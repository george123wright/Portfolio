from __future__ import annotations

"""
Direct multi-horizon equity forecaster with exogenous scenario simulation.

Mathematical summary
--------------------
Target:
   
    y = ∑_{h=1..K} r_{t+h}

Decomposition:

    y = μ_base(r_t; m, φ) + ε

ε | X ~ t_ν(μ_resid(X_{t − H + 1:t + K}), σ(X_{t − H + 1:t + K}))

Macro simulator:

- X^{macro} transformed to deltas

    PCA factors F_t
    
    BVAR(p) on F_t with a Minnesota prior
    
    Simulate forward and invert to macro deltas.

Factors:

- Stationary bootstrap with restart p.

Firm fundamentals:

- Revenue in log space; EPS in signed-log space; drifts anchored to combined
  analyst targets plus Gaussian shock.

Learning:

- Two-layer LSTM consumes historical returns (H) and exogenous deltas for both
  history and future (H+K), outputs quantiles for ε and Student-t parameters.

- Loss = 0.8 × average pinball loss + 0.2 × Student-t NLL + regularisers.

Uncertainty:

- Monte-Carlo dropout + scenario simulators produce predictive distributions of y 

  Transform to prices via P_q = (1 + q) P_0.

All scaling uses robust statistics; covariance matrices are shrunk and forced
SPD for numerical reliability.
"""

import os as _os

_os.environ.setdefault("PYTHONHASHSEED", "42")

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

for _v in (
    "OMP_NUM_THREADS", 
    "OPENBLAS_NUM_THREADS", 
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
):

    _os.environ.setdefault(_v, "1")

import cProfile, faulthandler, gc, logging, psutil, pstats, random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas.api.types import is_period_dtype, is_datetime64_any_dtype

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import shared_memory

from dataclasses import dataclass

import config
from data_processing.financial_forecast_data import FinancialForecastData
from TVP_GARCH_MC import _analyst_sigmas_and_targets_combined

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


STATE = None

FORECASTER_SINGLETON = None

SAVE_TO_EXCEL = True


def _init_worker(
    state_pack
):
    """
    Initialise a worker process for parallel forecasting.

    This function is used as a `ProcessPoolExecutor` initializer. It receives a
    prebuilt, read-only "state pack" containing all data and configuration needed to
    compute forecasts, and writes it into a module-level global (`STATE`) so that
    each worker can access the same memory-mapped resources without reloading or
    recomputing them.

    Parameters
    ----------
    state_pack : dict
        A serialisable dictionary built by `DirectHForecaster.build_state`. It
        contains preprocessed market, macroeconomic and firm-level data, shared
        memory pointers to Monte Carlo paths for exogenous variables, index
        alignment caches, and other metadata required by `forecast_one`.

    Notes
    -----
    The design avoids repeatedly shipping large arrays through inter-process pipes.
    Shared memory segments (created in `build_state`) let workers view large
    simulation tensors (e.g., macro/factor projections) with near-zero copy cost.

    Side Effects
    ------------
    Sets the module-level global `STATE` to `state_pack`, which downstream worker
    functions read.
    """

    global STATE

    STATE = state_pack


def _worker_forecast_one(
    ticker: str
):
    """
    Worker entry point that produces a forecast for a single ticker.

    This thin wrapper constructs a per-process singleton `DirectHForecaster`
    (the first time it is called in a worker) and delegates to
    `DirectHForecaster.forecast_one`. A singleton is used so that TensorFlow
    graphs, compiled functions and model weights are created once per process.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol to forecast.

    Returns
    -------
    dict
        A result record with keys such as {"Ticker", "status"} and, on success,
        price quantiles ("Min Price", "Avg Price", "Max Price"), expected
        return ("Returns") and dispersion ("SE").

    Notes
    -----
    Uses the module-level global `STATE` initialised by `_init_worker`.
    """

    global FORECASTER_SINGLETON

    if FORECASTER_SINGLETON is None:

        FORECASTER_SINGLETON = DirectHForecaster()

    return FORECASTER_SINGLETON.forecast_one(
        ticker = ticker
    )


@dataclass(frozen = True)
class ModelHP:
    """
    Container for model architecture hyperparameters.

    Attributes
    ----------
    hist_window : int
        Length H of the historical window (in weeks) fed into the network.
    horizon : int
        Forecast horizon (in weeks). The target is the sum of log returns across
        this horizon.
    lstm1, lstm2 : int
        Number of hidden units in the first and second LSTM layers respectively.
    l2_lambda : float
        ℓ₂ regularisation coefficient applied to kernel and recurrent weights.
    dropout : float
        Drop probability for input-to-hidden connections in each LSTM (Gal &
        Ghahramani style Monte-Carlo dropout when `training=True` is used at
        inference).
    """
   
    hist_window: int = 52
   
    horizon: int = 52
   
    lstm1: int = 128
   
    lstm2: int = 64
   
    l2_lambda: float = 1e-4
   
    dropout: float = 0.15


@dataclass(frozen = True)
class TrainHP:
    """
    Container for optimisation hyperparameters.

    Attributes
    ----------
    batch : int
        Mini-batch size used in model fitting.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Patience for early stopping and learning-rate reduction callbacks.
    lr : float
        Adam learning rate.
    small_floor : float
        Numerical floor used to prevent division by ~0 (e.g., clipping levels or
        scalers).
    """
   
    batch: int = 256
   
    epochs: int = 100
   
    patience: int = 10
   
    lr: float = 5e-4
   
    small_floor: float = 1e-6


@dataclass(frozen = True)
class DistHP:
    """
    Container for distributional head hyperparameters.

    Attributes
    ----------
    sigma_floor : float
        Additive floor for the scale parameter σ to ensure positivity and
        numerical stability.
    sigma_max : float
        Upper bound for σ to avoid explosive tails.
    nu_floor : float
        Minimum degrees of freedom ν for Student-t (ν>2 for finite variance).
    lambda_sigma, lambda_invnu : float
        Regularisation penalties for large σ and for light tails (via 1/(ν−2)),
        encouraging well-behaved predictive distributions.
    """
    
    sigma_floor: float = 1e-3
    
    sigma_max: float = 0.6
    
    nu_floor: float = 8.0
    
    lambda_sigma: float = 5e-4
    
    lambda_invnu: float = 5e-4


@dataclass(frozen = True)
class ScenarioHP:
    """
    Container for scenario/Monte-Carlo simulation hyperparameters.

    Attributes
    ----------
    n_sims : int
        Number of macro/factor/financial paths simulated per ticker.
    mc_dropout_samples : int
        Number of Monte-Carlo dropout forward passes used to draw predictive
        residuals from the LSTM's stochastic head.
    alpha_conf : float
        One-sided miscoverage for conformal/quantile adjustment (if applied).
    rev_noise_sd, eps_noise_sd : float
        Standard deviations for analyst-target shock priors (log-space for revenue,
        signed-log1p space for EPS).
    bootstrap_p : float
        Restart probability p of the stationary bootstrap; expected block length is
        1/p.
    """
   
    n_sims: int = 100
   
    mc_dropout_samples: int = 20
   
    alpha_conf: float = 0.10
   
    rev_noise_sd: float = 0.005
   
    eps_noise_sd: float = 0.005
   
    bootstrap_p: float = 1 / 6


@dataclass(frozen = True)
class HP:
   
    model: ModelHP = ModelHP()
   
    train: TrainHP = TrainHP()
   
    dist: DistHP = DistHP()
   
    scen: ScenarioHP = ScenarioHP()


class DirectHForecaster:
    """
    Direct multi-horizon forecaster that combines LSTM sequence modelling with
    factor-augmented VAR macro projections, analyst-target financial paths, and
    Monte-Carlo dropout for predictive uncertainty.

    Overview
    --------
    Given weekly equity log returns r_t and a panel of exogenous regressors X_t
    (macroeconomic levels transformed to weekly deltas, cross-sectional factor
    returns, firm revenue and EPS features, and higher-order moments of returns),
    the forecaster learns a direct mapping from a window of length H (history) plus
    a future exogenous path of length K (forecast horizon) to the cumulative future
    log return across the horizon.

    Let:
    
    - H = `HIST_WINDOW`,
    
    - K = `HORIZON`,
    
    - y = ∑_{h=1..K} r_{t+h}  (cumulative log return over the horizon),
    
    - X_{t−H+1:t}  be historical delta-features,
    
    - X_{t+1:t+K}  be future delta-features produced by stochastic simulators.

    The model learns an additive decomposition
    
        y = μ_base(r_t; θ_AR) + ε,                            (1)
    
    where μ_base is an AR(1) baseline for cumulative log returns and ε is a
    residual predicted by an LSTM with both quantile and parametric (Student-t)
    heads.

    Key modelling components
    ------------------------
    
    1) **AR(1) baseline.** The last observed log return r_t follows
    
        r_{t+1} = c + φ r_t + u_{t+1},
    
    with u_{t+1} white noise. The unconditional mean is m = c / (1 − φ).
    
    The K-step sum of returns conditional on r_t is
    
        μ_base(r_t; m, φ) = K m + (r_t − m) φ (1 − φ^K) / (1 − φ).      (2)

    2) **LSTM residual model.** The network receives a tensor of shape
    (H+K, 1+n_reg), where the first channel is scaled returns history for the
    H historical steps (and zeros for the K future steps), and the remaining
    channels are delta-features for both historical and simulated future steps.
    
    The LSTM stack outputs:
    
    - three monotone quantiles q_{0.1}, q_{0.5}, q_{0.9} for ε, enforced by
        softplus gaps; and
    
    - three unconstrained parameters that are transformed to a Student-t mean
        μ_resid, scale σ, and degrees of freedom ν.

    3) **Distributional head (Student-t).** With ε|X ∼ t_ν(μ_resid, σ), the
    negative log-likelihood per observation is
    
        −log p(ε) = − log Γ((ν + 1) / 2) + log Γ(ν / 2) + 0.5 log[(ν − 2)π] + log σm+ 0.5(ν + 1) log(1 + z^2/(ν − 2)), (3)
        
    where z = (ε − μ_resid)/σ and ν>2. Regularisation adds
    
        λ_σ E[σ^2] + λ_{invν} E[1 / (ν − 2)]                              (4)
    
    to discourage over-wide or overly light-tailed fits.

    4) **Quantile head.** The pinball (check) loss for a quantile level q∈(0,1)
    and residual ε is
      
        L_q(ε, hat{ε}_q) = max(q(ε − hat{ε}_q), (q − 1)(ε − hat{ε}_q)). (5)
    
    The training objective averages (5) over q∈{0.1, 0.5, 0.9}.

    5) **Exogenous simulators.**
   
    - *Macro features:* factor-augmented VAR (FAVAR). Weekly macro deltas are
        standardised, projected to principal components F_t = P' S (X_t − μ),
        fitted with a Bayesian VAR(p) under a Minnesota prior, then simulated
        forward K steps; finally inverted back to the macro delta space.
   
    - *Cross-sectional factors:* simulated by the stationary bootstrap with
        restart probability p; expected block length 1/p.
   
    - *Firm-level paths:* revenue and EPS deltas are centred on combined
        analyst targets with Gaussian noise in log / signed-log space.

    6) **Uncertainty propagation.** Future exogenous paths are combined with
    Monte-Carlo dropout draws of the distributional head, sampling
    ε ~ t_ν(μ_resid, σ) for each draw, and then adding μ_base to obtain K-step
    cumulative log returns. Simple returns are R = exp(y) − 1.

    Why these choices
    -----------------
    - The AR(1) baseline captures short-term mean reversion without burdening the
    LSTM; the LSTM focuses on conditional deviations tied to exogenous paths.
   
    - Student-t residuals accommodate heavy tails in equity returns.
   
    - FAVAR reduces macro dimensionality while preserving co-movement, and the
    Minnesota prior stabilises VAR inference in small samples.
   
    - Stationary bootstrap preserves short-range autocorrelation in factor returns.
   
    - Monte-Carlo dropout approximates Bayesian model averaging over network
    weights, providing a pragmatic route to predictive intervals.

    References
    ----------
    Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
    Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and Conditional
    Projection Using Realistic Prior Distributions.
    """
   
    SEED = 42
   
    random.seed(SEED)
   
    np.random.seed(SEED)

    SAVE_ARTIFACTS = False
   
    ARTIFACT_DIR = config.BASE_DIR / "lstm_artifacts"
   
    EXCEL_PATH = config.MODEL_FILE

    HIST_WINDOW = 52
   
    HORIZON = 52
   
    SEQUENCE_LENGTH = HIST_WINDOW + HORIZON

    BATCH = 64
   
    EPOCHS = 100
   
    PATIENCE = 10
   
    SMALL_FLOOR = 1e-6
   
    L2_LAMBDA = 1e-4

    N_SIMS = 100
   
    MC_DROPOUT_SAMPLES = 20
   
    ALPHA_CONF = 0.10

    SIGMA_FLOOR = 1e-3
   
    SIGMA_MAX = 0.15#0.60
   
    MU_MAX = 0.80
   
    REV_NOISE_SD = 0.005
   
    EPS_NOISE_SD = 0.005
   
    LAMBDA_SIGMA = 5e-4
   
    LAMBDA_INVNU = 5e-4
   
    NU_FLOOR = 8.0

    MACRO_REGRESSORS = [
        "Interest",
        "Cpi",
        "Gdp",
        "Unemp",
        "Balance Of Trade",
        "Corporate Profits",
        "Balance On Current Account",
    ]
    
    FACTORS = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]
    
    FIN_REGRESSORS = ["Revenue", "EPS (Basic)"]
    
    MOMENT_COLS = ["skew_104w_lag52", "kurt_104w_lag52"]
    
    NON_FIN_REGRESSORS = MACRO_REGRESSORS + FACTORS
    
    ALL_REGRESSORS = NON_FIN_REGRESSORS + FIN_REGRESSORS + MOMENT_COLS

    _MODEL_CACHE: Dict[int, Tuple[Any, List[np.ndarray]]] = {}
    
    _FN_CACHE: Dict[int, Dict[str, Any]] = {}
    
    _EYE_CACHE = {}


    def __init__(
        self, 
        tickers: Optional[List[str]] = None,
        hp: Optional["DirectHForecaster.HP"] = None
    ):
        """
        Construct a forecaster with explicit hyperparameters and prepare index maps.

        Parameters
        ----------
        tickers : list[str] or None
            Optional list of tickers to forecast. If `None`, a default set is used.
        hp : DirectHForecaster.HP or None
            Aggregated hyperparameter container. If `None`, defaults are used.

        Side Effects
        ------------
        - Sets model, training, distributional and scenario hyperparameters.
        - Builds index maps for macro/factor/financial/moment columns.
        - Configures TensorFlow determinism and thread limits.

        Notes
        -----
        The object keeps small, immutable caches (e.g., `_MODEL_CACHE`) keyed by the
        number of regressors so that multiple tickers sharing the same input shape
        reuse the same compiled Keras graph and initial weights.
        """
    
        self.logger = self._configure_logger()
    
        self.tickers_arg = tickers
    
        self.hp = hp or HP()

        self.HIST_WINDOW = self.hp.model.hist_window
    
        self.HORIZON = self.hp.model.horizon
    
        self.SEQUENCE_LENGTH = self.HIST_WINDOW + self.HORIZON
    
        self.L2_LAMBDA = self.hp.model.l2_lambda
    
        self._LSTM1 = self.hp.model.lstm1
    
        self._LSTM2 = self.hp.model.lstm2
    
        self._DROPOUT = self.hp.model.dropout

        self.BATCH = self.hp.train.batch
    
        self.EPOCHS = self.hp.train.epochs
    
        self.PATIENCE = self.hp.train.patience
    
        self._LR = self.hp.train.lr
    
        self.SMALL_FLOOR = self.hp.train.small_floor

        self.SIGMA_FLOOR = self.hp.dist.sigma_floor
    
        self.SIGMA_MAX = self.hp.dist.sigma_max
    
        self.NU_FLOOR = self.hp.dist.nu_floor
    
        self.LAMBDA_SIGMA = self.hp.dist.lambda_sigma
    
        self.LAMBDA_INVNU = self.hp.dist.lambda_invnu

        self.N_SIMS = self.hp.scen.n_sims
    
        self.MC_DROPOUT_SAMPLES = self.hp.scen.mc_dropout_samples
    
        self.ALPHA_CONF = self.hp.scen.alpha_conf
    
        self.REV_NOISE_SD = self.hp.scen.rev_noise_sd
    
        self.EPS_NOISE_SD = self.hp.scen.eps_noise_sd
    
        self._BOOT_P = self.hp.scen.bootstrap_p

        self._created_shms: List[shared_memory.SharedMemory] = []

        self.regs = list(self.ALL_REGRESSORS)
    
        self.reg_pos = {name: i for i, name in enumerate(self.regs)}
    
        self.n_reg = len(self.regs)

        self._macro_idx = np.array([self.reg_pos[m] for m in self.MACRO_REGRESSORS], dtype = np.int64)
    
        self._factor_idx = np.array([self.reg_pos[f] for f in self.FACTORS], dtype = np.int64)
    
        self._fin_idx = np.array([self.reg_pos["Revenue"], self.reg_pos["EPS (Basic)"]], dtype = np.int64)
    
        self._mom_idx = np.array([self.reg_pos["skew_104w_lag52"], self.reg_pos["kurt_104w_lag52"]], dtype = np.int64)
        
        self._configure_tf()


    def _configure_tf(
        self
    ):
        """
        Configure TensorFlow determinism and thread usage.

        Enables op determinism (where available), seeds the per-process TF PRNG, and
        limits intra/inter op parallelism to 1 thread to reduce non-determinism across
        workers.

        This is essential when Monte-Carlo dropout is used to reflect only stochastic
        inference variability rather than thread scheduling artefacts.
        """

        try:
    
            tf.config.experimental.enable_op_determinism()
    
        except Exception:
    
            pass
    
        tf.random.set_seed(self.SEED)
    
        try:
    
            tf.config.threading.set_intra_op_parallelism_threads(1)
    
            tf.config.threading.set_inter_op_parallelism_threads(1)
    
        except Exception:
    
            pass


    def _configure_logger(
        self
    ) -> logging.Logger:
        """
        Create or reuse a process-local logger.

        Returns
        -------
        logging.Logger
            Logger named "lstm_directH_class" with consistent formatting.

        Notes
        -----
        Initialises the handler only once per process to avoid duplicate log lines.
        """
            
        logger = logging.getLogger("lstm_directH_class")
    
        logger.setLevel(logging.INFO)
    
        if not logger.handlers:
    
            h = logging.StreamHandler()
    
            h.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s"))
    
            logger.addHandler(h)
    
        return logger


    def choose_splits(
        self,
        N: int, 
        min_fold: Optional[int] = None
    ) -> int:
        """
        Choose the number of time-series cross-validation splits from sample size.

        Parameters
        ----------
        N : int
            Number of rolling windows available for training (i.e., T_reg − (H+K) + 1).
        min_fold : int or None
            Minimum number of windows per fold; defaults to 2×batch size.

        Returns
        -------
        int
            Number of splits to use (0, 1, or 2). For small N, falls back to a single
            hold-out or no split.

        Rationale
        ---------
        Time-series splitting must preserve order. Small samples cannot support many
        folds without leaking. The heuristic avoids unstable validation sets.
        """
    
        if min_fold is None:
    
            min_fold = 2 * self.BATCH
    
        if N < min_fold:
    
            return 0
    
        if N >= 2 * min_fold:
    
            return 2
    
        return 1

    
    def build_delta_matrix(
        self, 
        reg_mat: np.ndarray, 
        regs: list[str]
    ) -> np.ndarray:
        """
        Construct a matrix of feature deltas from level-valued regressors.

        Given a T×n_reg matrix of regressor levels (or already delta-like series for
        some factors), compute the (T−1)×n_reg matrix ΔX where each column is
        transformed as follows:

        - For difference-stationary series S_t (e.g., Interest, Unemp, Balance of
        Trade, Balance on Current Account, and moment columns), use
         
            ΔS_t = S_t − S_{t−1}.
       
        - For log-difference series L_t > 0 (e.g., Cpi, Gdp, Revenue, Corporate
        Profits), use
       
            Δlog L_t = log L_t − log L_{t−1}.
      
        - For EPS, apply the signed log transform g(x) = sign(x)·log(1+|x|) to handle
        negative values, then difference:
       
            Δg(EPS)_t = g(EPS_t) − g(EPS_{t−1}).
       
        - For factor returns (already stationary), pass through the contemporaneous
        value:
       
            ΔF_t = F_t  (no differencing).

        Parameters
        ----------
        reg_mat : ndarray, shape (T, n_reg)
            Level values for all regressors.
        regs : list[str]
            Column names aligned with `reg_mat`.

        Returns
        -------
        ndarray, shape (T−1, n_reg)
            Delta-transformed feature matrix.

        Notes
        -----
        The transformation ensures approximate stationarity and scale comparability.
        A small positive floor is used before log transforms to avoid log(0).
        """
    
        reg_mat = np.asarray(reg_mat, dtype = np.float32)
    
        T, n_reg = reg_mat.shape
    
        regs_arr = np.array(regs, dtype = object)

        is_diff = np.isin(regs_arr, ("Interest", "Unemp", "Balance Of Trade", "Balance On Current Account")) | np.isin(regs_arr, self.MOMENT_COLS)
      
        is_dlog = np.isin(regs_arr, ("Cpi", "Gdp", "Revenue", "Corporate Profits"))
      
        is_slog_eps = (regs_arr == "EPS (Basic)")
      
        is_passthrough = np.isin(regs_arr, self.FACTORS)

        DEL = np.empty((T - 1, n_reg), dtype = np.float32)

        if is_diff.any():
      
            cols = reg_mat[:, is_diff]
      
            DEL[:, is_diff] = cols[1:] - cols[:-1]

        if is_dlog.any():
      
            cols = np.maximum(reg_mat[:, is_dlog], self.SMALL_FLOOR)
      
            DEL[:, is_dlog] = np.log(cols[1:]) - np.log(cols[:-1])

        if is_slog_eps.any():
      
            cols = reg_mat[:, is_slog_eps].copy()
      
            prev = self.slog1p_signed(
                x = cols[:-1, :]
            )
      
            nxt = self.slog1p_signed(
                x = cols[1:, :]
            )
      
            DEL[:, is_slog_eps] = (nxt - prev)

        if is_passthrough.any():
            
            DEL[:, is_passthrough] = reg_mat[1:, is_passthrough]

        other = ~(is_diff | is_dlog | is_slog_eps | is_passthrough)
        
        if other.any():
            
            cols = reg_mat[:, other]
            
            DEL[:, other] = cols[1:] - cols[:-1]

        return DEL


    @staticmethod
    def fit_scale_deltas(
        DEL_tr: np.ndarray
    ):
        """
        Fit a robust scaler and compute clipping quantiles for delta features.

        Parameters
        ----------
        DEL_tr : ndarray, shape (N, n_reg)
            Training subset of delta features.

        Returns
        -------
        sc : sklearn.preprocessing.RobustScaler
            Fitted scaler using the median and inter-quartile range per column.
        q_low, q_high : ndarray
            Column-wise 1% and 99% quantiles used to winsorise extreme deltas.

        Rationale
        ---------
        Robust scaling (median/IQR) is resilient to heavy-tailed shocks, making neural
        optimisation numerically stable. Clipping at [1%, 99%] mitigates undue
        influence from outliers.
        """
    
        sc = RobustScaler().fit(DEL_tr)
    
        sc.scale_[sc.scale_ < 1e-6] = 1e-6
    
        q_low = np.quantile(DEL_tr, 0.01, axis=0).astype(np.float32)
    
        q_high = np.quantile(DEL_tr, 0.99, axis=0).astype(np.float32)
    
        return sc, q_low, q_high


    @staticmethod
    def fit_ar1_baseline(
        log_ret: np.ndarray
    ) -> tuple[float, float]:
        """
        Estimate an AR(1) baseline for weekly log returns.

        Fits 
        
            r_{t+1} = c + φ r_t + u_{t+1} 
        
        by OLS and returns the implied unconditional mean m = c / (1−φ) 
        (with protection if φ≈1) and autoregressive coefficient φ.

        Parameters
        ----------
        log_ret : array-like
            Vector of weekly log returns r_t.

        Returns
        -------
        m : float
            Unconditional mean of the AR(1) process.
        phi : float
            Autoregressive coefficient.

        Use
        ---
        The AR(1) is used to produce μ_base(r_t; m, φ) over the K-step horizon, as in
        equation (2) of the class docstring, which is subtracted from the target to
        form a residual for the LSTM to learn.
        """
    
        r = np.asarray(log_ret, float).ravel()
    
        if len(r) < 3:
    
            return 0.0, 0.0
    
        r0 = r[:-1]
        
        r1 = r[1:]
        
        X = np.column_stack([np.ones_like(r0), r0])
        
        beta, *_ = np.linalg.lstsq(X, r1, rcond = None)
        
        m = float(beta[0] / (1.0 - beta[1])) if abs(1.0 - beta[1]) > 1e-8 else 0.0
        
        phi = float(beta[1])
        
        return m, phi


    @staticmethod
    def ar1_sum_forecast(
        r_t: float, 
        m: float, 
        phi: float,
        H: int
    ) -> float:
        """
        Closed-form K-step sum of AR(1) returns conditional on r_t.

        Implements equation (2):
        - If φ ≈ 1, returns K·m (random-walk with drift limit).
        - Otherwise, returns 
        
            K·m + (r_t − m) φ (1 − φ^K) / (1 − φ).

        Parameters
        ----------
        r_t : float
            Last observed log return.
        m : float
            Unconditional mean of the AR(1).
        phi : float
            Autoregressive coefficient.
        H : int
            Horizon length K.

        Returns
        -------
        float
            Expected cumulative log return over H steps.
        """
    
        if abs(1.0 - phi) < 1e-8:
    
            return H * m
    
        geom = (1.0 - phi**H) / (1.0 - phi)
    
        return H * m + (r_t - m) * phi * geom


    @staticmethod
    def transform_deltas(
        DEL: np.ndarray, 
        sc: RobustScaler,
        q_low: np.ndarray,
        q_high: np.ndarray
    ):
        """
        Winsorise and robust-scale delta features with a fitted scaler.

        Parameters
        ----------
        DEL : ndarray, shape (N, n_reg) or (S·K, n_reg)
            Delta features to transform.
        sc : RobustScaler
            Fitted robust scaler.
        q_low, q_high : ndarray
            1% and 99% clipping quantiles per column.

        Returns
        -------
        ndarray
            Robust-scaled and clipped delta features in float32.
        """
    
        DEL_clip = np.clip(DEL, q_low, q_high)
    
        DEL_scaled = (DEL_clip - sc.center_) / sc.scale_
    
        return DEL_scaled.astype(np.float32)


    @staticmethod
    def _macro_levels_to_deltas_favar(
        df_levels: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert macroeconomic level series to weekly deltas for FAVAR.

        For each macro column X_t:
        - If X_t ∈ {Cpi, Gdp, Corporate Profits}, compute log-difference
        Δlog X_t = log X_t − log X_{t−1} with a positive floor before log.
       
        - Otherwise compute simple difference ΔX_t = X_t − X_{t−1}.

        Parameters
        ----------
        df_levels : pandas.DataFrame
            Weekly/periodic macro levels indexed by date.

        Returns
        -------
        pandas.DataFrame
            Differenced macro series aligned to input index.

        Rationale
        ---------
        FAVAR assumes stationary inputs; log-differences approximate percentage
        changes and stabilise variance for strictly positive aggregates.
        """
    
        cols = list(df_levels.columns)
    
        out = {}
    
        for c in cols:
    
            s = pd.to_numeric(df_levels[c], errors = "coerce").astype(float).ffill().bfill()
    
            if c in ("Cpi", "Gdp", "Corporate Profits"):
    
                s = s.clip(lower = 1e-12)
    
                out[c] = np.log(s).diff()
    
            else:
    
                out[c] = s.diff()
    
        return pd.DataFrame(out, index = df_levels.index)

    
    @staticmethod
    def _favar_fit(
        dX_macro: pd.DataFrame, 
        max_pcs: int = 4
    ) -> tuple[pd.DataFrame, PCA, StandardScaler]:
        """
        Fit a Factor-Augmented VAR projection basis via PCA.

        Steps
        -----
        1) Standardise delta macro series: Z = (X − μ) / σ.
        
        2) Compute principal components: F = Z W, where W are the top eigenvectors
        (n_components ≤ 4).
        
        3) Return component scores F_t together with the fitted scaler and PCA object.

        Parameters
        ----------
        dX_macro : pandas.DataFrame, shape (T, p)
            Macro deltas.
        max_pcs : int
            Maximum number of principal components to retain.

        Returns
        -------
        pcs : pandas.DataFrame
            Principal component scores indexed by time, columns "pc1", "pc2", ….
        pca : sklearn.decomposition.PCA
            Fitted PCA model for inverse transformation.
        sc : sklearn.preprocessing.StandardScaler
            Fitted standardiser (mean, variance) for macro deltas.

        Notes
        -----
        Using a small number of factors captures common macro variation with reduced
        dimension, stabilising BVAR estimation.
        """
    
        sc = StandardScaler()
    
        X = sc.fit_transform(dX_macro.values)
    
        pca = PCA(n_components = min(max_pcs, X.shape[1]))
    
        F = pca.fit_transform(X)
    
        pcs = pd.DataFrame(F, index = dX_macro.index, columns = [f"pc{i+1}" for i in range(pca.n_components_)])
    
        return pcs, pca, sc

 
    @staticmethod
    def _mn_prior_BVAR(
        dX: np.ndarray, 
        p: int
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Construct a Minnesota prior for a VAR(p) on factor scores.

        Let the VAR(p) be
        
            Y_t = c + A_1 Y_{t−1} + … + A_p Y_{t−p} + e_t,           (6)
        
        with e_t ~ N(0, Σ) and Y_t ∈ R^k. Stack coefficients into
        
            B = [c, A_1, …, A_p]ᵀ  ∈ R^{(1+kp)×k}.

        The Minnesota prior specifies a conjugate Normal-Inverse-Wishart-like form:
        
            vec(B) ~ N(vec(B₀), V₀),  Σ ~ fixed (or weakly informative).        (7)

        Here a diagonal V₀ is built with shrinkage hyperparameters λ₁, λ₂, λ₃, λ₄:
        
        - Own-lags: Var(A_{i,i}^{(L)}) = (λ₁ / L^{λ₃})².
        
        - Cross-lags: Var(A_{i,j}^{(L)}) = (λ₁ λ₂ / L^{λ₃})² · (σ_i / σ_j).
        
        - Intercept: Var(c_i) = (λ₁ λ₄ σ_i)².

        Parameters
        ----------
        dX : ndarray, shape (T, k)
            Time-series of factor scores.
        p : int
            VAR order.

        Returns
        -------
        B0 : ndarray
            Prior mean for coefficients (zeros).
        V0 : ndarray
            Prior covariance (diagonal) for vec(B).
        nu0 : float
            Prior degrees of freedom for Σ.
        S0 : ndarray
            Prior scale matrix for Σ.

        Notes
        -----
        Small λ₁ and λ₂ shrink coefficients toward zero and own-lag persistence,
        improving stability and forecast accuracy in small T.
        """
    
        T, k = dX.shape
    
        lam1 = 0.25
        
        lam2 = 0.5
        
        lam3 = 1.0
        
        lam4 = 100.0
    
        sig = np.std(dX, axis = 0, ddof = 1) + 1e-8
    
        m = 1 + k * p
    
        V0_diag = np.zeros((k, m))
    
        for i in range(k):
    
            V0_diag[i, 0] = (lam1 * lam4 * sig[i]) ** 2
    
            for L in range(1, p + 1):
    
                for j in range(k):
    
                    pos = 1 + (L - 1) * k + j
    
                    if i == j:
    
                        var = (lam1 / (L**lam3)) ** 2
    
                    else:
    
                        var = (lam1 * lam2 / (L ** lam3)) ** 2 * (sig[i] / sig[j])
    
                    V0_diag[i, pos] = var
    
        V0 = np.diag(np.mean(V0_diag, axis=0))
    
        B0 = np.zeros((m, k))
    
        nu0 = max(k + 2, 6)
    
        S0 = 0.1 * np.eye(k)
    
        return B0, V0, nu0, S0


    @staticmethod
    def _build_YX(
        dX: np.ndarray, 
        p: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Form the response and design matrices for a VAR(p) with intercept.

        Given dX ∈ R^{T×k}, produce:
        
        - Y = dX[p: ] ∈ R^{(T−p)×k}
        
        - X = [1, dX_{t−1}, …, dX_{t−p}] for t=p..T−1, i.e.
        
        X ∈ R^{(T−p)×(1+kp)}.

        Parameters
        ----------
        dX : ndarray, shape (T, k)
        p : int

        Returns
        -------
        Y : ndarray
        X : ndarray

        Raises
        ------
        ValueError
            If T ≤ p (insufficient history for lags).
        """
    
        T, k = dX.shape
    
        if T <= p:
    
            raise ValueError("Not enough history for VAR lags.")
    
        Y = dX[p:]
    
        rows = []
    
        for t in range(p, T):
    
            row = [1.0]
    
            for L in range(1, p + 1):
    
                row.extend(dX[t - L])
    
            rows.append(row)
    
        X = np.array(rows, float)
    
        return Y, X


    def _fit_BVAR_minnesota(
        self,
        dX: pd.DataFrame,
        p: int = 2
    ):
        """
        Fit a Bayesian VAR with a Minnesota prior and return posterior parameters.

        Using the prior (B0, V0, ν0, S0) and the likelihood implied by (6), the
        posterior is (conjugate):
           
            Vn = (V0^{-1} + Xᵀ X)^{-1}
           
            Bn = Vn (V0^{-1} B0 + Xᵀ Y)
           
            νn = ν0 + T−p
           
            Sn = S0 + Yᵀ Y + B0ᵀ V0^{-1} B0 − Bnᵀ (V0^{-1} + Xᵀ X) Bn.

        Parameters
        ----------
        dX : pandas.DataFrame
            Factor scores (pcs) over time.
        p : int
            VAR order.

        Returns
        -------
        dict
            Posterior parameters { "Bn", "Vn", "Sn", "nun", "k", "p" } ready for
            simulation via `_sample_BSigma`.
        """
    
        Y, X = self._build_YX(
            dX = dX.values, 
            p = p
        )
    
        k = dX.shape[1]
    
        B0, V0, nu0, S0 = self._mn_prior_BVAR(
            dX = dX.values, 
            p = p
        )
    
        V0inv = np.linalg.pinv(V0)
    
        XtX, XtY, YtY = X.T @ X, X.T @ Y, Y.T @ Y
    
        Vn = np.linalg.pinv(V0inv + XtX)
    
        Bn = Vn @ (V0inv @ B0 + XtY)
    
        Sn = S0 + YtY + B0.T @ V0inv @ B0 - Bn.T @ (V0inv + XtX) @ Bn
    
        nun = nu0 + Y.shape[0]
    
        return {
            "Bn": Bn, 
            "Vn": Vn, 
            "Sn": Sn, 
            "nun": nun,
            "k": k, 
            "p": p
        }


    @staticmethod
    def _sample_BSigma(
        post, 
        rng: np.random.Generator
    ):
        """
        Draw a single posterior sample of (B, Σ) from the BVAR posterior.

        Sampling scheme
        ---------------
        1) Draw Σ from an inverse-Wishart implied by Sn and νn by sampling a Wishart
        W ~ Wishart(νn, Sn^{-1}) and taking Σ = W^{-1}.
        
        2) Conditional on Σ, draw B from
           
            vec(B) ~ N(vec(Bn), Vn ⊗ Σ).

        Parameters
        ----------
        post : dict
            Posterior parameters from `_fit_BVAR_minnesota`.
        rng : np.random.Generator
            Random generator.

        Returns
        -------
        B : ndarray, shape ((1+kp), k)
            Sampled coefficient stack.
        Sigma : ndarray, shape (k, k)
            Sampled covariance.

        Notes
        -----
        This corresponds to the conjugate Normal-Inverse-Wishart posterior for a VAR
        under a Minnesota-type prior.
        """
    
        Bn = post["Bn"]
        
        Vn = post["Vn"]
        
        Sn = post["Sn"]
        
        nun = post["nun"]
        
        k = post["k"]
        
        p = post["p"]
    
        Sinv = np.linalg.inv(Sn)
        
        L = np.linalg.cholesky(Sinv)
        
        A = np.zeros((k, k))
        
        for i in range(k):
        
            A[i, i] = np.sqrt(rng.chisquare(nun - i))
        
            for j in range(i):
        
                A[i, j] = rng.normal()
        
        W = L @ A @ A.T @ L.T
        
        Sigma = np.linalg.inv(W)
        
        m = Vn.shape[0]
        
        Z = rng.standard_normal((m, k))
        
        L_V = np.linalg.cholesky(Vn + 1e-12 * np.eye(m))
        
        L_S = np.linalg.cholesky((Sigma + Sigma.T) / 2)
        
        E = L_V @ Z @ L_S.T
        
        B = Bn + E
        
        return B, Sigma


    @staticmethod
    def _simulate_VAR(
        B: np.ndarray, 
        Sigma: np.ndarray, 
        dX_lags: np.ndarray, 
        steps: int, 
        rng: np.random.Generator
    ):
        """
        Simulate a VAR(p) forward for a given number of steps.

        Given coefficients B and covariance Σ, simulate
          
            Y_{t} = c + ∑_{L=1..p} A_L Y_{t−L} + e_t,
       
        with e_t ~ N(0, Σ), starting from the provided lag stack.

        Parameters
        ----------
        B : ndarray
            Coefficient stack with intercept row then lag blocks.
        Sigma : ndarray
            Positive-definite innovation covariance.
        dX_lags : ndarray, shape (p, k)
            Last p observations in chronological order.
        steps : int
            Number of forward steps to simulate.
        rng : np.random.Generator

        Returns
        -------
        ndarray, shape (steps, k)
            Simulated path of factor scores.
        """
    
        p = (B.shape[0] - 1) // dX_lags.shape[1]
    
        k = dX_lags.shape[1]
    
        Lchol = np.linalg.cholesky((Sigma + Sigma.T) / 2 + 1e-9 * np.eye(k))
    
        c = B[0]
    
        Al = [B[1 + L * k: 1 + (L + 1) * k].T for L in range(p)]
    
        lags = dX_lags.copy()
    
        out = np.zeros((steps, k), float)
    
        for t in range(steps):
    
            pred = c.copy()
    
            for L in range(1, p + 1):
    
                pred += Al[L - 1] @ lags[-L]
    
            innov = Lchol @ rng.standard_normal(k)
    
            dx_next = pred + innov
    
            out[t] = dx_next
    
            lags = np.vstack([lags[1:], dx_next])
    
        return out


    @staticmethod
    def ensure_spd_for_cholesky(
        Sigma: np.ndarray,
        min_eig: float = 1e-8,
        shrink: float = 0.05, 
        max_tries: int = 6
    ) -> np.ndarray:
        """
        Project a covariance-like matrix to the cone of symmetric positive-definite matrices.

        Performs:
        1) Symmetrisation: 
        
            S ← (S + Sᵀ)/2.
        
        2) Linear shrinkage: 
        
            S ← (1−λ) S + λ τ I, where τ = tr(S)/p.
        
        3) Eigenvalue clipping: eigenvalues w_i ← max(w_i, min_eig).
        
        4) Optional jitter increase until Cholesky succeeds.

        Parameters
        ----------
        Sigma : ndarray
            Input symmetric matrix.
        min_eig : float
            Minimum eigenvalue after clipping.
        shrink : float
            Shrinkage intensity λ.
        max_tries : int
            Maximum power-of-ten jitter attempts.

        Returns
        -------
        ndarray
            SPD matrix suitable for Cholesky.

        Rationale
        ---------
        Sampling and likelihood evaluation require SPD matrices. Numerical
        regularisation avoids failures due to rounding or finite-sample noise.
        """
   
        S = np.asarray(Sigma, dtype = np.float64)
    
        if not np.isfinite(S).all():
       
            S = np.where(np.isfinite(S), S, 0.0)
       
        S = 0.5 * (S + S.T)
       
        p = S.shape[0]
       
        I = np.eye(p, dtype = np.float64)
        
        tau = (np.trace(S) / p) if p > 0 else 1.0
        
        S = (1.0 - shrink) * S + shrink * tau * I
        
        w, V = np.linalg.eigh(S)
        
        w_clipped = np.maximum(w, min_eig)
        
        S = (V * w_clipped) @ V.T
        
        S = 0.5 * (S + S.T)
        
        jitter = 0.0
        
        for k in range(max_tries):
        
            try:
        
                np.linalg.cholesky(S + (jitter * I))
        
                if jitter > 0:
        
                    S = S + jitter * I
        
                return S.astype(np.float32)
        
            except np.linalg.LinAlgError:
        
                jitter = max(min_eig, (10.0 ** k) * min_eig)
        
        S = 0.5 * (S + S.T)
        
        S = S + (10.0 ** max_tries) * min_eig * I
        
        np.linalg.cholesky(S)
        
        return S.astype(np.float32)


    @staticmethod
    def _lw_cov_or_eye(
        tail_vals: Optional[np.ndarray], 
        dim: int
    ) -> np.ndarray:
        """
        Estimate a covariance matrix with Ledoit–Wolf shrinkage, or return a small eye.

        Parameters
        ----------
        tail_vals : ndarray or None
            Data used for covariance estimation. If insufficient observations (< 11),
            an eye matrix scaled by 1e−4 is returned.
        dim : int
            Target dimension.

        Returns
        -------
        ndarray
            Shrunk covariance estimate in float32.

        Notes
        -----
        Ledoit–Wolf shrinkage is well-behaved in small samples and reduces estimator
        variance.
        """
    
        if tail_vals is not None and tail_vals.shape[0] > 10:
    
            cov = LedoitWolf().fit(tail_vals).covariance_.astype(np.float32, copy = False)
    
            return cov
    
        return np.eye(dim, dtype=np.float32) * 1e-4

    
    @staticmethod
    def fit_ret_scaler_from_logret(
        log_ret: np.ndarray
    ) -> RobustScaler:
        """
        Fit a robust scaler to weekly log returns.

        Parameters
        ----------
        log_ret : ndarray
            Vector of log returns; the first element is ignored in fitting to avoid the
            synthetic 0 inserted earlier.

        Returns
        -------
        RobustScaler
            Fitted scaler with median centre and IQR scale; scale is floored at 1e−6.
        """
    
        sc = RobustScaler().fit(log_ret[1:].reshape(-1, 1))
    
        sc.scale_[sc.scale_ < 1e-6] = 1e-6
    
        return sc


    @staticmethod
    def slog1p_signed(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Apply the signed log1p transform to accommodate negative values.

        Definition
        ----------
        For x ∈ ℝ,
            g(x) = sign(x) · log(1 + |x|).

        This is monotone, odd, and approximately linear near 0, but compresses the
        tails like a log. It is used for EPS to avoid domain issues in log space.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        ndarray
            Transformed values.
        """
    
        x = np.asarray(x, float)
    
        return np.sign(x) * np.log1p(np.abs(x))


    @staticmethod
    def inv_slog1p_signed(
        v: np.ndarray
    ) -> np.ndarray:
        """
        Inverse of the signed log1p transform.

        For v ∈ ℝ,
          
            g^{-1}(v) = sign(v) · (exp(|v|) − 1).

        Parameters
        ----------
        v : array-like

        Returns
        -------
        ndarray
            Inverse-transformed values.
        """
    
        v = np.asarray(v, float)
    
        return np.sign(v) * np.expm1(np.abs(v))


    @staticmethod
    def scale_logret_with(
        log_ret: np.ndarray,
        sc: RobustScaler
    ) -> np.ndarray:
        """
        Scale log returns with a fitted robust scaler, preserving the first zero.

        Parameters
        ----------
        log_ret : ndarray
            Vector of log returns with log_ret[0] = 0 by construction.
        sc : RobustScaler
            Fitted robust scaler with attributes `.center_` and `.scale_`.

        Returns
        -------
        ndarray
            Scaled series with the first element kept at 0.
        """
        
        return np.concatenate([[0.0], ((log_ret[1:] - sc.center_) / sc.scale_).ravel()]).astype(np.float32)


    def build_directH_model(
        self, 
        n_reg: int,
        seed: int = SEED
    ):
        """
        Build the two-layer LSTM model with quantile and Student-t heads.

        Network
        -------
        Input tensor shape: (H+K, 1 + n_reg), where:
      
        - channel 0 contains scaled returns for historical steps (and zeros for future);
      
        - channels 1.. contain delta-features for both historical and future steps.

        Layers:
      
        1) LSTM(units = lstm1, return_sequences = True, ℓ₂ regularisation, dropout).
      
        2) Layer normalisation.
      
        3) LSTM(units = lstm2, return_sequences = False, ℓ₂ regularisation, dropout).
      
        4) Two dense heads:
      
        - `q_raw`: 3 logits transformed to monotone quantiles via
      
            q10 = a,
      
            q50 = a + softplus(b) + ε,
      
            q90 = a + softplus(b) + softplus(c) + ε,
      
            where ε is a small constant to enforce strict monotonicity.
      
        - `dist_head`: 3 real outputs mapped to Student-t parameters via
      
            μ_resid = μ_max · tanh(θ_μ),
      
            σ = softplus(θ_σ) + σ_floor, then min with σ_max,
      
            ν = ν_floor + softplus(θ_ν).

        Returns
        -------
        tensorflow.keras.Model
            Compiled-ready Keras model mapping input sequences to both heads.

        Notes on LSTM dynamics
        ----------------------
        A standard LSTM cell at time t computes:
      
        - i_t = σ(W_i x_t + U_i h_{t−1} + b_i)       (input gate)
      
        - f_t = σ(W_f x_t + U_f h_{t−1} + b_f)       (forget gate)
      
        - o_t = σ(W_o x_t + U_o h_{t−1} + b_o)       (output gate)
      
        - g_t = tanh(W_g x_t + U_g h_{t−1} + b_g)    (candidate)
      
        - c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t            (cell state update)
      
        - h_t = o_t ⊙ tanh(c_t)                      (hidden output)

        Stacking two LSTMs increases temporal receptive field and nonlinearity;
        layer normalisation stabilises training across long sequences.
        """
    
        tf.random.set_seed(seed)
    
        inp = Input((self.SEQUENCE_LENGTH, 1 + n_reg), dtype = "float32")
    
        x = LSTM( 
            self._LSTM1,
            return_sequences = True,
            kernel_regularizer = l2(self.L2_LAMBDA),
            recurrent_regularizer = l2(self.L2_LAMBDA),
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
            recurrent_initializer = tf.keras.initializers.Orthogonal(seed = seed),
            dropout = self._DROPOUT, 
            recurrent_dropout = 0.0
        )(inp)
        
        x = LayerNormalization()(x)
        
        x = LSTM(
            self._LSTM2,
            return_sequences = False,
            kernel_regularizer = l2(self.L2_LAMBDA),
            recurrent_regularizer = l2(self.L2_LAMBDA),
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
            recurrent_initializer = tf.keras.initializers.Orthogonal(seed = seed),
            dropout = self._DROPOUT, 
            recurrent_dropout = 0.0
        )(x)
        
        z = Dense(3, name = "q_raw")(x)


        def monotone_quantiles(
            z
        ):
        
            a = z[:, 0:1]
        
            b_gap = tf.nn.softplus(z[:, 1:2]) + 1e-6
        
            c_gap = tf.nn.softplus(z[:, 2:3]) + 1e-6
        
            q10 = a
        
            q50 = a + b_gap
        
            q90 = a + b_gap + c_gap
        
            return tf.concat([q10, q50, q90], axis=-1)


        q_out = Lambda(monotone_quantiles, name = "q_head")(z)

        d_params = Dense(3, name = "dist_head")(x)

        return Model(inp, [q_out, d_params])


    def _make_callbacks(
        self, 
        monitor: str
    ):
        """
        Create early-stopping and learning-rate scheduling callbacks.

        Parameters
        ----------
        monitor : str
            Metric to monitor on the validation set (e.g., "val_loss").

        Returns
        -------
        list[keras.callbacks.Callback]
            EarlyStopping (restoring best weights) and ReduceLROnPlateau.

        Rationale
        ---------
        Prevents overfitting and adapts the learning rate when validation progress
        stalls.
        """
                
        return [
            EarlyStopping(
                monitor = monitor,
                patience = self.PATIENCE, 
                restore_best_weights = True
            ),
            ReduceLROnPlateau(
                monitor = monitor,
                patience = self.PATIENCE,
                factor = 0.5, 
                min_lr = self.SMALL_FLOOR
            ),
        ]


    @staticmethod
    def pinball_loss(
        y_true,
        y_pred, 
        qs = (0.1, 0.5, 0.9)
    ):
        """
        Compute the average pinball (quantile) loss over fixed quantiles.

        For each q in (0.1, 0.5, 0.9) and residual target ε = y_true, the loss is
        
            L_q(ε, hat{ε}_q) = max(q(ε − hat{ε}_q), (q − 1)(ε − hat{ε}_q)),
        
        as in equation (5). The function averages the three losses.

        Parameters
        ----------
        y_true : Tensor
            Ground-truth residuals (single column).
        y_pred : Tensor
            Predicted quantiles stacked by column: [q10, q50, q90].
        qs : tuple[float]
            Quantile levels.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
    
        y_true = tf.cast(y_true[:, 0:1], tf.float32)
    
        losses = []
    
        for i, q in enumerate(qs):
    
            e = y_true - y_pred[:, i:i + 1]
    
            losses.append(tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e)))
    
        return tf.add_n(losses) / float(len(qs))


    def student_t_nll(
        self, 
        y_true_res,
        params
    ):
        """
        Negative log-likelihood of a Student-t residual model with regularisation.

        The head predicts raw parameters (θ_μ, θ_σ, θ_ν) mapped to:
       
        - μ_resid = μ_max · tanh(θ_μ),
       
        - σ = min(σ_max, softplus(θ_σ) + σ_floor) scaled by a calibration constant,
       
        - ν = ν_floor + softplus(θ_ν).

        For residual y, define z = (y − μ_resid)/σ. The negative log-likelihood is
       
            −log p(y) = − log Γ((ν+1)/2) + log Γ(ν/2) + 0.5 log((ν−2)π) + log σ + 0.5(ν+1) log(1 + z²/(ν−2)),
       
        and the objective adds penalties
       
            λ_σ E[σ²] + λ_{invν} E[1/(ν−2)].

        Parameters
        ----------
        y_true_res : Tensor
            Residual target (y − μ_base).
        params : Tensor
            Raw network outputs of shape (N, 3).

        Returns
        -------
        Tensor
            Scalar loss.

        Notes
        -----
        The calibration factor `_SIGMA_CAL_SCALE` is set from the validation residuals
        via the ratio of robust dispersion measures (median absolute deviation to the
        median predicted σ), clipped to a plausible range, to correct any systematic
        under/over-dispersion from the head.
        """
    
        y = tf.cast(y_true_res, tf.float32)
    
        mu_raw = tf.cast(params[:, 0:1], tf.float32)
    
        mu_resid = self.MU_MAX * tf.tanh(mu_raw)
    
        log_sigma = tf.cast(params[:, 1:2], tf.float32)
    
        df_raw = tf.cast(params[:, 2:3], tf.float32)

        sigma = tf.nn.softplus(log_sigma) + self.SIGMA_FLOOR
    
        sigma = tf.minimum(sigma * getattr(self, "_SIGMA_CAL_SCALE", 1.0), self.SIGMA_MAX)
    
        nu = self.NU_FLOOR + tf.nn.softplus(df_raw)

        z = (y - mu_resid) / sigma
    
        logC = (tf.math.lgamma((nu + 1.0) / 2.0) - tf.math.lgamma(nu / 2.0) - 0.5 * tf.math.log((nu - 2.0) * np.pi) - tf.math.log(sigma))
        
        log_pdf = logC - 0.5 * (nu + 1.0) * tf.math.log(1.0 + (z * z) / (nu - 2.0))
       
        nll = -tf.reduce_mean(log_pdf)
       
        reg = self.LAMBDA_SIGMA * tf.reduce_mean(tf.square(sigma)) + self.LAMBDA_INVNU * tf.reduce_mean(1.0 / (nu - 2.0))
       
        return nll + reg


    def build_state(
        self
    ) -> Dict[str, Any]:
        """
        Assemble and serialise all data, simulators, and alignment caches for workers.

        Pipeline
        --------
        1) **Load data** via `FinancialForecastData`:
        - Weekly closes per ticker; compute weekly log returns.
        - Macroeconomic levels; attach countries; resample to weekly means.
        - Cross-sectional factor returns; resample to weekly means.
        - Firm revenue and EPS series; resample to weekly last.

        2) **FAVAR macro simulator** per country:
        - Transform levels to deltas (`_macro_levels_to_deltas_favar`).
        - Fit PCA basis (`_favar_fit`), keep up to 4 PCs.
        - Fit BVAR(2) under Minnesota prior (`_fit_BVAR_minnesota`).
        - Simulate S paths × K steps using posterior draws
            (`_sample_BSigma`, `_simulate_VAR`).
        - Store simulations in shared memory and record handles.

        3) **Factor simulator** (stationary bootstrap):
        - If factors available, draw indices via
            `stationary_bootstrap_indices(L, S, K, p)` and slice historical returns;
            otherwise use zeros. Store in shared memory.

        4) **Feature engineering and alignment**:
        - Compute higher-order return moments over rolling windows and lag by K.
        - Build weekly firm-level matrices for revenue and EPS.
        - For each ticker, align market dates to (macro, factors, firm data) using
            lower-bound indices; create masks of valid joint observations.

        5) **Pack state**:
        - Ticker list grouped by country, presence flags, weekly price arrays,
            factor arrays, moments, alignment caches, shared memory metadata for macro
            and factors, PCA basis metadata, and analyst target table.

        Returns
        -------
        dict
            A serialisable "state pack" dictionary used by worker processes.

        Notes
        -----
        Shared memory prevents duplication of large tensors across processes; only
        small metadata (shm name, shape, dtype) is pickled into the state pack.
        """
    
        self.logger.info("Building global state …")
    
        if self.SAVE_ARTIFACTS and not _os.path.exists(self.ARTIFACT_DIR):
    
            _os.makedirs(self.ARTIFACT_DIR, exist_ok=True)

        fdata = FinancialForecastData()
    
        macro = fdata.macro
    
        r = macro.r

        if self.tickers_arg:
    
            tickers = list(self.tickers_arg)
    
        else:
    
            tickers = ['GOOG', 'NVDA', 'TJX', 'TTWO']

        close_df = r.weekly_close
    
        dates_all = close_df.index.values
    
        tick_list = close_df.columns.values
    
        price_arr = close_df.to_numpy(dtype=np.float32, copy = False)
        T_dates, M = price_arr.shape

        price_rec = np.empty(
            T_dates * M,
            dtype = [("ds", "datetime64[ns]"),
                   ("Ticker", f"U{max(1, max(len(t) for t in tick_list) if len(tick_list) > 0 else 1)}"),
                   ("y", "float32")]
        )
        
        price_rec["ds"] = np.repeat(dates_all, M)
        
        price_rec["Ticker"] = np.tile(tick_list, T_dates)
        
        price_rec["y"] = price_arr.reshape(-1)
        
        price_rec = price_rec[~np.isnan(price_rec["y"])]

        analyst = r.analyst
        
        country_map = {t: str(c) for t, c in zip(analyst.index, analyst["country"])}
        
        country_arr = np.array([country_map.get(t, "") for t in price_rec["Ticker"]])
        
        price_rec = rfn.append_fields(price_rec, "country", country_arr, usemask = False)
        
        price_rec = price_rec[np.lexsort((price_rec["country"], price_rec["ds"]))]

        raw_macro = macro.assign_macro_history_large_non_pct().reset_index()
        
        raw_macro = raw_macro.rename(columns={"year": "ds"} if "year" in raw_macro else {raw_macro.columns[1]: "ds"})
        
        if isinstance(raw_macro.index, pd.PeriodIndex):
        
            raw_macro.index = raw_macro.index.to_timestamp(how="end")
        
        ds = raw_macro["ds"]
        
        if is_period_dtype(ds):
        
            raw_macro["ds"] = ds.dt.to_timestamp(how="end")
        
        elif is_datetime64_any_dtype(ds):
        
            pass
        
        elif ds.dtype == object and len(ds) and isinstance(ds.iloc[0], pd.Period):
        
            raw_macro["ds"] = pd.PeriodIndex(ds).to_timestamp(how="end")
        
        else:
        
            raw_macro["ds"] = pd.to_datetime(ds, errors="coerce")
        
        raw_macro["country"] = raw_macro["ticker"].map(country_map)
        
        macro_clean = raw_macro[["ds", "country"] + self.MACRO_REGRESSORS].dropna()
        
        len_macro_clean = len(macro_clean)

        macro_rec = np.empty(
            len_macro_clean,
            dtype=[("ds", "datetime64[ns]"),
                   ("country", f"U{max(1, max(len(c) for c in macro_clean['country']) if len_macro_clean > 0 else 1)}")]
                 + [(reg, "float32") for reg in self.MACRO_REGRESSORS]
        )
        
        macro_rec["ds"] = macro_clean["ds"].values
        
        macro_rec["country"] = macro_clean["country"].values
        
        for reg in self.MACRO_REGRESSORS:
        
            macro_rec[reg] = macro_clean[reg].to_numpy(dtype = np.float32, copy = False)
        
        macro_rec = macro_rec[np.lexsort((macro_rec["ds"], macro_rec["country"]))]

        unique_countries, first_idx = np.unique(macro_rec["country"], return_index = True)
        
        country_slices: Dict[str, Tuple[int, int]] = {}
        
        for i, ctry in enumerate(unique_countries):
        
            start = first_idx[i]
        
            end = first_idx[i + 1] if i + 1 < len(first_idx) else len(macro_rec)
        
            country_slices[ctry] = (start, end)

        country_var_results: Dict[str, Optional[dict]] = {}
        
        macro_future_by_country_meta: Dict[str, Optional[dict]] = {}
        
        macro_pca_basis: Dict[str, dict] = {}
        
        rng_global = np.random.default_rng(self.SEED)
        
        hor = self.HORIZON

        macro_weekly_by_country = {}
        
        macro_weekly_idx_by_country = {}

        for ctry, (s, e) in country_slices.items():
        
            rec = macro_rec[s:e]
        
            if len(rec) == 0:
        
                country_var_results[ctry] = None
        
                macro_future_by_country_meta[ctry] = None
        
                macro_weekly_by_country[ctry] = None
        
                macro_weekly_idx_by_country[ctry] = None
        
                continue

            dfm = pd.DataFrame(
                {reg: rec[reg] for reg in self.MACRO_REGRESSORS},
                index=pd.DatetimeIndex(rec["ds"])
            )

            dfw = (dfm[~dfm.index.duplicated(keep="first")]
                .sort_index()
                .resample("W-FRI").mean()
                .ffill()
                .dropna())

            macro_weekly_by_country[ctry] = dfw
        
            macro_weekly_idx_by_country[ctry] = dfw.index.values

            if dfw.shape[0] < 60:
        
                country_var_results[ctry] = None
        
                macro_future_by_country_meta[ctry] = None
        
                continue

            dX_macro = self._macro_levels_to_deltas_favar(
                df_levels = dfw
            ).dropna()
        
            dX_macro = dX_macro[self.MACRO_REGRESSORS].dropna()
            
            if dX_macro.shape[0] < 60:
            
                country_var_results[ctry] = None
            
                macro_future_by_country_meta[ctry] = None
            
                continue

            pcs, pca, sc = self._favar_fit(
                dX_macro = dX_macro, 
                max_pcs = 4
            )
            
            dX = pcs.dropna()
            
            if dX.shape[0] <= 60:
              
                country_var_results[ctry] = None
              
                macro_future_by_country_meta[ctry] = None
              
                continue

            post = self._fit_BVAR_minnesota(
                dX = dX, 
                p = 2
            )
            
            country_var_results[ctry] = {"post": post, "dX_last": dX.values[-post["p"]:]}

            S = self.N_SIMS
            
            rng_cty = np.random.default_rng(int(rng_global.integers(1_000_000_000)))
            
            sims_macro = np.zeros((S, hor, len(self.MACRO_REGRESSORS)), np.float32)

            for sidx in range(S):
            
                B, Sigma = self._sample_BSigma(
                    post = post, 
                    rng = rng_cty
                )
            
                pcs_path = self._simulate_VAR(
                    B = B,
                    Sigma = Sigma, 
                    dX_lags = country_var_results[ctry]["dX_last"].copy(),
                    steps = hor,
                    rng = rng_cty
                )
                
                try:
                
                    X_rec_std = pca.inverse_transform(
                        X = pcs_path
                    )
                
                except Exception:
                
                    X_rec_std = np.zeros((hor, dX_macro.shape[1]), np.float32)
                
                X_rec = (X_rec_std * sc.scale_) + sc.mean_
                
                sims_macro[sidx, :, :] = X_rec.astype(np.float32, copy = False)

            shm = shared_memory.SharedMemory(create = True, size = sims_macro.nbytes)
            
            buf = np.ndarray(sims_macro.shape, dtype = sims_macro.dtype, buffer = shm.buf)
            
            buf[:] = sims_macro
            
            macro_future_by_country_meta[ctry] = {
                "shm_name": shm.name,
                "shape": sims_macro.shape,
                "dtype": str(sims_macro.dtype),
            }
            
            self._created_shms.append(shm)
            
            macro_pca_basis[ctry] = {"macro_col_order": list(dX_macro.columns)}


        fac_w: pd.DataFrame = macro.r.factor_weekly_rets()
       
        if set(self.FACTORS).issubset(fac_w.columns):
       
            fac_vals = fac_w[self.FACTORS].dropna().to_numpy(dtype = np.float32)
       
        else:
            
            fac_vals = np.zeros((0, len(self.FACTORS)), np.float32)

        if fac_vals.size == 0:
       
            factor_future_global = np.zeros((self.N_SIMS, hor, len(self.FACTORS)), np.float32)
       
        else:
       
            L = fac_vals.shape[0]
       
            idx = self.stationary_bootstrap_indices(
                L = L,
                n_sims = self.N_SIMS, 
                H = hor, 
                p = self._BOOT_P,
                rng = rng_global
            )
       
            factor_future_global = fac_vals[idx, :]

        shm_factor = shared_memory.SharedMemory(create = True, size = factor_future_global.nbytes)
      
        buf_factor = np.ndarray(factor_future_global.shape, dtype = factor_future_global.dtype, buffer = shm_factor.buf)
      
        buf_factor[:] = factor_future_global
      
        self._created_shms.append(shm_factor)

        factor_weekly = (fac_w.sort_index().resample("W-FRI").mean().ffill())
      
        factor_weekly_index = factor_weekly.index.values
      
        if set(self.FACTORS).issubset(factor_weekly.columns):
      
            factor_weekly_values = factor_weekly[self.FACTORS].to_numpy(np.float32)
      
        else:
      
            factor_weekly_values = np.zeros((0, len(self.FACTORS)), np.float32)

      
        def _norm_country(
            x
        ) -> str:
        
            try:
        
                if pd.isna(x):
        
                    return "UNK"
        
            except Exception:
        
                pass
        
            s = str(x).strip()
        
            return s if s and s.lower() not in ("nan", "none") else "UNK"


        by_country: Dict[str, List[str]] = {}

        for t in tickers:

            c = _norm_country(analyst["country"].get(t, None))

            by_country.setdefault(c, []).append(t)

        grouped_tickers: List[str] = []

        for c in sorted(by_country.keys(), key=str):

            grouped_tickers.extend(sorted(by_country[c]))

        moments_by_ticker = {}

        weekly_price_by_ticker: Dict[str, Dict[str, np.ndarray]] = {}

        fd_weekly_by_ticker: Dict[str, Dict[str, np.ndarray]] = {}

        seq_len = self.SEQUENCE_LENGTH
        
        fin_raw = fdata.prophet_data

        fd_rec_dict: Dict[str, np.ndarray] = {}

        for t in tickers:  
          
            df_fd = (fin_raw.get(t, pd.DataFrame())
                    .reset_index()
                    .rename(columns = {"index": "ds", "rev": "Revenue", "eps": "EPS (Basic)"}))
          
            if df_fd.empty:
          
                continue

            df_fd["ds"] = pd.to_datetime(df_fd["ds"])
           
            df_fd = df_fd[["ds", "Revenue", "EPS (Basic)"]].dropna()
           
            if df_fd.empty:
           
                continue

            rec_fin = np.empty(
                len(df_fd),
                dtype = [("ds", "datetime64[ns]"),
                    ("Revenue", "float32"),
                    ("EPS (Basic)", "float32")]
            )
          
            rec_fin["ds"] = df_fd["ds"].values
          
            rec_fin["Revenue"] = df_fd["Revenue"].to_numpy(dtype = np.float32, copy = False)
          
            rec_fin["EPS (Basic)"] = df_fd["EPS (Basic)"].to_numpy(dtype = np.float32, copy = False)
          
            fd_rec_dict[t] = np.sort(rec_fin, order = "ds")
            
        for t in grouped_tickers:
            
            pr = price_rec[price_rec["Ticker"] == t]
            
            if len(pr) == 0:
            
                continue
            
            s = (pd.DataFrame({
                "ds": pr["ds"], 
                "y": pr["y"]
            }).set_index("ds").sort_index()["y"] .resample("W-FRI").last().ffill())
            
            y = s.to_numpy(dtype = np.float32, copy = False)
            
            ys = np.maximum(y, self.SMALL_FLOOR)
            
            lr = np.zeros_like(ys, dtype = np.float32)
            
            lr[1:] = np.log(ys[1:]) - np.log(ys[:-1])
            
            weekly_price_by_ticker[t] = {
                "index": s.index.values, 
                "y": y, 
                "lr": lr
            }

            s_lr = pd.Series(lr, index = s.index)
           
            skew = s_lr.rolling(seq_len, min_periods = seq_len).skew().shift(self.HORIZON)
           
            kurt = s_lr.rolling(seq_len, min_periods = seq_len).kurt().shift(self.HORIZON)
           
            moments_by_ticker[t] = np.column_stack((
                np.nan_to_num(skew.to_numpy(dtype = np.float32), nan = 0.0, posinf = 0.0, neginf = 0.0),
                np.nan_to_num(kurt.to_numpy(dtype = np.float32), nan = 0.0, posinf = 0.0, neginf = 0.0),
            )).astype(np.float32, copy = False)
            
            if t in fd_rec_dict:
              
                df_fd = (pd.DataFrame({
                    "ds": fd_rec_dict[t]["ds"],
                    "Revenue": fd_rec_dict[t]["Revenue"],
                    "EPS (Basic)": fd_rec_dict[t]["EPS (Basic)"],
                }).set_index("ds").sort_index())
              
                fdw = df_fd.resample("W-FRI").last().ffill()
                
                fd_weekly_by_ticker[t] = {
                    "index": fdw.index.values,
                    "values": fdw[["Revenue", "EPS (Basic)"]].to_numpy(dtype = np.float32, copy = False)
                }
                
        align_cache = {}
        
        for t in grouped_tickers:
        
            wp = weekly_price_by_ticker.get(t)
        
            if wp is None:
        
                continue
        
            dates = wp["index"]
        
            ctry_t = analyst["country"].get(t, None)
        
            dfw = macro_weekly_by_country.get(ctry_t)
        
            mw_idx = macro_weekly_idx_by_country.get(ctry_t)
        
            if dfw is None or mw_idx is None:
        
                continue
        
            idx_m = np.searchsorted(mw_idx, dates, side = "right") - 1
            
            valid_m = idx_m >= 0
            
            if factor_weekly_values.shape[0] > 0:
            
                idx_fa = np.searchsorted(factor_weekly_index, dates, side = "right") - 1
            
                valid_fa = idx_fa >= 0
            
            else:
            
                idx_fa = np.zeros_like(idx_m)
            
                valid_fa = np.ones_like(idx_m, dtype=bool)
           
            if t in fd_weekly_by_ticker:
           
                fd_idx = fd_weekly_by_ticker[t]["index"]
           
                idx_fd = np.searchsorted(fd_idx, dates, side = "right") - 1
              
                valid_fd = idx_fd >= 0
           
            else:
           
                idx_fd = None
           
                valid_fd = np.ones(len(dates), bool)
           
            keep = valid_m & valid_fa & (valid_fd if idx_fd is not None else True)
           
            idx_keep = np.nonzero(keep)[0]
           
            align_cache[t] = {
                "idx_m": idx_m,
                "valid_m": valid_m,
                "idx_fa": idx_fa, 
                "valid_fa": valid_fa,
                "idx_fd": idx_fd, 
                "valid_fd": valid_fd,
                "idx_keep": idx_keep
            }

        state_pack: Dict[str, Any] = {
            "tickers": grouped_tickers,
            "macro_rec": macro_rec,
            "country_slices": country_slices,
            "macro_future_by_country_meta": macro_future_by_country_meta,
            "fd_rec_dict": fd_rec_dict,
            "next_fc": fdata.next_period_forecast(),
            "latest_price": r.last_price,
            "analyst": analyst,
            "macro_weekly_by_country": macro_weekly_by_country,
            "moments_by_ticker": moments_by_ticker,
            "factor_weekly_values": factor_weekly_values,
            "align_cache": align_cache,
            "weekly_price_by_ticker": weekly_price_by_ticker,
            "fd_weekly_by_ticker": fd_weekly_by_ticker,
        }
        
        presence_flags = {
            t: {
                "has_factors": (factor_weekly_values.shape[0] > 0),
                "has_fin": (t in fd_weekly_by_ticker),
                "has_moms": (t in moments_by_ticker),
            } for t in grouped_tickers
        }
        
        state_pack["presence_flags"] = presence_flags
        
        state_pack["scaler_cache"] = {}

        self.logger.info("Global state built (%d tickers).", len(grouped_tickers))
        
        state_pack["factor_future_meta"] = {
            "shm_name": shm_factor.name,
            "shape": factor_future_global.shape,
            "dtype": str(factor_future_global.dtype),
        }

        state_pack["macro_favar_basis"] = macro_pca_basis
        
        return state_pack


    def forecast_one(
        self, 
        ticker: str
    ) -> Dict[str, Any]:
        """
        Produce a distributional K-step price forecast for a single ticker.

        Process
        -------
        1) **Validation and alignment.** Retrieve the country, align ticker's weekly
        dates to macro/factor/financial calendars, and check sample sufficiency.

        2) **Construct design tensors.**
       
        - Build `reg_mat` of level features; convert to delta features
            `DEL_full = build_delta_matrix(reg_mat)`.
       
        - Create rolling windows of length H for historical returns and H+K for
            delta features using `sliding_window_view`.
       
        - Form training windows (train/validation split via `choose_splits`).

        3) **Scaling and residualisation.**
      
        - Fit robust scalers for deltas and returns on training coverage; clip
            deltas at [1%, 99%].
      
        - Compute AR(1) baseline parameters (m, φ) via OLS; compute μ_base for each
            window and residual targets y_res = y − μ_base.

        4) **Model fitting.** Build the LSTM via `build_directH_model`, compile with
        a weighted sum of pinball loss and Student-t NLL, and fit with early
        stopping.

        5) **Calibration.** Run validation forward pass to compute predicted Student-t
        parameters; set `_SIGMA_CAL_SCALE` so that median predicted σ matches a
        robust estimate of residual dispersion.

        6) **Scenario generation.**
       
        - Macro deltas: read pre-simulated FAVAR paths from shared memory; if
            unavailable, draw from a (shrunk) Gaussian with covariance estimated by
            Ledoit–Wolf over the last 52 deltas.
       
        - Factor deltas: read stationary-bootstrap paths; if unavailable, zeros.
       
        - Revenue/EPS deltas: centre on combined analyst targets with Gaussian
            noise in log / signed-log space, spread evenly across K steps.

        7) **Monte-Carlo sampling with dropout.**
       
        - For each scenario block, concatenate historical segment (H steps) with
            future exogenous deltas (K steps) to form an (H+K)×(1+n_reg) input.
       
        - Repeat each block `mc_dropout_samples` times and perform a single
            `training=True` forward pass to sample parameters (i.e., per-example
            dropout masks).
        
        - Draw ε from the Student-t per sample and add μ_base (equation (2)) to get
            y (cumulative log return). Convert to simple returns R = exp(y) − 1.

        8) **Aggregation to prices.**
       
        - Compute q05, q50, q95 of R to form "Min/Avg/Max Price" as
         
            P_q = (1 + q) × P_0,
          
            where P_0 is the latest price.
       
        - Trim returns to [q05, q95] to compute robust mean and standard deviation.

        Returns
        -------
        dict
            A result with quantile prices, expected return and dispersion, or a skip
            record if data are insufficient.

        Why direct multi-horizon
        ------------------------
        Directly modelling the sum over the horizon avoids iterative error
        accumulation inherent in recursive 1-step forecasts and lets future exogenous
        paths condition the entire horizon at once.
        """
    
        assert STATE is not None, "Worker STATE not initialized"


        def skip(
            reason: str
        ) -> Dict[str, Any]:
        
            return {
                "Ticker": ticker, 
                "status": "skipped", 
                "reason": reason
            }


        try:
            
            rng = np.random.default_rng(self.SEED ^ (hash(ticker) & 0xFFFFFFFF))
          
            self.logger.info(
                "%s: RAM %.1f / %.1f GB",
                ticker,
                psutil.Process().memory_info().rss / (1024 ** 3),
                psutil.virtual_memory().total / (1024 ** 3)
            )

            HIST = self.HIST_WINDOW
            
            HOR = self.HORIZON
          
            flags = STATE["presence_flags"][ticker]
          
            macro_rec = STATE["macro_rec"]
           
            country_slices = STATE["country_slices"]
           
            fd_rec_dict = STATE["fd_rec_dict"]
           
            next_fc = STATE["next_fc"]
           
            latest_price = STATE["latest_price"]
           
            analyst = STATE["analyst"]
           
            fa_vals = STATE["factor_weekly_values"]
           
            moms_vals = STATE["moments_by_ticker"].get(ticker)
           
            align = STATE["align_cache"].get(ticker)

            if align is None:
           
                return skip(
                    reason = "alignment_missing"
                )

            ctry = analyst["country"].get(ticker)
           
            if ctry not in country_slices:
           
                return skip(
                    reason = "no_country_slice"
                )

            s, e = country_slices[ctry]
           
            dfm_ct = STATE["macro_weekly_by_country"].get(ctry)
           
            if dfm_ct is None or dfm_ct.shape[0] < 12:
           
                return skip(
                    reason = "insufficient_macro_history"
                )

            macro_meta = STATE["macro_future_by_country_meta"].get(ctry, None)
          
            macro_future_deltas = None
          
            macro_col_order = None
          
            macro_basis = STATE.get("macro_favar_basis", {}).get(ctry, {})
          
            if macro_meta:
          
                try:
          
                    shm_m = shared_memory.SharedMemory(name = macro_meta["shm_name"])
          
                    macro_future_deltas = np.ndarray(shape = tuple(macro_meta["shape"]),
                                                     dtype = np.dtype(macro_meta["dtype"]),
                                                     buffer = shm_m.buf)
                
                    macro_col_order = macro_basis.get("macro_col_order", list(self.MACRO_REGRESSORS))
                
                except Exception:
                
                    macro_future_deltas = None

            fac_meta = STATE.get("factor_future_meta", None)
           
            factor_future = None
           
            if fac_meta:
           
                try:
           
                    shm_f = shared_memory.SharedMemory(name = fac_meta["shm_name"])
                    
                    factor_future = np.ndarray(shape = tuple(fac_meta["shape"]),
                                               dtype = np.dtype(fac_meta["dtype"]),
                                               buffer = shm_f.buf)
              
                except Exception:
              
                    factor_future = None

            S = self.N_SIMS

            if macro_future_deltas is None:
            
                dX_macro_fbk = self._macro_levels_to_deltas_favar(
                    df_levels = dfm_ct
                ).dropna()
                
                dX_macro_fbk = dX_macro_fbk[self.MACRO_REGRESSORS].dropna()
                
                tail = dX_macro_fbk.tail(52).to_numpy(np.float32, copy = False)
                
                Sigma_f = self._lw_cov_or_eye(
                    tail_vals = tail if len(tail) else None,
                    dim = len(self.MACRO_REGRESSORS)
                )
                
                Sigma_f = self.ensure_spd_for_cholesky(
                    Sigma = Sigma_f
                )
                
                rng_fbk = np.random.default_rng(self.SEED ^ 0xFACEB00C)
                
                macro_future_deltas = rng_fbk.multivariate_normal(
                    mean = np.zeros(len(self.MACRO_REGRESSORS), np.float32),
                    cov = Sigma_f,
                    size = (S, HOR),
                    method = "cholesky"
                ).astype(np.float32)
                
                macro_col_order = list(self.MACRO_REGRESSORS)

            if factor_future is None:
               
                factor_future = np.zeros((S, HOR, len(self.FACTORS)), np.float32)

            idx_m = np.asarray(align["idx_m"], dtype = np.int64)
            
            idx_fa = np.asarray(align["idx_fa"], dtype = np.int64)
            
            idx_keep = np.asarray(align["idx_keep"], dtype = np.int64)
            
            idx_fd = align["idx_fd"]
            
            if idx_fd is not None:
            
                idx_fd = np.asarray(idx_fd, dtype = np.int64)

            sel_m = idx_m[idx_keep]
           
            sel_fa = idx_fa[idx_keep]
           
            if idx_fd is not None:
           
                sel_fd = idx_fd[idx_keep]
           
            else:
           
                sel_fd = None

            cur_p = latest_price.get(ticker, np.nan)
           
            if not np.isfinite(cur_p):
           
                return skip(
                    reason = "no_latest_price"
                )

            wp = STATE["weekly_price_by_ticker"].get(ticker)
            
            if wp is None:
            
                return skip(
                    reason = "no_price_history_weekly"
                )
            
            yv = wp["y"]

            macro_ct = macro_rec[s:e]
           
            if len(macro_ct) < 12:
           
                return skip(
                    reason = "insufficient_macro_history"
                )

            SEQ_LEN = self.SEQUENCE_LENGTH
           
            if len(yv) < SEQ_LEN + 2:
           
                return skip(
                    reason = "short_price_history"
                )

            if len(idx_keep) < SEQ_LEN + 1:
                
                return skip(
                    reason = "insufficient_joint_features"
                )

            sel_m = np.clip(sel_m, 0, len(dfm_ct) - 1)
         
            if fa_vals.shape[0] > 0:
         
                sel_fa = np.clip(sel_fa, 0, fa_vals.shape[0] - 1)
         
            else:
         
                sel_fa = np.zeros_like(sel_m)

            if ticker in fd_rec_dict:
         
                fdw = STATE["fd_weekly_by_ticker"].get(ticker, None)
         
                if fdw is not None:
         
                    sel_fd = np.clip(sel_fd, 0, len(fdw["index"]) - 1) if sel_fd is not None else None
         
            else:
         
                fdw = None
         
                sel_fd = None

            n_m = len(dfm_ct)
         
            n_fa = fa_vals.shape[0]
         
            if flags["has_factors"]:
         
                upper_ok = (sel_m < n_m) & (sel_fa < n_fa)
         
            else:
         
                upper_ok = (sel_m < n_m)

            idx_keep = idx_keep[upper_ok]
         
            sel_m = sel_m[upper_ok]
         
            sel_fa = sel_fa[upper_ok]
         
            if sel_fd is not None:
         
                sel_fd = sel_fd[upper_ok]

            if len(idx_keep) < SEQ_LEN + 1:
         
                return skip(
                    reason = "insufficient_joint_features_after_upperbound"
                )

            reg_pos = self.reg_pos
           
            n_reg = self.n_reg
           
            n_ch = 1 + n_reg

            reg_mat = np.zeros((len(idx_keep), n_reg), dtype = np.float32)
            
            macro_vals = dfm_ct[self.MACRO_REGRESSORS].to_numpy(np.float32, copy = False)
            
            reg_mat[:, self._macro_idx] = macro_vals[sel_m]

            if flags["has_factors"] and fa_vals.shape[0] > 0:
            
                reg_mat[:, self._factor_idx] = fa_vals[sel_fa, :]

            if flags["has_fin"] and (fdw is not None) and (align["idx_fd"] is not None):
            
                reg_mat[:, self._fin_idx] = fdw["values"][sel_fd, :]

            if flags["has_moms"] and moms_vals is not None:
            
                reg_mat[:, self._mom_idx] = moms_vals[idx_keep, :]

            if reg_mat.shape[0] < (SEQ_LEN + 4):
            
                return skip(
                    reason = "too_short_after_align"
                )

            lr_full = STATE["weekly_price_by_ticker"][ticker]["lr"][idx_keep].astype(np.float32)
          
            lr_core = lr_full[1:]
          
            last_r_now = float(lr_core[-1])

            DEL_full = self.build_delta_matrix(
                reg_mat = reg_mat,
                regs = self.regs
            )

            T_reg = DEL_full.shape[0]     
           
            SEQ_LEN = HIST + HOR
            
            n_all = T_reg - SEQ_LEN + 1
            
            if n_all <= 1:
            
                return skip(
                    reason = "no_training_windows"
                )

            n_splits = self.choose_splits(
                N = n_all
            )
            
            if n_splits < 2:
              
                cut = max(1, int(0.8 * n_all))
              
                train_idx = np.arange(cut)
              
                cal_idx = np.arange(cut, n_all)
           
            else:
            
                tscv = TimeSeriesSplit(n_splits = n_splits)
            
                train_idx, cal_idx = list(tscv.split(np.arange(n_all)))[-1]

            start = train_idx[0]
           
            end = min(train_idx[-1] + SEQ_LEN - 1, T_reg - 1)
           
            delta_mask = np.zeros(T_reg, dtype = bool)
           
            delta_mask[start:end+1] = True


            if delta_mask.sum() < n_reg:
                
                return skip(
                    reason = "too_few_training_deltas_for_scaler"
                )

            sc_reg_full, ql_full, qh_full = self.fit_scale_deltas(
                DEL_tr = DEL_full[delta_mask]
            )
           
            scaled_reg_full = self.transform_deltas(
                DEL = DEL_full,
                sc = sc_reg_full, 
                q_low = ql_full, 
                q_high = qh_full
            )

            end_r = min(train_idx[-1] + HIST - 1, T_reg - 1)
            
            ret_mask = np.zeros(T_reg, dtype = bool)
            
            ret_mask[start:end_r + 1] = True
            
            if ret_mask.sum() < 8:
               
                ret_scaler_full = self.fit_ret_scaler_from_logret(
                    log_ret = lr_core
                )
          
            else:
          
                sc = RobustScaler().fit(lr_core[ret_mask].reshape(-1, 1))
          
                sc.scale_[sc.scale_ < 1e-6] = 1e-6
          
                ret_scaler_full = sc

            scaled_ret_core = ((lr_core - ret_scaler_full.center_) / ret_scaler_full.scale_).astype(np.float32)

            pr_view  = sliding_window_view(scaled_ret_core, HIST)                       
          
            reg_view = sliding_window_view(scaled_reg_full, (HIST + HOR, n_reg))[:, 0]   
          
            fr_view  = sliding_window_view(lr_core, HOR)                              

           
            def _build_X_for(
                idx: np.ndarray
            ) -> np.ndarray:
            
                X = np.empty((len(idx), SEQ_LEN, 1 + n_reg), np.float32)
            
                X[:, :HIST, 0]  = pr_view[idx]
            
                X[:, HIST:, 0]  = 0.0
            
                X[:, :HIST, 1:] = reg_view[idx, :HIST, :]
            
                X[:, HIST:, 1:] = reg_view[idx, HIST:, :]
            
                return X

            
            def _build_y_for(
                idx: np.ndarray
            ) -> np.ndarray:

                y = np.sum(fr_view[idx + HIST], axis = 1, keepdims = True).astype(np.float32)
            
                return y


            X_tr = _build_X_for(
                idx = train_idx
            )
            
            y_tr = _build_y_for(
                idx = train_idx
            )
            
            X_cal = _build_X_for(
                idx = cal_idx
            )
            
            y_cal = _build_y_for(
                idx = cal_idx
            )

            last_train_end = train_idx[-1] + HIST - 1
           
            r_train = lr_core[: last_train_end + 1]
           
            m_hat, phi_hat = self.fit_ar1_baseline(
                log_ret = r_train
            )

            
            def ar1_sum(
                r_t, 
                H
            ):
            
                return self.ar1_sum_forecast(
                    r_t = r_t,
                    m = m_hat,
                    phi = phi_hat, 
                    H = H
                )


            last_r_tr = lr_core[train_idx + HIST - 1]
           
            last_r_cal = lr_core[cal_idx + HIST - 1]
           
            last_r_now = float(lr_core[-1])      

            mu_ar1_tr  = np.array([ar1_sum(
                r_t = rt, 
                H = HOR
            ) for rt in last_r_tr], dtype = np.float32)[:, None]
           
            mu_ar1_cal = np.array([ar1_sum(
                r_t = rt,
                H = HOR
            ) for rt in last_r_cal], dtype = np.float32)[:, None]

            mu_base_tr = mu_ar1_tr
           
            mu_base_cal = mu_ar1_cal
           
            y_tr_res = y_tr - mu_base_tr
           
            y_cal_res = y_cal - mu_base_cal

            X_hist = np.empty((1, HIST, n_ch), dtype = np.float32)
           
            X_hist[0, :, 0]  = scaled_ret_core[-HIST:]
           
            X_hist[0, :, 1:] = self._last_hist_reg_deltas(
                delta_reg = scaled_reg_full,
                hist = HIST,
                hor = HOR, 
                seq_length = HIST + HOR
            )

            global_cache = self._MODEL_CACHE
           
            cached = global_cache.get(n_reg)
           
            if cached is None:
           
                model = self.build_directH_model(
                    n_reg = n_reg,
                    seed = self.SEED
                )
           
                opt = Adam(
                    learning_rate = self._LR,
                    clipnorm = 1.0
                )
           
                model.compile(
                    optimizer = opt,
                    loss = {
                        "q_head": self.pinball_loss, 
                        "dist_head": self.student_t_nll
                    },
                    
                    loss_weights = {
                        "q_head": 0.8, 
                        "dist_head": 0.2
                    },
                )
                
                init_w = [w.copy() for w in model.get_weights()]
                
                global_cache[n_reg] = (model, init_w)
         
            model, init_w = global_cache[n_reg]
         
            model.set_weights(init_w)

            spec = tf.TensorSpec(shape = (None, SEQ_LEN, n_ch), dtype = tf.float32)
        
            if n_reg not in self._FN_CACHE:
        
        
                @tf.function(reduce_retracing = True, input_signature = [spec])
                def _fwd(
                    x
                ):
                
                    return model(x, training = False)


                @tf.function(reduce_retracing = True, input_signature = [spec])
                def _fwd_train(
                    x
                ):
                
                    return model(x, training = True)

                self._FN_CACHE[n_reg] = {
                    "fwd": _fwd, 
                    "fwd_train": _fwd_train
                }
           
            fns = self._FN_CACHE[n_reg]

            effective_epochs = self.EPOCHS if len(X_tr) > 4 * self.BATCH else max(8, self.EPOCHS // 2)
           
            callbacks = self._make_callbacks(
                monitor = "val_loss"
            )
           
            model.fit(
                X_tr, 
                {
                    "q_head": y_tr_res, 
                    "dist_head": y_tr_res
                },
                epochs = effective_epochs, callbacks = callbacks, 
                verbose = 0,
                validation_data = (X_cal, {
                    "q_head": y_cal_res, 
                    "dist_head": y_cal_res
                })
            )

            outs_cal = fns["fwd"](tf.convert_to_tensor(X_cal))
           
            q_cal = (outs_cal["q_head"] if isinstance(outs_cal, dict) else outs_cal[0]).numpy().astype(np.float32)
           
            params_cal = (outs_cal["dist_head"] if isinstance(outs_cal, dict) else outs_cal[1]).numpy().astype(np.float32)
           
            q_cal = q_cal + mu_base_cal

            mu_c = self.MU_MAX * np.tanh(params_cal[:, 0:1])
           
            sigma_c = np.minimum(self.SIGMA_MAX, np.log1p(np.exp(params_cal[:, 1:2])) + self.SIGMA_FLOOR)
           
            resid = (y_cal_res.ravel() - mu_c.ravel())
           
            pred_sd = sigma_c.ravel()
           
            num = np.nanmedian(np.abs(resid)) / 0.6745
           
            den = max(np.nanmedian(pred_sd), 1e-6)
           
            self._SIGMA_CAL_SCALE = float(np.clip(num / den, 0.5, 1.2))

            deltas_future_all = np.zeros((S, HOR, n_reg), np.float32)

            if macro_future_deltas is not None:
           
                name2idx = {n: i for i, n in enumerate(macro_col_order or self.MACRO_REGRESSORS)}
           
                for m in self.MACRO_REGRESSORS:
           
                    if m in reg_pos and m in name2idx:
           
                        deltas_future_all[:, :, reg_pos[m]] = macro_future_deltas[:, :, name2idx[m]]

            if factor_future is not None and len(self.FACTORS):
           
                for f_idx, f in enumerate(self.FACTORS):
           
                    if f in reg_pos:
           
                        deltas_future_all[:, :, reg_pos[f]] = factor_future[:, :, f_idx]

            row_fore = next_fc.loc[ticker] if (ticker in next_fc.index) else pd.Series()
           
            n_yahoo = float(row_fore.get("num_analysts_y", np.nan))
           
            n_sa = float(row_fore.get("num_analysts", np.nan))

            if flags["has_fin"] and (fdw is not None) and (align["idx_fd"] is not None):
           
                last_rev = float(fdw["values"][sel_fd][-1, 0])
           
                last_eps = float(fdw["values"][sel_fd][-1, 1])
           
            else:
           
                last_rev = float(self.SMALL_FLOOR)
           
                last_eps = float(self.SMALL_FLOOR)

            comb = _analyst_sigmas_and_targets_combined(
                n_yahoo = n_yahoo,
                n_sa = n_sa,
                row_fore = row_fore
            )

            rng_scn = np.random.default_rng(self.SEED ^ 0xC1A0CAFE)
            
            d_rev = np.zeros((S, HOR), np.float32)
            
            if np.isfinite(comb["targ_rev"]) and last_rev > 0:
            
                mu_r = np.log(max(comb["targ_rev"], 1e-12)) - np.log(max(last_rev, 1e-12))
            
                dT_r = rng_scn.normal(loc=mu_r, scale=comb["rev_sigma"], size=S).astype(np.float32)
            
                d_rev[:] = (dT_r / HOR)[:, None]

            d_eps = np.zeros((S, HOR), np.float32)
            
            if np.isfinite(comb["targ_eps"]):
            
                mu_e = self.slog1p_signed(
                    x = np.array([comb["targ_eps"]])
                )[0] - self.slog1p_signed(
                    x = np.array([last_eps])
                )[0]
            
                dT_e = rng_scn.normal(loc = mu_e, scale = comb["eps_sigma"], size = S).astype(np.float32)
            
                d_eps[:] = (dT_e / HOR)[:, None]

            if "Revenue" in reg_pos:
        
                deltas_future_all[:, :, reg_pos["Revenue"]] = d_rev
        
            if "EPS (Basic)" in reg_pos:
        
                deltas_future_all[:, :, reg_pos["EPS (Basic)"]] = d_eps

            for mom in self.MOMENT_COLS:
        
                if mom in reg_pos:
        
                    j = reg_pos[mom]
        
                    mom_lastH = DEL_full[-HOR:, j]

                    deltas_future_all[:, :, j] = mom_lastH[None, :]

            deltas_future_all = self.transform_deltas(
                DEL = deltas_future_all.reshape(-1, n_reg),
                sc = sc_reg_full, 
                q_low = ql_full, 
                q_high = qh_full
            ).reshape(S, HOR, n_reg)

            hist_template = np.broadcast_to(X_hist, (1, HIST, n_ch))[0]
            
            zero_future_ret = np.zeros((S, HOR, 1), dtype = np.float32)
            
            mc = self.MC_DROPOUT_SAMPLES
            
            all_samples = []
            
            sim_chunk = 256
            
            mu_base_sum = float(self.ar1_sum_forecast(
                r_t = last_r_now,
                m = m_hat, 
                phi = phi_hat, 
                H = HOR
            ))

            for s0 in range(0, S, sim_chunk):
               
                s1 = min(s0 + sim_chunk, S)
               
                w = s1 - s0

                hist_chunk = np.broadcast_to(hist_template, (w, HIST, n_ch))
               
                future_reg = deltas_future_all[s0:s1]           
               
                future_ret = zero_future_ret[s0:s1]                  
               
                future_chunk = np.concatenate([future_ret, future_reg], axis = 2) 
               
                X_block = np.concatenate([hist_chunk, future_chunk], axis=1)     
                
                mu_base_block = np.full((w, 1), mu_base_sum, dtype = np.float32)

                X_block_tf = tf.convert_to_tensor(np.repeat(X_block, mc, axis = 0)) 
                
                outs = fns["fwd_train"](X_block_tf)
                
                dp = (outs["dist_head"] if isinstance(outs, dict) else outs[1]).numpy().astype(np.float32)
                
                dp = dp.reshape(mc, w, -1)
                
                mu = self.MU_MAX * np.tanh(dp[..., 0:1])
                
                sigma = np.log1p(np.exp(dp[..., 1:2])) + self.SIGMA_FLOOR
                
                sigma = np.minimum(self.SIGMA_MAX, sigma * getattr(self, "_SIGMA_CAL_SCALE", 1.0))
                
                nu = self.NU_FLOOR + np.log1p(np.exp(dp[..., 2:3]))
                
                t_draws = rng.standard_t(df = nu).astype(np.float32)
                
                samples_total = mu + sigma * t_draws + mu_base_block[None, ...]  
                
                all_samples.append(samples_total.reshape(-1, 1))

            logR_flat = np.concatenate(all_samples, axis = 0).ravel()
            
            rets = np.expm1(logR_flat)

            q05, q50, q95 = np.nanquantile(rets, [0.05, 0.5, 0.95])
            
            rets_trim = np.clip(rets, q05, q95)

            p_lower_raw = float((q05 + 1) * cur_p)
            
            p_median_raw = float((q50 + 1) * cur_p)
            
            p_upper_raw = float((q95 + 1) * cur_p)

            ret_exp = float(np.nanmean(rets_trim))
            
            se_ret = float(np.nanstd(rets_trim, ddof = 0))

            if self.SAVE_ARTIFACTS:
                try:
                    np.savez_compressed(
                        _os.path.join(self.ARTIFACT_DIR, f"{ticker}_artifacts_directH.npz"),
                        scaler_reg_center = sc_reg_full.center_.astype(np.float32),
                        scaler_reg_scale = sc_reg_full.scale_.astype(np.float32),
                        scaler_ret_center = ret_scaler_full.center_.astype(np.float32),
                        scaler_ret_scale = ret_scaler_full.scale_.astype(np.float32),
                        q_low = ql_full, 
                        q_high = qh_full,
                        hist = HIST, 
                        hor = HOR, 
                        n_reg = n_reg,
                    )
                    
                except Exception as ex:
                    
                    self.logger.warning("%s: artifact save failed: %s", ticker, ex)

            return {
                "Ticker": ticker,
                "status": "ok",
                "Min Price": float(p_lower_raw),
                "Avg Price": float(p_median_raw),
                "Max Price": float(p_upper_raw),
                "Returns": ret_exp,
                "SE": se_ret,
            }

        except Exception as e:
           
            self.logger.error("Error processing %s: %s", ticker, e)
           
            gc.collect()
           
            return {"Ticker": ticker, "status": "error", "reason": str(e)}


    @staticmethod
    def _last_hist_reg_deltas(
        delta_reg: np.ndarray,
        hist: int,
        hor: int, 
        seq_length: int
    ) -> np.ndarray:
        """
        Extract the last H rows of delta features aligned to the immediate past.

        Given the full delta matrix of shape (T_reg, n_reg), return the submatrix that
        fills the historical part (length H) of the (H+K) sequence, immediately
        preceding the K future steps.

        Parameters
        ----------
        delta_reg : ndarray
        hist : int
            Historical length H.
        hor : int
            Horizon K.
        seq_length : int
            H + K.

        Returns
        -------
        ndarray, shape (H, n_reg)
            The historical block used to initialise the sequence input.
        """
    
        start = max(0, delta_reg.shape[0] - seq_length)
    
        end = max(start, delta_reg.shape[0] - hor)
    
        out = delta_reg[start:end, :]
    
        if out.shape[0] < hist and delta_reg.shape[0] >= seq_length:
    
            out = delta_reg[-(seq_length):-hor, :]
    
        if out.shape[0] != hist:
    
            out = delta_reg[-hist:, :]
    
        return out.astype(np.float32, copy = False)


    @staticmethod
    def stationary_bootstrap_indices(
        L: int,
        n_sims: int, 
        H: int,
        p: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate stationary-bootstrap indices for block-resampling time series.

        Implements Politis & Romano's stationary bootstrap. For each simulation and
        each step h=1..H:
       
        - With probability p, start a new block at a random index S₀ ∼ Uniform{0..L−1}.
       
        - Otherwise advance by one from the last index (wrapping modulo L).

        Parameters
        ----------
        L : int
            Length of the source series to resample from.
        n_sims : int
            Number of simulated paths.
        H : int
            Path length (forecast horizon K).
        p : float
            Restart probability (expected block length 1/p).
        rng : np.random.Generator
            Random generator.

        Returns
        -------
        ndarray, shape (n_sims, H)
            Integer indices into the source series to assemble resampled paths.

        Notes
        -----
        Preserves short-range dependence while randomising block boundaries, better
        reflecting empirical serial correlation than IID sampling.
        """
    
        if L <= 0 or H <= 0 or n_sims <= 0:
    
            return np.zeros((n_sims, H), dtype = np.int64)
    
        R = (rng.random((n_sims, H)) < p)
        
        R[:, 0] = True
        
        G = np.cumsum(R, axis = 1) - 1
        
        group_starts = rng.integers(0, L, size = (n_sims, H), dtype = np.int64)
        
        S0 = np.take_along_axis(group_starts, G, axis = 1)
        
        t_idx = np.arange(H, dtype = np.int64)[None, :]
        
        last_restart_idx = np.maximum.accumulate(np.where(R, t_idx, -1), axis = 1)
        
        O = t_idx - last_restart_idx
        
        pos = (S0 + O) % L
        
        return pos.astype(np.int64, copy = False)
    

    def run(self):
        """
        Run the end-to-end forecasting pipeline across all tickers in parallel.

        Steps
        -----
        1) Enable fault handler; build the global state via `build_state`.
        
        2) Spawn a process pool (start method 'spawn') and initialise each worker with
        the state.
        
        3) Submit `forecast_one` tasks for all tickers; collect results with timeouts.
        
        4) Aggregate successes and failure reasons to two DataFrames.
        
        5) Optionally write results to Excel.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            (df_ok, df_bad) where df_ok contains quantile prices, expected returns and
            SE per ticker, and df_bad lists skip/error reasons.

        Notes
        -----
        The pool size is capped to a small number to limit CPU and memory contention,
        particularly important when TensorFlow is present in workers.
        """

        faulthandler.enable()

        state_pack = self.build_state()

        tickers: List[str] = state_pack["tickers"]

        ctx = mp.get_context("spawn")

        max_workers = min(4, _os.cpu_count())

        self.logger.info("Starting pool with %d workers …", max_workers)

        results: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(
            max_workers = max_workers,
            mp_context = ctx,
            initializer = _init_worker,      
            initargs = (state_pack,)
        ) as exe:
           
            futures = [exe.submit(_worker_forecast_one, t) for t in tickers]
           
            for fut in as_completed(futures):
               
                try:
               
                    res = fut.result(timeout = 1800)
               
                    results.append(res)
               
                    if res.get("status") == "ok":
               
                        self.logger.info(
                            "Ticker %s: Min %.2f, Avg %.2f, Max %.2f, Return %.4f, SE %.4f",
                            res["Ticker"], res["Min Price"], res["Avg Price"], res["Max Price"], res["Returns"],  res["SE"]
                        )
                 
                    else:
                 
                        self.logger.info("Ticker %s: %s (%s)", res.get("Ticker"), res.get("status"), res.get("reason"))
                
                except TimeoutError:
                
                    self.logger.error("A worker timed out; continuing.")
                
                except Exception as ex:
                
                    self.logger.error("Worker failed: %s", ex)

        ok_rows = [r for r in results if r.get("status") == "ok"]
       
        bad_rows = [r for r in results if r.get("status") != "ok"]

        if ok_rows:
  
            df_ok = pd.DataFrame(ok_rows).set_index("Ticker")[["Min Price", "Avg Price", "Max Price", "Returns", "SE"]]
  
        else:
  
            df_ok = pd.DataFrame(columns = ["Min Price", "Avg Price", "Max Price", "Returns", "SE"])

        if bad_rows:

            df_bad = pd.DataFrame(bad_rows).set_index("Ticker")[["status", "reason"]]

        else:

            df_bad = pd.DataFrame(columns = ["status", "reason"])

        if SAVE_TO_EXCEL:
          
            try:
          
                if _os.path.exists(self.EXCEL_PATH):
          
                    with pd.ExcelWriter(self.EXCEL_PATH, mode = "a", engine = "openpyxl", if_sheet_exists = "replace") as writer:
          
                        df_ok.to_excel(writer, sheet_name = "LSTM_DirectH")
          
                        df_bad.to_excel(writer, sheet_name = "LSTM_DirectH_skips")
          
                else:
          
                    with pd.ExcelWriter(self.EXCEL_PATH, engine = "openpyxl") as writer:
          
                        df_ok.to_excel(writer, sheet_name="LSTM")
          
                        df_bad.to_excel(writer, sheet_name="LSTM_skips")
          
                self.logger.info("Saved results to %s", self.EXCEL_PATH)
          
            except Exception as ex:
          
                self.logger.error("Failed to write Excel: %s", ex)

  

        self.logger.info("Forecasting complete. ok=%d, skipped/error=%d", len(ok_rows), len(bad_rows))
      
        return df_ok, df_bad
    
    
if __name__ == "__main__":
  
    try:
  
        mp.set_start_method("spawn", force = True)
  
    except RuntimeError:
  
        pass

    profiler = cProfile.Profile()

    profiler.enable()

    try:

        forecaster = DirectHForecaster(
            tickers = config.tickers
        )

        forecaster.run()

    finally:

        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats("cumtime")

        stats.print_stats(20)
