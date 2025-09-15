from __future__ import annotations

"""
GRU-based direct-H forecasting of cumulative weekly log-returns with
macro, factor, and financial scenario engines, post-hoc calibration, and
parallel execution.

Purpose
-------
This module trains and applies a gated recurrent unit (GRU) model to
forecast the H-week cumulative log return of an equity, while conditioning
on forward paths for exogenous drivers. It produces distributional price
summaries (intervals and moments) for each ticker and can persist results
to Excel.

High-level pipeline
-------------------
1) Data assembly and alignment:
  
   - Weekly prices are converted to log returns.
 
   - Macro series are differenced or log-differenced and resampled to a
     weekly cadence per country.
 
   - Factor returns are aligned to weekly frequency.
 
   - Optional financial levels (revenue and basic EPS) are aligned to a
     weekly last-observation-carried-forward series.
 
   - Rolling higher moments (skewness and kurtosis) of returns are
     computed on a 104-week window and shifted forward by the forecast
     horizon to avoid leakage.

2) Scenario engines (forward regressors for the forecast horizon H):
 
   - Macro: a per-country VAR(1) with intercept, fitted in stationary
     space X_t = c + A X_{t−1} + epsilon_t, epsilon_t normal with
     covariance Sigma. If estimable, multi-step paths are simulated in
     closed form; otherwise, Gaussian innovations with Ledoit–Wolf
     covariance are used. Deltas are expanded from quarterly blocks to
     weeks.
 
   - Factors: forward paths are drawn using the Politis–Romano stationary
     bootstrap with restart probability p, preserving marginal
     distributions and short-range dependence.
 
   - Financials: next-period revenue and EPS point estimates are converted
     to four sequential quarterly log-increments whose weekly deltas are
     debiased noisy versions that preserve the quarterly totals.
 
   - Moments: the last observed H deltas are persisted over the forecast
     horizon to represent slow movement in higher-order structure.

3) Direct-H learning target and baseline:
 
   - The target for a window ending at t is y_t = sum from k=1 to H of
     r_{t+k}, where r_t is the weekly log return.
 
   - A mean-reverting AR(1) baseline is fitted on training returns,
     
        r_t = m + phi (r_{t−1} − m) + epsilon_t. 
        
    The direct-H expectation is 
    
        E[sum from k=1 to H of r_{t+k} | r_t] = H m + (r_t − m) phi (1 − phi^H) divided by (1 − phi). 
        
    The neural model learns residuals y_t minus this baseline sum, i
    mproving stationarity and calibration.

4) Model architecture (direct-H, single-layer GRU):
   - Inputs are length L = H_hist + H sequences with channels:
       channel 0: scaled historical returns for the first H_hist steps and
                  zeros for the final H steps;
       channels 1..p: robust-scaled regressor deltas, where the future H
                      steps are filled by scenario draws.
  
   - A GRU with h units produces the final hidden state. Two heads follow:
       Quantile head: a 3-parameter monotone transform yields the
         residual quantiles (q_0.1, q_0.5, q_0.9). The pinball loss
         averages quantile losses and enforces non-crossing by
         construction via positive gaps.
       Distribution head: a Student-t residual model with parameters
         mu, sigma, nu obtained from bounded transforms of dense outputs:
         mu = mu_max times tanh(mu_raw);
         sigma = softplus(sigma_raw) plus sigma_floor, capped at sigma_max;
         nu = nu_floor plus softplus(nu_raw).
         Training minimises negative log-likelihood plus mild penalties on
         log sigma, on one over (nu minus two), and on the unbounded
         location parameter to discourage extreme values.

   - GRU internals: at time t, with input x_t and previous state h_{t−1},
     the update gate 
     
        z_t = sigmoid(W_z x_t + U_z h_{t−1} + b_z), 
        
     the reset gate r_t = sigmoid(W_r x_t + U_r h_{t−1} + b_r), the candidate
     state 
     
        h_t_tilde = tanh(W_h x_t + U_h (r_t elementwise h_{t−1}) + b_h),
        
     and the new state
     
        h_t = (1 − z_t) elementwise h_{t−1} + z_t elementwise h_t_tilde. 
        
    This gating controls information flow and mitigates vanishing gradients with fewer parameters 
    than LSTMs.

5) Calibration and uncertainty:
   - Scale calibration: a single multiplicative factor s on sigma is
     chosen on the validation set to minimise the mean squared difference
     between empirical and nominal coverages for several central
     intervals. For a tail mass alpha, the interval is
     
     [mu + s sigma t_{nu}^{−1}(alpha / 2),
      mu + s sigma t_{nu}^{−1}(1 − alpha / 2)].
   
   - PIT isotonic calibration: probability integral transforms
     
     u_i = F_t((y_i − mu_i)/sigma_i; nu_i) 
     
     are mapped to uniformity by an increasing isotonic function g, learned 
     on validation PITs. The inverse corrected levels alpha' satisfy g(alpha') 
     approximately equals alpha. Residual draws may be recalibrated by mapping u 
     to v = g(u) and inverting with the Student-t quantile.

6) Inference with epistemic and aleatoric uncertainty:
   - Epistemic uncertainty is approximated by Monte Carlo dropout with T
     stochastic forward passes; aleatoric uncertainty arises from the
     Student-t head. For each scenario path and pass, a residual sample
     epsilon_hat = mu + sigma times z is drawn with z from the Student-t
     with nu degrees of freedom, optionally recalibrated via PIT. The
     AR(1) baseline sum at now is added back. Sums of log returns are
     converted to multiplicative price factors via the exponential minus
     one mapping, and empirical quantiles and moments are computed.

Feature engineering and scaling
-------------------------------
Regressor deltas are column-wise:
- simple differences for interest, unemployment, and higher-moment
  columns;
- log-differences for price-level-like series such as CPI, GDP, revenue,
  and EPS;
- passthrough for factor returns which are already stationary.
Deltas are robust-scaled using training medians and interquartile ranges
with per-feature clipping to the 1st and 99th percentiles to stabilise
gradients.

Shapes and notation
-------------------
- Sequence length L equals H_hist plus H.
- Input tensor has shape (N, L, 1 + p), where p is the number of
  regressors.
- Target has shape (N, 1) and equals the direct sum of future H log
  returns.
- All tensors are float32 for performance and numerical consistency.

Parallelism, determinism, and caching
-------------------------------------
- Multi-process execution uses the 'spawn' start method for TensorFlow
  safety. An initialiser injects a read-only state pack into each worker.
- A process-local singleton forecaster avoids repeated model creation and
  JIT compilation.
- NumPy, Python, and TensorFlow seeds are set; intra- and inter-op thread
  pools are limited to one where supported to improve reproducibility.
- Covariances are regularised to be symmetric positive definite, using
  adaptive jitter and spectral shifting when necessary.

Outputs
-------
On success, per-ticker dictionaries are produced for raw and
PIT-calibrated forecasts containing:
- "Min Price", "Avg Price", "Max Price" from central intervals at
  corrected levels and the median;
- "Returns" and "SE" as the mean and standard deviation of cumulative log
  returns.
Skipped and error cases include a status and reason. If enabled, results
are written to Excel as separate sheets for raw forecasts, calibrated
forecasts, and skip diagnostics.

Assumptions and limitations
---------------------------
- Direct-H modelling assumes that providing future regressor paths is
  informative about future returns; the quality of these paths directly
  affects forecast accuracy.
- The Student-t residual assumption handles heavy tails but remains
  parametric; extreme regimes may still be under-represented.
- The AR(1) baseline captures only first-order persistence; higher-order
  auto-correlations, if present, are left to the GRU to learn.
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
from sklearn.isotonic import IsotonicRegression  
from scipy.stats import t as student_t           

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.api import VAR

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from sklearn.covariance import LedoitWolf

from dataclasses import dataclass

import config
from financial_forecast_data4 import FinancialForecastData

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, GRU, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


STATE = None

FORECASTER_SINGLETON = None  

SAVE_TO_EXCEL = True 


def _init_worker(
    state_pack
):
    """
    Initialise a worker process with an immutable snapshot of global state.

    This function is executed in each child process created by
    `ProcessPoolExecutor`. It binds the serialisable `state_pack` (produced
    by `GRU_Forecaster.build_state`) to the module-level variable `STATE`
    so worker functions can access read-only data without reloading large
    artefacts per task.

    Parameters
    ----------
    state_pack : Dict[str, Any]
        The pre-built, serialisable state dictionary containing data and
        caches required by forecasting workers. See `GRU_Forecaster.build_state`
        for a complete description of its contents.

    Side effects
    ------------
    Sets the module-level variable `STATE` for the lifetime of the worker
    process.

    Notes
    -----
    - The design avoids repeated I/O and CPU-heavy preprocessing in workers.
    - The state is treated as read-only; mutation would create hard-to-debug
    cross-process inconsistencies.
    """
   
    global STATE
   
    STATE = state_pack


def _worker_forecast_one(
    ticker: str
):
    """
    Worker entrypoint that forecasts for a single ticker using a process-local
    forecaster singleton.

    This function lazily constructs a `GRU_Forecaster` per worker process
    (the 'singleton') to amortise TensorFlow graph creation and model
    instantiation, then delegates to `GRU_Forecaster.forecast_one`.

    Parameters
    ----------
    ticker : str
        The equity identifier to forecast.

    Returns
    -------
    Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]
        On success, returns a pair `(raw_dict, calibrated_dict)` where each
        dictionary contains interval prices and return moments. On skip/error,
        returns a single dictionary with keys `status` and `reason`.

    Notes
    -----
    - The approach reduces overhead by reusing a compiled Keras model and
    JIT-compiled `tf.function`s within the process.
    - The module-level `STATE` must have been set by `_init_worker`.
    """
  
    global FORECASTER_SINGLETON
  
    if FORECASTER_SINGLETON is None:
  
        FORECASTER_SINGLETON = GRU_Forecaster()
  
    return FORECASTER_SINGLETON.forecast_one(ticker)


REV_KEYS = ["low_rev_y", "avg_rev_y", "high_rev_y", "low_rev", "avg_rev", "high_rev"]

EPS_KEYS = ["low_eps_y", "avg_eps_y", "high_eps_y", "low_eps", "avg_eps", "high_eps"]

SCENARIOS = [(r, e) for r in REV_KEYS for e in EPS_KEYS]


@dataclass(frozen = True)
class ModelHP:
   
    hist_window: int = 52
   
    horizon: int = 52
   
    GRU1: int = 128
      
    l2_lambda: float = 1e-4
   
    dropout: float = 0.15


@dataclass(frozen = True)
class TrainHP:
   
    batch: int = 256
   
    epochs: int = 100
   
    patience: int = 10
   
    lr: float = 5e-4
   
    small_floor: float = 1e-6


@dataclass(frozen = True)
class DistHP:
   
    sigma_floor: float = 1e-3
   
    sigma_max: float = 0.3
   
    nu_floor: float = 8.0
   
    lambda_sigma: float = 5e-4
   
    lambda_invnu: float = 5e-4

    lambda_mu_raw: float = 1e-4


@dataclass(frozen = True)
class ScenarioHP:
   
    n_sims: int = 100
   
    mc_dropout_samples: int = 20
   
    alpha_conf: float = 0.10
   
    rev_noise_sd: float = 0.005
   
    eps_noise_sd: float = 0.005
   
    bootstrap_p: float = 1/6


@dataclass(frozen = True)
class HP:
    """
    Container(s) for hyper-parameters grouped by concern:

    - `ModelHP`: architecture and sequence geometry.
        * hist_window : number of historical weekly steps H_h used as input.
        * horizon     : forecasting horizon H_f (weekly steps) for a direct-H target.
        * GRU1        : number of GRU units in the single recurrent layer.
        * l2_lambda   : ℓ2 regularisation coefficient for GRU kernel and recurrent
                        weights.
        * dropout     : (input) dropout rate for GRU.

    - `TrainHP`: optimisation and training stability.
        * batch, epochs, patience, lr, small_floor.

    - `DistHP`: Student-t head constraints and regularisers.
        * sigma_floor ≤ σ ≤ sigma_max ensures positive, bounded scales.
        * nu_floor sets ν ≥ ν_min > 2 to guarantee finite variance.
        * lambda_sigma, lambda_invnu, lambda_mu_raw regularise log(σ), 1/(ν−2),
        and raw location respectively.

    - `ScenarioHP`: scenario simulation controls.
        * n_sims: number of scenario draws per path block.
        * mc_dropout_samples: MC-dropout passes for epistemic uncertainty.
        * alpha_conf: tail mass for central intervals (e.g., 0.10 → 90% CI).
        * rev_noise_sd, eps_noise_sd: small quarterly noise perturbations.
        * bootstrap_p: stationary bootstrap restart probability.

    - `HP`: top-level aggregate of the groups above for concise passing.
    """
   
    model: ModelHP = ModelHP()
   
    train: TrainHP = TrainHP()
   
    dist: DistHP = DistHP()
   
    scen: ScenarioHP = ScenarioHP()


class GRU_Forecaster:
    """
    GRU-based direct-H forecaster of cumulative weekly log-returns with
    scenario-driven regressors and calibrated predictive distributions.

    Overview
    --------
    Let r_t denote weekly log return at week t. The forecasting target is the
    direct H-step sum:

        y_t = sum_{k=1..H} r_{t+k}.

    A single-layer GRU consumes a sequence of length L = H_hist + H (history
    plus 'known' future regressors) with channels:

        - channel 0: scaled historical returns for the first H_hist steps,
        and zeros for the last H steps (the model is asked to infer future
        returns conditioned on future regressors),

        - channels 1..p: transformed regressor deltas, with the final H steps
        populated by simulated future deltas from macro/factor/financial
        scenario engines.

    Two output heads are trained jointly:

    1) Quantile head (monotone by construction) yields (q_0.1, q_0.5, q_0.9)
    of residuals ε_t = y_t − μ_AR1(t) via

        q_0.1 = a,

        q_0.5 = a + softplus(b) + ε,

        q_0.9 = a + softplus(b) + softplus(c) + ε,

    with ε = 0 here (the offsets enforce non-crossing). The pinball loss
    aggregates quantile losses.

    2) Distribution head parameterises a Student-t residual model:

        ε_t | θ_t ∼ t_ν(μ_t, σ_t),
    
    where 
    
        μ_t = μ_max * tanh(μ_raw), 
        
        σ_t = softplus(σ_raw) + σ_floor,
        
        ν_t = ν_floor + softplus(ν_raw). 
        
    Training minimises the negative log-likelihood with mild regularisation on
    log σ, 1/(ν − 2), and μ_raw.

    A simple AR(1) with intercept provides a baseline mean path for the sum
    of returns, used to model residuals rather than raw sums, improving
    stationarity and fit:

        r_t = m + φ (r_{t−1} − m) + ε_t,

    and the direct-H expectation is

        E[ sum_{k=1..H} r_{t+k} | r_t ] = H m + (r_t − m) φ (1 − φ^H)/(1 − φ).

    Scenario engines provide forward deltas for regressors:

    - Macro deltas per country from a VAR(1) with intercept fitted in
    stationary space; closed-form multi-step simulation is used.

    - Factor deltas via the Politis–Romano stationary bootstrap.

    - Financial deltas (revenue, EPS) constructed from next-period points and
    converted into weekly log-delta paths with small mean-zero quarter-wise
    noise that preserves quarterly totals.

    Calibration
    -----------
    Two complementary calibration devices are applied on the validation set:
  
    a) Scale calibration: choose s to minimise the mean squared discrepancy
    between nominal and empirical coverages for several central intervals.
   
    b) PIT isotonic calibration: compute u = F_t(ε_t) under the fitted t-model
    and learn an isotonic map g so that g(u) is uniform. The inverse map
    provides corrected quantile levels for evaluation and reporting.

    Uncertainty
    -----------
    Epistemic uncertainty is approximated via Monte Carlo dropout (T passes).
    Aleatoric uncertainty arises from the Student-t head. During simulation,
    the model samples residuals:
    
        ε̂ = μ_t + σ_t * z,   with z ∼ t_ν,
    
    optionally recalibrated via the PIT map. The baseline AR(1) sum is added
    back to obtain total return sums; exponentiation converts sums of log
    returns to multiplicative price factors.

    Rationale
    ---------
    - Direct-H targets minimise error accumulation common to recursive
    strategies.
    - GRUs capture long-horizon dependencies with fewer parameters than LSTMs
    and good training stability.
    - Student-t likelihood is robust to heavy tails and outliers in returns.
    - VAR(1) with intercept in stationary macro space captures persistence
    and cross-series dynamics while remaining estimable in short samples.
    - Stationary bootstrap preserves dependence structures in factor returns.

    Shapes and notation
    -------------------
    - Sequence length L = H_hist + H.
    - Input tensor: (N, L, 1 + p) where p is the number of regressors.
    - Targets: (N, 1) with y_t = sum future H log returns.
    - All deltas are robust-scaled using training-set statistics only.

    Threading/determinism
    ---------------------
    Operations are configured for single-threaded, deterministic execution
    where supported to enhance reproducibility.

    File outputs
    ------------
    If enabled, results are saved to Excel with separate sheets for raw and
    calibrated forecasts and a sheet containing skip reasons.
    """


    SEED = 42
  
    random.seed(SEED)
  
    np.random.seed(SEED)

    SAVE_ARTIFACTS = False
  
    ARTIFACT_DIR = config.BASE_DIR / "gru_artifacts"
  
    EXCEL_PATH = config.MODEL_FILE

    HIST_WINDOW = 52
  
    HORIZON = 52
  
    SEQUENCE_LENGTH = HIST_WINDOW + HORIZON
                   
    MU_MAX = 0.80            
                    
    MACRO_REGRESSORS = ["Interest", "Cpi", "Gdp", "Unemp"]
  
    FACTORS = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]
  
    FIN_REGRESSORS = ["Revenue", "EPS (Basic)"]
  
    MOMENT_COLS = ["skew_104w_lag52", "kurt_104w_lag52"]
  
    NON_FIN_REGRESSORS = MACRO_REGRESSORS + FACTORS
  
    ALL_REGRESSORS = NON_FIN_REGRESSORS + FIN_REGRESSORS + MOMENT_COLS

    _w = HORIZON // 4
  
    _rem = HORIZON - 4 * _w
  
    REPEATS_QUARTER = np.array([_w, _w, _w, _w + _rem], dtype = int)

    _MODEL_CACHE: Dict[int, Tuple[Any, List[np.ndarray]]] = {}
    
    _EYE_CACHE = {4: (np.eye(4, dtype = np.float32) * 1e-4)} 
    
    _FN_CACHE: Dict[int, Dict[str, Any]] = {}


    def __init__(
        self, 
        tickers: Optional[List[str]] = None,
        hp: Optional["GRU_Forecaster.HP"] = None
    ):
        """
        Construct a forecaster instance, bind hyper-parameters, and configure
        deterministic TensorFlow execution.

        Parameters
        ----------
        tickers : Optional[List[str]]
            User-provided list of tickers to forecast. If None, a default set is
            used when building state.
        hp : Optional[HP]
            Aggregate hyper-parameters. If None, defaults are used.

        Initialisation details
        ----------------------
        - Seeds NumPy, Python, and TensorFlow RNGs for reproducibility.
        
        - Derives convenience attributes (e.g., `SEQUENCE_LENGTH = hist + horizon`).
        
        - Configures TensorFlow intra/inter-op thread pools to 1 for deterministic
        behaviour where supported.
        
        - Prepares caches:
            * `_MODEL_CACHE[n_reg]` → (compiled Keras model, initial weights),
            * `_FN_CACHE[n_reg]`    → bound `tf.function`s and meta (reg indices),
            * `_DEL_MASKS[tuple(regs)]` → indices for difference/log-difference rules.

        Notes
        -----
        No heavy I/O occurs here; data acquisition and alignment are performed
        in `build_state`.
        """
       
        self.logger = self._configure_logger()
       
        self.tickers_arg = tickers
       
        self.hp = hp or HP()  

        self.HIST_WINDOW = self.hp.model.hist_window
       
        self.HORIZON = self.hp.model.horizon
       
        self.SEQUENCE_LENGTH = self.HIST_WINDOW + self.HORIZON
       
        self.L2_LAMBDA = self.hp.model.l2_lambda
       
        self._GRU1 = self.hp.model.GRU1
              
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
        
        self.LAMBDA_MU_RAW = self.hp.dist.lambda_mu_raw
        
        self._DEL_MASKS: Dict[Tuple[str, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        
        try:

            tf.config.experimental.enable_op_determinism()

        except Exception:

            pass

        tf.random.set_seed(self.SEED)

        tf.config.threading.set_intra_op_parallelism_threads(1)

        tf.config.threading.set_inter_op_parallelism_threads(1)

        
    def _configure_logger(
        self
    ) -> logging.Logger:
        """
        Create and configure an `INFO`-level logger for this class.

        Returns
        -------
        logging.Logger
            A logger named `"gru_directH_class"` configured with a concise
            timestamped formatter and a single `StreamHandler`.

        Notes
        -----
        Idempotent across multiple instantiations; additional handlers are not
        added if one already exists.
        """
        
        logger = logging.getLogger("gru_directH_class")
        
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
        
            h = logging.StreamHandler()
        
            h.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s"))
        
            logger.addHandler(h)
        
        return logger

        
    @staticmethod
    def _shape_ok(
        X: np.ndarray, 
        seq_length: int, 
        n_reg: int
    ) -> bool:
        """
        Validate that an input tensor `X` matches the expected 3-D shape.

        Parameters
        ----------
        X : np.ndarray
            Candidate tensor.
        seq_length : int
            Expected sequence length L.
        n_reg : int
            Expected number of regressors p.

        Returns
        -------
        bool
            True if X.shape == (N, L, 1 + p) for some N; False otherwise.

        Rationale
        ---------
        Explicit shape checks guard against subtle broadcasting errors early in
        the pipeline.
        """
    
        N = len(X)
    
        return X.shape == (N, seq_length, 1 + n_reg)


    def choose_splits(
        self,
        N: int, 
        min_fold: Optional[int] = None
    ) -> int:
        """
        Select the number of time-series cross-validation folds based on sample size.

        Parameters
        ----------
        N : int
            Number of rolling windows available for training/validation.
        min_fold : Optional[int]
            Minimum number of samples per fold; defaults to 2 × batch size.

        Returns
        -------
        int
            0, 1, or 2 folds. Returns 0 if `N < min_fold`, 2 if `N ≥ 2*min_fold`,
            else 1.

        Rationale
        ---------
        Prevents over-fragmentation of limited data and keeps validation segments
        large enough to estimate calibration reliably.
        """
        
        if min_fold is None:
            
            min_fold = 2 * self.BATCH
    
        if N < min_fold :
    
            return 0
    
        if N >= 2 * min_fold:
    
            return 2
    
        return 1


    def _ensure_meta_cache(
        self, 
        n_reg: int, 
        regs: list[str]
    ) -> Dict[str, Any]:
        """
        Ensure that index metadata for a given regressor list is materialised and cached.

        Attaches to `_FN_CACHE[n_reg]`:
        - `reg_names` : tuple of names,
        - `reg_pos`   : mapping name → column index,
        - `delta_idx_masks` : four index arrays classifying columns into
        simple difference, log-difference, passthrough, and 'other' groups.
        Masks are produced (lazily) by `build_transform_matrix`.

        Parameters
        ----------
        n_reg : int
            Number of regressor columns.
        regs : list[str]
            Ordered regressor names.

        Returns
        -------
        Dict[str, Any]
            The cache entry containing indices and mappings.

        Notes
        -----
        Caching avoids recomputation of masks in tight loops and guarantees
        consistency across calls.
        """
        
        entry = self._FN_CACHE.get(n_reg)
        
        if entry is None:
        
            entry = {}
        
            self._FN_CACHE[n_reg] = entry

        if entry.get("reg_names") != tuple(regs):

            entry["reg_names"] = tuple(regs)

            entry["reg_pos"] = {name: i for i, name in enumerate(regs)}

            _ = self._DEL_MASKS.get(tuple(regs))

            if _ is None:

                fake = np.zeros((2, len(regs)), dtype = np.float32)

                self.build_transform_matrix(
                    reg_mat = fake, 
                    regs = regs
                )

            entry["delta_idx_masks"] = self._DEL_MASKS[tuple(regs)]

        return entry


    def build_transform_matrix(
        self,
        reg_mat: np.ndarray,
        regs: list[str]
    ) -> np.ndarray:
        """
        Transform raw regressor levels into weekly 'delta' features with per-column rules.

        Let X_t be the level at week t. The returned matrix Δ has shape (T−1, p)
        and is defined column-wise as follows:
        
        - For "Interest", "Unemp", and moment columns:
        
            Δ_t = X_t − X_{t−1}      (simple difference).
        
        - For "Cpi", "Gdp", "Revenue", "EPS (Basic)":
        
            Δ_t = log(max(X_t, ε)) − log(max(X_{t−1}, ε))   (log-difference),
        
        where ε = SMALL_FLOOR ensures numerical stability.
        
        - For factor columns (MTUM, QUAL, SIZE, USMV, VLUE):
        
            Δ_t = X_t               (already stationary).
        
        - For any other column:
        
            Δ_t = X_t − X_{t−1}.

        Parameters
        ----------
        reg_mat : np.ndarray, shape (T, p)
            Level values aligned to weekly frequency.
        regs : list[str], length p
            Column names corresponding to `reg_mat`.

        Returns
        -------
        np.ndarray, shape (T−1, p)
            Per-week deltas.

        Raises
        ------
        ValueError
            If dimensionality does not match expectations.

        Rationale
        ---------
        Differences and log-differences render many macro/financial series
        approximately stationary, which improves learning stability and the
        appropriateness of distributional assumptions downstream.
        """

        reg_mat = np.asarray(reg_mat, dtype = np.float32, order = "C")
        
        if reg_mat.ndim != 2:
        
            raise ValueError(f"reg_mat must be 2D, got {reg_mat.ndim}D with shape {reg_mat.shape}")
        
        T, n_reg = reg_mat.shape
        
        if n_reg != len(regs):
        
            raise ValueError(f"reg_mat has {n_reg} columns but regs has {len(regs)}")
        
        if T < 2:
        
            return np.zeros((0, n_reg), dtype = np.float32)

        regs_key = tuple(regs)

        masks = self._DEL_MASKS.get(regs_key)

        if masks is None:
            
            regs_arr = np.asarray(regs, dtype = object)

            is_diff = np.isin(regs_arr, ("Interest", "Unemp")) | np.isin(regs_arr, tuple(self.MOMENT_COLS))

            is_dlog = np.isin(regs_arr, ("Cpi", "Gdp", "Revenue", "EPS (Basic)"))

            is_pass = np.isin(regs_arr, tuple(self.FACTORS))

            other = ~(is_diff | is_dlog | is_pass)

            idx_diff = np.flatnonzero(is_diff)
           
            idx_dlog = np.flatnonzero(is_dlog)
           
            idx_pass = np.flatnonzero(is_pass)
           
            idx_other = np.flatnonzero(other)

            masks = (idx_diff, idx_dlog, idx_pass, idx_other)
           
            self._DEL_MASKS[regs_key] = masks

        idx_diff, idx_dlog, idx_pass, idx_other = masks
       
        DEL = np.empty((T - 1, n_reg), dtype = np.float32)

        if idx_diff.size:
           
            cols = reg_mat[:, idx_diff]
           
            DEL[:, idx_diff] = cols[1:] - cols[:-1]

        if idx_dlog.size:
          
            cols = np.maximum(reg_mat[:, idx_dlog], self.SMALL_FLOOR)
          
            DEL[:, idx_dlog] = np.log(cols[1:]) - np.log(cols[:-1])

        if idx_pass.size:

            DEL[:, idx_pass] = reg_mat[1:, idx_pass]

        if idx_other.size:
           
            cols = reg_mat[:, idx_other]
           
            DEL[:, idx_other] = cols[1:] - cols[:-1]

        return DEL


    @staticmethod
    def fit_scale_deltas(
        DEL_tr: np.ndarray
    ):
        """
        Fit a robust scaler on training deltas and compute clipping quantiles.

        Parameters
        ----------
        DEL_tr : np.ndarray, shape (N_tr, p)
            Training subset of deltas.

        Returns
        -------
        sc : RobustScaler
            Fitted scaler using median and IQR. Its `scale_` is floored at 1e−6.
        q_low, q_high : np.ndarray, shape (p,)
            1st and 99th percentile per feature for robust clipping prior to scaling.

        Rationale
        ---------
        Robust scaling mitigates the influence of outliers and heavy tails;
        pre-clipping further stabilises gradients and reduces the impact of
        rare extreme values in recurrent inputs.
        """
    
        sc = RobustScaler().fit(DEL_tr)
    
        sc.scale_[sc.scale_ < 1e-6] = 1e-6
    
        q_low = np.quantile(DEL_tr, 0.01, axis = 0).astype(np.float32)
    
        q_high = np.quantile(DEL_tr, 0.99, axis = 0).astype(np.float32)
    
        return sc, q_low, q_high
    
    
    @staticmethod
    def fit_ar1_baseline(
        log_ret: np.ndarray
    ) -> tuple[float, float]:
        """
        Estimate an AR(1) with intercept for weekly log returns and derive the
        long-run mean.

        Model
        -----
        r_t = m + φ (r_{t−1} − m) + ε_t.
        
        This is equivalent to 
        
            r_t = α + φ r_{t−1} + ε_t with α = m (1 − φ).
        
        Ordinary least squares on (r_{t−1}, r_t) yields β = (α̂, φ̂).
        
        The implied mean is m̂ = α̂ / (1 − φ̂), provided |1 − φ̂| > 0.

        Parameters
        ----------
        log_ret : np.ndarray, shape (T,)
            Weekly log returns.

        Returns
        -------
        (m_hat, phi_hat) : Tuple[float, float]
            Estimated long-run mean and AR(1) coefficient, or (0, 0) if too short.

        Rationale
        ---------
        Using an explicit baseline for the H-step sum (see `ar1_sum_forecast`)
        allows the neural network to model residual structure rather than raw
        trend and persistence, improving stability and calibration.
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
        Closed-form expectation of the sum of future H AR(1) returns.

        For r_t = m + φ (r_{t−1} − m) + ε_t, the expected H-step sum is:
        E[∑_{k=1..H} r_{t+k} | r_t]
        = H m + (r_t − m) φ (1 − φ^H) / (1 − φ),
        with the limit H m when φ ≈ 1.

        Parameters
        ----------
        r_t : float
            Last observed return r_t.
        m : float
            Long-run mean.
        phi : float
            AR(1) coefficient.
        H : int
            Horizon.

        Returns
        -------
        float
            The expected cumulative return under the AR(1) model.
        """

        if abs(1.0 - phi) < 1e-8:
    
            return H * m
    
        geom = (1.0 - phi ** H) / (1.0 - phi)
    
        return H * m + (r_t - m) * phi * geom


    @staticmethod
    def transform_deltas(
        DEL: np.ndarray, 
        sc: RobustScaler, 
        q_low: np.ndarray,
        q_high: np.ndarray
    ):
        """
        Clip deltas to robust quantile bounds and scale using a fitted `RobustScaler`.

        Parameters
        ----------
        DEL : np.ndarray, shape (..., p)
            Deltas to be transformed; leading dimensions are preserved.
        sc : RobustScaler
            Fitted scaler with `center_` and `scale_`.
        q_low, q_high : np.ndarray, shape (p,)
            Clipping bounds computed on the training set.

        Returns
        -------
        np.ndarray
            Transformed deltas as float32.

        Notes
        -----
        Clipping is applied prior to centring/scaling to suppress outliers.
        """
    
        DEL_clip = np.clip(DEL, q_low, q_high)
    
        DEL_scaled = (DEL_clip - sc.center_) / sc.scale_
    
        return DEL_scaled.astype(np.float32)
    
    
    @staticmethod
    def ensure_spd_for_cholesky(
        Sigma: np.ndarray, 
        min_eig: float = 1e-8,
        jitter_mult: float = 10.0, max_tries: int = 6
    ) -> np.ndarray:
        """
        Project a symmetric matrix to be positive definite for Cholesky
        factorisation, with adaptive jitter.

        Parameters
        ----------
        Sigma : np.ndarray, shape (d, d)
            Symmetric covariance estimate.
        min_eig : float
            Minimum allowable eigenvalue.
        jitter_mult : float
            Multiplicative factor to escalate jitter on repeated failures.
        max_tries : int
            Maximum attempts before a final spectral shift.

        Returns
        -------
        np.ndarray, shape (d, d), dtype float32
            A numerically SPD matrix.

        Algorithm
        ---------
        1) Symmetrise S ← ½ (S + Sᵀ).
        
        2) Try Cholesky with additive jitter λ I, doubling λ adaptively.
        
        3) If still failing, shift by (min_eig − λ_min + ε) I based on eigvals.

        Rationale
        ---------
        VAR simulation and multivariate normal sampling require SPD covariance
        matrices; this utility produces stable factors in the presence of
        estimation noise.
        """
    
        S = np.asarray(Sigma, dtype = np.float64)
    
        S = np.where(np.isfinite(S), S, 0.0)
    
        S = 0.5 * (S + S.T)
    
        I = np.eye(S.shape[0], dtype = np.float64)

        jitter = 0.0
       
        for _ in range(max_tries):
       
            try:
       
                np.linalg.cholesky(S + jitter * I)
       
                return (S + (jitter * I) if jitter else S).astype(np.float32, copy = False)
       
            except np.linalg.LinAlgError:
              
                jitter = jitter_mult * (jitter + min_eig)

        lam_min = float(np.min(np.linalg.eigvalsh(S)))
        
        shift = (min_eig - lam_min) + 1e-12 if lam_min < min_eig else min_eig
        
        S = 0.5 * (S + S.T) + (jitter + shift) * I
        
        np.linalg.cholesky(S)
        
        return S.astype(np.float32, copy = False)

    
    @staticmethod
    def _lw_cov_or_eye(
        tail_vals: Optional[np.ndarray], 
        dim: int
    ) -> np.ndarray:
        """
        Return a Ledoit–Wolf shrinkage covariance if enough samples are present,
        else return a small multiple of the identity (cached).

        Parameters
        ----------
        tail_vals : Optional[np.ndarray], shape (N, d)
            Observations for covariance estimation.
        dim : int
            Desired dimensionality if falling back to identity.

        Returns
        -------
        np.ndarray, shape (d, d)
            SPD covariance estimate or ε I with ε = 1e−4.

        Rationale
        ---------
        The shrinkage estimator is well-behaved in small samples; if samples are
        insufficient, a tiny identity regularises downstream simulation without
        injecting spurious structure.
        """

        if tail_vals is not None and tail_vals.shape[0] > 10:
    
            cov = LedoitWolf().fit(tail_vals).covariance_.astype(np.float32, copy = False)
    
            return cov
     
        if dim in GRU_Forecaster._EYE_CACHE:
     
            return GRU_Forecaster._EYE_CACHE[dim]

        GRU_Forecaster._EYE_CACHE[dim] = np.eye(dim, dtype = np.float32) * 1e-4
     
        return GRU_Forecaster._EYE_CACHE[dim]
    
    
    @staticmethod
    def make_windows_directH(
        scaled_ret, 
        scaled_reg, 
        hist,
        hor, 
        seq_length
    ):
        """
        Construct model input tensors for the direct-H setup by stitching
        historical returns and regressor deltas.

        Design
        ------
        - For each rolling window i (i = 0..N−1), build a length-L tensor:
        * positions 0..hist−1: channel 0 contains scaled returns; channels 1..p
            contain past regressor deltas,
        * positions hist..hist+hor−1: channel 0 is zero (the future returns are
            to be predicted); channels 1..p contain *future* regressor deltas
            aligned to the horizon.

        Parameters
        ----------
        scaled_ret : np.ndarray, shape (T_r,)
            Robust-scaled returns with a prepended zero at index 0.
        scaled_reg : np.ndarray, shape (T_d, p)
            Robust-scaled regressor deltas.
        hist : int
            Number of historical weeks.
        hor : int
            Forecast horizon in weeks.
        seq_length : int
            Must equal hist + hor.

        Returns
        -------
        X : np.ndarray, shape (N, seq_length, 1 + p)
            Model inputs.
        _ : None
            Placeholder for API compatibility.

        Notes
        -----
        The alignment uses `sliding_window_view` to avoid copies. When insufficient
        length exists, an empty design matrix is returned.
        """
       
        T_ret = scaled_ret.shape[0]
       
        n_reg = scaled_reg.shape[1]
       
        n_windows = T_ret - seq_length + 1
       
        if n_windows <= 0:
       
            return (np.zeros((0, seq_length, 1 + n_reg), np.float32), np.zeros((0, 1), np.float32))

        i0 = np.arange(n_windows, dtype = np.int64)

        pr = sliding_window_view(scaled_ret, hist)[i0]                        

        pR = sliding_window_view(scaled_reg, (hist, n_reg))[:, 0][i0]             

        fR = sliding_window_view(scaled_reg, (hor, n_reg))[:, 0][(hist - 1) + i0]   

        X = np.empty((n_windows, seq_length, 1 + n_reg), dtype = np.float32)

        X[:, :hist, 0] = pr

        X[:, hist:, 0] = 0.0

        X[:, :hist, 1:] = pR

        X[:, hist:, 1:] = fR

        return X, None


    @staticmethod
    def make_target_directH(
        log_ret: np.ndarray,
        hist: int,
        hor: int,
        seq_len: int,
        N_expected: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the direct-H target: sums of future H log returns for each window.

        Let r_t be weekly log return. For each window aligned so that its last
        historical index ends at t, the target is:
            y_t = sum_{k=1..H} r_{t+k}.

        Parameters
        ----------
        log_ret : np.ndarray, shape (T,)
            Raw log returns aligned to the same dates as the windowing.
        hist : int
            Historical length per design window.
        hor : int
            Horizon H.
        seq_len : int
            Must be hist + hor (used for sanity checks).
        N_expected : Optional[int]
            If provided, truncate to match the number of constructed windows.

        Returns
        -------
        np.ndarray, shape (N, 1), dtype float32
            Direct-H target vector.

        Notes
        -----
        Targets are in log-return space and later paired with a baseline AR(1)
        expectation to form residuals for model training.
        """

        fr_full = sliding_window_view(
            x = log_ret[hist:], 
            window_shape = hor
        )  
        
        if fr_full.size == 0:
       
            return np.zeros((0, 1), np.float32)
        
        y = np.sum(fr_full, axis = 1, keepdims = True).astype(np.float32)  
        
        if N_expected is not None:
        
            N = min(int(N_expected), y.shape[0])
        
            y = y[:N]
        
        return y


    @staticmethod
    def _to_stationary_macro(
        dfm: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Transform macro series to approximate stationarity via first differences
        and log-differences.

        Transformations
        ---------------
        - Interest, Unemp: ΔX_t = X_t − X_{t−1}.
        - Cpi, Gdp: Δx_t = log(max(X_t, ε)) − log(max(X_{t−1}, ε)), ε = 1e−6.

        Parameters
        ----------
        dfm : pd.DataFrame
            Columns: Interest, Cpi, Gdp, Unemp.

        Returns
        -------
        dfm_stat : pd.DataFrame
            Stationary macro deltas (drops first NA row).
        macro_action_dict : Dict[str, str]
            Mapping from column to applied action ("diff" or "dlog").

        Rationale
        ---------
        VAR models are typically fitted to stationary processes; using deltas
        avoids spurious regression and supports meaningful impulse propagation.
        """
    
        out = pd.DataFrame(index = dfm.index)
    
        out["Interest"] = dfm["Interest"].diff()
       
        out["Cpi"] = np.log(np.maximum(dfm["Cpi"], 1e-6)).diff()
       
        out["Gdp"] = np.log(np.maximum(dfm["Gdp"], 1e-6)).diff()
       
        out["Unemp"] = dfm["Unemp"].diff()
        
        macro_action_dict = {
            "Interest": "diff",
            "Cpi": "dlog", 
            "Gdp": "dlog", 
            "Unemp": "diff"
        }
       
        return out.dropna(), macro_action_dict

   
    @staticmethod
    def _stable_A(
        A: np.ndarray
    ) -> np.ndarray:
        """
        Scale a square matrix A to ensure spectral radius strictly less than one.

        Parameters
        ----------
        A : np.ndarray, shape (d, d)
            Estimated VAR(1) transition matrix in stationary space.

        Returns
        -------
        np.ndarray, shape (d, d), dtype float32
            Possibly scaled matrix A / (ρ + ε) if the spectral radius ρ ≥ 1.

        Rationale
        ---------
        Ensures stability of forward simulations and convergence of series such
        as ∑ A^k used in the drift term of the closed-form VAR path.
        """
    
        vals = np.linalg.eigvals(A)
    
        rho = float(np.max(np.abs(vals)))
    
        if rho >= 1.0:
    
            A = A / (rho + 1e-3)
    
        return A.astype(np.float32)


    @staticmethod
    def _coverage_loss(
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        df: np.ndarray,
        scale: float,
        alphas = (0.10, 0.20, 0.50)
    ) -> float:
        """
        Mean squared discrepancy between empirical and nominal central-interval
        coverages under a Student-t model with global scale multiplier.

        Given residuals y (validation), model parameters (μ, σ, ν), and a scalar
        s ≥ 0 applied to σ, compute, for each α in `alphas`, the central (1−α)
        interval:
       
            [ μ + (sσ) t_{ν}^{−1}(α/2),  μ + (sσ) t_{ν}^{−1}(1−α/2) ],
       
        and compare its empirical coverage to 1 − α. Return the mean squared
        error across α levels.

        Parameters
        ----------
        y_true : np.ndarray, shape (N, 1)
        mu, sigma, df : np.ndarray, shape (N, 1)
        scale : float
            Global multiplier s for σ.
        alphas : Iterable[float]
            Set of tail masses.

        Returns
        -------
        float
            Average squared coverage error.

        Rationale
        ---------
        Choosing s by minimising this loss provides a simple post-hoc variance
        calibration using held-out data.
        """

        s = max(scale, 1e-6)
       
        sig = np.minimum(10.0, sigma.ravel() * s)  
       
        dfv = df.ravel()
       
        muv = mu.ravel()
       
        y = y_true.ravel()

        errs = []
       
        for a in alphas:
       
            lo = muv + sig * student_t.ppf(a / 2.0, dfv)
       
            hi = muv + sig * student_t.ppf(1.0 - a / 2.0, dfv)
       
            cov = np.mean((y >= lo) & (y <= hi))
       
            errs.append((cov - (1.0 - a)) ** 2)
       
        return float(np.mean(errs))

    
    def _calibrate_sigma_scale(
        self,
        y_cal_res: np.ndarray,
        mu_c: np.ndarray,
        sigma_c: np.ndarray,
        df_c: np.ndarray
    ) -> float:
        """
        Select a scalar σ multiplier by grid search to match validation coverage.

        Parameters
        ----------
        y_cal_res, mu_c, sigma_c, df_c : np.ndarray
            Validation residuals and corresponding predicted distribution
            parameters.

        Returns
        -------
        float
            Best scale factor s* ∈ [0.6, 1.5] on a 20-point grid.

        Notes
        -----
        A coarse grid suffices because the coverage loss is typically smooth in s.
        """

        grid = np.linspace(0.6, 1.5, 20) 
        
        losses = [self._coverage_loss(
            y_true = y_cal_res, 
            mu = mu_c, 
            sigma = sigma_c,
            df = df_c, 
            scale = g
        ) for g in grid]
        
        best = float(grid[int(np.argmin(losses))])
        
        return best


    def fit_var1_with_intercept(
        self, 
        dfm_stationary: pd.DataFrame
    ):
        """
        Fit a VAR(1) with intercept in stationary macro space, returning a
        stable transition matrix, intercept, residual covariance, and last state.

        Model
        -----
        X_t = c + A X_{t−1} + ε_t,
        
        with ε_t ∼ N(0, Σ). Statsmodels is used with trend='c'. When the sample
        is too short (T ≤ d + 4) or k_ar < 1, returns None.

        Returns
        -------
        Optional[dict]
            On success: {"A": A_stable, "c": c, "Sigma": Σ_SPD, "last_x": X_T, "neqs": d}.
            On failure: None.

        Rationale
        ---------
        VAR(1) balances parsimony and the need to capture cross-series dynamics
        for macro drivers, supporting tractable closed-form simulation.
        """
    
        if dfm_stationary.shape[0] <= dfm_stationary.shape[1] + 4:
    
            return None
    
        vr = VAR(dfm_stationary).fit(maxlags=1, trend="c")
    
        if vr.k_ar < 1:
    
            return None
    
        A = vr.coefs[0].astype(np.float32)
    
        c = vr.intercept_.astype(np.float32)
    
        Sigma = np.asarray(vr.sigma_u, dtype = np.float32) 
    
        Sigma = self.ensure_spd_for_cholesky(
            Sigma = Sigma
        )
        
        last_x = dfm_stationary.iloc[-1].to_numpy(np.float32)
        
        return {
            "A": self._stable_A(
                A = A
            ), 
            "c": c, 
            "Sigma": Sigma, 
            "last_x": last_x, 
            "neqs": vr.neqs
        }


    @staticmethod
    def _precompute_var_aux(
        A: np.ndarray, 
        Hq: int
    ):
        """
        Precompute powers of A and the lower-triangular block Toeplitz tensor
        needed for closed-form simulation of a VAR(1) over Hq steps.

        Definitions
        -----------
        Let P[i] = A^i for i = 0..Hq. Define T_full[h, w] = A^{h−w−1} for h > w,
        else zero; shapes are (Hq, Hq, d, d).

        Parameters
        ----------
        A : np.ndarray, shape (d, d)
        Hq : int
            Number of quarterly (or block) steps to simulate.

        Returns
        -------
        P : np.ndarray, shape (Hq+1, d, d)
        T_full : np.ndarray, shape (Hq, Hq, d, d)

        Usage
        -----
        Used by `simulate_var_paths_closed_form` to build both the deterministic
        and random components of the path via efficient einsum operations.
        """
    
        neqs = A.shape[0]
    
        P = np.empty((Hq + 1, neqs, neqs), dtype = np.float32)
    
        P[0] = np.eye(neqs, dtype = np.float32)
       
        for i in range(1, Hq + 1):
       
            P[i] = P[i - 1] @ A
       
        i_idx = np.arange(Hq)[:, None]
       
        j_idx = np.arange(Hq)[None, :]
       
        idx = i_idx - j_idx - 1
       
        mask = idx >= 0
       
        T_full = np.zeros((Hq, Hq, neqs, neqs), dtype = np.float32)
     
        T_full[mask] = P[idx[mask]]
     
        return P, T_full

    
    @classmethod
    def simulate_var_paths_closed_form(
        cls,
        A, 
        c, 
        Sigma,
        x0,
        Hq, 
        n_sims,
        rng
    ):
        """
        Closed-form simulation of VAR(1) paths with intercept and Gaussian noise.

        For X_t = c + A X_{t−1} + ε_t, let ε_t i.i.d. ∼ N(0, Σ). Over Hq steps,
        the path can be decomposed as:
            X_{t+h} = A^h x0 + ∑_{i=0}^{h−1} A^i c + ∑_{j=0}^{h−1} A^j ε_{t+h−j},
        h = 1..Hq. The last term is a linear transform of stacked ε’s; the
        precomputed block Toeplitz tensor T_full implements this mapping.

        Parameters
        ----------
        A : np.ndarray, shape (d, d)
        c : np.ndarray, shape (d,)
        Sigma : np.ndarray, shape (d, d), SPD
        x0 : np.ndarray, shape (d,)
        Hq : int
        n_sims : int
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (n_sims, Hq, d)
            Realisations of the stationary deltas.

        Notes
        -----
        Noise is sampled with method='cholesky' for stability; A is pre-stabilised
        and Σ ensured SPD before this call.
        """
    
        P, T_full = cls._precompute_var_aux(
            A = A, 
            Hq = Hq
        )
        
        neqs = A.shape[0]
        
        eps = rng.multivariate_normal(
            mean = np.zeros(neqs, np.float32), 
            cov = Sigma, 
            size = (n_sims, Hq), 
            method = "cholesky"
        ).astype(np.float32)  
       
        noise = np.einsum("swj,hwij->shi", eps, T_full).astype(np.float32)  
       
        det_x0 = np.einsum("hij,j->hi", P[1:], x0)                        
       
        sums = np.cumsum(P[:-1], axis = 0)                                   
       
        drift = np.einsum("hij,j->hi", sums, c)                            
       
        return det_x0[None, :, :] + drift[None, :, :] + noise              


    @staticmethod
    def stationary_bootstrap_indices(
        L: int, 
        n_sims: int, 
        H: int, 
        p: float, 
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate indices for the Politis–Romano stationary bootstrap.

        Algorithm
        ---------
        - For each simulation, define restart indicators R[:, t] ~ Bernoulli(p)
        with R[:, 0] = 1. Each run between restarts forms a block copied from
        the original series.
       
        - On a restart, choose a start index S0 uniformly in {0, .., L−1}.
       
        - For subsequent positions in the same block, advance by +1 mod L.

        Parameters
        ----------
        L : int
            Length of the source series.
        n_sims : int
        H : int
            Length of each resampled path.
        p : float
            Restart probability (expected block length 1/p).
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (n_sims, H), dtype int64
            Bootstrap indices.

        Rationale
        ---------
        Preserves short-range dependence and marginal distribution, unlike i.i.d.
        resampling, which is inappropriate for returns data.
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


    @staticmethod
    def fit_ret_scaler_from_logret(
        log_ret: np.ndarray
    ) -> RobustScaler:
        """
        Fit a robust scaler on log returns (excluding the prepended zero).

        Parameters
        ----------
        log_ret : np.ndarray, shape (T,)
            Weekly log returns.

        Returns
        -------
        RobustScaler
            Fitted scaler with floor on `scale_` at 1e−6.

        Notes
        -----
        Scaling the return channel stabilises GRU optimisation across assets.
        """
    
        sc = RobustScaler().fit(log_ret[1:].reshape(-1, 1))
    
        sc.scale_[sc.scale_ < 1e-6] = 1e-6
    
        return sc


    @staticmethod
    def scale_logret_with(
        log_ret: np.ndarray, 
        sc: RobustScaler
    ) -> np.ndarray:
        """
        Apply a fitted robust scaler to log returns and prepend a leading zero.

        Parameters
        ----------
        log_ret : np.ndarray, shape (T,)
        sc : RobustScaler

        Returns
        -------
        np.ndarray, shape (T,), dtype float32
            Scaled returns s with s[0] = 0 to align with window construction.
        """
    
        return np.concatenate([[0.0], ((log_ret[1:] - sc.center_) / sc.scale_).ravel()]).astype(np.float32)
    
    
    def build_directH_model(
        self, 
        n_reg: int,
        seed: int = SEED
    ):
        """
        Construct the Keras computation graph for the direct-H forecaster.

        Architecture
        ------------
        Input: shape (L, 1 + p).
        GRU(L→h): single GRU layer with `h = self._GRU1` units, L2 regularisation
        on kernel and recurrent weights, `reset_after=True`, and input dropout.
        
        Two dense heads operate on the final hidden state:
       
        1) Quantile head ("q_head"):
        
        - Dense(3) → raw vector z = (a, b, c).
        
        - Monotone transform:
        
                q_0.1 = a,
        
                q_0.5 = a + softplus(b) + 1e−6,
        
                q_0.9 = a + softplus(b) + softplus(c) + 2×1e−6.
        
        The transform enforces q_0.1 ≤ q_0.5 ≤ q_0.9 everywhere.

        2) Distribution head ("dist_head"):
        
        - Dense(3) → params = (μ_raw, σ_raw, ν_raw).
        
        - Location: μ = MU_MAX * tanh(μ_raw) (bounds location).
        
        - Scale:    σ = softplus(σ_raw) + σ_floor, later min with σ_max.
        
        - Dof:      ν = ν_floor + softplus(ν_raw).

        Losses
        ------
        - Pinball loss for quantiles:
        
            L_q = mean_i max(q_i * e_i, (q_i − 1) * e_i),  e_i = y − q̂_i,
        
        averaged over q ∈ {0.1, 0.5, 0.9}.
        
        - Student-t negative log-likelihood for residuals (see `student_t_nll`),
        plus regularisers.

        Parameters
        ----------
        n_reg : int
            Number of regressor columns p.
        seed : int
            Random seed for deterministic initialisation.

        Returns
        -------
        tf.keras.Model
            A model with outputs [q_head, dist_head].

        Why GRU
        -------
        GRUs maintain a memory of the sequence via gated updates:
       
            z_t = σ(W_z x_t + U_z h_{t−1} + b_z)         (update gate)
       
            r_t = σ(W_r x_t + U_r h_{t−1} + b_r)         (reset gate)
       
            h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t−1}) + b_h)
       
            h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t.

        Here σ is the logistic sigmoid and ⊙ denotes elementwise product.
        
        This mechanism adaptively controls information flow and mitigates vanishing
        gradients, enabling learning of dependencies across dozens of weeks with
        moderate parameter count and good sample efficiency.
        
        The GRU trades off copying the previous state (when z_t ≈ 0) and writing a
        new candidate (when z_t ≈ 1), while r_t controls the degree to which the
        previous state informs the candidate. This gating reduces vanishing
        gradients relative to plain RNNs and needs fewer parameters than LSTMs.
        """
        
        tf.random.set_seed(seed)

        inp = Input((self.SEQUENCE_LENGTH, 1 + n_reg), dtype = "float32")

        x = GRU(
            self._GRU1,
            return_sequences = False,  
            kernel_regularizer = l2(self.L2_LAMBDA),
            recurrent_regularizer = l2(self.L2_LAMBDA),
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
            recurrent_initializer = tf.keras.initializers.Orthogonal(seed = seed),
            dropout = self._DROPOUT,
            recurrent_dropout = 0.0,
            reset_after = True
        )(inp)

        z = Dense(3, name = "q_raw")(x)


        def monotone_quantiles(
            z
        ):
        
            a = z[:, 0:1]
        
            b_gap = tf.nn.softplus(z[:, 1:2]) + 1e-6
        
            c_gap = tf.nn.softplus(z[:, 2:3]) + 1e-6
        
            return tf.concat([a, a + b_gap, a + b_gap + c_gap], axis = -1)


        q_out = Lambda(monotone_quantiles, name = "q_head")(z)

        d_params = Dense(3, name = "dist_head")(x)

        return Model(inp, [q_out, d_params])

        
    def _make_callbacks(
        self,
        monitor: str
    ):
        """
        Create early-stopping and learning-rate-reduction callbacks.

        Parameters
        ----------
        monitor : str
            Validation metric to monitor (e.g., "val_loss").

        Returns
        -------
        List[tf.keras.callbacks.Callback]
            EarlyStopping with patience and best-weight restore; ReduceLROnPlateau
            with multiplicative decay and floor.

        Rationale
        ---------
        These safeguards prevent overfitting and adapt the learning rate in the
        presence of plateauing validation performance.
        """
    
        callback =  [
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
        
        return callback


    @staticmethod
    def _fit_pit_isotonic(
        y_true_res_cal: np.ndarray,
        mu_c: np.ndarray,
        sigma_c: np.ndarray,
        df_c: np.ndarray
    ):
        """
        Fit an isotonic regression on probability integral transforms (PITs) and
        derive inverse-mapped quantile levels for reporting.

        Procedure
        ---------
        1) Compute u_i = F_t((y_i − μ_i)/(σ_i) ; ν_i), where F_t is the CDF of the
        Student-t distribution with ν_i degrees of freedom; clip to (1e−6, 1−1e−6).
       
        2) Sort u_i and regress the empirical CDF v_i against u_sorted using an
        increasing isotonic mapping g : [0,1]→[0,1].
        
        3) On a dense grid, evaluate g and invert the map numerically to obtain
        corrected α levels for {0.10, 0.50, 0.90}.

        Returns
        -------
        inv_levels : np.ndarray, shape (3,)
            Corrected levels α′ so that g(α′) ≈ α; used when extracting quantiles.
        iso_model : IsotonicRegression
            The fitted forward map; retained to recalibrate residual draws.

        Rationale
        ---------
        PITs should be uniform under perfect calibration; isotonic regression
        adjusts monotone distortions without imposing parametric form.
        """

        u = student_t.cdf((y_true_res_cal.ravel() - mu_c.ravel()) / np.maximum(sigma_c.ravel(), 1e-8), df = df_c.ravel())
        
        u = np.clip(u, 1e-6, 1 - 1e-6)

        u_sorted = np.sort(u)
        
        v = np.linspace(1.0 / (len(u) + 1), len(u) / (len(u) + 1), len(u))

        iso = IsotonicRegression(y_min = 0.0, y_max = 1.0, increasing = True)
        
        iso.fit(u_sorted, v)

        grid = np.linspace(0.0, 1.0, 2001)
        
        g_vals = iso.predict(grid)
        
        want = np.array([0.10, 0.50, 0.90])
        
        inv_levels = np.interp(want, g_vals, grid)  

        return inv_levels.astype(np.float32), iso


    @staticmethod
    def pinball_loss(
        y_true, 
        y_pred,
        qs = (0.1, 0.5, 0.9)):
        """
        Quantile (pinball) loss averaged across specified quantiles.

        For quantile q and error e = y − ŷ:
            ρ_q(e) = max(q e, (q − 1) e).
        The returned loss is mean over batch and over all q in `qs`.

        Parameters
        ----------
        y_true : tf.Tensor, shape (N, 1)
        y_pred : tf.Tensor, shape (N, len(qs))
        qs : Tuple[float, ...]
        """
        
        y_true = tf.cast(y_true[:, 0:1], tf.float32)
        
        losses = []
        
        for i, q in enumerate(qs):
        
            e = y_true - y_pred[:, i:i+1]
        
            losses.append(tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e)))
        
        return tf.add_n(losses) / float(len(qs))


    def _recalibrate_residual_draws(
        self, 
        z: np.ndarray, 
        nu: np.ndarray
    ) -> np.ndarray:
        """
        Apply PIT-based isotonic recalibration to raw Student-t residual draws.

        Given z ∼ t_ν, compute u = F_t(z; ν), apply the learned isotonic map
        v = g(u), and return z′ = F_t^{-1}(v; ν). Shapes (mc, N) are supported;
        a trailing singleton dimension is squeezed if present.

        Parameters
        ----------
        z : np.ndarray, shape (mc, N) or (mc, N, 1)
        nu : np.ndarray, shape (mc, N)
            Degrees of freedom per draw.

        Returns
        -------
        np.ndarray, shape (mc, N)
            Calibrated residual draws.

        Notes
        -----
        If no isotonic model has been trained yet, returns the input unchanged.
        """

        if not hasattr(self, "_iso_model") or (self._iso_model is None):
       
            return z
       
        if z.ndim == 3 and z.shape[-1] == 1:
       
            z = z[..., 0]  
            
        u = student_t.cdf(z, df = nu)
        
        u = np.clip(u, 1e-6, 1 - 1e-6)
        
        v = self._iso_model.predict(u.ravel()).reshape(u.shape)
        
        v = np.clip(v, 1e-6, 1 - 1e-6)
        
        z_cal = student_t.ppf(v, df=nu)
        
        z_cal = np.where(np.isfinite(z_cal), z_cal, z)
        
        return z_cal.astype(np.float32, copy = False)

    
    def student_t_nll(
        self, 
        y_true_res, 
        params
    ):
        """
        Negative log-likelihood of residuals under a heteroscedastic Student-t
        with mild regularisation.

        Parameterisation
        ----------------
        Given Dense(3) outputs `params = (μ_raw, σ_raw, ν_raw)`:
       
        - μ = MU_MAX * tanh(μ_raw)               (bounded location),
       
        - σ = softplus(σ_raw) + σ_floor          (strictly positive),
        later multiplied by a global calibration factor and capped at σ_max,
       
        - ν = ν_floor + softplus(ν_raw)          (ν > ν_floor > 2).

        For residual y, define z = (y − μ)/σ. The per-sample log density is:
       
            log f(y) = log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(νπ) − log σ − ½ (ν+1) log(1 + z²/ν).
        
        The loss is the minibatch mean of −log f(y) plus penalties:
        
        - λ_sigma * mean( log(σ)² ),
        
        - λ_invnu * mean( 1 / (ν − 2) ),
        
        - λ_mu_raw * mean( μ_raw² ).

        Parameters
        ----------
        y_true_res : tf.Tensor, shape (N, 1)
        params : tf.Tensor, shape (N, 3)

        Returns
        -------
        tf.Tensor (scalar)
            Regularised negative log-likelihood.
        """
    
        y = tf.cast(y_true_res, tf.float32)

        mu_raw = tf.cast(params[:, 0:1], tf.float32)       
       
        mu = self.MU_MAX * tf.tanh(mu_raw)

        log_sigma = tf.cast(params[:, 1:2], tf.float32)
       
        sigma = tf.nn.softplus(log_sigma) + self.SIGMA_FLOOR
        
        sigma = tf.minimum(sigma * getattr(self, "_SIGMA_CAL_SCALE", 1.0), self.SIGMA_MAX)

        df_raw = tf.cast(params[:, 2:3], tf.float32)
        
        df = self.NU_FLOOR + tf.nn.softplus(df_raw)

        z = (y - mu) / sigma

        logZ = (tf.math.lgamma((df + 1.0) / 2.0)
                - tf.math.lgamma(df / 2.0)
                - 0.5 * tf.math.log(df * np.pi)
                - tf.math.log(sigma))
        
        log_pdf = logZ - 0.5 * (df + 1.0) * tf.math.log(1.0 + tf.square(z) / df)

        nll = -tf.reduce_mean(log_pdf)

        reg_sigma = self.LAMBDA_SIGMA * tf.reduce_mean(tf.square(tf.math.log(sigma)))

        reg_invnu = self.LAMBDA_INVNU * tf.reduce_mean(1.0 / (df - 2.0))

        reg_mu_raw = self.LAMBDA_MU_RAW * tf.reduce_mean(tf.square(mu_raw))

        return nll + reg_sigma + reg_invnu + reg_mu_raw


    def build_state(
        self
    ) -> Dict[str, Any]:
        """
        Load raw data, construct stationary macro series, align weekly prices and
        regressors, compute feature moments, and precompute scenario engines.

        Outputs
        -------
        A serialisable dictionary with keys (non-exhaustive):
        - "tickers": ordered list of tickers grouped by country.
        - "macro_weekly_by_country": dict[country] → weekly macro DataFrame.
        - "macro_weekly_idx_by_country": dict[country] → macro weekly index (np.ndarray).
        - "country_var_results": dict[country] → VAR(1) fit artefacts or None.
        - "macro_future_by_country": dict[country] → simulated macro deltas
        of shape (S, H, 4) in stationary space (either VAR-based or white noise
        with Ledoit–Wolf covariance when insufficient sample).
        - "factor_future_global": np.ndarray, shape (S, H, n_factors), stationary
        bootstrap resamples of factor returns.
        - "factor_weekly_index", "factor_weekly_values": weekly factor panel.
        - "weekly_price_by_ticker": dict[ticker] → {index, y (levels), lr (log-ret)}.
        - "moments_by_ticker": dict[ticker] → 104-week rolling skew/kurt shifted
        by +H to avoid leakage into the target window.
        - "fd_weekly_by_ticker": dict[ticker] → resampled weekly revenue/EPS levels.
        - "align_cache": dict[ticker] → indices mapping from price dates to macro,
        factor, and financial weekly indices; includes a mask of valid positions.
        - "next_fc": DataFrame of next-period revenue/EPS points used to build
        scenario paths.
        - "latest_price": dict-like mapping ticker → latest level.
        - "ticker_country": Series mapping ticker → country string.
        - "presence_flags": dict[ticker] → {has_factors, has_fin, has_moms} booleans.

        Key steps and rationale
        -----------------------
        - Macro: convert to weekly frequency via 'W-FRI' resampling and forward
        fill; differencing/log-differencing enforces approximate stationarity.
        - VAR(1): fit per country in stationary space; ensure stability and SPD Σ.
        When not estimable, fall back to Gaussian innovations with LW Σ.
        - Factors: weekly returns panel; generate future paths using stationary
        bootstrap to preserve dependence.
        - Financials: resample to weekly last observation carried forward to align
        with price/factor/macro cadence.
        - Moments: compute 104-week rolling skewness and kurtosis of returns and
        shift by +H weeks so that only information available at forecast origin
        is used as a 'known future' regressor.

        Determinism
        -----------
        Relies on `self.SEED` for NumPy RNGs. All constructed arrays are float32
        where appropriate to match model expectations.

        Performance
        -----------
        Uses vectorised operations (e.g., `np.searchsorted`, views) and caches
        to avoid repeated passes over large frames.
        """
    
        self.logger.info("Building global state …")
    
        if self.SAVE_ARTIFACTS and not _os.path.exists(self.ARTIFACT_DIR):
    
            _os.makedirs(self.ARTIFACT_DIR, exist_ok = True)

        fdata = FinancialForecastData()
      
        macro = fdata.macro
    
        r = macro.r

        if self.tickers_arg:
    
            tickers = list(self.tickers_arg)
    
        else:
    
            tickers = ['NVDA', 'GOOG', 'TTWO', 'TJX']
            
            #tickers = ['AMZN', 'GOOG', 'KO', 'MSFT', 'NVDA', 'TJX', 'TTD', 'TTWO']

        close_df = r.weekly_close
    
        dates_all = close_df.index.values
        
        tick_list = close_df.columns.values
        
        price_arr = close_df.to_numpy(dtype = np.float32, copy = False)

        T_dates, M = price_arr.shape
    
        price_rec = np.empty(
            T_dates * M,
            dtype=[("ds", "datetime64[ns]"),
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

        raw_macro = macro.assign_macro_history_non_pct().reset_index()
      
        raw_macro = raw_macro.rename(columns = {"year": "ds"} if "year" in raw_macro else {raw_macro.columns[1]: "ds"})
       
        if isinstance(raw_macro.index, pd.PeriodIndex):
       
            raw_macro.index = raw_macro.index.to_timestamp(how = "end")
       
        ds = raw_macro["ds"]
       
        if is_period_dtype(ds):
       
            raw_macro["ds"] = ds.dt.to_timestamp(how = "end")
       
        elif is_datetime64_any_dtype(ds):
       
            pass
       
        elif ds.dtype == object and len(ds) and isinstance(ds.iloc[0], pd.Period):
     
            raw_macro["ds"] = pd.PeriodIndex(ds).to_timestamp(how = "end")
     
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
            
            if i + 1 < len(first_idx):
        
                end = first_idx[i + 1] 
            
            else:
                
                end = len(macro_rec)
        
            country_slices[ctry] = (start, end)

        country_var_results: Dict[str, Optional[dict]] = {}
      
        macro_future_by_country: Dict[str, Optional[np.ndarray]] = {}
      
        rng_global = np.random.default_rng(self.SEED)
              
        hor = self.HORIZON
        
        seq_len = self.SEQUENCE_LENGTH
        
        S = self.N_SIMS

        macro_weekly_by_country = {}
        
        macro_weekly_idx_by_country = {}

        for ctry, (s, e) in country_slices.items():
      
            rec = macro_rec[s:e]
      
            if len(rec) == 0:
      
                country_var_results[ctry] = None
      
                macro_future_by_country[ctry] = None

                macro_weekly_by_country[ctry] = None
        
                macro_weekly_idx_by_country[ctry] = None
                
                continue
      
            dfm = pd.DataFrame({reg: rec[reg] for reg in self.MACRO_REGRESSORS},
                               index = pd.DatetimeIndex(rec["ds"]))
         
            dfw = (dfm[~dfm.index.duplicated(keep = "first")]
                   .sort_index()
                   .resample("W-FRI").mean()
                   .ffill()
                   .dropna())
            
            macro_weekly_by_country[ctry] = dfw
           
            macro_weekly_idx_by_country[ctry] = dfw.index.values   
                  
            if dfw.shape[1] == 0 or len(dfw) <= dfw.shape[1] + 4:
         
                country_var_results[ctry] = None
         
                dfm_stat, _ = self._to_stationary_macro(
                    dfm = dfw
                )
                
                tail = dfm_stat.tail(52)
                
                Sigma_f = self._lw_cov_or_eye(
                    tail_vals = tail.values if len(tail) else None, 
                    dim = 4
                )
                
                Sigma_f = self.ensure_spd_for_cholesky(
                    Sigma = Sigma_f
                )
                
                eps = rng_global.multivariate_normal(
                    mean = np.zeros(4, np.float32), 
                    cov = Sigma_f, 
                    size = (S, hor),
                    method = "cholesky"
                ).astype(np.float32)
                
                macro_future_by_country[ctry] = eps  
              
                continue

            dfm_stat, _ = self._to_stationary_macro(
                dfm = dfw
            )
            
            try:
                vr = self.fit_var1_with_intercept(
                    dfm_stationary = dfm_stat
                )
            
            except Exception:
            
                vr = None
            
            country_var_results[ctry] = vr

            if vr is not None:
            
                A = vr["A"]
              
                c =  vr["c"]
              
                Sigma = vr["Sigma"]
              
                x0 = vr["last_x"]
              
                Xq = self.simulate_var_paths_closed_form(
                    A = A, 
                    c = c, 
                    Sigma = Sigma, 
                    x0 = x0, 
                    Hq = 4,
                    n_sims = S, 
                    rng = rng_global
                )  
              
                macro_future_by_country[ctry] = np.repeat(Xq, self.REPEATS_QUARTER, axis = 1)  
            
            else:
            
                tail = dfm_stat.tail(52)
            
                Sigma_f = self._lw_cov_or_eye(
                    tail_vals = tail.values if len(tail) else None, 
                    dim = 4
                )
                
                Sigma_f = self.ensure_spd_for_cholesky(
                    Sigma = Sigma_f
                )
                
                eps = rng_global.multivariate_normal(
                    mean = np.zeros(4, np.float32), 
                    cov = Sigma_f, 
                    size = (S, hor), 
                    method = "cholesky"
                ).astype(np.float32)
                
                macro_future_by_country[ctry] = eps
        
        fin_raw = fdata.prophet_data
     
        fd_rec_dict: Dict[str, np.ndarray] = {}
     
        for t in tickers:
     
            df_fd = (fin_raw.get(t, pd.DataFrame())
                     .reset_index()
                     .rename(columns = {
                         "index": "ds", 
                         "rev": "Revenue",
                         "eps": "EPS (Basic)"
                         })
            )
            
            if df_fd.empty:
            
                continue
            
            df_fd["ds"] = pd.to_datetime(df_fd["ds"])
            
            df_fd = df_fd[["ds", "Revenue", "EPS (Basic)"]].dropna()
            
            if df_fd.empty:
            
                continue
            
            rec = np.empty(len(df_fd), dtype = [
                ("ds", "datetime64[ns]"), 
                ("Revenue", "float32"), 
                ("EPS (Basic)", "float32")
            ])
            
            rec["ds"] = df_fd["ds"].values
            
            rec["Revenue"] = df_fd["Revenue"].to_numpy(dtype = np.float32, copy = False)
            
            rec["EPS (Basic)"] = df_fd["EPS (Basic)"].to_numpy(dtype = np.float32, copy = False)
            
            fd_rec_dict[t] = np.sort(rec, order="ds")

        fac_w: pd.DataFrame = r.factor_weekly_rets()  
        
        if set(self.FACTORS).issubset(fac_w.columns):
       
            fac_vals = fac_w[self.FACTORS].dropna().to_numpy(dtype = np.float32) 
            
        else:
            
            fac_vals = np.zeros((0, len(self.FACTORS)), np.float32)
       
        if fac_vals.size == 0:
    
            factor_future_global = np.zeros((S, hor, len(self.FACTORS)), np.float32)
    
        else:
    
            fac_vals_zero_mean = fac_vals - np.nanmean(fac_vals, axis=0, keepdims=True) 
    
            L = fac_vals_zero_mean.shape[0]
    
            idx = self.stationary_bootstrap_indices(
                L = L,
                n_sims = S,
                H = hor,
                p = self._BOOT_P,
                rng = rng_global
            )
          
            factor_future_global = fac_vals_zero_mean[idx, :]  

        factor_weekly = (fac_w.sort_index()
                         .resample("W-FRI").mean()
                         .ffill())
        
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
   
            c = _norm_country(
                x = analyst["country"].get(t, None)
            )
            
            by_country.setdefault(c, []).append(t)

        grouped_tickers: List[str] = []
      
        for c in sorted(by_country.keys(), key = str):
      
            grouped_tickers.extend(sorted(by_country[c]))
            
        moments_by_ticker = {}
       
        weekly_price_by_ticker: Dict[str, Dict[str, np.ndarray]] = {} 
       
        fd_weekly_by_ticker: Dict[str, Dict[str, np.ndarray]] = {}   
     
        for t in grouped_tickers:
         
            pr = price_rec[price_rec["Ticker"] == t]
         
            if len(pr) == 0:
         
                continue

            s = (pd.DataFrame({"ds": pr["ds"], "y": pr["y"]})
                   .set_index("ds").sort_index()["y"]
                   .resample("W-FRI").last()
                   .ffill())
          
            y = s.to_numpy(dtype = np.float32, copy = False)
          
            ys = np.maximum(y, self.SMALL_FLOOR)
          
            lr = np.zeros_like(ys, dtype = np.float32)
           
            lr[1:] = np.log(ys[1:]) - np.log(ys[:-1])
           
            weekly_price_by_ticker[t] = {
                "index": s.index.values,
                "y": y,
                "lr": lr,
            }

            s_lr = pd.Series(lr, index = s.index)
           
            skew = s_lr.rolling(seq_len, min_periods = seq_len).skew().shift(hor)
           
            kurt = s_lr.rolling(seq_len, min_periods = seq_len).kurt().shift(hor)
           
            moments_by_ticker[t] = np.column_stack((
                np.nan_to_num(skew.to_numpy(dtype = np.float32), nan = 0.0, posinf = 0.0, neginf = 0.0),
                np.nan_to_num(kurt.to_numpy(dtype = np.float32), nan = 0.0, posinf = 0.0, neginf = 0.0),
            )).astype(np.float32, copy = False)

            if t in fd_rec_dict:

                df_fd = (pd.DataFrame({
                            "ds": fd_rec_dict[t]["ds"],
                            "Revenue": fd_rec_dict[t]["Revenue"],
                            "EPS (Basic)": fd_rec_dict[t]["EPS (Basic)"],
                        })
                        .set_index("ds")
                        .sort_index())
              
                fdw = df_fd.resample("W-FRI").last().ffill()
               
                fd_weekly_by_ticker[t] = {
                    "index": fdw.index.values,
                    "values": fdw[["Revenue", "EPS (Basic)"]].to_numpy(dtype = np.float32, copy = False),
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
       
            idx_m = np.searchsorted(mw_idx, dates, side="right") - 1
       
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
           
                valid_fd= np.ones(len(dates), bool)
           
            keep = valid_m & valid_fa & (valid_fd if idx_fd is not None else True)
           
            idx_keep = np.nonzero(keep)[0]
           
            align_cache[t] = {
                "idx_m": idx_m, "valid_m": valid_m,
                "idx_fa": idx_fa, "valid_fa": valid_fa,
                "idx_fd": idx_fd, "valid_fd": valid_fd,
                "idx_keep": idx_keep,
            }

        state_pack: Dict[str, Any] = {
            "tickers": grouped_tickers,
            "country_var_results": country_var_results,
            "macro_future_by_country": macro_future_by_country,  
            "next_fc": fdata.next_period_forecast(),
            "latest_price": r.last_price,
            "ticker_country": analyst["country"],
            "factor_future_global": factor_future_global, 
            "macro_weekly_by_country": macro_weekly_by_country,
            "moments_by_ticker": moments_by_ticker,
            "macro_weekly_idx_by_country": macro_weekly_idx_by_country,
            "factor_weekly_index": factor_weekly_index,
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
            }
            for t in grouped_tickers
        }
        state_pack["presence_flags"] = presence_flags  
              
        self.logger.info("Global state built (%d tickers).", len(grouped_tickers))
        
        return state_pack


    def forecast_one(
        self, 
        ticker: str
    ) -> Dict[str, Any]:
        """
        Train a ticker-specific model on aligned windows, calibrate on a hold-out,
        simulate residual-augmented scenarios, and summarise predictive prices.

        Pipeline
        --------
        1) Preconditions and alignment
        - Retrieve aligned indices mapping the ticker's price dates to
            country macro, factor, and financial streams (`align_cache`).
        - Enforce index upper bounds and drop invalid positions.
        - Require at least L+1 aligned observations to create ≥1 training
            window.

        2) Feature construction
        - Build regressor level matrix, then transform to deltas using
            `build_transform_matrix` and robust-scale via `fit_scale_deltas`.
        - Scale returns using a robust scaler (training mask only).
        - Construct windows `X_all` and direct-H targets `y_all`.

        3) Train/validation split
        - Use `choose_splits` to select a time-series split; default to 80/20
            split when only 1 fold is available.

        4) Baseline residuals
        - Fit AR(1) with intercept on training returns; compute baseline sums
            μ_base for train, validation, and 'now' using `ar1_sum_forecast`.
        - Residual targets: y_res = y − μ_base.

        5) Model training
        - (Re)initialise a cached Keras model for this `n_reg`.
        - Compile with Adam(clipnorm=1), dual loss (`pinball_loss`, `student_t_nll`)
            and equal weights.
        - Train with shuffled mini-batches, early stopping, and LR plateau
            reduction. Reduce epochs when data are scarce.

        6) Validation calibration
        - Forward pass on validation windows to obtain (μ, σ, ν).
        - Grid search for global σ multiplier `s*` that matches central
            coverages (`_calibrate_sigma_scale`).
        - Fit isotonic regression on PITs; store corrected levels α′ and the
            isotonic model for draw recalibration.

        7) Scenario simulation
        - Build the last historical slice `X_hist` and assemble future deltas
            by combining:
            * macro deltas from per-country VAR paths (weekly expanded),
            * factor deltas from stationary bootstrap,
            * financial deltas derived from next-period revenue/EPS guidance,
                converted into weekly log-increments with mean-zero noise per
                quarter (noise debiased to preserve quarterly aggregates),
            * moment deltas equal to the last observed H deltas (persistence).
        - Robust-scale the future deltas with training scaler and clip bounds.

        8) Residual sampling with MC-dropout
        - For blocks of S scenarios and T Monte Carlo passes, forward-propagate
            `X_block` with `training=True` to activate dropout. Collect per-pass
            (μ_t, σ_t, ν_t).
        - Draw z ∼ t_ν, recalibrate via PIT if available, form residuals
            ε̂ = μ_t + σ_t z, and add μ_base_now to obtain total sums.
        - Convert sums of log returns to price factors via expm1, accumulate
            across passes to form a large sample.

        9) Summaries
        - Extract central interval quantiles using corrected α′ on both the raw
            sample and the PIT-calibrated sample.
        - Report price levels by scaling the latest price; also report sample
            mean and standard deviation of return sums.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]] on success
            raw_dict  : {"Ticker", "status", "Min Price", "Avg Price",
                        "Max Price", "Returns", "SE"}
            calib_dict: same keys but using PIT-calibrated residual draws.
        Dict[str, Any] on skip/error
            {"Ticker", "status": "skipped"/"error", "reason": "..."}.

        Failure modes and skip reasons
        ------------------------------
        - "insufficient_macro_history", "alignment_missing",
        "no_latest_price", "no_price_history_weekly",
        "short_price_history", "insufficient_joint_features",
        "insufficient_joint_features_after_upperbound",
        "too_short_after_align", "no_training_windows",
        "too_few_training_deltas_for_scaler", "window_count_mismatch".

        Rationale
        ---------
        Training on residuals of a simple, interpretable baseline yields better
        calibration; combining epistemic (dropout) and aleatoric (t-head) sources
        of uncertainty enables realistic interval estimates.
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
            
            S = self.N_SIMS
            
            flags = STATE["presence_flags"][ticker]

            next_fc = STATE["next_fc"]

            latest_price = STATE["latest_price"]

            ticker_country = STATE["ticker_country"]

            fa_vals = STATE["factor_weekly_values"]

            macro_future_by_country = STATE["macro_future_by_country"]

            factor_future_global = STATE["factor_future_global"]

            moms_vals = STATE["moments_by_ticker"].get(ticker)
            
            ctry = STATE["ticker_country"].get(ticker)
            
            dfm_ct = STATE["macro_weekly_by_country"].get(ctry)
            
            n_m = len(dfm_ct)
            
            if dfm_ct is None or n_m < 12:
                
                return skip(
                    reason = "insufficient_macro_history"
                )
            
            align = STATE["align_cache"].get(ticker)
            
            if align is None:

                return skip(
                    reason = "alignment_missing"
                )
            
            idx_m = np.asarray(align["idx_m"],  dtype = np.int64)

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
          
            dates = wp["index"]
          
            yv = wp["y"]
          
            lr = wp["lr"]
                      
            fdw = STATE["fd_weekly_by_ticker"].get(ticker, None)
            
            SEQ_LEN = self.SEQUENCE_LENGTH
           
            if len(yv) < SEQ_LEN + 2:
                
                return skip(
                    reason = "short_price_history"
                )
                
            small_floor = self.SMALL_FLOOR
                                       
            len_idx_keep = len(idx_keep)

            if len_idx_keep < SEQ_LEN + 1:
                
                return skip(
                    reason = "insufficient_joint_features"
                )

            sel_m = np.clip(sel_m, 0, n_m - 1)
            
            n_fa = fa_vals.shape[0]

            if fa_vals.shape[0] > 0:
                
                sel_fa = np.clip(sel_fa, 0, fa_vals.shape[0] - 1)
            
            if flags["has_factors"]:
          
                upper_ok = (sel_m < n_m) & (sel_fa < n_fa)
          
            else:
               
                sel_fa = np.zeros_like(sel_m)  
               
                upper_ok = (sel_m < n_m)
          
            idx_keep = idx_keep[upper_ok]
          
            sel_m = sel_m[upper_ok]
          
            sel_fa = sel_fa[upper_ok]
          
            if sel_fd is not None:
          
                sel_fd = sel_fd[upper_ok]
          
            if idx_keep.shape[0] < SEQ_LEN + 1:
          
                return skip(
                    reason = "insufficient_joint_features_after_upperbound"
                )

            log_ret_full = lr[idx_keep].astype(np.float32)
        
            regs = list(self.ALL_REGRESSORS)

            reg_pos = {name: i for i, name in enumerate(regs)}

            n_reg = len(regs)
            
            n_ch = 1 + n_reg

            len_idx_keep = idx_keep.shape[0]

            reg_mat = np.zeros((len_idx_keep, n_reg), dtype = np.float32)

            macro_vals = dfm_ct[["Interest", "Cpi", "Gdp", "Unemp"]].to_numpy(np.float32, copy = False)

            reg_mat[:, [reg_pos["Interest"], reg_pos["Cpi"], reg_pos["Gdp"], reg_pos["Unemp"]]] = macro_vals[sel_m]

            if flags["has_factors"] and fa_vals.shape[0] > 0:

                reg_mat[:, [reg_pos[f] for f in self.FACTORS]] = fa_vals[sel_fa, :]

            if flags["has_fin"] and (fdw is not None) and (sel_fd is not None):
                
                sel_fd = np.clip(sel_fd, 0, len(fdw["index"]) - 1)
                
                reg_mat[:, [reg_pos["Revenue"], reg_pos["EPS (Basic)"]]] = fdw["values"][sel_fd, :]

            if flags["has_moms"] and moms_vals is not None:

                reg_mat[:, [reg_pos["skew_104w_lag52"], reg_pos["kurt_104w_lag52"]]] = moms_vals[idx_keep, :]

            
            DEL_full = self.build_transform_matrix(
                reg_mat = reg_mat,
                regs = regs
            )

            global_cache = self._MODEL_CACHE
        
            cached = global_cache.get(n_reg)
            
            N_total = reg_mat.shape[0]
        
            if N_total < (SEQ_LEN + 4):
        
                return skip(
                    reason = "too_short_after_align"
                )
        
            if cached is None:
             
                model = self.build_directH_model(
                    n_reg = n_reg, 
                    seed = self.SEED
                )
                
                opt = Adam(learning_rate = self._LR, clipnorm = 1.0)
                
                model.compile(
                    optimizer = opt,
                    loss = {
                        "q_head": self.pinball_loss, 
                        "dist_head": self.student_t_nll
                    },
                    loss_weights = {
                        "q_head": 0.5, 
                        "dist_head": 0.5
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
                
                    return model(
                        x, 
                        training = False
                    )
               
               
                @tf.function(reduce_retracing = True, input_signature = [spec])
                def _fwd_train(
                    x
                ):
                
                    return model(
                        x, 
                        training = True
                    )
                    
                
                self._FN_CACHE[n_reg] = {
                    "fwd": _fwd,
                    "fwd_train": _fwd_train
                }
            
            meta = self._ensure_meta_cache(
                n_reg = n_reg, 
                regs = regs
            )
            
            reg_pos = meta["reg_pos"]  
                      
            fns = self._FN_CACHE[n_reg]
                                   
            HIST = self.HIST_WINDOW
          
            HOR = self.HORIZON

            n_all = len_idx_keep - SEQ_LEN + 1
           
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
           
            del_len = DEL_full.shape[0]
           
            starts = train_idx
           
            ends = np.minimum(train_idx + SEQ_LEN - 1, del_len - 1)

            d = np.zeros(del_len + 1, dtype = np.int32)
           
            np.add.at(d, starts, 1)
           
            np.add.at(d, ends + 1, -1)
           
            train_mask = np.cumsum(d[:-1]) > 0

            if train_mask.sum() < n_reg:
                
                return skip(
                    reason = "too_few_training_deltas_for_scaler"
                )

            sc_reg_full, ql_full, qh_full = self.fit_scale_deltas(
                DEL_tr = DEL_full[train_mask]
            )

            scaled_reg_full = self.transform_deltas(
                DEL = DEL_full, 
                sc = sc_reg_full, 
                q_low = ql_full, 
                q_high = qh_full
            )
                      
            ret_mask = np.zeros_like(log_ret_full, dtype = bool)
            
            ret_mask[train_idx.min(): train_idx.max() + self.HIST_WINDOW] = True
            
            ret_mask[0] = False
                        
            if ret_mask.sum() < 8:
            
                ret_scaler_full = self.fit_ret_scaler_from_logret(
                    log_ret = log_ret_full
                )  
            
            else:
            
                sc = RobustScaler().fit(log_ret_full[ret_mask].reshape(-1, 1))
            
                sc.scale_[sc.scale_ < 1e-6] = 1e-6
            
                ret_scaler_full = sc
                
            scaled_ret_full = self.scale_logret_with(
                log_ret = log_ret_full, 
                sc = ret_scaler_full
            )

            X_all, _ = self.make_windows_directH(
                scaled_ret = scaled_ret_full,
                scaled_reg = scaled_reg_full,
                hist = HIST,
                hor = HOR,
                seq_length = SEQ_LEN
            )
            
            y_all = self.make_target_directH(
                log_ret = log_ret_full, 
                hist = HIST,
                hor = HOR, 
                seq_len = SEQ_LEN, 
                N_expected = len(X_all)
            )
           
            if len(X_all) != n_all:
                
                return skip(
                    reason = "window_count_mismatch"
                )

            X_tr = X_all[train_idx]
            
            y_tr = y_all[train_idx]
           
            X_cal = X_all[cal_idx]
        
            y_cal = y_all[cal_idx]
            
            last_train_end = train_idx[-1] + HIST  
         
            r_train = log_ret_full[:last_train_end + 1]   
         
            m_hat, phi_hat = self.fit_ar1_baseline(
                log_ret = r_train
            )


            def baseline_for_windows(
                last_returns: np.ndarray, 
                H: int
            ) -> np.ndarray:

                if abs(1.0 - phi_hat) < 1e-8:
            
                    return (H * m_hat) * np.ones((last_returns.shape[0], 1), dtype = np.float32)
            
                geom = (1.0 - (phi_hat ** H)) / (1.0 - phi_hat)
            
                return (H * m_hat + (last_returns - m_hat) * phi_hat * geom).astype(np.float32)[:, None]


            last_r_tr = log_ret_full[train_idx + HIST - 1]

            last_r_cal = log_ret_full[cal_idx + HIST - 1]

            mu_base_tr = baseline_for_windows(
                last_returns = last_r_tr,  
                H = HOR
            )
            
            mu_base_cal = baseline_for_windows(
                last_returns = last_r_cal, 
                H = HOR
            )

            y_tr_res = y_tr - mu_base_tr
            
            y_cal_res = y_cal - mu_base_cal
            
            last_r_now = float(log_ret_full[-1])

            mu_base_now = float(self.ar1_sum_forecast(
                r_t = last_r_now, 
                m = m_hat, 
                phi = phi_hat, 
                H = HOR
            ))
            
            mu_base_now = np.float32(mu_base_now) 

            Xc = tf.convert_to_tensor(X_cal)

            outs_cal = fns["fwd"](Xc)

            if isinstance(outs_cal, dict):
                
                params_cal = outs_cal["dist_head"]  
            
            else:
                
                params_cal = outs_cal[1]

            params_cal = params_cal.numpy().astype(np.float32, copy = False)

            mu_c = self.MU_MAX * np.tanh(params_cal[:, 0:1])
           
            sigma_c = np.log1p(np.exp(params_cal[:, 1:2])) + self.SIGMA_FLOOR
           
            sigma_c = np.minimum(self.SIGMA_MAX, sigma_c)
           
            df_c = self.NU_FLOOR + np.log1p(np.exp(params_cal[:, 2:3]))

            self._SIGMA_CAL_SCALE = float(self._calibrate_sigma_scale(
                y_cal_res = y_cal_res, 
                mu_c = mu_c, 
                sigma_c = sigma_c, 
                df_c = df_c
            ))

            sigma_c_cal = np.minimum(self.SIGMA_MAX, sigma_c * self._SIGMA_CAL_SCALE)

            self._alpha_levels_adj, self._iso_model = self._fit_pit_isotonic(
                y_true_res_cal = y_cal_res,
                mu_c = mu_c,
                sigma_c = sigma_c_cal,
                df_c = df_c
            )
            
            if len(X_tr) > 4 * self.BATCH:
                
                effective_epochs = self.EPOCHS  
            
            else:
                
                effective_epochs = max(8, self.EPOCHS // 2)

            callbacks = self._make_callbacks(
                monitor = "val_loss"
            )

            ds_tr = tf.data.Dataset.from_tensor_slices(
                (
                    X_tr, 
                    {
                        "q_head": y_tr_res,
                        "dist_head": y_tr_res
                    }
                )
            ).shuffle(
                buffer_size = min(len(X_tr), 4096), 
                seed = self.SEED, 
                reshuffle_each_iteration = True
            ).batch(self.BATCH).prefetch(tf.data.AUTOTUNE)

            ds_val = tf.data.Dataset.from_tensor_slices(
                (
                    X_cal, 
                    {
                        "q_head": y_cal_res, 
                        "dist_head": y_cal_res
                    }
                )
            ).batch(self.BATCH).prefetch(tf.data.AUTOTUNE)

            model.fit(
                ds_tr, 
                epochs = effective_epochs, 
                callbacks = callbacks, 
                verbose = 0, 
                validation_data = ds_val
            )

            X_hist = np.empty((1, HIST, n_ch), dtype = np.float32)
            
            sr = scaled_ret_full[-(SEQ_LEN):]  
       
            X_hist[0, :, 0] = sr[:HIST]
       
            X_hist[0, :, 1:] = self._last_hist_reg_deltas(
                delta_reg = scaled_reg_full,
                hist = HIST, 
                hor = HOR, 
                seq_length = SEQ_LEN
            )

            country_t = ticker_country.get(ticker, "UNK")
            
            macro_future_deltas = macro_future_by_country.get(country_t)
            
            if macro_future_deltas is None:
            
                dfm_ct_stat, _ = self._to_stationary_macro(
                    dfm = dfm_ct
                )
            
                tail = dfm_ct_stat.tail(52)
            
                Sigma_f = self._lw_cov_or_eye(
                    tail_vals = tail.values if len(tail) else None, 
                    dim = 4
                )
            
                Sigma_f = self.ensure_spd_for_cholesky(
                    Sigma = Sigma_f
                )
            
                macro_future_deltas = rng.multivariate_normal(
                    mean = np.zeros(4, np.float32), 
                    cov = Sigma_f, 
                    size = (S, HOR),
                    method = "cholesky"
                ).astype(np.float32)  
                
            S_all = factor_future_global.shape[0]
            
            sel = rng.choice(S_all, size = S, replace = (S_all < S))
           
            factor_future = factor_future_global[sel]  
           
            n_scn = len(SCENARIOS)

            rev_points_arr = np.array([float(next_fc.at[ticker, r_key]) if (ticker in next_fc.index and r_key in next_fc.columns) else np.nan
                                       for r_key, _ in SCENARIOS], dtype = np.float32)
           
            eps_points_arr = np.array([float(next_fc.at[ticker, e_key]) if (ticker in next_fc.index and e_key in next_fc.columns) else np.nan
                                       for _, e_key in SCENARIOS], dtype = np.float32)

            if flags["has_fin"]:

                last_rev = float(fdw["values"][sel_fd][-1, 0])

                last_eps = float(fdw["values"][sel_fd][-1, 1])

            else:

                last_rev = small_floor

                last_eps = small_floor

            last_rev = max(last_rev, small_floor)
           
            last_eps = max(last_eps, small_floor)

            mu_rev = np.zeros(n_scn, np.float32)
           
            vr = rev_points_arr > small_floor
           
            mu_rev[vr] = (np.log(rev_points_arr[vr]) - np.log(last_rev)) / 4.0
           
            q_rev = last_rev * np.exp(np.cumsum(np.repeat(mu_rev.reshape(n_scn, 1), 4, axis = 1), axis = 1)).astype(np.float32)

            mu_eps = np.zeros(n_scn, np.float32)
          
            ve = eps_points_arr > small_floor
          
            mu_eps[ve] = (np.log(eps_points_arr[ve]) - np.log(last_eps)) / 4.0
           
            q_eps = last_eps * np.exp(np.cumsum(np.repeat(mu_eps.reshape(n_scn, 1), 4, axis = 1), axis = 1)).astype(np.float32)

            with np.errstate(divide = "ignore", invalid = "ignore"):
          
                log_prev_r = np.log(last_rev)
          
                log_prev_e = np.log(last_eps)
          
                log_q_rev = np.log(np.maximum(q_rev, small_floor))
          
                log_q_eps = np.log(np.maximum(q_eps, small_floor))

            shifts_rev = np.concatenate([(log_q_rev[:, :1] - log_prev_r), (log_q_rev[:, 1:] - log_q_rev[:, :-1])], axis = 1)
          
            shifts_eps = np.concatenate([(log_q_eps[:, :1] - log_prev_e), (log_q_eps[:, 1:] - log_q_eps[:, :-1])], axis = 1)

            d_rev_week = np.repeat(shifts_rev.reshape(n_scn, 1, 4), S, axis = 1)
          
            d_rev_week = np.repeat(d_rev_week, self.REPEATS_QUARTER, axis = 2)  
          
            d_eps_week = np.repeat(shifts_eps.reshape(n_scn, 1, 4), S, axis = 1)
          
            d_eps_week = np.repeat(d_eps_week, self.REPEATS_QUARTER, axis = 2)

            rev_noise = rng.normal(0.0, self.REV_NOISE_SD, size = d_rev_week.shape).astype(np.float32)
           
            eps_noise = rng.normal(0.0, self.EPS_NOISE_SD, size = d_eps_week.shape).astype(np.float32)

            q_repeats = self.REPEATS_QUARTER.astype(int)
            
            q_offsets = np.cumsum(np.concatenate([[0], q_repeats[:-1]])).astype(int)


            def _debias_per_quarter(
                arr,
                noise
            ):
            
                out = arr.copy()
            
                for qi, qlen in enumerate(q_repeats):
            
                    sl = slice(q_offsets[qi], q_offsets[qi] + qlen)

                    m = noise[:, :, sl].mean(axis = 2, keepdims = True)

                    out[:, :, sl] = out[:, :, sl] + (noise[:, :, sl] - m)

                return out


            d_rev_week = _debias_per_quarter(
                arr = d_rev_week, 
                noise = rev_noise
            )
           
            d_eps_week = _debias_per_quarter(
                arr = d_eps_week, 
                noise = eps_noise
            )
            
            deltas_future_all = np.empty((n_scn, S, HOR, n_reg), dtype = np.float32)

            for m_idx, m_name in enumerate(self.MACRO_REGRESSORS):
              
                j = reg_pos[m_name]
              
                deltas_future_all[:, :, :, j] = np.broadcast_to(
                    macro_future_deltas[:, :, m_idx], (n_scn, S, HOR)
                )

            for f_idx, f_name in enumerate(self.FACTORS):
             
                j = reg_pos[f_name]
             
                deltas_future_all[:, :, :, j] = np.broadcast_to(
                    factor_future[:, :, f_idx], (n_scn, S, HOR)
                )

            if "Revenue" in reg_pos:
             
                j = reg_pos["Revenue"]
             
                deltas_future_all[:, :, :, j] = d_rev_week

            if "EPS (Basic)" in reg_pos:
             
                j = reg_pos["EPS (Basic)"]
             
                deltas_future_all[:, :, :, j] = d_eps_week

          
            for mom in self.MOMENT_COLS:
             
                if mom in reg_pos:
             
                    j = reg_pos[mom]

                    mom_deltas_lastH = DEL_full[-HOR:, j]               
             
                    deltas_future_all[:, :, :, j] = np.broadcast_to(
                        mom_deltas_lastH[None, None, :], (n_scn, S, HOR)
                    )

            deltas_future_all = self.transform_deltas(
                DEL = deltas_future_all.reshape(-1, n_reg),
                sc = sc_reg_full, 
                q_low = ql_full,
                q_high = qh_full
            ).reshape(n_scn, S, HOR, n_reg)
                    
            mc = self.MC_DROPOUT_SAMPLES

            hist_template = X_hist                         
            
            zero_future_ret = np.zeros((n_scn, 1, HOR, 1), np.float32)

            cols_total = mc * S

            samples_buf = np.empty((n_scn, cols_total), dtype = np.float32)
            
            samples_buf_q = np.empty((n_scn, cols_total), dtype = np.float32)

            fill = 0
            
            fill_q = 0

            sim_chunk = 256
            
            for s0 in range(0, S, sim_chunk):
            
                s1 = min(s0 + sim_chunk, S)
            
                w = s1 - s0  

                hist_chunk = np.broadcast_to(hist_template, (n_scn, w, HIST, n_ch))

                future_chunk = np.concatenate(
                    (np.broadcast_to(zero_future_ret, (n_scn, w, HOR, 1)),
                    deltas_future_all[:, s0:s1, :, :]),
                    axis = 3
                )
                
                X_block = np.concatenate((hist_chunk, future_chunk), axis = 2) 
               
                X_block = X_block.reshape(-1, SEQ_LEN, n_ch)               
               
                X_block_tf = tf.convert_to_tensor(X_block)

                mu_stack = np.empty((mc, X_block.shape[0], 1), dtype = np.float32)
               
                sigma_stack = np.empty_like(mu_stack)
               
                nu_stack = np.empty((mc, X_block.shape[0]), dtype = np.float32)

                for mci in range(mc):
                 
                    outs = fns["fwd_train"](X_block_tf)
                 
                    d_params = outs["dist_head"] if isinstance(outs, dict) else outs[1]
                 
                    dp = d_params.numpy().astype(np.float32, copy = False)

                    mu_stack[mci] = self.MU_MAX * np.tanh(dp[:, 0:1])
                 
                    sigma_raw = np.log1p(np.exp(dp[:, 1:2])) + self.SIGMA_FLOOR
                 
                    sigma_stack[mci] = np.minimum(self.SIGMA_MAX, sigma_raw * getattr(self, "_SIGMA_CAL_SCALE", 1.0))
                 
                    nu_stack[mci] = (self.NU_FLOOR + np.log1p(np.exp(dp[:, 2:3]))).ravel()

                t_draws = rng.standard_t(df = nu_stack).astype(np.float32)
                
                t_draws_q = self._recalibrate_residual_draws(t_draws, nu_stack)

                samples_resid   = mu_stack + sigma_stack * t_draws[..., None]
                samples_resid_q = mu_stack + sigma_stack * t_draws_q[..., None]

                samples_total = samples_resid + mu_base_now            
                
                samples_total_q = samples_resid_q + mu_base_now

                block = samples_total.reshape(mc, n_scn, w).transpose(1, 0, 2).reshape(n_scn, mc * w)
                
                block_q = samples_total_q.reshape(mc, n_scn, w).transpose(1, 0, 2).reshape(n_scn, mc * w)

                samples_buf[:, fill:fill + mc * w] = np.expm1(block)
                
                samples_buf_q[:, fill:fill + mc * w] = np.expm1(block_q)    
                
                mc_w = mc * w

                fill += mc_w
                
                fill_q += mc_w

            rets = samples_buf.reshape(-1)
            
            rets_q = samples_buf_q.reshape(-1)

            alpha_adj = getattr(self, "_alpha_levels_adj", np.array([0.10, 0.50, 0.90], dtype=np.float32))

            p10_raw, p50_raw, p90_raw = np.quantile(rets, alpha_adj)

            p10_q, p50_q, p90_q = np.quantile(rets_q, alpha_adj)

            ret_exp = float(np.mean(rets))
            
            se_ret = float(np.std(rets, ddof = 0))
            
            ret_exp_q = float(np.mean(rets_q))
            
            sc_ret_q = float(np.std(rets_q, ddof = 0))

            p_lower_raw = float((p10_raw + 1.0) * cur_p)
            
            p_median_raw = float((p50_raw + 1.0) * cur_p)
            
            p_upper_raw = float((p90_raw + 1.0) * cur_p)

            p_lower_raw_q = float((p10_q + 1.0) * cur_p)
            
            p_median_raw_q = float((p50_q + 1.0) * cur_p)
            
            p_upper_raw_q = float((p90_q + 1.0) * cur_p)

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
                        N = len(X_all),
                        hist = HIST,
                        hor = HOR,
                        n_reg = n_reg,
                    )
                    
                except Exception as ex:
                  
                    self.logger.warning("%s: artifact save failed: %s", ticker, ex)
           
            gru_dict = {
                "Ticker": ticker,
                "status": "ok",
                "Min Price": float(p_lower_raw),
                "Avg Price": float(p_median_raw),
                "Max Price": float(p_upper_raw),
                "Returns": ret_exp,
                "SE": se_ret,
            }
            
            gru_q_dict = {
                "Ticker": ticker,
                "status": "ok",
                "Min Price": float(p_lower_raw_q),
                "Avg Price": float(p_median_raw_q),
                "Max Price": float(p_upper_raw_q),
                "Returns": ret_exp_q,
                "SE": sc_ret_q,
            }
            
            return gru_dict, gru_q_dict

        except Exception as e:
           
            self.logger.error("Error processing %s: %s", ticker, e)
           
            gc.collect()
           
            return {
                "Ticker": ticker, 
                "status": "error", 
                "reason": str(e)
            }


    @staticmethod
    def _last_hist_reg_deltas(
        delta_reg: np.ndarray, 
        hist: int, 
        hor: int, 
        seq_length: int
    ) -> np.ndarray:
        """
        Extract the final `hist` rows of regressor deltas from the composite
        sequence used at inference time.

        Parameters
        ----------
        delta_reg : np.ndarray, shape (T, p)
        hist : int
        hor : int
        seq_length : int
            Expected to be hist + hor.

        Returns
        -------
        np.ndarray, shape (hist, p)
            The historical portion of the (hist + hor) regressor delta block.

        Notes
        -----
        Handles several boundary cases to ensure the returned slice has exactly
        `hist` rows even when T is near the minimum required length.
        """
    
        start = max(0, delta_reg.shape[0] - seq_length)
       
        end = max(start, delta_reg.shape[0] - hor)
       
        out = delta_reg[start:end, :]
       
        if out.shape[0] < hist and delta_reg.shape[0] >= seq_length:
       
            out = delta_reg[-(seq_length):-hor, :]
       
        if out.shape[0] != hist:
       
            out = delta_reg[-hist:, :]
       
        return out.astype(np.float32, copy = False)


    def run(self):
        """
        End-to-end orchestration: build state, launch a process pool, collect per-ticker
        results, compile summary DataFrames, and optionally write Excel outputs.

        Workflow
        --------
        1) Build immutable `state_pack` (data alignment, scenarios).
        
        2) Launch a `ProcessPoolExecutor` with `spawn` context for TensorFlow
        compatibility; pass `state_pack` to each worker via `_init_worker`.
        
        3) Submit `_worker_forecast_one(t)` for all tickers and collect results
        as they complete with per-ticker logging.
        
        4) Partition outputs into:
        
        - ok_rows_raw : list of raw forecast dictionaries,
        
        - ok_rows_cal : list of calibrated forecast dictionaries,
        
        - bad_rows    : list of error/skip dictionaries.
        
        5) Build:
        
        - `df_ok_raw` : ["Min Price","Avg Price","Max Price","Returns","SE"],
        
        - `df_ok_cal` : same as above (calibrated),
        
        - `df_bad`    : ["status","reason"].
        
        6) If `SAVE_TO_EXCEL` is True, write sheets "GRU_raw", "GRU_cal",
        and "GRU_skips" (replacing if they exist).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (df_ok_raw, df_ok_cal, df_bad).

        Notes
        -----
        - Timeouts and worker exceptions are logged and do not abort the run.
        - Process-local singletons reduce TensorFlow start-up overhead per task.
        """

        faulthandler.enable()

        state_pack = self.build_state()

        tickers: List[str] = state_pack["tickers"]

        ctx = mp.get_context("spawn")

        max_workers = min(4, _os.cpu_count())

        self.logger.info("Starting pool with %d workers …", max_workers)

        ok_rows_raw: List[Dict[str, Any]] = []
       
        ok_rows_cal: List[Dict[str, Any]] = []
       
        bad_rows: List[Dict[str, Any]] = []

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
      
                    if isinstance(res, tuple) and len(res) == 2:
      
                        raw, cal = res
                  
                        ok_rows_raw.append(raw)
                   
                        ok_rows_cal.append(cal)
                   
                        self.logger.info(
                            "Ticker %s [raw]: Min %.2f, Avg %.2f, Max %.2f, Return %.4f, SE %.4f",
                            raw["Ticker"], raw["Min Price"], raw["Avg Price"], raw["Max Price"], raw["Returns"], raw["SE"]
                        )
                   
                        self.logger.info(
                            "Ticker %s [cal]: Min %.2f, Avg %.2f, Max %.2f, Return %.4f, SE %.4f",
                            cal["Ticker"], cal["Min Price"], cal["Avg Price"], cal["Max Price"], cal["Returns"], cal["SE"]
                        )
                   
                    elif isinstance(res, dict) and res.get("status") == "ok":
                   
                        ok_rows_raw.append(res)  
                       
                        self.logger.info(
                            "Ticker %s [raw]: Min %.2f, Avg %.2f, Max %.2f, Return %.4f, SE %.4f",
                            res["Ticker"], res["Min Price"], res["Avg Price"], res["Max Price"], res["Returns"], res["SE"]
                        )
                  
                    else:
                  
                        bad_rows.append(res if isinstance(res, dict) else {"status":"error","reason":"bad result type"})
                  
                        self.logger.info("Ticker %s: %s (%s)",
                                        res.get("Ticker") if isinstance(res, dict) else "?", 
                                        res.get("status") if isinstance(res, dict) else "error",
                                        res.get("reason") if isinstance(res, dict) else "bad result type")
                
                except TimeoutError:
                
                    self.logger.error("A worker timed out; continuing.")
                
                except Exception as ex:
                
                    self.logger.error("Worker failed: %s", ex)

        if ok_rows_raw:
        
            df_ok_raw = (pd.DataFrame(ok_rows_raw).set_index("Ticker")[["Min Price", "Avg Price", "Max Price", "Returns", "SE"]])
        
        else:
       
            df_ok_raw = pd.DataFrame(columns=["Min Price", "Avg Price", "Max Price", "Returns", "SE"])

        if ok_rows_cal:

            df_ok_cal = (pd.DataFrame(ok_rows_cal).set_index("Ticker")[["Min Price", "Avg Price", "Max Price", "Returns", "SE"]])

        else:

            df_ok_cal = pd.DataFrame(columns=["Min Price", "Avg Price", "Max Price", "Returns", "SE"])

        if bad_rows:
     
            df_bad = pd.DataFrame(bad_rows).set_index("Ticker")[["status", "reason"]]
     
        else:
     
            df_bad = pd.DataFrame(columns=["status", "reason"])

        if SAVE_TO_EXCEL:
          
            try:
          
                file_exists = _os.path.exists(self.EXCEL_PATH)
          
                if file_exists:
          
                    with pd.ExcelWriter(self.EXCEL_PATH, mode = "a", engine = "openpyxl", if_sheet_exists = "replace") as writer:
          
                        df_ok_raw.to_excel(writer, sheet_name = "GRU_raw")
          
                        df_ok_cal.to_excel(writer, sheet_name = "GRU_cal")
          
                        df_bad.to_excel(writer, sheet_name = "GRU_skips")
             
                else:
             
                    with pd.ExcelWriter(self.EXCEL_PATH, engine = "openpyxl") as writer:
             
                        df_ok_raw.to_excel(writer, sheet_name = "GRU_raw")
             
                        df_ok_cal.to_excel(writer, sheet_name = "GRU_cal")
             
                        df_bad.to_excel(writer, sheet_name = "GRU_skips")
             
                self.logger.info("Saved results to %s", self.EXCEL_PATH)
            
            except Exception as ex:
            
                self.logger.error("Failed to write Excel: %s", ex)

        self.logger.info("Forecasting complete. ok_raw=%d, ok_cal=%d, skipped/error=%d",
                        len(df_ok_raw), len(df_ok_cal), len(df_bad))
        
        return df_ok_raw, df_ok_cal, df_bad


if __name__ == "__main__":
    """
    Profiled script entrypoint.

    - Ensures the 'spawn' start method (required for TensorFlow in many
    environments).
    - Profiles the end-to-end `run()` call and prints the top 20 functions by
    cumulative time.
    - Returns three DataFrames: raw forecasts, calibrated forecasts, and
    skip/error diagnostics.

    Rationale
    ---------
    Explicit profiling aids performance tuning of data preparation and model
    loops; using 'spawn' avoids inherited state conflicts that the 'fork'
    mode may introduce.
    """
  
    try:
  
        mp.set_start_method("spawn", force = True)
  
    except RuntimeError:
  
        pass

    profiler = cProfile.Profile()

    profiler.enable()

    try:

        forecaster = GRU_Forecaster(
            tickers = config.tickers
        )

        df_ok_raw, df_ok_cal, df_bad = forecaster.run()

    finally:

        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats("cumtime")

        stats.print_stats(20)
