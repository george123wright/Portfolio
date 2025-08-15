from __future__ import annotations

import os as _os

_os.environ.setdefault("PYTHONHASHSEED", "42")
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import cProfile, faulthandler, gc, logging, psutil, pstats, random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas.api.types import is_period_dtype, is_datetime64_any_dtype

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.api import VAR

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from sklearn.covariance import LedoitWolf

from dataclasses import dataclass

import config
from data_processing.financial_forecast_data import FinancialForecastData

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
   
    global STATE
   
    STATE = state_pack


def _worker_forecast_one(
    ticker: str
):
  
    global FORECASTER_SINGLETON
  
    if FORECASTER_SINGLETON is None:
  
        FORECASTER_SINGLETON = DirectHForecaster()
  
    return FORECASTER_SINGLETON.forecast_one(ticker)


REV_KEYS = ["low_rev_y", "avg_rev_y", "high_rev_y", "low_rev", "avg_rev", "high_rev"]

EPS_KEYS = ["low_eps_y", "avg_eps_y", "high_eps_y", "low_eps", "avg_eps", "high_eps"]

SCENARIOS = [(r, e) for r in REV_KEYS for e in EPS_KEYS]


@dataclass(frozen = True)
class ModelHP:
   
    hist_window: int = 52
   
    horizon: int = 52
   
    lstm1: int = 96
   
    lstm2: int = 64
   
    l2_lambda: float = 1e-4
   
    dropout: float = 0.15


@dataclass(frozen = True)
class TrainHP:
   
    batch: int = 64
   
    epochs: int = 30
   
    patience: int = 5
   
    lr: float = 5e-4
   
    small_floor: float = 1e-6


@dataclass(frozen = True)
class DistHP:
   
    sigma_floor: float = 1e-3
   
    sigma_max: float = 0.60
   
    nu_floor: float = 8.0
   
    lambda_sigma: float = 5e-4
   
    lambda_invnu: float = 5e-4


@dataclass(frozen = True)
class ScenarioHP:
   
    n_sims: int = 100
   
    mc_dropout_samples: int = 20
   
    alpha_conf: float = 0.10
   
    rev_noise_sd: float = 0.005
   
    eps_noise_sd: float = 0.010
   
    bootstrap_p: float = 1/6


@dataclass(frozen = True)
class HP:
   
    model: ModelHP = ModelHP()
   
    train: TrainHP = TrainHP()
   
    dist: DistHP = DistHP()
   
    scen: ScenarioHP = ScenarioHP()


class DirectHForecaster:
    """
    Direct-H (sum of H weekly log-returns) LSTM forecaster with:
      - train-only scaling/quantiles
      - macro scenarios (country VAR(1) closed-form, with drift) precomputed once
      - factor scenarios (stationary bootstrap) precomputed once
      - financial scenario engine from next_period_forecast()
      - additional moment regressors (skew/kurt) lagged 52w, rolling 104w
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
  
    EPOCHS = 30
  
    PATIENCE = 5
  
    SMALL_FLOOR = 1e-6
  
    L2_LAMBDA = 1e-4

    N_SIMS = 100                 
  
    MC_DROPOUT_SAMPLES = 20    
  
    ALPHA_CONF = 0.10

    SIGMA_FLOOR = 1e-3
    
    SIGMA_MAX = 0.60         
   
    MU_MAX = 0.80            
    
    REV_NOISE_SD = 0.005   
   
    EPS_NOISE_SD = 0.010   
      
    LAMBDA_SIGMA = 5e-4     
   
    LAMBDA_INVNU = 5e-4     
    
    NU_FLOOR = 8.0

    MACRO_REGRESSORS = ["Interest", "Cpi", "Gdp", "Unemp"]
  
    FACTORS = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]
  
    FIN_REGRESSORS = ["Revenue", "EPS (Basic)"]
  
    MOMENT_COLS = ["skew_104w_lag52", "kurt_104w_lag52"]
  
    NON_FIN_REGRESSORS = MACRO_REGRESSORS + FACTORS
  
    ALL_REGRESSORS = NON_FIN_REGRESSORS + FIN_REGRESSORS + MOMENT_COLS

    _w = HORIZON // 4
  
    _rem = HORIZON - 4 * _w
  
    REPEATS_QUARTER = np.array([_w, _w, _w, _w + _rem], dtype=int)

    _MODEL_CACHE: Dict[int, Tuple[Any, List[np.ndarray]]] = {}
    
    _EYE_CACHE = {4: (np.eye(4, dtype=np.float32) * 1e-4)} 
    
    _FN_CACHE: Dict[int, Dict[str, Any]] = {}

    def __init__(
        self, 
        tickers: Optional[List[str]] = None,
        hp: Optional["DirectHForecaster.HP"] = None
    ):
       
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

        
    def _configure_logger(
        self
    ) -> logging.Logger:
        
        logger = logging.getLogger("lstm_directH_class")
        
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
    
        N = len(X)
    
        return X.shape == (N, seq_length, 1 + n_reg)


    def choose_splits(
        self,
        N: int, 
        min_fold: Optional[int] = None
    ) -> int:
        
        if min_fold is None:
            
            min_fold = 2 * self.BATCH
    
        if N < min_fold :
    
            return 0
    
        if N >= 2 * min_fold:
    
            return 2
    
        return 1


    def build_delta_matrix(
        self, 
        reg_mat: np.ndarray, 
        regs: List[str]
    ) -> np.ndarray:

        reg_mat = np.asarray(reg_mat, dtype = np.float32)
        
        if reg_mat.ndim != 2:
        
            raise ValueError(f"reg_mat must be 2D, got {reg_mat.ndim}D with shape {reg_mat.shape}")
        
        T, n_reg = reg_mat.shape
        
        if n_reg != len(regs):
        
            raise ValueError(f"reg_mat has {n_reg} columns but regs has {len(regs)}")

        DEL = np.empty((T - 1, n_reg), dtype=np.float32)
        
        for j, name in enumerate(regs):
        
            col = reg_mat[:, j]
        
            if name in ("Interest", "Unemp") or (name in self.MOMENT_COLS):
        
                d = col[1:] - col[:-1]
        
            elif name in ("Cpi", "Gdp", "Revenue", "EPS (Basic)"):
        
                col_safe = np.maximum(col, self.SMALL_FLOOR)
        
                d = np.log(col_safe[1:]) - np.log(col_safe[:-1])
        
            elif name in self.FACTORS:
        
                d = col[1:]
        
            else:
        
                d = col[1:] - col[:-1]
        
            DEL[:, j] = d
        
        return DEL


    @staticmethod
    def fit_scale_deltas(
        DEL_tr: np.ndarray
    ):
    
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
        Estimate r_t = m + phi (r_{t-1} - m) + eps via OLS on training returns.
        Returns (m, phi).
        """
       
        r = np.asarray(log_ret, float).ravel()
       
        if len(r) < 3:
       
            return 0.0, 0.0
       
        r0 = r[:-1]
       
        r1 = r[1:]
       
        X = np.column_stack([np.ones_like(r0), r0])
       
        beta, *_ = np.linalg.lstsq(X, r1, rcond=None)
       
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
        E[sum_{k=1..H} r_{t+k} | r_t] under AR(1) with intercept m.
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
    
        DEL_clip = np.clip(DEL, q_low, q_high)
    
        DEL_scaled = (DEL_clip - sc.center_) / sc.scale_
    
        return DEL_scaled.astype(np.float32)
    
    
    @staticmethod
    def ensure_spd_for_cholesky(
        Sigma: np.ndarray,
        min_eig: float = 1e-8,
        shrink: float = 0.05,
        max_tries: int = 6
    ) -> np.ndarray:
        """
        Return a symmetric positive-definite covariance suitable for Cholesky.
        Steps: symmetrize -> shrink to identity -> clip eigenvalues -> escalate
        diagonal loading until np.linalg.cholesky succeeds. Keeps float64 until end.
        """

        S = np.asarray(Sigma, dtype = np.float64)

        if not np.isfinite(S).all():
          
            S = np.where(np.isfinite(S), S, 0.0)

        S = 0.5 * (S + S.T)

        p = S.shape[0]
       
        I = np.eye(p, dtype = np.float64)

        if p > 0:
            
            tau = (np.trace(S) / p)  
        
        else:
            
            tau = 1.0
        
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
        If enough data, return LedoitWolf covariance.
        Otherwise return a small, cached eye(dim) * 1e-4 (do NOT mutate).
        """
    
        if tail_vals is not None and tail_vals.shape[0] > 10:
    
            cov = LedoitWolf().fit(tail_vals).covariance_.astype(np.float32, copy = False)
    
            return cov
     
        if dim in DirectHForecaster._EYE_CACHE:
     
            return DirectHForecaster._EYE_CACHE[dim]

        DirectHForecaster._EYE_CACHE[dim] = np.eye(dim, dtype=np.float32) * 1e-4
     
        return DirectHForecaster._EYE_CACHE[dim]
    
    
    @staticmethod
    def make_windows_directH(
        scaled_ret, 
        scaled_reg, 
        hist, 
        hor,
        seq_length
    ):
       
        T_ret = scaled_ret.shape[0]
                
        n_reg = scaled_reg.shape[1]

        n_windows = T_ret - seq_length + 1
        
        if n_windows <= 0:
        
            return (np.zeros((0, seq_length, 1 + n_reg), np.float32),
                    np.zeros((0, 1), np.float32))

        i0 = np.arange(n_windows, dtype = np.int64)   

        pr_full = sliding_window_view(
            x = scaled_ret, 
            window_shape = hist
        )                  
        
        pr = pr_full[i0]                                                        

        pR_full = sliding_window_view(
            x = scaled_reg, 
            window_shape = (hist, n_reg)
        )[:, 0]   
        
        pR = pR_full[i0]                                                     

        reg_future_full = sliding_window_view(
            x = scaled_reg, 
            window_shape = (hor, n_reg)
        )[:, 0]  
        
        fR = reg_future_full[(hist - 1) + i0]   
        
        ret_ch = np.empty((n_windows, seq_length, 1), dtype=np.float32)
        
        ret_ch[:, :hist, 0] = pr
        
        ret_ch[:, hist:, 0] = 0.0

        reg_ch = np.concatenate([pR, fR], axis = 1)     
        
        X = np.concatenate([ret_ch, reg_ch], axis = 2)  
        
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
        y: sum of future H log-returns for each window, shape (N,1).
        If N_expected is provided, y is truncated to that length so it
        always matches X built from make_windows_directH.
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
    
        vals = np.linalg.eigvals(A)
    
        rho = float(np.max(np.abs(vals)))
    
        if rho >= 1.0:
    
            A = A / (rho + 1e-3)
    
        return A.astype(np.float32)


    def fit_var1_with_intercept(
        self, 
        dfm_stationary: pd.DataFrame
    ):
    
        if dfm_stationary.shape[0] <= dfm_stationary.shape[1] + 4:
    
            return None
    
        vr = VAR(dfm_stationary).fit(maxlags=1, trend="c")
    
        if vr.k_ar < 1:
    
            return None
    
        A = vr.coefs[0].astype(np.float32)
    
        c = vr.intercept_.astype(np.float32)
    
        Sigma = np.asarray(vr.sigma_u, dtype=np.float32) 
    
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
       
        sums = np.cumsum(P[:-1], axis=0)                                   
       
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
        Return indices of shape (n_sims, H) for Politis-Romano stationary bootstrap.
        R[:, t] ~ Bernoulli(p) indicates a restart at time t (with R[:,0]=True).
        At a restart, start ~ Uniform{0,..,L-1}; else continue previous index + 1 mod L.
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
    
        sc = RobustScaler().fit(log_ret[1:].reshape(-1, 1))
    
        sc.scale_[sc.scale_ < 1e-6] = 1e-6
    
        return sc


    @staticmethod
    def scale_logret_with(
        log_ret: np.ndarray, 
        sc: RobustScaler
    ) -> np.ndarray:
    
        return np.concatenate([[0.0], ((log_ret[1:] - sc.center_) / sc.scale_).ravel()]).astype(np.float32)


    def build_directH_model(
        self, 
        n_reg: int, 
        seed: int = SEED
    ):
   
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
        
        z = Dense(3, name="q_raw")(x)
        
        def monotone_quantiles(
            z
        ):
            a = z[:, 0:1]               
           
            b_gap = tf.nn.softplus(z[:, 1:2]) + 1e-6  
           
            c_gap = tf.nn.softplus(z[:, 2:3]) + 1e-6  
           
            q10 = a
           
            q50 = a + b_gap
           
            q90 = a + b_gap + c_gap
           
            return tf.concat([q10, q50, q90], axis = -1)


        q_out = Lambda(monotone_quantiles, name="q_head")(z)
        
        d_params = Dense(
            3, 
            name = "dist_head"
        )(x) 

        return Model(
            inp, 
            [q_out, d_params]
        )
        
        
    def _make_callbacks(
        self,
        monitor: str
    ):
    
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
    def pinball_loss(
        y_true, 
        y_pred,
        qs = (0.1, 0.5, 0.9)):
        
        y_true = tf.cast(y_true[:, 0:1], tf.float32)
        
        losses = []
        
        for i, q in enumerate(qs):
        
            e = y_true - y_pred[:, i:i+1]
        
            losses.append(tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e)))
        
        return tf.add_n(losses) / float(len(qs))


    def student_t_nll(
        self,
        y_true_res, 
        params
    ):
        
        y = tf.cast(y_true_res, tf.float32)
        
        mu_raw = tf.cast(params[:, 0:1], tf.float32)   
        
        mu_resid = self.MU_MAX * tf.tanh(mu_raw)        
        
        log_sigma = tf.cast(params[:, 1:2], tf.float32)
        
        df_raw = tf.cast(params[:, 2:3], tf.float32)

        sigma = tf.nn.softplus(log_sigma) + self.SIGMA_FLOOR
       
        sigma = tf.minimum(sigma * getattr(self, "_SIGMA_CAL_SCALE", 1.0), self.SIGMA_MAX)
       
        nu = self.NU_FLOOR + tf.nn.softplus(df_raw)

        z = (y - mu_resid) / sigma
       
        logC = (tf.math.lgamma((nu + 1.0) / 2.0) - tf.math.lgamma(nu / 2.0)
                - 0.5 * tf.math.log((nu - 2.0) * np.pi) - tf.math.log(sigma))
      
        log_pdf = logC - 0.5 * (nu + 1.0) * tf.math.log(1.0 + (z * z)/(nu - 2.0))
      
        nll = -tf.reduce_mean(log_pdf)
      
        reg = self.LAMBDA_SIGMA * tf.reduce_mean(tf.square(sigma)) + self.LAMBDA_INVNU * tf.reduce_mean(1.0 / (nu - 2.0))
       
        return nll + reg

    
    def build_state(
        self
    ) -> Dict[str, Any]:
    
        self.logger.info("Building global state …")
    
        if self.SAVE_ARTIFACTS and not _os.path.exists(self.ARTIFACT_DIR):
    
            _os.makedirs(self.ARTIFACT_DIR, exist_ok = True)

        fdata = FinancialForecastData()
      
        macro = fdata.macro
    
        r = macro.r

        if self.tickers_arg:
    
            tickers = list(self.tickers_arg)
    
        else:
    
            tickers = ['NVDA', 'GOOG', 'META', 'TTWO', 'AMZN', 'MSFT', 'IAG.L', '1211.HK']

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
  
            macro_rec[reg] = macro_clean[reg].to_numpy(dtype=np.float32, copy=False)
  
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

        for ctry, (s, e) in country_slices.items():
      
            rec = macro_rec[s:e]
      
            if len(rec) == 0:
      
                country_var_results[ctry] = None
      
                macro_future_by_country[ctry] = None
      
                continue
      
            dfm = pd.DataFrame({reg: rec[reg] for reg in self.MACRO_REGRESSORS},
                               index = pd.DatetimeIndex(rec["ds"]))
         
            dfm = (dfm[~dfm.index.duplicated(keep = "first")]
                   .sort_index()
                   .resample("W").mean()
                   .ffill()
                   .dropna())
         
            if dfm.shape[1] == 0 or len(dfm) <= dfm.shape[1] + 4:
         
                country_var_results[ctry] = None
         
                dfm_stat, _ = self._to_stationary_macro(
                    dfm = dfm
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
                    size = (self.N_SIMS, hor),
                    method = "cholesky"
                ).astype(np.float32)
                
                macro_future_by_country[ctry] = eps  
              
                continue

            dfm_stat, _ = self._to_stationary_macro(
                dfm = dfm
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
                    n_sims = self.N_SIMS, 
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
                    size = (self.N_SIMS, hor), 
                    method = "cholesky"
                ).astype(np.float32)
                
                macro_future_by_country[ctry] = eps
        
        macro_weekly_by_country = {}
       
        macro_weekly_idx_by_country = {}
       
        for ctry, (s, e) in country_slices.items():
       
            rec = macro_rec[s:e]
       
            if len(rec) == 0:
       
                macro_weekly_by_country[ctry] = None
       
                macro_weekly_idx_by_country[ctry] = None
       
                continue
       
            dfm = pd.DataFrame({reg: rec[reg] for reg in self.MACRO_REGRESSORS},
                            index = pd.DatetimeIndex(rec["ds"]))
         
            dfw = (dfm[~dfm.index.duplicated(keep="first")]
                   .sort_index()
                   .resample("W-FRI").mean()  
                   .ffill()
                   .dropna())
          
            macro_weekly_by_country[ctry] = dfw    
           
            macro_weekly_idx_by_country[ctry] = dfw.index.values

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
          
            lr = np.zeros_like(ys, dtype=np.float32)
           
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
       
                idx_fa = np.searchsorted(factor_weekly_index, dates, side="right") - 1
       
                valid_fa = idx_fa >= 0
       
            else:
       
                idx_fa = np.zeros_like(idx_m)
       
                valid_fa = np.ones_like(idx_m, dtype=bool)
       
            if t in fd_weekly_by_ticker:
                
                fd_idx = fd_weekly_by_ticker[t]["index"]
                
                idx_fd = np.searchsorted(fd_idx, dates, side="right") - 1
                
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
            "price_rec": price_rec,
            "macro_rec": macro_rec,
            "country_slices": country_slices,
            "country_var_results": country_var_results,
            "macro_future_by_country": macro_future_by_country,  
            "fd_rec_dict": fd_rec_dict,
            "next_fc": fdata.next_period_forecast(),
            "latest_price": r.last_price,
            "analyst": analyst,
            "ticker_country": analyst["country"],
            "factor_weekly": fac_w,
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
      
        state_pack["scaler_cache"] = {}  
        
        self.logger.info("Global state built (%d tickers).", len(grouped_tickers))
        
        return state_pack


    def forecast_one(
        self, 
        ticker: str
    ) -> Dict[str, Any]:
    
        assert STATE is not None, "Worker STATE not initialized"
        
        try:

            tf.config.experimental.enable_op_determinism()

        except Exception:

            pass

        tf.random.set_seed(self.SEED)

        tf.config.threading.set_intra_op_parallelism_threads(1)

        tf.config.threading.set_inter_op_parallelism_threads(1)


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
            
            flags = STATE["presence_flags"][ticker]

            price_rec = STATE["price_rec"]

            macro_rec = STATE["macro_rec"]

            country_slices = STATE["country_slices"]

            fd_rec_dict = STATE["fd_rec_dict"]

            next_fc = STATE["next_fc"]

            latest_price = STATE["latest_price"]

            analyst = STATE["analyst"]

            ticker_country = STATE["ticker_country"]

            fa_vals = STATE["factor_weekly_values"]

            macro_future_by_country = STATE["macro_future_by_country"]

            factor_future_global = STATE["factor_future_global"]

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

            cur_p = latest_price.get(ticker, np.nan)

            if not np.isfinite(cur_p):

                return skip(
                    reason = "no_latest_price"
                )

            mask_t = price_rec["Ticker"] == ticker

            wp = STATE["weekly_price_by_ticker"].get(ticker)
          
            if wp is None:
          
                return skip(
                    reason = "no_price_history_weekly"
                )
          
            dates = wp["index"]
          
            yv = wp["y"]
          
            lr = wp["lr"]
          
            macro_ct = macro_rec[s:e]
            
            fdw = STATE["fd_weekly_by_ticker"].get(ticker, None)

            if len(macro_ct) < 12:
               
                return skip(
                    reason = "insufficient_macro_history"
                )
            
            SEQ_LEN = self.SEQUENCE_LENGTH
           
            if len(yv) < SEQ_LEN + 2:
                
                return skip(
                    reason = "short_price_history"
                )
                
            small_floor = self.SMALL_FLOOR
            
            ys = np.where(yv > small_floor, yv, small_floor)
           
            ly = np.log(ys)
           
            lr = np.empty_like(ly)
           
            lr[0] = 0.0
           
            lr[1:] = ly[1:] - ly[:-1]
                
            len_idx_keep = len(idx_keep)

            if len_idx_keep < SEQ_LEN + 1:
                
                return skip(
                    reason = "insufficient_joint_features"
                )

            sel_m = np.clip(sel_m, 0, len(dfm_ct) - 1)


            if fa_vals.shape[0] == 0:

                fac_block = np.zeros((len_idx_keep, len(self.FACTORS)), dtype=np.float32)
         
            else:
         
                sel_fa = np.clip(sel_fa, 0, fa_vals.shape[0] - 1)
         
                fac_block = fa_vals[sel_fa, :]  
         
           
            if ticker in fd_rec_dict:

                fd_rec = fd_rec_dict[ticker]

                sel_fd = np.clip(sel_fd, 0, len(fdw["index"]) - 1)
                
            else:
                
                fd_rec = None
                
                sel_fd = None
                
            n_m = len(dfm_ct)
            
            n_fa = fa_vals.shape[0]
            
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

            len_idx_keep = len(idx_keep)
          
            if len_idx_keep < SEQ_LEN + 1:
          
                return skip(
                    reason = "insufficient_joint_features_after_upperbound"
                )

            regs = list(self.ALL_REGRESSORS)
        
            n_reg = len(regs)
            
            sig = tuple(regs)  
           
            if sig not in self._FN_CACHE:
           
                self._FN_CACHE[sig] = {}
                
            if "reg_pos" not in self._FN_CACHE[sig]:
            
                self._FN_CACHE[sig]["reg_pos"] = {name: i for i, name in enumerate(regs)}
                
            reg_pos = self._FN_CACHE[sig]["reg_pos"]

            n_ch = 1 + n_reg
            
            log_ret_full = lr[idx_keep].astype(np.float32)

            reg_list = []
        
            reg_list.append(dfm_ct["Interest"].values[sel_m].astype(np.float32))
        
            reg_list.append(dfm_ct["Cpi"].values[sel_m].astype(np.float32))
        
            reg_list.append(dfm_ct["Gdp"].values[sel_m].astype(np.float32))
        
            reg_list.append(dfm_ct["Unemp"].values[sel_m].astype(np.float32))

            if flags["has_factors"]:

                reg_list.extend([fac_block[:, k].astype(np.float32) for k in range(len(self.FACTORS))])
           
            else:
               
                reg_list.extend([np.zeros(len_idx_keep, np.float32) for _ in self.FACTORS])

            if flags["has_fin"] and fdw is not None and (align["idx_fd"] is not None):
               
                sel_fd = np.clip(sel_fd, 0, len(fdw["index"]) - 1)
               
                reg_list.append(fdw["values"][sel_fd, 0].astype(np.float32))
               
                reg_list.append(fdw["values"][sel_fd, 1].astype(np.float32))
         
            else:
         
                reg_list.append(np.zeros(len_idx_keep, np.float32))
         
                reg_list.append(np.zeros(len_idx_keep, np.float32))

            if flags["has_moms"] and moms_vals is not None:

                reg_list.append(moms_vals[idx_keep, 0].astype(np.float32, copy=False))

                reg_list.append(moms_vals[idx_keep, 1].astype(np.float32, copy=False))

            else:

                reg_list.append(np.zeros(len_idx_keep, np.float32))

                reg_list.append(np.zeros(len_idx_keep, np.float32))

            reg_mat = np.column_stack(reg_list) 
            
            DEL_full = self.build_delta_matrix(
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
                      
                      
            fns = self._FN_CACHE[n_reg]
                                   
            HIST = self.HIST_WINDOW
          
            HOR = self.HORIZON
             
            ret_scaler_full = self.fit_ret_scaler_from_logret(
                log_ret = log_ret_full
            )
          
            scaled_ret_full = self.scale_logret_with(
                log_ret = log_ret_full, 
                sc = ret_scaler_full
            )

            sc_reg_full, ql_full, qh_full = self.fit_scale_deltas(
                DEL_tr = DEL_full
            )
            
            scaled_reg_full = self.transform_deltas(
                DEL = DEL_full, 
                sc = sc_reg_full, 
                q_low = ql_full, 
                q_high = qh_full
            )

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

            scaler_cache = STATE["scaler_cache"]  
          
            ctry_key = (ctry, sig)             
 
            del_len = DEL_full.shape[0]
           
            train_mask = np.zeros(del_len, dtype=bool)

            for i in train_idx:

                a = i

                b = min(i + SEQ_LEN - 1, del_len - 1)

                if a <= b:

                    train_mask[a:b + 1] = True

            if train_mask.sum() < n_reg:
                
                return skip(
                    reason = "too_few_training_deltas_for_scaler"
                )

          
            if ctry_key in scaler_cache:
          
                sc_reg_full, ql_full, qh_full = scaler_cache[ctry_key]
          
            else:
                
                sc_reg_full, ql_full, qh_full = self.fit_scale_deltas(
                    DEL_tr = DEL_full[train_mask]
                )
                
                scaler_cache[ctry_key] = (sc_reg_full, ql_full, qh_full)

            scaled_reg_full = self.transform_deltas(
                DEL = DEL_full, 
                sc = sc_reg_full, 
                q_low = ql_full, 
                q_high = qh_full
            )
          
            ret_len = log_ret_full.shape[0]
          
            ret_mask = np.zeros(ret_len, dtype = bool)
          
            for i in train_idx:
          
                a = max(0, i)                   
          
                b = min(i + HIST - 1, ret_len-1)
          
                if a <= b:
          
                    ret_mask[a:b + 1] = True
            
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
                
                return np.array([
                    self.ar1_sum_forecast(
                        r_t = rt, 
                        m = m_hat, 
                        phi = phi_hat,
                        H = H
                    ) for rt in last_returns], dtype = np.float32)[:, None]


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
            
            last_r_now = float(log_ret_full[-(HOR + 1)])

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
            
            sigma_c = np.minimum(self.SIGMA_MAX, np.log1p(np.exp(params_cal[:, 1:2])) + self.SIGMA_FLOOR)

            resid = (y_cal_res.ravel() - mu_c.ravel())

            pred_sd = sigma_c.ravel()

            num = np.nanmedian(np.abs(resid)) / 0.6745   # MAD -> sigma
            
            den = np.nanmedian(pred_sd)
            
            sigma_scale = np.clip(num / max(den, 1e-6), 0.5, 1.2)

            self._SIGMA_CAL_SCALE = float(sigma_scale)
            
            if len(X_tr) > 4 * self.BATCH:
                
                effective_epochs = self.EPOCHS  
            
            else:
                
                effective_epochs = max(8, self.EPOCHS // 2)

            callbacks = self._make_callbacks(
                monitor = "val_loss"
            )

            model.fit(
                X_tr, 
                {
                    "q_head": y_tr_res, 
                    "dist_head": y_tr_res
                },
                epochs = effective_epochs,
                callbacks = callbacks,
                verbose = 0,
                validation_data = (X_cal, {
                    "q_head": y_cal_res, 
                    "dist_head": y_cal_res
                })
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
                    size = (self.N_SIMS, HOR),
                    method = "cholesky"
                ).astype(np.float32)  
                
            S_all = factor_future_global.shape[0]
            
            sel = rng.choice(S_all, size = self.N_SIMS, replace = (S_all < self.N_SIMS))
           
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

            d_rev_week = np.repeat(shifts_rev.reshape(n_scn, 1, 4), self.N_SIMS, axis = 1)
          
            d_rev_week = np.repeat(d_rev_week, self.REPEATS_QUARTER, axis = 2)  
          
            d_eps_week = np.repeat(shifts_eps.reshape(n_scn, 1, 4), self.N_SIMS, axis = 1)
          
            d_eps_week = np.repeat(d_eps_week, self.REPEATS_QUARTER, axis = 2)

            rev_noise = rng.normal(0.0, self.REV_NOISE_SD, size = d_rev_week.shape).astype(np.float32)
            
            eps_noise = rng.normal(0.0, self.EPS_NOISE_SD, size = d_eps_week.shape).astype(np.float32)
            
            d_rev_week = d_rev_week + rev_noise
            
            d_eps_week = d_eps_week + eps_noise

            S = self.N_SIMS
            
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
            all_samples = []

            hist_template = X_hist
        
            zero_future_ret = np.zeros((n_scn, 1, HOR, 1), dtype = np.float32)
        
            sim_chunk = 256

            for s0 in range(0, S, sim_chunk):
             
                s1 = min(s0 + sim_chunk, S)
             
                w = s1 - s0

                hist_chunk = np.broadcast_to(hist_template, (n_scn, w, HIST, n_ch))
           
                future_chunk = np.concatenate(
                    [np.broadcast_to(zero_future_ret, (n_scn, w, HOR, 1)),
                    deltas_future_all[:, s0:s1, :, :]],
                    axis=3
                )
             
                X_block = np.concatenate([hist_chunk, future_chunk], axis = 2).reshape(-1, SEQ_LEN, n_ch)
             
                X_block_tf = tf.convert_to_tensor(X_block)

                mu_stack = []
              
                sigma_stack = []
              
                nu_stack = []

                for _ in range(mc):
                 
                    outs = fns["fwd_train"](X_block_tf)
                 
                    d_params = outs["dist_head"] if isinstance(outs, dict) else outs[1]
                 
                    dp = d_params.numpy().astype(np.float32, copy = False)

                    mu = self.MU_MAX * np.tanh(dp[:, 0:1])
                    
                    sigma = np.log1p(np.exp(dp[:, 1:2])) + self.SIGMA_FLOOR
                    
                    sigma = np.minimum(self.SIGMA_MAX, sigma * getattr(self, "_SIGMA_CAL_SCALE", 1.0))
                    
                    nu = self.NU_FLOOR + np.log1p(np.exp(dp[:, 2:3]))

                    mu_stack.append(mu)
                 
                    sigma_stack.append(sigma)
                 
                    nu_stack.append(nu.squeeze(-1))  
                    
                mu_stack = np.stack(mu_stack, axis = 0)      
              
                sigma_stack = np.stack(sigma_stack, axis = 0)  
              
                nu_stack = np.stack(nu_stack, axis = 0)     

                t_draws = rng.standard_t(df = nu_stack).astype(np.float32)[..., None] 

                samples_resid = mu_stack + sigma_stack * t_draws

                samples_total = samples_resid + mu_base_now

                all_samples.append(samples_total.reshape(-1, 1))

            logR_flat = np.concatenate(all_samples, axis = 0).ravel() 
                        
            q_lo, q_hi = np.nanquantile(logR_flat, [0.05, 0.95])
            
            mask = (logR_flat >= q_lo) & (logR_flat <= q_hi)
            
            logR_flat = logR_flat[mask]
            
            se = float(np.nanstd(logR_flat, ddof=0))
            
            alpha = float(self.ALPHA_CONF)
            
            y_lower = np.quantile(logR_flat, alpha / 2.0)
            
            y_med = np.quantile(logR_flat, 0.5)
            
            y_upper = np.quantile(logR_flat, 1.0 - alpha / 2.0)

            p_lower_raw = float(cur_p * np.exp(y_lower))
            
            p_median_raw = float(cur_p * np.exp(y_med))
            
            p_upper_raw = float(cur_p * np.exp(y_upper))
            
            se_ret = np.exp(se) - 1 
            
            ret_exp = np.exp(np.nanmean(logR_flat)) - 1.0

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
    
        start = max(0, delta_reg.shape[0] - seq_length)
       
        end = max(start, delta_reg.shape[0] - hor)
       
        out = delta_reg[start:end, :]
       
        if out.shape[0] < hist and delta_reg.shape[0] >= seq_length:
       
            out = delta_reg[-(seq_length):-hor, :]
       
        if out.shape[0] != hist:
       
            out = delta_reg[-hist:, :]
       
        return out.astype(np.float32, copy = False)


    def run(self):

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
  
        mp.set_start_method("spawn", force=True)
  
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
