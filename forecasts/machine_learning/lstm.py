from __future__ import annotations


SAVE_ARTIFACTS = True
ARTIFACT_DIR = config.BASE_DIR / "lstm_artifacts"   
EXCEL_PATH = config.MODEL_FILE
PRED_BLOCK = 20000 


import os as _os


_os.environ.setdefault("PYTHONHASHSEED", "42")      

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1") 

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
):

    _os.environ.setdefault(_v, "1")

import cProfile
import faulthandler
import gc
import logging
import math
import psutil
import pstats
import random
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.api import VAR

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import config
from data_processing.financial_forecast_data import FinancialForecastData

warnings.filterwarnings(
    "ignore",
    message = r"Protobuf gencode version .* is exactly one major version older",
    module = r"google\.protobuf\.runtime_version",
)

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

REV_KEYS = ["low_rev_y", "avg_rev_y", "high_rev_y", "low_rev", "avg_rev", "high_rev"]

EPS_KEYS = ["low_eps_y", "avg_eps_y", "high_eps_y", "low_eps", "avg_eps", "high_eps"]

SCENARIOS = [(r, e) for r in REV_KEYS for e in EPS_KEYS]

TECHNICAL_REGRESSORS = ["MA52_ret"]

MACRO_REGRESSORS = ["Interest", "Cpi", "Gdp", "Unemp"]

FIN_REGRESSORS = ["Revenue", "EPS (Basic)"]

NON_FIN_REGRESSORS = MACRO_REGRESSORS + TECHNICAL_REGRESSORS

ALL_REGRESSORS = NON_FIN_REGRESSORS + FIN_REGRESSORS

HIST_WINDOW = 52

HORIZON = 52

SEQUENCE_LENGTH = HIST_WINDOW + HORIZON

w = HORIZON // 4

rem  = HORIZON - 4 * w

repeats_quarter = np.array([w, w, w, w + rem], dtype = int)

SMALL_FLOOR = 1e-6

L2_LAMBDA = 1e-4

PATIENCE = 5

EPOCHS = 30

BATCH = 64

N_SIMS = 100

STATE: Optional[Dict[str, Any]] = None

_MODEL_CACHE: Dict[int, Tuple[Any, List[np.ndarray]]] = {}   


def configure_logger() -> logging.Logger:

    logger = logging.getLogger("lstm_forecast")

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        h = logging.StreamHandler()

        h.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s"))

        logger.addHandler(h)

    return logger


logger = configure_logger()


def make_scaled_arrays(
    df: pd.DataFrame,
    regs: List[str],
    q_clip: Tuple[float, float] = (0.01, 0.99)
) -> Tuple[np.ndarray, np.ndarray, RobustScaler, RobustScaler, np.ndarray, np.ndarray]:
    """
    Returns:
      scaled_ret: 1D length T standardized log-returns
      scaled_reg: (T-1, n_reg) standardized Δlog(regressor)
      scaler_reg, scaler_ret
      q_low, q_high : 1% and 99% quantiles of Δlog(reg) for robust clipping (#9)
    """
    
    y = df["y"].to_numpy(np.float32)
    
    y_safe = np.maximum(y, SMALL_FLOOR)
    
    log_y = np.log(y_safe)
    
    log_ret = np.concatenate([[0.0], np.diff(log_y)]).astype(np.float32)

    R = df[regs].to_numpy(np.float32)
  
    R_safe  = np.maximum(R, SMALL_FLOOR)
  
    delta_R = np.diff(np.log(R_safe), axis = 0).astype(np.float32)

    scaler_ret = RobustScaler().fit(log_ret[1:].reshape(-1, 1))

    scaler_ret.scale_[scaler_ret.scale_ < SMALL_FLOOR] = SMALL_FLOOR

    scaled_ret = np.concatenate(
        [[0.0], ((log_ret[1:] - scaler_ret.center_) / scaler_ret.scale_).ravel()]
    ).astype(np.float32)

    scaler_reg = RobustScaler().fit(delta_R)

    scaler_reg.scale_[scaler_reg.scale_ < SMALL_FLOOR] = SMALL_FLOOR

    scaled_reg = ((delta_R - scaler_reg.center_) / scaler_reg.scale_).astype(np.float32)

    q_low = np.quantile(delta_R, q_clip[0], axis = 0).astype(np.float32)

    q_high = np.quantile(delta_R, q_clip[1], axis = 0).astype(np.float32)

    return (scaled_ret, scaled_reg, scaler_reg, scaler_ret, q_low, q_high)


def make_windows(
    ret: np.ndarray,
    reg: np.ndarray,
    hist: int,
    hor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build aligned windows (fixes off-by-hor errors).
      X: (N, hist+hor, 1 + n_reg)
      y: (N, hor)
    """
   
    T = len(ret) - 1
   
    N = T - (hist + hor) + 1
   
    if N <= 0:
   
        return (np.zeros((0, hist + hor, reg.shape[1] + 1), np.float32),
                np.zeros((0, hor), np.float32))

    pr_full = sliding_window_view(ret[:-1], window_shape = hist)    
  
    pr = pr_full[:N]                                        

    fr = sliding_window_view(ret[hist:-1], window_shape = hor)      

    pR_full = sliding_window_view(reg, window_shape = (hist, reg.shape[1]))   
   
    pR = pR_full[:, 0, :, :][:N]                                        

    fR_full = sliding_window_view(reg[hist:], window_shape = (hor, reg.shape[1])) 
    
    fR = fR_full[:, 0, :, :]                                                

    X = np.zeros((N, hist+hor, 1 + reg.shape[1]), np.float32)
    
    X[:, :hist, 0] = pr
  
    X[:, :hist, 1:] = pR
  
    X[:, hist:, 1:] = fR
  
    return X, fr.astype(np.float32)


def _shape_ok(
    X_full: np.ndarray,
    y_full: np.ndarray, 
    hist: int, 
    hor: int, 
    n_reg: int
) -> bool:
   
    N = len(X_full)
   
    return (X_full.shape == (N, hist + hor, 1 + n_reg)) and (y_full.shape == (N, hor))


def _no_nans(
    *arrs: np.ndarray
) -> bool:

    return all(np.isfinite(a).all() for a in arrs)


def choose_splits(
    N: int, 
    hist: int, 
    hor: int,
    min_fold: int = 2 * BATCH
) -> int:
    """
    Smarter CV sizing (#3).
    usable = N - hor (reserve gap for last fold). Need each side >= min_fold.
    """
    
    usable = N - hor
    
    if usable < min_fold:
    
        return 0
    
    if usable >= 2 * min_fold:
        
        return 2  
    
    else:
        
        return 1


def predict_in_blocks(
    model, 
    X: np.ndarray, 
    block: int = PRED_BLOCK
) -> np.ndarray:
    """
    Memory-friendly prediction (#4).
    """
    
    out = np.empty((len(X), HORIZON), dtype = np.float32)
    
    for i in range(0, len(X), block):
    
        out[i: i + block] = model.predict(X[i: i + block], batch_size = 256, verbose = 0)
    
    return out


def _init_worker(
    state_pack: Dict[str, Any]
):

    global STATE

    STATE = state_pack


def _stable_A(
    A: np.ndarray
) -> np.ndarray:

    vals = np.linalg.eigvals(A)

    rho = float(np.max(np.abs(vals)))
   
    if rho >= 1.0:
        
        A = A / (rho + 1e-3) 
    
    return A.astype(np.float32)


def _precompute_var_aux(
    A: np.ndarray, 
    neqs: int,
    Hq: int
) -> Dict[str, np.ndarray]:
    """
    Precompute powers P[0..Hq] and block-lower-triangular T_full for VAR(1) (#5).
    """
    
    P = np.empty((Hq + 1, neqs, neqs), dtype = np.float32)
  
    P[0] = np.eye(neqs, dtype = np.float32)
  
    for i in range(1, Hq + 1):
  
        P[i] = P[i - 1] @ A

    i_idx = np.arange(Hq)[:, None]
   
    j_idx = np.arange(Hq)[None, :]
   
    idx = i_idx - j_idx
   
    mask = idx >= 0
   
    T_full = np.zeros((Hq, Hq, neqs, neqs), dtype = np.float32)
   
    T_full[mask] = P[idx[mask]]
   
    return {
        "P": P, 
        "T_full": T_full
    }


def build_state() -> Dict[str, Any]:
    """
    Build read-only data structures once in parent:
    price/macro/VAR per country, fin statements, mappings, next forecasts.
    """
   
    logger.info("Building global state …")

    if SAVE_ARTIFACTS and not _os.path.exists(ARTIFACT_DIR):
   
        _os.makedirs(ARTIFACT_DIR, exist_ok = True)

    fdata = FinancialForecastData()
   
    macro = fdata.macro
   
    r = macro.r

    close_df = r.weekly_close
   
    tickers = list(config.tickers)
   
    fin_raw = fdata.prophet_data
   
    next_fc = fdata.next_period_forecast()
   
    next_macro = macro.macro_forecast_dict()
      
    analyst = r.analyst
   
    ticker_country = analyst["country"]

    dates_all = close_df.index.values

    tick_list = close_df.columns.values

    price_arr = close_df.values.astype(np.float32)

    T_dates, M = price_arr.shape

    total_rows = T_dates * M

    ds_col = np.repeat(dates_all, M)
    
    tick_col = np.tile(tick_list, T_dates)
   
    y_col = price_arr.reshape(-1)

    if len(tick_list) > 0:
        
        max_tlen = max(len(t) for t in tick_list) 
    
    else:
        
        max_tlen = 1
        
    price_rec = np.zeros(
        total_rows,
        dtype = [
            ("ds", "datetime64[ns]"), 
            ("ticker", f"U{max_tlen}"), 
            ("y", "float32")
        ]
    )
    
    price_rec["ds"] = ds_col
    
    price_rec["ticker"] = tick_col
    
    price_rec["y"] = y_col
    
    price_rec = price_rec[~np.isnan(price_rec["y"])]

    country_map = {t: str(c) for t, c in zip(analyst.index, analyst["country"])}
   
    country_arr = np.array([country_map.get(t, "") for t in price_rec["ticker"]])
   
    price_rec = rfn.append_fields(price_rec, "country", country_arr, usemask = False)
    
    price_rec = price_rec[np.lexsort((price_rec["ds"], price_rec["country"]))]

    raw_macro = macro.assign_macro_history_non_pct().reset_index()
    
    raw_macro = raw_macro.rename(
        columns = {"year": "ds"} if "year" in raw_macro else {raw_macro.columns[1]: "ds"}
    )
    
    raw_macro["ds"] = (
        raw_macro["ds"].dt.to_timestamp()
        if isinstance(raw_macro["ds"].dtype, pd.PeriodDtype)
        else pd.to_datetime(raw_macro["ds"])
    )
    
    raw_macro["country"] = raw_macro["ticker"].map(country_map)
    
    macro_clean = raw_macro[["ds", "country"] + MACRO_REGRESSORS].dropna()

    if len(macro_clean) > 0:
        
        max_clen = max(len(c) for c in macro_clean["country"])  
    
    else:
        
        max_clen = 1
        
    macro_rec = np.zeros(
        len(macro_clean),
        dtype = [("ds", "datetime64[ns]"), ("country", f"U{max_clen}")]
              + [(reg, "float32") for reg in MACRO_REGRESSORS]
    )
   
    macro_rec["ds"] = macro_clean["ds"].values
   
    macro_rec["country"] = macro_clean["country"].values
   
    for reg in MACRO_REGRESSORS:
   
        macro_rec[reg] = macro_clean[reg].values.astype(np.float32)
   
    macro_rec = macro_rec[np.lexsort((macro_rec["ds"], macro_rec["country"]))]

    unique_countries, first_idx = np.unique(macro_rec["country"], return_index = True)
    
    country_slices: Dict[str, Tuple[int,int]] = {}
    
    for i, ctry in enumerate(unique_countries):
       
        start = first_idx[i]
        
        if i+1 < len(first_idx):
       
            end = first_idx[i+1]  
        
        else:
            
            end = len(macro_rec)
       
        country_slices[ctry] = (start, end)

    country_var_results: Dict[str, Optional[dict]] = {}

    Hq = HORIZON // 4

    for ctry, (s, e) in country_slices.items():

        rec = macro_rec[s:e]

        if len(rec) == 0:

            country_var_results[ctry] = None

            continue

        dfm = pd.DataFrame(
            {reg: rec[reg] for reg in MACRO_REGRESSORS},
            index = pd.DatetimeIndex(rec["ds"])
        )
        
        dfm = (dfm[~dfm.index.duplicated(keep = "first")]
                 .sort_index().resample("W").mean().ffill().dropna())
       
        if dfm.shape[1] == 0 or len(dfm) <= dfm.shape[1]:
       
            country_var_results[ctry] = None
       
            continue

        try:
       
            vr = VAR(dfm).fit(maxlags=1)
       
        except Exception:
       
            country_var_results[ctry] = None
       
            continue

        if vr.k_ar < 1:
       
            country_var_results[ctry] = None
       
            continue

        A = _stable_A(
            A = vr.coefs[0].astype(np.float32)
        )
        
        Σdf = vr.resid.cov()
        
        Σ = Σdf.to_numpy(dtype = np.float32)
        Σ = 0.5 * (Σ + Σ.T)
       
        eigvals = np.linalg.eigvalsh(Σ)
       
        min_eig = float(eigvals.min())
       
        if min_eig < 0:
       
            Σ += np.eye(Σ.shape[0], dtype = Σ.dtype) * (-min_eig + SMALL_FLOOR)

        neqs = vr.neqs
       
        last_state = rec[-vr.k_ar:]
       
        init = (np.column_stack([last_state[r] for r in MACRO_REGRESSORS])
                .astype(np.float32).ravel())

        aux  = _precompute_var_aux(
            A = A,
            neqs = neqs, 
            Hq = Hq
        )    
        
        country_var_results[ctry] = {
            "init": init,
            "A": A, 
            "Sigma": Σ, 
            "neqs": neqs, 
            "aux": aux
        }

    fd_rec_dict: Dict[str, np.ndarray] = {}

    for t in tickers:

        df_fd = (fin_raw.get(t, pd.DataFrame())
                 .reset_index()
                 .rename(columns = {
                     "index": "ds",
                     "rev": "Revenue",
                     "eps": "EPS (Basic)"
                }))

        if df_fd.empty: 

            continue

        df_fd["ds"] = pd.to_datetime(df_fd["ds"])

        df_fd = df_fd[["ds", "Revenue", "EPS (Basic)"]].dropna()

        if df_fd.empty:

            continue
      
        rec = np.zeros(
            len(df_fd),
            dtype = [
                ("ds", "datetime64[ns]"), 
                ("Revenue", "float32"), 
                ("EPS (Basic)", "float32")
            ]
        )
        
        rec["ds"] = df_fd["ds"].values
        
        rec["Revenue"] = df_fd["Revenue"].values.astype(np.float32)
        
        rec["EPS (Basic)"] = df_fd["EPS (Basic)"].values.astype(np.float32)
        
        fd_rec_dict[t]= np.sort(rec, order = "ds")

    def _norm_country(
        x
    ) -> str:
    
        try:
    
            if pd.isna(x):
    
                return "UNK"
    
        except Exception:
    
            pass
    
        s = str(x)

        if s.strip().lower() in ("", "nan", "none"):
            
            return "UNK"  
        
        else:
            
            return s


    by_country: Dict[str, List[str]] = {}
    
    for t in tickers:
    
        c_raw = ticker_country.get(t, None)
        c = _norm_country(
            x = c_raw
        )
        
        by_country.setdefault(c, []).append(t)

    logger.info("Country groups: %d (%s ...)", len(by_country), ", ".join(list(by_country.keys())[:5]))

    grouped_tickers: List[str] = []

    for c in sorted(by_country.keys(), key=str):  

        grouped_tickers.extend(sorted(by_country[c]))

    assert all(isinstance(k, str) for k in by_country.keys()), "by_country keys not normalized to str"

    state_pack: Dict[str, Any] = {
        "tickers": grouped_tickers,
        "price_rec": price_rec,
        "macro_rec": macro_rec,
        "country_slices": country_slices,
        "country_var_results": country_var_results,
        "fd_rec_dict": fd_rec_dict,
        "next_fc": next_fc,
        "next_macro": next_macro,
        "latest_price": r.last_price,
        "analyst": analyst,
        "ticker_country": ticker_country,
    }
   
    logger.info("Global state built (%d tickers).", len(grouped_tickers))
   
    return state_pack


def forecast_one(
    ticker: str
) -> Dict[str, Any]:
    """
    Returns a dict:
      status: "ok" | "skipped" | "error"
      reason: if skipped/error
      min/avg/max/return/se if ok
    """
  
    assert STATE is not None, "Worker STATE not initialized"

    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    try:
       
        tf.config.experimental.enable_op_determinism()
  
    except Exception:
    
        pass

    tf.config.optimizer.set_jit(True)
  
    tf.random.set_seed(SEED)
  
    tf.config.threading.set_intra_op_parallelism_threads(1)
  
    tf.config.threading.set_inter_op_parallelism_threads(1)

    def build_model(
        n_reg: int
    ) -> Model:
       
        inp = Input((SEQUENCE_LENGTH, 1 + n_reg), dtype = "float32")
       
        x = LSTM(
            64, 
            return_sequences = True,
            kernel_regularizer = l2(L2_LAMBDA), 
            recurrent_regularizer = l2(L2_LAMBDA),
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = SEED),
            recurrent_initializer = tf.keras.initializers.Orthogonal(seed = SEED),
        )(inp)
        
        x = Dropout(0.1, seed = SEED)(x)
       
        x = LSTM(
            32,
            kernel_regularizer = l2(L2_LAMBDA), 
            recurrent_regularizer = l2(L2_LAMBDA),
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = SEED),
            recurrent_initializer = tf.keras.initializers.Orthogonal(seed = SEED),
        )(x)
        
        x = Dropout(0.1, seed = SEED)(x)
        
        out = Dense(HORIZON, kernel_regularizer = l2(L2_LAMBDA))(x)
    
        model = Model(inp, out)
     
        model.compile(
            optimizer = Adam(5e-4),
            loss = Huber(delta = 1.0),
            metrics = [tf.keras.metrics.MeanAbsoluteError(name = "mae")],
        )
        
        return model


    def get_cached_model(
        n_reg: int
    ) -> Tuple[Model, List[np.ndarray]]:
    
        global _MODEL_CACHE
    
        cached = _MODEL_CACHE.get(n_reg)
    
        if cached is None:
    
            m = build_model(
                n_reg = n_reg
            )
    
            init_w = [w.copy() for w in m.get_weights()]
    
            _MODEL_CACHE[n_reg] = (m, init_w)
    
        return _MODEL_CACHE[n_reg]


    def make_tf_dataset(
        X, 
        y, 
        batch = BATCH, 
        shuffle = False, 
        buffer = None,
        repeat = False
    ):
    
        ds = tf.data.Dataset.from_tensor_slices(
            (X.astype(np.float32), y.astype(np.float32))
        )
    
        if shuffle:
    
            ds = ds.shuffle(buffer or len(X), seed = SEED)
      
        if repeat:
      
            ds = ds.repeat()
      
        ds = ds.batch(batch)
      
        ds = ds.map(
            lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
            num_parallel_calls = tf.data.AUTOTUNE
        )
        
        return ds.prefetch(tf.data.AUTOTUNE)


    CALLBACKS = [
        EarlyStopping("loss", patience = PATIENCE, restore_best_weights = True),
        ReduceLROnPlateau("loss", patience = PATIENCE, factor = 0.5, min_lr = SMALL_FLOOR),
    ]

    def skip(
        reason: str
    ) -> Dict[str, Any]:
    
        return {"ticker": ticker, "status": "skipped", "reason": reason}

    try:
        rng = np.random.default_rng(SEED ^ (hash(ticker) & 0xFFFFFFFF))
      
        logger.info("%s: RAM %.1f / %.1f GB",
                    ticker,
                    psutil.Process().memory_info().rss / (1024 ** 3),
                    psutil.virtual_memory().total / (1024 ** 3))

        price_rec = STATE["price_rec"]

        macro_rec = STATE["macro_rec"]

        country_slices = STATE["country_slices"]

        country_var_results = STATE["country_var_results"]

        fd_rec_dict = STATE["fd_rec_dict"]

        next_fc = STATE["next_fc"]

        next_macro = STATE["next_macro"]

        latest_price = STATE["latest_price"]

        analyst = STATE["analyst"]

        ticker_country = STATE["ticker_country"]

        cur_p = latest_price.get(ticker, np.nan)

        if not np.isfinite(cur_p):

            return skip("no_latest_price")

        mask_t = price_rec["ticker"] == ticker

        pr_t = price_rec[mask_t]

        if len(pr_t) == 0:

            return skip("no_price_history")

        ctry = analyst["country"].get(ticker)

        if ctry not in country_slices:

            return skip("no_country_slice")

        s, e = country_slices[ctry]

        macro_ct = macro_rec[s:e]

        if len(macro_ct) < 8:

            return skip("insufficient_macro_history")

        idx_m = np.searchsorted(macro_ct["ds"], pr_t["ds"], "right") - 1

        valid = idx_m >= 0
       
        pr_t = pr_t[valid]
       
        idx_m = idx_m[valid]
       
        if len(pr_t) < SEQUENCE_LENGTH:
       
            return skip("short_price_history")

        yv = pr_t["y"].astype(np.float32)

        ys = np.where(yv > SMALL_FLOOR, yv, SMALL_FLOOR)

        ly = np.log(ys)

        lr = np.empty_like(ly)

        lr[0] = 0.0
      
        lr[1:] = ly[1:] - ly[:-1]

        ma = np.convolve(lr, np.ones(HIST_WINDOW) / HIST_WINDOW, mode = "valid")
        
        MA52 = np.concatenate([np.full(HIST_WINDOW - 1, np.nan, dtype = np.float32), ma])
       
        valid_ma = ~np.isnan(MA52)

        macro_vals = np.vstack([macro_ct[r][idx_m] for r in MACRO_REGRESSORS]).T
       
        if ticker in fd_rec_dict:
           
            fd_rec = fd_rec_dict[ticker]
           
            idx_f = np.searchsorted(fd_rec["ds"], pr_t["ds"], "right") - 1
           
            valid_fd = idx_f >= 0
        
        else:
        
            valid_fd = np.zeros(len(pr_t), bool)

        keep = valid_ma & valid_fd
        
        idx_keep = np.nonzero(keep)[0]
        
        if len(idx_keep) < SEQUENCE_LENGTH:
        
            return skip("insufficient_joint_features")

        dfm = pd.DataFrame({
            "ds": pr_t["ds"][idx_keep],
            "y": yv[idx_keep],
            "log_ret": lr[idx_keep],
            "Interest": macro_vals[idx_keep, 0],
            "Cpi": macro_vals[idx_keep, 1],
            "Gdp": macro_vals[idx_keep, 2],
            "Unemp": macro_vals[idx_keep, 3],
            "MA52_ret": MA52[idx_keep],
        })

        if ticker in fd_rec_dict:
          
            dfm["Revenue"] = fd_rec["Revenue"][idx_f[idx_keep]]
          
            dfm["EPS (Basic)"] = fd_rec["EPS (Basic)"][idx_f[idx_keep]]
          
            regs = ALL_REGRESSORS
        
        else:
        
            regs = NON_FIN_REGRESSORS

        n_reg = len(regs)
        
        n_ch = 1 + n_reg

        model, init_w = get_cached_model(
            n_reg = n_reg
        )
        
        model.set_weights(init_w)

        scaled_ret, scaled_reg, sc_reg, sc_ret, q_low, q_high = make_scaled_arrays(
            df = dfm[["y"] + regs], 
            regs = regs
        )
        
        X_full, y_full = make_windows(
            ret = scaled_ret, 
            reg = scaled_reg, 
            hist = HIST_WINDOW, 
            hor = HORIZON
        )

        N = len(X_full)
       
        if N == 0:
       
            return skip("no_training_windows")

        if not _shape_ok(
            X_full = X_full, 
            y_full = y_full, 
            hist = HIST_WINDOW, 
            hor = HORIZON, 
            n_reg = n_reg
        ):
       
            return skip(
                reason = "bad_window_shapes"
            )

        if not _no_nans(X_full, y_full):
       
            return skip(
                reason = "nan_in_windows"
            )

        n_splits = choose_splits(
            N = N, 
            hist = HIST_WINDOW, 
            hor = HORIZON
        )

        logger.info("%s: N=%d hist=%d hor=%d n_reg=%d splits=%d",
                    ticker, N, HIST_WINDOW, HORIZON, n_reg, n_splits)

        sigma_model = 0.0

        if n_splits >= 1:

            tscv = TimeSeriesSplit(n_splits = n_splits, gap = HORIZON)
          
            resids: List[np.ndarray] = []
          
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full), start = 1):
         
                model.set_weights(init_w)

                X_tr = X_full[train_idx]
                
                y_tr = y_full[train_idx]
                
                X_va = X_full[val_idx]
                
                y_va = y_full[val_idx]

                steps_per_epoch = max(1, math.ceil(len(train_idx)/BATCH))

                repeat_flag = steps_per_epoch > 1 and len(train_idx) >= BATCH

                train_ds = make_tf_dataset(
                    X = X_tr, 
                    y = y_tr,
                    shuffle = True,
                    buffer = len(X_tr),
                    repeat = repeat_flag
                )
                
                val_ds = make_tf_dataset(
                    X = X_va, 
                    y = y_va, 
                    repeat = False
                )

                model.fit(
                    train_ds,
                    validation_data = val_ds,   
                    epochs = EPOCHS,
                    steps_per_epoch = steps_per_epoch,
                    callbacks = CALLBACKS,
                    verbose = 0,
                )

                preds = model.predict(val_ds, verbose = 0)
                
                resids.append((y_va - preds).ravel())

            if resids:
               
                sigma_model = float(np.std(np.concatenate(resids), ddof = 1))

        model.set_weights(init_w)

        total_steps = max(1, math.ceil(N / BATCH))

        repeat_flag = total_steps > 1 and N >= BATCH

        ds_full = make_tf_dataset(
            X = X_full, 
            y = y_full, 
            shuffle = True, 
            buffer = N, 
            repeat = repeat_flag
        )

        model.fit(
            ds_full,
            epochs = EPOCHS, 
            steps_per_epoch = total_steps, 
            callbacks = CALLBACKS, 
            verbose = 0
        )

        hist_df = dfm.iloc[-HIST_WINDOW:].copy()

        last_hist_macro = hist_df[MACRO_REGRESSORS].values[-1].astype(np.float32)

        last_hist_macro = np.where(last_hist_macro > SMALL_FLOOR, last_hist_macro, SMALL_FLOOR)

        rev_points_arr = np.array([float(next_fc.at[ticker, rev_key]) for rev_key,_ in SCENARIOS], dtype = np.float32)

        eps_points_arr = np.array([float(next_fc.at[ticker, eps_key]) for _,eps_key in SCENARIOS], dtype = np.float32)
       
        n_scn = len(SCENARIOS)

        country_t = ticker_country[ticker]
       
        var_info = STATE["country_var_results"].get(country_t)

        if var_info is not None:
         
            init_state = var_info["init"]
         
            Σ = var_info["Sigma"]
         
            neqs = var_info["neqs"]
         
            aux = var_info["aux"]
         
            P = aux["P"]        
         
            T_full = aux["T_full"]   
         
            Hq = P.shape[0] - 1

            eps_shocks = rng.multivariate_normal(
                mean = np.zeros(neqs, dtype = np.float32),
                cov = Σ,
                size = (N_SIMS, Hq),
                method = "cholesky"
            ).astype(np.float32)

            noise_term = np.einsum("ski,tkij->stj", eps_shocks, T_full)

            init_term = init_state.astype(np.float32) @ P[1:].astype(np.float32)
          
            init_term = np.broadcast_to(init_term[np.newaxis, ...], (N_SIMS, Hq, neqs))

            sims_uncentered = np.clip(init_term + noise_term, SMALL_FLOOR, None)
           
            sims_tiled = np.repeat(sims_uncentered[None, ...], n_scn, axis = 0)
      
        else:

            int_array = np.array(next_macro.get(ticker, {}).get("InterestRate", [np.nan]), dtype = np.float32)
            
            inf_array = np.array(next_macro.get(ticker, {}).get("Consumer_Price_Index_Cpi", [np.nan]), dtype = np.float32)
            
            gdp_array = np.array(next_macro.get(ticker, {}).get("GDP", [np.nan, np.nan, np.nan]), dtype = np.float32)
            
            unemp_array = np.array(next_macro.get(ticker, {}).get("Unemployment", [np.nan]), dtype = np.float32)
            
            quarterly_fc = np.zeros((4, 4), dtype=np.float32)
            
            for i, arr in enumerate([int_array, inf_array, gdp_array, unemp_array]):
               
                if arr.size >= 4:
               
                    quarterly_fc[:, i] = arr[:4]
               
                else:
               
                    fill = arr.size
               
                    if fill > 0:
               
                        quarterly_fc[:fill, i] = arr
               
                        quarterly_fc[fill:, i] = arr[-1]
               
                    else:
               
                        quarterly_fc[:, i] = SMALL_FLOOR
            
            sims_tiled = np.broadcast_to(
                quarterly_fc[np.newaxis, np.newaxis, :, :], (n_scn, N_SIMS, 4, 4)
            )

        last_vals_all = np.broadcast_to(
            last_hist_macro.reshape(1, 1, 1, 4).astype(np.float32),
            (n_scn, N_SIMS, 1, 4)
        )

        with np.errstate(divide = "ignore", invalid = "ignore"):
           
            log_prev_all = np.log(np.where(last_vals_all > 0, last_vals_all, SMALL_FLOOR))
           
            log_q_all = np.log(np.where(sims_tiled > 0, sims_tiled, SMALL_FLOOR))

        cat = np.concatenate([log_prev_all, log_q_all], axis = 2)
       
        diffs_all = np.diff(cat, axis = 2)
       
        quarter_shocks = diffs_all[..., :4, :]
       
        deltas_macro_weekly = np.repeat(quarter_shocks, repeats_quarter, axis = 2)

        deltas_future_all = np.zeros((n_scn, N_SIMS, HORIZON, n_reg), dtype = np.float32)
       
        for m_idx, macro_name in enumerate(MACRO_REGRESSORS):
      
            reg_idx = regs.index(macro_name)
      
            deltas_future_all[..., reg_idx] = deltas_macro_weekly[..., m_idx]

        if "Revenue" in regs:

            last_rev = float(hist_df.iloc[-1]["Revenue"]) if "Revenue" in hist_df else SMALL_FLOOR
           
            last_rev = last_rev if last_rev > SMALL_FLOOR else SMALL_FLOOR
           
            last_rev_arr = np.full((n_scn,), last_rev, dtype = np.float32)
            
            valid_rev = rev_points_arr > SMALL_FLOOR
            
            mu_rev = np.zeros(n_scn, dtype = np.float32)
           
            mu_rev[valid_rev] = (np.log(rev_points_arr[valid_rev]) - np.log(last_rev_arr[valid_rev])) / 4.0
           
            q_rev = last_rev_arr[:, None] * np.exp(np.cumsum(mu_rev.reshape(n_scn, 1).repeat(4, axis=1), axis = 1)).astype(np.float32)
         
            with np.errstate(divide="ignore", invalid="ignore"):
         
                log_q_rev = np.log(np.where(q_rev > 0, q_rev, SMALL_FLOOR))
         
                log_prev_r = np.log(last_rev_arr)
         
            shifts_rev = np.concatenate([
                (log_q_rev[:, :1] - log_prev_r[:, None]),
                (log_q_rev[:, 1:] - log_q_rev[:, :-1])
            ], axis = 1)
            
            shifts_rev_exp = shifts_rev.reshape((n_scn, 1, 4))
           
            deltas_rev_weekly = np.repeat(shifts_rev_exp, repeats_quarter, axis = 2)
           
            deltas_rev_weekly = np.repeat(deltas_rev_weekly, N_SIMS, axis = 1)[..., None]
           
            deltas_future_all[..., regs.index("Revenue")] = deltas_rev_weekly[..., 0]

        if "EPS (Basic)" in regs:
          
            last_eps = float(hist_df.iloc[-1]["EPS (Basic)"]) if "EPS (Basic)" in hist_df else SMALL_FLOOR
          
            last_eps = last_eps if last_eps > SMALL_FLOOR else SMALL_FLOOR
          
            last_eps_arr = np.full((n_scn,), last_eps, dtype = np.float32)
          
            valid_eps = eps_points_arr > SMALL_FLOOR
          
            mu_eps = np.zeros(n_scn, dtype = np.float32)
           
            mu_eps[valid_eps] = (np.log(eps_points_arr[valid_eps]) - np.log(last_eps_arr[valid_eps])) / 4.0
        
            q_eps = last_eps_arr[:, None] * np.exp(np.cumsum(mu_eps.reshape(n_scn, 1).repeat(4, axis = 1), axis = 1)).astype(np.float32)
          
            with np.errstate(divide = "ignore", invalid = "ignore"):
               
                log_q_eps  = np.log(np.where(q_eps > 0, q_eps, SMALL_FLOOR))
               
                log_prev_e = np.log(last_eps_arr)
            
            shifts_eps = np.concatenate([
                (log_q_eps[:, :1] - log_prev_e[:, None]),
                (log_q_eps[:, 1:] - log_q_eps[:, :-1])
            ], axis = 1)
            
            shifts_eps_exp = shifts_eps.reshape((n_scn, 1, 4))
         
            deltas_eps_weekly = np.repeat(shifts_eps_exp, repeats_quarter, axis = 2)
         
            deltas_eps_weekly = np.repeat(deltas_eps_weekly, N_SIMS, axis = 1)[..., None]
         
            deltas_future_all[..., regs.index("EPS (Basic)")] = deltas_eps_weekly[..., 0]

        center = sc_reg.center_.reshape((1, 1, 1 ,n_reg)).astype(np.float32)
       
        scale = sc_reg.scale_.reshape((1, 1, 1, n_reg)).astype(np.float32)
      
        q_low_b = q_low.reshape((1, 1, 1, n_reg))
     
        q_high_b = q_high.reshape((1, 1, 1, n_reg))
     
        deltas_future_all = (np.clip(deltas_future_all, q_low_b, q_high_b) - center) / scale

        X_hist = np.zeros((1, HIST_WINDOW, n_ch), dtype = np.float32)
       
        last_hist_returns = hist_df["log_ret"].values[-HORIZON:].astype(np.float32).reshape(-1, 1)
    
        lr2 = (last_hist_returns - sc_ret.center_) / sc_ret.scale_
    
        X_hist[0, :, 0] = lr2.ravel()

        full_regs_array = dfm[regs].astype(np.float32).values
      
        with np.errstate(divide = "ignore", invalid = "ignore"):
      
            log_regs_full = np.log(np.where(full_regs_array > SMALL_FLOOR, full_regs_array, SMALL_FLOOR))
      
        delta_regs_full = np.diff(log_regs_full, axis = 0).astype(np.float32)
       
        delta_regs_full = (delta_regs_full - sc_reg.center_) / sc_reg.scale_
      
        X_hist[0, :, 1:] = delta_regs_full[-HORIZON:, :]

        zeros_future_ret = np.zeros((n_scn, N_SIMS, HORIZON, 1), dtype = np.float32)
       
        X_future = np.concatenate([zeros_future_ret, deltas_future_all], axis = 3)
       
        hist_block = np.broadcast_to(X_hist, (n_scn, N_SIMS, HIST_WINDOW, n_ch))
       
        X_all = np.concatenate([hist_block, X_future], axis = 2)
       
        X_all_flat = X_all.reshape(-1, SEQUENCE_LENGTH, n_ch)

        logger.info("%s: scenarios=%d sims=%d flat=%d",
                    ticker, n_scn, N_SIMS, len(X_all_flat))

        pred_scaled_all = predict_in_blocks(
            model = model, 
            X = X_all_flat, 
            block = PRED_BLOCK
        )

        if not (np.isfinite(sigma_model) and sigma_model >= SMALL_FLOOR):
          
            sigma_model = 0.0
      
        noise_all = rng.normal(loc = 0.0, scale = sigma_model, size = pred_scaled_all.shape).astype(np.float32)
     
        pred_scaled_noisy = pred_scaled_all + noise_all

        median = float(sc_ret.center_[0])
        
        iqr = float(sc_ret.scale_[0])
        
        pred_returns = pred_scaled_noisy * iqr + median

        sum_returns = np.sum(pred_returns, axis=1)
      
        final_prices_flat = cur_p * np.exp(sum_returns)

        p_lower, p_median, p_upper = np.nanpercentile(final_prices_flat, [2.5, 50.0, 97.5])
      
        rets = final_prices_flat / cur_p - 1.0
      
        avg_ret_val = float(np.nanmean(rets))
      
        std_ret_val = float(np.nanstd(rets, ddof=0))

        if SAVE_ARTIFACTS:

            try:

                np.savez_compressed(
                    _os.path.join(ARTIFACT_DIR, f"{ticker}_artifacts.npz"),
                    scaler_reg_center = sc_reg.center_.astype(np.float32),
                    scaler_reg_scale = sc_reg.scale_.astype(np.float32),
                    scaler_ret_center = sc_ret.center_.astype(np.float32),
                    scaler_ret_scale = sc_ret.scale_.astype(np.float32),
                    q_low = q_low, 
                    q_high = q_high,
                    n_splits = n_splits,
                    N = N, 
                    hist = HIST_WINDOW, 
                    hor = HORIZON,
                    n_reg = n_reg
                )
                
            
            except Exception as ex:
            
                logger.warning("%s: artifact save failed: %s", ticker, ex)

        gc.collect()

        return {
            "ticker": ticker, 
            "status": "ok",
            "min": float(p_lower), 
            "avg": float(p_median),
            "max": float(p_upper),
            "Returns": avg_ret_val, 
            "SE": std_ret_val
        }

    except Exception as e:
        
        logger.error("Error processing %s: %s", ticker, e)

        gc.collect()
        
        return {
            "ticker": ticker, 
            "status": "error",
            "reason": str(e)
        }

def main() -> None:
    
    faulthandler.enable()

    state_pack = build_state()
    
    tickers: List[str] = state_pack["tickers"]  

    ctx = mp.get_context("spawn")
    
    max_workers = min(4, _os.cpu_count() or 2)
    
    logger.info("Starting pool with %d workers …", max_workers)

    results: List[Dict[str, Any]] = []
   
    with ProcessPoolExecutor(
        max_workers = max_workers,
        mp_context = ctx,
        initializer = _init_worker,
        initargs = (state_pack,)
    ) as exe:
        
        futures = [exe.submit(forecast_one, t) for t in tickers]
      
        for fut in as_completed(futures):
        
            try:
               
                res = fut.result(timeout = 1800)  
                
                results.append(res)
                
                if res.get("status") == "ok":
                    logger.info("Ticker %s: Min %.2f, Avg %.2f, Max %.2f, Return %.4f, SE %.4f",
                                res["ticker"], res["min"], res["avg"], res["max"], res["Returns"], res["SE"])
              
                else:
              
                    logger.info("Ticker %s: %s (%s)",
                                res.get("ticker"), res.get("status"), res.get("reason"))
           
            except TimeoutError:
           
                logger.error("A worker timed out; continuing.")
           
            except Exception as ex:
           
                logger.error("Worker failed: %s", ex)

    ok_rows = [r for r in results if r.get("status") == "ok"]
    
    bad_rows = [r for r in results if r.get("status") != "ok"]

    if ok_rows:
     
        df_ok = pd.DataFrame(ok_rows).set_index("ticker")[["min", "avg", "max", "Returns", "SE"]]
  
    else:
  
        df_ok = pd.DataFrame(columns = ["min", "avg", "max", "Returns", "SE"])

    if bad_rows:
       
        df_bad = pd.DataFrame(bad_rows).set_index("ticker")[["status", "reason"]]
    
    else:
    
        df_bad = pd.DataFrame(columns = ["status", "reason"])

    try:

        if _os.path.exists(EXCEL_PATH):

            with pd.ExcelWriter(EXCEL_PATH, mode="a", engine = "openpyxl", if_sheet_exists = "replace") as writer:

                df_ok.to_excel(writer, sheet_name = "LSTM")

                df_bad.to_excel(writer, sheet_name = "LSTM_skips")
       
        else:
       
            with pd.ExcelWriter(EXCEL_PATH, engine = "openpyxl") as writer:
       
                df_ok.to_excel(writer, sheet_name = "LSTM")
       
                df_bad.to_excel(writer, sheet_name = "LSTM_skips")
       
        logger.info("Saved results to %s", EXCEL_PATH)
   
    except Exception as ex:
       
        logger.error("Failed to write Excel: %s", ex)

    logger.info("Forecasting complete. ok=%d, skipped/error=%d", len(ok_rows), len(bad_rows))


if __name__ == "__main__":
    
    try:
    
        mp.set_start_method("spawn", force = True)
    
    except RuntimeError:
  
        pass

    profiler = cProfile.Profile()
  
    profiler.enable()
  
    try:
  
        main()
  
    finally:
  
        profiler.disable()
  
        stats = pstats.Stats(profiler).sort_stats("cumtime")
  
        stats.print_stats(20)
