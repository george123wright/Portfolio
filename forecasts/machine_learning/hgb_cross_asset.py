"""
Annual equity-return modelling with cross-asset lags, bootstrap ensembling, and
quasi–Monte Carlo simulation.

Overview
--------
This module implements a full pipeline to estimate and simulate one-period
(typically annual) equity returns across a universe of tickers. It:

1) Builds a long panel of per-ticker fundamentals and returns.

2) Engineers cross-asset lagged features (market, sector, peer-K means).

3) Fits a base forecaster per ticker (HistGradientBoostingRegressor or Ridge)
   using time-series cross-validation with safeguards for small samples.

4) Trains a bootstrap ensemble via circular block resampling to capture model
   and sampling uncertainty.

5) Generates predictive distributions by quasi–Monte Carlo simulation of key
   drivers (revenue and EPS growth), using analyst-informed targets and
   dispersions.

6) Summarises the predictive distribution per ticker (10th, 50th, 90th
   percentiles and standard error) and exports results.

Notation and feature construction
---------------------------------
Let r_{t,j} denote the realised return of ticker j at date t (from column
'Return'). Cross-asset lag features are computed from lagged returns r_{t-1,·}
to avoid look-ahead:

• MKT_L1 (leave-one-out market mean):

    MKT_L1_{t,j} = (sum over l in A_t \ {j} of r_{t-1,l}) / (|A_t| - 1),

  where A_t is the set of tickers with non-missing r_{t-1,·} on date t.

• SECTOR_L1 (leave-one-out sector mean):

    SECTOR_L1_{t,j} = (sum over l in A_{t,S(j)} \ {j} of r_{t-1,l}) / (|A_{t,S(j)}| - 1),
  
  where S(j) is ticker j’s sector and A_{t,S(j)} are tickers in that sector
  with non-missing r_{t-1,·}. If the denominator is zero, the value falls
  back to MKT_L1_{t,j}.

• PEERK_L1 (top-K peer mean by absolute correlation):
  
  For each j, compute Pearson correlations C(j,k) between the histories of j
  and k on pairwise complete observations (minimum sample size enforced).
  
  Select the K tickers with largest |C(j,k)| and define:
  
    PEERK_L1_{t,j} = mean over k in TopK(j) of r_{t-1,k}.
  
  If no peers qualify, fall back to MKT_L1_{t,j}.
  For moderate universes (N ≤ threshold) a full float32 correlation matrix is
  formed; otherwise, correlations are computed on the fly to bound memory.

Modelling
---------
Base learner selection depends on sample size n:

• For n < 2: a trivial Ridge model is fitted (API compliance).

• For n < 8: Ridge regression with L2 penalty selected by time-series CV.

• For 8 ≤ n < 20: HistGradientBoostingRegressor (HGBR) with conservative,
  capped complexity is trained (no hyperparameter search).

• For n ≥ 20: a one-off global hyperparameter search for HGBR is performed
  using HalvingGridSearchCV with TimeSeriesSplit, and the resulting best
  parameters are cached and reused across tickers.

Time-series cross-validation preserves temporal order; folds are constructed
so that training windows respect a minimum length, and validation always
occurs on future observations relative to training.

Bootstrap ensembling uses a circular block bootstrap of the index:
blocks of length b are drawn with replacement by selecting random start points
and wrapping modulo n; blocks are concatenated and truncated to length n.
Each resample is used to fit a clone of the base model, yielding an ensemble
of predictors that captures sampling variability and model instability.

Driver simulation and transforms
--------------------------------
Predictive uncertainty is propagated by simulating revenue and EPS growth.

• Revenue growth G_rev is modelled as log-normal:

    Z_rev ~ Normal(mu_rev, sigma_rev^2),  G_rev = exp(Z_rev),

  where mu_rev = log(max(targ_rev, 1e-12)) and sigma_rev is inferred from
  analyst information.

• EPS growth G_eps is modelled on the signed log-one-plus scale:

    Transform T(x) = sign(x) * log(1 + |x|),

    Inverse  T^{-1}(y) = sign(y) * (exp(|y|) - 1).

  Draw Z_eps ~ Normal(mu_eps, sigma_eps^2) with mu_eps = T(targ_eps), and set
  G_eps = T^{-1}(Z_eps). This accommodates negative and heavy-tailed growth.

Quasi–Monte Carlo sampling uses a scrambled Sobol sequence u in [0,1)^2,
mapped via Z = Phi^{-1}(u) (componentwise Gaussian inverse CDF) to obtain
(Z_rev, Z_eps). Quasi-random points reduce integration variance relative to
independent sampling for smooth functionals.

Prediction aggregation
----------------------
For each draw, the feature template vector is copied and the revenue and EPS
entries are replaced by the simulated values. Predictions are produced either
by:

• Mixture sampling: choose a bootstrap model at random per draw and predict; or

• Fixed averaging: predict with a fixed subset of models and average.

Chunked inference avoids materialising a dense n_draws × p matrix; features are
constructed and predicted in bounded-size chunks to control peak memory.

Outputs
-------
The main entry point `main()` logs progress, trains per-ticker bundles in
parallel, runs the global Monte Carlo simulation, and writes an output table
with index 'Ticker' and columns:

  'Low Returns' (10th percentile),
  'Returns' (50th percentile),
  'High Returns' (90th percentile),
  'SE' (sample standard deviation of draws).

Results are exported to the Excel workbook indicated by `config.MODEL_FILE`.

Performance and reproducibility
-------------------------------
• Arrays are stored as float32 where appropriate to reduce auxiliary space.

• Native thread pools (BLAS/MKL/OpenBLAS) are capped per process to prevent
  oversubscription under process-based parallelism.

• `HalvingGridSearchCV` is used for a single global HGBR parameter search to
  amortise hyperparameter tuning.

• Seeds are derived deterministically per ticker by hashing the symbol and
  combining with a base seed (reproducible runs).

Assumptions and limitations
---------------------------
• The per-ticker DataFrames must contain a numeric 'Return' and the features
  'Revenue Growth' and 'EPS Growth' for simulation-time injection.

• Cross-asset features rely on sufficient overlap of return histories.

• Analyst-informed targets and dispersions are expected to be coherent across
  providers; missing values are handled with sensible fallbacks.

• The system is intended for research and operational forecasting; it does not
  constitute investment advice.

Quick start
-----------
Typical usage:

    if __name__ == "__main__":
        logging.info("Starting annual return modelling with cross-asset lags")
        main()

Configuration and dependencies
------------------------------
Configuration is read from `config`, including the ticker universe and output
file path. The pipeline depends on NumPy, pandas, scikit-learn, joblib,
threadpoolctl, tqdm, SciPy (for Sobol and Gaussian CDF), and psutil (optional
memory logging).

"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Mapping, Optional

import numpy as np
import pandas as pd
from numpy.random import default_rng

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, Ridge
from joblib import Parallel, delayed
import time
import joblib
from contextlib import contextmanager
from tqdm.auto import tqdm
import psutil
from threadpoolctl import threadpool_limits
import hashlib

GLOBAL_HGBR_PARAMS = None

from scipy.stats import qmc, norm

from TVP_GARCH_MC import _analyst_sigmas_and_targets_combined

from sklearn.experimental import enable_halving_search_cv   
from sklearn.model_selection import HalvingGridSearchCV

from data_processing.financial_forecast_data import FinancialForecastData
from functions.export_forecast import export_results
import config

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


USE_HALVING = True

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s %(levelname)-8s %(message)s",
    datefmt = "%H:%M:%S",
)

logger = logging.getLogger(__name__)


class LogTimer:
    """
    Context manager that logs wall-clock duration of a code block.

    Behaviour
    ---------
    Upon entering the context, the timer records a high-resolution start time
    using `time.perf_counter`. Upon exit, it logs either a success message
    with the elapsed seconds or, if an exception escaped the block, an error
    message together with the runtime up to the failure.

    Parameters
    ----------
    name : str
        Human-readable label for the timed block.
    logger : logging.Logger
        Logger used for messages; defaults to the module logger.

    Rationale
    ---------
    Long pipelines benefit from lightweight instrumentation. Wall-clock
    timing provides a practical measure of end-to-end latency (including
    I/O and process scheduling), which is typically more informative for
    model-building workflows than pure CPU time.
    """
    
    def __init__(
        self,
        name: str, 
        logger = logger
    ):
    
        self.name = name
    
        self.logger = logger
    
        self.t0 = None
    
    
    def __enter__(self):
    
        self.t0 = time.perf_counter()
    
        self.logger.info("%s ...", self.name)
    
        return self
    
    
    def __exit__(
        self,
        exc_type,
        exc, 
        tb
    ):
    
        dt = time.perf_counter() - self.t0
    
        if exc is None:
    
            self.logger.info("%s done in %.2fs", self.name, dt)
    
        else:
    
            self.logger.exception("✖ %s failed after %.2fs", self.name, dt)


@contextmanager
def tqdm_joblib(
    tqdm_object
):
    """
    Tie joblib's progress callbacks to a tqdm progress bar.

    This context manager temporarily replaces joblib's internal
    `BatchCompletionCallBack` class so that completion of each batch of
    parallel tasks advances the provided tqdm progress bar.

    Parameters
    ----------
    tqdm_object : tqdm.tqdm
        A tqdm instance configured with the desired total and description.

    Notes
    -----
    The monkey-patch is scoped to the context block and restored afterwards.
    This avoids persistent global state changes and prevents interference
    with unrelated joblib usage.
    """
    
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
    
        def __call__(
            self,
            *args,
            **kwargs
        ):
        
            tqdm_object.update(n = self.batch_size)
        
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    
    try:
      
        yield tqdm_object
   
    finally:
   
        joblib.parallel.BatchCompletionCallBack = old_cb
   
        tqdm_object.close()


def log_mem(
    note = ""
):
    """
    Log the current process resident set size (RSS) in MiB.

    Parameters
    ----------
    note : str
        Optional free-text suffix to include in the log line.

    Rationale
    ---------
    Memory diagnostics are important when processing large panels or when
    building many models in parallel. Monitoring RSS helps identify peak
    usage and potential pressure from array materialisation.
    """
    
    p = psutil.Process(os.getpid())

    rss = p.memory_info().rss / (1024 ** 2)

    logger.info("MEM %.1f MiB %s", rss, note)


class LimitThreads:
    """
    Context manager that limits thread usage of native libraries (BLAS, MKL,
    OpenBLAS, etc.) via `threadpoolctl`.

    Parameters
    ----------
    n : int
        Maximum number of threads to allow for libraries controlled by
        threadpoolctl.

    Rationale
    ---------
    When using joblib's process-based parallelism, oversubscription can occur
    if each worker internally spawns many BLAS threads. Capping threads per
    process prevents multiplicative contention and stabilises wall-clock
    performance.
    """
        
    def __init__(
        self, 
        n = 1
    ): 
        
        self.n=n
   
   
    def __enter__(
        self
    ): 
        
        self.cm = threadpool_limits(limits = self.n)
        
        return self
    
    
    def __exit__(
        self,
        *exc
    ): 
        self.cm.__exit__(*exc)
        
    
def _slog1p_signed(
    x: float
) -> float:
    """
    Signed log-one-plus transform applied elementwise.

    Definition
    ----------
    For any real x, define:
       
        y = sign(x) * log(1 + |x|)

    Properties
    ----------
    • Maps the real line to the real line.
    • Is odd, monotone in |x|, and approximately linear near zero:
       
        y ≈ x for small |x|.
    
    • Compresses heavy tails symmetrically, unlike a plain log.

    Usage
    -----
    Applied to EPS growth to obtain a near-Gaussian working scale even when
    raw values may be negative and heavy-tailed. The transform is numerically
    stable around zero and for large magnitudes.
    """
    
    x = np.asarray(x, dtype = np.float64)
    
    return np.sign(x) * np.log1p(np.abs(x))


def _slog1p_signed_inv(
    y: float
) -> float:
    """
    Inverse of the signed log-one-plus transform applied elementwise.

    Definition
    ----------
    For any real y, define:
       
        x = sign(y) * (exp(|y|) - 1)

    Relationship
    ------------
    This is the exact inverse of `_slog1p_signed`, i.e., composing the two
    returns the original value (subject to floating-point precision).
    """
        
    y = np.asarray(y, dtype = np.float64)
    
    return np.sign(y) * np.expm1(np.abs(y))


def _union_feature_columns(
    growth_hist: Dict[str, pd.DataFrame | str | None]
) -> List[str]:
    """
    Compute the sorted union of feature column names across ticker DataFrames,
    excluding the target column 'Return'.

    Parameters
    ----------
    growth_hist : dict[str, DataFrame | str | None]
        Mapping from ticker to a per-ticker feature/target DataFrame, or a
        sentinel (string/None) indicating unavailable data.

    Returns
    -------
    list[str]
        Sorted list of feature names present across any available DataFrame.

    Rationale
    ---------
    A consistent feature set is required to stack a multi-ticker panel with
    aligned columns and to initialise model design matrices uniformly.
    """
    
    cols: set[str] = set()

    for df in growth_hist.values():

        if isinstance(df, str) or df is None:

            continue

        if not isinstance(df, pd.DataFrame) or df.empty:

            continue

        for c in df.columns:

            if c != "Return":

                cols.add(c)

    return sorted(cols)


def fit_or_get_global_params(
    X_arr,
    y_arr, 
    cv, 
    random_state
):
    """
    Run a one-off hyperparameter search for HistGradientBoostingRegressor
    and cache the resulting parameterisation for subsequent calls.

    Procedure
    ---------
    1. Construct a pipeline with a single HGBR step (early stopping disabled).
    
    2. Search over a compact grid of structural parameters using
       `HalvingGridSearchCV` with time-series cross-validation.
    
    3. Cache the best HGBR step's parameters (not the whole pipeline).

    Parameters
    ----------
    X_arr : ndarray, shape (n_samples, n_features)
        Training design matrix.
    y_arr : ndarray, shape (n_samples,)
        Training target vector.
    cv : sklearn.model_selection.BaseCrossValidator
        Time-series-aware cross-validation splitter.
    random_state : int
        Seed for HGBR reproducibility.

    Returns
    -------
    dict
        Dictionary of keyword parameters for initialising HGBR.

    Rationale
    ---------
    Re-running hyperparameter search per ticker is expensive and often
    unnecessary when the task (annual returns with similar feature sets)
    is homogeneous. Caching a competent global setting reduces latency while
    retaining good out-of-sample performance.
    """
    
    global GLOBAL_HGBR_PARAMS
    
    if GLOBAL_HGBR_PARAMS is not None:
    
        return GLOBAL_HGBR_PARAMS

    pipe = Pipeline([("model", HistGradientBoostingRegressor(
        random_state = random_state, 
        early_stopping = False
    ))])
    
    param_grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_leaf": [5, 10],
        "model__l2_regularization": [0.0, 1.0],
        "model__max_bins": [255],
    }
    
    max_iter_cap = _cap_max_iter(
        n_samples = X_arr.shape[0]
    )
    
    hs = HalvingGridSearchCV(
        pipe, 
        param_grid = param_grid, 
        cv = cv, 
        scoring = "neg_mean_squared_error",
        resource = "model__max_iter",
        max_resources = max_iter_cap,
        min_resources = max(30, min(50, max_iter_cap // 4)),
        factor = 3, 
        aggressive_elimination = True,
        n_jobs = 1, 
        refit = True,
    )
    
    with LimitThreads(1):
    
        hs.fit(X_arr, y_arr)
    
    GLOBAL_HGBR_PARAMS = hs.best_estimator_.named_steps["model"].get_params(deep = False)
    
    return GLOBAL_HGBR_PARAMS


def _build_panel(
    growth_hist: Dict[str, pd.DataFrame | str | None],
    tickers: List[str],
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Assemble a multi-index (Date, Ticker) panel from per-ticker histories.

    Steps
    -----
    1. Determine the union of feature columns across available tickers.
    2. For each ticker, copy, sort by date, drop duplicate dates, add any
       missing features filled with NaN, and keep only the union plus 'Return'.
    3. Add a '__Ticker__' column and set a MultiIndex (Date, Ticker).

    Parameters
    ----------
    growth_hist : dict
        Ticker -> DataFrame mapping of features plus 'Return'.
    tickers : list[str]
        Universe of tickers to include.
    sector_map : dict[str, str]
        Presently unused here (sector features are added downstream).

    Returns
    -------
    DataFrame
        Long panel with levels ('Date', 'Ticker') and columns being features
        plus 'Return'.

    Design notes
    ------------
    Using a long panel with a MultiIndex enables simple stacking/unstacking
    operations for cross-asset feature engineering and aligns time indices
    cleanly across assets with sparse histories.
    """
        
    feat_cols = _union_feature_columns(
        growth_hist = growth_hist
    )
    
    frames: List[pd.DataFrame] = []
    
    for tk in tickers:
    
        df = growth_hist.get(tk)
    
        if isinstance(df, str) or df is None:
    
            continue
    
        if not isinstance(df, pd.DataFrame) or df.empty:
    
            continue
    
        base = df.copy()
    
        try:
    
            base.index = pd.to_datetime(base.index)
    
        except Exception:
    
            pass
    
        base = base.sort_index()
    
        base = base[~base.index.duplicated(keep = "last")]
    
        for c in feat_cols:
    
            if c not in base.columns:
    
                base[c] = np.nan
    
        keep_cols = [c for c in feat_cols if c in base.columns] + ["Return"]
    
        base = base[keep_cols]
        
        base["__Ticker__"] = tk
    
        base.index.name = "Date"
    
        frames.append(base)
    
    if not frames:
    
        return pd.DataFrame()
    
    panel = pd.concat(frames, axis = 0)
    
    panel = panel.set_index("__Ticker__", append = True)
    
    panel.index.rename(["Date", "Ticker"], inplace = True)
    
    return panel


def _cross_asset_features(
    panel: pd.DataFrame,
    sector_map: Mapping[str, str],
    k_peers: int = 5,
    min_periods: int = 3,
    full_corr_threshold: int = 220,  
) -> pd.DataFrame:
    """
    Construct cross-asset lagged features and append them to the panel.

    Features
    --------
    Let r_{t,j} denote the one-period return of ticker j at time t
    (stored in column 'Return'). Define the following, using only values
    observed at time t-1 to avoid look-ahead bias.

    1) MKT_L1 (leave-one-out market mean):
    
       For each (t, j), let A_{t} be tickers with non-missing r_{t-1,·}.
    
       Define:
    
         MKT_L1_{t,j} = (sum_{l in A_t \\ {j}} r_{t-1,l}) / (|A_t| - 1)
    
       Computed with NaN-aware counts per date, thus handling unbalanced panels.

    2) SECTOR_L1 (leave-one-out sector mean):
    
       Let S(j) be the sector of ticker j and A_{t,S(j)} be tickers in
       S(j) with non-missing r_{t-1,·}. 
       
       Define:
   
         SECTOR_L1_{t,j} = (sum_{l in A_{t,S(j)} \\ {j}} r_{t-1,l}) / (|A_{t,S(j)}| - 1)
   
       If the denominator is zero at a given date, fall back to MKT_L1_{t,j}.

    3) PEERK_L1 (top-k peer mean by absolute correlation):
   
       Let C(j,k) be the Pearson correlation between the return histories
       of tickers j and k computed over pairwise complete observations with
       a minimum of `min_periods`. Select the k tickers with the largest
       |C(j,k)| for each j, and define:
     
         PEERK_L1_{t,j} = mean_{k in TopK(j)} r_{t-1,k}
     
       If no eligible peers exist, fall back to MKT_L1_{t,j}.

    Implementation
    --------------
    • Uses an efficient leave-one-out aggregation for market and sector means.
    • For peers:
      - If the number of tickers N ≤ `full_corr_threshold`, forms a dense
        correlation matrix in float32 for speed and selects top-k via
        argpartition.
      - Otherwise, computes correlations on the fly per ticker using
        pairwise complete observations to avoid materialising an N×N matrix.

    Returns
    -------
    DataFrame
        Copy of the input panel with three additional float32 columns:
        'MKT_L1', 'SECTOR_L1', and 'PEERK_L1'.

    Rationale
    ---------
    Cross-sectional context improves predictive power in equity return models.
    The leave-one-out market and sector means provide robust, low-variance
    proxies for broad and industry-specific momentum, respectively, while the
    peer component adapts to idiosyncratic co-movement structure.
    """
   
    if panel.empty or "Return" not in panel.columns:
   
        return panel

    ret = panel["Return"].unstack("Ticker").sort_index()
   
    ret_l1 = ret.shift(1)

    X = ret_l1.to_numpy(dtype = np.float32, copy = True)  
   
    T, N = X.shape
   
    valid = np.isfinite(X)

    with np.errstate(invalid = "ignore"):
       
        sum_all = np.nansum(X, axis = 1, keepdims = True, dtype = np.float64)
        
        cnt_all = np.sum(valid, axis = 1, keepdims = True)
        
        denom = (cnt_all - 1).astype(np.float32)
       
        mkt_vals = (sum_all.astype(np.float32) - X) / np.where(denom > 0, denom, np.nan)
    
    mkt_l1 = pd.DataFrame(mkt_vals, index = ret_l1.index, columns = ret_l1.columns)

    tickers = ret_l1.columns.to_list()
    
    sectors = np.array([sector_map.get(tk, "Unknown") for tk in tickers], dtype = object)
    
    sector_codes, sector_idx = np.unique(sectors, return_inverse = True)
    
    S = len(sector_codes)

    C = np.zeros((N, S), dtype = bool)
    
    C[np.arange(N), sector_idx] = True

    sum_sec = np.nan_to_num(X, nan = 0.0, posinf = 0.0, neginf = 0.0, copy = False) @ C.astype(np.float32)
   
    cnt_sec = valid.astype(np.int32) @ C.astype(np.int32)

    sec_vals = np.full_like(X, np.nan, dtype = np.float32)
   
    for j in range(N):
   
        s = sector_idx[j]
   
        num = sum_sec[:, s] - np.nan_to_num(X[:, j], nan = 0.0)
   
        den = (cnt_sec[:, s] - 1).astype(np.float32)
   
        with np.errstate(invalid = "ignore", divide = "ignore"):
   
            sec_vals[:, j] = num / np.where(den > 0, den, np.nan)

    sector_l1 = pd.DataFrame(sec_vals, index = ret_l1.index, columns = ret_l1.columns)
   
    sector_l1 = sector_l1.combine_first(mkt_l1)  

    if N <= full_corr_threshold:
        
        corr = ret_l1.astype(np.float32).corr(min_periods = min_periods)
        
        corr = corr.fillna(0.0).to_numpy(dtype = np.float32, copy = True)
        
        np.fill_diagonal(corr, 0.0)  
        
        peerk_arr = np.full((T, N), np.nan, dtype = np.float32)
       
        k_eff = min(k_peers, max(0, N - 1))
       
        if k_eff > 0:
       
            for j in range(N):
       
                cj = corr[:, j]
       
                if N - 1 > k_eff:
       
                    idx = np.argpartition(np.abs(cj), -(k_eff))[-k_eff:]
       
                else:
       
                    idx = np.flatnonzero(np.arange(N) != j) 
       
                peers = [tickers[i] for i in idx]
       
                if peers:
       
                    pj = ret_l1[peers].mean(axis = 1).astype(np.float32)
       
                    peerk_arr[:, j] = pj.to_numpy(dtype = np.float32, copy = False)

        peerk_l1 = pd.DataFrame(peerk_arr, index = ret_l1.index, columns = ret_l1.columns)
        
        peerk_l1 = peerk_l1.combine_first(mkt_l1)

    else:
       
        peerk_arr = np.full((T, N), np.nan, dtype = np.float32)
       
        for j in range(N):
       
            xj = X[:, j]
            
            vj = np.isfinite(xj)
            
            if vj.sum() < min_periods:
            
                continue

            corrs = np.full(N, np.nan, dtype = np.float32)
            
            for k in range(N):
            
                if k == j:
            
                    continue
            
                xk = X[:, k]
            
                mk = vj & np.isfinite(xk)
            
                n = int(mk.sum())
            
                if n < min_periods:
            
                    continue
            
                xx = xj[mk].astype(np.float64, copy=False)
            
                yy = xk[mk].astype(np.float64, copy=False)
            
                xx -= xx.mean()
            
                yy -= yy.mean()
            
                num = np.dot(xx, yy)
            
                den = np.sqrt(np.dot(xx, xx) * np.dot(yy, yy))
            
                if den > 0.0:
            
                    corrs[k] = float(num / den)

            idx = np.flatnonzero(np.isfinite(corrs))
            
            if idx.size == 0:
            
                continue
            
            if idx.size > k_peers:
            
                top_local = np.argpartition(np.abs(corrs[idx]), -k_peers)[-k_peers:]
            
                peers_idx = idx[top_local]
            
            else:
            
                peers_idx = idx

            if peers_idx.size > 0:
            
                pj = np.nanmean(X[:, peers_idx], axis = 1, dtype = np.float64)
            
                peerk_arr[:, j] = pj.astype(np.float32, copy = False)

        peerk_l1 = pd.DataFrame(peerk_arr, index = ret_l1.index, columns = ret_l1.columns)
       
        peerk_l1 = peerk_l1.combine_first(mkt_l1)

    out = panel.copy()
    
    out["MKT_L1"] = mkt_l1.stack().reindex(out.index).astype(np.float32)
    
    out["SECTOR_L1"] = sector_l1.stack().reindex(out.index).astype(np.float32)
    
    out["PEERK_L1"]= peerk_l1.stack().reindex(out.index).astype(np.float32)
    
    return out


def _min_train_tss(
    n_samples: int,
    n_splits: int,
    min_train: int = 6
) -> TimeSeriesSplit:
    """
    Construct a time-series cross-validator that respects a minimum training
    length and avoids degenerate splitting.

    Constraints
    -----------
    • At least two splits are produced.
    • No split requires more than n_samples - 1 observations.
    • Each training fold contains at least `min_train` observations where
      feasible.

    Returns
    -------
    sklearn.model_selection.TimeSeriesSplit
        A splitter whose `n_splits` is adapted to sample size.

    Rationale
    ---------
    Time-series CV preserves temporal order (train on past, validate on future),
    which is essential for realistic generalisation assessment in forecasting.
    """
    
    if n_samples < 4:
        
        raise ValueError("n_samples too small for TimeSeriesSplit")

    if n_samples < (min_train + 2):
        
        splits = max(2, n_samples - min_train)   
   
    else:
   
        splits = n_splits

    splits = max(2, min(splits, n_samples - 1))  
    
    return TimeSeriesSplit(n_splits = splits)


def _block_bootstrap_indices(
    n: int, 
    block_size: int, 
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Generate circular block bootstrap indices for a univariate time series.

    Construction
    ------------
    • Let b = block_size. The number of blocks is ceil(n / b).
    
    • Draw starting positions s_1, ..., s_B uniformly from {0, 1, ..., n-1}.
    
    • For each start s_b, form a length-b block:
        (s_b + 0) mod n, (s_b + 1) mod n, ..., (s_b + b - 1) mod n.
    
    • Concatenate blocks and truncate to length n.

    Returns
    -------
    ndarray of shape (n,)
        Index vector with replacement and dependence preserved within blocks.

    Rationale
    ---------
    Block bootstrapping acknowledges serial dependence by resampling segments
    of consecutive observations rather than IID points. The circular variant
    avoids edge effects by wrapping blocks around the series end.
    """
    
    if n <= 1:

        return np.zeros(1, dtype = int)

    b = max(1, min(block_size, n))

    n_blocks = int(np.ceil(n / b))

    starts = rng.randint(0, n, size = n_blocks)

    idx = [(s + np.arange(b)) % n for s in starts]

    return np.concatenate(idx)[:n]


def _clean_df_once(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series]:
    """
    Convert a per-ticker DataFrame into a numeric design matrix and target,
    performing basic cleaning and alignment.

    Steps
    -----
    1. Ensure monotone datetime index and drop duplicate dates (keep last).
  
    2. Drop non-numeric columns from the feature set and replace ±inf with NaN.
  
    3. Build a mask for rows with complete numeric features and non-missing
       target 'Return'.
  
    4. Return NumPy arrays X (float32), y (float32), a feature name list, and
       per-feature medians computed on the filtered data.

    Returns
    -------
    X : ndarray, shape (n, p)
        Numeric features.
    y : ndarray, shape (n,)
        Target returns.
    feature_names : list[str]
        Column names corresponding to X.
    train_medians : pd.Series
        Per-feature medians for later templating and imputation.

    Rationale
    ---------
    Many scikit-learn estimators require dense numeric arrays. Computing
    medians once enables reproducible feature templating that is independent
    of the sampling variability of future Monte Carlo draws.
    """

    if "Return" not in df.columns:
       
        raise ValueError("Input df must contain a 'Return' column.")

    df2 = df.copy()
   
    try:
   
        df2.index = pd.to_datetime(df2.index)
   
    except Exception:
   
        pass
   
    df2 = df2.sort_index()
   
    df2 = df2[~df2.index.duplicated(keep = "last")]

    X_raw = df2.drop(columns = ["Return"])
    
    X_num = X_raw.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    y = df2["Return"]

    mask = X_num.notna().all(axis = 1) & y.notna()
    
    Xc = X_num.loc[mask].astype(np.float32)
  
    yc = y.loc[mask].astype(np.float32)
  
    feature_names = list(Xc.columns)
  
    train_medians = Xc.median(axis = 0).astype(np.float32)
  
    return Xc.to_numpy(), yc.to_numpy(), feature_names, train_medians


def seed_for_ticker(
    base_seed: int, 
    tk: str
) -> int:
    """
    Derive a stable per-ticker integer seed from a base seed and ticker symbol.

    Method
    ------
    • Compute an 8-byte BLAKE2b digest of the ticker string.
    • Convert to an unsigned integer and combine with `base_seed` modulo 2^31 − 1.

    Rationale
    ---------
    Ensures independence and reproducibility of Monte Carlo sampling across
    tickers and runs, while remaining deterministic for a given input.
    """
    
    digest = hashlib.blake2b(tk.encode("utf-8"), digest_size = 8).digest()

    off = int.from_bytes(digest, "little") & 0x7fffffff

    return (base_seed + off) % (2 ** 31 - 1)


def _fit_ridge_fallback(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3
) -> object:
    """
    Fit a ridge regression with either analytical leave-one-out (when small)
    or time-series cross-validated alpha selection.

    Model
    -----
    Ridge solves:
      
        minimise over β:  ||y - X β||_2^2 + α ||β||_2^2
   
    where α > 0 controls shrinkage.

    Procedure
    ---------
    • For very small samples, use `RidgeCV` without CV (equivalent to GCV).
    • Otherwise, select α from a log-spaced grid via time-series CV with
      a minimum training size enforced by `_min_train_tss`.

    Rationale
    ---------
    Ridge is robust for low-n, collinear designs and serves as a fast and
    stable fallback when gradient boosting may overfit or be underdetermined.
    """
    
    n = len(y)
    
    alphas = np.logspace(-3, 3, 25)

    if n < 4:

        model = RidgeCV(alphas = alphas, cv = None)
        
        model.fit(X, y)
        
        return model

    min_train = min(6, max(3, n // 2))

    if n < (min_train + 2):

        model = RidgeCV(alphas = alphas, cv = None)

        model.fit(X, y)

        return model

    cv = _min_train_tss(
        n_samples = n,
        n_splits = min(n_splits, n - 1),
        min_train = min_train,
    )
    model = RidgeCV(alphas = alphas, cv = cv)
    
    model.fit(X, y)
    
    return model


def _n_boot_eff(
    n_samples: int, 
    n_boot_max: int = 30
) -> int:
    """
    Heuristic for the effective number of bootstrap resamples.

    Definition
    ----------
    n_boot = clamp(10, 3 * sqrt(n_samples), n_boot_max)

    Rationale
    ---------
    The standard error of a bootstrap estimate scales roughly like
    1 / sqrt(n_boot). The rule aims to reduce computational cost for small
    panels while still averaging over a meaningful ensemble. An upper bound
    prevents excessive training when the series is long.
    """
    
    return int(max(10, min(n_boot_max, 3 * np.sqrt(max(n_samples, 1)))))


def fit_base_model_from_arrays(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    n_splits: int = 4,
    random_state: int = 42,
    cv_n_jobs: int = 1
) -> object:
    """
    Fit a base forecaster given numeric arrays, choosing an estimator that
    is appropriate for the sample size and using time-series validation.

    Strategy
    --------
    • n_samples < 2: fit a trivial ridge to satisfy API requirements.
    • n_samples < 8: fit ridge with CV via `_fit_ridge_fallback`.
    • 8 ≤ n_samples < 20: fit a small HistGradientBoostingRegressor (HGBR)
      with conservative hyperparameters and capped `max_iter`.
    • n_samples ≥ 20: obtain a cached global HGBR parameterisation via
      `fit_or_get_global_params` and fit an HGBR with those settings.
      If `USE_HALVING` were not short-circuited by the above, a halving
      grid search could be performed with time-series CV.

    Models
    ------
    HistGradientBoostingRegressor is a tree ensemble trained by gradient
    descent on a least-squares loss. It uses histogram-based splits for
    efficiency. Ridge regression is linear with L2 regularisation.

    Cross-validation
    ----------------
    TimeSeriesSplit is used to maintain temporal order. This avoids leakage
    from the future into the past that would otherwise bias validation error
    downward.

    Returns
    -------
    estimator : object
        A fitted scikit-learn estimator ready for `.predict`.
    """

    n_samples = X_arr.shape[0]
    
    if n_samples < 2:
    
        m = Ridge(alpha = 1.0)
    
        m.fit(X_arr, y_arr)  
    
        return m

    if n_samples < 8:
        
        return _fit_ridge_fallback(
            X = X_arr, 
            y = y_arr,
            n_splits = min(3, max(2, n_samples - 1))
        )

    cv = _min_train_tss(
        n_samples = n_samples,
        n_splits = n_splits, 
        min_train = 6
    )
    
    max_iter_cap = _cap_max_iter(
        n_samples = n_samples
    )
    
    if n_samples >= 20:
        
        params = fit_or_get_global_params(
            X_arr = X_arr, 
            y_arr = y_arr, 
            cv = cv, 
            random_state = random_state
        )
        
        model = HistGradientBoostingRegressor(**params)
        
        with LimitThreads(1):
        
            model.fit(X_arr, y_arr)
        
        return model        

    if n_samples < 20:
       
        model = HistGradientBoostingRegressor(
            random_state = random_state,
            early_stopping = False,
            max_iter = max_iter_cap,
            learning_rate = 0.1,
            max_depth = 2,
            min_samples_leaf = 5,
            l2_regularization = 0.0,
            max_bins = 255,
        )
        
        with LimitThreads(1):
        
            model.fit(X_arr, y_arr)
        
        return model

    pipe = Pipeline([
        ("model", HistGradientBoostingRegressor(
            random_state = random_state,
            early_stopping = False,      
        ))
    ])

    param_grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_leaf": [5, 10],
        "model__l2_regularization": [0.0, 1.0],
        "model__max_bins": 255, 
    }

    if USE_HALVING:
        
        hs = HalvingGridSearchCV(
            pipe,
            param_grid = param_grid,
            cv = cv,
            scoring = "neg_mean_squared_error",
            resource = "model__max_iter",
            max_resources = max_iter_cap,
            min_resources = max(30, min(50, max_iter_cap // 4)),
            factor = 3,
            aggressive_elimination = True,
            n_jobs = cv_n_jobs,
            refit = True,
        )
        
        with LimitThreads(1):
           
            hs.fit(X_arr, y_arr)
       
        return hs.best_estimator_

    gs = GridSearchCV(pipe, param_grid, cv = cv, scoring = "neg_mean_squared_error",  n_jobs = cv_n_jobs, refit = True)
    
    with LimitThreads(1):
    
        gs.fit(X_arr, y_arr)
    
    return gs.best_estimator_


def _fit_one_bootstrap(
    base_model,
    X_feat, 
    y_arr, 
    random_state, 
    block_size = None, 
    idx = None
):
    """
    Fit a cloned base model on a single circular block-bootstrap resample.

    Parameters
    ----------
    base_model : estimator
        A fitted estimator whose hyperparameters are used for cloning.
    X_feat : ndarray, shape (n, p)
        Original design matrix.
    y_arr : ndarray, shape (n,)
        Original target vector.
    random_state : int
        Seed for generating bootstrap indices when `idx` is not supplied.
    block_size : int or None
        Block length; defaults to ceil(sqrt(n)) if None.
    idx : ndarray[int] or None
        Precomputed index vector; when provided it is used directly.

    Rationale
    ---------
    Bootstrapping provides a simple model-uncertainty ensemble by refitting
    the estimator on resampled series whilst preserving short-range temporal
    dependence through blocks. Cloning ensures the same hyperparameters are
    used in each resample.
    """
    
    n = X_feat.shape[0]
    
    if idx is None:
    
        rng = np.random.RandomState(random_state)
    
        if block_size is None:
    
            block_size = int(max(2, min(n, np.sqrt(n))))
    
        idx = _block_bootstrap_indices(
            n = n, 
            block_size = block_size, 
            rng = rng
        )
    
    Xb = X_feat[idx, :]
    
    yb = y_arr[idx]
    
    m = clone(base_model)
    
    m.fit(Xb, yb)
    
    return m


def bootstrap_models_from_arrays(
    base_model, 
    X_arr,
    y_arr,
    *, 
    n_resamples = 50, 
    block_size = None,
    boot_n_jobs = 1
):
    """
    Train an ensemble of bootstrap models in parallel.

    Procedure
    ---------
    1. Precompute `n_resamples` circular block-bootstrap index vectors using
       `_precompute_boot_indices` for reproducibility.
   
    2. For each index vector, clone `base_model` and fit on the resampled data.
   
    3. Return the list of fitted models.

    Parameters
    ----------
    base_model : estimator
        A fitted estimator used as a template for cloning.
    X_arr : ndarray, shape (n, p)
        Original design matrix.
    y_arr : ndarray, shape (n,)
        Original target vector.
    n_resamples : int
        Size of the bootstrap ensemble.
    block_size : int or None
        Block length for the circular bootstrap.
    boot_n_jobs : int
        Parallelism for the bootstrap training.

    Returns
    -------
    list[estimator]
        Ensemble of independently fitted models.

    Rationale
    ---------
    Ensembles reduce variance and propagate sampling uncertainty through to
    predictions without imposing parametric assumptions on residuals.
    """
    
    idxs = _precompute_boot_indices(
        n = X_arr.shape[0], 
        n_resamples = n_resamples,
        block_size = block_size
    )
    
    models = Parallel(n_jobs = boot_n_jobs, prefer = "processes", backend = "loky")(
        delayed(_fit_one_bootstrap)(base_model, X_arr, y_arr, seed, block_size, idxs[r])
        for r, seed in enumerate(range(n_resamples))
    )
    return models


def _precompute_boot_indices(
    n, n_resamples, 
    block_size, 
    seed = 0
):
    """
    Precompute circular block-bootstrap index vectors using NumPy's Generator.

    Parameters
    ----------
    n : int
        Length of the series.
    n_resamples : int
        Number of index vectors to produce.
    block_size : int or None
        Length of blocks; defaults to ceil(sqrt(n)).
    seed : int
        Seed for reproducibility.

    Returns
    -------
    list[np.ndarray]
        List of int64 arrays of length n containing bootstrap indices.

    Notes
    -----
    Precomputing indices allows deterministic ensembles across processes and
    avoids re-creating RNG state inside each worker.
    """
    
    rng = np.random.default_rng(seed)
   
    idxs = []
   
    for _ in range(n_resamples):
   
        b = block_size or int(max(2, min(n, np.sqrt(n))))
   
        n_blocks = int(np.ceil(n / b))
   
        starts = rng.integers(0, n, size = n_blocks)
   
        idx = (starts[:, None] + np.arange(b)[None, :]) % n
   
        idxs.append(idx.ravel()[:n].astype(np.int64, copy = False))
   
    return idxs


def _row_fore_from_maps(
    rev_map: Mapping[str, float] | None,
    eps_map: Mapping[str, float] | None
) -> pd.Series:
    """
    Assemble the forecast summary expected by `_analyst_sigmas_and_targets_combined`.

    Input schema
    ------------
    rev_map / eps_map may contain keys such as:
      - 'low_y', 'avg_y', 'high_y'    : point forecasts for year-over-year changes
      - 'low_fs', 'avg_fs', 'high_fs' : forward score or scenario points
      - 'n_yahoo', 'n_sa'             : analyst count metadata

    Returns
    -------
    pd.Series (float32)
        A canonical set of fields:
        ['low_rev_y', 'avg_rev_y', 'high_rev_y',
         'low_rev', 'avg_rev', 'high_rev',
         'low_eps_y', 'avg_eps_y', 'high_eps_y',
         'low_eps', 'avg_eps', 'high_eps'].

    Rationale
    ---------
    Normalising input shapes decouples the Monte Carlo logic from upstream
    data providers and ensures the downstream sigma/target inference sees a
    consistent interface.
    """
    
    rev_map = rev_map or {}
    
    eps_map = eps_map or {}
    
    s = pd.Series({
        "low_rev_y": rev_map.get("low_y", np.nan),
        "avg_rev_y": rev_map.get("avg_y", np.nan),
        "high_rev_y": rev_map.get("high_y", np.nan),
        "low_rev": rev_map.get("low_fs", rev_map.get("low", np.nan)),
        "avg_rev": rev_map.get("avg_fs", rev_map.get("avg", np.nan)),
        "high_rev": rev_map.get("high_fs", rev_map.get("high", np.nan)),
        "low_eps_y": eps_map.get("low_y", np.nan),
        "avg_eps_y": eps_map.get("avg_y", np.nan),
        "high_eps_y": eps_map.get("high_y", np.nan),
        "low_eps": eps_map.get("low_fs", eps_map.get("low", np.nan)),
        "avg_eps": eps_map.get("avg_fs", eps_map.get("avg", np.nan)),
        "high_eps": eps_map.get("high_fs", eps_map.get("high", np.nan)),
    }, dtype = np.float32)
    
    return s


def _pick_analyst_counts(
    rev_map: Mapping[str, float] | None,
    eps_map: Mapping[str, float] | None
) -> tuple[float, Optional[float]]:
    """
    Extract analyst sample sizes from forecast maps.

    Returns
    -------
    (n_yahoo, n_sa) : tuple[float, Optional[float]]
        Best-effort float counts, defaulting to 0.0 for Yahoo and None for SA
        if absent.

    Rationale
    ---------
    Analyst sample sizes inform uncertainty calibration in
    `_analyst_sigmas_and_targets_combined`, for example via shrinkage of
    dispersion parameters as counts increase.
    """
   
    rev_map = rev_map or {}
   
    eps_map = eps_map or {}
   
   
    def _first(
        *keys
    ):
   
        for k in keys:
   
            if k in rev_map and np.isfinite(rev_map[k]): return float(rev_map[k])
   
            if k in eps_map and np.isfinite(eps_map[k]): return float(eps_map[k])
   
        return None
    
   
    n_y = _first("n_yahoo", "n_y") or 0.0
   
    n_sa = _first("n_sa", "n_s")
   
    return float(n_y), (float(n_sa) if n_sa is not None else None)


def predict_chunked(
    models: list,
    base_vec: np.ndarray,
    i_rev: int,
    i_eps: int,
    rev_vec: np.ndarray,
    eps_vec: np.ndarray,
    *,
    choose_random_model: bool = True,
    m_idx: Optional[np.ndarray] = None,
    sel: Optional[np.ndarray] = None,
    K: int = 16,
    clip: Optional[tuple[float, float]] = None,
    chunk: int = 2000,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Predict Monte Carlo outcomes in memory-bounded chunks.

    Design
    ------
    • A base feature template `base_vec` encodes medians, macro overlays,
      and anchored cross-asset lags for the ticker.
    
    • For each draw d, two entries of the feature vector are replaced:
      
        X[d, i_rev] = rev_vec[d]          (revenue growth draw)
      
        X[d, i_eps] = eps_vec[d]          (EPS growth draw)
    
    • Prediction is computed in chunks of size `chunk` to control the peak
      auxiliary memory required to build X.

    Model aggregation
    -----------------
    • If `choose_random_model` is True:
       
        For each draw d, select a bootstrap model index m_idx[d] uniformly
        and set yhat[d] = models[m_idx[d]].predict(X[d]).
    
    • Else:
        
        Select a subset `sel` of at most K model indices and compute the
        per-draw mean prediction across that subset:
       
          yhat[d] = (1 / |sel|) * sum_{j in sel} f_j(X[d]).

    Parameters
    ----------
    models : list[estimator]
        Bootstrap ensemble for the ticker.
    base_vec : ndarray, shape (p,)
        Feature template vector.
    i_rev, i_eps : int
        Column indices for 'Revenue Growth' and 'EPS Growth'.
    rev_vec, eps_vec : ndarray, shape (n_draws,)
        Monte Carlo draws on the natural scales expected by the model.
    choose_random_model : bool
        Whether to sample a model per draw (mixture prediction) or to
        average a fixed subset of models (bagging).
    m_idx : ndarray[int] or None
        Precomputed per-draw model indices; generated if None.
    sel : ndarray[int] or None
        Fixed subset of model indices to average when `choose_random_model` is False.
    K : int
        Upper bound on the size of `sel` when not provided.
    clip : tuple[float, float] or None
        Optional lower/upper bounds applied to predictions per chunk.
    chunk : int
        Chunk size for building temporary design matrices.
    rng : np.random.RandomState or None
        RNG used when sampling model indices.

    Returns
    -------
    ndarray, shape (n_draws,)
        Predicted returns per draw.

    Rationale
    ---------
    Chunking avoids allocating an n_draws × p dense matrix when n_draws is
    large. The two aggregation modes reflect two notions of uncertainty:
    mixture sampling preserves model heterogeneity at the draw level; fixed
    averaging reduces variance akin to bagging.
    """
    
    base_vec = np.asarray(base_vec, dtype = np.float32).ravel()
    
    rev_vec = np.asarray(rev_vec, dtype = np.float32).ravel()
    
    eps_vec = np.asarray(eps_vec, dtype = np.float32).ravel()
    
    n_draws = rev_vec.size
    
    assert eps_vec.size == n_draws

    n_models = len(models)
   
    if n_models == 0:
   
        return np.zeros(n_draws, dtype = np.float32)

    if choose_random_model:
        
        if m_idx is None:
        
            if rng is None:
        
                rng = np.random.RandomState(0)
        
            m_idx = rng.integers(0, len(models), size = n_draws)
            
        else:
            
            m_idx = np.asarray(m_idx, dtype = int).ravel()
            
            assert m_idx.size == n_draws
    
    else:
        
        if sel is None:
        
            if rng is not None:
        
                sel = rng.choice(n_models, size = min(K, n_models), replace = False)
        
            else:
        
                sel = np.arange(min(K, n_models), dtype = int)
        
        else:
        
            sel = np.asarray(sel, dtype = int).ravel()
        
            assert sel.size > 0

    yhat = np.empty(n_draws, dtype = np.float32)
    
    p = base_vec.size
    
    lo_hi = clip if (clip is not None and all(v is not None for v in clip)) else None

    Xbuf = np.empty((chunk, p), dtype = np.float32)

    for s in range(0, n_draws, chunk):
       
        e = min(n_draws, s + chunk)
       
        m = e - s

        Xbuf[:m, :] = base_vec
       
        Xbuf[:m, i_rev] = rev_vec[s:e]
       
        Xbuf[:m, i_eps] = eps_vec[s:e]
       
        X = Xbuf[:m, :]

        if choose_random_model:
       
            idx_slice = m_idx[s:e]
       
            yh = np.empty(m, dtype = np.float32)

            for j in np.unique(idx_slice):
               
                mask = (idx_slice == j)
              
                if mask.any():
              
                    yh[mask] = models[int(j)].predict(X[mask, :])
          
            if lo_hi is not None:
          
                yh = np.clip(yh, lo_hi[0], lo_hi[1])
          
            yhat[s:e] = yh
        
        else:

            yh = np.zeros(m, dtype = np.float32)
        
            denom = float(sel.size)
           
            for j in sel:
           
                yh += models[int(j)].predict(X)
           
            yh /= denom
           
            if lo_hi is not None:
           
                yh = np.clip(yh, lo_hi[0], lo_hi[1])
           
            yhat[s:e] = yh

    return yhat


def prepare_per_ticker_arrays(
    panel_x: pd.DataFrame,
    tickers: list[str]
):
    """
    Materialise cleaned numeric arrays per ticker for downstream training.

    Parameters
    ----------
    panel_x : DataFrame
        Multi-index panel containing features and 'Return'.
    tickers : list[str]
        Tickers to extract.

    Returns
    -------
    dict[str, tuple]
        Mapping from ticker to (X, y, feature_cols, train_medians), where
        X and y are float32 arrays, feature_cols is a tuple of names, and
        train_medians is a float32 Series indexed by feature name.

    Rationale
    ---------
    Precomputing arrays once prevents repeated DataFrame slicing in parallel
    workers and reduces serialisation overhead by sending compact NumPy
    buffers across process boundaries.
    """
       
    out = {}
   
    for tk in tickers:
   
        try:
   
            df_t = panel_x.xs(tk, level = "Ticker")
   
        except KeyError:
   
            continue
   
        X, y, feat_cols, med = _clean_df_once(
            df = df_t
        )
   
        out[tk] = (
            X.astype(np.float32, copy = False),
            y.astype(np.float32, copy = False),
            tuple(feat_cols),                  
            pd.Series(med, dtype = np.float32),
        )
    return out


def train_ticker_bundle_from_arrays(
    tk: str,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    feature_cols: List[str],
    train_medians: pd.Series,
    macro_fc: Mapping[str, float],
    n_boot: int = 30,
    random_state: int = 42,
) -> Optional[dict]:
    """
    Train the per-ticker model bundle from numeric arrays and build cached
    artefacts for fast Monte Carlo prediction.

    Bundle contents
    ---------------
    • 'models'                 : bootstrap ensemble of fitted estimators
    • 'i_rev', 'i_eps'         : column indices for growth controls
    • 'template_vec'           : feature template (medians + macro + anchor)
    • 'anchor_rev_baseline'    : last observed revenue growth (fallback)
    • 'anchor_eps_baseline'    : last observed EPS growth (fallback)
    • 'feat_idx'               : name->index mapping for features

    Template construction
    ---------------------
    1. Start from per-feature medians estimated on the cleaned training rows.
   
    2. Overlay macro forecasts where provided (vectorised name lookup).
   
    3. Anchor all remaining non-macro, non-growth features to the last
       observed row to preserve recent cross-asset and idiosyncratic state.

    Bootstrap size
    --------------
    The effective number of resamples is `_n_boot_eff(n_samples, n_boot)`,
    i.e., min(n_boot, max(10, 3 * sqrt(n_samples))).

    Returns
    -------
    dict or None
        Model bundle, or None when data are insufficient or growth columns
        are missing.

    Rationale
    ---------
    Separating training from MC prediction allows caching of indices and
    the template vector so that the Monte Carlo loop performs no pandas
    manipulation and minimal Python overhead.
    """
   
    if X_arr is None or y_arr is None:
   
        return None
   
    if not isinstance(X_arr, np.ndarray) or not isinstance(y_arr, np.ndarray):
   
        return None
   
    n_samples, n_features = X_arr.shape
   
    if n_samples < 3 or n_features == 0:
   
        return None

    X_arr = np.asarray(X_arr, dtype = np.float32, order = "C")

    y_arr = np.asarray(y_arr, dtype = np.float32).ravel()

    try:
       
        i_rev = feature_cols.index("Revenue Growth")
       
        i_eps = feature_cols.index("EPS Growth")
    
    except ValueError:
    
        return None

    anchor_row_vec = X_arr[-1, :].astype(np.float32, copy = False)

    template = np.array(
        [float(train_medians.get(c, 0.0)) for c in feature_cols],
        dtype = np.float32
    )

    feat_idx: Dict[str, int] = {c: i for i, c in enumerate(feature_cols)}
   
    if macro_fc:
       
        for k, v in macro_fc.items():
       
            j = feat_idx.get(k)
       
            if j is not None and np.isfinite(v):
       
                template[j] = float(v)

    macro_idx: set[int] = {feat_idx[k] for k in macro_fc.keys() if k in feat_idx}

    for j, c in enumerate(feature_cols):
       
        if j in (i_rev, i_eps) or j in macro_idx:
       
            continue
       
        val = anchor_row_vec[j]
       
        if np.isfinite(val):
       
            template[j] = float(val)

    template_vec = np.ascontiguousarray(template, dtype = np.float32)

    _n_boot = _n_boot_eff(
        n_samples = n_samples, 
        n_boot_max = n_boot
    )

    with LimitThreads(1):
        
        base = fit_base_model_from_arrays(
            X_arr = X_arr,
            y_arr = y_arr,
            n_splits = 4,
            random_state = random_state,
            cv_n_jobs = 1,
        )
        boots = bootstrap_models_from_arrays(
            base_model = base,
            X_arr = X_arr,
            y_arr = y_arr,
            n_resamples = _n_boot,
            block_size = None,
            boot_n_jobs = 1,
        )

    anchor_rev_baseline = float(anchor_row_vec[i_rev]) if np.isfinite(anchor_row_vec[i_rev]) else np.nan
    
    anchor_eps_baseline = float(anchor_row_vec[i_eps]) if np.isfinite(anchor_row_vec[i_eps]) else np.nan

    return {
        "models": boots,
        "i_rev": i_rev,
        "i_eps": i_eps,
        "template_vec": template_vec,
        "anchor_rev_baseline": anchor_rev_baseline,
        "anchor_eps_baseline": anchor_eps_baseline,
        "feat_idx": feat_idx,
    }


def _train_one_ticker_arrays(
    tk: str,
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: tuple[str, ...],
    train_medians: pd.Series,
    macro_fc: Mapping[str, float],
    n_boot: int,
    random_state: int
):
    """
    Thin wrapper to train a per-ticker bundle from prepared arrays.

    Returns
    -------
    (ticker, bundle) : tuple[str, Optional[dict]]
        The key–value pair suitable for dict construction in parallel
        training pipelines.

    Rationale
    ---------
    Structuring the worker to return a tuple facilitates direct conversion
    of a list of results into a mapping without additional re-ordering.
    """
    
    return tk, train_ticker_bundle_from_arrays(
        tk = tk,
        X_arr = X, 
        y_arr = y,
        feature_cols = list(feat_cols),
        train_medians = train_medians,
        macro_fc = macro_fc,
        n_boot = n_boot, 
        random_state = random_state,
    )



def _cap_max_iter(
    n_samples: int
) -> int:
    """
    Heuristic cap for the number of boosting iterations in HGBR.

    Definition
    ----------
    max_iter = min(256, max(60, 20 * n_samples))

    Rationale
    ---------
    Small annual panels cannot support deep ensembles without overfitting.
    The cap scales gently with sample size to allow more complexity when
    sufficient history exists while imposing a hard upper bound for speed.
    """
        
    return int(min(256, max(60, 20 * n_samples)))


def _mc_predict_one_ticker(
    tk: str, 
    b: dict, 
    rev_map: Mapping[str, float], 
    eps_map: Mapping[str, float],
    n_draws: int, 
    seed: int, 
    choose_random_model: bool,
    clip: Optional[tuple[float, float]] = None
) -> dict:
    """
    Monte Carlo (quasi-random) prediction for a single ticker using a cached
    model bundle and analyst-informed priors for growth drivers.

    Distributional assumptions
    --------------------------
    Let G_rev denote revenue growth on a multiplicative scale and G_eps the
    EPS growth on an additive scale.

    • Revenue growth:
        
        Draw Z_rev ~ Normal(mu_rev, sigma_rev^2) in log space, where
       
          mu_rev  = log(max(targ_rev, 1e-12)),
      
          sigma_rev is provided by `_analyst_sigmas_and_targets_combined`.
       
        Then set:
      
          G_rev = exp(Z_rev)
      
        This yields a log-normal distribution for G_rev, ensuring positivity.

    • EPS growth:
      
        Work in the signed log-one-plus space:
      
          Y = sign(G_eps) * log(1 + |G_eps|)
      
        Draw Z_eps ~ Normal(mu_eps, sigma_eps^2) with
      
          mu_eps  = sign(targ_eps) * log(1 + |targ_eps|),
      
          sigma_eps as returned by `_analyst_sigmas_and_targets_combined`.
      
        Map back via:
      
          G_eps = sign(Z_eps) * (exp(|Z_eps|) - 1)

    Quasi-Monte Carlo
    -----------------
    A scrambled Sobol sequence u in [0, 1)^2 is generated (using base-2 size,
    truncated to `n_draws`) and mapped via the Gaussian inverse CDF:
    
      Z = Phi^{-1}(u)
    
    This reduces integration variance relative to IID sampling for smooth
    functionals of the draws.

    Prediction
    ----------
    For each draw d:
      1. Copy the cached feature template.
    
      2. Set feature[i_rev] = G_rev[d] and feature[i_eps] = G_eps[d].
    
      3. Predict either with a randomly chosen bootstrap model (mixture)
         or by averaging a fixed subset (bagging); implemented in
         `predict_chunked` for memory efficiency.

    Summary statistics
    ------------------
    The function returns the 10th, 50th, and 90th percentiles of the
    predictive distribution and the sample standard deviation:
      
      SE = sqrt( sum_i (yhat_i - mean)^2 / (n_draws - 1) )

    Parameters
    ----------
    tk : str
        Ticker identifier.
    b : dict
        Per-ticker bundle as produced by `train_ticker_bundle_from_arrays`.
    rev_map, eps_map : Mapping[str, float]
        Analyst forecast maps used to infer targets and sigmas.
    n_draws : int
        Number of quasi-random draws.
    seed : int
        Seed for the Sobol generator and RNG used for model selection.
    choose_random_model : bool
        Select mixture vs fixed-subset averaging for ensemble aggregation.
    clip : tuple[float, float] or None
        Optional bounds applied to predictions.

    Returns
    -------
    dict
        {'Ticker', 'Low Returns', 'Returns', 'High Returns', 'SE'}.

    Rationale
    ---------
    The approach integrates three uncertainty sources: input uncertainty
    (growth draws), model uncertainty (bootstrap ensemble), and finite-sample
    noise captured by the learner. Quasi-random sequences improve precision
    for a fixed computational budget.
    """
    
    rng = default_rng(seed)

    models: List[object] = b["models"]
   
    i_rev: int = b["i_rev"]
   
    i_eps: int = b["i_eps"]
   
    base_vec: np.ndarray = b["template_vec"]
   
    anchor_rev_baseline: float = b["anchor_rev_baseline"]
   
    anchor_eps_baseline: float = b["anchor_eps_baseline"]

    row = _row_fore_from_maps(
        rev_map = rev_map,
        eps_map = eps_map
    )
    
    n_y, n_sa = _pick_analyst_counts(
        rev_map = rev_map, 
        eps_map = eps_map
    )
    
    pars = _analyst_sigmas_and_targets_combined(
        n_yahoo = n_y,
        n_sa = n_sa,
        row_fore = row
    )

    targ_rev = pars.get("targ_rev", anchor_rev_baseline)

    mu_rev_T = np.log(max(float(targ_rev if np.isfinite(targ_rev) else 0.0), 1e-12))

    sig_rev_T = float(pars.get("rev_sigma", 0.05))

    targ_eps = pars.get("targ_eps", anchor_eps_baseline)

    mu_eps_T = _slog1p_signed(
        x = float(targ_eps if np.isfinite(targ_eps) else 0.0)
    )

    sig_eps_T = float(pars.get("eps_sigma", 0.05))
    
    m = int(np.ceil(np.log2(n_draws)))
   
    sampler = qmc.Sobol(d = 2, scramble = True, seed = seed)
    
    u = sampler.random_base2(m)[:n_draws]  

    z_rev = norm.ppf(u[:, 0]) * sig_rev_T + mu_rev_T
   
    z_eps = norm.ppf(u[:, 1]) * sig_eps_T + mu_eps_T

    rev_vec = np.exp(z_rev).astype(np.float32, copy = False)
   
    eps_vec = _slog1p_signed_inv(
        y = z_eps
    ).astype(np.float32, copy = False)
    
    if choose_random_model:
        
        m_idx = rng.integers(0, len(models), size = n_draws)

        yhat = predict_chunked(
            models = models,
            base_vec = base_vec,
            i_rev = i_rev,
            i_eps = i_eps,
            rev_vec = rev_vec,
            eps_vec = eps_vec,
            choose_random_model = True,
            m_idx = m_idx,
            clip = clip,
            chunk = 2000,
        )
        
    else:
        
        K_eff = min(16, len(models))
        
        sel = rng.choice(len(models), size = K_eff, replace = False)
        
        yhat = predict_chunked(
            models = models,
            base_vec = base_vec,
            i_rev = i_rev,
            i_eps = i_eps,
            rev_vec = rev_vec,
            eps_vec = eps_vec,
            choose_random_model = False,
            sel = sel,
            K = K_eff,
            clip = clip,
            chunk = 2000,
        )

    if clip is not None:
        
        lo, hi = clip
        
        yhat = np.clip(yhat, lo, hi)

    low, med, high = np.percentile(yhat, [10, 50, 90])
    
    se = float(yhat.std(ddof = 1)) if yhat.size > 1 else 0.0
    
    return {
        "Ticker": tk, 
        "Low Returns": float(low),
        "Returns": float(med),
        "High Returns": float(high), 
        "SE": se
    }


def global_mc_prediction(
    bundles,
    fin_fc_map, 
    *,
    n_draws = 1000,
    random_state = 42, 
    choose_random_model = True,
    n_jobs_mc = -1
) -> pd.DataFrame:
    """
    Run Monte Carlo prediction across all tickers in parallel.

    Parameters
    ----------
    bundles : dict[str, dict]
        Mapping from ticker to trained bundle.
    fin_fc_map : dict[str, tuple[Mapping[str, float], Mapping[str, float]]]
        For each ticker, a pair (rev_map, eps_map) providing analyst inputs.
    n_draws : int
        Number of draws per ticker.
    random_state : int
        Base seed; per-ticker seeds are derived by `seed_for_ticker`.
    choose_random_model : bool
        Whether to sample a model per draw or average a fixed subset.
    n_jobs_mc : int
        Joblib parallelism for the Monte Carlo loop.

    Returns
    -------
    DataFrame
        Index of tickers and columns: 'Low Returns', 'Returns',
        'High Returns', 'SE'.

    Rationale
    ---------
    Process-based parallelism scales efficiently across tickers because
    each ticker's Monte Carlo simulation is independent given its bundle
    and analyst inputs.
    """

    clip = (getattr(config, "lbr", None), getattr(config, "ubr", None))
    
    clip = clip if all(v is not None for v in clip) else None

    tickers = [tk for tk, b in bundles.items() if b]
    
    seeds = {
        tk: seed_for_ticker(
            base_seed = random_state, 
            tk = tk
        ) for tk in tickers
    }


    def _do(
        tk
    ):
    
        b = bundles[tk]
    
        rev_map, eps_map = fin_fc_map.get(tk, ({}, {}))
    
        return _mc_predict_one_ticker(
            tk = tk, 
            b = b,
            rev_map = rev_map, 
            eps_map = eps_map,
            n_draws = n_draws, 
            seed = seeds[tk],
            choose_random_model = choose_random_model,
            clip = clip,
        )


    rows = Parallel(n_jobs = n_jobs_mc, prefer = "processes", backend = "loky", batch_size = "auto")(
        delayed(_do)(tk) for tk in tickers
    )
    
    return pd.DataFrame(rows).set_index("Ticker").sort_index()


def main() -> None:
    """
    Orchestrate the full modelling pipeline:

    1) Data ingestion:
     
       Load fundamental and macro data via `FinancialForecastData()`.
       Retrieve per-ticker regression histories and sector labels.

    2) Panel construction:
     
       Build a long multi-index panel with `_build_panel`, then augment with
       cross-asset lagged features using `_cross_asset_features`.

    3) Macro overlays:
     
       Compute per-ticker macro forecast dictionaries.

    4) Array preparation:
     
       Convert per-ticker panels into dense arrays using
       `prepare_per_ticker_arrays`.

    5) Model training:
     
       Train a bootstrap ensemble per ticker in parallel using
       `train_ticker_bundle_from_arrays` wrapped by `_train_one_ticker_arrays`.
       Thread counts of native libraries are constrained to avoid
       oversubscription.

    6) Monte Carlo prediction:
     
       Build the (rev_map, eps_map) inputs per ticker and run
       `global_mc_prediction` with quasi-random draws to obtain predictive
       summaries.

    7) Export:
     
       Write the resulting DataFrame to the Excel workbook designated by
       `config.MODEL_FILE` using `export_results`.

    Instrumentation
    ---------------
    Each stage is wrapped in `LogTimer` blocks; joblib stages are connected
    to tqdm progress bars via `tqdm_joblib`. Memory snapshots may be logged
    with `log_mem` where useful.

    Outcome
    -------
    Produces a sheet 'hgb_cross_asset' containing per-ticker predictive
    statistics suitable for portfolio construction or reporting.
    """

    with LogTimer("Load data"):
       
        fdata = FinancialForecastData()
       
        macro = fdata.macro
       
        tickers: List[str] = list(config.tickers)
       
        growth_hist: Dict[str, pd.DataFrame | str] = fdata.regression_dict()
       
        sector_map: Dict[str, str] = (
            macro.r.sector.reindex(tickers).fillna("Unknown").astype(str).to_dict()
        )

    with LogTimer(f"Build panel ({len(tickers)} tickers)"):
       
        panel = _build_panel(
            growth_hist = growth_hist, 
            tickers = tickers, 
            sector_map = sector_map
        )
       
        if panel.empty:
       
            logger.warning("No panel data; exiting.")
       
            return

    with LogTimer("Cross-asset features"):
        
        panel_x = _cross_asset_features(
            panel = panel, 
            sector_map = sector_map,
            k_peers = 5
        )

    with LogTimer("Assign macro forecasts"):
        
        macro_forecast: Dict[str, Mapping[str, float]] = macro.assign_macro_forecasts()

    N_BOOT = 30
    
    OUTER_JOBS = -1

    with LogTimer("Prepare per-ticker arrays"):
        
        per_ticker = prepare_per_ticker_arrays(
            panel_x = panel_x, 
            tickers = tickers
        )

    with LogTimer(f"Train bundles in parallel (n_jobs={OUTER_JOBS})"):
        
        with tqdm_joblib(tqdm(total = len(tickers), desc = "Train bundles", unit = "stk")):
            
            results = Parallel(n_jobs = OUTER_JOBS, prefer = "processes", backend = "loky", batch_size = 1, verbose = 0)(
                
                delayed(_train_one_ticker_arrays)(
                    tk,
                    *per_ticker[tk],                                  
                    macro_forecast.get(tk, {}),
                    N_BOOT,
                    42
                )
                
                for tk in tickers if tk in per_ticker
            )
            
    bundles = {tk: b for tk, b in results}

    with LogTimer("Build fin_fc_map"):
        
        fin_fc_map: Dict[str, Tuple[Mapping[str, float], Mapping[str, float]]] = {
            tk: fdata.get_forecast_pct_changes(ticker = tk) or ({}, {}) for tk in tickers
        }

    with LogTimer("Global MC prediction"):
        
        with tqdm_joblib(tqdm(total = len(bundles), desc = "MC per ticker", unit = "stk")):
            
            df_global_mc = global_mc_prediction(
                bundles = bundles,
                fin_fc_map = fin_fc_map,
                n_draws = 1000,
                random_state = 42,
                choose_random_model = True,
                n_jobs_mc = -1,
            )


    logger.info("\n%s", df_global_mc)

    with LogTimer("Export results"):
        
        out_file = Path(config.MODEL_FILE)
        
        sheets = {
            "hgb_cross_asset": df_global_mc
        }
        
        export_results(
            sheets = sheets, 
            output_excel_file = out_file
        )


if __name__ == "__main__":
    
    logger.info("Starting annual return modelling with cross-asset lags")
    
    main()
