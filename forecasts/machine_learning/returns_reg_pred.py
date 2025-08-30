"""
Annual return modelling with time-aware cross-validation, moving-block bootstrap,
and scenario evaluation via boosted trees (HGBR) with a ridge fallback.

Overview
--------
This module estimates a mapping from annual features to an annual equity return,
then evaluates forecast scenarios for revenue growth, EPS growth, and macro
changes. The primary estimator is a regularised histogram gradient boosting
regressor (HGBR), selected by time-series aware cross-validation. For small
samples a ridge regression fallback is used. Model uncertainty is assessed by a
circular moving-block bootstrap that preserves short-run dependence in annual
panels. Scenario uncertainty enters through a Cartesian product of revenue and
EPS growth cases overlaid with fixed macro forecasts. The predictive mixture is
the set of all bootstrap-by-scenario predictions; summary statistics comprise
the 10th, 50th, and 90th percentiles and the standard deviation.

Mathematical specification
--------------------------
Let { (x_t, y_t) } for t = 1,…,T denote an annual panel for a single ticker,
with y_t the target *Return* (e.g., percentage price change for the year) and
x_t ∈ R^p the feature vector containing, for example, "Revenue Growth",
"EPS Growth", and macro changes such as "InterestRate_pct_change",
"Inflation_rate", "GDP_growth", "Unemployment_pct_change".

1) Ridge fallback (used when T is very small):
  
   Estimate β ∈ R^p by minimising
  
       J(β; α) = (1 / T) * Σ_{t=1..T} (y_t − x_t^⊤ β)^2 + α * ||β||_2^2,
  
   with α selected by time-series cross-validation. Prediction is ŷ = X β̂.

2) Histogram Gradient Boosting Regressor (HGBR):
  
   Model f(x) as a stagewise sum of regression trees:
  
       f_0(x) = argmin_c Σ (y_t − c)^2  = mean(y),
  
       for m = 1..M:
  
           r_t^{(m)} = − ∂/∂f ½(y_t − f_{m−1}(x_t))^2 = y_t − f_{m−1}(x_t),
  
           fit tree h_m(x) to residuals r^{(m)} with depth and leaf constraints,
           update f_m(x) = f_{m−1}(x) + η * h_m(x).
  
   Here η ∈ (0,1] is the learning rate. HGBR uses histogram binning to form
   contiguous value buckets per feature and applies L2 regularisation on leaf
   values: for a leaf with n samples and residuals r_i, the leaf prediction is
  
       v_leaf = Σ r_i / (n + λ),
  
   where λ ≥ 0 is the regularisation strength. The final predictor is f_M(x).

3) Time-series cross-validation:
  
   Folds respect time ordering. For split k, the training indices precede the
   validation indices. The selected model minimises validation mean squared error:
  
       MSE = (1 / n_val) * Σ (y_val − ŷ_val)^2.

4) Circular moving-block bootstrap (time-aware resampling):
  
   For block size b and sample length T, draw starting points s_j ∼ Uniform{0,…,T−1}
   for j = 1,…,⌈T / b⌉, form blocks { (s_j + i) mod T : i = 0,…,b−1 }, concatenate
   to length T, and refit the estimator on the resampled sequence. This preserves
   short-run dependence up to lag ≈ b.

5) Scenario construction and predictive mixture:
  
   Let S be the set of scenarios formed by the Cartesian product of revenue and
   EPS labels (e.g., low/avg/high) with a fixed macro overlay. Each scenario s
   maps to a feature vector x_s constructed by strict alignment to the training
   features (drop extra columns; impute missing with training medians). For a set
   of bootstrap models { f^{(b)} }, the predictive mixture is
  
       Z = { f^{(b)}(x_s) : s ∈ S, b = 1,…,B }.
  
   Report:
  
       Low  = percentile_{10}(Z),
       Med  = percentile_{50}(Z),
       High = percentile_{90}(Z),
       SE   = sqrt( Σ (z_i − mean(Z))^2 / (|Z| − 1) ).

All equations are provided in plain text and UK spelling is used throughout.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Mapping, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed

from data_processing.financial_forecast_data import FinancialForecastData
from functions.export_forecast import export_results
import config


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s %(levelname)-8s %(message)s",
    datefmt = "%H:%M:%S",
)

logger = logging.getLogger(__name__)


def _min_train_tss(
    n_samples: int,
    n_splits: int,
    min_train: int = 6
) -> TimeSeriesSplit:
    """
    Construct a TimeSeriesSplit that guarantees at least `min_train` training
    observations in the first fold, subject to the total sample size.

    Behaviour
    ---------
    - If n_samples < (min_train + 2), the number of splits is reduced so that the
      first training window has at least `min_train` observations while leaving at
      least one observation for validation: splits = max(2, n_samples − min_train).
    - Otherwise, use `n_splits`.
    - Constrain to 2 ≤ splits ≤ n_samples − 1.

    Rationale
    ---------
    Time-series cross-validation requires training sets that precede validation
    sets. Ensuring a minimum training size in the first fold stabilises model
    selection under small samples.
    """

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
    Generate circular moving-block bootstrap indices of length n.

    Construction
    ------------
    Let b = clamp(block_size, lower = 1, upper = n). Draw the number of blocks
    as B = ceil(n / b). For each block j = 1..B, sample a start point s_j
    uniformly from {0,…,n−1}, then take indices
        
        { (s_j + i) mod n : i = 0,…,b−1 }.
    
    Concatenate blocks and truncate to the first n indices.

    Purpose
    -------
    The moving-block bootstrap preserves short-run dependence up to lag ≈ b in
    annual data, in contrast to i.i.d. resampling which breaks temporal structure.

    Returns
    -------
    numpy.ndarray
        Array of length n containing resampled indices in [0, n−1].
    """
    
    if n <= 1:

        return np.zeros(1, dtype=int)

    b = max(1, min(block_size, n))

    n_blocks = int(np.ceil(n / b))

    starts = rng.randint(0, n, size = n_blocks)
    
    idx = []
    
    for s in starts:
    
        block = (s + np.arange(b)) % n
    
        idx.append(block)
    
    idx = np.concatenate(idx)[:n]
    
    return idx


def _align_scenario_to_training(
    scen_df: pd.DataFrame, 
    feature_cols: List[str],
    train_medians: pd.Series
) -> pd.DataFrame:
    """
    Align scenario features to the training design strictly.

    Steps
    -----
    1) Drop any scenario columns not present in `feature_cols`.
   
    2) For each required feature c ∈ feature_cols that is missing, create it and
       fill with the training median train_medians[c] (default 0.0 if absent).
   
    3) Reorder columns to match `feature_cols` exactly.

    Rationale
    ---------
    Strict alignment prevents accidental leakage of unseen variables and ensures
    that the estimator receives features in the identical order and scale as used
    for training.
    """
    
    scen = scen_df.copy()

    scen = scen[[c for c in scen.columns if c in feature_cols]].copy()

    for c in feature_cols:

        if c not in scen.columns:

            scen[c] = train_medians.get(c, 0.0)

    scen = scen[feature_cols]

    return scen


def _clean_df_once(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series]:
    """
    Prepare a design matrix and target vector from an annual panel.

    Inputs
    ------
    df : DataFrame
        Must contain a target column named "Return"; all other columns are treated
        as candidate features. The index is expected to be time-ordered.

    Cleaning and outputs
    --------------------
    - Sort by index timestamp and drop duplicate timestamps (keep the last).
  
    - Replace ±∞ in features with NaN; drop any row with NaN in either features or target.
  
    - Return:
        X_arr         : numpy array of shape (n_valid, p) with cleaned features,
        y_arr         : numpy array of shape (n_valid,) with target,
        feature_names : list of p feature column names,
        train_medians : Series of per-feature medians used for scenario imputation.

    Mathematical target
    -------------------
    The target y_t is an annual return (e.g., arithmetic price change ratio) as
    supplied in the input frame; no transformation is applied here.
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

    X = df2.drop(columns=["Return"]).replace([np.inf, -np.inf], np.nan)
  
    y = df2["Return"]

    mask = X.notna().all(axis = 1) & y.notna()
  
    Xc = X.loc[mask]
  
    yc = y.loc[mask]

    feature_names = list(Xc.columns)
  
    train_medians = Xc.median(axis = 0)

    return Xc.to_numpy(), yc.to_numpy(), feature_names, train_medians


def _fit_ridge_fallback(
    X: np.ndarray, 
    y: np.ndarray, 
    n_splits: int = 3
) -> RidgeCV:
    """
    Fit a ridge regression with time-aware cross-validation for very small samples.

    Model
    -----
    Estimate β by minimising
      
        J(β; α) = (1 / n) * Σ_i (y_i − x_i^⊤ β)^2 + α * ||β||_2^2,
   
    over α from a log-spaced grid. The selected α minimises the validation mean
    squared error under TimeSeriesSplit.

    Returns
    -------
    RidgeCV
        Fitted ridge regressor with `.predict(X)` available.
    """
    
    alphas = np.logspace(-3, 3, 25)
    
    cv = _min_train_tss(
        n_samples = len(y), 
        n_splits = n_splits,
        min_train = min(6, max(3, len(y) // 2)))
    
    model = RidgeCV(alphas = alphas, cv = cv)
    
    model.fit(X, y)
    
    return model


def fit_base_model_from_arrays(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_cols: List[str],
    *,
    n_splits: int = 4,
    random_state: int = 42,
) -> object:
    """
    Fit a regularised histogram gradient boosting regressor with time-aware CV;
    fall back to ridge when the panel is too small.

    Decision rule
    -------------
    - If n_samples < 8, use the ridge fallback with a reduced number of splits.
  
    - Otherwise, perform grid-searched HGBR with TimeSeriesSplit and refit on the
      full sample using the best hyperparameters (minimum validation MSE).

    HGBR specification
    ------------------
    Objective: minimise squared error via stagewise additive trees:
  
        f_m(x) = f_{m−1}(x) + η * h_m(x),
  
    with depth constraints (`max_depth`), minimum samples per leaf
    (`min_samples_leaf`), L2 leaf regularisation (`l2_regularization`), a bounded
    number of histogram bins (`max_bins`), and iterations (`max_iter`).

    Cross-validation scoring
    ------------------------
    The grid search uses scoring = "neg_mean_squared_error"; the best estimator is
    the one with the largest (least negative) value, i.e., the smallest MSE.

    Returns
    -------
    object
        Fitted estimator with `.predict(X)` implemented (either HGBR in a Pipeline
        or a RidgeCV instance).
    """
    
    n_samples = X_arr.shape[0]
    
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

    pipe = Pipeline([
        ("model",
         HistGradientBoostingRegressor(
             random_state = random_state,
             early_stopping = False,  
         ))
    ])

    param_grid = {
        "model__max_iter": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_leaf": [5, 10],
        "model__l2_regularization": [0.0, 1.0],
        "model__max_bins": [64, 255],
    }

    gs = GridSearchCV(
        pipe,
        param_grid,
        cv = cv,
        scoring = "neg_mean_squared_error",
        n_jobs = -1,
        refit = True,
    )
    
    gs.fit(X_arr, y_arr)
    
    return gs.best_estimator_


def _fit_one_bootstrap(
    base_model,
    X_feat: np.ndarray,
    y_arr: np.ndarray,
    random_state: int,
    block_size: Optional[int] = None,
) -> object:
    """
    Fit one bootstrap replicate using a circular moving-block resample.

    Method
    ------
    - Choose block size b = max(2, min(√n, n)) if `block_size` is None.
   
    - Generate indices via `_block_bootstrap_indices(n, b, rng)`.
   
    - Clone the `base_model` (to avoid shared state), fit on the resampled (X_b, y_b),
      and return the fitted clone.

    Purpose
    -------
    Each replicate approximates the sampling distribution under serial dependence,
    providing model-uncertainty dispersion for scenario evaluation.
    """

    n = X_feat.shape[0]
   
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
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    n_resamples: int = 50,
    block_size: Optional[int] = None,
) -> List[object]:
    """
    Produce a time-aware bootstrap ensemble by fitting clones across resamples.

    Procedure
    ---------
    For seeds s = 0,…,n_resamples−1:
      1) Generate circular moving-block indices of length n.
      2) Fit a cloned estimator on the resampled data.
    The operation is parallelised across CPU cores.

    Returns
    -------
    list[object]
        List of fitted estimator clones suitable for `.predict`.
    """
    
    seeds = list(range(n_resamples))
    
    models = Parallel(n_jobs=-1)(
        delayed(_fit_one_bootstrap)(base_model, X_arr, y_arr, seed, block_size)
        for seed in seeds
    )
    
    return models


def scenario_matrix(
    rev_covs: Mapping[str, float],
    eps_covs: Mapping[str, float],
    macro_fc: Mapping[str, float],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct a Cartesian product of revenue and EPS scenarios with a macro overlay.

    Inputs
    ------
    rev_covs : mapping
        Keys label revenue scenarios (e.g., 'low_y','avg_y','high_y','low_fs',…),
        values are the associated revenue growth inputs (proportions or changes).
    eps_covs : mapping
        Keys label EPS scenarios; values are EPS growth inputs.
    macro_fc : mapping
        Fixed macro features for the forecast (e.g., 'InterestRate_pct_change').

    Outputs
    -------
    (scenario_df, scenario_keys)
        scenario_df has columns:
          - "Revenue Growth" and "EPS Growth" filled according to labels, and
          - macro columns copied from `macro_fc`.
        scenario_keys are composite identifiers of the form
          "<rev_label>_rev__<eps_label>_eps".

    Notes
    -----
    The resulting features are not yet aligned to the training design; alignment
    and imputation are performed by `_align_scenario_to_training`.
    """
    
    rev_items = list(rev_covs.items())
   
    eps_items = list(eps_covs.items())
   
    rev_labels, _ = zip(*rev_items)
   
    eps_labels, _ = zip(*eps_items)

    idx = pd.MultiIndex.from_product([rev_labels, eps_labels], names = ["rev_lbl", "eps_lbl"])
   
    df = pd.DataFrame(index = idx).reset_index()

    rev_map = dict(rev_items)
   
    eps_map = dict(eps_items)
   
    df["Revenue Growth"] = df["rev_lbl"].map(rev_map)
   
    df["EPS Growth"] = df["eps_lbl"].map(eps_map)

    for col, val in macro_fc.items():
   
        df[col] = val

    keys = (df["rev_lbl"] + "_rev__" + df["eps_lbl"] + "_eps").tolist()
   
    scen_df = df.drop(columns=["rev_lbl", "eps_lbl"]).reset_index(drop = True)
   
    return scen_df, keys


def process_ticker(
    tk: str,
    growth: pd.DataFrame | str,
    macro_forecasts: Mapping[str, float],
    fin_forecast: Tuple[Mapping[str, float], Mapping[str, float]] | None = None,
    *,
    n_boot: int = 50,
    random_state: int = 42,
) -> Dict:
    """
    End-to-end modelling for a single ticker: fit the base model, build a
    moving-block bootstrap ensemble, evaluate across revenue × EPS × macro
    scenarios, and aggregate to a predictive mixture.

    Pipeline
    --------
    1) Data cleaning: construct (X, y) from `growth` where y = "Return" and
       features are the remaining columns. Rows with NaN or ±∞ are dropped.

    2) Base model: fit via `fit_base_model_from_arrays`. If the sample is small,
       use ridge regression with time-series CV; otherwise perform grid search for
       a regularised HGBR.

    3) Bootstrap ensemble: fit `n_boot` clones using circular moving-block
       bootstrap indices of length n to preserve short-run dependence.

    4) Scenario features: build `scenario_df` via `scenario_matrix`, then apply
       strict alignment to the training features, imputing missing columns with
       training medians.

    5) Predictive mixture: for each bootstrap model m_b and scenario s, compute
       z_{b,s} = m_b(x_s). Let Z = { z_{b,s} } over all b and s.

    Reported statistics
    -------------------
    - Low Returns  : percentile_{10}(Z),
    - Returns      : percentile_{50}(Z) (median),
    - High Returns : percentile_{90}(Z),
    - SE           : sqrt( Σ (z_i − mean(Z))^2 / (|Z| − 1) ).

    Edge cases
    ----------
    If growth data are insufficient or no financial forecast is available,
    zeros are returned for all outputs with an explanatory log message.

    Returns
    -------
    dict
        {"Ticker", "Low Returns", "Returns", "High Returns", "SE"} for the ticker.
    """
    
    if isinstance(growth, str) or len(growth) < 3:
      
        logger.warning("Skip %s: Insufficient Growth Data, Returning Zeros", tk)
      
        return {"Ticker": tk, "Low Returns": 0.0, "Returns": 0.0, "High Returns": 0.0, "SE": 0.0}

    X_arr, y_arr, feature_cols, train_medians = _clean_df_once(
        df = growth
    )

    base_model = fit_base_model_from_arrays(
        X_arr = X_arr,
        y_arr = y_arr,
        feature_cols = feature_cols,
        n_splits = 4,
        random_state = random_state,
    )

    bs_models = bootstrap_models_from_arrays(
        base_model = base_model,
        X_arr = X_arr,
        y_arr = y_arr,
        n_resamples = n_boot,
        block_size = None,  
    )

    if fin_forecast is None:
     
        logger.warning("No financial forecast for %s; returning zeros.", tk)
     
        return {"Ticker": tk, "Low Returns": 0.0, "Returns": 0.0, "High Returns": 0.0, "SE": 0.0}

    rev_covs, eps_covs = fin_forecast
    
    scen_df, keys = scenario_matrix(
        rev_covs = rev_covs, 
        eps_covs = eps_covs,
        macro_fc = macro_forecasts
    )

    scen_df = _align_scenario_to_training(
        scen_df = scen_df,
        feature_cols = feature_cols, 
        train_medians = train_medians
    )
    
    scen_arr = scen_df.to_numpy()

    _ = base_model.predict(scen_arr)

    preds_bs = np.vstack([m.predict(scen_arr) for m in bs_models])

    pred_mix = preds_bs.reshape(-1)

    low, med, high = np.percentile(pred_mix, [10.0, 50.0, 90.0])
   
    avg = np.nanmean(pred_mix)
    
    if pred_mix.size > 1:
   
        se = pred_mix.std(ddof = 1)  
    
    else:
        
        se = 0.0

    logger.info("%s: Low %.4f, Med %.4f, Avg %.4f, High %.4f, SE %.4f", tk, low, med, avg, high, se)

    return {"Ticker": tk, "Low Returns": float(low), "Returns": float(med), "High Returns": float(high), "SE": float(se)}


def main() -> None:
    """
    Orchestrate data acquisition, per-ticker modelling, scenario evaluation, and
    aggregation of results.

    Steps
    -----
    1) Fetch annual growth/return panels via `FinancialForecastData.regression_dict()`.
       Each panel typically includes columns:
         {"Revenue Growth","EPS Growth","Return",
          "InterestRate_pct_change","Inflation_rate","GDP_growth","Unemployment_pct_change"}.
    
    2) Obtain macro forecast inputs per ticker via `macro.assign_macro_forecasts()`.
    
    3) For each ticker:
         a) Build revenue/EPS forecast-to-history changes via
            `FinancialForecastData.get_forecast_pct_changes(ticker)` producing
            (rev_covs, eps_covs).
         b) Call `process_ticker` to fit the model, bootstrap, construct the
            scenario matrix, and compute mixture percentiles and SE.
    
    4) Collect per-ticker outputs into a DataFrame and (optionally) write to disk.

    Outputs
    -------
    A DataFrame with index "Ticker" and columns:
      "Low Returns", "Returns" (median), "High Returns", "SE".

    Notes
    -----
    Logging emits progress and summaries. The commented Excel export section can
    be re-enabled to append results to `config.MODEL_FILE`.
    """
      
    fdata = FinancialForecastData()
   
    macro = fdata.macro
   
    tickers = config.tickers

    growth_hist = fdata.regression_dict()
    
    macro_forecast = macro.assign_macro_forecasts()

    results: List[Dict] = []
    
    for ticker in tickers:
    
        ticker_hist = growth_hist.get(ticker, "No Growth Data")
    
        ticker_forecast = fdata.get_forecast_pct_changes(
            ticker = ticker
        ) 
    
        ticker_macro_forecast = macro_forecast[ticker]  

        res = process_ticker(
            tk = ticker,
            growth = ticker_hist,
            macro_forecasts = ticker_macro_forecast,
            fin_forecast = ticker_forecast,
            n_boot = 50,
        )
        
        if res:
        
            results.append(res)

    if not results:
        
        logger.warning("No results produced – exiting")
        
        return

    df = pd.DataFrame(results).set_index("Ticker")
    
    out_file = Path(config.MODEL_FILE)
    
    sheets = {"HGB Returns": df}
    
    export_results(
        sheets = sheets,
        output_excel_file = out_file,
    )


if __name__ == "__main__":
   
    logger.info("Starting lin_reg_returns10_improved.py")
   
    main()
