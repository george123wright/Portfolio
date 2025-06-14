from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed
from sklearn.base import clone

from data_processing.financial_forecast_data import FinancialForecastData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fit_base_model_from_arrays(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_cols: List[str],
    *,
    n_splits: int = 4,
) -> HistGradientBoostingRegressor:
    """
    Run exactly the same GridSearchCV on (X_arr, y_arr) but avoiding pandas overhead.
    Return the fitted estimator (best_estimator_).
    """
    n_samples = X_arr.shape[0]
    cv = TimeSeriesSplit(n_splits=min(n_splits, max(2, n_samples - 1)))

    pipe = Pipeline([
        (
            "model",
            HistGradientBoostingRegressor(
                random_state=42,
                early_stopping=False,
                min_samples_leaf=1,
                max_depth=3,
            ),
        )
    ])

    param_grid = {
        "model__max_iter": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    }

    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1, 
    )
    gs.fit(X_arr, y_arr)
    return gs.best_estimator_

def _clean_df_once(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Clean out rows with NaN / ±Inf in either features or target.
    Return:
      - X_arr: numpy array of shape (n_valid, p)
      - y_arr: numpy array of shape (n_valid,)
      - feature_names: list of column names (p long)
    """
    X = df.drop(columns=["Return"])
    y = df["Return"]
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()

    X_clean_df = X.loc[mask, :]
    y_clean_ser = y.loc[mask]

    feature_names = list(X_clean_df.columns)
    X_arr = X_clean_df.to_numpy()      
    y_arr = y_clean_ser.to_numpy()     
    return X_arr, y_arr, feature_names


def bootstrap_models(
    base_model,
    data_hist: pd.DataFrame,
    feature_cols: List[str],
    *,
    n_resamples: int = 50
) -> List:
    """Return a list of models each fitted on a bootstrap resample."""
    models = []
    for _ in range(n_resamples):
        samp = data_hist.sample(frac=0.8, replace=True)
        Xs = samp.drop(columns=["Return"])
        ys = samp["Return"]

        Xs.replace([np.inf, -np.inf], np.nan, inplace=True)
        mask = Xs.notna().all(axis=1) & ys.notna()
        Xs, ys = Xs[mask], ys[mask]

        m = clone(base_model)
        m.fit(Xs[feature_cols], ys)
        models.append(m)
    return models

def _fit_one_bootstrap(
    base_model,
    X_feat: np.ndarray,
    y_arr: np.ndarray,
    random_state: int,
) -> HistGradientBoostingRegressor:
    """
    - Sample with replacement from X_feat, y_arr (NumPy arrays).
    - Fit a clone of base_model on that sample.
    - Return the fitted clone.
    """
    n_samples = X_feat.shape[0]
    rng = np.random.RandomState(random_state)
    idx = rng.randint(0, n_samples, size=n_samples)  

    X_samp = X_feat[idx, :]
    y_samp = y_arr[idx]

    m = clone(base_model)
    m.fit(X_samp, y_samp)
    return m

def bootstrap_models_from_arrays(
    base_model: HistGradientBoostingRegressor,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_cols: List[str],
    *,
    n_resamples: int = 50,
) -> List[HistGradientBoostingRegressor]:
    """
    - Extract X_feat once (shape [n_samples, p]) as X_arr[:, feature_indices].
    - Launch n_resamples parallel jobs, each calling _fit_one_bootstrap(...) with
      a different random_state seed.
    """
   
    p = X_arr.shape[1]  
    X_feat = X_arr[:, :]  

    seeds = list(range(n_resamples))
    models = Parallel(n_jobs=-1)(
        delayed(_fit_one_bootstrap)(base_model, X_feat, y_arr, seed)
        for seed in seeds
    )
    return models


def scenario_matrix(
    rev_covs: Dict[str, float],
    eps_covs: Dict[str, float],
    macro_fc: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return DataFrame (n_scenarios × p) with all scenario feature-vectors,
    plus a list of scenario keys.
    Vectorized via pandas.MultiIndex cartesian product.
    """
    rev_items = list(rev_covs.items())
    eps_items = list(eps_covs.items())

    rev_labels, _ = zip(*rev_items)
    eps_labels, _ = zip(*eps_items)

    idx = pd.MultiIndex.from_product(
        [rev_labels, eps_labels],
        names=["rev_lbl", "eps_lbl"]
    )
    df = pd.DataFrame(index=idx).reset_index()

    rev_map = dict(rev_items)
    eps_map = dict(eps_items)
    df["Revenue Growth"] = df["rev_lbl"].map(rev_map)
    df["EPS Growth"]     = df["eps_lbl"].map(eps_map)

    for col, val in macro_fc.items():
        df[col] = val

    keys = (df["rev_lbl"] + "_rev__" + df["eps_lbl"] + "_eps").tolist()

    scen_df = df.drop(columns=["rev_lbl", "eps_lbl"]).reset_index(drop=True)

    return scen_df, keys


def process_ticker(
    tk: str,
    growth: pd.DataFrame | str,
    macro_forecasts: Dict[str, pd.DataFrame],
    fin_forecast: Tuple[Dict[str, float], Dict[str, float]] | None = None,
    *,
    n_boot: int = 50,
) -> Dict:
    """Return results dictionary for a single ticker."""
    if isinstance(growth, str) or len(growth) < 3:
        logger.warning("Skip %s: Insufficient Growth Data, Returning Zeros", tk)
        return {
            "Ticker": tk,
            "Lowest Predicted Return": 0.0,
            "Mean Predicted Return": 0.0,
            "Highest Predicted Return": 0.0,
            "Return SE": 0.0,
        }

    X_arr, y_arr, feature_cols = _clean_df_once(growth)    
    base_model = fit_base_model_from_arrays(X_arr, y_arr, feature_cols)
    
    bs_models = bootstrap_models_from_arrays(
        base_model,
        X_arr,
        y_arr,
        feature_cols,
        n_resamples=n_boot
    )

    rev_covs, eps_covs = fin_forecast 
    scen_df, keys = scenario_matrix(rev_covs, eps_covs, macro_forecasts)

    scen_df = scen_df.reindex(columns=feature_cols, fill_value=0.0)
    scen_arr = scen_df.to_numpy() 
    
    preds_base = base_model.predict(scen_arr)
    preds_bs = np.vstack([m.predict(scen_arr) for m in bs_models]).T

    se_bs = preds_bs.std(ddof=1)
    scen_var = preds_base.var(ddof=1)
    se_final = np.sqrt(se_bs**2 + scen_var)

    min_ret = preds_base.min()
    max_ret = preds_base.max()
    avg_ret = preds_base.mean()

    logger.info(f" {tk}: Low: {min_ret:.4f},  Avg: {avg_ret:.4f},  High: {max_ret:.4f}, SE: {se_final:.4f}")

    return {
        "Ticker": tk,
        "Lowest Predicted Return": min_ret,
        "Mean Predicted Return": avg_ret,
        "Highest Predicted Return": max_ret,
        "Return SE": se_final,
    }


def main() -> None:

    fdata = FinancialForecastData(tickers)

    macro = fdata.macro
    r = macro.r
    tickers = r.tickers

    growth_hist = fdata.regression_dict()
    macro_forecast = macro.assign_macro_forecasts()

    results: List[Dict] = []
    for ticker in tickers:
        ticker_hist = growth_hist.get(ticker, "No Growth Data")
        ticker_forecast = fdata.get_forecast_pct_changes(ticker)
        ticker_macro_forecast = macro_forecast[ticker]
        res = process_ticker(
            ticker,
            ticker_hist,
            ticker_macro_forecast,
            ticker_forecast,
            n_boot=50,
        )
        if res:
            results.append(res)

    if not results:
        logger.warning("No results produced – exiting")
        return

    df = pd.DataFrame(results).set_index("Ticker")
    out_file = Path("/Users/georgewright/Portfolio_Optimisation_DCF.xlsx")
    with pd.ExcelWriter(out_file, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="Lin Reg Returns")


if __name__ == "__main__":
    logger.info("Starting lin_reg_returns10.py")
    main()
