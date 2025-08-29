"""
Prophet + residual ARMA–GARCH forecasting in **return space** with quarterly-to-weekly
feature engineering, macro PCA, publication lags, and price-space reconstruction.

How this version differs from the other Prophet model
--------------------------------------------------------
1) Return-space residual modelling and price reconstruction:
  
   - The other pipeline adjusted Prophet **level** predictions (price space) with ARIMA
     mean and GARCH volatility applied to Prophet residuals, then clipped price bands.
  
   - This version fits residual models on **log-return** residuals. Let L_t = log P_t,
     and r_t = ΔL_t = L_t − L_{t−1}. Prophet is fit to L_t, its increment r̂_t = Δ L̂_t
     forms Prophet step returns. Residual step returns e_r_t = r_t − r̂_t are modelled by
     ARMA and GARCH. Combined horizon step means and sigmas are **integrated** back to
     price: P̂_{t+h} = exp( L_{t} + sum_{i=1..h} μ_i ) with bands using cumulative sigma.

2) Quarterly fundamentals and macro to weekly features with lags:
  
   - Quarterly inputs (Revenue TTM, EPS TTM, macro levels) are converted to quarter-end,
     transformed into QoQ and YoY growth, shifted by reporting (publication) lags in
     weeks, and **broadcast** to a weekly grid using backward as-of matching.
  
   - The previous version fed weekly macro or level regressors directly.

3) Macro dimensionality reduction:
  
   - This version fits StandardScaler + PCA **inside each training fold** and transforms
     macro QoQ/YoY features into a small set of principal components Macro_PCk used as
     regressors. The earlier version used raw macro regressors.

4) Scenario construction for TTM fundamentals:
  
   - Revenue and EPS **TTM** paths are ramped linearly at quarter-end dates between the
     last known TTM and target scenario values, then converted to QoQ/YoY and lagged
     before weekly broadcasting. The previous model ramped daily/weekly regressors.

5) Frequency, lags, clipping, and CV:
  
   - Weekly frequency uses 'W-FRI'. Both macro and financial features honour explicit
     publication lags. Clipping is in **price space** using config.lbp and config.ubp
     relative to the forecast origin price. Cross-validation is implemented as a full
     rolling pipeline (`pipeline_cv_rmse`) that recomputes features and macro PCA per
     fold, and scores **price-space RMSE**.

Key equations (text form)
-------------------------

- Prophet (log-price): L_t = g(t) + s_y(t) + β' x_t + ε_t.

- Returns: r_t = ΔL_t = L_t − L_{t−1}; Prophet step returns r̂_t = Δ L̂_t.

- Return residual: e_r_t = r_t − r̂_t.

- ARMA(p,q) on e_r_t: e_r_t = Σ_{i=1..p} φ_i e_r_{t−i} + Σ_{j=1..q} θ_j η_{t−j} + η_t,
  with η_t white noise; AIC selects p,q.

- GARCH(1,1) on u_r_t (ARMA residual): σ_t^2 = ω + α u_{t−1}^2 + β σ_{t−1}^2.

- Prophet step sigma from interval: σ_p,step = (r̂_upper − r̂) / 1.96 (Normal).

- Combined step sigma: σ_step = sqrt( max(σ_p,step,0)^2 + max(σ_g,step,0)^2 ).

- Cumulative mean and sigma over horizon h:

    μ_cum(h) = Σ_{i=1..h} ( r̂_i + μ_ARMA,i ),

    σ_cum(h) = sqrt( Σ_{i=1..h} σ_step,i^2 ).

- Price reconstruction and bands:

    P̂_{t+h} = exp( L_t + μ_cum(h) ),

    PI_low = exp( L_t + μ_cum(h) − 1.96 σ_cum(h) ),

    PI_high = exp( L_t + μ_cum(h) + 1.96 σ_cum(h) ).
"""


import logging
from typing import List, Tuple, Dict, Iterable
import numpy as np
import pandas as pd
from itertools import product

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from functions.export_forecast import export_results
from data_processing.financial_forecast_data import FinancialForecastData
import config


REV_KEYS = ['low_rev_y', 'avg_rev_y', 'high_rev_y', 'low_rev', 'avg_rev', 'high_rev']

EPS_KEYS = ['low_eps_y', 'avg_eps_y', 'high_eps_y', 'low_eps', 'avg_eps', 'high_eps']

SCENARIOS = list(product(REV_KEYS, EPS_KEYS))

MACRO_COLS = ['Interest', 'Cpi', 'Gdp', 'Unemp']

FIN_TTM_COLS = ['Revenue', 'EPS (Basic)'] 

FREQ_WEEK = 'W-FRI'

PUB_LAG_WEEKS_FIN = 1

PUB_LAG_WEEKS_MACRO = 1


def configure_logger() -> logging.Logger:
    """
    Configure a module logger for informational progress and diagnostics.

    Behaviour
    ---------
    - Sets the logger level to INFO.
    - Adds a StreamHandler with format "%(asctime)s - %(levelname)s - %(message)s".
    - Avoids duplicate handlers if called multiple times.

    Returns
    -------
    logging.Logger
        The configured logger bound to this module's name.
    """
       
    logger = logging.getLogger(__name__)
   
    logger.setLevel(logging.INFO)
   
    if not logger.handlers:
   
        ch = logging.StreamHandler()
   
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
   
        logger.addHandler(ch)
   
    return logger


logger = configure_logger()


def _to_quarter_end(
    df: pd.DataFrame, 
    date_col: str, 
    cols: List[str]
) -> pd.DataFrame:
    """
    Collapse arbitrary-dated rows to quarter-end observations for selected columns.

    Method
    ------
  
    1) Coerce `date_col` to datetime and sort ascending.
  
    2) Index by PeriodIndex at quarterly frequency; within each quarter take the **last**
       row for each requested column.
  
    3) Convert the PeriodIndex to timestamps at quarter **end** (how='end').

    Parameters
    ----------
    df : DataFrame
        Source frame with at least `date_col` and the requested `cols`.
    date_col : str
        Column name containing dates.
    cols : list of str
        Columns to retain.

    Returns
    -------
    DataFrame
        Columns ['ds'] + cols with ds at quarter ends.
    """

    d = df.copy()

    d[date_col] = pd.to_datetime(d[date_col])

    d = d.sort_values(date_col)

    q = d.set_index(pd.PeriodIndex(d[date_col], freq = 'Q'))[cols].groupby(level = 0).last()

    q = q.copy()

    q['ds'] = q.index.to_timestamp(how='end')   

    q = q.reset_index(drop = True)

    return q[['ds'] + cols]



def _qoq_yoy(
    qdf: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """
    Compute quarter-over-quarter and year-over-year percentage changes.

    Definitions
    -----------
    For a quarterly series x_t sampled at quarter ends:
  
      QoQ change:   qoq_t = (x_t − x_{t−1}) / |x_{t−1}|
  
      YoY change:   yoy_t = (x_t − x_{t−4}) / |x_{t−4}|
  
    where division by zero is avoided by pandas NA semantics.

    Parameters
    ----------
    qdf : DataFrame
        Quarterly time series with a 'ds' column and the input `cols`.
    cols : list of str
        Column names for which to compute QoQ and YoY.

    Returns
    -------
    DataFrame
        Columns: 'ds' plus f'{col}_qoq' and f'{col}_yoy' for each input column.
    """
    
    qdf = qdf.sort_values('ds').reset_index(drop = True)

    out = qdf[['ds']].copy()
   
    for c in cols:
   
        out[f'{c}_qoq'] = qdf[c].pct_change(1)
   
        out[f'{c}_yoy'] = qdf[c].pct_change(4)
   
    return out


def _apply_pub_lag(
    qfeat: pd.DataFrame, 
    weeks_lag: int
) -> pd.DataFrame:
    """
    Shift quarterly features forward by a publication lag measured in weeks.

    Rationale
    ---------
    Quarterly fundamentals or macro releases become observable to markets with delay.
    A lag of L weeks moves the effective date by +7L days, ensuring causal usage in
    weekly forecasting.

    Parameters
    ----------
    qfeat : DataFrame
        Quarterly features with 'ds' at quarter end.
    weeks_lag : int
        Number of weeks to shift forward.

    Returns
    -------
    DataFrame
        Same structure with 'ds' shifted by the lag.
    """
    
    if weeks_lag <= 0:

        return qfeat

    q = qfeat.copy()

    q['ds'] = q['ds'] + pd.to_timedelta(7 * weeks_lag, unit = 'D')

    return q


def _broadcast_quarterly_to_weekly(
    weekly_ds: pd.Series,
    qfeat: pd.DataFrame
) -> pd.DataFrame:
    """
    Broadcast quarter-end features to a weekly grid by backward as-of matching.

    Method
    ------
    Given weekly dates w_i and quarterly feature dates q_j, assign to each w_i the
    most recent q_j ≤ w_i (direction='backward'). This holds features constant within
    the quarter (post-publication) until the next release.

    Parameters
    ----------
    weekly_ds : pandas.Series
        Weekly timestamps (target grid).
    qfeat : DataFrame
        Quarterly features with 'ds' and numeric columns.

    Returns
    -------
    DataFrame
        Weekly 'ds' with quarterly features carried forward.
    """
    
    w = pd.DataFrame({'ds': pd.to_datetime(weekly_ds)}).sort_values('ds')

    feat = pd.merge_asof(w, qfeat.sort_values('ds'), on = 'ds', direction = 'backward')

    return feat


def _lag_quarters(
    qfeat: pd.DataFrame, 
    lag_q: int
) -> pd.DataFrame:
    """
    Lag quarterly features by an integer number of quarters.

    Parameters
    ----------
    qfeat : DataFrame
        Quarterly feature frame with 'ds'.
    lag_q : int
        Number of quarters to shift (e.g., 1 for one-quarter lag).

    Returns
    -------
    DataFrame
        Same columns with non-'ds' columns shifted by `lag_q` rows.
    """
    
    if lag_q <= 0:
        
        return qfeat
  
    q = qfeat.copy()
  
    for c in q.columns:
  
        if c != 'ds':
  
            q[c] = q[c].shift(lag_q)
  
    return q


def _macro_cols_from_df(
    df: pd.DataFrame, 
    macro_names = ('Interest', 'Cpi', 'Gdp', 'Unemp'),
    suffixes = ('_qoq','_yoy')
) -> List[str]:
    """
    Collect available macro feature column names following a naming convention.

    Convention
    ----------
    Macro base names in `macro_names` combined with suffixes in `suffixes` produce
    candidate names like 'Interest_qoq', 'Cpi_yoy', etc. Only existing columns are
    returned.

    Returns
    -------
    list[str]
        Present macro columns to be used for PCA or as regressors.
    """
    
    cand = []

    for m in macro_names:

        for s in suffixes:

            col = f'{m}{s}'

            if col in df.columns:

                cand.append(col)

    return cand

def fit_macro_pca(
    df_train: pd.DataFrame, 
    macro_cols: List[str],
    n_components: int = 3
):
    """
    Fit StandardScaler + PCA on the **training** macro block and transform the frame.

    Mathematics
    -----------
  
    - Let X be the training macro matrix with columns macro_cols.
  
    - Standardisation: Z = (X − μ) / σ, where μ and σ are column means and std devs.
  
    - PCA: find orthonormal loadings W ∈ R^{d×k} maximising explained variance of Z.
      Principal components are PC = Z W. The k components are chosen with
      k = min(n_components, d).

    Outputs
    -------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted on non-missing rows of X.
    pca : sklearn.decomposition.PCA
        Fitted on scaled macro data.
    pc_colnames : list[str]
        ['Macro_PC1', ..., 'Macro_PCk'].
    train_with_pcs : DataFrame
        df_train with macro_cols dropped and PC columns appended.

    Notes
    -----
    Missing macro entries are imputed with the scaler column means before transformation.
    If there are insufficient rows or columns, returns the input frame unchanged.
    """

    if not macro_cols:
     
        return None, None, [], df_train

    X = df_train[macro_cols].astype(float)
  
    mask = X.notna().all(axis=1)
  
    X_fit = X.loc[mask]

    if X_fit.shape[1] == 0 or X_fit.shape[0] < 10:
  
        return None, None, [], df_train

    scaler = StandardScaler().fit(X_fit)
  
    k = min(n_components, X_fit.shape[1])
  
    pca = PCA(n_components = k, random_state = 0).fit(scaler.transform(X_fit))

    X_full = X.fillna(pd.Series(scaler.mean_, index = macro_cols))
   
    Z = pca.transform(scaler.transform(X_full))
   
    pc_cols = [f'Macro_PC{i+1}' for i in range(pca.n_components_)]
   
    pcs_df = pd.DataFrame(Z, columns=pc_cols, index = df_train.index)

    df_out = pd.concat([df_train.drop(columns = macro_cols), pcs_df], axis = 1)
   
    return scaler, pca, pc_cols, df_out


def transform_macro_with_pca(
    df_macro_features: pd.DataFrame,
    macro_cols: List[str],
    scaler: StandardScaler,
    pca: PCA
) -> pd.DataFrame:
    """
    Apply a previously fitted scaler and PCA to a macro feature frame.

    Steps
    -----
    1) Select macro_cols and fill missing entries with scaler means.

    2) Standardise using scaler; project onto PCA loadings to obtain Macro_PCk.

    3) Return a frame with columns ['ds','Macro_PC1',...].

    Returns
    -------
    DataFrame
        Projection of macro features to principal component space, aligned on 'ds'.

    Caution
    -------
    The scaler and PCA must come from the **train** window to avoid look-ahead bias.
    """
    
    if scaler is None or pca is None or not macro_cols:
     
        return df_macro_features[['ds']].copy()

    X = df_macro_features[macro_cols].astype(float)
   
    X_full = X.fillna(pd.Series(scaler.mean_, index = macro_cols))
   
    Z = pca.transform(scaler.transform(X_full))
   
    pc_cols = [f'Macro_PC{i+1}' for i in range(pca.n_components_)]
   
    pcs = pd.DataFrame(Z, columns=pc_cols)
   
    pcs.insert(0, 'ds', df_macro_features['ds'].values)
   
    return pcs


def build_quarterly_features_for_weekly(
    weekly_ds: pd.Series,
    quarterly_df: pd.DataFrame,
    cols: List[str],
    publication_lag_weeks: int,
    eq: Iterable[int] = ()
) -> pd.DataFrame:
    """
    Produce weekly QoQ/YoY features from a quarterly source with publication lag.

    Pipeline
    --------
    1) Collapse input to quarter end via `_to_quarter_end`.
   
    2) Compute QoQ and YoY percentage changes via `_qoq_yoy`.
   
    3) Apply quarter lag(s) given by `eq` (e.g., [1,2]) to create lagged predictors.
   
    4) Shift by `publication_lag_weeks` to mimic reporting delay.
   
    5) Broadcast to a weekly grid by backward as-of merge.

    Returns
    -------
    DataFrame
        Weekly 'ds' plus f'{col}_qoq' and f'{col}_yoy' (and their lags).
    """
    
    if quarterly_df is None or quarterly_df.empty:
      
        return pd.DataFrame({'ds': pd.to_datetime(weekly_ds)})

    q = _to_quarter_end(
        df = quarterly_df, 
        date_col = 'ds', 
        cols = cols
    )
  
    qchg = _qoq_yoy(
        qdf = q, 
        cols = cols
    )
   
    for lag in eq:
  
        qchg = _lag_quarters(
            qfeat = qchg,
            lag_q = lag
        )

    qchg = _apply_pub_lag(
        qfeat = qchg, 
        weeks_lag = publication_lag_weeks
    )

    return _broadcast_quarterly_to_weekly(
        weekly_ds = weekly_ds,
        qfeat = qchg
    )


def _interp_path_weekly(
    h_idx: np.ndarray,
    arr: np.ndarray,
    H: int
) -> np.ndarray:
    """
    Linearly interpolate a path of length H on horizon indices 1..H from sparse points.

    Definition
    ----------
    If arr = [v1, v2, ..., vL] with L ≥ 2, construct a grid x = linspace(1, H, L)
    and define v(h) by linear interpolation at integer h ∈ {1,...,H}. Returns None
    if arr is empty or all NaN.

    Returns
    -------
    numpy.ndarray or None
        Interpolated horizon path or None if not feasible.
    """
    
    if arr is None or len(arr) <= 1 or np.all(np.isnan(arr)):

        return None

    L = len(arr)

    x = np.linspace(1, H, L)

    return np.interp(h_idx, x, arr)


def _future_quarter_grid(
    last_qe: pd.Timestamp,
    horizon_weeks: int
) -> pd.DatetimeIndex:
    """
    Construct future quarter-end timestamps covering a weekly horizon.

    Method
    ------
    - Start at the quarter **after** `last_qe` and go to last_qe + horizon_weeks.
    - Return quarter ends with frequency 'Q' as timestamps.

    Returns
    -------
    pandas.DatetimeIndex
        Quarter-end dates used for future TTM ramps and change calculations.
    """
    
    start = (pd.Period(last_qe, freq = 'Q') + 1).to_timestamp(how = 'end')

    end = pd.to_datetime(last_qe) + pd.to_timedelta(7 * horizon_weeks, unit = 'D')

    q = pd.period_range(start = start, end = end, freq = 'Q')

    return q.to_timestamp(how = 'end')


def build_future_macro_features(
    weekly_all_ds: pd.Series,
    horizon_mask: np.ndarray,
    last_hist_qe: pd.Timestamp,
    macro_arrays: Dict[str, np.ndarray],
    pub_lag_weeks: int
) -> pd.DataFrame:
    """
    Create weekly **future** macro QoQ/YoY features from sparse horizon arrays.

    Steps
    -----
    1) For each macro key in {'InterestRate','Consumer_Price_Index_Cpi','GDP','Unemployment'}:
   
         a) Interpolate a weekly horizon path v(h) on h = 1..H via `_interp_path_weekly`.
   
         b) Build a horizon-only frame with 'ds' and the raw level series.
   
         c) Collapse to quarter end, apply `pub_lag_weeks`, and compute QoQ/YoY changes.
   
         d) Broadcast back to the weekly grid and merge into the output.
   
    2) Only columns present are produced; missing arrays are ignored.

    Returns
    -------
    DataFrame
        Weekly 'ds' plus macro QoQ/YoY features for the horizon (history will be NaN).
    """
      
    future = pd.DataFrame({'ds': pd.to_datetime(weekly_all_ds)}).sort_values('ds')
  
    H = int(np.sum(horizon_mask))
  
    h_idx = np.arange(1, H + 1)
  
    out = future[['ds']].copy()
  
    for key, col in [('InterestRate', 'Interest'), ('Consumer_Price_Index_Cpi', 'Cpi'), ('GDP', 'Gdp'), ('Unemployment', 'Unemp')]:
        
        arr = macro_arrays.get(key, None)
        
        v = _interp_path_weekly(
            h_idx = h_idx, 
            arr = np.asarray(arr) if arr is not None else None, 
            H = H
        )
        
        if v is None:
        
            continue
        
        wf = future.loc[horizon_mask, ['ds']].copy()
        
        wf[col] = v
        
        qlvl = _to_quarter_end(
            df = wf,
            date_col = 'ds', 
            cols = [col]
        )
        
        qlvl = _apply_pub_lag(
            qfeat = qlvl, 
            weeks_lag = pub_lag_weeks
        )
        
        qchg = _qoq_yoy(
            qdf = qlvl, 
            cols = [col]
        )
        
        feat = _broadcast_quarterly_to_weekly(
            weekly_ds = future['ds'], 
            qfeat = qchg
        )
        
        out = out.merge(feat, on = 'ds', how = 'left')
    
    return out


def _rescale_for_sarimax(
    y: pd.Series
) -> tuple[pd.Series, float]:
    """
    Rescale a series to improve SARIMAX optimisation stability.

    Rule
    ----
    Let s = median absolute value of y (ignoring NaNs). Define a scale factor
    k = clamp( 1 / s, lower = 1, upper = 1e6 ). Return y' = k * y and k.
    If s is not finite or s = 0, return y unchanged and scale 1.

    Rationale
    ---------
    Rescaling reduces pathologies in likelihood optimisation when residuals are very
    small in magnitude. The stored scale is later inverted on forecasts.
    """
    
    y = pd.Series(y).astype(float).dropna()

    s = np.nanmedian(np.abs(y))

    if not np.isfinite(s) or s == 0:

        return y, 1.0

    scale = max(1.0, 1.0 / s)

    scale = min(scale, 1e6)

    return y * scale, scale


def prepare_prophet_model(
    df_model: pd.DataFrame, 
    regressors: List[str]
) -> Prophet:
    """
    Fit a Prophet model on log-prices with user-supplied regressors.

    Specification
    -------------
    - Target: 
    
        y = log price; 
        
        time index 'ds'; 
        
        regressors supplied in `regressors`.
   
    - Additive structure: 
    
        y_t = g(t) + s_y(t) + β' x_t + ε_t,

      with yearly seasonality enabled; change-point prior 0.05; range 0.9.
   
    - Each regressor is standardised (z-score) and regularised by a Gaussian prior
      on its coefficient: 
      
        β_r ~ Normal(0, 0.1^2) (prior_scale = 0.1).

    Returns
    -------
    Prophet
        Fitted model ready for future construction and prediction.
    """
    
    m = Prophet(
        changepoint_prior_scale = 0.05,
        changepoint_range = 0.9,
        daily_seasonality = False,
        weekly_seasonality = False,
        yearly_seasonality = True
    )
    
    for r in regressors:
    
        m.add_regressor(r, standardize = True, prior_scale = 0.1)
    
    m.fit(df_model[['ds', 'y'] + regressors])
    
    return m


def _fit_garch(
    resid_after_arima: pd.Series
):
    """
    Fit a zero-mean GARCH(1,1) with Student-t innovations to return residuals.

    Model
    -----
    Let u_t denote ARMA residuals of step returns. The conditional variance follows:
   
      σ_t^2 = ω + α u_{t−1}^2 + β σ_{t−1}^2,
   
    with
    
        u_t = σ_t z_t and z_t ~ Student-t(ν). 
        
    Mean is constrained to zero.
    
    Persistence condition α + β < 1 implies covariance stationarity.

    Requirements
    ------------
    At least 200 observations are required (guard against small samples).

    Returns
    -------
    arch.univariate.base.ARCHModelResult or None
        Fitted result or None upon failure.
    """
    
    z = pd.Series(resid_after_arima).astype(float).dropna()

    if len(z) < 200:

        return None

    try:

        am = arch_model(z, mean = 'Zero', vol = 'GARCH', p = 1, q = 1, dist = 't', rescale = False)

        return am.fit(disp = 'off')

    except Exception:

        return None


def _forecast_arima(
    res, 
    steps: int
) -> np.ndarray:
    """
    Produce ARMA mean forecasts for step-return residuals over the next `steps`.

    Details
    -------
   
    - If `res` was fit on rescaled data y' = k y, the stored scale k is inverted:
   
      μ = predicted_mean / k.
   
    - NaNs are replaced with zero and sequences are padded as needed.

    Returns
    -------
    numpy.ndarray
        Length-`steps` array of mean adjustments μ_i to be added to Prophet step returns.
    """

    if res is None or steps <= 0:

        return np.zeros(steps, dtype = float)

    try:
      
        fc = res.get_forecast(steps = steps)
      
        mu = np.asarray(fc.predicted_mean)
      
        scale = getattr(res, "_y_scale", 1.0)
      
        mu = mu / (scale if np.isfinite(scale) and scale > 0 else 1.0)
      
        if mu.shape[0] < steps:
      
            mu = np.pad(mu, (0, steps - mu.shape[0]))
      
        return np.nan_to_num(mu, nan = 0.0)[:steps]
    
    except Exception:
    
        return np.zeros(steps, dtype = float)


def _forecast_garch(
    garch_res, 
    steps: int
) -> np.ndarray:
    """
    Generate GARCH conditional standard deviations for the next `steps`.

    Method
    ------
    Use `garch_res.forecast(horizon=steps, reindex=False)` to obtain forecast
    variances, take square roots, forward-fill if shorter than `steps`, and replace
    NaNs with zero.

    Returns
    -------
    numpy.ndarray
        Length-`steps` array of σ_g,i for step-return uncertainty.
    """
    
    if garch_res is None or steps <= 0:

        return np.zeros(steps, dtype = float)

    try:

        f = garch_res.forecast(horizon = steps, reindex = False)

        var = np.asarray(f.variance.values)[-1]

        sigma = np.sqrt(np.maximum(var, 0.0))

        if sigma.shape[0] < steps:

            sigma = np.pad(sigma, (0, steps - sigma.shape[0]), constant_values = sigma[-1] if sigma.size else 0.0)

        return np.nan_to_num(sigma, nan = 0.0)[:steps]

    except Exception:

        return np.zeros(steps, dtype = float)


def clip_to_bounds(
    df: pd.DataFrame, 
    price: float
) -> pd.DataFrame:
    """
    Constrain price forecasts to a multiplicative band around the origin price.

    Rule
    ----
    For columns {yhat, yhat_lower, yhat_upper} apply:
   
      lower = config.lbp * price,
   
      upper = config.ubp * price,
   
    and clip values to [lower, upper].

    Rationale
    ---------
    Guards against unrealistic extrapolation in volatile regimes or sparse data.

    Returns
    -------
    DataFrame
        The same frame with clipped price predictions (returned for chaining).
    """
    
    lower = config.lbp * price
   
    upper = config.ubp * price
   
    cols = ['yhat', 'yhat_lower', 'yhat_upper']
   
    df.loc[:, cols] = df.loc[:, cols].clip(lower = lower, upper = upper)
   
    return df


def fit_residual_models(
    prophet_model: Prophet,
    df_model: pd.DataFrame, 
    regressors: List[str]
):
    """
    Fit ARMA and GARCH models on **return** residuals derived from Prophet's log-price fit.

    Construction
    ------------
   
    - Let L_t be the observed log price and L̂_t the Prophet fitted log price.
      Step returns are 
      
        r_t = ΔL_t and r̂_t = ΔL̂_t.
      
      Return residuals: 
      
        e_r_t = r_t − r̂_t.
   
    - ARMA(p,q) on e_r_t with p,q ∈ {0,1,2} selected by minimum AIC:
   
        e_r_t = Σ_{i=1..p} φ_i e_r_{t−i} + Σ_{j=1..q} θ_j η_{t−j} + η_t.
   
      Fitting uses SARIMAX with trend='n' and relaxed stationarity/invertibility.
      The series is rescaled by k = max(1, 1 / median|e_r|) (capped at 1e6) for stability;
      the scale is stored and undone in forecasting.
   
    - GARCH(1,1) with Student-t innovations is fit to the ARMA residuals u_r_t to
      model conditional variance:
   
        σ_t^2 = ω + α u_{t−1}^2 + β σ_{t−1}^2.

    Returns
    -------
    (arima_res, garch_res, e_r, u_r)
        arima_res : statsmodels result or None (ARMA mean on e_r)
        garch_res : arch result or None (GARCH on ARMA residuals)
        e_r       : pandas.Series of return residuals
        u_r       : pandas.Series of ARMA residuals (input to GARCH)

    Purpose
    -------
    The ARMA forecast adjusts Prophet's step-return mean; GARCH widens uncertainty
    where residual heteroskedasticity is present.
    """

    ins  = prophet_model.predict(df_model[['ds'] + regressors].copy())

    L = pd.Series(df_model['y'].values, index = df_model['ds'])   

    Lhat = pd.Series(ins['yhat'].values, index = df_model['ds'])    

    r = L.diff()
   
    r_hat = Lhat.diff()
   
    rr = pd.concat([r, r_hat], axis = 1, keys = ['r', 'r_hat']).dropna()
   
    e_r = (rr['r'] - rr['r_hat']).astype(float).dropna()


    def _choose_sarimax_returns(
        resid: pd.Series,
        p_grid = (0, 1, 2),
        q_grid = (0, 1, 2)
    ):
    
        y = pd.Series(resid).astype(float).dropna()
    
        y_s, y_scale = _rescale_for_sarimax(
            y = y
        )    
    
        best_aic = np.inf
        
        best_res = None
    
        for p in p_grid:
    
            for q in q_grid:
    
                try:
    
                    mod = SARIMAX(
                        endog = y_s, 
                        order = (p, 0, q), 
                        trend = 'n',
                        enforce_stationarity = False,
                        enforce_invertibility = False,
                        dates = y.index, 
                        freq = FREQ_WEEK
                    )
    
                    res = mod.fit(disp = False, method = 'lbfgs', maxiter = 400)
    
                    res._y_scale = y_scale
    
                    if res.aic < best_aic:
    
                        best_aic = res.aic
                        
                        best_res = res
    
                except Exception:
    
                    continue
    
        return best_res


    arima_res = _choose_sarimax_returns(
        resid = e_r
    )

    if arima_res is not None:
      
        fitted = pd.Series(arima_res.fittedvalues, index = e_r.index)
       
        u_r = (e_r - fitted).dropna()
    
    else:
    
        u_r = e_r.copy()

    garch_res = _fit_garch(
        resid_after_arima = u_r
    )

    return arima_res, garch_res, e_r, u_r


def build_model_frame(
    price_series: pd.Series,
    fin_q_df: pd.DataFrame,
    macro_q_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], float]:
    """
    Construct the modelling frame with log price and weekly QoQ/YoY features.

    Steps
    -----
   
    1) Price: L_t = log P_t on a weekly 'ds' index; drop NaNs.
   
    2) Financial features: from `fin_q_df` build weekly f'{col}_qoq' and f'{col}_yoy'
       with publication lag and optional quarter lags.
   
    3) Macro features: similarly from `macro_q_df`.
   
    4) Merge price with features, keep rows with non-missing y and regressors.

    Outputs
    -------
    df_model : DataFrame
        Columns: 'ds', 'y' = log price, and weekly QoQ/YoY regressors.
    reg_cols : list[str]
        Names of regressor columns (those ending with '_qoq' or '_yoy').
    last_price : float
        Last observed spot price (for subsequent clipping and diagnostics).
    """
        
    dfp = pd.DataFrame({'ds': pd.to_datetime(price_series.index), 'price': price_series.values}).sort_values('ds')
    
    dfp['y'] = np.log(dfp['price'])        
    
    dfp = dfp.dropna().reset_index(drop = True)

    fin_feat = build_quarterly_features_for_weekly(
        weekly_ds = dfp['ds'], 
        quarterly_df = fin_q_df[['ds'] + FIN_TTM_COLS], 
        cols = FIN_TTM_COLS, 
        publication_lag_weeks = PUB_LAG_WEEKS_FIN, 
        eq = [1]
    )
    
    mac_feat = build_quarterly_features_for_weekly(
        weekly_ds = dfp['ds'], 
        quarterly_df = macro_q_df[['ds'] + MACRO_COLS], 
        cols = MACRO_COLS, 
        publication_lag_weeks = PUB_LAG_WEEKS_MACRO, 
        eq = [1, 2]
    )

    df_model = dfp.merge(fin_feat, on = 'ds', how = 'left').merge(mac_feat, on = 'ds', how = 'left')
   
    reg = [c for c in df_model.columns if c.endswith('_qoq') or c.endswith('_yoy')]
   
    df_model = df_model.dropna(subset=['y'] + reg).reset_index(drop = True)
   
    last_price = dfp['price'].iloc[-1]
   
    return df_model, reg, last_price


def forecast_returns_to_prices(
    model: Prophet,
    df_hist: pd.DataFrame,
    reg_cols: List[str],
    future_reg_df: pd.DataFrame,
    arima_res=None,
    garch_res=None,
    last_spot: float = None,   
) -> pd.DataFrame:
    """
    Forecast in **return space** and reconstruct prices and intervals over the horizon.

    Procedure and equations
    -----------------------
    1) Build the full future frame (history + horizon) with regressors forward/back-filled.
   
    2) Prophet predictions on log price: obtain { L̂_t, L̂_upper_t } for all t.
   
    3) Prophet step returns:
   
         r̂_t = L̂_t − L̂_{t−1},
   
         r̂_upper_t = L̂_upper_t − L̂_{t−1}.
   
       Step sigma from Prophet interval (Normal):
   
         σ_p,step_t = max( (r̂_upper_t − r̂_t) / 1.96, 0 ).
   
    4) Residual ARMA and GARCH forecasts over H steps:
   
         μ_ARMA,i = _forecast_arima(arima_res, H)[i],
   
         σ_g,i     = _forecast_garch(garch_res, H)[i].
   
       Combined step sigma in quadrature:
   
         σ_step,i = sqrt( max(σ_p,step_i,0)^2 + max(σ_g,i,0)^2 ).
   
    5) Cumulative mean and sigma:
   
         μ_cum(h) = Σ_{i=1..h} ( r̂_i + μ_ARMA,i ),
   
         σ_cum(h) = sqrt( Σ_{i=1..h} σ_step,i^2 ).
   
    6) Price reconstruction from last observed log price L_T:
   
         P̂_{T+h}     = exp( L_T + μ_cum(h) ),
   
         PI_low(h)    = exp( L_T + μ_cum(h) − 1.96 σ_cum(h) ),
   
         PI_high(h)   = exp( L_T + μ_cum(h) + 1.96 σ_cum(h) ).
   
       Insert these into the forecast frame at horizon rows; null out history rows.
   
    7) Clip to [config.lbp * last_spot, config.ubp * last_spot] in price space.

    Parameters
    ----------
    last_spot : float
        Spot price at the forecast origin used for price-space clipping.

    Returns
    -------
    DataFrame
        Prophet output with 'yhat','yhat_lower','yhat_upper' replaced by price-space
        reconstructions over the horizon and clipped bands; adds 'yhat_log_adj' for
        adjusted horizon log path.
    """
    
    if last_spot is None or not np.isfinite(last_spot):
     
        raise ValueError("last_spot must be provided and finite for price reconstruction/clipping.")

    future = model.make_future_dataframe(
        periods = future_reg_df.shape[0] - df_hist.shape[0],
        freq = FREQ_WEEK,
        include_history = True
    )
    
    future = future.merge(future_reg_df, on = 'ds', how = 'left')
  
    future[reg_cols] = future[reg_cols].ffill().bfill()

    fc = model.predict(future[['ds'] + reg_cols])

    hmask = future['ds'] > df_hist['ds'].max()
   
    H = int(hmask.sum())
   
    if H <= 0:
   
        return fc

    L_hat_all = fc['yhat'].to_numpy()
   
    L_up_all = fc['yhat_upper'].to_numpy()
   
    r_hat_all = L_hat_all[1:] - L_hat_all[:-1]
   
    r_up_all = L_up_all[1:] - L_up_all[:-1]
   
    sig_p_all = np.maximum((r_up_all - r_hat_all) / 1.96, 0.0)  

    hmask_r = hmask.to_numpy()[1:] 

    r_prop_h = r_hat_all[hmask_r]
   
    sig_p_h = sig_p_all[hmask_r]

    mu_r  = _forecast_arima(
        res = arima_res, 
        steps = H
    )       
   
    sig_g = _forecast_garch(
        garch_res = garch_res, 
        steps = H
    )         

    sig_step = np.sqrt(np.maximum(sig_p_h, 0.0) ** 2 + np.maximum(sig_g, 0.0) ** 2)

    cum_mu = np.cumsum(r_prop_h + mu_r)
    
    cum_sig = np.sqrt(np.cumsum(sig_step ** 2))

    last_logp = float(df_hist['y'].iloc[-1])
  
    P_mid = np.exp(last_logp + cum_mu)
  
    P_low = np.exp(last_logp + cum_mu - 1.96 * cum_sig)
  
    P_high = np.exp(last_logp + cum_mu + 1.96 * cum_sig)

    idx_h = np.where(hmask.to_numpy())[0]
  
    fc.loc[fc.index[idx_h], 'yhat'] = P_mid
  
    fc.loc[fc.index[idx_h], 'yhat_lower'] = P_low
  
    fc.loc[fc.index[idx_h], 'yhat_upper'] = P_high

    fc.loc[fc.index[idx_h], 'yhat_log_adj'] = last_logp + cum_mu

    fc.loc[~hmask, ['yhat','yhat_lower','yhat_upper', 'yhat_log_adj']] = np.nan

    fc = clip_to_bounds(
        df = fc,
        price = last_spot
    )
    
    return fc


def build_future_financial_features_ttm(
    weekly_all_ds: pd.Series,
    horizon_mask: np.ndarray,
    last_hist_qe: pd.Timestamp,
    last_known_ttm: Dict[str, float],
    rev_target: float,
    eps_target: float,
    pub_lag_weeks: int
) -> pd.DataFrame:
    """
    Create weekly future **TTM** financial features (QoQ/YoY) under target scenarios.

    Steps
    -----
    1) Generate a future quarter-end grid covering the weekly horizon.
    
    2) For each TTM column in {'Revenue','EPS (Basic)'}:
    
         a) Determine start (last known TTM) and end (scenario target).
    
         b) Form a linear ramp on quarter ends between start and end.
    
         c) Apply publication lag in weeks.
    
         d) Convert the ramped TTM levels into QoQ and YoY percentage changes.
    
         e) Broadcast to weekly dates via backward as-of.
    
    3) Merge all financial features on weekly 'ds'.

    Returns
    -------
    DataFrame
        Weekly features for financial QoQ and YoY under the scenario.
    """
    
    if not any(np.isfinite(v) for v in last_known_ttm.values()) and  not np.isfinite(rev_target) and not np.isfinite(eps_target):
   
        return pd.DataFrame({'ds': pd.to_datetime(weekly_all_ds)})

    future = pd.DataFrame({'ds': pd.to_datetime(weekly_all_ds)}).sort_values('ds')

    H = int(np.sum(horizon_mask))

    out = future[['ds']].copy()

    qh = _future_quarter_grid(
        last_qe = last_hist_qe,
        horizon_weeks = H
    )

    if len(qh) == 0:

        return out

    for col, end_val in [('Revenue', rev_target), ('EPS (Basic)', eps_target)]:
    
        start_val = last_known_ttm.get(col, np.nan)
    
        if not np.isfinite(start_val) and not np.isfinite(end_val):
    
            continue
    
        if not np.isfinite(start_val):
    
            start_val = end_val
    
        if not np.isfinite(end_val):
    
            end_val = start_val
    
        q_ttm = pd.DataFrame({'ds': qh, col: np.linspace(start_val, end_val, len(qh))})
    
        q_ttm = _apply_pub_lag(
            qfeat = q_ttm, 
            weeks_lag = pub_lag_weeks
        )
    
        qchg = _qoq_yoy(
            qdf = q_ttm, 
            cols = [col]
        )
    
        feat = _broadcast_quarterly_to_weekly(
            weekly_ds = future['ds'],
            qfeat = qchg
        )
    
        out = out.merge(feat, on = 'ds', how = 'left')

    return out


def rolling_cutoffs(
    ds: pd.Series,
    initial: str,
    period: str, 
    horizon: str
) -> List[pd.Timestamp]:
    """
    Generate rolling cutoffs for expanding-window cross-validation.

    Definition
    ----------
    Let t_min = min(ds), t_max = max(ds).
    The first cutoff is t_min + initial, and the last allowable cutoff is
    t_max − horizon. Cutoffs are spaced by `period`. If the first cutoff
    is not before the last allowable cutoff, return a single cutoff at t_max − horizon.

    Returns
    -------
    list[pandas.Timestamp]
        Sequence of cutoff timestamps.
    """
    
    ds = pd.to_datetime(ds).sort_values()

    start  = ds.min() + pd.to_timedelta(initial)

    last_c = ds.max() - pd.to_timedelta(horizon)

    if start >= last_c:

        return [last_c]

    return list(pd.date_range(start, last_c, freq = period))


def pipeline_cv_rmse(
    df_full: pd.DataFrame,         
    mac_q_df: pd.DataFrame,           
    fin_q_df: pd.DataFrame,      
    make_model_fn,                    
    fit_residuals_fn,                
    forecast_fn,                     
    initial: str, period: str, horizon: str,
    reg_cols_all: List[str],
    freq: str = FREQ_WEEK,
    clip_price_at_cutoff: bool = True,
) -> float:
    """
    End-to-end **price-space** RMSE cross-validation of the full pipeline, per fold.

    For each cutoff c:
   
      1) Split TRAIN = {ds ≤ c}, TEST = {c < ds ≤ c + horizon}; TRAIN must be non-empty.
   
      2) Build realised TRAIN features:
   
           - Macro: quarterly to weekly QoQ/YoY with publication lags and lags eq=[1,2].
   
           - Financial: similar for TTM with eq=[1].
   
      3) Determine which macro columns exist in this fold; fit StandardScaler + PCA
         on the TRAIN macro block only. Transform TRAIN to Macro_PCk.
   
      4) Select fold regressors = Macro_PCk + available financial QoQ/YoY columns.
   
      5) Fit Prophet on TRAIN with those regressors.
   
      6) Fit residual models (ARMA on return residuals; GARCH on ARMA residuals).
   
      7) Build FUTURE realised frames for TEST horizon, transform with the **fold**
         scaler and PCA, and assemble a complete future regressor frame (history PCs
         plus horizon PCs and financials).
   
      8) Forecast with `forecast_fn` (which reconstructs price and clips).
   
      9) Compute RMSE in **price space** over TEST:
   
            RMSE = sqrt( mean( (P_pred − P_true)^2 ) ).

    Returns
    -------
    float
        Pipeline RMSE (price units) averaged over all folds; NaN if no valid folds.

    Notes
    -----
    Fitting the scaler and PCA within each TRAIN fold prevents look-ahead bias.
    """
    
    cutoffs = rolling_cutoffs(
        ds = df_full['ds'],
        initial = initial, 
        period = period, 
        horizon = horizon
    )
    
    if not cutoffs:
    
        return np.nan

    errs, cnt = 0.0, 0
    
    df_full = df_full.sort_values('ds').reset_index(drop = True)

    for co in cutoffs:
      
        train_y = df_full[df_full['ds'] <= co][['ds','y']].copy()
      
        test = df_full[(df_full['ds'] > co) & (df_full['ds'] <= co + pd.to_timedelta(horizon))][['ds','y']].copy()
      
        if train_y.empty or test.empty:
      
            continue

        hist_mac_raw = build_quarterly_features_for_weekly(
            weekly_ds = train_y['ds'],
            quarterly_df = mac_q_df[['ds'] + MACRO_COLS],
            cols = MACRO_COLS, 
            publication_lag_weeks = PUB_LAG_WEEKS_MACRO,
            eq = [1,2]
        )
        
        hist_fin = build_quarterly_features_for_weekly(
            weekly_ds = train_y['ds'],
            quarterly_df = fin_q_df[['ds'] + FIN_TTM_COLS] if (fin_q_df is not None and not fin_q_df.empty) else pd.DataFrame(),
            cols = FIN_TTM_COLS,
            publication_lag_weeks = PUB_LAG_WEEKS_FIN, 
            eq = [1]
        )

        fold_macro_cols = _macro_cols_from_df(
            df = hist_mac_raw,
            macro_names = MACRO_COLS, 
            suffixes = ('_qoq','_yoy')
        )

        train_for_pca = train_y.merge(hist_mac_raw, on = 'ds', how = 'left').merge(hist_fin, on = 'ds', how = 'left')

        scaler, pca, pc_cols, train_with_pcs = fit_macro_pca(
            df_train = train_for_pca, 
            macro_cols = fold_macro_cols,
            n_components = 3
        )

        fin_cols_present = [c for c in train_with_pcs.columns
                            if any(c.startswith(x) for x in FIN_TTM_COLS)
                            and (c.endswith('_qoq') or c.endswith('_yoy'))]

        fold_reg_cols = pc_cols + fin_cols_present
        
        if not fold_reg_cols:

            fold_reg_cols = []

        m = make_model_fn(train_with_pcs[['ds','y'] + fold_reg_cols], fold_reg_cols)

        arima_res, garch_res, _, _ = fit_residuals_fn(m, train_with_pcs[['ds','y'] + fold_reg_cols], fold_reg_cols)

        future_all = m.make_future_dataframe(periods = len(test), include_history = True, freq = freq)
       
        hmask = (future_all['ds'] > train_y['ds'].max()).values

        fut_mac_raw = build_quarterly_features_for_weekly(
            weekly_ds = future_all['ds'],
            quarterly_df = mac_q_df[['ds'] + MACRO_COLS],
            cols = MACRO_COLS,
            publication_lag_weeks = PUB_LAG_WEEKS_MACRO,
            eq = [1, 2]
        )
        
        fut_fin = build_quarterly_features_for_weekly(
            weekly_ds = future_all['ds'],
            quarterly_df = fin_q_df[['ds'] + FIN_TTM_COLS] if (fin_q_df is not None and not fin_q_df.empty) else pd.DataFrame(),
            cols = FIN_TTM_COLS, 
            publication_lag_weeks = PUB_LAG_WEEKS_FIN,
            eq = [1]
        )

        fut_mac_raw = fut_mac_raw[['ds'] + [c for c in fold_macro_cols if c in fut_mac_raw.columns]]

        fut_mac_pcs = transform_macro_with_pca(
            df_macro_features = fut_mac_raw, 
            macro_cols = fold_macro_cols,
            scaler = scaler, 
            pca = pca
        )

        hist_pcs_only = train_with_pcs[['ds'] + fold_reg_cols]   
        
        fut_realized = fut_mac_pcs.merge(fut_fin, on = 'ds', how = 'left')
        
        future_reg_df = pd.concat([hist_pcs_only, fut_realized.loc[hmask, fold_reg_cols + ['ds']]], axis = 0, ignore_index = True)
        
        future_reg_df = future_reg_df.drop_duplicates('ds').sort_values('ds').reset_index(drop = True)

        last_spot = float(np.exp(train_y['y'].iloc[-1]))

        fc = forecast_fn(
            model = m,
            df_hist = train_with_pcs[['ds','y'] + fold_reg_cols],
            reg_cols = fold_reg_cols,
            future_reg_df = future_reg_df[['ds'] + fold_reg_cols],
            arima_res = arima_res,
            garch_res = garch_res,
            last_spot = last_spot,
        )

        comp = fc.merge(test[['ds','y']], on = 'ds', how = 'inner')
       
        if comp.empty:
       
            continue

        y_true = np.exp(comp['y'].to_numpy())
      
        y_pred = comp['yhat'].to_numpy()
      
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
      
        if mask.any():
      
            errs += np.sum((y_pred[mask] - y_true[mask]) ** 2)
      
            cnt += mask.sum()

    return float(np.sqrt(errs / cnt)) if cnt > 0 else np.nan


def main() -> None:
    """
    Orchestrate data preparation, model fitting, residual modelling, CV scoring,
    scenario forecasting, aggregation, and export.

    Outline
    -------
    - Load weekly closes and the latest prices.

    - Build per-ticker quarterly macro and financial tables; produce weekly modelling
      frames with QoQ/YoY features and macro PCs.
 
    - Fit Prophet on log prices with regressors; fit ARMA–GARCH on return residuals.
 
    - Score the **full pipeline** RMSE in price space via `pipeline_cv_rmse`.
 
    - For production horizon:
 
        * Generate macro future PCs from horizon arrays and publication lags.
 
        * If sufficient financial history exists, iterate over revenue/EPS TTM scenarios
          and forecast; else produce a macro-only baseline.
 
        * Reconstruct price bands from step-return forecasts and clip in price space.
 
    - Aggregate scenario end-point statistics:
        min, mean, max price; average return = mean price / spot − 1.
 
    - Combine band-implied sigma with CV RMSE in quadrature to obtain a final SE:
 
        SE_final = sqrt( band_sigma^2 + rmse_cv^2 ),
 
      where band_sigma = (max − min) / (2 * 1.96 * spot).
 
    - Export the per-ticker summary to Excel.

    Logging
    -------
    Emits per-ticker Low, Avg, High, Returns, and SE.

    Side effects
    ------------
    Writes sheet "Prophet Pred" to `config.MODEL_FILE`.
    """
       
    fdata = FinancialForecastData()
   
    macro = fdata.macro
   
    r = macro.r

    tickers = config.tickers
   
    forecast_weeks = 52
   
    cv_initial = f"{forecast_weeks * 3} W"
   
    cv_period = f"{int(forecast_weeks * 0.5)} W"
   
    cv_horizon = f"{forecast_weeks} W"

    close = r.weekly_close
   
    latest_prices = r.last_price

    next_fc = fdata.next_period_forecast() 
   
    next_macro_dict = macro.macro_forecast_dict()

    raw_macro = macro.assign_macro_history_non_pct()
   
    macro_history = (
        raw_macro.reset_index().rename(columns={'year': 'ds'})[['ticker', 'ds'] + MACRO_COLS]
    )
   
    if pd.api.types.is_period_dtype(macro_history['ds']):
   
        macro_history['ds'] = macro_history['ds'].dt.to_timestamp()
   
    else:
   
        macro_history['ds'] = pd.to_datetime(macro_history['ds'])
   
    macro_history = macro_history.sort_values(['ticker', 'ds'])
    
    macro_groups = macro_history.groupby('ticker')

    fin_raw: Dict[str, pd.DataFrame] = fdata.prophet_data  
    
    fin_proc: Dict[str, pd.DataFrame] = {}
    
    for tk in tickers:
    
        df = fin_raw.get(tk, pd.DataFrame()).reset_index().rename(columns = {'index': 'ds', 'rev': 'Revenue', 'eps': 'EPS (Basic)'})
    
        if 'ds' in df.columns:
    
            df['ds'] = pd.to_datetime(df['ds'])
    
        fin_proc[tk] = df.sort_values('ds')

    min_price = {}
    
    max_price = {}
    
    avg_price = {}
    
    avg_ret = {}
    
    se = {}
   
    final_rmse = {}
   
    logger.info("Running models ...")

    for tk in tickers:

        logger.info("Ticker: %s", tk)

        spot = latest_prices.get(tk, np.nan)

        if not np.isfinite(spot):

            logger.warning("No price for %s", tk)

            continue

        if tk not in macro_groups.groups:
     
            logger.warning("No macro history for %s", tk)
      
            continue
      
        tm_hist = macro_groups.get_group(tk).drop(columns = 'ticker').copy()

        fd = fin_proc.get(tk, pd.DataFrame()).copy()

        price_ser = close[tk].dropna()
      
        if price_ser.empty:
      
            logger.warning("No price history for %s", tk)
      
            continue

        if fd.empty:
      
            fin_q_df = pd.DataFrame({'ds': [], 'Revenue': [], 'EPS (Basic)': []})
      
        else:
      
            fin_q_df = _to_quarter_end(
                df = fd, 
                date_col = 'ds', 
                cols = FIN_TTM_COLS
            )  

        mac_q_df = _to_quarter_end(
            df = tm_hist, 
            date_col = 'ds', 
            cols = MACRO_COLS
        )

        df_model, reg_cols, last_price = build_model_frame(
            price_series = price_ser, 
            fin_q_df = fin_q_df, 
            macro_q_df = mac_q_df
        )
        
        if df_model.empty:
        
            logger.warning("Insufficient data for %s", tk)
        
            continue
        
        macro_cols = _macro_cols_from_df(
            df = df_model, 
            macro_names = MACRO_COLS, 
            suffixes = ('_qoq','_yoy')
        )

        macro_scaler, macro_pca, macro_pc_cols, df_model = fit_macro_pca(
            df_train = df_model, 
            macro_cols = macro_cols,
            n_components = 3
        )

        fin_cols = [c for c in df_model.columns
                    if any(c.startswith(x) for x in FIN_TTM_COLS)
                    and (c.endswith('_qoq') or c.endswith('_yoy'))]

        reg_cols = macro_pc_cols + fin_cols
        
        q_lo = df_model[reg_cols].quantile(0.01)

        q_hi = df_model[reg_cols].quantile(0.99)


        def clamp_future_feats(
            df: pd.DataFrame
            ) -> pd.DataFrame:

            df = df.copy()

            for c in reg_cols:

                if c in df:

                    df[c] = df[c].clip(lower = q_lo[c], upper = q_hi[c])

            return df
        

        try:
      
            m = prepare_prophet_model(
                df_model = df_model, 
                regressors = reg_cols
            )
      
        except Exception as e:
      
            logger.error("Prophet fit failed for %s: %s", tk, e)
      
            continue

        try:
            
            arima_res, garch_res, _, _ = fit_residual_models(
                prophet_model = m, 
                df_model = df_model, 
                regressors = reg_cols
            )
     
        except Exception as e:
     
            logger.warning("Residual models failed for %s: %s", tk, e)
     
            arima_res = None
     
            garch_res = None

        try:

            rmse_full = pipeline_cv_rmse(
                df_full = df_model[['ds','y'] + reg_cols], 
                mac_q_df = mac_q_df, fin_q_df=fin_q_df,
                make_model_fn = prepare_prophet_model,
                fit_residuals_fn = fit_residual_models,     
                forecast_fn = forecast_returns_to_prices,   
                initial = cv_initial, 
                period = cv_period,
                horizon = cv_horizon,
                reg_cols_all = reg_cols,
                freq = FREQ_WEEK
            )
            
            final_rmse[tk] = rmse_full   

        except Exception:
           
            final_rmse[tk] = np.nan

        macro_forecasts = next_macro_dict.get(tk, {})
     
        int_array = np.array(macro_forecasts.get('InterestRate', [np.nan]))
     
        cpi_array = np.array(macro_forecasts.get('Consumer_Price_Index_Cpi', [np.nan]))
     
        gdp_array = np.array(macro_forecasts.get('GDP', [np.nan]))
     
        unemp_array = np.array(macro_forecasts.get('Unemployment', [np.nan]))
     
        macro_arrays = {
            'InterestRate': int_array,
            'Consumer_Price_Index_Cpi': cpi_array,
            'GDP': gdp_array,
            'Unemployment': unemp_array
        }

        H = forecast_weeks
     
        future_all = m.make_future_dataframe(
            periods = H,
            include_history = True, 
            freq = FREQ_WEEK
        )
     
        horizon_mask = (future_all['ds'] > df_model['ds'].max()).values

        last_known_ttm = {}
       
        for col in FIN_TTM_COLS:
       
            if not fin_q_df.empty:
       
                last_known_ttm[col] = fin_q_df[col].dropna().iloc[-1]
       
            else:
       
                last_known_ttm[col] = np.nan
                        
        hist_fin = build_quarterly_features_for_weekly(
            weekly_ds = df_model['ds'],
            quarterly_df = fin_q_df[['ds'] + FIN_TTM_COLS] if not fin_q_df.empty else pd.DataFrame(),
            cols = FIN_TTM_COLS,
            publication_lag_weeks = PUB_LAG_WEEKS_FIN, 
            eq = [1]
        )

        hist_mac_raw = build_quarterly_features_for_weekly(
            df_model['ds'],
            mac_q_df[['ds'] + MACRO_COLS],
            MACRO_COLS, PUB_LAG_WEEKS_MACRO, eq=[1,2]
        )
        
        hist_mac_raw = hist_mac_raw[['ds'] + [c for c in macro_cols if c in hist_mac_raw.columns]]
        
        hist_mac_pcs = transform_macro_with_pca(
            df_macro_features = hist_mac_raw, 
            macro_cols = macro_cols,
            scaler = macro_scaler,
            pca = macro_pca
        )

        hist_all = df_model[['ds']].merge(hist_mac_pcs, on = 'ds', how = 'left').merge(hist_fin, on = 'ds', how = 'left')

        yf = future_all['ds']
      
        fut_mac_raw = build_future_macro_features(
            weekly_all_ds = yf,
            horizon_mask = horizon_mask,
            last_hist_qe = pd.to_datetime(yf.iloc[-1]),
            macro_arrays = macro_arrays,
            pub_lag_weeks = PUB_LAG_WEEKS_MACRO
        )
        
        fut_mac_raw = fut_mac_raw[['ds'] + [c for c in macro_cols if c in fut_mac_raw.columns]]
        
        fut_mac_pcs = transform_macro_with_pca(
            df_macro_features = fut_mac_raw,
            macro_cols = macro_cols, 
            scaler = macro_scaler,
            pca = macro_pca
        )

        has_financials = not fin_q_df.empty and (len(fin_q_df) >= 5)  

        results = []

        if has_financials:
           
            for rev_key, eps_key in SCENARIOS:
                
                if tk in next_fc.index and rev_key in next_fc.columns:
                    
                    rev_target = next_fc.at[tk, rev_key]  
                    
                else:
                    
                    rev_target = np.nan

                if tk in next_fc.index and eps_key in next_fc.columns:
                   
                    eps_target = next_fc.at[tk, eps_key]
                    
                else:
                   
                    eps_target = np.nan

                fut_fin = build_future_financial_features_ttm(
                    weekly_all_ds = yf,
                    horizon_mask = horizon_mask,
                    last_hist_qe = pd.to_datetime(yf.iloc[-1]),
                    last_known_ttm = last_known_ttm,
                    rev_target = rev_target,
                    eps_target = eps_target,
                    pub_lag_weeks = PUB_LAG_WEEKS_FIN
                )

                future_reg_df = pd.concat([
                    hist_all,
                    fut_mac_pcs.merge(fut_fin, on = 'ds', how = 'left').loc[horizon_mask, :]
                ], axis = 0, ignore_index = True).drop_duplicates('ds').sort_values('ds').reset_index(drop = True)

                future_reg_df = future_reg_df[['ds'] + reg_cols]
                
                future_reg_df = clamp_future_feats(future_reg_df)

                try:
                    
                    fc = forecast_returns_to_prices(
                        model = m,
                        df_hist = df_model,
                        reg_cols = reg_cols,
                        future_reg_df = future_reg_df,
                        arima_res = arima_res,
                        garch_res = garch_res,
                        last_spot = spot,
                    )
                    last = fc.dropna(subset = ['yhat']).iloc[-1]
               
                    results.append({
                        'Ticker': tk,
                        'Scenario': f'{rev_key}|{eps_key}',
                        'yhat': last['yhat'],
                        'yhat_lower': last['yhat_lower'],
                        'yhat_upper': last['yhat_upper']
                    })
         
                except Exception as e:
           
                    logger.error("Scenario failed %s %s %s", tk, (rev_key, eps_key), e)
       
        else:

            fut_fin_empty = pd.DataFrame({'ds': yf})

            future_reg_df = pd.concat([
                hist_all,
                fut_mac_pcs.merge(fut_fin_empty, on = 'ds', how = 'left').loc[horizon_mask, :]
            ], axis = 0, ignore_index = True).drop_duplicates('ds').sort_values('ds').reset_index(drop = True)

            future_reg_df = future_reg_df[['ds'] + reg_cols]

            future_reg_df = clamp_future_feats(
                df = future_reg_df
            )

            try:
                
                fc = forecast_returns_to_prices(
                    model = m,
                    df_hist = df_model,
                    reg_cols = reg_cols,
                    future_reg_df = future_reg_df,
                    arima_res = arima_res,
                    garch_res = garch_res,
                    last_spot = spot,
                )
                
                last = fc.dropna(subset=['yhat']).iloc[-1]
                
                results.append({
                    'Ticker': tk,
                    'Scenario': 'BASE_MACRO_ONLY',
                    'yhat': last['yhat'],
                    'yhat_lower': last['yhat_lower'],
                    'yhat_upper': last['yhat_upper']
                })

            except Exception as e:
         
                logger.error("Scenario failed %s %s", tk, e)

        if not results:
        
            logger.warning("No scenarios for %s", tk)
        
            continue

        scen = pd.DataFrame(results).set_index('Ticker')
      
        yvals = scen['yhat'].values
      
        lows = scen['yhat_lower'].values
      
        highs = scen['yhat_upper'].values

        if lows.size:
            
            min_price[tk] = np.nanmin(lows)  
        
        else:
            
            min_price[tk] = np.nan
            
        if highs.size:
            
            max_price[tk] = np.nanmax(highs)
            
        else:
            
            max_price[tk] = np.nan
            
        if yvals.size:
            
            avg_price[tk] = np.nanmean(yvals)
            
        else:
            
            avg_price[tk] = np.nan

        if np.isfinite(spot) and spot > 0:
      
            avg_price[tk] = np.nanmean(yvals)
      
            avg_ret[tk] = avg_price[tk] / spot - 1.0
      
        else:
      
            avg_ret[tk] = np.nan

    rmse_series = pd.Series(final_rmse).dropna()
    
    if not rmse_series.empty:
        
        max_rmse = rmse_series.max()
    
    else:
        
        max_rmse = np.nan
   
    for tk in tickers:
      
        spot = latest_prices.get(tk, np.nan)
      
        if not np.isfinite(spot) or spot <= 0:
      
            se[tk] = np.nan
      
            continue
      
        if tk in avg_price and tk in min_price and tk in max_price and np.isfinite(avg_price[tk]):
      
            band = (max_price[tk] - min_price[tk]) / (2 * 1.96 * spot)
      
        else:
      
            band = np.nan
      
        fr = final_rmse.get(tk, np.nan)
      
        use_rmse = fr if np.isfinite(fr) else (max_rmse if np.isfinite(max_rmse) else 0.0)
     
        if np.isfinite(band):
      
            se[tk] = float(np.sqrt(band ** 2 + (use_rmse ** 2)))
      
        else:
      
            se[tk] = float(use_rmse)

        logger.info(
            "Ticker: %s, Low: %.2f, Avg: %.2f, High: %.2f, Returns: %.4f, SE: %.4f",
            tk, min_price.get(tk, np.nan), avg_price.get(tk, np.nan), max_price.get(tk, np.nan),
            avg_ret.get(tk, np.nan), se.get(tk, np.nan)
        )

    prophet_results = pd.DataFrame({
        'Ticker': tickers,
        'Current Price': [latest_prices.get(tk, np.nan) for tk in tickers],
        'Avg Price': [avg_price.get(tk, np.nan) for tk in tickers],
        'Low Price': [min_price.get(tk, np.nan) for tk in tickers],
        'High Price': [max_price.get(tk, np.nan) for tk in tickers],
        'Returns': [avg_ret.get(tk, np.nan) for tk in tickers],
        'SE': [se.get(tk, np.nan) for tk in tickers]
    }).set_index('Ticker')

    sheets_to_write = {"Prophet PCA": prophet_results}
   
    export_results(
        output_excel_file = config.MODEL_FILE,
        sheets = sheets_to_write, 
    )
   
    logger.info("Done.")


if __name__ == "__main__":
    
    main()
