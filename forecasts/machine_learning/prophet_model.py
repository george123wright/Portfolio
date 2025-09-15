"""
Hybrid Prophet + ARIMA–GARCH equity price forecasting with macro and financial regressors.

Overview
--------
This module implements a hybrid forecasting pipeline that models weekly equity prices
using Meta Prophet with exogenous regressors, then captures residual mean
dynamics via ARIMA and residual conditional volatility via GARCH. Forecasts are
generated under multiple fundamental scenarios (revenue/EPS targets), macro paths
are interpolated over the forecast horizon, and predictive intervals are widened by
combining Prophet and GARCH uncertainty. Rolling cross-validation provides out-of-sample
error estimates which are merged with scenario dispersion to form a final standard error.

Core model components
---------------------
1) Prophet with regressors:
    
    y_t = g(t) + s_y(t) + β' x_t + ε_t,
   
   where g(t) is a piecewise-linear trend with changepoints, s_y(t) is yearly seasonality,
   x_t are exogenous regressors (macro and financial), β are coefficients with Gaussian
   priors N(0, prior_scale^2) on standardised regressors, and ε_t is i.i.d. noise
   within Prophet’s additive error model.

2) ARIMA on Prophet residuals:
    e_t = y_t − ŷ_t^Prophet.
   
   Select d ∈ {0,1} using the Augmented Dickey–Fuller test on levels. 
   
   Fit ARIMA(p,d,q) with small grid over p,q to minimise AIC = 2k − 2 log L. This captures 
   short-run mean reversion or autocorrelation left by the Prophet stage.

3) GARCH on ARIMA residuals:
   
    u_t = e_t − ê_t^ARIMA.
   
   Model conditional variance with zero-mean GARCH(1,1):
    
    σ_t^2 = ω + α u_{t−1}^2 + β σ_{t−1}^2, with Student-t innovations.
   
   Stationarity requires α + β < 1; the one-step conditional sigma enters the
   predictive interval combination.

Hybrid forecast combination
---------------------------
Let μ_h be the ARIMA mean forecast for horizon h, and σ_g,h the GARCH conditional
sigma. For Prophet’s horizon predictions (ŷ_h, ŷ_h,lower, ŷ_h,upper), convert its
interval to an implied sigma σ_p,h via half-width / 1.96 under a Normal assumption.
Adjust the mean and fuse uncertainties as:
  
   ŷ_h,hyb       = ŷ_h + μ_h
  
   σ_comb,h      = sqrt( max(σ_p,h,0)^2 + max(σ_g,h,0)^2 )
  
   PI_h,lower    = ŷ_h,hyb − 1.96 σ_comb,h
  
   PI_h,upper    = ŷ_h,hyb + 1.96 σ_comb,h

Scenario aggregation and final error
------------------------------------
Across revenue/EPS scenarios, compute end-horizon ŷ, lower, upper. Convert to returns
by dividing by the current price and subtracting 1. The scenario standard deviation
of returns provides a dispersion measure. Cross-validated RMSE from Prophet (scaled
by price) is combined in quadrature:
  
   SE_final = sqrt( scenario_se^2 + rmse_cv^2 ).

If rmse_cv is missing, substitute the sample maximum across tickers as a conservative proxy.

All equations are in plain text, with UK spelling (e.g., normalise).
"""


import logging
from typing import List, Tuple, Union, Dict
import numpy as np
import pandas as pd
from itertools import product

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

from export_forecast import export_results
from data_processing.financial_forecast_data import FinancialForecastData
import config


REV_KEYS = ['low_rev_y', 'avg_rev_y', 'high_rev_y', 'low_rev', 'avg_rev', 'high_rev']

EPS_KEYS = ['low_eps_y', 'avg_eps_y', 'high_eps_y', 'low_eps', 'avg_eps', 'high_eps']

SCENARIOS = list(product(REV_KEYS, EPS_KEYS))

MACRO_REGRESSORS = ['Interest', 'Cpi', 'Gdp', 'Unemp']

FIN_REGRESSORS = ['Revenue', 'EPS (Basic)']

ALL_REGRESSORS = MACRO_REGRESSORS + FIN_REGRESSORS


def configure_logger() -> logging.Logger:
    """
    Create a module logger emitting time-stamped INFO-level messages.

    Behaviour
    ---------
    - Sets logger level to INFO.
    - Adds a StreamHandler with the format: "%(asctime)s - %(levelname)s - %(message)s".
    - Avoids duplicate handlers on repeated calls.

    Returns
    -------
    logging.Logger
        Configured logger instance bound to this module’s namespace.
    """
    
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        ch = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger

logger = configure_logger()


def prepare_prophet_model(
    df_model: pd.DataFrame, 
    regressors: List[str]
) -> Prophet:
    """
    Construct and fit a Prophet model with specified exogenous regressors.

    Model
    -----
    Prophet specifies an additive structure:
    
        y_t = g(t) + s_y(t) + β' x_t + ε_t,
    
    where g(t) is a piecewise-linear trend with automatic changepoints
    (changepoint_range = 0.9 and changepoint_prior_scale = 0.05), s_y(t) denotes
    yearly seasonality, x_t are supplied regressors (macro/financial), β their
    coefficients, and ε_t is idiosyncratic noise.

    Regularisation
    --------------
    For each regressor r in `regressors`, the model calls:
    
        add_regressor(r, standardize = True, prior_scale = 0.01).
    
    Standardisation applies z-scoring per regressor (mean 0, variance 1).
    The coefficient prior is Gaussian: β_r ~ N(0, 0.01^2), acting as L2 shrinkage.

    Inputs
    ------
    df_model : DataFrame with columns ['ds', 'y'] + regressors
        'ds' (datetime) is the time index, 'y' the target (price), regressors as columns.
    regressors : list[str]
        Names of exogenous regressors in df_model.

    Returns
    -------
    Prophet
        Fitted model object ready for `make_future_dataframe` and `predict`.
    """
    
    model = Prophet(
        changepoint_prior_scale = 0.05,
        changepoint_range = 0.9,
        daily_seasonality = False,
        weekly_seasonality = False,
        yearly_seasonality = True
    )
   
    for reg in regressors:
        
        model.add_regressor(reg, standardize = True, prior_scale = 0.01)
   
    model.fit(df_model)
   
    return model


def add_financials(
    daily_df: pd.DataFrame, 
    fd: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge quarterly (or lower-frequency) financial data onto daily/weekly price dates.

    Method
    ------
    Uses `pd.merge_asof` with direction='backward' to carry the most recent available
    financial observation forward to each trading date.

    Requirements
    ------------
    Both frames contain a 'ds' datetime column sorted ascending.

    Returns
    -------
    pandas.DataFrame
        Daily/weekly frame augmented with the latest prior financial values at each date.
    """

    return pd.merge_asof(
        daily_df.sort_values('ds'),
        fd.sort_values('ds'),
        on = 'ds',
        direction = 'backward'
    )


def clip_to_bounds(
    df: pd.DataFrame, 
    price: float
) -> pd.DataFrame:
    """
    Truncate Prophet predictions to a reasonable band relative to current price.

    Rule
    ----
    For columns {yhat, yhat_lower, yhat_upper}, apply:
        lower bound = 0.2 * price,
        upper bound = 5.0 * price.
    This avoids pathological extrapolations in thin data regimes.

    Returns
    -------
    pandas.DataFrame
        Same object with clipped prediction columns (returned for chaining).
    """

    lower = config.lbp * price
    
    upper = config.ubp * price

    df[['yhat', 'yhat_lower', 'yhat_upper']] = df[
        ['yhat', 'yhat_lower', 'yhat_upper']
    ].clip(lower = lower, upper = upper)

    return df


def evaluate_forecast(
    model: Prophet, 
    initial: str, 
    period: str, 
    horizon: str
) -> pd.DataFrame:
    """
    Run rolling cross-validation and compute performance metrics for Prophet.

    Procedure
    ---------
    - `cross_validation(model, initial, period, horizon)` forms expanding windows
      with cutoffs spaced by `period`. For each cutoff, the model is trained on
      data up to the cutoff and scored over the next `horizon`.
    - `performance_metrics` computes RMSE, MAPE, MAE, coverage etc.

    Parameters
    ----------
    initial : str (pandas offset, e.g., "156 W")
        Initial training span before first cutoff.
    period : str
        Spacing between successive cutoffs.
    horizon : str
        Forecast horizon evaluated after each cutoff.

    Returns
    -------
    pandas.DataFrame
        Metrics per horizon; last row often corresponds to the full horizon window.

    Notes
    -----
    RMSE is defined as sqrt( mean( (y_true − y_pred)^2 ) ).
    """

    try:
        
        cv_results = cross_validation(
            model = model, 
            initial = initial, 
            period = period, 
            horizon = horizon
        )
        
        metrics = performance_metrics(
            df = cv_results
        )
        
        return metrics
    
    except Exception as e:
        
        logger.error("Error during cross-validation: %s", e)
        
        return pd.DataFrame()


def _linear(
    series: pd.Series, 
    end_value: Union[float, np.nan]
) -> pd.Series:
    """
    Replace a series over its index with a linear ramp from its first value to an end value.

    Definition
    ----------
    Let start = series.iloc[0], length = len(series), and v* = end_value.
    If v* is NaN, return the input unchanged. Otherwise construct:
     
        ramp[i] = start + (i / (length − 1)) * (v* − start), for i = 0..length−1,
    
    and return ramp aligned to the original index.

    Returns
    -------
    pandas.Series
        Linearly interpolated path from start to end_value.
    """

    if pd.isna(end_value):
        
        return series

    start = series.iloc[0]

    length = len(series)

    return pd.Series(
        np.linspace(start, end_value, length),
        index = series.index,
        dtype = series.dtype
    )


def build_base_future(
    model: Prophet,
    forecast_period: int,
    macro_df: pd.DataFrame,
    fin_df: pd.DataFrame,
    last_vals: pd.Series,
    regressors: List[str],
    int_array: Union[np.ndarray, None],
    inf_array: Union[np.ndarray, None],
    gdp_array: Union[np.ndarray, None],
    unemp_array: Union[np.ndarray, None]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a future design matrix with historical backfill and interpolated macro paths.

    Steps
    -----
   
    1) future_base = model.make_future_dataframe(periods = H, freq = 'W', include_history = True).
   
    2) Merge financials and macro histories using backward asof joins; forward/backward fill.
   
    3) Determine a boolean horizon_mask for dates strictly greater than last_vals['ds'].
   
    4) Interpolate macro arrays over the H horizon positions:
   
         - For Interest, Cpi, Unemp: if arrays length L ≥ 2, construct grid x = {0, seg, 2seg, ...}
           with seg = H / (L − 1) and set interp(h) = linear_interpolation(x, values) at h = 1..H.
   
         - For Gdp: if array length ≥ 3, use a simple linear path from element 2 to element 3
           over H steps; otherwise return None.

    Outputs
    -------
    future_base : DataFrame
        Contains ds plus all regressors, filled through the last historical date.
    horizon_mask : ndarray[bool]
        Mask selecting the H forecast rows inside future_base.
    interp_int_allH, interp_inf_allH, interp_gdp_allH, interp_unemp_allH : ndarray or None
        Interpolated macro values ready to assign into the forecast horizon.

    Rationale
    ---------
    Interpolation provides a smooth, scenario-agnostic macro trajectory to combine
    with scenario-specific financial ramps for revenue/EPS.
    """

    future_base = model.make_future_dataframe(
        periods = forecast_period,
        freq = 'W',
        include_history = True
    )

    future_base = pd.merge_asof(
        future_base,
        fin_df,       
        on = 'ds',
        direction = 'backward'
    )

    future_base = pd.merge_asof(
        future_base,
        macro_df,     
        on = 'ds',
        direction = 'backward'
    )

    future_base[regressors] = future_base[regressors].ffill().bfill()

    horizon_mask = future_base['ds'] > last_vals['ds']
   
    H = horizon_mask.sum()
   
    h_idx = np.arange(1, H + 1)

    if 'Interest' in regressors:
   
        if int_array is not None and len(int_array) > 1 and not np.all(np.isnan(int_array)):
   
            L_int = len(int_array)
   
            seg_int = forecast_period / (L_int - 1)
   
            x_int = np.arange(L_int) * seg_int
   
            interp_int_allH = np.interp(h_idx, x_int, int_array)
   
        else:
   
            interp_int_allH = None
   
    else:
   
        interp_int_allH = None

    if 'Cpi' in regressors:
   
        if inf_array is not None and len(inf_array) > 1 and not np.all(np.isnan(inf_array)):
   
            L_inf = len(inf_array)
   
            seg_inf = forecast_period / (L_inf - 1)
   
            x_inf = np.arange(L_inf) * seg_inf
   
            interp_inf_allH = np.interp(h_idx, x_inf, inf_array)
   
        else:
   
            interp_inf_allH = None
   
    else:
   
        interp_inf_allH = None

    if 'Gdp' in regressors:
   
        if gdp_array is not None and len(gdp_array) > 2 and not np.all(np.isnan(gdp_array)):
   
            start_gdp = gdp_array[1]
            end_gdp = gdp_array[2]
   
            interp_gdp_allH = np.linspace(start_gdp, end_gdp, H)
   
        else:
   
            interp_gdp_allH = None
   
    else:
   
        interp_gdp_allH = None

    if 'Unemp' in regressors:
        
        if unemp_array is not None and len(unemp_array) > 1 and not np.all(np.isnan(unemp_array)):
        
            L_unemp = len(unemp_array)
        
            seg_unemp = forecast_period / (L_unemp - 1)
        
            x_unemp = np.arange(L_unemp) * seg_unemp
        
            interp_unemp_allH = np.interp(h_idx, x_unemp, unemp_array)
        
        else:
        
            interp_unemp_allH = None
    else:
    
        interp_unemp_allH = None

    return (
        future_base,
        horizon_mask.values,      
        interp_int_allH,
        interp_inf_allH,
        interp_gdp_allH,
        interp_unemp_allH
    )
    
    
def _adf_d_order(
    x: pd.Series
) -> int:
    """
    Select the integration order d ∈ {0, 1} using the Augmented Dickey–Fuller test.

    Test
    ----
    Null hypothesis: the series has a unit root (non-stationary).
    If p-value > 0.05, fail to reject → treat as non-stationary and choose d = 1.
    Else choose d = 0.

    Safeguards
    ----------
    - If fewer than 30 observations or the test fails, return 0.

    Returns
    -------
    int
        1 or 0 according to the ADF decision criterion.
    """
   
    x = pd.Series(x).dropna()
   
    if len(x) < 30:
   
        return 0
   
    try:
   
        p = adfuller(x, autolag = 'AIC')[1]

        if p > 0.05:
            
            return 1.0
        
        else:
            
            return 0.0
   
    except Exception:
   
        return 0.0


def _choose_arima(
    resid: pd.Series,
    p_grid=(0,1,2),
    q_grid=(0,1,2)
):
    """
    Fit ARIMA(p,d,q) on Prophet residuals using a small AIC grid search.

    Procedure
    ---------
    1) Determine d via `_adf_d_order` on `resid`.
    2) For p in p_grid, q in q_grid, fit SARIMAX with order (p,d,q) and trend='n'
       (trend suppressed because Prophet already models trend/level).
    3) Select the model minimising Akaike Information Criterion:
         AIC = 2k − 2 log L,
       where k is the number of estimated parameters and L is the maximised likelihood.

    Returns
    -------
    statsmodels result or None
        The best fitted ARIMA result object, or None if all fits fail.

    Notes
    -----
    Stationarity and invertibility constraints are relaxed to avoid failures
    under small samples or near-unit-root behaviour.
    """
   
    y = pd.Series(resid).astype(float).dropna()
   
    d = _adf_d_order(
        x = y
    )
    
    best = None
    
    best_res = None
    
    for p in p_grid:
    
        for q in q_grid:
    
            try:
    
                mod = SARIMAX(
                    endog = y, 
                    order = (p,d,q),
                    trend = 'n',
                    enforce_stationarity = False,
                    enforce_invertibility = False
                )
                
                res = mod.fit(disp = False)
                
                if best is None or res.aic < best:
                
                    best, best_res = res.aic, res

            except Exception:

                continue

    return best_res 


def _fit_garch(
    resid_after_arima: pd.Series
):
    """
    Fit a zero-mean GARCH(1,1) with Student-t innovations to ARIMA residuals.

    Model
    -----
    Let u_t be ARIMA residuals. The conditional variance follows:
    
        σ_t^2 = ω + α u_{t−1}^2 + β σ_{t−1}^2,
    
    with z_t ~ Student-t(ν) and u_t = σ_t z_t. Mean is constrained to zero.
    
    Persistence α + β < 1 implies covariance stationarity.

    Fitting
    -------
    - Uses `arch_model(..., mean='Zero', vol='GARCH', p=1, q=1, dist='t')`.
    - Requires at least 50 observations; else returns None.

    Returns
    -------
    arch.univariate.base.ARCHModelResult or None
        Fitted GARCH result or None on failure.
    """
   
    z = pd.Series(resid_after_arima).astype(float).dropna()
   
    if len(z) < 50:
   
        return None
   
    try:
   
        am = arch_model(z, mean = 'Zero', vol = 'GARCH', p = 1, q = 1, o = 0, dist = 't')
   
        garch_res = am.fit(disp = 'off')
   
        return garch_res
   
    except Exception:
   
        return None


def _forecast_arima(
    res, 
    steps: int
) -> np.ndarray:
    """
    Produce the ARIMA mean forecast μ_h for h = 1..steps.

    Definition
    ----------
    If `res` is a fitted ARIMA result, return:
   
        μ = predicted_mean for the next `steps` points,
   
    padded with zeros if fewer elements are returned and with NaNs replaced by 0.

    Returns
    -------
    numpy.ndarray
        Array of length `steps` containing the residual mean forecast.
    """
    
    if res is None or steps <= 0:
  
        return np.zeros(steps, dtype = float)
  
    try:
  
        fc = res.get_forecast(steps = steps)
  
        mu = np.asarray(fc.predicted_mean)
  
        if mu.shape[0] < steps:
  
            mu = np.pad(mu, (0, steps - mu.shape[0]))
  
        mu = np.nan_to_num(mu, nan = 0.0)
  
        return mu[:steps]
  
    except Exception:
  
        return np.zeros(steps, dtype = float)


def _forecast_garch(
    garch_res,
    steps: int
) -> np.ndarray:
    """
    Produce the GARCH conditional sigma path σ_g,h for h = 1..steps.

    Method
    ------
    - Call `garch_res.forecast(horizon = steps, reindex = False)`.
    - Extract the forecasted variance for the final available period,
      take the square root, and pad to `steps` if necessary.
    - Replace NaNs with zeros.

    Returns
    -------
    numpy.ndarray
        Conditional standard deviation array (length `steps`). Zeros if unavailable.
    """
  
    if garch_res is None or steps <= 0:
    
        return np.zeros(steps, dtype = float)
    
    try:
     
        f = garch_res.forecast(horizon = steps, reindex = False)

        var = np.asarray(f.variance.values)[-1]

        sigma = np.sqrt(np.maximum(var, 0.0))

        if sigma.shape[0] < steps:

            sigma = np.pad(sigma, (0, steps - sigma.shape[0]), constant_values=sigma[-1] if sigma.size else 0.0)

        return np.nan_to_num(sigma, nan=0.0)[:steps]

    except Exception:

        return np.zeros(steps, dtype = float)


def fit_arima_garch_on_prophet_residuals(
    prophet_model: Prophet,
    df_model: pd.DataFrame,
    regressors: List[str]
):
    """
    Fit ARIMA and GARCH layers on top of Prophet’s in-sample residuals.

    Pipeline
    --------
   
    1) Compute in-sample Prophet predictions on df_model[['ds','y'] + regressors].
   
       Residuals: e_t = y_t − ŷ_t^Prophet.
   
    2) Fit ARIMA to e_t to capture residual mean: obtain fitted values ê_t^ARIMA
       and residuals u_t = e_t − ê_t^ARIMA.
   
    3) Fit GARCH(1,1) with Student-t innovations to u_t to capture conditional sigma.

    Returns
    -------
    (arima_res, garch_res, resid_mean, resid_after_arima)
        arima_res : statsmodels result or None
        garch_res : arch result or None
        resid_mean : pandas.Series of e_t
        resid_after_arima : pandas.Series of u_t

    Interpretation
    --------------
    The ARIMA component adjusts the Prophet mean forecast; the GARCH component
    widens predictive intervals where residual heteroskedasticity is present.
    """

    ins = prophet_model.predict(df_model[['ds'] + regressors].copy())

    e = pd.Series(df_model['y'].values - ins['yhat'].values, index=df_model['ds'])

    e = e.dropna()

    arima_res = _choose_arima(
        resid = e
    )

    if arima_res is not None:

        u = e - pd.Series(arima_res.fittedvalues, index=e.index)

    else:

        u = e.copy()

    garch_res = _fit_garch(
        resid_after_arima = u
    )

    return arima_res, garch_res, e, u


def apply_hybrid_adjustment(
    forecast, 
    horizon_mask,
    arima_res, 
    garch_res,
    widen_with_prophet_pi = True
):
    """
    Add ARIMA mean to Prophet forecast and fuse Prophet and GARCH uncertainty.

    Equations
    ---------
    Let H be the number of forecast points (sum of horizon_mask).
  
    - Mean adjustment:
  
        μ = _forecast_arima(arima_res, H)      (zeros if unavailable)
  
        ŷ_h,hyb = ŷ_h + μ_h
  
    - Uncertainty combination:
  
        If Prophet intervals are present and `widen_with_prophet_pi` is True:
  
            σ_p,h = (yhat_upper_h − ŷ_h,hyb) / 1.96
  
            σ_comb,h = sqrt( max(σ_p,h,0)^2 + max(σ_g,h,0)^2 )
  
        Else:
  
            σ_comb,h = σ_g,h
  
        Predictive interval:
  
            [ŷ_h,hyb − 1.96 σ_comb,h,  ŷ_h,hyb + 1.96 σ_comb,h]

    Inputs
    ------
    forecast : DataFrame
        Prophet forecast output with columns including {'ds','yhat','yhat_lower','yhat_upper'}.
    horizon_mask : array-like[bool]
        Boolean mask for horizon rows inside `forecast`.
    arima_res, garch_res
        Fitted ARIMA and GARCH results or None.
    widen_with_prophet_pi : bool
        If True, combine Prophet and GARCH sigmas in quadrature; otherwise use GARCH only.

    Returns
    -------
    pandas.DataFrame
        Adjusted forecast frame with updated yhat and intervals on the horizon.
    """
       
    fc = forecast.copy()
   
    H = int(np.sum(horizon_mask))
   
    if H <= 0:
   
        return fc

    if arima_res is not None:
        
        mu = _forecast_arima(
            res = arima_res, 
            steps = H
        ) 
    
    else: 
    
        mu = np.zeros(H)
    
    if garch_res is not None:
        
        sig_g = _forecast_garch(
            garch_res = garch_res,
            H = H
        )  
        
    else:
        
        sig_g = np.zeros(H)

    yhat_h = fc.loc[horizon_mask, 'yhat'].to_numpy() + mu
    
    fc.loc[horizon_mask, 'yhat'] = yhat_h

    if {'yhat_lower','yhat_upper'} <= set(fc.columns) and widen_with_prophet_pi:
    
        lo_p = fc.loc[horizon_mask, 'yhat_lower'].to_numpy()
    
        hi_p = fc.loc[horizon_mask, 'yhat_upper'].to_numpy()
    
        sig_p = (hi_p - yhat_h) / 1.96
    
        sig_comb = np.sqrt(np.maximum(sig_p, 0.0) ** 2 + np.maximum(sig_g, 0.0) ** 2)
    
        fc.loc[horizon_mask, 'yhat_lower'] = yhat_h - 1.96 * sig_comb
    
        fc.loc[horizon_mask, 'yhat_upper'] = yhat_h + 1.96 * sig_comb

    else:

        fc.loc[horizon_mask, 'yhat_lower'] = yhat_h - 1.96 * sig_g

        fc.loc[horizon_mask, 'yhat_upper'] = yhat_h + 1.96 * sig_g

    return fc


def forecast_with_prophet(
    model: Prophet,
    current_price: float,
    last_vals: pd.Series,
    regressors: List[str],
    future_base: pd.DataFrame,
    horizon_mask: np.ndarray,
    interp_int_allH: np.ndarray,
    interp_inf_allH: np.ndarray,
    interp_gdp_allH: np.ndarray,
    interp_unemp_allH: np.ndarray,
    rev_target: Union[float, np.nan] = None,
    eps_target: Union[float, np.nan] = None,
    arima_res=None,
    garch_res=None,
    widen_with_prophet_pi: bool = True
) -> pd.DataFrame:
    """
    Produce a scenario forecast given a prepared future design and optional targets.

    Scenario mechanics
    ------------------
    - Financial ramps:
   
        If 'Revenue' is in regressors and `rev_target` is not NaN, replace the
        horizon path by a linear ramp from its first horizon value to `rev_target`.
        Similarly for 'EPS (Basic)' and `eps_target`. See `_linear`.
   
    - Macro paths:
   
        If interpolated arrays for Interest, Cpi, Gdp, Unemp are provided, assign
        them on the horizon. If only a scalar is provided, linearly ramp from the
        last observed value to that scalar.

    Prediction and hybridisation
    ----------------------------
    - Ensure no missing regressor values remain; raise on NaNs.
    - Obtain Prophet predictions, then optionally apply hybrid adjustment:
        yhat_hyb as mean, and intervals widened using σ_comb (see module overview).
    - Clip {yhat, yhat_lower, yhat_upper} to [lb * current_price, ub * current_price].

    Returns
    -------
    pandas.DataFrame
        Scenario forecast with adjusted means and intervals.
    """
   
    future = future_base.copy()
   
    if 'Revenue' in regressors:
       
        if not pd.isna(rev_target):
       
            future.loc[horizon_mask, 'Revenue'] = _linear(
                series = future.loc[horizon_mask, 'Revenue'],
                end_value = rev_target
            )
   
    if 'EPS (Basic)' in regressors:
        
        if not pd.isna(eps_target):
        
            future.loc[horizon_mask, 'EPS (Basic)'] = _linear(
                series = future.loc[horizon_mask, 'EPS (Basic)'],
                end_value = eps_target
            )

    if interp_int_allH is not None and 'Interest' in regressors:
        
        future.loc[horizon_mask, 'Interest'] = interp_int_allH
   
    elif 'Interest' in regressors and not pd.isna(interp_int_allH):
       
        future.loc[horizon_mask, 'Interest'] = _linear(
            series = future.loc[horizon_mask, 'Interest'],
            end_value = float(interp_int_allH) if np.isscalar(interp_int_allH) else None
        )

    if interp_inf_allH is not None and 'Cpi' in regressors:
       
        future.loc[horizon_mask, 'Cpi'] = interp_inf_allH
   
    elif 'Cpi' in regressors and not pd.isna(interp_inf_allH):
        
        future.loc[horizon_mask, 'Cpi'] = _linear(
            
            series = future.loc[horizon_mask, 'Cpi'],
            end_value = float(interp_inf_allH) if np.isscalar(interp_inf_allH) else None
        )

    if interp_gdp_allH is not None and 'Gdp' in regressors:
        
        future.loc[horizon_mask, 'Gdp'] = interp_gdp_allH
   
    elif 'Gdp' in regressors and not pd.isna(interp_gdp_allH):
        
        future.loc[horizon_mask, 'Gdp'] = _linear(
            series = future.loc[horizon_mask, 'Gdp'],
            end_value = float(interp_gdp_allH) if np.isscalar(interp_gdp_allH) else None
        )

    if interp_unemp_allH is not None and 'Unemp' in regressors:
        
        future.loc[horizon_mask, 'Unemp'] = interp_unemp_allH
   
    elif 'Unemp' in regressors and not pd.isna(interp_unemp_allH):
        
        future.loc[horizon_mask, 'Unemp'] = _linear(
            series = future.loc[horizon_mask, 'Unemp'],
            end_value = float(interp_unemp_allH) if np.isscalar(interp_unemp_allH) else None
        )

    if future[regressors].isna().any().any():
       
        missing = future[regressors].isna().sum()
       
        raise ValueError(f"NaNs remain in regressors after filling:\n{missing}")

    forecast = model.predict(future)
    
    if (arima_res is not None) or (garch_res is not None):
       
        forecast = apply_hybrid_adjustment(
            forecast = forecast,
            horizon_mask = horizon_mask,
            arima_res = arima_res,
            garch_res = garch_res,
            widen_with_prophet_pi = widen_with_prophet_pi
        )

    
    forecast = clip_to_bounds(
        df = forecast, 
        price = current_price
    )

    return forecast


def forecast_with_prophet_without_fd(
    model: Prophet,
    forecast_period: int,
    macro_df: pd.DataFrame,
    last_vals: pd.Series,
    current_price: float,
    regressors: List[str],
    arima_res=None,
    garch_res=None,
    widen_with_prophet_pi: bool = True
) -> pd.DataFrame:
    """
    Forecast when no financial (Revenue/EPS) data are available.

    Steps
    -----
   
    1) Create weekly future for H = forecast_period.
   
    2) Merge macro history, forward-fill missing values.
   
    3) For each regressor r in `regressors`, set future[r] = last_vals[r] on the horizon,
       which implies a flat macro path unless superseded by ARIMA/GARCH adjustments.
   
    4) Predict with Prophet; optionally apply hybrid adjustment (ARIMA mean and GARCH sigma).
   
    5) Clip predictions to [lb * current_price, ub * current_price].

    Returns
    -------
    pandas.DataFrame
        Forecast frame with hybrid intervals if enabled.

    Notes
    -----
    When ARIMA/GARCH layers are used, only the mean/variance of the residual
    component changes; regressors remain fixed at last observed levels.
    """

    future = model.make_future_dataframe(
        periods = forecast_period, 
        freq = 'W'
    )
    
    future['ds'] = pd.to_datetime(future['ds'])

    macro_df['ds'] = pd.to_datetime(macro_df['ds'])

    future = future.merge(
        macro_df, 
        on = 'ds', 
        how = 'left'
    )
    
    future.ffill(inplace=True)

    for reg in regressors:
        
        future[reg] = last_vals[reg]

    forecast = model.predict(future)

    horizon_mask = forecast['ds'] > last_vals['ds']

    if (arima_res is not None) or (garch_res is not None):
        forecast = apply_hybrid_adjustment(
            forecast = forecast,
            horizon_mask = horizon_mask.values,
            arima_res = arima_res,
            garch_res = garch_res,
            widen_with_prophet_pi = widen_with_prophet_pi
        )
   
    forecast = clip_to_bounds(
        df = forecast, 
        price = current_price
    )
    
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]

    return forecast


def main() -> None:
    """
    Orchestrate the end-to-end hybrid forecasting, cross-validation, scenarios, and export.

    High-level workflow
    -------------------
    For each ticker in `config.tickers`:
   
      1) Load weekly closes y_t, macro history (Interest, Cpi, Gdp, Unemp), and
         Prophet-ready financials (Revenue, EPS (Basic)) if available.
   
      2) Join price with macro (and financials if present) to form df_model = {ds, y, regressors}.
   
      3) Fit Prophet with yearly seasonality and standardised regressors with Gaussian priors.
   
      4) Fit ARIMA on Prophet residuals e_t to model residual mean; obtain ARIMA residuals u_t.
   
      5) Fit GARCH(1,1) with Student-t innovations on u_t to model conditional sigma.
   
      6) Cross-validate Prophet with (initial, period, horizon) windows; record RMSE
         and scale by current price (capped at 2 for stability).
   
      7) Build a future base (history + H weekly steps), merge and forward-fill regressors,
         and interpolate macro arrays over H.
   
      8) If no financials: forecast once. Else, iterate over scenario pairs
         (rev_key, eps_key) from REV_KEYS × EPS_KEYS:
            - Obtain targets from `next_period_forecast()` for that ticker.
            - Apply linear ramps for Revenue/EPS to those targets.
            - Predict with Prophet, apply hybrid adjustment, and record end-horizon
              {yhat, yhat_lower, yhat_upper}.
   
      9) Aggregate scenarios:
            min_price  = min end-horizon yhat_lower across scenarios (or single run),
            max_price  = max end-horizon yhat_upper,
            avg_price  = mean of end-horizon yhat across scenarios,
            avg_return = avg_price / current_price − 1,
            scenario_se:
      
                • No-financials case: ((max − min) / 2) / (1.96 * current_price),
                  i.e., infer sigma from half-width of a 95% interval under Normality.
      
                • Scenario case: standard deviation of end-horizon returns across
                  all collected {yhat, yhat_lower, yhat_upper}.
     10) Final standard error:
      
            SE_final = sqrt( scenario_se^2 + rmse_cv^2 ),
      
         with rmse_cv substituted by the maximum available RMSE across tickers if missing.

    Export
    ------
    Writes a sheet "Prophet Pred" with columns:
        Current Price, Avg Price, Low Price, High Price, Returns, SE,
    indexed by ticker, to `config.MODEL_FILE` via `export_results`.

    Key equations (summary)
    -----------------------
    Prophet:     y_t = g(t) + s_y(t) + β' x_t + ε_t
   
    ARIMA mean:  μ_h = E[e_{t+h} | ARIMA(p,d,q)]
   
    GARCH var:   σ_{t+h}^2 from GARCH(1,1) recursion
   
    Hybrid mean: ŷ_h,hyb = ŷ_h^Prophet + μ_h
   
    Hybrid PI:   σ_comb,h = sqrt( σ_p,h^2 + σ_g,h^2 ),
                 PI_h = ŷ_h,hyb ± 1.96 σ_comb,h
   
    Return:      r̂ = avg_price / current_price − 1
   
    Final SE:    SE_final = sqrt( scenario_se^2 + rmse_cv^2 )

    Logging
    -------
    Emits per-ticker summary:
       Ticker, Low, Avg, High, Returns, SE.

    Side effects
    ------------
    Produces an Excel workbook with the results and logs progress to stdout.
    """
    
    fdata = FinancialForecastData()
   
    macro = fdata.macro
   
    r = macro.r
   
    tickers = config.tickers

    forecast_period = 52  
   
    cv_initial = f"{forecast_period * 3} W"
   
    cv_period = f"{int(forecast_period * 0.5)} W"
   
    cv_horizon = f"{forecast_period} W"

    logger.info("Importing data from Excel ...")

    close = r.weekly_close

    next_fc = fdata.next_period_forecast()
   
    next_macro_dict = macro.macro_forecast_dict()

    latest_prices = r.last_price
   
    raw_macro = macro.assign_macro_history_non_pct()
   
    macro_history = (
        raw_macro
        .reset_index()
        .rename(columns = {'year': 'ds'})
        [['ticker', 'ds'] + MACRO_REGRESSORS]
    )

    if pd.api.types.is_period_dtype(macro_history['ds']):
       
        macro_history['ds'] = macro_history['ds'].dt.to_timestamp()
   
    else:
        
        macro_history['ds'] = pd.to_datetime(macro_history['ds'])

    macro_history.sort_values(['ticker', 'ds'], inplace = True)
    
    macro_groups = macro_history.groupby('ticker')

    fin_data_raw: Dict[str, pd.DataFrame] = fdata.prophet_data
    
    fin_data_processed: Dict[str, pd.DataFrame] = {}

    for tk in tickers:
       
        df_fd = fin_data_raw.get(tk, pd.DataFrame()).reset_index().rename(
            
            columns = {
                'index': 'ds',
                'rev': 'Revenue',
                'eps': 'EPS (Basic)'
            }
        )

        if 'ds' in df_fd.columns:
          
            df_fd['ds'] = pd.to_datetime(df_fd['ds'])

        df_fd.sort_values('ds', inplace = True)
      
        fin_data_processed[tk] = df_fd

    min_price = {}
    
    max_price = {}
    
    avg_price = {}
    
    avg_returns_dict = {}
    
    scenario_se = {}
    
    se = {}
    
    final_rmse = {}

    logger.info("Computing Prophet Forecasts ...")

    for ticker in tickers:
       
        logger.info("Processing ticker: %s", ticker)

        current_price = latest_prices.get(ticker, np.nan)

        if pd.isna(current_price):
           
            logger.warning("No current price for %s. Skipping.", ticker)
           
            continue

        macro_forecasts = next_macro_dict.get(ticker, {})

        int_array = np.array(macro_forecasts.get('InterestRate', [np.nan]))
      
        inf_array = np.array(macro_forecasts.get('Consumer_Price_Index_Cpi', [np.nan]))
      
        gdp_array = np.array(macro_forecasts.get('GDP', [np.nan, np.nan, np.nan]))
      
        unemp_array = np.array(macro_forecasts.get('Unemployment', [np.nan]))

        df_price = pd.DataFrame({
            'ds': close.index,
            'y':  close[ticker]
        })

        df_price['ds'] = pd.to_datetime(df_price['ds'])
      
        df_price.sort_values('ds', inplace = True) 

        if ticker not in macro_groups.groups:
           
            logger.warning("No macro history for %s. Skipping.", ticker)
           
            min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
           
            continue

        tm = macro_groups.get_group(ticker).drop(columns = 'ticker').copy()

        fd_ticker = fin_data_processed.get(ticker, pd.DataFrame()).copy()

        if fd_ticker.empty:
            
            df_model = df_price.merge(tm, on = 'ds', how = 'left')
            
            df_model.ffill(inplace = True)
            
            df_model.dropna(inplace = True)
            
            regressors = MACRO_REGRESSORS
       
        else:
       
            df_price_fd = add_financials(df_price, fd_ticker)  
       
            df_model = df_price_fd.merge(tm, on='ds', how='left')
       
            df_model.ffill(inplace = True)
       
            df_model.dropna(inplace = True)
       
            regressors = ALL_REGRESSORS

        if df_model.empty:
       
            logger.warning("Insufficient data for ticker %s. Skipping.", ticker)
       
            min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
       
            continue

        try:
       
            m_prophet = prepare_prophet_model(
                df_model = df_model[['ds', 'y'] + regressors], 
                regressors = regressors
            )
       
        except Exception as e:
       
            logger.error("Failed to fit Prophet model for %s: %s", ticker, e)
       
            min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
       
            continue

        try:
          
            arima_res, garch_res, resid_e, resid_u = fit_arima_garch_on_prophet_residuals(
                prophet_model = m_prophet,
                df_model = df_model[['ds','y'] + regressors],
                regressors = regressors
            )
        except Exception as e:
        
            logger.warning("ARIMA/GARCH fit failed for %s: %s", ticker, e)
        
            arima_res = None
            
            garch_res = None

        cv_metrics = evaluate_forecast(
            model = m_prophet, 
            initial = cv_initial, 
            period = cv_period, 
            horizon = cv_horizon
        )

        if not cv_metrics.empty:
            
            final_rmse[ticker] = min(cv_metrics['rmse'].iat[-1] / current_price, 2)
        
        else:
           
            final_rmse[ticker] = np.nan

        last_vals = df_model.iloc[-1]

        if fd_ticker.empty:
            
            try:
               
                forecast = forecast_with_prophet_without_fd(
                    model = m_prophet,
                    forecast_period = forecast_period,
                    macro_df = tm,          
                    last_vals = last_vals,
                    current_price = current_price,
                    regressors = regressors,
                    arima_res = arima_res,
                    garch_res = garch_res,
                    widen_with_prophet_pi = True                    
                )
            
                min_price[ticker] = forecast['yhat_lower'].iloc[-1]
                
                max_price[ticker] = forecast['yhat_upper'].iloc[-1]
                
                avg_price[ticker] = forecast['yhat'].iloc[-1]
            
                avg_returns_dict[ticker] = ((avg_price[ticker] / current_price) - 1 if current_price != 0 else np.nan)
            
                scenario_se[ticker] = ((max_price[ticker] - min_price[ticker]) / (2 * 1.96 * current_price) if current_price != 0 else np.nan)

            except Exception as e:
               
                logger.error("Forecasting without financial data failed for %s: %s", ticker, e)
               
                min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0

        else:
            (
                future_base,
                horizon_mask,
                interp_int_allH,
                interp_inf_allH,
                interp_gdp_allH,
                interp_unemp_allH
            ) = build_base_future(
                model = m_prophet,
                forecast_period = forecast_period,
                macro_df = tm,
                fin_df = fd_ticker,
                last_vals = last_vals,
                regressors = regressors,
                int_array = int_array,
                inf_array = inf_array,
                gdp_array = gdp_array,
                unemp_array = unemp_array
            )

            results = []
            
            for rev_key, eps_key in SCENARIOS:
                
                label = f"{rev_key}|{eps_key}"
                
                rev_target = next_fc.at[ticker, rev_key]
                eps_target = next_fc.at[ticker, eps_key]

                try:
                
                    fc = forecast_with_prophet(
                        model = m_prophet,
                        current_price = current_price,
                        last_vals = last_vals,
                        regressors = regressors,
                        future_base = future_base,
                        horizon_mask = horizon_mask,
                        interp_int_allH = interp_int_allH,
                        interp_inf_allH = interp_inf_allH,
                        interp_gdp_allH = interp_gdp_allH,
                        interp_unemp_allH = interp_unemp_allH,
                        rev_target = rev_target,
                        eps_target = eps_target,
                        arima_res = arima_res,
                        garch_res = garch_res,
                        widen_with_prophet_pi = True                        
                    )
                    
                    yhat_val = fc['yhat'].iloc[-1]
                    
                    yhat_lower = fc['yhat_lower'].iloc[-1]
                    
                    yhat_upper = fc['yhat_upper'].iloc[-1]
               
                except Exception as e:
               
                    logger.error("Scenario %s failed for %s: %s", label, ticker, e)
               
                    yhat_val = 0.0
                    
                    yhat_lower = 0.0
                    
                    yhat_upper = 0.0

                results.append({
                    'Ticker': ticker,
                    'Scenario': label,
                    'RevTarget': rev_target,
                    'EpsTarget': eps_target,
                    'yhat': yhat_val,
                    'yhat_lower': yhat_lower,
                    'yhat_upper': yhat_upper
                })

            if not results:
                
                logger.warning("No scenarios available for ticker %s. Skipping.", ticker)
                
                min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
           
            else:
           
                scenario_df = (
                    pd.DataFrame(results)
                    .set_index('Ticker')
                    .sort_index()
                )

                min_price[ticker] = scenario_df['yhat_lower'].min()
                
                max_price[ticker] = scenario_df['yhat_upper'].max()

                all_y = scenario_df['yhat'].values
              
                all_low = scenario_df['yhat_lower'].values
              
                all_high= scenario_df['yhat_upper'].values
            
                scenario_array = np.concatenate([all_y, all_low, all_high])

                avg_price[ticker] = (all_y.mean() if all_y.size > 0 else 0.0)
              
                returns_arr = ((scenario_array / current_price) - 1 if (current_price != 0 and scenario_array.size > 0) else np.zeros_like(scenario_array))
              
                avg_returns_dict[ticker] = (returns_arr.mean() if returns_arr.size > 0 else 0.0)
            
                scenario_vol = (returns_arr.std() if returns_arr.size > 0 else 0.0)
              
                scenario_se[ticker] = (scenario_vol)

    max_rmse = max(pd.Series(final_rmse).dropna())
    
    for ticker in tickers:

        if ticker in final_rmse:

            if pd.isna(final_rmse[ticker]):
             
                se[ticker] = np.sqrt(scenario_se[ticker] ** 2 + (max_rmse ** 2))
            else:
               
                se[ticker] = np.sqrt((scenario_se[ticker] ** 2) + (final_rmse[ticker] ** 2))

        else:
          
            se[ticker] = 0

        logger.info(
            "Ticker: %s, Low: %.2f, Avg: %.2f, High: %.2f, Returns: %.4f, SE: %.4f", 
            ticker, 
            min_price[ticker], 
            avg_price[ticker],
            max_price[ticker], 
            avg_returns_dict[ticker],
            se[ticker]
        )

    prophet_results = pd.DataFrame({
        'Ticker': tickers,
        'Current Price': [latest_prices.get(tk, np.nan) for tk in tickers],
        'Avg Price': [avg_price.get(tk, np.nan) for tk in tickers],
        'Low Price': [min_price.get(tk, np.nan) for tk in tickers],
        'High Price': [max_price.get(tk, np.nan) for tk in tickers],
        'Returns': [avg_returns_dict.get(tk, np.nan) for tk in tickers],
        'SE': [se.get(tk, np.nan) for tk in tickers]
    }).set_index('Ticker')

    sheets_to_write = {
        "Prophet Pred": prophet_results,
    }
    export_results(
        sheets = sheets_to_write, 
        output_excel_file = config.MODEL_FILE
    )
    
    logger.info("Prophet forecasting, cross-validation, and export completed.")
    
    
if __name__ == "__main__":
    main()
