from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def _winsorize(
    s: pd.Series, 
    p: float = 0.01
) -> pd.Series:
    """
    Perform two-sided winsorisation of a numeric series at probability
    cut-offs [p, 1−p].

    Purpose
    -------
    Winsorisation replaces extreme observations in the tails by the
    corresponding quantile values in order to reduce the influence of
    outliers on downstream estimates (e.g., regression coefficients,
    volatilities).

    Method
    ------
    Let Q_lo = quantile_p(s) and Q_hi = quantile_{1−p}(s). The winsorised
    series s_w is defined elementwise by:
     
        s_w[t] = min( max(s[t], Q_lo), Q_hi ).

    Parameters
    ----------
    s : pandas.Series
        Input numeric series.
    p : float, default 0.01
        Tail probability for winsorisation; must satisfy 0 < p < 0.5.

    Returns
    -------
    pandas.Series
        Series with values clamped to [Q_lo, Q_hi].

    Notes
    -----
    - Winsorisation preserves order statistics between the two cut-offs.
   
    - This function operates silently on empty inputs and returns them unchanged.
    """
   
    if len(s) == 0:
   
        return s
   
    lo, hi = s.quantile([p, 1 - p])
   
    return s.clip(lo, hi)


def _weekly_same_day_last_prices(
    px: pd.Series, 
    weekday: str = "W-FRI"
) -> pd.Series:
    """
    Resample a price series to weekly frequency using the last observation
    on a fixed weekday (default Friday).

    Method
    ------
    1) Sort the input by index.
    
    2) Resample with rule = weekday.
    
    3) Take the last observation for each weekly bin.
    
    4) Drop missing values created by resampling.

    Parameters
    ----------
    px : pandas.Series
        Price level series indexed by a monotonic DateTimeIndex.
    weekday : str, default "W-FRI"
        Pandas offset alias specifying the weekly anchor day.

    Returns
    -------
    pandas.Series
        Weekly prices aligned to the specified weekday.

    Rationale
    ---------
    Aligning weekly prices to a consistent weekday mitigates day-of-week
    sampling effects and ensures stock and benchmark returns are computed
    on comparable horizons.
    """
    
    px = px.sort_index()
    
    return px.resample(weekday).last().dropna()


def _weekly_returns_from_prices(
    px: pd.Series,
    weekday: str = "W-FRI"
) -> pd.Series:
    """
    Compute simple weekly returns from a daily (or higher-frequency) price series.

    Method
    ------
    1) Obtain weekly prices with `_weekly_same_day_last_prices`.
    
    2) Compute simple returns:
      
         r_t = P_t / P_{t-1} − 1
      
       where P_t is the weekly close on the anchor weekday.

    Parameters
    ----------
    px : pandas.Series
        Price levels indexed by date.
    weekday : str, default "W-FRI"
        Weekly anchor weekday for resampling.

    Returns
    -------
    pandas.Series
        Weekly simple returns with NaNs removed.
    """
    
    wk = _weekly_same_day_last_prices(
        px = px, 
        weekday = weekday
    )
    
    return wk.pct_change().dropna()


def _align_returns(
    r1: pd.Series,
    r2: pd.Series
) -> pd.DataFrame:
    """
    Align two return series on their common dates and label the columns.

    Method
    ------
    The function performs an inner join on the index and drops rows
    containing NaNs. The resulting DataFrame has columns:
    
        'p' : first series (e.g., portfolio/stock),
    
        'b' : second series (e.g., benchmark/market).

    Parameters
    ----------
    r1, r2 : pandas.Series
        Return series to align.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame with aligned returns.
    """
    
    df = pd.concat([r1, r2], axis = 1, join = "inner").dropna()
    
    df.columns = ["p", "b"]
    
    return df


def _looks_like_returns(
    s: pd.Series
) -> bool:
    """
    Heuristically determine whether a numeric series resembles returns rather
    than prices.

    Heuristic
    ---------
    Treat a series as returns if:
    
      (a) at least 90% of absolute values are ≤ 0.5, and
    
      (b) at least 99% of absolute values are ≤ 2.0.

    Rationale
    ---------
    Daily or weekly returns should typically lie in a narrow range
    (e.g., ±50% is already extreme), whereas prices are positive with much
    larger magnitudes and variability.

    Parameters
    ----------
    s : pandas.Series
        Input numeric series.

    Returns
    -------
    bool
        True if the heuristic deems the series to be returns; False otherwise.
    """

    s = s.dropna()

    if s.empty:

        return True

    share_small = (s.abs() <= 0.5).mean()

    share_reasonable = (s.abs() <= 2.0).mean()

    return (share_small > 0.90) and (share_reasonable > 0.99)


def convert_prices_to_base_ccy(
    px_local: pd.Series,
    fx_series: Optional[pd.Series],
    conv: str = "local_to_base",
) -> pd.Series:
    """
    Convert price levels quoted in a local currency into a base currency using
    an FX series.

    Conventions
    -----------
    The function assumes the FX series is quoted as a multiplicative converter:
    
      - 'local_to_base': price_base = price_local * FX
    
                         Example: local = GBP, base = USD, FX = GBPUSD (USD per GBP).
    
      - 'base_to_local': price_base = price_local / FX
                         
                         Example: local = GBP, base = USD, FX = USDGBP (GBP per USD).

    Parameters
    ----------
    px_local : pandas.Series
        Local-currency prices indexed by date.
    fx_series : pandas.Series or None
        FX converter series aligned by date. If None, the input is returned unchanged.
    conv : {"local_to_base","base_to_local"}, default "local_to_base"
        Directionality of the FX series.

    Returns
    -------
    pandas.Series
        Prices expressed in the base currency.

    Notes
    -----
    - The series are inner-joined on dates to prevent spurious extrapolation.
    
    - FX handling is multiplicative at the price level so that subsequent
      returns computed from converted prices are FX-adjusted correctly.
    """
   
    px_local = px_local.sort_index()
   
    if fx_series is None:
        return px_local
   
    fx_series = fx_series.sort_index()
   
    df = pd.concat([px_local, fx_series], axis = 1, join = "inner").dropna()
   
    pl = df.iloc[:, 0]
    
    fx = df.iloc[:, 1]
    
    if conv == "local_to_base":
    
        return (pl * fx).rename(px_local.name)
    
    elif conv == "base_to_local":
    
        return (pl / fx).rename(px_local.name)
    
    else:
    
        raise ValueError("conv must be 'local_to_base' or 'base_to_local'")


def weekly_returns_in_base(
    px_local: pd.Series,
    fx_series: Optional[pd.Series],
    weekday: str = "W-FRI",
    conv: str = "local_to_base",
) -> pd.Series:
    """
    Convert a local-currency price series into a base currency (if an FX
    series is provided), resample to weekly frequency, and compute weekly returns.

    Method
    ------
    1) Convert prices with `convert_prices_to_base_ccy`, respecting 'conv'.
   
    2) Resample to weekly anchor weekday.
   
    3) Compute weekly simple returns r_t = P_t / P_{t-1} − 1.

    Parameters
    ----------
    px_local : pandas.Series
        Local-currency price series.
    fx_series : pandas.Series or None
        FX converter series. If None, prices are treated as already in base currency.
    weekday : str, default "W-FRI"
        Weekly anchor weekday.
    conv : {"local_to_base","base_to_local"}, default "local_to_base"
        Directionality of the FX series.

    Returns
    -------
    pandas.Series
        Weekly simple returns in the base currency.
    """
   
    px_base = convert_prices_to_base_ccy(
        px_local = px_local, 
        fx_series = fx_series, 
        conv = conv
    )
   
    return _weekly_returns_from_prices(
        px = px_base,
        weekday = weekday
    )


def _infer_base_ccy_from_default_pair(
    default_pair: str, 
    conv: str = "local_to_base"
) -> str:
    """
    Infer the base currency from a 6-letter pair like 'GBPUSD'.
    If conv='local_to_base' → base is last 3 chars; 'base_to_local' → first 3.
    """
    
    if isinstance(default_pair, str) and len(default_pair) >= 6:
    
        return default_pair[3:] if conv == "local_to_base" else default_pair[:3]
    
    return "USD"


def _convert_returns_to_base_ccy(
    r_local: pd.Series, 
    fx_price: Optional[pd.Series], 
    conv: str = "local_to_base"
) -> pd.Series:
    """
    Translate arithmetic *returns* from local into base currency using the FX *price* series.
   
      - conv = 'local_to_base':  
      
      (1 + R_base) = (1 + R_local) * (1 + R_fx)
   
      - conv = 'base_to_local':  
      
      (1 + R_base) = (1 + R_local) / (1 + R_fx)
   
    If fx_price is None, return the input unchanged.
    """
    
    if fx_price is None:
    
        return r_local.sort_index()
    
    r_fx = fx_price.sort_index().pct_change().reindex(r_local.index).fillna(0.0)
    
    if conv == "local_to_base":
    
        return (1.0 + r_local).mul(1.0 + r_fx, fill_value=0.0).sub(1.0)
    
    elif conv == "base_to_local":
    
        return (1.0 + r_local).div(1.0 + r_fx.replace(-1.0, np.nan)).sub(1.0)
    
    else:
    
        raise ValueError("conv must be 'local_to_base' or 'base_to_local'")
    

def _extract_param(
    res, 
    name: str, 
    fallback_pos: int = -1
) -> tuple[float, float]:
    """
    Extract a coefficient and its standard error from a statsmodels result,
    accommodating both labelled (pandas) and positional (numpy) storage.

    Parameters
    ----------
    res : statsmodels regression result
        The fitted model or robust-covariance wrapper.
    name : str
        Parameter name to retrieve (e.g., "b", "const").
    fallback_pos : int, default -1
        Positional index to use if the parameter lacks a label.

    Returns
    -------
    (coef, se) : tuple[float, float]
        Coefficient estimate and standard error, with NaN returned where unavailable.

    Notes
    -----
    This helper reduces fragility when statsmodels objects differ in how
    they expose parameters (e.g., through `.params`/`.bse` as arrays or Series).
    """
   
    params = res.params
   
    bse = res.bse

    try:

        if hasattr(params, "index") and (name in getattr(params, "index", [])):

            coef = float(params[name])

            se = float(bse[name]) if hasattr(bse, "__getitem__") else float(np.nan)

            return coef, se

    except Exception:

        pass

    try:

        coef = float(params[fallback_pos])
   
    except Exception:

        coef = float(params[-1]) 
    try:
  
        se = float(bse[fallback_pos])
  
    except Exception:
  
        se = float(bse[-1])
  
    return coef, se


def beta_hac_ols(
    stock_weekly: pd.Series,
    mkt_weekly: pd.Series,
    rf_per_week: float,
    winsor: float = 0.01,
    hac_lags: Optional[int] = None,
) -> Tuple[float, float, int]:
    """
    Estimate market beta by OLS on weekly excess returns with Newey–West
    (HAC) standard errors.

    Model
    -----
    Define weekly excess returns:
   
      y_t = R_i,t − rf_week,
   
      x_t = R_m,t − rf_week,
   
    and estimate the linear model:
   
      y_t = alpha + beta * x_t + epsilon_t.

    Estimator
    ---------
    - OLS coefficient:
   
        beta_hat = Cov(x, y) / Var(x).
   
    - The intercept alpha is included but not returned.

    Robust Standard Errors
    ----------------------
    Heteroskedasticity- and autocorrelation-consistent (HAC) standard errors
    are computed using the Newey–West estimator with bandwidth L:
   
      Var_HAC(beta_hat) = (X'X)^{-1} (X' S_X X) (X'X)^{-1},
   
    where S_X involves weighted sums of lagged score covariances up to L.
   
    The default lag is:
   
      L = max( floor( 4 * (T / 100)^{2/9} ), 1 ),
   
    a common plug-in rule.

    Pre-processing
    --------------
    - Both x_t and y_t are winsorised at quantiles [winsor, 1−winsor].

    Parameters
    ----------
    stock_weekly : pandas.Series
        Weekly simple returns for the stock.
    mkt_weekly : pandas.Series
        Weekly simple returns for the market index.
    rf_per_week : float
        Constant risk-free rate per week.
    winsor : float, default 0.01
        Two-sided winsorisation percentile.
    hac_lags : int or None
        Newey–West lag. If None, use the plug-in rule above.

    Returns
    -------
    beta : float
        OLS estimate of market beta.
    se : float
        HAC standard error of beta.
    T : int
        Number of observations used in the regression.
    """
   
    df = _align_returns(
        r1 = stock_weekly, 
        r2 = mkt_weekly
    )
    
    sp = df["p"] - rf_per_week
    
    sb = df["b"] - rf_per_week
    
    y = _winsorize(
        s = sp, 
        p = winsor
    ).rename("y")
   
    x = _winsorize(
        s = sb, 
        p = winsor
    ).rename("b")

    X = pd.concat([x], axis = 1)

    X = sm.add_constant(X) 

    ols = sm.OLS(y, X, missing = "drop").fit()
   
    T = int(ols.nobs)
   
    if hac_lags is None:
   
        hac_lags = max(int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))), 1)

    hac = ols.get_robustcov_results(cov_type = "HAC", maxlags = hac_lags)

    beta, se = _extract_param(
        res = hac,
        name = "b",
        fallback_pos = -1
    )
    
    return beta, se, T


def beta_dimson_hac(
    stock_weekly: pd.Series,
    mkt_weekly: pd.Series,
    rf_per_week: float,
    L: int = 1,
    winsor: float = 0.01,
    hac_lags: Optional[int] = None,
) -> Tuple[float, float, int, Dict[str, float]]:
    """
    Estimate Dimson beta to address non-synchronicity (stale prices) by
    regressing on multiple leads and lags of market excess returns, and compute
    a HAC standard error for the sum of coefficients.

    Model
    -----
    Let y_t = R_i,t − rf_week. For k = −L, ..., 0, ..., +L define:
    
      x_{t,k} = R_m,t+k − rf_week.
    
    Estimate
    
      y_t = alpha + Σ_{k=−L}^{L} beta_k x_{t,k} + epsilon_t,
    
    and define the Dimson beta as the sum:
    
      beta_Dimson = Σ_{k=−L}^{L} beta_k.

    Standard Error of the Sum
    -------------------------
    Let b be the vector of slope coefficients [beta_{−L}, ..., beta_{+L}]',
    and let V be the HAC covariance matrix of b. Then
    
      Var(beta_Dimson) = 1' V 1,
    
    where 1 is a vector of ones of conformable dimension. The standard
    error is se = sqrt(Var(beta_Dimson)).

    Pre-processing
    --------------
    - y_t and each x_{t,k} are winsorised at [winsor, 1−winsor].
    
    - Missing values introduced by shifting are dropped.

    Parameters
    ----------
    stock_weekly : pandas.Series
        Weekly simple returns for the stock.
    mkt_weekly : pandas.Series
        Weekly simple returns for the market index.
    rf_per_week : float
        Constant risk-free rate per week.
    L : int, default 1
        Number of symmetric leads/lags included.
    winsor : float, default 0.01
        Two-sided winsorisation percentile.
    hac_lags : int or None
        Newey–West lag for HAC covariance. If None, use the plug-in rule.

    Returns
    -------
    beta_dimson : float
        Sum of lead/lag betas Σ beta_k.
    se_dimson : float
        HAC standard error of the sum.
    T : int
        Number of observations used in the regression.
    coeffs : dict[str, float]
        Mapping from coefficient names (e.g., 'b_{−1}', 'b_0', 'b_{+1}') to estimates.

    Remarks
    -------
    - Dimson (1979) corrects bias when both the security and the market do not
      trade synchronously or when the weekly sampling induces measurement timing
      differences.
    """

    df = _align_returns(
        r1 = stock_weekly, 
        r2 = mkt_weekly
    )
    
    sp = df["p"] - rf_per_week
    
    y = _winsorize(
        s = sp, 
        p = winsor
    ).rename("y")

    X_parts = []
    
    names = []
    
    for k in range(-L, L + 1):
   
        name = f"b_{k}"
        
        sb = df["b"].shift(k) - rf_per_week
   
        xk = _winsorize(
            s = sb, 
            p = winsor
        ).rename(name)
   
        X_parts.append(xk)
   
        names.append(name)
   
    X = pd.concat(X_parts, axis = 1).dropna()
   
    y = y.loc[X.index]

    X = sm.add_constant(X)  

    ols = sm.OLS(y, X, missing  ="drop").fit()
    
    T = int(ols.nobs)
    
    if hac_lags is None:
    
        hac_lags = max(int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))), 1)

    hac = ols.get_robustcov_results(cov_type = "HAC", maxlags = hac_lags)

    coeffs: Dict[str, float] = {}

    for i, nm in enumerate(["const"] + names):

        if nm == "const":

            continue

        coef_i, _ = _extract_param(
            res = hac, 
            name = nm, 
            fallback_pos = i
        )

        coeffs[nm] = coef_i

    beta = float(sum(coeffs.values()))

    cov = hac.cov_params()
    
    if isinstance(cov, pd.DataFrame):
        
        cols = [nm for nm in names if nm in cov.columns]
       
        if cols:
       
            cov_sub = cov.loc[cols, cols]
       
            ones = np.ones((len(cols), 1))
          
            var_sum = float(ones.T @ cov_sub.values @ ones)
       
            se = np.sqrt(max(var_sum, 0.0))
       
        else:
       
            se = np.nan
    
    else:

        
        p = len(names)

        start = 1  
        
        stop = start + p
        
        cov_sub = np.asarray(cov)[start: stop, start: stop]
        
        ones = np.ones((p, 1))
        
        var_sum = float(ones.T @ cov_sub @ ones)
        
        se = np.sqrt(max(var_sum, 0.0))

    return beta, se, T, coeffs


def beta_kalman_random_walk(
    stock_weekly: pd.Series,
    mkt_weekly: pd.Series,
    rf_per_week: float,
    q: float = 5e-5,     
    r: Optional[float] = None,  
    winsor: float = 0.01,
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Estimate a time-varying market beta using a state-space model with a
    random-walk state and Gaussian observation noise, then apply a fixed-interval
    (Rauch–Tung–Striebel) smoother.

    State-Space Formulation
    -----------------------
    Measurement equation:
   
      y_t = alpha + beta_t * x_t + epsilon_t,
   
      epsilon_t ~ N(0, r).
   
    State equation (random walk):
   
      beta_t = beta_{t−1} + eta_t,
   
      eta_t ~ N(0, q).

    Here:
   
      y_t = R_i,t − rf_week,
   
      x_t = R_m,t − rf_week.

    Algorithm
    ---------
    1) Initialise alpha by OLS on (y_t, x_t); use the OLS beta as the initial
       state mean, and set the initial state variance to 1.0.
   
    2) For t = 1, ..., T, perform the Kalman filter recursions:
   
         Prediction:
   
           beta_{t|t − 1} = beta_{t − 1|t − 1}
   
           P_{t|t − 1} = P_{t − 1|t − 1} + q
   
         Update:
   
           S_t = P_{t|t − 1} * x_t^2 + r
   
           K_t = (P_{t|t − 1} * x_t) / S_t
   
           beta_{t|t} = beta_{t|t − 1} + K_t * ( (y_t − alpha) − x_t * beta_{t|t − 1} )
   
           P_{t|t} = (1 − K_t * x_t) * P_{t|t − 1}
   
    3) Apply fixed-interval smoothing backward for t = T − 1, ..., 1:
   
           C_t = P_{t|t} / (P_{t|t} + q)
   
           beta_{t|T} = beta_{t|t} + C_t * (beta_{t + 1|T} − beta_{t|t})
   
           P_{t|T} = P_{t|t} + C_t^2 * (P_{t + 1|T} − (P_{t|t} + q))


    Hyper-parameters
    ----------------
    q : state innovation variance. Larger q allows beta to vary more rapidly.
    r : observation noise variance. If None, r is set to the sample variance
        of residuals from the initial OLS fit.

    Pre-processing
    --------------
    - y_t and x_t are winsorised at [winsor, 1−winsor].

    Parameters
    ----------
    stock_weekly : pandas.Series
        Weekly simple returns for the stock.
    mkt_weekly : pandas.Series
        Weekly simple returns for the market index.
    rf_per_week : float
        Constant risk-free rate per week.
    q : float, default 5e-5
        State noise variance.
    r : float or None
        Observation noise variance. If None, estimated from OLS residuals.
    winsor : float, default 0.01
        Two-sided winsorisation percentile.

    Returns
    -------
    beta_smooth : pandas.Series
        Smoothed state sequence {beta_{t|T}}.
    var_beta_smooth : pandas.Series
        Smoothed state variances {P_{t|T}}.
    alpha_hat : float
        Intercept estimate from the initial OLS fit.

    Notes
    -----
    - The final “point” beta used elsewhere is typically the last smoothed
      value beta_{T|T}.
    """
   
    df = _align_returns(
        r1 = stock_weekly, 
        r2 = mkt_weekly
    )
    
    sp = df["p"] - rf_per_week
    
    sb = df["b"] - rf_per_week
    
    y = _winsorize(
        s = sp, 
        p = winsor
    )
   
    x = _winsorize(
        s = sb, 
        p = winsor
    )

    X = sm.add_constant(x)
    
    ols = sm.OLS(y, X, missing = "drop").fit()
    
    alpha_hat = float(ols.params["const"])
    
    beta_ols = float(ols.params.get("b", ols.params.iloc[-1]))
    
    resid = y - (alpha_hat + beta_ols * x)
    
    if r is None:
    
        r = float(resid.var(ddof = 1)) if len(resid) > 1 else 1e-4

    dates = y.index
    
    T = len(dates)
    
    beta_pred = np.zeros(T)
    
    P_pred = np.zeros(T)
    
    beta_filt = np.zeros(T)
    
    P_filt = np.zeros(T)

    beta_pred[0] = beta_ols
    
    P_pred[0] = 1.0

    xv = x.values
    
    yv = (y - alpha_hat).values
    
    for t in range(T):
    
        if t > 0:
    
            beta_pred[t] = beta_filt[t - 1]
          
            P_pred[t] = P_filt[t - 1] + q
    
        Ht = float(xv[t])
    
        St = P_pred[t] * (Ht ** 2) + r
    
        Kt = (P_pred[t] * Ht) / St
    
        beta_filt[t] = beta_pred[t] + Kt * (float(yv[t]) - Ht * beta_pred[t])
    
        P_filt[t] = (1.0 - Kt * Ht) * P_pred[t]

    beta_smooth = np.zeros(T)
    
    P_smooth = np.zeros(T)
    
    beta_smooth[-1] = beta_filt[-1]
    
    P_smooth[-1] = P_filt[-1]
    
    for t in range(T - 2, -1, -1):
    
        C = P_filt[t] / (P_filt[t] + q)
    
        beta_smooth[t] = beta_filt[t] + C * (beta_smooth[t + 1] - beta_filt[t])
    
        P_smooth[t] = P_filt[t] + (C ** 2) * (P_smooth[t + 1] - (P_filt[t] + q))

    return (
        pd.Series(beta_smooth, index = dates, name = "beta_kalman"),
        pd.Series(P_smooth, index = dates, name = "var_beta_kalman"),
        alpha_hat
    )


def combine_betas_inverse_variance(
    beta_ols: float, 
    var_ols: float,
    beta_dim: float, 
    var_dim: float,
    beta_kal: float, 
    var_kal: float,
    pref_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, float]:
    """
    Combine multiple beta estimators using inverse-variance (precision) weighting,
    optionally scaled by user preferences.

    Weighting Scheme
    ----------------
    Let the three beta estimates be b_i with variances v_i (i ∈ {OLS, Dimson, Kalman}),
    and preference multipliers p_i ≥ 0. Define weights:
   
      w_i = p_i / v_i,
   
    with the convention that non-finite or non-positive v_i are replaced by a
    large variance surrogate.

    The combined beta and its variance are:
   
      beta_combined = sum_i( w_i * b_i ) / sum_i( w_i ),
   
      Var(beta_combined) = 1 / sum_i( w_i ).

    Parameters
    ----------
    beta_ols, beta_dim, beta_kal : float
        Input beta estimates.
    var_ols, var_dim, var_kal : float
        Corresponding variance estimates.
    pref_weights : tuple of three floats, default (1,1,1)
        Relative user preferences for the three estimators.

    Returns
    -------
    (beta, var) : tuple[float, float]
        Combined beta and its variance.

    Robustness
    ----------
    - If all weights collapse to zero, the function falls back to the
      arithmetic mean of available betas and variances.
    """
    
    betas = np.array([beta_ols, beta_dim, beta_kal], dtype = float)
    
    vars_ = np.array([var_ols, var_dim, var_kal], dtype = float)
    
    prefs = np.array(pref_weights, dtype = float)

    vars_[~np.isfinite(vars_) | (vars_ <= 0)] = 1e3

    prefs[~np.isfinite(prefs) | (prefs <= 0)] = 0.0

    w = prefs / vars_

    if not np.isfinite(w).any() or np.allclose(w.sum(), 0.0):

        beta = float(np.nanmean(betas))

        var = float(np.nanmean(vars_))

        return beta, var

    beta = float(np.nansum(w * betas) / np.nansum(w))

    var = float(1.0 / np.nansum(w))

    return beta, var


def vasicek_shrinkage(
    betas: pd.Series,
    ses: pd.Series,
    mean_target: Optional[float] = None,
    min_tau2: float = 1e-6,
) -> Tuple[pd.Series, float]:
    """
    Apply Vasicek (empirical Bayes) shrinkage to a cross-section of beta
    estimates, shrinking noisy estimates toward a common mean.

    Model
    -----
    Suppose beta_i | μ ~ N(μ, τ^2) across names, and the observed estimate
    \hat{beta}_i has sampling variance s_i^2. The posterior mean (shrunken beta)
    is:
     
      beta_i* = w_i * \hat{beta}_i + (1 − w_i) * μ,
     
      w_i = τ^2 / (τ^2 + s_i^2).

    Estimation of τ^2
    -----------------
    Let Var_hat = sample variance of {\hat{beta}_i}, and let s_i^2 be the
    squared standard errors. A method-of-moments estimator is:
    
      τ^2_hat = max( Var_hat − mean(s_i^2), min_tau2 ),
    
    enforcing a non-negative floor.

    Mean Target
    -----------
    μ is set to the provided `mean_target` if not None; otherwise μ is the
    cross-sectional mean of \hat{beta}_i.

    Parameters
    ----------
    betas : pandas.Series
        Observed beta estimates.
    ses : pandas.Series
        Corresponding standard errors.
    mean_target : float or None
        Shrinkage target mean μ. If None, use sample mean of betas.
    min_tau2 : float, default 1e-6
        Minimum τ^2 to avoid degenerate weights.

    Returns
    -------
    (beta_star, tau2_hat) : tuple
        Shrunken beta series and the estimated τ^2.

    Practicalities
    --------------
    - Non-positive or missing standard errors are replaced by the positive
      median of available s_i to stabilise weights.
    """
   
    betas = betas.dropna()
   
    ses = ses.reindex(betas.index)

    pos_med = float(ses[ses > 0].median()) if (ses > 0).any() else 0.2
   
    ses = ses.fillna(pos_med)
   
    ses = ses.mask(ses <= 0, pos_med)

    mu = float(mean_target) if mean_target is not None else float(betas.mean())
   
    s2 = ses ** 2
   
    var_betas = float(betas.var(ddof = 1)) if len(betas) > 1 else 0.0
   
    tau2_hat = max(var_betas - float(s2.mean()), min_tau2)

    w = tau2_hat / (tau2_hat + s2)
   
    beta_star = w * betas + (1.0 - w) * mu
   
    return beta_star, tau2_hat


def unlever_beta_from_levered(
    beta_levered: float, 
    tax_rate: float,
    d_to_e: float
) -> float:
    """
    Compute the asset (unlevered) beta given the equity (levered) beta,
    corporate tax rate, and the debt-to-equity ratio using the standard
    Modigliani–Miller tax shield adjustment.

    Formula
    -------
      beta_unlevered = beta_levered / ( 1 + (1 − τ) * D / E ).

    Parameters
    ----------
    beta_levered : float
        Equity beta.
    tax_rate : float
        Corporate tax rate τ in [0, 1].
    d_to_e : float
        Debt-to-equity ratio D/E.

    Returns
    -------
    float
        Unlevered beta.
    """
    
    unlevered_beta = beta_levered / max(1.0 + (1.0 - tax_rate) * d_to_e, 1e-12)
    
    return unlevered_beta


def relever_beta_from_unlevered(
    beta_unlevered: float, 
    tax_rate: float, 
    d_to_e: float
) -> float:
    """
    Re-lever an asset beta back to an equity beta given a debt-to-equity
    ratio and tax rate.

    Formula
    -------
      beta_levered = beta_unlevered * ( 1 + (1 − τ) * D/E ).

    Parameters
    ----------
    beta_unlevered : float
        Asset beta.
    tax_rate : float
        Corporate tax rate τ in [0, 1].
    d_to_e : float
        Debt-to-equity ratio D/E.

    Returns
    -------
    float
        Equity beta after re-levering.
    """
    
    levered_beta =  beta_unlevered * (1.0 + (1.0 - tax_rate) * d_to_e)
    
    return levered_beta


def calculate_cost_of_equity(
    tickers: list[str],
    rf: float,
    returns: pd.DataFrame,                 
    index_close: pd.Series,               
    per_ticker_market_weekly: Optional[Dict[str, pd.Series]], 
    spx_expected_return: float,
    crp_df: pd.DataFrame,
    currency_bl_df: pd.Series | pd.DataFrame,
    country_to_pair: Dict[str, str],
    ticker_country_map: dict[str, str],
    tax_rate: pd.Series,
    d_to_e: pd.Series,
    debt: pd.Series,
    equity: pd.Series,
    r: object,
    fx_price_by_pair: Optional[Dict[str, pd.Series]] = None,  
    base_ccy_conv: str = "local_to_base",                
    weekly_day: str = "W-FRI",
    use_dimson_L: int = 1,
    kalman_q: float = 5e-5,
    winsor: float = 0.01,
    method_prefs: Tuple[float, float, float] = (1.0, 1.0, 1.0),  
    target_d_to_e: Optional[pd.Series] = None,                 
) -> pd.DataFrame:
    """
    Estimate the cost of equity (COE) for each ticker by:
     
      (i) constructing weekly excess returns,
     
      (ii) estimating multiple betas (OLS, Dimson, Kalman),
     
      (iii) combining betas by inverse-variance weighting,
     
      (iv) optionally re-levering to target capital structure, and
     
      (v) applying a CAPM-style premium augmented by country risk premium (CRP)
          and a currency premium.

    Return Construction
    -------------------
    For each ticker i:
     
      • If the provided `returns[i]` resembles prices (heuristic), first convert
        to the base currency using the tickers traded currency inferred from it's
        suffix to select the FX pair (fallback to country-based pair; then compute 
        weekly returns on a fixed weekday.
     
      • Otherwise treat the series as returns and align/sort.

    The market weekly return series is taken from `per_ticker_market_weekly[i]`
    when available; otherwise from `index_close` resampled to weekly.

    Weekly excess returns are defined as:
     
      y_t = R_i,t − RF_week,
     
      x_t = R_m,t − RF_week,
    
    where RF_week = config.RF_PER_WEEK.

    Beta Estimation
    ---------------
    1) OLS beta with HAC/Newey–West standard error:
   
         y_t = alpha + beta_OLS x_t + epsilon_t.
   
    2) Dimson beta with leads / lags k = −L, ...,+L:
   
         y_t = alpha + Σ_{k=−L}^{L} beta_k x_{t+k} + epsilon_t,
   
         beta_Dimson = Σ beta_k,
   
         se_Dimson from 1' V 1, with V the HAC covariance of {beta_k}.
   
    3) Kalman random-walk beta:
   
         y_t = alpha + beta_t x_t + epsilon_t,
   
         beta_t = beta_{t − 1} + eta_t,
   
         epsilon_t ~ N(0, r), eta_t ~ N(0, q),
   
       returning the smoothed beta at T (last date).

    Combining Betas
    ---------------
    The three betas are combined using inverse-variance weights scaled by
    user preferences p_i:

      w_i = p_i / Var_i,

      beta_combined = sum_i w_i * beta_i / sum_i w_i,

      Var(beta_combined) = 1 / sum_i w_i.

    Capital Structure Adjustment
    ----------------------------
    The combined beta is interpreted as a levered equity beta. If
    `target_d_to_e` is supplied, a re-levering is applied to align with the
    target debt-to-equity:
   
      beta_levered_used = beta_unlevered * ( 1 + (1 − τ) * D/E_target ).
   
    In this implementation the combined beta is treated as already levered,
    so the function directly uses a re-levered adjustment relative to the
    provided target.

    Premium Components and COE
    --------------------------
    Define the equity risk premium (ERP) as:
    
      ERP = max( spx_expected_return − rf, 0 ).
    
    For country c, define:
    
      CRP_c = country risk premium from `crp_df` (fallback to mean across countries),
    
      CP_c = currency premium for the FX pair mapped by `country_to_pair`
             and provided in `currency_bl_df` (0 for the United Kingdom by convention).

    The total additive premium for ticker i is:
    
      Premium_i = max( beta_levered_used_i * ERP + CRP_{c_i} + CP_{c_i}, 0 ),
    
    and the cost of equity is:
    
      COE_i = rf + Premium_i.

    Inputs
    ------
    tickers : list[str]
        Universe of tickers to process.
    rf : float
        Annual risk-free rate used in the final COE construction.
    returns : pandas.DataFrame
        Either price levels or returns by column; detection is heuristic.
    index_close : pandas.Series
        Benchmark price series for the market.
    per_ticker_market_weekly : dict[str, pandas.Series] or None
        Optional per-ticker market weekly returns (overrides `index_close`).
    spx_expected_return : float
        Expected annual return for the S&P 500 (or chosen market proxy),
        used to form ERP = max(spx_expected_return − rf, 0).
    crp_df : pandas.DataFrame
        Table with a 'CRP' column indexed by country names.
    currency_bl_df : pandas.Series or pandas.DataFrame
        Currency premium per FX pair (e.g., 'GBPUSD'); if a DataFrame, pairs
        are looked up in the index and reduced to scalars.
    country_to_pair : dict[str, str]
        Map from country name to FX pair code (e.g., 'United States' → 'GBPUSD').
    ticker_country_map : dict[str, str]
        Map from ticker to country.
    tax_rate : pandas.Series
        Corporate tax rate per ticker τ in [0, 1].
    d_to_e : pandas.Series
        Current debt-to-equity ratio D/E per ticker.
    debt, equity : pandas.Series
        Debt and equity levels; carried to the output for reference.
    fx_price_by_pair : dict[str, pandas.Series] or None
        FX price converters used for price-level currency conversion.
    base_ccy_conv : {"local_to_base","base_to_local"}, default "local_to_base"
        FX directionality in price conversion.
    weekly_day : str, default "W-FRI"
        Weekly anchor weekday for resampling.
    use_dimson_L : int, default 1
        Number of lead/lag weeks in the Dimson regression.
    kalman_q : float, default 5e-5
        State innovation variance for the random-walk beta.
    winsor : float, default 0.01
        Two-sided winsorisation percentile applied to inputs in regressions.
    method_prefs : tuple[float, float, float], default (1,1,1)
        Preference multipliers for (OLS, Dimson, Kalman) in the combination step.
    target_d_to_e : pandas.Series or None
        Optional target D/E used to re-lever the combined beta.

    Returns
    -------
    pandas.DataFrame
        Index by ticker with columns:
          • Country
          • Beta_OLS, SE_OLS
          • Beta_Dimson, SE_Dimson
          • Beta_Kalman, SE_Kalman
          • Beta_Combined, SE_Combined
          • Beta_Levered_Used
          • TaxRate, D_to_E_Current, D_to_E_Target
          • CRP, Currency Premium
          • COE
          • Debt, Equity

    Implementation Notes
    --------------------
    - Observations with fewer than 30 aligned weeks are skipped.
    
    - ERP and the total premium are floored at zero to avoid negative
      premia entering COE.
    
    - Currency conversion is performed at the price level (multiplicative)
      before computing returns, which correctly introduces FX effects into
      returns; if `returns` are already currency-aligned, no conversion is applied.
    """

    index_weekly = _weekly_returns_from_prices(
        px = index_close, 
        weekday = weekly_day
    )

    is_price_like = {}

    for t in tickers:

        if t in returns.columns:

            is_price_like[t] = not _looks_like_returns(
                s = returns[t]
            )

    recs = []

    beta_comb_map: dict[str, float] = {}

    se_comb_map: dict[str, float] = {}

    raw_ols_map: dict[str, float] = {}

    rf_w = float(config.RF_PER_WEEK)
    
    default_pair = "GBPUSD"

    for t in tickers:
      
        if t not in returns.columns:
      
            continue

        ser = returns[t].dropna()
      
        if ser.empty:
      
            continue

        ctry = ticker_country_map.get(t, "NA")
       
        ccy_pair = country_to_pair.get(ctry, default_pair)

        if is_price_like.get(t, False):

            fx_series = None

            if fx_price_by_pair is not None:

                local_ccy = r._ticker_currency_from_suffix(
                    ticker = t
                )

                base_ccy  = _infer_base_ccy_from_default_pair(
                    default_pair = default_pair, 
                    conv = base_ccy_conv
                )

                pair_by_ticker = (local_ccy + base_ccy) if base_ccy_conv == "local_to_base" else (base_ccy + local_ccy)

                fx_series = fx_price_by_pair.get(pair_by_ticker)

                if fx_series is None:

                    ccy_pair = country_to_pair.get(ctry, default_pair)

                    fx_series = fx_price_by_pair.get(ccy_pair)

            if fx_series is not None:
                
                px_base = convert_prices_to_base_ccy(
                    px_local = ser,
                    fx_series = fx_series,
                    conv = base_ccy_conv
                )
                
                stock_weekly = _weekly_returns_from_prices(
                    px = px_base,
                    weekday = weekly_day
                )
                
            else:

                stock_weekly = _weekly_returns_from_prices(
                    px = ser,
                    weekday = weekly_day
                )

        else:

            r_local = ser.sort_index()

            fx_series = None

            if fx_price_by_pair is not None:
                
                local_ccy = r._ticker_currency_from_suffix(
                    ticker = t
                )

                base_ccy  = _infer_base_ccy_from_default_pair(
                    default_pair = default_pair, 
                    conv = base_ccy_conv
                )
                
                pair_by_ticker = (local_ccy + base_ccy) if base_ccy_conv == "local_to_base" else (base_ccy + local_ccy)
                
                fx_series = fx_price_by_pair.get(pair_by_ticker)
                
                if fx_series is None:
                
                    ccy_pair = country_to_pair.get(ctry, default_pair)
                
                    fx_series = fx_price_by_pair.get(ccy_pair)
            
            stock_weekly = _convert_returns_to_base_ccy(
                r_local = r_local, 
                fx_price = fx_series, 
                conv = base_ccy_conv
            )

        mkt_w = None

        if per_ticker_market_weekly is not None:

            mkt_w = per_ticker_market_weekly.get(t, None)

        if mkt_w is None:

            index_weekly = _weekly_returns_from_prices(
                px = index_close, 
                weekday = weekly_day
            )
        
        else:
        
            index_weekly = mkt_w

        df = pd.concat([stock_weekly, index_weekly], axis = 1, join = "inner").dropna()

        if len(df) < 30:

            continue

        stock_w = df.iloc[:, 0]
       
        mkt_w = df.iloc[:, 1]

        b_ols, se_ols, _ = beta_hac_ols(
            stock_weekly = stock_w,
            mkt_weekly = mkt_w, 
            rf_per_week = rf_w, 
            winsor = winsor
        )
       
        v_ols = se_ols ** 2 if np.isfinite(se_ols) else np.nan
       
        raw_ols_map[t] = b_ols

        b_dim, se_dim, _, _ = beta_dimson_hac(
            stock_weekly = stock_w,
            mkt_weekly = mkt_w,
            rf_per_week = rf_w,
            L = use_dimson_L, 
            winsor = winsor
        )
        
        v_dim = se_dim ** 2 if np.isfinite(se_dim) else np.nan

        beta_path, var_path, _ = beta_kalman_random_walk(
            stock_weekly = stock_w, 
            mkt_weekly = mkt_w, 
            rf_per_week = rf_w, 
            q = kalman_q, 
            winsor = winsor
        )
        
        b_kal = float(beta_path.iloc[-1])
        
        v_kal = float(var_path.iloc[-1])

        b_cmb, v_cmb = combine_betas_inverse_variance(
            beta_ols = b_ols,
            var_ols = v_ols,
            beta_dim = b_dim,
            var_dim = v_dim, 
            beta_kal = b_kal, 
            var_kal = v_kal, 
            pref_weights = method_prefs
        )
        
        se_cmb = float(np.sqrt(v_cmb)) if np.isfinite(v_cmb) and v_cmb >= 0 else np.nan

        beta_comb_map[t] = b_cmb
        
        se_comb_map[t] = se_cmb

        recs.append({
            "Ticker": t,
            "Country": ctry,
            "Beta_OLS": b_ols,    
            "SE_OLS": se_ols,
            "Beta_Dimson": b_dim,  
            "SE_Dimson": se_dim,
            "Beta_Kalman": b_kal,  
            "SE_Kalman": (np.sqrt(v_kal) if np.isfinite(v_kal) and v_kal >= 0 else np.nan),
            "Beta_Combined": b_cmb,
            "SE_Combined": se_cmb,
        })

    beta_df = pd.DataFrame(recs).set_index("Ticker")

    if beta_df.empty:
        
        return beta_df  

    levered_used: dict[str, float] = {}

    dte_target_out: dict[str, float] = {}

    for t in beta_df.index:
        
        beta_levered_combined = float(beta_df.loc[t, "Beta_Combined"])
       
        tau_t = float(tax_rate.get(t, 0.20))
       
        dte_cur = float(d_to_e.get(t, 0.0))
       
        dte_tgt = float(target_d_to_e.get(t, dte_cur)) if target_d_to_e is not None else dte_cur

        beta_relev = relever_beta_from_unlevered(
            beta_unlevered = beta_levered_combined,
            tax_rate = tau_t, 
            d_to_e = dte_tgt
        )

        levered_used[t] = beta_relev

        dte_target_out[t] = dte_tgt

    beta_df["Beta_Levered_Used"] = pd.Series(levered_used)

    beta_df["TaxRate"] = pd.Series({t: float(tax_rate.get(t, 0.20)) for t in beta_df.index})

    beta_df["D_to_E_Current"] = pd.Series({t: float(d_to_e.get(t, 0.0)) for t in beta_df.index})

    beta_df["D_to_E_Target"] = pd.Series(dte_target_out)

    crp_mean = float(crp_df["CRP"].mean()) if "CRP" in crp_df.columns else 0.0

    coe_rows = []
    
    for t in beta_df.index:
    
        ctry = beta_df.loc[t, "Country"]
    
        crp_val = float(crp_df["CRP"].get(ctry, crp_mean)) if "CRP" in crp_df.columns else 0.0

        if ctry == "United Kingdom":

            curr_prem = 0.0

        else:

            pair = country_to_pair.get(ctry, default_pair)

            if isinstance(currency_bl_df, pd.DataFrame):

                if pair in currency_bl_df.index:

                    curr_prem = float(currency_bl_df.loc[pair].squeeze())

                else:

                    curr_prem = 0.0

            else:

                curr_prem = float(currency_bl_df.get(pair, 0.0))

        beta_used = float(beta_df.loc[t, "Beta_Levered_Used"])

        erp = max(spx_expected_return - rf, 0.0)

        prem = max(beta_used * erp + crp_val + curr_prem, 0.0)

        coe = rf + prem

        coe_rows.append({
            "Ticker": t,
            "Country": ctry,
            "CRP": crp_val,
            "Currency Premium": curr_prem,
            "COE": coe
        })

    coe_df = pd.DataFrame(coe_rows).set_index("Ticker")

    if "Country" in coe_df.columns:
        
        coe_df = coe_df.rename(columns = {"Country": "Country_from_crp"})

    out = beta_df.join(coe_df, how = "left")

    out["Debt"] = pd.Series({t: float(debt.get(t, np.nan)) for t in out.index})

    out["Equity"] = pd.Series({t: float(equity.get(t, np.nan)) for t in out.index})

    cols_first = [
        "Country",
        "Beta_OLS",
        "Beta_Dimson", 
        "Beta_Kalman",
        "Beta_Combined", 
        "SE_Combined",
        "Beta_Levered_Used",
        "TaxRate", 
        "D_to_E_Current", 
        "D_to_E_Target",
        "CRP",
        "Currency Premium", 
        "COE",
        "Debt", 
        "Equity",
    ]
    
    cols_first = [c for c in cols_first if c in out.columns]
    
    other_cols = [c for c in out.columns if c not in cols_first]
    
    out = out[cols_first + other_cols]

    return out
