from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ratio_data import RatioData
import config

from coe2 import (
    beta_hac_ols,
    beta_dimson_hac,
    beta_kalman_random_walk,
    combine_betas_inverse_variance,
)


r = RatioData()


def _to_series(
    s
) -> pd.Series:
    """
    Coerce arbitrary 1-dimensional input to a pandas Series, drop missing values,
    and sort by index ascending.

    Purpose
    -------
    Normalises heterogeneous inputs (lists, arrays, existing Series) into a
    consistent, clean time series suitable for regression alignment.

    Returns
    -------
    pandas.Series
        A strictly increasing-index series with NaNs removed.

    Notes
    -----
    - Preserves the original index where present; otherwise creates a default
      RangeIndex, then sorts.
    
    - Sorting ensures subsequent inner joins across multiple series time-align properly.
    """
    
    return pd.Series(s).dropna().sort_index()


def _add_const(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Add an explicit intercept column named 'const' to a regressor matrix.

    Rationale
    ---------
    Linear models y = α + Xβ + ε require an intercept term α to absorb the mean.
    statsmodels' `add_constant(..., has_constant="add")` ensures the column is present
    and named 'const', which simplifies downstream parameter extraction and
    variance propagation.

    Returns
    -------
    pandas.DataFrame
        X augmented with a leading intercept column.
    """
    
    return sm.add_constant(X, has_constant = "add")


def _daily_from_annual(
    x_ann: float
) -> float:
    """
    Convert an annual simple return to an equivalent *daily* simple return under
    geometric compounding.

    Equation
    --------
    Let R_ann be the (simple) annual return. The daily (simple) return R_day satisfies:
   
        (1 + R_ann) = (1 + R_day)^(N),
   
    with N = 252 (trading days). Hence:
   
        R_day = (1 + R_ann)^(1 / 252) − 1.

    Parameters
    ----------
    x_ann : float
        Annual simple return.

    Returns
    -------
    float
        Daily simple return consistent with geometric compounding.
    """
    
    return float((1.0 + float(x_ann)) ** (1.0 / 252.0) - 1.0)


def _excess_daily_from_annual_total(
    x_ann_total: float, 
    rf_ann: float
) -> float:
    """
    Convert an *annual total* return into an *annual excess* return by subtracting
    the annual risk-free rate, then convert the latter to a *daily* simple return.

    Steps
    -----
    1) Annual excess return:
   
         R_ex_ann = R_tot_ann − r_f_ann.
   
    2) Daily excess (simple) return:
   
         r_ex_day = (1 + R_ex_ann)^(1/252) − 1.

    Parameters
    ----------
    x_ann_total : float
        Annual total (simple) expected return.
    rf_ann : float
        Annual risk-free rate (simple).

    Returns
    -------
    float
        Daily simple excess return aligned to 252 trading days.
    """
    
    x_ann_excess = float(x_ann_total) - float(rf_ann)
    
    return _daily_from_annual(
        x_ann = x_ann_excess
    )
    

def _residualise(
    y: pd.Series,
    X: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, sm.OLS]:
    """
    Residualise a target series on a regressor matrix using the
    Frisch–Waugh–Lovell (FWL) theorem.

    Model
    -----
    y = α + Xβ + ε.

    Outputs
    -------
    - y_res : residuals 
    
        ε̂ = y − ŷ.
        
    - y_hat : fitted values 
    
        ŷ = α̂ + Xβ̂.
        
    - fit   : the fitted OLS result (unrobust).

    Notes on FWL
    ------------
    FWL states that the coefficient of a covariate in a multivariate OLS equals
    the coefficient from regressing the residualised dependent variable
    (after projecting out other regressors) on the residualised covariate.
    
    This function performs the first projection step for the dependent variable.
    """
    
    Xc = _add_const(
        X = X
    )
    
    fit = sm.OLS(y, Xc, missing = "drop").fit()
    
    y_hat = pd.Series(fit.fittedvalues, index = fit.fittedvalues.index)
    
    y_res = (y - y_hat).dropna()
    
    return y_res, y_hat, fit


def _partial_out(
    series: pd.Series, 
    others: pd.DataFrame
) -> pd.Series:
    """
    Residualise `series` on the columns of `others` via the
    Frisch–Waugh–Lovell theorem (partialling-out).

    Operation
    ---------
    If others = [Z], compute:
   
        series_res = series − P_Z(series),
   
    where P_Z denotes the OLS projection onto the column space of Z (with intercept).
    When `others` is empty, the input series is simply cleaned (NaNs dropped, sorted).

    Returns
    -------
    pandas.Series
        Residualised series aligned to the intersection of indices.

    Practical impact
    ----------------
    Isolates the unique variation in a regressor or dependent variable that is
    orthogonal to other controls, ensuring per-regressor beta estimation reflects
    marginal, not joint, effects.
    """
   
    if others.shape[1] == 0:
   
        return _to_series(
            s = series
        )
   
    s = _to_series(
        s = series
    )
   
    df = pd.concat([s, others], axis=1, join = "inner").dropna()
   
    y = df.iloc[:, 0]
   
    X = df.iloc[:, 1:]
   
    res, _, _ = _residualise(
        y = y, 
        X = X
    )
   
    return res


def _blend_beta_for_regressor(
    y_excess: pd.Series,
    x: pd.Series,
    others_matrix: pd.DataFrame,
    *,
    rf_per_day: float,
    winsor: float,
    L_dimson: int,
    kalman_q: float,
    pref_weights: Tuple[float, float, float],
    hac_lags_for_daily: int,
) -> Tuple[float, float, Dict[str, float], int]:
    """
    Estimate a *per-regressor* sensitivity (beta) of a daily excess return series
    to a candidate driver using FWL residualisation and a multi-estimator blend.

    Pipeline
    --------
    1) Partialling-out (FWL):
     
       - x_res = residualise(x | others),  y_res = residualise(y_excess | others).
    
       - This isolates the marginal effect of x controlling for `others`.
    
    2) Three beta estimators on (y_res, x_res):
    
       (a) HAC OLS beta:
    
           y_t = α + β x_t + ε_t,
    
           β̂_OLS estimated by OLS; Newey–West/HAC standard errors with `maxlags`
           = `hac_lags_for_daily` guard against heteroskedasticity and serial correlation.
           Winsorisation at both tails (prob = `winsor`) reduces outlier influence.
    
       (b) Dimson beta (lead/lag correction):
    
           y_t = α + Σ_{k=-L..L} β_k x_{t+k} + ε_t,
    
           β̂_Dimson = Σ_k β̂_k (sum of coefficients). 
           
           This attenuates biases due to non-synchronous trading and microstructure delays; 
           `L_dimson` controls ±lead/lag span.
           
           HAC covariance yields a robust s.e. for the sum.
           
       (c) Kalman random-walk beta (time-varying):
           
           State-space model:
           
             Observation:  
             
                y_t = α + β_t x_t + ε_t,   ε_t ~ (0, σ^2)
           
             State:        
             
                β_t = β_{t-1} + η_t,       η_t ~ (0, q)
                
           with process variance q = `kalman_q`. Filtering + fixed-interval smoothing
           provide {β̂_t}; the terminal β̂_T and its variance Var(β̂_T) are used here.
          
           This captures gradual evolution in beta.
    
    3) Inverse-variance blending:
    
       Let (β_i, v_i) for i ∈ {OLS, Dimson, Kalman}. Define preferences p_i ≥ 0
       (from `pref_weights`). 
       
       Weights:
          
           w_i = p_i / v_i,
      
       β̂_blend = Σ_i w_i β_i / Σ_i w_i,
       
       Var(β̂_blend) = 1 / Σ_i w_i.
       
       When v_i is missing/non-finite, a large placeholder variance is used to
       down-weight that estimator.

    Inputs
    ------
    y_excess : pandas.Series
        Daily *excess* returns for the dependent variable (already minus risk-free).
        In this routine, `rf_per_day` is not used because residuals are regressed with
        rf=0 explicitly.
    x : pandas.Series
        Candidate driver (e.g., index excess return, factor return).
    others_matrix : pandas.DataFrame
        Control regressors to partial-out via FWL.
    rf_per_day : float
        Daily risk-free rate (not applied inside the residual regressions here).
    winsor : float
        Two-sided winsorisation probability (p) for both tails before OLS.
    L_dimson : int
        Number of leads/lags for the Dimson specification (±L).
    kalman_q : float
        State innovation variance in the random-walk beta model.
    pref_weights : tuple of 3 floats
        Relative preferences for (OLS, Dimson, Kalman) in blending.
    hac_lags_for_daily : int
        Newey–West/HAC maximum lag for daily data.

    Returns
    -------
    (beta_blend, se_blend, details, nobs) : tuple
        beta_blend : float
            Inverse-variance combined beta.
        se_blend : float
            √Var(beta_blend).
        details : dict
            Diagnostic map with component betas, s.e.s, and sample size.
        nobs : int
            Effective number of aligned observations used.

    Modelling considerations
    ------------------------
    - HAC (Newey–West) robustifies inference to mild serial correlation induced by
      partial-out operations and overlapping effects.
    
    - Dimson’s correction directly targets attenuation bias from timing mismatch.
    
    - The Kalman filter provides a structural method for slowly varying loadings.
    
    - Blending stabilises estimation by pooling complementary estimators with
      reliability weighting.
    """

    x_res = _partial_out(
        series = x, 
        others = others_matrix
    )

    y_res = _partial_out(
        series = y_excess, 
        others = others_matrix
    )

    df = pd.concat([y_res, x_res], axis = 1, join = "inner").dropna()
   
    if len(df) < 30:
   
        return np.nan, np.nan, {"nobs": len(df)}, len(df)

    y_r = df.iloc[:, 0]
  
    x_r = df.iloc[:, 1]
    
    b_ols, se_ols, _ = beta_hac_ols(
        stock_weekly = y_r,
        mkt_weekly = x_r,
        rf_per_week = 0.0,         
        winsor = winsor,
        hac_lags = hac_lags_for_daily,
    )

    b_dim, se_dim, _, _ = beta_dimson_hac(
        stock_weekly = y_r,
        mkt_weekly = x_r,
        rf_per_week = 0.0,         
        L = L_dimson,
        winsor = winsor,
        hac_lags = hac_lags_for_daily,
    )

    beta_path, var_path, _ = beta_kalman_random_walk(
        stock_weekly = y_r,
        mkt_weekly = x_r,
        rf_per_week = 0.0,       
        q = kalman_q,
        winsor = winsor,
    )
    
    b_kal = float(beta_path.iloc[-1]) if len(beta_path) else np.nan
    
    v_kal = float(var_path.iloc[-1]) if len(var_path) else np.nan
    
    se_kal = float(np.sqrt(v_kal)) if np.isfinite(v_kal) and v_kal >= 0 else np.nan

    v_ols = se_ols ** 2 if np.isfinite(se_ols) else np.nan
   
    v_dim = se_dim ** 2 if np.isfinite(se_dim) else np.nan
   
    b_cmb, v_cmb = combine_betas_inverse_variance(
        beta_ols = b_ols, 
        var_ols = v_ols,
        beta_dim = b_dim, 
        var_dim = v_dim, 
        beta_kal = b_kal,
        var_kal = v_kal,
        pref_weights = pref_weights
    )
    
    se_cmb = float(np.sqrt(v_cmb)) if np.isfinite(v_cmb) and v_cmb >= 0 else np.nan

    details = {
        "beta_ols": b_ols, "se_ols": se_ols,
        "beta_dim": b_dim, "se_dim": se_dim,
        "beta_kal": b_kal, "se_kal": se_kal,
        "beta_combined": b_cmb, "se_combined": se_cmb,
        "nobs": len(df),
    }
    
    return b_cmb, se_cmb, details, len(df)


def _extract_param_and_var(
    res, 
    name: str, 
    fallback_pos: int = 0
) -> tuple[float, float]:
    """
    Extract a parameter’s point estimate and variance from a fitted statsmodels result.

    Behaviour
    ---------
    Works across both pandas-labelled (`.params` as Series; `.cov_params()` as DataFrame)
    and positional numpy arrays. If `name` is not present, the positional index
    `fallback_pos` is used.

    Returns
    -------
    (coef, var) : tuple[float, float]
        Parameter estimate and the corresponding diagonal element of the covariance matrix.
    """
    
    params = res.params
   
    cov = res.cov_params()

    try:

        if hasattr(params, "index") and name in params.index:

            coef = float(params[name])

            var = float(cov.loc[name, name]) if hasattr(cov, "loc") else float(np.nan)

            return coef, var

    except Exception:

        pass

    try:

        coef = float(params[fallback_pos])

    except Exception:

        coef = float(params[0])

    try:

        if hasattr(cov, "__getitem__"):

            var = float(cov[fallback_pos, fallback_pos])

        else:

            var = float(np.nan)

    except Exception:

        var = float(np.nan)

    return coef, var


def exp_fac_reg(
    tickers: Optional[List[str]] = None,
    *,
    expected_scale: str = "annual",
    L_dimson: int = 1,
    kalman_q: float = 5e-5,
    winsor: float = 0.01,
    method_prefs: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hac_lags_for_daily: int = 10,
    td_per_year: int = 252,
) -> pd.DataFrame:
    """
    Cross-sectional expected-return synthesis by regressing each ticker’s *daily excess*
    return on a set of drivers, *blending* per-regressor betas, and *projecting* the
    expectation using forward inputs. Produces an annualised total expected return with
    an uncertainty estimate.

    Data assembly
    -------------
    For each ticker t:
   
    - Build a daily design from `RatioData.exp_factor_data(t)` comprising:
   
        y_t  = 'Ticker Excess Return',
   
        X_t  ∈ { 'Index Excess Return', 'Sector Return', 'Industry Return',
                 'MTUM','QUAL','SIZE','USMV','VLUE' } (subset, as available).
   
      These series are pre-translated into the ticker’s local currency and already
      excessed for the index series (−RF_PER_DAY).
   
    - Obtain annual expected values μ (industry/sector/index/factors) from
      `RatioData.exp_factors(t)`; map {MTUM→Momentum, QUAL→Quality, …}.

    Stage A — baseline OLS with HAC on the full design
    --------------------------------------------------
   
    Fit:
   
        y = α + Xβ + ε,
   
    by OLS; compute HAC (Newey–West) covariance with `maxlags = hac_lags_for_daily`.
    Extract:
   
        α̂ (intercept), Var(α̂), R².

    Stage B — per-regressor beta *blending* (marginal effects)
    ----------------------------------------------------------
    For each included regressor j ∈ X:
   
      1) Apply FWL to obtain residuals (y_res, x_j_res | X_{−j}).
   
      2) Estimate three β_j variants on (y_res, x_j_res):
   
         • OLS with HAC s.e.,
   
         • Dimson (±L_dimson) with HAC s.e. on the sum,
   
         • Kalman random-walk β_t (terminal β̂_T with variance).
   
      3) Combine by inverse-variance with preferences p_i:
   
           w_i = p_i / v_i,
   
           β̂_j,blend = Σ_i w_i β̂_{j,i} / Σ_i w_i,
   
           Var(β̂_j,blend) = 1 / Σ_i w_i.

    Forecast construction (daily scale)
    -----------------------------------
    Convert annual expectations to daily:
   
      - For total *index* expectation, convert to daily *excess*:
   
          μ_day['Index Excess Return'] = (1 + μ_ann_index − rf_ann)^(1/252) − 1.
   
      - For other drivers (sector/industry/factors), convert annual simple to daily simple:
   
          μ_day[k] = (1 + μ_ann[k])^(1/252) − 1.
   
    Form the daily expected excess return:
   
        E[r_ex,day] = α̂ + Σ_j β̂_j,blend · μ_day[j].

    Annualisation to total return
    -----------------------------
    Convert the daily *total* expectation using geometric compounding:
   
        E[R_tot,ann] = (1 + rf_day + E[r_ex,day])^(td_per_year) − 1,
   
    where rf_day = config.RF_PER_DAY, td_per_year = 252.

    Uncertainty propagation (delta method)
    --------------------------------------
    Let θ = [α, β'] be the parameter vector from the HAC fit and
    Σ = Cov_HAC(θ). Let z be the vector of scenario means with
    z = [1, μ_day'] corresponding to [α, β'].
   
    The forecast variance per day is approximated by:
   
        Var(ŷ_day) ≈ zᵀ Σ z + σ²,
   
    where σ² = OLS residual variance (mse_resid) accounts for idiosyncratic noise
    around the conditional mean. The daily standard error is:
   
        se_day = sqrt(Var(ŷ_day)).
   
    Annual standard error adopts a √N scaling:
   
        se_ann = sqrt(td_per_year) · se_day,
   
    which is conservative under weak dependence and arises from a linearisation
    of the compounding map around the mean (delta method).

    Filters and guards
    ------------------
    - Requires at least 60 aligned daily observations; otherwise returns zeros and NaN r².
    
    - For each regressor’s blend, requires at least 30 aligned residual observations;
      otherwise returns NaN for that beta.

    Parameters
    ----------
    tickers : list[str] | None
        Universe; defaults to `config.tickers`.
    expected_scale : {'annual'}
        Reserved for future scaling options; currently expectations are annual inputs.
    L_dimson : int
        Number of symmetrical leads/lags in Dimson beta (±L).
    kalman_q : float
        State innovation variance for the time-varying beta model.
    winsor : float
        Two-sided winsor probability applied inside component beta estimators.
    method_prefs : tuple[float, float, float]
        Preferences for (OLS, Dimson, Kalman) in the inverse-variance blend.
    hac_lags_for_daily : int
        HAC max lag for daily regressions to accommodate short-run dependence.
    td_per_year : int
        Trading days per year (default 252) for scaling and compounding.

    Returns
    -------
    pandas.DataFrame indexed by ticker with columns:
    
      - 'Returns'        : annual total expected return,
    
      - 'SE'             : approximate annual standard error (delta-method),
      
      - 'r2'             : OLS R² from the baseline fit,
    
      - 'nobs_min'       : minimum per-regressor sample size used in blending,
    
      - 'alpha'          : baseline intercept α̂ (daily),
    
      - 'beta[<name>]'   : blended beta for each included regressor.

    Modelling remarks
    -----------------
    - The separation of baseline HAC-OLS (for Σ) and per-regressor blending
      (for β_j) deliberately balances covariance realism with estimator robustness.
   
    - Dimson’s correction is retained even at daily frequency to mitigate
      asynchronous prints and closing-time mismatches across markets.
   
    - The Kalman random-walk beta accommodates structural drift in exposures
      (e.g., business mix shifts) without over-fitting.
    """

    if tickers is None:
    
        tickers = list(config.tickers)

    factor_data_dict = r.exp_factor_data(
        tickers = tickers
    )
    
    factor_preds_df = r.exp_factors(
        tickers = tickers
    )

    y_col = "Ticker Excess Return"
  
    base_regressors = [
        "Index Excess Return", 
        "Sector Return", 
        "Industry Return",
        "MTUM", 
        "QUAL",
        "SIZE", 
        "USMV",
        "VLUE",
    ]
    
    fac_map = {
        "MTUM": "Momentum",
        "QUAL": "Quality",
        "SIZE": "Size",
        "USMV": "Volatility",
        "VLUE": "Value"
    }

    rf_day = float(config.RF_PER_DAY)
    
    rf_ann = float(config.RF)

    rows: List[Dict[str, float]] = []

    for t in sorted(set(factor_data_dict.keys()) & set(factor_preds_df.index)):
    
        df = factor_data_dict[t].copy()
    
        if y_col not in df.columns or df[y_col].dropna().empty:
    
            rows.append({
                "Ticker": t, 
                "Returns": 0.0,
                "SE": 0.0, 
                "r2": np.nan
            })
    
            continue

        X_cols = [c for c in base_regressors if c in df.columns]
       
        if not X_cols:
       
            rows.append({
                "Ticker": t,
                "Returns": 0.0, 
                "SE": 0.0, 
                "r2": np.nan
            })
       
            continue

        sub = df[[y_col] + X_cols].dropna()
      
        if len(sub) < 60:
      
            rows.append({
                "Ticker": t, 
                "Returns": 0.0, 
                "SE": 0.0,
                "r2": np.nan
            })
      
            continue

        y = sub[y_col]
      
        X = sub[X_cols]
      
        Xc = _add_const(
            X = X
        )
      
        ols = sm.OLS(y, Xc).fit()
      
        hac = ols.get_robustcov_results(cov_type = "HAC", maxlags = hac_lags_for_daily)

        alpha, var_alpha = _extract_param_and_var(
            res = hac,
            name = "const", 
            fallback_pos = 0
        )
        
        r2 = float(ols.rsquared)

        betas_blend: Dict[str, float] = {}
        
        ses_blend: Dict[str, float] = {}
        
        nobs_min = len(sub)

        for c in X_cols:
           
            others = X.drop(columns = [c])
           
            b_cmb, se_cmb, _info, nobs = _blend_beta_for_regressor(
                y_excess = y,
                x = X[c],
                others_matrix = others,
                rf_per_day = rf_day,
                winsor = winsor,
                L_dimson = L_dimson,
                kalman_q = kalman_q,
                pref_weights = method_prefs,
                hac_lags_for_daily = hac_lags_for_daily,
            )
            
            betas_blend[c] = b_cmb
            
            ses_blend[c] = se_cmb
            
            nobs_min = min(nobs_min, nobs)

        mu_day: Dict[str, float] = {}
       
        if "Index Excess Return" in X_cols and "Index" in factor_preds_df.columns:
       
            idx_ann_total = float(factor_preds_df.at[t, "Index"])
       
            mu_day["Index Excess Return"] = _excess_daily_from_annual_total(
                x_ann_total = idx_ann_total, 
                rf_ann = rf_ann
            )
       
        if "Sector Return" in X_cols and "Sector" in factor_preds_df.columns:
       
            mu_day["Sector Return"] = _daily_from_annual(
                x_ann = float(factor_preds_df.at[t, "Sector"])
            )
       
        if "Industry Return" in X_cols and "Industry" in factor_preds_df.columns:
       
            mu_day["Industry Return"] = _daily_from_annual(
                x_ann = float(factor_preds_df.at[t, "Industry"])
            )
       
        for etf, pred_col in fac_map.items():
       
            if etf in X_cols and pred_col in factor_preds_df.columns:
       
                mu_day[etf] = _daily_from_annual(
                    x_ann = float(factor_preds_df.at[t, pred_col])
                )

        pred_excess_day = alpha + float(np.nansum([betas_blend.get(k, 0.0) * mu_day.get(k, 0.0) for k in X_cols]))
       
        pred_total_ann = (1.0 + rf_day + pred_excess_day) ** td_per_year - 1.0

        if hasattr(hac.params, "index"):

            param_names = list(hac.params.index)

        elif hasattr(ols.params, "index"):
           
            param_names = list(ols.params.index)

        else:

            param_names = list(Xc.columns)

        z = []
        
        for nm in param_names:
      
            if nm == "const":
      
                z.append(1.0)
      
            else:
                
                z.append(float(mu_day.get(nm, 0.0)))
      
        z = np.asarray(z, dtype = float)

        Cov = np.asarray(hac.cov_params(), dtype = float)
        
        if Cov.shape[0] != len(z):  
        
            Cov = np.eye(len(z)) * 0.0
        
        sigma2 = float(ols.mse_resid)

        var_mean_day = float(z @ Cov @ z)
        
        var_obs_day = max(var_mean_day + sigma2, 0.0)
        
        se_day = np.sqrt(var_obs_day)
        
        se_ann = np.sqrt(td_per_year) * se_day  

        row = {
            "Ticker": t,
            "Returns": pred_total_ann,
            "SE": se_ann,
            "r2": r2,
            "nobs_min": int(nobs_min),
            "alpha": alpha,
        }
     
        for c in X_cols:
     
            row[f"beta[{c}]"] = betas_blend.get(c, np.nan)

        rows.append(row)

    out = pd.DataFrame(rows).set_index("Ticker").sort_index()
    
    return out


