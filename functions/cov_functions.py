"""
Provides covariance estimators
- sample
- constant‑correlation
- shrinkage versions
- predicted
"""


import numpy as np
import pandas as pd
import config
from sklearn.covariance import LedoitWolf


def multi_horizon_scale_correlation(
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    daily_3y: pd.DataFrame,
    weekly_3y: pd.DataFrame,
    monthly_3y: pd.DataFrame,
    daily_1y: pd.DataFrame,
    weekly_1y: pd.DataFrame,
    monthly_1y: pd.DataFrame,
    w_freq: tuple[float,float,float] = (0.2, 0.6, 0.2),
    h_horizon: tuple[float,float,float] = (0.5, 0.3, 0.2)
) -> pd.DataFrame:
    """
    Blend correlations across frequencies (daily/weekly/monthly)
    and horizons (5y/3y/1y), then return a combined correlation matrix.
    """

    w_d, w_w, w_m = w_freq

    h_5, h_3, h_1 = h_horizon

    def blend_freq(
        daily, 
        weekly, 
        monthly
    ):
       
        c_d = daily.corr().values
       
        c_w = weekly.corr().values
       
        c_m = monthly.corr().values
       
        return w_d * c_d + w_w * c_w + w_m * c_m

    c5 = blend_freq(
        daily = daily_5y, 
        weekly = weekly_5y, 
        monthly = monthly_5y
    )
    
    c3 = blend_freq(
        daily = daily_3y, 
        weekly = weekly_3y, 
        monthly = monthly_3y
    )
    
    c1 = blend_freq(
        daily = daily_1y, 
        weekly = weekly_1y, 
        monthly = monthly_1y
    )

    comb = h_5 * c5 + h_3 * c3 + h_1 * c1

    idx = daily_5y.columns

    corr_multi = pd.DataFrame(comb, index=idx, columns=idx)

    return corr_multi


def _shrink_correlation(
    corr: pd.DataFrame,
    n_eff: int
) -> pd.DataFrame:
    """
    Analytic shrinkage of off-diagonal entries toward zero (identity)
    per Schäfer & Strimmer (2005).
    """
   
    R = corr.values.copy()
   
    p = R.shape[0]
   
    mask = ~np.eye(p, dtype=bool)

    r = R[mask]

    var_r = (1 - r**2)**2 / max(n_eff - 1, 1)

    num = var_r.sum()
   
    den = (r ** 2).sum()
   
    if den <= 0:
   
        lam = 0.0
   
    else:
       
        lam = min(max(num/den, 0.0), 1.0)

    R_shr = R.copy()

    R_shr[mask] = (1 - lam) * R[mask]
   
    np.fill_diagonal(R_shr, 1.0)
    
    return pd.DataFrame(R_shr, index=corr.index, columns=corr.columns)


def sample_covariance(
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    daily_3y: pd.DataFrame,
    weekly_3y: pd.DataFrame,
    monthly_3y: pd.DataFrame,
    daily_1y: pd.DataFrame,
    weekly_1y: pd.DataFrame,
    monthly_1y: pd.DataFrame,
    w_freq: tuple[float,float,float] = (0.2, 0.6, 0.2),
    h_horizon: tuple[float,float,float] = (0.5, 0.3, 0.2)
) -> pd.DataFrame:
    """
    Multi-horizon, multi-scale sample covariance in *weekly* units.
    """
    
    w_d, w_w, w_m = w_freq
    
    h_5, h_3, h_1 = h_horizon

    def cov_for_horizon(
        daily, 
        weekly, 
        monthly
    ):

        cov_d = daily.cov(ddof = 0) * 252
       
        cov_w = weekly.cov(ddof = 0) * 52
       
        cov_m = monthly.cov(ddof = 0) * 12

        cov_ann = w_d * cov_d.values + w_w * cov_w.values + w_m * cov_m.values
        
        return cov_ann / 52

    cov5 = cov_for_horizon(
        daily = daily_5y, 
        weekly = weekly_5y, 
        monthly = monthly_5y
    )
    
    cov3 = cov_for_horizon(
        daily = daily_3y, 
        weekly = weekly_3y, 
        monthly = monthly_3y
    )
    
    cov1 = cov_for_horizon(
        daily = daily_1y, 
        weekly = weekly_1y, 
        monthly = monthly_1y
    )

    cov_wk = h_5 * cov5 + h_3 * cov3 + h_1 * cov1
    
    idx = daily_5y.columns
    
    return pd.DataFrame(cov_wk, 
                        index = idx, 
                        columns = idx)


def cc_covariance(
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    daily_3y: pd.DataFrame,
    weekly_3y: pd.DataFrame,
    monthly_3y: pd.DataFrame,
    daily_1y: pd.DataFrame,
    weekly_1y: pd.DataFrame,
    monthly_1y: pd.DataFrame,
    w_freq: tuple[float,float,float] = (0.2, 0.6, 0.2),
    h_horizon: tuple[float,float,float] = (0.5, 0.3, 0.2)
) -> pd.DataFrame:
    """
    Constant-correlation prior using multi-horizon, multi-scale correlation,
    placed on the weekly-volatility diagonal.
    """

    corr_ms = multi_horizon_scale_correlation(
        daily_5y = daily_5y, 
        weekly_5y = weekly_5y, 
        monthly_5y = monthly_5y,
        daily_3y = daily_3y, 
        weekly_3y = weekly_3y, 
        monthly_3y = monthly_3y,
        daily_1y = daily_1y, 
        weekly_1y = weekly_1y, 
        monthly_1y = monthly_1y,
        w_freq = w_freq, 
        h_horizon = h_horizon
    )

    n_eff = (
        h_horizon[0] * len(daily_5y) +
        h_horizon[1] * len(daily_3y) +
        h_horizon[2] * len(daily_1y)
    )

    corr_shr = _shrink_correlation(
        corr = corr_ms, 
        n_eff = int(n_eff)
    )

    S = sample_covariance(
        daily_5y = daily_5y, 
        weekly_5y = weekly_5y, 
        monthly_5y = monthly_5y,
        daily_3y = daily_3y, 
        weekly_3y = weekly_3y, 
        monthly_3y = monthly_3y,
        daily_1y = daily_1y, 
        weekly_1y = weekly_1y, 
        monthly_1y = monthly_1y,
        w_freq = w_freq, 
        h_horizon = h_horizon
    )
    std_wk = np.sqrt(np.diag(S))

    cov = corr_shr.values * np.outer(std_wk, std_wk)
   
    return pd.DataFrame(cov, index=S.index, columns=S.columns)


def pred_covariance(
    comb_std: pd.Series,
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    daily_3y: pd.DataFrame,
    weekly_3y: pd.DataFrame,
    monthly_3y: pd.DataFrame,
    daily_1y: pd.DataFrame,
    weekly_1y: pd.DataFrame,
    monthly_1y: pd.DataFrame,
    w_freq: tuple[float,float,float] = (0.2, 0.6, 0.2),
    h_horizon: tuple[float,float,float] = (0.5, 0.3, 0.2),
    periods_per_year: int = 52
) -> pd.DataFrame:
    """
    Forecast-error covariance (weekly units) using multi-horizon,
    multi-scale shrunk correlation and annual SEs.
    """
   
    corr_ms = multi_horizon_scale_correlation(
        daily_5y = daily_5y, 
        weekly_5y = weekly_5y, 
        monthly_5y = monthly_5y,
        daily_3y = daily_3y, 
        weekly_3y = weekly_3y, 
        monthly_3y = monthly_3y,
        daily_1y = daily_1y, 
        weekly_1y = weekly_1y, 
        monthly_1y = monthly_1y,
        w_freq = w_freq, 
        h_horizon = h_horizon
    )
  
    n_eff = (
        h_horizon[0] * len(daily_5y) +
        h_horizon[1] * len(daily_3y) +
        h_horizon[2] * len(daily_1y)
    )
   
    corr_shr = _shrink_correlation(
        corr = corr_ms, 
        n_eff = int(n_eff)
    )

    se_wk = comb_std / np.sqrt(periods_per_year)

    cov = np.outer(se_wk, se_wk) * corr_shr.values

    idx = comb_std.index

    return pd.DataFrame(cov, 
                        index = idx, 
                        columns = idx)


def shrinkage_covariance(
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    comb_std: pd.Series,
    common_idx: list[str],
    w_freq: tuple[float,float,float] = (0.2, 0.6, 0.2),
    h_horizon: tuple[float,float,float] = (0.5, 0.3, 0.2),
    delta: float = 1/3,
    alpha: float = 1/3
) -> pd.DataFrame:
    """
    Shrink between:
      - sample covariance
      - constant-correlation prior
      - forecast-error covariance
    in weekly units, then annualize once.
    """
    
    daily_5y = daily_5y.loc[:, common_idx]
    weekly_5y = weekly_5y.loc[:, common_idx]
    monthly_5y = monthly_5y.loc[:, common_idx]
    
    daily_3y = daily_5y.loc[
        daily_5y.index >= pd.to_datetime(config.THREE_YEAR_AGO), common_idx
    ]
    
    weekly_3y = weekly_5y.loc[
        weekly_5y.index >= pd.to_datetime(config.THREE_YEAR_AGO), common_idx
    ]
    
    monthly_3y = monthly_5y.loc[
        monthly_5y.index >= pd.to_datetime(config.THREE_YEAR_AGO), common_idx
    ]
    
    daily_1y = daily_5y.loc[
        daily_5y.index >= pd.to_datetime(config.YEAR_AGO), common_idx
    ]
    
    weekly_1y = weekly_5y.loc[
        weekly_5y.index >= pd.to_datetime(config.YEAR_AGO), common_idx
    ]
    
    monthly_1y = monthly_5y.loc[
        monthly_5y.index >= pd.to_datetime(config.YEAR_AGO), common_idx
    ]
    
    S = sample_covariance(
        daily_5y = daily_5y, 
        weekly_5y = weekly_5y, 
        monthly_5y = monthly_5y,
        daily_3y = daily_3y, 
        weekly_3y = weekly_3y, 
        monthly_3y = monthly_3y,
        daily_1y = daily_1y, 
        weekly_1y = weekly_1y, 
        monthly_1y = monthly_1y,
        w_freq = w_freq, 
        h_horizon = h_horizon
    ).values
   
    P = cc_covariance(
        daily_5y = daily_5y, 
        weekly_5y = weekly_5y, 
        monthly_5y = monthly_5y,
        daily_3y = daily_3y, 
        weekly_3y = weekly_3y, 
        monthly_3y = monthly_3y,
        daily_1y = daily_1y, 
        weekly_1y = weekly_1y, 
        monthly_1y = monthly_1y,
        w_freq = w_freq, 
        h_horizon = h_horizon
    ).values
   
    F = pred_covariance(
        comb_std = comb_std,
        daily_5y = daily_5y, 
        weekly_5y = weekly_5y, 
        monthly_5y = monthly_5y,
        daily_3y = daily_3y, 
        weekly_3y = weekly_3y, 
        monthly_3y = monthly_3y,
        daily_1y = daily_1y, 
        weekly_1y = weekly_1y, 
        monthly_1y = monthly_1y,
        w_freq = w_freq, 
        h_horizon = h_horizon
    ).values

    C_wk = delta * P + (1 - delta - alpha) * S + alpha * F
   
    C_ann = C_wk * 52
   
    idx = daily_5y.columns
   
    return pd.DataFrame(C_ann, index=idx, columns=idx)


def optimal_ledoit_wolf(
    returns_weekly: pd.DataFrame,
    periods_per_year: int = 52
) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage on weekly returns, annualized.
    """
    
    lw = LedoitWolf().fit(returns_weekly.values)
    
    cov_wk = lw.covariance_
    
    cov_ann = cov_wk * periods_per_year
    
    return pd.DataFrame(cov_ann,
                        index = returns_weekly.columns,
                        columns = returns_weekly.columns)
