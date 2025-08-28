"""
Simple CAPM utility returning volatility and expected return for a stock relative to a market index.
"""

import numpy as np
import pandas as pd


def capm_model(
    beta_stock: float,
    market_volatility: float,
    risk_free_rate: float,
    market_return: float,
    weekly_ret: pd.Series,
    index_weekly_ret: pd.Series
) -> tuple[float, float]:
    """
    Compute CAPM predicted return and back out stock volatility from β and market stats.

    CAPM expected (excess) return:
       
        E[R_i] = r_f + β_i ( E[R_m] − r_f ).

    Link between β, volatility, and correlation:
       
        β_i = Cov(R_i, R_m) / Var(R_m) = ρ_{i,m} (σ_i / σ_m)
    
    ⇒  σ_i = β_i σ_m / ρ_{i,m}.

    This routine:
    
    1) Estimates ρ_{i,m} using the sample correlation of weekly returns
        (`weekly_ret` vs `index_weekly_ret`). If correlation is zero/NaN, it is set to 1
        to avoid division by zero (conservative).
    
    2) Computes σ_i = |β| · σ_m / ρ.
    
    3) Computes CAPM predicted return: 
    
        r_f + β (E[R_m] − r_f).

    Parameters
    ----------
    beta_stock : float
        Asset β relative to the market index.
    market_volatility : float
        σ_m, volatility of the market index in the same periodicity as β (e.g., weekly).
    risk_free_rate : float
        r_f, risk-free rate in the same periodicity (e.g., weekly).
    market_return : float
        E[R_m], expected market return in the same periodicity.
    weekly_ret : pandas.Series
        Asset weekly returns used to compute correlation with the index.
    index_weekly_ret : pandas.Series
        Market index weekly returns.

    Returns
    -------
    (stock_volatility, predicted_return) : tuple[float, float]
        σ_i implied by β and ρ, and CAPM expected return E[R_i].
    """

    corr = weekly_ret.corr(index_weekly_ret)
   
    if corr == 0 or np.isnan(corr):
   
        corr = 1.0
   
    stock_volatility = abs(beta_stock * market_volatility / corr)
   
    predicted_return = risk_free_rate + beta_stock * (market_return - risk_free_rate)
   
    return stock_volatility, predicted_return

