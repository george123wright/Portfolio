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
    Calculate the stock’s volatility and predicted return using CAPM.
    """
   
    corr = weekly_ret.corr(index_weekly_ret)
   
    if corr == 0 or np.isnan(corr):
   
        corr = 1.0
   
    stock_volatility = abs(beta_stock * market_volatility / corr)
   
    predicted_return = risk_free_rate + beta_stock * (market_return - risk_free_rate)
   
    return stock_volatility, predicted_return
