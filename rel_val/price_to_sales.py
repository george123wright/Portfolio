"""
Price‑to‑sales valuation for projected revenue scenarios.
"""

import numpy as np
import pandas as pd


def price_to_sales_price_pred(
    price: float,
    low_rev_y: float,
    avg_rev_y: float,
    high_rev_y: float,
    low_rev: float,
    avg_rev: float,
    high_rev: float,
    shares_outstanding: int,
    ps: float,
    avg_ps_fs: float,
    ind_ps: float
) -> tuple[float, float, float, float, float]:
    """
    Compute predicted low, average, and high prices as well as return error
    based on the price-to-sales ratio.
    """

    price = float(price)
   
    low_rev = float(low_rev)
    avg_rev = float(avg_rev)
    high_rev = float(high_rev)
   
    low_rev_y = float(low_rev_y)
    avg_rev_y = float(avg_rev_y)
    high_rev_y = float(high_rev_y)
   
    shares_outstanding = float(shares_outstanding)
    
    if avg_rev == 0:
   
        return 0, 0, 0, 0, 0, 0
   
    lb = 0.2 * price
    ub = 5 * price
   
    rev_list  = pd.Series([low_rev, avg_rev, high_rev, low_rev_y, avg_rev_y, high_rev_y]).dropna()
    
    ps_list = pd.Series(
        [avg_ps_fs, ps, ind_ps['Region-Industry'], ind_ps['Industry-MC']]
    ).dropna()
    
    avg_ps = np.mean(ps_list)
    
    prices = []
    
    for r in rev_list:
      
        for ps in ps_list:
      
            prices.append(np.clip(r * ps / shares_outstanding, lb,ub))
    
    price_low = min(prices)
    price_avg = np.mean(prices)
    price_high = max(prices)
    
    returns = {i: (price_i / price) - 1 for i, price_i in enumerate(prices)}
    
    returns_avg = np.mean(list(returns.values()))
    
    returns_std = np.std(list(returns.values()))
    
    return price_low, price_avg, price_high, returns_avg, returns_std, avg_ps
