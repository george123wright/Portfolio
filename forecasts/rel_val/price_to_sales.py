"""
Price‑to‑sales valuation for projected revenue scenarios.
"""

import numpy as np
import pandas as pd

import config


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
    Price range from Price/Sales (P/S) across revenue and multiple scenarios.

    For each revenue scenario R and P/S multiple S ∈ {avg_ps_fs, ps, ind_ps['Region-Industry'], ind_ps['Industry-MC']},
    implied price per share is
  
        Price(R, S) = (R · S) / Shares.
  
    Clip and aggregate across combos.

    Outputs include price range, mean/std of relative returns to current price P_0,
    and the average multiple used.

    Parameters
    ----------
    price : float
        Current stock price P_0.
    low_rev_y, avg_rev_y, high_rev_y : float
        Next-period revenue scenarios.
    low_rev, avg_rev, high_rev : float
        Current-period revenue scenarios.
    shares_outstanding : int
        Shares denominator for per-share price.
    ps : float
        Company P/S multiple.
    avg_ps_fs : float
        Average P/S from financial statement comps / peer set.
    ind_ps : dict-like
        Industry P/S with keys 'Region-Industry' and 'Industry-MC'.

    Returns
    -------
    (price_low, price_avg, price_high, returns_avg, returns_std, avg_ps) : tuple[float,...]
        Range and return stats; avg_ps is the mean P/S multiple used.

    Notes
    -----
    • If `avg_rev` equals 0 (no scale), the function returns zeros by design guard.
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
   
    lb = config.lbp * price
    
    ub = config.ubp * price
   
    rev_list  = pd.Series([low_rev, avg_rev, high_rev, low_rev_y, avg_rev_y, high_rev_y]).dropna()
    
    ps_list = pd.Series(
        [avg_ps_fs, ps, ind_ps['Region-Industry'], ind_ps['Industry-MC']]
    ).dropna()
    
    avg_ps = np.mean(ps_list)
    
    prices = []
    
    for r in rev_list:
      
        for ps in ps_list:
      
            prices.append(np.clip(r * ps / shares_outstanding, lb, ub))
    
    price_low = min(prices)
   
    price_avg = np.mean(prices)
   
    price_high = max(prices)
    
    returns = {
        i: (price_i / price) - 1 for i, price_i in enumerate(prices)
    }
    
    returns_avg = np.mean(list(returns.values()))
    
    returns_std = np.std(list(returns.values()))
    
    return price_low, price_avg, price_high, returns_avg, returns_std, avg_ps
