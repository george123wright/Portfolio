"""
Price‑to‑earnings valuation producing price range and return statistics.
"""

import numpy as np
import pandas as pd

import config


def pe_price_pred(
    eps_low: float,
    eps_avg: float,
    eps_high: float,
    eps_low_y: float,
    eps_avg_y: float,
    eps_high_y: float,
    pe_c: float,
    pe_t: float,
    pe_ind: float,
    avg_pe_fs: float,
    price: float
) -> tuple[float, float, float, float, float]:
    """
    Price range from P/E methodology across EPS and multiple scenarios.

    For each EPS scenario E ∈ {eps_low, eps_avg, eps_high, eps_low_y, eps_avg_y, eps_high_y}
    and P/E multiple P ∈ {pe_c, pe_t, avg_pe_fs, pe_ind['Region-Industry'], pe_ind['Industry-MC']},
    compute
   
        Price(E, P) = E · P,
   
    Aggregate across all combos.

    Outputs include the average multiple used and relative return distribution to
    current price P_0.

    Parameters
    ----------
    eps_low, eps_avg, eps_high : float
        Current-period EPS scenarios (floored at 0).
    eps_low_y, eps_avg_y, eps_high_y : float
        Next-period EPS scenarios (floored at 0).
    pe_c : float
        Company current P/E.
    pe_t : float
        Company trailing / target P/E (as provided).
    pe_ind : dict-like
        Industry P/E with keys 'Region-Industry' and 'Industry-MC'].
    avg_pe_fs : float
        Average P/E from financial-statement comps / peer set.
    price : float
        Current stock price P_0.

    Returns
    -------
    (price_min, price_avg, price_max, rets_avg, rets_std, avg_pe) : tuple[float,...]
        Range, average price, average/std of (Price/P_0 − 1), and mean multiple used.
    """

    prices = []
  
    eps_low = max(eps_low, 0)
  
    eps_avg = max(eps_avg, 0)
  
    eps_high = max(eps_high, 0)
  
    eps_low_y = max(eps_low_y, 0)
  
    eps_avg_y = max(eps_avg_y, 0)
  
    eps_high_y = max(eps_high_y, 0)
  
    lb = config.lbp * price
    
    ub = config.ubp * price
  
    eps_list  = pd.Series([eps_low, eps_avg, eps_high, eps_low_y, eps_avg_y, eps_high_y]).dropna()
    
    pe_list = pd.Series(
        [pe_c, pe_t, avg_pe_fs, pe_ind['Region-Industry'], pe_ind['Industry-MC']]
    ).dropna()
    
    avg_pe = np.mean(pe_list)
    
    for e in eps_list:
  
        for p in pe_list:
            
            e = max(e, 0)
  
            price_i = np.clip(e * p, lb, ub)
  
            prices.append(max(price_i, 0))
            
    price_avg = np.mean(prices)
    
    price_min = np.min(prices)
    
    price_max = np.max(prices)
    
    rets = {
        i: (price_i / price) - 1 for i, price_i in enumerate(prices)
    }
  
    rets_avg = np.mean(list(rets.values()))
  
    rets_std = np.std(list(rets.values()))
    
    return price_min, price_avg, price_max, rets_avg, rets_std, avg_pe


    

