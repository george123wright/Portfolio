"""
EV‑to‑sales valuation adjusted for market cap / enterprise value multiples
"""

import numpy as np
import pandas as pd

def ev_to_sales_price_pred(price: float,
                           low_rev: float,
                           avg_rev: float,
                           high_rev: float,
                           low_rev_y: float,
                           avg_rev_y: float,
                           high_rev_y: float,
                           shares_outstanding: int,
                           evs: float,
                           avg_fs_ev: float,
                           ind_evs: float, 
                           mc_ev: float) -> tuple[float, float, float, float, float]:
    """
    Compute predicted prices based on EV-to-sales metrics.
    """
    price = float(price)
    low_rev = max(float(low_rev), 0)
    avg_rev = max(float(avg_rev), 0)
    high_rev = max(float(high_rev), 0)
    low_rev_y = max(float(low_rev_y), 0)
    avg_rev_y = max(float(avg_rev_y), 0)
    high_rev_y = max(float(high_rev_y), 0)
    shares_outstanding = float(shares_outstanding)
    
    evts = pd.Series(
        [evs, avg_fs_ev, ind_evs['Region-Industry'],ind_evs['Industry-MC']]
    ).dropna()
    ub = 5 * price
    lb = 0.2 * price
    
    rev = pd.Series([low_rev, avg_rev, high_rev, low_rev_y, avg_rev_y, high_rev_y]).dropna().tolist()
    
    avg_evts = np.mean(evts)
    
    prices = []
    for e in evts:
        for r in rev:
            p = e * r / shares_outstanding
            if not pd.isna(p):
                prices.append(np.clip(p * mc_ev, lb, ub))
            
    prices = pd.Series(prices).dropna().tolist()
    
    if prices:
        price_low = np.min(prices)
        price_avg = np.mean(prices)
        price_high = np.max(prices)
        
        rets = {i: (price_i / price) - 1 for i, price_i in enumerate(prices)}
        
        rets_avg = np.mean(list(rets.values()))
        rets_std = np.std(list(rets.values()))

        return price_low, price_avg, price_high, rets_avg, rets_std, avg_evts
    
    else:
        return 0, 0, 0, 0, 0, 0



