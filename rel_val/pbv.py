"""
Calculates price targets using price‑to‑book ratios and book‑value growth.
"""

import numpy as np
import pandas as pd

def bvps(eps, prev_bvps, dps):
    """
    Calculate BVPS for the next period:
    BVPSₙ = BVPSₙ₋₁ + EPSₙ - DPSₙ.
    """
    return max(float(prev_bvps) + float(eps) - float(dps), 0)

def price_to_book_pred(low_eps: float, 
                       avg_eps: float, 
                       high_eps: float, 
                       low_eps_y: float,
                       avg_eps_y: float,
                       high_eps_y: float,
                       ptb: float,
                       avg_ptb_fs: float,
                       ptb_ind: float, 
                       book_fs: float,
                       dps: float,
                       price: float) -> tuple[float, float, float, float, float]:
    
    low_eps = float(low_eps)
    avg_eps = float(avg_eps)
    high_eps = float(high_eps)
    low_eps_y = float(low_eps_y)
    avg_eps_y = float(avg_eps_y)
    high_eps_y = float(high_eps_y)
    
    if pd.isna(dps):
        dps = 0
    
    eps = pd.Series([low_eps, avg_eps, high_eps, low_eps_y, avg_eps_y, high_eps_y]).dropna().tolist()
        
    ptb = pd.Series(
        [ptb, avg_ptb_fs, ptb_ind['Region-Industry'], ptb_ind['Industry-MC']]
    ).dropna()
        
    book = pd.Series([book_fs]).dropna().tolist()
    
    prices = []
    lb = 0.2 * price
    ub = 5 * price
    for e in eps:
        for b in book:
            book_val = bvps(e, b, dps)
            for k in ptb:
                p = book_val * k
                if not pd.isna(p):
                    prices.append(np.clip(p, lb, ub))

    avg_ptb = np.mean(ptb)
    
    if prices:
        price_low = np.min(prices)
        price_avg = np.mean(prices)
        price_high = np.max(prices)
        
        rets = {i: (price_i / price) - 1 for i, price_i in enumerate(prices)}
        
        rets_avg = np.mean(list(rets.values()))
        rets_std = np.std(list(rets.values()))

        return price_low, price_avg, price_high, rets_avg, rets_std, avg_ptb
    
    else:
        return 0, 0, 0, 0, 0, 0
