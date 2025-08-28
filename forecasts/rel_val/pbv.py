"""
Calculates price targets using price‑to‑book ratios and book‑value growth.
"""

import numpy as np
import pandas as pd

import config


def bvps(
    eps, 
    prev_bvps, 
    dps
):
    """
    One-step book value per share (BVPS) roll-forward.

    Accounting identity:
        BVPS_next = BVPS_prev + EPS − DPS.

    All inputs are interpreted per-share; the result is floored at 0.

    Parameters
    ----------
    eps : float
        Earnings per share for the period.
    prev_bvps : float
        Starting book value per share.
    dps : float
        Dividends per share.

    Returns
    -------
    float
        Next period BVPS_next ≥ 0.
    """
    
    return max(float(prev_bvps) + float(eps) - float(dps), 0)


def price_to_book_pred(
    low_eps: float, 
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
    price: float
) -> tuple[float, float, float, float, float]:
    """
    Price range from P/B methodology with forward BVPS across EPS and multiple scenarios.

    For each EPS scenario E and base book B, compute forward book value:
    
        BVPS_next = max( B + E − DPS, 0 ),
    
    then for each P/B multiple K ∈ {ptb, avg_ptb_fs, ptb_ind['Region-Industry'], ptb_ind['Industry-MC']},
    compute
    
        Price(E, B, K) = BVPS_next · K,
    
    clip, and aggregate across combinations.

    Return price range and the distribution of returns relative to current price,
    as well as the average P/B multiple used.

    Parameters
    ----------
    low_eps, avg_eps, high_eps : float
        Current EPS scenarios.
    low_eps_y, avg_eps_y, high_eps_y : float
        Next-period EPS scenarios.
    ptb : float
        Company P/B multiple.
    avg_ptb_fs : float
        Average P/B from financial statement comps / peer set.
    ptb_ind : dict-like
        Industry P/B with keys 'Region-Industry' and 'Industry-MC'.
    book_fs : float
        Base book value per share B used in the roll-forward.
    dps : float
        Dividends per share; NaN is treated as 0.
    price : float
        Current price used for clipping and returns.

    Returns
    -------
    (price_low, price_avg, price_high, rets_avg, rets_std, avg_ptb) : tuple[float,...]
        Range, average price, return stats, and mean multiple used; zeros if no prices.
    """
    
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
    
    lb = config.lbp * price
   
    ub = config.ubp * price
    
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
        
        rets = {
            i: (price_i / price) - 1 for i, price_i in enumerate(prices)
        }
        
        rets_avg = np.mean(list(rets.values()))
    
        rets_std = np.std(list(rets.values()))

        return price_low, price_avg, price_high, rets_avg, rets_std, avg_ptb
    
    else:
    
        return 0, 0, 0, 0, 0, 0
