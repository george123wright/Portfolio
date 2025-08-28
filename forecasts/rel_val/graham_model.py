"""
Computes Graham number–style valuations using industry averages instead of 22.5, combining P/E and P/B metrics.
"""

import numpy as np
import pandas as pd

from pbv import bvps
import config


def graham_number(
    pe_ind,
    eps_low, 
    eps_avg, 
    eps_high,
    price,
    pb_ind,
    bvps_0, 
    dps,
    low_eps_y, 
    avg_eps_y, 
    high_eps_y
):
    """
    Generalised Graham valuation combining earnings and book via industry multiples.
    
    The origional Graham number is defined as:
    
        Graham(E; PE, PB) = sqrt(22.5 · EPS · BVPS )
    
    where 22.5 = 15 · 1.5 is the product of Graham's preferred P/E and P/B ratios.
    
    I have adapted this to use industry averages isntead as I believe these valuations
    are outdated in todays markets.
        
    For each EPS scenario E and for each pair of industry multiples (PE_ind, PB_ind),
    form a Graham-style value using the geometric mean of earnings and book drivers:
      
        BVPS_next = BVPS_prev + EPS − DPS
      
        Graham(E; PE, PB) = sqrt( PE · E · PB · BVPS_next )

    We aggregate across:
   
    • EPS scenarios: {eps_low, eps_avg, eps_high, low_eps_y, avg_eps_y, high_eps_y}.
   
    • PE_ind ∈ { pe_ind['Region-Industry'], pe_ind['Industry-MC'] }.
   
    • PB_ind ∈ { pb_ind['Region-Industry'], pb_ind['Industry-MC'] }.

    Each Graham value is clipped to [0.2·Price_current, 5·Price_current] to guard outliers.
    We return min/mean/max and the mean/std of the relative returns to current price.

    Parameters
    ----------
    pe_ind : dict-like
        Industry P/E multipliers with keys 'Region-Industry' and 'Industry-MC'.
    eps_low, eps_avg, eps_high : float
        Current-period EPS scenarios.
    price : float
        Current stock price used for clipping and relative returns.
    pb_ind : dict-like
        Industry P/B multipliers with keys 'Region-Industry' and 'Industry-MC'.
    bvps_0 : float
        Starting book value per share (BVPS_prev).
    dps : float
        Dividends per share.
    low_eps_y, avg_eps_y, high_eps_y : float
        Next-period EPS scenarios.

    Returns
    -------
    (min_price, mean_price, max_price, mean_rel_ret, std_rel_ret) : tuple
        Summary of Graham valuations across all combinations.

    Notes
    -----
    • BVPS update equation: BVPS_next = BVPS_prev + EPS − DPS.
    • Any NaNs from inputs are ignored; if no values remain, zeros are returned.
    """    
    
    eps_list  = pd.Series([eps_low, eps_avg, eps_high, low_eps_y, avg_eps_y, high_eps_y]).dropna()
    
    pe_list = pd.Series(
        [pe_ind['Region-Industry'], pe_ind['Industry-MC']]
    ).dropna()
    
    ptb_list = pd.Series(
        [pb_ind['Region-Industry'], pb_ind['Industry-MC']]
    ).dropna()
    
    
    lb = config.lbp * price
    
    ub = config.ubp * price
    
    gn_vals = []
    
    for e in eps_list:
    
        fv = bvps(
            eps = e, 
            prev_bvps = bvps_0, 
            dps = dps
        )
    
        for p in pe_list:
    
            for pb in ptb_list:
                
                e = max(e, 0)
    
                val = np.sqrt(p * float(e) * float(pb) * fv)
    
                if not np.isnan(val):
    
                    gn_vals.append(np.clip(val, lb, ub))

    gn = np.array(pd.Series(gn_vals).dropna())
    
    if len(gn) != 0:
    
        rel_ret = (gn / price) - 1

        return (
            np.min(gn),           
            np.mean(gn),           
            np.max(gn),           
            np.mean(rel_ret),      
            np.std(rel_ret)   
        )
    
    else:
        
        return 0, 0, 0, 0, 0
