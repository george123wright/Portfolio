"""
Computes Graham number–style valuations using industry averages instead of 22.5, combining P/E and P/B metrics.
"""

import numpy as np
import pandas as pd
from pbv import bvps


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
    
    eps_list  = pd.Series([eps_low, eps_avg, eps_high, low_eps_y, avg_eps_y, high_eps_y]).dropna()
    
    pe_list = pd.Series(
        [pe_ind['Region-Industry'], pe_ind['Industry-MC']]
    ).dropna()
    
    ptb_list = pd.Series(
        [pb_ind['Region-Industry'], pb_ind['Industry-MC']]
    ).dropna()
    
    lb = 0.2 * price
    ub = 5 * price
    
    gn_vals = []
    
    for e in eps_list:
    
        fv = bvps(
            eps = e, 
            prev_bvps = bvps_0, 
            dps = dps
        )
    
        for p in pe_list:
    
            for pb in ptb_list:
    
                val = np.sqrt(p * float(e) * float(pb) * fv)
    
                if not np.isnan(val):
    
                    gn_vals.append(np.clip(val, lb, ub))

    gn = np.array(pd.Series(gn_vals).dropna())
    
    if len(gn)!=0:
    
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
