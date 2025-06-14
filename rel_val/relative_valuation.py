import numpy as np
import pandas as pd
from pbv import bvps

def rel_val_model(low_eps: float, 
                       avg_eps: float, 
                       high_eps: float, 
                       low_eps_y: float,
                       avg_eps_y: float,
                       high_eps_y: float,
                       low_rev: float,
                       avg_rev: float,
                       high_rev: float,
                       low_rev_y: float,
                       avg_rev_y: float,
                       high_rev_y: float,
                       pe_c: float,
                       pe_t: float,
                       pe_ind: float,
                       avg_pe_fs: float,
                       ps: float,
                       avg_ps_fs: float,
                       ind_ps: float,                       
                       ptb: float,
                       avg_ptb_fs: float,
                       ptb_ind: float, 
                       evs: float,
                       avg_fs_ev: float,
                       ind_evs: float, 
                       mc_ev: float,                       
                       bvps_0: float,
                       dps: float,
                       shares_outstanding: int,
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
    
    rev = pd.Series([low_rev, avg_rev, high_rev, low_rev_y, avg_rev_y, high_rev_y]).dropna().tolist()
        
    ptb = pd.Series(
        [ptb, avg_ptb_fs, ptb_ind['Region-Industry'], ptb_ind['Industry-MC']]
    ).dropna()
    
    evts = pd.Series(
        [evs, avg_fs_ev, ind_evs['Region-Industry'],ind_evs['Industry-MC']]
    ).dropna()
    
    pe_list = pd.Series(
        [pe_c, pe_t, avg_pe_fs, pe_ind['Region-Industry'], pe_ind['Industry-MC']]
    ).dropna()
    
    ps_list = pd.Series(
        [avg_ps_fs, ps, ind_ps['Region-Industry'], ind_ps['Industry-MC']]
    ).dropna()
    
    pe_graham_list = pd.Series(
        [pe_ind['Region-Industry'], pe_ind['Industry-MC']]
    ).dropna()
    
    ptb_graham_list = pd.Series(
        [ptb_ind['Region-Industry'], ptb_ind['Industry-MC']]
    ).dropna()
      
    lb = 0.2 * price
    ub = 5 * price
    
    prices = []
    
    for e in eps:
        for p in pe_list:
            price_i = np.clip(e * p, lb, ub)
            if not pd.isna(price_i):
                prices.append(price_i)
            
        book_val = bvps(e, bvps_0, dps)
        for k in ptb:
            p = book_val * k
            if not pd.isna(p):
                prices.append(np.clip(p, lb, ub))
                
        for peg in pe_graham_list:
            for ptbg in ptb_graham_list:
                val = np.sqrt(peg * float(e) * float(ptbg) * book_val)
                if not pd.isna(val):
                    prices.append(np.clip(val, lb, ub))
            
    for r in rev:
        for ps in ps_list:
            price_i = np.clip(r * ps / shares_outstanding, lb, ub)
            if not pd.isna(price_i):
                prices.append(price_i)
        for ev in evts:
            price_i = np.clip(ev * r * mc_ev / shares_outstanding, lb, ub)
            if not pd.isna(price_i):
                prices.append(price_i)
            
    if prices:
        price_low = np.min(prices)
        price_avg = np.mean(prices)
        price_high = np.max(prices)
        
        rets = {i: (price_i / price) - 1 for i, price_i in enumerate(prices)}
        
        rets_avg = np.mean(list(rets.values()))
        rets_std = np.std(list(rets.values()))

        return price_low, price_avg, price_high, rets_avg, rets_std
    
    else:
        return 0, 0, 0, 0, 0, 0
