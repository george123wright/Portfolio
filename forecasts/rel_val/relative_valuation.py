"""
Consolidated relative‑valuation model blending P/E, P/S, P/BV and EV/Sales signals.
"""

import numpy as np
import pandas as pd

from pbv import bvps
import config

def rel_val_model(
    low_eps: float, 
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
    price: float,
    discount_rate: float
) -> tuple[float, float, float, float, float]:
    """
    Composite relative valuation blending P/E, P/B (forward BVPS), Graham, P/S, and EV/Sales.
    
    All prices are discounted by the discount rate.

    We generate a panel of implied prices from the following models:

    1) **P/E** (earnings multiple):
    
        Price_PE(E, P) = E · P,
    
    with E over EPS scenarios
    
        {low_eps, avg_eps, high_eps, low_eps_y, avg_eps_y, high_eps_y},
    
    and P over multiples
    
        {pe_c, pe_t, avg_pe_fs, pe_ind['Region-Industry'], pe_ind['Industry-MC']}.

    2) **P/B** (book multiple with forward BVPS):
    
        BVPS_next = max( bvps_0 + E − dps, 0 ),
    
        Price_PB(E, K) = BVPS_next · K,
    
    with K over
    
        {ptb, avg_ptb_fs, ptb_ind['Region-Industry'], ptb_ind['Industry-MC']}.

    3) **Graham** (geometric blend of earnings and book via industry multiples):
    
        Price_Graham(E; PE_g, PB_g) = sqrt( PE_g · E · PB_g · BVPS_next ),
    
    with PE_g over {pe_ind[...]}, PB_g over {ptb_ind[...]}.

    4) **P/S** (sales multiple):
    
        Price_PS(R, S) = (R · S) / Shares,
    
    with R over revenue scenarios
    
        {low_rev, avg_rev, high_rev, low_rev_y, avg_rev_y, high_rev_y},
    
    and S over {avg_ps_fs, ps, ind_ps['Region-Industry'], ind_ps['Industry-MC']}.

    5) **EV/Sales** (enterprise value multiple to equity value per share):
    
        Price_EV(R, M) = (M · R · MC_EV) / Shares,
    
    with M over {evs, avg_fs_ev, ind_evs['Region-Industry'], ind_evs['Industry-MC']}.

    For robustness, each implied price is clipped to [0.2·Price_current, 5·Price_current].
    
    Pool all valid prices from the above to produce the final summary:

        price_low  = min pooled prices
        price_avg  = mean pooled prices
        price_high = max pooled prices
        rets_avg   = mean_i( Price_i / Price_current − 1 )
        rets_std   = std_i( Price_i / Price_current − 1 ) / sqrt(5)

    (The divisor sqrt(5) down-weights dispersion because five model families are blended.)

    Parameters
    ----------
    low_eps, avg_eps, high_eps, low_eps_y, avg_eps_y, high_eps_y : float
        EPS scenarios (current and next period).
    low_rev, avg_rev, high_rev, low_rev_y, avg_rev_y, high_rev_y : float
        Revenue scenarios (current and next period).
    pe_c, pe_t, avg_pe_fs : float
        Company and peer P/E multiples.
    pe_ind : dict-like
        Industry P/E with keys 'Region-Industry' and 'Industry-MC'].
    ps, avg_ps_fs : float
        Company and peer P/S multiples.
    ind_ps : dict-like
        Industry P/S with keys 'Region-Industry' and 'Industry-MC'].
    ptb, avg_ptb_fs : float
        Company and peer P/B multiples.
    ptb_ind : dict-like
        Industry P/B with keys 'Region-Industry' and 'Industry-MC'].
    evs, avg_fs_ev : float
        Company and peer EV/Sales multiples.
    ind_evs : dict-like
        Industry EV/Sales with keys 'Region-Industry' and 'Industry-MC'].
    mc_ev : float
        EV alignment multiplier (e.g., to translate EV to equity basis).
    bvps_0 : float
        Starting book value per share for the BVPS roll-forward.
    dps : float
        Dividends per share (NaN treated as 0).
    shares_outstanding : int
        Shares denominator for per-share prices.
    price : float
        Current price used for clipping and relative return computation.

    Returns
    -------
    (price_low, price_avg, price_high, rets_avg, rets_std) : tuple[float, float, float, float, float]
        Summary statistics of the pooled relative-valuation prices. Returns zeros if no
        valid prices are generated.

    Notes
    -----
    • All EPS and revenue inputs are cast to float and NaNs dropped per scenario list.
    
    • The Graham component uses the same BVPS_next as the P/B component.
        
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
      
    lb = config.lbp * price
    
    ub = config.ubp * price
    
    prices = []
    
    for e in eps:
     
        for p in pe_list:
     
            price_i = np.clip(e * p, lb, ub)
     
            if not pd.isna(price_i):
     
                prices.append(price_i)
            
        book_val = bvps(
            eps = e, 
            prev_bvps = bvps_0, 
            dps = dps
        )
     
        for k in ptb:
     
            p = book_val * k
     
            if not pd.isna(p):
     
                prices.append(np.clip(p, lb, ub))
                
        for peg in pe_graham_list:
     
            for ptbg in ptb_graham_list:
                
                e = max(e, 0.0)
     
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
        
        prices = prices / (1 + discount_rate)
     
        price_low = np.min(prices)
     
        price_avg = np.mean(prices)
     
        price_high = np.max(prices)
        
        rets = {
            i: (price_i / price) - 1 for i, price_i in enumerate(prices)
        }
        
        rets_avg = np.mean(list(rets.values()))
     
        rets_std = np.std(list(rets.values())) / np.sqrt(5)

        return price_low, price_avg, price_high, rets_avg, rets_std
    
    else:
       
        return 0, 0, 0, 0, 0, 0
