"""
EV‑to‑sales valuation adjusted for market cap / enterprise value multiples
"""

import numpy as np
import pandas as pd
import config

def ev_to_sales_price_pred(
    price: float,
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
    mc_ev: float
) -> tuple[float, float, float, float, float]:
    """
    Price prediction from EV/Sales (EV/Revenue) multiples across revenue scenarios.

    For each EV/Sales multiple m in the set M and revenue level R in the scenario set S,
    the implied equity value per share is:
   
        Price(m, R) = m · R · MC_EV / Shares,
   
    where
   
    • m ∈ {evs, avg_fs_ev, ind_evs['Region-Industry'], ind_evs['Industry-MC']},
   
    • R ∈ { low_rev, avg_rev, high_rev, low_rev_y, avg_rev_y, high_rev_y } (all ≥ 0),
   
    • Shares is `shares_outstanding`,
   
    • MC_EV is an optional capitalisation/matching coefficient (`mc_ev`) applied to EV
        to align metrics to market conventions (set to 1 if not needed).

    We clip each price to the band [0.2·Price_current, 5·Price_current] for robustness.
    We then report summary statistics and distribution properties relative to the
    current price P_0:

    low  = min_i Price_i
    avg  = mean_i Price_i
    high = max_i Price_i
    rets_avg = mean_i( Price_i / P_0 − 1 )
    rets_std = std_i( Price_i / P_0 − 1 )
    avg_evts = mean of the EV/Sales multiples used

    Parameters
    ----------
    price : float
        Current price P_0 used for clipping and return computation.
    low_rev, avg_rev, high_rev : float
        Revenue scenarios for the current period (≥ 0).
    low_rev_y, avg_rev_y, high_rev_y : float
        Revenue scenarios for the next period (≥ 0).
    shares_outstanding : int
        Share count denominator.
    evs : float
        Company EV/Sales multiple.
    avg_fs_ev : float
        Average EV/Sales from financial statement comps / peer set.
    ind_evs : pandas.Series or dict-like
        Industry EV/Sales dictionary with keys 'Region-Industry' and 'Industry-MC'.
    mc_ev : float
        Multiplicative alignment factor applied to EV before translating to equity.

    Returns
    -------
    (low, avg, high, rets_avg, rets_std, avg_evts) : tuple[float, float, float, float, float, float]
        Price range and return distribution stats; avg_evts is the mean multiple used.

    Notes
    -----
    • If no valid prices are produced (all NaN), zeros are returned.
    • All revenues are floored at 0; negative inputs are treated as 0.
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
   
    ub = config.lbp * price
  
    lb = config.ubp * price
    
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
        
        rets = {
            i: (price_i / price) - 1 for i, price_i in enumerate(prices)
        }
        
        rets_avg = np.mean(list(rets.values()))
   
        rets_std = np.std(list(rets.values()))

        return price_low, price_avg, price_high, rets_avg, rets_std, avg_evts
    
    else:
   
        return 0, 0, 0, 0, 0, 0



