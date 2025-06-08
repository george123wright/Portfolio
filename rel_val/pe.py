import numpy as np
import pandas as pd

def pe_price_pred(eps_low: float,
                  eps_avg: float,
                  eps_high: float,
                  eps_low_y: float,
                  eps_avg_y: float,
                  eps_high_y: float,
                  pe_c: float,
                  pe_t: float,
                  pe_ind: float,
                  avg_pe_fs: float,
                  price: float) -> tuple[float, float, float, float, float]:
    """
    Compute the predicted price range based on P/E estimates.
    """
    prices = []
    eps_low = max(eps_low, 0)
    eps_avg = max(eps_avg, 0)
    eps_high = max(eps_high, 0)
    eps_low_y = max(eps_low_y, 0)
    eps_avg_y = max(eps_avg_y, 0)
    eps_high_y = max(eps_high_y, 0)
    lb = 0.2 * price
    ub = 5 * price
    eps_list  = pd.Series([eps_low, eps_avg, eps_high, eps_low_y, eps_avg_y, eps_high_y]).dropna()
    
    pe_list = pd.Series(
        [pe_c, pe_t, avg_pe_fs, pe_ind['Region-Industry'], pe_ind['Industry-MC']]
    ).dropna()
    
    avg_pe = np.mean(pe_list)
    
    for e in eps_list:
        for p in pe_list:
            price_i = np.clip(e * p, lb, ub)
            prices.append(max(price_i, 0))
        
    price_avg = np.mean(prices)
    price_min = np.min(prices)
    price_max = np.max(prices)
    
    rets = {i: (price_i / price) - 1 for i, price_i in enumerate(prices)}
    rets_avg = np.mean(list(rets.values()))
    rets_std = np.std(list(rets.values()))
    
    return price_min, price_avg, price_max, rets_avg, rets_std, avg_pe

