from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def factor_sim(
    factor_data: pd.DataFrame,
    num_factors: int,
    n_sims: int = 1_000,
    horizon: int = 4,
    max_lag: int = 4,
    seed: int | None = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:

    if num_factors == 5:
        
        factors = ["smb", "hml", "rmw", "cma"]
    
    elif num_factors == 3:
        
        factors = ["smb", "hml"]
    
    else:
       
        raise ValueError("num_factors must be 3 or 5")

    fac_df = factor_data[factors].dropna()

    var_mod = VAR(fac_df)
    
    p_opt = var_mod.select_order(max_lag).aic or 1
    
    var_res = var_mod.fit(int(max(1, p_opt)))

    c = var_res.params.loc["const"].values        

    A = var_res.coefs                      

    Sigma_u = var_res.sigma_u        
               
    chol = np.linalg.cholesky(Sigma_u)

    p = var_res.k_ar
   
    history = fac_df.values[-p:] 

    k = len(factors)
   
    sims = np.empty((horizon, k, n_sims))
   
    rng = np.random.default_rng(seed)

    for j in range(n_sims):
       
        buf = history.copy() 
   
        for t in range(horizon):
   
            u_t = chol @ rng.standard_normal(k)    

            x_t = c.copy()

            for lag in range(p):

                x_t += A[lag].dot(buf[-lag-1])

            x_t += u_t

            sims[t, :, j] = x_t

            buf = np.vstack([buf, x_t])[ -p: ]

    idx_q = [f"Q{i}" for i in range(1, horizon+1)]

    mean_q = pd.DataFrame(sims.mean(axis=2), index=idx_q, columns=factors)

    cov_q  = {
        q: pd.DataFrame(np.cov(sims[i], 
                        rowvar = True),
                        index = factors, 
                        columns = factors
                        )
        for i, q in enumerate(idx_q)
    }

    sims_q = {
        q: pd.DataFrame(
            sims[i],
            index = factors,
            columns = [f"{n+1}" for n in range(n_sims)]
        )
        for i, q in enumerate(idx_q)
    }
    
    print(sims_q)

    return cov_q, mean_q, sims_q
