import config
import statsmodels.api as sm
import pandas as pd
import numpy as np


def ff5_pred(
    tickers: list[str],
    factor_data: pd.DataFrame,
    weekly_ret: pd.DataFrame,
    Cov_tv,                        
    E_factors_12m: pd.DataFrame,    
    E_mkt_ret: pd.Series,
    sims: dict[str, pd.DataFrame] | None = None,
    rf: float = config.RF,
) -> pd.DataFrame:
    """
    Expected twelve-month return from the Fama–French 5-factor model.

    If Cov_tv is a dict { "Q1": Σ1, … }, the function will use Σ1 for the
    Q1 expectation, Σ2 for Q2, etc.  If it is a single DataFrame the
    same matrix is used for all quarters (back-compat).
    """

    non_mkt = ["smb", "hml", "rmw", "cma"]

    if isinstance(Cov_tv, dict):
        
        cov_dict = {q: cov.loc[non_mkt, non_mkt] for q, cov in Cov_tv.items()}
  
    else:

        cov_fixed = Cov_tv.loc[non_mkt, non_mkt]
        
        cov_dict = {q: cov_fixed for q in E_mkt_ret.index}

    if isinstance(E_factors_12m, pd.Series):
       
        E_factors_12m = pd.DataFrame(
            np.tile(E_factors_12m.values, (len(E_mkt_ret), 1)),
            index=E_mkt_ret.index,
            columns=E_factors_12m.index,
        )

    results = []

    for ticker in tickers:

        df = (
            factor_data
            .join(weekly_ret[[ticker]], how="inner")
            .assign(ticker_excess=lambda d: d[ticker] - d["rf"])
            .dropna()
        )
      
        df_5y = df.loc[df.index.max() - pd.DateOffset(years=5):]

        X = sm.add_constant(df_5y[["mkt_excess", *non_mkt]])
      
        y = df_5y["ticker_excess"]
       
        betas = sm.OLS(y, X).fit().params.drop("const").values

        β_mkt = betas[0]
        β_oth = betas[1:]
        
        quarters = E_mkt_ret.index
        
        if sims is not None:
            
            n_sims = sims[next(iter(sims))].shape[1]
                        
            cum_ret = np.ones(n_sims)
            var_analytical = 0
                                
            for q in quarters:
                    
                mkt_xs = E_mkt_ret.loc[q] - config.RF_PER_QUARTER
                
                fac_paths = sims[q].values  

                E_q_xs_all = β_mkt * mkt_xs + β_oth.dot(fac_paths)

                E_q_tot_all = config.RF_PER_QUARTER + E_q_xs_all

                cum_ret *= (1 + E_q_tot_all)
                
                Σ_q = cov_dict[q].values
                
                var_analytical += β_oth @ Σ_q @ β_oth
                
                    
            tot_ret_sims = cum_ret - 1          
            tot_ret = tot_ret_sims.mean()
            
            se_sim = tot_ret_sims.std(ddof=1)

            sd_analytical = np.sqrt(var_analytical)

            sd_tot = np.sqrt(sd_analytical ** 2 + se_sim ** 2)   
        
        else:

            cum_ret = 1.0
            var_tot = 0.0

            for q in quarters:
            
                mkt_xs = E_mkt_ret.loc[q] - config.RF_PER_QUARTER
            
                fac_xs = E_factors_12m.loc[q, non_mkt].values

                E_q_xs = β_mkt * mkt_xs + β_oth.dot(fac_xs)

                E_q_tot = config.RF_PER_QUARTER + E_q_xs

                cum_ret *= 1 + E_q_tot

                Σ_q = cov_dict[q].values
                
                var_tot += β_oth @ Σ_q @ β_oth        

            tot_ret = cum_ret - 1
            
            sd_tot = np.sqrt(var_tot)            
            
            
        results.append({"Ticker": ticker, "Returns": tot_ret, "SE": sd_tot})

    return pd.DataFrame(results).set_index("Ticker")
