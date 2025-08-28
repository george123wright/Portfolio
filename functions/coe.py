"""
Calculates cost of equity per ticker using betas, country risk premiums and currency premiums.
"""

import pandas as pd
import statsmodels.api as sm
import config


def calculate_cost_of_equity(
    tickers: list[str],
    rf: float,
    returns: pd.DataFrame, 
    index_close: pd.Series,
    spx_expected_return: float,
    crp_df: pd.DataFrame,
    currency_bl_df: pd.DataFrame,
    country_to_pair: pd.DataFrame,
    ticker_country_map: dict[str, str]
) -> pd.DataFrame:
    """
    Estimate cost of equity (COE) per ticker combining CAPM beta, country risk premium,
    and currency risk premium.

    Workflow per ticker t:
   
    1) Compute weekly excess-return beta by OLS:
   
        y_t = R_t − r_f,    
        
        X_t = R_m,t − r_f,    
        
    and fit 
    
        y_t = α + β X_t + ε_t,
        
    where r_f is the *weekly* risk-free rate (e.g., `config.RF_PER_WEEK`).
    We keep the slope β = beta_t.

    2) Compose expected equity premium:
    
        ERP_country_currency = max( (β_t · max(E[R_m] − r_f, 0)) + CRP_country + FX_premium, 0 ),
    
    where
    
        • E[R_m] is the *posterior* expected S&P500 return (e.g. from Black–Litterman),
    
        • CRP_country is the country risk premium looked up in `crp_df['CRP']`,
        defaulting to its cross-sectional mean if missing,
      
        • FX_premium is the currency premium for the issuer country, looked up
        via `country_to_pair` → currency pair string and then into `currency_bl_df`
        (e.g., expected currency BL return). If the country is 'United Kingdom',
        FX_premium is set to 0 in this implementation.

    3) Final cost of equity:
      
        COE_t = r_f + ERP_country_currency.

    Inputs
    ------
    tickers : list[str]
        Ticker symbols to process.
    rf : float
        Annual or periodic risk-free rate used in the final COE formula (same periodicity as `spx_expected_return`).
    returns : pandas.DataFrame
        Asset price series transformed to returns; columns include each ticker.
    index_close : pandas.Series
        Benchmark index closing prices; used to build weekly returns for the OLS.
    spx_expected_return : float
        E[R_m] used in CAPM premium (same periodicity as `rf`).
    crp_df : pandas.DataFrame
        Country risk premia; index must contain country names, column 'CRP'.
    currency_bl_df : pandas.DataFrame or pandas.Series
        Currency premiums (e.g., BL posterior). Indexed by currency pair code; accessed via `.at[code]`.
    country_to_pair : dict-like
        Mapping from country name to currency pair (e.g., 'United States' → 'GBPUSD').
    ticker_country_map : dict[str,str]
        Mapping from ticker to issuer country name.

    Returns
    -------
    pandas.DataFrame
        Indexed by ticker, columns:
        ['Country','Beta','CRP','Currency Premium','COE'].

    Notes
    -----
    • Weekly OLS uses: 
    
        y = R_stock_weekly − rf_weekly, 
        
        X = R_index_weekly − rf_weekly, with an intercept.
    
    • Non-negative guards:
    
    – The term (E[R_m] − r_f) is floored at 0 before multiplying by β.
    
    – The aggregate premium added to r_f is floored at 0.
    """

   
    records = []
   
    def_cur = 'GBPUSD'
    
    index_rets = index_close.resample('W').last().pct_change().dropna()
    
    crp_mean = crp_df['CRP'].mean()
   
    for t in tickers:
           
        df = pd.concat([returns[t], index_rets], axis=1).dropna()
    
        df.columns = ['p', 'b']
        
                
        y = df['p'] - config.RF_PER_WEEK
        X = df['b'] - config.RF_PER_WEEK
        
        X = sm.add_constant(X)
       
        model = sm.OLS(y, X).fit()

        country = ticker_country_map.get(t, 'NA')
       
        crp_val = crp_df['CRP'].get(country, crp_mean)
        
        beta = model.params['b']
   
        if country == 'United Kingdom':
       
            curr_prem = 0
       
        else:
       
            ccy = country_to_pair.get(country, def_cur)
       
            curr_prem = currency_bl_df.at[ccy]
   
        coe = rf + max(((beta * max((spx_expected_return - rf), 0)) + crp_val + curr_prem), 0)
   
        records.append({
            'Ticker': t,
            'Country': country,
            'Beta': beta,
            'CRP': crp_val,
            'Currency Premium': curr_prem,
            'COE': coe
        })
   
    return pd.DataFrame(records).set_index('Ticker')




