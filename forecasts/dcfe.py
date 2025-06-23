"""
Calculates discounted cash‑flow to equity valuations using constrained regression on financials and analyst forecasts.
"""

import pandas as pd
import datetime as dt
import numpy as np
import logging
from data_processing.financial_forecast_data import FinancialForecastData
from functions.fast_regression import constrained_regression


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

today_ts = pd.Timestamp(config.TODAY)

fdata = FinancialForecastData()
macro = fdata.macro
r = macro.r
tickers = r.tickers 
latest_prices = r.last_price.copy()

ticker_metadata = r.analyst

market_cap = ticker_metadata['marketCap']
shares_out = ticker_metadata['sharesOutstanding']

tax_rate = {ticker: ticker_metadata['Tax Rate'][ticker] if pd.notna(ticker_metadata['Tax Rate'][ticker]) else 0.22 for ticker in tickers}

coe = pd.read_excel(config.FORECAST_FILE, sheet_name='COE', index_col=0, usecols=['Ticker', 'COE'], engine='openpyxl')

dicts = r.dicts()

growth_dict = dicts['eps1y_5']
pe_ind_dict = dicts['PE']

low_price = {}
avg_price = {}
high_price = {}
se_dict = {}

for ticker in tickers:
    
    fin_df = fdata.annuals.get(ticker)
    fc_df = fdata.forecast[ticker].copy()
    kpis = fdata.kpis.get(ticker)
    
    if fin_df is None or fc_df is None or kpis is None:
        logger.info(f"Skipping {ticker}: missing data")
        low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = 0
        continue
    
    fc_df = fc_df.dropna(
        subset=['low_rev','avg_rev','high_rev','low_eps','avg_eps','high_eps']
    )
   
    fc_df.index = pd.to_datetime(fc_df.index)
    fc_df = fc_df.sort_index()  
   
    if fc_df.empty:
        logger.warning(f"{ticker}: no valid forecast data after dropping all‐NaN rows, skipping")
        low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = np.nan
        continue

    reg = fin_df.dropna(subset=["Revenue","EPS","OCF","NetBorrowing","Capex"])
   
    X = reg[['Revenue','EPS']].values
   
    y = (reg['OCF'] + reg['NetBorrowing'] + reg['Capex']).values
   
    if len(y) < 2:
   
        logger.warning(f"Not enough regression data for {ticker}")
        low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = 0
        continue

    beta = constrained_regression(X, y)

    mc_debt = kpis['market_value_debt'].iat[0]
       
    coe_t = coe.loc[ticker].values

    E = market_cap[ticker]
   
    V = E + mc_debt
    
    cost_debt = 0.042 * (1 - tax_rate[ticker])
       
    years = len(fc_df)

    for col in ['low_rev','avg_rev','high_rev']:
        fc_df[col] = fc_df[col].replace({'T':'e12','B':'e9','M':'e6'}, regex=True).astype(float)

    rev_opts = fc_df[['low_rev','avg_rev','high_rev']].values  
    eps_opts = fc_df[['low_eps','avg_eps','high_eps']].values  

    grids = np.meshgrid(*[range(3) for _ in range(years)], indexing='ij')

    idxs = np.stack([g.flatten() for g in grids], axis=1)

    rev_matrix = rev_opts[np.arange(years)[:,None], idxs.T].T
    eps_matrix = eps_opts[np.arange(years)[:,None], idxs.T].T

    days = (fc_df.index.to_numpy() - np.datetime64(today_ts)) / np.timedelta64(1, 'D')
    discount = 1.0 / ((1 + coe_t) ** (days / 365.0))

    fcfs = beta[0] + (rev_matrix * beta[1]) + (eps_matrix * beta[2])
   
    dcf_years = fcfs * discount[np.newaxis, :]  

    sum_dcf = dcf_years.sum(axis=1)

    final_eps = eps_matrix[:, -1]
    
    exp_pe = kpis['exp_pe'].iat[0] 
    pe_used = exp_pe if exp_pe > 0 else pe_ind_dict[ticker]['Industry-MC']
    pe_list = np.array([pe_used, pe_ind_dict[ticker]['Region-Industry']])
    
    tv_raw = pe_list[:, None] * final_eps[None, :] * shares_out[ticker]
    tv_disc = tv_raw / ((1 + coe_t) ** years) 

    dcf_vals = sum_dcf[None, :] + tv_disc

    prices = (dcf_vals / shares_out[ticker]).clip(0.2 * latest_prices[ticker],
                                                  5.0 * latest_prices[ticker])
    
    flat_prices = prices.flatten()

    low_price[ticker] = max(flat_prices.min(), 0)
    avg_price[ticker] = max(flat_prices.mean(), 0)
    high_price[ticker] = max(flat_prices.max(), 0)

    dfcf = dcf_years 
    stds = dfcf.std(axis=0)
  
    n = fc_df["num_analysts"].iloc[:years].astype(float).values
    n[n == 0] = 1.0 
    
    ses = stds / np.sqrt(n)

    tv_flat = tv_disc.flatten()
    
    std_term = tv_flat.std()
    
    n_term = float(fc_df["num_analysts"].iat[-1])
    
    if n_term < 1:
        n_term = 1.0
    
    se_term = std_term / np.sqrt(n_term)

    se_total = np.sqrt((ses**2).sum() + se_term**2)
    se_dict[ticker] = se_total / shares_out[ticker]

    logger.info(f"{ticker}: Low {low_price[ticker]}, Avg {avg_price[ticker]}, High {high_price[ticker]}, SE {se_dict[ticker]}")


dcf_df = pd.DataFrame({
    'Low Price': low_price,
    'Avg Price': avg_price,
    'High Price': high_price,
    'SE': se_dict
})

dcf_df.index.name = 'Ticker'

with pd.ExcelWriter(config.MODEL_FILE, mode='a',
                     engine='openpyxl', if_sheet_exists='replace') as writer:
     dcf_df.to_excel(writer, sheet_name='DCFE')
