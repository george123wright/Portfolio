"""
Performs discounted cash‑flow valuation using regression‑based cash‑flow forecasts and Monte‑Carlo scenarios.
"""

import pandas as pd
import datetime as dt
import numpy as np
import logging
from sklearn.model_selection import KFold
from functions.fast_regression import grid_search_regression, constrained_regression, ordinary_regression
from data_processing.financial_forecast_data import FinancialForecastData
import config


pd.set_option('future.no_silent_downcasting', True)

logging.basicConfig(
    level = logging.INFO, 
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

today_ts = pd.Timestamp.today().normalize()

fdata = FinancialForecastData()

macro = fdata.macro

r = macro.r

tickers = r.tickers

close = r.close

latest_prices_series = pd.Series({ticker: close[ticker].iloc[-1] for ticker in tickers})

latest_prices = latest_prices_series.to_dict()

ticker_data = r.analyst

market_cap = pd.Series({ticker: ticker_data.loc[ticker, 'marketCap'] for ticker in tickers})

shares_outstanding = pd.Series({ticker: ticker_data.loc[ticker, 'sharesOutstanding'] for ticker in tickers})

tax_rate = pd.Series({ticker: ticker_data.loc[ticker, 'Tax Rate'] for ticker in tickers})

coe = pd.read_excel(config.FORECAST_FILE, 
                    sheet_name = 'COE', 
                    index_col = 0, 
                    usecols = ['Ticker', 'COE'], 
                    engine = 'openpyxl'
                    )

r_dicts = r.dicts()

evs_ind_dict = r_dicts['EVS']

alphas = np.linspace(0.3, 0.7, 5)

lambdas = np.logspace(-4, 1, 20)

huber_M_values = (0.25, 1.0, 4.0)

param_grid = [(a, l, m)
                for a in alphas
                for l in lambdas
                for m in huber_M_values]
cv_folds = 5

kf = KFold(n_splits = cv_folds, shuffle = True, random_state = 123)

column_map = {
    "ocf": "OCF",
    "interest": "InterestAfterTax",
    "ebit": "EBIT",
    "da": "Depreciation & Amortization",
    "sbc": "Share-Based Compensation",
    "aq": "Acquisitions",
    "ni": "Net Income",
    "fcf": "FCF",
    "ebitda": "EBITDA",
    "cwc": "Change in Working Capital",
    "ooa": "Other Operating Activities",
}

keys = ["ocf", "interest", "ebit", "da", "sbc", "aq", "ni", "fcf", "ebitda", "cwc", "ooa"]

funcs = [
    constrained_regression, 
    constrained_regression, 
    constrained_regression,
    ordinary_regression, 
    ordinary_regression, 
    ordinary_regression, 
    constrained_regression, 
    constrained_regression, 
    constrained_regression, 
    ordinary_regression, 
    ordinary_regression
]


def _zero_dicts(
    ticker
):
    
    for d in (low_price_dict, avg_price_dict, high_price_dict, returns_dict, se_dict):
        
        d[ticker] = 0


low_price_dict, avg_price_dict, high_price_dict, returns_dict, se_dict = {}, {}, {}, {}, {}

required_cols = [
    "Revenue", 
    "EPS", 
    "OCF", 
    "InterestAfterTax", 
    "Capex",
    "EBIT", 
    "Depreciation & Amortization", 
    "Share-Based Compensation",
    "Acquisitions", 
    "Net Income", 
    "EBITDA",
    "Change in Working Capital", 
    "Other Operating Activities", 
    "FCF"
]

for ticker in tickers:
    
    logger.info("Processing ticker: %s", ticker)
    
    fin_df = fdata.annuals.get(ticker)
    
    if fin_df is None:
    
        logger.info("Skipping %s: missing financials.", ticker)
    
        _zero_dicts(
            ticker = ticker
        )
        
        continue

    present = [c for c in required_cols if c in fin_df.columns]
   
    missing = set(required_cols) - set(present)
   
    if missing:
        
        logger.warning(f"{ticker} missing columns {missing}. Skipping regression.")
        
        _zero_dicts(
            ticker = ticker
        )
        
        continue

    regression_data = fin_df.dropna(subset=present)
   
    if regression_data.empty:
        
        logger.warning(f"Regression data empty for {ticker}. Skipping.")
        
        _zero_dicts(
            ticker = ticker
        )
        
        continue

    X = regression_data[["Revenue", "EPS"]].values
   
    y_series = {col: regression_data[col].values for col in ["OCF", 
                                                             "InterestAfterTax",
                                                             "EBIT", 
                                                             "Depreciation & Amortization",
                                                             "Share-Based Compensation", 
                                                             "Acquisitions",
                                                             "Net Income", 
                                                             "FCF", 
                                                             "EBITDA",
                                                             "Change in Working Capital", 
                                                             "Other Operating Activities"
                                                             ]}

    forecast_df = fdata.forecast[ticker].dropna()

    if forecast_df.empty:
        
        logger.warning(f"No forecast data for {ticker}, skipping DCF.")
        
        _zero_dicts(
            ticker = ticker
        )
        
        continue
 
    kpis = fdata.kpis[ticker]
 
    evs_list = pd.Series([kpis['exp_evs'].iat[0], evs_ind_dict[ticker]['Industry-MC']]).dropna()
    
    mc_ev = kpis['mc_ev'].iat[0]
 
    years = len(forecast_df)
    
    cv_splits = list(kf.split(X)) 
    
    regs = {}
    
    for key, func in zip(keys, funcs):
    
        y_col = column_map[key]
    
        regs[key] = grid_search_regression(
            X = X,
            y = y_series[y_col],
            regression_func = func,
            param_grid = param_grid,    
            cv_splits = cv_splits     
        )[0]

    beta_ocf = regs['ocf']; beta_interest = regs['interest']; beta_ebit = regs['ebit']
   
    beta_da = regs['da']; beta_sbc = regs['sbc']; beta_aq = regs['aq']
   
    beta_ni = regs['ni']; beta_fcf = regs['fcf']; beta_ebitda = regs['ebitda']
   
    beta_cwc = regs['cwc']; beta_ooa = regs['ooa']

    coe_t = coe.loc[ticker].values
    
    mv_debt = kpis['market_value_debt'].iat[0]
    
    E = market_cap[ticker]; V = E + mv_debt
    
    interest_rates = 0.042
    
    if pd.isna(tax_rate[ticker]): 
        
        tax_rate[ticker] = 0.22
    
    cost_of_debt = interest_rates * (1 - tax_rate[ticker])
   
    WACC = (coe_t * E / V) + (cost_of_debt * mv_debt / V)
    
    capex_rev_ratio = kpis['capex_rev_ratio'].iat[0]

    rev_vals = forecast_df[['low_rev','avg_rev','high_rev']].to_numpy()
    
    eps_vals = forecast_df[['low_eps','avg_eps','high_eps']].to_numpy()

    REVS_mesh = np.empty((years, 3, 3)); EPS_mesh = np.empty((years, 3, 3))
   
    for i in range(years):
       
        REVS_mesh[i], EPS_mesh[i] = np.meshgrid(rev_vals[i], eps_vals[i], indexing='ij')

    rev_flat = REVS_mesh.reshape(years, -1); eps_flat = EPS_mesh.reshape(years, -1)
   
    idx = np.indices([rev_flat.shape[1]] * years).reshape(years, -1)
  
    REVS = rev_flat[np.arange(years)[:, None], idx].T  
   
    EPS = eps_flat[np.arange(years)[:, None], idx].T 

    ocf_p = beta_ocf[0] + (beta_ocf[1] * REVS) + (beta_ocf[2] * EPS)
    
    int_p = beta_interest[0] + (beta_interest[1] * REVS) + (beta_interest[2] * EPS)
    
    ebit_p = beta_ebit[0] + (beta_ebit[1] * REVS) + beta_ebit[2] * EPS
    
    da_p = beta_da[0] + (beta_da[1] * REVS )+ (beta_da[2] * EPS)
    
    sbc_p = beta_sbc[0] + (beta_sbc[1] * REVS) + (beta_sbc[2] * EPS)
    
    aq_p = beta_aq[0] + (beta_aq[1] * REVS) + (beta_aq[2] * EPS)
    
    ni_p = beta_ni[0] + (beta_ni[1] * REVS) + (beta_ni[2] * EPS)
    
    fcf_p = beta_fcf[0] + (beta_fcf[1] * REVS) + (beta_fcf[2] * EPS)
    
    ebitda_p = beta_ebitda[0] + (beta_ebitda[1] * REVS) + (beta_ebitda[2] * EPS)
    
    cwc_p = beta_cwc[0] + (beta_cwc[1] * REVS) + (beta_cwc[2] * EPS)
    
    ooa_p = beta_ooa[0] + (beta_ooa[1] * REVS) + (beta_ooa[2] * EPS)

    capex = capex_rev_ratio * REVS  
    
    fcff_formulas = {
        "ocf_int_capex": lambda: ocf_p  + int_p +   - capex,
        "ebitda_based":  lambda: ebit_p + da_p + sbc_p + ooa_p + aq_p - cwc_p - capex,
        "ni_based":      lambda: ni_p   + int_p + da_p + sbc_p + ooa_p + aq_p - cwc_p - capex,
        "fcf_plus_int":  lambda: fcf_p  + int_p,
        "ebitda_minus":  lambda: ebitda_p       - capex + aq_p - cwc_p,
    }
    
    valid_ff = []
    
    for name, fn in fcff_formulas.items():
        
        arr = fn()                   
        
        if np.isfinite(arr).all():   
       
            valid_ff.append(np.maximum(arr, 0))
       
        else:
       
            logger.info(f"  • skipping FCFF method {name} (missing inputs)")

    if not valid_ff:
       
        logger.warning(f"No complete FCFF formulas for {ticker}; skipping.")
       
        _zero_dicts(
            ticker = ticker
        )
       
        continue

    fcff_raw = np.stack(valid_ff, axis=0)

    days = (forecast_df.index.to_numpy() - np.datetime64(today_ts)) / np.timedelta64(1, 'D')
    discount_factors = 1.0 / ((1.0 + WACC) ** (days / 365.0)) 

    disc_ff = fcff_raw * discount_factors[None, None, :]
    sum_ff  = disc_ff.sum(axis = 2)

    final_rev = REVS[:, -1] 

    evs_arr = evs_list.values
    
    tv_raw = evs_arr[:, None] * final_rev[None, :]
    
    tv_disc = tv_raw / ((1 + WACC) ** years)

    dcf = sum_ff[None, :, :] + tv_disc[:, None, :]

    lb = config.lbp * latest_prices_series[ticker] 
    ub = config.ubp * latest_prices_series[ticker]
    
    shares_out = shares_outstanding[ticker]
    
    prices = np.clip(mc_ev * dcf / shares_out, lb, ub)  

    all_prices = prices.flatten()
    
    low_price_dict[ticker] = max(all_prices.min(), 0)
    avg_price_dict[ticker] = max(all_prices.mean(), 0)
    high_price_dict[ticker] = max(all_prices.max(), 0)

    disc_ff_expanded = np.broadcast_to(disc_ff[None, :, :, :], (len(evs_arr),) + disc_ff.shape)
    
    na = forecast_df['num_analysts'].to_numpy(dtype=float)
    
    se_by_year = np.std(disc_ff_expanded, axis=(0,1,2), ddof=1) / np.sqrt(na)
   
    tv_rep = np.repeat(tv_disc[:, :, None], 5, axis=2).flatten()
    
    tv_se = np.std(tv_rep, ddof=1) / np.sqrt(float(forecast_df['num_analysts'].iat[-1]))

    se = np.sqrt(np.sum(se_by_year**2) + tv_se**2) * mc_ev / shares_out
    
    se_dict[ticker] = se

    latest_price = latest_prices[ticker]
    
    returns_dict[ticker] = max((avg_price_dict[ticker] / latest_price) - 1, -1)

    logger.info(f"Ticker: {ticker}, Low: {low_price_dict[ticker]:.2f}, Avg: {avg_price_dict[ticker]:.2f}, "
                f"High: {high_price_dict[ticker]:.2f}, SE: {se_dict[ticker]:.2f}")

dcf_df = pd.DataFrame({
    'Low Price': low_price_dict,
    'Avg Price': avg_price_dict,
    'High Price': high_price_dict,
    'SE': se_dict
})

dcf_df.index.name = 'Ticker'

with pd.ExcelWriter(
    config.MODEL_FILE,
    mode = 'a', 
    engine = 'openpyxl', 
    if_sheet_exists = 'replace'
) as writer:
    
  dcf_df.to_excel(writer, sheet_name='DCF')
