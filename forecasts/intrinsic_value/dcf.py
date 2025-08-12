"""
Performs discounted cash‑flow valuation using regression‑based cash‑flow forecasts and Monte‑Carlo scenarios.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import datetime as dt
import itertools
from sklearn.model_selection import TimeSeriesSplit

from functions.fast_regression import HuberENetCV  

from data_processing.financial_forecast_data import FinancialForecastData
import config


pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(
    level = logging.INFO, 
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

today_ts = pd.Timestamp.today().normalize()

fdata = FinancialForecastData()

macro = fdata.macro

r = macro.r

tickers = list(config.tickers)

latest_prices = r.last_price

market_cap = r.mcap

shares_outstanding = r.shares_outstanding

tax_rate = r.tax_rate

interest_rates = 0.042

cost_of_debt = interest_rates * (1.0 - tax_rate)

coe = pd.read_excel(
    config.FORECAST_FILE,
    sheet_name = "COE",
    index_col = 0,
    usecols = ["Ticker", "COE"],
    engine = "openpyxl",
)

r_dicts = r.dicts()

evs_ind_dict = r_dicts["EVS"]

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

keys = [
    "ocf", 
    "interest",
    "ebit", 
    "da",
    "sbc",
    "aq", 
    "ni", 
    "fcf", 
    "ebitda", 
    "cwc", 
    "ooa",
]

constrained_keys = {"ocf", "ebit", "ni", "fcf", "ebitda"}  

constrained_map = {k: (k in constrained_keys) for k in keys}

alphas = np.linspace(0.3, 0.7, 5)

lambdas = np.logspace(0, -4, 20)  

huber_M_values = (0.25, 1.0, 4.0)

cv_folds = 5

def _zero_dicts(
    ticker: str
):

    for d in (low_price_dict, avg_price_dict, high_price_dict, returns_dict, se_dict):

        d[ticker] = 0.0


low_price_dict: dict = {}

avg_price_dict: dict = {}

high_price_dict: dict = {}

returns_dict: dict = {}

se_dict: dict = {}

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
    "FCF",
]

lb = config.lbp * latest_prices 

ub = config.ubp * latest_prices

tscv = TimeSeriesSplit(n_splits = cv_folds)

kpis_dict = fdata.kpis

forecast_dict = fdata.forecast

fin_dict = fdata.annuals

cv = HuberENetCV(
    alphas = alphas,
    lambdas = lambdas,
    Ms = huber_M_values,
    n_splits = cv_folds,
    n_jobs = -1,                 
)

for ticker in tickers:
    
    logger.info("Processing ticker: %s", ticker)

    fin_df = fin_dict[ticker]
  
    if fin_df is None:
    
        logger.info("Skipping %s: missing financials.", ticker)
    
        _zero_dicts(
            ticker = ticker
        )
    
        continue

    present = [c for c in required_cols if c in fin_df.columns]
   
    missing = set(required_cols) - set(present)
   
    if missing:
       
        logger.warning("%s missing columns %s. Skipping regression.", ticker, missing)
       
        _zero_dicts(
            ticker = ticker
        )
       
        continue

    regression_data = fin_df.dropna(subset = present)
   
    if regression_data.empty:
        
        logger.warning("Regression data empty for %s. Skipping.", ticker)
       
        _zero_dicts(
            ticker = ticker
        )
        
        continue

    X = regression_data[["Revenue", "EPS"]].to_numpy(dtype = float)

    y_dict = {
        k: regression_data[column_map[k]].to_numpy(dtype = float) for k in keys
    }

    forecast_df = forecast_dict[ticker].dropna()
    
    if forecast_df.empty:
        
        logger.warning("No forecast data for %s, skipping DCF.", ticker)
       
        _zero_dicts(
            ticker = ticker
        )
      
        continue

    kpis = kpis_dict[ticker]
    
    evs_list = pd.Series(
        [kpis["exp_evs"].iat[0], evs_ind_dict[ticker]["Industry-MC"]]
    ).dropna()
    
    mc_ev = kpis["mc_ev"].iat[0]
    
    capex_rev_ratio = kpis["capex_rev_ratio"].iat[0]

    years = len(forecast_df)
    
    cv_splits = list(tscv.split(X))

    betas_by_key, best_lambda, best_alpha, best_M = cv.fit_joint(
        X = X,
        y_dict = y_dict,
        constrained_map = constrained_map,
        cv_splits = cv_splits,
        scorer = None, 
    )

    B = np.stack([betas_by_key[k] for k in keys], axis = 0) 
    
    rev_vals = forecast_df[["low_rev", "avg_rev", "high_rev"]].to_numpy(dtype = float)

    eps_vals = forecast_df[["low_eps", "avg_eps", "high_eps"]].to_numpy(dtype = float)


    all_idx = np.array(list(itertools.product(range(3), repeat = years)), dtype = int)

    n_combo = all_idx.shape[0]

    REVS = np.take_along_axis(rev_vals, all_idx.T, axis = 1).T 

    EPS = np.take_along_axis(eps_vals, all_idx.T, axis = 1).T 

    F = np.stack([np.ones_like(REVS), REVS, EPS], axis = -1)

    Y = np.einsum("tj,cyj->tcy", B, F)

    (
        ocf_p, 
        int_p, 
        ebit_p, 
        da_p, 
        sbc_p,
        aq_p,
        ni_p, 
        fcf_p, 
        ebitda_p,
        cwc_p, 
        ooa_p
    ) = Y

    capex = capex_rev_ratio * REVS 

    fcff_methods = [
        ("ocf_int_capex", lambda: ocf_p + int_p - capex),
        ("ebitda_based", lambda: ebit_p + da_p + sbc_p + ooa_p + aq_p - cwc_p - capex),
        ("ni_based", lambda: ni_p + int_p + da_p + sbc_p + ooa_p + aq_p - cwc_p - capex),
        ("fcf_plus_int", lambda: fcf_p + int_p),
        ("ebitda_minus", lambda: ebitda_p - capex + aq_p - cwc_p),
    ]

    valid_ff = []
   
    for name, fn in fcff_methods:
   
        arr = fn()  
   
        if np.isfinite(arr).all():
   
            valid_ff.append(arr)
   
        else:
   
            logger.info("• skipping FCFF method %s (non-finite inputs)", name)

    if not valid_ff:
       
        logger.warning("No complete FCFF formulas for %s; skipping.", ticker)
       
        _zero_dicts(
            ticker = ticker
        )
        
        continue

    fcff_raw = np.stack(valid_ff, axis = 0) 

    days = (forecast_df.index.to_numpy() - np.datetime64(today_ts)) / np.timedelta64(1, "D")

    years_frac = days / 365.0

    coe_t = float(coe.loc[ticker].iat[0])

    mv_debt = float(kpis["market_value_debt"].iat[0])

    E = float(market_cap[ticker])

    V = E + mv_debt
     
    WACC = (coe_t * E / V) + (cost_of_debt[ticker] * mv_debt / V)

    discount_factors = 1.0 / np.power(1.0 + WACC, years_frac) 

    disc_ff = fcff_raw * discount_factors[None, None, :]   
   
    sum_ff = disc_ff.sum(axis = 2)                              

    evs_arr = evs_list.to_numpy(dtype = float) 
   
    final_rev = REVS[:, -1]             
   
    tv_raw = evs_arr[:, None] * final_rev[None, :]                 
   
    tv_disc = tv_raw / np.power(1.0 + WACC, years)                 

    dcf = sum_ff[None, :, :] + tv_disc[:, None, :]

    shares_out = float(shares_outstanding[ticker])
    
    mc_ev_shares = mc_ev / shares_out 
   
    prices = (mc_ev_shares * dcf).clip(lb[ticker], ub[ticker])
   
    px = prices.reshape(-1)
    
    disc_ff_expanded = np.broadcast_to(disc_ff[None, :, :, :], (len(evs_arr),) + disc_ff.shape)
    
    na = forecast_df['num_analysts'].to_numpy(dtype = float)
    
    se_by_year = np.std(disc_ff_expanded, axis = (0, 1, 2), ddof = 1) / np.sqrt(na)
    
    tv_rep = np.repeat(tv_disc[:, :, None], 5, axis = 2).flatten()
    
    tv_se = np.std(tv_rep, ddof = 1) / np.sqrt(float(forecast_df['num_analysts'].iat[-1]))

    se = np.sqrt(np.sum(se_by_year ** 2) + tv_se ** 2) * mc_ev_shares

    low_price = float(np.nanpercentile(px, 10))
    med_price = float(np.nanpercentile(px, 50))
    high_price = float(np.nanpercentile(px, 90))

    low_price_dict[ticker] = low_price
    avg_price_dict[ticker] = med_price
    high_price_dict[ticker] = high_price
    se_dict[ticker] = se

    latest_price = latest_prices[ticker]
    returns_dict[ticker] = float((med_price / latest_price - 1.0))

    logger.info(
        "Ticker: %s | Low (P10): %.2f | Med (P50): %.2f | High (P90): %.2f | SE: %.2f ",
        ticker, low_price, med_price, high_price, se
    )

dcf_df = pd.DataFrame(
    {
        "Low Price": low_price_dict,
        "Avg Price": avg_price_dict, 
        "High Price": high_price_dict,
        "SE": se_dict,
    }
)

dcf_df.index.name = "Ticker"

with pd.ExcelWriter(
    config.MODEL_FILE, mode = "a", engine = "openpyxl", if_sheet_exists = "replace"
) as writer:
   
    dcf_df.to_excel(writer, sheet_name = "DCF")
