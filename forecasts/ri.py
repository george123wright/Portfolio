"""
Implements a residual income valuation model combining book value growth, cost of equity and analyst estimates.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import itertools
import datetime as dt
import logging
from data_processing.financial_forecast_data import FinancialForecastData

df_opts = {'future.no_silent_downcasting': True}
for opt, val in df_opts.items():
    pd.set_option(opt, val)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

today     = dt.date.today()
today_ts  = pd.Timestamp.today().normalize()

EXCEL_OUT_FILE   = f'Portfolio_Optimisation_Forecast_{today}.xlsx'
EXCEL_IN_FILE    = f'Portfolio_Optimisation_Data_{today}.xlsx'

fdata   = FinancialForecastData()
macro  = fdata.macro
r = macro.r
tickers = r.tickers
latest_prices = r.last_price

shares_outstanding = pd.Series({
    t: float(r.analyst.at[t, 'sharesOutstanding'])
    for t in tickers
})

rf = 0.046

COE_MEASURE = 'black' 
sheet_name  = 'CAPM BL Pred' if COE_MEASURE == 'black' else 'CAPM Pred'
coe_table   = pd.read_excel(
    EXCEL_OUT_FILE,
    sheet_name=sheet_name,
    header=0,
    index_col=0,
    na_values=0,
    engine='openpyxl'
)
coe_series  = coe_table['Returns'].astype(float)

def bvps(eps, prev_bvps, dps):
    return float(prev_bvps) + float(eps) - float(dps)

def growth(roe, payout_ratio, ind_g):
    return ind_g if roe < 0 else roe * (1 - payout_ratio)

def calc_div_growth(div):
    div = div.abs().fillna(0).replace(0, np.nan).dropna()
    if len(div) < 2:
        return 0.0
    pct = div.pct_change().dropna()
    return float((1 + pct).prod()**(1/len(pct)) - 1)

results      = r.dicts()
growth_dict  = results['eps1y_5']
low_price    = {}
avg_price    = {}
high_price   = {}
returns_dict = {}
se_dict      = {}

for ticker in tickers:
    fin_df      = fdata.annuals.get(ticker)
    forecast_df = fdata.forecast[ticker]
    kpis        = fdata.kpis[ticker]

    if fin_df.empty:
        logger.info("Skipping %s: missing financials.", ticker)
        low_price[ticker] = avg_price[ticker] = high_price[ticker] = returns_dict[ticker] = se_dict[ticker] = 0
        continue

    bvps_0       = kpis['bvps_0'].iat[0]
    price_book   = kpis['exp_ptb'].iat[0]
    roe          = kpis['roe'].iat[0]
    payout_ratio = kpis['payout_ratio'].iat[0] if not pd.isna(kpis['payout_ratio'].iat[0]) else 0
    ind_g        = growth_dict[ticker]['Region-Industry']
    g            = roe if roe < 0 else roe * (1 - payout_ratio)

    div          = fin_df['Div'].abs().fillna(0)
    div_growth   = calc_div_growth(div)
    dps          = float(div.mean()) / shares_outstanding[ticker]
    
    lp = latest_prices[ticker]
    lb = 0.2 * lp
    ub = 5.0 * lp
    
    coe = coe_series[ticker]
    
    valid = forecast_df[['low_eps','avg_eps','high_eps','num_analysts']].dropna(how='any')
    
    years    = valid.shape[0]
    eps_opts = valid[['low_eps','avg_eps','high_eps']].to_numpy()
    eps_grid = np.array(list(itertools.product(*eps_opts)))
    combos   = eps_grid.shape[0]
    
    dps_vec  = dps * (1 + div_growth)**np.arange(years)
    delta    = eps_grid - dps_vec[None, :]
    cs       = np.cumsum(delta, axis=1)
    BVPS_prev = np.hstack([
        np.full((combos,1), bvps_0),
        bvps_0 + cs[:, :-1]
    ])

    days     = (valid.index.to_numpy() - np.datetime64(today_ts)) / np.timedelta64(1,'D')
    disc1d   = np.power(1 + coe, - days / 365.0)
 
    ri_terms = (eps_grid - coe * BVPS_prev) * disc1d[None, :]
    ri_sum   = ri_terms.sum(axis=1)

    last_eps  = eps_grid[:, -1]
    last_bvps = BVPS_prev[:, -1]
    eps_tp1   = last_eps * (1 + g)
    term_raw  = eps_tp1 - coe * last_bvps
    good      = (coe - g) > 0
    term_val  = np.where(good,
                          term_raw / (coe - g),
                          price_book * last_bvps)
    term_disc = term_val * disc1d[-1]

    total_RI      = bvps_0 + ri_sum + term_disc
    prices_all    = np.clip(total_RI, lb, ub)
    low_price[ticker]  = max(prices_all.min(), 0)
    avg_price[ticker]  = max(prices_all.mean(), 0)
    high_price[ticker] = max(prices_all.max(), 0)
    returns_dict[ticker] = max((avg_price[ticker]/lp) - 1, -1)

    na       = valid['num_analysts'].to_numpy(dtype=float)
    stds_by_year     = ri_terms.std(axis=0, ddof=1)
    ses_by_year      = stds_by_year / np.sqrt(na)
    se_term          = term_disc.std(ddof=1) / np.sqrt(na[-1])
    se_dict[ticker]  = np.sqrt((ses_by_year**2).sum() + se_term**2)

    logger.info(
        f"{ticker}: Low {low_price[ticker]:.2f}, Avg {avg_price[ticker]:.2f}, "
        f"High {high_price[ticker]:.2f}, SE {se_dict[ticker]:.4f}"
    )

df_ri = pd.DataFrame({
    'Low Price': low_price,
    'Avg Price': avg_price,
    'High Price': high_price,
    'SE': se_dict
})
df_ri.index.name = 'Ticker'
excel_file3 = ""
with pd.ExcelWriter(excel_file3, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
   df_ri.to_excel(writer, sheet_name='RI')
