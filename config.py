import datetime as dt
from pathlib import Path
import numpy as np

TODAY = dt.date.today()

YEAR_AGO = TODAY - dt.timedelta(days = 365)
THREE_YEAR_AGO = TODAY - dt.timedelta(days = 3 * 365)
FIVE_YEAR_AGO = TODAY - dt.timedelta(days = 5 * 365)

BASE_DIR = Path("")

DATA_FILE = BASE_DIR / f"Portfolio_Optimisation_Data_{TODAY}.xlsx"
FORECAST_FILE = BASE_DIR / f"Portfolio_Optimisation_Forecast_{TODAY}.xlsx"
PORTFOLIO_FILE = BASE_DIR / f"Portfolio_Optimisation_Portfolio_{TODAY}.xlsx"
MODEL_FILE = BASE_DIR / "Portfolio_Optimisation_DCF.xlsx"
REL_VAL_FILE = BASE_DIR / f"Portfolio_Optimisation_Relative_Valuation_{TODAY}.xlsx"
STOCK_SCREENER_FILE = BASE_DIR / "stock_screener/screener-stocks-2025-07-US.xlsx"
IND_DATA_FILE = BASE_DIR / 'ind_data_mc_all_simple_mean.xlsx'


ROOT_FIN_DIR = BASE_DIR / "modelling" / "stock_analysis_data"

RF = 0.0435
RF_PER_WEEK = (1 + RF) ** (1/52) - 1
RF_PER_QUARTER = (1 + RF) ** (1/4) - 1

lbp, ubp = 0.2, 5
lbr, ubr = -0.8, 4

MAX_WEIGHT = 0.1

MONEY_IN_PORTFOLIO = 4000

TICKER_EXEMPTIONS = [""]  #Insert Ticker excemptions, typically for use of an etf or a commodity, where a balance sheet or analyst targets do not exist.

benchmark = 'SP500'

sector_limits = {
    'Technology': 0.3,
    'Healthcare': 0.1    
}

BL_DELTA = 2.5
BL_TAU = 0.02


tickers = np.sort([""]).tolist() #Insert Tickers here
