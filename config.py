import datetime as dt
from pathlib import Path
import pickle
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

STOCK_SCREENER_FILE = BASE_DIR / "stock_screener/screener-stocks-2026-02-US.xlsx"

IND_DATA_FILE = BASE_DIR / 'ind_data_mc_all_simple_mean.xlsx'

CALIBRATION_FILE = BASE_DIR / "indicator_calibration.xlsx"

BETA_CACHE_FILE = BASE_DIR / "beta_cache.pkl"

ROOT_FIN_DIR = BASE_DIR / "modelling" / "stock_analysis_data"

RF = 0.038

RF_PER_WEEK = (1 + RF) ** (1/52) - 1

RF_PER_QUARTER = (1 + RF) ** (1/4) - 1

RF_PER_DAY = (1 + RF) ** (1/252) - 1

lbp, ubp = 0.2, 5

lbr, ubr = -0.8, 4

lbr_s, ubr_s = -0.6, 2

MAX_WEIGHT = 0.08

MONEY_IN_PORTFOLIO = 4000

TICKER_EXEMPTIONS = [""] #Insert Ticker excemptions, typically for use of an etf or a commodity, where a balance sheet or analyst targets do not exist.

benchmark = 'SP500'

IND_MAX_WEIGHT = 0.1

SECTOR_MAX_WEIGHT = 0.15

sector_limits = {
    'Technology': 0.4,
    "Financials": 0.2,
    'Communication Services': 0.2
}

BL_DELTA = 2.5

BL_TAU = 0.02

GAMMA = (1.0, 1.0, 1.0, 1.0, 1.0)


COV_USE_LOG_RETURNS = True

COV_USE_OAS = True

COV_USE_BLOCK_PRIOR = True

COV_USE_FX_FACTORS = True

COV_USE_REGIME_EWMA = True

COV_USE_GLOSSO = True

COV_USE_TERM_STRUCTURE = False

COV_USE_FUND_FACTORS = True   

COV_CACHE_DIR = BASE_DIR / "cov_cache"

COV_CACHE_MODE = "manual"

USE_CACHE = True

RESIMULATE_SCENARIOS = False

REUSE_SCENARIO_CACHE = True

REUSE_BVAR_POSTERIOR = True

REUSE_PREPROC_MATRICES = True

REUSE_WEIGHTS_CACHE = True

REUSE_PANEL_CACHE = True

CACHE_DIR = BASE_DIR / "lstm_cache"

ALLOW_GPU = False

SKIP_TRAIN_IF_WEIGHTS = True

SAVE_WEIGHTS = True

PROFILE = False

ENABLE_ISOTONIC_CAL = False

RESID_AR1_RHO = 0.30

EXOG_POSTERIOR_DRAWS = 1

tickers = np.sort(['']).tolist()#Insert Tickers here
