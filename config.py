import datetime as dt

TODAY = dt.date.today()
YEAR_AGO = TODAY - dt.timedelta(days=365)
FIVE_YEAR_AGO = TODAY - dt.timedelta(days=5 * 365)

BASE_DIR = "/Users/georgewright"

DATA_FILE = BASE_DIR / f"Portfolio_Optimisation_Data_{TODAY}.xlsx"
FORECAST_FILE = BASE_DIR / f"Portfolio_Optimisation_Forecast_{TODAY}.xlsx"
MODEL_FILE = BASE_DIR / f"/Users/georgewright/Portfolio_Optimisation_DCF.xlsx"

ROOT_FIN_DIR = BASE_DIR / "modelling/stock_analysis_data"

RF = 0.046

lbr = 0.2
ubr = 5

MAX_WEIGHT = 0.1

MONEY_IN_PORTFOLIO = 4000
