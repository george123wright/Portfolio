"""
Downloads historical market data, computes technical indicators, scrapes Trading Economics forecasts and exports all series to Excel.
"""

import datetime as dt
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt import expected_returns
from pandas_datareader import data as web
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
from maps.SecMap import sec_map
from maps.industry_mapping import IndustryMap
from functions.export_forecast import export_results


BASE_FORECAST_URL = "https://tradingeconomics.com/forecast"


FORECAST_SPECS: Dict[str, Dict[str, Any]] = {
    "interest-rate": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last", 
            3: "Q2/25", 
            4: "Q3/25", 
            5: "Q4/25", 
            6: "Q1/26"
        }
    },
    "stock-market": {
        "index_pos": 1, 
        "mapping": {
            1: "Index", 
            2: "Last", 
            4: "Q2/25", 
            5: "Q3/25", 
            6: "Q4/25", 
            7: "Q1/26"
        }
    },
    "currency": {
        "index_pos": 1,
        "mapping": {
            1: "Currency Pair",
            2: "Last",
            4: "Q2/25", 
            5: "Q3/25",
            6: "Q4/25", 
            7: "Q1/26"
        }
    },
    "wages": {
        "index_pos": 0,
        "mapping": {
            0: "Country", 
            1: "Last", 
            3: "Q2/25", 
            4: "Q3/25", 
            5: "Q4/25", 
            6: "Q1/26"
        }
    },
    "unemployment-rate": {
        "index_pos": 0,
        "mapping": {
            0: "Country", 
            1: "Last", 
            3: "Q2/25",
            4: "Q3/25",
            5: "Q4/25", 
            6: "Q1/26"
        }
    },
    "gdp": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last",
            3: "2025", 
            4: "2026", 
            5: "2027"
        }
    },
    "consumer-price-index-cpi": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last", 
            3: "Q2/25", 
            4: "Q3/25", 
            5: "Q4/25",
            6: "Q1/26"
        }
    },
    "inflation-rate": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last", 
            3: "Q2/25", 
            4: "Q3/25", 
            5: "Q4/25", 
            6: "Q1/26"
        }
    },
    "consumer-confidence": {
        "index_pos": 0, 
        "mapping": {
            0: "Country", 
            1: "Last",
            3: "Q2/25",
            4: "Q3/25", 
            5: "Q4/25",
            6: "Q1/26"
        }
    },
    "business-confidence": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last", 
            3: "Q2/25",
            4: "Q3/25",
            5: "Q4/25", 
            6: "Q1/26"
        }
    },
    "balance-of-trade": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last",
            3: "Q2/25",
            4: "Q3/25",
            5: "Q4/25",
            6: "Q1/26"
        }
    },
    "corporate-profits": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last",
            3: "Q2/25",
            4: "Q3/25", 
            5: "Q4/25",
            6: "Q1/26"
        }
    },
}


COMMODITY_CATEGORIES: Dict[str, Tuple[str, int, Dict[int, str]]] = {
    "Crude Oil": (
        "Energy", 
        0,
        {
            0: "Category",  
            1: "Price",      
            2: "Signal",
            3: "Q2/25",
            4: "Q3/25",
            5: "Q4/25",
            6: "Q1/26",
        }
    ),
    "Gold": (
        "Metals",
        0,
        {
            0: "Category",   
            1: "Signal",
            2: "Q2/25",
            3: "Q3/25",
            4: "Q4/25",
            5: "Q1/26",
        }
    ),
}


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def configure_logging(
    level: int = logging.INFO
) -> None:
    """
    Configure the root logger’s format and level.

    Parameters
    ----------
    level : int, default logging.INFO
        Logging threshold passed to `logging.basicConfig`.

    Effects
    -------
    - Sets a simple formatter: ``'%(asctime)s - %(levelname)s - %(message)s'``.
    - Affects the root logger (submodules inherit unless overridden).

    Notes
    -----
    Call this once at program start to ensure consistent log formatting across the app.
    """

    fmt = '%(asctime)s - %(levelname)s - %(message)s'
   
    logging.basicConfig(level=level, format=fmt)


def download_data(
    tickers: List[str], 
    start: dt.date, 
    end: dt.date
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols supported by Yahoo Finance. Can include international tickers (e.g., 'VOD.L').
    start : datetime.date
        Inclusive start date for the download.
    end : datetime.date
        Exclusive (Yahoo-convention) end date for the download window.

    Returns
    -------
    pd.DataFrame
        A multi-indexed DataFrame with top-level columns ['Open','High','Low','Close','Adj Close','Volume'].
        The row index is a DatetimeIndex (timezone-naive).

    Raises
    ------
    ValueError
        If the returned DataFrame is empty (e.g., invalid tickers or empty date range).

    Notes
    -----
    - `yfinance.download` returns a wide panel with one column per field per ticker.
    - Index is converted to pandas `datetime64[ns]` for downstream resampling.
    """

    logging.info("Downloading data for %d tickers: %s to %s", len(tickers), start, end)
   
    data = yf.download(tickers, start = start, end = end)
   
    if data.empty:
      
        raise ValueError("No data downloaded. Verify tickers or date range.")
   
    data.index = pd.to_datetime(data.index)
   
    return data


def scrape_forecast(
    endpoint: str, 
    index_pos: int, 
    col_map: Dict[int, str]
) -> pd.DataFrame:
    """
    Scrape a TradingEconomics forecast table for a given endpoint.

    Parameters
    ----------
    endpoint : str
        Endpoint appended to the base URL (e.g., 'interest-rate', 'gdp').
    index_pos : int
        Column index (0-based in the HTML table) used as the DataFrame index.
    col_map : dict[int, str]
        Mapping from HTML table cell index → desired column name.

    Returns
    -------
    pd.DataFrame
        Parsed table with columns named according to `col_map` and indexed by the
        column specified via `index_pos`.

    Raises
    ------
    RuntimeError
        If no <table> is found on the page.

    Implementation details
    ----------------------
    - Requests page at ``f"{BASE_FORECAST_URL}/{endpoint}"`` with a desktop UA.
    - Parses the first HTML <table> using BeautifulSoup.
    - Skips header row; for each body row, extracts cells specified by `col_map`.
    - Sets the DataFrame index using `col_map[index_pos]` and returns.

    Caveats
    -------
    - HTML shape can change; if TradingEconomics modifies the table structure,
    you may need to update `FORECAST_SPECS`.
    """
    
    url = f"{BASE_FORECAST_URL}/{endpoint}"
    
    logging.info("Scraping forecast for %s", endpoint)
    
    resp = requests.get(url, headers=HEADERS, timeout=10)
    
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    table = soup.find('table')
    
    if not table:
        
        raise RuntimeError(f"No table on page {url}")
    
    records: List[Dict[str, Any]] = []
    
    for row in table.find_all('tr')[1:]:
        
        cells = row.find_all('td')
    
        if len(cells) <= max(col_map):
           
            continue
    
        rec: Dict[str, Any] = {}
    
        for idx, name in col_map.items():
           
            rec[name] = cells[idx].get_text(strip=True)
    
        records.append(rec)
    
    df = pd.DataFrame(records)
    
    df.set_index(col_map[index_pos], inplace=True)
    
    return df


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}


def scrape_commodity_forecast(
    category: str, 
    index_pos: int, 
    col_map: Dict[int, str]
) -> pd.DataFrame:
    """
    Scrape the TradingEconomics "Commodity" page and extract a specific category table.

    Parameters
    ----------
    category : str
        Display name of the commodity category in the table header (e.g., 'Crude Oil', 'Gold').
    index_pos : int
        Column index (within the matched category table) to use as DataFrame index.
    col_map : dict[int, str]
        Mapping from column indices (0-based) to output column names.

    Returns
    -------
    pd.DataFrame
        Parsed category table with renamed columns and index set by `col_map[index_pos]`.

    Raises
    ------
    RuntimeError
        If no table with a header matching `category` is found.

    Implementation details
    ----------------------
    - Fetches ``f"{BASE_FORECAST_URL}/commodity"``.
    - Iterates over all tables; matches a table if its first <th> in <thead> equals `category`.
    - Extracts rows from <tbody>, applies `col_map`, and returns a DataFrame.
    """
  
    url = f"{BASE_FORECAST_URL}/commodity"
   
    resp = requests.get(url, headers = HEADERS, timeout = 10)
    
    resp.raise_for_status()
   
    soup = BeautifulSoup(resp.text, "html.parser")

    for tbl in soup.find_all("table"):
       
        thead = tbl.find("thead")
       
        if not thead:
          
            continue
       
        first_th = thead.find("th")
       
        if not first_th or first_th.get_text(strip = True) != category:
            
            continue

        records = []
       
        for row in tbl.find("tbody").find_all("tr"):
           
            cells = row.find_all("td")
       
            rec = {
                col_map[i]: cells[i].get_text(strip=True)
                for i in col_map
                if i < len(cells)
            }
       
            records.append(rec)

        df = pd.DataFrame(records)
        
        df.set_index(col_map[index_pos], inplace=True)
        
        return df

    raise RuntimeError(f"No '{category}' table found on commodity page.")


def get_close_series(
    data: pd.DataFrame, 
    tickers: List[str]
) -> pd.DataFrame:
    """
    Extract and clean the 'Close' price panel for selected tickers.

    Parameters
    ----------
    data : pd.DataFrame
        The full OHLCV panel returned by `download_data` (or equivalent) containing a 'Close' column block.
    tickers : list of str
        Tickers to keep in the output.

    Returns
    -------
    pd.DataFrame
        A prices DataFrame of shape (T, N_available), indexed by date, containing only
        the requested tickers that are present. Missing values are linearly interpolated
        in time per ticker.

    Side effects
    ------------
    - For UK tickers ending with '.L', values are divided by 100 to convert pence to pounds
    (Yahoo convention).

    Logging
    -------
    - Warns about any requested tickers that are not present in the 'Close' columns.

    Raises
    ------
    KeyError
        If 'Close' is not present in `data`.
    """

    try:
       
        close = data['Close'].copy()

    except KeyError:
        
        logging.exception("'Close' missing in data.")
        
        raise

    uk = [t for t in close.columns if t.endswith('.L')]

    if uk:
       
        close.loc[:, uk] = close.loc[:, uk].div(100)

    avail = [t for t in tickers if t in close.columns]

    miss = set(tickers) - set(avail)
   
    if miss:
       
        logging.warning("Missing tickers: %s", miss)
   
    return close[avail].interpolate()


def fetch_fred_series(
    symbol: str, 
    start: str, 
    end: str, 
    freq: str = 'M'
) -> pd.Series:
    """
    Fetch a FRED time series and resample to a specified frequency with forward fill.

    Parameters
    ----------
    symbol : str
        FRED series code (e.g., 'CPIAUCSL', 'GDP').
    start : str (YYYY-MM-DD)
        Start date for the download.
    end : str (YYYY-MM-DD)
        End date for the download.
    freq : str, default 'M'
        Pandas offset alias to resample to (e.g., 'M' monthly, 'W' weekly). If falsy, no resampling.

    Returns
    -------
    pd.Series
        A (potentially resampled) series indexed by period end dates, forward-filled
        after resampling to avoid gaps.

    Notes
    -----
    Uses `pandas_datareader.data.DataReader(symbol, 'fred', start, end)`.
    """

    s = web.DataReader(symbol, 'fred', start, end).squeeze()

    if freq:
       
        s = s.resample(freq).ffill()

    return s


def macro_data() -> pd.DataFrame:
    """
    Assemble a monthly macro dataset joined with S&P 500 history and returns.

    Returns
    -------
    pd.DataFrame
        A monthly panel with columns:
        - Several macro series (Inflation, US_GDP, GBP_USD, etc.)
        - 'SP500_Close' : S&P 500 month-end close
        - 'SP500_Return': Monthly percentage return (in %) computed as
                r_t = 100 * (P_t / P_{t-1} - 1)

    Process
    -------
    1) Download S&P 500 daily, resample to month-end (ME), forward-fill within month.
    2) Compute monthly % return for S&P 500:
        SP500_Return_t = 100 * (Close_t / Close_{t-1} - 1)
    3) Concurrently fetch FRED series via `ThreadPoolExecutor`.
    4) Join all series on the common monthly index, keeping only intersection rows.

    Robustness
    ----------
    - Each FRED fetch is wrapped in try/except; failures are logged and omitted.
    - Returns an "inner" join to avoid mismatched frequencies and missing data.
    """


    start, end = '2010-01-01', config.TODAY.isoformat()

    sp = yf.download('^GSPC', start = start, end = end)

    sp500_monthly = pd.DataFrame(sp["Close"].resample("ME").ffill())
  
    sp500_monthly.columns = ["SP500_Close"]
   
    sp500_monthly["SP500_Return"] = sp500_monthly["SP500_Close"].pct_change() * 100
    
    sp500_monthly.dropna(inplace = True)

    series_map = [
        ('Inflation', 'CPIAUCSL'),
        ('US_GDP', 'GDP'),
        ('GBP_USD', 'DEXUSUK'),
        ('US_Consumer_Sentiment', 'UMCSENT'),
        ('US_Unemployment_Rate', 'UNRATE'),
        ('US_Interest_Rate', 'DGS10'),
        ('US_Wages', 'AWHNONAG')
    ]
  
    results: Dict[str, pd.Series] = {}
  
    with ThreadPoolExecutor(max_workers=4) as ex:
  
        futures = {ex.submit(fetch_fred_series, sym, start, end): name
                   for name, sym in series_map}
       
        for fut in as_completed(futures):
       
            name = futures[fut]
       
            try:
       
                results[name] = fut.result()
       
            except Exception:
       
                logging.exception("Failed FRED %s", name)
  
    df = pd.DataFrame(results)
    
    macro_df = df.join(sp500_monthly, how = 'inner')
  
    return macro_df


def get_sector_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute realised and exponentially-weighted expected stats for sector indices.

    Returns
    -------
    (df, sec_close) : (pd.DataFrame, pd.DataFrame)
        - df: index 'Sector' with columns
        
            * 'Returns'            : Cumulative return over the last year window
                                    R_ann = ∏_{t ∈ last_year} (1 + r_t) - 1
          
            * 'Volatility'         : Annualised volatility using full-sample daily std
                                    scaled by √(T_last_year):
                                        σ_ann = std_full * √T_last_year
          
            * 'Sharpe Ratio'       : (R_ann - RF) / σ_ann
          
            * 'Exp Returns'        : EWM mean of last-year daily returns scaled to 1y
                                    Let T_y = #obs in last year.
                                    Using pandas EWM with halflife h = 0.1·T_y:
                                        μ̂_EWM = EWM_mean_last_year[-1] · T_y
          
            * 'Exp Volatility'     : EWM std of last-year daily returns, annualised
                                        σ̂_EWM = EWM_std_last_year[-1] · √T_y
          
            * 'Exp Sharpe Ratio'   : (μ̂_EWM - RF) / σ̂_EWM
        
        - sec_close: sector close-price panel aligned to `sec_map` names.

    Method
    ------
    - Download sector ETFs/indices given by `sec_map` for 5 years, compute daily returns.
    - Restrict a 1-year window (by date) to form “last-year” aggregates (T_last_year points).
    - EWM uses pandas' halflife parameter: with halflife h, decay factor λ = 0.5^{1/h}.
    The effective weight for an observation k steps in the past is λ^k.

    Notes
    -----
    - `RF` is expected to be an annualised risk-free rate in `config.RF`.
    - Volatility uses `rets.std()` over the *entire* sample, then scales by √(T_last_year),
    matching the implementation here.
    """
    
    sec_data = yf.download(list(sec_map.keys()), start = config.FIVE_YEAR_AGO, end = config.TODAY)['Close']
    
    sec_data = sec_data.rename(columns = sec_map)
    
    rets = sec_data.pct_change().dropna()
        
    rets_last_year = rets.loc[rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_len = len(rets_last_year)
    
    rets_ann = (1 + rets_last_year).prod() - 1
    
    vol = rets.std() * np.sqrt(rets_last_year_len)
    
    sr = (rets_ann - config.RF) / vol
    
    exp_ret_ind = rets_last_year.ewm(halflife = 0.1 * rets_last_year_len, adjust = False).mean().iloc[-1] * rets_last_year_len
   
    exp_std_ind = rets_last_year.ewm(halflife = 0.1 * rets_last_year_len, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_len)
   
    exp_sr_ind = (exp_ret_ind - config.RF) / exp_std_ind
    
    df = pd.DataFrame({
        "Sector": sec_data.columns,
        "Returns": rets_ann,
        "Volatility": vol,
        "Sharpe Ratio": sr,
        "Exp Returns": exp_ret_ind,
        "Exp Volatility": exp_std_ind,
        "Exp Sharpe Ratio": exp_sr_ind
    }).set_index("Sector")
    
    return df, sec_data


def get_industry_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build industry-level aggregates and compute realised/expected statistics.

    Returns
    -------
    (df, ind_close) : (pd.DataFrame, pd.DataFrame)
      
        - df indexed by 'Industry' with columns:
      
            * 'Returns'            : Last-year cumulative return
      
            * 'Volatility'         : Annualised volatility ≈ std_full · √T_last_year
      
            * 'Sharpe Ratio'       : (R_ann - RF) / σ_ann
      
            * 'Exp Returns'        : EWM mean (last year) × T_last_year
      
            * 'Exp Volatility'     : EWM std (last year) × √T_last_year
      
            * 'Exp Sharpe Ratio'   : (μ̂_EWM - RF) / σ̂_EWM
      
        - ind_close: industry close panel after mapping/grouping.

    Procedure
    ---------
   
    1) Convert an industry universe (`IndustryMap` and `yf.Industry`) into a
    ticker→industry mapping, download closes for 5y.
   
    2) Rename columns twice:
    - to the raw industry names from `ind_ticker_map`
    - to standardised names via `IndustryMap`
   
    3) Aggregate columns by the first level (group names), using column sum as the group proxy.
   
    4) Compute daily returns; slice the last year and compute statistics:
        R_ann = ∏ (1 + r_t) - 1
        σ_ann ≈ std_full · √(T_last_year)
        Sharpe = (R_ann - RF) / σ_ann
   
    5) EWM expected moments use halflife = 0.1·T_last_year, as in sectors.

    Notes
    -----
    - Any gaps or NaNs from download are implicitly handled by pandas;
    downstream `.dropna()` removes non-overlapping rows in returns.
    """

    ind_list = list(IndustryMap.keys())
    
    ind_ticker_map = {}
    
    for ind in ind_list:
    
        sec_data = yf.Industry(ind).ticker.ticker
    
        ind_ticker_map[sec_data] = ind
    
    ind_ticker_map

    ind_data = yf.download(list(ind_ticker_map.keys()), start = config.FIVE_YEAR_AGO, end = config.TODAY)['Close']
    
    ind_data = ind_data.rename(columns = ind_ticker_map)

    ind_data = ind_data.rename(columns = IndustryMap)
    
    ind_data = ind_data.groupby(axis = 1, level=0).sum()

    rets_ind = ind_data.pct_change().dropna()
        
    rets_last_year_ind = rets_ind.loc[rets_ind.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_ind_len = len(rets_last_year_ind)
    
    rets_ann_ind = (1 + rets_last_year_ind).prod() - 1        
    
    vol_ind = rets_ind.std() * np.sqrt(rets_last_year_ind_len)
    
    sr_ind = ((rets_ann_ind - config.RF) / vol_ind).fillna(0)
    
    exp_ret_ind = rets_last_year_ind.ewm(halflife = 0.1 * rets_last_year_ind_len, adjust = False).mean().iloc[-1] * rets_last_year_ind_len
   
    exp_std_ind = rets_last_year_ind.ewm(halflife = 0.1 * rets_last_year_ind_len, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_ind_len)
   
    exp_sr_ind = (exp_ret_ind - config.RF) / exp_std_ind
    
    df = pd.DataFrame({
        "Industry": ind_data.columns,
        "Returns": rets_ann_ind,
        "Volatility": vol_ind,
        "Sharpe Ratio": sr_ind,
        "Exp Returns": exp_ret_ind,
        "Exp Volatility": exp_std_ind,
        "Exp Sharpe Ratio": exp_sr_ind
    }).set_index("Industry")
    
    return df, ind_data


def get_factor_etfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute stats for a set of factor ETFs (Value, Quality, Momentum, MinVol, Size).

    Returns
    -------
    (df, rets) : (pd.DataFrame, pd.DataFrame)
        - df indexed by 'Sector' (ETF tickers) with the same columns as sector stats:
            * 'Returns', 'Volatility', 'Sharpe Ratio',
            'Exp Returns', 'Exp Volatility', 'Exp Sharpe Ratio'
        - rets : daily returns for the factor ETFs.

    Method
    ------
    
    - Download close prices for ['VLUE','QUAL','MTUM','USMV','SIZE'] since 2000-01-01.
    
    - Compute daily returns, and slice a last-year window (length T_y).
    
    - Annualised measures follow:
        R_ann = ∏_{t∈Y} (1 + r_t) - 1
        σ_ann = std_full · √T_y
        Sharpe = (R_ann - RF) / σ_ann
    
    - EWM expected moments use halflife h = 0.1·T_y:
        μ̂_EWM = EWM_mean_last_year[-1] · T_y
        σ̂_EWM = EWM_std_last_year[-1] · √T_y
        Sharpe_EWM = (μ̂_EWM - RF) / σ̂_EWM
    """
        
    tickers = ['VLUE', 'QUAL', 'MTUM', 'USMV', 'SIZE']
    
    fac_data = yf.download(tickers, start = '2000-01-01', end = config.TODAY)['Close']
        
    rets = fac_data.pct_change().dropna()
        
    rets_last_year = rets.loc[rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_len = len(rets_last_year)
    
    rets_ann = (1 + rets_last_year).prod() - 1
    
    vol = rets.std() * np.sqrt(rets_last_year_len)
    
    sr = (rets_ann - config.RF) / vol
    
    exp_ret_ind = rets_last_year.ewm(halflife = 0.1 * rets_last_year_len, adjust = False).mean().iloc[-1] * rets_last_year_len
   
    exp_std_ind = rets_last_year.ewm(halflife = 0.1 * rets_last_year_len, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_len)
   
    exp_sr_ind = (exp_ret_ind - config.RF) / exp_std_ind
    
    df = pd.DataFrame({
        "Sector": fac_data.columns,
        "Returns": rets_ann,
        "Volatility": vol,
        "Sharpe Ratio": sr,
        "Exp Returns": exp_ret_ind,
        "Exp Volatility": exp_std_ind,
        "Exp Sharpe Ratio": exp_sr_ind
    }).set_index("Sector")
    
    return df, rets
    

def get_index_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute stats for a set of global equity indices.

    Returns
    -------
    (df, index_close) : (pd.DataFrame, pd.DataFrame)
        - df indexed by 'Sector' (index tickers) with columns:
            * 'Returns', 'Volatility', 'Sharpe Ratio',
            'Exp Returns', 'Exp Volatility', 'Exp Sharpe Ratio'
        - index_close : close-price panel for the selected indices.

    Universe
    --------
    ['^GSPC', '^NDX', '^FTSE', '^GDAXI', '^FCHI', '^AEX', '^IBEX', '^GSPTSE', '^HSI', '^SSMI', '^IXIC']

    Formulas
    --------
    Let Y be the last-year window with T_y observations and r_t daily returns.

    - Cumulative return:
        R_ann = ∏_{t∈Y} (1 + r_t) - 1
   
    - Annualised volatility (as implemented):
        σ_ann = std_full · √T_y
   
    - Sharpe:
        Sharpe = (R_ann - RF) / σ_ann
   
    - EWM expected moments (halflife = 0.1·T_y):
        μ̂_EWM = EWM_mean_last_year[-1] · T_y
        σ̂_EWM = EWM_std_last_year[-1] · √T_y
        Sharpe_EWM = (μ̂_EWM - RF) / σ̂_EWM
    """
   
    tickers = ['^GSPC', '^NDX', '^FTSE', '^GDAXI', '^FCHI', '^AEX', '^IBEX', '^GSPTSE', '^HSI', '^SSMI', '^IXIC']
    
    index_data = yf.download(tickers, start = config.FIVE_YEAR_AGO, end = config.TODAY)['Close']
        
    rets = index_data.pct_change().dropna()
        
    rets_last_year = rets.loc[rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_len = len(rets_last_year)
    
    rets_ann = (1 + rets_last_year).prod() - 1
    
    vol = rets.std() * np.sqrt(rets_last_year_len)
    
    sr = (rets_ann - config.RF) / vol
    
    exp_ret_ind = rets_last_year.ewm(halflife = 0.1 * rets_last_year_len, adjust = False).mean().iloc[-1] * rets_last_year_len
   
    exp_std_ind = rets_last_year.ewm(halflife = 0.1 * rets_last_year_len, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_len)
   
    exp_sr_ind = (exp_ret_ind - config.RF) / exp_std_ind
    
    df = pd.DataFrame({
        "Sector": index_data.columns,
        "Returns": rets_ann,
        "Volatility": vol,
        "Sharpe Ratio": sr,
        "Exp Returns": exp_ret_ind,
        "Exp Volatility": exp_std_ind,
        "Exp Sharpe Ratio": exp_sr_ind
    }).set_index("Sector")
    
    return df, index_data


def main() -> None:
    """
    End-to-end pipeline:
    - Market data download
    - Price/return transforms
    - Macro and forecast scraping
    - Cross-section stats (sectors, industries, factors, indices)
    - Single-pass Excel export

    Steps
    -----
    1) **Download & panels**
    - Download OHLCV for `config.tickers` from '2000-01-01' to `config.TODAY`.
    - Extract 'Close' panel (UK '.L' scaled by 1/100) and interpolate gaps.
    - Construct `high`, `low`, `volume` panels.

    2) **Transforms**
    - Weekly close:  last value per week.
    - Daily returns:  r_t = Close_t / Close_{t-1} - 1
    - Weekly returns: analogous on weekly_close.

    **EMA volatilities**
    - Daily EMA std:  `ema_vol = daily_ret.ewm(span=252).std()`
    - Weekly EMA std: `weekly_ema_vol = weekly_ret.ewm(span=52).std()`

    Pandas EWM with `span` uses α = 2/(span+1), i.e., weights w_k ∝ (1-α)^k.

    3) **Forecast scraping**
    - For each entry in `FORECAST_SPECS`, fetch a TradingEconomics forecast table
        using `scrape_forecast`. Store in `forecasts` under a readable sheet name.
    - For each commodity spec in `COMMODITY_CATEGORIES`, scrape the commodity page
        via `scrape_commodity_forecast`.

    4) **Macro panel**
    - Build monthly macro dataset with S&P500 close and monthly % returns
        (in percent):  100·(P_t/P_{t-1}−1).

    5) **Cross-section summaries**
    - Sectors: `get_sector_data()`
    - Industries: `get_industry_data()`
    - Factor ETFs: `get_factor_etfs()`
    - Indices: `get_index_data()`

    6) **EMA expected returns (PyPortfolioOpt)**
    - On daily closes (1Y window length `l`):
        `expected_returns.ema_historical_return(close, span=l, frequency=l, compounding=True)`
        This computes an exponentially weighted mean return and **annualises** using `frequency=l`.
    - On weekly closes (1Y weekly length `L_w` ≈ ½ * len(weekly_1y) here):
        `expected_returns.ema_historical_return(weekly_close, span=0.5*L_w, frequency=0.5*L_w, compounding=True)`

    Notes:
    - With compounding=True, PyPortfolioOpt converts mean log/linear returns into an annualised
        compounded expectation consistent with `frequency`.
    - Latest EWM volatilities are reported as:
        SE_daily = ema_vol.iloc[-1] · √252
        SE_weekly = weekly_ema_vol.iloc[-1]

    7) **Assemble Excel outputs**
    - Core workbook (`config.DATA_FILE`) includes prices, returns, weekly panels,
        forecasts, macro, and close panels.
    - Forecast workbook (`config.FORECAST_FILE`) includes:
        "Exponential Returns" (EMA expected returns & vol),
        "Daily Returns" (1Y raw returns & std),
        and the cross-section summary sheets.

    I/O
    ---
    - Uses `export_results(...)` to write each workbook in a single pass.
    - Logging provides progress and error messages for scraping and data fetching.

    Caveats
    -------
    - External data sources (Yahoo, TradingEconomics, FRED) are subject to outages and schema changes.
    - Some annualisation conventions in the cross-section summaries follow the code
    exactly (e.g., std_full × √T_last_year).
    """

   
    configure_logging()

    tickers = config.tickers
    
    start_date = '2000-01-01'  
    
    data = download_data(
        tickers = tickers, 
        start = start_date, 
        end = config.TODAY
    )
    
    close = get_close_series(
        data = data, 
        tickers = tickers
    )
    
    high = data['High']
    
    low = data['Low']
        
    volume = data['Volume']
   
    weekly_close = close.resample('W').last()
   
    daily_ret = close.pct_change()
    
    weekly_ret = weekly_close.pct_change()
   
    ema_vol = daily_ret.ewm(span = 252, adjust = False).std()
   
    weekly_ema_vol = weekly_ret.ewm(span = 52, adjust = False).std()
   
    forecasts: Dict[str, pd.DataFrame] = {}
   
    for ep, spec in FORECAST_SPECS.items():
       
        try:
       
            df = scrape_forecast(
                endpoint = ep, 
                index_pos = spec['index_pos'], 
                col_map = spec['mapping']
            )
       
            forecasts[ep.replace('-', '_').title()] = df
       
        except Exception:
       
            logging.exception("Forecast failed: %s", ep)

    for key, (cat, idx, cmap) in COMMODITY_CATEGORIES.items():
       
        try:
       
            df = scrape_commodity_forecast(
                endpoint = cat, 
                index_pos = idx, 
                col_map = cmap
            )
            
            forecasts[key.replace(' ', '_')] = df
        
        except Exception:
        
            logging.exception("Commodity forecast failed: %s", key)
   
    macro_df = macro_data()
    
    sector_data, sec_close = get_sector_data()
    
    industry_data, ind_close = get_industry_data()
    
    factor_etfs, factor_rets = get_factor_etfs()
    
    index_data, index_close = get_index_data()
   
    sheets_data = {
        "Close": close,
        "High": high,
        "Low": low,
        "Volume": volume,
        "Weekly Close": weekly_close,
        "Historic Returns": daily_ret,
        "Historic Weekly Returns": weekly_ret,
        **forecasts,
        "Macro Data": macro_df,
        "Index Close": index_close,
        "Sector Close": sec_close,
        "Industry Close": ind_close,
        "Factor Returns": factor_rets,
    }
   
    close_1y = close.loc[close.index >= pd.to_datetime(config.YEAR_AGO)]
    
    weekly_close_1y = weekly_close.loc[weekly_close.index >= pd.to_datetime(config.YEAR_AGO)]
   
    l = len(close_1y)
   
    ret_1y = close_1y.pct_change().dropna()
   
    exp_ret_ema = expected_returns.ema_historical_return(
        close, 
        span = l, 
        frequency = l, 
        compounding = True, 
        returns_data = False
    )
   
    exp_ret_week = expected_returns.ema_historical_return(
        weekly_close, 
        span = (0.5 * len(weekly_close_1y)), 
        frequency = (0.5 * len(weekly_close_1y)), 
        compounding = True
    )
   
    latest_ema_vol = ema_vol.iloc[-1]
    
    latest_wk_vol = weekly_ema_vol.iloc[-1]
   
    one_year_raw = close_1y.iloc[-1] / close_1y.iloc[0] - 1

    er_ema_df = pd.DataFrame({
        "Returns": exp_ret_ema,
        "EMA Weekly Returns": exp_ret_week,
        "SE": latest_ema_vol * np.sqrt(252),
        "Weekly SE": latest_wk_vol
    })
   
    dr_df = pd.DataFrame({
        "Returns": one_year_raw,
        "SE": ret_1y.std() * np.sqrt(len(ret_1y))
    })

    export_results(
        sheets = sheets_data, 
        output_excel_file = config.DATA_FILE,
        formatting = False
    )
    
    forecast_sheets = {
        "Exponential Returns": er_ema_df, 
        "Daily Returns": dr_df,
        "Sector Data": sector_data,
        "Industry Data": industry_data,
        "Factor ETFs": factor_etfs,
        "Index Data": index_data
    }
   
    export_results(
        sheets = forecast_sheets,
        output_excel_file = config.FORECAST_FILE
    )


if __name__ == '__main__':
    
    main()
  
  
