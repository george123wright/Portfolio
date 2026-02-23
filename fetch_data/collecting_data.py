"""
Downloads historical market data, computes technical indicators, scrapes Trading Economics forecasts and exports all series to Excel.
"""

import datetime as dt
import logging
from typing import List, Dict, Any, Tuple, Iterable, Optional
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
from maps.currency_mapping import country_to_ccy, country_to_pair
from maps.ccy_exchange_map import _CCY_BY_SUFFIX

BASE_FORECAST_URL = "https://tradingeconomics.com/forecast"


FORECAST_SPECS: Dict[str, Dict[str, Any]] = {
    "interest-rate": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last", 
            3: "Q3/25", 
            4: "Q4/25", 
            5: "Q1/26", 
            6: "Q2/26"
        }
    },
    "stock-market": {
        "index_pos": 1, 
        "mapping": {
            1: "Index", 
            2: "Last", 
            4: "Q3/25", 
            5: "Q4/25", 
            6: "Q1/26", 
            7: "Q2/26"
        }
    },
    "currency": {
        "index_pos": 1,
        "mapping": {
            1: "Currency Pair",
            2: "Last",
            4: "Q3/25", 
            5: "Q4/25",
            6: "Q1/26", 
            7: "Q2/26"
        }
    },
    "wages": {
        "index_pos": 0,
        "mapping": {
            0: "Country", 
            1: "Last", 
            3: "Q3/25", 
            4: "Q4/25", 
            5: "Q1/26", 
            6: "Q2/26"
        }
    },
    "unemployment-rate": {
        "index_pos": 0,
        "mapping": {
            0: "Country", 
            1: "Last", 
            3: "Q3/25",
            4: "Q4/25",
            5: "Q1/26", 
            6: "Q2/26"
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
            3: "Q3/25", 
            4: "Q4/25", 
            5: "Q1/26",
            6: "Q2/26"
        }
    },
    "inflation-rate": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last", 
            3: "Q3/25", 
            4: "Q4/25", 
            5: "Q1/26", 
            6: "Q2/26"
        }
    },
    "consumer-confidence": {
        "index_pos": 0, 
        "mapping": {
            0: "Country", 
            1: "Last",
            3: "Q3/25",
            4: "Q4/25", 
            5: "Q1/26",
            6: "Q2/26"
        }
    },
    "business-confidence": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last", 
            3: "Q3/25",
            4: "Q4/25",
            5: "Q1/26", 
            6: "Q2/26"
        }
    },
    "balance-of-trade": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last",
            3: "Q3/25",
            4: "Q4/25",
            5: "Q1/26",
            6: "Q2/26"
        }
    }, 
    "corporate-profits": {
        "index_pos": 0, 
        "mapping": {
            0: "Country",
            1: "Last",
            3: "Q3/25",
            4: "Q4/25", 
            5: "Q1/26",
            6: "Q2/26"
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
            3: "Q3/25",
            4: "Q4/25",
            5: "Q1/26",
            6: "Q2/26",
        }
    ),
    "Gold": (
        "Metals",
        0,
        {
            0: "Category",   
            1: "Signal",
            2: "Q3/25",
            3: "Q4/25",
            4: "Q1/26",
            5: "Q2/26",
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
   
    logging.basicConfig(level = level, format = fmt)


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
        A multi-indexed DataFrame with top-level columns ['Open','High','Low','Close','Volume'].
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
   
    data = yf.download(tickers, start = start, end = end, auto_adjust = True)
   
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
    
    resp = requests.get(url, headers=HEADERS, timeout = 10)
    
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
           
            rec[name] = cells[idx].get_text(strip = True)
    
        records.append(rec)
    
    df = pd.DataFrame(records)
    
    df.set_index(col_map[index_pos], inplace = True)
    
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
                col_map[i]: cells[i].get_text(strip = True)
                for i in col_map
                if i < len(cells)
            }
       
            records.append(rec)

        df = pd.DataFrame(records)
        
        df.set_index(col_map[index_pos], inplace = True)
        
        return df

    raise RuntimeError(f"No '{category}' table found on commodity page.")


def _ensure_ends_at_today(
    df: pd.DataFrame,
    today: Any,
    *,
    is_returns: bool = False,
) -> pd.DataFrame:
    """
    Ensure df has a row at `today`. For level series, forward-fill.
    For return series, set NaN in the last row to 0.

    - Trims any rows after `today`.
  
    - If `today` is missing, adds it.
    """
  
    if df.empty:
  
        return df

    today = pd.to_datetime(today)
  
    out = df.copy()
  
    out = out.sort_index()
  
    out = out.loc[out.index <= today]

    if today not in out.index:
  
        out = out.reindex(out.index.union([today])).sort_index()

    if is_returns:

        out.loc[today] = out.loc[today].fillna(0.0)

    else:

        out = out.ffill()

    return out


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

    uk_all = [t for t in close.columns if isinstance(t, str) and t.endswith('.L')]

    if uk_all:

        close = _fix_london_unit_flips_df(
            df = close,
            uk_tickers = uk_all,
            factor = 100.0,
            max_jump = 20.0,
        )

        close.loc[:, uk_all] = close.loc[:, uk_all].div(100)

    avail = [t for t in tickers if t in close.columns]

    miss = set(tickers) - set(avail)

    if miss:

        logging.warning("Missing tickers: %s", miss)

    close = close[avail]

    return close.interpolate()


def _fix_london_unit_flips_series(
    s: pd.Series,
    *,
    factor: float = 100.0,
    max_jump: float = 20.0,
) -> pd.Series:
    """
    Detect and repair 100x GBp/GBP unit flips in a single price series.

    Logic
    -----
    Yahoo / yfinance sometimes mix GBp (pence) and GBP (pounds) for .L tickers,
    so a single day can appear ~100x too large or too small relative to the
    previous close.

    This routine walks the series and, for each t > 0:

        ratio_t = price_t / price_{t-1}

    If ratio_t is implausibly large or small (ratio_t > max_jump or
    ratio_t < 1 / max_jump), it tries two candidate rescalings of price_t:

        cand_down = price_t / factor      # assume this point is 100x too big
        cand_up   = price_t * factor      # assume this point is 100x too small

    For each candidate it computes the *new* ratio to the previous day:

        cand_ratio = cand / price_{t-1}

    and keeps the candidate (including the original value) whose
    cand_ratio lies in [1/max_jump, max_jump] and is *closest to 1*.
    That way:

    - A 100x drop (ratio ~ 0.01) will typically be fixed by *multiplying*
      that point by 100, bringing the ratio back to ~1–2.

    - A 100x spike (ratio ~ 100) will typically be fixed by *dividing*
      that point by 100.

    Parameters
    ----------
    s : pd.Series
        Price series (one ticker), in whatever units yfinance returned.
    factor : float, default 100.0
        Suspected unit conversion factor between GBp and GBP.
    max_jump : float, default 20.0
        Threshold for "implausible" one-day moves. Moves larger than
        max_jump (or smaller than 1/max_jump) in absolute ratio are
        treated as likely unit errors rather than genuine price action.

    Returns
    -------
    pd.Series
        Series with suspicious 100x jumps/drops rescaled in-place.

    Notes
    -----
    - This assumes genuine 100-for-1 stock splits are essentially absent.
      If such a split did occur, this filter would undo it, so for very
      exotic corporate actions you may want to whitelist specific dates.
    """
   
    s = s.copy()
   
    vals = s.values
   
    n = len(vals)

    for i in range(1, n):
   
        prev = vals[i - 1]
   
        cur = vals[i]

        if not np.isfinite(prev) or not np.isfinite(cur) or prev <= 0 or cur <= 0:
   
            continue

        ratio = cur / prev
   
        if ratio <= 0:
   
            continue

        if ratio > max_jump or ratio < 1.0 / max_jump:

            best_val = cur

            best_score = abs(np.log(ratio))

            cand_down = cur / factor

            cand_ratio_down = cand_down / prev

            if 1.0 / max_jump <= cand_ratio_down <= max_jump:

                score_down = abs(np.log(cand_ratio_down))

                if score_down < best_score:

                    best_score = score_down

                    best_val = cand_down

            cand_up = cur * factor

            cand_ratio_up = cand_up / prev

            if 1.0 / max_jump <= cand_ratio_up <= max_jump:

                score_up = abs(np.log(cand_ratio_up))

                if score_up < best_score:

                    best_score = score_up

                    best_val = cand_up

            if best_val != cur:

                logging.warning(
                    "Fixed suspected GBp/GBP flip for %s at %s: %.6f -> %.6f "
                    "(prev=%.6f, ratio=%.4f)",
                    getattr(s, "name", "<unknown>"),
                    s.index[i],
                    cur,
                    best_val,
                    prev,
                    ratio,
                )
                
                vals[i] = best_val

    return pd.Series(vals, index = s.index, name = s.name)


def _fix_london_unit_flips_df(
    df: pd.DataFrame,
    uk_tickers: Iterable[str],
    *,
    factor: float = 100.0,
    max_jump: float = 20.0,
) -> pd.DataFrame:
    """
    Apply `_fix_london_unit_flips_series` column-wise to a price panel.

    Parameters
    ----------
    df : pd.DataFrame
        Wide price panel (one field, many tickers).
    uk_tickers : iterable of str
        Tickers ending with '.L' that should be checked for GBp/GBP flips.
        Only those present as columns in df are processed.
    factor : float, default 100.0
        Unit conversion factor between GBp and GBP.
    max_jump : float, default 20.0
        Threshold forwarded to `_fix_london_unit_flips_series`.

    Returns
    -------
    pd.DataFrame
        Copy of df with any 100x unit flips repaired for `.L` tickers.
    """
   
    out = df.copy()
   
    cols = [t for t in uk_tickers if t in out.columns]

    for col in cols:
       
        out[col] = _fix_london_unit_flips_series(
            s = out[col],
            factor = factor, 
            max_jump = max_jump
        )

    return out


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

    start = '2010-01-01'
    
    end = config.TODAY.isoformat()

    rate_tickers = ['^TNX', '^IRX']
    
    other_tickers = ['CL=F', 'GC=F', 'HG=F', '^VIX']
    
    macro_tickers = rate_tickers + other_tickers
    
    macro = yf.download(macro_tickers, start = start, end = end, auto_adjust = True)
    
    rates = macro['Close'][rate_tickers].ffill().bfill() / 100
    
    macro_ret = macro['Close'][other_tickers].pct_change().ffill().bfill()
    
    yh_df = macro_ret.join(rates, how = 'outer')

    series_map = [
        ('Inflation', 'T5YIE'),
        ('US_GDP', 'GDP'),
        ('GBP_USD', 'DEXUSUK'),
        ('US_Consumer_Sentiment', 'UMCSENT'),
        ('US_Unemployment_Rate', 'UNRATE'),
        ('US_Wages', 'CES0500000003')
    ]
  
    results: Dict[str, pd.Series] = {}
    
    growth_names = ['US_GDP', 'US_Consumer_Sentiment', 'US_Wages']
  
    with ThreadPoolExecutor(max_workers = 4) as ex:
  
        futures = {ex.submit(fetch_fred_series, sym, start, end): name
                   for name, sym in series_map}
       
        for fut in as_completed(futures):
       
            name = futures[fut]
            
            if name in growth_names:
                
                try:
                
                    results[name] = fut.result().pct_change().ffill()
                    
                except Exception:
               
                    logging.exception("Failed FRED %s", name)

            else:
                
                try:
        
                    results[name] = fut.result()
        
                except Exception:
        
                    logging.exception("Failed FRED %s", name)
  
    df = pd.DataFrame(results)
    
    macro_df = df.join(yh_df, how = 'outer').ffill().bfill()

    macro_df = _ensure_ends_at_today(
        df = macro_df, 
        today = config.TODAY, 
        is_returns = False
    )
  
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
    
    sec_data = yf.download(list(sec_map.keys()), start = config.FIVE_YEAR_AGO, end = config.TODAY, auto_adjust = True)['Close']
    
    sec_data = sec_data.rename(columns = sec_map)
    
    sec_data = sec_data.ffill().bfill()
    
    rets = sec_data.pct_change().dropna()
        
    rets_last_year = rets.loc[rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_len = len(rets_last_year)
    
    rets_ann = (1 + rets_last_year).prod() - 1
    
    vol = rets.std() * np.sqrt(rets_last_year_len)
    
    sr = (rets_ann - config.RF) / vol
    
    hl = 0.1 * rets_last_year_len
    
    exp_ret_ind = rets_last_year.ewm(halflife = hl, adjust = False).mean().iloc[-1] * rets_last_year_len
   
    exp_std_ind = rets_last_year.ewm(halflife = hl, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_len)
   
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
    
    sec_close = _ensure_ends_at_today(
        df = sec_data, 
        today = config.TODAY, 
        is_returns = False
    )
    
    return df, sec_close


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
    
    ind_data = yf.download(list(ind_ticker_map.keys()), start = config.FIVE_YEAR_AGO, end = config.TODAY, auto_adjust = True)['Close']
    
    ind_data = ind_data.ffill().bfill()
    
    ind_data = ind_data.rename(columns = ind_ticker_map)

    ind_data = ind_data.rename(columns = IndustryMap)
    
    ind_data = ind_data.groupby(axis = 1, level = 0).sum()

    rets_ind = ind_data.pct_change().dropna()
        
    rets_last_year_ind = rets_ind.loc[rets_ind.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_ind_len = len(rets_last_year_ind)
    
    rets_ann_ind = (1 + rets_last_year_ind).prod() - 1        
    
    vol_ind = rets_ind.std() * np.sqrt(rets_last_year_ind_len)
    
    sr_ind = ((rets_ann_ind - config.RF) / vol_ind).fillna(0)
    
    hl = 0.1 * rets_last_year_ind_len
    
    exp_ret_ind = rets_last_year_ind.ewm(halflife = hl, adjust = False).mean().iloc[-1] * rets_last_year_ind_len
   
    exp_std_ind = rets_last_year_ind.ewm(halflife = hl, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_ind_len)
   
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
    
    ind_close = _ensure_ends_at_today(
        df = ind_data, 
        today = config.TODAY,
        is_returns = False
    )
    
    return df, ind_close


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
    
    fac_data = yf.download(tickers, start = '2000-01-01', end = config.TODAY, auto_adjust = True)['Close']
    
    fac_data = fac_data.ffill().bfill()
        
    rets = fac_data.pct_change().dropna()
        
    rets_last_year = rets.loc[rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_len = len(rets_last_year)
    
    rets_ann = (1 + rets_last_year).prod() - 1
    
    vol = rets.std() * np.sqrt(rets_last_year_len)
    
    sr = (rets_ann - config.RF) / vol
    
    hl = 0.1 * rets_last_year_len
    
    exp_ret_ind = rets_last_year.ewm(halflife = hl, adjust = False).mean().iloc[-1] * rets_last_year_len
   
    exp_std_ind = rets_last_year.ewm(halflife = hl, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_len)
   
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
    

def get_factor_etfs_all_regions(
    regions: Optional[Iterable[str]] = None,
    start: str = "2000-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute stats for factor ETFs (Value, Quality, Momentum, MinVol, Size) across regions.
    Regions available in this mapping: {"US", "EU", "EM"}.

    Returns
    -------
    (df_all, rets_all)
        df_all : pd.DataFrame (index: MultiIndex[Region, Ticker])
            Columns: 'Factor','Returns','Volatility','Sharpe Ratio',
                     'Exp Returns','Exp Volatility','Exp Sharpe Ratio'
        rets_all : pd.DataFrame
            Daily returns with MultiIndex columns (Region, Ticker).
    """

    FACTOR_TICKERS = {
        "US": ["VLUE", "QUAL", "MTUM", "USMV", "SIZE"],
        "EU": ["IEVL.MI", "IEQU.MI", "IEMO.MI", "MVEU.MI", "IEFS.L"],
        "EM": ["EMVL.L", "IWQU.L", "EEMO", "EEMV", "IWSZ.L"],
    }
   
    FACTOR_NAMES = ["Value", "Quality", "Momentum", "MinVol", "Size"]

    if regions is None:
   
        regions = list(FACTOR_TICKERS.keys())
   
    else:
   
        regions = [r.upper() for r in regions if r.upper() in FACTOR_TICKERS]

    all_stats: list[pd.DataFrame] = []
   
    rets_blocks: list[pd.DataFrame] = []

    year_ago = pd.to_datetime(getattr(config, "YEAR_AGO"))
   
    rf_annual = float(getattr(config, "RF"))
   
    end_date = getattr(config, "TODAY")

    for region in regions:
   
        tickers = FACTOR_TICKERS.get(region, [])
   
        if not tickers:
   
            continue

        px = yf.download(tickers, start = start, end = end_date, progress = False, auto_adjust = True)

        if isinstance(px, pd.DataFrame) and "Close" in px.columns:

            px = px["Close"]

        elif isinstance(px, pd.Series):

            px = px.to_frame(name = tickers[0])

        if px is None or px.empty:

            continue
        
        uk_cols = [c for c in px.columns if isinstance(c, str) and c.endswith(".L")]
        
        if uk_cols:
            
            px = _fix_london_unit_flips_df(
                df = px, 
                uk_tickers = uk_cols
            )
           
            px.loc[:, uk_cols] = px.loc[:, uk_cols].div(100)

        idx = pd.to_datetime(px.index, errors = "coerce")
        
        px = px[~idx.isna()].copy()
        
        px.index = pd.to_datetime(px.index).tz_localize(None)
        
        px = px.sort_index()

        if px.empty:

            continue
        
        px = px.ffill().bfill()

        px = px.dropna(how = "all")
        
        if px.empty:
         
            continue

        rets = px.pct_change()

        keep = rets.notna().any()

        rets = rets.loc[:, keep]

        if rets.empty:

            continue

        rets = rets.astype(float)

        rets = rets.dropna(how = "all")

        rets_y = rets.loc[rets.index >= year_ago]

        if rets_y.empty:

            rets_y = rets.tail(252)

        if rets_y.empty:

            continue

        T = len(rets_y)  

        R_ann = (1.0 + rets_y).prod() - 1.0
        
        vol_ann = rets_y.std() * np.sqrt(T)  
        
        sr = (R_ann - rf_annual) / vol_ann.replace(0.0, np.nan)

        hl = max(int(0.1 * T), 1)

        mu_EWM = rets_y.ewm(halflife = hl, adjust = False).mean().iloc[-1] * T

        sig_EWM = rets_y.ewm(halflife = hl, adjust = False).std().iloc[-1] * np.sqrt(T)

        sr_EWM = (mu_EWM - rf_annual) / sig_EWM.replace(0.0, np.nan)

        cols = list(rets.columns)

        factor_labels = (FACTOR_NAMES + ["Other"] * max(0, len(cols) - len(FACTOR_NAMES)))[: len(cols)]

        factor_map = dict(zip(cols, factor_labels))

        df_region = pd.DataFrame(
            {
                "Factor": [factor_map[c] for c in cols],
                "Region": region,
                "Returns": R_ann.reindex(cols),
                "Volatility": vol_ann.reindex(cols),
                "Sharpe Ratio": sr.reindex(cols),
                "Exp Returns": mu_EWM.reindex(cols),
                "Exp Volatility": sig_EWM.reindex(cols),
                "Exp Sharpe Ratio": sr_EWM.reindex(cols),
            },
            index = pd.Index(cols, name = "Ticker"),
        ).sort_index()

        all_stats.append(df_region)

        rets_block = rets.copy()

        rets_blocks.append(rets_block)

    if not all_stats:
        
        raise ValueError("No data retrieved for the requested regions/tickers.")

    df_all = pd.concat(all_stats, axis = 0)

    rets_all = pd.concat(rets_blocks, axis = 1).ffill().bfill()
    
    rets_all = rets_all.sort_index(axis = 1)

    rets_all = _ensure_ends_at_today(
        df = rets_all,
        today = end_date, 
        is_returns = True
    )

    return df_all, rets_all


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
    
    index_data = yf.download(tickers, start = '2000-01-01', end = config.TODAY, auto_adjust = True)['Close']
    
    index_data = index_data.ffill().bfill()
        
    rets = index_data.pct_change().dropna()
        
    rets_last_year = rets.loc[rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    rets_last_year_len = len(rets_last_year)
    
    rets_ann = (1 + rets_last_year).prod() - 1
    
    vol = rets.std() * np.sqrt(rets_last_year_len)
    
    sr = (rets_ann - config.RF) / vol
    
    hl = 0.1 * rets_last_year_len
    
    exp_ret_ind = rets_last_year.ewm(halflife = hl, adjust = False).mean().iloc[-1] * rets_last_year_len
   
    exp_std_ind = rets_last_year.ewm(halflife = hl, adjust = False).std().iloc[-1] * np.sqrt(rets_last_year_len)
   
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
    
    index_close = _ensure_ends_at_today(
        df = index_data, 
        today = config.TODAY, 
        is_returns = False
    )
    
    return df, index_close


def infer_fx_universe(
    tickers: list[str],
    country_to_pair: dict | None = None,
    extra: tuple[str, ...] = country_to_ccy.values(),
) -> list[str]:
    """
    Build the set of currency codes we must include in the FX sheet.
    Union of:
   
      - suffix->ccy map
   
      - country->ccy map
   
      - any currencies appearing in country_to_pair pairs (LHS/RHS)
   
      - a conservative 'extra' list
    """
    
    ccys = set(map(str.upper, extra))

    try:

        ccys.update(map(str.upper, _CCY_BY_SUFFIX.values()))

    except Exception:

        pass

    try:

        ccys.update(map(str.upper, country_to_ccy.values()))

    except Exception:

        pass

    if country_to_pair:

        for p in country_to_pair.values():

            if isinstance(p, str) and len(p) >= 6:

                ccys.add(p[: 3].upper())

                ccys.add(p[3: 6].upper())

    return sorted(ccys)


def build_fx_usd_per_ccy(
    ccys: list[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Build a daily (business-day) panel with columns 'USD_per_{CCY}' for a superset
    of currencies inferred from ticker suffix mapping plus any extras.

    Logic replicates RatioData._usd_per_one:
    
      - If CCY == 'USD' → 1.0 flat series
    
      - Else try '{CCY}USD=X' (USD per CCY). If missing, try 'USD{CCY}=X' and invert.
    
      - Reindex to business-day calendar and forward-fill.

    Returns
    -------
    pd.DataFrame
        Index: business days; Columns: 'USD_per_{CCY}'.
    """
   
    if start is None:
   
        start = pd.to_datetime(config.FIVE_YEAR_AGO)
   
    if end is None:
   
        end = pd.to_datetime(config.TODAY)

    idx = pd.date_range(start = start, end = end, freq = "B")
    
    out = {}

    for ccy in ccys:
    
        col = f"USD_per_{ccy}"
    
        if ccy == "USD":
    
            out[col] = pd.Series(1.0, index = idx, name = col)
    
            continue

        s = yf.download(f"{ccy}USD=X", start = start, end = end, progress = False, auto_adjust = True)["Close"].ffill()
       
        if not isinstance(s, pd.Series):
       
            s = s.squeeze()
       
        if s is not None and not s.empty:
       
            out[col] = s.reindex(idx).ffill().rename(col)
       
            continue

        s = yf.download(f"USD{ccy}=X", start = start, end = end, progress = False, auto_adjust = True)["Close"].ffill()
       
        if not isinstance(s, pd.Series):
       
            s = s.squeeze()
       
        if s is not None and not s.empty:
       
            out[col] = (1.0 / s).reindex(idx).ffill().rename(col)
       
            continue

        out[col] = pd.Series(index = idx, dtype = float, name = col)

    fx_df = pd.DataFrame(out, index = idx).sort_index()

    fx_df.index.name = "Date"

    return fx_df


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
    
    uk = [t for t in tickers if t.endswith('.L')]
    
    start_date = '2000-01-01'  
    
    data = download_data(
        tickers = tickers, 
        start = start_date, 
        end = config.TODAY
    )
    
    close = get_close_series(
        data = data, 
        tickers = tickers
    ).ffill()
    
    high = data['High'].ffill()
   
    low = data['Low'].ffill()
   
    volume = data['Volume'].ffill()
   
    open = data['Open'].ffill()

    if uk:
       
        high = _fix_london_unit_flips_df(
            df = high, 
            uk_tickers = uk
        )
       
        low = _fix_london_unit_flips_df(
            df = low, 
            uk_tickers = uk
        )
       
        open = _fix_london_unit_flips_df(
            df = open,
            uk_tickers = uk
        )

        high.loc[:, uk] = high.loc[:, uk].div(100)
       
        low.loc[:, uk] = low.loc[:, uk].div(100)
       
        open.loc[:, uk] = open.loc[:, uk].div(100)

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
    
    index_data, index_close = get_index_data()
    
    factor_etfs, factor_rets = get_factor_etfs_all_regions()
    
    fx_ccys = infer_fx_universe(
        tickers = tickers,
        country_to_pair = country_to_pair  
    )
    
    fx_usd_per_ccy = build_fx_usd_per_ccy(
        ccys = fx_ccys
    )
    
    missing_cols = [c for c in fx_usd_per_ccy.columns if fx_usd_per_ccy[c].isna().all()]
    
    if missing_cols:
    
        print("WARNING: missing FX series:", missing_cols)
   
    sheets_data = {
        "Close": close,
        "Open": open,
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
        "FX USD per CCY": fx_usd_per_ccy,
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
  
