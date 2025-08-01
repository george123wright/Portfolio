"""
Downloads historical market data, computes technical indicators, scrapes Trading Economics forecasts and exports all series to Excel.
"""

import datetime as dt
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import yfinance as yf
import ta
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
    "interest-rate": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "stock-market": {"index_pos": 1, "mapping": {1: "Index", 2: "Last", 4: "Q2/25", 5: "Q3/25", 6: "Q4/25", 7: "Q1/26"}},
    "currency": {"index_pos": 1, "mapping": {1: "Currency Pair", 2: "Last", 4: "Q2/25", 5: "Q3/25", 6: "Q4/25", 7: "Q1/26"}},
    "wages": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "unemployment-rate": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "gdp": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "2025", 4: "2026", 5: "2027"}},
    "consumer-price-index-cpi": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "inflation-rate": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "consumer-confidence": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "business-confidence": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "balance-of-trade": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
    "corporate-profits": {"index_pos": 0, "mapping": {0: "Country", 1: "Last", 3: "Q2/25", 4: "Q3/25", 5: "Q4/25", 6: "Q1/26"}},
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

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def configure_logging(
    level: int = logging.INFO
) -> None:
    """
    Configure logging format and level.
    """
   
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
   
    logging.basicConfig(level=level, format=fmt)


def download_data(
    tickers: List[str], 
    start: dt.date, 
    end: dt.date
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance for given tickers.
    Raises ValueError if the result is empty.
    """
   
    logging.info("Downloading data for %d tickers: %s to %s", len(tickers), start, end)
   
    data = yf.download(tickers, start=start, end=end)
   
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
    Scrape a Trading Economics forecast table by endpoint.
    Uses col_map to map td indices to column names, and index_pos for the index column.
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
   
    url = f"{BASE_FORECAST_URL}/commodity"
   
    resp = requests.get(url, headers=HEADERS, timeout=10)
    
    resp.raise_for_status()
   
    soup = BeautifulSoup(resp.text, "html.parser")

    for tbl in soup.find_all("table"):
       
        thead = tbl.find("thead")
       
        if not thead:
          
            continue
       
        first_th = thead.find("th")
       
        if not first_th or first_th.get_text(strip=True) != category:
            
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
    Extract and scale 'Close' prices; interpolate missing values.
    Returns DataFrame of closes.
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


def compute_technical_indicators(
    data: pd.DataFrame,
    tickers: List[str],
    rsi_window: int = 14
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute RSI and MACD (+ signal) for each ticker in 'Close'.
    Returns: (RSI_df, MACD_df, MACD_signal_df) with MultiIndex columns.
    """
   
    close = get_close_series(
        data = data, 
        tickers = tickers
    )
   
    rsi_d, macd_d, macd_sig_d = {}, {}, {}
   
    for t in tickers:
       
        if t not in close:
       
            logging.warning("Skipping indicators for %s", t)
       
            continue
   
        s = close[t]
      
        rsi_d[t] = ta.momentum.RSIIndicator(s, window=rsi_window).rsi()
      
        mac = ta.trend.MACD(s)
      
        macd_d[t] = mac.macd()
      
        macd_sig_d[t] = mac.macd_signal()
   
    rsi_df = pd.DataFrame(rsi_d)
    macd_df = pd.DataFrame(macd_d)
    macd_sig_df = pd.DataFrame(macd_sig_d)
   
    rsi_df.columns = pd.MultiIndex.from_product([['RSI'], rsi_df.columns])
    macd_df.columns = pd.MultiIndex.from_product([['MACD'], macd_df.columns])
    macd_sig_df.columns = pd.MultiIndex.from_product([['MACD Signal'], macd_sig_df.columns])
   
    return rsi_df, macd_df, macd_sig_df


def fetch_fred_series(
    symbol: str, 
    start: str, 
    end: str, 
    freq: str = 'M'
) -> pd.Series:
    """
    Fetch a FRED series and resample forward-fill to freq.
    """

    s = web.DataReader(symbol, 'fred', start, end).squeeze()

    if freq:
       
        s = s.resample(freq).ffill()

    return s


def macro_data() -> pd.DataFrame:
    """
    Compile monthly macro indicators and SP500 history into one clean DataFrame.
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
    
    macro_df = df.join(sp500_monthly, how='inner')
  
    return macro_df


def get_sector_data() -> pd.DataFrame:
    
    sec_data = yf.download(list(sec_map.keys()), start = config.YEAR_AGO, end = config.TODAY)['Close']
    
    sec_data = sec_data.rename(columns = sec_map)
    
    rets = sec_data.pct_change().dropna()
    
    rets_len = len(rets)
    
    rets_ann = (1 + rets).prod() - 1
    
    vol = rets.std() * np.sqrt(rets_len)
    
    sr = (rets_ann - config.RF) / vol
    
    exp_ret_ind = rets.ewm(halflife = 0.2 * rets_len, adjust = False).mean().iloc[-1] * rets_len
   
    exp_std_ind = rets.ewm(halflife = 0.2 * rets_len, adjust = False).std().iloc[-1] * np.sqrt(rets_len)
   
    exp_sr_ind = (exp_ret_ind - 0.0435) / exp_std_ind
    
    df = pd.DataFrame({
        "Sector": sec_data.columns,
        "Returns": rets_ann,
        "Volatility": vol,
        "Sharpe Ratio": sr,
        "Exp Returns": exp_ret_ind,
        "Exp Volatility": exp_std_ind,
        "Exp Sharpe Ratio": exp_sr_ind
    }).set_index("Sector")
    
    return df


def get_industry_data() -> pd.DataFrame:
    """
    Download industry data, compute returns, volatility, and Sharpe ratio.
    Returns a DataFrame with industry metrics.
    """
    
    ind_list = list(IndustryMap.keys())
    
    ind_ticker_map = {}
    
    for ind in ind_list:
    
        sec_data = yf.Industry(ind).ticker.ticker
    
        ind_ticker_map[sec_data] = ind
    
    ind_ticker_map

    ind_data = yf.download(list(ind_ticker_map.keys()), start = config.YEAR_AGO, end = config.TODAY)['Close']
    
    ind_data = ind_data.rename(columns=ind_ticker_map)

    ind_data = ind_data.rename(columns=IndustryMap)
    
    ind_data = ind_data.groupby(axis=1, level=0).sum()

    rets_ind = ind_data.pct_change().dropna()
    
    rets_len = len(rets_ind)
    
    rets_ann_ind = (1 + rets_ind).prod() - 1        
    
    vol_ind = rets_ind.std() * np.sqrt(rets_len)
    
    sr_ind = ((rets_ann_ind - 0.0465) / vol_ind).fillna(0)
    
    exp_ret_ind = rets_ind.ewm(halflife = 0.2 * rets_len, adjust = False).mean().iloc[-1] * rets_len
   
    exp_std_ind = rets_ind.ewm(halflife=0.2 * rets_len, adjust = False).std().iloc[-1] * np.sqrt(rets_len)
   
    exp_sr_ind = (exp_ret_ind - 0.0435) / exp_std_ind
    
    df = pd.DataFrame({
        "Industry": ind_data.columns,
        "Returns": rets_ann_ind,
        "Volatility": vol_ind,
        "Sharpe Ratio": sr_ind,
        "Exp Returns": exp_ret_ind,
        "Exp Volatility": exp_std_ind,
        "Exp Sharpe Ratio": exp_sr_ind
    }).set_index("Industry")
    
    return df


def main() -> None:
    """
    Orchestrate data download, indicator computation, forecast scraping,
    macro data retrieval, and single-pass Excel export.
    """
   
    configure_logging()

    tickers = config.tickers
    
    start_date = '2000-01-01'  
    
    index_tickers = ['^GSPC', '^NDX', '^FTSE', '^GDAXI', '^FCHI', '^AEX', '^IBEX', '^GSPTSE', '^HSI', '^SSMI', 'VWRL.L', '^IXIC']
   
    data = download_data(
        tickers = tickers, 
        start = start_date, 
        end = config.TODAY
    )
    
    close = get_close_series(
        data = data, 
        tickers = tickers
    )
    
    index_data = download_data(
        tickers = index_tickers, 
        start = config.FIVE_YEAR_AGO, 
        end = config.TODAY
    )
   
    index_close = get_close_series(
        data = index_data, 
        tickers = index_tickers
    )
    
    high = data['High']
    
    low = data['Low']
    
    open_p = data['Open']
    
    volume = data['Volume']
   
    weekly_close = close.resample('W').last()
   
    daily_ret = close.pct_change()
    
    weekly_ret = weekly_close.pct_change()
   
    rsi_df, macd_df, macd_sig_df = compute_technical_indicators(
        data = data, 
        tickers = tickers
    )
   
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
    
    sector_data = get_sector_data()
    
    industry_data = get_industry_data()
   
    sheets_data = {
        "Close": close,
        "High": high,
        "Low": low,
        "Open": open_p,
        "Volume": volume,
        "Weekly Close": weekly_close,
        "Historic Returns": daily_ret,
        "Historic Weekly Returns": weekly_ret,
        "RSI": rsi_df,
        "MACD": macd_df,
        "MACD Signal": macd_sig_df,
        **forecasts,
        "Macro Data": macro_df,
        "Sector Data": sector_data,
        "Industry Data": industry_data,
        "Index Close": index_close
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
        output_excel_file = config.DATA_FILE
    )
   
    export_results(
        sheets = {"Exponential Returns": er_ema_df, "Daily Returns": dr_df},
        output_excel_file = config.FORECAST_FILE
    )


if __name__ == '__main__':
    main()
  
