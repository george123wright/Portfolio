"""
fetch_data/fetch_macro_data.py
Retrieves macroeconomic time series (interest, CPI, GDP, etc.) from FRED for multiple countries and saves them to Excel..
"""

import pandas as pd
import datetime as dt
import logging
import yfinance as yf
from pandas_datareader import data as pdr

macro_series = {
    
    "United States": {
        "interest": "FEDFUNDS",
        "cpi":      "CPIAUCSL",
        "gdp":      "GDP",
        "unemp":    "UNRATE",
        "balance of trade": "BOPGSTB",
        "corporate profits": "CPATAX",
        "balance on current account": "IEABC"
    },
    
    "United Kingdom": {
        "interest": "IRSTCI01GBM156N",
        "cpi":      "GBRCPIALLMINMEI",
        "gdp":      "MKTGDPGBA646NWDB",
        "unemp":    "LRHUTTTTGBM156S"
    },
    
    "Germany": {
        "interest": "IRSTCI01DEM156N",
        "cpi":      "DEUCPIALLMINMEI",
        "gdp":      "MKTGDPDEA646NWDB",
        "unemp":    "LRHUTTTTDEM156S"
    },
    
    "France": {
        "interest": "IRSTCI01FRM156N",
        "cpi":      "FRACPIALLMINMEI",
        "gdp":      "MKTGDPFRA646NWDB",
        "unemp":    "LRHUTTTTFRM156S"
    },
    
    "Canada": {
        "interest": "IRSTCI01CAM156N",
        "cpi":      "CANCPIALLMINMEI",
        "gdp":      "MKTGDPCAA646NWDB",
        "unemp":    "LRUNTTTTCAM156S"
    },
    
    "Spain": {
        "interest": "IRSTCI01ESM156N",
        "cpi":      "CP0000ESM086NEST",
        "gdp":      "MKTGDPESA646NWDB",
        "unemp":    "LRHUTTTTESM156S"
    },
    
    "FX": {
        "USD per GBP": "DEXUSUK",
        "USD per EUR": "DEXUSEU",
        "USD per CHF": "DEXCZUS",
        "USD per CAD": "DEXCAUS"
    },
    
    "CHINA": {
        "interest": "INTDSRCNM193N",
        "cpi":      "CHNCPIALLMINMEI",
        "gdp":      "MKTGDPCNA646NWDB"
    },
    
    "Israel": {
        "interest": "IRSTCI01ILM156N",
    }
    
}

tickers = ["^GSPC", "^FTSE","^GDAXI", "^FCHI", "^GSPTSE", "^IBEX", "^SSMI", 'CL=F', 'GC=F']

close = (
    yf.download(tickers, start="2000-01-01", end=dt.date.today())['Close']
      .resample("Q").last()                     
      .to_period("Q")                     
)
close_df = pd.DataFrame(close)

def fetch_country_df(codes, start, end):
    dfs = []
    for key, series in codes.items():
        try:
            df = (
                pdr.DataReader(series, "fred", start, end)
                   .resample("Q").last()          
            )
            df.index = df.index.to_period("Q")      
            df.rename(columns={series: key.title().replace('_', ' ')}, inplace=True)
            dfs.append(df)
        except Exception:
            logging.warning("Failed to fetch %s for series %s", key, series)
    if dfs:
        return pd.concat(dfs, axis=1)
    else:
        return None


def main():
 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    today = dt.date.today()
    
    start = pd.to_datetime("2000-01-01")
    end   = pd.Timestamp(today)
    
    macro_by_country = {}

    for country, codes in macro_series.items():
    
        logging.info("Fetching macro data for %s...", country)
        
        df = fetch_country_df(codes, start, end)

        if df is not None and not df.empty:
            macro_by_country[country] = df
            logging.info("%s: %d quarters of data", country, len(df))
        else:
            logging.error("No data for %s (skipping)", country)

   
    out_file = "macro_and_index_data.xlsx"
    
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        for country, df in macro_by_country.items():
            df.to_excel(writer, sheet_name=country)
        close_df.to_excel(writer, sheet_name="Stock Indexes and Commodities")
    
    logging.info("Saved macro + stock index data to %s", out_file)


if __name__ == "__main__":
    main()
