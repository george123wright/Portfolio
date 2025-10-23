import pandas as pd
from typing import Dict
from functions.export_forecast import export_results
from data_processing.ratio_data import RatioData
import config


SHEET_NAMES = [
    "DCF",
    "DCFE",
    "RI",
    "SARIMAX Monte Carlo",
    "Prophet Pred",
    "Prophet PCA",
    "LSTM_DirectH",
    "LSTM_Cross_Asset",
    "GRU_raw",
    "GRU_cal",
    "Advanced MC",
    "HGB Returns",
    "hgb_cross_asset",
    "TVP + GARCH Monte Carlo",
    "SARIMAX Factor",
]

PRICE_COLS = ["Low Price", "Avg Price", "High Price"]


def _read_model_sheets() -> Dict[str, pd.DataFrame]:
    """
    Read all required model sheets from the Excel file.
    """
    
    sheets = pd.read_excel(
        config.MODEL_FILE,
        sheet_name = SHEET_NAMES,
        engine = "openpyxl",
        index_col = 0,
    )

    return {
        name: sheets[name] for name in SHEET_NAMES
    }


def clean_headers(cols):
   
    seen, out = set(), []
   
    for c in map(lambda x: str(x).strip(), cols):
        
        if c == "" or c.lower() == "nan":
            
            c = "Col" 
            
        else:
            
            c = c
   
        base, i, name = c, 1, c
   
        while name in seen:
   
            i += 1
   
            name = f"{base}_{i}"
   
        seen.add(name)
   
        out.append(name[:200]) 
   
    return out


def _reindex_and_cast(
    dfs: Dict[str, pd.DataFrame], 
    tickers
) -> Dict[str, pd.DataFrame]:
    """
    Reindex each DataFrame to the target tickers and normalise dtypes.
    """
    
    out = {}

    tickers = [str(t) for t in list(tickers)]

    numeric_cols_any = {"SE", "Returns", "Current Price", "Low Price", "Avg Price", "High Price"}

    for name, df in dfs.items():
       
        df = pd.DataFrame(df).copy()

        df.index = df.index.map(str)

        df.columns = df.columns.map(str)

        df = df.reindex(tickers)

        present_num = list(numeric_cols_any.intersection(df.columns))

        if present_num:

            df[present_num] = df[present_num].apply(pd.to_numeric, errors = "coerce")

        out[name] = df

    return out


def _add_prices_and_metrics(
    dfs: Dict[str, pd.DataFrame], 
    tickers
) -> None:
  
    r = RatioData()
  
    latest_prices = pd.Series(r.last_price)
  
    latest_prices.index = latest_prices.index.map(str)    
  
    latest_prices = latest_prices.reindex([str(t) for t in tickers])


    def _apply_return_and_se(
        df: pd.DataFrame
    ) -> None:
   
        cp = latest_prices.reindex(df.index)
   
        df.loc[:, "Current Price"] = cp.values

        valid = cp.notna()

        if "Avg Price" in df.columns:

            df.loc[valid, "Returns"] = df.loc[valid, "Avg Price"].div(cp[valid]).sub(1.0)

        if "SE" in df.columns:

            df.loc[:, "SE"] = pd.to_numeric(df["SE"], errors = "coerce")

            df.loc[valid, "SE"] = df.loc[valid, "SE"].div(cp[valid])

    for key in ("DCF", "DCFE", "RI"):
      
        if key in dfs:
      
            _apply_return_and_se(
                df = dfs[key]
            )

    if "SARIMAX Monte Carlo" in dfs:
       
        sarimax_df = dfs["SARIMAX Monte Carlo"]
      
        sarimax_df.loc[:, "Current Price"] = latest_prices.reindex(sarimax_df.index).values


def _postprocess_clip_and_clean(
    dfs: Dict[str, pd.DataFrame]
) -> None:
    """
    Apply row filters, clip ranges, and final NaN handling.
    """

    MODEL_COLS = ["Returns", "SE", "Current Price", "Low Price", "Avg Price", "High Price"]

    for name, df in dfs.items():

        if "SE" not in df.columns:

            df["SE"] = 0.0

        if "Returns" not in df.columns:

            df["Returns"] = 0.0

        present_model_cols = [c for c in MODEL_COLS if c in df.columns]

        if present_model_cols:
            
            df[present_model_cols] = df[present_model_cols].apply(pd.to_numeric, errors = "coerce")

        if present_model_cols:
            
            row_has_nan = df[present_model_cols].isna().any(axis=1)  
        
        else:
            
            row_has_nan = df.isna().any(axis = 1)
        
        mask = (df["SE"] <= 0.02) | (df["Returns"] <= -0.8) | row_has_nan
      
        if present_model_cols:
          
            df.loc[mask, present_model_cols] = 0

        if "Returns" in df.columns:

            df["Returns"] = df["Returns"].clip(lower = config.lbr, upper = config.ubr)

        if "SE" in df.columns:

            df["SE"] = df["SE"].clip(upper=2)

        num_cols = df.select_dtypes(include = ["number"]).columns
        
        df[num_cols] = df[num_cols].fillna(0)


def process_and_export_forecasts(
    tickers
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end pipeline:
      1) Load model sheets
    
      2) Align to the provided tickers and normalise dtypes
    
      3) Add current prices and compute Returns/SE where required
    
      4) Apply filtering and clipping rules
    
      5) Export all sheets via export_results
    
    Returns a dict of the processed DataFrames.
    """
    
    raw = _read_model_sheets()
    
    dfs = _reindex_and_cast(
        dfs = raw, 
        tickers = tickers
    )
    
    _add_prices_and_metrics(
        dfs = dfs, 
        tickers = tickers
    )

    name_map = {
        "DCF": "DCF",
        "DCFE": "DCFE",
        "RI": "RI",
        "SARIMAX Monte Carlo": "SARIMAX Monte Carlo",
        "Prophet Pred": "Prophet Pred",
        "Prophet PCA": "Prophet PCA",
        "LSTM_DirectH": "LSTM_DirectH",
        "LSTM_Cross_Asset": "LSTM_Cross_Asset",
        "GRU_cal": "GRU_cal",
        "GRU_raw": "GRU_raw",
        "Advanced MC": "Advanced MC",
        "HGB Returns": "HGB Returns",
        "hgb_cross_asset": "HGB Returns CA",  
        "TVP + GARCH Monte Carlo": "TVP + GARCH Monte Carlo",
        "SARIMAX Factor": "SARIMAX Factor",
    }

    sheets = {out_name: dfs[in_name] for in_name, out_name in name_map.items() if in_name in dfs}

    _postprocess_clip_and_clean(
        dfs = sheets
    )
    
    for name, df in sheets.items():
        
        df.columns = clean_headers(
            cols = df.columns
        )
        
        assert df.columns.is_unique
        
        if not df.index.name or str(df.index.name).strip().lower() in ("", "nan", "none"):
        
            df.index.name = "Ticker"
   
    export_results(
        sheets = sheets
    )
   
    return sheets


if __name__ == "__main__":

    process_and_export_forecasts(
        tickers = config.tickers
    )
