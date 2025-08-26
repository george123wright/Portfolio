"""
Loads historical macroeconomic data per country and provides annualised statistics and forecasts.
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from functools import cached_property
from ratio_data import RatioData
import config


r = RatioData()


COUNTRY_FALLBACK = {   
    "Guernsey": "United Kingdom",
    "Taiwan": "China",
}


_QVARS = ["Interest", "Cpi", "Gdp", "Unemp"]


_VAR_KEY = {
    "Interest": "Interest_Rate",
    "Cpi": "Inflation_Rate",
    "Gdp": "Gdp",
    "Unemp": "Unemployment_Rate",
}


class MacroData:
    """
    MacroData

    Convenience loader/adapter for (1) historical macro time series by country
    and (2) TradingEconomics-style macro forecasts, plus (3) equity/commodity
    index price panels and (4) FX conversion helpers.

    Files
    -----
    - Historical:  BASE_DIR / "macro_and_index_data.xlsx"
    * Sheets: one per country (e.g. "United States"), plus
        "Stock Indexes and Commodities" (price levels), optional "FX".
    * Quarterly columns expected (per country): "Interest", "Cpi", "Gdp", "Unemp".

    - Forecasts:   config.DATA_FILE (already produced earlier in your pipeline)
    * Sheets used: "Inflation_Rate", "Gdp", "Unemployment_Rate", "Interest_Rate",
        and "Currency" (FX table).

    Key behavior
    ------------
  
    - Country name canonicalization with fallbacks (e.g., "Guernsey" → "United Kingdom").
    
    - Historical quarterly series are annualized into year-end panels and converted to
    percentage changes where appropriate.
    
    - If a country's quarterly series is missing a macro column, US data are used
    as fallback on a per-period basis.
    
    - Forecast tables are looked up by country name with US fallback when missing.

    Attributes
    ----------
    today : datetime.date
    hist_path : pathlib.Path
    forecast_path : pathlib.Path
    r : RatioData
    tickers : list[str]
    SHEET_INDEXES : str
    _FORECAST_SHEETS : dict[str, str]
    """

    
    SHEET_INDEXES = "Stock Indexes and Commodities"
    
    _FORECAST_SHEETS = {
        "Inflation_Rate": "Inflation_Rate",
        "Gdp": "Gdp",
        "Unemployment_Rate": "Unemployment_Rate",
        "Interest_Rate": "Interest_Rate",
    }


    def __init__(
        self
    ) -> None:
        """
        Initialize macro data paths, helpers, and the working ticker universe.

        Side effects
        ------------
        - Sets `today`, resolves Excel paths from `config`.
        - Instantiates an internal `RatioData` for analyst country metadata.
        - Reads the working tickers from `config.tickers`.
        """
            
        self.today = dt.date.today()
        
        self.hist_path = Path(config.BASE_DIR / "macro_and_index_data.xlsx")
        
        self.forecast_path = Path(
            config.DATA_FILE
        )
        
        self.r = RatioData()
        
        self.tickers = config.tickers


    @staticmethod
    def _canon(
        name: str
    ) -> str:
        """
        Normalise a country/name key for dictionary lookups.

        Parameters
        ----------
        name : str
            Raw country or sheet name.

        Returns
        -------
        str
            Lowercased, stripped key (e.g., "United States" → "united states").
        """

        return name.strip().lower()


    @staticmethod
    def _annualise(
        series: pd.Series, 
        method: str, 
        col: str, 
        pct: str
    ) -> pd.DataFrame:
        """
        Convert a quarterly `Series` to an annual panel and compute a % change column.

        Parameters
        ----------
        series : pd.Series
            Quarterly series indexed by PeriodIndex('Q') or datetime-like (will be coerced).
        method : {'mean','last'}
            Aggregation for the year: 'mean' over quarters or 'last' quarter.
        col : str
            Output column name for the aggregated level.
        pct : str
            Output column name for the derived percent change.

        Returns
        -------
        pd.DataFrame
            Index: PeriodIndex('Y'), columns: [col, pct] where pct = df[col].pct_change().

        Notes
        -----
        - If index is a PeriodIndex(Q), it is converted to timestamp for resampling, then
        re-cast to PeriodIndex('Y').
        """

        if isinstance(series.index, pd.PeriodIndex):
            
            series = series.to_timestamp(how="end")
       
        ann = series.resample("YE").mean() if method == "mean" else series.resample("YE").last()
        
        ann.index = ann.index.to_period("Y")
       
        df = ann.to_frame(col)
       
        df[pct] = df[col].pct_change(fill_method = None)
       
        return df


    @staticmethod
    def _with_us_fallback(
        df: pd.DataFrame, 
        us: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fill missing values of a country's annual panel with US values, aligned by index/columns.

        Parameters
        ----------
        df : pd.DataFrame
            Annual country panel with potentially missing values.
        us : pd.DataFrame
            US annual panel with the same measure columns.

        Returns
        -------
        pd.DataFrame
            `df` with NAs filled from `us` for matching year rows/columns.
        """

        df = df.reindex(columns = us.columns)

        return df.combine_first(us.loc[df.index])


    @cached_property
    def _sheet_key(
        self
    ) -> dict[str,str]:
        """
        Build a mapping {normalized sheet name → exact sheet name} for the historical workbook.

        Returns
        -------
        dict[str, str]
            Includes 'united states' → 'United States' and all other country sheets,
            excluding the index/commodity sheet.

        Notes
        -----
        - Cached for reuse.
        """

        wb = pd.ExcelFile(self.hist_path, engine="openpyxl")
       
        key = {
            self._canon(
                name = "United States"
            ): "United States"
        }

        for s in wb.sheet_names:
          
            if s not in (self.SHEET_INDEXES, "United States"):
          
                key[self._canon(s)] = s

        return key


    def _sheet_for(
        self, 
        country: str | None
    ) -> str:
        """
        Resolve the historical sheet name for a given country with fallbacks.

        Parameters
        ----------
        country : str | None
            Country from analyst metadata. If None or not recognized, defaults to US.

        Returns
        -------
        str
            Exact sheet name present in the historical workbook.
        """

        if not country:
         
            country = "United States"

        country = COUNTRY_FALLBACK.get(country, country)

        return self._sheet_key.get(self._canon(country), "United States")


    @cached_property
    def historical(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Annualized macro *rate-of-change* panel by country (with US fallbacks).

        Process
        -------
        1) Load quarterly macro for each country and US.
        2) For any missing quarterly macro column in a country, fill from US for the
        overlapping quarters.
        3) Annualize:
        - Interest: year mean → 'Interest_Rate' then % change → 'InterestRate_pct_change'
        - CPI:      year mean → 'CPI'           then % change → 'Inflation_rate'
        - GDP:      year last → 'Gdp'           then % change → 'GDP_growth'
        - Unemp:    year mean → 'Unemployment_Rate' then % change → 'Unemployment_pct_change'
        4) Join annual % change columns and fill remaining gaps from US.

        Returns
        -------
        dict[str, pd.DataFrame]
            country → DataFrame indexed by PeriodIndex('Y') with columns:
            ['InterestRate_pct_change','Inflation_rate','GDP_growth','Unemployment_pct_change'].
        """
      
        wb = pd.ExcelFile(self.hist_path, engine = "openpyxl")

        us_q = pd.read_excel(
            self.hist_path,
            sheet_name = "United States", 
            index_col = 0,
            engine = "openpyxl", 
            parse_dates = True
        )

        us_q.index = us_q.index.to_period("Q")

        us_ir = self._annualise(
            series = us_q["Interest"], 
            method = "mean",
            col = "Interest_Rate", 
            pct = "InterestRate_pct_change"
        )

        us_inf = self._annualise(
            series = us_q["Cpi"], 
            method = "mean",
            col = "CPI", 
            pct = "Inflation_rate"
        )

        us_gdp = self._annualise(
            series = us_q["Gdp"], 
            method = "last",
            col = "Gdp", 
            pct = "GDP_growth"
        )

        us_ue = self._annualise(
            series = us_q["Unemp"], 
            method = "mean",
            col = "Unemployment_Rate", 
            pct = "Unemployment_pct_change"
        )

        us_hist = us_ir.join([
            us_inf["Inflation_rate"],
            us_gdp["GDP_growth"],
            us_ue["Unemployment_pct_change"]
        ], how = "inner")

        data = {"United States": us_hist}

        for sheet in wb.sheet_names:
            
            if sheet in (self.SHEET_INDEXES, "United States"):
               
                continue

            df_q = pd.read_excel(
                self.hist_path,
                sheet_name = sheet, 
                index_col = 0,
                engine = "openpyxl", 
                parse_dates = True
            )

            df_q.index = df_q.index.to_period("Q")

            for v in _QVARS:
            
                if v not in df_q.columns:
               
                    df_q[v] = us_q[v].loc[df_q.index].values
               
                else:
               
                    df_q[v] = df_q[v].fillna(us_q[v].loc[df_q.index])

            ir  = self._annualise(
                series = df_q["Interest"],
                method = "mean",
                col = "Interest_Rate", 
                pct = "InterestRate_pct_change"
            )
       
            inf = self._annualise(
                series = df_q["Cpi"], 
                method = "mean",
                col = "CPI", 
                pct = "Inflation_rate"
            )
       
            gdp = self._annualise(
                series = df_q["Gdp"], 
                method = "last",
                col = "Gdp", 
                pct = "GDP_growth"
            )
       
            ue  = self._annualise(
                series = df_q["Unemp"], 
                method = "mean",
                col = "Unemployment_Rate", 
                pct = "Unemployment_pct_change"
            )

            df_y = ir.join([
                inf["Inflation_rate"],
                gdp["GDP_growth"],
                ue["Unemployment_pct_change"]
            ], how="inner")
       
            df_y = self._with_us_fallback(
                df = df_y, 
                us = us_hist
            )
       
            data[sheet] = df_y

        return data


    @cached_property
    def historical_non_pct(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Quarterly macro *levels* by country (no % transforms), 2010Q1 onward.

        Process
        -------
        - Load quarterly macro for each sheet, fill missing columns forward from US per-quarter.
        - Convert special units for selected countries (e.g., divide 'Gdp' by 1e9 for CHINA/UK/Germany).

        Returns
        -------
        dict[str, pd.DataFrame]
            country → DataFrame with columns ['Interest','Cpi','Gdp','Unemp'],
            index PeriodIndex('Q'), filtered to ≥ 2010Q1.
        """
       
        wb = pd.ExcelFile(
            self.hist_path, 
            engine = "openpyxl"
        )
       
        us_q = pd.read_excel(
            self.hist_path,
            sheet_name = "United States",
            index_col = 0, 
            engine = "openpyxl",
            parse_dates = True
        )
       
        us_q.index = us_q.index.to_period("Q")
       
        us_q = us_q[us_q.index >= pd.Period("2010Q1")]

        data = {
            "United States": us_q[_QVARS].ffill().fillna(0)
        }

        for sheet in wb.sheet_names:
       
            if sheet in (self.SHEET_INDEXES, "United States", "FX"):
              
                continue

            df_q = pd.read_excel(
                self.hist_path,
                sheet_name = sheet,
                index_col = 0, 
                engine = "openpyxl",
                parse_dates = True
            )
            
            df_q.index = df_q.index.to_period("Q")
           
            df_q = df_q[df_q.index >= pd.Period("2010Q1")]

            for v in _QVARS:
             
                df_q[v] = df_q.get(v, us_q[v].loc[df_q.index]).ffill().fillna(0)

            if sheet in ["CHINA", "United Kingdom", "Germany"]:
              
                df_q["Gdp"] = df_q["Gdp"] / 1e9

            data[sheet] = df_q[_QVARS]

        return data


    @cached_property
    def currency(
        self
    ) -> pd.Series:
        """
        Load latest FX quotes from the forecasts workbook.

        Returns
        -------
        pd.Series
            'Last' column from the 'Currency' sheet, indexed by currency pair (e.g., 'GBPUSD').
        """
      
        df = pd.read_excel(
            self.forecast_path,
            sheet_name = "Currency",
            index_col = 0,
            engine = "openpyxl",
            parse_dates = True
        )
       
        return df["Last"]
    
    
    @cached_property
    def interest(
        self
    ) -> pd.Series:
        """
        Load latest Interest Rates from the forecasts workbook.

        Returns
        -------
        pd.Series
            'Last' column from the 'Interest_Rate' sheet, indexed by Country.
        """
      
        df = pd.read_excel(
            self.forecast_path,
            sheet_name = "Interest_Rate",
            index_col = 0,
            engine = "openpyxl",
            parse_dates = True
        )
       
        return df["Last"]  


    @cached_property
    def prices(
        self
    ) -> pd.DataFrame:
        """
        Load index/commodity price levels.

        Returns
        -------
        pd.DataFrame
            From the 'Stock Indexes and Commodities' sheet, datetime index (converted), raw prices.
        """
      
        df = pd.read_excel(
            self.hist_path,
            sheet_name = self.SHEET_INDEXES,
            index_col = 0,
            engine = "openpyxl",
            parse_dates = True
        )
       
        df.index = pd.to_datetime(df.index)
       
        return df


    @cached_property
    def forecasts(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Load macro forecast tables for Interest, CPI/Inflation, GDP, and Unemployment.

        Returns
        -------
        dict[str, pd.DataFrame]
            {
            'Interest_Rate': df,
            'Inflation_Rate': df,
            'Gdp': df,
            'Unemployment_Rate': df,
            }
        where DataFrames are indexed by country and columns are horizon labels
        (e.g., 'Last', 'Q1/26', or calendar years for GDP).

        Side effects
        ------------
        - Builds `_forecast_key` for fast country name → index resolution.

        Notes
        -----
        - Attempts to parse yearly indices to PeriodIndex('Y') when possible.
        """
       
        out = {}
       
        wb = pd.ExcelFile(
            self.forecast_path, 
            engine = "openpyxl"
        )
       
        for var, sheet in self._FORECAST_SHEETS.items():
       
            df = pd.read_excel(
                self.forecast_path,
                sheet_name = sheet,
                index_col = 0,
                engine = "openpyxl"
            )
       
            try:
          
                df.index = pd.to_datetime(df.index, format="%Y").to_period("Y")
          
            except (ValueError, TypeError):
          
                pass
       
            out[var] = df
       
        self._forecast_key = {
            self._canon(idx): idx
            for idx in out["Interest_Rate"].index
        }
       
        return out


    def _forecast_for(
        self, 
        country: str
    ) -> str:
        """
        Resolve the forecast table row key (country) with US fallback.

        Parameters
        ----------
        country : str

        Returns
        -------
        str
            Exact index label present in the forecast tables (default 'United States').
        """
    
        return self._forecast_key.get(
            self._canon(country), "United States"
        )


    def _fc_row(
        self, 
        var_short: str, 
        sheet: str
    ) -> pd.Series:
        """
        Get the forecast row for a macro variable and sheet (country), handling fallbacks.

        Parameters
        ----------
        var_short : {'Interest','Cpi','Gdp','Unemp'}
            Short variable alias mapped via `_VAR_KEY`.
        sheet : str
            Historical sheet/country name. If in `_hist_fallback`, US forecasts are used.

        Returns
        -------
        pd.Series
            A single country's forecast row for the specified macro variable.
        """

        if sheet in getattr(self, "_hist_fallback", {}):
         
            country = "United States"
        
        else:
        
            country = sheet
       
        return self.forecasts[_VAR_KEY[var_short]].loc[
            self._forecast_for(
                country = country
            )
        ]


    def get_base_macro_fc(
        self, 
        country: str | None
    ) -> dict[str, float]:
        """
        Compute a compact macro forecast dictionary for one country.

        Parameters
        ----------
        country : str | None
            Country name; resolves via `_sheet_for` and COUNTRY_FALLBACK.

        Returns
        -------
        dict[str, float]
            {
            'InterestRate_pct_change': (IR_Q1/26 - IR_Last)/IR_Last,
            'Inflation_rate':          CPI_Q1/26 (level as rate),
            'GDP_growth':              (GDP_2026 - GDP_Last)/GDP_Last,
            'Unemployment_pct_change': (U_Q1/26  - U_Last)/U_Last,
            }

        Notes
        -----
        - GDP uses calendar-year columns (e.g., '2026'); other variables use quarter labels.
        """
       
        sheet = self._sheet_for(
            country = country
        )

        ir = self._fc_row(
            var_short = "Interest", 
            sheet = sheet
        )
       
        inf = self._fc_row(
            var_short = "Cpi", 
            sheet = sheet
        )
       
        gdp = self._fc_row(
            var_short = "Gdp", 
            sheet = sheet
        )
       
        ue = self._fc_row(
            var_short = "Unemp", 
            sheet = sheet
        )

        ir_last = ir["Last"]
        
        gdp_last = gdp["Last"]
        
        ue_last = ue["Last"]

        return {
            "InterestRate_pct_change": (ir["Q1/26"] - ir_last) / ir_last,
            "Inflation_rate": inf["Q1/26"],
            "GDP_growth": (gdp["2026"] - gdp_last) / gdp_last,
            "Unemployment_pct_change": (ue["Q1/26"] - ue_last) / ue_last,
        }


    def macro_forecast_dict(
        self
    ) -> dict[str, dict[str, float]]:
        """
        Build per-ticker macro forecast bundles (raw row arrays).

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            {
            TICKER: {
                'InterestRate':              <forecast row values>,
                'Consumer_Price_Index_Cpi':  <forecast row values>,
                'GDP':                       <['2025','2026'] values>,
                'Unemployment':              <forecast row values>,
            },
            ...
            }

        Notes
        -----
        - Country for each ticker comes from `r.analyst['country']` with sheet resolution
        and fallbacks as in `_sheet_for`.
        """
  
        records: dict[str, dict[float]] = {}

        for ticker in self.tickers:
       
            country_raw = (
                self.r.analyst.loc[ticker, "country"]
                if "country" in self.r.analyst.columns
                else None
            )
       
            sheet = self._sheet_for(
                country = country_raw
            )
            
            ir_values = self._fc_row(
                var_short = "Interest", 
                sheet = sheet
            ).values
            
            cpi_values = self._fc_row(
                var_short = "Cpi", 
                sheet = sheet
            ).values
            
            gdp_values = self._fc_row(
                var_short = "Gdp", 
                sheet = sheet
            )[["2025", "2026"]].values
            
            unemp_values = self._fc_row(
                var_short = "Unemp", 
                sheet = sheet
            ).values

            records[ticker] = {
                "InterestRate": ir_values,
                "Consumer_Price_Index_Cpi": cpi_values,
                "GDP": gdp_values,
                "Unemployment": unemp_values,
            }

        return records


    def assign_macro_history(
        self
    ) -> pd.DataFrame:
        """
        Attach annual *rate-of-change* macro history to each ticker based on its country.

        Returns
        -------
        pd.DataFrame
            MultiIndex [ticker, year] with columns:
            ['InterestRate_pct_change','Inflation_rate','GDP_growth','Unemployment_pct_change'].

        Notes
        -----
        - Ticker → country from `RatioData.analyst['country']`; falls back to US when unknown.
        """
       
        frames: dict[str, pd.DataFrame] = {}
       
        for ticker in self.tickers:
       
            sheet = self._sheet_for(
                country = self.r.analyst.loc[ticker, "country"]
                if "country" in self.r.analyst.columns
                else None
            )
       
            if sheet in self.historical:
              
                frames[ticker] = self.historical[sheet]
       
        return pd.concat(frames, names = ["ticker", "year"])


    def assign_macro_history_non_pct(
        self
    ) -> pd.DataFrame:
        """
        Attach quarterly *level* macro history (2010Q1+) to each ticker based on its country.

        Returns
        -------
        pd.DataFrame
            MultiIndex [ticker, year(Q)] with columns ['Interest','Cpi','Gdp','Unemp'].
        """

       
        frames: dict[str, pd.DataFrame] = {}
       
        for ticker in self.tickers:
       
            sheet = self._sheet_for(
                country = self.r.analyst.loc[ticker, "country"]
                if "country" in self.r.analyst.columns
                else None
            )
       
            if sheet in self.historical_non_pct:
           
                frames[ticker] = self.historical_non_pct[sheet]
       
        return pd.concat(frames, names=["ticker", "year"])
    

    def assign_macro_forecasts(
        self
    ) -> dict[str, dict[str, float]]:
        """
        Build a per-ticker dictionary of compact macro forecast metrics.

        Returns
        -------
        dict[str, dict[str, float]]
            Each ticker mapped to the output of `get_base_macro_fc(...)`.

        Notes
        -----
        - Applies COUNTRY_FALLBACK (e.g., 'Guernsey' → 'United Kingdom', 'Taiwan' → 'China').
        """
  
        records: dict[str, dict[str, float]] = {}
        
        for ticker in self.tickers:
        
            country = (self.r.analyst.loc[ticker, 'country']
                       if 'country' in self.r.analyst.columns else None)
        
            if country in COUNTRY_FALLBACK:
           
                country = COUNTRY_FALLBACK[country]
        
            records[ticker] = self.get_base_macro_fc(
                country = country
            )
        
        return records
    
    
    def convert_to_gbp_rates(
        self,
        current_col: str = 'Last',
        future_col:  str = 'Q1/26'
    ) -> pd.DataFrame:
        """
        Convert 'Currency' forecast table into GBP-cross rates and compute predicted changes.

        Parameters
        ----------
        current_col : str, default 'Last'
            Column name for spot.
        future_col : str, default 'Q1/26'
            Column name for forecast horizon.

        Returns
        -------
        pd.DataFrame
            Index: synthetic GBP crosses (e.g., 'GBPUSD','GBPEUR','GBPCAD', ...).
            Columns:
            - 'GBP-X Today'     : current GBP→X rate
            - 'GBP-X Q1/26'     : forecast GBP→X rate
            - 'Pred Change (%)' : (future / current) - 1

        Raises
        ------
        ValueError
            If 'GBPUSD' row is missing in the 'Currency' sheet.

        Logic
        -----
        - If pair == 'GBPUSD': use directly.
        - If pair endswith 'USD' (e.g., 'EURUSD'): GBP→EUR ≈ GBPUSD / EURUSD.
        - If pair startswith 'USD' (e.g., 'USDCAD'): GBP→CAD ≈ USD→CAD * GBPUSD.
        - Non-USD-anchored pairs are skipped.
        """

        df = pd.read_excel(
            self.forecast_path,
            sheet_name = "Currency",
            engine = "openpyxl",
            usecols = [0, 1, 5],
            index_col = 0,
            thousands = ','
        )
       
        try:
           
            gbp_usd = df.at['GBPUSD', current_col]
      
        except KeyError:
      
            raise ValueError("Couldn't find a 'GBPUSD' row in the DataFrame.") 

        records = []
      
        for pair, row in df.iterrows():
      
            cur = row[current_col]
      
            fut = row[future_col]

            if pair == 'GBPUSD':
      
                curr_rate = gbp_usd
      
                fut_rate  = df.at['GBPUSD', future_col]
      
            elif pair.endswith('USD'):
      
                curr_rate = gbp_usd / cur
      
                fut_rate  = gbp_usd / fut
      
            elif pair.startswith('USD'):
      
                curr_rate = cur * gbp_usd
      
                fut_rate  = fut * gbp_usd
      
            else:
                continue

            pct_change = (fut_rate / curr_rate - 1)
      
            records.append({
                'Currency Pair': pair,
                'GBP-X Today':   curr_rate,
                'GBP-X Q1/26':   fut_rate,
                'Pred Change (%)': pct_change,
            })
            
        df = pd.DataFrame(records)


        def _label(
            pair: str
        ) -> str:
      
            if pair == 'GBPUSD':
      
                x = 'USD'
      
            elif pair.endswith('USD'):
      
                x = pair[:3]
      
            elif pair.startswith('USD'):
      
                x = pair[3:]
      
            else:
      
                x = pair
      
            return f"GBP{x}"

        df.index = df['Currency Pair'].map(_label)

        df = df.drop(columns=['Currency Pair'])

        return df
        
        
