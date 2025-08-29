from __future__ import annotations

"""
Financial data ingestion, normalisation, forecasting, KPI construction, and
macro-join utilities.

Overview
--------
This module defines helper functions and the `FinancialForecastData` class to
load issuer financial statements and ratio sheets from Excel, convert abbreviated
magnitudes (e.g., '1.2B') to floats, align currencies, derive annual aggregates
and cash-flow components, and build forward analyst or synthetic forecasts for
revenue and EPS across a five-year horizon. Additional outputs include KPI
summaries using exponentially weighted smoothing, Prophet-ready (also used for 
other machine learning models) time series (revenue, EPS), first-period forecast 
extraction, forecast change diagnostics, a macro-joined regression panel, and annual 
revenue/EPS growth aligned to simple price returns around reporting dates.

Key conventions and formulae
----------------------------

- Abbreviation parsing: 'T','B','M' map to 1e12, 1e9, 1e6 respectively.

- Growth parsing: strings like '12.5%' → 0.125; values greater than 1.0 are
  interpreted as percentages (e.g., 12.5 → 0.125), otherwise as proportions.

- IQR outlier filter: IQR = Q3 − Q1; keep values in [Q1 − m·IQR, Q3 + m·IQR].

- Currency adjustments: ticker-specific business rules convert series to a
  common currency basis (documented in `currency_adjustment`).

- Exponentially weighted mean (EWMA) with span s: smoothing factor
  alpha = 2 / (s + 1); recursion m_t = alpha·x_t + (1 − alpha)·m_{t−1}.

- Synthetic forecast compounding: for growth rate g and base level L_0,
  L_p = L_{p−1} × (1 + g) for horizon step p.

- Effective interest after tax: InterestAfterTax = InterestExpense × (1 − TaxRate),
  where TaxRate = IncomeTax / PretaxIncome.

- ROE geometric mean over n years: ROE_g = (∏_{i=1..n} (1 + ROE_i))^(1/n) − 1.

- Percentage change for baseline B ≠ 0: pct = (New − B) / |B|; otherwise 0.

All equations are shown in plain text and use UK spelling throughout.
"""


import re
from pathlib import Path
from typing import Final
import numpy as np
import pandas as pd
from functools import cached_property

from data_processing.macro_data import MacroData
import config


macro = MacroData()

_TODAY_TS: Final = pd.Timestamp(config.TODAY)

_HORIZON_YEARS = 5

_SCALES = {
    "T": 1e12, 
    "B": 1e9, 
    "M": 1e6
}


def _parse_abbrev(
    val
) -> float:
    """
    Convert an abbreviated magnitude or numeric to a float.

    Accepted inputs
    ---------------
    - Numeric (int or float): returned as a float if not NaN.
    - String with optional suffix among {'T','B','M'}:
      pattern "^[0-9.,]+\\s*([TMB])?$".
        * 'T' → multiply by 1e12
        * 'B' → multiply by 1e9
        * 'M' → multiply by 1e6
      Commas are ignored (e.g., "1,234.5M" → 1_234_500_000).

    Returns
    -------
    float
        Parsed value in base units. Returns NaN if parsing fails.

    Notes
    -----
    The function is idempotent for plain numerics and robust to leading/trailing
    whitespace. Use to normalise heterogeneous spreadsheet entries into floats.
    """ 
   
    if isinstance(val, (int, float)) and not pd.isna(val):
        
        return float(val)
 
    s = str(val).strip().upper()
 
    m = re.match(r"^([0-9.,]+)\s*([TMB])?$", s)
 
    if not m:
        
        return np.nan
 
    num = float(m.group(1).replace(",", ""))
 
    return num * _SCALES.get(m.group(2), 1)


def _safe_stat(
    series: pd.Series, 
    fn, 
    default: float = 0.0
) -> float:
    """
    Apply a statistic to a Series with NaNs dropped; return a default if empty.

    Parameters
    ----------
    series : pandas.Series
        Input data possibly containing NaNs.
    fn : callable
        Function mapping a non-empty Series to a scalar (e.g., mean, median).
    default : float, default 0.0
        Value returned when the Series is empty after dropping NaNs.

    Returns
    -------
    float
        Statistic value as float, or `default` when no valid data exist.

    Rationale
    ---------
    Many sheet extracts are sparse. This helper centralises the “drop-then-stat”
    pattern and prevents exceptions on empty inputs.
    """
       
    sr = series.dropna()
   
    return float(fn(sr)) if not sr.empty else default


class FinancialForecastData:
    """
    Lazy loader and feature factory for issuer fundamentals, forecasts, KPIs,
    Prophet inputs, macro-joined regression panels, and growth diagnostics.

    Responsibilities
    ----------------
   
    - Read issuer Excel workbooks (Income TTM/Annual, Cash-flow TTM, Balance-sheet
      TTM, Ratios TTM/Annual) and expose typed, date-indexed frames.
   
    - Convert abbreviated magnitudes to floats and forward-fill sparse histories.
   
    - Apply ticker-specific currency normalisation rules for comparability.
   
    - Build an “annuals” table per ticker with cash-flow and working-capital
      components, tax rate, and interest after tax.
   
    - Ingest analyst forecasts when available or synthesise five-year revenue/EPS
      paths by compounding historical growth statistics.
   
    - Compute KPIs with exponentially weighted smoothing.
   
    - Produce Prophet-ready series (revenue and EPS).
   
    - Extract next-period forecasts and compute forecast-to-history percentage
      changes for diagnostic use.
   
    - Join annual growth and simple returns with macro series to form regression
      panels.

    Key modelling elements
    ----------------------
    - EWMA smoothing using span 20 (alpha = 2/21), providing a bias toward recent
      observations for valuation ratios.
   
    - Synthetic forecasts compound using lower/mean/upper growth triplets derived
      from interquartile-filtered annual growth series.
   
    - Returns around report dates are approximated by the percentage change in a
      2-month window mean price (one month either side of the date) to reduce
      microstructure noise.

    Attributes (selected)
    ---------------------
    tickers : list[str]
        Universe operated on by the instance.
    root_dir : pathlib.Path
        Root directory containing per-ticker workbooks.
    macro : MacroData
        Macro data provider used for joins and currency rates.
    analyst_df : pandas.DataFrame
        Analyst summary table used for direct estimates where present.
    """

    _SHEET_INCOME = "Income-TTM"
   
    _SHEET_CASHFLOW = "Cash-Flow-TTM"
   
    _SHEET_BALANCE = "Balance-Sheet-TTM"
   
    _SHEET_RATIOS = "Ratios-TTM"
   
    _SHEET_RATIOS_ANN = "Ratios-Annual"
   
    _SHEET_INCOME_ANN = "Income-Annual"

    _INCOME_ROWS = ["Revenue", "EPS (Basic)", "EBIT", "Net Income", "Income Tax",
                        "Pretax Income", "Interest Expense / Income", "EBITDA", "Depreciation & Amortization"]
    
    _CASHFLOW_ROWS = ["Operating Cash Flow", "Capital Expenditures", "Acquisitions",
                        "Share-Based Compensation", "Free Cash Flow", "Other Operating Activities", "Dividends Paid"]
    
    _BALANCE_ROWS = ["Working Capital", "Total Debt", "Cash & Cash Equivalents", "Book Value Per Share"]
   
    _RATIOS_ROWS = ["EV/Revenue", "Enterprise Value", "Market Capitalization", "PE Ratio", "PS Ratio", "PB Ratio", "EV/EBITDA", "EV/EBIT", "EV/FCF"]
   
    _RATIOS_ROWS_ANN = ["Return on Equity (ROE)", "Payout Ratio"]
   
    _INCOME_ROWS_ANN = ["Revenue Growth", "EPS Growth"]

    _FORECAST_COLS = ["num_analysts", "low_rev", "avg_rev", "high_rev", "low_eps", "avg_eps", "high_eps"]
   
    _KPI_COLS = ["mc_ev", "market_value_debt", "capex_rev_ratio", "exp_evs", "exp_pe",
                      "exp_ps", "exp_ptb", "bvps_0", "roe", "payout_ratio"]


    def __init__(
        self,
        tickers: list[str] = config.tickers,
        quiet: bool = False,
    ):
     
        self.tickers = list(tickers)
     
        self.root_dir = config.ROOT_FIN_DIR
     
        self.macro = macro
     
        self.analyst_df = macro.r.analyst
     
        self.quiet = quiet


    def currency_adjustment(
        self, 
        ticker: str, s: pd.Series | pd.DataFrame
    ):
        """
        Apply ticker-specific currency or unit adjustments to a series/frame.

        Rules
        -----
        - For tickers in {"BABA","PDD","XIACY"}: assumed local CNY input converted to
        USD using spot rate r = USDCNY; output = input / r.
        - For ticker "BP.L": assumed GBP input converted to USD using GBPUSD rate g,
        and then scaled by 1/10 (business rule to align with reporting units);
        output = input * g / 10.
        - For ticker "ASX": apply a constant multiplier 30 (heuristic placeholder):
        output = input * 30.
        - Otherwise: return input unchanged.

        Parameters
        ----------
        ticker : str
            Issuer symbol controlling the rule selection.
        s : pandas.Series or pandas.DataFrame
            Numeric data to be scaled.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Adjusted data with the same index/columns as `s`.

        Notes
        -----
        These mappings encode pragmatic normalisations based on available sheets and
        may be refined when consistent currency metadata become available.
        """       
        
        ex = self.macro.currency
        
        if ticker in ["BABA", "PDD", "XIACY"]:
           
            rate = float(ex.loc["USDCNY"])
           
            return s / rate
        
        if ticker in ["BP.L"]:
           
            rate = float(ex.loc["GBPUSD"])
           
            return s * rate / 10
        
        if ticker == "ASX":
           
            rate = 30
           
            return s * rate
        
        return s


    def _parse_growth(
        self, 
        x
    ):
        """
        Parse a growth input into a proportion in [−∞, +∞).

        Parsing logic
        -------------
        - If string ends with '%': strip percent and divide by 100.
       
        - Else if string is numeric with commas: parse and, if value > 1.0, treat as
        a percentage (divide by 100); otherwise interpret as already a proportion.
       
        - If numeric (int/float): same 'value > 1.0' rule as above.
       
        - Otherwise: return NaN.

        Returns
        -------
        float
            Growth as a decimal (e.g., 0.125 for 12.5%).
        """        
        
        if isinstance(x, str):
        
            s = x.strip()
        
            if s.endswith("%"):
        
                try:
               
                    return float(s.rstrip("%"))/100.0
               
                except ValueError:
               
                    return np.nan
        
            try:
               
                v = float(s.replace(",",""))
           
            except ValueError:
               
                return np.nan
           
            return v / 100.0 if v > 1.0 else v
        
        if isinstance(x,(int,float)):
          
            return float(x)/100.0 if x>1.0 else float(x)
        
        return np.nan


    def _filter_outliers_iqr(
        self, 
        series: pd.Series, 
        m: float = 1.5
    ) -> pd.Series:
        """
        Remove outliers using the Tukey IQR rule.

        Method
        ------
        Compute Q1 = 25th percentile and Q3 = 75th percentile; define IQR = Q3 − Q1.
        Keep values x satisfying:
            Q1 − m·IQR ≤ x ≤ Q3 + m·IQR.
        The default multiplier m = 1.5 corresponds to the standard Tukey fences.

        Parameters
        ----------
        series : pandas.Series
            Input data.
        m : float, default 1.5
            Fence width multiplier.

        Returns
        -------
        pandas.Series
            Filtered series preserving the original index for retained entries.
        """      
        
        q1, q3 = series.quantile([0.25, 0.75])
        
        iqr = q3 - q1
        
        lo,hi = q1 - (m * iqr), q3 + (m * iqr)
       
        return series[(series >= lo) & (series <= hi)]


    def _empty_forecast(
        self
    ) -> pd.DataFrame:
        """
        Construct an empty five-year forecast frame with annual December endpoints.

        Index
        -----
        Year-end dates from the next 31 December onward for `_HORIZON_YEARS + 1`
        periods.

        Columns
        -------
        ["num_analysts", "low_rev", "avg_rev", "high_rev",
        "low_eps", "avg_eps", "high_eps"] filled with NaN.

        Returns
        -------
        pandas.DataFrame
            Shape ((HORIZON_YEARS + 1) × 7) with NaN entries.
        """    
        
        dates = pd.date_range(
            start = _TODAY_TS + pd.offsets.YearEnd(1),
            periods = _HORIZON_YEARS+1,
            freq= " YE-DEC",
        )
       
        return pd.DataFrame(np.nan, index=dates, columns = self._FORECAST_COLS)


    def _empty_kpis(
        self
    ) -> pd.DataFrame:
        """
        Construct a one-row KPI frame initialised with NaNs at today's date.

        Returns
        -------
        pandas.DataFrame
            Index [TODAY], columns matching `_KPI_COLS`, values NaN.
        """
        
        return pd.DataFrame(np.nan, index=[_TODAY_TS], columns = self._KPI_COLS)


    def _read_sheet(
        self, 
        tkr: str, 
        sheet: str, 
        rows: list[str]
    ) -> pd.DataFrame:
        """
        Load a single worksheet for a ticker, coerce date columns, and reindex rows.

        Steps
        -----
        1) Open "{tkr}-financials.xlsx" from the ticker folder.
        2) Read `sheet` using 'openpyxl'; treat ['—','-',''] as NaN.
        3) Coerce column labels to pandas Timestamps (NaT if invalid).
        4) Reindex to `rows` and sort columns ascending by date.

        Parameters
        ----------
        tkr : str
            Ticker symbol.
        sheet : str
            Worksheet name.
        rows : list[str]
            Expected row order for the resulting frame.

        Returns
        -------
        pandas.DataFrame
            DataFrame reindexed to requested rows with chronological columns.
        """
        
        folder = self.root_dir / tkr
        
        xlsx = folder / f"{tkr}-financials.xlsx"
        xls = pd.ExcelFile(xlsx, engine="openpyxl")
        
        df = pd.read_excel(
            xls, 
            sheet_name = sheet, 
            index_col=0,
            na_values=["—","-",""]
        )
        
        df.columns = pd.to_datetime(df.columns, errors="coerce")
        
        return df.reindex(rows).reindex(sorted(df.columns), axis=1)


    @cached_property
    def income(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Income TTM DataFrame.

        Each DataFrame has date columns and rows defined by `_INCOME_ROWS`, with
        magnitudes as found in the workbook (not yet currency-adjusted).
        Failed loads yield an empty DataFrame with a warning when `quiet` is False.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Income-TTM frame (possibly empty).
        """
        
        d = {}
       
        for t in self.tickers:
       
            try:
               
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_INCOME, 
                    rows = self._INCOME_ROWS
                )
            
            except Exception as e:
            
                if not self.quiet: 
            
                    print(f"[WARN] income {t}: {e}")
               
                d[t] = pd.DataFrame()
      
        return d


    @cached_property
    def cash(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Cash-flow TTM DataFrame.

        Rows correspond to `_CASHFLOW_ROWS`. Missing or unreadable sheets produce an
        empty DataFrame entry with an optional warning.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Cash-flow-TTM frame (possibly empty).
        """
        
        d = {}

        for t in self.tickers:

            try:
        
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_CASHFLOW, 
                    rows = self._CASHFLOW_ROWS
                )
        
            except Exception as e:
        
                if not self.quiet: 
        
                    print(f"[WARN] cash {t}: {e}")
                
                d[t] = pd.DataFrame()

        return d


    @cached_property
    def bal(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Balance-sheet TTM DataFrame.

        Rows correspond to `_BALANCE_ROWS`. On failure, returns an empty DataFrame
        for the ticker and optionally logs a warning.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Balance-sheet-TTM frame (possibly empty).
        """

        d = {}

        for t in self.tickers:

            try:
        
                d[t] = self._read_sheet(t, self._SHEET_BALANCE, self._BALANCE_ROWS)
        
            except Exception as e:
        
                if not self.quiet: 
        
                    print(f"[WARN] bal {t}: {e}")

                d[t] = pd.DataFrame()

        return d


    @cached_property
    def ratios(self) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Ratios TTM DataFrame.

        Rows include valuation and coverage ratios listed in `_RATIOS_ROWS`. Values
        are raw as read; use `_parse_abbrev` where necessary to obtain floats.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Ratios-TTM frame (possibly empty).
        """   
        
        d = {}
       
        for t in self.tickers:
       
            try:
      
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_RATIOS, 
                    rows = self._RATIOS_ROWS
                )
      
            except Exception as e:
      
                if not self.quiet: 
      
                    print(f"[WARN] ratios {t}: {e}")
               
                d[t] = pd.DataFrame()
     
        return d


    @cached_property
    def ratios_ann(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Ratios Annual DataFrame.

        Typically includes "Return on Equity (ROE)" and "Payout Ratio" across years.
        Values may be abbreviations or percentages and require parsing downstream.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Ratios-Annual frame (possibly empty).
        """
        
        d = {}
       
        for t in self.tickers:
       
            try:
        
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_RATIOS_ANN, 
                    rows = self._RATIOS_ROWS_ANN
                )
        
            except Exception as e:
        
                if not self.quiet: 
        
                    print(f"[WARN] ratios_ann {t}: {e}")
               
                d[t] = pd.DataFrame()
     
        return d


    @cached_property
    def income_ann(self) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Income Annual DataFrame.

        Rows are `_INCOME_ROWS_ANN` (e.g., "Revenue Growth", "EPS Growth").
        Values are parsed to floats downstream via `_parse_growth` and `_parse_abbrev`.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Income-Annual frame (possibly empty).
        """
        
        d = {}

        for t in self.tickers:

            try:
       
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_INCOME_ANN, 
                    rows = self._INCOME_ROWS_ANN
                )
       
            except Exception as e:
       
                if not self.quiet: 
       
                    print(f"[WARN] income_ann {t}: {e}")
               
                d[t] = pd.DataFrame()
      
        return d


    @cached_property
    def annuals(self) -> dict[str,pd.DataFrame]:
        """
        Build per-ticker annual aggregates with currency normalisation and derived fields.

        Construction
        ------------
        For each ticker:
       
        1) Map TTM sheets through `_parse_abbrev`, forward-fill across columns, then
            fill NaNs with zero; convert to common currency using `currency_adjustment`.
       
        2) Derive:
            - TaxRate = IncomeTax / PretaxIncome (guarding PretaxIncome = 0 → NaN).
            - InterestAfterTax = InterestExpense × (1 − TaxRate).
            - Change in Working Capital = WorkingCapital_t − WorkingCapital_{t−1}.
            - NetBorrowing = TotalDebt_t − TotalDebt_{t−1}, reindexed to OCF index.
       
        3) Assemble columns including Revenue, EPS, OCF, Capex, Acquisitions (as a
            positive cash outflow), FCF, EBITDA, EBIT, Net Income, Total Debt, SBC,
            Other Operating Activities, Dividends, PretaxIncome, Depreciation &
            Amortisation, InterestAfterTax, Change in Working Capital, NetBorrowing.
       
        4) Drop rows all-NaN.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → annual aggregate table aligned on reporting dates.

        Notes
        -----
        EPS is coerced via `pd.to_numeric(..., errors="coerce")` before adjustment.
        """   
        
        out = {}
       
        for t in self.tickers:
       
            try:
         
                inc = self.income[t].map(_parse_abbrev).ffill(axis = 1).fillna(0)
         
                cf = self.cash[t].map(_parse_abbrev).ffill(axis = 1).fillna(0)
         
                bal = self.bal[t].map(_parse_abbrev).ffill(axis = 1).fillna(0)

                rev_hist = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Revenue"]
                )
         
                eps_hist = self.currency_adjustment(
                    ticker = t,
                    s = pd.to_numeric(
                        self.income[t].loc["EPS (Basic)"], errors="coerce"
                    ).ffill().fillna(0)
                )

                ocf = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Operating Cash Flow"]
                )
         
                capex = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Capital Expenditures"]
                )
         
                aq = -self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Acquisitions"]
                )
         
                fcf = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Free Cash Flow"]
                )
         
                sbc = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Share-Based Compensation"]
                )
         
                ooa = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Other Operating Activities"]
                )
         
                div = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Dividends Paid"]
                )

                wc = self.currency_adjustment(
                    ticker = t, 
                    s = bal.loc["Working Capital"]
                )
         
                tot_d = self.currency_adjustment(
                    ticker = t, 
                    s = bal.loc["Total Debt"]
                )
         
                bvps_0 = self.currency_adjustment(
                    ticker = t, 
                    s = bal.loc["Book Value Per Share"].iat[-1]
                )

                ebit = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["EBIT"]
                )
         
                da = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Depreciation & Amortization"]
                )
         
                ni = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Net Income"]
                )
         
                ebitda = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["EBITDA"]
                )
         
                tax = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Income Tax"]
                )
         
                pretax = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Pretax Income"]
                )
         
                iexp = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Interest Expense / Income"]
                )

                tax_rate = tax / pretax.replace(0, np.nan)
         
                int_at = iexp * (1 - tax_rate)
         
                d_wc = wc.diff()
         
                net_borrow = tot_d.diff().reindex(ocf.index, fill_value = 0)

                df = pd.DataFrame({
                    "Revenue": rev_hist,
                    "EPS": eps_hist,
                    "OCF": ocf,
                    "PretaxIncome": pretax,
                    "InterestAfterTax": int_at,
                    "NetBorrowing": net_borrow,
                    "Capex": capex,
                    "EBIT": ebit,
                    "Depreciation & Amortization": da,
                    "Share-Based Compensation": sbc,
                    "Acquisitions": aq,
                    "Net Income": ni,
                    "EBITDA": ebitda,
                    "Total Debt": tot_d,
                    "Change in Working Capital": d_wc,
                    "Other Operating Activities": ooa,
                    "FCF": fcf,
                    "Div": div,
                }).T.dropna(how="all").T

                out[t] = df

            except Exception as e:
          
                if not self.quiet: 
          
                    print(f"[WARN] annuals {t}: {e}")
               
                out[t] = pd.DataFrame()
      
        return out


    def _read_analyst_sheet(
        self, 
        path: Path, 
        tkr: str
    ) -> pd.DataFrame:
     
        df = pd.read_excel(
            path, 
            index_col = 0, 
            engine = "openpyxl"
        )
        """
        Read a per-ticker analyst workbook and return a forecast panel.

        Inputs
        ------
        Expected rows (index) include "Period Ending", "Analysts",
        revenue triplet {"Revenue Low","Revenue","Revenue High"} and
        EPS triplet {"EPS Low","EPS","EPS High"}.

        Processing
        ----------
        - Columns are normalised to strings and dates parsed from "Period Ending".
        - Revenue entries are parsed via `_parse_abbrev` and currency-adjusted.
        - EPS entries are numerically coerced and currency-adjusted.
        - A DataFrame is returned with columns:
            ["num_analysts","low_rev","avg_rev","high_rev",
            "low_eps","avg_eps","high_eps"] and date index.

        Returns
        -------
        pandas.DataFrame
            Chronologically sorted analyst forecast panel.
        """
 
        df.columns = [str(c).strip() for c in df.columns]
     
        dates = pd.to_datetime(df.loc["Period Ending"].dropna())
     
        data = {
            "num_analysts": df.loc["Analysts"].astype(int).values,
            "low_rev": self.currency_adjustment(
                ticker = tkr, 
                series = df.loc["Revenue Low"].apply(_parse_abbrev)
            ).values,
            "avg_rev": self.currency_adjustment(
                ticker = tkr, 
                series = df.loc["Revenue"].apply(_parse_abbrev)
            ).values,
            "high_rev": self.currency_adjustment(
                ticker = tkr, 
                series = df.loc["Revenue High"].apply(_parse_abbrev)
            ).values,
            "low_eps": self.currency_adjustment(
                ticker = tkr, 
                series = pd.to_numeric(df.loc["EPS Low"], errors = "coerce")
            ).values,
            "avg_eps": self.currency_adjustment(
                ticker = tkr,
                series = pd.to_numeric(df.loc["EPS"], errors = "coerce")
            ).values,
            "high_eps": self.currency_adjustment(
                ticker = tkr, 
                series = pd.to_numeric(df.loc["EPS High"], errors = "coerce")
            ).values,
        }
     
        return pd.DataFrame(data, index=dates).sort_index()


    def _make_synthetic_forecast(
        self,
        tkr: str,
        rev_hist: pd.Series,
        eps_hist: pd.Series,
        income_ann_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create a five-year synthetic revenue/EPS forecast by compounding filtered growth.

        Method
        ------
       
        1) Parse annual growth rows ("Revenue Growth","EPS Growth") to proportions,
        forward-fill across years, and replace missing with zero.
       
        2) Filter extreme growth observations using the IQR rule.
       
        3) Compute triplets for revenue growth g_rev = (min, mean, max) and for EPS
        growth g_eps = (min, mean, max).
       
        4) Determine base levels:
            - If analyst one-period estimates are present in `analyst_df` for `tkr`,
            use Low/Avg/High for both revenue and EPS.
            - Otherwise, use the last observed historical values.
       
        5) For horizon steps p = 1..H:
            - For p = 1: carry forward base values (no compounding in the first stub).
            - For p ≥ 2: compound from the previous average level:
                avg_rev_p = avg_rev_{p−1} × (1 + mean(g_rev))
                low_rev_p = avg_rev_{p−1} × (1 + min(g_rev))
                high_rev_p = avg_rev_{p−1} × (1 + max(g_rev))
            and analogously for EPS using g_eps.

        Output
        ------
        Index: year-end dates for H + 1 periods starting next year.
        Columns: num_analysts (set to 1), and the six revenue/EPS forecast columns.

        Returns
        -------
        pandas.DataFrame
            Synthetic forecast panel suitable for joining with KPI logic.
        """
        
        ann_num = income_ann_df.map(self._parse_growth).ffill(axis=1).fillna(0)

        rev_g = self._filter_outliers_iqr(
            series = ann_num.loc["Revenue Growth"].dropna()
        )
        
        eps_g = self._filter_outliers_iqr(
            series = ann_num.loc["EPS Growth"].dropna()
        )

        g_rev = (rev_g.min(), rev_g.mean(), rev_g.max())
     
        g_eps = (eps_g.min(), eps_g.mean(), eps_g.max())

        if tkr in self.analyst_df.index:

            base = self.analyst_df.loc[tkr]

            br, ar, hr = (base[c] for c in ["Low Revenue Estimate", "Avg Revenue Estimate", "High Revenue Estimate"])
          
            br, ar, hr = map(_parse_abbrev, [br, ar ,hr])
          
            be, ae, he = (base[c] for c in ["Low EPS Estimate", "Avg EPS Estimate", "High EPS Estimate"])
          
            be, ae, he = map(lambda x: pd.to_numeric(x, errors="coerce"), [be, ae, he])

        else:

            br = ar = hr = rev_hist.iloc[-1]
          
            be = ae = he = eps_hist.iloc[-1]

        low_r, avg_r, high_r = [ [br], [ar], [hr] ]
        
        low_e, avg_e, high_e = [ [be], [ae], [he] ]

        for p in range(1, _HORIZON_YEARS+1):

            if p==1:

                for lst in (low_r, avg_r, high_r, low_e, avg_e, high_e):
                    lst.append(lst[-1])

            else:
               
                low_r.append(avg_r[-1] * (1 + g_rev[0]))
             
                avg_r.append(avg_r[-1] * (1 + g_rev[1]))
             
                high_r.append(avg_r[-1] * (1 + g_rev[2]))
               
                low_e.append(avg_e[-1] * (1 + g_eps[0]))
             
                avg_e.append(avg_e[-1] * (1 + g_eps[1]))
             
                high_e.append(avg_e[-1] * (1 + g_eps[2]))

        dates = pd.date_range(
            start = _TODAY_TS + pd.offsets.YearEnd(1),
            periods = _HORIZON_YEARS+1,
            freq = "YE-DEC",
        )

        return pd.DataFrame({
            "num_analysts": [1]*len(dates),
            "low_rev": np.array(low_r),
            "avg_rev": np.array(avg_r),
            "high_rev": np.array(high_r),
            "low_eps": np.array(low_e),
            "avg_eps": np.array(avg_e),
            "high_eps": np.array(high_e),
        }, index = dates)


    @cached_property
    def forecast(self) -> dict[str,pd.DataFrame]:
        """
        Map each ticker to its forecast panel, preferring analyst sheets over synthesis.

        Resolution
        ----------
        - If "{t}-analyst-forecasts.xlsx" exists, parse via `_read_analyst_sheet`.
        - Else, synthesise via `_make_synthetic_forecast` using historical series.

        Failure mode
        ------------
        On exception, produce `_empty_forecast()` and optionally warn.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → forecast DataFrame with seven columns across H + 1 periods.
        """
        
        out = {}

        for t in self.tickers:

            try:
                fpath = self.root_dir / t / f"{t}-analyst-forecasts.xlsx"

                if fpath.exists():
                   
                    df = self._read_analyst_sheet(
                        path = fpath, 
                        tkr = t
                    )

                else:
                    ann_df = self.income_ann[t]
                  
                    rev = self.annuals[t]["Revenue"]
                  
                    eps = self.annuals[t]["EPS"]
                  
                    df = self._make_synthetic_forecast(
                        tkr = t, 
                        rev_hist = rev, 
                        eps_hist = eps, 
                        income_ann_df = ann_df
                    )

                out[t] = df

            except Exception as e:

                if not self.quiet: 
                    
                    print(f"[WARN] forecast {t}: {e}")
                
                out[t] = self._empty_forecast()
        
        return out


    @cached_property
    def kpis(self) -> dict[str,pd.DataFrame]:
        """
        Compute smoothed valuation KPIs and capital structure quantities per ticker.

        Metrics (smoothed with span=20 unless noted)
        --------------------------------------------
        - exp_evs       : EWMA of EV/Revenue.
        - exp_pe        : EWMA of PE Ratio.
        - exp_ps        : EWMA of PS Ratio.
        - exp_ptb       : EWMA of PB Ratio.
        - exp_evfcf     : EWMA of EV/FCF.
        - exp_evebitda  : EWMA of EV/EBITDA.
        - exp_evebit    : EWMA of EV/EBIT.
        - mc_ev         : EWMA of MarketCapitalisation / EnterpriseValue.
        - eve_t         : mc_ev × exp_pe (a proxy for expected earnings yield scaled
                        by capital structure mix).
        - market_value_debt (md): last MarketCapitalisation − last EnterpriseValue.
        - capex_rev_ratio: mean over last 20 observations of Capex / Revenue.
        - bvps_0        : last Book Value Per Share.
        - roe           : geometric mean of last five annual ROE values:
                        if ROE_i denotes each proportion, then
                        ROE_g = (product over i of (1 + ROE_i))^(1/n) − 1.
        - payout_ratio  : mean absolute payout ratio over the last five annual values.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → one-row KPI DataFrame indexed by the last annual observation date.

        Notes
        -----
        EWMA recursion uses alpha = 2/(1 + 20) = 2/21. Missing values are coerced to
        NaN and handled by pandas EWMs. Where ROE data are absent, an optional
        industry fallback may be used if available.
        """       
        
        out = {}
       
        for t in self.tickers:
       
            try:
               
                ratios_df = self.ratios[t]
               
                capex = self.cash[t].map(_parse_abbrev).loc["Capital Expenditures"]
               
                rev_hist = self.annuals[t]["Revenue"]
               
                last_idx = self.annuals[t].index[-1]
               
                bvps0 = self.bal[t].map(_parse_abbrev).loc["Book Value Per Share"].iat[-1]
               
                out[t] = self._compute_kpis(t, ratios_df, capex, rev_hist, last_idx, bvps0)
       
            except Exception as e:
               
                if not self.quiet: 
               
                    print(f"[WARN] kpis {t}: {e}")
                
                out[t] = self._empty_kpis()
     
        return out

    
    def _compute_kpis(
        self, 
        tkr, 
        ratios_df, 
        capex, rev, 
        annual_index, 
        bvps_0
    ) -> pd.DataFrame:
        """
        Internal KPI constructor using smoothed ratio time series and balance items.

        Inputs
        ------
        ratios_df : DataFrame with rows including
            {"EV/Revenue","PE Ratio","PS Ratio","PB Ratio","EV/FCF","EV/EBITDA","EV/EBIT",
            "Market Capitalization","Enterprise Value"}.
        capex, rev : Series aligned in time to compute Capex/Revenue.
        annual_index : Timestamp
            Index label for the resulting single-row KPI frame.
        bvps_0 : float
            Book Value Per Share at the latest date.

        Computations
        ------------
        - EWMA smoothing of valuation ratios with span 20.
        - mc_ev = EWMA( MarketCap / EnterpriseValue ), guarding infinities.
        - eve_t = mc_ev × exp_pe.
        - capex_rev_ratio = mean over last 20 valid points of Capex / Revenue.
        - market_value_debt = last MarketCap − last EnterpriseValue.
        - roe = geometric mean of last five annual ROE observations (if any).
        - payout_ratio = mean absolute payout ratio over last five annual values.

        Returns
        -------
        pandas.DataFrame
            Single-row KPI frame indexed by `annual_index`.
        """
    
        ev_rev = pd.to_numeric(ratios_df.loc["EV/Revenue"], errors = "coerce")
       
        evs_t = ev_rev.ewm(span = 20, adjust=False).mean().iat[-1]
       
        pe = pd.to_numeric(ratios_df.loc["PE Ratio"], errors = "coerce")
       
        ps = pd.to_numeric(ratios_df.loc["PS Ratio"], errors = "coerce")
       
        ptb = pd.to_numeric(ratios_df.loc["PB Ratio"], errors = "coerce")
        
        evfcf = pd.to_numeric(ratios_df.loc["EV/FCF"], errors = "coerce")
        
        evebitda = pd.to_numeric(ratios_df.loc["EV/EBITDA"], errors = "coerce")
        
        evebit = pd.to_numeric(ratios_df.loc["EV/EBIT"], errors = "coerce")
       
        pe_a = pe.ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]
       
        ps_a = ps.ewm(
            span = 20,
            adjust = False
        ).mean().iat[-1]
       
        ptb_a = ptb.ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]
        
        evfcf_a = evfcf.ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]
        
        evebitda_a = evebitda.ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]
        
        evebit_a = evebit.ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]

        mc = ratios_df.loc["Market Capitalization"].apply(_parse_abbrev)
       
        ev = ratios_df.loc["Enterprise Value"].apply(_parse_abbrev)
       
        mc_ev = (mc / ev).replace([np.inf, -np.inf], np.nan).ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]
        
        eve_t = mc_ev * pe_a

        cap_rev = (capex / rev).tail(20).mean()
        md = mc.iat[-1] - ev.iat[-1]

        roe_list = (
            self.ratios_ann[tkr]
                .loc["Return on Equity (ROE)"]
                .map(_parse_abbrev)
                .dropna().tail(5)
        )
        
        if not roe_list.empty:
           
            roe = (roe_list+1).prod() ** (1 / len(roe_list)) - 1
       
        else:
       
            roe = self.ind_dict["ROE"][tkr]["Region-Industry"] or 0.0

        payout = abs(
            self.ratios_ann[tkr]
                .loc["Payout Ratio"]
                .map(_parse_abbrev)
                .dropna()
                .tail(5)
                .mean()
        )

        return pd.DataFrame({
            "mc_ev": [mc_ev],
            "market_value_debt": [md],
            "capex_rev_ratio": [cap_rev],
            "exp_evs": [evs_t],
            "exp_pe": [pe_a],
            "exp_ps": [ps_a],
            "exp_ptb": [ptb_a],
            "bvps_0": [bvps_0],
            "roe": [roe],
            "payout_ratio": [payout],
            "exp_evfcf": [evfcf_a],
            "exp_evebitda": [evebitda_a],
            "exp_evebit": [evebit_a],
            "eve_t": [eve_t],
        }, index = [annual_index])


    @cached_property
    def prophet_data(self) -> dict[str,pd.DataFrame]:
        """
        Provide Prophet-ready per-ticker time series for revenue and EPS.

        Output format
        -------------
        For each ticker with available income data since 2000-01-01:
            DataFrame with columns:
                - "rev": currency-adjusted, forward-filled Revenue series
                - "eps": currency-adjusted, numeric-coerced, forward-filled EPS
        If prerequisites are missing, returns an empty DataFrame for that ticker.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Prophet-ready series (or empty).
        """       
        
        out = {}
       
        for t in self.tickers:
       
            try:
       
                inc = self.income.get(t, pd.DataFrame())
       
                if inc.empty:
                  
                    out[t] = pd.DataFrame()
                  
                    continue

                inc.columns = pd.to_datetime(inc.columns)
       
                inc2 = inc.loc[:, inc.columns >= "2000-01-01"]
       
                if not {"Revenue","EPS (Basic)"}.issubset(inc2.index):
                  
                    out[t] = pd.DataFrame()
                  
                    continue

                rev = self.currency_adjustment(
                    ticker = t, 
                    s = inc2.loc["Revenue"].ffill().fillna(0)
                )
              
                eps = self.currency_adjustment(
                    ticker = t, 
                    s = pd.to_numeric(inc2.loc["EPS (Basic)"], errors="coerce").ffill().fillna(0)
                )
       
                out[t] = pd.DataFrame({"rev":rev, "eps":eps})
       
            except Exception as e:
       
                if not self.quiet: 
                    print(f"[WARN] prophet_data {t}: {e}")
                
                out[t] = pd.DataFrame()
       
        return out


    def next_period_forecast(
        self
    ) -> pd.DataFrame:
        """
        Assemble next-period analyst estimates and the first forecast row per ticker.

        Components
        ----------
        - Analyst “_y” columns (per ticker index in `analyst_df`):
            low_rev_y, avg_rev_y, high_rev_y,
            low_eps_y, avg_eps_y, high_eps_y,
            num_analysts_y.
        - First future row from `forecast[t]` mapped into
            ["num_analysts","low_rev","avg_rev","high_rev","low_eps","avg_eps","high_eps"].

        Join logic
        ----------
        Outer join on ticker index to retain all tickers observed in either source.

        Returns
        -------
        pandas.DataFrame
            Ticker-indexed table combining analyst one-period estimates and the first
            row of the internally stored forecast panel.
        """
        
        est = pd.DataFrame({
            "low_rev_y": self.analyst_df["Low Revenue Estimate"].apply(_parse_abbrev),
            "avg_rev_y": self.analyst_df["Avg Revenue Estimate"].apply(_parse_abbrev),
            "high_rev_y": self.analyst_df["High Revenue Estimate"].apply(_parse_abbrev),
            "low_eps_y": pd.to_numeric(self.analyst_df["Low EPS Estimate"], errors = "coerce"),
            "avg_eps_y": pd.to_numeric(self.analyst_df["Avg EPS Estimate"], errors = "coerce"),
            "high_eps_y": pd.to_numeric(self.analyst_df["High EPS Estimate"], errors = "coerce"),
            "num_analysts_y": self.analyst_df["numberOfAnalystOpinions"].astype(float)
        }, index = self.analyst_df.index)
       
        first_fc = {
            t: (
                {c: df.iloc[0].get(c, np.nan) for c in self._FORECAST_COLS}
                if not df.empty else dict.fromkeys(self._FORECAST_COLS, np.nan)
            )
            for t,df in self.forecast.items()
        }
      
        first_df = pd.DataFrame.from_dict(first_fc, orient = "index")
      
        return est.join(first_df, how = "outer")
    
    
    def get_forecast_pct_changes(
        self, 
        ticker: str
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute percentage changes of revenue and EPS forecasts relative to history.

        Definitions
        -----------
        Let B_rev be the last historical Revenue; B_eps the last historical EPS.
        For any new estimate N (analyst or first forecast), define:
            pct_change = (N − B) / |B|   if B ≠ 0 and N is not NaN,
                        0              otherwise.

        Outputs
        -------
        rev_covs : dict[str, float]
            Keys: {"low_y","avg_y","high_y","low_fs","avg_fs","high_fs"}
            where *_y are analyst one-period estimates and *_fs are first-row
            internal forecasts. Values are percentage changes vs B_rev.
        eps_covs : dict[str, float]
            Same keys as above; percentage changes vs B_eps.

        Returns
        -------
        (rev_covs, eps_covs) : tuple of dictionaries.

        Notes
        -----
        Guards against division by zero or missing data by returning 0 in such cases.
        """

        def _zero_covs():
            
            return {k: 0.0 for k in ("low_y", "avg_y", "high_y", "low_fs", "avg_fs", "high_fs")}


        if ticker not in self.annuals or self.annuals[ticker].empty:
           
            return _zero_covs(), _zero_covs()

        hist = self.annuals[ticker]
       
        base_rev = hist["Revenue"].iloc[-1]
       
        base_eps = hist["EPS"].iloc[-1]

        if self.analyst_df is not None and ticker in self.analyst_df.index:
            
            est = pd.Series({
                "low_rev_y": _parse_abbrev(
                    val = self.analyst_df.at[ticker, "Low Revenue Estimate"]
                ),
                "avg_rev_y": _parse_abbrev(
                    val = self.analyst_df.at[ticker, "Avg Revenue Estimate"]
                ),
                "high_rev_y": _parse_abbrev(
                    val = self.analyst_df.at[ticker, "High Revenue Estimate"]
                ),
                "low_eps_y": pd.to_numeric(self.analyst_df.at[ticker, "Low EPS Estimate"], errors = "coerce"),
                "avg_eps_y": pd.to_numeric(self.analyst_df.at[ticker, "Avg EPS Estimate"], errors = "coerce"),
                "high_eps_y": pd.to_numeric(self.analyst_df.at[ticker, "High EPS Estimate"], errors = "coerce"),
            })
       
        else:
       
            est = pd.Series(index = ["low_rev_y", "avg_rev_y", "high_rev_y", "low_eps_y", "avg_eps_y", "high_eps_y"], dtype = float)

        first_fc = self.forecast.get(ticker, pd.DataFrame())

        if first_fc is not None and not first_fc.empty:

            first = first_fc.iloc[0]

        else:

            first = pd.Series(index = ["low_rev", "avg_rev", "high_rev", "low_eps", "avg_eps", "high_eps"], dtype = float)


        def pct(
            new,
            old
        ):
            
            return (new - old) / abs(old) if pd.notna(new) and old not in (0, 0.0) else 0.0


        rev_covs = {
            "low_y": pct(
                new = est.get("low_rev_y"), 
                old = base_rev
            ),
            "avg_y": pct(
                new = est.get("avg_rev_y"), 
                old = base_rev
            ),
            "high_y": pct(
                new = est.get("high_rev_y"), 
                old = base_rev
            ),
            "low_fs": pct(
                new = first.get("low_rev"), 
                old = base_rev
            ),
            "avg_fs": pct(
                new = first.get("avg_rev"), 
                old = base_rev
            ),
            "high_fs": pct(
                new = first.get("high_rev"), 
                old = base_rev
            ),
        }
        
        eps_covs = {
            "low_y": pct(
                new = est.get("low_eps_y"), 
                old = base_eps
            ),
            "avg_y": pct(
                new = est.get("avg_eps_y"), 
                old = base_eps
            ),
            "high_y": pct(
                new = est.get("high_eps_y"), 
                old = base_eps
            ),
            "low_fs": pct(
                new = first.get("low_eps"), 
                old = base_eps
            ),
            "avg_fs": pct(
                new = first.get("avg_eps"), 
                old = base_eps
            ),
            "high_fs": pct(
                new = first.get("high_eps"), 
                old = base_eps
            ),
        }
        
        return rev_covs, eps_covs
    
    
    def regression_dict(
        self
    ) -> dict[str, pd.DataFrame]:
        """
        Construct per-ticker panels combining annual growth, price returns, and macro history.

        Pipeline
        --------
       
        1) Obtain macro history from `macro.assign_macro_history()`; this is expected
        to provide a MultiIndex with level 'ticker' and annual period index.
       
        2) Compute annual income growth via `get_annual_income_growth()`, yielding
        per-ticker DataFrames with "Revenue Growth", "EPS Growth", and "Return".
       
        3) For each ticker, join on annual period with macro subframe for that ticker
        and select the following columns:
            - "Revenue Growth", "EPS Growth", "Return"
            - "InterestRate_pct_change", "Inflation_rate", "GDP_growth",
            "Unemployment_pct_change"

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → joined panel suitable for downstream regression modelling.

        Interpretation
        --------------
        The panel is designed for reduced-form regressions linking firm growth and
        simple returns to macro covariates; it does not impose structural dynamics.
        """
    
        macro_hist = self.macro.assign_macro_history()
       
        growth_dict = self.get_annual_income_growth()
       
        combined: dict[str, pd.DataFrame] = {}

        for tkr, df_growth in growth_dict.items():
    
            if df_growth.empty:
       
                continue 

            df = df_growth.copy()
       
            df.index = pd.to_datetime(df.index).to_period("Y")

            try:
       
                macro_tkr = macro_hist.xs(tkr, level="ticker")
       
            except KeyError:
       
                continue

            joined = df.join(macro_tkr, how="inner")
    
            combined[tkr] = joined[[
                "Revenue Growth", 
                "EPS Growth", 
                "Return",
                "InterestRate_pct_change",
                "Inflation_rate", 
                "GDP_growth", 
                "Unemployment_pct_change"
            ]]

        return combined
    
    
    def get_annual_income_growth(
        self
    ) -> dict[str, pd.DataFrame]:
        """
        Derive annual revenue/EPS growth and approximate returns around report dates.

        Steps
        -----
       
        1) For each ticker with an Income Annual sheet:
            - Parse numeric values via `_parse_abbrev` and forward-fill across years.
            - Extract "Revenue Growth" and "EPS Growth" rows as proportions.
       
        2) Construct a mean price series around each annual date using a symmetric
        window of ±1 month:
            mean_price(date) = average of daily closes in [date − 1M, date + 1M].
       
        3) Define an annual return proxy as:
            Return_t = mean_price_t / mean_price_{t−1} − 1,
        i.e., the percentage change in the windowed mean price.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → DataFrame with columns {"Revenue Growth","EPS Growth","Return"}
            indexed by annual period, with NaNs dropped.

        Notes
        -----
        The windowed price mean reduces sensitivity to day-specific noise or
        announcement-day jumps. Growth rows are assumed to be on an annual basis in
        the workbook, expressed as percentages or proportions, and are parsed to
        decimals before use.
        """
    
        growth: dict[str, pd.DataFrame] = {}
     
        window = pd.DateOffset(months = 1)

        for tkr in self.tickers:
    
            income_ann_df = self.income_ann.get(tkr)
    
            if income_ann_df is None or income_ann_df.empty:
     
                growth[tkr] = pd.DataFrame() 
     
                continue

            ann_num = income_ann_df.map(_parse_abbrev).ffill(axis = 1).fillna(0)
    
            rev_g = ann_num.loc["Revenue Growth"]
     
            eps_g = ann_num.loc["EPS Growth"]

            closes = macro.r.close[tkr].sort_index()
    
    
            def window_mean(
                dt: pd.Timestamp
            ) -> float:
     
                return closes.loc[dt - window : dt + window].mean()


            mean_price = pd.Series(
                {dt: window_mean(dt) for dt in rev_g.index}
            ).sort_index()
    
            ret_g = mean_price.pct_change()

            df = pd.DataFrame({
                "Revenue Growth": rev_g,
                "EPS Growth": eps_g,
                "Return": ret_g,
            }).dropna()
    
            growth[tkr] = df

        return growth    
            
    
