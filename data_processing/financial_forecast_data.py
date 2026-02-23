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
from typing import Final, Sequence, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from functools import cached_property
from data_processing.macro_data import MacroData
import config

import os
import pickle
import tempfile
import zlib
import time
from contextlib import contextmanager

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

    _INCOME_ROWS = ["Revenue", "EPS (Basic)", "EBIT", "Net Income", "Income Tax", "Pretax Income", "Effective Tax Rate", "Interest Expense / Income", "EBITDA", "Depreciation & Amortization"]
    
    _CASHFLOW_ROWS = ["Operating Cash Flow", "Capital Expenditures", "Acquisitions", "Share-Based Compensation", "Free Cash Flow", "Other Operating Activities", "Dividends Paid"]
    
    _BALANCE_ROWS = ["Working Capital", "Total Debt", "Cash & Cash Equivalents", "Book Value Per Share", "Total Assets", "Short-TermInvestments", "Long-Term Investments", "Goodwill and Intangibles", "Total Liabilities", "Current Debt", "Long Term Debt"]
   
    _RATIOS_ROWS = ["EV/Revenue", "Enterprise Value", "Market Capitalization", "PE Ratio", "PS Ratio", "PB Ratio", "EV/EBITDA", "EV/EBIT", "EV/FCF", "Payout Ratio", "Return on Equity (ROE)", "Return on Assets (ROA)", "Debt/Equity"]
   
    _RATIOS_ROWS_ANN = ["Return on Equity (ROE)", "Payout Ratio"]
   
    _INCOME_ROWS_ANN = ["Revenue Growth", "EPS Growth"]

    _FORECAST_COLS = ["num_analysts", "low_rev", "avg_rev", "high_rev", "low_eps", "avg_eps", "high_eps"]
   
    _KPI_COLS = ["mc_ev", "market_value_debt", "capex_rev_ratio", "exp_evs", "exp_pe", "exp_ps", "exp_ptb", "bvps_0", "roe", "payout_ratio", "exp_evfcf", "exp_evebitda", "exp_evebit", "eve_t", "d_to_e_mean", "tax_rate_avg"]

    _CACHE_SUBDIR = "_ffdata_daily_cache"
    
    _CACHE_VERSION = 1
    
    _CACHE_LOCK_TIMEOUT_SEC = 300
    
    _CACHE_LOCK_POLL_SEC = 0.25

    _FUND_EXPOS_CACHE_VERSION = 1


    def __init__(
        self,
        tickers: list[str] = config.tickers,
        quiet: bool = False,
    ):
        """
        Initialise the financial forecast data engine and prime the daily in-memory cache.

        Parameters
        ----------
        tickers : list[str], default `config.tickers`
            Universe of issuer symbols to process. The order supplied here becomes
            the canonical order used by cached properties and panel constructors.
        quiet : bool, default False
            Controls warning verbosity. When `True`, recoverable ingestion and parsing
            failures are suppressed; when `False`, warnings are printed to aid
            diagnostics.

        Initialisation sequence
        -----------------------
        1. Store the ticker universe as a concrete list.
      
        2. Store filesystem root (`config.ROOT_FIN_DIR`) and shared macro provider.
      
        3. Snapshot analyst summary data from `macro.r.analyst`.
      
        4. Build or load the daily cache via `_init_daily_excel_cache()`.

        Side effects
        ------------
        - May create cache directories and cache files under `FFDATA_CACHE_DIR` (or
          `_ffdata_daily_cache` beneath the financial root).
      
        - May read a large set of Excel workbooks when cache misses occur.

        Notes
        -----
        Cache priming at construction time converts later feature calls into mostly
        in-memory operations, reducing repeated workbook parsing and lowering runtime
        variance across downstream modelling tasks.
        """
     
        self.tickers = list(tickers)
     
        self.root_dir = config.ROOT_FIN_DIR
     
        self.macro = macro
     
        self.analyst_df = macro.r.analyst
     
        self.quiet = quiet
    
        self._init_daily_excel_cache()


    def _lock_path_for_today(
        self
    ) -> Path:
        """
        Return the lock-file path associated with today’s daily cache object.

        Returns
        -------
        pathlib.Path
            Path of the form:
            `<cache_path_for_today>.lock`.

        Rationale
        ---------
        A dedicated lock path enables mutual exclusion across concurrent processes
        that may attempt to build or update the same daily cache file.
        """
    
        p = self._cache_path_for_today()
    
        return p.with_suffix(p.suffix + ".lock")

    
    @contextmanager
    def _daily_cache_lock(
        self
    ):
        """
        Acquire and release an inter-process lock for daily cache operations.

        Lock algorithm
        --------------
        - Atomic acquisition uses `os.open(..., O_CREAT | O_EXCL | O_WRONLY)`.
          This guarantees that exactly one process creates the lock file.
    
        - If the lock exists, polling continues every `_CACHE_LOCK_POLL_SEC` seconds
          until acquisition or timeout at `_CACHE_LOCK_TIMEOUT_SEC`.
    
        - On context exit, the lock file is removed in a best-effort manner.

        Yields
        ------
        None
            Control is yielded to the caller while the lock is held.

        Raises
        ------
        TimeoutError
            Raised when lock acquisition exceeds the configured timeout.

        Operational advantages
        ----------------------
        - Prevents race conditions during cache writes.
    
        - Avoids partial-file visibility by serialising writers.
    
        - Provides deterministic contention handling via explicit timeout bounds.
    
        """
    
        lock_path = self._lock_path_for_today()
    
        start = time.time()

        while True:
    
            try:
    
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    
                os.close(fd)
    
                break
    
            except FileExistsError:
    
                if (time.time() - start) > self._CACHE_LOCK_TIMEOUT_SEC:
    
                    raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
    
                time.sleep(self._CACHE_LOCK_POLL_SEC)

        try:
    
            yield
    
        finally:
    
            try:
    
                os.remove(lock_path)
    
            except FileNotFoundError:
    
                pass
    
            except Exception:
    
                pass
    
            
    def _cache_path_for_today(
        self
    ) -> Path:
        """
        Return the canonical per-day cache path for pre-extracted financial sheets.

        Path construction
        -----------------
        Let `D` denote today in `YYYYMMDD` form. The returned path is:
        `FFDATA_CACHE_DIR/ffdata_cache_D.pkl`, where `FFDATA_CACHE_DIR` resolves to:
      
        - `config.FFDATA_CACHE_DIR` when defined; otherwise
      
        - `ROOT_FIN_DIR/_ffdata_daily_cache`.

        Returns
        -------
        pathlib.Path
            Cache file path for today.

        Notes
        -----
        The directory is created eagerly with `parents=True, exist_ok=True` so that
        subsequent atomic write operations cannot fail due to missing parents.
        """
     
        cache_dir = getattr(config, "FFDATA_CACHE_DIR", self.root_dir / self._CACHE_SUBDIR)
       
        cache_dir.mkdir(parents = True, exist_ok = True)
       
        tag = _TODAY_TS.strftime("%Y%m%d")
       
        return cache_dir / f"ffdata_cache_{tag}.pkl"


    def _load_daily_cache(
        self
    ) -> dict | None:
        """
        Load today’s daily cache from disk and validate structural metadata.

        Validation checks
        -----------------
        A loaded object is accepted only when all checks pass:
     
        1. `meta.version == _CACHE_VERSION`.
     
        2. `meta.date == today` in `YYYY-MM-DD` form.
     
        3. Keys `"financials"` and `"analyst_forecasts"` are present.

        Returns
        -------
        dict | None
            Valid cache payload on success; otherwise `None`.

        Failure handling
        ----------------
        Any deserialisation or schema error returns `None` rather than raising.
        This fail-soft strategy favours automatic rebuild over hard failure.
        """
    
        path = self._cache_path_for_today()
    
        if not path.exists():
    
            return None
    
        try:
    
            with open(path, "rb") as f:
    
                obj = pickle.load(f)
    
            meta = obj.get("meta", {})
    
            if meta.get("version") != self._CACHE_VERSION:
    
                return None
    
            if meta.get("date") != _TODAY_TS.strftime("%Y-%m-%d"):
    
                return None
    
            if "financials" not in obj or "analyst_forecasts" not in obj:
    
                return None
    
            return obj
    
        except Exception:
    
            return None


    def _save_daily_cache(
        self,
        obj: dict
    ) -> None:
        """
        Persist the daily cache object using an atomic write-then-replace protocol.

        Persistence algorithm
        ---------------------
        1. Create a temporary file in the target directory.
     
        2. Serialise with `pickle.HIGHEST_PROTOCOL`.
     
        3. Promote the temp file via `os.replace(tmp, final)`.

        Atomicity property
        ------------------
        `os.replace` is atomic on a single filesystem. Readers observe either the
        previous complete file or the new complete file, but not an intermediate
        partially written state.

        Parameters
        ----------
        obj : dict
            Cache payload to serialise.

        Returns
        -------
        None
        """
    
        path = self._cache_path_for_today()
    
        tmp_dir = path.parent
    
        tmp_dir.mkdir(parents = True, exist_ok = True)


        fd, tmp_path = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(tmp_dir))
       
        try:
       
            with os.fdopen(fd, "wb") as f:
       
                pickle.dump(obj, f, protocol = pickle.HIGHEST_PROTOCOL)
       
            os.replace(tmp_path, path)
       
        finally:
            
            try:
            
                if os.path.exists(tmp_path):
            
                    os.remove(tmp_path)
            
            except Exception:
            
                pass


    def _fundexpos_cache_key(
        self,
        *,
        tickers: Sequence[str],
        weekly_index: pd.DatetimeIndex,
        report_lag_days: int,
        price_lag_weeks: int,
        winsor_p: tuple[float, float],
        robust_scale: str,
        min_coverage: float,
    ) -> str:
        """
        Create a deterministic cache key for weekly fundamental exposure settings.

        Key design
        ----------
        The key encodes both universe and modelling hyperparameters:
      
        - ticker count and CRC32 hash of ticker ordering,
      
        - most recent weekly index date,
      
        - reporting lag, price lag,
      
        - winsorisation quantiles,
      
        - scaling method,
      
        - minimum cross-sectional coverage threshold.

        A second CRC32 digest is then applied to the concatenated descriptor string,
        producing a compact hexadecimal identifier.

        Parameters
        ----------
        tickers : Sequence[str]
            Ordered ticker set used in exposure construction.
        weekly_index : pd.DatetimeIndex
            Weekly evaluation grid; only the maximum date is encoded.
        report_lag_days : int
            Publication lag used for point-in-time alignment.
        price_lag_weeks : int
            Lag applied to prices in valuation ratio calculations.
        winsor_p : tuple[float, float]
            Lower and upper clipping quantiles.
        robust_scale : str
            Cross-sectional normalisation mode (for example `mad` or `zscore`).
        min_coverage : float
            Minimum valid-name fraction required per weekly cross-section.

        Returns
        -------
        str
            Eight-character hexadecimal key.

        Notes
        -----
        CRC32 is not cryptographic; the objective is compact reproducibility, not
        adversarial collision resistance.
        """
      
        tag = "None"
       
        if weekly_index is not None and len(weekly_index) > 0:
       
            tag = str(pd.to_datetime(weekly_index.max()).date())
       
        tickers_list = list(tickers)
       
        t_hash = zlib.crc32(",".join(tickers_list).encode("utf-8")) & 0xFFFFFFFF
       
        parts = [
            f"T{len(tickers_list)}",
            f"TW{tag}",
            f"H{t_hash}",
            f"lag{report_lag_days}",
            f"plag{price_lag_weeks}",
            f"w{winsor_p[0]:.2f}-{winsor_p[1]:.2f}",
            f"scale{robust_scale}",
            f"cov{min_coverage:.2f}",
        ]
       
        raw = "|".join(parts)
       
        return f"{zlib.crc32(raw.encode('utf-8')) & 0xFFFFFFFF:08x}"


    def _fundexpos_cache_path(
        self, 
        key: str
    ) -> Path:
        """
        Return the per-day cache path for weekly fundamental exposure artefacts.

        Parameters
        ----------
        key : str
            Cache key produced by `_fundexpos_cache_key`.

        Returns
        -------
        pathlib.Path
            Path of the form `ffdata_fundexpos_YYYYMMDD_<key>.pkl`.
        """
    
        cache_dir = getattr(config, "FFDATA_CACHE_DIR", self.root_dir / self._CACHE_SUBDIR)
    
        cache_dir.mkdir(parents = True, exist_ok = True)
    
        tag = _TODAY_TS.strftime("%Y%m%d")
    
        return cache_dir / f"ffdata_fundexpos_{tag}_{key}.pkl"


    def _load_fundexpos_cache(
        self,
        path: Path
    ) -> dict | None:
        """
        Load and validate a cached weekly fundamental exposure object.

        Validation criteria
        -------------------
        A payload is accepted only when:
     
        - `meta.version` equals `_FUND_EXPOS_CACHE_VERSION`,
     
        - `meta.date` equals today (`YYYY-MM-DD`),
     
        - key `"exposures"` exists.

        Parameters
        ----------
        path : pathlib.Path
            Candidate cache file.

        Returns
        -------
        dict | None
            Valid cache object or `None` when absent, stale, malformed, or unreadable.
        """
    
        if not path.exists():
    
            return None
    
        try:
    
            with open(path, "rb") as f:
    
                obj = pickle.load(f)
    
            meta = obj.get("meta", {})
    
            if meta.get("version") != self._FUND_EXPOS_CACHE_VERSION:
    
                return None
    
            if meta.get("date") != _TODAY_TS.strftime("%Y-%m-%d"):
    
                return None
    
            if "exposures" not in obj:
    
                return None
    
            return obj
    
        except Exception:
    
            return None


    def _save_fundexpos_cache(
        self, 
        path: Path,
        obj: dict
    ) -> None:
        """
        Persist fundamental exposure cache data atomically.

        Parameters
        ----------
        path : pathlib.Path
            Final cache location.
        obj : dict
            Serialisable cache object, typically containing `meta` and `exposures`.

        Returns
        -------
        None

        Notes
        -----
        The same atomic replacement strategy as `_save_daily_cache` is used to avoid
        exposing partially written files to readers.
        """
    
        tmp_dir = path.parent
    
        tmp_dir.mkdir(parents = True, exist_ok = True)
        
        fd, tmp_path = tempfile.mkstemp(prefix = path.stem + "_", suffix = ".tmp", dir = str(tmp_dir))
        
        try:
        
            with os.fdopen(fd, "wb") as f:
        
                pickle.dump(obj, f, protocol = pickle.HIGHEST_PROTOCOL)
        
            os.replace(tmp_path, path)
        
        finally:
            
            try:
            
                if os.path.exists(tmp_path):
            
                    os.remove(tmp_path)
            except Exception:
            
                pass


    def _read_sheet_from_excelfile(
        self,
        xls: pd.ExcelFile,
        sheet: str,
        rows: list[str]
    ) -> pd.DataFrame:
        """
        Read and normalise a single worksheet from an open `pandas.ExcelFile` handle.

        Normalisation steps
        -------------------
        1. Read with first column as row index and standard missing-value sentinels.
       
        2. Convert all column labels to datetimes using coercive parsing.
       
        3. Reindex rows to the requested ordered row set.
       
        4. Sort columns chronologically.

        Parameters
        ----------
        xls : pd.ExcelFile
            Already-open workbook handle to avoid repeated file open costs.
        sheet : str
            Worksheet name.
        rows : list[str]
            Canonical row ordering expected by downstream code.

        Returns
        -------
        pd.DataFrame
            Date-column matrix aligned to requested rows. Missing rows are retained
            as all-NaN to preserve schema consistency across issuers.
        """
        
        df = pd.read_excel(
            xls,
            sheet_name = sheet,
            index_col = 0,
            na_values = ["—", "-", ""],
        )
        
        df.columns = pd.to_datetime(df.columns, errors = "coerce")
        
        return df.reindex(rows).reindex(sorted(df.columns), axis = 1)
    
    
    def _build_cache_for_ticker(self, tkr: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Build all cached sheet objects for a single ticker from source workbooks.

        Processing logic
        ----------------
        - Financial workbook (`<ticker>-financials.xlsx`) is opened once.
      
        - Required sheets are extracted through `_read_sheet_from_excelfile`.
      
        - Missing or unreadable sheets are replaced with empty DataFrames.
      
        - Analyst workbook (`<ticker>-analyst-forecasts.xlsx`) is parsed through
          `_read_analyst_sheet_uncached` when present.

        Parameters
        ----------
        tkr : str
            Ticker symbol whose workbook directory is `root_dir / tkr`.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
   
            - `fin_sheets`: dictionary containing every expected financial sheet key.
   
            - `analyst_fc`: analyst forecast frame (possibly empty).

        Design advantages
        -----------------
        - Single-open workbook parsing reduces I/O overhead.
   
        - Guaranteed key presence removes repeated guard logic in downstream accessors.
   
        - Normalised empty-frame fallback keeps the cache schema deterministic.
   
        """
       
        folder = self.root_dir / tkr

        expected = [
            (self._SHEET_INCOME, self._INCOME_ROWS),
            (self._SHEET_CASHFLOW, self._CASHFLOW_ROWS),
            (self._SHEET_BALANCE, self._BALANCE_ROWS),
            (self._SHEET_RATIOS, self._RATIOS_ROWS),
            (self._SHEET_RATIOS_ANN, self._RATIOS_ROWS_ANN),
            (self._SHEET_INCOME_ANN, self._INCOME_ROWS_ANN),
        ]

        fin_sheets: dict[str, pd.DataFrame] = {sheet: pd.DataFrame() for sheet, _ in expected}

        fin_path = folder / f"{tkr}-financials.xlsx"
      
        if fin_path.exists():
      
            try:
      
                with pd.ExcelFile(fin_path, engine = "openpyxl") as xls:
      
                    for sheet, rows in expected:
      
                        try:
      
                            fin_sheets[sheet] = self._read_sheet_from_excelfile(
                                xls = xls, 
                                sheet = sheet,
                                rows = rows
                            )
      
                        except Exception as e:
      
                            if not self.quiet:
      
                                print(f"[WARN] build_cache financials {tkr} {sheet}: {e}")
      
                            fin_sheets[sheet] = pd.DataFrame()
      
            except Exception as e:
      
                if not self.quiet:
      
                    print(f"[WARN] build_cache financials open {tkr}: {e}")

        analyst_fc = pd.DataFrame()

        analyst_path = folder / f"{tkr}-analyst-forecasts.xlsx"

        if analyst_path.exists():

            try:

                analyst_fc = self._read_analyst_sheet_uncached(
                    path = analyst_path, 
                    tkr = tkr
                )

            except Exception as e:

                if not self.quiet:

                    print(f"[WARN] build_cache analyst {tkr}: {e}")

                analyst_fc = pd.DataFrame()

        return fin_sheets, analyst_fc


    def _init_daily_excel_cache(
        self
    ) -> None:
        """
        Initialise the daily in-memory sheet cache with lock-protected rebuild logic.

        Workflow
        --------
        1. Acquire the daily lock to prevent concurrent writers.
     
        2. Attempt to load a valid on-disk cache unless forced rebuild is enabled.
     
        3. Detect missing ticker entries or incomplete sheet sets.
     
        4. Rebuild only required tickers via `_build_cache_for_ticker`.
     
        5. Persist updated cache atomically.
     
        6. Ensure analyst forecast key presence for every ticker.
     
        7. Publish `self._fin_sheet_cache` and `self._analyst_fc_cache`.

        Returns
        -------
        None

        Guarantees
        ----------
        - `financials[ticker]` exists for each ticker and contains all expected sheet keys.
     
        - `analyst_forecasts[ticker]` exists for each ticker (possibly empty).

        Practical advantages
        --------------------
        - Incremental rebuild minimises unnecessary workbook parsing.
     
        - Locked writes and atomic replacement provide robustness under contention.
     
        - In-memory publication makes later data access deterministic and fast.
     
        """
      
        force = bool(getattr(config, "FORCE_REBUILD_FFDATA_CACHE", False))

        expected_sheets = [
            self._SHEET_INCOME,
            self._SHEET_CASHFLOW,
            self._SHEET_BALANCE,
            self._SHEET_RATIOS,
            self._SHEET_RATIOS_ANN,
            self._SHEET_INCOME_ANN,
        ]


        def _ticker_fin_complete(
            fin_all: dict,
            t: str
        ) -> bool:
        
            if t not in fin_all:
        
                return False
        
            d = fin_all.get(t, {})
        
            return all(s in d for s in expected_sheets)


        with self._daily_cache_lock():

            cache = None if force else self._load_daily_cache()

            if cache is None:

                cache = {
                    "meta": {
                        "version": self._CACHE_VERSION,
                        "date": _TODAY_TS.strftime("%Y-%m-%d"),
                    },
                    "financials": {},
                    "analyst_forecasts": {},
                }

            fin_all: dict[str, dict[str, pd.DataFrame]] = cache.get("financials", {})

            an_all: dict[str, pd.DataFrame] = cache.get("analyst_forecasts", {})

            need_build = [t for t in self.tickers if not _ticker_fin_complete(
                fin_all = fin_all,
                t = t
            )]

            if need_build:
             
                for t in need_build:
             
                    fin_sheets, analyst_fc = self._build_cache_for_ticker(
                        tkr = t
                    )
             
                    fin_all[t] = fin_sheets
             
                    an_all[t] = analyst_fc if analyst_fc is not None else pd.DataFrame()

                cache["financials"] = fin_all

                cache["analyst_forecasts"] = an_all

                self._save_daily_cache(
                    obj = cache
                )

            changed = False

            for t in self.tickers:

                if t not in an_all:

                    an_all[t] = pd.DataFrame()

                    changed = True

            if changed:

                cache["analyst_forecasts"] = an_all

                self._save_daily_cache(
                    obj = cache
                )

            self._fin_sheet_cache = fin_all

            self._analyst_fc_cache = an_all
        
        
    def currency_adjustment(
        self, 
        ticker: str, s: pd.Series | pd.DataFrame
    ):
        """
        Apply hard-coded issuer-specific currency/unit normalisation rules.

        Piecewise transformation
        ------------------------
        Let `x` denote the input series or frame, and let spot rates be obtained from
        `macro.currency`.

        - For `ticker` in `{ "BABA", "PDD" }`:
          `x_adj = x / USDCNY`.
          This maps CNY-denominated values to USD when the quoted rate is CNY per USD.

        - For `ticker` in `{ "BP.L", "AZN.L", "HSBA.L", "GSK.L", "IWG.L" }`:
          `x_adj = x * GBPUSD / 10`.
          The GBP-to-USD conversion is combined with a fixed unit harmonisation factor
          of `1/10` used by the local data convention.

        - For `ticker == "ASX"`:
          `x_adj = x / 31`.
          This applies a fixed divisor required by the source-unit convention.

        - For all other tickers:
          `x_adj = x`.

        Parameters
        ----------
        ticker : str
            Issuer identifier selecting the adjustment regime.
        s : pd.Series | pd.DataFrame
            Numeric input to be transformed.

        Returns
        -------
        pd.Series | pd.DataFrame
            Adjusted object with unchanged shape and index/column structure.

        Notes
        -----
        - Rules are intentionally explicit and conservative; they should be replaced by
          metadata-driven conversion if dependable currency metadata become available.
    
        - The method is vectorised through pandas broadcasting and therefore scales
          efficiently across long histories.
        
        """
        
        ex = self.macro.currency
        
        if ticker in ["BABA", "PDD"]:
           
            rate = float(ex.loc["USDCNY"])
           
            return s / rate
        
        if ticker in ["BP.L", "AZN.L", "HSBA.L", "GSK.L", "IWG.L"]:
           
            rate = float(ex.loc["GBPUSD"])
           
            return s * rate / 10
        
        if ticker == "ASX":
           
            rate = 31
           
            return s / rate
        
        return s


    def _parse_growth(
        self, 
        x
    ):
        """
        Parse heterogeneous growth representations into decimal proportions.

        Conversion equations
        --------------------
        Let raw input be `x_raw`. The parser returns `g` where:

        - If `x_raw` is a percent string such as `"12.5%"`:
          `g = 12.5 / 100 = 0.125`.
     
        - If `x_raw` is numeric text or numeric scalar with absolute magnitude above
          unity and intended as percentage points, the same rule applies:
          `g = x_raw / 100`.
     
        - Otherwise:
          `g = x_raw` (already assumed to be in decimal form).
     
        - Parsing failure returns `NaN`.

        Heuristic rationale
        -------------------
        Source sheets typically mix percentages and decimal proportions. The
        threshold rule (`value > 1.0`) provides pragmatic standardisation while
        preserving already scaled decimals.

        Returns
        -------
        float
            Growth in decimal form.
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
        Remove outliers using Tukey interquartile fences.

        Definition
        ----------
        For input values `x`:
     
        - `Q1 = quantile_0.25(x)`
     
        - `Q3 = quantile_0.75(x)`
     
        - `IQR = Q3 - Q1`
     
        - lower fence `L = Q1 - m * IQR`
     
        - upper fence `U = Q3 + m * IQR`

        Retained observations satisfy `L <= x_i <= U`.
        With `m = 1.5`, the method corresponds to standard Tukey boxplot fences.

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

        Advantages
        ----------
        - Distribution-free and simple to interpret.
     
        - Robust to heavy tails relative to mean-standard-deviation clipping.
     
        - Effective as a pre-processing step before growth compounding.
     
        """      
        
        q1, q3 = series.quantile([0.25, 0.75])
        
        iqr = q3 - q1
        
        lo,hi = q1 - (m * iqr), q3 + (m * iqr)
       
        return series[(series >= lo) & (series <= hi)]


    def _empty_forecast(
        self
    ) -> pd.DataFrame:
        """
        Construct a schema-stable empty forecast panel for a fixed horizon.

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

        Use case
        --------
        This output acts as a deterministic fallback object, allowing downstream
        joins and column selection to proceed without special-case logic.
        """    
        
        dates = pd.date_range(
            start = _TODAY_TS + pd.offsets.YearEnd(1),
            periods = _HORIZON_YEARS+1,
            freq = "YE-DEC",
        )
       
        return pd.DataFrame(np.nan, index = dates, columns = self._FORECAST_COLS)


    def _empty_kpis(
        self
    ) -> pd.DataFrame:
        """
        Construct a one-row KPI frame initialised with NaN placeholders.

        Returns
        -------
        pandas.DataFrame
            Index [TODAY], columns matching `_KPI_COLS`, values NaN.

        Rationale
        ---------
        A shape-consistent null object preserves pipeline contracts when KPI
        computation fails for an issuer.
        """
        
        return pd.DataFrame(np.nan, index = [_TODAY_TS], columns = self._KPI_COLS)


    def _read_sheet(
        self,
        tkr: str, 
        sheet: str, 
        rows: list[str]
    ) -> pd.DataFrame:
        """
        Retrieve a preloaded financial sheet from the in-memory daily cache.

        Parameters
        ----------
        tkr : str
            Ticker symbol.
        sheet : str
            Worksheet key.
        rows : list[str]
            Expected row schema (retained for API compatibility).

        Returns
        -------
        pd.DataFrame
            Defensive copy of the cached sheet. Returns an empty frame when the
            cache entry is absent.

        Notes
        -----
        No workbook I/O occurs in this method; all disk access is delegated to the
        cache build stage.
        """
      
        cached = getattr(self, "_fin_sheet_cache", {}).get(tkr, {}).get(sheet, None)
      
        if cached is None:
      
            if not self.quiet:
      
                print(f"[WARN] cache-miss financial sheet {tkr}:{sheet} (returning empty)")
      
            return pd.DataFrame()
      
        return cached.copy()


    @cached_property
    def income(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Income TTM DataFrame.

        Extraction details
        ------------------
        - Source key: `_SHEET_INCOME`.
     
        - Row schema: `_INCOME_ROWS`.
     
        - Access path: in-memory cache through `_read_sheet`, therefore no workbook
          disk I/O is performed here.

        Data semantics
        --------------
        Values are preserved as supplied by source sheets (for example abbreviated
        magnitudes). Currency harmonisation and numeric coercion are applied later in
        downstream feature builders such as `annuals`.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Income-TTM frame (possibly empty).

        Operational advantage
        ---------------------
        `cached_property` ensures one-pass extraction per instance, reducing repeated
        lookup and allocation overhead during model construction.
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


    def build_pit_fundamental_exposures_weekly(
        self,
        *,
        tickers: Sequence[str],
        weekly_index: pd.DatetimeIndex,
        weekly_prices: pd.DataFrame,
        shares_outstanding: Optional[pd.Series],
        report_lag_days: int = 90,
        price_lag_weeks: int = 1,
        winsor_p: tuple[float, float] = (0.05, 0.95),
        robust_scale: str = "mad",
        min_coverage: float = 0.6,
    ) -> dict[str, pd.DataFrame]:
        """
        Build point-in-time weekly fundamental style exposures for a ticker universe.

        Objective
        ---------
        Construct cross-sectional weekly signals that are:
        - point-in-time valid (explicit reporting lag),
   
        - economically interpretable (profitability, growth, leverage, cash-flow
          stability, value),
   
        - robust to outliers and sparse coverage.

        Inputs
        ------
        tickers : Sequence[str]
            Universe to score.
        weekly_index : pd.DatetimeIndex
            Weekly observation grid.
        weekly_prices : pd.DataFrame
            Weekly close levels by ticker.
        shares_outstanding : Optional[pd.Series]
            Shares outstanding used in price-to-sales construction.
        report_lag_days : int, default 90
            Delay applied to statement dates to prevent look-ahead bias.
        price_lag_weeks : int, default 1
            Additional lag on prices for ratio denominators.
        winsor_p : tuple[float, float], default (0.05, 0.95)
            Lower and upper clipping quantiles per weekly cross-section.
        robust_scale : str, default "mad"
            Scaling mode:
            - `"mad"`: median/MAD scaling,
            - otherwise: mean/standard-deviation scaling.
        min_coverage : float, default 0.6
            Minimum fraction of non-missing names required to publish a weekly score.

        Feature construction
        --------------------
        For ticker `i` and report-date-aligned series:

        1. PROFIT
           `PROFIT_raw = mean(ROE, ROA)`.

        2. GROWTH
           `g_rev_t = Revenue_t / Revenue_{t-k} - 1`,
           `g_eps_t = EPS_t / EPS_{t-k} - 1`,
           where `k = 4` when sufficient quarterly history exists, otherwise `k = 1`.
           `GROWTH_raw = mean(g_rev_t, g_eps_t)`.

        3. LEVER
           `LEVER_raw = (TotalDebt - CashAndEquivalents) / TotalAssets`.

        4. CASH_STAB
           Compute rolling four-period coefficient of variation of operating cash flow:
           `CV_t = std_4(OCF)_t / abs(mean_4(OCF)_t)`,
           then define stability as `CASH_STAB_raw = -CV_t`.

        5. VALUE
           Build lagged valuation ratios:
           - `PE = Price_lag / EPS` (where EPS > 0),
           - `PB = Price_lag / BVPS` (where BVPS > 0),
           - `PS = Price_lag / (Revenue / SharesOutstanding)` (where valid).
           Aggregate as:
           `VALUE_raw = -mean(log(PE), log(PB), log(PS))`.

        Point-in-time alignment
        -----------------------
        Each fundamental series is shifted by `report_lag_days` and then forward-filled
        onto `weekly_index`:
        `x_weekly(w) = last_available( x_reported(t + lag) <= w )`.

        Cross-sectional normalisation
        -----------------------------
        For each week:
        1. Winsorise at quantiles `(q_low, q_high)`.
        2. Scale:
           - MAD mode:
             `z_i = (x_i - median(x)) / (1.4826 * MAD(x))`,
             where `MAD(x) = median(|x - median(x)|)`.
           - Standard mode:
             `z_i = (x_i - mean(x)) / std(x)`.
        3. If coverage `< min_coverage`, emit NaN for that week.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with keys:
            `{"PROFIT", "GROWTH", "LEVER", "CASH_STAB", "VALUE"}`.
            Each value is a weekly DataFrame indexed by `weekly_index`, columns=tickers.

        Modelling rationale and advantages
        ----------------------------------
        - Reporting lag and price lag reduce inadvertent forward-looking leakage.
   
        - Multi-signal construction captures complementary style dimensions.
   
        - Log-ratio value aggregation mitigates multiplicative scale distortions.
   
        - Winsorisation and robust scaling improve stability under fat tails and
          accounting outliers.
   
        - Coverage gating avoids publishing fragile cross-sections with too few names.
   
        - Deterministic cache keys make repeated runs reproducible and efficient.
   
        """
        
        if weekly_index is None or len(weekly_index) == 0:
        
            return {}

        tickers = list(tickers)
     
        weekly_index = pd.DatetimeIndex(weekly_index).sort_values()

        key = self._fundexpos_cache_key(
            tickers = tickers,
            weekly_index = weekly_index,
            report_lag_days = report_lag_days,
            price_lag_weeks = price_lag_weeks,
            winsor_p = winsor_p,
            robust_scale = robust_scale,
            min_coverage = min_coverage,
        )
      
        cache_path = self._fundexpos_cache_path(
            key = key
        )

        force = bool(getattr(config, "FORCE_REBUILD_FUND_EXPOS_CACHE", False))
     
        force = force or bool(getattr(config, "FORCE_REBUILD_FFDATA_CACHE", False))

        with self._daily_cache_lock():
       
            if not force:
       
                cached = self._load_fundexpos_cache(
                    path = cache_path
                )
       
                if cached is not None and cached.get("meta", {}).get("key") == key:
       
                    return cached.get("exposures", {})

            weekly_prices = (
                weekly_prices.reindex(index = weekly_index)
                if weekly_prices is not None
                else pd.DataFrame(index = weekly_index)
            )
          
            weekly_prices = weekly_prices.reindex(columns = tickers)
          
            weekly_prices_lag = weekly_prices.shift(price_lag_weeks)

            if shares_outstanding is None:
          
                shares_outstanding = pd.Series(index = tickers, dtype = float)
          
            else:
          
                shares_outstanding = shares_outstanding.reindex(tickers)
          
                shares_outstanding = pd.to_numeric(shares_outstanding, errors = "coerce")

          
            def _parse_ratio_value(
                x
            ) -> float:
            
                try:
            
                    v = float(x)
            
                except Exception:
            
                    return np.nan
            
                if not np.isfinite(v):
            
                    return np.nan
            
                if abs(v) > 1.5:
            
                    return v / 100.0 if abs(v) <= 100.0 else v
            
                return v


            def _parse_number(
                x
            ) -> float:
            
                v = pd.to_numeric(x, errors = "coerce")
            
                if np.isnan(v):
            
                    try:
            
                        v = _parse_abbrev(
                            val = x
                        )
            
                    except Exception:
            
                        v = np.nan
            
                return v


            def _row_series(
                df: pd.DataFrame,
                candidates: list[str],
                *,
                parser: Optional[Callable] = None,
            ) -> Optional[pd.Series]:
        
                if df is None or df.empty:
        
                    return None
        
                for row in candidates:
        
                    if row in df.index:
        
                        s = df.loc[row]
        
                        if parser is not None:
        
                            s = s.map(parser)
        
                        else:
        
                            s = pd.to_numeric(s, errors = "coerce")
        
                        try:
        
                            s.index = pd.to_datetime(s.index, errors = "coerce")
        
                        except Exception:
        
                            pass
        
                        s = s[~s.index.isna()]
        
                        if not s.empty:
        
                            return s.sort_index()
        
                return None


            def _yoy_growth(
                s: Optional[pd.Series]
            ) -> Optional[pd.Series]:
            
                if s is None or s.empty:
            
                    return None
            
                s = s.sort_index()
            
                s = s.ffill()
            
                if len(s) >= 5:
            
                    g = s.pct_change(4, fill_method = None)
            
                    if g.notna().sum() == 0:
            
                        g = s.pct_change(fill_method = None)
            
                else:
            
                    g = s.pct_change(fill_method = None)
            
                return g


            def _asof_weekly(
                s: Optional[pd.Series]
            ) -> Optional[pd.Series]:
            
                if s is None or s.empty:
            
                    return None
            
                s = s.dropna()
            
                if s.empty:
            
                    return None
            
                s = s.sort_index()
            
                s.index = pd.to_datetime(s.index, errors = "coerce")
            
                s = s[~s.index.isna()]
            
                if s.empty:
            
                    return None
            
                s = s.copy()
            
                s.index = s.index + pd.Timedelta(days = report_lag_days)
            
                return s.reindex(weekly_index, method = "ffill")


            def _init_frame() -> pd.DataFrame:
              
                return pd.DataFrame(index = weekly_index, columns = tickers, dtype = float)


            profit_df = _init_frame()

            growth_df = _init_frame()

            lever_df = _init_frame()

            cash_stab_df = _init_frame()

            value_df = _init_frame()

            for t in tickers:

                if hasattr(config, "TICKER_EXEMPTIONS") and t in config.TICKER_EXEMPTIONS:

                    continue

                ratios = self.ratios.get(t, pd.DataFrame())

                income = self.income.get(t, pd.DataFrame())

                bal = self.bal.get(t, pd.DataFrame())

                cash = self.cash.get(t, pd.DataFrame())

                roe = _row_series(
                    df = ratios, 
                    candidates = ["Return on Equity (ROE)"], 
                    parser = _parse_ratio_value
                )

                roa = _row_series(
                    df = ratios, 
                    candidates = ["Return on Assets (ROA)"], 
                    parser = _parse_ratio_value
                )

                if roe is not None or roa is not None:
             
                    dfp = pd.concat([s for s in [roe, roa] if s is not None], axis = 1)
             
                    profit_raw = dfp.mean(axis = 1, skipna = True)
             
                    profit_df[t] = _asof_weekly(
                        s = profit_raw
                    )

                rev = _row_series(
                    df = income, 
                    candidates = ["Revenue"], 
                    parser = _parse_abbrev
                )
              
                eps = _row_series(
                    df = income, 
                    candidates = ["EPS (Basic)"], 
                    parser = _parse_number
                )
              
                rev_g = _yoy_growth(
                    s = rev
                )
              
                eps_g = _yoy_growth(
                    s = eps
                )
                
                if rev_g is not None or eps_g is not None:
                
                    dfg = pd.concat([s for s in [rev_g, eps_g] if s is not None], axis = 1)
                
                    growth_raw = dfg.mean(axis = 1, skipna = True)
                
                    growth_df[t] = _asof_weekly(
                        s = growth_raw
                    )

                total_debt = _row_series(
                    df = bal,
                    candidates = ["Total Debt"], 
                    parser = _parse_abbrev
                )
            
                cash_eq = _row_series(
                    df = bal,
                    candidates = ["Cash & Cash Equivalents"],
                    parser = _parse_abbrev
                )
            
                total_assets = _row_series(
                    df = bal,
                    candidates = ["Total Assets"],
                    parser = _parse_abbrev
                )
                
                if total_debt is not None and total_assets is not None:
                 
                    cash_eq = cash_eq.reindex(total_debt.index).fillna(0.0) if cash_eq is not None else 0.0
                 
                    net_debt = total_debt - cash_eq
                 
                    leverage_raw = net_debt / total_assets.replace(0.0, np.nan)
                 
                    leverage_raw = leverage_raw.replace([np.inf, -np.inf], np.nan)
                 
                    lever_df[t] = _asof_weekly(
                        s = leverage_raw
                    )

                cfo = _row_series(
                    df = cash,
                    candidates = ["Operating Cash Flow"], 
                    parser = _parse_abbrev
                )
                
                if cfo is not None:
                
                    rolling_mean = cfo.rolling(4, min_periods = 2).mean()
                
                    rolling_std = cfo.rolling(4, min_periods = 2).std()
                
                    cv = rolling_std / rolling_mean.abs().replace(0.0, np.nan)
                
                    cash_stab_raw = (-cv).replace([np.inf, -np.inf], np.nan)
                
                    cash_stab_df[t] = _asof_weekly(
                        s = cash_stab_raw
                    )

                price_wk = weekly_prices_lag[t] if t in weekly_prices_lag.columns else None
               
                eps_wk = _asof_weekly(
                    s = eps
                )
               
                bvps = _row_series(
                    df = bal,
                    candidates = ["Book Value Per Share"], 
                    parser = _parse_number
                )
               
                bvps_wk = _asof_weekly(
                    s = bvps
                )
               
                rev_wk = _asof_weekly(
                    s = rev
                )

                ratios_list = []
             
                if price_wk is not None and eps_wk is not None:
             
                    pe = (price_wk / eps_wk).where(eps_wk > 0)
             
                    ratios_list.append(pe)
             
                if price_wk is not None and bvps_wk is not None:
             
                    pb = (price_wk / bvps_wk).where(bvps_wk > 0)
             
                    ratios_list.append(pb)
             
                so = shares_outstanding.get(t, np.nan)
             
                if price_wk is not None and rev_wk is not None and np.isfinite(so) and so > 0:
             
                    ps = price_wk / (rev_wk / so)
             
                    ratios_list.append(ps.where(ps > 0))

                if ratios_list:
             
                    val = -pd.concat(
                        [np.log(r.where(r > 0)) for r in ratios_list],
                        axis = 1
                    ).mean(axis = 1, skipna = True)
             
                    value_df[t] = val.replace([np.inf, -np.inf], np.nan)

            
            def _robust_scale_frame(
                df: pd.DataFrame
            ) -> pd.DataFrame:
            
                out = pd.DataFrame(index = df.index, columns = df.columns, dtype = float)
            
                n = len(df.columns)
            
                if n == 0:
            
                    return out
            
                for dt, row in df.iterrows():
            
                    valid = row.dropna()
            
                    if valid.empty or (valid.size / n) < min_coverage:
            
                        out.loc[dt] = np.nan
            
                        continue
            
                    lo = valid.quantile(winsor_p[0])
            
                    hi = valid.quantile(winsor_p[1])
            
                    row = row.clip(lo, hi)
            
                    if robust_scale.lower() == "mad":
            
                        med = row.median(skipna = True)
            
                        mad = (row - med).abs().median(skipna = True)
            
                        scale = 1.4826 * mad if np.isfinite(mad) and mad > 0 else row.std(skipna = True)
            
                        if not np.isfinite(scale) or scale == 0:
            
                            out.loc[dt] = (row - med) * 0.0
            
                        else:
            
                            out.loc[dt] = (row - med) / scale
            
                    else:
            
                        mean = row.mean(skipna = True)
            
                        std = row.std(skipna = True)
            
                        if not np.isfinite(std) or std == 0:
            
                            out.loc[dt] = (row - mean) * 0.0
            
                        else:
            
                            out.loc[dt] = (row - mean) / std
            
                return out.replace([np.inf, -np.inf], np.nan)


            exposures = {
                "PROFIT": _robust_scale_frame(
                    df = profit_df
                ),
                "GROWTH": _robust_scale_frame(
                    df = growth_df
                ),
                "LEVER": _robust_scale_frame(
                    df = lever_df
                ),
                "CASH_STAB": _robust_scale_frame(
                    df = cash_stab_df
                ),
                "VALUE": _robust_scale_frame(
                    df = value_df
                ),
            }

            obj = {
                "meta": {
                    "version": self._FUND_EXPOS_CACHE_VERSION,
                    "date": _TODAY_TS.strftime("%Y-%m-%d"),
                    "key": key,
                    "report_lag_days": report_lag_days,
                    "price_lag_weeks": price_lag_weeks,
                    "winsor_p": winsor_p,
                    "robust_scale": robust_scale,
                    "min_coverage": min_coverage,
                },
                "exposures": exposures,
            }
            
            self._save_fundexpos_cache(
                path = cache_path, 
                obj = obj
            )

            return exposures


    @cached_property
    def cash(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Cash-flow TTM DataFrame.

        Extraction details
        ------------------
        - Source key: `_SHEET_CASHFLOW`.
       
        - Row schema: `_CASHFLOW_ROWS`.
       
        - Values are not parsed numerically in this accessor; parsing occurs where
          explicit units are required.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Cash-flow-TTM frame (possibly empty).

        Notes
        -----
        Empty DataFrames are emitted for missing sheets, preserving a stable mapping
        structure across the ticker universe.
        
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

        Extraction details
        ------------------
        - Source key: `_SHEET_BALANCE`.
    
        - Row schema: `_BALANCE_ROWS`.
    
        - Output frames preserve source cell formats for downstream controlled
          parsing.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Balance-sheet-TTM frame (possibly empty).

        Advantage
        ---------
        Centralised retrieval gives a single canonical balance-sheet view for all
        higher-level calculations (for example NOA and leverage).
        
        """

        d = {}

        for t in self.tickers:

            try:
        
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_BALANCE,
                    rows = self._BALANCE_ROWS
                )
        
            except Exception as e:
        
                if not self.quiet: 
        
                    print(f"[WARN] bal {t}: {e}")

                d[t] = pd.DataFrame()

        return d


    @cached_property
    def ratios(self) -> dict[str,pd.DataFrame]:
        """
        Lazy map from ticker to Ratios TTM DataFrame.

        Content
        -------
        Rows include valuation and profitability ratios listed in `_RATIOS_ROWS`
        (for example EV/Revenue, PE, PB, EV/EBITDA, ROE, ROA, Debt/Equity).

        Parsing policy
        --------------
        Raw workbook values are retained at this stage. Numeric coercion and unit
        harmonisation are deferred to the specific modelling routine that consumes
        each ratio.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Ratios-TTM frame (possibly empty).

        Advantage
        ---------
        Deferred parsing avoids irreversible assumptions when source formats differ
        by metric family.
        
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

        Content
        -------
        Typically includes long-horizon metrics such as annual ROE and payout ratio
        according to `_RATIOS_ROWS_ANN`.

        Data treatment
        --------------
        Values are returned in source representation. Downstream methods apply
        explicit parsers before statistical aggregation.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Ratios-Annual frame (possibly empty).

        Advantage
        ---------
        Annual ratio access is isolated from TTM retrieval, simplifying horizon-
        specific modelling code.
        
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

        Content
        -------
        Rows follow `_INCOME_ROWS_ANN`, typically annual revenue and EPS growth
        series used in synthetic forecast generation and growth diagnostics.

        Parsing policy
        --------------
        No conversion is imposed in this accessor. Consumers choose between
        `_parse_growth` and `_parse_abbrev` depending on whether source cells encode
        percentages, decimal rates, or abbreviated numerics.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Ticker → Income-Annual frame (possibly empty).

        Advantage
        ---------
        Keeping raw annual rows available supports transparent audit and alternative
        growth-normalisation strategies.
        
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
    def annuals(
        self
    ) -> dict[str,pd.DataFrame]:
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

        Advantages of this construction
        -------------------------------
        - Centralises sign conventions (for example acquisitions as cash outflow),
          improving consistency across downstream valuation steps.
    
        - Aligns all major cash-flow and balance-sheet drivers in one panel, which
          simplifies model specification and auditability.
    
        - Uses explicit tax and financing decompositions:
          `InterestAfterTax = InterestExpense * (1 - TaxRate)`,
          supporting capital-cost and enterprise-value analyses.
    
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


    def _read_analyst_sheet_uncached(
        self, 
        path: Path,
        tkr: str
    ) -> pd.DataFrame:
        """
        Parse analyst forecast workbook data directly from disk.

        Purpose
        -------
        This method is used only during cache construction. It converts raw analyst
        workbook rows into a normalised time-indexed forecast frame and applies
        issuer-specific currency adjustments exactly once.

        Input mapping
        -------------
        Source rows are mapped to target columns:
       
        - Analysts -> `num_analysts`
       
        - Revenue Low / Revenue / Revenue High -> `low_rev`, `avg_rev`, `high_rev`
       
        - EPS Low / EPS / EPS High -> `low_eps`, `avg_eps`, `high_eps`
       
        - Period Ending row provides the datetime index.

        Parameters
        ----------
        path : pathlib.Path
            Analyst workbook path (`<ticker>-analyst-forecasts.xlsx`).
        tkr : str
            Ticker used for currency adjustment logic.

        Returns
        -------
        pd.DataFrame
            Forecast panel sorted by period ending date with columns in
            `_FORECAST_COLS`.

        Notes
        -----
        Revenue values are parsed through `_parse_abbrev` before conversion. EPS
        values are coerced numerically with invalid entries mapped to NaN.
        """
      
        df = pd.read_excel(path, index_col = 0, engine = "openpyxl")
      
        df.columns = [str(c).strip() for c in df.columns]

        dates = pd.to_datetime(df.loc["Period Ending"].dropna())

        data = {
            "num_analysts": df.loc["Analysts"].astype(int).values,
            "low_rev": self.currency_adjustment(
                ticker = tkr, 
                s = df.loc["Revenue Low"].apply(_parse_abbrev)
            ).values,
            "avg_rev": self.currency_adjustment(
                ticker = tkr, 
                s = df.loc["Revenue"].apply(_parse_abbrev)
            ).values,
            "high_rev": self.currency_adjustment(
                ticker = tkr, 
                s = df.loc["Revenue High"].apply(_parse_abbrev)
            ).values,
            "low_eps": self.currency_adjustment(
                ticker = tkr, 
                s = pd.to_numeric(df.loc["EPS Low"], errors="coerce")
            ).values,
            "avg_eps": self.currency_adjustment(
                ticker = tkr, 
                s = pd.to_numeric(df.loc["EPS"], errors="coerce")
            ).values,
            "high_eps": self.currency_adjustment(
                ticker = tkr, 
                s = pd.to_numeric(df.loc["EPS High"], errors="coerce")
            ).values,
        }

        return pd.DataFrame(data, index = dates).sort_index()


    def _read_analyst_sheet(
        self, 
        path: Path, 
        tkr: str
    ) -> pd.DataFrame:
        """
        Return analyst forecasts from the in-memory cache for the requested ticker.

        Parameters
        ----------
        path : pathlib.Path
            Ignored argument retained for backward compatibility with older call
            signatures.
        tkr : str
            Ticker key in the cached analyst dictionary.

        Returns
        -------
        pd.DataFrame
            Defensive copy of cached analyst forecasts. Empty DataFrame when absent.

        Notes
        -----
        No file I/O occurs here; this accessor exists to preserve a stable API while
        enforcing cache-only reads.
        """
       
        cached = getattr(self, "_analyst_fc_cache", {}).get(tkr, None)
      
        if cached is None:
      
            if not self.quiet:
      
                print(f"[WARN] cache-miss analyst forecast {tkr} (returning empty)")
      
            return pd.DataFrame()
      
        return cached.copy()


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
        
        ann_num = income_ann_df.map(self._parse_growth).ffill(axis = 1).fillna(0)

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
    def forecast(
        self
    ) -> dict[str,pd.DataFrame]:
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

        Advantages
        ----------
        - Hierarchical fallback (analyst first, synthetic second) maximises coverage.
    
        - Common output schema enables direct panel joins without per-ticker guards.
    
        - Synthetic generation preserves scenario spread via low/average/high bands.
        
        """
        
        out = {}

        for t in self.tickers:

            try:
                
                df_a = self._read_analyst_sheet(
                    path = Path(),
                    tkr = t
                )

                if df_a is not None and not df_a.empty:
              
                    df = df_a
              
                else:
              
                    ann_df = self.income_ann[t]
              
                    rev = self.annuals[t]["Revenue"]
              
                    eps = self.annuals[t]["EPS"]
              
                    df = self._make_synthetic_forecast(
                        tkr = t,
                        rev_hist = rev,
                        eps_hist = eps,
                        income_ann_df = ann_df,
                    )

                out[t] = df

            except Exception as e:

                if not self.quiet: 
                    
                    print(f"[WARN] forecast {t}: {e}")
                
                out[t] = self._empty_forecast()
        
        return out


    @cached_property
    def kpis(
        self
    ) -> dict[str,pd.DataFrame]:
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

        Advantages
        ----------
        - EWMA damping reduces sensitivity to stale extrema while retaining recency.
    
        - Geometric ROE aggregation respects compounding behaviour of returns on
          equity over multi-year windows.
    
        - Consolidating leverage, valuation, and payout metrics supports coherent
          downstream cost-of-capital and terminal-value assumptions.
    
        """       
        
        out = {}
       
        for t in self.tickers:
       
            try:
               
                ratios_df = self.ratios[t]
               
                capex = self.cash[t].map(_parse_abbrev).loc["Capital Expenditures"]
               
                rev_hist = self.annuals[t]["Revenue"]
               
                last_idx = self.annuals[t].index[-1]
               
                bvps0 = self.bal[t].map(_parse_abbrev).loc["Book Value Per Share"].iat[-1]
                
                tax_rate = self.income[t].loc['Effective Tax Rate']
               
                out[t] = self._compute_kpis(t, ratios_df, capex, rev_hist, last_idx, bvps0, tax_rate)
       
            except Exception as e:
               
                if not self.quiet: 
               
                    print(f"[WARN] kpis {t}: {e}")
                
                out[t] = self._empty_kpis()
     
        return out
    
    
    def payout_hist(
        self
    ):
        """
        Extract historical payout-ratio series for each ticker.

        Method
        ------
        For each ticker, select row `"Payout Ratio"` from `ratios[ticker]`, then
        apply forward-fill, backward-fill, and final zero-fill:
        `x_clean = ffill(bfill(x_raw)).fillna(0)`.

        Returns
        -------
        dict[str, pd.Series | None]
            Mapping ticker -> cleaned payout-ratio history, or `None` when
            extraction fails.

        Advantage
        ---------
        Normalised histories simplify downstream summary statistics and avoid
        repeated missing-data handling.
        """
        
        payout = {}
        
        for t in self.tickers:
            
            try:
            
                payout[t] = self.ratios[t].loc["Payout Ratio"].ffill().bfill().fillna(0)
                
            except Exception as e:
                
                payout[t] = None
                        
        return payout
    
    
    def roe_hist(
        self
    ):
        """
        Extract historical return-on-equity series for each ticker.

        Method
        ------
        For each ticker, select `"Return on Equity (ROE)"` from ratio sheets and
        apply:
        `x_clean = ffill(bfill(x_raw)).fillna(0)`.

        Returns
        -------
        dict[str, pd.Series | None]
            Ticker-indexed dictionary of ROE histories, with `None` for failures.
        """
        
        roe = {}
        
        for t in self.tickers:
            
            try:
            
                roe[t] = self.ratios[t].loc["Return on Equity (ROE)"].ffill().bfill().fillna(0)
                
            except Exception as e:
                
                roe[t] = None
                        
        return roe


    def roa_hist(
        self
    ):
        """
        Extract historical return-on-assets series for each ticker.

        Method
        ------
        For each ticker, select `"Return on Assets (ROA)"`, then apply:
        `x_clean = ffill(bfill(x_raw)).fillna(0)`.

        Returns
        -------
        dict[str, pd.Series | None]
            Ticker-indexed dictionary of ROA histories, with `None` where data are
            unavailable.
        """
        
        roa = {}
        
        for t in self.tickers:
            
            try:
            
                roa[t] = self.ratios[t].loc["Return on Assets (ROA)"].ffill().bfill().fillna(0)
                
            except Exception as e:
                
                roa[t] = None
                        
        return roa
    
    
    def assets_noa_hist(
        self
    ):
        """
        Build adjusted tangible-asset and net-operating-asset histories by ticker.

        Definitions
        -----------
        Let:
        - `TA = TotalAssets`
        - `GW = GoodwillAndIntangibles`
        - `STI = ShortTermInvestments`
        - `LTI = LongTermInvestments`
        - `TL = TotalLiabilities`
        - `CD = CurrentDebt`
        - `LD = LongTermDebt`

        Then:
        - Tangible assets:
          `Assets_ex_intangibles = TA - GW`
        - Net operating assets:
          `NOA = Assets_ex_intangibles - STI - LTI - (TL - CD - LD)`

        Both series are passed through `currency_adjustment`.

        Returns
        -------
        tuple[dict[str, pd.Series | None], dict[str, pd.Series | None]]
            `(assets_dict, noa_dict)` keyed by ticker.

        Modelling motivation
        --------------------
        Removing financial investments and non-operating financing liabilities yields
        an operating-capital view suitable for profitability and valuation analyses.
        """
        
        assets = {}
        
        noa = {}
        
        for t in self.tickers:
            
            try:
            
                tot_assets_t = self.bal[t].loc["Total Assets"].ffill().bfill().fillna(0)
                
                int_assets_t = self.bal[t].loc["Goodwill and Intangibles"].ffill().bfill().fillna(0)
                
                assets_t = tot_assets_t - int_assets_t
                
                assets[t] = self.currency_adjustment(
                    ticker = t, 
                    s = assets_t
                )
                
                short_inv_t = self.bal[t].loc["Short-TermInvestments"].ffill().bfill().fillna(0)
                
                long_inv_t = self.bal[t].loc["Long-Term Investments"].ffill().bfill().fillna(0)
                
                tot_liab_t = self.bal[t].loc["Total Liabilities"].ffill().bfill().fillna(0)

                curr_debt_t = self.bal[t].loc["Current Debt"].ffill().bfill().fillna(0)

                long_debt_t = self.bal[t].loc["Long Term Debt"].ffill().bfill().fillna(0)

                noa_t = assets_t - short_inv_t - long_inv_t - (tot_liab_t - curr_debt_t - long_debt_t)    
                            
                noa[t] = self.currency_adjustment(
                    ticker = t, 
                    s = noa_t
                )
                
            except Exception as e:
                
                assets[t] = None
                
                noa[t] = None
                        
        return assets, noa
    
    
    def _compute_kpis(
        self, 
        tkr, 
        ratios_df, 
        capex, 
        rev, 
        annual_index, 
        bvps_0,
        tax_rate
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

        Technique advantages
        --------------------
        - EWMA smooths transitory accounting noise without discarding history.
    
        - Log-domain ROE compounding (`exp(mean(log(1 + roe))) - 1`) is numerically
          stable for multi-period geometric averaging.
    
        - Concurrent inclusion of market-value debt and valuation multiples captures
          both capital-structure and pricing dimensions in one coherent state vector.
    
        """
    
        ev_rev = pd.to_numeric(ratios_df.loc["EV/Revenue"], errors = "coerce")
       
        evs_t = ev_rev.ewm(span = 20, adjust = False).mean().iat[-1]
       
        pe = pd.to_numeric(ratios_df.loc["PE Ratio"], errors = "coerce")
       
        ps = pd.to_numeric(ratios_df.loc["PS Ratio"], errors = "coerce")
       
        ptb = pd.to_numeric(ratios_df.loc["PB Ratio"], errors = "coerce")
        
        evfcf = pd.to_numeric(ratios_df.loc["EV/FCF"], errors = "coerce")
        
        evebitda = pd.to_numeric(ratios_df.loc["EV/EBITDA"], errors = "coerce")
        
        evebit = pd.to_numeric(ratios_df.loc["EV/EBIT"], errors = "coerce")
        
        d_to_e = pd.to_numeric(ratios_df.loc['Debt/Equity'], errors = "coerce")
        
        tax_rate_avg = tax_rate.tail(20).mean()
        
        d_to_e_mean = d_to_e.tail(20).mean()
        
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
           
            roe_arr = np.asarray(roe_list, dtype=float)
        
            roe_arr = roe_arr[np.isfinite(roe_arr)]

            valid = roe_arr > -0.999

            if valid.sum() >= 2:
       
                roe = float(np.exp(np.mean(np.log1p(roe_arr[valid]))) - 1.0)
       
            elif valid.sum() == 1:
       
                roe = float(roe_arr[valid][0])
       
            else:
                
                if roe_arr.size:
                    
                    roe = float(np.nanmean(roe_arr))  
                
                else:
                    
                    roe = np.nan
       
        else:
       
            roe = self.ind_dict["ROE"][tkr]["Region-Industry"] or 0.0

        payout = abs(self.ratios_ann[tkr].loc["Payout Ratio"].map(_parse_abbrev).dropna().tail(5).mean())

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
            "d_to_e_mean": d_to_e_mean,
            "tax_rate_avg": tax_rate_avg,
        }, index = [annual_index])


    @cached_property
    def prophet_data(
        self
    ) -> dict[str,pd.DataFrame]:
        """
        Construct cleaned per-ticker Revenue/EPS series for forecasting models.

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

        Modelling rationale
        -------------------
        - Restricting to modern history reduces structural breaks from legacy
          reporting regimes.
     
        - Forward-fill then zero-fill yields a deterministic dense design matrix for
          time-series learners that do not natively support ragged observations.
     
        - Currency harmonisation improves cross-ticker comparability of scale.
                
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


    def fundamentals_asof_weekly(
        self,
        *,
        tickers: Sequence[str],
        weekly_index: pd.DatetimeIndex,
        report_lag_days: int = 90,
    ) -> dict[str, pd.DataFrame]:
        """
        Build point-in-time weekly Revenue/EPS panels aligned to report dates.

        Each ticker's annual fundamentals are shifted by a reporting lag and then
        aligned to the provided weekly index using merge_asof (last observation
        carried forward only after the report date).

        Parameters
        ----------
        tickers : Sequence[str]
            Tickers to process.
        weekly_index : pd.DatetimeIndex
            Weekly dates to align to (e.g., Friday week-ends).
        report_lag_days : int, default 90
            Reporting lag added to annual report dates before alignment.

        Returns
        -------
        dict[str, pd.DataFrame]
            Ticker → DataFrame indexed by weekly dates with columns
            ["Revenue", "EPS (Basic)"] where available.

        Alignment equation
        ------------------
        For weekly date `w`, exposure uses:
        `x_asof(w) = x(t*)` where
        `t* = max{ t_report + lag : t_report + lag <= w }`.

        Advantages
        ----------
        - Eliminates look-ahead leakage by enforcing report-date availability.
    
        - Produces a weekly panel compatible with return-frequency modelling.
    
        - `merge_asof` is computationally efficient for monotone time indices.
        
        """

        if weekly_index is None or len(weekly_index) == 0:
     
            return {}

        weekly_index = pd.DatetimeIndex(weekly_index).sort_values()

        out: dict[str, pd.DataFrame] = {}

        for t in list(tickers):
       
            ann = self.annuals.get(t)
       
            if ann is None or ann.empty:
       
                out[t] = pd.DataFrame()
       
                continue

            cols = []
        
            if "Revenue" in ann.columns:
        
                cols.append("Revenue")
        
            if "EPS" in ann.columns:
        
                cols.append("EPS")

            if not cols:
          
                out[t] = pd.DataFrame()
          
                continue

            df = ann[cols].copy()
          
            df = df.sort_index()
          
            df.index = pd.to_datetime(df.index) + pd.Timedelta(days = report_lag_days)
          
            df = df.reset_index().rename(columns = {"index": "ds", "EPS": "EPS (Basic)"})

            wk = pd.DataFrame({"ds": weekly_index})

            merged = pd.merge_asof(wk, df, on = "ds", direction = "backward")

            merged = merged.set_index("ds")

            if "EPS (Basic)" not in merged.columns and "EPS" in merged.columns:

                merged = merged.rename(columns = {"EPS": "EPS (Basic)"})

            out[t] = merged

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

        Advantage
        ---------
        Consolidating external analyst consensus and internal forecast state in a
        single table supports rapid forecast-consistency diagnostics.
                
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

        Interpretation advantage
        ------------------------
        Expressing revisions in relative terms normalises across issuers with very
        different absolute revenue and EPS scales.
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

        Advantages
        ----------
        - Common annual period index simplifies model fitting and validation.
     
        - Inner joins restrict estimation to jointly observed firm-macro periods,
          reducing implicit missing-data assumptions.
        
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
       
                macro_tkr = macro_hist.xs(tkr, level = "ticker")
       
            except KeyError:
       
                continue

            joined = df.join(macro_tkr, how = "inner")
    
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

        Advantages
        ----------
        - Symmetric window averaging attenuates event-day microstructure noise.
      
        - Joint reporting of fundamental growth and return proxies supports direct
          reduced-form linkage analyses.
      
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
    
            ret_g = mean_price.ffill().pct_change(fill_method = None)

            df = pd.DataFrame({
                "Revenue Growth": rev_g,
                "EPS Growth": eps_g,
                "Return": ret_g,
            }).dropna()
    
            growth[tkr] = df

        return growth    
            
    
