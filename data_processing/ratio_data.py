"""
Loads industry/sector valuation ratios, analyst data and price history, providing helper methods for region mapping and index returns.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Sequence
from collections import OrderedDict
from maps.index_exchange_mapping import ExchangeMapping
from maps.ccy_exchange_map import _CCY_BY_SUFFIX, _EU_SUFFIXES, _REGION_FACTOR_MAP, _FACTORS, _FACTOR_LABEL_TO_CODE, _USD_EXCEPTIONS
import config


class RatioData:
    """
    RatioData

    Central loader/adapter for valuation ratios, analyst snapshots, prices/returns,
    time series data for factor models and group-level aggregates (industry, sector, 
    region, market-cap buckets). Exposes convenience helpers to align group signals
    (Sharpe/momentum), construct factor inputs, and map tickers to benchmark indexes.

    Responsibilities
    ----------------
    • Load structured inputs from three workbooks:
    
        path1 = 'ind_data_mc_all_simple_mean.xlsx'
            - industry
            - sector × MC group
            - region × industry
            - region × sector tables
            - MC bucket bounds
            
        path2 = config.FORECAST_FILE
            - Analyst per-ticker data
            - per-group summaries (industry/sector/index),
            - factor ETF metadata ('Factor ETFs')
            
        path3 = config.DATA_FILE
            - Panels of closes
            - returns (daily/weekly)
            - rindex/sector/industry closes,
            - factor ETF returns
            
    • Provide helpers to:
        
        - map countries to regions and Fama-French regions,
       
        - classify market-cap buckets,
       
        - assemble per-ticker design matrices for factor regressions in the ticker’s
          local currency (including FX alignment),
       
        - compute index, sector, and industry momentum and Sharpe proxies,
       
        - select regional factor returns/expectations and apply FX translations,
       
        - fetch FX series from Yahoo Finance and construct pairwise FX return series.
        
    Currency & ticker conventions
    -----------------------------
    - Quote currency inference:
       
        1) explicit USD exceptions in `_USD_EXCEPTIONS` (e.g., 'EMVL.L', 'IWQU.L', 'IWSZ.L'),
       
        2) suffix mapping `_CCY_BY_SUFFIX`,
       
        3) default to USD when unknown.
    
    - Regional factor selection:
        `_factor_region_for_ticker` uses suffixes `_EU_SUFFIXES` and special handling for
        Canadian tickers ('.TO' → 'US' factor set). Regional factor returns are mapped via
        `_REGION_FACTOR_MAP` and canonicalised to MTUM, QUAL, SIZE, USMV, VLUE.

    Time conventions
    ----------------
    - Daily → weekly/quarterly resampling uses `.resample('W-XXX').last()` or
      `.resample('QE').last()` and simple percentage change unless specified
      (e.g., log-sums in `factor_weekly_rets`).


    Key attributes
    --------------
    today : datetime.date
    path1, path2, path3 : pathlib.Path
    year_ago, five_year_ago : pandas.Timestamp/str (from config)
    industry, sector_mc, region_ind, region_sec, bounds, industry_mc : pd.DataFrame
    analyst, sector_data, industry_data, index_data, factor_data : pd.DataFrame
    close, weekly_close, daily_rets, weekly_rets, index_close, sector_close, industry_close : pd.DataFrame
    quarterly_close, quarterly_rets : pd.DataFrame
    last_price : pd.Series
    mcap, shares_outstanding, tax_rate : pd.Series
    currency : pd.Series
    tickers : Sequence[str]

    Notes
    -----
    - All frames are sorted ascending by index where relevant.
    - Returns are simple returns unless noted; some methods log-sum weekly returns.
    - No network calls happen here except where explicitly stated (e.g., FX via yfinance).
    """
    
    INDEX_ROUTING = {
        'NasdaqGS': '^NDX',
        'NasdaqGM': '^NDX',
        'NasdaqCM': '^NDX',
        'LSE': '^FTSE',
        'NYSE': '^GSPC',
        'XETRA': '^GDAXI',
        'MCE': '^IBEX',
        'Amsterdam': '^AEX',
        'Paris': '^FCHI',
        'Toronto': '^GSPTSE',
        'HKSE': '^HSI',
        'Swiss': '^SSMI',
    }

    _ATTR_LOADERS = {
        # path1 group tables
        "industry": "_load_group_tables",
        "sector_mc": "_load_group_tables",
        "region_ind": "_load_group_tables",
        "region_sec": "_load_group_tables",
        "bounds": "_load_group_tables",
        "industry_mc": "_load_group_tables",

        # path2 analyst / meta
        "analyst": "_load_analyst_tables",
        "sector_data": "_load_analyst_tables",
        "industry_data": "_load_analyst_tables",
        "index_data": "_load_analyst_tables",
        "factor_data": "_load_analyst_tables",
        "mcap": "_load_analyst_tables",
        "country": "_load_analyst_tables",
        "sector": "_load_analyst_tables",
        "shares_outstanding": "_load_analyst_tables",
        "tax_rate": "_load_analyst_tables",
        "last_price": "_load_analyst_tables",

        # path3 core market panels
        "close": "_load_market_panels",
        "open": "_load_market_panels",
        "weekly_close": "_load_market_panels",
        "weekly_rets": "_load_market_panels",
        "quarterly_close": "_load_market_panels",
        "quarterly_rets": "_load_market_panels",
        "index_close": "_load_market_panels",
        "daily_rets": "_load_market_panels",
        "sector_close": "_load_market_panels",
        "industry_close": "_load_market_panels",
        "factor_rets": "_load_market_panels",

        # path3 other sheets
        "macro_data": "_load_macro_sheet",
        "currency": "_load_currency_sheet",
        "fx_usd_per_ccy": "_load_fx_sheet",
    }
    def __init__(
        self
    ):
        """
        Initialise static paths and date anchors from `config`, then call `_load()`
        to populate all public attributes.

        Side effects
        ------------
        - Sets: today, path1/2/3, year_ago, five_year_ago.
        - Triggers workbook IO through `_load()`.

        Raises
        ------
        Any exception propagated by `_load()` (e.g., missing files/sheets).
        """

        self.today = config.TODAY
        self.path1 = config.BASE_DIR / "ind_data_mc_all_simple_mean.xlsx"
        self.path2 = config.FORECAST_FILE
        self.path3 = config.DATA_FILE
        self.year_ago = config.YEAR_AGO
        self.five_year_ago = config.FIVE_YEAR_AGO

        # universe
        self.tickers = config.tickers

        # lazy-load flag
        self._loaded = False

        # caches
        self._index_returns_cache: tuple[pd.Series, pd.DataFrame, pd.DataFrame] | None = None
        self._factor_weekly_cache: dict[tuple[str, pd.Timestamp | None], pd.DataFrame] = {}
        self._usd_per_one_cache: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.Series] = {}
        self._fx_ret_cache: dict[tuple[str, str, pd.Timestamp, pd.Timestamp, int], pd.Series] = {}
        
        self._loaded = False           
        self._loaded_path1 = False   
        self._loaded_path2 = False        
        self._loaded_market_panels = False 
        self._loaded_currency = False    
        self._loaded_fx = False          
        self._loaded_macro = False  


    def _load(
        self
    ):
        """
        Load all dependent workbooks and derive secondary structures.

        IO
        --
        - path1 ('ind_data_mc_all_simple_mean.xlsx'):
           
            Sheets: 'Industry', 'Sector MC Group', 'Region-Industry', 'Region-Sector',
                    'MC Bounds', 'Industry MC Group'.
            
            Renames 'Industry_grouped' → 'Industry' where applicable and sets multi-indices.
       
        - path2 (config.FORECAST_FILE):
       
            Sheets: 'Analyst Data', 'Analyst Target', 'Sector Data', 'Industry Data',
                    'Index Data', 'Factor ETFs'. Stores frequently used columns as
           
            convenience series (e.g., marketCap, sharesOutstanding, Tax Rate).
       
        - path3 (config.DATA_FILE):
           
            Sheets: 'Close', 'Weekly Close', 'Historic Returns', 'Historic Weekly Returns',
                    'Currency', 'Index Close', 'Sector Close', 'Industry Close', 'Factor Returns'.

        Loads
        -----
        - Group ratio tables and MC bounds from `path1` (industry/sector/region views).
       
        - Analyst/target/aggregate performance tables from `path2`.
       
        - Price/return panels (close, returns, index/sector/industry/factors) from `path3`.

        Post-processing
        ---------------
        - Normalises key index levels (e.g., renames 'Industry_grouped' → 'Industry').
       
        - Sets default tax rate to 22% if missing.
       
        - Sorts time series ascending.
       
        - Creates quarterly closes/returns.
       
        - Caches 'Current Price' from 'Analyst Target' as `last_price`.
        
        - Quarterly closes: `quarterly_close = close.resample('QE').last()`
        and returns `quarterly_rets = quarterly_close.pct_change()`.

        Returns
        -------
        None

        Raises
        ------
        - Any pandas/openpyxl IO errors on missing files/sheets.
        - KeyError if required columns are missing in the provided workbooks.
        """

        if self._loaded:
    
            return
    
        self._load_group_tables()
    
        self._load_analyst_tables()
    
        self._load_market_panels()
    
        self._load_currency_sheet()
    
        self._load_fx_sheet()
    
        self._load_macro_sheet()
    
        self._loaded = True


    def _load_group_tables(
        self
    ) -> None:
        """
        Load group-level lookup tables from `self.path1`.

        This reads the six structural sheets used by ratio aggregation logic and
        prepares their indices so downstream lookups can use direct `.reindex`
        calls on scalar keys or MultiIndex tuples.

        Loaded attributes
        -----------------
        industry : pd.DataFrame
            Industry-level metrics indexed by `Industry` (after renaming
            `Industry_grouped` to `Industry` when present).
        sector_mc : pd.DataFrame
            Sector x market-cap bucket table indexed by
            (`Sector`, `MC_Group_Merged`).
        region_ind : pd.DataFrame
            Region x industry table indexed by (`Region`, `Industry`).
        region_sec : pd.DataFrame
            Region x sector table indexed by (`Region`, `Sector`).
        bounds : pd.DataFrame
            Market-cap bounds indexed by `MC_Group_Merged`.
        industry_mc : pd.DataFrame
            Industry x market-cap bucket table indexed by
            (`Industry`, `MC_Group_Merged`).

        Returns
        -------
        None

        Notes
        -----
        - Idempotent: returns immediately when `_loaded_path1` is already `True`.
    
        - Exceptions from `pandas.read_excel` propagate unchanged.
    
        """
    
        if self._loaded_path1:
    
            return

        sheets1 = pd.read_excel(
            self.path1,
            sheet_name=[
                "Industry",
                "Sector MC Group",
                "Region-Industry",
                "Region-Sector",
                "MC Bounds",
                "Industry MC Group",
            ],
            engine = "openpyxl",
        )

        df_ind = sheets1["Industry"].rename(columns={"Industry_grouped": "Industry"})
      
        self.industry = df_ind.set_index("Industry")

        self.sector_mc = sheets1["Sector MC Group"].set_index(["Sector", "MC_Group_Merged"])

        df_ri = sheets1["Region-Industry"].rename(columns={"Industry_grouped": "Industry"})
      
        self.region_ind = df_ri.set_index(["Region", "Industry"])

        self.region_sec = sheets1["Region-Sector"].set_index(["Region", "Sector"])

        self.bounds = sheets1["MC Bounds"].set_index("MC_Group_Merged")

        self.industry_mc = sheets1["Industry MC Group"].set_index(["Industry", "MC_Group_Merged"])

        self._loaded_path1 = True


    def _load_analyst_tables(
        self
    ) -> None:
        """
        Load ticker- and group-level analyst metadata from `self.path2`.

        Reads:
        - `Analyst Data`
        - `Analyst Target`
        - `Sector Data`
        - `Industry Data`
        - `Index Data`
        - `Factor ETFs`

        and exposes both raw tables and frequently used derived series.

        Loaded attributes
        -----------------
        analyst, sector_data, industry_data, index_data, factor_data : pd.DataFrame
        mcap : pd.Series
            `analyst['marketCap']` cast to float64.
        country : pd.Series
            `analyst['country']`.
        sector : pd.Series
            `analyst['Sector']`.
        shares_outstanding : pd.Series
            `analyst['sharesOutstanding']`.
        tax_rate : pd.Series
            `analyst['Tax Rate']` with missing values filled to `0.22`, cast to float32.
        last_price : pd.Series
            `Analyst Target['Current Price']`.

        Returns
        -------
        None

        Notes
        -----
        - Idempotent: returns immediately when `_loaded_path2` is already `True`.
      
        - Exceptions from `pandas.read_excel` propagate unchanged.
      
        """
     
        if self._loaded_path2:
     
            return

        sheets2 = pd.read_excel(
            self.path2,
            sheet_name = [
                "Analyst Data",
                "Analyst Target",
                "Sector Data",
                "Industry Data",
                "Index Data",
                "Factor ETFs",
            ],
            index_col = 0,
            engine = "openpyxl",
        )

        self.analyst = sheets2["Analyst Data"]
       
        temp_target = sheets2["Analyst Target"]
       
        self.sector_data = sheets2["Sector Data"]
       
        self.industry_data = sheets2["Industry Data"]
       
        self.index_data = sheets2["Index Data"]
       
        self.factor_data = sheets2["Factor ETFs"]

        self.mcap = self.analyst["marketCap"].astype("float64")

        self.country = self.analyst["country"]

        self.sector = self.analyst["Sector"]

        self.shares_outstanding = self.analyst["sharesOutstanding"]

        self.tax_rate = self.analyst["Tax Rate"].fillna(0.22).astype("float32")

        self.last_price = temp_target["Current Price"]

        self._loaded_path2 = True


    def _load_market_panels(
        self
    ) -> None:
        """
        Load market price/return panels from `self.path3`.

        Reads close/open panels, precomputed return panels, and index/sector/
        industry/factor closes. Also derives quarterly close and quarterly return
        panels from daily closes.

        Loaded attributes
        -----------------
        close, open, weekly_close : pd.DataFrame
            Price-level panels sorted by ascending datetime index.
        daily_rets, weekly_rets : pd.DataFrame
            Historic simple returns from workbook sheets.
        quarterly_close, quarterly_rets : pd.DataFrame
            Derived as `close.resample('QE').last()` and `pct_change()`.
        index_close, sector_close, industry_close : pd.DataFrame
            Group/index price panels sorted by date.
        factor_rets : pd.DataFrame
            Daily factor ETF returns panel.

        Returns
        -------
        None

        Notes
        -----
        - Idempotent: returns immediately when `_loaded_market_panels` is `True`.
     
        - Exceptions from `pandas.read_excel` propagate unchanged.
     
        """
    
        if self._loaded_market_panels:
    
            return

        sheets3 = pd.read_excel(
            self.path3,
            sheet_name = [
                "Close",
                "Weekly Close",
                "Open",
                "Historic Returns",
                "Historic Weekly Returns",
                "Index Close",
                "Sector Close",
                "Industry Close",
                "Factor Returns",
            ],
            index_col = 0,
            engine = "openpyxl",
        )

        self.close = sheets3["Close"].sort_index(ascending = True).astype("float32")
     
        self.open = sheets3["Open"].sort_index(ascending = True).astype("float32")

        self.weekly_close = sheets3["Weekly Close"].sort_index(ascending = True).astype("float32")
       
        self.weekly_rets = sheets3["Historic Weekly Returns"].sort_index(ascending = True).astype("float32")

        self.quarterly_close = self.close.resample("QE").last()
      
        self.quarterly_rets = self.quarterly_close.pct_change()

        self.index_close = sheets3["Index Close"].sort_index().astype("float32")

        self.daily_rets = sheets3["Historic Returns"].sort_index(ascending = True).astype("float32")

        self.sector_close = sheets3["Sector Close"].sort_index(ascending = True).astype("float32")
       
        self.industry_close = sheets3["Industry Close"].sort_index(ascending = True).astype("float32")

        self.factor_rets = sheets3["Factor Returns"].sort_index(ascending = True).astype("float32")

        self._loaded_market_panels = True


    def _load_macro_sheet(
        self
    ) -> None:
        """
        Load and weekly-resample macro data from `self.path3`.

        The source sheet (`Macro Data`) is read at its native frequency, sorted
        by datetime index, then resampled to weekly end-of-period values using
        `.resample('W').last()`.

        Loaded attributes
        -----------------
        macro_data : pd.DataFrame
            Weekly macro panel sorted by ascending datetime index.

        Returns
        -------
        None

        Notes
        -----
        - Idempotent: returns immediately when `_loaded_macro` is `True`.
    
        """
    
        if self._loaded_macro:
    
            return

        macro_df = pd.read_excel(
            self.path3,
            sheet_name = "Macro Data",
            index_col = 0,
            engine = "openpyxl",
        ).sort_index(ascending = True)

        macro_weekly = macro_df.resample("W").last().sort_index(ascending=True)
     
        self.macro_data = macro_weekly

        self._loaded_macro = True


    def _load_currency_sheet(
        self
    ) -> None:
        """
        Load per-ticker currency codes from the `Currency` sheet in `self.path3`.

        The method expects a column named `Last` and stores it as a Series for
        downstream ticker-to-currency alignment.

        Loaded attributes
        -----------------
        currency : pd.Series
            Currency code (or currency-like label) per ticker.

        Returns
        -------
        None

        Notes
        -----
        - Idempotent: returns immediately when `_loaded_currency` is `True`.
     
        - Raises `KeyError` if the `Last` column is absent.
     
        """
    
        if self._loaded_currency:
    
            return

        df = pd.read_excel(
            self.path3,
            sheet_name = "Currency",
            index_col = 0,
            engine = "openpyxl",
        )

        self.currency = df["Last"]

        self._loaded_currency = True


    def _load_fx_sheet(
        self
    ) -> None:
        """
        Load cached FX level data (`USD_per_<CCY>`) from `self.path3`.

        The method attempts to read sheet `FX USD per CCY`. If the sheet is
        missing, it stores an empty DataFrame so FX-dependent methods can fail
        explicitly later with informative key errors.

        Loaded attributes
        -----------------
        fx_usd_per_ccy : pd.DataFrame
            Daily FX level panel sorted by index. Index is coerced to datetime
            when possible.

        Returns
        -------
        None

        Notes
        -----
        - Idempotent: returns immediately when `_loaded_fx` is `True`.
     
        - Missing-sheet handling is intentional: `ValueError` from
          `read_excel(..., sheet_name='FX USD per CCY')` is converted into an
          empty DataFrame.
     
        """
    
        if self._loaded_fx:
    
            return

        try:
        
            df = pd.read_excel(
                self.path3,
                sheet_name="FX USD per CCY",
                index_col=0,
                engine="openpyxl",
            )
    
        except ValueError:
    
            df = pd.DataFrame()

        self.fx_usd_per_ccy = df.sort_index(ascending=True)
    
        try:
    
            self.fx_usd_per_ccy.index = pd.to_datetime(self.fx_usd_per_ccy.index)
    
        except Exception:
    
            pass

        self._loaded_fx = True

    
    def _ensure_loaded(
        self
    ) -> None:
        """
        Backwards-compat stub.
        Data is loaded lazily per attribute in __getattr__.
        Call self._load() explicitly if you really want an eager full load.
                
        """
    
        return

    def __getattr__(
        self, 
        name: str
    ):
        """
        Lazily materialise known data attributes on first access.

        This hook is invoked only when normal attribute lookup fails. If `name`
        appears in `_ATTR_LOADERS`, the mapped loader is executed, then the
        attribute is looked up again and returned.

        Parameters
        ----------
        name : str
            Missing attribute requested by the caller.

        Returns
        -------
        Any
            The lazily loaded attribute value.

        Raises
        ------
        AttributeError
            If `name` is unknown, or if the selected loader runs but fails to set
            the expected attribute.
       
        """
    
        loader_name = self._ATTR_LOADERS.get(name)
    
        if loader_name is not None:
    
            loader = getattr(self, loader_name)
    
            loader()

            try:

                return object.__getattribute__(self, name)

            except AttributeError:

                raise AttributeError(
                    f"{type(self).__name__!s} loader {loader_name} did not set attribute {name!r}"
                )

        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")


    @staticmethod
    def determine_region(
        country
    ):
        """
        Heuristic mapping from a free-form country string to a coarse region bucket.

        Mapping (case-insensitive)
        --------------------------
        'United States' or 'USA' → 'United States'
        
        {'Germany','France','Spain','Europe','Italy','United Kingdom','Ireland'} → 'Europe'
        
        {'Australia','Canada'} → 'Canada/Australia'
        
        {'China','Hong Kong','India','Japan','South Korea','Taiwan','Singapore',
        'Thailand','Vietnam'} → 'Asia'
        
        anything else or non-string → 'Emerging Mkts (ex-Asia)'

        Parameters
        ----------
        country : Any
            Country name; case-insensitive. Non-str values map to 'Emerging Mkts (ex-Asia)'.

        Returns
        -------
        str
            One of {'United States', 'Europe', 'Canada/Australia', 'Asia', 'Emerging Mkts (ex-Asia)'}.

        Notes
        -----
        - Uses substring checks on a lowercased string; conservative fallbacks on unknowns.
                
        """
      
        if not isinstance(country, str):
           
            return 'Emerging Mkts (ex-Asia)'
      
        c = country.strip().lower()
      
        if 'united states' in c or 'usa' in c:
            
            return 'United States'
      
        elif any(w in c for w in ['germany', 'france', 'spain', 'europe', 'italy', 'united kingdom', 'ireland']):
          
            return 'Europe'
      
        elif any(w in c for w in ['australia', 'canada']):
           
            return 'Canada/Australia'
      
        elif any(w in c for w in ['china', 'hong kong', 'india', 'japan', 'south korea', 'taiwan', 'singapore', 'thailand', 'vietnam']):
           
            return 'Asia'
      
        else:
           
            return 'Emerging Mkts (ex-Asia)'


    @staticmethod
    def determine_region_fama(
        country
    ):
        """
        Map a country label to a Fama-French-style regional bucket.

        Mapping (substring-based)
        -------------------------
        - North America keywords -> `'North_America'`
        - Europe keywords -> `'Europe'`
        - Asia-Pacific keywords -> `'Asia_Pacific'`
        - Japan keyword -> `'Japan'`
        - Otherwise -> `'Emerging_Markets'`

        Parameters
        ----------
        country : Any
            Country name or free-form text. Non-string inputs return
            `'Emerging Mkts (ex-Asia)'`.

        Returns
        -------
        str
            One of:
            `{'North_America', 'Europe', 'Asia_Pacific', 'Japan',
            'Emerging_Markets', 'Emerging Mkts (ex-Asia)'}`.

        Notes
        -----
        - Matching is implemented with substring checks on a lowercased input.
      
        - Keyword lists should remain lowercase for reliable matches.
      
        - For unknown strings, the fallback is `'Emerging_Markets'`.
      
        """
      
        if not isinstance(country, str):
            
            return 'Emerging Mkts (ex-Asia)'
      
        c = country.strip().lower()
        
        if any(w in c for w in ['United States', 'Canada']):
            
            return 'North_America'
        
        if any(w in c for w in ['Germany', 'France', 'Spain', 'Europe', 'Italy', 'United Kingdom', 'Ireland']):
            
            return 'Europe'
        
        if any(w in c for w in ['China', 'Hong Kong', 'India', 'South Korea', 'Taiwan', 'Singapore', 'Thailand', 'Vietnam', 'Australia']):
            
            return 'Asia_Pacific'
        
        if 'Japan' in c:
            
            return 'Japan'
      
        else:
            return 'Emerging_Markets'
        
    
    def default(
        self, 
        val, 
        fallback
    ):
        """
        Return `val` if it is not NA; otherwise return `fallback`.

        Parameters
        ----------
        val : Any
        fallback : Any

        Returns
        -------
        Any
        
        """
        
        if pd.notna(val):
            
            return val 
        
        else:
            
            return fallback

    
    def mc_group(
        self, 
    ):
        """
        Assign each ticker to a market-cap bucket using configured bounds.

        For each ticker market cap `cap` and each bound row `(low, high)`, the
        first matching interval `low <= cap < high` determines the bucket label.
        Tickers that do not match any interval (or have missing caps) default to
        `'Mid-Cap'`.

        Parameters
        ----------
        None
            Uses `self.mcap` and `self.bounds` loaded from workbook data.

        Returns
        -------
        pd.Series
            Index = ticker, name = `MC_Group_Merged`, values = matched bucket
            labels.

        Notes
        -----
        - Lower bound is inclusive; upper bound is exclusive.
      
        - Vectorised NumPy masking is used for performance across all tickers.
                
        """
      
        self._ensure_loaded()

        caps = self.mcap.astype(float)
      
        bounds = self.bounds[["Low Bound", "High Bound"]].astype(float)

        cap_vals = caps.values.reshape(-1, 1)
      
        lo = bounds["Low Bound"].values.reshape(1, -1)
      
        hi = bounds["High Bound"].values.reshape(1, -1)

        mask = (cap_vals >= lo) & (cap_vals < hi)
      
        idx = mask.argmax(axis = 1)
      
        has_match = mask.any(axis = 1)

        groups = np.where(has_match, bounds.index.values[idx], "Mid-Cap")

        return pd.Series(groups, index = caps.index, name = "MC_Group_Merged")

            
    def dicts(
        self
    ):
        """
        Build a nested dictionary of core metrics for each ticker across four group lenses:
        'Sector-MC', 'Region-Industry', 'Region-Sector', and 'Industry-MC'.

        Metrics 
        -------
        ['eps1y_5','eps1y','rev1y_5','rev1y','PB','PE','PS','EVS','FPS','FEVS',
        'ROE','ROA','ROIC','PE10y','EVE','EVFCF','EVEBITDA','EVEBIT']

        Fallbacks
        ---------
        If a metric is NA, may fall back to its forward analogue:
        
            PS ← FPS
            
            EVS ← FEVS
       
        Additional fallbacks within region/industry lens:
       
        - Region-Industry → Industry
       
        - Industry-MC → Industry

        Returns
        -------
        dict[str, dict[str, dict[str, float]]]
            results[metric][ticker][bucket] = value

        Notes
        -----
        - Tickers sourced from `self.tickers`.
        
        - Region derived from `determine_region(country)`.
        
        - Market-cap bucket derived via `mc_group`.
        
        - Missing keys handled via try/except; NA preserved where no fallback resolves.
       
        """
      
        self._ensure_loaded()

        core_metrics = [
            "eps1y_5",
            "eps1y",
            "rev1y_5",
            "rev1y",
            "PB",
            "PE",
            "PS",
            "EVS",
            "FPS",
            "FEVS",
            "ROE",
            "ROA",
            "ROIC",
            "PE10y",
            "EVE",
            "EVFCF",
            "EVEBITDA",
            "EVEBIT",
        ]

        fallback_map = {"PS": "FPS", "EVS": "FEVS"}

        tickers = pd.Index(self.tickers)
     
        analyst = self.analyst.reindex(tickers)

        ticker_ind = analyst["Industry"]
     
        ticker_sec = analyst["Sector"]
     
        if "country" in analyst.columns:
     
            ticker_country = analyst["country"]
     
        else:
     
            ticker_country = pd.Series(index=tickers, dtype=object)

        ticker_country = ticker_country.reindex(tickers)

        unique_countries = ticker_country.dropna().unique()

        country_to_region = {c: self.determine_region(c) for c in unique_countries}

        region = ticker_country.map(country_to_region).fillna("Emerging Mkts (ex-Asia)")

        mc_series = self.mc_group().reindex(tickers)

        industry_table = self.industry[core_metrics]

        sector_mc_table = self.sector_mc[core_metrics]

        region_ind_table = self.region_ind[core_metrics]

        region_sec_table = self.region_sec[core_metrics]

        industry_mc_table = self.industry_mc[core_metrics]

        industry_vals = industry_table.reindex(ticker_ind)

        industry_vals.index = tickers

        for base, fb in fallback_map.items():

            if base in industry_vals.columns and fb in industry_vals.columns:

                mask = industry_vals[base].isna()

                industry_vals.loc[mask, base] = industry_vals.loc[mask, fb]

        sm_index = pd.MultiIndex.from_arrays(
            [ticker_sec.values, mc_series.values],
            names = ["Sector", "MC_Group_Merged"],
        )
      
        sector_mc_vals = sector_mc_table.reindex(sm_index)
      
        sector_mc_vals.index = tickers

        for base, fb in fallback_map.items():
         
            if base in sector_mc_vals.columns and fb in sector_mc_vals.columns:
         
                mask = sector_mc_vals[base].isna()
         
                sector_mc_vals.loc[mask, base] = sector_mc_vals.loc[mask, fb]

        rs_index = pd.MultiIndex.from_arrays(
            [region.values, ticker_sec.values],
            names = ["Region", "Sector"],
        )
        
        region_sec_vals = region_sec_table.reindex(rs_index)
        
        region_sec_vals.index = tickers

        for base, fb in fallback_map.items():
        
            if base in region_sec_vals.columns and fb in region_sec_vals.columns:
        
                mask = region_sec_vals[base].isna()
        
                region_sec_vals.loc[mask, base] = region_sec_vals.loc[mask, fb]

        im_index = pd.MultiIndex.from_arrays(
            [ticker_ind.values, mc_series.values],
            names = ["Industry", "MC_Group_Merged"],
        )
       
        industry_mc_vals = industry_mc_table.reindex(im_index)
       
        industry_mc_vals.index = tickers

        industry_mc_vals = industry_mc_vals.fillna(industry_vals)

        ri_index = pd.MultiIndex.from_arrays(
            [region.values, ticker_ind.values],
            names = ["Region", "Industry"],
        )

        region_ind_vals = pd.DataFrame(index=tickers, columns=core_metrics, dtype=float)

        non_fb_metrics = [m for m in core_metrics if m not in fallback_map]

        if non_fb_metrics:

            tmp = region_ind_table[non_fb_metrics].reindex(ri_index)

            tmp.index = tickers

            tmp = tmp.fillna(industry_vals[non_fb_metrics])

            region_ind_vals[non_fb_metrics] = tmp[non_fb_metrics]

        for base, fb in fallback_map.items():

            col_reg = region_ind_table.get(base)

            if col_reg is not None:

                col_reg = col_reg.reindex(ri_index)

                col_reg.index = tickers

            else:

                col_reg = pd.Series(np.nan, index=tickers)

            col_ind = industry_table.get(base)

            if col_ind is not None:

                col_ind = col_ind.reindex(ticker_ind)

                col_ind.index = tickers

            else:

                col_ind = pd.Series(np.nan, index=tickers)

            val = col_reg.fillna(col_ind)

            reg_fb = region_ind_table.get(fb)

            if reg_fb is not None:

                reg_fb = reg_fb.reindex(ri_index)

                reg_fb.index = tickers

            else:

                reg_fb = pd.Series(np.nan, index=tickers)

            mask = val.isna()

            val.loc[mask] = reg_fb.loc[mask]

            im_fb = industry_mc_table.get(fb)

            if im_fb is not None:

                im_fb = im_fb.reindex(im_index)

                im_fb.index = tickers

            else:

                im_fb = pd.Series(np.nan, index=tickers)

            mask = val.isna()

            val.loc[mask] = im_fb.loc[mask]

            ind_fb = industry_table.get(fb)

            if ind_fb is not None:

                ind_fb = ind_fb.reindex(ticker_ind)

                ind_fb.index = tickers

            else:

                ind_fb = pd.Series(np.nan, index=tickers)

            mask = val.isna()

            val.loc[mask] = ind_fb.loc[mask]

            region_ind_vals[base] = val

        results: dict[str, dict[str, dict[str, float]]] = {m: {} for m in core_metrics}

        for t in tickers:

            for m in core_metrics:

                results[m][t] = {
                    "Industry": float(industry_vals.at[t, m]) if t in industry_vals.index else np.nan,
                    "Sector-MC": float(sector_mc_vals.at[t, m]) if t in sector_mc_vals.index else np.nan,
                    "Region-Industry": float(region_ind_vals.at[t, m]) if t in region_ind_vals.index else np.nan,
                    "Region-Sector": float(region_sec_vals.at[t, m]) if t in region_sec_vals.index else np.nan,
                    "Industry-MC": float(industry_mc_vals.at[t, m]) if t in industry_mc_vals.index else np.nan,
                }

        return results


    def index_returns(
        self
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Build index-level return panels and a long-horizon annualised proxy.

        Computed outputs:
    
        - Weekly returns from `index_close.resample('W').last().pct_change()`
    
        - Quarterly returns from `index_close.resample('QE').last().pct_change()`
    
        - Annualised proxy per index:
          `(last_close / first_close) ** 0.2 - 1.0`

        Missing closes are forward/backward filled before resampling so sparse
        holes do not break period returns.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame, pd.DataFrame]
            `(annualised_rets, index_weekly_rets, index_quarterly_rets)`.

        Notes
        -----
        - Results are cached in `_index_returns_cache` after first computation.
       
        - The annualised exponent is fixed at `0.2` by implementation.

        """

        self._ensure_loaded()

        if self._index_returns_cache is not None:
            return self._index_returns_cache

        index_close = self.index_close.ffill().bfill()

        index_weekly_close = index_close.resample("W").last()
    
        index_weekly_rets = index_weekly_close.pct_change(fill_method=None).dropna()

        index_quarterly_close = index_close.resample("QE").last()
    
        index_quarterly_rets = index_quarterly_close.pct_change().dropna()

        annualised_rets = (index_close.iloc[-1] / index_close.iloc[0]) ** 0.2 - 1.0

        self._index_returns_cache = (annualised_rets, index_weekly_rets, index_quarterly_rets)
    
        return self._index_returns_cache
    
    
    def match_index_rets(
        self,
        exchange: str,
        index_rets: pd.Series,
        index_weekly_rets: pd.DataFrame,
        index_quarter_rets: pd.DataFrame,
        bl_market_returns: pd.Series = None,
        freq: str = "weekly"
    ) -> tuple[float, pd.Series]:
        """
        Route an exchange name to its canonical benchmark index and return both
        the corresponding scalar (e.g., expected return/Sharpe) and the time series
        of returns at the requested frequency.

        Routing examples
        ----------------
        Nasdaq* → ^NDX
        NYSE → ^GSPC
        LSE → ^FTSE
        XETRA → ^GDAXI
        MCE → ^IBEX
        Amsterdam → ^AEX
        Paris → ^FCHI
        Toronto → ^GSPTSE
        HKSE → ^HSI
        Swiss → ^SSMI
        default → ^GSPC

        Scalar selection
        ----------------
        If `bl_market_returns` is provided and contains the symbol, use its value;
        else use `index_rets[symbol]`; else default to 0.0.

        Parameters
        ----------
        exchange : str
        index_rets : pandas.Series
            Scalars by index symbol (e.g., expected Sharpe).
        index_weekly_rets, index_quarter_rets : pandas.DataFrame
            Return panels keyed by the same index symbols.
        bl_market_returns : pandas.Series, optional
            Overrides `index_rets` where present.
        freq : {'weekly','quarterly'}

        Returns
        -------
        (scalar, series) : tuple[float, pandas.Series]
            Selected scalar and the corresponding return series.
            
        Notes
        -----
        - Hard-coded routing for major exchanges to symbols: ^NDX, ^GSPC, ^FTSE, ^GDAXI, etc.
      
        - Defaults to S&P 500 (^GSPC) if exchange not recognised.
                
        """
        
        def pick_ret(
            idx_name: str
        ) -> float:
        
            if bl_market_returns is not None and idx_name in bl_market_returns.index:
        
                return float(bl_market_returns[idx_name])
        
            return float(index_rets.get(idx_name, 0.0))


        symbol = self.INDEX_ROUTING.get(exchange, "^GSPC")

        if freq == "quarterly":

            return pick_ret(symbol), index_quarter_rets[symbol]

        else:

            return pick_ret(symbol), index_weekly_rets[symbol]
        
        
    def load_index_pred(
        self
    ) -> pd.DataFrame:
        """
        Load stock market forecast table from `DATA_FILE` ('Stock_Market' sheet), numericise, and average duplicate rows.

        Processing
        ----------
        - Read sheet 'Stock_Market' with '0' treated as NA.
        
        - Strip thousands separators and coerce to numeric.
        
        - Average duplicate index rows using simple mean.
        
        Returns
        -------
        pd.DataFrame
            Table with numeric columns (commas removed), mean-aggregated over duplicate indices
                
        """

        data = pd.read_excel(
            self.path3,
            header = 0, 
            index_col = 0, 
            na_values = 0, 
            sheet_name = 'Stock_Market'
        )

        for col in data.columns:

            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '', regex = False), errors = 'coerce')

        return data.groupby(level = 0).mean()
    
    
    def crp(
        self
    ):
        """
        Load country risk premium (CRP) table from 'crp.xlsx'.

        IO
        --
        File: config.BASE_DIR / 'crp.xlsx'
        Columns used: ['Country','CRP']; 'Country' is set as the index.

        Returns
        -------
        pandas.DataFrame
            A two-column frame with index 'Country' and column 'CRP'.

        Raises
        ------
        FileNotFoundError or parser errors if the file/sheet is missing.
        
        """

        crp_file = config.BASE_DIR / 'crp.xlsx'
      
        crp = pd.read_excel(
            crp_file, 
            index_col = 0, 
            usecols = ['Country', 'CRP']
        )
      
        return crp
    
    
    @staticmethod
    def _factor_region_for_ticker(
        ticker: str
    ) -> str:
        """
        Determine which regional factor set (US/EU/EM) to use for a ticker when
        selecting factor ETFs and expected returns.

        Rules
        -----
        - '.TO' (Canada) → 'US' factor set.
       
        - Any suffix in `_EU_SUFFIXES` (including '.L') → 'EU'.
       
        - No dot in the symbol (typical US listing) → 'US'.
       
        - All other suffixes → 'EM'.

        Returns
        -------
        str
            One of {'US','EU','EM'}.
        """

        if ticker.endswith(".TO"):

            return "US"

        if any(ticker.endswith(suf) for suf in _EU_SUFFIXES):

            return "EU"

        if "." not in ticker:

            return "US"

        return "EM"


    @staticmethod
    def _factor_currency_for_region(
        region: str
    ) -> str:
        """
        Return the base currency in which factor ETF returns are quoted for a region.

        Mapping
        -------
        'EU' → 'EUR'
        'US', 'EM' → 'USD' (assumed)

        Returns
        -------
        str
            ISO currency code.
        """
        
        if region == "EU":
        
            return "EUR"

        return "USD"


    def _usd_per_one(
        self, 
        ccy: str, 
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.Series:
        """
        Build a daily level series of 'USD per 1 CCY' for the interval [start, end].

        Method
        ------
        1) If CCY == 'USD', return a constant 1.0 series.
       
        2) Try Yahoo ticker '{CCY}USD=X' (USD per CCY). If present, forward-fill on a
        business-day index.
        
        3) Else try 'USD{CCY}=X' (CCY per USD) and invert.

        Returns
        -------
        pandas.Series
            Series named 'USD_per_{CCY}' on a business-day index.

        Notes
        -----
        This series is a level (not a return). It is used to construct cross FX
        pairs via triangular arbitrage.
        """
    
        self._ensure_loaded()

        ccy = str(ccy).upper()
    
        start = pd.to_datetime(start)
    
        end = pd.to_datetime(end)
    
        key = (ccy, start, end)

        if key in self._usd_per_one_cache:
    
            return self._usd_per_one_cache[key]

        idx = pd.date_range(start=start, end=end, freq="B")

        if ccy == "USD":
    
            s = pd.Series(1.0, index=idx, name="USD_per_USD")
    
        else:
    
            fx_df = self.fx_usd_per_ccy
    
            col = f"USD_per_{ccy}"
    
            if col not in fx_df.columns:
    
                raise KeyError(f"'{col}' not found in 'FX USD per CCY'. Available: {list(fx_df.columns)[:8]}...")
            
            s = fx_df[col].reindex(idx).ffill().rename(col)

        self._usd_per_one_cache[key] = s
        
        return s

    def _pair_close_via_usd(
        self,
        pair: str,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.Series:
        """
        Recreate Yahoo '{PAIR}=X' close from USD crosses:
     
            close(pair) = (USD_per_LHS) / (USD_per_RHS)
     
        where pair = LHS+RHS (e.g., 'GBPUSD' → USD per GBP).
                
        """
        
        lhs = pair[:3]
        
        rhs = pair[3:]
       
        lhs_usd = self._usd_per_one(
            ccy= lhs, 
            start = start, 
            end = end
        )
       
        rhs_usd = self._usd_per_one(
            ccy = rhs, 
            start = start, 
            end = end
        )
       
        s = (lhs_usd / rhs_usd).rename(f"{pair}=X")
       
        return s


    def _fx_pair_level(
        self,
        from_ccy: str, 
        to_ccy: str, 
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.Series:
        """
        Construct the FX level series 'to_ccy per 1 from_ccy' using USD crosses:
           
            to_per_from = (USD_per_FROM) / (USD_per_TO).

        Special case
        ------------
        If from_ccy == to_ccy, return a flat 1.0 series.

        Returns
        -------
        pandas.Series
            Named '{to_ccy}_per_{from_ccy}', business-day index.
        """
        
        idx = pd.date_range(start = start, end = end, freq = "B")

        if from_ccy == to_ccy:
           
            return pd.Series(1.0, index = idx, name = f"{to_ccy}_per_{from_ccy}")

        usd_per_from = self._usd_per_one(
            ccy = from_ccy, 
            start = start, 
            end = end
        )

        usd_per_to = self._usd_per_one(
            ccy = to_ccy, 
            start = start, 
            end = end
        )

        pair = (usd_per_from / usd_per_to).rename(f"{to_ccy}_per_{from_ccy}")

        return pair.reindex(idx).ffill()


    def _fx_returns(
        self, 
        from_ccy: str,
        to_ccy: str,
        index_like: pd.Index
    ) -> pd.Series:
        """
        Compute daily arithmetic FX returns for the pair 'to_ccy per 1 from_ccy'
        aligned to a target index.

        Procedure
        ---------
        1) Build the level series L_t = {to_ccy per 1 from_ccy}.
        
        2) Compute arithmetic returns:
        
            r_t = L_t / L_{t-1} − 1.
        
        3) Reindex to `index_like` and fill NaNs with 0.0 (neutral assumption).

        Returns
        -------
        pandas.Series
            Named 'r_{to_ccy}_per_{from_ccy}', aligned to index_like.
      
        """
      
        if len(index_like) == 0:
   
            return pd.Series(dtype = float)

        key = (from_ccy, to_ccy, index_like[0], index_like[-1], len(index_like))
     
        if key in self._fx_ret_cache:
     
            s = self._fx_ret_cache[key]
     
            return s.reindex(index_like).fillna(0.0) 

        start = pd.Timestamp(index_like[0]) - pd.Timedelta(days = 7)

        end = pd.Timestamp(index_like[-1]) + pd.Timedelta(days = 7)

        level = self._fx_pair_level(from_ccy, to_ccy, start, end)

        r = level.pct_change()

        self._fx_ret_cache[key] = r

        return r.reindex(index_like).fillna(0.0)
        
    
    @staticmethod
    def _ticker_currency_from_suffix(
        ticker: str
    ) -> str:
        """
        Infer a ticker’s quote currency.

        Priority
        --------
        1) Explicit USD exceptions `_USD_EXCEPTIONS` (e.g., 'EMVL.L','IWQU.L','IWSZ.L') → 'USD'.
        
        2) Suffix mapping `_CCY_BY_SUFFIX` (e.g., '.L' → 'GBP', '.PA' → 'EUR', etc.).
        
        3) Default to 'USD' if no rule matches.

        Returns
        -------
        str
            ISO currency code.
        """
        
        if ticker in _USD_EXCEPTIONS:
        
            return "USD"
        
        for suf, ccy in _CCY_BY_SUFFIX.items():
        
            if ticker.endswith(suf):
        
                return ccy
        
        return "USD"
    

    @staticmethod
    def _convert_returns_currency(
        returns: pd.DataFrame, 
        from_ccy: str,
        to_ccy: str,
        r_fx: pd.Series
    ) -> pd.DataFrame:
        """
        Translate arithmetic returns from one currency into another by compounding
        with the FX return.

        Currency translation identity
        -----------------------------
        If R_local,t are returns in 'from_ccy' and r_fx,t are returns on the FX pair
        'to_ccy per 1 from_ccy', then returns in 'to_ccy' satisfy:
          
            R_to,t = (1 + R_local,t) * (1 + r_fx,t) − 1.

        Parameters
        ----------
        returns : pandas.DataFrame
            Columns are return series in 'from_ccy'.
        from_ccy, to_ccy : str
        r_fx : pandas.Series
            Daily FX returns aligned to `returns.index`.

        Returns
        -------
        pandas.DataFrame
            Returns expressed in 'to_ccy'. If currencies are identical the input is
            returned unchanged.
   
        """
        
        if from_ccy == to_ccy:
    
            return returns
    
        r_fx = r_fx.reindex(returns.index).fillna(0.0)

        return ((1.0 + returns).mul((1.0 + r_fx), axis = 0) - 1.0)
        
        
    def get_currency_annual_returns(
        self,
        country_to_pair: dict[str, str],
        base: str = "GBP"
    ) -> pd.DataFrame:
        """
        Compute weekly FX returns and total-period returns for FX pairs vs `base`.

        Inputs
        ------
        country_to_pair : dict[str, str]
            Mapping from country/label to six-character pair code (without `=X`),
            e.g. `{'United States': 'USDGBP'}`.
        base : str, default 'GBP'
            Target base currency used to reconstruct pair closes via
            `_pair_close_in_base`.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame]
            `ann_ret`:
                Total period return per pair, computed as
                `(1 + weekly_returns).prod() - 1`.
            `rets`:
                Weekly simple return panel (`W-FRI` resample, last close then
                `pct_change`).

        Notes
        -----
        - Invalid or short pair strings are ignored.
     
        - If no valid pairs are supplied, returns `(empty_series, empty_frame)`.
     
        """
        
        pairs = [str(p).upper() for p in country_to_pair.values() if isinstance(p, str) and len(str(p)) >= 6]
     
        if not pairs:
     
            return pd.Series(dtype = float), pd.DataFrame()

        cols = {}
      
        for p in pairs:
      
            s = self._pair_close_in_base(p, base = base, start = self.year_ago, end = self.today)
      
            w = s.resample('W-FRI').last()
      
            cols[p] = w

        close = pd.concat(cols, axis = 1)
       
        rets = close.pct_change().dropna(how = "all")
       
        ann_ret = (1.0 + rets).prod() - 1.0
       
        return ann_ret, rets


    def get_currency_annual_returns_both(
        self,
        country_to_ccy: dict[str, str],
    ) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
        """
        Compute `get_currency_annual_returns` outputs for both USD and GBP bases.

        Parameters
        ----------
        country_to_ccy : dict[str, str]
            Mapping from country/label to local currency code (e.g. `'NZD'`).

        Returns
        -------
        dict[str, tuple[pd.Series, pd.DataFrame]]
            Dictionary with keys `'USD'` and `'GBP'`. Each value is the tuple
            returned by `get_currency_annual_returns`:
            `(total_returns_by_pair, weekly_returns_panel)`.

        Examples
        --------
        If `country_to_ccy = {'NZ': 'NZD', 'EU': 'EUR'}`, this method builds
        pairs `{'NZDUSD', 'EURUSD'}` for the USD run and
        `{'NZDGBP', 'EURGBP'}` for the GBP run.
     
        """
       
        bases = ("USD", "GBP")
      
        out = {}
      
        for base in bases:
      
            pairs = {k: f"{v}{base}" for k, v in country_to_ccy.items()}
      
            out[base] = self.get_currency_annual_returns(country_to_pair=pairs, base=base)
      
        return out


    def get_fx_price_by_pair_local_to_base(
        self,
        country_to_pair: dict[str, str],
        *,
        base_ccy: str = "GBP",
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        interval: str = "1d",
        index: pd.Index | None = None,
    ) -> dict[str, pd.Series]:
        """
        Produce per-pair converters that map local-currency prices into a chosen
        base currency, suitable for multiplicative conversion at the price level.

        Converter semantics
        -------------------
        The output for pair P is a series FX_local_to_base such that:
           
            Price_base = Price_local * FX_local_to_base.

        Ambiguity resolution
        --------------------
        If pair is quoted BASE/LOCAL (e.g., 'GBPUSD' when base='GBP'):
       
            FX_local_to_base = 1 / Price(pair).
       
        If quoted LOCAL/BASE (e.g., 'USDGBP'):
       
            FX_local_to_base = Price(pair).
       
        If neither pattern matches, default to inverting (conservative assumption).

        Parameters
        ----------
        country_to_pair : dict[str, str]
            Pair map where each value is a six-character FX pair code.
        base_ccy : str, default 'GBP'
            Desired base currency for the output converters.
        start, end : date-like | None
            Default to (self.five_year_ago, self.today).
        interval : {'1d','1wk','1mo'}, default '1d'
            Output sampling frequency.
        index : pd.Index | None
            Optional explicit target index. If provided, `start`/`end` are taken
            from this index span and each converter is aligned to that range.

        Returns
        -------
        dict[str, pandas.Series]
            Mapping `pair -> FX_local_to_base` series where:
            `price_in_base = price_in_local * FX_local_to_base`.
     
        """

        if index is None:
            
            if start is None:
        
                start = self.five_year_ago
        
            if end is None:
        
                end = self.today
        
        else:
            
            start = index.min()
            
            end   = index.max()
       
        start = pd.to_datetime(start)
       
        end = pd.to_datetime(end)

        pairs = sorted(set(p for p in country_to_pair.values() if isinstance(p, str)))

        if not pairs:

            return {}

        out: dict[str, pd.Series] = {}

        for p in pairs:

            s = self._pair_close_via_usd(
                pair = p, 
                start = start, 
                end = end
            ).sort_index().dropna()
           
            if interval in ("1wk", "1w"):
           
                s = s.resample("W-FRI").last()
           
            elif interval in ("1mo", "1m"):
           
                s = s.resample("ME").last()
           
            else:

                s = s.asfreq("B")

            s = s.ffill().bfill()

            if p.startswith(base_ccy):

                fx_local_to_base = 1.0 / s

            elif p.endswith(base_ccy):

                fx_local_to_base = s

            else:

                fx_local_to_base = 1.0 / s

            fx_local_to_base.name = f"{p}_local_to_{base_ccy}"

            out[p] = fx_local_to_base

        return out


    def _region_factor_returns(
        self, 
        region: str
    ) -> pd.DataFrame:
        """
        Select factor ETF return columns for a region and rename columns to the
        canonical factor codes {MTUM, QUAL, SIZE, USMV, VLUE}.

        Selection logic
        ---------------
        - Use `_REGION_FACTOR_MAP[region]` to obtain {code → ETF ticker}.
        
        - Invert and keep only tickers present in `self.factor_rets`.
        
        - If none are present, fall back to `_REGION_FACTOR_MAP['US']`.
        
        - Return a DataFrame with columns ordered as `_FACTORS` and missing columns filled.

        Returns
        -------
        pandas.DataFrame
            Factor return panel for the chosen region, with canonical names.
     
        """
        
        if not isinstance(self.factor_rets, pd.DataFrame) or self.factor_rets.empty:
        
            return pd.DataFrame(index = self.daily_rets.index, columns = _FACTORS, dtype = float)

        ticker_map = _REGION_FACTOR_MAP.get(region, {})

        inv = {etf: code for code, etf in ticker_map.items()}

        present = [etf for etf in inv if etf in self.factor_rets.columns]

        if not present:

            ticker_map = _REGION_FACTOR_MAP["US"]

            inv = {etf: code for code, etf in ticker_map.items()}

            present = [etf for etf in inv if etf in self.factor_rets.columns]

        if not present:

            return pd.DataFrame(index = self.factor_rets.index, columns = _FACTORS, dtype = float)

        df = self.factor_rets[present]
        
        df = df.rename(columns = inv)  

        df = df.reindex(columns = _FACTORS)

        return df


    def _region_factor_exp(
        self,
        region: str
    ) -> dict:
        """
        Extract expected annual returns for canonical factors in a region from the
        'Factor ETFs' metadata table.

        Parsing rules
        -------------
        - If a 'Region' column exists, filter rows by case-insensitive match.
        
        - Prefer mapping from the 'Factor' label via `_FACTOR_LABEL_TO_CODE`.
        
        - Otherwise (or in addition), map from ETF tickers using `_REGION_FACTOR_MAP`.
        
        - If all expected values are NA/non-finite after parsing, fall back to 'US'.

        Returns
        -------
        dict[str, float]
            {'MTUM': μ_M, 'QUAL': μ_Q, 'SIZE': μ_S, 'USMV': μ_Vol, 'VLUE': μ_Val}
            with NaN where unavailable.

        Notes
        -----
        Values are assumed to be in the factor ETF’s base currency; FX adjustment
        is applied later where required.
    
        """
       
        out = {k: np.nan for k in _FACTORS}
       
        if not isinstance(self.factor_data, pd.DataFrame) or self.factor_data.empty:
       
            return out

        df = self.factor_data

        if "Region" in df.columns:

            df_r = df[df["Region"].astype(str).str.upper() == region.upper()]

        else:

            df_r = df

        if "Factor" in df_r.columns:

            for _, row in df_r.iterrows():

                label = str(row["Factor"]).strip()

                code = _FACTOR_LABEL_TO_CODE.get(label)

                if code in out and "Exp Returns" in df_r.columns and pd.notna(row.get("Exp Returns")):

                    out[code] = float(row["Exp Returns"])
        else:

            ticker_map = _REGION_FACTOR_MAP.get(region, {})

            for code, etf in ticker_map.items():

                if etf in df.index and "Exp Returns" in df.columns and pd.notna(df.at[etf, "Exp Returns"]):

                    out[code] = float(df.at[etf, "Exp Returns"])

        if all(not np.isfinite(v) for v in out.values()):
            
            return self._region_factor_exp(
                region = "US"
            )

        return out
    
    
    def align_sector_sharpe(
        self
    ) -> pd.Series:
        """
        Map each ticker to the expected Sharpe ratio of its sector from the
        'Sector Data' sheet.

        Output
        ------
        pandas.Series indexed by ticker. Missing sectors default to 0.0.

        Definition
        ----------
        The function does not compute Sharpe; it relays the 'Exp Sharpe Ratio'
        column for the matching sector.
                
        """
                
        sec_sharpe = self.sector_data["Exp Sharpe Ratio"]
        
        out = self.analyst["Sector"].map(sec_sharpe).reindex(self.tickers).fillna(0.0)
        
        return out.sort_index()
    
    
    def align_ind_sharpe(
        self
    ) -> pd.Series:
        """
        Map each ticker to the expected Sharpe ratio of its industry from the
        'Industry Data' sheet. Missing industries default to 0.0.

        Output
        ------
        pandas.Series indexed by ticker with the referenced 'Exp Sharpe Ratio'.
        
        """
                
        ind_sharpe = self.industry_data["Exp Sharpe Ratio"]
       
        out = self.analyst["Industry"].map(ind_sharpe).reindex(self.tickers).fillna(0.0)
       
        return out.sort_index()
    
    
    def index_sharpe(
        self
    ) -> pd.Series:
        """
        Compute an exchange-aligned expected Sharpe proxy per ticker using index
        weekly returns over the last year and an exponential estimator.

        Estimation
        ----------
        Let R_t be weekly index returns for each index over the last year,
        N = number of weekly observations.

        Exponentially weighted estimates (halflife = 0.1*N):
            
            μ_hat = E_ewm[R_t] at the last date,
        
            σ_hat = Std_ewm[R_t] at the last date.
        
        Annualisation:
        
            μ_ann = μ_hat * N,
        
            σ_ann = σ_hat * sqrt(N).

        Expected Sharpe proxy per index:
        
            SR = (μ_ann − rf_ann) / σ_ann,
        
        with rf_ann = 0.0435 (4.35% annual).

        Ticker mapping
        --------------
        For each ticker, map 'fullExchangeName' to a canonical index symbol via
        `match_index_rets` (weekly frequency) and take that index’s SR.

        Returns
        -------
        pandas.Series
            Expected Sharpe proxy per ticker.
     
        """

        index_weekly = self.index_returns()[1].loc[self.year_ago:]
     
        N = len(index_weekly)
     
        if N == 0:
     
            return pd.Series(0.0, index=self.tickers)

        mu = index_weekly.ewm(halflife=0.1 * N, adjust = False).mean().iloc[-1] * N
     
        sigma = index_weekly.ewm(halflife=0.1 * N, adjust = False).std().iloc[-1] * np.sqrt(N)

        sr_by_index = (mu - config.RF) / sigma

        idx_series = self.ticker_to_index_series()

        out = idx_series.map(sr_by_index).reindex(self.tickers).fillna(0.0)

        return out.sort_index()

        
        
    def _group_momentum_z(
        self,
        price_df: pd.DataFrame,
        period: int = 52
    ) -> pd.Series:
        """
        Compute cross-sectional z-scores of momentum across columns.

        Momentum definition
        -------------------
        For each column j:
        
            mom_j = P_j(T) / P_j(T−period) − 1.
        
        Standardise cross-sectionally at the last date:
            
            z_j = (mom_j − mean_j(mom)) / std_j(mom).

        Parameters
        ----------
        price_df : pandas.DataFrame
            Price levels per group (columns).
        period : int, default 52
            Lookback horizon (weeks).

        Returns
        -------
        pandas.Series
            z-scores by column at the last available date.
     
        """

        mom = price_df.pct_change(periods = period).iloc[-1]   

        return (mom - mom.mean()) / mom.std()


    def align_sector_momentum_z(
        self,
        price_df: pd.DataFrame,
        tickers: Sequence[str],
        ticker_to_sector: pd.Series,
        period: int = 52
    ) -> pd.Series:
        """
        Map each ticker to the z-scored momentum of its sector.

        Parameters
        ----------
        price_df : pd.DataFrame
            Sector price levels indexed by date.
        tickers : Sequence[str]
        ticker_to_sector : pd.Series
            Map ticker → sector string present in `price_df.columns`.
        period : int, default 52
            Lookback in weeks.

        Returns
        -------
        pd.Series
            Ticker-aligned sector momentum z-scores. Missing sectors get the mean z.
        
        """


        z = self._group_momentum_z(
            price_df = price_df, 
            period = period
        )    
        
        fallback = z.mean()
        
        idx = pd.Index(tickers)
        
        out = {  
            t: z.get(ticker_to_sector.get(t, None), fallback)
            for t in idx
        }
        
        return pd.Series(out).sort_index()


    def align_industry_momentum_z(
        self,
        price_df: pd.DataFrame,
        tickers: pd.Index,
        ticker_to_industry: pd.Series,
        period: int = 52
    ) -> pd.Series:
        """
        Map each ticker to the z-scored momentum of its industry.

        Parameters
        ----------
        price_df : pd.DataFrame
            Industry price levels indexed by date.
        tickers : pd.Index
        ticker_to_industry : pd.Series
        period : int, default 52

        Returns
        -------
        pd.Series
            Ticker-aligned industry momentum z-scores. Missing industries get the mean z.
        
        """

        
        z = self._group_momentum_z(
            price_df = price_df, 
            period = period
        )
        
        fallback = z.mean()
        
        idx = pd.Index(tickers)
        
        out = {  
            t: z.get(ticker_to_industry.get(t, None), fallback)
            for t in idx
        }
        
        return pd.Series(out).sort_index()
        

    def align_index_momentum_z(
        self,
        index_price_df: pd.DataFrame,
        tickers: pd.Index,
        ticker_to_index: pd.Series,
        period: int = 52
    ) -> pd.Series:
        """
        Map each ticker to the z-scored momentum of its benchmark index.

        Parameters
        ----------
        index_price_df : pd.DataFrame
            Index price levels (weekly close), columns=index symbols.
        tickers : pd.Index
        ticker_to_index : pd.Series
            Ticker → index symbol present in `index_price_df.columns`.
        period : int, default 52

        Returns
        -------
        pd.Series
            Ticker-aligned index momentum z-scores. Missing indexes get the mean z.
      
        """

        
        z = self._group_momentum_z(
            price_df = index_price_df, 
            period = period
        )
        
        fallback = z.mean()
        
        idx = pd.Index(tickers)
        
        out = {  
            t: z.get(ticker_to_index.get(t, None), fallback)
            for t in idx
        }
        
        return pd.Series(out).sort_index()
    
    
    def pick_index(
        self,
        exchange_name: str
    ) -> str:
        """
        Return the primary benchmark index symbol for a given exchange.

        Parameters
        ----------
        exchange_name : str

        Returns
        -------
        str
            Index symbol (e.g., '^GSPC'). Falls back to '^GSPC' if not in `ExchangeMapping`.

        Notes
        -----
        - Uses `ExchangeMapping.get(exchange_name.upper(), '^GSPC')`.
        
        """
        
        try:
            ex = ExchangeMapping.get(exchange_name.upper(), '^GSPC')
            
        except Exception:
            
            ex = '^GSPC'
        
        return ex
    
    
    def ticker_to_index_series(
        self
    ) -> pd.Series:
        """
        Build a per-ticker Series of benchmark index symbols using 'fullExchangeName'.

        Returns
        -------
        pd.Series
            Index=ticker, value=index symbol (e.g., '^GSPC', '^NDX', '^FTSE', ...).
       
        """

        ticker_to_index_series = self.analyst['fullExchangeName'].map(self.pick_index)
        
        return ticker_to_index_series


    def exp_factor_data(
        self,
        tickers = None
    ):
        """
        Assemble a per-ticker DAILY design frame in the ticker’s local currency,
        combining factor returns, group returns, index excess returns, and the
        ticker’s own excess returns.

        Components (per ticker)
        -----------------------
        • 'MTUM','QUAL','SIZE','USMV','VLUE' :
        
            Regional factor ETF returns chosen via `_region_factor_returns(region)`
            and then translated from the factor base currency (USD or EUR) into the
            ticker’s local currency using daily FX returns r_fx.
        
            Currency translation equation:
        
                r_factor_local,t = (1 + r_factor_base,t) * (1 + r_fx,t) − 1,
                
            where r_fx,t are returns on the FX pair 'local per 1 base'.
        
        • 'Industry Return' : daily pct-change for the ticker’s industry.
        
        • 'Sector Return'   : daily pct-change for the ticker’s sector.
        
        • 'Index Excess Return' : market index daily return minus RF_PER_DAY.
        
        • 'Ticker Excess Return' : ticker’s daily return minus RF_PER_DAY.

        Index mapping
        -------------
        The index symbol for each ticker is given by `ticker_to_index_series()`.

        Missing data handling
        ---------------------
        - If the regional factor columns are unavailable, an empty frame with
        canonical columns is used.
       
        - Group/index series missing for a key are replaced by NaN series aligned
        to the factor index.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Per-ticker design frames indexed by date.
       
        """
       
        self._ensure_loaded()

        if tickers is None:
    
            tickers = list(config.tickers)

        tickers = pd.Index(tickers)

        ind_ret = self.industry_close.pct_change()
    
        sec_ret = self.sector_close.pct_change()
    
        idx_daily = self.index_close.pct_change().dropna() - config.RF_PER_DAY
    
        daily_rets = self.daily_rets - config.RF_PER_DAY

        FACTORS = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]

        region = tickers.to_series().map(self._factor_region_for_ticker)
    
        local_ccy = tickers.to_series().map(self._ticker_currency_from_suffix)
    
        meta = pd.DataFrame({"region": region, "local_ccy": local_ccy}, index=tickers)

        idx_map = self.ticker_to_index_series().reindex(tickers)

        fac_region_cache: dict[str, pd.DataFrame] = {}
    
        for r in meta["region"].unique():
    
            fac_region_cache[r] = self._region_factor_returns(r)

        fx_fac_cache: dict[tuple[str, str], pd.Series] = {}
    
        fac_local_cache: dict[tuple[str, str], pd.DataFrame] = {}

        for (r, ccy), _ in meta.groupby(["region", "local_ccy"]).groups.items():
    
            fac_region = fac_region_cache[r]
    
            if fac_region.empty:
    
                fac_local = pd.DataFrame(index = self.factor_rets.index, columns = FACTORS, dtype = float)
    
            else:
    
                factor_ccy = self._factor_currency_for_region(r)
    
                r_fx = self._fx_returns(
                    from_ccy = factor_ccy,
                    to_ccy = ccy,
                    index_like = fac_region.index,
                )
               
                fx_fac_cache[(r, ccy)] = r_fx
               
                fac_local = self._convert_returns_currency(
                    returns = fac_region,
                    from_ccy = factor_ccy,
                    to_ccy = ccy,
                    r_fx = r_fx,
                )
           
            fac_local_cache[(r, ccy)] = fac_local

        ticker_factor: dict[str, pd.DataFrame] = {}

        for t in tickers:
       
            r = meta.at[t, "region"]
       
            ccy = meta.at[t, "local_ccy"]

            fac_local = fac_local_cache[(r, ccy)]

            t_ind = self.analyst.at[t, "Industry"]
       
            t_sec = self.analyst.at[t, "Sector"]

            ind_series = ind_ret.get(t_ind, pd.Series(np.nan, index = fac_local.index))
       
            sec_series = sec_ret.get(t_sec, pd.Series(np.nan, index = fac_local.index))

            mkt_idx = idx_map.at[t]
          
            if pd.isna(mkt_idx) or mkt_idx not in idx_daily.columns:
          
                mkt_series = pd.Series(np.nan, index = fac_local.index, name = "Index Excess Return")
         
            else:
         
                mkt_series = idx_daily[mkt_idx].reindex(fac_local.index).rename("Index Excess Return")

            daily_rets_t = daily_rets.get(t, pd.Series(np.nan, index=fac_local.index)).rename(
                "Ticker Excess Return"
            )

            df = pd.concat(
                [
                    fac_local,
                    ind_series.rename("Industry Return"),
                    sec_series.rename("Sector Return"),
                    mkt_series,
                    daily_rets_t,
                ],
                axis = 1,
            )

            ticker_factor[t] = df

        return ticker_factor


    def exp_factors(
        self, 
        tickers = None
    ) -> pd.DataFrame:
        """
        Construct expected ANNUAL returns for each ticker across:
        
            {'Industry','Sector','Index','Momentum','Quality','Size','Volatility','Value'}.

        Group expectations
        ------------------
        - Industry/Sector/Index: read directly from 'Exp Returns' columns in their
        respective aggregate tables; values are assumed to be already in relevant
        base units and are not FX-translated here.

        Factor expectations with FX drift
        ---------------------------------
        - Regional factor expected returns μ_fac (per canonical factor) are read via
        `_region_factor_exp(region)` in the factor ETF base currency c_base ∈ {USD, EUR}.
        
        - If the ticker local currency c_local differs from c_base, an FX drift
        adjustment is applied using daily FX returns r_fx over the factor sample window:
        
            DRIFT_FX_ann = ( Π_t (1 + r_fx,t) )^(252 / N) − 1,
        
            where N = count of non-missing r_fx over the sample.
        
        The local-currency expectation becomes:
        
            μ_local = (1 + μ_base) * (1 + DRIFT_FX_ann) − 1.

        Returns
        -------
        pandas.DataFrame
            Index = ticker; columns:
            ['Industry','Sector','Index','Momentum','Quality','Size','Volatility','Value'].

        Notes
        -----
        RF is not added here; outputs are expectations (means), not premia.
    
        """
       
        self._ensure_loaded()

        if tickers is None:
     
            tickers = list(config.tickers)
     
        tickers = pd.Index(tickers)

        idx_exp = self.index_data.get("Exp Returns", pd.Series(dtype = float))
     
        ind_exp = self.industry_data.get("Exp Returns", pd.Series(dtype = float))
     
        sec_exp = self.sector_data.get("Exp Returns", pd.Series(dtype = float))

        idx_map = self.ticker_to_index_series().reindex(tickers)

        if isinstance(self.factor_rets, pd.DataFrame) and not self.factor_rets.empty:

            factor_index = self.factor_rets.index

        else:

            factor_index = getattr(self.daily_rets, "index", pd.DatetimeIndex([]))

        region = tickers.to_series().map(self._factor_region_for_ticker)

        local_ccy = tickers.to_series().map(self._ticker_currency_from_suffix)

        meta = pd.DataFrame({"region": region, "local_ccy": local_ccy}, index=tickers)

        mu_fac_by_region = {r: self._region_factor_exp(r) for r in meta["region"].unique()}

        fx_ann_by_pair: dict[tuple[str, str], float] = {}

        for (r, ccy), _ in meta.groupby(["region", "local_ccy"]).groups.items():

            factor_ccy = self._factor_currency_for_region(
                region = r
            )

            if factor_ccy == ccy or len(factor_index) <= 5:

                fx_ann_by_pair[(r, ccy)] = 0.0

                continue

            r_fx = self._fx_returns(
                from_ccy = factor_ccy, 
                to_ccy = ccy,
                index_like = factor_index
            )
            
            n = r_fx.notna().sum()
            
            if n > 0:
            
                fx_ann = (1.0 + r_fx.fillna(0.0)).prod() ** (252.0 / n) - 1.0
            
            else:
            
                fx_ann = 0.0
            
            fx_ann_by_pair[(r, ccy)] = fx_ann

        rows = []
        
        for t in tickers:
        
            r = meta.at[t, "region"]
        
            ccy = meta.at[t, "local_ccy"]
        
            fx_ann = fx_ann_by_pair[(r, ccy)]
        
            mu_fac = mu_fac_by_region[r]

            t_ind = self.analyst.at[t, "Industry"]
        
            t_sec = self.analyst.at[t, "Sector"]
        
            t_idx = idx_map.at[t]

            mu_ind = np.max(float(ind_exp.get(t_ind, np.nan)), -1) if len(ind_exp) else np.nan
        
            mu_sec = np.max(float(sec_exp.get(t_sec, np.nan)), -1) if len(sec_exp) else np.nan
        
            mu_idx = (
                np.max(float(idx_exp.get(t_idx, np.nan)), -1)
                if pd.notna(t_idx) and len(idx_exp)
                else np.nan
            )

        
            def _to_local(
                x: float
            ) -> float:
            
                return (1.0 + x) * (1.0 + fx_ann) - 1.0 if np.isfinite(x) else np.nan

            
            rows.append(
                {
                    "Ticker": t,
                    "Industry": mu_ind,
                    "Sector": mu_sec,
                    "Index": mu_idx,
                    "Momentum": _to_local(mu_fac.get("MTUM", np.nan)),
                    "Quality": _to_local(mu_fac.get("QUAL", np.nan)),
                    "Size": _to_local(mu_fac.get("SIZE", np.nan)),
                    "Volatility": _to_local(mu_fac.get("USMV", np.nan)),
                    "Value": _to_local(mu_fac.get("VLUE", np.nan)),
                }
            )

        return pd.DataFrame(rows).set_index("Ticker")


    def factor_weekly_rets(
        self,
        resample_str = "W-FRI",
        after_date = None,
    ) -> pd.DataFrame:
        """
        Compute weekly factor ETF returns by log-summing daily returns within each
        resampled week (additivity of log returns), then exponentiating implicitly
        when used downstream.

        Construction
        ------------
        Let r_t be daily arithmetic returns. The weekly log-return is:
        
            lr_week = Σ_{t∈week} log(1 + r_t).
       
        The arithmetic weekly return is:
            
            R_week = exp(lr_week) − 1.
        
        This function returns lr_week (log-summed values) in column space, which is
        additive and convenient; downstream consumers may exponentiate as needed.

        Parameters
        ----------
        resample_str : str, default "W-FRI"
        after_date : str | pandas.Timestamp | None
            Optionally trim the sample to dates ≥ after_date.

        Returns
        -------
        pandas.DataFrame
            Weekly log-summed returns for the intersection of {MTUM, QUAL, SIZE, USMV, VLUE}.
      
        """
        
        self._ensure_loaded()

        key = (resample_str, pd.to_datetime(after_date) if after_date is not None else None)
   
        if key in self._factor_weekly_cache:
   
            return self._factor_weekly_cache[key]

        FACTORS = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]

        factor_daily = self.factor_rets
   
        if after_date is not None:
   
            factor_daily = factor_daily.loc[after_date:]

        fac_w = np.log1p(factor_daily).resample(resample_str).sum()

        fac_w = fac_w.rename(columns=lambda c: str(c))

        fac_w = fac_w[[c for c in FACTORS if c in fac_w.columns]].dropna(how="all")

        self._factor_weekly_cache[key] = fac_w

        return fac_w

    
    def factor_index_ind_sec_weekly_rets(
        self,
        merge: bool = True
    ) -> pd.DataFrame:
        """
        Return weekly return panels for factor ETFs, major indexes, industries,
        and sectors on a common weekly calendar.

        Components
        ----------
        - Index: from `index_returns()[1]`.
       
        - Factors: `factor_weekly_rets(resample_str="W-SUN", after_date=self.five_year_ago)`.
        (weekly log sums; treat as log-returns for additive analyses)
       
        - Industry/Sector: resample closes with 'W-SUN' → last → pct_change.

        Parameters
        ----------
        merge : bool, default True
       
            If True, concatenate into a single DataFrame by columns; otherwise
            return the four components as a tuple.

        Returns
        -------
        pandas.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Combined panel or a 4-tuple: (factor_weekly, index_weekly, industry_weekly, sector_weekly).

        Notes
        -----
        - All components are aligned by outer join in the concatenation step and
        rows that are all-NA are dropped.
      
        """
                
        self._ensure_loaded()

        index_weekly = self.index_returns()[1]

        factor_weekly = self.factor_weekly_rets(
            resample_str="W-SUN",
            after_date=self.five_year_ago,
        )

        ind_weekly_rets = self.industry_close.resample("W-SUN").last().pct_change().dropna()
      
        sec_weekly_rets = self.sector_close.resample("W-SUN").last().pct_change().dropna()

        if merge:
            df = pd.concat(
                [index_weekly, factor_weekly, ind_weekly_rets, sec_weekly_rets],
                axis = 1,
            ).dropna(how = "all")
            return df

        return factor_weekly, index_weekly, ind_weekly_rets, sec_weekly_rets
          
             
    def _ticker_ccy(
        self, 
        tickers: pd.Index
    ) -> pd.Series:
        """
        Resolve ticker currency codes for a provided ticker index.

        Resolution order
        ----------------
        1. Use `self.currency` if available.
    
        2. Fall back to `self.currencies` or `self.ticker_currency` if exposed.
    
        3. If source is a DataFrame, use column `Currency` when present, else the
           first column.
    
        4. Fill unresolved tickers using `_ticker_currency_from_suffix`.

        Parameters
        ----------
        tickers : pd.Index
            Tickers requiring currency resolution.

        Returns
        -------
        pd.Series
            Currency code per ticker, indexed by `tickers`.

        Raises
        ------
        AttributeError
            If no currency map attribute is available on the object.
      
        """

        try:

            s = self.currency

        except AttributeError:

            s = None

            for attr_name in ("currencies", "ticker_currency"):

                if hasattr(self, attr_name):

                    s = getattr(self, attr_name)

                    break

            if s is None:

                raise AttributeError(
                    "RatioData does not expose a currency map. "
                    "Ensure the 'Currency' sheet or equivalent is present "
                    "and mapped to one of: 'currency', 'currencies', "
                    "'ticker_currency'."
                )

        if isinstance(s, pd.DataFrame):

            col = "Currency" if "Currency" in s.columns else s.columns[0]

            s = s[col]

        s = s.astype(str)

        out = s.reindex(tickers)

        missing = out.isna()

        if missing.any():

            inferred = pd.Series(
                [self._ticker_currency_from_suffix(t) for t in out.index[missing]],
                index = out.index[missing],
                dtype = "object",
            )
          
            out.loc[missing] = inferred

        return out

    
    def _base_per_one(
        self,
        ccy: str, 
        base: str, 
        start, 
        end
    ) -> pd.Series:
        """
        Build the level series `BASE_per_<CCY>` over a business-day calendar.

        Definition:
        `BASE_per_ccy = USD_per_ccy / USD_per_base`.

        Parameters
        ----------
        ccy : str
            Source currency code.
        base : str
            Target base currency code.
        start, end : datetime-like
            Inclusive range used to build the business-day index.

        Returns
        -------
        pd.Series
            FX level series named `"{base}_per_{ccy}"`.

        Notes
        -----
        - Returns a constant 1.0 series when `ccy == base`.
     
        """
      
        ccy = str(ccy).upper()
      
        base = str(base).upper()
      
        idx = pd.date_range(start = start, end = end, freq = "B")

        if ccy == base:
           
            return pd.Series(1.0, index=idx, name=f"{base}_per_{base}")

        usd_per_ccy = self._usd_per_one(
            ccy = ccy, 
            start = start,
            end = end
        )     
        
        usd_per_base = self._usd_per_one(
            ccy = base,
            start = start,
            end = end
        )  

        s = (usd_per_ccy / usd_per_base).reindex(idx).ffill()
     
        s.name = f"{base}_per_{ccy}"
     
        return s


    def _pair_close_in_base(
        self, 
        pair: str,
        base: str,
        start, 
        end
    ) -> pd.Series:
        """
        Reconstruct `{pair}=X` closes from base-currency crosses.

        For `pair = LHS + RHS` (for example `EURUSD`), this computes:
        `close(pair) = BASE_per_LHS / BASE_per_RHS`.

        Parameters
        ----------
        pair : str
            Six-character FX pair code.
        base : str
            Base currency used to synthesize both legs.
        start, end : datetime-like
            Inclusive range for the output level series.

        Returns
        -------
        pd.Series
            Reconstructed pair close series named `"{pair}=X"`.
      
        """
    
        pair = str(pair).upper()
    
        lhs, rhs = pair[: 3], pair[3: 6]
        
        lhs_b = self._base_per_one(
            ccy = lhs,
            base = base, 
            start = start, 
            end = end
        )
        
        rhs_b = self._base_per_one(
            ccy = rhs, 
            base = base,
            start = start,
            end = end
        )
        
        s = (lhs_b / rhs_b).rename(f"{pair}=X")
        
        return s


    def _fx_converters_by_ccy(
        self,
        *,
        ccys: pd.Series,
        base_ccy: str,
        interval: str,
        index: pd.DatetimeIndex,
    ) -> dict[str, pd.Series]:
        """
        Build local→base FX price series for each distinct currency in `ccys`,
        aligned to `index`.

        Parameters
        ----------
        ccys : pd.Series
            Per-ticker currency codes.
        base_ccy : str
            Base currency to convert into.
        interval : str
            Interval passed through to `get_fx_price_by_pair_local_to_base`.
        index : pd.DatetimeIndex
            Target index for all converter series.

        Returns
        -------
        dict[str, pd.Series]
            Mapping `{ccy: local_to_base_price_series}`.

        Raises
        ------
        ValueError
            If the FX provider returns `None` or an empty series for a required
            pair.
        KeyError
            If a required pair is missing from the provider output.
        
        """

        index = pd.DatetimeIndex(index)

        ccys  = ccys.astype(str)

        unique_ccys = pd.Index(sorted(set(ccys.dropna().tolist())))

        out: dict[str, pd.Series] = {}

        if len(unique_ccys) == 0:

            return out

        pair_map = {ccy: f"{ccy}{base_ccy}" for ccy in unique_ccys if ccy != base_ccy}

        fx_map: dict[str, pd.Series] = {}

        if pair_map:

            fx_map = self.get_fx_price_by_pair_local_to_base(
                country_to_pair = pair_map,
                base_ccy = base_ccy,
                interval = interval,
                index = index,
            )
            
            if fx_map is None:

                raise ValueError("get_fx_price_by_pair_local_to_base returned None – it must return a dict mapping pair codes to Series.")

        for ccy in unique_ccys:

            if ccy == base_ccy:

                out[ccy] = pd.Series(
                    1.0,
                    index = index,
                    name = f"{base_ccy}_local_to_{base_ccy}",
                )
                
                continue

            pair = pair_map[ccy]

            if pair not in fx_map:
                
                raise KeyError(f"FX pair {pair} missing from fx_map. country_to_pair passed to get_fx_price_by_pair_local_to_base was: {pair_map}")

            s = fx_map[pair]

            if s is None or len(s) == 0:
             
                raise ValueError(f"FX series for pair {pair} is empty or None. Check FX USD per CCY data (e.g. 'USD_per_{ccy}' and 'USD_per_{base_ccy}').")

            union_idx = index.union(s.index)

            s_aligned = (
                s.reindex(union_idx)
                .sort_index()
                .ffill()
                .bfill()
                .reindex(index)
            )

            out[ccy] = s_aligned.rename(f"{pair}_local_to_{base_ccy}")

        return out

    
    def convert_returns_to_base(
        self,
        ret_df: pd.DataFrame,
        *,
        base_ccy: str,
        interval: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Convert a DataFrame of local‐currency returns into base‐currency returns.

        Parameters
        ----------
        ret_df : pd.DataFrame
            Wide DataFrame of simple returns in local currency. Columns are tickers,
            index is a DatetimeIndex at the desired frequency.
        base_ccy : str
            Target base currency (e.g. 'USD', 'GBP').
        interval : str
            Sampling interval (e.g. '1d', '1wk', '1mo'). Passed through to the FX
            converters so they can pick the correct frequency.
        verbose : bool, default False
            If True, print some debug information about the FX series used.

        Returns
        -------
        pd.DataFrame
            Returns in `base_ccy`, same shape as `ret_df`, with dtype float64.
        
        """
      
        if ret_df is None:
      
            raise ValueError("ret_df must not be None.")

        if not isinstance(ret_df, pd.DataFrame):
      
            ret_df = pd.DataFrame(ret_df)

        if ret_df.empty:

            return ret_df.astype("float64")

        if not isinstance(ret_df.index, pd.DatetimeIndex):

            raise TypeError(
                "ret_df.index must be a pandas.DatetimeIndex for FX alignment."
            )

        try:

            col_ccy = self._ticker_ccy(
                tickers = ret_df.columns
            )
       
        except AttributeError:

            mapping = OrderedDict()
        
            for col in ret_df.columns:
       
                mapping[col] = self._ticker_currency_from_suffix(
                    ticker = col
                )
       
            col_ccy = pd.Series(mapping, index = ret_df.columns, dtype = "object")

        col_ccy = col_ccy.reindex(ret_df.columns)

        known_mask = col_ccy.notna()

        if not known_mask.all():

            unknown = col_ccy.index[~known_mask].tolist()

            if verbose and unknown:

                print("convert_returns_to_base: dropping columns with unknown currency:", unknown)
                
            ret_df = ret_df.loc[:, known_mask]
            
            col_ccy = col_ccy[known_mask]

        if col_ccy.empty:

            return ret_df.astype("float64")

        fx_by_ccy = self._fx_converters_by_ccy(
            ccys = col_ccy,
            base_ccy = base_ccy,
            interval = interval,
            index = ret_df.index,
        )

        out = ret_df.astype("float64").copy()

        groups: dict[str, list[str]] = {}

        for col, ccy in col_ccy.items():

            if pd.isna(ccy):

                continue

            groups.setdefault(str(ccy), []).append(col)

        for ccy, cols in groups.items():

            if ccy == base_ccy:

                continue

            fx_price = fx_by_ccy.get(ccy)
          
            if fx_price is None:
           
                if verbose:
                 
                    print(f"convert_returns_to_base: no FX converter for {ccy}→{base_ccy}; leaving columns {cols} unchanged.")
             
                continue

            fx_price = fx_price.reindex(out.index).ffill().bfill()

            fx_ret = fx_price.pct_change().fillna(0.0)

            out.loc[:, cols] = (1.0 + out.loc[:, cols]).mul(1.0 + fx_ret, axis=0) - 1.0

            if verbose:
               
                print(f"\nDEBUG FX price for {ccy}→{base_ccy}:")
               
                print(fx_price.head())
               
                print(fx_price.tail())
               
                print("nunique:", fx_price.nunique())
               
                print("fx_ret nonzero count:", (fx_ret != 0).sum())
               
                print("fx_ret describe:")
               
                print(fx_ret.describe())

        return out
    

    def convert_prices_to_base(
        self,
        price_df: pd.DataFrame,
        *,
        base_ccy: str,
        interval: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Convert a DataFrame of local-currency price levels into base-currency prices.

        Parameters
        ----------
        price_df : pd.DataFrame
            Wide DataFrame of price levels in local currency. Columns are tickers,
            index is a DatetimeIndex at the desired frequency.
        base_ccy : str
            Target base currency (e.g. 'USD', 'GBP').
        interval : str
            Sampling interval (e.g. '1d', '1wk', '1mo'). Passed through to the FX
            converters so they can pick the correct frequency.
        verbose : bool, default False
            If True, print debug information about FX series used.

        Returns
        -------
        pd.DataFrame
            Prices in `base_ccy`, same shape as `price_df`, with dtype float64.
        
        """
       
        if price_df is None:
       
            raise ValueError("price_df must not be None.")

        if not isinstance(price_df, pd.DataFrame):
       
            price_df = pd.DataFrame(price_df)

        if price_df.empty:
       
            return price_df.astype("float64")

        if not isinstance(price_df.index, pd.DatetimeIndex):
            
            raise TypeError("price_df.index must be a pandas.DatetimeIndex for FX alignment.")

        try:

            col_ccy = self._ticker_ccy(
                tickers = price_df.columns
            )

        except AttributeError:

            mapping = OrderedDict()
       
            for col in price_df.columns:
       
                mapping[col] = self._ticker_currency_from_suffix(
                    ticker = col
                )
       
            col_ccy = pd.Series(mapping, index = price_df.columns, dtype = "object")

        col_ccy = col_ccy.reindex(price_df.columns)

        known_mask = col_ccy.notna()

        if not known_mask.all():

            unknown = col_ccy.index[~known_mask].tolist()

            if verbose and unknown:

                print("convert_prices_to_base: dropping columns with unknown currency:",  unknown)
                
            price_df = price_df.loc[:, known_mask]
          
            col_ccy = col_ccy[known_mask]

        if col_ccy.empty:
          
            return price_df.astype("float64")

        fx_by_ccy = self._fx_converters_by_ccy(
            ccys = col_ccy,
            base_ccy = base_ccy,
            interval = interval,
            index = price_df.index,
        )

        out = price_df.astype("float64").copy()

        groups: dict[str, list[str]] = {}

        for col, ccy in col_ccy.items():

            if pd.isna(ccy):

                continue

            groups.setdefault(str(ccy), []).append(col)

        for ccy, cols in groups.items():

            if ccy == base_ccy:

                continue

            fx_price = fx_by_ccy.get(ccy)

            if fx_price is None:

                if verbose:

                    print(f"convert_prices_to_base: no FX converter for {ccy}→{base_ccy}; leaving columns {cols} unchanged.")
                    
                continue

            fx_price = fx_price.reindex(out.index).ffill().bfill()

            out.loc[:, cols] = out.loc[:, cols].mul(fx_price, axis=0)

            if verbose:
          
                print(f"\nDEBUG FX price for {ccy}→{base_ccy}:")
          
                print(fx_price.head())
          
                print(fx_price.tail())
          
                print("nunique:", fx_price.nunique())

        return out
