"""
Loads industry/sector valuation ratios, analyst data and price history, providing helper methods for region mapping and index returns.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Sequence
from maps.index_exchange_mapping import ExchangeMapping
import config


class RatioData:
    """
    RatioData

    Central loader/adapter for valuation ratios, analyst snapshots, prices/returns,
    and group-level aggregates (industry, sector, region, market-cap buckets).
    Exposes convenience helpers to align group signals (Sharpe/momentum), construct
    factor inputs, and map tickers to benchmark indexes.

    Loaded inputs
    -------------
    From `ind_data_mc_all_simple_mean.xlsx` (path1):
    - Industry:                per-industry metrics
    - Sector MC Group:         (Sector, MC_Group_Merged) → metrics
    - Region-Industry:         (Region, Industry) → metrics
    - Region-Sector:           (Region, Sector) → metrics
    - MC Bounds:               market-cap bucket bounds
    - Industry MC Group:       (Industry, MC_Group_Merged) → metrics

    From `FORECAST_FILE` (path2):
    - Analyst Data:            fundamentals/qualitative fields per ticker
    - Analyst Target:          includes 'Current Price'
    - Sector/Industry/Index Data, Factor ETFs: performance summaries

    From `DATA_FILE` (path3):
    - Close, Weekly Close, Historic Returns, Historic Weekly Returns
    - Currency, Index Close, Sector Close, Industry Close, Factor Returns

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

    
    def __init__(
        self
    ):
        """
        Initialise container paths/dates from `config` and load all dependent datasets.

        Side effects
        ------------
        - Stores `today`, workbook paths, and common lookback anchors.
        - Calls `self._load()` to populate all public attributes.
        """

        self.today = config.TODAY
       
        self.path1 = config.BASE_DIR / 'ind_data_mc_all_simple_mean.xlsx'  
       
        self.path2 = config.FORECAST_FILE
       
        self.path3 = config.DATA_FILE
       
        self.year_ago = config.YEAR_AGO
       
        self.five_year_ago = config.FIVE_YEAR_AGO
       
        self._load()


    def _load(
        self
    ):
        """
        Read all required Excel sheets and derive secondary structures.

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

        Returns
        -------
        None

        Raises
        ------
        - Any pandas/openpyxl IO errors on missing files/sheets.
        - KeyError if required columns are missing in the provided workbooks.
        """

        sheets1 = pd.read_excel(
            self.path1,
            sheet_name = [
                'Industry',
                'Sector MC Group',
                'Region-Industry',
                'Region-Sector',
                'MC Bounds',
                'Industry MC Group'
            ],
            engine = 'openpyxl'
        )

        df_ind = sheets1['Industry'].rename(columns = {'Industry_grouped': 'Industry'})
     
        self.industry = df_ind.set_index('Industry')

        self.sector_mc = sheets1['Sector MC Group'].set_index(['Sector', 'MC_Group_Merged'])

        df_ri = sheets1['Region-Industry'].rename(columns = {'Industry_grouped': 'Industry'})
     
        self.region_ind = df_ri.set_index(['Region', 'Industry'])

        self.region_sec = sheets1['Region-Sector'].set_index(['Region', 'Sector'])

        self.bounds = sheets1['MC Bounds'].set_index('MC_Group_Merged')

        self.industry_mc = sheets1['Industry MC Group'].set_index(['Industry', 'MC_Group_Merged'])

        sheets2 = pd.read_excel(
            self.path2,
            sheet_name = ['Analyst Data', 'Analyst Target', 'Sector Data', 'Industry Data', 'Index Data', 'Factor ETFs'],
            index_col = 0,
            engine = 'openpyxl'
        )
        
        self.analyst = sheets2['Analyst Data']
      
        temp_target = sheets2['Analyst Target']
        
        self.sector_data = sheets2['Sector Data']
        
        self.industry_data = sheets2['Industry Data']
        
        self.index_data = sheets2['Index Data']
        
        self.factor_data = sheets2['Factor ETFs']
        
        self.mcap = self.analyst['marketCap']
        
        self.shares_outstanding = self.analyst['sharesOutstanding']
        
        self.tax_rate = self.analyst['Tax Rate'].fillna(0.22)

        sheets3 = pd.read_excel(
            self.path3,
            sheet_name = ['Close', 'Weekly Close', 'Historic Returns', 'Historic Weekly Returns', 'Currency', 'Index Close', 'Sector Close', 'Industry Close', 'Factor Returns'],
            index_col = 0,
            engine = 'openpyxl'
        )
        
        self.close = sheets3['Close'].sort_index(ascending = True)
      
        self.weekly_close = sheets3['Weekly Close'].sort_index(ascending = True)
      
        self.weekly_rets = sheets3['Historic Weekly Returns'].sort_index(ascending = True)
      
        self.daily_rets = sheets3['Historic Returns'].sort_index(ascending = True)
      
        self.currency = sheets3['Currency']['Last']
      
        self.index_close = sheets3['Index Close'].sort_index(ascending = True)
      
        self.quarterly_close = self.close.resample('QE').last()
      
        self.quarterly_rets = self.quarterly_close.pct_change()

        self.tickers = config.tickers
      
        self.last_price = temp_target['Current Price']
        
        self.sector_close = sheets3['Sector Close'].sort_index(ascending = True)
        
        self.industry_close = sheets3['Industry Close'].sort_index(ascending = True)
        
        self.factor_rets = sheets3['Factor Returns'].sort_index(ascending = True)
    

    @staticmethod
    def determine_region(
        country
    ):
        """
        Heuristic mapping from a free-form country string to a coarse region bucket.

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
        Map a country to Fama-French–style regional buckets.

        Parameters
        ----------
        country : Any
            Country name; if not a str, returns 'Emerging_Markets'.

        Returns
        -------
        str
            One of {'North_America', 'Europe', 'Asia_Pacific', 'Japan', 'Emerging_Markets'}.

        Notes
        -----
        - Logic uses substring checks; ensure input casing is consistent if modifying.
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
        bounds, 
        market_cap
    ):
        """
        Assign a market-capitalisation bucket given bounds and a point estimate.

        Parameters
        ----------
        bounds : pd.DataFrame
            Indexed by 'MC_Group_Merged' with columns ['Low Bound', 'High Bound'].
        market_cap : float | int | NA
            Company market cap.

        Returns
        -------
        str
            Matching bucket label from `bounds.index`, or 'Mid-Cap' as a default.

        Notes
        -----
        - Lower bound inclusive, upper bound exclusive:  lower ≤ cap < upper.
        - Treats NA market cap as 'Mid-Cap'.
        """

        if pd.isna(market_cap):
            
            return 'Mid-Cap'

        for group, (lower, upper) in zip(
            bounds.index,
            bounds[['Low Bound', 'High Bound']].values
        ):
            
            if lower <= market_cap < upper:
                
                return group

        return 'Mid-Cap'

    
    def dicts(
        self
    ):
        """
        Build a nested dictionary of core metrics for each ticker across four group lenses:
        'Sector-MC', 'Region-Industry', 'Region-Sector', and 'Industry-MC'.

        Metrics (per `core_metrics`)
        ----------------------------
        ['eps1y_5', 'eps1y', 'rev1y_5', 'rev1y', 'PB', 'PE', 'PS', 'EVS',
        'FPE', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC', 'PE10y']

        Fallbacks
        ---------
        If a metric is NA, may fall back to its forward analogue:
        {'PS': 'FPS', 'PE': 'FPE', 'EVS': 'FEVS'}
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
      
        core_metrics = ['eps1y_5', 'eps1y', 'rev1y_5', 'rev1y', 'PB', 'PE', 'PS', 'EVS', 'FPE', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC', 'PE10y']
      
        fallback_map = {
            'PS': 'FPS', 
            'PE': 'FPE', 
            'EVS': 'FEVS'
        }
      
        buckets = ['Industry', 'Sector-MC', 'Region-Industry', 'Region-Sector']

        results = {
            m: {t: {} for t in self.tickers}
            for m in core_metrics
        }

        for t in self.tickers:
            
            country = self.analyst.loc[t, 'country'] if 'country' in self.analyst.columns else None
            
            industry = self.analyst.loc[t, 'Industry']
            
            sector = self.analyst.loc[t, 'Sector']
            
            mc_group = self.mc_group(
                bounds = self.bounds, 
                market_cap = self.mcap[t]
            )
            
            region = self.determine_region(
                country = country
            )

            for m in core_metrics:
      
                try:
                    
                    val = self.sector_mc.loc[(sector, mc_group), m]
                
                except KeyError:
                    
                    val = np.nan
      
                if m in fallback_map and pd.isna(val):
      
                    fb = fallback_map[m]
      
                    try:
                       
                        val_fb = self.sector_mc.loc[(sector, mc_group), fb]
                    
                    except KeyError:
                        
                        val_fb = np.nan
      
                    if pd.notna(val_fb):
                       
                        val = val_fb
      
                results[m][t]['Sector-MC'] = val

            for m in core_metrics:
      
                try:
                   
                    val = self.region_ind.at[(region, industry), m]
                
                except KeyError:
                    
                    val = np.nan

                if pd.isna(val):
      
                    try:
                     
                        val = self.industry.at[industry, m]
                   
                    except KeyError:
                       
                        val = np.nan

                if m in fallback_map and pd.isna(val):
      
                    fb = fallback_map[m]

                    try:
                    
                        fb_val = self.region_ind.at[(region, industry), fb]
                    
                    except KeyError:
                     
                        fb_val = np.nan

                    if pd.isna(fb_val):
      
                        try:
                      
                            fb_val = self.industry_mc.at[(industry, mc_group), fb]
                       
                        except KeyError:
                        
                            fb_val = np.nan

                    if pd.isna(fb_val):
      
                        try:
                       
                            fb_val = self.industry.at[industry, fb]
                       
                        except KeyError:
                        
                            fb_val = np.nan

                    if pd.notna(fb_val):
      
                        val = fb_val

                results[m][t]['Region-Industry'] = val

            for m in core_metrics:
      
                try:
               
                    val = self.region_sec.loc[(region, sector), m]
               
                except KeyError:
               
                    val = np.nan
      
                if m in fallback_map and pd.isna(val):
      
                    fb = fallback_map[m]
      
                    try:
               
                        fb_val = self.region_sec.loc[(region, sector), fb]
               
                    except KeyError:
               
                        fb_val = np.nan
      
                    if pd.notna(fb_val):
               
                        val = fb_val
      
                results[m][t]['Region-Sector'] = val
                
            for m in core_metrics:
      
                try:
               
                    val = self.industry_mc.at[(industry, mc_group), m]
               
                except KeyError:
               
                    val = np.nan

                if pd.isna(val):
      
                    try:
               
                        val = self.industry.at[industry, m]
               
                    except KeyError:
               
                        val = np.nan

                results[m][t]['Industry-MC'] = val

        return results


    def index_returns(
        self
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Compute index-level annualised return (5y), weekly returns, and quarterly returns.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame, pd.DataFrame]
            (annualised_rets, index_weekly_rets, index_quarterly_rets)
        where
        - annualised_rets : Series of (last/first)^(1/5) - 1 over the available window.
        - index_weekly_rets : weekly pct-change of index closes (W resample, last), dropna.
        - index_quarterly_rets : quarterly pct-change (QE resample, last), dropna.

        Notes
        -----
        - Assumes `self.index_close` starts ≥ 5y; if shorter, the annualisation uses full span.
        """

        index_close = self.index_close
        
        index_weekly_close = index_close.resample('W').last()
       
        index_weekly_rets = index_weekly_close.pct_change(fill_method=None).dropna()
        
        index_quarterly_close = index_close.resample('QE').last()
       
        index_quarterly_rets = index_quarterly_close.pct_change().dropna()

        annualised_rets = (index_close.iloc[-1] / index_close.iloc[0])**0.2 - 1

        return annualised_rets, index_weekly_rets, index_quarterly_rets
    
    
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
        Select the appropriate benchmark index series and its summary return per exchange.

        Parameters
        ----------
        exchange : str
            Exchange name (e.g., 'NasdaqGS', 'LSE', 'NYSE', etc.).
        index_rets : pd.Series
            Per-index summary scalar (e.g., expected return or Sharpe) keyed by index symbol.
        index_weekly_rets : pd.DataFrame
            Weekly returns by index symbol.
        index_quarter_rets : pd.DataFrame
            Quarterly returns by index symbol.
        bl_market_returns : pd.Series, optional
            Overrides `index_rets` if provided and contains the symbol.
        freq : {'weekly', 'quarterly'}
            Frequency for the returned time series.

        Returns
        -------
        tuple[float, pd.Series]
            (selected_index_scalar, selected_index_returns)

        Notes
        -----
        - Hard-coded routing for major exchanges to symbols: ^NDX, ^GSPC, ^FTSE, ^GDAXI, etc.
        - Defaults to S&P 500 (^GSPC) if exchange not recognised.
        """
        
        def pick_ret(
            idx_name
        ):

            if bl_market_returns is not None and idx_name in bl_market_returns.index:
                
                return bl_market_returns[idx_name]

            else:
                
                return index_rets.get(idx_name, 0.0)

        if exchange in ['NasdaqGS', 'NasdaqGM', 'NasdaqCM']:
           
            if freq=="quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^NDX'
                )
               
                return index_ret, index_quarter_rets['^NDX']
           
            else:
                
                index_ret = pick_ret(
                    idx_name = '^NDX'
                )
                
                return index_ret, index_weekly_rets['^NDX']
            
        elif exchange == 'LSE':
           
            if freq=="quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^FTSE'
                )
                
                return index_ret, index_quarter_rets['^FTSE']
               
            else:
                
                index_ret = pick_ret(
                    idx_name = '^FTSE'
                )
               
                return index_ret, index_weekly_rets['^FTSE']
            
        elif exchange == 'NYSE':
            
            if freq=="quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^GSPC'
                )
                
                return index_ret, index_quarter_rets['^GSPC']
               
            else:
                
                index_ret = pick_ret(
                    idx_name = '^GSPC'
                )
                
                return index_ret, index_weekly_rets['^GSPC']
            
        elif exchange == 'XETRA':
           
            if freq=="quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^GDAXI'
                )
                
                return index_ret, index_quarter_rets['^GDAXI']
               
            else:
                
                index_ret = pick_ret(
                    idx_name = '^GDAXI'
                )
                
                return index_ret, index_weekly_rets['^GDAXI']
        
        elif exchange == 'MCE':
            
            if freq == "quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^IBEX'
                )
                
                return index_ret, index_quarter_rets['^IBEX']

            else:
                
                index_ret = pick_ret(
                    idx_name = '^IBEX'
                )
                
                return index_ret, index_weekly_rets['^IBEX']
            
        elif exchange == 'Amsterdam':
            
            if freq == "quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^AEX'
                )
                
                return index_ret, index_quarter_rets['^AEX']
            
            else:
                
                index_ret = pick_ret(
                    idx_name = '^AEX'
                )
                
                return index_ret, index_weekly_rets['^AEX']
        
        elif exchange == 'Paris':
            
            if freq == "quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^FCHI'
                )
                
                return index_ret, index_quarter_rets['^FCHI']
            
            else:
                
                index_ret = pick_ret(
                    idx_name = '^FCHI'
                )
                
                return index_ret, index_weekly_rets['^FCHI']
            
        elif exchange == 'Toronto':
            
            if freq == "quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^GSPTSE'
                )
                
                return index_ret, index_quarter_rets['^GSPTSE']
            
            else:
                
                index_ret = pick_ret(
                    idx_name = '^GSPTSE'
                )
                
                return index_ret, index_weekly_rets['^GSPTSE']
            
        elif exchange == 'HKSE':
            
            if freq == "quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^HSI'
                )
                
                return index_ret, index_quarter_rets['^HSI']
            else:
               
                index_ret = pick_ret(
                    idx_name = '^HSI'
                )
                
                return index_ret, index_weekly_rets['^HSI']
        
        elif exchange == 'Swiss':
            
            if freq == "quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^SSMI'
                )
                
                return index_ret, index_quarter_rets['^SSMI']
            else:
               
                index_ret = pick_ret(
                    idx_name = '^SSMI'
                )
                
                return index_ret, index_weekly_rets['^SSMI']
            
        else:
            
            if freq == "quarterly":
                
                index_ret = pick_ret(
                    idx_name = '^GSPC'
                )
                
                return index_ret, index_quarter_rets['^GSPC']
            else:
               
                index_ret = pick_ret(
                    idx_name = '^GSPC'
                )
                
                return index_ret, index_weekly_rets['^GSPC']
        
        
    def load_index_pred(
        self
    ) -> pd.DataFrame:
        """
        Load stock market forecast table from `DATA_FILE` ('Stock_Market' sheet), numericise, and average duplicate rows.

        Returns
        -------
        pd.DataFrame
            Table with numeric columns (commas removed), mean-aggregated over duplicate indices.

        Notes
        -----
        - Treats '0' as NA via `na_values=0` in the Excel read, then coerces numerics.
        """

        data = pd.read_excel(
            self.path3,
            header = 0, 
            index_col = 0, 
            na_values = 0, 
            sheet_name = 'Stock_Market'
        )

        for col in data.columns:

            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '', regex=False), errors='coerce')

        return data.groupby(level=0).mean()
    
    
    def crp(
        self
    ):
        """
        Load country risk premium (CRP) table from 'crp.xlsx'.

        Returns
        -------
        pd.DataFrame
            DataFrame with index 'Country' and a 'CRP' column.

        Raises
        ------
            - FileNotFoundError or parser errors if sheet/path is missing.
        """

        crp_file = config.BASE_DIR / 'crp.xlsx'
      
        crp = pd.read_excel(
            crp_file, 
            index_col = 0, 
            usecols = ['Country', 'CRP']
        )
      
        return crp
        
        
    def get_currency_annual_returns(
        self,
        country_to_pair: dict[str, str],
    ) -> pd.DataFrame:
        """
        Download GBP-cross FX weekly closes for given pairs and compute period returns.

        Parameters
        ----------
        country_to_pair : dict[str, str]
            Map of country → currency pair code without '=X' suffix (e.g., {'United States': 'GBPUSD'}).

        Returns
        -------
        tuple[pd.Series, pd.DataFrame]
            (ann_ret, rets)
        where
        - ann_ret : Series of total return over [YEAR_AGO, TODAY] for each pair
                    computed as (1 + weekly_ret).prod() - 1.
        - rets    : DataFrame of weekly pct-change returns.

        Notes
        -----
        - Pulls `start=self.year_ago`, `end=self.today`, `interval='1wk'` from yfinance with auto-adjust.
        - Pairs are passed as '{PAIR}=X' tickers (e.g., 'GBPUSD=X').
        """

        
        yf_tickers = [f"{pair}=X" for pair in country_to_pair.values()]
   
        close = yf.download(
            yf_tickers,
            start = self.year_ago,
            end = self.today,
            interval = '1wk',
            auto_adjust = True,
            progress = False
        )["Close"]

        rets = close.pct_change().dropna()
   
        ann_ret = (1 + rets).prod() - 1
   
        return ann_ret, rets 
    
    
    def align_sector_sharpe(
        self
    ) -> pd.Series:
        """
        Map each ticker to its sector's expected Sharpe ratio.

        Returns
        -------
        pd.Series
            Index=ticker, value=sector expected Sharpe (from 'Sector Data' → 'Exp Sharpe Ratio'),
            defaulting to 0 when sector not found.
        """

                
        sharpe = {}
        
        sec_sharpe = self.sector_data['Exp Sharpe Ratio']
        
        for t in self.tickers:
            
            sector = self.analyst.loc[t, 'Sector']
            
            if sector in sec_sharpe.index:
                
                sharpe[t] = sec_sharpe[sector]
            else:
                
                sharpe[t] = 0
        
        return pd.Series(sharpe).sort_index()
    
    
    def align_ind_sharpe(
        self
    ) -> pd.Series:
        """
        Map each ticker to its industry's expected Sharpe ratio.

        Returns
        -------
        pd.Series
            Index=ticker, value=industry expected Sharpe (from 'Industry Data' → 'Exp Sharpe Ratio'),
            defaulting to 0 when industry not found.
        """

                
        sharpe = {}
        
        ind_sharpe = self.industry_data['Exp Sharpe Ratio']
        
        for t in self.tickers:
            
            ind = self.analyst.loc[t, 'Industry']
            
            if ind in ind_sharpe.index:
                
                sharpe[t] = ind_sharpe[ind]
            else:
                
                sharpe[t] = 0
        
        return pd.Series(sharpe).sort_index()
    
    def index_sharpe(
        self
    ) -> pd.Series:
        """
        Compute an exchange-aligned expected Sharpe ratio per ticker using index returns.

        Procedure
        ---------
        - Take index weekly returns (5y), focus on last year.
        - Compute exponentially-weighted expected return/std with halflife = 0.1 * N,
        then annualise to an expected Sharpe via:  (E[R] − 0.0435) / E[σ].
        - Map each ticker's `fullExchangeName` to an index via `match_index_rets`
        and pick the corresponding scalar.

        Returns
        -------
        pd.Series
            Index=ticker, value=exchange-matched expected Sharpe.

        Notes
        -----
        - Uses a hard-coded 4.35% annual risk-free in the Sharpe (consistent with your pipeline).
        - Prints intermediate Series for debugging.
        """

        sharpe = {}
        
        index_rets = self.index_returns()[1]
        
        index_rets_last_year = index_rets.loc[self.year_ago:]
        
        rets_len = len(index_rets_last_year)
        
        exp_ret_exch = index_rets_last_year.ewm(halflife = 0.1 * rets_len, adjust = False).mean().iloc[-1] * rets_len
    
        exp_std_exch = index_rets_last_year.ewm(halflife = 0.1 * rets_len, adjust = False).std().iloc[-1] * np.sqrt(rets_len)
        
        exch = self.analyst['fullExchangeName']
        
        print(exp_ret_exch)
        print(exp_std_exch)
    
        exp_sr_exch = (exp_ret_exch - 0.0435) / exp_std_exch   
        
        index_ann_ret = (1 + index_rets_last_year).prod() - 1
        
        print(exp_sr_exch)
        
        for t in self.tickers:
            
            exch_t = exch[t]
            
            index_sr, index_weekly_rets = self.match_index_rets(
                exchange = exch_t,
                index_rets = exp_sr_exch,
                index_weekly_rets = index_rets_last_year,
                index_quarter_rets = index_rets_last_year,
            )
 
                
            sharpe[t] = index_sr
        
        return pd.Series(sharpe).sort_index()
        
        
    def _group_momentum_z(
        self,
        price_df: pd.DataFrame,
        period: int = 52
    ) -> pd.Series:
        """
        Compute cross-sectional z-score of group momentum.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price level time series (columns = groups: sectors/industries/indices).
        period : int, default 52
            Lookback periods (weeks) for momentum as pct-change.

        Returns
        -------
        pd.Series
            z = (mom − mean(mom)) / std(mom), for the last available row.
        """

        mom = price_df.pct_change(periods=period).iloc[-1]   

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

        return ExchangeMapping.get(exchange_name.upper(), '^GSPC')
    
    
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
        Assemble a per-ticker daily factor/design frame for regression/attribution.

        Parameters
        ----------
        tickers : list[str] | None
            Universe; defaults to `config.tickers` when None.

        Returns
        -------
        dict[str, pd.DataFrame]
            For each ticker t, a DataFrame indexed by date with columns:
            - factor ETF returns (from `self.factor_rets`)
            - 'Industry Return'  (daily pct-change of ticker's industry, if available)
            - 'Sector Return'    (daily pct-change of ticker's sector, if available)
            - 'Index Excess Return' (daily index return − RF_PER_DAY)
            - 'Ticker Excess Return' (ticker daily return − RF_PER_DAY)

        Notes
        -----
        - If the industry/sector/index series is missing, a NaN-aligned Series is used.
        - Uses `ticker_to_index_series()` to map the market series.
        """

        if tickers is None:
            
            tickers = list(config.tickers)

        factor_data = self.factor_rets
        
        ind_ret = self.industry_close.pct_change() 
        
        sec_ret = self.sector_close.pct_change() 
        
        idx_map = self.ticker_to_index_series()          
        
        idx_daily = self.index_close.pct_change().dropna() - config.RF_PER_DAY
        
        daily_rets = self.daily_rets - config.RF_PER_DAY

        ticker_factor = {}

        for t in tickers:
            
            t_ind = self.analyst.loc[t, 'Industry']
            
            t_sec = self.analyst.loc[t, 'Sector']

            if t_ind in ind_ret.columns:
               
                ind_series = ind_ret[t_ind]
            else:
                
                ind_series = pd.Series(np.nan, index=factor_data.index)  

            if t_sec in sec_ret.columns:
          
                sec_series = sec_ret[t_sec]
          
            else:
          
                sec_series = pd.Series(np.nan, index=factor_data.index)

            idx_symbol = idx_map.get(t, '^GSPC')
          
            if idx_symbol in idx_daily.columns:
          
                mkt_series = idx_daily[idx_symbol]
          
            else:
          
                mkt_series = pd.Series(np.nan, index=factor_data.index)
                
            daily_rets_t = daily_rets[t].dropna()

            df = (factor_data
                .join(ind_series.rename('Industry Return'), how = 'left')
                .join(sec_series.rename('Sector Return'), how = 'left')
                .join(mkt_series.rename('Index Excess Return'), how = 'left')
                .join(daily_rets_t.rename('Ticker Excess Return'), how = 'left')
            )

            ticker_factor[t] = df

        return ticker_factor
        
        
    def exp_factors(
        self, 
        tickers = None
    ):
        """
        Construct a per-ticker table of expected returns from industry/sector/index and factor ETFs.

        Parameters
        ----------
        tickers : list[str] | None
            Universe; defaults to `config.tickers`.

        Returns
        -------
        pd.DataFrame
            Columns: ['Industry','Sector','Index','Momentum','Quality','Size','Volatility','Value'],
            indexed by ticker.

        Sources
        -------
        - Industry/Sector/Index: 'Exp Returns' from respective aggregate tables.
        - Factor exposures: 'Exp Returns' of ETFs MTUM, QUAL, SIZE, USMV, VLUE.
        """
        
        if tickers is None:
        
            tickers = list(config.tickers)

        idx_exp = self.index_data['Exp Returns']      
       
        ind_exp = self.industry_data['Exp Returns']  
       
        sec_exp = self.sector_data['Exp Returns']     
       
        fac_exp = self.factor_data['Exp Returns']    

        idx_map = self.ticker_to_index_series()

        rows = []
        
        for t in tickers:
           
            t_ind = self.analyst.loc[t, 'Industry']
           
            t_sec = self.analyst.loc[t, 'Sector']
           
            t_idx = idx_map.get(t, '^GSPC')

            rows.append({
                'Ticker': t,
                'Industry': ind_exp.get(t_ind, 0.0),
                'Sector': sec_exp.get(t_sec, 0.0),
                'Index': idx_exp.get(t_idx, 0.0),
                'Momentum': fac_exp.get('MTUM', 0.0),
                'Quality': fac_exp.get('QUAL', 0.0),
                'Size': fac_exp.get('SIZE', 0.0),
                'Volatility': fac_exp.get('USMV', 0.0),
                'Value': fac_exp.get('VLUE', 0.0),
            })

        return pd.DataFrame(rows).set_index('Ticker')
    
    
    def factor_weekly_rets(
        self,
        resample_str = "W-FRI",
        after_date = None,
    ) -> pd.DataFrame:
        """
        Compute weekly factor ETF returns (log-summed) with optional start date.

        Parameters
        ----------
        resample_str : str, default "W-FRI"
            Pandas resample rule for weekly bars.
        after_date : str | pd.Timestamp | None
            If provided, restricts the sample to dates ≥ after_date.

        Returns
        -------
        pd.DataFrame
            Weekly factor returns for the set ['MTUM','QUAL','SIZE','USMV','VLUE'] present in data,
            computed as log(1+r) summed by week and dropped if all-NA.
        """
        
        FACTORS = ['MTUM', 'QUAL', 'SIZE', 'USMV', 'VLUE']
        
        factor_daily = self.factor_rets
        
        if after_date is not None:
            
            factor_daily = factor_daily.loc[after_date:]
        
        fac_w = (np.log1p(factor_daily)).resample(resample_str).sum().rename(columns=lambda c: str(c))
       
        fac_w = fac_w[[c for c in FACTORS if c in fac_w.columns]].dropna(how = "all")
        
        return fac_w
                
    
    def factor_index_ind_sec_weekly_rets(
        self,
        merge: bool = True
    ) -> pd.DataFrame:
        """
        Return weekly returns for factor ETFs, major indexes, industries, and sectors.

        Parameters
        ----------
        merge : bool, default True
            If True, returns a single DataFrame with all groups concatenated.
            If False, returns a 4-tuple.

        Returns
        -------
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            If merge:
                columns = [indexes | factors | industries | sectors] weekly returns.
            Else:
                (factor_weekly, index_weekly, industry_weekly, sector_weekly),
                each resampled to 'W-SUN' and pct-change (dropna).

        Notes
        -----
        - Index weekly returns come from `index_returns()[1]`.
        - Factor weekly returns are built via `factor_weekly_rets(resample_str="W-SUN", after_date=five_year_ago)`.
        - Industry/Sector returns are resampled closes (last of week) → pct_change.
        """
                
        index_weekly = self.index_returns()[1]
        
        factor_weekly = self.factor_weekly_rets(
            resample_str = "W-SUN",
            after_date = self.five_year_ago
        )
        
        ind_weekly_rets = self.industry_close.resample("W-SUN").last().pct_change().dropna()
        
        sec_weekly_rets = self.sector_close.resample("W-SUN").last().pct_change().dropna()
        
        if merge:
        
            df = pd.concat(
                [index_weekly, factor_weekly, ind_weekly_rets, sec_weekly_rets],
                axis = 1
            ).dropna(how = "all")
            
            return df
        
        else:
            
            return factor_weekly, index_weekly, ind_weekly_rets, sec_weekly_rets
                    
            
            
            
