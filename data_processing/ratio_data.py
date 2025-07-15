"""
Loads industry/sector valuation ratios, analyst data and price history, providing helper methods for region mapping and index returns.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import config

class RatioData:
    """
    Loads weighted PB, PE and forward PE ratios by Industry, Sector and Region.
    """
    
    def __init__(self):
        
        self.today = config.TODAY
       
        self.path1 = config.BASE_DIR / 'ind_data_mc_all_simple_mean.xlsx'  
       
        self.path2 = config.FORECAST_FILE
       
        self.path3 = config.DATA_FILE
       
        self.year_ago = config.YEAR_AGO
       
        self.five_year_ago = config.FIVE_YEAR_AGO
       
        self._load()


    def _load(self):
      
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
            sheet_name = ['Analyst Data', 'Analyst Target'],
            index_col = 0,
            engine = 'openpyxl'
        )
        
        self.analyst = sheets2['Analyst Data']
      
        temp_target = sheets2['Analyst Target']

        sheets3 = pd.read_excel(
            self.path3,
            sheet_name = ['Close', 'Weekly Close', 'Historic Returns', 'Historic Weekly Returns', 'Currency', 'Index Close'],
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

        self.tickers = self.analyst.index
      
        self.last_price = temp_target['Current Price']


    @staticmethod
    def determine_region(
        country
    ):
      
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
        
    
    def default(self, 
                val, 
                fallback
    ):
        
        return val if pd.notna(val) else fallback

    
    def mc_group(self, 
                 bounds, 
                 market_cap
    ):
      
        if pd.isna(market_cap):
            
            return 'Mid-Cap'

        for group, (lower, upper) in zip(
            bounds.index,
            bounds[['Low Bound', 'High Bound']].values
        ):
            
            if lower <= market_cap < upper:
                
                return group

        return 'Mid-Cap'

    
    def dicts(self):
      
        core_metrics = ['eps1y_5', 'eps1y', 'rev1y_5', 'rev1y', 'PB', 'PE', 'PS', 'EVS', 'FPE', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC']
      
        fallback_map = {'PS': 'FPS', 'PE': 'FPE', 'EVS': 'FEVS'}
      
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
                market_cap = self.analyst.loc[t, 'marketCap']
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


    def index_returns(self) -> tuple[pd.Series, pd.DataFrame]:

        major_indexes = ['^GSPC', '^NDX', '^FTSE', '^GDAXI', '^FCHI', '^AEX', '^IBEX', '^GSPTSE', '^HSI', '^SSMI']

        index_close = yf.download(major_indexes, start=self.five_year_ago, end=self.today)['Close'].squeeze().dropna()

        index_weekly_close = index_close.resample('W').last()
       
        index_weekly_rets = index_weekly_close.pct_change(fill_method=None).dropna()
        
        index_quarterly_close = index_close.resample('QE').last()
       
        index_quarterly_rets = index_quarterly_close.pct_change().dropna()

        annualised_rets = (index_close.iloc[-1] / index_close.iloc[0])**0.2 - 1

        return annualised_rets, index_weekly_rets, index_quarterly_rets
    
    
    def match_index_rets(self,
                         exchange: str,
                         index_rets: pd.Series,
                         index_weekly_rets: pd.DataFrame,
                         index_quarter_rets: pd.DataFrame,
                         bl_market_returns: pd.Series = None,
                         freq: str = "weekly") -> tuple[float, pd.Series]:
        
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
        
        
    def load_index_pred(self) -> pd.DataFrame:

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
    
    
    def crp(self):

        crp_file = config.BASE_DIR / 'crp.xlsx'
      
        crp = pd.read_excel(
            crp_file, 
            index_col = 0, 
            usecols = ['Country', 'CRP']
        )
      
        return crp
        
        
    def get_currency_annual_returns(self,
        country_to_pair: dict[str, str],
    ) -> pd.DataFrame:
        """
        Download daily FX rates for all GBP-to-foreign currency pairs and compute annual returns.

        Args:
            country_to_pair: Mapping from country name to currency-pair code (e.g. 'GBPUSD').
            start_date: ISO date string 'YYYY-MM-DD' for the start of the history.
            end_date: ISO date string 'YYYY-MM-DD' for the end of the history.

        Returns:
            DataFrame indexed by calendar year (int), columns are currency codes (e.g. 'USD'),
            values are annual total returns (float).
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
