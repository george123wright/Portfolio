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

    
    def dicts(
        self
    ):
      
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


    def index_returns(
        self
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:

        #major_indexes = ['^GSPC', '^NDX', '^FTSE', '^GDAXI', '^FCHI', '^AEX', '^IBEX', '^GSPTSE', '^HSI', '^SSMI']

        #index_close = yf.download(major_indexes, start=self.five_year_ago, end=self.today)['Close'].squeeze().dropna()
        
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
        
        
    def get_currency_annual_returns(
        self,
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
    
    
    def align_sector_sharpe(
        self
    ) -> pd.Series:
        """
        Aligns ticker with Sector Sharpe Ratios.
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
        Aligns ticker with Sector Sharpe Ratios.
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
        Aligns ticker with Index Sharpe Ratios.
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
        Compute the last-period pct-change (momentum) for each column in price_df
        using `period` weeks, then return its cross-sectional z-score.
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
        Map each ticker to the z-score of its sector’s momentum, 
        defaulting missing sectors to the *mean* sector momentum.
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
        Map each ticker to the z-score of its industry’s momentum.
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
        Map each ticker to the z-score of its benchmark index’s momentum.
        `index_price_df` has one column per index (weekly close).
        `ticker_to_index` maps ticker → index symbol.
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
        Given an exchange name, returns the primary benchmark index ticker symbol.
        Defaults to S&P 500 if the exchange isn’t in the map.
        """

        return ExchangeMapping.get(exchange_name.upper(), '^GSPC')
    
    
    def ticker_to_index_series(
        self
    ) -> pd.Series:
        """
        Maps each ticker to its primary benchmark index based on the exchange.
        """
        
        ticker_to_index_series = self.analyst['fullExchangeName'].map(self.pick_index)
        
        return ticker_to_index_series
    
    
    def exp_factor_data(
        self, 
        tickers = None
    ):
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
                
                
                
            
            
            
            
