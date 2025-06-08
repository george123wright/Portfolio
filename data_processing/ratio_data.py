import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

class RatioData:
    """
    Loads weighted PB, PE and forward PE ratios by Industry, Sector and Region.
    """
    def __init__(self):
        self.today = dt.date.today()
        self.path1 = ''  
        self.path2 = '' 
        self.path3 = ''  
        self.year_ago = self.today - dt.timedelta(days=365)
        self._load()        

    def _load(self):
        sheets1 = pd.read_excel(
            self.path1,
            sheet_name=[
                'Industry',
                'Sector MC Group',
                'Region-Industry',
                'Region-Sector',
                'MC Bounds',
                'Industry MC Group'
            ],
            engine='openpyxl'
        )

        df_ind = sheets1['Industry'].rename(columns={'Industry_grouped': 'Industry'})
        self.industry = df_ind.set_index('Industry')

        self.sector_mc = sheets1['Sector MC Group'].set_index(['Sector', 'MC_Group_Merged'])

        df_ri = sheets1['Region-Industry'].rename(columns={'Industry_grouped': 'Industry'})
        self.region_ind = df_ri.set_index(['Region', 'Industry'])

        self.region_sec = sheets1['Region-Sector'].set_index(['Region', 'Sector'])

        self.bounds = sheets1['MC Bounds'].set_index('MC_Group_Merged')

        self.industry_mc = sheets1['Industry MC Group'].set_index(['Industry', 'MC_Group_Merged'])

        sheets2 = pd.read_excel(
            self.path2,
            sheet_name=['Analyst Data', 'Analyst Target'],
            index_col=0,
            engine='openpyxl'
        )
        self.analyst    = sheets2['Analyst Data']
        temp_target     = sheets2['Analyst Target']

        sheets3 = pd.read_excel(
            self.path3,
            sheet_name=['Close', 'Weekly Close', 'Historic Weekly Returns', 'Currency'],
            index_col=0,
            engine='openpyxl'
        )
        self.close        = sheets3['Close'].sort_index(ascending=True)
        self.weekly_close = sheets3['Weekly Close'].sort_index(ascending=True)
        self.weekly_rets  = sheets3['Historic Weekly Returns'].sort_index(ascending=True)
        self.currency     = sheets3['Currency']['Last']

        self.tickers    = self.analyst.index
        self.last_price = temp_target['Current Price']



    @staticmethod
    def determine_region(country):
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

    def default(self, val, fallback):
        return val if pd.notna(val) else fallback
    
    def mc_group(self, bounds, market_cap):
        if pd.isna(market_cap):
            return 'Mid-Cap'

        for group, (lower, upper) in zip(
            bounds.index,
            bounds[['Low Bound','High Bound']].values
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
            country  = self.analyst.loc[t, 'country'] if 'country' in self.analyst.columns else None
            industry = self.analyst.loc[t, 'Industry']
            sector   = self.analyst.loc[t, 'Sector']
            mc_group = self.mc_group(self.bounds, self.analyst.loc[t, 'marketCap'])
            region   = self.determine_region(country)

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
        index_close = yf.download(major_indexes, start=self.year_ago, end=self.today)['Close'].squeeze().dropna()
        index_weekly_close = index_close.resample('W').last()
        index_weekly_rets = index_weekly_close.pct_change(fill_method=None).dropna()
        annualised_rets = index_close.iloc[-1] / index_close.iloc[0] - 1
        return annualised_rets, index_weekly_rets
    
    def match_index_rets(self,
                         exchange: str,
                         index_rets: pd.Series,
                         index_weekly_rets: pd.DataFrame,
                         bl_market_returns: pd.Series = None) -> tuple[float, pd.Series]:
        
        def pick_ret(idx_name):
            if bl_market_returns is not None and idx_name in bl_market_returns.index:
                return bl_market_returns[idx_name]
            else:
                return index_rets.get(idx_name, 0.0)
        if exchange in ['NasdaqGS', 'NasdaqGM', 'NasdaqCM']:
            return pick_ret('^NDX'), index_weekly_rets['^NDX']
        elif exchange == 'LSE':
            return pick_ret('^FTSE'), index_weekly_rets['^FTSE']
        elif exchange == 'NYSE':
            return pick_ret('^GSPC'), index_weekly_rets['^GSPC']
        elif exchange == 'XETRA':
            return pick_ret('^GDAXI'), index_weekly_rets['^GDAXI']
        elif exchange == 'MCE':
            return pick_ret('^IBEX'), index_weekly_rets['^IBEX']
        elif exchange == 'Amsterdam':
            return pick_ret('^AEX'), index_weekly_rets['^AEX']
        elif exchange == 'Paris':
            return pick_ret('^FCHI'), index_weekly_rets['^FCHI']
        elif exchange == 'Toronto':
            return pick_ret('^GSPTSE'), index_weekly_rets['^GSPTSE']
        elif exchange == 'HKSE':
            return pick_ret('^HSI'), index_weekly_rets['^HSI']
        elif exchange == 'Swiss':
            return pick_ret('^SSMI'), index_weekly_rets['^SSMI']
        else:
            return pick_ret('^GSPC'), index_weekly_rets['^GSPC']
        
    def load_index_pred(self) -> pd.DataFrame:
        data = pd.read_excel(self.path3,
                             header=0, index_col=0, na_values=0, sheet_name='Stock_Market')
        for col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        return data.groupby(level=0).mean()
            
