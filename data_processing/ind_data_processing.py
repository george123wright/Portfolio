import pandas as pd
import numpy as np
import re
import config

FILE_IN   = config.STOCK_SCREENER_FILE
FILE_OUT  = config.IND_DATA_FILE

europe = [
    'Belgium', 'British Virgin Islands', 'Costa Rica', 'Cyprus', 'Denmark',
    'Finland', 'France', 'Germany', 'Gibraltar', 'Greece', 'Guernsey',
    'Ireland', 'Isle of Man', 'Italy', 'Luxembourg', 'Netherlands', 'Norway',
    'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom'
]
asia = [
    'China', 'Hong Kong', 'India', 'Indonesia', 'Israel', 'Japan',
    'Kazakhstan', 'Macau', 'Malaysia', 'Philippines', 'SAR China',
    'Singapore', 'South Korea', 'Taiwan', 'Thailand',
    'United Arab Emirates', 'Vietnam'
]
canada_aus = ['Australia', 'Canada']
em_exc_asia = [
    'Argentina', 'Bahamas', 'Bermuda', 'Brazil', 'Cayman Islands', 'Chile',
    'Colombia', 'Jordan', 'Mexico', 'Monaco', 'Panama', 'Peru',
    'South Africa', 'Uruguay'
]
us = ['United States']

industry_mapping = {
    r'(?i).*REIT.*': 'REIT',
    r'^(Airlines|Airports & Air Services)$': 'Airlines, Airports & Air Services',
    r'^(Apparel Manufacturing|Apparel Retail|Textile Manufacturing)$': 'Apparel/Textile Manufacturing',
    r'^Oil & Gas$': 'Oil & Gas',
    r'^Drug Manufacturers - (General|Specialty & Generic)$': 'Drug Manufacturers',
    r'^Insurance - (Brokers|Diversified|Life|Property & Casualty|Reinsurance|Specialty)$': 'Insurance',
    r'^Banks - (Diversified|Regional)$': 'Banks',
    r'^Real Estate - (Development|Diversified|Services)$': 'Real Estate',
    r'^(Thermal Coal|Coking Coal)$': 'Coal',
    r'^(Aluminum|Copper|Steel|Industrial Metals|Other Industrial Metals & Mining|Other Precious Metals & Mining)$': 'Industrial Metals',
    r'^(Beverages - Brewers|Beverages - Wineries & Distilleries)$': 'Beverages - Alcoholic',
    # fallback exact mappings
    'Asset Management - Income': 'Asset Management',
    'Financial - Credit Services': 'Credit Services',
    'Medical - Healthcare Plans': 'Healthcare Plans',
    'Consumer Discretionary': 'Specialty Retail',
    'Other': 'Conglomerates',
   # 'Semiconductor Equipment & Materials': 'Semiconductors',
    'Specialty Chemicals': 'Chemicals',
}

def std_industry(name):
    for pattern, target in industry_mapping.items():
        if re.match(pattern, name):
            return target
    return name

def assign_region(country):
    if country in europe:      return 'Europe'
    if country in asia:        return 'Asia'
    if country in canada_aus:  return 'Canada/Australia'
    if country in em_exc_asia: return 'Emerging Mkts (ex-Asia)'
    if country in us:          return 'United States'
    return 'Other'

def clip_outliers(s: pd.Series) -> pd.Series:
    Q1, Q3 = s.quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    lower  = Q1 - 1.5 * IQR
    upper  = Q3 + 1.5 * IQR
    return s.clip(lower, upper)

data = pd.read_excel(FILE_IN)
required = ['Industry','Country','MC Group','Sector','Market Cap']
sub = data.dropna(subset=required).copy()

sub['Industry_grouped']   = sub['Industry'].astype(str).map(std_industry)
sub['MC_Group_Merged']    = sub['MC Group'].replace({
    'Micro-Cap':'Micro/Nano-Cap','Nano-Cap':'Micro/Nano-Cap'
})
sub['Region']             = sub['Country'].map(assign_region)

group_max = sub.groupby('MC_Group_Merged')['Market Cap'].max().sort_values()
group_min = group_max.shift(1).fillna(0)
group_max.iloc[-1] = np.inf
bounds = pd.DataFrame({
    'MC_Group_Merged': group_max.index,
    'Low Bound':       group_min.values,
    'High Bound':      group_max.values
})

ratio_cols = [
    'PS Ratio','PB Ratio','PE Ratio',
    'EV/Sales','Forward PE','Fwd EV/S','Forward PS'
]
metrics = ratio_cols + ['EPS Gr. Next 5Y','Rev Gr. Next 5Y', 'EPS Growth', 'Rev. Growth', 'ROE', 'ROA', 'ROIC']

sub_clipped_ind = sub.copy()
sub_clipped_ind[metrics] = (
    sub_clipped_ind
    .groupby('Industry_grouped')[metrics]
    .transform(clip_outliers)
)
ind = (
    sub_clipped_ind
    .groupby('Industry_grouped')
    .agg(
        eps5y=('EPS Gr. Next 5Y','mean'),
        eps1y=('EPS Growth','mean'),
        rev1y=('Rev. Growth','mean'),
        rev5y=('Rev Gr. Next 5Y','mean'),
        PS    =('PS Ratio','mean'),
        PB    =('PB Ratio','mean'),
        PE    =('PE Ratio','mean'),
        EVS   =('EV/Sales','mean'),
        FPE   =('Forward PE','mean'),
        FPS   =('Forward PS','mean'),
        FEVS  =('Fwd EV/S','mean'),
        ROE   =('ROE','mean'),
        ROA   =('ROA','mean'),
        ROIC  =('ROIC','mean'),
    )
    .assign(
        eps1y_5=lambda df: (1 + df.eps5y)**(1/5) - 1,
        rev1y_5=lambda df: (1 + df.rev5y)**(1/5) - 1,
    )
    .loc[:, ['eps1y_5','eps1y','rev1y_5','rev1y','PS','PB','PE','EVS','FPE','FPS','FEVS','ROE','ROA','ROIC']]
    .sort_values('eps1y', ascending=False)
)

sub_clipped_sec_mc = sub.copy()
sub_clipped_sec_mc[metrics] = (
    sub_clipped_sec_mc
    .groupby(['Sector','MC_Group_Merged'])[metrics]
    .transform(clip_outliers)
)
sec_mc = (
    sub_clipped_sec_mc
    .groupby(['Sector','MC_Group_Merged'])[metrics]
    .mean()
    .rename(columns={
        'EPS Gr. Next 5Y':'eps5y', 'EPS Growth':'eps1y',
        'Rev Gr. Next 5Y':'rev5y', 'Rev. Growth':'rev1y',
        'PS Ratio':'PS','PB Ratio':'PB','PE Ratio':'PE',
        'EV/Sales':'EVS','Forward PE':'FPE',
        'Forward PS':'FPS','Fwd EV/S':'FEVS'
        
    })
    .assign(
        eps1y_5=lambda df: (1 + df.eps5y)**(1/5) - 1,
        rev1y_5=lambda df: (1 + df.rev5y)**(1/5) - 1,
    )
    .loc[:, ['eps1y_5','eps1y','rev1y_5','rev1y','PS','PB','PE','EVS','FPE','FPS','FEVS','ROE','ROA','ROIC']]
)

sub_clipped_ind_mc = sub.copy()
sub_clipped_ind_mc[metrics] = (
    sub_clipped_ind_mc
    .groupby(['Industry','MC_Group_Merged'])[metrics]
    .transform(clip_outliers)
)

ind_mc = (
    sub_clipped_ind_mc
    .groupby(['Industry','MC_Group_Merged'])[metrics]
    .mean()
    .rename(columns={
        'EPS Gr. Next 5Y':'eps5y', 'EPS Growth':'eps1y',
        'Rev Gr. Next 5Y':'rev5y', 'Rev. Growth':'rev1y',
        'PS Ratio':'PS','PB Ratio':'PB','PE Ratio':'PE',
        'EV/Sales':'EVS','Forward PE':'FPE',
        'Forward PS':'FPS','Fwd EV/S':'FEVS'
        
    })
    .assign(
        eps1y_5=lambda df: (1 + df.eps5y)**(1/5) - 1,
        rev1y_5=lambda df: (1 + df.rev5y)**(1/5) - 1,
    )
    .loc[:, ['eps1y_5','eps1y','rev1y_5','rev1y','PS','PB','PE','EVS','FPE','FPS','FEVS','ROE','ROA','ROIC']]
)

sub_clipped_reg_ind = sub.copy()
sub_clipped_reg_ind[metrics] = (
    sub_clipped_reg_ind
    .groupby(['Region','Industry_grouped'])[metrics]
    .transform(clip_outliers)
)
reg_ind = (
    sub_clipped_reg_ind
    .groupby(['Region','Industry_grouped'])
    .agg(
        eps5y=('EPS Gr. Next 5Y','mean'),
        eps1y=('EPS Growth','mean'),
        rev5y=('Rev Gr. Next 5Y','mean'),
        rev1y=('Rev. Growth','mean'),
        PS    =('PS Ratio','mean'),
        PB    =('PB Ratio','mean'),
        PE    =('PE Ratio','mean'),
        EVS   =('EV/Sales','mean'),
        FPE   =('Forward PE','mean'),
        FPS   =('Forward PS','mean'),
        FEVS  =('Fwd EV/S','mean'),
        ROE   =('ROE','mean'),
        ROA   =('ROA','mean'),
        ROIC  =('ROIC','mean'),
    )
    .assign(
        eps1y_5=lambda df: (1 + df.eps5y)**(1/5) - 1,
        rev1y_5=lambda df: (1 + df.rev5y)**(1/5) - 1,
    )
    .reset_index()
    .loc[:, ['Region','Industry_grouped','eps1y_5','eps1y','rev1y','rev1y_5','PS','PB','PE','EVS','FPE','FPS','FEVS','ROE','ROA','ROIC']]
    .sort_values(['Region','eps1y'], ascending=[True,False])
)

sub_clipped_reg_sec = sub.copy()
sub_clipped_reg_sec[metrics] = (
    sub_clipped_reg_sec
    .groupby(['Region','Sector'])[metrics]
    .transform(clip_outliers)
)
reg_sec = (
    sub_clipped_reg_sec
    .groupby(['Region','Sector'])
    .agg(
        eps5y = ('EPS Gr. Next 5Y','mean'),
        eps1y = ('EPS Growth','mean'),
        rev5y = ('Rev Gr. Next 5Y','mean'),
        rev1y = ('Rev. Growth','mean'),
        PS = ('PS Ratio','mean'),
        PB = ('PB Ratio','mean'),
        PE = ('PE Ratio','mean'),
        EVS = ('EV/Sales','mean'),
        FPE = ('Forward PE','mean'),
        FPS = ('Forward PS','mean'),
        FEVS = ('Fwd EV/S','mean'),
        ROE = ('ROE','mean'),
        ROA = ('ROA','mean'),
        ROIC = ('ROIC','mean'),
    )
    .assign(
        eps1y_5=lambda df: (1 + df.eps5y)**(1/5) - 1,
        rev1y_5=lambda df: (1 + df.rev5y)**(1/5) - 1,
    )
    .reset_index()
    .loc[:, ['Region','Sector','eps1y_5','eps1y','rev1y_5','rev1y','PS','PB','PE','EVS','FPE','FPS','FEVS', 'ROE','ROA','ROIC']]
    .sort_values(['Region','Sector'])
)

with pd.ExcelWriter(FILE_OUT, engine='openpyxl') as writer:
    ind.to_excel(writer, sheet_name='Industry', index=True)
    ind_mc.to_excel(writer, sheet_name='Industry MC Group', index=True, merge_cells=False)
    sec_mc.to_excel(writer, sheet_name='Sector MC Group', index=True, merge_cells=False)
    reg_ind.to_excel(writer, sheet_name='Region-Industry', index=False)
    reg_sec.to_excel(writer, sheet_name='Region-Sector', index=False)
    bounds.to_excel(writer, sheet_name='MC Bounds', index=False)
