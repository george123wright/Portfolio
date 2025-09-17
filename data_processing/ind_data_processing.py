"""
Industry/Sector Aggregation and Outlier-Robust Benchmarking from a Stock Screener
================================================================================

Purpose
-------
This module ingests a stock-screener Excel extract, standardises industry labels,
assigns companies to broad geographic regions, mitigates outliers within peer
groups, and produces a set of benchmark tables (industry, industry×market-cap,
sector×market-cap, region×industry, region×sector). The resulting tables are
exported to a multi-sheet Excel workbook for downstream analysis and reporting.

Inputs and outputs
------------------
- Input file  : `config.STOCK_SCREENER_FILE` (Excel)
- Output file : `config.IND_DATA_FILE` (Excel, written with `openpyxl`)

The input file is expected to contain at least the following columns
(others may be present and are ignored unless referenced below):

    ['Industry', 'Country', 'MC Group', 'Sector', 'Market Cap',
     'PS Ratio', 'PB Ratio', 'PE Ratio', 'EV/Sales', 'Fwd EV/S',
     'Forward PS', 'PE (10Y)', 'EV/Earnings', 'EV/FCF', 'EV/EBITDA',
     'EV/EBIT', 'EPS Gr. Next 5Y', 'Rev Gr. Next 5Y',
     'EPS Growth', 'Rev. Growth', 'ROE', 'ROA', 'ROIC']

Rows with missing values in the *required* subset
`['Industry', 'Country', 'MC Group', 'Sector', 'Market Cap']` are dropped.

Overview of processing
----------------------
1. **Industry normalisation**
   
   - `Industry_grouped` is created by mapping raw 'Industry' values through
     `industry_mapping`. The mapping is a mixture of anchored exact matches and
     case-insensitive regular expressions (e.g. `r'(?i).*REIT.*' → 'REIT'`).
     The first pattern that matches is applied; if none match, the original
     industry string is retained.

2. **Market-cap group consolidation**
  
   - 'MC Group' values 'Micro-Cap' and 'Nano-Cap' are merged into
     'Micro/Nano-Cap' and stored as `MC_Group_Merged`.

3. **Geographic region assignment**
   
   - A coarse region is assigned from 'Country' into one of
     {'Europe', 'Asia', 'Canada/Australia', 'Emerging Mkts (ex-Asia)',
     'United States', 'Other'} via list-based membership.

4. **Outlier mitigation (within peers)**
 
   - For a set of valuation/quality/growth metrics (see *Metrics* below),
     values are **clipped** within each peer group using the Tukey IQR rule:
 
       - Compute Q1 = 25th percentile and Q3 = 75th percentile of the series *within the group*.
 
       - Let IQR = Q3 − Q1.
 
       - Define lower bound L = Q1 − 1.5 × IQR and upper bound U = Q3 + 1.5 × IQR.
 
       - Replace values below L with L and above U with U.
 
     This preserves sample size while reducing the influence of extreme values.

5. **Groupwise aggregation**
  
   - After clipping, the module computes **group means** for each metric under
     four groupings, yielding the following output sheets:
  
     - **'Industry'**: by `Industry_grouped`.
  
     - **'Industry MC Group'**: by `['Industry', 'MC_Group_Merged']`.
  
     - **'Sector MC Group'**: by `['Sector', 'MC_Group_Merged']`.
  
     - **'Region-Industry'**: by `['Region', 'Industry_grouped']`.
  
     - **'Region-Sector'**: by `['Region', 'Sector']`.
  
   - In addition, a **'MC Bounds'** sheet is produced to document implied
     market-cap thresholds per `MC_Group_Merged` (see below).

6. **Derived growth-rate annualisations**
  
   - From 5-year growth expectations, compute equivalent constant annual rates:
  
       eps1y_5 = (1 + eps5y)^(1/5) − 1
  
       rev1y_5 = (1 + rev5y)^(1/5) − 1
  
     where `eps5y` and `rev5y` denote 5-year (cumulative) growth expectations.

Market-cap bounds construction
------------------------------
For each `MC_Group_Merged`, compute the maximum observed 'Market Cap' and sort
groups by this maximum in ascending order. The lower bound of a group is the
previous group's maximum (with 0 for the first), and the upper bound is the
group's own maximum; the last group's upper bound is set to infinity. This table
is written to the **'MC Bounds'** sheet with columns:
`['MC_Group_Merged', 'Low Bound', 'High Bound']`.

Metrics
-------
The clipping and aggregation operate on the following fields:

Valuation ratios:
    'PS Ratio', 'PB Ratio', 'PE Ratio', 'EV/Sales', 'Fwd EV/S',
    'Forward PS', 'PE (10Y)', 'EV/Earnings', 'EV/FCF', 'EV/EBITDA', 'EV/EBIT'

Growth:
    'EPS Gr. Next 5Y', 'Rev Gr. Next 5Y', 'EPS Growth', 'Rev. Growth'

Profitability/returns:
    'ROE', 'ROA', 'ROIC'

Each output table includes a subset of these (renamed for brevity where
appropriate), together with the derived annualised growth rates `eps1y_5` and
`rev1y_5`. The 'Industry' table is sorted by `eps1y` (descending). Region tables
are sorted by region (and sector for 'Region-Sector').

Assumptions and conventions
---------------------------
- **Data quality**: Percentiles/means are computed on the available values within
  each group; NaNs are ignored by `pandas` aggregation; rows missing any of the
  *required* columns are dropped at the outset.
- **Units**: Growth fields are treated as decimal rates (e.g., 0.12 = 12%).
- **Industry precedence**: The first regular-expression match in
  `industry_mapping` takes precedence; ordering of the mapping therefore matters.
- **Ordinality of market-cap groups**: The market-cap bounds rely on an implicit
  ordering induced by the observed maximum market caps per group.

Outputs (Excel sheets)
----------------------
- 'Industry'            : Outlier-clipped means by `Industry_grouped`, with
                          columns ['eps1y_5','eps1y','rev1y_5','rev1y', valuation,
                          profitability metrics].

- 'Industry MC Group'   : Means by `['Industry','MC_Group_Merged']` (no index merge).

- 'Sector MC Group'     : Means by `['Sector','MC_Group_Merged']`.

- 'Region-Industry'     : Means by `['Region','Industry_grouped']` (region and
                          industry retained as columns).

- 'Region-Sector'       : Means by `['Region','Sector']`.

- 'MC Bounds'           : Implied market-cap interval per `MC_Group_Merged`.

Dependencies
------------
- `pandas`, `numpy`, `openpyxl` (Excel writer engine), `re`
- A `config` module defining `STOCK_SCREENER_FILE` and `IND_DATA_FILE`.

This module is deterministic and contains no stochastic elements. It is intended
as a repeatable preprocessing and benchmarking step prior to security selection
or valuation analysis.
"""


import pandas as pd
import numpy as np
import re
import config

FILE_IN = config.STOCK_SCREENER_FILE

FILE_OUT = config.IND_DATA_FILE

europe = [
    'Belgium', 
    'British Virgin Islands', 
    'Costa Rica', 
    'Cyprus', 
    'Denmark', 
    'Finland', 
    'France',
    'Germany', 
    'Gibraltar', 
    'Greece',
    'Guernsey',
    'Ireland', 
    'Isle of Man',
    'Italy',
    'Luxembourg',
    'Netherlands',
    'Norway',
    'Spain',
    'Sweden', 
    'Switzerland', 
    'Turkey', 
    'United Kingdom'
]

asia = [
    'China',
    'Hong Kong', 
    'India', 
    'Indonesia', 
    'Israel', 
    'Japan', 
    'Kazakhstan', 
    'Macau', 
    'Malaysia',
    'Philippines',
    'SAR China', 
    'Singapore', 
    'South Korea',
    'Taiwan', 
    'Thailand', 
    'United Arab Emirates',
    'Vietnam'
]

canada_aus = [
    'Australia', 
    'Canada'
]

em_exc_asia = [
    'Argentina', 
    'Bahamas', 
    'Bermuda', 
    'Brazil', 
    'Cayman Islands',
    'Chile',
    'Colombia', 
    'Jordan',
    'Mexico', 
    'Monaco', 
    'Panama', 
    'Peru', 
    'South Africa',
    'Uruguay'
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


def std_industry(
    name
):
    
    for pattern, target in industry_mapping.items():
        
        if re.match(pattern, name):
        
            return target
    
    return name


def assign_region(
    country
):
    if country in europe:     
        
        return 'Europe'
    
    if country in asia:        
        
        return 'Asia'
    
    if country in canada_aus:  
        
        return 'Canada/Australia'
    
    if country in em_exc_asia: 
        
        return 'Emerging Mkts (ex-Asia)'
    
    if country in us:          
        
        return 'United States'
    
    return 'Other'


def clip_outliers(
    s: pd.Series
) -> pd.Series:
    
    Q1, Q3 = s.quantile([0.25, 0.75])
    
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return s.clip(lower, upper)


data = pd.read_excel(FILE_IN)

required = ['Industry', 'Country', 'MC Group', 'Sector', 'Market Cap']

sub = data.dropna(subset = required).copy()

sub['Industry_grouped'] = sub['Industry'].astype(str).map(std_industry)

sub['MC_Group_Merged'] = sub['MC Group'].replace({
    'Micro-Cap': 'Micro/Nano-Cap',
    'Nano-Cap': 'Micro/Nano-Cap'
})

sub['Region'] = sub['Country'].map(assign_region)

group_max = sub.groupby('MC_Group_Merged')['Market Cap'].max().sort_values()

group_min = group_max.shift(1).fillna(0)

group_max.iloc[-1] = np.inf

bounds = pd.DataFrame({
    'MC_Group_Merged': group_max.index,
    'Low Bound': group_min.values,
    'High Bound': group_max.values
})

ratio_cols = [
    'PS Ratio',
    'PB Ratio',
    'PE Ratio',
    'EV/Sales',
    'Fwd EV/S',
    'Forward PS',
    'PE (10Y)',
    'EV/Earnings',
    'EV/FCF',
    'EV/EBITDA',
    'EV/EBIT'
]

metrics = ratio_cols + ['EPS Gr. Next 5Y', 'Rev Gr. Next 5Y', 'EPS Growth', 'Rev. Growth', 'ROE', 'ROA', 'ROIC']

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
        eps5y = ('EPS Gr. Next 5Y', 'mean'),
        eps1y = ('EPS Growth', 'mean'),
        rev1y = ('Rev. Growth', 'mean'),
        rev5y = ('Rev Gr. Next 5Y', 'mean'),
        PS = ('PS Ratio', 'mean'),
        PB = ('PB Ratio', 'mean'),
        PE = ('PE Ratio', 'mean'),
        EVS = ('EV/Sales', 'mean'),
        FPS = ('Forward PS', 'mean'),
        FEVS = ('Fwd EV/S', 'mean'),
        ROE = ('ROE', 'mean'),
        ROA = ('ROA', 'mean'),
        ROIC = ('ROIC', 'mean'),
        PE10y = ('PE (10Y)', 'mean'),
        EVE = ('EV/Earnings', 'mean'),
        EVFCF = ('EV/FCF', 'mean'),
        EVEBITDA = ('EV/EBITDA', 'mean'),
        EVEBIT = ('EV/EBIT', 'mean')
    )
    .assign(
        eps1y_5 = lambda df: (1 + df.eps5y) ** (1/5) - 1,
        rev1y_5 = lambda df: (1 + df.rev5y) ** (1/5) - 1,
    )
    .loc[:, ['eps1y_5', 'eps1y', 'rev1y_5', 'rev1y', 'PS', 'PB', 'PE', 'EVS', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC', 'PE10y', 'EVE', 'EVFCF', 'EVEBITDA', 'EVEBIT']]
    .sort_values('eps1y', ascending = False)
)

sub_clipped_sec_mc = sub.copy()

sub_clipped_sec_mc[metrics] = (
    sub_clipped_sec_mc
    .groupby(['Sector', 'MC_Group_Merged'])[metrics]
    .transform(clip_outliers)
)

sec_mc = (
    sub_clipped_sec_mc
    .groupby(['Sector', 'MC_Group_Merged'])[metrics]
    .mean()
    .rename(columns={
        'EPS Gr. Next 5Y': 'eps5y', 
        'EPS Growth': 'eps1y',
        'Rev Gr. Next 5Y': 'rev5y', 
        'Rev. Growth': 'rev1y',
        'PS Ratio': 'PS',
        'PB Ratio': 'PB',
        'PE Ratio': 'PE',
        'EV/Sales':'EVS',
        'Forward PS': 'FPS',
        'Fwd EV/S': 'FEVS',
        'PE (10Y)': 'PE (10Y)',
        'EV/Earnings': 'EVE',
        'EV/FCF': 'EVFCF',
        'EV/EBITDA': 'EVEBITDA',
        'EV/EBIT': 'EVEBIT'
    })
    .assign(
        eps1y_5 = lambda df: (1 + df.eps5y) ** (1/5) - 1,
        rev1y_5 = lambda df: (1 + df.rev5y) ** (1/5) - 1,
    )
    .loc[:, ['eps1y_5', 'eps1y', 'rev1y_5', 'rev1y', 'PS', 'PB', 'PE', 'EVS', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC', 'PE (10Y)', 'EVE', 'EVFCF', 'EVEBITDA', 'EVEBIT']]
)

sub_clipped_ind_mc = sub.copy()

sub_clipped_ind_mc[metrics] = (
    sub_clipped_ind_mc
    .groupby(['Industry', 'MC_Group_Merged'])[metrics]
    .transform(clip_outliers)
)

ind_mc = (
    sub_clipped_ind_mc
    .groupby(['Industry', 'MC_Group_Merged'])[metrics]
    .mean()
    .rename(columns = {
        'EPS Gr. Next 5Y': 'eps5y', 
        'EPS Growth': 'eps1y',
        'Rev Gr. Next 5Y': 'rev5y',
        'Rev. Growth': 'rev1y',
        'PS Ratio': 'PS',
        'PB Ratio': 'PB',
        'PE Ratio': 'PE',
        'EV/Sales': 'EVS',
        'Forward PS': 'FPS',
        'Fwd EV/S': 'FEVS',
        'PE (10Y)': 'PE (10Y)',
        'EV/Earnings': 'EVE',
        'EV/FCF': 'EVFCF',
        'EV/EBITDA': 'EVEBITDA',
        'EV/EBIT': 'EVEBIT'
    })
    .assign(
        eps1y_5 = lambda df: (1 + df.eps5y) ** (1/5) - 1,
        rev1y_5 = lambda df: (1 + df.rev5y) ** (1/5) - 1,
    )
    .loc[:, ['eps1y_5', 'eps1y', 'rev1y_5', 'rev1y', 'PS', 'PB', 'PE', 'EVS', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC', 'PE (10Y)', 'EVE', 'EVFCF', 'EVEBITDA', 'EVEBIT']]
)

sub_clipped_reg_ind = sub.copy()

sub_clipped_reg_ind[metrics] = (
    sub_clipped_reg_ind
    .groupby(['Region', 'Industry_grouped'])[metrics]
    .transform(clip_outliers)
)

reg_ind = (
    sub_clipped_reg_ind
    .groupby(['Region','Industry_grouped'])
    .agg(
        eps5y = ('EPS Gr. Next 5Y', 'mean'),
        eps1y = ('EPS Growth', 'mean'),
        rev5y = ('Rev Gr. Next 5Y', 'mean'),
        rev1y = ('Rev. Growth', 'mean'),
        PS = ('PS Ratio', 'mean'),
        PB = ('PB Ratio', 'mean'),
        PE = ('PE Ratio', 'mean'),
        EVS = ('EV/Sales', 'mean'),
        FPS = ('Forward PS', 'mean'),
        FEVS = ('Fwd EV/S', 'mean'),
        ROE = ('ROE', 'mean'),
        ROA = ('ROA', 'mean'),
        ROIC = ('ROIC', 'mean'),
        PE10y = ('PE (10Y)', 'mean'),
        EVE = ('EV/Earnings', 'mean'),
        EVFCF = ('EV/FCF', 'mean'),
        EVEBITDA = ('EV/EBITDA', 'mean'),
        EVEBIT = ('EV/EBIT', 'mean')
    )
    .assign(
        eps1y_5 = lambda df: (1 + df.eps5y) ** (1/5) - 1,
        rev1y_5 = lambda df: (1 + df.rev5y) ** (1/5) - 1,
    )
    .reset_index()
    .loc[:, ['Region', 'Industry_grouped', 'eps1y_5', 'eps1y', 'rev1y', 'rev1y_5', 'PS', 'PB', 'PE', 'EVS', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC', 'PE10y', 'EVE', 'EVFCF', 'EVEBITDA', 'EVEBIT']]
    .sort_values(['Region', 'eps1y'], ascending = [True, False])
)

sub_clipped_reg_sec = sub.copy()

sub_clipped_reg_sec[metrics] = (
    sub_clipped_reg_sec
    .groupby(['Region', 'Sector'])[metrics]
    .transform(clip_outliers)
)

reg_sec = (
    sub_clipped_reg_sec
    .groupby(['Region', 'Sector'])
    .agg(
        eps5y = ('EPS Gr. Next 5Y', 'mean'),
        eps1y = ('EPS Growth', 'mean'),
        rev5y = ('Rev Gr. Next 5Y', 'mean'),
        rev1y = ('Rev. Growth', 'mean'),
        PS = ('PS Ratio', 'mean'),
        PB = ('PB Ratio', 'mean'),
        PE = ('PE Ratio', 'mean'),
        EVS = ('EV/Sales', 'mean'),
        FPS = ('Forward PS', 'mean'),
        FEVS = ('Fwd EV/S', 'mean'),
        ROE = ('ROE', 'mean'),
        ROA = ('ROA', 'mean'),
        ROIC = ('ROIC', 'mean'),
        PE10y = ('PE (10Y)', 'mean'),
        EVE = ('EV/Earnings', 'mean'),
        EVFCF = ('EV/FCF', 'mean'),
        EVEBITDA = ('EV/EBITDA', 'mean'),
        EVEBIT = ('EV/EBIT', 'mean')
    )
    .assign(
        eps1y_5 = lambda df: (1 + df.eps5y) ** (1/5) - 1,
        rev1y_5 = lambda df: (1 + df.rev5y) ** (1/5) - 1,
    )
    .reset_index()
    .loc[:, ['Region', 'Sector', 'eps1y_5', 'eps1y', 'rev1y_5', 'rev1y', 'PS', 'PB', 'PE', 'EVS', 'FPS', 'FEVS', 'ROE', 'ROA', 'ROIC', 'PE10y', 'EVE', 'EVFCF', 'EVEBITDA', 'EVEBIT']]
    .sort_values(['Region','Sector'])
)

with pd.ExcelWriter(FILE_OUT, engine = 'openpyxl') as writer:
    
    ind.to_excel(writer, sheet_name = 'Industry', index=True)
    
    ind_mc.to_excel(writer, sheet_name = 'Industry MC Group', index = True, merge_cells = False)
    
    sec_mc.to_excel(writer, sheet_name = 'Sector MC Group', index = True, merge_cells = False)
    
    reg_ind.to_excel(writer, sheet_name = 'Region-Industry', index = False)
    
    reg_sec.to_excel(writer, sheet_name = 'Region-Sector', index = False)
    
    bounds.to_excel(writer, sheet_name = 'MC Bounds', index = False)
    
