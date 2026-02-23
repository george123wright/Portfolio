"""
Uses Yahoo Finance to pull analyst info, financial statements and estimates for each ticker and exports processed metrics.
"""

import numpy as np
import yfinance as yf
import pandas as pd
import scipy.stats as st
import logging
from typing import Any, Dict 
import time
from maps.industry_mapping import IndustryMap
from maps.sector_map import SectorMap
from functions.export_forecast import export_results
import config 
 

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch.setFormatter(formatter)

logger.addHandler(ch)

try:

    close = (
        pd.read_excel(
            config.DATA_FILE, 
            sheet_name = "Close", 
            index_col = 0, 
            parse_dates = True, 
            engine = "openpyxl"
        ).sort_index(ascending = True)
    )

    close.columns = close.columns.astype(str)

    rets = (
        pd.read_excel(
            config.DATA_FILE, 
            sheet_name = "Historic Returns", 
            index_col = 0, 
            parse_dates = True, 
            engine = "openpyxl"
        ).sort_index(ascending = True)
    )

    rets.columns = rets.columns.astype(str)

except Exception as e:

    logger.error(
        "Failed to read data from Excel. Ensure the file '%s' is not open in another application. Error: %s",
        config.DATA_FILE, e
    )

    raise

currency = pd.read_excel(
    config.DATA_FILE,
    sheet_name = 'Currency',
    index_col = 0,
    parse_dates = True,
    engine = 'openpyxl'
)

usdcad = pd.to_numeric(currency.loc['USDCAD']['Last'], errors = 'coerce')

gbpusd = pd.to_numeric(currency.loc['GBPUSD']['Last'], errors = 'coerce')

eurusd = pd.to_numeric(currency.loc['EURUSD']['Last'], errors = 'coerce')

usdchf = pd.to_numeric(currency.loc['USDCHF']['Last'], errors = 'coerce')

usdhkd = pd.to_numeric(currency.loc['USDHKD']['Last'], errors = 'coerce')

usdcny = pd.to_numeric(currency.loc['USDCNY']['Last'], errors = 'coerce')

usddkk = pd.to_numeric(currency.loc['USDDKK']['Last'], errors = 'coerce')

usdjpy = pd.to_numeric(currency.loc['USDJPY']['Last'], errors = 'coerce')

logger.info("Downloading Analyst Data from Yahoo Finance ...")


def safe_first_value(
    df: pd.DataFrame,
    row_label: str,
    default: float = np.nan
) -> float:
    """
    Safely retrieve the first value for a given row label from a DataFrame.
    Logs a warning if the row or index does not exist.
    """
  
    try:
  
        return df.loc[row_label].fillna(default).iloc[0]
  
    except Exception as e:
  
        logger.warning("Missing or invalid data for '%s': %s", row_label, e)
  
        return default


tickers = config.tickers

ticker_objs: Dict[str, yf.Ticker] = {}

targetsYF: Dict[str, Any] = {}

for ticker in tickers:
    
    try:
    
        ticker_obj = yf.Ticker(ticker)
    
        ticker_objs[ticker] = ticker_obj
    
        targetsYF[ticker] = ticker_obj.info
    
    except Exception as e:
    
        logger.warning("Failed to retrieve info for %s: %s", ticker, e)

desired_keys = [
    'targetMeanPrice',
    'targetMedianPrice',
    'targetLowPrice',
    'targetHighPrice',
    'numberOfAnalystOpinions',
    'dividendYield',
    'beta',
    'overallRisk',
    'recommendationKey',
    'sharesShort',
    'sharesShortPriorMonth',
    'sharesOutstanding',
    'earningsGrowth',
    'revenueGrowth',
    'debtToEquity',
    'returnOnAssets',
    'returnOnEquity',
    'priceToBook',
    'trailingEps',
    'forwardEps',
    'industryKey',
    'sectorKey',
    'country',
    'priceToSalesTrailing12Months',
    'enterpriseValue',
    'enterpriseToRevenue',
    'priceEpsCurrentYear',
    'forwardPE',
    'trailingPE',
    'profitMargins',
    'fullExchangeName',
    'marketCap',
    'bookValue',
    'lastDividendValue',
    'totalRevenue'
]


def get_historical_data(
    tickers,
    ticker_objs = ticker_objs
):
  
    financial_data = {}

    for t in tickers:
    
        logger.info("Obtaining Data for ticker %s", t)

        tk = ticker_objs.get(t, yf.Ticker(t))

        fin = tk.financials if hasattr(tk, 'financials') else pd.DataFrame()
     
        cf = tk.cashflow if hasattr(tk, 'cashflow') else pd.DataFrame()
     
        bs = tk.balance_sheet if hasattr(tk, 'balance_sheet') else pd.DataFrame()

     
        def one_year_est(
            df
        ):
        
            try:
        
                return df.loc['+1y', ['low', 'avg', 'high', 'numberOfAnalysts']]
        
            except Exception:
        
                return pd.Series({
                    'low': np.nan,
                    'avg': np.nan,
                    'high': np.nan,
                    'numberOfAnalysts': 0
                })


        rev1y = one_year_est(
            df = tk.revenue_estimate
        )
        
        eps1y = one_year_est(
            df = tk.earnings_estimate
        )

        eps_revisions = tk.eps_revisions

        financial_data[t] = {
            'financials': fin,
            'cashflow': cf,
            'balance_sheet': bs,
            'rev_estimate': rev1y,
            'eps_estimate': eps1y,
            'eps_revisions': eps_revisions
        }

        time.sleep(0.5)

    return financial_data


financial_data = get_historical_data(
    tickers = tickers
)


def summarize_eps_revisions(
    eps_rev: pd.DataFrame
) -> tuple[float, float]:
    """
    Take yfinance .eps_revisions DataFrame and return two summary signals:
    net breadth last 7 days, net breadth last 30 days.

    breadth_7d  = sum[(#up - #down) over all non-NaN periods]
    breadth_30d = same for 30d

    If eps_rev is None or empty, returns (0.0, 0.0).
    """
    
    if eps_rev is None or not isinstance(eps_rev, pd.DataFrame) or eps_rev.empty:
      
        return 0.0, 0.0

    cols_needed = ["upLast7days", "downLast7Days", "upLast30days", "downLast30days"]
   
    available = [c for c in cols_needed if c in eps_rev.columns]
   
    eps_rev = eps_rev[available].apply(pd.to_numeric, errors="coerce")
   
    eps_rev = eps_rev.fillna(0.0)

    diff7 = (
        eps_rev["upLast7days"] - eps_rev["downLast7Days"]
        if "upLast7days" in eps_rev and "downLast7Days" in eps_rev
        else pd.Series(dtype = float)
    )
    
    diff30 = (
        eps_rev["upLast30days"] - eps_rev["downLast30days"]
        if "upLast30days" in eps_rev and "downLast30days" in eps_rev
        else pd.Series(dtype = float)
    )
    
    if "upLast7days" in eps_rev:
        
        eu7 = eps_rev["upLast7days"]  
    
    else:
        
        eu7 = pd.Series(dtype = float)
        
    if "downLast7Days" in eps_rev:
    
        ed7 = eps_rev["downLast7Days"] 
        
    else:
        
        ed7 = pd.Series(dtype = float)
    
    diff7  = eu7 - ed7
    
    if "upLast30days" in eps_rev:
        
        eu30 = eps_rev["upLast30days"] - eu7 
    
    else:
        
        eu30 = pd.Series(dtype = float)
    
    if "downLast30days" in eps_rev:
        
        ed30 = eps_rev["downLast30days"] - ed7  
    
    else:
        
        ed30 = pd.Series(dtype = float)
    
    diff30 = eu30 - ed30
    
    positive_count_7 = (diff7 > 0).sum()
    
    negative_count_7 = (diff7 < 0).sum()
    
    positive_count_30 = (diff30 > 0).sum()
    
    negative_count_30 = (diff30 < 0).sum()

    breadth7 = positive_count_7 - negative_count_7
    
    breadth30 = positive_count_30 - negative_count_30

    return breadth7, breadth30


def extract_metrics_for_ticker(
    ticker: str,
    data: dict
) -> dict:
    """
    data is the dict from financial_data[ticker]:
      {
        'financials': DataFrame,
        'cashflow': DataFrame,
        'balance_sheet': DataFrame,
        'rev_estimate': Series,
        'eps_estimate': Series,
        'eps_revisions': DataFrame or None
      }

    Returns a dict of scalar metrics for that ticker.
    """
    
    financials = data.get('financials')
  
    cashflow = data.get('cashflow')
  
    balance_sheet = data.get('balance_sheet')
  
    rev_estimate = data.get('rev_estimate')
  
    eps_estimate = data.get('eps_estimate')
  
    eps_revisions = data.get('eps_revisions')

    if financials is None or cashflow is None or balance_sheet is None:

        return {
            'Ticker': ticker
        }

    b7, b30 = summarize_eps_revisions(
        eps_rev = eps_revisions
    )

    net_income_val = safe_first_value(
        df = financials, 
        row_label = 'Net Income'
    )
    
    operating_cash_flow_val = safe_first_value(
        df = cashflow, 
        row_label = 'Operating Cash Flow'
    )

    total_assets_val = safe_first_value(
        df = balance_sheet, 
        row_label = 'Total Assets'
    )

    if 'Total Assets' in balance_sheet.index:
        
        total_assets_series = balance_sheet.loc['Total Assets'].dropna()[::-1]

        if len(total_assets_series) >= 3:
            
            prev_total_assets = total_assets_series.iloc[-2]
            
            prev_prev_total_assets = total_assets_series.iloc[-3]

            asset_growth_rates = total_assets_series.pct_change().dropna()
            
            asset_growth_rate = asset_growth_rates.mean()
            
            asset_growth_rate_vol = asset_growth_rates.std()
        
        else:
        
            prev_total_assets = np.nan
        
            prev_prev_total_assets = np.nan
        
            asset_growth_rate = np.nan
        
            asset_growth_rate_vol = np.nan

        if 'Research And Development' in financials.index:
        
            rnd = safe_first_value(
                df = financials, 
                row_label = 'Research And Development',
                default = 0
            )
        
            rnd_assets = rnd / total_assets_val if total_assets_val != 0 else np.nan
        
        else:
        
            rnd = np.nan
        
            rnd_assets = np.nan

        if 'Capital Expenditure' in cashflow.index:
           
            capital_expenditure = safe_first_value(
                df = cashflow, 
                row_label = 'Capital Expenditure',
                default = 0
            )
           
            capex_intensity = (
                capital_expenditure / total_assets_val if total_assets_val != 0 else np.nan
            )
        
        else:
        
            capital_expenditure = np.nan
        
            capex_intensity = np.nan

        if 'EBIT' in financials.index:
          
            ebit = safe_first_value(
                df = financials, 
                row_label = 'EBIT', 
                default = 0
            )
          
            ebit_assets = ebit / total_assets_val if total_assets_val != 0 else np.nan
       
        else:
       
            ebit = np.nan
       
            ebit_assets = np.nan
    else:
     
        prev_total_assets = np.nan
     
        prev_prev_total_assets = np.nan
     
        asset_growth_rate = np.nan
     
        asset_growth_rate_vol = np.nan
     
        rnd = np.nan
     
        rnd_assets = np.nan
     
        capital_expenditure = np.nan
     
        capex_intensity = np.nan
     
        ebit = np.nan
     
        ebit_assets = np.nan

    if 'Net Income' in financials.index and financials.loc['Net Income'].size > 1:
     
        prev_net_income = financials.loc['Net Income'].iloc[1]
   
    else:
   
        prev_net_income = 0

    if 'Common Stock Equity' in balance_sheet.index and not balance_sheet.loc['Common Stock Equity'].empty:

        book_value = balance_sheet.loc['Common Stock Equity', balance_sheet.columns[0]]

    elif 'Stockholders Equity' in balance_sheet.index and not balance_sheet.loc['Stockholders Equity'].empty:

        book_value = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]]

    else:

        book_value = np.nan

    if 'Total Liabilities Net Minority Interest' in balance_sheet.index:

        tot_liab_series = balance_sheet.loc['Total Liabilities Net Minority Interest'].dropna()[::-1]

        if len(tot_liab_series) >= 1:

            tot_liab_c = tot_liab_series.iloc[-1]

        else:

            tot_liab_c = np.nan

        if len(tot_liab_series) > 1:

            tot_liabs_growth = tot_liab_series.pct_change().dropna()

            tot_liab_growth = tot_liabs_growth.mean()

            tot_liab_growth_vol = tot_liabs_growth.std()

        else:

            tot_liab_growth = np.nan

            tot_liab_growth_vol = np.nan

    else:

        tot_liab_c = np.nan

        tot_liab_growth = np.nan

        tot_liab_growth_vol = np.nan

    average_assets_val = (
        total_assets_val if pd.isna(prev_total_assets)
        else (total_assets_val + prev_total_assets) / 2
    )

    previous_average_assets_val = (
        prev_total_assets if pd.isna(prev_prev_total_assets)
        else (prev_total_assets + prev_prev_total_assets) / 2
    )

    return_on_assets_val = (
        net_income_val / average_assets_val if average_assets_val != 0 else np.nan
    )

    previous_return_on_assets_val = (
        prev_net_income / previous_average_assets_val
        if previous_average_assets_val != 0 else np.nan
    )

    if 'Long Term Debt' in balance_sheet.index:

        long_term_debt_val = safe_first_value(
            df = balance_sheet, 
            row_label = 'Long Term Debt', 
            default = 0
        )

        try:

            prev_long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[1]

        except Exception:

            prev_long_term_debt = np.nan

    else:

        long_term_debt_val = 0

        prev_long_term_debt = 0

    if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
       
        current_assets_val = safe_first_value(
            df = balance_sheet, 
            row_label = 'Current Assets',
            default = 0
        )
       
        current_liabilities_val = safe_first_value(
            df = balance_sheet, 
            row_label = 'Current Liabilities',
            default = 0
        )
       
        try:
       
            prev_current_assets = balance_sheet.loc['Current Assets'].iloc[1]
       
        except Exception:
       
            prev_current_assets = np.nan
       
        try:
       
            prev_current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[1]
       
        except Exception:
       
            prev_current_liabilities = np.nan
    
    else:
    
        current_assets_val = 0
    
        current_liabilities_val = 0
    
        prev_current_assets = np.nan
    
        prev_current_liabilities = np.nan

    current_ratio_val = (
        current_assets_val / current_liabilities_val
        if current_liabilities_val != 0 else np.nan
    )

    previous_current_ratio_val = (
        prev_current_assets / prev_current_liabilities
        if pd.notna(prev_current_assets)
           and pd.notna(prev_current_liabilities)
           and prev_current_liabilities != 0
        else np.nan
    )

    if 'Issuance Of Capital Stock' in cashflow.index:

        new_shares_issued_val = safe_first_value(
            df = cashflow, 
            row_label = 'Issuance Of Capital Stock', 
            default = 0
        )
        
    else:
        
        new_shares_issued_val = 0

    if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index:

        total_revenue = safe_first_value(
            df = financials, 
            row_label = 'Total Revenue',
            default = 0
        )

        gross_profit = safe_first_value(
            df = financials, 
            row_label = 'Gross Profit', 
            default = 0
        )

        gross_margin_val = gross_profit / total_revenue if total_revenue != 0 else 0

        try:
       
            prev_total_revenue = financials.loc['Total Revenue'].iloc[1]
       
            prev_gross_profit = financials.loc['Gross Profit'].iloc[1]
       
            previous_gross_margin_val = (
                prev_gross_profit / prev_total_revenue
                if prev_total_revenue != 0 else np.nan
            )
       
        except Exception:
       
            previous_gross_margin_val = np.nan
    
    else:
    
        gross_margin_val = 0
    
        previous_gross_margin_val = 0

    if 'Tax Rate For Calcs' in financials.index:

        tax_rate = safe_first_value(
            df = financials, 
            row_label = 'Tax Rate For Calcs',
            default = 0.21
        )

    else:

        tax_rate = 0.21

    if 'Total Revenue' in financials.index and 'Current Assets' in balance_sheet.index:

        total_revenue = safe_first_value(
            df = financials, 
            row_label = 'Total Revenue', 
            default = 0
        )

        try:

            prev_total_revenue = financials.loc['Total Revenue'].iloc[1]

        except Exception:

            prev_total_revenue = np.nan

        asset_turnover = (
            total_revenue / average_assets_val
            if average_assets_val != 0 else np.nan
        )
        
        prev_asset_turnover = (
            prev_total_revenue / previous_average_assets_val
            if previous_average_assets_val != 0 else np.nan
        )
        
    else:
       
        asset_turnover = np.nan
       
        prev_asset_turnover = np.nan

    ticker_obj = ticker_objs.get(ticker, yf.Ticker(ticker))

    try:
        insider_trans = ticker_obj.insider_purchases

        if not insider_trans.empty and len(insider_trans) > 2:

            insider_purchase = insider_trans.iloc[2]['Shares']

        else:

            insider_purchase = np.nan

    except Exception as e:

        logger.warning("Failed to retrieve insider purchases for %s: %s", ticker, e)

        insider_purchase = np.nan

    rev_low_estimate = rev_estimate['low']

    rev_avg_estimate = rev_estimate['avg']

    rev_high_estimate = rev_estimate['high']

    rev_num_analysts = rev_estimate['numberOfAnalysts']

    eps_low_estimate = eps_estimate['low']

    eps_avg_estimate = eps_estimate['avg']

    eps_high_estimate = eps_estimate['high']

    eps_num_analysts = eps_estimate['numberOfAnalysts']

    return {
        'Ticker': ticker,
        'EPS Revision 7D': b7,
        'EPS Revision 30D': b30,
        "Net Income": net_income_val,
        "Operating Cash Flow": operating_cash_flow_val,
        "Total Assets": total_assets_val,
        "Average Assets": average_assets_val,
        "Return on Assets": return_on_assets_val,
        "Previous Return on Assets": previous_return_on_assets_val,
        "Long Term Debt": long_term_debt_val,
        "Previous Long Term Debt": prev_long_term_debt,
        "Current Assets": current_assets_val,
        "Current Liabilities": current_liabilities_val,
        "Previous Current Liabilities": prev_current_liabilities,
        "Current Ratio": current_ratio_val,
        "Previous Current Ratio": previous_current_ratio_val,
        "New Shares Issued": new_shares_issued_val,
        "Gross Margin": gross_margin_val,
        "Previous Gross Margin": previous_gross_margin_val,
        "Insider Purchases": insider_purchase,
        "Asset Turnover": asset_turnover,
        "Previous Asset Turnover": prev_asset_turnover,
        "Low Revenue Estimate": rev_low_estimate,
        "Avg Revenue Estimate": rev_avg_estimate,
        "High Revenue Estimate": rev_high_estimate,
        "Num Analyst Revenue": rev_num_analysts,
        "Low EPS Estimate": eps_low_estimate,
        "Avg EPS Estimate": eps_avg_estimate,
        "High EPS Estimate": eps_high_estimate,
        "Num Analyst EPS": eps_num_analysts,
        "Book Value": book_value,
        "Total Liabilities": tot_liab_c,
        "Total Liabilities Growth": tot_liab_growth,
        "Asset Growth Rate": asset_growth_rate,
        "Asset Growth Rate Vol": asset_growth_rate_vol,
        "Total Liabilities Growth Vol": tot_liab_growth_vol,
        'Capex Intensity': capex_intensity,
        "Tax Rate": tax_rate
    }


metric_rows = []

for ticker in tickers:

    try:

        row_metrics = extract_metrics_for_ticker(
            ticker = ticker,
            data = financial_data.get(ticker, {})
        )
      
        metric_rows.append(row_metrics)
    
    except Exception as e:
    
        logger.warning("Could not process financial metrics for %s: %s", ticker, e)
    
        continue

metrics_df = pd.DataFrame(metric_rows).set_index("Ticker")

targets_df = (
    pd.DataFrame(targetsYF)
    .T[desired_keys]
    .fillna(0)
    .infer_objects()
)

targets_df['fullExchangeName'] = targets_df['fullExchangeName'].fillna('NYSE')

targets_df['debtToEquity'] = targets_df['debtToEquity'].fillna(1)

targets_df.index.name = "Ticker"

targets_df.columns = targets_df.columns.astype(str)

targets_df = targets_df.reindex(sorted(targets_df.index))

exemptions = config.TICKER_EXEMPTIONS

numeric_cols = targets_df.select_dtypes(include = [np.number]).columns

for ex in exemptions:

    if ex in targets_df.index:

        targets_df.loc[ex, numeric_cols] = 0


def map_yfinance_industry(
    yf_industry: str
) -> str:

    yf_industry = (yf_industry or "").strip()

    return IndustryMap.get(yf_industry, "Diversified")


def map_yfinance_sector(
    yf_sector: str
) -> str:

    yf_sector = (yf_sector or "").strip()

    return SectorMap.get(yf_sector, "Other")


targets_df['industryKey'] = targets_df['industryKey'].apply(map_yfinance_industry)

targets_df['sectorKey'] = targets_df['sectorKey'].apply(map_yfinance_sector)

rename_map = {
    'industryKey': 'Industry',
    'sectorKey': 'Sector'
}

targets_df.rename(columns = rename_map, inplace = True)

if "SGLP.L" in targets_df.index:
    
    targets_df.loc["SGLP.L", ["Industry", "Sector", "country"]] = ["Gold", "Materials", "United Kingdom"]

targets_df = targets_df.join(metrics_df, how = "left")

if "SGLP.L" in targets_df.index:

    targets_df.loc["SGLP.L", 'Industry'] = 'Gold'

    targets_df.loc["SGLP.L", 'Sector'] = 'Materials'

    targets_df.loc["SGLP.L", 'country'] = 'United Kingdom'

logger.info("Download Complete")

uk_mask = targets_df.index.str.endswith('.L')

cad_mask = targets_df.index.str.endswith('.TO')

eur_mask = (
    targets_df.index.str.endswith('.MC') |
    targets_df.index.str.endswith('.PA') |
    targets_df.index.str.endswith('.AS') |
    targets_df.index.str.endswith('.DE')
)

chf_mask = targets_df.index.str.endswith('.SW')

hkd_mask = targets_df.index.str.endswith('.HK')

dkk_mask = targets_df.index.str.endswith('.CO')

jp_mask = targets_df.index.str.endswith('.T')

twd_mask = (
    targets_df.index.str.endswith('.TW') |
    targets_df.index.str.endswith('.TWO')
)

cny_mask = targets_df.index.isin(['BABA', 'PDD'])

gbp_big_mask = targets_df.index.isin(['HSBA.L', 'AZN.L', 'IWG.L', 'BP.L'])

targets_df.loc[uk_mask, 'targetMeanPrice'] = targets_df.loc[uk_mask, 'targetMeanPrice'] / 100

targets_df.loc[uk_mask, 'targetMedianPrice'] = targets_df.loc[uk_mask, 'targetMedianPrice'] / 100

targets_df.loc[uk_mask, 'targetLowPrice'] = targets_df.loc[uk_mask, 'targetLowPrice'] / 100

targets_df.loc[uk_mask, 'targetHighPrice'] = targets_df.loc[uk_mask, 'targetHighPrice'] / 100

targets_df.loc[uk_mask, 'forwardEps'] = targets_df.loc[uk_mask, 'forwardEps'] / 100

targets_df.loc[uk_mask, 'priceEpsCurrentYear'] = targets_df.loc[uk_mask, 'priceEpsCurrentYear'] / 100

targets_df.loc[uk_mask, 'priceToBook'] = targets_df.loc[uk_mask, 'priceToBook'] / 100

targets_df.loc[uk_mask, 'marketCap']  = targets_df.loc[uk_mask, 'marketCap'] / gbpusd

targets_df.loc[cad_mask, 'marketCap'] = targets_df.loc[cad_mask, 'marketCap'] / usdcad

targets_df.loc[cad_mask, 'enterpriseValue'] = targets_df.loc[cad_mask, 'enterpriseValue'] / usdcad

targets_df.loc[eur_mask, 'marketCap'] = targets_df.loc[eur_mask, 'marketCap'] * eurusd

targets_df.loc[eur_mask, 'enterpriseValue'] = targets_df.loc[eur_mask, 'enterpriseValue'] * eurusd

targets_df.loc[chf_mask, 'marketCap'] = targets_df.loc[chf_mask, 'marketCap'] * usdchf

targets_df.loc[chf_mask, 'enterpriseValue'] = targets_df.loc[chf_mask, 'enterpriseValue'] * usdchf

targets_df.loc[hkd_mask, 'marketCap'] = targets_df.loc[hkd_mask, 'marketCap'] / usdhkd

targets_df.loc[hkd_mask, 'enterpriseValue'] = targets_df.loc[hkd_mask, 'enterpriseValue'] / usdhkd

targets_df.loc[dkk_mask, 'marketCap'] = targets_df.loc[dkk_mask, 'marketCap'] / usddkk

targets_df.loc[dkk_mask, 'enterpriseValue'] = targets_df.loc[dkk_mask, 'enterpriseValue'] / usddkk

targets_df.loc[jp_mask, 'marketCap'] = targets_df.loc[jp_mask, 'marketCap'] / usdjpy

targets_df.loc[jp_mask, 'enterpriseValue'] = targets_df.loc[jp_mask, 'enterpriseValue'] / usdjpy

usdtwd = 31.6

targets_df.loc[twd_mask, 'Low Revenue Estimate'] = targets_df.loc[twd_mask, 'Low Revenue Estimate'] / usdtwd

targets_df.loc[twd_mask, 'Avg Revenue Estimate'] = targets_df.loc[twd_mask, 'Avg Revenue Estimate'] / usdtwd

targets_df.loc[twd_mask, 'High Revenue Estimate'] = targets_df.loc[twd_mask, 'High Revenue Estimate'] / usdtwd

targets_df.loc[twd_mask, 'marketCap'] = targets_df.loc[twd_mask, 'marketCap'] / usdtwd

targets_df.loc[twd_mask, 'enterpriseValue'] = targets_df.loc[twd_mask, 'enterpriseValue'] / usdtwd

targets_df.loc[twd_mask, 'bookValue'] = targets_df.loc[twd_mask, 'bookValue'] / usdtwd

targets_df.loc[twd_mask, 'Net Income'] = targets_df.loc[twd_mask, 'Net Income'] / usdtwd

targets_df.loc[twd_mask, 'Operating Cash Flow'] = targets_df.loc[twd_mask, 'Operating Cash Flow'] / usdtwd

targets_df.loc[twd_mask, 'Total Assets'] = targets_df.loc[twd_mask, 'Total Assets'] / usdtwd

targets_df.loc[twd_mask, 'Average Assets'] = targets_df.loc[twd_mask, 'Average Assets'] / usdtwd

targets_df.loc[twd_mask, 'Long Term Debt'] = targets_df.loc[twd_mask, 'Long Term Debt'] / usdtwd

targets_df.loc[twd_mask, 'Previous Long Term Debt'] = targets_df.loc[twd_mask, 'Previous Long Term Debt'] / usdtwd

targets_df.loc[twd_mask, 'Current Assets'] = targets_df.loc[twd_mask, 'Current Assets'] / usdtwd

targets_df.loc[twd_mask, 'Current Liabilities'] = targets_df.loc[twd_mask, 'Current Liabilities'] / usdtwd

targets_df.loc[twd_mask, 'Previous Current Liabilities'] = targets_df.loc[twd_mask, 'Previous Current Liabilities'] / usdtwd

targets_df.loc[twd_mask, 'Previous Long Term Debt'] = targets_df.loc[twd_mask, 'Previous Long Term Debt'] / usdtwd

targets_df.loc[twd_mask, 'Total Liabilities'] = targets_df.loc[twd_mask, 'Total Liabilities']  / usdtwd

targets_df.loc[twd_mask, 'Book Value'] = targets_df.loc[twd_mask, 'Book Value'] / usdtwd

targets_df.loc[cny_mask, 'Low Revenue Estimate'] = targets_df.loc[cny_mask, 'Low Revenue Estimate'] / usdcny

targets_df.loc[cny_mask, 'Avg Revenue Estimate'] = targets_df.loc[cny_mask, 'Avg Revenue Estimate'] / usdcny

targets_df.loc[cny_mask, 'High Revenue Estimate'] = targets_df.loc[cny_mask, 'High Revenue Estimate'] / usdcny

targets_df.loc[cny_mask, 'enterpriseValue'] = targets_df.loc[cny_mask, 'enterpriseValue'] / usdcny

targets_df.loc[cny_mask, 'bookValue'] = targets_df.loc[cny_mask, 'bookValue'] / usdcny

targets_df.loc[cny_mask, 'Net Income'] = targets_df.loc[cny_mask, 'Net Income'] / usdcny

targets_df.loc[cny_mask, 'Operating Cash Flow'] = targets_df.loc[cny_mask, 'Operating Cash Flow'] / usdcny

targets_df.loc[cny_mask, 'Total Assets'] = targets_df.loc[cny_mask, 'Total Assets'] / usdcny

targets_df.loc[cny_mask, 'Average Assets'] = targets_df.loc[cny_mask, 'Average Assets'] / usdcny

targets_df.loc[cny_mask, 'Long Term Debt'] = targets_df.loc[cny_mask, 'Long Term Debt'] / usdcny

targets_df.loc[cny_mask, 'Previous Long Term Debt'] = targets_df.loc[cny_mask, 'Previous Long Term Debt'] / usdcny

targets_df.loc[cny_mask, 'Current Assets'] = targets_df.loc[cny_mask, 'Current Assets'] / usdcny

targets_df.loc[cny_mask, 'Current Liabilities'] = targets_df.loc[cny_mask, 'Current Liabilities'] / usdcny

targets_df.loc[cny_mask, 'Previous Current Liabilities'] = targets_df.loc[cny_mask, 'Previous Current Liabilities'] / usdcny

targets_df.loc[cny_mask, 'Previous Long Term Debt'] = targets_df.loc[cny_mask, 'Previous Long Term Debt'] / usdcny

targets_df.loc[cny_mask, 'Total Liabilities'] = targets_df.loc[cny_mask, 'Total Liabilities'] / usdcny

targets_df.loc[cny_mask, 'Book Value'] = targets_df.loc[cny_mask, 'Book Value'] / usdcny

targets_df.loc[cny_mask, 'priceEpsCurrentYear'] = targets_df.loc[cny_mask, 'priceEpsCurrentYear'] / usdcny

targets_df.loc[cny_mask, 'Low EPS Estimate'] = targets_df.loc[cny_mask, 'Low EPS Estimate'] / usdcny

targets_df.loc[cny_mask, 'Avg EPS Estimate'] = targets_df.loc[cny_mask, 'Avg EPS Estimate'] / usdcny

targets_df.loc[cny_mask, 'High EPS Estimate'] = targets_df.loc[cny_mask, 'High EPS Estimate'] / usdcny

targets_df.loc[gbp_big_mask, 'Low Revenue Estimate'] = targets_df.loc[gbp_big_mask, 'Low Revenue Estimate'] / gbpusd

targets_df.loc[gbp_big_mask, 'Avg Revenue Estimate'] = targets_df.loc[gbp_big_mask, 'Avg Revenue Estimate'] / gbpusd

targets_df.loc[gbp_big_mask, 'High Revenue Estimate'] = targets_df.loc[gbp_big_mask, 'High Revenue Estimate'] / gbpusd

targets_df.loc[gbp_big_mask, 'enterpriseValue'] = targets_df.loc[gbp_big_mask, 'enterpriseValue'] / gbpusd

targets_df.loc[gbp_big_mask, 'bookValue'] = targets_df.loc[gbp_big_mask, 'bookValue'] / gbpusd

targets_df.loc[gbp_big_mask, 'Net Income'] = targets_df.loc[gbp_big_mask, 'Net Income'] / gbpusd

targets_df.loc[gbp_big_mask, 'Operating Cash Flow'] = targets_df.loc[gbp_big_mask, 'Operating Cash Flow'] / gbpusd

targets_df.loc[gbp_big_mask, 'Total Assets'] = targets_df.loc[gbp_big_mask, 'Total Assets'] / gbpusd

targets_df.loc[gbp_big_mask, 'Average Assets'] = targets_df.loc[gbp_big_mask, 'Average Assets'] / gbpusd

targets_df.loc[gbp_big_mask, 'Long Term Debt'] = targets_df.loc[gbp_big_mask, 'Long Term Debt'] / gbpusd

targets_df.loc[gbp_big_mask, 'Previous Long Term Debt'] = targets_df.loc[gbp_big_mask, 'Previous Long Term Debt'] / gbpusd

targets_df.loc[gbp_big_mask, 'Current Assets'] = targets_df.loc[gbp_big_mask, 'Current Assets'] / gbpusd

targets_df.loc[gbp_big_mask, 'Current Liabilities'] = targets_df.loc[gbp_big_mask, 'Current Liabilities'] / gbpusd

targets_df.loc[gbp_big_mask, 'Previous Current Liabilities'] = targets_df.loc[gbp_big_mask, 'Previous Current Liabilities'] / gbpusd

targets_df.loc[gbp_big_mask, 'Previous Long Term Debt'] = targets_df.loc[gbp_big_mask, 'Previous Long Term Debt'] / gbpusd

targets_df.loc[gbp_big_mask, 'Total Liabilities'] = targets_df.loc[gbp_big_mask, 'Total Liabilities'] / gbpusd

targets_df.loc[gbp_big_mask, 'Book Value'] = targets_df.loc[gbp_big_mask, 'Book Value'] / gbpusd

targets_df.loc[gbp_big_mask, 'Low EPS Estimate'] = targets_df.loc[gbp_big_mask, 'Low EPS Estimate'] / gbpusd

targets_df.loc[gbp_big_mask, 'Avg EPS Estimate'] = targets_df.loc[gbp_big_mask, 'Avg EPS Estimate'] / gbpusd

targets_df.loc[gbp_big_mask, 'High EPS Estimate'] = targets_df.loc[gbp_big_mask, 'High EPS Estimate'] / gbpusd

targets_df.loc[gbp_big_mask, 'priceEpsCurrentYear'] = targets_df.loc[gbp_big_mask, 'priceEpsCurrentYear'] / gbpusd


targets_df.loc[jp_mask, 'Low Revenue Estimate'] = targets_df.loc[jp_mask, 'Low Revenue Estimate'] / usdjpy

targets_df.loc[jp_mask, 'Avg Revenue Estimate'] = targets_df.loc[jp_mask, 'Avg Revenue Estimate'] / usdjpy

targets_df.loc[jp_mask, 'High Revenue Estimate'] = targets_df.loc[jp_mask, 'High Revenue Estimate'] / usdjpy

targets_df.loc[jp_mask, 'enterpriseValue'] = targets_df.loc[jp_mask, 'enterpriseValue'] / usdjpy

targets_df.loc[jp_mask, 'bookValue'] = targets_df.loc[jp_mask, 'bookValue'] / usdjpy

targets_df.loc[jp_mask, 'Net Income'] = targets_df.loc[jp_mask, 'Net Income'] / usdjpy

targets_df.loc[jp_mask, 'Operating Cash Flow'] = targets_df.loc[jp_mask, 'Operating Cash Flow'] / usdjpy

targets_df.loc[jp_mask, 'Total Assets'] = targets_df.loc[jp_mask, 'Total Assets'] / usdjpy

targets_df.loc[jp_mask, 'Average Assets'] = targets_df.loc[jp_mask, 'Average Assets'] / usdjpy

targets_df.loc[jp_mask, 'Long Term Debt'] = targets_df.loc[jp_mask, 'Long Term Debt'] / usdjpy

targets_df.loc[jp_mask, 'Previous Long Term Debt'] = targets_df.loc[jp_mask, 'Previous Long Term Debt'] / usdjpy

targets_df.loc[jp_mask, 'Current Assets'] = targets_df.loc[jp_mask, 'Current Assets'] / usdjpy

targets_df.loc[jp_mask, 'Current Liabilities'] = targets_df.loc[jp_mask, 'Current Liabilities'] / usdjpy

targets_df.loc[jp_mask, 'Previous Current Liabilities'] = targets_df.loc[jp_mask, 'Previous Current Liabilities'] / usdjpy

targets_df.loc[jp_mask, 'Previous Long Term Debt'] = targets_df.loc[jp_mask, 'Previous Long Term Debt'] / usdjpy

targets_df.loc[jp_mask, 'Total Liabilities'] = targets_df.loc[jp_mask, 'Total Liabilities'] / usdjpy

targets_df.loc[jp_mask, 'Book Value'] = targets_df.loc[jp_mask, 'Book Value'] / usdjpy

targets_df.loc[jp_mask, 'Low EPS Estimate'] = targets_df.loc[jp_mask, 'Low EPS Estimate'] / usdjpy

targets_df.loc[jp_mask, 'Avg EPS Estimate'] = targets_df.loc[jp_mask, 'Avg EPS Estimate'] / usdjpy

targets_df.loc[jp_mask, 'High EPS Estimate'] = targets_df.loc[jp_mask, 'High EPS Estimate'] / usdjpy

targets_df.loc[jp_mask, 'priceEpsCurrentYear'] = targets_df.loc[jp_mask, 'priceEpsCurrentYear'] / usdjpy

targets_df.rename(
    columns = {
        'targetLowPrice': 'Low Price',
        'targetHighPrice': 'High Price',
        'targetMedianPrice': 'Median Price',
        'targetMeanPrice': 'Avg Price'
    },
    inplace = True
)


def compute_z_score_vec(
    n_arr: pd.Series | np.ndarray
) -> np.ndarray:
    """
    Vectorised z-score calculation.
    If n == 2 -> bump to 3 to avoid alpha=0.5.
    Alpha = 1/n.
    z = Φ^{-1}(1 - alpha)
    """
    
    n_arr = np.asarray(n_arr, dtype = float).copy()
    
    n_arr = np.where(n_arr < 2, 2, n_arr)
    
    n_arr = np.where(n_arr == 2, 3, n_arr)
    
    alpha = 1.0 / n_arr
    
    return st.norm.ppf(1.0 - alpha)


def rets_variable_yahoo_vec(
    df: pd.DataFrame, 
    threshold: float = 0.10
)-> pd.DataFrame:
    """
    Vectorised version of rets_variable_yahoo for all tickers at once.
    Expects df to have columns:
    
        ['Avg Price','Median Price','Low Price','High Price', 
         'numberOfAnalystOpinions','Current Price','beta']

    Returns DataFrame with columns:
        ['exp_ret','sigma_ret','beta_out']
    """
   
    meanY = df['Avg Price'].astype(float).to_numpy()
   
    medianY = df['Median Price'].astype(float).to_numpy()
   
    minY = df['Low Price'].astype(float).to_numpy()
   
    maxY = df['High Price'].astype(float).to_numpy()
   
    nY = df['numberOfAnalystOpinions'].astype(float).to_numpy()
   
    price = df['Current Price'].astype(float).to_numpy()
   
    beta = df['beta'].astype(float).to_numpy()

    z = compute_z_score_vec(
        n_arr = nY
    )

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        
        skew_ratio = np.abs(meanY - medianY) / meanY
        
    cond_skew = (meanY > 0) & (skew_ratio > threshold)

    medRetsY = (medianY / price) - 1.0
    
    minRetsY = (minY / price) - 1.0
    
    maxRetsY = (maxY / price) - 1.0

    varY = (((medRetsY - minRetsY) ** 2) + ((maxRetsY - medRetsY) ** 2)) / 12.0
    
    exp_ret_skew = (minRetsY + 2.0 * medRetsY + maxRetsY) / 4.0
    
    sigma_skew = np.sqrt(varY)

    exp_ret_sym = (meanY / price) - 1.0
    
    sigma_sym = (maxY - minY) / (2.0 * z * price)

    exp_ret = np.where(cond_skew, exp_ret_skew, exp_ret_sym)
    
    sigma_ret = np.where(cond_skew, sigma_skew, sigma_sym)

    bad_price = ~np.isfinite(price) | (price == 0)
    
    exp_ret[bad_price] = 0.0
    
    sigma_ret[bad_price] = 0.0

    out = pd.DataFrame({
        "exp_ret": exp_ret,
        "sigma_ret": sigma_ret,
        "beta_out":  beta,
    }, index = df.index)

    return out


latest_prices_series = pd.Series({ticker: close[ticker].iloc[-1] for ticker in tickers})

latest_prices_series = latest_prices_series.reindex(targets_df.index)

latest_prices = latest_prices_series.to_dict()

targets_df['Current Price'] = targets_df.index.map(latest_prices)

logger.info("Computing Analyst Predictions ...")

analyst_calc = rets_variable_yahoo_vec(
    df = targets_df
)

Analyst_Target_df = pd.DataFrame({
    "Current Price": targets_df["Current Price"],
    "Avg Price":  targets_df["Current Price"] * (1.0 + analyst_calc["exp_ret"]),
    "Returns": analyst_calc["exp_ret"],
    "SE": analyst_calc["sigma_ret"],
    "Beta": analyst_calc["beta_out"],
}, index = targets_df.index)

num_cols = Analyst_Target_df.select_dtypes(include = [np.number]).columns

Analyst_Target_df.loc[exemptions, num_cols] = 0.0

Analyst_Target_df.loc[exemptions, "Current Price"] = targets_df.loc[exemptions, "Current Price"]

targets_df_num_cols = targets_df.select_dtypes(include = [np.number]).columns

targets_df.loc[exemptions, targets_df_num_cols] = 0.0

Analyst_Target_df.columns = Analyst_Target_df.columns.astype(str)

logger.info("Uploading Data to Excel ...")

sheets_to_write = {
    "Analyst Data": targets_df,
    "Analyst Target": Analyst_Target_df,
}

export_results(
    sheets = sheets_to_write
)

logger.info("Data has been uploaded to Excel with conditional formatting applied, and all sheets are now tables.")
