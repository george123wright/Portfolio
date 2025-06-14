import pandas as pd
import datetime as dt
from export_forecast import export_results
from ratio_data import RatioData

def get_data():
   
    dcf = pd.read_excel(
        '/Users/georgewright/Portfolio_Optimisation_DCF.xlsx',
        sheet_name = 'DCF',
        index_col=0,
        parse_dates=True,
        engine='openpyxl'
    )
   
    dcfe = pd.read_excel(
        '/Users/georgewright/Portfolio_Optimisation_DCF.xlsx',
        sheet_name = 'DCFE',
        index_col=0,
        parse_dates=True,
        engine='openpyxl'
    )
   
    epv = pd.read_excel(
        '/Users/georgewright/Portfolio_Optimisation_DCF.xlsx',
        sheet_name = 'EPV',
        index_col=0,
        parse_dates=True,
        engine='openpyxl'
    )
    
    ri = pd.read_excel(
        '/Users/georgewright/Portfolio_Optimisation_DCF.xlsx',
        sheet_name = 'RI',
        index_col=0,
        parse_dates=True,
        engine='openpyxl'
    )
    
    lin_reg = pd.read_excel(
        '/Users/georgewright/Portfolio_Optimisation_DCF.xlsx',
        sheet_name = 'Lin Reg Returns',
        index_col=0,
        parse_dates=True,
        engine='openpyxl'
    )
   
    return dcf, dcfe, epv, ri, lin_reg

dcf, dcfe, epv, ri, lin_reg = get_data()

dcf_df = pd.DataFrame(dcf)

dcfe_df = pd.DataFrame(dcfe)

epv_df = pd.DataFrame(epv)

ri_df = pd.DataFrame(ri)

lin_reg_df = pd.DataFrame(lin_reg)

today = dt.date.today() - dt.timedelta(days=1)

yesterday = today - dt.timedelta(days=1)

excel_file = f"/Users/georgewright/Portfolio_Optimisation_Forecast_{today}.xlsx"
excel_file2 = f"/Users/georgewright/Portfolio_Optimisation_Data_{today}.xlsx"

close = pd.read_excel(excel_file2, sheet_name="Close", index_col=0, parse_dates=True, engine="openpyxl").sort_index(ascending=True)

close.columns = close.columns.astype(str)

r = RatioData()

tickers = r.tickers
for t in tickers:
    print(f"Tickers: {t}")

latest_prices = r.last_price

currency = pd.read_excel(
    f'/Users/georgewright/Portfolio_Optimisation_Data_{today}.xlsx',
    sheet_name='Currency',
    index_col=0,
    parse_dates=True,
    engine='openpyxl'
)

usdcny = currency.loc['USDCNY']['Last']

usdcad = currency.loc['USDCAD']['Last']

usdcny = pd.to_numeric(currency.loc['USDCNY']['Last'], errors='coerce')

usdcad = pd.to_numeric(currency.loc['USDCAD']['Last'], errors='coerce')

gbpusd = pd.to_numeric(currency.loc['GBPUSD']['Last'], errors='coerce')

eurusd = pd.to_numeric(currency.loc['EURUSD']['Last'], errors='coerce')

usddkk = pd.to_numeric(currency.loc['USDDKK']['Last'], errors='coerce')

price_cols = ['Low Price', 'Avg Price', 'High Price']

dcf_df[price_cols] = dcf_df[price_cols].apply(pd.to_numeric, errors='coerce')

dcfe_df[price_cols] = dcfe_df[price_cols].apply(pd.to_numeric, errors='coerce')

epv_df[price_cols] = epv_df[price_cols].apply(pd.to_numeric, errors='coerce')

for ticker in tickers:

    dcfe_df.loc[ticker, 'Current Price'] = latest_prices[ticker]

    dcf_df.loc[ticker, 'Current Price'] = latest_prices[ticker] 

    epv_df.loc[ticker, 'Current Price'] = latest_prices[ticker]
    
    ri_df.loc[ticker, 'Current Price'] = latest_prices[ticker]
    
    lin_reg_df.loc[ticker, 'Current Price'] = latest_prices[ticker]

    dcf_df.loc[ticker, 'Returns'] = (dcf_df.loc[ticker, 'Avg Price'] / latest_prices[ticker]) - 1

    dcf_df.loc[ticker, 'SE'] /= latest_prices[ticker]

    dcfe_df.loc[ticker, 'Returns'] = (dcfe_df.loc[ticker, 'Avg Price'] / latest_prices[ticker]) - 1

    dcfe_df.loc[ticker, 'SE'] /= latest_prices[ticker]
    
    ri_df.loc[ticker, 'Current Price'] = latest_prices[ticker]
    
    ri_df.loc[ticker, 'Returns'] = (ri_df.loc[ticker, 'Avg Price'] / latest_prices[ticker]) - 1
    
    ri_df.loc[ticker, 'SE'] /= latest_prices[ticker]

    
dcf_df.columns = dcf_df.columns.astype(str)
dcfe_df.columns = dcfe_df.columns.astype(str)
epv_df.columns = epv_df.columns.astype(str)
ri_df.columns = ri_df.columns.astype(str)
lin_reg_df.columns = lin_reg_df.columns.astype(str)

dcf_df.index = dcf_df.index.astype(str)
dcfe_df.index = dcfe_df.index.astype(str)
epv_df.index = epv_df.index.astype(str)
ri_df.index = ri_df.index.astype(str)
lin_reg_df.index = lin_reg_df.index.astype(str)

for df in (dcf_df, dcfe_df, epv_df, ri_df, lin_reg_df):
    df.fillna(0, inplace=True)

sheets = {
    'DCF': dcf_df,
    'DCFE': dcfe_df,
    'EPV': epv_df,
    'RI': ri_df,
    'Lin Reg Returns': lin_reg_df
}
export_results(sheets)
