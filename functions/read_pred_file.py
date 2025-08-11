"""
Reads previously saved valuation sheets (DCF, DCFE, RI, GBM), updates them with latest market prices and exports formatted results.
"""


import pandas as pd
import datetime as dt
from export_forecast import export_results
from ratio_data import RatioData
import config

tickers = config.tickers

def get_data():
    
    sheet_names = [
        'DCF', 
        'DCFE',
        'RI',
        'Lin Reg Returns',
        'SARIMAX Monte Carlo',
        'Prophet Pred',
        'LSTM'
    ]

    sheets = pd.read_excel(
        config.MODEL_FILE,
        sheet_name = sheet_names,
        engine = 'openpyxl',
        index_col = 0
    )

    dcf = sheets['DCF']
    
    dcfe = sheets['DCFE']
    
    ri = sheets['RI']
    
    lin_reg = sheets['Lin Reg Returns']
    
    sarimax = sheets['SARIMAX Monte Carlo']
    
    prophet_pred = sheets['Prophet Pred']
    
    lstm_pred = sheets['LSTM']
    
    return dcf, dcfe, ri, lin_reg, sarimax, prophet_pred, lstm_pred


dcf, dcfe, ri, lin_reg, sarimax, prophet_pred, lstm_pred = get_data()

dcf_df = pd.DataFrame(dcf)
dcf_df = dcf_df.reindex(tickers)

dcfe_df = pd.DataFrame(dcfe)
dcfe_df = dcfe_df.reindex(tickers)

ri_df = pd.DataFrame(ri)
ri_df = ri_df.reindex(tickers)

lin_reg_df = pd.DataFrame(lin_reg)
lin_reg_df = lin_reg_df.reindex(tickers)

sarimax_df = pd.DataFrame(sarimax)
sarimax_df = sarimax_df.reindex(tickers)

prophet_pred_df = pd.DataFrame(prophet_pred)
prophet_pred_df = prophet_pred_df.reindex(tickers)

lstm_pred_df = pd.DataFrame(lstm_pred)
lstm_pred_df = lstm_pred_df.reindex(tickers)

r = RatioData()

latest_prices = r.last_price

price_cols = ['Low Price', 'Avg Price', 'High Price']

dcf_df[price_cols] = dcf_df[price_cols].apply(pd.to_numeric, errors = 'coerce')

dcfe_df[price_cols] = dcfe_df[price_cols].apply(pd.to_numeric, errors = 'coerce')

ticker_set = set(tickers)

dcf_df = dcf_df[dcf_df.index.isin(ticker_set)]

dcfe_df = dcfe_df[dcfe_df.index.isin(ticker_set)]

ri_df = ri_df[ri_df.index.isin(ticker_set)]

lin_reg_df = lin_reg_df[lin_reg_df.index.isin(ticker_set)]

sarimax_df = sarimax_df[sarimax_df.index.isin(ticker_set)]

prophet_pred_df = prophet_pred_df[prophet_pred_df.index.isin(ticker_set)]

lstm_pred_df = lstm_pred_df[lstm_pred_df.index.isin(ticker_set)]


for ticker in tickers:

    dcfe_df.loc[ticker, 'Current Price'] = latest_prices[ticker]

    dcf_df.loc[ticker, 'Current Price'] = latest_prices[ticker] 
    
    ri_df.loc[ticker, 'Current Price'] = latest_prices[ticker]
    
    lin_reg_df.loc[ticker, 'Current Price'] = latest_prices[ticker]

    dcf_df.loc[ticker, 'Returns'] = (dcf_df.loc[ticker, 'Avg Price'] / latest_prices[ticker]) - 1

    dcf_df.loc[ticker, 'SE'] /= latest_prices[ticker]

    dcfe_df.loc[ticker, 'Returns'] = (dcfe_df.loc[ticker, 'Avg Price'] / latest_prices[ticker]) - 1

    dcfe_df.loc[ticker, 'SE'] /= latest_prices[ticker]
    
    ri_df.loc[ticker, 'Current Price'] = latest_prices[ticker]
    
    ri_df.loc[ticker, 'Returns'] = (ri_df.loc[ticker, 'Avg Price'] / latest_prices[ticker]) - 1
    
    ri_df.loc[ticker, 'SE'] /= latest_prices[ticker]

for df in (dcf_df, dcfe_df, ri_df, lin_reg_df, sarimax_df, prophet_pred_df):
    
    df.drop(   
        index = [idx for idx in df.index if idx not in tickers],
        inplace = True
    )

sarimax_df['Current Price'] = sarimax_df.index.map(latest_prices)

sarimax_df.fillna(0, inplace=True)

dcf_df.columns = dcf_df.columns.astype(str)
dcfe_df.columns = dcfe_df.columns.astype(str)
ri_df.columns = ri_df.columns.astype(str)
lin_reg_df.columns = lin_reg_df.columns.astype(str)
sarimax_df.columns = sarimax_df.columns.astype(str)
prophet_pred_df.columns = prophet_pred_df.columns.astype(str)
lstm_pred_df.columns = lstm_pred_df.columns.astype(str)

dcf_df.index = dcf_df.index.astype(str)
dcfe_df.index = dcfe_df.index.astype(str)
ri_df.index = ri_df.index.astype(str)
lin_reg_df.index = lin_reg_df.index.astype(str)
sarimax_df.index = sarimax_df.index.astype(str)
prophet_pred_df.index = prophet_pred_df.index.astype(str)
lstm_pred_df.index = lstm_pred_df.index.astype(str)

for df in (dcf_df, dcfe_df, ri_df, lin_reg_df, sarimax_df, prophet_pred_df, lstm_pred_df):
    
    df.fillna(0, inplace=True)

sheets = {
    'DCF': dcf_df,
    'DCFE': dcfe_df,
    'RI': ri_df,
    'Lin Reg Returns': lin_reg_df,
    "SARIMAX Monte Carlo": sarimax_df,
    'Prophet Pred': prophet_pred_df,
    'LSTM': lstm_pred_df
}

for df in sheets.values():
   
    mask = (
        (df['SE'] <= 0.02) |
        (df['Returns'] <= -0.8) |
        (df.isna().any(axis=1))
    )
   
    df.loc[mask, :] = 0
    
for df in sheets.values():
   
    df.fillna(0, inplace=True)
    
export_results(
    sheets = sheets
)
