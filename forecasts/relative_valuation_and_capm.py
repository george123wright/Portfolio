"""
Runs relative‑valuation models 

- P/E
- P/S
- EV/Sales
- PBV
- Graham, 
- CAPM
- CAPM with Black–Litterman market views

as well as the cost‑of‑equity calculation.
"""

import datetime as dt
import numpy as np
import pandas as pd
import logging
from maps.index_mapping import INDEX_MAPPING
from functions.black_litterman_model import black_litterman
from functions.capm import capm_model
from rel_val.price_to_sales import price_to_sales_price_pred
from rel_val.pe import pe_price_pred
from rel_val.ev import ev_to_sales_price_pred
from rel_val.pbv import price_to_book_pred 
from data_processing.ratio_data import RatioData
from data_processing.financial_forecast_data import FinancialForecastData
from functions.export_forecast import export_results
from rel_val.graham_model import graham_number
from rel_val.relative_valuation import rel_val_model
from maps.currency_mapping import country_to_pair
from functions.coe import calculate_cost_of_equity
from data_processing.macro_data import MacroData
import config


logger = logging.getLogger(__name__)

def run_black_litterman_on_indexes(indr: RatioData,
                                   annual_index_ret: pd.Series,
                                   tau=1,
                                   delta=2.5):

    idx_df = indr.load_index_pred()

    for col in idx_df.columns:

        idx_df[col] = pd.to_numeric(idx_df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')

    mapped_idx = [INDEX_MAPPING.get(name, name) for name in idx_df.index]

    idx_df.index = pd.Index(mapped_idx, name=idx_df.index.name)

    idx_df['Q4_Return'] = (idx_df['Q1/26'] / idx_df['Last']) - 1

    logger.info("Q4/25 implied returns:\n%s", idx_df['Q4_Return'])

    common_idx = idx_df.index.intersection(annual_index_ret.index)

    if len(common_idx) == 0:

        logger.warning("No common indexes between predictions and annual market returns. BL cannot proceed.")

        return None, None

    idx_df = idx_df.loc[common_idx]

    prior = annual_index_ret.loc[common_idx]

    _, weekly_returns = indr.index_returns()

    common_cov_idx = idx_df.index.intersection(weekly_returns.columns)

    if len(common_cov_idx) == 0:

        logger.warning("No common indexes for covariance. Falling back to diagonal covariance.")

        n_idx = len(idx_df)

        sigma_prior = pd.DataFrame(np.diag([0.05] * n_idx), index=idx_df.index, columns=idx_df.index)

    else:

        sigma_prior = weekly_returns[common_cov_idx].cov() * 52

    sigma_prior_diag = pd.DataFrame(np.diag(np.diag(sigma_prior.values)),
                                    index=sigma_prior.index,
                                    columns=sigma_prior.columns)

    n_idx = len(idx_df)

    w_prior = pd.Series(1.0 / n_idx, index=idx_df.index, name='w_prior')

    P = pd.DataFrame(np.eye(n_idx), index=idx_df.index, columns=idx_df.index)

    Q = idx_df['Q4_Return']

    mu_bl, sigma_bl = black_litterman(
        w_prior=w_prior,
        sigma_prior=sigma_prior_diag,
        p=P,
        q=Q,
        omega=None,
        delta=delta,
        tau=tau,
        prior=prior
    )

    bl_df = pd.DataFrame({
        'Prior Market Return': prior,
        'View Return (Q4)': Q,
        'Posterior Market Return': mu_bl
    })
    
    bl_df.index.name = 'Index'

    logger.info("Black–Litterman posterior market returns:\n%s", bl_df)

    return bl_df, sigma_bl


def blend_fx_returns(hist_series, annual_forecast, weight=0.5):
    """
    Return a weighted average of historical and forecast FX returns.
    """
    
    hist_series.index = hist_series.index.str.replace(r'=X$', '', regex=True)

    hist, fc = hist_series.align(annual_forecast, join='inner')
    blended = (weight * hist) + ((1 - weight) * fc)
    
    return blended


def main():
    
    s5 = np.sqrt(5)
    s52 = np.sqrt(52)

    logger.info("Importing Data from Excel")
    
    macro = MacroData()

    r = RatioData()
    
    crp = r.crp()
    
    print("CRP", crp)
            
    fdata = FinancialForecastData()
    
    overall_ann_rets, overall_weekly_rets = r.index_returns()
    
    bl_df, bl_cov = run_black_litterman_on_indexes(r, overall_ann_rets)     
    
    print('bl_df', bl_df)

    tickers = r.tickers
    
    weekly_ret = r.weekly_rets
    
    temp_analyst = r.analyst

    latest_prices = r.last_price

    stock_exchange = temp_analyst['fullExchangeName']
    
    enterprise_value = temp_analyst['enterpriseValue']
    
    marketCap = temp_analyst['marketCap']
    
    country = temp_analyst['country']
    
    mc_ev = enterprise_value / marketCap
    
    bl_market_dict = bl_df['Posterior Market Return'] if bl_df is not None else None
    
    capm_bl_data = []
    capm_hist = []

    beta = temp_analyst['beta']

    for ticker in tickers:
        
        exchange = stock_exchange.loc[ticker]

        matched_index_ret_bl, matched_index_weekly_bl = r.match_index_rets(
            exchange,
            overall_ann_rets,
            overall_weekly_rets,
            bl_market_returns=bl_market_dict
        )
        
        matched_index_ret, matched_index_weekly = r.match_index_rets(
            exchange,
            overall_ann_rets,
            overall_weekly_rets
        )

        market_vol = np.sqrt(matched_index_weekly_bl.var())

        b_stock = beta.get(ticker, 1.0)

        vol_capm_bl, ret_capm_bl = capm_model(b_stock, market_vol, config.RF, matched_index_ret_bl, weekly_ret[ticker], matched_index_weekly_bl)
                
        vol_capm_hist, ret_capm_hist = capm_model(b_stock, market_vol, config.RF, matched_index_ret, weekly_ret[ticker], matched_index_weekly)
                
        cur_price = latest_prices.get(ticker, np.nan)

        capm_bl_data.append({
            "Ticker": ticker,
            "Current Price": cur_price,
            "Avg Price": cur_price * (1 + ret_capm_bl) if not pd.isna(cur_price) else np.nan,
            "Returns": ret_capm_bl,
            "Daily Volatility": vol_capm_bl / s5,
            "SE": vol_capm_bl * s52
        })
        
        capm_hist.append({
            "Ticker": ticker,
            "Current Price": cur_price,
            "Avg Price": cur_price * (1 + ret_capm_hist) if not pd.isna(cur_price) else np.nan,
            "Returns": ret_capm_hist,
            "Daily Volatility": vol_capm_hist / s5,
            "SE": vol_capm_hist * s52
        })
        
    hist_ann_fx, fx_rets = r.get_currency_annual_returns(country_to_pair)
    fx_fc = macro.convert_to_gbp_rates(current_col='Last', future_col='Q1/26')
    pred_fx_growth = fx_fc['Pred Change (%)']
        
    pred_cur_growth = blend_fx_returns(
        hist_series=hist_ann_fx,
        annual_forecast=pred_fx_growth,
        weight=0.5
    )
    
    spx_ret = bl_df.at['^GSPC', 'Posterior Market Return']

    capm_bl_pred_df = pd.DataFrame(capm_bl_data).set_index("Ticker")
    capm_hist_df = pd.DataFrame(capm_hist).set_index("Ticker")
    
    coe_df = calculate_cost_of_equity(
        tickers=tickers,
        rf=config.RF,
        beta_series=beta,
        spx_expected_return=spx_ret,
        crp_df=crp,
        currency_bl_df=pred_cur_growth,
        country_to_pair=country_to_pair,
        ticker_country_map=country,        
    )
    
    low_rev_y = temp_analyst['Low Revenue Estimate']
    avg_rev_y = temp_analyst['Avg Revenue Estimate']
    high_rev_y = temp_analyst['High Revenue Estimate']

    low_eps_y = temp_analyst['Low EPS Estimate']
    avg_eps_y = temp_analyst['Avg EPS Estimate']
    high_eps_y = temp_analyst['High EPS Estimate']
   
    ps = temp_analyst['priceToSalesTrailing12Months']
    cpe = temp_analyst['priceEpsCurrentYear']
    tpe = temp_analyst['trailingPE']
    evts = temp_analyst['enterpriseToRevenue'].copy()
    
    shares_outstanding = temp_analyst['sharesOutstanding']
    dps = temp_analyst['lastDividendValue']
    ptb_y = temp_analyst['priceToBook']
    
    results = r.dicts()

    pe_pred_list = []
    evs_pred_list = []
    ps_pred_list = []
    pbv_pred_list = []
    graham_pred_list = []
    rel_val_list = []

    for ticker in tickers:
                
        forecast_df = fdata.forecast[ticker][['low_eps', 'avg_eps', 'high_eps', 'low_rev', 'avg_rev', 'high_rev']].iloc[0]
                
        kpis = fdata.kpis[ticker][["exp_pe", "exp_ps", "exp_ptb", "exp_evs", "bvps_0"]].iloc[0]
                
        r_pe = pe_price_pred(
            forecast_df['low_eps'],
            forecast_df['avg_eps'],
            forecast_df['high_eps'],
            low_eps_y.get(ticker, np.nan),
            avg_eps_y.get(ticker, np.nan),
            high_eps_y.get(ticker, np.nan),
            cpe.get(ticker, np.nan),
            tpe.get(ticker, np.nan),
            results['PE'][ticker],
            kpis['exp_pe'],
            latest_prices.get(ticker, 0)
        )
     
        pe_pred_list.append({
            "Ticker": ticker,
            "Current Price": latest_prices.get(ticker, 0),
            "Low Price": r_pe[0],
            "Avg Price": r_pe[1],
            "High Price": r_pe[2],
            "Returns": r_pe[3],
            "Volatility": r_pe[4],
            "Avg PE": r_pe[5]
        })
    
        r_evs = ev_to_sales_price_pred(
            latest_prices.get(ticker, 0),
            forecast_df['low_rev'],
            forecast_df['avg_rev'],
            forecast_df['high_rev'],
            low_rev_y.get(ticker, np.nan),
            avg_rev_y.get(ticker, np.nan),
            high_rev_y.get(ticker, np.nan),
            shares_outstanding.get(ticker, 0),
            evts[ticker],
            kpis['exp_evs'], 
            results['EVS'][ticker],
            mc_ev.get(ticker, 1),
        )
    
        evs_pred_list.append({
            "Ticker": ticker,
            "Current Price": latest_prices.get(ticker, 0),
            "Low Price": r_evs[0],
            "Avg Price": r_evs[1],
            "High Price": r_evs[2],
            "Returns": r_evs[3],
            "Volatility": r_evs[4],
            "Avg EVS": r_evs[5]
        })
    
        r_ps = price_to_sales_price_pred(
            latest_prices.get(ticker, 0),
            low_rev_y.get(ticker, np.nan),
            avg_rev_y.get(ticker, np.nan),
            high_rev_y.get(ticker, np.nan),
            forecast_df['low_rev'],
            forecast_df['avg_rev'],
            forecast_df['high_rev'],
            shares_outstanding.get(ticker, 0),
            ps[ticker],
            kpis['exp_ps'],
            results['PS'][ticker],
        )
    
        ps_pred_list.append({
            "Ticker": ticker,
            "Current Price": latest_prices.get(ticker, 0),
            "Low Price": r_ps[0],
            "Avg Price": r_ps[1],
            "High Price": r_ps[2],
            "Returns": r_ps[3],
            "Volatility": r_ps[4],
            "Avg PS": r_ps[5]
        })

        r_pbv =  price_to_book_pred(
            forecast_df['low_eps'],
            forecast_df['avg_eps'],
            forecast_df['high_eps'],
            low_eps_y.get(ticker, np.nan),
            avg_eps_y.get(ticker, np.nan),
            high_eps_y.get(ticker, np.nan),
            ptb_y[ticker],
            kpis['exp_ptb'],
            results['PB'][ticker],
            kpis['bvps_0'],
            dps.get(ticker, 0),
            latest_prices.get(ticker, 0)
        )
      
        pbv_pred_list.append({
            "Ticker": ticker,
            "Current Price": latest_prices.get(ticker, 0),
            "Low Price": r_pbv[0],
            "Avg Price": r_pbv[1],
            "High Price": r_pbv[2],
            "Returns": r_pbv[3],
            "Volatility": r_pbv[4],
            "Avg PBV": r_pbv[5]
        })
        
        r_graham = graham_number(
            results['PE'][ticker],
            forecast_df['low_eps'],
            forecast_df['avg_eps'],
            forecast_df['high_eps'],
            latest_prices.get(ticker, 0),
            results['PB'][ticker],
            kpis['bvps_0'],
            dps.get(ticker, 0),
            low_eps_y.get(ticker, np.nan),
            avg_eps_y.get(ticker, np.nan),
            high_eps_y.get(ticker, np.nan)
        )
        
        graham_pred_list.append({
            "Ticker": ticker,
            "Current Price": capm_bl_pred_df.loc[ticker, "Current Price"],
            "Low Price": r_graham[0],
            "Avg Price": r_graham[1],
            "High Price": r_graham[2],
            "Returns": r_graham[3],
            "Volatility": r_graham[4]
        })
        
        r_rel_val = rel_val_model(
            forecast_df['low_eps'],
            forecast_df['avg_eps'],
            forecast_df['high_eps'],
            low_eps_y.get(ticker, np.nan),
            avg_eps_y.get(ticker, np.nan),
            high_eps_y.get(ticker, np.nan),
            forecast_df['low_rev'],
            forecast_df['avg_rev'],
            forecast_df['high_rev'],
            low_rev_y.get(ticker, np.nan),
            avg_rev_y.get(ticker, np.nan),
            high_rev_y.get(ticker, np.nan),
            cpe.get(ticker, np.nan),
            tpe.get(ticker, np.nan),
            results['PE'][ticker],
            kpis['exp_pe'],
            ps[ticker],
            kpis['exp_ps'],
            results['PS'][ticker],
            ptb_y[ticker],
            kpis['exp_ptb'],
            results['PB'][ticker],
            evts[ticker],
            kpis['exp_evs'], 
            results['EVS'][ticker],
            mc_ev.get(ticker, 1),
            kpis['bvps_0'],
            dps.get(ticker, 0),
            shares_outstanding.get(ticker, 0),
            latest_prices.get(ticker, 0)
        )
        
        rel_val_list.append({
            "Ticker": ticker,
            "Current Price": latest_prices.get(ticker, 0),
            "Low Price": r_rel_val[0],
            "Avg Price": r_rel_val[1],
            "High Price": r_rel_val[2],
            "Returns": r_rel_val[3],
            "SE": r_rel_val[4]
        })
    
    pe_pred_df = pd.DataFrame(pe_pred_list).set_index("Ticker")
    evs_pred_df = pd.DataFrame(evs_pred_list).set_index("Ticker")
    ps_pred_df = pd.DataFrame(ps_pred_list).set_index("Ticker")
    pbv_pred_df = pd.DataFrame(pbv_pred_list).set_index("Ticker")
    graham_pred_df = pd.DataFrame(graham_pred_list).set_index("Ticker")
    rel_val_pred_df = pd.DataFrame(rel_val_list).set_index("Ticker")
    
    sheets_to_write = {
        'BL Index Preds': bl_df,
        'CAPM Pred': capm_hist_df,
        'CAPM BL Pred': capm_bl_pred_df,
        'PS Price Pred': ps_pred_df,
        'EVS Price Pred': evs_pred_df,
        'PE Pred': pe_pred_df,
        'PBV Pred': pbv_pred_df,
        'Graham Pred': graham_pred_df,
        'Rel Val Pred': rel_val_pred_df,
        'COE': coe_df,
    }
    
    export_results(sheets_to_write)


if __name__ == "__main__":
    main()
