"""
Runs relative‑valuation models 

- P/E
- P/S
- EV/Sales
- PBV
- Graham, 
- CAPM
- CAPM with Black–Litterman market views
- FF3
- FF5

as well as the cost‑of‑equity calculation.
"""

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
from data_processing.macro_data3 import MacroData
from fetch_data.factor_data import load_factor_data
from functions.factor_simulations import factor_sim
from functions.fama_french_5_pred import ff5_pred
from functions.fama_french_3_pred import ff3_pred
import config

logger = logging.getLogger(__name__)


def run_black_litterman_on_indexes(
    annual_ret: pd.Series,
    hist_ret: pd.DataFrame,
    future_q_rets: pd.DataFrame,
    tau: float = None,
    delta: float = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs Black–Litterman for each forecast quarter:
      - Uses historical covariance (annualized)
      - Uses identity P for independent views
      - Prior (pi) = quarterlyized annual returns
      - Default tau and delta if not supplied
    Returns: bl_df (posterior returns) and last sigma_bl
    """

    assets = annual_ret.index.intersection(future_q_rets.index)

    pi_ann = annual_ret.loc[assets]

    pi_q = (1 + pi_ann) ** (1/4) - 1
    
    cov_hist = hist_ret[assets].cov()
  
    sigma_prior = cov_hist * 52/4
        
    k = len(assets)
    
    if delta is None:

        w_eq = pd.Series(1.0 / k, index = assets)
        
        delta = float(pi_q.dot(w_eq) / (w_eq.T.dot(sigma_prior).dot(w_eq)))

    if tau is None:
        
        tau = 1.0 / (len(hist_ret) - k - 1)

    w_prior = pd.Series(1.0 / len(assets), index=assets)

    bl_post = {}
    sigma_bl = None
    
    common = future_q_rets.index.intersection(annual_ret.index)
    
    future_q_rets = future_q_rets.loc[common]
            
    P = pd.DataFrame(np.eye(len(assets)), index = assets, columns = assets)

    for col in future_q_rets.columns:
        
        Q = future_q_rets[col].loc[assets]

        mu, sigma_bl = black_litterman(
            w_prior = w_prior,
            sigma_prior = sigma_prior,
            p = P,
            q = Q,
            omega = None,
            delta = delta,
            tau = tau,
            prior = pi_q, 
            confidence = 0.1
        )
       
        bl_post[col] = mu

    bl_df = pd.DataFrame(bl_post)

    bl_df['Ann'] = (1 + bl_df).prod(axis=1) - 1

    return bl_df, sigma_bl


def blend_fx_returns(
    hist_series: pd.Series,
    annual_forecast: pd.Series,
    weight: float = 0.5
) -> pd.Series:
  
    hist_series.index = hist_series.index.str.replace(r'=X$', '', regex=True)
  
    h, f = hist_series.align(annual_forecast, join='inner')
  
    return weight * h + (1 - weight) * f


def main():
   
    s5  = np.sqrt(5)
    s52 = np.sqrt(52)

    logger.info("Importing data…")
   
    macro = MacroData()
   
    r = RatioData()
   
    crp = r.crp()
   
    print("CRP:\n", crp)

    fdata = FinancialForecastData()

    overall_ann_rets, overall_weekly_rets, overall_quarter_rets = r.index_returns()

    idx_levels = r.load_index_pred()
   
    for col in idx_levels.columns:
   
        idx_levels[col] = pd.to_numeric(
            idx_levels[col].astype(str).str.replace(',', '', regex = False),
            errors = 'coerce'
        )
        
    idx_levels.index = [INDEX_MAPPING.get(i, i) for i in idx_levels.index]

    future_q_rets = (
        idx_levels
        .div(idx_levels.shift(axis=1))
        .sub(1)
        .iloc[:, 1:]
    )
    
    future_q_rets.columns = [f"Q{i+1}" for i in range(future_q_rets.shape[1])]

    bl_df, bl_cov = run_black_litterman_on_indexes(
        annual_ret = overall_ann_rets,
        hist_ret = overall_weekly_rets,
        future_q_rets = future_q_rets,
        tau = None,
        delta = None
    )
    
    print("BL posterior:\n", bl_df)
    
    print("BL covariance:\n", bl_cov)

    tickers = r.tickers
    
    weekly_ret = r.weekly_rets
    
    temp_analyst = r.analyst
    
    latest_prices = r.last_price
    
    stock_exchange = temp_analyst['fullExchangeName']
    
    enterprise_val = temp_analyst['enterpriseValue']
    
    market_cap = temp_analyst['marketCap']
    
    country = temp_analyst['country']
    
    mc_ev = enterprise_val / market_cap

    bl_market_dict = bl_df['Ann']

    capm_bl_list = []
    capm_hist_list = []
    
    beta = temp_analyst['beta']

    for ticker in tickers:
        
        exch = stock_exchange.loc[ticker]

        ann_bl, weekly_bl = r.match_index_rets(
            exchange = exch,
            index_rets = overall_ann_rets,
            index_weekly_rets = overall_weekly_rets,
            index_quarter_rets = overall_quarter_rets,
            bl_market_returns = bl_market_dict,
            freq = "annual"
        )

        ann_hist, weekly_hist = r.match_index_rets(
            exchange = exch,
            index_rets = overall_ann_rets,
            index_weekly_rets = overall_weekly_rets,
            index_quarter_rets = overall_quarter_rets
        )

        vol_market = np.sqrt(weekly_bl.var())
       
        b_stock = beta.get(ticker, 1.0)

        vol_bl, ret_bl = capm_model(
            beta_stock = b_stock, 
            market_volatility = vol_market, 
            risk_free_rate = config.RF,
            market_return = ann_bl, 
            weekly_ret = weekly_ret[ticker], 
            index_weekly_ret = weekly_bl
        )
        
        vol_hist, ret_hist = capm_model(
            beta_stock = b_stock, 
            market_volatility = vol_market, 
            risk_free_rate = config.RF,
            market_return = ann_hist, 
            weekly_ret = weekly_ret[ticker], 
            index_weekly_ret = weekly_hist
        )

        price = latest_prices.get(ticker, np.nan)
        
        capm_bl_list.append({
            "Ticker": ticker,
            "Current Price": price,
            "Avg Price": price * (1 + ret_bl) if not pd.isna(price) else np.nan,
            "Returns": ret_bl,
            "Daily Volatility": vol_bl / s5,
            "SE": vol_bl * s52
        })
        
        capm_hist_list.append({
            "Ticker": ticker,
            "Current Price": price,
            "Avg Price": price * (1 + ret_hist) if not pd.isna(price) else np.nan,
            "Returns": ret_hist,
            "Daily Volatility": vol_hist / s5,
            "SE": vol_hist * s52
        })
        
    capm_bl_pred_df = pd.DataFrame(capm_bl_list).set_index("Ticker")
   
    capm_hist_df = pd.DataFrame(capm_hist_list).set_index("Ticker")

    hist_ann_fx, fx_rets = r.get_currency_annual_returns(
        country_to_pair = country_to_pair
    )
   
    fx_fc = macro.convert_to_gbp_rates(
        current_col = 'Last', 
        future_col = 'Q1/26'
    )
   
    pred_fx_growth = fx_fc['Pred Change (%)']
   
    blended_fx = blend_fx_returns(
        hist_series = hist_ann_fx, 
        annual_forecast = pred_fx_growth
    )

    spx_ret = bl_df.loc['^GSPC']
    
    spx_ret_series = bl_df.loc['^GSPC', ['Q1','Q2','Q3','Q4']]

    index_close = r.index_close['^GSPC'].sort_index()
    
    coe_df = calculate_cost_of_equity(
        tickers = tickers,
        rf = config.RF,
        returns = weekly_ret,
        index_close = index_close,
        spx_expected_return = spx_ret['Ann'],
        crp_df = crp,
        currency_bl_df = blended_fx,
        country_to_pair = country_to_pair,
        ticker_country_map = country
    )

    ff5_m, ff3_m, ff5_q, ff3_q = load_factor_data()
    
    cov5, E5_q, sims_5 = factor_sim(
        factor_data = ff5_q, 
        num_factors = 5, 
        n_sims = 1_000, 
        horizon = 4
    )

    cov3, E3_q, sims_3 = factor_sim(
        factor_data = ff3_q, 
        num_factors = 3, 
        n_sims = 1_000, 
        horizon = 4
    )
        
    ff5_results = ff5_pred(
        tickers = tickers,
        factor_data = ff5_q,
        weekly_ret = r.quarterly_rets,
        Cov_tv = cov5,
        E_factors_12m = E5_q,
        E_mkt_ret = spx_ret_series,
        sims = sims_5,
        rf = config.RF
    )
    
    ff3_results = ff3_pred(
        tickers = tickers,
        factor_data = ff3_q,
        weekly_ret = r.quarterly_rets,
        Cov_tv = cov3,
        E_factors_12m = E3_q,
        E_mkt_ret = spx_ret_series,
        sims = sims_3,
        rf = config.RF
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

    shares_out = temp_analyst['sharesOutstanding']
    
    dps = temp_analyst['lastDividendValue']
    
    ptb_y = temp_analyst['priceToBook']

    results = r.dicts()

    pe_pred_list, evs_pred_list, ps_pred_list, pbv_pred_list, graham_pred_list, rel_val_list = ([] for _ in range(6))

    for ticker in tickers:
       
        forecast_df = (
            fdata.forecast[ticker]
            [['low_eps', 'avg_eps', 'high_eps', 'low_rev', 'avg_rev', 'high_rev']]
            .iloc[0]
        )
        
        kpis = (
            fdata.kpis[ticker]
            [["exp_pe", "exp_ps", "exp_ptb", "exp_evs", "bvps_0"]]
            .iloc[0]
        )
        
        r_pe = pe_price_pred(
            eps_low = forecast_df['low_eps'],
            eps_avg = forecast_df['avg_eps'],
            eps_high = forecast_df['high_eps'],
            eps_low_y = low_eps_y.get(ticker, np.nan),
            eps_avg_y = avg_eps_y.get(ticker, np.nan),
            eps_high_y = high_eps_y.get(ticker, np.nan),
            pe_c = cpe.get(ticker, np.nan),
            pe_t = tpe.get(ticker, np.nan),
            pe_ind = results['PE'][ticker],
            avg_pe_fs = kpis['exp_pe'],
            price = latest_prices.get(ticker, 0)
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
            price = latest_prices.get(ticker, 0),
            low_rev = forecast_df['low_rev'],
            avg_rev = forecast_df['avg_rev'],
            high_rev = forecast_df['high_rev'],
            low_rev_y = low_rev_y.get(ticker, np.nan),
            avg_rev_y = avg_rev_y.get(ticker, np.nan),
            high_rev_y = high_rev_y.get(ticker, np.nan),
            shares_outstanding = shares_out.get(ticker, 0),
            evs = evts[ticker],
            avg_fs_ev = kpis['exp_evs'], 
            ind_evs = results['EVS'][ticker],
            mc_ev = mc_ev.get(ticker, 1),
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
            price = latest_prices.get(ticker, 0),
            low_rev_y = low_rev_y.get(ticker, np.nan),
            avg_rev_y = avg_rev_y.get(ticker, np.nan),
            high_rev_y = high_rev_y.get(ticker, np.nan),
            low_rev = forecast_df['low_rev'],
            avg_rev = forecast_df['avg_rev'],
            high_rev = forecast_df['high_rev'],
            shares_outstanding = shares_out.get(ticker, 0),
            ps = ps[ticker],
            avg_ps_fs = kpis['exp_ps'],
            ind_ps = results['PS'][ticker],
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

        r_pbv = price_to_book_pred(
            low_eps = forecast_df['low_eps'],
            avg_eps = forecast_df['avg_eps'],
            high_eps = forecast_df['high_eps'],
            low_eps_y = low_eps_y.get(ticker, np.nan),
            avg_eps_y = avg_eps_y.get(ticker, np.nan),
            high_eps_y = high_eps_y.get(ticker, np.nan),
            ptb = ptb_y[ticker],
            avg_ptb_fs = kpis['exp_ptb'],
            ptb_ind = results['PB'][ticker],
            book_fs = kpis['bvps_0'],
            dps = dps.get(ticker, 0),
            price = latest_prices.get(ticker, 0)
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
            pe_ind = results['PE'][ticker],
            eps_low = forecast_df['low_eps'],
            eps_avg = forecast_df['avg_eps'],
            eps_high = forecast_df['high_eps'],
            price = latest_prices.get(ticker, 0),
            pb_ind = results['PB'][ticker],
            bvps_0 = kpis['bvps_0'],
            dps = dps.get(ticker, 0),
            low_eps_y = low_eps_y.get(ticker, np.nan),
            avg_eps_y = avg_eps_y.get(ticker, np.nan),
            high_eps_y = high_eps_y.get(ticker, np.nan)
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
            low_eps = forecast_df['low_eps'],
            avg_eps = forecast_df['avg_eps'],
            high_eps = forecast_df['high_eps'],
            low_eps_y = low_eps_y.get(ticker, np.nan),
            avg_eps_y = avg_eps_y.get(ticker, np.nan),
            high_eps_y = high_eps_y.get(ticker, np.nan),
            low_rev = forecast_df['low_rev'],
            avg_rev = forecast_df['avg_rev'],
            high_rev = forecast_df['high_rev'],
            low_rev_y = low_rev_y.get(ticker, np.nan),
            avg_rev_y = avg_rev_y.get(ticker, np.nan),
            high_rev_y = high_rev_y.get(ticker, np.nan),
            pe_c = cpe.get(ticker, np.nan),
            pe_t = tpe.get(ticker, np.nan),
            pe_ind = results['PE'][ticker],
            avg_pe_fs = kpis['exp_pe'],
            ps = ps[ticker],
            avg_ps_fs = kpis['exp_ps'],
            ind_ps = results['PS'][ticker],
            ptb = ptb_y[ticker],
            avg_ptb_fs = kpis['exp_ptb'],
            ptb_ind = results['PB'][ticker],
            evs = evts[ticker],
            avg_fs_ev = kpis['exp_evs'], 
            ind_evs = results['EVS'][ticker],
            mc_ev = mc_ev.get(ticker, 1),
            bvps_0 = kpis['bvps_0'],
            dps = dps.get(ticker, 0),
            shares_outstanding = shares_out.get(ticker, 0),
            price = latest_prices.get(ticker, 0)
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
        
    rel_val_sheets = {
        'BL Index Preds': bl_df,
        'CAPM BL Pred': capm_hist_df,
        'PS Price Pred': ps_pred_df,
        'EVS Price Pred': evs_pred_df,
        'PE Pred': pe_pred_df,
        'PBV Pred': pbv_pred_df,
        'Graham Pred': graham_pred_df,
    }
    export_results(
        sheets = rel_val_sheets, 
        output_excel_file = config.REL_VAL_FILE
    )

    sheets_to_write = {
        'CAPM BL Pred': capm_bl_pred_df,
        'Rel Val Pred': rel_val_pred_df,
        'COE': coe_df,
        'FF3 Pred': ff3_results,
        'FF5 Pred': ff5_results
    }
    export_results(
        sheets = sheets_to_write
    )


if __name__ == "__main__":
    main()
