"""
Main script orchestrating portfolio optimisation: loads Excel data, computes bounds, runs optimisers, simulates portfolio performance and writes all outputs.
"""

import datetime as dt
import logging
from typing import Tuple, List
import numpy as np
import pandas as pd
import yfinance as yf
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import PatternFill
import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule 

import portfolio_functions as pf
from functions.cov_functions import shrinkage_covariance
import portfolio_optimisers as po
from data_processing.ratio_data import RatioData
import config


r = RatioData()

tickers = config.tickers

money_in_portfolio = config.MONEY_IN_PORTFOLIO

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s"
)


def ensure_headers_are_strings(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure that all column names and the index name of the DataFrame are strings.
    """
    
    df.columns = [str(col) if col is not None else "" for col in df.columns]
    
    if df.index.name is None:
   
        df.index.name = "Index"
   
    else:
   
        df.index.name = str(df.index.name)
    
    return df


def load_excel_data() -> Tuple[
    pd.DataFrame,
    List[str],
    np.ndarray, 
    pd.DataFrame,  
    pd.DataFrame, 
    pd.Series,    
    pd.Series,
    pd.Series,   
    pd.Series,     
    pd.Series,      
    pd.Series,
    pd.Series,
    pd.Series, 
    pd.Series
]:
    """
    Loads required sheets from Excel, does index alignment, and returns them.
    """
        
    weekly_ret = r.weekly_rets
    daily_ret = r.daily_rets
    monthly_ret = r.close.resample('M').last().pct_change().dropna()
    
    weekly_ret.index = pd.to_datetime(weekly_ret.index)
    weekly_ret.sort_index(ascending=True, inplace=True)
    
    
    weekly_ret.columns = tickers
    
    comb_data = pd.read_excel(config.PORTFOLIO_FILE, sheet_name="Combination Forecast", index_col=0)
    comb_data.index = comb_data.index.str.upper()
    comb_data = comb_data.reindex(tickers)
    
    comb_std = comb_data['SE']
    score = comb_data["Score"]
    
    comb_rets = comb_data["Returns"]
    bear_rets = comb_data["Low Returns"]
    bull_rets = comb_data["High Returns"]
    
    ticker_data = pd.read_excel(config.FORECAST_FILE, sheet_name="Analyst Data", index_col=0)
    
    ticker_ind = ticker_data["Industry"]
    ticker_mcap = ticker_data['marketCap']
    ticker_sec = ticker_data['Sector']
        
    daily_ret_5y = daily_ret.loc[daily_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
    weekly_ret_5y = weekly_ret.loc[weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
    monthly_ret_5y = monthly_ret.loc[monthly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
            
    annCov = shrinkage_covariance(
        daily_5y = daily_ret_5y, 
        weekly_5y = weekly_ret_5y, 
        monthly_5y = monthly_ret_5y, 
        comb_std = comb_std, 
        common_idx = comb_data.index
    )
    
    sigma_prior = shrinkage_covariance(daily_ret_5y, weekly_ret_5y, monthly_ret_5y, comb_std, comb_data.index, delta=1/2, alpha = 0)
    
    annCov = annCov.reindex(index=comb_data.index, columns=comb_data.index)
        
    weeklyCov = annCov / 52
    
    cov_df = pd.DataFrame(annCov, index=comb_data.index, columns=comb_data.index)
    
    Signal = pd.read_excel(config.DATA_FILE, sheet_name="Signal Scores", index_col=0).iloc[-1]
    
    beta_data = pd.read_excel(config.FORECAST_FILE, sheet_name="Analyst Data", index_col=0)["beta"]
    beta_data.index = beta_data.index.str.upper()
    beta = beta_data.reindex(tickers)
    
    return (weekly_ret_5y, tickers, weeklyCov, annCov, cov_df,
            Signal, beta, 
            score, comb_rets, ticker_ind, ticker_sec, comb_std,
            bear_rets, bull_rets, ticker_mcap, sigma_prior)


def get_buy_sell_signals_from_scores(
    tickers: List[str],
    signal_score: pd.Series
) -> Tuple[pd.DataFrame, List[bool], List[bool]]:
    """
    For each ticker, if signal_score > 0 ⇒ buy flag, < 0 ⇒ sell flag.
    """
    
    signal_score.index = signal_score.index.str.upper()

    buy_flags  = [bool(signal_score.get(t, 0) > 0) for t in tickers]
    sell_flags = [bool(signal_score.get(t, 0) < 0) for t in tickers]

    df = pd.DataFrame({
        "Ticker": tickers,
        "Buy":    buy_flags,
        "Sell":   sell_flags,
        "Score":  [signal_score.get(t, 0) for t in tickers],
    })
    
    return df, buy_flags, sell_flags


def _add_table(
    ws, 
    table_name: str
) -> None:
    """
    Convert the used-range of *ws* to an Excel table called *table_name*.
    """
    
    last_col = get_column_letter(ws.max_column)
    
    table = Table(displayName=table_name, ref=f"A1:{last_col}{ws.max_row}")
    
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium9",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    
    ws.add_table(table)


def add_weight_cf(
    ws
) -> None:
    """
    Add the three traffic-light rules (red <0.01, yellow 1.95–2.05, green “valid”)
    to every numeric column (B…last) of the supplied worksheet.
    """
    
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow = PatternFill(start_color="FFEB9B", end_color="FFEB9B", fill_type="solid")

    first_data_col = 2                         
   
    last_data_col = ws.max_column

    for col_idx in range(first_data_col, last_data_col + 1):
   
        col_letter = get_column_letter(col_idx)
        
        rng = f"{col_letter}2:{col_letter}{ws.max_row}"

        ws.conditional_formatting.add(
            rng,
            CellIsRule(operator="lessThan", formula=['0.01'], fill=red, stopIfTrue=True)
        )
        
       
        ws.conditional_formatting.add(
            rng,
            CellIsRule(operator="between", formula=['1.95', '2.05'], fill=yellow, stopIfTrue=True)
        )
       
        ws.conditional_formatting.add(
            rng,
            FormulaRule(formula=[f"AND({col_letter}2<>0,{col_letter}2<>2)"], fill=green)
        )

    for row in ws.iter_rows(min_row=2, min_col=first_data_col, max_col=last_data_col):
        
        for cell in row:
            
            cell.number_format = "0.00"


def write_excel_results(
    excel_file: str, 
    sheets: dict[str, pd.DataFrame]
) -> None:
    """
    Write all result dataframes to *excel_file* and style them.
    Expected keys in *sheets*:
        Today_Buy_Sell, Covariance,
        Weights, Weights_Bear, Weights_Bull,
        Portfolio Performance, Portfolio Performance Bear, Portfolio Performance Bull,
        Industry Breakdown, Industry Breakdown Bear, Industry Breakdown Bull,
        Sector Breakdown, Sector Breakdown Bear, Sector Breakdown Bull
    """

    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    with pd.ExcelWriter(excel_file, engine = "openpyxl", mode = "a", if_sheet_exists = "replace") as writer:
        
        tperf = ensure_headers_are_strings(sheets["Ticker Performance"].copy())
        
        tperf.to_excel(writer, sheet_name="Ticker Performance", index=True)
        
        ws_tp = writer.sheets["Ticker Performance"]

        tbs = ensure_headers_are_strings(sheets["Today_Buy_Sell"].copy())
        
        tbs.to_excel(writer, sheet_name="Today_Buy_Sell", index=False)

        ws = writer.sheets["Today_Buy_Sell"]
        
        n = tbs.shape[0] + 1
        
        ws.conditional_formatting.add(f"B2:B{n}", CellIsRule(operator = "equal", formula = ["TRUE"],  fill = green))
        ws.conditional_formatting.add(f"B2:B{n}", CellIsRule(operator = "equal", formula = ["FALSE"], fill = red))
        ws.conditional_formatting.add(f"C2:C{n}", CellIsRule(operator = "equal", formula = ["TRUE"],  fill = green))
        ws.conditional_formatting.add(f"C2:C{n}", CellIsRule(operator = "equal", formula = ["FALSE"], fill = red))
        
        _add_table(writer.sheets['Ticker Performance'], "TickerPerformanceTable")
        
        _add_table(ws, "TodayBuySellTable")

        cov = ensure_headers_are_strings(sheets["Covariance"].copy())

        bnds = ensure_headers_are_strings(sheets["Bounds"].copy())
        
        bnds.to_excel(writer, sheet_name="Bounds")
        
        bnds_ws = writer.sheets["Bounds"]
        
        _add_table(bnds_ws, "BoundsTable")

        cov.to_excel(writer, sheet_name="Covariance")
        
        cov_ws = writer.sheets["Covariance"]
        
        _add_table(cov_ws, "CovarianceTable")

        data = cov.values.astype(float)

        min_val, mean_val, max_val = data.min(), data.mean(), data.max()
        
        color_rule = ColorScaleRule(
            start_type='num', start_value=str(min_val), start_color='FFFFFF',
            mid_type='num', mid_value=str(mean_val), mid_color='FFDD99',
            end_type='num', end_value=str(max_val), end_color='FF0000'
        )

        rng = f"B2:{get_column_letter(cov_ws.max_column)}{cov_ws.max_row}"
       
        cov_ws.conditional_formatting.add(rng, color_rule)

        def dump_weight_sheet(df_key: str, sheet_name: str, tablename: str) -> None:
       
            df = ensure_headers_are_strings(sheets[df_key].copy())
       
            df.to_excel(writer, sheet_name=sheet_name, index=False)
       
            ws = writer.sheets[sheet_name]
       
            add_weight_cf(ws)
       
            _add_table(ws, tablename)

        dump_weight_sheet("Weights", "Portfolio Weights", "WeightsTable")

        for key, sheet_name, tablename in [
            ("Portfolio Performance", "Portfolio Performance", "PortfolioPerformanceTable"),
            ("Industry Breakdown", "Industry Breakdown", "IndustryBreakdownTable"),
           ("Sector Breakdown", "Sector Breakdown", "SectorBreakdownTable"),
        ]:
       
            df = ensure_headers_are_strings(sheets[key].copy())
            df.to_excel(writer, sheet_name = sheet_name, index = "Industry" not in key)
            _add_table(writer.sheets[sheet_name], tablename)
            

def benchmark_rets(
    benchmark: str, 
    start: dt.date, 
    end: dt.date, 
    steps: int
) -> float:
    """
    Downloads benchmark data, computes returns, and annualises them.
    """
    
    if benchmark.upper() == 'SP500':
        
        close = yf.download('^GSPC', start=start, end=end)['Close'].squeeze()
    
    elif benchmark.upper() == 'NASDAQ':
        
        close = yf.download('^IXIC', start=start, end=end)['Close'].squeeze()
    
    elif benchmark.upper() == 'FTSE':
        
        close = yf.download('^FTSE', start=start, end=end)['Close'].squeeze()
        
    elif benchmark.upper() == 'FTSE ALL-WORLD':
        
        close = yf.download('VWRL.L', start=start, end=end)['Close'].squeeze()
    
    elif benchmark.upper() == 'ALL':
    
        close_sp = yf.download('^GSPC', start=start, end=end)['Close'].squeeze()
        close_nd = yf.download('^IXIC', start=start, end=end)['Close'].squeeze()
        close_ft = yf.download('^FTSE', start=start, end=end)['Close'].squeeze()
    
        rets_sp = close_sp.resample('W').last().pct_change().dropna()
        rets_nd = close_nd.resample('W').last().pct_change().dropna()
        rets_ft = close_ft.resample('W').last().pct_change().dropna()
    
        return float((pf.annualise_returns(rets_sp, steps) +
                      pf.annualise_returns(rets_nd, steps) +
                      pf.annualise_returns(rets_ft, steps)) / 3)
    
    else:
        raise ValueError("Unsupported benchmark")
    
    rets = close.resample('W').last().pct_change().dropna()
    
    last_year_rets = rets[rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    benchmark_ann_ret = (last_year_rets + 1).prod() - 1
    
    return benchmark_ann_ret, rets, last_year_rets 


def compute_mcap_bounds(
    market_cap: pd.Series,
    beta: pd.Series,
    er: pd.Series,
    vol: pd.Series,
    score: pd.Series,
    tickers: pd.Index,
    ticker_sec: pd.Series,
    min_all: float,
    max_all: float,
    max_all_l: float,
    min_all_u: float
):
    """
    Returns two dicts (lb, ub) mapping ticker -> lower / upper bound.
    """
    
    score_max = score.max()
    
    mcap_sqrt = market_cap
    
    sec_sharpe = r.align_sector_sharpe()
    
    ind_sharpe = r.align_ind_sharpe()
    
    mcap_vol = {}
            
    for t in tickers:
    
        if vol.loc[t] > 0 and beta.loc[t] > 0 and score.loc[t] > 0:
            
            sec_sharpe_t = sec_sharpe.loc[t]    
            
            ind_sharpe_t = ind_sharpe.loc[t]
            
            ind_sharpe_val = np.maximum(np.sqrt(np.maximum(ind_sharpe_t + 1, 0)), 0.1)
            
            mcap_vol[t] = (1 + sec_sharpe_t) * ind_sharpe_val * mcap_sqrt.loc[t] / vol.loc[t]
            
        else:
            mcap_vol[t] = 0.0
    
    mcap_vol_beta = pd.Series(mcap_vol)

    mcap_vol_beta = mcap_vol_beta.fillna(0)
    
    mask = (er > 0) & (score > 0)
    
    tot = float(mcap_vol_beta[mask].sum())
    
    if tot == 0:
        
        tot = 1e-12  

    lb = {}
    ub = {}
    
    for t in tickers:
        
        if t in config.TICKER_EXEMPTIONS:
            
            lb[t] = 0.01
            ub[t] = 0.075
    
        elif er.loc[t] > 0 and score.loc[t] > 0:
                            
            norm_score = score.loc[t] / score_max

            frac_l = mcap_vol_beta[t] / tot
            
            cl_l = min(frac_l, max_all_l)
            
            lb[t] = max(min_all, norm_score * cl_l)

            frac_u = np.sqrt(mcap_vol_beta[t] / tot)
            
            cl_u = min(frac_u, max_all)
            
            if ticker_sec.loc[t] == 'Healthcare' or ticker_sec.loc[t] == 'Consumer Staples':
            
                ub[t] = (norm_score * cl_u).clip(min_all_u, 0.025)
            
                ub_val = min(norm_score * cl_u, 0.025)
                
                lb[t] = max(norm_score * cl_l / 2, min_all)
            
                ub[t] = max(ub_val, lb[t])
            
            else:
            
                ub[t] = (norm_score * cl_u).clip(min_all_u, max_all)
    
        else:
    
            lb[t] = 0.0
            ub[t] = 0.0

    return pd.Series(lb), pd.Series(ub), mcap_vol_beta


def main() -> None:
   
    logging.info("Starting portfolio optimisation script...")
        
    benchmark_ret, benchmark_weekly_rets, last_year_benchmark_weekly_rets = benchmark_rets(
        benchmark = config.benchmark, 
        start = config.FIVE_YEAR_AGO, 
        end = config.TODAY, 
        steps = 52
    )

    logging.info("Loading Excel data...")
  
    (weekly_ret, tickers, weekly_cov, ann_cov, cov_df, signal_score, beta, comb_score, comb_rets, ticker_ind, ticker_sec, comb_ann_std, bear_rets, bull_rets, mcap, sigma_prior) = load_excel_data()
    
    print(comb_rets)
    print(cov_df)
    
    assert comb_rets.index.equals(pd.Index(tickers)), "comb_rets index ≠ tickers"
    
    assert cov_df.index.equals(pd.Index(tickers)), "cov_df index ≠ tickers"
    
    assert cov_df.columns.equals(pd.Index(tickers)), "cov_df columns ≠ tickers"
    
    assert comb_rets.isna().sum() == 0, "Some tickers have no forecasted return!"

    buy_sell_today_df, buy_flags, sell_flags = get_buy_sell_signals_from_scores(
        tickers = tickers, 
        signal_score = signal_score
    )
    
    logging.info("Data loaded and signals created.")
     
    rf_rate = config.RF
    
    w_max = config.MAX_WEIGHT
    
    w_min = 2 / money_in_portfolio
    
    max_all_l = w_max / 2
    
    min_all_u = w_min * 2
    
    mcap_bnd_l, mcap_bnd_h, mcap_vol_beta = compute_mcap_bounds(
        market_cap = mcap, 
        beta = beta, 
        er = comb_rets, 
        vol = comb_ann_std, 
        score = comb_score,
        tickers = tickers, 
        ticker_sec = ticker_sec,
        min_all = w_min, 
        max_all = w_max, 
        max_all_l = max_all_l, 
        min_all_u = min_all_u
    )
    
    mcap["SGLP.L"] = mcap.max()
    
    bounds_df = pd.DataFrame({
        'mid': mcap_vol_beta,
        'Low': mcap_bnd_l,
        'High': mcap_bnd_h
    })
    
    last_year_weekly_rets = weekly_ret.loc[weekly_ret.index >= pd.to_datetime(config.YEAR_AGO)]   
    
    last_5_year_weekly_rets = weekly_ret.loc[weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]         
    
    logging.info("Optimising MSR with constraints...")
    
    w_msr = po.msr(
        riskfree_rate = rf_rate, 
        er = comb_rets, 
        cov = ann_cov, 
        scores = comb_score, 
        ticker_ind = ticker_ind, 
        ticker_sec = ticker_sec, 
        bnd_h = mcap_bnd_h, 
        bnd_l = mcap_bnd_l
    )
    
    print("MSR Weights:", w_msr)
    
    print(comb_rets.loc['NVDA'])
    
    vol_msr_ann = pf.portfolio_volatility(
        weights = w_msr, 
        covmat = ann_cov
    )
    
    vol_msr = pf.portfolio_volatility(
        weights = w_msr, 
        covmat = weekly_cov
    )

    logging.info("Optimising Sortino with constraints...")
    
    w_sortino = po.msr_sortino(
        riskfree_rate = rf_rate, 
        er = comb_rets, 
        weekly_ret = last_year_weekly_rets,
        scores = comb_score, 
        ticker_ind = ticker_ind, 
        ticker_sec = ticker_sec, 
        bnd_h = mcap_bnd_h, 
        bnd_l = mcap_bnd_l
    )
    
    print("Sortino Weights:", w_sortino)
    
    vol_sortino_ann = pf.portfolio_volatility(
        weights = w_sortino, 
        covmat = ann_cov
    )
    
    vol_sortino = pf.portfolio_volatility(
        weights = w_sortino, 
        covmat = weekly_cov
    )
    
    logging.info("Optimising Black-Litterman with constraints...")
    
    w_bl, mu_bl, sigma_bl = po.black_litterman_weights(
        tickers = tickers,
        comb_rets = comb_rets,
        comb_std = comb_ann_std,
        cov_prior = sigma_prior,
        mcap = mcap,
        score = comb_score,
        ticker_ind = ticker_ind,
        ticker_sec = ticker_sec,
        bnd_l = mcap_bnd_l,
        bnd_h = mcap_bnd_h,
    )
    
    print('Black-Litterman Weights:', w_bl)
    
    vol_bl_ann = pf.portfolio_volatility(
        weights = w_bl, 
        covmat = ann_cov
    )
    
    vol_bl = pf.portfolio_volatility(
        weights = w_bl, 
        covmat = weekly_cov
    )

    logging.info("Optimising MIR with constraints...")
    
    weekly_hist = weekly_ret.tail(52)
    
    w_mir = po.MIR(
        benchmark = benchmark_ret, 
        benchmark_weekly_ret = last_year_benchmark_weekly_rets, 
        er = comb_rets, 
        scores = comb_score,
        er_hist = last_year_weekly_rets, 
        ticker_ind = ticker_ind, 
        ticker_sec = ticker_sec, 
        bnd_h = mcap_bnd_h, 
        bnd_l = mcap_bnd_l
    )
    
    print("MIR Weights:", w_mir)
    
    vol_mir_ann = pf.portfolio_volatility(
        weights = w_mir, 
        covmat = ann_cov
    )
    
    vol_mir = pf.portfolio_volatility(
        weights = w_mir, 
        covmat = weekly_cov
    )

    logging.info("Optimising MSP with constraints...")
    
    w_msp = po.msp(
        scores = comb_score,
        er = comb_rets,
        cov = ann_cov,
        weekly_ret = last_5_year_weekly_rets,
        level = 5.0,          
        ticker_ind = ticker_ind,
        ticker_sec = ticker_sec,
        bnd_h = mcap_bnd_h,
        bnd_l = mcap_bnd_l
    )
            
    print("MSP Weights:", w_msp)    
        
    vol_msp_ann = pf.portfolio_volatility(
        weights = w_msp, 
        covmat = ann_cov
    )
    
    vol_msp = pf.portfolio_volatility(
        weights = w_msp, 
        covmat = weekly_cov
    )

    logging.info("Building a 'Combination' portfolio ...")
        
    w_comb = po.comb_port(
        riskfree_rate = rf_rate,
        er = comb_rets,
        cov = ann_cov,
        weekly_ret_1y = last_year_weekly_rets,
        last_5_year_weekly_rets = last_5_year_weekly_rets,
        benchmark = benchmark_ret,
        last_year_benchmark_weekly_ret = last_year_benchmark_weekly_rets,
        bnd_h = mcap_bnd_h,
        bnd_l = mcap_bnd_l,
        scores = comb_score,
        ticker_ind = ticker_ind,
        ticker_sec = ticker_sec,
        tickers = tickers,
        comb_std = comb_ann_std,
        sigma_prior = sigma_prior,
        mcap = mcap,
        w_msr = w_msr,
        w_sortino = w_sortino,
        w_bl = w_bl,
        w_mir = w_mir,
        w_msp = w_msp,
        mu_bl = mu_bl,
        sigma_bl = sigma_bl,
        gamma = (1.0, 1.0, 1.0, 1.0, 1.0),
    )
    
    print("Combination Weights:", w_comb)

    vol_comb_ann = pf.portfolio_volatility(
        weights = w_comb, 
        covmat = ann_cov
    )
    
    vol_comb = pf.portfolio_volatility(
        weights = w_comb, 
        covmat = weekly_cov
    )

    weights_df = pd.DataFrame({
        "MSR": w_msr,
        "Sortino": w_sortino,
        "MIR": w_mir,
        "MSP": w_msp,
        "BL": w_bl,
        "Combination": w_comb,
    })
    
    portfolio_weights = weights_df.reindex(tickers).multiply(money_in_portfolio).reset_index().rename(columns = {"index": "Ticker"})
    
    portfolio_weights_with_ind = portfolio_weights.merge(
        ticker_ind.rename("Industry"), 
        left_on = "Ticker", 
        right_index = True
    )

    industry_breakdown = portfolio_weights_with_ind.groupby("Industry").sum(numeric_only = True)
    industry_breakdown_percent = (industry_breakdown / money_in_portfolio) * 100
    industry_breakdown_percent = industry_breakdown_percent.reset_index()
    
    portfolio_weights_with_sec = portfolio_weights.merge(
        ticker_sec.rename("Sector"), 
        left_on = "Ticker", 
        right_index = True
    )
    
    sector_breakdown = portfolio_weights_with_sec.groupby("Sector").sum(numeric_only = True)
    sector_breakdown_percent = (sector_breakdown / money_in_portfolio) * 100
    sector_breakdown_percent = sector_breakdown_percent.reset_index()
    
    port_performance = {
        "MSR": pf.simulate_and_report("MSR", w_msr, comb_rets, bear_rets, bull_rets, vol_msr, vol_msr_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets, benchmark_ret, mu_bl, sigma_bl),
        "Sortino": pf.simulate_and_report("Sortino", w_sortino, comb_rets, bear_rets, bull_rets, vol_sortino, vol_sortino_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets, benchmark_ret, mu_bl, sigma_bl),
        "BL": pf.simulate_and_report("Black-Litterman", w_bl, comb_rets, bear_rets, bull_rets, vol_bl, vol_bl_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets, benchmark_ret, mu_bl, sigma_bl),
        "MIR": pf.simulate_and_report("MIR", w_mir, comb_rets, bear_rets, bull_rets, vol_mir, vol_mir_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets, benchmark_ret, mu_bl, sigma_bl),
        "MSP": pf.simulate_and_report("MSP", w_msp, comb_rets, bear_rets, bull_rets, vol_msp, vol_msp_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets, benchmark_ret, mu_bl, sigma_bl),
        "Comb": pf.simulate_and_report("Combination", w_comb, comb_rets, bear_rets, bull_rets, vol_comb, vol_comb_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets, benchmark_ret, mu_bl, sigma_bl),
    }
    performance_df = pd.DataFrame(port_performance).T
    
    ticker_performance = pf.report_ticker_metrics(
        tickers = tickers, 
        weekly_rets = weekly_ret,
        weekly_cov = weekly_cov, 
        ann_cov = ann_cov, 
        comb_rets = comb_rets,
        bear_rets = bear_rets,
        bull_rets = bull_rets,
        comb_score = comb_score,
        rf = rf_rate,
        beta = beta,
        benchmark_weekly_rets = benchmark_weekly_rets,
        benchmark_ann_ret = benchmark_ret,
        bl_ret = mu_bl,
        bl_cov = sigma_bl,
        forecast_file = config.FORECAST_FILE
    )
    
    logging.info("Writing results to Excel...")
    
    ticker_performance_df = pd.DataFrame(ticker_performance)
    
    sheets_to_write = {
        "Ticker Performance": ticker_performance_df,
        "Today_Buy_Sell": buy_sell_today_df,
        "Covariance": cov_df,
        "Bounds": bounds_df,
        "Weights": portfolio_weights,
        "Industry Breakdown": industry_breakdown_percent,
        "Sector Breakdown": sector_breakdown_percent,
        "Portfolio Performance": performance_df,
    }

    write_excel_results(
        excel_file = config.PORTFOLIO_FILE, 
        sheets = sheets_to_write
    )
  
    logging.info("Upload Complete")


if __name__ == "__main__":
    main()
