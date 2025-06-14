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
from openpyxl.formatting.rule import ColorScaleRule 
import portfolio_functions as pf
from functions.cov_functions import shrinkage_covariance
import portfolio_optimisers as po
from data_processing.ratio_data import RatioData

r = RatioData()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def ensure_headers_are_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that all column names and the index name of the DataFrame are strings.
    """
    df.columns = [str(col) if col is not None else "" for col in df.columns]
    if df.index.name is None:
        df.index.name = "Index"
    else:
        df.index.name = str(df.index.name)
    return df


money_in_portfolio: float = 4000

def load_excel_data(excel_file: str, excel_file2: str) -> Tuple[
    pd.DataFrame,   # weekly_ret
    List[str],      # tickers
    np.ndarray,     # weeklyCov
    pd.DataFrame,   # cov_df
    pd.DataFrame,   # buySignal
    pd.Series,      # beta
    pd.Series,      # erPred
    pd.Series,      # score
    pd.Series,      # combRets
    pd.Series,       # ticker_ind
    pd.Series,
    pd.Series,
    pd.Series, 
    pd.Series
]:
    """
    Loads required sheets from Excel, does index alignment, and returns them.
    """
    weekly_ret = pd.read_excel(excel_file, sheet_name="Historic Weekly Returns", index_col=0)
    weekly_ret.index = pd.to_datetime(weekly_ret.index)
    weekly_ret.sort_index(ascending=True, inplace=True)
    tickers = [c.upper() for c in weekly_ret.columns]
    weekly_ret.columns = tickers
    comb_data = pd.read_excel(excel_file2, sheet_name="Combination Forecast", index_col=0)
    comb_data.index = comb_data.index.str.upper()
    comb_data = comb_data.reindex(tickers)
    comb_std = comb_data['SE']
    score = comb_data["Score"]
    comb_rets = comb_data["Returns"]
    bear_rets = comb_data["Low Returns"]
    bull_rets = comb_data["High Returns"]
    
    ticker_data = pd.read_excel(excel_file2, sheet_name="Analyst Data", index_col=0)
    ticker_ind = ticker_data["Industry"]
    ticker_mcap = ticker_data['marketCap']
    ticker_sec = ticker_data['Sector']
    
    ret_corr = weekly_ret.corr()
    
    annCov = shrinkage_covariance(weekly_ret, comb_std, ret_corr)
    weeklyCov = annCov / 52
    cov_df = pd.DataFrame(annCov, index=tickers, columns=tickers)
    
    Signal = pd.read_excel(excel_file, sheet_name="Signal Scores", index_col=0).iloc[-1]
    
    beta_data = pd.read_excel(excel_file2, sheet_name="Analyst Data", index_col=0)["beta"]
    beta_data.index = beta_data.index.str.upper()
    beta = beta_data.reindex(tickers)
    
    return (weekly_ret, tickers, weeklyCov, annCov, cov_df,
            Signal, beta, 
            score, comb_rets, ticker_ind, ticker_sec, comb_std,
            bear_rets, bull_rets, ticker_mcap)


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



def _add_table(ws, table_name: str) -> None:
    """Convert the used-range of *ws* to an Excel table called *table_name*."""
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


def add_weight_cf(ws) -> None:
    """
    Add the three traffic-light rules (red <0.01, yellow 1.95–2.05, green “valid”)
    to every numeric column (B…last) of the supplied worksheet.
    """
    red   = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow= PatternFill(start_color="FFEB9B", end_color="FFEB9B", fill_type="solid")

    first_data_col = 2                               # B
    last_data_col  = ws.max_column

    for col_idx in range(first_data_col, last_data_col + 1):
        col_letter = get_column_letter(col_idx)
        rng        = f"{col_letter}2:{col_letter}{ws.max_row}"

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

    # two-decimal number format
    for row in ws.iter_rows(min_row=2, min_col=first_data_col, max_col=last_data_col):
        for cell in row:
            cell.number_format = "0.00"


def write_excel_results(excel_file: str, sheets: dict[str, pd.DataFrame]) -> None:
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
    red   = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:

        tbs = ensure_headers_are_strings(sheets["Today_Buy_Sell"].copy())
        tbs.to_excel(writer, sheet_name="Today_Buy_Sell", index=False)
        ws = writer.sheets["Today_Buy_Sell"]
        n  = tbs.shape[0] + 1
        ws.conditional_formatting.add(f"B2:B{n}", CellIsRule(operator="equal", formula=["TRUE"],  fill=green))
        ws.conditional_formatting.add(f"B2:B{n}", CellIsRule(operator="equal", formula=["FALSE"], fill=red))
        ws.conditional_formatting.add(f"C2:C{n}", CellIsRule(operator="equal", formula=["TRUE"],  fill=green))
        ws.conditional_formatting.add(f"C2:C{n}", CellIsRule(operator="equal", formula=["FALSE"], fill=red))
        _add_table(ws, "TodayBuySellTable")

        cov = ensure_headers_are_strings(sheets["Covariance"].copy())

        bnds = ensure_headers_are_strings(sheets["Bounds"].copy())
        bnds.to_excel(writer, sheet_name="Bounds")
        bnds_ws = writer.sheets["Bounds"]
        _add_table(bnds_ws, "BoundsTable")

        cov.to_excel(writer, sheet_name="Covariance")
        cov_ws = writer.sheets["Covariance"]
        _add_table(cov_ws, "CovarianceTable")

        # now apply the gradient
        data = cov.values.astype(float)
        min_val, mean_val, max_val = data.min(), data.mean(), data.max()
        color_rule = ColorScaleRule(
            start_type='num', start_value=str(min_val), start_color='FFFFFF',
            mid_type='num',   mid_value=str(mean_val), mid_color='FFDD99',
            end_type='num',   end_value=str(max_val), end_color='FF0000'
        )
        # note: B2 to last column/row skips the index column
        rng = f"B2:{get_column_letter(cov_ws.max_column)}{cov_ws.max_row}"
        cov_ws.conditional_formatting.add(rng, color_rule)


        def dump_weight_sheet(df_key: str, sheet_name: str, tablename: str) -> None:
            df = ensure_headers_are_strings(sheets[df_key].copy())
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            add_weight_cf(ws)
            _add_table(ws, tablename)

        dump_weight_sheet("Weights",       "Portfolio Weights", "WeightsTable")
        dump_weight_sheet("Weights_Bear",  "Bear Weights",      "WeightsBearTable")
        dump_weight_sheet("Weights_Bull",  "Bull Weights",      "WeightsBullTable")

        for key, sheet_name, tablename in [
            ("Portfolio Performance",      "Portfolio Performance",      "PortfolioPerformanceTable"),
            ("Portfolio Performance Bear", "Portfolio Performance Bear", "BearMarketPerformanceTable"),
            ("Portfolio Performance Bull", "Portfolio Performance Bull", "BullMarketPerformanceTable"),
            ("Industry Breakdown",         "Industry Breakdown",         "IndustryBreakdownTable"),
            ("Industry Breakdown Bear",    "Industry Breakdown Bear",    "IndustryBreakdownBearTable"),
            ("Industry Breakdown Bull",    "Industry Breakdown Bull",    "IndustryBreakdownBullTable"),
            ("Sector Breakdown",         "Sector Breakdown",         "SectorBreakdownTable"),
            ("Sector Breakdown Bear",    "Sector Breakdown Bear",    "SectorBreakdownBearTable"),
            ("Sector Breakdown Bull",    "Sector Breakdown Bull",    "SectorBreakdownBullTable"),
        ]:
            df = ensure_headers_are_strings(sheets[key].copy())
            df.to_excel(writer, sheet_name=sheet_name, index="Industry" not in key)
            _add_table(writer.sheets[sheet_name], tablename)
            

def benchmark_rets(benchmark: str, start: dt.date, end: dt.date, steps: int) -> float:
    """
    Downloads benchmark data, computes returns, and annualises them.
    """
    if benchmark.upper() == 'SP500':
        close = yf.download('^GSPC', start=start, end=end)['Close'].squeeze()
    elif benchmark.upper() == 'NASDAQ':
        close = yf.download('^IXIC', start=start, end=end)['Close'].squeeze()
    elif benchmark.upper() == 'FTSE':
        close = yf.download('^FTSE', start=start, end=end)['Close'].squeeze()
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
    return float(pf.annualise_returns(rets, steps)), rets 


def compute_mcap_bounds(
    market_cap: pd.Series,
    beta: pd.Series,
    er: pd.Series,
    vol: pd.Series,
    score: pd.Series,
    tickers: pd.Index,
    min_all: float,
    max_all: float,
    max_all_l: float,
    min_all_u: float
):
    """
    Returns two dicts (lb, ub) mapping ticker -> lower / upper bound.
    """
    score_max = score.max()
    mcap_sqrt = np.sqrt(market_cap)
    mcap_vol = {}
    for t in tickers:
        if vol.loc[t] > 0 and beta.loc[t] > 0 and score.loc[t] > 0:
            mcap_vol[t] = mcap_sqrt.loc[t] / (vol.loc[t])# * abs(beta.loc[t]))
        else:
            mcap_vol[t] = 0.0
    mcap_vol_beta = pd.Series(mcap_vol)

    mcap_vol_beta = mcap_vol_beta.fillna(0)
    tot = mcap_vol_beta[er > 0].sum()
    if tot == 0:
        tot = 1e-12  

    lb = {}
    ub = {}
    for t in tickers:
        if er.loc[t] > 0 and score.loc[t] > 0:
            norm_score = score.loc[t] / score_max

            frac_l = mcap_vol_beta[t] / tot
            cl_l   = min(frac_l, max_all_l)
            lb[t]  = max(min_all, norm_score * cl_l)

            frac_u = np.sqrt(mcap_vol_beta[t] / tot)
            cl_u   = min(frac_u, max_all)
            ub[t]  = max(min_all_u, norm_score * cl_u)
        else:
            lb[t] = 0.0
            ub[t] = 0.0

    return pd.Series(lb), pd.Series(ub)

def main() -> None:
   
    logging.info("Starting portfolio optimisation script...")
    today = dt.date.today()  # Use yesterday's date for consistency
    start = today - dt.timedelta(days=5*365)
    
    tickers = r.tickers
    
    
    benchmark = 'SP500'
    benchmark_ret, benchmark_weekly_rets = benchmark_rets(benchmark, start, today, 252)
    excel_file = f"Portfolio_Optimisation_Data_{today}.xlsx"
    output_file = f"Portfolio_Optimisation_Forecast_{today}.xlsx"

    logging.info("Loading Excel data...")
    (weekly_ret, tickers, weekly_cov, ann_cov, cov_df,
     signal_score, beta,
     comb_score, comb_rets, ticker_ind, ticker_sec, comb_ann_std,
     bear_rets, bull_rets, mcap) = load_excel_data(excel_file, output_file)
    
    assert list(comb_rets.index) == tickers == list(cov_df.index) == list(cov_df.columns)
    assert comb_rets.isna().sum() == 0, "Some tickers have no forecasted return!"

    buy_sell_today_df, buy_flags, sell_flags = get_buy_sell_signals_from_scores(tickers, signal_score)
    logging.info("Data loaded and signals created.")
    
     
    rf_rate = 0.0465
    
    w_max = 0.1
    
    w_min = 2 / money_in_portfolio
    
    max_all_l = w_max / 2
    
    min_all_u = w_min * 2
    
    mcap_bnd_l, mcap_bnd_h = compute_mcap_bounds(
        mcap, beta, comb_rets, comb_ann_std, comb_score,
        tickers, w_min, w_max, max_all_l, min_all_u
    )
    
    mcap_bnd_bear_l, mcap_bnd_bear_h = compute_mcap_bounds(
        mcap, beta, bear_rets, comb_ann_std, comb_score,
        tickers, w_min, w_max, max_all_l, min_all_u
    )
    
    mcap_bnd_bull_l, mcap_bnd_bull_h = compute_mcap_bounds(
        mcap, beta, bull_rets, comb_ann_std, comb_score,
        tickers, w_min, w_max, max_all_l, min_all_u
    )
    
    bounds_df = pd.DataFrame({
        'Low': mcap_bnd_l,
        'High': mcap_bnd_h,
        'Low Bear': mcap_bnd_bear_l,
        'High Bear': mcap_bnd_bear_h,
        'Low Bull': mcap_bnd_bull_l,
        'High Bull': mcap_bnd_bull_h
    })
    
    
    logging.info("Optimising MSR with constraints...")
    w_msr = po.msr_2(rf_rate, comb_rets, ann_cov, w_max, buy_flags, sell_flags, comb_score, ticker_ind, mcap_bnd_h, mcap_bnd_l)
    vol_msr_ann = pf.portfolio_volatility(w_msr, ann_cov)
    vol_msr = pf.portfolio_volatility(w_msr, weekly_cov)

    logging.info("Optimising Sortino with constraints...")
    w_sortino = po.msr_sortino(rf_rate, comb_rets, weekly_ret.tail(52),
                            w_max, buy_flags, sell_flags, comb_score, ticker_ind, mcap_bnd_h, mcap_bnd_l)
    vol_sortino_ann = pf.portfolio_volatility(w_sortino, ann_cov)
    vol_sortino = pf.portfolio_volatility(w_sortino, weekly_cov)

    logging.info("Optimising MIR with constraints...")
    weekly_hist = weekly_ret.tail(52)
    w_mir = po.MIR(benchmark_ret, benchmark_weekly_rets, comb_rets, buy_flags, sell_flags, comb_score,
                   w_max, weekly_hist, ticker_ind, mcap_bnd_h, mcap_bnd_l)
    vol_mir_ann = pf.portfolio_volatility(w_mir, ann_cov)
    vol_mir = pf.portfolio_volatility(w_mir, weekly_cov)

    logging.info("Optimising MSP with constraints...")
    w_msp = po.msp(comb_score, ann_cov, buy_flags, sell_flags, comb_rets,
                w_max, ticker_ind, mcap_bnd_h, mcap_bnd_l)
    vol_msp_ann = pf.portfolio_volatility(w_msp, ann_cov)
    vol_msp = pf.portfolio_volatility(w_msp, weekly_cov)

    logging.info("Building a 'Combination' portfolio ...")
    w_comb = po.comb_port(rf_rate, comb_rets, ann_cov, w_max, buy_flags, sell_flags, comb_score, ticker_ind, weekly_ret, mcap_bnd_h, mcap_bnd_l, benchmark_ret, benchmark_weekly_rets)
    vol_comb_ann = pf.portfolio_volatility(w_comb, ann_cov)
    vol_comb = pf.portfolio_volatility(w_comb, weekly_cov)

    portfolio_weights = pd.DataFrame([
        {
            "Ticker": t,
            "MSR": w_msr[i] * money_in_portfolio,
            "Sortino": w_sortino[i] * money_in_portfolio,
            "MIR": w_mir[i] * money_in_portfolio,
            "MSP": w_msp[i] * money_in_portfolio,
            "Combination": w_comb[i] * money_in_portfolio,
        }
        for i, t in enumerate(tickers)
    ])
    
    portfolio_weights_with_ind = portfolio_weights.merge(
        ticker_ind.rename("Industry"), left_on="Ticker", right_index=True
    )

    industry_breakdown = portfolio_weights_with_ind.groupby("Industry").sum(numeric_only=True)
    industry_breakdown_percent = (industry_breakdown / money_in_portfolio) * 100
    industry_breakdown_percent = industry_breakdown_percent.reset_index()
    
    portfolio_weights_with_sec = portfolio_weights.merge(
        ticker_sec.rename("Sector"), left_on="Ticker", right_index=True
    )
    
    sector_breakdown = portfolio_weights_with_sec.groupby("Sector").sum(numeric_only=True)
    sector_breakdown_percent = (sector_breakdown / money_in_portfolio) * 100
    sector_breakdown_percent = sector_breakdown_percent.reset_index()
    
    
    logging.info("Optimising MSR for Bear Case...")
    w_msr_bear = po.msr_2(rf_rate, bear_rets, ann_cov, w_max, buy_flags, sell_flags, comb_score, ticker_ind, mcap_bnd_bear_h, mcap_bnd_bear_l)
    vol_msr_bear = pf.portfolio_volatility(w_msr_bear, weekly_cov)
    vol_msr_bear_ann = pf.portfolio_volatility(w_msr_bear, ann_cov)

    
    logging.info("Optimising Sortino for Bear Case...")
    w_sortino_bear = po.msr_sortino(rf_rate, bear_rets, weekly_ret.tail(52), 
                                  w_max, buy_flags, sell_flags, comb_score, beta, mcap_bnd_bear_h, mcap_bnd_bear_l)
    vol_sortino_bear = pf.portfolio_volatility(w_sortino_bear, weekly_cov)
    vol_sortino_bear_ann = pf.portfolio_volatility(w_sortino_bear, ann_cov)
    
    logging.info("Optimising MIR for Bear Case...")
    w_mir_bear = po.MIR(benchmark_ret, benchmark_weekly_rets, bear_rets, buy_flags, sell_flags, comb_score,
                     w_max, weekly_ret.tail(52), ticker_ind, mcap_bnd_bear_h, mcap_bnd_bear_l)
    vol_mir_bear = pf.portfolio_volatility(w_mir_bear, weekly_cov)
    vol_mir_bear_ann = pf.portfolio_volatility(w_mir_bear, ann_cov)


    logging.info("Optimising Combination for Bear Case...")
    w_comb_bear = po.comb_port(rf_rate, bear_rets, ann_cov, w_max, buy_flags, sell_flags, comb_score, ticker_ind, weekly_ret, mcap_bnd_bear_h, mcap_bnd_bear_l, benchmark_ret, benchmark_weekly_rets)
    vol_comb_bear = pf.portfolio_volatility(w_comb_bear, ann_cov)
    vol_comb_bear_ann = pf.portfolio_volatility(w_comb_bear, ann_cov)

    logging.info("Optimising MSR for Bull Case...")
    w_msr_bull = po.msr_2(rf_rate, bull_rets, ann_cov, w_max, buy_flags, sell_flags, comb_score, ticker_ind, mcap_bnd_bull_h, mcap_bnd_bull_l)
    vol_msr_bull = pf.portfolio_volatility(w_msr_bull, weekly_cov)
    vol_msr_bull_ann = pf.portfolio_volatility(w_msr_bull, ann_cov)

    logging.info("Optimising Sortino for Bull Case...")
    w_sortino_bull = po.msr_sortino(rf_rate, bull_rets, weekly_ret.tail(52),
                                  w_max, buy_flags, sell_flags, comb_score, ticker_ind, mcap_bnd_bull_h, mcap_bnd_bull_l)
    vol_sortino_bull = pf.portfolio_volatility(w_sortino_bull, weekly_cov)
    vol_sortino_bull_ann = pf.portfolio_volatility(w_sortino_bull, ann_cov)

    logging.info("Optimising MIR for Bull Case...")
    w_mir_bull = po.MIR(benchmark_ret, benchmark_weekly_rets, bull_rets, buy_flags, sell_flags, comb_score,
                     w_max, weekly_ret.tail(52), ticker_ind, mcap_bnd_bull_h, mcap_bnd_bull_l)
    vol_mir_bull = pf.portfolio_volatility(w_mir_bull, weekly_cov)
    vol_mir_bull_ann = pf.portfolio_volatility(w_mir_bull, ann_cov)

    logging.info("Optimising Combination for Bull Case...")
    w_comb_bull = po.comb_port(rf_rate, bull_rets, ann_cov, w_max, buy_flags, sell_flags, comb_score, ticker_ind, weekly_ret, mcap_bnd_bull_h, mcap_bnd_bull_l, benchmark_ret, benchmark_weekly_rets)
    vol_comb_bull = pf.portfolio_volatility(w_comb_bull, weekly_cov)
    vol_comb_bull_ann = pf.portfolio_volatility(w_comb_bull, ann_cov)

    
    port_performance = {
        "MSR": pf.simulate_and_report("MSR", w_msr, comb_rets, bear_rets, bull_rets, vol_msr, vol_msr_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "Sortino": pf.simulate_and_report("Sortino", w_sortino, comb_rets, bear_rets, bull_rets, vol_sortino, vol_sortino_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "MIR": pf.simulate_and_report("MIR", w_mir, comb_rets, bear_rets, bull_rets, vol_mir, vol_mir_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "MSP": pf.simulate_and_report("MSP", w_msp, comb_rets, bear_rets, bull_rets, vol_msp, vol_msp_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "Comb": pf.simulate_and_report("Combination", w_comb, comb_rets, bear_rets, bull_rets, vol_comb, vol_comb_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
    }
    performance_df = pd.DataFrame(port_performance)
    
    bear_port_performance = {
        "MSR": pf.simulate_and_report("MSR Bear", w_msr_bear, comb_rets, bear_rets, bull_rets, vol_msr_bear, vol_msr_bear_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "Sortino": pf.simulate_and_report("Sortino Bear", w_sortino_bear, comb_rets, bear_rets, bull_rets, vol_sortino_bear, vol_sortino_bear_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "MIR": pf.simulate_and_report("MIR Bear", w_mir_bear, comb_rets, bear_rets, bull_rets, vol_mir_bear, vol_mir_bear_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "Comb": pf.simulate_and_report("Combination Bear", w_comb_bear, comb_rets, bear_rets, bull_rets, vol_comb_bear, vol_comb_bear_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
    }
    bear_performance_df = pd.DataFrame(bear_port_performance)
    
    bull_port_performance = {
        "MSR": pf.simulate_and_report("MSR Bull", w_msr_bull, comb_rets, bear_rets, bull_rets, vol_msr_bull, vol_msr_bull_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "Sortino": pf.simulate_and_report("Sortino Bull", w_sortino_bull, comb_rets, bear_rets, bull_rets, vol_sortino_bull, vol_sortino_bull_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "MIR": pf.simulate_and_report("MIR Bull", w_mir_bull, comb_rets, bear_rets, bull_rets, vol_mir_bull, vol_mir_bull_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
        "Comb": pf.simulate_and_report("Combination Bull", w_comb_bull, comb_rets, bear_rets, bull_rets, vol_comb_bull, vol_comb_bull_ann, comb_score, weekly_ret, rf_rate, beta, benchmark_weekly_rets),
    }
    bull_performance_df = pd.DataFrame(bull_port_performance)
    
    portfolio_weights_bear = pd.DataFrame([
        {
            "Ticker": t,
            "MSR": w_msr_bear[i] * money_in_portfolio,
            "Sortino": w_sortino_bear[i] * money_in_portfolio,
            "MIR": w_mir_bear[i] * money_in_portfolio,
            "Combination": w_comb_bear[i] * money_in_portfolio,
        }
        for i, t in enumerate(tickers)
    ])
    
    portfolio_weights_bear_with_ind = portfolio_weights_bear.merge(
        ticker_ind.rename("Industry"), left_on="Ticker", right_index=True
    )
    
    portfolio_weights_bear_with_sec = portfolio_weights_bear_with_ind.merge(
        ticker_sec.rename("Sector"), left_on="Ticker", right_index=True
    )
    
    industry_breakdown_bear = portfolio_weights_bear_with_ind.groupby("Industry").sum(numeric_only=True)
    industry_breakdown_bear_percent = (industry_breakdown_bear / money_in_portfolio) * 100
    industry_breakdown_bear_percent = industry_breakdown_bear_percent.reset_index()
    
    sector_breakdown_bear = portfolio_weights_bear_with_sec.groupby("Sector").sum(numeric_only=True)
    sector_breakdown_bear_percent = (sector_breakdown_bear / money_in_portfolio) * 100
    sector_breakdown_bear_percent = sector_breakdown_bear_percent.reset_index()
    
    portfolio_weights_bull = pd.DataFrame([
        {
            "Ticker": t,
            "MSR": w_msr_bull[i] * money_in_portfolio,
            "Sortino": w_sortino_bull[i] * money_in_portfolio,
            "MIR": w_mir_bull[i] * money_in_portfolio,
            "Combination": w_comb_bull[i] * money_in_portfolio,
        }
        for i, t in enumerate(tickers)
    ])
    portfolio_weights_bull_with_ind = portfolio_weights_bull.merge(
        ticker_ind.rename("Industry"), left_on="Ticker", right_index=True
    )
    industry_breakdown_bull = portfolio_weights_bull_with_ind.groupby("Industry").sum(numeric_only=True)
    industry_breakdown_bull_percent = (industry_breakdown_bull / money_in_portfolio) * 100
    industry_breakdown_bull_percent = industry_breakdown_bull_percent.reset_index()
    
    portfolio_weights_bull_with_sec = portfolio_weights_bull_with_ind.merge(
        ticker_sec.rename("Sector"), left_on="Ticker", right_index=True
    )
    sector_breakdown_bull = portfolio_weights_bull_with_sec.groupby("Sector").sum(numeric_only=True)
    sector_breakdown_bull_percent = (sector_breakdown_bull / money_in_portfolio) * 100
    sector_breakdown_bull_percent = sector_breakdown_bull_percent.reset_index()
    
    logging.info("Writing results to Excel...")
    sheets_to_write = {
        "Today_Buy_Sell": buy_sell_today_df,
        "Covariance": cov_df,
        "Bounds": bounds_df,
        "Weights": portfolio_weights,
        "Weights_Bear": portfolio_weights_bear,
        "Weights_Bull": portfolio_weights_bull,
        "Industry Breakdown": industry_breakdown_percent,
        "Industry Breakdown Bear": industry_breakdown_bear_percent,
        "Industry Breakdown Bull": industry_breakdown_bull_percent,
        "Sector Breakdown": sector_breakdown_percent,
        "Sector Breakdown Bear": sector_breakdown_bear_percent,
        "Sector Breakdown Bull": sector_breakdown_bull_percent,
        "Portfolio Performance": performance_df,
        "Portfolio Performance Bear": bear_performance_df,
        "Portfolio Performance Bull": bull_performance_df,
    }

    write_excel_results(output_file, sheets_to_write)
    logging.info("Upload Complete")


if __name__ == "__main__":
    main()
