"""
End-to-end portfolio optimisation and reporting pipeline.

This script loads forecasts, returns, and classifications; builds risk models;  
runs a suite of long-only optimisers via `PortfolioOptimiser`; and exports both 
weights and diagnostics to an Excel workbook with conditional formatting.

Pipeline outline
----------------
1) **Data ingestion and alignment**
  
   - Loads recent daily/weekly/monthly returns, score/return forecasts, per-ticker 
     sector/industry and market caps, and factor/benchmark series.
  
   - Builds annual and weekly covariance estimates using a shrinkage routine that 
     can blend idiosyncratic and factor components.
  
   - Resamples, aligns, and prunes histories to a common ticker universe.

2) **Risk model construction**

   - `ann_cov` (annualised covariance) and `weekly_cov = ann_cov / 52`.
 
   - Optional Black–Litterman (BL) prior covariance (`sigma_prior`) and view 
     uncertainty from forecast SE to derive a BL posterior inside the optimiser.

3) **Bounds and gating**
 
   - Lower/upper per-ticker bounds computed by `compute_mcap_bounds` from market cap,
     forecasted return and score, ex-ante volatility and a factor-tilt predictor.
 
   - Enforced long-only, budget, industry and sector caps via the optimiser.

4) **Optimisers executed**
  
   - **MSR**: Maximum Sharpe ratio.
  
   - **Sortino**: Maximum Sortino ratio (downside risk, 1-year weekly).
  
   - **MIR**: Maximum Information Ratio (active mean over tracking error).
  
   - **MSP**: Score-over-CVaR (Rockafellar–Uryasev 1-year CVaR).
  
   - **BL**: MSR under Black–Litterman posterior (μ_bl, Σ_bl).
  
   - **Deflated MSR**: Maximises Deflated Sharpe (controls selection bias).
  
   - **Adjusted MSR**: Maximises Adjusted Sharpe (skew/kurtosis correction).
  
   - **Composite CCP portfolios** (all long-only, same constraint set):
  
       • `comb_port`   = Sharpe + Sortino + BL-Sharpe − proximity to MIR/MSP.  
  
            Advantage: stability from anchoring to two practical baselines.
  
       • `comb_port1`  = Sharpe + Sortino + BL-Sharpe + IR(1y) + Score/CVaR.  
  
            Advantage: balanced near-term alpha, risk and tails; RU CVaR exact at end.
  
       • `comb_port2`  = Sharpe + Sortino + BL-Sharpe − L1 proximity to MIR/MSP.  
  
            Advantage: sparser tilts, turnover-friendly.
  
       • `comb_port3`  = Sharpe + Sortino + IR(5y) + Score/CVaR − L2 to BL.  
            
            Advantage: long-horizon discipline around BL equilibrium.
       
       • `comb_port4`  = As in 3 with L1 to BL + **sector risk caps** (linearised).  
           
            Advantage: limits sector risk concentration with sparse BL tilts.
       
       • `comb_port5`  = Sharpe + Sortino + BL-Sharpe + IR(1y) + Score/CVaR + ASR.  
            
            Advantage: higher-moment quality via adjusted Sharpe.
       
       • `comb_port6`  = Sharpe + Sortino + IR(5y) + Score/CVaR + ASR − L2 to BL.  
            
            Advantage: long-horizon + tails + higher moments, regularised to BL.
       
       • `comb_port7`  = As in 6 with L1 to BL + **sector risk caps** (linearised).  
            
            Advantage: sector budgets + sparse BL tilts + long-horizon risk.

       • `comb_port8`  = Sharpe + Sortino + BL-Sharpe + Score/CVaR + Adjusted Sharpe 
            
            Advantage: pure reward composite that balances mean–variance, downside/tails,
            and higher moments; BL posterior alignment **without** proximity penalties; 
            auto-scaling equalises gradient pushes across terms.


5) **Reporting**
  
   - Exports per-portfolio weights, industry/sector breakdowns (% of capital), 
     per-portfolio and per-ticker performance diagnostics, covariance summary, 
     and gating bounds to named Excel sheets.
  
   - Applies conditional formatting (tables, colour scales, “traffic-light” rules).

Inputs & configuration
----------------------
- Paths, ticker list, benchmark, dates, sector caps, risk-free rates, gamma weights,
  and Excel sheet names are drawn from `config`.

- Market/factor/benchmark series are fetched through `RatioData` and `yfinance`.

Outputs
-------
- Excel workbook updated in place with the following sheets (replaced if present):
  `Ticker Performance`, `Today_Buy_Sell`, `Covariance`, `Covariance Description`,
  `Bounds`, `Weights`, `Industry Breakdown`, `Sector Breakdown`,
  `Portfolio Performance`.

Notes
-----
- `yfinance` download calls require network connectivity and may incur API delays.

- All optimisers enforce the same long-only, budget, and box/sector/industry caps;
  some composite variants also include CCP linearised sector risk-contribution caps.
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
from portfolio_optimisers import PortfolioOptimiser
from data_processing.ratio_data import RatioData
import config

r = RatioData()

tickers = config.tickers

money_in_portfolio = config.MONEY_IN_PORTFOLIO

MIN_STD = 1e-2

MAX_STD = 2


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s"
)


def ensure_headers_are_strings(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure all DataFrame headers are strings.

    Coerces each column name to `str` (empty string if None) and ensures the index
    has a non-None string name.

    Args:
        df: Input DataFrame.

    Returns:
        The same DataFrame with stringified column names and a string index name.
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
    Load and align all data required by the optimiser.

    This function pulls returns and forecast sheets, aligns indices to the configured
    ticker universe, computes annualised covariance with shrinkage, and prepares
    auxiliary inputs (scores, sectors, industries, market caps, factor predictor).

    Data sources:
        - Weekly/daily returns and factor series from `RatioData`.
        - Excel sheets:
            • PORTFOLIO_FILE / "Combination Forecast": forecasted Returns, SE, Score,
            Low/High Returns.
            • FORECAST_FILE / "Analyst Data": Industry, Sector, marketCap, beta.
            • DATA_FILE / "Signal Scores": daily signal snapshot (last row).

    Risk model:
        - `annCov`: annual covariance (possibly factor-shrunk); `weeklyCov = annCov/52`.
        - `sigma_prior`: prior covariance for BL (idiosyncratic-only variant).

    Returns:
        A tuple:
            weekly_ret_5y:   pd.DataFrame of weekly returns (5y window).
            tickers:         List[str] current universe from config.
            weeklyCov:       np.ndarray, annual covariance / 52.
            annCov:          np.ndarray, annual covariance (aligned).
            annCov_desc_df:  pd.DataFrame, shrinkage description/attribution.
            cov_df:          pd.DataFrame, annual covariance with labels.
            Signal:          pd.Series, latest signal score per ticker.
            beta:            pd.Series, analyst beta per ticker.
            score:           pd.Series, model score per ticker.
            comb_rets:       pd.Series, forecast expected returns per ticker.
            ticker_ind:      pd.Series, industry per ticker.
            ticker_sec:      pd.Series, sector per ticker.
            comb_std:        pd.Series, forecast standard error per ticker.
            bear_rets:       pd.Series, stressed (low) forecast returns.
            bull_rets:       pd.Series, optimistic (high) forecast returns.
            ticker_mcap:     pd.Series, market capitalisation.
            sigma_prior:     np.ndarray, prior covariance for BL (aligned).
            factor_pred:     pd.Series, factor-tilt predictor used in bounds.

    Raises:
        AssertionError: if key indices fail to align to `tickers`.
        FileNotFoundError / ValueError: if required sheets are missing.
    """

    weekly_ret = r.weekly_rets
   
    daily_ret = r.daily_rets
  
    monthly_ret = r.close.resample('M').last().pct_change().dropna()
    
    weekly_ret.index = pd.to_datetime(weekly_ret.index)
  
    weekly_ret.sort_index(ascending = True, inplace = True)
      
    weekly_ret.columns = tickers
    
    comb_data = pd.read_excel(config.PORTFOLIO_FILE, sheet_name = "Combination Forecast", index_col = 0)
    
    comb_data.index = comb_data.index.str.upper()
    
    comb_data = comb_data.reindex(tickers)
    
    factor_pred = pd.read_excel(config.FORECAST_FILE, sheet_name = "Factor Exponential Regression", index_col = 0)['Returns']
    
    comb_std = comb_data['SE']
    
    score = comb_data["Score"]
    
    comb_rets = comb_data["Returns"]
  
    bear_rets = comb_data["Low Returns"]
  
    bull_rets = comb_data["High Returns"]
    
    ticker_data = pd.read_excel(config.FORECAST_FILE, sheet_name = "Analyst Data", index_col = 0)
    
    ticker_ind = ticker_data["Industry"]
  
    ticker_mcap = ticker_data['marketCap']
  
    ticker_sec = ticker_data['Sector']
    
    d_to_e = ticker_data['debtToEquity'] / 100
    
    tax = ticker_data['Tax Rate']
        
    daily_ret_5y = daily_ret.loc[daily_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
  
    weekly_ret_5y = weekly_ret.loc[weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
  
    monthly_ret_5y = monthly_ret.loc[monthly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
    
    factor_weekly, index_weekly, ind_weekly, sec_weekly = r.factor_index_ind_sec_weekly_rets(
        merge = False
    )
    
    base_ccy = getattr(config, "BASE_CCY", "GBP")

    daily_ret_5y_base = r.convert_returns_to_base(
        ret_df = daily_ret_5y,
        base_ccy = base_ccy,
        interval = "1d"
    )
    
    weekly_ret_5y_base = r.convert_returns_to_base(
        ret_df = weekly_ret_5y,
        base_ccy = base_ccy,
        interval = "1wk"
    )
    
    monthly_ret_5y_base = r.convert_returns_to_base(
        ret_df = monthly_ret_5y,
        base_ccy = base_ccy,
        interval = "1mo"
    )
    
    factor_weekly_base = r.convert_returns_to_base(
        ret_df = factor_weekly, 
        base_ccy = base_ccy,
        interval = "1wk"
    )
    
    index_weekly_base = r.convert_returns_to_base(
        ret_df = index_weekly,
        base_ccy = base_ccy,
        interval = "1wk"
    )
    
    ind_weekly_base = r.convert_returns_to_base(
        ret_df = ind_weekly,
        base_ccy = base_ccy, 
        interval = "1wk"
    )
    
    sec_weekly_base = r.convert_returns_to_base(
        ret_df = sec_weekly,
        base_ccy = base_ccy, 
        interval = "1wk"
    )
     
    annCov, annCov_desc = shrinkage_covariance(
        daily_5y = daily_ret_5y_base, 
        weekly_5y = weekly_ret_5y_base, 
        monthly_5y = monthly_ret_5y_base, 
        comb_std = comb_std, 
        common_idx = comb_rets.index,
        ff_factors_weekly = factor_weekly_base,
        index_returns_weekly = index_weekly_base,
        industry_returns_weekly = ind_weekly_base,
        sector_returns_weekly = sec_weekly_base,
        use_excess_ff = False,
        description = True
    )
        
    sigma_prior = shrinkage_covariance(
        daily_5y = daily_ret_5y_base, 
        weekly_5y = weekly_ret_5y_base, 
        monthly_5y = monthly_ret_5y_base, 
        comb_std = comb_std, 
        common_idx = tickers, 
        w_F = 0,       
        w_FF = 0,     
        w_IDX = 0,  
        w_IND = 0,     
        w_SEC = 0,   
    )
    
    annCov = annCov.reindex(index = tickers, columns = tickers)
        
    weeklyCov = annCov / 52
    
    cov_df = pd.DataFrame(annCov, index = tickers, columns = tickers)
    
    annCov_desc_df = pd.DataFrame(annCov_desc.T)
        
    Signal = pd.read_excel(config.DATA_FILE, sheet_name = "Signal Scores", index_col = 0).iloc[-1]
    
    beta_data = pd.read_excel(config.FORECAST_FILE, sheet_name = "COE", index_col = 0)["Beta_Levered_Used"]

    beta_data.index = beta_data.index.str.upper()

    beta = beta_data.reindex(tickers)
    
    return (weekly_ret_5y, tickers, weeklyCov, annCov, annCov_desc_df, cov_df,
            Signal, beta, 
            score, comb_rets, ticker_ind, ticker_sec, comb_std,
            bear_rets, bull_rets, ticker_mcap, sigma_prior, factor_pred,
            d_to_e, tax)


def get_buy_sell_signals_from_scores(
    tickers: List[str],
    signal_score: pd.Series
) -> Tuple[pd.DataFrame, List[bool], List[bool]]:
    """
    Derive simple buy/sell flags from a score vector.

    Rule: for ticker t, Buy = (score_t > 0), Sell = (score_t < 0).

    Args:
        tickers: Ordered list of tickers to evaluate.
        signal_score: pd.Series of scores indexed by ticker (case-insensitive).

    Returns:
        df:   pd.DataFrame with columns [Ticker, Buy, Sell, Score].
        buy_flags:  List[bool] aligned to `tickers`.
        sell_flags: List[bool] aligned to `tickers`.

    Notes:
        Missing tickers in `signal_score` default to Score=0 (neither buy nor sell).
    """

    signal_score.index = signal_score.index.str.upper()

    buy_flags = [bool(signal_score.get(t, 0) > 0) for t in tickers]
   
    sell_flags = [bool(signal_score.get(t, 0) < 0) for t in tickers]

    df = pd.DataFrame({
        "Ticker": tickers,
        "Buy": buy_flags,
        "Sell": sell_flags,
        "Score": [signal_score.get(t, 0) for t in tickers],
    })
    
    return df, buy_flags, sell_flags


def _add_table(
    ws, 
    table_name: str
) -> None:
    """
    Convert the used range of an openpyxl worksheet into a styled Excel table.

    Creates a table covering A1:(lastcol,lastrow) with `TableStyleMedium9`.

    Args:
        ws: openpyxl worksheet object.
        table_name: Name to assign to the created table.

    Side Effects:
        Modifies `ws` in place (adds a table).
    """

    last_col = get_column_letter(ws.max_column)
    
    table = Table(displayName=table_name, ref=f"A1:{last_col}{ws.max_row}")
    
    table.tableStyleInfo = TableStyleInfo(
        name = "TableStyleMedium9",
        showFirstColumn = False,
        showLastColumn = False,
        showRowStripes = True,
        showColumnStripes = False,
    )
    
    ws.add_table(table)


def add_weight_cf(
    ws
) -> None:
    """
    Apply “traffic-light” conditional formatting to numeric weight columns.

    Rules applied to each numeric column (rows 2…max_row):
        - Red fill if value < 0.01 (tiny / effectively zero).
        - Yellow fill if 1.95 ≤ value ≤ 2.05 (aggregated check; e.g., sum validation).
        - Green fill if cell is neither 0 nor 2 (treated as a “valid” entry).
    Also sets number format "0.00" for all numeric cells in data rows.

    Args:
        ws: openpyxl worksheet containing a weight table with headers in row 1.

    Side Effects:
        Adds conditional formatting rules and number formats to `ws`.
    """

    red = PatternFill(start_color = "FFC7CE", end_color = "FFC7CE", fill_type = "solid")
   
    green = PatternFill(start_color = "C6EFCE", end_color = "C6EFCE", fill_type = "solid")
   
    yellow = PatternFill(start_color = "FFEB9B", end_color = "FFEB9B", fill_type = "solid")

    first_data_col = 2                         
   
    last_data_col = ws.max_column

    for col_idx in range(first_data_col, last_data_col + 1):
   
        col_letter = get_column_letter(col_idx)
        
        rng = f"{col_letter}2:{col_letter}{ws.max_row}"

        ws.conditional_formatting.add(
            rng,
            CellIsRule(operator = "lessThan", formula=['0.01'], fill = red, stopIfTrue = True)
        )
        
       
        ws.conditional_formatting.add(
            rng,
            CellIsRule(operator = "between", formula=['1.95', '2.05'], fill = yellow, stopIfTrue = True)
        )
       
        ws.conditional_formatting.add(
            rng,
            FormulaRule(formula = [f"AND({col_letter}2<>0,{col_letter}2<>2)"], fill = green)
        )

    for row in ws.iter_rows(min_row = 2, min_col = first_data_col, max_col = last_data_col):
        
        for cell in row:
            
            cell.number_format = "0.00"


def write_excel_results(
    excel_file: str, 
    sheets: dict[str, pd.DataFrame]
) -> None:
    """
    Write result DataFrames to Excel and apply styling.

    Creates/replaces the following sheets (keys must be present in `sheets`):
        - "Ticker Performance"
        - "Today_Buy_Sell"
        - "Covariance"
        - "Covariance Description"
        - "Bounds"
        - "Weights"
        - "Industry Breakdown"
        - "Sector Breakdown"
        - "Portfolio Performance"

    Formatting:
        - Converts each written range to an Excel table with a consistent style.
        - Applies green/red fills to Buy/Sell booleans.
        - Applies 3-colour scale to the covariance numeric area.
        - Applies traffic-light rules to weight sheets.

    Args:
        excel_file: Path to the workbook (will be opened with mode='a' and
            `if_sheet_exists='replace'`).
        sheets: Mapping from sheet name (key) to DataFrame (value).

    Raises:
        KeyError: if a required sheet key is missing.
        ValueError: if DataFrame shapes are incompatible with expected formatting.
    """

    green = PatternFill(start_color = "C6EFCE", end_color = "C6EFCE", fill_type = "solid")
   
    red = PatternFill(start_color = "FFC7CE", end_color = "FFC7CE", fill_type = "solid")

    with pd.ExcelWriter(excel_file, engine = "openpyxl", mode = "a", if_sheet_exists = "replace") as writer:
        
        tperf = ensure_headers_are_strings(
            df = sheets["Ticker Performance"].copy()
        )
        
        tperf.to_excel(writer, sheet_name="Ticker Performance", index = True)
        
        ws_tp = writer.sheets["Ticker Performance"]

        tbs = ensure_headers_are_strings(
            df = sheets["Today_Buy_Sell"].copy()
        )
        
        tbs.to_excel(writer, sheet_name = "Today_Buy_Sell", index = False)

        ws = writer.sheets["Today_Buy_Sell"]
        
        n = tbs.shape[0] + 1
        
        ws.conditional_formatting.add(f"B2:B{n}", CellIsRule(operator = "equal", formula = ["TRUE"],  fill = green))
       
        ws.conditional_formatting.add(f"B2:B{n}", CellIsRule(operator = "equal", formula = ["FALSE"], fill = red))
       
        ws.conditional_formatting.add(f"C2:C{n}", CellIsRule(operator = "equal", formula = ["TRUE"],  fill = green))
       
        ws.conditional_formatting.add(f"C2:C{n}", CellIsRule(operator = "equal", formula = ["FALSE"], fill = red))
        
        _add_table(
            ws = writer.sheets['Ticker Performance'], 
            table_name = "TickerPerformanceTable"
        )
        
        _add_table(
            ws = ws, 
            table_name = "TodayBuySellTable"
        )

        cov = ensure_headers_are_strings(
            df = sheets["Covariance"].copy()
        )

        bnds = ensure_headers_are_strings(
            df = sheets["Bounds"].copy()
        )
        
        bnds.to_excel(writer, sheet_name = "Bounds")
        
        bnds_ws = writer.sheets["Bounds"]
        
        _add_table(
            ws = bnds_ws, 
            table_name = "BoundsTable"
        )

        cov.to_excel(writer, sheet_name = "Covariance")
        
        cov_ws = writer.sheets["Covariance"]
        
        _add_table(
            ws = cov_ws, 
            table_name = "CovarianceTable"
        )

        data = cov.values.astype(float)

        min_val = data.min()
        
        mean_val = data.mean()
        
        max_val = data.max()
        
        color_rule = ColorScaleRule(
            start_type = 'num', start_value = str(min_val), start_color = 'FFFFFF',
            mid_type = 'num', mid_value = str(mean_val), mid_color = 'FFDD99',
            end_type = 'num', end_value = str(max_val), end_color = 'FF0000'
        )

        rng = f"B2:{get_column_letter(cov_ws.max_column)}{cov_ws.max_row}"
       
        cov_ws.conditional_formatting.add(rng, color_rule)
        
        cov_desc = ensure_headers_are_strings(
            df = sheets['Covariance Description'].copy()
        )
        
        cov_desc.to_excel(writer, sheet_name = "Covariance Description", index = True)
        
        cov_desc_ws = writer.sheets["Covariance Description"]
                
        _add_table(
            ws = cov_desc_ws, 
            table_name = "CovarianceDescriptionTable"
        )


        def dump_weight_sheet(
            df_key: str,
            sheet_name: str, 
            tablename: str
        ) -> None:
       
            df = ensure_headers_are_strings(
                df = sheets[df_key].copy()
            )
       
            df.to_excel(writer, sheet_name = sheet_name, index = False)
       
            ws = writer.sheets[sheet_name]
       
            add_weight_cf(
                ws = ws
            )
       
            _add_table(
                ws = ws, 
                table_name = tablename
            )


        dump_weight_sheet(
            df_key = "Weights", 
            sheet_name = "Portfolio Weights", 
            tablename = "WeightsTable"
        )

        for key, sheet_name, tablename in [
            ("Portfolio Performance", "Portfolio Performance", "PortfolioPerformanceTable"),
            ("Industry Breakdown", "Industry Breakdown", "IndustryBreakdownTable"),
           ("Sector Breakdown", "Sector Breakdown", "SectorBreakdownTable"),
        ]:
       
            df = ensure_headers_are_strings(
                df = sheets[key].copy()
            )
           
            df.to_excel(writer, sheet_name = sheet_name, index = "Industry" not in key)
           
            _add_table(
                ws = writer.sheets[sheet_name],
                table_name = tablename
            )
            

def benchmark_rets(
    benchmark: str, 
    start: dt.date, 
    end: dt.date, 
    steps: int
) -> float:
    """
    Download benchmark prices and compute weekly returns and trailing annual return.

    Supported `benchmark` strings (case-insensitive):
       
        'SP500'  → '^GSPC'
       
        'NASDAQ' → '^IXIC'
       
        'FTSE'   → '^FTSE'
       
        'FTSE ALL-WORLD' → 'VWRL.L'
       
        'ALL'    → equal-weighted average of SP500, NASDAQ, FTSE annualised returns

    Computation:
    
        - Price series resampled to weekly last; returns = pct_change().dropna().
        
        - `last_year_rets` filters to the last calendar year window (`config.YEAR_AGO`).
        
        - Trailing annual return = ∏(1 + weekly_ret) − 1 over `last_year_rets`.

    Args:
        benchmark: Name from the supported list above.
        start: Start date for download.
        end: End date for download.
        steps: Periods per year used by external annualisers when `benchmark='ALL'`.

    Returns:
        benchmark_ann_ret: float, trailing annual return over the last year window.
        rets: pd.Series, full weekly returns over [start, end].
        last_year_rets: pd.Series, weekly returns within the last year.

    Raises:
        ValueError: for unsupported `benchmark`.
        Exception: network/IO errors from `yfinance` and resampling.
    """
    
    if benchmark.upper() == 'SP500':
        
        close = yf.download('^GSPC', start = start, end = end)['Close'].squeeze()
    
    elif benchmark.upper() == 'NASDAQ':
        
        close = yf.download('^IXIC', start = start, end = end)['Close'].squeeze()
    
    elif benchmark.upper() == 'FTSE':
        
        close = yf.download('^FTSE', start = start, end = end)['Close'].squeeze()
        
    elif benchmark.upper() == 'FTSE ALL-WORLD':
        
        close = yf.download('VWRL.L', start = start, end = end)['Close'].squeeze()
    
    elif benchmark.upper() == 'ALL':
    
        close_sp = yf.download('^GSPC', start = start, end = end)['Close'].squeeze()
     
        close_nd = yf.download('^IXIC', start = start, end = end)['Close'].squeeze()
     
        close_ft = yf.download('^FTSE', start = start, end = end)['Close'].squeeze()
    
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
    er: pd.Series,
    vol: pd.Series,
    score: pd.Series,
    tickers: pd.Index,
    ticker_sec: pd.Series,
    min_all: float,
    max_all: float,
    max_all_l: float,
    min_all_u: float,
    factor_pred: pd.Series
):
    """
    Compute per-ticker lower/upper weight bounds informed by capacity and quality.

    Bound logic (long-only):
        
        - Base capacity proxy: mcap / vol, modulated by a factor predictor (>=0.1 floored).
        
        - Eligibility: assets with er>0 and score>0 receive positive bounds; others get 0.
        
        - Lower bound: increases with normalised score and relative capacity share,
        but never below `min_all`. Healthcare/Staples receive a conservative tweak.
        
        - Upper bound: sqrt(capacity share) scaled by normalised score, clipped to `max_all`
        (and optionally to 2.5% for Healthcare/Staples), and no lower than `min_all_u`.
        
        - Exempt tickers in `config.TICKER_EXEMPTIONS`: fixed [0.01, 0.075].

    Args:
        market_cap: pd.Series of market caps (index = tickers).
        er: pd.Series of forecast expected returns (index = tickers).
        vol: pd.Series of annualised vol/SE (index = tickers).
        score: pd.Series of model scores (index = tickers).
        tickers: pd.Index universe order.
        ticker_sec: pd.Series mapping ticker → sector.
        min_all: global minimum lower bound.
        max_all: global cap for upper bound.
        max_all_l: cap applied to the computed lower bound share.
        min_all_u: minimum upper bound floor (ensures ub ≥ min_all_u when eligible).
        factor_pred: pd.Series factor-tilt predictor; used multiplicatively in capacity.

    Returns:
        (lb, ub, frac):
            lb:   pd.Series of lower bounds in [0,1] aligned to `tickers`.
            ub:   pd.Series of upper bounds in [0,1] aligned to `tickers`.
            frac: pd.Series of normalised capacity shares (sums to 1 over eligible).

    Notes:
        - Prints intermediate per-ticker diagnostics (factor predictor, bounds).
        - Non-eligible tickers receive (lb, ub) = (0, 0).
    """

    score_max = score.max()
    
    mcap_sqrt = np.sqrt(market_cap)
    
    mcap_vol = {}
    
    fp = (factor_pred.reindex(tickers).fillna(0.0) + 1).clip(lower = 0.1)
    
    for t in tickers:
       
        if vol.loc[t] > 0 and score.loc[t] > 0 and er.loc[t] > 0:
       
            mcap_vol[t] = fp.loc[t] * mcap_sqrt.loc[t] / vol.loc[t]
                   
        else:
       
            mcap_vol[t] = 0.0
    
    mcap_vol_beta = pd.Series(mcap_vol)

    mcap_vol_beta = mcap_vol_beta.fillna(0)
    
    mask = (er > 0) & (score > 0)
    
    tot = float(mcap_vol_beta[mask].sum())
    
    frac = mcap_vol_beta / tot
    
    if tot == 0:
        
        tot = 1e-12  

    lb = {}
    
    ub = {}
    
    for t in tickers:
        
        if t in config.TICKER_EXEMPTIONS:
            
            lb[t] = 0.01
          
            ub[t] = 0.075
    
        elif er.loc[t] > 0 and score.loc[t] > 0:
            
            score_t = score.loc[t]#min(score.loc[t], 20)
            
            norm_score = score_t / score_max
            
            cl_l = min(frac[t], max_all_l)

            frac_u = np.sqrt(frac[t])
            
            cl_u = min(frac_u, max_all) #frax_u
            
            if ticker_sec.loc[t] == 'Healthcare' or ticker_sec.loc[t] == 'Consumer Staples':
                        
                ub_val = min(norm_score * cl_u, 0.025)
            
                ub[t] = ub_val #max(ub_val, lb[t])
                
                lb[t] = min(max(norm_score * cl_l / 2, min_all), ub[t] / 2)
            
            else:
            
                ub[t] = max(min(norm_score * cl_u, max_all), min_all_u)
                
            lb[t] = min(min(max(min_all, norm_score * cl_l), ub[t] / 2), 0.025)
    
        else:
    
            lb[t] = 0.0
       
            ub[t] = 0.0

    return pd.Series(lb), pd.Series(ub), frac


def main() -> None:
    """
    Run the full optimisation workflow and export results.

    Steps:
       
        1) Download benchmark data and compute weekly/annual returns.
      
        2) Load Excel-based forecasts and analytics; build annual/weekly covariance
        (including a BL prior covariance).
      
        3) Build per-ticker bounds using market cap, scores, volatility and factor tilt.
      
        4) Instantiate `PortfolioOptimiser` with returns, covariances, histories,
        benchmark, classifications, bounds, BL inputs, sector caps, and rates.
      
        5) Solve base optimisers (MSR, Sortino, MIR, MSP, BL), DSR and ASR variants,
        and composite CCP portfolios (`comb_port`, …, `comb_port7`).
      
        6) Compute annual and weekly volatilities for each solution.
      
        7) Build per-ticker weight tables and industry/sector breakdowns (% of capital).
      
        8) Produce portfolio and ticker performance reports via `PortfolioAnalytics`.
      
        9) Write all outputs to Excel with conditional formatting and tables.

    Side Effects:
     
        - Logs progress to the root logger.
     
        - Appends/replaces sheets in `config.PORTFOLIO_FILE`.
     
        - Prints intermediate optimised weights and volatility figures.

    Raises:
        AssertionError: if key indices do not align to the configured tickers.
        RuntimeError / ValueError: bubbled up from optimiser/calibration steps or IO.
    """
   
    logging.info("Starting portfolio optimisation script...")
        
    benchmark_ret, benchmark_weekly_rets, last_year_benchmark_weekly_rets = benchmark_rets(
        benchmark = config.benchmark, 
        start = config.FIVE_YEAR_AGO, 
        end = config.TODAY, 
        steps = 52
    )

    logging.info("Loading Excel data...")
  
    (weekly_ret, tickers, weekly_cov, ann_cov, ann_cov_desc, cov_df, signal_score, beta, comb_score, comb_rets, ticker_ind, ticker_sec, comb_ann_std, bear_rets, bull_rets, mcap, sigma_prior, factor_pred, d_to_e, tax) = load_excel_data()
    
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
    
    tickers_index = pd.Index(tickers)
    
    mcap_bnd_l, mcap_bnd_h, mcap_vol_beta = compute_mcap_bounds(
        market_cap = mcap, 
        er = comb_rets, 
        vol = comb_ann_std, 
        score = comb_score,
        tickers = tickers_index, 
        ticker_sec = ticker_sec,
        min_all = w_min, 
        max_all = w_max, 
        max_all_l = max_all_l, 
        min_all_u = min_all_u,
        factor_pred = factor_pred
    )
    
    mcap["SGLP.L"] = mcap.max()
    
    bounds_df = pd.DataFrame({
        'mid': mcap_vol_beta,
        'Low': mcap_bnd_l,
        'High': mcap_bnd_h
    })
    
    last_year_weekly_rets = weekly_ret.loc[weekly_ret.index >= pd.to_datetime(config.YEAR_AGO)]   
    
    n_last_year_weeks = len(last_year_weekly_rets)
    
    last_5_year_weekly_rets = weekly_ret.loc[weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]    
    
    pa = pf.PortfolioAnalytics(cache = False)     
    
    logging.info("Optimising Portfolios...")
   
    opt = PortfolioOptimiser(
        er = comb_rets, 
        cov = ann_cov, 
        scores = comb_score,
        weekly_ret_1y = last_year_weekly_rets,
        last_5_year_weekly_rets = last_5_year_weekly_rets,
        benchmark_weekly_ret = last_year_benchmark_weekly_rets,
        ticker_ind = ticker_ind, 
        ticker_sec = ticker_sec,
        bnd_h = mcap_bnd_h, 
        bnd_l = mcap_bnd_l,
        comb_std = comb_ann_std, 
        sigma_prior = sigma_prior, 
        mcap = mcap,
        sector_limits = config.sector_limits,
        rf_annual = config.RF, 
        rf_week = config.RF_PER_WEEK,
        gamma = config.GAMMA,
    )

    w_msr = opt.msr()
    
    print("MSR Weights:", w_msr)
   
    w_sortino = opt.sortino()

    print("Sortino Weights:", w_sortino)
   
    w_mir = opt.MIR()

    print("MIR Weights:", w_mir)
   
    w_msp = opt.msp()

    print("MSP Weights:", w_msp)    
    
    w_gmv = opt.min_variance()

    print("GMV Weights:", w_gmv)
    
    w_mdp = opt.max_diversification()
    
    print("MDP Weights:", w_mdp)
    
    w_bl, mu_bl, sigma_bl = opt.black_litterman_weights()

    print('Black-Litterman Weights:', w_bl)

    deflated_w_msr = opt.optimise_deflated_sharpe()
    
    print("Deflated MSR Weights:", deflated_w_msr)
    
    adjusted_w_msr = opt.optimise_adjusted_sharpe()

    print("Adjusted MSR Weights:", adjusted_w_msr)
    
    w_upm_lpm = opt.upm_lpm_port()
    
    print("UPM-LPM Weights:", w_upm_lpm)
    
    w_up_down_cap = opt.upside_downside_capture_port()
    
    print("Upside/Downside Capture Weights:", w_up_down_cap)
    
    w_comb10 = opt.comb_port10()
    
    print("Combination10 Weights:", w_comb10)
    
    w_comb11 = opt.comb_port11()
    
    print("Combination11 Weights:", w_comb11)
    
    w_comb = opt.comb_port()
    
    print("Combination Weights:", w_comb)
   
    w_comb1 = opt.comb_port1()
    
    print("Combination1 Weights:", w_comb1)
   
    w_comb2 = opt.comb_port2()
    
    print("Combination2 Weights:", w_comb2)
    
    w_comb3 = opt.comb_port3()
    
    print("Combination3 Weights:", w_comb3)
    
    w_comb4 = opt.comb_port4()
    
    print("Combination4 Weights:", w_comb4)
    
    w_comb5 = opt.comb_port5()
    
    print("Combination5 Weights:", w_comb5)
    
    w_comb6 = opt.comb_port6()
    
    print("Combination6 Weights:", w_comb6)
    
    w_comb7 = opt.comb_port7()
    
    print("Combination7 Weights:", w_comb7)
    
    w_comb8 = opt.comb_port8()
    
    print("Combination8 Weights:", w_comb8)
    
    w_comb9 = opt.comb_port9()
    
    print("Combination9 Weights:", w_comb9)
    

            
    vol_msr_ann = pa.portfolio_volatility(
        weights = w_msr, 
        covmat = ann_cov
    )
    
    vol_msr = pa.portfolio_volatility(
        weights = w_msr, 
        covmat = weekly_cov
    )
            
    vol_sortino_ann = pa.portfolio_volatility(
        weights = w_sortino, 
        covmat = ann_cov
    )
    
    vol_sortino = pa.portfolio_volatility(
        weights = w_sortino, 
        covmat = weekly_cov
    )
        
    vol_bl_ann = pa.portfolio_volatility(
        weights = w_bl, 
        covmat = ann_cov
    )
    
    vol_bl = pa.portfolio_volatility(
        weights = w_bl, 
        covmat = weekly_cov
    )
        
    vol_mir_ann = pa.portfolio_volatility(
        weights = w_mir, 
        covmat = ann_cov
    )
    
    vol_mir = pa.portfolio_volatility(
        weights = w_mir, 
        covmat = weekly_cov
    )
    
    vol_gmv_ann = pa.portfolio_volatility(
        weights = w_gmv, 
        covmat = ann_cov
    )
    
    vol_gmv = pa.portfolio_volatility(
        weights = w_gmv, 
        covmat = weekly_cov
    )
    
    vol_mdp_ann = pa.portfolio_volatility(
        weights = w_mdp, 
        covmat = ann_cov
    )
    
    vol_mdp = pa.portfolio_volatility(
        weights = w_mdp, 
        covmat = weekly_cov
    )
        
    vol_msp_ann = pa.portfolio_volatility(
        weights = w_msp, 
        covmat = ann_cov
    )
    
    vol_msp = pa.portfolio_volatility(
        weights = w_msp, 
        covmat = weekly_cov
    )
        
    vol_deflated_msr_ann = pa.portfolio_volatility(
        weights = deflated_w_msr, 
        covmat = ann_cov
    )
    
    vol_deflated_msr = pa.portfolio_volatility(
        weights = deflated_w_msr, 
        covmat = weekly_cov
    )
        
    vol_adjusted_msr_ann = pa.portfolio_volatility(
        weights = adjusted_w_msr, 
        covmat = ann_cov
    )
    
    vol_adjusted_msr = pa.portfolio_volatility(
        weights = adjusted_w_msr, 
        covmat = weekly_cov
    )
    
    vol_upm_lpm_ann = pa.portfolio_volatility(
        weights = w_upm_lpm, 
        covmat = ann_cov
    )
    
    vol_upm_lpm = pa.portfolio_volatility(
        weights = w_upm_lpm, 
        covmat = weekly_cov
    )
    
    vol_up_down_cap_ann = pa.portfolio_volatility(
        weights = w_up_down_cap, 
        covmat = ann_cov
    )
    
    vol_up_down_cap = pa.portfolio_volatility(
        weights = w_up_down_cap, 
        covmat = weekly_cov
    )
        
    vol_comb_ann = pa.portfolio_volatility(
        weights = w_comb, 
        covmat = ann_cov
    )
    
    vol_comb = pa.portfolio_volatility(
        weights = w_comb, 
        covmat = weekly_cov
    )
    
    vol_comb1_ann = pa.portfolio_volatility(
        weights = w_comb1, 
        covmat = ann_cov
    )
    
    vol_comb1 = pa.portfolio_volatility(
        weights = w_comb1, 
        covmat = weekly_cov
    )
        
    vol_comb2_ann = pa.portfolio_volatility(
        weights = w_comb2, 
        covmat = ann_cov
    )
    
    vol_comb2 = pa.portfolio_volatility(
        weights = w_comb2, 
        covmat = weekly_cov
    )
        
    vol_comb3_ann = pa.portfolio_volatility(
        weights = w_comb3, 
        covmat = ann_cov
    )
    
    vol_comb3 = pa.portfolio_volatility(
        weights = w_comb3, 
        covmat = weekly_cov
    )
        
    vol_comb4_ann = pa.portfolio_volatility(
        weights = w_comb4, 
        covmat = ann_cov
    )
    
    vol_comb4 = pa.portfolio_volatility(
        weights = w_comb4, 
        covmat = weekly_cov
    )
        
    vol_comb5_ann = pa.portfolio_volatility(
        weights = w_comb5, 
        covmat = ann_cov
    )
    
    vol_comb5 = pa.portfolio_volatility(
        weights = w_comb5, 
        covmat = weekly_cov
    )
        
    vol_comb6_ann = pa.portfolio_volatility(
        weights = w_comb6, 
        covmat = ann_cov
    )
    
    vol_comb6 = pa.portfolio_volatility(
        weights = w_comb6, 
        covmat = weekly_cov
    )
        
    vol_comb7_ann = pa.portfolio_volatility(
        weights = w_comb7, 
        covmat = ann_cov
    )
    
    vol_comb7 = pa.portfolio_volatility(
        weights = w_comb7, 
        covmat = weekly_cov
    )
        
    vol_comb8_ann = pa.portfolio_volatility(
        weights = w_comb8, 
        covmat = ann_cov
    )
    
    vol_comb8 = pa.portfolio_volatility(
        weights = w_comb8, 
        covmat = weekly_cov
    )
    
    vol_comb9_ann = pa.portfolio_volatility(
        weights = w_comb9, 
        covmat = ann_cov
    )
    
    vol_comb9 = pa.portfolio_volatility(
        weights = w_comb9, 
        covmat = weekly_cov
    )
    
    vol_comb10_ann = pa.portfolio_volatility(
        weights = w_comb10, 
        covmat = ann_cov
    )
    
    vol_comb10 = pa.portfolio_volatility(
        weights = w_comb10, 
        covmat = weekly_cov
    )
    
    vol_comb11_ann = pa.portfolio_volatility(
        weights = w_comb11, 
        covmat = ann_cov
    )
    
    vol_comb11 = pa.portfolio_volatility(
        weights = w_comb11, 
        covmat = weekly_cov
    )
    
    
    var = pd.Series(np.diag(ann_cov), index = ann_cov.index)
    
    std = np.sqrt(var).clip(lower = MIN_STD, upper = MAX_STD)
    
    weights_df = pd.DataFrame({
        "MSR": w_msr,
        "Sortino": w_sortino,
        "MIR": w_mir,
        "GMV": w_gmv,
        "MDP": w_mdp,
        "MSP": w_msp,
        "BL": w_bl,
        "Deflated_MSR": deflated_w_msr,
        "Adjusted_MSR": adjusted_w_msr,
        "UPM-LPM": w_upm_lpm,
        "Upside/Downside": w_up_down_cap,
        "Combination": w_comb,
        "Combination1": w_comb1,
        "Combination2": w_comb2,
        "Combination3": w_comb3,
        "Combination4": w_comb4,
        "Combination5": w_comb5,
        "Combination6": w_comb6,
        "Combination7": w_comb7,
        "Combination8": w_comb8,
        "Combination9": w_comb9,
        "Combination10": w_comb10,
        "Combination11": w_comb11,
        "Expected Return": comb_rets,
        "Vol": std,
        "Score": comb_score,
    })
    
    weights = weights_df.reindex(tickers)
    
    cols_to_mult = [col for col in weights.columns if col not in ["Expected Return", "Vol", "Score"]]
    
    weights[cols_to_mult] = weights[cols_to_mult] * money_in_portfolio
    
    portfolio_weights = weights.reset_index().rename(columns = {"index": "Ticker"})
    
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

    logging.info("Generating portfolio performance reports...")
    
    performance_df = pa.report_portfolio_metrics(
        w_msr = w_msr,
        w_sortino = w_sortino,
        w_mir = w_mir,
        w_gmv = w_gmv,
        w_mdp = w_mdp,
        w_msp = w_msp,
        w_bl = w_bl,
        w_comb = w_comb,
        w_comb1 = w_comb1,
        w_comb2 = w_comb2,
        w_comb3 = w_comb3,
        w_comb4 = w_comb4,
        w_comb5 = w_comb5,
        w_comb6 = w_comb6,
        w_comb7 = w_comb7,
        w_comb8 = w_comb8,
        w_comb9 = w_comb9,
        w_comb10 = w_comb10,
        w_comb11 = w_comb11,
        w_deflated_msr = deflated_w_msr,
        w_adjusted_msr = adjusted_w_msr,
        w_upm_lpm = w_upm_lpm,
        w_up_down_cap = w_up_down_cap,
        comb_rets = comb_rets,
        bear_rets = bear_rets,
        bull_rets = bull_rets,
        vol_msr = vol_msr,
        vol_sortino = vol_sortino,
        vol_mir = vol_mir,
        vol_gmv = vol_gmv,
        vol_mdp = vol_mdp,
        vol_msp = vol_msp,
        vol_bl = vol_bl,
        vol_comb = vol_comb,
        vol_comb1 = vol_comb1,
        vol_comb2 = vol_comb2,
        vol_comb3 = vol_comb3,
        vol_comb4 = vol_comb4,
        vol_comb5 = vol_comb5,
        vol_comb6 = vol_comb6,
        vol_comb7 = vol_comb7,
        vol_comb8 = vol_comb8,
        vol_comb9 = vol_comb9,
        vol_comb10 = vol_comb10,
        vol_comb11 = vol_comb11,
        vol_deflated_msr = vol_deflated_msr,
        vol_adjusted_msr = vol_adjusted_msr,
        vol_upm_lpm = vol_upm_lpm,
        vol_up_down_cap = vol_up_down_cap,
        vol_msr_ann = vol_msr_ann,
        vol_sortino_ann = vol_sortino_ann,
        vol_mir_ann = vol_mir_ann,
        vol_gmv_ann = vol_gmv_ann,
        vol_mdp_ann = vol_mdp_ann,
        vol_msp_ann = vol_msp_ann,
        vol_bl_ann = vol_bl_ann,
        vol_comb_ann = vol_comb_ann,
        vol_comb1_ann = vol_comb1_ann,
        vol_comb2_ann = vol_comb2_ann,
        vol_comb3_ann = vol_comb3_ann,
        vol_comb4_ann = vol_comb4_ann,
        vol_comb5_ann = vol_comb5_ann,
        vol_comb6_ann = vol_comb6_ann,
        vol_comb7_ann = vol_comb7_ann,
        vol_comb8_ann = vol_comb8_ann,
        vol_comb9_ann = vol_comb9_ann,
        vol_comb10_ann = vol_comb10_ann,
        vol_comb11_ann = vol_comb11_ann,
        vol_deflated_msr_ann = vol_deflated_msr_ann,
        vol_adjusted_msr_ann = vol_adjusted_msr_ann,
        vol_upm_lpm_ann = vol_upm_lpm_ann,
        vol_up_down_cap_ann = vol_up_down_cap_ann,
        comb_score = comb_score,
        last_year_weekly_rets = last_year_weekly_rets,
        last_5y_weekly_rets = last_5_year_weekly_rets,
        n_last_year_weeks = n_last_year_weeks,
        rf_rate = rf_rate,
        beta = beta,
        benchmark_weekly_rets = last_year_benchmark_weekly_rets,
        benchmark_ret = benchmark_ret,
        mu_bl = mu_bl,
        sigma_bl = sigma_bl,
        d_to_e = d_to_e,
        tax = tax,
    )
        
    ticker_performance = pa.report_ticker_metrics(
        tickers = tickers, 
        last_year_weekly_rets = last_year_weekly_rets,
        last_5y_weekly_rets = last_5_year_weekly_rets,
        n_last_year_weeks = n_last_year_weeks,
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
        tax = tax,
        d_to_e = d_to_e,
        forecast_file = config.FORECAST_FILE
    )
    
    logging.info("Writing results to Excel...")
    
    ticker_performance_df = pd.DataFrame(ticker_performance)
    
    sheets_to_write = {
        "Ticker Performance": ticker_performance_df,
        "Today_Buy_Sell": buy_sell_today_df,
        "Covariance": cov_df,
        "Covariance Description": ann_cov_desc,
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
