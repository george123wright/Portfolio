"""
Aggregates predictions from multiple models and derives a blended return forecast per ticker.
"""

import numpy as np
import pandas as pd
import datetime as dt
import logging
from typing import Dict, Tuple
from data_processing.ratio_data import RatioData
from functions.export_forecast import export_results
from functions.cov_functions import shrinkage_covariance
import optimiser.portfolio_functions as pf
import optimiser.Port_Optimisation as po
import config


r = RatioData()

pa  = pf.PortfolioAnalytics(cache = False)     

weekly_ret = r.weekly_rets

daily_ret = r.daily_rets

monthly_ret = r.close.resample('M').last().pct_change().dropna()

today = dt.date.today()

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

IND_DATA_FILE = config.IND_DATA_FILE

MIN_STD = 1e-2

MAX_STD = 2

MAX_MODEL_WT = 0.10


def ensure_headers_are_strings(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure column headers (and the index name) are strings for Excel export safety.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sanitize.

    Returns
    -------
    pd.DataFrame
        Same frame, but with `df.columns` coerced to `str` and `df.index.name`
        set to a non-None string.

    Rationale
    ---------
    OpenXML writers are brittle with non-string headers. This prevents
    type-related surprises when exporting multiple sheets or applying styles.
    """
    
    df.columns = [str(col) if col is not None else '' for col in df.columns]
    
    if df.index.name is None:
        
        df.index.name = 'Index'
    
    else:
        
        df.index.name = str(df.index.name)
    
    return df


def fix_header_cells(
    ws
):
    """
    Coerce the *first row* (header row) cell values in an openpyxl worksheet to strings.

    Parameters
    ----------
    ws : openpyxl.worksheet.worksheet.Worksheet
        Target worksheet object.

    Returns
    -------
    None

    Notes
    -----
    Useful when a Table style or conditional formatting expects string headers,
    or when a sheet was created by a third-party exporter with non-string cells.
    """
   
    for cell in ws[1]:
        
        cell.value = str(cell.value) if cell.value is not None else ''


class PortfolioOptimiser:
    """
    PortfolioOptimiser

    Coordinates ingest of model outputs, analyst/fundamental data, and sentiment
    to build a combined expected return (ER) and volatility (SE) forecast per ticker,
    plus an additive scoring framework that produces a final cross-sectional score.

    High-level flow
    ---------------
   
    1) Load:
    - Model sheets with per-ticker ER and SE (σ) for many forecasters.
    - Analyst sheet with fundamentals and qualitative signals.
    - Sentiment sheet (WSB) with average sentiment and mention counts.
    - Latest prices and "Signal Scores" (technical) from the input workbook.

    2) Combine models:
    - Compute per-ticker inverse-variance weights
        w_i ∝ 1/σ_i^2      (i indexes models available for that ticker),
        with caps and group limits (historical / intrinsic value / factor / ML).
    - Combined return:
        ER_comb(t) = div(t) + Σ_i w_i(t) · ER_i(t)
        where div(t) is the per-ticker dividend yield (as decimal).
    - Combined variance with a simple within/between decomposition:
        Var_comb(t) = Σ_i w_i(t) · σ_i^2(t)                 [within]
                        + (1/M_t) · Σ_i w_i(t) · (ER_i(t) − ER_comb(t))^2
        where M_t is the number of valid models for ticker t.

    3) Score:
    - Start with short-interest score; then sequentially apply additive
        adjustments (sentiment, earnings/revenue growth vs industry, ROE/ROA,
        P/B vs industry, EPS diagnostics, upside/downside capture, alpha flags,
        as well as analyst/insider/capital-structure/margin/efficiency and
        risk-return diagnostics such as skewness, Sharpe, Sortino).
    - Final score is the row-sum of the score breakdown (capped by number
        of analyst opinions). This is designed for ranking, not for sizing.

    Attributes
    ----------
    excel_file_out : str
        Path to the workbook that already contains the model/analyst/sentiment sheets.
    excel_file_in : str
        Path to the workbook that contains "Signal Scores".
    ratio_data : RatioData
        Fundamental/returns utility with latest prices and return matrices.
    today : datetime.date
        Convenience timestamp.
    LOWER_PERCENTILE, UPPER_PERCENTILE, NINETY_PERCENTILE : int
        Percentiles used in some scoring thresholds.

    Notes
    -----
    - All intermediate arithmetic is performed on aligned indices only.
    - Many signals use simple binary (+1/−1/… ) increments to keep scores interpretable.
    - Combined SE is bounded to [MIN_STD, MAX_STD] to avoid pathological values.
    """

    def __init__(
        self,
        excel_file: str, 
        ratio_data: RatioData
    ):
        """
        Parameters
        ----------
        excel_file : str
            Workbook path containing model/analyst/sentiment sheets (the "Forecast" file).
        ratio_data : RatioData
            Provider of latest prices, daily/weekly/monthly returns, and factor/index
            returns used later for covariance and diagnostics.

        Side effects
        ------------
        - Stores paths, initializes percentiles and date.
        - Calls `_load_all_data()` to read all required sheets and cache them.

        Raises
        ------
        Any exception propagated by `_load_all_data()` if sheets are missing or malformed.
        """
        
        self.excel_file_out = excel_file
        
        self.excel_file_in = config.DATA_FILE
        
        self.ratio_data = ratio_data
        
        self.today = dt.date.today()
        
        self.LOWER_PERCENTILE = 25
        
        self.UPPER_PERCENTILE = 75
        
        self.NINETY_PERCENTILE = 90
        
        self._load_all_data()

    
    def _load_all_data(
        self
    ) -> None:
        """
        Load and align all external inputs from Excel.

        Loads
        -----
        - Sentiment Findings: ['ticker','avg_sentiment','mentions'] → index upper-cased
        - Model sheets: a map from human sheet names to internal model keys:
            'Prophet Pred'→'Prophet', 'Analyst Target'→'AnalystTarget', ...,
        each containing columns ['Ticker','Returns','SE'] (plus 'Current Price' for Analyst Target).
        - Analyst Data: a wide table with fundamentals and qualitative variables.
        - Latest prices from RatioData.
        - Per-ticker "Signal Scores" (last row) from `excel_file_in` ("Signal Scores" sheet).

        Returns
        -------
        None

        Notes
        -----
        - Ensures string/upper case tickers for joins.
        - Retains the last row of "Signal Scores" as a Series to avoid mixing regimes.
        """

        xls = pd.ExcelFile(self.excel_file_out)

        self.wsb = (
            xls.parse('Sentiment Findings',
                      usecols = ['ticker', 'avg_sentiment', 'mentions'],
                      index_col = 'ticker')
            .sort_index()
        )
        
        self.wsb.index = self.wsb.index.str.upper()

        model_sheets = {
            'Prophet Pred': 'Prophet',
            'Analyst Target': 'AnalystTarget',
            'Exponential Returns':'EMA',
            'Lin Reg Returns': 'LinReg',
            'DCF': 'DCF',
            'DCFE': 'DCFE',
            'Daily Returns': 'Daily',
            'RI': 'RI',
            'CAPM BL Pred': 'CAPM',
            'FF3 Pred': 'FF3',
            'FF5 Pred': 'FF5',
            'Factor Exponential Regression': 'FER',
            'SARIMAX Monte Carlo': 'SARIMAX',
            'Rel Val Pred': 'RelVal',
            'LSTM': 'LSTM',
            'GRU': 'GRU',
            'Advanced MC MIDAS': 'AdvMidas'
        }
        
        self.models: Dict[str, pd.DataFrame] = {}
        
        for sheet_name, name in model_sheets.items():
           
            cols = ['Ticker', 'Returns', 'SE'] + (
                ['Current Price'] if sheet_name == 'Analyst Target' else []
            )
           
            df = (
                xls.parse(sheet_name, usecols = cols, index_col = 0)
                .sort_index()
            )
           
            self.models[name] = df

        analyst_cols = [
            'Ticker', 
            'dividendYield', 
            'recommendationKey', 
            'sharesShort', 
            'sharesShortPriorMonth',
            'sharesOutstanding', 
            'beta', 
            'earningsGrowth', 
            'revenueGrowth', 
            'debtToEquity',
            'Return on Assets', 
            'returnOnEquity', 
            'priceToBook', 
            'trailingEps', 
            'forwardEps',
            'Gross Margin', 
            'Current Price', 
            'Low Price', 
            'numberOfAnalystOpinions',
            'Net Income', 
            'Operating Cash Flow', 
            'Previous Return on Assets',
            'Long Term Debt', 
            'Previous Long Term Debt', 
            'Current Ratio', 
            'Previous Current Ratio',
            'New Shares Issued', 
            'Previous Gross Margin', 
            'Asset Turnover',
            'Previous Asset Turnover', 
            'Insider Purchases', 
            'Avg EPS Estimate', 
            'marketCap',
            'totalRevenue', 
            'Avg Revenue Estimate'
        ]
        
        self.analyst_df = (
            xls.parse('Analyst Data', usecols=analyst_cols, index_col=0)
            .sort_index()
        )
        
        if self.analyst_df.index.dtype == object:
            
            self.analyst_df.index = self.analyst_df.index.str.upper()

        self.latest_prices = self.ratio_data.last_price
        
        self.tickers = config.tickers

        self.signal_scores = (
            pd.read_excel(self.excel_file_in,
                          sheet_name = "Signal Scores",
                          index_col = 0)
            .iloc[-1]
        )
        
    def short_score(
        self,
        shares_short: pd.Series,
        shares_outstanding: pd.Series,
        shares_short_prior: pd.Series
    ) -> pd.Series:
        """
        Compute a short-interest score using current/prior short shares and shares outstanding.

        Let for ticker t:
            s_t   = current shares short
            S_t   = shares outstanding
            s^-_t = prior shares short

        Define the short ratio:
            r_t = s_t / S_t

        Scoring rules
        -------------
        - Penalty if r_t is in the upper percentile (default 75th):
            score_t -= 1{ r_t ≥ Q_{0.75}(r_{t over non-zero}) }
        - Penalty if ratio increased by ≥ 5% relative to prior (where prior>0):
            score_t -= 1{ r_t ≥ 1.05 · (s^-_t / S_t) , s^-_t>0 }
        - Bonus if ratio decreased by ≥ 5%:
            score_t += 1{ r_t ≤ 0.95 · (s^-_t / S_t) }

        Parameters
        ----------
        shares_short, shares_outstanding, shares_short_prior : pd.Series
            Current short, outstanding shares, and prior short shares.

        Returns
        -------
        pd.Series (int)
            Additive score over the common index.
        """

        common_idx = (
            shares_short.index
            .intersection(shares_outstanding.index)
            .intersection(shares_short_prior.index)
        )
        
        curr = shares_short.reindex(common_idx).fillna(0)
        
        out = shares_outstanding.reindex(common_idx)
        
        prior = shares_short_prior.reindex(common_idx).fillna(0)

        ratio = curr.div(out).fillna(0)

        ratio_nz = ratio[ratio != 0].dropna()
        
        if ratio_nz.empty:
            
            return pd.Series(0.0, index = common_idx)

        upper = np.percentile(ratio_nz, self.UPPER_PERCENTILE)

        scores = pd.Series(0.0, index=common_idx)

        scores -= (ratio >= upper).astype(int)

        prior_ratio = prior.div(out).fillna(0)
        
        mask_prior_nz = prior > 0
        
        scores -= ((ratio >= 1.05 * prior_ratio) & mask_prior_nz).astype(int)

        scores += (ratio <= 0.95 * prior_ratio).astype(int)

        return scores.astype(int)


    def wsb_score(
        self, 
        base_score: pd.Series
        ) -> pd.Series:
        """
        Adjust a base per-ticker score using WSB sentiment and mention intensity.

        Definitions
        -----------
        - Positive: avg_sentiment > 0
        - Negative: avg_sentiment < 0
        - High mentions: mentions > 4
        - Very high: mentions > 10
        - Very positive/negative: thresholds at ±0.2

        Increments
        ----------
        For tickers present in the WSB sheet:
        - +1 for positive; +1 more if positive & high mentions
        - +1 if very positive & high
        - +1 if very positive & very high.
        - −1 for negative
        - −1 more if negative & high
        - −1 if very negative & high
        - −1 if very negative & very high.

        Parameters
        ----------
        base_score : pd.Series
            Baseline score to adjust.

        Returns
        -------
        pd.Series (int)
            Adjusted scores.
        """


        common = self.wsb.index.intersection(base_score.index)

        adjusted = base_score.copy()

        wsb_slice = self.wsb.loc[common]

        pos = wsb_slice['avg_sentiment'] > 0

        neg = wsb_slice['avg_sentiment'] < 0

        high_mentions = wsb_slice['mentions'] > 4
        
        high_high_mentions = wsb_slice['mentions'] > 10
        
        very_pos = wsb_slice['avg_sentiment'] > 0.2
        
        very_neg = wsb_slice['avg_sentiment'] < -0.2

        adjusted.loc[common] += pos.astype(int)

        adjusted.loc[common] += (pos & high_mentions).astype(int)
        
        adjusted.loc[common] += (very_pos & high_high_mentions).astype(int)
        
        adjusted.loc[common] += (very_pos & high_mentions).astype(int)

        adjusted.loc[common] -= neg.astype(int)

        adjusted.loc[common] -= (neg & high_mentions).astype(int)
        
        adjusted.loc[common] -= (very_neg & high_high_mentions).astype(int)
        
        adjusted.loc[common] -= (very_neg & high_mentions).astype(int)

        return adjusted
    

    def earnings_growth_score(
        self, 
        earnings_growth: pd.Series, 
        score: pd.Series, 
        ind_earnings_growth: pd.Series, 
        eps: pd.Series, 
        eps_pred: pd.Series
        ) -> pd.Series:
        """
        Adjust score using earnings growth (EG), industry EG baseline, and EPS surprises.

        Let
        - eg_t  = earnings_growth(t)
        - eg_ind_t = industry baseline for earnings growth
        - eps_t = trailing EPS, eps_pred_t = next-year EPS estimate

        Rules
        -----
        - +1 if eg_t > 0; −1 if eg_t < 0
        - +1 if eg_t ≥ Q_{0.75}(eg over non-zero)
        - +1 if eg_t > eg_ind_t
        - +1 if eps_t < eps_pred_t (implying expected improvement)
        - −1 if eps_t > eps_pred_t

        Returns
        -------
        pd.Series (int)
            Updated scores on the common index.
        """

        common_idx = earnings_growth.index.intersection(score.index)

        eg = earnings_growth.reindex(common_idx).fillna(0)
        
        eps = eps.reindex(common_idx).fillna(0)
        
        eps_pred = eps_pred.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        eg_high = np.percentile(eg[eg != 0].dropna(), self.UPPER_PERCENTILE)

        sc += (eg > 0).astype(int)

        sc -= (eg < 0).astype(int)

        sc += (eg > eg_high).astype(int)
        
        sc +=(eg > ind_earnings_growth).astype(int)
        
        sc += (eps < eps_pred).astype(int)
        
        sc -= (eps > eps_pred).astype(int)

        return sc


    def operating_cash_flow_score(
        self, 
        operating_cash_flow: pd.Series, 
        score: pd.Series
        ) -> pd.Series:
        """
        Adjust score based on operating cash flow (OCF).

        Rule
        ----
        +1 if OCF > 0.

        Parameters
        ----------
        operating_cash_flow : pd.Series
        score : pd.Series

        Returns
        -------
        pd.Series (int)
        """

        common_idx = operating_cash_flow.index.intersection(score.index)

        ocf = operating_cash_flow.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        sc += (ocf > 0).astype(int)

        return sc


    def revenue_growth_score(
        self, 
        rev_growth: pd.Series, 
        score: pd.Series, 
        ind_rvg: pd.Series, 
        rev: pd.Series, 
        rev_pred: pd.Series
        ) -> pd.Series:
        """
        Adjust score using revenue growth (RG) relative to an industry baseline and forecasts.

        Let
        - rg_t   = revenue_growth(t)
        - rg_ind_t = industry baseline
        - rev_t  = current total revenue
        - rev_pred_t = forecast revenue (e.g., next year)

        Rules
        -----
        - +1 if rg_t > rg_ind_t (on non-zero rg only), −1 if rg_t < rg_ind_t
        - +1 if rev_t < rev_pred_t (implying growth runway)
        - −1 if rev_t > rev_pred_t

        Returns
        -------
        pd.Series (int)
        """

        common_idx = rev_growth.index.intersection(score.index).intersection(ind_rvg.index)

        rg = rev_growth.reindex(common_idx).fillna(0)
        
        rev = rev.reindex(common_idx).fillna(0)
        
        rev_pred = rev_pred.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        iv = ind_rvg.reindex(common_idx).fillna(0)

        nonzero_idx = rg[rg != 0].index

        sc.loc[nonzero_idx] += (rg.loc[nonzero_idx] > iv.loc[nonzero_idx]).astype(int)

        sc.loc[nonzero_idx] -= (rg.loc[nonzero_idx] < iv.loc[nonzero_idx]).astype(int)
        
        sc += (rev < rev_pred).astype(int)
        
        sc -= (rev > rev_pred).astype(int)

        return sc


    def return_on_equity_score(
        self, 
        return_on_equity: pd.Series, 
        score: pd.Series, 
        ind_roe: pd.Series
        ) -> pd.Series:
        """
        Adjust score using Return on Equity (ROE) vs industry baseline.

        Rules (on non-zero ROE only)
        ----------------------------
        - +1 if ROE > industry ROE
        - −1 if ROE < industry ROE
        - +1 if ROE > 0

        Parameters
        ----------
        return_on_equity : pd.Series
        score : pd.Series
        ind_roe : pd.Series

        Returns
        -------
        pd.Series (int)
        """

        common_idx = return_on_equity.index.intersection(score.index).intersection(ind_roe.index)

        roe = return_on_equity.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        iro = ind_roe.reindex(common_idx).fillna(0)

        nonzero_idx = roe[roe != 0].index

        sc.loc[nonzero_idx] += (roe.loc[nonzero_idx] > iro.loc[nonzero_idx]).astype(int)

        sc.loc[nonzero_idx] -= (roe.loc[nonzero_idx] < iro.loc[nonzero_idx]).astype(int)
        
        sc.loc[nonzero_idx] += (roe.loc[nonzero_idx] > 0).astype(int)

        return sc
    
    
    def return_on_assets_score(
        self, 
        return_on_assets: pd.Series, 
        prev_roa : pd.Series,
        score: pd.Series, 
        ind_roa: pd.Series
        ) -> pd.Series:
        """
        Adjust score using Return on Assets (ROA) vs industry baseline and prior ROA.

        Let roa_t be current ROA; p_roa_t previous ROA; ind_roa_t industry benchmark.

        Rules (on non-zero roa)
        -----------------------
        - +1 if roa_t > 0; −1 if roa_t < 0
        - +1 if roa_t > ind_roa_t; −1 if roa_t < ind_roa_t
        - +1 if roa_t > p_roa_t; −1 if roa_t < p_roa_t

        Returns
        -------
        pd.Series (int)
        """

        common_idx = return_on_assets.index.intersection(score.index)

        roa = return_on_assets.reindex(common_idx).fillna(0)
        
        p_roa = prev_roa.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)
        
        nonzero_idx = roa[roa != 0].index
        
        sc.loc[nonzero_idx] += (roa.loc[nonzero_idx] > 0).astype(int)
        
        sc.loc[nonzero_idx] -= (roa.loc[nonzero_idx] < 0).astype(int)
        
        sc.loc[nonzero_idx] += (roa.loc[nonzero_idx] > ind_roa.loc[nonzero_idx]).astype(int)
        
        sc.loc[nonzero_idx] -= (roa.loc[nonzero_idx] < ind_roa.loc[nonzero_idx]).astype(int)
        
        sc.loc[nonzero_idx] += (roa.loc[nonzero_idx] > p_roa.loc[nonzero_idx]).astype(int)
        
        sc.loc[nonzero_idx] -= (roa.loc[nonzero_idx] < p_roa.loc[nonzero_idx]).astype(int)

        return sc


    def price_to_book_score(
        self, 
        price_to_book: pd.Series, 
        score: pd.Series, 
        ind_pb: pd.Series
        ) -> pd.Series:
        """
        Adjust score using Price-to-Book (P/B) vs industry baseline.

        Scope
        -----
        - Excludes tickers ending with '.L' (presumably UK listing with different accounting).

        Rules (on the filtered tickers)
        -------------------------------
        - +1 if 0 < P/B ≤ 1 (value condition)
        - −1 if P/B ≤ 0
        - +1 if P/B < industry P/B
        - −1 if P/B > industry P/B

        Parameters
        ----------
        price_to_book : pd.Series
        score : pd.Series
        ind_pb : pd.Series

        Returns
        -------
        pd.Series (int)
        """

        common_idx = price_to_book.index.intersection(score.index).intersection(ind_pb.index)

        pb = price_to_book.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        ipb = ind_pb.reindex(common_idx).fillna(0)

        mask = ~pb.index.astype(str).to_series().str.endswith('.L')

        local_idx = pb.index[mask]
        
        value_cond = (pb[mask] > 0) & (pb[mask] <= 1)
        
        sc.loc[local_idx] += value_cond.astype(int)

        sc.loc[local_idx] -= (pb[mask] <= 0).astype(int)

        sc.loc[local_idx] += (pb[mask] < ipb[mask]).astype(int)

        sc.loc[local_idx] -= (pb[mask] > ipb[mask]).astype(int)

        return sc


    def ep_score(
        self, 
        trailing_eps: pd.Series, 
        forward_eps: pd.Series, 
        price: pd.Series,
        score: pd.Series, 
        ind_pe: pd.Series,
        eps_1y: pd.Series
        ) -> pd.Series:
        """
        Adjust score using EPS-based valuation signals.

        Let
        - P_t    = current price
        - EPS_tr = trailing EPS
        - EPS_f  = forward EPS (next period)
        - P/E_tr = P_t / EPS_tr (∞ if EPS_tr = 0)
        - P/E_f  = P_t / EPS_f  (∞ if EPS_f = 0)
        - PE_ind = industry P/E benchmark

        Rules (excluding tickers ending '.L')
        -------------------------------------
        - −1 if P/E_tr > PE_ind (trailing valuation rich vs industry)
        - +1 if P/E_f < P/E_tr (forward multiple compressing)
        - −1 if P/E_f > P/E_tr (forward multiple expanding)

        Returns
        -------
        pd.Series (int)
        """

        slist = [trailing_eps, forward_eps, price, score, ind_pe]

        common_idx = slist[0].index

        for s in slist[1:]:

            common_idx = common_idx.intersection(s.index)

        teps, feps, pr, sc, ipe, eps_1y = [s.reindex(common_idx).fillna(0) for s in (trailing_eps, forward_eps, price, score, ind_pe, eps_1y)]

        with np.errstate(divide='ignore', invalid='ignore'):

            trailing_ratio = np.where(teps != 0, pr / teps, np.inf)

            forward_ratio = np.where(feps != 0, pr / feps, np.inf)
            
        trailing_series = pd.Series(trailing_ratio, index = common_idx)

        forward_series = pd.Series(forward_ratio, index = common_idx)
        
        mask_no_l = ~pd.Series(common_idx, index = common_idx).str.endswith('.L')

        valid_idx = common_idx[mask_no_l]

        sc.loc[valid_idx] -= (trailing_series[valid_idx] > ipe[valid_idx]).astype(int)

        sc.loc[valid_idx] += (forward_series[valid_idx] < trailing_series[valid_idx]).astype(int)
        
        sc.loc[valid_idx] -= (forward_series[valid_idx] > trailing_series[valid_idx]).astype(int)

        return sc
    
    
    def upside_downside_score(
        self,
        hist_rets: pd.DataFrame,
        bench_hist_rets: pd.Series,
        score: pd.Series
    ) -> pd.Series:
        """
        Adjust `score` using upside and downside capture ratios vs a benchmark.

        Definitions
        -----------
        Let r_t be the ticker weekly returns, b_t the benchmark weekly returns.
        Define subsets:
        U = { weeks where b_t > 0 }   (up weeks)
        D = { weeks where b_t < 0 }   (down weeks)

        A conventional (weekly) capture ratio can be defined as:
        UpCapture = mean(r_t | t∈U) / mean(b_t | t∈U)
        DownCapture = mean(r_t | t∈D) / mean(b_t | t∈D)
        (The helper `pa.capture_ratios` provides these numbers; exact formula may
        be CAGR-based internally, but monotonic with the above.)

        Rules
        -----
        Let up = UpCapture, down = DownCapture.
        - +1 for each of:
            (up > 1.5 & down < 0.5),
            (up > 1.0 & down < 1.0),
            (up > 1.0 & down < 0.5),
            (up > 1.5 & down < 1.0)
        - −1 for each of:
            (down > 1.0 & up < down),
            (down > 1.5 & up < down)

        Returns
        -------
        pd.Series (int)
            Adjusted score, aligned to `hist_rets.columns`.
        """

        def one_ticker_bonus(
            tkr
        ):

            p = hist_rets[tkr].dropna()

            b = bench_hist_rets.reindex(p.index).dropna()

            if p.empty or b.empty:
                
                return 0.0, 0.0

            caps = pa.capture_ratios(p, b)
            
            return caps['Upside Capture'], caps['Downside Capture']

        caps_df = (
            pd.DataFrame(
                {t: one_ticker_bonus(t) for t in hist_rets.columns},
                index=['Up', 'Down']
            ).T
        )

        up = caps_df['Up']
        down = caps_df['Down']

        cond1 = (up > 1.5) & (down < 0.5)
        cond2 = (up > 1.0) & (down < 1.0)
        cond3 = (up > 1.0) & (down < 0.5)
        cond4 = (up > 1.5) & (down < 1.0)
        
        malus1 = (down > 1.0) & (up < down)
        malus2 = (down > 1.5) & (up < down)

        adjusted = score.copy().reindex(hist_rets.columns).fillna(0)
        
        adjusted += cond1.astype(int)
        adjusted += cond2.astype(int)
        adjusted += cond3.astype(int)
        adjusted += cond4.astype(int)
        
        adjusted -= malus1.astype(int)
        adjusted -= malus2.astype(int)

        return adjusted


    def alpha_adjustments(
        self,
        hist_rets: pd.DataFrame,
        benchmark_ann_ret: float,
        comb_ret: pd.Series,
        bench_hist_rets: pd.Series,
        rf: float,
        periods_per_year: int
    ) -> pd.Series:
        """
        Compute per-ticker Jensen's alpha (annualized) and a predicted-alpha flag, then
        return additive adjustments.

        Model
        -----
        Standard Jensen alpha regression (per ticker, only on dates where the ticker
        has data):
        (r_p,t − r_f) = α + β · (r_b,t − r_f) + ε_t
        where r_p,t is the weekly asset return, r_b,t is the weekly benchmark, and r_f
        is the weekly risk-free rate. Annualized alpha is derived from the fitted α.

        We also compute a "predicted alpha" from the combined expected return `comb_ret`
        passed to `pa.jensen_alpha_r2` (library-specific; conceptually assesses whether
        forecast ER implies a positive alpha vs benchmark trajectory).

        Rules
        -----
        - alpha_adj += 1 if α > 0; alpha_adj −= 1 if α < 0
        - pred_alpha_adj −= 5 if predicted α < 0 (strong penalty)

        Parameters
        ----------
        hist_rets : pd.DataFrame
            Weekly returns per ticker (5y window suggested).
        benchmark_ann_ret : float
            Benchmark annualized return over a longer window (e.g., 5y).
        comb_ret : pd.Series
            Combined expected return per ticker.
        bench_hist_rets : pd.Series
            Benchmark weekly returns over the same dates.
        rf : float
            Weekly risk-free rate.
        periods_per_year : int
            52 for weekly data.

        Returns
        -------
        (pd.Series, pd.Series)
            (alpha_adj, pred_alpha_adj), each int and aligned to `hist_rets.columns`.

        Notes
        -----
        - When insufficient data (<2 obs) on a ticker, returns 0 for that ticker.
        """

        def one_ticker_alpha(
            tkr
        ):
            print(tkr)
            
            p = hist_rets[tkr].dropna()
            
            b = bench_hist_rets.reindex(p.index).dropna()
            
            c_r = comb_ret.loc[tkr]
            
            print('Combined Returns:', c_r)

            if len(p) < 2 or b.empty:
            
                return (np.nan, np.nan)

            alpha, _, pred_alpha = pa.jensen_alpha_r2(p, benchmark_ann_ret, c_r, b, rf, periods_per_year)
            
            print('Alpha:', alpha, 'Predicted Alpha:', pred_alpha)
            print('_________________________________________')
            
            return alpha, pred_alpha

        alpha_df = pd.DataFrame.from_dict(
            {t: one_ticker_alpha(t) for t in hist_rets.columns},
            orient = 'index',
            columns = ['alpha', 'pred_alpha']
        )
                
        alpha_adj = pd.Series(0, index = self.tickers, dtype = int)
        pred_alpha_adj = pd.Series(0, index = self.tickers, dtype = int)
        
        alpha_adj += (alpha_df['alpha'] > 0).astype(int)
        alpha_adj -= (alpha_df['alpha'] < 0).astype(int)
        
        pred_alpha_adj -= 5 * (alpha_df['pred_alpha'].fillna(0) < 0).astype(int)
        
        return alpha_adj.reindex(hist_rets.columns).fillna(0), pred_alpha_adj.reindex(hist_rets.columns).fillna(0)

        
    
    def compute_combination_forecast(
        self,
        region_indicators: Dict[str, pd.Series]
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, pd.Series], pd.DataFrame, pd.Index]:
        """
        Compute the combined expected returns, combined standard errors (SE), 
        final scores, per-model weights, a full score breakdown, and the aligned index.

        Steps / Math
        ------------
        1) **Gather model ER and SE**:
   
        For each model i, per-ticker ER_i(t) and SE_i(t) = σ_i(t).
    
        Clip:
            ER_i(t) ∈ [config.lbr, config.ubr]
            σ_i(t)  ∈ [MIN_STD, MAX_STD]

        2) **Validity & caps**:
     
        Valid if ER_i(t) is not in {−1, 0} and finite, and σ_i(t) > 0.
        Let M_t = #valid models for ticker t.
        Cap per-ticker per-model weight:
            cap_t = max(MAX_MODEL_WT, 1 / M_t)

        3) **Inverse-variance weights** (pre-cap):
        For each ticker t:
            w_i^raw(t) = 1 / σ_i^2(t)
            Normalize across i:
            \tilde{w}_i(t) = w_i^raw(t) / Σ_j w_j^raw(t)

        4) **Cap-and-renormalize** (per ticker):
      
        Apply per-component cap_t (and zero invalid models) and redistribute any deficit
        proportionally to residual room until column sums reach 1 (bounded iterative
        proportional fitting).

        5) **Group limits** (per ticker):
     
        Let groups be:
            HIST={Daily, EMA}, IV={DCF, DCFE, RI, RelVal},
            F={FF3, FF5, CAPM, FER}, ML={Prophet, SARIMAX, LinReg, LSTM, GRU, AdvMidas}.
      
        Enforce:
            sum(HIST) ≤ 0.10, sum(IV) ≤ 0.30, sum(F) ≤ 0.30, sum(ML) ≤ 0.45
       
        If a group exceeds its cap, scale it down to the cap and renormalize the remainder.

        6) **Combined expected return**:
      
        ER_comb(t) = div(t) + Σ_i w_i(t) · ER_i(t)

        7) **Combined variance** (within/between):
       
        Within(t)  = Σ_i w_i(t) · σ_i^2(t)
        Between(t) = (1/M_t) · Σ_i w_i(t) · (ER_i(t) − ER_comb(t))^2
        Var_comb(t)= Within(t) + Between(t)
        SE_comb(t) = sqrt( clip( Var_comb(t), MIN_STD^2, MAX_STD^2 ) )

        8) **Risk/return diagnostics** (last-year window):
      
        - Skewness (per-ticker) on 5y returns.
        - Sharpe ratio: SR = (μ − r_f) / σ with μ,σ computed on last-year weekly returns.
        - Sortino ratio: SoR = (μ − r_f) / σ_down, where σ_down is downside std.

        9) **Additive score breakdown**:
       
        Start with short-interest score, then sequentially add adjustments:
        WSB sentiment, earnings growth, revenue growth, ROE, ROA, P/B, EPS,
        upside/downside capture, analyst lower target vs price, recommendations,
        insider activity, profitability/cash-flow/debt/liquidity/issuance/margins/efficiency,
        skewness, Sharpe, Sortino, alpha flags, and technical "Signal Scores".
        
        Finally:
            FinalScore(t) = min( sum(row signals), numberOfAnalystOpinions(t) )

        Returns
        -------
        (comb_rets, comb_stds, final_scores, weights, score_breakdown, common_idx)
        where
        - comb_rets : pd.Series
        - comb_stds : pd.Series (SE)
        - final_scores : pd.Series (int)
        - weights : dict[str, pd.Series]  (per-model weight series summing to 1 per ticker)
        - score_breakdown : pd.DataFrame  (each column is an additive component)
        - common_idx : pd.Index (tickers used)

        Notes
        -----
        - All inputs are reindexed to a common universe of tickers with finite data.
        - Dividend yield (decimal) is added directly to ER_comb.
        - SEs and ERs are clipped to stabilize optimization and guard against data errors.
        """

        names = list(self.models.keys())

        rets = [self.models[n]['Returns'] for n in names]
        
        ses = [self.models[n]['SE'] for n in names]
        
        benchmark_ret, benchmark_weekly_rets, last_year_benchmark_weekly_rets = po.benchmark_rets(
                                                                                    benchmark = config.benchmark, 
                                                                                    start = config.FIVE_YEAR_AGO, 
                                                                                    end = config.TODAY, 
                                                                                    steps = 52
                                                                                )

        benchmark_ann_ret_5y = (1 + benchmark_weekly_rets).prod() ** 0.2 - 1
       
        a = self.analyst_df
       
        div = a['dividendYield'] / 100
       
        recommendation = a['recommendationKey']
       
        shares_short = a['sharesShort']
       
        shares_outstanding = a['sharesOutstanding']
       
        shares_short_prior = a['sharesShortPriorMonth']
       
        earnings_growth = a['earningsGrowth']
       
        rev_growth = a['revenueGrowth']
       
        roa = a['Return on Assets']
        prev_roa = a['Previous Return on Assets']
       
        roe = a['returnOnEquity']
       
        pb = a['priceToBook']
       
        teps = a['trailingEps']      
        feps = a['forwardEps']
       
        price = a['Current Price']
       
        lower_target = a['Low Price']
       
        insider_purchases = a['Insider Purchases']
       
        net_income = a['Net Income']
       
        operating_cashflow = a['Operating Cash Flow']
       
        prev_long_debt = a['Previous Long Term Debt']
        long_debt = a['Long Term Debt']
       
        prev_current_ratio = a['Previous Current Ratio']
        current_ratio = a['Current Ratio']
       
        shares_issued = a['New Shares Issued']
       
        gross_margin = a['Gross Margin']
       
        prev_gm = a['Previous Gross Margin']
       
        at = a['Asset Turnover']
        prev_at = a['Previous Asset Turnover']
       
        eps_1y = a['Avg EPS Estimate']
       
        rev = a['totalRevenue']
        rev_1y = a['Avg Revenue Estimate']
       
        nY = a['numberOfAnalystOpinions']

        ind_pe = region_indicators['PE']
        ind_pb = region_indicators['PB']
        ind_roe = region_indicators['ROE']
        ind_roa = region_indicators['ROA']
        ind_rvg = region_indicators['RevG']
       
        ind_eg = region_indicators.get('EarningsG', pd.Series(0, index=price.index))

        all_series = rets + ses + [
            div, 
            recommendation, 
            shares_short,
            shares_outstanding, 
            shares_short_prior, 
            earnings_growth,
            rev_growth, 
            roa, 
            roe,
            pb, 
            teps, 
            feps, 
            price, 
            lower_target,
            insider_purchases, 
            net_income, 
            operating_cashflow, 
            prev_long_debt,
            long_debt, 
            prev_current_ratio, 
            current_ratio, 
            shares_issued,
            gross_margin, 
            prev_gm, 
            at, 
            prev_at, 
            eps_1y, 
            nY,
            ind_pe, 
            ind_pb, 
            ind_roe, 
            ind_roa, 
            ind_rvg, 
            ind_eg
        ]
       
        common_idx = set(all_series[0].index)
       
        for s in all_series[1:]:
            
            common_idx &= set(s.index)
       
        common_idx = sorted(common_idx)

        rets = [r.reindex(common_idx).clip(lower = config.lbr, upper = config.ubr) for r in rets]
       
        ses = [s.reindex(common_idx).clip(lower = MIN_STD, upper = MAX_STD) for s in ses]

        ret_df = pd.DataFrame({names[i]: rets[i] for i in range(len(names))}, index=common_idx)
        
        ret_df_clipped = ret_df

        model_vars = pd.DataFrame(
            { names[i]: ses[i] ** 2 for i in range(len(names)) },
            index = common_idx
        )

        se_df = pd.DataFrame(
            { names[i]: ses[i] for i in range(len(names)) },
            index=common_idx
        )

        valid = (
            (~ret_df.isin([-1, 0])) & ret_df.notna()  
        ) & (
            (se_df > 0) & se_df.notna()              
        )
        
        model_counts = valid.sum(axis = 1).replace(0, np.nan)
        
        cap_per_ticker = np.maximum(MAX_MODEL_WT, 1.0 / model_counts).fillna(MAX_MODEL_WT)

        inv_var = 1.0 / model_vars
        inv_var = inv_var.where(valid, other = 0.0)

        tot_inv = inv_var.sum(axis = 1)

        raw_w = inv_var.div(tot_inv, axis = 0)

        def cap_norm(
            w_arr, 
            cap, 
            mask = None
        ):
            """
            w_arr: shape (n_models, n_tickers)
            cap:   shape (n_tickers,)  # per-ticker maximum weight
            mask:  shape (n_models, n_tickers)
            """

            final = np.minimum(w_arr, cap[np.newaxis, :])

            if mask is not None:
                
                final = np.where(mask, final, 0.0)

            for _ in range(1000):

                deficit = 1.0 - final.sum(axis=0)  

                if np.all(deficit <= 1e-8):
                    
                    break

                room = np.maximum(cap[np.newaxis, :] - final, 0.0)

                if mask is not None:
                    
                    room = np.where(mask, room, 0.0)

                for j, d in enumerate(deficit):
                  
                    if d <= 0 or room[:, j].sum() == 0:
                        
                        continue
                  
                    alloc = room[:, j]
                   
                    final[:, j] += d * alloc / alloc.sum()
                  
                    final[:, j] = np.minimum(final[:, j], cap[j])

                    if mask is not None:
                        
                        final[:, j] = np.where(mask[:, j], final[:, j], 0.0)

            return final

        
        w_arr = cap_norm(
            w_arr = raw_w.values.T,          
            cap = cap_per_ticker.values,     
            mask = valid.values.T
        )
        
        group_hist_names = ['Daily', 'EMA']
       
        group_iv_names = ['DCF', 'DCFE', 'RI', 'RelVal']
       
        group_f_names = ['FF3', 'FF5', 'CAPM', 'FER']
       
        group_ml_names = ['Prophet', 'SARIMAX', 'LinReg', 'LSTM', 'GRU', 'AdvMidas']

        group_hist_idx = [names.index(m) for m in group_hist_names]
       
        group_iv_idx = [names.index(m) for m in group_iv_names]
       
        group_f_idx = [names.index(m) for m in group_f_names]
       
        group_ml_idx = [names.index(m) for m in group_ml_names]
        
        hist_limit = 0.10
        
        iv_limit = 0.3
        
        f_limit = 0.3
        
        ml_limit = 0.45
        
        for col in range(w_arr.shape[1]):
            
            current_hist = w_arr[group_hist_idx, col]

            if current_hist.sum() > hist_limit:

                w_arr[group_hist_idx, col] *= hist_limit / current_hist.sum()
                
                others = [i for i in range(w_arr.shape[0]) if i not in group_hist_idx]
               
                other_sum = w_arr[others, col].sum()
               
                if other_sum > 0:
               
                    w_arr[others, col] *= (1 - hist_limit) / other_sum

            current_iv = w_arr[group_iv_idx, col]

            if current_iv.sum() > iv_limit:

                w_arr[group_iv_idx, col] *=  iv_limit / current_iv.sum()

                others = [i for i in range(w_arr.shape[0]) if i not in group_hist_idx]
               
                other_sum = w_arr[others, col].sum()
               
                if other_sum > 0:
               
                    w_arr[others, col] *= (1 - hist_limit) / other_sum

            current_f = w_arr[group_f_idx, col]

            if current_f.sum() > f_limit:

                w_arr[group_f_idx, col] *= f_limit / current_f.sum()

                others = [i for i in range(w_arr.shape[0]) if i not in group_hist_idx]
               
                other_sum = w_arr[others, col].sum()
               
                if other_sum > 0:
               
                    w_arr[others, col] *= (1 - hist_limit) / other_sum
            
            col_sum = w_arr[:, col].sum()
            
            if col_sum > 0:
            
                w_arr[:, col] /= col_sum

            current_ml = w_arr[group_ml_idx, col]

            if current_ml.sum() > ml_limit:

                w_arr[group_ml_idx, col] *= ml_limit / current_ml.sum()
                others = [i for i in range(w_arr.shape[0]) if i not in group_hist_idx]
               
                other_sum = w_arr[others, col].sum()
               
                if other_sum > 0:
               
                    w_arr[others, col] *= (1 - hist_limit) / other_sum
                    
        weights = {
            names[i]: pd.Series(w_arr[i], index = common_idx)
            for i in range(len(names))
        }

        comb_rets = div.reindex(common_idx).fillna(0)

        for n in names:
            
            comb_rets += weights[n] * ret_df_clipped[n]

        w_df = pd.DataFrame(weights)

        within_var = (w_df * model_vars).sum(axis = 1)

        between_var = (
            w_df * (ret_df_clipped.sub(comb_rets, axis = 0) ** 2)
        ).sum(axis = 1).div(model_counts)

        total_var = within_var + between_var

        comb_stds = np.sqrt(total_var.clip(lower = MIN_STD ** 2, upper = MAX_STD ** 2))
        
        last_year_ret = weekly_ret.loc[
            weekly_ret.index >= pd.to_datetime(config.YEAR_AGO), common_idx
        ]
        
        last_5y_ret = weekly_ret.loc[
            weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO), common_idx
        ]
        
        skewness = pa.skewness(
            r = last_5y_ret
        )
        
        last_year_period = len(last_year_ret)
        
        sharpe = pa.sharpe_ratio(
            r = last_year_ret, 
            periods_per_year = last_year_period
        )
        
        sortino = pa.sortino_ratio(
            returns = last_year_ret, 
            riskfree_rate = config.RF, 
            periods_per_year = last_year_period
        )
        
        score_base = self.short_score(
            shares_short = shares_short, 
            shares_outstanding = shares_outstanding, 
            shares_short_prior = shares_short_prior
        )
        
        score_ws = self.wsb_score(
            base_score = score_base.copy()
        )
        
        score_eg = self.earnings_growth_score(
            earnings_growth = earnings_growth, 
            score = score_ws.copy(), 
            ind_earnings_growth = ind_eg, 
            eps = teps, 
            eps_pred = eps_1y
        )
        
        score_rg = self.revenue_growth_score(
            rev_growth = rev_growth, 
            score = score_eg.copy(), 
            ind_rvg = ind_rvg, 
            rev = rev, 
            rev_pred = rev_1y
        )
        
        score_roe = self.return_on_equity_score(
            return_on_equity = roe, 
            score = score_rg.copy(), 
            ind_roe = ind_roe
        )
        
        score_roa = self.return_on_assets_score(
            return_on_assets = roa, 
            prev_roa = prev_roa,
            score = score_roe.copy(), 
            ind_roa = ind_roa
        )
        
        score_pb = self.price_to_book_score(
            price_to_book = pb, 
            score = score_roa.copy(), 
            ind_pb = ind_pb
        )
        
        score_eps = self.ep_score(
            trailing_eps = teps, 
            forward_eps = feps, 
            price = price, 
            score = score_pb.copy(), 
            ind_pe = ind_pe, 
            eps_1y = eps_1y
        )
        
        score_up_down = self.upside_downside_score(
            hist_rets = last_5y_ret, 
            bench_hist_rets = benchmark_weekly_rets, 
            score = score_eps.copy()
        )

        lower_target_adj = (lower_target > price).astype(int)

        rec_strong_buy_adj = pd.Series(
            np.where(recommendation == 'strong_buy', 3, 0), 
            index = common_idx
        )
        
        rec_hold_adj = pd.Series(
            np.where(recommendation == 'hold', -1, 0), 
            index = common_idx
        )
        
        rec_sell_adj = pd.Series(
            np.where(recommendation.isin(['sell','strong_sell']), -5, 0), 
            index = common_idx
        )
        
        insider_pos_adj = pd.Series(
            np.where(insider_purchases > 0, 2, 0), 
            index = common_idx
        )
        
        insider_neg_adj = pd.Series(
            np.where(insider_purchases < 0, -1, 0), 
            index = common_idx
        )
        
        net_income_adj = pd.Series(
            np.where(net_income > 0, 1, 0), 
            index = common_idx
        )
        
        ocf_adj = pd.Series(
            np.where(operating_cashflow > net_income, 1, 0), 
            index = common_idx
        )
        
        ld_adj = pd.Series(
            np.where(prev_long_debt > long_debt, 1, 0), 
            index = common_idx
        )
        
        cr_adj = pd.Series(
            np.where(prev_current_ratio < current_ratio, 1, 0), 
            index = common_idx
        )
        
        no_new_shares_adj = pd.Series(
            np.where(shares_issued <= 0, 1, 0), 
            index = common_idx
        )
        
        gm_adj = pd.Series(
            np.where(gross_margin > prev_gm, 1, 0),
            index = common_idx
        )
        
        at_adj = pd.Series(
            np.where(prev_at < at, 1, 0), 
            index = common_idx
        )
        
        skewness_adj = pd.Series(
            np.where(skewness > 0, 1, 0), 
            index = common_idx
        )
        
        sharpe_adj = pd.Series(
            np.where(sharpe > 1, 1, np.where(sharpe <= 0, -1, 0)),
            index = common_idx
        )
        
        sortino_adj = pd.Series(
            np.where(sortino > 1, 1, np.where(sortino <= 0, -1, 0)), 
            index = common_idx
        )
        
        alpha_adj, pred_alpha_adj = self.alpha_adjustments(
            hist_rets = last_5y_ret,
            benchmark_ann_ret= benchmark_ann_ret_5y,
            comb_ret = comb_rets,
            bench_hist_rets = benchmark_weekly_rets,
            rf = config.RF_PER_WEEK,
            periods_per_year = 52
        )
        
        score_breakdown = pd.DataFrame({
            'Short Score': score_base,
            'WSB Adjustment': score_ws - score_base,
            'Earnings Growth Adjustment': score_eg - score_ws,
            'Revenue Growth Adjustment': score_rg - score_eg,
            'Return on Equity Adjustment': score_roe - score_rg,
            'Return on Assets Adjustment': score_roa - score_roe,
            'Price-to-Book Adjustment': score_pb - score_roa,
            'EPS Adjustment': score_eps - score_pb,
            'Upside/Downside Adjustment': score_up_down - score_eps,
            'Lower Target Price': lower_target_adj,
            'Recommendation Strong Buy': rec_strong_buy_adj,
            'Recommendation Hold': rec_hold_adj,
            'Recommendation Sell/Strong Sell': rec_sell_adj,
            'Insider Purchases Positive': insider_pos_adj,
            'Insider Purchases Negative': insider_neg_adj,
            'Net Income Positive': net_income_adj,
            'OCF > Net Income': ocf_adj,
            'Long-Debt Improvement': ld_adj,
            'Current-Ratio Improvement': cr_adj,
            'No New Shares Issued': no_new_shares_adj,
            'Gross-Margin Improvement': gm_adj,
            'Asset-Turnover Improvement': at_adj,
            'Skewness': skewness_adj,
            'Sharpe Ratio': sharpe_adj,
            'Sortino Ratio': sortino_adj,
            'Alpha': alpha_adj,
            'Pred Alpha': pred_alpha_adj,
            'Signal Scores': self.signal_scores.reindex(common_idx).fillna(0)
        })

        final_scores = score_breakdown.sum(axis=1)
       
        final_scores = pd.Series(np.minimum(final_scores, nY), index=common_idx)
        
        if "SGLP.L" in final_scores.index:
            
            final_scores.loc["SGLP.L"] = final_scores.quantile(0.75)
       
        score_breakdown['Final Score'] = final_scores

        return comb_rets, comb_stds, final_scores, weights, score_breakdown, common_idx

    
def main():
    """
    Run the full portfolio forecast and export pipeline.

    Sequence
    --------
    1) Build `PortfolioOptimiser` with the Forecast workbook and RatioData r.
    
    2) Collect region/industry baselines from `RatioData().dicts()`:
    
    PE, PB, ROE, ROA, revenue growth (RevG), earnings growth (EarningsG).
    
    3) Compute combined ER/SE and final scores:
    (comb_rets, comb_stds, final_scores, weights, score_breakdown, idx)
    = optimiser.compute_combination_forecast(region_indicators)
    
    4) Create a covariance matrix via `shrinkage_covariance` using multi-horizon,
    factor/index/industry/sector inputs; annualize inside that function.
    
    5) Derive per-ticker annualized variance and std:
        var_i = Σ_{ii},  std_i = sqrt(var_i); clip to [MIN_STD, MAX_STD].
    
    6) Construct bull/bear return bands under a normal approximation:
        bull_i = clip( comb_rets_i + 1.96 · std_i, lbr, ubr )
        bear_i = clip( comb_rets_i − 1.96 · std_i, lbr, ubr )
    And corresponding price bands:
        HighPrice_i = Price_i · (1 + bull_i)
        LowPrice_i  = Price_i · (1 + bear_i)
        AvgPrice_i  = Price_i · (1 + comb_rets_i)
    
    7) Assemble the “Combination Forecast” sheet:
    - Current Price, Avg/Low/High Price, Returns/Low/High Returns, SE, Volatility
    - Each model’s weight (in %)
    - Final cross-sectional Score
    
    8) Assemble the “Score Breakdown” sheet (all additive components + Final Score).
    
    9) Export both sheets to `config.PORTFOLIO_FILE`.

    Outputs
    -------
    - Excel file `config.PORTFOLIO_FILE` with:
        * 'Combination Forecast'
        * 'Score Breakdown'

    Logging
    -------
    - Uses `logging.INFO` to trace progress.

    Notes
    -----
    - The covariance builder takes daily/weekly/monthly returns (5y windows) plus
    factor/index/industry/sector weekly returns. It blends multiple targets with
    convex weights and applies a PSD correction (see its own docstring).
    """
    
    logging.info('Loading data...')
    
    optimiser = PortfolioOptimiser(
        excel_file = config.FORECAST_FILE, 
        ratio_data = r
    )

    metrics = r.dicts()
    
    region_ind = {
        'PE': pd.Series([metrics['PE'][t]['Region-Industry'] for t in optimiser.tickers], index = optimiser.tickers),
        'PB': pd.Series([metrics['PB'][t]['Region-Industry'] for t in optimiser.tickers], index = optimiser.tickers),
        'ROE': pd.Series([metrics['ROE'][t]['Region-Industry'] for t in optimiser.tickers], index = optimiser.tickers),
        'ROA': pd.Series([metrics['ROA'][t]['Region-Industry'] for t in optimiser.tickers], index = optimiser.tickers),
        'RevG': pd.Series([metrics['rev1y'][t]['Region-Industry'] for t in optimiser.tickers], index = optimiser.tickers),
        'EarningsG': pd.Series([metrics['eps1y'][t]['Region-Industry'] for t in optimiser.tickers], index = optimiser.tickers)
    }

    logging.info('Computing combination forecast...')
    
    comb_rets, comb_stds, final_scores, weights, score_breakdown, common_idx = (
        optimiser.compute_combination_forecast(
            region_indicators = region_ind
        )
    )
    
    print(comb_rets)
    
    daily_ret_5y = daily_ret.loc[daily_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
  
    weekly_ret_5y = weekly_ret.loc[weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
  
    monthly_ret_5y = monthly_ret.loc[monthly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
    
    factor_weekly, index_weekly, ind_weekly, sec_weekly = r.factor_index_ind_sec_weekly_rets(
        merge = False
    )
        
    cov = shrinkage_covariance(
        daily_5y = daily_ret_5y, 
        weekly_5y = weekly_ret_5y, 
        monthly_5y = monthly_ret_5y, 
        comb_std = comb_stds, 
        common_idx = comb_rets.index,
        ff_factors_weekly = factor_weekly,
        index_returns_weekly = index_weekly,
        industry_returns_weekly = ind_weekly,
        sector_returns_weekly = sec_weekly,
        use_excess_ff = False
    )

    var = pd.Series(np.diag(cov), index = cov.index)
    
    std = np.sqrt(var).clip(lower = MIN_STD, upper = MAX_STD)

    idx = optimiser.latest_prices.index.sort_values()
    
    price = optimiser.latest_prices.reindex(idx)
    
    bull = (comb_rets + 1.96 * std).clip(config.lbr, config.ubr)
  
    bear = (comb_rets - 1.96 * std).clip(config.lbr, config.ubr)

    df = pd.DataFrame({
            'Ticker': idx,
            'Current Price': price,
            'Avg Price': np.round(price * (comb_rets + 1), 2),
            'Low Price': np.round(price * (bear + 1), 2),
            'High Price': np.round(price * (bull + 1), 2),
            'Returns': comb_rets,
            'Low Returns': bear,
            'High Returns': bull,
            'SE': comb_stds,
            'Volatility': std,
        }, index = idx).set_index('Ticker')
    
    for name, w in weights.items():
        
        df[f'{name} (%)'] = w.reindex(idx) * 100
        
    df['Score'] = final_scores.reindex(idx)

    df = ensure_headers_are_strings(
        df = df
    )
    
    score_breakdown = ensure_headers_are_strings(
        df = score_breakdown
    )
    
    sheets_to_upload = {
        'Combination Forecast': df,
        'Score Breakdown': score_breakdown
    }
    
    export_results(
        sheets = sheets_to_upload, 
        output_excel_file = config.PORTFOLIO_FILE
    )
    
    logging.info('Done.')
    

if __name__ == '__main__':

    main()

