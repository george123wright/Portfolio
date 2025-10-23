"""
Portfolio combination-forecast and cross-sectional scoring module.

Purpose
-------
This module builds a combined expected-return forecast per equity ticker from a
set of heterogeneous forecasters, derives a conservative uncertainty (standard
error) for that forecast, computes a suite of return/quality/valuation/
behavioural diagnostics, and assembles an additive score used for ranking. It
also prepares export-ready tables for downstream portfolio construction and
reporting.

High-level workflow 
-------------------
1) **Ingest**: Read model sheets (per-ticker ER and SE), analyst fundamentals,
   sentiment/mentions, dividend-yield predictions, and technical “Signal
   Scores”. 
   
   Latest prices and multi-horizon return panels are sourced from `RatioData`.

2) **Combine forecasters**: For each ticker t and model i with standard error
   σ_i(t) and expected return ER_i(t), form inverse-variance weights with
   per-ticker caps and per-group limits; add a dividend-yield component.

3) **Quantify uncertainty**: Construct a total variance composed of 

    (i) the within-model uncertainty, 
    
    (ii) a between-model dispersion term (to penalise forecaster disagreement), 
    
    (iii) dividend-yield uncertainty. 
    
    Bound the resulting standard error to a defensible range for numerical 
    stability.

4) **Diagnostics & scoring**: Compute asymmetry captures, Sharpe/Sortino,
   higher moments (skewness/kurtosis), profitability/quality, growth,
   valuation, analyst overlays, insider activity, and technical signals. 
   
   Each contributes an integer increment to a cross-sectional score
   
   The final score is the row sum, optionally capped by the number of analyst 
   opinions.

5) **Covariance**: Build a shrinkage covariance matrix from multi-horizon
   returns and factor/index/industry/sector structures for use in portfolio
   optimisation and risk reporting.

6) **Export**: Produce the “Combination Forecast” and “Score Breakdown” sheets,
   including model weights, combined returns and uncertainty bands, and the
   full score decomposition.

Notation
--------
Let i index models and t index tickers.

- Model inputs:

  ER_i(t)  : model i expected return for ticker t (clipped to [lbr, ubr])

  σ_i(t)   : model i standard error for ticker t (clipped to [MIN_STD, MAX_STD])

  M_t      : number of valid models for ticker t

  cap_t    : per-ticker per-model cap, cap_t = max(MAX_MODEL_WT, 1 / M_t)

- Weighting:

  w_i^raw(t)   = 1 / σ_i(t)^2

  w_i(t) = w_i^raw(t) / Σ_j w_j^raw(t)
  
  w_i(t) = Cap-and-renormalise( w_i(t), cap_t, group limits )
  
  Group constraints enforce 
  
    Σ_{i∈G} w_i(t) ≤ L_G 
    
for predefined groups (Historical, Intrinsic Value, Factor,
ML, and subfamilies).

- Combined expected return:

    ER_comb(t) = YMean_t + Σ_i w_i(t) · ER_i(t)

  where YMean_t is the predicted dividend yield (decimal).

- Combined variance (within/between + dividend-yield uncertainty):

  Var_within(t)  = Σ_i w_i(t) · σ_i(t)^2

  Var_between(t) = (1 / M_t) · Σ_i w_i(t) · (ER_i(t) − ER_comb(t))^2

  Var_total(t)   = Var_within(t) + Var_between(t) + YStd_t^2

  SE_comb(t)     = sqrt( clip(Var_total(t), MIN_STD^2, MAX_STD^2) )

Scoring framework
-----------------
Each indicator returns an integer increment Δ_k(t) ∈ { …, −2, −1, 0, 1, 2, … }.

The **Final Score** is:

  Score_final(t) = Σ_k Δ_k(t),

  and for all tickers except 'SGLP.L':

    Score_final(t) = min( Score_final(t), NOpinions_t )

A deterministic +1 bonus is applied to 'SGLP.L' in several fundamentals
functions via a single helper to keep behaviour consistent and auditable.

Indicators (summary)
--------------------
- **Short interest**: 

    penalise high short ratio and worsening trends
    
    reward improvement
    
- **Sentiment (WSB)**: 

    polarity × attention rules with escalating thresholds.

- **Growth**: 

    earnings and revenue growth vs industry baselines, plus EPS/revenue
    trajectory checks.

- **Quality**:

    ROE vs industry 
    
    sign ROA vs industry, sign
    
    momentum
    positive net income
    
    OCF > net income
    
    improving current ratio
    
    falling long debt
    
    improving gross margin and asset turnover.

- **Valuation**:

    P/B value screen and relative level vs industry
    
    EPS-based trailing/forward P/E compression vs industry benchmark (excluding '.L' tickers).

- **Asymmetry**:

    upside/downside capture (slopes and ratios)
    
    reward strong upside with muted downside; penalise downside dominance.
    
- **Risk-adjusted returns**: 

    Sharpe and Sortino ternarised around 0 and 1.
    
- **Higher moments**:

    skewness sign (±1/0) and kurtosis threshold (<3.5).
    
- **Alpha**: 

    sign of realised (Jensen) alpha
    
    strong penalty for negative
    
    predicted alpha implied by the forecast.
    
- **Technical**: 

    pass-through “Signal Scores”.

Uncertainty bands
-----------------
Given annualised volatility σ_t = sqrt(Σ_tt) from the shrinkage covariance, a
normal approximation is used to form 95% return bands:
  
  bull_t = clip( ER_comb(t) + 1.96 · σ_t , lbr , ubr )
  
  bear_t = clip( ER_comb(t) − 1.96 · σ_t , lbr , ubr )

Price bands follow as:

  HighPrice_t = Price_t · (1 + bull_t)

  LowPrice_t  = Price_t · (1 + bear_t)

  AvgPrice_t  = Price_t · (1 + ER_comb(t))

Data alignment and robustness
-----------------------------
All operations are performed on a common index of tickers with finite data.
Inputs are clipped to defensible ranges to guard against sheet errors and to
prevent single-model dominance. Group limits reduce style concentration risk.

Inputs and outputs
------------------
Inputs:

- Excel workbook with model sheets, sentiment, dividend yield predictions, and
  analyst panel.

- `RatioData`: prices and return matrices at daily/weekly/monthly frequencies,
  factor/index/industry/sector returns.

- Configuration in `config` (file paths, bounds, benchmark, risk-free rates).

Outputs:

- 'Combination Forecast' sheet: prices, combined returns, bands, SE/volatility,
  model weights (%), and final score.

- 'Score Breakdown' sheet: each component increment and the final score.

Assumptions and limitations
---------------------------
- The between-model dispersion term is a simple convex penalty and not a full
  hierarchical Bayesian treatment of forecaster disagreement.

- Group caps are exogenous policy parameters; alternative caps or hierarchical
  shrinkage could be substituted.

- Sharpe/Sortino assume a stationary distribution within the last-year window;
  scores use coarse thresholds for interpretability.

- P/E diagnostics treat zero EPS as infinite multiple; this preserves monotonic
  comparisons but can be conservative for deep value with volatile earnings.

Reproducibility
---------------
The pipeline is deterministic for a fixed input workbook and `RatioData`
snapshot. Any randomness is confined to upstream model sheets (outside this
module’s scope).

"""

import numpy as np
import pandas as pd
import datetime as dt
import logging
from typing import Dict, Tuple

from ratio_data import RatioData
from export_forecast import export_results
from functions.cov_functions import shrinkage_covariance
import Optimiser.portfolio_functions as pf
import Optimiser.Port_Optimisation as po
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
       
        w_i ∝ 1 / σ_i^2      (i indexes models available for that ticker),
       
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
        Initialise the portfolio optimiser with file locations, a data provider, and
        scoring parameters.

        Parameters
        ----------
        excel_file : str
            Path to the workbook containing model forecasts and the analyst/sentiment
            sheets (the “Forecast” file).
        ratio_data : RatioData
            Provider of latest prices, return panels, and factor/index/industry/sector
            returns used later for diagnostics and covariance construction.

        Side Effects
        ------------
        - Stores file paths, the date stamp (`today`), and percentile cut-offs used by
        score rules (25th, 75th, and 90th).
        
        - Immediately calls `_load_all_data()` which loads and aligns all inputs, and
        caches them into instance attributes.

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
        Load, align, and cache all external inputs from Excel and `RatioData`.

        Inputs (Excel)
        --------------
        - 'Sentiment Findings':
            Columns: ['ticker', 'avg_sentiment', 'mentions'].
            The index is uppercase ticker symbols to ensure join compatibility.
       
        - Model sheets, mapped to internal model names:
            • 'Prophet Pred' → 'Prophet'
            • 'Prophet PCA' → 'ProphetPCA'
            • 'Analyst Target' → 'AnalystTarget'
            • 'Exponential Returns' → 'EMA'
            • 'DCF' → 'DCF'
            • 'DCFE' → 'DCFE'
            • 'Daily Returns' → 'Daily'
            • 'RI' → 'RI'
            • 'CAPM BL Pred' → 'CAPM'
            • 'FF3 Pred' → 'FF3'
            • 'FF5 Pred' → 'FF5'
            • 'Factor Exponential Regression' → 'FER'
            • 'SARIMAX Monte Carlo' → 'SARIMAX'
            • 'Rel Val Pred' → 'RelVal'
            • 'LSTM_DirectH' → 'LSTM_DirectH'
            • 'LSTM_Cross_Asset' → 'LSTM_Cross_Asset'
            • 'GRU_cal' → 'GRU_cal'
            • 'GRU_raw' → 'GRU_raw'
            • 'Advanced MC' → 'AdvMC'
            • 'HGB Returns' → 'HGB Returns'
            • 'HGB Returns CA' → 'HGB Returns CA'
            • 'TVP + GARCH Monte Carlo' → 'TVP + GARCH Monte Carlo'
            • 'SARIMAX Factor' → 'SARIMAX Factor'

        Expected columns per model: ['Ticker', 'Returns', 'SE'].
        'Analyst Target' also supplies 'Current Price'.

        - 'Div Yield Pred':
       
        Columns: ['Ticker', 'Yield Mean', 'Yield Std'] capturing expected dividend
        yield (decimal) and its uncertainty per ticker.

        - 'Analyst Data':
       
        A wide panel with fundamental and qualitative variables used by the
        scoring functions, including growth, profitability, leverage, liquidity,
        margins, efficiency, EPS, revenue forecasts, targets, recommendations,
        insider activity, and opinion count.

        Inputs (RatioData)
        ------------------
        - Latest close prices, daily/weekly returns, and factor/index/industry/sector
        returns accessed elsewhere.

        Returns
        -------
        None

        Guarantees
        ----------
        - Ticker symbols on all loaded frames/series are upper-cased and sorted.
       
        - Signal Scores are loaded from the last row of the “Signal Scores” sheet in
        `self.excel_file_in` and held as a 1-row Series for direct reindexing.

        Notes
        -----
        - All later computations reindex to a common universe to ensure arithmetic is
        performed on aligned arrays only.
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
            "Prophet PCA": 'ProphetPCA',
            'Analyst Target': 'AnalystTarget',
            'Exponential Returns':'EMA',
            'DCF': 'DCF',
            'DCFE': 'DCFE',
            'Daily Returns': 'Daily',
            'RI': 'RI',
            'CAPM BL Pred': 'CAPM',
            'FF3 Pred': 'FF3',
            'FF5 Pred': 'FF5',
            'Factor Exponential Regression': 'FER',
            'Gordon Growth Model': 'Gordon Growth Model',
            'SARIMAX Monte Carlo': 'SARIMAX',
            'Rel Val Pred': 'RelVal',
            'LSTM_DirectH': 'LSTM_DirectH',
            'LSTM_Cross_Asset': 'LSTM_Cross_Asset',
            'GRU_cal': 'GRU_cal',
            'GRU_raw': 'GRU_raw',
            'Advanced MC': 'AdvMC',
            'HGB Returns': 'HGB Returns',
            'HGB Returns CA': 'HGB Returns CA',
            'TVP + GARCH Monte Carlo': 'TVP + GARCH Monte Carlo',
            'SARIMAX Factor': 'SARIMAX Factor',
        }
        
        self.models: Dict[str, pd.DataFrame] = {}
         
        self.dividend_yield = (
            xls.parse(
                sheet_name = 'Div Yield Pred',
                usecols = ['Ticker', 'Yield Mean', 'Yield Std'],
                index_col = 'Ticker'
            )
            .sort_index()
        )

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
            'Avg Revenue Estimate',
            'Tax Rate'
        ]
        
        self.analyst_df = (
            xls.parse('Analyst Data', usecols=analyst_cols, index_col = 0)
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


    def convert_series_to_base(
        self,
        series_rets: pd.Series,
        *,
        series_ccy: str,
        base_ccy: str,
        interval: str,
    ) -> pd.Series:
        """
        Convert a single return series in `series_ccy` to base_ccy using the same formula.
        """
       
        if series_ccy == base_ccy:
       
            return series_rets
       
        fx_map = r.get_fx_price_by_pair_local_to_base(
            country_to_pair = {
                series_ccy: f"{series_ccy}{base_ccy}"
            },
            base_ccy = base_ccy,
            interval = interval,
        )
        
        fx_price = fx_map[f"{series_ccy}{base_ccy}"].reindex(series_rets.index).ffill().bfill()
       
        fx_ret = fx_price.pct_change().reindex(series_rets.index).fillna(0.0)
       
        return (1.0 + series_rets).mul(1.0 + fx_ret).sub(1.0)

    
    def _apply_sglp_bonus(
        self, 
        adj: pd.Series
    ) -> pd.Series:
        """
        Apply a deterministic +1 bonus to SGLP.L if present in the Series.

        Parameters
        ----------
        adj : pd.Series
            Any additive signal series indexed by tickers.

        Returns
        -------
        pd.Series
            The input `adj` coerced to integer dtype with +1 added at index 'SGLP.L'
            (no effect if not present). Missing values are treated as 0 before
            adjustment.

        Raises
        ------
        TypeError
            If `adj` is not a pandas Series.

        Use Case
        --------
        This centralises a special-case scoring preference for SGLP.L, keeping the
        per-metric functions uncluttered and ensuring consistent handling. This is in
        order to not discriminate against SGLP.L in scoring, due to it not containing
        financial data as it is an ETC rather than a stock.
        """
        
        if not isinstance(adj, pd.Series):
        
            raise TypeError("_apply_sglp_bonus expects a pandas Series.")
        
        s = adj.fillna(0).astype(int).copy()
        
        if 'SGLP.L' in s.index:
        
            s.loc['SGLP.L'] = s.loc['SGLP.L'] + 1
        
        return s
        
        
    def short_score(
        self,
        shares_short: pd.Series,
        shares_outstanding: pd.Series,
        shares_short_prior: pd.Series
    ) -> pd.Series:
        """
        Compute a short score using current/prior short shares and shares outstanding.

        Let for ticker t:
         
            s_t   = current shares short
         
            S_t   = shares outstanding
         
            s^-_t = prior shares short

        Define the short ratio:
         
            r_t = s_t / S_t

        Scoring Logic
        -------------
        The function builds an integer increment series `Δ_t` with three components:
        
        1) High short ratio penalty:
        
            Δ_t ← Δ_t − 1 if r_t ≥ Q_{0.75}({r_u : r_u > 0})   (upper-quartile threshold
       
        computed over the non-zero cross-section).
        
        2) Worsening short ratio penalty:
            
            Δ_t ← Δ_t − 1 if r_t ≥ 1.05 × (s^-_t / S_t) and s^-_t > 0  (≥ +5% relative rise).
        
        3) Improving short ratio bonus:
            
            Δ_t ← Δ_t + 1 if r_t ≤ 0.95 × (s^-_t / S_t)                (≥ −5% relative fall).

        Parameters
        ----------
        shares_short : pd.Series
            Current shorted shares.
        shares_outstanding : pd.Series
            Current shares outstanding.
        shares_short_prior : pd.Series
            Previous-period shorted shares (same scale as `shares_short`).

        Returns
        -------
        pd.Series (int)
            Additive score aligned to the intersection of provided indices. Returns 0
            if all short ratios are zero or missing.

        Notes
        -----
        - Ratios undefined because of missing denominators are treated as 0.
       
        - The percentile is computed on non-zero ratios to avoid concentrated zeros
        collapsing the threshold.
        """

        common = shares_short.index.intersection(shares_outstanding.index).intersection(shares_short_prior.index)
       
        curr = shares_short.reindex(common).fillna(0)
       
        out = shares_outstanding.reindex(common)
       
        prior = shares_short_prior.reindex(common).fillna(0)

        ratio = curr.div(out).fillna(0)
       
        nonzero = ratio[ratio != 0].dropna()
       
        if nonzero.empty:
       
            return pd.Series(0, index = common, dtype = int)

        upper = np.percentile(nonzero, self.UPPER_PERCENTILE)
       
        prior_ratio = prior.div(out).fillna(0)
       
        inc = pd.Series(0, index = common, dtype = int)

        inc -= (ratio >= upper).astype(int)
       
        mask_prior_nz = prior > 0
       
        inc -= ((ratio >= 1.05 * prior_ratio) & mask_prior_nz).astype(int)
       
        inc += (ratio <= 0.95 * prior_ratio).astype(int)
       
        return inc.astype(int)


    def wsb_score(
        self, 
        index: pd.Index
        ) -> pd.Series:
        """
        Construct an additive WSB sentiment/attention signal.

        Indicator Construction
        ----------------------
        Inputs (for a subset of tickers present in the 'Sentiment Findings' sheet):
      
        - Average sentiment `S_t` (continuous; positive vs negative polarity).
      
        - Mention count `M_t` (attention proxy).

        Thresholds
        ----------
        - Positive sentiment: S_t > 0
       
        - Negative sentiment: S_t < 0
       
        - Very positive: S_t > 0.2
       
        - Very negative: S_t < −0.2
       
        - High attention: M_t > 4
       
        - Very high attention: M_t > 10

        Scoring Rules
        -------------
        For tickers with sentiment data, the increment is the sum of the following
        terms (each contributes ±1):

        Positive cases:

        - +1 if S_t > 0

        - +1 if (S_t > 0) and (M_t > 4)

        - +1 if (S_t > 0.2) and (M_t > 4)

        - +1 if (S_t > 0.2) and (M_t > 10)

        Negative cases:

        - −1 if S_t < 0

        - −1 if (S_t < 0) and (M_t > 4)

        - −1 if (S_t < −0.2) and (M_t > 4)

        - −1 if (S_t < −0.2) and (M_t > 10)

        Parameters
        ----------
        index : pd.Index
            The desired output universe. The method will return zeros for tickers with
            no sentiment data.

        Returns
        -------
        pd.Series (int)
            Additive sentiment/attention adjustment on `index`.
        """

        common = self.wsb.index.intersection(index)
      
        adj = pd.Series(0, index = index, dtype = int)
      
        if common.empty:
      
            return adj
      
        w = self.wsb.loc[common]
      
        pos = w['avg_sentiment'] > 0
      
        neg = w['avg_sentiment'] < 0
      
        hi = w['mentions'] > 4
        
        vhi = w['mentions'] > 10
        
        vpos = w['avg_sentiment'] > 0.2
        
        vneg = w['avg_sentiment'] < -0.2

        add = pos.astype(int)
        
        add += (pos & hi).astype(int)
        
        add += (vpos & hi).astype(int)
            
        add += (vpos & vhi).astype(int)
            
        add -= neg.astype(int)
            
        add -= (neg & hi).astype(int)
        
        add -= (vneg & hi).astype(int)
            
        add -= (vneg & vhi).astype(int)

        adj.loc[common] = add

        return adj
    

    def earnings_growth_score(
        self, 
        earnings_growth: pd.Series,
        ind_earnings_growth: pd.Series,
        eps: pd.Series,
        eps_pred: pd.Series
        ) -> pd.Series:
        """
        Score earnings growth (EG) against absolute and relative benchmarks, and EPS
        trajectory.

        Definitions
        -----------
        For ticker t:
        
        - Reported/estimated earnings growth: EG_t
        
        - Industry/regional baseline for earnings growth: EG*_t
        
        - Trailing EPS: EPS_tr,t
        
        - Forward (1y) EPS estimate: EPS^f_t

        Thresholds and Statistics
        -------------------------
        
        - Positive/negative growth: EG_t > 0 vs EG_t < 0
        
        - High growth cut-off: EG_t ≥ Q_{0.75}({EG_u : EG_u ≠ 0}) computed cross-sectionally
        on non-zero EG to avoid undue compression at 0.

        Scoring Rules
        -------------
        Let Δ_t be the additive increment:

        1) Absolute growth sign:
        
        - Δ_t ← Δ_t + 1 if EG_t > 0
        
        - Δ_t ← Δ_t − 1 if EG_t < 0

        2) High growth tail:

        - Δ_t ← Δ_t + 1 if EG_t > Q_{0.75} (non-zero cross-section)

        3) Relative growth vs industry:

        - Δ_t ← Δ_t + 1 if EG_t > EG*_t

        4) EPS improvement proxy:

        - Δ_t ← Δ_t + 1 if EPS_tr,t < EPS^f_t  (expected improvement)

        - Δ_t ← Δ_t − 1 if EPS_tr,t > EPS^f_t  (expected deterioration)

        Parameters
        ----------
        earnings_growth : pd.Series
            Reported/estimated earnings growth measure (consistent scale across names).
        ind_earnings_growth : pd.Series
            Matched industry/region baseline for earnings growth.
        eps : pd.Series
            Trailing EPS.
        eps_pred : pd.Series
            Forward (1y) EPS estimate.

        Returns
        -------
        pd.Series (int)
            Additive score per ticker, with a +1 adjustment applied to 'SGLP.L' if present.

        Notes
        -----
        The high-growth quartile is computed over the cross-section excluding zeros to
        reflect genuine positive growth rather than missing or flat values.
        """

        idx = earnings_growth.index.intersection(ind_earnings_growth.index).intersection(eps.index).intersection(eps_pred.index)
        
        eg = earnings_growth.reindex(idx).fillna(0)
        
        eg_nonzero = eg[eg != 0].dropna()
        
        if not eg_nonzero.empty:
            
            
            eg_hi = np.percentile(eg_nonzero, self.UPPER_PERCENTILE) 
            
        else:
            
            eg_hi = np.inf
        
        adj = (eg > 0).astype(int) - (eg < 0).astype(int)
        
        adj += (eg > eg_hi).astype(int)
        
        adj += (eg > ind_earnings_growth.reindex(idx).fillna(0)).astype(int)
        
        adj += (eps.reindex(idx).fillna(0) < eps_pred.reindex(idx).fillna(0)).astype(int)
        
        adj -= (eps.reindex(idx).fillna(0) > eps_pred.reindex(idx).fillna(0)).astype(int)
        
        return self._apply_sglp_bonus(
            adj = adj
        )


    def operating_cash_flow_score(
        self, 
        operating_cash_flow: pd.Series, 
        ) -> pd.Series:
        """
        Award a profitability/quality point when operating cash flow is positive.

        Indicator
        ---------
        Operating cash flow (OCF), typically defined as cash generated from core
        operations prior to financing/investing flows.

        Scoring Rule
        ------------
        Δ_t = 1{ OCF_t > 0 }

        Parameters
        ----------
        operating_cash_flow : pd.Series
            Operating cash flow per ticker (consistent units).

        Returns
        -------
        pd.Series (int)
            +1 where OCF is strictly positive; 0 otherwise, with an additional +1 for
            'SGLP.L' if present.
        """

        v = operating_cash_flow.fillna(0)
        
        adj = (v > 0).astype(int).reindex(v.index).astype(int)

        return self._apply_sglp_bonus(
            adj = adj
        )
        

    def revenue_growth_score(
        self,
        rev_growth: pd.Series,
        ind_rvg: pd.Series,
        rev: pd.Series,
        rev_pred: pd.Series
        ) -> pd.Series:
        """
        Score revenue growth against an industry baseline and forecast trajectory.

        Definitions
        -----------
        For ticker t:
       
        - Revenue growth: RG_t
       
        - Industry baseline: RG*_t
       
        - Current total revenue: REV_t
       
        - Forward (1y) revenue estimate: REV^f_t

        Scoring Rules
        -------------
       
        1) Relative revenue growth (only where RG_t ≠ 0):
       
        - +1 if RG_t > RG*_t
       
        - −1 if RG_t < RG*_t

        2) Forward revenue trajectory:
       
        - +1 if REV_t < REV^f_t  (implied growth runway)
       
        - −1 if REV_t > REV^f_t

        Parameters
        ----------
        rev_growth : pd.Series
            Reported/estimated revenue growth.
        ind_rvg : pd.Series
            Industry/regional baseline for revenue growth.
        rev : pd.Series
            Current revenue.
        rev_pred : pd.Series
            Forward revenue estimate.

        Returns
        -------
        pd.Series (int)
            Additive adjustment with a deterministic +1 for 'SGLP.L' if present.

        Notes
        -----
        The relative comparison is restricted to non-zero RG to prevent awarding/penalising
        flat or undefined growth rates.
        """

        idx = rev_growth.index.intersection(ind_rvg.index).intersection(rev.index).intersection(rev_pred.index)
        
        rg = rev_growth.reindex(idx).fillna(0)
        
        iv = ind_rvg.reindex(idx).fillna(0)
        
        adj = pd.Series(0, index = idx, dtype = int)
        
        nz = rg != 0
        
        adj.loc[nz] += (rg.loc[nz] > iv.loc[nz]).astype(int)
        
        adj.loc[nz] -= (rg.loc[nz] < iv.loc[nz]).astype(int)
        
        adj += (rev.reindex(idx).fillna(0) < rev_pred.reindex(idx).fillna(0)).astype(int)
        
        adj -= (rev.reindex(idx).fillna(0) > rev_pred.reindex(idx).fillna(0)).astype(int)
        
        return self._apply_sglp_bonus(
            adj = adj
        )
        
        
    def return_on_equity_score(
        self, 
        roe: pd.Series, 
        ind_roe: pd.Series
        ) -> pd.Series:
        """
        Score Return on Equity (ROE) against industry baseline and absolute sign.

        Definitions
        -----------
        For ticker t:
        
        - Return on equity: ROE_t
        
        - Industry baseline: ROE*_t

        Scoring Rules (only where ROE_t ≠ 0)
        ------------------------------------
        
        - +1 if ROE_t > ROE*_t
        
        - −1 if ROE_t < ROE*_t
        
        - +1 if ROE_t > 0  (profitable equity)

        Parameters
        ----------
        roe : pd.Series
            ROE measure (e.g., trailing twelve months).
        ind_roe : pd.Series
            Industry/regional ROE benchmark.

        Returns
        -------
        pd.Series (int)
            Per-ticker additive score, plus +1 for 'SGLP.L' if present.
        """

        idx = roe.index.intersection(ind_roe.index)
        
        r = roe.reindex(idx).fillna(0)
        
        iro = ind_roe.reindex(idx).fillna(0)
        
        adj = pd.Series(0, index = idx, dtype = int)
       
        nz = r != 0
       
        adj.loc[nz] += (r.loc[nz] > iro.loc[nz]).astype(int)
       
        adj.loc[nz] -= (r.loc[nz] < iro.loc[nz]).astype(int)
       
        adj.loc[nz] += (r.loc[nz] > 0).astype(int)
        
        return self._apply_sglp_bonus(
            adj = adj
        )    
        
    
    def return_on_assets_score(
        self, 
        roa: pd.Series, 
        prev_roa : pd.Series,
        ind_roa: pd.Series
        ) -> pd.Series:
        """
        Score Return on Assets (ROA) against industry baseline and prior value.

        Definitions
        -----------
        For ticker t:
      
        - Current ROA: ROA_t
      
        - Prior ROA: ROA^-_t
      
        - Industry baseline: ROA*_t

        Scoring Rules (only where ROA_t ≠ 0)
        ------------------------------------
        - Sign:
      
            +1 if ROA_t > 0
            
            −1 if ROA_t < 0
       
        - Relative to industry:
            
            +1 if ROA_t > ROA*_t
            
            −1 if ROA_t < ROA*_t
        
        - Momentum:
            
            +1 if ROA_t > ROA^-_t
            
            −1 if ROA_t < ROA^-_t

        Parameters
        ----------
        roa : pd.Series
            Current return on assets.
        prev_roa : pd.Series
            Prior-period return on assets (same definition).
        ind_roa : pd.Series
            Industry baseline for ROA.

        Returns
        -------
        pd.Series (int)
            Additive score with 'SGLP.L' bonus (+1) when present.
        """

        idx = roa.index.intersection(prev_roa.index).intersection(ind_roa.index)
        
        r = roa.reindex(idx).fillna(0)
        
        p = prev_roa.reindex(idx).fillna(0)
        
        ir = ind_roa.reindex(idx).fillna(0)
        
        adj = pd.Series(0, index = idx, dtype = int)
        
        nz = r != 0
        
        adj.loc[nz] += (r.loc[nz] > 0).astype(int)
        
        adj.loc[nz] -= (r.loc[nz] < 0).astype(int)
        
        adj.loc[nz] += (r.loc[nz] > ir.loc[nz]).astype(int)
        
        adj.loc[nz] -= (r.loc[nz] < ir.loc[nz]).astype(int)
        
        adj.loc[nz] += (r.loc[nz] > p.loc[nz]).astype(int)
        
        adj.loc[nz] -= (r.loc[nz] < p.loc[nz]).astype(int)

        return self._apply_sglp_bonus(
            adj = adj
        )
        

    def price_to_book_score(
        self, 
        pb: pd.Series, 
        ind_pb: pd.Series
        ) -> pd.Series:
        """
        Score valuation using Price-to-Book (P/B), excluding tickers ending '.L'.

        Definitions
        -----------
        For ticker t:
       
        - Price-to-book: PB_t
       
        - Industry baseline: PB*_t

        Universe Filter
        ---------------
        Tickers whose symbol ends with '.L' are excluded from this indicator to avoid
        cross-market accounting and reporting distortions.

        Scoring Rules (on the filtered universe)
        ----------------------------------------
        - Value condition:
            
            +1 if 0 < PB_t ≤ 1
        
            −1 if PB_t ≤ 0 (non-sensical / distressed)
        
        - Relative valuation:
            
            +1 if PB_t < PB*_t
        
            −1 if PB_t > PB*_t

        Parameters
        ----------
        pb : pd.Series
            Price-to-book ratio.
        ind_pb : pd.Series
            Industry baseline for P/B.

        Returns
        -------
        pd.Series (int)
            Additive score on the full input index (zeros for '.L' tickers), then +1
            for 'SGLP.L' if present.

        Notes
        -----
        The P/B ≤ 0 rule catches pathological denominators (negative or near-zero book
        value), which is commonly treated as a warning sign for simple ratio screens.
        """

        idx = pb.index.intersection(ind_pb.index)
        
        p = pb.reindex(idx).fillna(0)
        
        ipb = ind_pb.reindex(idx).fillna(0)
        
        adj = pd.Series(0, index = idx, dtype = int)
        
        mask_arr = (~pd.Series(idx, index=idx).astype(str).str.endswith('.L')).to_numpy()
        
        local = pd.Index(idx)[mask_arr]
        
        val = (p.loc[local] > 0) & (p.loc[local] <= 1)
        
        adj.loc[local] += val.astype(int)
        
        adj.loc[local] -= (p.loc[local] <= 0).astype(int)
        
        adj.loc[local] += (p.loc[local] < ipb.loc[local]).astype(int)
        
        adj.loc[local] -= (p.loc[local] > ipb.loc[local]).astype(int)

        return self._apply_sglp_bonus(
            adj = adj
        )
        

    def ep_score(
        self, 
        trailing_eps: pd.Series, 
        forward_eps: pd.Series, 
        price: pd.Series,
        ind_pe: pd.Series,
        eps_1y: pd.Series
        ) -> pd.Series:
        """
        Score EPS-based valuation diagnostics using trailing and forward P/E, excluding
        tickers ending '.L'.

        Definitions
        -----------
        For ticker t:
      
        - Price: P_t
      
        - Trailing EPS: EPS_tr,t
      
        - Forward (1y) EPS: EPS^f_t
      
        - Industry forward/trailing P/E benchmark: PE*_t
      
        - Trailing P/E: PE_tr,t = P_t / EPS_tr,t    (∞ if EPS_tr,t = 0)
      
        - Forward P/E:  PE_f,t  = P_t / EPS^f_t     (∞ if EPS^f_t  = 0)

        Scoring Rules (excluding tickers ending '.L')
        ---------------------------------------------
        - Relative valuation:
      
            −1 if PE_tr,t > PE*_t    (rich vs industry on trailing)
        
        - Multiple compression/expansion:
            
            +1 if PE_f,t < PE_tr,t    (compression)
        
            −1 if PE_f,t > PE_tr,t    (expansion)

        Parameters
        ----------
        trailing_eps : pd.Series
            Trailing EPS.
        forward_eps : pd.Series
            Forward (1y) EPS estimate.
        price : pd.Series
            Current price.
        ind_pe : pd.Series
            Industry benchmark P/E.
        eps_1y : pd.Series
            Not used by the scoring rules here; carried for API consistency.

        Returns
        -------
        pd.Series (int)
            Additive valuation score, with an additional +1 for 'SGLP.L' if present.

        Notes
        -----
        Division by zero is handled by mapping 1/0 to +∞, ensuring the comparison
        operators behave monotonically.
        """

        idx = trailing_eps.index.intersection(forward_eps.index).intersection(price.index).intersection(ind_pe.index)
       
        teps = trailing_eps.reindex(idx).fillna(0)
       
        feps = forward_eps.reindex(idx).fillna(0)
        
        pr = price.reindex(idx).fillna(0)
        
        ipe = ind_pe.reindex(idx).fillna(np.inf)
        
        with np.errstate(divide='ignore', invalid='ignore'):
           
            pe_tr = pd.Series(np.where(teps != 0, pr / teps, np.inf), index = idx)
           
            pe_f = pd.Series(np.where(feps != 0, pr / feps, np.inf), index = idx)
        
        mask = ~pd.Index(idx).str.endswith('.L')
        
        valid = idx[mask]
        
        adj = pd.Series(0, index = idx, dtype = int)
        
        adj.loc[valid] -= (pe_tr.loc[valid] > ipe.loc[valid]).astype(int)
        
        adj.loc[valid] += (pe_f.loc[valid] < pe_tr.loc[valid]).astype(int)
        
        adj.loc[valid] -= (pe_f.loc[valid] > pe_tr.loc[valid]).astype(int)

        return self._apply_sglp_bonus(
            adj = adj
        )        
    
    
    def upside_downside_score(
        self,
        hist_rets: pd.DataFrame,
        bench_hist_rets: pd.Series,
    ) -> pd.Series:
        """
        Score risk asymmetry using upside/downside capture metrics against a benchmark.

        Concept
        -------
        Capture statistics measure proportional participation in the benchmark’s up
        and down periods. Two families are used here:

        1) Slope-based capture (from a conditional regression/fit):
       
        - Upside slope (Up Slope): sensitivity of the asset to the benchmark on up
            periods (benchmark return > 0).
       
        - Downside slope (Down Slope): analogous for down periods (benchmark < 0).

        2) Ratio-based capture:
       
        - Upside ratio (Up Ratio): mean(asset | benchmark > 0) / mean(benchmark | benchmark > 0).
       
        - Downside ratio (Down Ratio): mean(asset | benchmark < 0) / mean(benchmark | benchmark < 0).

        Scoring Rules
        -------------
        Bonuses are awarded for strong upside with muted downside, and penalties for
        excess downside dominance:

        Using slopes:
       
        - +1 if (Up Slope > 1.5) and (Down Slope < 0.5)
       
        - +1 if (Up Slope > 1.0) and (Down Slope < 1.0)
       
        - +1 if (Up Slope > 1.0) and (Down Slope < 0.5)
       
        - +1 if (Up Slope > 1.5) and (Down Slope < 1.0)
       
        - +1 if (Down Slope < 0)  (anti-cyclical behaviour on down weeks)
       
        - −1 if (Down Slope > 1.0) and (Up Slope < Down Slope)
       
        - −1 if (Down Slope > 1.5) and (Up Slope < Down Slope)

        Using ratios (same logic on ratio space):
       
        - +1 if (Up Ratio > 1.5) and (Down Ratio < 0.5)
       
        - +1 if (Up Ratio > 1.0) and (Down Ratio < 1.0)
        
        - +1 if (Up Ratio > 1.0) and (Down Ratio < 0.5)
       
        - +1 if (Up Ratio > 1.5) and (Down Ratio < 1.0)
       
        - +1 if (Down Ratio < 0) (pathological but included symmetrically)
       
        - −1 if (Down Ratio > 1.0) and (Up Ratio < Down Ratio)
       
        - −1 if (Down Ratio > 1.5) and (Up Ratio < Down Ratio)

        Parameters
        ----------
        hist_rets : pd.DataFrame
            Panel of asset returns with columns = tickers and rows = datestamps. A
            multi-year weekly window is recommended for stability.
        bench_hist_rets : pd.Series
            Benchmark returns aligned on the same dates as `hist_rets`.

        Returns
        -------
        pd.Series (int)
            Additive capture-based score per ticker (columns of `hist_rets`).

        Notes
        -----
        - The helper functions `pa.capture_slopes` and `pa.capture_ratios` supply the
        slope and ratio measures. The internal estimation details are library-specific,
        but both are monotonically consistent with the informal definitions above.
      
        - Empty or insufficient return histories yield zeros.
        """

        def caps_for(
            tkr
        ):
        
            p = hist_rets[tkr].dropna()
        
            b = bench_hist_rets.reindex(p.index).dropna()
        
            if p.empty or b.empty:
        
                return 0.0, 0.0
        
            d = pa.capture_slopes(
                port_rets = p,
                bench_rets = b
            )
        
            m = pa.capture_ratios(
                port_rets = p, 
                bench_rets = b
            )
            
            return d['Upside Capture'], d['Downside Capture'], m['Upside Capture'], m['Downside Capture']
        
       
        caps = pd.DataFrame({t: caps_for(t) for t in hist_rets.columns}, index = ['Up Slope', 'Down Slope', 'Up Ratio', 'Down Ratio']).T
       
        up, down = caps['Up Slope'], caps['Down Slope']
       
        up_r, down_r = caps['Up Ratio'], caps['Down Ratio']
        
        adj = (up > 1.5).astype(int) * (down < 0.5).astype(int)
        
        adj += (up > 1.0).astype(int) * (down < 1.0).astype(int)
        
        adj += (up > 1.0).astype(int) * (down < 0.5).astype(int)
        
        adj += (up > 1.5).astype(int) * (down < 1.0).astype(int)
        
        adj += (down < 0).astype(int)  
        
        adj -= ((down > 1.0) & (up < down)).astype(int)
        
        adj -= ((down > 1.5) & (up < down)).astype(int)
        
        adj += (up_r > 1.5).astype(int) * (down_r < 0.5).astype(int)
        
        adj += (up_r > 1.0).astype(int) * (down_r < 1.0).astype(int)
        
        adj += (up_r > 1.0).astype(int) * (down_r < 0.5).astype(int)
        
        adj += (up_r > 1.5).astype(int) * (down_r < 1.0).astype(int)
        
        adj += (down_r < 0).astype(int)
        
        adj -= ((down_r > 1.0) & (up_r < down_r)).astype(int)
        
        adj -= ((down_r > 1.5) & (up_r < down_r)).astype(int)
        
        return adj
    
    
    def lower_target_adjustment(
        self, 
        lower_target: pd.Series,
        price: pd.Series
    ) -> pd.Series:
        """
        Award +1 when the analyst low target exceeds the current price.

        Rationale
        ---------
        If the most conservative sell-side target ('Low Price') is above the current
        price, the analyst consensus envelope is supportive even under pessimistic
        assumptions.

        Scoring Rule
        ------------
        Δ_t = 1{ LowTarget_t > Price_t }

        Parameters
        ----------
        lower_target : pd.Series
            Analyst low price target.
        price : pd.Series
            Current price.

        Returns
        -------
        pd.Series (int)
            Per-ticker increment with the SGLP.L +1 bonus applied if present.
        """
                
        idx = lower_target.index.intersection(price.index)
        
        adj = pd.Series((lower_target.reindex(idx) > price.reindex(idx)).astype(int), index = idx)

        return self._apply_sglp_bonus(
            adj = adj
        )
        

    def recommendation_adjustment(
        self, 
        recommendation: pd.Series
    ) -> pd.Series:
        """
        Translate headline analyst recommendation into an additive score.

        Mapping
        -------
        - 'strong_buy' → +3
        - 'hold'       → −1
        - {'sell', 'strong_sell'} → −10
        - Any other value → 0

        Parameters
        ----------
        recommendation : pd.Series
            Text recommendations; case-insensitive string comparison is applied.

        Returns
        -------
        pd.Series (int)
            Additive score per ticker.

        Notes
        -----
        This metric is intentionally coarse and asymmetric to emphasise outright sell
        flags and strong-buy conviction.
        """

        idx = recommendation.index
    
        rec = recommendation.reindex(idx).astype(str).str.lower()
    
        return pd.Series(
            np.select(
                [
                    rec.eq('strong_buy'),
                    rec.eq('hold'),
                    rec.isin(['sell', 'strong_sell']),
                ],
                [3, -1, -10],
                default = 0
            ),
            index = idx
        ).astype(int)


    def insider_purchases_adjustment(
        self, 
        insider_purchases: pd.Series
    ) -> pd.Series:
        """
        Score insider trading flow as a governance/quality proxy.

        Scoring Rule
        ------------
        - +2 if insider purchases > 0 (net buying)
        - −1 if insider purchases < 0 (net selling)
        -  0 otherwise (flat or missing)

        Parameters
        ----------
        insider_purchases : pd.Series
            Net insider purchases over a recent assessment window (units as provided).

        Returns
        -------
        pd.Series (int)
            Governance/flow-based adjustment.
        """

        idx = insider_purchases.index
    
        v = insider_purchases.reindex(idx).fillna(0)
    
        return pd.Series(np.where(v > 0, 2, np.where(v < 0, -1, 0)), index = idx).astype(int)


    def net_income_positive_adjustment(
        self, 
        net_income: pd.Series
    ) -> pd.Series:
        """
        Award +1 if net income is strictly positive.

        Indicator
        ---------
        Net income (after tax) is a coarse profitability metric used as a binary gate.

        Scoring Rule
        ------------
        Δ_t = 1{ NetIncome_t > 0 }

        Parameters
        ----------
        net_income : pd.Series
            Latest net income figure (units as reported).

        Returns
        -------
        pd.Series (int)
            +1 where positive; otherwise 0. Includes SGLP.L +1 bonus if present.
        """

        idx = net_income.index
        
        v = net_income.reindex(idx).fillna(0)
        
        adj =  pd.Series((v > 0).astype(int), index = idx)

        return self._apply_sglp_bonus(
            adj = adj
        )
        

    def ocf_gt_net_income_adjustment(
        self, 
        operating_cf: pd.Series, 
        net_income: pd.Series
    ) -> pd.Series:
        """
        Award +1 when operating cash flow exceeds net income (earnings quality signal).

        Rationale
        ---------
        Sustainable earnings tend to be underpinned by cash conversion. A rule of thumb
        is that OCF ≥ Net Income over recent periods is supportive.

        Scoring Rule
        ------------
        Δ_t = 1{ OCF_t > NetIncome_t }

        Parameters
        ----------
        operating_cf : pd.Series
            Operating cash flow.
        net_income : pd.Series
            Net income.

        Returns
        -------
        pd.Series (int)
            +1 if OCF dominates; otherwise 0. Includes SGLP.L +1 bonus if present.
        """
    
        idx = operating_cf.index.intersection(net_income.index)
    
        adj = pd.Series((operating_cf.reindex(idx).fillna(0) > net_income.reindex(idx).fillna(0)).astype(int), index = idx)

        return self._apply_sglp_bonus(
            adj = adj
        )
        
        
    def long_debt_improvement_adjustment(
        self, 
        prev_long_debt: pd.Series, 
        long_debt: pd.Series
    ) -> pd.Series:
        """
        Award +1 when long-term debt has fallen relative to the prior observation.

        Scoring Rule
        ------------
        Δ_t = 1{ LongDebt^-_t > LongDebt_t }

        Parameters
        ----------
        prev_long_debt : pd.Series
            Prior-period long-term debt.
        long_debt : pd.Series
            Current long-term debt.

        Returns
        -------
        pd.Series (int)
            +1 for a reduction; otherwise 0. SGLP.L +1 bonus applied if present.

        Notes
        -----
        Both series are compared after `fillna(np.inf)` which treats missing as
        non-improving by default.
        """

        idx = prev_long_debt.index.intersection(long_debt.index)
    
        adj =  pd.Series((prev_long_debt.reindex(idx).fillna(np.inf) > long_debt.reindex(idx).fillna(np.inf)).astype(int), index = idx)

        return self._apply_sglp_bonus(
            adj = adj
        )
        

    def current_ratio_improvement_adjustment(
        self, 
        prev_cr: pd.Series, 
        curr_cr: pd.Series
    ) -> pd.Series:
        """
        Award +1 when the current ratio has improved (liquidity enhancement).

        Definition
        ----------
        Current ratio CR_t = CurrentAssets_t / CurrentLiabilities_t

        Scoring Rule
        ------------
        Δ_t = 1{ CR_t > CR^-_t }

        Parameters
        ----------
        prev_cr : pd.Series
            Prior-period current ratio.
        curr_cr : pd.Series
            Current-period current ratio.

        Returns
        -------
        pd.Series (int)
            +1 for improvement; else 0. SGLP.L +1 bonus applied if present.
        """

        idx = prev_cr.index.intersection(curr_cr.index)
        
        adj =  pd.Series((curr_cr.reindex(idx).fillna(0) > prev_cr.reindex(idx).fillna(0)).astype(int), index = idx)

        return self._apply_sglp_bonus(
            adj = adj
        )
        
        
    def no_new_shares_adjustment(
        self, 
        shares_issued: pd.Series
    ) -> pd.Series:
        """
        Award +1 if no new shares have been issued (anti-dilution signal).

        Scoring Rule
        ------------
        Δ_t = 1{ NewSharesIssued_t ≤ 0 }

        Parameters
        ----------
        shares_issued : pd.Series
            Net new shares issued over the observation window (units as provided).

        Returns
        -------
        pd.Series (int)
            +1 for non-dilutive or shrinking share count; 0 otherwise.
        """

        idx = shares_issued.index
    
        v = shares_issued.reindex(idx).fillna(0)
    
        return pd.Series((v <= 0).astype(int), index = idx)


    def gross_margin_improvement_adjustment(
        self,
        gm: pd.Series, 
        prev_gm: pd.Series
    ) -> pd.Series:
        """
        Award +1 when gross margin has improved relative to the prior observation.

        Definition
        ----------
        Gross margin GM_t = (Revenue_t − CostOfGoodsSold_t) / Revenue_t

        Scoring Rule
        ------------
        Δ_t = 1{ GM_t > GM^-_t }

        Parameters
        ----------
        gm : pd.Series
            Current gross margin (ratio or percentage on a consistent basis).
        prev_gm : pd.Series
            Prior gross margin.

        Returns
        -------
        pd.Series (int)
            +1 for improvement; 0 otherwise. SGLP.L +1 bonus applied if present.
        """

        idx = gm.index.intersection(prev_gm.index)
        
    
        adj = pd.Series((gm.reindex(idx).fillna(0) > prev_gm.reindex(idx).fillna(0)).astype(int), index = idx)

        return self._apply_sglp_bonus(
            adj = adj
        )
        

    def asset_turnover_improvement_adjustment(
        self, 
        at: pd.Series,
        prev_at: pd.Series
    ) -> pd.Series:
        """
        Award +1 when asset turnover has improved (efficiency signal).

        Definition
        ----------
        Asset turnover AT_t = Revenue_t / AverageTotalAssets_t

        Scoring Rule
        ------------
        Δ_t = 1{ AT_t > AT^-_t }

        Parameters
        ----------
        at : pd.Series
            Current asset turnover.
        prev_at : pd.Series
            Prior asset turnover.

        Returns
        -------
        pd.Series (int)
            +1 for improvement; 0 otherwise. SGLP.L +1 bonus applied if present.
        """

        idx = at.index.intersection(prev_at.index)
       
        adj = pd.Series((at.reindex(idx).fillna(0) > prev_at.reindex(idx).fillna(0)).astype(int), index = idx)

        return self._apply_sglp_bonus(
            adj = adj
        )


    def skewness_adjustment(
        self, 
        r: pd.Series
    ) -> pd.Series:
        """
        Translate return skewness into a ternary score.

        Definition
        ----------
        Sample skewness is a third central moment normalised by σ^3. Positive skew
        (Skew_t > 0) implies a right tail and occasionally large positive realisations;
        negative skew (Skew_t < 0) implies a left tail and crash-prone distribution.

        Scoring Rule
        ------------
        For each ticker t:
        
        - +1 if Skew_t > 0
        
        - −1 if Skew_t < 0
        
        -  0 if Skew_t = 0 (or missing after fill)

        Parameters
        ----------
        r : pd.Series or pd.DataFrame-like accepted by `pa.skewness`
            Return history used by `pa.skewness` to produce per-ticker skewness.

        Returns
        -------
        pd.Series (int)
            Skewness-based adjustment per ticker.

        Notes
        -----
        The internal `pa.skewness` returns a Series aligned to tickers; this wrapper
        converts it to +1/−1/0 according to the sign.
        """

        skewness = pa.skewness(
            r = r
        )
        idx = skewness.index
        
        s = skewness.reindex(idx).fillna(0)
        
        skew_cond = np.where(s > 0, 1, np.where(s < 0, -1, 0))
        
        return pd.Series(skew_cond, index = idx).astype(int)
    
    
    def kurtosis_adjustment(
        self, 
        r: pd.Series
    ) -> pd.Series:
        """
        Award +1 when excess tail risk is not pronounced according to kurtosis.

        Definition
        ----------
        Kurtosis measures tail thickness. Under a normal distribution, kurtosis = 3.
        Values substantially above 3 suggest fat tails (greater crash or jump risk).

        Scoring Rule
        ------------
        Δ_t = 1{ Kurtosis_t < 3.5 }

        Parameters
        ----------
        r : pd.Series or pd.DataFrame-like accepted by `pa.kurtosis`
            Return history used by `pa.kurtosis` to produce per-ticker kurtosis.

        Returns
        -------
        pd.Series (int)
            +1 where kurtosis is below the threshold; 0 otherwise.

        Notes
        -----
        The 3.5 threshold is a conservative “near-normal” tolerance allowing mild
        excess kurtosis.
        """

        kurtosis = pa.kurtosis(
            r = r
        )
        
        idx = kurtosis.index
        
        k = kurtosis.reindex(idx).fillna(0)
        
        kurt_cond = np.where(k < 3.5, 1, 0)
        
        return pd.Series(kurt_cond, index = idx)


    def sharpe_adjustment(
        self, 
        r: pd.Series, 
        n: int
    ) -> pd.Series:
        """
        Translate the Sharpe ratio into a ternary score using annualisation implied by
        the input frequency.

        Definition
        ----------
        Sharpe ratio 
        
            SR_t = (μ_t − r_f) / σ_t, 
            
        where μ_t and σ_t are mean and standard deviation of periodic returns, annualised consistent 
        with `periods_per_year`, and r_f is the risk-free rate at the same periodicity (as handled by
        `pa.sharpe_ratio`).

        Scoring Rule
        ------------
        For each ticker t:
        
        - +1 if SR_t > 1         (acceptable risk-adjusted performance)
        
        -  0 if 0 < SR_t ≤ 1     (marginal)
        
        - −1 if SR_t ≤ 0         (uncompensated risk)

        Parameters
        ----------
        r : pd.Series or DataFrame
            Return history used by `pa.sharpe_ratio` (per ticker).
        n : int
            Number of periods per year (e.g., 52 for weekly, 252 for daily).

        Returns
        -------
        pd.Series (int)
            Sharpe-based adjustment per ticker.
        """

        sharpe = pa.sharpe_ratio(
            r = r, 
            periods_per_year = n
        )
        
        idx = sharpe.index
    
        s = sharpe.reindex(idx).fillna(0)
    
        return pd.Series(np.where(s > 1, 1, np.where(s <= 0, -1, 0)), index = idx).astype(int)


    def sortino_adjustment(
        self,
        r: pd.Series,
        n: int
    ) -> pd.Series:
        
        sortino = pa.sortino_ratio(
            returns = r, 
            riskfree_rate = config.RF, 
            periods_per_year = n
        )
        """
        Translate the Sortino ratio into a ternary score using an annualisation
        consistent with `n`.

        Definition
        ----------
        Sortino ratio 
        
            SoR_t = (μ_t − r_f) / σ_down,t,
            
        where σ_down,t is the downside standard deviation (computed using only negative 
        deviations from a threshold, typically r_f or 0), annualised consistent with `n`.
        Implemented via `pa.sortino_ratio`.

        Scoring Rule
        ------------
        For each ticker t:
      
        - +1 if SoR_t > 1
      
        -  0 if 0 < SoR_t ≤ 1
      
        - −1 if SoR_t ≤ 0

        Parameters
        ----------
        r : pd.Series or DataFrame
            Return history used by `pa.sortino_ratio`.
        n : int
            Number of periods per year.

        Returns
        -------
        pd.Series (int)
            Sortino-based adjustment per ticker.
        """
        
        idx = sortino.index
        
        s = sortino.reindex(idx).fillna(0)
        
        return pd.Series(np.where(s > 1, 1, np.where(s <= 0, -1, 0)), index = idx).astype(int)


    def signal_scores_adjustment(
        self, 
        signal_scores: pd.Series,
        index: pd.Index
    ) -> pd.Series:
        """
        Pass through externally-computed technical/quantitative “Signal Scores”.

        Purpose
        -------
        This function simply aligns a pre-computed additive score (e.g., from technical
        models) to the requested ticker index. It is treated as another additive term
        in the final score breakdown.

        Parameters
        ----------
        signal_scores : pd.Series
            A single-row Series of additive scores indexed by ticker.
        index : pd.Index
            The desired output ordering/universe.

        Returns
        -------
        pd.Series (int)
            `signal_scores` reindexed to `index` with missing values filled by 0.
        """

        return signal_scores.reindex(index).fillna(0).astype(int)


    def alpha_adjustments(
        self,
        hist_rets: pd.DataFrame,
        benchmark_ann_ret: float,
        comb_ret: pd.Series,
        last_year_ret: pd.Series,
        bench_hist_rets: pd.Series,
        rf: float,
        periods_per_year: int,
        d_to_e: pd.Series,
        tax: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute two alpha-based adjustments: 
        
            1) realised (Jensen) alpha sign 
        
            2) predicted-alpha flag implied by the combination forecast.

        Models
        ------
        1) Jensen's alpha (realised):
       
        For ticker t, regress excess returns on benchmark excess returns:
        
            (r_{p,t} − r_f) = α + β (r_{b,t} − r_f) + ε_t
        
        where r_{p,t} is the asset periodic return, r_{b,t} is the benchmark
        periodic return, and r_f is the periodic risk-free rate. The intercept α is
        annualised internally (via `periods_per_year`).

        2) Predicted alpha:
       
        A library function `pa.jensen_alpha_r2` also returns `pred_alpha` that
        compares the combined expected return (annualised) against the benchmark
        trajectory and fitted β, indicating whether the forecast implies a
        positive/negative alpha versus the market.

        Scoring Rules
        -------------
        - alpha_adj:
       
            +1 if α_ann > 0
            
            −1 if α_ann < 0
            
            0 otherwise.
            
        - pred_alpha_adj:
        
            −5 if `pred_alpha` < 0
            
            0 otherwise. (A strong penalty for forecasts that imply negative alpha.)

        Parameters
        ----------
        hist_rets : pd.DataFrame
            Weekly (or other periodic) returns per ticker.
        benchmark_ann_ret : float
            Benchmark annualised return over the longer lookback window.
        comb_ret : pd.Series
            Combined expected returns per ticker (annualised or mapped so that the
            library can infer an annualised prediction).
        bench_hist_rets : pd.Series
            Benchmark periodic returns aligned with `hist_rets`.
        rf : float
            Risk-free rate per period (e.g., per week).
        periods_per_year : int
            Annualisation factor (e.g., 52 for weekly).

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (alpha_adj, pred_alpha_adj), both integer Series aligned to
            `hist_rets.columns`.

        Notes
        -----
        - If a ticker has fewer than two valid observations or the benchmark series is
        empty over the intersecting dates, (NaN, NaN) is returned from the worker
        and converted to 0 in the final alignment step.
       
        - The magnitude of the predicted-alpha penalty (−5) is designed to dominate
        smaller style/quality signals.
        
        """
        ann_hist_ret = (1 + last_year_ret).prod()
        
        
        def one_ticker_alpha(
            tkr
        ):
            print(tkr)
            
            p = hist_rets[tkr].dropna()
            
            b = bench_hist_rets.reindex(p.index).dropna()
            
            c_r = comb_ret.loc[tkr]
            
            r = ann_hist_ret.loc[tkr]
            
            tax_t = tax.loc[tkr]
            
            d_to_e_t = d_to_e.loc[tkr]
            
            print('Combined Returns:', c_r)

            if len(p) < 2 or b.empty:
            
                return (np.nan, np.nan)

            alpha_dict = pa.jensen_alpha_r2(
                port_rets = p, 
                bench_rets = b,
                rf_per_period = config.RF_PER_WEEK,
                periods_per_year = periods_per_year,
                bench_ann_ret = benchmark_ann_ret,
                port_ann_ret_pred = c_r,             
                lever_beta = True,
                d_to_e = d_to_e_t,
                tax = tax_t,
                port_ann_ret_hist = r,
                scale_alpha_with_leverage = True
            )
            
            alpha = alpha_dict['alpha_ann']
            
            pred_alpha = alpha_dict['pred_alpha']
            
            print('Alpha:', alpha, 'Predicted Alpha:', pred_alpha)
            
            print('_________________________________________')
            
            return alpha, pred_alpha


        alpha_df = pd.DataFrame.from_dict(
            {
                t: one_ticker_alpha(
                    tkr = t
                ) for t in hist_rets.columns
            },
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
        Build combined expected returns and standard errors via inverse-variance
        weighting with per-ticker caps and per-group limits; compute a rich set of
        diagnostics and a fully additive score breakdown.

        Steps / Math
        ------------
        1) **Collect per-model forecasts**:
        
        For each model i, obtain per-ticker expected return ER_i(t) and standard
        error σ_i(t). 
       
        Clip:
           
            ER_i(t) ∈ [config.lbr, config.ubr]
           
            σ_i(t)  ∈ [MIN_STD, MAX_STD]

        2) **Validity and per-ticker caps**:
     
            A model is valid for a ticker if ER_i(t) ∉ {−1, 0}, finite, and σ_i(t) > 0.
           
            Let M_t be the number of valid models for t. Define the per-model cap for
            ticker t as:
           
                cap_t = max(MAX_MODEL_WT, 1 / M_t)
           
            This prevents a single model dominating when few models are available.
        
        3) **Raw inverse-variance weights** (pre-cap):
        
        For each ticker t:
           
            w_i^raw(t) = 1 / σ_i^2(t)
           
            Normalize across i:
                
                w_i(t) = w_i^raw(t) / Σ_j w_j^raw(t)
            
        Invalid models receive weight 0.

        4) **Cap-and-renormalize** (per ticker):
      
        Apply the cap `cap_t` elementwise and iteratively redistribute the deficit
        across remaining headroom until the column sums reach exactly 1. This is a
        bounded iterative proportional fitting step.

        5) **Group limits** (per ticker):
     
        Partition models into groups (Historical, Intrinsic Value, Factor, Machine
        Learning, etc.) and enforce per-ticker group caps, e.g.:
           
            Σ_{i ∈ HIST} w_i(t) ≤ 0.10
           
            Σ_{i ∈ IV}   w_i(t) ≤ 0.25
           
            Σ_{i ∈ F}    w_i(t) ≤ 0.25
           
            Σ_{i ∈ ML}   w_i(t) ≤ 0.40
        
        and analogous sub-caps for specific families (Prophet, LSTM, GRU, HGB,
        SARIMAX, FF). 
        
        If a group exceeds its cap, scale it down to the limit and
        renormalise the remaining mass proportionally.

        6) **Combined expected return**:
      
        Add dividend yield and form the convex combination:

            ER_comb(t) = YMean_t + Σ_i w_i(t) · ER_i(t)
       
        where YMean_t is the predicted dividend yield (decimal).
       
        7) **Combined variance** (within/between):
       
        Let 
        
            V_i(t) = σ_i^2(t) 
            
            ER* = ER_comb(t).
       
        - Within variance:
       
            Var_within(t)  = Σ_i w_i(t) · V_i(t)
       
        - Between variance (disagreement/dispersion term):
       
            Var_between(t) = (1/M_t) · Σ_i w_i(t) · (ER_i(t) − ER*)^2
       
        The total variance adds dividend yield uncertainty:
       
            Var_total(t) = Var_within(t) + Var_between(t) + (YStd_t)^2
       
        where YStd_t is the yield std. The combined standard error is:
       
            SE_comb(t) = sqrt( clip(Var_total(t), MIN_STD^2, MAX_STD^2) )

        8) **Diagnostics**
        Compute risk/return diagnostics on appropriate windows:
      
        - Skewness and kurtosis on multi-year returns.
      
        - Sharpe and Sortino ratios on the last-year window with appropriate
            annualisation.
      
        - Upside/Downside capture (slopes and ratios).
      
        - Jensen’s alpha and “predicted alpha” using combined ER.

        9) **Additive score breakdown**:
       
        Build an additive DataFrame whose columns are all component signals:
        
            - short-interest
            
            - WSB sentiment, 
            
            - growth (earnings, revenue), 
            
            - quality (ROA/ROE), 
            
            - valuation (P/B, EPS/P-E diagnostics),
            
            - momentum-asymmetry (up/down capture),
            
            - analyst overlays (targets, recommendations), 
            
            - insider,
        
            - profitability/cash conversion/leverage/liquidity/issuance/margins/efficiency,
        
            - higher moments (skewness, kurtosis), 
            
            - risk-adjusted returns (Sharpe, Sortino), 
            
            - alpha flags, 
            
            - technical Signal Scores.

        The final cross-sectional score is the row sum of the components; for all
        names except 'SGLP.L', it is capped above by `numberOfAnalystOpinions` to
        moderate spurious specificity:
          
            Final(t) = min( Σ_k Score_k(t) , Opinions_t )  for t ≠ 'SGLP.L'
          
            Final('SGLP.L') = Σ_k Score_k('SGLP.L')

        Parameters
        ----------
        region_indicators : Dict[str, pd.Series]
            Cross-sectional industry/region baselines keyed by:
            {'PE','PB','ROE','ROA','RevG','EarningsG'}. 
            
            All are aligned to the optimiser universe.

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
        
        The aligned outputs suitable for export and downstream optimisation.

        Notes
        -----
        - All inputs are reindexed to a common ticker set before any arithmetic.
        
        - Hard clipping of ER and SE protects against outliers and sheet errors.
        
        - Group caps prevent style concentration and improve robustness under model
        misspecification.
        """

        names = list(self.models.keys())

        rets = [self.models[n]['Returns'] for n in names]
        
        for n in names:
            
            print(n, self.models[n]['Returns'])
        
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
        
        d_to_e = a['debtToEquity'] / 100
       
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
        
        tax = a['Tax Rate']

        ind_pe = region_indicators['PE']
        
        ind_pb = region_indicators['PB']
        
        ind_roe = region_indicators['ROE']
        
        ind_roa = region_indicators['ROA']
        
        ind_rvg = region_indicators['RevG']
       
        ind_eg = region_indicators.get('EarningsG', pd.Series(0, index = price.index))

        all_series = rets + ses + [
            div, 
            recommendation, 
            shares_short,
            shares_outstanding, 
            shares_short_prior, 
            d_to_e,
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
            ind_eg,
            tax
        ]
       
        common_idx = set(all_series[0].index)
       
        for s in all_series[1:]:
            
            common_idx &= set(s.index)
       
        common_idx = sorted(common_idx)

        rets = [r.reindex(common_idx).fillna(0).clip(lower = config.lbr, upper = config.ubr) for r in rets]
       
        ses = [s.reindex(common_idx).fillna(0).clip(lower = MIN_STD, upper = MAX_STD) for s in ses]

        ret_df = pd.DataFrame({names[i]: rets[i] for i in range(len(names))}, index = common_idx)
        
        ret_df_clipped = ret_df

        model_vars = pd.DataFrame(
            { names[i]: ses[i] ** 2 for i in range(len(names)) },
            index = common_idx
        )

        se_df = pd.DataFrame(
            { names[i]: ses[i] for i in range(len(names)) },
            index = common_idx
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

                deficit = 1.0 - final.sum(axis = 0)  

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
       
        group_iv_names = ['DCF', 'DCFE', 'RI', 'RelVal', 'Gordon Growth Model']
       
        group_f_names = ['FF3', 'FF5', 'CAPM', 'FER']
       
        group_ml_names = ['Prophet', 'SARIMAX', 'LSTM_DirectH', 'LSTM_Cross_Asset', 'GRU_cal', 'GRU_raw', 'HGB Returns', 'ProphetPCA', 'HGB Returns CA']
        
        group_mc_names = ['AdvMC', 'TVP + GARCH Monte Carlo', 'SARIMAX Factor']
        
        group_prophet_names = ['Prophet', 'ProphetPCA']
        
        group_lstm_names = ['LSTM_DirectH', 'LSTM_Cross_Asset']
        
        group_gru_names = ['GRU_cal', 'GRU_raw']
        
        group_hgb_names = ['HGB Returns', 'HGB Returns CA']
        
        group_sarimax_names = ['SARIMAX', 'SARIMAX Factor']
        
        group_ff_names = ['FF3', 'FF5']

        group_hist_idx = [names.index(m) for m in group_hist_names if m in names]
       
        group_iv_idx = [names.index(m) for m in group_iv_names if m in names]
       
        group_f_idx = [names.index(m) for m in group_f_names if m in names]
       
        group_ml_idx = [names.index(m) for m in group_ml_names if m in names]
        
        group_mc_idx = [names.index(m) for m in group_mc_names if m in names]
        
        group_prophet_idx = [names.index(m) for m in group_prophet_names if m in names]
        
        group_lstm_idx = [names.index(m) for m in group_lstm_names if m in names]
        
        group_gru_idx = [names.index(m) for m in group_gru_names if m in names]
        
        group_hgb_idx = [names.index(m) for m in group_hgb_names if m in names]
        
        group_sarimax_idx = [names.index(m) for m in group_sarimax_names if m in names]
        
        group_ff_idx = [names.index(m) for m in group_ff_names if m in names]
        
        hist_limit = 0.1
        
        iv_limit = 0.3
        
        f_limit = 0.25
        
        ml_limit = 0.4
        
        mc_limit = 0.25
        
        prophet_limit = 0.1
        
        lstm_limit = 0.1
        
        gru_limit = 0.1
        
        hgb_limit = 0.1
        
        sarimax_limit = 0.1
        
        ff_limit = 0.1
        
        
        def cap_group(
            col, 
            group_idx, 
            limit
        ):
        
            current = w_arr[group_idx, col]
        
            if current.sum() > limit:

                w_arr[group_idx, col] *= limit / current.sum()

                others = [i for i in range(w_arr.shape[0]) if i not in group_idx]

                other_sum = w_arr[others, col].sum()

                if other_sum > 0:

                    w_arr[others, col] *= (1 - limit) / other_sum

            s = w_arr[:, col].sum()

            if s > 0:

                w_arr[:, col] /= s
                

        for col in range(w_arr.shape[1]):
            
            cap_group(
                col = col,
                group_idx = group_hist_idx, 
                limit = hist_limit
            )
            
            cap_group(
                col = col,
                group_idx = group_iv_idx, 
                limit = iv_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_f_idx,        
                limit = f_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_ml_idx, 
                limit = ml_limit
            )
            
            cap_group(
                col = col,
                group_idx = group_mc_idx,
                limit = mc_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_prophet_idx,
                limit = prophet_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_lstm_idx,
                limit = lstm_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_gru_idx,
                limit = gru_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_hgb_idx,
                limit = hgb_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_sarimax_idx,
                limit = sarimax_limit
            )
            
            cap_group(
                col = col, 
                group_idx = group_ff_idx,
                limit = ff_limit
            )
            
        weights = {
            names[i]: pd.Series(w_arr[i], index = common_idx)
            for i in range(len(names))
        }
        
        pred_div_yield = self.dividend_yield.reindex(common_idx)
        
        yield_pred = pred_div_yield['Yield Mean']
        
        yield_std = pred_div_yield['Yield Std'].fillna(0)

        comb_rets = yield_pred.fillna(0)

        for n in names:
            
            comb_rets += weights[n] * ret_df_clipped[n]

        w_df = pd.DataFrame(weights)

        within_var = (w_df * model_vars).sum(axis = 1)

        between_var = (
            w_df * (ret_df_clipped.sub(comb_rets, axis = 0) ** 2)
        ).sum(axis = 1)
        
        total_var = within_var + between_var + (yield_std ** 2)

        comb_stds = np.sqrt(total_var.clip(lower = MIN_STD ** 2, upper = MAX_STD ** 2))
        
        last_year_ret = weekly_ret.loc[
            weekly_ret.index >= pd.to_datetime(config.YEAR_AGO), common_idx
        ]
        
        last_5y_ret = weekly_ret.loc[
            weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO), common_idx
        ]
        
        last_year_period = len(last_year_ret)
        
        base_ccy = getattr(config, "BASE_CCY", "GBP")
        
        bench_ccy = getattr(config, "BENCHMARK_CCY", "USD") 


        last_year_ret_base = r.convert_returns_to_base(
            ret_df = last_year_ret, 
            base_ccy = base_ccy, 
            interval = "1wk"
        )

        last_5y_ret_base = r.convert_returns_to_base(
            ret_df = last_5y_ret, 
            base_ccy = base_ccy, 
            interval = "1wk"
        )

        benchmark_weekly_rets_base = self.convert_series_to_base(
            benchmark_weekly_rets,
            series_ccy = bench_ccy,
            base_ccy = base_ccy,
            interval = "1wk",
        )
        
        short_base = self.short_score(
            shares_short = shares_short, 
            shares_outstanding = shares_outstanding, 
            shares_short_prior = shares_short_prior
        ).reindex(common_idx).fillna(0)
        
        wsb_adj = self.wsb_score(
            index = pd.Index(common_idx)
        ).reindex(common_idx).fillna(0)
        
        eg_adj = self.earnings_growth_score(
            earnings_growth = earnings_growth, 
            ind_earnings_growth = ind_eg, 
            eps = teps, 
            eps_pred = eps_1y
        ).reindex(common_idx).fillna(0)
        
        rg_adj = self.revenue_growth_score(
            rev_growth = rev_growth, 
            ind_rvg = ind_rvg, 
            rev = rev, 
            rev_pred = rev_1y
        ).reindex(common_idx).fillna(0)
        
        roe_adj = self.return_on_equity_score(
            roe = roe, 
            ind_roe = ind_roe
        ).reindex(common_idx).fillna(0)
        
        roa_adj = self.return_on_assets_score(
            roa = roa, 
            prev_roa = prev_roa,
            ind_roa = ind_roa
        ).reindex(common_idx).fillna(0)
        
        pb_adj = self.price_to_book_score(
            pb = pb, 
            ind_pb = ind_pb
        ).reindex(common_idx).fillna(0)
        
        eps_adj = self.ep_score(
            trailing_eps = teps, 
            forward_eps = feps, 
            price = price, 
            ind_pe = ind_pe, 
            eps_1y = eps_1y
        ).reindex(common_idx).fillna(0)
        
        up_down_adj = self.upside_downside_score(
            hist_rets = last_5y_ret_base, 
            bench_hist_rets = benchmark_weekly_rets_base, 
        ).reindex(common_idx).fillna(0)

        lower_target_adj = self.lower_target_adjustment(
            lower_target = lower_target, 
            price = price
        ).reindex(common_idx).fillna(0)
        
        rec_adj = self.recommendation_adjustment(
            recommendation = recommendation
        ).reindex(common_idx).fillna(0)
        
        insider_adj = self.insider_purchases_adjustment(
            insider_purchases = insider_purchases
        ).reindex(common_idx).fillna(0)
        
        net_income_adj = self.net_income_positive_adjustment(
            net_income = net_income
        ).reindex(common_idx).fillna(0)
        
        ocf_adj = self.ocf_gt_net_income_adjustment(
            operating_cf = operating_cashflow, 
            net_income = net_income
        ).reindex(common_idx).fillna(0)
        
        ld_adj = self.long_debt_improvement_adjustment(
            prev_long_debt = prev_long_debt,
            long_debt = long_debt
        ).reindex(common_idx).fillna(0)
        
        cr_adj = self.current_ratio_improvement_adjustment(
            prev_cr = prev_current_ratio, 
            curr_cr = current_ratio
        ).reindex(common_idx).fillna(0)
        
        no_new_shares_adj = self.no_new_shares_adjustment(
            shares_issued = shares_issued
        ).reindex(common_idx).fillna(0)
        
        gm_adj = self.gross_margin_improvement_adjustment(
            gm = gross_margin,
            prev_gm = prev_gm
        ).reindex(common_idx).fillna(0)
        
        at_adj = self.asset_turnover_improvement_adjustment(
            at = at,
            prev_at = prev_at
        ).reindex(common_idx).fillna(0)
        
        skew_adj= self.skewness_adjustment(
            r = last_5y_ret
        ).reindex(common_idx).fillna(0)
        
        kurt_adj = self.kurtosis_adjustment(
            r = last_5y_ret
        ).reindex(common_idx).fillna(0)
        
        sharpe_adj = self.sharpe_adjustment(
            r = last_year_ret, 
            n = last_year_period
        ).reindex(common_idx).fillna(0)
        
        sortino_adj = self.sortino_adjustment(
            r = last_year_ret, 
            n = last_year_period
        ).reindex(common_idx).fillna(0)
        
        alpha_adj, pred_alpha_adj = self.alpha_adjustments(
            hist_rets = last_5y_ret_base,
            benchmark_ann_ret = benchmark_ann_ret_5y,
            comb_ret = comb_rets,
            last_year_ret = last_year_ret_base,
            bench_hist_rets = benchmark_weekly_rets_base,
            rf = config.RF_PER_WEEK,
            periods_per_year = 52,
            d_to_e = d_to_e,
            tax = tax
        )
        
        alpha_adj = alpha_adj.reindex(common_idx).fillna(0)
        
        pred_alpha_adj = pred_alpha_adj.reindex(common_idx).fillna(0)
        
        signal_adj = self.signal_scores_adjustment(self.signal_scores, pd.Index(common_idx)).reindex(common_idx).fillna(0)
        
        score_breakdown = pd.DataFrame({
            'Short Score': short_base,
            'WSB Adjustment': wsb_adj,
            'Earnings Growth Adjustment': eg_adj,
            'Revenue Growth Adjustment': rg_adj,
            'Return on Equity Adjustment': roe_adj,
            'Return on Assets Adjustment': roa_adj,
            'Price-to-Book Adjustment': pb_adj,
            'EPS Adjustment': eps_adj,
            'Upside/Downside Adjustment': up_down_adj,
            'Lower Target Price': lower_target_adj,
            'Recommendation': rec_adj,
            'Insider Purchases': insider_adj,
            'Net Income Positive': net_income_adj,
            'OCF > Net Income': ocf_adj,
            'Long-Debt Improvement': ld_adj,
            'Current-Ratio Improvement': cr_adj,
            'No New Shares Issued': no_new_shares_adj,
            'Gross-Margin Improvement': gm_adj,
            'Asset-Turnover Improvement': at_adj,
            'Skewness': skew_adj,
            'Kurtosis': kurt_adj,
            'Sharpe Ratio': sharpe_adj,
            'Sortino Ratio': sortino_adj,
            'Alpha': alpha_adj,
            'Pred Alpha': pred_alpha_adj,
            'Signal Scores': signal_adj,
        }).reindex(index = common_idx)

        final_scores = score_breakdown.sum(axis = 1)
        
        nY_aligned = np.maximum(nY, 2).reindex(common_idx).fillna(np.inf)
        
        mask = final_scores.index != "SGLP.L"
        
        final_scores.loc[mask] = final_scores.loc[mask].clip(upper = nY_aligned.loc[mask])
       
        score_breakdown['Final Score'] = final_scores

        return comb_rets, comb_stds, final_scores, weights, score_breakdown

    
def main():
    """
    Orchestrate the end-to-end pipeline: load inputs, compute combined forecasts,
    build covariance, assemble export tables, and write to Excel.

    Sequence
    --------
    1) Instantiate `PortfolioOptimiser` with the forecast workbook and `RatioData`.
    
    2) Collect region/industry baselines from `RatioData().dicts()`:
    
    PE, PB, ROE, ROA, revenue growth (RevG), earnings growth (EarningsG).
    
    3) Compute combined ER/SE and final scores:
    
    (comb_rets, comb_stds, final_scores, weights, score_breakdown, idx)
    = optimiser.compute_combination_forecast(region_indicators)
    
    4) Construct a shrinkage covariance matrix using daily/weekly/monthly returns
   and multi-source factor/index/industry/sector inputs:
   
        `Σ = shrinkage_covariance(...)`. 
    
    Annualisation is handled within that routine.
    
    5) Extract diagonal variance and compute per-name annualised volatility:

        `σ_i = sqrt(Σ_ii)`, clipping to [MIN_STD, MAX_STD].
    
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
    - The final DataFrames are passed through `ensure_headers_are_strings` to
    eliminate writer issues.
    
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
    
    comb_rets, comb_stds, final_scores, weights, score_breakdown = (
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

    cov = shrinkage_covariance(
        daily_5y = daily_ret_5y_base, 
        weekly_5y = weekly_ret_5y_base, 
        monthly_5y = monthly_ret_5y_base, 
        comb_std = comb_stds, 
        common_idx = comb_rets.index,
        ff_factors_weekly = factor_weekly_base,
        index_returns_weekly = index_weekly_base,
        industry_returns_weekly = ind_weekly_base,
        sector_returns_weekly = sec_weekly_base,
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
