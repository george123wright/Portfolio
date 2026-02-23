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
    
    EPS-based trailing/forward P/E compression vs industry benchmark.

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
from typing import Dict, Tuple, Sequence, Mapping, Hashable, Any, Callable

from data_processing.ratio_data import RatioData
from data_processing.financial_forecast_data import FinancialForecastData
from functions.export_forecast import export_results
from functions.cov_functions import shrinkage_covariance
import Optimiser.portfolio_functions as pf
import Optimiser.Port_Optimisation as po
import config

IND_DATA_FILE: str = config.IND_DATA_FILE

MIN_STD: float = 1e-2

MAX_STD: float = 2.0

MAX_MODEL_WT: float = 0.10

SHORT_RATIO_INCREASE: float = 1.05

SHORT_RATIO_DECREASE: float = 0.95

WSB_POS_THRESHOLD: float = 0.0

WSB_VPOS_THRESHOLD: float = 0.2

WSB_HIGH_ATTENTION: int = 4

WSB_VERY_HIGH_ATTENTION: int = 10

KURT_NEAR_NORMAL: float = 3.5

RISK_RATIO_GOOD: float = 1.0

PRED_ALPHA_PENALTY: int = 5

EPS_REV_CLIP: int = 5

HIST_GROUP_LIMIT: float = 0.10

IV_GROUP_LIMIT: float = 0.40

F_GROUP_LIMIT: float = 0.2

ML_GROUP_LIMIT: float = 0.40

MC_GROUP_LIMIT: float = 0.15

SUBGROUP_LIMIT: float = 0.10


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
    UPPER_PERCENTILE : int
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

        self.analytics = pf.PortfolioAnalytics(cache = False)

        self.today = dt.date.today()

        self.UPPER_PERCENTILE: int = 75

        self.weekly_ret = ratio_data.weekly_rets

        self.daily_ret = ratio_data.daily_rets

        self.daily_close = ratio_data.close

        self.daily_open = ratio_data.open

        self.macro_weekly = ratio_data.macro_data

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
            'DCF CapIQ': 'DCF CapIQ',
            'FCFE CapIQ': 'FCFE CapIQ',
            'DDM CapIQ': 'DDM CapIQ',
            'Daily Returns': 'Daily',
            'RI': 'RI',
            'RI CapIQ': 'RI CapIQ',
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
            'Tax Rate',
            'EPS Revision 30D',
            'EPS Revision 7D',
            'Asset Growth Rate',
            'Asset Growth Rate Vol'
        ]
        
        self.analyst_df = (
            xls.parse('Analyst Data', usecols = analyst_cols, index_col = 0)
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
        Convert a return series from local currency into a base currency using
        multiplicative FX compounding.

        Mathematical definition
        -----------------------
        Let:

        - r_local,t denote the asset return in source-currency terms.
       
        - P_fx,t denote the FX price for pair series_ccy/base_ccy at time t.
       
        - r_fx,t = (P_fx,t / P_fx,t-1) - 1 denote the FX return.

        The converted base-currency return is:

            r_base,t = (1 + r_local,t) * (1 + r_fx,t) - 1.

        This preserves exact one-period compounding and avoids approximation
        errors that arise from additive conversions.

        Implementation details
        ----------------------
        - If `series_ccy` equals `base_ccy`, the input is returned unchanged.
    
        - FX prices are fetched from `RatioData`, aligned to the return index,
          and forward/backward filled before differencing.
    
        - The first missing FX return is set to 0.0 to keep the aligned output
          finite and neutral.

        Parameters
        ----------
        series_rets : pd.Series
            Return series in source-currency units.
        series_ccy : str
            Source currency code.
        base_ccy : str
            Target base currency code.
        interval : str
            Sampling interval used by the FX data lookup.

        Returns
        -------
        pd.Series
            Return series expressed in base-currency units.

        Advantages
        ----------
        The multiplicative mapping is consistent with portfolio return algebra
        and is therefore suitable for benchmark-relative scoring and covariance
        construction on a common currency basis.
        """
       
        if series_ccy == base_ccy:
       
            return series_rets
       
        fx_map = self.ratio_data.get_fx_price_by_pair_local_to_base(
            country_to_pair = {
                series_ccy: f"{series_ccy}{base_ccy}"
            },
            base_ccy = base_ccy,
            interval = interval,
        )
        
        fx_price = fx_map[f"{series_ccy}{base_ccy}"].reindex(series_rets.index).ffill().bfill()
       
        fx_ret = fx_price.pct_change(fill_method=None).reindex(series_rets.index).fillna(0.0)
       
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

        inc -= ((ratio >= SHORT_RATIO_INCREASE * prior_ratio) & mask_prior_nz).astype(int)

        inc += (ratio <= SHORT_RATIO_DECREASE * prior_ratio).astype(int)
       
        return inc.astype(int)


    def wsb_score(
        self, 
        universe: pd.Index
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

        common = self.wsb.index.intersection(universe)
      
        adj = pd.Series(0, index = universe, dtype = int)
      
        if common.empty:
      
            return adj
      
        w = self.wsb.loc[common]
      
        pos = w['avg_sentiment'] > WSB_POS_THRESHOLD
        
        neg = w['avg_sentiment'] < WSB_POS_THRESHOLD

        hi = w['mentions'] > WSB_HIGH_ATTENTION
        
        vhi = w['mentions'] > WSB_VERY_HIGH_ATTENTION

        vpos = w['avg_sentiment'] > WSB_VPOS_THRESHOLD
        
        vneg = w['avg_sentiment'] < -WSB_VPOS_THRESHOLD

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
        
        adj = (v > 0).astype(int)

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
        
    
    def asset_growth_rate_sharpe(
        self,
        asset_growth_rate: pd.Series,
        asset_growth_rate_vol: pd.Series
    ) -> pd.Series:
        """
        Score the asset-growth signal using a Sharpe-like signal-to-noise
        statistic.

        Mathematical definition
        -----------------------
        For each ticker t, define:

            q_t = g_t / s_t,

        where g_t is the estimated asset growth rate and s_t is the volatility
        of that growth rate.

        The additive score increment is:

            +1 if q_t > 1,
            -1 if q_t < 0,
             0 otherwise.

        Inputs are aligned to a common index and missing values are replaced
        with zero before forming q_t.

        Parameters
        ----------
        asset_growth_rate : pd.Series
            Cross-sectional asset growth rate estimates.
        asset_growth_rate_vol : pd.Series
            Cross-sectional volatility estimates of asset growth.

        Returns
        -------
        pd.Series
            Integer score contribution with the deterministic SGLP.L bonus
            applied via `_apply_sglp_bonus`.

        Modelling rationale
        -------------------
        The ratio q_t rewards growth that is large relative to its own
        instability, so persistent balance-sheet expansion receives higher
        scores than equally large but noisy growth episodes.
        """
        
        asset_growth_rate = asset_growth_rate.reindex(asset_growth_rate_vol.index).fillna(0)
        
        asset_growth_rate_vol = asset_growth_rate_vol.fillna(0)
        
        asset_sharpe = asset_growth_rate.div(asset_growth_rate_vol).fillna(0)
        
        sc = (asset_sharpe > 1).astype(int)
        
        sc -= (asset_sharpe < 0).astype(int)
        
        return self._apply_sglp_bonus(
            adj = sc
        )
        

    def price_to_book_score(
        self, 
        pb: pd.Series, 
        ind_pb: pd.Series
        ) -> pd.Series:
        """
        Score valuation using Price-to-Book (P/B)'.

        Definitions
        -----------
        For ticker t:
       
        - Price-to-book: PB_t
       
        - Industry baseline: PB*_t

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
            Additive score on the full input index, then +1
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
                                        
        val = (p> 0) & (p <= 1)
        
        adj += val.astype(int)
        
        adj -= (p <= 0).astype(int)
        
        adj += (p < ipb).astype(int)
        
        adj -= (p > ipb).astype(int)

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
        Score EPS-based valuation diagnostics using trailing and forward P/E.

        Definitions
        -----------
        For ticker t:
      
        - Price: P_t
      
        - Trailing EPS: EPS_tr,t
      
        - Forward (1y) EPS: EPS^f_t
      
        - Industry forward/trailing P/E benchmark: PE*_t
      
        - Trailing P/E: PE_tr,t = P_t / EPS_tr,t    (∞ if EPS_tr,t = 0)
      
        - Forward P/E:  PE_f,t  = P_t / EPS^f_t     (∞ if EPS^f_t  = 0)

        Scoring Rules
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
        
        adj = pd.Series(0, index = idx, dtype = int)
        
        adj -= (pe_tr > ipe).astype(int)
        
        adj += (pe_f < pe_tr).astype(int)
        
        adj -= (pe_f > pe_tr).astype(int)

        return self._apply_sglp_bonus(
            adj = adj
        )        


    def peg_ratio_adjustment(
        self, 
        price: pd.Series, 
        forward_eps: pd.Series, 
        earnings_growth: pd.Series
    ) -> pd.Series:
        """
        Score based on the PEG ratio (Price/Earnings divided by Growth Rate).
        
        Formula
        -------
        PEG = (Price / Forward EPS) / (Earnings Growth * 100)
        
        (Note: earnings growth in the sheet appears as a decimal, for example
        0.15 for 15 percent, so the denominator is scaled accordingly.)

        Parameters
        ----------
        price : pd.Series
            Current Price.
        forward_eps : pd.Series
            Forward EPS estimate.
        earnings_growth : pd.Series
            projected earnings growth (decimal).

        Returns
        -------
        pd.Series (int)
            +1 for attractive PEG, negative scores for expensive/distressed PEG.
        """
        
        idx = price.index.intersection(forward_eps.index).intersection(earnings_growth.index)
        
        p = price.reindex(idx).fillna(0)
        
        eps = forward_eps.reindex(idx).fillna(0)
        
        g = earnings_growth.reindex(idx).fillna(0)
        
        pe = p.div(eps).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        peg = pe.div(g * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        scores = pd.Series(0, index=idx)
        
        scores.loc[(peg > 0) & (peg < 1.0)] = 1
        
        scores.loc[peg > 2.0] = -1
        
        scores.loc[peg < 0] = -1

        return self._apply_sglp_bonus(
            adj = scores
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
        - The helper functions `self.analytics.capture_slopes` and `self.analytics.capture_ratios` supply the
        slope and ratio measures. The internal estimation details are library-specific,
        but both are monotonically consistent with the informal definitions above.
      
        - Empty or insufficient return histories yield zeros.
        """

        def caps_for(
            tkr
        ):
            """
            Compute upside/downside capture diagnostics for a single ticker
            against the benchmark series.

            Conceptual measures
            -------------------
            Two complementary diagnostics are requested from the analytics
            utility:

            1. Capture slopes (regression-style sensitivities in up/down states).
           
            2. Capture ratios:

                   Upside ratio   = mean(r_asset | r_bench > 0) / mean(r_bench | r_bench > 0)
                   Downside ratio = mean(r_asset | r_bench < 0) / mean(r_bench | r_bench < 0)

            Parameters
            ----------
            tkr : Hashable
                Ticker key in `hist_rets`.

            Returns
            -------
            tuple[float, float, float, float]
                (up_slope, down_slope, up_ratio, down_ratio). Where data are
                insufficient, the current implementation returns placeholder
                zeros.

            Modelling rationale
            -------------------
            Using both slope and ratio views reduces sensitivity to a single
            estimator choice and provides a richer view of upside participation
            versus downside protection.
            """
        
            p = hist_rets[tkr].dropna()
        
            b = bench_hist_rets.reindex(p.index).dropna()
        
            if p.empty or b.empty:
        
                return 0.0, 0.0
        
            d = self.analytics.capture_slopes(
                port_rets = p,
                bench_rets = b
            )
        
            m = self.analytics.capture_ratios(
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
        

    def beta_adjustment(
        self,
        beta: pd.Series
    ) -> pd.Series:
        """
        Score market sensitivity (Beta).

        Scoring Rule
        ------------
        +1 for defensive/stable assets (0.5 < beta < 1.1).
        
        -1 for high sensitivity (beta > 1.5).
        
        0 otherwise.

        Parameters
        ----------
        beta : pd.Series
            The asset's beta relative to the benchmark.

        Returns
        -------
        pd.Series (int)
            Risk-based adjustment.
        """
        
        idx = beta.index
        
        b = beta.reindex(idx).fillna(1.0) 
        
        scores = pd.Series(0, index = idx)
        
        scores.loc[(b < 1.0)] = 1
        
        scores.loc[b > 1.5] = -1
                
        return self._apply_sglp_bonus(
            adj = scores
        )
        

    def volatility_trend_score(
        self,
        daily_rets: pd.DataFrame,
    ) -> pd.Series:
        """
        Score the volatility regime. 
        Reward volatility compression (Short Term < Long Term).
        Penalize volatility spikes.

        Parameters
        ----------
        daily_rets : pd.DataFrame
            Daily return history.

        Returns
        -------
        pd.Series (int)
            +1 for calming vol, -1 for spiking vol.
        """

        short_vol = daily_rets.iloc[-21:].std() 
       
        long_vol = daily_rets.iloc[-252:].std() 
        
        idx = daily_rets.columns
       
        s_vol = short_vol.reindex(idx)
       
        l_vol = long_vol.reindex(idx)
        
        scores = pd.Series(0, index=idx)
        
        scores.loc[s_vol < l_vol] = 0
        
        scores.loc[s_vol > l_vol] = -1
        
        scores.loc[s_vol > (l_vol * 1.5)] = -1
        
        scores.loc[s_vol > (l_vol * 2)] = -1
        
        scores.loc[s_vol < (l_vol * 0.5)] = +1
        
        return scores
        
        
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


    def debt_to_equity_adjustment(
        self, 
        d_to_e: pd.Series
    ) -> pd.Series:
        """
        Award +1 if 0 < debt / equity < 1. 
       
        Award -1 if debt / equity > 2.
       
        Award -2 if debt / equity > 4.
       
        Neutral (0) if debt / equity is 0 or between 1 and 2.
        
        Parameters
        ----------
        d_to_e : pd.Series
            Debt to Equity ratio values.

        Returns
        -------
        pd.Series (int)
            +1, 0, -1, or -2 based on the debt/equity thresholds. SGLP.L +1 bonus applied if present.
        """
        
        scores = pd.Series(0, index=d_to_e.index)
        
        scores.loc[(d_to_e > 0) & (d_to_e < 1)] = 1
        
        scores.loc[d_to_e > 2] = -1
        
        scores.loc[d_to_e > 4] = -2
        
        return self._apply_sglp_bonus(
            adj = scores
        )    
        

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
        

    def gross_profitability_score(
        self,
        gross_margin: pd.Series,
        asset_turnover: pd.Series
    ) -> pd.Series:
        """
        Score based on Robert Novy-Marx's 'Gross Profitability' premium.
        
        Formula
        -------
        
        GPA = (Revenue - COGS) / Total Assets
        
            = Gross Margin * Asset Turnover
            
        Scoring Rule
        ------------
        +1 if GPA > 0.45 (Highly productive assets)
       
        -1 if GPA < 0.10 (Unproductive assets)
        
        0 otherwise.
        
        Parameters
        ----------
        gross_margin : pd.Series
            Gross Profit / Revenue (decimal).
        asset_turnover : pd.Series
            Revenue / Total Assets.

        Returns
        -------
        pd.Series (int)
            +1 for high productivity, -1 for low.
        """
       
        idx = gross_margin.index.intersection(asset_turnover.index)
        
        gm = gross_margin.reindex(idx).fillna(0)
       
        at = asset_turnover.reindex(idx).fillna(0)
        
        gpa = gm * at
        
        scores = pd.Series(0, index=idx)
        
        scores.loc[gpa > 0.45] = 1
        
        scores.loc[gpa < 0.10] = -1
        
        return self._apply_sglp_bonus(
            adj = scores
        )


    def asset_growth_anomaly_score(
        self,
        asset_growth: pd.Series
    ) -> pd.Series:
        """
        Score based on the 'Asset Growth Anomaly' (Cooper, Gulen, Schill).
        
        Concept
        -------
        High asset growth predicts NEGATIVE future returns (Empire Building).
        Low/Negative asset growth predicts POSITIVE future returns (Efficiency).
        
        Scoring Rule
        ------------
        
        -1 if Asset Growth > 15% (High risk of empire building)
       
        +1 if Asset Growth < 5% (Conservative/Shrinking)
       
        0 otherwise.

        Parameters
        ----------
        asset_growth : pd.Series
            Year-over-year growth in total assets (decimal).

        Returns
        -------
        pd.Series (int)
            Penalty for high growth, reward for discipline.
        """
        
        idx = asset_growth.index
        
        ag = asset_growth.reindex(idx).fillna(0)
        
        scores = pd.Series(0, index = idx)
        
        scores.loc[ag > 0.0] = -1
        
        scores.loc[ag > 0.15] = -1
        
        scores.loc[ag < 0.05] = 1
        
        return self._apply_sglp_bonus(
            adj = scores
        )


    def shareholder_yield_score(
        self,
        div_yield: pd.Series,
        net_issuance: pd.Series,
        market_cap: pd.Series
    ) -> pd.Series:
        """
        Score based on Shareholder Yield (Dividends + Buybacks).
        
        Formula
        -------
        Buyback Yield = -(Net New Shares Issued Currency) / Market Cap
        Shareholder Yield = Dividend Yield + Buyback Yield
        
        Scoring Rule
        ------------
        +1 if Shareholder Yield > 5%
      
        +2 if Shareholder Yield > 8% (Strong conviction)
      
        -1 if Net Issuance > 5% of Market Cap (Dilution > 5%)

        Parameters
        ----------
        div_yield : pd.Series
            Dividend yield (decimal or percentage, assumed decimal 0.05 here).
        net_issuance : pd.Series
            Value of net new shares issued (Currency units). Positive = Dilution. Negative = Buyback.
        market_cap : pd.Series
            Total Market Cap (Currency units).

        Returns
        -------
        pd.Series (int)
            Reward for high total payout.
        """
        
        idx = div_yield.index.intersection(net_issuance.index).intersection(market_cap.index)
        
        dy = div_yield.reindex(idx).fillna(0)
       
        issuance = net_issuance.reindex(idx).fillna(0)
       
        mc = market_cap.reindex(idx).replace(0, np.inf)
        
        buyback_yield = -issuance / mc
        
        total_yield = dy + buyback_yield
        
        scores = pd.Series(0, index=idx)
        
        is_percentage = dy.max() > 5.0 
        
        threshold_low = 5.0 if is_percentage else 0.05
        
        threshold_high = 8.0 if is_percentage else 0.08
        
        scores.loc[total_yield > threshold_low] = 1
        
        scores.loc[total_yield > threshold_high] = 2 
        
        dilution_ratio = issuance / mc
        
        scores.loc[dilution_ratio > (0.05 if not is_percentage else 5.0)] = -1
        
        return self._apply_sglp_bonus(
            adj = scores
        )
        

    def accruals_score(
        self,
        net_income: pd.Series,
        ocf: pd.Series,
        total_revenue: pd.Series,
        asset_turnover: pd.Series
    ) -> pd.Series:
        """
        Score based on the Sloan Ratio (Earnings Quality).
        
        High accruals (Income > Cash Flow) indicate poor quality earnings and potential reversals.
        
        Formula
        -------
        Total Assets = Revenue / Asset Turnover
        Accruals Ratio = (Net Income - Operating Cash Flow) / Total Assets
        
        Scoring Rule
        ------------
        -1 if Accruals > 0.10 (High Accruals -> Bad Future Returns)
        
        +1 if Accruals < 0    (Conservative/Cash-rich Earnings -> Good Future Returns)
        
        Parameters
        ----------
        net_income : pd.Series
        ocf : pd.Series
        total_revenue : pd.Series
        asset_turnover : pd.Series

        Returns
        -------
        pd.Series (int)
            Earnings quality adjustment.
        """
        
        idx = net_income.index.intersection(ocf.index).intersection(total_revenue.index)
        
        ni = net_income.reindex(idx).fillna(0)
      
        cf = ocf.reindex(idx).fillna(0)
      
        rev = total_revenue.reindex(idx).fillna(0)
      
        at = asset_turnover.reindex(idx).replace(0, np.inf)
        
        assets = rev / at

        assets = assets.replace(0, np.inf) 
        
        accruals_ratio = (ni - cf) / assets
        
        scores = pd.Series(0, index = idx)
        
        scores.loc[accruals_ratio > 0.10] = -1
        
        scores.loc[accruals_ratio < 0.0] = 1
        
        return self._apply_sglp_bonus(
            adj = scores
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
        r: pd.Series | pd.DataFrame,
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
        r : pd.Series or pd.DataFrame-like accepted by `self.analytics.skewness`
            Return history used by `self.analytics.skewness` to produce per-ticker skewness.

        Returns
        -------
        pd.Series (int)
            Skewness-based adjustment per ticker.

        Notes
        -----
        The internal `self.analytics.skewness` returns a Series aligned to tickers; this wrapper
        converts it to +1/−1/0 according to the sign.
        """

        skewness = self.analytics.skewness(
            r = r
        )
        
        s = skewness.fillna(0)
        
        skew_cond = np.where(s > 0, 1, np.where(s < 0, -1, 0))
        
        return pd.Series(skew_cond, index = skewness.index).astype(int)
    
    
    def kurtosis_adjustment(
        self,
        r: pd.Series | pd.DataFrame,
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
        r : pd.Series or pd.DataFrame-like accepted by `self.analytics.kurtosis`
            Return history used by `self.analytics.kurtosis` to produce per-ticker kurtosis.

        Returns
        -------
        pd.Series (int)
            +1 where kurtosis is below the threshold; 0 otherwise.

        Notes
        -----
        The 3.5 threshold is a conservative “near-normal” tolerance allowing mild
        excess kurtosis.
        """

        kurtosis = self.analytics.kurtosis(
            r = r
        )
        
        k = kurtosis.fillna(0)

        kurt_cond = np.where(k < KURT_NEAR_NORMAL, 1, 0)
        
        kurt_cond = np.where(k > 5, kurt_cond - 1, kurt_cond)
        
        kurt_cond = np.where(k > 10, kurt_cond - 1, kurt_cond)
        
        return pd.Series(kurt_cond, index = kurtosis.index)


    def sharpe_adjustment(
        self,
        r: pd.Series | pd.DataFrame,
        n: int,
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
        `self.analytics.sharpe_ratio`).

        Scoring Rule
        ------------
        For each ticker t:
        
        - +1 if SR_t > 1         (acceptable risk-adjusted performance)
        
        -  0 if 0 < SR_t ≤ 1     (marginal)
        
        - −1 if SR_t ≤ 0         (uncompensated risk)

        Parameters
        ----------
        r : pd.Series or DataFrame
            Return history used by `self.analytics.sharpe_ratio` (per ticker).
        n : int
            Number of periods per year (e.g., 52 for weekly, 252 for daily).

        Returns
        -------
        pd.Series (int)
            Sharpe-based adjustment per ticker.
        """

        sharpe = self.analytics.sharpe_ratio(
            r = r, 
            periods_per_year = n
        )
        
        s = sharpe.fillna(0)

        scores = np.where(
            s > RISK_RATIO_GOOD,
            1,
            np.where(s <= 0, -1, 0),
        )
        
        return pd.Series(scores, index = sharpe.index).astype(int)
    

    def sortino_adjustment(
        self,
        r: pd.Series | pd.DataFrame,
        n: int,
    ) -> pd.Series:
        """
        Translate the Sortino ratio into a ternary score using an annualisation
        consistent with `n`.

        Definition
        ----------
        Sortino ratio 
        
            SoR_t = (μ_t − r_f) / σ_down,t,
            
        where σ_down,t is the downside standard deviation (computed using only negative 
        deviations from a threshold, typically r_f or 0), annualised consistent with `n`.
        Implemented via `self.analytics.sortino_ratio`.

        Scoring Rule
        ------------
        For each ticker t:
      
        - +1 if SoR_t > 1
      
        -  0 if 0 < SoR_t ≤ 1
      
        - −1 if SoR_t ≤ 0

        Parameters
        ----------
        r : pd.Series or DataFrame
            Return history used by `self.analytics.sortino_ratio`.
        n : int
            Number of periods per year.

        Returns
        -------
        pd.Series (int)
            Sortino-based adjustment per ticker.
        """
                
        sortino = self.analytics.sortino_ratio(
            returns = r, 
            riskfree_rate = config.RF, 
            periods_per_year = n
        )
        
        s = sortino.fillna(0)

        scores = np.where(
            s > RISK_RATIO_GOOD,
            1,
            np.where(s <= 0, -1, 0),
        )
        return pd.Series(scores, index = sortino.index).astype(int)


    def signal_scores_adjustment(
        self, 
        signal_scores: pd.Series,
        universe: pd.Index
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

        return signal_scores.reindex(universe).fillna(0).astype(int)


    def alpha_adjustments(
        self,
        hist_rets: pd.DataFrame,
        benchmark_ann_ret: float,
        comb_ret: pd.Series,
        last_year_ret: pd.Series,
        bench_hist_rets: pd.Series,
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
       
        A library function `self.analytics.jensen_alpha_r2` also returns `pred_alpha` that
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
            """
            Estimate realised and forecast-implied Jensen alpha for one ticker.

            Model concept
            -------------
            The analytics backend applies a leverage-aware market model of the
            form:

                r_p,t - r_f,t = alpha + beta_lev * (r_b,t - r_f,t) + epsilon_t.

            The function returns:

            - `alpha_ann`: annualised realised alpha from historical returns.
            - `pred_alpha`: forecast-implied alpha from the combined expected
              return relative to the benchmark trajectory and fitted beta.

            Parameters
            ----------
            tkr : Hashable
                Ticker symbol in the estimation universe.

            Returns
            -------
            tuple[float, float]
                (alpha_ann, pred_alpha). If the history is insufficient, both
                values are returned as NaN and converted to neutral scores by
                the caller.

            Advantages
            ----------
            Separating realised alpha from forecast-implied alpha enables a
            direct consistency check between historical skill and forward
            expectations, which improves robustness of the final additive score.
            """
                        
            p = hist_rets[tkr].dropna()
            
            b = bench_hist_rets.reindex(p.index).dropna()
            
            c_r = comb_ret.loc[tkr]
            
            r = ann_hist_ret.loc[tkr]
            
            tax_t = tax.loc[tkr]
            
            d_to_e_t = d_to_e.loc[tkr]
            
            if len(p) < 2 or b.empty:
            
                return (np.nan, np.nan)

            alpha_dict = self.analytics.jensen_alpha_r2(
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
        
        pred_alpha_adj -= PRED_ALPHA_PENALTY * (alpha_df['pred_alpha'].fillna(0) < 0).astype(int)
        
        return alpha_adj.reindex(hist_rets.columns).fillna(0), pred_alpha_adj.reindex(hist_rets.columns).fillna(0)
    
    
    def eps_revision_score(
        self,
        eps_rev_7: pd.Series,
        eps_rev_30: pd.Series,
        w7: float = 2,
        w30: float = 1,
    ) -> pd.Series:
        """
        Compute a weighted EPS revision score with clipping.
        
        Parameters
        ----------
        eps_rev_7 : pd.Series
            EPS revision over 7 days.
        eps_rev_30 : pd.Series
            EPS revision over 30 days.
        w7 : float, optional
            Weight for the 7-day revision (default is 2).
        w30 : float, optional
            Weight for the 30-day revision (default is 1).
       
        Returns
        -------
        pd.Series
            Combined EPS revision score, clipped to [-EPS_REV_CLIP, EPS_REV_CLIP].
        """
                
        score = (eps_rev_7 * w7) + (eps_rev_30 * w30)

        score = score.clip(lower = -EPS_REV_CLIP, upper = EPS_REV_CLIP)
        
        return score.fillna(0)
        

    @staticmethod
    def _cap_norm(
        w_arr: np.ndarray,
        cap: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Cap-and-renormalise weights column-wise, respecting per-ticker caps.

        Parameters
        ----------
        w_arr : np.ndarray, shape (n_models, n_tickers)
            Raw weights (columns sum to ~1 before capping).
        cap : np.ndarray, shape (n_tickers,)
            Per-ticker maximum weight.
        mask : np.ndarray, shape (n_models, n_tickers), optional
            Boolean mask of valid model/ticker cells.

        Returns
        -------
        np.ndarray
            Capped weights with each column summing to (approximately) 1.
        """
        
        final = np.minimum(w_arr, cap[np.newaxis, :])

        if mask is not None:
            
            final = np.where(mask, final, 0.0)

        n_models, n_tickers = final.shape

        for j in range(n_tickers):
            
            col = final[:, j]

            if mask is not None:
                
                active = mask[:, j]
                
            else:
                
                active = np.ones_like(col, dtype = bool)

            for _ in range(n_models): 
                
                s = col.sum()
                
                deficit = 1.0 - s

                if deficit <= 1e-8:
             
                    break

                room = np.maximum(cap[j] - col, 0.0)
             
                if mask is not None:
             
                    room = np.where(active, room, 0.0)

                room_sum = room.sum()
             
                if room_sum <= 0.0:
             
                    break

                delta = deficit * room / room_sum
             
                col = np.minimum(col + delta, cap[j])

                if mask is not None:
             
                    col = np.where(active, col, 0.0)

            s = col.sum()

            if s > 0.0:

                col /= s

            final[:, j] = col

        return final
    

    @staticmethod
    def _apply_group_caps_iterative(
        w_arr: np.ndarray,                
        raw_pref: np.ndarray,              
        cap_per_ticker: np.ndarray,        
        valid_mask: np.ndarray,             
        group_constraints: list[tuple[str, list[int], float]],
        *,
        tol: float = 1e-10,
        max_passes: int = 30,
    ) -> np.ndarray:
        """
        Enforce group and subgroup caps for each ticker while preserving
        per-model cap constraints and near-simplex normalisation.

        Constraint system (per ticker)
        ------------------------------
        For model weights w_i and per-model cap c:

            0 <= w_i <= c,
        
            sum_i w_i = 1,
        
            sum_{i in G_k} w_i <= L_k   for each group k.

        Group sets can overlap. A single projection is therefore insufficient,
        so this routine iterates constraint enforcement passes until changes are
        below tolerance or `max_passes` is reached.

        Method summary
        --------------
        - Build boolean masks for each group constraint.
        
        - For each ticker column:
        
          - clip to non-negative weights and per-model cap;
        
          - repeatedly enforce each group cap with `_enforce_one_constraint`;
        
          - if required, renormalise active weights to sum to one only when the
            renormalisation itself respects the cap.

        Parameters
        ----------
        w_arr : np.ndarray
            Initial weights with shape (n_models, n_tickers).
        raw_pref : np.ndarray
            Preference matrix used as redistribution weights.
        cap_per_ticker : np.ndarray
            Per-ticker individual model caps.
        valid_mask : np.ndarray
            Boolean validity matrix for model-ticker pairs.
        group_constraints : list[tuple[str, list[int], float]]
            Group definitions in the form (name, model indices, cap).
        tol : float, optional
            Numerical tolerance for convergence and feasibility checks.
        max_passes : int, optional
            Maximum number of full constraint sweeps per ticker.

        Returns
        -------
        np.ndarray
            Weight matrix after iterative feasibility enforcement.

        Key properties:
      
        - Never increases any model weight above cap_per_ticker[ticker].
      
        - Applies a group cap only when feasible to move mass outside the group.
      
        - Uses iterative passes so overlapping caps settle.
        """

        n_models, n_tickers = w_arr.shape
    
        out = w_arr.copy()

        g_masks: list[tuple[str, np.ndarray, float]] = []
     
        for gname, gidx, glimit in group_constraints:
     
            m = np.zeros(n_models, dtype=bool)
     
            if gidx:
     
                m[np.asarray(gidx, dtype=int)] = True
     
            g_masks.append((gname, m, float(glimit)))

     
        def _waterfill_add(
            w: np.ndarray,
            add_mask: np.ndarray,
            need: float,
            cap: float,
            pref: np.ndarray,
        ) -> float:
            """
            Reallocate a required mass `need` across eligible positions using
            constrained proportional allocation.

            For eligible indices i with remaining room:

                room_i = cap - w_i,

            a proposed update is:

                delta_i = need * p_i / sum_j p_j,

            where p_i are positive preferences. The update is clipped by room_i:

                w_i <- w_i + min(delta_i, room_i).

            The loop continues until residual need is negligible or capacity is
            exhausted.

            Parameters
            ----------
            w : np.ndarray
                Mutable ticker-level weight vector.
            add_mask : np.ndarray
                Boolean mask identifying recipient positions.
            need : float
                Mass that must be added outside a constrained group.
            cap : float
                Per-model upper bound for the ticker.
            pref : np.ndarray
                Preference vector used for proportional redistribution.

            Returns
            -------
            float
                Residual unallocated mass after capacity-constrained updates.
            """

            while need > tol:

                idx = np.where(add_mask)[0]

                if idx.size == 0:

                    break

                room = cap - w[idx]

                ok = room > tol

                if not np.any(ok):

                    break

                idx = idx[ok]

                room = room[ok]

                p = pref[idx]

                p = np.where(np.isfinite(p) & (p > 0), p, 1.0)

                p_sum = float(p.sum())

                if p_sum <= tol:

                    p = np.ones_like(p)

                    p_sum = float(p.sum())

                delta = need * (p / p_sum)

                delta = np.minimum(delta, room)

                w[idx] += delta

                need -= float(delta.sum())

                if float(delta.sum()) <= tol:

                    break

            return need


        def _enforce_one_constraint(
            w: np.ndarray,
            active: np.ndarray,
            gmask: np.ndarray,
            limit: float,
            cap: float,
            pref: np.ndarray,
        ) -> np.ndarray:
            """
            Enforce a single group cap with feasibility-aware scaling and
            redistribution.

            Let G be the constrained group and:

                s_in = sum_{i in G} w_i.

            If s_in exceeds `limit`, in-group weights are scaled towards an
            effective limit:

                limit_eff = max(limit, s_in - room_out),

            where room_out is total available capacity outside G:

                room_out = sum_{i not in G} max(cap - w_i, 0).

            The removed mass is then redistributed outside the group with
            `_waterfill_add`, preserving non-negativity and individual caps.

            Parameters
            ----------
            w : np.ndarray
                Mutable ticker-level weight vector.
            active : np.ndarray
                Active positions permitted to hold weight.
            gmask : np.ndarray
                Group membership mask.
            limit : float
                Group cap.
            cap : float
                Per-model cap for the ticker.
            pref : np.ndarray
                Preference vector for redistribution.

            Returns
            -------
            np.ndarray
                Updated weight vector after enforcing the single constraint.
            """
          
            in_mask = active & gmask
          
            if not np.any(in_mask):
          
                return w

            s_in = float(w[in_mask].sum())
          
            if s_in <= limit + tol:
          
                return w

            out_mask = active & (~gmask)
          
            if not np.any(out_mask):
          
                return w

            room_out = np.maximum(cap - w[out_mask], 0.0)

            room_out_sum = float(room_out.sum())

            if room_out_sum <= tol:

                return w

            min_feasible = s_in - room_out_sum

            limit_eff = max(limit, min_feasible)

            if limit_eff >= s_in - tol:
                
                return w  

            scale = limit_eff / s_in

            w[in_mask] *= scale

            deficit = s_in - limit_eff

            need = deficit

            need = _waterfill_add(
                w = w, 
                add_mask = out_mask,
                need = need,
                cap = cap,
                pref = pref
            )

            if need > tol:

                idx = np.where(out_mask)[0]

                room = cap - w[idx]

                j = idx[np.argmax(room)] if idx.size else None

                if j is not None and room.max() > tol:

                    add = min(need, float(room.max()))

                    w[j] += add

                    need -= add

            return w


        for col in range(n_tickers):

            active = valid_mask[:, col].astype(bool)

            if not np.any(active):

                continue

            cap = float(cap_per_ticker[col])

            w = out[:, col].copy()

            w = np.where(active, w, 0.0)

            w = np.minimum(np.maximum(w, 0.0), cap)

            pref = raw_pref[:, col].copy()

            pref = np.where(np.isfinite(pref) & (pref > 0), pref, 1.0)

            for _ in range(max_passes):

                w_prev = w.copy()

                for _gname, gmask, limit in g_masks:

                    w = _enforce_one_constraint(
                        w = w, 
                        active = active, 
                        gmask = gmask, 
                        limit = limit, 
                        cap = cap, 
                        pref = pref
                    )

                if np.max(np.abs(w - w_prev)) < 1e-10:
                  
                    break

            s = float(w[active].sum())

            if s > 0 and abs(s - 1.0) > 1e-8:

                w_scaled = w.copy()

                w_scaled[active] /= s

                if np.all(w_scaled[active] <= cap + 1e-12):

                    w = w_scaled

            out[:, col] = w

        return out


    def _score(
        self,
        func: Callable[..., pd.Series],
        *,
        index: pd.Index,
        **kwargs: Any,
    ) -> pd.Series:
        """
        Apply a scoring function and standardise the result onto a specified
        ticker universe.

        Post-processing map
        -------------------
        If `s_raw` is the output of `func(**kwargs)`, the returned series is:

            s_out = astype_int( fillna( reindex(s_raw, index), 0 ) ).

        This enforces:

        - deterministic index alignment;
     
        - neutral treatment of missing observations;
     
        - integer-valued contributions for additive score aggregation.

        Parameters
        ----------
        func : Callable[..., pd.Series]
            Scoring function to evaluate.
        index : pd.Index
            Target index for alignment.
        **kwargs : Any
            Keyword arguments forwarded to `func`.

        Returns
        -------
        pd.Series
            Integer score vector aligned to `index`.

        Advantages
        ----------
        A single normalisation path removes repetitive alignment code and
        prevents subtle cross-sectional mismatches when combining many scoring
        components.
        """
        
        raw = func(**kwargs)
        
        return raw.reindex(index).fillna(0).astype(int)
 
 
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
        
        ses = [self.models[n]['SE'] for n in names]
        
        _, benchmark_weekly_rets, _ = po.benchmark_rets(
            benchmark = config.benchmark, 
            start = config.FIVE_YEAR_AGO, 
            end = config.TODAY, 
            steps = 52
        )

        benchmark_ann_ret_5y = (1 + benchmark_weekly_rets).prod() ** 0.2 - 1
       
        a = self.analyst_df
       
        div = a['dividendYield'] / 100
        
        mc = a['marketCap']
       
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
        
        beta = a['beta']
       
        prev_gm = a['Previous Gross Margin']
       
        at = a['Asset Turnover']
        
        prev_at = a['Previous Asset Turnover']
       
        eps_1y = a['Avg EPS Estimate']
       
        rev = a['totalRevenue']
        
        rev_1y = a['Avg Revenue Estimate']
       
        nY = a['numberOfAnalystOpinions']
        
        tax = a['Tax Rate']
        
        eps_rev_7 = a['EPS Revision 7D']
        
        eps_rev_30 = a['EPS Revision 30D']
        
        asset_growth_rate = a['Asset Growth Rate']
        
        asset_growth_rate_vol = a['Asset Growth Rate Vol']

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
            tax,
            eps_rev_7,
            eps_rev_30,
            asset_growth_rate,
            asset_growth_rate_vol
        ]
       
        common_idx = common_index(
            series_list = all_series
        )

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
            (se_df > 0.02) & se_df.notna()              
        )
        
        model_counts = valid.sum(axis = 1).replace(0, np.nan)
        
        cap_per_ticker = np.maximum(MAX_MODEL_WT, 1.0 / model_counts).fillna(MAX_MODEL_WT)

        inv_var = 1.0 / model_vars
        
        inv_var = inv_var.where(valid, other = 0.0)

        tot_inv = inv_var.sum(axis = 1)

        raw_w = inv_var.div(tot_inv, axis = 0)

        w_arr = self._cap_norm(
            w_arr = raw_w.values.T,          
            cap = cap_per_ticker.values,     
            mask = valid.values.T
        )
        
        group_hist_names = ['Daily', 'EMA']
       
        group_iv_names = ['DCF', 'DCFE', 'RI', 'RelVal', 'Gordon Growth Model', 'DCF CapIQ', 'FCFE CapIQ', 'RI CapIQ', 'DDM CapIQ']
        
        group_dcf_names = ['DCF', 'DCF CapIQ']
        
        group_dcfe_names = ['DCFE', 'FCFE CapIQ']
        
        group_ri_names = ['RI', 'RI CapIQ']
        
        group_ddm_names = ['Gordon Growth Model', 'DDM CapIQ']
       
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
        
        group_dcf_idx = [names.index(m) for m in group_dcf_names if m in names]
        
        group_ri_idx = [names.index(m) for m in group_ri_names if m in names]
        
        group_ddm_idx = [names.index(m) for m in group_ddm_names if m in names]
        
        group_dcfe_idx = [names.index(m) for m in group_dcfe_names if m in names]
        
        group_constraints = [
            ("HIST", group_hist_idx, HIST_GROUP_LIMIT),
            ("IV", group_iv_idx, IV_GROUP_LIMIT),
            ("F", group_f_idx, F_GROUP_LIMIT),
            ("ML", group_ml_idx, ML_GROUP_LIMIT),
            ("MC", group_mc_idx, MC_GROUP_LIMIT),
            ("PROPHET", group_prophet_idx, SUBGROUP_LIMIT),
            ("LSTM", group_lstm_idx, SUBGROUP_LIMIT),
            ("GRU", group_gru_idx, SUBGROUP_LIMIT),
            ("HGB", group_hgb_idx, SUBGROUP_LIMIT),
            ("SARIMAX", group_sarimax_idx, SUBGROUP_LIMIT),
            ("FF", group_ff_idx, SUBGROUP_LIMIT),
            ("DCF", group_dcf_idx, SUBGROUP_LIMIT),
            ("RI", group_ri_idx, SUBGROUP_LIMIT),
            ("DDM", group_ddm_idx, SUBGROUP_LIMIT),
            ("DCFE", group_dcfe_idx, SUBGROUP_LIMIT),
        ]
        
        w_arr = self._apply_group_caps_iterative(
            w_arr = w_arr,
            raw_pref = raw_w.values.T,
            cap_per_ticker = cap_per_ticker.values,
            valid_mask = valid.values.T,
            group_constraints = group_constraints,
            max_passes = 30,
        )

        col_sums = w_arr.sum(axis = 0)

        if np.max(np.abs(col_sums - 1.0)) > 1e-6:

            logging.warning("Weight columns not summing to 1 (max dev=%.3e)", np.max(np.abs(col_sums - 1.0)))

        if np.max(w_arr - cap_per_ticker.values[np.newaxis, :]) > 1e-8:
       
            logging.warning("Per-model cap violated (max over=%.3e)", np.max(w_arr - cap_per_ticker.values[np.newaxis, :]))

        weights = {
            names[i]: pd.Series(w_arr[i], index = common_idx)for i in range(len(names))
        }
        
        pred_div_yield = self.dividend_yield.reindex(common_idx)
        
        yield_pred = pred_div_yield['Yield Mean']
        
        yield_std = pred_div_yield['Yield Std'].fillna(0)

        er_bar = pd.Series(0.0, index = common_idx)

        for n in names:
            
            er_bar += weights[n] * ret_df_clipped[n]

        comb_rets = yield_pred.fillna(0) + er_bar

        w_df = pd.DataFrame(weights)

        within_var = (w_df.pow(2) * model_vars).sum(axis = 1)

        between_var = (w_df * (ret_df_clipped.sub(er_bar, axis = 0) ** 2)).sum(axis = 1)
                
        between_var = between_var.fillna(0.0)
        
        total_var = within_var + between_var + (yield_std ** 2)

        comb_stds = np.sqrt(total_var.clip(lower = MIN_STD ** 2, upper = MAX_STD ** 2))
        
        weekly_ret_5y = self.weekly_ret.loc[
            self.weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO),
            common_idx,
        ]

        base_ccy = getattr(config, "BASE_CCY", "GBP")
        
        bench_ccy = getattr(config, "BENCHMARK_CCY", "USD")

        weekly_ret_5y_base = self.ratio_data.convert_returns_to_base(
            ret_df = weekly_ret_5y,
            base_ccy = base_ccy,
            interval = "1wk",
        )

        last_5y_ret_base = weekly_ret_5y_base

        last_year_ret_base = weekly_ret_5y_base.loc[
            weekly_ret_5y_base.index >= pd.to_datetime(config.YEAR_AGO)
        ]

        last_year_period = len(last_year_ret_base)

        benchmark_weekly_rets_base = self.convert_series_to_base(
            benchmark_weekly_rets,
            series_ccy = bench_ccy,
            base_ccy = base_ccy,
            interval = "1wk",
        )

        idx = pd.Index(common_idx)
        
        short_base = self._score(
            func = self.short_score,
            index = idx,
            shares_short = shares_short, 
            shares_outstanding = shares_outstanding, 
            shares_short_prior = shares_short_prior
        )
        
        wsb_adj = self._score(
            func = self.wsb_score, 
            index = idx,
            universe = idx
        )
        
        eg_adj = self._score(
            func = self.earnings_growth_score,
            index = idx,
            earnings_growth = earnings_growth, 
            ind_earnings_growth = ind_eg, 
            eps = teps, 
            eps_pred = eps_1y
        )
        
        beta_adj = self._score(
            func = self.beta_adjustment,
            index = idx,
            beta = beta
        )

        peg_adj = self._score(
            func = self.peg_ratio_adjustment,
            index = idx,
            price = price,
            forward_eps = feps,
            earnings_growth = earnings_growth
        )
        
        vol_trend_adj = self._score(
            func = self.volatility_trend_score,
            index = idx,
            daily_rets = last_year_ret_base
        )
        
        rg_adj = self._score(
            func = self.revenue_growth_score,
            index = idx,
            rev_growth = rev_growth, 
            ind_rvg = ind_rvg, 
            rev = rev, 
            rev_pred = rev_1y
        )
        
        roe_adj = self._score(
            func = self.return_on_equity_score,
            index = idx,
            roe = roe, 
            ind_roe = ind_roe
        )
        
        roa_adj = self._score(
            func = self.return_on_assets_score,
            index = idx,
            roa = roa, 
            prev_roa = prev_roa,
            ind_roa = ind_roa
        )
        
        pb_adj = self._score(
            func = self.price_to_book_score,
            index = idx,
            pb = pb, 
            ind_pb = ind_pb
        )
        
        eps_adj = self._score(
            func = self.ep_score,
            index = idx,
            trailing_eps = teps, 
            forward_eps = feps, 
            price = price, 
            ind_pe = ind_pe, 
            eps_1y = eps_1y
        )
        
        up_down_adj = self._score(
            func = self.upside_downside_score,
            index = idx,
            hist_rets = last_5y_ret_base, 
            bench_hist_rets = benchmark_weekly_rets_base, 
        )

        lower_target_adj = self._score(
            func = self.lower_target_adjustment,
            index = idx,
            lower_target = lower_target, 
            price = price
        )
        
        rec_adj = self._score(
            func = self.recommendation_adjustment,
            index = idx,
            recommendation = recommendation
        )
        
        insider_adj = self._score(
            func = self.insider_purchases_adjustment,
            index = idx,
            insider_purchases = insider_purchases
        )
        
        net_income_adj = self._score(
            func = self.net_income_positive_adjustment,
            index = idx,
            net_income = net_income
        )
        
        ocf_adj = self._score(
            func = self.ocf_gt_net_income_adjustment,
            index = idx,
            operating_cf = operating_cashflow, 
            net_income = net_income
        )
        
        ld_adj = self._score(
            func = self.long_debt_improvement_adjustment,
            index = idx,
            prev_long_debt = prev_long_debt,
            long_debt = long_debt
        )
        
        cr_adj = self._score(
            func = self.current_ratio_improvement_adjustment,
            index = idx,
            prev_cr = prev_current_ratio, 
            curr_cr = current_ratio
        )
        
        ags_adj = self._score(
            func = self.asset_growth_rate_sharpe,
            index = idx,
            asset_growth_rate = asset_growth_rate,
            asset_growth_rate_vol = asset_growth_rate_vol
        )
        
        no_new_shares_adj = self._score(
            func = self.no_new_shares_adjustment,
            index = idx,
            shares_issued = shares_issued
        )
        
        de_adj = self._score(
            func = self.debt_to_equity_adjustment,
            index = idx,
            d_to_e = d_to_e
        )
        
        gm_adj = self._score(
            func = self.gross_margin_improvement_adjustment,
            index = idx,
            gm = gross_margin,
            prev_gm = prev_gm
        )
        
        at_adj = self._score(
            func = self.asset_turnover_improvement_adjustment,
            index = idx,
            at = at,
            prev_at = prev_at
        )
        
        skew_adj = self._score(
            func = self.skewness_adjustment,
            index = idx,
            r = last_5y_ret_base
        )
        
        kurt_adj = self._score(
            func = self.kurtosis_adjustment,
            index = idx,
            r = last_5y_ret_base
        )
        
        sharpe_adj = self._score(
            func = self.sharpe_adjustment,
            index = idx,
            r = last_year_ret_base, 
            n = last_year_period
        )
        
        sortino_adj = self._score(
            func = self.sortino_adjustment,
            index = idx,
            r = last_year_ret_base, 
            n = last_year_period
        )
        
        alpha_adj, pred_alpha_adj = self.alpha_adjustments(
            hist_rets = last_5y_ret_base,
            benchmark_ann_ret = benchmark_ann_ret_5y,
            comb_ret = comb_rets,
            last_year_ret = last_year_ret_base,
            bench_hist_rets = benchmark_weekly_rets_base,
            periods_per_year = 52,
            d_to_e = d_to_e,
            tax = tax
        )
        
        eps_rev_score = self._score(
            func = self.eps_revision_score,
            index = idx,
            eps_rev_7 = eps_rev_7,
            eps_rev_30 = eps_rev_30
        )
        
        gpa_adj = self._score(
            func = self.gross_profitability_score,
            index = idx,
            gross_margin = gross_margin,
            asset_turnover = at
        )

        aga_adj = self._score(
            func = self.asset_growth_anomaly_score,
            index = idx,
            asset_growth = asset_growth_rate
        )

        sh_yield_adj = self._score(
            func = self.shareholder_yield_score,
            index = idx,
            div_yield = div,
            net_issuance = shares_issued,
            market_cap = mc
        )

        accruals_adj = self._score(
            func = self.accruals_score,
            index = idx,
            net_income = net_income,
            ocf = operating_cashflow,
            total_revenue = rev,
            asset_turnover = at
        )
        
        alpha_adj = alpha_adj.reindex(common_idx).fillna(0)
        
        pred_alpha_adj = pred_alpha_adj.reindex(common_idx).fillna(0)
        
        signal_adj = self._score(
            func = self.signal_scores_adjustment,
            index = idx,
            signal_scores = self.signal_scores, 
            universe = idx
        )
        
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
            'Debt-to-Equity': de_adj,
            'Long-Debt Improvement': ld_adj,
            'Current-Ratio Improvement': cr_adj,
            'Asset-Growth-Rate Sharpe': ags_adj,
            'Gross Profitability': gpa_adj,
            'Asset Growth Anomaly': aga_adj,
            'Shareholder Yield': sh_yield_adj,
            'Accruals Quality': accruals_adj,
            'No New Shares Issued': no_new_shares_adj,
            'Gross-Margin Improvement': gm_adj,
            'Asset-Turnover Improvement': at_adj,
            'Beta Adjustment': beta_adj,
            'PEG Adjustment': peg_adj,
            'Vol Trend Adjustment': vol_trend_adj,
            'Skewness': skew_adj,
            'Kurtosis': kurt_adj,
            'Sharpe Ratio': sharpe_adj,
            'Sortino Ratio': sortino_adj,
            'Alpha': alpha_adj,
            'Pred Alpha': pred_alpha_adj,
            'EPS Revision Score': eps_rev_score,
            'Signal Scores': signal_adj,
        }).reindex(index = common_idx) 

        final_scores = score_breakdown.sum(axis = 1)
       
        score_breakdown['Final Score'] = final_scores

        return comb_rets, comb_stds, final_scores, weights, score_breakdown


def safe_region_series(
    metric_dict: Mapping[Hashable, Mapping[str, Any]],
    tickers: Sequence[Hashable],
    key: str = 'Region-Industry',
    default: float = np.nan,
) -> pd.Series:
    """
    Extract a region/industry baseline field from a nested mapping for a given
    ordered ticker universe.

    Mapping rule
    ------------
    For each ticker t in `tickers`:

        x_t = metric_dict[t][key], if available;
        x_t = default, otherwise.

    The resulting vector x is returned as a Series indexed in the same order as
    `tickers`.

    Parameters
    ----------
    metric_dict : Mapping[Hashable, Mapping[str, Any]]
        Nested dictionary keyed first by ticker and then by metric field.
    tickers : Sequence[Hashable]
        Target ticker ordering.
    key : str, optional
        Field name to retrieve from each ticker-level mapping.
    default : float, optional
        Fallback value used when ticker or field is absent.

    Returns
    -------
    pd.Series
        Extracted baseline metric aligned to `tickers`.

    Advantages
    ----------
    The helper centralises sparse-dictionary extraction and guarantees stable
    ordering, which prevents alignment errors in subsequent cross-sectional
    scoring functions.
    """

    return pd.Series(
        [metric_dict.get(t, {}).get(key, default) for t in tickers],
        index = tickers
    )


def macro_to_weekly_returns(
    macro_weekly: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert a mixed weekly macro panel (levels, rates, and returns) into a
    coherent return-like panel using variable-specific transforms.

    Transformation equations
    -----------------------
    For a macro series x_t:

    - Known return-like variables:

          r_t = x_t.

    - Rate-like variables:

          r_t = x_t - x_t-1.

    - Remaining level-like variables:

          r_t = (x_t / x_t-1) - 1.

    Rows that are entirely missing after conversion are removed.

    Parameters
    ----------
    macro_weekly : pd.DataFrame
        Weekly macro dataset containing heterogeneous variable types.

    Returns
    -------
    pd.DataFrame
        Converted panel in return-compatible units.

    Modelling rationale
    -------------------
    Percentage changes are appropriate for strictly positive index-like levels,
    whereas differencing is more meaningful for quoted rates and yields.
    Combining both rules preserves economic interpretation while producing
    numerically consistent inputs for covariance targets.
    """

    if macro_weekly is None or macro_weekly.empty:
        return macro_weekly

    df = macro_weekly.copy()

    already_returns = {
        "CL=F", "GC=F", "HG=F", "^VIX",
        "US_GDP", "US_Consumer_Sentiment", "US_Wages",
    }

    rate_like = {
        "^TNX", "^IRX", "Inflation", "US_Unemployment_Rate",
    }

    rets = df.pct_change()

    for col in already_returns.intersection(df.columns):
  
        rets[col] = df[col]

    for col in rate_like.intersection(df.columns):
    
        rets[col] = df[col].diff()

    return rets.dropna(how = "all")
    

def common_index(
    series_list: Sequence[pd.Series]
) -> pd.Index:
    """
    Compute the sorted intersection of index labels across multiple Series.

    Mathematical definition
    -----------------------
    For index sets I_1, I_2, ..., I_n from `series_list`, the function returns:

        I_common = I_1 intersect I_2 intersect ... intersect I_n,

    sorted in ascending order.

    Parameters
    ----------
    series_list : Sequence[pd.Series]
        Input series whose shared support is required.

    Returns
    -------
    pd.Index
        Sorted common index. If no input series are supplied, an empty index is
        returned.

    Advantages
    ----------
    Enforcing a strict common support avoids silent broadcasting and missing-
    data artefacts when constructing combined forecasts and additive scores.
    """
    
    if not series_list:
    
        return pd.Index([])
    
    idx = series_list[0].index
    
    for s in series_list[1:]:
    
        idx = idx.intersection(s.index)
    
    return idx.sort_values()

    
def build_fx_factor_returns(
    ratio_data: RatioData,
    tickers: Sequence[str],
    weekly_index: pd.DatetimeIndex,
    base_ccy: str,
) -> pd.DataFrame | None:
    """
    Build weekly FX factor returns for non-base currencies represented in the
    asset universe.

    Construction method
    -------------------
    1. Infer ticker currencies from `ratio_data`.
   
    2. Request local-to-base FX price series for each currency.
    
    3. Convert each price series to weekly returns:

           r_fx,c,t = (P_c,t / P_c,t-1) - 1.

    4. Store each series in a factor column named `FX_<CCY>`.

    Parameters
    ----------
    ratio_data : RatioData
        Data source providing ticker-currency mapping and FX converter series.
    tickers : Sequence[str]
        Asset universe from which currencies are inferred.
    weekly_index : pd.DatetimeIndex
        Target weekly index for factor alignment.
    base_ccy : str
        Base currency code.

    Returns
    -------
    pd.DataFrame | None
        FX factor return matrix indexed by `weekly_index`, or `None` if inputs
        are unavailable or conversion fails.

    Advantages
    ----------
    Explicit FX factors isolate currency co-movement from local return
    dynamics, improving factor-model coverage in multi-currency covariance
    estimation.
    """

    try:
 
        col_ccy = ratio_data._ticker_ccy(pd.Index(tickers)).fillna(base_ccy)
 
    except Exception:
 
        return None

    try:
 
        fx_by_ccy = ratio_data._fx_converters_by_ccy(
            ccys = col_ccy,
            base_ccy = base_ccy,
            interval = "1wk",
            index = weekly_index,
        )
 
    except Exception:
 
        return None

    fx_rets = {}

    for ccy, fx_price in fx_by_ccy.items():

        if ccy == base_ccy:
 
            continue

        s = fx_price.pct_change().reindex(weekly_index).fillna(0.0)

        fx_rets[f"FX_{ccy}"] = s

    if not fx_rets:
 
        return None

    return pd.DataFrame(fx_rets, index = weekly_index)


def main() -> None:
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
    
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s",
    )    
    
    logging.info('Loading data...')
    
    ratio_data = RatioData()
    
    optimiser = PortfolioOptimiser(
        excel_file = config.FORECAST_FILE, 
        ratio_data = ratio_data
    )

    sector_map = None
  
    industry_map = None
  
    if hasattr(optimiser, "analyst_df"):
  
        if "Sector" in optimiser.analyst_df.columns:
  
            sector_map = optimiser.analyst_df["Sector"]
  
        if "Industry" in optimiser.analyst_df.columns:
  
            industry_map = optimiser.analyst_df["Industry"]

    metrics = ratio_data.dicts()
    
    tickers = optimiser.tickers
    
    daily_close = ratio_data.close
    
    daily_open = ratio_data.open
    
    macro_weekly = ratio_data.macro_data
    
    daily_ret = ratio_data.daily_rets
    
    weekly_ret = ratio_data.weekly_rets
    
    region_ind = {
        'PE': safe_region_series(
            metric_dict = metrics['PE'],        
            tickers = tickers
        ),
        'PB': safe_region_series(
            metric_dict = metrics['PB'],        
            tickers = tickers
        ),
        'ROE': safe_region_series(
            metric_dict = metrics['ROE'],        
            tickers = tickers
        ),
        'ROA': safe_region_series(
            metric_dict = metrics['ROA'],        
            tickers = tickers
        ),
        'RevG': safe_region_series(
            metric_dict = metrics['rev1y'],    
            tickers = tickers
        ),
        'EarningsG': safe_region_series(
            metric_dict = metrics['eps1y'],    
            tickers = tickers
        ),
    }

    logging.info('Computing combination forecast...')
    
    comb_rets, comb_stds, final_scores, weights, score_breakdown = (
        optimiser.compute_combination_forecast(
            region_indicators = region_ind
        )
    )

    cov_idx = comb_rets.index

    for t in tickers:
        
        print(f'{t}:    ', comb_rets[t])
    
    daily_close_5y = daily_close.loc[
        daily_close.index >= pd.to_datetime(config.FIVE_YEAR_AGO),
        cov_idx,
    ]
    
    daily_open_5y = daily_open.loc[
        daily_open.index >= pd.to_datetime(config.FIVE_YEAR_AGO),
        cov_idx,
    ]
    
    macro_weekly_5y = macro_weekly.loc[
        macro_weekly.index >= pd.to_datetime(config.FIVE_YEAR_AGO),
        :,
    ].astype("float32").copy()

    daily_ret_5y = daily_ret.loc[
        daily_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO),
        cov_idx,
    ]
  
    weekly_ret_5y = weekly_ret.loc[
        weekly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO),
        cov_idx,
    ]
  
    monthly_ret = daily_close_5y.resample('ME').last().pct_change().dropna()
    
    monthly_ret_5y = monthly_ret.loc[
        monthly_ret.index >= pd.to_datetime(config.FIVE_YEAR_AGO)
    ]
    
    factor_weekly, index_weekly, ind_weekly, sec_weekly = ratio_data.factor_index_ind_sec_weekly_rets(
        merge = False
    )
    
    other_tickers = ['CL=F', 'GC=F', 'HG=F', '^VIX']
    
    macro_weekly_rets = macro_to_weekly_returns(
        macro_weekly = macro_weekly_5y
    )
    
    macro_cols = [c for c in other_tickers if c in macro_weekly_rets.columns]
    
    macro_ret = macro_weekly_rets[macro_cols] if macro_cols else pd.DataFrame(index = macro_weekly_rets.index)
        
    base_ccy = getattr(config, "BASE_CCY", "USD")
    
    daily_ret_5y_base = ratio_data.convert_returns_to_base(
        ret_df = daily_ret_5y,
        base_ccy = base_ccy,
        interval = "1d",
    )
    
    macro_ret_base = ratio_data.convert_returns_to_base(
        ret_df = macro_ret,
        base_ccy = base_ccy,
        interval = "1wk",
    )
        
    if macro_cols:
        macro_weekly_rets.loc[:, macro_cols] = macro_ret_base
    
    daily_close_5y_base = ratio_data.convert_prices_to_base(
        price_df = daily_close_5y,
        base_ccy = base_ccy,
        interval = "1d"
    )
    
    daily_open_5y_base = ratio_data.convert_prices_to_base(
        price_df = daily_open_5y,
        base_ccy = base_ccy,
        interval = "1d"
    )
    
    weekly_ret_5y_base = ratio_data.convert_returns_to_base(
        ret_df = weekly_ret_5y,
        base_ccy = base_ccy,
        interval = "1wk"
    )
    
    monthly_ret_5y_base = ratio_data.convert_returns_to_base(
        ret_df = monthly_ret_5y,
        base_ccy = base_ccy,
        interval = "1mo"
    )

    fund_exposures_weekly = None
    
    if getattr(config, "COV_USE_FUND_FACTORS", True):

        try:

            fdata = FinancialForecastData(tickers = list(comb_rets.index), quiet = True)

            weekly_prices = ratio_data.weekly_close.reindex(
                index = weekly_ret_5y_base.index
            ).reindex(columns = comb_rets.index)
          
            shares_out = getattr(ratio_data, "shares_outstanding", pd.Series(index = comb_rets.index, dtype = float))
            
            fund_exposures_weekly = fdata.build_pit_fundamental_exposures_weekly(
                tickers = list(comb_rets.index),
                weekly_index = weekly_ret_5y_base.index,
                weekly_prices = weekly_prices,
                shares_outstanding = shares_out,
                report_lag_days = getattr(config, "FUND_REPORT_LAG_DAYS", 90),
                price_lag_weeks = getattr(config, "FUND_PRICE_LAG_WEEKS", 1),
                winsor_p = getattr(config, "FUND_WINSOR_P", (0.05, 0.95)),
                robust_scale = getattr(config, "FUND_ROBUST_SCALE", "mad"),
                min_coverage = getattr(config, "FUND_MIN_COVERAGE", 0.6),
            )
     
        except Exception as exc:
     
            logging.warning("Failed to build PIT fundamental exposures: %s", exc)

    fx_factors_weekly = build_fx_factor_returns(
        ratio_data = ratio_data,
        tickers = comb_rets.index,
        weekly_index = weekly_ret_5y_base.index,
        base_ccy = base_ccy,
    )
    
    factor_weekly_base = ratio_data.convert_returns_to_base(
        ret_df = factor_weekly, 
        base_ccy = base_ccy,
        interval = "1wk"
    )
    
    index_weekly_base = ratio_data.convert_returns_to_base(
        ret_df = index_weekly,
        base_ccy = base_ccy,
        interval = "1wk"
    )
    
    ind_weekly_base = ratio_data.convert_returns_to_base(
        ret_df = ind_weekly,
        base_ccy = base_ccy, 
        interval = "1wk"
    )
    
    sec_weekly_base = ratio_data.convert_returns_to_base(
        ret_df = sec_weekly,
        base_ccy = base_ccy, 
        interval = "1wk"
    )
    
    print('Building covariance matrix...')

    solve_method = getattr(config, "COV_SOLVE_METHOD", "cvxpy")

    macro_factors_weekly = (
        macro_weekly_rets
        if (macro_weekly_rets is not None and not macro_weekly_rets.empty)
        else None
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
        macro_factors_weekly = macro_factors_weekly,
        fx_factors_weekly = fx_factors_weekly,
        fund_exposures_weekly = fund_exposures_weekly,
        sector_map = sector_map,
        industry_map = industry_map,
        daily_open_5y = daily_open_5y_base,
        daily_close_5y = daily_close_5y_base,
        use_excess_ff = False,
        use_log_returns = getattr(config, "COV_USE_LOG_RETURNS", True),
        use_oas = getattr(config, "COV_USE_OAS", True),
        use_block_prior = getattr(config, "COV_USE_BLOCK_PRIOR", True),
        use_regime_ewma = getattr(config, "COV_USE_REGIME_EWMA", True),
        use_glasso = getattr(config, "COV_USE_GLOSSO", True),
        use_fund_factors = getattr(config, "COV_USE_FUND_FACTORS", True),
        use_fx_factors = getattr(config, "COV_USE_FX_FACTORS", True),
        use_factor_term_structure = getattr(config, "COV_USE_TERM_STRUCTURE", False),
        solve_method = solve_method,
    )

    cov_path = config.BASE_DIR / f"cov_matrix_{config.TODAY}.pkl"
  
    try:
  
        cov.to_pickle(cov_path)
  
    except Exception as exc:
  
        logging.warning("Failed to save covariance to %s: %s", cov_path, exc)

    var = pd.Series(np.diag(cov), index = cov.index)
    
    std = np.sqrt(var).clip(lower = MIN_STD, upper = MAX_STD)

    idx = optimiser.latest_prices.index.sort_values()
    
    price = optimiser.latest_prices.reindex(idx)
    
    bull = (comb_rets + 1.96 * std).clip(config.lbr, config.ubr)
  
    bear = (comb_rets - 1.96 * std).clip(config.lbr, config.ubr)
    
    margin_safety = 1.96 * std / np.sqrt(252)
    
    month_ago = pd.to_datetime(config.TODAY) - pd.DateOffset(months=3)

    last_month_daily_ret_base = daily_ret_5y_base.loc[
        daily_ret_5y_base.index >= month_ago,
        comb_rets.index,
    ]

    alpha = 0.05  
    
    hist_var_daily = last_month_daily_ret_base.quantile(alpha, axis = 0)

    tail_mask = last_month_daily_ret_base.le(hist_var_daily, axis = "columns")
    
    tail_losses = last_month_daily_ret_base.where(tail_mask)

    hist_etl_daily = tail_losses.mean(axis = 0, skipna=True)

    hist_etl_margin = (-hist_etl_daily).clip(lower = 0.0)

    margin_safety = hist_etl_margin.reindex(comb_rets.index).fillna(0.0)
  
    safe_returns = comb_rets - margin_safety
    
    df = pd.DataFrame({
            'Ticker': idx,
            'Current Price': price,
            'Avg Price': np.round(price * (comb_rets + 1), 2),
            'Low Price': np.round(price * (bear + 1), 2),
            'High Price': np.round(price * (bull + 1), 2),
            'Returns': comb_rets,
            'Safe Returns': safe_returns,
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
        'Score Breakdown': score_breakdown,
    }
    
    export_results(
        sheets = sheets_to_upload, 
        output_excel_file = config.PORTFOLIO_FILE
    )
    
    logging.info('Done.')
    

if __name__ == '__main__':

    main()
