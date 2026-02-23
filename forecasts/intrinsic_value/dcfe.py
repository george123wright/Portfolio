"""
Discounted cash-flow to equity (DCFE) valuation with robust, constrained regression and PE-based terminals

Purpose
-------
This module estimates per-share equity values using a discounted cash-flow to equity
(DCFE) framework. Forecast revenue/EPS paths are combined with a robust,
constrained regression that maps fundamentals to free cash flow to equity (FCFE).
Period FCFE is discounted on an irregular calendar using a flat cost of equity,
and a price/earnings (PE)–based terminal value is appended. Multiple forecast
paths and terminal PE candidates are evaluated; the resulting distribution is
summarised into low/average/high price points and a standard error (SE).
Results are exported to Excel.

Data inputs and provenance
--------------------------
- Historical annual financials per ticker with at least:
  ["Revenue", "EPS", "OCF", "NetBorrowing", "Capex"].

- Forecast frame per ticker with, per period:
  low/average/high revenue and EPS, and analyst counts:
  ['low_rev','avg_rev','high_rev','low_eps','avg_eps','high_eps','num_analysts'].

- KPI frame with expected PE (exp_pe) and dictionaries for industry PE and 10-year
  industry PE benchmarks.

- Cost of equity (COE) per ticker from the workbook sheet 'COE'.

- Shares outstanding and latest prices from `fdata.macro.r`.

Pre-processing and normalisation
--------------------------------
1) Forecast cleaning
  
   - Revenue strings suffixed with magnitudes are coerced to floats via:
   
     "T" → "e12", "B" → "e9", "M" → "e6", then `float`.
   
   - Index is coerced to `DatetimeIndex` and sorted ascending.

2) Regression target construction
   
   - From the historical financials, build:
    
     X_t = [Revenue_t, EPS_t]
     
     y_t = OCF_t + NetBorrowing_t + Capex_t
     
     Notes on sign conventions:
     
       Many definitions use FCFE_t = CFO_t − Capex_t + NetBorrowing_t.
       If Capex is stored as a negative spend (as in the data used here),
       y_t as written matches the intended sign.

Modelling technique: robust constrained regression
--------------------------------------------------
The mapping from fundamentals to FCFE is approximated linearly and estimated
with a Huberised Elastic-Net model selected by time-series cross-validation:

- Model: FCFE_t ≈ β0 + β1·Revenue_t + β2·EPS_t

- Estimation: `HuberENetCV` with grids over:

    • alpha (elastic-net mixing),
    
    • lambda (penalty strength),
    
    • M (Huber loss parameter),
  
  using `TimeSeriesSplit` to respect temporal ordering.

- Optionally, constraints can be imposed on targets (e.g., positivity) via
  the `constrained_map` settings passed through to the joint fit routine.

Scenario construction: path enumeration
---------------------------------------
For a horizon of T forecast periods, low/average/high paths are enumerated for
both revenue and EPS:

- For each t in {0,…,T−1}:

    Revenue_t ∈ {low_rev_t, avg_rev_t, high_rev_t}  
    EPS_t     ∈ {low_eps_t, avg_eps_t, high_eps_t}

- The Cartesian product yields P = 3^T distinct paths.

- Path matrices have shape (P, T):
    rev[i, t] = chosen revenue for path i at period t
    eps[i, t] = chosen EPS     for path i at period t

Cash-flows, discounting, and present values
-------------------------------------------
1) FCFE per path and period:
 
   FCFE_{i,t} = β0 + β1·Revenue_{i,t} + β2·EPS_{i,t}

2) Irregular-calendar discount factors (flat COE):
  
   For each forecast date τ_t and valuation date τ_0 (today),
  
     days_t = (τ_t − τ_0) in days
     
     Δ_t    = days_t / 365
     
     disc_t = (1 + COE)^(−Δ_t)
   
   Discount factors form a vector disc ∈ ℝ^T aligned with the forecast index.

3) Present value per period and horizon sum:
  
   DCF_{i,t} = FCFE_{i,t} × disc_t
   
   SumDCF_i  = Σ_{t=0}^{T−1} DCF_{i,t}

Terminal value via PE multiples
-------------------------------
A PE-based terminal value is formed for each path using the final-year EPS and
up to three PE candidates, then discounted to present using the last discount
factor disc_T:

- Candidate multiples, in order of preference:
    PE_used  = exp_pe (if positive) else Industry-MC
    PE_ind   = Region-Industry (current)
    PE_10y   = Region-Industry (10-year)
- For candidate j and path i:
    TV_raw[j,i]  = PE_j × EPS_{i,T−1} × SharesOut
    TV_disc[j,i] = TV_raw[j,i] × disc_T
- Stacking over candidates yields TV_disc ∈ ℝ^{K×P}, K ∈ {1,2,3}.

Equity value per path and terminal candidate
--------------------------------------------
For candidate j and path i:
 
  V_{j,i} = SumDCF_i + TV_disc[j,i]

In array form:
 
  dcfe_vals = SumDCF[None, :] + TV_disc

so dcfe_vals has shape (K, P).

Per-share prices and clipping
-----------------------------
- Convert to per-share prices via:
   
    Price_{j,i} = dcfe_vals[j,i] / SharesOut

- Clip each price to user-configured bounds per ticker:
    lb = lower bound multiple × latest price
    ub = upper bound multiple × latest price
  so that: Price_{j,i} ← min( max(Price_{j,i}, lb), ub )

Uncertainty quantification (standard error)
-------------------------------------------
A conservative SE combines cross-path dispersion of discounted period cash-flows
and dispersion of terminal values, with analyst-count scaling:

1) Per-period dispersion over paths:
 
   σ_t = std( DCF[:, t], ddof = 1 )
   
   n_t = analyst count at period t (non-positive or NaN coerced to 1)
   
   SE_t = σ_t / sqrt(n_t)

2) Terminal dispersion over all candidates and paths:
   
   σ_term = std( vec(TV_disc), ddof = 1 )
   
   n_term = analyst count at the terminal year (≤ 0 → 1)
   
   SE_term = σ_term / sqrt(n_term)

3) Combine in quadrature and convert to per-share:
   
   SE_total = sqrt( Σ_t SE_t^2 + SE_term^2 )
   
   SE_price = SE_total / SharesOut

Outputs
-------
- For each ticker:
  • Low Price = max over finite clipped prices’ minimum, floored at 0.
  • Avg Price = max over finite clipped prices’ mean, floored at 0.
  • High Price = max over finite clipped prices’ maximum, floored at 0.
  • SE = SE_price from the uncertainty routine.
- A DataFrame with columns ['Low Price', 'Avg Price', 'High Price', 'SE']
  is written to the 'DCFE' sheet of `config.MODEL_FILE`.

Operational notes
-----------------
- Logging reports per-ticker price points and SE.

- Missing or invalid inputs (e.g., absent columns, zero/NaN shares, empty
  forecasts) result in skips with logged warnings.

- Analyst counts are used as a heuristic precision weight in SE computation,
  not as a formal sampling model.

Reproducibility
---------------
- Given fixed inputs, the regression cross-validation and valuation are
  deterministic.
- Date alignment matters: discount factors depend on the forecast index and the
  valuation date `config.TODAY`.

This module is intended for analytical support. It is not investment advice.
"""

import pandas as pd
import numpy as np
import logging
from typing import List
from financial_forecast_data5 import FinancialForecastData
from fast_regression6 import HuberENetCV
from export_forecast import export_results 

from sklearn.model_selection import TimeSeriesSplit
import itertools
 
import config

from pathlib import Path

import beta_cache as bc

RUN_REGRESSION = False 

BETA_CACHE_MODEL = "dcfe"

BETA_CACHE_FILE = Path(config.BETA_CACHE_FILE)
 
REQUIRED_COLUMNS = ["Revenue", "EPS", "OCF", "NetBorrowing", "Capex"]

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

TODAY_TS = pd.Timestamp(config.TODAY)

fdata = FinancialForecastData()

macro = fdata.macro

r = macro.r


def _clean_fc_df(
    fc_df: pd.DataFrame
 ) -> pd.DataFrame:
    """
    Clean and normalize the forecast DataFrame.

    **Type hints**
    ------------
    fc_df : pandas.DataFrame
        A forecast frame indexed by (parseable) dates, containing revenue columns
        ['low_rev', 'avg_rev', 'high_rev'] that may include textual suffixes, plus
        EPS columns such as ['low_eps', 'avg_eps', 'high_eps'].
    Returns
    -------
    pandas.DataFrame
        The same DataFrame with:
    
          (i) revenue columns coerced to float, mapping textual magnitudes
              {'T'→1e12, 'B'→1e9, 'M'→1e6},
    
         (ii) a DatetimeIndex (`pd.to_datetime`),
    
        (iii) rows sorted by date (ascending).

    **Method & transformations**
    ----------------------------
    For each revenue column `c ∈ {low_rev, avg_rev, high_rev}`:
    
      • If dtype is object (string), apply the magnitude mapping via a regex replace:
            "xT" → "x e12", "xB" → "x e9", "xM" → "x e6",
        then coerce to float (`errors='coerce'`).
    
      • Otherwise, coerce directly to float.

    The index is coerced to a `DatetimeIndex`, and the frame is sorted by index.

    **Notes**
    --------
    • Entries that cannot be parsed are set to NaN.
    • This function does *not* drop rows; callers should handle missingness later.
    """
    
    for col in ['low_rev', 'avg_rev', 'high_rev']:
    
        s = fc_df[col]
       
        if s.dtype == 'O':
       
            fc_df[col] = pd.to_numeric(s.replace({
                'T': 'e12',
                'B': 'e9',
                'M': 'e6'
            }, regex = True), errors = 'coerce')
       
        else:
       
            fc_df[col] = pd.to_numeric(s, errors = 'coerce')
   
    fc_df.index = pd.to_datetime(fc_df.index)
    
    fc_df = fc_df.sort_index()  
    
    return fc_df


def _get_float(
    d: dict,
    key: str,
    default = np.nan
) -> float:
    """
    Safely extract a numeric value from a mapping.

    **Type hints**
    ------------
    d : dict
        Source dictionary (e.g., industry PE dictionary).
    key : str
        Key to retrieve.
    default : float, optional (default = np.nan)
        Value to return when key is absent or not finite.

    Returns
    -------
    float
        A finite float if possible; otherwise `default`.

    **Method**
    ----------
    Fetch `v = d.get(key, default)`, attempt `float(v)`, and return it if finite
    (`np.isfinite`). On any exception or non-finite value, return `default`.

    **Notes**
    --------
    • This shields downstream math (e.g., PE terminals) from KeyError / NaN cascades.
    """
       
    try:
   
        v = d.get(key, default)
   
        v = float(v)
        
        if np.isfinite(v):
   
            return v 
        
        else:
            
            return default
   
    except Exception:
   
        return default


def regression_data_prep(
    fin_df: pd.DataFrame,
    required_columns: List = REQUIRED_COLUMNS,
 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare regression design and target from historical financials.

    **Type hints**
    ------------
    fin_df : pandas.DataFrame
        Historical financials with at least the columns in `required_columns`.
    required_columns : list[str]
        Columns required to compute the target and features. Default:
      
        ["Revenue", "EPS", "OCF", "NetBorrowing", "Capex"].

    Returns
    -------
    X : numpy.ndarray, shape (N, 2), dtype=float
        Feature matrix with columns [Revenue, EPS].
    y : numpy.ndarray, shape (N,), dtype=float
    
        Target vector defined as:
        
            y_t = OCF_t + NetBorrowing_t + Capex_t
        
        (see Notes regarding Capex sign convention).

    **Equations**
    -------------
    Let for period t:
  
        X_t = [ Revenue_t , EPS_t ]
  
        y_t = OCF_t + NetBorrowing_t + Capex_t
  
    Then:
  
        X = [X_1; X_2; …; X_N],  y = [y_1, y_2, …, y_N]^⊤

    **Notes**
    --------
    • Many FCFE formulas are y_t = CFO_t − CapEx_t + NetBorrowing_t.
      If `Capex` is stored as a positive spend, the correct sign is negative; if it is
      stored negative, the above additive form is equivalent. The data that I use has capex as 
      negative.

    • Rows with any NaNs in `required_columns` are dropped prior to construction.
    """
       
    reg = fin_df.dropna(subset = required_columns)
   
    X = reg[['Revenue', 'EPS']].to_numpy(dtype = float)
   
    y = (reg['OCF'] + reg['NetBorrowing'] + reg['Capex']).to_numpy(dtype = float)
    
    return X, y


def build_rev_eps_matrices(
    fc_df: pd.DataFrame,
    years: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all low/avg/high revenue and EPS paths across the forecast horizon.

    **Type hints**
    ------------
    fc_df : pandas.DataFrame
        Forecast frame containing columns
        ['low_rev','avg_rev','high_rev','low_eps','avg_eps','high_eps'] in each row.
    years : int
        Number of forecast periods T (should match len(fc_df)).

    Returns
    -------
    rev_matrix : numpy.ndarray, shape (3^T, T), dtype=float
        Matrix of all revenue paths, where each row is a path and each column a year.
    eps_matrix : numpy.ndarray, shape (3^T, T), dtype=float
        Matrix of all EPS paths, aligned with `rev_matrix`.

    **Combinatorics & equations**
    -----------------------------
    For each period t ∈ {0,…,T−1}, we have three options:
        Revenue_t ∈ { low_rev_t, avg_rev_t, high_rev_t }
        EPS_t     ∈ { low_eps_t, avg_eps_t, high_eps_t }

    All path indices are formed via the Cartesian product:
        I = {0,1,2}^T  (3 choices per period)
        |I| = 3^T

    With `idxs ∈ I`, the pathwise selections are:
        rev_matrix[i, t] = rev_opts[t, idxs[i, t]]
        eps_matrix[i, t] = eps_opts[t, idxs[i, t]]
    """
    
    for col in ['low_rev', 'avg_rev', 'high_rev']:
        
        fc_df[col] = fc_df[col].replace({'T':'e12', 'B':'e9', 'M': 'e6'}, regex = True).astype(float)

    rev_opts = fc_df[['low_rev', 'avg_rev', 'high_rev']].to_numpy(dtype = float)
    
    eps_opts = fc_df[['low_eps', 'avg_eps', 'high_eps']].to_numpy(dtype = float)

    idxs = np.array(list(itertools.product(range(3), repeat = years)), dtype = int)
    
    rev_matrix = np.take_along_axis(rev_opts, idxs.T, axis = 1).T
    
    eps_matrix = np.take_along_axis(eps_opts, idxs.T, axis = 1).T
    
    return rev_matrix, eps_matrix


def terminal_value_pe(
    final_eps: np.ndarray,
    shares_out: float,
    exp_pe: float,
    pe_ind_dict: dict,
    pe_10y_ind_dict: dict,
    disc_T: float,
) -> np.ndarray:
    """
    Construct discounted PE-based terminal values for each EPS path using up to
    three PE multiples, then stack them as alternative terminal candidates.

    **Type hints**
    ------------
    final_eps : numpy.ndarray, shape (P,), dtype=float
        The last-year EPS along each path (P = number of paths).
    shares_out : float
        Shares outstanding (scalar).
    exp_pe : float
        Company-specific expected PE. If `exp_pe <= 0`, fallback is used.
    pe_ind_dict : dict
        Dictionary with at least keys 'Industry-MC' and 'Region-Industry' (floats).
    pe_10y_ind_dict : dict
        Dictionary with at least key 'Region-Industry' (10-year industry PE).
    disc_T : float
        Final discount factor that maps terminal nominal value to present value.

    Returns
    -------
    tv_disc : numpy.ndarray, shape (K, P), dtype=float
        Discounted terminal values where K ∈ {1,2,3} is the number of valid PE
        candidates constructed in the order:
          1) `pe_used` := (exp_pe if exp_pe>0 else pe_ind_dict['Industry-MC'])
          2) `pe_ind`  := pe_ind_dict['Region-Industry']
          3) `pe_10y`  := pe_10y_ind_dict['Region-Industry']

    **Equations**
    -------------
    For each candidate multiple PE_j and path i:
        TV_raw[j, i]  = PE_j × EPS_T(i) × SharesOut
        TV_disc[j, i] = TV_raw[j, i] × disc_T

    Hence:
        tv_disc = [TV_disc(j, :)]_{j=1..K}  ∈ ℝ^{K×P}

    **Notes**
    --------
    • `disc_T` should be the *last* entry from `build_discount_factor_vector`.
    """    
    
    if exp_pe > 0:
    
        pe_used = exp_pe  
    
    else:
        
        pe_used = pe_ind_dict['Industry-MC']
        
    pe_ind = _get_float(
        d = pe_ind_dict,
        key = 'Region-Industry'
    )
    
    pe_10y_ind = _get_float(
        d = pe_10y_ind_dict,
        key = 'Region-Industry'
    ) 
    
    pe_list = np.array([pe_used, pe_ind, pe_10y_ind], dtype = float)
    
    tv_raw = pe_list[:, None] * final_eps[None, :] * shares_out
    
    tv_disc = tv_raw * disc_T
    
    return tv_disc


def build_discount_factor_vector(
    fc_df: pd.DataFrame,
    coe_t: float,
    today_ts: pd.Timestamp = TODAY_TS,
) -> np.ndarray:
    """
    Build per-period discount factors from a potentially irregular forecast calendar.

    **Type hints**
    ------------
    fc_df : pandas.DataFrame
        Forecast frame whose index can be coerced to datetime; one row per cashflow period.
    coe_t : float
        Cost of equity (nominal annual rate, in decimal, e.g., 0.10 for 10%).
    today_ts : pandas.Timestamp, optional
        Valuation date used as present time; defaults to module-level `TODAY_TS`.

    Returns
    -------
    discount : numpy.ndarray, shape (T,), dtype=float
        Discount factors for each period, aligned with `fc_df.index`.

    **Equations**
    -------------
    Let τ_t be the date for period t and Δ_t the year fraction to present:
        days_t = (τ_t − today_ts) / 1 day
        Δ_t    = days_t / 365
    Then the discount factor under flat coe is:
        disc_t = (1 + coe_t)^{−Δ_t}

    Hence:
        discount = [disc_0, disc_1, …, disc_{T−1}]^⊤
    """
        
    days = (fc_df.index.to_numpy() - np.datetime64(today_ts)) / np.timedelta64(1, 'D')
    
    discount = 1.0 / ((1 + coe_t) ** (days / 365.0))
    
    return discount


def _se(
    dcfe_years: np.ndarray,
    tv_disc: np.ndarray,
    shares_out: float,
    n: np.ndarray, 
    n_term: float,
 ) -> float:
    """
    Estimate a standard error (SE) for the DCFE price by combining
    period-by-period dispersion and terminal dispersion, scaled by analyst counts.

    **Type hints**
    ------------
    dcfe_years : numpy.ndarray, shape (P, T), dtype=float
        Discounted cash flows to equity per path and period. Row i is a path,
        column t is year t's present value.
    tv_disc : numpy.ndarray, shape (K, P), dtype=float
        Discounted terminal values: K terminal candidates stacked over P paths.
    shares_out : float
        Shares outstanding (scalar).
    n : numpy.ndarray, shape (T,), dtype=float
        Analyst counts per year; entries ≤ 0 or non-finite are treated as 1.
    n_term : float
        Analyst count for the final year (terminal scaling); ≤ 0 or non-finite → 1.

    Returns
    -------
    se : float
        Standard error per share.

    **Equations**
    -------------
    1) Per-year standard deviations (across paths):
        σ_t = std( dcfe_years[:, t], ddof=1 )

       Per-year standard errors with analyst scaling:
        SE_t = σ_t / sqrt( n_t )

    2) Terminal dispersion (across all terminal candidates and paths):
        σ_term = std( vec(tv_disc), ddof=1 )
        SE_term = σ_term / sqrt( n_term )

    3) Combine in quadrature and convert to per-share:
        SE_total = sqrt(  Σ_{t=0}^{T−1} SE_t^2  +  SE_term^2 )
        SE_price = SE_total / SharesOut

    **Notes**
    --------
    • Uses `np.nanstd(..., ddof=1)` to ignore NaNs in dispersion computation.
    • This SE is computed on *totals* and then divided by shares; if your reported
      statistic is the *clipped per-share* mean, consider computing SE directly on
      that distribution for alignment.
    """
    
    stds = np.nanstd(dcfe_years, axis = 0, ddof = 1)
    
    n = np.where(np.isfinite(n) & (n > 0), n, 1.0)
    
    ses = stds / np.sqrt(n)
    
    tv_flat = tv_disc.ravel()
    
    std_term = np.nanstd(tv_flat, ddof = 1)
    
    if not (np.isfinite(n_term) and n_term > 0):
        
        n_term = 1.0
            
    se_term = std_term / np.sqrt(n_term)

    se_total = np.sqrt((ses ** 2).sum() + se_term ** 2)
    
    se = se_total / shares_out
    
    return se
    
    
def _dcfe_vals(
    fc_df: pd.DataFrame,
    years: int, 
    shares_out: float,
    coe_t: float,
    exp_pe: float,
    pe_ind_dict: dict,
    pe_10y_ind_dict: dict,
    beta: np.ndarray,
    n: np.ndarray, 
    n_term: float,
) -> tuple[np.ndarray, float]:
    """
    Compute DCFE valuations across all enumerated paths and their standard error.

    **Type hints**
    ------------
    fc_df : pandas.DataFrame
        Cleaned forecast frame with revenue & EPS options per period.
    years : int
        Horizon length T (should equal len(fc_df)).
    shares_out : float
        Shares outstanding (scalar).
    coe_t : float
        Cost of equity (decimal).
    exp_pe : float
        Company expected PE multiple; if ≤0, a fallback is used in `terminal_value_pe`.
    pe_ind_dict : dict
        Industry PE dictionary for the ticker (keys like 'Industry-MC', 'Region-Industry').
    pe_10y_ind_dict : dict
        10-year industry PE dictionary (key 'Region-Industry').
    beta : numpy.ndarray, shape (3,), dtype=float
        Regression coefficients [β0, β1, β2] for:
            FCFE_t ≈ β0 + β1·Revenue_t + β2·EPS_t
    n : numpy.ndarray, shape (T,), dtype=float
        Analyst counts per year (used in SE scaling).
    n_term : float
        Analyst count for the terminal year.

    Returns
    -------
    dcfe_vals : numpy.ndarray, shape (K, P), dtype=float
        Present value per path including terminal for each candidate multiple:
            dcfe_vals = 1·Σ_t (FCFE_t × disc_t) + TV_disc
        stacked over K terminal candidates and P paths.
    se : float
        Standard error per share from `_se`.

    **Equations & pipeline**
    ------------------------
    1) Path construction:
        (rev_matrix, eps_matrix) ∈ ℝ^{P×T} with P = 3^T paths.

    2) Free cash flow to equity per path & period (linear proxy):
        FCFE_{i,t} = β0 + β1·Revenue_{i,t} + β2·EPS_{i,t}

    3) Discount factors (irregular calendar):
        disc_t = (1 + coe_t)^{ − ( (τ_t − today)/365 ) }

    4) Present values per period:
        DCF_{i,t} = FCFE_{i,t} × disc_t

    5) Horizon sum per path:
        SumDCF_i = Σ_{t=0}^{T−1} DCF_{i,t}

    6) Terminal (PE-based) per candidate j:
        TV_disc[j, i] = (PE_j × EPS_{i,T−1} × SharesOut) × disc_T

    7) Total present value per candidate & path:
        V_{j,i} = SumDCF_i + TV_disc[j,i]

       In array terms:
        dcfe_vals = SumDCF[None, :] + TV_disc

    8) Standard error:
        se = _se(DCF, TV_disc, SharesOut, n, n_term)

    **Shapes**
    ----------
    • T = years, P = 3^T, K ∈ {1,2,3}
    • rev_matrix, eps_matrix : (P, T)
    • discount : (T,)
    • dcf_years : (P, T)
    • tv_disc : (K, P)
    • dcfe_vals : (K, P)
    """
   
    core_cols = ['low_rev', 'avg_rev', 'high_rev', 'low_eps', 'avg_eps', 'high_eps']
    
    valid = fc_df.loc[fc_df[core_cols].notna().all(axis = 1)].copy()

    if valid.empty:

        return np.array([[0.0]]), 0.0

    years = len(valid)
    
    rev_matrix, eps_matrix = build_rev_eps_matrices(
        fc_df = valid,
        years = years
    )
    
    fcfes = beta[0] + (rev_matrix * beta[1]) + (eps_matrix * beta[2])
    
    discount = build_discount_factor_vector(
        fc_df = valid,
        coe_t = coe_t,
        today_ts = TODAY_TS,
    )
   
    dcf_years = fcfes * discount[np.newaxis, :]  

    sum_dcf = dcf_years.sum(axis = 1)
    
    final_eps = eps_matrix[:, -1]
    
    disc_T = float(discount[-1])
    
    tv_disc = terminal_value_pe(
        final_eps = final_eps,
        shares_out = shares_out,
        exp_pe = exp_pe, 
        pe_ind_dict = pe_ind_dict,
        pe_10y_ind_dict = pe_10y_ind_dict,
        disc_T = disc_T
    )
    
    dcfe_vals = sum_dcf[None, :] + tv_disc
    
    n = (
        valid['num_analysts']
        .astype(float)
        .bfill()
        .fillna(1.0)
        .clip(lower = 1.0)
        .to_numpy()
    )
    n_term = float(n[-1])
    
    se = _se(
        dcfe_years = dcf_years,
        tv_disc = tv_disc,
        shares_out = shares_out,
        n = n,
        n_term = n_term
    )    
    
    return dcfe_vals, se


def main():
    """
    Run the DCFE valuation pipeline across all tickers and export results.

    Steps
    -----
    1) Load inputs (tickers, last prices, shares outstanding, COE, industry PE dictionaries).
    2) For each ticker:
       • Validate inputs; clean forecast frame via `_clean_fc_df`.
       • Build regression design with `regression_data_prep` and fit coefficients with
         `HuberENetCV` (time-series CV).
       • Build all revenue/EPS paths; compute FCFE via linear proxy β0 + β1·Revenue + β2·EPS.
       • Construct discount factors and horizon PVs; compute PE terminals via `terminal_value_pe`.
       • Aggregate to present values `dcfe_vals`, compute per-share prices (clipped to bounds).
       • Compute SE via `_dcfe_vals`'s call to `_se`.
       • Log and accumulate Low/Avg/High/SE.
    3) Export results to an Excel sheet named 'DCFE' via `export_results`.

    Outputs
    -------
    • DataFrame with columns ['Low Price','Avg Price','High Price','SE'] indexed by ticker.
    • Written to the Excel file designated by `config.MODEL_FILE`.
    """
    
    tickers = config.tickers 

    latest_prices = r.last_price

    shares_out = r.shares_outstanding

    lbp = config.lbp * latest_prices

    ubp = config.ubp * latest_prices

    coe = pd.read_excel(config.FORECAST_FILE, sheet_name = 'COE', index_col = 0, usecols = ['Ticker', 'COE'], engine = 'openpyxl')

    dicts = r.dicts()

    pe_ind_dict = dicts['PE']
    
    pe_10y_ind_dict = dicts['PE10y']

    low_price = {}

    avg_price = {}

    high_price = {}

    se_dict = {}

    alphas = np.linspace(0.3, 0.7, 5)

    lambdas = np.logspace(0, -4, 20)  

    huber_M_values = (0.25, 1.0, 4.0)

    cv_folds = 3

    tscv = TimeSeriesSplit(n_splits = cv_folds)

    cv = HuberENetCV(
        alphas = alphas,
        lambdas = lambdas,
        Ms = huber_M_values,
        n_splits = cv_folds,
        n_jobs = -1,               
    )

    fin_dict = fdata.annuals

    fc_dict = fdata.forecast

    kpis_dict = fdata.kpis

    constrained_map = {
        'y': True
    } 

    for ticker in tickers:
        
        shares_t = shares_out[ticker]
        
        if not (np.isfinite(shares_t) and shares_t > 0):
            
            logger.warning(f"{ticker}: invalid shares_out={shares_t}, skipping")
        
        fin_df = fin_dict[ticker]

        fc_df = fc_dict[ticker]

        kpis = kpis_dict[ticker]
        
        if fin_df is None or fc_df is None or kpis is None:

            logger.info(f"Skipping {ticker}: missing data")

            low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = 0

            continue

        fc_df = _clean_fc_df(
            fc_df = fc_df
        )
    
        if fc_df.empty:
        
            logger.warning(f"{ticker}: no valid forecast data after dropping all‐NaN rows, skipping")
        
            low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = np.nan
        
            continue
        
            
        if not set(REQUIRED_COLUMNS).issubset(fin_df.columns):
        
            logger.warning(f"Missing required columns in fin_df for {ticker}, skipping ticker")
        
            low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = 0
        
            continue
        
        X, y = regression_data_prep(
            fin_df = fin_df
        )

        y_dict = {
            'y': y
        }

        if len(y) < 2:
    
            logger.warning(f"Not enough regression data for {ticker}")
        
            low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = 0
        
            continue
        
        if not RUN_REGRESSION:
            
            cached = bc.get_betas(
                path = BETA_CACHE_FILE,
                model_key = BETA_CACHE_MODEL, 
                ticker = ticker
            )
            
            if cached is None or "y" not in cached:
                
                logger.warning("%s: missing cached beta. Set RUN_REGRESSION=True once to populate cache. Skipping.", ticker)
                
                low_price[ticker] = avg_price[ticker] = high_price[ticker] = se_dict[ticker] = 0
                
                continue

            beta = np.asarray(cached["y"], dtype=float)

        else:
            
            cv_splits = list(tscv.split(X))

            betas_by_key, best_lambda, best_alpha, best_M = cv.fit_joint(
                X = X,
                y_dict = y_dict,
                constrained_map = constrained_map,
                cv_splits = cv_splits,
                scorer = None,
            )

            beta = betas_by_key["y"]

            bc.upsert_betas(
                path = BETA_CACHE_FILE,
                model_key = BETA_CACHE_MODEL,
                ticker = ticker,
                betas_by_key = {
                    "y": beta
                },
                meta = {
                    "best_lambda": float(best_lambda),
                    "best_alpha": float(best_alpha),
                    "best_M": float(best_M),
                },
            )
        
        coe_t = coe.loc[ticker].iat[0]
            
        years = len(fc_df)
        
        exp_pe = kpis['exp_pe'].iat[0] 
        
        n = fc_df["num_analysts"].iloc[:years].astype(float).values
        
        n[n == 0] = 1.0 
        
        n_term = float(fc_df["num_analysts"].iat[-1])
        
        if n_term < 1:
        
            n_term = 1.0
                    
        dcfe_vals, se = _dcfe_vals(
            fc_df = fc_df,
            years = years,
            shares_out = shares_t,
            coe_t = coe_t,
            exp_pe = exp_pe,
            pe_ind_dict = pe_ind_dict[ticker],
            pe_10y_ind_dict = pe_10y_ind_dict[ticker],
            beta = beta,
            n = n, 
            n_term = n_term
        )
        
        prices = np.clip(dcfe_vals / shares_t, lbp[ticker], ubp[ticker])
        
        flat_prices = prices.flatten()

        low_price[ticker] = max(np.nanmin(flat_prices), 0)
    
        avg_price[ticker] = max(np.nanmean(flat_prices), 0)
    
        high_price[ticker] = max(np.nanmax(flat_prices), 0)    
    
        se_dict[ticker] = se

        logger.info(f"{ticker}: Low {low_price[ticker]}, Avg {avg_price[ticker]}, High {high_price[ticker]}, SE {se_dict[ticker]}")


    dcfe_df = pd.DataFrame({
        'Low Price': low_price,
        'Avg Price': avg_price,
        'High Price': high_price,
        'SE': se_dict
    })

    dcfe_df.index.name = 'Ticker'
    
    sheet = {
        "DCFE": dcfe_df
    }
    
    export_results(
        sheets = sheet,
        output_excel_file = config.MODEL_FILE
    )
         
        
if __name__ == "__main__":

    main()
