"""
Implements a residual income valuation model combining book value growth, cost of equity and analyst estimates.
"""

import numpy as np
import pandas as pd
import numpy.typing as npt
import logging

from data_processing.financial_forecast_data import FinancialForecastData
import config


df_opts = {'future.no_silent_downcasting': True}

for opt, val in df_opts.items():
    
    pd.set_option(opt, val)


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


TODAY_TS = pd.Timestamp.today().normalize()

MIN_DENOM = 5e-3


fdata = FinancialForecastData()

macro = fdata.macro

r = macro.r


def bvps(
    eps: float, 
    prev_bvps: float, 
    dps: float
) -> float:
    """
    Compute next-period Book Value Per Share (BVPS) given earnings and dividends.

    Mathematically, if BVPS evolves additively as retained earnings are reinvested,
    the one-step update is:
        BVPS_{t+1} = BVPS_t + EPS_{t+1} - DPS_{t+1}

    Parameters
    ----------
    eps : float
        Forecast earnings per share for the next period, EPS_{t+1}.
    prev_bvps : float
        Current (previous) book value per share, BVPS_t.
    dps : float
        Forecast dividends per share for the next period, DPS_{t+1}.

    Returns
    -------
    float
        Next-period book value per share, BVPS_{t+1}.

    Notes
    -----
    • This is the standard clean-surplus relation in per-share terms.
    • Inputs are cast to float to avoid dtype surprises.
    • No clipping or bounds are imposed here; caller should enforce realism if needed.
    """
    
    bvps_t1 = float(prev_bvps) + float(eps) - float(dps)
    
    return bvps_t1


def growth(
    kpis, 
    ind_g
):
    """
    Compute the company-specific growth rate used for terminal residual-income logic.

    The function returns a growth proxy g:
      • If ROE < 0 (loss-making), fall back to the industry/region growth `ind_g`.
      • Otherwise, use retention-based growth:
            g = ROE × (1 - payout_ratio)
        where payout_ratio defaults to 0 if missing.

    Parameters
    ----------
    kpis : pandas.DataFrame
        Must contain columns 'roe' and 'payout_ratio' (first row is used).
        'roe' is Return on Equity; 'payout_ratio' in [0, 1] ideally.
    ind_g : float
        Region-Industry growth fallback used when ROE < 0 or as an alternative
        terminal denominator.

    Returns
    -------
    float
        Growth rate g to use in:
          • terminal EPS_{T+1} = EPS_T × (1 + g)
          • terminal denominator (coe - g) when RI terminal is active.

    Notes
    -----
    • When payout_ratio is NaN, it is set to 0 (i.e., full retention).
    • This is a simplified fundamental identity that ties growth to reinvested
      earnings via ROE.
    """
    
    roe = kpis['roe'].iat[0]
    
    if roe < 0:
    
        return ind_g  
    
    else:
        
        payout_ratio = kpis['payout_ratio'].iat[0] 
        
        if pd.isna(payout_ratio):
            
            payout_ratio = 0
                    
        return roe * (1 - payout_ratio)


def calc_div_growth(
    div: pd.Series
) -> float:
    """
    Estimate long-run dividend growth rate from a dividend history.

    Steps:
    1) Coerce to float, back/forward fill for small gaps, keep strictly positive values.
    2) Compute period-over-period growth rates:
           r_t = (Div_t / Div_{t-1}) - 1
       (implemented via pct_change()).
    3) Compute the geometric mean growth via logs:
           g = exp( mean( log(1 + r_t) ) ) - 1
    4) Clip to a reasonable range to avoid absurd tails:
           g ∈ [-0.8, 5.0]

    Parameters
    ----------
    div : pandas.Series
        Historical dividends (level, not growth). Should be indexed in time order.

    Returns
    -------
    float
        Geometric-average dividend growth estimate.

    Notes
    -----
    • If there are no valid growth observations, returns 0.0.
    • Clipping is intentionally wide to preserve signal but block extreme outliers.
    • The geometric mean is more appropriate than an arithmetic mean for chained
      growth processes.
    """
    
    div = div.astype(float)
    
    div = div.bfill().ffill()
    
    div = div.where(div > 0)
    
    pct = div.pct_change().dropna()
    
    if pct.empty:
        
        return 0.0
    
    g = np.exp(np.nanmean(np.log1p(pct).mean())) - 1.0
    
    return float(np.clip(g, -0.8, 5))


def build_dps_vector(
    dps: float,
    div_growth: float,
    years: int
) -> npt.NDArray[np.float64]: 
    """
    Build a vector of forecast dividends per share (DPS) across a multi-year horizon.

    Assumes a constant growth model for dividends:
        DPS_t = DPS_0 × (1 + g_div)^t,    for t = 0, 1, ..., years-1

    Parameters
    ----------
    dps : float
        Baseline (current) dividend per share, DPS_0.
    div_growth : float
        Constant dividend growth rate g_div applied each period.
    years : int
        Number of forecast periods (T).

    Returns
    -------
    numpy.ndarray of shape (T,)
        DPS forecasts [DPS_0, DPS_1, ..., DPS_{T-1}].

    Notes
    -----
    • This simplifies dividend dynamics
    """
    
    vec = dps * (1 + div_growth) ** np.arange(years)     
    
    return vec


def build_eps_grid(
    valid: pd.DataFrame
) -> np.ndarray:
    """
    Enumerate all EPS path combinations over the forecast horizon using
    low/avg/high per-period options.

    Let T = number of valid forecast periods (rows in `valid`).
    For each period t, let {L_t, A_t, H_t} be (low_eps, avg_eps, high_eps).
    The function constructs all 3^T paths:
        ε = [ε_0, ε_1, ..., ε_{T-1}]  with ε_t ∈ {L_t, A_t, H_t}

    Implementation uses `np.meshgrid` to form the Cartesian product, then reshapes
    to an array of shape:
        (3^T, T)

    Parameters
    ----------
    valid : pandas.DataFrame
        Must contain columns: ['low_eps', 'avg_eps', 'high_eps'], indexed by forecast dates.

    Returns
    -------
    numpy.ndarray
        EPS path grid of shape (combos, T) where combos = 3**T.

    Notes
    -----
    • The paths are unweighted here. If you wish to weight L/A/H differently
      (e.g., by forecast probabilities), handle that downstream.
    """
        
    eps_cols = ['low_eps', 'avg_eps', 'high_eps']
    
    opts = [valid[c].to_numpy(dtype=float) for c in eps_cols]
    
    m = np.meshgrid(*opts, indexing = 'xy') 
    
    eps_grid = np.stack(m, axis = -1).reshape(-1, len(valid))
    
    return eps_grid


def build_discount_factor_vector(
    coe_t: float,
    valid: pd.DataFrame,
    today_ts = TODAY_TS
) -> np.ndarray:
    """
    Construct per-period discount factors from a (possibly irregular) forecast index.

    For each forecast date τ_t (taken from `valid.index`), compute the time-to-cash in days:
        days_t = (τ_t - today_ts)  in days
    Then the discount factor under a flat cost of equity `coe_t` is:
        disc_t = (1 + coe_t)^{ - days_t / 365 }

    Parameters
    ----------
    coe_t : float
        Cost of equity (nominal annual rate, decimal), assumed constant across the horizon.
    valid : pandas.DataFrame
        Forecast frame whose index is (or can be coerced to) datetime64; one row per period.
    today_ts : numpy.datetime64 or pandas.Timestamp, optional
        Valuation date used as the present time. Defaults to module-level TODAY_TS.

    Returns
    -------
    numpy.ndarray of shape (T,)
        Discount factors [disc_0, disc_1, ..., disc_{T-1}] matching `valid` rows.

    Notes
    -----
    • If the forecast index is not a regular yearly grid, the exponent uses fractional years.
    • Assumes simple annual compounding with 365-day convention.
    • Ensure `valid.index` is timezone-naive or consistently localized prior to calling.
    """

    days = (valid.index.to_numpy() - np.datetime64(today_ts)) / np.timedelta64(1,'D')
    
    disc1d = np.power(1 + coe_t, - days / 365.0)
    
    return disc1d
    

def build_prev_bvps_vector(
    bvps_0: np.ndarray,
    eps_grid: np.ndarray,
    dps_vec: npt.NDArray[np.float64],
    combos: int
) -> np.ndarray:
    """
    Build the matrix of 'previous' BVPS along each EPS path, used in residual income terms.

    For path i and period t≥1, the 'previous' book value is:
        BVPS_{t}^{prev}(i) = BVPS_0 + sum_{k=0}^{t-1} ( EPS_k(i) - DPS_k )
    and BVPS_{0}^{prev}(i) = BVPS_0.

    This function returns a matrix BVPS_prev of shape (combos, T) where:
      • BVPS_prev[i, 0]      = BVPS_0
      • BVPS_prev[i, t>0]    = BVPS_0 + cumulative sum_{k=0}^{t-1}(EPS_k(i) - DPS_k)

    Parameters
    ----------
    bvps_0 : float
        Starting book value per share at valuation time (t=0).
    eps_grid : numpy.ndarray, shape (combos, T)
        All EPS paths (from `build_eps_grid`).
    dps_vec : numpy.ndarray, shape (T,)
        Dividend-per-share forecasts for each period.
    combos : int
        Number of EPS paths (usually 3**T).

    Returns
    -------
    numpy.ndarray, shape (combos, T)
        Matrix of previous-period BVPS along each path.

    Notes
    -----
    • The construction uses cumulative sums of (EPS - DPS), shifted by one period.
    • This is consistent with the clean-surplus relation in per-share terms.
    """
        
    delta = eps_grid - dps_vec[None, :]
   
    cs = np.cumsum(delta, axis = 1)
   
    BVPS_prev = np.hstack([
        np.full((combos, 1), bvps_0),
        bvps_0 + cs[:, :-1]
    ])
    
    return BVPS_prev


def notna(
    x
): 
    """
    Convenience predicate for 'is not NA' that mirrors pandas semantics.

    Parameters
    ----------
    x : Any
        Value to test for missingness (supports NaN, None, pandas.NA).

    Returns
    -------
    bool
        True if `x` is not a pandas-style NA; False otherwise.
    """    
    return not pd.isna(x)

    
def build_tv_mat(
    coe_t,
    g_comp,
    ind_g_ri,
    ind_pb,
    price_book,
    term_raw,
    last_bvps,
    na_m1,
    disc1d,
    min_threshold = MIN_DENOM,
):
    """
    Assemble terminal value candidates for residual-income valuation and estimate
    their contribution to standard error (SE) given analyst counts.

    Terminal candidates (each yields a vector over EPS paths of length `combos`):

    1) Company RI terminal (if denominator is sufficiently positive):
           D_comp = coe_t - g_comp
           TV_comp = (term_raw / D_comp) × disc_T
       with
           term_raw = EPS_{T+1} - coe_t × BVPS_{T-1}^{prev}
           EPS_{T+1} = EPS_T × (1 + g_comp)
           disc_T = disc1d[-1]
       Included only if D_comp > min_threshold.

    2) Industry RI terminal (if denominator is sufficiently positive):
           D_ind = coe_t - ind_g_ri
           TV_ind = (term_raw / D_ind) × disc_T
       Included only if D_ind > min_threshold.

    3) Company P/B terminal (always included if price_book is finite):
           TV_PB = (price_book × BVPS_{T-1}^{prev}) × disc_T

    4) Industry P/B terminal (always included if ind_pb is finite):
           TV_PB_ind = (ind_pb × BVPS_{T-1}^{prev}) × disc_T

    The function stacks whichever candidates are available into a matrix:
        term_disc_mat ∈ ℝ^{combos × n_terms}
    where n_terms ∈ {0,1,2,3,4} depending on gating/availability.

    For the terminal block SE, it computes:
        SE_term = sd( vec(term_disc_mat) ) / sqrt(na_last_eff * n_terms)
    where:
        • vec(·) flattens the matrix,
        • na_last_eff = max(na_m1, 1.0) is the analyst count floor on the final period,
        • n_terms is the number of active terminal methods.

    Parameters
    ----------
    coe_t : float
        Cost of equity (annual rate, decimal).
    g_comp : float
        Company growth g used in EPS_{T+1} and (coe_t - g_comp) denominator.
    ind_g_ri : float or NaN
        Region-industry growth used in alternative RI denominator (coe_t - ind_g_ri).
    ind_pb : float or NaN
        Region-industry price-to-book multiple (per-share terminal alternative).
    price_book : float or NaN
        Company expected price-to-book multiple (per-share terminal alternative).
    term_raw : numpy.ndarray, shape (combos,)
        Vector of (EPS_{T+1} - coe_t × BVPS_{T-1}^{prev}) across paths.
    last_bvps : numpy.ndarray, shape (combos,)
        Vector of BVPS_{T-1}^{prev} across paths.
    na_m1 : float
        Analyst count for the last forecast period.
    disc1d : numpy.ndarray, shape (T,)
        Discount factor vector; only disc1d[-1] is used for terminal discounting.
    min_threshold : float
        Minimal positive denominator required to include an RI terminal (default MIN_DENOM).

    Returns
    -------
    term_disc_mat : numpy.ndarray, shape (combos, n_terms) or (combos, 1) if none
        Discounted terminal value candidates stacked column-wise. If no candidates
        are available, returns a zero column to preserve broadcast shape.
    se_term : float
        Terminal-block standard error contribution, computed from the flattened
        candidate matrix and scaled by analyst counts.

    Notes
    -----
    • The RI terminal follows a standard continuing-value expression for residual income:
          TV_RI = ( RI_{T+1} ) / (coe - g ) × disc_T
      where RI_{T+1} = EPS_{T+1} - coe × BVPS_{T}^{book}.
    • The P/B terminals  capitalise BVPS_{T}^{book} with a multiple and discount it.
    • This implementation divides SE_term by sqrt(n_terms) to reflect combining multiple
      candidate terminals.
    """
       
    term_candidates = []

    if notna(
        x = coe_t
    ):
    
        if notna(
            x = g_comp
        ):
           
            denom_comp = float(coe_t) - float(g_comp)
           
            if denom_comp is not None and denom_comp > min_threshold:
           
                term_candidates.append(term_raw / denom_comp)

        if notna(
            x = ind_g_ri
        ):
        
            denom_ind = float(coe_t) - float(ind_g_ri)
        
            if denom_ind > min_threshold and denom_ind is not None:
     
                term_candidates.append(term_raw / denom_ind)

    if notna(
        x = price_book
    ):
   
        term_candidates.append(float(price_book) * last_bvps)

    if notna(
        x = ind_pb
    ):
      
        term_candidates.append(float(ind_pb) * last_bvps)

    combos = last_bvps.shape[0]
  
    na_last_eff = max(float(na_m1), 1.0)

    if term_candidates:
       
        term_vals = np.column_stack(term_candidates)            
       
        term_disc_mat = term_vals * disc1d[-1]              

        se_term = np.nanstd(term_disc_mat.ravel(), ddof = 1) / np.sqrt(na_last_eff * len(term_candidates))
     
        return term_disc_mat, se_term
    
    else:

        term_disc_mat = np.zeros((combos, 1), dtype = float)

        se_term = 0.0

        return term_disc_mat, se_term


def total_ri_preds(
    kpis,
    fin_df,
    forecast_df,
    coe_t,
    ind_g,
    ind_g_ri,
    ind_pb,
    shares_out
):
    """
    Compute the distribution of present values from residual-income (RI) valuation
    across all EPS paths and estimate overall SE accounting for per-period analyst counts.

    Pipeline & Equations
    --------------------
    1) Inputs & coercions:
       • Extract BVPS_0, price_book (P/B), cast coe_t to float if provided.
       • Build `valid` forecast frame with columns:
           {low_eps, avg_eps, high_eps, num_analysts}
         ensuring datetime index and an analysts floor of 1.0.

    2) Dividend model:
       • Estimate long-run dividend growth g_div = calc_div_growth(div_series).
       • Baseline dividend per share DPS_0:
            DPS_0 = mean(Div) / Shares_Out
         (or 0 if shares_out is missing/nonpositive).
       • Build DPS vector for T periods:
            DPS_t = DPS_0 × (1 + g_div)^t

    3) EPS path enumeration:
       • For each year t with options {L_t, A_t, H_t}, build all paths:
            ε ∈ ℝ^{3^T × T}, ε_{i,t} ∈ {L_t, A_t, H_t}

    4) Book value propagation along each path:
       • Previous BVPS before recognising EPS_t:
            BVPS_{t}^{prev}(i) = BVPS_0 + ∑_{k=0}^{t-1} ( ε_{i,k} - DPS_k )
         Construct BVPS_prev ∈ ℝ^{3^T × T}.

    5) Discount factors:
       • For forecast date τ_t, days_t = (τ_t - today) in days, then:
            disc_t = (1 + coe_t)^{ - days_t / 365 }
         Collect disc ∈ ℝ^{T}.

    6) Residual income (per period, per path):
       • RI_{i,t} = ( ε_{i,t} - coe_t × BVPS_{i,t}^{prev} ) × disc_t
       • Sum over horizon:
            RI_sum_i = ∑_{t=0}^{T-1} RI_{i,t}

    7) Terminal value candidates (per path):
       • Let g = growth(kpis, ind_g) and EPS_{T+1}(i) = ε_{i,T-1} × (1 + g).
       • term_raw_i = EPS_{T+1}(i) - coe_t × BVPS_{i,T-1}^{prev}.
       • Call `build_tv_mat` to assemble discounted terminal candidates:
            term_disc_mat ∈ ℝ^{3^T × n_terms}, se_term ∈ ℝ
         (RI terminals gated on denominators coe_t - g and coe_t - ind_g_ri,
          plus P/B terminals when available).

    8) Total present value distribution:
       • base_ri_i = BVPS_0 + RI_sum_i
       • total_RI_mat = base_ri[:, None] + term_disc_mat
       • Flatten to a 1-D vector, drop NaNs:
            total_RI = vec(total_RI_mat) ∈ ℝ^{3^T × n_terms}

    9) Standard error (SE):
       • Period-by-period uncertainty from RI terms:
            σ_t = std( RI_{:,t}, ddof=1 )
            SE_t = σ_t / sqrt( n_t ), with n_t = max(num_analysts_t, 1)
       • Combine by quadrature and include terminal-block contribution:
            SE = sqrt( ∑_{t=0}^{T-1} SE_t^2  +  se_term^2 )

    Parameters
    ----------
    kpis : pandas.DataFrame
        Requires columns ['bvps_0', 'exp_ptb', 'roe', 'payout_ratio']; first row used.
    fin_df : pandas.DataFrame
        Financial history with 'Div' column (dividends level). Index chronological.
    forecast_df : pandas.DataFrame
        Forecasts with columns ['low_eps', 'avg_eps', 'high_eps', 'num_analysts'].
        Index must be (or will be coerced to) datetime for discounting.
    coe_t : float
        Cost of equity (annual rate, decimal).
    ind_g : float
        Region-Industry growth used by `growth(kpis, ind_g)` and as an alternative RI denominator.
    ind_g_ri : float or NaN
        Region-Industry growth used specifically for the alternative RI terminal (coe_t - ind_g_ri).
    ind_pb : float or NaN
        Region-Industry price-to-book multiple used as a P/B terminal.
    shares_out : float
        Shares outstanding used to convert total dividends to per-share DPS_0. If missing or
        nonpositive, DPS_0 is set to 0.

    Returns
    -------
    total_RI : pandas.Series
        Flattened distribution of present values across all EPS paths and terminal candidates.
    se : float
        Overall standard error estimate combining per-year (analyst-weighted) RI uncertainty and
        terminal-block uncertainty from `build_tv_mat`.

    Shapes
    ------
    • T = number of forecast periods after filtering ('valid' rows).
    • combos = 3**T = number of EPS paths.
    • eps_grid : (combos, T), BVPS_prev : (combos, T), disc : (T,)
    • term_disc_mat : (combos, n_terms), total_RI_mat : (combos, n_terms)

    Notes
    -----
    • The RI framework values equity via:
          V_0 = BVPS_0 + Σ_{t=0}^{T-1} [ (EPS_t - coe × BVPS_{t}^{prev}) × disc_t ] + TV
      with different specifications for the continuing value TV.
    • This implementation enumerates all low/avg/high EPS paths.
    • The SE composition assumes independence across years for the by-year term; if serial
      correlation is material, consider a delta method with a full covariance or bootstrap.
    """
       
    bvps_0 = kpis['bvps_0'].iat[0]
   
    price_book = kpis['exp_ptb'].iat[0]
    
    if not pd.isna(price_book):
        
        price_book = float(price_book)
    
    if not pd.isna(coe_t):

        coe_t = float(coe_t)

    fdf = forecast_df.copy()
    
    fdf.index = pd.to_datetime(fdf.index, errors = 'coerce')

    core = fdf[['low_eps', 'avg_eps', 'high_eps']]
   
    valid = fdf[core.notna().all(axis = 1)].copy()
   
    valid['num_analysts'] = (
        fdf['num_analysts']
        .reindex(valid.index)
        .bfill()
        .fillna(1.0)            
        .clip(lower = 1.0)
    )   
    
    if valid.empty:

        total_RI = pd.Series([float(kpis['bvps_0'].iat[0])])

        se = 0.0

        return total_RI, se

    years = valid.shape[0]
   
    na = valid['num_analysts'].to_numpy(dtype = float)
   
    na[-1] = max(na[-1], 1.0)  

    g = growth(
        kpis = kpis, 
        ind_g = ind_g
    )

    div = fin_df['Div'].abs().fillna(0)
    
    if shares_out:
   
        dps = float(div.mean()) / float(shares_out)
    
    else:
        
        dps = 0.0
   
    div_growth = calc_div_growth(
        div = div
    )

    eps_grid = build_eps_grid(
        valid = valid
    )
    
    combos = eps_grid.shape[0]

    dps_vec = build_dps_vector(
        dps = dps, 
        div_growth = div_growth,
        years = years
    )

    BVPS_prev = build_prev_bvps_vector(
        bvps_0 = bvps_0, 
        eps_grid = eps_grid, 
        dps_vec = dps_vec,
        combos = combos
    )

    disc1d = build_discount_factor_vector(
        coe_t = coe_t, 
        valid = valid
    )

    ri_terms = (eps_grid - coe_t * BVPS_prev) * disc1d[None, :]
   
    ri_sum = ri_terms.sum(axis = 1)

    last_eps = eps_grid[:, -1]
   
    last_bvps = BVPS_prev[:, -1]
   
    eps_tp1 = last_eps * (1 + g)
   
    term_raw = eps_tp1 - coe_t * last_bvps

    term_disc_mat, se_term = build_tv_mat(
        coe_t = coe_t,
        g_comp = g,
        ind_g_ri = ind_g_ri,
        ind_pb = ind_pb,
        price_book = price_book,
        term_raw = term_raw,
        last_bvps = last_bvps,
        na_m1 = na[-1],
        disc1d = disc1d
    )

    base_ri = bvps_0 + ri_sum
   
    total_RI_mat = base_ri[:, None] + term_disc_mat
   
    total_RI = pd.Series(total_RI_mat.ravel()).dropna()

    stds_by_year = np.nanstd(ri_terms, axis = 0, ddof = 1)
   
    ses_by_year = stds_by_year / np.sqrt(na)
   
    se = np.sqrt((ses_by_year ** 2).sum() + se_term ** 2)

    return total_RI, se


def main():
    """
    Orchestrate the residual-income valuation across tickers and write outputs.

    Steps
    -----
    1) Load inputs:
       • Ticker list, latest prices, shares outstanding from `macro.r`.
       • Cost of equity series (COE) from Excel (sheet 'COE').
       • Lower/upper price clamps (lb/ub) from config × latest prices.
       • Dictionaries for Region-Industry growth (`eps1y_5`) and P/B (`PB`).

    2) For each ticker:
       • Validate presence of financials, forecasts, KPIs, COE.
       • Extract Region-Industry growth and P/B where available.
       • Call `total_ri_preds` to obtain the present-value distribution and SE.
       • Clip prices to [lb, ub], then compute:
            low  = max(min(price_dist), 0)
            avg  = max(mean(price_dist), 0)
            high = max(max(price_dist), 0)
            return = max(avg / last_price - 1, 0)
       • Log a one-line summary.

    3) Export:
       • Assemble a DataFrame with Low/Avg/High/SE per ticker.
       • Write to Excel (sheet 'RI'), replacing if it exists.

    Outputs
    -------
    • Excel sheet 'RI' with columns:
         ['Low Price', 'Avg Price', 'High Price', 'SE']
      indexed by ticker.

    Notes
    -----
    • Missing inputs for a ticker trigger a skip with zeros recorded.
    • Price clipping prevents extreme terminals from dominating the reported stats.
    • Logging level is INFO; switch to DEBUG upstream for more granular diagnostics.
    """
        
    tickers = config.tickers
    
    latest_prices = r.last_price
    
    shares_outstanding = r.shares_outstanding

    coe = pd.read_excel(
        config.FORECAST_FILE, 
        sheet_name = 'COE',
        index_col = 0, 
        usecols = ['Ticker', 'COE'], 
        engine = 'openpyxl'
    )

    lb = config.lbp * latest_prices
    
    ub = config.ubp * latest_prices

    results = r.dicts()
    
    growth_dict = results['eps1y_5']
    
    pb_dict = results['PB']

    low_price = {}
    
    avg_price = {}
    
    high_price = {}
    
    returns_dict = {}
    
    se_dict = {}

    for ticker in tickers:
       
        fin_df = fdata.annuals.get(ticker)
       
        forecast_df = fdata.forecast.get(ticker)
       
        kpis = fdata.kpis.get(ticker)

        if (fin_df is None or fin_df.empty or forecast_df is None or kpis is None or ticker not in coe.index):
            
            logger.info("Skipping %s: missing inputs.", ticker)
            
            low_price[ticker] = avg_price[ticker] = high_price[ticker] = returns_dict[ticker] = se_dict[ticker] = 0.0
            
            continue

        gi = growth_dict.get(ticker, {})

        pi = pb_dict.get(ticker, {})

        ind_g = gi.get('Region-Industry', np.nan)
        
        if not pd.isna(ind_g):

            ind_g_ri = float(ind_g)  
        
        else:
            
            ind_g_ri = np.nan

        ind_pb_val = pi.get('Region-Industry', np.nan)
        
        if not pd.isna(ind_pb_val):
            
            ind_pb = float(ind_pb_val)  
        
        else:
            
            ind_pb = np.nan

        coe_t = coe.loc[ticker, 'COE']
        
        shares_out = shares_outstanding.get(ticker, np.nan)
        
        total_RI, se = total_ri_preds(
            kpis = kpis,
            fin_df = fin_df,
            forecast_df = forecast_df,
            coe_t = coe_t,
            ind_g = ind_g,
            ind_g_ri = ind_g_ri,
            ind_pb = ind_pb,
            shares_out = shares_out,
        )

        prices_all = np.clip(total_RI.to_numpy(), lb[ticker], ub[ticker])

        lp = float(latest_prices[ticker])
       
        low_price[ticker] = max(prices_all.min(), 0.0)
       
        avg_price[ticker] = max(prices_all.mean(), 0.0)
       
        high_price[ticker] = max(prices_all.max(), 0.0)
       
        returns_dict[ticker] = max((avg_price[ticker] / lp) - 1.0, 0.0)
       
        se_dict[ticker] = float(se)

        logger.info(
            f"{ticker}: Low {low_price[ticker]:.2f}, Avg {avg_price[ticker]:.2f}, High {high_price[ticker]:.2f}, SE {se_dict[ticker]:.4f}"
        )

    df_ri = pd.DataFrame({
        'Low Price': low_price,
        'Avg Price': avg_price,
        'High Price': high_price,
        'SE': se_dict
    })
   
    df_ri.index.name = 'Ticker'

    with pd.ExcelWriter(config.MODEL_FILE, mode = 'a', engine = 'openpyxl', if_sheet_exists = 'replace') as writer:
       
        df_ri.to_excel(writer, sheet_name='RI')

    
if __name__ == "__main__":
    
    main()
