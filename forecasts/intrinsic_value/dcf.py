"""

Deterministic-scenario discounted cash-flow (DCF) with multiple FCFF constructions
and terminal value by valuation multiples.

Overview
--------

For each ticker, historical financials are regressed jointly on (Revenue, EPS) to
obtain linear predictors for operating lines. Using analysts' low/average/high paths
for (Revenue, EPS), a Cartesian grid of scenarios is generated. For each scenario,
several Free Cash Flow to Firm (FCFF) constructions are computed under structural
constraints (Capex floors, margin floors, and working-capital scaling). 

Cash flows are discounted at a scalar WACC to present values and combined with terminal values
derived from valuation multiples.

Notation
--------

- Years T (indexed t = 1,…,T), scenarios C = 3^T (Cartesian product of low/avg/high).

- For a given scenario c, denote Revenue_t(c) and EPS_t(c).

- FCFF methods produce FCFF_t^{(m)}(c) for method m.

- WACC is constant across t and scenarios; discount factors are D_t = 1 / (1+WACC)^{τ_t},
  with τ_t the year fraction from today to period t.

- Terminal value for a multiple family f with base B_f(c) is TV_f(c) = m_f · B_f(c),
  where m_f is a valuation multiple (ticker-specific and/or region–industry).

Regression Layer
----------------

For each target line y^k (e.g., EBIT, EBITDA, OCF, DA), a linear model in (1, Revenue, EPS)
is fitted on historical annuals:

    ŷ^k_t = β^k_0 + β^k_1 · Revenue_t + β^k_2 · EPS_t.

Coefficients {β^k} are obtained by the Huber–elastic-net procedure defined in
`fast_regression.py`. These are then applied to scenario paths to produce predicted
primitives needed for FCFF constructions.

FCFF Constructions
------------------

Let OCF = operating cash flow, INT = interest after tax, CAPEX, EBIT, DA, SBC, OOA,
AQ (acquisitions), NI (net income), EBITDA, CWC = change in working capital. The
following FCFF variants are computed per scenario c and year t:

1) OCF–INT–CAPEX:
      
       FCFF_t = OCF_t + INT_t − CAPEX_t.

2) EBITDA-minus:
      
       FCFF_t = EBITDA_t − CAPEX_t + AQ_t − CWC_t.

3) EBIT-based:
      
       FCFF_t = EBIT_t + DA_t + SBC_t + OOA_t + AQ_t − CWC_t − CAPEX_t.

4) NI-based:
      
       FCFF_t = NI_t + INT_t + DA_t + SBC_t + OOA_t + AQ_t − CWC_t − CAPEX_t.

5) FCF+INT:
     
       FCFF_t = FCF_t + INT_t.

Structural Constraints
----------------------

To avoid pathological paths, primitives are modified as:

- Capex floor vs depreciation:
    
      CAPEX_t = max( capex_rev_ratio · Revenue_t,  capex_da_floor · DA_t ).

- Margin floors:
    
      EBITDA_t ← max( EBITDA_t,  ebitda_margin_floor · Revenue_t ),
    
      EBIT_t   ← max( EBIT_t,    ebit_margin_floor   · Revenue_t ).

- Working capital change scales to revenue growth (if wc_alpha is available):
   
      ΔRevenue_t = Revenue_t − Revenue_{t−1}  (with ΔRevenue_1 = 0),
   
      CWC_t = clip( wc_alpha · ΔRevenue_t,  −wc_clip_frac · Revenue_t,  +wc_clip_frac · Revenue_t ).

Discounting
-----------

Let E denote equity market value (MCAP) and D the market value of debt. With a scalar
cost of equity `coe` and after-tax cost of debt `cod`, the weighted average cost of
capital is

    V = E + D,       
    
    WACC = (coe · E / V) + (cod · D / V).

For the date grid of forecast periods with year fractions τ_t, the discount factors are

    D_t = 1 / (1 + WACC)^{τ_t} 
    
        = exp( − τ_t · log(1+WACC) ).

Terminal Value by Multiples
---------------------------
Given a base B_f(c) and a positive multiple m_f>0, the terminal enterprise value for
scenario c and family f is

    TV_f(c) = m_f · B_f(c),

which is then discounted by D_Terminal = (1+WACC)^T (or equivalently exp(T · log(1+WACC))).
O
nly positive multiples and positive bases are used. In particular, for the earnings
route the base is EPS_T(c) · Shares, and the route is enabled only if EPS_T(c) > 0.
Other bases are Revenue_T(c), FCF_T(c), EBITDA_T(c), EBIT_T(c). 

Available multiple families are EV/Sales, EV/Earnings, EV/FCF, EV/EBITDA, EV/EBIT; families 
depending on EBITDA or EBIT are only used if the corresponding FCFF recipe is feasible.

Uncertainty Aggregation
-----------------------
Let disc_ff be the discounted FCFF tensor of shape (M, C, T) over methods M and scenarios C.

The present value per method–scenario is PV^{(m)}(c) = ∑_t disc_ff^{(m)}_t(c).

For each terminal route r, the discounted terminal value TV_r(c) is computed. The per-share 
price for route r, method m, scenario c is obtained by converting enterprise value to equity
value using a ticker-specific ratio (MC/EV) and dividing by shares outstanding:

    Price_{r,m}(c) = [ PV^{(m)}(c) + TV_r(c) ] · ( MC / EV ) / Shares.

Point estimates (e.g., percentiles across routes×methods×scenarios) are formed from the
distribution of Price_{r,m}(c). A simple standard-error proxy combines dispersion from
cash-flow legs and terminal legs:

    SE_years = std(disc_ff, axes=(methods, scenarios)) / sqrt(#analysts_t)  (per t),

    SE_TV    = std(TV_r(c) across r,c) / sqrt(#analysts_T),
 
    SE_total = sqrt( ∑_t SE_years(t)^2 + SE_TV^2 ) · ( MC / EV ) / Shares.

All arrays are implemented with NumPy and Pandas, and all transforms are vectorised.
"""


from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import datetime as dt
import itertools
from sklearn.model_selection import TimeSeriesSplit

from fast_regression6 import HuberENetCV  
from export_forecast import export_results
from financial_forecast_data4 import FinancialForecastData
import config


pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(
    level = logging.INFO, 
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

TODAY_TS = pd.Timestamp.today().normalize()

fdata = FinancialForecastData()

macro = fdata.macro

r = macro.r


REQUIRED_COLS = [
    "Revenue",
    "EPS",
    "OCF",
    "InterestAfterTax",
    "Capex",
    "EBIT",
    "Depreciation & Amortization",
    "Share-Based Compensation",
    "Acquisitions",
    "Net Income",
    "EBITDA",
    "Change in Working Capital",
    "Other Operating Activities",
    "FCF",
]

COLUMN_MAP = {
    "ocf": "OCF",
    "interest": "InterestAfterTax",
    "ebit": "EBIT",
    "da": "Depreciation & Amortization",
    "sbc": "Share-Based Compensation",
    "aq": "Acquisitions",
    "ni": "Net Income",
    "fcf": "FCF",
    "ebitda": "EBITDA",
    "cwc": "Change in Working Capital",
    "ooa": "Other Operating Activities",
}

KEYS = [
    "ocf", 
    "interest",
    "ebit", 
    "da",
    "sbc",
    "aq", 
    "ni", 
    "fcf", 
    "ebitda", 
    "cwc", 
    "ooa",
]

CONSTRAINED_KEYS = {
    "ocf", 
    "ebit", 
    "ni",
    "fcf",
    "ebitda"
}  

CONSTRAINED_MAP = {
    k: (k in CONSTRAINED_KEYS) for k in KEYS
}

ALPHAS = np.linspace(0.3, 0.7, 5)

LAMBDAS = np.logspace(0, -4, 20)  

HUBER_M_VALUES = (0.25, 1.0, 4.0)

CV_FOLDS = 5

TSCV = TimeSeriesSplit(n_splits = CV_FOLDS)


def match_ticker_interest_rate(
    tickers,
    country,
) -> float:
    """
    Map each ticker to an interest-rate series based on its country of risk.

    If a ticker's country is missing or not present in `macro.interest`, the
    United States series is used as a fallback.

    Parameters
    ----------
   
    tickers : iterable
        Collection of ticker symbols.
   
    country : pandas.Series-like
        Mapping from ticker to country string.

    Returns
    -------
    dict
        Mapping {ticker → interest_rate_series}.
    """
       
    ir = {}
    
    for ticker in tickers:
        
        ctry = country.get(ticker, np.nan)
   
        if pd.isna(ctry) or ctry not in macro.interest.index:
    
            ir[ticker] = macro.interest.loc['United States']
            
        else:
            
            ir[ticker] = macro.interest.loc[ctry]
    
    return ir


def cod(
    tickers,
    country,
):
    """
    Compute after-tax cost of debt per ticker.

    For ticker i, with nominal interest rate r_i and tax rate τ_i (in percent),
    the after-tax cost of debt is

        CoD_i = r_i · (1 − τ_i) / 100.

    Returns a pandas.Series named "Cost of Debt" indexed by ticker.
    """
        
    interest_rates = match_ticker_interest_rate(
        tickers = tickers,
        country = country,
    )
    
    tax_rate = r.tax_rate

    cost_of_debt_dict = {ticker: interest_rates[ticker] * (1.0 - tax_rate[ticker]) for ticker in tickers}

    cost_of_debt = pd.Series(cost_of_debt_dict, name = "Cost of Debt") / 100
    
    cost_of_debt.index.name = "Ticker"
    
    return cost_of_debt


def _prepare_regression_data(
    regression_data: pd.DataFrame,
):
    """
    Build the regression design and target dictionary.

    Selects the two predictors [Revenue, EPS] as X ∈ ℝ^{n×2} and the target
    dictionary Y = {key → column(COLUMN_MAP[key])}. Also constructs time-series
    cross-validation splits via `TimeSeriesSplit`.

    Returns
    -------
  
    X : np.ndarray, shape (n, 2)
  
    y_dict : Dict[str, np.ndarray]
        Targets for keys in KEYS.
  
    cv_splits : List[Tuple[np.ndarray, np.ndarray]]
        Train/test index pairs for CV.
    """
       
    X = regression_data[["Revenue", "EPS"]].to_numpy(dtype = float)

    y_dict = {
        k: regression_data[COLUMN_MAP[k]].to_numpy(dtype = float) for k in KEYS
    }
        
    cv_splits = list(TSCV.split(X))
        
    return X, y_dict, cv_splits
    

def _calc_wc_alpha(
    regression_data: pd.DataFrame,
):
    """
    Estimate a proportionality coefficient for working-capital changes.

    Using historical series of Revenue and Change in Working Capital (CWC), the
    median slope of ΔCWC on ΔRevenue is computed:

        ΔRevenue_t = Revenue_t − Revenue_{t−1},
  
        ΔCWC_t     = CWC_t − CWC_{t−1},
  
        wc_alpha   = median( ΔCWC_t / ΔRevenue_t ),  over finite ratios.

    The estimate is clipped to [-0.4, 0.4]. Returns None if insufficient data.
    """
    
    hist = regression_data.loc[:, ["Revenue","Change in Working Capital"]].dropna()
    
    if len(hist) >= 4:
    
        d_rev = hist["Revenue"].diff().to_numpy()
    
        d_wc = hist["Change in Working Capital"].diff().to_numpy()
      
        mask = np.isfinite(d_rev) & np.isfinite(d_wc) & (d_rev != 0)
        
        if mask.any():
      
            wc_alpha = np.median(d_wc[mask] / d_rev[mask])  
        
        else:
            
            return None
      
        wc_alpha = float(np.clip(wc_alpha, -0.4, 0.4))
        
        return wc_alpha
   
    else:
   
        return None


def _build_rev_eps_grid(
    forecast_df: pd.DataFrame,
    years: int,
):  
    """
    Construct the Cartesian scenario grid for (Revenue, EPS).

    For each forecast year t, the low/avg/high triplet is taken from columns:
    (low_rev, avg_rev, high_rev) and (low_eps, avg_eps, high_eps). The full
    scenario set is the Cartesian product across T years, yielding C = 3^T
    scenarios. The output arrays have shape (C, T).

    Returns
    -------
   
    REVS : np.ndarray, shape (C, T)
   
    EPS  : np.ndarray, shape (C, T)
   
    """   
    rev_vals = forecast_df[["low_rev", "avg_rev", "high_rev"]].to_numpy(dtype = float)

    eps_vals = forecast_df[["low_eps", "avg_eps", "high_eps"]].to_numpy(dtype = float)

    all_idx = np.array(list(itertools.product(range(3), repeat = years)), dtype = int)

    n_combo = all_idx.shape[0]

    REVS = np.take_along_axis(rev_vals, all_idx.T, axis = 1).T 

    EPS = np.take_along_axis(eps_vals, all_idx.T, axis = 1).T 
    
    return REVS, EPS


def fcff_methods(
    betas,
    REVS,
    EPS,
    capex_rev_ratio: float,
    shares_out: float,
    capex_da_floor: float = 1.0,      
    ebitda_margin_floor: float = 0.0, 
    ebit_margin_floor: float = 0.0, 
    wc_alpha: float | None = None, 
    wc_clip_frac: float = 0.25,     
):
    """
    Generate FCFF paths under structural constraints and collect terminal bases.

    Regression Application
    ----------------------
  
    Let F(c,t) = [1, Revenue_t(c), EPS_t(c)] for scenario c and year t, and
    β^k = (β^k_0, β^k_1, β^k_2) the coefficients for target k ∈ KEYS. 
    
    Predicted lines are assembled as

        Y_k(c,t) = β^k_0 + β^k_1 · Revenue_t(c) + β^k_2 · EPS_t(c),

    implemented via an einsum over the stacked tensors.

    Structural Constraints
    ----------------------
    With base_capex_t(c) = capex_rev_ratio · Revenue_t(c), constraints are

        CAPEX_t = max( base_capex_t,  capex_da_floor · DA_t ),

        EBITDA_t ← max( EBITDA_t,  ebitda_margin_floor · Revenue_t ),

        EBIT_t   ← max( EBIT_t,    ebit_margin_floor   · Revenue_t ),

    and, if wc_alpha is present,

        ΔRev_t = Revenue_t − Revenue_{t−1}   (ΔRev_1 = 0),

        CWC_t  = clip( wc_alpha · ΔRev_t,  −wc_clip_frac · Revenue_t,  +wc_clip_frac · Revenue_t ).

    FCFF Variants
    -------------
    Per scenario and year:

      (1) OCF–INT–CAPEX      : FCFF = OCF + INT − CAPEX

      (2) EBITDA-minus       : FCFF = EBITDA − CAPEX + AQ − CWC

      (3) EBIT-based         : FCFF = EBIT + DA + SBC + OOA + AQ − CWC − CAPEX

      (4) NI-based           : FCFF = NI + INT + DA + SBC + OOA + AQ − CWC − CAPEX

      (5) FCF+INT            : FCFF = FCF + INT

    Terminal Bases and Route Eligibility
    ------------------------------------
    Terminal base vector at T is

        revenue_T  = Revenue_T,

        earnings_T = EPS_T · Shares   (set NaN when EPS_T ≤ 0),

        fcf_T      = FCF_T,

        ebitda_T   = EBITDA_T,

        ebit_T     = EBIT_T.

    Allowed route flags are:

      - EV/Sales, EV/Earnings, EV/FCF: always allowed (earnings only if any EPS_T>0),

      - EV/EBITDA: allowed iff the "ebitda_minus" FCFF recipe is finite,

      - EV/EBIT  : allowed iff the "ebitda_based"   FCFF recipe is finite.

    Returns
    -------

    valid_ff : List[np.ndarray]

        FCFF tensors of shape (C, T) for each feasible recipe.

    terminals : dict[str, np.ndarray]

        Terminal bases at T for {"revenue","earnings","fcf","ebitda","ebit"}.

    allowed : dict[str, bool]

        Eligibility flags for multiple families: {"rev","earnings","fcf","ebitda","ebit"}.
    """
   
    B = np.stack([betas[k] for k in KEYS], axis = 0)        
   
    F = np.stack([np.ones_like(REVS), REVS, EPS], axis = -1)  

    Y = np.einsum("tj,cyj->tcy", B, F)

    (
        ocf_p,
        int_p,
        ebit_p,
        da_p,
        sbc_p,
        aq_p,
        ni_p,
        fcf_p,
        ebitda_p,
        cwc_p,
        ooa_p,
    ) = Y

    base_capex = capex_rev_ratio * REVS
   
    capex = np.maximum(base_capex, capex_da_floor * da_p)
    
    ebitda_p = np.maximum(ebitda_p, ebitda_margin_floor * REVS)
   
    ebit_p = np.maximum(ebit_p, ebit_margin_floor * REVS)

    if wc_alpha is not None:
       
        d_rev = np.diff(REVS, axis = 1, prepend=REVS[:, :1])
       
        cwc_scaled = wc_alpha * d_rev
       
        cwc_clip = wc_clip_frac * REVS

        cwc_p = np.clip(cwc_scaled, -cwc_clip, cwc_clip)

    recipes = [
        ("ocf_int_capex", lambda: ocf_p + int_p - capex),
        ("ebitda_minus", lambda: ebitda_p - capex + aq_p - cwc_p),
        ("ebitda_based", lambda: ebit_p + da_p + sbc_p + ooa_p + aq_p - cwc_p - capex),
        ("ni_based", lambda: ni_p + int_p + da_p + sbc_p + ooa_p + aq_p - cwc_p - capex),
        ("fcf_plus_int", lambda: fcf_p + int_p),
    ]

    valid_ff = []
    
    valid_names = []

    for name, fn in recipes:
    
        arr = fn()
    
        if np.isfinite(arr).all():
    
            valid_ff.append(arr)
    
            valid_names.append(name)
    
        else:
    
            logger.info("• skipping FCFF method %s (non-finite inputs)", name)
            
    eps_T = EPS[:, -1]
    
    earnings_base = np.where(eps_T > 0, eps_T * shares_out, np.nan).astype(np.float32)

    terminals = {
        "revenue": REVS[:, -1],
        "earnings": earnings_base, 
        "fcf": fcf_p[:, -1],
        "ebitda": ebitda_p[:, -1],
        "ebit": ebit_p[:, -1],
    }

    allowed = {
        "rev": True,                
        "earnings": (np.isfinite(earnings_base).any()),          
        "fcf": True,               
        "ebitda": ("ebitda_minus" in valid_names),  
        "ebit": ("ebitda_based" in valid_names),   
    }
    
    
    if logger.isEnabledFor(logging.DEBUG):
       
       
        def _count_valid(
            x
        ): 
            
            return int(np.isfinite(x).sum())
       
       
        logger.debug(
            "TV valid cols — EV/S:%d, EV/E:%d, EV/FCF:%d, EV/EBITDA:%d, EV/EBIT:%d",
            _count_valid(
                x = terminals["revenue"]
            ),
            _count_valid(
                x = terminals["earnings"]
            ),
            _count_valid(
                x = terminals["fcf"]
            ),
            _count_valid(
                x = terminals["ebitda"]
            ),
            _count_valid(
                x = terminals["ebit"]
            ),
        )

    return valid_ff, terminals, allowed


def _regression(
    regression_data: pd.DataFrame,
    years: int,
    forecast_df: pd.DataFrame,
    capex_rev_ratio: float,
    shares_out: float,
    cv: HuberENetCV,
):
    """
    Run the regression layer and build FCFF paths and terminal bases.

    Steps
    -----
  
    1) Prepare regression data (X = [Revenue, EPS], targets in KEYS) and CV splits.
  
    2) Fit joint Huber–elastic-net across targets; obtain {β^k}.
  
    3) Build the scenario grid REVS, EPS with C = 3^years combinations.
  
    4) Estimate wc_alpha from history.
  
    5) Call `fcff_methods` to obtain feasible FCFF arrays, terminal bases, and
       allowed multiple flags.

    Returns
    -------
    valid_ff : List[np.ndarray]
    terminals : dict[str, np.ndarray]
    allowed : dict[str, bool]
  
    """    
  
    X, y_dict, cv_splits = _prepare_regression_data(
        regression_data = regression_data
    )

    betas_by_key, best_lambda, best_alpha, best_M = cv.fit_joint(
        X = X,
        y_dict = y_dict,
        constrained_map = CONSTRAINED_MAP,
        cv_splits = cv_splits,
        scorer = None, 
    )
    
    REVS, EPS = _build_rev_eps_grid(
        forecast_df = forecast_df,
        years = years,
    )
    
    wc_alpha = _calc_wc_alpha(
        regression_data = regression_data
    )

    valid_ff, terminals, allowed = fcff_methods(
        betas = betas_by_key,
        REVS = REVS,
        EPS = EPS,
        capex_rev_ratio = capex_rev_ratio,
        shares_out = shares_out,
        wc_alpha = wc_alpha,
    )
    
    return valid_ff, terminals, allowed


def _build_discount_factor_vector(
    forecast_df: pd.DataFrame,
    coe: float,
    E: float,
    cost_of_debt: float,
    mv_debt: float,
    today_ts: pd.Timestamp = TODAY_TS,
):
    """
    Construct discount factors and a scalar WACC.

    With equity value E (market cap), debt value D, cost of equity coe, and after-tax
    cost of debt cod, define V = E + D and

        WACC = (coe · E / V) + (cod · D / V).

    For forecast dates t with year fractions τ_t = (date_t − today)/365, the discount
    factors are computed as

        D_t = 1 / (1 + WACC)^{τ_t} = exp( − τ_t · log(1 + WACC) ).

    Returns
    -------
  
    discount_factors : np.ndarray, shape (T,)
    WACC : float
    """
    
    days = (forecast_df.index.to_numpy() - np.datetime64(today_ts)) / np.timedelta64(1, "D")

    years_frac = days / 365.0

    V = E + mv_debt
    
    WACC = (coe * E / V) + (cost_of_debt * mv_debt / V)

    discount_factors = 1.0 / np.exp(years_frac * np.log1p(WACC))
    
    return discount_factors, WACC


def _m_list(
    val_ticker,
    val_industry
):
    """
    Build a two-element vector of multiples [ticker_value, industry_value], dropping NaNs.

    Returns
    -------
    np.ndarray
        Array of length 1 or 2 containing finite multiples for the given family.
    """
    
    return pd.Series([val_ticker, val_industry]).dropna().to_numpy(dtype = float)


def terminal_values(
    evs_ind_dict,
    eve_ind_dict,
    evfcf_ind_dict,
    evebitda_ind_dict,
    evebit_ind_dict,
    kpis,
    terminals: dict[str, np.ndarray],   
    allowed: dict[str, bool],        
    WACC: float,
    years: int,
):
    """
    Compute discounted terminal enterprise values across multiple families and sources.

    For each multiple family f ∈ {EV/S, EV/E, EV/FCF, EV/EBITDA, EV/EBIT}, two sources
    (ticker-specific expected value and region–industry value) are collected. 
    
    For each source m_f and scenario base B_f(c), a route is formed:

        TV_f(c) = m_f · B_f(c),

    with the following positivity filters:
    
      - m_f > 0 and finite,
    
      - B_f(c) > 0 and finite (earnings base uses EPS_T·Shares and is NaN if EPS_T ≤ 0).

    Each route is discounted by

        D_Terminal = (1 + WACC)^T = exp( T · log(1 + WACC) ),

    yielding a route-by-scenario matrix of discounted terminal values. If no routes
    are eligible, an empty matrix with shape (0, C) is returned.

    Returns
    -------
    tv_disc_all : np.ndarray, shape (R, C)
        Discounted terminal values across routes R and scenarios C.
    n_routes : int
        Number of routes included.
    """

    evs_arr = _m_list(
        val_ticker = kpis["exp_evs"].iat[0], 
        val_industry = evs_ind_dict["Region-Industry"]
    )
    
    eve_arr = _m_list(
        val_ticker = kpis["eve_t"].iat[0],
        val_industry = eve_ind_dict["Region-Industry"]
    )
    
    evfcf_arr = _m_list(
        val_ticker = kpis["exp_evfcf"].iat[0],  
        val_industry = evfcf_ind_dict["Region-Industry"]
    )
   
    evebitda_arr = _m_list(
        val_ticker = kpis["exp_evebitda"].iat[0], 
        val_industry = evebitda_ind_dict["Region-Industry"]
    )
   
    evebit_arr = _m_list(
        val_ticker = kpis["exp_evebit"].iat[0],
        val_industry = evebit_ind_dict["Region-Industry"]
    )

    routes = []  
    
    disc_T = np.exp(years * np.log1p(WACC))  
    
    routes = []


    def _push(
        multiples: np.ndarray, 
        base: np.ndarray
    ):
    
        if multiples.size == 0 or base.size == 0:
    
            return
    
        pos_m = np.isfinite(multiples) & (multiples > 0)
    
        pos_b = np.isfinite(base) & (base > 0)
    
        if not pos_m.any() or not pos_b.any():
    
            return
    
        m = multiples[pos_m][:, None]
    
        b = base[None, pos_b]
    
        tv = np.full((pos_m.sum(), base.size), np.nan, dtype = np.float32)
    
        tv[:, pos_b] = (m * b) / disc_T
    
        routes.append(tv)


    _push(
        multiples = evs_arr, 
        base = terminals["revenue"]
    )
   
    if allowed["earnings"]: 
        
        _push(
            multiples = eve_arr,     
            base = terminals["earnings"]
        )
    
    if allowed["fcf"]: 
             
        _push(
            multiples = evfcf_arr,   
            base = terminals["fcf"]
        )
    
    if allowed["ebitda"]:  
        
        _push(
            multiples = evebitda_arr, 
            base = terminals["ebitda"]
        )
    
    if allowed["ebit"]:
        
        _push(
            multiples = evebit_arr,   
            base = terminals["ebit"]
        )
    
    if not routes:
    
        return np.empty((0, terminals["revenue"].shape[0]), dtype = np.float32), 0

    tv_disc_all = np.vstack(routes).astype(np.float32)

    return tv_disc_all, tv_disc_all.shape[0]

 
def build_DCF(
    kpis,
    forecast_df: pd.DataFrame,
    coe,
    mcap,
    cost_of_debt,
    valid_ff,
    evs_ind,
    eve_ind,
    evfcf_ind,
    evebitda_ind,
    evebit_ind,
    shares_out,
    years: int,
    terminals: dict[str, np.ndarray],  
    allowed: dict[str, bool],        
):
    """
    Assemble discounted cash flows and terminal values to produce per-share valuations.

    Definitions
    -----------
  
    - Let FCFF^{(m)}_t(c) be FCFF for method m, year t, scenario c.
  
    - Discounted FCFF: disc_ff^{(m)}_t(c) = FCFF^{(m)}_t(c) · D_t.
  
    - Present-value leg: PV^{(m)}(c) = ∑_{t=1}^T disc_ff^{(m)}_t(c).
  
    - Terminal leg: TV_r(c) from `terminal_values`, for route r.

    Enterprise Value to Equity Value
    --------------------------------
  
    The enterprise-value result is converted to per-share price via a ticker-specific
    scalar κ = (MC / EV) / Shares, where MC/EV is supplied by KPIs:

        Price_{r,m}(c) = [ PV^{(m)}(c) + TV_r(c) ] · κ,
  
        κ = (MC / EV) / Shares.

    Uncertainty Proxy
    -----------------
  
    A standard-error proxy combines dispersion from cash-flow legs and terminal legs:

        SE_years(t) = std( disc_ff[:,:,t] ) / sqrt(num_analysts_t),
  
        SE_TV       = std( tv_disc_all )     / sqrt(num_analysts_T),
  
        SE_total    = sqrt( ∑_t SE_years(t)^2 + SE_TV^2 ) · κ.

    Returns
    -------
    fut_mcap_pred : np.ndarray
  
        Tensor of per-share valuations across routes×methods×scenarios with clipping
        applied at configured lower/upper bounds downstream.
  
    se : float
  
        Aggregate standard error proxy on the per-share scale.
   
    """
    
    mv_debt_t = float(kpis["market_value_debt"].iat[0])
        
    mc_ev = kpis["mc_ev"].iat[0]
    
    na = forecast_df['num_analysts'].to_numpy(dtype = float)
    
    discount_factors, WACC = _build_discount_factor_vector(
        forecast_df = forecast_df,
        coe = coe,
        E = mcap,
        cost_of_debt = cost_of_debt,
        mv_debt = mv_debt_t,
    )

    fcff_raw = np.stack(valid_ff, axis = 0) 

    disc_ff = fcff_raw * discount_factors[None, None, :]   

    sum_ff = disc_ff.sum(axis = 2)                              

    tv_disc_all, n_routes = terminal_values(
        evs_ind_dict = evs_ind,
        eve_ind_dict = eve_ind,
        evfcf_ind_dict = evfcf_ind,
        evebitda_ind_dict = evebitda_ind,
        evebit_ind_dict = evebit_ind,
        kpis = kpis,
        terminals = terminals,
        allowed = allowed,
        WACC = WACC,
        years = years,
    )

    dcf = sum_ff[None, :, :] + tv_disc_all[:, None, :]    
    
    se_by_year = np.std(disc_ff, axis = (0, 1), ddof = 1) / np.sqrt(na)  
        
    tv_se = np.std(tv_disc_all, ddof = 1) / np.sqrt(float(forecast_df['num_analysts'].iat[-1]))
    
    mc_ev_shares = mc_ev / shares_out
    
    fut_mcap_pred = dcf * mc_ev_shares

    se = np.sqrt(np.sum(se_by_year ** 2) + tv_se ** 2) * mc_ev_shares
    
    return fut_mcap_pred, se


def main():
    """
    Entry point to run the DCF pipeline across the configured ticker universe.

    Steps
    -----
   
    1) Load macro series, KPIs, forecasts, and historical annuals.
   
    2) Compute after-tax cost of debt per ticker.
   
    3) For each ticker:
   
         - Validate required columns and forecast availability.
   
         - Run the regression layer and build FCFF paths and terminal bases.
   
         - Discount cash flows, compute terminal values, and convert EV to price.
   
         - Aggregate into low/median/mean/high point summaries and SE.
   
    4) Assemble a DataFrame of valuation summaries indexed by ticker.
   
    """
    tickers = list(config.tickers)

    latest_prices = r.last_price

    market_cap = r.mcap

    shares_outstanding = r.shares_outstanding

    country = r.country

    coe = pd.read_excel(
        config.FORECAST_FILE,
        sheet_name = "COE",
        index_col = 0,
        usecols = ["Ticker", "COE"],
        engine = "openpyxl",
    )
    
    cost_of_debt = cod(
        tickers = tickers,
        country = country,
    )

    r_dicts = r.dicts()

    evs_ind_dict = r_dicts["EVS"]
    
    eve_ind_dict = r_dicts["EVE"]
    
    evfcf_ind_dict = r_dicts["EVFCF"]
    
    evebitda_ind_dict = r_dicts["EVEBITDA"]
    
    evebit_ind_dict = r_dicts["EVEBIT"]


    def _zero_dicts(
        ticker: str
    ):

        for d in (low_price_dict, avg_price_dict, high_price_dict, returns_dict, se_dict):

            d[ticker] = 0.0


    low_price_dict: dict = {}

    avg_price_dict: dict = {}

    high_price_dict: dict = {}

    returns_dict: dict = {}

    se_dict: dict = {}

    lb = config.lbp * latest_prices 

    ub = config.ubp * latest_prices
    
    kpis_dict = fdata.kpis
    
    forecast_dict = fdata.forecast

    fin_dict = fdata.annuals

    cv = HuberENetCV(
        alphas = ALPHAS,
        lambdas = LAMBDAS,
        Ms = HUBER_M_VALUES,
        n_splits = CV_FOLDS,
        n_jobs = -1,                 
    )

    for ticker in tickers:
        
        logger.info("Processing ticker: %s", ticker)

        fin_df = fin_dict[ticker]
    
        if fin_df is None:
        
            logger.info("Skipping %s: missing financials.", ticker)
        
            _zero_dicts(
                ticker = ticker
            )
        
            continue

        present = [c for c in REQUIRED_COLS if c in fin_df.columns]
    
        missing = set(REQUIRED_COLS) - set(present)
    
        if missing:
        
            logger.warning("%s missing columns %s. Skipping regression.", ticker, missing)
        
            _zero_dicts(
                ticker = ticker
            )
        
            continue

        regression_data = fin_df.dropna(subset = present)
    
        if regression_data.empty:
            
            logger.warning("Regression data empty for %s. Skipping.", ticker)
        
            _zero_dicts(
                ticker = ticker
            )
            
            continue

        forecast_df = forecast_dict[ticker].dropna()
        
        if forecast_df.empty:
            
            logger.warning("No forecast data for %s, skipping DCF.", ticker)
        
            _zero_dicts(
                ticker = ticker
            )
        
            continue

        kpis = kpis_dict[ticker]
        
        capex_rev_ratio = kpis["capex_rev_ratio"].iat[0]

        years = len(forecast_df)
        
        shares_out_t = float(shares_outstanding[ticker])
        
        valid_ff, terminals, allowed = _regression(
            regression_data = regression_data,
            years = years,
            forecast_df = forecast_df,
            capex_rev_ratio = capex_rev_ratio,
            shares_out = shares_out_t,
            cv = cv,
        )

        if not valid_ff:
        
            logger.warning("No complete FCFF formulas for %s; skipping.", ticker)
        
            _zero_dicts(
                ticker = ticker
            )
            
            continue
        
        coe_t = float(coe.loc[ticker].iat[0])
        
        evs_ind_t = evs_ind_dict[ticker]
        
        eve_ind_t = eve_ind_dict[ticker]
        
        evfcf_ind_t = evfcf_ind_dict[ticker]
        
        evebitda_ind_t = evebitda_ind_dict[ticker]
        
        evebit_ind_t = evebit_ind_dict[ticker]
        
        cost_of_debt_t = float(cost_of_debt[ticker])
        
        mcap_t = float(market_cap[ticker])
        
        fut_mcap_pred, se = build_DCF(
            kpis = kpis,
            forecast_df = forecast_df,
            coe = coe_t,
            mcap = mcap_t,
            cost_of_debt = cost_of_debt_t,
            valid_ff = valid_ff,
            evs_ind = evs_ind_t,
            eve_ind = eve_ind_t,
            evfcf_ind = evfcf_ind_t,
            evebitda_ind = evebitda_ind_t,
            evebit_ind = evebit_ind_t,
            shares_out = shares_out_t,
            years = years,
            terminals = terminals,      
            allowed = allowed,   
        )

        prices = (fut_mcap_pred).clip(lb[ticker], ub[ticker])
    
        px = prices.reshape(-1)
    
        low_price = float(np.nanpercentile(px, 10))
    
        med_price = float(np.nanpercentile(px, 50))
        
        mean_price = float(np.nanmean(px))
    
        high_price = float(np.nanpercentile(px, 90))

        low_price_dict[ticker] = low_price
    
        avg_price_dict[ticker] = med_price
    
        high_price_dict[ticker] = high_price
    
        se_dict[ticker] = se

        latest_price = latest_prices[ticker]
        
        returns_dict[ticker] = float((med_price / latest_price - 1.0))

        logger.info(
            "Ticker: %s | Low (P10): %.2f | Med (P50): %.2f | Mean P: %.2f | High (P90): %.2f | SE: %.2f ",
            ticker, low_price, med_price, mean_price, high_price, se
        )

    dcf_df = pd.DataFrame(
        {
            "Low Price": low_price_dict,
            "Avg Price": avg_price_dict, 
            "High Price": high_price_dict,
            "SE": se_dict,
        }
    )

    dcf_df.index.name = "Ticker"
    
    sheets = {
        'DCF': dcf_df,
    }
    
    export_results(
        sheets = sheets,
        output_excel_file = config.MODEL_FILE,
    )


if __name__ == "__main__":
    
    main()
