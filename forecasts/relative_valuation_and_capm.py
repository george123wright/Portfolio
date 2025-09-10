"""
End-to-end equity valuation and risk pipeline with Black–Litterman market views,
macro BVAR simulations, factor VARX conditioning, and Monte-Carlo pricing overlays.

Overview
--------
This module orchestrates the construction of one-year equity return forecasts and
valuation summaries by combining:

  1) Black–Litterman (BL) quarterly index views derived from forecast index levels;
 
  2) A Minnesota-prior Bayesian VAR (BVAR) for quarterly macro variables with
     simulation under heavy-tailed or bootstrap innovations;
 
  3) A VARX model of Fama–French factors conditional on simulated macro paths;
 
  4) A quarter-wise alignment of the market factor to BL beliefs (mean and variance),
     using either resampling, rescaling, or covariance-mapping to preserve factor
     dependence;
 
  5) Per-stock factor regressions with Newey–West (HAC) covariance and Monte Carlo
     propagation of factor and idiosyncratic risks to annual total returns;
 
  6) Fundamental price overlays from comparables-based models (P/E, P/S, EV/S, P/B)
     and Graham-style heuristics; and
 
  7) CAPM-based diagnostics and cost-of-equity (CoE) estimates including currency
     adjustments.

Data requirements
-----------------
Inputs expected by the pipeline include:
  
  • Historical weekly and quarterly index returns and forecast index levels by quarter;
  
  • Quarterly macroeconomic panel for a chosen domain (e.g., United States);
  
  • Quarterly Fama–French factors (3- or 5-factor sets) including the risk-free rate;
  
  • Security-level historical returns, analyst forecasts, and valuation multiples;
  
  • Currency series and forecasts to form blended FX expectations.

Core models and equations
-------------------------
Black–Litterman:
  
  Let π_ann denote annual prior means per index; quarterly prior means are
  
  π_q = (1 + π_ann)^(1/4) − 1.
  
  Let Σ_w be the weekly covariance; quarterly prior covariance is Σ_q = 13 · Σ_w.
  
  For quarter q with identity view matrix P = I and view vector Q^(q),
  the posterior mean and covariance are
  
    μ_post^(q) = π_q + τ Σ_q P' (P τ Σ_q P' + Ω)^{-1} (Q^(q) − P π_q),
  
    Σ_post^(q) = ((τ Σ_q)^{-1} + P' Ω^{-1} P)^{-1}.
  
  Risk aversion δ is inferred from equal-weight equilibrium if not supplied:
  
    δ = (w_eq' π_q) / (w_eq' Σ_q w_eq), with w_eq = (1/k) · 1.
  
  The scalar τ is set to 1/(T − k − 1) by default, where T is weekly sample size.

Minnesota BVAR for macro:
  For macro vector y_t (k×1), the VAR(p) is

    y_t = c + ∑_{ℓ=1}^p A_ℓ y_{t−ℓ} + u_t,
 
  with u_t ~ (0, Σ). Per-equation ridge regression implements the Minnesota prior:
 
    argmin_b ||y − X b||^2 + (b − m)' D (b − m),
 
  where D encodes shrinkage by series and lag, with cross-lag penalties scaled by
  relative volatilities and a lag-decay term. Hyperparameters are selected by
  expanding-window backtest on one-step RMSE. Simulations use Gaussian, Student-t,
  or circular block bootstrap innovations.

Factor VARX:
  For factor vector f_t and macro exogenous z_t,
   
    f_t = α + ∑_{ℓ=1}^p Φ_ℓ f_{t−ℓ} + B z_t + ε_t,
  
  with ε_t ~ (0, Σ_f). Parameters are estimated by OLS per equation. Factor paths
  are simulated conditional on macro simulation draws z_t.

BL alignment of the market factor:
  The factor named "mkt_excess" is adjusted, per quarter, to match BL beliefs for
  total market return μ_BL^(q) and variance Var_BL^(q). The target excess mean is
  
    μ_target^(q) = μ_BL^(q) − r_f,  where r_f is the per-quarter risk-free rate.

  Three modes are supported:
    • resample: replace market draws by independent samples with the target mean
      and variance (Gaussian or Student-t);
    • rescale: shift/scale existing market draws to match moments (preserves ranks
      and approximate correlations);
    • covmap: apply a covariance-mapping transform Y = μ* + L* L_0^{-1} (X − μ_0)
      that sets the market mean/variance to targets while preserving cross-factor
      correlations with the market and leaving the non-market covariance block
      unchanged.

Per-stock factor model and Monte Carlo:
  For stock i, quarterly excess return r_{i,t}^ex is modelled as

    r_{i,t}^ex = β_i' f_t + e_{i,t},
  
  with β_i estimated by OLS and HAC standard errors (Newey–West). Idiosyncratic
  shocks e_{i,t} are simulated as Gaussian, Student-t, or bootstrap resamples
  calibrated to the residual variance. Quarterly total returns are formed as

    r_{i,t} = r_f + r_{i,t}^ex,

  and annual compounding is

    R_i = exp(∑_{q=1}^4 log(1 + r_{i,q})) − 1.

CAPM and cost of equity:
  
  CAPM expected return uses E[R_i] = r_f + β_i (E[R_M] − r_f), with β_i estimated
  from weekly data. The cost of equity integrates market, currency, and country-risk
  adjustments as supplied by supporting utilities.

Configuration and outputs
-------------------------
Key knobs include innovation types, Student-t degrees of freedom, bootstrap block
lengths, Minnesota hyperparameter grids, and BL confidence settings. The module
emits:

  • Posterior BL means per index and posterior covariances per quarter;

  • Macro BVAR simulation panels (Q1..QH);

  • Factor simulations before and after BL market alignment, with per-quarter
    empirical means/covariances;

  • Per-ticker Monte-Carlo distributions of annual returns (mean, quantiles);

  • Valuation sheets from fundamental overlays; and

  • Diagnostic CAPM/CoE summaries.

Assumptions and conventions
---------------------------
  • All returns are decimals. Weekly-to-quarterly covariance scaling uses 13.

  • Quarter labels are canonicalised to quarter-end timestamps.

  • Random seeds are set where appropriate for reproducibility.

  • Monte-Carlo statistics match targets up to sampling error.

Dependencies
------------
Relies on `pandas`, `numpy`, `statsmodels` (VAR), and project-local helpers:
Black–Litterman assembly, BVAR fitter/simulator, VARX simulator, factor predictor,
and valuation utilities. Logging is available via the module-level logger.

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
from data_processing.macro_data import MacroData
from functions.factor_data import load_factor_data
from functions.factor_simulations import factor_sim_varx, apply_bl_market_to_factor_sims
from functions.factor_model_pred import ff_pred_mc
from functions.bvar_minnesota import fit_bvar_minnesota, simulate_bvar
from functions.factor_exponential_regression import exp_fac_reg
import config

logger = logging.getLogger(__name__)


MACRO_COLS = ["Interest", "Cpi", "Gdp", "Unemp", "Balance Of Trade", "Corporate Profits", "Balance On Current Account"]


def run_black_litterman_on_indexes(
    annual_ret: pd.Series,
    hist_ret: pd.DataFrame,
    future_q_rets: pd.DataFrame,
    tau: float = None,
    delta: float = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Compute quarter-by-quarter Black–Litterman (BL) posterior index return
    expectations and associated posterior covariances, using historical
    index co-movement and externally supplied quarterly index views.

    Modelling set-up
    ----------------
    Let there be k indices (assets), with:
      • prior mean (annual) π_ann (vector length k) converted to quarterly
        via π_q = (1 + π_ann)^(1/4) − 1,
      • prior covariance Σ_q for quarterly returns, formed from historical
        weekly covariance Σ_w by quarterly scaling: Σ_q = 13 · Σ_w,
        recognising approximately 13 weeks per quarter,
      • view matrix P = I_k (identity), so each view applies to a single
        index independently,
      • view vector for quarter q: Q^(q) (k × 1), taken from `future_q_rets`,
      • risk aversion δ and scalar τ governing prior mean uncertainty.

    The standard BL posterior for a single quarter q is:
     
      μ_post^(q) = π_q + τ Σ_q P' (P τ Σ_q P' + Ω)^{-1} (Q^(q) − P π_q),
     
      Σ_post^(q) = ( (τ Σ_q)^{-1} + P' Ω^{-1} P )^{-1},
    
    where Ω is the view covariance (set inside `black_litterman`, commonly
    Ω = diag( P τ Σ_q P' ) · c for a confidence scalar c).

    Estimation of δ and τ
    ---------------------
    If δ is not provided, it is inferred from equal-weight equilibrium:
   
      w_eq = (1/k) · 1, 
     
      δ = (w_eq' π_q) / (w_eq' Σ_q w_eq).
    
    If τ is not provided, the default τ = 1 / (T − k − 1) is used, where T is
    the number of historical weekly observations in `hist_ret`.

    Inputs
    ------
    annual_ret : pandas.Series
        Annualised prior mean returns per index (decimal), indexed by tickers.
    hist_ret : pandas.DataFrame
        Historical weekly index returns (decimal), columns matching
        `annual_ret.index`. Used to estimate Σ_w.
    future_q_rets : pandas.DataFrame
        Quarter-specific views, columns named "Q1","Q2",…; rows indexed by the
        same tickers as `annual_ret`. Entry Q^(q)_i is the view for index i in
        quarter q (decimal).

    Parameters
    ----------
    tau : float, optional
        BL τ parameter. If None, set to 1 / (T − k − 1).
    delta : float, optional
        Risk-aversion parameter. If None, inferred from equal-weight equilibrium.

    Returns
    -------
    bl_df : pandas.DataFrame
        Posterior mean returns by index for each forecast quarter, with an
        additional column 'Ann' equal to the compounded annual return across
        the provided quarters:
          Ann = (∏_q (1 + μ_post^(q))) − 1.
    sigma_bl_by_q : dict[str, pandas.DataFrame]
        Mapping from quarter label (e.g., "Q1") to the BL posterior covariance
        matrix Σ_post^(q) (k × k, indexed by tickers).

    Notes
    -----
    • Identity P assumes mutually independent index views in Q^(q). Cross-asset
      relationships still enter via Σ_q.
    • `black_litterman` is called per quarter with the same Σ_q and π_q but
      quarter-specific Q^(q), returning μ_post^(q) and Σ_post^(q).
    """
   
    assets = annual_ret.index.intersection(future_q_rets.index)

    pi_ann = annual_ret.loc[assets]
   
    pi_q = (1 + pi_ann) ** (1 / 4) - 1  

    cov_hist = hist_ret[assets].cov()

    sigma_prior = cov_hist * 13

    k = len(assets)

    if delta is None:
       
        w_eq = pd.Series(1.0 / k, index = assets)

        delta = float(pi_q.dot(w_eq) / (w_eq.T.dot(sigma_prior).dot(w_eq)))

    if tau is None:
        
        tau = 1.0 / (len(hist_ret) - k - 1)

    w_prior = pd.Series(1.0 / k, index = assets)

    bl_post_means = {}
   
    sigma_bl_by_q: dict[str, pd.DataFrame] = {}

    qcols = [c for c in future_q_rets.columns if c.upper().startswith("Q")]

    future_q_rets = future_q_rets.loc[assets, qcols]

    P = pd.DataFrame(np.eye(len(assets)), index = assets, columns = assets)

    for q in qcols:

        Q = future_q_rets[q].loc[assets]

        mu_q, sigma_bl_q = black_litterman(
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
        
        bl_post_means[q] = mu_q
        
        sigma_bl_by_q[q] = sigma_bl_q

    bl_df = pd.DataFrame(bl_post_means)
   
    bl_df['Ann'] = (1 + bl_df).prod(axis = 1) - 1

    return bl_df, sigma_bl_by_q


def blend_fx_returns(
    hist_series: pd.Series,
    annual_forecast: pd.Series,
    weight: float = 0.5
) -> pd.Series:
    """
    Form a convex blend of historical and forecast foreign-exchange (FX) annual
    returns after aligning tickers.

    Procedure
    ---------
    1) Remove trailing '=X' from `hist_series` indices for compatibility with
       forecast tickers.
   
    2) Align the two series on the intersection of tickers.
   
    3) Return the convex combination:
   
         blended = weight · hist + (1 − weight) · forecast.

    Parameters
    ----------
    hist_series : pandas.Series
        Historical annual FX returns (decimal), indexed by tickers, potentially
        with a '=X' suffix.
    annual_forecast : pandas.Series
        External forecast of annual FX returns (decimal), indexed by tickers
        without '=X'.
    weight : float, default 0.5
        Weight on the historical component. Must lie in [0, 1].

    Returns
    -------
    pandas.Series
        Blended annual FX return per currency on the common index.

    Notes
    -----
    The function performs no scaling or currency conversion; it assumes both
    inputs are comparable annual percentage changes expressed in decimal form.
    """
      
    hist_series.index = hist_series.index.str.replace(r'=X$', '', regex=True)
  
    h, f = hist_series.align(annual_forecast, join = 'inner')
  
    return weight * h + (1 - weight) * f


def _parse_quarter_col(
    s: pd.Series
) -> pd.DatetimeIndex:
    """
    Parse quarter labels into quarter-end timestamps.

    Accepted formats
    ----------------
    Strings of the form 'YYYYQn' (case-insensitive), e.g., '2010Q1'.
    Any non-alphanumeric separators are stripped before validation.

    Parameters
    ----------
    s : pandas.Series
        Series of quarter labels to parse.

    Returns
    -------
    pandas.DatetimeIndex
        Quarter-end timestamps corresponding to the input labels.

    Raises
    ------
    ValueError
        If any label does not match the pattern 'YYYYQn' with n in {1,2,3,4}.
    """
  
    s = s.astype(str).str.strip().str.upper()
  
    s = s.str.replace(r"[^0-9Q]", "", regex = True)
  
    bad = ~s.str.match(r"^\d{4}Q[1-4]$")
  
    if bad.any():
  
        raise ValueError(f"Quarter strings malformed at rows: {list(np.where(bad)[0][:5])} (showing first 5)")
  
    return pd.PeriodIndex(s, freq = "Q").to_timestamp("Q")


def make_macro_panel_for_domain(
    macro_raw: pd.DataFrame,             
    country_by_ticker: pd.Series,         
    target_country: str = "United States", 
    macro_cols: list[str] = MACRO_COLS,
) -> pd.DataFrame:
    """
    Build a country-specific quarterly macroeconomic panel with a uniform
    quarter-end index and selected macro variables.

    Data model
    ----------
    The input `macro_raw` is expected to be a flat table with columns including:
      • 'ticker' identifying the security/series source,
      • 'year' containing human-readable quarter labels (e.g., '2019Q4'),
      • a superset of macro variables in `macro_cols`.

    The series `country_by_ticker` maps tickers to country names. Rows of
    `macro_raw` are filtered to those whose ticker maps to `target_country`.
    When multiple rows exist in the same quarter for the same variable, the
    first value is taken.

    Transformation
    --------------
    1) Parse 'year' into quarter-end timestamps.
  
    2) Filter to `macro_cols` and group by quarter, taking the first
       observation within the quarter for each variable.
  
    3) Coerce values to numeric and drop any row with missing values across
       the selected variables.

    Parameters
    ----------
    macro_raw : pandas.DataFrame
        Raw macro table with at least columns {'ticker','year'} plus the
        chosen macro variables.
    country_by_ticker : pandas.Series
        Mapping from ticker to country name.
    target_country : str, default "United States"
        Country to select.
    macro_cols : list of str, default MACRO_COLS
        Macro variable names to include in the output panel.

    Returns
    -------
    pandas.DataFrame
        A clean quarterly panel indexed by quarter-end timestamp, with columns
        `macro_cols` and no missing values.

    Raises
    ------
    ValueError
        If required columns are missing, if the ticker→country mapping yields
        no rows for `target_country`, or if quarter strings are malformed.
    """
    
    df = macro_raw.copy()

    if "ticker" not in df.columns or "year" not in df.columns:
       
        raise ValueError("macro_raw must have 'ticker' and 'year' columns (use .reset_index() on your MultiIndex).")

    tick2ctry = country_by_ticker.dropna().to_dict()
  
    df["country"] = df["ticker"].map(tick2ctry)
  
    df = df[df["country"] == target_country]

    if df.empty:
       
        raise ValueError(f"No rows found for country '{target_country}'. Check your ticker→country map.")

    qe = _parse_quarter_col(
        s = df["year"]
    )
    
    df = df.assign(qe = qe)

    keep = ["qe"] + [c for c in macro_cols if c in df.columns]

    df = df[keep].copy()

    panel = df.groupby("qe")[macro_cols].first().sort_index()

    panel = panel.apply(pd.to_numeric, errors = "coerce").dropna(how = "any")

    return panel


def main():
    """
    End-to-end pipeline to produce valuation, risk, and return forecasts by
    combining Black–Litterman index expectations, macroeconomic simulations,
    factor-based equity return modelling, and fundamental relative valuation.

    Processing outline
    ------------------
    1) Data acquisition:
      
       • Load macroeconomic histories (`MacroData`), index return histories and
         levels (`RatioData`), analyst fundamentals and KPI forecasts
         (`FinancialForecastData`), and auxiliary configuration (`config`).

    2) Quarterly index views:
      
       • Construct future quarterly index returns from predicted index levels:
         for each quarter q, r_q = level_q / level_{q−1} − 1. Label columns
         "Q1","Q2",….
      
       • Run Black–Litterman per quarter via `run_black_litterman_on_indexes`.
         Inputs include quarterly prior means π_q derived from annual priors
         (π_q = (1 + π_ann)^(1/4) − 1), and prior covariance Σ_q = 13 · Σ_w
         from weekly history. The function returns posterior means μ_post^(q)
         per index and posterior covariance Σ_post^(q).

    3) Cost of equity and CAPM:
      
       • Compute CAPM-based annual predictions using both BL-adjusted market
         returns and historical market returns as references (`capm_model`),
         with market volatility estimated from weekly series. The cost of
         equity per ticker is computed in `calculate_cost_of_equity`, blending
         currency risk with `blend_fx_returns`.

    4) Macro simulation (BVAR):
       
       • Fit a Minnesota-prior Bayesian VAR to the selected macro panel
         (`fit_bvar_minnesota`), selecting hyperparameters via expanding-window
         cross-validation on one-step RMSE. Simulate H quarters of macro
         scenarios using Student–t or bootstrap shocks (`simulate_bvar`).

    5) Factor simulation with macro conditioning:
      
       • Align quarterly Fama–French factor data with the macro panel and fit a
         VARX model (`factor_sim_varx`) for 5- and 3-factor sets. Simulate
         factor paths conditional on macro scenarios.

    6) Inject BL market beliefs into factor simulations:
       
       • Enforce quarter-specific BL market moments on the simulated market
         factor using `apply_bl_market_to_factor_sims` with mode "covmap".
         This shifts the market factor’s scenario mean to μ_BL^(q) − r_f and
         rescales its variance to Var_BL^(q) while preserving the empirical
         correlation structure across factors within each quarter.

    7) Equity return forecasting via factor model:
      
       • For each ticker, estimate factor loadings by OLS with Newey–West HAC
         covariance, then simulate one-year outcomes by applying the simulated
         factor paths and idiosyncratic innovations (`ff_pred_mc`). Annual
         returns are formed by compounding quarterly total returns:
          
           R_path = exp( Σ_q log(1 + r_q) ) − 1.

    8) Fundamental valuation overlays:
      
       • Produce price targets and return/volatility summaries using pricing
         models conditioned on forecast fundamentals and comparables:
           price-to-sales (`price_to_sales_price_pred`),
           enterprise-value-to-sales (`ev_to_sales_price_pred`),
           price-to-earnings (`pe_price_pred`),
           price-to-book (`price_to_book_pred`),
           Graham heuristic (`graham_number`),
           relative valuation blend (`rel_val_model`).

    9) Output:
      
       • Aggregate results into DataFrames and export to Excel workbooks via
         `export_results`. Diagnostic prints include checks that the enforced
         market factor moments match the BL targets per quarter.

    Side effects
    ------------
    Prints intermediate tables and diagnostics to stdout, and writes result
    sheets to the output Excel file specified in `config.REL_VAL_FILE` as
    well as a default export for selected additional sheets.

    Notes
    -----
    • Risk-free inputs appear in decimal form; quarterly risk-free r_f is
      provided by `config.RF_PER_QUARTER`.
    • The pipeline assumes factor simulations already reflect macro dynamics;
      the BL overlay aligns the market factor’s marginal moments to external
      beliefs while maintaining intra-quarter factor dependence.
    """
       
    s5 = np.sqrt(5)
   
    s52 = np.sqrt(52)

    logger.info("Importing data…")
   
    macro = MacroData()
   
    r = RatioData()
   
    crp = r.crp()
   
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
    
    tickers = config.tickers
    
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
    
    print("CAPM (BL-adjusted) Predictions:\n", capm_bl_pred_df)
   
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
    
    mkt_ticker = '^GSPC'  
    
    spx_ret = bl_df.loc[mkt_ticker]
    
    bl_mu_market_by_q = bl_df.loc[mkt_ticker, [c for c in bl_df.columns if c.upper().startswith('Q')]]  
   
    bl_var_market_by_q = {
        q: float(sigma.loc[mkt_ticker, mkt_ticker])
        for q, sigma in bl_cov.items()
        if (mkt_ticker in sigma.index) and (mkt_ticker in sigma.columns)
    }

    index_close = r.index_close[mkt_ticker].sort_index()
    
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
    
    macro_raw = macro.assign_macro_history_large_non_pct().reset_index()
    
    macro_q = make_macro_panel_for_domain(
        macro_raw = macro_raw,
        country_by_ticker = country,
        target_country = "United States",   
    )

    bvar = fit_bvar_minnesota(
        macro_df = macro_q,
        innovation="bootstrap"
    )
    macro_sims = simulate_bvar(
        model = bvar, 
        n_sims=10000, 
        horizon = 4,
        seed = 42, 
        innovation = 'bootstrap'
    )
    
    ff5_m, ff3_m, ff5_q, ff3_q = load_factor_data()

    factor_q_5 = ff5_q[["mkt_excess", "smb", "hml", "rmw", "cma", "rf"]].dropna()
    
    macro_hist = macro_q.reindex(factor_q_5.index).dropna()
    
    factor_q_5 = factor_q_5.loc[macro_hist.index]
    
    factor_q_3 = ff3_q[["mkt_excess", "smb", "hml", "rf"]].dropna()
    
    macro_hist = macro_q.reindex(factor_q_3.index).dropna()
    
    factor_q_3 = factor_q_3.loc[macro_hist.index]
    
    cov5, E5_q, sims_5 = factor_sim_varx(
        factor_data = factor_q_5,
        num_factors = 5,
        macro_hist = macro_hist,
        macro_sims = macro_sims,
        max_lag = 4,
        seed = 123,
        innovation = "bootstrap",
        df_t = 7.0,
    )

    sims_5, cov5_adj, E5_q_adj = apply_bl_market_to_factor_sims(
        sims_q = sims_5,
        bl_mu_market_by_q = bl_mu_market_by_q,
        bl_var_market_by_q = bl_var_market_by_q,
        rf_per_quarter = config.RF_PER_QUARTER,
        market_factor_name = "mkt_excess",
        mode = "covmap",          
        dist = "gaussian",          
        df_t = 7.0,
        seed = 12345
    )

    for q in sims_5.keys():
       
        m = sims_5[q].loc["mkt_excess"].values
       
        print(f"[BL override] {q} mean={np.mean(m):.6f}, var={np.var(m, ddof=1):.6f}")

    ff5_results = ff_pred_mc(
        tickers = tickers,
        factor_data = factor_q_5,
        returns_quarterly = r.quarterly_rets,
        sims = sims_5,                 
        rf_per_quarter = config.RF_PER_QUARTER,
        idio_df = 7.0,
        sample_betas = True,
        beta_lags_hac = 4,
        idio_innovation = "bootstrap"
    )

    cov3, E3_q, sims_3 = factor_sim_varx(
        factor_data = factor_q_3,
        num_factors = 3,
        macro_hist = macro_hist,
        macro_sims = macro_sims,
        max_lag = 4,
        seed = 123,
        innovation = "bootstrap",
        df_t = 7.0,
    )

    sims_3, cov3_adj, E3_q_adj = apply_bl_market_to_factor_sims(
        sims_q = sims_3,
        bl_mu_market_by_q = bl_mu_market_by_q,
        bl_var_market_by_q = bl_var_market_by_q,
        rf_per_quarter = config.RF_PER_QUARTER,
        market_factor_name = "mkt_excess",
        mode = "covmap",
        dist = "gaussian",
        df_t = 7.0,
        seed = 67890
    )

    ff3_results = ff_pred_mc(
        tickers = tickers,
        factor_data = factor_q_3,
        returns_quarterly = r.quarterly_rets,
        sims = sims_3,                
        rf_per_quarter = config.RF_PER_QUARTER,
        idio_df = 7.0,
        sample_betas = True,
        beta_lags_hac = 4,
        idio_innovation = "bootstrap"
    )
    
    exp_factor_results = exp_fac_reg()

    low_rev_y = temp_analyst['Low Revenue Estimate']
   
    avg_rev_y = temp_analyst['Avg Revenue Estimate']
   
    high_rev_y = temp_analyst['High Revenue Estimate']

    low_eps_y = temp_analyst['Low EPS Estimate']
   
    avg_eps_y = temp_analyst['Avg EPS Estimate']
   
    high_eps_y = temp_analyst['High EPS Estimate']

    ps = temp_analyst['priceToSalesTrailing12Months']
   
    cpe = temp_analyst['priceEpsCurrentYear']
   
    tpe = temp_analyst['trailingPE']
    
    evts = temp_analyst['enterpriseToRevenue']
 
    shares_out = temp_analyst['sharesOutstanding']
    
    dps = temp_analyst['lastDividendValue']
    
    ptb_y = temp_analyst['priceToBook']

    results = r.dicts()

    pe_pred_list, evs_pred_list, ps_pred_list, pbv_pred_list, graham_pred_list, rel_val_list = ([] for _ in range(6))

    for ticker in tickers:
        
        if ticker not in config.TICKER_EXEMPTIONS:
        
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
        
        else:
            
            pe_pred_list.append({
                "Ticker": ticker,
                "Current Price": latest_prices.get(ticker, 0),
                "Low Price": 0,
                "Avg Price": 0,
                "High Price": 0,
                "Returns": 0,
                "Volatility": 0,
                "Avg PE": 0
            })
            
            evs_pred_list.append({
                "Ticker": ticker,
                "Current Price": latest_prices.get(ticker, 0),
                "Low Price": 0,
                "Avg Price": 0,
                "High Price": 0,
                "Returns": 0,
                "Volatility": 0,
                "Avg EVS": 0
            })
            
            ps_pred_list.append({
                "Ticker": ticker,
                "Current Price": latest_prices.get(ticker, 0),
                "Low Price": 0,
                "Avg Price": 0,
                "High Price": 0,
                "Returns": 0,
                "Volatility": 0,
                "Avg PS": 0
            })
            
            pbv_pred_list.append({
                "Ticker": ticker,
                "Current Price": latest_prices.get(ticker, 0),
                "Low Price": 0,
                "Avg Price": 0,
                "High Price": 0,
                "Returns": 0,
                "Volatility": 0,
                "Avg PBV": 0
            })
            
            graham_pred_list.append({
                "Ticker": ticker,
                "Current Price": latest_prices.get(ticker, 0),
                "Low Price": 0,
                "Avg Price": 0,
                "High Price": 0,
                "Returns": 0,
                "Volatility": 0
            })
            
            rel_val_list.append({
                "Ticker": ticker,
                "Current Price": latest_prices.get(ticker, 0),
                "Low Price": 0,
                "Avg Price": 0,
                "High Price": 0,
                "Returns": 0,
                "SE" : np.nan
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
        'FF5 Pred': ff5_results,
        'Factor Exponential Regression': exp_factor_results,
    }
    
    export_results(
        sheets = sheets_to_write
    )


if __name__ == "__main__":
    
    main()
