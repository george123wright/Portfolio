import numpy as np
import pandas as pd
import statsmodels.api as sm
from factor_simulations import _circular_block_bootstrap


def _standardise_factor_cols(
    f
):
    """
    Standardise Fama–French column names to the internal convention.

    Parameters
    ----------
    f : pandas.DataFrame
        DataFrame containing Fama–French factor columns. Any subset of the
        following original names may be present: {"Mkt-RF","SMB","HML","RMW",
        "CMA","RF"}.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with columns renamed (when present) as:
        "Mkt-RF" -> "mkt_excess", "SMB" -> "smb", "HML" -> "hml",
        "RMW" -> "rmw", "CMA" -> "cma", "RF" -> "rf". Columns not in the map
        are left unchanged.

    Notes
    -----
    The Fama–French convention reports the market factor as excess market
    return "Mkt-RF" and the risk-free rate "RF" in decimal units. The internal
    code expects the factor names {"mkt_excess","smb","hml","rmw","cma"} and a
    separate risk-free series "rf". This helper performs only a rename; it
    does not alter values or frequencies.
    """
    
    mapper = {
        'Mkt-RF': 'mkt_excess',
        'SMB': 'smb',
        'HML': 'hml',
        'RMW': 'rmw',
        'CMA': 'cma',
        'RF': 'rf'
    }

    return f.rename(columns = {k:v for k,v in mapper.items() if k in f.columns})


def _to_qe_ts(
    idx
):
    """
    Convert a date-like index to quarter-end timestamps.

    Parameters
    ----------
    idx : pandas.Index or sequence
        An index or sequence that can be parsed to dates or quarter periods.
        Examples include datetime-like indices, period indices with quarterly
        frequency, or strings such as "2019Q3".

    Returns
    -------
    pandas.DatetimeIndex
        A quarter-end timestamp index, computed as:
        1) try: PeriodIndex(idx, freq="Q").to_timestamp("Q");
        2) except: to_datetime(idx).to_period("Q").to_timestamp("Q").

    Notes
    -----
    The conversion collapses any intra-quarter information to quarter-end.
    All further alignment in this module is performed at quarterly granularity.
    """
    
    try:

        return pd.PeriodIndex(idx, freq = 'Q').to_timestamp('Q')

    except Exception:

        ts = pd.to_datetime(idx)

        return ts.to_period('Q').to_timestamp('Q')


def _align_quarterly(
    factors: pd.DataFrame, 
    rets: pd.DataFrame
):
    """
    Align factor and asset return series to a common quarter-end index.

    Parameters
    ----------
    factors : pandas.DataFrame
        Factor time series with an index convertible to quarter-end (see
        `_to_qe_ts`). Columns are expected to include Fama–French factors and
        the risk-free rate "rf" after standardisation.
    rets : pandas.DataFrame
        Asset return time series with an index convertible to quarter-end.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Tuple (F, R) of the two inputs with indices converted to quarter-end
        timestamps and sorted by time. No intersection is taken here; that
        occurs downstream when joining per ticker.

    Notes
    -----
    This function performs only index transformation and sorting. Downstream
    routines will join on the intersection of quarter-end dates per ticker.
    """
    
    F = factors.copy()
    
    R = rets.copy()

    F.index = _to_qe_ts(
        idx = F.index
    )
    
    R.index = _to_qe_ts(
        idx = R.index
    )

    return F.sort_index(), R.sort_index()


def _nw_cov(
    X: pd.DataFrame, 
    y: pd.Series, 
    lags: int = 1
):
    """
    Estimate OLS coefficients with a Newey–West (HAC) covariance estimator.

    Model
    -----
    The regression is
   
        y_t = α + x_t' β + ε_t,
        
    where y_t is a scalar dependent variable, x_t is a k×1 regressor vector,
    α is an intercept and β is a k×1 coefficient vector. Let X_c be the design
    matrix with a leading column of ones.

    Estimation
    ----------
    Ordinary least squares yields
   
        β̂_OLS = argmin_b Σ_t (y_t − α − x_t' b)^2,
   
    implemented via `statsmodels.OLS(y, add_constant(X))`.

    The estimator of Var(β̂) is Newey–West HAC with maximum lag L = lags,
    accounting for heteroskedasticity and autocorrelation in ε_t. Let û_t be
    residuals and S_0 = Σ_t û_t^2 x_t x_t', and for ℓ ≥ 1,
   
        S_ℓ = Σ_{t=ℓ+1} û_t û_{t−ℓ} x_t x_{t−ℓ}'.
   
    The HAC long-run covariance uses Bartlett weights w_ℓ = 1 − ℓ/(L+1):
   
        Ω̂_HAC = S_0 + Σ_{ℓ=1}^L w_ℓ (S_ℓ + S_ℓ').
   
    Then
   
        Var̂(β̂) = (X_c' X_c)^{-1} (X_c' Ω̂_HAC X_c) (X_c' X_c)^{-1}.

    Returns
    -------
    beta : pandas.Series
        The estimated β̂ (excluding the intercept), indexed by X’s columns.
    Vbeta : pandas.DataFrame
        The HAC covariance matrix Var̂(β̂), aligned to `beta.index`.
    sigma2 : float
        The OLS residual variance, equal to `res.scale` from statsmodels, i.e.
        SSR / (T − (k+1)), where k is the number of regressors excluding the
        intercept.
    resid : numpy.ndarray
        The residual vector ε̂_t.

    Parameters
    ----------
    X : pandas.DataFrame
        Regressor matrix (no constant column; it is added internally).
    y : pandas.Series
        Dependent variable.
    lags : int, default 1
        Newey–West maximum lag L for the HAC estimator.

    Notes
    -----
    The returned `Vbeta` is used to quantify coefficient uncertainty in the
    Monte-Carlo routine; `sigma2` quantifies idiosyncratic (unexplained) return
    variance conditional on factors.
    """
    
    Xc = sm.add_constant(X)

    res = sm.OLS(y, Xc, hasconst = True).fit(cov_type = "HAC", cov_kwds = {"maxlags": lags})

    beta = res.params.drop("const")

    Vbeta = res.cov_params().loc[beta.index, beta.index]

    sigma2 = max(float(res.scale), 0.0)

    resid = res.resid.to_numpy()

    return beta, Vbeta, sigma2, resid


def ff_pred_mc(
    tickers: list[str],
    factor_data: pd.DataFrame,
    returns_quarterly: pd.DataFrame,
    sims: dict[str, pd.DataFrame],
    rf_per_quarter: float,
    *,
    sample_betas: bool = False,        
    beta_var_scale: float = 0.25,     
    beta_lags_hac: int = 4,         
    idio_innovation: str | None = "gaussian",  
    idio_df: float = 20.0,         
    block_len: int = 4,            
    min_obs_fallback: int = 20,         
    extra_min_cushion: int = 10,        
    clip_min_qret: float = -0.5,       
    clip_max_qret: float = 0.5,
    winsorize_pct: float = 0.0,         
) -> pd.DataFrame:
    """
    Monte-Carlo forecast of annual equity returns using Fama–French factor
    exposures and simulated factor paths.

    Overview
    --------
    For each ticker, the procedure:
   
    1) aligns factors and the asset’s quarterly returns, computes excess returns,
   
    2) estimates factor loadings β by OLS with Newey–West (HAC) covariance,
   
    3) draws (optionally) coefficient vectors β* to reflect estimation
       uncertainty, and
   
    4) simulates quarterly excess returns using factor scenarios and an
       idiosyncratic innovation, cumulating to an annual path return via log
       compounding.

    Data alignment
    --------------
    Let factors be quarterly series including "mkt_excess" and a subset of
    {"smb","hml","rmw","cma"}, plus the risk-free rate "rf". Let asset returns
    be quarterly total returns. After alignment, excess returns are computed as
        r_excess,t = r_total,t − rf_t.

    Regression model (per ticker)
    -----------------------------
    The factor model is
   
        r_excess,t = x_t' β + ε_t,
   
    where x_t is the k×1 vector of contemporaneous factors (k ≥ 1), β is a
    k×1 exposure vector and ε_t is an idiosyncratic shock. OLS provides β̂ and
    the residual variance σ̂²_ε. HAC with `beta_lags_hac` lags is used to form
    Var̂(β̂) = V̂_β. Sufficient data are required: at least
    `len(cols) + extra_min_cushion` quarters, where cols includes "mkt_excess"
    and any available style factors.

    Coefficient uncertainty (optional)
    ----------------------------------
    If `sample_betas=True`, coefficient draws are taken as
   
        β* ~ N(β̂,  Σ̂_β),
   
    where Σ̂_β = beta_var_scale · V̂_β (a scalar shrink applied to the HAC
    covariance). If `sample_betas=False`, β* = β̂ deterministically.

    Idiosyncratic shock simulation
    ------------------------------
    Given residual variance σ̂²_ε, the per-quarter idiosyncratic shock ε_q is
    simulated via one of:
   
      • Gaussian:      ε_q ~ N(0, σ̂²_ε).
   
      • Student–t:     ε_q = z_q / sqrt(g_q) with z_q ~ N(0, σ̂²_ε) and
                       g_q ~ χ²_ν / ν; here ν = `idio_df`.
   
      • Bootstrap:     draw a circular block bootstrap path from the centred and
                       variance-matched residuals using block length `block_len`.
   
    Shocks are drawn independently across quarters within each scenario path.

    Quarterly return generation
    ---------------------------
    Let F_q be the simulated factor vector for quarter q from `sims[q]`
    (dimension k), arranged to match the order of `cols`. For a coefficient
    draw β*, the model implies the quarterly excess return
   
        r_ex,q = F_q' β* + ε_q.
   
    The quarterly total return adds the risk-free leg:
   
        r_q = rf_per_quarter + r_ex,q.
   
    Per-quarter winsorisation (if `winsorize_pct>0`) and clipping to
    [`clip_min_qret`, `clip_max_qret`] are applied to stabilise tails.

    Annual compounding per simulation path
    --------------------------------------
    Returns are compounded via log-aggregation:
   
        G = Σ_{q=1}^Q log(1 + r_q),        (Q = number of simulated quarters)
   
        R_path = exp(G) − 1.
   
    The distribution of R_path across simulation paths yields the scenario
    statistics.

    Outputs
    -------
    A DataFrame indexed by ticker with columns:
      • "Returns": mean(R_path),
      • "SE": standard deviation of R_path across paths (dispersion measure),
      • "p05","p25","p50","p75","p95": percentiles of R_path.
    If no sufficient data exist for a ticker, a row of zeros/NaNs is emitted.

    Parameters
    ----------
    tickers : list of str
        Tickers to forecast.
    factor_data : pandas.DataFrame
        Fama–French factors and "RF", at least quarterly, before standardisation.
    returns_quarterly : pandas.DataFrame
        Quarterly total returns per ticker (in decimal).
    sims : dict[str, pandas.DataFrame]
        Factor simulation set keyed by quarter labels (e.g., "Q1", "Q2", ...).
        Each DataFrame is (factors × n_sims), with rows containing at least
        "mkt_excess" and any of {"smb","hml","rmw","cma"} used for estimation,
        and columns indexing simulation paths.
    rf_per_quarter : float
        Risk-free rate per quarter (decimal), added back to excess returns.
    sample_betas : bool, default False
        If True, draw β* ~ N(β̂, beta_var_scale · V̂_β) per simulation path.
    beta_var_scale : float, default 0.25
        Scalar shrink applied to the HAC covariance V̂_β when sampling β*.
    beta_lags_hac : int, default 4
        Newey–West maximum lag used in HAC covariance for β̂.
    idio_innovation : {"gaussian","student_t","bootstrap"} or None, default "gaussian"
        Idiosyncratic shock distribution.
    idio_df : float, default 20.0
        Degrees of freedom for Student–t idiosyncratic shocks.
    block_len : int, default 4
        Block length for circular block bootstrap of idiosyncratic shocks.
    min_obs_fallback : int, default 20
        Minimum number of quarters desired in the most recent 5-year window;
        if not met, the full sample is used instead.
    extra_min_cushion : int, default 10
        Additional cushion beyond the number of regressors required to run the
        regression (stability filter).
    clip_min_qret, clip_max_qret : float, defaults −0.5 and 0.5
        Bounds applied to each quarterly total return to control extreme tails.
    winsorize_pct : float, default 0.0
        If > 0, per-quarter winsorisation at the given lower/upper tail mass
        (e.g., 0.01 implies 1%/99%).

    Notes
    -----
    • The factor simulations determine the conditional distribution of
      systematic returns, whilst the idiosyncratic component is modelled
      independently.
    • The procedure forecasts one-year returns by compounding Q quarters of
      simulated quarterly returns (Q equals the number of keys in `sims`).
    • Reported dispersion "SE" is the standard deviation across simulated
      annual outcomes; it is not divided by sqrt(n_sims).
    """
    
    f = _standardise_factor_cols(
        f = factor_data
    )
   
    f, rq = _align_quarterly(
        factors = f, 
        rets = returns_quarterly
    )

    non_mkt = [c for c in ["smb","hml","rmw","cma"] if c in f.columns]
   
    cols = ["mkt_excess"] + non_mkt
   
    assert "rf" in f.columns, "factor_data must include 'rf' (decimals)."

    quarters = list(sims.keys())
   
    n_q = len(quarters)
   
    n_sims = sims[quarters[0]].shape[1]

    out = []

    for tk in tickers:
   
        df = (
            f[cols + ["rf"]]
            .join(rq[[tk]], how = "inner")
            .rename(columns = {tk: "ret"})
            .dropna()
        )
   
        if df.empty:
   
            out.append({
                "Ticker": tk, 
                "Returns": 0,
                "SE": 0,
                "p05": 0,
                "p25": 0,
                "p50": 0, 
                "p75": 0, 
                "p95": 0
            })
   
            continue

        df["excess"] = df["ret"] - df["rf"]
   
        df5 = df.loc[df.index.max() - pd.DateOffset(years = 5):]
   
        use = df5 if len(df5) >= min_obs_fallback else df

        min_needed = len(cols) + extra_min_cushion

        if len(use) < min_needed:

            out.append({
                "Ticker": tk, 
                "Returns": 0,
                "SE": 0, 
                "p05": 0,
                "p25": 0,
                "p50": 0,
                "p75": 0,
                "p95": 0
            })

            continue

        try:

            lags = max(1, min(beta_lags_hac, len(use) // 4))

            beta_hat, Vbeta, sigma2_eps, resid_vec = _nw_cov(
                X = use[cols], 
                y = use["excess"], 
                lags = lags
            )

        except Exception:
           
            out.append({
                "Ticker": tk,
                "Returns": 0,
                "SE": 0, 
                "p05": 0, 
                "p25": 0,
                "p50": 0,
                "p75": 0,
                "p95": 0
            })
           
            continue

        rng = np.random.default_rng(abs(hash(tk)) % (2**32))

        if sample_betas:
           
            V = Vbeta.values * float(beta_var_scale) 
           
            try:
           
                Lb = np.linalg.cholesky(V + 1e-12 * np.eye(V.shape[0]))
           
            except np.linalg.LinAlgError:
           
                w, Q = np.linalg.eigh((V + V.T) / 2)
           
                w = np.maximum(w, 1e-12)
                
                Lb = Q @ np.diag(np.sqrt(w))
           
            betas_draw = rng.standard_normal((n_sims, len(cols))) @ Lb.T + beta_hat.values
       
        else:
       
            betas_draw = np.repeat(beta_hat.values[None, :], n_sims, axis = 0)

        mode = (idio_innovation or "gaussian").lower()
       
        if mode == "bootstrap":
       
            resid_centered = resid_vec - resid_vec.mean()
       
            emp_var = float(np.var(resid_centered, ddof = 1)) if len(resid_centered) > 1 else 0.0
       
            scale = (np.sqrt(sigma2_eps / emp_var) if emp_var > 1e-12 else 1.0)
       
            resid_scaled = resid_centered * scale
       
            eps = np.empty((n_q, n_sims))
       
            for j in range(n_sims):
       
                eps[:, j] = _circular_block_bootstrap(
                    resid = resid_scaled.reshape(-1, 1), 
                    T = n_q, 
                    L = block_len, 
                    rng = rng
                ).ravel()
       
        elif mode == "student_t":
       
            g = rng.chisquare(idio_df, size = (n_q, n_sims)) / idio_df
       
            eps = rng.standard_normal((n_q, n_sims)) * np.sqrt(sigma2_eps) / np.sqrt(g)
       
        else:  
       
            eps = rng.standard_normal((n_q, n_sims)) * np.sqrt(sigma2_eps)

        log_growth = np.zeros(n_sims)
       
        valid = np.ones(n_sims, dtype = bool)

        for qi, q in enumerate(quarters):

            fac_names = [c for c in cols if c in sims[q].index]

            Fq = sims[q].loc[fac_names].T.values  

            if Fq.shape[1] != len(cols):
            
                tmp = np.zeros((n_sims, len(cols)))
            
                for ci, c in enumerate(cols):
            
                    if c in fac_names:
            
                        tmp[:, ci] = sims[q].loc[c].values
            
                Fq = tmp

            r_ex_q = (Fq * betas_draw).sum(axis = 1) + eps[qi]
            
            r_q = rf_per_quarter + r_ex_q

            if winsorize_pct > 0:

                lo, hi = np.percentile(r_q, [100*winsorize_pct, 100 * (1 - winsorize_pct)])

                r_q = np.clip(r_q, lo, hi)

            r_q = np.clip(r_q, clip_min_qret, clip_max_qret)

            valid &= np.isfinite(r_q)

            log_growth += np.log1p(r_q)

        ann_paths = np.expm1(log_growth)

        if not np.any(valid):

            out.append({
                "Ticker": tk,
                "Returns": 0, 
                "SE": 0, 
                "p05": 0, 
                "p25": 0, 
                "p50": 0,
                "p75": 0,
                "p95": 0
            })

            continue

        vals = ann_paths[np.isfinite(ann_paths)]

        sd_paths = float(vals.std(ddof = 1)) if len(vals) > 1 else 0.0

        mean_ret = float(vals.mean())

        if len(vals) > 10:

            q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])

        else:

            q05 = q25 = q50 = q75 = q95 = float("nan")

        out.append({
            "Ticker": tk,
            "Returns": mean_ret,
            "SE": sd_paths,                   
            "p05": q05,
            "p25": q25,
            "p50": q50,
            "p75": q75,
            "p95": q95
        })

    return pd.DataFrame(out).set_index("Ticker")
