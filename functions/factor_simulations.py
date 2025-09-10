from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from numpy.linalg import eigvals


def _companion_spectral_radius(
    A: np.ndarray
) -> float:
    """
    Compute the spectral radius of the companion matrix associated with a VAR(p).

    Parameters
    ----------
    A : np.ndarray
        Array of lag coefficient matrices with shape (p, k, k), where p is the
        VAR order and k the number of variables. A[ℓ-1] corresponds to A_ℓ.

    Returns
    -------
    float
        The spectral radius ρ(C) = max_i |λ_i(C)| of the companion matrix C.

    Notes
    -----
    The VAR(p) process for y_t ∈ ℝ^k,

        y_t = ∑_{ℓ=1}^p A_ℓ y_{t-ℓ} + u_t,

    can be written in companion form as

        Y_t = C Y_{t-1} + U_t,

    where 
        
        Y_t = [y_t', y_{t-1}', …, y_{t-p+1}']' ∈ ℝ^{k p}, 
        
    and the companion matrix C ∈ ℝ^{k p × k p} has block structure

        C = [ A_1  A_2  …  A_{p-1}  A_p ]
            [  I    0   …     0      0  ]
            [  0    I   …     0      0  ]
            [  ⋮    ⋮   ⋱     ⋮      ⋮  ]
            [  0    0   …     I      0  ],

    with I the k×k identity and 0 a k×k zero block. The VAR is stable (second-order
    stationary) if and only if ρ(C) < 1. This routine builds C and returns ρ(C),
    which is used to enforce stability via coefficient shrinkage elsewhere.
    """

    p, k, _ = A.shape
   
    top = np.concatenate(A, axis = 1)  
   
    if p == 1:
   
        C = top
   
    else:
   
        I = np.eye(k * (p - 1))
   
        Z = np.zeros((k * (p - 1), k))
    
        bottom = np.concatenate([I, Z], axis = 1)
    
        C = np.vstack([top, bottom])
   
    return float(np.max(np.abs(eigvals(C))))


def _shrink_to_stability(
    A: np.ndarray, 
    max_radius: float = 0.98
) -> np.ndarray:
    """
    Convexly shrink VAR(p) lag coefficients towards zero until the companion spectral radius is below a target.

    Parameters
    ----------
    A : np.ndarray
        Array of lag coefficient matrices with shape (p, k, k).
    max_radius : float, default 0.98
        Stability target for the companion spectral radius. Values below 1 enforce
        a margin of stability.

    Returns
    -------
    np.ndarray
        Shrunk coefficient array Ã = λ A with 0 < λ ≤ 1 such that ρ(C(Ã)) ≤ max_radius,
        or the final λ after a finite number of back-offs if the target is not reached.

    Rationale
    ---------
    For any scalar λ > 0, the spectral radius scales as ρ(λC) = |λ| ρ(C). Shrinking
    A ↦ λ A consequently shrinks the spectral radius linearly in λ. This routine
    decreases λ multiplicatively (by 0.9) until ρ(C(λA)) ≤ max_radius or the iteration
    limit is reached. Shrinkage preserves the relative structure of lags and offers
    a simple stabilisation for borderline or mildly explosive VAR estimates.
    """
   
    if A.size == 0:
   
        return A
   
    lam = 1.0
   
    for _ in range(50):
   
        r = _companion_spectral_radius(
            A = lam * A
        )
   
        if r <= max_radius:
   
            return lam * A
   
        lam *= 0.9 
   
    return lam * A


def _circular_block_bootstrap(
    resid: np.ndarray, 
    T: int, 
    L: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a residual path via circular block bootstrap.

    Parameters
    ----------
    resid : np.ndarray
        Centred residual matrix of shape (T_resid, k), where rows index time and columns index variables.
    T : int
        Desired length of the bootstrapped path.
    L : int
        Block length. Larger L preserves longer-range dependence; smaller L increases mixing.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (T, k) formed by concatenating circular blocks of length L sampled
        from `resid`. Wrap-around indexing ensures all starting positions are valid.

    Notes
    -----
    Circular block bootstrapping preserves short-run temporal dependence and contemporaneous
    cross-sectional correlation across series. It is well-suited to re-using VAR residuals
    when simulating alternative realisations under weak dependence.
    """
    
    T_resid, k = resid.shape
   
    n_blocks = int(np.ceil(T / L))
   
    starts = rng.integers(0, T_resid, size=n_blocks)
   
    pieces = []
   
    for s in starts:
   
        idx = (np.arange(L) + s) % T_resid
   
        pieces.append(resid[idx])
   
    out = np.vstack(pieces)[:T]
   
    return out


def _student_t_innovations(
    T: int,
    k: int, 
    chol: np.ndarray,
    df: float, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Draw multivariate Student–t innovations using a Gaussian scale mixture.

    Parameters
    ----------
    T : int
        Number of time periods to draw.
    k : int
        Dimension of the innovation vector at each t.
    chol : np.ndarray
        Upper (or lower) triangular Cholesky factor of the target covariance Σ (k×k).
    df : float
        Degrees of freedom ν > 0. Finite variance requires ν > 2.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Matrix of shape (T, k) with rows u_t constructed as

            u_t = (z_t / sqrt(g_t)) chol^T,

        where z_t ~ N(0, I_k) and g_t ~ χ²_ν / ν are independent across t.

    Important Property
    ------------------
    The scale-mixture construction yields

        E[u_t] = 0,     
        
        Cov(u_t) = (ν / (ν − 2)) Σ  for ν > 2.

    Hence the marginal covariance of the returned t-innovations is larger than Σ by
    a factor ν / (ν − 2). This deliberate inflation introduces heavy tails relative
    to Gaussian shocks with covariance Σ and is often used in stressable simulations.
    """
   
    z = rng.standard_normal((T, k))
   
    g = rng.chisquare(df, size = T) / df
   
    scaled = z / np.sqrt(g)[:, None]
   
    return scaled @ chol.T  


def factor_sim(
    factor_data: pd.DataFrame,
    num_factors: int,
    n_sims: int = 1_000,
    horizon: int = 4,
    max_lag: int = 4,
    seed: int | None = 42,
    include_market: bool = True,
    innovation: str = "student_t",         
    df_t: float = 7.0,                         
    block_len: int = 8,                          
    exog: pd.DataFrame | None = None,          
    lag_criterion: str = "bic",            
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Fit a reduced-form VAR(p) to selected Fama–French factors and simulate forward paths with heavy-tailed or bootstrapped shocks.

    Model
    -----
    Let f_t ∈ ℝ^k denote the vector of chosen factors (e.g., mkt_excess, SMB, HML, …).
    The VAR(p) with intercept is

        f_t = c + ∑_{ℓ=1}^p A_ℓ f_{t−ℓ} + u_t,      u_t ∼ (0, Σ),

    estimated equation-by-equation via OLS using `statsmodels.VAR`. A stability
    adjustment is applied via `_shrink_to_stability(A)` so that the companion
    spectral radius is below a target, reducing explosive dynamics.

    Lag Selection
    -------------
    The lag order p is selected from {1, …, max_lag} using `VAR.select_order` and
    the criterion indicated by `lag_criterion` ("bic", "aic", or "hqic"). If the
    chosen criterion is unavailable, p defaults conservatively to 1.

    Innovations
    -----------
    • Gaussian:
        u_t = z_t L', with z_t ~ N(0, I_k) and Σ = L L'.

    • Student–t:
        u_t = (z_t / sqrt(g_t)) L', with z_t ~ N(0, I_k), g_t ~ χ²_ν / ν, independent.
        The marginal covariance inflates to (ν/(ν−2)) Σ for ν > 2.

    • Bootstrap:
        u_t is generated by circular block bootstrapping of centred residuals, with
        block length `block_len`. This preserves short-run dependence.

    Parameters
    ----------
    factor_data : pd.DataFrame
        Historical factor returns. Must contain the required factor columns and a
        time index. Missing rows for selected factors are dropped.
    num_factors : int
        Either 3 or 5. If `include_market` is True, the set is
        {mkt_excess, SMB, HML} or {mkt_excess, SMB, HML, RMW, CMA}; otherwise
        {SMB, HML} or {SMB, HML, RMW, CMA}.
    n_sims : int, default 1_000
        Number of Monte Carlo paths.
    horizon : int, default 4
        Number of forward steps (e.g., quarters).
    max_lag : int, default 4
        Maximum VAR order considered in selection.
    seed : int or None, default 42
        Random seed for reproducibility.
    include_market : bool, default True
        Whether to include the market excess factor among the endogenous variables.
    innovation : {"gaussian","student_t","bootstrap"}, default "student_t"
        Shock generator for u_t.
    df_t : float, default 7.0
        Degrees of freedom for Student–t innovations.
    block_len : int, default 8
        Block size for the residual bootstrap (tune for sampling frequency).
    exog : pd.DataFrame or None, default None
        Reserved for future use (exogenous regressors are not utilised in this function).
    lag_criterion : {"bic","aic","hqic"}, default "bic"
        Information criterion used for lag selection.

    Returns
    -------
    cov_q : dict[str, pd.DataFrame]
        Dictionary mapping "Q1", …, f"Q{horizon}" to k×k covariance matrices of
        simulated factor realisations at each step.
    mean_q : pd.DataFrame
        DataFrame of shape (horizon × k) with the simulation mean per step.
    sims_q : dict[str, pd.DataFrame]
        Dictionary mapping "Q1", …, to k×n_sims matrices of simulated paths.

    Notes
    -----
    • The initial state uses the last p observed rows (“history”).
    • Under Student–t innovations, the unconditional variance is inflated by ν/(ν−2)
      relative to Σ; this is intentional to introduce heavy tails.
    • The residual bootstrap should be applied to mean-centred residuals to avoid
      reintroducing a non-zero mean in u_t.
    """
  
    if num_factors == 5:
  
        base = ["mkt_excess", "smb", "hml", "rmw", "cma"] if include_market else ["smb", "hml", "rmw", "cma"]
  
    elif num_factors == 3:
  
        base = ["mkt_excess", "smb", "hml"] if include_market else ["smb", "hml"]
  
    else:
  
        raise ValueError("num_factors must be 3 or 5")

    fac_df = factor_data[base].dropna().copy()
  
    if exog is not None:
  
        exog = exog.loc[fac_df.index].astype(float)

    var_mod = VAR(fac_df)
  
    order_sel = var_mod.select_order(max_lag)
  
    crit = getattr(order_sel, lag_criterion) if getattr(order_sel, lag_criterion) is not None else 1
  
    p = int(max(1, crit))

    var_res = var_mod.fit(p, trend = "c")

    c = var_res.params.loc["const"].values  
  
    A = var_res.coefs.copy()               
  
    A = _shrink_to_stability(
        A = A
    )             

    Sigma_u = var_res.sigma_u              
   
    chol = np.linalg.cholesky(Sigma_u)
    
    resid = var_res.resid.values          

    history = fac_df.values[-p:]        
    
    k = len(base)

    rng = np.random.default_rng(seed)
    
    sims = np.empty((horizon, k, n_sims))

    if innovation == "gaussian":
    
        shocks = rng.standard_normal((n_sims, horizon, k)) @ chol.T
    
    elif innovation == "student_t":
    
        shocks = np.stack([_student_t_innovations(horizon, k, chol, df_t, rng) for _ in range(n_sims)], axis=0)
    
    elif innovation == "bootstrap":

        shocks = np.stack([_circular_block_bootstrap(
            resid = resid,
            T = horizon, 
            L = block_len,
            rng = rng
        ) for _ in range(n_sims)], axis = 0)
    
    else:
    
        raise ValueError("innovation must be 'gaussian', 'student_t', or 'bootstrap'")

    for j in range(n_sims):
      
        buf = history.copy()
      
        for t in range(horizon):
      
            x_t = c.copy()

            for lag in range(p):
      
                x_t += A[lag] @ buf[-lag - 1]
      
            x_t += shocks[j, t]
      
            sims[t, :, j] = x_t
      
            buf = np.vstack([buf, x_t])[-p:]

    idx_q = [f"Q{i}" for i in range(1, horizon + 1)]
    
    mean_q = pd.DataFrame(sims.mean(axis = 2), index = idx_q, columns = base)

    cov_q = {
        q: pd.DataFrame(np.cov(sims[i], rowvar = True), index = base, columns = base)
        for i, q in enumerate(idx_q)
    }

    sims_q = {
        q: pd.DataFrame(sims[i], index=base, columns=[f"{n+1}" for n in range(n_sims)])
        for i, q in enumerate(idx_q)
    }

    return cov_q, mean_q, sims_q



def _companion_radius(
    A: np.ndarray
) -> float:
    """
    Compute the spectral radius of the companion matrix for a VAR(p) (duplicate utility for VARX).

    Parameters
    ----------
    A : np.ndarray
        Array of lag coefficient matrices with shape (p, k, k).

    Returns
    -------
    float
        Spectral radius ρ(C) of the corresponding companion matrix C.

    Notes
    -----
    The construction of C follows the standard block companion form used to assess
    VAR stability. This helper mirrors `_companion_spectral_radius` and is used
    by the VARX simulation pipeline.
    """
        
    p, k, _ = A.shape
   
    top = np.concatenate(A, axis = 1)
   
    if p == 1:
   
        C  =  top
   
    else:
   
        I = np.eye(k * (p - 1))
   
        Z = np.zeros((k * (p - 1), k))
   
        C = np.vstack([top, np.concatenate([I, Z], axis = 1)])
   
    return float(np.max(np.abs(eigvals(C))))


def factor_sim_varx(
    factor_data: pd.DataFrame,          
    num_factors: int,
    macro_hist: pd.DataFrame,            
    macro_sims: dict[str, pd.DataFrame], 
    n_sims: int | None = None,           
    horizon: int | None = None,    
    max_lag: int = 4,
    seed: int | None = 42,
    innovation: str = "student_t",       
    df_t: float = 7.0,
    block_len: int = 4,                 
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Fit a VARX(p) of factors on macro exogenous variables and simulate factors conditional on macro scenarios.

    Model
    -----
    Let f_t ∈ ℝ^k denote the vector of factors and z_t ∈ ℝ^m the macro exogenous
    vector at time t. The VARX(p) with intercept is

        f_t = c + ∑_{ℓ=1}^p A_ℓ f_{t−ℓ} + B z_t + u_t,     u_t ∼ (0, Σ),

    where c ∈ ℝ^k, A_ℓ ∈ ℝ^{k×k}, and B ∈ ℝ^{k×m}. Estimation is performed with
    `statsmodels.VAR(endog=factors, exog=macro_hist)`, which fits the system via
    OLS equation-by-equation. The lag order p is selected using information
    criteria on the endogenous block; exogenous regressors do not affect lag
    selection in `statsmodels`.

    Stability
    ---------
    The companion spectral radius ρ(C) is computed for A. If ρ(C) > 0.99, the
    lag matrices are homothetically scaled as A ← (0.99 / ρ(C)) A to enforce a
    margin of stability prior to simulation.

    Innovations
    -----------
    • Gaussian:
       
        u_t = z_t L', z_t ~ N(0, I_k), Σ = L L'.

    • Student–t:
      
        u_t = (z_t / sqrt(g_t)) L', z_t ~ N(0, I_k), g_t ~ χ²_ν / ν. Marginal
        covariance inflates by ν/(ν−2) for ν > 2.

    • Bootstrap:
        u_t sampled via circular block bootstrap from centred residuals with block
        length `block_len`.

    Macro Scenario Injection
    ------------------------
    `macro_sims` is a dictionary with keys "Q1", "Q2", …, each mapping to a DataFrame
    whose rows are named exactly as the columns of `macro_hist` and whose columns are
    simulation identifiers ("1", "2", …). For each quarter q and simulation j, the
    exogenous vector z_t is taken from `macro_sims[q][exog_names].iloc[:, j]`. Missing
    or misaligned names are excluded to preserve ordering consistency.

    Parameters
    ----------
    factor_data : pd.DataFrame
        Historical factor returns containing the required factor columns.
    num_factors : {3,5}
        Select the 3- or 5-factor specification, always including mkt_excess.
    macro_hist : pd.DataFrame
        Historical macro exogenous series aligned by index to factor_data.
    macro_sims : dict[str, pd.DataFrame]
        Quarter-labelled macro scenario draws as described above.
    n_sims : int or None, default None
        Number of simulations. If None, inferred from the number of columns in macro_sims["Q1"].
    horizon : int or None, default None
        Simulation horizon. If None, set to len(macro_sims).
    max_lag : int, default 4
        Maximum lag considered in selection.
    seed : int or None, default 42
        Random seed.
    innovation : {"gaussian","student_t","bootstrap"}, default "student_t"
        Innovation generator for u_t.
    df_t : float, default 7.0
        Degrees of freedom for Student–t.
    block_len : int, default 4
        Block size for bootstrap.

    Returns
    -------
    cov_q : dict[str, pd.DataFrame]
        Quarter-indexed covariance matrices of simulated factors.
    mean_q : pd.DataFrame
        Quarter-indexed simulation means of factors.
    sims_q : dict[str, pd.DataFrame]
        Quarter-indexed factor draws (rows=factors, columns=simulation IDs).

    Notes
    -----
    • Residuals are centred before bootstrapping to avoid reintroducing a mean in u_t.
    • The exogenous coefficient matrix B is extracted from `res.params` using the
      ordering of macro_hist columns present in the fitted parameter table.
    """
  
    if num_factors == 5:
  
        fac_cols = ["mkt_excess", "smb", "hml", "rmw", "cma"]
  
    elif num_factors == 3:
  
        fac_cols = ["mkt_excess", "smb", "hml"]
  
    else:
  
        raise ValueError("num_factors must be 3 or 5")

    F = factor_data[fac_cols].dropna().copy()
  
    Z = macro_hist.loc[F.index].astype(float).dropna()

    common = F.index.intersection(Z.index)
  
    F = F.loc[common]
  
    Z = Z.loc[common]

    var_mod = VAR(F, exog = Z)             
  
    sel = var_mod.select_order(max_lag)       
  
    p = int(max(1, sel.bic or sel.aic or 1))
  
    res = var_mod.fit(p, trend = "c")     

    A = res.coefs.copy()          
   
    Sigma_u = res.sigma_u          
   
    const = res.params.loc["const"].values 

    exog_names = [c for c in Z.columns if c in res.params.index]
   
    if not exog_names:

        exog_names = [r for r in res.params.index if r not in ["const"] and not r.startswith("L")]
   
    B = res.params.loc[exog_names].values 

    radius = _companion_radius(
        A = A
    )
   
    if radius > 0.99:
   
        A *= (0.99 / radius)

    try:
    
        chol = np.linalg.cholesky(Sigma_u + 1e-12 * np.eye(Sigma_u.shape[0]))
    
    except np.linalg.LinAlgError:
    
        w, Q = np.linalg.eigh((Sigma_u + Sigma_u.T) / 2)
    
        w = np.maximum(w, 1e-12)
    
        chol = Q @ np.diag(np.sqrt(w))

    resid = res.resid.values  
    
    resid = resid - resid.mean(0, keepdims = True) 

    qkeys = sorted(macro_sims.keys(), key=lambda s: int(s[1:])) 
   
    H = len(qkeys) if horizon is None else horizon
   
    if n_sims is None:
   
        n_sims = macro_sims[qkeys[0]].shape[1]

    m = len(exog_names)
   
    k = len(fac_cols)

    macro_paths = np.zeros((H, m, n_sims))
   
    for t, q in enumerate(qkeys[:H]):
   
        Mq = macro_sims[q] 
   
        macro_paths[t] = Mq.loc[exog_names].values

    hist = F.values[-p:] 
   
    sims = np.empty((H, k, n_sims))

    rng = np.random.default_rng(seed)

    for j in range(n_sims):
   
        buf = hist.copy()

        if innovation == "bootstrap":
   
            boot_path = _circular_block_bootstrap(
                resid = resid, 
                T = H, 
                L = block_len, 
                rng = rng
            ) 

        for t in range(H):

            if innovation == "student_t":
             
                g = rng.chisquare(df_t) / df_t
             
                u = (rng.standard_normal(k) / np.sqrt(g)) @ chol.T
            
            elif innovation == "bootstrap":
            
                u = boot_path[t]
            
            else:  
                
                u = rng.standard_normal(k) @ chol.T

            x = const.copy()
            
            for ell in range(p):
            
                x += A[ell] @ buf[-ell - 1]

            zt = macro_paths[t, :, j]       
            
            x += B.T @ zt                       
            
            x += u                              

            sims[t, :, j] = x
            
            buf = np.vstack([buf, x])[-p:]

    idx_q = [f"Q{i}" for i in range(1, H + 1)]
   
    mean_q = pd.DataFrame(sims.mean(axis = 2), index = idx_q, columns = fac_cols)
   
    cov_q = {q: pd.DataFrame(np.cov(sims[i], rowvar = True), index = fac_cols, columns = fac_cols)
             for i, q in enumerate(idx_q)}
   
    sims_q = {q: pd.DataFrame(sims[i], index = fac_cols, columns = [f"{n+1}" for n in range(n_sims)])
              for i, q in enumerate(idx_q)}

    return cov_q, mean_q, sims_q


def _recompute_moments_from_sims(
    sims_q: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Recompute per-quarter sample means and covariances from simulated factor panels.

    Parameters
    ----------
    sims_q : dict[str, pd.DataFrame]
        Mapping from "Q1", … to DataFrames of shape (k × n_sims), where rows are
        factor names and columns are simulation identifiers.

    Returns
    -------
    cov_q : dict[str, pd.DataFrame]
        Mapping from quarter label to the k×k sample covariance of that quarter’s draws.
    mean_q : pd.DataFrame
        DataFrame of shape (Q × k) with per-quarter sample means across simulations.

    Notes
    -----
    Each covariance is computed with `np.cov(S, rowvar=True)` where S is the k×n_sims
    matrix of draws for a quarter, thus treating rows as variables and columns as
    realisations.
    """
       
    qkeys = sorted(sims_q.keys(), key=lambda s: int(s[1:]))
   
    factors = list(sims_q[qkeys[0]].index)
   
    mean_q = pd.DataFrame(index=qkeys, columns=factors, dtype=float)
   
    cov_q: dict[str, pd.DataFrame] = {}
   
    for q in qkeys:
   
        S = sims_q[q].values
   
        mean_q.loc[q] = sims_q[q].mean(axis = 1)
   
        cov_q[q] = pd.DataFrame(np.cov(S, rowvar = True), index = factors, columns = factors)
   
    return cov_q, mean_q


def apply_bl_market_to_factor_sims(
    sims_q: dict[str, pd.DataFrame],
    *,
    bl_mu_market_by_q: pd.Series,        
    bl_var_market_by_q: dict[str, float], 
    rf_per_quarter: float,
    market_factor_name: str = "mkt_excess",
    mode: str = "resample",              
    dist: str = "gaussian",             
    df_t: float = 7.0,                   
    var_floor: float = 1e-12,
    seed: int | None = 42
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Impose Black–Litterman (BL) quarterly beliefs on the **market excess factor**
    within a panel of factor simulations, quarter by quarter.

    For each quarter q, the function enforces that the sample mean and variance
    across simulation paths of the row `market_factor_name` (e.g. "mkt_excess")
    match the BL-implied *total* market mean μ_BL^{(q)} and variance σ²_BL^{(q)},
    after converting the mean to an excess return by subtracting the per-quarter
    risk-free rate r_f:

        μ_target^{(q)}  =  μ_BL^{(q)} − r_f,
      
        σ²_target^{(q)} =  max(σ²_BL^{(q)}, var_floor).

    Three adjustment modes are available:

    1) mode = "resample"  (does **not** preserve correlations)
   
       Replace the market draws with i.i.d. samples having the target mean/variance.
   
       - Gaussian:   
       
            m_new = μ_target + σ_target · Z,  with Z ~ N(0,1).
   
       - Student–t:  
       
            m_new = μ_target + s · T_ν, with T_ν ~ t(ν) and
         
            s = σ_target · sqrt((ν−2)/ν) for ν > 2 (so Var(s·T_ν)=σ²_target). 
            
            If ν≤2, fall back to the Gaussian formula. This mode breaks the 
            contemporaneous correlation between the market factor and the other factors.

    2) mode = "rescale"  (preserves pairwise correlations with other factors)
       
       Affine rescaling of the *existing* market draws to match target moments:
       
           m_new = μ_target + (σ_target / σ_curr) · (m_old − μ_curr),
       
       where μ_curr and σ_curr are the sample mean and standard deviation of the
       current market draws in quarter q. For σ_curr≈0, the code falls back to
       "resample". For positive scaling, Pearson correlations with every other
       factor are preserved exactly, since Corr(a + b·m_old, x) = Corr(m_old, x)
       for b>0.

    3) mode = "covmap"   (preserves the *entire* joint covariance structure,
                          except for the market variance which is set to BL)
       A whiten→re-colour linear transform is applied to the **whole** factor
       vector, quarter by quarter, to achieve a *target* covariance that (i) sets
       the market variance to σ²_target while (ii) leaving non-market variances
       unchanged and (iii) preserving all correlations exactly.

       Let X ∈ ℝ^{k×n} be the simulated panel for the quarter (k = number of
       factors = number of rows; n = number of simulation paths = number of
       columns), with row order equal to `df.index`. Define sample moments

           μ₀ = mean(X, axis=1) ∈ ℝ^{k}, arranged as a k×1 column,
       
           Z  = X − μ₀·1ᵀ,
       
           S₀ = (Z Zᵀ) / (n−1)  ∈ ℝ^{k×k}.

       Index the market row by m. Let σ_m = sqrt(S₀_{mm}) be the current market
       standard deviation, and set s = σ_target / max(σ_m, ε). Construct the **target**
       covariance S★ ∈ ℝ^{k×k} by
       
           S★_{mm} = σ²_target,
       
           S★_{mi} = S★_{im} = s · S₀_{mi}   for all i ≠ m,
       
           S★_{ij} = S₀_{ij}                 for all i,j ≠ m.
       
       This keeps non-market variances and **all** correlations intact while
       changing only the market variance.

       Compute robust Cholesky factors (with a tiny ridge εI for numerical
       stability): 
       
            S₀ + εI = L₀ L₀ᵀ 
            
            S★+εI = L★ L★ᵀ. 
        
        The whiten→re-colour map is
          
           Y = μ★ + L★ L₀^{-1} (X − μ₀),
      
       where μ★ equals μ₀ except μ★_m = μ_target (excess). The rows of Y replace X,
       yielding the adjusted quarter panel. Empirically, sample means/variances
       of Y match targets up to Monte-Carlo error; the empirical covariance of Y
       equals S★ up to numerical tolerance.

    Parameters
    ----------
    sims_q : dict[str, DataFrame]
        Dictionary keyed by quarter labels (e.g., "Q1","Q2",…) with values
        DataFrames of shape (k × n), **rows=factors** (must include the row
        named by `market_factor_name`) and **columns=simulation IDs**.
    bl_mu_market_by_q : Series
        BL *total* market mean per quarter. Index must align with sims_q keys.
    bl_var_market_by_q : dict[str, float]
        BL *total* market variance per quarter. Keys must align with sims_q keys.
    rf_per_quarter : float
        Risk-free rate per quarter (decimal). Subtracted from BL total mean to
        obtain the excess mean target (μ_target).
    market_factor_name : str, default "mkt_excess"
        Name of the market excess factor row in each quarter’s DataFrame.
    mode : {"resample","rescale","covmap"}, default "resample"
        Adjustment mode; see above. Use "rescale" to preserve correlations,
        "covmap" to preserve the entire covariance structure.
    dist : {"gaussian","student_t"}, default "gaussian"
        Distribution used only when `mode="resample"`.
    df_t : float, default 7.0
        Degrees of freedom for Student–t in the resampling mode.
    var_floor : float, default 1e-12
        Lower bound on the target variance for numerical stability.
    seed : int or None, default 42
        RNG seed used where randomness is involved.

    Returns
    -------
    sims_new : dict[str, DataFrame]
        Adjusted simulations by quarter (same shapes and indices as input).
    cov_q_new : dict[str, DataFrame]
        Recomputed per-quarter sample covariance matrices from `sims_new`.
    mean_q_new : DataFrame
        Recomputed per-quarter sample means from `sims_new`.

    Notes
    -----
    • Mode "rescale" preserves Pearson correlations with other factors **exactly**
      (for positive scaling). Mode "covmap" preserves **all** correlations and all
      non-market variances by construction, while setting the market variance to
      the BL target.

    • BL inputs are for total returns; converting to excess via μ_target = μ_BL − r_f
      aligns them with the definition of the market factor “mkt_excess”.

    • All matching is in sample (across simulated paths), so targets are attained
      up to Monte-Carlo error.
    """

    rng = np.random.default_rng(seed)
   
    sims_new: dict[str, pd.DataFrame] = {}
    
    for q, df in sims_q.items():
   
        if market_factor_name not in df.index:
   
            sims_new[q] = df.copy()
   
            continue

        mu_bl_total = float(bl_mu_market_by_q.get(q, np.nan))
   
        if not np.isfinite(mu_bl_total) or q not in bl_var_market_by_q:
   
            sims_new[q] = df.copy()
   
            continue

        mu_target_excess = mu_bl_total - rf_per_quarter
   
        var_target = max(float(bl_var_market_by_q[q]), var_floor)
   
        sd_target = np.sqrt(var_target)
   
        n_sims = df.shape[1]

        df_new = df.copy()
   
        m_old = df.loc[market_factor_name].values.astype(float)

        if mode.lower() == "resample":

            if dist.lower() == "student_t":

                v = df_t

                if v <= 2:

                    z = rng.standard_normal(n_sims)

                    m_new = mu_target_excess + sd_target * z

                else:

                    scale = sd_target / np.sqrt(v / (v - 2))

                    tdraws = rng.standard_t(v, size = n_sims)

                    m_new = mu_target_excess + scale * tdraws
            else:
            
                z = rng.standard_normal(n_sims)
            
                m_new = mu_target_excess + sd_target * z
                
        elif mode.lower() == "covmap":
            
            df_new = df.copy()
            
            X = df.values.astype(float)            
            
            k, n = X.shape

            mu0 = X.mean(axis = 1, keepdims = True)         
           
            Z = X - mu0
           
            S0 = (Z @ Z.T) / max(1, n-1)

            eps = 1e-10

            S0_reg = S0 + eps * np.eye(k)

            m_idx = list(df.index).index(market_factor_name)

            mu_star = mu0.copy()

            mu_star[m_idx, 0] = mu_target_excess

            S_star = S0.copy()
          
            sd_m = np.sqrt(S0_reg[m_idx, m_idx])
          
            s = float(sd_target / max(sd_m, 1e-12))
          
            S_star[m_idx, m_idx] = sd_target**2
          
            for i in range(k):
          
                if i == m_idx: 
          
                    continue
          
                S_star[m_idx, i] = S0_reg[m_idx, i] * s
          
                S_star[i, m_idx] = S_star[m_idx, i]

            L0 = np.linalg.cholesky(S0_reg)

            S_star_reg = S_star + eps * np.eye(k)

            try:

                Ls = np.linalg.cholesky(S_star_reg)

            except np.linalg.LinAlgError:

                w, Q = np.linalg.eigh((S_star_reg + S_star_reg.T) / 2)

                w = np.maximum(w, 1e-12)

                Ls = Q @ np.diag(np.sqrt(w))

            W = np.linalg.solve(L0, Z)

            Y = (mu_star + (Ls @ W))             

            df_new.iloc[:, :] = Y
            
            m_new = Y[m_idx, :].ravel()  

        else:  
           
            mu_curr = float(np.mean(m_old))
           
            sd_curr = float(np.std(m_old, ddof = 1)) if n_sims > 1 else 0.0
           
            if sd_curr < 1e-12:

                z = rng.standard_normal(n_sims)

                m_new = mu_target_excess + sd_target * z

            else:

                m_new = mu_target_excess + (sd_target / sd_curr) * (m_old - mu_curr)

        df_new.loc[market_factor_name] = m_new

        sims_new[q] = df_new

    cov_q_new, mean_q_new = _recompute_moments_from_sims(
        sims_q = sims_new
    )

    return sims_new, cov_q_new, mean_q_new
