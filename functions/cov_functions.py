"""
Provides covariance estimators
- sample
- constant‑correlation
- shrinkage versions
- predicted
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from scipy.optimize import nnls


def _nearest_psd_preserve_diag(
    C: np.ndarray, 
    eps: float = 1e-8
) -> np.ndarray:
    """
    Project a symmetric covariance matrix to the nearest PSD matrix **while preserving its diagonal**.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Input (possibly indefinite) covariance matrix.
    eps : float, default 1e-8
        Eigenvalue floor used during projection in correlation space.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric positive semidefinite matrix with the **same diagonal as C**.

    Method
    ------
    
    1) Symmetrise:  C ← (C + Cᵀ)/2.
    
    2) Extract variances d = diag(C), enforce d_i ≥ eps, and form D^{-1/2}:
        s_i = √d_i  
        D^{-1/2} = diag(1/s_i).
    
    3) Convert to a correlation matrix:
        R = D^{-1/2} C D^{-1/2};  symmetrise R.
    
    4) Eigen-decompose:  R = V diag(λ) Vᵀ. Floor eigenvalues: λ̂_i = max(λ_i, eps).
    
    5) Rebuild R_psd = V diag(λ̂) Vᵀ, then **force unit diagonal**: diag(R_psd) = 1.
    
    6) Map back to covariance space with original variances:
        C_psd = D^{1/2} R_psd D^{1/2}.
    
    7) Symmetrise the result.

    Notes
    -----
    - Working in correlation space keeps relative cross-correlations but lets us
    enforce PSD via eigenvalue flooring cleanly.
    - The diagonal of the output equals the original diagonal of C (up to numeric
    rounding), which preserves marginal variances.
    """

   
    C = np.asarray(C, dtype = float)
   
    C = 0.5 * (C + C.T)
   
    d = np.clip(np.diag(C).copy(), eps, None)
   
    s = np.sqrt(d)
   
    Dinv = np.diag(1.0 / s)
   
    R = Dinv @ C @ Dinv
   
    R = 0.5 * (R + R.T)
   
    w, V = np.linalg.eigh(R)
   
    w = np.maximum(w, eps)
   
    R_psd = (V * w) @ V.T
   
    np.fill_diagonal(R_psd, 1.0)
   
    C_psd = np.diag(s) @ R_psd @ np.diag(s)
   
    return 0.5 * (C_psd + C_psd.T)


def _clean_cov_matrix(
    M: np.ndarray,
    min_var: float = 1e-10
) -> np.ndarray:
    """
    Make a covariance-like matrix finite, symmetric, and strictly positive on the diagonal.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        Input matrix with possible NaN/Inf/small negatives on the diagonal.
    min_var : float, default 1e-10
        Minimum variance enforced on the diagonal.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric matrix with finite entries and diag(M) ≥ min_var.

    Operations
    ----------
    - Replace non-finite entries with 0.
    - Symmetrise: (M + Mᵀ)/2.
    - Coerce diagonal to be finite and ≥ min_var.
    """

    
    M = np.asarray(M, dtype = float)
    
    M = np.where(np.isfinite(M), M, 0.0)
  
    M = 0.5 * (M + M.T)
  
    d = np.diag(M).copy()
  
    d = np.where(np.isfinite(d), d, min_var)
  
    d = np.clip(d, min_var, None)
  
    np.fill_diagonal(M, d)
  
    return M


def _clean_corr_matrix(
    R: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """
    Normalise a correlation-ish matrix to be finite, symmetric, and have unit diagonal.

    Parameters
    ----------
    R : pd.DataFrame | np.ndarray, shape (n, n)
        Matrix expected to be close to a correlation matrix.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric matrix with finite entries and diag(R)=1.

    Notes
    -----
    This function does **not** force positive semidefiniteness. Use with care if PSD
    is required.
    """

    
    R = np.asarray(R, dtype = float)
    
    R = np.where(np.isfinite(R), R, 0.0)
    
    R = 0.5 * (R + R.T)
    
    np.fill_diagonal(R, 1.0)
    
    return R


def _assert_finite(
    label: str,
    M: np.ndarray
):
    """
    Validate that a covariance matrix is finite, symmetric, and has strictly positive diagonal.

    Parameters
    ----------
    label : str
        Name used in error messages.
    M : np.ndarray, shape (n, n)
        Matrix to validate.

    Raises
    ------
    ValueError
        If any entry is non-finite, if M is not symmetric within atol=1e-10,
        or if any diagonal entry is ≤ 0.
    """

    if not np.all(np.isfinite(M)):

        raise ValueError(f"{label} contains non-finite entries.")

    if not np.allclose(M, M.T, atol = 1e-10):

        raise ValueError(f"{label} is not symmetric.")

    d = np.diag(M)

    if np.any(d <= 0):

        raise ValueError(f"{label} has non-positive variances.")


def _winsorize_df(
    df: pd.DataFrame, 
    z: float = 8.0
) -> pd.DataFrame:
    """
    Robustly winsorize each column using median and MAD to limit extreme outliers.

    Parameters
    ----------
    df : pd.DataFrame, shape (T, n)
        Data to winsorize (e.g., returns); NaNs allowed.
    z : float, default 8.0
        Winsorization band in MAD units.

    Returns
    -------
    pd.DataFrame
        Winsorized values with same index/columns.

    Details
    -------
    For each column x:
    - Compute median m = median(x) and MAD = median(|x - m|).
    - Convert MAD to a robust σ via c ≈ 1.4826:  σ̂ ≈ c * MAD.
    - Clip to [m - z σ̂, m + z σ̂].

    This reduces the influence of rare spikes before covariance estimation.
    """

    X = df.to_numpy(dtype = float)
   
    med = np.nanmedian(X, axis = 0)
   
    mad = np.nanmedian(np.abs(X - med), axis = 0) + 1e-12
   
    lo = med - 1.4826 * mad * z
    
    hi = med + 1.4826 * mad * z
    
    X = np.clip(X, lo, hi)
    
    return pd.DataFrame(X, index = df.index, columns = df.columns)


def _ewma_cov(
    returns_weekly: pd.DataFrame, 
    lam: float = 0.97, 
    eps: float = 1e-12
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average (EWMA) covariance (weekly units).

    Parameters
    ----------
    returns_weekly : pd.DataFrame, shape (T, n)
        Weekly returns. No annualisation is applied here.
    lam : float, default 0.97
        Decay λ (higher = slower decay). Effective weight at lag k is (1-λ) λ^k.
    eps : float, default 1e-12
        Guard against division by zero in weight normalisation.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric EWMA covariance in *weekly* units.

    Math
    ----
    Let x_t ∈ ℝ^n be the column vector of returns at time t (t=0 oldest).
    We compute backwards:

    S₀ = 0,  w₀ = 0
    For t = T-1 .. 0:
        S ← λ S + (1-λ) x_t x_tᵀ
        w ← λ w + (1-λ)

    Normalise:
        Σ_EWMA = S / max(w, eps)

    Finally symmetrise: (Σ_EWMA + Σ_EWMAᵀ)/2.
    """


    X = returns_weekly.to_numpy(float)

    n = X.shape[1]

    S = np.zeros((n, n), float)

    w_sum = 0.0

    for t in range(X.shape[0] - 1, -1, -1):

        x = X[t, :][:, None]

        S = lam * S + (1 - lam) * (x @ x.T)

        w_sum = lam * w_sum + (1 - lam)

    S = S / max(w_sum, eps)

    S = 0.5 * (S + S.T)

    return S


def _constant_correlation_prior(
    corr: pd.DataFrame,
    std_vec: np.ndarray
) -> np.ndarray:
    """
    Build a constant-correlation style prior covariance using a correlation matrix and target vols.

    Parameters
    ----------
    corr : pd.DataFrame, shape (n, n)
        Correlation matrix (will be cleaned to have unit diagonal).
    std_vec : np.ndarray, shape (n,)
        Vector of target standard deviations σ.

    Returns
    -------
    np.ndarray, shape (n, n)
        Prior covariance Σ_prior = R ⊙ (σ σᵀ) = R * (σ ⊗ σ), where ⊙ is the Hadamard product.

    Notes
    -----
    - The diagonal is σ_i^2.
    - Off-diagonals are ρ_{ij} σ_i σ_j.
    """

    R = _clean_corr_matrix(
        R = corr.values
    )

    return R * np.outer(std_vec, std_vec)


def _lw_reference_nan_safe(
    weekly_returns: pd.DataFrame,
    S_weekly: np.ndarray,
    min_complete_weeks: int = 26,
    policy: str = "complete_rows"
) -> np.ndarray:
    """
    Construct a Ledoit–Wolf covariance estimator on complete weekly observations (NaN-safe).

    Parameters
    ----------
    weekly_returns : pd.DataFrame, shape (T, n)
        Weekly returns with possible NaNs.
    S_weekly : np.ndarray, shape (n, n)
        Fallback covariance (same units) if not enough complete rows.
    min_complete_weeks : int, default 26
        Minimum number of complete (non-NaN) rows required to fit Ledoit–Wolf.
    policy : {"complete_rows","truncate_common_start"}, default "complete_rows"
        If "truncate_common_start", align all columns to the latest common first-valid index
        before dropping NaNs.

    Returns
    -------
    np.ndarray, shape (n, n)
        Ledoit–Wolf covariance (weekly units) or `S_weekly` if insufficient data.

    Notes
    -----
    `sklearn.covariance.LedoitWolf` shrinks the sample covariance toward a scaled
    identity target with an analytically chosen shrinkage intensity, reducing
    estimation error in small samples or with noisy data.
    """

  
    W = weekly_returns

    if policy == "truncate_common_start":
     
        starts = [W[c].first_valid_index() for c in W.columns]
     
        starts = [d for d in starts if d is not None]
     
        if len(starts) > 0:
     
            W = W.loc[max(starts):]

    W_cc = W.dropna(how="any")
   
    X = W_cc.to_numpy(float)
   
    if X.shape[0] >= min_complete_weeks:
   
        return LedoitWolf().fit(X).covariance_
   
    return S_weekly


def factor_model_cov(
    returns_weekly: pd.DataFrame,
    factors_weekly: pd.DataFrame,
    index_factors_weekly: pd.DataFrame | None = None,
    use_excess: bool = False,
    ridge: float = 1e-4,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Factor-model covariance on *weekly* data:  Σ = B Σ_F Bᵀ + Ψ.

    Parameters
    ----------
    returns_weekly : pd.DataFrame, shape (T, N)
        Asset weekly returns.
    factors_weekly : pd.DataFrame, shape (T, K₁)
        Factor returns (e.g., styles); if `use_excess` and column "RF" present, will be handled.
    index_factors_weekly : pd.DataFrame | None, shape (T, K₂), optional
        Additional index or macro factors (concatenated with `factors_weekly`).
    use_excess : bool, default False
        If True and "RF" exists in factors_weekly or combined frame, convert asset returns to
        excess returns and drop the "RF" column from factors.
    ridge : float, default 1e-4
        ℓ₂ regularisation on (FᵀF) for stable regression.
    eps : float, default 1e-12
        Floor for idiosyncratic variances.

    Returns
    -------
    pd.DataFrame, shape (N, N)
        Cleaned covariance estimate on *weekly* units (no annualisation).

    Procedure
    ---------
    1) Align times across all inputs; drop rows with any NaN after robust winsorisation.
    2) If `use_excess` and "RF" available, set R ← R - RF and drop RF from factors.
    3) Regress returns on factors with ridge:
        F ∈ ℝ^{T×K}, X ∈ ℝ^{T×N}
        B = (Fᵀ F + ridge I)^{-1} Fᵀ X
    Residuals: E = X - F B
    Residual variances: ψ_i = max( (∑_t E_{t,i}²) / (T-K), eps )
    4) Factor covariance:
    - Prefer Ledoit–Wolf on F:  Σ_F = LW(F)
    - Fallback: sample covariance of F with ddof=0.
    5) Assemble:
        Σ = Bᵀ Σ_F B + diag(ψ)
    6) Clean with `_clean_cov_matrix`.

    Notes
    -----
    - Units are weekly; annualise externally if needed.
    - Using Ledoit–Wolf for Σ_F reduces noise in factor covariance estimation.
    """

   
    parts = [returns_weekly, factors_weekly]
   
    if index_factors_weekly is not None:
   
        parts.append(index_factors_weekly)

    idx = parts[0].index
   
    for p in parts[1:]:
   
        idx = idx.intersection(p.index)

    R = returns_weekly.loc[idx].astype(float)
   
    F_all = factors_weekly.loc[idx].astype(float).copy()
   
    if index_factors_weekly is not None:
   
        F_all = pd.concat([F_all, index_factors_weekly.loc[idx].astype(float)], axis = 1)

    if use_excess and "RF" in F_all.columns:
    
        rf = F_all["RF"].copy()
     
        F_all = F_all.drop(columns = ["RF"])
     
        R = R.sub(rf.values, axis = 0)

    R = R.replace([np.inf, -np.inf], np.nan)
    
    F_all = F_all.replace([np.inf, -np.inf], np.nan)
    
    R = _winsorize_df(
        df = R
    )
    
    F_all = _winsorize_df(
        df = F_all
    )

    mask = R.notna().all(axis = 1) & F_all.notna().all(axis = 1)
   
    R = R.loc[mask]
    
    F_all = F_all.loc[mask]

    F = F_all.to_numpy(dtype = float)
    
    X = R.to_numpy(dtype = float)
   
    T, K = F.shape

    if T < K + 5: 
       
        Sigma = np.cov(X, rowvar=False, ddof=0)
       
        return pd.DataFrame(
            _clean_cov_matrix(
                M = Sigma
            ), index = R.columns, columns = R.columns
        )

    FtF = F.T @ F + ridge * np.eye(K)
  
    FtX = F.T @ X
  
    B = np.linalg.solve(FtF, FtX)          
  
    E = X - F @ B                            
  
    dof = max(T - K, 1)
  
    psi = np.maximum((E * E).sum(axis = 0) / dof, eps)

    try:
  
        Sigma_F = LedoitWolf().fit(F).covariance_
  
    except Exception:
  
        Sigma_F = np.cov(F, rowvar = False, ddof = 0)

    Sigma = B.T @ Sigma_F @ B + np.diag(psi)
  
    Sigma = _clean_cov_matrix(
        M = Sigma
    )
  
    return pd.DataFrame(Sigma, index = R.columns, columns = R.columns)


def _pca_stat_cov_on_residuals(
    X: np.ndarray,
    F: np.ndarray | None,
    K_max: int = 20,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Statistical (PCA) factor covariance on residuals (optional factor-residualisation).

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Weekly asset returns.
    F : np.ndarray | None, shape (T, K), optional
        Factor matrix. If provided, regress X on F and use residuals.
    K_max : int, default 20
        Max number of principal components retained.
    eps : float, default 1e-8
        Eigenvalue floor.

    Returns
    -------
    np.ndarray, shape (N, N)
        PSD statistical covariance constructed from top-K eigenmodes of residual Σ_E.

    Math
    ----
    If F provided:
        B = argmin_B ||X - F B||_F² = (F⁺) X  (least squares)
        E = X - F B
    Else:
        E = X

    Residual covariance:
        Σ_E = cov(E) with ddof=0.

    Eigen-decompose:
        Σ_E = Q diag(λ) Qᵀ,  λ sorted ascending.
    Choose K = min(K_max, max(1, ⌊N/5⌋)).
    Let U = Q[:, -K:],  Λ = diag(λ_{N-K+1..N}).

    Statistical approximation:
        Σ_stat = U Λ Uᵀ

    Finally, clean via `_clean_cov_matrix`.

    Notes
    -----
    This captures the strongest common residual modes beyond named factors and
    is useful as a robust, low-rank “statistical” target.
    """

  
    if F is not None and F.size > 0:
  
        B = np.linalg.lstsq(F, X, rcond = None)[0]
  
        E = X - F @ B
  
    else:
  
        E = X

    Sigma_E = np.cov(E, rowvar = False, ddof = 0)
   
    evals, evecs = np.linalg.eigh(Sigma_E)
   
    evals = np.clip(evals, eps, None)

    N = Sigma_E.shape[0]
   
    K = min(K_max, max(1, N // 5)) 

    U = evecs[:, -K:]
  
    Lam = np.diag(evals[-K:])
  
    Sigma_stat = U @ Lam @ U.T
  
    return _clean_cov_matrix(
        M = Sigma_stat
    )


def _project_boxed_simplex_leq(
    v: np.ndarray, 
    lb: np.ndarray,
    ub: np.ndarray, 
    s: float = 1.0,
    tol: float = 1e-12, 
    max_iter: int = 100
) -> np.ndarray:
    """
    Euclidean projection onto the **boxed simplex with inequality**:
        Π(v) = argmin_x ½||x - v||²  s.t.  lb ≤ x ≤ ub,  1ᵀx ≤ s.

    Parameters
    ----------
    v : np.ndarray, shape (m,)
        Vector to project.
    lb, ub : np.ndarray, shape (m,)
        Lower/upper bounds (componentwise). Must satisfy lb_i ≤ ub_i.
    s : float, default 1.0
        Simplex size (upper bound on the sum).
    tol : float, default 1e-12
        Tolerance for sum constraint feasibility.
    max_iter : int, default 100
        Max iterations for bisection when enforcing the sum.

    Returns
    -------
    np.ndarray, shape (m,)
        Projected vector.

    Algorithm
    ---------
    1) Clip v to [lb, ub]:  x = clip(v, lb, ub). If 1ᵀx ≤ s + tol, return x.
    2) Else enforce equality 1ᵀx = s via scalar Lagrange multiplier τ:
        x(τ) = clip(v - τ, lb, ub).
    The function g(τ) = 1ᵀ x(τ) is monotonically decreasing in τ. Find τ by
    bisection on [min(v-ub), max(v-lb)], then set x = x(τ).

    KKT intuition
    -------------
    The projection onto box∩simplex has piecewise-linear structure; bisection on τ
    is guaranteed to converge because g is continuous and strictly decreasing
    whenever the active set is nonempty.
    """


    v = np.asarray(v, dtype = float)

    lb = np.asarray(lb, dtype = float)

    ub = np.asarray(ub, dtype = float)
  
    if np.any(lb > ub):
  
        raise ValueError("Infeasible: some lower bound exceeds upper bound.")
  
    if lb.sum() - s > 1e-12:
  
        raise ValueError("Infeasible: sum of lower bounds exceeds simplex size s.")

    x = np.clip(v, lb, ub)
  
    sx = float(x.sum())
  
    if sx <= s + tol:
  
        return x

    tau_lo = float(np.min(v - ub))

    tau_hi = float(np.max(v - lb))

    for _ in range(max_iter):

        tau = 0.5 * (tau_lo + tau_hi)

        x = np.clip(v - tau, lb, ub)

        diff = float(x.sum() - s)

        if abs(diff) <= tol:

            return x

        if diff > 0:

            tau_lo = tau

        else:

            tau_hi = tau

    return np.clip(v - tau, lb, ub)


def _make_bounds_for_targets(
    names: list[str], 
    **kw
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lower/upper bounds (lb, ub) for shrinkage target weights in Σ(w).

    Parameters
    ----------
    names : list[str]
        Target names, e.g. ["P", "S_EWMA", "C_EWMA", "F", "FF", "IDX", "IND", "SEC", "STAT"].
    **kw : dict
        Parameter bag containing either fixed weights `w_<NAME>` or bounds
        `<param>_min` / `<param>_max` per target family.
        See PARAM_MAP inside for the exact mapping.

    Returns
    -------
    (lb, ub) : tuple[np.ndarray, np.ndarray], shape (m,)
        Bound vectors aligned with `names`. `ub` may contain +inf entries.

    Raises
    ------
    ValueError
        If any lower bound exceeds the corresponding upper bound, or if ∑ lb > 1.

    Notes
    -----
    - Fixed weights are applied by setting lb_i = ub_i = value.
    - This function does not enforce ∑ ub ≥ 1; feasibility in the solver is
    handled by projecting onto {w ≥ lb, w ≤ ub, ∑w ≤ 1}.
    """


    PARAM_MAP = {
        "P": ("p_min", "p_max"),
        "S_EWMA": ("s_ewma_min", "s_ewma_max"),
        "C_EWMA": ("c_ewma_min", "c_ewma_max"),
        "F": ("fpred_min", "fpred_max"),
        "FF": ("ff_min", "ff_max"),
        "IDX": ("idx_min", "idx_max"),
        "IND": ("ind_min", "ind_max"),
        "SEC": ("sec_min", "sec_max"),
        "STAT": ("stat_min", "stat_max"),
    }

    lb = np.zeros(len(names), dtype = float)
   
    ub = np.full(len(names), np.inf, dtype = float)

    for j, nm in enumerate(names):
     
        mn_key, mx_key = PARAM_MAP.get(nm, (None, None))

        if mn_key:
            
            mn = float(kw.get(mn_key, 0.0)) 
        
        else:
            
            mn = 0.0

        if mx_key:
            
            mx_val = kw.get(mx_key, None) 
            
        else:
            
            mx_val = None

        if mx_val is not None:
            
            ux = float(mx_val) 
        
        else:
            
            ux = np.inf

        fixed_key = f"w_{nm}"

        if fixed_key in kw and kw[fixed_key] is not None:

            val = max(0.0, float(kw[fixed_key]))

            lb[j] = val

            ub[j] = val

        else:

            lb[j] = max(0.0, mn)

            ub[j] = max(lb[j], ux)

    if lb.sum() > 1.0 + 1e-12:

        raise ValueError(f"Infeasible bounds: sum of mins {lb.sum():.4f} exceeds 1.0")

    return lb, ub


def shrinkage_covariance(
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    comb_std: pd.Series,
    common_idx: list[str],
    *,
    ff_factors_weekly: pd.DataFrame | None = None,
    index_returns_weekly: pd.DataFrame | None = None,
    industry_returns_weekly: pd.DataFrame | None = None,
    sector_returns_weekly: pd.DataFrame | None = None,
    use_excess_ff: bool = True,
    description: bool = False,

    # Recency/regularisation
    ewma_lambda: float = 0.97,
    tau_logdet: float = 1e-3,
    logdet_eps: float = 1e-6,

    # Optional fixed weights (w_<name>) or caps
    w_P: float | None = None,
    w_S_EWMA: float | None = None,
    w_C_EWMA: float | None = None,
    w_F: float | None = None,
    w_FF: float | None = None,
    w_IDX: float | None = None,
    w_IND: float | None = None,
    w_SEC: float | None = None,
    w_STAT: float | None = None,

    # Box constraints (mins/maxes). ***Ensure F has at least 30%.***
    p_min: float = 0.10,  
    p_max: float | None = None,
    s_ewma_min: float = 0.05, 
    s_ewma_max: float = 0.20,
    c_ewma_min: float = 0.05,
    c_ewma_max: float = 0.20,
    fpred_min: float = 0.30, 
    fpred_max: float | None = None,  
    ff_min: float = 0.05, 
    ff_max: float = 0.15,
    idx_min: float = 0.02, 
    idx_max: float = 0.10,
    ind_min: float = 0.02,
    ind_max: float = 0.10,
    sec_min: float = 0.02, 
    sec_max: float = 0.10,
    stat_min: float = 0.05,
    stat_max: float = 0.15,

    periods_per_year: int = 52,
) -> pd.DataFrame:
    """
    Build an **annualised** covariance matrix by convexly blending multiple targets
    with a log-det barrier to ensure SPD:

        Σ(w) = S_base + ∑_j w_j (M_j - S_base),
        w ≥ 0, 1ᵀw ≤ 1,

    i.e., Σ(w) = (1 - 1ᵀw) S_base + ∑_j w_j M_j.

    Inputs
    ------
    daily_5y, weekly_5y, monthly_5y : pd.DataFrame
        5-year panels of daily / weekly / monthly returns for the same asset list.
    comb_std : pd.Series
        Combined/forecast standard deviations per asset (annualised); used to build F target.
    common_idx : list[str]
        Ordered list of asset tickers to keep and align across all inputs.

    Optional factors (weekly units)
    -------------------------------
    ff_factors_weekly : pd.DataFrame | None
        Fama–French (or similar ETFs) style factors. If `use_excess_ff` and column "RF"
        exists, assets are converted to excess returns and "RF" is dropped.
    index_returns_weekly, industry_returns_weekly, sector_returns_weekly : pd.DataFrame | None
        Additional factor blocks for INDEX / INDUSTRY / SECTOR targets.

    Tuning / constraints
    --------------------
    use_excess_ff : bool, default True
        Use excess returns for FF if RF available.
    description : bool, default False
        If True, return (cov, cov.describe()).
    ewma_lambda : float, default 0.97
        Decay for EWMA target(s).
    tau_logdet : float, default 1e-3
        Weight of the log-det barrier in the objective (SPD encouragement).
    logdet_eps : float, default 1e-6
        Minimal eigenvalue enforced: Σ(w) ≻ logdet_eps·I.
    Fixed weights and box constraints:
        w_<NAME> fixes a target's weight exactly (by setting lb=ub=value).
        <family>_min / <family>_max define bounds. **By default F has fpred_min=0.30.**
    periods_per_year : int, default 52
        Annualisation factor at the **end** (C_wk → C_ann = C_wk * periods_per_year).

    Returns
    -------
    pd.DataFrame, shape (N, N)
        Annualised covariance for tickers in `common_idx`.
    If description=True:
        (cov_df, cov_df.describe()).

    Construction steps
    ------------------
    
    1) **Base multi-horizon covariance (weekly units):**
    For each horizon H ∈ {5y, 3y, 1y}, compute annualised covariances:
        Σ_D = cov(daily) * 252,  Σ_W = cov(weekly) * 52,  Σ_M = cov(monthly) * 12
        Σ_ann(H) = 0.2·Σ_D + 0.6·Σ_W + 0.2·Σ_M
    Then average across horizons and de-annualise to weekly:
        S_base = 0.5·Σ_ann(5y) + 0.3·Σ_ann(3y) + 0.2·Σ_ann(1y)
        S_base ← S_base / periods_per_year
    Clean with `_clean_cov_matrix`.

    2) **Targets M_j (weekly units):**
   
    - P (constant-correlation prior):
        Corr_ms = 0.5·Corr(5y) + 0.3·Corr(3y) + 0.2·Corr(1y),
        Σ_P = Corr_ms ⊙ (σ σᵀ), where σ = √diag(S_base).
   
    - S_EWMA:  direct EWMA covariance on weekly returns, Σ_S_EWMA.
   
    - C_EWMA:  EWMA correlation R_EW on weekly returns, then
        Σ_C_EWMA = R_EW ⊙ (σ σᵀ) with σ from S_base.
   
    - F (forecast error prior): using annualised comb_std,
        se_wk = comb_std / √periods_per_year,  Σ_F = Corr_ms ⊙ (se_wk se_wkᵀ).
   
    - FF / IDX / IND / SEC:  weekly **factor-model covariances**
        Σ = B Σ_F Bᵀ + Ψ (see `factor_model_cov`), one per factor block.
   
    - STAT: statistical residual covariance via PCA (see `_pca_stat_cov_on_residuals`).

    Each target is cleaned via `_clean_cov_matrix`.

    3) **Reference covariance T_ref (weekly units):**
    Build a NaN-safe Ledoit–Wolf covariance on complete weekly rows; fall back to
    S_base if insufficient data. Clean the result.

    4) **Optimisation (convex):**
    
    Variables: w ∈ ℝ^m, w ≥ 0, 1ᵀw ≤ 1.
    
    Decision covariance:
    
        Σ(w) = S_base + ∑_j w_j (M_j - S_base).
    
    Constraints: Σ(w) ≻ logdet_eps·I, and per-target bounds from `_make_bounds_for_targets`.
    
    Objective:
    
        minimise  ‖Σ(w) - T_ref‖_F²  −  τ·logdet(Σ(w)),
    
    where τ = tau_logdet ≥ 0. The barrier keeps Σ(w) well-conditioned.

    Solver: MOSEK if available, else SCS. If solving fails:
    
    **Fallback NNLS** on the linearised system vec(T_ref − S_base) ≈ A w, then
    project onto the boxed simplex (∑w ≤ 1) via `_project_boxed_simplex_leq`.

    5) **Assemble and post-process:**
    
    Σ_wk = S_base + ∑_j ŵ_j (M_j − S_base).
    Project to nearest PSD with preserved diagonal via `_nearest_psd_preserve_diag`.
    Annualise: Σ_ann = Σ_wk · periods_per_year.

    6) **Output & diagnostics:**
    
    Returns Σ_ann as a DataFrame. Prints the final shrinkage weights, with
    “S (implicit)” = 1 − ∑_j ŵ_j.

    Caveats
    -------
    - All target covariances are built in **weekly** units; annualisation is applied only
    at the very end.
    - Bounds should be chosen to ensure feasibility (∑ lb ≤ 1).
    """

    idx = list(common_idx)
    
    daily_5y = daily_5y.loc[:, idx]
    
    weekly_5y = weekly_5y.loc[:, idx]
    
    monthly_5y = monthly_5y.loc[:, idx]
    
    comb_std = comb_std.loc[idx]


    def _cov_for_horizon(
        d,
        w, 
        m
    ):
        """
        Blend daily/weekly/monthly sample covariances into an annualised covariance for one horizon,
        then convert back to weekly units.

        Parameters
        ----------
        d, w, m : pd.DataFrame
            Daily, weekly, monthly return panels (aligned on the same asset set).

        Returns
        -------
        np.ndarray, shape (N, N)
            Weekly-unit covariance:
                Σ_ann = 0.2·cov(d) * 252 + 0.6·cov(w) * 52 + 0.2·cov(m) * 12
                Σ_wk  = Σ_ann / periods_per_year
        """
            
        cov_d = d.cov(ddof = 0) * 252.0
       
        cov_w = w.cov(ddof = 0) * 52.0
       
        cov_m = m.cov(ddof = 0) * 12.0
       
        cov_ann = 0.2 * cov_d.values + 0.6 * cov_w.values + 0.2 * cov_m.values
       
        return cov_ann / periods_per_year


    d3 = daily_5y[daily_5y.index >= daily_5y.index.max() - pd.DateOffset(years = 3)]
   
    w3 = weekly_5y[weekly_5y.index >= weekly_5y.index.max() - pd.DateOffset(years = 3)]
   
    m3 = monthly_5y[monthly_5y.index >= monthly_5y.index.max() - pd.DateOffset(years = 3)]

    d1 = daily_5y[daily_5y.index >= daily_5y.index.max() - pd.DateOffset(years = 1)]
    
    w1 = weekly_5y[weekly_5y.index >= weekly_5y.index.max() - pd.DateOffset(years = 1)]
    
    m1 = monthly_5y[monthly_5y.index >= monthly_5y.index.max() - pd.DateOffset(years = 1)]

    S5 = _cov_for_horizon(
        d = daily_5y, 
        w = weekly_5y, 
        m = monthly_5y
    )
   
    S3 = _cov_for_horizon(
        d = d3, 
        w = w3, 
        m = m3
    )
   
    S1 = _cov_for_horizon(
        d = d1, 
        w = w1, 
        m = m1
    )
   
    S_base = _clean_cov_matrix(
        M = 0.5 * S5 + 0.3 * S3 + 0.2 * S1
    )

    S_EWMA = _clean_cov_matrix(
        M = _ewma_cov(
            returns_weekly = weekly_5y, 
            lam = ewma_lambda
        )
    )

    std_wk = np.sqrt(np.clip(np.diag(S_base), 1e-12, None))

    Corr5 = daily_5y.corr()

    Corr3 = d3.corr()

    Corr1 = d1.corr()

    Corr_ms = (0.5 * Corr5 + 0.3 * Corr3 + 0.2 * Corr1).reindex(index = idx, columns = idx)

    Corr_ms = pd.DataFrame(
        _clean_corr_matrix(
            R = Corr_ms.values
        ), index = idx, columns = idx
    )
    
    P_const = _clean_cov_matrix(
        M = _constant_correlation_prior(
            corr = Corr_ms, 
            std_vec = std_wk
        )
    )

    S_ew = _ewma_cov(
        returns_weekly = weekly_5y, 
        lam = ewma_lambda
    )
   
    s_ew = np.sqrt(np.clip(np.diag(S_ew), 1e-12, None))
   
    R_ew = _clean_corr_matrix(
        R = S_ew / np.outer(s_ew, s_ew)
    )
    
    C_EWMA = _clean_cov_matrix(
        M = R_ew * np.outer(std_wk, std_wk)
    )

    se_wk = comb_std.values / np.sqrt(periods_per_year)

    F_pred = _clean_cov_matrix(
        M = np.outer(se_wk, se_wk) * Corr_ms.values
    )

    mats: list[np.ndarray] = []
  
    names: list[str] = []

    names += ["P", "S_EWMA", "C_EWMA", "F"]
   
    mats += [ P_const, S_EWMA,  C_EWMA,  F_pred ]

    if ff_factors_weekly is not None:

        Sigma_FF = factor_model_cov(
            returns_weekly = weekly_5y, 
            factors_weekly = ff_factors_weekly,
            use_excess = use_excess_ff
        )

        mats.append(Sigma_FF.values)

        names.append("FF")

    if index_returns_weekly is not None:
       
        Sigma_IDX = factor_model_cov(
            returns_weekly = weekly_5y, 
            factors_weekly = index_returns_weekly, 
            use_excess = False
        )
       
        mats.append(Sigma_IDX.values)
       
        names.append("IDX")

    if industry_returns_weekly is not None:
       
        Sigma_IND = factor_model_cov(
            returns_weekly = weekly_5y, 
            factors_weekly = industry_returns_weekly, 
            use_excess = False
        )
       
        mats.append(Sigma_IND.values)
       
        names.append("IND")

    if sector_returns_weekly is not None:
        
        Sigma_SEC = factor_model_cov(
            returns_weekly = weekly_5y, 
            factors_weekly = sector_returns_weekly, 
            use_excess = False
        )
        
        mats.append(Sigma_SEC.values)
        
        names.append("SEC")

    frames = []
    
    if ff_factors_weekly is not None:
    
        ff = ff_factors_weekly.copy()
    
        if "RF" in ff.columns:
    
            ff = ff.drop(columns = ["RF"])
    
        frames.append(ff)
    
    for fb in (index_returns_weekly, industry_returns_weekly, sector_returns_weekly):
    
        if fb is not None:
    
            frames.append(fb)

    if len(frames) == 0:
    
        Xw = weekly_5y.to_numpy(dtype = float)

        F_all = None

    else:

        idx_resid = weekly_5y.index

        for fr in frames:

            idx_resid = idx_resid.intersection(fr.index)

        if len(idx_resid) < 26:

            Xw = weekly_5y.to_numpy(dtype = float)

            F_all = None
     
        else:
     
            R_align = weekly_5y.loc[idx_resid].replace([np.inf, -np.inf], np.nan)
     
            big = {
                "RET": R_align
            }
     
            for k, fr in enumerate(frames):
            
                big[f"F{k}"] = fr.loc[idx_resid].replace([np.inf, -np.inf], np.nan)

            joint = pd.concat(big.values(), axis = 1).dropna(how = "any")
         
            n_ret = R_align.shape[1]
         
            Xw = joint.iloc[:, :n_ret].to_numpy(dtype = float) 
         
            Fblocks = []
         
            c0 = n_ret
         
            for fr in frames:
         
                k = fr.shape[1]
         
                Fblocks.append(joint.iloc[:, c0: c0 + k].to_numpy(dtype = float))
             
                c0 += k
            
            F_all = np.concatenate(Fblocks, axis = 1) if len(Fblocks) else None

    Sigma_STAT = _pca_stat_cov_on_residuals(
        X = Xw,
        F = F_all,
        K_max = 20
    )
    
    mats.append(Sigma_STAT)
    
    names.append("STAT")

    mats = [
        _clean_cov_matrix(
            M = M
        ) for M in mats
    ]

    S = _clean_cov_matrix(
        M = S_base.copy()
    )

    T_ref = _lw_reference_nan_safe(
        weekly_returns = weekly_5y.loc[:, idx],
        S_weekly = S,
        min_complete_weeks = 26,
        policy = "complete_rows"
    )
    
    T_ref = _clean_cov_matrix(
        M = T_ref
    )

    _assert_finite(
        label = "S", 
        M = S
    )
    
    _assert_finite(
        label = "T_ref", 
        M = T_ref
    )
    
    for nm, M in zip(names, mats):
       
        _assert_finite(
            label = f"Target[{nm}]", 
            M = M
        )

    m = len(mats)
  
    w = cp.Variable(m, nonneg = True)

    Sigma_w = S.copy()

    for j in range(m):

        Sigma_w = Sigma_w + w[j] * (mats[j] - S)

    Sigma_w = 0.5 * (Sigma_w + Sigma_w.T)

    lb, ub = _make_bounds_for_targets(
        names,
        w_P = w_P, 
        w_S_EWMA = w_S_EWMA, 
        w_C_EWMA = w_C_EWMA,
        w_F = w_F,
        w_FF = w_FF,
        w_IDX = w_IDX,
        w_IND = w_IND,
        w_SEC = w_SEC,
        w_STAT = w_STAT,
        p_min = p_min,
        p_max = p_max,
        s_ewma_min = s_ewma_min, 
        s_ewma_max = s_ewma_max,
        c_ewma_min = c_ewma_min, 
        c_ewma_max = c_ewma_max,
        fpred_min = fpred_min, 
        fpred_max = fpred_max,    
        ff_min = ff_min,
        ff_max = ff_max,
        idx_min = idx_min, 
        idx_max = idx_max,
        ind_min = ind_min, 
        ind_max = ind_max,
        sec_min = sec_min, 
        sec_max = sec_max,
        stat_min = stat_min, 
        stat_max = stat_max,
    )

    cons = [
        cp.sum(w) <= 1.0, 
        w >= lb
    ]
   
    finite_ub = np.isfinite(ub)
    
    if np.any(finite_ub):
    
        cons.append(w[finite_ub] <= ub[finite_ub])

    n = len(idx)

    eps_ld = float(logdet_eps)

    cons.append(Sigma_w >> eps_ld * np.eye(n))

    obj = cp.Minimize(
        cp.sum_squares(Sigma_w - T_ref) - float(tau_logdet) * cp.log_det(Sigma_w)
    )

    solver_choice = cp.MOSEK if hasattr(cp, "MOSEK") else cp.SCS

    prob = cp.Problem(obj, cons)

    try:

        if solver_choice is cp.SCS:

            prob.solve(solver = solver_choice, verbose = False, eps = 1e-5, max_iters = 20000)
        else:
            prob.solve(solver = solver_choice, verbose = False)
  
    except Exception:

        pass

    if (w.value is None) or (not np.all(np.isfinite(w.value))):

        cols = [(M - S).ravel() for M in mats]

        A = np.column_stack(cols) if len(cols) else np.zeros((S.size, 0))

        b = (T_ref - S).ravel()

        if A.shape[1] > 0:

            w0 = nnls(A, b)[0]

        else:

            w0 = np.zeros(m)

        w_hat = _project_boxed_simplex_leq(
            v = w0, 
            lb = lb,
            ub = ub, 
            s = 1.0
        )

    else:
       
        w_hat = np.clip(w.value, lb, ub)

        w_hat = _project_boxed_simplex_leq(
            v = w_hat, 
            lb = lb, 
            ub = ub, 
            s = 1.0
        )

    C_wk = S.copy()

    for wj, M in zip(w_hat, mats):

        C_wk += wj * (M - S)

    C_wk = _nearest_psd_preserve_diag(
        C = C_wk, 
        eps = 1e-10
    )

    C_ann = C_wk * float(periods_per_year)

    out = pd.DataFrame(C_ann, index = idx, columns = idx)

    remaining = float(max(0.0, 1.0 - np.sum(w_hat)))

    w_map = {nm: float(wj) for nm, wj in zip(names, w_hat)}

    w_map["S (implicit)"] = remaining

    print("Shrinkage weights (v2):", w_map)
    
    if description:
        
        desc = out.describe()

        return out, desc
    
    else:
         
        return out

