"""
Primitives for differentiable portfolio objectives and composite optimisation.

This module provides value–and–gradient implementations for a set of common
portfolio performance measures (Sharpe ratio, Sortino ratio, information ratio,
tracking-error standard deviation) and a smoothed Score/CVaR objective based on
a convex approximation of lower-tail risk. It also includes light-weight
context/caching helpers and a composer to create linear combinations of
objectives for convex–concave procedures (CCP) or projected gradient methods.

Notation and shapes (all vectors are column-interpreted):

- w ∈ ℝ^N: portfolio weights (non-negative and summing to 1 in most use cases).

- μ ∈ ℝ^N: annualised expected returns.

- Σ ∈ ℝ^{N×N}: annualised covariance matrix (symmetric positive semi-definite).

- rf ∈ ℝ: annual risk-free rate used in Sharpe computations.

- R ∈ ℝ^{T×N}: matrix of historical returns (rows are time, columns are assets).

- b ∈ ℝ^T: per-period benchmark (or risk-free) return vector.

- scores ∈ ℝ^N: cross-sectional asset “scores” for the Score/CVaR objective.

- T ∈ ℕ: number of return observations.

- eps > 0: numerical stabiliser added inside square-roots/denominators.

Unless otherwise stated, inner products are standard Euclidean; norms are ℓ2;
and “mean” refers to arithmetic mean over time.

All gradients are with respect to w and returned in ℝ^N.
"""


import numpy as np
from dataclasses import dataclass

try:
  
    from numba import njit
  
    _HAS_NUMBA = True

except Exception:

    _HAS_NUMBA = False

    def njit(
        *args, 
        **kwargs
    ):

        def deco(
            f
        ): 
            
            return f

        return deco


@dataclass
class GradCtx:
    """
    Context object carrying model inputs and hyper-parameters for value/gradient
    evaluations.

    Fields
    ------
    mu : np.ndarray, shape (N,)
        Forecast annualised expected returns μ.
    Sigma : np.ndarray, shape (N, N)
        Forecast annualised covariance Σ.
    rf : float, default 0.0
        Risk-free rate used in Sharpe-type numerators.
    eps : float, default 1e-12
        Small positive constant added to denominators to preserve numerical stability.
    R1, R5, Rr : np.ndarray | None
        Return histories used by various objectives (e.g., 1-year or 5-year weekly).
    b1, b5 : np.ndarray | None
        Per-period benchmark vectors to form active returns y = Rw − b.
    target : float | None
        Target (e.g., weekly risk-free) for downside measures in Sortino.
    scores : np.ndarray | None
        Cross-sectional scores s used in Score/CVaR = (sᵀw) / CVaR(w).
    cvar_alpha : float, default 0.05
        Lower-tail probability level α for CVaR (e.g., 5%).
    cvar_beta : float, default 50.0
        Softplus/Logistic smoothing steepness β used in the smoothed CVaR surrogate.
    denom_floor : float, default 1e-12
        Lower bound applied to denominators such as CVaR to avoid division by zero.

    Notes
    -----
    This class is intentionally permissive: different primitives read only the
    subset of fields they require.
    """

    mu: np.ndarray            
   
    Sigma: np.ndarray         
   
    rf: float = 0.0
   
    eps: float = 1e-12
   
    R1: np.ndarray | None = None
   
    b1: np.ndarray | None = None
   
    R5: np.ndarray | None = None
   
    b5: np.ndarray | None = None
   
    Rr: np.ndarray | None = None    
   
    target: float | None = None     
   
    scores: np.ndarray | None = None 
   
    cvar_alpha: float = 0.05        
   
    cvar_beta: float = 50.0        
   
    denom_floor: float = 1e-12   
   
    
@dataclass
class Work:
    """
    Per-call working cache to avoid repeated matrix–vector products.

    Fields
    ------
    Sigma_w : np.ndarray | None
        Cache for Σw. If None, it is computed once by `ensure_work` and reused by
        objectives that require both wᵀΣw and Σw.
    """
    
    Sigma_w: np.ndarray | None = None  


def ensure_work(
    w: np.ndarray,
    ctx: GradCtx, 
    work: Work | None
) -> Work:
    """
    Ensure that a `Work` cache is available and that Σw has been computed.

    Parameters
    ----------
    w : np.ndarray, shape (N,)
        Portfolio weights.
    ctx : GradCtx
        Context carrying Σ.
    work : Work | None
        Optional working cache.

    Returns
    -------
    Work
        A cache with `Sigma_w = Σ @ w` populated.

    Notes
    -----
    Many objectives need both the quadratic form wᵀΣw and the vector Σw. Computing
    Σw once and reusing it reduces repeated cost in gradient evaluations.
    """

    if work is None:

        work = Work()

    if work.Sigma_w is None:

        work.Sigma_w = ctx.Sigma @ w

    return work


def sharpe_val_grad(
    w: np.ndarray, 
    ctx: GradCtx,
    work: Work | None = None
):
    """
    Sharpe ratio value and gradient.

    Mathematics
    -----------
    Define
    
        S(w) = ( μᵀ w − rf ) / sqrt( wᵀ Σ w + eps ).

    Let 
    
        A = μᵀ w − rf 
        
        B = wᵀ Σ w + eps. 
        
    Then
    
        S(w) = A / sqrt(B).

    Gradient derivation:
    
    - dA/dw = μ.
    
    - dB/dw = (Σ + Σᵀ) w = 2 Σ w because Σ is symmetric.
    
    - d sqrt(B)/dw = (1/(2 sqrt(B))) dB/dw = (Σ w) / sqrt(B).

    By the quotient rule,
    
        ∇S = ( sqrt(B) * μ − A * (Σ w) / sqrt(B) ) / B
           
           = μ / sqrt(B) − A * (Σ w) / B^{3/2}.

    Parameters
    ----------
    w : np.ndarray, shape (N,)
        Portfolio weights.
    ctx : GradCtx
        Uses `mu`, `Sigma`, `rf`, `eps`.
    work : Work | None
        Optional cache for Σw.

    Returns
    -------
    val : float
        S(w).
    grad : np.ndarray, shape (N,)
        ∇S(w) as given above.
    """

    work = ensure_work(
        w = w, 
        ctx = ctx, 
        work = work
    )

    A = float(ctx.mu @ w - ctx.rf)

    B = float(w @ work.Sigma_w) + ctx.eps

    sqrtB = np.sqrt(B)

    val = A / sqrtB

    grad = ctx.mu / sqrtB - (A * work.Sigma_w) / (B ** 1.5)

    return val, grad


def ir_val_grad_from(
    R: np.ndarray, 
    b: np.ndarray, 
    eps: float
):
    """
    Factory for an Information Ratio (IR) objective bound to a specific history.

    Mathematics
    -----------
    Given returns R ∈ ℝ^{T×N}, benchmark b ∈ ℝ^T, define active returns
    
        y(w) = R w − b  ∈ ℝ^T.

    Mean active return:
    
        A(w) = mean(y) = (1/T) 1ᵀ y.

    Tracking-error (RMS) with stabiliser eps:
    
        TE(w) = sqrt( mean( (y − A)^2 ) + eps )
            
               = sqrt( (1/T) || y − A 1 ||_2^2 + eps ).

    Information Ratio:
    
        IR(w) = A(w) / TE(w).

    Gradients:
    
    - dA/dw = mean(R, axis=0)  ∈ ℝ^N.
    
    - Let yc = y − A. Then TE = sqrt( (1/T) ycᵀ yc + eps ).
    
        dTE/dw = (Rᵀ yc) / (T * TE).

    By quotient rule,
    
        ∇IR = dA/TE − A/TE^2 * dTE.

    Parameters
    ----------
    R : np.ndarray, shape (T, N)
        Return matrix.
    b : np.ndarray, shape (T,)
        Benchmark vector aligned to R's rows.
    eps : float
        Positive stabiliser inside TE.

    Returns
    -------
    callable
        A function (w, ctx, work) → (val, grad) that computes IR(w) and ∇IR(w)
        using the bound (R, b, eps). The ctx/work arguments are accepted to match
        the project’s objective signature and are not required here (R and b are
        already closed over).
    """

    T = R.shape[0]
   
    R_mean = R.mean(axis = 0)  


    def _val_grad(
        w: np.ndarray, 
        ctx: GradCtx,
        work: Work | None = None
    ):
       
        y = R @ w - b                 
       
        A = y.mean()
       
        yc = y - A
       
        TE = np.sqrt((yc @ yc) / T + eps)
       
        val = A / TE
       
        dA = R_mean
       
        dTE = (R.T @ yc) / (T * TE)
       
        grad = dA / TE - (A / (TE ** 2)) * dTE
       
        return val, grad

    return _val_grad


def te_std_val_grad_from(
    R: np.ndarray, 
    b: np.ndarray, 
    eps: float
):
    """
    Factory for tracking-error (TE) standard deviation value and gradient.

    Mathematics
    -----------
    Given 
    
        y(w) = R w − b 
        
        yc = y − mean(y),
    
    define
    
        TE(w) = sqrt( (1/T) ||yc||_2^2 + eps ).

    Its gradient is
    
        ∇TE = (Rᵀ yc) / (T * TE).

    Parameters
    ----------
    R : np.ndarray, shape (T, N)
        Return matrix.
    b : np.ndarray, shape (T,)
        Benchmark vector.
    eps : float
        Positive stabiliser inside the square-root.

    Returns
    -------
    callable
        A function (w, ctx, work) → (TE(w), ∇TE(w)).
    """

    T = R.shape[0]


    def _val_grad(
        w: np.ndarray,
        ctx: GradCtx, 
        work: Work | None = None
    ):
    
        y = R @ w - b
    
        yc = y - y.mean()
    
        TE = np.sqrt((yc @ yc) / T + eps)
    
        grad = (R.T @ yc) / (T * TE)
    
        return TE, grad
   
    return _val_grad


def sortino_val_grad_from(
    R: np.ndarray,
    target: float, 
    eps: float,
    annualise_by: float = np.sqrt(52)
):
    """
    Factory for the Sortino ratio value and gradient against a fixed target.

    Mathematics
    -----------
    Let portfolio per-period returns be r_p(w) = R w ∈ ℝ^T and a scalar target τ.
    
    The Sortino ratio is
    
        Sortino(w) = ( mean(r_p) − τ ) / D(w),

    where the downside deviation is
    
        D(w) = sqrt( mean( min(0, r_p − τ)^2 ) + eps ).

    Define 
    
        y = r_p − τ 
        
    and the indicator g_i = 1 if y_i < 0, else 0.
    
    Then 
    
        min(0, y_i) = g_i * y_i 
    
        D(w) = sqrt( (1/T) ∑_i g_i y_i^2 + eps ).

    Derivatives:
    
    - Numerator A(w) = mean(r_p) − τ has gradient dA/dw = mean(R, axis=0).
    
    - For the denominator, letting G = diag(g), one obtains
    
        dD/dw = (Rᵀ (g ⊙ y)) / (T * D),
    
    where "⊙" is elementwise multiplication. If D = 0, we return zero gradient
    for numerical safety.

    Quotient rule:
    
        ∇ Sortino = dA/D − A/D^2 * dD.

    Parameters
    ----------
    R : np.ndarray, shape (T, N)
        Return matrix.
    target : float
        Target τ against which downside is computed (e.g., weekly risk-free).
    eps : float
        Positive stabiliser inside D(w).

    Returns
    -------
    callable
        A function (w, ctx, work) → (Sortino(w), ∇Sortino(w)).
    """
  
    T = R.shape[0]

    R_mean = R.mean(axis = 0)
    

    def _val_grad(
        w: np.ndarray,
        ctx: GradCtx,
        work: Work | None = None
    ):

        rp = R @ w                         

        y = rp - target
        
        g = (y < 0.0).astype(float)

        Dw_sq = float(np.mean(g * (y ** 2))) + float(eps)
        
        Dw = float(np.sqrt(Dw_sq))      

        D  = annualise_by * Dw

        if getattr(ctx, "mu", None) is not None:

            A  = float(ctx.mu @ w - ctx.rf)   

            dA = ctx.mu
            
        else:

            periods_per_year = annualise_by ** 2

            A  = float((rp.mean() - target) * periods_per_year)
        
            dA = R_mean * periods_per_year

        if Dw <= 0.0 or not np.isfinite(Dw):

            return (A / max(D, float(eps))), np.zeros_like(w, dtype = float)

        dDw = (R.T @ (g * y)) / (T * Dw)

        dD = annualise_by * dDw

        val  = A / D
      
        grad = dA / D - (A / (D ** 2)) * dD

        return float(val), np.asarray(grad, dtype = float)
    

    return _val_grad


def compose(
    terms
):
    """
    Create a linear combination of objective primitives.

    Given terms {(c_k, f_k)} with f_k : (w, ctx, work) → (v_k, g_k), this returns a
    callable that evaluates
    
        v(w)   = ∑_k c_k v_k(w),
   
        ∇v(w)  = ∑_k c_k g_k(w).

    Parameters
    ----------
    terms : list[tuple[float, callable]]
        Each item is (coefficient, objective_fn). The objective function must accept
        (w, ctx, work) and return (value, gradient).

    Returns
    -------
    callable
        A function (w, ctx, work) → (value, gradient) implementing the linear
        combination above.

    Notes
    -----
    The `work` cache is passed through unmodified to allow sharing intermediates,
    e.g., Σw.
    """

    def _val_grad(
        w: np.ndarray, 
        ctx: GradCtx, 
        work: Work | None = None
    ):
    
        total_v = 0.0
    
        total_g = np.zeros_like(w)
    
        for coef, fn in terms:
    
            v, g = fn(w, ctx, work)
    
            total_v += coef * v
    
            total_g += coef * g
    
        return total_v, total_g
    
    
    return _val_grad


def _softplus(
    x: np.ndarray, 
    beta: float
) -> np.ndarray:
    """
    Stable softplus used to smooth the hinge in the CVaR surrogate.

    Definition
    ----------
    softplus_β(x) = (1 / β) * log(1 + exp(β x)).

    Properties
    ----------
   
    - softplus_β(x) ≥ max(0, x) with equality as β → ∞.
   
    - d/dx softplus_β(x) = σ(β x), where σ is the logistic sigmoid.

    This implementation uses `logaddexp` for numerical stability.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    beta : float
        Slope parameter β.

    Returns
    -------
    np.ndarray
        softplus_β(x).
    """

    return np.logaddexp(0.0, beta * x) / beta


def _sigmoid(
    x: np.ndarray
) -> np.ndarray:
    """
    Numerically stable logistic sigmoid.

    Definition
    ----------
    σ(x) = 1 / (1 + exp(−x)).

    To avoid overflow for large |x|, inputs are clipped to [−50, 50].

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        σ(x).
    """
    
    x = np.clip(x, -50.0, 50.0)
    
    return 1.0 / (1.0 + np.exp(-x))


def _solve_z_star_softplus(
    r: np.ndarray, 
    alpha: float, 
    beta: float, 
    iters: int = 60
) -> float:
    """
    Solve the scalar minimisation that defines the smoothed CVaR surrogate.

    Problem
    -------
    Given per-period portfolio returns r ∈ ℝ^T, define
    
        φ(z; r) = z + (1/(α T)) ∑_{i=1}^T softplus_β( −(r_i + z) ).

    We seek z* = argmin_z φ(z; r). This is the smoothed version of the Rockafellar–
    Uryasev CVaR formulation with the hinge replaced by softplus_β.

    First-order condition
    ---------------------
    Let y = −z. Using d/dx softplus_β(x) = σ(β x), the optimality condition is
    
        ∑_{i=1}^T σ( β ( y − r_i ) ) = α T.

    The left-hand side is monotone in y, so bisection on y is appropriate.

    Parameters
    ----------
    r : np.ndarray, shape (T,)
        Portfolio per-period returns.
    alpha : float
        Tail probability α ∈ (0, 1).
    beta : float
        Softplus steepness β > 0.
    iters : int
        Number of bisection iterations.

    Returns
    -------
    float
        z*, the optimal auxiliary variable.
    """

    T = r.shape[0]

    lo = np.min(r) - 10.0 / max(beta, 1.0)
   
    hi = np.max(r) + 10.0 / max(beta, 1.0)

    target = alpha * T
   
    for _ in range(iters):
   
        mid = 0.5 * (lo + hi)
   
        s = _sigmoid(
            x = beta * (mid - r)
        ).sum()
   
        if s < target:
   
            lo = mid
   
        else:
   
            hi = mid
   
    y_star = 0.5 * (lo + hi)
   
    z_star = -y_star
   
    return float(z_star)


@njit(cache = True, fastmath = True)
def _zstar_and_sigmoid_nb(
    r: np.ndarray, 
    alpha: float, 
    beta: float, 
    iters: int = 60
):
    """
    Numba-accelerated solver returning z* and the final sigmoid vector.

    This implements the same bisection as `_solve_z_star_softplus`, but returns
    both:
   
    - z*  (the optimal auxiliary variable), and
    
    - s ∈ ℝ^T with s_i = σ( β ( −(r_i + z*) ) ).

    These s_i are precisely the derivatives of softplus_β( −(r_i + z) ) evaluated
    at z = z*, which are required by the envelope theorem when differentiating the
    smoothed CVaR with respect to w.

    Parameters
    ----------
    r : np.ndarray, shape (T,)
        Per-period returns.
    alpha : float
        Tail probability α.
    beta : float
        Softplus steepness β.
    iters : int
        Bisection iterations.

    Returns
    -------
    z_star : float
        The optimal z*.
    svec : np.ndarray, shape (T,)
        Sigmoid vector s at the optimum.
    """

   
    T = r.shape[0]
   
    lo = r.min() - 10.0 / (beta if beta > 1.0 else 1.0)
   
    hi = r.max() + 10.0 / (beta if beta > 1.0 else 1.0)

    target = alpha * T
  
    for _ in range(iters):
  
        mid = 0.5 * (lo + hi)
  
        ssum = 0.0
  
        for i in range(T):
  
            z = beta * (mid - r[i])  
  
            if z > 50.0:
  
                s = 1.0
  
            elif z < -50.0:
  
                s = 0.0
  
            else:
  
                s = 1.0 / (1.0 + np.exp(-z))
  
            ssum += s
  
        if ssum < target:
  
            lo = mid
  
        else:
  
            hi = mid
  
    y_star = 0.5 * (lo + hi)
  
    z_star = -y_star

    svec = np.empty(T)
  
    for i in range(T):
  
        z = beta * (-(r[i] + z_star))
  
        if z > 50.0:
  
            s = 1.0
  
        elif z < -50.0:
  
            s = 0.0
  
        else:
  
            s = 1.0 / (1.0 + np.exp(-z))
  
        svec[i] = s
  
    return z_star, svec


def score_over_cvar_val_grad(
    w: np.ndarray, 
    ctx: "GradCtx",
    work: object | None = None
):
    """
    Score-over-CVaR objective (smoothed denominator) with value and gradient.

    Objective
    ---------
    Let r(w) = R1 w ∈ ℝ^T be per-period portfolio returns and s ∈ ℝ^N be a vector
    of asset scores. Define the smoothed CVaR (Rockafellar–Uryasev form with
    softplus_β) as
    
        CVaR_smooth(w) = min_z [ z + (1/(α T)) ∑ softplus_β( −(r_i(w) + z) ) ].

    Let 
        S(w) = sᵀ w. 
        
    The objective is
    
        F(w) = S(w) / max( CVaR_smooth(w), denom_floor ).

    Denominator smoothing
    ---------------------
    The softplus produces a convex, differentiable surrogate of CVaR whose gradient
    with respect to w can be obtained via the envelope theorem:
    
        ∂/∂w CVaR_smooth(w) = (1/(α T)) ∑ softplus_β'( −(r_i + z*) ) * ( −∂r_i/∂w )
                        
                            = − (R1ᵀ s_vec) / (α T),

    where z* solves the inner minimisation and
    
        s_vec[i] = σ( β ( −(r_i + z*) ) ).

    Gradient of F
    -------------
    Write C = max( CVaR_smooth(w), denom_floor ). Then
    
        ∇F = (∇S) / C − ( S / C^2 ) * ∇C,
   
    where ∇S = s and ∇C = ∂ CVaR_smooth / ∂w (or zero if the floor is active).

    Parameters
    ----------
    w : np.ndarray, shape (N,)
        Portfolio weights.
    ctx : GradCtx
        Requires: R1 (T×N), scores (N,), cvar_alpha, cvar_beta, denom_floor.
    work : object | None
        Unused placeholder to match the common signature.

    Returns
    -------
    val : float
        F(w) = (sᵀ w) / max(CVaR_smooth(w), denom_floor).
    grad : np.ndarray, shape (N,)
        ∇F(w) as given above.

    Notes
    -----
    - Using β → ∞ recovers the non-smooth hinge; finite β trades accuracy for
    smoothness and stable gradients.
    - denom_floor prevents divisions by values arbitrarily close to zero.
    """

    if ctx.R1 is None or ctx.scores is None:
   
        raise ValueError("score_over_cvar_val_grad: ctx.R1 and ctx.scores must be set.")
   
    R = ctx.R1                    
   
    s_vec = ctx.scores             
   
    alpha = float(ctx.cvar_alpha)
   
    beta = float(ctx.cvar_beta)
    
    denom = float(ctx.denom_floor)

    r = R @ w                  
    
    T = r.shape[0]
    
    if T == 0:

        return 0.0, np.zeros_like(w)

    if _HAS_NUMBA:
     
        z_star, s = _zstar_and_sigmoid_nb(
            r = r.astype(np.float64), 
            alpha = float(alpha), 
            beta = float(beta)
        )
        
    else:
       
        z_star = _solve_z_star_softplus(
            r = r, 
            alpha = float(alpha),
            beta = float(beta)
        )
       
        s = _sigmoid(
            x = beta * (-(r + z_star))
        )

    sp = _softplus(
        x = -(r + z_star),
        beta = beta
    )
    
    cvar = z_star + (sp.sum() / (alpha * T))

    dC_dw = - (R.T @ s) / (alpha * T)

    S_num = float(s_vec @ w)
    
    dS_dw = s_vec
    
    cvar_eps = max(cvar, denom)

    val = S_num / cvar_eps
    
    grad = dS_dw / cvar_eps - (S_num / (cvar_eps ** 2)) * dC_dw

    return val, grad


def _mdp_val_grad(
    w, 
    ctx,
    work = None, 
    eps: float = 1e-12
):
    """
    Diversification Ratio (MDP proxy) value and gradient.


    Definition
    ----------
    DR(w) = (σᵀ w) / sqrt(wᵀ Σ w), where σ = sqrt(diag Σ).

    Gradient
    --------
    ∇DR = σ / D − (N / D^3) (Σ w), with N = σᵀ w, D = sqrt(wᵀ Σ w).

    Notes
    -----
    • Reuses Σw from `g.ensure_work` to avoid recomputation.
    
    • Guards denominators with `self._denom_floor`.
    """

    work = ensure_work(w, ctx, work)
    
    denom = float(ctx.denom_floor)
    
    Sigma_w = work.Sigma_w


    D = float(np.sqrt(max(float(w @ Sigma_w), denom)))
   
    sigma_vec = np.sqrt(np.maximum(np.diag(ctx.Sigma).astype(float), 0.0))
   
    N = float(sigma_vec @ w)

    val = N / D

    grad = sigma_vec / D - (N / (D ** 3)) * Sigma_w

    return float(val), np.asarray(grad, dtype = float)
