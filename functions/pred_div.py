import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


def _safe_clip_mode(
    low: float, 
    mode: float,
    high: float
) -> float:
    """
    Safely constrain a proposed PERT mode to the open interval (low, high).

    Purpose
    -------
    Many PERT parameterisations require the mode m to lie strictly between the
    minimum and maximum. This helper guards against invalid inputs (e.g. m ≤ low or
    m ≥ high) by softly clipping the proposed `mode` into (low, high) using a tiny
    ε > 0 margin.

    Parameters
    ----------
    low : float
        The lower bound of the support.
    mode : float
        The proposed mode. If not strictly inside (low, high), it will be adjusted.
    high : float
        The upper bound of the support.

    Returns
    -------
    float
        A numerically safe mode: max(low + ε, min(mode, high − ε)).

    Notes
    -----
    - Uses ε = 1×10⁻⁸ to avoid degeneracy when `mode` lands exactly on a bound.
    
    - This function does not reorder bounds
    
    - `low` is assumed < `high`.
    """

    eps = 1e-8

    if not (low < mode < high):

        mode = min(max(mode, low + eps), high - eps)

    return mode


def _beta_pert_params(
    low: float, 
    mode: float,
    high: float,
    lam: float
) -> Tuple[float, float]:
    """
    Compute Beta parameters (α, β) for a PERT distribution on [low, high].

    Model
    -----
    The classical PERT distribution is a scaled Beta on [0, 1] with shape parameter
    λ ≥ 0 that controls peakedness around the mode. Let m be the (clipped) mode and
    let span = high − low. Define the canonicalised mode in [0, 1] as
     
        m* = (m − low) / span.
    
    Then the Beta parameters are:
    
        α = 1 + λ · m*
    
        β = 1 + λ · (1 − m*)
    
    A draw U ~ Beta(α, β) is mapped to X on [low, high] by the affine transform:
    
        X = low + U · (high − low).

    Parameters
    ----------
    low : float
        Minimum of the support.
    mode : float
        Mode of the target PERT (soft-clipped into (low, high) if necessary).
    high : float
        Maximum of the support (must satisfy high > low).
    lam : float
        Peakedness parameter λ. Larger λ concentrates mass near the mode; λ = 4 is
        the standard PERT choice.

    Returns
    -------
    Tuple[float, float]
        (α, β) suitable for `np.random.Generator.beta(α, β, ...)`.

    Defensive behaviour
    -------------------
    - If `mode` is outside (low, high), it is softly clipped to avoid α or β ≤ 1.
   
    - α and β are lower-bounded at 1e−6 for numerical robustness in random samplers.
    """

    mode = _safe_clip_mode(
        low = low, 
        mode = mode,
        high = high
    )
    
    span = high - low
    
    if span <= 0:
    
        return 1e6, 1e6

    alpha = 1.0 + lam * (mode - low) / span

    beta = 1.0 + lam * (high - mode) / span

    alpha = max(alpha, 1e-6)

    beta = max(beta, 1e-6)

    return alpha, beta


def _draw_beta_pert(
    low: float,
    mode: float, 
    high: float, 
    lam: float,
    size: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Draw samples from a Beta-PERT distribution on [low, high].

    Construction
    ------------
    Given bounds (low, high), a mode m, and λ, compute (α, β) as in
    `_beta_pert_params`.
    
    Sample U ~ Beta(α, β) and linearly rescale to
        
        X = low + U · (high − low).

    Parameters
    ----------
    low : float
        Lower support bound.
    mode : float
        Mode (will be safely clipped into (low, high) inside the helper).
    high : float
        Upper support bound.
    lam : float
        Peakedness parameter λ (≥ 0).
    size : int
        Number of draws.
    rng : np.random.Generator
        Numpy random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (size,) of PERT draws on [low, high].

    Notes
    -----
    - The transformation preserves order and mean under the Beta law scaling.
   
    - For λ → 0, α → 1, β → 1 (Uniform on [low, high])
    
    - for large λ, samples concentrate around the mode.
    """

    a, b = _beta_pert_params(
        low = low, 
        mode = mode, 
        high = high, 
        lam = lam
    )

    u = rng.beta(a, b, size = size)

    return low + u * (high - low)


def _truncnorm(
    mu: float,
    sigma: float,
    low: float, 
    high: float,
    size: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample from a univariate truncated normal via rejection sampling.

    Target distribution
    -------------------
    Let Z ~ N(μ, σ²) and define the truncated variable
   
        X = Z | (low ≤ Z ≤ high).
   
    This function draws i.i.d. samples from X using simple accept-reject.

    Algorithm
    ---------
    Iteratively:
   
    1) Draw a batch B ~ N(μ, σ²).
   
    2) Keep values in [low, high].
   
    3) Append up to the remaining quota.
   
    4) Double the batch size adaptively up to a hard cap for efficiency.

    Parameters
    ----------
    mu : float
        Mean μ of the underlying (untruncated) normal.
    sigma : float
        Standard deviation σ (> 0). If σ ≤ 0, returns the clipped constant
        `clip(mu, low, high)` for all samples.
    low : float
        Lower truncation bound (inclusive).
    high : float
        Upper truncation bound (inclusive).
    size : int
        Number of samples to return.
    rng : np.random.Generator
        Numpy PRNG to use for normal draws.

    Returns
    -------
    np.ndarray
        Array of shape (size,) containing truncated normal samples.

    Complexity and caveats
    ----------------------
    - Acceptance rate equals P(low ≤ Z ≤ high) which can be small when μ is far
    outside the interval or σ is tiny; the routine adapts batch sizes, and if it
    detects pathological non-acceptance it returns the clipped constant for
    numerical safety.
   
    - This “lightweight” approach avoids external dependencies. For extreme
    truncations or heavy use, specialised samplers (e.g. inverse-CDF or
    exponential-tilting methods) are more efficient.
    """

    if sigma <= 0:

        return np.full(size, np.clip(mu, low, high))

    out = np.empty(size)
    
    filled = 0

    batch = max(256, int(size * 0.1))
    
    while filled < size:
    
        x = rng.normal(mu, sigma, size = batch)
       
        x = x[(x >= low) & (x <= high)]
       
        take = min(x.size, size - filled)
       
        if take > 0:
          
            out[filled: filled + take] = x[:take]
          
            filled += take

        if filled == 0 and batch > 1_000_000:

            return np.full(size, np.clip(mu, low, high))

        batch = min(batch * 2, 1_000_000)

    return out


def _draw_roe_from_history(
    roe_hist,                         
    size: int,
    rng: np.random.Generator,
    clip_low: float = -1.0,           
    clip_high: float = 1.0,        
    ewm_span: int = 40,
    min_sigma_frac: float = 0.10,   
    min_sigma_abs: float = 0.02       
) -> np.ndarray:
    """
    Calibrate a truncated normal ROE distribution from historical ROE.

    - Location: μ̂ = EWM_mean(roe_hist, span=ewm_span)
    - Scale:    σ̂ = max( min_sigma_abs, min_sigma_frac * |μ̂|, EWM_std(roe_hist) )
    - Draws:    ROE ~ TruncNormal(μ̂, σ̂, [clip_low, clip_high])
    """
    
    if roe_hist is None or len(roe_hist) == 0:

        mu_hat = 0.10
        
        sigma_hat = 0.05
        
    else:
        
        ewm = roe_hist.ewm(span = ewm_span, adjust = False)
        
        mu_hat = float(ewm.mean().iat[-1])
        
        s_hat = float(ewm.std().iat[-1])
        
        sigma_hat = max(min_sigma_abs, min_sigma_frac * abs(mu_hat), s_hat if np.isfinite(s_hat) else 0.0)

    roe_draws = _truncnorm(
        mu = mu_hat, 
        sigma = sigma_hat, 
        low = clip_low,
        high = clip_high,
        size = size,
        rng = rng
    )
    
    return roe_draws


def simulate_terminal_growth(
    n_sims: int,
    rng: np.random.Generator,
    g_low: Optional[float] = None,
    g_mode: Optional[float] = None,
    g_high: Optional[float] = None,
    g_lam: float = 6.0,                 
    roe_low: Optional[float] = None,
    roe_mode: Optional[float] = None,
    roe_high: Optional[float] = None,
    roe_hist = None, 
    payout_draws: Optional[np.ndarray] = None,
    g_clip_low: float = -1.0,
    g_clip_high: float = 1.0,
    roe_clip_low: float = -1.0,             
    roe_clip_high: float = 1.0
) -> np.ndarray:
    """
        Simulate terminal dividend growth rates:
    
        Route A (direct PERT):
            If g_low/mode/high are provided, draw g_term ~ PERT(g_low, g_mode, g_high, g_lam).
    
        Route B (sustainable growth):
            If roe_* bounds and payout_draws are provided, draw ROE ~ PERT(roe_low, roe_mode, roe_high)
            and set g_term = (1 - payout) * ROE. Then clip to [g_clip_low, g_clip_high].
    
        Priority: If both routes are fully specified, Route B (sustainable) is used.
        
    """
    
    if (roe_hist is not None) and (payout_draws is not None):
    
        roe_draws = _draw_roe_from_history(
            roe_hist = roe_hist, 
            size = n_sims, 
            rng = rng,
            clip_low = roe_clip_low,
            clip_high = roe_clip_high
        )
        
        g = (1.0 - np.clip(payout_draws, 0.0, 10.0)) * roe_draws
        
        return np.clip(g, g_clip_low, g_clip_high)


    can_sustainable = (
        (roe_low is not None) and (roe_mode is not None) and (roe_high is not None) and
        (payout_draws is not None)
    )

    if can_sustainable:

        roe_draws = _draw_beta_pert(
            low = roe_low, 
            mode = roe_mode,
            high = roe_high,
            lam = g_lam,
            size = n_sims,
            rng = rng
        )

        g = (1.0 - np.clip(payout_draws, 0.0, 10.0)) * roe_draws
        
        return np.clip(g, g_clip_low, g_clip_high)
        
      
    if (g_low is None) or (g_mode is None) or (g_high is None):

        g_low = 0.00
        
        g_mode = 0.02
        
        g_high = 0.04
        
    g = _draw_beta_pert(
        low = g_low, 
        mode = g_mode,
        high = g_high, 
        lam = g_lam, 
        size = n_sims,
        rng = rng
    )

    return np.clip(g, g_clip_low, g_clip_high)


@dataclass
class DivSimResult:
    """
    Container for dividend simulation outputs.

    Fields
    ------
    dps : np.ndarray
        Simulated dividends per share (DPS) path, one value per simulation.
    payout : np.ndarray
        Simulated payout ratios paired 1-to-1 with `dps` draws.
    eps : np.ndarray
        Simulated earnings per share (EPS) driving DPS via payout × max(EPS, 0).
    yield_ : np.ndarray
        Simulated dividend yields, `dps / price`.
    summary : Dict[str, float]
        Pre-computed descriptive statistics for each simulated dimension, including
        mean, median, 10th/90th percentiles, and standard deviations for DPS,
        payout, EPS, and dividend yield. Keys:
      
        - 'dps_mean', 'dps_median', 'dps_p10', 'dps_p90', 'dps_std'
      
        - 'yield_mean', 'yield_median', 'yield_p10', 'yield_p90', 'yield_std'
      
        - 'eps_mean', 'eps_p10', 'eps_p90', 'eps_std'
      
        - 'payout_mean', 'payout_p10', 'payout_p90', 'payout_std'

    Purpose
    -------
    Provides a structured return for downstream reporting and calibration, avoiding
    re-computation of summaries and facilitating serialisation.
    """

    dps: np.ndarray           
   
    payout: np.ndarray             
   
    eps: np.ndarray                
   
    yield_: np.ndarray           
    
    g_term: np.ndarray  
   
    summary: Dict[str, float]     


def div_yield(
    payout_hist,            
    price: float, 
    eps_y_high: float,
    eps_y_avg: float,
    eps_y_low: float,
    eps_s_high: float,
    eps_s_avg: float,
    eps_s_low: float,
    n_y: int,              
    n_s: int,               
    n_sims: int,
    rng: Optional[np.random.Generator] = None,
    payout_cap_high: float = 1.5,   
    payout_cap_low: float = 0.0,
    g_low: Optional[float] = None,
    g_mode: Optional[float] = None,
    g_high: Optional[float] = None,
    g_lam: float = 6.0,
    roe_low: Optional[float] = None,
    roe_mode: Optional[float] = None,
    roe_high: Optional[float] = None,
    roe_hist = None,     
    g_clip_low: float = -1.0,
    g_clip_high: float = 1.0        
) -> DivSimResult:
    """
    Simulate dividends per share (DPS) and dividend yield using a PERT-mixture EPS
    model and a truncated-normal payout process.

    Overview
    --------
    The simulator marries (i) a mixture distribution for forward EPS formed from two
    independent PERT laws—representing, for example, “year-ahead” and “street”
    sources—with (ii) a truncated normal distribution for the payout ratio. DPS is
    computed as payout × max(EPS, 0), and dividend yield as DPS / price.

    EPS component (mixture of Beta-PERT)
    ------------------------------------
    For each source j ∈ {y, s}, define:
    
    • Bounds: 
    
        (L_j, H_j) = (eps_*_low, eps_*_high).
    
    • Mode:   
    
        M_j = eps_*_avg (soft-clipped into (L_j, H_j)).

    Given a peakedness parameter λ_j ≥ 0, obtain Beta parameters:
    
        α_j = 1 + λ_j · (M_j − L_j) / (H_j − L_j)
    
        β_j = 1 + λ_j · (H_j − M_j) / (H_j − L_j)

    Draw U_j ~ Beta(α_j, β_j) and rescale to X_j = L_j + U_j · (H_j − L_j).

    Peakedness as a function of analyst coverage:
    
        λ_j = 4 + 2 · log(1 + n_j),
    
    so that larger coverage n_j concentrates mass near the mode (the provided
    average). The “standard” PERT choice is λ = 4.

    Form a convex mixture across sources with data-driven weights:
    
        w_y = n_y / (n_y + n_s)
        
        w_s = n_s / (n_y + n_s),
    
    and sample EPS from:
    
        EPS ∼ { X_y with prob. w_y;  X_s with prob. w_s }.
   
    If n_y + n_s = 0, a neutral w_y = 0.5 is used.

    Payout ratio component (truncated normal)
    -----------------------------------------
    Let {P_t} be the historical payout ratio series. Compute an exponentially
    weighted mean and standard deviation with span = 40:
    
        μ̂ = EWM_mean_40(P_t),   
        
        σ̂ = EWM_std_40(P_t).
    
    Sample payout ∼ 
    
        TruncNormal(μ = μ̂, σ = σ̂, bounds = [payout_cap_low, payout_cap_high]),
    
    using accept-reject (see `_truncnorm`). If σ̂ is missing or numerically tiny,
    stabilise via:
   
        σ̂ ← max(0.05, 0.10 · |μ̂|).

    DPS and dividend yield
    ----------------------
    Let EPS⁺ = max(EPS, 0) to avoid negative dividends mechanically arising from
    transient negative EPS. For each simulation:
    
        DPS      = payout × EPS⁺
    
        DY (yield) = DPS / price

    Inputs
    ------
    payout_hist
        A pandas Series-like time series of historical payout ratios; most recent at
        the end. If `None`, returns a zeroed result with a summary of zeros.
    price : float
        Current share price used to map DPS to dividend yield.
    eps_y_high, eps_y_avg, eps_y_low : float
        High/mode(=avg)/low triplet for the “y” EPS source.
    eps_s_high, eps_s_avg, eps_s_low : float
        High/mode(=avg)/low triplet for the “s” EPS source.
    n_y : int
        Analyst count for the “y” source, influencing λ_y and the mixture weight.
    n_s : int
        Analyst count for the “s” source, influencing λ_s and the mixture weight.
    n_sims : int
        Number of Monte Carlo replications.
    rng : Optional[np.random.Generator], default None
        PRNG; if None, `np.random.default_rng()` is used.
    payout_cap_high : float, default 1.5
        Upper bound for the truncated payout distribution (e.g. to curb windfall
        ratios).
    payout_cap_low : float, default 0.0
        Lower bound for the truncated payout distribution.

    Returns
    -------
    DivSimResult
        A dataclass with arrays of `dps`, `payout`, `eps`, `yield_`, and a `summary`
        dict containing mean/median/10th/90th percentiles and standard deviations
        for each dimension.

    Statistical properties and rationale
    ------------------------------------
    - **Mixture PERT**: 
    
        The scaled-Beta PERT provides a flexible, bounded law whose mean lies between 
        low and high and whose concentration increases with λ. By setting the mode 
        to the supplied average and linking λ to coverage, the simulator encodes greater 
        confidence as coverage deepens while respecting analyst-provided bounds.
   
    - **Truncation**: 
    
        Payout ratios are naturally bounded below at 0 and often have practical caps
        due to policy and cash-flow constraints. Truncation produces realistic tails 
        and prevents DPS explosions from outlier payout draws.
        
    - **Non-negative dividends**: 
    
        Setting DPS ∝ max(EPS, 0) reflects the typical policy of not paying dividends out of
        losses.
        
    - **Summary statistics**: 
    
        Percentiles are reported via `np.quantile` to support robust reporting (e.g. P10/P90 bands).

    Edge cases
    ----------
    - If `payout_hist` is None, the routine returns zeros and a zero summary to
    signal the absence of calibrating history.
   
    - If (n_y + n_s) = 0, mixture weights default to (0.5, 0.5).
   
    - If `eps_*_avg` lies outside its [low, high], it is softly clipped internally
    to avoid invalid Beta parameters.
    """

    if payout_hist is None:
        
        summary = {
            "dps_mean": 0,
            "dps_median": 0,
            "dps_p10": 0,
            "dps_p90": 0,
            "dps_std": 0,
            "yield_mean": 0,
            "yield_median": 0,
            "yield_p10": 0,
            "yield_p90": 0,
            "yield_std": 0,
            "eps_mean": 0,
            "eps_p10": 0,
            "eps_p90": 0,
            "eps_std": 0,
            "payout_mean": 0,
            "payout_p10": 0,
            "payout_p90": 0,
            "payout_std": 0,
        }
        
        return DivSimResult(
            dps = np.zeros(n_sims),
            payout = np.zeros(n_sims),
            eps = np.zeros(n_sims),
            yield_ = np.zeros(n_sims),
            g_term = np.zeros(n_sims),
            summary = summary
        )
        
    if rng is None:
        
        rng = np.random.default_rng()

    ewm = payout_hist.ewm(span = 40, adjust = False)
   
    exp_payout = float(ewm.mean().iat[-1])
   
    payout_std = float(ewm.std().iat[-1])

    if not np.isfinite(payout_std) or payout_std <= 1e-6:

        payout_std = max(0.05, abs(exp_payout) * 0.10)  

    lam_y = 4.0 + 2.0 * np.log1p(max(int(n_y), 0))

    lam_s = 4.0 + 2.0 * np.log1p(max(int(n_s), 0))

    eps_y_draws = _draw_beta_pert(
        low = eps_y_low, 
        mode = eps_y_avg, 
        high = eps_y_high, 
        lam = lam_y, 
        size = n_sims, 
        rng = rng
    )

    eps_s_draws = _draw_beta_pert(
        low = eps_s_low, 
        mode = eps_s_avg, 
        high = eps_s_high, 
        lam = lam_s, 
        size = n_sims, 
        rng = rng
    )

    tot = max(n_y + n_s, 0)

    if tot == 0:

        w_y = 0.5

    else:
        
        w_y = n_y / tot

    choose_y = rng.random(n_sims) < w_y

    eps_mix = np.where(choose_y, eps_y_draws, eps_s_draws)

    payout_draws = _truncnorm(
        mu = exp_payout,
        sigma = payout_std,
        low = payout_cap_low,
        high = payout_cap_high,
        size = n_sims,
        rng = rng
    )

    eps_nonneg = np.maximum(eps_mix, 0.0)

    dps = payout_draws * eps_nonneg

    div_yield_sim = dps / float(price)
    
    div_std = np.nanstd(dps)
    
    yield_std = np.nanstd(div_yield_sim)
    
    eps_std = np.nanstd(eps_mix)
    
    payout_std_emp = float(np.nanstd(payout_draws))
    
    g_term = simulate_terminal_growth(
        n_sims = n_sims, 
        rng = rng,
        g_low = g_low,
        g_mode = g_mode, 
        g_high = g_high, 
        g_lam = g_lam,
        roe_low = roe_low,
        roe_mode = roe_mode, 
        roe_high = roe_high,
        payout_draws = payout_draws,
        roe_hist = roe_hist,     
        g_clip_low = g_clip_low,
        g_clip_high = g_clip_high
    )
    
    g_std = float(np.nanstd(g_term))

    def _q(
        x,
        p
    ):
        
        return float(np.quantile(x, p))
    
    summary = {
        "dps_mean": float(np.mean(dps)),
        "dps_median": _q(
            x = dps,
            p = 0.5
        ),
        "dps_p10": _q(
            x = dps, 
            p = 0.10
        ),
        "dps_p90": _q(
            x = dps, 
            p = 0.90
        ),
        "dps_std": float(div_std),
        "yield_mean": float(np.mean(div_yield_sim)),
        "yield_median": _q(
            x = div_yield_sim, 
            p = 0.5
        ),
        "yield_p10": _q(
            x = div_yield_sim, 
            p = 0.10
        ),
        "yield_p90": _q(
            x = div_yield_sim, 
            p = 0.90
        ),
        "yield_std": float(yield_std),
        "eps_mean": float(np.mean(eps_mix)),
        "eps_p10": _q(
            x = eps_mix, 
            p = 0.10
        ),
        "eps_p90": _q(
            x = eps_mix, 
            p = 0.90
        ),
        "eps_std": float(eps_std),
        "payout_mean": float(np.mean(payout_draws)),
        "payout_p10": _q(
            x = payout_draws, 
            p = 0.10
        ),
        "payout_p90": _q(
            x = payout_draws, 
            p = 0.90
        ),
        "payout_std": float(payout_std_emp),
        "g_mean": float(np.mean(g_term)),
        "g_p10": _q(
            x = g_term, 
            p = 0.10
        ),
        "g_p90": _q(
            x = g_term, 
            p = 0.90
        ),
        "g_std": g_std,
    }

    return DivSimResult(
        dps = dps,
        payout = payout_draws,
        eps = eps_mix,
        yield_ = div_yield_sim,
        g_term = g_term, 
        summary = summary
    )
