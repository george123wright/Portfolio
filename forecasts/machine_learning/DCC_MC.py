from __future__ import annotations

"""
Monte-Carlo MIDAS + Bayesian MS-AR + DCC engine

This module builds a weekly multi-asset Monte-Carlo engine driven by:

1) Macro factors aggregated from quarterly to weekly with exponential-kernel MIDAS features,

2) A per-ticker Bayesian two-state Markov-switching AR with exogenous regressors (MS-ARX),

3) A jump process whose occurrence intensity exhibits self-excitation (Hawkes-style) and jump sizes follow a Generalised Pareto tail,

4) Cross-sectional dependence across assets via a DCC(1,1) correlation process on standardised shocks,

5) A Direct Quantile Regression (Direct-QR) layer used as a distribution-calibration step to align simulated quantiles to supervised conditional quantiles.

Key modelling blocks and equations (in text form):

• MIDAS weekly macro features:

  Given quarterly stationary deltas z_q for a macro series, define weights
  
    w_j ∝ exp(−θ·j), j = 0,…,L−1,
  
  normalised to sum to 1. The MIDAS signal at quarter q is 
  
    m_q = Σ_{j=0}^{L−1} w_j z_{q−j−report_lag}.
    
  This is forward-filled to weekly dates.

• Per-ticker return model (MS-ARX):
  
  Weekly log return y_t follows a two-state Markov switching ARX(0) with exogenous design x_t
  (constant, scaled macro slice, and a jump indicator):
  
    y_t = μ_{s_t} + x_t'β + ε_t,  ε_t | s_t ~ N(0, σ^2_{s_t}),  s_t ∈ {0,1}.

  Hidden state s_t is a Markov chain with transition matrix P, where
   
    P = [[p00, 1−p00],
         [1−p11, p11]].
  
  The stationary distribution is π = [π_0, π_1] with
  
    π_0 = (1−p11) / ((1−p11) + (1−p00)),  
    
    π_1 = 1 − π_0.

  The posterior for (β, μ_0, μ_1, σ^2_0, σ^2_1, P) is sampled by Gibbs;
  β uses a Gaussian ridge prior N(0, τ^2 I) and state-specific Normal-Inverse-Gamma updates.

• DCC(1,1) correlation:

  Let u_t be standardised residual vectors across tickers. The DCC latent covariance Q_t obeys
   
    Q_t = (1 − α − β) S + α u_{t−1} u_{t−1}' + β Q_{t−1},
  
  with long-run correlation S (sample correlation). The correlation is
  
    R_t = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}.
  
  The quasi log-likelihood increment is 0.5·( log det R_t + u_t' R_t^{−1} u_t ).

• Jump process:
  
  Jump occurrences J_t ∈ {0,1} follow a two-state Markov background with emission probabilities p_emit[state],
  
  augmented by Hawkes-style self-excitation:
  
    P(J_t=1 | history, state_t) = clip( p_emit[state_t] + α·exp(−β·age_since_last_jump), 0, 1 ).
  
  Jump sizes |Z_t| have a threshold-excess model:
  
    Z_t = sign_t · (threshold + excess_t − E[excess]) where excess_t ~ Generalised Pareto(c, scale).
  
  Signs are symmetric ±1.

• Direct-QR calibration:
  
  For a forecast horizon H, Direct-QR fits conditional quantiles Q_{τ}(Y_H | features) for τ ∈ {0.05, 0.25, 0.50, 0.75, 0.95}
  via quantile regression: 
  
    minimise Σ ρ_τ(y − Xβ) + λ||β||_1.
  
  After MC simulation of the H-week cumulative return S (sum of weekly y_t),
  a piecewise-linear, monotone mapping f is built so that f(q_sim(τ_j)) = q_tgt(τ_j) at the specified knots,
  and samples are pushed through f to align simulated quantiles with Direct-QR targets.

• Output:
  
  Simulated terminal prices P_T = P_0 · exp(S) with optional clipping to bounds; report p5/p50/p95, mean returns and standard error.

Numerical notes:
  – Extensive vectorisation and preallocation are used to keep complexity manageable: O(T·N_sims·N_assets) for the MC core.
  – Stability guards: clipping of scales, tiny epsilons in denominators, and capped step sizes.

"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scipy import stats
from scipy.linalg import cholesky, inv

from joblib import Parallel, delayed

from scipy.optimize import minimize

from scipy.signal import lfilter

from sklearn.linear_model import QuantileRegressor

from arch import arch_model

from numba import njit
import hashlib


from export_forecast import export_results
from macro_data3 import MacroData
import config


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if not logger.handlers:

    _h = logging.StreamHandler()

    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(_h)


BASE_REGRESSORS: List[str] = ["Interest", "Cpi", "Gdp", "Unemp", "Balance Of Trade", "Corporate Profits", "Balance On Current Account"]

LOG_DIFF = {"cpi", "gdp", "corporate profits"}   

FORECAST_WEEKS: int = 52

N_SIMS: int = 2000

RNG_SEED: int = 42

USE_BOUNDS: bool = False

JOH_MAX_K_AR_DIFF: int = 2

JOH_BOOTSTRAP_REPS: int = 200

JOH_DET_OPTS: Tuple[str, ...] = ("nc", "c", "uc")

MN_GRID_L1 = [0.1, 0.2, 0.3, 0.5]

MN_GRID_L2 = [0.3, 0.5, 0.7]

MN_GRID_L3 = [0.5, 1.0]

MN_GRID_L4 = [10.0, 50.0, 100.0]

L1 = np.array(MN_GRID_L1)

L2 = np.array(MN_GRID_L2)

L3 = np.array(MN_GRID_L3)

L4 = np.array(MN_GRID_L4)

g1, g2, g3, g4 = np.meshgrid(L1, L2, L3, L4, indexing="ij")

G = g1.size

l1g = g1.ravel()

l2g = g2.ravel()

l3g = g3.ravel()

l4g = g4.ravel()

BVAR_P: int = 2

BVAR_USE_T: bool = True

BVAR_T_DOF: float = 7.0

BVAR_SV_LITE: bool = True

BVAR_SV_AR: float = 0.95

TVP_ENABLED: bool = False

USE_JOH_BOOTSTRAP = False

QUARTERLY_HOLD_ONLY = True

AR_LAGS_DIRECT: int = 2

DIRECT_QUANTILES: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

JUMP_Q: float = 0.97

HAWKES_ALPHA: float = 0.25

HAWKES_BETA: float = 3.0

EVT_FRACTION: float = 0.05

rng_global = np.random.default_rng(RNG_SEED)

MACRO_IS_QUARTERLY: bool = True     

WEEKS_PER_QUARTER: int = 13         

MACRO_DISAGG_NOISE: float = 0.0  

MIDAS_LAGS_Q: int = 8      

MIDAS_THETA: float = 0.5     

MIDAS_REPORT_LAG_Q: int = 1  

VOL_TARGET_WEEKLY = 0.025   

MAX_WEEKLY_STD = 0.06   

MAX_WEEKLY_DRIFT = None

JUMP_CAP = 0.10   

STEP_CAP = 0.15 

STEP_CAP_MULT = 4.0


def _macro_stationary_deltas(
    df_levels: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert macro levels to stationary deltas.

    For strictly positive level series we use log-differences:
    
        Δlog X_t = log(X_t) − log(X_{t−1}).
    
    For rate-like series we use first differences:
    
        ΔX_t = X_t − X_{t−1}.

    The function:
   
    1) forward/backward fills temporary non-positive entries for log transforms,
   
    2) computes the differences as above,
   
    3) drops initial NaNs induced by differencing.

    Parameters
    ----------
    df_levels : DataFrame
        Columns include BASE_REGRESSORS in levels at (typically) weekly dates.

    Returns
    -------
    DataFrame
        Same columns, transformed to stationary deltas and aligned to the input index, with NaNs removed.
    """

    df = df_levels[BASE_REGRESSORS].astype(float).copy()

    for c in BASE_REGRESSORS:
        
        if c in LOG_DIFF:

            s = df[c].where(df[c] > 0).ffill().bfill()

            df[c] = np.log(s).diff()

        else:

            df[c] = df[c].diff()

    return df.dropna()


def stable_hash(
    s: str
) -> int:
    """
    Hash a string to a stable 32-bit integer seed.

    Uses SHA-256(s) and keeps the first 8 hex digits as an integer, which is deterministic across runs.

    Parameters
    ----------
    s : str
        Arbitrary string.

    Returns
    -------
    int
        Deterministic integer in [0, 2^32).
    """

    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _fast_scale(
    arr: np.ndarray, 
    sc: StandardScaler,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Scale features using pre-fitted StandardScaler statistics without allocating a new scaler.

    Given scaler mean m and scale s (with small-value protection), compute
    
        z = (x − m) / s
    
    elementwise, clipping to [−8, 8] for robustness.

    Parameters
    ----------
    arr : ndarray, shape (..., k)
        Raw features.
    sc : StandardScaler
        Fitted scaler providing mean_ and scale_ of length k.
    eps : float
        Minimal scale to avoid division by tiny numbers.

    Returns
    -------
    ndarray
        Scaled features with same shape as arr.
    """

    
    arr = np.nan_to_num(arr, nan = 0.0, posinf = 0.0, neginf = 0.0)  
    
    scale = np.where(np.abs(sc.scale_) > eps, sc.scale_, 1.0)
    
    out = (arr - sc.mean_) / scale
    
    return np.clip(out, -8.0, 8.0)


def _exp_midas_weights(
    L: int, 
    theta: float
) -> np.ndarray:
    """
    Compute exponential MIDAS lag weights.

    Weights are 
    
        w_j ∝ exp(−θ·j), j = 0,…,L−1, 
    
    normalised so Σ_j w_j = 1. If Σ_j w_j = 0 numerically, fall back 
    to uniform 1 / L.

    Parameters
    ----------
    L : int
        Number of quarterly lags.
    theta : float
        Decay parameter θ > 0; larger θ emphasizes recent lags.

    Returns
    -------
    ndarray, shape (L,)
        Non-negative weights summing to 1.
    """

   
    j = np.arange(L, dtype = float)
   
    w = np.exp(-theta * j)
   
    s = w.sum()
    
    if s > 0:
   
        return (w / s)  
    
    else:
        
        return np.ones(L) / L


def _quarterly_levels_to_deltas(
    dfq_levels: pd.DataFrame
) -> pd.DataFrame:
    """
    Quarterly stationarity transform identical to the weekly transform.

    See `_macro_stationary_deltas` for equations.

    Parameters
    ----------
    dfq_levels : DataFrame
        Quarterly index with BASE_REGRESSORS in levels.

    Returns
    -------
    DataFrame
        Quarterly deltas with NaNs removed.
    """


    df = dfq_levels[BASE_REGRESSORS].astype(float).copy()

    for c in ("Cpi", "Gdp"):

        s = df[c].where(df[c] > 0).ffill().bfill()

        df[c] = np.log(s).diff()

    for c in ("Interest", "Unemp"):

        df[c] = df[c].diff()

    return df.dropna()


def build_weekly_midas_from_quarterly(
    df_quarterly: pd.DataFrame,               
    weekly_index: pd.DatetimeIndex,
    L: int = MIDAS_LAGS_Q,
    theta: float = MIDAS_THETA,
    report_lag_q: int = MIDAS_REPORT_LAG_Q,
) -> pd.DataFrame:
    """
    Build weekly MIDAS exogenous features from quarterly inputs.

    Steps:
    
    1) Resample per-quarter levels to a regular quarter-end index.
    
    2) Convert to stationary deltas at quarterly frequency (see `_quarterly_levels_to_deltas`).
    
    3) Apply an exponentially weighted MIDAS filter:
    
        m_q = Σ_{j=0}^{L−1} w_j z_{q−j−report_lag},   w_j ∝ exp(−θ·j).
    
    4) Forward-fill m_q to weekly dates and align to `weekly_index`.

    Parameters
    ----------
    df_quarterly : DataFrame
        Columns BASE_REGRESSORS and a date column 'ds' at (possibly irregular) quarterly timestamps.
    weekly_index : DatetimeIndex
        Target weekly calendar.
    L : int
        Number of quarterly lags.
    theta : float
        Exponential decay parameter.
    report_lag_q : int
        Non-neg integer; shifts the quarterly input back to avoid look-ahead.

    Returns
    -------
    DataFrame
        Weekly matrix with columns BASE_REGRESSORS and index `weekly_index`.
    """


    dfq_levels = (
        df_quarterly.set_index("ds")[BASE_REGRESSORS]
        .sort_index()
        .groupby(level = 0).mean()      
        .resample("QE").mean()
        .ffill()
    )

    dXq = _quarterly_levels_to_deltas(
        dfq_levels = dfq_levels
    )      
    
    Z = dXq.shift(report_lag_q)                         

    W = _exp_midas_weights(
        L = L, 
        theta = theta
    )
    
    M = pd.DataFrame(index = Z.index, columns = Z.columns, dtype = float)
    
    b = W[::-1].astype(float)
   
    a = np.array([1.0])
   
    Z_vals = Z.values.astype(float) 
   
    M_vals = lfilter(b, a, Z_vals, axis = 0)

    M_vals[:L-1, :] = np.nan
   
    M = pd.DataFrame(M_vals, index=Z.index, columns=Z.columns).ffill().bfill().fillna(0.0)
   
    M_w = M.resample("W-SUN").ffill().bfill()

    M_w = M_w.reindex(weekly_index).ffill().bfill() 
   
    M_w = M_w.iloc[:len(weekly_index)]
   
    M_w.index = weekly_index
    
    return M_w


def build_midas_future_exog_for_country(
    df_country_quarterly: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    n_sims: int,
    L: int = MIDAS_LAGS_Q,
    theta: float = MIDAS_THETA,
    report_lag_q: int = MIDAS_REPORT_LAG_Q,
) -> np.ndarray:
    """
    Construct future weekly exogenous MIDAS paths for a country.

    Two modes:
   
    • QUARTERLY_HOLD_ONLY = True: hold a single quarterly delta value within each forecast quarter
    (after shifting by `report_lag_q`), updating only when moving into a new quarter.
   
    • Otherwise: compute full weekly MIDAS via `build_weekly_midas_from_quarterly`.

    Output is broadcast to sims:
   
        X ∈ R^{n_sims × T × k}, k = len(BASE_REGRESSORS).

    Parameters
    ----------
    df_country_quarterly : DataFrame
        'ds' plus BASE_REGRESSORS in levels for one country.
    weekly_index : DatetimeIndex
        Weekly forecast grid.
    n_sims : int
        Number of MC scenarios to create (identical copies of the mean exog).
    L, theta, report_lag_q : see `build_weekly_midas_from_quarterly`.

    Returns
    -------
    ndarray
        Array of shape (n_sims, T, k) with weekly exogenous features.
    """

    if QUARTERLY_HOLD_ONLY:
        
        dfq = (df_country_quarterly.set_index("ds")[BASE_REGRESSORS]
               .sort_index()
               .groupby(level = 0)
               .mean()
               .resample("Q")
               .mean()
               .ffill()
               .dropna()
            )

        dXq = _quarterly_levels_to_deltas(
            dfq_levels = dfq
        ).shift(report_lag_q).dropna()
        
        dXq = dXq.copy()
        
        dXq.index = dXq.index.to_period("Q")

        q_fore = pd.PeriodIndex(weekly_index, freq="Q")

        q_unique = q_fore.unique()

        dXq_aligned = dXq.reindex(dXq.index.union(q_unique)).sort_index().ffill()

        dXq_future = dXq_aligned.loc[q_unique]

        M_w = dXq_future.loc[q_fore].copy()   

        M_w.index = weekly_index

    else:

        M_w = build_weekly_midas_from_quarterly(
            df_quarterly = df_country_quarterly,
            weekly_index = weekly_index,
            L = L, 
            theta = theta, 
            report_lag_q = report_lag_q
        )

    X = M_w.to_numpy(dtype=np.float32)

    return np.tile(X[None, :, :], (n_sims, 1, 1))


def _project_ab(
    z: np.ndarray, 
    cap: float = 0.999
) -> np.ndarray:
    """
    Project a 2-vector z = (a, b) to the non-negative wedge a ≥ 0, b ≥ 0 with a+b ≤ cap.

    If a+b ≤ cap after flooring at 0, return as is; otherwise rescale by factor cap / (a+b).

    Parameters
    ----------
    z : ndarray, shape (2,)
    cap : float
        Upper bound on a+b, typically < 1 for DCC stationarity.

    Returns
    -------
    ndarray, shape (2,)
        Projected vector.
    """

    x = np.maximum(z, 0.0)
    
    s = x.sum()
    
    if s <= cap:
    
        return x
    
    return (cap / (s + 1e-12)) * x

    
def _fit_garch_std_resid(
    y: np.ndarray,
    dist: str = "t"
) -> np.ndarray:
    """
    Fit a zero-mean GARCH(1,1) to returns and return standardized residuals.

    Model:
    
        y_t = σ_t ε_t,  ε_t ~ i.i.d. dist
    
        σ_t^2 = ω + α y_{t−1}^2 + β σ_{t−1}^2.
    
    Fit with `arch` (dist='t' by default) and compute e_t = y_t / σ_t.
    On failure, fall back to z-scores.

    Parameters
    ----------
    y : ndarray
        1D array of returns.
    dist : {'normal','t',...}
        Innovation distribution used by `arch`.

    Returns
    -------
    ndarray
        Standardised residuals e_t, same length as y.
    """

    y = np.asarray(y, dtype = float)
   
    am = arch_model(
        y, 
        mean = "Zero", 
        vol = "GARCH", 
        p = 1, 
        q = 1, 
        dist = dist,
        rescale = False
    )
    
    try:
        res = am.fit(
            disp = "off",
            update_freq = 0,
            tol = 1e-6,
            options = {
                "maxiter": 1000
            }  
        )
        
        e = res.std_resid
    
    except Exception:
        
        e = (y - np.mean(y)) / (np.std(y, ddof=1) + 1e-12)

    if np.isnan(e).any():

        first_valid = np.flatnonzero(~np.isnan(e))

        if len(first_valid):

            e[:first_valid[0]] = e[first_valid[0]]

        else:

            e = np.nan_to_num(e)

    return np.asarray(e, dtype = float)


@njit(cache = True, fastmath = True)
def _dcc_negloglik(
    alpha, 
    beta, 
    U, 
    S
):
    """
    DCC(1,1) negative quasi-log-likelihood for standardised residuals.

    Given U ∈ R^{T×k} and long-run correlation S:
   
        Q_t = (1−α−β) S + α u_{t−1} u_{t−1}' + β Q_{t−1}
    
        R_t = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}.
    
    Per-t contribution (up to constants) is:
    
        ℓ_t = 0.5 · (log det R_t + u_t' R_t^{−1} u_t).
    
    We sum from t=1 to T−1 (using u_{t−1} to update Q_t). Returns Σ ℓ_t.

    Inputs must satisfy α ≥ 0, β ≥ 0, α+β < 1; otherwise return a large penalty.

    Parameters
    ----------
    alpha, beta : float
    U : ndarray, shape (T, k)
    S : ndarray, shape (k, k)

    Returns
    -------
    float
        Total negative quasi-log-likelihood.
    """

    if alpha < 0 or beta < 0 or alpha + beta >= 0.999:

        return 1e12

    T, k = U.shape

    Qt = S.copy()

    nll = 0.0

    for t in range(1, T):

        utm1 = U[t - 1]

        for i in range(k):

            for j in range(k):

                Qt[i, j] = (1.0 - alpha - beta) * S[i, j] + alpha * utm1[i] * utm1[j] + beta * Qt[i, j]

        d = np.empty(k)

        for i in range(k):

            d[i] = np.sqrt(max(Qt[i, i], 1e-12))

        Rt = Qt.copy()

        for i in range(k):

            for j in range(k):

                Rt[i, j] = Rt[i, j] / (d[i] * d[j])

        L = np.linalg.cholesky((Rt + Rt.T) * 0.5 + 1e-9 * np.eye(k))

        z = np.linalg.solve(L, U[t])

        quad = np.dot(z, z)

        logdet = 2.0 * np.log(np.diag(L)).sum()

        nll += 0.5 * (logdet + quad)

    return nll


def _estimate_dcc_qmle(
    U: np.ndarray,
    max_iter: int = 300,
    lr_init: float = 0.2,
    tol: float = 1e-6,
    cap: float = 0.999
    ) -> Tuple[float, float, np.ndarray]:
    """
    Estimate DCC(1,1) parameters (α, β) by projected gradient on the QMLE.

    We initialise S as a near-PSD shrinkage of the sample correlation of U.
    
    Objective is f(α,β) = _dcc_negloglik(α,β; U,S).
    
    We compute finite-difference gradients, take backtracking steps, and project (α,β)
    onto a ≥ 0, b ≥ 0, a+b ≤ cap. Returns (α*, β*, S).

    Parameters
    ----------
    U : ndarray, shape (T, k)
        Standardized residuals.
    max_iter, lr_init, tol, cap : optimization controls.

    Returns
    -------
    alpha : float
    beta : float
    S : ndarray
        Long-run correlation (fixed during optimization).
    """

    C = np.corrcoef(U, rowvar = False)
   
    k = C.shape[0]
   
    S = 0.995 * ((C + C.T) * 0.5) + 0.005 * np.eye(k)


    def nll(
        ab: np.ndarray
    ) -> float:
       
        return _dcc_negloglik(
            alpha = ab[0], 
            beta = ab[1], 
            U = U.astype(np.float64), 
            S = S.astype(np.float64)
        )
        

    def grad_fd(
        ab: np.ndarray, 
        h: float = 1e-5
    ) -> np.ndarray:
    
        f0 = nll(
            ab = ab
        )
    
        g = np.zeros_like(ab, dtype = float)
       
        for i in range(2):
       
            e = np.zeros(2)
            
            e[i] = h
       
            f1 = nll(
                _project_ab(
                    z = ab + e, 
                    cap = cap
                )
            )
            
            g[i] = (f1 - f0) / h
        
        return g


    x = _project_ab(
        z = np.array([0.02, 0.95]),
        cap = cap
    )
    
    f = nll(
        ab = x
    )
    
    best_x, best_f = x.copy(), f
   
    lr = float(lr_init)

    for it in range(max_iter):
      
        g = grad_fd(
            ab = x
        )
      
        gnorm = float(np.linalg.norm(g))
       
        if gnorm < tol:
       
            break

        improved = False
       
        step = lr
       
        for _ in range(12): 
           
            cand = _project_ab(
                z = x - step * g, 
                cap = cap
            )
           
            f_cand = nll(
                ab = cand
            )

            if f_cand + 1e-10 < f:
           
                x, f = cand, f_cand
           
                improved = True
           
                if f < best_f:
           
                    best_x = x.copy()
                    
                    best_f = f
           
                break
           
            step *= 0.5

        if not improved:

            lr *= 0.5

            if lr < 1e-6:

                break

    return float(best_x[0]), float(best_x[1]), S


def _quantile_map_piecewise(
    samples: np.ndarray,      
    q_probs: np.ndarray,     
    q_sim: np.ndarray,        
    q_tgt: np.ndarray       
) -> np.ndarray:
    """
    Monotone piecewise-linear quantile mapping for distribution calibration.

    Given simulated samples S, simulated quantiles x_j = q_sim(τ_j), and target quantiles
    y_j = q_tgt(τ_j) at probabilities τ_j (sorted), construct a piecewise linear function f such that
    f(x_j) = y_j for all knots, and for s between x_j and x_{j+1}:
    
        f(s) = y_j + ( (y_{j+1} − y_j) / (x_{j+1} − x_j) ) · (s − x_j).
    
    For s below the lowest knot or above the highest knot, linearly extrapolate using the end segments.
    To ensure strict monotonicity, its enforced that x_{j+1} > x_j by tiny jitter if needed.

    Parameters
    ----------
    samples : ndarray, shape (n_sims,)
    q_probs : ndarray, shape (m,)
    q_sim : ndarray, shape (m,)
    q_tgt : ndarray, shape (m,)

    Returns
    -------
    ndarray, shape (n_sims,)
        Calibrated samples f(samples).
    """

    s = samples.astype(np.float64, copy=False)
   
    x = q_sim.astype(np.float64, copy=False)
   
    y = q_tgt.astype(np.float64, copy=False)

    for j in range(1, len(x)):
   
        if x[j] <= x[j - 1]:
   
            x[j] = x[j - 1] + 1e-12

    idx = np.searchsorted(x, s, side="right") - 1
   
    idx = np.clip(idx, 0, len(x) - 2)

    x0 = x[idx]
    
    x1 = x[idx + 1]
    
    y0 = y[idx]
    
    y1 = y[idx + 1]

    t = (s - x0) / (x1 - x0)
    
    return y0 + t * (y1 - y0)



def _estimate_hawkes_from_jumps(
    jump_ind: np.ndarray,
    beta_grid: Tuple[float, ...] = (2., 3., 4., 6., 8., 10., 12.),
    alpha_cap: float = 1.0
) -> Tuple[float, float]:
    """
    Estimate Hawkes-style excitation parameters (alpha, beta) from a binary jump series.

    Posit a logistic Bernoulli model for jump occurrence:
    
        P(J_t = 1 | history) = sigmoid( μ + α h_t ),
    
    where 
    
        h_t = Σ_{s<t} exp(−β (t−s)) J_s 
    
    is computed recursively:
    
        h_t = exp(−β) · h_{t−1} + J_{t−1}, with h_0 = 0.
    
    For each β on a grid, we minimise the Bernoulli negative log-likelihood
    
        NLL(μ,α; h) = Σ_t [ log(1 + exp(μ + α h_t)) − J_t (μ + α h_t) ]
    
    with constraint α ≥ 0 (L-BFGS-B). We select the β with the smallest NLL and
    clip α to `alpha_cap` for stability.

    Parameters
    ----------
    jump_ind : ndarray of {0,1}
    beta_grid : tuple of float
    alpha_cap : float

    Returns
    -------
    alpha_hat : float
    beta_hat : float
    """

    y = jump_ind.astype(float).ravel()
   
    T = len(y)
   
    if T < 20 or y.mean() < 1e-4:
   
        return 0.0, float(beta_grid[0])


    def build_h(
        beta: float
    )-> np.ndarray:
       
        decay = np.exp(-beta)
       
        h = np.zeros(T, dtype = float)
      
        ht = 0.0

        for t in range(1, T):

            ht = decay * ht + y[t - 1]

            h[t] = ht

        return h


    def nll_logistic(
        params: np.ndarray, 
        h: np.ndarray
    ) -> float:
    
        mu = params[0]
        
        alpha = max(params[1], 0.0)  
    
        eta = mu + alpha * h

        nll = np.sum(np.log1p(np.exp(eta)) - y * eta)

        if not np.isfinite(nll):

            return 1e12

        return float(nll)
    

    best = (np.inf, 0.0, float(beta_grid[0])) 

    for beta in beta_grid:
       
        h = build_h(
            beta = beta
        )

        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)

        mu0 = np.log(p0 / (1 - p0))

        x0 = np.array([mu0, 0.05], dtype = float)

        res = minimize(
            fun = lambda p: nll_logistic(p, h),
            x0 = x0,
            method = "L-BFGS-B",
            bounds = ((None, None), (0.0, None)),  
            options = {"maxiter": 500}
        )
       
        if res.success:
       
            mu_hat, alpha_hat = res.x[0], max(res.x[1], 0.0)
       
            nll = nll_logistic(np.array([mu_hat, alpha_hat]), h)
       
            if nll < best[0]:
       
                best = (nll, float(alpha_hat), float(beta))

    alpha_hat = min(best[1], float(alpha_cap))
    
    beta_hat = float(best[2])
    
    return alpha_hat, beta_hat


def weekly_country_macro(
    df_country: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate potentially duplicated macro observations to a unique weekly series.

    For each week (Sun-ending), average duplicates then forward-fill missing weeks.

    Parameters
    ----------
    df_country : DataFrame
        Columns ['ds'] + BASE_REGRESSORS; 'ds' may contain duplicates.

    Returns
    -------
    DataFrame
        Weekly mean and forward-filled macro levels (not differenced).
    """


    df = (df_country
          .set_index("ds")[BASE_REGRESSORS]
          .sort_index())

    df = df.groupby(level=0).mean()

    dfw = (df
           .resample("W-SUN")
           .mean()
           .ffill()
    )
    
    return dfw


@dataclass
class JumpProcess:
    """
    Two-component jump process with Markov background, Hawkes excitation, and GPD sizes.

    State dynamics:
    
        state_t ∈ {0,1} with transition matrix P (rows sum to 1).
    
    Baseline emission probability depends on state: p_emit[state_t].
    
    Self-excitation adds α·exp(−β·age_since_last_jump) to the probability (clipped to [0, 0.99]).
    
    Conditional on a jump, the excess over a threshold u has a Generalised Pareto distribution:
    
        excess ~ GPD(shape = gpd_shape, scale = gpd_scale), so jump_size = u + excess − E[u + excess].

    Attributes
    ----------
    P : ndarray (2×2)
    p_emit : ndarray (2,)
    hawkes_alpha, hawkes_beta : float
    gpd_shape, gpd_scale : float
    jump_thr : float
    evt_cutoff : float
    """

    P: np.ndarray
   
    p_emit: np.ndarray
   
    hawkes_alpha: float
   
    hawkes_beta: float
   
    gpd_shape: float
   
    gpd_scale: float
   
    jump_thr: float
    
    evt_cutoff: float
    

    def simulate_batch(
        self, 
        steps: int, 
        n_sims: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate jump indicators and absolute sizes across many paths.

        At each time t and scenario n:
        
        1) Compute Hawkes term h = α·exp(−β·age) if a prior jump exists; else 0.
        
        2) Set p_t = clip( p_emit[state] + h, 0, 0.99 ), draw J_t ~ Bernoulli(p_t).
        
        3) If J_t=1, draw excess ~ GPD(c=gpd_shape, scale=gpd_scale) and set size = threshold + excess.
        
        4) Update age (0 if jump, else +1) and transition state via P.

        Parameters
        ----------
        steps : int
        n_sims : int
        rng : np.random.Generator

        Returns
        -------
        jumps : ndarray, shape (n_sims, steps)
        sizes : ndarray, shape (n_sims, steps)
        """

        P = self.P
     
        p_emit = self.p_emit
     
        alpha = self.hawkes_alpha
     
        beta = self.hawkes_beta

        state = np.zeros(n_sims, dtype = int)
     
        age = np.full(n_sims, 1e9, dtype = np.float32)  
    
        jumps = np.zeros((n_sims, steps), dtype = np.float32)
       
        sizes = np.zeros((n_sims, steps), dtype = np.float32)

        for t in range(steps):

            hawkes_term = np.where(age < 1e8, alpha * np.exp(-beta * age), 0.0)
           
            p_t = np.clip(p_emit[state] + hawkes_term, 0.0, 0.99)

            u = rng.uniform(size = n_sims)
           
            j = (u < p_t)
           
            jumps[:, t] = j.astype(np.float32)

            n_j = int(j.sum())

            if n_j > 0:

                sizes_j = stats.genpareto.rvs(
                    c = self.gpd_shape, loc = 0.0, scale = self.gpd_scale, size = n_j, random_state = rng
                )
             
                sizes[j, t] = sizes_j.astype(np.float32)

            age = np.where(j, 0.0, age + 1.0)

            u2 = rng.uniform(size=n_sims)

            next_state = np.where(u2 < P[state, 0], 0, 1)
           
            state = next_state

        return jumps, sizes


def _fit_jump_process(
    y: pd.Series, 
) -> JumpProcess:
    """
    Fit JumpProcess parameters from historical absolute returns.

    • Jump indicator: 
    
        J_t = 1{|y_t| ≥ q_{JUMP_Q}}, where q_{JUMP_Q} is the empirical quantile.
   
    • Background Markov P is estimated by counting 0 → 0, 0 → 1, 1 → 0, 1 → 1 transitions of states_{t}=J_{t−1}.
   
    • State-conditional baseline probabilities p_emit[s] are mean(J_t | states_{t−1}=s), clipped to [1e−4, 0.5].
   
    • EVT cutoff u = q_{1−EVT_FRACTION}(|y|). Fit threshold-excess GPD to |y|−u (≥0) with loc=0.
  
    • Hawkes parameters (α,β) via `_estimate_hawkes_from_jumps`.

    Parameters
    ----------
    y : Series
        Weekly log returns.

    Returns
    -------
    JumpProcess
        Calibrated jump process.
    """

    abs_y = y.abs().values

    jump_thr = np.quantile(abs_y, JUMP_Q)

    jump_ind = (abs_y >= jump_thr).astype(int)

    states = np.concatenate([[0], jump_ind[:-1]])

    counts = np.zeros((2, 2))

    np.add.at(counts, (states[:-1], states[1:]), 1)

    P = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)

    P = np.nan_to_num(P, nan=0.5)

    p_emit = np.array([jump_ind[states == s].mean() if np.any(states == s) else 0.01 for s in (0, 1)])

    p_emit = np.clip(p_emit, 1e-4, 0.5)

    evt_cutoff = np.quantile(abs_y, 1.0 - EVT_FRACTION)

    tail = abs_y[abs_y >= evt_cutoff] - evt_cutoff

    if len(tail) < 20:

        gpd_shape = 0.2

        gpd_scale = max(np.std(abs_y) * 0.5, 1e-6)

    else:

        gpd_shape, _, gpd_scale = stats.genpareto.fit(tail, floc = 0.0)

        gpd_scale = max(gpd_scale, 1e-6)

    alpha_hat, beta_hat = _estimate_hawkes_from_jumps(
        jump_ind = jump_ind
    )

    return JumpProcess(
        P = P,
        p_emit = p_emit,
        hawkes_alpha = alpha_hat,
        hawkes_beta = beta_hat,
        gpd_shape = gpd_shape,
        gpd_scale = gpd_scale,
        jump_thr = float(jump_thr),
        evt_cutoff = float(evt_cutoff),
    )


@dataclass
class MSARPosterior:
    """
    Posterior draws from Gibbs sampler for the two-state MS-ARX model.

    Shapes
    ------
    beta_draws : (D, p)         mean effect of exogenous x_t (including constant and jump_ind)
    mu_draws   : (D, 2)         state means (μ_0, μ_1)
    sig2_draws : (D, 2)         state variances (σ_0^2, σ_1^2)
    P_draws    : (D, 2, 2)      transition matrices
    """

    beta_draws: np.ndarray       
   
    mu_draws: np.ndarray          
   
    sig2_draws: np.ndarray       
   
    P_draws: np.ndarray          


@dataclass
class MSARModel:
    """
    Point estimates and standardised residuals for MS-ARX.

    beta_mean : posterior mean of β
   
    mu_mean   : posterior mean of (μ_0, μ_1)
   
    sig2_med  : per-state posterior median variances
   
    P_mean    : posterior mean transition matrix
   
    std_resid : standardised residuals computed using the stationary mixture moments:
   
    π_0 = (1 − p11) / ((1 − p11) + (1 − p00)), π_1 = 1 − π_0,
   
    mix_var = π_0 σ_0^2 + π_1 σ_1^2,
   
    μ_mix = μ'π,
   
    std_resid_t = (y_t − μ_mix − x_t'β) / sqrt(mix_var).
    """

    beta_mean: np.ndarray

    mu_mean: np.ndarray

    sig2_med: np.ndarray

    P_mean: np.ndarray

    std_resid: np.ndarray


def _forward_backward_likelihood(
    y: np.ndarray, 
    X: np.ndarray,
    beta: np.ndarray, 
    mu: np.ndarray, 
    sig2: np.ndarray,
    P: np.ndarray,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Forward-filter backward-sample (FFBS) for hidden states in a two-state Gaussian HMM.

    Emission densities:
    
        y_t | s_t = s ~ N( μ_s + x_t'β, σ_s^2 ).
    
    Forward recursion accumulates log-α_t(s) ∝ log p(s_t=s, y_{1:t}).
   
    Backward sampling draws s_T ~ p(s_T | y_{1:T}) and then s_t ~ p(s_t | s_{t+1}, y_{1:t}).

    Parameters
    ----------
    y : ndarray, shape (T,)
    X : ndarray, shape (T,p)
    beta : ndarray, shape (p,)
    mu : ndarray, shape (2,)
    sig2 : ndarray, shape (2,)
    P : ndarray, shape (2,2)
    rng : np.random.Generator

    Returns
    -------
    ndarray, shape (T,)
        Sampled state path S.
    """

    T = len(y)
    
    S = np.empty(T, dtype = int)
   
    mean0 = mu[0] + X @ beta
   
    mean1 = mu[1] + X @ beta
   
    l0 = -0.5 * (np.log(2 * np.pi * sig2[0]) + (y - mean0) ** 2 / sig2[0])
   
    l1 = -0.5 * (np.log(2 * np.pi * sig2[1]) + (y - mean1) ** 2 / sig2[1])
   
    L = np.vstack([l0, l1]).T 
  
    logalpha = np.zeros((T, 2))
   
    logP = np.log(P + 1e-16)
   
    logpi0 = np.log(np.array([0.5, 0.5]))
   
    logalpha[0] = logpi0 + L[0]
   
    for t in range(1, T):
   
        m = logalpha[t-1][:, None] + logP
   
        logalpha[t] = L[t] + (m.max(axis = 0) + np.log(np.exp(m - m.max(axis = 0)).sum(axis = 0)))
   
    probs_T = np.exp(logalpha[-1] - logalpha[-1].max())
   
    probs_T /= probs_T.sum()
    
    if rng.uniform() < probs_T[0]:
   
        S[-1] = 0 
    
    else:
        
        S[-1] = 1
        
    for t in range(T - 2, -1, -1):
       
        logp = logalpha[t] + logP[:, S[t+1]]
      
        p = np.exp(logp - logp.max())
        
        p /= p.sum()
        
        if rng.uniform() < p[0]:
      
            S[t] = 0  
        
        else:
            
            S[t] = 1
            
    return S


def _gibbs_msar(
    y: np.ndarray,
    X: np.ndarray, 
    rng: np.random.Generator,
    draws: int = 1000,
    burn: int = 500,
) -> MSARPosterior:
    """
    Gibbs sampler for the two-state MS-ARX(0) with Gaussian emissions and ridge prior on β.

    At each iteration:
    
    1) Sample S via FFBS using current parameters.
    
    2) Conditional on S, draw β from a Gaussian posterior:
    
        Prior β ~ N(0, τ^2 I).
    
        Let 
            
            w_t = 1 / σ_{S_t}^2, 
        
            ỹ_t = y_t − μ_{S_t}.
    
        Posterior precision: 
            
            V_n^{-1} = τ^{−2} I + X' diag(w) X,
        
        Posterior mean: 
        
            m_n = V_n X' (w ⊙ ỹ).
    
    3) For s ∈ {0,1}, update σ_s^2 from Inverse-Gamma:
    
        σ_s^2 ~ IG(a0 + n_s/2
        
        b0 + 0.5 Σ_{t:S_t=s} (y_t − x_t'β − μ_s)^2).
    
    4) Update μ_s from Normal with variance (n_s/σ_s^2 + 1/(c0 σ_s^2))^{−1}.
    
    5) Update each row of P from a Dirichlet over transition counts.

    Burn the first `burn` draws; keep D = draws − burn.

    Parameters
    ----------
    y : ndarray (T,)
    X : ndarray (T,p)
    rng : np.random.Generator
    draws, burn : int

    Returns
    -------
    MSARPosterior
        Posterior draws for (β, μ, σ^2, P).
    """
  
    if rng is None:
    
        rng = np.random.default_rng()

    T, p = X.shape

    tau2 = 100.0

    a0 = 2.0
    
    b0 = 0.1
    
    c0 = 10.0
    
    beta = np.zeros(p)
    
    mu = np.array([y.mean() - 0.1, y.mean() + 0.1])
    
    sig2 = np.array([np.var(y), np.var(y)])
   
    P = np.array([
        [0.95, 0.05],
        [0.10, 0.90]
    ])

    D = draws - burn
   
    beta_draws = np.empty((D, p))
   
    mu_draws = np.empty((D, 2))
   
    sig2_draws = np.empty((D, 2))
   
    P_draws = np.empty((D, 2, 2))

    for it in range(draws):  

        S = _forward_backward_likelihood(
            y = y,
            X = X, 
            beta = beta, 
            mu = mu, 
            sig2 = sig2,
            P = P,
            rng = rng
        )

        mean_S = mu[S]
       
        y_tilde = y - mean_S

        w = 1.0 / sig2[S]

        Xw = X * w[:, None]         

        Vn_inv = (np.eye(p) / tau2) + (X.T @ Xw)

        Vn = inv(Vn_inv)

        bn = X.T @ (w * y_tilde)

        beta = rng.multivariate_normal(mean=Vn @ bn, cov=Vn)

        for s in (0, 1):
          
            ys = y[S == s] - X[S == s] @ beta
          
            ns = max(1, len(ys))
          
            a_post = a0 + ns/2
          
            b_post = b0 + 0.5 * np.sum(ys**2)
          
            sig2[s] = 1.0 / rng.gamma(shape = a_post, scale = 1.0 / b_post)
          
            var_mu = 1.0 / (ns / sig2[s] + 1.0 / (c0 * sig2[s]))
          
            mean_mu = var_mu * np.sum(ys) / sig2[s]
          
            mu[s] = rng.normal(mean_mu, np.sqrt(var_mu))

        counts = np.zeros((2, 2))
       
        np.add.at(counts, (S[:-1], S[1:]), 1)
       
        P[0] = rng.dirichlet(alpha = counts[0] + 1.0)
       
        P[1] = rng.dirichlet(alpha = counts[1] + 1.0)

        if it >= burn:
       
            j = it - burn
       
            beta_draws[j] = beta
       
            mu_draws[j] = mu
       
            sig2_draws[j] = sig2
       
            P_draws[j] = P

    return MSARPosterior(
        beta_draws = beta_draws,
        mu_draws = mu_draws, 
        sig2_draws = sig2_draws,
        P_draws = P_draws
    )


def fit_bayesian_msar(
    y: pd.Series,
    X: np.ndarray, 
    rng: np.random.Generator,
    draws: int = 1200, 
    burn: int = 600,
) -> MSARModel:
    """
    Fit the MS-ARX model via Gibbs, then form point estimates and standardised residuals.

    Steps:
    
    • Run `_gibbs_msar` to obtain posterior draws.
    
    • β̂ = E[β], μ̂ = E[μ], σ̂_s^2 = median[σ_s^2], P̂ = E[P].
    
    • Stationary weights:
    
        p01 = 1 − P̂_{00},
        
        p10 = 1 − P̂_{11},
    
        π_0 = p10 / (p01 + p10),  
        
        π_1 = 1 − π_0.
    
    • Mixture variance: mix_var = π_0 σ̂_0^2 + π_1 σ̂_1^2.
    
    • Mixture mean: μ_mix = μ̂' π.
    
    • Standardised residuals: (y_t − μ_mix − x_t'β̂) / sqrt(mix_var).

    Parameters
    ----------
    y : Series
    X : ndarray (T,p)
    rng : np.random.Generator
    draws, burn : int

    Returns
    -------
    MSARModel
        Point estimates and residuals.
    """

    post = _gibbs_msar(
        y = y.values,
        X = X, 
        draws = draws, 
        burn = burn,
        rng = rng
    )

    beta_mean = post.beta_draws.mean(axis = 0)

    mu_mean = post.mu_draws.mean(axis = 0)

    sig2_med = np.median(post.sig2_draws, axis = 0)
    
    hist_var = np.var(y.values, ddof = 1)
    
    sig2_med = np.clip(sig2_med, 1e-8, 4.0 * hist_var)

    P_mean = post.P_draws.mean(axis = 0)

    mean_all = (X @ beta_mean)

    p01 = 1.0 - P_mean[0, 0]
   
    p10 = 1.0 - P_mean[1, 1]
   
    den = max(p01 + p10, 1e-12)
   
    pi0 = p10 / den
    
    pi0 = float(np.clip(pi0, 0.0, 1.0))
   
    pi1 = 1.0 - pi0
    
    pi = np.array([pi0, pi1], dtype=float)
   
    mix_var = pi0 * sig2_med[0] + pi1 * sig2_med[1]
    
    mu_mix = float(mu_mean @ pi)

    resid_std = (y.values - (mu_mix + mean_all)) / np.sqrt(mix_var + 1e-9)

    return MSARModel(
        beta_mean = beta_mean,
        mu_mean = mu_mean, 
        sig2_med = sig2_med, 
        P_mean = P_mean, 
        std_resid = resid_std
    )


@dataclass
class DirectQRModel:
    """
    Direct quantile regression container for a given horizon h.

    For each τ in DIRECT_QUANTILES, stores a fitted QuantileRegressor on
    design [1, scaled lags of y, macro deltas, jump indicator].

    Attributes
    ----------
    h : int
    scaler : StandardScaler
    models : dict[float -> QuantileRegressor]
    cols : list[str]
    """
   
    h: int
   
    scaler: StandardScaler
   
    models: Dict[float, any]    
   
    cols: List[str]


def fit_direct_qr(
    df: pd.DataFrame, 
    horizons: List[int], 
    quantiles: Tuple[float, ...]
) -> Dict[int, DirectQRModel]:
    """
    Fit Direct-QR models for specified horizons and quantiles.

    Target for horizon H is the rolling H-week sum:
    
        Y_H(t) = Σ_{i=0}^{H−1} y_{t+i}.
    
    Features:
        [1, y_{t−1}, …, y_{t−AR_LAGS_DIRECT}, macro_t, jump_ind_t] scaled by StandardScaler.
   
    For each τ, solve:
    
        β̂_τ = argmin_β Σ ρ_τ( Y_H − Xβ ) + α ||β||_1,
    
    where 
    
        ρ_τ(u) = u·(τ − 1{u<0}). 
    
    Set a small L1 penalty for numerical stability.

    Parameters
    ----------
    df : DataFrame with columns ['y'] + BASE_REGRESSORS + ['jump_ind']
    horizons : list of int
    quantiles : tuple of float

    Returns
    -------
    dict[int -> DirectQRModel]
        One entry per horizon with fitted models and the scaler.
    """
  
    base = df.copy()
  
    for L in range(1, AR_LAGS_DIRECT + 1):
  
        base[f"y_lag{L}"] = base["y"].shift(L)
  
    base.dropna(inplace = True)
   
    X_cols = [f"y_lag{L}" for L in range(1, AR_LAGS_DIRECT + 1)] + BASE_REGRESSORS + ["jump_ind"]

    out: Dict[int, DirectQRModel] = {}
   
    for H in horizons:
   
        yH = base["y"].rolling(H).sum().shift(-H + 1)
   
        tmp = base.copy()
   
        tmp["yH"] = yH
   
        tmp.dropna(inplace = True)
   
        if len(tmp) < 40:
   
            continue
   
        sc = StandardScaler().fit(tmp[X_cols].values)
   
        X = sc.transform(tmp[X_cols].values)
   
        X = np.column_stack([np.ones(len(X)), X])  
   
        models = {}
   
        for q in quantiles:
   
            mdl = QuantileRegressor(
                quantile = q, 
                alpha = 1e-4, 
                fit_intercept = False
            )
            
            mdl.fit(X, tmp["yH"].values)
            
            models[q] = mdl
   
        out[H] = DirectQRModel(
            h = H, 
            scaler = sc, 
            models = models, 
            cols = X_cols
        )
   
    return out


@dataclass
class DCCModelWrap:
    """
    Wrapper for DCC(1,1) parameters and standardised residuals.

    alpha, beta : scalar DCC parameters
    S : long-run correlation matrix
    L0 : Cholesky of S (for seeding)
    std_resids : historical standardised residuals used in fitting
    tickers : list of asset identifiers
    """
    
    alpha: float
    
    beta: float
    
    S: np.ndarray        
    
    L0: np.ndarray       
    
    std_resids: np.ndarray  
    
    tickers: List[str]


def fit_dcc_from_returns(
    returns_map: Dict[str, np.ndarray]
) -> DCCModelWrap:
    """
    Fit per-ticker univariate GARCH(1,1) to obtain standardised residuals, then DCC(1,1).

    Steps:
    
    1) For each ticker k, fit 
        
            y_{k,t} = σ_{k,t} ε_{k,t}
        
            σ_{k,t}^2 = ω + α y_{k,t−1}^2 + β σ_{k,t−1}^2,
        
        and compute 
        
            e_{k,t} = y_{k,t} / σ_{k,t}.
    
    2) Stack the last T common residuals to form U ∈ R^{T×K}.
    
    3) Estimate (α,β) by `_estimate_dcc_qmle` and set S to the shrunk sample correlation.

    Returns
    -------
    DCCModelWrap
        Calibrated DCC with standardised residual history.
    """

    tickers = list(returns_map.keys())

    std_e: Dict[str, np.ndarray] = {}

    lengths = []

    for tk in tickers:

        y = np.asarray(returns_map[tk], dtype = float)

        e = _fit_garch_std_resid(
            y = y, 
            dist = "t"
        )

        std_e[tk] = e

        lengths.append(len(e))

    mlen = int(np.min(lengths))

    U = np.column_stack([std_e[tk][-mlen:] for tk in tickers]).astype(float)   

    U = np.clip(U, -4, 4)

    alpha, beta, S = _estimate_dcc_qmle(
        U = U
    )

    L0 = cholesky(S + 1e-9 * np.eye(S.shape[0]), lower = True)

    return DCCModelWrap(
        alpha = float(alpha), 
        beta = float(beta),
        S = S,
        L0 = L0,
        std_resids = U,
        tickers = tickers
    )


def build_shock_stream(
    dccw: Optional[DCCModelWrap],
    full_tickers: List[str],
    T: int,
    n_sims: int,
    rng: np.random.Generator,
    dtype = np.float32
):
    """
    Create a generator of cross-sectional shocks per time step.

    If a DCC model is provided and the requested tickers are a subset of its fitted
    universe, shocks for those tickers follow DCC-correlated N(0,R_t); missing tickers
    receive iid N(0,1) shocks. Otherwise, all shocks are iid.

    Parameters
    ----------
    dccw : DCCModelWrap or None
    full_tickers : list[str]
    T : int
    n_sims : int
    rng : np.random.Generator
    dtype : np.dtype

    Returns
    -------
    callable
        A zero-arg generator that yields arrays of shape (n_sims, len(full_tickers)) at each t.
    """

   
    k_full = len(full_tickers)

    if (dccw is None) or (not hasattr(dccw, "tickers")) or (len(dccw.tickers) == 0):
     
     
        def _iid():
          
            for _ in range(T):
         
                yield rng.standard_normal((n_sims, k_full)).astype(dtype)
    
        return _iid


    pos = {tk: i for i, tk in enumerate(dccw.tickers)}
   
    mask = np.array([tk in pos for tk in full_tickers], dtype = bool)
   
    idx = np.array([pos[tk] for tk in full_tickers if tk in pos], dtype = int)


    def _gen():
        
        dcc_gen = simulate_dcc_shocks(
            dccw = dccw, 
            T = T,
            n_sims = n_sims, 
            rng = rng, 
            dtype = dtype
        )
       
        for U_small in dcc_gen:
       
            U = rng.standard_normal((n_sims, k_full)).astype(dtype) 
       
            if idx.size:
       
                U[:, mask] = U_small[:, idx]                    
       
            yield U
            
    
    return _gen


def simulate_dcc_shocks(
    dccw: DCCModelWrap,
    T: int,
    n_sims: int,
    rng: np.random.Generator,
    dtype=np.float32,
):
    """
    Simulate DCC(1,1) shocks over T steps for n_sims scenarios.

    We maintain a single Qt and Rt per t (using the mean outer product across sims):
    
        Q_t = (1−α−β) S + α E[U_{t−1} U_{t−1}'] + β Q_{t−1}.
    
    Given 
    
        Rt = corr(Q_t), draw Z ~ N(0, I) and set U_t = Z L', where L is the Cholesky of Rt.

    Parameters
    ----------
    dccw : DCCModelWrap
    T : int
    n_sims : int
    rng : np.random.Generator
    dtype : dtype

    Yields
    ------
    ndarray, shape (n_sims, k)
        Simulated standardised shocks for time t.
    """
  
    k = dccw.S.shape[0]
  
    S = dccw.S.astype(dtype)
  
    Qt = S.copy() 
  
    alpha = np.float32(dccw.alpha)
  
    beta = np.float32(dccw.beta)
  
    one_minus = np.float32(1.0 - alpha - beta)
  
    eye = np.eye(k, dtype = dtype)

    eps_prev_mean_outer = np.eye(k, dtype = dtype)

    for _ in range(T):

        Qt = one_minus * S + alpha * eps_prev_mean_outer + beta * Qt

        d = np.sqrt(np.clip(np.diag(Qt), 1e-12, None))
       
        Dinv = 1.0 / d
       
        Rt = Qt * (Dinv[:, None] * Dinv[None, :])

        L = cholesky((Rt + Rt.T) * 0.5 + 1e-9 * eye, lower = True)
       
        Z = rng.standard_normal((n_sims // 2, k)).astype(dtype)
       
        Z = np.vstack([Z, -Z])
        
        if Z.shape[0] < n_sims:  
        
            Z = np.vstack([Z, rng.standard_normal((1, k)).astype(dtype)])
        
        rng.shuffle(Z, axis=0)
        
        U_t = Z @ L.T

        eps_prev_mean_outer = (U_t.T @ U_t) / np.float32(n_sims)

        yield U_t


def build_features_for_ticker(
    dfp: pd.DataFrame, 
    dfm_levels: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct per-ticker modelling frame: returns, aligned macro deltas, and jump indicator.

    Steps:
    
    1) y_t = log(price_t) − log(price_{t−1}).
    
    2) Build weekly macro deltas: if `MACRO_IS_QUARTERLY` use MIDAS from quarterly levels,
        else apply weekly stationarity transform.
   
    3) Align on weekly index and drop NaNs.
   
    4) Jump indicator: J_t = 1{|y_t| ≥ q_{JUMP_Q}(|y|)}.

    Parameters
    ----------
    dfp : DataFrame
        Columns ['ds','price'] at weekly dates.
    dfm_levels : DataFrame
        Macro levels (weekly or quarterly depending on settings).

    Returns
    -------
    DataFrame
        Index weekly 'ds'; columns: ['y'] + BASE_REGRESSORS + ['jump_ind'].
    """

    dfp = dfp.copy()

    dfp["y"] = np.log(dfp["price"]).diff()

    dfp.dropna(inplace = True)

    weekly_idx = pd.DatetimeIndex(dfp["ds"])
    
    if MACRO_IS_QUARTERLY:
    
        dX = build_weekly_midas_from_quarterly(
            df_quarterly = dfm_levels if "ds" in dfm_levels.columns else dfm_levels.reset_index(),
            weekly_index = weekly_idx,
            L = MIDAS_LAGS_Q,
            theta = MIDAS_THETA,
            report_lag_q = MIDAS_REPORT_LAG_Q,
        )

    else:
      
        dX = _macro_stationary_deltas(
            df_levels = dfm_levels
        ).reindex(weekly_idx).ffill().dropna()

    dfr = dfp.set_index("ds").join(dX, how = "inner").dropna()

    thr = dfr["y"].abs().quantile(JUMP_Q)

    dfr["jump_ind"] = (dfr["y"].abs() >= thr).astype(float)

    return dfr


@dataclass
class ReturnModels:
    """
    Container for all per-ticker fitted components.

    Attributes
    ----------
    msar : MSARModel
    direct_qr : dict[int -> DirectQRModel]
    jump : JumpProcess
    scaler_macro : StandardScaler or _NullScaler
    macro_cols : list[str]
    qr_raw_latest : dict[str,float]
        Latest raw features for Direct-QR inference (lags, macro values, jump_ind).
    """
    
    msar: MSARModel
    
    direct_qr: Dict[int, DirectQRModel]
    
    jump: JumpProcess
    
    scaler_macro: StandardScaler
    
    macro_cols: List[str]
    
    qr_raw_latest: Dict[str, float]   


class _NullScaler:
    """
    No-op scaler exposing the same interface as StandardScaler for zero-column cases.

    transform(X) returns X unchanged; mean_ and scale_ are empty arrays.
    """
    
    mean_ = np.array([], dtype = float)
    
    scale_ = np.array([], dtype = float)
    
    
    def transform(
        self, 
        X
    ): 
        return X
    

def fit_return_models(
    dfr: pd.DataFrame, 
    horizons: List[int],
    rng: Optional[np.random.Generator] = None
) -> ReturnModels:
    """
    Fit all per-ticker models: MS-ARX, Direct-QR, and JumpProcess.

    Macro preprocessing:
    
    • Drop macro columns with near-zero variance to avoid numerical issues.
    
    • Fit a StandardScaler on the remaining macro columns and clip scaled values to [−8, 8].
    
    Design for MS-ARX:
    
        X_ms = [1, scaled_macro_subset, jump_ind].
    
    Gibbs sampling produces MSARModel; Direct-QR is fitted for the requested horizons;
    JumpProcess is learned from |y| with EVT and Hawkes components.
    Also stores `qr_raw_latest` (latest lags, macro and jump_ind) for prediction/calibration.

    Parameters
    ----------
    dfr : DataFrame
        Output of `build_features_for_ticker`.
    horizons : list[int]
    rng : np.random.Generator or None

    Returns
    -------
    ReturnModels
        Calibrated components for this ticker.
    """
   
    macro_cols = BASE_REGRESSORS
    
    macro_vals = dfr[macro_cols].values
    
    var = macro_vals.var(axis = 0)
    
    keep = var > 1e-8
    
    macro_cols_used = [c for c, k in zip(macro_cols, keep) if k]
   
    if len(macro_cols_used) == 0:

        sc_macro = _NullScaler()
       
        Xm = np.zeros((len(dfr), 0), dtype = float)
  
    else:
        sc_macro = StandardScaler().fit(dfr[macro_cols_used].values)
        
        Xm = np.clip(sc_macro.transform(dfr[macro_cols_used].values), -8.0, 8.0)
     
    X_ms = np.column_stack([
        np.ones(len(Xm)),           
        Xm,                        
        dfr["jump_ind"].values      
    ])
   
    y = dfr["y"]
   
    msar = fit_bayesian_msar(
        y = y, 
        X = X_ms, 
        draws = 1200, 
        burn = 600,
        rng = rng
    )

    direct_qr = fit_direct_qr(
        df = dfr[["y"] + BASE_REGRESSORS + ["jump_ind"]].copy(),
        horizons = horizons,
        quantiles = DIRECT_QUANTILES
    )

    jump = _fit_jump_process(
        y = y
    )
    
    qr_raw_latest: Dict[str, float] = {}

    y_vals = dfr["y"].values

    for L in range(1, AR_LAGS_DIRECT + 1):
        
        qr_raw_latest[f"y_lag{L}"] = float(y_vals[-L]) if len(y_vals) >= L else 0.0


    for c in BASE_REGRESSORS:

        if c in dfr.columns :
            
            qr_raw_latest[c] = float(dfr[c].iloc[-1])
        
        else:
            
            qr_raw_latest[c] = 0.0

    qr_raw_latest["jump_ind"] = float(dfr["jump_ind"].iloc[-1]) if "jump_ind" in dfr.columns else 0.0
   
    return ReturnModels(
        msar = msar,
        direct_qr = direct_qr,
        jump = jump,
        scaler_macro = sc_macro,      
        macro_cols = macro_cols_used,
        qr_raw_latest = qr_raw_latest      
    )


def simulate_return_paths_joint(
    ret_models: Dict[str, ReturnModels],
    exog_by_ticker: Dict[str, np.ndarray], 
    cp_vec: pd.Series,
    lb_vec: pd.Series,
    ub_vec: pd.Series,
    forecast_period: int,
    n_sims: int,
    rng: np.random.Generator,
    dccw: Optional[DCCModelWrap],
    tickers_order: List[str],
    hist_weekly_std_map: Optional[Dict[str, float]] = None, 
    scale_mode: str = "down",  
) -> Dict[str, Dict[str, float]]:
    """
    Simulate terminal prices jointly across tickers with DCC shocks, MS-ARX states, jumps, and Direct-QR calibration.

    At a high level, for each t = 1..T:
    
    1) Exogenous design X_full[:,:,t,:] = [1, scaled_macro, jump_ind].
    
    2) Markov states per ticker evolve with P_mean; we simulate s_t using pstay = diag(P_mean).
    
    3) MS-ARX conditional mean: μ_design = X·β, plus state mean μ_{s_t}.
    
    4) Conditional std: sqrt(σ^2_{s_t}) scaled by γ to match historical weekly std if requested.
    
    5) Correlated shocks ε_t drawn via the DCC stream; add jump increments governed by JumpProcess.
    
    6) Increment: 
    
        Δlog P = μ_design + μ_{s_t} + ε_t·std + jump.
    
    7) Cap per-step increments and accumulate over time: S = Σ Δlog P.

    After the T-week horizon (H=T), apply Direct-QR distribution calibration per ticker:
    
    • Compute simulated quantiles q_sim(τ_j) for the sum S.
    
    • Predict target quantiles q_tgt(τ_j) with Direct-QR.
    
    • Map S → f(S) using `_quantile_map_piecewise` so that quantiles match at knots.

    Finally, form terminal prices:
    
        P_T = P_0 · exp(S), clip to [lb, ub] if USE_BOUNDS, then report p5/p50/p95, mean returns, and SE.

    Shapes
    ------
    ret_models : dict[ticker -> ReturnModels]
    exog_by_ticker : dict[ticker -> ndarray], each of shape (n_sims, T, k_reg)
    cp_vec, lb_vec, ub_vec : pd.Series of current price and optional bounds
    forecast_period : int = T
    n_sims : int
    rng : np.random.Generator
    dccw : DCCModelWrap or None
    tickers_order : list[str]
    hist_weekly_std_map : dict[ticker -> float] or None
    scale_mode : {'down','match'}

    Returns
    -------
    dict[ticker -> dict[str,float]]
        For each ticker: {'low','avg','high','returns','se'}.
    """

    dtype = np.float32
  
    T = int(forecast_period)
  
    if T <= 0:
  
        return {}

    k_reg = len(BASE_REGRESSORS)

    sim_tickers: List[str] = []

    for tkr in tickers_order:

        if (
            (tkr in ret_models) and
            (tkr in exog_by_ticker) and
            np.isfinite(float(cp_vec.get(tkr, np.nan)))
        ):

            sim_tickers.append(tkr)

    if not sim_tickers:

        return {}

    kx = len(sim_tickers)

    X_macro_list: List[np.ndarray] = []

    B_stack_list: List[np.ndarray] = [] 

    for tkr in sim_tickers:
       
        ex_raw = exog_by_ticker[tkr]  
       
        if ex_raw.ndim != 3 or ex_raw.shape[0] != n_sims or ex_raw.shape[2] != k_reg:
       
            raise ValueError(
                f"exog_by_ticker[{tkr}] has shape {ex_raw.shape}; "
                f"expected (n_sims={n_sims}, T, k_reg={k_reg})"
            )
       
        if ex_raw.shape[1] < T:
   
            raise ValueError(
                f"exog_by_ticker[{tkr}] has T={ex_raw.shape[1]} < forecast_period={T}"
            )
   
        if ex_raw.shape[1] > T:
         
            ex_raw = ex_raw[:, :T, :]

        cols_used = ret_models[tkr].macro_cols 
       
        if not cols_used:

            X_full = np.zeros((n_sims, T, k_reg), dtype=dtype)

        else:

            idx = np.array([BASE_REGRESSORS.index(c) for c in cols_used], dtype = int)

            Xsub = ex_raw[:, :, idx] 
            
            Xsub = np.where(np.isfinite(Xsub), Xsub, 0.0)

            sc = ret_models[tkr].scaler_macro  

            Xm = _fast_scale(
                arr = Xsub.reshape(-1, len(idx)),
                sc = sc
            ).reshape(n_sims, T, len(idx))
          
            Xm = np.clip(Xm, -8.0, 8.0)

            X_full = np.zeros((n_sims, T, k_reg), dtype=dtype)
         
            X_full[:, :, idx] = Xm.astype(dtype, copy=False)

        X_macro_list.append(X_full)

        b = ret_models[tkr].msar.beta_mean  

        expected_len = 1 + (len(cols_used) if cols_used else 0) + 1
        
        if b.shape[0] != expected_len:
        
            raise ValueError(
                f"{tkr}: beta_mean length {b.shape[0]} != expected {expected_len} "
                f"(1 + len(cols_used) + 1). cols_used={cols_used}"
            )

        b_full = np.zeros(1 + k_reg + 1, dtype=dtype)
       
        b_full[0] = b[0]                         
       
        if cols_used:

            macro_block = b[1:1 + len(cols_used)]

            b_full[1:1 + k_reg][idx] = macro_block

        b_full[-1] = b[-1]                       

        B_stack_list.append(b_full)

    X_macro_stack = np.stack(X_macro_list, axis = 0)         
   
    B_stack = np.stack(B_stack_list, axis=0)              

    jump_inds_list: List[np.ndarray] = []

    jump_adds_list: List[np.ndarray] = []
    
    sign_choices = np.array([-1.0, 1.0], dtype=dtype)

    for tkr in sim_tickers:
       
        jp = ret_models[tkr].jump
       
        ji, sizes_exc = jp.simulate_batch(
            steps = T,
            n_sims = n_sims,
            rng = rng
        )  
        
        signs = rng.choice(sign_choices, size=ji.shape)
        
        if (jp.gpd_shape < 1.0):

            mean_excess = (jp.gpd_scale / (1.0 - jp.gpd_shape)) 
        
        else:
            
            mean_excess = jp.gpd_scale
        
        jump_mean = jp.evt_cutoff + mean_excess
        
        js = ji * (jp.evt_cutoff + sizes_exc - jump_mean) * signs
        
        js = np.clip(js, -np.float32(JUMP_CAP), np.float32(JUMP_CAP)).astype(dtype, copy = False)

        jump_inds_list.append(ji.astype(dtype, copy = False))
       
        jump_adds_list.append(js)

    jump_inds_stack = np.stack(jump_inds_list, axis = 0).astype(dtype, copy = False) 
    
    jump_adds_stack = np.stack(jump_adds_list, axis = 0).astype(dtype, copy = False) 

    const = np.ones((kx, n_sims, T, 1), dtype=dtype)
    
    jump_col = jump_inds_stack[:, :, :, None]
    
    X_full = np.concatenate([const, X_macro_stack, jump_col], axis = 3)  

    mu_stack = np.stack([ret_models[tkr].msar.mu_mean for tkr in sim_tickers], axis = 0).astype(dtype, copy = False) 
    
    sig2_stack = np.stack([ret_models[tkr].msar.sig2_med for tkr in sim_tickers], axis = 0).astype(dtype, copy = False)  
    
    p_stays_stack = np.stack(
        [np.clip(np.diag(ret_models[tkr].msar.P_mean).astype(dtype), 0.6, 0.99) for tkr in sim_tickers],
        axis = 0
    )  

    p = X_full.shape[3]

    if B_stack.shape[1] != p:

        raise ValueError(f"B_stack has p={B_stack.shape[1]} but expected {p} = 1 + {k_reg} + 1")

    states_arr = np.zeros((kx, n_sims, T), dtype = np.int8)

    states_arr[:, :, 0] = (rng.uniform(size = (kx, n_sims)) > 0.5).astype(np.int8)

    final_log_arr = np.zeros((kx, n_sims), dtype=dtype)

    shock_stream = build_shock_stream(
        dccw = dccw if (dccw is None or hasattr(dccw, "tickers")) else None,  
        full_tickers = sim_tickers,
        T = T,
        n_sims = n_sims,
        rng = rng,
        dtype = dtype
    )

    idx_all = np.arange(kx)
    
    Pm = np.stack([ret_models[tkr].msar.P_mean for tkr in sim_tickers], axis = 0).astype(dtype)

    p01 = 1.0 - Pm[:, 0, 0]

    p10 = 1.0 - Pm[:, 1, 1]

    denom = np.clip(p01 + p10, 1e-12, None)

    pi0 = p10 / denom

    pi1 = 1.0 - pi0

    mix_var_stack = (pi0 * sig2_stack[:, 0] + pi1 * sig2_stack[:, 1]).astype(dtype)

    mix_sd_stack = np.sqrt(np.clip(mix_var_stack, 1e-12, None))

    hist_vec = np.array([
        float(hist_weekly_std_map.get(tkr, float(mix_sd_stack[i]))) if hist_weekly_std_map is not None else float(mix_sd_stack[i])
        for i, tkr in enumerate(sim_tickers)
    ], dtype = dtype)

    if scale_mode == "down":

        gamma = np.minimum(1.0, hist_vec / np.clip(mix_sd_stack, 1e-12, None)).astype(dtype)
    
    elif scale_mode == "match":

        gamma = np.clip(hist_vec / np.clip(mix_sd_stack, 1e-12, None), 0.25, 2.0).astype(dtype)
    
    else:
        
        raise ValueError(f"Unknown scale_mode={scale_mode!r}")
    
    step_cap_vec = (STEP_CAP_MULT * hist_vec).astype(dtype)     
    
    state_std_cap = (3.0 * hist_vec).astype(dtype)
    
    u_mat = rng.uniform(size=(T, kx, n_sims)).astype(dtype)

    for t, U_t in enumerate(shock_stream()):  
     
        if t >= 1:
     
            prev = states_arr[:, :, t - 1]                                 
     
            pstay = p_stays_stack[idx_all[:, None], prev]                  
     
            u = u_mat[t]
            
            states_arr[:, :, t] = np.where(u < pstay, prev, 1 - prev)

        mu_design = np.einsum('knp,kp->kn', X_full[:, :, t, :], B_stack, optimize = True)  
      
        if MAX_WEEKLY_DRIFT is not None:
            
            mu_design = np.clip(mu_design, -MAX_WEEKLY_DRIFT, +MAX_WEEKLY_DRIFT)

        st_t = states_arr[:, :, t]                                    

        mu_state = mu_stack[idx_all[:, None], st_t]                          

        state_std = np.sqrt(sig2_stack[idx_all[:, None], st_t]).astype(dtype) * gamma[:, None]

        state_std = np.minimum(state_std, state_std_cap[:, None])
                            
        eps_std = U_t.T                                             

        inc = mu_design + mu_state + eps_std * state_std + jump_adds_stack[:, :, t]
        
        np.clip(inc, -step_cap_vec[:, None], step_cap_vec[:, None], out=inc)

        final_log_arr += inc.astype(dtype, copy = False)

    H = T
   
    for i, tkr in enumerate(sim_tickers):
   
        dq = ret_models[tkr].direct_qr.get(H)
   
        if dq is None or not dq.models:
   
            continue

        raw = ret_models[tkr].qr_raw_latest

        x_raw = np.array([raw.get(col, 0.0) for col in dq.cols], dtype = float).reshape(1, -1)

        x_scaled = dq.scaler.transform(x_raw)

        x_design = np.concatenate([np.ones((1, 1)), x_scaled], axis=1)

        q_probs = np.array(sorted(dq.models.keys()), dtype = float)
        
        if q_probs.size < 3:
        
            continue 

        q_tgt = np.array([float(dq.models[q].predict(x_design)[0]) for q in q_probs], dtype = float)

        S = final_log_arr[i].astype(np.float64, copy = False)
       
        q_sim = np.quantile(S, q_probs)

        if not np.isfinite(q_sim).all() or (q_sim[-1] - q_sim[0] < 1e-12):

            continue

        final_log_arr[i] = _quantile_map_piecewise(
            samples = S, 
            q_probs = q_probs, 
            q_sim = q_sim, 
            q_tgt = q_tgt
        )


    cp_arr = np.array([float(cp_vec[tkr]) for tkr in sim_tickers], dtype = np.float64)[:, None] 
   
    F = final_log_arr.astype(np.float64, copy = False)          
       
    final_prices = cp_arr * np.exp(F)
    
    final_prices[~np.isfinite(final_prices)] = np.nan

    mask = np.isfinite(final_prices).all(axis=1)
    
    final_prices = np.where(mask[:, None], final_prices, np.nan)

    if USE_BOUNDS:
      
        lo = np.array([float(lb_vec.get(tkr, -np.inf)) for tkr in sim_tickers], dtype = np.float64)[:, None]
      
        hi = np.array([float(ub_vec.get(tkr, np.inf)) for tkr in sim_tickers], dtype = np.float64)[:, None]
      
        final_prices = np.clip(final_prices, lo, hi)

    q05, q50, q95 = np.nanquantile(final_prices, [0.05, 0.5, 0.95], axis = 1)
   
    rets = final_prices / cp_arr - 1.0
    
    mean_r = np.nanmean(rets, axis = 1)
    
    se_r = np.nanstd(rets, axis = 1, ddof = 1)

    results: Dict[str, Dict[str, float]] = {}
   
    for i, tkr in enumerate(sim_tickers):
       
        results[tkr] = {
            "low": float(q05[i]),
            "avg": float(q50[i]),
            "high": float(q95[i]),
            "returns": float(mean_r[i]),
            "se": float(se_r[i]),
        }
    
    return results


def main() -> None:
    """
    End-to-end runner:
   
    • Load macro and market data,
   
    • Build future weekly MIDAS exogenous matrices per country,
   
    • Fit per-ticker ReturnModels in parallel,
   
    • Fit cross-asset DCC from historical standardised residuals,
   
    • Run the joint Monte-Carlo with calibration,
   
    • Export or print summary statistics.

    Side effects: logging and optional Excel export.
    """
  
    macro = MacroData()
  
    r = macro.r

    tickers: List[str] = config.tickers
  
    forecast_period: int = FORECAST_WEEKS

    close = r.weekly_close
    
    returns_map = {}
    
    for tk in tickers:
    
        if tk in close.columns:
    
            s = pd.Series(close[tk]).astype(float)
   
            rts = np.log(s).diff().dropna().values
   
            if rts.size >= 30:
   
                returns_map[tk] = rts
                
    hist_weekly_std = {
        tk: float(np.std(np.asarray(returns_map[tk], dtype=float), ddof=1))
        for tk in returns_map.keys()
    }
  
    latest_prices = r.last_price
  
    analyst = r.analyst

    if USE_BOUNDS:
  
        lb = config.lbr * latest_prices
  
        ub = config.ubr * latest_prices
  
    else:
  
        lb = -1
  
        ub = np.inf

    logger.info("Importing macro history …")
   
    raw_macro = macro.assign_macro_history_non_pct().reset_index()
   
    raw_macro = raw_macro.rename(columns = {
        "year": "ds"
    } if "year" in raw_macro.columns else {
        raw_macro.columns[1]: "ds"
    })
   
    raw_macro["ds"] = raw_macro["ds"].dt.to_timestamp()

    country_map = {
        t: str(c) for t, c in zip(analyst.index, analyst["country"])
    }
   
    raw_macro["country"] = raw_macro["ticker"].map(country_map)
   
    macro_clean = raw_macro[["ds", "country"] + BASE_REGRESSORS].dropna()

    logger.info("Fitting macro simulators (ECVAR / Hierarchical BVAR) …")
    
    future_index = pd.date_range(
        start = close.index[-1] + pd.offsets.Week(weekday = 6),
        periods = forecast_period,
        freq = "W-SUN",
    )

    country_quarterly = {
        c: (macro_clean[macro_clean["country"] == c][["ds"] + BASE_REGRESSORS]
            .dropna()
            .sort_values("ds")
            .reset_index(drop=True))
        for c in macro_clean["country"].unique()
    }

    midas_future_by_country = {
        c: build_midas_future_exog_for_country(
            df_country_quarterly = country_quarterly[c],
            weekly_index = future_index,    
            n_sims = N_SIMS,
        )
        for c in country_quarterly.keys()
    }

    exog_future_by_ticker = {
        tk: midas_future_by_country[c]
        for tk in tickers
        if (c := country_map.get(tk)) in midas_future_by_country
    }

    missing_country = [tk for tk in tickers if country_map.get(tk) not in midas_future_by_country]
    
    bad_price = [tk for tk in tickers if not np.isfinite(float(latest_prices.get(tk, np.nan)))]

    logger.info("Dropped (no macro for country): %s", missing_country)
    
    logger.info("Dropped (no current price): %s", bad_price)

    ret_models: Dict[str, ReturnModels] = {}
        
    if MACRO_IS_QUARTERLY:

        country_levels = {
            c: (macro_clean[macro_clean["country"] == c][["ds"] + BASE_REGRESSORS]
                .dropna()
                .sort_values("ds")
                .reset_index(drop=True))
            for c in macro_clean["country"].unique()
        }
    else:

        country_levels = {
            c: weekly_country_macro(
                df_country = macro_clean[macro_clean["country"] == c][["ds"] + BASE_REGRESSORS]
            )
            for c in macro_clean["country"].unique()
        }


    def _fit_one_ticker(
        tk
    ):
       
        cp = latest_prices.get(tk, np.nan)
      
        if not np.isfinite(cp): 
      
            return tk, None, None
        
        rng_tk = np.random.default_rng(
            RNG_SEED + stable_hash(
                s = tk
            )
        )
      
        dfp = pd.DataFrame({
            "ds": close.index, 
            "price": close[tk].values
        })
        
        ctry = country_map.get(tk)
        
        if ctry not in country_levels:
    
            return tk, None, None
        
        dfr = build_features_for_ticker(
            dfp = dfp, 
            dfm_levels = country_levels[ctry]
        )
        
        rm = fit_return_models(
            dfr = dfr, 
            horizons = [forecast_period],
            rng = rng_tk
        )
        
        return tk, rm, rm.msar.std_resid

    logger.info("Fitting return models per ticker (Bayesian MS-AR + Direct QR) …")
    
    results_parallel = Parallel(n_jobs = -1, prefer = "threads")(delayed(_fit_one_ticker)(tk) for tk in tickers)
    
    ret_models = {tk: rm for tk, rm, _ in results_parallel if rm is not None}
    
    no_model = [tk for tk in tickers if tk not in ret_models]

    logger.info("Dropped (no return model): %s", no_model)
    
    logger.info("Fitting DCC(1,1) from returns …")

    dccw = (fit_dcc_from_returns(
        returns_map = returns_map
    ) if (len(returns_map) >= 2) else (None))
            
    logger.info("Simulating joint price paths with DCC & Bayesian MS-AR …")

    results = simulate_return_paths_joint(
        ret_models = ret_models,
        exog_by_ticker = exog_future_by_ticker,
        cp_vec = latest_prices,
        lb_vec = lb,
        ub_vec = ub,
        forecast_period = forecast_period,
        n_sims = N_SIMS,
        rng = rng_global,
        dccw = dccw,
        tickers_order = tickers,
        hist_weekly_std_map = hist_weekly_std,
        scale_mode = 'down',
    )

    if not results:
     
        logger.warning("No results produced."); return

    rows = [{
        "Ticker": tk,
        "Current Price": latest_prices.get(tk),
        "Avg Price (p50)": rdict["avg"],
        "Low Price (p5)": rdict["low"],
        "High Price (p95)": rdict["high"],
        "Expected Return": (rdict["avg"] / latest_prices.get(tk) - 1.0) if latest_prices.get(tk) else np.nan,
        "SE": rdict["se"],
    } for tk, rdict in results.items()]

    df_out = pd.DataFrame(rows).set_index("Ticker")
    
    export_results(
        sheets = {"Advanced MC": df_out},
        output_excel_file = config.MODEL_FILE
    )
   
    logger.info("Run completed.")
   
    print(df_out)


if __name__ == "__main__":

    main()
