from __future__ import annotations

"""
TVP + (GARCH or EWMA) + Markov-regime Monte Carlo equity forecaster.

Overview
--------

This module forecasts per-ticker price distributions over a fixed horizon
by combining:

1) Macro/factor scenario generation via FAVAR + BVAR:
   
   - Macro level series are converted to weekly differences (log-diff for
     multiplicative series; level-diff otherwise).
  
   - Static principal components (PCs) summarise macro variation.
  
   - A Bayesian VAR(p) with a Minnesota prior generates joint paths for macro PCs
     and asset-pricing factors. We simulate exogenous paths by drawing (B, Σ)
     from the NIW posterior and propagating the VAR forward with Gaussian shocks.

2) Time-varying-parameter (TVP) return model with stochastic volatility:
  
   - Weekly log return y_t is modeled as 
   
        y_t = x_t'β_t + ε_t.
   
   - Coefficients follow a random walk: 
    
        β_t = β_{t-1} + η_t, diag(Q) unknown.
   
   - Observation variance σ_t^2 comes from either EWMA or GARCH(1,1).
   
   - A 2-state Markov regime multiplies σ_t^2 by v_low or v_high to capture bursts.

3) Fundamental guidance (optional):
   
   - When revenue/EPS histories and analyst next-period ranges exist, targets are
     combined across sources and translated to transformed scales:
       * revenue uses log level (log R)
       * EPS uses signed log1p (sign(x)*log(1+|x|))
  
   - A single standard deviation per metric is inferred from the cross-source
     min–max band: sigma = (max(T(v)) - min(T(v))) / (2 * 1.645 * (n_yahoo + n_sa)).
   
   - Terminal transformed deltas are drawn Normal around target deltas and
     linearly apportioned over the horizon as weekly exogenous increments.

Key mathematical components
---------------------------
- Bai–Ng IC to pick macro PC count:
   
    IC(k) = log(mean squared residual after removing k PCs) + g(N, T, k),
   
    where g differs by criterion (IC1 / IC2 / IC3).

- Minnesota prior for VAR(p):
   
    Var(intercept_i) = (λ1 * λ4 * σ_i)^2
   
    Var(own lag ℓ of eq i) = (λ1 / ℓ^{λ3})^2
   
    Var(cross lag ℓ j->i)  = (λ1 * λ2 / ℓ^{λ3})^2 * (σ_i / σ_j)

- Conjugate NIW posterior:
   
    Vn = (V0^{-1} + X'X)^{-1}
   
    Bn = Vn (V0^{-1}B0 + X'Y)
   
    Sn = S0 + Y'Y + B0'V0^{-1}B0 − Bn'(V0^{-1}+X'X)Bn
   
    νn = ν0 + T

- TVP predictive log-likelihood for given diag(Q):
   
    Sum over t:  −0.5 * [ log(2π F_t) + v_t^2 / F_t ],
   
    where F_t = x_t' P_{t|t-1} x_t + σ_t^2 and v_t is the one-step innovation.

- Discount-factor selection for Q:
    
    Approximate diag(Q) ∝ (1 − δ) / δ 
    
    search δ on a grid and maximize TVP predictive log-likelihood.

- EWMA volatility:
   
    s2_t = λ s2_{t-1} + (1 − λ) r_{t-1}^2.

- Two-state variance regimes:
   
    States from thresholding |z_t| at a high quantile; transition matrix from
    empirical transitions; v_high / v_low from variance ratio in the two states.

Outputs
-------
For each ticker: 5th/50th/95th percentiles of terminal price, mean return,
and Monte Carlo standard deviation of returns. Results are exported to Excel.

Notes
-----
All equations are written in plain text to render well in any environment.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

try:

    from arch import arch_model
 
    HAS_ARCH = True

except Exception:

    HAS_ARCH = False

from functions.export_forecast import export_results
from data_processing.financial_forecast_data import FinancialForecastData
from fetch_data.factor_data import load_factor_data
import config


from pandas.api.types import is_datetime64_any_dtype

LOG_DIFF = {"cpi", "gdp", "corporate profits"}   

LEVEL_DIFF = {"interest", "unemp", "balance of trade", "balance on current account"}

FUND_COLS = ["d_log_rev", "d_slog1p_eps"]

REV_KEYS = ["low_rev_y", "avg_rev_y", "high_rev_y", "low_rev", "avg_rev", "high_rev"]

EPS_KEYS = ["low_eps_y", "avg_eps_y", "high_eps_y", "low_eps", "avg_eps", "high_eps"]

REV_WEIGHTS = np.ones(len(REV_KEYS)) / len(REV_KEYS)

EPS_WEIGHTS = np.ones(len(EPS_KEYS)) / len(EPS_KEYS)

MACRO_COLS: List[str] = ["Interest", "Cpi", "Gdp", "Unemp", "Balance Of Trade", "Corporate Profits", "Balance On Current Account"]

N_MACRO_PCS: int = 2  

USE_FF5: bool = True

FACTOR_COLS5: List[str] = ["mkt_excess", "smb", "hml", "rmw", "cma"]

FACTOR_COLS3: List[str] = ["mkt_excess", "smb", "hml"]

if USE_FF5:
    
    FACTOR_COLS: List[str] = FACTOR_COLS5  

else:
    
    FACTOR_COLS = FACTOR_COLS3

FORECAST_WEEKS: int = 52

N_SIMS: int = 5000

N_JOBS: int = -1

BVAR_P: int = 2

MN_LAMBDA1: float = 0.25

MN_LAMBDA2: float = 0.5

MN_LAMBDA3: float = 1.0

MN_LAMBDA4: float = 100.0

NIW_NU0: int = 6

NIW_S0_SCALE: float = 0.1

TVP_Q0: float = 1e-5   

TVP_Q_MIN: float = 1e-7

TVP_Q_MAX: float = 1e-3

RSCALE_HI: float = 3.0  

EWMA_LAMBDA: float = 0.94

TRI_PROBS = np.array([0.2, 0.6, 0.2]) 

SA_ANALYST_PRIOR = 5                    

REV_SIGMA0 = 0.20  

EPS_SIGMA0 = 0.35   

RNG_SEED: int = 42

rng_global = np.random.default_rng(RNG_SEED)


def configure_logger() -> logging.Logger:
    """
    Configure and return a module-level logger.

    Creates a logger named after the module with a single stream handler and the
    format: "%(asctime)s - %(levelname)s - %(message)s". The handler is added only
    once. The level is set to INFO.

    Returns
    -------
    logging.Logger
        Configured logger.
    """


    logger = logging.getLogger(__name__)
   
    logger.setLevel(logging.INFO)
   
    if not logger.handlers:
   
        ch = logging.StreamHandler()
   
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
   
        ch.setFormatter(logging.Formatter(fmt))
   
        logger.addHandler(ch)
   
    return logger


logger = configure_logger()


def _coerce_quarter_to_timestamp(
    s: pd.Series
) -> pd.Series:
    """
    Coerce quarterly labels to quarter-end timestamps.

    Accepts:
   
    - strings like "2010Q1" (non-digits except 'Q' are removed before parsing),
   
    - pandas Period[Q-*],
   
    - datetime-like Series.

    Returns quarter-end pd.Timestamp values.

    Parameters
    ----------
    s : pd.Series
        Series of quarter labels.

    Returns
    -------
    pd.Series
        Quarter-end timestamps aligned to the input index.
    """

  
    if isinstance(s.dtype, pd.PeriodDtype):        

        return s.dt.to_timestamp(how = "end")

    if is_datetime64_any_dtype(s):

        return s

    s_str = s.astype(str).str.strip().str.upper()

    s_str = s_str.str.replace(r"[^0-9Q]", "", regex = True) 

    p = pd.PeriodIndex(s_str, freq = "Q")

    return p.to_timestamp(how = "end")


def _bai_ng_ic(
    dX_macro: pd.DataFrame, 
    k: int, 
    which: str = "IC2"
) -> float:
    """
    Compute Bai–Ng information criterion for k static factors (plain text).

    Given X with shape T x N (columns standardised), remove k principal components
    and compute residual mean squared error: sigma2_hat(k) = mean(residual^2).
   
    The criterion is:

    IC(k) = log(sigma2_hat(k)) + g(N, T, k),

    with penalty g:

    - IC1: g = [k * (N + T) / (N * T)] * log( (N * T) / (N + T) )
   
    - IC2: g = k * log( min(N, T) ) / min(N, T)
   
    - IC3: g = k * log( min(N, T) ) * (N + T)/(N * T)

    Parameters
    ----------
    dX_macro : pd.DataFrame
   
    k : int
    which : {'IC1','IC2','IC3'}, default 'IC2'

    Returns
    -------
    float
        Value of the selected criterion.
    """



    X = StandardScaler().fit_transform(dX_macro.values)

    T, N = X.shape

    if k == 0:

        resid = X

    else:

        pca = PCA(n_components = min(k, N)).fit(X)

        F = pca.transform(X)

        L = pca.components_.T  

        Xhat = F @ L.T

        resid = X - Xhat

    sigma2 = np.mean(resid ** 2)

    if which.upper() == "IC1":

        g = k * (N + T) / (N * T) * math.log((N * T) / (N + T))

    elif which.upper() == "IC2":

        g = k * math.log(min(N, T)) / (min(N, T))

    else:  

        g = k * math.log(min(N, T)) * (N + T) / (N * T)
 
    return math.log(sigma2) + g


def select_n_pcs(
    dX_macro: pd.DataFrame,
    max_pcs: int = 4,
    method: str = "bai-ng",
    expl_var_target: float = 0.90
) -> int:
    """
    Select the number of macro principal components.

    Methods:
  
    - 'bai-ng' (default): choose k that minimizes IC2; returns at least 1.
  
    - 'explained-var': smallest k with cumulative explained variance >= target.

    Parameters
    ----------
  
    dX_macro : pd.DataFrame
    max_pcs : int, default 4
    method : {'bai-ng','explained-var'}, default 'bai-ng'
    expl_var_target : float, default 0.90

    Returns
    -------
  
    int
        Selected number of PCs (1..max_pcs).
    """

    X = StandardScaler().fit_transform(dX_macro.values)

    if method.lower() == "bai-ng":

        ks = list(range(0, max_pcs + 1))

        ics = [
            _bai_ng_ic(
                dX_macro = dX_macro, 
                k = k, 
                which = "IC2"
            ) for k in ks
        ]

        k_star = int(ks[int(np.argmin(ics))])

        return max(1, k_star)

    else:

        pca = PCA(n_components = min(max_pcs, X.shape[1])).fit(X)

        cexp = np.cumsum(pca.explained_variance_ratio_)

        k_star = int(1 + np.argmax(cexp >= expl_var_target))

        return min(k_star, max_pcs)


def _mn_prior_with_lambdas(
    dX: pd.DataFrame,
    p: int,
    lambda1: float, 
    lambda2: float, 
    lambda3: float, 
    lambda4: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minnesota prior moments for a VAR(p) with k variables.

    Model: 
    
        y_t = c + sum_{ℓ=1..p} A_ℓ y_{t-ℓ} + ε_t,  ε_t ~ N(0, Σ).
    
    Stack coefficients as B with shape (1 + k*p) x k: first row is intercepts.

    Prior: 
    
        vec(B) ~ Normal( vec(B0), V0 ), 
    
    with diagonal V0 defined as:

    - Intercept in equation i:
  
        Var(c_i) = (lambda1 * lambda4 * sigma_i)^2
  
    - Own lag ℓ in equation i:
  
        Var(A_ℓ(i,i)) = (lambda1 / ℓ^{lambda3})^2
  
    - Cross lag ℓ from variable j to equation i (j ≠ i):
  
        Var(A_ℓ(i,j)) = (lambda1 * lambda2 / ℓ^{lambda3})^2 * (sigma_i / sigma_j)

    sigma_i is an equation scale estimated from AR(1) residual sd (with guards).

    Parameters
    ----------
    dX : pd.DataFrame
    p : int
    lambda1, lambda2, lambda3, lambda4 : float

    Returns
    -------
    (B0, V0) : Tuple[np.ndarray, np.ndarray]
    
        B0 is zeros ((1 + k * p) x k); V0 is diagonal ((1 + k * p) x (1 + k * p)).
    """


    k = dX.shape[1]
    
    sig = []

    for i in range(k):

        s = dX.iloc[:, i].values

        if len(s) < 3: 
            
            sig.append(np.std(s) + 1e-6)
            
            continue
     
        y = s[1:]
        
        x = s[:-1]
       
        a = np.dot(x, y) / (np.dot(x, x) + 1e-12)
       
        resid = y - a * x

        if len(resid) > 3:
            
            sig_i = np.std(resid, ddof = 1) 
        
        else:
            
            sig_i = np.std(s)
            
        sig.append(sig_i + 1e-8)
   
    sig = np.array(sig)

    V0_diag = np.zeros((k, 1 + k * p))
   
    for i in range(k):
   
        V0_diag[i, 0] = (lambda1 * lambda4 * sig[i]) ** 2
     
        for l in range(1, p + 1):
     
            for j in range(k):
     
                pos = 1 + (l - 1) * k + j
     
                if i == j:
     
                    var = (lambda1 / (l ** lambda3)) ** 2
     
                else:
     
                    var = (lambda1 * lambda2 / (l ** lambda3))**2 * (sig[i] / sig[j])
     
                V0_diag[i, pos] = var
    
    V0 = np.diag(np.mean(V0_diag, axis = 0))
    
    B0 = np.zeros((1 + k * p, k))
    
    return B0, V0


def _bvar_posterior_and_log_evidence(
    dX: pd.DataFrame, 
    p: int,
    lambda1: float,
    lambda2: float, 
    lambda3: float, 
    lambda4: float,
    nu0: int, 
    s0_scale: float
) -> Tuple[BVARModel, float]:
    """
    Conjugate BVAR posterior (NIW) and a log-evidence proxy.

    Given Y (T x k) and X (T x (1 + k * p)):

    Posterior updates:
  
        - Vn = inverse( V0^{-1} + X'X )
    
        - Bn = Vn * ( V0^{-1} * B0 + X'Y )
    
        - Sn = S0 + Y'Y + B0' V0^{-1} B0 − Bn' (V0^{-1} + X'X) Bn
    
        - nun = nu0 + T

    Log-evidence proxy (constant-offset) used for tuning:
  
        log ev ∝ 0.5 * k * ( log|Vn| − log|V0| ) + 0.5 * ( nu0 * log|S0| − nun * log|Sn| )

    Parameters
    ----------
    dX : pd.DataFrame
    p : int
    lambda1, lambda2, lambda3, lambda4 : float
    nu0 : int
    s0_scale : float

    Returns
    -------
    (BVARModel, float)
        The posterior model and the log-evidence proxy.
    """

    k = dX.shape[1]

    Y, X = _build_lagged_xy(
        dX = dX, 
        p = p
    )
   
    B0, V0 = _mn_prior_with_lambdas(
        dX = dX, 
        p = p, 
        lambda1 = lambda1, 
        lambda2 = lambda2, 
        lambda3 = lambda3, 
        lambda4 = lambda4
    )
    
    V0_inv = np.linalg.pinv(V0)
    
    S0 = s0_scale * np.eye(k)
    
    nu0 = max(nu0, k + 2)

    XtX = X.T @ X
    
    XtY = X.T @ Y
    
    YtY = Y.T @ Y

    Vn = np.linalg.pinv(V0_inv + XtX)

    Bn = Vn @ (V0_inv @ B0 + XtY)

    Kmat = V0_inv + XtX

    Sn = S0 + YtY + B0.T @ V0_inv @ B0 - Bn.T @ Kmat @ Bn

    nun = nu0 + Y.shape[0]

    model = BVARModel(
        post = BVARPosterior(
            Bn = Bn, 
            Vn = Vn, 
            Sn = Sn,
            nun = nun,
            p = p,
            k = k
        )
    )

    sign_v0, logdet_v0 = np.linalg.slogdet(V0)

    sign_vn, logdet_vn = np.linalg.slogdet(Vn)

    sign_s0, logdet_s0 = np.linalg.slogdet(S0)

    sign_sn, logdet_sn = np.linalg.slogdet(Sn)

    log_ev = 0.5 * k * (logdet_vn - logdet_v0) + 0.5 * (nu0 * logdet_s0 - nun * logdet_sn)

    return model, float(log_ev)


def tune_bvar(
    dX: pd.DataFrame,
    p_grid: List[int] = [1, 2, 3, 4],
    lambda1_grid: List[float] = [0.15, 0.25, 0.5],
    lambda2_grid: List[float] = [0.25, 0.5, 1.0],
    lambda3_grid: List[float] = [0.5, 1.0],
    lambda4_grid: List[float] = [50.0, 100.0, 200.0],
    nu0: int = 6,
    s0_scale: float = 0.1
) -> Tuple[BVARModel, dict]:
    """
    Grid-search BVAR hyperparameters using the log-evidence proxy.

    Search over p and Minnesota prior hyperparameters. For each combination,
    fit the NIW posterior and keep the best (largest) evidence proxy. If all
    fail numerically, fall back to a default Minnesota fit.

    Parameters
    ----------
    dX : pd.DataFrame
   
    p_grid : List[int], default [1,2,3,4]
    lambda1_grid, lambda2_grid, lambda3_grid, lambda4_grid : List[float]
   
    nu0 : int, default 6
   
    s0_scale : float, default 0.1

    Returns
    -------
    (BVARModel, dict)
        Best model and selected hyperparameters.
    """
    
    best = (-np.inf, None, None)  
  
    for p in p_grid:
  
        if len(dX) <= p: 
            
            continue
        
        for l1 in lambda1_grid:
        
            for l2 in lambda2_grid:
        
                for l3 in lambda3_grid:
        
                    for l4 in lambda4_grid:
        
                        try:
        
                            mdl, lev = _bvar_posterior_and_log_evidence(
                                dX = dX,
                                p = p, 
                                lambda1 = l1, 
                                lambda2 = l2, 
                                lambda3 = l3, 
                                lambda4 = l4, 
                                nu0 = nu0, 
                                s0_scale = s0_scale
                            )
        
                            if lev > best[0]:
        
                                best = (lev, mdl, {
                                    "p": p, 
                                    "lambda1": l1, 
                                    "lambda2": l2,
                                    "lambda3": l3, 
                                    "lambda4": l4,
                                    "nu0": nu0,
                                    "s0_scale": s0_scale
                                })
        
                        except np.linalg.LinAlgError:
        
                            continue
  
    if best[1] is None:

        return _fit_bvar_minnesota(
            dX = dX, 
            p = 2
        ), {
            "p": 2, 
            "lambda1": 0.25, 
            "lambda2": 0.5, 
            "lambda3": 1.0,
            "lambda4": 100.0,
            "nu0": nu0,
            "s0_scale": s0_scale
        }
        
    return best[1], best[2]


def _tvp_filter_loglik(
    y: np.ndarray, 
    X: np.ndarray, 
    sigma2: np.ndarray, 
    q_diag: np.ndarray
) -> float:
    """
    Predictive log-likelihood for a TVP regression with random-walk betas.

    State equations:
   
    - beta_t = beta_{t-1} + w_t,  w_t ~ Normal(0, Q)
   
    - y_t = x_t' beta_t + e_t, e_t ~ Normal(0, sigma2_t)

    Kalman filter gives innovation variance 
    
        F_t = x_t' P_{t|t-1} x_t + sigma2_t
    
    and innovation 
    
        v_t = y_t − x_t' a_{t|t-1}.

    Log-likelihood:
    
        sum over t of  −0.5 * [ log(2*pi*F_t) + (v_t^2) / F_t ].

    Parameters
    ----------
    y : np.ndarray (T,)
    X : np.ndarray (T,p)
    sigma2 : np.ndarray (T,)
    q_diag : np.ndarray (p,)

    Returns
    -------
    float
    """

    T, p = X.shape
    
    Q = np.diag(q_diag)
    
    a = np.zeros(p)
    
    P = np.eye(p) * 1e2
    
    ll = 0.0
    
    for t in range(T):

        P = P + Q

        x = X[t]

        F = x @ P @ x + sigma2[t]

        if F <= 0: F = sigma2[t] + 1e-8

        v = y[t] - x @ a

        ll += -0.5 * (math.log(2 * math.pi * F) + (v * v) / F)

        K = (P @ x) / F

        a = a + K * v

        P = P - np.outer(K, x) @ P

    return float(ll)


def tune_tvp_q(
    y: np.ndarray,
    X: np.ndarray,
    sigma2: np.ndarray,
    delta_grid: List[float] = [0.96, 0.97, 0.98, 0.985, 0.99, 0.992, 0.995]
) -> np.ndarray:
    """
    Pick TVP drift scale via discount-factor search.

    Approximation: diag(Q) is proportional to (1 − delta) / delta.
    
    Starting from q0 (from _auto_scale_q), rescale:

    q(delta) = q0 * [ ((1 − delta) / delta) / ((1 − 0.99) / 0.99) ].

    Clip to [TVP_Q_MIN, TVP_Q_MAX], evaluate predictive log-likelihood via
    _tvp_filter_loglik, and choose the delta with the highest value.

    Parameters
    ----------
    y : np.ndarray (T,)
    X : np.ndarray (T,p)
    sigma2 : np.ndarray (T,)
    delta_grid : List[float]

    Returns
    -------
    np.ndarray (p,)
        Selected diagonal of Q.
    """

    q0 = _auto_scale_q(
        X = X, 
        y = y
    )
    
    base = (1.0 - 0.99) / 0.99  
  
    best_ll = -np.inf
    
    best_q = q0
    
    for δ in delta_grid:
    
        scale = ((1.0 - δ) / δ) / base
    
        q = np.clip(q0 * scale, TVP_Q_MIN, TVP_Q_MAX)
    
        ll = _tvp_filter_loglik(
            y = y, 
            X = X, 
            sigma2 = sigma2, 
            q_diag = q
        )
    
        if ll > best_ll:
    
            best_ll, best_q = ll, q
    
    return best_q


def _val(
    x
):
    """
    Return float(x) if convertible and finite; else return NaN.

    Parameters
    ----------
    x : Any

    Returns
    -------
    float
    """

    v = _to_float_or_nan(
        x = x
    )

    if np.isfinite(v):
        
        return v  
    
    else:
        
        return np.nan


def _sigma_from_bounds_transformed(
    vals, 
    transform, 
    denom: float,
    fallback: float
) -> float:
    """
    Estimate a standard deviation on a transformed scale from endpoints.

    Given candidate endpoints vals = {v_i} and a transform T(·), compute:

    sigma_hat = ( max_i T(v_i) − min_i T(v_i) ) / ( 2 * 1.645 * denom ).

    Interpretation: the observed min–max is treated as covering roughly
    ± 1.645 * sigma on the transformed scale. If fewer than two finite endpoints
    exist, return fallback / denom.

    Parameters
    ----------
    vals : iterable of float
    transform : callable
    denom : float
    fallback : float

    Returns
    -------
    float
    """

  
    tv = []
  
    for v in vals:
  
        if not np.isfinite(v):
  
            continue
  
        try:
  
            tv.append(float(transform(v)))
  
        except Exception:
  
            continue
  
    if len(tv) >= 2:
  
        width = max(tv) - min(tv)
       
        return float(width / (2.0 * 1.645 * max(denom, 1.0)))
  
    return float(fallback / max(denom, 1.0))


def _analyst_sigmas_and_targets_combined(
    n_yahoo: float,
    n_sa: Optional[float],
    row_fore: pd.Series,
) -> Dict[str, float]:
    """
    Combine analyst ranges into one sigma and one target per metric.

    For revenue:
   
    - Transform with log: T_rev(x) = log(max(x, tiny)).
   
    For EPS:
   
    - Transform with signed log1p: T_eps(x) = sign(x) * log(1 + |x|).

    Let n = max(n_yahoo, 0) + max(n_sa, 0); if n == 0 use n = 1 for stability.
   
    Compute:
   
    - rev_sigma = ( max T_rev(v) − min T_rev(v) ) / ( 2 * 1.645 * n )
   
    - eps_sigma = ( max T_eps(v) − min T_eps(v) ) / ( 2 * 1.645 * n )

    Targets:
   
    - targ_rev = mean of {avg_rev_y, avg_rev} over finite values (fallback to one).
   
    - targ_eps = mean of {avg_eps_y, avg_eps} over finite values (fallback to one).

    Returned sigmas are clipped to [1e-5, 2.0].

    Parameters
    ----------
    n_yahoo : float
    n_sa : Optional[float]
    row_fore : Mapping/Series with keys:
        low_* , avg_* , high_* for both *_y (Yahoo) and non-suffix (SA).

    Returns
    -------
    dict with keys: 'rev_sigma', 'eps_sigma', 'targ_rev', 'targ_eps'
    """

    ny = _to_float_or_nan(
        x = n_yahoo
    )
    
    ns = _to_float_or_nan(
        x = n_sa
    )
    
    if not (np.isfinite(ny) and ny >= 0):
        
        ny = 0.0
    
    if not (np.isfinite(ns) and ny >= 0):
        
        ns = 0.0
        
    denom = max(ny + ns, 1.0)

    rev_vals = [
        _val(
            x = row_fore.get("low_rev_y")
        ),  
        _val(
            x = row_fore.get("avg_rev_y")
        ),  
        _val(
            x = row_fore.get("high_rev_y")
        ),
        _val(
            x = row_fore.get("low_rev")
        ),   
        _val(
            x = row_fore.get("avg_rev")
        ),   
        _val(
            x = row_fore.get("high_rev")
        ),
    ]
    
    eps_vals = [
        _val(
            x = row_fore.get("low_eps_y")
        ),  
        _val(
            x = row_fore.get("avg_eps_y")
        ),  
        _val(
            x = row_fore.get("high_eps_y")
        ),
        _val(
            x = row_fore.get("low_eps")
        ),   
        _val(
            x = row_fore.get("avg_eps")
        ),    
        _val(
            x = row_fore.get("high_eps")
        ),
    ]

    logp = lambda x: np.log(max(float(x), 1e-12))
    
    slog1p = lambda x: float(_slog1p_signed(
        s = pd.Series([x])
    )[0])

    rev_sigma = _sigma_from_bounds_transformed(
        vals = rev_vals, 
        transform = logp, 
        denom = denom, 
        fallback = REV_SIGMA0
    )
    
    eps_sigma = _sigma_from_bounds_transformed(
        vals = eps_vals, 
        transform = slog1p, 
        denom = denom, 
        fallback = EPS_SIGMA0
    )

    avg_rev_vals = [
        _val(
            x = row_fore.get("avg_rev_y")
        ),
        _val(
            x = row_fore.get("avg_rev")
        ) 
    ]
   
    avg_eps_vals = [
        _val(
            x = row_fore.get("avg_eps_y")
        ), 
        _val(
            x = row_fore.get("avg_eps")
        )
    ]
    
    if np.isfinite(np.nanmean(avg_rev_vals)):
        
        targ_rev = float(np.nanmean(avg_rev_vals))  
    
    else:
        
        targ_rev = np.nan
    
    if np.isfinite(np.nanmean(avg_eps_vals)):
        
        targ_eps = float(np.nanmean(avg_eps_vals))  
    
    else:
        
        targ_eps = np.nan

    rev_sigma = float(np.clip(rev_sigma, 1e-5, 2.0))
   
    eps_sigma = float(np.clip(eps_sigma, 1e-5, 2.0))

    return {
        "rev_sigma": rev_sigma,
        "eps_sigma": eps_sigma,
        "targ_rev": targ_rev,
        "targ_eps": targ_eps,
    }




def estimate_regime_params(
    std_resid: np.ndarray, 
    q_hi: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Estimate a 2-state variance regime from standardized residuals.

    Procedure:
   
    1) Compute a = |std_resid| and threshold at quantile q_hi.
   
    State s_t = 1 if a_t > threshold else 0.
   
    2) Estimate transition counts c_ij for transitions i->j and form:
   
        P = [[p00, 1 − p00],
            [1 − p11, p11]],  
    
    where
    
        p00 = c00 / (c00 + c01) 
        
        p11 = c11 / (c10 + c11)
    
    3) Stationary distribution pi = normalized left eigenvector of P for eigenvalue 1.
    
    4) 
    
        v_low = Var(z_t | s_t = 0)  (fallback 1.0 if empty).
        
        v_high_raw = Var(z_t | s_t = 1)  (fallback RSCALE_HI if empty).
    
    Return 
    
        v_low = 1.0 
        
        v_high = clip( v_high_raw / max(v_low, 1e-6), lower = 1.2, upper = 10.0 ).

    Returns
    -------
    (P, pi, v_low, v_high) : Tuple[np.ndarray, np.ndarray, float, float]
    """

   
    a = np.abs(std_resid)
   
    thr = np.quantile(a, q_hi)
   
    s = (a > thr).astype(int)
   
    if s.sum() == 0 or s.sum() == len(s):

        return np.array([[0.95, 0.05],[0.05, 0.95]]), np.array([0.5, 0.5]), 1.0, RSCALE_HI
   
    c00 = np.sum((s[: -1] == 0) & (s[1: ] == 0))
    
    c01 = np.sum((s[: -1] == 0) & (s[1: ] == 1))
    
    c10 = np.sum((s[: -1] == 1) & (s[1: ] == 0))
    
    c11 = np.sum((s[: -1] == 1) & (s[1: ] == 1))
    
    p00 = c00 / max(1, (c00 + c01))
    
    p11 = c11 / max(1, (c10 + c11))
    
    P = np.array([[p00, 1 - p00],[1 - p11, p11]])

    eigvals, eigvecs = np.linalg.eig(P.T)
   
    i = np.argmin(np.abs(eigvals - 1))
    
    pi = np.real(eigvecs[:, i])
    
    pi = pi / pi.sum()

    z = std_resid

    if np.any(s == 0):
        
        v0 = np.var(z[s == 0])
    
    else:
        
        v0 = 1.0
        
    if np.any(s == 1):

        v1 = np.var(z[s == 1])
    
    else:
        
        v1 = RSCALE_HI

    r = float(v1 / max(v0, 1e-6))

    return P, pi, 1.0, float(max(1.2, min(r, 10.0)))  


def pick_vol_model(
    resid: np.ndarray, 
    horizon: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Choose conditional variance model by in-sample likelihood: 
    
        EWMA vs GARCH(1,1).

    Setting
    -------
    Model demeaned residuals r_t (mean removed before fitting) using one of:
        
        A) EWMA volatility
        
        B) GARCH(1,1) with zero conditional mean and Normal shocks.

    For both models, the conditional density at time t is:
  
        r_t | F_{t-1} ~ Normal(0, h_t),
    
    and the log-likelihood contribution is:
    
        ell_t = -0.5 * [ log(2*pi*h_t) + r_t^2 / h_t ].

    MODEL A: EWMA VOLATILITY (RISKMETRICS-TYPE)
    -------------------------------------------
    Conditional variance recursion (demeaned residual r_t):

        h_t = lambda * h_{t-1} + (1 - lambda) * r_{t-1}^2,

    with 0 < lambda < 1 and h_0 = sample variance over the first min(20, T) 
    observations.

    The multi-step forecast used here is flat:

        h_{T+h|T} = h_T for all h >= 1.

    MODEL B: GARCH(1,1) (GENERALIsED ARCH)
    --------------------------------------
    Classical GARCH(1, 1) is used so with z_t ~ Normal(0,1) with internal data rescaling for numerical stability.

    GARCH(1,1) specification:
    
        r_t = sqrt(h_t) * z_t,  z_t ~ Normal(0, 1) i.i.d.
        
        h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}.

    PARAMETER CONSTRAINTS (for identifiability and positivity):
    omega > 0,
    alpha >= 0,
    beta  >= 0,
    alpha + beta < 1  (covariance-stationary case; ensures finite unconditional var).

    UNCONDITIONAL VARIANCE (if alpha + beta < 1):

        E[h_t] = omega / (1 - alpha - beta).

    PERSISTENCE:
    
    Persistence = alpha + beta. Values close to 1 imply very slowly mean-reverting h_t.

    Multistep forecast:

        Let k = alpha + beta and m = omega / (1 - k) (the long-run variance).
        Then the h-step-ahead forecast from time T satisfies:
    
            h_{T+1|T} = omega + k * h_T,
    
            h_{T+h|T} = m + k^h * (h_T - m)   for h >= 1.

    MODEL SELECTION IN THIS FUNCTION
    --------------------------------
    1) Compute EWMA h_t and LL_EWMA.

    2) If GARCH is available and T >= 50, fit GARCH(1,1) and compute
    
        LL_GARCH = sum_t -0.5 * [ log(2*pi*h_t^G) + r_t^2 / h_t^G ].

    3) If LL_GARCH > LL_EWMA, return (in-sample h_t^G, forecast h_{T+1..T+h}^G, "garch")

    otherwise return (in-sample h_t^EWMA, flat forecast h_T, "ewma").

    All returned variances are floored at a tiny epsilon to avoid division-by-zero.

    RETURNS
    -------
    (in_sample_var, forecast_var, name)
    in_sample_var : np.ndarray of shape (T,)
    forecast_var : np.ndarray of shape (horizon,)
    name : "garch" or "ewma"
    """

    r = resid - resid.mean()

    T = len(r)

    lam = EWMA_LAMBDA

    s2 = np.empty(T)
    
    s2[0] = np.var(r[ :min(20, T)], ddof = 1)
   
    for t in range(1, T):
        
        s2[t] = lam * s2[t-1] + (1-lam) * r[t-1] ** 2
        
    s2 = np.maximum(s2, 1e-12)
  
    ll_ewma = -0.5 * np.sum(np.log(2 * np.pi * s2) + (r ** 2) / s2)

    if HAS_ARCH and T >= 50:

        try:

            am = arch_model(r, mean = "Zero", vol = "GARCH", p = 1, q = 1, dist = "normal", rescale = True)

            res = am.fit(disp = "off")
           
            s2_g = res.conditional_variance.values
           
            s2_g = np.maximum(s2_g, 1e-12)
            
            ll_garch = -0.5 * np.sum(np.log(2 * np.pi * s2_g) + (r ** 2) / s2_g)
            
            if ll_garch > ll_ewma:
          
                fc = res.forecast(horizon = horizon, reindex = False)
          
                s2_f = np.array(fc.variance.values[-1])
          
             
                return s2_g, s2_f, "garch"
      
        except Exception:
      
            pass

    s2_f = np.full(horizon, s2[-1])

    return s2, s2_f, "ewma"


def stable_quantiles(
    simulator_fn,
    min_sims = 2000, 
    max_sims = 200000,
    step = 1000,
    tol = 0.002
) -> Tuple[np.ndarray, dict]:
    """
    Increase simulation size until p50 and p95 stabilize within tolerance.

    Call simulator_fn(n) -> array of length n (terminal prices). Recompute until
    max relative change in {p50, p95} between iterations < tol or n >= max_sims.

    Parameters
    ----------
    simulator_fn : Callable[[int], np.ndarray]
   
    min_sims : int, default 2000
   
    max_sims : int, default 200000
   
    step : int, default 1000
   
    tol : float, default 0.002

    Returns
    -------
    (sample, info) : (np.ndarray, dict)
        info contains 'n_sims', 'p50', 'p95', 'rel_move'.
    """
    
    prev = None
    
    n = min_sims

    while True:

        arr = simulator_fn(n)  
       
        q = np.quantile(arr, [0.5, 0.95])
       
        if prev is not None:
       
            rel = np.max(np.abs((q - prev) / np.maximum(1e-8, prev)))
       
            if rel < tol or n >= max_sims:
       
                return arr, {
                    "n_sims": n, 
                    "p50": float(q[0]), 
                    "p95": float(q[1]), 
                    "rel_move": float(rel)
                }
       
        prev = q
        
        n += step


def _standardise_factor_cols(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Rename common Fama–French columns to canonical snake_case names.

    Returns
    -------
    pd.DataFrame
        Copy of input with standardized names where applicable.
    """

    mapper = {
        "Mkt-RF": "mkt_excess",
        "SMB": "smb",
        "HML": "hml",
        "RMW": "rmw",
        "CMA": "cma",
        "RF": "rf"
    }
    
    return df.rename(columns = {k:v for k,v in mapper.items() if k in df.columns})


def _load_factors_weekly(
    use_ff5: bool = True
) -> pd.DataFrame:
    """
    Load Fama–French factors, standardize names, and resample to weekly.

    - Choose 5-factor or 3-factor set.
    - Forward-fill to weekly ('W-SUN').
    - Keep only FACTOR_COLS.

    Parameters
    ----------
    use_ff5 : bool, default True

    Returns
    -------
    pd.DataFrame
    """

    ff5_m, ff3_m, ff5_q, ff3_q = load_factor_data()

    if use_ff5:

        f = _standardise_factor_cols(
            df = ff5_m
        ).copy()

        keep = [c for c in FACTOR_COLS + ["rf"] if c in f.columns]
        
        f = f[keep]
    
    else:
    
        f = _standardise_factor_cols(
            df = ff3_m
        ).copy()
    
        keep = [c for c in FACTOR_COLS + ["rf"] if c in f.columns]
    
        f = f[keep]
    
    f_w = f.resample("W-SUN").ffill()
    
    f_w = f_w[[c for c in FACTOR_COLS if c in f_w.columns]].dropna(how = "all")
    
    return f_w


def _macro_levels_to_deltas(
    df_levels: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert macro level series to weekly differences.

    For each column:
  
    - If its lowercase name is in LOG_DIFF, compute d_log = log(x_t) − log(x_{t-1})
    with a small lower clamp to ensure positivity.
  
    - Otherwise compute d_level = x_t − x_{t-1}.

    Returns a DataFrame of differences named "d_<Column>".

    Parameters
    ----------
    df_levels : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    out = {}

    for c in MACRO_COLS:

        s = df_levels[c].astype(float).ffill().bfill()

        name = c.lower()

        if name in LOG_DIFF:

            s = s.clip(lower = 1e-12)
          
            out[f"d_{c}"] = np.log(s).diff()
    
        else:   
    
            out[f"d_{c}"] = s.diff()

    return pd.DataFrame(out, index = df_levels.index).dropna()


def _favar_pcs(
    dX_macro: pd.DataFrame, 
    n_pcs: int
) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    Compute macro principal components (FAVAR-style).

    Standardize columns, fit PCA with n_pcs, and return scores as columns
    pc1..pcK plus the fitted PCA and scaler objects.

    Parameters
    ----------
    dX_macro : pd.DataFrame
    n_pcs : int

    Returns
    -------
    (pcs, pca, scaler) : (pd.DataFrame, PCA, StandardScaler)
    """

    
    sc = StandardScaler()
    
    X = sc.fit_transform(dX_macro.values)
    
    pca = PCA(n_components = min(n_pcs, X.shape[1]))
    
    F = pca.fit_transform(X) 
    
    pcs = pd.DataFrame(F, index = dX_macro.index, columns = [f"pc{i + 1}" for i in range(pca.n_components_)])
    
    return pcs, pca, sc


@dataclass
class BVARPosterior:
    """
    Posterior parameters for the conjugate BVAR.

    Attributes
    ----------
    Bn : np.ndarray
    
        (1 + k * p) x k
    
    Vn : np.ndarray    
    
        (1 + k*p) x (1 + k*p)
    
    Sn : np.ndarray    
    
        k x k
    
    nun : int
    p : int
    k : int
    """
  
    Bn: np.ndarray  
  
    Vn: np.ndarray  
  
    Sn: np.ndarray  
  
    nun: int
  
    p: int
  
    k: int


@dataclass
class BVARModel:
  
    post: BVARPosterior

    def sample_coeffs_and_sigma(
        self, 
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw a posterior sample (B, Sigma) from the NIW posterior.

        Steps:
        
        1) Draw Sigma ~ Inverse-Wishart(nun, Sn). Implementation uses a Bartlett
        decomposition of a Wishart W ~ Wishart(nun, inverse(Sn)) and sets
        Sigma = inverse(W).
        
        2) Conditional on Sigma, draw vec(B) ~ Normal( vec(Bn), Vn ⊗ Sigma ).
        Implemented via a matrix-normal construction with Cholesky factors.

        Parameters
        ----------
        rng : np.random.Generator

        Returns
        -------
        (B, Sigma) : Tuple[np.ndarray, np.ndarray]
        """
  
        nun = self.post.nun
        
        Sn = self.post.Sn
        
        k = self.post.k
    
        Sinv = np.linalg.inv(Sn)

        L = np.linalg.cholesky(Sinv)

        A = np.zeros((k, k))

        for i in range(k):

            A[i, i] = np.sqrt(rng.chisquare(nun - i))
           
            for j in range(i):
           
                A[i, j] = rng.normal()
       
        W = L @ A @ A.T @ L.T
       
        Sigma = np.linalg.inv(W)

        m = self.post.Vn.shape[0]

        Z = rng.standard_normal((m, k))

        L_V = np.linalg.cholesky(self.post.Vn + 1e-12 * np.eye(m))

        L_S = np.linalg.cholesky((Sigma + Sigma.T) / 2.0)

        E = L_V @ Z @ L_S.T

        B = self.post.Bn + E

        return B, Sigma


    def simulate(
        self, 
        dX_lags: np.ndarray, 
        steps: int, 
        z: np.ndarray, 
        B: np.ndarray,
        Sigma: np.ndarray
    ) -> np.ndarray:
        """
        Simulate a VAR(p) path given (B, Sigma) and Gaussian shocks.

        Partition B into c (intercepts) and lag blocks A_ℓ. For t = 1..steps:
            
            x_{t} = c + sum_{ℓ=1..p} A_ℓ * x_{t−ℓ} + L*z_t,
        
        where L is the Cholesky of Sigma (or stabilized), and z_t ~ Normal(0, I).

        Parameters
        ----------
        dX_lags : np.ndarray (p, k)
        steps : int
        z : np.ndarray (steps, k)
        B : np.ndarray
        Sigma : np.ndarray

        Returns
        -------
        np.ndarray (steps, k)
        """
    
        p, k = self.post.p, self.post.k
    
        assert z.shape == (steps, k)
    
        Lchol = np.linalg.cholesky((Sigma + Sigma.T) / 2.0 + 1e-9 * np.eye(k))
       
        c = B[0]
       
        Al = [B[1 + l * k : 1 + (l + 1) * k].T for l in range(p)]
       
        lags = dX_lags.copy()
       
        out = np.zeros((steps, k))
       
        for t in range(steps):
       
            pred = c.copy()
       
            for l in range(1, p + 1):
       
                pred += Al[l - 1] @ lags[-l]
       
            innov = Lchol @ z[t]
       
            dx_next = pred + innov
       
            out[t] = dx_next
       
            lags = np.vstack([lags[1: ], dx_next])
       
        return out


def _build_lagged_xy(
    dX: pd.DataFrame,
    p: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Y and X for a VAR(p): Y = dX[p:], X has rows [1, y_{t-1}, ..., y_{t-p}].

    Parameters
    ----------
    dX : pd.DataFrame
    p : int

    Returns
    -------
    (Y, X) : Tuple[np.ndarray, np.ndarray]

    Raises
    ------
    ValueError
        If number of rows <= p.
    """

    T = dX.shape[0]

    if T <= p: 
        
        raise ValueError("Not enough observations for VAR lags.")
    
    Y = dX.values[p: ]
    
    rows = []

    for t in range(p, T):

        xrow = [1.0]

        for l in range(1, p + 1):
   
            xrow.extend(dX.values[t - l])
   
        rows.append(xrow)
   
    X = np.array(rows)
   
    return Y, X


def _mn_prior(
    dX: pd.DataFrame, 
    p: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minnesota prior using module-level hyperparameters (MN_LAMBDA1..4).

    Same construction as _mn_prior_with_lambdas.

    Parameters
    ----------
    dX : pd.DataFrame
    p : int

    Returns
    -------
    (B0, V0) : Tuple[np.ndarray, np.ndarray]
    """

    k = dX.shape[1]
    
    sig = []
    
    for i in range(k):
    
        s = dX.iloc[:, i].values
    
        if len(s) < 3:
     
            sig.append(np.std(s) + 1e-6)
            
            continue
     
        y = s[1: ]
        
        x = s[: -1]
      
        a = np.dot(x, y) / (np.dot(x, x) + 1e-12)
       
        resid = y - a * x
       
        if len(resid) > 3:
            
            sig_i = np.std(resid, ddof = 1) 
        
        else:
            
            sig_i = np.std(s)
       
        sig.append(sig_i + 1e-8)
    
    sig = np.array(sig)
    
    V0_diag = np.zeros((k, 1 + k * p))
    
    for i in range(k):
     
        V0_diag[i,0] = (MN_LAMBDA1 * MN_LAMBDA4 * sig[i]) ** 2
    
        for l in range(1, p + 1):
           
            for j in range(k):
             
                pos = 1 + (l - 1) * k + j
              
                if i == j:
                  
                    var = (MN_LAMBDA1 / (l**MN_LAMBDA3)) ** 2
              
                else:
              
                    var = (MN_LAMBDA1 * MN_LAMBDA2 / (l ** MN_LAMBDA3)) ** 2 * (sig[i] / sig[j])
                
                V0_diag[i, pos] = var
   
    V0 = np.diag(np.mean(V0_diag, axis = 0))
   
    B0 = np.zeros((1 + k * p, k))
   
    return B0, V0


def _fit_bvar_minnesota(
    dX: pd.DataFrame, 
    p: int
) -> BVARModel:
    """
    Fit a BVAR(p) under the Minnesota prior and NIW prior on Sigma.

    Applies the conjugate updates:
    
        Vn = inverse(V0^{-1} + X'X), 
        
        Bn = Vn (V0^{-1}B0 + X'Y),
        
        Sn = S0 + Y'Y + B0'V0^{-1}B0 − Bn' (V0^{-1}+X'X) Bn,
        
        nun = nu0 + T.

    Parameters
    ----------
    dX : pd.DataFrame
    p : int

    Returns
    -------
    BVARModel
    """

    k = dX.shape[1]

    Y, X = _build_lagged_xy(
        dX = dX, 
        p = p
    )
   
    B0, V0 = _mn_prior(
        dX = dX, 
        p = p
    )
   
    V0_inv = np.linalg.pinv(V0)
   
    nu0 = max(NIW_NU0, k + 2)
    
    S0 = NIW_S0_SCALE * np.eye(k)
   
    XtX = X.T @ X
    
    XtY = X.T @ Y
    
    YtY = Y.T @ Y
   
    Vn = np.linalg.pinv(V0_inv + XtX)
   
    Bn = Vn @ (V0_inv @ B0 + XtY)
   
    Kmat = V0_inv + XtX
   
    Sn = S0 + YtY + B0.T @ V0_inv @ B0 - Bn.T @ Kmat @ Bn
   
    nun = nu0 + Y.shape[0]
   
    return BVARModel(
        post = BVARPosterior(
            Bn = Bn,
            Vn = Vn,
            Sn = Sn,
            nun = nun,
            p = p,
            k = k
        )
    )


def _ensure_ds_col(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure DataFrame has a 'ds' datetime column.

    If already present, return as is. If index is datetime-like, lift it to 'ds'.
    Otherwise raise.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    KeyError
    """

   
    if "ds" in df.columns:
   
        return df
   
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
   
        return df.reset_index().rename(columns = {df.index.name or "index": "ds"})
   
    raise KeyError("'ds' not found as a column or datetime-like index.")


@dataclass
class TVPResult:
    """
    Outputs of the TVP Kalman filter and smoother.

    beta_filt   : T x p filtered means (beta_{t|t})
    
    beta_smooth : T x p smoothed means (beta_{t|T})
    
    V_smooth    : T x p x p smoothed covariances (P_{t|T})
    
    Q_diag      : p diagonal elements used for state noise
    """
    
    beta_filt: np.ndarray   
  
    beta_smooth: np.ndarray 
  
    V_smooth: np.ndarray    
  
    Q_diag: np.ndarray     


def _tvp_kalman(
    y: np.ndarray,
    X: np.ndarray,
    sigma2_t: np.ndarray, 
    q_diag: np.ndarray
) -> TVPResult:
    """
    Kalman filter + RTS smoother for random-walk-beta regression.

    Prediction:
   
        a_t = a_{t-1},  
        
        P_{t|t-1} = P_{t-1} + Q
    
    Innovation variance:
        
        S_t = x_t' P_{t|t-1} x_t + sigma2_t
   
    Update:
        
        K_t = P_{t|t-1} x_t / S_t
        
        a_t <- a_t + K_t (y_t − x_t' a_t)
    
        P_t <- P_{t|t-1} − K_t x_t' P_{t|t-1}

    Smoother:
    
        C_t = P_t * inverse(P_{t+1|t})
    
        beta_{t|T} = beta_{t|t} + C_t (beta_{t+1|T} − beta_{t+1|t})
    
        P_{t|T} = P_t + C_t (P_{t+1|T} − P_{t+1|t}) C_t'

    Parameters
    ----------
    y : np.ndarray (T,)
    X : np.ndarray (T,p)
    sigma2_t : np.ndarray (T,)
    q_diag : np.ndarray (p,)

    Returns
    -------
    TVPResult
    """

   
    T, p = X.shape
   
    Q = np.diag(q_diag)
   
    beta_pred = np.zeros((T, p))
   
    P_pred = np.zeros((T, p, p))
   
    beta_filt = np.zeros((T, p))
   
    P_filt = np.zeros((T, p, p))

    beta0 = np.zeros(p)

    P0 = np.eye(p) * 1e2  

    a = beta0
   
    P = P0

    for t in range(T):

        a = a  
       
        P = P + Q
       
        P_pred[t] = P
       
        beta_pred[t] = a

        x = X[t]

        S = x @ P @ x + sigma2_t[t]

        if S <= 0:

            S = sigma2_t[t] + 1e-8

        K = (P @ x) / S  

        v = y[t] - x @ a

        a = a + K * v

        P = P - np.outer(K, x) @ P

        beta_filt[t] = a
       
        P_filt[t] = P

    beta_smooth = beta_filt.copy()

    V_smooth = P_filt.copy()

    for t in range(T - 2, -1, -1):
       
        P_f = P_filt[t]
       
        P_p1 = P_pred[t + 1]  
      
        C = P_f @ np.linalg.pinv(P_p1) 
      
        beta_smooth[t] = beta_filt[t] + C @ (beta_smooth[t + 1] - beta_pred[t + 1])
      
        V_smooth[t] = P_f + C @ (V_smooth[t + 1] - P_p1) @ C.T

    return TVPResult(
        beta_filt = beta_filt,
        beta_smooth = beta_smooth,
        V_smooth = V_smooth, 
        Q_diag = q_diag
    )


def _can_use_fundamentals(
    df_fd: pd.DataFrame,
    row_fore: Optional[dict],
    last_rev: float,
    last_eps: float
)-> bool:
    """
    Return True if fundamental simulation is feasible.

    Requirements:
   
    - Non-empty financials history,
   
    - Forecast row has required low/avg/high keys (either *_y or non-suffix),
   
    - Last revenue and EPS levels are finite.

    Parameters
    ----------
    df_fd : pd.DataFrame
    row_fore : Optional[dict]
    last_rev : float
    last_eps : float

    Returns
    -------
    bool
    """

    if df_fd is None or df_fd.empty or row_fore is None:

        return False

    rk = ["low_rev_y", "avg_rev_y", "high_rev_y"]
   
    ek = ["low_eps_y", "avg_eps_y", "high_eps_y"]

    if not all(k in row_fore for k in rk) or not all(k in row_fore for k in ek):

        rk = ["low_rev", "avg_rev", "high_rev"]

        ek = ["low_eps", "avg_eps", "high_eps"]

        if not all(k in row_fore for k in rk) or not all(k in row_fore for k in ek):

            return False

    if not (np.isfinite(last_rev) and np.isfinite(last_eps)):

        return False

    return True


def _auto_scale_q(
    X: np.ndarray, 
    y: np.ndarray
) -> np.ndarray:
    """
    Heuristic scaling for diag(Q) from OLS residual variance.

    Fit OLS with intercept to get residual variance 
    
        s2 = Var(y − [1 X] b).
    
    Let var_x_j = Var(X[:,j]) with a small floor. 
    
    Define:

        q_j = TVP_Q0 * s2 / max(var_x_j, 1e-6), 
        
    then clip each q_j to [TVP_Q_MIN, TVP_Q_MAX].

    Parameters
    ----------
    X : np.ndarray (T,p)
    y : np.ndarray (T,)

    Returns
    -------
    np.ndarray (p,)
    """


    Xc = np.column_stack([np.ones(len(y)), X])
    
    beta_hat, *_ = np.linalg.lstsq(Xc, y, rcond = None)
    
    resid = y - Xc @ beta_hat
   
    s2 = np.var(resid)

    var_x = np.var(X, axis = 0) + 1e-12

    q = TVP_Q0 * (s2 / np.maximum(var_x, 1e-6))
    
    q = np.clip(q, TVP_Q_MIN, TVP_Q_MAX)
    
    return q


@dataclass
class MSVol:
    """
    Two-state variance regime parameters.

    P       : 2x2 transition matrix
   
    pi      : length-2 stationary distribution
   
    v_low   : low-variance multiplier (relative scale, typically 1.0)
   
    v_high  : high-variance multiplier (ratio vs v_low)
   
    thresh  : threshold on |z| used to define states
    """
   
    P: np.ndarray           
   
    pi: np.ndarray          
   
    v_low: float        
   
    v_high: float        
   
    thresh: float          


def _estimate_ms_on_stdres(
    resid_std: np.ndarray
) -> MSVol:
    """
    Estimate a coarse 2-state Markov model from |standardized residuals|.

    Split states by the 75th percentile of |z|. Count transitions to form P.
    
    Compute stationary pi from the unit-eigenvalue left eigenvector. 
    
    Set v_low = 1.0 and v_high = RSCALE_HI (coarse default). Store the threshold.

    Parameters
    ----------
    resid_std : np.ndarray (T,)

    Returns
    -------
    MSVol
    """

    a = np.abs(resid_std)
   
    thr = np.quantile(a, 0.75)
   
    s = (a > thr).astype(int)  

    c00 = np.sum((s[: -1] == 0) & (s[1: ] == 0))
   
    c01 = np.sum((s[: -1] == 0) & (s[1: ] == 1))
   
    c10 = np.sum((s[: -1] == 1) & (s[1: ] == 0))
   
    c11 = np.sum((s[: -1] == 1) & (s[1: ] ==1 ))
   
    p00 = c00 / max(1, (c00 + c01))
   
    p11 = c11 / max(1, (c10 + c11))
   
    P = np.array([[p00, 1 - p00],[1 - p11, p11]], dtype = float)

    eigvals, eigvecs = np.linalg.eig(P.T)
   
    i = np.argmin(np.abs(eigvals - 1))
   
    pi = np.real(eigvecs[:, i])
    
    pi = pi / pi.sum()
   
    v_low = 1.0
   
    v_high = RSCALE_HI
   
    return MSVol(
        P = P,
        pi = pi,
        v_low = v_low,
        v_high = v_high,
        thresh = float(thr)
    )


def _garch_or_ewma(
    resid: np.ndarray, 
    horizon: int
) -> np.ndarray:
    """
    Compute conditional variances using GARCH(1,1) if available; else EWMA.

    - If 'arch' is installed and T >= 50: 
    
        fit zero-mean GARCH(1,1)
        
        return its in-sample conditional variance and h-step forecast.
   
    - Otherwise: 
    
        EWMA with s2_t = lambda * s2_{t-1} + (1−lambda) * r_{t-1}^2,
        
        and flat forecast equal to last in-sample s2.

    Parameters
    ----------
    resid : np.ndarray
    horizon : int

    Returns
    -------
    (in_sample_var, forecast_var) : Tuple[np.ndarray, np.ndarray]
    """

    resid = resid - resid.mean()

    T = len(resid)

    if HAS_ARCH and T >= 50:

        try:

            am = arch_model(resid, mean = "Zero", vol = "GARCH", p = 1, q = 1, dist = "normal", rescale = True)

            res = am.fit(disp = "off")

            sigma2_in = res.conditional_variance.values
           
            fc = res.forecast(horizon = horizon, reindex = False)
           
            sigma2_f = np.array(fc.variance.values[-1])
            
            sigma2_in = np.maximum(sigma2_in, 1e-12)   
            
            sigma2_f = np.maximum(sigma2_f, 1e-12)     
           
            return sigma2_in, sigma2_f
       
        except Exception:
       
            pass

    sigma2 = np.empty(T)

    sigma2[0] = np.var(resid[: min(20, T)], ddof = 1)
    
    lam = EWMA_LAMBDA
    
    for t in range(1, T):
    
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * resid[t - 1] ** 2

    sigma2_f = np.full(horizon, sigma2[-1])

    return sigma2, sigma2_f


def _simulate_regime_path(
    P: np.ndarray, 
    pi: np.ndarray,
    T: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate a length-T path of {0,1} states from a 2-state Markov chain.

    Parameters
    ----------
    P : np.ndarray (2,2)
    pi : np.ndarray (2,)
    T : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray (T,)
    """

    s = np.zeros(T, dtype = int)

    s[0] = int(rng.choice([0, 1], p = pi))
   
    for t in range(1, T):
   
        s[t] = int(rng.choice([0, 1], p = P[s[t - 1]]))
   
    return s


def _build_joint_exog_country(
    df_country: pd.DataFrame,
    factors_w: pd.DataFrame,
    n_pcs: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], PCA, StandardScaler, List[str]]:
    """
    Construct country-level exogenous drivers: macro PCs + FF factors.

    Steps:
   
    1) Resample macro to weekly, forward-fill, drop NAs.
   
    2) Difference levels using _macro_levels_to_deltas.
   
    3) Compute n_pcs principal components.
   
    4) Join with weekly Fama–French factors.
   
    5) Return aligned design matrix, PC scores, and PCA artifacts.

    Returns
    -------
    (dX, pcs, exog_cols, pca, scaler, macro_delta_cols)
    """
   
    dfm_levels_w = (df_country.set_index("ds")[MACRO_COLS]
                    .sort_index().resample("W-SUN").mean().ffill().dropna())
   
    dX_macro = _macro_levels_to_deltas(
        df_levels = dfm_levels_w
    )
   
    macro_delta_cols = list(dX_macro.columns)         
   
    pcs, pca, sc = _favar_pcs(
        dX_macro = dX_macro, 
        n_pcs = n_pcs
    )

    fct = factors_w.reindex(pcs.index).ffill().dropna()
    
    
    common = pcs.index.intersection(fct.index)

    dX = pd.concat([pcs.loc[common], fct.loc[common]], axis = 1).dropna()
    
    exog_cols = list(dX.columns)
  
    return dX, pcs.loc[common], exog_cols, pca, sc, macro_delta_cols


def _prepare_ticker_frame(
    tk: str,
    prices_w: pd.Series,
    country_rows: pd.DataFrame,
    factors_w: pd.DataFrame,
    exog_cols_country: List[str],
    country_basis: dict,                     
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-ticker modeling frame and exogenous matrix.

    Compute weekly log-return y = log(price_t) − log(price_{t-1}). Merge with
    country macro PCs and factor returns, drop non-finite and near-constant
    exogenous columns, and return (full frame, exog submatrix).

    Parameters
    ----------
    tk : str
    prices_w : pd.Series
    country_rows : pd.DataFrame
    factors_w : pd.DataFrame
    exog_cols_country : List[str]
    country_basis : dict with keys: 'pca', 'sc', 'k_pcs', 'macro_delta_cols'

    Returns
    -------
    (dfr, dX) : Tuple[pd.DataFrame, pd.DataFrame]
        Returns (None, None) if insufficient clean data remains.
    """
   
    country_rows = _ensure_ds_col(
        df = country_rows
    )

    dfp = pd.DataFrame(
        {"price": prices_w}
    )
    
    dfp["y"] = np.log(dfp["price"]).diff()
    
    dfp = dfp.dropna()
    
    dfp = dfp.reset_index().rename(columns = {dfp.index.name or "index": "ds"})

    cr = country_rows[["ds"] + MACRO_COLS].dropna().sort_values("ds").drop_duplicates("ds", keep = "last")

    dfm_levels = (pd.merge_asof(dfp.sort_values("ds"), cr, on = "ds", direction = "backward")
                    .set_index("ds").asfreq("W-SUN").ffill().bfill().dropna())

    macro_delta_cols = country_basis["macro_delta_cols"]

    dX_macro_t = _macro_levels_to_deltas(
        df_levels = dfm_levels
    )

    dX_macro_t = dX_macro_t.reindex(columns = macro_delta_cols).dropna()

    sc = country_basis["sc"]
    
    pca = country_basis["pca"]
    
    k_pcs = country_basis["k_pcs"]
    
    Z = sc.transform(dX_macro_t.values)
    
    F = pca.transform(Z)
    
    pcs = pd.DataFrame(F, index = dX_macro_t.index, columns = [f"pc{i + 1}" for i in range(k_pcs)])

    fct = factors_w.reindex(pcs.index).ffill()

    dfp_w = dfp.set_index("ds").reindex(pcs.index).dropna()

    dfr = (dfp_w.join(pcs, how = "inner").join(fct, how = "inner")).dropna()

    use_cols = [c for c in exog_cols_country if c in dfr.columns]
  
    dfr = dfr[["price", "y"] + use_cols]

    keep_cols = []
   
    for c in use_cols:
   
        col = dfr[c].to_numpy()
   
        if np.isfinite(col).all() and (np.nanstd(col) > 1e-14):
   
            keep_cols.append(c)
   
    if len(keep_cols) != len(use_cols):
   
        logger.info("Dropping constant/near-constant exogs: %s", [c for c in use_cols if c not in keep_cols])
   
    use_cols = keep_cols

    if dfr.shape[0] < max(60, len(use_cols) + 5):

        logger.warning("Too little clean data for %s after NA/const cleanup.", tk)

        return None, None

    dX = dfr[use_cols]
  
    return dfr, dX


def _normalize_financials_df(
    obj
) -> pd.DataFrame:
    """
    Normalize arbitrary financials input into columns: 'ds', 'Revenue', 'EPS (Basic)'.

    Accepts dict/list/Series/DataFrame. Coerces 'ds' to datetime, values to float.
    Missing columns are filled with NaN.

    Returns
    -------
    pd.DataFrame with columns ['ds', 'Revenue', 'EPS (Basic)'].
    """
    
    if obj is None:
   
        return pd.DataFrame(columns = ["ds", "Revenue", "EPS (Basic)"])

    if isinstance(obj, pd.DataFrame):
       
        df = obj  
    
    else:
    
        df = pd.DataFrame(obj)
   
    df = df.reset_index(drop = False).copy()

    cols_lower = {
        c: str(c).strip().lower() for c in df.columns
    }
   
    df.columns = list(cols_lower.values())

    ds_candidates = ["ds", "date", "period", "time", "index"]

    ds_col = next((c for c in ds_candidates if c in df.columns), None)

    rev_candidates = ["revenue", "rev", "revenue_ttm", "ttm_revenue"]

    eps_candidates = ["eps (basic)", "eps_basic", "eps", "ttm_eps", "eps ttm", "eps-ttm"]

    rev_col = next((c for c in rev_candidates if c in df.columns), None)

    eps_col = next((c for c in eps_candidates if c in df.columns), None)

    out = pd.DataFrame()
   
    if ds_col is not None:
   
        out["ds"] = pd.to_datetime(df[ds_col], errors = "coerce")

    if rev_col is not None:
   
        out["Revenue"] = pd.to_numeric(df[rev_col], errors = "coerce")
   
    else:
   
        out["Revenue"] = np.nan

    if eps_col is not None:
   
        out["EPS (Basic)"] = pd.to_numeric(df[eps_col], errors = "coerce")
   
    else:
   
        out["EPS (Basic)"] = np.nan

    if "ds" in out.columns:

        out = out.dropna(subset = ["ds"])
    
    else:

        return pd.DataFrame(columns = ["ds", "Revenue", "EPS (Basic)"])

    return out[["ds", "Revenue", "EPS (Basic)"]]


def _fund_paths_from_combined(
    last_rev: float,
    last_eps: float,
    targ_rev: float,
    targ_eps: float,
    sigma_rev: float,
    sigma_eps: float,
    horizon: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate weekly fundamental increments from combined targets and sigmas.

    Transforms:
   
    - Revenue: log level, L(x) = log(max(x, tiny)).
   
    - EPS: signed log1p, S(x) = sign(x) * log(1 + |x|).

    Terminal transformed deltas:
   
    - dT_rev ~ Normal( L(targ_rev) − L(last_rev),  sigma_rev^2 )
   
    - dT_eps ~ Normal( S(targ_eps) − S(last_eps),  sigma_eps^2 )

    Weekly increments are constant per simulation: each delta is divided by the
    horizon and repeated across weeks, producing columns:
    
        [d_log_rev, d_slog1p_eps].

    Parameters
    ----------
    last_rev, last_eps : float
    targ_rev, targ_eps : float
    sigma_rev, sigma_eps : float
    horizon : int
    n_sims : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of shape (n_sims, horizon, 2)
    """

    out = np.zeros((n_sims, horizon, 2), dtype = float)
   
    if not (np.isfinite(last_rev) and np.isfinite(last_eps)) or n_sims <= 0:
   
        return out

    log_rev0 = np.log(max(last_rev, 1e-12))
   
    slog1p_eps0 = float(_slog1p_signed(
        s = pd.Series([last_eps])
    )[0])

    if np.isfinite(targ_rev) and targ_rev > 0:
     
        mu_rev = np.log(max(float(targ_rev), 1e-12)) - log_rev0
     
        dT_rev = rng.normal(loc = mu_rev, scale = float(sigma_rev), size = n_sims)
     
        out[:, :, 0] = (dT_rev / horizon)[:, None]

    if np.isfinite(targ_eps):
        
        mu_eps = float(_slog1p_signed(
            s = pd.Series([targ_eps])
        )[0]) - slog1p_eps0
        
        dT_eps = rng.normal(loc=mu_eps, scale = float(sigma_eps), size = n_sims)
        
        out[:, :, 1] = (dT_eps / horizon)[:, None]

    return out


def _safe_ols(
    X: np.ndarray, 
    y: np.ndarray
) -> np.ndarray:
    """
    Robust OLS with guards for non-finite rows and ill-conditioning.

    - Drop rows with non-finite values.
   
    - If underdetermined (rows <= cols), solve ridge-regularized normal equations
    with tiny data-scaled penalty.
   
    - Otherwise try lstsq; on failure fall back to ridge solve.

    Parameters
    ----------
    X : np.ndarray (T,p)
    y : np.ndarray (T,)

    Returns
    -------
    np.ndarray (p,)
    """

    
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    
    X2 = X[mask]; y2 = y[mask]
    
    if X2.shape[0] <= X2.shape[1]:

        XtX = X2.T @ X2

        lam = 1e-6 * (np.trace(XtX) / max(1, XtX.shape[0]))

        return np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X2.T @ y2)

    try:

        beta, *_ = np.linalg.lstsq(X2, y2, rcond=None)

        if not np.all(np.isfinite(beta)):

            raise np.linalg.LinAlgError("non-finite beta")

        return beta

    except Exception:

        XtX = X2.T @ X2

        lam = 1e-6 * (np.trace(XtX) / max(1, XtX.shape[0]))

        return np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X2.T @ y2)


def _slog1p_signed(
    s: pd.Series
) -> pd.Series:
    """
    Signed log1p transform for possibly negative values.

    S(x) = sign(x) * log(1 + |x|). 
    
    Inverse is sign(y) * (exp(|y|) − 1).

    Parameters
    ----------
    s : pd.Series

    Returns
    -------
    pd.Series
    """

   
    s = s.astype(float)
   
    return np.sign(s) * np.log1p(np.abs(s))


def _fundamentals_weekly_history(
    df_fd: pd.DataFrame,
    weekly_index: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, float, float]:
    """
    Resample financials to weekly and compute transformed increments.

    - Weekly forward-fill 'Revenue' and 'EPS (Basic)'.
   
    - d_log_rev = log(Revenue_t) − log(Revenue_{t-1}) with small clamp > 0.
   
    - d_slog1p_eps = S(EPS_t) − S(EPS_{t-1}), S is signed log1p.
    
    - Return last observed Revenue and EPS levels.

    Parameters
    ----------
    df_fd : pd.DataFrame
    weekly_index : pd.DatetimeIndex

    Returns
    -------
    (history, last_rev, last_eps) : (pd.DataFrame, float, float)
    """

    if df_fd is None or df_fd.empty:

        return (pd.DataFrame(index = weekly_index, columns = FUND_COLS), np.nan, np.nan)

    keep = ["ds"] + [c for c in ["Revenue", "EPS (Basic)"] if c in df_fd.columns]

    dfq = (df_fd[keep].sort_values("ds").set_index("ds"))

    w = dfq.resample("W-SUN").ffill()

    if "Revenue" in w:
        
        last_rev = float(w["Revenue"].dropna().iloc[-1])
    
    else:
        
        last_rev = np.nan

    if "EPS (Basic)" in w:
        
        last_eps = float(w["EPS (Basic)"].dropna().iloc[-1])  
    
    else:
        
        last_eps = np.nan

    d_log_rev = (np.log(w["Revenue"].clip(lower = 1e-12)).diff()
                 if "Revenue" in w else pd.Series(index = w.index, dtype = float))
    
    d_slog1p_eps = (_slog1p_signed(
        s = w["EPS (Basic)"]
    ).diff() if "EPS (Basic)" in w else pd.Series(index = w.index, dtype = float))

    out = pd.DataFrame({"d_log_rev": d_log_rev, "d_slog1p_eps": d_slog1p_eps})

    out = out.reindex(weekly_index).ffill().bfill()

    return out, last_rev, last_eps


def _extract_next_fore_row(
    next_fore: object, 
    tk: str
) -> Optional[dict]:
    """
    Extract forecast row for ticker from heterogeneous containers.

    Supports:
    
    - dict[ticker] -> dict,
    
    - DataFrame indexed by ticker or with a 'ticker' column,
    
    - list of dicts with key 'ticker'.

    Parameters
    ----------
    next_fore : object
    tk : str

    Returns
    -------
    dict or None
    """

  
    if next_fore is None:
  
        return None

    if isinstance(next_fore, dict):

        if tk in next_fore and isinstance(next_fore[tk], dict):

            return next_fore[tk]

        return None

    try:

        if isinstance(next_fore, pd.DataFrame):
          
            if tk in next_fore.index:
          
                row = next_fore.loc[tk]
          
                return row.to_dict()
          
            if "ticker" in next_fore.columns:
          
                hit = next_fore[next_fore["ticker"] == tk]
          
                if len(hit):
          
                    return hit.iloc[0].to_dict()
    except Exception:
        
        pass
   
    if isinstance(next_fore, (list, tuple)) and next_fore and isinstance(next_fore[0], dict):
   
        for row in next_fore:
   
            if row.get("ticker") == tk:
   
                return row
   
    return None


def simulate_price_paths_for_ticker(
    tk: str,
    dfr: pd.DataFrame,            
    exog_sims: np.ndarray,            
    exog_cols: List[str],
    horizon: int,
    rng_seed: int,
    lb: float,
    ub: float,
) -> Dict[str, float]:
    """
    Simulate terminal price distribution under TVP mean (random-walk betas),  
    GARCH (or EWMA) SV and 2-state Markov regime on variance.
    
    Mean:
        
        r_t = x_t' beta_t + epsilon_t,
    
        beta_t = beta_{t-1} + eta_t,  diag(Q) chosen by tune_tvp_q.

    Volatility:
        
        sigma_t^2 from pick_vol_model (EWMA or GARCH(1,1)).
        
        Two-state regime multiplies sigma_t^2 by v_low or v_high using
        transitions P estimated from standardized residuals.

    Price:
    
        Starting at current price P0, path evolves as:
    
            P_t = P_{t-1} * exp(r_t).
    
        Final prices are clipped to [lb, ub].  
    
    MC steps:
     
      1) Fit fixed-beta OLS → residuals → GARCH (or EWMA) variance path
     
      2) Estimate 2-state Markov P from standardized residuals
     
      3) Fit TVP betas via Kalman given sigma2_t (in-sample)
     
      4) For each sim: draw regime path, build sigma2_future, run TVP forward
      
    If no exogenous regressors (p == 0), simulate epsilon_t only with the same
    volatility and regime mechanics.

    Parameters
    ----------
    tk : str
    dfr : pd.DataFrame with columns ['price', 'y'] + exog_cols
    exog_sims : np.ndarray (n_sims, horizon, p)
    exog_cols : List[str]
    horizon : int
    rng_seed : int
    lb, ub : float

    Returns
    -------
    dict with keys: 'low', 'avg', 'high', 'returns', 'se'
    """
    
    y = dfr["y"].to_numpy()
    
    X = dfr[exog_cols].to_numpy()
    
    T, p = X.shape
    
    if p == 0:
       
        rng = np.random.default_rng(rng_seed)
       
        n_sims = exog_sims.shape[0]
       
        cp = float(dfr["price"].iloc[-1])

        sigma2_in, sigma2_f = _garch_or_ewma(
            resid = dfr["y"].values - dfr["y"].mean(), 
            horizon = horizon
        )
       
        ms = _estimate_ms_on_stdres(
            resid_std = (dfr["y"].values - dfr["y"].mean()) / (dfr["y"].std() or 1.0)
        )

        final_prices = np.empty(n_sims, dtype = float)
       
        for i in range(n_sims):
          
            s = _simulate_regime_path(
                P = ms.P, 
                pi = ms.pi,
                T = horizon, 
                rng = rng
            )
            
            norm = ms.pi[0] * ms.v_low + ms.pi[1] * ms.v_high
           
            v0 = ms.v_low / norm
            
            v1 = ms.v_high / norm
          
            sigma2_path = sigma2_f * np.where(s == 0, v0, v1)
          
            eps = rng.standard_normal(horizon) * np.sqrt(np.maximum(sigma2_path, 1e-12))
          
            path = cp * np.exp(np.cumsum(eps))
          
            final_prices[i] = float(np.clip(path[-1], lb, ub))

        q05, q50, q95 = np.quantile(final_prices, [0.05, 0.50, 0.95])
        
        rets = final_prices / cp - 1.0
        
        return {
            "low": float(q05),
            "avg": float(q50),
            "high": float(q95),
            "returns": float(np.mean(rets)),
            "se": float(np.std(rets, ddof=1)),
        }

    Xc = np.column_stack([np.ones(X.shape[0]), X])
    
    beta_hat = _safe_ols(
        X = Xc, 
        y = y
    )
  
    resid = y - Xc @ beta_hat

    sigma2_in, sigma2_f, vol_name = pick_vol_model(
        resid = resid, 
        horizon = horizon
    )
    
    std_resid = resid / np.sqrt(np.maximum(sigma2_in, 1e-12))

    P, pi, v_low, v_high = estimate_regime_params(
        std_resid = std_resid, 
        q_hi = 0.75
    )
    
    ms = MSVol(
        P = P,
        pi = pi,
        v_low = v_low,
        v_high = v_high,
        thresh = np.quantile(np.abs(std_resid), 0.75)
    )

    q_diag = tune_tvp_q(
        y = y, 
        X = X, 
        sigma2 = sigma2_in,
        delta_grid = [0.96, 0.97, 0.98, 0.985, 0.99, 0.992, 0.995]
    )

    tvp_res = _tvp_kalman(
        y = y,
        X = X, 
        sigma2_t = sigma2_in,
        q_diag = q_diag
    )
   
    beta_T = tvp_res.beta_smooth[-1]          
   
    P_T = tvp_res.V_smooth[-1]                

    rng = np.random.default_rng(rng_seed)
   
    n_sims = exog_sims.shape[0]
   
    cp = float(dfr["price"].iloc[-1])
   
    final_prices = np.empty(n_sims, dtype = float)

    try:
   
        L_P = np.linalg.cholesky(P_T + 1e-12 * np.eye(p))
   
    except np.linalg.LinAlgError:
   
        w, V = np.linalg.eigh((P_T + P_T.T) / 2)
   
        w = np.maximum(w, 1e-12)
   
        L_P = V @ np.diag(np.sqrt(w))
   
    q_vec = np.sqrt(np.maximum(tvp_res.Q_diag, 1e-12)) 

    for i in range(n_sims):

        s = _simulate_regime_path(
            P = ms.P, 
            pi = ms.pi, 
            T = horizon, 
            rng = rng
        )

        norm = ms.pi[0] * ms.v_low + ms.pi[1] * ms.v_high
        
        v0 = ms.v_low / norm
        
        v1 = ms.v_high / norm
        
        sigma2_path = sigma2_f * np.where(s == 0, v0, v1)

        Xf = exog_sims[i] 

        z0 = rng.standard_normal(p)

        beta = beta_T + L_P @ z0

        mu = np.zeros(horizon)
       
        for t in range(horizon):

            beta = beta + q_vec * rng.standard_normal(p)   

            mu[t] = float(Xf[t] @ beta)


        eps = rng.standard_normal(horizon) * np.sqrt(np.maximum(sigma2_path, 1e-12))

        r_path = mu + eps

        path = cp * np.exp(np.cumsum(r_path))
       
        final_prices[i] = float(np.clip(path[-1], lb, ub))

    q05, q50, q95 = np.quantile(final_prices, [0.05, 0.50, 0.95])
   
    rets = final_prices / cp - 1.0
   
    ret_mean = float(np.mean(rets))
   
    ret_std = float(np.std(rets, ddof = 1))

    return {
        "low": float(q05), 
        "avg": float(q50), 
        "high": float(q95), 
        "returns": ret_mean, 
        "se": ret_std
    }
    

def _to_float_or_nan(
    x
) -> float:
    """
    Safely convert to float; return NaN if conversion fails or value empty.

    Parameters
    ----------
    x : Any

    Returns
    -------
    float
    """
   
    try:

        arr = np.asarray(x, dtype = float).ravel()

        return float(arr[0]) if arr.size else float("nan")
    
    except Exception:
      
        return float("nan")


def main() -> None:
    """
    End-to-end driver: build exogenous scenario paths and run per-ticker simulations.

    Workflow
    --------
   
    1. Load data (financials, macro, factors, prices, analyst metadata).
   
    2. For each country:
   
        a. Build macro differences and select PC count via Bai–Ng.
   
        b. Fit a BVAR with Minnesota prior; simulate joint exogenous paths.
   
    3. For each ticker:
   
        a. Prepare modeling frame with price log-returns and exogs.
   
        b. If sufficient fundamentals, build simulated fundamental increments
        from combined targets/σ and inject into exog simulations.
   
        c. Run TVP + (GARCH or EWMA) + Markov regime Monte Carlo simulation.
   
    4. Export summary quantiles and diagnostics to an Excel workbook.

    Exports
    -------
    Writes a sheet "TVP + GARCH Monte Carlo" to config.MODEL_FILE with per-ticker
    summary statistics.
    
    Returns
    -------
    None
    """
    
    fdata = FinancialForecastData()
    
    fin_raw = fdata.prophet_data             
    
    next_fore = fdata.next_period_forecast() 
        
    macro = fdata.macro
    
    r = macro.r

    tickers: List[str] = config.tickers
    
    forecast_period: int = FORECAST_WEEKS

    close = r.weekly_close 

    latest_prices = r.last_price

    analyst = r.analyst

    lb = config.lbp * latest_prices
   
    ub = config.ubp * latest_prices

    factors_w = _load_factors_weekly(
        use_ff5 = USE_FF5
    )
    
    raw_macro = macro.assign_macro_history_large_non_pct().reset_index()
   
    raw_macro = raw_macro.rename(
        columns={"year": "ds"} if "year" in raw_macro.columns else {raw_macro.columns[1]: "ds"}
    )
    
    raw_macro["ds"] = _coerce_quarter_to_timestamp(
        s = raw_macro["ds"]
    )

    raw_macro["country"] = raw_macro["ticker"].map(analyst["country"].astype(str).to_dict())
   
    macro_clean = raw_macro[["ds", "country"] + MACRO_COLS].dropna()

    logger.info("Building FAVAR and simulating joint exogenous scenarios …")
   
    country_exog_paths: Dict[str, Optional[np.ndarray]] = {}
   
    country_exog_cols: Dict[str, List[str]] = {}
   
    country_rows_map: Dict[str, pd.DataFrame] = {}
   
    country_pca: Dict[str, dict] = {}

    for ctry, dfc in macro_clean.groupby("country"):
    
        try:

            dfm_levels_w = (dfc.set_index("ds")[MACRO_COLS].sort_index().resample("W-SUN").mean().ffill().dropna())

            dX_macro = _macro_levels_to_deltas(
                df_levels = dfm_levels_w
            )

            k_pcs = select_n_pcs(
                dX_macro = dX_macro, 
                max_pcs = 4,
                method = "bai-ng"
            )  

            dX, pcs, ex_cols, pca, sc, macro_delta_cols = _build_joint_exog_country(
                df_country = dfc,
                factors_w = factors_w,
                n_pcs = k_pcs
            )
         
            country_pca[ctry] = {
                "pca": pca,
                "sc": sc, 
                "k_pcs": k_pcs, 
                "macro_delta_cols": macro_delta_cols
            }

            mdl_bvar, bpar = tune_bvar(
                dX = dX, 
                p_grid = [1, 2, 3, 4]
            )

            if len(dX) <= bpar["p"]:
          
                raise ValueError("Insufficient history for tuned BVAR lags.")
          
            dX_lags = dX.values[-bpar["p"]:]
          
            k = dX.shape[1]
          
            half = N_SIMS // 2
          
            rng = np.random.default_rng(int(rng_global.integers(1_000_000_000)))
          
            z_cube = rng.standard_normal((half, forecast_period, k))
          
            sims = np.zeros((N_SIMS, forecast_period, k))
          
            for i in range(half):
          
                B, Sigma = mdl_bvar.sample_coeffs_and_sigma(
                    rng = rng
                )
          
                z = z_cube[i]
                
                z_neg = -z
          
                sims[i] = mdl_bvar.simulate(
                    dX_lags = dX_lags.copy(), 
                    steps = forecast_period, 
                    z = z,      
                    B = B, 
                    Sigma = Sigma
                )
          
                sims[i + half]  = mdl_bvar.simulate(
                    dX_lags = dX_lags.copy(), 
                    steps = forecast_period,
                    z = z_neg,  
                    B = B, 
                    Sigma = Sigma
                )
          
            if N_SIMS % 2 == 1:
          
                sims[-1] = sims[0]

            country_exog_paths[ctry] = sims
            
            country_exog_cols[ctry] = ex_cols
            
            country_rows_map[ctry] = _ensure_ds_col(
                df = dfc
            )[["ds"] + MACRO_COLS].copy()

        except Exception as e:
           
            logger.warning("Country %s exog build failed (%s). Using flat exogs.", ctry, e)
           
            ex_cols = [f"pc{i + 1}" for i in range(N_MACRO_PCS)] + FACTOR_COLS
           
            country_exog_paths[ctry] = np.zeros((N_SIMS, forecast_period, len(ex_cols)))
           
            country_exog_cols[ctry] = ex_cols
           
            country_rows_map[ctry] = _ensure_ds_col(
                df = dfc
            )[["ds"] + MACRO_COLS].copy()


    logger.info("Running TVP + GARCH (MS) Monte Carlo per ticker …")

   
    def _process_ticker(
        tk: str
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        """
        Per-ticker pipeline: build design, inject simulated exogs, run TVP+SV MC.

        Steps
        
        1) Validate price history and country metadata; build weekly 'price' and 'y'.
        
        2) Acquire country exogenous simulations and columns; build per-ticker frame
        via _prepare_ticker_frame (macro PCs + factors).
       
        3) If fundamentals are usable:
        
        - Normalise financials and extract last revenue/eps levels.
       
        - Compute combined sigmas and targets via _analyst_sigmas_and_targets_combined.
       
        - Simulate weekly increments for [d_log_rev, d_slog1p_eps] and inject them
            into the exogenous simulation cube in matching columns.
        
        4) Run simulate_price_paths_for_ticker with bounds [lb_tk, ub_tk].
        
        5) Log and return summary quantiles and MC diagnostics.

        Returns
        -------
        (ticker, result_dict or None)
        """

        cp = latest_prices.get(tk, np.nan)

        if not np.isfinite(cp):

            logger.warning("No price for %s, skipping.", tk)
            
            return tk, None

        if tk not in close.columns:

            logger.warning("No close series for %s, skipping.", tk)
            
            return tk, None

        prices_w = close[tk].dropna()

        prices_w = prices_w.asfreq("W-SUN").ffill().dropna()

        ctry = str(analyst["country"].get(tk, ""))

        exog_sims = country_exog_paths.get(ctry, None)

        exog_cols = country_exog_cols.get(ctry, [f"pc{i + 1}" for i in range(N_MACRO_PCS)] + FACTOR_COLS)

        rows_ctry = country_rows_map.get(ctry, macro_clean[macro_clean["country"]==ctry][["ds"] + MACRO_COLS])

        if rows_ctry is None or rows_ctry.empty:

            logger.warning("No macro rows for country '%s' (ticker %s). Using flat exogs.", ctry, tk)

        basis = country_pca.get(ctry, None)

        if basis is None:

            logger.warning("No PCA basis for country %s; using flat exogs for %s.", ctry, tk)

        try:

            dfr, dX = _prepare_ticker_frame(
                tk = tk,
                prices_w = prices_w,
                country_rows = rows_ctry,
                factors_w = factors_w,
                exog_cols_country = exog_cols,
                country_basis = basis,
            )
        except Exception as e:
         
            logger.warning("Prep failed for %s (%s).", tk, e)
         
            return tk, None
        
        if dfr is None:
            
            logger.warning("Prep failed or insufficient history for %s, skipping.", tk)
            
            return tk, None

        if dfr.shape[0] < forecast_period + 40:
         
            logger.warning("Insufficient history for %s, skipping.", tk)
         
            return tk, None

        lb_tk = float(lb.get(tk, -np.inf))        
       
        ub_tk = float(ub.get(tk, np.inf))
        
        raw_fin = fin_raw.get(tk, None)
       
        df_fd = _normalize_financials_df(
            obj = raw_fin
        )

        row_fore = _extract_next_fore_row(
            next_fore = next_fore,
            tk = tk
        )

        fund_hist, last_rev, last_eps = _fundamentals_weekly_history(
            df_fd = df_fd,
            weekly_index = dfr.index
        )

        has_fund = _can_use_fundamentals(
            df_fd = df_fd,
            row_fore = row_fore, 
            last_rev = last_rev, 
            last_eps = last_eps
        )

        base_cols = [c for c in exog_cols if c in dfr.columns]

        if has_fund:

            dfr = dfr.join(fund_hist, how="left").ffill().dropna(subset = ["y"])

            use_cols = base_cols + [c for c in FUND_COLS if c in dfr.columns and c not in base_cols]

        else:

            use_cols = base_cols

        macro_cols_only = [c for c in use_cols if c not in FUND_COLS]

        if macro_cols_only:
            
            idx_macro = [exog_cols.index(c) for c in macro_cols_only] 
        
        else:
            
            idx_macro = []
       
        macro_sims = (exog_sims[..., idx_macro]
                    if (exog_sims is not None and idx_macro)
                    else np.zeros((N_SIMS, forecast_period, 0)))

        exog_sims_use = np.zeros((N_SIMS, forecast_period, len(use_cols)))

        for j, c in enumerate(macro_cols_only):

            exog_sims_use[:, :, use_cols.index(c)] = macro_sims[:, :, j]

        if has_fund:
            
            if isinstance(row_fore, dict):
            
                n_yahoo = _to_float_or_nan(
                    x = row_fore.get("num_analysts_y")
                ) 
            
            else:
                
                n_yahoo = np.nan
            
            if isinstance(row_fore, dict):
                
                n_sa = _to_float_or_nan(
                    x = row_fore.get("num_analysts")
                )    
            
            else:
                
                n_sa = np.nan

            logger.debug("%s counts → yahoo=%s, sa=%s", tk, n_yahoo, n_sa) 

            params = _analyst_sigmas_and_targets_combined(
                n_yahoo = n_yahoo,
                n_sa = n_sa,
                row_fore = row_fore
            )

            rng_fund = np.random.default_rng(int(rng_global.integers(1_000_000_000)))

            fund_cube = _fund_paths_from_combined(
                last_rev = last_rev,
                last_eps = last_eps,
                targ_rev = params["targ_rev"],
                targ_eps = params["targ_eps"],
                sigma_rev = params["rev_sigma"],
                sigma_eps = params["eps_sigma"],
                horizon = forecast_period,
                n_sims = N_SIMS,
                rng = rng_fund,
            )

            col_to_idx = {
                "d_log_rev": 0, 
                "d_slog1p_eps": 1 
            }
            
            for c in FUND_COLS:
            
                if c in use_cols:
            
                    exog_sims_use[:, :, use_cols.index(c)] = fund_cube[:, :, col_to_idx[c]]

        out = simulate_price_paths_for_ticker(
            tk = tk,
            dfr = dfr[["price","y"] + use_cols],
            exog_sims = exog_sims_use,
            exog_cols = use_cols,
            horizon = forecast_period,
            rng_seed = int(rng_global.integers(1_000_000_000)),
            lb = lb_tk,
            ub = ub_tk,
        )
    
        logger.info("%s -> p5 %.2f, p50 %.2f, p95 %.2f, MC-σ %.4f", tk, out["low"], out["avg"], out["high"], out["se"])
       
        print(f"{tk}: Avg {out['avg']:.2f} | Low {out['low']:.2f} | High {out['high']:.2f} | "
              f"Ret {out['returns']:.4f} | MC σ {out['se']:.4f}")
      
        return tk, out


    results: Dict[str, Dict[str, float]]  =  {}
   
    for tk, res in Parallel(n_jobs = N_JOBS, prefer = "processes")(
        delayed(_process_ticker)(tk) for tk in tickers
    ):
    
        if res is not None:
    
            results[tk] = res

    rows = []
    
    for tk in tickers:
   
        if tk not in results:
   
            continue
   
        cp = latest_prices.get(tk)
   
        r_out = results[tk]
        
        rows.append(
            {
                "Ticker": tk,
                "Current Price": cp,
                "Avg Price (p50)": r_out["avg"],
                "Low Price (p5)": r_out["low"],
                "High Price (p95)": r_out["high"],
                "Returns": r_out["returns"],
                "SE": r_out["se"],
            }
        )

    if not rows:
   
        logger.warning("No results produced.")
   
        return

    df_out = pd.DataFrame(rows).set_index("Ticker")
  
    export_results(
        sheets = {"TVP + GARCH Monte Carlo": df_out},
        output_excel_file = config.MODEL_FILE,
    )
   
    logger.info("Run completed.")


if __name__ == "__main__":
   
    main()
