from __future__ import annotations

"""
Module: SARIMAX ensemble Monte Carlo with BVAR-driven exogenous scenarios

Overview
--------
This module produces forward-looking price distributions for equities by combining:

    1) A small ensemble of SARIMAX(p, d=0, q) models with exogenous drivers.

    2) Joint simulations of exogenous variables generated from a Minnesota-prior Bayesian VAR (BVAR).

    3) A jump indicator to capture tail events.

    4) A moving-block bootstrap for serial dependence in residual shocks with optional Student-t scaling.

    5) A calibration step that matches H-step uncertainty from model residuals to observed return uncertainty.

    6) A path clustering step to reduce the number of expensive SARIMAX forecasts.

Frequency and indexing are weekly throughout. The pipeline forecasts terminal price quantiles and summary
statistics over a horizon of H weeks by simulating S joint scenarios for macro and factor drivers, passing
them through a SARIMAX ensemble, and compounding simulated returns.

Data inputs and transformation
------------------------------
Let y_t denote the weekly log return of a ticker and x_t ∈ R^K the vector of exogenous drivers composed of:
  
    - Macro deltas: (derived from macro levels).
    - Risk factors: either Fama-French 5 factors {mkt_excess, smb, hml, rmw, cma} or FF3 subset.

Macro levels are merged by date to the price index, resampled to weekly, forward/backward filled, and
converted to deltas. Factors are aligned to weekly dates. Exogenous columns used per-ticker are the subset
present in its merged dataset. A binary jump indicator J_t is constructed by thresholding |y_t| at quantile q.

Models
------

1) SARIMAX with exogenous regressors:
   For candidate orders (p, 0, q) ∈ CANDIDATE_ORDERS:
   
       y_t = c + sum_{i=1..p} phi_i y_{t-i} + sum_{j=1..q} theta_j e_{t-j} + β' x_t + e_t,
   
   where e_t are innovations. Stationarity and invertibility are enforced. Exogenous regressors include the
   standardised macro/factor deltas and the jump indicator. Models are fitted with L-BFGS.


2) Minnesota-prior BVAR for exogenous simulation:
  
   Let X_t ∈ R^K collect the exogenous drivers (macro deltas and factors). A VAR(p) is:
    
       X_t = a + A_1 X_{t-1} + ... + A_p X_{t-p} + u_t,  with u_t ~ Normal(0, Σ).
   
   The Minnesota prior shrinks own-lags toward random-walk behavior and cross-lags toward zero, calibrated by
   series variances. Posterior draws (B, Σ) are simulated, and forward paths X_{t+1:t+H} are generated using
   standard-normal innovations. Antithetic variance reduction uses z and −z with the same (B, Σ).

Ensemble weighting
------------------
The candidate SARIMAX models are combined via either:

    - AIC weights:
        
        w_i ∝ exp(−0.5 * (AIC_i − min_j AIC_j)), normalised to sum to 1.

    - Stacked ridge weights (if enabled): collect 1-step-ahead in-sample predictions for each model into a
      design P ∈ R^{T×M} and solve minimise 
      
        ||y − P w||_2^2 + α ||w||_2^2.
      
      Negative coefficients are clamped to zero and renormalised to sum to 1. If stacking fails, AIC weights are used.

Optionally, the ensemble is pruned to the top K models by weight and renormalised, reducing forecast cost.

Uncertainty and simulation
--------------------------
For each ticker:

    1) Standardise exogenous columns using a scaler fitted on historical data; append the jump indicator.
   
    2) Fit the SARIMAX ensemble; extract residuals e_t from the highest-weight model.
   
    3) Calibrate a scalar shock scale ε such that the standard deviation of H-step residual sums matches the
       empirical standard deviation of H-step return sums:
   
           ε ≈ std( sum_{h=1..H} y_{t+h} ) / std( sum_{h=1..H} e_{t+h} ),
   
       computed by convolving with a length-H vector of ones (rolling sums).
   
    4) Simulate S exogenous paths:
   
         • If a country-level BVAR cube is available, subset columns to the ticker’s exog set.
   
         • Otherwise, use zeros (no-change baseline).
   
       Apply the historical scaler to each path to preserve scale comparability.
   
    5) Reduce S via K-means clustering on the flattened exogenous paths:
   
         • Flatten each (H×K) path to length H*K, cluster into K_CLUSTERS centroids.
   
         • Precompute SARIMAX forecasts for jump=0 and jump=1 at each centroid and for each ensemble model.
   
         • For a given simulation with jump path J_{i,h} ∈ {0,1}, linearly combine means:
   
               μ_h(i) = μ_h(jump=0) + (μ_h(jump=1) − μ_h(jump=0)) * J_{i,h}.
   
    6) Generate the shock path via a moving-block bootstrap:
   
         • If N ≥ block: concatenate random contiguous residual blocks until length H, else sample IID.
   
         • Optional Student-t scaling: draw χ^2 with df = ν > 2 and set s = sqrt(ν / χ^2) / sqrt(ν / (ν − 2)),
           so E[s^2] = 1, yielding heavy-tailed shocks e*_h = s_h * e_h.
   
    7) Form the simulated return path:
   
           r_h(i) = μ_h(i) + ε * e*_h(i),
   
       then compound geometrically to terminal price:
   
           P_T(i) = P_0 * exp( sum_{h=1..H} r_h(i) ),
   
       and clip to [lb, ub].

Outputs
-------
For each ticker, compute across simulations:
    
    - Low (5th percentile of P_T), Avg (50th percentile), High (95th percentile).
    
    - Returns = mean(P_T / P_0 − 1).
    
    - SE = standard deviation of (P_T / P_0 − 1).

Computational considerations
----------------------------

- Complexity is dominated by SARIMAX forecasting. Clustering reduces calls from O(S) to O(K_CLUSTERS).

- Pre-drawing jump paths, bootstrap indices, and t-scales moves work outside inner loops.

- Parallelism across tickers uses processes to avoid GIL contention.

- Antithetic draws reduce Monte Carlo variance for exogenous simulations.

Reproducibility
---------------

- A global RNG (rng_global) seeds country-level BVAR simulations and provides per-ticker seeds.

- All stochastic elements (BVAR draws, antithetic normals, clustering seeds, jump paths, bootstrap indices,
  t-scales, and model selection per simulation) depend on these seeds.

Assumptions and limitations
---------------------------

- SARIMAX residuals are treated as weakly stationary around zero with short-run dependence captured by

  ARMA terms, and remaining dependence handled by block bootstrap.

- The jump indicator is a simple threshold rule on |y_t|; it captures outliers but not signed asymmetry.

- The linear exogenous effect allows interpolation between jump=0 and jump=1 forecasts; nonlinearity is not modeled.

- BVAR simulations assume linear dynamics and Gaussian innovations; macro regime shifts are not explicitly modeled.

Notation (shapes)
-----------------
- H: forecast horizon in weeks, S: number of simulations, K: number of exogenous columns for a ticker.
- y ∈ R^T: log returns. X ∈ R^{T×K}: standardised exogenous matrix; J ∈ {0,1}^T: jump indicator.
- Exogenous simulations: shape (S, H, K); jump paths: shape (S, H).
- Forecast means μ_h(i) ∈ R for h=1..H; residual bootstrap indices: shape (S, H).

Key configuration
-----------------
- CANDIDATE_ORDERS: list of (p, d=0, q) candidates for SARIMAX.
- USE_STACK_WEIGHTS, RIDGE_ALPHA: stacking control and ridge strength.
- BVAR_P: VAR lag order for exogenous simulations with Minnesota prior.
- JUMP_Q: quantile threshold for jump detection (e.g., 0.97).
- RESID_BLOCK: bootstrap block length; T_DOF: Student-t degrees of freedom (>2).
- N_SIMS: number of Monte Carlo paths; K_CLUSTERS: number of exogenous path clusters.
"""


from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge

import warnings

from sklearn.cluster import KMeans
from numba import njit


import TVP_GARCH_MC as mc



from functions.export_forecast import export_results
from data_processing.macro_data import MacroData
import config


MACRO_COLS: List[str] = ["Interest", "Cpi", "Gdp", "Unemp", "Balance Of Trade", "Corporate Profits", "Balance On Current Account"]

USE_FF5: bool = True

FACTOR_COLS5: List[str] = ["mkt_excess", "smb", "hml", "rmw", "cma"]

FACTOR_COLS3: List[str] = ["mkt_excess", "smb", "hml"]

FACTOR_COLS: List[str] = FACTOR_COLS5 if USE_FF5 else FACTOR_COLS3

EXOG_COLS: List[str] = [f"d_{c}" for c in MACRO_COLS] + FACTOR_COLS

CANDIDATE_ORDERS: List[Tuple[int, int, int]] = [(1, 0, 0), (0, 0, 1), (1, 0, 1)]

FORECAST_WEEKS: int = 52

N_SIMS: int = 1000

JUMP_Q: float = 0.97

N_JOBS: int = -1

RESID_BLOCK: int = 4

T_DOF: int = 5

BVAR_P: int = 2

K_CLUSTERS = max(8, int(np.sqrt(N_SIMS))) 

RNG_SEED: int = 42

rng_global = np.random.default_rng(RNG_SEED)

USE_STACK_WEIGHTS: bool = True       

RIDGE_ALPHA: float = 1e-2    

logger = mc.configure_logger()


def simulate_joint_exog_paths(
    dX: pd.DataFrame,
    steps: int,
    n_sims: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate joint exogenous-regressor paths using a Minnesota-prior Bayesian VAR (BVAR)
    fitted on the differenced macro and factor set `dX`.

    Model:
        Let X_t be a K-dimensional vector of exogenous drivers (macro deltas and factors).
       
        A VAR(p) specifies:
       
            X_t = c + A_1 X_{t-1} + A_2 X_{t-2} + ... + A_p X_{t-p} + u_t,
       
        where u_t ~ Normal(0, Sigma), and p = BVAR_P.

        The Minnesota prior (implemented inside mc._fit_bvar_minnesota) shrinks own-lag
        coefficients of each series toward 1 at lag 1 and toward 0 at higher lags,
        while cross-lag coefficients are shrunk toward 0. Innovations are scaled by
        series-specific variances. The routine samples (B, Sigma) from the posterior
        and simulates forward trajectories conditional on the last p lags.

    Antithetic variance reduction:
      
        For each simulation i, a standard normal innovation array z of shape (steps, K)
        is drawn and an antithetic counterpart −z is constructed. Using the same
        (B, Sigma) draw for z and −z reduces Monte Carlo variance of path summaries.

    Inputs
    -------
    dX : pandas.DataFrame
        History of exogenous regressors after transformation, shape (T, K), aligned to
        weekly frequency and ordered as expected by EXOG_COLS.
    steps : int
        Forecast horizon H in periods (weeks).
    n_sims : int
        Number of simulated paths S to produce (antithetic pairing is used internally).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sims : numpy.ndarray
        Array with shape (n_sims, steps, K) containing simulated paths for X_{t+1:t+H}.

    Notes
    -----
    - The function raises an error if T <= BVAR_P, because p lags are required.
    - The simulation is conditional on the last BVAR_P rows of dX (the state vector).
    - mc._fit_bvar_minnesota provides two methods used here:
        sample_coeffs_and_sigma(rng) -> (B, Sigma) posterior draw,
        simulate(lags, steps, z, B, Sigma) -> forward path.
    - Complexity is O(n_sims * steps * K + fitting cost), with antithetic pairing
      implemented by reusing (B, Sigma) for z and −z.
    """
   
    k = dX.shape[1]

    if len(dX) <= BVAR_P:
     
        raise ValueError("Insufficient data for BVAR lags.")
    
    model = mc._fit_bvar_minnesota(
        dX = dX, 
        p = BVAR_P
    )
    
    dX_lags = dX.values[-BVAR_P:]  

    half = n_sims // 2

    rng = np.random.default_rng(seed)

    z_cube = rng.standard_normal((half, steps, k))

    sims = np.zeros((n_sims, steps, k))

    for i in range(half):

        z = z_cube[i]

        z_neg = -z

        B, Sigma = model.sample_coeffs_and_sigma(
            rng = rng
        )

        path = model.simulate(dX_lags.copy(), steps, z, B, Sigma)

        path_b = model.simulate(dX_lags.copy(), steps, z_neg, B, Sigma)
      
        sims[i] = path
      
        sims[i + half] = path_b
    
    if n_sims % 2 == 1:
    
        sims[-1] = sims[0]
    
    return sims


@dataclass
class Ensemble:
    """
    Container for a small ensemble of SARIMAX fits and their mixture weights.

    Attributes
    ----------
   
    fits : List
   
        A list of fitted statsmodels SARIMAXResults objects. Each corresponds to
   
        a (p, d, q) order from CANDIDATE_ORDERS and includes exogenous regressors.
   
    weights : numpy.ndarray
   
        A length-M nonnegative vector w that sums to 1, assigning each model a
        probability/weight for combination. When stacking is enabled, weights are
        obtained via ridge regression on 1-step-ahead predictions; otherwise,
        Akaike weights are used:
   
            w_i = exp(-0.5 * (AIC_i - min_j AIC_j)) / sum_k exp(-0.5 * (AIC_k - min_j AIC_j)).
    """
  
    fits: List
  
    weights: np.ndarray  


def _fit_sarimax_candidates(
    y: pd.Series, 
    X: np.ndarray
) -> Ensemble:
    """
    Fit a small ensemble of SARIMAX(p, d, q) models with exogenous regressors and
    compute combination weights either via AIC or stacked ridge.

    Model:
    
        For log returns y_t and exogenous vector x_t (including macro/factor drivers
        and possibly jump indicators), each candidate specifies:
    
            y_t = c + sum_{i=1..p} phi_i y_{t-i} + sum_{j=1..q} theta_j e_{t-j} + beta' x_t + e_t,
    
        where e_t is a zero-mean innovation. Here d = 0 (no differencing) and p, q
        come from CANDIDATE_ORDERS. The trend is a constant c.

    Weighting schemes:
    
        1) AIC weights (default fallback):
    
            Let AIC_i be the Akaike Information Criterion of model i. The unnormalised
            weight is 
            
                w_i* = exp( -0.5 * (AIC_i - min_j AIC_j) ). Normalise to sum to 1.
        
        2) Stacked ridge weights (if USE_STACK_WEIGHTS is True):
          
            Compute 1-step-ahead predictions for each fitted model over the in-sample
            period to form a T x M design matrix P. Fit a ridge regression
              
                minimise || y - P w ||_2^2 + alpha * || w ||_2^2
           
            with α = RIDGE_ALPHA. Clamp negative coefficients to zero and renormalise
            to sum to 1. If stacking fails due to insufficient data or instability,
            fall back to AIC weights.

    Frequency handling:
      
        If y has no frequency, coerce to weekly (W-SUN) either by asfreq or by
        resampling with last-and-ffill to ensure proper date handling in statsmodels.

    Inputs
    -------
    y : pandas.Series
        Target series of log returns, indexed by weekly DatetimeIndex.
    X : numpy.ndarray
        Exogenous regressor matrix with shape (T, K), aligned with y.

    Returns
    -------
    Ensemble
        Dataclass holding the list of fitted SARIMAX models and their weights.

    Notes
    -----
    - Stationarity and invertibility are enforced in estimation.
    - Optimisation uses L-BFGS with a high maxfun cap to encourage convergence.
    - Errors during fitting of a particular order are caught and that candidate
      is skipped.
    """   
    if getattr(y.index, "freq", None) is None:
   
        try:
   
            y = y.asfreq("W-SUN")
   
        except ValueError:
   
            y = y.resample("W-SUN").last().ffill()

    fits = []
    
    aics = []

    for order in CANDIDATE_ORDERS:
     
        try:
     
            mdl = SARIMAX(
                endog = y, 
                exog = X, 
                order = order,
                trend = "c",
                enforce_stationarity = True,
                enforce_invertibility = True,
            )
           
            with warnings.catch_warnings():
       
                warnings.simplefilter("ignore", ConvergenceWarning)
       
                res = mdl.fit(disp = False, method = "lbfgs", maxfun = 5000)
       
            fits.append(res)
       
            aics.append(res.aic)
       
        except Exception:
       
            continue

    if not fits:
       
        raise RuntimeError("No SARIMAX fits succeeded")


    aics = np.array(aics, dtype=float)

    aic_w = np.exp(-0.5 * (aics - aics.min()))

    aic_w /= aic_w.sum()

    if USE_STACK_WEIGHTS:

        w_stack = _stack_ridge_weights(
            fits = fits, 
            y = y
        )

        if w_stack is not None and np.isfinite(w_stack).all():

            return Ensemble(
                fits = fits,
                weights = w_stack
            )

    return Ensemble(
        fits = fits,
        weights = aic_w
    )


def cluster_exog_paths(
    exog_sims: np.ndarray,
    k: int, 
    rng: np.random.Generator
):
    """
    Cluster simulated exogenous paths to reduce the number of expensive SARIMAX
    forecast calls by representing S paths with K centroids.

    Method:
     
        Flatten each simulated path of shape (H, Kx) into a feature vector of
        length H * Kx. Run K-means on the S vectors with k clusters. For cluster c,
        the centroid is reshaped back to (H, Kx). Each simulation receives a label.

    Inputs
    -------
    exog_sims : numpy.ndarray
  
        Simulated exogenous paths, shape (S, H, Kx), already scaled if needed.
  
    k : int
  
        Number of clusters K.
  
    rng : numpy.random.Generator
  
        RNG for reproducibility; used to seed KMeans random_state.

    Returns
    -------
    labels : numpy.ndarray
  
        Integer labels in [0, K-1] of shape (S,), mapping each path to a centroid.
  
    cents : numpy.ndarray
  
        Centroids with shape (K, H, Kx).
  
    w : numpy.ndarray
        Cluster weights of length K, where w_c is the fraction of simulations in cluster c.

    Notes
    -----
    - Complexity is O(S * H * Kx * iterations) for K-means.
    - These centroids enable precomputing forecasts for jump=0 and jump=1 only K times
      per model instead of S times.
    """
   
    S, H, Kx = exog_sims.shape
   
    X = exog_sims.reshape(S, H*Kx)
   
    km = KMeans(n_clusters = k, n_init = 5, random_state = int(rng.integers(1e9)))
   
    labels = km.fit_predict(X)
   
    cents = km.cluster_centers_.reshape(k, H, Kx)
   
    w = np.bincount(labels, minlength = k).astype(float)
   
    w /= w.sum()
   
    return labels, cents, w


@njit
def _assemble_prices(
    resid, 
    boot_idx_row, 
    t_scales_row,
    mu_row, cp, 
    eps_scale, 
    lb,
    ub
):
    """
    Assemble a terminal price from model-implied conditional mean returns and bootstrapped
    residual shocks, with optional Student-t scaling, using geometric compounding.

    Computation:

        For horizon H:

            r_h = mu_row[h] + eps_scale * shock_h,

        where 
        
            shock_h = resid[boot_idx_row[h]] * t_scales_row[h] 
        
        if t_scales_row is provided, else 
        
            shock_h = resid[boot_idx_row[h]].

        The terminal price is:

            P_T = cp * exp( sum_{h=0..H-1} r_h ).

        Finally P_T is clipped to [lb, ub].

    Inputs
    -------
    resid : numpy.ndarray
        Residual vector from a fitted SARIMAX model, length N.
    boot_idx_row : numpy.ndarray
        Indices selecting a block-bootstrap path into resid, length H.
    t_scales_row : numpy.ndarray or None
        Optional length-H Student-t scale multipliers; use None to disable fat tails.
    mu_row : numpy.ndarray
        Conditional mean returns for the H-step horizon, length H.
    cp : float
        Current price at time 0.
    eps_scale : float
        Scalar scale factor that calibrates residual volatility to match H-step
        uncertainty empirically.
    lb, ub : float
        Lower and upper clipping bounds for the terminal price.

    Returns
    -------
    float
        Clipped terminal price for this simulation.
    """
    
    H = mu_row.shape[0]

    acc = 0.0

    price = cp

    for h in range(H):

        boot = resid[boot_idx_row[h]]

        if t_scales_row is not None:

            boot *= t_scales_row[h]

        r = mu_row[h] + eps_scale * boot

        acc += r

    price = cp * np.exp(acc)

    if price < lb:

        price = lb

    elif price > ub:

        price = ub

    return price


def make_jump_indicator_from_returns(
    y: pd.Series, 
    q: float = JUMP_Q
) -> Tuple[pd.Series, float]:
    """
    Construct a binary jump indicator based on extreme returns and estimate the jump
    probability.

    Method:
        Let thr be the q-quantile of |y_t|. Define jump_ind_t = 1 if |y_t| >= thr
        and 0 otherwise. The empirical jump probability is p_jump = mean(jump_ind_t).

    Inputs
    -------
    y : pandas.Series
        Return series (e.g., log returns), indexed by time.
    q : float
        Quantile threshold in (0, 1). Typical values near 0.95 to 0.99 define
        rare, large-magnitude moves.

    Returns
    -------
    jump_ind : pandas.Series
        Binary indicator series with the same index as y.
    p_jump : float
        Empirical frequency of jumps (mean of the indicator).

    Notes
    -----
    - This nonparametric rule is robust and easy to calibrate. The indicator can be
      included as an exogenous regressor in SARIMAX or used to draw Bernoulli jump
      paths in simulation.
    """
    
    thr = y.abs().quantile(q)

    jump_ind = (y.abs() >= thr).astype(float)

    p_jump = float(jump_ind.mean())

    return jump_ind, p_jump


def moving_block_bootstrap(
    resid: np.ndarray,
    length: int, 
    block: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a bootstrap time series by concatenating random contiguous blocks of the
    residuals to preserve short-run dependence.

    Method:
        If N < block, sample IID with replacement. Otherwise, draw starting indices
        s_1, ..., s_B uniformly from [0, N - block], take blocks resid[s_b : s_b + block],
        concatenate until reaching the desired length, and truncate to 'length'.

    Inputs
    -------
    resid : numpy.ndarray
        Vector of residuals of length N.
    length : int
        Desired output length H.
    block : int
        Block size L for dependence preservation.
    rng : numpy.random.Generator
        RNG for reproducibility.

    Returns
    -------
    numpy.ndarray
        Bootstrap series of length H.

    Notes
    -----
    - This function is not used when pre-drawn bootstrap indices are employed, but
      documents the intended resampling model: a stationary bootstrap approximation
      via fixed-length blocks.
    """
    
    n = resid.shape[0]

    if n < block:

        return rng.choice(resid, size = length, replace = True)

    starts = rng.integers(0, n - block + 1, size = int(np.ceil(length / block)))
   
    out = np.concatenate([resid[s:s + block] for s in starts])
   
    return out[:length]


def calibrate_eps_scale_from_resid(
    resid: np.ndarray, 
    horizon: int, 
    y: pd.Series
) -> float:
    """
    Calibrate a scalar residual scale factor (eps_scale) so that the standard deviation
    of the H-step sum of residual shocks matches the empirical standard deviation of
    H-step sums of actual returns.

    Idea:
    
        We want std( sum_{h=1..H} eps_scale * e_{t+h} ) ≈ std( sum_{h=1..H} y_{t+h} ),
        which implies:
           
            eps_scale ≈ std( H-sum of y ) / std( H-sum of residuals ).

    Implementation:
       
        - Compute rolling sums over horizon H via a convolution with a length-H vector
          of ones. Let rs be the H-sum of residuals and y_H be the H-sum of y.
       
        - Set rs_std = std(rs) and err_std = std(y_H). Return
              eps_scale = clip(err_std / rs_std, 1e-3, 10.0),
          with guards for small-sample or degenerate cases.

    Inputs
    -------
    resid : numpy.ndarray
        Residual vector e_t from a fitted SARIMAX model.
    horizon : int
        H-step horizon in periods (weeks).
    y : pandas.Series
        Observed return series, aligned to the same frequency as residuals.

    Returns
    -------
    float
        Scale factor eps_scale.

    Notes
    -----
    - This replaces costly rolling-origin cross-validation with a moment-matching
      calibration that uses a single fit.
    - If rs_std is extremely small or non-finite, the function returns 1.0.
    """
      
    resid = np.asarray(resid, float)
  
    n = resid.size
  
    if n < max(5, horizon):
  
        return 1.0

    ker = np.ones(horizon, dtype = float)
  
    rs = np.convolve(resid, ker, mode = "valid")
    
    if rs.size > 1:
        
        rs_std = float(np.std(rs, ddof = 1)) 
    
    else:
        
        rs_std = float(np.std(resid) * np.sqrt(horizon))

    y_H = np.convolve(y.values.astype(float), ker, mode = "valid")
    
    if y_H.size > 1:
        
        err_std = float(np.std(y_H, ddof = 1)) 
        
    else:
        
        err_std = rs_std

    if rs_std <= 1e-12 or not np.isfinite(rs_std) or not np.isfinite(err_std):
   
        return 1.0
   
    return float(np.clip(err_std / rs_std, 1e-3, 10.0))


def _stack_ridge_weights(
    fits: List, 
    y: pd.Series,
    alpha: float = RIDGE_ALPHA
) -> Optional[np.ndarray]:
    """
    Compute ensemble weights by stacking 1-step-ahead predictions using ridge
    regression, then project to the simplex.

    Construction:
     
        For each fitted SARIMAX model m, obtain the 1-step-ahead prediction series
        p_m,t over the in-sample period (dynamic=False). Stack these into a T x M
        matrix P with columns p_m. Solve the ridge problem:
       
            minimise_w || y - P w ||_2^2 + alpha * || w ||_2^2,
       
        with intercept fitted internally. Extract the coefficient vector w, clamp
        negative entries to zero, and renormalise to sum to 1.

    Inputs
    -------
    fits : list
        List of fitted SARIMAXResults for M candidate models.
    y : pandas.Series
        Target series of returns aligned to the prediction matrix.
    alpha : float
        L2 regularisation strength (RIDGE_ALPHA).

    Returns
    -------
    w : numpy.ndarray or None
        Length-M nonnegative weights summing to 1, or None if stacking is not feasible.

    Notes
    -----
    - Requires that the number of aligned observations T exceeds M by a small margin
      (here T >= M + 3) for numerical stability.
    - If any model fails to produce predictions, a column of NaNs is inserted and
      dropped via alignment.
    """
    
    if not fits:
    
        return None

    preds = []

    for res in fits:

        try:

            pm = res.get_prediction(dynamic = False)  

            s = pm.predicted_mean

            preds.append(s.rename(None))

        except Exception:

            preds.append(pd.Series(index = y.index, dtype = float))  

    P = pd.concat(preds, axis = 1)
  
    P.columns = range(P.shape[1])
  
    Y, X = y.align(P, join = "inner")
  
    X = X.dropna(axis = 0, how = "any")
  
    Y = Y.loc[X.index]

    if X.shape[0] < X.shape[1] + 3: 
  
        return None

    try:
        
        reg = Ridge(alpha = alpha, fit_intercept = True, positive = False)
        
        reg.fit(X.values, Y.values)
        
        w = reg.coef_.astype(float)

        w = np.maximum(w, 0.0)

        s = w.sum()

        if s <= 0 or not np.isfinite(s):
       
            return None

        w /= s

        return w
 
    except Exception:
 
        return None


def _predraw_jump_paths(
    n_sims: int, 
    horizon: int, 
    p_jump: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Pre-draw Bernoulli jump indicators for all simulations and horizons.

    Model:
    
        For each simulation i and step h, draw 
        
            J_{i,h} ~ Bernoulli(p_jump), 
            
        independently. Values are 1 with probability p_jump and 0 otherwise.

    Inputs
    -------
    n_sims : int
        Number of simulations S.
    horizon : int
        Forecast horizon H.
    p_jump : float
        Jump probability in [0, 1].
    rng : numpy.random.Generator
        RNG for reproducibility.

    Returns
    -------
    numpy.ndarray
        Binary array of shape (S, H) containing jump indicators.
    """
    
    return (rng.uniform(size = (n_sims, horizon)) < p_jump).astype(float)


def _predraw_bootstrap_indices(
    resid: np.ndarray,
    horizon: int,
    block: int,
    n_sims: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Pre-compute indices into the residual vector to realise moving-block bootstrap
    shock paths for all simulations.

    Model:
        
        For each simulation i, concatenate random contiguous blocks of length `block`
        selected uniformly from [0, N - block], where N is len(resid), until reaching
        length H. Truncate to exactly H. If N < block, sample IID indices with replacement.

    Inputs
    -------
    resid : numpy.ndarray
        Residual vector of length N.
    horizon : int
        Output path length H.
    block : int
        Block size L for the bootstrap.
    n_sims : int
        Number of simulations S.
    rng : numpy.random.Generator
        RNG for reproducibility.

    Returns
    -------
    numpy.ndarray
        Integer index array of shape (S, H) to index into `resid` for each shock.
    """
    
    n = resid.shape[0]
   
    out_idx = np.empty((n_sims, horizon), dtype = int)

    if n < block:

        for i in range(n_sims):

            out_idx[i] = rng.integers(0, n, size = horizon)

        return out_idx

    num_blocks = int(np.ceil(horizon / block))

    max_start = n - block

    for i in range(n_sims):

        starts = rng.integers(0, max_start + 1, size = num_blocks)

        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:horizon]

        out_idx[i, :len(idx)] = idx

    return out_idx


def _predraw_t_scales(
    n_sims: int, 
    horizon: int, 
    df: int, 
    rng: np.random.Generator
) -> Optional[np.ndarray]:
    """
    Pre-draw scale multipliers to induce Student-t-like fat tails in the bootstrapped
    shocks, normalised to unit variance.

    Construction:
        For degrees of freedom nu = df > 2, draw 
        
            chi2_{i,h} ~ ChiSquare(nu)
        
        and set
        
            s_{i,h} = sqrt(nu / chi2_{i,h}). 
        
        Then normalise so that E[s_{i,h}^2] = 1 by dividing by sqrt(nu / (nu - 2)). 
        If df <= 2, return None to disable t-scaling.

    Inputs
    -------
    n_sims : int
        Number of simulations S.
    horizon : int
        Forecast horizon H.
    df : int
        Degrees of freedom nu for the Student-t proxy (must exceed 2 for finite variance).
    rng : numpy.random.Generator
        RNG for reproducibility.

    Returns
    -------
    numpy.ndarray or None
        Array of shape (S, H) of scales, or None if df <= 2.
    """
    
    if not (df and df > 2):
    
        return None
    
    chi2 = rng.chisquare(df, size = (n_sims, horizon))
    
    scales = np.sqrt(df / chi2)
    
    scales /= np.sqrt(df / (df - 2.0))
    
    return scales


def _prune_ensemble(
    ens: Ensemble, 
    max_models: int = 3
) -> Ensemble:
    """
    Keep only the top `max_models` models by weight and renormalise, to reduce
    forecast time while preserving most of the ensemble mass.

    Inputs
    -------
    ens : Ensemble
        Fitted ensemble with M models and weights.
    max_models : int
        Number of models to retain (e.g., 2 or 3).

    Returns
    -------
    Ensemble
        Pruned ensemble with at most `max_models` fits and weights summing to 1.

    Notes
    -----
    - This is safe when the weight distribution is concentrated. Computational cost
      scales with the number of retained models, so pruning gives a near-linear speedup.
    """

    idx = np.argsort(ens.weights)[::-1][:max_models]
  
    w = ens.weights[idx]
  
    w = w / w.sum()
  
    fits = [ens.fits[i] for i in idx]
  
    return Ensemble(
        fits = fits,
        weights = w
    )


def simulate_price_paths_for_ticker(
    tk: str,
    df_tk: pd.DataFrame,
    exog_cols: List[str],
    cp: float,
    lb: float,
    ub: float,
    exog_sims: Optional[np.ndarray],
    horizon: int,
    rng_seed: int,
) -> Dict[str, float]:
    """
    Simulate terminal price quantiles for one ticker by combining a SARIMAX ensemble,
    scenario exogenous paths (macro and factors), Bernoulli jump indicators, and a
    block-bootstrap shock process with optional Student-t scaling. 
   
    Residual volatility is calibrated to match H-step uncertainty via rolling sums.

    Workflow:
      
        1) Standardise present exogenous columns exog_cols using a StandardScaler fit
           on the ticker's history, append the jump indicator, and fit a SARIMAX
           ensemble on y_t (log returns):
      
               y_t = c + AR(p) + MA(q) + beta' x_t + e_t.
      
        2) Prune to the top few models by weight. Extract residuals from the modal
           (highest-weight) fit and compute eps_scale such that:
      
               std( sum_{h=1..H} eps_scale * e_{t+h} ) ≈ std( sum_{h=1..H} y_{t+h} ).
      
        3) For each simulation:
      
           - Prepare a future exog path X_future[i] of shape (H, Kx); if clustering is
             enabled, map to a centroid sequence.
      
           - Incorporate jump paths J_{i,h} ~ Bernoulli(p_jump). Since SARIMAX is
             linear in exog, precompute forecasts for two settings (jump=0, jump=1)
             at the centroid and combine as:
      
                 mu_h(i) = mu_h(jump=0) + (mu_h(jump=1) - mu_h(jump=0)) * J_{i,h}.
      
           - Draw bootstrap residual shocks using moving-block indices and optionally
             multiply by t scales to fatten tails.
      
           - Form the return path:
      
                 r_h(i) = mu_h(i) + eps_scale * shock_h(i).
      
           - Compound geometrically:
      
                 P_T(i) = cp * exp( sum_{h=1..H} r_h(i) ),
      
             then clip to [lb, ub].
      
        4) Aggregate across simulations and report p5, p50, p95 quantiles, mean
           return E[P_T / cp - 1], and standard deviation of returns.

    Inputs
    -------
    tk : str
        Ticker identifier (for logging only).
    df_tk : pandas.DataFrame
        Ticker frame with columns: "price", "y" (log returns), exog_cols, "jump_ind".
        Indexed by weekly dates.
    exog_cols : list of str
        List of present exogenous driver column names for this ticker.
    cp : float
        Current price P_0.
    lb, ub : float
        Lower and upper clamps for the terminal price.
    exog_sims : numpy.ndarray or None
        Pre-simulated exogenous paths of shape (S, H, len(exog_cols)). If None,
        zeros are used (effectively a no-change baseline for exog).
    horizon : int
        H-step horizon in periods (weeks).
    rng_seed : int
        Seed for per-ticker simulation reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
            "low"   : 5th percentile of terminal price,
            "avg"   : 50th percentile of terminal price,
            "high"  : 95th percentile of terminal price,
            "returns": mean of (P_T / cp - 1) over simulations,
            "se"    : standard deviation of (P_T / cp - 1) over simulations.

    Notes
    -----
    - Exogenous clustering reduces forecast calls from S to roughly K_CLUSTERS,
      since SARIMAX is not vectorised over exog paths.
    - The model index per simulation is sampled from the ensemble weights, which
      approximates a mixture-of-models predictive distribution.
    - All scaling is performed on the historical distribution of exog_cols, applied
      to simulated exog paths to preserve comparability.
    """

    scaler = StandardScaler().fit(df_tk[exog_cols].values)
   
    X_full = scaler.transform(df_tk[exog_cols].values)
  
    X_full = np.column_stack([X_full, df_tk[["jump_ind"]].values])

    ens = _fit_sarimax_candidates(
        y = df_tk["y"],
        X = X_full
    )
    
    ens = _prune_ensemble(
        ens = ens
    )
    
    best_idx = int(np.argmax(ens.weights))
  
    best = ens.fits[best_idx]
  
    resid = pd.Series(best.resid, index = df_tk.index).dropna().values

    eps_scale = calibrate_eps_scale_from_resid(
        resid = resid, 
        horizon = horizon, 
        y = df_tk["y"]
    )
    
    p_jump = float(df_tk["jump_ind"].mean())

    rng = np.random.default_rng(rng_seed)
    
    if exog_sims is not None:
        
        n_sims = exog_sims.shape[0]  
    
    else:
        
        n_sims = N_SIMS


    if exog_sims is None:

        X_future_all = np.zeros((n_sims, horizon, len(exog_cols)), dtype = float)

    else:

        X_future_all = exog_sims  

    X_scaled_all = scaler.transform(X_future_all.reshape(-1, len(exog_cols))).reshape(n_sims, horizon, len(exog_cols))
    
    if n_sims > K_CLUSTERS:
       
        labels, cents_scaled, _ = cluster_exog_paths(
            exog_sims = X_scaled_all, 
            k = K_CLUSTERS,
            rng = rng
        )

        mu_cents0 = np.empty((len(ens.fits), K_CLUSTERS, horizon), dtype = float)

        mu_cents1 = np.empty((len(ens.fits), K_CLUSTERS, horizon), dtype = float)

        for m_idx, res in enumerate(ens.fits):

            for c_idx in range(K_CLUSTERS):

                exog0 = np.column_stack([cents_scaled[c_idx], np.zeros((horizon, 1))])

                exog1 = np.column_stack([cents_scaled[c_idx], np.ones((horizon, 1))])

                mu_cents0[m_idx, c_idx] = res.get_forecast(steps = horizon, exog = exog0).predicted_mean.values

                mu_cents1[m_idx, c_idx] = res.get_forecast(steps = horizon, exog = exog1).predicted_mean.values

    else:
     
        labels = None

    jump_paths_all = _predraw_jump_paths(
        n_sims = n_sims, 
        horizon = horizon,
        p_jump = p_jump, 
        rng = rng
    )

    boot_idx_all = _predraw_bootstrap_indices(
        resid = resid, 
        horizon = horizon, 
        block = RESID_BLOCK,
        n_sims = n_sims,
        rng = rng
    )

    t_scales_all = _predraw_t_scales(
        n_sims = n_sims, 
        horizon = horizon,
        df = T_DOF,
        rng = rng
    )

    mdl_idx = rng.choice(len(ens.fits), size = n_sims, p = ens.weights)

    final_prices = np.empty(n_sims, dtype = float)

    for i in range(n_sims):
       
        m_idx = mdl_idx[i]
       
        if labels is not None:
       
            c_idx = labels[i]
       
            mu0 = mu_cents0[m_idx, c_idx]
       
            mu1 = mu_cents1[m_idx, c_idx]
      
            mu = mu0 + (mu1 - mu0) * jump_paths_all[i]
      
        else:
      
            exog = np.column_stack([X_scaled_all[i], jump_paths_all[i].reshape(-1, 1)])
      
            res = ens.fits[m_idx]
      
            mu = res.get_forecast(steps = horizon, exog = exog).predicted_mean.values


        boot = resid[boot_idx_all[i]]
      
        if t_scales_all is not None:
      
            trow = t_scales_all[i]
      
            final_prices[i] = _assemble_prices(
                resid = resid, 
                boot_idx_row = boot_idx_all[i], 
                t_scales_row = trow, 
                mu_row = mu, 
                cp = cp, 
                eps_scale = eps_scale,
                lb = lb, 
                ub = ub
            )

        else:
          
            r_path = mu + eps_scale * boot
          
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
        "se": ret_std,
    }



def _process_ticker(
    tk: str,
    close_idx,
    close_col,
    latest_price: float,
    raw_macro: pd.DataFrame,
    MACRO_COLS: List[str],
    factors_w: pd.DataFrame,
    EXOG_COLS: List[str],
    JUMP_Q: float,
    forecast_period: int,
    lb_tk: float,
    ub_tk: float,
    country_map: Dict[str, str],
    country_exog_paths: Dict[str, Optional[np.ndarray]],
    rng_seed: int
) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    End-to-end pipeline for a single ticker: construct the modelling dataset,
    create jump indicators, align simulated exogenous paths to the ticker’s
    exog set, run Monte Carlo price simulations, and return summary statistics.

    Steps:
    
        1) Build a price and log-return frame 'dfp' from close_idx, close_col.
    
        2) Merge macro levels 'tm' by date using merge_asof, convert to weekly
           frequency, forward/backward fill, drop incomplete rows.
    
        3) Convert macro levels to deltas via mc._macro_levels_to_deltas for
           the columns MACRO_COLS; join with factor returns 'fct' on the same index.
    
        4) Construct the combined dataframe 'dfr' with ["price", "y"] + present_exogs.
           Determine 'present_exogs' as those EXOG_COLS actually available in dfr.
    
        5) Build a binary jump indicator from returns using make_jump_indicator_from_returns
           with threshold quantile JUMP_Q and add to the frame.
    
        6) Ensure enough history is available (at least forecast_period + 32 points).
    
        7) Look up the country of the ticker and retrieve the country exog simulation
           cube; reduce to ticker-specific present_exogs.
    
        8) Call simulate_price_paths_for_ticker to obtain low/median/high prices,
           mean return, and Monte Carlo standard deviation.
    
        9) Log a concise summary and return (ticker, result_dict).

    Inputs
    -------
    tk : str
        Ticker code.
    close_idx : array-like
        DatetimeIndex or compatible array of weekly dates for the price series.
    close_col : array-like
        Price levels aligned with close_idx.
    latest_price : float
        Most recent price (used as cp).
    raw_macro : pandas.DataFrame
        Long panel of macro levels that includes columns "ticker", "ds", and MACRO_COLS.
    MACRO_COLS : list of str
        Macro columns to use (levels that will be differenced).
    factors_w : pandas.DataFrame
        Weekly factor dataframe with columns matching FACTOR_COLS.
    EXOG_COLS : list of str
        Global exogenous column names; ticker uses the subset present in its data.
    JUMP_Q : float
        Quantile for jump thresholding.
    forecast_period : int
        Horizon H in weeks.
    lb_tk, ub_tk : float
        Lower/upper clamps for terminal price.
    country_map : dict
        Mapping {ticker -> country} for selecting country-level exog simulations.
    country_exog_paths : dict
        Mapping {country -> exog simulation cube of shape (S, H, len(EXOG_COLS))}.
    rng_seed : int
        Seed used for the ticker’s simulation.

    Returns
    -------
    (str, dict or None)
        The ticker and a result dictionary as returned by simulate_price_paths_for_ticker,
        or (ticker, None) if insufficient data.
    """
    
    if not np.isfinite(latest_price):
    
        logger.warning("No price for %s, skipping", tk)
    
        return tk, None

    dfp = pd.DataFrame({
        "ds": close_idx, 
        "price": close_col
    })

    dfp["y"] = np.log(dfp["price"]).diff()

    dfp.dropna(inplace = True)
    
    dfp.set_index("ds", inplace = True)

    tm = raw_macro[raw_macro["ticker"] == tk][["ds"] + MACRO_COLS]

    dfm_levels = (
        pd.merge_asof(dfp[[]].reset_index().sort_values("ds"), tm.sort_values("ds"), on = "ds")
        .set_index("ds")
        .asfreq("W-SUN")
        .ffill()
        .bfill()
        .dropna()
    )

    dfm_d = mc._macro_levels_to_deltas(
        df_levels = dfm_levels, 
        cols = MACRO_COLS
    )

    fct = factors_w.reindex(dfp.index).ffill()

    dfr = dfp.join(dfm_d, how = "inner").join(fct, how = "inner").dropna()
    
    present_exogs = [c for c in EXOG_COLS if c in dfr.columns]
    
    if any(c not in dfr.columns for c in EXOG_COLS):

        logger.warning("Missing exog columns for %s: %s", tk, [c for c in EXOG_COLS if c not in dfr.columns])

    dfr = dfr[["price", "y"] + present_exogs].dropna()

    jump_ind, p_jump = make_jump_indicator_from_returns(
        y = dfr["y"], 
        q = JUMP_Q
    )

    dfr["jump_ind"] = jump_ind.astype(float)

    if len(dfr) < forecast_period + 32:

        logger.warning("Insufficient history for %s, skipping", tk)

        return tk, None

    ctry = country_map.get(tk, None)

    exog_sims = country_exog_paths.get(ctry)
  
    exog_sims_tk = None
  
    if exog_sims is not None:
  
        idx = [EXOG_COLS.index(c) for c in present_exogs]
  
        exog_sims_tk = exog_sims[:, :, idx]

    out = simulate_price_paths_for_ticker(
        tk = tk,
        df_tk = dfr[["price","y"] + present_exogs + ["jump_ind"]],
        exog_cols = present_exogs,
        cp = float(latest_price),
        lb = float(lb_tk),
        ub = float(ub_tk),
        exog_sims = exog_sims_tk,
        horizon = forecast_period,
        rng_seed = int(rng_global.integers(1_000_000_000)),
    )
    
    logger.info("%s -> p5 %.2f, p50 %.2f, p95 %.2f, Rets %.2f, MC-σ %.4f", tk, out["low"], out["avg"], out["high"], out['returns'], out["se"])
    
    return tk, out


def main() -> None:
    """
    Orchestrate the full workflow:
     
        - Load macro and market data.
     
        - Build country-level joint exogenous simulation cubes via a BVAR with a
          Minnesota prior on differenced macro and factor drivers.
     
        - For each ticker, run the per-ticker pipeline in parallel to produce
          Monte Carlo price summaries.
     
        - Collect and format results.

    Country-level exogenous simulations:
     
        For each country, construct dX with columns in EXOG_COLS (macro deltas plus
        factor returns), aligned on weekly dates. Fit a BVAR(p) with Minnesota prior
        (via mc._fit_bvar_minnesota), then simulate S exogenous paths of length H
        using antithetic normal innovations. If a country lacks sufficient history,
        use a flat exog cube of zeros as a fallback (baseline).

    Parallel execution:
     
        Build a list of tasks, each calling _process_ticker with serialisable inputs,
        and execute them with joblib. Results are collected into a dataframe with
        current price, price quantiles, expected return, and Monte Carlo standard error.

    Logging and output:
     
        Intermediate steps log the status of BVAR simulation and per-ticker results.
        The final dataframe is constructed but not exported by default.

    Notes
    -----
    - Weekly frequency ("W-SUN") is used consistently for macro, factors, prices,
      and SARIMAX models to ensure alignment.
    - Randomness:
        * rng_global seeds BVAR simulations per country.
        * Each ticker’s simulation uses its own seed drawn from rng_global.
    - This entry point does not take arguments; configuration is controlled by
      module-level constants such as FORECAST_WEEKS, N_SIMS, and K_CLUSTERS.
    """
       
    macro = MacroData()
   
    r = macro.r

    tickers: List[str] = list(config.tickers)
   
    forecast_period: int = FORECAST_WEEKS

    close = r.weekly_close
   
    latest_prices = r.last_price
   
    analyst = r.analyst

    lb = config.lbr * latest_prices
   
    ub = config.ubr * latest_prices 

    factors_w = mc._load_factors_weekly(
        use_ff5 = USE_FF5
    )

    logger.info("Importing macro history …")
    
    raw_macro = macro.assign_macro_history_large_non_pct().reset_index()

    raw_macro = raw_macro.rename(columns = {"year": "ds"} if "year" in raw_macro.columns else {raw_macro.columns[1]: "ds"})
    
    raw_macro["ds"] = raw_macro["ds"].dt.to_timestamp()

    country_map = {t: str(c) for t, c in zip(analyst.index, analyst["country"])}

    raw_macro["country"] = raw_macro["ticker"].map(country_map)

    macro_clean = raw_macro[["ds", "country"] + MACRO_COLS].dropna()

    logger.info("Simulating joint macro+factor exogenous scenarios …")

    country_exog_paths: Dict[str, Optional[np.ndarray]] = {}

    for ctry, dfc in macro_clean.groupby("country"):

        dfm_levels_w = (
            dfc.set_index("ds")[MACRO_COLS]
               .sort_index()
               .resample("W-SUN")
               .mean()
               .ffill()
               .dropna()
        )

        if dfm_levels_w.empty:

            logger.warning("No macro levels for %s; using flat exogs.", ctry)

            country_exog_paths[ctry] = np.zeros((N_SIMS, forecast_period, len(EXOG_COLS)))

            continue

        dfm_d = mc._macro_levels_to_deltas(
            dfm_levels_w,
            cols = MACRO_COLS
        )

        fct = factors_w.reindex(dfm_d.index).ffill().dropna()

        common_idx = dfm_d.index.intersection(fct.index)

        dfm_d = dfm_d.loc[common_idx]

        fct = fct.loc[common_idx]

        dX = pd.concat([dfm_d[[c for c in dfm_d.columns if c in EXOG_COLS]], fct[[c for c in FACTOR_COLS if c in fct.columns]]], axis = 1)

        dX = dX[[c for c in EXOG_COLS if c in dX.columns]].dropna()

        if dX.shape[0] < BVAR_P + 8:

            logger.warning("Too little history to fit joint BVAR for %s; using flat exogs.", ctry)

            country_exog_paths[ctry] = np.zeros((N_SIMS, forecast_period, len(EXOG_COLS)))

            continue

        try:

            sims = simulate_joint_exog_paths(
                dX = dX, 
                steps = forecast_period, 
                n_sims = N_SIMS, 
                seed = int(rng_global.integers(1_000_000_000))
            )

     
            country_exog_paths[ctry] = sims
     
        except Exception as e:
     
            logger.warning("Joint exog simulation failed for %s (%s). Using flat exogs.", ctry, e)
           
            country_exog_paths[ctry] = np.zeros((N_SIMS, forecast_period, len(EXOG_COLS)))

    logger.info("Fitting SARIMAX ensemble and running Monte Carlo …")

    tasks = []
   
    for tk in tickers:
   
        tasks.append(delayed(_process_ticker)(
            tk = tk,
            close_idx = close.index,
            close_col = close[tk].values,
            latest_price = float(latest_prices.get(tk, np.nan)),
            raw_macro = raw_macro,
            MACRO_COLS = MACRO_COLS,
            factors_w = factors_w,
            EXOG_COLS = EXOG_COLS,
            JUMP_Q = JUMP_Q,
            forecast_period = forecast_period,
            lb_tk = float(lb.get(tk, -np.inf)),
            ub_tk = float(ub.get(tk, np.inf)),
            country_map = country_map,
            country_exog_paths = country_exog_paths,
            rng_seed = int(rng_global.integers(1_000_000_000))
        ))



    results: Dict[str, Dict[str, float]] = {}
    
    for tk, res in Parallel(n_jobs = N_JOBS, prefer = "processes")(tasks):

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
        sheets = {"SARIMAX Factor": df_out},
        output_excel_file = "MODEL_FILE",
    )
    
    logger.info("Run completed.")

if __name__ == "__main__":
  
    main()
