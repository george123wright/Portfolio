from __future__ import annotations

"""
Monte-Carlo Price Forecasting with Macro-Driven SARIMAX Ensembles, ECVAR/BVAR Macro
Scenarios, Stochastic/Regime-Switching Volatility, and POT-GPD Jumps
==================================================================================

Overview
--------
This module implements an end-to-end pipeline to simulate future price distributions
for multiple tickers over a fixed horizon (e.g., 52 weeks). The approach couples
asset-level SARIMAX models (with exogenous macroeconomic deltas, Fourier seasonality,
and a jump indicator) to macro-scenario generators fitted at the country level
(ECVAR, Minnesota-BVAR, and VAR). Volatility dynamics are enriched using both
stochastic volatility (SV) and a 2-state Markov-switching (MS) volatility layer.
Tail risk is modelled via peaks-over-threshold (POT) Generalised Pareto (GPD)
shocks conditioned on the latent MS state. The final price distribution is obtained
by Monte-Carlo propagation of the SARIMAX ensemble conditional on simulated
exogenous paths and jump processes, blending model-based and bootstrap noise.

Notation
--------
Time index t = 1, …, H (forecast horizon). Let:
- y_t            : asset log return at time t.
- P_t            : asset price; P_t = P_{t−1} × exp(y_t).
- x_t            : exogenous feature vector (lagged macro deltas, jump flag, Fourier terms).
- ΔX_t ∈ R^k     : vector of macro deltas (Interest, Cpi, Gdp, Unemp), k = 4.
- L_t ∈ R^k      : macro levels (often log levels for price-like variables).
- S_t ∈ {0,1}    : MS volatility state (0 = calm, 1 = stressed).
- J_t ∈ {0,1}    : jump indicator.
- ε_t            : model innovation.

Core components and equations
-----------------------------
1) **SARIMAX with exogenous regressors (per ticker)**
  
   The observation equation is
  
       y_t = μ + Σ_{i=1..p} φ_i y_{t−i} + Σ_{j=1..q} θ_j ε_{t−j} + x_t' β + ε_t,
  
       ε_t ~ N(0, σ^2),
  
   with seasonality captured by Fourier terms (period 52, K harmonics) rather than
   seasonal ARIMA. Candidate orders (p,0,q) are fitted and AICc-weighted to form
   an ensemble. Optionally, non-negative stacking weights are learned by
   rolling-origin cross-validation targeting H-step sums.

   Ensemble mixture for step-wise predictive moments:
  
       μ_mix,t = Σ_k w_k μ_{k,t},
  
       Var_mix,t = Σ_k w_k [ Var_{k,t} + (μ_{k,t} − μ_mix,t)^2 ].

2) **Macro dynamics (per country)**
  
   a) **ECVAR(1):** When cointegration is indicated (Johansen test), the system
      in differences is
  
          ΔX_t = c + A ΔX_{t−1} + K (β' L_{t−1}) + η_t,   η_t ~ N(0, Σ),
  
      and levels evolve as L_t = L_{t−1} + ΔX_t. The cointegration vectors β
      encode long-run equilibria; K are adjustment loadings.

   b) **Minnesota-BVAR(p):** Zero-mean Minnesota prior on stacked coefficients B,
      with variances
  
          Var(B on var j at lag ℓ in eq i) = (λ1 / ℓ^{λ3})^2 × [ 1 if i=j; (λ2 × σ_i / σ_j) if i≠j ],
        
          Var(constant in eq i) = (λ1 λ4 σ_i)^2.
     
      Posterior is matrix-normal–inverse-Wishart. Hyperparameters (λ1..λ4) are
      tuned by empirical Bayes (maximising marginal evidence) or by maximising
      predictive log-score on a holdout.

   c) **VAR(1) fallback:** ΔX_t = c + A ΔX_{t−1} + η_t, for robustness.

   Candidates are weighted by an information-criterion proxy:

       IC = T × log |Σ̂| + pcount × log T,

   soft-maxed to produce model-mixing weights; an optional regime-aware reweighting
   uses HMM posteriors (see below).

3) **Volatility layers**

   a) **Stochastic volatility (SV):** For residual component r_tj per dimension j,
      proxy log-variance h_tj = log(r_tj^2 + ε) follows AR(1):

          h_tj = μ_j + φ_j (h_{t−1,j} − μ_j) + σ_{η,j} e_tj,   e_tj ~ N(0,1).

      Simulated scales are S_tj = exp(0.5 h_tj).

   b) **Markov-switching (MS) volatility:** Residuals follow

          r_t | S_t = s ~ N(0, g_s^2 Σ),

      with S_t a 2-state Markov chain (transition P, initial π). EM with the
      forward–backward algorithm estimates (P, g, π, Σ). Simulated state paths
      produce a scale g_{S_t}. The macro simulator uses the elementwise product
      of SV and MS scales; ticker-level residual noise is also scaled by g_{S_t}.

4) **Tail risk and jumps**
  
   a) **Jump identification:** From historical y_t, define a high-quantile threshold
  
      τ = quantile_q(|y_t|); set J_t = 1{|y_t| ≥ τ}.

   b) **State-conditional tails (POT-GPD):** For exceedances e = |r| − τ > 0,
      fit GPD(e | ξ, β) (location fixed at 0) by MLE, optionally weighted by
      HMM posteriors γ_t(s) for state-conditional fits. Also estimate p_neg, the
      proportion of negative signs on exceedances, to randomise shock signs.

   c) **Jump simulation:** For each t, draw J_t ~ Bernoulli(p_jump[S_t]).
      If J_t = 1, draw 
      
        |shock_t| = τ_{S_t} + GPD_rv(ξ_{S_t}, β_{S_t}) 
        
      and assign sign negative with probability p_neg_{S_t}.

5) **Noise blending and bootstrap**

   Rolling-origin CV provides:

   - v_fore_mean: average model-based forecast variance of H-step sums;

   - rs_std: standard deviation of rolling H-sum residuals, a bootstrap variance proxy.

   Blend model and bootstrap noise via

       η_t = α η_model,t + (1 − α) η_boot,t,

   where α minimises α^2 V_fore + (1 − α)^2 V_boot on a small grid, V_boot = rs_std^2.

   The bootstrap uses the stationary bootstrap (Politis–Romano) on residuals and
   can be fattened by Student-t scale factors

       s = sqrt(ν / χ^2_ν) / sqrt(ν / (ν − 2)),

   so E[s^2] = 1 for ν > 2.

6) **Exogenous construction and seasonality**
   Exogenous matrices include lagged macro deltas ΔX_{t−L}, a jump indicator, and
   Fourier sin/cos pairs for weekly seasonality. At forecast time, the future
   design X_f is built strictly in the training column order and standardised
   using the training scaler, ensuring positional alignment with SARIMAX
   coefficients.

Simulation loop
---------------
For each ticker:

1) Fit SARIMAX candidates on historical data; learn ensemble/stacking weights.

2) Estimate residual diagnostics; set Student-t df; fit the ticker-level HMM.

3) Estimate state-conditional jump probabilities and GPD tails (or unconditional fallback).

4) For each Monte-Carlo replication:

   a) Obtain a macro delta path from the country-level simulator (or zeros).

   b) Simulate HMM states; draw jumps and magnitudes; build future exogenous X_f.

   c) Forecast SARIMAX ensemble; sample blended noise; apply regime scales; add jumps.

   d) Accumulate returns and exponentiate to prices, clipped to [lb, ub] at horizon.

Outputs
-------
Per ticker summary:
- low (5th percentile of final price),
- avg (median p50),
- high (95th percentile),
- returns (mean simple return of final/initial − 1),
- se (Monte-Carlo s.d. of returns).
Results are exported to an Excel workbook.

Why these modelling choices
---------------------------
- **ECVAR/BVAR/VAR mix:** hedges structural uncertainty (presence of cointegration,
  lag decay, cross-lag effects) and stabilises long-horizon macro dynamics.

- **Fourier seasonality:** parsimonious periodicity without seasonal ARIMA overhead.

- **AICc and stacking:** reduce small-sample overfitting and target decision-relevant
  H-step aggregates.

- **SV + MS:** replicate volatility clustering and regime shifts observed in financial
  series beyond homoskedastic innovations.

- **POT-GPD jumps:** provide asymptotically justified tail modelling for threshold
  exceedances and capture sign asymmetry.

- **Bootstrap blending:** accounts for model misspecification and residual dependence.

Assumptions and limitations
---------------------------
- Exogeneity: macro deltas are treated as exogenous to asset returns.

- Release-lag discipline: macro regressors are lagged by default to reduce
  contemporaneous look-ahead risk in forecasting mode.

- Stationarity: macro differences ΔX_t are assumed stationary; levels may be I(1).

- Gaussian base errors: SARIMAX innovations are conditionally normal; heavy tails are
  introduced via t-scales and jumps rather than Student-t innovations within SARIMAX.

- Data sufficiency: evidence tuning and HMM/GPD fits require non-trivial sample sizes;
  fallbacks are used when data are scarce.

- Column order fidelity: exogenous forecast design must match training order exactly.

Computational considerations
----------------------------
- Extensive caching of SARIMAX fits and Fourier blocks reduces repeated cost across folds.

- Antithetic Gaussian innovations halve Monte-Carlo variance for macro simulations.

- Parallel processing (Joblib) is used at the ticker level; logging is deduplicated across processes.

Optional extensions (not implemented here)
------------------------------------------
A gated recurrent unit (GRU) could replace or augment SARIMAX for the y_t dynamics:
  
   z_t = σ(W_z x_t + U_z h_{t−1} + b_z)        (update gate)
  
   r_t = σ(W_r x_t + U_r h_{t−1} + b_r)        (reset gate)
  
   h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t−1}) + b_h)
  
   h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t,

with σ the logistic function. GRUs mitigate vanishing gradients and can summarise long
histories with fewer parameters than LSTMs, providing a complementary non-linear
sequence model whose predictive distributions could be ensembled with SARIMAX.

Dependencies and data
---------------------
- `statsmodels` for SARIMAX, VAR, and Johansen tests;
- `scikit-learn` for scaling;
- `scipy` for GPD;
- `pandas`/`numpy` for data handling;
- An external `MacroData` provider and `export_results` utility.

This module is intended for research and production-adjacent forecasting where
transparent statistical structure, scenario-based exogenous drivers, and explicit
tail/volatility modelling are required.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pandas.tseries.frequencies import to_offset
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import OAS
from joblib import Parallel, delayed

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from scipy.stats import genpareto, johnsonsu, skew, kurtosis, norm, t as student_t
from scipy.special import logsumexp
import multiprocessing
import time
import warnings
import hashlib
import json
import os
import pickle
import tempfile

from data_processing.macro_data import MacroData
import config

from export_forecast import export_results

BASE_REGRESSORS: List[str] = ["Interest", "Cpi", "Gdp", "Unemp"]

lenBR = len(BASE_REGRESSORS)

STACK_ORDERS: List[Tuple[int,int,int]] = [(1, 0, 0), (0, 0, 1), (1, 0, 1)]

RIDGE_EPS = 1e-6  

EXOG_LAGS: tuple[int, ...] = (1, 2, 3)  

FOURIER_K: int = 2

USE_SV_MACRO: bool = True       

USE_MS_VOL: bool = True          

USE_RANK_UNCERTAINTY: bool = True  

MN_TUNE_METHOD: str = "evidence"   

SV_MIN_VAR = 1e-6  

MN_TUNE_L1_GRID = (0.1, 0.2, 0.3)     

MN_TUNE_L2_GRID = (0.3, 0.5, 0.7)     

MN_TUNE_L3_GRID = (0.5, 1.0, 1.5)  

MN_TUNE_L4_GRID = (50.0, 100.0, 200.0)  

MN_TUNE_JITTER = 1e-9               

two_pi = 2.0 * np.pi

_MAX_CACHE = 320

FORECAST_WEEKS: int = 52

N_SIMS: int = 2000

JUMP_Q: float = 0.97

N_JOBS: int = -1

CV_SPLITS: int = 3

RESID_BLOCK: int = 4 

T_DOF: int = 5             

INNOV_DIST_MODE: str = "johnson_su"

HM_MIN_OBS: int = 120

HM_CLIP_SKEW: float = 2.5

HM_CLIP_EXCESS_KURT: float = 15.0

COPULA_SHRINK: float = 0.05

START_AR_CLIP: float = 0.6

START_MA_CLIP: float = 0.4

U_EPS: float = 1e-10

BVAR_P: int = 2          

MN_LAMBDA1: float = 0.2 

MN_LAMBDA2: float = 0.5

MN_LAMBDA3: float = 1.0 

MN_LAMBDA4: float = 100.0

NIW_NU0: int = 5

NIW_S0_SCALE: float = 0.1  

RNG_SEED: int = 42

rng_global = np.random.default_rng(RNG_SEED)

AUTO_TUNE_HYPERPARAMS: bool = True

RUNTIME_PROFILE: str = "CUSTOM"

SPEED_FIRST_MODE: bool = False

SKIP_BACKTEST_ON_PROD_RUN: bool = True

USE_STATIC_MODEL_COV: bool = True

MC_DIAG_EVERY_BATCHES: int = 3

MACRO_CACHE_FILE: Path = Path(__file__).with_name("sarimax20_macro_cache.pkl")

TARGET_COUNTRIES_ONLY: bool = True

SUPPRESS_EXPECTED_CONVERGENCE_WARNINGS: bool = True

TARGET_Q_CI_HALF_WIDTH: float = 0.006 

TARGET_COV_REL_ERR: float = 0.02

LOW_MEMORY_MODE: bool = False

MAX_TUNE_SECONDS_PER_TICKER: float = 180.0

MAX_FITS_PER_TICKER: int = 1000

FULL_RERUN: bool = True

CACHE_FILE: Path = Path(__file__).with_name("sarimax20_cache.pkl")

CACHE_SCHEMA_VERSION: int = 1

_FOURIER_CACHE: Dict[Tuple[int, int, int, int], np.ndarray] = {}

_T_SCALES_CACHE = {}

_fit_memo_np: Dict[tuple, Ensemble] = {}         
           
_fit_by_order_cache_np: Dict[tuple, object] = {}               

_MODE_COMPARISON_CACHE: Dict[tuple, Dict[str, float]] = {}

_COMPILED_EXOG_LAYOUT_CACHE: Dict[tuple, Dict[str, Any]] = {}

_FOLD_DESIGN_CACHE: Dict[tuple, Any] = {}

_HP_MODE_COMPARISON_CACHE: Dict[tuple, Dict[str, float]] = {}


_RUNTIME_PRESETS: Dict[str, Dict[str, float]] = {
    "FAST": {
        "cv_draws_tune": 80,
        "cv_draws_backtest": 120,
        "max_order": 2,
        "max_cv_splits": 3,
        "fit_maxiter_a": 350,
        "fit_maxiter_b": 700,
        "fit_maxiter_c": 120,
        "fit_maxiter_final_a": 900,
        "fit_maxiter_final_b": 1600,
        "fit_maxiter_final_c": 220,
        "cov_reps_min": 80,
        "cov_reps_max": 320,
        "cov_batch": 60,
        "cov_target_rel_err": 0.07,
        "n_sims_min": 250,
        "n_sims_max": 900,
        "n_sims_batch": 80,
        "q_ci_half_width": 0.015,
        "macro_weight_eps": 0.03,
        "macro_tau_floor": 0.60,
        "ljungbox_min_p": 0.0005,
        "max_stack_keep": 3,
        "fourier_k_cap": 3,
        "mc_diag_every_batches": 3,
        "opt_time_floor": 20.0,
        "opt_max_fits_floor": 80,
    },
    "BALANCED": {
        "cv_draws_tune": 120,
        "cv_draws_backtest": 240,
        "max_order": 3,
        "max_cv_splits": 5,
        "fit_maxiter_a": 700,
        "fit_maxiter_b": 1400,
        "fit_maxiter_c": 220,
        "fit_maxiter_final_a": 1200,
        "fit_maxiter_final_b": 2200,
        "fit_maxiter_final_c": 320,
        "cov_reps_min": 120,
        "cov_reps_max": 700,
        "cov_batch": 80,
        "cov_target_rel_err": 0.05,
        "n_sims_min": 350,
        "n_sims_max": 1500,
        "n_sims_batch": 120,
        "q_ci_half_width": 0.012,
        "macro_weight_eps": 0.02,
        "macro_tau_floor": 0.50,
        "ljungbox_min_p": 0.001,
        "max_stack_keep": 3,
        "fourier_k_cap": 3,
        "mc_diag_every_batches": 1,
        "opt_time_floor": 40.0,
        "opt_max_fits_floor": 180,
    },
    "THOROUGH": {
        "cv_draws_tune": 220,
        "cv_draws_backtest": 420,
        "max_order": 4,
        "max_cv_splits": 6,
        "fit_maxiter_a": 1300,
        "fit_maxiter_b": 2600,
        "fit_maxiter_c": 450,
        "fit_maxiter_final_a": 2200,
        "fit_maxiter_final_b": 4200,
        "fit_maxiter_final_c": 650,
        "cov_reps_min": 220,
        "cov_reps_max": 1200,
        "cov_batch": 120,
        "cov_target_rel_err": 0.03,
        "n_sims_min": 600,
        "n_sims_max": 2600,
        "n_sims_batch": 200,
        "q_ci_half_width": 0.009,
        "macro_weight_eps": 0.015,
        "macro_tau_floor": 0.45,
        "ljungbox_min_p": 0.002,
        "max_stack_keep": 5,
        "fourier_k_cap": 5,
        "mc_diag_every_batches": 1,
        "opt_time_floor": 50.0,
        "opt_max_fits_floor": 260,
    },
    "RESEARCH": {
        "cv_draws_tune": 360,
        "cv_draws_backtest": 720,
        "max_order": 5,
        "max_cv_splits": 7,
        "fit_maxiter_a": 2200,
        "fit_maxiter_b": 5000,
        "fit_maxiter_c": 900,
        "fit_maxiter_final_a": 3200,
        "fit_maxiter_final_b": 7000,
        "fit_maxiter_final_c": 1200,
        "cov_reps_min": 320,
        "cov_reps_max": 2200,
        "cov_batch": 160,
        "cov_target_rel_err": 0.02,
        "n_sims_min": 1000,
        "n_sims_max": 4200,
        "n_sims_batch": 260,
        "q_ci_half_width": 0.006,
        "macro_weight_eps": 0.01,
        "macro_tau_floor": 0.40,
        "ljungbox_min_p": 0.003,
        "max_stack_keep": 5,
        "fourier_k_cap": 5,
        "mc_diag_every_batches": 1,
        "opt_time_floor": 80.0,
        "opt_max_fits_floor": 420,
    },
    "CUSTOM": {
        "cv_draws_tune": 120,
        "cv_draws_backtest": 240,
        "max_order": 4,
        "max_cv_splits": 3,
        "fit_maxiter_a": 700,
        "fit_maxiter_b": 1400,
        "fit_maxiter_c": 220,
        "fit_maxiter_final_a": 1200,
        "fit_maxiter_final_b": 2200,
        "fit_maxiter_final_c": 320,
        "cov_reps_min": 120,
        "cov_reps_max": 700,
        "cov_batch": 80,
        "cov_target_rel_err": 0.05,
        "n_sims_min": 350,
        "n_sims_max": 1500,
        "n_sims_batch": 120,
        "q_ci_half_width": 0.012,
        "macro_weight_eps": 0.02,
        "macro_tau_floor": 0.50,
        "ljungbox_min_p": 0.001,
        "max_stack_keep": 3,
        "fourier_k_cap": 3,
        "mc_diag_every_batches": 1,
        "opt_time_floor": 40.0,
        "opt_max_fits_floor": 180,
    },
}


def _effective_runtime_profile() -> str:
    """
    Resolve the active runtime preset name after applying hard overrides and validation.

    Selection rule
    --------------
    Let `p0 = upper(RUNTIME_PROFILE)`.

    The effective profile `p` is computed as:

    1) If `SPEED_FIRST_MODE` is true, set `p = "FAST"` irrespective of `p0`.
  
    2) Otherwise set `p = p0`.
  
    3) If `p` is not a key in `_RUNTIME_PRESETS`, set `p = "BALANCED"`.

    In compact form:

        p = "FAST"                       if SPEED_FIRST_MODE
        p = p0                           if not SPEED_FIRST_MODE and p0 in presets
        p = "BALANCED"                   otherwise

    Returns
    -------
    str
        A validated preset label present in `_RUNTIME_PRESETS`.

    Why this process
    ----------------
    Runtime tuning is centralised into named presets to guarantee consistent behaviour
    across hyperparameter search, covariance simulation, and Monte-Carlo budgets.
    A deterministic fallback prevents accidental misconfiguration from propagating to
    expensive model-fitting stages.

    Advantages
    ----------
    - Provides deterministic execution policy in both research and production modes.
  
    - Protects against invalid configuration strings without raising user-facing errors.
  
    - Allows a single global speed override (`SPEED_FIRST_MODE`) for emergency runs.
  
    """

    prof = str(RUNTIME_PROFILE).upper()

    if SPEED_FIRST_MODE:
      
        prof = "FAST"

    if prof not in _RUNTIME_PRESETS:
      
        prof = "BALANCED"

    return prof


def _runtime_settings() -> Dict[str, float]:
    """
    Return the numeric runtime settings for the active preset.

    Mathematical mapping
    --------------------
    Let `p = _effective_runtime_profile()`. The function returns:

        settings = _RUNTIME_PRESETS[p].

    The mapping is a pure dictionary lookup; no interpolation or stochastic adjustment
    is applied.

    Returns
    -------
    dict[str, float]
        Preset-specific controls such as optimisation iteration caps, simulation budgets,
        and accuracy targets.

    Modelling relevance
    -------------------
    Although this is a utility function, it governs model-selection pressure and Monte-Carlo
    precision indirectly by controlling draw counts, fit budgets, and convergence thresholds.

    Advantages
    ----------
    - Single source of truth for compute-versus-accuracy trade-offs.
  
    - Reproducible behaviour because every downstream stage reads the same preset.
  
    - Simple retrieval cost, enabling frequent use inside tight loops.
  
    """
    
    prof = _effective_runtime_profile()
    
    return _RUNTIME_PRESETS[prof]


@dataclass(frozen = True)
class OptimizationProfile:

    enable_two_stage_search: bool

    max_tune_seconds: float

    max_fits: int

    reuse_fold_metrics: bool


def _optimization_profile() -> OptimizationProfile:
    """
    Build the optimisation-budget profile used by ticker-level hyperparameter search.

    Construction
    ------------
    The function combines preset floors with global hard limits:

    - `floor_time = runtime_settings["opt_time_floor"]`
 
    - `floor_fits = runtime_settings["opt_max_fits_floor"]`

    For profile `"FAST"`:

        max_tune_seconds = min(floor_time, MAX_TUNE_SECONDS_PER_TICKER)
        max_fits         = min(floor_fits, MAX_FITS_PER_TICKER)

    For all other profiles:

        max_tune_seconds = max(floor_time, MAX_TUNE_SECONDS_PER_TICKER)
        max_fits         = max(floor_fits, MAX_FITS_PER_TICKER)

    The two-stage search flag and fold-metric reuse are enabled in both branches.

    Returns
    -------
    OptimizationProfile
        Immutable configuration containing time budget, fit budget, and search toggles.

    Why this process
    ----------------
    Hyperparameter tuning is a constrained optimisation over a noisy objective. The
    pipeline therefore uses explicit budget control to avoid pathological runs while
    preserving exploratory depth in non-fast profiles.

    Advantages
    ----------
    - Enforces predictable wall-time envelopes for each ticker.
  
    - Prevents silent under-search in thorough modes and over-search in fast mode.
  
    - Supplies stable policy parameters for reproducible comparative experiments.
  
    """

    prof = _effective_runtime_profile()

    rs = _runtime_settings()

    floor_time = float(rs.get("opt_time_floor", MAX_TUNE_SECONDS_PER_TICKER))

    floor_fits = int(rs.get("opt_max_fits_floor", MAX_FITS_PER_TICKER))

    if prof == "FAST":

        return OptimizationProfile(
            enable_two_stage_search = True,
            max_tune_seconds = min(floor_time, MAX_TUNE_SECONDS_PER_TICKER),
            max_fits = min(floor_fits, MAX_FITS_PER_TICKER),
            reuse_fold_metrics = True,
        )

    return OptimizationProfile(
        enable_two_stage_search = True,
        max_tune_seconds = max(floor_time, MAX_TUNE_SECONDS_PER_TICKER),
        max_fits = max(floor_fits, MAX_FITS_PER_TICKER),
        reuse_fold_metrics = True,
    )


class _DedupFilter(logging.Filter):
  
    def __init__(
        self, 
        window_sec: float = 5.0
    ):
       
        super().__init__()
       
        self._last_t: dict[tuple[int, str], float] = {}
       
        self._window = window_sec

    
    def filter(
        self, 
        record: logging.LogRecord
    ) -> bool:
        """
        Filter log records to suppress near-duplicate messages within a sliding time window.

        This filter keeps at most one instance of an identically formatted log message
        (per log level) over `window_sec` seconds. Deduplication is keyed by the fully
        rendered `record.getMessage()`, not by the format string, ensuring that messages
        with distinct interpolated values are treated as distinct.

        Parameters
        ----------
        record : logging.LogRecord
            The logging record proposed for emission.

        Returns
        -------
        bool
            True if the record should be emitted; False if it is suppressed as a recent duplicate.

        Rationale
        ---------
        Parallel and iterative estimation steps (e.g., repeated SARIMAX fits or cross-validation
        folds) can trigger bursts of identical informational messages. Deduplicating reduces
        console noise and improves signal without altering program semantics.
        """

        msg = record.getMessage()

        key = (record.levelno, msg)

        now = time.time()

        last = self._last_t.get(key, 0.0)

        if now - last < self._window:

            return False

        self._last_t[key] = now

        return True


def configure_logger() -> logging.Logger:
    """
    Create a process-aware logger with sensible defaults.

    - In worker processes (name ≠ "MainProcess"), disable propagation to avoid
    duplicate handlers; attach a simple stream handler at INFO level.
    - In the main process, attach a stream handler with a de-duplication filter
    and set third-party libraries (statsmodels, joblib) to quieter levels.

    Returns
    -------
    logging.Logger
        Configured logger for this module.

    Advantages
    ----------
    Process-aware configuration prevents multiprocess duplication; a lightweight
    console formatter preserves readability; lowering third-party verbosity reduces
    log spam during large model grids.
    """
       
    logger = logging.getLogger(__name__)
   
    if multiprocessing.current_process().name != "MainProcess":
   
        logger.propagate = False
   
        if not logger.handlers:                   
   
            ch = logging.StreamHandler()
   
            ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
   
            logger.addHandler(ch)
   
        logger.setLevel(logging.INFO)              
   
        return logger

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        ch = logging.StreamHandler()

        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        ch.addFilter(_DedupFilter(window_sec=4.0))

        logger.addHandler(ch)

    logging.getLogger("statsmodels").setLevel(logging.ERROR)

    logging.getLogger("joblib").setLevel(logging.WARNING)

    return logger


logger = configure_logger()


def _hash_file_sha256(
    path: Path
) -> str:
    """
    Compute SHA256 for a file.
    """

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(lambda: f.read(1 << 20), b""):

            h.update(chunk)

    return h.hexdigest()


def _compute_run_signature(
    tickers: List[str],
    forecast_period: int,
    close: pd.DataFrame,
    raw_macro: pd.DataFrame,
    run_cfg: Dict[str, Any],
    dep_hashes_override: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a deterministic signature for cache validity.
    """

    close_last_ts = str(pd.to_datetime(close.index.max()).isoformat()) if len(close.index) else ""

    macro_last_ts = str(pd.to_datetime(raw_macro["ds"].max()).isoformat()) if len(raw_macro) else ""

    close_anchor = {
        tk: float(close[tk].dropna().iloc[-1]) if tk in close.columns and close[tk].dropna().size else np.nan
        for tk in tickers
    }

    if dep_hashes_override is None:
       
        dep_files = [
            Path(__file__),
            Path(__file__).with_name("config.py"),
            Path(__file__).with_name("macro_data3.py"),
        ]

        dep_hashes: Dict[str, str] = {}

        for p in dep_files:
  
            try:
  
                dep_hashes[p.name] = _hash_file_sha256(
                    path = p
                )
  
            except Exception:
  
                dep_hashes[p.name] = ""
    else:
        dep_hashes = dict(dep_hashes_override)

    script_name = Path(__file__).name
 
    script_hash = dep_hashes.get(script_name, "")
 
    if not script_hash:
 
        script_hash = _hash_file_sha256(
            path = Path(__file__)
        )

    sig_obj = {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "script_sha256": script_hash,
        "dependency_hashes": dep_hashes,
        "tickers": list(tickers),
        "forecast_period": int(forecast_period),
        "close_last_ts": close_last_ts,
        "macro_last_ts": macro_last_ts,
        "close_anchor": close_anchor,
        "run_cfg": run_cfg,
    }

    packed = json.dumps(sig_obj, sort_keys=True, separators=(",", ":"), default=str)

    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _load_run_cache(
    path: Path
) -> Optional[RunCachePayload]:
    """
    Load cache payload from pickle.
    """

    if not path.exists():

        return None

    try:

        with path.open("rb") as f:

            raw = pickle.load(f)

        if isinstance(raw, RunCachePayload):

            return raw

        if not isinstance(raw, dict):

            return None

        return RunCachePayload(
            schema_version = int(raw.get("schema_version", -1)),
            signature = str(raw.get("signature", "")),
            created_utc = str(raw.get("created_utc", "")),
            results_by_ticker = dict(raw.get("results_by_ticker", {})),
            df_out_records = list(raw.get("df_out_records", [])),
            fitted_hyperparams_by_ticker = dict(raw.get("fitted_hyperparams_by_ticker", {})),
            backtest_summary_by_ticker = dict(raw.get("backtest_summary_by_ticker", {})),
            ticker_diagnostics_by_ticker = dict(raw.get("ticker_diagnostics_by_ticker", {})),
            warning_summary = dict(raw.get("warning_summary", {})),
            search_trace_by_ticker = dict(raw.get("search_trace_by_ticker", {})),
            run_meta = dict(raw.get("run_meta", {})),
        )

    except Exception as e:

        logger.warning("Cache load failed (%s). Recomputing.", e)

        return None


def _save_run_cache_atomic(
    path: Path,
    payload: RunCachePayload,
) -> None:
    """
    Write cache atomically to avoid partial files.
    """

    path.parent.mkdir(parents = True, exist_ok = True)

    fd, tmp_name = tempfile.mkstemp(prefix = path.name + ".", suffix=".tmp", dir=str(path.parent))

    try:

        with os.fdopen(fd, "wb") as f:

            pickle.dump(asdict(payload), f, protocol = pickle.HIGHEST_PROTOCOL)

            f.flush()

            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    finally:

        if os.path.exists(tmp_name):

            try:

                os.remove(tmp_name)

            except OSError:

                pass


def _is_cache_usable(
    payload: Optional[RunCachePayload],
    signature: str,
) -> bool:
    """
    Cache is usable only when schema and signature both match.
    """

    if payload is None:

        return False

    return (
        int(payload.schema_version) == int(CACHE_SCHEMA_VERSION)
        and payload.signature == signature
    )


def _dict_to_fitted_hyperparams(
    d: Dict[str, Any]
) -> Optional[FittedHyperparams]:
    """
    Convert a raw dictionary payload into a validated `FittedHyperparams` instance.

    Process
    -------
    The function performs structural normalisation before dataclass construction:

    - `stack_orders` entries are coerced to tuples so each order is hashable and stable.
  
    - `exog_lags` is coerced to a tuple to preserve deterministic cache-key behaviour.
  
    - `mn_lambdas` is coerced to a 4-tuple, falling back to module defaults when missing.

    Let `d_raw` be the input dictionary and `T(.)` denote type coercion:

        d_norm["stack_orders"] = [T_tuple(x) for x in d_raw["stack_orders"]]
  
        d_norm["exog_lags"]    = T_tuple(d_raw["exog_lags"])
  
        d_norm["mn_lambdas"]   = T_tuple4(d_raw["mn_lambdas"], defaults)

    Then:

        hp = FittedHyperparams(**d_norm).

    Parameters
    ----------
    d : dict[str, Any]
        Deserialised cache object representing fitted hyperparameters.

    Returns
    -------
    FittedHyperparams | None
        Parsed dataclass on success, otherwise `None` if coercion or construction fails.

    Why this process
    ----------------
    Pickled or JSON-like cache payloads may lose tuple fidelity and can contain partially
    stale fields across schema revisions. Defensive coercion preserves compatibility without
    interrupting pipeline execution.

    Advantages
    ----------
    - Robust partial-cache reuse after non-breaking schema drift.
   
    - Preserves deterministic typing required by downstream hashing and comparisons.
   
    - Fails closed (`None`) rather than propagating malformed objects.
   
    """

    try:

        dd = dict(d)

        dd["stack_orders"] = [tuple(x) for x in dd.get("stack_orders", [])]

        dd["exog_lags"] = tuple(dd.get("exog_lags", ()))

        dd["mn_lambdas"] = tuple(dd.get("mn_lambdas", (MN_LAMBDA1, MN_LAMBDA2, MN_LAMBDA3, MN_LAMBDA4)))

        return FittedHyperparams(**dd)

    except Exception:

        return None


def _dict_to_backtest_summary(
    d: Dict[str, Any]
) -> Optional[BacktestSummary]:
    """
    Convert a dictionary payload into a `BacktestSummary` dataclass safely.

    Method
    ------
    The function applies a direct dataclass constructor:

        summary = BacktestSummary(**dict(d)).

    Any exception from missing fields, incompatible types, or malformed input is trapped,
    and `None` is returned.

    Parameters
    ----------
    d : dict[str, Any]
        Raw cache object expected to contain backtest metrics.

    Returns
    -------
    BacktestSummary | None
        Valid summary object on success, otherwise `None`.

    Why this process
    ----------------
    Backtest summaries are optional during partial cache recovery. Hard failure on a single
    malformed ticker record would unnecessarily force complete recomputation.

    Advantages
    ----------
    - Graceful degradation when historical cache payloads are incomplete.
   
    - Maintains strict typing for consumers that rely on dataclass fields.
   
    - Simplifies upstream control flow by using `None` as an explicit invalid marker.
   
    """

    try:

        return BacktestSummary(**dict(d))

    except Exception:

        return None


def _maybe_trim_cache(
    d: dict
):
    """
    Bound the size of an in-memory cache with incremental eviction.

    Parameters
    ----------
    d : dict
        Cache dictionary to check and potentially clear.

    Notes
    -----
    Caches for SARIMAX fits and Fourier blocks accelerate repeated calls but can grow
    unbounded across instruments and folds. Incremental eviction avoids memory blow-ups
    without discarding all hot entries at once.
    """

    if len(d) > _MAX_CACHE:
 
        trim_to = max(1, int(_MAX_CACHE * 0.90))
 
        while len(d) > trim_to:
 
            try:
 
                d.pop(next(iter(d)))
 
            except Exception:
 
                break


def _cache_get_touch(
    d: Dict[Any, Any],
    key: Any,
) -> Any:
    """
    LRU-like access: on hit, move key to dict end.
    """

    if key not in d:
 
        return None

    try:
 
        val = d.pop(key)
 
        d[key] = val
 
        return val
 
    except Exception:
 
        return d.get(key)


def _load_macro_cache(
    path: Path,
) -> Dict[str, Any]:
    """
    Load the macro-scenario cache payload from disk with defensive type checks.

    Behaviour
    ---------
    1) If `path` does not exist, return an empty dictionary.
  
    2) Attempt to unpickle the file.
  
    3) Return the object only when it is a dictionary; otherwise return an empty dictionary.
  
    4) On any exception, log a warning and return an empty dictionary.

    Parameters
    ----------
    path : pathlib.Path
        Location of the macro cache pickle file.

    Returns
    -------
    dict[str, Any]
        Cache payload keyed by deterministic macro scenario signatures.

    Why this process
    ----------------
    Macro scenario generation is computationally expensive; cache reuse materially reduces
    runtime. At the same time, cache files can become stale or corrupted, so the loader
    must be fail-safe.

    Advantages
    ----------
    - Prevents pipeline interruption due to cache corruption.
 
    - Keeps stale-cache handling local rather than spreading try/except logic.
 
    - Supports deterministic recomputation path when cache validity is uncertain.
 
    """
 
    if not path.exists():
 
        return {}
 
    try:
 
        with path.open("rb") as f:
 
            raw = pickle.load(f)
 
        if isinstance(raw, dict):
 
            return raw
 
        return {}
 
    except Exception as e:
 
        logger.warning("Macro cache load failed (%s). Recomputing macro scenarios.", e)
 
        return {}


def _save_macro_cache_atomic(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    """
    Persist macro cache data atomically to avoid partial-file corruption.

    Atomic write protocol
    ---------------------
    The function implements a write-then-rename sequence:

    1) Create parent directory if needed.
   
    2) Create a temporary file in the same directory.
   
    3) Serialize payload via pickle (highest protocol).
   
    4) Flush user-space buffers and force an `fsync`.
   
    5) Replace the target path using `os.replace`, which is atomic on the same filesystem.
   
    6) Remove the temporary file in a guarded `finally` block if it still exists.

    Parameters
    ----------
    path : pathlib.Path
        Final cache-file location.
    payload : dict[str, Any]
        Macro cache object to serialize.

    Why this process
    ----------------
    Long-running forecasting jobs may be interrupted mid-write. Non-atomic writes can leave
    truncated pickles, causing downstream deserialisation failures.

    Advantages
    ----------
    - Strong crash consistency for cache persistence.
  
    - Avoids readers observing half-written files.
  
    - Keeps cache invalidation minimal by preserving either old or new complete content.
  
    """
 
    path.parent.mkdir(parents = True, exist_ok = True)

    fd, tmp_name = tempfile.mkstemp(prefix = path.name + ".", suffix = ".tmp", dir = str(path.parent))
 
    try:
 
        with os.fdopen(fd, "wb") as f:
 
            pickle.dump(payload, f, protocol = pickle.HIGHEST_PROTOCOL)
 
            f.flush()
 
            os.fsync(f.fileno())
 
        os.replace(tmp_name, path)
 
    finally:
 
        if os.path.exists(tmp_name):
 
            try:
 
                os.remove(tmp_name)
 
            except OSError:
 
                pass


def _macro_cache_key(
    country: str,
    horizon: int,
    n_sims: int,
    macro_last_ts: str,
    macro_hp: Dict[str, Any],
) -> str:
    """
    Construct a deterministic SHA-256 key for macro-scenario cache lookup.

    Definition
    ----------
    Let `obj` be the canonical request descriptor:

        obj = {
            "country": country,
            "horizon": horizon,
            "n_sims": n_sims,
            "macro_last_ts": macro_last_ts,
            "macro_hp": macro_hp
        }.

    The key is:

        key = SHA256(JSON(obj, sort_keys=True, compact_separators, default=str)).

    Parameters
    ----------
    country : str
        Country identifier for the macro model.
    horizon : int
        Forecast horizon in weekly steps.
    n_sims : int
        Number of Monte-Carlo macro paths.
    macro_last_ts : str
        Timestamp anchor for available macro history.
    macro_hp : dict[str, Any]
        Fitted macro hyperparameters affecting scenario dynamics.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest suitable as a dictionary key.

    Why this process
    ----------------
    Macro simulations depend jointly on country data, horizon, simulation count, and
    hyperparameter choices. Any change in these inputs should invalidate prior scenarios.

    Advantages
    ----------
    - Collision-resistant cache identity for practical workloads.
   
    - Stable across processes due to canonical JSON serialisation.
   
    - Compact fixed-length key, efficient for in-memory and pickled dictionaries.
   
    """
  
    obj = {
        "country": str(country),
        "horizon": int(horizon),
        "n_sims": int(n_sims),
        "macro_last_ts": str(macro_last_ts),
        "macro_hp": macro_hp,
    }
  
    packed = json.dumps(obj, sort_keys = True, separators = (",", ":"), default = str)
  
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _mode_comparison_cache_key(
    df: pd.DataFrame,
    regressors: List[str],
    horizon: int,
    n_splits: int,
    draws: int,
    seed: int,
    exog_lags: tuple[int, ...],
    fourier_k: int,
    orders: Optional[List[Tuple[int, int, int]]],
    hm_min_obs: int,
    hm_clip_skew: float,
    hm_clip_excess_kurt: float,
) -> tuple:
    """
    Build a stable in-memory key for rolling-origin mode-comparison reuse.
    """

    h = hashlib.blake2b(digest_size=16)

    if len(df.index):
      
        idx_vec = np.asarray(
            [int(df.index[0].value), int(df.index[-1].value), int(len(df.index))],
            dtype = np.int64,
        )
     
        h.update(idx_vec.tobytes())

    y_arr = np.asarray(df.get("y", pd.Series(dtype=float)).values, float)
  
    if y_arr.size:
  
        y_tail = np.nan_to_num(y_arr[-min(512, y_arr.size):], nan = 0.0, posinf = 0.0, neginf = 0.0)
  
        h.update(np.ascontiguousarray(y_tail, dtype=np.float64).tobytes())

    if regressors:
   
        try:
   
            x_tail = np.asarray(df[regressors].tail(min(128, len(df))).values, float)
   
            if x_tail.size:
   
                x_tail = np.nan_to_num(x_tail, nan = 0.0, posinf = 0.0, neginf = 0.0)
   
                h.update(np.ascontiguousarray(x_tail, dtype = np.float64).tobytes())
   
        except Exception:
   
            pass

    orders_key = tuple(tuple(o) for o in (orders if orders is not None else list(STACK_ORDERS)))

    return (
        int(horizon),
        int(n_splits),
        int(draws),
        int(seed),
        tuple(int(x) for x in exog_lags),
        int(fourier_k),
        orders_key,
        int(hm_min_obs),
        float(hm_clip_skew),
        float(hm_clip_excess_kurt),
        h.hexdigest(),
    )


def _hp_mode_comparison_cache_key(
    df: pd.DataFrame,
    horizon: int,
    hp: "FittedHyperparams",
) -> tuple:
    """
    Build a draw/seed-agnostic key for tuned-hyperparameter mode comparison reuse.
    """

    h = hashlib.blake2b(digest_size = 16)

    if len(df.index):
   
        idx_vec = np.asarray(
            [int(df.index[0].value), int(df.index[-1].value), int(len(df.index))],
            dtype = np.int64,
        )
        h.update(idx_vec.tobytes())

    y_arr = np.asarray(df.get("y", pd.Series(dtype = float)).values, float)
 
    if y_arr.size:
 
        y_tail = np.nan_to_num(y_arr[-min(512, y_arr.size):], nan = 0.0, posinf = 0.0, neginf = 0.0)
 
        h.update(np.ascontiguousarray(y_tail, dtype=np.float64).tobytes())

    if BASE_REGRESSORS:
 
        try:
 
            x_tail = np.asarray(df[BASE_REGRESSORS].tail(min(128, len(df))).values, float)
 
            if x_tail.size:
 
                x_tail = np.nan_to_num(x_tail, nan = 0.0, posinf = 0.0, neginf = 0.0)
 
                h.update(np.ascontiguousarray(x_tail, dtype=np.float64).tobytes())
                
        except Exception:
            
            pass

    return (
        int(horizon),
        int(hp.cv_splits),
        tuple(int(x) for x in hp.exog_lags),
        int(hp.fourier_k),
        tuple(tuple(o) for o in hp.stack_orders),
        int(hp.hm_min_obs),
        float(hp.hm_clip_skew),
        float(hp.hm_clip_excess_kurt),
        h.hexdigest(),
    )


def _macro_stationary_deltas(
    df_levels: pd.DataFrame,
    cointegration: bool = False
) -> pd.DataFrame:
    """
    Transform macroeconomic levels to stationary differences (and, optionally, a levels
    copy suitable for cointegration testing).

    Let the set of base regressors be:
        BASE_REGRESSORS = {Interest, Cpi, Gdp, Unemp}.

    Transformation
    --------------
    For c in {Cpi, Gdp}: use logarithmic differences:
        Δ log c_t = log(c_t) − log(c_{t−1}).
    For c in {Interest, Unemp}: use level differences:
        Δ c_t = c_t − c_{t−1}.

    If `cointegration=True`, also return a levels DataFrame (possibly logged for prices)
    to permit Johansen cointegration analysis on (potentially) I(1) series.

    Parameters
    ----------
    df_levels : pandas.DataFrame
        Weekly macro series in levels, indexed by a DatetimeIndex; must contain the
        base regressor columns.
    cointegration : bool, default False
        If True, return a tuple (deltas, levels_for_coint) where levels_for_coint are
        the transformed levels used in VECM (e.g., log levels for price-like series).

    Returns
    -------
    pandas.DataFrame | tuple[pandas.DataFrame, pandas.DataFrame]
        Stationary deltas; or (deltas, levels-for-cointegration) when `cointegration=True`.

    Why this transformation
    -----------------------
    Log-differences for expenditure/price-level variables approximate continuously
    compounded growth rates and stabilise variance; first differences for rates and
    unemployment remove unit roots without undue distortion. Stationarity is a prerequisite
    for VAR-type dynamics and for well-behaved residuals in downstream modelling.
    """
   
    df = pd.DataFrame(index = df_levels.index)
    
    if cointegration:
        
        df_coint = pd.DataFrame(index = df_levels.index)
   
        for c in BASE_REGRESSORS:
    
            if c.lower() in ("cpi", "gdp"):
                    
                s = df_levels[c].astype(float)
                    
                s = s.where(s > 0).ffill().bfill()
                
                ls = np.log(s)
                                
                df_coint[c] = ls
    
                df[c] = ls.diff()
    
            else:
    
                df[c] = df_levels[c].astype(float).diff()
                
                df_coint[c] = df_levels[c].astype(float)
        
        return df.dropna(), df_coint.dropna()
    
    else:
        
        for c in BASE_REGRESSORS:
    
            if c.lower() in ("cpi", "gdp"):
    
                s = df_levels[c].astype(float)
    
                s = s.where(s > 0).ffill().bfill()
    
                df[c] = np.log(s).diff()
    
            else:
    
                df[c] = df_levels[c].astype(float).diff()
   
        return df.dropna()


@dataclass
class ECVARModel:
  
    c: np.ndarray
  
    A: np.ndarray
  
    K: np.ndarray
  
    beta: np.ndarray
  
    Sigma: np.ndarray
  
    k: int

    def simulate(
        self,
        L0: np.ndarray,
        dX0: np.ndarray,
        steps: int,
        z: np.ndarray,
        scale_path: Optional[np.ndarray] = None,  
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate vector error-correction VAR(1) dynamics with Gaussian innovations and optional
        time-varying scale factors.

        Model
        -----
        Let L_t denote the (log) levels vector (k × 1) of macro variables and ΔX_t = L_t − L_{t − 1}
        the first differences. The ECVAR(1) dynamics used here are:

            ΔX_t = c + A ΔX_{t − 1} + K (β' L_{t − 1}) + ε_t,

        where:
       
        - c is a k × 1 intercept,
       
       
        - A is a k × k short-run dynamics matrix,
       
        - β is a k × r cointegration matrix (r cointegrating relations),
       
        - K is a k × r loading (adjustment) matrix,
       
        - ε_t ~ N(0, Σ) are Gaussian innovations.

        The levels then evolve by:
        
            L_t = L_{t−1} + ΔX_t.

        Optional heteroskedasticity is introduced by elementwise scaling:
       
            ε_t = chol(Σ) z_t ⊙ s_t,
       
        with z_t ~ N(0, I_k) and s_t supplied via `scale_path` (shape steps×k).

        Parameters
        ----------
        L0 : numpy.ndarray, shape (k,)
            Last observed levels (or log-levels).
        dX0 : numpy.ndarray, shape (k,)
            Last observed differences ΔX_{t−1}.
        steps : int
            Forecast horizon.
        z : numpy.ndarray, shape (steps, k)
            Base standard-normal innovations.
        scale_path : numpy.ndarray, optional, shape (steps, k)
            Non-negative scale multipliers per dimension and step (e.g., from stochastic
            volatility and/or Markov switching).

        Returns
        -------
        dX_path : numpy.ndarray, shape (steps, k)
            Simulated differences.
        L_path : numpy.ndarray, shape (steps, k)
            Simulated levels.

        Advantages of ECVAR
        -------------------
        - Error-correction restores long-run equilibrium relations through K β^T L_{t − 1}
        while allowing rich short-run dynamics in A ΔX_{t − 1}
      
        - Adjustment speeds (K) quantify how quickly disequilibria are corrected.
      
        - When cointegration is present, ECVAR yields better long-horizon behaviour than an
        unrestricted VAR in differences, which discards long-run information. This prevents
        drift from cointegrating manifolds and improves multi-step predictive realism for
        exogenous macro scenarios.
        """
     
        assert z.shape == (steps, self.k)
     
        if scale_path is not None:
     
            assert scale_path.shape == (steps, self.k)
     
        L = L0.copy()
     
        dX = dX0.copy()
     
        dX_path = np.zeros((steps, self.k))
     
        L_path = np.zeros((steps, self.k))
     
        Lchol = np.linalg.cholesky((self.Sigma + self.Sigma.T) / 2.0 + 1e-9 * np.eye(self.k))
     
        for t in range(steps):
     
            ect = self.beta.T @ L
     
            innov = Lchol @ z[t]
     
            if scale_path is not None:
     
                innov = innov * scale_path[t]         
     
            dx_next = self.c + self.A @ dX + self.K @ ect + innov
     
            L = L + dx_next
     
            dX = dx_next
     
            dX_path[t] = dx_next
     
            L_path[t] = L
     
        return dX_path, L_path


@dataclass
class VARModel:
   
    c: np.ndarray
   
    A: np.ndarray
   
    Sigma: np.ndarray
   
    k: int

    def simulate(
        self, 
        dX0: np.ndarray, 
        steps: int,
        z: np.ndarray,
        scale_path: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate paths from a VAR(1) in deltas with Gaussian shocks (fallback).

        Model
        -----
        ΔX_t = c + A ΔX_{t−1} + ε_t,  
        
        ε_t = chol(Σ) z_t ⊙ s_t,
        
        where:
       
        - c is k × 1, A is k × k, Σ is k × k positive definite,
       
        - z_t ~ N(0, I_k), and s_t are optional scale multipliers (steps × k).


        Parameters
        ----------
        dX0 : ndarray of shape (k,)
            Last observed ΔX.
        steps : int
            Number of steps to simulate.
        z : ndarray of shape (H, k)
            Standard normal innovations.
        scale_path : ndarray of shape (H, k), optional
            Elementwise multipliers to imprint volatility structure.

        Returns
        -------
        ndarray of shape (H, k)
            Simulated ΔX path.

        Why
        ---
        A stable VAR in differences captures short-run dynamics when cointegration evidence
        is weak or unavailable. It is robust, fast to estimate, and serves as a safety net
        when more structured candidates fail.
        """
      
        assert z.shape == (steps, self.k)
       
        if scale_path is not None:
       
            assert scale_path.shape == (steps, self.k)
       
        dX = dX0.copy()
       
        out = np.zeros((steps, self.k))
       
        Lchol = np.linalg.cholesky((self.Sigma + self.Sigma.T) / 2.0 + 1e-9 * np.eye(self.k))
       
        for t in range(steps):
       
            innov = Lchol @ z[t]
       
            if scale_path is not None:
       
                innov = innov * scale_path[t]
       
            dx_next = self.c + self.A @ dX + innov
       
            out[t] = dx_next
       
            dX = dx_next
       
        return out


@dataclass
class BVARPosterior:
  
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
        Draw a parameter sample (B, Σ) from the Minnesota-Normal-Inverse-Wishart posterior.

        Posterior structure
        -------------------
        Let the VAR(p) in differences be:

            ΔX_t = c + Σ_{ℓ=1..p} A_ℓ ΔX_{t−ℓ} + ε_t,  with ε_t ~ N(0, Σ).

        Stack coefficients by equation so that:
      
        - B is an (m × k) matrix with m = 1 + k p (constant then lag blocks),
      
        - For observation t, regressor x_t is (m × 1) and outcome y_t is (k × 1).

        Prior and posterior:
      
        - B | Σ ~ MatrixNormal(B_n, V_n, Σ),
      
        - Σ ~ InverseWishart(ν_n, S_n).

        Sampling scheme
        ---------------
        1) Draw Σ ~ IW(ν_n, S_n) using Bartlett decomposition.
       
        2) Draw B ~ MN(B_n, V_n, Σ) by B = B_n + L_V Z L_Σ', where:
       
        - L_V is chol(V_n), L_Σ is chol(Σ), Z ~ N(0, I_{m×k}).

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        B : numpy.ndarray, shape (m, k)
            Sampled stacked coefficient matrix.
        Sigma : numpy.ndarray, shape (k, k)
            Sampled covariance matrix.

        Advantages
        ----------
        - Bayesian shrinkage stabilises estimates in small samples and reduces overfitting.
       
        - The Minnesota prior encodes beliefs that own-lags dominate cross-lags and that
        higher-order lags decay, improving forecast accuracy under typical macro time-series
        properties.
        
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
        Sigma: np.ndarray,
        scale_path: Optional[np.ndarray] = None  
    ) -> np.ndarray:
        """
        Simulate a VAR(p) path of differences given sampled (B, Σ) and past lags.

        Model
        -----
        
        ΔX_t = c + Σ_{ℓ=1..p} A_ℓ ΔX_{t−ℓ} + ε_t,
        
        ε_t = chol(Σ) z_t ⊙ s_t,

        with B stacking [c, A_1, ..., A_p] row-wise (m×k, m = 1 + k p). The simulation
        maintains a rolling buffer of the last p differences.

        Parameters
        ----------
        dX_lags : numpy.ndarray, shape (p, k)
            The last p differences ordered from oldest to most recent.
        steps : int
            Simulation horizon.
        z : numpy.ndarray, shape (steps, k)
            Standard-normal shocks.
        B : numpy.ndarray, shape (m, k)
            Stacked coefficients.
        Sigma : numpy.ndarray, shape (k, k)
            Innovation covariance.
        scale_path : numpy.ndarray, optional, shape (steps, k)
            Elementwise scale multipliers.

        Returns
        -------
        numpy.ndarray, shape (steps, k)
            Simulated ΔX path.

        Notes
        -----
        This routine is used inside Monte Carlo scenario generation to propagate
        macro deltas under parameter uncertainty, reflecting the posterior predictive.
        """
      
        p = self.post.p
        
        k = self.post.k
      
        assert z.shape == (steps, k)
      
        Lchol = np.linalg.cholesky((Sigma + Sigma.T) / 2.0 + 1e-9 * np.eye(k))

        c = B[0]                    
      
        Al = [B[1 + l * k : 1 + (l + 1) * k].T for l in range(p)] 

        if p <= 0:
       
            out = np.zeros((steps, k))
       
            for t in range(steps):
       
                innov = Lchol @ z[t]
       
                if scale_path is not None:
       
                    innov = innov * scale_path[t]
       
                out[t] = c + innov
       
            return out

        lags = np.asarray(dX_lags, float)[-p:, :].copy()
       
        out = np.zeros((steps, k))

        head = p - 1

        for t in range(steps):

            pred = c.copy()

            for l in range(1, p + 1):
       
                idx = (head - (l - 1)) % p
       
                pred += Al[l - 1] @ lags[idx]

            innov = Lchol @ z[t]

            if scale_path is not None:
       
                innov = innov * scale_path[t]

            dx_next = pred + innov
       
            out[t] = dx_next

            head = (head + 1) % p
       
            lags[head] = dx_next
      
        return out


def _build_lagged_xy(
    dX: pd.DataFrame, 
    p: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct (Y, X) matrices for a VAR(p) in differences.

    Given a T × k matrix of differences dX (in BASE_REGRESSORS order), build:
   
    - Y of shape (T − p, k), containing ΔX_p, ..., ΔX_{T−1}.
   
    - X of shape (T − p, 1 + k p), containing a constant and stacked lags:
    
    [1, ΔX_{t−1}', ΔX_{t − 2}', ..., ΔX_{t − p}'].

    Parameters
    ----------
    dX : pandas.DataFrame
        Stationary differences with columns ordered as BASE_REGRESSORS.
    p : int
        Lag order.

    Returns
    -------
    Y : numpy.ndarray
    X : numpy.ndarray

    Raises
    ------
    ValueError
        If T ≤ p.
    """

    T = dX.shape[0]
  
    if T <= p:
  
        raise ValueError("Not enough observations for VAR lags.")
  
    Y = dX.values[p:]                    
  
    rows = []
  
    for t in range(p, T):
  
        xrow = [1.0]
  
        for l in range(1, p + 1):
  
            xrow.extend(dX.values[t - l])
  
        rows.append(xrow)
  
    X = np.array(rows)                  
  
    return Y, X


@dataclass
class SVParams:
  
    mu: np.ndarray       
  
    phi: np.ndarray      
  
    sigma_eta: np.ndarray 


@dataclass
class HigherMomentParams:

    dist_name: str

    params: tuple

    mean: float

    std: float

    skew: float

    exkurt: float

    n_eff: int

    mix_weight: float = 1.0


@dataclass
class ModelStepDependence:

    cov: np.ndarray

    corr: np.ndarray

    std: np.ndarray

    mu: np.ndarray


@dataclass
class FittedHyperparams:

    stack_orders: List[Tuple[int, int, int]]

    exog_lags: Tuple[int, ...]

    fourier_k: int

    cv_splits: int

    resid_block_len: int

    student_df: int

    jump_q: float

    gpd_min_exceed: int

    hm_min_obs: int

    hm_clip_skew: float

    hm_clip_excess_kurt: float

    copula_shrink: float

    model_cov_reps: int

    bvar_p: int

    mn_lambdas: Tuple[float, float, float, float]

    niw_nu0: int

    niw_s0_scale: float

    n_sims_required: int

    hm_enabled: bool

    hm_prior_weight: float


@dataclass
class BacktestSummary:

    logscore: float

    rmse: float

    wis90: float

    coverage90: float

    tail_low_exceed: float

    tail_high_exceed: float

    n_folds: int


@dataclass
class RunCachePayload:

    schema_version: int

    signature: str

    created_utc: str

    results_by_ticker: Dict[str, Dict[str, float]]

    df_out_records: List[Dict[str, Any]]

    fitted_hyperparams_by_ticker: Dict[str, Dict[str, Any]]

    backtest_summary_by_ticker: Dict[str, Dict[str, Any]]

    ticker_diagnostics_by_ticker: Dict[str, Dict[str, Any]]

    warning_summary: Dict[str, Any]

    search_trace_by_ticker: Dict[str, Dict[str, Any]]

    run_meta: Dict[str, Any]


@dataclass
class SARIMAXFitStats:

    attempts: int = 0

    converged: int = 0

    non_converged: int = 0

    warnings_start: int = 0

    warnings_conv: int = 0

    warnings_other: int = 0


@dataclass
class HyperparamSearchTrace:

    selected: Dict[str, Any]

    objective: float

    candidates_evaluated: int

    elapsed_sec: float


@dataclass
class TickerDiagnostics:

    fit_stats: Dict[str, Any]

    backtest: Dict[str, Any]

    hp: Dict[str, Any]

    runtime_sec: float


@dataclass
class RunPerfStats:

    sim_forecast_calls: int = 0

    sim_cov_calls: int = 0

    fit_calls: int = 0

    cache_hits: int = 0

    cache_misses: int = 0

    wall_sec_by_phase: Dict[str, float] = None


def estimate_sv_params(
    resid: np.ndarray
) -> SVParams:
    """
    Estimate univariate log-variance AR(1) stochastic volatility proxies per dimension.

    Proxy model
    -----------
    For residuals r_tj (j = 1..k), approximate latent log-variance h_tj by:

        h_tj ≈ log(r_tj^2 + ε),

    and fit an AR(1):
   
        h_tj = μ_j + φ_j (h_{t−1,j} − μ_j) + σ_{η,j} e_tj,  with e_tj ~ N(0, 1).

    Estimation
    ----------
    Ordinary least squares on (h_{t−1}, h_t) with intercept yields estimates of φ_j
    and c_j, with μ_j = c_j / (1 − φ_j) (for |φ_j| < 1). The innovation s.d. σ_{η,j}
    is the residual standard deviation. Estimates are clipped to ensure stability
    (|φ_j| ≤ 0.98) and numerical robustness.

    Parameters
    ----------
    resid : numpy.ndarray, shape (T, k)
        Residuals from a macro model.

    Returns
    -------
    SVParams
        Vectors μ, φ, σ_η.

    Why stochastic volatility
    -------------------------
    Financial and macro series exhibit time-varying volatility. Using SV scales to
    modulate Gaussian shocks produces more realistic dispersion and tail behaviour in
    simulations than homoskedastic alternatives.
    """

    _, k = resid.shape

    mu = np.zeros(k); phi = np.zeros(k); sigma_eta = np.zeros(k)

    H = np.log(np.maximum(resid**2, 1e-10))
   
    for j in range(k):
   
        hj = H[:, j]
   
        hj = hj[np.isfinite(hj)]
   
        if hj.size < 5:
   
            mu[j] = float(np.mean(hj)) if hj.size else 0.0
   
            phi[j] = 0.0
   
            sigma_eta[j] = 0.0
   
            continue
   
        h0 = hj[:-1]
        
        h1 = hj[1:]
      
        X = np.column_stack([np.ones_like(h0), h0])
      
        beta = np.linalg.lstsq(X, h1, rcond = None)[0]
      
        c = float(beta[0])

        a = float(np.clip(beta[1], -0.98, 0.98))

        phi[j] = a

        mu[j] = c / (1.0 - a) if abs(1.0 - a) > 1e-6 else np.mean(hj)

        e = h1 - (c + a * h0)
      
        sigma_eta[j] = float(np.std(e, ddof=1)) if e.size > 1 else 0.0
        
    return SVParams(
        mu = mu, 
        phi = phi, 
        sigma_eta = sigma_eta
    )


def simulate_sv_scales(
    steps: int,
    params: SVParams,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate per-dimension stochastic volatility scale factors.

    Given AR(1) parameters (μ, φ, σ_η) for log-variance h_t, simulate:

        h_0 = μ,
     
        h_t = μ + φ (h_{t−1} − μ) + σ_η e_t,  with e_t ~ N(0, 1),

    and return 
    
        S_t = exp(0.5 h_t). 
        
    Values are clipped below to sqrt(SV_MIN_VAR) to avoid degeneracy.

    Parameters
    ----------
    steps : int
    params : SVParams
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, shape (steps, k)
        Elementwise volatility scales.
    """

    k = params.mu.size
   
    h = np.zeros((steps, k))
   
    h[0] = params.mu
   
    for t in range(1, steps):
   
        e = rng.standard_normal(k)
   
        h[t] = params.mu + params.phi * (h[t-1] - params.mu) + params.sigma_eta * e
   
    S = np.exp(0.5 * h)
   
    S = np.clip(S, np.sqrt(SV_MIN_VAR), None)
   
    return S


@dataclass
class MSVolParams:
  
    P: np.ndarray      
  
    g: np.ndarray        
  
    pi: np.ndarray       
  
    Sigma: np.ndarray    


def _forward_backward(
    loglik_t: np.ndarray, 
    P: np.ndarray,
    pi: np.ndarray
):
    """
    Compute HMM filtered/posterior state probabilities (α, β, γ) and log-likelihood.

    Inputs
    ------
    loglik_t : numpy.ndarray, shape (T, S)
        Per-time, per-state log-likelihoods log p(y_t | S_t = s).

    P : numpy.ndarray, shape (S, S)
        Transition matrix, P_{ij} = p(S_t = j | S_{t−1} = i).

    pi : numpy.ndarray, shape (S,)
        Initial state probabilities.

    Outputs
    -------
    gamma : numpy.ndarray, shape (T, S)
        Filtered/posterior probabilities p(S_t = s | y_{1:T}).
  
    xi : numpy.ndarray, shape (T−1, S, S)
        Pairwise posteriors p(S_{t−1} = i, S_t = j | y_{1:T}).
  
    loglik : float
        Total log-likelihood log p(y_{1:T}).

    Algorithmic notes
    -----------------
    Implements a numerically stable forward–backward algorithm with per-step
    normalisation to prevent underflow. Used within EM-style updates and to obtain
    regime weights.
    """

    T, S = loglik_t.shape

    logP = np.log(np.maximum(P, 1e-300))

    logpi = np.log(np.maximum(pi, 1e-300))

    log_alpha = np.zeros((T, S))

    log_beta = np.zeros((T, S))

    log_alpha[0] = logpi + loglik_t[0]

    for t in range(1, T):

        for s in range(S):

            log_alpha[t, s] = loglik_t[t, s] + logsumexp(log_alpha[t - 1] + logP[:, s])

    loglik = float(logsumexp(log_alpha[-1]))

    log_beta[-1] = 0.0

    for t in range(T - 2, -1, -1):

        for s in range(S):

            log_beta[t, s] = logsumexp(logP[s, :] + loglik_t[t + 1, :] + log_beta[t + 1, :])

    log_gamma = log_alpha + log_beta - loglik

    gamma = np.exp(log_gamma)

    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi = np.zeros((T - 1, S, S))

    for t in range(T - 1):

        log_xi_t = (
            log_alpha[t][:, None]
            + logP
            + loglik_t[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - loglik
        )

        xi_t = np.exp(log_xi_t)

        xi_t /= np.maximum(xi_t.sum(), 1e-300)

        xi[t] = xi_t

    return gamma, xi, loglik


def estimate_ms_vol_params_hmm(
    resid: np.ndarray,
    max_iter: int = 200, 
    tol: float = 1e-5
) -> MSVolParams:
    """
    Estimate a 2-state Gaussian-scale HMM for multivariate residuals.

    Observation model
    -----------------
    Residuals r_t ∈ R^k follow:
     
        r_t | S_t = s ~ N(0, g_s^2 Σ),
    with:
   
    - Σ a base covariance (k×k), positive definite,
   
    - g_0, g_1 > 0 state-specific scale multipliers.

    Hidden state S_t ∈ {0, 1} follows a first-order Markov chain with transition P.

    Estimation
    ----------
    Initialise Σ by the sample covariance; π by a median-split on the Mahalanobis
    distance q_t = r_t' Σ^{-1} r_t; P to a near-sticky matrix. Iterate:

    1) E-step: compute γ_t(s) ∝ p(S_t = s | r_{1:T}) and ξ_t(i, j).
   
    2) M-step:
   
    - P_{ij} ← Σ_t ξ_t(i, j) / Σ_t γ_t(i),
   
    - π ← γ_0,
   
    - g_s^2 ← (Σ_t γ_t(s) q_t) / (k Σ_t γ_t(s)).

    Stop when log-likelihood improvement < tol.

    Parameters
    ----------
    resid : numpy.ndarray, shape (T, k)
    max_iter : int
    tol : float

    Returns
    -------
    MSVolParams
        Estimated P, g, π, Σ.

    Advantages
    ----------
    A Markov-switching volatility layer captures regime shifts (e.g., calm vs stressed
    periods) that simple SV cannot represent alone, and improves realism of simulated
    paths by clustering volatility.
    """
  
    _, k = resid.shape

    Sigma = np.cov(resid, rowvar = False)
  
    Sigma = (Sigma + Sigma.T) * 0.5 + 1e-9 * np.eye(k)
  
    Sinv = np.linalg.inv(Sigma)
  
    _, logdetS = np.linalg.slogdet(Sigma)

    q = np.einsum('ti,ij,tj->t', resid, Sinv, resid)
  
    thr = np.median(q)
  
    z0 = (q <= thr).astype(float)
  
    w0 = np.clip(z0.mean(), 1e-2, 1-1e-2)
  
    pi = np.array([w0, 1-w0])

    P = np.array([
        [0.90, 0.10],
        [0.10, 0.90]
    ])

    
    def _g_from_weights(
        w
    ):
    
        num = (w * q).sum()
    
        den = k * np.maximum(w.sum(), 1e-12)
    
        return np.sqrt(np.maximum(num/den, 1e-6))
    
    
    g = np.array([
        _g_from_weights(
            w = z0
        ), 
        _g_from_weights(
            w = 1 - z0
        )
    ])

    prev_ll = -np.inf
  
    for _ in range(max_iter):

        loglik_t = np.column_stack([
            -0.5 * (k * np.log(g[0] ** 2) + logdetS + q / (g[0] ** 2)),
            -0.5 * (k * np.log(g[1] ** 2) + logdetS + q / (g[1] ** 2))
        ])
        
        gamma, xi, ll = _forward_backward(
            loglik_t = loglik_t, 
            P = P, 
            pi = pi
        )

        P = xi.sum(axis = 0)
        
        P = P / np.maximum(P.sum(axis = 1, keepdims = True), 1e-12)

        pi = gamma[0] / np.maximum(gamma[0].sum(), 1e-12)

        g[0] = np.sqrt(np.maximum((gamma[:,0] @ q) / (k * np.maximum(gamma[:,0].sum(), 1e-12)), 1e-6))
        
        g[1] = np.sqrt(np.maximum((gamma[:,1] @ q) / (k * np.maximum(gamma[:,1].sum(), 1e-12)), 1e-6))

        if ll - prev_ll < tol:
     
            break
     
        prev_ll = ll

    return MSVolParams(
        P = P,
        g = g, 
        pi = pi,
        Sigma = Sigma
    )


def simulate_ms_states_conditional(
    steps: int,
    P: np.ndarray, 
    rng: np.random.Generator, 
    p_init: np.ndarray
) -> np.ndarray:
    """
    Simulate a Markov chain of states S_t given transition matrix P and an initial
    distribution p_init.

    Parameters
    ----------
    steps : int
    P : numpy.ndarray, shape (2, 2)
    rng : numpy.random.Generator
    p_init : numpy.ndarray, shape (2,)
        Probabilities for S_0.

    Returns
    -------
    numpy.ndarray, shape (steps,)
        Simulated states in {0, 1}.
    """

    S = np.zeros(steps, dtype = int)

    S[0] = 1 if rng.random() < p_init[1] else 0

    for t in range(1, steps):

        p = P[S[t-1]]

        u = rng.random()

        S[t] = 1 if u > p[0] else 0

    return S


def _mn_prior(
    dX: pd.DataFrame, 
    p: int,
    lambda1: float, 
    lambda2: float, 
    lambda3: float,
    lambda4: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct Minnesota prior mean and covariance for a VAR(p) coefficient matrix.

    Prior structure
    ---------------
    Let B be an (m×k) stacked coefficient matrix (m = 1 + k p). The prior is:

    - Mean: E[B] = 0 (per-equation zero mean).
    - Covariance: V_0 diagonal at the design level with entries:

    For coefficient on variable j at lag ℓ in equation i:
      
        Var(B_{pos, i}) =   (λ1 / ℓ^{λ3})^2 × 1, if i = j;
                        
                            (λ2 × σ_i / σ_j), if i ≠ j.

    For constant term in equation i:
   
        Var(B_{0, i}) = (λ1 × λ4 × σ_i)^2.

    Here σ_i are scale estimates for each equation, e.g., AR(1) residual s.d.

    Parameters
    ----------
    dX : pandas.DataFrame
        Stationary differences (T×k).
    p : int
    lambda1, lambda2, lambda3, lambda4 : float
        Minnesota hyperparameters.

    Returns
    -------
    B0 : numpy.ndarray, shape (m, k)
        Zero matrix.
    V0 : numpy.ndarray, shape (m, m)
        Diagonal prior covariance.

    Why Minnesota
    -------------
    This shrinkage encodes that own-lags are more informative than cross-lags and
    that higher-order lags decay in influence. It mitigates overfitting and improves
    forecasting in small and moderate samples, a common situation with weekly macro data.
    """
   
    k = dX.shape[1]
   
    m = 1 + k * p

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
   
        sig_i = np.std(resid, ddof=1) if len(resid) > 3 else np.std(s)
   
        sig.append(sig_i + 1e-8)
   
    sig = np.array(sig)

    V0_diag = np.zeros((k, m))
    
    for i in range(k):
    
        V0_diag[i, 0] = (lambda1 * lambda4 * sig[i]) ** 2
    
        for l in range(1, p + 1):
    
            for j in range(k):
    
                pos = 1 + (l - 1) * k + j
    
                if i == j:
    
                    var = (lambda1 / (l ** lambda3)) ** 2
    
                else:
    
                    var = (lambda1 * lambda2 / (l ** lambda3)) ** 2 * ((sig[i] / sig[j]) ** 2)
    
                V0_diag[i, pos] = var

    V0 = np.diag(np.mean(V0_diag, axis=0))
    
    B0 = np.zeros((m, k))
    
    return B0, V0


def _chol_logdet_psd(
    A: np.ndarray,
    jitter: float = MN_TUNE_JITTER
) -> Tuple[np.ndarray, float]:
    """
    Compute a numerically robust Cholesky factor and log-determinant of a PSD matrix.

    Attempts a Cholesky factorisation with increasing ridge jitter until success, then
    returns:
    
        L such that L L' ≈ A + jitter I,
    
        logdet = log |A + jitter I| = 2 Σ_i log L_{ii}.

    Parameters
    ----------
    A : numpy.ndarray, shape (n, n)
    jitter : float
        Initial ridge level.

    Returns
    -------
    L : numpy.ndarray
    logdet : float

    Raises
    ------
    np.linalg.LinAlgError
        If the matrix remains indefinite after multiple jitter increases.
    """

    A = (A + A.T) * 0.5

    jj = jitter

    for _ in range(7): 

        try:

            L = np.linalg.cholesky(A + jj * np.eye(A.shape[0]))

            logdet = 2.0 * np.sum(np.log(np.diag(L)))

            return L, float(logdet)

        except np.linalg.LinAlgError:

            jj *= 10.0

    s, ld = np.linalg.slogdet(A + jj * np.eye(A.shape[0]))

    if s <= 0:

        raise np.linalg.LinAlgError("Matrix not PD even after jitter.")

    return np.linalg.cholesky(A + jj * np.eye(A.shape[0])), float(ld)


def _bvar_posterior_and_log_evidence(
    Y: np.ndarray,
    X: np.ndarray,
    V0_diag: np.ndarray,
    S0: np.ndarray,
    nu0: int
) -> Tuple[BVARPosterior, float]:
    """
    Compute the MN-IW posterior for a VAR and the marginal log evidence (up to constants).

    Given Y (T×k) and X (T×m), with prior:

        B | Σ ~ MN(0, V0, Σ)   and   Σ ~ IW(S0, ν0),

    the posterior is:

        K = X'X + V0^{-1},

        B_n = K^{-1} X'Y,

        S_n = S0 + Y'Y − B_n' K B_n,

        ν_n = ν0 + T.

    The constant-free log evidence (marginal likelihood of Y | prior hyperparameters) is:

        log p(Y | λ) ∝ (k/2)(log|V_n| − log|V_0|) + (ν0/2) log|S0| − (ν_n/2) log|S_n|.

    Parameters
    ----------
    Y : numpy.ndarray, shape (T, k)
    X : numpy.ndarray, shape (T, m)
    V0_diag : numpy.ndarray, shape (m,)
        Diagonal of V0 (for speed).
    S0 : numpy.ndarray, shape (k, k)
    nu0 : int

    Returns
    -------
    post : BVARPosterior
    log_evd : float

    Use
    ---
    The evidence is used to tune Minnesota hyperparameters via empirical Bayes:
    higher evidence indicates a better balance of fit and parsimony.
    """

    T, k = Y.shape
    
    V0_inv_diag = 1.0 / (V0_diag + 1e-18)
    
    XtX = X.T @ X
    
    K = XtX + np.diag(V0_inv_diag)

    logdet_V0 = float(np.sum(np.log(V0_diag + 1e-18)))
    
    _, logdet_K = _chol_logdet_psd(
        A = K
    )
    
    logdet_Vn = -logdet_K

    XtY = X.T @ Y
    
    Bn = np.linalg.solve(K, XtY)

    YtY = Y.T @ Y
    
    Sn = S0 + YtY - (Bn.T @ K @ Bn)
    
    _, logdet_S0 = _chol_logdet_psd(
        A = S0
    )
    
    _, logdet_Sn = _chol_logdet_psd(
        A = Sn
    )

    nu_n = nu0 + T

    log_evd = (k * 0.5) * (logdet_Vn - logdet_V0) + (nu0 * 0.5) * logdet_S0 - (nu_n * 0.5) * logdet_Sn

    post = BVARPosterior(
        Bn = Bn,
        Vn = np.linalg.inv(K),
        Sn = Sn, 
        nun = nu_n,
        p = None,
        k = k
    )  
    
    return post, float(log_evd)


def _mn_prior_diag(
    dX: pd.DataFrame, 
    p: int,
    lambda1: float, 
    lambda2: float, 
    lambda3: float, 
    lambda4: float
) -> np.ndarray:
    """
    Fast construction of diag(V0) for the Minnesota prior.

    Identical in meaning to `_mn_prior`, but returns only the diagonal of V0, enabling
    efficient posterior algebra with diagonal priors.

    Parameters
    ----------
    dX : pandas.DataFrame
    p : int
    lambda1, lambda2, lambda3, lambda4 : float

    Returns
    -------
    V0_diag : numpy.ndarray, shape (1 + k p,)
    """

  
    k = dX.shape[1]
  
    m = 1 + k * p
  
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
  
        sig_i = np.std(resid, ddof = 1) if len(resid) > 3 else np.std(s)
  
        sig.append(sig_i + 1e-8)
  
    sig = np.asarray(sig)

    V0_diag_eq = np.zeros((k, m))

    for i in range(k):

        V0_diag_eq[i, 0] = (lambda1 * lambda4 * sig[i]) ** 2

        for l in range(1, p + 1):

            for j in range(k):

                pos = 1 + (l - 1) * k + j

                if i == j:

                    var = (lambda1 / (l ** lambda3)) ** 2

                else:

                    var = (lambda1 * lambda2 / (l ** lambda3)) ** 2 * ((sig[i] / sig[j]) ** 2)

                V0_diag_eq[i, pos] = var

    V0_diag = np.mean(V0_diag_eq, axis = 0)  

    return V0_diag


def _tune_minnesota_hyperparams(
    dX: pd.DataFrame, p: int,
    nu0: int, s0_scale: float,
    grid_l1: Tuple[float, ...] = MN_TUNE_L1_GRID,
    grid_l2: Tuple[float, ...] = MN_TUNE_L2_GRID,
    grid_l3: Tuple[float, ...] = MN_TUNE_L3_GRID,
    grid_l4: Tuple[float, ...] = MN_TUNE_L4_GRID
) -> Tuple[BVARPosterior, Tuple[float, float, float, float]]:
    """
    Empirical-Bayes tuning of Minnesota hyperparameters by maximising log evidence.

    Procedure
    ---------
    1) Split differences dX into (Y, X) for VAR(p).
    
    2) Construct S0 as a shrinkage blend of identity and the diagonal of OLS residual covariance.
    
    3) For each grid point (λ1, λ2, λ3, λ4), compute the posterior and log evidence
    using `_bvar_posterior_and_log_evidence`.
    
    4) Return the posterior at the maximising grid point.

    Parameters
    ----------
    dX : pandas.DataFrame
    p : int
    nu0 : int
    s0_scale : float
    grid_l1..grid_l4 : tuple[float, ...]

    Returns
    -------
    BVARPosterior, tuple[float, float, float, float]
        Posterior at the best hyperparameters and the chosen λ tuple.

    Why evidence maximisation
    -------------------------
    Evidence trades off in-sample fit (via S_n) against model complexity (via V_n),
    guarding against over-aggressive or too-weak shrinkage.
    """
   
    Y, X = _build_lagged_xy(
        dX = dX, 
        p = p
    )
   
    k = dX.shape[1]
   
    B_ols = np.linalg.lstsq(X, Y, rcond = None)[0]
   
    resid_ols = Y - X @ B_ols
   
    cov_ols = np.cov(resid_ols, rowvar = False)
   
    rho = 0.5
   
    S0 = s0_scale * ((1 - rho) * np.eye(k) + rho * np.diag(np.diag(cov_ols)))
   
    nu0 = int(max(nu0, k + 2))

    best_logev = -np.inf
    
    best_post = None
    
    best_lmb =  None
   
    for l1 in grid_l1:
    
        for l2 in grid_l2:
     
            for l3 in grid_l3:
     
                for l4 in grid_l4:
     
                    V0_diag = _mn_prior_diag(
                        dX = dX, 
                        p = p, 
                        lambda1 = l1,
                        lambda2 = l2, 
                        lambda3 = l3, 
                        lambda4 = l4
                    )
     
                    try:
     
                        post, logev = _bvar_posterior_and_log_evidence(
                            Y = Y, 
                            X = X,
                            V0_diag = V0_diag, 
                            S0 = S0,
                            nu0 = nu0
                        )
     
                    except Exception:
     
                        continue
     
                    if logev > best_logev:
     
                        best_logev = logev
                        
                        best_post = post
                        
                        best_lmb = (l1, l2, l3, l4)

    if best_post is None:

        raise RuntimeError("EB tuning failed for Minnesota prior (no valid candidates).")
 
    best_post.p = p
 
    return best_post, best_lmb


def _fit_bvar_minnesota(
    dX: pd.DataFrame, 
    p: int
) -> BVARModel:
    """
    Fit a Minnesota BVAR(p) using either evidence maximisation or predictive-logscore tuning.

    If `MN_TUNE_METHOD == "logscore"`, selects λ to maximise average one-step-ahead
    log predictive density on a holdout block; otherwise maximises marginal evidence.

    Returns
    -------
    BVARModel
        Encapsulates the posterior used for simulation.

    Fallback
    --------
    On tuning failure, falls back to fixed hyperparameters MN_LAMBDA1..4 and computes
    the closed-form posterior.
    """

    try:

        if MN_TUNE_METHOD.lower() == "logscore":

            post, lmb = _tune_minnesota_by_logscore(
                dX = dX, 
                p = p,
                nu0 = NIW_NU0,
                s0_scale = NIW_S0_SCALE,
                grid_l1 = MN_TUNE_L1_GRID, 
                grid_l2 = MN_TUNE_L2_GRID,
                grid_l3 = MN_TUNE_L3_GRID, 
                grid_l4 = MN_TUNE_L4_GRID
            )

        else:

            post, lmb = _tune_minnesota_hyperparams(
                dX = dX, 
                p = p,
                nu0 = NIW_NU0, 
                s0_scale = NIW_S0_SCALE,
                grid_l1 = MN_TUNE_L1_GRID, 
                grid_l2 = MN_TUNE_L2_GRID,
                grid_l3 = MN_TUNE_L3_GRID,
                grid_l4 = MN_TUNE_L4_GRID
            )

        logger.info("Minnesota tuned λ = (%.3f, %.3f, %.3f, %.3f) via %s", *lmb, MN_TUNE_METHOD)

        return BVARModel(
            post = post
        )

    except Exception as e:

        logger.warning("Minnesota tuning failed (%s). Reverting to fixed λ.", e)

        k = dX.shape[1]

        Y, X = _build_lagged_xy(
            dX = dX,
            p = p
        )
        
        B0, V0 = _mn_prior(
            dX = dX,
            p = p,
            lambda1 = MN_LAMBDA1,
            lambda2 = MN_LAMBDA2, 
            lambda3 = MN_LAMBDA3, 
            lambda4 = MN_LAMBDA4
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
        
        post = BVARPosterior(
            Bn = Bn,
            Vn = Vn, 
            Sn = Sn,
            nun = nun,
            p = p, 
            k = k
        )
        
        return BVARModel(
            post = post
        )


def _logpdf_gaussian(
    y: np.ndarray,
    mu: np.ndarray, 
    Sigma: np.ndarray
) -> float:
    """
    Evaluate the multivariate normal log-density log N(y | μ, Σ) robustly.

    Returns −∞ (a large negative sentinel) if Σ is not numerically positive definite.

    Parameters
    ----------
    y, mu : numpy.ndarray, shape (k,)
    Sigma : numpy.ndarray, shape (k, k)

    Returns
    -------
    float
    """

    k = y.size

    S = (Sigma + Sigma.T) * 0.5

    try:

        L = np.linalg.cholesky(S + 1e-9 * np.eye(k))

        alpha = np.linalg.solve(L, (y - mu))

        qf = alpha @ alpha

        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        return float(-0.5 * (k * np.log(two_pi) + logdet + qf))

    except np.linalg.LinAlgError:

        return -1e12


def _posterior_from_prior_and_data(
    Y: np.ndarray, 
    X: np.ndarray, 
    V0_diag: np.ndarray,
    S0: np.ndarray,
    nu0: int
):
    """
    Compute K, B_n, S_n, ν_n for a diagonal-V0 Minnesota prior and given data.

    Parameters
    ----------
    Y : numpy.ndarray, shape (T, k)
    X : numpy.ndarray, shape (T, m)
    V0_diag : numpy.ndarray, shape (m,)
    S0 : numpy.ndarray, shape (k, k)
    nu0 : int

    Returns
    -------
    K : numpy.ndarray, shape (m, m)
    Bn : numpy.ndarray, shape (m, k)
    Sn : numpy.ndarray, shape (k, k)
    nun : int
    """

    V0_inv_diag = 1.0 / (V0_diag + 1e-18)

    XtX = X.T @ X

    K = XtX + np.diag(V0_inv_diag)

    XtY = X.T @ Y

    Bn = np.linalg.solve(K, XtY)

    YtY = Y.T @ Y

    Sn = S0 + YtY - (Bn.T @ K @ Bn)

    nun = nu0 + Y.shape[0]

    return K, Bn, Sn, nun


def _tune_minnesota_by_logscore(
    dX: pd.DataFrame, p: int,
    nu0: int, s0_scale: float,
    holdout_frac: float = 0.2,
    grid_l1: Tuple[float, ...] = MN_TUNE_L1_GRID,
    grid_l2: Tuple[float, ...] = MN_TUNE_L2_GRID,
    grid_l3: Tuple[float, ...] = MN_TUNE_L3_GRID,
    grid_l4: Tuple[float, ...] = MN_TUNE_L4_GRID
) -> Tuple[BVARPosterior, Tuple[float,float,float,float]]:
    """
    Select Minnesota hyperparameters by maximising average one-step-ahead log predictive density.

    Procedure
    ---------
    1) Split (Y, X) into training (first T0 points) and holdout (remaining).

    2) For each λ grid point:
 
    a) Form V0_diag with `_mn_prior_diag`, compute K, Bn, Sn, ν_n on the training block.
 
    b) For each holdout x_t, compute predictive mean μ_t = x_t' Bn and predictive
        covariance Σ_pred = s_t × (Sn / (ν_n − k − 1)), where:
 
            s_t = 1 + x_t' K^{-1} x_t.
 
    c) Accumulate Σ_t log N(Y_t | μ_t, Σ_pred).
 
    3) Return the posterior parameters of the best λ.

    Advantages
    ----------
    Directly targets out-of-sample density forecasting, which is often aligned with
    decision-making under uncertainty.
    """

    Y, X = _build_lagged_xy(
        dX = dX, 
        p = p
    )
   
    T = X.shape[0]
    
    k = Y.shape[1]
   
    if T < 30:
   
        raise RuntimeError("Not enough data for predictive-score tuning.")
   
    T0 = int(max(20, np.floor((1.0-holdout_frac) * T)))
   
    Y_tr = Y[:T0]
    
    X_tr = X[:T0]
   
    Y_te = Y[T0:]
    
    X_te = X[T0:]

    B_ols = np.linalg.lstsq(X_tr, Y_tr, rcond = None)[0]
    
    resid_ols = Y_tr - X_tr @ B_ols
    
    cov_ols = np.cov(resid_ols, rowvar = False)
    
    rho = 0.5
  
    S0 = s0_scale * ((1 - rho) * np.eye(k) + rho * np.diag(np.diag(cov_ols)))
  
    nu0 = int(max(nu0, k + 2))

    best_score, best_post, best_lmb = -np.inf, None, None
  
    for l1 in grid_l1:
  
        for l2 in grid_l2:
  
            for l3 in grid_l3:
  
                for l4 in grid_l4:
  
                    V0_diag = _mn_prior_diag(
                        dX = dX,
                        p = p, 
                        lambda1 = l1,
                        lambda2 = l2, 
                        lambda3 = l3,
                        lambda4 = l4
                    )
  
                    K, Bn, Sn, nun = _posterior_from_prior_and_data(
                        Y = Y_tr, 
                        X = X_tr, 
                        V0_diag = V0_diag,
                        S0 = S0,
                        nu0 = nu0
                    )

                    Ksym = (K + K.T) * 0.5 + 1e-9 * np.eye(K.shape[0])
  
                    L = np.linalg.cholesky(Ksym)

                    dof = max(nun - k - 1, k + 2)
  
                    inv_dof = 1.0 / dof

                    score = 0.0
  
                    for t in range(X_te.shape[0]):
  
                        x = X_te[t]                 
  
                        mu = x @ Bn              

                        y = np.linalg.solve(L, x)
  
                        z = np.linalg.solve(L.T, y)
  
                        s = float(1.0 + x @ z)

                        Sigma_pred = s * (Sn * inv_dof)
  
                        score += _logpdf_gaussian(
                            y = Y_te[t], 
                            mu = mu, 
                            Sigma = Sigma_pred
                        )

                    if score > best_score:
  
                        best_score = score
  
                        best_post = BVARPosterior(
                            Bn = Bn, 
                            Vn = None,
                            Sn = Sn,
                            nun = nun, 
                            p = p, 
                            k = k
                        ) 
  
                        best_lmb = (l1, l2, l3, l4)
  
    if best_post is None:
  
        raise RuntimeError("Predictive-score tuning failed.")
  
    logger.info("Minnesota predictive-score tuned λ = (%.3f, %.3f, %.3f, %.3f)", *best_lmb)
  
    return best_post, best_lmb


def build_future_exog(
    hist_tail_vals: np.ndarray,          
    macro_path: np.ndarray,              
    col_order: list[str],          
    scaler: StandardScaler,
    lags: tuple[int, ...] = EXOG_LAGS,
    jump: Optional[np.ndarray] = None,   
    fourier_period: int = 52,
    fourier_k: int = FOURIER_K,
    t_start: int = 0,
    layout: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Construct a future exogenous design matrix for SARIMAX in **exact training column order**.

    Inputs
    ------
    hist_tail_vals : numpy.ndarray, shape (context_rows, k_macro)
        The final `context_rows` rows of the base macro regressors in the training order.
    
    macro_path : numpy.ndarray, shape (H, k_macro)
        Simulated macro deltas for the forecast horizon H.
   
    col_order : list[str]
        Exact SARIMAX training column order (e.g., ["Cpi_L0", "Cpi_L1", ..., "jump_ind",
        "fourier_sin_1", "fourier_cos_1", ...]).
   
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler from the training design.
   
    lags : tuple[int, ...]
        Lag indices L used when fitting (e.g., (0, 1, 2)).
   
    jump : numpy.ndarray, optional, shape (H,)
        Bernoulli jump indicators to include as an exogenous column if present.
    fourier_period : int, default 52
        Seasonal period used in Fourier construction.
    fourier_k : int, default FOURIER_K
        Number of Fourier harmonics.
    t_start : int, default 0
        Absolute phase offset to ensure train/forecast continuity.

    Construction
    ------------
   
    1) Vertically stack `hist_tail_vals` and `macro_path` to create a contiguous series.
   
    2) For each unique lag L in `lags`, slice the lagged H×k block needed for the horizon.
   
    3) For each name in `col_order`, fill the corresponding column:
   
    - "{Base}_L{L}" → the appropriate lag block and variable column,
   
    - "jump_ind" → `jump` (or zeros),
   
    - "fourier_sin_k", "fourier_cos_k" → from internally generated Fourier block.
   
    4) Apply the training `scaler.transform` to standardise exactly as during fitting.

    Returns
    -------
    numpy.ndarray, shape (H, n_cols)
        Scaled exogenous design in training column order.

    Why column-order fidelity matters
    ---------------------------------
    Statsmodels aligns exogenous columns by **position**, not by name. Matching the
    training order guarantees coefficients multiply the intended signals at forecast time.
    """

    H = macro_path.shape[0]

    if layout is None:
  
        key = (tuple(col_order), tuple(lags))
  
        layout = _cache_get_touch(
            d = _COMPILED_EXOG_LAYOUT_CACHE, 
            key = key
        )
  
        if layout is None:
  
            jump_col = -1
  
            lag_cols: List[int] = []
  
            lag_L: List[int] = []
  
            lag_ridx: List[int] = []
  
            sin_cols: List[int] = []
  
            sin_kidx: List[int] = []
  
            cos_cols: List[int] = []
  
            cos_kidx: List[int] = []

            for j, name in enumerate(col_order):
              
                if name == "jump_ind":
              
                    jump_col = j
              
                    continue

                if name.startswith("fourier_sin_"):
              
                    try:
              
                        sin_cols.append(j)
              
                        sin_kidx.append(int(name.split("_")[-1]) - 1)
              
                    except Exception:
              
                        pass
              
                    continue

                if name.startswith("fourier_cos_"):
              
                    try:
              
                        cos_cols.append(j)
              
                        cos_kidx.append(int(name.split("_")[-1]) - 1)
              
                    except Exception:
              
                        pass
              
                    continue

                if "_L" in name:
              
                    base, Ls = name.rsplit("_L", 1)
              
                    try:
              
                        L = int(Ls)
              
                        ridx = BASE_REGRESSORS.index(base)
              
                    except Exception:
              
                        continue
              
                    lag_cols.append(j)
              
                    lag_L.append(L)
              
                    lag_ridx.append(ridx)

            layout = {
                "jump_col": int(jump_col),
                "lag_cols": np.asarray(lag_cols, dtype = int),
                "lag_L": np.asarray(lag_L, dtype = int),
                "lag_ridx": np.asarray(lag_ridx, dtype = int),
                "sin_cols": np.asarray(sin_cols, dtype = int),
                "sin_kidx": np.asarray(sin_kidx, dtype = int),
                "cos_cols": np.asarray(cos_cols, dtype = int),
                "cos_kidx": np.asarray(cos_kidx, dtype = int),
                "n_cols": int(len(col_order)),
            }
          
            _COMPILED_EXOG_LAYOUT_CACHE[key] = layout
          
            _maybe_trim_cache(
                d = _COMPILED_EXOG_LAYOUT_CACHE
            )

    F_fourier = _fourier_block(
        H = H,
        period = fourier_period,
        K = fourier_k,
        t_start = t_start,
    )

    series = np.vstack([hist_tail_vals, macro_path])
   
    base_rows_start = series.shape[0] - H

    lag_L_arr = layout["lag_L"]

    uniq_lags = sorted(set(lag_L_arr.tolist())) if lag_L_arr.size else []
   
    lag_blocks = {L: series[base_rows_start - L: base_rows_start - L + H, :] for L in uniq_lags}

    X = np.zeros((H, int(layout["n_cols"])), dtype=float)

    lag_cols = layout["lag_cols"]
 
    lag_ridx = layout["lag_ridx"]
 
    if lag_cols.size:
 
        for L in uniq_lags:
 
            m = (lag_L_arr == L)
 
            if not np.any(m):
 
                continue
 
            X[:, lag_cols[m]] = lag_blocks[L][:, lag_ridx[m]]

    sin_cols = layout["sin_cols"]
 
    sin_kidx = layout["sin_kidx"]
 
    if sin_cols.size:
 
        X[:, sin_cols] = F_fourier[:, 2 * sin_kidx]

    cos_cols = layout["cos_cols"]
 
    cos_kidx = layout["cos_kidx"]
 
    if cos_cols.size:
 
        X[:, cos_cols] = F_fourier[:, 2 * cos_kidx + 1]

    jump_col = int(layout["jump_col"])
 
    if jump_col >= 0:
 
        if jump is None:
 
            X[:, jump_col] = 0.0
 
        else:
 
            X[:, jump_col] = np.asarray(jump, float)

    X_scaled = scaler.transform(X)
    
    return np.clip(X_scaled, -5.0, 5.0)


def _fit_macro_candidates(
    df_levels_weekly: pd.DataFrame
    ) -> List[Tuple[str, object, dict, float]]:
    """
    Fit macro-dynamics candidates (ECVAR with multiple ranks, BVAR(p), VAR(1) fallback)
    and score them with an information criterion proxy.

    Steps
    -----
    1) Build stationary differences ΔX and cointegration levels L.
   
    2) Attempt Johansen cointegration; for ranks r = 1..min(k−1, 3), fit ECVAR by OLS on:
  
        ΔX_t = c + A ΔX_{t−1} + K β' L_{t−1} + ε_t.
    3) Fit Minnesota-BVAR(p) and compute OLS residual covariance as a surrogate for IC.
  
    4) If all else fails, fit a VAR(1) in differences.
  
    5) For each candidate, compute an IC of the form:
  
        IC = T_eff × log|Σ̂| + pcount × log(T_eff),
  
    which mimics Schwarz-type penalties.

    Returns
    -------
    list of tuples
        (label, model_object, aux_dict, ic_value), where aux_dict contains starting
        states/lags and residuals for volatility estimation.

    Rationale
    ---------
    Combining models with different long-run structures (cointegration vs unrestricted)
    hedges specification risk. The simple IC stabilises the weight allocation without
    over-indexing on within-sample AIC alone.
    """
   
    dX, L = _macro_stationary_deltas(
        df_levels = df_levels_weekly, 
        cointegration = True
    )
   
    df = L.join(dX, how = "inner", lsuffix = "_L", rsuffix = "")
   
    L = L.loc[df.index]
    
    dX = dX.loc[df.index]
    
    k = lenBR
    
    candidates: List[Tuple[str, object, dict, float]] = []

    rank_max = max(0, min(k - 1, 3))  
    
    try:
    
        joh = coint_johansen(L.values, det_order = 0, k_ar_diff = 1)
    
    except Exception:
    
        joh = None

    if joh is not None and USE_RANK_UNCERTAINTY:
       
        for r in range(1, rank_max + 1):
       
            try:
       
                beta = joh.evec[:, :r]
       
                ect = (L.values @ beta)
       
                Y = dX.values[1:]
       
                R = np.column_stack([np.ones((len(dX) - 1, 1)), dX.values[:-1], ect[:-1]])
       
                RtR = R.T @ R
       
                RtY = R.T @ Y
       
                B = np.linalg.pinv(RtR) @ RtY
       
                c = B[0]
       
                A = B[1:1 + k].T
       
                K = B[1 + k:].T
       
                resid = Y - R @ B
       
                T_eff = resid.shape[0]
       
                Sigma = np.cov(resid, rowvar = False)

                pcount = k + k * k + k * r
               
                _, logdet = _chol_logdet_psd(
                    A = Sigma + 1e-9 * np.eye(k)
                )
               
                ic = T_eff * logdet + pcount * np.log(max(T_eff, 2))
               
                model = ECVARModel(
                    c = c, 
                    A = A,
                    K = K, 
                    beta = beta, 
                    Sigma = Sigma, 
                    k = k
                )
               
                aux = {
                    "L0": L.values[-1], 
                    "dX0": dX.values[-1], 
                    "resid": resid
                }
               
                candidates.append((f"ecvar_r{r}", model, aux, float(ic)))
            
            except Exception:
            
                continue

    try:
       
        bvar = _fit_bvar_minnesota(
            dX = dX,
            p = BVAR_P
        )
       
        Yv, Xv = _build_lagged_xy(
            dX = dX, 
            p = BVAR_P
        )
       
        B_ols = np.linalg.lstsq(Xv, Yv, rcond = None)[0]
       
        resid_ols = Yv - Xv @ B_ols
       
        Sigma_ols = np.cov(resid_ols, rowvar = False)
       
        T_eff = resid_ols.shape[0]
       
        pcount = k + k*k*BVAR_P
       
        _, logdet = _chol_logdet_psd(Sigma_ols + 1e-9 * np.eye(k))
       
        ic = T_eff * logdet + pcount * np.log(max(T_eff, 2))
       
        aux = {
            "dX_lags": dX.values[-BVAR_P:], 
            "resid": resid_ols
        }
       
        candidates.append(("bvar", bvar, aux, float(ic)))
    
    except Exception as e:
    
        logger.warning("BVAR candidate failed (%s)", e)

    if not candidates:
      
        var = VAR(dX).fit(maxlags = 1, trend = "c")
      
        if var.k_ar < 1:
      
            c = dX.mean().values
      
            A = np.zeros((k, k))
      
            Sigma = np.cov((dX - dX.mean()).values, rowvar = False)
      
            resid = (dX - dX.mean()).values
      
        else:
      
            c = var.intercept
      
            A = var.coefs[0]
      
            Sigma = var.sigma_u
      
            resid = var.resid
      
        model = VARModel(
            c = c, 
            A = A,
            Sigma = Sigma,
            k = k
        )
      
        T_eff = resid.shape[0]
      
        pcount = k + k*k
      
        _, logdet = _chol_logdet_psd(
            A = Sigma + 1e-9 * np.eye(k)
        )
      
        ic = T_eff * logdet + pcount * np.log(max(T_eff, 2))
      
        aux = {
            "dX0": dX.values[-1], 
            "resid": resid
        }
      
        candidates.append(("var", model, aux, float(ic)))

    return candidates


def simulate_macro_paths_for_country(
    df_levels_weekly: pd.DataFrame,
    steps: int,
    n_sims: int,
    seed: int
) -> np.ndarray:
    """
    Simulate macro delta scenarios by mixture of fitted ECVAR/BVAR/VAR candidates,
    optionally layered with stochastic volatility and Markov-switching volatility.

    Process
    -------
    1) Fit candidates and compute weights w ∝ exp(−0.5 (IC − min(IC))).
    
    2) Estimate residual-driven volatility layers:
    
    - Stochastic volatility (per dimension) via `estimate_sv_params` and
        `simulate_sv_scales`.
    
    - 2-state HMM volatility: scales g_s obtained by `estimate_ms_vol_params_hmm`,
        states simulated by `simulate_ms_states_conditional`.
    
    The final scale_path is elementwise product (SV × MS).
    
    3) For antithetic sampling, draw z for half the simulations and reuse −z for the
    remainder to reduce Monte Carlo variance.
    
    4) For each candidate:
    
    - ECVAR: propagate (L, ΔX) using `ECVARModel.simulate`.
    
    - BVAR: sample (B, Σ) and simulate with `BVARModel.simulate`.
    
    - VAR: simulate with `VARModel.simulate`.
    
    5) Return an array of shape (n_sims, steps, k).

    Parameters
    ----------
    df_levels_weekly : pandas.DataFrame
    steps : int
    n_sims : int
    seed : int

    Returns
    -------
    numpy.ndarray
        Macro delta scenarios.

    Advantages
    ----------
    - Mixture over structural assumptions spreads model risk.

    - SV and MS volatility layers deliver clustered volatility and fat tails.

    - Antithetic variates improve efficiency without biasing moments.
    
    """
    
    k = lenBR
    
    rng = np.random.default_rng(seed)
    
    sims = np.zeros((n_sims, steps, k), dtype = np.float32)

    candidates = _fit_macro_candidates(
        df_levels_weekly = df_levels_weekly
    )
   
    ics = np.array([ic for (_, _, _, ic) in candidates], dtype = float)

    rs = _runtime_settings()

    spread = float(np.nanstd(ics)) if np.isfinite(ics).any() else 1.0
 
    tau = float(max(float(rs["macro_tau_floor"]), spread, 1e-6))

    scaled = (ics - np.min(ics)) / tau

    w = np.exp(-scaled)
  
    if not np.isfinite(w).any():
  
        w = np.ones_like(ics)
  
    w = w / w.sum()

    eps = float(np.clip(rs["macro_weight_eps"], 0.0, 0.20))
 
    if w.size > 0:
 
        w = (1.0 - eps) * w + eps / w.size
 
        w = w / np.maximum(w.sum(), 1e-12)
  
    labels = [lab for (lab, _, _, _) in candidates]

    resid_for_vol = None
   
    try:
   
        best_idx = int(np.argmin(ics))
   
        resid_for_vol = candidates[best_idx][2].get("resid", None)
   
    except Exception:
   
        resid_for_vol = None

    if resid_for_vol is None or not np.isfinite(resid_for_vol).all():
   
        dX = _macro_stationary_deltas(
            df_levels = df_levels_weekly
        )
   
        try:
   
            var = VAR(dX).fit(maxlags = max(1, BVAR_P), trend = "c")
   
            resid_for_vol = var.resid
   
        except Exception:
   
            resid_for_vol = dX.values - dX.values.mean(0)

    sv_params = estimate_sv_params(
        resid = resid_for_vol
    ) if USE_SV_MACRO else None

    ms_hmm = estimate_ms_vol_params_hmm(
        resid = resid_for_vol
    ) if USE_MS_VOL else None


    def _filtered_last_probs(
        resid,
        ms
    ):
        _, k = resid.shape
    
        Sinv = np.linalg.inv(ms.Sigma)
    
        _, logdetS = np.linalg.slogdet(ms.Sigma)
    
        q = np.einsum('ti,ij,tj->t', resid, Sinv, resid)
    
        loglik_t = np.column_stack([
            -0.5*(k * np.log(ms.g[0] ** 2) + logdetS + q / (ms.g[0] ** 2)),
            -0.5*(k * np.log(ms.g[1] ** 2) + logdetS + q / (ms.g[1] ** 2))
        ])
        
        gamma, _, _ = _forward_backward(
            loglik_t = loglik_t, 
            P = ms.P, 
            pi = ms.pi
        )
        
        return gamma[-1]  


    p_init = _filtered_last_probs(
        resid = resid_for_vol, 
        ms = ms_hmm
    ) if ms_hmm else np.array([0.5, 0.5])

    gamma_full = None
   
    if ms_hmm is not None:
   
        _, kdim = resid_for_vol.shape
   
        Sinv = np.linalg.inv(ms_hmm.Sigma)
   
        _, logdetS = np.linalg.slogdet(ms_hmm.Sigma)
   
        q_macro = np.einsum('ti,ij,tj->t', resid_for_vol, Sinv, resid_for_vol)
   
        loglik_t = np.column_stack([
            -0.5 * (kdim * np.log(ms_hmm.g[0] ** 2) + logdetS + q_macro / (ms_hmm.g[0] ** 2)),
            -0.5 * (kdim * np.log(ms_hmm.g[1] ** 2) + logdetS + q_macro / (ms_hmm.g[1] ** 2)),
        ])
        
        gamma_full, _, _ = _forward_backward(
            loglik_t = loglik_t, 
            P = ms_hmm.P, 
            pi = ms_hmm.pi
        )

    w_rw = _regime_weighted_macro_weights(
        candidates = candidates, 
        ms_hmm = ms_hmm, 
        gamma_full = gamma_full,
        k = lenBR
    )
    
    if w_rw is not None and np.all(np.isfinite(w_rw)) and w_rw.sum() > 0:
    
        w = w_rw

    half = n_sims // 2
   
    z_cube = rng.standard_normal((half, steps, k)).astype(np.float32)

    counts = rng.multinomial(half, w)  
   
    start = 0
   
    for cand_idx, cnt in enumerate(counts):
   
        if cnt == 0:
   
            continue
   
        label, model, aux, _ = candidates[cand_idx]
       
        end = start + cnt
       
        for i in range(start, end):

            S_sv = simulate_sv_scales(
                steps = steps, 
                params = sv_params,
                rng = rng
            ) if sv_params is not None else np.ones((steps, k))

            if ms_hmm is not None:
               
                S_states = simulate_ms_states_conditional(
                    steps = steps,
                    P = ms_hmm.P, 
                    rng = rng, 
                    p_init = p_init
                )

                gpath = np.where(S_states == 0, ms_hmm.g[0], ms_hmm.g[1]).astype(float)

                S_ms = np.tile(gpath.reshape(-1, 1), (1, k))
        
            else:
        
                S_ms = np.ones((steps, k))

            scale_path = S_sv * S_ms

            z = z_cube[i]
            
            z_neg = -z

            if label.startswith("ecvar"):
             
                dX_path, _ = model.simulate(aux["L0"].copy(), aux["dX0"].copy(), steps, z, scale_path = scale_path)
             
                dX_path_b, _ = model.simulate(aux["L0"].copy(), aux["dX0"].copy(), steps, z_neg, scale_path = scale_path)
          
            elif label == "bvar":
          
                B, Sigma = model.sample_coeffs_and_sigma(rng)
          
                dX_path = model.simulate(aux["dX_lags"].copy(), steps, z, B, Sigma, scale_path = scale_path)
          
                dX_path_b = model.simulate(aux["dX_lags"].copy(), steps, z_neg, B, Sigma, scale_path = scale_path)
          
            else:  
          
                dX_path = model.simulate(aux["dX0"].copy(), steps, z, scale_path = scale_path)
          
                dX_path_b = model.simulate(aux["dX0"].copy(), steps, z_neg, scale_path = scale_path)

            sims[i] = dX_path
     
            sims[i + half] = dX_path_b
     
        start = end

    if n_sims % 2 == 1:
     
        sims[-1] = sims[0]
        
    logger.info("Macro candidates: %s with weights %s (count=%d, tau=%.3f, eps=%.3f)", labels, np.round(w, 3), len(labels), tau, eps)
    
    if USE_SV_MACRO:
    
        logger.info("SV params: mean phi=%.2f, mean sigma_eta=%.3f", float(np.mean(sv_params.phi)), float(np.mean(sv_params.sigma_eta)))

    return sims


def _regime_weighted_macro_weights(
    candidates: List[Tuple[str, object, dict, float]],
    ms_hmm: Optional[MSVolParams],
    gamma_full: Optional[np.ndarray],
    k: int,
) -> Optional[np.ndarray]:
    """
    Re-weight macro candidates using regime-aware scores derived from HMM posteriors.

    Score
    -----
    For candidate c with residuals R (T×k) and per-time HMM posteriors γ_t(s), define:

    q_t = R_t' Σ_c^{-1} R_t,
    
    score_c = 0.5 Σ_t [ γ_t(0) × (k log g_0^2 + log|Σ_c| + q_t / g_0^2) + γ_t(1) × (k log g_1^2 + log|Σ_c| + q_t / g_1^2) ] + pcount × log(T),

    which is the negative (up to constants) of a regime-weighted Gaussian likelihood
    with BIC-style penalty. Convert scores to weights by a softmax over −score.

    Parameters
    ----------
    candidates : list
    ms_hmm : MSVolParams | None
    gamma_full : numpy.ndarray | None
    k : int

    Returns
    -------
    numpy.ndarray | None
        Normalised weights aligned to `candidates`, or None if inputs are insufficient.

    Rationale
    ---------
    When volatility regimes are present, a candidate’s residual variance interacts with
    regime intensity. Regime-aware weighting prefers candidates that fit high-volatility
    periods without over-penalising low-volatility spans.
    """
   
    if ms_hmm is None or gamma_full is None or gamma_full.ndim != 2 or gamma_full.shape[1] != 2:
   
        return None
   
    g0, g1 = ms_hmm.g
   
    weights = []
   
    labels = []
   
    T_gamma = gamma_full.shape[0]
   
    for (label, _model, aux, _ic) in candidates:
   
        resid = aux.get("resid", None)
   
        if resid is None or resid.ndim != 2 or resid.shape[1] != k:
   
            continue
   
        T = resid.shape[0]
   
        Tm = min(T, T_gamma)
   
        if Tm < 10:
   
            continue
   
        R = resid[-Tm:]
   
        G = gamma_full[-Tm:]

        Sigma_c = np.cov(R, rowvar = False)
   
        Sigma_c = (Sigma_c + Sigma_c.T) * 0.5 + 1e-9 * np.eye(k)
   
        Sinv_c = np.linalg.inv(Sigma_c)
      
        _, logdet_c = np.linalg.slogdet(Sigma_c)
      
        q = np.einsum('ti,ij,tj->t', R, Sinv_c, R)
      
        term0 = G[:, 0] * (k * np.log(g0**2) + logdet_c + q / (g0**2))
      
        term1 = G[:, 1] * (k * np.log(g1**2) + logdet_c + q / (g1**2))

        if label.startswith("ecvar_r"):
      
            r = int(label.split("r")[-1])
      
            pcount = k + k * k + k * r
      
        elif label == "bvar":
      
            pcount = k + k * k * BVAR_P
      
        else:
      
            pcount = k + k * k
      
        score = 0.5 * (term0.sum() + term1.sum()) + pcount * np.log(max(Tm, 2))
      
        weights.append(score)
      
        labels.append(label)
   
    if not weights:
   
        return None
   
    scores = np.array(weights, float)
   
    rs = _runtime_settings()
    
    spread = float(np.nanstd(scores)) if np.isfinite(scores).any() else 1.0
    
    tau = float(max(float(rs["macro_tau_floor"]), spread, 1e-6))

    w = np.exp(-(scores - scores.min()) / tau)
    
    w /= np.maximum(w.sum(), 1e-12)

    eps = float(np.clip(rs["macro_weight_eps"], 0.0, 0.20))
    
    w = (1.0 - eps) * w + eps / max(1, w.size)
    
    w /= np.maximum(w.sum(), 1e-12)

    w_full = np.zeros(len(candidates), float)
    
    label_to_pos = {lab: j for j, lab in enumerate(labels)}
   
    for i, (lab, *_rest) in enumerate(candidates):
   
        j = label_to_pos.get(lab, None)

        if j is not None:
   
            w_full[i] = w[j]

    s = w_full.sum()
   
    if s <= 0:
   
        return None
   
    return w_full / s


@dataclass
class Ensemble:
    """
    Container for an ensemble of SARIMAX fits and their AICc-derived weights.

    Attributes
    ----------
    fits : list
        Statsmodels SARIMAXResults objects.
    weights : numpy.ndarray
        Non-negative, summing to one.
    """
  
    fits: List
  
    weights: np.ndarray

    meta: Optional[Dict[str, int]] = None


def _fourier_block(
    H: int, 
    period: int, 
    K: int,
    t_start: int = 0,
) -> np.ndarray:
    """
    Generate a deterministic Fourier seasonal block for weekly seasonality.

    For horizon H, period P (typically 52), and K harmonics, returns an H×(2K) matrix.
    The phase starts at `t_start` to preserve continuity across train and forecast windows.

    Columns are:
      
        sin(2π k t / P), cos(2π k t / P),  for k = 1..K and t = 0..H−1.

    Parameters
    ----------
    H : int
    period : int
    K : int

    Returns
    -------
    numpy.ndarray, shape (H, 2K)

    Why Fourier terms
    -----------------
    Fourier pairs concisely capture periodic seasonality without discontinuities
    or the parameter explosion of full seasonal ARIMA terms in short weekly samples.
    """

    key = (H, period, K, int(t_start))

    blk = _FOURIER_CACHE.get(key)

    if blk is None:

        t = np.arange(int(t_start), int(t_start) + H)

        cols = []

        for kf in range(1, K+1):

            cols.append(np.sin(two_pi * kf * t / period))

            cols.append(np.cos(two_pi * kf * t / period))

        blk = np.column_stack(cols)

        _FOURIER_CACHE[key] = blk

    return blk


def make_exog(
    df: pd.DataFrame,
    base: list[str],
    lags: tuple[int, ...] = EXOG_LAGS,
    add_fourier: bool = True,
    K: int = 2,
    include_jump_exog: bool = False,
    t_start: int = 0,
) -> pd.DataFrame:
    """
    Assemble an exogenous design matrix from lagged macro regressors, optional jump indicator,
    and optional Fourier seasonal terms.

    Construction
    ------------
    1) For each lag L in `lags`, add a block with columns "{name}_L{L}".
   
    2) If `include_jump_exog=True` and "jump_ind" exists, append it as a separate column.
   
    3) If `add_fourier=True`, append sin/cos pairs from `_fourier_block`.
   
    4) Drop rows with any NaNs (due to lagging) and enforce a weekly frequency tag.

    Parameters
    ----------
    df : pandas.DataFrame
    base : list[str]
    lags : tuple[int, ...], default (0, 1, 2)
    add_fourier : bool, default True
    K : int, default 2
    include_jump_exog : bool, default False
    t_start : int, default 0
        Fourier phase offset.

    Returns
    -------
    pandas.DataFrame
        Exogenous matrix indexed by time with exact column names used downstream.

    Reasoning
    ---------
    Explicit lags permit lead/lag relationships between macro deltas and returns;
    jump indicators capture tail-risk asymmetries; Fourier terms model weekly seasonality.
    """

    blocks: list[pd.DataFrame] = []

    base_cols = [c for c in base if c in df.columns]

    if not base_cols:

        return pd.DataFrame(index=df.index)

    for L in lags:

        block = df.loc[:, base_cols].shift(L)

        block.columns = [f"{c}_L{L}" for c in base_cols]

        blocks.append(block)

    if include_jump_exog and "jump_ind" in df.columns:

        blocks.append(df[["jump_ind"]])

    if add_fourier:

        F = _fourier_block(
            H = len(df.index), 
            period = 52, 
            K = K, 
            t_start = t_start
        )

        cols = []

        for kf in range(1, K + 1):

            cols += [f"fourier_sin_{kf}", f"fourier_cos_{kf}"]

        blocks.append(pd.DataFrame(F, index = df.index, columns = cols))

    if not blocks:
   
        return pd.DataFrame(index = df.index)

    X = pd.concat(blocks, axis = 1, copy = False)
   
    X = X.dropna()
   
    freq = getattr(df.index, "freq", None) or to_offset("W-SUN")
    
    try:
   
        X = X.asfreq(freq)
   
    except Exception:
   
        try:
   
            X.index.freq = to_offset("W-SUN")
   
        except Exception:
   
            pass
   
    return X


def _sample_autocorr(
    x: np.ndarray,
    lag: int
) -> float:
    """
    Estimate sample autocorrelation at a positive lag using centred dot products.

    Definition
    ----------
    For lag `k > 0`, define overlapping vectors:

    - `x0 = (x_1, ..., x_{T-k})`
  
    - `x1 = (x_{1+k}, ..., x_T)`.

    After centring each segment by its own sample mean, the estimator is:

        rho_k = <x0 - mean(x0), x1 - mean(x1)>
                / sqrt( ||x0 - mean(x0)||^2 * ||x1 - mean(x1)||^2 ).

    The function returns `0.0` when:

    - `lag <= 0`,
  
    - insufficient observations (`T <= lag`),
  
    - or the denominator is numerically negligible.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional series.
    lag : int
        Requested lag in observations.

    Returns
    -------
    float
        Finite sample autocorrelation estimate in approximately `[-1, 1]`.

    Why this process
    ----------------
    Multiple hyperparameter heuristics in this module rely on short-lag persistence
    diagnostics. A minimal, allocation-light estimator is preferable to repeated heavy
    library calls inside tuning loops.

    Advantages
    ----------
    - Numerically stable due to explicit denominator guard.
 
    - Fast enough for repeated use during candidate screening.
 
    - Independent centring of both segments reduces bias from local level shifts.
 
    """

    if lag <= 0 or x.size <= lag:

        return 0.0

    x0 = x[:-lag]

    x1 = x[lag:]

    x0 = x0 - np.mean(x0)

    x1 = x1 - np.mean(x1)

    den = np.sqrt(np.dot(x0, x0) * np.dot(x1, x1))

    if den <= 1e-12:

        return 0.0

    return float(np.dot(x0, x1) / den)


def _warning_bucket(
    message: str
) -> str:
    """
    Map SARIMAX warning text to coarse diagnostic categories.

    Classification
    --------------
    The function performs substring matching and returns:

    - `"start"` for non-stationary AR starts or non-invertible MA starts,
   
    - `"conv"` for optimiser non-convergence warnings,
   
    - `"other"` for all remaining messages.

    Parameters
    ----------
    message : str
        Warning message emitted during model fitting.

    Returns
    -------
    str
        One of `{"start", "conv", "other"}`.

    Why this process
    ----------------
    Hyperparameter search produces many warnings across folds and orders. Coarse
    bucketing creates interpretable diagnostics without storing unbounded raw text.

    Advantages
    ----------
    - Low-overhead warning telemetry for large fit grids.
   
    - Separates start-value issues from true convergence failures.
   
    - Enables concise run-level reporting and regression monitoring.
   
    """

    msg = str(message)

    if "Non-stationary starting autoregressive parameters found" in msg:
      
        return "start"

    if "Non-invertible starting MA parameters found" in msg:
      
        return "start"

    if "Maximum Likelihood optimization failed to converge" in msg:
      
        return "conv"

    return "other"


def _accumulate_fit_warning_stats(
    wrns: List[warnings.WarningMessage],
    fit_stats: Optional[SARIMAXFitStats],
) -> None:
    """
    Accumulate warning counters from a single fitting attempt into shared statistics.

    Process
    -------
    For each captured warning object `w`:

    1) Extract textual content from `w.message`.
 
    2) Classify with `_warning_bucket`.
 
    3) Increment exactly one counter in `fit_stats`:
       `warnings_start`, `warnings_conv`, or `warnings_other`.

    Parameters
    ----------
    wrns : list[warnings.WarningMessage]
        Warnings captured within a `warnings.catch_warnings(record=True)` context.
    fit_stats : SARIMAXFitStats | None
        Mutable statistics container. When `None`, the function exits immediately.

    Returns
    -------
    None

    Why this process
    ----------------
    Warning volume can be high in staged optimisation. Aggregation by category retains
    operational signal while avoiding large memory overhead from full warning histories.

    Advantages
    ----------
    - Constant-memory warning accounting.
  
    - Consistent categorisation shared by all fitting pathways.
  
    - Supports model-quality diagnostics in cached and uncached workflows.
  
    """

    if fit_stats is None:
   
        return

    for w in wrns:
   
        bucket = _warning_bucket(str(getattr(w, "message", "")))

        if bucket == "start":
   
            fit_stats.warnings_start += 1
   
        elif bucket == "conv":
   
            fit_stats.warnings_conv += 1
   
        else:
   
            fit_stats.warnings_other += 1


def _build_sarimax_start_params(
    mdl: SARIMAX,
    y_arr: np.ndarray,
    X_arr: np.ndarray,
    p: int,
    q: int
) -> np.ndarray:
    """
    Construct data-driven start params aligned to `mdl.param_names`.
    """

    y = np.asarray(y_arr, float).reshape(-1)

    X = np.asarray(X_arr, float)

    if X.ndim == 1:

        X = X.reshape(-1, 1)

    n = y.size

    kx = X.shape[1] if X.ndim == 2 else 0

    if n >= 5 and kx > 0:

        Xd = np.column_stack([np.ones(n), X])

        try:

            b = np.linalg.lstsq(Xd, y, rcond=None)[0]

        except Exception:

            b = np.zeros(kx + 1)

        intercept = float(b[0]) if b.size > 0 else float(np.mean(y))

        beta = b[1:1 + kx] if b.size >= (kx + 1) else np.zeros(kx)

        resid = y - Xd @ np.r_[intercept, beta]

    else:

        intercept = float(np.mean(y)) if y.size else 0.0

        beta = np.zeros(kx)

        resid = y - intercept

    ar_starts = np.zeros(max(p, 0), float)

    if p > 0 and resid.size > p + 5:

        Y_ar = resid[p:]

        X_ar = np.column_stack([resid[p - l : -l] for l in range(1, p + 1)])

        try:

            ar_starts = np.linalg.lstsq(X_ar, Y_ar, rcond=None)[0]

        except Exception:

            ar_starts = np.zeros(p, float)

    ar_starts = np.clip(ar_starts, -START_AR_CLIP, START_AR_CLIP)

    ma_starts = np.zeros(max(q, 0), float)

    for i in range(q):

        rho = _sample_autocorr(resid, i + 1)

        ma_starts[i] = np.clip(-rho, -START_MA_CLIP, START_MA_CLIP)

    sigma2 = float(np.var(resid, ddof=1)) if resid.size > 2 else float(np.var(resid))

    sigma2 = max(sigma2, 1e-6)

    start = np.zeros(mdl.k_params, dtype=float)

    for i, name in enumerate(mdl.param_names):

        if name in {"intercept", "const", "trend"}:

            start[i] = intercept

        elif name.startswith("x"):

            try:

                j = int(name[1:]) - 1

                start[i] = float(beta[j]) if 0 <= j < beta.size else 0.0

            except Exception:

                start[i] = 0.0

        elif name.startswith("ar.L"):

            try:

                l = int(name.split("L")[-1]) - 1

                start[i] = float(ar_starts[l]) if 0 <= l < ar_starts.size else 0.0

            except Exception:

                start[i] = 0.0

        elif name.startswith("ma.L"):

            try:

                l = int(name.split("L")[-1]) - 1

                start[i] = float(ma_starts[l]) if 0 <= l < ma_starts.size else 0.0

            except Exception:

                start[i] = 0.0

        elif name == "sigma2":

            start[i] = sigma2

        else:

            start[i] = 0.0

    start = np.where(np.isfinite(start), start, 0.0)

    return start


def _is_sarimax_fit_usable(
    res
) -> bool:
    """
    Apply strict validity checks to a fitted SARIMAX result object.

    Validation criteria
    -------------------
    The result is considered usable only if all checks pass:

    1) `res` is not `None`.
   
    2) Parameter vector exists, is non-empty, and all entries are finite.
   
    3) `AIC` is finite.
   
    4) Scale estimate (`sigma^2` proxy) is finite and strictly positive.
   
    5) Optimiser convergence flag is true (from `mle_retvals` when available).
   
    6) Parameter covariance matrix is retrievable and finite.

    Parameters
    ----------
    res : Any
        Candidate statsmodels SARIMAX fit result.

    Returns
    -------
    bool
        `True` when the fit is numerically and diagnostically admissible.

    Why this process
    ----------------
    Convergence in numerical optimisation does not guarantee a reliable model object.
    Subsequent scoring, forecasting, and covariance extraction require finite estimates.

    Advantages
    ----------
    - Prevents silent propagation of degenerate fits into ensemble weighting.
   
    - Reduces downstream exceptions in forecast simulation.
   
    - Enforces consistent quality gate across fallback optimisation paths.
   
    """

    if res is None:

        return False

    params = np.asarray(getattr(res, "params", np.array([])), float)

    if params.size == 0 or not np.all(np.isfinite(params)):

        return False

    aic = float(getattr(res, "aic", np.nan))

    if not np.isfinite(aic):

        return False

    scale = float(getattr(res, "scale", np.nan))

    if (not np.isfinite(scale)) or (scale <= 0.0):

        return False

    mle = getattr(res, "mle_retvals", None)

    if isinstance(mle, dict):

        converged = bool(mle.get("converged", True))

    else:

        converged = bool(getattr(mle, "converged", True))

    if not converged:

        return False

    try:

        covp = np.asarray(res.cov_params(), float)

        if covp.size and not np.all(np.isfinite(covp)):

            return False

    except Exception:

        return False

    return True


def _fit_sarimax_result_with_starts(
    mdl: SARIMAX,
    y_arr: np.ndarray,
    X_arr: np.ndarray,
    order: Tuple[int, int, int],
    fit_stats: Optional[SARIMAXFitStats] = None,
    fit_mode: str = "auto",
    start_override: Optional[np.ndarray] = None,
):
    """
    Fit SARIMAX with staged fallback and warning accounting.
    """

    p, _, q = order

    base_start = _build_sarimax_start_params(
        mdl = mdl,
        y_arr = y_arr,
        X_arr = X_arr,
        p = p,
        q = q,
    )

    start_params = base_start
  
    if start_override is not None:
  
        try:
  
            ov = np.asarray(start_override, float).reshape(-1)
  
            if ov.size == mdl.k_params and np.all(np.isfinite(ov)):

                start_params = 0.65 * ov + 0.35 * base_start

        except Exception:

            start_params = base_start

    rs = _runtime_settings()

    if str(fit_mode).lower() == "final":

        max_a = int(rs["fit_maxiter_final_a"])

        max_b = int(rs["fit_maxiter_final_b"])

        max_c = int(rs["fit_maxiter_final_c"])

    else:

        max_a = int(rs["fit_maxiter_a"])

        max_b = int(rs["fit_maxiter_b"])

        max_c = int(rs["fit_maxiter_c"])

    if fit_stats is not None:

        fit_stats.attempts += 1

    def _run_fit(
        *,
        method: str,
        maxiter: int,
        start: Optional[np.ndarray] = None,
        transformed: Optional[bool] = None,
    ):

        with warnings.catch_warnings(record=True) as wrns:

            warnings.simplefilter("always")

            if SUPPRESS_EXPECTED_CONVERGENCE_WARNINGS:

                warnings.filterwarnings(
                    "ignore",
                    message = "Non-stationary starting autoregressive parameters found.*",
                    category = UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message = "Non-invertible starting MA parameters found.*",
                    category = UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message = "Maximum Likelihood optimization failed to converge.*",
                    category = Warning,
                )

            kwargs: Dict[str, Any] = {
                "disp": False,
                "method": method,
                "maxiter": int(maxiter),
            }

            if start is not None:
      
                kwargs["start_params"] = start

            if transformed is not None:
      
                kwargs["transformed"] = transformed

            try:
      
                res = mdl.fit(**kwargs)
      
            except Exception:
      
                _accumulate_fit_warning_stats(
                    wrns = wrns, 
                    fit_stats = fit_stats
                )
      
                return None

            _accumulate_fit_warning_stats(
                wrns = wrns, 
                fit_stats = fit_stats
            )

            if _is_sarimax_fit_usable(
                res = res
            ):
            
                if fit_stats is not None:
            
                    fit_stats.converged += 1
            
                return res

            if fit_stats is not None:
                
                fit_stats.non_converged += 1

            return None

    res = _run_fit(
        method = "lbfgs",
        maxiter = max_a,
        start = start_params,
        transformed = False,
    )

    if res is not None:
  
        return res

    res = _run_fit(
        method = "lbfgs",
        maxiter = max_b,
        start = None,
        transformed = None,
    )

    if res is not None:
   
        return res

    warm = _run_fit(
        method = "powell",
        maxiter = max_c,
        start = start_params,
        transformed = False,
    )

    if warm is None:
 
        return None

    try:
 
        warm_start = np.asarray(warm.params, float)
 
    except Exception:
 
        return None

    return _run_fit(
        method = "lbfgs",
        maxiter = max_b,
        start = warm_start,
        transformed = False,
    )


def _extract_forecast_cov_from_simulation(
    res,
    exog_f: np.ndarray,
    horizon: int,
    reps: int,
    rng: np.random.Generator,
    target_rel_err: float = TARGET_COV_REL_ERR,
) -> np.ndarray:
    """
    Estimate HxH forecast covariance using repeated path simulation.
    """

    H = int(horizon)

    rs = _runtime_settings()

    Rmax = int(np.clip(reps, int(rs["cov_reps_min"]), int(rs["cov_reps_max"])))
 
    target = float(max(1e-3, target_rel_err))
 
    batch = int(min(max(20, int(rs["cov_batch"])), Rmax))

    try:
 
        cov = np.eye(H, dtype = float)
 
        cov_prev: Optional[np.ndarray] = None
 
        reps_done = 0
 
        mean = np.zeros(H, dtype = float)
 
        M2 = np.zeros((H, H), dtype = float)

        while reps_done < Rmax:
      
            take = int(min(batch, Rmax - reps_done))

            sim = res.simulate(
                nsimulations = H,
                repetitions = take,
                exog = exog_f,
                anchor = "end",
                random_state = rng,
            )

            A = np.asarray(sim, float)

            if A.ndim == 3:
          
                if A.shape[1] == 1:
          
                    A = A[:, 0, :]
          
                elif A.shape[0] == 1:
          
                    A = A[0, :, :]
          
                else:
          
                    A = A[:, 0, :]

            if A.ndim == 1:

                A = A.reshape(H, 1)

            if A.ndim != 2:

                A = A.reshape(H, -1)

            if A.shape[0] == H:

                samples = A.T

            elif A.shape[1] == H:

                samples = A

            else:

                samples = A.reshape(-1, H)

            if samples.shape[0] < 1:

                break

            m = int(samples.shape[0])

            batch_mean = np.mean(samples, axis = 0)

            centered = samples - batch_mean

            batch_M2 = centered.T @ centered

            if reps_done == 0:
        
                mean = batch_mean
        
                M2 = batch_M2
        
                reps_done = m
        
            else:
        
                total_n = reps_done + m
        
                delta = batch_mean - mean
        
                mean = mean + delta * (m / max(total_n, 1))
        
                M2 = M2 + batch_M2 + np.outer(delta, delta) * (reps_done * m / max(total_n, 1))
        
                reps_done = total_n

            if reps_done < int(rs["cov_reps_min"]):
        
                continue

            if reps_done < 2:
        
                continue

            cov_now = np.atleast_2d(M2 / max(reps_done - 1, 1)).astype(float)
        
            cov_now = (cov_now + cov_now.T) * 0.5

            if cov_prev is not None:
        
                den = float(np.linalg.norm(cov_prev, ord="fro"))
        
                den = max(den, 1e-10)
        
                rel = float(np.linalg.norm(cov_now - cov_prev, ord="fro") / den)
        
                if rel <= target:
        
                    cov = cov_now
        
                    break

            cov_prev = cov_now
        
            cov = cov_now

        if reps_done < 2:
        
            raise RuntimeError("Not enough simulation repetitions for covariance.")

    except Exception:

        f = res.get_forecast(steps = H, exog = exog_f)

        v = np.maximum(np.asarray(f.var_pred_mean, float), 1e-12)

        cov = np.diag(v)

    cov = (cov + cov.T) * 0.5

    eig, vec = np.linalg.eigh(cov)

    eig = np.maximum(eig, 1e-10)

    cov = (vec * eig) @ vec.T

    cov = (cov + cov.T) * 0.5

    return cov


def _nearest_psd_corr(
    R: np.ndarray
) -> np.ndarray:
    """
    Project a symmetric matrix onto a valid positive-semidefinite correlation matrix.

    Algorithm
    ---------
    Given an input matrix `R`:

    1) Symmetrise: `R <- (R + R^T) / 2`.
   
    2) Set unit diagonal to enforce correlation scale.
   
    3) Eigen-decompose: `R = Q diag(lambda) Q^T`.
   
    4) Clip eigenvalues: `lambda_i <- max(lambda_i, 1e-8)`.
   
    5) Reconstruct PSD matrix: `C = Q diag(lambda_clipped) Q^T`.
   
    6) Renormalise to unit diagonal:
   
       `C_ij <- C_ij / sqrt(C_ii C_jj)`.
   
    7) Re-symmetrise and force exact ones on the diagonal.

    Parameters
    ----------
    R : numpy.ndarray
        Square matrix intended to represent correlation structure.

    Returns
    -------
    numpy.ndarray
        Symmetric PSD correlation matrix with diagonal equal to one.

    Why this process
    ----------------
    Finite-sample covariance manipulations and shrinkage can produce slight indefiniteness.
    Copula simulation requires a valid correlation matrix for Cholesky- or eigen-based
    Gaussian sampling.

    Advantages
    ----------
    - Guarantees admissible correlation geometry.
   
    - Preserves most of the original dependence pattern while removing negative eigenmodes.
   
    - Numerically robust for repeated Monte-Carlo dependence construction.
   
    """

    R = np.asarray(R, float)

    R = (R + R.T) * 0.5

    np.fill_diagonal(R, 1.0)

    eig, vec = np.linalg.eigh(R)

    eig = np.maximum(eig, 1e-8)

    C = (vec * eig) @ vec.T

    C = (C + C.T) * 0.5

    d = np.sqrt(np.maximum(np.diag(C), 1e-12))

    C = C / np.outer(d, d)

    C = (C + C.T) * 0.5

    np.fill_diagonal(C, 1.0)

    return C


def _mixture_mean_cov(
    mus: List[np.ndarray],
    covs: List[np.ndarray],
    weights: np.ndarray,
    copula_shrink: float = COPULA_SHRINK,
) -> ModelStepDependence:
    """
    Combine model-specific predictive moments into mixture moments and a copula correlation.

    Mathematical formulation
    ------------------------
    For model index `k = 1..K` with weight `w_k`, mean vector `mu_k` (length `H`), and
    covariance `Sigma_k` (`H x H`), define normalised weights:

        w_k = w_k / sum_j w_j.

    Mixture mean:

        mu_mix = sum_k w_k mu_k.

    Mixture covariance (law of total covariance):

        Sigma_mix = sum_k w_k [ Sigma_k + (mu_k - mu_mix)(mu_k - mu_mix)^T ].

    The implementation then:

    - symmetrises `Sigma_mix`,
  
    - clips eigenvalues to enforce PSD,
  
    - derives per-step standard deviations `std_i = sqrt(Sigma_mix_ii)`,
  
    - converts to correlation `Corr = Sigma_mix / (std std^T)`,
  
    - applies linear shrinkage towards identity:
  
      `Corr_shrunk = (1 - s) Corr + s I`, with `s = clip(copula_shrink, 0, 0.95)`,
  
    - projects to nearest PSD correlation via `_nearest_psd_corr`.

    Parameters
    ----------
    mus : list[numpy.ndarray]
        Predictive mean vectors from ensemble members.
    covs : list[numpy.ndarray]
        Predictive covariance matrices from ensemble members.
    weights : numpy.ndarray
        Non-negative ensemble weights.
    copula_shrink : float, default COPULA_SHRINK
        Identity-shrinkage intensity for dependence stabilisation.

    Returns
    -------
    ModelStepDependence
        Dataclass containing mixture covariance, correlation, standard deviations, and mean.

    Why this process
    ----------------
    Scenario simulation requires a single coherent horizon dependence structure, whereas
    the ensemble produces model-specific moments. Total-covariance aggregation preserves both
    within-model uncertainty and between-model mean dispersion.

    Advantages
    ----------
    - Statistically principled moment aggregation for model mixtures.

    - Shrinkage plus PSD projection improves numerical stability for high-horizon sampling.

    - Retains cross-step dependence information required by copula-based innovation draws.

    """

    w = np.asarray(weights, float)

    w = w / np.maximum(w.sum(), 1e-12)

    mu_arr = np.asarray(mus, float)

    mu_mix = np.tensordot(w, mu_arr, axes = (0, 0))

    H = mu_mix.size

    Sigma = np.zeros((H, H), float)

    for wk, muk, Sk in zip(w, mus, covs):

        dm = (muk - mu_mix).reshape(-1, 1)

        Sigma += wk * (Sk + dm @ dm.T)

    Sigma = (Sigma + Sigma.T) * 0.5

    eig, vec = np.linalg.eigh(Sigma)

    eig = np.maximum(eig, 1e-10)

    Sigma = (vec * eig) @ vec.T

    Sigma = (Sigma + Sigma.T) * 0.5

    std = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))

    corr = Sigma / np.outer(std, std)

    shrink = float(np.clip(copula_shrink, 0.0, 0.95))

    corr = (1.0 - shrink) * corr + shrink * np.eye(H)

    corr = _nearest_psd_corr(corr)

    return ModelStepDependence(
        cov = Sigma,
        corr = corr,
        std = std,
        mu = mu_mix,
    )


def _fit_sarimax_by_orders_cached_np(
    y: pd.Series,
    X_arr: np.ndarray,
    index: pd.DatetimeIndex,
    orders: List[Tuple[int, int, int]],
    col_order: List[str],
    time_varying: bool = False,
    fit_mode: str = "auto",
    warm_start_by_order: Optional[Dict[Tuple[int, int, int], np.ndarray]] = None,
) -> Dict[Tuple[int, int, int], object]:
    """
    Fit multiple SARIMAX order candidates on NumPy exogenous arrays with memoised reuse.

    Model specification
    -------------------
    For each candidate order `(p, 0, q)`, the fitted mean equation is:

        y_t = c + sum_{i=1..p} phi_i y_{t-i}
                + sum_{j=1..q} theta_j epsilon_{t-j}
                + x_t^T beta + epsilon_t,

        epsilon_t ~ N(0, sigma^2),

    with no seasonal component (`seasonal_order = (0, 0, 0, 0)`), constant trend, and
    optional time-varying regression coefficients.

    Caching process
    ---------------
    A per-order cache key is constructed from training shape/signature metadata and the
    order itself:

        key = ("NP", n_obs, last_timestamp, n_features,
               hash(col_order), hash(fit_mode), hash(order)).

    On cache hit, stored fit objects are returned immediately. On miss, the function:

    1) Builds a statsmodels `SARIMAX` instance.
   
    2) Fits via `_fit_sarimax_result_with_starts`, including staged optimisation fallback.
   
    3) Stores valid fits in `_fit_by_order_cache_np`.
   
    4) Trims cache size incrementally via `_maybe_trim_cache`.

    Parameters
    ----------
    y : pandas.Series
        Endogenous return series aligned to `index`.
    X_arr : numpy.ndarray
        Standardised exogenous design matrix.
    index : pandas.DatetimeIndex
        Time index for SARIMAX date alignment.
    orders : list[tuple[int, int, int]]
        Candidate non-seasonal ARMA orders.
    col_order : list[str]
        Exogenous column order used in cache signature construction.
    time_varying : bool, default False
        If true, fit time-varying exogenous coefficients in the state-space model.
    fit_mode : {"auto", "final"}, default "auto"
        Controls optimiser iteration budgets in staged fitting.
    warm_start_by_order : dict[order, numpy.ndarray] | None
        Optional previously fitted parameter vectors used for order-specific warm starts.

    Returns
    -------
    dict[tuple[int, int, int], object]
        Mapping from order to usable fitted SARIMAX results.

    Why this process
    ----------------
    Repeated order fitting occurs inside rolling-origin procedures and hyperparameter
    search. Without memoisation, identical subproblems are re-solved many times.

    Advantages
    ----------
    - Substantial runtime reduction through fold-level fit reuse.
  
    - Improved optimiser stability via optional warm starts from earlier folds.
  
    - Strict usability gating prevents invalid fits from polluting downstream ensembles.
  
    """
  
    out: Dict[Tuple[int, int, int], object] = {}

    freq = getattr(index, "freqstr", None) or pd.infer_freq(index) or "W-SUN"
   
    base_key = (
        "NP",                     
        len(y),
        int(index[-1].value) if len(index) else 0,
        X_arr.shape[1],
        hash(tuple(col_order)),  
        hash(str(fit_mode).lower()),
    )

    for od in orders:
       
        key = (*base_key, hash(od))
       
        cached = _cache_get_touch(
            d = _fit_by_order_cache_np, 
            key = key
        )
        
        if cached is not None:
        
            out[od] = cached
        
            continue
       
        try:
       
            mdl = SARIMAX(
                y.values,
                exog = X_arr,
                order = od,
                seasonal_order = (0, 0, 0, 0),
                trend = "c",
                enforce_stationarity = True,
                enforce_invertibility = True,
                validate_specification=False,
                time_varying_regression = time_varying,
                mle_regression = not time_varying,
                dates = index,
                freq = freq,
            )
            
            local_stats = SARIMAXFitStats()

            res = _fit_sarimax_result_with_starts(
                mdl = mdl,
                y_arr = y.values,
                X_arr = X_arr,
                order = od,
                fit_stats = local_stats,
                fit_mode = fit_mode,
                start_override = (warm_start_by_order.get(od) if warm_start_by_order else None),
            )

            if res is None:

                continue
            
            _fit_by_order_cache_np[key] = res

            _maybe_trim_cache(
                d = _fit_by_order_cache_np
            )
            
            out[od] = res
        
        except Exception:
        
            continue

    return out


def _fit_sarimax_candidates_np(
    y: pd.Series,
    X_arr: np.ndarray,
    index: pd.DatetimeIndex,
    time_varying: bool = False,
    orders: Optional[List[Tuple[int, int, int]]] = None,
    fit_mode: str = "auto",
) -> Ensemble:
    """
    Fit a set of SARIMAX candidates and convert AICc scores to normalised weights.

    AICc
    ----
    AICc = AIC + (2 k (k + 1)) / (n − k − 1), 
    
    where k is the number of free parameters and n is the effective sample size. 
    It penalises small-sample overfitting more than AIC.

    Parameters
    ----------
    y : pandas.Series
    X_arr : numpy.ndarray
    index : pandas.DatetimeIndex
    time_varying : bool, default False
    orders : list[tuple[int, int, int]] | None

    Returns
    -------
    Ensemble
        Fitted candidates and softmax-normalised weights w ∝ exp(−0.5 (AICc − min AICc)).
    """
   
    if orders is None:

        orders = list(STACK_ORDERS)

    freq = getattr(index, "freqstr", None) or pd.infer_freq(index) or "W-SUN"

    fits, crit = [], []

    fit_attempts = 0

    fit_converged = 0

    fit_skipped = 0

    agg_stats = SARIMAXFitStats()

    rs_local = _runtime_settings()

    for od in orders:

        try:
            p, _, q = od

            min_n = max(30, 8 + 3 * (p + q) + int(np.sqrt(max(X_arr.shape[1], 1))))

            if len(y) < min_n:
          
                fit_skipped += 1
          
                continue

            fit_attempts += 1

            mdl = SARIMAX(
                y.values,
                exog = X_arr,
                order = od,
                seasonal_order = (0,0,0,0),
                trend = "c",
                enforce_stationarity = True,
                enforce_invertibility = True,
                time_varying_regression = time_varying,
                mle_regression = not time_varying,
                dates = index,
                freq = freq,
            )
          
            res = _fit_sarimax_result_with_starts(
                mdl = mdl,
                y_arr = y.values,
                X_arr = X_arr,
                order = od,
                fit_stats = agg_stats,
                fit_mode = fit_mode,
            )

            if res is None:

                fit_skipped += 1

                continue

            if (p + q) > 1 and str(fit_mode).lower() != "final":
           
                try:
           
                    lag_lb = min(12, max(4, len(y) // 10))
           
                    lb_df = acorr_ljungbox(np.asarray(res.resid, float), lags = [lag_lb], return_df = True)
           
                    lb_p = float(lb_df["lb_pvalue"].iloc[-1])
           
                    if np.isfinite(lb_p) and lb_p < float(rs_local["ljungbox_min_p"]):
               
                        fit_skipped += 1
               
                        continue
               
                except Exception:
               
                    pass

            fit_converged += 1

            fits.append(res)
        
            kparams = res.params.size
        
            n = len(res.model.endog)
        
            aicc = res.aic + (2*kparams*(kparams+1))/max(n-kparams-1, 1)
        
            crit.append(aicc)
        
        except Exception:

            fit_skipped += 1

            continue
        
    if not fits:

        od = (1, 0, 0)

        fit_attempts += 1

        try:

            mdl_fb = SARIMAX(
                y.values,
                exog = X_arr,
                order = od,
                seasonal_order = (0, 0, 0, 0),
                trend = "c",
                enforce_stationarity = False,
                enforce_invertibility = False,
                validate_specification = False,
                time_varying_regression = time_varying,
                mle_regression = not time_varying,
                dates = index,
                freq = freq,
            )

            with warnings.catch_warnings():

                warnings.filterwarnings(
                    "ignore",
                    message = "Non-stationary starting autoregressive parameters found.*",
                    category = UserWarning,
                )

                warnings.filterwarnings(
                    "ignore",
                    message = "Non-invertible starting MA parameters found.*",
                    category = UserWarning,
                )

                res_fb = mdl_fb.fit(
                    disp = False,
                    method = "lbfgs",
                    maxiter = int(rs_local["fit_maxiter_final_b"]),
                )

            if _is_sarimax_fit_usable(res_fb):

                fits = [res_fb]

                crit = [float(res_fb.aic)]

                fit_converged += 1

            else:

                fit_skipped += 1

        except Exception:

            fit_skipped += 1

    if not fits:

        raise RuntimeError("No SARIMAX fits succeeded (array path).")

    aicc = np.asarray(crit, float)
  
    w = np.exp(-0.5 * (aicc - aicc.min()))
  
    w /= w.sum()
  
    return Ensemble(
        fits = fits,
        weights = w,
        meta = {
            "attempts": int(fit_attempts),
            "converged": int(fit_converged),
            "skipped": int(fit_skipped),
            "warnings_start": int(agg_stats.warnings_start),
            "warnings_conv": int(agg_stats.warnings_conv),
            "warnings_other": int(agg_stats.warnings_other),
            "non_converged": int(agg_stats.non_converged),
        },
    )


def _fit_sarimax_candidates_cached_np(
    y: pd.Series,
    X_arr: np.ndarray,
    index: pd.DatetimeIndex,
    col_order: List[str],
    time_varying: bool = False,
    orders: Optional[List[Tuple[int, int, int]]] = None,
    fit_mode: str = "auto",
) -> Ensemble:
    """
    Return a cached SARIMAX ensemble for a training design, fitting only on cache miss.

    Cache key definition
    --------------------
    The key is:

        ("NP",
         len(y),
         last_index_timestamp,
         n_exog_columns,
         hash(tuple(col_order)),
         hash(tuple(orders)) or 0,
         hash(lower(fit_mode))).

    This key intentionally includes `fit_mode` because optimiser budgets differ between
    regular and final fits, which can change the selected likelihood optimum.

    Parameters
    ----------
    y, X_arr, index, col_order, time_varying, orders, fit_mode
        As in `_fit_sarimax_candidates_np`.

    Returns
    -------
    Ensemble

    Why this process
    ----------------
    Rolling-origin validation and tuning repeatedly request identical train-block fits.
    Memoisation avoids redundant optimisation while preserving deterministic behaviour.

    Advantages
    ----------
    - Reduces repeated SARIMAX fitting cost substantially.
   
    - Maintains coherence between fitting policy (`fit_mode`) and cached artefacts.
   
    - Preserves stable ensemble objects across repeated calls in the same run.
   
    """

    key = (
        "NP",
        len(y),
        int(index[-1].value) if len(index) else 0,
        X_arr.shape[1],
        hash(tuple(col_order)),
        hash(tuple(orders)) if orders is not None else 0,
        hash(str(fit_mode).lower()),
    )
  
    cached = _cache_get_touch(
        d = _fit_memo_np, 
        key = key
    )
 
    if cached is not None:
 
        return cached
  
    ens = _fit_sarimax_candidates_np(
        y = y, 
        X_arr = X_arr, 
        index = index,
        time_varying = time_varying,
        orders = orders,
        fit_mode = fit_mode,
    )
  
    _fit_memo_np[key] = ens
  
    _maybe_trim_cache(
        d = _fit_memo_np
    )
  
    return ens


@dataclass
class FoldDesignArtifacts:

    X_tr_arr: np.ndarray

    y_tr: pd.Series

    index: pd.DatetimeIndex

    col_order: List[str]

    X_vas: np.ndarray

    valid_y: np.ndarray


def _df_fold_signature(
    df: pd.DataFrame,
) -> Tuple[Any, ...]:
    """
    Build a compact signature describing the fold-defining state of a time-indexed frame.

    Signature components
    --------------------
    The returned tuple is:

        (n_rows, first_timestamp_ns, last_timestamp_ns, y_tail_signature),

    where `y_tail_signature` is the finite tail of up to eight `y` values rounded to
    eight decimal places.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame expected to contain the target column `y` and a datetime-like index.

    Returns
    -------
    tuple[Any, ...]
        Hashable summary used as part of fold-design cache keys.

    Why this process
    ----------------
    Fold design matrices depend on both index boundaries and recent target alignment.
    A lightweight signature provides robust cache invalidation without hashing full arrays.

    Advantages
    ----------
    - Cheap to compute inside repeated cross-validation loops.
   
    - Sensitive to key temporal changes affecting fold construction.
   
    - Hashable and deterministic, suitable for dictionary-backed memoisation.
   
    """

    idx = df.index
 
    first = int(idx[0].value) if len(idx) else 0
 
    last = int(idx[-1].value) if len(idx) else 0

    y_tail = np.asarray(df["y"].tail(8), float) if "y" in df.columns else np.asarray([], float)
 
    y_tail = y_tail[np.isfinite(y_tail)]
 
    y_sig = tuple(np.round(y_tail, 8).tolist())

    return (int(len(df)), first, last, y_sig)


def _get_fold_design_artifacts(
    df: pd.DataFrame,
    regressors: List[str],
    exog_lags: tuple[int, ...],
    fourier_k: int,
    horizon: int,
    train_end: int,
    include_jump: bool,
) -> Optional[FoldDesignArtifacts]:
    """
    Create and cache all design artefacts required for one rolling-origin fold.

    Fold construction
    -----------------
    For a fold split at `train_end` with forecast horizon `H`:

    1) Training block: `train = df[:train_end]`.
  
    2) Validation block: `valid = df[train_end : train_end + H]`.
  
    3) Build training exogenous matrix `X_tr` via `make_exog` using configured lags
       and Fourier harmonics.
  
    4) Fit `StandardScaler` on `X_tr` and transform to `X_tr_arr`.
  
    5) Build future exogenous matrix `X_vas` for validation horizon via `build_future_exog`,
       using trailing macro context rows from training history and validation macro path.

    Caching
    -------
    The function memoises the complete artefact bundle using a key composed of:

    - dataframe fold signature,
  
    - split position and horizon,
  
    - regressor set,
  
    - lag tuple,
  
    - Fourier order,
  
    - jump-inclusion flag.

    Parameters
    ----------
    df : pandas.DataFrame
        Time-ordered modelling frame containing `y` and macro regressor columns.
    regressors : list[str]
        Base macro regressor names.
    exog_lags : tuple[int, ...]
        Lag structure used for exogenous feature construction.
    fourier_k : int
        Number of Fourier harmonic pairs for seasonal representation.
    horizon : int
        Validation horizon in periods.
    train_end : int
        End index of the training window.
    include_jump : bool
        Whether to include `jump_ind` in future exogenous construction when available.

    Returns
    -------
    FoldDesignArtifacts | None
        Fully prepared artefacts, or `None` when insufficient validation rows or invalid
        design construction prevents fold evaluation.

    Why this process
    ----------------
    Rolling-origin workflows repeatedly rebuild identical train/validation matrices. Centralising
    and caching fold artefacts improves consistency and removes duplicated preprocessing logic.

    Advantages
    ----------
    - Ensures strict train/validation separation and consistent scaling.
  
    - Substantially reduces repeated preprocessing overhead in cross-validation.
  
    - Produces reusable artefacts for both stacking and mode-comparison routines.
  
    """

    key = (
        _df_fold_signature(
            df = df
        ),
        int(train_end),
        int(horizon),
        tuple(regressors),
        tuple(exog_lags),
        int(fourier_k),
        bool(include_jump),
    )

    cached = _cache_get_touch(
        d = _FOLD_DESIGN_CACHE, 
        key = key
    )
    
    if cached is not None:
    
        return cached

    train = df.iloc[:train_end]
    
    valid = df.iloc[train_end : train_end + horizon]

    if len(valid) < horizon:
    
        return None

    X_tr = make_exog(
        df = train,
        base = regressors,
        lags = exog_lags,
        add_fourier = True,
        K = fourier_k,
    )

    if len(X_tr) == 0:
      
        return None

    y_tr = train["y"].loc[X_tr.index]
  
    sc = StandardScaler().fit(X_tr.values)
  
    X_tr_arr = sc.transform(X_tr.values)
  
    col_order = list(X_tr.columns)

    max_lag = max(exog_lags) if exog_lags else 0
  
    context_rows = max_lag + 2
  
    hist_tail_vals = train[regressors].iloc[-context_rows:].values
  
    macro_path = valid[regressors].values
  
    jump = None
  
    if include_jump and "jump_ind" in valid.columns:
  
        jump = np.asarray(valid["jump_ind"].values, float)

    layout = _cache_get_touch(
        d = _COMPILED_EXOG_LAYOUT_CACHE, 
        key = (tuple(col_order), tuple(exog_lags))
    )
  
    X_vas = build_future_exog(
        hist_tail_vals = hist_tail_vals,
        macro_path = macro_path,
        col_order = col_order,
        scaler = sc,
        lags = exog_lags,
        jump = jump,
        t_start = train_end,
        layout = layout,
    )

    art = FoldDesignArtifacts(
        X_tr_arr = X_tr_arr,
        y_tr = y_tr,
        index = X_tr.index,
        col_order = col_order,
        X_vas = X_vas,
        valid_y = np.asarray(valid["y"].values, float),
    )

    _FOLD_DESIGN_CACHE[key] = art
   
    _maybe_trim_cache(
        d = _FOLD_DESIGN_CACHE
    )
    
    return art


def learn_stacking_weights(
    df: pd.DataFrame,
    regressors: List[str],
    orders: List[Tuple[int,int,int]],
    n_splits: int,
    horizon: int,
    exog_lags: tuple[int, ...] = EXOG_LAGS,
    fourier_k: int = FOURIER_K,
) -> Tuple[Dict[Tuple[int,int,int], float], List[Tuple[int,int,int]]]:
    """
    Learn non-negative stacking weights over SARIMAX orders by rolling-origin cross-validation
    on **H-step sums**.

    Procedure
    ---------
    1) Split the series into `n_splits` rolling training/validation windows with a fixed
    validation horizon H.
   
    2) For each fold and each order in `orders`, fit on the training block and forecast
    H steps ahead using the candidate exogenous future path built by `build_future_exog`.
   
    3) Record the sum of the H predictive means per model, forming a design F (fold × model).
   
    4) Solve a non-negative ridge-regularised least squares:
   
        minimise ||F w − y||_2^2 + ε ||w||_2^2,  s.t. w ≥ 0,
   
    where y are validation H-sums. Normalise w to sum to one.

    Parameters
    ----------
    df : pandas.DataFrame
    regressors : list[str]
    orders : list[tuple[int, int, int]]
    n_splits : int
    horizon : int

    Returns
    -------
    weights : dict[order, float]
    valid_orders : list[order]

    Why sum-based stacking
    ----------------------
    For many portfolio or P&L applications the H-period sum (or average) is the relevant
    target. Directly matching H-sums reduces temporal mis-alignment between models.
    Non-negative weights preserve interpretability and guard against cancellation.
    """

    N = len(df)
    
    if N <= horizon:
    
        return {}, []
    
    fold_size = (N - horizon) // (n_splits + 1)
    
    if fold_size < 1:
    
        return {}, []

    rows_F: List[Dict[Tuple[int,int,int], float]] = []
    
    y_vec: List[float] = []
    
    valid_orders = set()

    warm_starts: Dict[Tuple[int, int, int], np.ndarray] = {}

    for kfold in range(n_splits):
    
        train_end = (kfold + 1) * fold_size
       
        fold_art = _get_fold_design_artifacts(
            df = df,
            regressors = regressors,
            exog_lags = exog_lags,
            fourier_k = fourier_k,
            horizon = horizon,
            train_end = train_end,
            include_jump = True,
        )

        if fold_art is None:
     
            break

        fits = _fit_sarimax_by_orders_cached_np(
            y = fold_art.y_tr, 
            X_arr = fold_art.X_tr_arr, 
            index = fold_art.index, 
            orders = orders, 
            col_order = fold_art.col_order,
            warm_start_by_order = warm_starts,
        )
        
        valid_orders |= set(fits.keys())

        for od, res in fits.items():
        
            try:
        
                warm_starts[od] = np.asarray(res.params, float)
        
            except Exception:
        
                continue

        row = {}
      
        for od in fits.keys():
      
            f = fits[od].get_forecast(steps = horizon, exog = fold_art.X_vas)
      
            mu_k = f.predicted_mean
      
            row[od] = float(np.sum(mu_k))
      
        rows_F.append(row)
      
        y_vec.append(float(np.sum(fold_art.valid_y)))

    if not rows_F or not valid_orders:
       
        return {}, []

    valid_orders = sorted(valid_orders)
   
    F = np.array([[r.get(od, np.nan) for od in valid_orders] for r in rows_F], dtype = float)
    
    mask_rows = ~np.isnan(F).any(axis = 1)
   
    F = F[mask_rows]
   
    y_arr = np.array(y_vec, dtype = float)[mask_rows]
   
    if F.shape[0] < 1 or F.shape[1] < 1:
   
        return {}, []

    G = F.T @ F + RIDGE_EPS * np.eye(F.shape[1])
   
    b = F.T @ y_arr
   
    try:
   
        w = np.linalg.solve(G, b)
   
    except np.linalg.LinAlgError:
   
        w = np.linalg.lstsq(G, b, rcond = None)[0]
   
    w = np.clip(w, 0.0, None)
   
    s = float(w.sum())
   
    if s <= 0:
   
        return {}, []
   
    w = w / s
   
    return {od: float(wi) for od, wi in zip(valid_orders, w)}, valid_orders


def residual_diagnostics(
    resid: np.ndarray,
    lags: int = 12
) -> Dict[str, float]:
    """
    Compute simple residual diagnostics: serial correlation, conditional heteroskedasticity,
    and kurtosis.

    Metrics
    -------
    - Ljung–Box p-value at lag L for r_t and r_t^2.
    - Engle's ARCH LM test p-value.
    - Excess kurtosis estimate.

    Parameters
    ----------
    resid : numpy.ndarray
    lags : int, default 12

    Returns
    -------
    dict
        {"lb_p", "lb2_p", "arch_p", "kurt"}.

    Use
    ---
    Informs the Student-t degrees-of-freedom heuristic and flags model mis-specification.
    """

    r = pd.Series(resid).dropna().values

    if r.size < max(20, lags + 5):

        return {
            "lb_p": np.nan, 
            "lb2_p": np.nan, 
            "arch_p": np.nan, 
            "kurt": np.nan
        }

    lb = acorr_ljungbox(
        x = r, 
        lags = [lags], 
        return_df = True
    )

    lb2 = acorr_ljungbox(
        r ** 2, 
        lags = [lags], 
        return_df = True
    )

    _, arch_lm_p, _, _ = het_arch(
        resid = r,
        nlags = min(lags, max(1, r.size // 10))
    )

    kurt = pd.Series(r).kurtosis()  

    return {
        "lb_p": float(lb["lb_pvalue"].iloc[-1]),
        "lb2_p": float(lb2["lb_pvalue"].iloc[-1]),
        "arch_p": float(arch_lm_p),
        "kurt": float(kurt),
    }


def choose_student_df_from_diag(
    diag: Dict[str, float]
) -> int:
    """
    Heuristic mapping from residual diagnostics to Student-t degrees of freedom for tail thickening.

    Rules of thumb
    --------------
    - Strong ARCH (p < 0.05): prefer lower df (5–6).
    - Near-Gaussian excess kurtosis (≤ 0.5): higher df (≈ 12).
    - Moderate excess kurtosis (≤ 2): medium df (≈ 8).
    - Otherwise: df = 5.

    Parameters
    ----------
    diag : dict

    Returns
    -------
    int
    """

    kurt = diag.get("kurt", np.nan)

    arch_p = diag.get("arch_p", np.nan)

    if not np.isfinite(kurt):

        return max(6, T_DOF)

    if np.isfinite(arch_p) and arch_p < 0.05:

        return 6 if kurt <= 5 else 5

    if kurt <= 0.5:

        return 12

    if kurt <= 2.0:

        return 8

    return 5


def rolling_origin_cv_rmse_return_sum(
    df: pd.DataFrame,
    regressors: List[str],
    n_splits: int,
    horizon: int,
    y_full: Optional[pd.Series] = None,
    X_full_arr: Optional[np.ndarray] = None,
    col_order: Optional[List[str]] = None,
    ens_full: Optional[Ensemble] = None,
    exog_lags: tuple[int, ...] = EXOG_LAGS,
    fourier_k: int = FOURIER_K,
    orders: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate forecast variance for H-step sums via rolling-origin cross-validation and
    collect residuals from a full-data ensemble for bootstrap calibration.

    Outputs
    -------
    - `resid`: residuals of the best-weighted full-data SARIMAX fit.
    
    - `rs_std`: standard deviation of rolling H-step sums of residuals (or scaled proxy),
    used as a bootstrap variance proxy.
    
    - `v_fore_mean`: average model-based forecast variance of H-sums across folds, computed
    from the mixture variance of the ensemble.

    Procedure
    ---------
    For each fold:
    
    1) Build training exogenous matrix `X_tr` and fit an ensemble of SARIMAX orders.
    
    2) Construct a fold-specific future exogenous path (`build_future_exog`) from the last
    `context_rows` regressors and the validation macro slice.
    
    3) Obtain H-step predictive means and per-step predictive variances per model.
    
    4) Form the ensemble mixture mean and variance via:
    
        μ_mix_t = Σ_k w_k μ_{k,t},
    
        Var_mix_t = Σ_k w_k [Var_{k,t} + (μ_{k,t} − μ_mix_t)^2],
    
    then sum across t = 1..H and record Σ_t Var_mix_t as the fold’s forecast variance.

    Finally, if a full-data ensemble is not supplied, fit one on all available data, take
    its residuals, and compute the standard deviation of the rolling H-sum.

    Parameters
    ----------
    df : pandas.DataFrame
    regressors : list[str]
    n_splits : int
    horizon : int
    y_full : pandas.Series, optional
    X_full_arr : numpy.ndarray, optional
    col_order : list[str], optional
    ens_full : Ensemble, optional

    Returns
    -------
    resid : numpy.ndarray
    rs_std_arr : numpy.ndarray, shape (1,)
    v_fore_mean : float

    Why both model and bootstrap variance
    -------------------------------------
    Model variance (from SARIMAX) captures parametric and innovation uncertainty
    conditional on the specification; bootstrap variance captures residual serial
    dependence and model misspecification. A convex blend is calibrated subsequently.
    """

    N = len(df)
   
    if N <= horizon:

        return np.array([]), np.array([np.nan]), np.nan

    fold_size = (N - horizon) // (n_splits + 1)
   
    if fold_size < 1:

        return np.array([]), np.array([np.nan]), np.nan

    v_fore_list: List[float] = []

    for kfold in range(n_splits):

        train_end = (kfold + 1) * fold_size

        fold_art = _get_fold_design_artifacts(
            df = df,
            regressors = regressors,
            exog_lags = exog_lags,
            fourier_k = fourier_k,
            horizon = horizon,
            train_end = train_end,
            include_jump = True,
        )

        if fold_art is None:
         
            break

        try:
            
            ens = _fit_sarimax_candidates_cached_np(
                y = fold_art.y_tr, 
                X_arr = fold_art.X_tr_arr,
                index = fold_art.index, 
                col_order = fold_art.col_order,
                orders = orders,
            )
            
        except Exception:
           
            continue

        mu_stack, var_stack = [], []
        
        for res in ens.fits:
        
            f = res.get_forecast(steps = horizon, exog = fold_art.X_vas)
        
            mu_k = f.predicted_mean
        
            var_k = np.asarray(f.var_pred_mean)
        
            mu_stack.append(mu_k)
        
            var_stack.append(var_k)

        mu_stack = np.asarray(mu_stack)                 
      
        var_stack = np.asarray(var_stack)               
      
        wv = np.asarray(ens.weights).reshape(-1, 1)      
      
        mu_mix = (wv * mu_stack).sum(axis = 0)            
      
        var_mix = (wv * (var_stack + (mu_stack - mu_mix) ** 2)).sum(axis = 0)  
      
        v_fore_list.append(float(np.sum(np.maximum(var_mix, 1e-12))))

    v_fore_mean = float(np.mean(v_fore_list)) if v_fore_list else np.nan

    if ens_full is None:

        if y_full is None or X_full_arr is None or col_order is None:

            X_full = make_exog(
                df = df,
                base = regressors,
                lags = exog_lags,
                add_fourier = True,
                K = fourier_k,
            )

            y_full = df["y"].loc[X_full.index]

            sc_full = StandardScaler().fit(X_full.values)

            X_full_arr = sc_full.transform(X_full.values)

            col_order = list(X_full.columns)

        try:
            ens_full = _fit_sarimax_candidates_cached_np(
                y = y_full,
                X_arr = X_full_arr,
                index = y_full.index,
                col_order = col_order,
                orders = orders,
            )
        except Exception:
            ens_full = None

    if ens_full is None or len(ens_full.fits) == 0:

        return np.array([]), np.array([np.nan]), v_fore_mean

    best = ens_full.fits[int(np.argmax(ens_full.weights))]
    
    resid = pd.Series(best.resid, index=y_full.index).dropna().values

    if len(resid) >= horizon:
        
        rs = pd.Series(resid).rolling(window = horizon).sum().dropna().values
        
        rs_std = np.std(rs, ddof=1) if len(rs) > 1 else np.std(rs)
  
    else:
  
        rs_std = np.std(resid) * np.sqrt(horizon)

    return resid, np.array([float(rs_std)]), v_fore_mean


def calibrate_alpha_from_cv(
    v_fore_mean: float, 
    rs_std: float,
    grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
) -> float:
    """
    Choose the convex blending weight α between model-based noise and bootstrap noise
    to minimise expected squared H-sum error.

    Objective
    ---------
    For V_fore = mean forecast variance of H-sums and V_boot = rs_std^2:

        objective(α) = α^2 V_fore + (1 − α)^2 V_boot,
        α* = argmin_{α ∈ [0, 1]} objective(α),

    evaluated on a small grid.

    Parameters
    ----------
    v_fore_mean : float
    rs_std : float
    grid : tuple[float, ...], default (0.0, 0.25, 0.5, 0.75, 1.0)

    Returns
    -------
    float
        α in [0, 1].

    Rationale
    ---------
    Balances parametric uncertainty and model misspecification in a transparent manner.
    """
  
    if not np.isfinite(v_fore_mean) or not np.isfinite(rs_std):
  
        return 0.5
  
    V_fore = max(float(v_fore_mean), 1e-12)
  
    V_boot = max(float(rs_std) ** 2, 1e-12)
  
    best_a = 0.5
    
    best_obj = float("inf")
  
    for a in grid:
  
        obj = a * a * V_fore + (1.0 - a) * (1.0 - a) * V_boot
  
        if obj < best_obj:
  
            best_obj = obj
            
            best_a = a
  
    return float(np.clip(best_a, 0.0, 1.0))


def rolling_origin_mode_comparison(
    df: pd.DataFrame,
    regressors: List[str],
    horizon: int,
    n_splits: int = CV_SPLITS,
    draws: int = 400,
    seed: int = RNG_SEED,
    exog_lags: tuple[int, ...] = EXOG_LAGS,
    fourier_k: int = FOURIER_K,
    orders: Optional[List[Tuple[int, int, int]]] = None,
    hm_min_obs: int = HM_MIN_OBS,
    hm_clip_skew: float = HM_CLIP_SKEW,
    hm_clip_excess_kurt: float = HM_CLIP_EXCESS_KURT,
) -> Dict[str, float]:
    """
    Compare Gaussian vs higher-moment innovations on rolling-origin folds.

    Metrics are computed on H-step return sums:
    - RMSE of predictive mean.
    - Approximate log score from Monte Carlo kernel density.
    - 90% interval coverage.
    - Tail exceedance rates at 5% and 95%.
    """

    cache_key = _mode_comparison_cache_key(
        df = df,
        regressors = regressors,
        horizon = horizon,
        n_splits = n_splits,
        draws = draws,
        seed = seed,
        exog_lags = exog_lags,
        fourier_k = fourier_k,
        orders = orders,
        hm_min_obs = hm_min_obs,
        hm_clip_skew = hm_clip_skew,
        hm_clip_excess_kurt = hm_clip_excess_kurt,
    )
    cached_cmp = _cache_get_touch(
        d = _MODE_COMPARISON_CACHE, 
        key = cache_key
    )
    
    if cached_cmp is not None:
    
        return dict(cached_cmp)

    rng = np.random.default_rng(seed)

    N = len(df)

    if N <= horizon:

        _MODE_COMPARISON_CACHE[cache_key] = {}
      
        _maybe_trim_cache(
            d = _MODE_COMPARISON_CACHE
        )
      
        return {}

    fold_size = (N - horizon) // (n_splits + 1)

    if fold_size < 1:

        _MODE_COMPARISON_CACHE[cache_key] = {}
     
        _maybe_trim_cache(
            d = _MODE_COMPARISON_CACHE
        )
     
        return {}

    stats = {
        "gauss_sq": [],
        "hm_sq": [],
        "gauss_log": [],
        "hm_log": [],
        "gauss_wis90": [],
        "hm_wis90": [],
        "gauss_cov90": [],
        "hm_cov90": [],
        "gauss_low_exc": [],
        "hm_low_exc": [],
        "gauss_high_exc": [],
        "hm_high_exc": [],
    }

    for kfold in range(n_splits):

        train_end = (kfold + 1) * fold_size

        fold_art = _get_fold_design_artifacts(
            df = df,
            regressors = regressors,
            exog_lags = exog_lags,
            fourier_k = fourier_k,
            horizon = horizon,
            train_end = train_end,
            include_jump = False,
        )

        if fold_art is None:
            break

        if len(fold_art.y_tr) < 30:

            continue

        try:
            
            ens = _fit_sarimax_candidates_cached_np(
                y = fold_art.y_tr,
                X_arr = fold_art.X_tr_arr,
                index = fold_art.index,
                col_order = fold_art.col_order,
                orders = orders,
            )
        except Exception:
            
            continue

        mu_stack: List[np.ndarray] = []

        var_stack: List[np.ndarray] = []

        for res in ens.fits:

            f = res.get_forecast(
                steps = horizon, 
                exog = fold_art.X_vas
            )

            mu_stack.append(np.asarray(f.predicted_mean, float))

            var_stack.append(np.asarray(f.var_pred_mean, float))

        mu_arr = np.asarray(mu_stack, float)

        var_arr = np.asarray(var_stack, float)

        wv = np.asarray(ens.weights, float).reshape(-1, 1)

        mu_mix = (wv * mu_arr).sum(axis = 0)

        var_mix = (wv * (var_arr + (mu_arr - mu_mix) ** 2)).sum(axis = 0)

        var_mix = np.maximum(var_mix, 1e-12)

        pred_sum = float(np.sum(mu_mix))

        actual_sum = float(np.sum(fold_art.valid_y))

        resid_best = np.asarray(ens.fits[int(np.argmax(ens.weights))].resid, float)

        resid_best = resid_best[np.isfinite(resid_best)]

        resid_best = resid_best - np.mean(resid_best) if resid_best.size else np.array([0.0])

        resid_sd = np.std(resid_best) + 1e-9

        hm_min_obs_old = HM_MIN_OBS
        
        hm_clip_skew_old = HM_CLIP_SKEW
        
        hm_clip_exk_old = HM_CLIP_EXCESS_KURT
        
        try:
        
            globals()["HM_MIN_OBS"] = int(hm_min_obs)
        
            globals()["HM_CLIP_SKEW"] = float(hm_clip_skew)
        
            globals()["HM_CLIP_EXCESS_KURT"] = float(hm_clip_excess_kurt)
        
            hm_params = fit_higher_moment_params(
                resid_standardized = resid_best / resid_sd
            )
        
        finally:
        
            globals()["HM_MIN_OBS"] = hm_min_obs_old
        
            globals()["HM_CLIP_SKEW"] = hm_clip_skew_old
        
            globals()["HM_CLIP_EXCESS_KURT"] = hm_clip_exk_old

        sd_step = np.sqrt(var_mix)

        z_g = rng.standard_normal((draws, horizon))

        z_h = draw_higher_moment_z(hm_params, draws * horizon, rng).reshape(draws, horizon)

        s_g = pred_sum + np.sum(sd_step[None, :] * z_g, axis=1)

        s_h = pred_sum + np.sum(sd_step[None, :] * z_h, axis=1)

        qg = np.quantile(s_g, [0.05, 0.95])

        qh = np.quantile(s_h, [0.05, 0.95])

        bw_g = max(np.std(s_g, ddof=1), 1e-6) * max(draws, 2) ** (-1.0 / 5.0)

        bw_h = max(np.std(s_h, ddof=1), 1e-6) * max(draws, 2) ** (-1.0 / 5.0)

        den_g = np.mean(np.exp(-0.5 * ((actual_sum - s_g) / bw_g) ** 2) / (np.sqrt(two_pi) * bw_g))

        den_h = np.mean(np.exp(-0.5 * ((actual_sum - s_h) / bw_h) ** 2) / (np.sqrt(two_pi) * bw_h))

        pred_g = float(np.mean(s_g))

        pred_h = float(np.mean(s_h))

        stats["gauss_sq"].append((pred_g - actual_sum) ** 2)
       
        stats["hm_sq"].append((pred_h - actual_sum) ** 2)
       
        stats["gauss_log"].append(np.log(max(float(den_g), 1e-300)))
       
        stats["hm_log"].append(np.log(max(float(den_h), 1e-300)))
       
        wis_g = float((qg[1] - qg[0]) + 20.0 * max(qg[0] - actual_sum, 0.0) + 20.0 * max(actual_sum - qg[1], 0.0))
       
        wis_h = float((qh[1] - qh[0]) + 20.0 * max(qh[0] - actual_sum, 0.0) + 20.0 * max(actual_sum - qh[1], 0.0))
       
        stats["gauss_wis90"].append(wis_g)
       
        stats["hm_wis90"].append(wis_h)
       
        stats["gauss_cov90"].append(float(qg[0] <= actual_sum <= qg[1]))
       
        stats["hm_cov90"].append(float(qh[0] <= actual_sum <= qh[1]))
       
        stats["gauss_low_exc"].append(float(actual_sum < qg[0]))
       
        stats["hm_low_exc"].append(float(actual_sum < qh[0]))
       
        stats["gauss_high_exc"].append(float(actual_sum > qg[1]))
       
        stats["hm_high_exc"].append(float(actual_sum > qh[1]))

    if len(stats["gauss_sq"]) == 0:

        _MODE_COMPARISON_CACHE[cache_key] = {}
       
        _maybe_trim_cache(
            d = _MODE_COMPARISON_CACHE
        )
       
        return {}

    out = {
        "gauss_rmse": float(np.sqrt(np.mean(stats["gauss_sq"]))),
        "hm_rmse": float(np.sqrt(np.mean(stats["hm_sq"]))),
        "gauss_logscore": float(np.mean(stats["gauss_log"])),
        "hm_logscore": float(np.mean(stats["hm_log"])),
        "gauss_wis90": float(np.mean(stats["gauss_wis90"])),
        "hm_wis90": float(np.mean(stats["hm_wis90"])),
        "gauss_cov90": float(np.mean(stats["gauss_cov90"])),
        "hm_cov90": float(np.mean(stats["hm_cov90"])),
        "gauss_low_exc": float(np.mean(stats["gauss_low_exc"])),
        "hm_low_exc": float(np.mean(stats["hm_low_exc"])),
        "gauss_high_exc": float(np.mean(stats["gauss_high_exc"])),
        "hm_high_exc": float(np.mean(stats["hm_high_exc"])),
    }

    _MODE_COMPARISON_CACHE[cache_key] = out
   
    _maybe_trim_cache(
        d = _MODE_COMPARISON_CACHE
    )

    return out


def _fit_student_df_mle(
    resid: np.ndarray,
    lo: float = 3.5,
    hi: float = 40.0,
) -> int:
    """
    Estimate Student-t degrees of freedom by maximum likelihood on standardised residuals.

    Estimation workflow
    -------------------
    Let raw residuals be `r_t`. The function:

    1) Drops non-finite values.
  
    2) Requires at least 40 observations; otherwise returns clipped default `T_DOF`.
  
    3) Centres and scales:

           z_t = (r_t - mean(r)) / sd(r).

    4) Fits a Student-t distribution with fixed location (`loc = 0`) via MLE:

           z_t ~ t_nu(0, s),

       and extracts `nu = df_hat`.
  
    5) Clips `nu` to `[lo, hi]`, rounds to nearest integer, and returns it.

    If MLE fails, a diagnostic fallback is used:

        nu_fallback = choose_student_df_from_diag(residual_diagnostics(z)).

    Parameters
    ----------
    resid : numpy.ndarray
        Residual series used to infer tail thickness.
    lo : float, default 3.5
        Lower admissible bound for degrees of freedom.
    hi : float, default 40.0
        Upper admissible bound for degrees of freedom.

    Returns
    -------
    int
        Estimated and bounded degrees of freedom.

    Why this process
    ----------------
    Tail heaviness materially affects extreme return simulation. Data-driven estimation of
    Student-t `nu` provides a direct control on kurtosis, while bounded fallback logic
    prevents unstable estimates under small or degenerate samples.

    Advantages
    ----------
    - Adapts heavy-tail intensity to observed residual behaviour.
 
    - Preserves numerical robustness through clipping and diagnostic fallback.
 
    - Produces an interpretable scalar hyperparameter for subsequent t-scaling.
 
    """

    x = np.asarray(resid, float)

    x = x[np.isfinite(x)]

    if x.size < 40:

        return int(np.clip(T_DOF, lo, hi))

    x = x - np.mean(x)

    sd = np.std(x, ddof = 1) if x.size > 1 else np.std(x)

    if (not np.isfinite(sd)) or sd <= 1e-12:

        return int(np.clip(T_DOF, lo, hi))

    x = x / sd

    try:

        df_hat, _, _ = student_t.fit(x, floc=0.0)

        if not np.isfinite(df_hat):

            raise ValueError("non-finite df")

        return int(np.round(np.clip(df_hat, lo, hi)))

    except Exception:

        return int(np.clip(choose_student_df_from_diag(
            diag = residual_diagnostics(
                resid = x
            )
        ), lo, hi))


def _auto_stationary_block_length(
    resid: np.ndarray,
    max_lag: int = 52,
) -> int:
    """
    Approximate Politis-White automatic stationary bootstrap block length.
    """

    x = np.asarray(resid, float)
  
    x = x[np.isfinite(x)]

    n = int(x.size)

    if n < 30:
  
        return int(np.clip(RESID_BLOCK, 2, 12))

    x = x - np.mean(x)
  
    L = int(min(max_lag, max(5, n // 4)))
  
    acf = np.array([_sample_autocorr(
        x = x,
        lag = l
    ) for l in range(1, L + 1)], float)

    if not np.all(np.isfinite(acf)):
     
        return int(np.clip(RESID_BLOCK, 2, 24))

    sig = 2.0 / np.sqrt(n)
    
    m = L
    
    for i, v in enumerate(acf, start=1):
    
        if abs(v) < sig:
    
            m = i
    
            break

    g = 0.0
    
    for l in range(1, m + 1):
    
        w = 1.0 - l / (m + 1.0)
    
        g += 2.0 * w * acf[l - 1]

    g = float(max(g, 1e-6))

    b = ((2.0 * (g ** 2)) ** (1.0 / 3.0)) * (n ** (1.0 / 3.0))

    if not np.isfinite(b):

        b = float(RESID_BLOCK)

    return int(np.clip(np.round(b), 2, min(52, max(4, n // 3))))


def _select_jump_threshold_evt(
    resid: np.ndarray,
    min_exceed: int,
) -> Tuple[float, int]:
    """
    Select an EVT jump quantile threshold by balancing tail fit quality and stability.

    Method
    ------
    Candidate quantiles `q` are scanned on a grid from `0.90` to `0.995`. For each `q`:

    1) Fit peaks-over-threshold GPD on `|resid|` via `fit_gpd_tail`.
    
    2) Compute exceedances `e = |r| - tau_q`, where `tau_q` is the `q`-quantile threshold.
    
    3) Compute a per-exceedance negative log-likelihood proxy under fitted `(xi, beta)`:

           nll = mean( log(beta) + (1/xi + 1) * log(1 + xi * e / beta) ),

       with numerical floors on `xi` and `beta`.
    
    4) Score each candidate with:

           obj = instability + 0.20 * nll + 0.001 * exceedance_balance,

       where:
   
       - `instability = |xi - median(xi)| + 0.5 * |log(beta) - median(log(beta))|`,
   
       - `exceedance_balance = |n_exceed - 1.5 * min_exceed|`.

    The quantile with minimum objective is selected. If candidates are insufficient,
    fallback is `(JUMP_Q, min_exceed)`.

    Parameters
    ----------
    resid : numpy.ndarray
        Residual series used for jump-threshold calibration.
    min_exceed : int
        Minimum exceedance count required for stable GPD estimation.

    Returns
    -------
    tuple[float, int]
        Selected quantile threshold and the unchanged exceedance floor.

    Why this process
    ----------------
    EVT tail fitting is sensitive to threshold choice. Extremely low thresholds violate
    asymptotic assumptions, while very high thresholds leave too few exceedances. The
    composite objective explicitly trades off fit quality and parameter stability.

    Advantages
    ----------
    - More robust than a fixed quantile under varying sample sizes.
   
    - Encourages stable GPD parameters across nearby thresholds.
   
    - Preserves sufficient tail sample mass for downstream jump simulation.
   
    """

    x = np.asarray(resid, float)
   
    x = x[np.isfinite(x)]

    if x.size < max(120, 2 * min_exceed):
   
        return float(JUMP_Q), int(min_exceed)

    q_grid = np.linspace(0.90, 0.995, 20)
   
    cand: List[Dict[str, float]] = []

    for qv in q_grid:
   
        fit = fit_gpd_tail(
            resid = x,
            q = float(qv),
            min_exceed = int(min_exceed),
            min_obs = 80,
        )
    
        if fit is None:
     
            continue
     
        thr = float(fit["thr"])
     
        exc = np.abs(x[np.abs(x) > thr]) - thr
     
        if exc.size < min_exceed:
     
            continue
     
        xi = float(fit["xi"])
     
        beta = float(max(fit["beta"], 1e-12))
     
        z = np.maximum(exc, 1e-12) / beta
     
        nll = float(np.mean(np.log(beta) + (1.0 / max(xi, 1e-8) + 1.0) * np.log1p(max(xi, 1e-8) * z)))
     
        cand.append({"q": float(qv), "xi": xi, "beta": beta, "nll": nll, "exc": float(exc.size)})

    if not cand:
     
        return float(JUMP_Q), int(min_exceed)

    xi_med = float(np.median([c["xi"] for c in cand]))
 
    logb_med = float(np.median([np.log(c["beta"]) for c in cand]))

    best_q = float(cand[0]["q"])
 
    best_obj = float("inf")

    for c in cand:
 
        instability = abs(c["xi"] - xi_med) + 0.5 * abs(np.log(c["beta"]) - logb_med)
 
        tail_pen = 0.20 * c["nll"]
 
        balance = 0.001 * abs(c["exc"] - min_exceed * 1.5)
 
        obj = instability + tail_pen + balance
 
        if obj < best_obj:
 
            best_obj = obj
 
            best_q = float(c["q"])

    return best_q, int(min_exceed)


def _estimate_oas_shrinkage(
    resid: np.ndarray,
    horizon: int,
) -> float:
    """
    Estimate dependence shrinkage intensity using OAS on rolling residual windows.

    Construction
    ------------
    Let `x_t` be residuals and `H = clip(horizon, 2, 12)`. Form overlapping windows:

        W_t = (x_t, x_{t+1}, ..., x_{t+H-1}),

    producing matrix `W` of shape `(n_windows, H)`. Fit Oracle Approximating Shrinkage:

        Sigma_hat = (1 - alpha) * S + alpha * mu * I,

    where `S` is sample covariance, `mu = trace(S) / H`, and `alpha` is the OAS
    shrinkage coefficient estimated by `sklearn.covariance.OAS`.

    The returned value is:

        clip(alpha, 0.01, 0.95).

    If data are insufficient or fitting fails, fallback is `COPULA_SHRINK`.

    Parameters
    ----------
    resid : numpy.ndarray
        Residual sequence used to infer horizon-wise dependence regularisation.
    horizon : int
        Forecast horizon requested by the simulation.

    Returns
    -------
    float
        Shrinkage intensity used later for correlation regularisation.

    Why this process
    ----------------
    Horizon covariance estimation from finite samples is noisy, especially when effective
    dimension grows with horizon. OAS provides data-adaptive regularisation with low
    estimation variance.

    Advantages
    ----------
    - Automatic, sample-size-aware shrinkage intensity.
  
    - Better-conditioned dependence matrices for copula simulation.
  
    - Reduced overfitting to noisy off-diagonal sample correlations.
  
    """

    x = np.asarray(resid, float)
    
    x = x[np.isfinite(x)]

    H = int(max(2, min(horizon, 12)))

    if x.size < (H + 20):
    
        return float(COPULA_SHRINK)

    try:
    
        W = np.lib.stride_tricks.sliding_window_view(x, window_shape = H)
        if W.ndim != 2 or W.shape[0] < 20:
    
            return float(COPULA_SHRINK)
    
        oas = OAS(store_precision = False, assume_centered = True).fit(W)
    
        sh = float(getattr(oas, "shrinkage_", COPULA_SHRINK))
    
        return float(np.clip(sh, 0.01, 0.95))
    
    except Exception:
    
        return float(COPULA_SHRINK)


def _quantile_ci_half_widths(
    x: np.ndarray,
    qs: np.ndarray,
) -> np.ndarray:
    """
    Kernel-based asymptotic CI half-widths for multiple quantiles in one pass.
    """

    a = np.asarray(x, float)
  
    a = a[np.isfinite(a)]

    n = int(a.size)

    qv = np.asarray(qs, float).reshape(-1)

    if n < 80:
        
        return np.full(qv.shape, float("inf"), dtype = float)

    sd = float(np.std(a, ddof=1))

    if (not np.isfinite(sd)) or sd <= 1e-12:
   
        return np.zeros(qv.shape, dtype = float)

    qq = np.quantile(a, qv)

    bw = max(sd * n ** (-1.0 / 5.0), 1e-6)
 
    z = (qq[:, None] - a[None, :]) / bw
 
    pdf = np.mean(np.exp(-0.5 * z * z) / (np.sqrt(two_pi) * bw), axis=1)
 
    pdf = np.maximum(pdf, 1e-6)
 
    se = np.sqrt(np.maximum(qv * (1.0 - qv), 1e-8) / n) / pdf
 
    return 1.96 * se


def _candidate_orders_from_data(
    n_obs: int,
    y: Optional[np.ndarray] = None,
    max_order: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Build an ARMA order candidate set from sample size.
    """

    max_pq = 3 if n_obs >= 220 else (2 if n_obs >= 120 else 1)

    if max_order is not None:
 
        max_pq = int(min(max_pq, max(1, int(max_order))))

    out: List[Tuple[int, int, int]] = [(1, 0, 0), (0, 0, 1), (1, 0, 1)]

    sig_lags: set[int] = set()

    if y is not None and np.asarray(y).size >= 40:
 
        yy = np.asarray(y, float)
 
        yy = yy[np.isfinite(yy)]
 
        if yy.size >= 40:
 
            thresh = float(1.96 / np.sqrt(yy.size))
 
            for l in range(1, max_pq + 1):
 
                if abs(_sample_autocorr(
                    x = yy, 
                    lag = l
                )) > thresh:
 
                    sig_lags.add(l)
 
            try:
          
                pacf_vals = np.asarray(pacf(yy, nlags=max_pq, method="ywm"), float)
          
                for l in range(1, min(max_pq, pacf_vals.size - 1) + 1):
          
                    if np.isfinite(pacf_vals[l]) and abs(float(pacf_vals[l])) > thresh:
          
                        sig_lags.add(l)
          
            except Exception:
          
                pass

    for p in range(max_pq + 1):
     
        for q in range(max_pq + 1):
     
            if p == 0 and q == 0:
     
                continue
     
            if p + q > max_pq + 1:
     
                continue
     
            if sig_lags and (p not in sig_lags) and (q not in sig_lags) and (p + q > 1):
     
                continue
     
            out.append((p, 0, q))

    uniq = sorted(set(out), key=lambda z: (z[0] + z[2], z[0], z[2]))

    return uniq


def fit_hyperparams_from_data(
    df_tk: pd.DataFrame,
    horizon: int,
    seed: int = RNG_SEED,
    trace_out: Optional[Dict[str, Any]] = None,
) -> FittedHyperparams:
    """
    Data-driven hyperparameter fitting for a ticker.
    """

    t0 = time.time()

    rs = _runtime_settings()

    rng = np.random.default_rng(seed)

    y = np.asarray(df_tk["y"].dropna(), float)

    T = int(y.size)

    if T == 0:
        hp0 = FittedHyperparams(
            stack_orders = list(STACK_ORDERS),
            exog_lags = tuple(EXOG_LAGS),
            fourier_k = int(FOURIER_K),
            cv_splits = int(np.clip(CV_SPLITS, 2, int(rs["max_cv_splits"]))),
            resid_block_len = int(RESID_BLOCK),
            student_df = int(T_DOF),
            jump_q = float(JUMP_Q),
            gpd_min_exceed = 30,
            hm_min_obs = int(HM_MIN_OBS),
            hm_clip_skew = float(HM_CLIP_SKEW),
            hm_clip_excess_kurt = float(HM_CLIP_EXCESS_KURT),
            copula_shrink = float(COPULA_SHRINK),
            model_cov_reps = int(rs["cov_reps_min"]),
            bvar_p = int(BVAR_P),
            mn_lambdas = (MN_LAMBDA1, MN_LAMBDA2, MN_LAMBDA3, MN_LAMBDA4),
            niw_nu0 = int(NIW_NU0),
            niw_s0_scale = float(NIW_S0_SCALE),
            n_sims_required = int(rs["n_sims_min"]),
            hm_enabled = True,
            hm_prior_weight = 0.5,
        )
       
        if trace_out is not None:
       
            trace_out.update(
                asdict(
                    HyperparamSearchTrace(
                        selected = {"reason": "empty_series"},
                        objective = float("nan"),
                        candidates_evaluated = 0,
                        elapsed_sec = float(time.time() - t0),
                    )
                )
            )
        return hp0

    cv_splits = int(np.clip((T - horizon) // max(1, horizon), 2, int(rs["max_cv_splits"])))

    opt = _optimization_profile()
   
    tune_deadline = t0 + float(max(5.0, opt.max_tune_seconds))
   
    max_evals = int(max(4, opt.max_fits))

    order_candidates = _candidate_orders_from_data(
        n_obs = T,
        y = y,
        max_order = int(rs["max_order"]),
    )

    max_lag = int(np.clip(np.sqrt(max(T, 4)) // 2, 2, 6))

    lag_candidates = [tuple(range(1, L + 1)) for L in range(1, max_lag + 1)]

    k_cap = int(np.clip(rs.get("fourier_k_cap", 5), 0, 8))
 
    k_max = int(np.clip(T // 104, 0, k_cap))

    fourier_candidates = list(range(0, k_max + 1))

    candidate_pairs = [(tuple(lag_set), int(kf)) for lag_set in lag_candidates for kf in fourier_candidates]

    if candidate_pairs:
 
        sig_lags: set[int] = set()
 
        if T >= 40:
 
            thr = float(1.96 / np.sqrt(max(T, 1)))
 
            for l in range(1, min(max_lag, 6) + 1):
 
                if abs(_sample_autocorr(
                    x = y, 
                    lag = l
                )) > thr:
 
                    sig_lags.add(l)
            try:
   
                pacf_vals = np.asarray(pacf(y, nlags=min(max_lag, 6), method="ywm"), float)
   
                for l in range(1, min(max_lag, pacf_vals.size - 1) + 1):
   
                    if np.isfinite(pacf_vals[l]) and abs(float(pacf_vals[l])) > thr:
   
                        sig_lags.add(l)
   
            except Exception:
   
                pass


        def _pair_score(
            pair: Tuple[Tuple[int, ...], int]
        ) -> float:
        
            lag_set, kf = pair
        
            overlap = len(sig_lags.intersection(lag_set)) if sig_lags else 0
         
            return (
                2.0 * overlap
                - 0.35 * len(lag_set)
                - 0.40 * int(kf)
            )

        max_screen = int(max(8, max_evals * 2))
        
        if len(candidate_pairs) > max_screen:
        
            candidate_pairs = sorted(candidate_pairs, key=_pair_score, reverse=True)[:max_screen]

    best_score = -np.inf
    
    best_lags = tuple(EXOG_LAGS)
    
    best_k = int(FOURIER_K)
    
    best_cmp: Dict[str, float] = {}

    draws_tune = int(rs["cv_draws_tune"])

    n_eval = 0


    def _cmp_objective(
        cmp: Dict[str, float]
    ) -> float:
    
        if not cmp:
    
            return -np.inf
    
        cov_pen = abs(float(cmp.get("hm_cov90", 0.0)) - 0.90)
    
        return (
            float(cmp.get("hm_logscore", -np.inf))
            - 0.03 * float(cmp.get("hm_wis90", 0.0))
            - 4.0 * cov_pen
        )

    stage_pairs = list(candidate_pairs)
    
    if opt.enable_two_stage_search and len(candidate_pairs) > 6:
    
        coarse_draws = int(max(40, draws_tune // 2))
    
        coarse_splits = int(max(2, min(cv_splits, 3)))
    
        coarse_rank: List[Tuple[float, Tuple[int, ...], int]] = []

        for lag_set, kf in candidate_pairs:

            if (n_eval >= max_evals) or (time.time() >= tune_deadline):

                break

            try:

                cmp = rolling_origin_mode_comparison(
                    df = df_tk,
                    regressors = BASE_REGRESSORS,
                    horizon = horizon,
                    n_splits = coarse_splits,
                    draws = coarse_draws,
                    seed = int(rng.integers(1_000_000_000)),
                    exog_lags = lag_set,
                    fourier_k = kf,
                    orders = order_candidates,
                )
         
            except Exception:
         
                continue
         
            n_eval += 1
         
            coarse_rank.append((_cmp_objective(cmp), lag_set, kf))

        if coarse_rank:
         
            coarse_rank.sort(key=lambda z: z[0], reverse=True)
         
            keep_top = int(min(len(coarse_rank), max(4, min(10, int(np.sqrt(len(candidate_pairs)) * 2)))))
         
            stage_pairs = [(l, kf) for _, l, kf in coarse_rank[:keep_top]]

    for lag_set, kf in stage_pairs:
        
        if (n_eval >= max_evals) or (time.time() >= tune_deadline):
        
            break
        try:
        
            cmp = rolling_origin_mode_comparison(
                df = df_tk,
                regressors = BASE_REGRESSORS,
                horizon = horizon,
                n_splits = cv_splits,
                draws = draws_tune,
                seed = int(rng.integers(1_000_000_000)),
                exog_lags = lag_set,
                fourier_k = kf,
                orders = order_candidates,
            )
            
        except Exception:
            
            continue

        n_eval += 1
        
        score = _cmp_objective(
            cmp = cmp
        )
        
        if score > best_score:
        
            best_score = score
        
            best_lags = tuple(lag_set)
        
            best_k = int(kf)
        
            best_cmp = dict(cmp)

    if time.time() < tune_deadline and n_eval < max_evals:
    
        try:
    
            w_map, valid_orders = learn_stacking_weights(
                df = df_tk,
                regressors = BASE_REGRESSORS,
                orders = order_candidates,
                n_splits = cv_splits,
                horizon = horizon,
                exog_lags = best_lags,
                fourier_k = best_k,
            )
            
        except Exception:
            
            w_map, valid_orders = {}, []
 
    else:
 
        w_map, valid_orders = {}, []

    if valid_orders:
 
        ordered = sorted(valid_orders, key=lambda od: -float(w_map.get(od, 0.0)))
 
        max_keep = int(max(1, rs.get("max_stack_keep", 5)))
 
        stack_orders = ordered[: min(max_keep, len(ordered))]
 
    else:
 
        max_keep = int(max(1, rs.get("max_stack_keep", 5)))
 
        stack_orders = order_candidates[: min(max_keep, len(order_candidates))]

    if not stack_orders:
 
        stack_orders = list(STACK_ORDERS)

    y_centered = y - np.mean(y)

    student_df = int(np.clip(_fit_student_df_mle(y_centered), 4, 40))

    rho1 = abs(_sample_autocorr(y_centered, 1))
 
    resid_block = _auto_stationary_block_length(y_centered)

    gpd_min_exceed = int(np.clip(np.sqrt(T) * 2.5, 25, 120))
 
    best_jump_q, gpd_min_exceed = _select_jump_threshold_evt(
        resid = y_centered,
        min_exceed = gpd_min_exceed,
    )

    sk = float(skew(y_centered, bias = False, nan_policy = "omit")) if T > 8 else 0.0
    ek = float(kurtosis(y_centered, fisher = True, bias = False, nan_policy = "omit")) if T > 8 else 0.0

    hm_prior_weight = float(np.clip(T / (T + 180.0), 0.15, 0.98))
 
    hm_shrink = float(np.clip(T / (T + 220.0), 0.10, 0.95))

    sk_post = hm_shrink * sk
 
    ek_post = hm_shrink * ek

    hm_min_obs = int(np.clip(max(45, 0.20 * T), 45, 280))
 
    hm_clip_skew = float(np.clip(1.2 + 1.6 * abs(sk_post), 1.2, 4.2))
 
    hm_clip_exkurt = float(np.clip(3.0 + 1.8 * abs(ek_post), 3.0, 26.0))
 
    hm_enabled = bool(T >= max(40, hm_min_obs // 2))

    copula_shrink = _estimate_oas_shrinkage(
        resid = y_centered, 
        horizon = horizon
    )

    cov_target = float(max(1e-3, rs.get("cov_target_rel_err", TARGET_COV_REL_ERR)))
    
    reps_raw = int(np.ceil((horizon + 8) / (cov_target * cov_target)))
    
    model_cov_reps = int(np.clip(
        max(int(rs["cov_reps_min"]), reps_raw),
        int(rs["cov_reps_min"]),
        int(rs["cov_reps_max"]),
    ))

    y_std = float(np.std(y_centered, ddof=1)) if T > 1 else float(np.std(y_centered))
   
    n_sims_required = int(np.clip(
        300 + 18.0 * horizon * min(max(y_std, 1e-5) / 0.03, 4.0),
        int(rs["n_sims_min"]),
        int(rs["n_sims_max"]),
    ))

    bvar_p = int(np.clip(np.sqrt(max(T, 16)) / 8.0, 1, 4))
   
    l1 = float(np.clip(0.08 + 0.55 * rho1, 0.08, 0.60))
   
    l2 = float(np.clip(0.20 + 0.85 * rho1, 0.20, 1.20))
   
    l3 = float(np.clip(0.60 + 1.20 * (1.0 - rho1), 0.50, 2.00))
   
    l4 = float(np.clip(40.0 + 300.0 * y_std, 20.0, 250.0))
   
    niw_nu0 = int(np.clip(lenBR + 2 + np.log(max(T, 20)), lenBR + 2, 30))
   
    niw_s0_scale = float(np.clip(np.var(y_centered), 0.02, 0.8))

    hp = FittedHyperparams(
        stack_orders = stack_orders,
        exog_lags = best_lags,
        fourier_k = best_k,
        cv_splits = cv_splits,
        resid_block_len = resid_block,
        student_df = student_df,
        jump_q = best_jump_q,
        gpd_min_exceed = gpd_min_exceed,
        hm_min_obs = hm_min_obs,
        hm_clip_skew = hm_clip_skew,
        hm_clip_excess_kurt = hm_clip_exkurt,
        copula_shrink = copula_shrink,
        model_cov_reps = model_cov_reps,
        bvar_p = bvar_p,
        mn_lambdas = (l1, l2, l3, l4),
        niw_nu0 = niw_nu0,
        niw_s0_scale = niw_s0_scale,
        n_sims_required = n_sims_required,
        hm_enabled = hm_enabled,
        hm_prior_weight = hm_prior_weight,
    )

    if best_cmp:
    
        hp_cmp_key = _hp_mode_comparison_cache_key(
            df = df_tk,
            horizon = horizon,
            hp = hp,
        )
     
        _HP_MODE_COMPARISON_CACHE[hp_cmp_key] = dict(best_cmp)
     
        _maybe_trim_cache(
            d = _HP_MODE_COMPARISON_CACHE
        )

    if trace_out is not None:
        trace_out.update(
            asdict(
                HyperparamSearchTrace(
                    selected = {
                        "lags": list(best_lags),
                        "fourier_k": int(best_k),
                        "orders": [list(o) for o in stack_orders],
                        "student_df": int(student_df),
                        "jump_q": float(best_jump_q),
                        "copula_shrink": float(copula_shrink),
                        "model_cov_reps": int(model_cov_reps),
                        "n_sims_required": int(n_sims_required),
                        "n_eval": int(n_eval),
                        "max_evals": int(max_evals),
                        "tune_time_budget_sec": float(opt.max_tune_seconds),
                    },
                    objective = float(best_score if np.isfinite(best_score) else np.nan),
                    candidates_evaluated = int(n_eval),
                    elapsed_sec = float(time.time() - t0),
                )
            )
        )

    return hp


def fit_macro_hyperparams_from_levels(
    df_levels_weekly: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    Data-driven macro-model hyperparameters by country history.
    """

    dX = _macro_stationary_deltas(
        df_levels = df_levels_weekly
    )
    
    T = int(len(dX))

    if T < 20:
    
        return {
            "bvar_p": int(BVAR_P),
            "mn_lambdas": (MN_LAMBDA1, MN_LAMBDA2, MN_LAMBDA3, MN_LAMBDA4),
            "niw_nu0": int(NIW_NU0),
            "niw_s0_scale": float(NIW_S0_SCALE),
        }

    acfs = []
    
    for col in dX.columns:
    
        acfs.append(abs(_sample_autocorr(np.asarray(dX[col], float), 1)))
    
    ac1 = float(np.nanmean(acfs)) if acfs else 0.0

    bvar_p = int(np.clip(np.sqrt(T) / 8.0, 1, 4))
    
    l1 = float(np.clip(0.08 + 0.45 * ac1, 0.08, 0.60))
    
    l2 = float(np.clip(0.20 + 0.70 * ac1, 0.20, 1.20))
    
    l3 = float(np.clip(0.60 + 1.10 * (1.0 - ac1), 0.50, 2.00))
    
    l4 = float(np.clip(40.0 + 3.5 * horizon, 20.0, 250.0))
    
    nu0 = int(np.clip(lenBR + 2 + np.log(max(T, 20)), lenBR + 2, 30))
    
    s0_scale = float(np.clip(np.nanmean(np.var(dX.values, axis=0)), 0.02, 0.8))

    return {
        "bvar_p": bvar_p,
        "mn_lambdas": (l1, l2, l3, l4),
        "niw_nu0": nu0,
        "niw_s0_scale": s0_scale,
    }


def build_backtest_summary(
    df_tk: pd.DataFrame,
    horizon: int,
    hp: FittedHyperparams,
    seed: int,
) -> BacktestSummary:
    """
    Build cached backtest diagnostics for a ticker under fitted hyperparameters.
    """

    rs = _runtime_settings()

    draw_budget = int(rs["cv_draws_backtest"])
    
    if SKIP_BACKTEST_ON_PROD_RUN and (not FULL_RERUN):
    
        draw_budget = int(min(draw_budget, rs["cv_draws_tune"]))

    cmp: Dict[str, float] = {}
    
    opt = _optimization_profile()
    
    if (not FULL_RERUN) and opt.reuse_fold_metrics:
        hp_cmp_key = _hp_mode_comparison_cache_key(
            df = df_tk,
            horizon = horizon,
            hp = hp,
        )
        cached_cmp = _cache_get_touch(_HP_MODE_COMPARISON_CACHE, hp_cmp_key)
        if cached_cmp is not None:
            cmp = dict(cached_cmp)

    if not cmp:
       
        cmp = rolling_origin_mode_comparison(
            df = df_tk,
            regressors = BASE_REGRESSORS,
            horizon = horizon,
            n_splits = hp.cv_splits,
            draws = draw_budget,
            seed = seed,
            exog_lags = hp.exog_lags,
            fourier_k = hp.fourier_k,
            orders = hp.stack_orders,
            hm_min_obs = hp.hm_min_obs,
            hm_clip_skew = hp.hm_clip_skew,
            hm_clip_excess_kurt = hp.hm_clip_excess_kurt,
        )
        
        if cmp:
            
            hp_cmp_key = _hp_mode_comparison_cache_key(
                df = df_tk,
                horizon = horizon,
                hp = hp,
            )
          
            _HP_MODE_COMPARISON_CACHE[hp_cmp_key] = dict(cmp)
          
            _maybe_trim_cache(
                d = _HP_MODE_COMPARISON_CACHE
            )

    if not cmp:
        
        return BacktestSummary(
            logscore = float("nan"),
            rmse = float("nan"),
            wis90 = float("nan"),
            coverage90 = float("nan"),
            tail_low_exceed = float("nan"),
            tail_high_exceed = float("nan"),
            n_folds = 0,
        )

    return BacktestSummary(
        logscore = float(cmp.get("hm_logscore", np.nan)),
        rmse = float(cmp.get("hm_rmse", np.nan)),
        wis90 = float(cmp.get("hm_wis90", np.nan)),
        coverage90 = float(cmp.get("hm_cov90", np.nan)),
        tail_low_exceed = float(cmp.get("hm_low_exc", np.nan)),
        tail_high_exceed = float(cmp.get("hm_high_exc", np.nan)),
        n_folds = int(hp.cv_splits),
    )


def make_jump_indicator_from_returns(
    y: np.ndarray,
    q: float
) -> tuple[np.ndarray, float]:
    """
    Create a binary jump indicator from absolute returns via a high quantile threshold.

    Definition
    ----------
    Let a_t = |y_t| and τ = quantile_q(a). Then:

        J_t = 1{ a_t ≥ τ }.

    Returns both the indicator vector and its empirical probability.

    Parameters
    ----------
    y : numpy.ndarray
    q : float
        Tail quantile (e.g., 0.97).

    Returns
    -------
    J : numpy.ndarray (0/1)
    p_jump : float

    Why
    ---
    Rare but large shocks materially affect price paths; explicit jump flags enable
    state-conditional jump intensities and magnitudes.
    """

    a = np.abs(y)

    thr = np.quantile(a, q)

    j = (a >= thr).astype(np.float32)

    return j, float(j.mean())


def fit_gpd_tail(
    resid: np.ndarray,
    q: float = JUMP_Q,
    min_exceed: int = 30,
    min_obs: int = 100,
) -> Optional[dict]:
    """
    Fit a Generalised Pareto Distribution (GPD) to the tail of |residuals| using peaks-over-threshold.

    Procedure
    ---------
    1) Compute absolute residuals x_t = |r_t| and threshold τ = quantile_q(x).
    
    2) Form excesses e_t = x_t − τ for x_t > τ.
    
    3) Fit GPD(e | ξ, β) with fixed location 0 via MLE, returning shape ξ and scale β.
    
    4) Record p_neg = P(residual < 0 | exceedance) to model sign asymmetry.

    Parameters
    ----------
    resid : numpy.ndarray
    q : float, default JUMP_Q

    Returns
    -------
    dict | None
        {"thr": τ, "xi": ξ, "beta": β, "p_neg": p_neg} or None if insufficient data.

    Advantages
    ----------
    The peaks-over-threshold method is asymptotically justified for threshold exceedances
    and offers a flexible model for heavy tails beyond Gaussian assumptions.
    """

    if resid.size < max(min_obs, 2 * min_exceed):
    
        return None
    
    x = np.abs(resid)
    
    thr = float(np.quantile(x, q))
    
    mask = x > thr
    
    excess = x[mask] - thr
    
    if excess.size < int(min_exceed):
    
        return None
    
    try:
    
        xi, _, beta = genpareto.fit(excess, floc = 0.0)
    
        xi = float(np.clip(xi, -0.49, 0.95))  
    
        beta = float(max(beta, 1e-8))

    except Exception:
    
        return None
    
    p_neg = float((resid[mask] < 0).mean())
    
    return {
        "thr": thr,
        "xi": xi, 
        "beta": beta, 
        "p_neg": p_neg
    }


def _weighted_quantile(
    x: np.ndarray, 
    w: np.ndarray, 
    q: float
) -> float:
    """
    Compute a weighted quantile of a vector with positive weights using inverse-CDF interpolation.

    Parameters
    ----------
    x : numpy.ndarray
    w : numpy.ndarray
    q : float ∈ [0, 1]

    Returns
    -------
    float
        Weighted q-quantile.

    Notes
    -----
    Weights are normalised to sum to one after masking non-finite values.
    """

    x = np.asarray(x, float)

    w = np.asarray(w, float)

    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if not np.any(mask):

        return float(np.nan)

    x = x[mask]; w = w[mask]

    idx = np.argsort(x)

    x_sorted = x[idx]; w_sorted = w[idx]

    cw = np.cumsum(w_sorted)

    cw /= cw[-1]

    return float(np.interp(q, cw, x_sorted))


def fit_gpd_tail_weighted(
    resid: np.ndarray, 
    weights: np.ndarray,
    q: float = JUMP_Q, 
    min_exceed: int = 30
) -> Optional[dict]:
    """
    Fit a weighted GPD to |residuals| exceedances using importance-resampling.

    Method
    ------
    1) Compute a weighted quantile τ_w of x = |resid| using weights w.
    
    2) Select exceedances and resample them proportionally to weights to approximate a
    weighted likelihood.
    
    3) Fit a GPD(ξ, β) on the resampled excesses via MLE.

    Parameters
    ----------
    resid : numpy.ndarray
    weights : numpy.ndarray
    q : float, default JUMP_Q
    min_exceed : int, default 30

    Returns
    -------
    dict | None
        {"thr", "xi", "beta", "p_neg"} or None.

    Use
    ---
    Enables state-conditional tail estimation by weighting observations with HMM posteriors.
    """

    x = np.abs(np.asarray(resid, float))
    
    w = np.asarray(weights, float)
    
    thr = _weighted_quantile(
        x = x, 
        w = w, 
        q = q
    )
    
    if not np.isfinite(thr):
    
        return None
    
    mask = x > thr
    
    if mask.sum() < min_exceed:
    
        return None
    
    exc = x[mask] - thr
    
    w_exc = w[mask]
    
    w_exc = np.maximum(w_exc, 1e-12)
    
    p = w_exc / w_exc.sum()
    
    m = int(np.clip(mask.sum() * 3, 100, 5000)) 
    
    idx = rng_global.choice(exc.size, size=m, replace=True, p=p)
    
    sample = exc[idx]
    
    try:
    
        xi, _, beta = genpareto.fit(sample, floc=0.0)
    
        xi = float(np.clip(xi, -0.49, 0.95))
    
        beta = float(max(beta, 1e-8))
    
    except Exception:
    
        return None
    
    p_neg = float((resid[mask] < 0).mean())
    
    return {
        "thr": float(thr), 
        "xi": xi, 
        "beta": beta,
        "p_neg": p_neg
    }


def _hmm_posteriors_for_resid_1d(
    resid_1d: np.ndarray,
    ms: MSVolParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtain filtered/posterior state probabilities for a 1-D residual series under a
    pre-estimated 2-state scale-HMM.

    Model
    -----
    r_t | S_t = s ~ N(0, g_s^2 Σ), with Σ scalar (from `ms.Sigma`).

    Parameters
    ----------
    resid_1d : numpy.ndarray
    ms : MSVolParams

    Returns
    -------
    gamma : numpy.ndarray, shape (T, 2)
    p_last : numpy.ndarray, shape (2,)
    """

    r = np.asarray(resid_1d, float).reshape(-1, 1)

    Sigma = ms.Sigma

    Sinv = np.linalg.inv(Sigma)

    _, logdetS = np.linalg.slogdet(Sigma)

    q = np.einsum('ti,ij,tj->t', r, Sinv, r)  

    loglik_t = np.column_stack([
        -0.5 * (1 * np.log(ms.g[0] ** 2) + logdetS + q / (ms.g[0] ** 2)),
        -0.5 * (1 * np.log(ms.g[1] ** 2) + logdetS + q / (ms.g[1] ** 2)),
    ])

    gamma, _, _ = _forward_backward(
        loglik_t = loglik_t, 
        P = ms.P,
        pi = ms.pi
    )

    return gamma, gamma[-1]


def estimate_state_conditional_jump_params(
    resid_1d: np.ndarray, 
    jump_ind: np.ndarray,
    gamma: np.ndarray,
    q: float = JUMP_Q,
    min_exceed: int = 30,
) -> Tuple[np.ndarray, List[Optional[dict]]]:
    """
    Estimate state-conditional jump intensities and GPD tail parameters.

    For HMM posteriors γ_t(s) and jump indicators J_t:

    - Jump probabilities:
        p_jump[s] = (Σ_t γ_t(s) J_t) / (Σ_t γ_t(s)).

    - Tail magnitudes:
        Fit weighted GPD on |resid| with weights γ_t(s) to obtain (τ_s, ξ_s, β_s),
        and estimate sign asymmetry p_neg from signs on exceedances.

    Parameters
    ----------
    resid_1d : numpy.ndarray
    jump_ind : numpy.ndarray
    gamma : numpy.ndarray, shape (T, 2)
    q : float

    Returns
    -------
    p_jump : numpy.ndarray, shape (2,)
    gpd_params : list[dict | None], length 2
    """

    resid_1d = np.asarray(resid_1d, float).reshape(-1)
    
    jump_ind = np.asarray(jump_ind, int).reshape(-1)
    
    assert gamma.shape[0] == resid_1d.shape[0] == jump_ind.shape[0] and gamma.shape[1] == 2
    
    p_jump = np.zeros(2, float)
    
    gpd_by_state: List[Optional[dict]] = [None, None]
    
    for s in (0, 1):
    
        w = gamma[:, s]
    
        wsum = np.sum(w)
    
        if wsum <= 0:
    
            p_jump[s] = float(jump_ind.mean())
    
            gpd_by_state[s] = None
    
            continue
    
        p_jump[s] = float(np.sum(w * jump_ind) / wsum)
    
        gpd_by_state[s] = fit_gpd_tail_weighted(
            resid = resid_1d, 
            weights = w, 
            q = q,
            min_exceed = min_exceed,
        )
    
    return p_jump, gpd_by_state


def draw_state_conditional_jumps(
    states: np.ndarray, 
    p_jump: np.ndarray, 
    gpd_params: List[Optional[dict]], 
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate binary jump occurrences and signed magnitudes conditional on HMM states.

    For each step t with state s = states[t]:
  
    1) Draw J_t ~ Bernoulli(p_jump[s]).
    
    2) If J_t = 1 and parameters are available, draw magnitude:
    
        |shock_t| = τ_s + GPD_rv(ξ_s, β_s),
    
    and assign sign negative with probability p_neg_s.

    Parameters
    ----------
    states : numpy.ndarray
    p_jump : numpy.ndarray
    gpd_params : list[dict | None]
    rng : numpy.random.Generator

    Returns
    -------
    J : numpy.ndarray (0/1)
    shocks : numpy.ndarray (signed)
    """

    H = int(len(states))
   
    states = np.asarray(states, int)
   
    J = (rng.uniform(size=H) < p_jump[states]).astype(int)
   
    shocks = np.zeros(H, float)
   
    for s in (0, 1):
   
        idx = np.where((J == 1) & (states == s))[0]
   
        if idx.size == 0:
   
            continue
   
        params = gpd_params[s]
   
        if params is None:
   
            continue
   
        m = idx.size
   
        mags = params["thr"] + genpareto.rvs(c=params["xi"], scale=params["beta"], size=m, random_state=rng)
   
        neg_flags = rng.uniform(size = m) < params["p_neg"]
   
        signs = np.where(neg_flags, -1.0, 1.0)
   
        shocks[idx] = signs * mags
   
    return J, shocks


def t_scale_factors(
    length: int, 
    df: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate Student-t variance-normalised scale factors for fattening bootstrap residuals.

    Construction
    ------------
    Let ν be degrees of freedom. Draw χ^2 ~ ChiSquare(ν) and set:

        s = sqrt(ν / χ^2) / sqrt(ν / (ν − 2)),

    so that E[s^2] = 1 for ν > 2. Multiplying Gaussian or bootstrap noise by s introduces
    heavy tails without changing variance on average.

    Parameters
    ----------
    length : int
    df : int
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, shape (length,)
    """

    chi2 = rng.chisquare(df, size = length)
   
    scales = np.sqrt(df / chi2)
   
    scales /= np.sqrt(df / (df - 2.0))  
   
    return scales


def get_t_scales(
    H: int,
    df: int,
    rng
):
    """
    Cache and return Student-t scale factors for (H, df).

    Parameters
    ----------
    H : int
    df : int
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
    """

    key = (H, df)

    s = _T_SCALES_CACHE.get(key)

    if s is None:

        s = t_scale_factors(
            length = H, 
            df = df,
            rng = rng
        )

        _T_SCALES_CACHE[key] = s

    return s


def _gaussian_hm_params(
    x: np.ndarray
) -> HigherMomentParams:
    """
    Build Gaussian-based higher-moment parameter summary from a sample.

    Statistical definitions
    -----------------------
    For finite observations `x_1, ..., x_n`, compute:

        mean      = (1/n) * sum_i x_i,
        std       = sample standard deviation (ddof=1 when n > 1),
        skew      = sample skewness,
        exkurt    = sample excess kurtosis (Fisher definition).

    Skewness and excess kurtosis are clipped to configured bounds:

        skew   in [-HM_CLIP_SKEW, HM_CLIP_SKEW],
        exkurt in [-1.0, HM_CLIP_EXCESS_KURT].

    When `n = 0`, the function returns the neutral Gaussian baseline:
    mean `0`, std `1`, skew `0`, exkurt `0`.

    Parameters
    ----------
    x : numpy.ndarray
        Standardised residual sample.

    Returns
    -------
    HigherMomentParams
        Parameter container with `dist_name="gaussian"` and empty distribution parameters.

    Why this process
    ----------------
    This function provides a conservative fallback when non-Gaussian fitting is unavailable
    or unsupported by sample size, while still preserving empirical moment diagnostics.

    Advantages
    ----------
    - Always returns a valid distributional descriptor.
   
    - Retains informative sample moments for diagnostics and blending decisions.
   
    - Avoids unstable tail fits under sparse data conditions.
   
    """

    xs = np.asarray(x, float)

    xs = xs[np.isfinite(xs)]

    n_eff = int(xs.size)

    if n_eff == 0:

        return HigherMomentParams(
            dist_name = "gaussian",
            params = (),
            mean = 0.0,
            std = 1.0,
            skew = 0.0,
            exkurt = 0.0,
            n_eff = 0,
        )

    mu = float(np.mean(xs))

    sd = float(np.std(xs, ddof = 1)) if n_eff > 1 else float(np.std(xs))

    sd = float(max(sd, 1e-8))

    sk_raw = skew(xs, bias = False, nan_policy = "omit")

    ek_raw = kurtosis(xs, fisher = True, bias = False, nan_policy = "omit")

    sk = float(np.clip(float(sk_raw) if np.isfinite(sk_raw) else 0.0, -HM_CLIP_SKEW, HM_CLIP_SKEW))

    ek = float(
        np.clip(
            float(ek_raw) if np.isfinite(ek_raw) else 0.0,
            -1.0,
            HM_CLIP_EXCESS_KURT,
        )
    )

    return HigherMomentParams(
        dist_name = "gaussian",
        params = (),
        mean = mu,
        std = sd,
        skew = sk,
        exkurt = ek,
        n_eff = n_eff,
        mix_weight = 0.0,
    )


def fit_higher_moment_params(
    resid_standardized: np.ndarray
) -> HigherMomentParams:
    """
    Fit a higher-moment innovation distribution to standardized residuals.

    Uses Johnson SU by default with Gaussian fallback when sample size is low
    or fitting fails.
    """

    x = np.asarray(resid_standardized, float)

    x = x[np.isfinite(x)]

    n_eff = int(x.size)

    if n_eff < HM_MIN_OBS or INNOV_DIST_MODE.lower() == "gaussian":

        return _gaussian_hm_params(x)

    try:

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            a, b, loc, scale = johnsonsu.fit(x)

            m, v, sk, ek = johnsonsu.stats(a, b, loc=loc, scale=scale, moments="mvsk")

        m = float(m)

        sd = float(np.sqrt(float(v)))

        if not np.isfinite(m) or not np.isfinite(sd) or sd <= 1e-8:

            return _gaussian_hm_params(
                x = x
            )

        sk = float(np.clip(float(sk), -HM_CLIP_SKEW, HM_CLIP_SKEW))

        ek = float(np.clip(float(ek), -1.0, HM_CLIP_EXCESS_KURT))

        return HigherMomentParams(
            dist_name = "johnson_su",
            params = (float(a), float(b), float(loc), float(scale)),
            mean = m,
            std = sd,
            skew = sk,
            exkurt = ek,
            n_eff = n_eff,
            mix_weight = float(np.clip(n_eff / (n_eff + HM_MIN_OBS), 0.05, 0.98)),
        )

    except Exception:

        return _gaussian_hm_params(
            x = x
        )


def _weighted_resample_1d(
    x: np.ndarray,
    w: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a weighted bootstrap sample from a one-dimensional array.

    Method
    ------
    After masking to finite observations with strictly positive weights, probabilities are
    normalised:

        p_i = w_i / sum_j w_j.

    Indices are then sampled with replacement:

        i_1, ..., i_m ~ Categorical(p),  where m = size.

    The returned sample is `(x_{i_1}, ..., x_{i_m})`.

    Parameters
    ----------
    x : numpy.ndarray
        Source values.
    w : numpy.ndarray
        Non-negative importance weights aligned with `x`.
    size : int
        Number of bootstrap draws.
    rng : numpy.random.Generator
        Random generator used for reproducible sampling.

    Returns
    -------
    numpy.ndarray
        Weighted resample of length `size`; empty array when no valid weighted observations
        are available.

    Why this process
    ----------------
    State-conditional higher-moment fitting requires approximating weighted empirical
    distributions implied by HMM posterior probabilities. Weighted resampling provides a
    straightforward Monte-Carlo approximation to weighted likelihood fitting.

    Advantages
    ----------
    - Simple and robust approximation to posterior-weighted data generation.
  
    - Preserves support of observed residuals.
  
    - Compatible with existing unweighted fitting routines downstream.
  
    """

    x = np.asarray(x, float)

    w = np.asarray(w, float)

    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if not np.any(mask):

        return np.asarray([], float)

    xv = x[mask]

    wv = w[mask]

    p = wv / np.maximum(wv.sum(), 1e-300)

    idx = rng.choice(xv.size, size = int(size), replace = True, p = p)

    return xv[idx]


def fit_state_conditional_hm_params(
    resid_standardized: np.ndarray,
    gamma: Optional[np.ndarray],
    rng: np.random.Generator,
    fallback: HigherMomentParams,
    prior_weight: float = 0.5,
) -> List[HigherMomentParams]:
    """
    Fit state-conditional higher-moment parameters with posterior-weighted resampling.
    """

    if gamma is None or gamma.ndim != 2 or gamma.shape[1] != 2:

        return [fallback, fallback]

    x = np.asarray(resid_standardized, float)

    T = min(x.size, gamma.shape[0])

    if T < HM_MIN_OBS:

        return [fallback, fallback]

    x = x[-T:]

    G = gamma[-T:]

    out: List[HigherMomentParams] = []

    for s in (0, 1):

        ws = G[:, s]

        eff = float((np.sum(ws) ** 2) / np.maximum(np.sum(ws ** 2), 1e-300))

        if eff < max(40.0, HM_MIN_OBS * 0.35):

            out.append(fallback)

            continue

        sample = _weighted_resample_1d(
            x = x,
            w = ws,
            size = max(HM_MIN_OBS, T),
            rng = rng,
        )

        if sample.size < HM_MIN_OBS:

            out.append(fallback)

            continue

        pfit = fit_higher_moment_params(
            resid_standardized = sample
        )

        w = float(np.clip(prior_weight * eff / (eff + HM_MIN_OBS), 0.05, 0.98))

        out.append(
            HigherMomentParams(
                dist_name = pfit.dist_name,
                params = pfit.params,
                mean = pfit.mean,
                std = pfit.std,
                skew = pfit.skew,
                exkurt = pfit.exkurt,
                n_eff = pfit.n_eff,
                mix_weight = w,
            )
        )

    return out


def draw_higher_moment_z(
    params: HigherMomentParams,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw standardized innovations (mean ~ 0, variance ~ 1) with optional higher moments.
    """

    size = int(size)

    if size <= 0:

        return np.asarray([], float)

    if params.dist_name == "johnson_su" and len(params.params) == 4:

        a, b, loc, scale = params.params

        raw = johnsonsu.rvs(a, b, loc = loc, scale = scale, size = size, random_state = rng)

        z = (raw - params.mean) / max(params.std, 1e-8)

        w = float(np.clip(getattr(params, "mix_weight", 1.0), 0.0, 1.0))
       
        if w < 0.999:
       
            zg = rng.standard_normal(size = size)
       
            z = w * np.asarray(z, float) + (1.0 - w) * zg

        return np.clip(np.asarray(z, float), -12.0, 12.0)

    return rng.standard_normal(size = size)


def _draw_correlated_hm_z(
    dep: ModelStepDependence,
    hm_params_by_state: List[HigherMomentParams],
    states: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw horizon-correlated standardized innovations with non-Gaussian marginals
    via a Gaussian copula transform.
    """

    states = np.asarray(states, int).reshape(-1)

    H = int(states.size)

    if H <= 0:

        return np.asarray([], float)

    R = np.asarray(dep.corr, float)

    if R.shape != (H, H):

        R = np.eye(H, dtype=float)

    R = _nearest_psd_corr(R)

    try:

        z_g = rng.multivariate_normal(mean = np.zeros(H), cov = R, check_valid = "ignore")

    except Exception:

        z_g = rng.standard_normal(H)

    u = norm.cdf(z_g)

    u = np.clip(u, U_EPS, 1.0 - U_EPS)
   
    z_g_std = norm.ppf(u)
   
    z_out = np.asarray(z_g_std, float).copy()

    if len(hm_params_by_state) == 0:
   
        return np.clip(np.where(np.isfinite(z_out), z_out, 0.0), -12.0, 12.0)

    s_idx = np.where((states >= 0) & (states < len(hm_params_by_state)), states, 0).astype(int)

    for s in np.unique(s_idx):
   
        p = hm_params_by_state[int(s)]
   
        if not (p.dist_name == "johnson_su" and len(p.params) == 4):
   
            continue
   
        mask = (s_idx == s)
   
        if not np.any(mask):
   
            continue
   
        a, b, loc, scale = p.params
   
        raw = johnsonsu.ppf(u[mask], a, b, loc=loc, scale=scale)
   
        z_hm = (np.asarray(raw, float) - p.mean) / max(p.std, 1e-8)
   
        w = float(np.clip(getattr(p, "mix_weight", 1.0), 0.0, 1.0))
   
        z_out[mask] = w * z_hm + (1.0 - w) * z_g_std[mask]

    z_out = np.where(np.isfinite(z_out), z_out, 0.0)

    return np.clip(z_out, -12.0, 12.0)


def simulate_price_paths_for_ticker(
    tk: str,
    df_tk: pd.DataFrame,  
    cp: float,
    lb: float,
    ub: float,
    macro_sims: Optional[np.ndarray],
    horizon: int,
    rng_seed: int,
    fitted_hp: Optional[FittedHyperparams] = None,
    backtest_out: Optional[Dict[str, BacktestSummary]] = None,
    diag_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Simulate price scenarios for a single ticker using a SARIMAX ensemble with macro
    exogenous paths, regime-conditional jumps, and blended innovation noise.

    Pipeline
    --------
    
    1) Build historical exogenous design X_hist (lagged macros and Fourier)
    and standardise; fit a SARIMAX ensemble and cache.
    
    2) Cross-validation calibration:
    
    - Call `rolling_origin_cv_rmse_return_sum` to obtain residuals, a bootstrap H-sum
        variance proxy rs_std, and a model-based forecast variance proxy v_fore_mean.
    
    - Choose α by `calibrate_alpha_from_cv`.
    
    3) Residual diagnostics → set Student-t degrees of freedom for bootstrap scaling.
    
    4) Learn stacking weights over a small order set via `learn_stacking_weights` (optional).
    
    5) Estimate a 2-state HMM on residuals for volatility regimes; obtain γ and last-state probs.
    
    6) Estimate state-conditional jump probabilities and tail magnitudes; or fall back to
    unconditional tail fitting.
    
    7) For each simulation i:
    
    a) Select a macro delta path `macro_path[i]` (or zeros).
    
    b) Simulate HMM states S_t and obtain volatility scales gpath_t.
    
    c) Simulate binary jumps and signed magnitudes; construct exogenous future Xf with
        `build_future_exog` using the macro path and jump vector.
    
    d) Forecast the SARIMAX ensemble H steps ahead; form mixture mean μ_mix and
       full horizon covariance Σ_mix from per-model simulated forecast covariance.
    
    e) Generate noise as:
    
            η_model_t = std_mix_t * z_t,
            z_t drawn from a Gaussian-copula process with correlation from Σ_mix and
            Johnson SU / Gaussian marginals,
    
            η_bootstrap_t from stationary bootstrap of residuals, fattened by t-scales,
    
            η_t = α η_model_t + (1 − α) η_bootstrap_t,
    
        then scale by gpath_t (regime volatility), and add jump shocks separately.
    
    f) Returns path r_t = μ_mix_t + η_t + jump_shock_t.
    
    g) Price path P_t = P_0 × exp(Σ_{u=1..t} r_u), clipped to [lb, ub] at horizon.

    8) Summarise final prices with 5th/50th/95th percentiles, and return return-mean and s.d.

    Parameters
    ----------
    tk : str
    df_tk : pandas.DataFrame
        Columns: ["price", "y"] + BASE_REGRESSORS + ["jump_ind"].
    cp : float
    lb, ub : float
    macro_sims : numpy.ndarray | None, shape (n_sims, H, k_macro)
    horizon : int
    rng_seed : int

    Returns
    -------
    dict
        {"low", "avg", "high", "returns", "se"}.

    Why this design
    ---------------
    - SARIMAX with exogenous regressors: couples asset returns to macro deltas and seasonality.
    
    - Ensemble weighting (AICc/stacking): hedges over ARMA orders to reduce specification risk.
    
    - Regime-switching volatility and jumps: match empirical clustering and heavy tails.
    
    - Bootstrap blending: accounts for residual dependence beyond Gaussian innovations.
    """

    rng = np.random.default_rng(rng_seed)
  
    sim_perf = RunPerfStats(
        wall_sec_by_phase = {}
    )

    rs = _runtime_settings()

    hp = fitted_hp if fitted_hp is not None else fit_hyperparams_from_data(
        df_tk = df_tk,
        horizon = horizon,
        seed = rng_seed,
    )

    exog_lags = tuple(hp.exog_lags)

    fourier_k = int(hp.fourier_k)

    cv_splits = int(hp.cv_splits)

    resid_block = int(hp.resid_block_len)

    model_cov_reps = int(hp.model_cov_reps)

    copula_shrink = float(hp.copula_shrink)

    jump_q = float(hp.jump_q)

    gpd_min_exceed = int(hp.gpd_min_exceed)

    n_sims_budget = int(np.clip(
        int(hp.n_sims_required),
        int(rs["n_sims_min"]),
        int(rs["n_sims_max"]),
    ))
    
    n_sims_budget = max(1, n_sims_budget)

    n_sims_min = int(max(int(rs["n_sims_min"]), min(120, n_sims_budget)))
  
    n_sims_batch = int(max(20, min(int(rs["n_sims_batch"]), n_sims_budget)))
  
    q_target = float(max(1e-4, rs.get("q_ci_half_width", TARGET_Q_CI_HALF_WIDTH)))
  
    cov_target_rel_err = float(max(1e-3, rs.get("cov_target_rel_err", TARGET_COV_REL_ERR)))

    macro_pool = macro_sims if macro_sims is not None else None

    k_macro = lenBR

    X_hist = make_exog(
        df = df_tk, 
        base = BASE_REGRESSORS, 
        lags = exog_lags,
        add_fourier = True,
        K = fourier_k
    )
    
    y_hist = df_tk["y"].loc[X_hist.index]
    
    sc = StandardScaler().fit(X_hist.values)
    
    zero_var = np.isclose(sc.scale_, 0)

    if zero_var.any():

        X_hist = X_hist[X_hist.columns[~zero_var]]

        y_hist = y_hist.loc[X_hist.index]

        sc = StandardScaler().fit(X_hist.values)

    col_order = list(X_hist.columns)
    
    X_hist_arr = sc.transform(X_hist.values)

    max_lag = max(exog_lags) if exog_lags else 0

    hist_tail_base = df_tk.loc[:, BASE_REGRESSORS].to_numpy()[- (max_lag + 2):, :]

    try:
        
        ens = _fit_sarimax_candidates_cached_np(
            y = y_hist, 
            X_arr = X_hist_arr, 
            index = X_hist.index, 
            col_order = col_order,
            orders = hp.stack_orders,
            fit_mode = "final",
        )
    except Exception:
        
        ens = _fit_sarimax_candidates_np(
            y = y_hist,
            X_arr = X_hist_arr,
            index = X_hist.index,
            orders = [(1, 0, 0)],
            fit_mode = "final",
        )

    resid, rs_std_arr, v_fore_mean = rolling_origin_cv_rmse_return_sum(
        df = df_tk,
        regressors = BASE_REGRESSORS,
        n_splits = cv_splits,
        horizon = horizon,
        y_full = y_hist,
        X_full_arr = X_hist_arr,
        col_order = col_order,
        ens_full = ens,
        exog_lags = exog_lags,
        fourier_k = fourier_k,
        orders = hp.stack_orders,
    )
    
    resid_c = resid - np.mean(resid) if resid.size else np.array([0.0])
  
    resid_std_hist = np.std(resid_c) + 1e-9
  
    resid_standardized = resid_c / resid_std_hist 

    rs_std = float(rs_std_arr[0]) if rs_std_arr.size else np.nan

    alpha = calibrate_alpha_from_cv(
        v_fore_mean = v_fore_mean,
        rs_std = rs_std,
    )
    
    df_local = int(np.clip(hp.student_df, 4, 30))

    if not hp.hm_enabled:
        
        hm_params = _gaussian_hm_params(
            x = resid_standardized
        )
    
    else:
    
        hm_min_obs_old = HM_MIN_OBS
    
        hm_clip_skew_old = HM_CLIP_SKEW
    
        hm_clip_exk_old = HM_CLIP_EXCESS_KURT
    
        try:
    
            globals()["HM_MIN_OBS"] = int(hp.hm_min_obs)
    
            globals()["HM_CLIP_SKEW"] = float(hp.hm_clip_skew)
    
            globals()["HM_CLIP_EXCESS_KURT"] = float(hp.hm_clip_excess_kurt)
    
    
            hm_params = fit_higher_moment_params(
                resid_standardized = resid_standardized
            )
            
        finally:
      
            globals()["HM_MIN_OBS"] = hm_min_obs_old
      
            globals()["HM_CLIP_SKEW"] = hm_clip_skew_old
      
            globals()["HM_CLIP_EXCESS_KURT"] = hm_clip_exk_old

    hm_params = HigherMomentParams(
        dist_name = hm_params.dist_name,
        params = hm_params.params,
        mean = hm_params.mean,
        std = hm_params.std,
        skew = hm_params.skew,
        exkurt = hm_params.exkurt,
        n_eff = hm_params.n_eff,
        mix_weight = float(np.clip(getattr(hm_params, "mix_weight", 1.0) * hp.hm_prior_weight, 0.0, 1.0)),
    )

    try:
    
        ms_tk = estimate_ms_vol_params_hmm(
            resid = resid_c.reshape(-1, 1)
        )
    
        gamma_tk, p_last_tk = _hmm_posteriors_for_resid_1d(
            resid_1d = resid_c,
            ms = ms_tk
        )
    
    except Exception as _e:
    
        logger.warning("%s HMM failed on ticker residuals (%s). Falling back to no MS at ticker level.", tk, _e)
    
        ms_tk = None
    
        gamma_tk = None
    
        p_last_tk = np.array([0.5, 0.5], float)
        
    jump_hist = df_tk["jump_ind"].loc[X_hist.index].to_numpy(dtype=int)
   
    if ms_tk and gamma_tk is not None:

        p_jump_by_state, gpd_params_by_state = estimate_state_conditional_jump_params(
            resid_1d = resid_c[-len(jump_hist):],  
            jump_ind = jump_hist,
            gamma = gamma_tk[-len(jump_hist):],
            q = jump_q,
            min_exceed = gpd_min_exceed,
        )
        
    else:
     
        p_u = float(np.mean(jump_hist)) if jump_hist.size else 0.0
     
        p_jump_by_state = np.array([p_u, p_u])
     
        gpd_params_by_state = [
            fit_gpd_tail(
                resid = resid_c, 
                q = jump_q,
                min_exceed = gpd_min_exceed,
            )
        ] * 2

    if not hp.hm_enabled:
 
        hm_params_by_state = [hm_params, hm_params]
 
    else:
 
        hm_min_obs_old = HM_MIN_OBS
 
        hm_clip_skew_old = HM_CLIP_SKEW
 
        hm_clip_exk_old = HM_CLIP_EXCESS_KURT
 
        try:
 
            globals()["HM_MIN_OBS"] = int(hp.hm_min_obs)
 
            globals()["HM_CLIP_SKEW"] = float(hp.hm_clip_skew)
 
            globals()["HM_CLIP_EXCESS_KURT"] = float(hp.hm_clip_excess_kurt)
 
            hm_params_by_state = fit_state_conditional_hm_params(
                resid_standardized = resid_standardized,
                gamma = gamma_tk if ms_tk is not None else None,
                rng = rng,
                fallback = hm_params,
                prior_weight = float(hp.hm_prior_weight),
            )
    
        finally:
    
            globals()["HM_MIN_OBS"] = hm_min_obs_old
    
            globals()["HM_CLIP_SKEW"] = hm_clip_skew_old
    
            globals()["HM_CLIP_EXCESS_KURT"] = hm_clip_exk_old

    layout_key = (tuple(col_order), tuple(exog_lags))
    
    layout = _cache_get_touch(
        d = _COMPILED_EXOG_LAYOUT_CACHE, 
        key = layout_key
    )
    
    if layout is None:

        _ = build_future_exog(
            hist_tail_vals = hist_tail_base,
            macro_path = np.zeros((horizon, k_macro), dtype = float),
            col_order = col_order,
            scaler = sc,
            lags = exog_lags,
            jump = None,
            t_start = len(df_tk),
            layout = None,
        )
        
        layout = _cache_get_touch(
            d = _COMPILED_EXOG_LAYOUT_CACHE, 
            key = layout_key
        )
        
    has_jump_exog = bool(layout is not None and int(layout.get("jump_col", -1)) >= 0)

    mean_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
   
    exog_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
   
    cov_cache: Dict[Tuple[int, int], np.ndarray] = {}

    static_covs: List[np.ndarray] = []
   
    use_static_cov = bool(USE_STATIC_MODEL_COV)
   
    t_cov0 = time.time()
   
    if use_static_cov:
   
        try:
   
            macro_ref = np.zeros((horizon, k_macro), dtype=float)
   
            if macro_pool is not None and macro_pool.size:
   
                macro_ref = np.asarray(np.mean(macro_pool, axis=0), float)
   
            Xf_ref = build_future_exog(
                hist_tail_vals = hist_tail_base,
                macro_path = macro_ref,
                col_order = col_order,
                scaler = sc,
                lags = exog_lags,
                jump = np.zeros(horizon, dtype = float),
                t_start = len(df_tk),
                layout = _cache_get_touch(
                    d = _COMPILED_EXOG_LAYOUT_CACHE,
                    key = layout_key
                ),
            )
            
            for res in ens.fits:
            
                cov_k = _extract_forecast_cov_from_simulation(
                    res = res,
                    exog_f = Xf_ref,
                    horizon = horizon,
                    reps = model_cov_reps,
                    rng = rng,
                    target_rel_err = cov_target_rel_err,
                )
        
                sim_perf.sim_cov_calls += 1
        
                static_covs.append(cov_k)
        
            if len(static_covs) != len(ens.fits):
        
                use_static_cov = False
        
            elif not all(np.all(np.isfinite(np.diag(c))) and np.min(np.diag(c)) > 0 for c in static_covs):
        
                use_static_cov = False
        
        except Exception:
        
            use_static_cov = False
        
            static_covs = []
    
    sim_perf.wall_sec_by_phase["static_cov_precompute"] = float(time.time() - t_cov0)

    final_prices_buf = np.empty(n_sims_budget, dtype=float)
    
    final_prices_count = 0

    std_mix_med_sum = 0.0
    
    abs_offdiag_corr_sum = 0.0
    
    diag_obs = 0

    sims_done = 0
    
    batch_counter = 0
    
    mean_cache_enabled = True
    
    mc_diag_every = int(max(1, rs.get("mc_diag_every_batches", MC_DIAG_EVERY_BATCHES)))
    
    t_sim_loop0 = time.time()

    while sims_done < n_sims_budget:
    
        batch_n = int(min(n_sims_batch, n_sims_budget - sims_done))
    
        batch_counter += 1

        if macro_pool is None:
    
            batch_macro_idx = np.full(batch_n, -1, dtype=int)
    
        else:
    
            batch_macro_idx = rng.integers(0, macro_pool.shape[0], size=batch_n)

        for b in range(batch_n):

            midx = int(batch_macro_idx[b])

            if midx < 0:
    
                macro_path = np.zeros((horizon, k_macro))
    
            else:
    
                macro_path = macro_pool[midx]

            if ms_tk:

                S_tk = simulate_ms_states_conditional(
                    steps = horizon,
                    P = ms_tk.P,
                    rng = rng,
                    p_init = p_last_tk
                )

                gpath_tk = ms_tk.g[S_tk].astype(float)

            else:

                S_tk = np.zeros(horizon, dtype=int)

                gpath_tk = np.ones(horizon, float)

            J, jump_shocks = draw_state_conditional_jumps(
                states = S_tk,
                p_jump = p_jump_by_state,
                gpd_params = gpd_params_by_state,
                rng = rng
            )

            jump_key: Optional[bytes]
           
            if has_jump_exog:
           
                jump_key = bytes(np.packbits(J.astype(np.uint8), bitorder="little"))
           
                exog_key = (midx, jump_key)
           
            else:
           
                jump_key = None
           
                exog_key = (midx,)
           
            Xf = _cache_get_touch(
                d = exog_cache, 
                key = exog_key
            )
            
            if Xf is None:
            
                layout = _cache_get_touch(
                    d = _COMPILED_EXOG_LAYOUT_CACHE,
                    key = layout_key
                )
                
                Xf = build_future_exog(
                    hist_tail_vals = hist_tail_base,
                    macro_path = macro_path,
                    col_order = col_order,
                    scaler = sc,
                    lags = exog_lags,
                    jump = (J.astype(float) if has_jump_exog else None),
                    t_start = len(df_tk),
                    layout = layout,
                )
                
                exog_cache[exog_key] = Xf
                
                _maybe_trim_cache(
                    d = exog_cache
                )
                
                sim_perf.cache_misses += 1
                
            else:
                
                sim_perf.cache_hits += 1

            mu_stack: List[np.ndarray] = []

            cov_stack: List[np.ndarray] = []

            for k_res, res in enumerate(ens.fits):
                
                mean_key = (int(k_res), midx, jump_key) if has_jump_exog else (int(k_res), midx)
                
                mu_k = _cache_get_touch(
                    d = mean_cache, 
                    key = mean_key
                ) if mean_cache_enabled else None
                
                if mu_k is None:
                    
                    f = res.get_forecast(steps = horizon, exog = Xf)
                 
                    sim_perf.sim_forecast_calls += 1

                    mu_k = np.asarray(f.predicted_mean, float).reshape(-1)

                    if mean_cache_enabled:
                 
                        mean_cache[mean_key] = mu_k
                 
                        _maybe_trim_cache(
                            d = mean_cache
                        )
                 
                        sim_perf.cache_misses += 1
                
                else:
                
                    sim_perf.cache_hits += 1

                if use_static_cov and k_res < len(static_covs):
                
                    cov_k = static_covs[k_res]
                
                else:
                
                    cov_k = None
                
                    if not has_jump_exog:
                
                        cov_key = (int(k_res), int(midx))
                
                        cov_k = _cache_get_touch(
                            d = cov_cache,
                            key = cov_key
                        )
                
                    if cov_k is None:
                
                        cov_k = _extract_forecast_cov_from_simulation(
                            res = res,
                            exog_f = Xf,
                            horizon = horizon,
                            reps = model_cov_reps,
                            rng = rng,
                            target_rel_err = cov_target_rel_err,
                        )
                     
                        sim_perf.sim_cov_calls += 1
                     
                        if not has_jump_exog:
                     
                            cov_cache[cov_key] = cov_k
                     
                            _maybe_trim_cache(
                                d = cov_cache
                            )

                mu_stack.append(mu_k)

                cov_stack.append(cov_k)

            dep = _mixture_mean_cov(
                mus = mu_stack,
                covs = cov_stack,
                weights = np.asarray(ens.weights, float),
                copula_shrink = copula_shrink,
            )

            mu_mix = dep.mu

            std_mix = dep.std

            std_mix_med_sum += float(np.median(std_mix))

            if dep.corr.shape[0] > 1:

                mask = ~np.eye(dep.corr.shape[0], dtype=bool)

                abs_offdiag_corr_sum += float(np.mean(np.abs(dep.corr[mask])))

            else:

                abs_offdiag_corr_sum += 0.0

            diag_obs += 1

            boot_z = stationary_bootstrap(
                resid = resid_standardized,
                length = horizon,
                p = 1.0 / max(1, resid_block),
                rng = rng
            )

            if df_local > 2:

                boot_z *= get_t_scales(
                    H = horizon,
                    df = df_local,
                    rng = rng
                )

            eta_boot = boot_z * resid_std_hist

            z_hm = _draw_correlated_hm_z(
                dep = dep,
                hm_params_by_state = hm_params_by_state,
                states = S_tk,
                rng = rng,
            )

            eta_model = std_mix * z_hm

            eta_blended = alpha * eta_model + (1.0 - alpha) * eta_boot

            eta_blended *= gpath_tk

            r_path = mu_mix + eta_blended + jump_shocks

            path = cp * np.exp(np.cumsum(r_path))

            final_prices_buf[final_prices_count] = float(np.clip(path[-1], lb, ub))
         
            final_prices_count += 1

        sims_done = final_prices_count

        if sims_done >= n_sims_min and (batch_counter % mc_diag_every == 0):
        
            rets_partial = final_prices_buf[:final_prices_count] / cp - 1.0
        
            hw = _quantile_ci_half_widths(
                x = rets_partial, 
                qs = np.array([0.05, 0.50, 0.95], dtype = float)
            )
        
            if np.all(np.isfinite(hw)) and np.all(hw <= q_target):
        
                break

        if mean_cache_enabled and (sim_perf.cache_hits + sim_perf.cache_misses) >= 250:
         
            hit_rate = float(sim_perf.cache_hits) / max(sim_perf.cache_hits + sim_perf.cache_misses, 1)
         
            if hit_rate < 0.05:
         
                mean_cache_enabled = False
         
                mean_cache.clear()

    final_prices_arr = final_prices_buf[:final_prices_count]
    
    sim_perf.wall_sec_by_phase["simulation_loop"] = float(time.time() - t_sim_loop0)

    q05, q50, q95 = np.quantile(final_prices_arr, [0.05, 0.50, 0.95])

    rets = final_prices_arr / cp - 1.0

    se = float(np.std(rets, ddof=1)) if rets.size > 1 else float(np.std(rets))

    fit_meta = ens.meta or {}
    
    sim_perf.fit_calls = int(fit_meta.get("attempts", 0))

    logger.info(
        "%s diagnostics: alpha=%.2f, std_mix_med=%.4f, mean|corr_offdiag|=%.4f, fit attempts=%d converged=%d skipped=%d, warning(start/conv/other)=%d/%d/%d, n_sims=%d, return_se=%.4f, lags=%s, K=%d, orders=%s, block=%d, jump_q=%.3f",
        tk,
        float(alpha),
        float(std_mix_med_sum / max(diag_obs, 1)),
        float(abs_offdiag_corr_sum / max(diag_obs, 1)),
        int(fit_meta.get("attempts", 0)),
        int(fit_meta.get("converged", 0)),
        int(fit_meta.get("skipped", 0)),
        int(fit_meta.get("warnings_start", 0)),
        int(fit_meta.get("warnings_conv", 0)),
        int(fit_meta.get("warnings_other", 0)),
        int(final_prices_arr.size),
        se,
        tuple(exog_lags),
        int(fourier_k),
        list(hp.stack_orders),
        int(resid_block),
        float(jump_q),
    )

    if diag_out is not None:
     
        diag_out.update(
            {
                "fit_stats": dict(fit_meta),
                "n_sims_used": int(final_prices_arr.size),
                "target_q_ci_half_width": float(q_target),
                "copula_shrink": float(copula_shrink),
                "model_cov_reps": int(model_cov_reps),
                "perf_stats": asdict(sim_perf),
            }
        )

    if backtest_out is not None and tk not in backtest_out:
      
        try:
     
            backtest_out[tk] = build_backtest_summary(
                df_tk = df_tk,
                horizon = horizon,
                hp = hp,
                seed = int(rng_seed + 17),
            )
       
        except Exception as e:
       
            logger.warning("%s backtest summary failed (%s)", tk, e)

    mean_cache.clear()
   
    exog_cache.clear()
   
    cov_cache.clear()
   
    static_covs.clear()

    return {
        "low": float(q05),
        "avg": float(q50),
        "high": float(q95),
        "returns": float(np.mean(rets)),
        "se": se,
    }


def stationary_bootstrap(
    resid: np.ndarray,
    length: int,
    p: float, 
    rng
) -> np.ndarray:
    """
    Generate a stationary bootstrap (Politis–Romano) sequence from residuals.

    Algorithm
    ---------
    Start at a random index i; repeatedly draw geometric block lengths L ~ Geometric(p)
    (with support {1, 2, ...}), copy segments r[i : i+L] wrapping around as needed,
    and with probability p start a new block at a new random i, otherwise continue
    from i + L (circularly). Stops when `length` points are produced.

    Parameters
    ----------
    resid : numpy.ndarray
    length : int
    p : float
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, shape (length,)

    Why stationary bootstrap
    ------------------------
    Preserves short-range dependence without imposing strict block boundaries, producing
    more realistic multi-step residual patterns than i.i.d. resampling.
    """

    x = np.asarray(resid, float)
   
    x = x[np.isfinite(x)]
   
    n = int(x.size)

    if length <= 0:
   
        return np.zeros(0, dtype=float)

    if n == 0:
   
        return np.zeros(int(length), dtype=float)

    p_use = float(np.clip(p, 1e-6, 1.0))

    restart = rng.random(int(length)) < p_use
   
    restart[0] = True

    run_starts = rng.integers(0, n, size=int(restart.sum()))
   
    run_ids = np.cumsum(restart) - 1
   
    last_restart_idx = np.maximum.accumulate(np.where(restart, np.arange(int(length)), -1))
   
    run_pos = np.arange(int(length)) - last_restart_idx
   
    idx = (run_starts[run_ids] + run_pos) % n

    return x[idx]


def main() -> None:
    """
    End-to-end orchestration of the macro-to-price simulation and export workflow.

    Steps
    -----
    1) Load macro history and map tickers to countries; clean to weekly frequency.
   
    2) For each country, simulate macro delta scenarios via `simulate_macro_paths_for_country`.
   
    3) For each ticker:
   
    a) Build a weekly returns frame with aligned macro deltas and jump indicator.
   
    b) Simulate price paths via `simulate_price_paths_for_ticker`.
   
    4) Aggregate per-ticker summaries into a table and export via `export_results`.

    Side effects
    -----------
    Writes an Excel workbook containing p50/p5/p95 price scenarios and expected returns.
    Logs progress and warnings about insufficient history or failed macro fits.
    """
    
    run_start = time.time()

    if N_JOBS != 1:
    
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    macro = MacroData()

    r = macro.r

    tickers: List[str] = config.tickers

    forecast_period: int = FORECAST_WEEKS

    close = r.weekly_close

    latest_prices = r.last_price

    analyst = r.analyst

    lb = config.lbp * latest_prices

    ub = config.ubp * latest_prices

    logger.info("Importing macro history …")

    raw_macro = macro.assign_macro_history_non_pct().reset_index()

    raw_macro = raw_macro.rename(
        columns={"year": "ds"} if "year" in raw_macro.columns else {raw_macro.columns[1]: "ds"}
    )

    raw_macro["ds"] = raw_macro["ds"].dt.to_timestamp()

    effective_profile = _effective_runtime_profile()
  
    rs_runtime = _runtime_settings()
  
    opt_runtime = _optimization_profile()
  
    resolved_runtime = {k: float(v) for k, v in rs_runtime.items()}

    run_cfg = {
        "base_regressors": list(BASE_REGRESSORS),
        "n_jobs": int(N_JOBS),
        "auto_tune_hyperparams": bool(AUTO_TUNE_HYPERPARAMS),
        "n_sims_default": int(N_SIMS),
        "innov_dist_mode": str(INNOV_DIST_MODE),
        "runtime_profile_requested": str(RUNTIME_PROFILE).upper(),
        "runtime_profile_effective": effective_profile,
        "runtime_preset_resolved": resolved_runtime,
        "optimization_profile": asdict(opt_runtime),
        "target_countries_only": bool(TARGET_COUNTRIES_ONLY),
        "suppress_expected_convergence_warnings": bool(SUPPRESS_EXPECTED_CONVERGENCE_WARNINGS),
        "target_q_ci_half_width_requested": float(TARGET_Q_CI_HALF_WIDTH),
        "target_cov_rel_err_requested": float(TARGET_COV_REL_ERR),
        "target_q_ci_half_width_resolved": float(rs_runtime.get("q_ci_half_width", TARGET_Q_CI_HALF_WIDTH)),
        "target_cov_rel_err_resolved": float(rs_runtime.get("cov_target_rel_err", TARGET_COV_REL_ERR)),
        "speed_first_mode": bool(SPEED_FIRST_MODE),
        "skip_backtest_on_prod_run": bool(SKIP_BACKTEST_ON_PROD_RUN),
        "use_static_model_cov": bool(USE_STATIC_MODEL_COV),
        "mc_diag_every_batches_requested": int(MC_DIAG_EVERY_BATCHES),
        "mc_diag_every_batches_resolved": int(max(1, rs_runtime.get("mc_diag_every_batches", MC_DIAG_EVERY_BATCHES))),
        "low_memory_mode": bool(LOW_MEMORY_MODE),
    }
   
    run_cfg_hash = hashlib.sha256(
        json.dumps(run_cfg, sort_keys = True, separators=(",", ":"), default = str).encode("utf-8")
    ).hexdigest()

    dep_files = [
        Path(__file__),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("macro_data3.py"),
    ]
 
    dep_hashes: Dict[str, str] = {}
 
    for p in dep_files:
 
        try:
 
            dep_hashes[p.name] = _hash_file_sha256(p)
 
        except Exception:
 
            dep_hashes[p.name] = ""

    current_script_sha = dep_hashes.get(Path(__file__).name, "")
 
    if not current_script_sha:
 
        current_script_sha = _hash_file_sha256(
            path = Path(__file__)
        )

    signature = _compute_run_signature(
        tickers = tickers,
        forecast_period = forecast_period,
        close = close,
        raw_macro = raw_macro,
        run_cfg = run_cfg,
        dep_hashes_override = dep_hashes,
    )

    payload = None if FULL_RERUN else _load_run_cache(CACHE_FILE)

    current_macro_last_ts = str(pd.to_datetime(raw_macro["ds"].max()).isoformat()) if len(raw_macro) else ""
   
    partial_reuse_tickers: set[str] = set()

    if (not FULL_RERUN) and _is_cache_usable(
        payload = payload, 
        signature = signature
    ):
 
        logger.info("Cache hit at %s (created %s). Reusing outputs.", CACHE_FILE, payload.created_utc)
 
        if payload.df_out_records:
 
            df_out = pd.DataFrame(payload.df_out_records).set_index("Ticker")
 
        else:
 
            logger.warning("Cache payload has no tabular records.")
 
        logger.info("Run completed.")
 
        return

    if (not FULL_RERUN) and payload is not None:
    
        pm = payload.run_meta if isinstance(payload.run_meta, dict) else {}
    
        old_anchor = pm.get("ticker_close_anchor", {})
    
        can_partial = (
            str(pm.get("script_sha256", "")) == current_script_sha
            and str(pm.get("macro_last_ts", "")) == current_macro_last_ts
            and str(pm.get("run_profile", "")).upper() == effective_profile
            and str(pm.get("run_cfg_hash", "")) == run_cfg_hash
        )

        if can_partial and isinstance(old_anchor, dict):
       
            for tk in tickers:
       
                if tk not in payload.results_by_ticker:
       
                    continue
       
                if tk not in close.columns:
       
                    continue
       
                old_v = old_anchor.get(tk, np.nan)
       
                s = close[tk].dropna()
       
                cur_v = float(s.iloc[-1]) if s.size else np.nan
       
                if np.isfinite(cur_v) and np.isfinite(old_v) and abs(cur_v - float(old_v)) <= 1e-12:
       
                    partial_reuse_tickers.add(tk)

            if partial_reuse_tickers:
       
                logger.info(
                    "Partial cache reuse enabled for %d tickers (signature mismatch but compatible run metadata).",
                    len(partial_reuse_tickers),
                )

    logger.info("Cache miss or full rerun requested. Running full pipeline …")
  
    logger.info(
        "Parallel/BLAS: n_jobs=%d, profile=%s, backend_hint=%s, OMP=%s, MKL=%s, OPENBLAS=%s",
        int(N_JOBS),
        effective_profile,
        "threads" if LOW_MEMORY_MODE else "processes",
        os.environ.get("OMP_NUM_THREADS", ""),
        os.environ.get("MKL_NUM_THREADS", ""),
        os.environ.get("OPENBLAS_NUM_THREADS", ""),
    )

    country_map = {t: str(c) for t, c in zip(analyst.index, analyst["country"])}

    raw_macro["country"] = raw_macro["ticker"].map(country_map)

    macro_clean = raw_macro[["ds", "country"] + BASE_REGRESSORS].dropna()

    if TARGET_COUNTRIES_ONLY:
   
        target_countries = {country_map.get(tk) for tk in tickers}
   
        target_countries = {c for c in target_countries if c is not None and str(c) != "nan"}
   
        macro_clean = macro_clean[macro_clean["country"].isin(target_countries)]
   
        logger.info("Macro scope limited to requested ticker countries: %s", sorted(target_countries))

    logger.info("Simulating macro scenarios (ECVAR/BVAR/VAR) …")

    macro_phase_start = time.time()
   
    macro_cache_payload = {} if FULL_RERUN else _load_macro_cache(
        path = MACRO_CACHE_FILE
    )
   
    macro_cache_hits = 0
   
    macro_cache_misses = 0

    country_paths: Dict[str, Optional[np.ndarray]] = {}
   
    macro_hyperparams_by_country: Dict[str, Dict[str, Any]] = {}

    for ctry, dfc in macro_clean.groupby("country"):

        dfm_raw = (
            dfc.set_index("ds")[BASE_REGRESSORS]
               .sort_index()
               .resample("W")
               .mean()
               .ffill()
               .dropna()
        )

        macro_hp = fit_macro_hyperparams_from_levels(
            df_levels_weekly = dfm_raw,
            horizon = forecast_period,
        )
        macro_hyperparams_by_country[ctry] = macro_hp

        logger.info(
            "Country %s macro hyperparams: hist_weeks=%d, bvar_p=%d, lambdas=%s, niw=(%d, %.4f)",
            ctry,
            int(len(dfm_raw)),
            int(macro_hp["bvar_p"]),
            tuple(np.round(macro_hp["mn_lambdas"], 3)),
            int(macro_hp["niw_nu0"]),
            float(macro_hp["niw_s0_scale"]),
        )

        old_vals = (
            BVAR_P,
            MN_LAMBDA1, MN_LAMBDA2, MN_LAMBDA3, MN_LAMBDA4,
            NIW_NU0, NIW_S0_SCALE,
            MN_TUNE_L1_GRID, MN_TUNE_L2_GRID, MN_TUNE_L3_GRID, MN_TUNE_L4_GRID,
        )

        try:
            mkey = _macro_cache_key(
                country = str(ctry),
                horizon = int(forecast_period),
                n_sims = int(N_SIMS),
                macro_last_ts = current_macro_last_ts,
                macro_hp = macro_hp,
            )

            cached = macro_cache_payload.get(mkey) if isinstance(macro_cache_payload, dict) else None
           
            if (not FULL_RERUN) and isinstance(cached, np.ndarray):
             
                ok_shape = (
                    cached.ndim == 3
                    and cached.shape[0] == int(N_SIMS)
                    and cached.shape[1] == int(forecast_period)
                    and cached.shape[2] == int(lenBR)
                )
              
                if ok_shape:
              
                    country_paths[ctry] = cached
              
                    macro_cache_hits += 1
              
                    continue

            globals()["BVAR_P"] = int(macro_hp["bvar_p"])
            
            l1, l2, l3, l4 = macro_hp["mn_lambdas"]
            
            globals()["MN_LAMBDA1"] = float(l1)
            
            globals()["MN_LAMBDA2"] = float(l2)
            
            globals()["MN_LAMBDA3"] = float(l3)
            
            globals()["MN_LAMBDA4"] = float(l4)
            
            globals()["NIW_NU0"] = int(macro_hp["niw_nu0"])
            
            globals()["NIW_S0_SCALE"] = float(macro_hp["niw_s0_scale"])
            
            globals()["MN_TUNE_L1_GRID"] = tuple(sorted({max(0.05, l1 * 0.75), l1, min(1.50, l1 * 1.25)}))
            
            globals()["MN_TUNE_L2_GRID"] = tuple(sorted({max(0.10, l2 * 0.75), l2, min(1.50, l2 * 1.25)}))
            
            globals()["MN_TUNE_L3_GRID"] = tuple(sorted({max(0.50, l3 * 0.75), l3, min(2.50, l3 * 1.25)}))
            
            globals()["MN_TUNE_L4_GRID"] = tuple(sorted({max(20.0, l4 * 0.75), l4, min(400.0, l4 * 1.25)}))

            sims = simulate_macro_paths_for_country(
                df_levels_weekly = dfm_raw,
                steps = forecast_period,
                n_sims = N_SIMS,
                seed = int(rng_global.integers(1_000_000_000)),
            )
           
            country_paths[ctry] = sims
           
            macro_cache_payload[mkey] = sims
           
            _maybe_trim_cache(macro_cache_payload)
           
            macro_cache_misses += 1

        except Exception as e:
    
            logger.warning("Macro simulation failed for %s (%s). Using flat deltas.", ctry, e)
    
            country_paths[ctry] = np.zeros((N_SIMS, forecast_period, lenBR))

        finally:
          
            (
                bvar_old,
                l1_old, l2_old, l3_old, l4_old,
                nu_old, s0_old,
                g1_old, g2_old, g3_old, g4_old,
            ) = old_vals
            globals()["BVAR_P"] = bvar_old
            globals()["MN_LAMBDA1"] = l1_old
            globals()["MN_LAMBDA2"] = l2_old
            globals()["MN_LAMBDA3"] = l3_old
            globals()["MN_LAMBDA4"] = l4_old
            globals()["NIW_NU0"] = nu_old
            globals()["NIW_S0_SCALE"] = s0_old
            globals()["MN_TUNE_L1_GRID"] = g1_old
            globals()["MN_TUNE_L2_GRID"] = g2_old
            globals()["MN_TUNE_L3_GRID"] = g3_old
            globals()["MN_TUNE_L4_GRID"] = g4_old

    macro_deltas_by_ticker: Dict[str, pd.DataFrame] = {}

    requested_set = set(tickers)
  
    raw_macro_req = raw_macro[raw_macro["ticker"].isin(requested_set)]

    for tk, g in raw_macro_req.groupby("ticker", sort=False):

        lvl = (g.set_index("ds")[BASE_REGRESSORS]
                .sort_index()
                .resample("W-SUN").mean().ffill())

        d = _macro_stationary_deltas(
            df_levels = lvl
        ).reindex(close.index).ffill()

        macro_deltas_by_ticker[tk] = d

    del raw_macro_req
    
    del macro_clean

    macro_phase_sec = float(time.time() - macro_phase_start)

    try:
        
        _save_macro_cache_atomic(
            path = MACRO_CACHE_FILE, 
            payload = macro_cache_payload
        )
    
    except Exception as e:
    
        logger.warning("Failed to save macro cache (%s).", e)

    logger.info("Fitting SARIMAX ensemble and running Monte Carlo …")

    ticker_seeds: Dict[str, int] = {
        tk: int(rng_global.integers(1_000_000_000))
        for tk in tickers
    }

    rs_main = _runtime_settings()

    def _process_ticker(
        tk: str
    ) -> Tuple[
        str,
        Optional[Dict[str, float]],
        Optional[FittedHyperparams],
        Optional[BacktestSummary],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:

        tk_start = time.time()

        cp = latest_prices.get(tk, np.nan)

        if not np.isfinite(cp):
        
            logger.warning("No price for %s, skipping", tk)
        
            return tk, None, None, None, None, None

        dfp = pd.DataFrame({"ds": close.index, "price": close[tk].values})
        
        y = np.diff(np.log(dfp["price"].values))
        
        ticker_idx = dfp["ds"]
        
        dfm_base = macro_deltas_by_ticker.get(tk)
        
        if dfm_base is None:
        
            dfm_deltas = np.zeros((len(ticker_idx), lenBR), dtype=float)
        
        else:
        
            dfm_deltas = dfm_base.reindex(ticker_idx).ffill().bfill().values

        dfr = pd.DataFrame(
            np.column_stack([dfp["price"].values[1:], y, dfm_deltas[1:]]),
            index = pd.DatetimeIndex(ticker_idx[1:]),
            columns = ["price", "y"] + BASE_REGRESSORS
        )

        try:
       
            dfr.index.freq = to_offset(
                freq = "W-SUN"
            )
       
        except Exception:
       
            pass

        dfr = dfr.replace([np.inf, -np.inf], np.nan)
     
        first_valid = dfr["price"].first_valid_index()
     
        if first_valid is not None:
     
            dfr = dfr.loc[first_valid:]
     
        dfr = dfr.dropna(subset = ["y"] + BASE_REGRESSORS)

        if len(dfr) < forecast_period + 32:
      
            logger.warning("Insufficient history for %s, skipping", tk)
      
            return tk, None, None, None, None, None

        tk_seed = int(ticker_seeds.get(tk, RNG_SEED))

        rng_tk = np.random.default_rng(tk_seed)

        hp_trace: Dict[str, Any] = {}

        hp = fit_hyperparams_from_data(
            df_tk = dfr,
            horizon = forecast_period,
            seed = int(rng_tk.integers(1_000_000_000)),
            trace_out = hp_trace,
        ) if AUTO_TUNE_HYPERPARAMS else FittedHyperparams(
            stack_orders = list(STACK_ORDERS),
            exog_lags = tuple(EXOG_LAGS),
            fourier_k = int(FOURIER_K),
            cv_splits = int(np.clip(CV_SPLITS, 2, int(rs_main["max_cv_splits"]))),
            resid_block_len = int(RESID_BLOCK),
            student_df = int(T_DOF),
            jump_q = float(JUMP_Q),
            gpd_min_exceed = 30,
            hm_min_obs = int(HM_MIN_OBS),
            hm_clip_skew = float(HM_CLIP_SKEW),
            hm_clip_excess_kurt = float(HM_CLIP_EXCESS_KURT),
            copula_shrink = float(COPULA_SHRINK),
            model_cov_reps = int(rs_main["cov_reps_min"]),
            bvar_p = int(BVAR_P),
            mn_lambdas = (MN_LAMBDA1, MN_LAMBDA2, MN_LAMBDA3, MN_LAMBDA4),
            niw_nu0 = int(NIW_NU0),
            niw_s0_scale = float(NIW_S0_SCALE),
            n_sims_required = int(rs_main["n_sims_min"]),
            hm_enabled = True,
            hm_prior_weight = 0.5,
        )

        jump_ind, _ = make_jump_indicator_from_returns(
            y = dfr["y"],
            q = float(hp.jump_q),
        )
        dfr["jump_ind"] = jump_ind.astype(float)

        ctry = country_map.get(tk)
     
        macro_sims = country_paths.get(ctry)
     
        if macro_sims is None:
     
            macro_sims = np.zeros((N_SIMS, forecast_period, lenBR))

        lb_tk = float(lb[tk]) if np.isfinite(lb.get(tk, np.nan)) else -np.inf
     
        ub_tk = float(ub[tk]) if np.isfinite(ub.get(tk, np.nan)) else np.inf

        bt_holder: Dict[str, BacktestSummary] = {}
     
        diag_holder: Dict[str, Any] = {}
     
        bt_target = None if (SKIP_BACKTEST_ON_PROD_RUN and (not FULL_RERUN)) else bt_holder
     
        out = simulate_price_paths_for_ticker(
            tk = tk,
            df_tk = dfr[["price", "y"] + BASE_REGRESSORS + ["jump_ind"]],
            cp = float(cp),
            lb = lb_tk,
            ub = ub_tk,
            macro_sims = macro_sims,
            horizon = forecast_period,
            rng_seed = int(rng_tk.integers(1_000_000_000)),
            fitted_hp = hp,
            backtest_out = bt_target,
            diag_out = diag_holder,
        )

        bt = bt_holder.get(tk)
      
        if bt is None:
      
            if SKIP_BACKTEST_ON_PROD_RUN and (not FULL_RERUN):
      
                bt = BacktestSummary(
                    logscore = float("nan"),
                    rmse = float("nan"),
                    wis90 = float("nan"),
                    coverage90 = float("nan"),
                    tail_low_exceed = float("nan"),
                    tail_high_exceed = float("nan"),
                    n_folds = 0,
                )
       
            else:
              
                bt = build_backtest_summary(
                    df_tk = dfr[["price", "y"] + BASE_REGRESSORS + ["jump_ind"]],
                    horizon = forecast_period,
                    hp = hp,
                    seed = int(rng_tk.integers(1_000_000_000)),
                )

        print(
            f"{tk}: Avg: {out['avg']:.2f}, Low: {out['low']:.2f}, High: {out['high']:.2f}, "
            f"Return: {out['returns']:.4f}, MC Std: {out['se']:.4f}"
        )

        tdiag = asdict(
            TickerDiagnostics(
                fit_stats = dict(diag_holder.get("fit_stats", {})),
                backtest = (asdict(bt) if bt is not None else {}),
                hp = asdict(hp),
                runtime_sec = float(time.time() - tk_start),
            )
        )
      
        tdiag.update(diag_holder)

        return tk, out, hp, bt, tdiag, dict(hp_trace)

    results: Dict[str, Dict[str, float]] = {}
  
    fitted_hyperparams_by_ticker: Dict[str, FittedHyperparams] = {}
  
    backtest_summary_by_ticker: Dict[str, BacktestSummary] = {}
  
    ticker_diagnostics_by_ticker: Dict[str, Dict[str, Any]] = {}
  
    search_trace_by_ticker: Dict[str, Dict[str, Any]] = {}
  
    skipped_tickers: List[str] = []

    if payload is not None and partial_reuse_tickers:
  
        for tk in sorted(partial_reuse_tickers):
  
            try:
  
                results[tk] = dict(payload.results_by_ticker[tk])
  
            except Exception:
  
                continue

            hp_d = payload.fitted_hyperparams_by_ticker.get(tk)
  
            if isinstance(hp_d, dict):
  
                hp_obj = _dict_to_fitted_hyperparams(
                    d = hp_d
                )
  
                if hp_obj is not None:
  
                    fitted_hyperparams_by_ticker[tk] = hp_obj

            bt_d = payload.backtest_summary_by_ticker.get(tk)
          
            if isinstance(bt_d, dict):
          
                bt_obj = _dict_to_backtest_summary(
                    d = bt_d
                )
          
                if bt_obj is not None:
          
                    backtest_summary_by_ticker[tk] = bt_obj

            dg_d = payload.ticker_diagnostics_by_ticker.get(tk)
           
            if isinstance(dg_d, dict):
           
                ticker_diagnostics_by_ticker[tk] = dict(dg_d)
                
            tr_d = payload.search_trace_by_ticker.get(tk)
          
            if isinstance(tr_d, dict):
          
                search_trace_by_ticker[tk] = dict(tr_d)

    to_process = [tk for tk in tickers if tk not in partial_reuse_tickers]

    ticker_phase_start = time.time()

    effective_low_mem = bool(LOW_MEMORY_MODE or (SPEED_FIRST_MODE and int(N_JOBS) != 1 and len(to_process) >= 6))
 
    parallel_prefer = "threads" if effective_low_mem else "processes"

    for tk, res, hp, bt, tdiag, htrace in Parallel(n_jobs=N_JOBS, prefer=parallel_prefer)(delayed(_process_ticker)(tk) for tk in to_process):
     
        if res is None:
     
            skipped_tickers.append(tk)
     
            continue
     
        results[tk] = res
     
        if hp is not None:
     
            fitted_hyperparams_by_ticker[tk] = hp
     
        if bt is not None:
     
            backtest_summary_by_ticker[tk] = bt
     
        if tdiag is not None:
     
            ticker_diagnostics_by_ticker[tk] = tdiag
     
        if htrace is not None:
     
            search_trace_by_ticker[tk] = htrace

    ticker_phase_sec = float(time.time() - ticker_phase_start)

    rows = []

    for tk in tickers:
    
        if tk not in results:
    
            continue
    
        cp = latest_prices.get(tk)
    
        r_tk = results[tk]
    
        rows.append(
            {
                "Ticker": tk,
                "Current Price": cp,
                "Avg Price (p50)": r_tk["avg"],
                "Low Price (p5)": r_tk["low"],
                "High Price (p95)": r_tk["high"],
                "Returns": (r_tk["avg"] / cp - 1.0) if cp else np.nan,
                "SE": r_tk["se"],
            }
        )

    if not rows:
     
        logger.warning("No results produced.")
     
        return

    df_out = pd.DataFrame(rows).set_index("Ticker")
    
    print(df_out)


    def _safe_num(
        v: Any
    ) -> float:
   
        try:
   
            f = float(v)
   
            return f if np.isfinite(f) else 0.0
   
        except Exception:
   
            return 0.0

    warning_summary = {
        "warnings_start": int(np.nansum([_safe_num(d.get("fit_stats", {}).get("warnings_start", 0)) for d in ticker_diagnostics_by_ticker.values()])),
        "warnings_conv": int(np.nansum([_safe_num(d.get("fit_stats", {}).get("warnings_conv", 0)) for d in ticker_diagnostics_by_ticker.values()])),
        "warnings_other": int(np.nansum([_safe_num(d.get("fit_stats", {}).get("warnings_other", 0)) for d in ticker_diagnostics_by_ticker.values()])),
        "attempts": int(np.nansum([_safe_num(d.get("fit_stats", {}).get("attempts", 0)) for d in ticker_diagnostics_by_ticker.values()])),
        "converged": int(np.nansum([_safe_num(d.get("fit_stats", {}).get("converged", 0)) for d in ticker_diagnostics_by_ticker.values()])),
    }

    ticker_close_anchor = {
        tk: (float(close[tk].dropna().iloc[-1]) if tk in close.columns and close[tk].dropna().size else np.nan)
        for tk in tickers
    }

    payload = RunCachePayload(
        schema_version = int(CACHE_SCHEMA_VERSION),
        signature = signature,
        created_utc = datetime.now(timezone.utc).isoformat(),
        results_by_ticker = results,
        df_out_records = df_out.reset_index().to_dict(orient = "records"),
        fitted_hyperparams_by_ticker = {k: asdict(v) for k, v in fitted_hyperparams_by_ticker.items()},
        backtest_summary_by_ticker = {k: asdict(v) for k, v in backtest_summary_by_ticker.items()},
        ticker_diagnostics_by_ticker = ticker_diagnostics_by_ticker,
        warning_summary = warning_summary,
        search_trace_by_ticker = search_trace_by_ticker,
        run_meta = {
            "runtime_sec": float(time.time() - run_start),
            "tickers_requested": list(tickers),
            "tickers_completed": sorted(results.keys()),
            "tickers_skipped": sorted(skipped_tickers),
            "ticker_seeds": ticker_seeds,
            "ticker_close_anchor": ticker_close_anchor,
            "macro_hyperparams_by_country": macro_hyperparams_by_country,
            "macro_last_ts": current_macro_last_ts,
            "script_sha256": current_script_sha,
            "run_profile": effective_profile,
            "run_profile_requested": str(RUNTIME_PROFILE).upper(),
            "run_cfg_hash": run_cfg_hash,
            "run_cfg": run_cfg,
            "cache_file": str(CACHE_FILE),
            "full_rerun": bool(FULL_RERUN),
            "perf_summary": {
                "macro_phase_sec": macro_phase_sec,
                "ticker_phase_sec": ticker_phase_sec,
                "partial_reuse_tickers": int(len(partial_reuse_tickers)),
                "processed_tickers": int(len(to_process)),
                "parallel_backend": parallel_prefer,
                "low_memory_mode": bool(effective_low_mem),
                "macro_cache_hits": int(macro_cache_hits),
                "macro_cache_misses": int(macro_cache_misses),
            },
        },
    )

    try:
  
        _save_run_cache_atomic(
            path = CACHE_FILE, 
            payload = payload
        )
  
        logger.info("Saved cache to %s", CACHE_FILE)
  
    except Exception as e:
  
        logger.warning("Failed to save cache (%s)", e)
  
    export_results(
        sheets = {"SARIMAX Monte Carlo": df_out},
        output_excel_file = config.MODEL_FILE,
    )
  
    logger.info("Run completed.")


if __name__ == "__main__":
    
    main()
