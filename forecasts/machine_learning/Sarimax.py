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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pandas.tseries.frequencies import to_offset

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from scipy.stats import genpareto  
import multiprocessing
import time
import warnings

from functions.export_forecast import export_results
from data_processing.macro_data import MacroData
import config


BASE_REGRESSORS: List[str] = ["Interest", "Cpi", "Gdp", "Unemp"]

lenBR = len(BASE_REGRESSORS)

STACK_ORDERS: List[Tuple[int,int,int]] = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2), (2, 0, 1)]

RIDGE_EPS = 1e-6  

EXOG_LAGS: tuple[int, ...] = (0, 1, 2)  

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

_MAX_CACHE = 512

FORECAST_WEEKS: int = 52

N_SIMS: int = 1000

JUMP_Q: float = 0.97

N_JOBS: int = -1

CV_SPLITS: int = 3

RESID_BLOCK: int = 4 

T_DOF: int = 5             

BVAR_P: int = 2          

MN_LAMBDA1: float = 0.2 

MN_LAMBDA2: float = 0.5

MN_LAMBDA3: float = 1.0 

MN_LAMBDA4: float = 100.0

NIW_NU0: int = 5

NIW_S0_SCALE: float = 0.1  

RNG_SEED: int = 42

rng_global = np.random.default_rng(RNG_SEED)

_FOURIER_CACHE: Dict[Tuple[int,int,int], np.ndarray] = {}

_T_SCALES_CACHE = {}

_fit_memo_np: Dict[tuple, Ensemble] = {}         
           
_fit_by_order_cache_np: Dict[tuple, object] = {}               


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

        last = self._last_t.get(
            key = key, 
            default = 0.0
        )

        if now - last < self._window:

            return False

        self._last_t[key] = now

        return True


def configure_logger() -> logging.Logger:
    """Create a process-aware logger with sensible defaults.

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


def _maybe_trim_cache(
    d: dict
):
    """
    Bound the size of an in-memory cache and clear it if the limit is exceeded.

    Parameters
    ----------
    d : dict
        Cache dictionary to check and potentially clear.

    Notes
    -----
    Caches for SARIMAX fits and Fourier blocks accelerate repeated calls but can grow
    unbounded across instruments and folds. Clearing opportunistically avoids memory blow-ups.
    """

    if len(d) > _MAX_CACHE:

        d.clear()


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

        lags = dX_lags.copy()  
      
        out = np.zeros((steps, k))

        for t in range(steps):

            pred = c.copy()
      
            for l in range(1, p + 1):
      
                pred += Al[l - 1] @ lags[-l]
      
            innov = Lchol @ z[t]
           
            if scale_path is not None:
            
                innov = innov * scale_path[t]
                  
            dx_next = pred + innov
      
            out[t] = dx_next
      
            lags = np.vstack([lags[1: ], dx_next])
      
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

    T, k = resid.shape

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
      
        c, a = float(beta[0]), float(beta[1])
      
        mu[j] = c / (1.0 - a) if abs(1.0 - a) > 1e-6 else np.mean(hj)
      
        phi[j] = np.clip(a, -0.98, 0.98)
      
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

    T = loglik_t.shape[0]

    logpi = np.log(pi + 1e-16)

    alpha = np.zeros((T,2))
  
    c = np.zeros(T)
  
    alpha[0] = logpi + loglik_t[0]
  
    m = np.max(alpha[0]); alpha[0] = np.exp(alpha[0] - m); c[0] = np.sum(alpha[0]); alpha[0] /= c[0]
  
    loglik = m + np.log(c[0] + 1e-300)

    for t in range(1, T):
   
        a = alpha[t-1] @ P
   
        a = np.maximum(a, 1e-300)
   
        alpha[t] = a * np.exp(loglik_t[t])
   
        ct = np.sum(alpha[t])
        
        alpha[t] /= (ct + 1e-300)
   
        loglik += np.log(ct + 1e-300)

    beta = np.ones((T, 2))
  
    gamma = np.zeros((T, 2))
  
    xi = np.zeros((T - 1, 2, 2))
  
    gamma[-1] = alpha[-1] 

    for t in range(T-2, -1, -1):
      
        tmp = P * np.exp(loglik_t[t + 1]) * beta[t + 1]
      
        denom = np.maximum(np.sum(tmp, axis = 1, keepdims = True), 1e-300)
      
        beta[t] = (tmp @ np.ones(2)) / denom.squeeze()

        z = alpha[t] * beta[t]
      
        gamma[t] = z / np.maximum(z.sum(), 1e-300)

        denom2 = np.maximum((alpha[t][:, None] * P * np.exp(loglik_t[t + 1])[None, :] * beta[t + 1][None, :]).sum(), 1e-300)
      
        xi[t] = (alpha[t][:, None] * P * np.exp(loglik_t[t + 1])[None, :] * beta[t + 1][None, :]) / denom2

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
  
    T, k = resid.shape

    Sigma = np.cov(resid, rowvar = False)
  
    Sigma = (Sigma + Sigma.T) * 0.5 + 1e-9 * np.eye(k)
  
    Sinv = np.linalg.inv(Sigma)
  
    sign, logdetS = np.linalg.slogdet(Sigma)

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
    
                    var = (lambda1 * lambda2 / (l ** lambda3)) ** 2 * (sig[i] / sig[j])
    
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
    B0: np.ndarray, 
    V0_diag: np.ndarray,
    S0: np.ndarray,
    nu0: int
) -> Tuple[BVARPosterior, float]:
    """
    Compute the MN-IW posterior for a VAR and the marginal log evidence (up to constants).

    Given Y (T×k) and X (T×m), with prior:

        B | Σ ~ MN(B0, V0, Σ)   and   Σ ~ IW(S0, ν0),

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
    B0 : numpy.ndarray, shape (m, k)
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
    
    LK, logdet_K = _chol_logdet_psd(
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast construction of (B0, diag(V0)) for the Minnesota prior.

    Identical in meaning to `_mn_prior`, but returns only the diagonal of V0, enabling
    efficient posterior algebra with diagonal priors.

    Parameters
    ----------
    dX : pandas.DataFrame
    p : int
    lambda1, lambda2, lambda3, lambda4 : float

    Returns
    -------
    B0 : numpy.ndarray, shape (1 + k p, k)
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

                    var = (lambda1 * lambda2 / (l ** lambda3)) ** 2 * (sig[i] / sig[j])

                V0_diag_eq[i, pos] = var

    V0_diag = np.mean(V0_diag_eq, axis = 0)  

    B0 = np.zeros((1 + k * p, k))

    return B0, V0_diag


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

    best_logev -np.inf
    
    best_post = None
    
    best_lmb =  None
   
    for l1 in grid_l1:
    
        for l2 in grid_l2:
     
            for l3 in grid_l3:
     
                for l4 in grid_l4:
     
                    B0, V0_diag = _mn_prior_diag(
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
                            B0 = B0, 
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
  
                    _, V0_diag = _mn_prior_diag(
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
    F_fourier: np.ndarray,              
    col_order: list[str],          
    scaler: StandardScaler,
    lags: tuple[int, ...] = EXOG_LAGS,
    jump: Optional[np.ndarray] = None,   
) -> np.ndarray:
    """
    Construct a future exogenous design matrix for SARIMAX in **exact training column order**.

    Inputs
    ------
    hist_tail_vals : numpy.ndarray, shape (context_rows, k_macro)
        The final `context_rows` rows of the base macro regressors in the training order.
    
    macro_path : numpy.ndarray, shape (H, k_macro)
        Simulated macro deltas for the forecast horizon H.
   
    F_fourier : numpy.ndarray, shape (H, 2K)
        Fourier seasonal block with sin/cos pairs for K harmonics and period = 52 (weekly).
   
    col_order : list[str]
        Exact SARIMAX training column order (e.g., ["Cpi_L0", "Cpi_L1", ..., "jump_ind",
        "fourier_sin_1", "fourier_cos_1", ...]).
   
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler from the training design.
   
    lags : tuple[int, ...]
        Lag indices L used when fitting (e.g., (0, 1, 2)).
   
    jump : numpy.ndarray, optional, shape (H,)
        Bernoulli jump indicators to include as an exogenous column if present.

    Construction
    ------------
   
    1) Vertically stack `hist_tail_vals` and `macro_path` to create a contiguous series.
   
    2) For each unique lag L in `lags`, slice the lagged H×k block needed for the horizon.
   
    3) For each name in `col_order`, fill the corresponding column:
   
    - "{Base}_L{L}" → the appropriate lag block and variable column,
   
    - "jump_ind" → `jump` (or zeros),
   
    - "fourier_sin_k", "fourier_cos_k" → from `F_fourier`.
   
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

    series = np.vstack([hist_tail_vals, macro_path])
   
    base_rows_start = series.shape[0] - H

    uniq_lags = sorted(set(lags))
   
    lag_blocks = {L: series[base_rows_start - L: base_rows_start - L + H, :] for L in uniq_lags}

    X = np.empty((H, len(col_order)), dtype = float)

    for j, name in enumerate(col_order):
   
        if name == "jump_ind":
   
            if jump is None:
   
                X[:, j] = 0.0
   
            else:
   
                X[:, j] = jump.astype(float)
   
            continue

        if name.startswith("fourier_sin_"):
           
            kf = int(name.split("_")[-1]) - 1
      
            X[:, j] = F_fourier[:, 2 * kf + 0]
      
            continue
      
        if name.startswith("fourier_cos_"):
      
            kf = int(name.split("_")[-1]) - 1
      
            X[:, j] = F_fourier[:, 2 * kf + 1]
      
            continue

        if "_L" in name:

            base, Ls = name.rsplit("_L", 1)

            L = int(Ls)

            try:

                r_idx = BASE_REGRESSORS.index(base)

            except ValueError:

                X[:, j] = 0.0

                continue

            X[:, j] = lag_blocks[L][:, r_idx]

        else:

            X[:, j] = 0.0

    return scaler.transform(X)


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

    w = np.exp(-0.5 * (ics - np.min(ics)))
  
    if not np.isfinite(w).any():
  
        w = np.ones_like(ics)
  
    w = w / w.sum()
  
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
        T, k = resid.shape
    
        Sinv = np.linalg.inv(ms.Sigma)
    
        sign, logdetS = np.linalg.slogdet(ms.Sigma)
    
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
   
        Tm, kdim = resid_for_vol.shape
   
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
        
    logger.info("Macro candidates: %s with weights %s", labels, np.round(w, 3))
    
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
   
    for (label, model, aux, _ic) in candidates:
   
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
   
    w = np.exp(-0.5 * (scores - scores.min()))
   
    w /= w.sum()

    w_full = np.zeros(len(candidates), float)
   
    for i, (lab, *_rest) in enumerate(candidates):
   
        if lab in labels:
   
            j = labels.index(lab)
   
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


def _fourier_block(
    H: int, 
    period: int, 
    K: int
) -> np.ndarray:
    """
    Generate a deterministic Fourier seasonal block for weekly seasonality.

    For horizon H, period P (typically 52), and K harmonics, returns an H×(2K) matrix:

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

    key = (H, period, K)

    blk = _FOURIER_CACHE.get(key)

    if blk is None:

        t = np.arange(H)

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
    lags: tuple[int, ...] = (0, 1, 2),
    add_fourier: bool = True,
    K: int = 2
) -> pd.DataFrame:
    """
    Assemble an exogenous design matrix from lagged macro regressors, optional jump indicator,
    and optional Fourier seasonal terms.

    Construction
    ------------
    1) For each lag L in `lags`, add a block with columns "{name}_L{L}".
   
    2) If "jump_ind" exists in the input frame, append it as a separate column.
   
    3) If `add_fourier=True`, append sin/cos pairs from `_fourier_block`.
   
    4) Drop rows with any NaNs (due to lagging) and enforce a weekly frequency tag.

    Parameters
    ----------
    df : pandas.DataFrame
    base : list[str]
    lags : tuple[int, ...], default (0, 1, 2)
    add_fourier : bool, default True
    K : int, default 2

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

    if "jump_ind" in df.columns:

        blocks.append(df[["jump_ind"]])

    if add_fourier:

        F = _fourier_block(len(df.index), 52, K)

        cols = []

        for kf in range(1, K+1):

            cols += [f"fourier_sin_{kf}", f"fourier_cos_{kf}"]

        blocks.append(pd.DataFrame(F, index = df.index, columns = cols))

    if not blocks:
   
        return pd.DataFrame(index=df.index)

    X = pd.concat(blocks, axis=1, copy=False)
   
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


def _fit_sarimax_by_orders_cached_np(
    y: pd.Series,
    X_arr: np.ndarray,
    index: pd.DatetimeIndex,
    orders: List[Tuple[int, int, int]],
    col_order: List[str],
    time_varying: bool = False,
) -> Dict[Tuple[int, int, int], object]:
    out: Dict[Tuple[int, int, int], object] = {}
    """
    Fit multiple SARIMAX(p, d=0, q) candidates with exogenous regressors (NumPy path) and cache results.

    Model
    -----
    For observed return y_t and exogenous vector x_t:

        y_t = μ + Σ_{i=1..p} φ_i y_{t−i} + Σ_{j=1..q} θ_j ε_{t−j} + x_t' β + ε_t,
      
        ε_t ~ N(0, σ^2).

    The implementation uses statsmodels' `SARIMAX` with:
    
    - seasonal_order = (0, 0, 0, 0),
    
    - time-varying regression disabled by default,
    
    - stationarity and invertibility enforced.

    Parameters
    ----------
    y : pandas.Series
    X_arr : numpy.ndarray
    index : pandas.DatetimeIndex
    orders : list[tuple[int, int, int]]
    col_order : list[str]
    time_varying : bool, default False

    Returns
    -------
    dict
        Mapping order → fitted results (cached by a stable key).

    Why ACF/ARMA terms
    ------------------
    AR/MA components absorb serial correlation left after exogenous effects, improving
    density forecasts and calibration.
    """

    freq = getattr(index, "freqstr", None) or pd.infer_freq(index) or "W-SUN"
   
    base_key = (
        "NP",                     
        len(y),
        int(index[-1].value) if len(index) else 0,
        X_arr.shape[1],
        hash(tuple(col_order)),  
    )

    for od in orders:
       
        key = (*base_key, hash(od))
       
        if key in _fit_by_order_cache_np:
       
            out[od] = _fit_by_order_cache_np[key]
       
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
                time_varying_regression = time_varying,
                mle_regression = not time_varying,
                dates = index,
                freq = freq,
            )
            
            with warnings.catch_warnings():
            
                warnings.simplefilter("ignore", ConvergenceWarning)
            
                res = mdl.fit(disp = False, method = "lbfgs", maxfun = 5000)
            
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
   
        orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2), (2, 0, 1)]

    freq = getattr(index, "freqstr", None) or pd.infer_freq(index) or "W-SUN"

    fits, crit = [], []

    for od in orders:

        try:

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
          
            with warnings.catch_warnings():
          
                warnings.simplefilter("ignore", ConvergenceWarning)
          
                res = mdl.fit(disp = False, method = "lbfgs", maxfun = 5000)
        
            fits.append(res)
        
            kparams = res.params.size
        
            n = len(res.model.endog)
        
            aicc = res.aic + (2*kparams*(kparams+1))/max(n-kparams-1, 1)
        
            crit.append(aicc)
        
        except Exception:
        
            continue
    if not fits:
    
        raise RuntimeError("No SARIMAX fits succeeded (array path).")

    aicc = np.asarray(crit, float)
  
    w = np.exp(-0.5 * (aicc - aicc.min()))
  
    w /= w.sum()
  
    return Ensemble(
        fits = fits,
        weights = w
    )


def _fit_sarimax_candidates_cached_np(
    y: pd.Series,
    X_arr: np.ndarray,
    index: pd.DatetimeIndex,
    col_order: List[str],
    time_varying: bool = False,
    orders: Optional[List[Tuple[int, int, int]]] = None,
) -> Ensemble:
    """
    Return a cached SARIMAX ensemble for the given training block; fit and cache if necessary.

    Cache key
    ---------
    A stable tuple comprising:
    ("NP", len(y), last_index_value, X_arr.shape[1], hash(col_order), hash(orders)).

    Parameters
    ----------
    y, X_arr, index, col_order, time_varying, orders
        As in `_fit_sarimax_candidates_np`.

    Returns
    -------
    Ensemble
    """

    key = (
        "NP",
        len(y),
        int(index[-1].value) if len(index) else 0,
        X_arr.shape[1],
        hash(tuple(col_order)),
        hash(tuple(orders)) if orders is not None else 0,
    )
  
    if key in _fit_memo_np:
  
        return _fit_memo_np[key]
  
    ens = _fit_sarimax_candidates_np(
        y = y, 
        X_arr = X_arr, 
        index = index,
        time_varying = time_varying,
        orders = orders
    )
  
    _fit_memo_np[key] = ens
  
    _maybe_trim_cache(
        d = _fit_memo_np
    )
  
    return ens


def learn_stacking_weights(
    df: pd.DataFrame,
    regressors: List[str],
    orders: List[Tuple[int,int,int]],
    n_splits: int,
    horizon: int,
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

    for kfold in range(n_splits):
    
        train_end = (kfold + 1) * fold_size
    
        train = df.iloc[:train_end].copy()
    
        valid = df.iloc[train_end : train_end + horizon].copy()
    
        if len(valid) < horizon:
    
            break

        X_tr = make_exog(
            df = train, 
            base = regressors,
            lags = EXOG_LAGS,
            add_fourier = True,
            K = FOURIER_K
        )
    
        y_tr = train["y"].loc[X_tr.index]

        max_lag = max(EXOG_LAGS) if EXOG_LAGS else 0
        
        context_rows = max_lag + 2
        
        hist_tail_vals = train[regressors].iloc[-context_rows:].values       
        
        macro_path = valid[regressors].values                              
        
        add_jump_arr = valid["jump_ind"].values if "jump_ind" in valid.columns else None

        sc = StandardScaler().fit(X_tr.values)
       
        X_tr_arr = sc.transform(X_tr.values)
       
        col_order = list(X_tr.columns)
       
        F_val = _fourier_block(
            H = horizon, 
            period = 52, 
            K = FOURIER_K
        )

        X_vas = build_future_exog(
            hist_tail_vals = hist_tail_vals,
            macro_path = macro_path,
            F_fourier = F_val,
            col_order = col_order,
            scaler = sc,
            lags = EXOG_LAGS,
            jump = (add_jump_arr.astype(float) if add_jump_arr is not None else None),
        )

        fits = _fit_sarimax_by_orders_cached_np(
            y = y_tr, 
            X_arr = X_tr_arr, 
            index = X_tr.index, 
            orders = orders, 
            col_order = col_order
        )
        
        valid_orders |= set(fits.keys())

        row = {}
      
        for od in fits.keys():
      
            f = fits[od].get_forecast(steps = horizon, exog = X_vas)
      
            mu_k = f.predicted_mean
      
            row[od] = float(np.sum(mu_k))
      
        rows_F.append(row)
      
        y_vec.append(float(np.sum(valid["y"].values)))

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

    arch_lm, arch_lm_p, _, _ = het_arch(
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
    - Near-Gaussian kurtosis (≤ 3.5): higher df (≈ 12).
    - Moderate excess kurtosis (≤ 5): medium df (≈ 8).
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

    if kurt <= 3.5:

        return 12

    if kurt <= 5.0:

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
   
        return np.nan, np.array([]), np.array([]), np.nan, []

    fold_size = (N - horizon) // (n_splits + 1)
   
    if fold_size < 1:
   
        return np.nan, np.array([]), np.array([]), np.nan, []

    v_fore_list: List[float] = []

    for kfold in range(n_splits):

        train_end = (kfold + 1) * fold_size

        train = df.iloc[:train_end].copy()

        valid = df.iloc[train_end : train_end + horizon].copy()

        if len(valid) < horizon:

            break

        max_lag = max(EXOG_LAGS) if EXOG_LAGS else 0

        context_rows = max_lag + 2

        hist_tail_vals = train[regressors].iloc[-context_rows:].values          

        macro_path = valid[regressors].values                               

        add_jump_arr = valid["jump_ind"].values if "jump_ind" in valid.columns else None

        X_tr = make_exog(
            df = train, 
            base = regressors, 
            lags = EXOG_LAGS, 
            add_fourier = True,
            K = FOURIER_K
        )
       
        y_tr = train["y"].loc[X_tr.index]
       
        sc = StandardScaler().fit(X_tr.values)
       
        X_tr_arr = sc.transform(X_tr.values)
       
        col_order = list(X_tr.columns)
       
        F_val = _fourier_block(
            H = horizon, 
            period = 52, 
            K = FOURIER_K
        )

        X_vas = build_future_exog(
            hist_tail_vals = hist_tail_vals,
            macro_path = macro_path,
            F_fourier = F_val,
            col_order = col_order,
            scaler = sc,
            lags = EXOG_LAGS,
            jump = (add_jump_arr.astype(float) if add_jump_arr is not None else None),
        )

        ens = _fit_sarimax_candidates_cached_np(
            y = y_tr, 
            X_arr = X_tr_arr,
            index = X_tr.index, 
            col_order = col_order
        )

        mu_stack, var_stack = [], []
        
        for res, w in zip(ens.fits, ens.weights):
        
            f = res.get_forecast(steps = horizon, exog = X_vas)
        
            mu_k = f.predicted_mean
        
            var_k = np.asarray(f.var_pred_mean)
        
            mu_stack.append(mu_k)
        
            var_stack.append(var_k)

        mu_stack = np.asarray(mu_stack)                 
      
        var_stack = np.asarray(var_stack)               
      
        wv = np.asarray(ens.weights).reshape(-1, 1)      
      
        mu_mix = (wv * mu_stack).sum(axis = 0)            
      
        var_mix = (wv * (var_stack + (mu_stack - mu_mix)**2)).sum(axis = 0)  
      
        v_fore_list.append(float(np.sum(np.maximum(var_mix, 1e-12))))

    v_fore_mean = float(np.mean(v_fore_list)) if v_fore_list else np.nan

    if ens_full is None:
        
        if y_full is None or X_fulls is None or ens_full is None:
        
            X_full = make_exog(
                df = df,
                base = regressors,
                lags = EXOG_LAGS, 
                add_fourier = True, 
                K = FOURIER_K
            )
        
            y_full = df["y"].loc[X_full.index]
        
            sc_full = StandardScaler().fit(X_full.values)
        
            X_fulls = pd.DataFrame(sc_full.transform(X_full.values), index = X_full.index, columns = X_full.columns)
            
        else:
            ens_full = _fit_sarimax_candidates_cached_np(
                y = y_full, 
                X_arr = X_full_arr, 
                index = y_full.index, 
                col_order = col_order
            )

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
    q: float = JUMP_Q
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

    if resid.size < 100:  
    
        return None
    
    x = np.abs(resid)
    
    thr = float(np.quantile(x, q))
    
    mask = x > thr
    
    excess = x[mask] - thr
    
    if excess.size < 30:
    
        return None
    
    try:
    
        xi, loc, beta = genpareto.fit(excess, floc = 0.0)
    
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
    
        xi, loc, beta = genpareto.fit(sample, floc=0.0)
    
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
    q: float = JUMP_Q
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
            q = q
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
   
        neg_flags = rng.uniform(size=m) < params["p_neg"]
   
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


def simulate_price_paths_for_ticker(
    tk: str,
    df_tk: pd.DataFrame,  
    cp: float,
    lb: float,
    ub: float,
    macro_sims: Optional[np.ndarray],
    horizon: int,
    rng_seed: int,
) -> Dict[str, float]:
    """
    Simulate price scenarios for a single ticker using a SARIMAX ensemble with macro
    exogenous paths, regime-conditional jumps, and blended innovation noise.

    Pipeline
    --------
    
    1) Build historical exogenous design X_hist (lagged macros, jump indicator, Fourier)
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
    
    d) Forecast the SARIMAX ensemble H steps ahead; form mixture mean μ_mix and variance Var_mix.
    
    e) Generate noise as:
    
            η_model_t ~ N(0, Var_mix_t),
    
            η_bootstrap_t from stationary bootstrap of residuals, fattened by t-scales,
    
            η_t = α η_model_t + (1 − α) η_bootstrap_t,
    
        then scale by gpath_t (regime volatility).
    
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
   
    n_sims = macro_sims.shape[0] if macro_sims is not None else N_SIMS
   
    k_macro = lenBR

    X_hist = make_exog(
        df = df_tk, 
        base = BASE_REGRESSORS, 
        lags = EXOG_LAGS, 
        add_fourier = True,
        K = FOURIER_K
    )
   
    y_hist = df_tk["y"].loc[X_hist.index]
   
    sc = StandardScaler().fit(X_hist.values)

    zero_var = np.isclose(sc.scale_, 0)
   
    if zero_var.any():
   
        keep_cols = X_hist.columns[~zero_var]
   
        logger.warning("%s: dropping %d constant exog cols: %s", tk, int(zero_var.sum()), list(X_hist.columns[zero_var]))
   
        X_hist = X_hist[keep_cols]
   
        y_hist = y_hist.loc[X_hist.index]
   
        sc = StandardScaler().fit(X_hist.values)

    col_order = list(X_hist.columns)
   
    X_hist_arr = sc.transform(X_hist.values)

    F_fourier = _fourier_block(
        H = horizon,
        period = 52, 
        K = FOURIER_K
    )
   
    max_lag = max(EXOG_LAGS) if EXOG_LAGS else 0
   
    context_rows = max_lag + 2
    
    hist_tail_base = df_tk.loc[:, BASE_REGRESSORS].to_numpy()[-context_rows:, :]

    ens = _fit_sarimax_candidates_cached_np(
        y = y_hist, 
        X_arr = X_hist_arr, 
        index = X_hist.index, 
        col_order = col_order
    )

    resid, rs_std_arr, v_fore_mean = rolling_origin_cv_rmse_return_sum(
        df = df_tk,
        regressors = BASE_REGRESSORS,
        n_splits = CV_SPLITS,
        horizon = horizon,
        y_full = y_hist,
        X_full_arr = X_hist_arr,
        col_order = col_order,
        ens_full = ens,
    )
    
    rs_std = float(rs_std_arr[0]) if rs_std_arr.size else (np.std(resid) * np.sqrt(horizon))
    
    alpha = calibrate_alpha_from_cv(
        v_fore_mean = v_fore_mean,
        rs_std = rs_std
    )

    resid_c = resid - np.mean(resid) if resid.size else np.array([0.0])
   
    diag = residual_diagnostics(
        resid = resid_c, 
        lags = 12
    )
    
    df_local = choose_student_df_from_diag(
        diag = diag
    )

    w_stack, stack_orders = learn_stacking_weights(
        df = df_tk, 
        regressors = BASE_REGRESSORS,
        orders = STACK_ORDERS,
        n_splits = CV_SPLITS, 
        horizon = horizon
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

    jump_hist = df_tk["jump_ind"].loc[X_hist.index].to_numpy(dtype = int)
  
    if ms_tk is not None and gamma_tk is not None and gamma_tk.shape[0] == jump_hist.shape[0]:
       
        p_jump_by_state, gpd_params_by_state = estimate_state_conditional_jump_params(
            resid_1d = resid_c[-len(jump_hist):],  
            jump_ind = jump_hist,
            gamma = gamma_tk[-len(jump_hist):],
            q = JUMP_Q,
        )
        
    else:
       
        p_jump_uncond = float(np.mean(jump_hist)) if jump_hist.size else 0.0
       
        p_jump_by_state = np.array([p_jump_uncond, p_jump_uncond], float)
       
        gpd_params_by_state = [
            fit_gpd_tail(
                resid = resid_c, 
                q = JUMP_Q
            ), 
            fit_gpd_tail(
                resid = resid_c, 
                q = JUMP_Q
            )
        ]

    use_stack = bool(w_stack)
   
    if use_stack:
   
        fits_by_order = _fit_sarimax_by_orders_cached_np(
            y = y_hist, 
            X_arr = X_hist_arr,
            index = X_hist.index, 
            orders = list(stack_orders), 
            col_order = col_order
        )
   
        s = sum(w_stack.get(od, 0.0) for od in stack_orders)
   
        if s <= 0:
   
            use_stack = False
   
        else:
   
            for od in stack_orders:
   
                w_stack[od] = w_stack[od] / s

    final_prices = np.empty(n_sims, dtype = float)

    eps_scale = 1.0  

    w_vec = []

    if use_stack:

        for od in stack_orders:

            if od not in fits_by_order:

                continue

            w_vec.append(w_stack[od])

        w_vec = np.array(w_vec, dtype=float).reshape(-1,1)

    else:

        w_vec = np.asarray(ens.weights, dtype=float).reshape(-1,1)        
    
    for i in range(n_sims):

        macro_path = np.zeros((horizon, k_macro)) if macro_sims is None else macro_sims[i]

        if ms_tk is not None:

            S_tk = simulate_ms_states_conditional(
                steps = horizon, 
                P = ms_tk.P, 
                rng = rng, 
                p_init = p_last_tk
            )

            gpath_tk = ms_tk.g[S_tk].astype(float) 

        else:

            S_tk = np.zeros(horizon, dtype = int)

            gpath_tk = np.ones(horizon, float)

        J, jump_shocks = draw_state_conditional_jumps(
            states = S_tk,
            p_jump = p_jump_by_state,
            gpd_params = gpd_params_by_state,
            rng = rng
        )

        Xf = build_future_exog(
            hist_tail_vals = hist_tail_base,
            macro_path = macro_path,
            F_fourier = F_fourier,
            col_order = col_order,
            scaler = sc,
            lags = EXOG_LAGS,
            jump = J.astype(float),
        )

        mu_list = []
        
        var_list = []
        
        if use_stack:
           
            for od in stack_orders:
           
                if od not in fits_by_order:
           
                    continue
           
                res = fits_by_order[od]
           
                fc = res.get_forecast(steps = horizon, exog = Xf)  
           
                mu = fc.predicted_mean
           
                va = np.asarray(fc.var_pred_mean)

                if not (np.isfinite(mu).all() and np.isfinite(va).all()):
              
                    continue
               
                mu_list.append(mu)
               
                var_list.append(np.asarray(va))

        else:
        
            for res, w in zip(ens.fits, ens.weights):
        
                fc = res.get_forecast(steps = horizon, exog = Xf)  
        
                mu = fc.predicted_mean
        
                va = np.asarray(fc.var_pred_mean)

                if not (np.isfinite(mu).all() and np.isfinite(va).all()):
                
                    continue
              
                mu_list.append(mu)
              
                var_list.append(np.asarray(va))
      
        if not mu_list: 
      
            mu_mix = np.zeros(horizon)
      
            var_mix = np.full(horizon, np.var(resid) if resid.size else 1e-6)
      
        else:
      
            mu_arr = np.asarray(mu_list)               
      
            var_arr = np.asarray(var_list)                    
      
            mu_mix = (w_vec * mu_arr).sum(axis = 0)           
      
            var_mix = (w_vec * (var_arr + (mu_arr - mu_mix)**2)).sum(axis = 0)  

        boot = stationary_bootstrap(
            resid = resid_c, 
            length = horizon,
            p = (1.0 / max(1, RESID_BLOCK)), 
            rng = rng
        )
       
        if df_local and df_local > 2:
       
            boot *= get_t_scales(
                H = horizon, 
                df = df_local, 
                rng = rng
            )

        noise_model = alpha * rng.standard_normal(horizon) * np.sqrt(np.maximum(var_mix, 1e-12))
        
        noise_boot = (1.0 - alpha) * eps_scale * boot

        r_path = mu_mix + (noise_model + noise_boot) * gpath_tk

        r_path = r_path + jump_shocks


        path = cp * np.exp(np.cumsum(r_path))
       
        final_prices[i] = float(np.clip(path[-1], lb, ub))

    q05, q50, q95 = np.quantile(final_prices, [0.05, 0.50, 0.95])

    rets = final_prices / cp - 1.0
   
    ret_mean = float(np.mean(rets))
   
    ret_std = float(np.std(rets, ddof=1))

    return {
        "low": float(q05),
        "avg": float(q50),
        "high": float(q95),
        "returns": ret_mean,
        "se": ret_std,
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

    out = []
    
    n = resid.shape[0]

    i = rng.integers(0, n)
    
    while len(out) < length:
    
        L = 1 + rng.geometric(p)
    
        seg = resid[i: i+L]
    
        if len(seg) < L: seg = np.r_[seg, resid[:L-len(seg)]]
    
        out.extend(seg.tolist())
    
        if rng.random() < p:
    
            i = rng.integers(0, n)
    
        else:
    
            i = (i + L) % n
    
    return np.array(out[:length])


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
    
    macro = MacroData()
    
    r = macro.r

    tickers: List[str] =  list(config.tickers)
    
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

    country_map = {t: str(c) for t, c in zip(analyst.index, analyst["country"])}
    
    raw_macro["country"] = raw_macro["ticker"].map(country_map)

    macro_clean = raw_macro[["ds", "country"] + BASE_REGRESSORS].dropna()

    logger.info("Simulating macro scenarios (ECVAR/BVAR/VAR) …")
    
    country_paths: Dict[str, Optional[np.ndarray]] = {}

    for ctry, dfc in macro_clean.groupby("country"):
    
        dfm_raw = (
            dfc.set_index("ds")[BASE_REGRESSORS]
               .sort_index()
               .resample("W")  
               .mean()
               .ffill()
               .dropna()
        )
     
        try:
     
            sims = simulate_macro_paths_for_country(
                df_levels_weekly = dfm_raw, 
                steps = forecast_period,
                n_sims = N_SIMS, 
                seed = rng_global.integers(1_000_000_000)
            )
           
            country_paths[ctry] = sims
       
        except Exception as e:
       
            logger.warning("Macro simulation failed for %s (%s). Using flat deltas.", ctry, e)
       
            country_paths[ctry] = np.zeros((N_SIMS, forecast_period, lenBR))

    macro_deltas_by_ticker: Dict[str, pd.DataFrame] = {}
   
    for tk, g in raw_macro.groupby("ticker"):
   
        lvl = (g.set_index("ds")[BASE_REGRESSORS]
                .sort_index()
                .resample("W-SUN").mean().ffill())
   
        d = _macro_stationary_deltas(
            df_levels = lvl
        ).reindex(close.index).ffill()  
   
        macro_deltas_by_ticker[tk] = d
   
    logger.info("Fitting SARIMAX ensemble and running Monte Carlo …")

   
    def _process_ticker(
        tk: str
    ) -> Tuple[str, Optional[Dict[str, float]]]:
    
        cp = latest_prices.get(tk, np.nan)
    
        if not np.isfinite(cp):
    
            logger.warning("No price for %s, skipping", tk)
    
            return tk, None

        dfp = pd.DataFrame({"ds": close.index, "price": close[tk].values})
    
        y = np.diff(np.log(dfp["price"].values))
    
        ticker_idx = dfp["ds"]
    
        dfm_deltas = macro_deltas_by_ticker[tk].reindex(ticker_idx).values

        dfr = pd.DataFrame(
            np.column_stack([dfp["price"].values[1:], y, dfm_deltas[1:]]),
            index = pd.DatetimeIndex(ticker_idx[1:]),
            columns = ["price", "y"] + BASE_REGRESSORS
        )
        
        try:
     
            dfr.index.freq = to_offset("W-SUN")
     
        except Exception:
     
            pass
     
        jump_ind, p_jump = make_jump_indicator_from_returns(
            y = dfr["y"], 
            q = JUMP_Q
        )
        
        dfr["jump_ind"] = jump_ind.astype(float)
        
        dfr = dfr.replace([np.inf, -np.inf], np.nan)
        
        first_valid = dfr["price"].first_valid_index()
        
        if first_valid is not None:
        
            dfr = dfr.loc[first_valid:]
            
        dfr = dfr.dropna(subset = ["y"] + BASE_REGRESSORS)

        if len(dfr) < forecast_period + 32:
     
            logger.warning("Insufficient history for %s, skipping", tk)
     
            return tk, None
        
        ctry = country_map.get(tk)
     
        macro_sims = country_paths.get(ctry)
     
        if macro_sims is None:
     
            macro_sims = np.zeros((N_SIMS, forecast_period, lenBR))

        lb_tk = float(lb[tk]) if np.isfinite(lb.get(tk, np.nan)) else -np.inf
     
        ub_tk = float(ub[tk]) if np.isfinite(ub.get(tk, np.nan)) else np.inf

        out = simulate_price_paths_for_ticker(
            tk = tk,
            df_tk = dfr[["price", "y"] + BASE_REGRESSORS + ["jump_ind"]],
            cp = float(cp),
            lb = lb_tk,
            ub = ub_tk,
            macro_sims = macro_sims,
            horizon = forecast_period,
            rng_seed = int(rng_global.integers(1_000_000_000)),
        )

        print(f"{tk}: Avg: {out['avg']:.2f}, Low: {out['low']:.2f}, High: {out['high']:.2f}, "
              f"Return: {out['returns']:.4f}, MC Std: {out['se']:.4f}")
      
        return tk, out

    results: Dict[str, Dict[str, float]] = {}
   
    for tk, res in Parallel(n_jobs=N_JOBS, prefer="processes")(delayed(_process_ticker)(tk) for tk in tickers):
   
        if res is not None:
   
            results[tk] = res

    rows = []
   
    for tk in tickers:
   
        if tk not in results:
   
            continue
   
        cp = latest_prices.get(tk)
   
        r = results[tk]
   
        rows.append(
            {
                "Ticker": tk,
                "Current Price": cp,
                "Avg Price (p50)": r["avg"],
                "Low Price (p5)": r["low"],
                "High Price (p95)": r["high"],
                "Returns": (r["avg"] / cp - 1.0) if cp else np.nan,
                "SE": r["se"],
            }
        )
   
    if not rows:
   
        logger.warning("No results produced.")
   
        return

    df_out = pd.DataFrame(rows).set_index("Ticker")

    export_results(
        sheets = {"SARIMAX Monte Carlo": df_out},
        output_excel_file = getattr(config, "MODEL_FILE", "model_output.xlsx"),
    )
    
    logger.info("Run completed.")


if __name__ == "__main__":
    
    main()
