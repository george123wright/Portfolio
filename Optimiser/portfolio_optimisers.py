from __future__ import annotations

"""
Portfolio optimisation suite: ratio maximisation, CCP composites, BL integration, and
risk-aware constraints.

This module implements several optimiser families for long-only portfolios, built
around convex subproblems and first-order methods:

1) Fractional “ratio” objectives solved via Dinkelbach transforms:
  
   - Maximum Sharpe ratio (MSR):    
   
        maximise S(w) = (μᵀw − r_f) / sqrt(wᵀΣw).
   
   - Maximum Sortino ratio:        
   
        maximise (E[r_p] − τ) / D(w) with downside D(w).
   
   - Maximum Information Ratio:     
   
        maximise mean active / tracking-error RMS.
   
   - Score-over-CVaR (Rockafellar–Uryasev): 
   
        maximise sᵀw / CVaR_α(r_p).
   
    Each ratio becomes a sequence of convex problems of the form
   maximise  numerator(w) − λ · denominator(w), with the scalar λ updated as
   λ ← numerator(w*) / denominator(w*) until |numerator − λ · denominator| is small.

2) CCP (convex–concave procedure) combinations of smooth risk–reward terms with optional
   penalties and linearised non-convex constraints (e.g., sector risk caps). Each CCP step
   solves a convex subproblem obtained by linearising the concave part around the current
   iterate and, when enabled, linearising quadratic caps about the current point.

3) Black–Litterman (BL) integration to derive posterior (μ_bl, Σ_bl) from a prior and
   views, then solve MSR under the BL posterior.

4) Deflated Sharpe ratio (DSR) and Adjusted Sharpe ratio (ASR) variants:
   
   - DSR(w) = SR_ann(w) / SE{SR_week(w)} − E[max_i Z_i] where Z_i ~ N(0,1).
   
   - ASR(w) ≈ SR(w) · [1 + (γ₃ SR)/6 + ((γ₄ − 3) SR²)/24], with γ₃, γ₄ sample
     skewness and kurtosis from recent weekly returns.

CCP composite portfolios (overview)
-----------------------------------
All `comb_port*` optimisers maximise a calibrated linear combination of components
(e.g., Sharpe, Sortino, IR, Score/CVaR, ASR), using CCP to linearise the non-convex
parts and (optionally) sector risk-contribution caps. Modelling distinctions:

- `comb_port`   : Sharpe + Sortino + BL-Sharpe − quadratic proximity to MIR/MSP anchors.
                  Emphasises stability by keeping close to two practical anchor portfolios.

- `comb_port1`  : Sharpe + Sortino + BL-Sharpe + IR(1y) + Score/CVaR(1y).
                  Score/CVaR uses a smoothed RU CVaR for gradients and exact RU LP in final scoring.

- `comb_port2`  : Sharpe + Sortino + BL-Sharpe − L1 proximity to MIR/MSP (Huber-smoothed).
                  Promotes sparse tilts relative to anchors; turnover-friendly.

- `comb_port3`  : Sharpe + Sortino + IR(5y) + Score/CVaR(1y) − L2 proximity to BL.
                  Includes Armijo backtracking on the true smoothed composite; long-horizon discipline.

- `comb_port4`  : As in comb_port3 but with L1 proximity to BL and **sector risk caps** (linearised).
                  Controls sector-level risk concentration while encouraging sparse BL tilts.

- `comb_port5`  : Sharpe + Sortino + BL-Sharpe + IR(1y) + Score/CVaR(1y) + **ASR**.
                  Adds higher-moment quality via adjusted Sharpe (skew/kurtosis aware).

- `comb_port6`  : Sharpe + Sortino + IR(5y) + Score/CVaR(1y) + **ASR** − L2 proximity to BL.
                  Backtracking on smoothed composite; regularised towards BL equilibrium.

- `comb_port7`  : As in comb_port6 but with L1 proximity to BL and **sector risk caps** (linearised).
                  Enforces sector budgets, sparse BL tilts, long-horizon and tail-risk features.

Constraints and modelling
-------------------------
All optimisers use a common feasible set:

- Budget: sum_i w_i = 1.  Long-only: w_i ≥ 0.

- Asset-level gating/bounds: lb_i ≤ w_i ≤ ub_i from screening.

- Industry/sector caps: A_caps w ≤ caps (row sums of industry/sector masks).

- Optional single-portfolio envelope: lo_i ≤ w_i ≤ hi_i derived from base solutions.

- Optional sector risk-contribution caps: wᵀ M_s w ≤ α_s · φ(w; w_t) for each sector s,
  where M_s = D_s Σ D_s (D_s diagonal mask), and φ is an affine majoriser at w_t (CCP).

Notation
--------
w ∈ ℝ^N  : weights; μ ∈ ℝ^N : expected returns; Σ ∈ ℝ^{N×N} : covariance (PSD).
r_f, τ   : annual risk-free and weekly target; R ∈ ℝ^{T×N} : weekly returns.
b ∈ ℝ^T : benchmark weekly returns; s ∈ ℝ^N : score vector.
All gradients are with respect to w (column interpretation); ℓ₂ norms unless stated.

Numerical safeguards
--------------------
Small ε > 0 floors denominators and square-roots; Cholesky factors are built from
nearest-PSD projections if required.

External references
-------------------
- Dinkelbach (1967): Nonlinear fractional programming.
- Rockafellar & Uryasev (2000): Optimisation of Conditional Value-at-Risk.
- Black & Litterman (1992): Global portfolio optimisation.
"""


import numpy as np
import pandas as pd
import cvxpy as cp
import functools
from typing import Tuple, Dict, List, Optional
from scipy.stats import norm
from scipy.integrate import quad

import portfolio_grad as g
from functions.black_litterman_model import black_litterman
from functions.cov_functions import _nearest_psd_preserve_diag
import config


def generate_bounds_for_asset(
    bnd_h: float, 
    bnd_l: float,
    er: float, 
    score: float
) -> Tuple[float, float]:
    """
    Screen-driven asset bounds for long-only portfolios.

    Logic
    -----
    Given upper and lower per-asset caps (bnd_h, bnd_l) along with an asset’s
    expected return er and cross-sectional score, this function *gates out* assets
    failing either signal: if score ≤ 0 or er ≤ 0 (or either is non-finite), it
    returns (0, 0). Otherwise, it returns (bnd_l, bnd_h).

    Use in optimisation
    -------------------
    The pair (lb_i, ub_i) is used in the box constraints:
       
        lb_i ≤ w_i ≤ ub_i,  with w_i ≥ 0.

    Parameters
    ----------
    bnd_h, bnd_l : float
        Proposed high/low caps for the asset weight.
    er : float
        Expected return signal for screening.
    score : float
        Cross-sectional score for screening.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Element-wise bounds to be enforced in the feasible set.
    """

    if not np.isfinite(score) or not np.isfinite(er) or score <= 0 or er <= 0:
    
        return (0.0, 0.0)
    
    return (float(bnd_l), float(bnd_h))


def expected_max_std_normal(
    N: int
) -> float:
    """
    Expectation of the maximum of N iid standard normal variables.

    Definition
    ----------
    Let Z_1, …, Z_N ~ iid N(0, 1). This returns E[max_i Z_i]. The value is computed
    exactly by one-dimensional quadrature using the identity
    
        E[max_i Z_i] = ∫ x * N * φ(x) * Φ(x)^{N-1} dx  over x ∈ (−∞, ∞),
    
    where φ and Φ are the standard normal density and CDF.

    Use
    ---
    In deflated Sharpe computations, the term E[max_i Z_i] estimates the bias from
    multiple testing over N strategies/starts.

    Parameters
    ----------
    N : int
        Number of iid standard normals in the maximum.

    Returns
    -------
    float
        E[max_{i≤N} Z_i], or 0 if N ≤ 0.
    """

    if N <= 0:

        return 0.0

    phi = norm.pdf

    Phi = norm.cdf

    integrand = lambda x: x * N * phi(x) * (Phi(x) ** (N - 1))

    val, _ = quad(integrand, -np.inf, np.inf, limit = 200)

    return float(val)


class PortfolioOptimiser:
    """
    Stateful optimiser holding universe-specific data, caches, and reusable
    constraints.

    Design
    ------
    Create a single instance per universe; expensive objects (Cholesky factors,
    nearest-PSD projections, Black–Litterman posteriors, masks) are computed once
    and cached via `cached_property`. Methods share a common constraint builder to
    ensure identical feasible regions across optimisers.
    """

    def __init__(
        self,
        
        # core
        er: pd.Series,              
        cov: pd.DataFrame | np.ndarray,
        scores: pd.Series,
        
        # weekly data
        weekly_ret_1y: pd.DataFrame,
        last_5_year_weekly_rets: pd.DataFrame,
        benchmark_weekly_ret: Optional[pd.Series] = None,
        
        # classifications + box bounds
        ticker_ind: pd.Series = None,
        ticker_sec: pd.Series = None,
        bnd_h: pd.Series = None,
        bnd_l: pd.Series = None,
        
        # BL inputs (prior & views)
        comb_std: Optional[pd.Series] = None,
        sigma_prior: Optional[pd.DataFrame] = None,
        mcap: Optional[pd.Series] = None,
        
        # caps & limits
        sector_limits: Optional[Dict[str, float]] = None,
        default_max_industry_pct: float = config.IND_MAX_WEIGHT,
        default_max_sector_pct: float = config.SECTOR_MAX_WEIGHT,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0), 
        
        # rates
        rf_annual: Optional[float] = None,         
        rf_week: Optional[float] = None,
        cvar_level_pct = 5.0,
        random_state: int = 42,
        sample_size: int = 5000,        
    ):
        """
        Construct an optimiser over a given universe.

        Inputs
        ------
        er : pd.Series
            Forecast annualised expected returns μ indexed by tickers.
        cov : pd.DataFrame | np.ndarray
            Forecast annualised covariance Σ (PSD is enforced by a nearest-PSD projection).
        scores : pd.Series
            Cross-sectional score vector s used in Score/CVaR and seeding.
        weekly_ret_1y : pd.DataFrame
            1-year history of weekly returns R1 aligned to the universe.
        last_5_year_weekly_rets : pd.DataFrame
            Up to 5-year history of weekly returns R5 for IR(5y).
        benchmark_weekly_ret : pd.Series, optional
            Weekly benchmark series b for active/TE computations; falls back to rf_week.
        ticker_ind, ticker_sec : pd.Series, optional
            Industry and sector labels per ticker for group caps and risk caps.
        bnd_h, bnd_l : pd.Series, optional
            Per-asset high/low caps used after gating.
        comb_std, sigma_prior, mcap : optional
            Inputs for Black–Litterman posterior construction.
        sector_limits : dict[str, float], optional
            Per-sector maximum weight caps; otherwise defaults from `config`.
        default_max_industry_pct, default_max_sector_pct : float
            Global caps used if a group is missing from `sector_limits`.
        gamma : tuple
            Default weights for composite objectives; used where indicated.
        rf_annual, rf_week : float, optional
            Annual and weekly risk-free rates; default to `config` if omitted.
        cvar_level_pct : float
            CVaR tail probability percentage (e.g., 5.0 for α = 0.05).
        random_state : int, sample_size : int
            Seeding and sample size for calibration/sketching routines.

        Behaviour
        ---------
        - Ticker normalisation (optional) and strict realignment of all inputs to the
        universe index.
        - Gating and feasibility checks: verifies that the induced box constraints and
        group caps admit a feasible long-only solution (via a small LP).
        - Caching of key matrices (Σ, Σ_bl), Cholesky factors (A, Ab), returns (R1, R5),
        and helper structures (group masks, rows for cap matrices).
        """

        if not isinstance(er, pd.Series) and not isinstance(scores, pd.Series):

            raise ValueError("Provide at least one of er or scores as pd.Series to infer the universe.")

        if isinstance(er, pd.Series):
            
            base = er 
        
        else:
            
            base = scores
            
        self._universe: List[str] = list(base.index)
        
        idx = pd.Index(self._universe)
        
        self.random_state = int(random_state)
        
        self.sample_size = sample_size
        
        self.cvar_level_pct = float(cvar_level_pct)
        
        if idx.has_duplicates:
        
            dupes = idx[idx.duplicated()].unique().tolist()
        
            raise ValueError(f"Duplicate tickers in universe: {dupes}")
        
        self._universe = list(idx)
        
        NORMALIZE_TICKER_CASE = getattr(config, "NORMALIZE_TICKER_CASE", False)
        
        if NORMALIZE_TICKER_CASE:
            
            
            def _canon_tkr(
                x: str
            ) -> str:
                
                return x.upper().strip()


            canon = pd.Index(map(_canon_tkr, self._universe))
           
            if canon.has_duplicates:

                collided = canon[canon.duplicated()].unique().tolist()

                raise ValueError(f"Ticker collisions after case normalization: {collided}")

            mapping = dict(zip(self._universe, canon))

            self._universe = list(canon)


            def _canon_index(
                obj
            ):
                
                if obj is None: 
                
                    return None
                
                if isinstance(obj, pd.Series):
                
                    s = obj.copy()
                
                    s.index = s.index.map(lambda t: mapping.get(t, _canon_tkr(t)))
                
                    return s
                
                if isinstance(obj, pd.DataFrame):
                
                    df = obj.copy()
                
                    df.columns = df.columns.map(lambda t: mapping.get(t, _canon_tkr(t)))
                
                    return df
                
                return obj
            

            er = _canon_index(
                obj = er
            )
            
            scores = _canon_index(
                obj = scores
            )
            
            weekly_ret_1y = _canon_index(
                obj = weekly_ret_1y
            )
            
            last_5_year_weekly_rets = _canon_index(
                obj = last_5_year_weekly_rets
            )
            
            benchmark_weekly_ret = _canon_index(
                obj = benchmark_weekly_ret
            )
            
            ticker_ind = _canon_index(
                obj = ticker_ind
            )
            
            ticker_sec = _canon_index(
                obj = ticker_sec
            )
            
            bnd_h = _canon_index(
                obj= bnd_h
            )
            
            bnd_l = _canon_index(
                obj = bnd_l
            )
            
            comb_std = _canon_index(
                obj = comb_std
            )
            
            sigma_prior = _canon_index(
                obj = sigma_prior
            )
            
            mcap = _canon_index(
                obj = mcap
            )
        
        self._er = self._reindex_series(
            s = er
        ).astype(float).fillna(0.0)
        
        self._scores = self._reindex_series(
            s = scores
        ).astype(float).fillna(0.0)
        
        self.gamma = gamma

        if isinstance(cov, pd.DataFrame):
            
            cov = cov.reindex(index = self._universe, columns = self._universe).values
        
        self._cov = np.asarray(cov, float)

        self._weekly_ret_1y = self._reindex_df_cols(
            df = weekly_ret_1y
        ).dropna(how = "any")
        
        self._last_5y = self._reindex_df_cols(
            df = last_5_year_weekly_rets
        ).dropna(how = "any")
        
        self._benchmark_weekly = self._align_bench(
            bench = benchmark_weekly_ret
        )

        if ticker_ind is not None:
             
            self._ticker_ind = self._reindex_series(
                s = ticker_ind
            ) 
        
        else:
            
            self._ticker_ind = pd.Series(index = self._universe, dtype = object)
        
        if ticker_sec is not None:
            
            self._ticker_sec = self._reindex_series(
                s = ticker_sec
            )  
        
        else:
            self._ticker_sec = pd.Series(index = self._universe, dtype = object)

        if bnd_h is not None:
            
            self._bnd_h = (
                self._reindex_series(
                    s = bnd_h
                ).fillna(0.0).astype(float)
            )  
        
        else:
            
            self._bnd_h = pd.Series(1.0, index = self._universe)
        
        if bnd_l is not None:
            
            self._bnd_l = (
                self._reindex_series(
                    s = bnd_l
                ).fillna(0.0).astype(float)
            )  
        
        else: 
            
            self._bnd_l = pd.Series(0.0, index = self._universe)

        if comb_std is not None:
            
            self._comb_std = self._reindex_series(
                s = comb_std
            )  
        
        else:
            
            self._comb_std = None
        
        if isinstance(sigma_prior, pd.DataFrame):
            
            self._sigma_prior = sigma_prior.reindex(index = self._universe, columns = self._universe).values
        
        else:
            
            self._sigma_prior = None
        
        if mcap is not None:
            
            self._mcap = self._reindex_series(
                s = mcap
            )  
        
        else:
            
            self._mcap = None

        self.sector_limits = sector_limits or config.sector_limits
        
        self.default_max_industry_pct = float(default_max_industry_pct)
       
        self.default_max_sector_pct = float(default_max_sector_pct)

        if rf_annual is not None:
            
            self._rf_annual = float(rf_annual)  
        
        else:
            
            self._rf_annual = None
        
        if rf_week is not None:
       
            self._rf_week = float(rf_week)  
        
        else:
            self._rf_week = None

        self._mu_bl: Optional[pd.Series] = None
        
        self._sigma_bl: Optional[pd.DataFrame] = None

        self._last_diag: Optional[Dict[str, float]] = None
        
        self._solver_hint = None

        self._A_ind_rows, self._A_sec_rows, self._ind_order, self._sec_order = self._build_caps_rows()
        
        self.sector_masks = self._sector_masks_from_series(
            sector_series = self._ticker_sec, 
            universe = self.universe
        )
        
        self.alpha_dict = {
            "Technology": 0.4,
            "Communication Services": 0.3,
        }
        
        self._check_box_caps_feasible()
        

    def _reindex_series(
        self, 
        s: Optional[pd.Series]
    ) -> Optional[pd.Series]:
        """
        Reindex a Series to the current universe order, preserving dtype and allowing missing inputs.
        Returns None if the input is None.
        """

        if s is None: 
            
            return None
        
        return s.reindex(self._universe)


    def _reindex_df_cols(
        self, 
        df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Reindex a DataFrame’s columns to the current universe order, preserving rows; returns None if input is None.
        """

        if df is None: 
            
            return None
        
        return df.reindex(columns=self._universe)


    def _align_bench(
        self, 
        bench: Optional[pd.Series]
    ) -> Optional[pd.Series]:
        """
        Align and clean the benchmark weekly series: reindex to available history elsewhere and drop NaNs.
        """

        if bench is None: 
            
            return None
       
        return bench.dropna()


    @functools.cached_property
    def universe(
        self
    ) -> List[str]:
        """
        List[str]: canonical ticker order used throughout optimisation and reporting.
        """

        return self._universe


    @functools.cached_property
    def n(
        self
    ) -> int:
        """
        int: number of assets in the current universe.
        """

        return len(self._universe)


    @functools.cached_property
    def er_arr(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): annual expected returns vector μ aligned to the universe (float).
        """

        return self._er.to_numpy(dtype = float)


    @functools.cached_property
    def scores_arr(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): asset score vector s aligned to the universe (float).
        """

        return self._scores.to_numpy(dtype = float)


    @functools.cached_property
    def Σ(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,n): nearest-PSD version of the input covariance with diagonal preserved; small ridge added if needed.
        """

        return _nearest_psd_preserve_diag(
            C = self._cov,
            eps = 1e-10
        )


    @functools.cached_property
    def A(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,n): upper-triangular factor A with AᵀA ≈ Σ (Cholesky of a PSD-repaired Σ).
        """

        try:
        
            Lc = np.linalg.cholesky(self.Σ)
        
        except np.linalg.LinAlgError:
        
            w, V = np.linalg.eigh(0.5 * (self.Σ + self.Σ.T))
        
            w = np.maximum(w, 1e-12)
        
            Lc = np.linalg.cholesky((V * w) @ V.T)
        
        return Lc.T
    
    
    @functools.cached_property
    def Σb(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,n): Black–Litterman posterior covariance Σ_bl made PSD (diag preserved) with a small diagonal ridge.
        """

        if self._sigma_bl is None:       
            
            _ = self._bl    
        
        X = np.asarray(self._sigma_bl.to_numpy(dtype = float))
        
        X = _nearest_psd_preserve_diag(
            X, eps = 1e-10
        )
        
        return X + 1e-10 * np.eye(X.shape[0])

    
    @functools.cached_property
    def Ab(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,n): upper-triangular factor Ab such that Abᵀ Ab ≈ Σb (Cholesky on PSD-repaired BL covariance).
        """

        try:
    
            Lb = np.linalg.cholesky(self.Σb)
    
        except np.linalg.LinAlgError:
    
            w, V = np.linalg.eigh(0.5*(self.Σb + self.Σb.T))
    
            w = np.maximum(w, 1e-12)
    
            Lb = np.linalg.cholesky((V * w) @ V.T)
        
        return Lb.T


    @functools.cached_property
    def R1(
        self
    ) -> np.ndarray:
        """
        Weekly return matrix for the last year, aligned to the universe.

        Returns
        -------
        np.ndarray, shape (T1, n)
            Rows are weekly observations; columns are assets (universe order).
        Notes
        -----
        History is pruned to rows with at least `min_cov` coverage and NaNs filled to 0.
        """
        
        R1_df_raw = self._weekly_ret_1y.reindex(columns = self.universe)
        
        R1_df = self._prune_history(
            df = R1_df_raw, 
            min_cov = 0.8
        )

        if R1_df is None or R1_df.empty:

            raise RuntimeError("comb_port3: no overlapping 1y weekly history")

        R1_local = R1_df.to_numpy()
        
        return R1_local
    
    
    @functools.cached_property
    def R1T(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,T₁): transpose of the 1-year weekly return matrix (for gradient reuse).
        """

        return self.R1.T


    @functools.cached_property
    def R5(
        self
    ) -> np.ndarray:
        """
        Weekly return matrix for up to five years, aligned to the universe.

        Returns
        -------
        np.ndarray, shape (T5, n)
            Used for IR(5y) calculations (possibly against a weekly benchmark).
        """

        return self._R5_b_te[0]


    @functools.cached_property
    def T1(
        self
    ) -> int:      
        """
        int: number of weekly observations in the 1-year history.
        """

        return self.R1.shape[0]
    
    
    @functools.cached_property
    def sqrtT1(
        self
    ) -> float:
        """
        float: √T₁, used to RMS-normalise weekly sums and standard errors.
        """

        return float(np.sqrt(self.T1))
    

    @functools.cached_property
    def sqrtT5(
        self
    ) -> float:
        """
        float: √T₅, used to RMS-normalise 5-year tracking error.
        """

        return float(np.sqrt(self.T5))    


    @functools.cached_property
    def T5(
        self
    ) -> int:
        """
        int: number of weekly observations in the 5-year history (after pruning/alignment).
        """

        return self._R5_b_te[0].shape[0]


    @functools.cached_property
    def rf_ann(
        self
    ) -> float:
        """
        Annual risk-free rate used in Sharpe-type numerators.
        """ 
        
        if self._rf_annual is not None: 
            
            return float(self._rf_annual)
        
        return float(getattr(config, "RF", 0.0))


    @functools.cached_property
    def rf_week(
        self
    ) -> float:
        """
        Weekly risk-free rate used in Sortino/IR numerators or benchmark fallback.
        """ 

        if self._rf_week is not None: 
            
            return float(self._rf_week)
        
        return float(getattr(config, "RF_PER_WEEK", 0.0))

    
    @functools.cached_property
    def sqrt52(
        self
    ) -> float:
        """
        Square-root of 52; used to annualise weekly standard deviations.
        """  

        return float(np.sqrt(52.0))
    
    
    @functools.cached_property
    def _R5_b_te(
        self
    ):
        """
        Tuple[np.ndarray, Optional[np.ndarray]]:
        Aligned 5-year weekly returns matrix (R5, shape T₅×n) and matching benchmark series b⁵ (length T₅) if provided.
        Rows with insufficient coverage are dropped before conversion to NumPy.
        """

        R5_df_raw = self._last_5y.reindex(columns = self.universe)
        
        R5_df = self._prune_history(
            df = R5_df_raw,
            min_cov = 0.8
        )
       
        if R5_df is None or R5_df.empty:
       
            raise RuntimeError("comb_port4: no overlapping 5y weekly history")

        b5_vec = None
       
        if self._benchmark_weekly is not None:
       
            b5_al = self._benchmark_weekly.reindex(R5_df.index)
       
            ok5 = b5_al.notna()
       
            if ok5.any():
       
                R5_local = R5_df.loc[ok5].to_numpy()
       
                b5_vec = b5_al.loc[ok5].to_numpy()
       
            else:
              
                R5_local = R5_df.to_numpy()
              
                b5_vec = None
                
        else:
           
            R5_local = R5_df.to_numpy()
           
            b5_vec = None
        
        return R5_local, b5_vec
    
    
    @functools.cached_property
    def R5T(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,T₅): transpose of the 5-year weekly returns matrix.
        """

        return self._R5_b_te[0].T
    
    
    @functools.cached_property
    def _ones5(
        self
    ) -> np.ndarray:
        """
        np.ndarray (T₅,): vector of ones used for simple means over the 5-year window.
        """

        return np.ones(self.T5)

    
    @functools.cached_property
    def bound_arr(
        self
    ) -> np.ndarray:
        """
        Element-wise lower/upper bounds (lb, ub) after gating.

        Returns
        -------
        (lb, ub) : tuple[np.ndarray, np.ndarray], each shape (n,)
            Vectors of per-asset bounds used in the box constraints.
        """

        out_li = np.zeros(self.n)
        
        out_ui = np.zeros(self.n)
        
        for i, t in enumerate(self.universe):
        
            li, ui = generate_bounds_for_asset(
                bnd_h = float(self._bnd_h.loc[t]), 
                bnd_l = float(self._bnd_l.loc[t]),
                er = float(self._er.loc[t]), 
                score = float(self._scores.loc[t])
            )
            
            out_li[i] = li
            
            out_ui[i] = ui
        
        return out_li, out_ui


    @functools.cached_property
    def ind_idxs(
        self
    ) -> Dict[object, np.ndarray]:
        """
        Dict[industry, np.ndarray]: map from industry label to index array of assets in that industry (for caps).
        """

        return {ind: np.where(self._ticker_ind.values == ind)[0] for ind in self._ticker_ind.dropna().unique()}


    @functools.cached_property
    def sec_idxs(
        self
    ) -> Dict[object, np.ndarray]:
        """
        Dict[sector, np.ndarray]: map from sector label to index array of assets in that sector (for caps).
        """

        return {sec: np.where(self._ticker_sec.values == sec)[0] for sec in self._ticker_sec.dropna().unique()}
    

    @functools.cached_property
    def _cvar_beta_calib(
        self
    ) -> float:
        """
        float: softplus/sigmoid temperature β used to smooth RU-CVaR **for gradients** during calibration/CCP.
        """

        return 30.0
    
    
    @functools.cached_property
    def _cvar_beta_solve(
        self
    ) -> float:
        """
        float: higher softplus/sigmoid temperature β used when solving/evaluating smoothed RU-CVaR (sharper tail).
        """

        return 50.0
    
    
    @functools.cached_property
    def uniform_weights(
        self
    ) -> np.ndarray:
        """
        A feasible uniform portfolio on the active box.

        Definition
        ----------
        Start from equal weights over assets with ub_i > 0, clip to [lb_i, ub_i], renormalise
        to sum to one, and project into the feasible set (budget, boxes, caps).
        """

        self._assert_alignment()
        
        lb, ub = self.bound_arr
        
        target = np.clip(np.ones(self.n) / max((ub > 1e-12).sum(), 1), lb, ub)
        
        target /= target.sum()
        
        return self._project_feasible(
            target = target
        )  


    @functools.cached_property
    def _cvar_level(
        self
    ) -> float:
        
        return self.cvar_level_pct / 100.0
    
    
    @functools.cached_property
    def _msr(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): cached MSR solution re-coerced to the current universe and normalised.
        """

        w_msr = self.msr()
        
        w_msr_arr = self._coerce_weights(
            w = w_msr, 
            name = "w_msr"
        ).to_numpy()
        
        return w_msr_arr
    
    
    @functools.cached_property
    def _sortino(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): cached maximum Sortino solution (1-year downside), coerced and normalised.
        """
        
        w_so = self.sortino()
        
        w_so_arr = self._coerce_weights(
            w = w_so, 
            name = "w_so"
        ).to_numpy()
        
        return w_so_arr
    
    
    @functools.cached_property
    def _mir(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): cached 5-year Information Ratio solution, coerced and normalised.
        """

        w_mir = self.MIR()
        
        w_mir_arr = self._coerce_weights(
            w = w_mir, 
            name = "w_mir"
        ).to_numpy()
        
        return w_mir_arr
    
    
    @functools.cached_property
    def _bl(
        self
    ) -> np.ndarray:
        """
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        (w_BL as array, μ_bl as array, Σ_bl as DataFrame) from the Black–Litterman solver, all aligned to the universe.
        """
        
        w_bl, mu_bl, sigma_bl = self.black_litterman_weights()
        
        w_bl_arr = self._coerce_weights(
            w = w_bl, 
            name = "w_bl"
        ).to_numpy()
        
        return w_bl_arr, mu_bl.to_numpy(), sigma_bl
    
    
    @functools.cached_property
    def _msp(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): cached Score/CVaR portfolio (RU formulation), coerced and normalised.
        """

        w_msp = self.msp()
        
        w_msp_arr = self._coerce_weights(
            w = w_msp, 
            name = "w_msp"
        ).to_numpy()
        
        return w_msp_arr
    
    
    def _initials(
        self
    ) -> np.ndarray:
        """
        List[np.ndarray]: a small menu of diverse initial portfolios (MSR, Sortino, MIR, BL, MSP, a Dirichlet mix, and EW).
        """

        init_mix = self._initial_mixes_L2(
            w_msr = self._msr,
            w_sortino = self._sortino,
            w_mir = self._mir,
            w_bl = self._bl[0],
            w_msp = self._msp
        )
                
        return [self._msr, self._sortino, self._mir, self._bl[0], self._msp, init_mix, self.uniform_weights]

    
    @functools.cached_property
    def _initial_seeds(
        self
    ) -> List[np.ndarray]:
        """
        List[np.ndarray]: validated, finite initial weight vectors (shape (n,)), suitable for CCP/Dinkelbach starts.
        """

        seeds = []
    
        for w in self._initials():
    
            a = np.asarray(w, float)
    
            if a.ndim == 1 and a.size == self.n and np.all(np.isfinite(a)):
    
                seeds.append(a)
    
        return seeds
    
    
    @functools.cached_property
    def _grad_m5(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): ∂ mean(R5 w)/∂w = R5ᵀ 1 / T₅, used inside IR(5y) gradients.
        """

        return (self.R5T @ self._ones5) / self.T5
        

    @functools.cached_property
    def _grad_m1(
        self
    ) -> np.ndarray:
        """
        np.ndarray (n,): ∂ mean(R1 w)/∂w = R1ᵀ 1 / T₁, used inside IR(1y) gradients.
        """

        return (self.R1T @ np.ones(self.T1)) / self.T1
    

    def _assert_alignment(
        self
    ):
        """
        Internal consistency checks on dimensions, column order, and shared universes across all stored inputs.
        """

        n = self.n
       
        assert self._cov.shape == (n, n)
       
        assert self._weekly_ret_1y.shape[1] == n
       
        assert self._last_5y.shape[1] == n
       
        assert len(self._er) == n and len(self._scores) == n
       
        assert (self._weekly_ret_1y.columns == self._last_5y.columns).all()
       
        assert self._weekly_ret_1y.columns.equals(pd.Index(self.universe))
        
        assert self._last_5y.columns.equals(pd.Index(self.universe))


    def _quad_vol(
        self, 
        w: np.ndarray, 
        Σ: np.ndarray, 
        Σw: Optional[np.ndarray] = None, 
        eps: float = None
    ):
        """
        Compute portfolio volatility from quadratic form with optional cache.

        Returns
        -------
        (vol, Σw) : tuple[float, np.ndarray]
            vol = sqrt(max(wᵀΣw, eps)); Σw optionally returned to reuse upstream.
        """
        
        if Σw is None: 
            
            Σw = Σ @ w
        
        q = float(w @ Σw)
        
        if eps is None: 
            
            eps = self._denom_floor
        
        return np.sqrt(max(q, eps)), Σw


    def _build_caps_rows(
        self
    ):
        """
        Precompute binary rows encoding industry/sector memberships.

        Each row r selects the assets in a group; group caps enforce rᵀ w ≤ cap_group.
        """
      
        A_ind_rows = []
        
        ind_order = []
        
        for ind, idxs in self.ind_idxs.items():
         
            row = np.zeros(self.n, dtype = float)
         
            row[idxs] = 1.0
         
            A_ind_rows.append(row)
         
            ind_order.append(ind)

        A_sec_rows = []
        
        sec_order = []
        
        for sec, idxs in self.sec_idxs.items():
           
            row = np.zeros(self.n, dtype = float)
           
            row[idxs] = 1.0
           
            A_sec_rows.append(row)
           
            sec_order.append(sec)

        return A_ind_rows, A_sec_rows, ind_order, sec_order
    
    
    def _check_box_caps_feasible(
        self
    ) -> None:
        """
        Feasibility check for boxes + group caps.

        Solves: maximise 1ᵀ w  subject to 0 ≤ w ≤ ub, A_caps w ≤ caps.
        Raises if the maximum achievable mass is < 1, i.e., the caps are too tight.
        """

        lb, ub = self.bound_arr
    
        if lb.sum() > 1 + 1e-12:
         
            raise ValueError(f"Lower bounds sum to {lb.sum():.4f} > 1")

        A_caps, caps = self._caps_mats(
            max_industry_pct = self.default_max_industry_pct,
            max_sector_pct = self.default_max_sector_pct
        )
        
        if A_caps.size:

            w = cp.Variable(self.n, nonneg = True)
            
            cons = [
                w <= ub,
                A_caps @ w <= caps
            ]
           
            prob = cp.Problem(cp.Maximize(cp.sum(w)), cons)
          
            if not self._solve(prob) or prob.value is None or prob.value < 1 - 1e-6:
             
                raise ValueError("Caps/boxes infeasible after gating; loosen caps or gating.")


    def _caps_mats(
        self, 
        max_industry_pct: float, 
        max_sector_pct: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble (A_caps, caps_vec) for industry and sector caps.

        A_caps : stack of group membership rows; caps_vec : corresponding group caps.
        If no caps are active, both are zero-sized arrays.
        """
       
        mats = []
       
        caps = []

        if len(self._A_ind_rows) > 0:

            mats.append(np.vstack(self._A_ind_rows))

            caps.extend([max_industry_pct] * len(self._A_ind_rows))

        if len(self._A_sec_rows) > 0:

            A_sec = np.vstack(self._A_sec_rows)

            caps_sec = []

            for sec in self._sec_order:

                cap = self.sector_limits.get(sec, max_sector_pct)

                caps_sec.append(cap)

            mats.append(A_sec)

            caps.extend(caps_sec)

        if mats:

            A_caps = np.vstack(mats)

            caps_vec = np.asarray(caps, float)

        else:

            A_caps = np.zeros((0, self.n), float)

            caps_vec = np.zeros((0,), float)

        return A_caps, caps_vec


    def _solve(
        self, 
        prob: cp.Problem
    ) -> bool:
        """
        Robust CVXPY solve helper: tries multiple solvers with warm-starts and tolerances.
        """

        order = []
    
        if self._solver_hint is not None:
    
            order.append(self._solver_hint)

        order += [
            dict(solver=cp.CLARABEL, warm_start = True),
            dict(solver=cp.ECOS, feastol = 1e-7, reltol = 1e-7, abstol = 1e-7, max_iters = 3000, warm_start = True),
            dict(solver = cp.SCS, eps = 5e-4, max_iters = 20000, acceleration_lookback = 20, warm_start = True),
            dict(solver = cp.OSQP, warm_start = True),
        ]

        tried = set()
       
        for kwargs in order:
       
            key = kwargs.get("solver")
       
            if key in tried: 
                
                continue
           
            tried.add(key)
           
            try:
           
                prob.solve(**kwargs)
           
            except cp.error.SolverError:
           
                continue
           
            if prob.status in ("optimal", "optimal_inaccurate"):
           
                self._solver_hint = kwargs  
           
                return True
        
        return False
    

    def _prune_history(
        self, 
        df: pd.DataFrame, 
        min_cov: float = 0.8
    ) -> pd.DataFrame:
        """
        Row-prune a returns DataFrame to achieve at least `min_cov` non-missing coverage per row; fill remaining NaNs with 0.
        """
    
        need = int(np.ceil(min_cov * df.shape[1]))
    
        keep = df.notna().sum(axis = 1) >= need
    
        out = df.loc[keep].copy()
    
        return out.fillna(0.0)


    def _sector_masks_from_series(
        self, 
        sector_series: pd.Series, 
        universe: list[str]
    ) -> dict[str, np.ndarray]:
        """
        Build one-hot masks per sector to compute sector variances and risk caps.

        For sector s, the mask m_s has m_s[i] = 1 if asset i is in s, else 0.
        """

        sec = sector_series.reindex(universe)
    
        masks = {}
    
        for s in sec.dropna().unique().tolist():
    
            m = (sec.values == s).astype(float)
    
            masks[str(s)] = m
    
        return masks
    
    
    def _single_port_cap(
        self
    ):
        """
        Element-wise bounds from a set of precomputed single-portfolio solutions.
        lo[i] = min_k w_k[i],  hi[i] = max_k w_k[i]
        """
        
        W = np.vstack([self._msr, self._sortino, self._mir, self._bl[0], self._msp])
       
        lo = W.min(axis = 0)
       
        hi = W.max(axis = 0)

        eps = 1e-12
       
        lo = np.clip(lo - eps, 0.0, None)
       
        hi = np.clip(hi + eps, 0.0, 1.0)
       
        return lo, hi
    

    def build_constraints(
        self,
        w_var: "cp.Variable",               
        *,
        A_caps: np.ndarray | None = None,
        caps_vec: np.ndarray | None = None,
        max_industry_pct: float | None = None,
        max_sector_pct: float | None = None,
        add_box: bool = True,
        sector_risk_cap: bool = False,
        Sigma: np.ndarray | None = None,           
        sector_masks: dict[str, np.ndarray] | None = None,
        sector_alpha: dict[str, float] | None = None,
        sector_series: "pd.Series" | None = None,
        universe: list[str] | None = None,
        single_port_cap: bool = False,
    ):
        """
        Build the common feasible set and (optionally) linearised sector risk caps.

        Feasible set (always on)
        ------------------------
       
        1) Budget:                 sum_i w_i = 1.
       
        2) Long-only box:          if `add_box` is True, lb_i ≤ w_i ≤ ub_i, with w_i ≥ 0.
       
        3) Group caps:             A_caps w ≤ caps_vec (industry/sector aggregate caps).
       
        4) Single-portfolio envelope (optional):
       
        If `single_port_cap=True`, enforce lo_i ≤ w_i ≤ hi_i where
       
            lo_i = min_k w_k[i]
            
            hi_i = max_k w_k[i] 
            
        over a set of precomputed base solutions (MSR, Sortino, MIR, BL, MSP). This yields
        a conservative “tube”.

        Sector risk-contribution caps (optional, CCP)
        ---------------------------------------------
        If `sector_risk_cap=True`, define, for each sector s with mask m_s,
        
            D_s = diag(m_s),  
            
            M_s = D_s Σ D_s  (sector covariance block)
        
        and enforce a quadratic cap on the sector variance 
        
            wᵀ M_s w ≤ α_s V_sector_cap(w).

        Because wᵀ M_s w is convex but the *right-hand-side* must depend on w (to scale
        with total risk), CCP linear majoriser is used for the total variance at the
        current iterate w_t:
        
            total_var(w) = wᵀ Σ w
            
            ∇ total_var(w_t) = (Σ + Σᵀ) w_t
            
            total_var(w) ≤ off + gradᵀ w
            
        with 
        
            off = total_var(w_t) − gradᵀ w_t. 
        
        The constraint becomes 
        
            wᵀ M_s w ≤ α_s * ( off + gradᵀ w ).

        This function therefore returns:
        
        - a list of CVXPY constraints,
        
        - `update_ccp(w_t)` which, when called with the current iterate, updates the
        (off, grad) parameters used in the linearisation.

        Parameters
        ----------
        w_var : cp.Variable, shape (n,)
            Decision variable (weights).
        A_caps, caps_vec : np.ndarray, optional
            Precomputed group cap matrix/vector; if omitted, built from cached rows.
        max_industry_pct, max_sector_pct : float, optional
            Default caps for groups not explicitly listed in `sector_limits`.
        add_box : bool
            Whether to add lb/ub long-only box constraints.
        sector_risk_cap : bool
            Whether to add CCP linearised sector risk-contribution caps.
        Sigma : np.ndarray, sector_masks, sector_alpha : optional
            Inputs required if sector risk caps are on.

        Returns
        -------
        constraints : list
            CVXPY constraints implementing the feasible set.
        update_ccp : callable | None
            Function `update_ccp(w_t)` to refresh linearisation; None if not used.
        """

        n = self.n
        
        lb_arr, ub_arr = self.bound_arr

        cons: list = [cp.sum(w_var) == 1]
       
        if add_box:
           
            cons += [w_var >= lb_arr, w_var <= ub_arr]
            
        if max_industry_pct is None:
            
            max_industry_pct = self.default_max_industry_pct
        
        if max_sector_pct is None:
            
            max_sector_pct = self.default_max_sector_pct

        if A_caps is None or caps_vec is None:
            
            A_caps, caps_vec = self._caps_mats(
                max_industry_pct = max_industry_pct,
                max_sector_pct = max_sector_pct,
            )
      
        if A_caps is not None and A_caps.size:
      
            cons.append(A_caps @ w_var <= caps_vec)
            
        if single_port_cap:
            
            lo_cap, hi_cap = self._single_port_cap()
            
            cons += [w_var >= lo_cap, w_var <= hi_cap]

        if sector_risk_cap:
            
            if Sigma is None:
            
                Sigma = self.Σ
            
            if sector_masks is None:
               
                if sector_series is None or universe is None:
               
                    raise ValueError("Provide sector_masks or (sector_series and universe) when sector_risk_cap=True.")
               
                sector_masks = self._sector_masks_from_series(
                    sector_series = sector_series, 
                    universe = universe
                )

            if sector_alpha is None:

                sector_alpha = {s: 0.25 for s in sector_masks.keys()}

            M_s_dict: dict[str, np.ndarray] = {}
        
            for s, m in sector_masks.items():
        
                D = np.diag(np.asarray(m, float))
        
                M = D @ Sigma @ D
        
                M_s_dict[s] = 0.5 * (M + M.T)  

            grad_param = cp.Parameter(n, name = "grad_sig2")    
           
            off_param = cp.Parameter(name = "off_sig2")

            for s, M_s in M_s_dict.items():
           
                alpha_s = float(sector_alpha.get(s, 1.0))
           
                lhs = cp.quad_form(w_var, M_s)
           
                rhs = alpha_s * (off_param + grad_param @ w_var)
           
                cons.append(lhs <= rhs)


            def update_ccp(
                w_t: np.ndarray
            ) -> None:
                """
                Update linearisation at current iterate w_t.
                """
                
                w_t = np.asarray(w_t, float).reshape(-1)

                sig2_t = float(w_t @ Sigma @ w_t)
               
                grad = (Sigma + Sigma.T) @ w_t
               
                grad_param.value = grad
               
                off_param.value = sig2_t - float(grad @ w_t)  

            return cons, update_ccp


        return cons
    

    def _dinkelbach_sharpe_max(
        self,
        μ: np.ndarray,
        A: np.ndarray,
        rf: float,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> np.ndarray:
        """
        Maximum Sharpe via Dinkelbach’s transform over the long-only feasible set.

        Objective
        ---------
            
            Sharpe(w) = (μᵀ w − r_f) / ||A w||_2,  
        
        where A is a Cholesky factor with AᵀA ≈ Σ.

        At iteration k with scalar λ_k, solve the convex subproblem
            
            maximise   (μᵀ w − r_f) − λ_k ||A w||_2
        
        subject to (common constraints: budget, boxes, group caps, etc.).
        
        Update
            
            λ_{k+1} = (μᵀ w* − r_f) / max(||A w*||_2, denom_floor)
        
        and stop when 
        
            |(μᵀ w* − r_f) − λ_k ||A w*||_2| < tol.

        Parameters
        ----------
        μ : np.ndarray, shape (n,)
        A : np.ndarray, shape (n, n)
        rf : float
        tol : float
        max_iter : int

        Returns
        -------
        np.ndarray, shape (n,)
            Optimal weights for the MSR problem.
        """
       
        n = self.n
       
        lam = 0.0
       
        w0 = self.uniform_weights
        
        ret0 = self.ew_excess_er
        
        vol0 = max(float(np.linalg.norm(A @ w0)), self._denom_floor)
        
        lam = ret0 / vol0

        for _ in range(max_iter):

            w = cp.Variable(n, nonneg = True)

            ret = μ @ w - rf

            vol = cp.norm(A @ w, 2)
                      
            obj = cp.Maximize(ret - lam * vol)
          
            cons = self.build_constraints(
                w_var = w, 
            )
            
            prob = cp.Problem(obj, cons)
            
            if not self._solve(prob) or w.value is None:
            
                raise RuntimeError("Dinkelbach Sharpe: infeasible subproblem")
            
            w_val = w.value
            
            ret_v = float(μ @ w_val) - rf
            
            vol_v = max(float(np.linalg.norm(A @ w_val)), self._denom_floor)
            
            if abs(ret_v - lam * vol_v) < tol:
            
                return w_val
            
            lam = ret_v / vol_v
        
        raise RuntimeError("Dinkelbach Sharpe: no convergence")


    def msr(
        self,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> pd.Series:
        """
        Maximum Sharpe ratio portfolio.

        Mathematics
        -----------
            Sharpe(w) = (μᵀ w − r_f) / sqrt(wᵀ Σ w),
            
        solved via Dinkelbach as described in `_dinkelbach_sharpe_max`. 
        
        Feasible set is the common constraints from`build_constraints`.

        Returns
        -------
        pd.Series
            Long-only MSR weights indexed by tickers.
        """
        
        self._assert_alignment()
        
        μ = self.er_arr
        
        A = self.A
        
        rf = self.rf_ann
            
        w = self._dinkelbach_sharpe_max(
            μ = μ, 
            A = A, 
            rf = rf,
            tol = tol, 
            max_iter = max_iter,
        )
        
        return pd.Series(w, index = self.universe, name = "msr")


    def sortino(
        self,
        tol: float = 1e-10,
        max_iter: int = 100,
        riskfree_rate: Optional[float] = None,
    ) -> pd.Series:
        """
        Maximum Sortino ratio via Dinkelbach with epigraph for downside deviation.

        Objective
        ---------
        
            Sortino(w) = (μᵀ w − r_f^ann) / D_ann(w),  
            
        where D_ann(w) is the annualised downside deviation relative to weekly rf (or 
        a provided weekly target).
        
        Let R be weekly returns, 
        
            r_p = R w
            
            u_i = max(0, r_f^week − r_{p,i}).

        Downside deviation (weekly) is
            
            D_week(w) = sqrt( (1/T) ∑_i u_i^2 ),
        
        and 
        
            D_ann(w) = D_week(w) * sqrt(52).

        Dinkelbach subproblem at λ:
            
            maximise   (μᵀ w − r_f^ann) − λ * max(D_ann(w), denom_floor)
            
            subject to u ≥ r_f^week − R w,  u ≥ 0,
                    and the common constraints.

        The `max(·, denom_floor)` ensures numerical robustness.

        Returns
        -------
        pd.Series
            Long-only maximum Sortino weights.
        """

        self._assert_alignment()
       
        n = self.n
       
        μ = self.er_arr
        
        if riskfree_rate is None:
       
            rf_ann = float(self.rf_ann)  
        
        else:
            
            rf_ann = riskfree_rate
            
        rf_week = self.rf_week
       
        T = self.T1
       
        if T == 0:
       
            raise RuntimeError("sortino: no weekly data")
        
        sqrtT1 = self.sqrtT1
       
        R = self.R1
       
        sqrt52 = self.sqrt52
                    
        w0 = self.uniform_weights
      
        exc0 = self.ew_excess_er
      
        u0 = np.maximum(rf_week - (R @ w0), 0.0)
      
        dd0 = (np.linalg.norm(u0) / np.sqrt(T)) * sqrt52
      
        dd0 = max(dd0, self._denom_floor)
      
        lam = exc0 / dd0

        for _ in range(max_iter):
           
            w = cp.Variable(n, nonneg = True)
           
            u = cp.Variable(T, nonneg = True)

            port_week = R @ w
            
            dd_week = cp.norm(u, 2) / sqrtT1
            
            dd_ann = dd_week * sqrt52
            
            dd_ann = cp.maximum(dd_ann, self._denom_floor)

            ret_ann = μ @ w - rf_ann
            
            obj = cp.Maximize(ret_ann - lam * dd_ann)

            cons = self.build_constraints(
                w_var = w
            )
            
            cons += [
                u >= rf_week - port_week, 
            ]
            
            prob = cp.Problem(obj, cons)
            
            if not self._solve(prob) or w.value is None:
            
                raise RuntimeError("sortino@Dinkelbach: infeasible")

            w_opt = w.value

            exc = float(μ @ w_opt) - rf_ann
            
            u_val = np.maximum(rf_week - (R @ w_opt), 0.0)
            
            dd_val = (np.linalg.norm(u_val) / sqrtT1) * sqrt52

            dd_val = max(dd_val, self._denom_floor)

            if abs(exc - lam * dd_val) < tol:
            
                return pd.Series(w_opt, index = self.universe, name = "sortino")

            lam = exc / dd_val

        raise RuntimeError("sortino@Dinkelbach: no convergence")


    def MIR(
        self,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> pd.Series:
        """
        Maximum Information Ratio (1y or 5y version chosen elsewhere).

        Objective
        ---------
        Given weekly returns R and benchmark/weekly rf vector b, define active returns
        
            a = R w − b. 
        
        Mean active is 
        
            A = mean(a)
            
        Tracking-error RMS is
        
            TE(w) = ||a||_2 / sqrt(T). The Information Ratio is IR(w) = A / TE(w).

        Dinkelbach subproblem at λ:
            maximise   A(w) − λ * max(TE(w), denom_floor)
            subject to common constraints.

        Returns
        -------
        pd.Series
            Long-only MIR weights.
        """
    
        self._assert_alignment()
            
        R, b_vec = self._R5_b_te
        
        if b_vec is None:
            
            b_vec = np.full(R.shape[0], self.rf_week)

        n = self.n
        
        T = self.T5
        
        sqrtT = np.sqrt(T)
        
        ones_T = self._ones5
                
        w0 = self.uniform_weights
        
        active0 = (R @ w0) - b_vec
        
        mean_a0 = float(active0.mean())
        
        te0 = float(np.linalg.norm(active0) / sqrtT)
        
        te0 = max(te0, self._denom_floor)
        
        lam = mean_a0 / te0

        for _ in range(max_iter):
          
            w = cp.Variable(n, nonneg = True)
          
            active = R @ w - b_vec
          
            mean_a = (ones_T @ active) / T
          
            te_rms = cp.norm(active, 2) / sqrtT
          
            te_eps = cp.maximum(te_rms, self._denom_floor)
            
            obj = cp.Maximize(mean_a - lam * te_eps)

            cons = self.build_constraints(
                w_var = w, 
            )
          
            prob = cp.Problem(obj, cons)
            
            if not self._solve(prob) or w.value is None:
            
                raise RuntimeError("MIR@Dinkelbach: infeasible")

            w_opt = w.value
            
            active_np = (R @ w_opt) - b_vec
            
            mean_a_v = float(active_np.mean())
            
            te = max(float(np.linalg.norm(active_np) / sqrtT), self._denom_floor)

            if abs(mean_a_v - lam * te) < tol:
            
                return pd.Series(w_opt, index = self.universe, name = "MIR")

            lam = mean_a_v / te

        raise RuntimeError("MIR@Dinkelbach: no convergence")


    def msp(
        self,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> pd.Series:
        """
        Score-over-CVaR (Rockafellar–Uryasev form) via Dinkelbach.

        Objective
        ---------
        Let r = R w (weekly portfolio returns), s ∈ ℝ^n be asset scores, and α ∈ (0,1).
        Rockafellar–Uryasev CVaR epigraph (exact, no smoothing in the subproblem):

        Introduce variables z ∈ ℝ and u ∈ ℝ^T with u ≥ 0, u ≥ −(r + z). Then
            
            CVaR_α(r) = min_{z,u≥0}  z + (1/(α T)) ∑ u_i  subject to  u_i ≥ −(r_i + z).

        The Dinkelbach subproblem at λ:
           
            maximise   sᵀ w − λ * ( z + (1/(α T)) ∑ u_i )
           
            subject to u ≥ −(R w + z), u ≥ 0, and common constraints.

        Iterate λ ← (sᵀw) / max(CVaR_α(r), denom_floor).

        Returns
        -------
        pd.Series
            Long-only solution to the Score/CVaR problem.
        """

        self._assert_alignment()
        
        univ = self.universe
    
        n = self.n
            
        hist = pd.DataFrame(self.R1, columns = univ)
      
        if hist.empty:
      
            raise ValueError("MSP: no complete weekly data")

        R = hist.to_numpy()
      
        T = R.shape[0]
      
        α = self._cvar_level

        w = cp.Variable(n, nonneg = True)
      
        z = cp.Variable()
      
        u = cp.Variable(T, nonneg = True)
        
        cons = [
            u >= -(R @ w) - z
        ]
        
        cons += self.build_constraints(
            w_var = w, 
        )

        scores_arr = self.scores_arr
      
        f_expr = scores_arr @ w
      
        g_expr = z + (1.0 / (α * T)) * cp.sum(u)

        lam = 0.0
      
        w_val = None
      
        for _ in range(max_iter):
      
            prob = cp.Problem(cp.Maximize(f_expr - lam * g_expr), cons)
      
            if not self._solve(prob) or w.value is None:
      
                raise RuntimeError("MSP: subproblem failed")
      
            w_val = w.value

            f_val = float(scores_arr @ w_val)
           
            z_val = float(z.value)
           
            u_sum = float(np.sum(u.value))
           
            g_val = z_val + (1.0 / (α * T)) * u_sum

            delta = f_val - lam * g_val
           
            if abs(delta) < tol:
           
                break

            lam = f_val / max(g_val, self._denom_floor)

        if w_val is None:
           
            raise RuntimeError("MSP: Dinkelbach did not converge")

        w_opt = np.maximum(w_val, 0.0)
       
        w_opt /= w_opt.sum()
       
        return pd.Series(w_opt, index = univ, name = "msp")


    @staticmethod
    def _lhs_dirichlet1(
        m: int, 
        d: int, 
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Latin Hypercube sampling from Dirichlet(1) on the simplex.

        Method
        ------
        Sample Exp(1) via stratified uniforms per coordinate, then normalise rows to sum 1.
        Used to mix base portfolios or produce random convex combinations with better space-filling.

        Parameters
        ----------
        m : int
            Number of samples (rows).
        d : int
            Dimension (columns).
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (m, d)
            Row-stochastic matrix of mixing weights.
        """

        U = np.empty((m, d))

        for j in range(d):

            strata = (np.arange(m) + rng.random(m)) / m

            rng.shuffle(strata)

            U[:, j] = strata

        E = -np.log(U)  
       
        W = E / E.sum(axis = 1, keepdims = True)
      
        return W


    @staticmethod
    def _mix_base(
        W_stack: np.ndarray, 
        A_mix: np.ndarray
    ) -> np.ndarray:
        """
        Linearly mix base portfolio stack W_stack with rows of A_mix.

        If W_stack has shape (nbases, nassets) and A_mix has shape (k, nbases),
        returns k mixed portfolios:  A_mix @ W_stack.
        """

        return A_mix @ W_stack
    

    @staticmethod
    def _safe_scale(
        target: float,
        weight: float,
        avg_norm: float,
        eps: float = 1e-8,
        cap: float = 1e6
    ) -> float:
        """
        Stable rescaling factor to equalise average gradient norms.

        Returns s ≈ target / (weight * avg_norm), clipped to [1/cap, cap] to avoid
        pathological multipliers during calibration.
        """

        s = target / (weight * max(avg_norm, eps))
       
        return float(np.clip(s, 1 / cap, cap))
    
    
    def avg_norm(
        self,
        G
    ): 
        """
        Average ℓ2 norm across rows of a gradient matrix G (diagnostic helper).
        """

        return float(np.mean(np.linalg.norm(G, axis = 1)))

    
    @functools.cached_property
    def ew_excess_er(
        self
    ) -> float:
        """
        Equal-weighted expected excess return.

        Returns
        -------
        float
            (μᵀ w_eq − r_f), with w_eq the uniform feasible portfolio.
        """

        return (self.er_arr @ self.uniform_weights) - self.rf_ann


    @functools.cached_property
    def _denom_floor(
        self, 
        lam = 0.97
    ):
        """
        Adaptive denominator floor from an EW portfolio’s EWMA volatility.

        Computes an EWMA of weekly squared returns of the uniform portfolio, annualises
        by sqrt(52), and returns max(1e-8, 1e-3 * σ_EWMA). Used to guard denominators.
        """ 
        
        r = self.R1 @ self.uniform_weights
        
        v = 0.0
        
        for x in r:
        
            v = lam * v + (1 - lam) * (x ** 2)
      
        sig = np.sqrt(v) * self.sqrt52
      
        return max(1e-8, 1e-3 * sig) 
    
    
    @functools.cached_property
    def _denom_floor_week(
        self
    ):
        """
        Weekly version of `_denom_floor` obtained by dividing by sqrt(52).
        """  

        return self._denom_floor / self.sqrt52


    def _project_feasible(
        self,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Projection onto the feasible set via a quadratic programme.

        Problem
        -------
            minimise  ||w − target||_2^2
        
        subject to common constraints (budget, boxes, group caps, etc.).

        If the solver fails, fall back to clipped renormalisation or uniform weights.

        Parameters
        ----------
        target : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
            The feasible point closest (in ℓ2) to `target`.
        """

     
        w = cp.Variable(self.n, nonneg = True)
       
        obj = cp.Minimize(cp.sum_squares(w - target))
        
        cons = self.build_constraints(
            w_var = w
        )
        
        prob = cp.Problem(obj, cons)
       
        if not self._solve(prob) or w.value is None:

            w0 = np.clip(target, 0, None)

            s = w0.sum()
            
            if s > 0:

                return (w0 / s)  
            
            else:
                
                return self.uniform_weights

        return w.value


    def _feasible_seeds(
        self,
        nb_spikes: int,
        nb_rand_objs: int,
        rng: np.random.Generator
    ) -> List[np.ndarray]:
        """
        Generate feasible initial seeds: spikes and random LP solutions.

        Method
        ------
        1) “Spike” seeds: project e_i onto the feasible set for up to `nb_spikes` assets.
       
        2) Random linear objectives: repeatedly solve
       
                maximise cᵀ w  subject to constraints
       
        for random c, to obtain diverse extreme points.

        Returns
        -------
        list[np.ndarray]
            Feasible seeds to warm-start non-convex routines.
        """

        seeds = []

        k = min(nb_spikes, self.n)

        for i in range(k):

            e = np.zeros(self.n)
            
            e[i] = 1.0

            seeds.append(
                
                self._project_feasible(
                    target = e
                )
            )
            
        c_param = cp.Parameter(self.n)
        
        w = cp.Variable(self.n, nonneg = True)
        
        cons = self.build_constraints(
            w_var = w,
            single_port_cap = True
        )
        
        prob = cp.Problem(cp.Maximize(c_param @ w), cons)
                
        for _ in range(nb_rand_objs):
           
            c = rng.random(self.n)
            
            c_param.value = c
                        
            if self._solve(prob) and w.value is not None:
            
                seeds.append(w.value.copy())
        
        return seeds


    def black_litterman_weights(
        self,
        delta: float = 2.5,
        tau: float = 0.02,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Black–Litterman posterior MSR portfolio.

        Pipeline
        --------
        1) Prior equilibrium weights w_prior from market caps restricted to passed screens.
        
        2) Identity views P = I with view vector Q = er (ticker-aligned).
        
        3) View uncertainty Ω diagonal from `comb_std`² with a small floor proportional
        to the average prior variance.
        
        4) Prior covariance Σ_prior projected to nearest PSD and slightly regularised.
        
        5) Posterior (μ_bl, Σ_bl) from the BL update:
        
            μ_bl = [ (τ Σ_prior)^{-1} + Pᵀ Ω^{-1} P ]^{-1} [ (τ Σ_prior)^{-1} Π + Pᵀ Ω^{-1} Q ],
        
        with Π the implied equilibrium returns from w_prior and risk aversion δ,
        and posterior covariance computed accordingly (see `black_litterman` helper).
        
        6) Solve MSR with (μ_bl, Σ_bl) via Dinkelbach.

        Returns
        -------
        (w_bl, mu_bl, sigma_bl) : tuple[pd.Series, pd.Series, pd.DataFrame]
            Posterior MSR weights and BL posterior moments aligned to the universe.
        """
        
        self._assert_alignment()

        if self._mu_bl is not None and self._sigma_bl is not None:

            mu_bl = self._mu_bl

            sigma_bl = self._sigma_bl

        else:

            if self._comb_std is None or self._sigma_prior is None or self._mcap is None:

                raise ValueError("BL inputs missing: comb_std, sigma_prior, mcap required.")

            tickers = self.universe

            mask = (self._scores > 0) & (self._er > 0)

            total = self._mcap[mask].sum()

            if total <= 0:

                raise ValueError("BL: nothing passes >0 screen")

            w_prior = pd.Series(0.0, index = tickers)

            w_prior.loc[mask] = self._mcap.loc[mask] / total

            P = pd.DataFrame(np.eye(len(tickers)), index = tickers, columns = tickers)
           
            Q = self._er
           
            omega_diag = np.asarray(self._comb_std.reindex(tickers).values, float) ** 2
            
            omega_floor = 1e-6 * float(np.mean(np.diag(self._sigma_prior)))
            
            omega_diag = np.clip(omega_diag, omega_floor, np.inf)
            
            Omega = pd.DataFrame(np.diag(omega_diag), index = tickers, columns = tickers)

            Sigma_prior_np = _nearest_psd_preserve_diag(
                C = self._sigma_prior, 
                eps = 1e-10
            )
            
            Sigma_prior_np += 1e-10 * np.eye(Sigma_prior_np.shape[0])
            
            Sigma_prior = pd.DataFrame(Sigma_prior_np, index = tickers, columns = tickers)

            mu_bl, sigma_bl = black_litterman(
                w_prior = w_prior, 
                sigma_prior = Sigma_prior,
                p = P, 
                q = Q, 
                omega = Omega, 
                delta = delta, 
                tau = tau,
                omega_floor = omega_floor
            )
            
            mu_bl = mu_bl.reindex(tickers)
          
            sigma_bl = sigma_bl.reindex(index = self.universe, columns = self.universe)
            
            if (mu_bl.isna().any()) or (sigma_bl.isna().any().any()):
            
                raise ValueError("BL: NaNs after reindexing to universe")
          
            self._mu_bl = mu_bl
            
            self._sigma_bl = sigma_bl

        μ = mu_bl.to_numpy()
        
        Ab = self.Ab
        
        rf = float(self.rf_ann)

        w = self._dinkelbach_sharpe_max(
            μ = μ, 
            A = Ab, 
            rf = rf,
            tol = tol, 
            max_iter = max_iter,
        )
        
        w_bl = pd.Series(w, index = self.universe, name = "BL Weight")
        
        return w_bl, mu_bl, sigma_bl


    def project_feasible_qp(
        self, 
        v,
        lb,
        ub
    ):
        """
        Quadratic projection with the module’s common constraints.

        Problem
        -------
        
            minimise  ||w − v||_2^2
        
        subject to  
        
            sum_i w_i = 1
            
            lb ≤ w ≤ ub
            
            A_caps w ≤ caps_vec,
        
        and any optional extras configured by `build_constraints`.

        If the QP is infeasible or fails numerically, clip v to [lb, ub] and renormalise.

        Parameters
        ----------
        v : array-like, shape (n,)
        lb, ub : array-like, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
            Feasible projection of v.
        """

      
        n = len(v)
       
        w = cp.Variable(n, nonneg = True)
       
        obj = cp.Minimize(cp.sum_squares(w - v))
        
        cons = self.build_constraints(
            w_var = w
        )
              
        prob = cp.Problem(obj, cons)
       
        self._solve(prob)
       
        if w.value is None:
       
            x = np.clip(v, lb, ub)
       
            s = x.sum()
       
            return x / s if s > 0 else np.full(n, 1.0 / n)
       
        return w.value


    def dsr_objective_and_grad(
        self,
        R_weekly: np.ndarray,
        rf_week: float,
        w: np.ndarray,
        EmaxZ: float,
        er_ann_vec: np.ndarray,    
        rf_ann: float,            
        Sigma_ann: np.ndarray,        
        eps: float = 1e-12,
    ):
        """
        Deflated Sharpe ratio (DSR) using annualised Sharpe in the numerator and
        sampling error of weekly Sharpe in the denominator; returns value and gradient.

        Definition
        ----------
        1) Annualised Sharpe from forecasts:
        
            Let 
            
                E_ann(w) = μ_annᵀ w − r_f^ann
                
                σ_ann(w) = sqrt(wᵀ Σ_ann w).
                
                SR_ann(w) = E_ann(w) / σ_ann(w).

        2) Weekly SR used in the sampling-error term:
        
            SR_week(w) = SR_ann(w) / sqrt(52).

        3) From weekly returns R_weekly, compute centred residuals 
        
            z = r − mean(r),
            
        sample standard deviation 
        
            σ_w = sqrt( (1/(T−1)) ∑ z_i² ), 
        
        sample skewness
        
            γ_3 = mean(z³) / σ_w³, 
            
        and sample kurtosis 
        
            γ_4 = mean(z⁴) / σ_w⁴.

        4) The asymptotic variance of SR_week admits a second-order Edgeworth correction,
        yielding (Lo 2002–style):
            
            H(w) = 1 − γ_3 SR_week + 0.25 (γ_4 − 1) SR_week²,
            
            SE{SR_week} = sqrt( H(w) / (T − 1) ).

        5) Deflated Sharpe ratio:
       
            DSR(w) = SR_ann(w) / SE{SR_week(w)} − E_max,
        
        where
        
            E_max = E[max_{i≤N} Z_i], Z_i ~ N(0,1), 
            
        adjusts for multiplicity.

        Gradients (w.r.t. w)
        --------------------
        - dE_ann = μ_ann,  
        
        - dσ_ann = Σ_ann w / σ_ann, 
        
        - hence
          
            dSR_ann = (σ_ann μ_ann − E_ann Σ_ann w / σ_ann) / σ_ann².

        - From weekly data:
        
            Let 
            
                r = R w − r_f^week, 
                
                z = r − mean(r).
                
                d mean(r) = mean(R, axis=0).
                
                ds² = (2/(T−1)) [ Rᵀ z − T * mean(r) * d mean(r) ].
        
                dσ_w = ds² / (2 σ_w).
        
                dγ_3 = d[ mean(z³) / σ_w³ ] 
                
                dγ_4 = d[ mean(z⁴) / σ_w⁴ ] 
                
        via quotient rule, using 
        
            d mean(z^k) = (k/T)[ Rᵀ z^{k−1} − (∑ z^{k−1}) d mean(r) ].

        - d SR_week = d SR_ann / sqrt(52).
        
            dH = −(dγ_3) SR_week − γ_3 d SR_week + 0.25 (dγ_4) SR_week² + 0.5 (γ_4 − 1) SR_week d SR_week.
        
            d SE = 0.5 / SE * dH / (T − 1).

        Finally,
        
            ∇ DSR = ( dSR_ann * SE − SR_ann * dSE ) / SE².

        Parameters
        ----------
        R_weekly : np.ndarray, shape (T, n)
        rf_week : float
        w : np.ndarray, shape (n,)
        EmaxZ : float
        er_ann_vec : np.ndarray, shape (n,)
        rf_ann : float
        Sigma_ann : np.ndarray, shape (n, n)
        eps : float
            Numerical floor for denominators.

        Returns
        -------
        (val, grad) : tuple[float, np.ndarray]
            DSR(w) and its gradient.
        """

        T = self.T1
        
        one = np.ones(T)

        y = R_weekly @ w
        
        r = y - rf_week

        mu_samp = float(r.mean())
       
        z = r - mu_samp
       
        s2 = float(np.dot(z, z) / max(T - 1, 1))
       
        sigma_w = np.sqrt(max(s2, eps))  

        z2 = z ** 2
      
        z3 = z ** 3
      
        m3 = float(np.mean(z3))
      
        m4 = float(np.mean(z ** 4))
      
        skew = m3 / (sigma_w ** 3 + eps)
      
        kurt = m4 / (sigma_w ** 4 + eps)

        dmu_samp = (R_weekly.T @ one) / T

        ds2 = (2.0 / max(T - 1, 1)) * (R_weekly.T @ z - T * mu_samp * dmu_samp)

        dsigma_w = ds2 / (2.0 * max(sigma_w, eps))

        dm3 = (3.0 / T) * (R_weekly.T @ z2 - z2.sum() * dmu_samp)

        dm4 = (4.0 / T) * (R_weekly.T @ z3 - z3.sum() * dmu_samp)

        dskew = dm3 / (sigma_w ** 3 + eps) - 3.0 * skew / max(sigma_w, eps) * dsigma_w

        dkurt = dm4 / (sigma_w ** 4 + eps) - 4.0 * kurt / max(sigma_w, eps) * dsigma_w

        E_ann = float(er_ann_vec @ w - rf_ann)

        dE = er_ann_vec.astype(float)

        Sw = Sigma_ann @ w
       
        q = float(w @ Sw)
       
        sigma_ann = np.sqrt(max(q, eps))
       
        dsigma_ann = Sw / max(sigma_ann, eps)

        SR_ann = E_ann / max(sigma_ann, eps)

        dSR_ann = (max(sigma_ann, eps) * dE - E_ann * dsigma_ann) / (max(sigma_ann, eps) ** 2)
                
        SR_week = SR_ann / self.sqrt52
        
        dSR_week = dSR_ann / self.sqrt52

        H = 1.0 - skew * SR_week + 0.25 * (kurt - 1.0) * (SR_week ** 2)
        
        H = float(max(H, eps))

        dH = (-(dskew) * SR_week - skew * dSR_week + 0.25 * dkurt * (SR_week ** 2) + 0.5 * (kurt - 1.0) * SR_week * dSR_week)
       
        SE = np.sqrt(H / max(T - 1, 1))
       
        dSE = 0.5 / max(SE, eps) * (dH / max(T - 1, 1))
       
        DSR = SR_ann / max(SE, eps) - EmaxZ
       
        dDSR = (dSR_ann * max(SE, eps) - SR_ann * dSE) / (max(SE, eps) ** 2)
       
        return float(DSR), dDSR


    def optimise_deflated_sharpe(
        self,
        R_weekly: np.ndarray | None = None,
        Sigma_ann: np.ndarray | None = None,  
        N_strategies: int = 50,
        starts: list[np.ndarray] | None = None,
        max_iter: int = 200,
        step0: float = 1.0,
        tol: float = 1e-8,
    ):
        """
        Projected gradient ascent for the Deflated Sharpe ratio.

        Algorithm
        ---------
        Starting from each feasible initial weight vector:
        
        1) Compute (value, gradient) of DSR(w) using `dsr_objective_and_grad`.
        
        2) Take a step w + η ∇DSR, then project onto the feasible set via
        `project_feasible_qp`.
        
        3) Accept with Armijo-style improvement (≥ 1e−6). Adapt η up/down. Stop when
        both ||w_new − w|| and |val_new − val| fall below `tol`, or the step underflows.

        Feasible set is the common constraint set (`build_constraints`).

        Parameters
        ----------
        N_strategies : int
            Used to compute E[max Z_i] (multiplicity correction).
        starts : list[np.ndarray] | None
            Optional initialisations; defaults to the uniform feasible portfolio.
        max_iter, step0, tol : control parameters.

        Returns
        -------
        pd.Series
            Best DSR weights over the starts.
        """

        Rw = self.R1 if R_weekly is None else R_weekly
        
        rf_week = self.rf_week
        
        er_ann_vec = self.er_arr
        
        rf_ann = self.rf_ann

        if Sigma_ann is None:
          
            Sigma_ann = self.Σ
       
        if not starts:
         
            starts = [self.uniform_weights]

        best_w = None
        
        best_val = -np.inf
        
        EmaxZ = expected_max_std_normal(
            N = max(1, int(N_strategies))
        )
        
        lb, ub = self.bound_arr
        
        for w_start in starts:
            
            w = self.project_feasible_qp(
                v = w_start,
                lb = lb,
                ub = ub
            )

            step = step0

            for it in range(max_iter):
                
                val, grad = self.dsr_objective_and_grad(
                    R_weekly = Rw,
                    rf_week = rf_week,
                    w = w,
                    EmaxZ = EmaxZ,
                    er_ann_vec = er_ann_vec, 
                    rf_ann = rf_ann,
                    Sigma_ann = Sigma_ann
                )

                v = w + step * grad
               
                w_new = self.project_feasible_qp(
                    v = v,
                    lb = lb,
                    ub = ub
                )

                val_new, _ = self.dsr_objective_and_grad(
                    R_weekly = Rw,
                    rf_week = rf_week,
                    w = w_new,
                    EmaxZ = EmaxZ,
                    er_ann_vec = er_ann_vec,
                    rf_ann = rf_ann,
                    Sigma_ann = Sigma_ann
                )

                if val_new >= val + 1e-6:
                   
                    if np.linalg.norm(w_new - w) < tol and abs(val_new - val) < tol:
                   
                        w = w_new
                   
                        val = val_new
                   
                        break
                   
                    w = w_new
                    
                    val = val_new
                    
                    step = min(step * 1.25, 10.0)
               
                else:
               
                    step *= 0.5
               
                    if step < 1e-6:
               
                        break

            final_val, _ = self.dsr_objective_and_grad(
                R_weekly = Rw, 
                rf_week = rf_week,
                w = w,
                EmaxZ = EmaxZ,
                er_ann_vec = er_ann_vec,
                rf_ann = rf_ann,
                Sigma_ann = Sigma_ann
            )
            
            if final_val > best_val:
               
                best_val = final_val 
                
                best_w = w.copy()
                     
        return pd.Series(best_w, index = self.universe, name = "Deflated MSR Weight")


    def _asr_grad_batch(
        self, 
        W_chunk: np.ndarray, 
        eps: float = 1e-12
    ):
        """
        Adjusted Sharpe ratio (ASR) value and gradient for a batch of portfolios.

        Definition
        ----------
       
        Let 
        
            SR(w) = (μᵀ w − r_f) / σ(w) 
        
        where 
        
            σ(w) = sqrt(wᵀ Σ w). 
            
        Using weekly data R, compute sample skewness γ_3 and kurtosis γ_4 of the portfolio weekly excess
        returns 
        
            r = R w − mean(R w).

        A second-order adjustment (Cornish–Fisher / Edgeworth style) yields
       
            ASR(w) = SR(w) * [ 1 + (γ_3 SR(w))/6 + ((γ_4 − 3) SR(w)²)/24 ].

        Batch notation
        --------------
        Given W_chunk with rows w^{(k)}, k = 1..K:
       
        - For each k, compute r^{(k)}, its centred residuals Z, sample σ_w, γ_3, γ_4.
       
        - Forecast Sharpe uses annual μ, Σ, r_f:
       
            E_k = μᵀ w^{(k)} − r_f,
       
            σ_k = sqrt( (w^{(k)})ᵀ Σ w^{(k)} ),
       
            SR_k = E_k / σ_k.

        Gradients
        ---------
        Write
        
            s0 = SR
            
            g1 = 1 + (γ_3 s0) / 6 + ((γ_4 − 3) s0²) / 24.
        
        Then
        
            ASR = s0 * g1.

        By product / chain rules (holding weekly moments as functions of w):
        
            d s0 = (σ μ − E Σw / σ) / σ²,
        
        d γ_3 and d γ_4 as in the DSR derivation (with y = R w − mean(R w)).

        Conveniently,
        
            d ASR = C1 * d s0 + (s0²/6) dγ_3 + (s0³ / 24) dγ_4,
        
        where
        
            C1 = 1 + (γ_3 s0) / 3 + ((γ_4 − 3) s0²) / 8.

        Implementation notes
        --------------------
        - Returns `dASR` as shape (n, K): each column is ∇ ASR(w^{(k)}).
        - Also returns `norms` (ℓ2 norms of the gradients) and `ASR_vals` (scalar ASR per k).
        - All denominators are floored by `eps` for stability.

        Parameters
        ----------
        W_chunk : np.ndarray, shape (K, n)
        eps : float

        Returns
        -------
        norms : np.ndarray, shape (K,)
            2-norms of gradients per portfolio.
        dASR : np.ndarray, shape (n, K)
            Gradient matrix (columns correspond to portfolios).
        ASR_vals : np.ndarray, shape (K,)
            Scalar ASR values per portfolio.
        """

        
        R = self.R1          
        
        RT = self.R1T        
        
        T = self.T1
        
        one = np.ones(T)
        
        WcT = W_chunk.T  

        Rw = R @ WcT              
       
        mu = Rw.mean(axis = 0)          
       
        Z = Rw - mu[None, :]            
       
        Z2 = Z * Z
       
        Z3 = Z2 * Z

        s2 = Z2.sum(axis = 0) / max(T - 1, 1)             
       
        sigma_w = np.sqrt(np.maximum(s2, eps))               
       
        m3 = Z3.mean(axis = 0)                                 
       
        m4 = (Z2 * Z2).mean(axis = 0)                         
        
        skew = m3 / (sigma_w ** 3 + eps)                    
        
        kurt = m4 / (sigma_w ** 4 + eps)                   

        dmu = (RT @ one) / T                               

        RTZ = RT @ Z      
                                            
        ds2 = (2.0 / max(T - 1, 1)) * (RTZ - (dmu[:, None] * (T * mu[None, :])))

        dsigma_w = ds2 / (2.0 * np.maximum(sigma_w[None, :], eps))

        z2_sum = Z2.sum(axis = 0)     
                                  
        z3_sum = Z3.sum(axis = 0)                           

        dm3 = (3.0 / T) * ((RT @ Z2) - dmu[:, None] * z2_sum[None, :])    
       
        dm4 = (4.0 / T) * ((RT @ Z3) - dmu[:, None] * z3_sum[None, :])   

        denom_s3 = (sigma_w**3 + eps)[None, :]
       
        denom_s4 = (sigma_w**4 + eps)[None, :]

        dskew = dm3 / denom_s3 - 3.0 * (skew[None, :] / np.maximum(sigma_w[None, :], eps)) * dsigma_w
        
        dkurt = dm4 / denom_s4 - 4.0 * (kurt[None, :] / np.maximum(sigma_w[None, :], eps)) * dsigma_w

        mu_ann = self.er_arr @ WcT                  
        
        E = mu_ann - self.rf_ann                            

        SigmaW = self.Σ @ WcT                         
       
        q = np.einsum("kn,nk->k", W_chunk, SigmaW)          
       
        sigma_ann = np.sqrt(np.maximum(q, eps))               

        dE = self.er_arr[:, None]                              
        
        dsigma = SigmaW / np.maximum(sigma_ann[None, :], eps)   

        dSR = (np.maximum(sigma_ann[None, :], eps) * dE - E[None, :] * dsigma) / (np.maximum(sigma_ann[None, :], eps) ** 2)
       
        s0 = E / np.maximum(sigma_ann, eps)                    
        
        g1 = 1.0 + (skew * s0) / 6.0 + ((kurt - 3.0) * (s0 ** 2)) / 24.0
       
        C1 = 1.0 + (skew * s0) / 3.0 + ((kurt - 3.0) * (s0 ** 2)) / 8.0

        dASR = C1[None, :] * dSR + (s0[None, :] ** 2) * (dskew / 6.0) + (s0[None, :] ** 3) * (dkurt / 24.0)  

        norms = np.sqrt(np.sum(dASR * dASR, axis = 0))    
       
        ASR_vals = s0 * g1 
       
        return norms, dASR, ASR_vals


    def _asr_val_grad_single(
        self, 
        w: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Single-portfolio wrapper around `_asr_grad_batch`.

        Parameters
        ----------
        w : np.ndarray, shape (n,)

        Returns
        -------
        (val, grad) : tuple[float, np.ndarray]
            ASR(w) and its gradient.
        """
    
        w1 = np.asarray(w, float).reshape(1, -1)
    
        norms, dASR, vals = self._asr_grad_batch(
            W_chunk =w1
        )
    
        return float(vals[0]), dASR[:, 0]
    
    
    def optimise_adjusted_sharpe(
        self,
        starts: list[np.ndarray] | None = None,
        max_iter: int = 200,
        step0: float = 1.0,
        tol: float = 1e-8,
    ) -> pd.Series:
        """
        Projected gradient ascent for the Adjusted Sharpe ratio (ASR).

        Objective
        ---------
      
            ASR(w) = SR(w) * [ 1 + (γ_3 SR(w))/6 + ((γ_4 − 3) SR(w)²)/24 ],
        
        where:
        
        - SR(w) = (μᵀ w − r_f) / sqrt(wᵀ Σ w) uses forecast μ, Σ, r_f (annual).
        
        - γ_3, γ_4 are sample skewness and kurtosis of weekly portfolio returns R w.

        Constraints
        -----------
        Feasible set from `build_constraints`:
        
        - sum_i w_i = 1,
        
        - lb_i ≤ w_i ≤ ub_i and w_i ≥ 0 (gated boxes),
        
        - industry/sector caps A_caps w ≤ caps_vec,
        
        - (optional, if enabled in the projection helper) single-portfolio envelope.

        Algorithm
        ---------
        From each feasible start:
        
        1) Evaluate (ASR, ∇ASR) via `_asr_grad_batch(w[None, :])`.
        
        2) Consider a short backtracking schedule of step sizes {η, η/2, η/4, …},
        project each trial w + η ∇ASR onto the feasible set with `project_feasible_qp`,
        then batch-evaluate ASR on all trial points.
        
        3) Accept the first trial with improvement ≥ 1e−6; expand step if the full
        step was accepted, otherwise keep the accepted η. Terminate when both the
        weight change and objective improvement fall below `tol`, or η underflows.

        Returns
        -------
        pd.Series
            The best-ASR feasible portfolio over the provided starts.
        """


        if not starts:

            starts = [self.uniform_weights]

        best_w, best_val = None, -np.inf
        
        lb, ub = self.bound_arr

        for w0 in starts:

            w = self.project_feasible_qp(
                v = np.asarray(w0, float),
                lb = lb,
                ub = ub
            )
            
            step = step0

            MAX_BACKTRACKS = 4  
                   
            ARMİJO_SLACK  = 1e-6   

            for _ in range(max_iter):
              
                _, dASR_cur, vals_cur = self._asr_grad_batch(
                    W_chunk = w[None, :]
                )
              
                grad = dASR_cur[:, 0]
              
                val  = float(vals_cur[0])

                etas = [step * (0.5 ** j) for j in range(MAX_BACKTRACKS)]

                W_trials = []
                
                for eta in etas:
                
                    v = w + eta * grad
                
                    W_trials.append(self.project_feasible_qp(
                        v = v,
                        lb = lb, 
                        ub = ub
                    ))
                
                W_trials = np.vstack(W_trials)  

                _, _, vals_trials = self._asr_grad_batch(
                    W_chunk = W_trials
                )  

                picked = -1
               
                for j, val_new in enumerate(vals_trials):
               
                    if float(val_new) >= val + ARMİJO_SLACK:
               
                        picked = j
               
                        break

                if picked >= 0:
               
                    w_new  = W_trials[picked]
               
                    val_new = float(vals_trials[picked])

                    if picked == 0:
               
                        step = min(step * 1.25, 10.0)
               
                    else:
               
                        step = etas[picked]  

                    if np.linalg.norm(w_new - w) < tol and abs(val_new - val) < tol:
               
                        w = w_new
               
                        val = val_new
               
                        break

                    w = w_new
               
                    val = val_new
               
                    continue

                step *= 0.5
               
                if step < 1e-6:
               
                    break

            final_val, _ = self._asr_val_grad_single(
                w = w
            )
            
            if final_val > best_val:
            
                best_val = final_val
                
                best_w = w.copy()

        if best_w is None:
            
            raise RuntimeError("optimise_adjusted_sharpe: no feasible solution found")

        return pd.Series(best_w, index = self.universe, name = "ASR Weight")


    def _initial_mixes_L2(
        self, 
        w_msr, 
        w_sortino, 
        w_mir,
        w_bl, 
        w_msp, 
        sample_size = 2000, 
        random_state = 42
    ):
        """
        Heuristic initializer: choose a convex combination of base portfolios that scores best
        under a simple, differentiability-friendly proxy.

        Inputs
        ------
        w_msr, w_sortino, w_mir, w_bl, w_msp : array-like, shape (n,)
            Base feasible portfolios (MSR, Sortino, MIR, Black–Litterman MSR, MSP).
        sample_size : int
            Number of Latin–Hypercube Dirichlet(1) mixes to evaluate.
        random_state : int
            RNG seed.

        Method
        ------
        1) Stack the base portfolios into W_stack ∈ R^{5×n}.
       
        2) Draw A_mix ∈ R^{m×5} with each row on the simplex (Dirichlet(1) via Latin–Hypercube).
       
        3) Form W_mix = A_mix W_stack; each row is a candidate feasible mix.
       
        4) Score each candidate w using the proxy score:
            score(w) = Sharpe(w) + Sortino(w) + BL_Sharpe(w) − penalty_L2(w),
       
        where:
       
        - Sharpe(w) = (μᵀ w − r_f) / vol(w), with vol(w) = sqrt(wᵀ Σ w).
       
        - Sortino(w) uses weekly downside deviation D_week(w) = ||max(r_f^week − R w, 0)||_2 / sqrt(T),
            annualised by sqrt(52); numerator is μᵀ w − r_f^ann.
       
        - BL_Sharpe(w) is Sharpe computed with (μ_bl, Σ_bl).
       
        - penalty_L2(w) = ||w − w_mir||_2^2 / den_mir0 + ||w − w_msp||_2^2 / den_msp0,
            with denominators den_mir0, den_msp0 taken as the *average* over W_mix to keep the
            two penalty channels comparable. A small floor eps is applied.

        5) Return the argmax of the proxy score.

        Numerical safeguards
        --------------------
        - Quadratic forms and denominators are floored by `_denom_floor` / eps to avoid division by 0.
        - All computations are purely on candidate mixes; feasibility holds as each mix is a convex
        combination of feasible bases.

        Returns
        -------
        np.ndarray, shape (n,)
            The highest-scoring initial mix to be used as a warm start.
        """
 
        rng = np.random.default_rng(random_state)
        
        W = np.vstack([w_msr, w_sortino, w_mir, w_bl, w_msp])

        A_mix = self._lhs_dirichlet1(
            m = sample_size, 
            d = 5, 
            rng = rng
        )
        
        W_mix = self._mix_base(
            W_stack = W, 
            A_mix = A_mix
        )

        T = self.T1

        rf_ann = self.rf_ann

        rf_week = self.rf_week
       
        scores = np.zeros(W_mix.shape[0])
               
        eps = 1e-12
        
        den_mir0 = max(np.mean(((W_mix - w_mir) ** 2).sum(axis = 1)), eps)
        
        den_msp0 = max(np.mean(((W_mix - w_msp) ** 2).sum(axis = 1)), eps)
       
        for i, w in enumerate(W_mix):
       
            ret = float(self.er_arr @ w)
       
            vol, Σw = self._quad_vol(
                w = w, 
                Σ = self.Σ
            )
       
            dd_week = float(np.linalg.norm(np.maximum(rf_week - (self.R1 @ w), 0.0)) / np.sqrt(T))
       
            dd_ann = max(dd_week * self.sqrt52, self._denom_floor)
       
            bl_ret = float(self._mu_bl.values @ w)
       
            bl_vol, Σb_w = self._quad_vol(
                w = w, 
                Σ = self.Σb
            )

            exc = ret - rf_ann
            
            sh = exc / vol
            
            so = exc / max(dd_ann, self._denom_floor)
            
            bl = (bl_ret - rf_ann) / bl_vol
            
            pen_comb = (((w - w_mir) ** 2).sum() / den_mir0) + (((w - w_msp) ** 2).sum() / den_msp0)
            
            scores[i] = sh + so + bl - pen_comb

        return W_mix[np.argmax(scores)]


    def _cvar_ru_smooth_batch(
        self, 
        R1W: np.ndarray,
        alpha: float,
        beta: float
    ):
        """
        Smoothed Rockafellar–Uryasev CVaRα for multiple portfolios, with gradient w.r.t. per-period returns.

        Definitions
        -----------
        For a given portfolio’s weekly returns r ∈ R^T, the exact RU representation is
         
            CVaR_α(r) = min_z  z + (1/(α T)) ∑_i max(0, −(r_i + z)).
        
        This function uses a smooth surrogate of the hinge via softplus:
        
            softplus(x; β) = (1/β) log(1 + exp(β x)),
        
            σ(x; β) = 1 / (1 + exp(−β x))  (sigmoid).

        Smoothed objective for fixed β > 0:
        
            C_α^β(r) = z* + (1/(α T)) ∑_i softplus( −(r_i + z*) ; β ),

        where z* solves the first-order optimality condition of the RU surrogate:
        
            ∑_i σ(β (−(r_i + z*))) = α T.
        
        This code computes z* per column by bisection in a robust bracket around the empirical losses.

        Gradients
        ---------
        By the envelope theorem (treating z* as the argmin),
        
            ∂ C_α^β / ∂ r_i = − (1/(α T)) σ(β (−(r_i + z*))).
        
        Hence, for a matrix of portfolio return paths R1W ∈ R^{T×K} (T weeks, K portfolios),
        the function returns:
        
            cvar : length-K vector of C_α^β values,
        
            dC_dr : T×K matrix with [dC_dr]_{i,k} = ∂ C_α^β / ∂ r_i^{(k)}.

        Parameters
        ----------
        R1W : np.ndarray, shape (T, K)
            Weekly portfolio returns for K portfolios (each column a portfolio path).
        alpha : float
            CVaR tail probability in (0,1).
        beta : float
            Softplus/sigmoid temperature; larger values approximate the hinge more tightly.

        Returns
        -------
        cvar : np.ndarray, shape (K,)
            Smoothed RU CVaR values for each portfolio.
        dC_dr : np.ndarray, shape (T, K)
            Gradients of C_α^β with respect to the T weekly returns of each portfolio.

        Notes
        -----
        - The gradient w.r.t. weights is obtained upstream as ∇_w C = Rᵀ dC_dr.
        - Bracketing for the bisection uses percentiles of −R1W to locate a stable search interval.
        - All divisions are robustified by small floors to avoid numerical issues.
        """

        T = self.T1
        
        alpha_T = alpha * T
               
        Z = -R1W  
       
        lo = np.percentile(Z, 100 * alpha * 0.5, axis = 0)
       
        hi = np.percentile(Z, 100 * min(0.99, alpha * 1.5), axis = 0)
       
        for _ in range(40):
       
            mid = 0.5 * (lo + hi)                    
          
            sig = 1.0 / (1.0 + np.exp(-beta * (-R1W - mid))) 
            
            g = 1.0 - sig.sum(axis = 0) / alpha_T
            
            pick_lo = (g > 0)
            
            lo = np.where(pick_lo, mid, lo)
          
            hi = np.where(~pick_lo, mid, hi)
     
        z_star = 0.5 * (lo + hi)                       
        
        A = -R1W - z_star[None, :]         
        
        bA = beta * A       
        
        sp = np.log1p(np.exp(bA)) / beta          
        
        cvar = z_star + sp.sum(axis = 0) / alpha_T    
        
        sig = 1.0 / (1.0 + np.exp(-bA))
      
        dC_dr = -(sig / (alpha * T))                 
      
        return cvar, dC_dr


    @functools.cached_property
    def _W_rand_vals(
        self
    ):
        """
        Calibration statistics from random feasible mixes and extremal seeds.

        Purpose
        -------
        Produce *comparable scales* for multi-term objectives by estimating the typical
        magnitude of gradient norms for each constituent term under realistic portfolios.
        This enables scale factors that make each term contribute similar “push” during
        CCP or line-search procedures.

        Sampling and sketching
        ----------------------
        - Base set W_base = [MSR, Sortino, MIR, BL, MSP].
       
        - Generate K ≈ `sample_size` random convex mixes via Dirichlet(1) LHS on the simplex.
       
        - Augment with additional feasible seeds (projected spikes e_i and random LP extremes).
       
        - For computational economy, IR(5y) is computed on a subsample of the 5y history
        (uniform without replacement up to `CALIB_T5_SUB` rows).

        For each candidate W_chunk (rows are portfolios), compute:

        1) Sharpe (S)
       
        Gradient per portfolio k:
       
            ∇ S = (μ σ − excess Σ w / σ) / σ²,
       
        where excess = μᵀ w − r_f and σ = sqrt(wᵀ Σ w).
        Accumulate average ℓ2 norms to produce A_s.

        2) Sortino (SO)
       
        Weekly downside deviation 
        
            D_week(w) = ||max(r_f^week − R w, 0)||_2 / sqrt(T).
        
        Annualise by sqrt(52). Using chain rule,
        
            ∇ D_week = − Rᵀ pos(u) / (T * D_week),  
            
        where 
        
            u = r_f^week − R w,
        
        and ∇ Sortino follows the quotient rule. Accumulate norms into A_so.

        3) BL Sharpe (BL)
        Same as (1) but with (μ_bl, Σ_bl) to yield A_bl.

        4) Information Ratio (IR 1y and IR 5y)
        Active returns 
        
            a = R w − b. TE = ||a||_2 / sqrt(T).
        
        Mean active gradient is mean(R) and
        
            ∇ TE = Rᵀ a / (T * TE).
        
        Apply quotient rule to get ∇ IR and accumulate into A_ir_1y and A_ir_5y.

        5) Score/CVaR (SC) with smoothed RU CVaR
        
        Using `_cvar_ru_smooth_batch` to obtain 
        
            C = C_α^β and dC_dr, 
        
        then
        
            ∇_w C = Rᵀ dC_dr,
        
            ∇ SC = (scores * C − (scoresᵀ w) ∇ C) / C².
        
        Accumulate norms into A_sc.

        6) Adjusted Sharpe (ASR)
        Use `_asr_grad_batch` to obtain gradient norms and sum into A_asr.

        Penalty denominators
        --------------------
        Compute L1 and L2 baselines for penalties against MIR and MSP anchors:
        
            den_mir_l1 = average ||w − w_mir||_1 over candidates (floored by eps),
        
            den_msp_l1 = average ||w − w_msp||_1,
        
            den_mir_l2 = average ||w − w_mir||_2^2,
        
            den_msp_l2 = average ||w − w_msp||_2^2.
        
        Also compute average *gradient magnitudes* for BL-centred penalties (L1 and L2)
        to calibrate penalty scales in BL-regularised composites.

        Returns
        -------
        dict
            {
            "A_s", "A_so", "A_bl", "A_ir_1y", "A_ir_5y", "A_sc", "A_asr",
            "den_mir_l1", "den_msp_l1", "den_mir_l2", "den_msp_l2",
            "A_pen_mir_msp_l1", "A_pen_mir_msp_l2", "A_pen_bl_l1", "A_pen_bl_l2",
            }

        Notes
        -----
        - Chunked computation reduces peak memory and improves cache locality.
        - All denominators are floored by small eps or `_denom_floor` variants.
        """

        w_bl, μ_bl_arr, Σb = self._bl
        
        w_mir = self._mir
        
        w_msp = self._msp

        rng = np.random.default_rng(self.random_state)

        W_base = np.vstack([self._msr, self._sortino, w_mir, w_bl, self._msp])

        A_mix = self._lhs_dirichlet1(
            m = self.sample_size,
            d = W_base.shape[0],
            rng = rng
        )

        seeds = self._feasible_seeds(
            nb_spikes = min(10, self.n),
            nb_rand_objs = 10,
            rng = rng
        )
        
        if len(seeds) > 0:
        
            seeds = np.vstack(seeds)
        
        else:
        
            seeds = np.zeros((0, self.n))

        R1 = self.R1
       
        R1T = self.R1T
       
        T1 = self.T1
        
        sqrtT1 = self.sqrtT1

        R5_full, b5_full = self._R5_b_te
        
        T5 = self.T5
        
        t5_sub = min(T5, int(getattr(config, "CALIB_T5_SUB", 156)))
       
        if t5_sub >= T5:
       
            idx5 = np.arange(T5)
       
        else:
       
            idx5 = rng.choice(T5, size = t5_sub, replace = False)   

        R5  = R5_full[idx5]
       
        R5T = R5.T

        if b5_full is not None:
       
            b5 = b5_full[idx5]
       
        else:
       
            b5 = np.full(R5.shape[0], self.rf_week)

        sqrtT5 = self.sqrtT5

        Σ = self.Σ
      
        er = self.er_arr
      
        rf_a = self.rf_ann
      
        rf_w = self.rf_week
      
        sqrt52 = self.sqrt52
      
        denom = self._denom_floor
      
        denom_w = self._denom_floor_week
      
        scores = self.scores_arr
      
        α = self._cvar_level

        K = A_mix.shape[0]
        
        grad_m1 = self._grad_m1
        
        grad_m5 = self._grad_m5
        
        chunk_k = 512

        cnt = 0

        sum_A_s = 0.0
        
        sum_A_so = 0.0
        
        sum_A_bl = 0.0
      
        sum_A_ir1 = 0.0
        
        sum_A_ir5 = 0.0
        
        sum_A_sc = 0.0
      
        sum_A_asr = 0.0                       

        sum_den_mir_l1 = 0.0
        
        sum_den_msp_l1 = 0.0
        
        sum_den_mir_l2 = 0.0
        
        sum_den_msp_l2 = 0.0


        def process_chunk(
            W_chunk
        ):
        
            nonlocal cnt, sum_A_s, sum_A_so, sum_A_bl, sum_A_ir1, sum_A_ir5, sum_A_sc, sum_A_asr
        
            nonlocal sum_den_mir_l1, sum_den_msp_l1, sum_den_mir_l2, sum_den_msp_l2

            k = W_chunk.shape[0]
        
            cnt += k
            
            WcT = W_chunk.T

            ΣW = Σ @ WcT
           
            q = np.einsum("kn,nk->k", W_chunk, ΣW)
           
            vol = np.sqrt(np.maximum(q, denom))
           
            ret = er @ WcT
           
            excess = ret - rf_a
           
            g_sh_nK = (er[:, None] * vol[None, :] - excess[None, :] * (ΣW / vol[None, :])) / (vol[None, :] ** 2)
           
            sum_A_s += float(np.linalg.norm(g_sh_nK, axis = 0).sum())

            R1W = R1 @ WcT

            u = rf_w - R1W

            pos_u = np.clip(u, 0.0, None)

            dd_week = np.linalg.norm(pos_u, axis = 0) / sqrtT1

            dd_week_eps = np.maximum(dd_week, denom_w)
           
            grad_dd_week = -(R1T @ pos_u) / (T1 * dd_week_eps[None, :])
           
            dd_ann = np.maximum(dd_week * sqrt52, denom)
           
            grad_dd = grad_dd_week * sqrt52
           
            g_so_nK = (er[:, None] * dd_ann[None, :] - excess[None, :] * grad_dd) / (dd_ann[None, :] ** 2)
           
            sum_A_so += float(np.linalg.norm(g_so_nK, axis = 0).sum())

            ΣbW = Σb @ WcT

            qb = np.einsum("kn,nk->k", W_chunk, ΣbW)

            bl_vol = np.sqrt(np.maximum(qb, denom))

            bl_ret = μ_bl_arr @ WcT

            g_bl_nK = (μ_bl_arr[:, None] * bl_vol[None, :] - (bl_ret - rf_a)[None, :] * (ΣbW / bl_vol[None, :])) / (bl_vol[None, :] ** 2)

            sum_A_bl += float(np.linalg.norm(g_bl_nK, axis = 0).sum())

            a1 = R1W - rf_w
           
            mean_a1 = a1.mean(axis = 0)
           
            te1 = np.linalg.norm(a1, axis = 0) / sqrtT1
          
            te1_eps = np.maximum(te1, denom)
          
            dte1_nK = (R1T @ a1) / (te1_eps[None, :] * T1)
          
            g_ir1_nK = (grad_m1[:, None] * te1_eps[None, :] - mean_a1[None, :] * dte1_nK) / (te1_eps[None, :] ** 2)
          
            sum_A_ir1 += float(np.linalg.norm(g_ir1_nK, axis = 0).sum())

            R5W = R5 @ WcT

            a5 = R5W - b5[:, None]

            mean_a5 = a5.mean(axis = 0)

            te5 = np.linalg.norm(a5, axis=0) / sqrtT5

            te5_eps = np.maximum(te5, denom)

            dte5_nK = (R5T @ a5) / (te5_eps[None, :] * R5.shape[0])
            
            g_ir5_nK = (grad_m5[:, None] * te5_eps[None, :] - mean_a5[None, :] * dte5_nK) / (te5_eps[None, :] ** 2)

            sum_A_ir5 += float(np.linalg.norm(g_ir5_nK, axis = 0).sum())

            cvar, dC_dr = self._cvar_ru_smooth_batch(
                R1W = R1W,
                alpha = α,
                beta = self._cvar_beta_calib
            )
            
            g_cvar_nK = R1T @ dC_dr
            
            f = scores @ WcT
            
            cvar_eps = np.maximum(cvar, denom)
            
            g_sc_nK = (scores[:, None] * cvar_eps[None, :] - f[None, :] * g_cvar_nK) / (cvar_eps[None, :] ** 2)
            
            sum_A_sc += float(np.linalg.norm(g_sc_nK, axis = 0).sum())
                
            asr_norms, _, _ = self._asr_grad_batch(W_chunk)
            
            sum_A_asr += float(asr_norms.sum())
            
            diff_mir = W_chunk - w_mir[None, :]

            diff_msp = W_chunk - w_msp[None, :]

            sum_den_mir_l1 += float(np.sum(np.abs(diff_mir)))

            sum_den_msp_l1 += float(np.sum(np.abs(diff_msp)))

            sum_den_mir_l2 += float(np.sum(diff_mir * diff_mir))

            sum_den_msp_l2 += float(np.sum(diff_msp * diff_msp))

        for lo in range(0, K, chunk_k):

            hi = min(K, lo + chunk_k)

            A_chunk = A_mix[lo: hi]

            W_chunk = A_chunk @ W_base

            process_chunk(
                W_chunk = W_chunk
            )

        if seeds.shape[0] > 0:

            process_chunk(
                W_chunk = seeds
            )

        eps = 1e-12
       
        den_mir_l1 = max(sum_den_mir_l1 / max(cnt, 1), eps)
       
        den_msp_l1 = max(sum_den_msp_l1 / max(cnt, 1), eps)
       
        den_mir_l2 = max(sum_den_mir_l2 / max(cnt, 1), eps)
       
        den_msp_l2 = max(sum_den_msp_l2 / max(cnt, 1), eps)

        A_s = sum_A_s / max(cnt, 1)
      
        A_so = sum_A_so / max(cnt, 1)
      
        A_bl = sum_A_bl / max(cnt, 1)
      
        A_ir_1y = sum_A_ir1 / max(cnt, 1)
      
        A_ir_5y = sum_A_ir5 / max(cnt, 1)
      
        A_sc = sum_A_sc / max(cnt, 1)
      
        A_asr = sum_A_asr / max(cnt, 1)     

        sum_pen_mir_msp_l1 = 0.0
        
        sum_pen_mir_msp_l2 = 0.0
       
        sum_pen_bl_l1 = 0.0
        
        sum_pen_bl_l2 = 0.0
        
        cnt2 = 0


        def process_pen_chunk(
            W_chunk
        ):
        
            nonlocal sum_pen_mir_msp_l1, sum_pen_mir_msp_l2, sum_pen_bl_l1, sum_pen_bl_l2, cnt2
        
            cnt2 += W_chunk.shape[0]

            g1 = np.sign(W_chunk - w_mir[None, :]) / den_mir_l1 + np.sign(W_chunk - w_msp[None, :]) / den_msp_l1
        
            sum_pen_mir_msp_l1 += float(np.linalg.norm(g1, axis = 1).sum())

            g2 = 2.0 * (W_chunk - w_mir[None, :]) / den_mir_l2 + 2.0 * (W_chunk - w_msp[None, :]) / den_msp_l2
            
            sum_pen_mir_msp_l2 += float(np.linalg.norm(g2, axis = 1).sum())

            g_bl1 = np.sign(W_chunk - w_bl[None, :])
            
            g_bl2 = 2.0 * (W_chunk - w_bl[None, :])

            sum_pen_bl_l1 += float(np.linalg.norm(g_bl1, axis = 1).sum())
            
            sum_pen_bl_l2 += float(np.linalg.norm(g_bl2, axis = 1).sum())

    
        for lo in range(0, K, chunk_k):
      
            hi = min(K, lo + chunk_k)
      
            A_chunk = A_mix[lo:hi]
      
            W_chunk = A_chunk @ W_base
      
            process_pen_chunk(
                W_chunk = W_chunk
            )

        if seeds.shape[0] > 0:
           
            process_pen_chunk(
                W_chunk = seeds
            )

        A_pen_mir_msp_l1 = sum_pen_mir_msp_l1 / max(cnt2, 1)
       
        A_pen_mir_msp_l2 = sum_pen_mir_msp_l2 / max(cnt2, 1)
       
        A_pen_bl_l1 = sum_pen_bl_l1 / max(cnt2, 1)
       
        A_pen_bl_l2 = sum_pen_bl_l2 / max(cnt2, 1)

        return {
            "A_s": A_s,
            "A_so": A_so,
            "A_bl": A_bl,
            "A_ir_1y": A_ir_1y,
            "A_ir_5y": A_ir_5y,
            "A_sc": A_sc,
            "A_asr": A_asr,                     
            "den_mir_l1": den_mir_l1,
            "den_msp_l1": den_msp_l1,
            "den_mir_l2": den_mir_l2,
            "den_msp_l2": den_msp_l2,
            "A_pen_mir_msp_l1": A_pen_mir_msp_l1,
            "A_pen_mir_msp_l2": A_pen_mir_msp_l2,
            "A_pen_bl_l1": A_pen_bl_l1,
            "A_pen_bl_l2": A_pen_bl_l2,
        }
        
          
    def _calibrate_scales_by_grad_rewards_penalties(
        self,
        gamma: tuple,
        use_l1_penalty: bool = False,
        eps: float = 1e-12,
    ):
        """
        Calibrate scale factors for Sharpe, Sortino, BL-Sharpe, and (MIR,MSP) penalties
        by equalising average gradient norms.

        Inputs
        ------
        gamma : tuple
            User weights (γ_s, γ_so, γ_bl, γ_pen, _). Only the first four are used here.
        use_l1_penalty : bool
            If True, calibrate penalty scale using L1 geometry; otherwise L2.
        eps : float
            Numerical floor when computing ratios.

        Method
        ------
        Let A_s, A_so, A_bl be the average ℓ2 norms of the gradients of Sharpe, Sortino,
        and BL-Sharpe across random feasible portfolios (from `_W_rand_vals`).
       
        Let A_pen be the average norm of the penalty gradient in the chosen geometry.

        Choose a common target magnitude
         
            target = mean( γ_s A_s, γ_so A_so, γ_bl A_bl, γ_pen A_pen ).

        Define scale factors to make γ_i * scale_i * A_i ≈ target:
        
            scale_i = clip( target / (γ_i * max(A_i, eps)), 1/cap, cap ),
        
        with a large `cap` to avoid extreme multipliers (implemented in `_safe_scale`).

        Return scales and the penalty denominators used by L2 geometry:
            (scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, Ab)

        where:
        - den_mir, den_msp are the L1 or L2 denominators chosen for the MIR/MSP penalty terms.
        - Ab is the BL Cholesky factor retained for downstream use.

        Returns
        -------
        tuple
            (scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, Ab)
        """
        
        vals = self._W_rand_vals
        
        A_s = vals["A_s"]
            
        A_so = vals["A_so"]
            
        A_bl = vals["A_bl"]
     
        if use_l1_penalty:
            
            den_mir = vals["den_mir_l1"]
            
            den_msp = vals["den_msp_l1"]
            
            A_pen = vals["A_pen_mir_msp_l1"]
       
        else:     

            den_mir = vals["den_mir_l2"]
            
            den_msp = vals["den_msp_l2"]
            
            A_pen = vals["A_pen_mir_msp_l2"]

        γ_s, γ_so, γ_bl, γ_pen, _ = gamma 
       
        target = np.mean([
            γ_s * A_s,
            γ_so * A_so, 
            γ_bl * A_bl, 
            γ_pen * A_pen
        ])
        
        scale_s = self._safe_scale(
            target = target,
            weight = γ_s,
            avg_norm = A_s,
            eps = eps
        )
        
        scale_so = self._safe_scale(
            target = target,
            weight = γ_so,
            avg_norm = A_so,
            eps = eps
        )
        
        scale_bl = self._safe_scale(
            target = target,
            weight = γ_bl,
            avg_norm = A_bl,
            eps = eps
        )
        
        scale_pen = self._safe_scale(
            target = target,
            weight = γ_pen,
            avg_norm = A_pen,
            eps = eps
        )

        return (scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, self.Ab)


    def _calibrate_scales_by_grad_ir_sc(
        self,
        gamma: tuple,
        eps: float = 1e-12
    ):
        """
        Calibrate scale factors for Sharpe, Sortino, BL-Sharpe, Information Ratio (1y),
        and Score/CVaR by equalising average gradient norms.

        Inputs
        ------
        gamma : tuple
            (γ_s, γ_so, γ_bl, γ_ir, γ_sc).
        eps : float
            Numerical floor.

        Method
        ------
        From `_W_rand_vals`, obtain A_s, A_so, A_bl, A_ir_1y, A_sc (average gradient norms).
        Set
            target = mean( γ_s A_s, γ_so A_so, γ_bl A_bl, γ_ir A_ir_1y, γ_sc A_sc ).
        Then compute scale_i = _safe_scale(target, γ_i, A_i, eps) for each term.

        Returns
        -------
        tuple
            (scale_s, scale_so, scale_bl, scale_ir, scale_sc, Ab)
        """

        vals = self._W_rand_vals
      
        A_s = vals["A_s"]
      
        A_so = vals["A_so"]
      
        A_bl = A_bl = vals["A_bl"]
      
        A_ir = vals["A_ir_1y"]
      
        A_sc = vals["A_sc"]

        γ_s, γ_so, γ_bl, γ_ir, γ_sc = gamma
       
        target = np.mean([
            γ_s * A_s, 
            γ_so * A_so,
            γ_bl * A_bl,
            γ_ir * A_ir, 
            γ_sc * A_sc
        ])
        
        scale_s = self._safe_scale(
            target = target,
            weight = γ_s,
            avg_norm = A_s,
            eps = eps
        )
        
        scale_so = self._safe_scale(
            target = target,
            weight = γ_so,
            avg_norm = A_so,
            eps = eps
        )
        
        scale_bl = self._safe_scale(
            target = target,
            weight = γ_bl,
            avg_norm = A_bl,
            eps = eps
        )
        
        scale_ir = self._safe_scale(
            target = target,
            weight = γ_ir,
            avg_norm = A_ir,
            eps = eps
        )
        
        scale_sc = self._safe_scale(
            target = target,
            weight = γ_sc,
            avg_norm = A_sc,
            eps = eps
        )

        return (scale_s, scale_so, scale_bl, scale_ir, scale_sc, self.Ab)


    def _calibrate_scales_bl_pen(
        self,
        gamma: tuple,
        use_l1_penalty: bool = False,
        eps: float = 1e-12,
    ):
        """
        Calibrate scale factors for Sharpe, Sortino (1y), IR(5y), Score/CVaR(1y),
        and a BL-centred penalty term (either L1 or L2).

        Inputs
        ------
        gamma : tuple
            (γ_sh, γ_so, γ_ir, γ_sc, γ_pen).
        use_l1_penalty : bool
            Select the geometry of the BL penalty used in calibration.
        eps : float
            Numerical floor.

        Method
        ------
        From `_W_rand_vals`, obtain average norms A_sh, A_so, A_ir_5y, A_sc, and A_pen_bl_(L1/L2).
        Compute
           
            target = mean( γ_sh A_sh, γ_so A_so, γ_ir A_ir_5y, γ_sc A_sc, γ_pen A_pen ).
       
        Then 
        
            scale_i = _safe_scale(target, γ_i, A_i, eps).

        Returns
        -------
        tuple
            (scale_sh, scale_so, scale_ir, scale_sc, scale_pen)
        """
        
        vals = self._W_rand_vals
       
        A_sh = vals["A_s"]
       
        A_so = vals["A_so"]
       
        A_ir = vals["A_ir_5y"]
       
        A_sc = vals["A_sc"]
        
        if use_l1_penalty:
       
            A_pen = vals["A_pen_bl_l1"]
       
        else:
       
            A_pen = vals["A_pen_bl_l2"]   

        γ_sh, γ_so, γ_ir, γ_sc, γ_pen = gamma
       
        target = np.mean([
            γ_sh * A_sh,
            γ_so * A_so,
            γ_ir * A_ir,
            γ_sc * A_sc,
            γ_pen * A_pen
        ])

        scale_sh = self._safe_scale(
            target = target, 
            weight = γ_sh, 
            avg_norm = A_sh, 
            eps = eps
        )
       
        scale_so = self._safe_scale(
            target = target, 
            weight = γ_so, 
            avg_norm = A_so, 
            eps = eps
        )
       
        scale_ir = self._safe_scale(
            target = target, 
            weight = γ_ir, 
            avg_norm = A_ir, 
            eps = eps
        )
       
        scale_sc = self._safe_scale(
            target = target, 
            weight = γ_sc, 
            avg_norm = A_sc, 
            eps = eps
        )
       
        scale_pen = self._safe_scale(
            target = target, 
            weight = γ_pen, 
            avg_norm = A_pen, 
            eps = eps
        )

        return (scale_sh, scale_so, scale_ir, scale_sc, scale_pen)
    
    
    def _calibrate_scales_by_grad_ir_sc_asr(
        self,
        gamma: tuple,
        eps: float = 1e-12,
    ):
        """
        Calibrate scale factors for Sharpe, Sortino, BL-Sharpe, IR(1y), Score/CVaR, and ASR.

        Inputs
        ------
        gamma : tuple
            (γ_s, γ_so, γ_bl, γ_ir, γ_sc, γ_asr).
        eps : float
            Numerical floor.

        Method
        ------
        Use averages A_s, A_so, A_bl, A_ir_1y, A_sc, A_asr from `_W_rand_vals`.
        Set
       
            target = mean(γ_s A_s, γ_so A_so, γ_bl A_bl, γ_ir A_ir_1y, γ_sc A_sc, γ_asr A_asr),
       
        and 
        
            scale_i = _safe_scale(target, γ_i, A_i, eps).

        Returns
        -------
        tuple
            (scale_s, scale_so, scale_bl, scale_ir, scale_sc, scale_asr)
        """
       
        vals = self._W_rand_vals

        A_s = vals["A_s"]
       
        A_so = vals["A_so"]
       
        A_bl = vals["A_bl"]
       
        A_ir = vals["A_ir_1y"]
       
        A_sc = vals["A_sc"]
       
        A_asr = vals["A_asr"]

        γ_s, γ_so, γ_bl, γ_ir, γ_sc, γ_asr = gamma

        target = np.mean([
            γ_s * A_s, 
            γ_so * A_so, 
            γ_bl * A_bl,
            γ_ir * A_ir, 
            γ_sc * A_sc, 
            γ_asr * A_asr
        ])

        scale_s = self._safe_scale(
            target = target, 
            weight = γ_s,   
            avg_norm = A_s,  
            eps = eps
        )
        
        scale_so = self._safe_scale(
            target = target, 
            weight = γ_so,  
            avg_norm = A_so,
            eps = eps
        )
        
        scale_bl = self._safe_scale(
            target = target, 
            weight = γ_bl,  
            avg_norm = A_bl, 
            eps = eps
        )
        
        scale_ir = self._safe_scale(
            target = target, 
            weight = γ_ir,  
            avg_norm = A_ir, 
            eps = eps
        )
        
        scale_sc = self._safe_scale(
            target = target, 
            weight = γ_sc,  
            avg_norm = A_sc, 
            eps = eps
        )
        
        scale_asr = self._safe_scale(
            target = target, 
            weight = γ_asr, 
            avg_norm = A_asr,
            eps = eps
        )
        
        return (scale_s, scale_so, scale_bl, scale_ir, scale_sc, scale_asr)


    def _calibrate_scales_ir5_sc_pen_asr(
        self,
        gamma: tuple,
        use_l1_penalty: bool = False,
        eps: float = 1e-12,
    ):
        """
        Calibrate scale factors for Sharpe, Sortino, IR(5y), Score/CVaR, ASR, and a BL-penalty.

        Inputs
        ------
        gamma : tuple
            (γ_sh, γ_so, γ_ir5, γ_sc, γ_asr, γ_pen).
        use_l1_penalty : bool
            If True, use L1 geometry for the BL penalty; else L2.
        eps : float
            Numerical floor.

        Method
        ------
        From `_W_rand_vals`, obtain A_sh, A_so, A_ir_5y, A_sc, A_asr, and A_pen_bl_(L1/L2).
       
        Set
       
            target = mean(γ_sh A_sh, γ_so A_so, γ_ir5 A_ir_5y, γ_sc A_sc, γ_asr A_asr, γ_pen A_pen),
       
        and 
        
            scale_i = _safe_scale(target, γ_i, A_i, eps).

        Returns
        -------
        tuple
            (scale_sh, scale_so, scale_ir5, scale_sc, scale_asr, scale_pen)
        """

        vals = self._W_rand_vals

        A_sh = vals["A_s"]
       
        A_so = vals["A_so"]
       
        A_ir5 = vals["A_ir_5y"]
       
        A_sc = vals["A_sc"]
       
        A_asr = vals.get("A_asr", A_sh)

        if use_l1_penalty:
       
            A_pen = vals["A_pen_bl_l1"]
       
        else:
       
            A_pen = vals["A_pen_bl_l2"]

        γ_sh, γ_so, γ_ir5, γ_sc, γ_asr, γ_pen = gamma

        target = np.mean([
            γ_sh * A_sh,
            γ_so * A_so,
            γ_ir5 * A_ir5,
            γ_sc * A_sc,
            γ_asr * A_asr,
            γ_pen * A_pen,
        ])

        scale_sh = self._safe_scale(
            target = target, 
            weight = γ_sh, 
            avg_norm = A_sh, 
            eps = eps
        )
        
        scale_so = self._safe_scale(
            target = target, 
            weight = γ_so, 
            avg_norm = A_so, 
            eps = eps
        )
        
        scale_ir5 = self._safe_scale(
            target = target, 
            weight = γ_ir5, 
            avg_norm = A_ir5, 
            eps = eps
        )
        
        scale_sc = self._safe_scale(
            target = target, 
            weight = γ_sc, 
            avg_norm = A_sc, 
            eps = eps
        )
        
        scale_asr = self._safe_scale(
            target = target, 
            weight = γ_asr, 
            avg_norm = A_asr, 
            eps = eps
        )
        
        scale_pen = self._safe_scale(
            target = target, 
            weight = γ_pen, 
            avg_norm = A_pen, 
            eps = eps
        )

        return (scale_sh, scale_so, scale_ir5, scale_sc, scale_asr, scale_pen)


    def _coerce_weights(
        self, 
        w: Optional[pd.Series], 
        name: str
    ) -> pd.Series:
        """
        Convert various weight inputs to a feasible pd.Series aligned to the universe.

        Inputs
        ------
        w : pd.Series | array-like
            If a Series, it is reindexed to the universe; if an array, it must be length n.
        name : str
            Name used in error messages and as a fallback for the output Series name.

        Processing
        ----------
        - Reindex to the current universe (Series case).
        
        - Validate shape (array case).
        
        - Non-negativity: clip below at 0.
        
        - Renormalise to sum 1 (raises if total mass ≤ 0).
        
        - Preserve input name where possible.

        Returns
        -------
        pd.Series
            Weight vector aligned to `self.universe`, non-negative, summing to one.

        Raises
        ------
        ValueError
            If the input is None, has incompatible shape, or sums to a non-positive value.
        """
        
        if w is None:
        
            raise ValueError(f"{name} is None")
        
        if isinstance(w, pd.Series):
        
            out = w.reindex(self.universe).astype(float).fillna(0.0)
        
        else:
        
            arr = np.asarray(w, float)
        
            if arr.shape != (self.n,):
        
                raise ValueError(f"{name} has wrong shape {arr.shape}; expected {(self.n,)}")
        
            out = pd.Series(arr, index = self.universe)
        
        s = float(out.sum())
        
        if s <= 0:
        
            raise ValueError(f"{name} sums to {s}; cannot renormalize")
        
        out = out.clip(lower = 0.0)
        
        out /= out.sum()
        
        out.name = getattr(w, "name", name)
        
        return out
    
    
    def _np1(
        self, 
        x
    ) -> np.ndarray:
        """
        Convert input to a 1-D NumPy float array of length n.

        Inputs
        ------
        x : array-like

        Behaviour
        ---------
        - If x is already a 1-D array of length n, return it.
       
        - If x has higher dimension but squeezes to a 1-D length-n array, return the squeeze.
       
        - Otherwise, raise a ValueError.

        Returns
        -------
        np.ndarray, shape (n,)
            1-D float array.

        Raises
        ------
        ValueError
            If the squeezed shape is not exactly (n,).
        """
    
        a = np.asarray(x, dtype = float)
        
        if a.ndim == 1: 
            
            if a.size != self.n:
                
                raise ValueError(f"Expected length {self.n}, got {a.size}")
    
            return a 
        
        else:
            
            return a.squeeze()


    def comb_port(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        max_iter: int = 1000,
        tol: float = 1e-12,
    ) -> pd.Series:
        """
        Convex–concave procedure (CCP) optimiser for a composite **reward** built from:
       
            Sharpe(annual), Sortino(1y), Black–Litterman Sharpe, Information Ratio (1y),
            and a Score/CVaR(1y) ratio (gradients via smoothed RU CVaR; final scoring via
            exact RU LP).

        Objective (maximised)
        ---------------------
        Let:
        
        - μ, Σ, r_f^ann be the annual forecast mean, covariance, and risk-free rate.
        
        - R₁ ∈ R^{T×n} be 1y weekly returns; r_f^wk the weekly risk-free rate; T the number of weeks.
        
        - b¹ ∈ R^T be the 1y benchmark (rf or provided series).
        
        - s ∈ R^n be asset scores; α ∈ (0,1) the CVaR level.
        
        - (μ_bl, Σ_bl) be BL mean and covariance (calibrated earlier).
        
        - γ = (γ_S, γ_SO, γ_BL, γ_IR, γ_SC) are user weights.
        
        - κ = (κ_S, κ_SO, κ_BL, κ_IR, κ_SC) are calibration scales returned by
        `_calibrate_scales_by_grad_ir_sc`.

        Define, for weights w ≥ 0, 1ᵀw = 1:

        1) Sharpe:
        
            S(w) = (μᵀw − r_f^ann) / σ(w),  where σ(w) = sqrt(wᵀ Σ w).

        2) Sortino (annualised):
        
            Downside deviation per week:
        
                D_week(w) = ||max(r_f^wk − R₁ w, 0)||₂ / sqrt(T).
       
                D_ann(w) = max(D_week(w) * sqrt(52), ε).
        
                SO(w) = (μᵀ w − r_f^ann) / D_ann(w).

        3) BL Sharpe:
        
            S_bl(w) = (μ_blᵀw − r_f^ann) / σ_bl(w),  σ_bl(w) = sqrt(wᵀ Σ_bl w).

        4) Information Ratio (1y):
        
            a¹ = R₁ w − b¹,
        
            TE₁(w) = ||a¹||₂ / sqrt(T),
        
            IR₁(w) = mean(a¹) / max(TE₁(w), ε).

        5) Score/CVaR (1y): use smoothed RU CVaR for gradients and exact RU LP for final scoring.
        Smoothed RU (for gradients) with softplus temperature β_cal:
            
                C_α^β(w) = z* + (1/(α T)) ∑ softplus(−(R₁ w + z*); β_cal),
        
        where z* solves ∑ σ(β_cal (−(R₁w + z*))) = α T.
        
        Score ratio (gradients): SC_grad(w) = (sᵀ w) / max(C_α^β(w), ε).
        
        Final scoring replaces C_α^β(w) by the **exact** RU LP CVaR.

        Composite reward to maximise:
        
            F(w) = γ_S κ_S S(w)
                    + γ_SO κ_SO SO(w)
                    + γ_BL κ_BL S_bl(w)
                    + γ_IR κ_IR IR₁(w)
                    + γ_SC κ_SC SC(w),

        subject to feasibility constraints (sum-to-one, boxes, industry/sector caps).

        Gradients used in the CCP linearisation
        ---------------------------------------
        1) ∇ S(w) = [ μ σ(w) − (μᵀw − r_f) Σ w / σ(w) ] / σ(w)².
        
        2) ∇ SO(w) via quotient rule with
            
            ∇ D_week = − R₁ᵀ max(r_f^wk − R₁w, 0) / (T * D_week),
        then annualised by sqrt(52) and floored by ε.
        
        3) ∇ S_bl(w) as in (1) with (μ_bl, Σ_bl).
        
        4) ∇ IR₁(w) = [ mean(R₁) * TE₁ − mean(a¹) * ∇ TE₁ ] / TE₁²,
        with ∇ TE₁ = R₁ᵀ a¹ / (T * TE₁).
       
        5) For smoothed RU:
            ∂ C_α^β / ∂ r_i = − σ(β (−(r_i + z*))) / (α T),
        so 
        
            ∇ C_α^β(w) = R₁ᵀ (∂ C_α^β / ∂ r),
        
            ∇ SC = [ s * C − (sᵀw) ∇ C ] / C².

        CCP linearisation and subproblem
        --------------------------------
        At iterate w_t, form the first-order surrogate of −F(w) about w_t:
        
            −F(w) ≈ const − ⟨∇F(w_t), w⟩.
        
        Solve the convex subproblem
        
            minimise    − ⟨lin_param, w⟩
        
        subject to feasibility constraints, where 
        
            lin_param = ∇F(w_t). 
        
        This yields w_{t+1}.

        Stopping and acceptance
        -----------------------
        Iterate until both ||w_{t+1} − w_t||₂ and |subproblem objective change| fall below tolerance.
        Final scoring recomputes SC via the exact RU LP CVaR.

        Modelling advantage
        -------------------
        Balances traditional mean–variance efficiency (Sharpe) with downside-aware risk (Sortino),
        incorporates Black–Litterman views for stability, controls tracking quality via 1y IR,
        and promotes robustness to tail risk via a Score/CVaR ratio; gradients remain smooth and
        well-behaved due to the CVaR softplus surrogate used during optimisation.
        """

        self._assert_alignment()
        
        if self.gamma is not None:
        
            gamma = self.gamma

        tol_floor = (np.trace(self.Σ) / max(1, self.n)) * 1e-8
        
        tol = max(tol, tol_floor)

        w_mir_a = self._mir
        
        w_msp_a = self._msp
        
        _, μ_bl_arr, Σb_cal = self._bl
        
        initials = self._initial_seeds

        scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, _ = self._calibrate_scales_by_grad_rewards_penalties(
            gamma = gamma,
            use_l1_penalty = False,
        )

        ctx = g.GradCtx(
            mu = self.er_arr,          
            Sigma = self.Σ,            
            rf = self.rf_ann,       
            eps = 1e-12,
            R1 = self.R1,             
            target = self.rf_week,    
        )

        sharpe = g.sharpe_val_grad

        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week,
            eps = ctx.eps
        )

        ctx_bl = g.GradCtx(
            mu = μ_bl_arr, 
            Sigma = Σb_cal, 
            rf = self.rf_ann,
            eps = ctx.eps
        )
        
        
        def bl_sharpe_val_grad(
            w: np.ndarray, 
            _ctx: g.GradCtx,
            work: g.Work | None = None
        ):
        
            return g.sharpe_val_grad(
                w = w, 
                ctx = ctx_bl, 
                work = work
            )


        obj = g.compose([
            (gamma[0] * scale_s, sharpe),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_bl, bl_sharpe_val_grad),
        ])

        w_var = cp.Variable(self.n, nonneg = True)
        
        lin_param = cp.Parameter(self.n)

        pen_comb = (cp.sum_squares(w_var - w_mir_a) / den_mir) + (cp.sum_squares(w_var - w_msp_a) / den_msp)
     
        obj_expr = gamma[3] * scale_pen * pen_comb - lin_param @ w_var
     
        cons = self.build_constraints(
            w_var = w_var, 
            single_port_cap = True
        )
     
        prob = cp.Problem(cp.Minimize(obj_expr), cons)


        def _from_initial(
            w0: np.ndarray
        ) -> tuple[np.ndarray, float, dict[str, float]]:
        
            w = w0.copy()
        
            prev_obj = None

            for _ in range(max_iter):
        
                work = g.Work()

                _, grad = obj(w, ctx, work)
        
                lin_param.value = self._np1(grad)

                if not self._solve(prob) or w_var.value is None:
                  
                    raise RuntimeError("comb_port CCP: solver failed")

                w_new = w_var.value
               
                obj_val = float(obj_expr.value)

                if prev_obj is not None and np.linalg.norm(w_new - w) < tol and abs(obj_val - prev_obj) < tol:
                  
                    w = w_new
                  
                    break

                prev_obj = obj_val
               
                w = w_new

            work = g.Work()
           
            sh_val, sh_grad = sharpe(
                w = w, 
                ctx = ctx, 
                work = work
            )
           
            so_val, so_grad = sortino(
                w = w, 
                ctx = ctx,
                work = work
            )
           
            bl_val, bl_grad = bl_sharpe_val_grad(
                w = w, 
                _ctx = ctx, 
                work = work
            )

            pen_val = float(np.sum((w - w_mir_a) ** 2)) / den_mir + float(np.sum((w - w_msp_a) ** 2)) / den_msp
           
            score = (
                gamma[0] * scale_s * sh_val +
                gamma[1] * scale_so * so_val +
                gamma[2] * scale_bl * bl_val -
                gamma[3] * scale_pen * pen_val
            )

            diag = {
                "Sharpe_push": gamma[0] * scale_s * float(np.linalg.norm(sh_grad)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(so_grad)),
                "BL_push": gamma[2] * scale_bl * float(np.linalg.norm(bl_grad)),
                "Penalty_push": gamma[3] * scale_pen * float(
                    np.linalg.norm(2.0 * (w - w_mir_a) / den_mir + 2.0 * (w - w_msp_a) / den_msp)
                ),
            }
            
            return w, score, diag

        best_w = None
        
        best_s = -np.inf
        
        best_diag = {}
        
        for w0 in initials:
        
            try:
        
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
        
            except RuntimeError:
        
                continue
        
            if s_cand > best_s:
        
                best_s = s_cand
                
                best_w = w_cand
                
                best_diag = diag

        if best_w is None:
        
            raise RuntimeError("comb_port: all initialisations failed")

        self._last_diag = best_diag

        return pd.Series(best_w, index = self.universe, name = "comb_port")


    def comb_port1(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        max_iter: int = 1000,
        tol: float = 1e-12,
    ) -> pd.Series:
        """
        Convex–concave procedure (CCP) optimiser for a composite **reward** built from:
        Sharpe(annual), Sortino(1y), Black–Litterman Sharpe, Information Ratio (1y),
        and a Score/CVaR(1y) ratio (gradients via smoothed RU CVaR; final scoring via exact RU LP).

        Objective (maximised)
        ---------------------
        Let:
       
        - μ, Σ, r_f^ann be the annual forecast mean, covariance, and risk-free rate.
       
        - R₁ ∈ R^{T×n} be 1y weekly returns; r_f^wk the weekly risk-free rate; T the number of weeks.
       
        - b¹ ∈ R^T be the 1y benchmark (rf or provided series).
       
        - s ∈ R^n be asset scores; α ∈ (0,1) the CVaR level.
       
        - (μ_bl, Σ_bl) be BL mean and covariance (calibrated earlier).
       
        - γ = (γ_S, γ_SO, γ_BL, γ_IR, γ_SC) are user weights.
       
        - κ = (κ_S, κ_SO, κ_BL, κ_IR, κ_SC) are calibration scales returned by
        `_calibrate_scales_by_grad_ir_sc`.

        Define, for weights w ≥ 0, 1ᵀw = 1:

        1) Sharpe:
       
            S(w) = (μᵀw − r_f^ann) / σ(w),  where σ(w) = sqrt(wᵀ Σ w).

        2) Sortino (annualised):
        
        Downside deviation per week:
        
            
            D_week(w) = ||max(r_f^wk − R₁ w, 0)||₂ / sqrt(T).
        
            D_ann(w) = max(D_week(w) * sqrt(52), ε).
        
            SO(w) = (μᵀ w − r_f^ann) / D_ann(w).

        3) BL Sharpe:
        
            S_bl(w) = (μ_blᵀw − r_f^ann) / σ_bl(w),  σ_bl(w) = sqrt(wᵀ Σ_bl w).

        4) Information Ratio (1y):
        
            a¹ = R₁ w − b¹,
        
            TE₁(w) = ||a¹||₂ / sqrt(T),
        
            IR₁(w) = mean(a¹) / max(TE₁(w), ε).

        5) Score/CVaR (1y): use smoothed RU CVaR for gradients and exact RU LP for final scoring.
        Smoothed RU (for gradients) with softplus temperature β_cal:
            
            C_α^β(w) = z* + (1/(α T)) ∑ softplus(−(R₁ w + z*); β_cal),
            where z* solves ∑ σ(β_cal (−(R₁w + z*))) = α T.
        
        Score ratio (gradients): SC_grad(w) = (sᵀ w) / max(C_α^β(w), ε).
        Final scoring replaces C_α^β(w) by the **exact** RU LP CVaR.

        Composite reward to maximise:
        
            F(w) = γ_S κ_S S(w)
                    + γ_SO κ_SO SO(w)
                    + γ_BL κ_BL S_bl(w)
                    + γ_IR κ_IR IR₁(w)
                    + γ_SC κ_SC SC(w),

        subject to feasibility constraints (sum-to-one, boxes, industry/sector caps).

        Gradients used in the CCP linearisation
        ---------------------------------------
        1) ∇ S(w) = [ μ σ(w) − (μᵀw − r_f) Σ w / σ(w) ] / σ(w)².
       
        2) ∇ SO(w) via quotient rule with
       
            ∇ D_week = − R₁ᵀ max(r_f^wk − R₁w, 0) / (T * D_week),
        then annualised by sqrt(52) and floored by ε.
        
        3) ∇ S_bl(w) as in (1) with (μ_bl, Σ_bl).
        
        4) ∇ IR₁(w) = [ mean(R₁) * TE₁ − mean(a¹) * ∇ TE₁ ] / TE₁²,
        with ∇ TE₁ = R₁ᵀ a¹ / (T * TE₁).
        
        5) For smoothed RU:
        
            ∂ C_α^β / ∂ r_i = − σ(β (−(r_i + z*))) / (α T),
        so 
        
            ∇ C_α^β(w) = R₁ᵀ (∂ C_α^β / ∂ r),
            
            ∇ SC = [ s * C − (sᵀw) ∇ C ] / C².

        CCP linearisation and subproblem
        --------------------------------
        At iterate w_t, form the first-order surrogate of −F(w) about w_t:
        
            −F(w) ≈ const − ⟨∇F(w_t), w⟩.
        
        Solve the convex subproblem
        
            minimise    − ⟨lin_param, w⟩
        
        subject to  feasibility constraints, where 
        
            lin_param = ∇F(w_t). 
        
        This yields w_{t+1}.

        Stopping and acceptance
        -----------------------
        Iterate until both ||w_{t+1} − w_t||₂ and |subproblem objective change| fall below tolerance.
        Final scoring recomputes SC via the exact RU LP CVaR.

        Modelling advantage
        -------------------
        Balances traditional mean–variance efficiency (Sharpe) with downside-aware risk (Sortino),
        incorporates Black–Litterman views for stability, controls tracking quality via 1y IR,
        and promotes robustness to tail risk via a Score/CVaR ratio; gradients remain smooth and
        well-behaved due to the CVaR softplus surrogate used during optimisation.
        """
        
        self._assert_alignment()
        
        cvar_level = self._cvar_level
        
        if self.gamma is not None:
        
            gamma = self.gamma

        tol_floor = (np.trace(self.Σ) / max(1, self.n)) * 1e-8
        
        tol = max(tol, tol_floor)

        initials = self._initial_seeds
        
        _, μ_bl_arr, Σb_cal = self._bl

        _r1_b = getattr(self, "_R1_b_te", (None, None))[1]
        
        b1_vec = _r1_b if _r1_b is not None else np.full(self.T1, self.rf_week)

        scale_s, scale_so, scale_bl, scale_ir, scale_sc, _ = self._calibrate_scales_by_grad_ir_sc(
            gamma = gamma
        )

        ctx_grad = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ, 
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1,
            target = self.rf_week,
        )
        
        ctx_grad.scores = self.scores_arr
        
        ctx_grad.cvar_alpha = cvar_level
        
        ctx_grad.cvar_beta = self._cvar_beta_calib
        
        ctx_grad.denom_floor = self._denom_floor

        ctx_eval = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ, 
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week,
        )
        
        ctx_eval.scores = self.scores_arr
        
        ctx_eval.cvar_alpha = cvar_level
        
        ctx_eval.cvar_beta = self._cvar_beta_solve
        
        ctx_eval.denom_floor = self._denom_floor

        ctx_bl = g.GradCtx(
            mu = μ_bl_arr,
            Sigma = Σb_cal, 
            rf = self.rf_ann, 
            eps = 1e-12
        )
        
        
        def bl_sharpe_val_grad(
            w, 
            _ctx, 
            work = None
        ):
        
            return g.sharpe_val_grad(
                w = w,
                ctx = ctx_bl, 
                work = work
            )

       
        ir1 = g.ir_val_grad_from(
            R = self.R1, 
            b = b1_vec, 
            eps = ctx_grad.eps
        )
       
        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week, 
            eps = ctx_grad.eps
        )
       
        obj = g.compose([
            (gamma[0] * scale_s, g.sharpe_val_grad),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_bl, bl_sharpe_val_grad),
            (gamma[3] * scale_ir, ir1),
            (gamma[4] * scale_sc, g.score_over_cvar_val_grad),  
        ])

        w_var = cp.Variable(self.n, nonneg = True)
       
        lin_param = cp.Parameter(self.n)
       
        obj_expr = -lin_param @ w_var
       
        cons = self.build_constraints(
            w_var = w_var, 
            single_port_cap = True
        )
       
        prob = cp.Problem(cp.Minimize(obj_expr), cons)


        def _from_initial(
            w0: np.ndarray
        ):
        
            w = w0.copy()
        
            prev_obj = None

            for _ in range(max_iter):
        
                work = g.Work()
        
                _, grad = obj(w, ctx_grad, work)
        
                lin_param.value = self._np1(grad)

                if not self._solve(prob) or w_var.value is None:
        
                    raise RuntimeError("comb_port1 CCP: solver failed")
        
                w_new = w_var.value
        
                obj_val = float(obj_expr.value)

                if prev_obj is not None and np.linalg.norm(w_new - w) < tol and abs(obj_val - prev_obj) < tol:
        
                    w = w_new
        
                    break
        
                prev_obj = obj_val
        
                w = w_new

            work = g.Work()
        
            sh_val, sh_g = g.sharpe_val_grad(
                w = w, 
                ctx = ctx_eval,
                work = work
            )
        
            so_val, so_g = sortino(
                w = w, 
                ctx = ctx_eval,
                work = work
            )
        
            bl_val, bl_g = bl_sharpe_val_grad(
                w = w, 
                _ctx = ctx_eval, 
                work = work
            )
        
            ir_val, ir_g = ir1(
                w = w, 
                ctx = ctx_eval,
                work = work
            )

            r = ctx_eval.R1 @ w
           
            T = r.shape[0]
           
            z = cp.Variable()
           
            u = cp.Variable(T, nonneg = True)
           
            alpha = float(cvar_level)
          
            cvar_obj = z + (1.0 / (alpha * T)) * cp.sum(u)
          
            cvar_cons = [u >= -(r + z)]
          
            cvar_prob = cp.Problem(cp.Minimize(cvar_obj), cvar_cons)
          
            if not self._solve(cvar_prob) or z.value is None or u.value is None:
          
                raise RuntimeError("comb_port1: CVaR LP failed in final scoring")
          
            cvar_exact = float(cvar_obj.value)
          
            sc_ratio = float(self.scores_arr @ w) / max(cvar_exact, self._denom_floor)

            score = (
                gamma[0] * scale_s  * sh_val +
                gamma[1] * scale_so * so_val +
                gamma[2] * scale_bl * bl_val +
                gamma[3] * scale_ir * ir_val +
                gamma[4] * scale_sc * sc_ratio
            )

            _, sc_g = g.score_over_cvar_val_grad(w, ctx_eval, work)
          
            diag = {
                "Sharpe_push": gamma[0] * scale_s * float(np.linalg.norm(sh_g)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(so_g)),
                "BL_push": gamma[2] * scale_bl * float(np.linalg.norm(bl_g)),
                "IR_push": gamma[3] * scale_ir * float(np.linalg.norm(ir_g)),
                "SC_push": gamma[4] * scale_sc * float(np.linalg.norm(sc_g)),
            }
            
            return w, score, diag

        best_w = None 
        
        best_s = -np.inf
        
        best_diag = {}
        
        for w0 in initials:
          
            try:
          
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
          
            except RuntimeError:
          
                continue
          
            if s_cand > best_s:
          
                best_s = s_cand
                
                best_w = w_cand
                
                best_diag = diag

        if best_w is None:
    
            raise RuntimeError("comb_port1: all initialisations failed")

        self._last_diag = best_diag
    
        return pd.Series(best_w, index = self.universe, name = "comb_port1")


    def comb_port2(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        max_iter: int = 1000,
        tol: float = 1e-12,
        huber_delta_l1: float = 1e-4,
    ) -> pd.Series:
        """
        CCP optimiser for a composite reward with **L1 (Huber-smoothed) proximity penalties** to
        two anchor portfolios (MIR and MSP): maximise Sharpe + Sortino + BL-Sharpe − L1 penalties.

        Objective (maximised)
        ---------------------
        Using the same definitions for S(w), SO(w), S_bl(w) as in comb_port1, and anchors
        w_MIR, w_MSP, with denominators (den_MIR, den_MSP) from calibration, define
        the Huber-smoothed L1 penalty for a scalar x as:
       
            huber(x; δ) = { 0.5 x²/δ      if |x| ≤ δ
                          { |x| − 0.5 δ   otherwise.

        Penalty term:
       
            P₁(w) = (∑_i huber(w_i − w_MIR,i; δ)) / den_MIR + (∑_i huber(w_i − w_MSP,i; δ)) / den_MSP.

        Composite objective:
        
            F(w) = γ_S κ_S S(w) + γ_SO κ_SO SO(w) + γ_BL κ_BL S_bl(w) − γ_P κ_P P₁(w).

        Gradients used
        --------------
        - ∇ S(w), ∇ SO(w), ∇ S_bl(w) as in comb_port1.
        - ∇ P₁(w): componentwise derivative of the Huber penalty,
        
            g_i = sign(w_i − a_i) 
            
        when |w_i − a_i| > δ, else (w_i − a_i)/δ; scaled by the corresponding denominators and γ_P κ_P.

        CCP linearisation and subproblem
        --------------------------------
        As in comb_port1, linearise the reward part and keep the convex Huber penalties explicit:
        
            minimise   γ_P κ_P P₁(w) − ⟨lin_param, w⟩
        
        subject to feasibility, with 
        
            lin_param = ∇ (γ_S κ_S S + γ_SO κ_SO SO + γ_BL κ_BL S_bl) 
            
        evaluated at w_t.

        Stopping and acceptance
        -----------------------
        Stop when both ||w_{t+1} − w_t||₂ and |subproblem objective change| fall below tolerance.

        Modelling advantage
        -------------------
        Encourages solutions close (in L1 geometry) to two practical anchors (MIR, MSP),
        which improves interpretability and turnover control, while keeping a differentiable
        objective via Huber smoothing; the reward balances total and downside efficiency and
        stability through BL-Sharpe.
        """
       
        self._assert_alignment()
       
        if self.gamma is not None:
       
            gamma = self.gamma

        tol_floor = (np.trace(self.Σ) / max(1, self.n)) * 1e-8
       
        tol = max(tol, tol_floor)

        w_mir_a = self._mir
       
        w_msp_a = self._msp
       
        _, μ_bl_arr, Σb_cal = self._bl
       
        initials = self._initial_seeds

        scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, _ = self._calibrate_scales_by_grad_rewards_penalties(
                gamma = gamma, 
                use_l1_penalty = True
            )

        ctx = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ, 
            rf = self.rf_ann,
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week
        )
        
        ctx_bl = g.GradCtx(
            mu = μ_bl_arr, 
            Sigma = Σb_cal, 
            rf = self.rf_ann, 
            eps = 1e-12
        )
        
        
        def bl_sharpe_val_grad(
            w, 
            _ctx,
            work = None
        ):
           
            return g.sharpe_val_grad(
                w = w, 
                ctx = ctx_bl, 
                work = work
            )
        
        
        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week, 
            eps = ctx.eps
        )

        obj = g.compose([
            (gamma[0] * scale_s, g.sharpe_val_grad),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_bl, bl_sharpe_val_grad),
        ])

        w_var = cp.Variable(self.n, nonneg = True)
       
        lin_param = cp.Parameter(self.n)
       
        pen_comb = (
            cp.sum(cp.huber(w_var - w_mir_a, huber_delta_l1)) / den_mir
            + cp.sum(cp.huber(w_var - w_msp_a, huber_delta_l1)) / den_msp
        )
       
        obj_expr = gamma[3] * scale_pen * pen_comb - lin_param @ w_var
       
        cons = self.build_constraints(
            w_var = w_var, 
            single_port_cap = True
        )
        
        prob = cp.Problem(cp.Minimize(obj_expr), cons)


        def _from_initial(
            w0: np.ndarray
        ):
        
            w = w0.copy()
        
            prev_obj = None

            for _ in range(max_iter):
        
                work = g.Work()
        
                _, grad = obj(w, ctx, work)
        
                lin_param.value = self._np1(
                    x = grad
                )

                if not self._solve(prob) or w_var.value is None:
                
                    raise RuntimeError("comb_port2 CCP: solver failed")
               
                w_new = w_var.value
               
                obj_val = float(obj_expr.value)

                if prev_obj is not None and np.linalg.norm(w_new - w) < tol and abs(obj_val - prev_obj) < tol:
               
                    w = w_new
               
                    break
               
                prev_obj = obj_val
               
                w = w_new

            work = g.Work()
            
            sh_val, sh_g = g.sharpe_val_grad(
                w = w, 
                ctx = ctx,
                work = work
            )
            
            so_val, so_g = sortino(
                w = w,
                ctx = ctx, 
                work = work
            )
            
            bl_val, bl_g = bl_sharpe_val_grad(
                w = w, 
                _ctx = ctx, 
                work = work
            )

            pen_val = float(np.sum(np.abs(w - w_mir_a))) / den_mir + float(np.sum(np.abs(w - w_msp_a))) / den_msp
           
            score = (
                gamma[0] * scale_s  * sh_val +
                gamma[1] * scale_so * so_val +
                gamma[2] * scale_bl * bl_val -
                gamma[3] * scale_pen * pen_val
            )

            g_pen = np.sign(w - w_mir_a) / den_mir + np.sign(w - w_msp_a) / den_msp
           
            diag = {
                "Sharpe_push": gamma[0] * scale_s * float(np.linalg.norm(sh_g)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(so_g)),
                "BL_push":  gamma[2] * scale_bl * float(np.linalg.norm(bl_g)),
                "Penalty_push": gamma[3] * scale_pen * float(np.linalg.norm(g_pen)),
            }
            
            return w, score, diag

        best_w = None
        
        best_s = -np.inf
        
        best_diag = {}
        
        for w0 in initials:
        
            try:
        
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
        
            except RuntimeError:
        
                continue
        
            if s_cand > best_s:
        
                best_s = s_cand
                
                best_w = w_cand
                
                best_diag = diag

        if best_w is None:
           
            raise RuntimeError("comb_port2: all initialisations failed")

        self._last_diag = best_diag
        
        return pd.Series(best_w, index = self.universe, name = "comb_port2")


    def comb_port3(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        max_iter: int = 1000,
    ) -> pd.Series:
        """
        CCP optimiser with **IR(5y) and Score/CVaR(1y)** in the reward and an **L2 proximity**
        penalty to the BL portfolio; includes an **Armijo backtracking line-search** on the
        true smoothed composite.

        Objective (maximised)
        ---------------------
        Let b⁵ be the 5y benchmark series (rf if absent), R₅ the 5y weekly returns, T₅ its length.
       
        Define:
       
        - IR₅(w) = mean(a⁵) / TE₅, 
        
        where 
        
            a⁵ = R₅ w − b⁵,  
            
            TE₅ = ||a⁵||₂ / sqrt(T₅).
       
        - SC(w) = (sᵀ w) / C_α^β(w) as in comb_port1 (β_cal used for gradients).
       
        - BL proximity (L2): 
        
            P₂(w) = ||w − w_BL||₂².

        Composite objective:
       
            F(w) = γ_SH κ_SH S(w) + γ_SO κ_SO SO(w) + γ_IR κ_IR IR₅(w) + γ_SC κ_SC SC(w) − γ_P κ_P P₂(w).

        Gradients used
        --------------
        - ∇ S, ∇ SO as before.
        
        - ∇ IR₅(w) via quotient rule with ∇ TE₅ = R₅ᵀ a⁵ / (T₅ * TE₅).
        
        - ∇ SC(w) from smoothed RU as in comb_port1.
        
        - ∇ P₂(w) = 2 (w − w_BL).

        CCP linearisation and subproblem
        --------------------------------
        Linearise the reward at w_t and solve
        
            minimise   γ_P κ_P P₂(w) − ⟨lin_param, w⟩
        
        subject to feasibility, with
        
            lin_param = ∇(γ_SH κ_SH S + γ_SO κ_SO SO + γ_IR κ_IR IR₅ + γ_SC κ_SC SC) |_{w_t}.

        Armijo line-search on the true smoothed objective
        -------------------------------------------------
        Let w_cand be the subproblem solution and d = w_cand − w_t.
        Form the **true** smoothed reward G(w) (same terms as F but SC using smoothed RU,
        and the penalty evaluated exactly) and backtrack η ∈ {1, 1/2, 1/4, …} to accept
        
            w_{t+1} = w_t + η d 
            
        if it yields sufficient increase:
        
            G(w_{t+1}) ≥ G(w_t) + c η ||d||₂²
        
        with a small c > 0. If backtracking fails, fall back to w_cand.

        Final scoring
        -------------
        For reporting, recompute Score/CVaR with the **exact** RU LP CVaR.

        Modelling advantage
        -------------------
        The 5y IR component brings persistence and benchmark alignment over a longer horizon;
        the Score/CVaR term injects tail-risk awareness; the L2 proximity to BL stabilises
        estimates and prevents drift away from equilibrium views; Armijo backtracking adds
        monotonic ascent in the smoothed objective, improving robustness.
        """

        self._assert_alignment()

        cvar_level = self._cvar_level

        if self.gamma is not None:

            gamma = self.gamma

        w_bl_a, _, _ = self._bl

        initials = [np.asarray(w, float) for w in self._initial_seeds if w is not None and np.ndim(w) == 1]

        if not initials:

            raise RuntimeError("comb_port3: no valid initial weights")

        scale_sh, scale_so, scale_ir, scale_sc, scale_pen = self._calibrate_scales_bl_pen(
            gamma = gamma
        )

        b5_vec = self._R5_b_te[1]
     
        if b5_vec is None:
     
            b5_vec = np.full(self.R5.shape[0], self.rf_week)

        ctx_grad = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ, 
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week
        )
        
        ctx_grad.scores = self.scores_arr
        
        ctx_grad.cvar_alpha = cvar_level
        
        ctx_grad.cvar_beta = self._cvar_beta_calib
        
        ctx_grad.denom_floor = self._denom_floor

        ctx_eval = g.GradCtx(
            mu = self.er_arr,
            Sigma = self.Σ, 
            rf = self.rf_ann,
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week
        )
        
        ctx_eval.scores = self.scores_arr
        
        ctx_eval.cvar_alpha = cvar_level
        
        ctx_eval.cvar_beta = self._cvar_beta_solve
        
        ctx_eval.denom_floor = self._denom_floor

        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week,
            eps = ctx_grad.eps
        )
        
        ir5 = g.ir_val_grad_from(
            R = self.R5, 
            b = b5_vec, 
            eps = ctx_grad.eps
        )

        obj = g.compose([
            (gamma[0] * scale_sh, g.sharpe_val_grad),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_ir, ir5),
            (gamma[3] * scale_sc, g.score_over_cvar_val_grad),
        ])

        w_var = cp.Variable(self.n, nonneg = True)
       
        lin_param = cp.Parameter(self.n)
       
        pen_bl = cp.sum_squares(w_var - w_bl_a)
       
        obj_expr = gamma[4] * scale_pen * pen_bl - lin_param @ w_var
       
        cons = self.build_constraints(
            w_var = w_var,
            single_port_cap = True
        )
       
        prob = cp.Problem(cp.Minimize(obj_expr), cons)


        def _from_initial(
            w0: np.ndarray
        ):
        
            w = w0.copy()
        
            for _ in range(max_iter):
        
                work = g.Work()
        
                _, grad = obj(w, ctx_grad, work)
        
                lin_param.value = self._np1(
                    x = grad
                )

                if not self._solve(prob) or w_var.value is None:
               
                    raise RuntimeError("comb_port3 CCP: subproblem solve failed")

                w_candidate = w_var.value

                eta = 1.0
               
                c = 1e-4
               
                direction = w_candidate - w

                work_eval = g.Work()
               
                sh_v, _ = g.sharpe_val_grad(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
               
                so_v, _ = sortino(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
               
                ir_v, _ = ir5(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
               
                sc_v, _ = g.score_over_cvar_val_grad(
                    w = w, 
                    ctx = ctx_eval,
                    work = work_eval
                )
               
                prev_true = (
                    gamma[0] * scale_sh * sh_v
                    + gamma[1] * scale_so * so_v
                    + gamma[2] * scale_ir * ir_v
                    + gamma[3] * scale_sc * sc_v
                    - gamma[4] * scale_pen * float(np.sum((w - w_bl_a) ** 2))
                )

                while eta > 1e-6:
                 
                    w_trial = w + eta * direction
                 
                    work_eval = g.Work()
                 
                    sh_t, _ = g.sharpe_val_grad(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                 
                    so_t, _ = sortino(
                        w = w_trial, 
                        ctx  = ctx_eval,
                        work = work_eval
                    )
                 
                    ir_t, _ = ir5(
                        w = w_trial,
                        ctx = ctx_eval,
                        work = work_eval
                    )
                 
                    sc_t, _ = g.score_over_cvar_val_grad(
                        w = w_trial, 
                        ctx = ctx_eval,
                        work = work_eval
                    )
                 
                    s_trial = (
                        gamma[0] * scale_sh * sh_t
                        + gamma[1] * scale_so * so_t
                        + gamma[2] * scale_ir * ir_t
                        + gamma[3] * scale_sc * sc_t
                        - gamma[4] * scale_pen * float(np.sum((w_trial - w_bl_a) ** 2))
                    )
                 
                    if s_trial >= prev_true + c * eta * float(np.dot(direction, direction)):
                 
                        w = w_trial
                 
                        prev_true = s_trial
                 
                        break
                 
                    eta *= 0.5
                
                else:
                
                    w = w_candidate

            work = g.Work()
            
            sh_val, sh_g = g.sharpe_val_grad(
                w = w, 
                ctx = ctx_eval,
                work = work
            )
            
            so_val, so_g = sortino(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )
            
            ir_val, ir_g = ir5(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )

            r = ctx_eval.R1 @ w
            
            T = r.shape[0]
            
            z = cp.Variable()
            
            u = cp.Variable(T, nonneg = True)
            
            alpha = float(cvar_level)
            
            cvar_obj = z + (1.0 / (alpha * T)) * cp.sum(u)
            
            cvar_cons = [u >= -(r + z)]
            
            cvar_prob = cp.Problem(cp.Minimize(cvar_obj), cvar_cons)
            
            if not self._solve(cvar_prob) or z.value is None or u.value is None:
            
                raise RuntimeError("comb_port3: CVaR LP failed in final scoring")
            
            cvar_exact = float(cvar_obj.value)
            
            sc_ratio = float(self.scores_arr @ w) / max(cvar_exact, self._denom_floor)

            pen_val = float(np.sum((w - w_bl_a) ** 2))
            
            score = (
                gamma[0] * scale_sh * sh_val
                + gamma[1] * scale_so * so_val
                + gamma[2] * scale_ir * ir_val
                + gamma[3] * scale_sc * sc_ratio
                - gamma[4] * scale_pen * pen_val
            )

            _, sc_g = g.score_over_cvar_val_grad(
                w = w,
                ctx = ctx_eval,
                work = work
            )
           
            g_pen = 2.0 * (w - w_bl_a)
           
            diag = {
                "Sharpe_push": gamma[0] * scale_sh * float(np.linalg.norm(sh_g)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(so_g)),
                "IR5y_push": gamma[2] * scale_ir * float(np.linalg.norm(ir_g)),
                "SC_push": gamma[3] * scale_sc * float(np.linalg.norm(sc_g)),
                "BLpen_push": gamma[4] * scale_pen * float(np.linalg.norm(g_pen)),
            }
            
            return w, score, diag

        best_w = None
        
        best_s = -np.inf
        
        best_diag = {}
      
        for w0 in initials:
       
            try:
        
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
        
            except RuntimeError:
        
                continue
        
            if s_cand > best_s:
        
                best_w = w_cand
                
                best_s = s_cand
                
                best_diag = diag

        if best_w is None:
           
            raise RuntimeError("comb_port3: all initialisations failed")

        self._last_diag = best_diag
      
        return pd.Series(best_w, index = self.universe, name = "comb_port3")
    
    
    def comb_port4(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        max_iter: int = 1000,
    ) -> pd.Series:
        """
        CCP optimiser with **IR(5y)** and **Score/CVaR(1y)** in the reward, **L1 proximity**
        to the BL portfolio, and **sector risk-contribution caps** enforced via
        successive linearisation (CCP). Includes Armijo backtracking on the true smoothed objective.

        Objective (maximised)
        ---------------------
        Same reward terms as comb_port3, but L1 BL penalty:
        
            P₁^BL(w) = ||w − w_BL||₁.

        Composite:
       
        F(w) = γ_SH κ_SH S(w) + γ_SO κ_SO SO(w) + γ_IR κ_IR IR₅(w) + γ_SC κ_SC SC(w) − γ_P κ_P P₁^BL(w).

        Gradients used
        --------------
        - ∇ S, ∇ SO, ∇ IR₅, ∇ SC as before.
       
        - Subgradient of L1 penalty: ∂ P₁^BL(w) contains sign(w − w_BL) componentwise.

        Sector risk-contribution caps (linearised)
        ------------------------------------------
        For each sector s with mask m_s (diagonal D_s), define sector variance
       
            V_s(w) = wᵀ D_s Σ D_s w.
        
        Impose V_s(w) ≤ α_s · φ_t(w), where φ_t is the first-order linearisation at w_t:
        
            V_s(w) ≈ V_s(w_t) + ⟨∇ V_s(w_t), w − w_t⟩,
        
            ∇ V_s(w_t) = (Σ + Σᵀ) w_t masked to sector s.
        
        This yields convex constraints
            
            V_s(w) ≤ α_s (off_param + grad_paramᵀ w),
        
        with parameters updated each CCP iteration via `rc_update(w_t)`.

        CCP subproblem and line-search
        ------------------------------
        Subproblem:
        
            minimise   γ_P κ_P ||w − w_BL||₁ − ⟨lin_param, w⟩
        
        subject to feasibility + linearised sector caps.

        Armijo backtracking on the true smoothed composite as in comb_port3.

        Final scoring
        -------------
        Score/CVaR evaluated with exact RU LP CVaR.

        Modelling advantage
        -------------------
        Explicit control of sector-level risk concentration via linearised variance caps;
        L1 proximity promotes sparse deviations from BL and can mitigate turnover;
        the mix of 5y IR and Score/CVaR balances benchmark discipline and tail-risk robustness.
        """

        self._assert_alignment()
     
        cvar_level = self._cvar_level
     
        if self.gamma is not None:
     
            gamma = self.gamma

        w_bl_a, _, _ = self._bl
     
        initials = self._initial_seeds

        scale_sh, scale_so, scale_ir, scale_sc, scale_pen = self._calibrate_scales_bl_pen(
            gamma = gamma, 
            use_l1_penalty = True
        )

        b5_vec = self._R5_b_te[1]
        
        if b5_vec is None:
        
            b5_vec = np.full(self.R5.shape[0], self.rf_week)

        ctx_grad = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ,
            rf = self.rf_ann,
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week
        )
        
        ctx_grad.scores = self.scores_arr
        
        ctx_grad.cvar_alpha = cvar_level
        
        ctx_grad.cvar_beta = self._cvar_beta_calib
        
        ctx_grad.denom_floor = self._denom_floor

        ctx_eval = g.GradCtx(
            mu = self.er_arr,
            Sigma = self.Σ, 
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1,
            target = self.rf_week
        )
        
        ctx_eval.scores = self.scores_arr
        
        ctx_eval.cvar_alpha = cvar_level
        
        ctx_eval.cvar_beta = self._cvar_beta_solve
        
        ctx_eval.denom_floor = self._denom_floor

        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week, 
            eps = ctx_grad.eps
        )
        
        ir5 = g.ir_val_grad_from(
            R = self.R5, 
            b = b5_vec, 
            eps = ctx_grad.eps
        )

        obj = g.compose([
            (gamma[0] * scale_sh, g.sharpe_val_grad),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_ir, ir5),
            (gamma[3] * scale_sc, g.score_over_cvar_val_grad),
        ])

        w_var = cp.Variable(self.n, nonneg = True)
     
        lin_param = cp.Parameter(self.n)
     
        pen_bl = cp.norm1(w_var - w_bl_a)

        cons, rc_update = self.build_constraints(
            w_var = w_var,
            sector_risk_cap = True,
            Sigma = self.Σ,
            sector_masks = self.sector_masks,
            sector_alpha = self.alpha_dict,
            single_port_cap = True
        )
       
        obj_expr = gamma[4] * scale_pen * pen_bl - lin_param @ w_var
       
        prob = cp.Problem(cp.Minimize(obj_expr), cons)


        def _from_initial(
            w0: np.ndarray
        ):
        
            w = w0.copy()
           
            for _ in range(max_iter):
           
                work = g.Work()
           
                _, grad = obj(w, ctx_grad, work)
           
                lin_param.value = self._np1(
                    x = grad
                )

                rc_update(
                    w_t = w
                )

                if not self._solve(prob) or w_var.value is None:
                
                    raise RuntimeError("comb_port4 CCP: subproblem solve failed")


                w_candidate = w_var.value

                eta = 1.0

                c = 1e-4

                direction = w_candidate - w

                work_eval = g.Work()

                sh_v, _ = g.sharpe_val_grad(
                    w = w, 
                    ctx = ctx_eval,
                    work = work_eval
                )

                so_v, _ = sortino(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )

                ir_v, _ = ir5(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )

                sc_v, _ = g.score_over_cvar_val_grad(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )

                prev_true = (
                    gamma[0] * scale_sh * sh_v
                    + gamma[1] * scale_so * so_v
                    + gamma[2] * scale_ir * ir_v
                    + gamma[3] * scale_sc * sc_v
                    - gamma[4] * scale_pen * float(np.abs(w - w_bl_a).sum())
                )

                while eta > 1e-6:
                 
                    w_trial = w + eta * direction
                 
                    work_eval = g.Work()
                 
                    sh_t, _ = g.sharpe_val_grad(
                        w = w_trial,
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                 
                    so_t, _ = sortino(
                        w = w_trial,
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                 
                    ir_t, _ = ir5(
                        w = w_trial, 
                        ctx = ctx_eval,
                        work = work_eval
                    )
                 
                    sc_t, _ = g.score_over_cvar_val_grad(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                 
                    s_trial = (
                        gamma[0] * scale_sh * sh_t
                        + gamma[1] * scale_so * so_t
                        + gamma[2] * scale_ir * ir_t
                        + gamma[3] * scale_sc * sc_t
                        - gamma[4] * scale_pen * float(np.abs(w_trial - w_bl_a).sum())
                    )
                  
                    if s_trial >= prev_true + c * eta * float(np.dot(direction, direction)):
                  
                        w = w_trial
                  
                        prev_true = s_trial
                  
                        break
                  
                    eta *= 0.5
                
                else:
                
                    w = w_candidate

            work = g.Work()
          
            sh_val, sh_g = g.sharpe_val_grad(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )
          
            so_val, so_g = sortino(
                w = w, 
                ctx = ctx_eval,
                work = work
            )
          
            ir_val, ir_g = ir5(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )

            r = ctx_eval.R1 @ w
            
            T = r.shape[0]
            
            z = cp.Variable()
            
            u = cp.Variable(T, nonneg = True)
            
            alpha = float(cvar_level)
            
            cvar_obj = z + (1.0 / (alpha * T)) * cp.sum(u)
            
            cvar_cons = [u >= -(r + z)]
            
            cvar_prob = cp.Problem(cp.Minimize(cvar_obj), cvar_cons)
            
            if not self._solve(cvar_prob) or z.value is None or u.value is None:
            
                raise RuntimeError("comb_port4: CVaR LP failed in final scoring")
            
            cvar_exact = float(cvar_obj.value)
            
            sc_ratio = float(self.scores_arr @ w) / max(cvar_exact, self._denom_floor)

            pen_val = float(np.abs(w - w_bl_a).sum())
            
            score = (
                gamma[0] * scale_sh * sh_val
                + gamma[1] * scale_so * so_val
                + gamma[2] * scale_ir * ir_val
                + gamma[3] * scale_sc * sc_ratio
                - gamma[4] * scale_pen * pen_val
            )

            _, sc_g = g.score_over_cvar_val_grad(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )
            
            g_pen = np.sign(w - w_bl_a)
           
            diag = {
                "Sharpe_push": gamma[0] * scale_sh * float(np.linalg.norm(sh_g)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(so_g)),
                "IR5y_push": gamma[2] * scale_ir * float(np.linalg.norm(ir_g)),
                "SC_push": gamma[3] * scale_sc * float(np.linalg.norm(sc_g)),
                "BLpen_push": gamma[4] * scale_pen * float(np.linalg.norm(g_pen)),
            }
           
            return w, score, diag

        best_w = None
        
        best_s = -np.inf
        
        best_diag = {}
        
        for w0 in initials:
         
            try:
         
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
         
            except RuntimeError:
         
                continue
         
            if s_cand > best_s:
         
                best_w = w_cand
                
                best_s = s_cand
                
                best_diag = diag

        if best_w is None:
       
            raise RuntimeError("comb_port4: all initialisations failed")

       
        self._last_diag = best_diag
       
        return pd.Series(best_w, index = self.universe, name = "comb_port4")


    def comb_port5(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),  # SH, SO, BL, IR(1y), SC, ASR
        max_iter: int = 1000,
        tol: float = 1e-12,
    ) -> pd.Series:
        """
        CCP optimiser for a rich composite that **adds Adjusted Sharpe Ratio (ASR)** to
        Sharpe, Sortino, BL-Sharpe, IR(1y), and Score/CVaR(1y). Final Score/CVaR uses exact RU LP.

        Objective (maximised)
        ---------------------
        Let ASR(w) denote the adjusted Sharpe that accounts for skewness and kurtosis
        (via the practitioner’s approximation implemented in `_asr_val_grad_single`).
        With γ = (γ_SH, γ_SO, γ_BL, γ_IR, γ_SC, γ_ASR) and scales κ from
        `_calibrate_scales_by_grad_ir_sc_asr`, define
        
            F(w) = γ_SH κ_SH S(w) + γ_SO κ_SO SO(w) + γ_BL κ_BL S_bl(w) + γ_IR κ_IR IR₁(w) + γ_SC κ_SC SC(w) + γ_ASR κ_ASR ASR(w).

        Gradients used
        --------------
        - ∇ S, ∇ SO, ∇ S_bl, ∇ IR₁, ∇ SC as in comb_port1.
        
        - ∇ ASR(w): taken from `_asr_val_grad_single`, which internally computes the gradient
        of the adjusted Sharpe term wrt w using vectorised weekly central moments and the
        annual (μ, Σ, r_f) Sharpe pieces.

        CCP linearisation and subproblem
        --------------------------------
        Linearise the full reward at w_t:
       
            minimise   − ⟨lin_param, w⟩
        
        subject to feasibility, with 
        
            lin_param = ∇ F(w_t).

        Stopping and acceptance
        -----------------------
        Stop on small step and subproblem objective change thresholds. For reporting,
        recompute SC with exact RU LP CVaR.

        Modelling advantage
        -------------------
        ASR rewards portfolios that achieve Sharpe under favourable higher moments
        (skew/kurtosis), complementing Sortino’s downside sensitivity and IR’s benchmark
        discipline; BL-Sharpe adds stability to expected returns; the calibration of scales
        harmonises the “push” of all terms during the CCP iterations.
        """

        self._assert_alignment()
       
        cvar_level = self._cvar_level
       
        if gamma is not None:
       
            gamma = self.gamma
       
        if len(gamma) == 5:
       
            gamma = (*gamma, 1.0)
       
        if len(gamma) != 6:
       
            raise ValueError("comb_port5 expects gamma with 6 entries: (SH, SO, BL, IR1y, SC, ASR).")

        tol_floor = (np.trace(self.Σ) / max(1, self.n)) * 1e-8
       
        tol = max(tol, tol_floor)

        initials = self._initial_seeds
       
        _, μ_bl_arr, Σb_cal = self._bl

        _r1_b = getattr(self, "_R1_b_te", (None, None))[1]
       
        b1_vec = _r1_b if _r1_b is not None else np.full(self.R1.shape[0], self.rf_week)

        scale_s, scale_so, scale_bl, scale_ir, scale_sc, scale_asr = self._calibrate_scales_by_grad_ir_sc_asr(
            gamma = gamma
        )

        ctx_grad = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ,
            rf = self.rf_ann,
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week,
        )
        
        ctx_grad.scores = self.scores_arr
        
        ctx_grad.cvar_alpha = cvar_level
        
        ctx_grad.cvar_beta = self._cvar_beta_calib
        
        ctx_grad.denom_floor = self._denom_floor

        ctx_eval = g.GradCtx(
            mu = self.er_arr,
            Sigma = self.Σ,
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week,
        )
        
        ctx_eval.scores = self.scores_arr
      
        ctx_eval.cvar_alpha = cvar_level
      
        ctx_eval.cvar_beta = self._cvar_beta_solve
      
        ctx_eval.denom_floor = self._denom_floor

        ctx_bl = g.GradCtx(
            mu = μ_bl_arr,
            Sigma = Σb_cal, 
            rf = self.rf_ann,
            eps = 1e-12
        )
      
      
        def bl_sharpe_val_grad(
            w, 
            _ctx, 
            work = None
        ):
        
            return g.sharpe_val_grad(
                w = w,
                ctx = ctx_bl, 
                work = work
            )
            

        ir1 = g.ir_val_grad_from(
            R = self.R1, 
            b = b1_vec, 
            eps = ctx_grad.eps
        )
        
        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week,
            eps = ctx_grad.eps
        )


        def asr_val_grad(
            w: np.ndarray, 
            _ctx: g.GradCtx,
            work: g.Work | None = None
        ):
        
            return self._asr_val_grad_single(
                w = w
            )


        obj = g.compose([
            (gamma[0] * scale_s, g.sharpe_val_grad),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_bl, bl_sharpe_val_grad),
            (gamma[3] * scale_ir, ir1),
            (gamma[4] * scale_sc, g.score_over_cvar_val_grad),  
            (gamma[5] * scale_asr, asr_val_grad),
        ])

        w_var = cp.Variable(self.n, nonneg = True)
        
        lin_param = cp.Parameter(self.n)
        
        obj_expr = -lin_param @ w_var
        
        cons = self.build_constraints(w_var = w_var, single_port_cap = True)
        
        prob = cp.Problem(cp.Minimize(obj_expr), cons)

       
        def _from_initial(
            w0: np.ndarray
        ):
        
            w = w0.copy()
        
            prev_obj = None

            for _ in range(max_iter):
        
                work = g.Work()
        
                _, grad = obj(w, ctx_grad, work)
        
                lin_param.value = self._np1(
                    x = grad
                )

                if not self._solve(prob) or w_var.value is None:
               
                    raise RuntimeError("comb_port5 CCP: solver failed")
               
                w_new = w_var.value
               
                obj_val = float(obj_expr.value)

                if prev_obj is not None and np.linalg.norm(w_new - w) < tol and abs(obj_val - prev_obj) < tol:
               
                    w = w_new
               
                    break
               
                prev_obj = obj_val
               
                w = w_new

            work = g.Work()
            
            sh_val, sh_g = g.sharpe_val_grad(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )
            
            so_val, so_g = sortino(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )
            
            bl_val, bl_g = bl_sharpe_val_grad(
                w = w, 
                _ctx = ctx_eval,
                work = work
            )
            
            ir_val, ir_g = ir1(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )

            r = ctx_eval.R1 @ w
            
            T = r.shape[0]
            
            z = cp.Variable()
            
            u = cp.Variable(T, nonneg = True)
            
            alpha = float(cvar_level)
           
            cvar_obj = z + (1.0 / (alpha * T)) * cp.sum(u)
           
            cvar_cons = [u >= -(r + z)]
           
            cvar_prob = cp.Problem(cp.Minimize(cvar_obj), cvar_cons)
           
            if not self._solve(cvar_prob) or z.value is None or u.value is None:
           
                raise RuntimeError("comb_port5: CVaR LP failed in final scoring")
           
            cvar_exact = float(cvar_obj.value)
           
            sc_ratio = float(self.scores_arr @ w) / max(cvar_exact, self._denom_floor)

            asr_val, asr_g = self._asr_val_grad_single(
                w = w
            )

            score = (
                gamma[0] * scale_s * sh_val +
                gamma[1] * scale_so * so_val +
                gamma[2] * scale_bl * bl_val +
                gamma[3] * scale_ir * ir_val +
                gamma[4] * scale_sc  * sc_ratio +
                gamma[5] * scale_asr * asr_val
            )

            _, sc_g = g.score_over_cvar_val_grad(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )
           
            diag = {
                "Sharpe_push": gamma[0] * scale_s * float(np.linalg.norm(sh_g)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(so_g)),
                "BL_push": gamma[2] * scale_bl * float(np.linalg.norm(bl_g)),
                "IR_push": gamma[3] * scale_ir * float(np.linalg.norm(ir_g)),
                "SC_push": gamma[4] * scale_sc * float(np.linalg.norm(sc_g)),
                "ASR_push": gamma[5] * scale_asr * float(np.linalg.norm(asr_g)),
            }
         
            return w, score, diag

        best_w = None
        
        best_s = -np.inf
        
        best_diag = {}
     
        for w0 in initials:
     
            try:
     
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
     
            except RuntimeError:
     
                continue
     
            if s_cand > best_s:
     
                best_s = s_cand
                
                best_w = w_cand
                
                best_diag = diag

        if best_w is None:
         
            raise RuntimeError("comb_port5: all initialisations failed")

        self._last_diag = best_diag
     
        return pd.Series(best_w, index = self.universe, name = "comb_port5")


    def comb_port6(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 
        max_iter: int = 1000,
    ) -> pd.Series:
        """
        CCP optimiser that **combines IR(5y), Score/CVaR(1y), and ASR** with an **L2 proximity**
        penalty to BL; includes Armijo backtracking on the true smoothed composite.

        Objective (maximised)
        ---------------------
        Let
        
            γ = (γ_SH, γ_SO, γ_IR5, γ_SC, γ_ASR, γ_P)
            
        with scales κ from `_calibrate_scales_ir5_sc_pen_asr` (L2 penalty mode).
        
        Define
        
            F(w) = γ_SH κ_SH S(w) + γ_SO κ_SO SO(w) + γ_IR5 κ_IR5 IR₅(w) + γ_SC κ_SC SC(w) + γ_ASR κ_ASR ASR(w) − γ_P κ_P ||w − w_BL||₂².

        Gradients used
        --------------
        - ∇ S, ∇ SO, ∇ IR₅, ∇ SC as in comb_port3.
       
        - ∇ ASR from `_asr_val_grad_single`.
       
        - ∇ ||w − w_BL||₂² = 2 (w − w_BL).

        CCP subproblem and line-search
        ------------------------------
        Subproblem:
        
            minimise   γ_P κ_P ||w − w_BL||₂² − ⟨lin_param, w⟩
       
        subject to feasibility, with 
        
            lin_param = ∇(reward) at w_t.

        Armijo backtracking on the true smoothed objective as in comb_port3.

        Final scoring
        -------------
        Score/CVaR via exact RU LP CVaR for reporting.

        Modelling advantage
        -------------------
        Pairs long-horizon tracking quality (IR over 5y) with tail-risk robustness and higher-moment
        awareness (ASR). The L2 proximity to BL regularises towards equilibrium allocations,
        improving stability against estimation error. Backtracking improves robustness of ascent.
        """


        self._assert_alignment()
        
        cvar_level = self._cvar_level

        if gamma is not None:
       
            gamma = self.gamma
       
        if len(gamma) == 5:
       
            gamma = (*gamma, 1.0)
       
        if len(gamma) != 6:
       
            raise ValueError("comb_port6 expects gamma with 6 entries: (SH, SO, IR5y, SC, ASR, PEN).")

        w_bl_a, _, _ = self._bl
       
        initials = [np.asarray(w, float) for w in self._initial_seeds if w is not None and np.ndim(w) == 1]
       
        if not initials:
       
            raise RuntimeError("comb_port6: no valid initial weights")

        b5_vec = self._R5_b_te[1]
       
        if b5_vec is None:
       
            b5_vec = np.full(self.R5.shape[0], self.rf_week)

        scale_sh, scale_so, scale_ir, scale_sc, scale_asr, scale_pen =  self._calibrate_scales_ir5_sc_pen_asr(
            gamma = gamma,
            use_l1_penalty = False
        )

        ctx_grad = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ,
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week
        )
        
        ctx_grad.scores = self.scores_arr
        
        ctx_grad.cvar_alpha = cvar_level
        
        ctx_grad.cvar_beta = self._cvar_beta_calib
        
        ctx_grad.denom_floor = self._denom_floor

        ctx_eval = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ, 
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week
        )
        
        ctx_eval.scores = self.scores_arr
        
        ctx_eval.cvar_alpha = cvar_level
        
        ctx_eval.cvar_beta = self._cvar_beta_solve
        
        ctx_eval.denom_floor = self._denom_floor

        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week,
            eps = ctx_grad.eps
        )
        
        ir5 = g.ir_val_grad_from(
            R = self.R5, 
            b = b5_vec, 
            eps = ctx_grad.eps
        )


        def asr_val_grad(
            w: np.ndarray,
            _ctx: g.GradCtx,
            work: g. Work | None = None
        ):
        
            return self._asr_val_grad_single(
                w = w
                )


        obj = g.compose([
            (gamma[0] * scale_sh, g.sharpe_val_grad),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_ir, ir5),
            (gamma[3] * scale_sc, g.score_over_cvar_val_grad),
            (gamma[4] * scale_asr, asr_val_grad),
        ])

        w_var = cp.Variable(self.n, nonneg = True)
        
        lin_param = cp.Parameter(self.n)
        
        pen_bl = cp.sum_squares(w_var - w_bl_a)
        
        obj_expr = gamma[5] * scale_pen * pen_bl - lin_param @ w_var
        
        cons = self.build_constraints(
            w_var = w_var,
            single_port_cap = True
        )
        
        prob = cp.Problem(cp.Minimize(obj_expr), cons)


        def _from_initial(
            w0: np.ndarray
        ):
        
            w = w0.copy()
        
            for _ in range(max_iter):
        
                work = g.Work()
        
                _, grad = obj(w, ctx_grad, work)
        
                lin_param.value = self._np1(grad)

                if not self._solve(prob) or w_var.value is None:
        
                    raise RuntimeError("comb_port6 CCP: subproblem solve failed")

                w_candidate = w_var.value

                eta = 1.0
        
                c = 1e-4
        
                direction = w_candidate - w

                work_eval = g.Work()
        
                sh_v, _ = g.sharpe_val_grad(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
        
                so_v, _ = sortino(
                    w = w, 
                    ctx = ctx_eval,
                    work = work_eval
                )
        
                ir_v, _ = ir5(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
        
                sc_v, _ = g.score_over_cvar_val_grad(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
        
                asr_v, _ = asr_val_grad(
                    w = w, 
                    _ctx = ctx_eval, 
                    work = work_eval
                )
        
                prev_true = (
                    gamma[0] * scale_sh  * sh_v
                    + gamma[1] * scale_so * so_v
                    + gamma[2] * scale_ir * ir_v
                    + gamma[3] * scale_sc * sc_v
                    + gamma[4] * scale_asr * asr_v
                    - gamma[5] * scale_pen * float(np.sum((w - w_bl_a) ** 2))
                )

                while eta > 1e-6:
                   
                    w_trial = w + eta * direction
                   
                    work_eval = g.Work()
                   
                    sh_t, _ = g.sharpe_val_grad(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                   
                    so_t, _ = sortino(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                   
                    ir_t, _ = ir5(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                   
                    sc_t, _ = g.score_over_cvar_val_grad(
                        w = w_trial, 
                        ctx = ctx_eval,
                        work = work_eval
                    )
                   
                    asr_t, _ = asr_val_grad(
                        w = w_trial, 
                        _ctx = ctx_eval,
                        work = work_eval
                    )
                   
                    s_trial = (
                        gamma[0] * scale_sh  * sh_t
                        + gamma[1] * scale_so * so_t
                        + gamma[2] * scale_ir * ir_t
                        + gamma[3] * scale_sc * sc_t
                        + gamma[4] * scale_asr * asr_t
                        - gamma[5] * scale_pen * float(np.sum((w_trial - w_bl_a) ** 2))
                    )
                   
                    if s_trial >= prev_true + c * eta * float(np.dot(direction, direction)):
                   
                        w = w_trial
                   
                        prev_true = s_trial
                   
                        break
                   
                    eta *= 0.5
                
                else:
                
                    w = w_candidate

            work = g.Work()
           
            sh_val, sh_g = g.sharpe_val_grad(
                w = w, 
                ctx = ctx_eval,
                work = work
            )
           
            so_val, so_g = sortino(
                w = w, 
                ctx = ctx_eval, 
                work = work
            )
           
            ir_val, ir_g = ir5(
                w = w,
                ctx = ctx_eval,
                work = work
            )

            r = ctx_eval.R1 @ w
           
            T = r.shape[0]
           
            z = cp.Variable()
           
            u = cp.Variable(T, nonneg = True)
           
            alpha = float(cvar_level)
           
            cvar_obj = z + (1.0 / (alpha * T)) * cp.sum(u)
           
            cvar_cons = [u >= -(r + z)]
           
            cvar_prob = cp.Problem(cp.Minimize(cvar_obj), cvar_cons)
           
            if not self._solve(cvar_prob) or z.value is None or u.value is None:
           
                raise RuntimeError("comb_port6: CVaR LP failed in final scoring")
           
            cvar_exact = float(cvar_obj.value)
           
            sc_ratio = float(self.scores_arr @ w) / max(cvar_exact, self._denom_floor)

            asr_val, asr_g = self._asr_val_grad_single(
                w = w
            )
            
            pen_val = float(np.sum((w - w_bl_a) ** 2))
            
            score = (
                gamma[0] * scale_sh  * sh_val
                + gamma[1] * scale_so * so_val
                + gamma[2] * scale_ir * ir_val
                + gamma[3] * scale_sc * sc_ratio
                + gamma[4] * scale_asr * asr_val
                - gamma[5] * scale_pen * pen_val
            )

            _, sc_g = g.score_over_cvar_val_grad(
                w = w,
                ctx = ctx_eval, 
                work = work
            )
            
            g_pen = 2.0 * (w - w_bl_a)
            
            diag = {
                "Sharpe_push": gamma[0] * scale_sh * float(np.linalg.norm(sh_g)),
                "Sortino_push": gamma[1] * scale_so  * float(np.linalg.norm(so_g)),
                "IR5y_push": gamma[2] * scale_ir * float(np.linalg.norm(ir_g)),
                "SC_push": gamma[3] * scale_sc * float(np.linalg.norm(sc_g)),
                "ASR_push": gamma[4] * scale_asr * float(np.linalg.norm(asr_g)),
                "BLpen_push": gamma[5] * scale_pen * float(np.linalg.norm(g_pen)),
            }
            
            return w, score, diag

        best_w = None
        
        best_s = -np.inf
        
        best_diag = {}
        
        for w0 in initials:
        
            try:
        
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
        
            except RuntimeError:
        
                continue
        
            if s_cand > best_s:
        
                best_w = w_cand
                
                best_s = s_cand
                
                best_diag = diag

        if best_w is None:
       
            raise RuntimeError("comb_port6: all initialisations failed")

        self._last_diag = best_diag
       
        return pd.Series(best_w, index = self.universe, name = "comb_port6")


    def comb_port7(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 
        max_iter: int = 1000,
    ) -> pd.Series:
        """
        CCP optimiser that **combines IR(5y), Score/CVaR(1y), and ASR** with an **L1 proximity**
        penalty to BL and **sector risk-contribution caps**, enforced via linearisation.
        Includes Armijo backtracking on the true smoothed composite.

        Objective (maximised)
        ---------------------
        With γ = (γ_SH, γ_SO, γ_IR5, γ_SC, γ_ASR, γ_P) and scales κ from
        `_calibrate_scales_ir5_sc_pen_asr` (L1 penalty mode):
        
            F(w) = γ_SH κ_SH S(w) + γ_SO κ_SO SO(w) + γ_IR5 κ_IR5 IR₅(w) + γ_SC κ_SC SC(w) + γ_ASR κ_ASR ASR(w) − γ_P κ_P ||w − w_BL||₁.

        Gradients used
        --------------
        - ∇ S, ∇ SO, ∇ IR₅, ∇ SC as in comb_port4.
       
        - ∇ ASR from `_asr_val_grad_single`.
       
        - L1 penalty subgradient: sign(w − w_BL) componentwise.

        Sector risk-contribution caps (linearised)
        ------------------------------------------
        For each sector s:  
        
            V_s(w) = wᵀ D_s Σ D_s w.
        
        Impose 
        
            V_s(w) ≤ α_s · φ_t(w) 
            
        using first-order linearisation at w_t, yielding convex constraints of the form 
        
            V_s(w) ≤ α_s (off_param + grad_paramᵀ w),
            
        with parameters updated each iteration via `rc_update(w_t)`.

        CCP subproblem and line-search
        ------------------------------
        Subproblem:
        
            minimise   γ_P κ_P ||w − w_BL||₁ − ⟨lin_param, w⟩
        
        subject to feasibility + linearised sector caps, with 
        
            lin_param = ∇(reward) at w_t.

        Armijo backtracking on the true smoothed objective as in comb_port3.

        Final scoring
        -------------
        Score/CVaR evaluated with exact RU LP CVaR.

        Modelling advantage
        -------------------
        Simultaneously enforces sector-level risk budgets, penalises large deviations from BL
        in L1 geometry (promoting sparse tilts), and captures long-horizon benchmark discipline,
        tail-risk robustness, and higher-moment quality via ASR.
        """

        self._assert_alignment()
       
        cvar_level = self._cvar_level

        if gamma is not None:
       
            gamma = self.gamma
            
        if len(gamma) == 5:
       
            gamma = (*gamma, 1.0)
       
        if len(gamma) != 6:
       
            raise ValueError("comb_port7 expects gamma with 6 entries: (SH, SO, IR5y, SC, ASR, PEN).")

        w_bl_a, _, _ = self._bl
       
        initials = self._initial_seeds

        b5_vec = self._R5_b_te[1]
       
        if b5_vec is None:
       
            b5_vec = np.full(self.R5.shape[0], self.rf_week)

        scale_sh, scale_so, scale_ir, scale_sc, scale_asr, scale_pen = self._calibrate_scales_ir5_sc_pen_asr(
            gamma = gamma, 
            use_l1_penalty = True
        )

        ctx_grad = g.GradCtx(
            mu = self.er_arr,
            Sigma = self.Σ, 
            rf = self.rf_ann,
            eps = 1e-12,
            R1 = self.R1,
            target = self.rf_week
        )
        
        ctx_grad.scores = self.scores_arr
        
        ctx_grad.cvar_alpha = cvar_level
        
        ctx_grad.cvar_beta = self._cvar_beta_calib
        
        ctx_grad.denom_floor = self._denom_floor

        ctx_eval = g.GradCtx(
            mu = self.er_arr, 
            Sigma = self.Σ,
            rf = self.rf_ann, 
            eps = 1e-12,
            R1 = self.R1, 
            target = self.rf_week
        )
        
        ctx_eval.scores = self.scores_arr
        
        ctx_eval.cvar_alpha = cvar_level
        
        ctx_eval.cvar_beta = self._cvar_beta_solve
        
        ctx_eval.denom_floor = self._denom_floor

        sortino = g.sortino_val_grad_from(
            R = self.R1, 
            target = self.rf_week,
            eps = ctx_grad.eps
        )
       
        ir5 = g.ir_val_grad_from(
            R = self.R5,
            b = b5_vec, 
            eps = ctx_grad.eps
        )

        
        def asr_val_grad(
            w: np.ndarray,
            _ctx: g.GradCtx,
            work: g.Work | None = None
        ):
        
            return self._asr_val_grad_single(
                w = w
            )


        obj = g.compose([
            (gamma[0] * scale_sh, g.sharpe_val_grad),
            (gamma[1] * scale_so, sortino),
            (gamma[2] * scale_ir, ir5),
            (gamma[3] * scale_sc, g.score_over_cvar_val_grad),
            (gamma[4] * scale_asr, asr_val_grad),
        ])

        w_var = cp.Variable(self.n, nonneg = True)
        
        lin_param = cp.Parameter(self.n)
        
        pen_bl = cp.norm1(w_var - w_bl_a)

        cons, rc_update = self.build_constraints(
            w_var = w_var,
            sector_risk_cap = True,
            Sigma = self.Σ,
            sector_masks = self.sector_masks,
            sector_alpha = self.alpha_dict,
            single_port_cap =True
        )
        
        obj_expr = gamma[5] * scale_pen * pen_bl - lin_param @ w_var
        
        prob = cp.Problem(cp.Minimize(obj_expr), cons)

        
        def _from_initial(
            w0: np.ndarray
        ):
        
            w = w0.copy()
        
            for _ in range(max_iter):
        
                work = g.Work()
        
                _, grad = obj(w, ctx_grad, work)
        
                lin_param.value = self._np1(
                    x = grad
                )

                rc_update(
                    w_t = w
                )

                if not self._solve(prob) or w_var.value is None:
                   
                    raise RuntimeError("comb_port7 CCP: subproblem solve failed")

                w_candidate = w_var.value

                eta = 1.0
               
                c = 1e-4
               
                direction = w_candidate - w

                work_eval = g.Work()
               
                sh_v, _ = g.sharpe_val_grad(
                    w = w, 
                    ctx = ctx_eval,
                    work = work_eval
                )
               
                so_v, _ = sortino(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
               
                ir_v, _ = ir5(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
               
                sc_v, _ = g.score_over_cvar_val_grad(
                    w = w, 
                    ctx = ctx_eval, 
                    work = work_eval
                )
               
                asr_v, _ = asr_val_grad(
                    w = w,
                    _ctx = ctx_eval,
                    work = work_eval
                )
                
                prev_true = (
                    gamma[0] * scale_sh  * sh_v
                    + gamma[1] * scale_so * so_v
                    + gamma[2] * scale_ir * ir_v
                    + gamma[3] * scale_sc * sc_v
                    + gamma[4] * scale_asr * asr_v
                    - gamma[5] * scale_pen * float(np.abs(w - w_bl_a).sum())
                )

                while eta > 1e-6:
                
                    w_trial = w + eta * direction
                
                    work_eval = g.Work()
                
                    sh_t, _ = g.sharpe_val_grad(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                
                    so_t, _ = sortino(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                
                    ir_t, _ = ir5(
                        w = w_trial,
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                
                    sc_t, _ = g.score_over_cvar_val_grad(
                        w = w_trial, 
                        ctx = ctx_eval, 
                        work = work_eval
                    )
                
                    asr_t, _ = asr_val_grad(
                        w = w_trial, 
                        _ctx = ctx_eval, 
                        work = work_eval
                    )
                
                    s_trial = (
                        gamma[0] * scale_sh * sh_t
                        + gamma[1] * scale_so * so_t
                        + gamma[2] * scale_ir * ir_t
                        + gamma[3] * scale_sc * sc_t
                        + gamma[4] * scale_asr * asr_t
                        - gamma[5] * scale_pen * float(np.abs(w_trial - w_bl_a).sum())
                    )
                    
                    if s_trial >= prev_true + c * eta * float(np.dot(direction, direction)):
                     
                        w = w_trial
                     
                        prev_true = s_trial
                     
                        break
                   
                    eta *= 0.5
                
                else:
                
                    w = w_candidate

            work = g.Work()
           
            sh_val, sh_g = g.sharpe_val_grad(w, ctx_eval, work)
           
            so_val, so_g = sortino(
                w = w, 
                ctx = ctx_eval,
                work = work
            )
           
            ir_val, ir_g = ir5(
                w = w, 
                ctx = ctx_eval,
                work = work
            )

            r = ctx_eval.R1 @ w
           
            T = r.shape[0]
           
            z = cp.Variable()
           
            u = cp.Variable(T, nonneg = True)
           
            alpha = float(cvar_level)
           
            cvar_obj = z + (1.0 / (alpha * T)) * cp.sum(u)
           
            cvar_cons = [u >= -(r + z)]
           
            cvar_prob = cp.Problem(cp.Minimize(cvar_obj), cvar_cons)
           
            if not self._solve(cvar_prob) or z.value is None or u.value is None:
           
                raise RuntimeError("comb_port7: CVaR LP failed in final scoring")
           
            cvar_exact = float(cvar_obj.value)
           
            sc_ratio = float(self.scores_arr @ w) / max(cvar_exact, self._denom_floor)

            asr_val, asr_g = self._asr_val_grad_single(
                w = w
            )

            pen_val = float(np.abs(w - w_bl_a).sum())
          
            score = (
                gamma[0] * scale_sh  * sh_val
                + gamma[1] * scale_so * so_val
                + gamma[2] * scale_ir * ir_val
                + gamma[3] * scale_sc * sc_ratio
                + gamma[4] * scale_asr * asr_val
                - gamma[5] * scale_pen * pen_val
            )

            _, sc_g = g.score_over_cvar_val_grad(
                w = w,
                ctx = ctx_eval, 
                work = work
            )
          
            g_pen = np.sign(w - w_bl_a)
          
            diag = {
                "Sharpe_push": gamma[0] * scale_sh * float(np.linalg.norm(sh_g)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(so_g)),
                "IR5y_push": gamma[2] * scale_ir * float(np.linalg.norm(ir_g)),
                "SC_push": gamma[3] * scale_sc * float(np.linalg.norm(sc_g)),
                "ASR_push": gamma[4] * scale_asr * float(np.linalg.norm(asr_g)),
                "BLpen_push": gamma[5] * scale_pen * float(np.linalg.norm(g_pen)),
            }
           
            return w, score, diag

        best_w = None
        
        best_s = -np.inf
        
        best_diag = {}
        
        for w0 in initials:
      
            try:
      
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
      
            except RuntimeError:
      
                continue
      
            if s_cand > best_s:
      
                best_w = w_cand
                
                best_s = s_cand
                
                best_diag = diag

        if best_w is None:
         
            raise RuntimeError("comb_port7: all initialisations failed")

        self._last_diag = best_diag
       
        return pd.Series(best_w, index = self.universe, name = "comb_port7")
