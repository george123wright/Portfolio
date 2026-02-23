"""
Portfolio analytics, risk metrics, and non-Gaussian Monte Carlo simulation.

This module provides:

1) Vectorised portfolio arithmetic
  
   - Portfolio return:            r_p(t) = ∑_{i=1}^N w_i r_i(t)
  
   - Volatility:                  σ_p = √(w^⊤ Σ w)
  
   - Tracking error (TE):         TE = stdev(r_a − r_b)

2) Annualisation and ratios
  
   - Annualised return (geometric):      (∏_{t=1}^n (1 + r_t))^{P/n} − 1
  
   - Annualised volatility:              σ_ann = σ_pp √P
  
   - Sharpe ratio:                       SR = (μ_ann − R_f) / σ_ann
  
   - Sortino ratio:                      Sortino = (μ_ann − R_f)/σ_down, with
                                         σ_down = √( E[(min(0, r − τ))^2] ) · √P
   - Calmar ratio:                       Calmar = CAGR / |MaxDD|
  
   - Treynor ratio:                      (μ_p − R_f) / β_p
  
   - Information ratio:                  IR = (μ_p − μ_b) / TE
  
   - Modigliani–Modigliani (M²):         M² = R_f + SR · σ_bench
  
   - Omega ratio (threshold τ):          Ω = (∫_{x > τ}(x − τ)dF) / (∫_{x ≤ τ}(τ − x)dF)

3) Tail risk and drawdown metrics
  
   - Gaussian VaR_α:                     VaR_α = −(μ + z_α σ)
  
   - Cornish–Fisher modified VaR_α:      z_cf = z + (z²−1)s / 6 + (z³ − 3z)(k − 3) / 24 − (2z³ − 5z) s² / 36
                                        
                                         mVaR_α = −(μ + z_cf σ)
                                         
   - Historical VaR/CVaR:                empirical quantile/mean of tail

   - Ulcer index:                        UI = √(mean(DD_t²))
  
   - Conditional Drawdown at Risk (CDaR): mean drawdown over the worst α-tail
  
   - Pain index:                         PI = −E[DD_t]
  
   - Pain ratio:                         (CAGR − R_f)/PI
  
   - Tail ratio:                         TR = q_{0.90} / |q_{0.10}| (if q_{0.10} < 0)

4) CAPM diagnostics with HAC (Newey–West)
 
   - CAPM: r_p − r_f = α + β (r_b − r_f) + ε
 
   - Annualised alpha via compounding:    α_ann = (1 + α_pp)^P − 1
 
   - Delta-method SE:                     se(α_ann) ≈ |∂α_ann/∂α_pp| · se(α_pp)
 
                                           with ∂α_ann/∂α_pp = P (1 + α_pp)^{P−1}
 
   - R² and predicted alpha relative to CAPM fair return

5) Non-Gaussian Monte Carlo (Edgeworth/CF) and QMC
 
   - Price dynamics: log S_{t+1} = log S_t + a + b ε_t
 
     where b = σ √Δt, ε_t is standard normal or Edgeworth-adjusted
 
   - Drift calibration per step to hit geometric growth G_target:
 
       choose a such that E[exp(a + b ε)] = G_target
 
     For Gaussian ε, E[exp(bε)] = exp(½ b²) ⇒ a = log G_target − ½ b².
 
     For Edgeworth ε, E[exp(bε)] is estimated by pilot simulation.

Caching
-------
An optional cache stores portfolio series, drawdowns, and annualised statistics.
Two key modes:
 
  • 'identity' : keyed by object identity (WeakKeyDictionary)
 
  • 'content'  : keyed by a content fingerprint (LRU)

All computations are implemented with NumPy/Pandas, with statsmodels for OLS/HAC.
"""


from __future__ import annotations

from numbers import Number
from typing import Any, Dict, List, Tuple, Iterable
from collections import OrderedDict
import weakref
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import qmc
import statsmodels.api as sm

import config


class LRUCache:
    """
    Bounded least-recently-used (LRU) cache for arbitrary in-memory objects.

    The cache stores key→value pairs in an OrderedDict and evicts the least-recently-
    used entry once the number of items exceeds `capacity`.

    Parameters
    ----------
    capacity : int, default 256
        Maximum number of items maintained.

    Notes
    -----
    - GET: moves the key to the end (most-recently used) on a hit.
  
    - SET: inserts/updates, then evicts the oldest item if size > capacity.
    """

    
    def __init__(
        self,
        capacity: int = 256
    ):
    
        self.capacity = int(capacity)
    
        self._od: OrderedDict[Any, Any] = OrderedDict()

    
    def get(
        self, 
        key: Any, 
        default: Any = None
    ) -> Any:
        """
        Return the cached value for `key`, marking it most-recently used.

        Parameters
        ----------
        key : Any
        default : Any, optional
            Returned when `key` is not present.

        Returns
        -------
        Any
            The cached object or `default` if absent.
        """    
        
        if key in self._od:
    
            self._od.move_to_end(key)
    
            return self._od[key]
    
        return default


    def set(
        self,
        key: Any, 
        value: Any
    ) -> None:
        """
        Insert or update the cache entry and enforce the capacity constraint.

        Evicts the least-recently used entry if `len(cache) > capacity`.
        """   
        
        if key in self._od:
    
            self._od.move_to_end(key)
    
        self._od[key] = value
    
        if len(self._od) > self.capacity:
    
            self._od.popitem(last=False)

    
    def clear(
        self
    ) -> None:
        """
        Remove all entries from the cache.
        """
        
        self._od.clear()


class PortfolioAnalytics:
    """
    Performance, risk and attribution metrics, with optional caching and
    non-Gaussian Monte Carlo simulation.

    Caching
    -------
    When enabled, repeatedly used inputs (return series, drawdown series,
    annualised stats) are memoised. Two strategies are supported:

    - cache_key='identity' :
  
        per-DataFrame caches keyed by the object's identity (WeakKeyDictionary).
  
        Fast, but identical content in distinct objects is treated separately.
   
    - cache_key='content' :
  
        a compact content fingerprint of the DataFrame is used as the key inside
  
        a bounded LRU cache.

    Methods cover:
  
    • Portfolio arithmetic (masked dot products with per-row renormalisation),
  
    • Volatility, tracking error, betas and scores,
  
    • Annualisation and ratios (Sharpe/Treynor/Sortino/Calmar/M²/Omega/IR),
  
    • VaR/CVaR (Gaussian, Cornish–Fisher, historical), Ulcer, CDaR, Pain,
  
    • CAPM Jensen alpha with HAC (Newey–West) covariance,
  
    • Edgeworth/QMC Monte Carlo for final price distributions.
  
    """

    def __init__(
        self,
        *,
        cache: bool = False,
        cache_key: str = "identity",
        cache_maxsize: int = 256,
    ) -> None:
       
        self._cache_enabled = bool(cache)
       
        if cache_key not in ("identity", "content"):
       
            raise ValueError("cache_key must be 'identity' or 'content'")
       
        self._cache_key_mode = cache_key
       
        self._cache_maxsize = int(cache_maxsize)

        self._port_series_cache_id: weakref.WeakKeyDictionary[pd.DataFrame, LRUCache] = weakref.WeakKeyDictionary()

        self._dd_cache_id: weakref.WeakKeyDictionary[pd.Series, pd.Series] = weakref.WeakKeyDictionary()

        self._annret_cache_id: weakref.WeakKeyDictionary[pd.Series, Dict[int, float]] = weakref.WeakKeyDictionary()

        self._annvol_cache_id: weakref.WeakKeyDictionary[pd.Series, Dict[int, float]] = weakref.WeakKeyDictionary()

        self._port_series_cache_content = LRUCache(capacity=cache_maxsize)


    def enable_cache(
        self, 
        key_mode: str | None = None
    ) -> None:
        """
        Enable caching and optionally set the keying strategy.

        Parameters
        ----------
        key_mode : {"identity", "content"}, optional
      
            If provided, switches the cache keying strategy.
      
        """    
        self._cache_enabled = True
    
        if key_mode:
    
            if key_mode not in ("identity", "content"):
    
                raise ValueError("cache_key must be 'identity' or 'content'")
    
            self._cache_key_mode = key_mode


    def disable_cache(
        self
    ) -> None:
        """
        Disable all memoisation (no cached reads/writes).
        """
        
        self._cache_enabled = False

    
    def clear_cache(
        self
    ) -> None:
        """
        Clear all internal caches (portfolio series, drawdowns, annualised stats).
        """
    
        self._port_series_cache_content.clear()
    
        self._port_series_cache_id.clear()
    
        self._dd_cache_id.clear()
    
        self._annret_cache_id.clear()
    
        self._annvol_cache_id.clear()


    @staticmethod
    def _as_scalar(
        x, 
        *, 
        weights = None, 
        take = 'last'
    ):
        """
        Coerce x to a float.
      
        - If scalar -> float(x).
      
        - If pd.Series with index matching weights -> weighted average.
      
        - If pd.Series time-series -> last valid (take='last') or mean (take='mean').
      
        - If 1-length array/Series -> the single value.
      
        """

        if np.isscalar(x):
    
            return float(x)

        if isinstance(x, pd.Series):
     
            if weights is not None and isinstance(weights, pd.Series):
     
                w = weights.reindex(x.index).fillna(0.0)
     
                return float((x.fillna(0.0) * w).sum())
     
            s = x.dropna()
     
            if s.empty:
     
                return float('nan')
     
            if take == 'mean':
     
                return float(s.mean())
     
            return float(s.iloc[-1])

        arr = np.asarray(x, dtype = float).ravel()
   
        if arr.size == 0:
   
            return float('nan')
   
        if arr.size == 1:
   
            return float(arr[0])

        return float(np.nanmean(arr))


    @staticmethod
    def portfolio_return(
        weights: np.ndarray,
        returns: Any
    ) -> Any:
        """
        Portfolio return aggregation.

        For weights w ∈ ℝ^N and asset returns r(t) ∈ ℝ^N at time t,
       
            r_p(t) = ∑_{i=1}^N w_i r_i(t) = w^⊤ r(t).

        Parameters
        ----------
    
        weights : np.ndarray, shape (N,)
    
        returns : pd.DataFrame | pd.Series | np.ndarray | scalar
    
            If DataFrame/2D array: computes the time-series r_p(t).
    
            If Series/1D array: returns a scalar w^⊤ r.
    
            If scalar and N=1: returns w_1 * scalar.

        Returns
        -------
        Same type as input when feasible (Series for DataFrame input, float otherwise).
        """


        if isinstance(returns, pd.DataFrame):
    
            return returns.dot(weights)
    
        elif isinstance(returns, pd.Series):
    
            return float(weights @ returns.to_numpy(copy = False))
    
        elif isinstance(returns, np.ndarray):
    
            return float(weights @ returns)
    
        if isinstance(returns, Number):
    
            if len(weights) == 1:
    
                return float(weights[0] * returns)
    
            raise TypeError("Cannot compute a multi-asset portfolio return from a single scalar")
    
        raise TypeError("Expected returns to be a Series, DataFrame, ndarray, or scalar")


    @staticmethod
    def portfolio_volatility(
        weights: np.ndarray,
        covmat: np.ndarray
    ) -> float:
        """
        Portfolio standard deviation from covariance.

            σ_p = √( w^⊤ Σ w ).

        Parameters
        ----------
        weights : np.ndarray, shape (N,)
        covmat : np.ndarray, shape (N,N)

        Returns
        -------
        float
            Portfolio volatility.
        """
        
        return float(np.sqrt(np.einsum("i,ij,j->", weights, covmat, weights, optimize = True)))


    @staticmethod
    def tracking_error(
        r_a: pd.Series,
        r_b: pd.Series
    ) -> float:
        """
        Tracking error between two return series, aligned on their common index:

            TE = stdev( r_a − r_b ),  (sample stdev, ddof=1).

        Missing values are dropped pairwise before computation.
        """

        idx = r_a.index.intersection(r_b.index)

        a = r_a.loc[idx].to_numpy(copy = False, dtype = float)
       
        b = r_b.loc[idx].to_numpy(copy = False, dtype = float)
       
        mask = ~(np.isnan(a) | np.isnan(b))
       
        if not mask.any():
       
            return float("nan")
       
        diff = a[mask] - b[mask]
       
        return float(diff.std(ddof=1))


    @staticmethod
    def port_beta(
        weights: np.ndarray, 
        beta: pd.Series
    ) -> float:
        """
        Portfolio beta as a weighted average of asset betas:

            β_p = w^⊤ β.
            
        """
        
        return float(weights @ beta)
    
    
    @staticmethod
    def port_tax(
        weights: np.ndarray,
        tax: pd.Series
    ) -> float:
        """
        Portfolio tax rate as a weighted average of asset tax rates:

            tax_p = w^⊤ tax.
            
        """
        
        return float(weights @ tax)
    
    
    @staticmethod
    def port_d_to_e(
        weights: np.ndarray,
        d_to_e: pd.Series
    ) -> float:
        """
        Portfolio debt-to-equity ratio as a weighted average of asset d/e ratios:

            (d/e)_p = w^⊤ (d/e).
            
        """
        
        return float(weights @ d_to_e.to_numpy(copy = False, dtype = float))


    @staticmethod
    def port_d_to_e_batch(
        W: np.ndarray,
        d_to_e: pd.Series,
        names: list[str]
    ) -> pd.Series:

        v = d_to_e.to_numpy(dtype=float)

        out = v @ W                   

        return pd.Series(out, index=names, name="D_to_E")


    @staticmethod
    def port_tax_batch(
        W: np.ndarray, 
        tax: pd.Series,
        names: list[str]
    ) -> pd.Series:
    
        v = tax.to_numpy(dtype=float)
    
        out = v @ W
    
        return pd.Series(out, index=names, name="Tax")


    @staticmethod
    def compute_treynor_ratio(
        port_ret: float, 
        rf: float,
        port_beta_val: float
    ) -> float:
        """
        Treynor ratio (systematic risk-adjusted return):

            Treynor = ( μ_p − R_f ) / β_p,

        where μ_p is an annualised or target-horizon portfolio return and β_p is the
        portfolio beta. Returns NaN if β_p = 0.
        """
        
        if port_beta_val == 0:
    
            return float("nan")
    
        return (port_ret - rf) / port_beta_val

    
    @staticmethod
    def port_score(
        weights: np.ndarray, 
        score: pd.Series
    ) -> float:
        """
        Weighted average of a cross-sectional score:

            s_p = w^⊤ s.
        """
        
        return float(weights @ score.to_numpy(copy = False, dtype = float))


    @staticmethod
    def annualise_vol(
        r: pd.Series | pd.DataFrame, 
        periods_per_year: int
    ) -> float | pd.Series:
        """
        Annualised volatility:

            σ_ann = σ_pp · √P,

        where σ_pp is the per-period sample stdev (columnwise for DataFrame) and P is
        `periods_per_year`.
        """

        if isinstance(r, pd.DataFrame):
    
            return r.std() * np.sqrt(periods_per_year)
    
        return float(r.std() * np.sqrt(periods_per_year))


    @staticmethod
    def annualise_returns(
        ret_series: pd.Series | pd.DataFrame,
        periods_per_year: int
    ) -> float | pd.Series:
        """
        Geometric annualisation of simple returns.

        For a series r_1,…,r_n (per-period), cumulative growth is
        
            G = ∏_{t=1}^n (1 + r_t),
       
        and the annualised return is
       
            μ_ann = G^{P/n} − 1,

        with P = periods_per_year. Vectorised columnwise for DataFrame input.
        """
        
        if isinstance(ret_series, pd.DataFrame):
    
            n = len(ret_series)
    
            if n <= 1:
    
                return pd.Series(0.0, index = ret_series.columns)
    
            cum = (1.0 + ret_series).prod()
    
            return cum ** (periods_per_year / n) - 1.0
    
        n = len(ret_series)
    
        if n <= 1:
    
            return 0.0
    
        cum = (1.0 + ret_series).prod()
    
        return float(cum ** (periods_per_year / n) - 1.0)


    @staticmethod
    def sharpe_ratio(
        r: pd.Series | pd.DataFrame,
        periods_per_year: int,
        ann_ret: float | pd.Series | None = None,
        ann_vol: float | pd.Series | None = None,
    ) -> float | pd.Series:
        """
        Sharpe ratio with optional pre-computed annualised return/volatility.

            SR = ( μ_ann − R_f ) / σ_ann.

        If `ann_ret` is None, the excess returns are formed per period as
        r − RF_per_period (from config) before geometric annualisation. If
        `ann_vol` is None, annualised volatility is computed as σ_pp √P.

        Returns scalar or a Series aligned to columns.
        """       
      
        if ann_ret is None:
       
            excess = r - config.RF_PER_WEEK
       
            ann_ex_ret = PortfolioAnalytics.annualise_returns(
                ret_series = excess, 
                periods_per_year = periods_per_year
            )
       
        else:
       
            ann_ex_ret = ann_ret - config.RF

        if ann_vol is None:

            ann_vol = PortfolioAnalytics.annualise_vol(
                r = r, 
                periods_per_year = periods_per_year
            )

        if isinstance(ann_ex_ret, pd.Series) or isinstance(ann_vol, pd.Series):

            denom = (ann_vol.replace(0, np.nan) if isinstance(ann_vol, pd.Series) else ann_vol)

            return ann_ex_ret / denom

        if ann_vol > 0:

            return float(ann_ex_ret / ann_vol)

        return np.nan


    @staticmethod
    def drawdown(
        return_series: pd.Series
    ) -> pd.DataFrame:
        """
        Drawdown path and wealth index.

        With initial wealth W_0 = 1000,
          
            W_t = W_0 ∏_{u=1}^t (1 + r_u),
           
            P_t = max_{1≤u≤t} W_u,
           
            DD_t = (W_t − P_t) / P_t.

        Returns a DataFrame with columns ["Wealth", "Previous Peak", "Drawdown"].
        """
        
        wealth_index = 1000.0 * (1.0 + return_series).cumprod()
    
        previous_peaks = wealth_index.cummax()
    
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
        df = pd.DataFrame({
            "Wealth": wealth_index,
            "Previous Peak": previous_peaks, 
            "Drawdown": drawdowns
        })
        
        return df


    @staticmethod
    def skewness(
        r: pd.Series | pd.DataFrame
    ) -> float | pd.Series:
        """
        Third standardised moment:

            γ_1 = E[(r − μ)^3] / σ^3.

        Returns scalar for Series or applies columnwise for DataFrame.
        """

        if isinstance(r, pd.DataFrame):
    
            return r.apply(PortfolioAnalytics.skewness)
    
        demeaned = r - r.mean()
    
        sigma = r.std(ddof = 0)
      
        if float(sigma) == 0.0:
      
            return 0.0
      
        return float(((demeaned ** 3).mean()) / (sigma ** 3))


    @staticmethod
    def kurtosis(
        r: pd.Series | pd.DataFrame
    ) -> float | pd.Series:
        """
        Fourth standardised moment (raw kurtosis, not excess):

            κ = E[(r − μ)^4] / σ^4.

        Returns 3 when σ=0 (degenerate). Columnwise for DataFrame.
        """
        
        if isinstance(r, pd.DataFrame):
    
            return r.apply(PortfolioAnalytics.kurtosis)
    
        demeaned = r - r.mean()
    
        sigma = r.std(ddof = 0)
    
        if float(sigma) == 0.0:
    
            return 3.0
    
        return float(((demeaned ** 4).mean()) / (sigma ** 4))


    @staticmethod
    def var_gaussian(
        r: pd.Series,
        level: float = 5.0,
        s: float | None = None,
        k: float | None = None,
        modified: bool = False,
    ) -> float:
        """
        Parametric (Gaussian or Cornish–Fisher) Value-at-Risk at level α=level%.

        Let z = Φ^{-1}(α). The Gaussian VaR is
       
            VaR_α = −( μ + z σ ).

        The Cornish–Fisher adjusted quantile is
       
            z_cf = z + (z²−1)s/6 + (z³−3z)(k−3)/24 − (2z³−5z)s²/36,

        where s is skewness and k is kurtosis (raw). The modified VaR is
       
            mVaR_α = −( μ + z_cf σ ).

        Mean and stdev are computed with ddof=0. Returns a positive loss number.
        """
       
        z = norm.ppf(level / 100.0)
      
        if modified:
      
            if s is None:
      
                s = PortfolioAnalytics.skewness(
                    r = r
                )
      
            if k is None:
      
                k = PortfolioAnalytics.kurtosis(
                    r = r
                )
      
      
            z = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * (k - 3) / 24 - (2 * z ** 3 - 5 * z) * (s ** 2) / 36)
      
        return float(-(r.mean() + z * r.std(ddof = 0)))

   
    @staticmethod
    def var_historic(
        r: pd.Series | pd.DataFrame, 
        level: float = 5.0
    ) -> float | pd.Series:
        """
        Historical (empirical) VaR at level α=level%:

            VaR_α = − quantile_α( r ).

        Returns a positive loss number. Columnwise for DataFrame.
        """
        
        if isinstance(r, pd.DataFrame):
    
            return r.aggregate(PortfolioAnalytics.var_historic, level = level)
    
        if isinstance(r, pd.Series):
           
            return float(-np.percentile(r.to_numpy(copy = False), level))
    
        raise TypeError("Expected r to be a Series or DataFrame")


    @staticmethod
    def cvar_historic(
        r: pd.Series | pd.DataFrame, 
        level: float = 5.0
    ) -> float | pd.Series:
        """
        Historical Conditional VaR (Expected Shortfall) at level α=level%:

            CVaR_α = − E[ r | r ≤ −VaR_α ].

        Returns a positive loss number. Columnwise for DataFrame.
        """
        
        if isinstance(r, pd.Series):
    
            v = PortfolioAnalytics.var_historic(
                r = r,
                level = level
            )
    
            tail = r[r <= -v]

            if not tail.empty:
                
                return float(-tail.mean())
            
            else:
                
                return 0.0
    
        if isinstance(r, pd.DataFrame):
    
            return r.aggregate(PortfolioAnalytics.cvar_historic, level = level)
    
        raise TypeError("Expected r to be a Series or DataFrame")


    @staticmethod
    @lru_cache(maxsize = 256)
    def _hac_lags(
        n: int
    ) -> int:
        """
        Newey–West lag selection:

            maxlags = ⌊ n^{1/4} ⌋.

        Cached for small integers.
        """
        
        return max(1, int(np.floor(n ** 0.25)))
    

    @staticmethod
    def jensen_alpha_r2(
        port_rets: pd.Series,
        bench_rets: pd.Series,
        rf_per_period: float,
        periods_per_year: int,
        bench_ann_ret: float,
        port_ann_ret_pred: float,
        lever_beta: bool = False,
        d_to_e: float = 0.0,
        tax: float = 0.2,
        port_ann_ret_hist: float = None,
        scale_alpha_with_leverage: bool = False
    ) -> Dict[str, float]:
        """
        Jensen’s alpha with HAC covariance, plus annualisation, t-stats, R², and
        a CAPM-based predicted alpha.

        Model
        -----
            y_t = (r_p,t − r_f) = α_pp + β (r_b,t − r_f) + ε_t.

        Estimation is OLS with HAC (Newey–West, maxlags=⌊n^{1/4}⌋).

        Annualisation and SE
        --------------------
            α_ann = (1 + α_pp)^P − 1,
          
            se(α_ann) ≈ |∂α_ann/∂α_pp| · se(α_pp),   with  ∂α_ann/∂α_pp = P(1 + α_pp)^{P−1}.

        Predicted alpha (annual), relative to CAPM fair return:
          
            pred_α = μ_p,ann − [ R_f + β ( μ_b,ann − R_f ) ].

        Returns
        -------
        dict with keys:
      
        {"alpha_ann", "beta", "alpha_ann_se", "alpha_t", "beta_se", "beta_t", "r2", "pred_alpha"}.
        """ 

        df = pd.concat([port_rets, bench_rets], axis = 1)
       
        df.columns = ["p", "b"]
       
        df = df.dropna()
       
        y = (df["p"] - rf_per_period).to_numpy(copy = False)
       
        x = (df["b"] - rf_per_period).to_numpy(copy = False)
        
        try:
            
            X = sm.add_constant(x, has_constant = 'add')
            
        except:

            X = np.column_stack([np.ones(len(x)), x])
       
        model = sm.OLS(y, X)
       
        n = X.shape[0]
       
        if n < 5:
            
            return {
                "alpha_ann": np.nan, 
                "beta": np.nan, 
                "alpha_ann_se": np.nan, 
                "alpha_t": np.nan,
                "beta_se": np.nan,
                "beta_t": np.nan, 
                "r2": np.nan, 
                "pred_alpha": np.nan
            }
       
        lags = PortfolioAnalytics._hac_lags(n)
       
        res = model.fit(cov_type = "HAC", cov_kwds = {"maxlags": lags})

        alpha_pp = float(res.params[0])
       
        beta_u = float(res.params[1])

        alpha_se_pp = float(np.sqrt(res.cov_params()[0, 0]))
       
        beta_se_u = float(np.sqrt(res.cov_params()[1, 1]))
                
        if lever_beta and d_to_e > 0.0:
            
            L = 1.0 + (1.0 - tax) * d_to_e
            
            beta = beta_u * L
            
            beta_se = beta_se_u * abs(L)   
            
            if scale_alpha_with_leverage:
                
                alpha_pp *= L
                
                alpha_se_pp *= abs(L)
                
            alpha_ann = port_ann_ret_hist - (config.RF + beta * (bench_ann_ret - config.RF))
      
        else:
      
            beta = beta_u
      
            beta_se = beta_se_u
      
            alpha_ann = (1.0 + alpha_pp) ** periods_per_year - 1.0

        jac = periods_per_year * (1.0 + alpha_pp) ** (periods_per_year - 1.0)

        alpha_ann_se = abs(jac) * alpha_se_pp

        pred_alpha = port_ann_ret_pred - (config.RF + beta * (bench_ann_ret - config.RF))

        alpha_t = alpha_pp / (alpha_se_pp + 1e-16)

        beta_t = beta / (beta_se + 1e-16)

        return {
            "alpha_ann": float(alpha_ann),
            "beta": beta,
            "alpha_ann_se": float(alpha_ann_se),
            "alpha_t": float(alpha_t),
            "beta_se": float(beta_se),
            "beta_t": float(beta_t),
            "r2": float(res.rsquared),
            "pred_alpha": float(pred_alpha),
        }


    @staticmethod
    def alpha_beta_hac(
        port: pd.Series,
        bench: pd.Series,
        *,
        rf_per_period: float,
        periods_per_year: int
    ) -> dict[str, float]:
        """
        CAPM regression with HAC errors; reports annualised alpha and robust SE/t-stats.

        Alpha is compounded to annual:
           
            α_ann = (1 + α_pp)^P − 1.

        Returns a dictionary with alpha_ann, alpha_se (per-period), alpha_t, beta,
        beta_se, beta_t, and R².
        """   
        
        df = pd.concat({
            "p": port, 
            "b": bench
        }, axis = 1, join = "inner").dropna()
       
        if len(df) < 10:  
       
            return dict(
                alpha_ann = np.nan,
                alpha_se = np.nan, 
                alpha_t = np.nan,
                beta = np.nan,
                beta_se = np.nan, 
                beta_t = np.nan, 
                r2 = np.nan
            )

        y = df["p"] - rf_per_period
       
        X = sm.add_constant(df["b"] - rf_per_period)
       
        model = sm.OLS(y, X, hasconst = True).fit()

        lag = int(np.floor(len(df) ** 0.25))

        try:
            
            hac = model.get_robustcov_results(cov_type = "HAC", maxlags = lag)

        except Exception:

            hac = model.get_robustcov_results(cov_type = "HC1")  

        alpha_pp = float(hac.params["const"])
       
        alpha_ann = (1.0 + alpha_pp) ** periods_per_year - 1.0

        alpha_se = float(hac.bse["const"])
        
        alpha_t = float(hac.tvalues["const"])
        
        beta = float(hac.params[df.columns[1]])  
        
        beta_se = float(hac.bse[df.columns[1]])
        
        beta_t = float(hac.tvalues[df.columns[1]])

        return dict(
            alpha_ann = alpha_ann, 
            alpha_se = alpha_se,
            alpha_t = alpha_t,
            beta = beta,
            beta_se = beta_se, 
            beta_t = beta_t, 
            r2 = float(model.rsquared)
        )


    def _align_betas_to_universe(
        self,
        universe: pd.Index | list[str],
        beta_ext: pd.Series,
        beta_fallback: float = 1.0
    ) -> pd.Series:
        """
        Align an external beta Series to the optimiser's universe.
        Any missing tickers get a sensible fallback (default 1.0, or change to np.nan and drop).
        """
        
        uni = pd.Index([str(t).strip().upper() for t in universe])
        
        b = beta_ext.reindex(uni)

        b = b.fillna(beta_fallback)
        
        b.name = "Beta_Levered_Used"
        
        return b
    
    
    def estimate_alpha_with_external_beta(
        self,
        beta0: pd.Series,
        exp_ret: pd.Series,            
        market_exp_ret: float,           
        *,
        risk_free: float = 0.0,          
        treat_exp_ret_as_excess: bool = True,
        universe: pd.Index | list[str] | None = None,
        beta_fallback: float = 1.0
    ) -> pd.Series:
        """
        Compute alphas using external levered betas from coe2.py (Excel 'COE' sheet).
        All inputs must be in the SAME return convention and horizon (e.g., annual).
   
        - If treat_exp_ret_as_excess=True, exp_ret and market_exp_ret are excess returns.
   
        - Otherwise they are total returns; alpha formula adjusts for Rf.
   
        """

        if universe is None:

            universe = exp_ret.index

        beta = self._align_betas_to_universe(
            universe = universe,
            beta0 = beta0, 
            beta_fallback = beta_fallback
        )

        exp_ret = exp_ret.reindex(beta.index)

        if treat_exp_ret_as_excess:

            alpha = exp_ret - beta * market_exp_ret
    
        else:

            alpha = exp_ret - beta * market_exp_ret - (1.0 - beta) * risk_free

        alpha.name = "Alpha"

        return alpha


    @staticmethod
    def port_pred_cvar(
        r_pred: float,
        std_pred: float,
        skew: float,
        kurt: float,
        level: float = 5.0,
        periods: int = 52,
    ) -> float:
        """
        Predicted one-period Expected Shortfall (left tail) from annual inputs using
        Cornish–Fisher quantile and normal ES formula.

        Steps
        -----
        
        1) Convert annual mean/vol to per-period:
        
            μ_pp = (1 + r_pred)^{1 / P} − 1,   σ_pp = std_pred / √P.
        
        2) α = level / 100, z = Φ^{-1}(α), build z_cf via Cornish–Fisher (skew/kurt).
      
        3) Normal ES formula (left tail):
      
            ES_α = μ_pp − σ_pp · φ(z_cf)/α.
      
        4) Return −ES_α as a positive loss number.

        Here φ is the standard normal pdf.
        """
        
        alpha = level / 100.0
       
        z = norm.ppf(alpha)
       
        z_cf = (z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * (kurt - 3) / 24 - (2 * z ** 3 - 5 * z) * (skew ** 2) / 36)
       
        mu_pp = (1.0 + r_pred) ** (1.0 / periods) - 1.0
       
        sigma_pp = std_pred / np.sqrt(periods)
       
        es_left = mu_pp - sigma_pp * norm.pdf(z_cf) / alpha
       
        return float(-es_left) 
    

    @staticmethod
    def IR(
        w: np.ndarray,
        er: pd.DataFrame | pd.Series,
        te: float | None,
        benchmark_ret: pd.Series,
        port_series: pd.Series | None = None,
        ann_hist_ret: float | None = None,
        ann_hist_bench_ret: float | None = None,
        periods_per_year: int = 52,
    ) -> float:
        """
        Information ratio:

            IR = ( μ_p,ann − μ_b,ann ) / TE.

        If TE is not supplied it is computed from `port_series` vs `benchmark_ret`.
      
        If annualised returns are not supplied they are computed from series using
        geometric annualisation with `periods_per_year`.
        """
        
        if te is None:
      
            if port_series is None:
      
                port_series = PortfolioAnalytics.portfolio_return_robust(
                    weights = w,
                    returns = er
                )
      
            te = PortfolioAnalytics.tracking_error(
                r_a = port_series, 
                r_b = benchmark_ret
            )
      
        te = max(float(te), 1e-12)

        if ann_hist_ret is None:
          
            if port_series is None:
          
                port_series = PortfolioAnalytics.portfolio_return_robust(
                    weights = w, 
                    returns = er
                )  
          
            ann_hist_ret = PortfolioAnalytics.annualise_returns(
                ret_series = port_series,
                periods_per_year = periods_per_year
            )

        if ann_hist_bench_ret is None:
          
            ann_hist_bench_ret = PortfolioAnalytics.annualise_returns(
                ret_series = benchmark_ret,
                periods_per_year = periods_per_year
            )

        return float((ann_hist_ret - ann_hist_bench_ret) / te)

   
    @staticmethod
    def ulcer_index(
        return_series: pd.Series, 
        dd: pd.Series | None = None
    ) -> float:
        """
        Ulcer Index (UI):

            UI = √( mean( DD_t^2 ) ),

        where DD_t is the drawdown path. If `dd` is provided, it is used directly.
        """
        
        if dd is None:
    
            wealth = (1 + return_series).cumprod()
    
            peak = wealth.cummax()
    
            dd = (wealth - peak) / peak
    
        return float(np.sqrt((dd ** 2).mean()))


    @staticmethod
    def cdar(
        r: pd.Series, 
        level: float = 5.0, 
        dd: pd.Series | None = None
    ) -> float:
        """
        Conditional Drawdown at Risk at α=level%:

            CDaR_α = mean( DD_t | DD_t ≤ q_α(DD) ).

        Returns the (negative) mean drawdown in the worst α tail. If no tail points
        exist, returns 0.0.
        """
        
        if dd is None:
    
            wealth = (1 + r).cumprod()
    
            peak = wealth.cummax()
    
            dd = (wealth - peak) / peak
    
        thresh = dd.quantile(level / 100.0)
    
        worst = dd[dd <= thresh]

        if not worst.empty:
            
            return float(worst.mean())  
        
        else: 
            
            return 0.0


    @staticmethod
    def probabilistic_sharpe_ratio(
        sr: float, 
        sr_star: float, 
        T: int, 
        skew: float, 
        kurt: float
    ) -> float:
        """
        Probabilistic Sharpe Ratio (PSR), following the finite-sample, non-normal
        adjustment:

            PSR = Φ( (SR − SR*) √(T − 1) / √( 1 − γ₁ SR + ((κ − 1) / 4) SR² ) ),

        where γ₁ is skewness, κ is kurtosis (raw), and T is the number of observations.
        """
        
        if T <= 1 or not np.isfinite(sr):
            
            return np.nan
        
        z_denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4.0) * (sr ** 2))
        
        if z_denom <= 0: 
            
            return np.nan
        
        z = (sr - sr_star) * np.sqrt(T - 1) / z_denom
        
        return float(norm.cdf(z))


    @staticmethod
    def deflated_sharpe_ratio(
        sr: float, 
        T: int,
        skew: float, 
        kurt: float,
        N: int = 1, 
        sr_max: float = 0.0
    ) -> float:
        """
        Deflated Sharpe Ratio (approximation via PSR).

        Interprets `sr_max` as the benchmark Sharpe SR* (e.g., expected maximum SR
        across N trials). Returns:

            DSR ≈ PSR(SR, SR*, T, skew, kurt).

        The input `N` is not used directly here; it is retained for API compatibility.
        """
        
        return PortfolioAnalytics.probabilistic_sharpe_ratio(
            sr = sr,
            sr_star = sr_max, 
            T = T, 
            skew = skew, 
            kurt = kurt
        )


    @staticmethod
    def capture_slopes(
        port_rets: pd.Series,
        bench_rets: pd.Series
    ) -> Dict[str, float]:
        """
        Upside/Downside Capture via conditional OLS slopes.

        Let β_up be the slope from OLS of p on b for periods with b > 0, and β_down for b < 0.
       
        Returns {"Upside Capture": β_up, "Downside Capture": β_down}.
        """
        
        df = pd.concat([port_rets, bench_rets], axis = 1).dropna()
    
        df.columns = ["p", "b"]

    
        def slope(
            y, 
            x
        ):
        
            if len(x) < 3:
        
                return np.nan
        
            X = sm.add_constant(x)
        
            return float(sm.OLS(y, X).fit().params["b"])

        
        up = df[df["b"] > 0]
        
        down = df[df["b"] < 0]
        
        return {
            "Upside Capture": slope(
                y = up["p"], 
                x = up["b"]
            ), 
            "Downside Capture": slope(
                y = down["p"],
                x = down["b"]
            )
        }


    @staticmethod
    def capture_ratios(
        port_rets: pd.Series, 
        bench_rets: pd.Series
    ) -> Dict[str, float]:
        """
        Upside/Downside Capture ratios via mean returns.

            UC = mean(r_p | r_b > 0) / mean(r_b | r_b > 0),
          
            DC = mean(r_p | r_b < 0) / mean(r_b | r_b < 0).

        Handles empty subsets by returning NaN for the respective ratio.
        """
       
        df = pd.concat([port_rets, bench_rets], axis = 1).dropna()
    
        df.columns = ["p", "b"]
    
        up = df[df["b"] > 0]
    
        down = df[df["b"] < 0]
        
        up_b = up["b"].mean()
        
        down_b = down["b"].mean()
        
        
        if not up.empty and abs(up_b) > 1e-12:
            
            up_cap = float(up["p"].mean() / up_b)  
        
        else:
            
            up_cap = np.nan
        
        
        if not down.empty and abs(down_b) > 1e-12:
            
            down_cap = float(down["p"].mean() / down_b)  
        
        else:
            
            down_cap = np.nan
    

        caps = {
            "Upside Capture": up_cap, 
            "Downside Capture": down_cap
        }
        
        return caps
    
    
    @staticmethod
    def portfolio_return_robust(
        weights: np.ndarray,
        returns: pd.DataFrame | pd.Series,
        *,
        renormalize: bool = True,        
        min_coverage: float | None = None,   
        mask_nonfinite: bool = False         
    ) -> pd.Series:
        """
        Row-wise portfolio return with per-row renormalisation and missing-data masking.

        For row t with observed mask m_i(t)∈{0,1}:
        
            numerator_t   = ∑_i w_i m_i(t) r_{it},
        
            denominator_t = ∑_i w_i m_i(t).

        If `renormalize=True`, returns
        
            r_p(t) = numerator_t / denominator_t         when denominator_t ≠ 0;
        
        otherwise returns the unnormalised numerator.

        If `min_coverage` is set, a row is kept only if
        
            |denominator_t / ∑_i w_i| ≥ min_coverage.

        Non-finite entries are masked as missing when `mask_nonfinite=True`, otherwise
        only NaNs are treated as missing.
        """
       
        w = np.asarray(weights, float).reshape(-1)

        if isinstance(returns, pd.Series):
        
            R = returns.to_frame() 
        
        else:
            
            R = returns
        
        V = R.to_numpy(copy = False, dtype = float)
        
        if mask_nonfinite:

            miss = ~np.isfinite(V) 
        
        else:
            
            miss = np.isnan(V)
        
        valid = ~miss

        V0 = np.where(valid, V, 0.0)

        num = V0 @ w               
           
        denom = valid.astype(float) @ w 

        out = np.full(V.shape[0], np.nan, dtype = float)

        has_any = valid.any(axis = 1)
        
        safe = has_any & (np.abs(denom) > 1e-12)
        
        if min_coverage is not None:
      
            total_w = np.sum(w) + 1e-12
      
            cov = denom / total_w
            
            safe &= (np.abs(cov) >= float(min_coverage))

        if renormalize:
      
            out[safe] = num[safe] / denom[safe] 
      
        else:
            
            out[safe] = num[safe]
                
        return pd.Series(out, index = R.index, name = "portfolio_return")


    @staticmethod
    def portfolio_return_robust_batch(
        W: np.ndarray,
        returns: pd.DataFrame,
        *,
        renormalize: bool = True,
        min_coverage: float | None = None,
        mask_nonfinite: bool = False
    ) -> pd.DataFrame:
        """
        Batch version for K portfolios with weight matrix W ∈ ℝ^{N × K}.

        For each time t and portfolio k:
     
            num_{t,k}   = ∑_i m_i(t) r_{it} w_{ik},
     
            denom_{t,k} = ∑_i m_i(t) w_{ik},

        and (if renormalize) r_{t,k} = num_{t,k} / denom_{t,k} when |denom_{t,k}| > 0.
        
        Rows with insufficient coverage (if provided) are set to NaN.
        """   
     
        V = returns.to_numpy(copy = False, dtype = float)
       
        W = np.asarray(W, float)

        if mask_nonfinite:
        
            miss = (~np.isfinite(V))  
        
        else:
            
            miss = np.isnan(V)
        
        valid = (~miss).astype(float)

        V0 = np.where(miss, 0.0, V)
      
        num = V0 @ W               
      
        denom = valid @ W            

        out = np.full_like(num, np.nan, float)

        has_any = (~miss).any(axis=1)[:, None]
        
        safe = has_any & (np.abs(denom) > 1e-12)

        if min_coverage is not None:
        
            total_w = np.sum(W, axis = 0, keepdims = True) + 1e-12
          
            cov = denom / total_w
          
            safe &= (np.abs(cov) >= float(min_coverage))

        if renormalize:
        
            out[safe] = num[safe] / denom[safe]
        
        else:
        
            out[safe] = num[safe]

        return pd.DataFrame(out, index=returns.index)


    @staticmethod
    def _edgeworth_eps(
        n: int,
        skew: float,
        kurt: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Edgeworth expansion to inject skewness and kurtosis into standardised shocks.

        Let 
        
            Z ~ N(0,1), 
            
            g₁ = skew, 
            
            g₂ = kurt − 3 (excess kurtosis). 
            
        Construct

            ε = Z + (g₁ / 6)(Z² − 1) + (g₂ / 24)(Z³ − 3Z) − (g₁² / 36)(2Z³ − 5Z),

        then rescale to zero mean and unit variance. Returns ε ∈ ℝ^n.
        """
        
        Z = rng.standard_normal(n).astype(np.float32)
    
        g1 = float(skew)
    
        g2 = float(kurt) - 3.0
    
        eps = (Z + (g1 / 6.0) * (Z ** 2 - 1.0) + (g2 / 24.0) * (Z ** 3 - 3.0 * Z) - ((g1 ** 2) / 36.0) * (2.0 * Z ** 3 - 5.0 * Z))
    
        eps = (eps - eps.mean()) / (eps.std() + 1e-12)
    
        return eps


    @staticmethod
    def gbm_final_non_gaussian(
        n_years: int = 10,
        n_scenarios: int = 1_000_000,
        mu: float = 0.07,
        sigma: float = 0.15,
        steps_per_year: int = 12,
        s_0: float = 100.0,
        skew: float = 0.0,
        kurt: float = 3.0,
        method: str = "edgeworth", 
        qmc_mode: bool = False,   
        dtype=np.float32,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Final-price simulation under log dynamics with non-Gaussian increments and optional QMC.

        Dynamics
        --------
      
        Let Δt = 1/steps_per_year, and define
      
            b = σ √Δt,  ε_t ~ N(0,1) or Edgeworth( skew, kurt ).

        The log-price evolves as
      
            log S_{t+1} = log S_t + a + b ε_t.

        Drift calibration
        -----------------
      
        Choose a per step such that E[exp(a + b ε)] = G_target, with
      
            G_target = (1 + μ)^{1 / steps_per_year}.

        • If ε ~ N(0,1): 
        
            E[exp(b ε)] = exp(½ b²) ⇒ a = log G_target − ½ b².
      
        • If ε is Edgeworth: E[exp(b ε)] is estimated via a pilot simulation,
        then  a = log G_target − log E[exp(b ε)].

        QMC
        ---
      
        If `qmc_mode = True`, standard normals are produced by Sobol points U ∼ U(0,1)
        mapped via z = Φ^{-1}(U).

        Returns
        -------
        np.ndarray
            Final prices S_T for `n_scenarios` paths (float64).
        """
       
        dt = 1.0 / steps_per_year
       
        n_steps = int(n_years * steps_per_year)
       
        rng = np.random.default_rng(random_state)

        b = sigma * np.sqrt(dt)
       
        G_target = (1.0 + mu) ** (1.0 / steps_per_year)

        if method == "gaussian":

            a = np.log(G_target) - 0.5 * (b * b)

        else:

            pilot = 200_000
            
            eps = PortfolioAnalytics._edgeworth_eps(
                n = pilot, 
                skew = skew,
                kurt = kurt, 
                rng = rng
            )
            
            exp_beps = np.exp(b * eps).mean()
            
            a = np.log(G_target) - np.log(exp_beps + 1e-15)

        log_s = np.full(n_scenarios, np.log(s_0), dtype = dtype)

        for step in range(n_steps):
        
            if qmc_mode:
        
                engine = qmc.Sobol(d=1, scramble=True, seed=(None if random_state is None else random_state + step))
        
                u = engine.random(n_scenarios).reshape(-1)
        
                z = norm.ppf(u).astype(dtype)
        
            else:
        
                z = rng.standard_normal(n_scenarios).astype(dtype)

            if method == "edgeworth":

                g1 = float(skew)
                
                g2 = float(kurt) - 3.0
               
                eps = (z + (g1 / 6.0) * (z ** 2 - 1.0) + (g2 / 24.0) * (z ** 3 - 3.0 * z) - ((g1 ** 2) / 36.0) * (2.0 * z ** 3 - 5.0 * z))
               
                eps = (eps - eps.mean()) / (eps.std() + 1e-12)
         
            else:
         
                eps = z

            log_s += (a + b * eps).astype(dtype)

        return np.exp(log_s, dtype = np.float64)


    @staticmethod
    def simulate_portfolio_stats(
        mu: float,
        sigma: float,
        steps: int = 252,
        s0: float = 100.0,
        scenarios: int = 100_000,
        skew: float = 0.0,
        kurt: float = 3.0,
        method: str = "edgeworth",
        qmc_mode: bool = False,
        random_state: int | None = 42,
    ) -> Dict[str, Any]:
        """
        One-year final-price Monte Carlo summary from annual μ, σ and higher moments.

        Uses `gbm_final_non_gaussian` with n_years=1 and steps=steps. Converts prices
        to returns r = S_T / S_0 − 1 and reports:

        • mean_returns = E[r],
      
        • loss_percentage = 100·Pr(r < 0),
       
        • mean_loss_amount = E[r | r < 0],
       
        • mean_gain_amount = E[r | r ≥ 0],
       
        • variance = Var[r],
       
        • deciles: 10th/90th percentiles,
       
        • quartile bands: means within narrow bands around Q1/Q3,
       
        • scenarios_up_down = p90/p10 (ratio of positive deciles),
       
        • upper_returns_mean = mean of r | r ≥ Q3,
       
        • min_return, max_return.

        Returns a dict of scalars.
        """
        
        final_prices = PortfolioAnalytics.gbm_final_non_gaussian(
            n_years = 1,
            n_scenarios = scenarios,
            mu = mu,
            sigma = sigma,
            steps_per_year = steps,
            s_0 = s0,
            skew = skew,
            kurt = kurt,
            method = method,
            qmc_mode = qmc_mode,
            dtype = np.float32,
            random_state = random_state,
        )
        
        r = final_prices / s0 - 1.0

        q = np.quantile(r, [0.10, 0.245, 0.255, 0.745, 0.755, 0.75, 0.90])

        p10, q25_l, q25_h, q75_l, q75_h, q75, p90 = map(float, q)

        lower_quart = float(r[(r >= q25_l) & (r <= q25_h)].mean())

        upper_quart = float(r[(r >= q75_l) & (r <= q75_h)].mean())

        upper_mean = float(r[r >= q75].mean())

        return {
            "mean_returns": float(r.mean()),
            "loss_percentage": float(100.0 * (r < 0).mean()),
            "mean_loss_amount": float(r[r < 0].mean()),
            "mean_gain_amount": float(r[r >= 0].mean()),
            "variance": float(r.var()),
            "10th_percentile": p10,
            "lower_quartile": lower_quart,
            "upper_quartile": upper_quart,
            "90th_percentile": p90,
            "scenarios_up_down": float((p90 / p10) if p10 != 0 else np.inf),
            "upper_returns_mean": upper_mean,
            "min_return": float(final_prices.min() / s0) - 1.0,
            "max_return": float(final_prices.max() / s0) - 1.0,
        }


    def _df_key(
        self,
        df: pd.DataFrame
    ) -> Any:
        """
        Content fingerprint for DataFrames (used when cache_key='content').

        The key aggregates: shape, size, (nan)mean, (nan)stdev of values, and a
        coarse index/column signature. It is not a cryptographic hash but robust for
        memoisation within a session.
        """
        
        if self._cache_key_mode == "identity":
    
            return id(df)
    
    
        vals = df.to_numpy(copy=False)
       
        h = (
            vals.size,
            vals.shape,
            float(np.nanmean(vals)) if vals.size else 0.0,
            float(np.nanstd(vals)) if vals.size else 0.0,
        )
        
        return (tuple(df.columns), tuple(df.index[:1]) + tuple(df.index[-1:]), h)


    def _maybe_cached_port_series(
        self,
        weights: np.ndarray, 
        rets_df: pd.DataFrame
    ) -> pd.Series:
        """
        Return the portfolio series for (weights, rets_df), using the active cache
        strategy if caching is enabled, else compute via `portfolio_return_robust`.
        """
        
        if not self._cache_enabled:
    
            return self.portfolio_return_robust(
                weights = weights, 
                returns = rets_df
            )

        wkey = np.asarray(weights, float).tobytes()
      
        if self._cache_key_mode == "identity":
      
            sub = self._port_series_cache_id.get(rets_df)
      
            if sub is None:
      
                sub = LRUCache(
                    capacity = self._cache_maxsize
                )
      
                self._port_series_cache_id[rets_df] = sub
      
            s = sub.get(wkey)
           
            if s is None:
           
                s = self.portfolio_return_robust(
                    weights = weights,
                    returns = rets_df
                )
           
                sub.set(wkey, s)
           
            return s
        
        else:
            
            key = (self._df_key(df = rets_df), wkey)
           
            s = self._port_series_cache_content.get(key)
           
            if s is None:
           
                s = self.portfolio_return_robust(
                    weights = weights,
                    returns = rets_df
                )
           
                self._port_series_cache_content.set(key, s)
           
            return s


    def _maybe_cached_drawdown_series(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Return the drawdown series for `series`, memoised by object identity when
        caching is enabled. Otherwise computes `drawdown(series)["Drawdown"]`.
        """

        if not self._cache_enabled:
    
            return self.drawdown(
                return_series = series
            )["Drawdown"]
    
        sub = self._dd_cache_id.get(series)
    
        if sub is None:
    
            sub = self.drawdown(
                return_series = series
            )["Drawdown"]
    
            self._dd_cache_id[series] = sub
    
        return sub


    def _maybe_cached_ann_ret(
        self,
        series: pd.Series, 
        periods: int
    ) -> float:
        """
        Return the annualised return for `series` at frequency `periods`, memoised
        per (series, periods) when caching is enabled.
        """
        
        if not self._cache_enabled:
    
            return self.annualise_returns(
                ret_series = series, 
                periods_per_year = periods
            )
    
        sub = self._annret_cache_id.get(series)
    
        if sub is None:
    
            sub = {}
    
            self._annret_cache_id[series] = sub
    
        if periods not in sub:
    
            sub[periods] = float(self.annualise_returns(
                ret_series = series,
                periods_per_year = periods
            ))
    
        return float(sub[periods])


    def _maybe_cached_ann_vol(
        self, 
        series: pd.Series,
        periods: int
    ) -> float:
        """
        Return the annualised volatility for `series` at frequency `periods`, memoised
        per (series, periods) when caching is enabled.
        """
        
        if not self._cache_enabled:
    
            return self.annualise_vol(
                r = series, 
                periods_per_year = periods
            )
    
        sub = self._annvol_cache_id.get(series)
    
        if sub is None:
    
            sub = {}
    
            self._annvol_cache_id[series] = sub
    
        if periods not in sub:
    
            sub[periods] = float(self.annualise_vol(
                r = series,
                periods_per_year = periods
            ))
    
        return float(sub[periods])


    def simulate_and_report(
        self,
        *,
        name: str,
        wts: np.ndarray,
        comb_rets: pd.Series,
        bear_rets: pd.Series,
        bull_rets: pd.Series,
        vol: float,
        vol_ann: float,
        comb_score: pd.Series,
        last_year_weekly_rets: pd.DataFrame,
        last_5y_weekly_rets: pd.DataFrame,
        n_last_year_weeks: int,
        rf: float,
        beta: pd.Series | np.ndarray,
        benchmark_weekly_rets: pd.Series,
        benchmark_ann_ret: float,
        bl_ret: pd.Series,
        bl_cov: pd.DataFrame | np.ndarray,
        tax: pd.Series,
        d_to_e: pd.Series,
        sims: int = 1_000_000,
        n_trials: int = 1,
        qmc_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute a comprehensive set of portfolio diagnostics and scenario statistics.

        The routine:
      
        1) Builds cached 1y / 5y portfolio series via masked dot with per-row renormalisation.
       
        2) Computes annualised historical return/volatility from 1y series.
       
        3) Computes point forecasts (average returns), Black–Litterman counterparts,
        betas, Treynor, scores, Sharpe (predicted, BL, historical), drawdowns, and
        distribution moments (skew/kurt).
       
        4) Computes tail metrics: Cornish–Fisher VaR, historical CVaR, predicted ES.
       
        5) Computes TE and IR versus the benchmark.
       
        6) Computes Ulcer Index, CDaR, Sortino (pred and hist), Calmar, Omega, M²,
        Pain index/ratio, Tail ratio, RAROC, win/loss streaks.
       
        7) Runs CAPM with HAC to obtain annualised alpha, SE/t-stats, R², and a CAPM
        predicted alpha.
       
        8) Computes capture slopes and capture ratios.
       
        9) Computes PSR and DSR approximations from SR, T, skew, kurt.
       
        10) Runs a one-year non-Gaussian Monte Carlo for scenario summaries.

        Returns
        -------
        Dict[str, Any]
            A dictionary of scalar metrics covering return, risk, drawdown, tail,
            CAPM diagnostics, captures, PSR/DSR, and Monte Carlo summaries.
        """
      
        port_1y = self._maybe_cached_port_series(
            weights = wts, 
            rets_df = last_year_weekly_rets
        )
        
        port_5y = self._maybe_cached_port_series(
            weights = wts, 
            rets_df = last_5y_weekly_rets
        )

        ann_hist_ret = self._maybe_cached_ann_ret(
            series = port_1y,
            periods = n_last_year_weeks
        )
        
        ann_hist_vol = self._maybe_cached_ann_vol(
            series = port_1y, 
            periods = n_last_year_weeks
        )

        port_rets = self.portfolio_return(
            weights = wts,
            returns = comb_rets
        )
        
        port_bear_rets = self.portfolio_return(
            weights = wts,
            returns = bear_rets
        )
        
        port_bull_rets = self.portfolio_return(
            weights = wts,
            returns = bull_rets
        )
        
        port_bl_rets = self.portfolio_return(
            weights = wts,
            returns = bl_ret
        )

        port_bl_vol = self.portfolio_volatility(
            weights = wts,
            covmat = np.asarray(bl_cov, float)
        )
        
        score_val = self.port_score(
            weights = wts,
            score = pd.Series(comb_score)
        )

        sr_pred = self.sharpe_ratio(
            r = port_1y,
            periods_per_year = n_last_year_weeks,
            ann_ret = port_rets,
            ann_vol = vol_ann
        )  
        
        bl_sr = self.sharpe_ratio(
            r = port_1y,
            periods_per_year = n_last_year_weeks,
            ann_ret = port_bl_rets,
            ann_vol = port_bl_vol
        )
        
        sr_hist = self.sharpe_ratio(
            r = port_1y,
            periods_per_year = n_last_year_weeks,
            ann_ret = ann_hist_ret,
            ann_vol = ann_hist_vol
        )

        dd_1y = self._maybe_cached_drawdown_series(
            series = port_1y
        )
        
        dd_max = float(dd_1y.min())
        
        dd_5y = self._maybe_cached_drawdown_series(
            series = port_5y
        )

        skew_val = float(self.skewness(
            r = port_5y
        ))
       
        kurt_val = float(self.kurtosis(
            r = port_5y
        ))

        cf_var5 = self.var_gaussian(
            r = port_5y,
            s = skew_val, 
            k = kurt_val, 
            level = 5.0, 
            modified = True
        )
       
        hist_cvar5 = self.cvar_historic(
            r = port_5y, 
            level = 5.0
        )
       
        pred_cvar = self.port_pred_cvar(
            r_pred = port_rets,
            std_pred = vol_ann,
            skew = skew_val,
            kurt = kurt_val, 
            level = 5.0,
            periods = 52
        )

        te = self.tracking_error(
            r_a = benchmark_weekly_rets,
            r_b = port_5y
        )
       
        ir = self.IR(
            w = wts,
            er = last_5y_weekly_rets,
            te = te,
            benchmark_ret = benchmark_weekly_rets,
            port_series = port_1y,
            ann_hist_ret = ann_hist_ret,
            ann_hist_bench_ret = benchmark_ann_ret,
            periods_per_year = n_last_year_weeks,
        )

        ui = self.ulcer_index(
            return_series = port_5y, 
            dd = dd_5y
        )
        
        cd = self.cdar(
            r = port_5y, 
            dd = dd_5y
        )
        
        sortino = self.sortino_ratio(
            returns = port_1y, 
            riskfree_rate = rf,
            periods_per_year = 52,
            target = config.RF_PER_WEEK, 
            er = port_rets
        )
        
        sortino_hist = self.sortino_ratio(
            returns = port_1y, 
            riskfree_rate = rf,
            periods_per_year = n_last_year_weeks
        )
        
        calmar = self.calmar_ratio(
            returns = port_5y, 
            periods_per_year = n_last_year_weeks,
            ann_hist_ret = ann_hist_ret,
            max_dd = dd_max
        )
        
        omega = self.omega_ratio(
            returns = port_5y
        )
        
        m2 = self.modigliani_ratio(
            returns = port_5y,
            bench_returns = benchmark_weekly_rets,
            riskfree_rate = rf, 
            periods_per_year = n_last_year_weeks, 
            sr = sr_hist
        )
        
        pi, pr = self.pain_index_and_ratio(
            returns = port_5y, 
            riskfree_rate = rf,
            periods_per_year = n_last_year_weeks,
            dd = dd_5y,
            cagr = ann_hist_ret
        )
       
        tail = self.tail_ratio(
            returns = port_5y
        )
       
        raroc_val = self.raroc(
            returns = port_5y, 
            riskfree_rate = rf,
            periods_per_year = n_last_year_weeks,
            ann_return = ann_hist_ret
        )
       
        pct_pos, win_streak, loss_streak = self.percent_positive_and_streaks(
            returns = port_5y
        )

        hac = self.jensen_alpha_r2(
            port_rets = port_1y,
            bench_rets = benchmark_weekly_rets,
            rf_per_period = config.RF_PER_WEEK,
            periods_per_year = 52,
            bench_ann_ret = benchmark_ann_ret,
            port_ann_ret_pred = port_rets,
            lever_beta = True,
            d_to_e = d_to_e,
            tax = tax,
            port_ann_ret_hist = ann_hist_ret,
            scale_alpha_with_leverage = True,
        )
        
        beta_i = hac['beta']
            
        treynor = self.compute_treynor_ratio(
            port_ret = port_rets,
            rf = rf,
            port_beta_val = beta_i
        )        

        caps = self.capture_slopes(
            port_rets = port_1y,
            bench_rets = benchmark_weekly_rets
        )
        
        caps_ratios = self.capture_ratios(
            port_rets = port_5y,
            bench_rets = benchmark_weekly_rets
        )
        
        n_obs = len(port_1y.dropna())

        if isinstance(sr_hist, (float, np.floating)):
            
            sr_sample = float(sr_hist)  
        
        else:
            
            sr_sample = float("nan")

        if np.isfinite(sr_sample):
            
            psr = self.probabilistic_sharpe_ratio(
                sr = sr_sample,
                sr_star = 0.0, 
                T = n_obs, 
                skew = skew_val, 
                kurt = kurt_val
            )  
        
        else:
            
            psr = np.nan

        if np.isfinite(sr_sample):
            
            dsr = self.deflated_sharpe_ratio(
                sr = sr_sample, 
                T = n_obs, 
                skew = skew_val, 
                kurt = kurt_val,
            )  
            
        else:
            
            dsr = np.nan

        stats = self.simulate_portfolio_stats(
            mu = port_rets,
            sigma = vol_ann,
            steps = 252,
            s0 = 100.0,
            scenarios = sims,
            skew = skew_val,
            kurt = kurt_val,
            method = "edgeworth",
            qmc_mode = qmc_mode,
            random_state = 42,
        )

        out = {
            "Average Returns": port_rets,
            "Average Bear Returns": port_bear_rets,
            "Average Bull Returns": port_bull_rets,
            "BL Returns": port_bl_rets,
            "Weekly Volatility": vol,
            "Annual Volatility": vol_ann,
            "BL Volatility": port_bl_vol,
            "Scenario Average Returns": stats["mean_returns"],
            "Scenario Loss Incurred": stats["loss_percentage"],
            "Scenario Average Loss": stats["mean_loss_amount"],
            "Scenario Average Gain": stats["mean_gain_amount"],
            "Scenario Variance": stats["variance"],
            "Scenario 10th Percentile": stats["10th_percentile"],
            "Scenario Lower Quartile": stats["lower_quartile"],
            "Scenario Upper Quartile": stats["upper_quartile"],
            "Scenario 90th Percentile": stats["90th_percentile"],
            "Scenario Up/Down": stats["scenarios_up_down"],
            "Scenario Min Returns": stats["min_return"],
            "Scenario Max Returns": stats["max_return"],
            "Portfolio Beta": beta_i,
            "Treynor Ratio": treynor,
            "Portfolio Score": score_val,
            "Portfolio Tracking Error": te,
            "Information Ratio": ir,
            "Sortino Ratio": sortino,
            "Sortino Ratio (Historical)": sortino_hist,
            "Calmar Ratio": calmar,
            "Omega Ratio": omega,
            "M2 (Modigliani)": m2,
            "Pain Index": pi,
            "Pain Ratio": pr,
            "Tail Ratio": tail,
            "RAROC": raroc_val,
            "Percent Positive Periods": pct_pos,
            "Max Win Streak": win_streak,
            "Max Loss Streak": loss_streak,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "Cornish-Fisher VaR (5%)": cf_var5,
            "Historic CVaR (5%)": float(hist_cvar5),
            "Predicted CVaR (5%)": pred_cvar,
            "Sharpe Ratio (Predicted)": sr_pred,
            "Sharpe Hist Ratio": sr_hist,
            "PSR (SR* = 0)": psr,
            "DSR (approx)": dsr,
            "Bl Sharpe Ratio": bl_sr,
            "Historic Annual Returns": ann_hist_ret,
            "Max Drawdown": dd_max,
            "Ulcer Index": ui,
            "Conditional Drawdown at Risk": cd,
            "Jensen's Alpha (ann)": hac["alpha_ann"],
            "Alpha ann SE (HAC)": hac["alpha_ann_se"],
            "Alpha t (HAC)": hac["alpha_t"],
            "Beta (HAC)": hac["beta"],
            "Beta SE (HAC)": hac["beta_se"],
            "Beta t (HAC)": hac["beta_t"],
            "R-squared": hac["r2"],
            "Predicted Alpha": hac["pred_alpha"],
            "Upside Capture Ratio": caps.get("Upside Capture", np.nan),
            "Downside Capture Ratio": caps.get("Downside Capture", np.nan),
            "Upside Capture (Mean)": caps_ratios.get("Upside Capture", np.nan),
            "Downside Capture (Mean)": caps_ratios.get("Downside Capture", np.nan),
        }
        
        return out


    def report_ticker_metrics(
        self,
        *,
        tickers: List[str],
        last_year_weekly_rets: pd.DataFrame,
        last_5y_weekly_rets: pd.DataFrame,
        n_last_year_weeks: int,
        weekly_cov: pd.DataFrame,
        ann_cov: pd.DataFrame,
        comb_rets: pd.Series,
        bear_rets: pd.Series,
        bull_rets: pd.Series,
        comb_score: pd.Series,
        rf: float,
        beta: pd.Series,
        benchmark_weekly_rets: pd.Series,
        benchmark_ann_ret: float,
        bl_ret: pd.Series,
        bl_cov: pd.DataFrame,
        tax: pd.Series,
        d_to_e: pd.Series,
        forecast_file: str,
        sims: int = 10_000,
        n_trials: int = 1,
        qmc_mode: bool = False,
    ) -> pd.DataFrame:
        """
        Per-ticker diagnostics (single-asset portfolios with weight 1).

        Steps
        -----
      
        1) Normalise tickers to upper case; align and reindex all inputs.
      
        2) Load model return sheets from the forecast workbook to produce a joined table
        of model returns.
      
        3) For each ticker, assemble 1y/5y weekly series, volatilities, BL inputs, beta,
        and score, then call `simulate_and_report`.
      
        4) Combine model returns and diagnostics into a final DataFrame indexed by ticker.
        """
      
        tickers = [t.upper() for t in tickers]


        def _up(
            s: pd.Series
        ) -> pd.Series:
        
            s = s.copy()
        
            s.index = s.index.str.upper()
        
            return s


        def _up_df(
            df: pd.DataFrame
        ) -> pd.DataFrame:
        
            df = df.copy()
        
            df.columns = df.columns.str.upper()
        
            return df


        last_year_weekly_rets = _up_df(
            df = last_year_weekly_rets
        ).reindex(columns = tickers)
        
        last_5y_weekly_rets = _up_df(
            df = last_5y_weekly_rets
        ).reindex(columns = tickers)
        
        weekly_cov = _up_df(
            df = weekly_cov
        ).reindex(index = tickers, columns = tickers)
        
        ann_cov = _up_df(
            df = ann_cov
        ).reindex(index = tickers, columns = tickers)
        
        bl_cov = _up_df(
            df = bl_cov
        ).reindex(index = tickers, columns = tickers)

        comb_rets = _up(
            s = comb_rets
        ).reindex(tickers)
        
        bear_rets = _up(
            s = bear_rets
        ).reindex(tickers)
        
        bull_rets = _up(
            s = bull_rets
        ).reindex(tickers)
        
        comb_score = _up(
            s = comb_score
        ).reindex(tickers)
        
        beta = _up(
            s = beta
        ).reindex(tickers)
        
        bl_ret = _up(
            s = bl_ret
        ).reindex(tickers)
        
        tax = _up(
            s = tax
        ).reindex(tickers)
        
        d_to_e = _up(
            s = d_to_e
        ).reindex(tickers)

        if len(set(tickers)) != len(tickers):
           
            dup = pd.Index(tickers)[pd.Index(tickers).duplicated()].unique().tolist()
           
            raise ValueError(f"Duplicate tickers provided: {dup}")

        results: Dict[str, Dict] = {}

        wts = np.array([1.0], dtype = float)

        one_year = pd.to_datetime(config.YEAR_AGO)
        
        bench_sr_all = benchmark_weekly_rets.loc[benchmark_weekly_rets.index >= one_year]

        xls = pd.ExcelFile(forecast_file)
      
        model_sheets = {
            'Prophet Pred': 'Prophet',
            "Prophet PCA": 'ProphetPCA',
            'Analyst Target': 'AnalystTarget',
            'Exponential Returns':'EMA',
            'DCF': 'DCF',
            'DCFE': 'DCFE',
            'DCF CapIQ': 'DCF CapIQ',
            'FCFE CapIQ': 'FCFE CapIQ',
            'DDM CapIQ': 'DDM CapIQ',
            'Daily Returns': 'Daily',
            'RI': 'RI',
            'RI CapIQ': 'RI CapIQ',
            'CAPM BL Pred': 'CAPM',
            'FF3 Pred': 'FF3',
            'FF5 Pred': 'FF5',
            'Factor Exponential Regression': 'FER',
            'Gordon Growth Model': 'Gordon Growth Model',
            'SARIMAX Monte Carlo': 'SARIMAX',
            'Rel Val Pred': 'RelVal',
            'LSTM_DirectH': 'LSTM_DirectH',
            'LSTM_Cross_Asset': 'LSTM_Cross_Asset',
            'GRU_cal': 'GRU_cal',
            'GRU_raw': 'GRU_raw',
            'Advanced MC': 'AdvMC',
            'HGB Returns': 'HGB Returns',
            'HGB Returns CA': 'HGB Returns CA',
            'TVP + GARCH Monte Carlo': 'TVP + GARCH Monte Carlo',
            'SARIMAX Factor': 'SARIMAX Factor',
        }
        
        model_returns: Dict[str, pd.Series] = {}
        
        for sheet, name in model_sheets.items():
        
            df = xls.parse(sheet, usecols = ["Ticker", "Returns"], index_col = "Ticker")
        
            df.index = df.index.str.upper()
        
            model_returns[name] = df["Returns"].reindex(tickers)

        for t in tickers:
          
            if (t not in last_year_weekly_rets.columns) or (t not in last_5y_weekly_rets.columns):
          
                continue

            stock_df_1y = last_year_weekly_rets[[t]].dropna().copy(deep = True)
          
            stock_df_5y = last_5y_weekly_rets[[t]].dropna().copy(deep = True)

            vol_weekly = float(np.sqrt(weekly_cov.loc[t, t]))
           
            vol_annual = float(np.sqrt(ann_cov.loc[t, t]))

            common = stock_df_1y.index.intersection(bench_sr_all.index)
           
            stock_df_1y = stock_df_1y.loc[common]
           
            bench_sr_t = bench_sr_all.loc[common]

            bl_cov_t = np.array([[float(bl_cov.loc[t, t])]], dtype = float)
           
            beta_t = np.array([float(beta.loc[t])], dtype = float)
           
            score_t = np.array([float(comb_score.loc[t])], dtype = float)
            
            tax_t = tax.loc[t]
            
            d_to_e_t = d_to_e.loc[t]

            metrics = self.simulate_and_report(
                name = t,
                wts = wts,
                comb_rets = float(comb_rets.loc[t]),
                bear_rets = float(bear_rets.loc[t]),
                bull_rets = float(bull_rets.loc[t]),
                vol = vol_weekly,
                vol_ann = vol_annual,
                comb_score = score_t,
                last_year_weekly_rets = stock_df_1y,
                last_5y_weekly_rets = stock_df_5y,
                n_last_year_weeks = n_last_year_weeks,
                rf = rf,
                beta = beta_t,
                benchmark_weekly_rets = bench_sr_t,
                benchmark_ann_ret = benchmark_ann_ret,
                bl_ret = float(bl_ret.loc[t]),
                bl_cov = bl_cov_t,
                tax = tax_t,
                d_to_e = d_to_e_t,
                sims = sims,
                n_trials = n_trials,
                qmc_mode = qmc_mode,
            )

            results[t] = metrics

        metrics_df = pd.DataFrame.from_dict(results, orient = "index")
        
        ret_df = pd.DataFrame(model_returns)
        
        final_df = ret_df.join(metrics_df)
        
        final_df["Combined Return"] = comb_rets.astype(float)
        
        return final_df.reindex(tickers)


    def report_portfolio_metrics_batch(
        self,
        *,
        weights: Dict[str, np.ndarray],        
        vols_weekly: Dict[str, float],      
        vols_annual: Dict[str, float],        
        comb_rets: pd.Series,                
        bear_rets: pd.Series,                
        bull_rets: pd.Series,                  
        comb_score: pd.Series,            
        last_year_weekly_rets: pd.DataFrame,   
        last_5y_weekly_rets: pd.DataFrame,    
        n_last_year_weeks: int,
        rf_rate: float,
        beta: pd.Series,                       
        benchmark_weekly_rets: pd.Series,      
        benchmark_ret: float,            
        mu_bl: pd.Series,                     
        sigma_bl: pd.DataFrame,    
        d_to_e: pd.Series,
        tax: pd.Series,       
        sims: int = 1_000_000,
        n_trials: int = 1,
        qmc_mode: bool = False,
    ) -> pd.DataFrame:
        """
        Batch diagnostics for K portfolios (vectorised).

        Vectorises:
      
        • Portfolio average returns:   
            
                μ_p = comb_rets^⊤ W
        
        • BL returns:                   
        
                μ_BL = mu_bl^⊤ W
        
        • Portfolio volatilities:       
        
                σ_BL,k = √( w_k^⊤ Σ_BL w_k )
        
        • Betas and scores:             
        
                β_p = β^⊤ W,  score_p = score^⊤ W
        
        • 1y/5y series:                 
        
                masked dot via `portfolio_return_robust_batch`

        For each portfolio, computes the same suite of metrics as in `simulate_and_report`.
        
        Returns a DataFrame keyed by portfolio name.
        """
       
        names = list(weights.keys())
       
        if not names:
       
            return pd.DataFrame()

        W = np.column_stack([np.asarray(weights[n], dtype = float).reshape(-1) for n in names])


        def dot_series(
            s: pd.Series
        ) -> np.ndarray:
        
            return s.to_numpy(copy = False, dtype = float) @ W


        port_rets = dot_series(
            s = comb_rets
        )            
        
        port_bear_rets = dot_series(
            s = bear_rets
        )
        
        port_bull_rets = dot_series(
            s = bull_rets
        )
        
        port_bl_rets = dot_series(
            s = mu_bl
        )

        Sigma = sigma_bl.to_numpy(copy = False, dtype = float)
       
        bl_vols = np.sqrt(np.einsum("ik,ij,jk->k", W, Sigma, W, optimize = True)) 

        beta_v = beta.to_numpy(copy = False, dtype = float)
       
        score_v = comb_score.to_numpy(copy = False, dtype = float)
                
        port_scores = score_v @ W   

        port_1y_df = self.portfolio_return_robust_batch(
            W = W, 
            returns = last_year_weekly_rets
        ) 
        
        port_1y_df.columns = names

        port_5y_df = self.portfolio_return_robust_batch(
            W = W, 
            returns = last_5y_weekly_rets
        )    
        
        port_5y_df.columns = names

        bench_1y = benchmark_weekly_rets.reindex(port_1y_df.index).dropna()
                
        bench_5y = benchmark_weekly_rets.reindex(port_5y_df.index).dropna()
                
        n_i = port_1y_df.count(axis = 0).clip(lower = 1)

        ann_hist_ret = (1.0 + port_1y_df).prod(skipna = True) ** (n_last_year_weeks / n_i) - 1.0
         
        ann_hist_vol = port_1y_df.std(ddof = 1) * np.sqrt(n_last_year_weeks)      
        
        p_d_to_e = self.port_d_to_e_batch(                       
            W = W,
            d_to_e = d_to_e,
            names = names
        )
        
        p_tax = self.port_tax_batch(                       
            W = W,
            tax = tax,
            names = names
        )
        
        records: Dict[str, Dict[str, float]] = {}

        for i, name in enumerate(names):

            w = W[:, i]

            vol_w = vols_weekly[name]

            vol_ann_w = vols_annual[name]

            s1y = port_1y_df.iloc[:, i]

            s5y = port_5y_df.iloc[:, i]

            ann_ret_i = float(ann_hist_ret.iloc[i])

            ann_vol_i = float(ann_hist_vol.iloc[i])

            bl_vol_i = float(bl_vols[i])

            score_i = float(port_scores[i])
            
            d_to_e_i = float(p_d_to_e[name])
            
            tax_i = float(p_tax[name])

            sr_pred_i = self.sharpe_ratio(
                r = s1y, 
                periods_per_year = n_last_year_weeks, 
                ann_ret = port_rets[i], 
                ann_vol = vol_ann_w
            )
            
            bl_sr_i = self.sharpe_ratio(
                r = s1y, 
                periods_per_year = n_last_year_weeks, 
                ann_ret = port_bl_rets[i], 
                ann_vol = bl_vol_i
            )
            
            sr_hist_i = self.sharpe_ratio(
                r = s1y, 
                periods_per_year = n_last_year_weeks, 
                ann_ret = ann_ret_i, 
                ann_vol = ann_vol_i
            )
           
            dd_1y = self._maybe_cached_drawdown_series(
                series = s1y
            )
           
            dd_max = float(dd_1y.min())
           
            dd_5y = self._maybe_cached_drawdown_series(
                series = s5y
            )

            skew_val = float(self.skewness(
                r = s5y
            ))
            
            kurt_val = float(self.kurtosis(
                r = s5y
            ))

            cf_var5 = self.var_gaussian(
                r = s5y, 
                s = skew_val,
                k = kurt_val, 
                level = 5.0, 
                modified = True
            )
           
            hist_cvar5 = self.cvar_historic(
                r = s5y,
                level = 5.0
            )
            
            pred_cvar = self.port_pred_cvar(
                r_pred = port_rets[i], 
                std_pred = vol_ann_w, 
                skew = skew_val,
                kurt = kurt_val, 
                level = 5.0, 
                periods = 52
            )

            te = self.tracking_error(
                r_a = bench_5y, 
                r_b = s5y
            )
            
            ir = self.IR(
                w = w,
                er = last_5y_weekly_rets,
                te = te,
                benchmark_ret = bench_5y,
                port_series = s1y,
                ann_hist_ret = ann_ret_i,
                ann_hist_bench_ret = benchmark_ret,
                periods_per_year = n_last_year_weeks,
            )

            ui = self.ulcer_index(
                return_series = s5y, 
                dd = dd_5y
            )
           
            cd = self.cdar(
                r = s5y,
                dd = dd_5y
            )
           
            sortino = self.sortino_ratio(
                returns = s1y,
                riskfree_rate = rf_rate,
                periods_per_year = 52,
                target = config.RF_PER_WEEK, 
                er = port_rets[i]
            )
           
            sortino_hist = self.sortino_ratio(
                returns = s1y, 
                riskfree_rate = rf_rate, 
                periods_per_year = n_last_year_weeks
            )
           
            calmar = self.calmar_ratio(
                returns = s5y,
                periods_per_year = n_last_year_weeks, 
                ann_hist_ret = ann_ret_i, 
                max_dd = dd_max
            )
           
            omega = self.omega_ratio(
                returns = s5y
            )
           
            m2 = self.modigliani_ratio(
                returns = s5y,
                bench_returns = bench_1y, 
                riskfree_rate = rf_rate, 
                periods_per_year = n_last_year_weeks,
                sr = sr_hist_i
            )
           
            pi, pr = self.pain_index_and_ratio(
                returns = s5y, 
                riskfree_rate = rf_rate,
                periods_per_year = n_last_year_weeks,
                dd = dd_5y,
                cagr = ann_ret_i
            )
           
            tail = self.tail_ratio(
                returns = s5y
            )
           
            raroc_val = self.raroc(
                returns = s5y,
                riskfree_rate = rf_rate,
                periods_per_year = n_last_year_weeks, 
                ann_return = ann_ret_i)
           
            pct_pos, win_streak, loss_streak = self.percent_positive_and_streaks(
                returns = s5y
            )

            hac = self.jensen_alpha_r2(
                port_rets = s1y,
                bench_rets = bench_1y,
                rf_per_period = config.RF_PER_WEEK,
                periods_per_year = 52,
                bench_ann_ret = benchmark_ret,
                port_ann_ret_pred = port_rets[i],
                lever_beta = True,
                d_to_e = d_to_e_i,
                tax = tax_i,
                port_ann_ret_hist = ann_ret_i,
                scale_alpha_with_leverage = True,
            )
            
            beta_i = hac['beta']
            
            treynor_i = self.compute_treynor_ratio(
                port_ret = port_rets[i],
                rf = rf_rate,
                port_beta_val = beta_i
            )        

            caps = self.capture_slopes(
                port_rets = s1y, 
                bench_rets = bench_1y
            )
            
            caps_ratios = self.capture_ratios(
                port_rets = s5y,
                bench_rets = bench_5y
            )

            n_obs = len(s1y.dropna())
            
            if isinstance(sr_hist_i, (float, np.floating)):
                
                sr_sample = float(sr_hist_i)  
                
            else:
                
                sr_sample = float("nan")
            
            if np.isfinite(sr_sample):
                
                psr = self.probabilistic_sharpe_ratio(
                    sr = sr_sample,
                    sr_star = 0.0,
                    T = n_obs, 
                    skew = skew_val, 
                    kurt = kurt_val
                ) 
            
            else: 
                
                psr = np.nan
            
            if np.isfinite(sr_sample):
                
                dsr = self.deflated_sharpe_ratio(
                    sr = sr_sample, 
                    T = n_obs,
                    skew = skew_val,
                    kurt = kurt_val
                )  
                
            else:
                
                dsr = np.nan

            stats = self.simulate_portfolio_stats(
                mu = port_rets[i],
                sigma = vol_ann_w,
                steps = 252,
                s0 = 100.0,
                scenarios = sims,
                skew = skew_val,
                kurt = kurt_val,
                method = "edgeworth",
                qmc_mode = qmc_mode,
                random_state = 42,
            )

            records[name] = {
                "Average Returns": port_rets[i],
                "Average Bear Returns": port_bear_rets[i],
                "Average Bull Returns": port_bull_rets[i],
                "BL Returns": port_bl_rets[i],
                "Weekly Volatility": vol_w,
                "Annual Volatility": vol_ann_w,
                "BL Volatility": bl_vol_i,
                "Scenario Average Returns": stats["mean_returns"],
                "Scenario Loss Incurred": stats["loss_percentage"],
                "Scenario Average Loss": stats["mean_loss_amount"],
                "Scenario Average Gain": stats["mean_gain_amount"],
                "Scenario Variance": stats["variance"],
                "Scenario 10th Percentile": stats["10th_percentile"],
                "Scenario Lower Quartile": stats["lower_quartile"],
                "Scenario Upper Quartile": stats["upper_quartile"],
                "Scenario 90th Percentile": stats["90th_percentile"],
                "Scenario Up/Down": stats["scenarios_up_down"],
                "Scenario Min Returns": stats["min_return"],
                "Scenario Max Returns": stats["max_return"],
                "Portfolio Beta": beta_i,
                "Treynor Ratio": treynor_i,
                "Portfolio Score": score_i,
                "Portfolio Tracking Error": te,
                "Information Ratio": ir,
                "Sortino Ratio": sortino,
                "Sortino Ratio (Historical)": sortino_hist,
                "Calmar Ratio": calmar,
                "Omega Ratio": omega,
                "M2 (Modigliani)": m2,
                "Pain Index": pi,
                "Pain Ratio": pr,
                "Tail Ratio": tail,
                "RAROC": raroc_val,
                "Percent Positive Periods": pct_pos,
                "Max Win Streak": win_streak,
                "Max Loss Streak": loss_streak,
                "Skewness": skew_val,
                "Kurtosis": kurt_val,
                "Cornish-Fisher VaR (5%)": cf_var5,
                "Historic CVaR (5%)": float(hist_cvar5),
                "Predicted CVaR (5%)": pred_cvar,
                "Sharpe Ratio (Predicted)": sr_pred_i,
                "Sharpe Hist Ratio": sr_hist_i,
                "PSR (SR* = 0)": psr,
                "DSR (approx)": dsr,
                "Bl Sharpe Ratio": bl_sr_i,
                "Historic Annual Returns": ann_ret_i,
                "Max Drawdown": dd_max,
                "Ulcer Index": ui,
                "Conditional Drawdown at Risk": cd,
                "Jensen's Alpha (ann)": hac["alpha_ann"],
                "Alpha ann SE (HAC)": hac["alpha_ann_se"],
                "Alpha t (HAC)": hac["alpha_t"],
                "Beta (HAC)": hac["beta"],
                "Beta SE (HAC)": hac["beta_se"],
                "Beta t (HAC)": hac["beta_t"],
                "R-squared": hac["r2"],
                "Predicted Alpha": hac["pred_alpha"],
                "Upside Capture Ratio": caps.get("Upside Capture", np.nan),
                "Downside Capture Ratio": caps.get("Downside Capture", np.nan),
                "Upside Capture (Mean)": caps_ratios.get("Upside Capture", np.nan),
                "Downside Capture (Mean)": caps_ratios.get("Downside Capture", np.nan),
            }

        return pd.DataFrame.from_dict(records, orient="index")


    def report_portfolio_metrics(
        self,
        *,
        w_msr: np.ndarray,
        w_sortino: np.ndarray,
        w_bl: np.ndarray,
        w_mir: np.ndarray,
        w_gmv: np.ndarray,
        w_mdp: np.ndarray,
        w_msp: np.ndarray,
        w_comb: np.ndarray,
        w_comb1: np.ndarray,
        w_comb2: np.ndarray,
        w_comb3: np.ndarray,
        w_comb4: np.ndarray,
        w_comb5: np.ndarray,
        w_comb6: np.ndarray,
        w_comb7: np.ndarray,
        w_comb8: np.ndarray,
        w_comb9: np.ndarray,
        w_comb10: np.ndarray,
        w_comb11: np.ndarray,
        w_deflated_msr: np.ndarray,
        w_adjusted_msr: np.ndarray,
        w_upm_lpm: np.ndarray,
        w_up_down_cap: np.ndarray,
        comb_rets: pd.Series,
        bear_rets: pd.Series,
        bull_rets: pd.Series,
        vol_msr: float,
        vol_sortino: float,
        vol_bl: float,
        vol_mir: float,
        vol_gmv: float,
        vol_mdp: float,
        vol_msp: float,
        vol_comb: float,
        vol_comb1: float,
        vol_comb2: float,
        vol_comb3: float,
        vol_comb4: float,
        vol_comb5: float,
        vol_comb6: float,
        vol_comb7: float,
        vol_comb8: float,
        vol_comb9: float,
        vol_comb10: float,
        vol_comb11: float,
        vol_deflated_msr: float,
        vol_adjusted_msr: float,
        vol_upm_lpm: float,
        vol_up_down_cap: float,
        vol_msr_ann: float,
        vol_sortino_ann: float,
        vol_bl_ann: float,
        vol_mir_ann: float,
        vol_gmv_ann: float,
        vol_mdp_ann: float,
        vol_msp_ann: float,
        vol_comb_ann: float,
        vol_comb1_ann: float,
        vol_comb2_ann: float,
        vol_comb3_ann: float,
        vol_comb4_ann: float,
        vol_comb5_ann: float,
        vol_comb6_ann: float,
        vol_comb7_ann: float,
        vol_comb8_ann: float,
        vol_comb9_ann: float,
        vol_comb10_ann: float,
        vol_comb11_ann: float,
        vol_deflated_msr_ann: float,
        vol_adjusted_msr_ann: float,
        vol_upm_lpm_ann: float,
        vol_up_down_cap_ann: float,
        comb_score: pd.Series,
        last_year_weekly_rets: pd.DataFrame,
        last_5y_weekly_rets: pd.DataFrame,
        n_last_year_weeks: int,
        rf_rate: float,
        beta: pd.Series,
        benchmark_weekly_rets: pd.Series,
        benchmark_ret: float,
        mu_bl: pd.Series,
        sigma_bl: pd.DataFrame,
        d_to_e: pd.Series,
        tax: pd.Series,
        w_comb12: np.ndarray | None = None,
        vol_comb12: float | None = None,
        vol_comb12_ann: float | None = None,
        sims: int = 1_000_000,
        n_trials: int = 1,
        qmc_mode: bool = False,
    ) -> pd.DataFrame:
        """
        Convenience wrapper that assembles named weight/volatility dictionaries for
        a set of canonical portfolios, then delegates to `report_portfolio_metrics_batch`.
        """
        
        weights = {
            "MSR": w_msr, 
            "Sortino": w_sortino,
            "Black-Litterman": w_bl, 
            "MIR": w_mir,
            "GMV": w_gmv,
            "MDP": w_mdp,
            "MSP": w_msp,
            "Deflated MSR": w_deflated_msr,
            "Adjusted MSR": w_adjusted_msr,      
            "UPM-LPM": w_upm_lpm,     
            "Up/Down Cap": w_up_down_cap,
            "Combination": w_comb, 
            "Combination1": w_comb1, 
            "Combination2": w_comb2,
            "Combination3": w_comb3,
            "Combination4": w_comb4,
            "Combination5": w_comb5,
            "Combination6": w_comb6,
            "Combination7": w_comb7,
            "Combination8": w_comb8,
            "Combination9": w_comb9,
            "Combination10": w_comb10,
            "Combination11": w_comb11
        }
      
        if w_comb12 is not None:
      
            weights["Combination12"] = w_comb12
        
        vols_weekly = {
            "MSR": vol_msr, 
            "Sortino": vol_sortino, 
            "Black-Litterman": vol_bl, 
            "MIR": vol_mir,
            "GMV": vol_gmv,
            "MDP": vol_mdp,
            "MSP": vol_msp,
            "Deflated MSR": vol_deflated_msr,
            "Adjusted MSR": vol_adjusted_msr,      
            "UPM-LPM": vol_upm_lpm,     
            "Up/Down Cap": vol_up_down_cap,
            "Combination": vol_comb, 
            "Combination1": vol_comb1,
            "Combination2": vol_comb2, 
            "Combination3": vol_comb3, 
            "Combination4": vol_comb4,
            "Combination5": vol_comb5,
            "Combination6": vol_comb6,
            "Combination7": vol_comb7,
            "Combination8": vol_comb8,
            "Combination9": vol_comb9,
            "Combination10": vol_comb10,
            "Combination11": vol_comb11
        }
        
        if vol_comb12 is not None:
            
            vols_weekly["Combination12"] = vol_comb12
        
        vols_annual = {
            "MSR": vol_msr_ann,
            "Sortino": vol_sortino_ann, 
            "Black-Litterman": vol_bl_ann, 
            "MIR": vol_mir_ann,
            "GMV": vol_gmv_ann,
            "MDP": vol_mdp_ann,
            "MSP": vol_msp_ann,
            "Deflated MSR": vol_deflated_msr_ann, 
            "Adjusted MSR": vol_adjusted_msr_ann,
            "UPM-LPM": vol_upm_lpm_ann,
            "Up/Down Cap": vol_up_down_cap_ann,
            "Combination": vol_comb_ann,
            "Combination1": vol_comb1_ann, 
            "Combination2": vol_comb2_ann,
            "Combination3": vol_comb3_ann, 
            "Combination4": vol_comb4_ann,
            "Combination5": vol_comb5_ann,
            "Combination6": vol_comb6_ann,
            "Combination7": vol_comb7_ann,
            "Combination8": vol_comb8_ann,
            "Combination9": vol_comb9_ann,
            "Combination10": vol_comb10_ann,
            "Combination11": vol_comb11_ann
        }
      
        if vol_comb12_ann is not None:
          
            vols_annual["Combination12"] = vol_comb12_ann

        return self.report_portfolio_metrics_batch(
            weights = weights,
            vols_weekly = vols_weekly,
            vols_annual = vols_annual,
            comb_rets = comb_rets,
            bear_rets = bear_rets,
            bull_rets = bull_rets,
            comb_score = comb_score,
            last_year_weekly_rets = last_year_weekly_rets,
            last_5y_weekly_rets = last_5y_weekly_rets,
            n_last_year_weeks = n_last_year_weeks,
            rf_rate = rf_rate,
            beta = beta,
            benchmark_weekly_rets = benchmark_weekly_rets,
            benchmark_ret = benchmark_ret,
            mu_bl = mu_bl,
            sigma_bl = sigma_bl,
            d_to_e = d_to_e,
            tax = tax,
            sims = sims,
            n_trials = n_trials,
            qmc_mode = qmc_mode,
        )


    @staticmethod
    def omega_ratio(
        returns: pd.Series, 
        threshold: float = 0.0
    ) -> float:
        """
        Omega ratio at threshold τ:

            Ω(τ) =  ( ∑_{r_t>τ} (r_t − τ) ) / ( ∑_{r_t≤τ} (τ − r_t) ).

        Returns +∞ if the denominator is zero (no losses).
        """
        
        gains = (returns[returns > threshold] - threshold).sum()
    
        losses = (threshold - returns[returns <= threshold]).sum()
    
        if losses != 0:
    
            return float(gains / losses)
    
        return np.inf


    @staticmethod
    def modigliani_ratio(
        returns: pd.Series,
        bench_returns: pd.Series,
        riskfree_rate: float,
        periods_per_year: int,
        sr: float | None = None,
        ann_hist_return: float | None = None,
        ann_hist_vol: float | None = None,
        bench_vol_ann: float | None = None,
    ) -> float:
        """
        Modigliani–Modigliani risk-adjusted performance (M²):

            M² = R_f + SR · σ_bench,    where SR is the portfolio Sharpe ratio,
                                        σ_bench is the annualised benchmark volatility.

        If SR is not provided, it is computed via geometric annualisation and σ_ann = σ_pp √P.
        """      
        
        if sr is None:
      
            if ann_hist_return is None:
      
                ann_hist_return = PortfolioAnalytics.annualise_returns(
                    ret_series = returns, 
                    periods_per_year = periods_per_year
                )
      
            if ann_hist_vol is None:
      
                ann_hist_vol = PortfolioAnalytics.annualise_vol(
                    r = returns, 
                    periods_per_year = periods_per_year
                )
      
            sr = PortfolioAnalytics.sharpe_ratio(
                r = returns,
                periods_per_year = periods_per_year, 
                ann_ret = ann_hist_return, 
                ann_vol = ann_hist_vol
            )
      
        if bench_vol_ann is None:
      
            bench_vol_ann = PortfolioAnalytics.annualise_vol(
                r = bench_returns, 
                periods_per_year = periods_per_year
            )
      
        return float(riskfree_rate + sr * bench_vol_ann)


    @staticmethod
    def pain_index_and_ratio(
        returns: pd.Series,
        riskfree_rate: float,
        periods_per_year: int,
        dd: pd.Series | None = None,
        cagr: float | None = None,
    ) -> Tuple[float, float]:
        """
        Pain metrics based on drawdowns.

        Pain Index:
           
            PI = − mean(DD_t).

        Pain Ratio:
           
            PR = ( CAGR − R_f ) / PI,    returned as NaN if PI = 0.

        If `dd`/`cagr` are omitted they are computed internally.
        """
        
        if dd is None:
     
            dd = PortfolioAnalytics.drawdown(
                return_series = returns
            )["Drawdown"]
     
        pi = float(-dd.mean())
     
        if cagr is None:
     
            cagr = PortfolioAnalytics.annualise_returns(
                ret_series = returns, 
                periods_per_year = periods_per_year
            )

        if pi > 0:
            
            pr = float((cagr - riskfree_rate) / pi)
            
        else:
            
            pr = np.nan
     
        return pi, pr


    @staticmethod
    def tail_ratio(
        returns: pd.Series, 
        upper_q: float = 0.90, 
        lower_q: float = 0.10
    ) -> float:
        """
        Tail ratio:

            TR = q_{upper_q}(r) / | q_{lower_q}(r) |,

        returned as +∞ if q_{lower_q} ≥ 0.
        """
        
        up = float(returns.quantile(upper_q))
    
        down = float(returns.quantile(lower_q))
    
        if down < 0:
    
            return float(up / abs(down))
    
        return np.inf


    @staticmethod
    def raroc(
        returns: pd.Series,
        riskfree_rate: float,
        periods_per_year: int,
        var_level: float = 5.0,
        ann_return: float | None = None,
    ) -> float:
        """
        Risk-Adjusted Return on Capital (RAROC):

            RAROC = ( μ_ann − R_f ) / VaR_{α},

        with VaR_{α} the historical VaR at level α=var_level%. Returns NaN if VaR=0.
        """
        
        if ann_return is None:
       
            ann_return = PortfolioAnalytics.annualise_returns(
                ret_series = returns,
                periods_per_year = periods_per_year
            )
       
        excess = ann_return - riskfree_rate
       
        cap = PortfolioAnalytics.var_historic(
            r = returns,
            level = var_level
        )
       
        if cap > 0:
       
            return float(excess / cap)
       
        return np.nan


    @staticmethod
    def percent_positive_and_streaks(
        returns: pd.Series
    ) -> Tuple[float, int, int]:
        """
        Fraction of positive periods and maximum consecutive winning/losing streaks.

            percent_pos = mean( r_t > 0 ),
           
            max_win = max run length of positive returns,
           
            max_loss = max run length of non-positive returns.
        """

        is_pos = returns > 0
    
        percent_pos = float(is_pos.mean())
    
        max_win = max_loss = current_win = current_loss = 0
    
        for up in is_pos:
    
            if up:
    
                current_win += 1
    
                max_win = max(max_win, current_win)
    
                current_loss = 0
    
            else:
    
                current_loss += 1
    
                max_loss = max(max_loss, current_loss)
    
                current_win = 0
    
        return percent_pos, max_win, max_loss


    @staticmethod
    def sortino_ratio(
        returns: pd.Series | pd.DataFrame,
        riskfree_rate: float,
        periods_per_year: int,
        target: float = config.RF_PER_WEEK,
        er: float | pd.Series | None = None,
    ) -> float | pd.Series:
        """
        Sortino ratio with target return τ (per period).

        For a Series:
       
        • downside set: D = { t : r_t < τ },
       
        • semi-deviation (per period):  σ_down = √( mean( (r_t − τ)^2, t ∈ D ) ),
       
        • annualised σ_down,ann = σ_down √P,
       
        • excess return: μ_ann − R_f  (from `er` if supplied, else geometric annualisation),
       
        • Sortino = ( μ_ann − R_f ) / σ_down,ann.

        For DataFrame, the computation is vectorised columnwise.
        """
       
        if isinstance(returns, pd.DataFrame):
    
            downside = returns.subtract(target).clip(upper = 0.0)
    
            semidev = np.sqrt((downside ** 2).mean())
          
            ann_downside = semidev * np.sqrt(periods_per_year)
          
            if er is None:
          
                ann_return = PortfolioAnalytics.annualise_returns(
                    ret_series = returns,
                    periods_per_year = periods_per_year
                )
          
                ann_excess = ann_return - riskfree_rate
          
            else:
          
                ann_excess = er - riskfree_rate
          
            return ann_excess / ann_downside.replace(0, np.nan)

        downside = returns[returns < target]
       
        if downside.empty:
       
            return np.nan
       
        semidev = np.sqrt(np.mean((downside - target) ** 2))
       
        ann_downside = semidev * np.sqrt(periods_per_year)
       
        if er is None:
       
            ann_return = PortfolioAnalytics.annualise_returns(
                ret_series = returns, 
                periods_per_year = periods_per_year
            )
       
            ann_excess = ann_return - riskfree_rate
       
        else:
       
            ann_excess = er - riskfree_rate
       
        if ann_downside > 0:
       
            return float(ann_excess / ann_downside)
       
        return np.nan


    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int,
        ann_hist_ret: float | None = None,
        max_dd: float | None = None,
    ) -> float:
        """
        Calmar ratio:

            Calmar = CAGR / |MaxDD|,

        where CAGR is the annualised geometric return and MaxDD is the minimum drawdown
        (a negative number). 
        
        Returns NaN if MaxDD ≥ 0.
        """
        
        returns = returns.dropna()
       
        if ann_hist_ret is None:
       
            cagr = PortfolioAnalytics.annualise_returns(
                ret_series = returns, 
                periods_per_year = periods_per_year
            )
       
        else:
       
            cagr = ann_hist_ret
       
        if max_dd is None:
       
            max_dd = PortfolioAnalytics.drawdown(
                return_series = returns
            )["Drawdown"].min()
       
        if max_dd < 0:
       
            return float(cagr / abs(max_dd))
       
        return np.nan
