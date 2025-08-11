"""
Contains portfolio utility functions—returns, volatility, downside deviation, risk metrics and Monte‑Carlo simulation helpers.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

import config


class PortfolioAnalytics:
    """
    Performance & risk analytics.

      • Caching is OPTIONAL and OFF by default (set cache = True to enable).
      • When caching is on, choose key strategy:
          - cache_key='identity' -> (id(df), weights.tobytes()) [fast]
          - cache_key='content' -> content fingerprint per df [robust]
    """

    def __init__(
        self,
        *,
        cache: bool = False,
        cache_key: str = "identity", 
    ) -> None:
      
        self._cache_enabled = bool(cache)
      
        if cache_key not in ("identity", "content"):
      
            raise ValueError("cache_key must be 'identity' or 'content'")
      
        self._cache_key_mode = cache_key

        self._port_series_cache: Dict[Tuple[Any, bytes], pd.Series] = {}

        self._dd_cache: Dict[Any, pd.Series] = {}

        self._annret_cache: Dict[Tuple[Any, int], float] = {}

        self._annvol_cache: Dict[Tuple[Any, int], float] = {}

    
    def enable_cache(
        self, 
        key_mode: str | None = None
    ) -> None:
       
        self._cache_enabled = True
       
        if key_mode:
       
            if key_mode not in ("identity", "content"):
       
                raise ValueError("cache_key must be 'identity' or 'content'")
       
            self._cache_key_mode = key_mode


    def disable_cache(
        self
    ) -> None:
       
        self._cache_enabled = False


    def clear_cache(
        self
    ) -> None:
    
        self._port_series_cache.clear()
    
        self._dd_cache.clear()
    
        self._annret_cache.clear()
    
        self._annvol_cache.clear()


    @staticmethod
    def portfolio_return(
        weights: np.ndarray, 
        returns: Any
    ) -> Any:
    
        if isinstance(returns, pd.DataFrame):
    
            return returns.dot(weights)
    
        elif isinstance(returns, pd.Series):
    
            return float(weights @ returns)
    
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
    
        return float(np.sqrt(weights @ covmat @ weights))


    @staticmethod
    def tracking_error(
        r_a: pd.Series, 
        r_b: pd.Series
    ) -> float:
    
        return float(np.sqrt(((r_a - r_b) ** 2).mean()))


    @staticmethod
    def port_beta(
        weights: np.ndarray, 
        beta: pd.Series
    ) -> float:
    
        return float(weights @ beta)


    @staticmethod
    def compute_treynor_ratio(
        port_ret: float, 
        rf: float, 
        port_beta_val: float
    ) -> float:
    
        if port_beta_val == 0:
    
            return float("nan")
    
        return (port_ret - rf) / port_beta_val

    
    @staticmethod
    def port_score(
        weights: np.ndarray, 
        score: pd.Series
    ) -> float:
    
        return float(weights @ score)

    
    @staticmethod
    def annualise_vol(
        r: pd.Series, 
        periods_per_year: int
    ) -> float:
    
        return float(r.std() * np.sqrt(periods_per_year))

    @staticmethod
    def annualise_returns(
        ret_series: pd.Series, 
        periods_per_year: int
    ) -> float:
    
        n = len(ret_series)
    
        if n <= 1:
    
            return 0.0
    
        cum = float((1.0 + ret_series).prod())
    
        return float(cum ** (periods_per_year / n) - 1.0)

    
    @staticmethod
    def sharpe_ratio(
        r: pd.Series,
        periods_per_year: int,
        ann_ret: float | None = None,
        ann_vol: float | None = None,
    ) -> float:
      
        if ann_ret is None:
      
            excess_ret = r - config.RF_PER_WEEK
      
            ann_ex_ret = PortfolioAnalytics.annualise_returns(
                ret_series = excess_ret, 
                periods_per_year = periods_per_year
            )
      
        else:
      
            ann_ex_ret = ann_ret - config.RF
      
        if ann_vol is None:
      
            ann_vol = PortfolioAnalytics.annualise_vol(
                r = r, 
                periods_per_year = periods_per_year
            )
      
        return float(ann_ex_ret / ann_vol) if ann_vol > 0 else np.nan


    @staticmethod
    def drawdown(
        return_series: pd.Series
    ) -> pd.DataFrame:
        
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
        r: pd.Series
    ) -> float:
       
        demeaned = r - r.mean()
       
        sigma = r.std(ddof = 0)
       
        if sigma == 0:
       
            return 0.0
       
        return float(((demeaned ** 3).mean()) / (sigma ** 3))


    @staticmethod
    def kurtosis(
        r: pd.Series
    ) -> float:
    
        demeaned = r - r.mean()
    
        sigma = r.std(ddof = 0)
    
        if sigma == 0:
    
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
       
            z = (z
                 + (z ** 2 - 1) * s / 6
                 + (z ** 3 - 3 * z) * (k - 3) / 24
                 - (2 * z ** 3 - 5 * z) * (s ** 2) / 36)
       
        return float(-(r.mean() + z * r.std(ddof = 0)))


    @staticmethod
    def var_historic(
        r: pd.Series | pd.DataFrame, 
        level: float = 5.0
    ) -> float | pd.Series:
       
        if isinstance(r, pd.DataFrame):
       
            return r.aggregate(PortfolioAnalytics.var_historic, level = level)
       
        if isinstance(r, pd.Series):
       
            return float(-np.percentile(r, level))
       
        raise TypeError("Expected r to be a Series or DataFrame")
    

    @staticmethod
    def cvar_historic(
        r: pd.Series | pd.DataFrame, 
        level: float = 5.0
    ) -> float | pd.Series:
    
        if isinstance(r, pd.Series):
    
            var = PortfolioAnalytics.var_historic(r, level = level)
    
            tail = r[r <= -var]
       
            return float(-tail.mean()) if not tail.empty else 0.0
       
        if isinstance(r, pd.DataFrame):
       
            return r.aggregate(PortfolioAnalytics.cvar_historic, level = level)
       
        raise TypeError("Expected r to be a Series or DataFrame")


    @staticmethod
    def port_pred_cvar(
        r_pred: float,
        std_pred: float,
        skew: float,
        kurt: float,
        level: float = 5.0,
        periods: int = 52,
    ) -> float:
       
        alpha = level / 100.0
       
        z = norm.ppf(alpha)
       
        z_cf = (z
                + (z ** 2 - 1) * skew / 6
                + (z ** 3 - 3 * z) * (kurt - 3) / 24
                - (2 * z ** 3 - 5 * z) * (skew ** 2) / 36)
       
        r_pred_pp = (1.0 + r_pred) ** (1.0 / periods) - 1.0
      
        std_pp = std_pred / np.sqrt(periods)
      
        return float(r_pred_pp + std_pp * norm.pdf(z_cf) / alpha)


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
       
                port_series = PortfolioAnalytics.portfolio_return_robust(w, er)  # type: ignore[arg-type]
       
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
    
        if dd is None:
    
            wealth = (1 + r).cumprod()
    
            peak = wealth.cummax()
    
            dd = (wealth - peak) / peak
    
        thresh = dd.quantile(level / 100.0)
    
        worst = dd[dd <= thresh]
    
        return float(worst.mean()) if not worst.empty else 0.0


    @staticmethod
    def jensen_alpha_r2(
        port_rets: pd.Series,
        bench_ann_ret: float,
        port_ret: float,
        bench_rets: pd.Series,
        rf: float,
        periods_per_year: int,
    ) -> Tuple[float, float, float]:
      
        df = pd.concat([port_rets, bench_rets], axis = 1).dropna()
      
        df.columns = ["p", "b"]
      
        y = df["p"] - config.RF_PER_WEEK
       
        X = sm.add_constant(df["b"] - config.RF_PER_WEEK)
       
        model = sm.OLS(y, X).fit()
       
        alpha_per_period = float(model.params["const"])
       
        alpha_ann = (1.0 + alpha_per_period) ** periods_per_year - 1.0
       
        pred_alpha = port_ret - (config.RF + float(model.params["b"]) * (bench_ann_ret - config.RF))
       
        return float(alpha_ann), float(model.rsquared), float(pred_alpha)


    @staticmethod
    def capture_ratios(
        port_rets: pd.Series, 
        bench_rets: pd.Series
    ) -> Dict[str, float]:
    
        df = pd.concat([port_rets, bench_rets], axis = 1).dropna()
    
        df.columns = ["p", "b"]
    
        up = df[df["b"] > 0]
    
        down = df[df["b"] < 0]
    
        up_cap = float(up["p"].mean() / up["b"].mean()) if not up.empty else np.nan
    
        down_cap = float(down["p"].mean() / down["b"].mean()) if not down.empty else np.nan

        caps = {
            "Upside Capture": up_cap, 
            "Downside Capture": down_cap
        }
        
        return caps


    @staticmethod
    def portfolio_return_robust(
        weights: np.ndarray, 
        returns: pd.DataFrame | pd.Series
    ) -> pd.Series:
    
        w = np.asarray(weights, dtype = float)
    
        if isinstance(returns, pd.Series):
        
            returns = returns.to_frame()

        def row_ret(
            row: pd.Series
        ) -> float:
        
            vals = row.to_numpy()
        
            valid = ~np.isnan(vals)
        
            if not valid.any():
        
                return np.nan
        
            w_sub = w[valid]
        
            w_sub = w_sub / w_sub.sum()
        
            return float((vals[valid] * w_sub).sum())

        return returns.apply(row_ret, axis = 1)


    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        riskfree_rate: float,
        periods_per_year: int,
        target: float = config.RF_PER_WEEK,
        er: float | None = None,
    ) -> float:
      
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
        
        else:
            
            return np.nan


    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int,
        ann_hist_ret: float | None = None,
        max_dd: float | None = None,
    ) -> float:
       
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
        
        else:
            
            return np.nan
        

    @staticmethod
    def omega_ratio(
        returns: pd.Series, 
        threshold: float = 0.0
    ) -> float:
    
        gains = (returns[returns > threshold] - threshold).sum()
    
        losses = (threshold - returns[returns <= threshold]).sum()

        if losses != 0:
            
            return float(gains / losses)  
        
        else:
            
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
    
        up = float(returns.quantile(upper_q))
    
        down = float(returns.quantile(lower_q))

        if down < 0:
            
            return float(up / abs(down))  
        
        else:
            
            return np.inf


    @staticmethod
    def raroc(
        returns: pd.Series,
        riskfree_rate: float,
        periods_per_year: int,
        var_level: float = 5.0,
        ann_return: float | None = None,
    ) -> float:
      
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
        
        else:
            
            return np.nan


    @staticmethod
    def percent_positive_and_streaks(
        returns: pd.Series
    ) -> Tuple[float, int, int]:
        
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
    def _edgeworth_eps(
        n_steps: int,
        n_scenarios: int,
        skew: float,
        kurt: float,
        random_state: int | None = None,
    ) -> np.ndarray:
      
        rng = np.random.default_rng(random_state)
      
        Z = rng.standard_normal(size = (n_steps, n_scenarios))
      
        g1 = float(skew)
      
        g2 = float(kurt) - 3.0
     
        eps = (
            Z
            + (g1 / 6.0) * (Z**2 - 1.0)
            + (g2 / 24.0) * (Z**3 - 3.0 * Z)
            - ((g1**2) / 36.0) * (2.0 * Z**3 - 5.0 * Z)
        )
      
        eps -= eps.mean(axis = 1, keepdims = True)
      
        std = eps.std(axis = 1, keepdims = True)
      
        eps /= (std + 1e-12)
      
        return eps


    @staticmethod
    def gbm_non_gaussian(
        n_years: int = 10,
        n_scenarios: int = 1_000_000,
        mu: float = 0.07,
        sigma: float = 0.15,
        steps_per_year: int = 12,
        s_0: float = 100.0,
        skew: float = 0.0,
        kurt: float = 3.0,
        method: str = "edgeworth",
        prices: bool = True,
        random_state: int | None = None,
    ) -> np.ndarray:
      
        dt = 1.0 / steps_per_year
      
        n_steps = int(n_years * steps_per_year)

        if method == "gaussian":
      
            rng = np.random.default_rng(random_state)
      
            eps = rng.standard_normal(size = (n_steps, n_scenarios))
      
        else:
          
            eps = PortfolioAnalytics._edgeworth_eps(
                n_steps = n_steps, 
                n_scenarios = n_scenarios, 
                skew = skew, 
                kurt = kurt, 
                random_state = random_state
            )

        b = sigma * np.sqrt(dt)
       
        G_target = (1.0 + mu) ** (1.0 / steps_per_year)
       
        exp_b_eps_mean = np.exp(b * eps).mean(axis = 1, keepdims = True)
       
        a = np.log(G_target) - np.log(exp_b_eps_mean)
       
        gross = np.exp(a + b * eps)
       
        gross = np.vstack([np.ones((1, n_scenarios)), gross])

        if prices:
            
            paths = s_0 * np.cumprod(gross, axis = 0)  
        
        else:
            
            paths = (gross - 1.0)
       
        return paths


    @staticmethod
    def simulate_portfolio_stats(
        mu: float,
        sigma: float,
        steps: int = 252,
        s0: float = 100.0,
        scenarios: int = 1_000_000,
        skew: float = 0.0,
        kurt: float = 3.0,
        method: str = "edgeworth",
        random_state: int | None = 42,
    ) -> Dict[str, Any]:
      
        sim_paths = PortfolioAnalytics.gbm_non_gaussian(
            n_years = 1,
            n_scenarios = scenarios,
            mu = mu,
            sigma = sigma,
            steps_per_year = steps,
            s_0 = s0,
            skew = skew,
            kurt = kurt,
            method = method,
            prices = True,
            random_state = random_state,
        )
       
        final_prices = sim_paths[-1]
       
        final_returns = pd.Series((final_prices / s0) - 1.0)

        p10 = float(final_returns.quantile(0.10))
       
        p90 = float(final_returns.quantile(0.90))
       
        q25_l = float(final_returns.quantile(0.245))
       
        q25_h = float(final_returns.quantile(0.255))
       
        q75_l = float(final_returns.quantile(0.745))
       
        q75_h = float(final_returns.quantile(0.755))
       
        q75 = float(final_returns.quantile(0.75))

        if p10 != 0:
            
            scen_up_down = (p90 / p10)  
            
        else:
            
            scen_up_down = np.inf

        lower_quart = float(final_returns[(final_returns >= q25_l) & (final_returns <= q25_h)].mean())
        
        upper_quart = float(final_returns[(final_returns >= q75_l) & (final_returns <= q75_h)].mean())
        
        upper_mean = float(final_returns[final_returns >= q75].mean())

        return {
            "mean_returns": float(final_returns.mean()),
            "loss_percentage": float(100.0 * (final_returns < 0).mean()),
            "mean_loss_amount": float(final_returns[final_returns < 0].mean()),
            "mean_gain_amount": float(final_returns[final_returns >= 0].mean()),
            "variance": float((final_prices / s0).var()),
            "10th_percentile": p10,
            "lower_quartile": lower_quart,
            "upper_quartile": upper_quart,
            "90th_percentile": p90,
            "scenarios_up_down": float(scen_up_down),
            "upper_returns_mean": upper_mean,
            "min_return": float(final_prices.min() / s0) - 1.0,
            "max_return": float(final_prices.max() / s0) - 1.0,
        }


    def _df_key(
        self, 
        df: pd.DataFrame
    ) -> Any:
        """
        Return a cache key based on the selected mode.
        """
        
        if self._cache_key_mode == "identity":
        
            return id(df)

        vals = df.to_numpy()

        h = (vals.size, vals.shape, float(np.nanmean(vals)) if vals.size else 0.0, float(np.nanstd(vals)) if vals.size else 0.0)
        
        return (tuple(df.columns), tuple(df.index[:1]) + tuple(df.index[-1:]), h)


    def _maybe_cached_port_series(
        self, 
        weights: np.ndarray, 
        rets_df: pd.DataFrame
    ) -> pd.Series:
    
        if not self._cache_enabled:
    
            return self.portfolio_return_robust(
                weights = weights, 
                returns = rets_df
            )
    
        key = (self._df_key(rets_df), np.asarray(weights, float).tobytes())
    
        s = self._port_series_cache.get(key)
    
        if s is None:
    
            s = self.portfolio_return_robust(
                weights = weights, 
                returns = rets_df
            )
    
            self._port_series_cache[key] = s
    
        return s


    def _maybe_cached_drawdown_series(
        self, 
        series: pd.Series
    ) -> pd.Series:
    
        if not self._cache_enabled:
    
            dd =  self.drawdown(
                return_series = series
            )["Drawdown"]
            
            return dd
    
        k = id(series)
    
        s = self._dd_cache.get(k)
    
        if s is None:
    
            s = self.drawdown(
                return_series = series
            )["Drawdown"]
    
            self._dd_cache[k] = s
    
        return s


    def _maybe_cached_ann_ret(
        self, 
        series: pd.Series, 
        periods: int
    ) -> float:
    
        if not self._cache_enabled:
    
            return self.annualise_returns(
                ret_series = series, 
                periods_per_year = periods
            )
    
        k = (id(series), periods)
    
        v = self._annret_cache.get(k)
    
        if v is None:
    
            v = self.annualise_returns(
                ret_series = series, 
                periods_per_year = periods
            )
    
            self._annret_cache[k] = v
    
        return v


    def _maybe_cached_ann_vol(
        self, 
        series: pd.Series, 
        periods: int
    ) -> float:
    
        if not self._cache_enabled:
    
            return self.annualise_vol(
                r = series, 
                periods_per_year = periods
            )
    
        k = (id(series), periods)
    
        v = self._annvol_cache.get(k)
    
        if v is None:
    
            v = self.annualise_vol(
                r = series, 
                periods_per_year = periods
            )
     
            self._annvol_cache[k] = v
    
        return v


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
        sims: int = 1_000_000,
    ) -> Dict[str, Any]:

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
        
        b_val = self.port_beta(
            weights = wts, 
            beta = pd.Series(beta) if isinstance(beta, np.ndarray) else beta
        )
        
        treynor = self.compute_treynor_ratio(
            port_ret = port_rets, 
            rf = rf, 
            port_beta_val = b_val
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

        alpha, r2, pred_alpha = self.jensen_alpha_r2(
            port_rets = port_1y,
            bench_ann_ret = benchmark_ann_ret,
            port_ret = port_rets,
            bench_rets = benchmark_weekly_rets,
            rf = rf,
            periods_per_year = 52,
        )
        caps = self.capture_ratios(
            port_rets = port_1y, 
            bench_rets = benchmark_weekly_rets
        )

        stats = self.simulate_portfolio_stats(
            mu = port_rets,
            sigma = vol_ann,
            steps = 252,
            s0 = 100.0,
            scenarios = sims,
            skew = skew_val,
            kurt = kurt_val,
            method = "edgeworth",
            random_state = 42,
        )

        return {
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
            "Portfolio Beta": b_val,
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
            "Bl Sharpe Ratio": bl_sr,
            "Historic Annual Returns": ann_hist_ret,
            "Max Drawdown": dd_max,
            "Ulcer Index": ui,
            "Conditional Drawdown at Risk": cd,
            "Jensen's Alpha": alpha,
            "Predicted Alpha": pred_alpha,
            "R-squared": r2,
            "Upside Capture Ratio": caps.get("Upside Capture", np.nan),
            "Downside Capture Ratio": caps.get("Downside Capture", np.nan),
        }


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
        forecast_file: str,
        sims: int = 10_000,
    ) -> pd.DataFrame:

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
        
        last_5y_weekly_rets = _up_df(last_5y_weekly_rets).reindex(columns = tickers)
        
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

        if len(set(tickers)) != len(tickers):
           
            dup = pd.Index(tickers)[pd.Index(tickers).duplicated()].unique().tolist()
           
            raise ValueError(f"Duplicate tickers provided: {dup}")

        results: Dict[str, Dict] = {}
       
        wts = np.array([1.0], dtype = float)

        one_year = pd.to_datetime(config.YEAR_AGO)
       
        bench_sr_all = benchmark_weekly_rets.loc[benchmark_weekly_rets.index >= one_year]

        xls = pd.ExcelFile(forecast_file)

        model_sheets = {
            "Prophet Pred": "Prophet",
            "Analyst Target": "AnalystTarget",
            "Exponential Returns": "EMA",
            "Lin Reg Returns": "LinReg",
            "DCF": "DCF",
            "DCFE": "DCFE",
            "Daily Returns": "Daily",
            "RI": "RI",
            "CAPM BL Pred": "CAPM",
            "FF3 Pred": "FF3",
            "FF5 Pred": "FF5",
            'Factor Exponential Regression': 'FER',
            "SARIMAX Monte Carlo": "SARIMAX",
            "Rel Val Pred": "RelVal",
            'LSTM': 'LSTM',
        }
       
        model_returns: Dict[str, pd.Series] = {}
       
        for sheet, name in model_sheets.items():
       
            df = xls.parse(sheet, usecols=["Ticker", "Returns"], index_col = "Ticker")
          
            df.index = df.index.str.upper()
          
            model_returns[name] = df["Returns"].reindex(tickers)

        for t in tickers:
        
            col_ok = (t in last_year_weekly_rets.columns) and (t in last_5y_weekly_rets.columns)
          
            if not col_ok:

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
                sims = sims,
            )
            
            results[t] = metrics

        metrics_df = pd.DataFrame.from_dict(results, orient = "index")

        ret_df = pd.DataFrame(model_returns)

        final_df = ret_df.join(metrics_df)

        final_df["Combined Return"] = comb_rets.astype(float)

        return final_df.reindex(tickers)


    def report_portfolio_metrics(
        self,
        *,
        w_msr: np.ndarray,
        w_sortino: np.ndarray,
        w_bl: np.ndarray,
        w_mir: np.ndarray,
        w_msp: np.ndarray,
        w_comb: np.ndarray,
        w_comb1: np.ndarray,
        w_comb2: np.ndarray,
        comb_rets: pd.Series,
        bear_rets: pd.Series,
        bull_rets: pd.Series,
        vol_msr: float,
        vol_sortino: float,
        vol_bl: float,
        vol_mir: float,
        vol_msp: float,
        vol_comb: float,
        vol_comb1: float,
        vol_comb2: float,
        vol_msr_ann: float,
        vol_sortino_ann: float,
        vol_bl_ann: float,
        vol_mir_ann: float,
        vol_msp_ann: float,
        vol_comb_ann: float,
        vol_comb1_ann: float,
        vol_comb2_ann: float,
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
        sims: int = 1_000_000,
    ) -> pd.DataFrame:
       
        reports: Dict[str, Dict[str, float]] = {}

        def run(
            name: str, 
            w: np.ndarray, 
            vol_w: float, 
            vol_ann_w: float
        ) -> Dict[str, float]:
        
            return self.simulate_and_report(
                name = name,
                wts = w,
                comb_rets = comb_rets,
                bear_rets = bear_rets,
                bull_rets = bull_rets,
                vol = vol_w,
                vol_ann = vol_ann_w,
                comb_score = comb_score,
                last_year_weekly_rets = last_year_weekly_rets,
                last_5y_weekly_rets = last_5y_weekly_rets,
                n_last_year_weeks = n_last_year_weeks,
                rf = rf_rate,
                beta = beta,
                benchmark_weekly_rets = benchmark_weekly_rets,
                benchmark_ann_ret = benchmark_ret,
                bl_ret = mu_bl,
                bl_cov = sigma_bl,
                sims = sims,
            )

        reports["MSR"] = run(
            name = "MSR", 
            w = w_msr, 
            vol_w = vol_msr, 
            vol_ann_w = vol_msr_ann
        )
        
        reports["Sortino"] = run(
            name = "Sortino", 
            w = w_sortino, 
            vol_w = vol_sortino, 
            vol_ann_w = vol_sortino_ann
        )
        
        reports["Black-Litterman"] = run(
            name = "Black-Litterman", 
            w = w_bl, 
            vol_w = vol_bl, 
            vol_ann_w = vol_bl_ann
        )
        
        reports["MIR"] = run(
            name = "MIR", 
            w = w_mir, 
            vol_w = vol_mir, 
            vol_ann_w = vol_mir_ann
        )
        
        reports["MSP"] = run(
            name = "MSP", 
            w = w_msp, 
            vol_w = vol_msp, 
            vol_ann_w = vol_msp_ann
        )
        
        reports["Combination"] = run(
            name = "Combination", 
            w = w_comb, 
            vol_w = vol_comb, 
            vol_ann_w = vol_comb_ann
        )
        
        reports["Combination1"] = run(
            name = "Combination1", 
            w = w_comb1, 
            vol_w = vol_comb1, 
            vol_ann_w = vol_comb1_ann
        )
        
        reports["Combination2"] = run(
            name = "Combination2", 
            w = w_comb2, 
            vol_w = vol_comb2, 
            vol_ann_w = vol_comb2_ann
        )

        return pd.DataFrame(reports).T
