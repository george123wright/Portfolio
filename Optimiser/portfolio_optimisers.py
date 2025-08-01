"""
Implements optimisation routines for max Sharpe, Sortino, information ratio, equal‑risk and combination strategies with bounds.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy.optimize import minimize
import portfolio_functions as pf
from functions.black_litterman_model import black_litterman

import config

sector_limits = {
    'Technology': 0.3,
    'Healthcare': 0.1    
}


def align_by_ticker(func):
    """
    Make every optimiser:
      • internally work with numpy arrays (fast),
      • externally always accept/return objects that share the same ticker order.

    The canonical order is taken from the tickers given in config.py.
    """
    @functools.wraps(func)
    
    def _wrapper(*args, **kwargs):

        tickers = config.tickers

        def _reindex(
            obj
        ):
           
            if isinstance(obj, pd.Series):
               
                return obj.reindex(tickers)
            
            if isinstance(obj, pd.DataFrame):

                if obj.index.equals(obj.columns):
                   
                    return obj.reindex(index = tickers, columns = tickers)

                return obj.reindex(columns = tickers)

            return obj

        args = list(args)

        for i, a in enumerate(args):
            
            args[i] = _reindex(
                obj = a
            )
       
        for k in list(kwargs):
       
            kwargs[k] = _reindex(
                obj = kwargs[k]
            )

        w = func(*args, **kwargs)   
        
        return pd.Series(w, index = tickers, name = func.__name__)

    return _wrapper


def generate_bounds_for_asset(
    bnd_h: float,
    bnd_l: float,
    er: float,
    score: float,
) -> Tuple[float, float]:
    """
    Decide (min, max) weight for each asset based on:
      - combination score
    """

    if score <= 0 or er <= 0:
       
        return (0, 0)

    else:
        
        return (bnd_l, bnd_h)
    
@align_by_ticker
def msr(
    riskfree_rate: float,
    er: pd.Series,
    cov: np.ndarray,
    scores: pd.Series,
    ticker_ind: pd.Series,
    ticker_sec: pd.Series,
    bnd_h: float,
    bnd_l: float,
    max_industry_pct: float = 0.1,
    max_sector_pct: float = 0.15
) -> np.ndarray:
    """
    Standard max Sharpe Ratio with industry constraints and (score, buy, sell)-based bounds.
    """
 
    n = er.shape[0]
 
    init_guess = np.zeros(n)
 
    nonzero_assets = np.where(er != 0)[0]
 
    if len(nonzero_assets) > 0:
        
        init_guess[nonzero_assets] = 1 / len(nonzero_assets)
 
    bnds = [
        generate_bounds_for_asset(
            bnd_h = bnd_h.iloc[i], 
            bnd_l = bnd_l.iloc[i], 
            er = er.iloc[i], 
            score = scores.iloc[i]
        )
        for i in range(n)
    ]
 
    constraints = [{
        "type": "eq", 
        "fun": lambda w: np.sum(w) - 1
    }]
 
    for industry in ticker_ind.unique():
        
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        
        constraints.append({
            "type": "ineq",
            "fun": lambda w,
            inds=idxs: max_industry_pct - np.sum(w[inds])
        })

    for sector in ticker_sec.unique():
       
        limit = sector_limits.get(sector, max_sector_pct)
       
        idxs = [i for i, tk in enumerate(er.index) if ticker_sec.loc[tk] == sector]
       
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs, 
            limit = limit: limit - np.sum(w[inds])
        })
        
   
    def neg_sharpe(
        w: np.ndarray
    ) -> float:
 
        pr = pf.portfolio_return(
            weights = w, 
            returns = er
        )
 
        vol = pf.portfolio_volatility(
            weights = w, 
            covmat = cov
        )
 
        vol = max(vol, 1e-12)
 
        return -(pr - riskfree_rate) / vol
  
    res = minimize(
        neg_sharpe, 
        init_guess, 
        method = "SLSQP",
        bounds = bnds, 
        constraints = constraints, 
        options = {"disp": False}
    )
 
    return res.x

@align_by_ticker
def msr_sortino(
    riskfree_rate: float,
    er: pd.Series,
    weekly_ret: pd.DataFrame,
    scores: pd.Series,
    ticker_ind: pd.Series,
    ticker_sec: pd.Series,
    bnd_h: float,
    bnd_l: float,
    max_industry_pct: float = 0.1,
    max_sector_pct: float = 0.15
) -> np.ndarray:
    """
    Maximises the Sortino ratio: (portfolio return - riskfree rate) / downside deviation.
    """
 
    n = er.shape[0]
 
    init_guess = np.zeros(n)
 
    nonzero_assets = np.where(er != 0)[0]
 
    if len(nonzero_assets) > 0:
        
        init_guess[nonzero_assets] = 1 / len(nonzero_assets)
 
    bnds = [
        generate_bounds_for_asset(
            bnd_h = bnd_h.iloc[i], 
            bnd_l = bnd_l.iloc[i], 
            er = er.iloc[i], 
            score = scores.iloc[i]
        )
        for i in range(n)
    ]
 
    constraints = [{
        "type": "eq", 
        "fun": lambda w: np.sum(w) - 1
    }]
 
    for industry in ticker_ind.unique():
        
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs: max_industry_pct - np.sum(w[inds])
        })
        
    for sector in ticker_sec.unique():
        
        limit = sector_limits.get(sector, max_sector_pct)
        
        idxs = [i for i, tk in enumerate(er.index) if ticker_sec.loc[tk] == sector]
        
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs, 
            limit = limit: limit - np.sum(w[inds])
        })

 
    def neg_sortino(
        w: np.ndarray
    ) -> float:
 
        port_r = pf.portfolio_return(
            weights = w, 
            returns = er
        )
 
        dd = pf.portfolio_downside_deviation(
            weights = w, 
            returns = weekly_ret, 
            target = config.RF_PER_WEEK
        )
 
        dd = max(dd, 1e-12)
 
        return -(port_r - riskfree_rate) / dd
 
 
    res = minimize(
        neg_sortino, 
        init_guess, 
        method = "SLSQP",
        bounds = bnds, 
        constraints = constraints, 
        options = {"disp": False}
    )
 
    return res.x


def risk_contribution(
    w, 
    cov
    ):
    """
    Compute the risk contributions of the portfolio constituents.
    """
   
    total_portfolio_var = pf.portfolio_volatility(
        weights = w, 
        covmat = cov
    ) ** 2
   
    marginal_contrib = cov @ w
   
    risk_contrib = np.multiply(w, marginal_contrib) / total_portfolio_var
   
    return risk_contrib


def target_risk_contributions(
    target_risk, 
    cov, 
    scores, 
    er, 
    ticker_ind, 
    ticker_sec, 
    max_industry_pct,
    max_sector_pct, 
    bnd_h, 
    bnd_l
    ):
    """
    Returns weights such that the risk contributions are as close as possible to target_risk.
    """
   
    n = cov.shape[0]
   
    init_guess = np.ones(n) / n
   
    bnds = [
        generate_bounds_for_asset(
            bnd_h = bnd_h.iloc[i], 
            bnd_l = bnd_l.iloc[i], 
            er = er.iloc[i], 
            score = scores.iloc[i]
        )
        for i in range(n)
    ]
   
    constraints = [{
        "type": "eq", 
        "fun": lambda w: np.sum(w) - 1
    }]
   
    for industry in ticker_ind.unique():
        
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs: max_industry_pct - np.sum(w[inds])
        })

    for sector in ticker_sec.unique():
        
        limit = sector_limits.get(sector, max_sector_pct)
        
        idxs = [i for i, tk in enumerate(er.index) if ticker_sec.loc[tk] == sector]
        
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs, 
            limit = limit: limit - np.sum(w[inds])
        })

    def msd_risk(
        weights, 
        target_risk, 
        cov
    ):
      
        w_contribs = risk_contribution(
            w = weights, 
            cov = cov
        )
      
        return ((w_contribs - target_risk) ** 2).sum()
    
    weights = minimize(
        msd_risk, 
        init_guess,
        args = (target_risk, cov), 
        method = 'SLSQP',
        options = {'disp': False},
        constraints = constraints,
        bounds = bnds
    )
    
    return weights.x


def equal_risk_contributions(
    cov, 
    scores, 
    er, 
    ticker_ind, 
    ticker_sec, 
    max_industry_pct, 
    max_sector_pct, 
    bnd_h, 
    bnd_l
    ):
    """
    Returns the portfolio weights that equalise the risk contributions.
    """
    
    n = cov.shape[0]
    
    target_risk = np.repeat(1 / n, n)
    
    return target_risk_contributions(
        target_risk = target_risk, 
        cov = cov, 
        scores = scores, 
        er = er, 
        ticker_ind = ticker_ind, 
        ticker_sec = ticker_sec, 
        max_industry_pct = max_industry_pct, 
        max_sector_pct = max_sector_pct, 
        bnd_h = bnd_h, 
        bnd_l = bnd_l
    )

@align_by_ticker
def MIR(
    benchmark: float,
    benchmark_weekly_ret: pd.Series,
    er: pd.Series,
    scores: pd.Series,
    er_hist: pd.DataFrame,
    ticker_ind: pd.Series,
    ticker_sec: pd.Series,
    bnd_h: float,
    bnd_l: float,
    max_industry_pct: float = 0.1,
    max_sector_pct: float = 0.15
) -> np.ndarray:
    """
    Maximises the Information Ratio: (average portfolio return - benchmark) / tracking error.
    """
    
    n = er.shape[0]
    
    init_guess = np.zeros(n)
    
    nonzero_assets = np.where(er != 0)[0]
    
    if len(nonzero_assets) > 0:
        
        init_guess[nonzero_assets] = 1 / len(nonzero_assets)
    
    bnds = [
        generate_bounds_for_asset(
            bnd_h = bnd_h.iloc[i], 
            bnd_l = bnd_l.iloc[i], 
            er = er.iloc[i], 
            score = scores.iloc[i]
        )
        for i in range(n)
    ]
    
    constraints = [{
        "type": "eq", 
        "fun": lambda w: np.sum(w) - 1
    }]
    
    for industry in ticker_ind.unique():
        
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs: max_industry_pct - np.sum(w[inds])
        })

    for sector in ticker_sec.unique():
       
        limit = sector_limits.get(sector, max_sector_pct)
       
        idxs = [i for i, tk in enumerate(er.index) if ticker_sec.loc[tk] == sector]
       
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs, 
            limit = limit: limit - np.sum(w[inds])
        })

    def neg_info_ratio(
        w: np.ndarray
    ) -> float:
    
        port_series = pf.portfolio_return_robust(
            weights = w, 
            returns = er_hist
        )
    
        te = pf.tracking_error(
            r_a = benchmark_weekly_ret, 
            r_b = port_series
        )
    
        te = max(te, 1e-12)
    
        avg_p = pf.annualise_returns(
            ret_series = port_series, 
            periods_per_year = 52
        )
    
        return - (avg_p - benchmark) / te
    
    res = minimize(
        neg_info_ratio, 
        init_guess, 
        method = "SLSQP",
        bounds = bnds, 
        constraints = constraints, 
        options = {"disp": False}
    )
    
    return res.x

def black_litterman_weights(
    tickers: pd.Index,
    comb_rets: pd.Series,
    comb_std: pd.Series,
    cov_prior: pd.DataFrame,
    mcap: pd.Series,
    score: pd.Series,
    ticker_ind: pd.Series,
    ticker_sec: pd.Series,
    bnd_h: float, 
    bnd_l: float = None,
    max_industry_pct: float = 0.1,
    max_sector_pct: float = 0.15,
    delta: float = 2.5,
    tau: float = 0.02
) -> pd.Series:
    """
    Compute long-only Black–Litterman portfolio weights.

    – comb_rets: your vector of views (Q)
    – comb_std: standard errors of those views (se → ω = diag(se²))
    – cov_prior: Σ (prior covariance)
    – mcap: market caps → w_prior = mcap / mcap.sum()
    – score: a screening score; any asset with score<=0 is zeroed out
    – delta, tau: BL parameters

    Returns
    -------
    pd.Series of weights summing to 1, with no shorts, and zeroed when comb_rets<=0 or score<=0.
    """
    
    mask = (score > 0) & (comb_rets > 0)

    total_mcap = mcap[mask].sum()
    
    if total_mcap == 0:
    
        raise ValueError("Nothing passes the >0 screen")
    
    w_prior = pd.Series(0.0, index=comb_std.index)
    
    w_prior[mask] = mcap[mask] / total_mcap

    P = pd.DataFrame(np.eye(len(tickers)), index=comb_std.index, columns=comb_std.index)
    
    Q = comb_rets.copy()
    
    omega = pd.DataFrame(np.diag(comb_std**2), index=comb_std.index, columns=comb_std.index)
    
    mu_bl, sigma_bl = black_litterman(
        w_prior = w_prior,
        sigma_prior = cov_prior,
        p = P, 
        q = Q, 
        omega = omega,
        delta = delta, 
        tau = tau
    )

    n = len(tickers)
    
    init_guess = np.zeros(n)
    
    init_guess[mask.values] = 1.0 / mask.sum()
    
    bnds = [
        generate_bounds_for_asset(
            bnd_h = bnd_h.iloc[i], 
            bnd_l = bnd_l.iloc[i], 
            er = comb_rets.iloc[i], 
            score = score.iloc[i]
        )
        for i in range(n)
    ]
 
    constraints = [{
        "type": "eq", 
        "fun": lambda w: np.sum(w) - 1
    }]
 
    for industry in ticker_ind.unique():
       
        idxs = [i for i, tk in enumerate(comb_rets.index) if ticker_ind.loc[tk] == industry]
       
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs: max_industry_pct - np.sum(w[inds])
        })
        
    for sector in ticker_sec.unique():
        
        limit = sector_limits.get(sector, max_sector_pct)
        
        idxs = [i for i, tk in enumerate(comb_rets.index) if ticker_sec.loc[tk] == sector]
        
        constraints.append({
            "type": "ineq",
            "fun": lambda w, 
            inds = idxs, 
            limit = limit: limit - np.sum(w[inds])
        })
    
    def neg_sharpe(
        w: np.ndarray
    ) -> float:
 
        pr = pf.portfolio_return(
            weights = w, 
            returns = mu_bl
        )
 
        vol = pf.portfolio_volatility(
            weights = w, 
            covmat = sigma_bl
        )
 
        vol = max(vol, 1e-12)
 
        return - (pr - config.RF) / vol
    
    res = minimize(
        neg_sharpe, 
        init_guess, 
        method = "SLSQP",
        bounds = bnds, 
        constraints = constraints, 
        options = {"disp": False}
    )

    w_bl = pd.Series(res.x, index=tickers, name='BL Weight')
    
    return w_bl, mu_bl, sigma_bl

@align_by_ticker
def comb_port_init_w(
    riskfree_rate: float,
    er: pd.Series,
    cov: np.ndarray,
    weekly_ret_1y: pd.DataFrame,
    ticker_ind: pd.Series,
    ticker_sec: pd.Series,
    bnd_h: pd.Series,
    bnd_l: pd.Series,
    w_msr: pd.Series,
    w_sortino: pd.Series,
    w_mir: pd.Series,
    w_bl: pd.Series,
    w_msp: pd.Series,
    mu_bl: pd.Series,
    sigma_bl: pd.DataFrame,
    max_industry_pct: float = 0.1,
    max_sector_pct: float = 0.15,
    tickers: pd.Index = None,
    gamma: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    sample_size: int = 2000,
    random_state: int = 42
) -> np.ndarray:
    """
    Builds a convex combination of MSR, Sortino, MIR and BL weights, maximizing:
    scaled_msr + scaled_sortino + scaled_bl - scaled_mir_deviations
    Uses empirical CDF transform for scaling.
    """

    def empirical_cdf_transform(
        vals: List[float]
    ):
        
        sorted_vals = np.sort(vals)
       
        def cdf(
            x: float
        ) -> float:
            
            return np.searchsorted(sorted_vals, x, side='right') / len(sorted_vals)
        
        return np.vectorize(cdf)

    W = np.column_stack([w_msr, w_sortino, w_mir, w_bl, w_msp])

    if random_state is not None:
        
        np.random.seed(random_state)
    
    alphas = np.random.dirichlet([1, 1, 1, 1, 1], size = sample_size)

    sharpe_vals = []
    sortino_vals = []
    mir_pen_vals = []
    bl_reward_vals = []
    msp_pen_vals = []

    for α in alphas:
        
        w = W.dot(α)
        
        ret = pf.portfolio_return(
            weights = w, 
            returns = er
        )
        
        vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = cov
            ), 1e-12
        )
        
        dd = max(
            pf.portfolio_downside_deviation(
                weights = w, 
                returns = weekly_ret_1y, 
                target = config.RF_PER_WEEK
            ), 1e-12
        )
        
        bl_ret = pf.portfolio_return(
            weights = w,
            returns = mu_bl
        )
        
        bl_vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = sigma_bl
            ), 1e-12
        )
        
        sharpe_vals.append((ret - riskfree_rate) / vol)
        
        sortino_vals.append((ret - riskfree_rate) / dd)
        
        mir_pen_vals.append(np.sum((w - w_mir)**2))

        bl_reward_vals.append((bl_ret - riskfree_rate) / bl_vol)
        
        msp_pen_vals.append(np.sum((w - w_msp)**2))

    F_sharpe = empirical_cdf_transform(
        vals = sharpe_vals
    )
    
    F_sortino = empirical_cdf_transform(
        vals = sortino_vals
    )
    
    F_bl = empirical_cdf_transform(
        vals = bl_reward_vals
    )
    
    F_mir = empirical_cdf_transform(
        vals = mir_pen_vals
    )
    
    F_msp = empirical_cdf_transform(
        vals = msp_pen_vals
    )

    γ_s, γ_so, γ_pm, γ_bl, γ_sp = gamma

    def neg_obj(
        α: np.ndarray
    ) -> float:
       
        w = W.dot(α)
       
        ret = pf.portfolio_return(
            weights = w, 
            returns = er
        )
       
        vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = cov
            ), 1e-12
        )
       
        dd = max(
            pf.portfolio_downside_deviation(
                weights = w, 
                returns = weekly_ret_1y, 
                target = config.RF_PER_WEEK
            ), 1e-12
        )
        
        bl_ret = pf.portfolio_return(
            weights = w,
            returns = mu_bl
        )
        
        bl_vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = sigma_bl
            ), 1e-12
        )

        sharpe = (ret - riskfree_rate) / vol
       
        sortino = (ret - riskfree_rate) / dd
       
        mir_pen = np.sum((w - w_mir) ** 2)
       
        bl_reward = (bl_ret - riskfree_rate) / bl_vol
        
        msp_pen = np.sum((w - w_msp) ** 2)

        s_scaled = F_sharpe(sharpe)
        so_scaled = F_sortino(sortino)
        bl_scaled = F_bl(bl_reward)
        mir_scaled = F_mir(mir_pen)
        msp_scaled = F_msp(msp_pen)

        return - (γ_s * s_scaled + γ_so * so_scaled + γ_bl * bl_scaled) + (γ_pm * mir_scaled + γ_sp * msp_scaled)

    cons = [{
        "type": "eq", 
        "fun": lambda α: np.sum(α) - 1
    }]
    
    for i in range(len(tickers)):
        
        cons.append({
            "type": "ineq",
            "fun": lambda α, 
            i = i: (W.dot(α))[i] - bnd_l.iloc[i]
        })
        
        cons.append({
            "type": "ineq",
            "fun": lambda α, 
            i = i: bnd_h.iloc[i] - (W.dot(α))[i]
        })

    for industry in ticker_ind.unique():
        
        inds = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        
        cons.append({
            "type": "ineq",
            "fun": lambda α, 
            inds = inds: max_industry_pct - np.sum(W.dot(α)[inds])
        })

    for sector in ticker_sec.unique():
        
        limit = sector_limits.get(sector, max_sector_pct)
        
        inds = [i for i, tk in enumerate(er.index) if ticker_sec.loc[tk] == sector]
        
        cons.append({
            "type": "ineq",
            "fun": lambda α, 
            inds = inds, 
            limit = limit: limit - np.sum(W.dot(α)[inds])
        })

    k = W.shape[1]            
    
    bounds_α = [(0.0, 1.0)] * k

    init = np.full(k, 1.0 / k)
    
    res = minimize(
        neg_obj,
        init,
        method = 'SLSQP',
        bounds = bounds_α,
        constraints = cons,
        options = {'disp': True,  "maxiter": 4000}
    )
    
    if not res.success:
        
        raise RuntimeError("comb_port: optimization failed to converge")

    α_opt = res.x
    
    return W.dot(α_opt)

@align_by_ticker
def comb_port(
    riskfree_rate: float,
    er: pd.Series,
    cov: np.ndarray,
    weekly_ret_1y: pd.DataFrame,
    last_5_year_weekly_rets: pd.DataFrame,
    benchmark: float,
    last_year_benchmark_weekly_ret: pd.Series,
    scores: pd.Series,
    ticker_ind: pd.Series,
    ticker_sec: pd.Series,
    bnd_h: pd.Series,
    bnd_l: pd.Series,
    tickers: pd.Index,
    comb_std: pd.Series,
    sigma_prior: pd.DataFrame,
    mcap: pd.Series,
    w_msr: pd.Series = None,
    w_sortino: pd.Series = None,
    w_mir: pd.Series = None,
    w_bl: pd.Series = None,
    w_msp: pd.Series = None,
    mu_bl: pd.Series = None,
    sigma_bl: pd.DataFrame = None,
    max_industry_pct: float = 0.1,
    max_sector_pct: float = 0.15,
    sample_size: int = 2000,
    random_state: int = 42,
    gamma: tuple = (1, 1, 1, 1, 1)
) -> np.ndarray:
    """
    Scale each of [Sharpe, Sortino, BL-Sharpe, MIR-penalty, MSP-penalty]
    to unit sample–std so that a unit weight on each (γ=1,1,…1) means “equal impact.”
    """

    n = len(er)
    
    if w_msr is None:
        
        w_msr = msr(
            riskfree_rate = riskfree_rate,
            er = er,
            cov = cov,
            scores = scores,
            ticker_ind = ticker_ind,
            ticker_sec = ticker_sec,
            bnd_h = bnd_h,
            bnd_l = bnd_l,
            max_industry_pct = max_industry_pct,
            max_sector_pct = max_sector_pct
        )
    
    if w_sortino is None:

        w_sortino = msr_sortino(
            riskfree_rate = riskfree_rate,
            er = er,
            weekly_ret = weekly_ret_1y,
            scores = scores,
            ticker_ind = ticker_ind,
            ticker_sec = ticker_sec,
            bnd_h = bnd_h,
            bnd_l = bnd_l,
            max_industry_pct = max_industry_pct,
            max_sector_pct = max_sector_pct
        )
    
    if w_mir is None:
        
        w_mir = MIR(
            benchmark = benchmark,
            benchmark_weekly_ret = last_year_benchmark_weekly_ret,
            er = er,
            scores = scores,
            er_hist = weekly_ret_1y,
            ticker_ind = ticker_ind,
            ticker_sec = ticker_sec,
            bnd_h = bnd_h,
            bnd_l = bnd_l,
            max_industry_pct = max_industry_pct,
            max_sector_pct = max_sector_pct
        )
    if w_msp is None:
        
        w_msp = msp(
            scores = scores,
            er = er,
            cov = cov,
            weekly_ret = weekly_ret_1y,
            level = 5.0,
            ticker_ind = ticker_ind,
            ticker_sec = ticker_sec,
            bnd_h = bnd_h,
            bnd_l = bnd_l,
            max_industry_pct = max_industry_pct,
            max_sector_pct = max_sector_pct
        )
    
    if w_bl is None or mu_bl is None or sigma_bl is None:
    
        w_bl, mu_bl, sigma_bl = black_litterman_weights(
            tickers = tickers,
            comb_rets = er,
            comb_std = comb_std,
            cov_prior = sigma_prior,
            mcap = mcap, 
            score = scores,
            ticker_ind = ticker_ind,
            ticker_sec = ticker_sec,
            bnd_h=bnd_h,
            bnd_l=bnd_l,
            max_industry_pct=max_industry_pct,
            max_sector_pct=max_sector_pct
        )
    
    weights_mat = np.vstack([
        w_msr.values,
        w_sortino.values,
        w_mir.values,
        w_msp.values,
        w_bl.values        
    ])
    
    min_weight = weights_mat.min(axis=0)   
    max_weight = weights_mat.max(axis=0)
    
    rng = np.random.default_rng(random_state)
 
    W_rand = rng.random((sample_size, n))
 
    W_rand /= W_rand.sum(axis=1, keepdims=True)

    sharpe_samps = np.empty(sample_size)
    sortino_samps = np.empty(sample_size)
    bl_samps = np.empty(sample_size)
    mir_samps = np.empty(sample_size)
    msp_samps = np.empty(sample_size)

    for j, w in enumerate(W_rand):

        ret = pf.portfolio_return(
            weights = w, 
            returns = er
        )
       
        vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = cov
            ), 1e-12
        )

        dd = max(
            pf.portfolio_downside_deviation(
                weights = w, 
                returns = weekly_ret_1y, 
                target = config.RF_PER_WEEK
            ), 1e-12
        )

        bl_ret = pf.portfolio_return(
            weights = w, 
            returns = mu_bl
        )

        bl_vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = sigma_bl
            ), 1e-12
        )

        sharpe_samps[j] = (ret - riskfree_rate) / vol
     
        sortino_samps[j] = (ret - riskfree_rate) / dd
     
        bl_samps[j] = (bl_ret - riskfree_rate) / bl_vol
     
        mir_samps[j] = np.sum((w - w_mir) ** 2)
     
        msp_samps[j] = np.sum((w - w_msp) ** 2)


    scale_s = 1.0 / (sharpe_samps.std() + 1e-12)
    scale_so = 1.0 / (sortino_samps.std() + 1e-12)
    scale_bl = 1.0 / (bl_samps.std() + 1e-12)
    scale_mir = 1.0 / (mir_samps.std() + 1e-12)
    scale_msp = 1.0 / (msp_samps.std() + 1e-12)

    γ_s, γ_so, γ_bl, γ_pm, γ_sp = gamma

    bounds = [(float(bnd_l[i]), float(bnd_h[i])) for i in range(n)]

    cons = [{
        "type": "eq", 
        "fun": lambda w: np.sum(w) - 1.0
    }]

    for industry in ticker_ind.unique():

        idxs = [i for i, t in enumerate(er.index) if ticker_ind.loc[t] == industry]

        cons.append({
            "type":"ineq",
            "fun": lambda w, 
            inds = idxs: max_industry_pct - np.sum(w[inds])
        })

    for sector in ticker_sec.unique():
      
        limit = sector_limits.get(sector, max_sector_pct)
      
        idxs = [i for i, t in enumerate(er.index) if ticker_sec.loc[t]==sector]
      
        cons.append({
            "type":"ineq",
            "fun": lambda w, 
            inds = idxs, 
            lim = limit: lim - np.sum(w[inds])
        })

    
    for i in range(n):
        
        lo = min_weight[i]
        hi = max_weight[i]

        cons.append({"type": "ineq",
                    "fun": lambda w, 
                    i = i, 
                    lo = lo: w[i] - lo})   
        
        cons.append({"type": "ineq",
                    "fun": lambda w, 
                    i = i,
                    hi = hi: hi - w[i]}) 
    
    if (min_weight > max_weight).any():
        
        raise ValueError("Some min_weight > max_weight — check your individual models")

    def neg_obj(
        w: np.ndarray
    ) -> float:
        
        ret = pf.portfolio_return(
            weights = w, 
            returns = er
        )
        
        vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = cov
            ), 1e-12
        )
        
        dd = max(
            pf.portfolio_downside_deviation(
                weights = w, 
                returns = weekly_ret_1y, 
                target = config.RF_PER_WEEK
            ), 1e-12
        )
        
        bl_ret = pf.portfolio_return(
            weights = w, 
            returns = mu_bl
        )
        
        bl_vol = max(
            pf.portfolio_volatility(
                weights = w, 
                covmat = sigma_bl
            ), 1e-12
        )

        sharpe  = (ret - riskfree_rate) / vol * scale_s
        
        sortino = (ret - riskfree_rate) / dd * scale_so
        
        bl_sharpe = (bl_ret - riskfree_rate) / bl_vol * scale_bl

        mir_pen = np.sum((w - w_mir)**2) * scale_mir
        
        msp_pen = np.sum((w - w_msp)**2) * scale_msp

        return - (γ_s * sharpe + γ_so * sortino + γ_bl * bl_sharpe) + (γ_pm * mir_pen + γ_sp * msp_pen)

    init = comb_port_init_w(
        riskfree_rate = riskfree_rate,
        er = er,
        cov = cov,
        weekly_ret_1y = weekly_ret_1y,
        ticker_ind = ticker_ind,
        ticker_sec = ticker_sec,
        bnd_h = bnd_h,
        bnd_l = bnd_l,
        w_msr = w_msr,
        w_sortino = w_sortino,
        w_mir = w_mir,
        w_bl = w_bl,
        w_msp = w_msp,
        mu_bl = mu_bl,
        sigma_bl = sigma_bl,
        tickers = tickers
    )

    res = minimize(
        neg_obj,
        init,
        method = "SLSQP",
        bounds = bounds,
        constraints = cons,
        options = {"disp": True, "maxiter": 5000}
    )

    if not res.success:

        raise RuntimeError(f"comb_port failed to converge: {res.message}")

    return res.x


@align_by_ticker
def msp(
    scores: pd.Series,
    er: pd.Series,
    cov: np.ndarray,
    weekly_ret: pd.DataFrame,
    level: float,
    ticker_ind: pd.Series,
    ticker_sec: pd.Series,
    bnd_h: pd.Series,
    bnd_l: pd.Series,
    max_industry_pct: float = 0.1,
    max_sector_pct: float = 0.15
) -> np.ndarray:
    """
    Maximises the score-per-CVaR measure: (w @ scores) / historical CVaR at `level`%.
    - scores: Analyst scores for each asset
    - er: Forecast returns (used for bounds & init guess)
    - weekly_ret: Historical returns DataFrame (for CVaR)
    - level: Tail level for CVaR (e.g. 5.0 for 5%)
    - ticker_ind: Industry mapping
    - ticker_sec: Sector mapping
    - bnd_h, bnd_l: Upper/lower weight bounds per asset
    """
    
    n = scores.shape[0]

    init_guess = np.zeros(n)

    nonzero = np.where((scores > 0) & (er > 0))[0]

    if len(nonzero) > 0:

        init_guess[nonzero] = 1.0 / len(nonzero)

    bnds = [
        generate_bounds_for_asset(
            bnd_h = bnd_h.iloc[i],
            bnd_l = bnd_l.iloc[i],
            er = er.iloc[i],
            score = scores.iloc[i]
        )
        for i in range(n)
    ]

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    for industry in ticker_ind.unique():
        
        inds = [i for i, t in enumerate(scores.index) if ticker_ind.loc[t] == industry]
      
        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=inds: max_industry_pct - np.sum(w[inds])
        })

    for sector in ticker_sec.unique():
       
        limit = sector_limits.get(sector, max_sector_pct)
       
        inds = [i for i, t in enumerate(scores.index) if ticker_sec.loc[t] == sector]

        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=inds, limit=limit: limit - np.sum(w[inds])
        })

    def neg_score_cvar(w: np.ndarray) -> float:

        pr = pf.portfolio_return(
            weights = w, 
            returns = er
        )
        
        vol = pf.portfolio_volatility(
            weights = w, 
            covmat = cov
        )
        
        port_rets = pf.portfolio_return_robust(
            weights = w, 
            returns = weekly_ret
        )
        
        skew = pf.skewness(
            r = port_rets
        )
        
        kurt = pf.kurtosis(
            r = port_rets
        )
 
        vol = max(vol, 1e-12)
        
        tail_risk = pf.port_pred_cvar(
            r_pred = pr, 
            std_pred = vol,
            skew = skew,
            kurt = kurt,
            level = level,
        )

        tail_risk = max(tail_risk, 1e-12)

        return -((w @ scores) / tail_risk)

    res = minimize(
        neg_score_cvar,
        init_guess,
        method = "SLSQP",
        bounds = bnds,
        constraints = constraints,
        options = {"disp": False}
    )

    return res.x
