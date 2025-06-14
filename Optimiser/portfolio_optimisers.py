"""
Implements optimisation routines for max Sharpe, Sortino, information ratio, equal‑risk and combination strategies with bounds.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy.optimize import minimize
import portfolio_functions as pf

def generate_bounds_for_asset(
    bnd_h: float,
    bnd_l: float,
    er: float,
    score: float,
    buy: bool,
    sell: bool,
    max_all: float,
) -> Tuple[float, float]:
    """
    Decide (min, max) weight for each asset based on:
      - combination score
      - buy/sell signals
      - expected return, etc.
    """
    
    buy_min = min(bnd_l * 2, max_all)
    
    buy_max = min(bnd_h * 2, max_all)

    if score <= 0 or sell or er <= 0:
        return (0, 0)
   # elif buy:
    #    return (buy_min, buy_max)
    else:
        return (bnd_l, bnd_h)
    

def msr_2(
    riskfree_rate: float,
    er: pd.Series,
    cov: np.ndarray,
    max_all: float,
    buy: List[bool],
    sell: List[bool],
    scores: pd.Series,
    ticker_ind: pd.Series,
    bnd_h: float,
    bnd_l: float,
    max_industry_pct: float = 0.1
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
        generate_bounds_for_asset(bnd_h.iloc[i], bnd_l.iloc[i], er.iloc[i], scores.iloc[i], buy[i], sell[i], max_all)
        for i in range(n)
    ]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for industry in ticker_ind.unique():
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=idxs: max_industry_pct - np.sum(w[inds])
        })
        
    def neg_sharpe(w: np.ndarray) -> float:
        pr = pf.portfolio_return(w, er)
        vol = pf.portfolio_volatility(w, cov)
        vol = max(vol, 1e-12)
        return -(pr - riskfree_rate) / vol
    res = minimize(neg_sharpe, init_guess, method="SLSQP",
                   bounds=bnds, constraints=constraints, options={"disp": False})
    return res.x


def msr_sortino(
    riskfree_rate: float,
    er: pd.Series,
    weekly_ret: pd.DataFrame,
    max_all: float,
    buy: List[bool],
    sell: List[bool],
    scores: pd.Series,
    ticker_ind: pd.Series,
    bnd_h: float,
    bnd_l: float,
    max_industry_pct: float = 0.1
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
        generate_bounds_for_asset(bnd_h.iloc[i], bnd_l.iloc[i], er.iloc[i], scores.iloc[i], buy[i], sell[i], max_all)
        for i in range(n)
    ]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for industry in ticker_ind.unique():
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=idxs: max_industry_pct - np.sum(w[inds])
        })

    def neg_sortino(w: np.ndarray) -> float:
        port_r = pf.portfolio_return(w, er)
        dd = pf.portfolio_downside_deviation(w, weekly_ret, target=riskfree_rate)
        dd = max(dd, 1e-12)
        return -(port_r - riskfree_rate) / dd
    res = minimize(neg_sortino, init_guess, method="SLSQP",
                   bounds=bnds, constraints=constraints, options={"disp": False})
    return res.x


def risk_contribution(w, cov):
    """
    Compute the risk contributions of the portfolio constituents.
    """
    total_portfolio_var = pf.portfolio_volatility(w, cov)**2
    marginal_contrib = cov @ w
    risk_contrib = np.multiply(w, marginal_contrib) / total_portfolio_var
    return risk_contrib


def target_risk_contributions(target_risk, cov, scores, er, buy, sell, max_all, ticker_ind, max_industry_pct, bnd_h, bnd_l):
    """
    Returns weights such that the risk contributions are as close as possible to target_risk.
    """
    n = cov.shape[0]
    init_guess = np.ones(n) / n
    bnds = [
        generate_bounds_for_asset(bnd_h.iloc[i], bnd_l.iloc[i], er.iloc[i], scores.iloc[i], buy[i], sell[i], max_all)
        for i in range(n)
    ]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for industry in ticker_ind.unique():
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=idxs: max_industry_pct - np.sum(w[inds])
        })

    def msd_risk(weights, target_risk, cov):
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs - target_risk) ** 2).sum()
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=constraints,
                       bounds=bnds)
    return weights.x


def equal_risk_contributions(cov, scores, er, buy, sell, max_all, ticker_ind, max_industry_pct, bnd_h, bnd_l):
    """
    Returns the portfolio weights that equalise the risk contributions.
    """
    n = cov.shape[0]
    target_risk = np.repeat(1 / n, n)
    return target_risk_contributions(target_risk, cov, scores, er, buy, sell, max_all, ticker_ind, max_industry_pct, bnd_h, bnd_l)


def MIR(
    benchmark: float,
    benchmark_weekly_ret: pd.Series,
    er: pd.Series,
    buy: List[bool],
    sell: List[bool],
    scores: pd.Series,
    max_all: float,
    er_hist: pd.DataFrame,
    ticker_ind: pd.Series,
    bnd_h: float,
    bnd_l: float,
    max_industry_pct: float = 0.1
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
        generate_bounds_for_asset(bnd_h.iloc[i], bnd_l.iloc[i], er.iloc[i], scores.iloc[i], buy[i], sell[i], max_all)
        for i in range(n)
    ]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for industry in ticker_ind.unique():
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=idxs: max_industry_pct - np.sum(w[inds])
        })

    def neg_info_ratio(w: np.ndarray) -> float:
        port_series = er_hist.dot(w)
        te = pf.tracking_error(benchmark_weekly_ret, port_series)
        te = max(te, 1e-12)
        avg_p = pf.annualise_returns(port_series, 52)
        return -(avg_p - benchmark) / te
    res = minimize(neg_info_ratio, init_guess, method="SLSQP",
                   bounds=bnds, constraints=constraints, options={"disp": False})
    return res.x


def comb_port(
    riskfree_rate: float,
    er: pd.Series,
    cov: np.ndarray,
    max_all: float,
    buy: List[bool],
    sell: List[bool],
    scores: pd.Series,
    ticker_ind: pd.Series,
    weekly_ret: pd.DataFrame,
    bnd_h: float,
    bnd_l: float,
    benchmark: float,
    benchmark_weekly_ret: pd.Series,
    max_industry_pct: float = 0.1
) -> np.ndarray:
    """
    Builds a 'Combination' portfolio using a weighted average of MSR and Sortino portfolios
    plus an extra penalty based on risk contributions.
    """
    n = er.shape[0]
    init_guess = np.zeros(n)
    nonzero_assets = np.where(er != 0)[0]
    if len(nonzero_assets) > 0:
        init_guess[nonzero_assets] = 1 / len(nonzero_assets)
    max_score = scores.max()
    bnds = [
        generate_bounds_for_asset(bnd_h.iloc[i], bnd_l.iloc[i], er.iloc[i], scores.iloc[i], buy[i], sell[i], max_all)
        for i in range(n)
    ]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for industry in ticker_ind.unique():
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=idxs: max_industry_pct - np.sum(w[inds])
        })
        
    w_mir = MIR(benchmark, benchmark_weekly_ret, er, buy, sell, scores, max_all, weekly_ret, ticker_ind, bnd_h, bnd_l)
    def neg_comb(w: np.ndarray) -> float:
        pr = pf.portfolio_return(w, er)
        vol = pf.portfolio_volatility(w, cov)
        dd = pf.portfolio_downside_deviation(w, weekly_ret, target=riskfree_rate)
        dd = max(dd, 1e-12)
        vol = max(vol, 1e-12)
        penalty = np.sum((abs(w - w_mir) * (max_score / ((scores.values + 1e-12))) ** 2))
        ret = pr - riskfree_rate
        sign = np.sign(ret)
        return -(sign * ret ** 2/ (vol * dd)) + penalty
    
    res = minimize(neg_comb, init_guess, method="SLSQP",
                   bounds=bnds, constraints=constraints, options={"disp": False})

    return res.x


def msp(
    scores: pd.Series,
    cov: np.ndarray,
    buy: List[bool],
    sell: List[bool],
    er: pd.Series,
    max_all: float,
    ticker_ind: pd.Series,
    bnd_h: float,
    bnd_l: float,
    max_industry_pct: float = 0.1
) -> np.ndarray:
    """
    Maximises the score-per-risk measure: (w @ scores) / volatility.
    """
    n = er.shape[0]
    init_guess = np.zeros(n)
    bnds = [
        generate_bounds_for_asset(bnd_h.iloc[i], bnd_l.iloc[i], er.iloc[i], scores.iloc[i], buy[i], sell[i], max_all)
        for i in range(n)
    ]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for industry in ticker_ind.unique():
        idxs = [i for i, tk in enumerate(er.index) if ticker_ind.loc[tk] == industry]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, inds=idxs: max_industry_pct - np.sum(w[inds])
        })

    def neg_score_risk(w: np.ndarray) -> float:
        vol = pf.portfolio_volatility(w, cov)
        vol = max(vol, 1e-12)
        return -((w @ scores) / vol)
    res = minimize(neg_score_risk, init_guess, method="SLSQP",
                   bounds=bnds, constraints=constraints, options={"disp": False})
    return res.x

