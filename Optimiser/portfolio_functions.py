"""
Contains portfolio utility functions—returns, volatility, downside deviation, risk metrics and Monte‑Carlo simulation helpers.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from scipy.stats import norm
import statsmodels.api as sm
from numbers import Number
import config


def portfolio_return(
    weights: np.ndarray, 
    returns
) -> Any:
    """
    Computes portfolio returns.
    If `returns` is a Series, returns a scalar dot product.
    If `returns` is a DataFrame, returns a Series of portfolio returns (row-wise dot product).
    """
   
    if isinstance(returns, pd.DataFrame):
        
        return returns.dot(weights)
   
    elif isinstance(returns, pd.Series):
       
        return float(weights @ returns)
    
    elif isinstance(returns, np.ndarray):
        
        return float(weights @ returns)

    if isinstance(returns, Number):
    
        if len(weights) == 1:
        
            return float(weights[0] * returns)
        
        else:
            
            raise TypeError(
                f"Cannot compute a multi‑asset portfolio return from a single scalar ({returns})"
            )
    
    else:
        
         raise TypeError("Expected returns to be a Series or DataFrame")


def portfolio_volatility(
    weights: np.ndarray, 
    covmat: np.ndarray
) -> float:
    """ 
    sqrt( w^T * Cov * w ).
    """
   
    return float(np.sqrt(weights @ covmat @ weights))


def portfolio_downside_deviation(
    weights: np.ndarray, 
    returns: pd.DataFrame, 
    target: float = config.RF_PER_WEEK
) -> float:
    """
    Compute downside semideviation if portfolio returns are below target.
    """
   
    port_returns = portfolio_return_robust(
        weights = weights, 
        returns = returns
    )
    
    below_target = port_returns[port_returns < target]
   
    if below_target.empty:
        
        return 0.0
   
    return float(np.sqrt(np.mean((below_target - target) ** 2)))


def tracking_error(
    r_a: pd.Series, 
    r_b: pd.Series
) -> float:
    """ 
    sqrt( sum( (r_a - r_b)^2 ) ). 
    """
   
    return float(np.sqrt(((r_a - r_b) ** 2).mean()))


def portfolio_tracking_error(
    weights: np.ndarray, 
    ref_r: pd.Series, 
    rets: pd.DataFrame
)-> float:
    """ 
    Tracking error between reference returns and portfolio returns.
    """
   
    port_rets = portfolio_return_robust(
        weights = weights, 
        returns = rets
    )
   
    return tracking_error(
        r_a = ref_r, 
        r_b = port_rets
    )


def port_beta(
    weights: np.ndarray, 
    beta: pd.Series
) -> float:
    """ 
    Weighted sum of betas => portfolio beta.
    """
   
    return float(weights @ beta)


def compute_treynor_ratio(
    port_ret: float, 
    rf: float, 
    port_beta_val: float
) -> float:
    """ 
    (port_ret - rf) / beta. Returns nan if beta is 0.
    """
   
    if port_beta_val == 0:
        
        return float('nan')
   
    return (port_ret - rf) / port_beta_val


def port_score(
    weights: np.ndarray, 
    score: pd.Series
) -> float:
    """ 
    Weighted sum of a 'score' metric.
    """
   
    return float(weights @ score)


def sharpe_ratio(r, periods_per_year):
    """
    Computes the annualised Sharpe ratio of a set of returns.
    """
      
    excess_ret = r - config.RF_PER_WEEK
   
    ann_ex_ret = annualise_returns(
        ret_series = excess_ret, 
        periods_per_year = periods_per_year
    )
    
    ann_vol = annualise_vol(
        r = r, 
        periods_per_year = periods_per_year
    )
   
    return ann_ex_ret / ann_vol


def annualise_vol(
    r, 
    periods_per_year
):
    """
    Annualises the volatility of a set of returns.
    We should infer the periods per year, but that is left as an exercise :-)
    """
   
    return r.std() * (periods_per_year ** 0.5)


def drawdown(
    return_series: pd.Series
):
    """
    Takes a time series of asset returns and returns a DataFrame with columns for
    the wealth index, previous peaks, and the percentage drawdown.
    """
   
    wealth_index = 1000 * (1 + return_series).cumprod()
   
    previous_peaks = wealth_index.cummax()
   
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
   
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


def skewness(
    r
):
    """
    Computes the skewness of the supplied Series or DataFrame.
    """
   
    demeaned_r = r - r.mean()
   
    sigma_r = r.std(ddof=0)
   
    exp = (demeaned_r ** 3).mean()
   
    return exp / sigma_r**3


def var_gaussian(
    r, 
    level = 5, 
    modified = False
):
    """
    Returns the parametric Gaussian VaR of a Series or DataFrame.
    If modified is True, returns the modified VaR using the Cornish-Fisher expansion.
    """
   
    z = norm.ppf(level / 100)
   
    if modified:
       
        s = skewness(
            r = r
        )
       
        k = kurtosis(
            r = r
        )
       
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
            )
    
    return -(r.mean() + z * r.std(ddof=0))


def var_historic(
    r, 
    level=5
):
    """
    Returns the historic Value at Risk at a specified level.
    """
    
    if isinstance(r, pd.DataFrame):
        
        return r.aggregate(var_historic, level=level)
    
    elif isinstance(r, pd.Series):
       
        return -np.percentile(r, level)
    
    else:
       
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(
    r, 
    level=5
):
    """
    Computes the Conditional VaR of a Series or DataFrame.
    """
    if isinstance(r, pd.Series):
        
        var = var_historic(
            r = r, 
            level = level
        )

        tail = r[r <= -var]
        
        if tail.empty:
            
            return 0.0     
        
        return -tail.mean()
    
    elif isinstance(r, pd.DataFrame):
       
        return r.aggregate(cvar_historic, level=level)
   
    else:
       
        raise TypeError("Expected r to be a Series or DataFrame")


def port_pred_cvar(
    r_pred, 
    std_pred,
    skew,
    kurt, 
    level = 5.0,
    periods = 52
) -> float:
    
    alpha = level / 100
    
    z  = norm.ppf(alpha)
        
    z_cf = (
        z
        + (z ** 2 - 1) * skew / 6
        + (z ** 3 - 3 * z) * (kurt - 3) / 24
        - (2 * z ** 3 - 5 * z) * (skew ** 2) / 36
    )
    
    r_pred_per_period = (1 + r_pred)**(1/periods) - 1
    
    std_pred_per_period = std_pred / np.sqrt(periods)
    
    z_pdf = norm.pdf(z_cf)
    
    cvar = r_pred_per_period + (std_pred_per_period * z_pdf / alpha)
    
    return cvar


def kurtosis(
    r
):
    """
    Computes the kurtosis of the supplied Series or DataFrame.
    """
    
    demeaned_r = r - r.mean()
    
    sigma_r = r.std(ddof=0)
    
    exp = (demeaned_r ** 4).mean()
    
    return exp / sigma_r ** 4


def IR(
    w, 
    er, 
    te, 
    benchmark_ret
) -> float:
    
    port_series = portfolio_return_robust(
        weights = w,
        returns = er
    )
    
    te = max(te, 1e-12)
    
    avg_p = annualise_returns(
        ret_series = port_series, 
        periods_per_year = 52
    )
    
    bench = annualise_returns(
        ret_series = benchmark_ret, 
        periods_per_year = 52
    )
    
    return (avg_p - bench) / te


def annualise_returns(
    ret_series: pd.Series, 
    periods_per_year: int
) -> float:
    """
    Annualises returns: (product of (1 + r))^(periods_per_year / number_of_periods) - 1.
    """
    
    total_periods = len(ret_series)
    
    if total_periods <= 1:
        
        return 0.0
    
    cum = (1 + ret_series).prod()
    
    return cum ** (periods_per_year / total_periods) - 1


def ulcer_index(
    return_series: pd.Series
) -> float:
    """
    Ulcer Index = sqrt( (1/N) * sum(drawdown_t^2) )
    where drawdown_t is the percent drawdown at time t.
    """
    
    wealth = (1 + return_series).cumprod()
    
    peak = wealth.cummax()
    
    drawdowns = (wealth - peak) / peak
    
    ui = np.sqrt((drawdowns ** 2).mean())
    
    return float(ui)


def cdar(
    r: pd.Series, 
    level: float = 5.0
) -> float:
    
    wealth = (1 + r).cumprod()
    
    peak = wealth.cummax()
    
    drawdowns = (wealth - peak) / peak

    thresh = drawdowns.quantile(level/100)
    
    worst = drawdowns[drawdowns <= thresh]

    if worst.empty:
       
        return 0.0
    
    return float(worst.mean())


def jensen_alpha_r2(
    port_rets: pd.Series, 
    bench_ann_ret: float, 
    port_ret: float, 
    bench_rets: pd.Series, 
    rf: float, 
    periods_per_year: int
) -> Tuple[float, float]:
    """
    Regress excess portfolio returns on excess benchmark returns:
      (r_p - rf) = alpha + beta * (r_b - rf) + ε
    Returns (alpha_annualised, R-squared).
    """
    
    df = pd.concat([port_rets, bench_rets], axis=1).dropna()
    df.columns = ['p', 'b']
        
    y = df['p'] - config.RF_PER_WEEK
    X = df['b'] - config.RF_PER_WEEK
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    alpha_per_period = model.params['const']
    
    alpha_ann = (1 + alpha_per_period) ** periods_per_year - 1
    
    pred_alpha = port_ret - (config.RF + (model.params['b'] * (bench_ann_ret - config.RF)))
    
    return float(alpha_ann), float(model.rsquared), pred_alpha


def capture_ratios(
    port_rets: pd.Series, 
    bench_rets: pd.Series
) -> Dict[str, float]:
    """
    Upside Capture = avg(r_p | r_b > 0) / avg(r_b | r_b > 0)
    Downside Capture = avg(r_p | r_b < 0) / avg(r_b | r_b < 0)
    """
    
    df = pd.concat([port_rets, bench_rets], axis=1).dropna()
    df.columns = ['p', 'b']
    
    up = df[df['b'] > 0]
    down = df[df['b'] < 0]
    
    up_cap = up['p'].mean() / up['b'].mean() if not up.empty else np.nan
    down_cap = down['p'].mean() / down['b'].mean() if not down.empty else np.nan
    
    return {'Upside Capture': float(up_cap), 'Downside Capture': float(down_cap)}


def portfolio_return_robust(
    weights: np.ndarray,
    returns: pd.DataFrame
) -> pd.Series:
    """
    Row-wise dot product of `returns` and `weights`, but on each row
    we (a) drop any NaNs, (b) re-normalize the surviving weights to sum to 1,
    then (c) compute the weighted sum.
    """
    
    w = np.asarray(weights)
    
    if isinstance(returns, pd.Series):
    
        returns = returns.to_frame()

    def row_ret(
        row: pd.Series
    ) -> float:
        
        valid = row.notna().to_numpy()      
        
        if not valid.any():
           
            return np.nan                
        
        w_sub = w[valid]
        
        w_sub = w_sub / w_sub.sum()         
        
        return float((row.to_numpy()[valid] * w_sub).sum())

    return returns.apply(row_ret, axis=1)


def sortino_ratio(
    returns: pd.Series,
    riskfree_rate: float,
    periods_per_year: int,
    target: float = config.RF_PER_WEEK,
    er: float = None
) -> float:
    """
    Annualized Sortino ratio.

    - returns: periodic returns (e.g. weekly or daily)
    - riskfree_rate: ANNUAL risk-free rate (e.g. 0.02 for 2%)
    - periods_per_year: e.g. 252 for daily, 52 for weekly
    - target: optional per-period threshold; if None, use RF converted to per-period
    """

    downside = returns[returns < target]
    
    if downside.empty:
       
        return np.nan

    semidev = np.sqrt(np.mean((downside - target) ** 2))

    ann_downside = semidev * np.sqrt(periods_per_year)

    if er is None:
       
        ann_return = annualise_returns(
            ret_series = returns, 
            periods_per_year = periods_per_year
        )
       
        ann_excess = ann_return - riskfree_rate
    
    else:
        ann_excess = er - riskfree_rate

    return ann_excess / ann_downside


def calmar_ratio(
    returns: pd.Series, 
    periods_per_year: int
) -> float:
    """
    Calmar Ratio: CAGR divided by maximum drawdown (in absolute value).
    """
   
    cagr = annualise_returns(
        ret_series = returns, 
        periods_per_year = periods_per_year
    )
   
    max_dd = drawdown(
        return_series = returns
    )["Drawdown"].min()
   
    return float(cagr / abs(max_dd)) if max_dd < 0 else np.nan


def omega_ratio(
    returns: pd.Series, 
    threshold: float = 0.0
) -> float:
    """
    Omega Ratio: ratio of cumulative gains above threshold to cumulative losses below.
    """
   
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns <= threshold]).sum()
   
    if losses == 0:
        
        return np.inf
   
    return float(gains / losses)


def modigliani_ratio(
    returns: pd.Series, 
    bench_returns: pd.Series, 
    riskfree_rate: float, 
    periods_per_year: int
) -> float:
    """
    Modigliani–Modigliani (M²): risk-adjusted return, scaled to benchmark volatility.
    """
  
    sr = sharpe_ratio(
        r = returns, 
        periods_per_year = periods_per_year
    )
  
    bench_vol_ann = annualise_vol(
        r = bench_returns, 
        periods_per_year = periods_per_year
    )
  
    return float(riskfree_rate + sr * bench_vol_ann)


def mar_ratio(
    returns: pd.Series, 
    periods_per_year: int
) -> float:
    """
    MAR Ratio: annualised return divided by max drawdown (abs).
    """
   
    cagr = annualise_returns(
        ret_series = returns, 
        periods_per_year = periods_per_year
    )
   
    max_dd = drawdown(
        return_series = returns
    )["Drawdown"].min()
   
    return float(cagr / abs(max_dd)) if max_dd < 0 else np.nan


def pain_index_and_ratio(
    returns: pd.Series, 
    riskfree_rate: float, 
    periods_per_year: int
) -> Tuple[float, float]:
    """
    Pain Index: average drawdown level.
    Pain Ratio: (CAGR - rf) / Pain Index.
    """

    dd = drawdown(
        return_series = returns
    )["Drawdown"]
    
    pi = float(-dd.mean())
    
    cagr = annualise_returns(
        ret_series = returns, 
        periods_per_year = periods_per_year
    )
    
    pr = float((cagr - riskfree_rate) / pi) if pi > 0 else np.nan
    
    return pi, pr


def tail_ratio(
    returns: pd.Series, 
    upper_q: float = 0.90, 
    lower_q: float = 0.10
) -> float:
    """
    Tail Ratio: ratio of the 90th percentile gain to the absolute 10th percentile loss.
    """
    
    up = returns.quantile(upper_q)
    down = returns.quantile(lower_q)
    
    if down >= 0:
        return np.inf
    
    return float(up / abs(down))


def raroc(
    returns: pd.Series, 
    riskfree_rate: float, 
    periods_per_year: int, 
    var_level: float = 5.0
) -> float:
    """
    RAROC: annualised excess return over economic capital (VaR at given level).
    """
    
    ann_return = annualise_returns(
        ret_series = returns, 
        periods_per_year = periods_per_year
    )
    
    excess = ann_return - riskfree_rate
    
    cap = var_historic(
        r = returns, 
        level = var_level
    )
    
    return float(excess / cap) if cap > 0 else np.nan


def percent_positive_and_streaks(
    returns: pd.Series
) -> Tuple[float, int, int]:
    """
    Returns: percent of positive periods, longest win streak, longest loss streak.
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


def gbm(
    n_years = 10, 
    n_scenarios = 1_000_000,
    mu = 0.07, 
    sigma = 0.15, 
    steps_per_year = 12, 
    s_0 = 100.0, 
    prices = True
):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
   
    dt = 1 / steps_per_year
   
    n_steps = int(n_years * steps_per_year) + 1
   
    rets_plus_1 = np.random.normal(
        loc = (1 + mu) ** dt, 
        scale = (sigma * np.sqrt(dt)), 
        size = (n_steps, n_scenarios)
    )
    
    rets_plus_1[0] = 1
   
    ret_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
   
    return ret_val


def simulate_portfolio_stats(
    mu: float,
    sigma: float,
    steps: int = 252,
    s0: float = 100.0,
    scenarios: int = 1_000_000
) -> Dict[str, Any]:
    """
    Simulate 1-year outcomes via GBM and gather summary statistics about the returns distribution.
    """
  
    sim_paths = gbm(
        n_years = 1, 
        n_scenarios = scenarios, 
        mu = mu, 
        sigma = sigma,
        steps_per_year = steps, 
        s_0 = s0
    )
  
    final_prices = sim_paths.loc[steps]
  
    final_returns = (final_prices / s0) - 1
  
    q25_l = final_returns.quantile(0.245)
    q25_h = final_returns.quantile(0.255)
  
    q75_l = final_returns.quantile(0.745)
    q75 = final_returns.quantile(0.75)
    q75_h = final_returns.quantile(0.755)
    
    p10 = final_returns.quantile(0.10)
    p90 = final_returns.quantile(0.90)
    
    scen_up_down = p90 / p10
  
    stats = {
        "mean_returns": final_returns.mean(),
        "loss_percentage": 100 * (final_returns < 0).sum() / len(final_returns),
        "mean_loss_amount": final_returns[final_returns < 0].mean(),
        "mean_gain_amount": final_returns[final_returns >= 0].mean(),
        "variance": (final_prices / s0).var(),
        "10th_percentile": p10,
        "lower_quartile": final_returns[(final_returns >= q25_l) & (final_returns <= q25_h)].mean(),
        "upper_quartile": final_returns[(final_returns >= q75_l) & (final_returns <= q75_h)].mean(),
        "90th_percentile": p90,
        "scenarios_up_down": scen_up_down,
        "upper_returns_mean": final_returns[final_returns >= q75].mean(),
        "min_return": float(final_prices.min() / s0) - 1,
        "max_return": float(final_prices.max() / s0) - 1
    }
  
    return stats


def simulate_and_report(
    name: str, 
    wts: np.ndarray, 
    comb_rets: float, 
    bear_rets: float, 
    bull_rets: float, 
    vol: float, 
    vol_ann: float,
    comb_score: pd.Series, 
    weekly_rets: pd.DataFrame, 
    rf: float, beta: float, 
    benchmark_weekly_rets: pd.Series, 
    benchmark_ann_ret: float,
    bl_ret: pd.Series,
    bl_cov: pd.DataFrame,
    sims: int = 1_000_000
) -> Dict[str, Any]:
    
    last_year_weekly_rets = weekly_rets.loc[weekly_rets.index >= pd.to_datetime(config.YEAR_AGO)]
    
    last_5y_weekly_rets = weekly_rets.loc[weekly_rets.index >= pd.to_datetime(config.FIVE_YEAR_AGO)]
    
    port_rets = portfolio_return(
        weights = wts, 
        returns = comb_rets
    )
    
    port_bear_rets = portfolio_return(
        weights = wts, 
        returns = bear_rets
    )
    
    port_bull_rets = portfolio_return(
        weights = wts, 
        returns = bull_rets
    )
    
    port_bl_rets = portfolio_return(
        weights = wts, 
        returns = bl_ret
    )
    
    port_bl_vol = portfolio_volatility(
        weights = wts, 
        covmat = bl_cov
    )
    
    stats = simulate_portfolio_stats(
        mu = port_rets, 
        sigma = vol_ann, 
        steps = 252, 
        s0 = 100.0, 
        scenarios = sims
    )
    
    b_val = port_beta(
        weights = wts, 
        beta = beta
    )
    
    treynor = compute_treynor_ratio(
        port_ret = port_rets, 
        rf = rf, 
        port_beta_val = b_val
    )
    
    score_val = port_score(
        weights = wts, 
        score = comb_score
    )
        
    portfolio_rets_hist = portfolio_return_robust(
        weights = wts, 
        returns = last_year_weekly_rets
    )
            
    portfolio_rets_5year_hist = portfolio_return_robust(
        weights = wts,
        returns = last_5y_weekly_rets
    )    
        
    sr_pred = (port_rets - rf) / vol_ann
    
    bl_sr = (port_bl_rets - rf) / port_bl_vol
    
    ann_sr_hist = sharpe_ratio(
        r = portfolio_rets_hist, 
        periods_per_year = 52
    )
    
    dd = drawdown(
        return_series = portfolio_rets_hist
    )["Drawdown"].min()
    
    skew_val = skewness(
        r = portfolio_rets_5year_hist
    )
    
    kurt_val = kurtosis(
        r = portfolio_rets_5year_hist
    )
    
    cf_var5 = var_gaussian(
        r = portfolio_rets_5year_hist,
        modified = True
    )
    
    hist_cvar5 = cvar_historic(
        r = portfolio_rets_5year_hist
    )
    
    pred_cvar = port_pred_cvar(
        r_pred = port_rets,
        std_pred = vol_ann,
        skew = skew_val,
        kurt = kurt_val,
        level = 5.0,
        periods = 52
    )
    
    te = tracking_error(
        r_a = benchmark_weekly_rets, 
        r_b = portfolio_rets_5year_hist
    )
    
    ir = IR(
        w = wts, 
        er = last_5y_weekly_rets, 
        te = te, 
        benchmark_ret = benchmark_weekly_rets
    )
    
    ui = ulcer_index(
        return_series = portfolio_rets_5year_hist
    )
    
    cd = cdar(
        r = portfolio_rets_5year_hist, 
    )
    
    sortino = sortino_ratio(
        returns = portfolio_rets_hist, 
        riskfree_rate = rf, 
        periods_per_year = 52, 
        target = config.RF_PER_WEEK, 
        er = port_rets
    )
    
    sortino_hist = sortino_ratio(
        returns = portfolio_rets_hist, 
        riskfree_rate = rf, 
        periods_per_year = 52
    )
    
    calmar = calmar_ratio(
        returns = portfolio_rets_5year_hist, 
        periods_per_year = 52
    )
    
    omega = omega_ratio(
        returns = portfolio_rets_5year_hist
    )
    
    m2 = modigliani_ratio(
        returns = portfolio_rets_5year_hist, 
        bench_returns = benchmark_weekly_rets, 
        riskfree_rate = rf, 
        periods_per_year = 52
    )
    
    mar = mar_ratio(
        returns = portfolio_rets_5year_hist, 
        periods_per_year = 52
    )
    
    pi, pr = pain_index_and_ratio(
        returns = portfolio_rets_5year_hist, 
        riskfree_rate = rf, 
        periods_per_year = 52
    )
    
    tail = tail_ratio(
        returns = portfolio_rets_5year_hist
    )
    
    raroc_val = raroc(
        returns = portfolio_rets_5year_hist, 
        riskfree_rate = rf, 
        periods_per_year = 52
    )
    
    pct_pos, win_streak, loss_streak = percent_positive_and_streaks(
        returns = portfolio_rets_5year_hist
    )

    alpha, r2, pred_alpha = jensen_alpha_r2(
        port_rets = portfolio_rets_hist, 
        bench_ann_ret = benchmark_ann_ret, 
        port_ret = port_rets, 
        bench_rets = benchmark_weekly_rets, 
        rf = rf, 
        periods_per_year = 52
    )

    caps = capture_ratios(
        port_rets = portfolio_rets_hist, 
        bench_rets = benchmark_weekly_rets
    )
    
    summary = {
        "Average Returns": port_rets,
        "Average Bear Returns": port_bear_rets,
        "Average Bull Returns": port_bull_rets,
        "BL Returns": port_bl_rets,
        "Weekly Volatility": vol,
        "Annual Volatility": vol_ann,
        "BL Volatility": port_bl_vol,
        "Scenario Average Returns": stats['mean_returns'],
        "Scenario Loss Incurred": stats['loss_percentage'],
        "Scenario Average Loss": stats['mean_loss_amount'],
        "Scenario Average Gain": stats['mean_gain_amount'],
        "Scenario Variance": stats["variance"],
        "Scenario 10th Percentile": stats['10th_percentile'],
        "Scenario Lower Quartile": stats['lower_quartile'],
        "Scenario Upper Quartile": stats['upper_quartile'],
        "Scenario 90th Percentile": stats['90th_percentile'],
        "Scenario Up/Down": stats['scenarios_up_down'],
        "Scenario Min Returns": stats['min_return'],
        "Scenario Max Returns": stats['max_return'],
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
        "MAR Ratio": mar,
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
        "Historic CVaR (5%)": hist_cvar5,
        "Predicted CVaR (5%)": pred_cvar,
        "Sharpe Ratio (Predicted)": sr_pred,
        "Sharpe Hist Ratio": ann_sr_hist,
        "Bl Sharpe Ratio": bl_sr,
        "Historic Annual Returns": annualise_returns(portfolio_rets_hist, 52),
        "Max Drawdown": dd,
        "Ulcer Index": ui,
        "Conditional Drawdown at Risk": cd,
        "Jensen's Alpha": alpha,
        "Predicted Alpha": pred_alpha,
        "R-squared": r2,
        "Upside Capture Ratio": caps.get('Upside Capture', np.nan),
        "Downside Capture Ratio": caps.get('Downside Capture', np.nan)
    }
        
    return summary


def report_ticker_metrics(
    tickers: List[str],
    weekly_rets: pd.DataFrame,
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
    sims: int = 100_000
) -> pd.DataFrame:
    """
    For each ticker in `tickers`, simulate and report portfolio metrics using a single-asset portfolio,
    then include each of the forecast-return model predictions from `forecast_file`.

    Returns a DataFrame with one row per ticker, containing:
      - All metrics from simulate_and_report
      - Model return predictions (one column per model)
    """
    
    n = len(tickers)
    
    results: Dict[str, Dict] = {}

    wts = [1]

    for t in tickers:
        
        print(f"Simulating {t}...")
        stock_df = weekly_rets[[t]].dropna()
                
        one_year = pd.to_datetime(config.YEAR_AGO)
        stock_df = stock_df.loc[ stock_df.index >= one_year ]
        bench_sr = benchmark_weekly_rets.loc[ benchmark_weekly_rets.index >= one_year ]
        vol_weekly = np.sqrt(weekly_cov.loc[t, t])
        
        common = stock_df.index.intersection( bench_sr.index )
        stock_df = stock_df.loc[common]
        bench_sr = bench_sr.loc[common]
        
        vol_annual = np.sqrt(ann_cov.loc[t, t])
                
        bl_cov_t = np.array([[bl_cov.loc[t, t]]])
        
        beta_t = np.array([beta.loc[t]])
        
        score_t = np.array([comb_score.loc[t]])
                        
        metrics = simulate_and_report(
            name = t,
            wts = wts,
            comb_rets = comb_rets.loc[t],
            bear_rets = bear_rets.loc[t],
            bull_rets = bull_rets.loc[t],
            vol = vol_weekly,
            vol_ann = vol_annual,
            comb_score = score_t,
            weekly_rets = stock_df,
            rf = rf,
            beta = beta_t,
            benchmark_weekly_rets = bench_sr,
            benchmark_ann_ret = benchmark_ann_ret,
            bl_ret = bl_ret.loc[t],
            bl_cov = bl_cov_t,
            sims = sims
        )
        results[t] = metrics

    metrics_df = pd.DataFrame.from_dict(results, orient='index')

    xls = pd.ExcelFile(forecast_file)
    
    model_sheets = {
        'Prophet Pred': 'Prophet',
        'Analyst Target': 'AnalystTarget',
        'Exponential Returns':'EMA',
        'Lin Reg Returns': 'LinReg',
        'DCF': 'DCF',
        'DCFE': 'DCFE',
        'Daily Returns': 'Daily',
        'RI': 'RI',
        'CAPM BL Pred': 'CAPM',
        'FF3 Pred': 'FF3',
        'FF5 Pred': 'FF5',
        'SARIMAX Monte Carlo': 'SARIMAX',
        'Rel Val Pred': 'RelVal'
    }
    
    model_returns: Dict[str, pd.Series] = {}
    
    for sheet, name in model_sheets.items():
        
        df = xls.parse(sheet, usecols=['Ticker', 'Returns'], index_col='Ticker')
        
        df.index = df.index.str.upper()
        
        model_returns[name] = df['Returns']

    ret_df = pd.DataFrame(model_returns).reindex(tickers)


    final_df = ret_df.join(metrics_df)

    final_df['Combined Return'] = comb_rets.reindex(tickers)

    return final_df
