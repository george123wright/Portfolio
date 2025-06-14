import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from scipy.stats import norm


def portfolio_return(weights: np.ndarray, returns) -> Any:
    """
    Computes portfolio returns.
    If `returns` is a Series, returns a scalar dot product.
    If `returns` is a DataFrame, returns a Series of portfolio returns (row-wise dot product).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.dot(weights)
    elif isinstance(returns, pd.Series):
        return float(weights @ returns)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def portfolio_volatility(weights: np.ndarray, covmat: np.ndarray) -> float:
    """ 
    sqrt( w^T * Cov * w ).
    """
    return float(np.sqrt(weights @ covmat @ weights))


def portfolio_downside_deviation(weights: np.ndarray, returns: pd.DataFrame, target: float = 0) -> float:
    """
    Compute downside semideviation if portfolio returns are below target.
    """
    port_returns = returns.dot(weights)
    below_target = port_returns[port_returns < target]
    if below_target.empty:
        return 0.0
    return float(np.sqrt(np.mean((below_target - target) ** 2)))


def tracking_error(r_a: pd.Series, r_b: pd.Series) -> float:
    """ 
    sqrt( sum( (r_a - r_b)^2 ) ). 
    """
    return float(np.sqrt(((r_a - r_b) ** 2).mean()))


def portfolio_tracking_error(weights: np.ndarray, ref_r: pd.Series, rets: pd.DataFrame) -> float:
    """ 
    Tracking error between reference returns and portfolio returns.
    """
    port_rets = rets.dot(weights)
    return tracking_error(ref_r, port_rets)


def port_beta(weights: np.ndarray, beta: pd.Series) -> float:
    """ 
    Weighted sum of betas => portfolio beta.
    """
    return float(weights @ beta)


def compute_treynor_ratio(port_ret: float, rf: float, port_beta_val: float) -> float:
    """ 
    (port_ret - rf) / beta. Returns nan if beta is 0.
    """
    if port_beta_val == 0:
        return float('nan')
    return (port_ret - rf) / port_beta_val


def port_score(weights: np.ndarray, score: pd.Series) -> float:
    """ 
    Weighted sum of a 'score' metric.
    """
    return float(weights @ score)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualised Sharpe ratio of a set of returns.
    """
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualise_returns(excess_ret, periods_per_year)
    ann_vol = annualise_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def annualise_vol(r, periods_per_year):
    """
    Annualises the volatility of a set of returns.
    We should infer the periods per year, but that is left as an exercise :-)
    """
    return r.std() * (periods_per_year ** 0.5)


def drawdown(return_series: pd.Series):
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


def skewness(r):
    """
    Computes the skewness of the supplied Series or DataFrame.
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r**3


def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a Series or DataFrame.
    If modified is True, returns the modified VaR using the Cornish-Fisher expansion.
    """
    z = norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1) * s / 6 +
             (z**3 - 3 * z) * (k - 3) / 24 -
             (2 * z**3 - 5 * z) * (s**2) / 36
            )
    return -(r.mean() + z * r.std(ddof=0))


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of a Series or DataFrame.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def kurtosis(r):
    """
    Computes the kurtosis of the supplied Series or DataFrame.
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r**4


def annualise_returns(ret_series: pd.Series, periods_per_year: int) -> float:
    """
    Annualises returns: (product of (1 + r))^(periods_per_year / number_of_periods) - 1.
    """
    total_periods = len(ret_series)
    if total_periods <= 1:
        return 0.0
    cum = (1 + ret_series).prod()
    return cum ** (periods_per_year / total_periods) - 1


def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
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
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
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
    sim_paths = gbm(n_years=1, n_scenarios=scenarios, mu=mu, sigma=sigma,
                    steps_per_year=steps, s_0=s0)
    final_prices = sim_paths.loc[steps]
    final_returns = (final_prices / s0) - 1
    q25_l = final_returns.quantile(0.245)
    q25_h = final_returns.quantile(0.255)
    q75_l = final_returns.quantile(0.745)
    q75 = final_returns.quantile(0.75)
    q75_h = final_returns.quantile(0.755)
    stats = {
        "mean_returns": final_returns.mean(),
        "loss_percentage": 100 * (final_returns < 0).sum() / len(final_returns),
        "mean_loss_amount": final_returns[final_returns < 0].mean(),
        "mean_gain_amount": final_returns[final_returns >= 0].mean(),
        "variance": (final_prices / s0).var(),
        "lower_quartile": final_returns[(final_returns >= q25_l) & (final_returns <= q25_h)].mean(),
        "upper_quartile": final_returns[(final_returns >= q75_l) & (final_returns <= q75_h)].mean(),
        "upper_returns_mean": final_returns[final_returns >= q75].mean(),
        "min_return": float(final_prices.min() / s0) - 1,
        "max_return": float(final_prices.max() / s0) - 1
    }
    return stats


def simulate_and_report(name: str, wts: np.ndarray, comb_rets: float, bear_rets: float, bull_rets: float, vol: float, vol_ann: float,
                            comb_score: pd.Series, weekly_rets: pd.DataFrame, rf: float, beta: float, benchmark_weekly_rets) -> Dict[str, Any]:
        
    port_rets = portfolio_return(wts, comb_rets)
    
    port_bear_rets = portfolio_return(wts, bear_rets)
    
    port_bull_rets = portfolio_return(wts, bull_rets)
    
    stats = simulate_portfolio_stats(port_rets, vol_ann, steps=252, s0=100.0, scenarios=1000000)
    
    b_val = port_beta(wts, beta)
    
    treynor = compute_treynor_ratio(port_rets, rf, b_val)
    
    score_val = port_score(wts, comb_score)
    
    portfolio_rets_hist = portfolio_return(wts, weekly_rets)
    
    sr_pred = (port_rets - rf) / vol_ann
    
    ann_sr_hist = sharpe_ratio(portfolio_rets_hist, riskfree_rate=rf, periods_per_year=52)
    
    dd = drawdown(portfolio_rets_hist)["Drawdown"].min()
    
    skew_val = skewness(portfolio_rets_hist)
    
    kurt_val = kurtosis(portfolio_rets_hist)
    
    cf_var5 = var_gaussian(portfolio_rets_hist, modified=True)
    
    hist_cvar5 = cvar_historic(portfolio_rets_hist)
    
    summary = {
        "Average Returns": f"{port_rets * 100:.2f}%",
        "Average Bear Returns": f"{port_bear_rets * 100:.2f}%",
        "Average Bull Returns": f"{port_bull_rets * 100:.2f}%",
        "Daily Volatility": vol,
        "Annual Volatility": vol_ann,
        "Scenario Average Returns": f"{stats['mean_returns'] * 100:.2f}%",
        "Scenario Loss Incurred": f"{stats['loss_percentage']:.2f}%",
        "Scenario Average Loss": f"{stats['mean_loss_amount'] * -100:.2f}%",
        "Scenario Average Gain": f"{stats['mean_gain_amount'] * 100:.2f}%",
        "Scenario Variance": stats["variance"],
        "Scenario Lower Quartile": f"{stats['lower_quartile'] * 100:.2f}%",
        "Scenario Upper Quartile": f"{stats['upper_quartile'] * 100:.2f}%",
        "Scenario Upper Quartile Mean": f"{stats['upper_returns_mean'] * 100:.2f}%",
        "Scenario Min Returns": f"{stats['min_return'] * 100:.2f}%",
        "Scenario Max Returns": f"{stats['max_return'] * 100:.2f}%",
        "Portfolio Beta": f"{b_val:.4f}",
        "Treynor Ratio": f"{treynor:.4f}",
        "Portfolio Score": f"{score_val:.2f}",
        "Portfolio Tracking Error": tracking_error(benchmark_weekly_rets, portfolio_rets_hist),
        "Skewness": skew_val,
        "Kurtosis": kurt_val,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio (Predicted)": sr_pred,
        "Sharpe Hist Ratio": ann_sr_hist,
        "Historic Annual Returs": annualise_returns(portfolio_rets_hist, 52),
        "Max Drawdown": dd
    }
        
    return summary
