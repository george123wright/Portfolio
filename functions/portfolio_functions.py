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

