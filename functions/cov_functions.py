import numpy as np
import pandas as pd
import statsmodels.stats.moment_helpers as mh

def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Plain sample covariance.
    """
    return returns.cov()


def cc_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Constant-correlation model. Compute average correlation among all pairs,
    then build the covariance matrix from that correlation and each asset's std dev.
    """
    corr = returns.corr()
    n = corr.shape[0]
    rho_bar = (corr.values.sum() - n) / (n * (n - 1))
    ccor = np.full(corr.shape, rho_bar)
    np.fill_diagonal(ccor, 1.0)
    std_devs = returns.std()
    return mh.corr2cov(ccor, std_devs)


def pred_covariance(pred_vol: pd.Series, corr: pd.DataFrame) -> np.ndarray:
    """
    Given predicted volatilities and a correlation matrix,
    build a covariance matrix as outer(vol, vol) * averageCorr.
    """
    return np.outer(pred_vol, pred_vol) * corr.values


def shrinkage_covariance(
    returns: pd.DataFrame,
    comb_std: pd.Series = None,
    ret_corr: pd.DataFrame = None,
    delta: float = 1/3,
    alpha: float = 1/3
) -> np.ndarray:
    """
    Shrinks between:
      - sample covariance
      - constant correlation covariance
      - predicted covariance (using comb_std + ret_corr)
    Weighted by delta (constant corr), alpha (predicted), (1-delta-alpha) (sample).
    """
    prior = cc_covariance(returns) 
    sample = sample_covariance(returns)
    pred = pred_covariance(comb_std, ret_corr) if alpha != 0 else 0
    return (delta * prior + (1 - delta - alpha) * sample + alpha * pred) * 52
