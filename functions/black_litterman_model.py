"""
Implements Black–Litterman portfolio calculations: implied returns, proportional prior, and the posterior mean/covariance update
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv


def implied_returns(delta: float, sigma: pd.DataFrame, w: pd.Series) -> pd.Series:
    """
    Compute the implied returns given delta, covariance sigma, and weights w.
    """
    ir = delta * sigma.dot(w).squeeze()
    ir.name = 'Implied Returns'
    return ir


def proportional_prior(sigma: pd.DataFrame, tau: float, p: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a diagonal omega matrix from sigma, tau and the view matrix p.
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),
                        index=p.index, columns=p.index)


def black_litterman(w_prior: pd.Series,
                    sigma_prior: pd.DataFrame,
                    p: pd.DataFrame,
                    q: pd.Series,
                    omega: pd.DataFrame = None,
                    delta: float = 2.5,
                    tau: float = 0.02,
                    prior: pd.Series = None) -> (pd.Series, pd.DataFrame):
    """
    Compute the posterior (Black–Litterman) returns and covariance.
    """
                        
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
   
    if prior is None:
        pi = implied_returns(delta, sigma_prior, w_prior)
   
    else:
        pi = prior
   
    sigma_prior_scaled = tau * sigma_prior

    pi_array = pi.values.reshape(-1, 1)
    Q_array = q.values.reshape(-1, 1)
                        
    p_array = p.values
    sigma_scaled_array = sigma_prior_scaled.values
    omega_array = omega.values

    A = np.dot(np.dot(p_array, sigma_scaled_array), p_array.T) + omega_array
    A_inv = inv(A)

    step = np.dot(np.dot(np.dot(sigma_scaled_array, p_array.T), A_inv), (Q_array - np.dot(p_array, pi_array)))

    mu_bl = pi_array + step
    mu_bl_series = pd.Series(mu_bl.ravel(), index=pi.index, name='Posterior Return')

    sigma_bl_array = sigma_prior.values + sigma_scaled_array - np.dot(np.dot(np.dot(sigma_scaled_array, p_array.T), A_inv), np.dot(p_array, sigma_scaled_array))
    sigma_bl = pd.DataFrame(sigma_bl_array, index=sigma_prior.index, columns=sigma_prior.columns)

    return mu_bl_series, sigma_bl
