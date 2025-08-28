"""
Implements Black–Litterman portfolio calculations: implied returns, proportional prior, and the posterior mean/covariance update
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from typing import Tuple


def implied_returns(
    delta: float, 
    sigma: pd.DataFrame, 
    w: pd.Series
) -> pd.Series:
    """
    Compute Black–Litterman "implied equilibrium" expected returns.

    The reverse-optimisation identity under a mean–variance investor with risk-aversion
    parameter δ is
   
        π = δ · Σ · w,
   
    where
   
    • π (vector, n×1) are the equilibrium expected excess returns,
   
    • Σ (n×n) is the covariance matrix of asset excess returns,
   
    • w (vector, n×1) are the market (e.g., cap-weighted) portfolio weights,
   
    • δ > 0 is the representative investor’s risk aversion.

    This function returns π.

    Parameters
    ----------
    delta : float
        Risk-aversion coefficient δ. Typical equity values are in the 2–3 range.
    sigma : pandas.DataFrame
        Covariance matrix Σ of asset excess returns. Index/columns must align with `w`.
    w : pandas.Series
        Market weights vector w (sum to 1), indexed like `sigma` columns.

    Returns
    -------
    pandas.Series
        Equilibrium (implied) excess returns π = δ Σ w, named 'Implied Returns'.
    """
  
    ir = delta * sigma.dot(w).squeeze()
  
    ir.name = 'Implied Returns'
  
    return ir


def proportional_prior(
    sigma: pd.DataFrame, 
    tau: float, 
    p: pd.DataFrame, 
    confidence: float,
    omega_floor: float = 0.0
) -> pd.DataFrame:
    """
    Build a diagonal Black–Litterman view-error covariance Ω (Omega) proportional to
    the uncertainty of the views implied by the prior covariance.

    For a view matrix P (k×n), prior covariance scaled by τ (tau) as τΣ, and a
    scalar "confidence" ∈ (0,∞], the canonical "proportional view uncertainty"
    construction sets
    
        \tilde{Ω} = P (τΣ / confidence) P',
   
    and then uses only its diagonal elements as Ω = diag(diag( \tilde{Ω} )). An
    optional floor `omega_floor` ensures a minimum view variance, i.e.
       
        Ω_jj = max( \tilde{Ω}_jj , omega_floor ).

    Interpretation:
    
    • Smaller Ω ⇒ higher confidence in the corresponding view.
    
    • Increasing `confidence` (>1) reduces Ω (more confidence), while
        `confidence` < 1 inflates Ω (less confidence).

    Parameters
    ----------
    sigma : pandas.DataFrame
        Prior covariance Σ of assets (n×n).
    tau : float
        Scalar ∈ (0,1], the BL prior-uncertainty scaling; τΣ is the covariance of the
        prior on equilibrium returns π.
    p : pandas.DataFrame
        View‐exposure matrix P (k×n). Each row encodes a linear view on assets.
    confidence : float
        Global view confidence scaling. Larger ⇒ smaller Ω. Use 1 for the standard
        proportional prior.
    omega_floor : float, default 0.0
        Minimal variance per view to avoid degeneracy.

    Returns
    -------
    pandas.DataFrame
        Diagonal Ω (k×k), indexed by the view rows of `p`.
    """

    helit_omega = p.dot(tau * sigma / confidence).dot(p.T)
    
    diag_vals = np.diag(helit_omega.values).astype(float)
    
    if omega_floor > 0.0:
        
        diag_vals = np.maximum(diag_vals, omega_floor)

    return pd.DataFrame(np.diag(diag_vals),
                        index = p.index,
                        columns = p.index)


def black_litterman(
    w_prior: pd.Series,
    sigma_prior: pd.DataFrame,
    p: pd.DataFrame,
    q: pd.Series,
    omega: pd.DataFrame = None,
    delta: float = 2.5,
    tau: float = 0.02,
    prior: pd.Series = None,
    confidence: float = 1,
    omega_floor: float = 0.0
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Black–Litterman posterior returns and covariance.

    Given:
   
    • Prior on equilibrium expected returns: π (n×1). If not supplied, set by
   
        π = δ Σ w_prior.
   
    • Prior covariance of returns: Σ (n×n).
   
    • View system: P (k×n), Q (k×1) with view errors ε ~ N(0, Ω) independent of π.
   
    • Prior uncertainty scaling: τ (so Var[π] = τΣ).

    The Gaussian Bayesian update yields the posterior mean μ_BL and covariance Σ_BL:

    Posterior mean:
   
        μ_BL = π + τ Σ P' ( P τ Σ P' + Ω )^{-1} ( Q − P π )

    Posterior covariance:

        Σ_BL = Σ + τ Σ − τ Σ P' ( P τ Σ P' + Ω )^{-1} P τ Σ

    (Equivalent forms factor τΣ on both sides.)

    This function implements the above with:
    • `omega` supplied or, if None, constructed by `proportional_prior`.
    • `prior` supplied (π) or, if None, computed via `implied_returns(delta, Σ, w_prior)`.

    Parameters
    ----------
    w_prior : pandas.Series
        Prior (market) weights vector w (n×1) used to back out π if `prior` is None.
    sigma_prior : pandas.DataFrame
        Prior covariance Σ (n×n).
    p : pandas.DataFrame
        View matrix P (k×n). Each row defines a linear combination of assets.
    q : pandas.Series
        View targets Q (k×1) in the same units as returns (excess returns).
    omega : pandas.DataFrame, optional
        View error covariance Ω (k×k). If None, a diagonal Ω is built via
        `proportional_prior(sigma_prior, tau, p, confidence, omega_floor)`.
    delta : float, default 2.5
        Risk-aversion δ for implied returns if `prior` is None.
    tau : float, default 0.02
        Prior uncertainty scaling τ (typical values ≈ 0.01–0.05).
    prior : pandas.Series, optional
        Explicit equilibrium prior π. If provided, overrides `delta`/`w_prior`.
    confidence : float, default 1
        Global confidence scaling for Ω when `omega is None`.
    omega_floor : float, default 0.0
        Minimal diagonal element for Ω when constructed.

    Returns
    -------
    (mu_bl, sigma_bl) : tuple[pandas.Series, pandas.DataFrame]
        μ_BL as a Series indexed by assets and Σ_BL as a DataFrame.

    Notes
    -----
    Dimensions: assets n, views k. All pandas objects must be index-aligned:
    Σ and μ_BL over assets; P has asset columns and view rows; Q, Ω over views.
    """


    if omega is None:
        
        omega = proportional_prior(
            sigma = sigma_prior, 
            tau = tau, 
            p = p, 
            confidence = confidence,
            omega_floor = omega_floor
        ) 
   
    if prior is None:
        
        pi = implied_returns(
            delta = delta, 
            sigma = sigma_prior, 
            w = w_prior
        )
   
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
    
    mu_bl_series = pd.Series(mu_bl.ravel(), index = pi.index, name = 'Posterior Return')

    sigma_bl_array = sigma_prior.values + sigma_scaled_array - np.dot(np.dot(np.dot(sigma_scaled_array, p_array.T), A_inv), np.dot(p_array, sigma_scaled_array))
    
    sigma_bl = pd.DataFrame(sigma_bl_array, index = sigma_prior.index, columns = sigma_prior.columns)

    return mu_bl_series, sigma_bl
