from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import product

from factor_simulations import _circular_block_bootstrap


def _build_lagged(
    Y: np.ndarray,
    p: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct lagged design and target matrices for a VAR(p).

    Parameters
    ----------
    Y : np.ndarray
        Two-dimensional array of shape (T, k) containing k time series observed
        over T periods. Each column corresponds to a variable and each row to a
        time index.
    p : int
        VAR lag order. Must satisfy 1 ≤ p < T.

    Returns
    -------
    Xlag : np.ndarray
        Array of shape (T - p, k * p). Each row at time t (for t = p, …, T-1)
        concatenates the p lag vectors in reverse chronological order:
        [y_{t-1}', y_{t-2}', …, y_{t-p}'].
    Yp : np.ndarray
        Array of shape (T - p, k) containing the contemporaneous targets
        y_t aligned to the rows of Xlag.

    Notes
    -----
    The resulting pair (Xlag, Yp) provides the standard regression form for
    estimating a VAR(p):

        y_t = ∑_{ℓ=1}^p A_ℓ y_{t-ℓ} + u_t,

    with Xlag acting as the regressor matrix formed by stacked lags, and Yp
    the dependent variable matrix.
    """
   
    Tloc, kloc = Y.shape
    
    Xrows = []
    
    for tloc in range(p, Tloc):
        
        lags = [Y[tloc - ell] for ell in range(1, p + 1)]
        
        Xrows.append(np.concatenate(lags, axis=0))
   
    Xlag = np.asarray(Xrows)
   
    Yp = Y[p:]
   
    return Xlag, Yp


def _ols_ridge(
    X: np.ndarray, 
    y: np.ndarray, 
    D: np.ndarray, 
    m: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Solve a ridge (Tikhonov) regression with a general quadratic penalty centred at a prior mean.

    This routine computes the penalised least squares estimator b that minimises
    the objective

        (y - X b)'(y - X b) + (b - m)' D (b - m),

    where D is a symmetric positive semidefinite penalty matrix and m is a
    prior mean vector. When D is diagonal with positive entries, this is the
    usual ridge penalty but centred at m rather than zero.

    Parameters
    ----------
    X : np.ndarray
        Regressor matrix of shape (n, p). It may already include a column of
        ones if an intercept is desired.
    y : np.ndarray
        Target vector of shape (n,).
    D : np.ndarray
        Penalty (precision) matrix of shape (p, p). Larger diagonal entries
        impose stronger shrinkage towards the corresponding component of m.
    m : np.ndarray
        Prior mean vector of shape (p,). Acts as the centre of the penalty.

    Returns
    -------
    b : np.ndarray
        Penalised estimator of shape (p,), given by the closed form

            b = (X'X + D)^{-1} (X'y + D m).
   
    sigma2 : float
        Residual variance estimate computed as

            sigma2 = (e'e) / max(1, n - p),

        where e = y - X b.

    Bayesian Interpretation
    -----------------------
    The estimator is the posterior mode (MAP) of a Gaussian linear model with
    a conjugate Gaussian prior b ~ N(m, D^{-1}). The penalty matrix D acts as
    the prior precision. When D is diagonal, 1 / D_jj is the prior variance
    for coefficient j.
    """
    
    XtX = X.T @ X
   
    Xty = X.T @ y
   
    A = XtX + D
   
    b = np.linalg.solve(A, Xty + D @ m)
   
    resid = y - X @ b
   
    sigma2 = float((resid @ resid) / max(1, X.shape[0] - X.shape[1]))
   
    return b, sigma2


def fit_bvar_minnesota(
    macro_df: pd.DataFrame,
    p: int = 2,
    *,
    grid_lambda_overall: list[float] | None = None,
    grid_lambda_cross: list[float] | None = None,
    grid_lambda_lagdecay: list[float] | None = None,
    grid_lambda_const: list[float] | None = None,
    grid_prior_ownlag1_mean: list[float] | None = None,
    cv_min_train: int | None = None,   
    cv_max_folds: int | None = None,   
    innovation: str = "student_t",
    df_t: float = 7.0,
) -> dict:
    """
    Fit a Minnesota-style Bayesian VAR(p) via independent per-equation ridge
    regressions with a structured, scale-aware penalty, and select the
    hyperparameters by expanding-window, one-step-ahead cross-validation.

    Model
    -----
    For k variables collected in y_t ∈ ℝ^k, the VAR(p) is

        y_t = c + ∑_{ℓ=1}^p A_ℓ y_{t-ℓ} + u_t,     for t = p+1, …, T,

    where c ∈ ℝ^k is the intercept, A_ℓ ∈ ℝ^{k×k} are lag coefficient matrices,
    and u_t ∼ (0, Σ) are i.i.d. reduced-form innovations. Estimation proceeds
    equation-by-equation by penalised least squares (ridge) under a Minnesota-type
    structure.

    Minnesota Penalty Structure
    ---------------------------
    Let σ_j denote the scale of variable j, operationalised here as the sample
    standard deviation of Δy^j_t (first differences) on the estimation sample
    to mitigate non-stationary levels. For equation i and coefficient on the
    lag ℓ of regressor j, the shrinkage 'tightness' is defined as

        tight(i, j, ℓ) = λ_overall * (σ_i / σ_j) / (ℓ^{λ_lagdecay}) * κ(i, j),

    where κ(i, j) = 1 for own-lags (j = i) and κ(i, j) = λ_cross for cross-lags
    (j ≠ i). The corresponding diagonal entry of the penalty matrix D is

        D_{pos,pos} = [1 / max(tight, 1e-8)]^2,

    so a larger 'tightness' implies a weaker penalty (larger prior variance).
    
    The intercept uses

        D_{intercept,intercept} = (λ_const * σ_i)^2,

    which scales the prior precision of the constant by the equation’s scale.

    Prior Mean
    ----------
    The prior mean vector m is zero except for the first own-lag coefficient
    in each equation, which is set to prior_ownlag1_mean (a scalar applied
    identically across equations). This encodes the Minnesota belief that each
    series behaves approximately as a univariate AR(1) around its own first lag.

    Estimation
    ----------
    For each equation i, the coefficient vector b_i solves

        b_i = argmin_b (y_i − X b)'(y_i − X b) + (b − m)' D (b − m),

    with X = [1, Y_{t-1}', …, Y_{t-p}'] the regressor matrix including an intercept.
    This is equivalent to the MAP estimator under a Gaussian prior
    b ∼ N(m, D^{-1}). Residuals are centred before forming Σ̂ to remove any
    intercept drift:

        e_t = y_t − X_t b̂,      
        
        ē = (1 / (T_eff)) ∑_t e_t,
       
        Σ̂  = ((E − 1 ē')'(E − 1 ē')) / max(1, T_eff − (1 + k p)),

    where T_eff = T − p.

    Hyperparameter Selection (Expanding-Window CV)
    ----------------------------------------------
    The tuple (λ_overall, λ_cross, λ_lagdecay, λ_const, prior_ownlag1_mean)
    is chosen by minimising the one-step-ahead root mean squared error (RMSE)
    over an expanding-window backtest. For each fold index t ∈ {cv_min_train, …, T−1}:
     
      1. Fit the model on Y[:t].
     
      2. Form the one-step forecast
      
        ŷ_t = ĉ + ∑_{ℓ=1}^p Â_ℓ y_{t−ℓ}.
     
      3. Accumulate squared errors across all k variables.

    The final score is RMSE = sqrt(∑ errors^2 / (k × number_of_predictions)).

    Parameters
    ----------
    macro_df : pd.DataFrame
        Input panel with a DateTimeIndex (or orderable index) and k columns.
        Rows are time points; columns are variables. Missing rows are dropped.
    p : int, default 2
        VAR lag order.
    grid_lambda_overall, grid_lambda_cross, grid_lambda_lagdecay,
    grid_lambda_const, grid_prior_ownlag1_mean : list[float] or None
        Search grids for the respective hyperparameters. Reasonable defaults
        are provided for quarterly data if None.
    cv_min_train : int or None, default None
        Minimum number of initial observations in the expanding window.
        Defaults to max(40, 4 p), and is clipped to be at least p+5.
    cv_max_folds : int or None, default None
        If provided, caps the number of folds by sub-sampling evenly spaced
        fold indices to reduce computation.
    innovation : {"student_t", "gaussian", "bootstrap"}, default "student_t"
        Label stored in the returned model dict to guide simulation choice.
    df_t : float, default 7.0
        Degrees of freedom for Student-t innovations used at simulation time.

    Returns
    -------
    dict
        A dictionary with keys:
          - "A" : np.ndarray of shape (p, k, k), VAR lag coefficient matrices.
          - "c" : np.ndarray of shape (k,), intercept vector.
          - "Sigma" : np.ndarray of shape (k, k), centred residual covariance.
          - "resid" : np.ndarray of shape (T − p, k), uncentred residuals.
          - "resid_centered" : np.ndarray of shape (T − p, k), centred residuals.
          - "history" : np.ndarray of shape (p, k), last p observed rows of Y.
          - "columns" : list[str], column names of macro_df.
          - "index" : pd.Index, index of the input data after dropna/sort.
          - "innovation" : str, as passed.
          - "df_t" : float, as passed.
          - "hyperparams" : dict, the selected hyperparameters.
          - "cv_score" : float, RMSE from cross-validation.
          - "cv_meta" : dict, metadata on CV setup (fold count, grid sizes, etc.).

    Notes
    -----
    • The penalty structure mirrors the Minnesota prior: own-lags are allowed
      larger variance than cross-lags; higher lags are more tightly shrunk;
      scale ratios σ_i / σ_j produce unit-free penalties and stabilise across
      differently scaled series.

    • The approach estimates equations independently; cross-equation shrinkage
      is not applied. The residual covariance Σ̂ captures contemporaneous
      correlation across equations.

    • Residual centring before Σ̂ avoids inflating covariance estimates due to
      a non-zero mean in e_t induced by the intercept.
    """

    df = macro_df.dropna().copy().sort_index()
    
    cols = list(df.columns)
    
    Y = df.values.astype(float) 
    
    T, k = Y.shape
    
    if T <= p + 4:
    
        raise ValueError("Not enough macro observations for BVAR fit.")

    if grid_lambda_overall is None:

        grid_lambda_overall = [0.05, 0.1, 0.2, 0.5]

    if grid_lambda_cross is None:

        grid_lambda_cross = [0.25, 0.5, 0.75, 1.0]

    if grid_lambda_lagdecay is None:

        grid_lambda_lagdecay = [0.5, 1.0, 1.5, 2.0]

    if grid_lambda_const is None:

        grid_lambda_const = [1.0, 5.0, 10.0, 20.0]

    if grid_prior_ownlag1_mean is None:

        grid_prior_ownlag1_mean = [0.5, 0.8, 1.0]

    if cv_min_train is None:

        cv_min_train = max(40, 4 * p) 

    cv_min_train = max(cv_min_train, p + 5) 


    def _fit_once(
        Y_train: np.ndarray,
        lam_overall: float,
        lam_cross: float,
        lam_lagdecay: float,
        lam_const: float,
        prior_mu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the Minnesota-style ridge system once on a given training block.

        Parameters
        ----------
        Y_train : np.ndarray
            Training sample array of shape (T_tr, k).
        lam_overall : float
            Overall tightness scaling λ_overall.
        lam_cross : float
            Cross-lag tightness multiplier λ_cross applied when j ≠ i.
        lam_lagdecay : float
            Lag decay exponent applied as ℓ^{λ_lagdecay} in the denominator.
        lam_const : float
            Intercept penalty scale, used as (λ_const σ_i)^2.
        prior_mu : float
            Prior mean for the first own-lag coefficient in every equation i.

        Returns
        -------
        B_stack : np.ndarray
            Stacked coefficient matrix of shape (1 + k p, k). The first row is
            the intercept, followed by p blocks of k rows corresponding to lags
            1..p, each containing the coefficients of y_{t−ℓ}.
        Sigma : np.ndarray
            Centred residual covariance estimate of shape (k, k).
        resid : np.ndarray
            Uncentred residual matrix of shape (T_tr − p, k).

        Notes
        -----
        • The scale parameters σ_j are computed as the sample standard deviations
        of first differences of Y_train, providing robust scale normalisation.
        • Each equation uses a diagonal penalty matrix D constructed from the
        tightness values, with D_{intercept,intercept} = (λ_const σ_i)^2 and
        D_{pos,pos} = [1 / tight(i, j, ℓ)]^2 for lagged coefficients.
        • The posterior mode is computed via the closed-form ridge solution with
        prior mean m that sets the first own-lag coefficient of equation i to
        prior_mu and zeros elsewhere.
        """
       
        Ttr, ktr = Y_train.shape

        dY = np.diff(Y_train, axis = 0)

        sj = np.std(dY, axis = 0, ddof = 1)

        sj = np.where(sj <= 1e-10, 1.0, sj)

        X_lag, Yp = _build_lagged(
            Y = Y_train, 
            p = p
        )
        
        T_eff = X_lag.shape[0]
        
        X = np.hstack([np.ones((T_eff, 1)), X_lag])  

        B_stack = np.zeros((1 + k * p, ktr))

        for i in range(ktr):

            pen = np.zeros(1 + k * p)

            pen[0] = (lam_const * sj[i]) ** 2  

            pos = 1
          
            for j in range(ktr):
          
                for ell in range(1, p + 1):
          
                    tight = lam_overall * (sj[i] / sj[j]) / (ell ** lam_lagdecay)
          
                    if j != i:
          
                        tight *= lam_cross
          
                    pen[pos] = (1.0 / max(tight, 1e-8)) ** 2
          
                    pos += 1
          
            D = np.diag(pen)

            m = np.zeros(1 + k * p)

            idx_own_lag1 = 1 + i * p

            m[idx_own_lag1] = float(prior_mu)

            y_i = Yp[:, i]

            b_i, _ = _ols_ridge(
                X = X,
                y = y_i,
                D = D, 
                m = m
            )

            B_stack[:, i] = b_i

        resid = Yp - X @ B_stack                        
        
        resid_centered = resid - resid.mean(0, keepdims = True)
        
        Sigma = (resid_centered.T @ resid_centered) / max(1, T_eff - (1 + k * p))

        return B_stack, Sigma, resid


    def _forecast_next(
        B_stack: np.ndarray, 
        Y_hist: np.ndarray
    ) -> np.ndarray:
        """
        Produce a one-step-ahead forecast from stacked coefficients and recent history.

        Parameters
        ----------
        B_stack : np.ndarray
            Stacked coefficient matrix of shape (1 + k p, k), as returned by
            `_fit_once`. The first row is the intercept, followed by p lag blocks.
        Y_hist : np.ndarray
            Array of the most recent observations of shape (T_hist, k) with
            T_hist ≥ p. Only the last p rows are used.

        Returns
        -------
        yhat : np.ndarray
            Forecast vector ŷ_{t} ∈ ℝ^k computed as

                ŷ_t = ĉ + ∑_{ℓ=1}^p Â_ℓ y_{t−ℓ},

            where ĉ and Â_ℓ are extracted from B_stack and y_{t−ℓ} are the last
            p rows of Y_hist.
        """
    
        assert Y_hist.shape[0] >= p

        lags = [Y_hist[-ell] for ell in range(1, p + 1)]

        xrow = np.hstack([1.0, np.concatenate(lags, axis = 0)])  

        yhat = xrow @ B_stack                            

        return yhat


    start_fold = cv_min_train
    
    end_fold = T - 1  
    
    fold_indices = list(range(start_fold, end_fold))

    if cv_max_folds is not None and len(fold_indices) > cv_max_folds:

        step = max(1, len(fold_indices) // cv_max_folds)

        fold_indices = fold_indices[::step]

    best_score = np.inf

    best_params = None

    for lam_overall, lam_cross, lam_lagdecay, lam_const, prior_mu in product(
        grid_lambda_overall,
        grid_lambda_cross,
        grid_lambda_lagdecay,
        grid_lambda_const,
        grid_prior_ownlag1_mean,
    ):

        sse = 0.0

        n_pred = 0

        for t_idx in fold_indices:

            Y_train = Y[:t_idx, :] 

            if Y_train.shape[0] <= p + 4:

                continue

            try:
                
                B_stack, _, _ = _fit_once(
                    Y_train = Y_train,
                    lam_overall = lam_overall,
                    lam_cross = lam_cross,
                    lam_lagdecay = lam_lagdecay,
                    lam_const = lam_const,
                    prior_mu = prior_mu,
                )
                
            except np.linalg.LinAlgError:
                
                sse = np.inf
                
                break

            yhat = _forecast_next(
                B_stack = B_stack, 
                Y_hist = Y_train
            )

            ytrue = Y[t_idx, :]

            err = ytrue - yhat

            sse += float(err @ err)

            n_pred += len(err)

        if n_pred == 0:
        
            continue

        rmse = np.sqrt(sse / n_pred)

        if rmse < best_score:
         
            best_score = rmse
          
            best_params = dict(
                lambda_overall = lam_overall,
                lambda_cross = lam_cross,
                lambda_lagdecay = lam_lagdecay,
                lambda_const = lam_const,
                prior_ownlag1_mean = prior_mu,
            )

    if best_params is None or not np.isfinite(best_score):
      
        raise RuntimeError("Cross-validation failed to find valid hyperparameters.")

    B_stack, Sigma, resid = _fit_once(
        Y_train = Y,
        lam_overall = best_params["lambda_overall"],
        lam_cross = best_params["lambda_cross"],
        lam_lagdecay = best_params["lambda_lagdecay"],
        lam_const = best_params["lambda_const"],
        prior_mu = best_params["prior_ownlag1_mean"],
    )

    resid_centered = resid - resid.mean(0, keepdims = True)

    c = B_stack[0, :]
  
    A = np.zeros((p, k, k))
  
    for ell in range(p):
  
        block = B_stack[1 + ell * k: 1 + (ell + 1) * k, :]
  
        A[ell] = block.T

    return {
        "A": A,                                  
        "c": c,                                
        "Sigma": Sigma,                
        "resid": resid,              
        "resid_centered": resid_centered,   
        "history": Y[-p:],         
        "columns": cols,
        "index": df.index,
        "innovation": innovation,
        "df_t": df_t,
        "hyperparams": best_params,            
        "cv_score": float(best_score),         
        "cv_meta": {
            "folds": len(fold_indices),
            "cv_min_train": cv_min_train,
            "grid_sizes": {
                "lambda_overall": len(grid_lambda_overall),
                "lambda_cross": len(grid_lambda_cross),
                "lambda_lagdecay": len(grid_lambda_lagdecay),
                "lambda_const": len(grid_lambda_const),
                "prior_ownlag1_mean": len(grid_prior_ownlag1_mean),
            },
        },
    }


def simulate_bvar(
    model: dict,
    n_sims: int = 1000,
    horizon: int = 4,
    seed: int | None = 42,
    innovation: str | None = None,   
    df_t: float | None = None,       
    block_len: int = 4,              
) -> dict[str, pd.DataFrame]:
    """
    Generate Monte Carlo scenarios from a fitted Minnesota VAR(p).

    The state evolves according to

        y_t = c + ∑_{ℓ=1}^p A_ℓ y_{t-ℓ} + u_t,

    using parameters (A_ℓ, c, Σ) from `fit_bvar_minnesota`. The simulation
    starts from the last p observed rows ("history") and proceeds for the
    specified horizon. Innovations u_t are drawn according to the chosen
    innovation scheme.

    Innovation Schemes
    ------------------
    • "gaussian":
        u_t = z_t L',  where z_t ~ N(0, I_k) and L is the Cholesky factor of Σ
        (robustly computed via eigenvalue repair if needed).

    • "student_t":
            u_t = (z_t / sqrt(g_t)) L',  
        
        where 
        
            z_t ~ N(0, I_k) and g_t ~ χ²_ν / ν,
       
        independently across t. This is the standard Gaussian scale mixture
        representation that yields multivariate Student-t marginals with ν = df_t.
        Variance exists for ν > 2; heavy tails diminish as ν increases.

    • "bootstrap":
        u_t is obtained by circular block bootstrap applied to the centred
        residual matrix (T_eff × k). Blocks of length `block_len` are sampled
        with wrap-around to preserve short-run dependence structures across
        both time and equations.

    Parameters
    ----------
    model : dict
        The dictionary returned by `fit_bvar_minnesota`, containing "A", "c",
        "Sigma", "history", and optionally "resid_centered", "innovation",
        and "df_t".
    n_sims : int, default 1000
        Number of independent scenario paths to simulate.
    horizon : int, default 4
        Number of forward steps to generate. The output will contain keys
        "Q1", …, f"Q{horizon}".
    seed : int or None, default 42
        Seed for the NumPy random number generator.
    innovation : {"gaussian", "student_t", "bootstrap"} or None, default None
        If None, falls back to `model["innovation"]` or "student_t".
    df_t : float or None, default None
        Degrees of freedom for the Student-t scheme. If None, falls back to
        `model["df_t"]` or 7.0.
    block_len : int, default 4
        Block length for the circular block bootstrap when innovation="bootstrap".

    Returns
    -------
    dict[str, pd.DataFrame]
        A mapping from horizon labels to DataFrames with index equal to the
        variable names (model["columns"]) and columns representing simulation
        identifiers "1", …, f"{n_sims}". For example:

            {
              "Q1": DataFrame(k × n_sims),
              "Q2": DataFrame(k × n_sims),
              ...,
              f"Q{horizon}": ...
            }

    Notes
    -----
    • The routine uses the last p observed rows of the input data ("history")
      as the initial condition. At each step t, 
      
        x_t = c + ∑_{ℓ=1}^p A_ℓ y_{t-ℓ}
        
      is formed, then an innovation u_t is added.

    • When "bootstrap" is chosen, residuals are mean-centred prior to resampling.
      This avoids re-introducing a spurious mean into innovations.

    • A numerically robust Cholesky factor is computed by attempting a standard
      decomposition of Σ + εI_k; if this fails, an eigenvalue decomposition with
      non-negative repair is used to form a valid factor.

    • The output moments across simulations need not match the in-sample
      estimates exactly, especially when using Student-t or bootstrap
      innovations or short horizons.
    """
    
    A = model["A"]
    
    c = model["c"]
    
    Sigma = model["Sigma"]
    
    hist = model["history"]
    
    cols = model["columns"]

    k = len(cols)
   
    p = A.shape[0]

    rng = np.random.default_rng(seed)

    try:

        chol = np.linalg.cholesky(Sigma + 1e-12 * np.eye(k))

    except np.linalg.LinAlgError:

        w, Q = np.linalg.eigh((Sigma + Sigma.T) / 2)

        w = np.maximum(w, 1e-12)

        chol = Q @ np.diag(np.sqrt(w))

    mode = (innovation or model.get("innovation") or "student_t").lower()

    dof = float(df_t or model.get("df_t", 7.0))

    resid_cent = model.get("resid_centered", model["resid"] - model["resid"].mean(0, keepdims = True))

    sims = np.empty((horizon, k, n_sims))

    for j in range(n_sims):
     
        buf = hist.copy()

        if mode == "bootstrap":

            boot_path = _circular_block_bootstrap(
                resid = resid_cent, 
                T = horizon, 
                L = block_len,
                rng = rng
            )  

        for t in range(horizon):

            if mode == "student_t":

                g = rng.chisquare(dof) / dof

                shock = (rng.standard_normal(k) / np.sqrt(g)) @ chol.T

            elif mode == "bootstrap":

                shock = boot_path[t]

            else:

                shock = rng.standard_normal(k) @ chol.T

            x = c.copy()

            for ell in range(p):

                x += A[ell] @ buf[-ell-1]

            x += shock

            sims[t, :, j] = x

            buf = np.vstack([buf, x])[-p:]

    idx_q = [f"Q{i}" for i in range(1, horizon + 1)]

    sims_q = {
        q: pd.DataFrame(sims[i], index=cols, columns=[f"{n + 1}" for n in range(n_sims)])
        for i, q in enumerate(idx_q)
    }

    return sims_q
