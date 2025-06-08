from __future__ import annotations
from functools import lru_cache
import itertools
from typing import Iterable, Tuple, Optional, List
import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import cvxpy as cp


@lru_cache(maxsize=16)
def _get_solver(n: int, p: int, constrained: bool):
    """
    Build and cache one CVXPY Problem (with Parameters) for given data-shape and constraint.
    Returns: Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem
    """
    Xd_param   = cp.Parameter((n, p + 1))
    y_param    = cp.Parameter(n)
    lam1_param = cp.Parameter(nonneg=True)  
    lam2_param = cp.Parameter(nonneg=True)  
    M_param    = cp.Parameter(nonneg=True) 

    beta = cp.Variable(p + 1)

    resid = Xd_param @ beta - y_param
    loss  = cp.sum(cp.huber(resid, M=M_param))

    ridge_eps = 1e-8
    penalty  = (
        lam1_param * cp.norm1(beta)
        + lam2_param * cp.sum_squares(beta)
        + ridge_eps   * cp.sum_squares(beta)
    )

    constraints = [beta[1:] >= 0] if constrained else []
    problem    = cp.Problem(cp.Minimize(loss + penalty), constraints)

    return Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem


def _solve_core_std(
    Xs: np.ndarray,
    ys: np.ndarray,
    lam: float,
    alpha: float,
    huber_M: float,
    constrained: bool,
) -> np.ndarray:
    """
    Standardised‐scale solve: set Parameter values, re‐solve, and return beta.
    """
    n, p = Xs.shape
    Xd   = np.column_stack((np.ones(n), Xs))

    Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem = (
        _get_solver(n, p, constrained)
    )

    Xd_param.value   = Xd
    y_param.value    = ys
    lam1_param.value = lam * alpha
    lam2_param.value = lam * (1.0 - alpha)
    M_param.value    = huber_M

    for solver in (cp.ECOS, cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
            if beta.value is not None:
                return beta.value
        except cp.SolverError:
            continue

    raise RuntimeError("All solvers failed to converge.")


def _solve_enet_huber(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    alpha: float,
    huber_M: float,
    *,
    constrained: bool,
) -> np.ndarray:
    """
    Standardise → solve → de‐standardise → return β on original scale.
    """
    
    mean_x = X.mean(axis=0)
    std_x  = X.std(axis=0, ddof=0)
    std_x[std_x == 0] = 1.0

    mean_y = y.mean()
    std_y  = y.std(ddof=0) or 1.0

    Xs = (X - mean_x) / std_x
    ys = (y - mean_y) / std_y

    beta_std = _solve_core_std(Xs, ys, lam, alpha, huber_M, constrained)

    beta0_std     = beta_std[0]
    beta_std_rest = beta_std[1:]
    beta_rest     = std_y * beta_std_rest / std_x
    beta0 = mean_y + std_y * beta0_std - np.dot(beta_rest, mean_x)

    return np.concatenate(([beta0], beta_rest))


def constrained_regression(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    alpha: float = 0.5,
    huber_M: float = 1.0,
) -> np.ndarray:
    """Elastic‐Net + Huber with β[1:] ≥ 0."""
    return _solve_enet_huber(
        X, y, lam, alpha, huber_M, constrained=True
    )


def ordinary_regression(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    alpha: float = 0.5,
    huber_M: float = 1.0,
) -> np.ndarray:
    """
    Elastic‐Net + Huber with no sign constraint.
    """
    return _solve_enet_huber(
        X, y, lam, alpha, huber_M, constrained=False
    )


def _eval(fold_idx: int, param_idx: int, params: Tuple[float, float, float], Xd_full, cv_splits, X, y, regression_func, scorer):
    """
    Evaluate one (alpha, lam, M) on one fold.
    """
    
    alpha, lam, M = params
    train_idx, test_idx = cv_splits[fold_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = Xd_full[test_idx]
    y_test = y[test_idx]
    
    try:
        beta = regression_func(X_train, y_train, lam=lam, alpha=alpha, huber_M=M)
    
    except Exception:
        return (fold_idx, param_idx, np.inf)
    
    y_hat = X_test @ beta
    
    return (fold_idx, param_idx, scorer(y_test, y_hat))


def make_param_grid(
    alphas, 
    lambdas, 
    huber_M_values
) -> List[Tuple[float,float,float]]:
    
    return list(itertools.product(alphas, lambdas, huber_M_values))


def grid_search_regression(
    X: np.ndarray,
    y: np.ndarray,
    regression_func,
    *,
    alphas: Optional[Iterable[float]] = None,
    lambdas: Optional[Iterable[float]] = None,
    huber_M_values: Optional[Iterable[float]] = None,
    cv_folds: int = 5,
    param_grid: Optional[List[Tuple[float, float, float]]] = None,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    scorer=lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    n_jobs: int = -1,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Hyper-parameter search over (alpha, lambda, M) with optional pre-built grid and splits.

    Returns: (beta_best, lambda_best, alpha_best, M_best)
    """

    if param_grid is None:
        
        if alphas is None:
            alphas = np.linspace(0.3, 0.7, 5)
        
        if lambdas is None:
            lambdas = np.logspace(-4, 1, 20)
        
        if huber_M_values is None:
            huber_M_values = (1.0,)
        
        param_grid = make_param_grid(alphas, lambdas, huber_M_values)
   
    n_param = len(param_grid)

    if cv_splits is None:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
        cv_splits = list(kf.split(X))
    n_folds = len(cv_splits)

    Xd_full = np.column_stack((np.ones(X.shape[0]), X))

    jobs = []
    for fold_i in range(n_folds):
        for param_i, params in enumerate(param_grid):
            jobs.append(delayed(_eval)(fold_i, param_i, params, Xd_full, cv_splits, X, y, regression_func, scorer))
    results = Parallel(n_jobs=n_jobs)(jobs)

    errors = np.zeros((n_folds, n_param), dtype=float)
    for fold_i, param_i, err in results:
        errors[fold_i, param_i] = err
    mean_errors = errors.mean(axis=0)

    best_idx = int(np.argmin(mean_errors))
    best_alpha, best_lam, best_M = param_grid[best_idx]

    beta_best = regression_func(X, y, lam=best_lam, alpha=best_alpha, huber_M=best_M)
    return beta_best, best_lam, best_alpha, best_M

