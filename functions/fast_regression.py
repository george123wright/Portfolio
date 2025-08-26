"""

Huber–Elastic-Net regression with time-series cross-validation and optional
monotonicity constraints on slopes.

Overview
--------

Let X ∈ ℝ^{n×p} be the design matrix and y ∈ ℝ^n a target series. For a given
target the method solves, on *standardised and orthogonalised* features, a
Huber-loss + elastic-net objective with optional non-negativity constraints
on the slope coefficients:

    β̂ = argmin_{β ∈ ℝ^{p+1}}  ∑_{i=1}^n ρ_M( (x̄_i^⊤ β - ȳ_i) ) + λ_1 ‖β_{1:}‖_1 + λ_2 ‖β_{1:}‖_2^2 + ε ‖β_{1:}‖_2^2

where:

- β = (β_0, β_1, …, β_p) contains an intercept and p slopes; penalties exclude the intercept (only β_{1:} are penalised).

- ρ_M(·) is the Huber loss with threshold M>0:
     
      ρ_M(r) = 0.5 r^2                            if |r| ≤ M
             = M(|r| - 0.5M)                      if |r| > M .

- (x̄_i, ȳ_i) are the standardised / transformed observations (see below).

- (λ_1, λ_2) are the L1/L2 elastic-net weights; a small ε>0 ridge term ensures strong convexity.

- Optional constraints enforce β_{1:} ≥ 0 (monotone non-decreasing mapping).

Standardisation and Orthogonalisation
-------------------------------------

Before fitting, features are standardised columnwise using training means μ_X and
standard deviations σ_X. 

The second feature is orthogonalised to the first to reduce collinearity: with 

    z_1 = (x_1 - μ_1)/σ_1, z_2^raw = (x_2 - μ_2)/σ_2,

    c_{21} = ⟨z_2^raw, z_1⟩ / ⟨z_1, z_1⟩ ,    z_2 = z_2^raw - c_{21} z_1.

Targets are standardised as ȳ = (y - μ_y)/σ_y. 

Coefficients are mapped back to the original scale after solving so outputs are always
on the raw units.

Joint Hyper-parameter Selection
-------------------------------

For a set of targets {y^k}, a single triple (M, λ, α) is selected jointly by
time-series K-fold CV (sklearn TimeSeriesSplit), where λ_1 = α λ and λ_2 = (1-α) λ.

The selected triple minimises the mean CV error across the targets, where the CV
error per fold is the average Huber loss on the test part.

Implementation Notes
--------------------

- The solver is built with CVXPY as a parameterised problem; (X, y, λ_1, λ_2, M)
  are Parameters and β are Variables.

- Fold-level parallelism is performed with `joblib.Parallel`; parameter batches
  within a fold may be evaluated with threads.

- EPS ⟂ Revenue orthogonalisation is implemented exactly as described above.

- All returned coefficients include the intercept as the first element.
"""


from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import cvxpy as cp
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import TimeSeriesSplit


@lru_cache(maxsize = 32)
def _get_solver(
    n: int, 
    p_eff: int, 
    constrained: bool
):
    """
    Build and cache a parameterised CVXPY problem for a Huber–elastic-net model.

    Objective
    ---------
   
    For standardised design X̃ ∈ ℝ^{n×p_eff} (with an explicit intercept column added
    internally) and standardised response ỹ ∈ ℝ^n, the problem is:

        minimise_{β ∈ ℝ^{p_eff+1}}  ∑_{i=1}^n ρ_M( (X̃β - ỹ)_i )
                                     + λ_1 ‖β_{1:}‖_1 + λ_2 ‖β_{1:}‖_2^2 + ε ‖β_{1:}‖_2^2,

    with ε = 1e-8. The Huber loss is:

        ρ_M(r) = 0.5 r^2                          if |r| ≤ M,
                 M(|r| - 0.5 M)                   if |r| > M.

    Constraints
    -----------
    If `constrained` is True, the slope vector obeys β_{1:} ≥ 0. The intercept is
    unconstrained and is not penalised.

    Parameters
    ----------
    
    n : int
        Number of training observations in the fold.
   
    p_eff : int
        Number of *kept* features after standardisation / dropping near-constant columns.
   
    constrained : bool
        Whether to impose β_{1:} ≥ 0.

    Returns
    -------
    (Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem)
        CVXPY Parameters for the design (with intercept column), response, penalties,
        Huber threshold, the coefficient Variable β, and the compiled Problem.
    """

    Xd_param = cp.Parameter((n, p_eff + 1))

    y_param = cp.Parameter(n)

    lam1_param = cp.Parameter(nonneg = True)   

    lam2_param = cp.Parameter(nonneg = True)   

    M_param = cp.Parameter(nonneg = True)    

    beta = cp.Variable(p_eff + 1)           

    resid = Xd_param @ beta - y_param
    
    loss = cp.sum(cp.huber(resid, M = M_param))

    ridge_eps = 1e-8

    penalty = (
        lam1_param * cp.norm1(beta[1: ]) +
        lam2_param * cp.sum_squares(beta[1: ]) +
        ridge_eps * cp.sum_squares(beta[1: ])
    )

    if constrained:
        
        constraints = [beta[1:] >= 0]  
    
    else:
        
        constraints = []

    problem = cp.Problem(cp.Minimize(loss + penalty), constraints)

    return Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem


class HuberENetCV:
    """
    Huber–elastic-net regression with joint hyper-parameter selection across targets.

    For a collection of targets {y^k}, a single hyper-parameter triple (M, λ, α)
    is selected that minimises the mean time-series CV error across k. Within a
    fold, each target is fitted on standardised and orthogonalised features, with
    optional non-negativity constraints on slopes.

    Hyper-parameters
    ----------------
    
    - M ∈ Ms : Huber threshold.
    
    - λ ∈ lambdas : overall penalty weight; decomposed as λ_1 = α λ (L1) and
    
      λ_2 = (1 - α) λ (L2).
    
    - α ∈ alphas : elastic-net mixing parameter.

    Cross-validation
    ----------------
  
    A `TimeSeriesSplit(n_splits)` is used to avoid look-ahead bias. For each
    candidate (M, λ, α) and for each fold, the model is trained on the training
    segment and scored on the test segment using the Huber loss:

        CVError(M, λ, α) = (1/|K|) ∑_{k∈K} (1/n_te) ∑_{i∈te} ρ_M( y_i^k - ŷ_i^k(M,λ,α) ).

    The best triple minimises the average CV error across folds and targets.

    Scaling and Orthogonalisation
    -----------------------------
    Let x_j denote feature j and y the target. The training transform computes
    μ_xj, σ_xj and μ_y, σ_y. The first two standardised features z_1, z_2 are
    defined by:
   
        z_1 = (x_1 - μ_1)/σ_1,
   
        z_2 = (x_2 - μ_2)/σ_2 - c_{21} z_1,
   
    where 
    
        c_{21} = ⟨(x_2 - μ_2)/σ_2, (x_1 - μ_1)/σ_1⟩ / ⟨(x_1 - μ_1)/σ_1, (x_1 - μ_1)/σ_1⟩.

    Coefficient Mapping Back
    ------------------------
    If b_std are the coefficients in the standardised space (intercept first), the
    original-scale coefficients b are recovered componentwise with the same
    orthogonalisation inverse and:

        b_j = (σ_y / σ_{xj}) * b_std,j,    for j ≥ 1,
  
        b_0 = μ_y + σ_y b_std,0 - ⟨b_{1:}, μ_X⟩.

    Parallelism
    -----------
    Fold-level evaluation is parallelised with processes. Within a fold, the
    parameter grid is optionally partitioned into thread-level batches.
    """


    def __init__(
        self,
        alphas: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
        lambdas: Iterable[float] = tuple(np.logspace(0, -4, 20)),
        Ms: Iterable[float] = (0.25, 1.0, 4.0),
        *,
        n_splits: int = 5,
        tol: float = 1e-12,
        solver_order: Tuple = (cp.OSQP, cp.ECOS, cp.SCS),
        n_jobs: int = -1,
        param_threads: int = 1,
    ):   
        
        self.alphas = np.array(list(alphas), dtype = float)
       
        self.lambdas = np.array(list(lambdas), dtype = float)
       
        self.Ms = np.array(list(Ms), dtype=float)

        self.grid: List[Tuple[float, float, float]] = [
            (M, lmb, a)
            for M in self.Ms
            for lmb in self.lambdas
            for a in self.alphas
        ]

        self.n_splits = int(n_splits)
       
        self.tol = float(tol)
       
        self.solver_order = tuple(solver_order)
       
        self.n_jobs = int(n_jobs)
       
        self.param_threads = max(1, int(param_threads))


    def fit_joint(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        constrained_map: Dict[str, bool],
        *,
        cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        scorer = None,
    ) -> Tuple[Dict[str, np.ndarray], float, float, float]:
        """
        Fit all targets with a single (M, λ, α) selected jointly by time-series CV.

        Procedure
        ---------
        
        1) For each candidate (M, λ, α) on `self.grid`, compute the per-fold CV error
        for each target using Huber loss on the test segment; average over targets.
        
        2) Select the triple minimising the mean error; form a local refinement grid
        around the best triple (geometric for λ, linear for α, bounded for M).
        
        3) Repeat step (1) on the refine grid; choose the best triple.
        
        4) Refit each target on the full sample with the selected triple, returning
        coefficients on the original scale.

        Returns
        -------
      
        betas_by_key : Dict[str, np.ndarray]
            Mapping from target key to coefficient vector b ∈ ℝ^{p+1} (intercept first).
      
        best_lambda : float
      
        best_alpha : float
      
        best_M : float
            Selected hyper-parameters.
        """     
        
        X = np.ascontiguousarray(X, dtype = float)
      
        keys = list(y_dict.keys())
      
        Y = {k: np.ascontiguousarray(v, dtype = float) for k, v in y_dict.items()}

        if cv_splits is None:

            tscv = TimeSeriesSplit(n_splits = self.n_splits)

            cv_splits = list(tscv.split(X))

        mean_err, _ = self._cv_errors_for_grid(
            X = X,
            Y = Y, 
            keys = keys, 
            constrained_map = constrained_map, 
            cv_splits = cv_splits, 
            scorer = scorer, 
            grid = self.grid
        )
        
        best_idx = int(np.argmin(mean_err))
        
        best_M, best_lambda, best_alpha = self.grid[best_idx]

        lam_lo = best_lambda / np.sqrt(10.0)
       
        lam_hi = best_lambda * np.sqrt(10.0)
       
        alpha_lo = max(0.0, best_alpha - 0.15)
       
        alpha_hi = min(1.0, best_alpha + 0.15)
       
        M_candidates = np.unique(
            np.clip(
                np.array([best_M / 2.0, best_M, best_M * 2.0], dtype = float),
                self.Ms.min(), self.Ms.max()
            )
        )
       
        refine_grid = [
            (M, l, a)
            for M in M_candidates
            for l in np.geomspace(lam_lo, lam_hi, 7)
            for a in np.linspace(alpha_lo, alpha_hi, 7)
        ]

        ref_err, _ = self._cv_errors_for_grid(
            X = X, 
            Y = Y,
            keys = keys, 
            constrained_map = constrained_map, 
            cv_splits = cv_splits, 
            scorer = scorer, 
            grid = refine_grid
        )
        
        r_idx = int(np.argmin(ref_err))
        
        best_M, best_lambda, best_alpha = refine_grid[r_idx]

        betas_by_key: Dict[str, np.ndarray] = {}

        for k in keys:

            b = self.fit_single(
                X = X,
                y = Y[k],
                lam = best_lambda,
                alpha = best_alpha,
                huber_M = best_M,
                constrained = bool(constrained_map[k]),
                tol = self.tol,
                solvers = self.solver_order,
            )
            
            betas_by_key[k] = b

        return betas_by_key, best_lambda, best_alpha, best_M


    def _cv_errors_for_grid(
        self,
        X: np.ndarray,
        Y: Dict[str, np.ndarray],
        keys: List[str],
        constrained_map: Dict[str, bool],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        scorer,
        grid: List[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, int]:
        """
        Compute mean cross-validated errors for a given hyper-parameter grid.

        For each fold f and each (M, λ, α) in `grid`, a problem is solved on the
        training segment and scored on the test segment via the Huber loss,

            score_f(M,λ,α) = (1/|K|) ∑_{k∈K} (1/n_te) ∑_{i∈te} ρ_M( y_i^k - ŷ_i^k ).

        The method returns the vector of mean scores across folds, aligned to `grid`.

        Returns
        -------
    
        mean_err : np.ndarray, shape (len(grid),)
            Average CV error per triple.
    
        int
            The grid length (for convenience).
        """
        
        errs_mat = Parallel(n_jobs = self.n_jobs, prefer = "processes")(
            delayed(self._eval_one_fold)(
                X_full = X,
                y_all = Y,
                keys = keys,
                constrained_map = constrained_map,
                cv_splits = cv_splits,
                fold_idx = f,
                scorer = scorer,
                grid = grid,              
            )
            for f in range(len(cv_splits))
        )
        
        mean_err = np.vstack(errs_mat).mean(axis = 0)
        
        return mean_err, len(grid)


    @staticmethod
    def fit_single(
        X: np.ndarray,
        y: np.ndarray,
        lam: float,
        alpha: float,
        huber_M: float,
        *,
        constrained: bool,
        tol: float = 1e-12,
        solvers: Tuple = (cp.OSQP, cp.ECOS, cp.SCS),
    ) -> np.ndarray:
        """
        Final single fit (no CV) with Huber–elastic-net on transformed features.

        The procedure applies:
    
        (i) training-only standardisation of X and y,
    
        (ii) orthogonalisation of the second feature to the first, and
    
        (iii) convex optimisation with the objective

            ∑_{i=1}^n ρ_M( (X̃β - ỹ)_i ) + λ_1 ‖β_{1:}‖_1 + λ_2 ‖β_{1:}‖_2^2 + ε ‖β_{1:}‖_2^2,

        with optional constraints β_{1:} ≥ 0. The returned coefficients are mapped
        back to the original feature scale (intercept first).
        """
        
        X = np.ascontiguousarray(X, dtype = float)
        
        y = np.ascontiguousarray(y, dtype = float)

        xf = HuberENetCV._fit_x_transform(
            X_tr = X, 
            tol = tol
        )
       
        Xs = HuberENetCV._transform_X(
            X = X, 
            xf = xf
        )

        mean_y = y.mean()
       
        std_y = y.std(ddof = 0) or 1.0
       
        ys = (y - mean_y) / std_y
       
        yf = {
            "mean_y": mean_y, 
            "std_y": std_y
        }

        b_std = HuberENetCV._solve_core_std(
            Xs = Xs,
            ys = ys,
            lam = lam,
            alpha = alpha,
            huber_M = huber_M,
            constrained = constrained, 
            solvers = solvers
        )
        
        return HuberENetCV._destandardise_beta(
            b_std = b_std,
            yf = yf,
            xf = xf,
            p_full = X.shape[1]
        )


    @classmethod
    def clear_caches(
        cls
    ):
    
        _get_solver.cache_clear()

    
    def _eval_one_fold(
        self,
        X_full: np.ndarray,
        y_all: Dict[str, np.ndarray],
        keys: List[str],
        constrained_map: Dict[str, bool],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        fold_idx: int,
        scorer,
        grid: Optional[List[Tuple[float, float, float]]] = None,   
    ) -> np.ndarray:
        """
        Evaluate a hyper-parameter subset on a single time-series fold.

        Let tr and te denote the train/test indices for the fold. For each (M, λ, α)
        in `grid`, the solver is parameterised with (X̃_tr, ỹ_tr, λ_1=αλ, λ_2=(1-α)λ, M),
        solved, and the predictions on te are scored with the Huber loss. The errors
        are averaged across targets and returned aligned to `grid`.
        """

      
        tr, te = cv_splits[fold_idx]
      
        X_tr = X_full[tr]
      
        X_te = np.column_stack((np.ones(len(te)), X_full[te]))

        xf = HuberENetCV._fit_x_transform(
            X_tr = X_tr, 
            tol = self.tol
        )
        
        Xs_tr = HuberENetCV._transform_X(
            X = X_tr,
            xf = xf
        ) 

        n, p_eff = Xs_tr.shape
        
        stack = np.column_stack((np.ones(n), Xs_tr))

        used_flags = tuple(sorted(set(bool(constrained_map[k]) for k in keys)))
        
        prob_cache = {
            flag: _get_solver(
                n = n,
                p_eff = p_eff,
                constrained = flag
            ) for flag in used_flags
        }

        for flag in used_flags:

            Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem = prob_cache[flag]

            Xd_param.value = np.ascontiguousarray(stack, dtype = float)

        y_train_stats = {}

        for k in keys:

            y_tr = y_all[k][tr]

            mu = y_tr.mean()

            sd = y_tr.std(ddof = 0) or 1.0
          
            y_train_stats[k] = {
                "mean_y": mu, 
                "std_y": sd
            }

        if grid is None:
            
            local_grid = self.grid
        
        else:
            
            local_grid = list(grid)
            

        def _errs_for_params(
            params_subset: List[Tuple[float, float, float]]
        ) -> np.ndarray:
        
            out = np.empty(len(params_subset), dtype = float)
        
            for i, (M, lmb, a) in enumerate(params_subset):
              
                total_err = 0.0
             
                l1 = lmb * a
             
                l2 = lmb * (1.0 - a)

                for flag in used_flags:

                    Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem = prob_cache[flag]

                    lam1_param.value = l1

                    lam2_param.value = l2

                    M_param.value = float(M)

                for k in keys:

                    flag = constrained_map[k]

                    Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem = prob_cache[flag]

                    stats = y_train_stats[k]
                
                    y_tr = y_all[k][tr]
                
                    ys_tr = (y_tr - stats["mean_y"]) / stats["std_y"]
                
                    y_param.value = np.ascontiguousarray(ys_tr, dtype = float)

                    solved = False

                    for solver in self.solver_order:

                        try:
                            problem.solve(solver = solver, warm_start = True, verbose = False)
                     
                            if beta.value is not None:
                     
                                solved = True
                     
                                break
                     
                        except cp.SolverError:
                     
                            continue
                    
                    if not solved:
                    
                        total_err = np.inf
                    
                        break

                    b_std = beta.value

                    b = HuberENetCV._destandardise_beta(
                        b_std = b_std,
                        yf = stats,
                        xf = xf, 
                        p_full = X_full.shape[1]
                    )
                    
                    y_hat = X_te @ b
                  
                    y_te = y_all[k][te]
                  
                    if scorer is None:
                  
                        total_err += HuberENetCV._score_huber(
                            y_true = y_te, 
                            y_pred = y_hat,
                            M = float(M)
                        )
                  
                    else:
                  
                        total_err += float(scorer(y_te, y_hat))

                out[i] = total_err / len(keys)
         
            return out

        if len(local_grid) == 0:

            return np.empty(0, dtype = float)

        nthreads = max(1, self.param_threads)
      
        chunk_size = int(np.ceil(len(local_grid) / nthreads))
      
        chunks: List[List[Tuple[float, float, float]]] = [
            local_grid[i:i + chunk_size] for i in range(0, len(local_grid), chunk_size)
        ]

        with parallel_backend("threading"):
         
            parts = Parallel(n_jobs=nthreads)(
                delayed(_errs_for_params)(chunk) for chunk in chunks
            )
       
        errs = np.concatenate(parts, axis = 0)
      
        return errs


    @staticmethod
    def _fit_x_transform(
        X_tr: np.ndarray, 
        tol: float = 1e-12
    ) -> Dict:
        """
        Fit the feature transformation on the training segment.

        Steps
        -----
     
        1) Columnwise means μ_x and standard deviations σ_x are computed.
     
        2) Columns with σ_x ≤ tol are dropped (only the first is kept if all are dropped).
     
        3) Standardise kept columns: Z = (X - μ_x)/σ_x.
     
        4) Orthogonalise the second kept column to the first:
     
            c_{21} = ⟨Z[:,1], Z[:,0]⟩ / ⟨Z[:,0], Z[:,0]⟩,
     
            Z[:,1] ← Z[:,1] - c_{21} Z[:,0].

        Returns
        -------
     
        dict with keys {"mean_x", "std_x", "keep", "c_21"}.
        """
       
        X_tr = np.ascontiguousarray(X_tr, dtype = float)
      
        mean_x = X_tr.mean(axis = 0)
      
        std_x = X_tr.std(axis = 0, ddof = 0)
      
        keep = std_x > tol

        if not np.any(keep):
          
            keep = np.zeros_like(std_x, dtype = bool)
          
            keep[0] = True
          
            std_x = std_x.copy()
          
            std_x[0] = 1.0

        Z1 = (X_tr[:, keep] - mean_x[keep]) / std_x[keep]

        c_21 = 0.0
      
        if Z1.shape[1] >= 2:
      
            z1 = Z1[:, 0]
      
            z2 = Z1[:, 1]
      
            denom = float(np.dot(z1, z1))
      
            if denom > 0:
      
                c_21 = float(np.dot(z2, z1) / denom)
      
                Z1[:, 1] = z2 - c_21 * z1  

        return {"mean_x": mean_x, "std_x": std_x, "keep": keep, "c_21": c_21}


    @staticmethod
    def _transform_X(
        X: np.ndarray, 
        xf: Dict
    ) -> np.ndarray:
        """
        Apply a previously fitted feature transform.

        Standardises kept columns using xf["mean_x"], xf["std_x"], and applies the
        stored orthogonalisation coefficient xf["c_21"] to the second kept column.
        """
      
        X = np.ascontiguousarray(X, dtype=float)
      
        Z = (X[:, xf["keep"]] - xf["mean_x"][xf["keep"]]) / xf["std_x"][xf["keep"]]
      
        if Z.shape[1] >= 2:
      
            Z[:, 1] = Z[:, 1] - xf["c_21"] * Z[:, 0]
      
        return Z


    @staticmethod
    def _destandardise_beta(
        b_std: np.ndarray, 
        yf: Dict, 
        xf: Dict,
        p_full: int
    ) -> np.ndarray:
        """
        Map coefficients from the standardised, orthogonalised space back to original units.

        If b_std = (b0_std, b1_std, b2_std, …) are the coefficients on [1, Z1, Z2, …], then
        with the orthogonalisation adjustment for the second feature,

            b_1 = (σ_y/σ_{x1}) * (b1_std - c_{21} b2_std),
         
            b_2 = (σ_y/σ_{x2}) * b2_std,
         
            b_j = (σ_y/σ_{xj}) * b_{j,std},  j ≥ 3,

        and intercept

            b_0 = μ_y + σ_y b0_std − ⟨b_{1:}, μ_X⟩.

        Returns
        -------
        np.ndarray
            Coefficient vector (intercept first) on the original feature scale.
        """
        
        beta0_std = b_std[0]
    
        rest_std = b_std[1:]
    
        full_rest = np.zeros(p_full, dtype = float)

        kept_idx = np.where(xf["keep"])[0]
     
        if len(kept_idx) >= 1:
     
            coef_x1 = rest_std[0]
     
            if len(kept_idx) >= 2:
     
                coef_x1 = rest_std[0] - xf["c_21"] * rest_std[1]
     
            full_rest[kept_idx[0]] = yf["std_y"] * coef_x1 / xf["std_x"][kept_idx[0]]

        if len(kept_idx) >= 2:
     
            full_rest[kept_idx[1]] = yf["std_y"] * rest_std[1] / xf["std_x"][kept_idx[1]]

        if len(kept_idx) > 2:
     
            k = 2
     
            for j in kept_idx[2:]:
     
                full_rest[j] = yf["std_y"] * rest_std[k] / xf["std_x"][j]
     
                k += 1

        beta0 = yf["mean_y"] + yf["std_y"] * beta0_std - np.dot(full_rest, xf["mean_x"])
     
        return np.concatenate(([beta0], full_rest))


    @staticmethod
    def _score_huber(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        M: float
    ) -> float:
        """
        Compute the mean Huber loss with threshold M:

            ρ_M(r) = 0.5 r^2                          if |r| ≤ M,
                    M(|r| - 0.5 M)                   if |r| > M,

        where r = y_true − y_pred. Returns the average over samples.
        """    
        
        r = y_true - y_pred
    
        a = np.abs(r)
    
        quad = 0.5 * np.minimum(a, M) ** 2
    
        lin = M * (a - np.minimum(a, M))
    
        return float(np.mean(quad + lin))


    @staticmethod
    def _solve_core_std(
        Xs: np.ndarray,
        ys: np.ndarray,
        lam: float,
        alpha: float,
        huber_M: float,
        constrained: bool,
        solvers: Tuple,
    ) -> np.ndarray:
        """
        Solve the standardised Huber–elastic-net problem and return β on the standardised scale.

        The design matrix is X̃ (with an intercept column appended). Parameters are set as:
       
        - lam1 = α λ, lam2 = (1−α) λ,
       
        - M = huber_M,
       
        and the CVXPY problem is solved with the specified solver order, warm-started.
        The return value is β on [1, Z1, Z2, …].
        """
       
        Xs = np.ascontiguousarray(Xs, dtype = float)
      
        ys = np.ascontiguousarray(ys, dtype = float)
      
        n, p_eff = Xs.shape

        Xd = np.column_stack((np.ones(n), Xs))

        Xd_param, y_param, lam1_param, lam2_param, M_param, beta, problem = _get_solver(
            n = n, 
            p_eff = p_eff, 
            constrained = constrained
        )
        
        Xd_param.value = np.ascontiguousarray(Xd, dtype = float)
       
        y_param.value = np.ascontiguousarray(ys, dtype = float)
       
        lam1_param.value = lam * alpha
       
        lam2_param.value = lam * (1.0 - alpha)
       
        M_param.value = huber_M

        for solver in solvers:
       
            try:
       
                problem.solve(solver = solver, warm_start = True, verbose = False)
       
                if beta.value is not None:
       
                    return beta.value
       
            except cp.SolverError:
       
                continue
       
        raise RuntimeError("All solvers failed to converge.")


def constrained_regression(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    alpha: float = 0.5,
    huber_M: float = 1.0,
    *,
    tol: float = 1e-12,
    solvers: Tuple = (cp.OSQP, cp.ECOS, cp.SCS),
) -> np.ndarray:
    """
    Convenience wrapper for a single constrained Huber–elastic-net fit:
    solves with β_{1:} ≥ 0 on standardised / orthogonalised features and
    returns coefficients on the original scale (intercept first).
    """
    
    return HuberENetCV.fit_single(
        X = X,
        y = y,
        lam = lam,
        alpha = alpha,
        huber_M = huber_M,
        constrained = True, 
        tol = tol, 
        solvers = solvers
    )


def ordinary_regression(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    alpha: float = 0.5,
    huber_M: float = 1.0,
    *,
    tol: float = 1e-12,
    solvers: Tuple = (cp.OSQP, cp.ECOS, cp.SCS),
) -> np.ndarray:
    """
    Convenience wrapper for a single unconstrained Huber–elastic-net fit:
    solves without monotonicity constraints and returns coefficients on the
    original scale (intercept first).
    """

    return HuberENetCV.fit_single(
        X = X, 
        y = y,
        lam = lam,
        alpha = alpha,
        huber_M = huber_M,
        constrained = False, 
        tol = tol, 
        solvers = solvers
    )
