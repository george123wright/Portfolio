"""
Implements optimisation routines for max Sharpe, Sortino, information ratio, equal‑risk and combination strategies with bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
import functools
from typing import Tuple, Dict, List, Optional
from functions.black_litterman_model import black_litterman
import config


def make_pd(
    mat: np.ndarray, 
    tol: float = 1e-8
) -> np.ndarray:
    """
    Symmetrise and lift the minimum eigenvalue to >= tol.
    """
    
    M = 0.5 * (np.asarray(mat, float) + np.asarray(mat, float).T)
    
    min_eig = np.linalg.eigvalsh(M).min()
    
    if min_eig < tol:
    
        M += np.eye(M.shape[0]) * (tol - min_eig)
    
    return M


def generate_bounds_for_asset(
    bnd_h: float, 
    bnd_l: float,
    er: float, 
    score: float
) -> Tuple[float, float]:
    """
    Long-only gating by score & expected return.
    """
    
    if score <= 0 or er <= 0:
    
        return (0.0, 0.0)
    
    return (float(bnd_l), float(bnd_h))


def softplus(
    x: np.ndarray, 
    beta: float = 50.0
) -> np.ndarray:
    """
    Smooth ReLU: (1/beta) * log(1 + exp(beta * x))
    Large beta -> closer to ReLU, but still smooth.
    """

    bx = beta * x

    out = np.where(bx > 20, bx / beta, np.log1p(np.exp(bx)) / beta)

    return out


def huber_vec(
    x: cp.Expression, 
    delta: float
) -> cp.Expression:
    """
    Elementwise Huber; CVXPY's huber returns elementwise; we sum later.
    """
    
    return cp.huber(x, delta)


class PortfolioOptimiser:
    """
    Behaviour-preserving class refactor with aggressive caching.
    Create once per 'universe' and reuse methods; heavy terms are computed once.
    """


    def __init__(
        self,

        # core
        er: pd.Series,              
        cov: pd.DataFrame | np.ndarray,
        scores: pd.Series,

        # weekly data
        weekly_ret_1y: pd.DataFrame,
        last_5_year_weekly_rets: pd.DataFrame,
        benchmark_weekly_ret: Optional[pd.Series] = None,

        # classifications + box bounds
        ticker_ind: pd.Series = None,
        ticker_sec: pd.Series = None,
        bnd_h: pd.Series = None,
        bnd_l: pd.Series = None,

        # BL inputs (prior & views)
        comb_std: Optional[pd.Series] = None,
        sigma_prior: Optional[pd.DataFrame] = None,
        mcap: Optional[pd.Series] = None,

        # caps & limits
        sector_limits: Optional[Dict[str, float]] = None,
        default_max_industry_pct: float = config.IND_MAX_WEIGHT,
        default_max_sector_pct: float = config.SECTOR_MAX_WEIGHT,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0), 

        # rates
        rf_annual: Optional[float] = None,         
        rf_week: Optional[float] = None,            
    ):

        if not isinstance(er, pd.Series) and not isinstance(scores, pd.Series):

            raise ValueError("Provide at least one of er or scores as pd.Series to infer the universe.")

        if isinstance(er, pd.Series):
            
            base = er 
        
        else:
            
            base = scores
            
        self._universe: List[str] = list(base.index)

        self._er = self._reindex_series(
            s = er
        )
        
        self._scores = self._reindex_series(
            s = scores
        )
        
        self.gamma = gamma

        if isinstance(cov, pd.DataFrame):
            
            cov = cov.reindex(index = self._universe, columns = self._universe).values
        
        self._cov = np.asarray(cov, float)

        self._weekly_ret_1y = self._reindex_df_cols(
            df = weekly_ret_1y
        ).dropna(how = "any")
        
        self._last_5y = self._reindex_df_cols(
            df = last_5_year_weekly_rets
        ).dropna(how = "any")
        
        self._benchmark_weekly = self._align_bench(
            bench = benchmark_weekly_ret
        )

        if ticker_ind is not None:
             
            self._ticker_ind = self._reindex_series(
                s = ticker_ind
            ) 
        
        else:
            
            self._ticker_ind = pd.Series(index = self._universe, dtype = object)
        
        if ticker_sec is not None:
            
            self._ticker_sec = self._reindex_series(
                s = ticker_sec
            )  
        
        else:
            self._ticker_sec = pd.Series(index = self._universe, dtype = object)

        if bnd_h is not None:
            
            self._bnd_h = (
                self._reindex_series(
                    s = bnd_h
                ).fillna(0.0).astype(float)
            )  
        
        else:
            
            self._bnd_h = pd.Series(1.0, index = self._universe)
        
        if bnd_l is not None:
            
            self._bnd_l = (
                self._reindex_series(
                    s = bnd_l
                ).fillna(0.0).astype(float)
            )  
        
        else: 
            
            self._bnd_l = pd.Series(0.0, index = self._universe)

        if comb_std is not None:
            
            self._comb_std = self._reindex_series(
                s = comb_std
            )  
        
        else:
            
            self._comb_std = None
        
        if isinstance(sigma_prior, pd.DataFrame):
            
            self._sigma_prior = sigma_prior.reindex(index = self._universe, columns = self._universe)  
        
        else:
            
            self._sigma_prior = None
        
        if mcap is not None:
            
            self._mcap = self._reindex_series(
                s = mcap
            )  
        
        else:
            
            self._mcap = None

        self.sector_limits = sector_limits or config.sector_limits
        
        self.default_max_industry_pct = float(default_max_industry_pct)
       
        self.default_max_sector_pct = float(default_max_sector_pct)

        if rf_annual is not None:
            
            self._rf_annual = float(rf_annual)  
        
        else:
            
            self._rf_annual = None
        
        if rf_week is not None:
       
            self._rf_week = float(rf_week)  
        
        else:
            self._rf_week = None

        self._mu_bl: Optional[pd.Series] = None
        
        self._sigma_bl: Optional[pd.DataFrame] = None

        self._last_diag: Optional[Dict[str, float]] = None

        self._A_ind_rows, self._A_sec_rows, self._ind_order, self._sec_order = self._build_caps_rows()


    def _reindex_series(
        self, 
        s: Optional[pd.Series]
    ) -> Optional[pd.Series]:
        
        if s is None: 
            
            return None
        
        return s.reindex(self._universe)


    def _reindex_df_cols(
        self, 
        df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        
        if df is None: 
            
            return None
        
        return df.reindex(columns=self._universe)


    def _align_bench(
        self, 
        bench: Optional[pd.Series]
    ) -> Optional[pd.Series]:
        
        if bench is None: 
            return None
       
        return bench.dropna()


    @functools.cached_property
    def universe(
        self
    ) -> List[str]:
        
        return self._universe


    @functools.cached_property
    def n(
        self
    ) -> int:
        
        return len(self._universe)


    @functools.cached_property
    def er_arr(
        self
    ) -> np.ndarray:
        
        return self._er.values.astype(float)


    @functools.cached_property
    def scores_arr(
        self
    ) -> np.ndarray:
        
        return self._scores.values.astype(float)


    @functools.cached_property
    def Σ(
        self
    ) -> np.ndarray:
        
        return make_pd(
            mat = self._cov
        )

    @functools.cached_property
    def Lc(
        self
    ) -> np.ndarray:
       
        return np.linalg.cholesky(self.Σ)


    @functools.cached_property
    def A(
        self
    ) -> np.ndarray:
      
        return self.Lc.T


    @functools.cached_property
    def R1(
        self
    ) -> np.ndarray:
        """
        Weekly returns (1y) matrix aligned to universe.
        """
        
        return self._weekly_ret_1y.values


    @functools.cached_property
    def R5(
        self
    ) -> np.ndarray:
        """
        Weekly returns (5y) matrix aligned to universe.
        """
        
        return self._last_5y.values


    @functools.cached_property
    def T1(
        self
    ) -> int:
        
        return self.R1.shape[0]


    @functools.cached_property
    def T5(
        self
    ) -> int:
        
        return self.R5.shape[0]


    @functools.cached_property
    def rf_ann(
        self
    ) -> float:
        
        if self._rf_annual is not None: 
            
            return float(self._rf_annual)
        
        return float(getattr(config, "RF", 0.0))


    @functools.cached_property
    def rf_week(
        self
    ) -> float:
        
        if self._rf_week is not None: 
            
            return float(self._rf_week)
        
        return float(getattr(config, "RF_PER_WEEK", 0.0))

    
    @functools.cached_property
    def sqrt52(
        self
    ) -> float:
        
        return float(np.sqrt(52.0))


    @functools.cached_property
    def lb_arr(
        self
    ) -> np.ndarray:
        
        out = np.zeros(self.n)
        
        for i, t in enumerate(self.universe):
        
            li, _ = generate_bounds_for_asset(
                bnd_h = float(self._bnd_h.loc[t]), 
                bnd_l = float(self._bnd_l.loc[t]),
                er = float(self._er.loc[t]), 
                score = float(self._scores.loc[t])
            )
            
            out[i] = li
        
        return out


    @functools.cached_property
    def ub_arr(
        self
    ) -> np.ndarray:
       
        out = np.zeros(self.n)
       
        for i, t in enumerate(self.universe):
       
            _, ui = generate_bounds_for_asset(
                bnd_h = float(self._bnd_h.loc[t]), 
                bnd_l = float(self._bnd_l.loc[t]),
                er = float(self._er.loc[t]), 
                score = float(self._scores.loc[t])
            )
            
            out[i] = ui
       
        return out


    @functools.cached_property
    def ind_idxs(
        self
    ) -> Dict[object, np.ndarray]:
        
        return {ind: np.where(self._ticker_ind.values == ind)[0] for ind in self._ticker_ind.dropna().unique()}


    @functools.cached_property
    def sec_idxs(
        self
    ) -> Dict[object, np.ndarray]:
        
        return {sec: np.where(self._ticker_sec.values == sec)[0] for sec in self._ticker_sec.dropna().unique()}


    def _build_caps_rows(
        self
    ):
        """
        Build binary rows for industry and sector caps once.
        Returns (A_ind_rows, A_sec_rows, ind_order, sec_order)
        where each is a list of (row_index_list).
        """
      
        A_ind_rows = []
        
        ind_order = []
        
        for ind, idxs in self.ind_idxs.items():
         
            row = np.zeros(self.n, dtype=float)
         
            row[idxs] = 1.0
         
            A_ind_rows.append(row)
         
            ind_order.append(ind)

        A_sec_rows = []
        
        sec_order = []
        
        for sec, idxs in self.sec_idxs.items():
           
            row = np.zeros(self.n, dtype=float)
           
            row[idxs] = 1.0
           
            A_sec_rows.append(row)
           
            sec_order.append(sec)

        return A_ind_rows, A_sec_rows, ind_order, sec_order


    def _caps_mats(
        self, 
        max_industry_pct: float, 
        max_sector_pct: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compose A_caps and caps vector for current limits.
        """
       
        mats = []
       
        caps = []

        if len(self._A_ind_rows) > 0:

            mats.append(np.vstack(self._A_ind_rows))

            caps.extend([max_industry_pct] * len(self._A_ind_rows))

        if len(self._A_sec_rows) > 0:

            A_sec = np.vstack(self._A_sec_rows)

            caps_sec = []

            for sec in self._sec_order:

                cap = self.sector_limits.get(sec, max_sector_pct)

                caps_sec.append(cap)

            mats.append(A_sec)

            caps.extend(caps_sec)

        if mats:

            A_caps = np.vstack(mats)

            caps_vec = np.asarray(caps, float)

        else:

            A_caps = np.zeros((0, self.n), float)

            caps_vec = np.zeros((0,), float)

        return A_caps, caps_vec


    def _solve(
        self, 
        prob: cp.Problem
    ) -> bool:
        """
        Try multiple solvers; return True if solved to (in)optimal.
        """
        
        for kwargs in (
            dict(solver=cp.ECOS, feastol=1e-10, reltol=1e-10, abstol=1e-10),
            dict(solver=cp.SCS, warm_start=True, eps=1e-5),
            dict(solver=cp.OSQP, warm_start=True),
        ):
        
            try:
        
                prob.solve(**kwargs)
        
            except cp.error.SolverError:
        
                continue
        
            if prob.status in ("optimal", "optimal_inaccurate"):
        
                return True
        
        return False


    def _caps_constraints_param(
        self,
        w: cp.Variable,
        A_caps: np.ndarray,
        caps_vec: np.ndarray,
        add_box: bool = True
    ) -> List[cp.Constraint]:
      
        cons = [cp.sum(w) == 1]
      
        if add_box:
      
            cons += [w >= self.lb_arr, w <= self.ub_arr]
      
        if A_caps.shape[0] > 0:
      
            cons.append(A_caps @ w <= caps_vec)
      
        return cons


    def _dinkelbach_sharpe_max(
        self,
        μ: np.ndarray,
        A: np.ndarray,
        rf: float,
        max_industry_pct: float,
        max_sector_pct: float,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> np.ndarray:
       
        n = self.n
       
        lam = 0.0
       
        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = max_industry_pct,
            max_sector_pct = max_sector_pct
        )

        for _ in range(max_iter):

            w = cp.Variable(n, nonneg = True)

            ret = μ @ w - rf

            vol = cp.norm(A @ w, 2)
          
            obj = cp.Maximize(ret - lam * vol)
          
            cons = self._caps_constraints_param(
                w = w, 
                A_caps = A_caps, 
                caps_vec = caps_vec, 
                add_box = True
            )
            
            prob = cp.Problem(obj, cons)
            
            if not self._solve(prob) or w.value is None:
            
                raise RuntimeError("Dinkelbach Sharpe: infeasible subproblem")
            
            w_val = w.value
            
            ret_v = float(μ @ w_val) - rf
            
            vol_v = float(np.linalg.norm(A @ w_val))
            
            if vol_v <= 0:
            
                raise RuntimeError("Dinkelbach Sharpe: zero vol")
            
            if abs(ret_v - lam * vol_v) < tol:
            
                return w_val
            
            lam = ret_v / vol_v
        
        raise RuntimeError("Dinkelbach Sharpe: no convergence")


    def msr(
        self,
        max_industry_pct: Optional[float] = None,
        max_sector_pct: Optional[float] = None,
        tol: float = 1e-10,
        max_iter: int = 100,
        riskfree_rate: Optional[float] = None,
    ) -> pd.Series:
        
        μ = self.er_arr
        
        A = self.A
        
        if riskfree_rate is None:
            
            rf = float(self.rf_ann)  
        
        else:
            
            rf = riskfree_rate
            
        w = self._dinkelbach_sharpe_max(
            μ = μ, 
            A = A, 
            rf = rf,
            max_industry_pct = (max_industry_pct or self.default_max_industry_pct),
            max_sector_pct = (max_sector_pct or self.default_max_sector_pct),
            tol = tol, 
            max_iter = max_iter,
        )
        
        return pd.Series(w, index=self.universe, name="msr")


    def msr_sortino(
        self,
        tol: float = 1e-10,
        max_iter: int = 100,
        max_industry_pct: Optional[float] = None,
        max_sector_pct: Optional[float] = None,
        riskfree_rate: Optional[float] = None,
    ) -> pd.Series:
       
        n = self.n
       
        μ = self.er_arr
        
        if riskfree_rate is None:
       
            rf_ann = float(self.rf_ann)  
        
        else:
            
            rf_ann = riskfree_rate
            
        rf_week = self.rf_week
       
        T = self.T1
       
        if T == 0:
       
            raise RuntimeError("msr_sortino: no weekly data")
       
        R = self.R1
       
        sqrt52 = self.sqrt52

        lam = 0.0
       
        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = max_industry_pct or self.default_max_industry_pct,
            max_sector_pct = max_sector_pct or self.default_max_sector_pct
        )

        for _ in range(max_iter):
            w = cp.Variable(n, nonneg = True)
            u = cp.Variable(T, nonneg = True)

            port_week = R @ w
            
            cons = [u >= rf_week - port_week, u >= 0]
            
            dd_week = cp.norm(u, 2) / np.sqrt(T)
            
            dd_ann = dd_week * sqrt52

            ret_ann = μ @ w - rf_ann
            
            obj = cp.Maximize(ret_ann - lam * dd_ann)

            cons += self._caps_constraints_param(
                w = w, 
                A_caps = A_caps, 
                caps_vec = caps_vec, 
                add_box = True
            )
            
            prob = cp.Problem(obj, cons)
            
            if not self._solve(prob) or w.value is None:
            
                raise RuntimeError("msr_sortino@Dinkelbach: infeasible")

            w_opt = w.value

            exc = float(μ @ w_opt) - rf_ann
            
            u_val = np.maximum(rf_week - (R @ w_opt), 0.0)
            
            dd_val = (np.linalg.norm(u_val) / np.sqrt(T)) * sqrt52

            if dd_val <= 0:
            
                raise RuntimeError("msr_sortino: zero DD")

            if abs(exc - lam * dd_val) < tol:
            
                return pd.Series(w_opt, index=self.universe, name="msr_sortino")

            lam = exc / dd_val

        raise RuntimeError("msr_sortino@Dinkelbach: no convergence")


    def MIR(
        self,
        tol: float = 1e-10,
        max_iter: int = 100,
        max_industry_pct: Optional[float] = None,
        max_sector_pct: Optional[float] = None,
        benchmark_weekly_ret: Optional[pd.Series] = None,
    ) -> pd.Series:

        df = pd.DataFrame(self._last_5y, index = self._last_5y.index if isinstance(self._last_5y, pd.DataFrame) else None, columns=self.universe)
        
        if benchmark_weekly_ret is None:
        
            benchmark_weekly_ret = self._benchmark_weekly

        if benchmark_weekly_ret is not None:
        
            df = df.join(benchmark_weekly_ret.rename("_bench"), how="inner")
        
            if df.empty:
        
                raise RuntimeError("MIR: no overlap between er_hist and benchmark")
        
            b = df["_bench"].values
        
            R = df.drop(columns=["_bench"]).values
        
            univ = list(df.drop(columns=["_bench"]).columns)
        
        else:
        
            R = self.R5
        
            b = np.full(self.T5, self.rf_week)
        
            univ = self.universe

        n = len(univ)
        
        T = R.shape[0]
        
        ones_T = np.ones(T)

        lam = 0.0
        
        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = max_industry_pct or self.default_max_industry_pct,
            max_sector_pct = max_sector_pct or self.default_max_sector_pct
        )

        for _ in range(max_iter):
          
            w = cp.Variable(n, nonneg = True)
          
            active = R @ w - b
          
            mean_a = (ones_T @ active) / T
          
            te_rms = cp.norm(active, 2) / np.sqrt(T)
          
            obj = cp.Maximize(mean_a - lam * te_rms)

            cons = self._caps_constraints_param(
                w = w, 
                A_caps = A_caps, 
                caps_vec = caps_vec, 
                add_box = True
            )
          
            prob = cp.Problem(obj, cons)
            
            if not self._solve(prob) or w.value is None:
            
                raise RuntimeError("MIR@Dinkelbach: infeasible")

            w_opt = w.value
            
            active_np = (R @ w_opt) - b
            
            mean_a_v = float(active_np.mean())
            
            te = float(np.linalg.norm(active_np) / np.sqrt(T))
            
            if te <= 0:
            
                raise RuntimeError("MIR: zero TE")

            if abs(mean_a_v - lam * te) < tol:
            
                return pd.Series(w_opt, index = univ, name = "MIR")

            lam = mean_a_v / te

        raise RuntimeError("MIR@Dinkelbach: no convergence")


    def msp(
        self,
        level: float = 5.0,
        tol: float = 1e-6,
        max_iter: int = 1000,
        max_industry_pct: Optional[float] = None,
        max_sector_pct: Optional[float] = None,
    ) -> pd.Series:
    
        univ = self.universe
    
        n = self.n
    
        hist = pd.DataFrame(self.R1, columns = univ)
      
        if hist.empty:
      
            raise ValueError("MSP: no complete weekly data")

        R = hist.values
      
        T = R.shape[0]
      
        α = float(level) / 100.0

        w = cp.Variable(n, nonneg = True)
      
        z = cp.Variable()
      
        u = cp.Variable(T, nonneg = True)

        cons = [
            cp.sum(w) == 1,
            w >= self.lb_arr, w <= self.ub_arr,
            u >= 0,
            u >= -(R @ w) - z
        ]
       
        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = max_industry_pct or self.default_max_industry_pct,
            max_sector_pct = max_sector_pct or self.default_max_sector_pct
        )
        
        cons += self._caps_constraints_param(
            w = w, 
            A_caps = A_caps, 
            caps_vec = caps_vec, 
            add_box = False
        )

        scores_arr = self.scores_arr
      
        f_expr = scores_arr @ w
      
        g_expr = z + (1.0 / (α * T)) * cp.sum(u)

        lam = 0.0
      
        w_val = None
      
        for _ in range(max_iter):
      
            prob = cp.Problem(cp.Maximize(f_expr - lam * g_expr), cons)
      
            if not self._solve(prob) or w.value is None:
      
                raise RuntimeError("MSP: subproblem failed")
      
            w_val = w.value

            f_val = float(scores_arr @ w_val)
           
            z_val = float(z.value)
           
            u_sum = float(np.sum(u.value))
           
            g_val = z_val + (1.0 / (α * T)) * u_sum

            delta = f_val - lam * g_val
           
            if abs(delta) < tol:
           
                break

            lam = f_val / max(g_val, 1e-12)

        if w_val is None:
           
            raise RuntimeError("MSP: Dinkelbach did not converge")

        w_opt = np.maximum(w_val, 0.0)
       
        w_opt /= w_opt.sum()
       
        return pd.Series(w_opt, index = univ, name = "msp")


    @staticmethod
    def _lhs_dirichlet1(
        m: int, 
        d: int, 
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Latin Hypercube sample ~ Dirichlet(1) over dimension d (simplex).
        Uses LHS of Exp(1) via U ~ Uniform strata, then normalize.
        """

        U = np.empty((m, d))

        for j in range(d):

            strata = (np.arange(m) + rng.random(m)) / m

            rng.shuffle(strata)

            U[:, j] = strata

        E = -np.log(U)  
       
        W = E / E.sum(axis = 1, keepdims = True)
      
        return W


    @staticmethod
    def _mix_base(
        W_stack: np.ndarray, 
        A_mix: np.ndarray
    ) -> np.ndarray:
        """
        Mix base solutions with mixing weights rows A_mix: shape (k, nbases),
        returns k portfolios on asset space. W_stack shape (nbases, nassets).
        """

        return A_mix @ W_stack


    def _project_feasible(
        self,
        target: np.ndarray,
        add_box: bool,
        max_industry_pct: float,
        max_sector_pct: float
    ) -> np.ndarray:
        """
        Solve: minimize ||w - target||^2  s.t. constraints.
        """
     
        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = max_industry_pct, 
            max_sector_pct = max_sector_pct
        )
        
        w = cp.Variable(self.n, nonneg = True)
       
        obj = cp.Minimize(cp.sum_squares(w - target))
       
        cons = [cp.sum(w) == 1]
       
        if add_box:
       
            cons += [w >= self.lb_arr, w <= self.ub_arr]
       
        if A_caps.shape[0] > 0:
       
            cons.append(A_caps @ w <= caps_vec)
       
        prob = cp.Problem(obj, cons)
       
        if not self._solve(prob) or w.value is None:

            w0 = np.clip(target, 0, None)

            s = w0.sum()

            return (w0 / s) if s > 0 else np.full(self.n, 1.0/self.n)

        return w.value


    def _feasible_seeds(
        self,
        nb_spikes: int,
        nb_rand_objs: int,
        max_industry_pct: float,
        max_sector_pct: float,
        rng: np.random.Generator
    ) -> List[np.ndarray]:
        """
        Project spikes e_i and a few 'random linear objective' solutions into feasible set.
        """

        seeds = []

        k = min(nb_spikes, self.n)

        for i in range(k):

            e = np.zeros(self.n)
            
            e[i] = 1.0

            seeds.append(
                
                self._project_feasible(
                    target = e, 
                    add_box = True,
                    max_industry_pct = max_industry_pct,
                    max_sector_pct = max_sector_pct
                )
            )
            
        for _ in range(nb_rand_objs):
           
            c = rng.random(self.n)

            A_caps, caps_vec = self._caps_mats(
                max_industry_pct = max_industry_pct, 
                max_sector_pct = max_sector_pct
            )
            
            w = cp.Variable(self.n, nonneg = True)
            
            cons = [cp.sum(w) == 1, w >= self.lb_arr, w <= self.ub_arr]
            
            if A_caps.shape[0] > 0:
            
                cons.append(A_caps @ w <= caps_vec)
            
            prob = cp.Problem(cp.Maximize(c @ w), cons)
            
            if self._solve(prob) and w.value is not None:
            
                seeds.append(w.value)
        
        return seeds


    def black_litterman_weights(
        self,
        delta: float = 2.5,
        tau: float = 0.02,
        max_industry_pct: Optional[float] = None,
        max_sector_pct: Optional[float] = None,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:

        if self._mu_bl is not None and self._sigma_bl is not None:

            mu_bl = self._mu_bl

            sigma_bl = self._sigma_bl

        else:

            if self._comb_std is None or self._sigma_prior is None or self._mcap is None:

                raise ValueError("BL inputs missing: comb_std, sigma_prior, mcap required.")

            tickers = self.universe

            mask = (self._scores > 0) & (self._er > 0)

            total = self._mcap[mask].sum()

            if total <= 0:

                raise ValueError("BL: nothing passes >0 screen")

            w_prior = pd.Series(0.0, index = tickers)

            w_prior.loc[mask] = self._mcap.loc[mask] / total

            P = pd.DataFrame(np.eye(len(tickers)), index = tickers, columns = tickers)
           
            Q = self._er
           
            Omega = pd.DataFrame(np.diag(self._comb_std.values ** 2), index = tickers, columns = tickers)
           
            Sigma_prior = self._sigma_prior

            mu_bl, sigma_bl = black_litterman(
                w_prior = w_prior, 
                sigma_prior = Sigma_prior,
                p = P, 
                q = Q, 
                omega = Omega, 
                delta = delta, 
                tau = tau
            )
            
            mu_bl = mu_bl.reindex(tickers)
          
            sigma_bl = sigma_bl.reindex(index = tickers, columns = tickers)
          
            self._mu_bl = mu_bl
            
            self._sigma_bl = sigma_bl

        μ = mu_bl.values
        
        Σb = make_pd(
            mat = sigma_bl.values, 
            tol = 1e-10
        )
        
        Lb = np.linalg.cholesky(Σb)
        
        Ab = Lb.T
        
        rf = float(self.rf_ann)

        w = self._dinkelbach_sharpe_max(
            μ = μ, 
            A = Ab, 
            rf = rf,
            max_industry_pct = (max_industry_pct or self.default_max_industry_pct),
            max_sector_pct = (max_sector_pct or self.default_max_sector_pct),
            tol = tol, 
            max_iter = max_iter,
        )
        
        w_bl = pd.Series(w, index = self.universe, name = "BL Weight")
        
        return w_bl, mu_bl, sigma_bl


    def _initial_mixes_L2(
        self, 
        w_msr, 
        w_sortino, 
        w_mir,
        w_bl, 
        w_msp, 
        sample_size = 2000, 
        random_state = 42
    ):
        
        rng = np.random.default_rng(random_state)
        
        W = np.vstack([w_msr, w_sortino, w_mir, w_bl, w_msp])

        A_mix = self._lhs_dirichlet1(
            m = sample_size, 
            d = 5, 
            rng = rng
        )
        
        W_mix = self._mix_base(
            W_stack = W, 
            A_mix = A_mix
        )

        T = self.T1

        rf_ann = self.rf_ann

        rf_week = self.rf_week

        if self._mu_bl is not None and self._sigma_bl is not None:

            Σb = make_pd(
                mat = self._sigma_bl.values
            )

        else:
            
            Σb = make_pd(
                mat = self._cov
            )
       
        LbT = np.linalg.cholesky(Σb).T

        scores = np.zeros(W_mix.shape[0])
       
        LcT = self.Lc.T
       
        for i, w in enumerate(W_mix):
       
            ret = float(self.er_arr @ w)
       
            vol = float(np.linalg.norm(LcT @ w))
       
            dd_week = float(np.linalg.norm(np.maximum(rf_week - (self.R1 @ w), 0.0)) / np.sqrt(T))
       
            dd_ann = dd_week * self.sqrt52

            if self._mu_bl is not None:
       
                bl_ret = float(self._mu_bl.values @ w)
       
                bl_vol = float(np.linalg.norm(LbT @ w))
       
            else:
       
                bl_ret = ret
                
                bl_vol = vol

            exc = ret - rf_ann
            
            sh = exc / max(vol, 1e-12)
            
            so = exc / max(dd_ann, 1e-12)
            
            bl = (bl_ret - rf_ann) / max(bl_vol, 1e-12)

            mir_pen = float(np.sum((w - w_mir) ** 2))
            
            msp_pen = float(np.sum((w - w_msp) ** 2))
            
            pen_comb = ((np.sum((w - w_mir)**2) / mir_pen) + (np.sum((w - w_msp) ** 2) / msp_pen))
            
            scores[i] = sh + so + bl - pen_comb

        return W_mix[np.argmax(scores)]


    def _cvar_ru_smooth(
        self,
        r: np.ndarray,     
        alpha: float,
        beta: float = 50.0 
    ) -> Tuple[float, float, np.ndarray]:
        """
        Smooth Rockafellar–Uryasev CVaR:
        CVaR_α ≈ min_z [ z + (1/(αT)) Σ softplus( -(r_t + z) ) ]
        Returns (cvar, z*, weights grad wrt r: dCVaR/dr = -(1/(αT)) * σ(beta*(−r−z))
        where σ is logistic. We also return the per-sample softplus'(a) = 1/(1+exp(-β a)).
        """
      
        T = r.shape[0]
      
        α = float(alpha)

        q_low = np.percentile(-r, 100 * α * 0.5)
       
        q_hi = np.percentile(-r, 100 * min(0.99, α * 1.5))

        def F(
            z
        ):
            
            a = -(r + z)
            
            return z + (softplus(
                x = a, 
                beta = beta
            ).sum() / (α * T))
       
        def dF(
            z
        ):
            
            a = -(r + z)

            sig = 1.0 / (1.0 + np.exp(-beta * a))

            return 1.0 - (sig.sum() / (α * T))

        lo = q_low
        
        hi = q_hi
        
        for _ in range(40):
        
            mid = 0.5*(lo+hi)
        
            g = dF(mid)
        
            if g > 0:
        
                lo = mid
        
            else:
        
                hi = mid
        
        z_star = 0.5 * (lo + hi)
       
        a = -(r + z_star)
       
        sp = softplus(a, beta = beta)
       
        cvar = z_star + sp.sum() / (α * T)
       
        sig = 1.0 / (1.0 + np.exp(-beta * a))  

        dC_dr = -(sig / (α * T))
       
        return float(cvar), float(z_star), dC_dr


    def _calibrate_scales_by_grad_rewards_penalties(
        self,
        W_stack: np.ndarray,
        μ_bl_arr: np.ndarray,
        w_mir_arr: np.ndarray,
        w_msp_arr: np.ndarray,
        gamma: tuple,
        sample_size: int = 2000,
        random_state: int = 42,
        use_l1_penalty: bool = False,
        huber_delta: float = 1e-4,
        eps: float = 1e-12,
    ):
        """
        Calibrate SH, SO, BL and penalties by equalising average gradient 2-norms
        in the same geometry as used in the CCP step. Uses LHS sampling and
        feasible spikes/random LP seeds.
        """
        
        rng = np.random.default_rng(random_state)

        A_mix = self._lhs_dirichlet1(
            m = sample_size, 
            d = W_stack.shape[0], 
            rng = rng
        )
        
        W_rand = self._mix_base(
            W_stack = W_stack, 
            A_mix = A_mix
        )

        seeds = self._feasible_seeds(
            nb_spikes = min(10, self.n),
            nb_rand_objs = 10,
            max_industry_pct = self.default_max_industry_pct,
            max_sector_pct = self.default_max_sector_pct,
            rng = rng
        )
        
        if len(seeds) > 0:
          
            W_rand = np.vstack([W_rand, np.vstack(seeds)])

        Σ = self.Σ
        
        LcT = self.Lc.T
        
        R1 = self.R1
        
        T = self.T1
        
        rf_a = self.rf_ann
        
        rf_w = self.rf_week
        
        s52 = self.sqrt52

        if self._sigma_bl is not None:
           
            Σb = make_pd(
                mat = self._sigma_bl.values
            )
       
        else:
       
            Σb = Σ
       
        LbT = np.linalg.cholesky(Σb).T

        g_sh = np.zeros_like(W_rand)
       
        g_so = np.zeros_like(W_rand)
       
        g_bl = np.zeros_like(W_rand)

        if use_l1_penalty:

            den_mir = max(np.mean(np.sum(np.abs(W_rand - w_mir_arr), axis = 1)), eps)

            den_msp = max(np.mean(np.sum(np.abs(W_rand - w_msp_arr), axis = 1)), eps)

            g_pen = np.sign(W_rand - w_mir_arr) / max(den_mir, eps) + np.sign(W_rand - w_msp_arr) / max(den_msp, eps)
       
        else:
       
            den_mir = max(np.mean(np.sum((W_rand - w_mir_arr) ** 2, axis=1)), eps)
       
            den_msp = max(np.mean(np.sum((W_rand - w_msp_arr) ** 2, axis=1)), eps)

            g_pen = (2.0 * (W_rand - w_mir_arr)) / max(den_mir, eps) + (2.0 * (W_rand - w_msp_arr)) / max(den_msp, eps)

        for i, w in enumerate(W_rand):

            Σw = Σ @ w

            Σbw = Σb @ w

            ret = float(self.er_arr @ w)

            vol = float(np.linalg.norm(LcT @ w)) 
            
            vol_eps = max(vol, eps)

            g_sh[i] = (self.er_arr * vol - (ret - rf_a) * (Σw / vol_eps)) / (vol_eps ** 2)

            u = rf_w - (R1 @ w)

            pos = u > 0

            if np.any(pos):

                M = R1[pos]

                dd_week = np.linalg.norm(u[pos]) / np.sqrt(T)
                
                dd_week = max(dd_week, eps)

                dd_ann = dd_week * s52

                grad_dd_week = -(M.T @ u[pos]) / (T * dd_week)

                grad_dd = grad_dd_week * s52

                g_so[i] = (self.er_arr * dd_ann - (ret - rf_a) * grad_dd) / (dd_ann ** 2)

            else:

                g_so[i] = 0.0

            bl_ret = float(μ_bl_arr @ w)

            bl_vol = float(np.linalg.norm(LbT @ w))
            
            bl_eps = max(bl_vol, eps)

            g_bl[i] = (μ_bl_arr * bl_vol - (bl_ret - rf_a) * (Σbw / bl_eps)) / (bl_eps ** 2)

        def avg_norm(
            G
        ): 
            return float(np.mean(np.linalg.norm(G, axis = 1)))

        A_s, A_so, A_bl = map(avg_norm, (g_sh, g_so, g_bl))
        
        A_pen = avg_norm(
            G = g_pen
        )

        γ_s, γ_so, γ_bl, γ_pen, _ = gamma 
       
        target = np.mean([γ_s*A_s, γ_so*A_so, γ_bl*A_bl, γ_pen*A_pen])

        scale_s = target / (γ_s * max(A_s, eps))
      
        scale_so = target / (γ_so * max(A_so, eps))
      
        scale_bl = target / (γ_bl * max(A_bl, eps))
      
        scale_pen = target / (γ_pen * max(A_pen, eps))

        return (scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, Σb, LbT)

    
    def _calibrate_scales_by_grad_ir_sc(
        self,
        W_stack: np.ndarray,
        μ_bl_arr: np.ndarray,
        gamma: tuple,
        scores_arr: np.ndarray,
        R1_local: np.ndarray,
        b_te: Optional[np.ndarray],
        cvar_level: float = 5.0,
        sample_size: int = 2000,
        random_state: int = 42,
        eps: float = 1e-12
    ):
        """
        Calibrate SH, SO, BL, IR, SC (score/CVaR) by equalising average gradient norms.
        Uses LHS sampling + smoothed RU CVaR for SC gradient.
        """
        
        rng = np.random.default_rng(random_state)
        
        A_mix = self._lhs_dirichlet1(
            m = sample_size, 
            d = W_stack.shape[0], 
            rng = rng
        )
        
        W_rand = self._mix_base(
            W_stack = W_stack, 
            A_mix = A_mix
        )

        seeds = self._feasible_seeds(
            nb_spikes = min(10, self.n),
            nb_rand_objs = 10,
            max_industry_pct = self.default_max_industry_pct,
            max_sector_pct = self.default_max_sector_pct,
            rng=rng
        )
        
        if len(seeds) > 0:
        
            W_rand = np.vstack([W_rand, np.vstack(seeds)])

        Σ = self.Σ
        
        LcT = self.Lc.T
        
        rf_a = self.rf_ann
        
        rf_w = self.rf_week
        
        s52 = self.sqrt52

        if self._sigma_bl is not None:

            Σb = make_pd(self._sigma_bl.values)

        else:

            Σb = Σ

        LbT = np.linalg.cholesky(Σb).T

        T = R1_local.shape[0]

        ones_T = np.ones(T)

        if b_te is not None:
            
            b_vec = b_te 
        
        else:
            
            b_vec = np.full(T, rf_w)

        g_sh = np.zeros_like(W_rand)
      
        g_so = np.zeros_like(W_rand)
      
        g_bl = np.zeros_like(W_rand)
      
        g_ir = np.zeros_like(W_rand)
      
        g_sc = np.zeros_like(W_rand)

        for i, w in enumerate(W_rand):
      
            Σw = Σ @ w
      
            Σbw = Σb @ w

            ret = float(self.er_arr @ w)
           
            vol = float(np.linalg.norm(LcT @ w))
           
            vol_eps = max(vol, eps)
           
            g_sh[i] = (self.er_arr * vol - (ret - rf_a) * (Σw / vol_eps)) / (vol_eps ** 2)

            u = rf_w - (R1_local @ w)

            pos = u > 0

            if np.any(pos):

                M = R1_local[pos]

                dd_week = np.linalg.norm(u[pos]) / np.sqrt(T)
                
                dd_week = max(dd_week, eps)

                dd_ann = dd_week * s52

                grad_dd_week = -(M.T @ u[pos]) / (T * dd_week)

                grad_dd = grad_dd_week * s52
               
                g_so[i] = (self.er_arr * dd_ann - (ret - rf_a) * grad_dd) / (dd_ann ** 2)
            
            else:
            
                g_so[i] = 0.0

            bl_ret = float(μ_bl_arr @ w)
           
            bl_vol = float(np.linalg.norm(LbT @ w))
            
            bl_eps = max(bl_vol, eps)
            
            g_bl[i] = (μ_bl_arr * bl_vol - (bl_ret - rf_a) * (Σbw / bl_eps)) / (bl_eps ** 2)

            a = (R1_local @ w) - b_vec

            mean_a = float(a.mean())

            te = float(np.linalg.norm(a) / np.sqrt(T))
            
            te_eps = max(te, eps)

            grad_m = (R1_local.T @ ones_T) / T
            
            dte = (R1_local.T @ a) / (te_eps * T)
            
            g_ir[i] = (grad_m * te_eps - mean_a * dte) / (te_eps ** 2)

            r_week = R1_local @ w

            cvar, z_star, dC_dr = self._cvar_ru_smooth(
                r = r_week, 
                alpha = cvar_level/100.0, 
                beta = 50.0
            )

            g_cvar = (R1_local.T @ dC_dr)

            f = float(self.scores_arr @ w)

            df = self.scores_arr

            cvar_eps = max(cvar, eps)

            g_sc[i] = (df * cvar_eps - f * g_cvar) / (cvar_eps ** 2)

        def avg_norm(
            G
        ): 
            
            return float(np.mean(np.linalg.norm(G, axis = 1)))

        A_s, A_so, A_bl = map(avg_norm, (g_sh, g_so, g_bl))
        
        A_ir = avg_norm(
            G = g_ir
        )
        
        A_sc = avg_norm(
            G = g_sc
        )

        γ_s, γ_so, γ_bl, γ_ir, γ_sc = gamma
        
        target = np.mean([γ_s * A_s, γ_so * A_so, γ_bl * A_bl, γ_ir * A_ir, γ_sc * A_sc])

        scale_s = target / (γ_s * max(A_s, eps))
        
        scale_so = target / (γ_so * max(A_so, eps))
        
        scale_bl = target / (γ_bl * max(A_bl, eps))
        
        scale_ir = target / (γ_ir * max(A_ir, eps))
        
        scale_sc = target / (γ_sc * max(A_sc, eps))

        return (scale_s, scale_so, scale_bl, scale_ir, scale_sc, Σb, LbT)


    def last_diagnostics(
        self
    ) -> Optional[Dict[str, float]]:
        """
        Return diagnostic 'push' magnitudes from the last comb_port* call.
        """
        
        return self._last_diag


    def comb_port(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        sample_size: int = 5000,
        random_state: int = 42,
        max_iter: int = 1000,
        tol: float = 1e-12,
        w_msr: Optional[pd.Series] = None,
        w_sortino: Optional[pd.Series] = None,
        w_mir: Optional[pd.Series] = None,
        w_bl: Optional[pd.Series] = None,
        w_msp: Optional[pd.Series] = None,
        huber_delta_l1: float = 1e-4, 
    ) -> pd.Series:

        if self.gamma is not None:
            
            gamma = self.gamma

        tol_floor = (np.trace(self.Σ) / max(1, self.n)) * 1e-8

        tol = max(tol, tol_floor)

        if w_msr is not None:
            
            w_msr = w_msr  
        
        else:
            
            w_msr = self.msr()
        
        if w_sortino is not None:
            
            w_sort = w_sortino  
        
        else:
            
            w_sort = self.msr_sortino()
        
        if w_mir is not None:
            
            w_mir = w_mir  
        
        else:
            
            w_mir = self.MIR()

        if w_msp is not None:
            
            w_msp = w_msp
        
        else:
            
            w_msp = self.msp()
            
        if w_bl is None or self._mu_bl is None or self._sigma_bl is None:
            
            w_bl, mu_bl, _ = self.black_litterman_weights()
       
        else:
       
            mu_bl = self._mu_bl

        w_msr_a = w_msr.values
        
        w_so_a = w_sort.values
        
        w_mir_a = w_mir.values
        
        w_bl_a = w_bl.values
        
        w_msp_a = w_msp.values
        
        μ_bl_arr = mu_bl.values
        
        W_stack = np.vstack([w_msr_a, w_so_a, w_mir_a, w_bl_a, w_msp_a])

        init_mix = self._initial_mixes_L2(
            w_msr = w_msr_a, 
            w_sortino = w_so_a, 
            w_mir = w_mir_a, 
            w_bl = w_bl_a, 
            w_msp = w_msp_a,
            sample_size = 2000, 
            random_state = random_state
        )

        initials = [w_msr_a, w_so_a, w_mir_a, w_bl_a, w_msp_a, init_mix]

        scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, Σb_cal, LbT = \
            self._calibrate_scales_by_grad_rewards_penalties(
                W_stack=W_stack,
                μ_bl_arr=μ_bl_arr,
                w_mir_arr=w_mir_a,
                w_msp_arr=w_msp_a,
                gamma=gamma,
                sample_size=sample_size,
                random_state=random_state,
                use_l1_penalty=False,
            )

        Σ = self.Σ

        LcT = self.Lc.T

        R1 = self.R1

        T1 = self.T1

        rf_ann = self.rf_ann

        rf_week = self.rf_week

        sqrt52 = self.sqrt52

        lb_arr, ub_arr = self.lb_arr, self.ub_arr

        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = self.default_max_industry_pct, 
            max_sector_pct = self.default_max_sector_pct
        )

        w_var = cp.Variable(self.n, nonneg = True)
        
        lin_param = cp.Parameter(self.n)

        pen_comb = (cp.sum_squares(w_var - w_mir_a) / den_mir) + (cp.sum_squares(w_var - w_msp_a) / den_msp)
       
        obj_expr = gamma[3] * scale_pen * pen_comb - cp.sum(cp.multiply(lin_param, w_var))
       
        cons = [cp.sum(w_var) == 1, w_var >= lb_arr, w_var <= ub_arr]
       
        if A_caps.shape[0] > 0:
       
            cons.append(A_caps @ w_var <= caps_vec)
       
        prob = cp.Problem(cp.Minimize(obj_expr), cons)


        def _from_initial(
            w0: np.ndarray
        ) -> Tuple[np.ndarray, float, Dict[str, float]]:
            
            w = w0.copy()
            
            prev_obj = None
            
            for _ in range(max_iter):

                Σw = Σ @ w

                ret = float(self.er_arr @ w)

                vol = float(np.linalg.norm(LcT @ w))
                
                vol_eps = max(vol, 1e-12)

                grad_sh = (self.er_arr * vol - (ret - rf_ann) * (Σw / vol_eps)) / (vol_eps ** 2)

                u = rf_week - (R1 @ w)

                pos = (u > 0)

                if np.any(pos):

                    M = R1[pos]

                    dd_week = np.linalg.norm(u[pos]) / np.sqrt(T1)
                    
                    dd_week = max(dd_week, 1e-12)

                    dd_ann = dd_week * sqrt52

                    grad_dd_week = - (M.T @ u[pos]) / (T1 * dd_week)

                    grad_dd = grad_dd_week * sqrt52
                    
                    grad_so = (self.er_arr * dd_ann - (ret - rf_ann) * grad_dd) / (dd_ann ** 2)
                
                else:
                
                    grad_so = np.zeros(self.n)

                Σb_w = Σb_cal @ w

                bl_ret = float(μ_bl_arr @ w)

                bl_vol = float(np.linalg.norm(LbT @ w))
                
                bl_eps = max(bl_vol, 1e-12)

                grad_bl = (μ_bl_arr * bl_vol - (bl_ret - rf_ann) * (Σb_w / bl_eps)) / (bl_eps ** 2)

                lin = gamma[0] * scale_s * grad_sh + gamma[1] * scale_so * grad_so + gamma[2] * scale_bl * grad_bl

                lin_param.value = lin

                if not self._solve(prob) or w_var.value is None:

                    raise RuntimeError("comb_port CCP: solvers failed")

                w_new = w_var.value

                obj_val = float(obj_expr.value)
               
                dw = float(np.linalg.norm(w_new - w))

                if prev_obj is not None and dw < tol and abs(obj_val - prev_obj) < tol:
                
                    w = w_new
                
                    break

                prev_obj = obj_val
                
                w = w_new

            Σw = Σ @ w

            vol = max(float(np.linalg.norm(LcT @ w)), 1e-12)

            dd_week = np.linalg.norm(np.maximum(rf_week - (R1 @ w), 0.0)) / np.sqrt(T1)

            dd_ann = max(dd_week * sqrt52, 1e-12)

            bl_vol = max(float(np.linalg.norm(LbT @ w)), 1e-12)

            sharpe = (float(self.er_arr @ w) - rf_ann) / vol

            sortino = (float(self.er_arr @ w) - rf_ann) / dd_ann

            bl_sh = (float(μ_bl_arr @ w) - rf_ann) / bl_vol

            pen_val = float(np.sum((w - w_mir_a) ** 2)) / den_mir + float(np.sum((w - w_msp_a) ** 2)) / den_msp

            score = (
                gamma[0]*scale_s*sharpe 
                + gamma[1]*scale_so*sortino 
                + gamma[2]*scale_bl*bl_sh
                - gamma[3] * scale_pen * pen_val
            )

            ret = float(self.er_arr @ w)
            
            vol = float(np.linalg.norm(LcT @ w))
            
            vol_eps = max(vol, 1e-12)
            
            g_sh = (self.er_arr * vol - (ret - rf_ann) * (Σw / vol_eps)) / (vol_eps ** 2)

            u = rf_week - (R1 @ w)
            
            pos = (u > 0)
            
            if np.any(pos):
            
                M = R1[pos]
            
                dd_week = np.linalg.norm(u[pos]) / np.sqrt(T1)
                
                dd_week = max(dd_week, 1e-12)
            
                dd_ann = dd_week * sqrt52
            
                grad_dd_week = -(M.T @ u[pos]) / (T1 * dd_week)
            
                grad_dd = grad_dd_week * sqrt52
            
                g_so = (self.er_arr * dd_ann - (ret - rf_ann) * grad_dd) / (dd_ann ** 2)
            
            else:
            
                g_so = np.zeros(self.n)

            Σb_w = Σb_cal @ w
            
            bl_ret = float(μ_bl_arr @ w)
            
            bl_vol = float(np.linalg.norm(LbT @ w))
            
            bl_eps = max(bl_vol, 1e-12)
            
            g_bl = (μ_bl_arr * bl_vol - (bl_ret - rf_ann) * (Σb_w / bl_eps)) / (bl_eps ** 2)

            g_pen = (2.0 * (w - w_mir_a)) / den_mir + (2.0 * (w - w_msp_a)) / den_msp

            diag = {
                "Sharpe_push": gamma[0] * scale_s * float(np.linalg.norm(g_sh)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(g_so)),
                "BL_push": gamma[2] * scale_bl * float(np.linalg.norm(g_bl)),
                "Penalty_push": gamma[3] * scale_pen * float(np.linalg.norm(g_pen)),
            }
            
            return w, score, diag


        best_w = None
       
        best_s = -np.inf
       
        best_diag = {}
       
        for w0 in initials:
       
            try:
       
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
       
            except RuntimeError:
       
                continue
       
            if s_cand > best_s:
       
                best_s = s_cand
       
                best_w = w_cand
       
                best_diag = diag

        if best_w is None:
           
            raise RuntimeError("comb_port: all initialisations failed")

        
        self._last_diag = best_diag
      
        return pd.Series(best_w, index = self.universe, name = "comb_port")


    def comb_port1(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        cvar_level: float = 5.0,
        sample_size: int = 5000,
        random_state: int = 42,
        max_iter: int = 1000,
        tol: float = 1e-12,
        w_msr: Optional[pd.Series] = None,
        w_sortino: Optional[pd.Series] = None,
        w_mir: Optional[pd.Series] = None,
        w_bl: Optional[pd.Series] = None,
        w_msp: Optional[pd.Series] = None,
    ) -> pd.Series:

        if self.gamma is not None:
         
            gamma = self.gamma

        tol_floor = (np.trace(self.Σ) / max(1, self.n)) * 1e-8

        tol = max(tol, tol_floor)

        if w_msr is not None:
            
            w_msr = w_msr  
        
        else:
            
            w_msr = self.msr()
        
        if w_sortino is not None:
            
            w_sort = w_sortino  
        
        else:
            
            w_sort = self.msr_sortino()
        
        if w_mir is not None:
            
            w_mir = w_mir  
        
        else:
            
            w_mir = self.MIR()
        
        if w_msp is not None:
            
            w_msp = w_msp
        
        else:
            
            w_msp = self.msp()

        if w_bl is None or self._mu_bl is None or self._sigma_bl is None:
            
            w_bl, mu_bl, _ = self.black_litterman_weights()
        
        else:
        
            mu_bl = self._mu_bl

        w_msr_a = w_msr.values
       
        w_so_a =  w_sort.values
      
        w_mir_a = w_mir.values
      
        w_bl_a = w_bl.values
      
        w_msp_a = w_msp.values
      
        μ_bl_arr = mu_bl.values
      
        W_stack = np.vstack([w_msr_a, w_so_a, w_mir_a, w_bl_a, w_msp_a])

        init_mix = self._initial_mixes_L2(
            w_msr = w_msr_a, 
            w_sortino = w_so_a, 
            w_mir = w_mir_a, 
            w_bl = w_bl_a,
            w_msp = w_msp_a,
            sample_size = 2000, 
            random_state = random_state
        )

        initials = [w_msr_a, w_so_a, w_mir_a, w_bl_a, w_msp_a, init_mix]

        hist_df = self._weekly_ret_1y.reindex(columns = self.universe).dropna(how = "any")
        
        if hist_df is None or hist_df.empty:
        
            raise RuntimeError("comb_port1: no overlapping weekly history")

        b_te = None
        
        if self._benchmark_weekly is not None:
        
            b_aligned = self._benchmark_weekly.reindex(hist_df.index)
        
            ok = b_aligned.notna()
        
            if ok.any():
        
                hist_df = hist_df.loc[ok]
        
                b_te = b_aligned.loc[ok].to_numpy()
        
            else:
        
                b_te = None

        R1_local = hist_df.values
        
        T1_local = R1_local.shape[0]

        scale_s, scale_so, scale_bl, scale_ir, scale_sc, Σb_cal, LbT = \
            self._calibrate_scales_by_grad_ir_sc(
                W_stack=W_stack,
                μ_bl_arr=μ_bl_arr,
                gamma=gamma,
                scores_arr=self.scores_arr,
                R1_local=R1_local,
                b_te=b_te,
                cvar_level=cvar_level,
                sample_size=sample_size,
                random_state=random_state
            )

        Σ = self.Σ

        LcT = self.Lc.T

        rf_ann = self.rf_ann

        rf_week = self.rf_week

        sqrt52 = self.sqrt52

        lb_arr = self.lb_arr
       
        ub_arr = self.ub_arr

        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = self.default_max_industry_pct, 
            max_sector_pct = self.default_max_sector_pct
        )

        ones_local = np.ones(T1_local)

        if b_te is not None:
            
            b_vec = b_te  
        
        else:
        
            b_vec = np.full(T1_local, rf_week)

        w_var = cp.Variable(self.n, nonneg = True)
        
        lin_param = cp.Parameter(self.n)
        
        obj_expr = -cp.sum(cp.multiply(lin_param, w_var))
        
        cons = [cp.sum(w_var) == 1, w_var >= lb_arr, w_var <= ub_arr]
        
        if A_caps.shape[0] > 0:
        
            cons.append(A_caps @ w_var <= caps_vec)
        
        prob = cp.Problem(cp.Minimize(obj_expr), cons)

        def _from_initial(
            w0: np.ndarray
        ) -> Tuple[np.ndarray, float, Dict[str, float]]:
            
            w = w0.copy()
            
            prev_obj = None
            
            for _ in range(max_iter):

                Σw = Σ @ w

                Σb_w = Σb_cal @ w

                ret = float(self.er_arr @ w)

                vol = float(np.linalg.norm(LcT @ w))
                
                vol_eps = max(vol, 1e-12)

                grad_sh = (self.er_arr * vol - (ret - rf_ann) * (Σw / vol_eps)) / (vol_eps ** 2)

                u = rf_week - (R1_local @ w)

                pos = (u > 0)

                if np.any(pos):

                    M = R1_local[pos]

                    dd_week = np.linalg.norm(u[pos]) / np.sqrt(T1_local)
                    
                    dd_week = max(dd_week, 1e-12)

                    dd_ann = dd_week * sqrt52
                    
                    grad_dd_week = -(M.T @ u[pos]) / (T1_local * dd_week)
                    
                    grad_dd = grad_dd_week * sqrt52
                    
                    grad_so = (self.er_arr * dd_ann - (ret - rf_ann) * grad_dd) / (dd_ann ** 2)
                
                else:
                
                    grad_so = np.zeros(self.n)

                bl_ret = float(μ_bl_arr @ w)

                bl_vol = float(np.linalg.norm(LbT @ w))
                
                bl_eps = max(bl_vol, 1e-12)

                grad_bl = (μ_bl_arr * bl_vol - (bl_ret - rf_ann) * (Σb_w / bl_eps)) / (bl_eps ** 2)

                a = (R1_local @ w) - b_vec

                mean_a = float(a.mean())

                te = float(np.linalg.norm(a) / np.sqrt(T1_local))
                
                te_eps = max(te, 1e-12)
                
                grad_m = (R1_local.T @ ones_local) / T1_local
                
                dte = (R1_local.T @ a) / (te_eps * T1_local)
                
                grad_ir = (grad_m * te_eps - mean_a * dte) / (te_eps ** 2)

                r_week = R1_local @ w

                cvar, z_star, dC_dr = self._cvar_ru_smooth(
                    r = r_week, 
                    alpha = cvar_level / 100.0, 
                    beta = 50.0
                )
                
                g_cvar = (R1_local.T @ dC_dr)
                
                f = float(self.scores_arr @ w)
                
                df = self.scores_arr
                
                cvar_eps = max(cvar, 1e-12)
                
                grad_sc = (df * cvar_eps - f * g_cvar) / (cvar_eps ** 2)

                lin = (
                    gamma[0]*scale_s  * grad_sh
                    + gamma[1]*scale_so* grad_so
                    + gamma[2]*scale_bl* grad_bl
                    + gamma[3]*scale_ir* grad_ir
                    + gamma[4]*scale_sc* grad_sc
                )

                lin_param.value = lin
                
                if not self._solve(prob) or w_var.value is None:
                
                    raise RuntimeError("comb_port1 CCP: solvers failed")
                
                w_new = w_var.value

                obj_val = float(obj_expr.value)
                
                dw = float(np.linalg.norm(w_new - w))

                if prev_obj is not None and dw < tol and abs(obj_val - prev_obj) < tol:
                
                    w = w_new
                
                    break

                prev_obj = obj_val
                
                w = w_new

            Σw = Σ @ w

            vol = max(float(np.linalg.norm(LcT @ w)), 1e-12)

            dd_week = np.linalg.norm(np.maximum(self.rf_week - (R1_local @ w), 0.0)) / np.sqrt(T1_local)

            dd_ann = max(dd_week * self.sqrt52, 1e-12)

            bl_vol = max(float(np.linalg.norm(LbT @ w)), 1e-12)

            sharpe = (float(self.er_arr @ w) - self.rf_ann) / vol

            sortino = (float(self.er_arr @ w) - self.rf_ann) / dd_ann

            bl_sh = (float(μ_bl_arr @ w) - self.rf_ann) / bl_vol

            a = (R1_local @ w) - b_vec

            ir_h = float(a.mean()) / max(float(np.linalg.norm(a) / np.sqrt(T1_local)), 1e-12)

            r_week = R1_local @ w

            cvar, _, _ = self._cvar_ru_smooth(
                r = r_week, 
                alpha = cvar_level / 100.0, 
                beta = 50.0
            )

            sc_ratio = float(self.scores_arr @ w) / max(cvar, 1e-12)

            score = (
                gamma[0] * scale_s * sharpe 
                + gamma[1] * scale_so * sortino 
                + gamma[2] * scale_bl * bl_sh
                + gamma[3] * scale_ir * ir_h 
                + gamma[4] * scale_sc * sc_ratio
            )

            ret = float(self.er_arr @ w)

            vol = float(np.linalg.norm(LcT @ w))
            
            vol_eps = max(vol, 1e-12)

            g_sh = (self.er_arr * vol - (ret - self.rf_ann) * (Σw / vol_eps)) / (vol_eps ** 2)

            u = self.rf_week - (R1_local @ w)
           
            pos = (u > 0)
           
            if np.any(pos):
           
                M = R1_local[pos]
           
                dd_week = np.linalg.norm(u[pos]) / np.sqrt(T1_local)
                
                dd_week = max(dd_week, 1e-12)
                
                dd_ann = dd_week * self.sqrt52
                
                grad_dd_week = -(M.T @ u[pos]) / (T1_local * dd_week)
                
                grad_dd = grad_dd_week * self.sqrt52
                
                g_so = (self.er_arr * dd_ann - (ret - self.rf_ann) * grad_dd) / (dd_ann ** 2)
            
            else:
            
                g_so = np.zeros(self.n)

            Σb_w = Σb_cal @ w
            
            bl_ret = float(μ_bl_arr @ w)
            
            bl_vol = float(np.linalg.norm(LbT @ w))
            
            bl_eps = max(bl_vol, 1e-12)
            
            g_bl = (μ_bl_arr * bl_vol - (bl_ret - self.rf_ann) * (Σb_w / bl_eps)) / (bl_eps ** 2)

            a = (R1_local @ w) - b_vec
            
            mean_a = float(a.mean())
            
            te = float(np.linalg.norm(a) / np.sqrt(T1_local))
            
            te_eps = max(te, 1e-12)
            
            grad_m = (R1_local.T @ ones_local) / T1_local
            
            dte = (R1_local.T @ a) / (te_eps * T1_local)
            
            g_ir = (grad_m * te_eps - mean_a * dte) / (te_eps ** 2)

            cvar, z_star, dC_dr = self._cvar_ru_smooth(
                r = R1_local @ w, 
                alpha = cvar_level / 100.0, 
                beta = 50.0
            )
            
            g_cvar = (R1_local.T @ dC_dr)
            
            f = float(self.scores_arr @ w)
            
            df = self.scores_arr
            
            cvar_eps = max(cvar, 1e-12)
            
            g_sc = (df * cvar_eps - f * g_cvar) / (cvar_eps ** 2)

            diag = {
                "Sharpe_push": gamma[0] * scale_s * float(np.linalg.norm(g_sh)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(g_so)),
                "BL_push": gamma[2] * scale_bl * float(np.linalg.norm(g_bl)),
                "IR_push": gamma[3] * scale_ir * float(np.linalg.norm(g_ir)),
                "SC_push": gamma[4] * scale_sc * float(np.linalg.norm(g_sc)),
            }
           
            return w, score, diag


        best_w = None
      
        best_s = np.inf
      
        best_diag = {}
      
        for w0 in initials:
      
            try:
      
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
      
            except RuntimeError:
               
                continue
            
            if s_cand > best_s:
            
                best_s = s_cand
            
                best_w = w_cand
            
                best_diag = diag

        if best_w is None:
            
            raise RuntimeError("comb_port1: all initialisations failed")

        self._last_diag = best_diag
        
        return pd.Series(best_w, index = self.universe, name = "comb_port1")


    def comb_port2(
        self,
        gamma: tuple = (1.0, 1.0, 1.0, 1.0, 1.0),
        sample_size: int = 5000,
        random_state: int = 42,
        max_iter: int = 1000,
        tol: float = 1e-12,
        w_msr: Optional[pd.Series] = None,
        w_sortino: Optional[pd.Series] = None,
        w_mir: Optional[pd.Series] = None,
        w_bl: Optional[pd.Series] = None,
        w_msp: Optional[pd.Series] = None,
        huber_delta_l1: float = 1e-4,   
    ) -> pd.Series:

        if self.gamma is not None:

            gamma = self.gamma

        tol_floor = (np.trace(self.Σ) / max(1, self.n)) * 1e-8

        tol = max(tol, tol_floor)

        if w_msr is not None:
            
            w_msr = w_msr  
        
        else:
            
            w_msr = self.msr()
        
        if w_sortino is not None:
            
            w_sort = w_sortino  
        
        else:
            
            w_sort = self.msr_sortino()
        
        if w_mir is not None:
            
            w_mir = w_mir  
        
        else:
            
            w_mir = self.MIR()
        
        if w_msp is not None:
            
            w_msp = w_msp
        
        else:
            
            w_msp = self.msp()

        if w_bl is None or self._mu_bl is None or self._sigma_bl is None:
            
            w_bl, mu_bl, _ = self.black_litterman_weights()
        
        else:
        
            mu_bl = self._mu_bl

        if w_bl is None or self._mu_bl is None or self._sigma_bl is None:
        
            w_bl, mu_bl, _ = self.black_litterman_weights()
        
        else:
        
            mu_bl = self._mu_bl

        w_msr_a = w_msr.values
       
        w_so_a =  w_sort.values
      
        w_mir_a = w_mir.values
      
        w_bl_a = w_bl.values
      
        w_msp_a = w_msp.values
      
        μ_bl_arr = mu_bl.values

        W_stack = np.vstack([w_msr_a, w_so_a, w_mir_a, w_bl_a, w_msp_a])

        init_mix = self._initial_mixes_L2(
            w_msr = w_msr_a, 
            w_sortino = w_so_a,
            w_mir = w_mir_a, 
            w_bl = w_bl_a,
            w_msp = w_msp_a,
            sample_size = 2000, 
            random_state = random_state
        )

        initials = [w_msr_a, w_so_a, w_mir_a, w_bl_a, w_msp_a, init_mix]

        scale_s, scale_so, scale_bl, scale_pen, den_mir, den_msp, Σb_cal, LbT = \
            self._calibrate_scales_by_grad_rewards_penalties(
                W_stack = W_stack,
                μ_bl_arr = μ_bl_arr,
                w_mir_arr = w_mir_a,
                w_msp_arr = w_msp_a,
                gamma = gamma,
                sample_size = sample_size,
                random_state = random_state,
                use_l1_penalty = True,
            )

        Σ = self.Σ

        LcT = self.Lc.T

        R1 = self.R1

        T1 = self.T1

        rf_ann = self.rf_ann

        rf_week = self.rf_week

        sqrt52 = self.sqrt52

        lb_arr = self.lb_arr
        
        ub_arr = self.ub_arr

        A_caps, caps_vec = self._caps_mats(
            max_industry_pct = self.default_max_industry_pct, 
            max_sector_pct = self.default_max_sector_pct
        )

        w_var = cp.Variable(self.n, nonneg = True)
        
        lin_param = cp.Parameter(self.n)

        pen_comb = (cp.sum(huber_vec(w_var - w_mir_a, huber_delta_l1)) / den_mir) + (cp.sum(huber_vec(w_var - w_msp_a, huber_delta_l1)) / den_msp)

        obj_expr = gamma[3] * scale_pen * pen_comb - cp.sum(cp.multiply(lin_param, w_var))

        cons = [cp.sum(w_var) == 1, w_var >= lb_arr, w_var <= ub_arr]

        if A_caps.shape[0] > 0:

            cons.append(A_caps @ w_var <= caps_vec)

        prob = cp.Problem(cp.Minimize(obj_expr), cons)

        def _from_initial(
            w0: np.ndarray
        ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        
            w = w0.copy()
        
            prev_obj = None
        
            for _ in range(max_iter):
        
                Σw = Σ @ w

                ret = float(self.er_arr @ w)
               
                vol = float(np.linalg.norm(LcT @ w))
                
                vol_eps = max(vol, 1e-12)
                
                grad_sh = (self.er_arr * vol - (ret - rf_ann) * (Σw / vol_eps)) / (vol_eps ** 2)

                u = rf_week - (R1 @ w)

                pos = (u > 0)

                if np.any(pos):

                    M = R1[pos]

                    dd_week = np.linalg.norm(u[pos]) / np.sqrt(T1)
                    
                    dd_week = max(dd_week, 1e-12)

                    dd_ann = dd_week * sqrt52

                    grad_dd_week = -(M.T @ u[pos]) / (T1 * dd_week)

                    grad_dd = grad_dd_week * sqrt52

                    grad_so = (self.er_arr * dd_ann - (ret - rf_ann) * grad_dd) / (dd_ann ** 2)
               
                else:
               
                    grad_so = np.zeros(self.n)

                Σb_w = Σb_cal @ w

                bl_ret = float(μ_bl_arr @ w)

                bl_vol = float(np.linalg.norm(LbT @ w))
                
                bl_eps = max(bl_vol, 1e-12)
                
                grad_bl = (μ_bl_arr * bl_vol - (bl_ret - rf_ann) * (Σb_w / bl_eps)) / (bl_eps ** 2)

                lin = gamma[0] * scale_s * grad_sh + gamma[1] * scale_so * grad_so + gamma[2] * scale_bl * grad_bl

                lin_param.value = lin
               
                if not self._solve(prob) or w_var.value is None:
               
                    raise RuntimeError("comb_port2 CCP: solvers failed")
               
                w_new = w_var.value

                obj_val = float(obj_expr.value)
               
                dw = float(np.linalg.norm(w_new - w))

                if prev_obj is not None and dw < tol and abs(obj_val - prev_obj) < tol:
               
                    w = w_new
               
                    break

                prev_obj = obj_val
               
                w = w_new

            Σw = Σ @ w

            vol = max(float(np.linalg.norm(LcT @ w)), 1e-12)

            dd_week = np.linalg.norm(np.maximum(rf_week - (R1 @ w), 0.0)) / np.sqrt(T1)

            dd_ann = max(dd_week * sqrt52, 1e-12)

            bl_vol = max(float(np.linalg.norm(LbT @ w)), 1e-12)

            sharpe = (float(self.er_arr @ w) - rf_ann) / vol

            sortino = (float(self.er_arr @ w) - rf_ann) / dd_ann

            bl_sh = (float(μ_bl_arr @ w) - rf_ann) / bl_vol

            pen_val = float(np.sum(np.abs(w - w_mir_a))) / den_mir + float(np.sum(np.abs(w - w_msp_a))) / den_msp

            score = (
                gamma[0]*scale_s*sharpe 
                + gamma[1]*scale_so*sortino 
                + gamma[2]*scale_bl*bl_sh
                - gamma[3] * scale_pen * pen_val
            )

            ret = float(self.er_arr @ w)

            vol = float(np.linalg.norm(LcT @ w))
            
            vol_eps = max(vol, 1e-12)

            g_sh = (self.er_arr * vol - (ret - rf_ann) * (Σw / vol_eps)) / (vol_eps ** 2)

            u = rf_week - (R1 @ w)
           
            pos = (u > 0)
           
            if np.any(pos):
           
                M = R1[pos]
           
                dd_week = np.linalg.norm(u[pos]) / np.sqrt(T1)
                
                dd_week = max(dd_week, 1e-12)
                
                dd_ann = dd_week * sqrt52
                
                grad_dd_week = -(M.T @ u[pos]) / (T1 * dd_week)
                
                grad_dd = grad_dd_week * sqrt52
                
                g_so = (self.er_arr * dd_ann - (ret - rf_ann) * grad_dd) / (dd_ann ** 2)
            
            else:
            
                g_so = np.zeros(self.n)

            Σb_w = Σb_cal @ w
            
            bl_ret = float(μ_bl_arr @ w)
            
            bl_vol = float(np.linalg.norm(LbT @ w))
            
            bl_eps = max(bl_vol, 1e-12)
            
            g_bl = (μ_bl_arr * bl_vol - (bl_ret - rf_ann) * (Σb_w / bl_eps)) / (bl_eps ** 2)

            g_pen = np.sign(w - w_mir_a) / den_mir + np.sign(w - w_msp_a) / den_msp

            diag = {
                "Sharpe_push": gamma[0] * scale_s * float(np.linalg.norm(g_sh)),
                "Sortino_push": gamma[1] * scale_so * float(np.linalg.norm(g_so)),
                "BL_push": gamma[2] * scale_bl * float(np.linalg.norm(g_bl)),
                "Penalty_push": gamma[3] * scale_pen * float(np.linalg.norm(g_pen))
            }
            
            return w, score, diag


        best_w = None
      
        best_s = -np.inf
      
        best_diag = {}
        
        for w0 in initials:
        
            try:
        
                w_cand, s_cand, diag = _from_initial(
                    w0 = np.asarray(w0, float)
                )
            
            except RuntimeError:
            
                continue
            
            if s_cand > best_s:
            
                best_s = s_cand
            
                best_w = w_cand
            
                best_diag = diag

        if best_w is None:
            
            raise RuntimeError("comb_port2: all initialisations failed")

        self._last_diag = best_diag
        
        return pd.Series(best_w, index = self.universe, name = "comb_port2")
