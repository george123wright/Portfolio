from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path
import zlib
import warnings
from typing import Sequence, Optional, Callable
from sklearn.covariance import LedoitWolf, GraphicalLassoCV, OAS
from scipy.optimize import nnls
from statsmodels.tsa.api import VAR  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data_processing.financial_forecast_data import FinancialForecastData, _parse_abbrev
import config

import logging

logger = logging.getLogger(__name__)

if not logger.handlers:

    logger.addHandler(logging.NullHandler())

COV_CACHE_REBUILD: bool = True


def _safe_log_returns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert simple returns to log returns with domain-safe clipping.

    Mathematical transformation
    ---------------------------
    For each observation r_t, compute:

        r_log,t = log(1 + r_t_clipped),

    where:

        r_t_clipped = max(r_t, -0.999999).

    Clipping enforces 1 + r_t_clipped > 0, which guarantees the logarithm is
    defined.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of simple returns.

    Returns
    -------
    pd.DataFrame
        Log-return DataFrame with the same shape/index/columns as input.

    Advantages
    ----------
    Log returns are additive over time under compounding and often display more
    stable variance behaviour for risk modelling. The clipping guard prevents
    numerical failures from malformed or extreme negative return inputs.
    """
   
    if df is None or df.empty:
   
        return df
   
    clipped = df.clip(lower = -0.999999)
   
    return np.log1p(clipped)


def _cache_key_from_inputs(
    *,
    tickers: list[str],
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    flags: dict[str, object],
) -> str:
    """
    Build a deterministic cache key from data recency, universe identity, and
    modelling switches.

    Key construction
    ----------------
    A canonical token list is assembled from:

    - number of tickers;
   
    - latest date in each return panel (daily/weekly/monthly);
   
    - CRC32 hash of the ordered ticker list;
   
    - sorted feature flags (for example log-return mode, OAS target enabled).

    Let `raw` be the pipe-joined token string. The cache key is:

        key = CRC32(raw) rendered as 8 lowercase hexadecimal characters.

    Parameters
    ----------
    tickers : list[str]
        Ordered ticker universe.
    daily_5y, weekly_5y, monthly_5y : pd.DataFrame
        Return panels used to infer recency markers.
    flags : dict[str, object]
        Boolean or categorical modelling toggles included in cache identity.

    Returns
    -------
    str
        Deterministic short hash key suitable for cache filenames.

    Advantages
    ----------
    The design creates fast cache invalidation when data timestamps, universe,
    or modelling configuration changes, while keeping file names compact and
    filesystem-safe.
    """
  
    parts = [
        "T" + str(len(tickers)),
        "TD" + str(pd.to_datetime(daily_5y.index.max()).date()) if not daily_5y.empty else "TDNone",
        "TW" + str(pd.to_datetime(weekly_5y.index.max()).date()) if not weekly_5y.empty else "TWNone",
        "TM" + str(pd.to_datetime(monthly_5y.index.max()).date()) if not monthly_5y.empty else "TMNone",
        "H" + str(zlib.crc32(",".join(tickers).encode("utf-8")) & 0xFFFFFFFF),
    ]
  
    for k in sorted(flags.keys()):
  
        parts.append(f"{k}={flags[k]}")
  
    raw = "|".join(parts)
  
    return f"{zlib.crc32(raw.encode('utf-8')) & 0xFFFFFFFF:08x}"


def _cache_paths(
    cache_dir: Path, 
    key: str
) -> tuple[Path, Path]:
    """
    Derive filesystem paths for cached base-covariance and target-covariance
    artefacts.

    Parameters
    ----------
    cache_dir : Path
        Root cache directory.
    key : str
        Cache key produced by `_cache_key_from_inputs`.

    Returns
    -------
    tuple[Path, Path]
        `(base_path, target_path)` where:

        - `base_path` stores heavy base artefacts (S, T_ref, Corr_ms, etc.);
  
        - `target_path` stores constructed covariance targets and metadata.

    Advantages
    ----------
    Splitting base and target artefacts allows partial cache reuse when only
    downstream target construction is invalidated.
    """
  
    base_path = cache_dir / f"cov_base_{key}.pkl"
  
    tgt_path = cache_dir / f"cov_targets_{key}.pkl"
  
    return base_path, tgt_path


def _load_cov_cache(
    cache_path: Path
) -> dict | None:
    """
    Load a cached covariance artefact dictionary from disk, returning `None`
    when unavailable or unreadable.

    Parameters
    ----------
    cache_path : Path
        Pickle path to load.

    Returns
    -------
    dict | None
        Deserialised cache object on success; otherwise `None`.

    Failure policy
    --------------
    Missing files, format incompatibilities, or deserialisation errors are
    treated as cache misses rather than hard failures. This keeps the modelling
    pipeline robust and deterministic under cache corruption.
    """

    if not cache_path.exists():

        return None

    try:

        return pd.read_pickle(cache_path)

    except Exception:

        return None


def _save_cov_cache(
    cache_path: Path, 
    obj: dict
) -> None:
    """
    Persist a covariance artefact dictionary to disk.

    Parameters
    ----------
    cache_path : Path
        Destination pickle path.
    obj : dict
        Serializable cache object.

    Side effects
    ------------
    Parent directories are created if required, then the object is written via
    pandas pickle serialisation.

    Advantages
    ----------
    Serialising expensive intermediate matrices substantially reduces repeated
    runtime for iterative model calibration and parameter tuning workflows.
    """
  
    cache_path.parent.mkdir(parents = True, exist_ok = True)
 
    pd.to_pickle(obj, cache_path)


def _oas_reference_cov(
    weekly_returns: pd.DataFrame
) -> np.ndarray:
    """
    Estimate an Oracle Approximating Shrinkage (OAS) covariance matrix from
    complete weekly return observations.

    Method
    ------
    Rows containing any missing or infinite values are removed. On the cleaned
    matrix X (T observations by n assets), scikit-learn OAS estimates:

        Sigma_OAS = (1 - delta) * S + delta * mu * I,

    where:

    - S is the sample covariance of X,
   
    - mu = trace(S) / n,
   
    - I is the identity matrix,
   
    - delta is an analytically estimated shrinkage intensity.

    Parameters
    ----------
    weekly_returns : pd.DataFrame
        Weekly return panel.

    Returns
    -------
    np.ndarray
        OAS covariance matrix in weekly units. If no complete observations are
        available, a zero matrix with matching dimension is returned.

    Advantages
    ----------
    OAS offers strong finite-sample conditioning in high-dimensional settings
    by shrinking noisy sample covariance eigenvalues towards a structured
    identity-scaled target.
    """
  
    W = weekly_returns.replace([np.inf, -np.inf], np.nan).dropna(how = "any")
  
    if W.empty:
  
        return np.zeros((weekly_returns.shape[1], weekly_returns.shape[1]))
  
    X = W.to_numpy(float)
  
    return OAS().fit(X).covariance_


def _block_corr_prior(
    corr_ref: pd.DataFrame,
    *,
    sector_map: pd.Series | None,
    industry_map: pd.Series | None,
    base_rho: float | None = None,
) -> np.ndarray:
    """
    Build a hierarchical block-correlation prior using sector and industry
    labels.

    Construction
    ------------
    Let R_ref be a cleaned reference correlation matrix over n assets.
    Define:

    - base_rho: baseline off-diagonal correlation (empirical mean if omitted),
  
    - rho_sector: mean correlation among same-sector pairs,
  
    - rho_ind: mean correlation among same-industry pairs.

    The prior matrix R_block is initialised as:

        R_block(i, j) = base_rho for i != j,
    
        R_block(i, i) = 1.

    Sector and industry structure are then overlaid:

        if sector_i == sector_j:   R_block(i, j) <- rho_sector
     
        if industry_i == industry_j: R_block(i, j) <- rho_ind

    Industry assignment is applied after sector assignment and therefore
    overrides it where both conditions hold. A final cleaning pass enforces
    valid correlation properties.

    Parameters
    ----------
    corr_ref : pd.DataFrame
        Reference correlation matrix with asset index labels.
    sector_map : pd.Series | None
        Asset-to-sector mapping.
    industry_map : pd.Series | None
        Asset-to-industry mapping.
    base_rho : float | None, optional
        Optional fixed baseline correlation.

    Returns
    -------
    np.ndarray
        Cleaned block-structured correlation prior.

    Advantages
    ----------
    Block priors provide interpretable structural shrinkage, reduce estimation
    noise in sparse histories, and preserve economically plausible correlation
    hierarchy across related assets.
    """
 
    R = _clean_corr_matrix(
        R = corr_ref.values
    )
    
    n = R.shape[0]
    
    if base_rho is None:
    
        off = R[np.triu_indices_from(R, k=1)]
    
        base_rho = float(np.nanmean(off)) if off.size else 0.0

    rho_sector = base_rho
    
    rho_ind = base_rho

    if sector_map is not None:
    
        sec = sector_map.reindex(corr_ref.index)
    
        same_sec = (sec.values[:, None] == sec.values[None, :])
    
        mask = np.triu(same_sec, k=1)
    
        if mask.any():
    
            rho_sector = float(np.nanmean(R[mask]))

    if industry_map is not None:
    
        ind = industry_map.reindex(corr_ref.index)
    
        same_ind = (ind.values[:, None] == ind.values[None, :])
    
        mask = np.triu(same_ind, k=1)
    
        if mask.any():
    
            rho_ind = float(np.nanmean(R[mask]))

    out = np.full((n, n), base_rho, dtype=float)
    
    np.fill_diagonal(out, 1.0)

    if sector_map is not None:
    
        sec = sector_map.reindex(corr_ref.index)
    
        same_sec = (sec.values[:, None] == sec.values[None, :])
    
        out[same_sec] = rho_sector
    
        np.fill_diagonal(out, 1.0)

    if industry_map is not None:
    
        ind = industry_map.reindex(corr_ref.index)
    
        same_ind = (ind.values[:, None] == ind.values[None, :])
    
        out[same_ind] = rho_ind

        np.fill_diagonal(out, 1.0)

    return _clean_corr_matrix(
        R = out
    )


def _ewma_cov_regime(
    weekly_returns: pd.DataFrame,
    *,
    index_returns_weekly: pd.DataFrame | None,
    lam_calm: float = 0.98,
    lam_stress: float = 0.94,
) -> np.ndarray:
    """
    Estimate a regime-aware EWMA covariance matrix by switching the decay
    parameter according to current benchmark volatility state.

    Regime rule
    -----------
    Let r_b,t be a benchmark weekly return series (S&P 500 if available).
    Define:

        vol_short,t = std(r_b,t-4 : r_b,t),
   
        vol_long,t  = std(r_b,t-51 : r_b,t).

    The stress threshold is:

        q = 90th percentile of vol_long (or mean if percentile unavailable).

    Decay choice:

        lambda = lam_stress if latest(vol_short) > q,
     
        lambda = lam_calm   otherwise.

    The selected lambda is passed to `_ewma_cov`, and the result is cleaned via
    `_clean_cov_matrix`.

    Parameters
    ----------
    weekly_returns : pd.DataFrame
        Weekly asset return panel.
    index_returns_weekly : pd.DataFrame | None
        Weekly benchmark/index returns used for regime detection.
    lam_calm : float, optional
        EWMA decay in calm regimes (slower decay).
    lam_stress : float, optional
        EWMA decay in stress regimes (faster decay).

    Returns
    -------
    np.ndarray
        Regime-conditioned EWMA covariance matrix in weekly units.

    Advantages
    ----------
    Faster decay in stressed conditions increases responsiveness to volatility
    clustering, while calmer regimes retain longer memory for stability.
    """
 
    if weekly_returns.empty:
 
        return np.zeros((weekly_returns.shape[1], weekly_returns.shape[1]))

    lam = lam_calm

    if index_returns_weekly is not None and not index_returns_weekly.empty:
 
        if "^GSPC" in index_returns_weekly.columns:
 
            ref = index_returns_weekly["^GSPC"].dropna()
 
        else:
 
            ref = index_returns_weekly.iloc[:, 0].dropna()
 
        if not ref.empty:
 
            vol_short = ref.rolling(5).std()
 
            vol_long = ref.rolling(52).std()
 
            thresh = vol_long.quantile(0.9) if vol_long.notna().sum() else vol_long.mean()
 
            if vol_short.iloc[-1] > thresh:
 
                lam = lam_stress

    return _clean_cov_matrix(
        M = _ewma_cov(
            returns_weekly = weekly_returns,
            lam = lam
        )
    )


def _parse_ratio_value(x) -> float:
    """
    Parse ratio-like values from heterogeneous financial table formats into a
    numeric decimal representation.

    Parsing heuristic
    -----------------
    After float coercion and finite check:

    - If abs(v) <= 1.5, keep v unchanged (already likely decimal ratio).
  
    - If 1.5 < abs(v) <= 100, interpret as percentage points and scale:

          v_decimal = v / 100.

    - If abs(v) > 100, keep v unchanged (likely true level multiple, not
      percentage notation).

    Parameters
    ----------
    x : Any
        Raw cell value from source statements.

    Returns
    -------
    float
        Parsed numeric value or `np.nan` when parsing fails.

    Advantages
    ----------
    The mixed-unit normalisation reduces silent scale errors when combining
    vendor tables that alternate between decimal and percentage conventions.
    """
  
    try:
  
        v = float(x)
  
    except Exception:
  
        return np.nan
  
    if not np.isfinite(v):
  
        return np.nan
  
    if abs(v) > 1.5:
  
        return v / 100.0 if abs(v) <= 100.0 else v
  
    return v


def _row_series(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    parser: Optional[Callable] = None,
) -> pd.Series | None:
    """
    Extract the first available named row from a wide statement table and
    convert it into a clean time-indexed numeric series.

    Processing pipeline
    -------------------
    1. Iterate candidate row labels in priority order.
  
    2. Convert row values with either:
  
       - a custom parser, or
  
       - numeric coercion (`to_numeric`).
  
    3. Attempt datetime parsing of column labels.
  
    4. If datetime parsing fails entirely, attempt year-based conversion to
       year-end timestamps.
  
    5. Drop entries with invalid timestamps and return the cleaned series.

    Parameters
    ----------
    df : pd.DataFrame
        Source statement table with metric rows and period columns.
    candidates : list[str]
        Candidate row names searched in order.
    parser : Optional[Callable], optional
        Value parser applied elementwise when provided.

    Returns
    -------
    pd.Series | None
        Cleaned time series for the first matching candidate, else `None`.

    Advantages
    ----------
    Prioritised row matching and robust date parsing stabilise ingestion across
    issuer-specific statement formats and inconsistent period labelling.
    """
  
    if df is None or df.empty:
  
        return None
  
    for row in candidates:
  
        if row in df.index:
  
            s = df.loc[row]
  
            if parser is not None:
  
                s = s.map(parser)
  
            else:
  
                s = pd.to_numeric(s, errors = "coerce")
  
            try:
  
                s.index = pd.to_datetime(s.index, errors = "coerce")
  
            except Exception:
  
                pass
  
            if s.index.isna().all():
  
                try:
  
                    years = pd.Index(s.index).astype(int)
  
                    s.index = pd.to_datetime(years, format = "%Y", errors = "coerce") + pd.offsets.YearEnd(0)
  
                except Exception:
  
                    pass
  
            s = s[~s.index.isna()]
  
            return s
  
    return None


def _fundamental_factor_returns(
    weekly_returns: pd.DataFrame,
    tickers: Sequence[str],
    fund_exposures_weekly: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """
    Construct weekly fundamental factor return series from either supplied
    exposures or historical statement-derived proxies.

    Two operating modes
    -------------------
    Mode A: externally supplied weekly exposures (`fund_exposures_weekly`)

    For each factor f with exposure matrix z_f,t,i and asset returns r_t,i:

        w_f,t,i = z_f,t,i / sum_j |z_f,t,j|,
        f_t     = sum_i w_f,t,i * r_t,i.

    Mode B: internal construction from historical financial statements

    For each ticker, historical (not forecast-path) annual/ratio series are
    mapped into slow-moving exposures:

    - PROFIT:
          mean(ROE, ROA)
  
    - GROWTH:
          mean(RevenueGrowth, EPSGrowth)
  
    - LEVER:
          (TotalDebt - CashEquivalents) / TotalAssets
  
    - CASH_STAB:
          - rolling_std(CFO, 4) / abs(rolling_mean(CFO, 4))
  
    - VALUE:
          - mean(log(EV/EBITDA), log(EV/FCF), log(PE), log(PB))

    These exposures are aligned to weekly dates by forward fill, then
    cross-sectionally standardised each week:

        z_t,i = (e_t,i - mean_i e_t,i) / std_i e_t,i,

    followed by the same L1-normalised portfolio return construction:

        w_t,i = z_t,i / sum_j |z_t,j|,
    
        f_t   = sum_i w_t,i * r_t,i.

    Parameters
    ----------
    weekly_returns : pd.DataFrame
        Weekly asset return panel (dates x tickers).
    tickers : Sequence[str]
        Asset universe ordering.
    fund_exposures_weekly : dict[str, pd.DataFrame] | None, optional
        Optional pre-built weekly exposure matrices by factor name.

    Returns
    -------
    pd.DataFrame | None
        Weekly factor return matrix (dates x factors), or `None` if no factor
        could be formed.

    Advantages
    ----------
    The procedure transforms low-frequency fundamental information into
    tradable return factors with explicit long-short construction, improving
    structural coverage beyond price-only factor sets.
    """
  
    if weekly_returns.empty:

        return None

    if fund_exposures_weekly is not None:
  
        R = weekly_returns.reindex(columns = tickers)
  
        out = {}
  
        for name, expo in fund_exposures_weekly.items():
  
            if expo is None or expo.empty:
  
                continue
  
            z = expo.reindex(index = weekly_returns.index, columns = tickers)
  
            z = z.replace([np.inf, -np.inf], np.nan)
  
            if not np.isfinite(z.to_numpy(dtype = float)).any():
  
                continue
  
            denom = z.abs().sum(axis = 1).replace(0.0, np.nan)
  
            w = z.div(denom, axis = 0).fillna(0.0)
  
            out[name] = (w * R).sum(axis = 1)
  
        if not out:
  
            return None
  
        return pd.DataFrame(out, index = weekly_returns.index)

    fdata = FinancialForecastData(
        tickers = list(tickers), 
        quiet = True
    )

    profit_series = {}
  
    growth_series = {}
  
    leverage_series = {}
  
    cash_stab_series = {}
  
    value_series = {}

    for t in tickers:
  
        if hasattr(config, "TICKER_EXEMPTIONS") and t in config.TICKER_EXEMPTIONS:
  
            continue

        ratios = fdata.ratios.get(t, pd.DataFrame())
  
        income_ann = fdata.income_ann.get(t, pd.DataFrame())
  
        bal = fdata.bal.get(t, pd.DataFrame())
  
        cash = fdata.cash.get(t, pd.DataFrame())

        roe = _row_series(
            df = ratios, 
            candidates = ["Return on Equity (ROE)"], 
            parser = _parse_ratio_value
        )
  
        roa = _row_series(
            df = ratios, 
            candidates = ["Return on Assets (ROA)"], 
            parser = _parse_ratio_value
        )

        if roe is not None or roa is not None:
          
            dfp = pd.concat([s for s in [roe, roa] if s is not None], axis = 1)
          
            profit_series[t] = dfp.mean(axis = 1, skipna = True)

        rev_g = _row_series(
            df = income_ann, 
            candidates = ["Revenue Growth"], 
            parser = fdata._parse_growth
        )
      
        eps_g = _row_series(
            df = income_ann, 
            candidates = ["EPS Growth"], 
            parser = fdata._parse_growth
        )
      
        if rev_g is not None or eps_g is not None:
      
            dfg = pd.concat([s for s in [rev_g, eps_g] if s is not None], axis = 1)
      
            growth_series[t] = dfg.mean(axis = 1, skipna = True)

        total_debt = _row_series(
            df = bal, 
            candidates = ["Total Debt"], 
            parser = _parse_abbrev
        )
      
        cash_eq = _row_series(
            df = bal, 
            candidates = ["Cash & Cash Equivalents", "Cash & Cash Equivalents"], 
            parser = _parse_abbrev
        )
      
        total_assets = _row_series(
            df = bal, 
            candidates = ["Total Assets"], 
            parser = _parse_abbrev
        )
      
        if total_debt is not None and total_assets is not None:
     
            cash_eq = cash_eq.reindex(total_debt.index).fillna(0.0) if cash_eq is not None else 0.0
     
            net_debt = total_debt - cash_eq
     
            leverage = net_debt / total_assets.replace(0.0, np.nan)
     
            leverage_series[t] = leverage.replace([np.inf, -np.inf], np.nan)

        cfo = _row_series(
            df = cash, 
            candidates = ["Operating Cash Flow"],
            parser = _parse_abbrev
        )
     
        if cfo is not None:
     
            rolling_mean = cfo.rolling(4, min_periods = 2).mean()
     
            rolling_std = cfo.rolling(4, min_periods = 2).std()
     
            cv = rolling_std / rolling_mean.abs().replace(0.0, np.nan)
     
            cash_stab_series[t] = (-cv).replace([np.inf, -np.inf], np.nan)

        ev_ebitda = _row_series(
            df = ratios, 
            candidates = ["EV/EBITDA"],
            parser = _parse_ratio_value
        )
     
        ev_fcf = _row_series(
            df = ratios,
            candidates = ["EV/FCF"],
            parser = _parse_ratio_value
        )
    
        pe = _row_series(
            df = ratios,
            candidates = ["PE Ratio"],
            parser = _parse_ratio_value
        )
     
        pb = _row_series(
            df = ratios,
            candidates = ["PB Ratio"],
            parser = _parse_ratio_value
        )
      
        if ev_ebitda is not None or ev_fcf is not None or pe is not None or pb is not None:
      
            dvals = pd.concat([s for s in [ev_ebitda, ev_fcf, pe, pb] if s is not None], axis = 1)
      
            dvals = dvals.replace([np.inf, -np.inf], np.nan)
      
            val = np.log(dvals.where(dvals > 0))
      
            value_series[t] = (-val).mean(axis = 1, skipna = True)

    
    def _to_weekly_df(
        series_map: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Convert a dictionary of irregular, low-frequency ticker series into a
        weekly panel aligned to the return index.

        Processing steps
        ----------------
        - Build a DataFrame from the ticker-keyed series dictionary.
      
        - Sort by timestamp.
      
        - Reindex onto `weekly_returns.index`.
      
        - Forward-fill so each week carries the latest known fundamental state.

        Parameters
        ----------
        series_map : dict[str, pd.Series]
            Mapping from ticker to historical exposure series.

        Returns
        -------
        pd.DataFrame
            Weekly-aligned exposure panel. If input is empty, an empty DataFrame
            with weekly index is returned.

        Advantages
        ----------
        Forward-filled weekly alignment is a practical state-space
        approximation for accounting variables that update discretely and are
        observed with reporting lags.
        """
   
        if not series_map:
   
            return pd.DataFrame(index = weekly_returns.index)
   
        df = pd.DataFrame(series_map)
   
        df = df.sort_index()
   
        df = df.reindex(weekly_returns.index).ffill()
   
        return df


    factors = {
        "PROFIT": _to_weekly_df(
            profit_series
        ),
        "GROWTH": _to_weekly_df(
            growth_series
        ),
        "LEVER": _to_weekly_df(
            leverage_series
        ),
        "CASH_STAB": _to_weekly_df(
            cash_stab_series
        ),
        "VALUE": _to_weekly_df(
            value_series
        ),
    }

    R = weekly_returns.reindex(columns = tickers)
 
    out = {}

    for name, expo in factors.items():
 
        if expo.empty:
 
            continue
 
        z = expo.sub(expo.mean(axis = 1), axis = 0)
 
        z = z.div(expo.std(axis = 1).replace(0.0, np.nan), axis = 0)
 
        z = z.replace([np.inf, -np.inf], np.nan)
 
        denom = z.abs().sum(axis = 1).replace(0.0, np.nan)
 
        w = z.div(denom, axis = 0).fillna(0.0)
 
        out[name] = (w * R).sum(axis = 1)

    if not out:
 
        return None

    return pd.DataFrame(out, index = weekly_returns.index)


def _nearest_psd_preserve_diag(
    C: np.ndarray, 
    eps: float = 1e-10
) -> np.ndarray:
    """
    Project a symmetric covariance-like matrix to the nearest positive semi-definite
    (PSD) matrix while preserving its diagonal (marginal variances).

    This function takes an approximate covariance matrix C ∈ ℝ^{n×n} which may be
    indefinite due to sampling noise, numerical error, or subsequent algebraic
    manipulations, and returns a matrix C_psd that:

        1. Is symmetric and positive semi-definite.
    
        2. Retains the original diagonal, that is
        diag(C_psd) = diag(C) (up to numerical clipping to `eps`).
    
        3. Is constructed by cleaning the implied correlation structure rather than
        arbitrarily altering marginal variances.

    The procedure is:

        1. **Symmetrisation**

            The input is first symmetrised:
            
                C_sym = 0.5 · (C + Cᵀ).

            This ensures that subsequent eigen-decompositions are well defined and that
            numerical asymmetries do not propagate.

        2. **Normalisation to correlation space**

            Let d = diag(C_sym). The diagonal is clipped to be at least `eps`, then
            square-rooted:

                - d_i = max(diag(C_sym)_i, eps)
            
                - s_i = sqrt(d_i)

            Let S = diag(s) and D⁻¹ = diag(1 / s_i). A correlation-like matrix R is
            constructed as:

                R = D⁻¹ · C_sym · D⁻¹.

            In exact arithmetic, if C_sym were a true covariance matrix, then R would be
            the corresponding correlation matrix with unit diagonal and entries
        
                ρ_{ij} = Cov(X_i, X_j) / (σ_i σ_j).

        3. **Eigenvalue cleaning in correlation space**

            R is symmetrised again and then eigen-decomposed:

                - R_sym = 0.5 · (R + Rᵀ)
    
                - R_sym = V · diag(w) · Vᵀ,

            where w = (w₁,…,w_n) are the eigenvalues and V the orthonormal eigenvectors.
    
            Negative eigenvalues arise when sampling noise or model construction renders
            R indefinite. These are clipped:

                - w̃_i = max(w_i, eps),

            yielding non-negative eigenvalues w̃. A PSD approximation of R is then:

                R_psd_raw = V · diag(w̃) · Vᵀ.

        4. **Re-normalisation to unit diagonal in correlation space**

            Due to numerical effects, the diagonal of R_psd_raw may drift away from one.
            
             Let:

                - d_R = diag(R_psd_raw) clipped to at least `eps`;
    
                - S_R = diag(1 / sqrt(d_R)).

            Then a correlation matrix with unit diagonal is obtained:

                R_psd = S_R · R_psd_raw · S_R.

            This preserves PSD-ness while enforcing exact unit variances in correlation
            space.

        5. **Mapping back to covariance space with original diagonal**

            Finally, the cleaned covariance matrix is reconstructed by reintroducing the
            original standard deviations s_i:

                C_psd_raw = S · R_psd · S,

            so that diag(C_psd_raw) ≈ d (the original clipped variances). A final
            symmetrisation

                C_psd = 0.5 · (C_psd_raw + C_psd_rawᵀ)

            ensures symmetry.

    The algorithm performs eigenvalue shrinkage in correlation
    space whilst preserving marginal variances supplied by the input matrix.
    This is particularly advantageous in equity covariance modelling because
    marginal volatilities are often relatively well estimated (or tied to
    external forecasts), whereas cross-sectional correlations can be unstable
    and noisy. By intervening only in the correlation structure, the procedure
    avoids artificially altering the per-asset risk levels implied by the
    diagonal of C.

    Parameters
    ----------
    C : np.ndarray
        Symmetric covariance-like matrix (n×n). Need not be PSD.
    eps : float, optional
        Minimum eigenvalue and variance floor used to avoid numerical problems.

    Returns
    -------
    np.ndarray
        Symmetric positive semi-definite matrix with diagonal matching the
        (clipped) diagonal of C.

    Notes
    -----
    This routine is useful when combining multiple covariance targets,
    performing shrinkage, or applying term-structure transformations, all of
    which may slightly violate positive semi-definiteness. Ensuring PSD is
    essential for portfolio optimisation and for interpreting the result as a
    valid covariance operator.
    """

    C = np.asarray(C, dtype = float)

    C = 0.5 * (C + C.T)

    d = np.clip(np.diag(C), eps, None)
    
    s = np.sqrt(d)
    
    S = np.diag(s)
    
    Dinv = np.diag(1.0 / s)

    R = Dinv @ C @ Dinv

    R = 0.5 * (R + R.T)

    w, V = np.linalg.eigh(R)

    w = np.maximum(w, eps)

    R_psd = (V * w) @ V.T

    d_R = np.clip(np.diag(R_psd), eps, None)

    Dinv_R = np.diag(1.0 / np.sqrt(d_R))

    R_psd = Dinv_R @ R_psd @ Dinv_R

    C_psd = S @ R_psd @ S

    return 0.5 * (C_psd + C_psd.T)


def _clean_cov_matrix(
    M: np.ndarray,
    min_var: float = 1e-10
) -> np.ndarray:
    """
    Normalise a covariance-like matrix so that it is finite, symmetric, and has
    strictly positive diagonal entries.

    This function is intended as a robust pre- and post-processing step when
    handling empirical covariance estimates, which may contain NaNs, infinities,
    or very small/negative variances due to numerical error.

    The operations are:

        1. Replace non-finite entries (NaN, +∞, −∞) with zero.
    
        2. Symmetrise the matrix:
    
            M_sym = 0.5 · (M + Mᵀ).
    
        3. Extract the diagonal d = diag(M_sym), enforce finiteness and a minimum
        variance `min_var`, and write back:
    
            d_i ← max(min_var, d_i) for each i.

    The resulting matrix is:

        - symmetric by construction;
    
        - free of non-finite entries;
    
        - guaranteed to have strictly positive variances on the diagonal.

    Parameters
    ----------
    M : np.ndarray
        Input covariance-like matrix.
    min_var : float, optional
        Minimum allowable variance for each diagonal entry.

    Returns
    -------
    np.ndarray
        Cleaned covariance-like matrix with finite entries and positive diagonal.

    Notes
    -----
    This helper does not enforce positive semi-definiteness; it focuses on
    basic numerical hygiene. PSD enforcement is delegated to
    `_nearest_psd_preserve_diag` when required.
    """

    M = np.asarray(M, dtype = float)
   
    M = np.where(np.isfinite(M), M, 0.0)
   
    M = 0.5 * (M + M.T)

    d = np.diag(M).copy()
   
    d = np.where(np.isfinite(d), d, min_var)
   
    d = np.clip(d, min_var, None)
   
    np.fill_diagonal(M, d)

    return M


def _clean_corr_matrix(
    R: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """
    Clean a correlation-like matrix to ensure finiteness, symmetry, and unit
    diagonal.

    Given an approximate correlation matrix R, this function:

        1. Replaces non-finite entries with zero.
    
        2. Symmetrises: R_sym = 0.5 · (R + Rᵀ).
    
        3. Sets the diagonal exactly to one.

    Parameters
    ----------
    R : pd.DataFrame or np.ndarray
        Approximate correlation matrix.

    Returns
    -------
    np.ndarray
        Symmetric, finite matrix with ones on the diagonal.

    Notes
    -----
    No explicit positive semi-definiteness adjustment is applied here; the
    result is intended as a numerically sound correlation input to further
    cleaning or shrinkage procedures.
    """
   
    R = np.asarray(R, dtype = float)
   
    R = np.where(np.isfinite(R), R, 0.0)
   
    R = 0.5 * (R + R.T)
   
    np.fill_diagonal(R, 1.0)
   
    return R


def _assert_finite(
    label: str,
    M: np.ndarray
):
    """
    Validate basic numerical properties of a covariance matrix.

    The function checks three conditions:
    
        1. All entries are finite (no NaN, +∞, or −∞).
    
        2. Symmetry: M ≈ Mᵀ within an absolute tolerance of 1e−12.
    
        3. All diagonal elements (variances) are strictly positive.

    If any of these checks fail, a `ValueError` is raised with the provided
    label.

    Parameters
    ----------
    label : str
        Descriptive label used in error messages.
    M : np.ndarray
        Covariance matrix to validate.

    Raises
    ------
    ValueError
        If the matrix contains non-finite entries, is not symmetric, or has
        non-positive diagonal entries.

    Notes
    -----
    This validation step is designed to catch numerical or logical errors early
    in the covariance construction pipeline before optimisation or portfolio
    construction.
    """
    
    if not np.all(np.isfinite(M)):
    
        raise ValueError(f"{label} contains non-finite entries.")

    if not np.allclose(M, M.T, atol=1e-10):
    
        raise ValueError(f"{label} is not symmetric.")

    d = np.diag(M)

    if np.any(d <= 0):

        raise ValueError(f"{label} has non-positive variances.")


def _winsorize_df(
    df: pd.DataFrame, 
    z: float = 8.0
) -> pd.DataFrame:
    """
    Apply robust column-wise winsorisation to a DataFrame using the median and
    median absolute deviation (MAD).

    For each column x of the input DataFrame, the function:

        1.  Computes the median m and MAD:

                - m = median(x_i),
    
                - MAD = median(|x_i − m|),

            with a small numerical floor added to MAD to avoid division by zero.

        2.  Defines lower and upper clipping thresholds using a robust z-score
            scaling:

                - lo = m − 1.4826 · MAD · z,
    
                - hi = m + 1.4826 · MAD · z,

            where 1.4826 is the constant that makes MAD consistent with the standard
            deviation under a Gaussian assumption, and z is the desired robust
            cut-off (default 8).

        3.  Clips each observation x_i to the interval [lo, hi].

            The resulting dataset is robust to extreme outliers, which can otherwise
            distort covariance and regression estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (e.g. returns, factors) with variables in columns.
    z : float, optional
        Robust z-score limit in MAD units.

    Returns
    -------
    pd.DataFrame
        Winsorised DataFrame with outliers compressed towards the central
        distribution.

    Notes
    -----
    This approach is preferable to naive trimming when the objective is to
    retain all observations but dampen the influence of extreme values, which
    can be particularly important for covariance estimation and factor-model
    regressions.
    """

    X = df.to_numpy(dtype = float)

    med = np.nanmedian(X, axis = 0)

    mad = np.nanmedian(np.abs(X - med), axis = 0) + 1e-10

    lo = med - 1.4826 * mad * z
    
    hi = med + 1.4826 * mad * z
    
    X = np.clip(X, lo, hi)

    return pd.DataFrame(X, index = df.index, columns = df.columns)


def _ewma_cov(
    returns_weekly: pd.DataFrame,
    lam: float = 0.97,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute an Exponentially Weighted Moving Average (EWMA) covariance matrix of
    weekly returns in a fully vectorised single pass.

    Let X ∈ ℝ^{T×n} be a matrix of weekly asset returns with T observations and n
    assets. Denote the time index t = 1,…,T, where t = T corresponds to the
    most recent observation. Define μ as the sample mean vector:

        μ = (1 / T) · ∑_{t=1}^T x_t,

    where x_t is the t-th row of X (a 1×n row vector).

    The centred returns are:

        x̃_t = x_t − μ.

    For a decay parameter λ ∈ (0,1), the EWMA covariance is defined as:

        S = (1 / W) · ∑_{t=1}^T w_t · x̃_tᵀ x̃_t,

    where the weights are

        w_t = (1 − λ) · λ^{T − t}

    and the normalisation constant is

        W = ∑_{t=1}^T w_t.

    Thus, more recent observations (larger t) receive higher weight, with the
    effective memory length controlled by λ (larger λ implies slower decay).

    Implementation details:

        - The function computes a weight vector w ∈ ℝ^T with entries
        w_t = (1 − λ) · λ^{T − t}.
    
        - It then constructs a column vector W = w (T×1) and forms a weighted,
        centred matrix:

            Xc = X − μ,
    
            Xc = 0 where Xc is non-finite,

        and uses

            S = (Xc · sqrt(W))ᵀ · (Xc · sqrt(W)) / W_sum,

        where W_sum = ∑ w_t. This yields S ∈ ℝ^{n×n}.

        - A final symmetrisation enforces S = 0.5 · (S + Sᵀ).

    Parameters
    ----------
    returns_weekly : pd.DataFrame
        Weekly returns with dates in the index and assets in columns.
    lam : float, optional
        EWMA decay parameter λ. Values close to 1 place more weight on recent
        observations.
    eps : float, optional
        Minimum value for the normalisation constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Symmetric EWMA covariance matrix of weekly returns.

    Advantages
    ----------
    EWMA captures time-varying volatility in a parsimonious way, reacting more
    quickly to recent changes in market conditions than a simple rolling sample
    covariance. It is widely used in risk management (for example, in RiskMetrics)
    because it is easy to implement, computationally efficient (O(T · n²)), and
    produces smoother, more stable estimates than short-window sample covariances.
    """
    
    X = returns_weekly.to_numpy(dtype = float)

    mu = np.nanmean(X, axis = 0, keepdims = True)
    
    Xc = X - mu
    
    Xc = np.where(np.isfinite(Xc), Xc, 0.0)

    T = Xc.shape[0]

    w = (1.0 - lam) * lam ** np.arange(T - 1, -1, -1, dtype = float)

    w_sum = w.sum()

    if w_sum <= eps:

        return np.zeros((Xc.shape[1], Xc.shape[1]), float)

    W = w[:, None]

    S = (Xc * np.sqrt(W)).T @ (Xc * np.sqrt(W)) / w_sum

    S = 0.5 * (S + S.T)

    return S


def _build_base_covariances(
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    comb_std: pd.Series,
    common_idx: list[str],
    *,
    ewma_lambda: float = 0.97,
    periods_per_year: int = 52,
):
    """
    Construct base weekly covariance estimates and multi-scale correlation inputs
    from daily, weekly, and monthly return histories, together with EWMA and a
    Ledoit–Wolf reference estimator.

    This function prepares the core covariance objects used later in the
    shrinkage and term-structure pipeline. It:

        1. Aligns daily, weekly, and monthly return panels to a common asset set.
    
        2. Removes near-zero daily returns attributable to stale prices.
    
        3. Builds annualised covariance estimates over multiple horizons (1, 3, 5
        years) by combining daily, weekly, and monthly information.
    
        4. Aggregates these into a base weekly covariance S.
    
        5. Constructs a Ledoit–Wolf reference covariance T_ref from weekly data.
    
        6. Builds a multi-scale historical correlation estimate Corr_ms.
    
        7. Computes an EWMA weekly covariance S_EWMA.
    
        8. Returns cleaned weekly return data for further modelling.

    **Multi-horizon covariance construction**

        For each horizon window, the function defines an annualised covariance as a
        weighted combination of daily, weekly, and monthly sample covariances:

            - Let Σ_d be the sample covariance of daily returns in the window,
            annualised by a factor of 252.
    
            - Let Σ_w be the sample covariance of weekly returns in the window,
            annualised by a factor of 52.
    
            - Let Σ_m be the sample covariance of monthly returns in the window,
            annualised by a factor of 12.

        For each horizon h ∈ {1 year, 3 years, 5 years}, the annualised covariance is

            Σ_ann(h) = 0.2 · Σ_d(h) + 0.6 · Σ_w(h) + 0.2 · Σ_m(h).

        This is then converted to a weekly covariance by dividing by the number of
        periods per year (typically 52):

            Σ_weekly(h) = Σ_ann(h) / periods_per_year.

        This yields Σ_weekly(1), Σ_weekly(3), Σ_weekly(5), denoted S1, S3, S5.

        A base weekly covariance S_base is then defined as the convex combination:

            S_base = 0.5 · S5 + 0.3 · S3 + 0.2 · S1.

        After application of `_clean_cov_matrix`, the result S is the primary
        historical covariance estimate used for shrinkage and comparison.

    **Reference covariance via Ledoit–Wolf**

        The function computes a reference covariance T_ref using `_lw_reference_nan_safe`,
        which applies the Ledoit–Wolf shrinkage estimator to weekly returns with
        sufficient complete data. Ledoit–Wolf covariance estimation can be viewed
        as:

            T_ref = (1 − δ) · Σ_sample + δ · Σ_prior,

        where δ ∈ [0,1] is a data-driven shrinkage intensity and Σ_prior is a
        simple structured prior (for example, scaled identity), chosen to reduce
        estimation error in high dimensions.

    **Multi-scale correlation Corr_ms**

        Daily returns (cleaned for staleness) are used to compute multi-year
        correlations:

            - Corr5: full history daily correlation.
    
            - Corr3: last 3 years daily correlation.
    
            - Corr1: last 1 year daily correlation.

        A convex average is then formed:

            Corr_ms = 0.5 · Corr5 + 0.3 · Corr3 + 0.2 · Corr1,

        and passed through `_clean_corr_matrix` to enforce symmetry and unit
        diagonal.

        This correlation matrix captures cross-sectional co-movement mostly from
        daily-frequency data, but with some emphasis on more recent history.

    **EWMA covariance**

        A weekly EWMA covariance S_EWMA is constructed using `_ewma_cov`, with
        decay λ = ewma_lambda. This provides a more reactive estimate of recent
        volatility compared with the multi-horizon sample covariances.

    **Role of comb_std**

        The `comb_std` Series carries per-asset forecast volatilities (or forecast
        standard errors). These are aligned to the common index and returned
        unchanged here; they are used later in `_build_cov_targets` to construct a
        factor-like covariance target F_pred with forecast variances on the
        diagonal.

    Parameters
    ----------
    daily_5y : pd.DataFrame
        Daily returns over approximately 5 years.
    weekly_5y : pd.DataFrame
        Weekly returns over approximately 5 years.
    monthly_5y : pd.DataFrame
        Monthly returns over approximately 5 years.
    comb_std : pd.Series
        Forecast standard deviations per asset (annualised), reflecting forecast
        error or expected volatility.
    common_idx : list of str
        List of tickers/assets to retain.
    ewma_lambda : float, optional
        Decay parameter for EWMA covariance.
    periods_per_year : int, optional
        Number of weekly periods per year (typically 52).

    Returns
    -------
    tuple
        (
            idx,           # list[str], aligned tickers
            comb_std,      # pd.Series, aligned forecast vol/SE
            daily_ns,      # pd.DataFrame, cleaned daily returns
            weekly_5y,     # pd.DataFrame, aligned weekly returns
            monthly_5y,    # pd.DataFrame, aligned monthly returns
            S,             # np.ndarray, base weekly covariance
            T_ref,         # np.ndarray, reference weekly covariance (Ledoit–Wolf)
            Corr_ms,       # pd.DataFrame, multi-scale daily correlation
            S_EWMA,        # np.ndarray, weekly EWMA covariance
            S1, S3, S5,    # np.ndarray, 1y/3y/5y weekly covariances
            weekly_clean,  # pd.DataFrame, weekly returns without missing values
        )

    Advantages
    ----------
    Combining multiple sampling frequencies and horizons reduces the sensitivity
    of covariance estimates to arbitrary window choices. The mixture of daily,
    weekly, and monthly information balances responsiveness with robustness.
    The Ledoit–Wolf reference further mitigates overfitting in high-dimensional
    settings by shrinking towards a structured target, while Corr_ms and
    S_EWMA provide alternative views of cross-sectional correlation and
    recency-weighted volatility, respectively.
    """
    
    idx = list(common_idx)

    daily_5y = daily_5y.loc[:, idx]
    
    weekly_5y = weekly_5y.loc[:, idx]
    
    monthly_5y = monthly_5y.loc[:, idx]
    
    comb_std = comb_std.loc[idx]

    daily_ns = _clean_daily_stale(
        daily = daily_5y
    )


    def _cov_for_horizon(
        d: pd.DataFrame, 
        w: pd.DataFrame,
        m: pd.DataFrame
    ) -> np.ndarray:
        """
        Build a horizon-specific weekly covariance estimate by blending
        annualised sample covariances from daily, weekly, and monthly returns.

        Equations
        ---------
        Let:

            C_d = cov(daily)   * 252,
            C_w = cov(weekly)  * 52,
            C_m = cov(monthly) * 12.

        Form an annualised blend:

            C_ann = 0.2 * C_d + 0.6 * C_w + 0.2 * C_m.

        Convert back to weekly units:

            C_weekly = C_ann / periods_per_year.

        Parameters
        ----------
        d, w, m : pd.DataFrame
            Daily, weekly, and monthly return panels for the same ticker set.

        Returns
        -------
        np.ndarray
            Weekly covariance matrix for the requested horizon subset.

        Advantages
        ----------
        Multi-frequency blending reduces dependence on any single sampling
        frequency and yields a more stable estimator under asynchronous market
        dynamics.
        """
    
        cov_d = d.cov(ddof = 0) * 252.0
    
        cov_w = w.cov(ddof = 0) * 52.0
    
        cov_m = m.cov(ddof = 0) * 12.0
    
        cov_ann = 0.2 * cov_d.values + 0.6 * cov_w.values + 0.2 * cov_m.values
    
        return cov_ann / float(periods_per_year)


    d3 = daily_ns[daily_ns.index >= daily_ns.index.max() - pd.DateOffset(years = 3)]
    
    w3 = weekly_5y[weekly_5y.index >= weekly_5y.index.max() - pd.DateOffset(years = 3)]
    
    m3 = monthly_5y[monthly_5y.index >= monthly_5y.index.max() - pd.DateOffset(years = 3)]

    d1 = daily_ns[daily_ns.index >= daily_ns.index.max() - pd.DateOffset(years = 1)]
    
    w1 = weekly_5y[weekly_5y.index >= weekly_5y.index.max() - pd.DateOffset(years = 1)]
    
    m1 = monthly_5y[monthly_5y.index >= monthly_5y.index.max() - pd.DateOffset(years = 1)]

    S5 = _cov_for_horizon(
        d = daily_ns,
        w = weekly_5y, 
        m = monthly_5y
    )
    
    S3 = _cov_for_horizon(
        d = d3,
        w = w3, 
        m = m3
    )
    
    S1 = _cov_for_horizon(
        d = d1, 
        w = w1, 
        m = m1
    )

    S_base = _clean_cov_matrix(
        M = 0.5 * S5 + 0.3 * S3 + 0.2 * S1
    )
    
    S = _clean_cov_matrix(
        M = S_base.copy()
    )

    T_ref = _lw_reference_nan_safe(
        weekly_returns = weekly_5y,
        S_weekly = S,
        min_complete_weeks = 26,
        policy = "complete_rows",
    )
    
    T_ref = _clean_cov_matrix(
        M = T_ref
    )

    _assert_finite(
        label = "S", 
        M = S
    )
    
    _assert_finite(
        label = "T_ref", 
        M = T_ref
    )

    Corr5 = daily_ns.corr()
    
    Corr3 = d3.corr()
    
    Corr1 = d1.corr()

    Corr_ms = (0.5 * Corr5 + 0.3 * Corr3 + 0.2 * Corr1).reindex(index = idx, columns = idx)
    
    Corr_ms = pd.DataFrame(
        _clean_corr_matrix(
            R = Corr_ms.values
        ),
        index = idx,
        columns = idx,
    )

    weekly_clean = weekly_5y.replace([np.inf, -np.inf], np.nan).dropna(how = "any")
    
    S_EWMA = _clean_cov_matrix(
        M = _ewma_cov(
            returns_weekly = weekly_clean,
            lam = ewma_lambda
        )
    )

    return (
        idx,
        comb_std,
        daily_ns,
        weekly_5y,
        monthly_5y,
        S,
        T_ref,
        Corr_ms,
        S_EWMA,
        S1,
        S3,
        S5,
        weekly_clean,
    )


def _constant_correlation_prior(
    corr: pd.DataFrame,
    std_vec: np.ndarray
) -> np.ndarray:
    """
    Construct a constant-correlation style prior covariance matrix given a
    correlation matrix and a vector of target standard deviations.

    Let R ∈ ℝ^{n×n} be a correlation matrix (or correlation-like matrix) and
    σ ∈ ℝ^n a vector of target standard deviations. This function first cleans R
    via `_clean_corr_matrix`, and then forms a covariance matrix Σ_prior as:

        Σ_prior = R_clean ⊙ (σ σᵀ),

    where ⊙ denotes elementwise multiplication and σ σᵀ is the outer product of
    σ with itself (so that Σ_prior_{ij} = R_clean_{ij} · σ_i · σ_j).

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix with assets in both rows and columns.
    std_vec : np.ndarray
        Vector of target standard deviations σ.

    Returns
    -------
    np.ndarray
        Covariance matrix consistent with the cleaned correlation and the given
        standard deviations.

    Notes
    -----
    Constant-correlation style priors are useful as shrinkage anchors: they
    encode a structured belief that all correlations are similar, while allowing
    per-asset volatility levels to differ. This can significantly reduce
    estimation error when sample correlations are noisy or unstable.
    """

    R = _clean_corr_matrix(
        R = corr.values
    )

    return R * np.outer(std_vec, std_vec)


def _lw_reference_nan_safe(
    weekly_returns: pd.DataFrame,
    S_weekly: np.ndarray,
    min_complete_weeks: int = 26,
    policy: str = "complete_rows"
) -> np.ndarray:
    """
    Compute a Ledoit–Wolf shrinkage covariance estimator on weekly returns in a
    NaN-safe manner, with a fallback to a pre-existing weekly covariance.

    This function:

        1. Optionally truncates the weekly return panel to a common start date
        across assets (policy "truncate_common_start").
    
        2. Drops any rows containing NaNs to obtain a complete-case matrix X.
    
        3. If there are at least `min_complete_weeks` complete observations,
        applies `sklearn.covariance.LedoitWolf` to obtain a shrinkage covariance
        estimate Σ_LW.
    
        4. Otherwise, returns the supplied matrix S_weekly as a fallback.

    Mathematically, the Ledoit–Wolf estimator can be written as:

        Σ_LW = (1 − δ) · Σ_sample + δ · Σ_prior,

    where Σ_sample is the sample covariance of X, Σ_prior is a simple
    structured matrix (such as a scaled identity), and δ is a data-driven
    shrinkage intensity chosen to minimise the mean squared error between Σ_LW
    and the true covariance. This reduces variance of the estimate in
    high-dimensional, low-sample contexts.

    Parameters
    ----------
    weekly_returns : pd.DataFrame
        Weekly returns per asset.
    S_weekly : np.ndarray
        Existing weekly covariance matrix used as fallback.
    min_complete_weeks : int, optional
        Minimum number of complete observations required to run Ledoit–Wolf.
    policy : {"complete_rows", "truncate_common_start"}, optional
        Policy for handling missing data. Currently only affects truncation of
        the start date before dropping rows with any NaNs.

    Returns
    -------
    np.ndarray
        Shrinkage covariance matrix if enough data exist, otherwise S_weekly.

    Advantages
    ----------
    Ledoit–Wolf shrinkage produces better conditioned, more stable covariance
    matrices than pure sample covariances, especially when the number of assets
    is large relative to the number of observations. The NaN-safe handling and
    fallback guarantee a valid matrix is always returned.
    """
   
    W = weekly_returns

    if policy == "truncate_common_start":
   
        starts = [W[c].first_valid_index() for c in W.columns]
   
        starts = [d for d in starts if d is not None]
   
        if len(starts) > 0:
   
            W = W.loc[max(starts):]

    W_cc = W.dropna(how = "any")
   
    X = W_cc.to_numpy(float)

    if X.shape[0] >= min_complete_weeks:
   
        return LedoitWolf().fit(X).covariance_

    return S_weekly


def _build_cov_targets(
    idx: list[str],
    comb_std: pd.Series,
    daily_ns: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    S: np.ndarray,
    T_ref: np.ndarray,
    Corr_ms: pd.DataFrame,
    S_EWMA: np.ndarray,
    S1: np.ndarray,
    S3: np.ndarray,
    S5: np.ndarray,
    weekly_clean: pd.DataFrame,
    *,
    ff_factors_weekly: pd.DataFrame | None = None,
    index_returns_weekly: pd.DataFrame | None = None,
    industry_returns_weekly: pd.DataFrame | None = None,
    sector_returns_weekly: pd.DataFrame | None = None,
    macro_factors_weekly: pd.DataFrame | None = None,
    fx_factors_weekly: pd.DataFrame | None = None,
    fund_exposures_weekly: dict[str, pd.DataFrame] | None = None,
    sector_map: pd.Series | None = None,
    industry_map: pd.Series | None = None,
    daily_open_5y: pd.DataFrame | None = None,
    daily_close_5y: pd.DataFrame | None = None,
    use_excess_ff: bool = True,
    use_oas: bool = True,
    use_block_prior: bool = True,
    use_regime_ewma: bool = True,
    use_glasso: bool = True,
    use_fund_factors: bool = True,
    horizon_weeks: float = 52.0,
    periods_per_year: int = 52,
    var_trend: str = "c",
) -> tuple[list[str], list[np.ndarray], dict]:
    """
    Construct a collection of structured covariance targets in weekly units for
    use in shrinkage, together with their names and auxiliary metadata.

    The function synthesises multiple covariance estimates, each capturing
    different aspects of return dynamics, and returns them as a list of matrices
    `mats` with corresponding names `names`. The targets include:

        - P: constant-correlation prior using Corr_ms and base weekly volatility.
    
        - S_EWMA: EWMA covariance based on weekly returns.
    
        - C_EWMA: EWMA correlation structure combined with base volatilities.
    
        - F: forecast-error covariance derived from forecast standard deviations
        `comb_std` and a forward-looking correlation matrix.
    
        - OVN / INTRA: overnight and intraday covariances constructed from open–close
        data (if available).
    
        - MACRO: macro factor-model covariance.
   
        - FF: Fama–French (or similar) factor-model covariance.
    
        - IDX: index-factor covariance.
    
        - IND: industry-factor covariance.
        
        - SEC: sector-factor covariance.
    
        - HIER: hierarchical (market+sector+industry) factor-model covariance.
        
        - STAT: statistical (PCA) covariance on residuals.
        
        - STAT_RMT: random-matrix-theory (RMT) cleaned statistical covariance.
    
        - REGIME: high-volatility regime covariance (EWMA on selected weeks).
    
        - LDA_REGIME: regime-weighted covariance using a discriminant model.
    
        - VaR: covariance implied by VaR-based correlations and historical volatilities.

    **Constant-correlation prior (P)**

        Using base weekly covariance S, the per-asset weekly standard deviations are:

            σ_wk = sqrt(diag(S)).

        The constant-correlation prior is:

            P_const = R_ms_clean ⊙ (σ_wk σ_wkᵀ),

        where R_ms_clean is Corr_ms cleaned via `_clean_corr_matrix`.

    **EWMA targets (S_EWMA, C_EWMA)**

        S_EWMA is provided as an input weekly EWMA covariance. From S_EWMA one can
        compute its implied volatilities s_ew and correlation matrix R_ew:

            s_ew = sqrt(diag(S_EWMA)),
    
            R_ew = S_EWMA / (s_ew s_ewᵀ),

        then define

            C_EWMA = R_ew ⊙ (σ_wk σ_wkᵀ),

        so that the correlation structure is driven by EWMA but the volatility
        levels are aligned with the base S.

    **Forecast-error covariance (F)**

        A forward-looking correlation matrix R_forecast is obtained via
        `_build_fpred_corr`, which blends historical correlations, multi-scale
        correlations, a constant-correlation anchor, and (optionally) a
        factor-based long-run correlation. Let `comb_std` be the annualised forecast
        standard deviations for each asset and `periods_per_year` the number of
        weekly periods per year. The implied weekly standard errors of forecast
        errors are:

            se_wk = comb_std / sqrt(periods_per_year).

        The forecast covariance target is then:

            F_pred = R_forecast ⊙ (se_wk se_wkᵀ).

        This target ensures the diagonal of F_pred matches the squared forecast
        standard errors and cross-asset co-movements are given by R_forecast.

    **Factor-model targets (MACRO, FF, IDX, IND, SEC, HIER)**

        Each factor-model covariance `Sigma_X` is computed via `factor_model_cov`:

            Sigma_X = B · Σ_F · Bᵀ + Ψ,

        where B is the matrix of factor loadings, Σ_F the factor covariance, and Ψ
        the diagonal (or slightly structured) idiosyncratic variance. For HIER, the
        factor set comprises a market index, sector returns, and industry returns.

    **Statistical covariances (STAT, STAT_RMT)**

        `Sigma_STAT` and `Sigma_STAT_RMT` are constructed on residuals (after
        optionally removing factor contributions) using:

            - PCA-based low-rank reconstruction blended with idiosyncratic noise, and
    
            - Random matrix theory (Marchenko–Pastur) eigenvalue cleaning,

        respectively. These capture common modes of variation beyond the explicit
        factor models and provide robust estimates when factor sets are incomplete.

    **Regime-based and VaR-based targets**

        - REGIME: an EWMA covariance estimated only over weeks flagged as high
        volatility according to short- vs long-horizon volatility of a reference
        index (e.g. the S&P 500).
    
        - LDA_REGIME: a mixture of covariances for low and high volatility regimes,
        weighted by regime probabilities estimated via Linear Discriminant
        Analysis on macro and factor variables.
    
        - VaR: a covariance constructed from VaR-implied correlations (see
        `var_implied_corr_matrix`) and historical annualised volatilities, then
        converted back to weekly units.

    Parameters
    ----------
    idx : list of str
        Asset identifiers (tickers) to retain.
    comb_std : pd.Series
        Annualised forecast standard deviations per asset (forecast error).
    daily_ns, weekly_5y, monthly_5y : pd.DataFrame
        Cleaned daily, weekly, and monthly returns.
    S : np.ndarray
        Base weekly covariance from `_build_base_covariances`.
    T_ref : np.ndarray
        Reference weekly covariance (Ledoit–Wolf).
    Corr_ms : pd.DataFrame
        Multi-scale daily correlation.
    S_EWMA : np.ndarray
        EWMA weekly covariance.
    S1, S3, S5 : np.ndarray
        Weekly covariances over 1-, 3-, and 5-year horizons.
    weekly_clean : pd.DataFrame
        Weekly returns without NaNs.
    ff_factors_weekly, index_returns_weekly, industry_returns_weekly,
    sector_returns_weekly, macro_factors_weekly : pd.DataFrame or None
        Various factor and grouping returns.
    daily_open_5y, daily_close_5y : pd.DataFrame or None
        Daily open and close prices for overnight/intraday decomposition.
    use_excess_ff : bool
        Whether to subtract risk-free rate from returns in FF factor models.
    horizon_weeks : float
        Forecast horizon (in weeks) for long-run correlation.
    periods_per_year : int
        Number of weekly periods per year.
    var_trend : str
        Trend specification passed to VAR in long-run factor-based correlation.

    Returns
    -------
    names : list of str
        Identifiers of the constructed targets (e.g. "P", "S_EWMA", "F", "VaR").
    mats : list of np.ndarray
        Corresponding covariance matrices (weekly units).
    aux_meta : dict
        Auxiliary metadata, currently including "S", "T_ref", "S1", "S3", "S5".

    Advantages
    ----------
    Using a diverse set of structured targets reduces model risk: different
    targets respond differently to market conditions (recent vs long-history,
    histogram-based vs factor-based vs statistical). The subsequent shrinkage
    optimisation can exploit this diversity to form a convex combination that
    matches reference properties (e.g. T_ref) while maintaining stability and
    positive definiteness.
    """

    mats: list[np.ndarray] = []

    names: list[str] = []

    std_wk = np.sqrt(np.clip(np.diag(S), 1e-10, None))

    P_const = _clean_cov_matrix(
        M = _constant_correlation_prior(
            corr = Corr_ms, 
            std_vec = std_wk
        )
    )

    S_ew = S_EWMA.copy()

    s_ew = np.sqrt(np.clip(np.diag(S_ew), 1e-10, None))

    R_ew = _clean_corr_matrix(
        R = S_ew / np.outer(s_ew, s_ew)
    )

    C_EWMA = _clean_cov_matrix(
        M = R_ew * np.outer(std_wk, std_wk)
    )

    R_forecast = _build_fpred_corr(
        idx = idx,
        T_ref = T_ref,
        Corr_ms = Corr_ms,
        weekly_5y = weekly_5y,
        ff_factors_weekly = ff_factors_weekly,
        index_returns_weekly = index_returns_weekly,
        macro_factors_weekly = macro_factors_weekly,
        industry_returns_weekly = industry_returns_weekly,
        sector_returns_weekly = sector_returns_weekly,
        horizon_weeks = horizon_weeks,
        trend = var_trend,
    )
    
    se_wk = comb_std.values / np.sqrt(periods_per_year)
    
    F_pred = _clean_cov_matrix(
        M = np.outer(se_wk, se_wk) * R_forecast
    )

    names += ["P", "S_EWMA", "C_EWMA", "F"]
    
    mats += [P_const, S_EWMA, C_EWMA, F_pred]

    if use_oas:
  
        try:
  
            Sigma_OAS = _clean_cov_matrix(
                M = _oas_reference_cov(weekly_5y)
            )
  
            mats.append(Sigma_OAS)
  
            names.append("OAS")
  
        except Exception:
  
            pass

    if use_block_prior and (sector_map is not None or industry_map is not None):
  
        try:
  
            R_block = _block_corr_prior(
                corr_ref = Corr_ms,
                sector_map = sector_map,
                industry_map = industry_map,
            )
  
            Sigma_BLOCK = _clean_cov_matrix(
                M = R_block * np.outer(std_wk, std_wk)
            )
  
            mats.append(Sigma_BLOCK)
  
            names.append("BLOCK")
  
        except Exception:
  
            pass

    if use_regime_ewma:
  
        try:
  
            Sigma_EWMA_REG = _ewma_cov_regime(
                weekly_returns = weekly_clean,
                index_returns_weekly = index_returns_weekly,
            )
  
            mats.append(Sigma_EWMA_REG)
  
            names.append("S_EWMA_REGIME")
  
        except Exception:
  
            pass

    if (daily_open_5y is not None) and (daily_close_5y is not None):

        Sigma_OVN, Sigma_INTRA = _overnight_intraday_cov(
            daily_open = daily_open_5y.loc[:, idx],
            daily_close = daily_close_5y.loc[:, idx],
            target_periods_per_year = periods_per_year,
        )
        
        if Sigma_OVN.any():
        
            mats.append(Sigma_OVN)
        
            names.append("OVN")
        
        if Sigma_INTRA.any():
        
            mats.append(Sigma_INTRA)
        
            names.append("INTRA")

    if macro_factors_weekly is not None:

        Sigma_MACRO = factor_model_cov(
            returns_weekly = weekly_5y,
            factors_weekly = macro_factors_weekly,
            use_excess = False,
            max_lag = "auto",
        )
        
        mats.append(Sigma_MACRO.values)
        
        names.append("MACRO")

    if fx_factors_weekly is not None:

        Sigma_FX = factor_model_cov(
            returns_weekly = weekly_5y,
            factors_weekly = fx_factors_weekly,
            use_excess = False,
            max_lag = "auto",
        )

        mats.append(Sigma_FX.values)

        names.append("FX")

    if ff_factors_weekly is not None:
        
        Sigma_FF = factor_model_cov(
            returns_weekly = weekly_5y,
            factors_weekly = ff_factors_weekly,
            use_excess = use_excess_ff,
            max_lag = "auto",
        )
        
        mats.append(Sigma_FF.values)
        
        names.append("FF")

    if index_returns_weekly is not None:
        
        Sigma_IDX = factor_model_cov(
            returns_weekly = weekly_5y,
            factors_weekly = index_returns_weekly,
            use_excess = False,
            max_lag = "auto",
        )
      
        mats.append(Sigma_IDX.values)
      
        names.append("IDX")

    if industry_returns_weekly is not None:
     
        Sigma_IND = factor_model_cov(
            returns_weekly = weekly_5y,
            factors_weekly = industry_returns_weekly,
            use_excess = False,
            max_lag = "auto",
        )
     
        mats.append(Sigma_IND.values)
     
        names.append("IND")

    if sector_returns_weekly is not None:
       
        Sigma_SEC = factor_model_cov(
            returns_weekly = weekly_5y,
            factors_weekly = sector_returns_weekly,
            use_excess = False,
            max_lag = "auto",
        )
       
        mats.append(Sigma_SEC.values)
       
        names.append("SEC")

    F_hier_for_ts = None

    if (
        (index_returns_weekly is not None)
        and (industry_returns_weekly is not None)
        and (sector_returns_weekly is not None)
    ):

        if "^GSPC" in index_returns_weekly.columns:

            F_mkt = index_returns_weekly[["^GSPC"]]

        else:

            F_mkt = index_returns_weekly.iloc[:, [0]]

        F_hier_for_ts = pd.concat(
            [F_mkt, sector_returns_weekly, industry_returns_weekly],
            axis = 1,
        )
        
        Sigma_HIER = factor_model_cov(
            returns_weekly = weekly_5y,
            factors_weekly = F_hier_for_ts,
            use_excess = False,
            max_lag = "auto",
        )
     
        mats.append(Sigma_HIER.values)
     
        names.append("HIER")

    if use_fund_factors:
   
        try:
   
            fund_factors = _fundamental_factor_returns(
                weekly_returns = weekly_5y.loc[:, idx],
                tickers = idx,
                fund_exposures_weekly = fund_exposures_weekly,
            )
   
            if fund_factors is not None and not fund_factors.empty:
   
                Sigma_FUND = factor_model_cov(
                    returns_weekly = weekly_5y,
                    factors_weekly = fund_factors,
                    use_excess = False,
                    max_lag = "auto",
                )
   
                mats.append(Sigma_FUND.values)
   
                names.append("FUND")
   
        except Exception:
   
            pass

    frames = []

    if ff_factors_weekly is not None:

        ff = ff_factors_weekly.copy()

        if "RF" in ff.columns:

            ff = ff.drop(columns = ["RF"])

        frames.append(ff)

    for fb in (
        index_returns_weekly,
        sector_returns_weekly,
        macro_factors_weekly,
        fx_factors_weekly,
    ):
      
        if fb is not None:
      
            frames.append(fb)

    if len(frames) == 0:
      
        W = weekly_5y.replace([np.inf, -np.inf], np.nan).dropna(how = "any")
      
        Xw = W.to_numpy(dtype=float)
      
        F_all = None
   
    else:
   
        idx_resid = weekly_5y.index
   
        for fr in frames:
   
            idx_resid = idx_resid.intersection(fr.index)

        if len(idx_resid) < 26:
            
            Xw = weekly_5y.to_numpy(dtype=float)
            
            F_all = None
        
        else:
        
            R_align = weekly_5y.loc[idx_resid].replace([np.inf, -np.inf], np.nan)
        
            big = {"RET": R_align}
        
            for k, fr in enumerate(frames):
        
                big[f"F{k}"] = fr.loc[idx_resid].replace([np.inf, -np.inf], np.nan)

            joint_stat = pd.concat(big.values(), axis = 1).dropna(how = "any")
          
            n_ret = R_align.shape[1]
          
            Xw = joint_stat.iloc[:, :n_ret].to_numpy(dtype = float)

            Fblocks = []
          
            c0 = n_ret
          
            for fr in frames:
          
                k = fr.shape[1]
          
                Fblocks.append(joint_stat.iloc[:, c0: c0 + k].to_numpy(dtype = float))
          
                c0 += k
                
            if len(Fblocks):
          
                F_all = np.concatenate(Fblocks, axis = 1)  
            
            else:
                
                F_all = None

    Sigma_STAT = _pca_stat_cov_on_residuals(
        X = Xw,
        F = F_all,
        K_max = 20,
    )
    
    Sigma_STAT_RMT = _pca_stat_cov_on_residuals_rmt(
        X = Xw,
        F = F_all,
    )

    mats.append(Sigma_STAT)

    names.append("STAT")

    mats.append(Sigma_STAT_RMT)

    names.append("STAT_RMT")

    if use_glasso and weekly_clean.shape[0] >= 20:
  
        X_glasso = weekly_clean.to_numpy(dtype = float)
  
        n_samples, n_features = X_glasso.shape
  
        if n_features >= 2 and np.isfinite(X_glasso).all():
  
            col_mean = X_glasso.mean(axis = 0)
  
            col_std = X_glasso.std(axis = 0, ddof = 0)
  
            col_std_safe = np.where(
                np.isfinite(col_std) & (col_std > 0.0),
                col_std,
                1.0,
            )
  
            X_std = (X_glasso - col_mean) / col_std_safe

            emp_cov = np.cov(X_std, rowvar = False, ddof = 0)
  
            off_diag = emp_cov.copy()
  
            np.fill_diagonal(off_diag, 0.0)
  
            alpha_max = np.max(np.abs(off_diag))
  
            if not np.isfinite(alpha_max) or alpha_max <= 0.0:
  
                alpha_max = 0.1
  
            alpha_max = max(alpha_max, 1e-3)
  
            alpha_min = max(alpha_max * 0.1, 1e-3)
  
            alphas_base = np.logspace(
                np.log10(alpha_max),
                np.log10(alpha_min),
                8,
            )

            cov_std = None
  
            for scale in (1.0, 2.0, 5.0):
  
                alphas = alphas_base * scale
  
                try:
  
                    with warnings.catch_warnings(record = True) as caught:
  
                        warnings.simplefilter("always", RuntimeWarning)
  
                        model = GraphicalLassoCV(
                            alphas = alphas,
                            assume_centered = True,
                        )
  
                        model.fit(X_std)
  
                    if not any(
                        issubclass(w.category, RuntimeWarning)
                        for w in caught
                    ):
 
                        cov_std = model.covariance_
 
                        break
 
                    cov_std = model.covariance_
 
                except Exception:
 
                    cov_std = None

            if cov_std is not None:
 
                scale = np.where(np.isfinite(col_std), col_std, 0.0)
 
                cov_orig = cov_std * (scale[:, None] * scale[None, :])
 
                Sigma_GLASSO = _clean_cov_matrix(
                    M = cov_orig
                )
   
                mats.append(Sigma_GLASSO)
   
                names.append("GLASSO")

    regime_lda = None

    if index_returns_weekly is not None and "^GSPC" in index_returns_weekly.columns:

        spx = index_returns_weekly["^GSPC"].dropna()

        vol_short = spx.rolling(5).std()

        vol_long  = spx.rolling(52).std()

        if vol_long.notna().sum() > 0:
            
            thresh = vol_long.quantile(0.9)  
            
        else:
            
            thresh = vol_long.mean()

        high_vol_flag = vol_short > thresh

        hv_idx = high_vol_flag[high_vol_flag].index

        regime_lda = high_vol_flag.astype(int).reindex(weekly_5y.index)

        reg_ret = weekly_5y.loc[weekly_5y.index.intersection(hv_idx)]
        
        if reg_ret.shape[0] >= 26:
        
            Sigma_REGIME = _clean_cov_matrix(
                M = _ewma_cov(
                    returns_weekly = reg_ret.replace([np.inf, -np.inf], np.nan).dropna(how = "any"),
                    lam = 0.94
                )
            )
            
            mats.append(Sigma_REGIME)
            
            names.append("REGIME")

    if (
        (macro_factors_weekly is not None)
        and (ff_factors_weekly is not None)
        and (index_returns_weekly is not None)
        and (regime_lda is not None)
    ):
      
        Z_blocks: list[pd.DataFrame] = []
      
        Z_blocks.append(macro_factors_weekly)
      
        ff_no_rf = ff_factors_weekly.drop(columns = ["RF"], errors = "ignore")
       
        Z_blocks.append(ff_no_rf)
       
        Z_blocks.append(index_returns_weekly)

        Z_full = pd.concat(Z_blocks, axis = 1)
 
        lda_joint = pd.concat([Z_full, regime_lda.rename("regime")], axis = 1).dropna(how = "any")
      
        if lda_joint.shape[0] >= 30:
      
            Z_aligned = lda_joint.iloc[:, :-1]
      
            y = lda_joint["regime"].astype(int)

            if y.nunique() >= 2:
             
                lda = LinearDiscriminantAnalysis()
             
                lda.fit(Z_aligned.to_numpy(float), y.to_numpy(int))

                R_lda = weekly_5y.loc[lda_joint.index].replace([np.inf, -np.inf], np.nan)
             
                R_joint = pd.concat([R_lda, y], axis=1).dropna(how="any")

                if R_joint.shape[0] >= 20:
             
                    X_ret = R_joint.iloc[:, :-1].to_numpy(float)
             
                    y_ret = R_joint.iloc[:, -1].to_numpy(int)

                    R_low  = X_ret[y_ret == 0]
             
                    R_high = X_ret[y_ret == 1]

                    if R_low.shape[0] >= 5 and R_high.shape[0] >= 5:
             
                        Sigma_low  = np.cov(R_low,  rowvar = False, ddof = 0)
             
                        Sigma_high = np.cov(R_high, rowvar = False, ddof = 0)

                        Z_T = Z_full.iloc[[-1]].to_numpy(float)
                      
                        p_reg = lda.predict_proba(Z_T)[0]
                      
                        p_low = float(p_reg[0])
                        
                        p_high = float(p_reg[1])

                        Sigma_LDA = p_low * Sigma_low + p_high * Sigma_high
                      
                        Sigma_LDA = _clean_cov_matrix(
                            M = Sigma_LDA
                        )
                      
                        mats.append(Sigma_LDA)
                      
                        names.append("LDA_REGIME")

    R_var = var_implied_corr_matrix(
        returns = weekly_5y.loc[:, idx], 
        alpha = 0.99,
        w_pair = 0.5
    )

    R_var_clean = _clean_corr_matrix(
        R = R_var.values
    )

    weekly_hist = weekly_5y.loc[:, idx].replace([np.inf, -np.inf], np.nan).dropna(how = "any")

    vol_wk_hist = weekly_hist.std(ddof = 0).to_numpy()

    vols_ann_hist = vol_wk_hist * np.sqrt(periods_per_year)

    Sigma_VaR_ann = R_var_clean * np.outer(vols_ann_hist, vols_ann_hist)
    
    Sigma_VaR_wk  = Sigma_VaR_ann / float(periods_per_year)

    Sigma_VaR_wk = _clean_cov_matrix(
        M = Sigma_VaR_wk
    )
    
    mats.append(Sigma_VaR_wk)
    
    names.append("VaR")

    mats = [
        _clean_cov_matrix(
            M = M
        ) for M in mats
    ]

    for nm, M in zip(names, mats):
       
        _assert_finite(
            label = f"Target[{nm}]", 
            M = M
        )

    aux_meta = {
        "S": S,
        "T_ref": T_ref,
        "S1": S1,
        "S3": S3,
        "S5": S5,
    }
    
    return names, mats, aux_meta


def _build_lagged_factors(
    F: np.ndarray,
    max_lag: int
) -> np.ndarray:
    """
    Construct a stacked factor matrix with current and lagged values for
    distributed-lag regressions.

    Given a factor matrix F_raw ∈ ℝ^{T×K} with T time points and K factors, and
    a maximum lag L = max_lag, the function builds:

        F_reg = [F_t, F_{t−1}, …, F_{t−L}] for t = L,…,T−1,

    so that:

        - The output F_reg has shape ((T − L) × (K (L + 1))).
    
        - Each row contains the contemporaneous factor vector and up to L lags.

    This is suitable for regressions of the form:

        X_t = α + ∑_{ℓ=0}^L B_ℓ F_{t−ℓ} + ε_t,

    where X_t is a vector of asset returns at time t.

    If `max_lag <= 0` or there are insufficient observations (T ≤ max_lag), the
    function returns the original F_raw unchanged.

    Parameters
    ----------
    F : np.ndarray
        Raw factor matrix F_raw of shape (T, K).
    max_lag : int
        Maximum lag order L.

    Returns
    -------
    np.ndarray
        Stacked factor matrix F_reg suitable for distributed-lag regression.

    Notes
    -----
    Including lags in factor regressions allows the model to capture delayed
    responses of asset returns to factor shocks and to absorb some serial
    correlation in residuals, thereby improving the realism of factor-based
    covariance estimates.
    """
  
    F = np.asarray(F, dtype = float)
  
    T, K = F.shape
  
    if max_lag <= 0 or T <= max_lag:
  
        return F

    blocks = [F[max_lag:]]  
  
    for L in range(1, max_lag + 1):
  
        blocks.append(F[max_lag - L : T - L])
  
    return np.hstack(blocks)


def _select_optimal_lag_for_regression(
    F_raw: np.ndarray,
    X_full: np.ndarray,
    max_lag_bound: int = 3,
    ridge: float = 1e-4,
    criterion: str = "bic",
    min_extra_obs: int = 5,
    lambda_lag: float = 1.0, 
) -> int:
    """
    Select the optimal number of factor lags in a distributed-lag regression
    using an information criterion with an explicit lag penalty.

    The function considers lag orders L in {0,…, max_lag_bound} for regressions
    of the form:

        X_t = α + ∑_{ℓ=0}^L B_ℓ F_{t−ℓ} + ε_t,

    where:

        - F_raw ∈ ℝ^{T×K} is the raw factor matrix,
    
        - X_full ∈ ℝ^{T×N} is the matrix of asset returns,
    
        - α ∈ ℝ^{1×N} and B_ℓ ∈ ℝ^{K×N}.

    For each candidate lag L:

        1. A lagged factor matrix F_reg(L) is constructed using
        `_prebuild_all_lags` / `_build_lagged_factors`.
    
        2. If L > 0, the first L observations of X_full are discarded so that
        factors and returns remain aligned.
    
        3. An augmented regressor matrix is formed:

            F_aug = [1, F_reg(L)],

        with shape (T_L × (1 + K_L)) where T_L is the effective number of
        observations and K_L = K (L + 1) (for single-block factors).
    
        4. A ridge-regularised least squares solution is computed:

            B_full(L) = argmin_B ∥X − F_aug B∥_F² + ∑_{j=1}^{K_tot−1} λ · B_j²,

        where the first column (intercept) is not penalised and `ridge` is the
        penalty λ. The closed-form solution is:

            B_full(L) = (F_augᵀ F_aug + diag(ridge_diag))⁻¹ F_augᵀ X.

        5. The residuals E(L) = X − F_aug B_full(L) are computed and the residual
        sum of squares (RSS) = ∑_{t,i} E_{t,i}² is obtained.

        6. An information criterion is computed:

            - Number of parameters: k_params = N · K_tot.
        
            - Effective sample size: T_eff = N · T_L.
        
            - Residual variance estimate: σ² = RSS / T_eff.

    For AIC:
    
        IC_struct = 2·k_params + T_eff · log(σ²)

    For BIC:
    
        IC_struct = k_params · log(T_eff) + T_eff · log(σ²)

    with 
    
        IC(L) = IC_struct + λ_lag · L,
        
    where λ_lag is an explicit penalty on the lag order to discourage overfitting.

    The lag order L* with the minimum IC(L) is chosen, subject to a feasibility
    requirement T_L ≥ K_L + min_extra_obs. If no candidate satisfies this
    constraint or the optimisation encounters numerical issues, the function
    defaults to L* = 0.

    Parameters
    ----------
    F_raw : np.ndarray
        Factor matrix (T×K).
    X_full : np.ndarray
        Asset returns matrix (T×N).
    max_lag_bound : int, optional
        Maximum lag order considered.
    ridge : float, optional
        Ridge penalty applied to slope coefficients in the lag-selection regression.
    criterion : {"aic", "bic"}, optional
        Information criterion used to compare lag orders.
    min_extra_obs : int, optional
        Minimum number of extra observations beyond the number of parameters
        required to consider a given lag.
    lambda_lag : float, optional
        Additional linear penalty on lag order.

    Returns
    -------
    int
        Selected lag order L* in [0, max_lag_bound].

    Advantages
    ----------
    This approach stabilises lag selection by balancing goodness-of-fit (via
    RSS and σ²) against model complexity (through k_params and λ_lag), thereby
    reducing the risk of spurious lag inclusion which would otherwise inflate
    parameter uncertainty and degrade out-of-sample covariance forecasts.
    """

    F_raw = np.asarray(F_raw, dtype = float)
    
    X_full = np.asarray(X_full, dtype = float)

    T, K = F_raw.shape
    
    N = X_full.shape[1]

    best_lag = 0
    
    best_ic = np.inf

    max_lag_bound = int(max_lag_bound)
    
    lag_cache = _prebuild_all_lags(
        F_raw = F_raw, 
        max_lag = max_lag_bound
    )

    for L in range(0, max_lag_bound + 1):
        
        if L not in lag_cache:
        
            continue
        
        F_reg, _ = lag_cache[L]
        
        if L > 0:
        
            X = X_full[L:, :]
        
        else:
        
            X = X_full

        T_L, K_L = F_reg.shape

        if T_L <= K_L + min_extra_obs:

            continue

        ones = np.ones((T_L, 1), float)

        F = np.hstack([ones, F_reg])

        K_tot = F.shape[1]

        ridge_diag = np.concatenate(([0.0], np.full(K_tot - 1, ridge)))

        FtF = F.T @ F + np.diag(ridge_diag)

        FtX = F.T @ X

        try:
       
            B_full = np.linalg.solve(FtF, FtX)
       
        except np.linalg.LinAlgError:
       
            logger.warning("LinAlgError during lag selection at L=%d; skipping this lag.", L)
       
            continue

        E = X - F @ B_full
       
        rss = float((E * E).sum())
       
        if rss <= 0:
       
            continue

        T_eff = T_L * N

        k_params = K_tot * N

        sigma2 = rss / T_eff

        if criterion.lower() == "aic":

            ic_struct = 2.0 * k_params + T_eff * np.log(sigma2)

        else:  
            
            ic_struct = k_params * np.log(T_eff) + T_eff * np.log(sigma2)

        ic = ic_struct + lambda_lag * L

        if ic < best_ic:

            best_ic = ic

            best_lag = L

    if best_ic == np.inf:

        return 0

    return best_lag


def var_implied_corr_matrix(
    returns: pd.DataFrame,
    alpha: float = 0.99,
    w_pair: float = 0.5,
    clip: float = 0.99,
) -> pd.DataFrame:
    """
    Construct a VaR-implied correlation matrix by treating each pair of assets as
    a two-asset portfolio and using the closed-form relationship between VaR and
    correlation.

    Let X ∈ ℝ^{T×N} be a matrix of returns with T observations and N assets.
    Fix a confidence level α (for example, 0.99) and a two-asset portfolio with
    weights w₁ and w₂ on assets i and j respectively. Denote:

        - VaR_i = −q_{1−α}(X_i),
    
        - VaR_j = −q_{1−α}(X_j),
    
        - VaR_port = −q_{1−α}(w₁ X_i + w₂ X_j),

    where q_{1−α}(·) denotes the (1−α)-quantile of the empirical return
    distribution and all VaR values are expressed as positive losses (hence the
    leading minus sign).

    Under a linear VaR approximation consistent with elliptical returns, the
    portfolio VaR satisfies:

        VaR_port² ≈ w₁² VaR_i² + w₂² VaR_j² + 2 w₁ w₂ ρ_{ij}^{VaR} VaR_i VaR_j,

    where ρ_{ij}^{VaR} is the VaR-implied correlation between assets i and j.
    Rearranging yields:

    ρ   _{ij}^{VaR} = [VaR_port² − w₁² VaR_i² − w₂² VaR_j²] / [2 w₁ w₂ VaR_i VaR_j].

    This function:

        1. Cleans the return matrix by removing rows with any NaNs or infinities.
        
        2. Computes per-asset VaR_i at level α.
        
        3. For each pair (i,j), constructs a two-asset portfolio with weights
        (w_pair, 1 − w_pair) and computes VaR_port for that portfolio.
        
        4. Uses the above formula to compute ρ_{ij}^{VaR}, clipping the result to
        [−clip, clip] to mitigate extreme values.
        
        5. Sets the diagonal entries ρ_{ii} = 1.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with dates in the index and asset identifiers in columns.
    alpha : float, optional
        Confidence level for VaR (e.g. 0.99).
    w_pair : float, optional
        Portfolio weight w₁ on the first asset in each pair; the second asset
        receives weight w₂ = 1 − w_pair.
    clip : float, optional
        Maximum absolute value allowed for the implied correlation before
        clipping.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of VaR-implied correlations.

    Advantages
    ----------
    VaR-implied correlations focus on co-movements in the tails of the return
    distribution, rather than in the bulk as in standard Pearson correlation.
    This can provide additional insight into joint extreme losses and is
    particularly relevant for risk management, where dependence during stress
    periods is more important than average co-movement.
    """

    R = returns.replace([np.inf, -np.inf], np.nan).dropna(how = "any")

    cols = list(R.columns)

    X = R.to_numpy(dtype = float)

    T, N = X.shape

    if T == 0 or N == 0:

        return pd.DataFrame(np.eye(N), index = cols, columns = cols)

    VaR_i = -np.quantile(X, 1.0 - alpha, axis = 0)

    w1 = float(w_pair)
    
    w2 = 1.0 - w1

    P = w1 * X[:, :, None] + w2 * X[:, None, :]

    VaR_port = -np.quantile(P, 1.0 - alpha, axis = 0) 

    V1 = VaR_i[:, None]   
    
    V2 = VaR_i[None, :]   

    denom = 2.0 * w1 * w2 * V1 * V2  
    
    num = VaR_port ** 2 - (w1 * V1) ** 2 - (w2 * V2) ** 2

    rho = np.zeros_like(denom, dtype = float)
    
    mask = denom > 0.0
    
    rho[mask] = num[mask] / denom[mask]

    rho = np.clip(rho, -clip, clip)
    
    np.fill_diagonal(rho, 1.0)

    return pd.DataFrame(rho, index = cols, columns = cols)


def factor_model_cov(
    returns_weekly: pd.DataFrame,
    factors_weekly: pd.DataFrame,
    index_factors_weekly: pd.DataFrame | None = None,
    use_excess: bool = False,
    ridge: float = 1e-4,
    eps: float = 1e-10,
    max_lag: int | str| None = None, 
) -> pd.DataFrame:
    """
    Estimate a weekly factor-model covariance matrix of asset returns, optionally
    with lagged factors, ridge regularisation, and excess-return adjustment.

    The factor model is:

        X_t = α + B F̃_t + ε_t,

    where:

        - X_t ∈ ℝ^N is the vector of asset returns at week t,
        
        - F̃_t ∈ ℝ^K is the stacked factor vector (current and lagged factors),
        
        - α ∈ ℝ^N is an intercept vector,
        
        - B ∈ ℝ^{K×N} is the matrix of factor loadings,
        
        - ε_t ∈ ℝ^N is the idiosyncratic return at week t.

    The covariance matrix of X_t is decomposed as:

        Σ = Cov(X_t) ≈ B Σ_F Bᵀ + Ψ,

    where:

        - Σ_F = Cov(F̃_t) is the factor covariance matrix,
       
        - Ψ = diag(ψ₁,…,ψ_N) is a diagonal matrix of idiosyncratic variances,
        possibly adjusted by cross-sectional shrinkage.

    The estimation steps are:

        1. **Alignment and excess-return adjustment**

            All inputs (returns_weekly, factors_weekly, optional index_factors_weekly)
            are intersected on common dates. If `use_excess` is True and a risk-free
            factor "RF" is present, returns are converted to excess returns:

                X_t ← R_t − RF_t,

            and "RF" is removed from the factor set.

        2. **Robust cleaning**

            Infinite values are replaced by NaN, followed by winsorisation via
            `_winsorize_df` to dampen outliers. Only rows where both returns and
            factors are finite across all columns are retained.

        3. **Lag selection and construction (optional)**

            The raw factor matrix F_raw and returns X_full are converted to numpy
            arrays. If `max_lag` is None or "auto", the optimal lag L* is selected
            via `_select_optimal_lag_for_regression`. Otherwise, `max_lag` is
            enforced (with non-negative truncation).

            If L* > 0, a stacked factor matrix F_reg is formed with current and
            lagged factors, and the first L* rows of X_full are discarded to
            maintain alignment. Otherwise, F_reg = F_raw.

        4. **Ridge-regularised regression**

            Let F_reg ∈ ℝ^{T×K_f} be the factor matrix and X ∈ ℝ^{T×N} the aligned
            return matrix. An augmented regressor matrix is defined:

                F_aug = [1, F_reg],

            with shape (T × (1 + K_f)). A ridge penalty is applied to the slope
            coefficients (but not the intercept). Denoting B_full = [α; B], the
            estimator is:

                B_full = (F_augᵀ F_aug + diag(ridge_diag))⁻¹ F_augᵀ X,

            where ridge_diag = (0, λ, λ,…,λ) and λ = ridge.

            The intercept and loadings are then:

                α = B_full[0:1, :],
            
                B = B_full[1:, :].

        5. **Idiosyncratic variance estimation with cross-sectional shrinkage**

            Residuals:

                E = X − (1 · α + F_reg B),

            are used to compute raw idiosyncratic variances:

                ψ_raw_i = (1 / dof) ∑_{t=1}^T E_{t,i}²,

            where dof = max(T − K_tot, 1) and K_tot is the number of columns in F_aug.

            Let ψ̄ be the cross-sectional mean of ψ_raw. A shrinkage estimator is
            formed:

                ψ_i = max(α_i ψ_raw_i + (1 − α_i) ψ̄, eps),

            where α_i ∈ [α_low, α_high] is chosen as an increasing function of the
            idiosyncratic variance share:

                total_var_i = Var(X_i),
            
                idio_share_i = ψ_raw_i / (total_var_i + 1e−12),
            
                α_i = α_low + (α_high − α_low) · idio_share_i.

            This increases shrinkage for names where residual variance is a smaller
            proportion of total variance, stabilising Ψ across the cross-section.

        6. **Factor covariance estimation**

            The covariance of the factor vector F_reg is estimated using the
            Ledoit–Wolf shrinkage estimator (with fallback to a sample covariance):

                Σ_F = LedoitWolf(F_reg).covariance_,

            improving stability when the number of factors is large relative to the
            sample size.

        7. **Model covariance construction**

            The factor-model covariance is then:

                Σ = Bᵀ Σ_F B + diag(ψ),

            which is cleaned via `_clean_cov_matrix` to ensure finite entries and
            positive diagonal.

    Parameters
    ----------
    returns_weekly : pd.DataFrame
        Weekly asset returns.
    factors_weekly : pd.DataFrame
        Weekly factor returns (e.g. Fama–French factors).
    index_factors_weekly : pd.DataFrame or None
        Additional index factors, if any.
    use_excess : bool, optional
        Whether to subtract the risk-free factor "RF" from returns.
    ridge : float, optional
        Ridge penalty applied to slope coefficients.
    eps : float, optional
        Minimum idiosyncratic variance to avoid degenerate diagonal.
    max_lag : int, str, or None, optional
        If None or "auto", selects lag via `_select_optimal_lag_for_regression`.
        If integer, uses that lag.

    Returns
    -------
    pd.DataFrame
        Factor-model covariance matrix Σ in weekly units, indexed by asset names.

    Advantages
    ----------
    This procedure combines:

        - Ridge-regularised regression to mitigate multicollinearity and control
        overfitting in factor loadings.
        
        - Ledoit–Wolf shrinkage for the factor covariance to improve conditioning.
        
        - Cross-sectional shrinkage of idiosyncratic variances, reducing dispersion
        caused by sampling noise.
        
        - Optional lagged factors, allowing the model to capture delayed responses.

    The resulting covariance is typically better conditioned and more robust
    than a naive sample covariance, while remaining interpretable via the factor
    decomposition Σ = B Σ_F Bᵀ + Ψ.
    """
    
    parts = [returns_weekly, factors_weekly]
    
    if index_factors_weekly is not None:
    
        parts.append(index_factors_weekly)

    idx = parts[0].index
    
    for p in parts[1:]:
    
        idx = idx.intersection(p.index)

    R = returns_weekly.loc[idx].astype(float)
    
    F_all = factors_weekly.loc[idx].astype(float).copy()
    
    if index_factors_weekly is not None:
    
        F_all = pd.concat([F_all, index_factors_weekly.loc[idx].astype(float)], axis = 1)

    if use_excess and "RF" in F_all.columns:
    
        rf = F_all["RF"].copy()
    
        F_all = F_all.drop(columns = ["RF"])
    
        R = R.sub(rf.values, axis = 0)

    R = R.replace([np.inf, -np.inf], np.nan)
    
    F_all = F_all.replace([np.inf, -np.inf], np.nan)

    R = _winsorize_df(
        df = R
    )
    
    F_all = _winsorize_df(
        df = F_all
    )

    mask = R.notna().all(axis = 1) & F_all.notna().all(axis = 1)
  
    R = R.loc[mask]
  
    F_all = F_all.loc[mask]

    F_raw = F_all.to_numpy(dtype = float)
  
    X_full = R.to_numpy(dtype = float)

    if max_lag is None or (isinstance(max_lag, str) and max_lag.lower() == "auto"):
      
        best_lag = _select_optimal_lag_for_regression(
            F_raw = F_raw,
            X_full = X_full,
            max_lag_bound = 3,     
            ridge = ridge,
            criterion = "bic",
        )
        
    else:
        
        best_lag = max(0, int(max_lag))

    if best_lag > 0 and F_raw.shape[0] > best_lag:
        
        F_reg = _build_lagged_factors(
            F = F_raw, 
            max_lag = best_lag
        )
        
        X = X_full[best_lag:, :]
        
    else:
        
        F_reg = F_raw
        
        X = X_full

    T, K_f = F_reg.shape

    if T < K_f + 5:
        
        Sigma = np.cov(X_full, rowvar = False, ddof = 0)
        
        return pd.DataFrame(
            _clean_cov_matrix(
                M = Sigma
            ),
            index = R.columns,
            columns = R.columns,
        )

    ones = np.ones((T, 1), dtype = float)
    
    F = np.hstack([ones, F_reg]) 
    
    K = F.shape[1]

    ridge_diag = np.concatenate(([0.0], np.full(K - 1, ridge)))
    
    FtF = F.T @ F + np.diag(ridge_diag)
    
    FtX = F.T @ X

    B_full = np.linalg.solve(FtF, FtX)
    
    alpha = B_full[0:1, :]
    
    B = B_full[1:, :]

    E = X - (ones @ alpha + F_reg @ B)

    dof = max(T - K, 1)
    
    psi_raw = (E * E).sum(axis=0) / dof
    
    psi_bar = float(psi_raw.mean())
    
    total_var = np.var(X, axis = 0, ddof = 1)
    
    idio_share = np.clip(psi_raw / (total_var + 1e-10), 0.0, 1.0) 

    alpha_low, alpha_high = 0.3, 0.9   
    
    alpha_vec = alpha_low + (alpha_high - alpha_low) * idio_share 

    psi = np.maximum(alpha_vec * psi_raw + (1.0 - alpha_vec) * psi_bar, eps)

    try:
   
        Sigma_F = LedoitWolf().fit(F_reg).covariance_
   
    except Exception:
   
        logger.warning("LedoitWolf failed (factor_model_cov); falling back to sample factor cov.")
   
        Sigma_F = np.cov(F_reg, rowvar = False, ddof = 0)

    Sigma = B.T @ Sigma_F @ B + np.diag(psi)
   
    Sigma = _clean_cov_matrix(
        M = Sigma
    )

    logger.info("factor_model_cov: selected lag = %d", best_lag)

    return pd.DataFrame(Sigma, index = R.columns, columns = R.columns)


def _orthogonalise_factor_blocks(
    factor_frames: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """
    Apply time-series orthogonalisation to a sequence of factor blocks in order
    to reduce overlap between different factor families (e.g. FF, IDX, IND, SEC,
    MACRO).

    Given a list of factor DataFrames [F^(0), F^(1), …, F^(J−1)], all indexed by
    the same time series, the function iterates over blocks j = 0,…,J−1 and:

        - For j = 0, keeps F^(0) unchanged.
    
        - For j > 0, regresses each column of F^(j) on all columns of the stacked
        previous blocks [F^(0),…,F^(j−1)] and replaces the block with the
        residuals.

    Formally, let F_prev ∈ ℝ^{T×K_prev} be the concatenation of all previous
    factors and X_j ∈ ℝ^{T×K_j} the current block. An augmented regressor matrix
    is formed:

        Z = [1, F_prev],

    and the least-squares solution is:

        B = (Zᵀ Z + ε I)⁻¹ Zᵀ X_j,

    with a small ridge ε to stabilise inversion. The residualised factors are:

        X_j,resid = X_j − Z B.

    These residuals are used as the new block F^(j)_ortho. The process is
    repeated sequentially, stacking all orthogonalised blocks to update
    F_prev.

    If there are insufficient overlapping observations or numerical problems,
    the block is left unchanged.

    Parameters
    ----------
    factor_frames : list of pd.DataFrame
        List of factor blocks to orthogonalise, all sharing the same index.

    Returns
    -------
    list of pd.DataFrame
        List of orthogonalised factor blocks of the same shape as the input.

    Advantages
    ----------
    Orthogonalisation reduces redundancy across factor blocks, making the
    subsequent factor regressions more stable and the interpretation of factor
    contributions clearer. It avoids double-counting risk that might otherwise
    occur if, for example, an index factor and sector factors explain similar
    variation.
    """

    if len(factor_frames) <= 1:
    
        return factor_frames

    ortho_frames: list[pd.DataFrame] = []
    
    prev_matrix = None

    for j, fr in enumerate(factor_frames):
    
        fr = fr.copy().astype(float)
    
        if j == 0:
    
            ortho_frames.append(fr)
    
            prev_matrix = fr.to_numpy()
    
            continue

        X = fr.to_numpy(float)
    
        if prev_matrix is None:
    
            ortho_frames.append(fr)
    
            prev_matrix = fr.to_numpy()
    
            continue

        F_prev = prev_matrix

        mask = np.isfinite(F_prev).all(axis=1) & np.isfinite(X).all(axis=1)

        if mask.sum() < max(30, F_prev.shape[1] + 5):

            ortho_frames.append(fr)

            prev_matrix = np.column_stack([prev_matrix, X])

            continue

        Fp = F_prev[mask, :]

        Xj = X[mask, :]

        ones = np.ones((Fp.shape[0], 1), float)

        Z = np.hstack([ones, Fp])

        ZtZ = Z.T @ Z

        ZtZ += 1e-10 * np.eye(ZtZ.shape[0])  

        try:
            
            ZtZ_inv = np.linalg.inv(ZtZ)
        
        except np.linalg.LinAlgError:
        
            logger.warning("LinAlgError during orthogonalisation; using pinv.")
        
            ZtZ_inv = np.linalg.pinv(ZtZ)

        B = ZtZ_inv @ (Z.T @ Xj)   
        
        residuals = np.empty_like(X)
        
        residuals[:] = np.nan
        
        residuals[mask, :] = Xj - Z @ B
 
        fr_ortho = pd.DataFrame(residuals, index = fr.index, columns = fr.columns)
        
        ortho_frames.append(fr_ortho)
        
        prev_matrix = np.column_stack([prev_matrix, residuals])

    return ortho_frames


def _prebuild_all_lags(
    F_raw: np.ndarray, 
    max_lag: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Precompute lagged factor matrices for all lag orders from 0 to max_lag.

    For a raw factor matrix F_raw ∈ ℝ^{T×K}, the function builds a dictionary:

        L → (F_reg(L), None),

    where:

        - For L = 0, F_reg(0) = F_raw.
    
        - For L ≥ 1, F_reg(L) is constructed by `_build_lagged_factors` and
        contains stacked [F_t, F_{t−1}, …, F_{t−L}] rows for t ≥ L.

    If T ≤ L for some L, lagging beyond T is not possible and higher lags are
    not included.

    Parameters
    ----------
    F_raw : np.ndarray
        Factor matrix of shape (T, K).
    max_lag : int
        Maximum lag order.

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping from lag L to (F_reg(L), None). The second element is reserved
        for potential future extensions (for example, pre-shifted X matrices).

    Notes
    -----
    This helper avoids recomputing lagged factor blocks repeatedly during lag
    selection.
    """

    F_raw = np.asarray(F_raw, dtype = float)
    
    T, K = F_raw.shape
    
    lags: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    lags[0] = (F_raw, None) 

    for L in range(1, max_lag + 1):
        
        if T <= L:
        
            break
        
        F_reg = _build_lagged_factors(
            F = F_raw, 
            max_lag = L
        )
        
        lags[L] = (F_reg, None)

    return lags


def _pca_stat_cov_on_residuals(
    X: np.ndarray,
    F: np.ndarray | None,
    K_max: int = 20,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute a statistical covariance matrix on residuals using PCA-based
    low-rank approximation blended with idiosyncratic variances.

    The function operates on a matrix X ∈ ℝ^{T×N} of asset returns or residuals,
    optionally after removing the contribution of known factors F ∈ ℝ^{T×K}.

        1. **Factor-residualisation (optional)**

            If F is provided and non-empty, the function first regresses X on F:

                B = argmin_B ∥X − F B∥_F²,

            with the ordinary least-squares solution:

                B = (Fᵀ F)⁻¹ Fᵀ X.

            Residuals are then:

                E = X − F B.

            Otherwise, E = X.

        2. **Sample covariance of residuals**

            The sample covariance of E is:

                Σ_E = (1 / T) · Eᵀ E,

            up to the choice of degrees of freedom (ddof=0).

        3. **PCA decomposition**

            Σ_E is eigen-decomposed:

                Σ_E = U diag(λ) Uᵀ,

            where λ = (λ₁,…,λ_N) are eigenvalues and U the orthonormal eigenvectors.
            Eigenvalues are clipped from below at `eps`.

            The eigenvalues are sorted descending by magnitude, and a minimal number
            K of principal components is chosen such that the cumulative explained
            variance ratio reaches at least 80 %:

            Let λ_{(1)} ≥ λ_{(2)} ≥ … ≥ λ_{(N)} be sorted eigenvalues and

            cum_k = (∑_{i=1}^k λ_{(i)}) / (∑_{i=1}^N λ_{(i)}).

            Then K is the smallest k with cum_k ≥ 0.8, capped by K_max.

        4. **Low-rank reconstruction**

            A low-rank approximation Σ_stat_lowrank is formed using the top K
            components (using the appropriate subspace of U and λ):

                Σ_stat_lowrank = U_K diag(λ_K) U_Kᵀ.

        5. **Blending with idiosyncratic variance**

            The diagonal of Σ_E, denoted diag_E, encodes per-asset residual variances.
            To avoid over-emphasising low-rank structure and to preserve reasonable
            marginal variances, the function computes a blending coefficient β* by
            regressing off-diagonal entries of Σ_E on those of Σ_stat_lowrank:

            Let A be the vector of off-diagonal entries of Σ_stat_lowrank and B_off
            those of Σ_E. Then

                β* = (Aᵀ B_off) / (Aᵀ A + 1e−8),

            clipped to [0,1]. The final covariance estimator is:

                Σ_stat = β* Σ_stat_lowrank + (1 − β*) diag(diag_E),

            where diag(diag_E) is the diagonal matrix with diag_E on the diagonal.

            This preserves a large fraction of the common-variation structure while
            anchoring variances to the empirical residual levels.

        6. **Cleaning**

            `_clean_cov_matrix` is applied to ensure finite entries and positive
            diagonal.

    Parameters
    ----------
    X : np.ndarray
        Matrix of returns or residuals (T×N).
    F : np.ndarray or None
        Factor matrix for residualisation (T×K), or None to skip this step.
    K_max : int, optional
        Maximum number of principal components to retain.
    eps : float, optional
        Minimum eigenvalue floor.

    Returns
    -------
    np.ndarray
        Statistical covariance matrix Σ_stat.

    Advantages
    ----------
    PCA-based statistical covariance estimation captures dominant modes of
    co-movement that may not be represented in the explicit factor set, while
    the blending with diagonal residual variances avoids over-fitting low-rank
    structure. This is particularly helpful when constructing shrinkage targets
    intended to complement explicit factor-based models.
    """

    if F is not None and F.size > 0:
    
        B = np.linalg.lstsq(F, X, rcond = None)[0]
    
        E = X - F @ B
    
    else:
    
        E = X

    Sigma_E = np.cov(E, rowvar = False, ddof = 0)
    
    evals, evecs = np.linalg.eigh(Sigma_E)
    
    evals = np.clip(evals, eps, None)

    N = Sigma_E.shape[0]
    
    idx_sorted = np.argsort(evals)[::-1]
    
    evals_sorted = evals[idx_sorted]
    
    cum = np.cumsum(evals_sorted)
    
    frac = cum / cum[-1]
    
    K = int(np.searchsorted(frac, 0.8)) + 1  
    
    K = min(K, K_max, N)
    
    logger.info("PCA stat cov: K=%d of N=%d", K, N)

    U = evecs[:, -K:]
    
    Lam = np.diag(evals[-K:])

    Sigma_stat_lowrank = U @ Lam @ U.T
    
    diag_E = np.diag(Sigma_E).copy()

    mask = ~np.eye(N, dtype = bool)
    
    A = Sigma_stat_lowrank[mask]
    
    B_off = Sigma_E[mask]

    num = float(np.dot(A, B_off))
  
    den = float(np.dot(A, A)) + 1e-8
    
    if den > 0:
        
        beta_star = num / den
    
    else:
        
        beta_star = 0.0
        
    beta_star = float(np.clip(beta_star, 0.0, 1.0))

    print(f"beta_star for stat cov blending: {beta_star:.4f}")

    Sigma_stat = beta_star * Sigma_stat_lowrank + (1.0 - beta_star) * np.diag(diag_E)
    
    return _clean_cov_matrix(
        M = Sigma_stat
    )


def _pca_stat_cov_on_residuals_rmt(
    X: np.ndarray,
    F: np.ndarray | None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute a statistical covariance matrix on residuals using random matrix
    theory (RMT) eigenvalue cleaning based on the Marchenko–Pastur (MP) bulk.

    The function operates on residuals E (optionally after regression on known
    factors F) and applies RMT-based shrinkage to the eigenvalues of the sample
    covariance.

        1. **Factor-residualisation (optional)**

            As in `_pca_stat_cov_on_residuals`, regress X on F if provided:

                E = X − F (Fᵀ F)⁻¹ Fᵀ X,

            otherwise E = X.

        2. **Sample covariance**

            The sample covariance is:

                Σ_E = (1 / T) · Eᵀ E,

            with eigen-decomposition:

                Σ_E = U diag(λ) Uᵀ.

        3. **Marchenko–Pastur bulk and noise level**

            Let T be the number of observations and N the number of assets. The
            aspect ratio is q = N / T. If q > 1, it is inverted to max(q, 1/q) to
            conform to the MP setting where q ≤ 1.

            The average eigenvalue is used as an estimate of the noise level σ²:

                σ² = mean(λ_i).

            Under the MP law, the upper edge of the noise bulk is:

                λ_plus = σ² · (1 + sqrt(q))².

        4. **Eigenvalue cleaning**

            All eigenvalues below λ_plus are shrunk to σ²:

                λ_clean,i = { λ_i,  if λ_i > λ_plus
                            { σ²,   otherwise }.

            The cleaned covariance is reconstructed as:

                Σ_clean = U diag(λ_clean) Uᵀ.

        5. **Cleaning and return**

            The matrix Σ_clean is passed through `_clean_cov_matrix`.

    Parameters
    ----------
    X : np.ndarray
        Matrix of returns or residuals (T×N).
    F : np.ndarray or None
        Factor matrix for residualisation, or None to skip.
    eps : float, optional
        Minimum eigenvalue floor.

    Returns
    -------
    np.ndarray
        RMT-cleaned covariance matrix Σ_clean.

    Advantages
    ----------
    RMT cleaning provides a principled way to differentiate between eigenvalues
    explained by random noise and those associated with genuine structure. By
    shrinking noisy eigenvalues towards a common level σ² and preserving large
    eigenvalues, the estimator reduces overfitting without explicitly choosing
    a rank. This is particularly useful when N is comparable to, or larger
    than, T.
    """
   
    if F is not None and F.size > 0:
   
        B = np.linalg.lstsq(F, X, rcond = None)[0]
   
        E = X - F @ B
   
    else:
   
        E = X

    T, N = E.shape
    
    if T <= 1 or N <= 1:
    
        Sigma_E = np.cov(E, rowvar = False, ddof = 0)
    
        return _clean_cov_matrix(
            M = Sigma_E
        )

    Sigma_E = np.cov(E, rowvar = False, ddof = 0)
    
    evals, evecs = np.linalg.eigh(Sigma_E)
    
    evals = np.clip(evals, eps, None)

    q = N / float(T)

    if q > 1.0:

        q = 1.0 / q

    sigma2 = float(evals.mean())

    lambda_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

    evals_clean = np.where(evals > lambda_plus, evals, sigma2)

    Sigma_clean = (evecs * evals_clean) @ evecs.T

    logger.info(
        "STAT_RMT: q=%.4f, sigma2=%.4e, lambda_plus=%.4e",
        q, sigma2, lambda_plus
    )

    return _clean_cov_matrix(
        M = Sigma_clean
    )


def _project_boxed_simplex_leq(
    v: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    s: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Project a vector onto a boxed simplex defined by lower and upper bounds and
    a total sum constraint.

    Given:

        - a vector v ∈ ℝ^m,
    
        - element-wise bounds lb ≤ x ≤ ub,
    
        - and a total sum constraint ∑ x_i ≤ s,

    this function computes a vector x that approximately solves:

        minimise   ∥x − v∥₂²
    
        subject to lb_i ≤ x_i ≤ ub_i   for all i, ∑ x_i ≤ s.

    The algorithm first clips v into [lb, ub]; if the sum of the clipped vector
    is ≤ s (within tolerance), the clipped vector is returned. Otherwise, a
    scalar dual variable τ is found via bisection such that:

        x(τ) = clip(v − τ, lb, ub)

    satisfies 
        
        ∑ x(τ)_i ≈ s. 
    
    The bisection proceeds over an interval for τ constructed from the differences 
    v_i − ub_i and v_i − lb_i for finite bounds.

    Parameters
    ----------
    v : np.ndarray
        Original vector to be projected.
    lb : np.ndarray
        Lower bounds for each component.
    ub : np.ndarray
        Upper bounds for each component (may be infinite).
    s : float, optional
        Simplex radius, i.e. upper bound on the sum of components.
    tol : float, optional
        Tolerance for the sum constraint.
    max_iter : int, optional
        Maximum number of bisection iterations.

    Returns
    -------
    np.ndarray
        Projected vector x satisfying box and sum constraints (within tolerance).

    Notes
    -----
    This projection is used to enforce non-negativity, box constraints, and a
    sum constraint on shrinkage weights in `_solve_shrinkage_weights`. It
    ensures feasibility of weights even when optimisation returns values that
    slightly violate constraints due to numerical error.
    """
   
    v = np.asarray(v, dtype = float)
   
    lb = np.asarray(lb, dtype = float)
   
    ub = np.asarray(ub, dtype = float)

    if np.any(lb > ub):
   
        raise ValueError("Infeasible: some lower bound exceeds upper bound.")
   
    if lb.sum() - s > 1e-10:
   
        raise ValueError("Infeasible: sum of lower bounds exceeds simplex size s.")

    x = np.clip(v, lb, ub)

    sx = float(x.sum())

    if sx <= s + tol:

        return x

    finite_ub = np.isfinite(ub)

    finite_lb = np.isfinite(lb)
    
    v_ub_fin = v[finite_ub] - ub[finite_ub]

    if np.any(finite_ub):
        
        tau_lo = float(np.min(v_ub_fin))
        
    else:
        
        tau_lo = 0.0
        
    if np.any(finite_lb):

        tau_hi = float(np.max(v_ub_fin))  
    
    else:
        
        tau_hi = 0.0

    for _ in range(max_iter):
        
        tau = 0.5 * (tau_lo + tau_hi)
        
        x = np.clip(v - tau, lb, ub)
        
        diff = float(x.sum() - s)

        if abs(diff) <= tol:
        
            return x
        
        if diff > 0:
        
            tau_lo = tau
        
        else:
        
            tau_hi = tau

    return np.clip(v - tau, lb, ub)


def _make_bounds_for_targets(
    names: list[str], 
    **kw
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct lower and upper bounds for shrinkage target weights given a list
    of target names and keyword arguments specifying minimums, maximums, or
    fixed weights.

    Let the shrinkage weights be w ∈ ℝ^m, where each component corresponds to
    a target name in `names`. The function outputs vectors lb and ub such that:

        lb_j ≤ w_j ≤ ub_j   for j = 1,…,m,

    with the following logic:

        - For each target name nm, a parameter pair (mn_key, mx_key) is looked up
        in PARAM_MAP. These keys point to entries in `kw` specifying the minimum
        and maximum weight for nm (e.g. "P" → ("p_min", "p_max")).
    
        - If a fixed weight parameter "w_nm" is present in `kw`, w_j is forced to
        that value: lb_j = ub_j = fixed_value.
    
        - Otherwise, lb_j is set to max(0, mn) and ub_j to max(lb_j, ux), where
        mn and ux are the supplied minimum and maximum or defaulted.

    The function also verifies that the sum of lower bounds does not exceed 1
    (with a small tolerance), otherwise it raises a ValueError, since such
    bounds would be infeasible under the constraint ∑ w_j ≤ 1.

    Parameters
    ----------
    names : list of str
        Names of shrinkage targets (e.g. "P", "S_EWMA", "F").
    **kw :
        Keyword arguments specifying bounds and optionally fixed weights.
        Expected keys include p_min, p_max, s_ewma_min, s_ewma_max, etc., as
        well as w_P, w_S_EWMA, etc. for fixed weights.

    Returns
    -------
    lb : np.ndarray
        Lower bounds for each weight.
    ub : np.ndarray
        Upper bounds for each weight.

    Raises
    ------
    ValueError
        If the sum of minimum weights exceeds 1, making the simplex constraint
        infeasible.

    Notes
    -----
    These bounds allow explicit control over the contribution of each target in
    the convex combination used by `_solve_shrinkage_weights`. They encode
    prior views or risk limits on how much weight should be allocated to each
    covariance component.
    """

    PARAM_MAP = {
        "P": ("p_min", "p_max"),
        "S_EWMA": ("s_ewma_min", "s_ewma_max"),
        "C_EWMA": ("c_ewma_min", "c_ewma_max"),
        "F": ("fpred_min", "fpred_max"),
        "OVN": ("ovn_min", "ovn_max"),
        "INTRA": ("intra_min", "intra_max"),
        "MACRO": ("macro_min", "macro_max"),
        "FF": ("ff_min", "ff_max"),
        "IDX": ("idx_min", "idx_max"),
        "IND": ("ind_min", "ind_max"),
        "SEC": ("sec_min", "sec_max"),
        "HIER": ("hier_min", "hier_max"),
        "STAT": ("stat_min", "stat_max"),
        "STAT_RMT": ("stat_rmt_min", "stat_rmt_max"), 
        "REGIME": ("regime_min", "regime_max"),
        "LDA_REGIME": ("lda_min", "lda_max"),
        "VaR": ("VaR_min", "VaR_max"),
        "OAS": ("oas_min", "oas_max"),
        "BLOCK": ("block_min", "block_max"),
        "S_EWMA_REGIME": ("s_ewma_regime_min", "s_ewma_regime_max"),
        "FX": ("fx_min", "fx_max"),
        "FUND": ("fund_min", "fund_max"),
        "GLASSO": ("glasso_min", "glasso_max"),
    }

    n_len = len(names)
    
    lb = np.zeros(n_len, dtype = float)

    ub = np.full(n_len, np.inf, dtype = float)

    for j, nm in enumerate(names):
        
        mn_key, mx_key = PARAM_MAP.get(nm, (None, None))

        if mn_key:
        
            mn = float(kw.get(mn_key, 0.0))
        
        else:
        
            mn = 0.0

        if mx_key:
        
            mx_val = kw.get(mx_key, None)
        
        else:
        
            mx_val = None

        if mx_val is not None:
        
            ux = float(mx_val)
        
        else:
        
            logger.debug("No max bound provided for target %s; defaulting to 1.0", nm)
        
            ux = 1.0

        fixed_key = f"w_{nm}"
        
        if fixed_key in kw and kw[fixed_key] is not None:
        
            val = max(0.0, float(kw[fixed_key]))
        
            lb[j] = val
        
            ub[j] = val
        
        else:
        
            lb[j] = max(0.0, mn)
        
            ub[j] = max(lb[j], ux)

    if lb.sum() > 1.0 + 1e-10:
        
        raise ValueError(f"Infeasible bounds: sum of mins {lb.sum():.4f} exceeds 1.0")

    return lb, ub


def _clean_daily_stale(
    daily: pd.DataFrame, 
    tol: float = 1e-10
) -> pd.DataFrame:
    """
    Remove near-zero daily returns indicative of stale prices by replacing them
    with zero.

    In daily return data, very small absolute returns (below `tol`) can
    correspond to days with stale marks or pricing errors, which can distort
    volatility estimates. This function:

        1. Replaces infinities with NaN.
    
        2. Replaces entries with absolute value < tol by zero, leaving other entries
        unchanged.

    Parameters
    ----------
    daily : pd.DataFrame
        Daily return data.
    tol : float, optional
        Threshold below which absolute returns are considered stale.

    Returns
    -------
    pd.DataFrame
        Daily returns with near-zero values replaced by zero.

    Notes
    -----
    Removing near-zero noise stabilises daily correlation and volatility
    estimates, especially when certain assets are thinly traded or quoted
    infrequently.
    """

    X = daily.replace([np.inf, -np.inf], np.nan)

    return X.where(X.abs() >= tol, 0.0)


def _overnight_intraday_cov(
    daily_open: pd.DataFrame,
    daily_close: pd.DataFrame,
    target_periods_per_year: int = 52,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct separate overnight and intraday covariance matrices in weekly
    units from daily open and close prices.

    Let O_t and C_t denote the opening and closing prices on day t. The function
    defines:

        - Overnight log-returns:

            r_ov_t = log(O_t / C_{t−1}),

        capturing the price move from previous close to current open.

        - Intraday log-returns:

            r_in_t = log(C_t / O_t),

        capturing the move from open to close on the same day.

    After aligning open and close prices on a common index and set of columns:

        1. Compute r_ov and r_in time series.
    
        2. Drop rows with any NaNs.
    
        3. If there are fewer than 10 observations for either r_ov or r_in, return
        zero matrices to avoid unstable estimates.
    
        4. Compute daily annualised covariances:

            Σ_ov,ann = Cov(r_ov) · daily_per_year,
    
            Σ_in,ann = Cov(r_in) · daily_per_year,

        where daily_per_year ≈ 252.

        5. Convert to weekly covariances by dividing by the number of target
        periods per year:

            Σ_OVN,wk = Σ_ov,ann / target_periods_per_year,
    
            Σ_INTRA,wk = Σ_in,ann / target_periods_per_year.

        6. Clean the resulting matrices with `_clean_cov_matrix`.

    Parameters
    ----------
    daily_open : pd.DataFrame
        Daily open prices.
    daily_close : pd.DataFrame
        Daily close prices.
    target_periods_per_year : int, optional
        Number of weekly periods per year (e.g. 52).
    eps : float, optional
        Minimum variance floor for cleaning.

    Returns
    -------
    Sigma_OVN_wk : np.ndarray
        Weekly covariance of overnight returns.
    Sigma_INTRA_wk : np.ndarray
        Weekly covariance of intraday returns.

    Advantages
    ----------
    Separating overnight and intraday risk allows more granular modelling of
    risk sources: overnight returns may capture macro news and gap risk, while
    intraday returns reflect continuous trading dynamics. These can be used as
    distinct shrinkage targets in a multi-target covariance blend.
    """
   
    idx = daily_close.index.intersection(daily_open.index)
   
    cols = [c for c in daily_close.columns if c in daily_open.columns]

    O = daily_open.loc[idx, cols]
   
    C = daily_close.loc[idx, cols]

    r_ov = np.log(O / C.shift(1))
   
    r_in = np.log(C / O)

    r_ov = r_ov.replace([np.inf, -np.inf], np.nan).dropna(how = "any")
   
    r_in = r_in.replace([np.inf, -np.inf], np.nan).dropna(how = "any")

    if r_ov.shape[0] < 10 or r_in.shape[0] < 10:
   
        n = len(cols)
   
        return np.zeros((n, n)), np.zeros((n, n))

    daily_per_year = 252.0
    
    cov_ov_ann = r_ov.cov(ddof = 0) * daily_per_year
    
    cov_in_ann = r_in.cov(ddof = 0) * daily_per_year

    Sigma_OVN_wk = cov_ov_ann.values / float(target_periods_per_year)
   
    Sigma_INTRA_wk = cov_in_ann.values / float(target_periods_per_year)

    Sigma_OVN_wk = _clean_cov_matrix(
        M = Sigma_OVN_wk,
        min_var = eps
    )
   
    Sigma_INTRA_wk = _clean_cov_matrix(
        M = Sigma_INTRA_wk,
        min_var = eps
    )

    return Sigma_OVN_wk, Sigma_INTRA_wk


def _decompose_factor_idio(
    returns_weekly: pd.DataFrame,
    factor_frames: list[pd.DataFrame],
    *,
    ridge: float = 1e-4,
    eps: float = 1e-10,
    keep_idio_offdiag: bool = False,
    idio_offdiag_shrink: float = 0.0,
    max_lag_factors: int | str | None = None,
    diag_target: np.ndarray | None = None,
    r2_cap: float = 0.9,
    factor_share_bounds: tuple[float, float] = (0.2, 0.8),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose the weekly return covariance into factor and idiosyncratic
    components, with optional lagged factors, off-diagonal idiosyncratic
    structure, and diagonal re-scaling to match a target variance.

    The decomposition aims to represent:

        Cov(R_t) ≈ Σ_factor + Σ_idio,

    where:

        - Σ_factor = B Σ_F Bᵀ is the factor-driven covariance;
        
        - Σ_idio is the idiosyncratic covariance, either diagonal or with shrunk
        off-diagonals;
        
        - R_t is the vector of weekly returns.

    The steps are:

        1. **Alignment and preliminary fallback**

            Intersect the index across `returns_weekly` and each DataFrame in
            `factor_frames`. If there are no factor frames or fewer than 10
            observations, a simple decomposition is used:

                - C = Cov(R),
                
                - Σ_idio = diag(diag(C)),
                
                - Σ_factor = C − Σ_idio.

            Both are cleaned via `_clean_cov_matrix`.

        2. **Joint data set**

            Construct Fcat by concatenating all factor frames. Build a joint DataFrame
            [R | Fcat] and drop rows with any NaNs. Let:

                - X_full ∈ ℝ^{T×N} be the asset returns,
            
                - F_raw ∈ ℝ^{T×K} be the combined factor matrix.

        3. **Lag selection and factor construction**

            Determine the best number of lags L* via `_select_optimal_lag_for_regression`
            on (F_raw, X_full) with ridge regularisation. If `max_lag_factors` is
            "auto" or None, automatic selection is used; otherwise, the user-supplied
            lag is used.

            Construct F_reg and aligned X as in `factor_model_cov`.

        4. **Ridge-regularised factor regression**

            For T observations and K_f lagged factors, define:

                F_aug = [1, F_reg] (T×(1+K)).

            The regression solution is:

                B_full = (F_augᵀ F_aug + diag(ridge_diag))⁻¹ F_augᵀ X,

            with ridge applied to slopes but not the intercept. Decompose:

                α = B_full[0:1, :],  
                
                B = B_full[1:, :].

            Residuals are:

                E = X − (1·α + F_reg B).

        5. **Idiosyncratic variances and Σ_idio**

            Raw idiosyncratic variances are:

                ψ_raw_i = (1 / dof) ∑_{t} E_{t,i}²,

            with dof = max(T − K_tot, 1). A cross-sectional shrinkage towards the
            average ψ̄ is applied:

                ψ_i = max(α_ψ ψ_raw_i + (1 − α_ψ) ψ̄, eps),

            where α_ψ is a fixed shrinkage parameter (e.g. 0.75).

            If `keep_idio_offdiag` is False, the idiosyncratic covariance is:

            Σ_idio = diag(ψ).

            If `keep_idio_offdiag` is True, an empirical covariance Σ_E of E is
            computed:

                Σ_E = Cov(E),

            decomposed into diagonal D and off-diagonal Off = Σ_E − D. The idio
            covariance is then:

                Σ_idio = D + idio_offdiag_shrink · Off,

            with off-diagonals shrunk towards zero.

        6. **Factor covariance Σ_F and Σ_factor**

            The factor covariance is estimated via Ledoit–Wolf:

                Σ_F = LedoitWolf(F_reg).covariance_,
                (fallback to sample covariance if needed).

            The factor covariance is:

                Σ_factor = Bᵀ Σ_F B.

        7. **Cleaning**

            Both Σ_factor and Σ_idio are passed through `_clean_cov_matrix`.

        8. **Optional diagonal calibration to diag_target**

            If `diag_target` is provided, the goal is to ensure that:

                diag(Σ_factor + Σ_idio) ≈ diag_target,

            while controlling the factor share p_i ∈ [f_min, f_max] of total variance
            attributed to factors for each asset i, with r2_cap as an upper bound.

    Let:

        - Σ_total_reg = Cov(X),
        
        - var_total_reg_i = diag(Σ_total_reg)_i,
        
        - diag_factor_reg_i = diag(Σ_factor)_i,
        
        - diag_idio_reg_i = diag(Σ_idio)_i,
        
        - denom_i = diag_factor_reg_i + diag_idio_reg_i.

    The raw factor share is:

        R2_raw_i = diag_factor_reg_i / denom_i.

    A shrunk share R2_shrunk_i is formed as a blend of R2_raw_i and its
    cross-sectional mean, then clipped to [0, r2_cap]. The final factor share
    p_i is clipped to [f_min, f_max].

    Target factor and idiosyncratic variances are:

        diag_factor_target_i = p_i · diag_target_i,
    
        diag_idio_target_i = (1 − p_i) · diag_target_i.

    Scaling vectors are:

        s_f_i = sqrt(diag_factor_target_i / diag_factor_reg_i),
    
        s_i_i = sqrt(diag_idio_target_i / diag_idio_reg_i),

    and the covariances are rescaled:

        Σ_factor ← S_f Σ_factor S_f,
    
        Σ_idio  ← S_i Σ_idio  S_i,

    where S_f and S_i are diagonal matrices of s_f and s_i.

    After re-scaling, both matrices are cleaned again.

    Parameters
    ----------
    returns_weekly : pd.DataFrame
        Weekly asset returns.
    factor_frames : list of pd.DataFrame
        List of factor blocks.
    ridge : float, optional
        Ridge penalty in factor regressions.
    eps : float, optional
        Minimum variance floor.
    keep_idio_offdiag : bool, optional
        Whether to retain off-diagonal idiosyncratic covariances with shrinkage.
    idio_offdiag_shrink : float, optional
        Shrinkage multiplier applied to idiosyncratic off-diagonals.
    max_lag_factors : int, str, or None, optional
        Lag selection for factors (see `_select_optimal_lag_for_regression`).
    diag_target : np.ndarray or None, optional
        Optional target diagonal for Σ_factor + Σ_idio.
    r2_cap : float, optional
        Maximum factor share per asset.
    factor_share_bounds : tuple[float, float], optional
        Lower and upper bounds for factor share p_i.

    Returns
    -------
    Sigma_factor : np.ndarray
        Factor-driven covariance matrix.
    Sigma_idio : np.ndarray
        Idiosyncratic covariance matrix.

    Advantages
    ----------
    This decomposition explicitly separates systematic and idiosyncratic
    sources of risk, incorporates lagged factor dynamics, allows for a modest
    degree of residual correlation, and can be calibrated to match target total
    variances and factor shares. This flexibility is crucial for building
    realistic term-structure models and for imposing economically meaningful
    constraints on how much variance is attributed to factors.
    """

    idx = returns_weekly.index

    for fr in factor_frames:

        idx = idx.intersection(fr.index)

    R = returns_weekly.loc[idx].replace([np.inf, -np.inf], np.nan).dropna(how = "any")

    if len(factor_frames) == 0 or R.shape[0] < 10:
        
        C = np.cov(R.to_numpy(float), rowvar=False, ddof = 0)
        
        diagC = np.diag(np.diag(C))
        
        Sigma_factor = _clean_cov_matrix(
            M = C - diagC
        )
        
        Sigma_idio = _clean_cov_matrix(
            M = diagC
        )
        
        return Sigma_factor, Sigma_idio

    Fcat = pd.concat([fr.loc[idx] for fr in factor_frames], axis = 1)
    
    joint = pd.concat([R, Fcat], axis = 1).replace([np.inf, -np.inf], np.nan).dropna(how = "any")

    N = R.shape[1]
    
    X_full = joint.iloc[:, :N].to_numpy(float)
    
    F_raw = joint.iloc[:, N:].to_numpy(float)

    if max_lag_factors is None or (isinstance(max_lag_factors, str) and max_lag_factors.lower() == "auto"):

        best_lag = _select_optimal_lag_for_regression(
            F_raw = F_raw,
            X_full = X_full,
            max_lag_bound = 3,    
            ridge = ridge,
            criterion = "bic",
        )
  
    else:
  
        best_lag = max(0, int(max_lag_factors))

    if best_lag > 0 and F_raw.shape[0] > best_lag:
  
        F_reg = _build_lagged_factors(
            F = F_raw,
            max_lag = best_lag
        )
  
        X = X_full[best_lag:, :]
  
    else:
  
        F_reg = F_raw
  
        X = X_full

    T, K_f = F_reg.shape
    
    ones = np.ones((T, 1), float)
    
    F = np.hstack([ones, F_reg])
    
    K = F.shape[1]

    ridge_diag = np.concatenate(([0.0], np.full(K - 1, ridge)))
    
    FtF = F.T @ F + np.diag(ridge_diag)
    
    FtX = F.T @ X

    B_full = np.linalg.solve(FtF, FtX)
    
    alpha = B_full[0:1, :]
    
    B = B_full[1:, :]

    E = X - (ones @ alpha + F_reg @ B)

    dof = max(T - K, 1)
    
    psi_raw = (E * E).sum(axis=0) / dof
    
    psi_bar = float(psi_raw.mean())
    
    alpha_psi = 0.75
    
    psi = np.maximum(alpha_psi * psi_raw + (1.0 - alpha_psi) * psi_bar, eps)

    try:

        Sigma_F = LedoitWolf().fit(F_reg).covariance_

    except Exception:

        logger.warning("LedoitWolf failed (_decompose_factor_idio); falling back to sample factor cov.")

        Sigma_F = np.cov(F_reg, rowvar = False, ddof = 0)

    Sigma_factor = B.T @ Sigma_F @ B

    if keep_idio_offdiag:
        
        Sigma_E = np.cov(E, rowvar = False, ddof = 0)
        
        D = np.diag(np.diag(Sigma_E))
        
        Off = Sigma_E - D
        
        Sigma_idio = D + float(idio_offdiag_shrink) * Off
    
    else:
    
        Sigma_idio = np.diag(psi)

    Sigma_factor = _clean_cov_matrix(
        M = Sigma_factor
    )
    
    Sigma_idio = _clean_cov_matrix(
        M = Sigma_idio
    )

    if diag_target is not None:

        diag_target = np.asarray(diag_target, dtype = float)

        if diag_target.shape[0] != N:

            raise ValueError("diag_target has incompatible length in _decompose_factor_idio.")

        Sigma_total_reg = np.cov(X, rowvar = False, ddof = 0)
        
        diag_factor_reg = np.clip(np.diag(Sigma_factor), 0.0, None)
        
        diag_idio_reg = np.clip(np.diag(Sigma_idio), 0.0, None)
        
        denom = np.clip(diag_factor_reg + diag_idio_reg, eps, None)

        R2_raw = diag_factor_reg / denom
        
        R2_bar = float(R2_raw.mean())
        
        R2_shrunk = 0.5 * R2_raw + 0.5 * R2_bar
        
        R2_shrunk = np.clip(R2_shrunk, 0.0, r2_cap)

        f_min, f_max = factor_share_bounds
        
        p = np.clip(R2_shrunk, f_min, f_max)

        diag_factor_target = p * diag_target
        
        diag_idio_target = (1.0 - p) * diag_target

        diag_factor_raw = np.clip(np.diag(Sigma_factor), eps, None)
        
        s_f = np.sqrt(diag_factor_target / diag_factor_raw)
        
        S_f = np.diag(s_f)
        
        Sigma_factor = S_f @ Sigma_factor @ S_f

        diag_idio_raw = np.clip(np.diag(Sigma_idio), eps, None)
        
        s_i = np.sqrt(diag_idio_target / diag_idio_raw)
        
        S_i = np.diag(s_i)
        
        Sigma_idio = S_i @ Sigma_idio @ S_i

        Sigma_factor = _clean_cov_matrix(
            M = Sigma_factor
        )
        
        Sigma_idio = _clean_cov_matrix(
            M = Sigma_idio
        )

    logger.info("_decompose_factor_idio: selected lag = %d", best_lag)

    return Sigma_factor, Sigma_idio


def _compute_factor_based_longrun_asset_corr(
    weekly_returns: pd.DataFrame,
    ff_factors_weekly: pd.DataFrame | None,
    index_returns_weekly: pd.DataFrame | None,
    macro_factors_weekly: pd.DataFrame | None,
    industry_returns_weekly: pd.DataFrame | None,
    sector_returns_weekly: pd.DataFrame | None,
    asset_idx: list[str],
    horizon_weeks: int,
    max_factors: int = 15,
    eps: float = 1e-10,
    trend: str = "c",
) -> np.ndarray | None:
    """
    Compute a long-run asset correlation matrix by fitting a VAR model on a
    PCA-reduced factor set and mapping the long-run factor covariance back to
    assets.

    The methodology is:

        1. **Factor set construction**

            Concatenate available factor blocks (Fama–French-like, index, macro,
            industry, sector) after dropping any risk-free column. Align all factors
            and weekly returns on a common date index. If there are insufficient
            observations (less than max(horizon_weeks, 60)), return None.

        2. **Standardisation and PCA reduction**

            Let F_mat_raw ∈ ℝ^{T×K} be the stacked factor matrix. Standardise each
            column to zero mean and unit variance (with floors for standard deviation)
            to obtain F_mat_std.

            Compute the sample covariance Cov_F and its eigen-decomposition. Retain
            up to `max_factors` principal components with non-negligible eigenvalues
            (e.g. > 1e−12) to obtain F_pc ∈ ℝ^{T×k_keep}. This reduces dimensionality
            and focuses on the dominant modes of factor variation.

        3. **VAR fitting**

            Fit a VAR(p) model on F_pc:

            F_pc,t = c + ∑_{ℓ=1}^p A_ℓ F_pc,t−ℓ + u_t,

            where u_t ∈ ℝ^{k_keep} is the innovation. The order p is selected by an
            information criterion (e.g. BIC) up to `maxlags`, using the `trend`
            argument for deterministic terms.

        4. **Companion form and long-run covariance**

            The VAR is recast in companion form with state vector y_t ∈ ℝ^{k_keep · p}:

                y_t = A_comp y_{t−1} + ε_t,

            where A_comp is the companion matrix built from {A_ℓ} and ε_t has
            covariance Σ_eps_comp, with Σ_eps_pc embedded in its top-left block.

            The h-step-ahead forecast error covariance for the companion process is
            constructed recursively for h = 0,…, horizon_weeks − 1, and the average
            per-week covariance of the principal-component factors is:

                Σ_F_pc_per_week = (1 / horizon_weeks) · ∑_{h=0}^{horizon_weeks−1}
                                Σ_k(h),

            where Σ_k(h) is the covariance of the first k_keep components at horizon h.

            To avoid explosive behaviour, the spectral radius ρ(A_comp) is checked
            and, if necessary, A_comp is scaled to keep ρ ≤ ρ_max (e.g. 0.98).

    5. **Mapping factors to asset returns**

        Align the asset returns X with F_pc; if lengths differ, use the last T₂
        rows of X where T₂ is the number of observations of F_pc. Perform a
        regression:

            X_t = α + B_pc F_pc,t + ε_t,

        yielding B_pc via least squares (without ridge). The long-run asset
        covariance implied by factors is:

            Σ_asset = B_pcᵀ Σ_F_pc_per_week B_pc.

    6. **Conversion to correlation**

        The long-run correlation matrix is:

            R_longrun = Σ_asset / (σ σᵀ),

        where σ_i = sqrt(Σ_asset_{ii}). This is cleaned using `_clean_corr_matrix`
        to enforce symmetry and unit diagonal.

    Parameters
    ----------
    weekly_returns : pd.DataFrame
        Weekly asset returns.
    ff_factors_weekly, index_returns_weekly, macro_factors_weekly,
    industry_returns_weekly, sector_returns_weekly : pd.DataFrame or None
        Factor and grouping series used for the long-run factor model.
    asset_idx : list of str
        Subset of asset columns for which the correlation is computed.
    horizon_weeks : int
        Horizon (in weeks) over which to compute the long-run covariance.
    max_factors : int, optional
        Maximum number of principal components to retain.
    eps : float, optional
        Minimum variance floor.
    trend : str, optional
        Trend specification for VAR (passed to statsmodels).

    Returns
    -------
    np.ndarray or None
        Long-run asset correlation matrix R_longrun if estimation succeeds,
        otherwise None.

    Advantages
    ----------
    This approach uses a dynamic factor model to incorporate temporal
    dependencies in factors when estimating long-run correlation. By modelling
    factors with a VAR and propagating their innovations over the horizon, the
    construction reflects both cross-sectional factor structure and time-series
    persistence, rather than treating factors as i.i.d. This yields a more
    forward-looking correlation estimate, especially relevant for longer-horizon
    portfolio and risk calculations.
    """
    
    factor_frames: list[pd.DataFrame] = []

    if ff_factors_weekly is not None:
      
        ff = ff_factors_weekly.copy()
      
        if "RF" in ff.columns:
      
            ff = ff.drop(columns=["RF"])
      
        factor_frames.append(ff)

    for fb in (index_returns_weekly, macro_factors_weekly, industry_returns_weekly, sector_returns_weekly):
      
        if fb is not None:
      
            factor_frames.append(fb)

    if len(factor_frames) == 0:
      
        return None

    idx = weekly_returns.index
    
    for fr in factor_frames:
    
        idx = idx.intersection(fr.index)

    if len(idx) < max(horizon_weeks, 60):
    
        return None

    R = weekly_returns.loc[idx, asset_idx].replace([np.inf, -np.inf], np.nan)
    
    F_all = pd.concat([fr.loc[idx] for fr in factor_frames], axis = 1)
    
    F_all = F_all.replace([np.inf, -np.inf], np.nan)

    joint = pd.concat([R, F_all], axis = 1).dropna(how = "any")
    
    if joint.shape[0] < max(horizon_weeks, 60):
    
        return None

    N = len(asset_idx)
    
    X = joint.iloc[:, :N].to_numpy(float)
    
    F_mat_raw = joint.iloc[:, N:].to_numpy(float)

    T, K = F_mat_raw.shape
    
    if T <= 5 or K == 0:
    
        return None

    F_mean = F_mat_raw.mean(axis = 0, keepdims = True)

    F_std = F_mat_raw.std(axis = 0, keepdims = True) + 1e-10

    F_std[F_std == 0.0] = 1.0

    F_std = np.where(np.isfinite(F_std), F_std, 1.0).astype(float)

    F_mat_std = (F_mat_raw - F_mean) / F_std

    Cov_F = np.cov(F_mat_std, rowvar = False, ddof = 0)
    
    evals, evecs = np.linalg.eigh(Cov_F)
    
    order = np.argsort(evals)[::-1]
    
    evals = evals[order]
    
    evecs = evecs[:, order]

    k_keep = min(max_factors, np.count_nonzero(evals > 1e-10))
    
    if k_keep == 0:
    
        return None

    evecs_k = evecs[:, :k_keep]
    
    F_pc = F_mat_std @ evecs_k  

    logger.debug("Factor VAR: PCA reduced from K=%d to k_keep=%d PCs", K, k_keep)

    ntrend = 0

    if trend and trend != "n":

        ntrend = 1

    k_pc = F_pc.shape[1]

    max_estimable = max((T - k_pc - ntrend) // (1 + k_pc), 0)

    if max_estimable == 0:

        logger.warning(
            "Factor VAR: not enough obs for VAR (T=%d, k=%d). Using static PCA cov.",
            T, k_pc,
        )

        Sigma_pc = np.cov(F_pc, rowvar = False, ddof = 0)
       
        diag = np.clip(np.diag(Sigma_pc), eps, None)
       
        std = np.sqrt(diag)
       
        R_pc = Sigma_pc / np.outer(std, std)
       
        R_pc = _clean_corr_matrix(
            R = R_pc
        )
       
        return R_pc

    maxlags = min(max_estimable, F_pc.shape[0] // 10, 4) or 1

    logger.info(
        "Factor VAR: T=%d, k=%d, maxlags=%d, max_estimable=%d, trend=%s",
        T, k_pc, maxlags, max_estimable, trend,
    )

    model = VAR(F_pc)

    try:

        sel = model.select_order(maxlags = maxlags, trend = trend)

        p_opt = getattr(sel, "bic", None)

        if p_opt is None or p_opt < 1:

            p_opt = 1

        res = model.fit(p_opt, trend = trend)
        
    except Exception as e:
        
        logger.warning("VAR fit failed: %s", e)
        
        return None

    coefs = res.coefs      
      
    Sigma_eps_pc = res.sigma_u 
    
    p = coefs.shape[0]

    Kp = k_keep * p

    A_comp = np.zeros((Kp, Kp))

    for i in range(p):

        A_comp[:k_keep, i * k_keep:(i + 1) * k_keep] = coefs[i]

    if p > 1:

        A_comp[k_keep:, :-k_keep] = np.eye(k_keep * (p - 1))

    Sigma_eps_comp = np.zeros((Kp, Kp))
    
    Sigma_eps_comp[:k_keep, :k_keep] = Sigma_eps_pc

    try:

        eigvals = np.linalg.eigvals(A_comp)

        rho = float(np.max(np.abs(eigvals)))

    except Exception:

        logger.warning("VAR companion spectral radius computation failed, rho set to NaN.")

        rho = np.nan

    rho_max = 0.98

    if np.isfinite(rho) and rho > rho_max:

        scale = rho_max / rho

        A_comp *= scale

        logger.debug(
            "VAR companion spectral radius %.4f > %.4f; scaled A_comp by %.4f",
            rho, rho_max, scale,
        )

    else:

        logger.debug(
            "VAR companion spectral radius=%.4f",
            rho if np.isfinite(rho) else -1.0,
        )

    logger.info("Factor VAR: selected order p = %d", p)

    Sigma_sum_comp = np.zeros((Kp, Kp))

    Ak = np.eye(Kp)

    for h in range(horizon_weeks):

        if h == 0:

            Sigma_k = Sigma_eps_comp

        else:

            Ak = Ak @ A_comp

            Sigma_k = Ak @ Sigma_eps_comp @ Ak.T

        Sigma_sum_comp += Sigma_k

    Sigma_F_pc_per_week = Sigma_sum_comp[:k_keep, :k_keep] / float(horizon_weeks)

    T2 = F_pc.shape[0]

    if X.shape[0] != T2:
        
        X_reg = X[-T2:, :] 
    
    else:
        
        X_reg = X

    ones = np.ones((T2, 1), float)
  
    F_ext = np.hstack([ones, F_pc])
  
    FtF = F_ext.T @ F_ext
  
    FtX = F_ext.T @ X_reg
  
    try:
  
        B_full = np.linalg.solve(FtF, FtX)
  
    except np.linalg.LinAlgError:
  
        logger.warning("Regression to map PCs to assets failed.")
  
        return None
  
    B_pc = B_full[1:, :] 

    Sigma_asset = B_pc.T @ Sigma_F_pc_per_week @ B_pc
 
    Sigma_asset = _clean_cov_matrix(
        M = Sigma_asset
    )

    diag = np.diag(Sigma_asset).copy()
    
    diag = np.clip(diag, eps, None)
    
    std = np.sqrt(diag)
    
    R_longrun = Sigma_asset / np.outer(std, std)
    
    R_longrun = _clean_corr_matrix(
        R = R_longrun
    )
    
    return R_longrun


def _build_fpred_corr(
    idx: list[str],
    T_ref: np.ndarray,
    Corr_ms: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    ff_factors_weekly: pd.DataFrame | None,
    index_returns_weekly: pd.DataFrame | None,
    macro_factors_weekly: pd.DataFrame | None,
    industry_returns_weekly: pd.DataFrame | None,
    sector_returns_weekly: pd.DataFrame | None,
    horizon_weeks: float,
    *,
    trend: str = "c",
) -> np.ndarray:
    """
    Construct a forward-looking asset correlation matrix R_forecast that blends
    multiple correlation views: historical, multi-scale, constant-correlation
    anchor, and optionally a factor-based long-run estimate.

    The components are:

        1. **Reference correlation R_ref**

            T_ref is a weekly covariance matrix (e.g. Ledoit–Wolf). Compute its
            implied correlation:

            
                σ_ref = sqrt(diag(T_ref)),
            
                R_ref = T_ref / (σ_ref σ_refᵀ),

            then clean via `_clean_corr_matrix`.

        2. **Multi-scale correlation R_ms**

            Corr_ms is a multi-horizon daily correlation (convex average of 1-, 3-,
            and 5-year daily correlations). Clean it to obtain:

            R_ms = _clean_corr_matrix(Corr_ms).

        3. **Constant-correlation anchor R_const**

            Extract off-diagonal entries of R_ms and compute their mean ρ̄:

                ρ̄ = mean(R_ms_{ij}, i < j).

            Construct a constant-correlation matrix:

                R_const = (1 − ρ̄) I + ρ̄ 1 1ᵀ,

            where I is the identity and 1 is the vector of ones. This encodes the
            belief that correlations are roughly homogeneous with average ρ̄.

        4. **Factor-based long-run correlation R_longrun (optional)**

            Attempt to compute a factor-based long-run correlation R_longrun using
            `_compute_factor_based_longrun_asset_corr`. If this step fails, this
            component is omitted.

        5. **Convex blending**

            A set of weights is assigned:

                - w_ref   (e.g. 0.45) to R_ref,
                
                - w_ms    (e.g. 0.30) to R_ms,
                
                - w_const (e.g. 0.05) to R_const,
                
                - w_long  (e.g. 0.20) to R_longrun if available, otherwise 0.

            These weights are renormalised to sum to 1. The forward correlation is:

                R_forecast = w_ref R_ref + w_ms R_ms + w_const R_const + w_long R_longrun,

            where the last term is included only if R_longrun is not None.

            Finally, R_forecast is cleaned via `_clean_corr_matrix`.

    Parameters
    ----------
    idx : list of str
        Asset identifiers.
    T_ref : np.ndarray
        Reference covariance matrix.
    Corr_ms : pd.DataFrame
        Multi-scale daily correlation.
    weekly_5y : pd.DataFrame
        Weekly returns.
    ff_factors_weekly, index_returns_weekly, macro_factors_weekly,
    industry_returns_weekly, sector_returns_weekly : pd.DataFrame or None
        Factor/grouping data for long-run factor-based correlation.
    horizon_weeks : float
        Horizon in weeks for the long-run factor model.
    trend : str, optional
        Trend specification passed to VAR for long-run factor-based correlation.

    Returns
    -------
    np.ndarray
        Forward-looking correlation matrix R_forecast.

    Advantages
    ----------
    This construction blends complementary information sources: robust
    historical covariance (R_ref), rich daily-frequency co-movement (R_ms),
    a simple prior belief (R_const), and dynamic factor structure over the
    forecast horizon (R_longrun). The resulting R_forecast is more robust to
    modelling and sampling error than any single source alone.
    """

    n = len(idx)

    std_ref = np.sqrt(np.clip(np.diag(T_ref), 1e-10, None))

    R_ref = _clean_corr_matrix(
        R = T_ref / np.outer(std_ref, std_ref)
    )

    R_ms = _clean_corr_matrix(
        R = Corr_ms.values
    )

    offs = R_ms[np.triu_indices_from(R_ms, k = 1)]
    
    if offs.size > 0:
        
        rho_bar = float(offs.mean())  
        
    else:
        
        rho_bar = 0.0
        
    R_const = (1 - rho_bar) * np.eye(n) + rho_bar * np.ones((n, n))

    R_longrun = _compute_factor_based_longrun_asset_corr(
        weekly_returns = weekly_5y,
        ff_factors_weekly = ff_factors_weekly,
        index_returns_weekly = index_returns_weekly,
        macro_factors_weekly = macro_factors_weekly,
        industry_returns_weekly = industry_returns_weekly,
        sector_returns_weekly = sector_returns_weekly,
        asset_idx = idx,
        horizon_weeks = int(max(horizon_weeks, 1)),
        trend = trend,
    )

    w_ref = 0.45
   
    w_ms = 0.3
   
    w_const = 0.05
    
    if R_longrun is not None:
        
        w_long = 0.2
        
    else:
        
        w_long = 0.0

    total = w_ref + w_ms + w_const + w_long
   
    w_ref /= total
   
    w_ms /= total
   
    w_const /= total
   
    if R_longrun is not None:
   
        w_long /= total

    R_forecast = (
        w_ref * R_ref
        + w_ms * R_ms
        + w_const * R_const
    )
   
    if R_longrun is not None:
   
        R_forecast = R_forecast + w_long * R_longrun

    R_forecast = _clean_corr_matrix(
        R = R_forecast
    )
   
    return R_forecast


def _calibrate_term_structure_params(
    Sigma_factor_wk: np.ndarray,
    Sigma_idio_wk: np.ndarray,
    horizon_covs: list[np.ndarray],
    horizon_weeks: list[float],
    gamma_min: float = 0.0,
    gamma_max: float = 2.0,
    gamma_step: float = 0.1,
    eps: float = 1e-10,
) -> tuple[float, float, float]:
    """
    Calibrate parameters (a, b, gamma) for a two-component volatility
    term-structure model:

        Σ_obs(h) ≈ a · h · Σ_factor_wk + b · h^{gamma} · Σ_idio_wk,

    where:

        - Σ_obs(h) is the observed covariance at horizon h (in weeks),
        
        - Σ_factor_wk is the weekly factor covariance,
        
        - Σ_idio_wk is the weekly idiosyncratic covariance,
        
        - a ≥ 0, b ≥ 0 are scaling parameters,
        
        - gamma ≥ 0 controls the non-linearity of idiosyncratic growth with horizon.

    The model posits that factor risk scales linearly with horizon, a · h, while
    idiosyncratic risk may scale at a different rate h^{gamma}, capturing
    mean-reversion or other non-linear behaviours.

    Given a set of horizon covariances and horizons:

        { (Σ_obs(h_k), h_k) } for k = 1,…,K,

    the calibration solves, for each candidate gamma on a grid [gamma_min,
    gamma_max] with step gamma_step, the linear regression problem in a and b
    under a least-squares criterion in Frobenius norm.

    For fixed gamma, define:

        Z₁(h_k) = h_k Σ_factor_wk,
   
        Z₂(h_k) = h_k^{gamma} Σ_idio_wk.

    The model covariance at horizon h_k is:

        Σ_model(h_k) = a · Z₁(h_k) + b · Z₂(h_k).

    The sum of squared errors is:

        J(a, b | gamma) = ∑_{k} ∥Σ_obs(h_k) − Σ_model(h_k)∥_F².

    This is quadratic in (a, b) and can be written in terms of inner products
    ip(A,B) = ∑_{ij} A_{ij} B_{ij}:

        A11 = ∑_k ip(Z₁(h_k), Z₁(h_k)),
        
        A22 = ∑_k ip(Z₂(h_k), Z₂(h_k)),
        
        A12 = ∑_k ip(Z₁(h_k), Z₂(h_k)),
        
        y1  = ∑_k ip(Z₁(h_k), Σ_obs(h_k)),
        
        y2  = ∑_k ip(Z₂(h_k), Σ_obs(h_k)).

    Then the optimal (a_hat, b_hat) for fixed gamma solves:

        [ A11  A12 ] [a] = [y1],
        [ A12  A22 ] [b]   [y2],

    provided the determinant A11 A22 − A12² is not near zero.

    The function:

        1. Computes inner products once for factor and idio matrices (FF, II, FI)
        to detect degenerate cases.
        
        2. Iterates gamma from gamma_min to gamma_max in steps of gamma_step,
        solves the 2×2 system for (a_hat, b_hat) if well-conditioned, and
        evaluates the objective J(a_hat, b_hat | gamma).
        
        3. Keeps the best (a_hat, b_hat, gamma) triple minimising J.
        
        4. Clips a_hat and b_hat to [0.1, 5.0] and gamma to [gamma_min, gamma_max].

    Parameters
    ----------
    Sigma_factor_wk : np.ndarray
        Weekly factor covariance Σ_factor_wk.
    Sigma_idio_wk : np.ndarray
        Weekly idiosyncratic covariance Σ_idio_wk.
    horizon_covs : list of np.ndarray
        Observed covariance matrices at different horizons Σ_obs(h_k).
    horizon_weeks : list of float
        Corresponding horizons in weeks h_k.
    gamma_min, gamma_max : float, optional
        Minimum and maximum gamma values considered.
    gamma_step : float, optional
        Step size in the gamma grid.
    eps : float, optional
        Threshold used to detect degeneracies in inner products.

    Returns
    -------
    a_hat : float
        Calibrated factor scaling parameter.
    b_hat : float
        Calibrated idiosyncratic scaling parameter.
    gamma_hat : float
        Calibrated idiosyncratic exponent.

    Advantages
    ----------
    By calibrating a and b jointly with gamma across multiple horizons, this
    approach exploits cross-horizon information to distinguish factor and
    idiosyncratic term-structure behaviour. It avoids overfitting by using a
    low-dimensional parametric form and ensures that the calibrated model
    remains interpretable and consistent with observed multi-horizon covariances.
    """
    
    Sigma_factor_wk = np.asarray(Sigma_factor_wk, dtype = float)
    
    Sigma_idio_wk = np.asarray(Sigma_idio_wk, dtype = float)


    def ip(
        A: np.ndarray, 
        B: np.ndarray
    ) -> float:
        """
        Compute the Frobenius inner product between two matrices.

        Definition
        ----------
        For conformable matrices A and B:

            ip(A, B) = sum_i sum_j A_ij * B_ij.

        Parameters
        ----------
        A, B : np.ndarray
            Matrices of identical shape.

        Returns
        -------
        float
            Frobenius inner product.

        Role in calibration
        -------------------
        The inner product supplies compact sufficient statistics for the normal
        equations used in term-structure parameter estimation.
        """
     
        return float(np.sum(A * B))


    FF = ip(
        A = Sigma_factor_wk, 
        B = Sigma_factor_wk
    )

    II = ip(
        A = Sigma_idio_wk, 
        B = Sigma_idio_wk
    )

    FI = ip(
        A = Sigma_factor_wk, 
        B = Sigma_idio_wk
    )

    if FF < eps or II < eps:

        return 1.0, 1.0, 0.5

    best_obj = np.inf
    
    best_params = (1.0, 1.0, 0.5)

    for gamma in np.arange(gamma_min, gamma_max + 1e-10, gamma_step):

        A11 = 0.0

        A22 = 0.0

        A12 = 0.0

        y1 = 0.0

        y2 = 0.0

        const_term = 0.0

        for S_h, h in zip(horizon_covs, horizon_weeks):

            S_h = np.asarray(S_h, dtype = float)

            h1 = float(h)

            h_g = h1 ** gamma

            Z1 = h1 * Sigma_factor_wk
            
            Z2 = h_g * Sigma_idio_wk

            A11 += ip(
                A = Z1, 
                B = Z1
            )
            
            A22 += ip(
                A = Z2,
                B = Z2
            )
            
            A12 += ip(
                A = Z1,
                B = Z2
            )
            
            y1 += ip(
                A = Z1, 
                B = S_h
            )
            
            y2 += ip(
                A = Z2, 
                B = S_h
            )
            
            const_term += ip(
                A = S_h, 
                B = S_h
            )

        det = A11 * A22 - A12 * A12

        if abs(det) < eps:

            continue

        a_hat = ( A22 * y1 - A12 * y2) / det

        b_hat = (-A12 * y1 + A11 * y2) / det

        obj = 0.0

        for S_h, h in zip(horizon_covs, horizon_weeks):

            h1 = float(h)

            h_g = h1 ** gamma

            S_model = a_hat * h1 * Sigma_factor_wk + b_hat * h_g * Sigma_idio_wk

            diff = S_h - S_model

            obj += ip(diff, diff)

        if obj < best_obj:
            
            best_obj = obj
            
            best_params = (a_hat, b_hat, gamma)

    a_hat, b_hat, gamma_hat = best_params

    a_hat = float(np.clip(a_hat, 0.1, 5.0))
    
    b_hat = float(np.clip(b_hat, 0.1, 5.0))
    
    gamma_hat = float(np.clip(gamma_hat, gamma_min, gamma_max))

    logger.info(
        "Term-structure calibration: a=%.4f, b=%.4f, gamma=%.4f, obj=%.4e",
        a_hat, b_hat, gamma_hat, best_obj,
    )

    return a_hat, b_hat, gamma_hat


def _solve_shrinkage_weights(
    S: np.ndarray,
    T_ref: np.ndarray,
    names: list[str],
    mats: list[np.ndarray],
    *,
    n_factor_directions: int,
    alpha_factor_tracking: float,
    tau_logdet: float,
    logdet_eps: float,
    alpha_trace: float,
    s_min: float,
    s_max: float | None,
    w_S: float | None,
    bounds_kwargs: dict,
    solve_method: str = "cvxpy",
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Solve for an optimal set of shrinkage weights w_hat that combines multiple
    covariance targets with a base covariance S into a single weekly covariance
    C_wk, subject to constraints and regularisation penalties.

    The combined covariance is:

        Σ(w) = S + ∑_{j=1}^m w_j (M_j − S),

    where:

        - S ∈ ℝ^{n×n} is the base weekly covariance,
        
        - M_j ∈ ℝ^{n×n} are shrinkage target covariances,
        
        - w ∈ ℝ^m is the vector of target weights.

    Equivalent form:

        Σ(w) = (1 − ∑ w_j) S + ∑ w_j M_j,

    so that the implicit weight on S is 1 − ∑ w_j.

    **Constraints**

        The optimisation is carried out in w-space with constraints:

            - w_j ≥ 0 for all j,
            
            - lb_j ≤ w_j ≤ ub_j (bounds from `_make_bounds_for_targets`),
            
            - ∑ w_j ≤ 1,
            
            - Additional simplex constraints ensuring 1 − ∑ w_j lies in [s_min, s_max]
            if s_max is not None,
            
            - Positive definiteness: Σ(w) ≽ logdet_eps I.

        If `w_S` is specified, ∑ w_j is fixed to 1 − w_S rather than being merely
        bounded.

    **Objective function**

        The objective minimises a regularised distance between Σ(w) and reference
        covariance T_ref, with additional penalties:

            J(w) = ∥Σ(w) − T_ref∥_F²
                + α_factor_tracking · ∑_{k=1}^{K_dir} (v_k(w) − v_ref,k)²
                + α_trace · trace(Σ(w))
                − τ_logdet · log det(Σ(w)),

        where:

            - v_k(w) is the variance of Σ(w) along the k-th factor direction q_k,
            
            - v_ref,k is the corresponding variance under T_ref:
            v_ref,k = q_kᵀ T_ref q_k,
            
            - q_k are the leading K_dir eigenvectors of T_ref (factor directions),
            
            - α_factor_tracking is `alpha_factor_tracking`,
            
            - α_trace is `alpha_trace`,
            
            - τ_logdet is `tau_logdet`.

        The factor-tracking term penalises deviations in key variance directions,
        ensuring Σ(w) is not only close to T_ref in Frobenius norm but also
        directionally similar in a low-dimensional subspace. The trace penalty
        discourages unnecessarily large variances, while the negative log-det term
        promotes well-conditioned, spread-out eigenvalues (acting as a barrier
        against degeneracy).

    **Computation**

        1.  Compute eigen-decomposition of T_ref and select K_dir leading eigenvectors
            Q_dirs. For each direction q_k:

                - c_dir,k = q_kᵀ S q_k,
                
                - v_ref,k = q_kᵀ T_ref q_k,
                
                - g_{k,j} = q_kᵀ (M_j − S) q_k,

            so that the directional variance under Σ(w) can be written as:

                v_k(w) = c_dir,k + ∑_j g_{k,j} w_j.

        2.  Define a cvxpy variable w and construct Σ(w) explicitly as:

                Σ(w) = S + ∑_j w_j (M_j − S),

            symmetrised.

        3.  Build constraints:

            - Box constraints w ≥ lb, w ≤ ub,
            
            - Simplex constraint ∑ w_j ≤ 1 and, if applicable,
                range constraints on ∑ w_j,
            
            - Positive definiteness Σ(w) ≽ logdet_eps I.

        4.   Formulate the objective as above and solve with a suitable solver (MOSEK
            if available, otherwise SCS).

        5.  If the solver fails or returns non-finite weights, fall back to a
            non-negative least-squares (NNLS) solution for:

                vec(T_ref − S) ≈ A w,

            where A has columns vec(M_j − S), followed by projection of the resulting
            w onto the constrained simplex using `_project_boxed_simplex_leq`.

        6.  Finally, recompute Σ(w_hat) and project it to the nearest PSD matrix with
            `_nearest_psd_preserve_diag`.

    Parameters
    ----------
    S : np.ndarray
        Base weekly covariance.
    T_ref : np.ndarray
        Reference weekly covariance.
    names : list of str
        Names of the covariance targets.
    mats : list of np.ndarray
        Covariance target matrices M_j.
    n_factor_directions : int
        Number of leading eigen-directions of T_ref used for factor tracking.
    alpha_factor_tracking : float
        Weight for variance tracking penalty in factor directions.
    tau_logdet : float
        Coefficient on the negative log-determinant barrier term.
    logdet_eps : float
        Minimum eigenvalue in the PSD constraint.
    alpha_trace : float
        Coefficient on trace regularisation.
    s_min : float
        Minimum implicit weight on S (through constraints on ∑ w_j).
    s_max : float or None
        Maximum implicit weight on S, if provided.
    w_S : float or None
        Fixed weight on S; if not None, ∑ w_j is fixed to 1 − w_S.
    bounds_kwargs : dict
        Keyword arguments passed to `_make_bounds_for_targets` to generate lb
        and ub.

    Returns
    -------
    w_hat : np.ndarray
        Optimal shrinkage weights for the targets.
    w_map : dict
        Mapping from target names to their weights, with an additional entry
        "S (implicit)" for the base covariance.
    C_wk : np.ndarray
        Final weekly covariance matrix Σ(w_hat), projected to be PSD with
        preserved diagonal.

    Advantages
    ----------
    This optimisation framework allows flexible blending of many heterogeneous
    covariance targets while ensuring numerical stability, positivity, and
    interpretability. The use of directional tracking, trace penalisation, and
    log-det regularisation balances fidelity to a reference with stability and
    robustness, and the fallback NNLS + projection ensures a valid solution is
    always obtained.
    """
   
    m = len(mats)
   
    n = S.shape[0]

    use_cvxpy = str(solve_method).lower() not in {"nnls", "ls", "least_squares"}

    lb, ub = _make_bounds_for_targets(
        names = names,
        **bounds_kwargs
    )

    if use_cvxpy:

        evals_ref, evecs_ref = np.linalg.eigh(T_ref)

        order = np.argsort(evals_ref)[::-1]

        evals_ref = evals_ref[order]

        evecs_ref = evecs_ref[:, order]

        K_dir = int(min(max(n_factor_directions, 0), n))

        if K_dir > 0:

            Q_dirs = evecs_ref[:, :K_dir]

            g = np.zeros((K_dir, m), dtype = float)

            c_dir = np.zeros(K_dir, dtype = float)

            v_ref_dir = np.zeros(K_dir, dtype = float)

            for k in range(K_dir):
                
                q = Q_dirs[:, k: k + 1]
                
                c_dir[k] = float(q.T @ S @ q)
                
                v_ref_dir[k] = float(q.T @ T_ref @ q)
                
                for j, M in enumerate(mats):
                
                    g[k, j] = float(q.T @ (M - S) @ q)
        
        else:
        
            g = None
        
            c_dir = None
        
            v_ref_dir = None

        w = cp.Variable(m, nonneg = True)

        Sigma_w = S.copy()
        
        for j in range(m):
        
            Sigma_w = Sigma_w + w[j] * (mats[j] - S)
        
        Sigma_w = 0.5 * (Sigma_w + Sigma_w.T)

        cons = [w >= lb, cp.sum(w) <= 1.0]
        
        finite_ub = np.isfinite(ub)
        
        if np.any(finite_ub):
        
            cons.append(w[finite_ub] <= ub[finite_ub])

        eps_ld = float(logdet_eps)
        
        cons.append(Sigma_w >> eps_ld * np.eye(n))

        if w_S is not None:

            cons.append(cp.sum(w) == 1.0 - float(w_S))

        else:

            if s_max is not None:

                cons.append(cp.sum(w) <= 1.0 - float(s_min))

                cons.append(cp.sum(w) >= 1.0 - float(s_max))

            else:

                cons.append(cp.sum(w) >= 1.0 - float(s_min))

        if (alpha_factor_tracking > 0.0) and (K_dir > 0):

            factor_terms = []

            for k in range(K_dir):

                dir_var = c_dir[k] + g[k, :] @ w

                factor_terms.append(cp.square(dir_var - v_ref_dir[k]))

            obj_factor = alpha_factor_tracking * cp.sum(factor_terms)

        else:

            obj_factor = cp.Constant(0.0)

        obj = cp.Minimize(
            cp.sum_squares(Sigma_w - T_ref)
            + obj_factor
            + alpha_trace * cp.trace(Sigma_w)
            - float(tau_logdet) * cp.log_det(Sigma_w)
        )

        prob = cp.Problem(obj, cons)

        solver_choice = None

        if hasattr(cp, "MOSEK"):

            try:

                _x = cp.Variable()

                _p = cp.Problem(cp.Minimize(_x), [_x >= 0])

                _p.solve(solver=cp.MOSEK, verbose=False)

                solver_choice = cp.MOSEK

            except Exception:

                solver_choice = None  

        if solver_choice is None:
            
            solver_choice = cp.SCS

        logger.info("Using CVXPY solver: %s", solver_choice)

        prob.solve(
            solver = solver_choice,
            verbose = False,
            **(
                {
                    "eps": 1e-10,
                    "max_iters": 50000
                }
                if solver_choice == cp.SCS
                else {}
            ),
        )

        logger.info("Shrinkage weights optimisation status: %s", prob.status)
        
        if (w.value is None) or (not np.all(np.isfinite(w.value))):
      
            cols = [(M - S).ravel() for M in mats]

            if len(cols):
                
                A = np.column_stack(cols)  
            
            else:
                
                A = np.zeros((S.size, 0))
      
            b_vec = (T_ref - S).ravel()

            if A.shape[1] > 0:
                
                w0 = nnls(A, b_vec)[0]
                
                logger.info("Fallback NNLS weights used: %s", w0)
            
            else:
            
                w0 = np.zeros(m)
            
                logger.info("No shrinkage targets; zero weights used.")

            w_hat = _project_boxed_simplex_leq(
                v = w0, 
                lb = lb, 
                ub = ub,
                s = 1.0
            )
            
        else:
            
            logger.info("Optimisation successful; CVXPY weights used.")
            
            w_hat = np.clip(w.value, lb, ub)
            
            w_hat = _project_boxed_simplex_leq(
                v = w_hat,
                lb = lb,
                ub = ub, 
                s = 1.0
            )
    else:

        cols = [(M - S).ravel() for M in mats]

        if len(cols):
            
            A = np.column_stack(cols)
        
        else:
            
            A = np.zeros((S.size, 0))
  
        b_vec = (T_ref - S).ravel()

        if A.shape[1] > 0:
            
            w0 = nnls(A, b_vec)[0]
            
            logger.info("NNLS weights used (solve_method=%s): %s", solve_method, w0)
        
        else:
        
            w0 = np.zeros(m)
        
            logger.info("No shrinkage targets; zero weights used.")

        w_hat = _project_boxed_simplex_leq(
            v = w0,
            lb = lb,
            ub = ub,
            s = 1.0,
        )

    if not np.all(np.isfinite(w_hat)):
        
        raise ValueError(f"Non-finite shrinkage weights after projection: {w_hat}")

    C_wk = S.copy()

    for wj, M in zip(w_hat, mats):

        C_wk += wj * (M - S)

    C_wk = _nearest_psd_preserve_diag(
        C = C_wk,
        eps = 1e-10
    )

    remaining = float(max(0.0, 1.0 - np.sum(w_hat)))
    
    w_map = {nm: float(wj) for nm, wj in zip(names, w_hat)}
    
    w_map["S (implicit)"] = remaining
    
    logger.info("Shrinkage weights: %s", w_map)

    return w_hat, w_map, C_wk


def _apply_term_structure(
    C_wk: np.ndarray,
    idx: list[str],
    periods_per_year: int,
    use_factor_term_structure: bool,
    *,
    weekly_5y: pd.DataFrame,
    ff_factors_weekly: pd.DataFrame | None,
    index_returns_weekly: pd.DataFrame | None,
    industry_returns_weekly: pd.DataFrame | None,
    sector_returns_weekly: pd.DataFrame | None,
    macro_factors_weekly: pd.DataFrame | None,
    names: list[str],
    S1: np.ndarray,
    S3: np.ndarray,
    S5: np.ndarray,
    horizon_weeks: float,
    gamma_idio: float,
    r2_cap: float,
    factor_share_min: float,
    factor_share_max: float,
) -> np.ndarray:
    """
    Apply an optional factor/idiosyncratic term-structure model to a weekly
    covariance matrix C_wk to obtain an annualised covariance matrix C_ann at a
    specified horizon.

    Two modes are supported:

        1. **Simple scaling (no factor term structure)**

            If `use_factor_term_structure` is False, the function simply scales the
            weekly covariance by the number of periods per year:

                C_ann = periods_per_year · C_wk,

            and then projects C_ann to a PSD matrix with preserved diagonal using
            `_nearest_psd_preserve_diag`.

        2. **Factor/idiosyncratic term structure**

            If `use_factor_term_structure` is True, the function:

                a. Builds a list of factor frames for term-structure modelling,
                    conditional on which shrinkage targets are present (FF, IDX, IND,
                    SEC, MACRO).

                b. Orthogonalises these factor blocks with
                    `_orthogonalise_factor_blocks` to reduce redundancy.

                c. Decomposes the covariance of weekly returns into factor and
                    idiosyncratic components via `_decompose_factor_idio`, using:

                        - Returns aligned to idx,
                        
                        - A target diagonal equal to diag(C_wk),
                        
                        - Bounds (factor_share_min, factor_share_max) and r2_cap for factor
                            shares.

                    This yields Σ_factor_wk and Σ_idio_wk.

                d. Cleans S1, S3, S5 using `_clean_cov_matrix` and constructs a list of
                    observed horizon covariances:

                        horizon_covs = [S1_clean, S3_clean, S5_clean],
                    
                        horizon_weeks_list = [52, 3·52, 5·52].

                e. Calibrates term-structure parameters (a_ts, b_ts, gamma_hat) via
                    `_calibrate_term_structure_params` by fitting:

                        Σ_obs(h) ≈ a_ts · h · Σ_factor_wk + b_ts · h^{gamma_hat} · Σ_idio_wk.

                f. Blends gamma_hat with the user-specified gamma_idio:

                        gamma_use = 0.5 · gamma_hat + 0.5 · gamma_idio,

                    to smooth the idiosyncratic scaling exponent.

                g. For the desired horizon h = horizon_weeks, computes the horizon
                    covariance:

                        Σ_h(h) = a_ts · h · Σ_factor_wk + b_ts · h^{gamma_use} · Σ_idio_wk,

                    then converts back to an annualised covariance:

                        C_ann = (Σ_h(h) / h) · periods_per_year.

                h. Projects C_ann to a PSD matrix with preserved diagonal via
                    `_nearest_psd_preserve_diag`.

    Parameters
    ----------
    C_wk : np.ndarray
        Weekly covariance matrix from shrinkage.
    idx : list of str
        Asset identifiers.
    periods_per_year : int
        Number of weekly periods per year (e.g. 52).
    use_factor_term_structure : bool
        Whether to apply factor/idiosyncratic term-structure or simple scaling.
    weekly_5y : pd.DataFrame
        Weekly returns.
    ff_factors_weekly, index_returns_weekly, industry_returns_weekly,
    sector_returns_weekly, macro_factors_weekly : pd.DataFrame or None
        Factor and grouping series for term-structure decomposition.
    names : list of str
        Names of shrinkage targets present (used to decide which factor frames
        to include).
    S1, S3, S5 : np.ndarray
        Weekly covariances at 1-, 3-, and 5-year horizons.
    horizon_weeks : float
        Target horizon in weeks for C_ann.
    gamma_idio : float
        User prior on idiosyncratic scaling exponent.
    r2_cap : float
        Upper bound on factor share per name in decomposition.
    factor_share_min, factor_share_max : float
        Bounds on factor shares per name.

    Returns
    -------
    np.ndarray
        Annualised covariance matrix C_ann.

    Advantages
    ----------
    This function allows the model to distinguish between the term-structure of
    factor and idiosyncratic risk. Factor risk is often closer to a random-walk
    behaviour (scaling roughly linearly with horizon), while idiosyncratic risk
    can exhibit mean-reversion or other non-linear scaling. Calibrating these
    patterns across multiple horizons makes the resulting annualised covariance
    better aligned with observed multi-horizon behaviour than simple linear
    scaling would allow.
    """

    if not use_factor_term_structure:
    
        C_ann = C_wk * float(periods_per_year)
    
        C_ann = _nearest_psd_preserve_diag(
            C = C_ann, 
            eps = 1e-10
        )
    
        return C_ann

    factor_frames_for_ts: list[pd.DataFrame] = []

    if ("FF" in names) and (ff_factors_weekly is not None):

        ff2 = ff_factors_weekly.copy()

        if "RF" in ff2.columns:

            ff2 = ff2.drop(columns = ["RF"])
            
        factor_frames_for_ts.append(ff2)
    
    if ("IDX" in names) and (index_returns_weekly is not None):
    
        factor_frames_for_ts.append(index_returns_weekly)
    
    if ("IND" in names) and (industry_returns_weekly is not None):
    
        factor_frames_for_ts.append(industry_returns_weekly)
    
    if ("SEC" in names) and (sector_returns_weekly is not None):
    
        factor_frames_for_ts.append(sector_returns_weekly)
    
    if ("MACRO" in names) and (macro_factors_weekly is not None):
    
        factor_frames_for_ts.append(macro_factors_weekly)

    if factor_frames_for_ts:
     
        idx_ts = factor_frames_for_ts[0].index
     
        for fr in factor_frames_for_ts[1:]:
     
            idx_ts = idx_ts.intersection(fr.index)

        if len(idx_ts) < 10:

            factor_frames_for_ts = []

        else:

            factor_frames_for_ts = [fr.loc[idx_ts].copy() for fr in factor_frames_for_ts]

            factor_frames_for_ts = _orthogonalise_factor_blocks(
                factor_frames = factor_frames_for_ts
            )
    
    returns_weekly = weekly_5y.loc[:, idx]

    try:
       
        Sigma_factor_wk, Sigma_idio_wk = _decompose_factor_idio(
            returns_weekly = returns_weekly,
            factor_frames = factor_frames_for_ts,
            ridge = 1e-4,
            eps = 1e-10,
            keep_idio_offdiag = True,
            idio_offdiag_shrink = 0.25,
            max_lag_factors = "auto",
            diag_target = np.diag(C_wk).copy(),
            r2_cap = r2_cap,
            factor_share_bounds = (factor_share_min, factor_share_max),
        )

        if (not np.all(np.isfinite(Sigma_factor_wk)) or not np.all(np.isfinite(Sigma_idio_wk))):
            
            raise ValueError("Non-finite factor/idio matrices.")
  
    except Exception:
  
        logger.warning("Term-structure factor/idio decomposition failed; using diag/idiosyncratic split.")
  
        diagC = np.diag(np.diag(C_wk))
  
        Sigma_factor_wk = _clean_cov_matrix(
            M = C_wk - diagC
        )
  
        Sigma_idio_wk = _clean_cov_matrix(
            M = diagC
        )

    S1_clean = _clean_cov_matrix(
        M = S1
    )

    S3_clean = _clean_cov_matrix(
        M = S3
    )

    S5_clean = _clean_cov_matrix(
        M = S5
    )

    horizon_covs = [S1_clean, S3_clean, S5_clean]
    
    horizon_weeks_list = [52.0, 3.0 * 52.0, 5.0 * 52.0]

    a_ts, b_ts, gamma_hat = _calibrate_term_structure_params(
        Sigma_factor_wk = Sigma_factor_wk,
        Sigma_idio_wk = Sigma_idio_wk,
        horizon_covs = horizon_covs,
        horizon_weeks = horizon_weeks_list,
        gamma_min = 0.0,
        gamma_max = 2.0,
        gamma_step = 0.1,
    )

    gamma_use = 0.5 * gamma_hat + 0.5 * gamma_idio

    h = float(max(horizon_weeks, 1e-10))

    Sigma_horizon_hweeks = (a_ts * h * Sigma_factor_wk) + (b_ts * h ** gamma_use * Sigma_idio_wk)

    C_ann = (Sigma_horizon_hweeks / h) * float(periods_per_year)
    
    C_ann = _nearest_psd_preserve_diag(
        C = C_ann, 
        eps = 1e-10
    )
    
    return C_ann


def shrinkage_covariance(
    daily_5y: pd.DataFrame,
    weekly_5y: pd.DataFrame,
    monthly_5y: pd.DataFrame,
    comb_std: pd.Series,
    common_idx: list[str],
    *,
    ff_factors_weekly: pd.DataFrame | None = None,
    index_returns_weekly: pd.DataFrame | None = None,
    industry_returns_weekly: pd.DataFrame | None = None,
    sector_returns_weekly: pd.DataFrame | None = None,
    macro_factors_weekly: pd.DataFrame | None = None,
    fx_factors_weekly: pd.DataFrame | None = None,
    fund_exposures_weekly: dict[str, pd.DataFrame] | None = None,
    daily_open_5y: pd.DataFrame | None = None,
    daily_close_5y: pd.DataFrame | None = None,
    sector_map: pd.Series | None = None,
    industry_map: pd.Series | None = None,
    use_excess_ff: bool = True,
    description: bool = False,
    use_log_returns: bool | None = None,
    use_oas: bool | None = None,
    use_block_prior: bool | None = None,
    use_regime_ewma: bool | None = None,
    use_glasso: bool | None = None,
    use_fund_factors: bool | None = None,
    use_fx_factors: bool | None = None,
    cache_dir: Path | None = None,
    cache_mode: str | None = None,

    # Recency/regularisation
    ewma_lambda: float = 0.97,
    tau_logdet: float = 1e-3,
    logdet_eps: float = 1e-10,
    alpha_factor_tracking: float = 1e-2,
    n_factor_directions: int = 5,
    alpha_trace: float = 1e-4,
    horizon_weeks: float = 52,

    # Factor term structure
    use_factor_term_structure: bool = False,
    gamma_idio: float = 0.5,
    r2_cap: float = 0.9,
    factor_share_min: float = 0.2,
    factor_share_max: float = 0.8,

    # Optional fixed weights
    w_S: float | None = None,
    w_P: float | None = None,
    w_S_EWMA: float | None = None,
    w_C_EWMA: float | None = None,
    w_F: float | None = None,
    w_FF: float | None = None,
    w_IDX: float | None = None,
    w_IND: float | None = None,
    w_SEC: float | None = None,
    w_STAT: float | None = None,
    w_OVN: float | None = None,
    w_INTRA: float | None = None,
    w_REGIME: float | None = None,
    w_HIER: float | None = None,
    w_MACRO: float | None = None,
    w_LDA_REGIME: float | None = None,
    w_VaR: float | None = None,

    # Box constraints (mins/maxes)
    s_min: float = 0.05,
    s_max: float | None = 0.20,
    p_min: float = 0.05,
    p_max: float = 0.1,
    s_ewma_min: float = 0.01,
    s_ewma_max: float = 0.08,
    c_ewma_min: float = 0.01,
    c_ewma_max: float = 0.08,
    fpred_min: float = 0.30,
    fpred_max: float = 0.35,
    ff_min: float = 0.01,
    ff_max: float = 0.08,
    idx_min: float = 0.01,
    idx_max: float = 0.08,
    ind_min: float = 0.01,
    ind_max: float = 0.08,
    sec_min: float = 0.01,
    sec_max: float = 0.08,
    stat_min: float = 0.05,
    stat_max: float = 0.15,
    ovn_min: float = 0.01,
    ovn_max: float = 0.08,
    intra_min: float = 0.01,
    intra_max: float = 0.08,
    regime_min: float = 0.01,
    regime_max: float = 0.08,
    hier_min: float = 0.01,
    hier_max: float = 0.08,
    macro_min: float = 0.01,
    macro_max: float = 0.08,
    stat_rmt_min: float = 0.08,
    stat_rmt_max: float = 0.15,
    lda_min: float = 0.01,
    lda_max: float = 0.08,
    VaR_min: float = 0.01,
    VaR_max: float = 0.08,
    oas_min: float = 0.01,
    oas_max: float = 0.08,
    block_min: float = 0.01,
    block_max: float = 0.08,
    glasso_min: float = 0.01,
    glasso_max: float = 0.08,
    s_ewma_regime_min: float = 0.01,
    s_ewma_regime_max: float = 0.08,
    fx_min: float = 0.01,
    fx_max: float = 0.08,
    fund_min: float = 0.05,
    fund_max: float = 0.1,
    periods_per_year: int = 52,
    var_trend: str = "c",
    solve_method: str = "cvxpy",
) -> pd.DataFrame:
    """
    Construct an annualised covariance matrix of equity returns by blending
    multiple structured covariance targets with a base covariance using
    optimised shrinkage weights, and optionally applying a factor-based
    term-structure model.

    High-level workflow
    -------------------
    1. **Base covariance construction**

        `_build_base_covariances` is called to produce:

            - S: base weekly covariance, derived from multi-horizon daily/weekly/
                monthly data.
            
            - T_ref: reference weekly covariance via Ledoit–Wolf shrinkage.
            
            - Corr_ms: multi-horizon daily correlation.
            
            - S_EWMA: EWMA weekly covariance.
            
            - S1, S3, S5: weekly covariances at 1-, 3-, and 5-year horizons.
            
            - weekly_clean: cleaned weekly return panel.
            
            - comb_std_aligned: forecast standard deviations (forecast errors)
                aligned to the asset index.

    2. **Construction of covariance targets**

        `_build_cov_targets` is invoked to compute a range of weekly covariance
        targets, including:

            - Constant-correlation prior (P),
            
            - EWMA-based targets (S_EWMA, C_EWMA),
            
            - Forecast-error covariance (F) based on comb_std_aligned and a
                forward-looking correlation R_forecast,
            
            - Overnight/intraday covariances (OVN, INTRA),
            
            - Factor-model covariances (FF, IDX, IND, SEC, MACRO, HIER),
            
            - Statistical covariances (STAT, STAT_RMT),
            
            - Regime-based covariances (REGIME, LDA_REGIME),
            
            - VaR-based covariance (VaR).

        All targets are in weekly units.

    3. **Shrinkage weight optimisation**

        Bounds and optional fixed weights for each target are constructed via
        `_make_bounds_for_targets` from the numerous Box-constraint parameters
        (p_min, p_max, fpred_min, fpred_max, etc.). These encode prior
        constraints on how much weight each target is allowed to have and allow
        hard-fixing certain weights (e.g. w_P).

        `_solve_shrinkage_weights` is then called with:

            - S and T_ref,
        
            - the list of targets and names,
        
            - hyperparameters controlling factor-direction tracking, trace and
            log-det penalties, and simplex constraints on the sum of weights.

        The output is:

            - w_hat: optimal weights for each target,
       
            - w_map: mapping of names to weights, with "S (implicit)" for the base
            covariance,
        
            - C_wk: final weekly covariance matrix after blending and PSD
            projection.

    4. **Term-structure application**

        `_apply_term_structure` scales C_wk to an annualised covariance C_ann.
        If `use_factor_term_structure` is False, this is simple multiplication
        by periods_per_year. Otherwise, factor/idiosyncratic term-structure
        modelling is used, as described in `_apply_term_structure`.

    5. **Optional diagnostics**

        If `description` is True, the function returns:

        - out: C_ann as a DataFrame,
        
        - desc: descriptive statistics of C_ann,
        
        - extras: a dict containing:
        
            - "weights": shrinkage weight map w_map,
        
            - "eig_min", "eig_max", "eig_trace": eigenvalue diagnostics,
        
            - "vol_forecast_corr": correlation between sqrt(diag(C_ann)) and
            comb_std_aligned (forecast vol), indicating how closely the
            covariance diagonal aligns with forecast volatilities.

    Role of `comb_std`
    ------------------
    `comb_std` is a per-asset forecast volatility or forecast error measure,
    annualised. It enters the model in two ways:

        1. **Forecast covariance target (F)**

            Weekly forecast standard errors are:

                se_wk = comb_std / sqrt(periods_per_year),

            and F_pred is constructed as:

                F_pred = R_forecast ⊙ (se_wk se_wkᵀ),

            where R_forecast is a forward-looking correlation matrix. This ensures
            the diagonal of F_pred reflects the forecast error variance while the
            off-diagonal structure reflects a blended forward-looking correlation.

        2. **Diagnostic correlation**

            The correlation between sqrt(diag(C_ann)) and comb_std_aligned measures
            how consistent the final annualised covariance is with the external
            forecast volatility inputs. A high correlation indicates that the
            shrinkage and term-structure pipeline respects the comb_std signal.

    Parameters
    ----------
    daily_5y, weekly_5y, monthly_5y : pd.DataFrame
        Daily, weekly, and monthly return matrices over a multi-year period.
    comb_std : pd.Series
        Per-asset forecast standard deviation (annualised).
    common_idx : list of str
        List of assets to include.
    ff_factors_weekly, index_returns_weekly, industry_returns_weekly,
    sector_returns_weekly, macro_factors_weekly : pd.DataFrame or None, optional
        Factor and grouping return series.
    daily_open_5y, daily_close_5y : pd.DataFrame or None, optional
        Daily open and close prices for overnight/intraday decomposition.
    use_excess_ff : bool, optional
        Whether to use excess returns for FF factor models.

    description : bool, optional
        If True, return diagnostics in addition to the covariance DataFrame.

    ewma_lambda, tau_logdet, logdet_eps, alpha_factor_tracking, n_factor_directions,
    alpha_trace, horizon_weeks : float or int, optional
        Hyperparameters for EWMA, log-det barrier, directional tracking, trace
        regularisation, and forecast horizon in weeks.

    use_factor_term_structure : bool, optional
        Whether to apply factor/idiosyncratic term-structure scaling.
    gamma_idio, r2_cap, factor_share_min, factor_share_max : float, optional
        Parameters controlling term-structure and factor share calibration.

    w_S, w_P, w_S_EWMA, ..., w_VaR : float or None, optional
        Optional fixed weights for specific targets (if not None).

    s_min, s_max, p_min, p_max, ..., VaR_max : float, optional
        Box constraints for target weights.

    ff_max_lag, idx_max_lag, ind_max_lag, sec_max_lag, macro_max_lag, hier_max_lag :
    int, str, or None, optional
        Maximum lags for the respective factor models.

    periods_per_year : int, optional
        Number of weekly periods per year (default 52).
    var_trend : str, optional
        Trend specification for VAR in long-run correlation estimation.

    Returns
    -------
    pd.DataFrame
        Annualised covariance matrix indexed by assets if description is False.

    If description is True:
        (pd.DataFrame, pd.DataFrame, dict)
            C_ann, descriptive statistics, and extras as described above.

    Advantages
    ----------
    This function orchestrates a covariance modelling pipeline that:

        - Integrates multiple sampling frequencies and historical horizons.
        
        - Incorporates forward-looking information via comb_std and long-run
        factor dynamics.
        
        - Uses multiple structured covariance targets capturing different aspects
        of return dependence (histogram-based, factor-based, statistical,
        regime-based, VaR-based).
        
        - Optimally blends these targets under convex constraints with
        regularisation, ensuring positive definiteness and good conditioning.
        
        - Optionally models differing term structures for factor and idiosyncratic
        risk.

    The result is a flexible, robust, and interpretable covariance estimate
    suitable for portfolio optimisation and risk management applications.
    """

    if use_log_returns is None:

        use_log_returns = bool(getattr(config, "COV_USE_LOG_RETURNS", False))

    if use_oas is None:

        use_oas = bool(getattr(config, "COV_USE_OAS", False))

    if use_block_prior is None:

        use_block_prior = bool(getattr(config, "COV_USE_BLOCK_PRIOR", False))

    if use_regime_ewma is None:

        use_regime_ewma = bool(getattr(config, "COV_USE_REGIME_EWMA", False))

    if use_glasso is None:

        use_glasso = bool(getattr(config, "COV_USE_GLOSSO", False))

    if use_fund_factors is None:

        use_fund_factors = bool(getattr(config, "COV_USE_FUND_FACTORS", False))

    if use_fx_factors is None:

        use_fx_factors = bool(getattr(config, "COV_USE_FX_FACTORS", False))

    if cache_mode is None:

        cache_mode = getattr(config, "COV_CACHE_MODE", "manual")

    if cache_dir is None:

        cache_dir = getattr(config, "COV_CACHE_DIR", Path.cwd() / "cov_cache")

    cache_dir = Path(cache_dir)

    if use_log_returns:
        
        daily_5y = _safe_log_returns(
            df = daily_5y
        )
        
        weekly_5y = _safe_log_returns(
            df = weekly_5y
        )
        
        monthly_5y = _safe_log_returns(
            df = monthly_5y
        )

    if not use_fx_factors:
        
        fx_factors_weekly = None

    flags = {
        "log": use_log_returns,
        "oas": use_oas,
        "block": use_block_prior,
        "regime": use_regime_ewma,
        "glasso": use_glasso,
        "fund": use_fund_factors,
        "fx": use_fx_factors,
        "term": use_factor_term_structure,
    }

    cache_allowed = str(cache_mode).lower() != "off"

    key = _cache_key_from_inputs(
        tickers = list(common_idx),
        daily_5y = daily_5y,
        weekly_5y = weekly_5y,
        monthly_5y = monthly_5y,
        flags = flags,
    )
  
    base_path, tgt_path = _cache_paths(cache_dir, key)

    base_cache = None
  
    tgt_cache = None

    if cache_allowed and (not COV_CACHE_REBUILD):
  
        base_cache = _load_cov_cache(
            cache_path = base_path
        )
  
        tgt_cache = _load_cov_cache(
            cache_path = tgt_path
        )

    if base_cache is not None:
   
        idx = base_cache["idx"]
   
        S = base_cache["S"]
   
        T_ref = base_cache["T_ref"]
   
        Corr_ms = base_cache["Corr_ms"]
   
        S_EWMA = base_cache["S_EWMA"]
   
        S1 = base_cache["S1"]
   
        S3 = base_cache["S3"]
   
        S5 = base_cache["S5"]
   
        comb_std_aligned = comb_std.reindex(idx)
   
        weekly_5y_idx = weekly_5y.loc[:, idx]
   
        monthly_5y_idx = monthly_5y.loc[:, idx]
   
        weekly_clean = weekly_5y_idx.replace([np.inf, -np.inf], np.nan).dropna(how = "any")
   
        daily_ns = None
   
    else:
        (
            idx,
            comb_std_aligned,
            daily_ns,
            weekly_5y_idx,
            monthly_5y_idx,
            S,
            T_ref,
            Corr_ms,
            S_EWMA,
            S1,
            S3,
            S5,
            weekly_clean,
        ) = _build_base_covariances(
            daily_5y = daily_5y,
            weekly_5y = weekly_5y,
            monthly_5y = monthly_5y,
            comb_std = comb_std,
            common_idx = common_idx,
            ewma_lambda = ewma_lambda,
            periods_per_year = periods_per_year,
        )
   
        if cache_allowed:
   
            _save_cov_cache(
                cache_path = base_path,
                obj = {
                    "idx": idx,
                    "S": S,
                    "T_ref": T_ref,
                    "Corr_ms": Corr_ms,
                    "S_EWMA": S_EWMA,
                    "S1": S1,
                    "S3": S3,
                    "S5": S5,
                },
            )

    if tgt_cache is not None:
 
        names = tgt_cache["names"]
 
        mats = tgt_cache["mats"]
 
        aux_meta = tgt_cache.get("aux_meta", {})
 
    else:
 
        if daily_ns is None:
 
            daily_ns = _clean_daily_stale(
                daily = daily_5y.loc[:, idx]
            )
 
        names, mats, aux_meta = _build_cov_targets(
            idx = idx,
            comb_std = comb_std_aligned,
            daily_ns = daily_ns,
            weekly_5y = weekly_5y_idx,
            monthly_5y = monthly_5y_idx,
            S = S,
            T_ref = T_ref,
            Corr_ms = Corr_ms,
            S_EWMA = S_EWMA,
            S1 = S1,
            S3 = S3,
            S5 = S5,
            weekly_clean = weekly_clean,
            ff_factors_weekly = ff_factors_weekly,
            index_returns_weekly = index_returns_weekly,
            industry_returns_weekly = industry_returns_weekly,
            sector_returns_weekly = sector_returns_weekly,
            macro_factors_weekly = macro_factors_weekly,
            fx_factors_weekly = fx_factors_weekly,
            fund_exposures_weekly = fund_exposures_weekly,
            sector_map = sector_map,
            industry_map = industry_map,
            daily_open_5y = daily_open_5y,
            daily_close_5y = daily_close_5y,
            use_excess_ff = use_excess_ff,
            use_oas = use_oas,
            use_block_prior = use_block_prior,
            use_regime_ewma = use_regime_ewma,
            use_glasso = use_glasso,
            use_fund_factors = use_fund_factors,
            horizon_weeks = horizon_weeks,
            periods_per_year = periods_per_year,
            var_trend = var_trend,
        )
 
        if cache_allowed:
 
            _save_cov_cache(
                cache_path = tgt_path,
                obj = {
                    "names": names,
                    "mats": mats,
                    "aux_meta": aux_meta,
                },
            )

    bounds_kwargs = dict(
        w_S = w_S,
        w_P = w_P,
        w_S_EWMA = w_S_EWMA,
        w_C_EWMA = w_C_EWMA,
        w_F = w_F,
        w_FF = w_FF,
        w_IDX = w_IDX,
        w_IND = w_IND,
        w_SEC = w_SEC,
        w_STAT = w_STAT,
        w_OVN = w_OVN,
        w_INTRA = w_INTRA,
        w_REGIME = w_REGIME,
        w_HIER = w_HIER,
        w_MACRO = w_MACRO,
        w_LDA_REGIME = w_LDA_REGIME,
        w_VaR = w_VaR,
        p_min = p_min,
        p_max = p_max,
        s_ewma_min = s_ewma_min,
        s_ewma_max = s_ewma_max,
        c_ewma_min = c_ewma_min,
        c_ewma_max = c_ewma_max,
        fpred_min = fpred_min,
        fpred_max = fpred_max,
        ff_min = ff_min,
        ff_max = ff_max,
        idx_min = idx_min,
        idx_max = idx_max,
        ind_min = ind_min,
        ind_max = ind_max,
        sec_min = sec_min,
        sec_max = sec_max,
        stat_min = stat_min,
        stat_max = stat_max,
        stat_rmt_min = stat_rmt_min,
        stat_rmt_max = stat_rmt_max,
        ovn_min = ovn_min,
        ovn_max = ovn_max,
        intra_min = intra_min,
        intra_max = intra_max,
        regime_min = regime_min,
        regime_max = regime_max,
        hier_min = hier_min,
        hier_max = hier_max,
        macro_min = macro_min,
        macro_max = macro_max,
        lda_min = lda_min,
        lda_max = lda_max,
        VaR_min = VaR_min,
        VaR_max = VaR_max,
        oas_min = oas_min,
        oas_max = oas_max,
        block_min = block_min,
        block_max = block_max,
        glasso_min = glasso_min,
        glasso_max = glasso_max,
        s_ewma_regime_min = s_ewma_regime_min,
        s_ewma_regime_max = s_ewma_regime_max,
        fx_min = fx_min,
        fx_max = fx_max,
        fund_min = fund_min,
        fund_max = fund_max,
    )

    w_hat, w_map, C_wk = _solve_shrinkage_weights(
        S = S,
        T_ref = T_ref,
        names = names,
        mats = mats,
        n_factor_directions = n_factor_directions,
        alpha_factor_tracking = alpha_factor_tracking,
        tau_logdet = tau_logdet,
        logdet_eps = logdet_eps,
        alpha_trace = alpha_trace,
        s_min = s_min,
        s_max = s_max,
        w_S = w_S,
        bounds_kwargs = bounds_kwargs,
        solve_method = solve_method,
    )

    C_ann = _apply_term_structure(
        C_wk = C_wk,
        idx = idx,
        periods_per_year = periods_per_year,
        use_factor_term_structure = use_factor_term_structure,
        weekly_5y = weekly_5y_idx,
        ff_factors_weekly = ff_factors_weekly,
        index_returns_weekly = index_returns_weekly,
        industry_returns_weekly = industry_returns_weekly,
        sector_returns_weekly = sector_returns_weekly,
        macro_factors_weekly = macro_factors_weekly,
        names = names,
        S1 = S1,
        S3 = S3,
        S5 = S5,
        horizon_weeks = horizon_weeks,
        gamma_idio = gamma_idio,
        r2_cap = r2_cap,
        factor_share_min = factor_share_min,
        factor_share_max = factor_share_max,
    )

    out = pd.DataFrame(C_ann, index = idx, columns = idx)
 
    if description:
        
        desc = out.describe()
        
        eigvals = np.linalg.eigvalsh(C_ann)
        
        diag_vol = np.sqrt(np.clip(np.diag(C_ann), 0, None))
        
        corr_vol = np.corrcoef(diag_vol, comb_std_aligned.loc[idx].to_numpy())[0, 1]
        
        extras = {
            "weights": w_map,
            "eig_min": float(eigvals[0]),
            "eig_max": float(eigvals[-1]),
            "eig_trace": float(eigvals.sum()),
            "vol_forecast_corr": float(corr_vol),
        }
        
        return out, desc, extras
    
    else:
    
        return out
