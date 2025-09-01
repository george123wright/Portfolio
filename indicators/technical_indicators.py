"""
Technical signal calibration, backtesting, and composite scoring from OHLCV data

Overview
--------
This module loads end-of-day OHLCV data from an Excel workbook, calibrates a
suite of classic technical analysis indicators by backtesting threshold grids,
and produces a per-ticker, per-date composite integer score that reflects the
weighted agreement of active buy/sell signals. The composite scores are written
back to the same workbook on a sheet named "Signal Scores". Optional
calibration results (best parameters, performance statistics, and assigned
weights per indicator) are persisted to a separate calibration workbook.

Data interface
--------------
Input workbook sheets (wide format, DateTime index; columns are tickers):
  - "High":   session highs
  - "Low":    session lows
  - "Close":  closing prices
  - "Volume": trading volume

Output workbook sheet:
  - "Signal Scores": a Date column followed by one column per ticker with an
    integer score clipped to a symmetric range.

All datetimes written to Excel are converted to timezone-naive UTC. Index order
is enforced as ascending by time.

Indicators and formulas
-----------------------
For each ticker, the following indicators are precomputed and later used to
generate entry and exit pulses. Where applicable, parameters are searched over
grids and tuned by in-sample backtesting with simple performance statistics.

1) Relative Strength Index (RSI)
   - RSI_t = 100 − 100 / (1 + RS_t), where RS_t is an exponentially smoothed
     ratio of average gains to average losses over a fixed window.
 
   - Signals: cross up of RSI above a buy threshold and cross down below a sell
     threshold.

2) Money Flow Index (MFI)
   - Typical price tp_t = (High_t + Low_t + Close_t) / 3.
  
   - Money flow mf_t = tp_t × Volume_t.
   
   - Positive and negative money flow are aggregated over a lookback depending
     on the sign of tp_t − tp_{t−1}.
   
   - MFI_t = 100 − 100 / (1 + positive_flow_sum / negative_flow_sum).
   
   - Signals: cross up of MFI above a buy threshold and cross down below a sell
     threshold.

3) Bollinger Bands (BB)
   - Mid_t = simple moving average of Close over a fixed window.
  
   - Std_t = rolling standard deviation of Close over the same window (ddof=0).
   
   - Upper_t = Mid_t + n_std × Std_t
   
   - Lower_t = Mid_t − n_std × Std_t.
   
   - Signals: price crossing above Lower_t (buy) and below Upper_t (sell).

4) Average True Range (ATR) breakouts
   - True range TR_t = max(High_t − Low_t, |High_t − Close_{t−1}|, |Low_t − Close_{t−1}|).
   
   - ATR_t = simple moving average of TR over a fixed window.
   
   - Rolling highs/lows of High/Low over a break window are shifted by one bar.
   
   - Up threshold_t = prior_window_high_t + m × ATR_{t−1}.
   
   - Down threshold_t = prior_window_low_t  − m × ATR_{t−1}.
   
   - Signals: Close_t > Up threshold_t (buy) and Close_t < Down threshold_t (sell).

5) Stochastic Oscillator (%K, %D)
   - %K_t = 100 × (Close_t − rolling_min(Low)) / (rolling_max(High) − rolling_min(Low)).
   
   - %D_t = simple moving average of %K over a short window.
   
   - Signals: 
   
        (%K < lower_bound and %K crosses above %D) is buy
        
        (%K > upper_bound and %K crosses below %D) is sell.

6) Exponential Moving Average (EMA) crossover
   - EMA is computed with the standard recursive formula with span s:
     
         EMA_t = EMA_{t−1} + alpha × (Close_t − EMA_{t−1}), 
         
    where alpha = 2 / (s + 1).
   
   - Signals: fast EMA crossing above slow EMA (buy) and below (sell).

7) MACD line vs signal
   - MACD line = EMA_fast(Close) − EMA_slow(Close).
   
   - Signal line = EMA of MACD line over a short window.
   
   - Signals: MACD line crossing above signal (buy) and below (sell).

8) Rolling VWAP crossover
   - Typical price tp_t = (High_t + Low_t + Close_t) / 3.
   
   - Rolling VWAP_t = sum(tp_i × Volume_i) / sum(Volume_i) over a fixed window.
   
   - Signals: Close crossing above VWAP (buy) and below VWAP (sell).

9) On-Balance Volume (OBV) divergence
   - Direction_t = sign(Close_t − Close_{t−1}).
   
   - OBV_t = cumulative sum of Direction_t × Volume_t.
   
   - Divergence conditions compare rolling highs/lows in price and OBV:
     buy if price makes a lower low while OBV does not; sell if price makes a
     higher high while OBV does not.

Trend and volume gating
-----------------------
Signals can be conditioned on trend and liquidity filters:

- Trend filter via ADX (Average Directional Index):
  
  * Directional movement is computed with the standard +DM and −DM rules.
  
  * +DI_t = 100 × EMA(+DM) / ATR; −DI_t = 100 × EMA(−DM) / ATR.
  
  * DX_t  = 100 × |+DI_t − −DI_t| / (+DI_t + −DI_t); ADX_t = EMA(DX).
  
  * A pro-trend gate requires ADX_t ≥ ADX_ENTRY; an anti-trend gate requires
    ADX_t < ADX_ENTRY.

- Volume confirmation:
  
  * A boolean mask flags bars with volume at or above a rolling median over a
    confirmation window.

Signal pulses, position state, and conflict handling
----------------------------------------------------
For any threshold θ, "cross up" at time t means value_t > θ and value_{t−1} ≤ θ,
and "cross down" means value_t < θ and value_{t−1} ≥ θ. Buy and sell pulses are
constructed as boolean time series (or matrices when evaluating parameter
grids). Pulses are converted into a position_t ∈ {−1, 0, +1} by a small state
machine that:

- applies an optional signal lag L bars,

- enforces a cooldown of K bars after any position change, and

- resolves simultaneous buy/sell pulses using a configurable policy:
  "sell wins", "buy wins", or "mutual exclude" (drop both).

Backtest returns alignment and statistics
-----------------------------------------
To avoid look-ahead bias when evaluating a position time series pos_t against
simple returns r_t = Close_t / Close_{t−1} − 1:

- effective position is pos_{t−1}, effective return is r_t;
- the first bar is discarded; returns with NaN are ignored.

For each scenario (parameter setting), the realised trade series is
X_t = pos_{t−1} × r_t.

Given n valid bars:

- Mean return: mean_X = (1 / n) × sum(X_t).

- Sample standard deviation: 

    sd_X = sqrt( (1 / (n − 1)) × sum( (X_t − mean_X)^2 ) ).

- Annualised Sharpe ratio: 

    Sharpe = sqrt(A) × mean_X / sd_X, where A is the
  annualisation constant (e.g., 252).

- One-sample t-statistic for mean > 0: 

    t = mean_X / (sd_X / sqrt(n)).

- One-sided p-value under Student-t with df = n − 1: p = survival_function(t, df).

If sd_X is zero or no valid returns exist, the scenario is penalised
(Sharpe = −infinity; t = 0; p = 1) to fall to the bottom of rankings.

Calibration and model selection
-------------------------------
For indicators with tunable parameters (e.g., RSI buy/sell thresholds, MFI
thresholds, Bollinger standard deviation multiplier, ATR breakout multiplier,
stochastic bounds), the module performs grid search:

1) Optional coarse screen on a small grid.

2) If the best coarse scenario passes a significance gate defined by
   p < alpha and Sharpe ≥ min_sharpe, refinement proceeds on either:

   - the full fine grid; or
   
   - a local neighbourhood around the coarse optimum.

3) The best scenario among significant candidates is chosen; if none are
   significant, the best overall Sharpe is chosen.

Each indicator receives a discrete weight in {0, 1, 2, 3, 4} based on
(significance, Sharpe level). The mapping thresholds are configurable.

Composite score construction
----------------------------
For a ticker and date t, each active indicator contributes weight × position_t,
with position_t ∈ {−1, 0, +1}. The composite integer score is the sum across
indicators, clipped to a symmetric cap to avoid extreme values. This score is a
heuristic proxy for consensus across momentum, reversal, breakout, and
volume-confirmed signals after trend gating and conflict resolution.

Persistence and caching
-----------------------
- Calibration results (best parameters, weights, Sharpe, t-stat, p-value, trade
  count, data start/end, last updated timestamp) are upserted to a calibration
  workbook. A JSON blob stores indicator parameters per ticker.

- If calibrations are available, they are reused to compute scores; otherwise
  the module calibrates on the fly.

Computation and implementation notes
------------------------------------
- Vectorised grid evaluation is used extensively by broadcasting time × grid
  arrays to build buy/sell pulse matrices in one pass per indicator.

- Numba is used to JIT-compile the pulse-to-position state machine for speed.

- Cross detection is implemented as elementwise comparisons between a series
  and its one-period lag.

- The ATR breakout uses ATR_{t−1} and prior window highs/lows (shifted) to
  avoid contemporaneous dependence.

- ADX and volume gates are applied multiplicatively to suppress pulses.

- All performance statistics are computed on aligned series to avoid look-ahead.

- Excel I/O uses openpyxl; conditional formatting highlights positive scores in
  green and negative scores in red.

Configuration
-------------
Key parameters are set at module scope:

- Significance gate: SIGNIFICANCE_ALPHA (p-value), MIN_SHARPE_FOR_SIG.

- Coarse-to-fine search toggles and grids.

- Indicator default thresholds, windows, and spans.

- Conflict resolution mode, signal lag, and cooldown.

- Annualisation constant.

- Per-indicator default weights (used if no calibration is present).

Assumptions and limitations
---------------------------
- The framework uses simple, in-sample grid search with basic statistics; no
  walk-forward validation, cross-validation, or transaction costs are applied.

- Signals are daily and assume end-of-day execution with lag L; intraday effects,
  slippage, and spreads are ignored.

- ADX/volume gates and window choices are heuristic; their impact is data- and
  asset-dependent.

- Multiple-testing inflation can occur when scanning large grids; the
  significance gate is a coarse filter rather than a formal correction.

Typical workflow
----------------
1) Read OHLCV sheets from `config.DATA_FILE`.

2) For each ticker:

   - Use cached calibrations if present; otherwise calibrate indicators.
   
   - Build per-indicator positions and aggregate to a composite score series.

3) Optionally upsert calibration rows to the calibration workbook.

4) Write the "Signal Scores" sheet to the data workbook.

The `main()` function orchestrates the above steps and logs progress,
skipped tickers, and file writes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
from numba import njit
import ta
import datetime as dt

from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule
from openpyxl import load_workbook, Workbook

import json
import config


UPDATE_PARAM: bool = False

CONFLICT_MODE: str = "sell_wins"  

SIGNIFICANCE_ALPHA = 0.05
MIN_SHARPE_FOR_SIG = 0.30

USE_COARSE_SCREEN: bool = True           
REFINE_STRATEGY: str = "full"           
LOCAL_WIDTH: int = 1                   

SCREEN_ALPHA: float = SIGNIFICANCE_ALPHA    
SCREEN_MIN_SHARPE: float = MIN_SHARPE_FOR_SIG

COARSE_GRIDS = {
    "rsi": {
        "buy": [25, 30, 35], 
        "sell": [65, 75, 85]
    },
    "mfi": {
        "buy": [20, 25, 30], 
        "sell": [75, 80, 85]
    },
    "bb": {
        "num_std": [1.5, 2.0, 2.5]
    },
    "atr": {
        "mult": [1.25, 1.5, 2.0, 2.5]
    },
    "stoch": {
        "lower": [15, 20, 25], 
        "upper": [75, 80, 85]
    },
}

try:

    CALIBRATION_FILE = Path(config.CALIBRATION_FILE)

except Exception:

    CALIBRATION_FILE = Path("indicator_calibration.xlsx")

CALIB_SHEET = "Calibration"

ANNUALISATION = 252

RSI_BUY_GRID = list(np.arange(1, 49, 0.5))
RSI_SELL_GRID = list(np.arange(51, 100, 0.5))

MFI_BUY_GRID = list(np.arange(1, 49, 0.5))
MFI_SELL_GRID = list(np.arange(51, 100, 0.5))

BB_STD_GRID = list(np.arange(0.75, 3, 0.05))

ATR_MULT_GRID = list(np.arange(0.75, 3, 0.05))

STOCH_LOW_GRID = list(np.arange(1, 49, 0.5))
STOCH_HIGH_GRID = list(np.arange(51, 100, 0.5))

MODE_SELL_WINS, MODE_BUY_WINS, MODE_MUTUAL = 0, 1, 2

_MODE_MAP = {
    "sell_wins": MODE_SELL_WINS, 
    "buy_wins": MODE_BUY_WINS, 
    "mutual_exclude": MODE_MUTUAL
}

DEFAULT_WEIGHTS = {
    "macd": 1,
    "ema": 1, 
    "atr": 1,
    "rsi": 1, 
    "bb": 1,
    "stoch": 1, 
    "obv": 1,
    "mfi": 1, 
    "vwap": 1
}

PRO_TREND = {"macd", "ema", "atr", "vwap", "obv"}
ANTI_TREND = {"rsi", "bb", "stoch"}
NEUTRAL_TREND = {"mfi"}
VOL_REQUIRED = {"atr", "vwap", "obv", "macd", "ema"}

INDICATOR_ORDER = ["macd", "ema", "atr", "rsi", "bb", "stoch", "obv", "mfi", "vwap"]

EMA_FAST: int = 12
EMA_SLOW: int = 26

BB_WINDOW: int = 20

STOCH_K: int = 14
STOCH_D: int = 3

ATR_WINDOW: int = 14
ATR_BREAK_WINDOW: int = 20
ATR_MULTIPLIER: float = 1.5  

OBV_LOOKBACK: int = 20

ADX_WINDOW: int = 14
ADX_ENTRY: float = 25.0
ADX_EXIT: float = 20.0 

RSI_WINDOW: int = 14
RSI_BUY_THRESH: float = 30.0
RSI_SELL_THRESH: float = 70.0

VWAP_WINDOW: int = 20
MFI_WINDOW: int = 14
MFI_BUY_THRESH: float = 20.0
MFI_SELL_THRESH: float = 80.0

SCORE_CLAMP: int = 10

SIGNAL_LAG: int = 0
COOLDOWN_BARS: int = 0


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if not logger.handlers:

    _h = logging.StreamHandler()

    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(_h)


def _col(
    df: pd.DataFrame, 
    tk: str, 
    name: str
) -> pd.Series:
    """
    Return a single ticker column from a sheet-like DataFrame with validation.

    Parameters
    ----------
    df : pd.DataFrame
        A wide DataFrame whose columns are ticker symbols.
    tk : str
        Ticker symbol to extract.
    name : str
        Human-readable sheet name used for error messages.

    Returns
    -------
    pd.Series
        The column df[tk].

    Raises
    ------
    KeyError
        If the ticker is not present in df.columns.

    Notes
    -----
    This is a convenience guard to produce informative errors when the expected
    ticker is missing in the input Excel sheet that was read into a DataFrame.
    """

    if tk not in df.columns:
    
        raise KeyError(f"{name} sheet does not contain ticker '{tk}'")
    
    return df[tk]


def read_sheet(
    excel_file: str | Path, 
    sheet: str
) -> pd.DataFrame:
    """
    Read a single Excel sheet as a DateTime-indexed DataFrame, sorted by index.

    Parameters
    ----------
    excel_file : str | Path
        Path to the workbook.
    sheet : str
        Sheet name to read.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime (parsed with `parse_dates=True`).

    Raises
    ------
    ValueError
        If the sheet’s index is not datetime-typed.
    Exception
        Any read errors are logged and re-raised.

    Notes
    -----
    The function enforces a datetime index because subsequent indicator and
    backtest logic assumes time-ordered data. The frame is sorted ascending
    by index to guarantee time order.
    """

    try:

        df = pd.read_excel(excel_file, sheet_name=sheet, index_col = 0, header = 0, parse_dates = True)

    except Exception as exc:

        logger.exception("Parsing %s!%s failed (%s)", excel_file, sheet, type(exc).__name__)

        raise

    if not pd.api.types.is_datetime64_any_dtype(df.index):

        raise ValueError(f"Sheet '{sheet}' has non-datetime index")

    return df.sort_index()


def load_data(
    excel_file: str | Path
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data from an Excel workbook into a dict of DataFrames.

    Parameters
    ----------
    excel_file : str | Path
        Path to workbook containing sheets: "High", "Low", "Close", "Volume".

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys: "high", "low", "close", "volume". Values are DateTime-indexed frames.

    Notes
    -----
    This function calls `read_sheet` for each sheet and provides a consistent
    mapping of logical names → sheet names expected downstream.
    """

    mapping = {
        "high": "High",
        "low": "Low", 
        "close": "Close", 
        "volume": "Volume"
    }
    
    out: Dict[str, pd.DataFrame] = {}
    
    for key, sheet in mapping.items():
    
        out[key] = read_sheet(
            excel_file = excel_file,
            sheet = sheet
        )
    
    return out


def _excel_safe_datetimes(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Return a copy with timezone-aware datetimes converted to tz-naive UTC.

    Parameters
    ----------
    df : pd.DataFrame
        Arbitrary DataFrame that may contain datetime-like columns or objects.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with:
        - tz-aware pandas datetime columns converted to UTC then made tz-naive.
        - object columns that contain datetime/timestamp objects coerced similarly.

    Why
    ---
    Excel (via openpyxl) cannot store tz-aware datetimes. This function ensures
    all datetimes are timezone-naive (interpreted as UTC) to avoid
    `ValueError: Excel does not support datetimes with timezones`.

    Details
    -------
    For pandas datetime columns with dtype `DatetimeTZDtype`, we apply:
        ts_utc_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    For object columns, each element `x` is mapped:
    - if `x` is pd.Timestamp with tz: `x.tz_convert("UTC").tz_localize(None)`
    - if `x` is datetime with tzinfo: `x.astimezone(UTC).replace(tzinfo=None)`
    - otherwise left unchanged.
    """
   
    out = df.copy()
   
    DatetimeTZDtype = getattr(pd, "DatetimeTZDtype", None)
   
    for col in out.columns:
   
        s = out[col]
   
        dtype = s.dtype
   
        is_tzaware = (
            (DatetimeTZDtype is not None and isinstance(dtype, DatetimeTZDtype))
            or getattr(dtype, "tz", None) is not None
        )
       
        if is_tzaware:
       
            out[col] = s.dt.tz_convert("UTC").dt.tz_localize(None)
       
        elif pd.api.types.is_datetime64_dtype(dtype):
       
            pass
       
        elif pd.api.types.is_object_dtype(dtype):
       
       
            def _strip_tz(
                x
            ):
       
                if isinstance(x, pd.Timestamp):
            
                    return x.tz_convert("UTC").tz_localize(None) if x.tz is not None else x
            
                if isinstance(x, dt.datetime):
            
                    return x.astimezone(dt.timezone.utc).replace(tzinfo=None) if x.tzinfo is not None else x
            
                return x
            
            
            out[col] = s.map(_strip_tz)
   
    return out


def load_calibration_df() -> pd.DataFrame:
    """
    Load the calibration table from the calibration workbook (if present).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        ["Ticker","Indicator","ParamJSON","Weight","Sharpe","TStat","PValue",
        "NTrades","Start","End","LastUpdated"].
        Empty frame if file/sheet is missing or upon read errors.

    Notes
    -----
    The sheet name is given by the module constant `CALIB_SHEET`.
    """

    cols = ["Ticker", "Indicator", "ParamJSON", "Weight", "Sharpe", "TStat", "PValue", "NTrades", "Start", "End", "LastUpdated"]
   
    if not CALIBRATION_FILE.exists():
   
        return pd.DataFrame(columns = cols)

    try:

        df = pd.read_excel(CALIBRATION_FILE, sheet_name = CALIB_SHEET, engine = "openpyxl")

        return df
  
    except Exception:
  
        return pd.DataFrame(columns = cols)


def save_calibration_df(
    df: pd.DataFrame
) -> None:
    """
    Write the calibration DataFrame into the calibration workbook.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration data. Columns will be reindexed to the canonical set.

    Notes
    -----
    All datetime columns are passed through `_excel_safe_datetimes` before write.
    The file is overwritten (mode="w") each time.
    """

    cols = ["Ticker", "Indicator", "ParamJSON", "Weight", "Sharpe", "TStat", "PValue", "NTrades", "Start", "End", "LastUpdated"]

    df = df.reindex(columns = cols)

    df = _excel_safe_datetimes(df)

    with pd.ExcelWriter(CALIBRATION_FILE, engine = "openpyxl", mode = "w", datetime_format = "yyyy-mm-dd hh:mm:ss") as w:
      
        df.to_excel(w, sheet_name = CALIB_SHEET, index = False)


def _is_sig(
    sharpe: float,
    p_value: float, 
    min_sharpe: float,
    alpha: float
) -> bool:
    """
    Return True if (Sharpe, p-value) qualifies as statistically significant.

    Parameters
    ----------
    sharpe : float
        Annualised Sharpe ratio estimate.
    p_value : float
        One-sided p-value for H1: mean > 0.
    min_sharpe : float
        Minimum Sharpe threshold.
    alpha : float
        Significance level.

    Returns
    -------
    bool
        True iff (p_value < alpha) and (sharpe >= min_sharpe).
    """

    return (p_value < alpha) and (sharpe >= min_sharpe)

def _neighbor_window(
    full_grid: List[float] | List[int], 
    center: float, 
    width: int = 1
) -> List[float]:
    """
    Return an index-based neighborhood window around a given grid center.

    Parameters
    ----------
    full_grid : list[int|float]
        The full, ordered grid (e.g., RSI thresholds).
    center : float
        Center value to locate in `full_grid`. If not exactly present, the
        closest element by absolute difference is used.
    width : int, default 1
        Number of neighbors to include on each side.

    Returns
    -------
    list[float]
        Slice of `full_grid` with up to ±width neighbors around center,
        preserving order and de-duplicating.
    """

    
    arr = list(full_grid)
    
    try:
    
        i = arr.index(center)  
   
    except ValueError:

        i = int(np.argmin([abs(x - center) for x in arr]))
  
    lo = max(0, i - width)
  
    hi = min(len(arr), i + width + 1)
  
    return arr[lo: hi]


def get_cached_for_ticker(
    tk: str
) -> dict[str, dict] | None:
    """
    Read and parse cached calibration entries for a given ticker.

    Parameters
    ----------
    tk : str
        Ticker symbol.

    Returns
    -------
    dict[str, dict] | None
        A mapping `indicator_name -> {params, weight, sharpe, t_stat, p_value, n_trades}`,
        lower-cased by indicator name. Returns None if no calibration for this ticker.

    Notes
    -----
    `ParamJSON` is decoded with `json.loads`; invalid JSON yields `{}`.
    """

    df = load_calibration_df()

    if df.empty:

        return None

    sub = df[df["Ticker"] == tk]

    if sub.empty:

        return None

    out: dict[str, dict] = {}

    for _, row in sub.iterrows():

        raw = row.get("ParamJSON")

        s = "" if pd.isna(raw) else str(raw)

        try:

            params = json.loads(s) if s else {}

        except Exception:

            params = {}

        out[str(row["Indicator"]).lower()] = {
            "params": params,
            "weight": int(row.get("Weight", 0) or 0),
            "sharpe": float(row.get("Sharpe", 0.0) or 0.0),
            "t_stat": float(row.get("TStat", 0.0) or 0.0),
            "p_value": float(row.get("PValue", 1.0) or 1.0),
            "n_trades": int(row.get("NTrades", 0) or 0),
        }
        
    return out


def write_calibration_rows(
    rows: list[dict]
) -> None:
    """
    Merge new calibration rows into the calibration sheet (upsert by Ticker+Indicator).

    Parameters
    ----------
    rows : list[dict]
        New rows to upsert. Keys must match calibration columns.

    Behavior
    --------
    1) Load current calibration frame (if any).
    2) Replace any existing rows with the same (Ticker, Indicator) keys.
    3) Append new rows for pairs that did not exist.
    4) Save back via `save_calibration_df`.

    Notes
    -----
    Datetimes are normalised to tz-naive via `_excel_safe_datetimes` before save.
    """

    old = load_calibration_df()

    if not old.empty:

        old = _excel_safe_datetimes(
            df = old
        )

    new = pd.DataFrame(rows)
   
    cols = ["Ticker", "Indicator", "ParamJSON", "Weight", "Sharpe", "TStat", "PValue", "NTrades", "Start", "End", "LastUpdated"]
   
    new = new.reindex(columns = cols)

    if old.empty:
  
        df = new
  
    else:
        old = old.reindex(columns = cols)
  
        key = ["Ticker", "Indicator"]
       
        old_idx = old.set_index(key).index
       
        new_idx = new.set_index(key).index
       
        keep_old_mask = ~old_idx.isin(new_idx)

        pieces = []
       
        old_keep = old.loc[keep_old_mask]
       
        if not old_keep.empty:
       
            pieces.append(old_keep.reset_index(drop = True))
       
        pieces.append(new.reset_index(drop = True))
       
        df = pd.concat(pieces, ignore_index = True)

    save_calibration_df(
        df = df
    )


def save_scores(
    excel_file: str | Path, 
    scores: pd.DataFrame
) -> None:
    """
    Write per-date signal scores to the workbook with conditional formatting.

    Parameters
    ----------
    excel_file : str | Path
        Target workbook.
    scores : pd.DataFrame
        DateTime-indexed DataFrame; columns are tickers; values are integer scores.

    Behavior
    --------
    - Creates/replaces a sheet named "Signal Scores".
    - Writes a "Date" column followed by one column per ticker.
    - Applies openpyxl conditional formatting:
        > 0 shaded green, < 0 shaded red.

    Notes
    -----
    Datetime index entries are converted to UTC tz-naive `datetime` objects for Excel.
    """

    green = PatternFill("solid", start_color = "90EE90", end_color = "90EE90")
    red = PatternFill("solid", start_color = "FFC7CE", end_color = "FFC7CE")

    excel_file = Path(excel_file)
  
    if excel_file.exists():
  
        wb = load_workbook(filename = excel_file)
   
    else:
   
        wb = Workbook()

    if "Signal Scores" in wb.sheetnames:
   
        ws = wb["Signal Scores"]
   
        wb.remove(ws)

    ws = wb.create_sheet("Signal Scores")
   
    cols = ["Date"] + scores.columns.tolist()
   
    for j, col in enumerate(cols, 1):
   
        ws.cell(row = 1, column = j, value = col)

    for i, (day, row) in enumerate(scores.iterrows(), 2):
    
        day_ts = pd.Timestamp(day)
    
        if day_ts.tz is not None:
    
            day_ts = day_ts.tz_convert("UTC").tz_localize(None)
    
        day_py = day_ts.to_pydatetime()
    
        ws.cell(row = i, column = 1, value = day_py)
      
        for j, val in enumerate(row, 2):
      
            ws.cell(row = i, column = j, value = int(val) if pd.notna(val) else None)

    rng = f"B2:{get_column_letter(ws.max_column)}{ws.max_row}"
    
    ws.conditional_formatting.add(rng, CellIsRule(operator = "greaterThan", formula = ["0"], fill = green))
    ws.conditional_formatting.add(rng, CellIsRule(operator = "lessThan", formula = ["0"], fill = red))

    if "Sheet" in wb.sheetnames and len(wb.sheetnames) > 1:
     
        try:
     
            default_ws = wb["Sheet"]
     
            if default_ws.max_row == 1 and default_ws.max_column == 1 and default_ws["A1"].value is None:
     
                wb.remove(default_ws)
     
        except Exception:
     
            pass

    wb.save(excel_file)


@dataclass
class IndicatorResult:
    """
    Container for the best indicator configuration and performance stats.

    Fields
    ------
    name : str
        Indicator identifier (e.g., "rsi").
    params : dict
        Chosen parameterisation (e.g., {"buy_thresh": 25.0, "sell_thresh": 75.0}).
    sharpe : float
        Annualised Sharpe (see `_eval_stats_from_pos` for the formula).
    t_stat : float
        One-sample t-statistic for mean daily return > 0.
    p_value : float
        One-sided p-value under Student-t with df = N-1.
    weight : int
        Discrete weight derived from `weight_from_stats` (0..4).
    n_trades : int
        Estimated number of flips (position changes) during the backtest.
    """

    name: str
  
    params: dict
  
    sharpe: float
  
    t_stat: float
  
    p_value: float
  
    weight: int
  
    n_trades: int


def _mode_id(
    mode: str
) -> int:
    """
    Map a human string conflict mode to an integer id for the Numba kernel.

    Parameters
    ----------
    mode : {"sell_wins","buy_wins","mutual_exclude"}
        Conflict resolution policy when buy and sell pulses occur simultaneously.

    Returns
    -------
    int
        0 for "sell_wins", 1 for "buy_wins", 2 for "mutual_exclude".

    Raises
    ------
    ValueError
        For unknown mode strings.
    """

    try:

        return _MODE_MAP[mode]

    except KeyError:

        raise ValueError(f"Unknown CONFLICT_MODE: {mode}")


def crosses_over(
    x: np.ndarray, 
    y: np.ndarray
) -> np.ndarray:
    """
    Vectorised "cross above" detector between two arrays aligned along time.

    Parameters
    ----------
    x : np.ndarray, shape (T, P) or (T, 1)
        Left-hand time series (or matrix of series).
    y : np.ndarray, shape (T, P) or (T, 1)
        Right-hand comparator.

    Returns
    -------
    np.ndarray of bool, shape broadcast from x and y
        True at t iff: x_t > y_t and x_{t-1} <= y_{t-1}. The t=0 row is False.

    Math
    ----
    For each column p,
        cross_up_t = 1{ x_t > y_t } * 1{ x_{t-1} <= y_{t-1} },  for t >= 1
    We treat t=0 as no cross (False). NaN comparisons follow NumPy rules (any
    comparison with NaN is False), which naturally suppresses signals on
    insufficient lookback.
    """


    x_prev = np.roll(x, 1, axis=0)
    
    x_prev[0, ...] = np.nan
    y_prev = np.roll(y, 1, axis=0)
    
    y_prev[0, ...] = np.nan
    
    return (x > y) & (x_prev <= y_prev)


def crosses_under(
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Vectorised "cross below" detector between two arrays aligned along time.

    Same as `crosses_over`, but detects:
        cross_down_t = 1{ x_t < y_t } * 1{ x_{t-1} >= y_{t-1} }, t >= 1.
    """

    x_prev = np.roll(x, 1, axis=0)
    
    x_prev[0, ...] = np.nan

    y_prev = np.roll(y, 1, axis=0)
    
    y_prev[0, ...] = np.nan

    return (x < y) & (x_prev >= y_prev)


@njit(cache = True)
def pulses_to_positions(
    buys, 
    sells, 
    signal_lag, 
    cooldown_bars, 
    mode_id
):
    """
    Convert buy/sell pulses into persistent positions with lag, cooldown, and conflict resolution.

    Parameters
    ----------
    buys, sells : np.ndarray of bool, shape (T, P)
        Pulse matrices: True indicates a buy (or sell) signal at time t for scenario p.
    signal_lag : int
        Apply a delay of `signal_lag` bars before a pulse can affect position.
    cooldown_bars : int
        After a pulse flips the position, suppress further pulses for this many bars.
    mode_id : int
        0 = sell_wins, 1 = buy_wins, 2 = mutual_exclude (drop both).

    Returns
    -------
    np.ndarray of int8, shape (T, P)
        Position ∈ {-1, 0, +1} per time/scenario. Position persists until flipped.

    Algorithm
    ---------
    At each t and scenario p:
    1) Read effective pulses at (t - signal_lag) if in range; else False.
    2) If cooldown[p] > 0, ignore pulses and decrement cooldown.
    3) Resolve simultaneous pulses according to `mode_id`.
    4) Update position:
    - buy → +1 and start cooldown
    - sell → -1 and start cooldown
    - else carry forward previous position

    Notes
    -----
    The kernel is Numba-compiled for speed. It expects C-contiguous boolean
    arrays for best performance.
    """

    T, P = buys.shape

    pos = np.zeros((T, P), np.int8)

    cool = np.zeros(P, np.int32)

    for t in range(T):
      
        for p in range(P):
      
            tt = t - signal_lag
      
            b = False; s = False
      
            if tt >= 0:
      
                b = bool(buys[tt, p])
                
                s = bool(sells[tt, p])
      
            if cool[p] > 0:
      
                b = False
                
                s = False
      
                cool[p] -= 1
      
            if b and s:
      
                if mode_id == 0: 
             
                    b = False
             
                elif mode_id == 1: 
             
                    s = False
             
                else:             
             
                    b = False
                    
                    s = False

            if t == 0:

                prev = 0 
            
            else:
                
                prev = pos[t - 1, p]
       
            if b:
       
                pos[t, p] = 1
                
                cool[p] = cooldown_bars
          
            elif s:
                
                pos[t, p] = -1
                
                cool[p] = cooldown_bars
           
            else:
           
                pos[t, p] = prev
   
    return pos


def _eval_stats_from_pos(
    pos: np.ndarray,
    r: np.ndarray,
    annualisation: float
):
    """
    Compute performance statistics (Sharpe, t-stat, p-value, N) from positions and returns.

    Parameters
    ----------
    pos : np.ndarray, shape (T, P), int8
        Position matrix over time (per scenario).
    r : np.ndarray, shape (T,), float
        Simple returns r_t = (C_t / C_{t-1}) - 1.
    annualisation : float
        Trading days per year (e.g., 252).

    Returns
    -------
    (sharpe, t_stat, p_val, n) : tuple
        sharpe : np.ndarray, shape (P,)
            Annualised Sharpe for each scenario p.
        t_stat : np.ndarray, shape (P,)
            One-sample t-statistic for daily mean > 0.
        p_val : np.ndarray, shape (P,)
            One-sided p-values under Student-t with df = n-1.
        n : int
            Number of valid daily returns used.

    Math
    ----
    We align positions and returns to avoid look-ahead:
    - Use `pos_eff = pos[:-1, p]` and `r_eff = r[1:]`.
    - Mask out NaN in r_eff.

    For scenario p:
        X_t = pos_eff_t * r_eff_t

    Mean and std:
        \bar{X} = (1/n) * Σ X_t
        s = sqrt( (1/(n-1)) * Σ (X_t - \bar{X})^2 )

    Annualised Sharpe:
        Sharpe = sqrt(A) * (\bar{X} / s)
    where A = `annualisation`.

    t-stat:
        t = \bar{X} / (s / sqrt(n))

    p-value (one-sided, H1: mean > 0):
        p = sf(max(t, 0), df = n-1)
    using `scipy.stats.t.sf`.

    Edge Cases
    ----------
    If no valid returns, Sharpe = -inf (to rank last), t = 0, p = 1.
    """

    valid = ~np.isnan(r)

    pos_eff = pos[:-1, :]

    r_eff = r[1:]

    valid_eff = valid[1:]

    if not valid_eff.any():

        P = pos.shape[1]

        return (np.full(P, -np.inf), np.zeros(P), np.ones(P), 0)

    pos_eff = pos_eff[valid_eff, :].astype(np.float64)

    r_eff = r_eff[valid_eff]

    ret_mat = pos_eff * r_eff[:, None]

    mean = np.nanmean(ret_mat, axis=0)

    std = np.nanstd(ret_mat, axis =0, ddof=1)

    n = ret_mat.shape[0]
    
    with np.errstate(divide = "ignore", invalid = "ignore"):
       
        sharpe = np.where(std > 0, np.sqrt(annualisation) * mean / std, -np.inf)
       
        t_stat = np.where(std > 0, mean / (std / np.sqrt(max(n, 1))), 0.0)
   
    df = max(n - 1, 1)
   
    p_val = stats.t.sf(np.maximum(t_stat, 0.0), df = df)
   
    return sharpe, t_stat, p_val, n


def _best_index(
    sharpe: np.ndarray, 
    t_stat: np.ndarray, 
    p_val: np.ndarray,
    min_sharpe: float, 
    alpha: float
) -> int:
    """
    Pick the best scenario index subject to significance gating.

    Parameters
    ----------
    sharpe, t_stat, p_val : np.ndarray, shape (P,)
        Scenario statistics.
    min_sharpe : float
        Minimum Sharpe to be considered significant.
    alpha : float
        p-value threshold.

    Returns
    -------
    int
        Index j that maximises Sharpe among significant scenarios if any exist;
        otherwise the global Sharpe argmax.

    Notes
    -----
    If any `sig = (p_val < alpha) & (sharpe >= min_sharpe)` is True, the argmax
    is restricted to those; else unrestricted.
    """
   
    sig = (p_val < alpha) & (sharpe >= min_sharpe)
    
    if sig.any():
   
        cand = np.where(sig, sharpe, -np.inf)  
    
    else:
        
        cand = sharpe
   
    return int(np.nanargmax(cand))


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    Compute the True Range (TR) series.

    Parameters
    ----------
    high, low, close : pd.Series
        Price series indexed by time.

    Returns
    -------
    pd.Series
        True range TR_t = max( 
            high_t - low_t,
            |high_t - close_{t-1}|,
            |low_t  - close_{t-1}|
        ).

    Notes
    -----
    This is the classic Wilder True Range used in ATR/ADX.
    """

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis = 1
    ).max(axis = 1)
    
    return tr


def atr_series(
    high: pd.Series,
    low: pd.Series, 
    close: pd.Series,
    window: int
) -> pd.Series:
    """
    Average True Range (ATR) via simple rolling mean of TR.

    Parameters
    ----------
    high, low, close : pd.Series
    window : int
        Length of the rolling mean.

    Returns
    -------
    pd.Series
        ATR_t = SMA_{window}( TR_t ).

    Notes
    -----
    This implementation uses a simple moving average (SMA) of TR. Some variants
    use Wilder’s RMA/EMA. Consistency matters more than the exact smoother.
    """

    tr = true_range(
        high = high,
        low = low, 
        close = close
    )

    return tr.rolling(window, min_periods = window).mean()


def adx_series(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    *, 
    window: int = ADX_WINDOW
) -> pd.Series:
    """
    Average Directional Index (ADX) following the classic DI+/DI- construction.

    Parameters
    ----------
    high, low, close : pd.Series
    window : int, default ADX_WINDOW
        Smoothing window for EMA used in DI and ADX.

    Returns
    -------
    pd.Series
        ADX_t in [0, 100].

    Math
    ----
    Let:
        TR_t        = True Range (see `true_range`)
        α           = 1 / window
        ATR_t       = EMA(TR_t; α)
        up_move_t   = max(high_t - high_{t-1}, 0)
        down_move_t = max(low_{t-1} - low_t, 0)

    Directional movement (Wilder):
        +DM_t = up_move_t   if up_move_t > down_move_t else 0
        -DM_t = down_move_t if down_move_t > up_move_t else 0

    Directional indicators (scaled to %):
        +DI_t = 100 * EMA(+DM_t; α) / ATR_t
        -DI_t = 100 * EMA(-DM_t; α) / ATR_t

    DX and ADX:
        DX_t  = 100 * |(+DI_t - -DI_t)| / (+DI_t + -DI_t)
        ADX_t = EMA(DX_t; α)

    Implementation Details
    ----------------------
    - Uses pandas `ewm(alpha=α, adjust=False)` for smoothing.
    - Divisions by zero are avoided via `replace(0, np.nan)`.
    """

    tr = true_range(high, low, close)
    
    alpha = 1.0 / window

    atr = tr.ewm(alpha = alpha, adjust=False).mean().replace(0, np.nan)

    up_move = (high.diff()).clip(lower = 0.0)

    down_move = (-low.diff()).clip(lower = 0.0)

    plus_dm = up_move.where(up_move > down_move, 0.0)

    minus_dm = down_move.where(down_move > up_move, 0.0)

    plus_di = 100 * plus_dm.ewm(alpha = alpha, adjust = False).mean() / atr

    minus_di = 100 * minus_dm.ewm(alpha = alpha, adjust = False).mean() / atr

    denom = (plus_di + minus_di).replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / denom

    adx = dx.ewm(alpha = alpha, adjust = False).mean()

    return adx


@dataclass
class Precomp:
    """
    Packed precomputations for one ticker to avoid repeated O(T) passes.

    Fields (all np.ndarray unless noted)
    ------------------------------------
    idx : pd.DatetimeIndex
        Index used to reconstruct Series at the edges.
    c : float64[T]
        Close prices.
    r : float64[T]
        Simple returns r_t = (c_t / c_{t-1}) - 1.
    trend_mask : bool[T]
        ADX gating mask: True where ADX >= ADX_ENTRY.
    vol_ok : bool[T]
        Volume confirmation mask (rolling median filter).
    rsi : float64[T]
        Relative Strength Index (RSI) series (from `ta.momentum.RSIIndicator`).
    bb_mid : float64[T]
        Bollinger midline = SMA_{BB_WINDOW}(close).
    bb_std : float64[T]
        Rolling std dev over BB_WINDOW (ddof=0).
    atr : float64[T]
        ATR window = ATR_WINDOW (SMA of TR).
    hi_prev, lo_prev : float64[T]
        Highest high / lowest low over ATR_BREAK_WINDOW, shifted by 1 bar.
    mfi : float64[T]
        Money Flow Index (see precompute_all for exact formula).
    stoch_k, stoch_d : float64[T]
        Stochastic %K and %D (SMA over STOCH_D).
    ema_fast, ema_slow : float64[T]
        Exponential moving averages (spans EMA_FAST/EMA_SLOW).
    macd_line, macd_signal : float64[T]
        MACD line and signal from `ta.trend.MACD`.
    rvwap : float64[T]
        Rolling VWAP_{VWAP_WINDOW} = sum(tp*vol)/sum(vol), tp=(H+L+C)/3.
    obv : float64[T]
        On-Balance Volume (signed cumulative volume).
    price_low_prev, price_high_prev : float64[T]
        Rolling min/max of close over OBV_LOOKBACK, shifted by 1 bar.
    obv_low_prev, obv_high_prev : float64[T]
        Rolling min/max of OBV over OBV_LOOKBACK, shifted by 1 bar.

    Notes
    -----
    All arrays are float64 or bool for consistent broadcasting in vectorised
    grid evaluators.
    """

    idx: pd.DatetimeIndex

    c: np.ndarray

    r: np.ndarray

    trend_mask: np.ndarray

    vol_ok: np.ndarray

    rsi: np.ndarray

    bb_mid: np.ndarray

    bb_std: np.ndarray

    atr: np.ndarray

    hi_prev: np.ndarray

    lo_prev: np.ndarray

    mfi: np.ndarray

    stoch_k: np.ndarray

    stoch_d: np.ndarray

    ema_fast: np.ndarray

    ema_slow: np.ndarray

    macd_line: np.ndarray

    macd_signal: np.ndarray

    rvwap: np.ndarray

    obv: np.ndarray

    price_low_prev: np.ndarray

    price_high_prev: np.ndarray

    obv_low_prev: np.ndarray

    obv_high_prev: np.ndarray


def precompute_all(
    h: pd.Series, 
    l: pd.Series, 
    c: pd.Series, 
    v: pd.Series
) -> Precomp:
    """
    Compute all indicator primitives once for a ticker and pack into `Precomp`.

    Parameters
    ----------
    h, l, c, v : pd.Series
        High, Low, Close, Volume series (aligned, DateTime-indexed).

    Returns
    -------
    Precomp
        See `Precomp` docstring for field definitions.

    Formulas (key)
    --------------
    RSI (via library):
        RSI_t = 100 - 100 / (1 + RS_t),  RS_t = EMA(Gains)/EMA(Losses)

    Bollinger:
        mid_t = SMA_{BB_WINDOW}(c_t)
        std_t = StdDev_{BB_WINDOW}(c_t; ddof=0)

    ATR:
        ATR_t = SMA_{ATR_WINDOW}( TR_t )

    ATR breakout thresholds:
        hi_prev_t = max(High_{t-ATR_BREAK_WINDOW..t-1})
        lo_prev_t = min(Low_{t-ATR_BREAK_WINDOW..t-1})

    MFI:
        tp_t = (H_t + L_t + C_t)/3
        mf_t = tp_t * Vol_t
        up_t = mf_t if tp_t - tp_{t-1} > 0 else 0
        dn_t = mf_t if tp_t - tp_{t-1} < 0 else 0
        MFI_t = 100 - 100 / (1 + sum(up) / sum(dn)) over MFI_WINDOW

    Stochastic:
        %K_t = 100 * (C_t - min(L)) / (max(H) - min(L)) over STOCH_K
        %D_t = SMA_{STOCH_D}(%K_t)

    EMA:
        EMA_t(span=s) = EMA_{t-1} + α*(C_t - EMA_{t-1}), α = 2/(s+1)

    MACD:
        MACD_line = EMA(C, EMA_FAST) - EMA(C, EMA_SLOW)
        Signal    = EMA(MACD_line, 9)

    Rolling VWAP:
        RVWAP_t = Σ_{i=t-W+1..t} (tp_i * vol_i) / Σ vol_i,  tp_i=(H_i+L_i+C_i)/3

    OBV:
        direction_t = sign(C_t - C_{t-1})
        OBV_t = Σ (direction_i * vol_i)
    """

    idx = c.index

    c_np = c.to_numpy(np.float64)

    r = c.pct_change().to_numpy(np.float64)

    adx = adx_series(
        high = h, 
        low = l, 
        close = c,
        window = ADX_WINDOW
    )

    trend_mask = (adx >= ADX_ENTRY).fillna(False).to_numpy(np.bool_)
   
    vol_ok = (v >= v.rolling(VOL_CONFIRM_WIN := 20, min_periods = 20).median()).fillna(False).to_numpy(np.bool_)

    rsi = ta.momentum.RSIIndicator(c, window = RSI_WINDOW).rsi().to_numpy(np.float64)

    mid = c.rolling(BB_WINDOW, min_periods = BB_WINDOW).mean()

    sd = c.rolling(BB_WINDOW, min_periods = BB_WINDOW).std(ddof = 0)
    
    bb_mid = mid.to_numpy(np.float64)
    
    bb_std = sd.to_numpy(np.float64)

    atr = atr_series(
        high = h,
        low = l, 
        close = c,
        window = ATR_WINDOW
    ).to_numpy(np.float64)
    
    hi_prev = h.rolling(ATR_BREAK_WINDOW, min_periods = ATR_BREAK_WINDOW).max().shift(1).to_numpy(np.float64)
   
    lo_prev = l.rolling(ATR_BREAK_WINDOW, min_periods = ATR_BREAK_WINDOW).min().shift(1).to_numpy(np.float64)

    tp = (h + l + c) / 3.0
    
    mf = tp * v
    
    up = mf.where(tp.diff() > 0, 0.0)
    
    dn = mf.where(tp.diff() < 0, 0.0)
   
    up_sum = up.rolling(MFI_WINDOW, min_periods = MFI_WINDOW).sum()
    
    dn_sum = dn.rolling(MFI_WINDOW, min_periods = MFI_WINDOW).sum().replace(0, np.nan)
    
    ratio = (up_sum / dn_sum).replace([np.inf, -np.inf], np.nan)
    
    mfi = (100 - (100 / (1 + ratio))).to_numpy(np.float64)

    low_k = l.rolling(STOCH_K, min_periods=STOCH_K).min()

    high_k = h.rolling(STOCH_K, min_periods=STOCH_K).max()

    den = (high_k - low_k).replace(0, np.nan)

    pct_k = 100 * (c - low_k) / den

    stoch_k = pct_k.replace([np.inf, -np.inf], np.nan).to_numpy(np.float64)

    stoch_d = pct_k.rolling(STOCH_D, min_periods = STOCH_D).mean().to_numpy(np.float64)

    ema_fast = c.ewm(span = EMA_FAST, adjust = False).mean().to_numpy(np.float64)

    ema_slow = c.ewm(span = EMA_SLOW, adjust = False).mean().to_numpy(np.float64)

    mac = ta.trend.MACD(c, window_fast = EMA_FAST, window_slow = EMA_SLOW, window_sign = 9)

    macd_line = mac.macd().to_numpy(np.float64)

    macd_signal = mac.macd_signal().to_numpy(np.float64)

    rvwap = ((tp * v).rolling(VWAP_WINDOW, min_periods = VWAP_WINDOW).sum() / v.rolling(VWAP_WINDOW, min_periods = VWAP_WINDOW).sum().replace(0, np.nan)).to_numpy(np.float64)

    direction = np.sign(c.diff()).fillna(0.0)

    obv = (direction * v).cumsum().to_numpy(np.float64)

    price_low_prev = c.rolling(OBV_LOOKBACK, min_periods=OBV_LOOKBACK).min().shift(1).to_numpy(np.float64)

    price_high_prev = c.rolling(OBV_LOOKBACK, min_periods=OBV_LOOKBACK).max().shift(1).to_numpy(np.float64)

    obv_low_prev = pd.Series(obv, index = idx).rolling(OBV_LOOKBACK, min_periods = OBV_LOOKBACK).min().shift(1).to_numpy(np.float64)

    obv_high_prev = pd.Series(obv, index = idx).rolling(OBV_LOOKBACK, min_periods = OBV_LOOKBACK).max().shift(1).to_numpy(np.float64)

    return Precomp(
        idx = idx,
        c = c_np,
        r = r, 
        trend_mask = trend_mask, 
        vol_ok = vol_ok,
        rsi = rsi,
        bb_mid = bb_mid,
        bb_std = bb_std,
        atr = atr, 
        hi_prev = hi_prev, 
        lo_prev = lo_prev,
        mfi = mfi,
        stoch_k = stoch_k, 
        stoch_d = stoch_d,
        ema_fast = ema_fast,
        ema_slow = ema_slow, 
        macd_line = macd_line,
        macd_signal = macd_signal,
        rvwap = rvwap,
        obv = obv,
        price_low_prev = price_low_prev,
        price_high_prev = price_high_prev,
        obv_low_prev = obv_low_prev,
        obv_high_prev = obv_high_prev
    )


def eval_bb_grid(
    pc: Precomp,
    num_std_grid: np.ndarray,
    is_pro_trend: bool, 
    is_anti_trend: bool, 
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
    ):
    """
    Evaluate Bollinger Band cross signals over a grid of num_std values (vectorised).

    Parameters
    ----------
    pc : Precomp
        Precomputed series container.
    num_std_grid : np.ndarray[float], shape (P,)
        Candidate standard deviation multipliers.
    is_pro_trend, is_anti_trend, vol_required : bool
        Gating flags for ADX and volume masks.
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        {
        "num_std": float, "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int
        } for the best grid point (per `_best_index` policy).

    Signals
    -------
    With mid = pc.bb_mid, sd = pc.bb_std, nσ in grid:
        upper_t(nσ) = mid_t + nσ * sd_t
        lower_t(nσ) = mid_t - nσ * sd_t

    Buy pulse:  C_t crosses over lower_t
    Sell pulse: C_t crosses under upper_t

    All scenarios are built in one matrix using broadcasting: T × P.
    Positions are produced by `pulses_to_positions`, then performance by
    `_eval_stats_from_pos`.
    """

    ns = num_std_grid.astype(np.float64) 
   
    upper = pc.bb_mid[:, None] + pc.bb_std[:, None] * ns[None, :]
   
    lower = pc.bb_mid[:, None] - pc.bb_std[:, None] * ns[None, :]
   
    c2 = pc.c[:, None]
   
    buy_mat = crosses_over(
        x = c2, 
        y = lower
    )
   
    sell_mat = crosses_under(
        x = c2, 
        y = upper
    )
    
    gate = np.ones(pc.c.shape[0], dtype = np.bool_)
    if is_pro_trend:
        
        gate &= pc.trend_mask
    
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
    
    if vol_required:    
        
        gate &= pc.vol_ok
    
    if not gate.all():
        
        buy_mat &= gate[:, None]
       
        sell_mat &= gate[:, None]
   
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag, 
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
    
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos, 
        r = pc.r, 
        annualisation = ANNUALISATION
    )
    
    j = _best_index(
        sharpe = sharpe, 
        t_stat = t_stat,
        p_val = p_val, 
        min_sharpe = MIN_SHARPE_FOR_SIG,
        alpha = SIGNIFICANCE_ALPHA
    )
    
    flips = int(np.sum(np.abs(np.diff(pos[:, j].astype(np.int16))) > 0))
    
    return {
        "num_std": float(ns[j]), 
        "sharpe": float(sharpe[j]), 
        "t_stat": float(t_stat[j]),
        "p_value": float(p_val[j]),
        "n": int(n),
        "n_trades": flips
    }


def eval_atr_grid(
    pc: Precomp, 
    mult_grid: np.ndarray,
    is_pro_trend: bool, 
    is_anti_trend: bool, 
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate ATR breakout signals for a multiplier grid (vectorised).

    Parameters
    ----------
    pc : Precomp
    mult_grid : np.ndarray[float], shape (P,)
        ATR multipliers m.
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        {
        "mult": float, "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int
        }

    Signals
    -------
    Let ATR_{t-1} be previous ATR and (hi_prev, lo_prev) be rolling extremes
    shifted by one bar:
        up_th_t(m) = hi_prev_t + m * ATR_{t-1}
        dn_th_t(m) = lo_prev_t - m * ATR_{t-1}

    Buy pulse:  C_t > up_th_t(m)
    Sell pulse: C_t < dn_th_t(m)
    """

    M = mult_grid.astype(np.float64)  

    atr_prev = np.roll(pc.atr, 1)
    
    atr_prev[0] = np.nan
    
    up_th = pc.hi_prev[:, None] + atr_prev[:, None] * M[None, :]
    
    dn_th = pc.lo_prev[:, None] - atr_prev[:, None] * M[None, :]
    
    c2 = pc.c[:, None]
    
    buy_mat = c2 > up_th
    
    sell_mat = c2 < dn_th
    
    gate = np.ones(pc.c.shape[0], dtype = np.bool_)
    
    if is_pro_trend:
        
        gate &= pc.trend_mask
    
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
    
    if vol_required:    
    
        gate &= pc.vol_ok
    
    if not gate.all():
    
        buy_mat &= gate[:, None]
    
        sell_mat &= gate[:, None]
    
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag,
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
    
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos, 
        r = pc.r, 
        annualisation = ANNUALISATION
    )
    
    j = _best_index(
        sharpe = sharpe, 
        t_stat = t_stat,
        p_val = p_val, 
        min_sharpe = MIN_SHARPE_FOR_SIG,
        alpha = SIGNIFICANCE_ALPHA
    )
    
    flips = int(np.sum(np.abs(np.diff(pos[:, j].astype(np.int16))) > 0))
    
    return {
        "mult": float(M[j]), 
        "sharpe": float(sharpe[j]),
        "t_stat": float(t_stat[j]),
        "p_value": float(p_val[j]),
        "n": int(n), 
        "n_trades": flips
    }


def eval_mfi_grid(
    pc: Precomp,
    buy_grid: np.ndarray,
    sell_grid: np.ndarray,
    is_pro_trend: bool, 
    is_anti_trend: bool, 
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG,
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate MFI threshold cross signals over (buy_grid × sell_grid) (vectorised).

    Parameters
    ----------
    pc : Precomp
    buy_grid, sell_grid : np.ndarray[float]
        Candidate thresholds B (enter) and S (exit).
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        {
        "buy_thresh": float, "sell_thresh": float,
        "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int
        }

    Signals
    -------
    Let MFI_t be `pc.mfi`, with previous MFI_{t-1}. For threshold θ:
        CrossUp_t(θ)   = 1{MFI_t > θ} * 1{MFI_{t-1} <= θ}
        CrossDown_t(θ) = 1{MFI_t < θ} * 1{MFI_{t-1} >= θ}

    Buy pulse uses θ=B, sell pulse uses θ=S. The full cartesian grid is formed
    by repeating/tiling the column vectors so we evaluate all (B,S) pairs at once.
    """
    
    mfi = pc.mfi.astype(np.float64)
    
    mfi_prev = np.roll(mfi, 1)
    
    mfi_prev[0] = np.nan
    
    B = buy_grid.astype(np.float64)  
    
    S = sell_grid.astype(np.float64) 
    
    buy_nb = (mfi[:, None] > B[None, :]) & (mfi_prev[:, None] <= B[None, :])
  
    sell_ns = (mfi[:, None] < S[None, :]) & (mfi_prev[:, None] >= S[None, :])
    
    Nb, Ns = buy_nb.shape[1], sell_ns.shape[1]
    
    buy_mat = np.repeat(buy_nb, Ns, axis = 1)
    
    sell_mat = np.tile(sell_ns, (1, Nb))
    
    gate = np.ones(mfi.shape[0], dtype = np.bool_)
    
    if is_pro_trend:  
        
        gate &= pc.trend_mask
        
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
        
    if vol_required:   
        
        gate &= pc.vol_ok
        
    if not gate.all():
        
        buy_mat &= gate[:, None]
      
        sell_mat &= gate[:, None]
   
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag, 
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
    
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos,
        r = pc.r, 
        annualisation = ANNUALISATION
    )
    
    j = _best_index(
        sharpe = sharpe, 
        t_stat = t_stat,
        p_val = p_val,
        min_sharpe = MIN_SHARPE_FOR_SIG,
        alpha = SIGNIFICANCE_ALPHA
    )
    
    bi = j // Ns
    
    si = j % Ns
    
    flips = int(np.sum(np.abs(np.diff(pos[:, j].astype(np.int16))) > 0))
    
    return {
        "buy_thresh": float(B[bi]), 
        "sell_thresh": float(S[si]),
        "sharpe": float(sharpe[j]), 
        "t_stat": float(t_stat[j]),
        "p_value": float(p_val[j]), 
        "n": int(n), 
        "n_trades": flips
    }


def eval_stoch_grid(
    pc: Precomp, 
    lower_grid: np.ndarray, 
    upper_grid: np.ndarray,
    is_pro_trend: bool, 
    is_anti_trend: bool, 
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate stochastic %K/%D cross signals conditioned by bounds (vectorised).

    Parameters
    ----------
    pc : Precomp
    lower_grid, upper_grid : np.ndarray[float]
        Candidate lower/upper bounds for %K conditioning.
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        {
        "lower": float, "upper": float,
        "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int
        }

    Signals
    -------
    Let K_t = %K and D_t = %D:
        CrossUp_t = 1{K crosses above D} = 1{K_t > D_t} 1{K_{t-1} <= D_{t-1}}
        CrossDn_t = 1{K crosses below D} = 1{K_t < D_t} 1{K_{t-1} >= D_{t-1}}

    Buy pulse:  (K_t < L) & CrossUp_t
    Sell pulse: (K_t > U) & CrossDn_t
    The cartesian grid over (L,U) is evaluated in one broadcasted matrix.
    """
    
    k = pc.stoch_k.astype(np.float64)[:, None]
    
    d = pc.stoch_d.astype(np.float64)[:, None]
    
    cross_up = crosses_over(
        x = k,
        y = d
    )    
    
    cross_down = crosses_under(
        x = k, 
        y = d
    )   
    
    L = lower_grid.astype(np.float64)    
   
    U = upper_grid.astype(np.float64)    
   
    buy_l = (k < L[None, :]) & cross_up    
   
    sell_u = (k > U[None, :]) & cross_down   
   
    Nl = buy_l.shape[1]
    
    Nu = sell_u.shape[1]
   
    buy_mat = np.repeat(buy_l, Nu, axis = 1)
    
    sell_mat = np.tile(sell_u, (1, Nl))
    
    gate = np.ones(k.shape[0], dtype = np.bool_)
    
    if is_pro_trend:  
        
        gate &= pc.trend_mask
    
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
        
    if vol_required:   
        
        gate &= pc.vol_ok
        
    if not gate.all():
       
        buy_mat &= gate[:, None]
       
        sell_mat &= gate[:, None]
    
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag, 
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
    
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos, 
        r = pc.r, 
        annualisation = ANNUALISATION
    )
    
    j = _best_index(
        sharpe = sharpe, 
        t_stat = t_stat,
        p_val = p_val,
        min_sharpe = MIN_SHARPE_FOR_SIG, 
        alpha = SIGNIFICANCE_ALPHA
    )
    
    li = j // Nu
    
    ui = j % Nu
    
    flips = int(np.sum(np.abs(np.diff(pos[:, j].astype(np.int16))) > 0))
    
    return {
        "lower": float(L[li]), 
        "upper": float(U[ui]),
        "sharpe": float(sharpe[j]), 
        "t_stat": float(t_stat[j]),
        "p_value": float(p_val[j]),
        "n": int(n), 
        "n_trades": flips
    }


def eval_rsi_grid(
    pc: Precomp,
    buy_grid: np.ndarray, 
    sell_grid: np.ndarray,
    is_pro_trend: bool, 
    is_anti_trend: bool, 
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate RSI threshold cross signals over (buy_grid × sell_grid) (vectorised).

    Parameters
    ----------
    pc : Precomp
    buy_grid, sell_grid : np.ndarray[float]
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        {
        "buy_thresh": float, "sell_thresh": float,
        "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int
        }

    Signals
    -------
    Let R_t be RSI and R_{t-1} previous:
        CrossUp_t(θ)   = 1{R_t > θ} * 1{R_{t-1} <= θ}
        CrossDown_t(θ) = 1{R_t < θ} * 1{R_{t-1} >= θ}

    Buy uses θ=B, sell uses θ=S. All pairs (B,S) are evaluated simultaneously.
    """

    rsi = pc.rsi.astype(np.float64)

    rsi_prev = np.roll(rsi, 1)
    
    rsi_prev[0] = np.nan

    B = buy_grid.astype(np.float64)  
   
    S = sell_grid.astype(np.float64) 
    
    buy_nb = (rsi[:, None] > B[None, :]) & (rsi_prev[:, None] <= B[None, :])
   
    sell_ns = (rsi[:, None] < S[None, :]) & (rsi_prev[:, None] >= S[None, :])
   
    Nb = buy_nb.shape[1]
    
    Ns = sell_ns.shape[1]
   
    buy_mat = np.repeat(buy_nb, Ns, axis = 1)
   
    sell_mat = np.tile(sell_ns, (1, Nb))
   
    gate = np.ones(rsi.shape[0], dtype = np.bool_)
    
    if is_pro_trend:  
        
        gate &= pc.trend_mask
    
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
    
    if vol_required:    
        
        gate &= pc.vol_ok
   
    if not gate.all():
      
        buy_mat &= gate[:, None]
      
        sell_mat &= gate[:, None]
    
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag,
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
    
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos,
        r = pc.r,
        annualisation = ANNUALISATION
    )
    
    j = _best_index(
        sharpe = sharpe, 
        t_stat = t_stat, 
        p_val = p_val,
        min_sharpe = MIN_SHARPE_FOR_SIG, 
        alpha = SIGNIFICANCE_ALPHA
    )
   
    bi = j // Ns
    
    si = j % Ns
    
    flips = int(np.sum(np.abs(np.diff(pos[:, j].astype(np.int16))) > 0))
   
    return {
        "buy_thresh": float(B[bi]),
        "sell_thresh": float(S[si]),
        "sharpe": float(sharpe[j]),
        "t_stat": float(t_stat[j]),
        "p_value": float(p_val[j]), 
        "n": int(n), 
        "n_trades": flips
    }


def eval_ema_single(
    pc: Precomp,
    is_pro_trend: bool, 
    is_anti_trend: bool,
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate EMA crossover strategy (fast vs slow) as a single scenario.

    Parameters
    ----------
    pc : Precomp
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        { "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int }

    Signals
    -------
    Buy pulse:  EMA_fast crosses above EMA_slow.
    Sell pulse: EMA_fast crosses below EMA_slow.
    """

    ema_f = pc.ema_fast[:, None]

    ema_s = pc.ema_slow[:, None]

    buy_mat = crosses_over(
        x = ema_f, 
        y = ema_s
    )

    sell_mat = crosses_under(
        x = ema_f, 
        y = ema_s
    )
   
    gate = np.ones(pc.c.shape[0], dtype = np.bool_)
   
    if is_pro_trend:  
        
        gate &= pc.trend_mask
    
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
   
    if vol_required:   
       
        gate &= pc.vol_ok
    
    if not gate.all():
       
        buy_mat &= gate[:, None]
       
        sell_mat &= gate[:, None]
   
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag, 
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
   
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos,
        r = pc.r, 
        annualisation = ANNUALISATION
    )
   
    flips = int(np.sum(np.abs(np.diff(pos[:, 0].astype(np.int16))) > 0))
   
    return {
        "sharpe": float(sharpe[0]), 
        "t_stat": float(t_stat[0]), 
        "p_value": float(p_val[0]),
        "n": int(n),
        "n_trades": flips
    }


def eval_macd_single(
    pc: Precomp,
    is_pro_trend: bool,
    is_anti_trend: bool,
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate MACD line vs signal crossover as a single scenario.

    Parameters
    ----------
    pc : Precomp
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        { "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int }

    Signals
    -------
    Buy pulse:  MACD_line crosses above MACD_signal.
    Sell pulse: MACD_line crosses below MACD_signal.
    """

    line = pc.macd_line[:, None]

    sig = pc.macd_signal[:, None]

    buy_mat = crosses_over(
        x = line, 
        y = sig
    )

    sell_mat = crosses_under(
        x = line, 
        y = sig
    )

    gate = np.ones(pc.c.shape[0], dtype = np.bool_)
    
    if is_pro_trend:   
        
        gate &= pc.trend_mask
    
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
    
    if vol_required:   
        
        gate &= pc.vol_ok
   
    if not gate.all():
        
        buy_mat &= gate[:, None]
        
        sell_mat &= gate[:, None]
   
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag,
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
   
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos, 
        r = pc.r,
        annualisation = ANNUALISATION
    )
   
    flips = int(np.sum(np.abs(np.diff(pos[:, 0].astype(np.int16))) > 0))
   
    return {
        "sharpe": float(sharpe[0]), 
        "t_stat": float(t_stat[0]),
        "p_value": float(p_val[0]),
        "n": int(n), 
        "n_trades": flips
    }


def eval_vwap_single(
    pc: Precomp,
    is_pro_trend: bool, 
    is_anti_trend: bool, 
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate rolling VWAP reversion/trend crossover as a single scenario.

    Parameters
    ----------
    pc : Precomp
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        { "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int }

    Signals
    -------
    Let RVWAP_t be rolling VWAP:
        Buy pulse:  C_t crosses above RVWAP_t
        Sell pulse: C_t crosses below RVWAP_t
    """

    c2 = pc.c[:, None]

    vw = pc.rvwap[:, None]

    buy_mat = crosses_over(
        x = c2, 
        y = vw
    )

    sell_mat = crosses_under(
        x = c2, 
        y = vw
    )
    
    gate = np.ones(pc.c.shape[0], dtype = np.bool_)
    
    if is_pro_trend:  
        
        gate &= pc.trend_mask
   
    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask
   
    if vol_required:    
        
        gate &= pc.vol_ok
   
    if not gate.all():
       
        buy_mat &= gate[:, None]
       
        sell_mat &= gate[:, None]
    
    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat,
        signal_lag = signal_lag, 
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
    
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos, 
        r = pc.r, 
        annualisation = ANNUALISATION
    )
    
    flips = int(np.sum(np.abs(np.diff(pos[:, 0].astype(np.int16))) > 0))
    
    return {
        "sharpe": float(sharpe[0]), 
        "t_stat": float(t_stat[0]), 
        "p_value": float(p_val[0]),
        "n": int(n), 
        "n_trades": flips
    }


def eval_obv_single(
    pc: Precomp,
    is_pro_trend: bool, 
    is_anti_trend: bool, 
    vol_required: bool,
    signal_lag: int = SIGNAL_LAG, 
    cooldown: int = COOLDOWN_BARS,
    conflict_mode: str = CONFLICT_MODE
):
    """
    Evaluate OBV divergence style pulses as a single scenario.

    Parameters
    ----------
    pc : Precomp
    is_pro_trend, is_anti_trend, vol_required : bool
    signal_lag : int
    cooldown : int
    conflict_mode : str

    Returns
    -------
    dict
        { "sharpe": float, "t_stat": float, "p_value": float,
        "n": int, "n_trades": int }

    Signals
    -------
    Bullish divergence (buy pulse):
        - Price makes a lower low vs prior window: C_t < price_low_prev_t
        - OBV does NOT make a lower low:         OBV_t >= obv_low_prev_t

    Bearish divergence (sell pulse):
        - Price makes higher high:                C_t > price_high_prev_t
        - OBV does NOT make higher high:          OBV_t <= obv_high_prev_t

    These conditions are evaluated pointwise without explicit crossings.
    """

    c2 = pc.c[:, None]

    price_low = pc.price_low_prev[:, None]

    price_high = pc.price_high_prev[:, None]
   
    obv2 = pc.obv[:, None]
   
    obv_low = pc.obv_low_prev[:, None]
   
    obv_high = pc.obv_high_prev[:, None]

    buy_mat = (c2 < price_low) & (obv2 >= obv_low)

    sell_mat = (c2 > price_high) & (obv2 <= obv_high)

    gate = np.ones(pc.c.shape[0], dtype=np.bool_)

    if is_pro_trend:   
        
        gate &= pc.trend_mask

    elif is_anti_trend: 
        
        gate &= ~pc.trend_mask

    if vol_required:    
        
        gate &= pc.vol_ok

    if not gate.all():
        
        buy_mat &= gate[:, None]
       
        sell_mat &= gate[:, None]

    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = signal_lag,
        cooldown_bars = cooldown, 
        mode_id = _mode_id(
            mode = conflict_mode
        )
    )
    
    sharpe, t_stat, p_val, n = _eval_stats_from_pos(
        pos = pos, 
        r = pc.r, 
        annualisation = ANNUALISATION
    )
    
    flips = int(np.sum(np.abs(np.diff(pos[:, 0].astype(np.int16))) > 0))
    
    return {
        "sharpe": float(sharpe[0]),
        "t_stat": float(t_stat[0]),
        "p_value": float(p_val[0]),
        "n": int(n), 
        "n_trades": flips
    }


def _coarse_then_full_rsi(
    pc: Precomp,
    is_pro: bool, 
    is_anti: bool, 
    vol_req: bool
):
    """
    Two-stage tuning: coarse screen then refine (full or local neighborhood).

    Parameters
    ----------
    pc : Precomp
    is_pro, is_anti, vol_req : bool
        Gating flags.
    Returns
    -------
    (best_dict, passed) : tuple
        best_dict : dict of the same shape returned by the corresponding `eval_*`.
        passed    : bool indicating whether the coarse screen met significance
                    (`SCREEN_MIN_SHARPE`, `SCREEN_ALPHA`) and thus refinement ran.

    Strategy
    --------
    1) Evaluate a small, cheap COARSE_GRIDS set.
    2) If best coarse result is not significant, return it with passed=False.
    3) Else, refine:
    - if REFINE_STRATEGY == "full": run the full grid;
    - if "local": restrict to ±LOCAL_WIDTH neighbors around the coarse best.
    """

    g = COARSE_GRIDS["rsi"]

    coarse = eval_rsi_grid(
        pc = pc,
        buy_grid = np.array(g["buy"], dtype = float),
        sell_grid = np.array(g["sell"], dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti,
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS,
        conflict_mode = CONFLICT_MODE
    )
    
    if not _is_sig(
        sharpe = coarse["sharpe"], 
        p_value = coarse["p_value"], 
        min_sharpe = SCREEN_MIN_SHARPE, 
        alpha = SCREEN_ALPHA):
       
        return coarse, False  

    if REFINE_STRATEGY == "local":
       
        buy_grid = _neighbor_window(
            full_grid = RSI_BUY_GRID, 
            center = coarse["buy_thresh"],
            width = LOCAL_WIDTH
        )
       
        sell_grid = _neighbor_window(
            full_grid = RSI_SELL_GRID,
            center = coarse["sell_thresh"],
            width = LOCAL_WIDTH
        )
        
    else:
        
        buy_grid = RSI_BUY_GRID
        
        sell_grid = RSI_SELL_GRID

    full = eval_rsi_grid(
        pc = pc,
        buy_grid = np.array(buy_grid, dtype = float),
        sell_grid = np.array(sell_grid, dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti, 
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG,
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    return full, True


def _coarse_then_full_mfi(
    pc: Precomp, 
    is_pro: bool,
    is_anti: bool,
    vol_req: bool
):
    """
    Two-stage tuning: coarse screen then refine (full or local neighborhood).

    Parameters
    ----------
    pc : Precomp
    is_pro, is_anti, vol_req : bool
        Gating flags.
    Returns
    -------
    (best_dict, passed) : tuple
        best_dict : dict of the same shape returned by the corresponding `eval_*`.
        passed    : bool indicating whether the coarse screen met significance
                    (`SCREEN_MIN_SHARPE`, `SCREEN_ALPHA`) and thus refinement ran.

    Strategy
    --------
    1) Evaluate a small, cheap COARSE_GRIDS set.
    2) If best coarse result is not significant, return it with passed=False.
    3) Else, refine:
    - if REFINE_STRATEGY == "full": run the full grid;
    - if "local": restrict to ±LOCAL_WIDTH neighbors around the coarse best.
    """

    g = COARSE_GRIDS["mfi"]

    coarse = eval_mfi_grid(
        pc = pc,
        buy_grid = np.array(g["buy"], dtype = float),
        sell_grid = np.array(g["sell"], dtype = float),
        is_pro_trend = is_pro,
        is_anti_trend = is_anti, 
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    if not _is_sig(
        sharpe = coarse["sharpe"],
        p_value = coarse["p_value"], 
        min_sharpe = SCREEN_MIN_SHARPE, 
        alpha = SCREEN_ALPHA
    ):
    
        return coarse, False
    
    if REFINE_STRATEGY == "local":
    
        buy_grid = _neighbor_window(
            full_grid = MFI_BUY_GRID,  
            center = coarse["buy_thresh"],  
            width = LOCAL_WIDTH
        )
        
        sell_grid = _neighbor_window(
            full_grid = MFI_SELL_GRID, 
            center = coarse["sell_thresh"], 
            width = LOCAL_WIDTH)
   
    else:
   
        buy_grid = MFI_BUY_GRID
        
        sell_grid = MFI_SELL_GRID
   
    full = eval_mfi_grid(
        pc = pc,
        buy_grid = np.array(buy_grid, dtype = float),
        sell_grid = np.array(sell_grid, dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti,
        vol_required = vol_req,
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    return full, True


def _coarse_then_full_bb(
    pc: Precomp, 
    is_pro: bool,
    is_anti: bool,
    vol_req: bool
):
    """
    Two-stage tuning: coarse screen then refine (full or local neighborhood).

    Parameters
    ----------
    pc : Precomp
    is_pro, is_anti, vol_req : bool
        Gating flags.
    Returns
    -------
    (best_dict, passed) : tuple
        best_dict : dict of the same shape returned by the corresponding `eval_*`.
        passed    : bool indicating whether the coarse screen met significance
                    (`SCREEN_MIN_SHARPE`, `SCREEN_ALPHA`) and thus refinement ran.

    Strategy
    --------
    1) Evaluate a small, cheap COARSE_GRIDS set.
    2) If best coarse result is not significant, return it with passed=False.
    3) Else, refine:
    - if REFINE_STRATEGY == "full": run the full grid;
    - if "local": restrict to ±LOCAL_WIDTH neighbors around the coarse best.
    """

    g = COARSE_GRIDS["bb"]

    coarse = eval_bb_grid(
        pc = pc,
        num_std_grid = np.array(g["num_std"], dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti, 
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    if not _is_sig(
        sharpe = coarse["sharpe"], 
        p_value = coarse["p_value"], 
        min_sharpe = SCREEN_MIN_SHARPE, 
        alpha = SCREEN_ALPHA
    ):
     
        return coarse, False
   
    if REFINE_STRATEGY == "local":
   
        grid = _neighbor_window(
            full_grid = BB_STD_GRID,
            center = coarse["num_std"], 
            width = LOCAL_WIDTH
        )
   
    else:
   
        grid = BB_STD_GRID
   
    full = eval_bb_grid(
        pc = pc, 
        num_std_grid = np.array(grid, dtype = float),
        is_pro_trend = is_pro,
        is_anti_trend = is_anti, 
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    return full, True


def _coarse_then_full_atr(
    pc: Precomp, 
    is_pro: bool,
    is_anti: bool,
    vol_req: bool
):
    """
    Two-stage tuning: coarse screen then refine (full or local neighborhood).

    Parameters
    ----------
    pc : Precomp
    is_pro, is_anti, vol_req : bool
        Gating flags.
    Returns
    -------
    (best_dict, passed) : tuple
        best_dict : dict of the same shape returned by the corresponding `eval_*`.
        passed    : bool indicating whether the coarse screen met significance
                    (`SCREEN_MIN_SHARPE`, `SCREEN_ALPHA`) and thus refinement ran.

    Strategy
    --------
    1) Evaluate a small, cheap COARSE_GRIDS set.
    2) If best coarse result is not significant, return it with passed=False.
    3) Else, refine:
    - if REFINE_STRATEGY == "full": run the full grid;
    - if "local": restrict to ±LOCAL_WIDTH neighbors around the coarse best.
    """

    g = COARSE_GRIDS["atr"]

    coarse = eval_atr_grid(
        pc = pc, 
        mult_grid = np.array(g["mult"], dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti, 
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    if not _is_sig(
        sharpe = coarse["sharpe"], 
        p_value = coarse["p_value"], 
        min_sharpe = SCREEN_MIN_SHARPE, 
        alpha = SCREEN_ALPHA
    ):
    
        return coarse, False
    
    if REFINE_STRATEGY == "local":
    
        grid = _neighbor_window(
            full_grid = ATR_MULT_GRID,
            center = coarse["mult"], 
            width = LOCAL_WIDTH
        )
    
    else:
    
        grid = ATR_MULT_GRID
    
    full = eval_atr_grid(
        pc = pc, 
        multi_grid = np.array(grid, dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti,
        vol_required = vol_req,
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    return full, True


def _coarse_then_full_rsi(
    pc: Precomp,
    is_pro: bool,
    is_anti: bool,
    vol_req: bool
):
    """
    Two-stage tuning: coarse screen then refine (full or local neighborhood).

    Parameters
    ----------
    pc : Precomp
    is_pro, is_anti, vol_req : bool
        Gating flags.
    Returns
    -------
    (best_dict, passed) : tuple
        best_dict : dict of the same shape returned by the corresponding `eval_*`.
        passed    : bool indicating whether the coarse screen met significance
                    (`SCREEN_MIN_SHARPE`, `SCREEN_ALPHA`) and thus refinement ran.

    Strategy
    --------
    1) Evaluate a small, cheap COARSE_GRIDS set.
    2) If best coarse result is not significant, return it with passed=False.
    3) Else, refine:
    - if REFINE_STRATEGY == "full": run the full grid;
    - if "local": restrict to ±LOCAL_WIDTH neighbors around the coarse best.
    """

    g = COARSE_GRIDS["rsi"]

    coarse = eval_rsi_grid(
        pc = pc,
        buy_grid = np.array(g["buy"], dtype = float),
        sell_grid = np.array(g["sell"], dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti, 
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
   
    if not _is_sig(
        sharpe = coarse["sharpe"],
        p_value = coarse["p_value"],
        min_sharpe = SCREEN_MIN_SHARPE, 
        alpha = SCREEN_ALPHA
    ):
   
        return coarse, False  

    if REFINE_STRATEGY == "local":

        buy_grid = _neighbor_window(
            full_grid = RSI_BUY_GRID,  
            center = coarse["buy_thresh"], 
            width = LOCAL_WIDTH
        )

        sell_grid = _neighbor_window(
            full_grid = RSI_SELL_GRID,
            center = coarse["sell_thresh"],
            width = LOCAL_WIDTH
        )

    else:

        buy_grid = RSI_BUY_GRID
        
        sell_grid = RSI_SELL_GRID

    full = eval_rsi_grid(
        pc = pc,
        buy_grid = np.array(buy_grid, dtype = float),
        sell_grid = np.array(sell_grid, dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti, 
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG,
        cooldown = COOLDOWN_BARS,
        conflict_mode = CONFLICT_MODE
    )
    
    return full, True


def _coarse_then_full_stoch(
    pc: Precomp,
    is_pro: bool, 
    is_anti: bool,
    vol_req: bool
):
    """
    Two-stage tuning: coarse screen then refine (full or local neighborhood).

    Parameters
    ----------
    pc : Precomp
    is_pro, is_anti, vol_req : bool
        Gating flags.
    Returns
    -------
    (best_dict, passed) : tuple
        best_dict : dict of the same shape returned by the corresponding `eval_*`.
        passed    : bool indicating whether the coarse screen met significance
                    (`SCREEN_MIN_SHARPE`, `SCREEN_ALPHA`) and thus refinement ran.

    Strategy
    --------
    1) Evaluate a small, cheap COARSE_GRIDS set.
    2) If best coarse result is not significant, return it with passed=False.
    3) Else, refine:
    - if REFINE_STRATEGY == "full": run the full grid;
    - if "local": restrict to ±LOCAL_WIDTH neighbors around the coarse best.
    """

    g = COARSE_GRIDS["stoch"]

    coarse = eval_stoch_grid(
        pc = pc,
        lower_grid = np.array(g["lower"], dtype = float),
        upper_grid = np.array(g["upper"], dtype = float),
        is_pro_trend = is_pro, 
        is_anti_trend = is_anti,
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG,
        cooldown = COOLDOWN_BARS,
        conflict_mode = CONFLICT_MODE
    )
    
    if not _is_sig(
        sharpe = coarse["sharpe"],
        p_value = coarse["p_value"], 
        min_sharpe = SCREEN_MIN_SHARPE, 
        alpha = SCREEN_ALPHA
    ):
    
        return coarse, False
    
    if REFINE_STRATEGY == "local":
    
        lower_grid = _neighbor_window(
            full_grid = STOCH_LOW_GRID, 
            center = coarse["lower"], 
            width = LOCAL_WIDTH
        )
    
        upper_grid = _neighbor_window(
            full_grid = STOCH_HIGH_GRID, 
            center = coarse["upper"],
            width = LOCAL_WIDTH
        )
    
    else:
    
        lower_grid = STOCH_LOW_GRID
        
        upper_grid = STOCH_HIGH_GRID
    
    full = eval_stoch_grid(
        pc = pc,
        lower_grid = np.array(lower_grid, dtype = float),
        upper_grid = np.array(upper_grid, dtype = float),
        is_pro_trend = is_pro,
        is_anti_trend = is_anti,
        vol_required = vol_req, 
        signal_lag = SIGNAL_LAG, 
        cooldown = COOLDOWN_BARS, 
        conflict_mode = CONFLICT_MODE
    )
    
    return full, True


def weight_from_stats(
    sharpe: float, 
    t_stat: float, 
    p_value: float
) -> int:
    """
    Map (Sharpe, p-value) to a discrete weight 0..4.

    Parameters
    ----------
    sharpe : float
    t_stat : float
    p_value : float

    Returns
    -------
    int
        0 if not significant (p >= α or Sharpe < min), else:
        1 for Sharpe in [min, 0.5), 2 for [0.5, 1.0), 3 for [1.0, 1.5), 4 for ≥ 1.5.

    Notes
    -----
    Uses `SIGNIFICANCE_ALPHA` and `MIN_SHARPE_FOR_SIG` module constants.
    `t_stat` is not directly thresholded here, but included for future policies.
    """
 
    if (p_value < SIGNIFICANCE_ALPHA) and (sharpe >= MIN_SHARPE_FOR_SIG):
       
        if sharpe >= 1.5: 
            
            return 4
        
        if sharpe >= 1.0: 
            
            return 3
       
        if sharpe >= 0.5:
            
            return 2
      
        return 1
   
    return 0


def default_params_for(
    name: str
) -> dict:
    """
    Return default parameterisation for an indicator (safe fallbacks).

    Parameters
    ----------
    name : str
        Indicator name.

    Returns
    -------
    dict
        Minimal parameter dict accepted by that indicator's evaluator.

    Notes
    -----
    Defaults ensure downstream evaluators always have a valid parameter set even
    when calibration data is missing.
    """
   
    if name == "rsi":  
        
        return {
            "buy_thresh": RSI_BUY_THRESH, 
            "sell_thresh": RSI_SELL_THRESH
        }
    
    if name == "mfi":   
        
        return {
            "buy_thresh": MFI_BUY_THRESH, 
            "sell_thresh": MFI_SELL_THRESH
        }
    
    
    if name == "bb":   
        
        return {
            "num_std": BB_STD_GRID[2]
        }
    
    if name == "atr":   
        return {
            "mult": ATR_MULTIPLIER
        }
    
    if name == "stoch": 
        
        return {
            "lower": 20.0, 
            "upper": 80.0
        }
    
    return {}


def tune_thresholds_if_significant(
    name: str,
    base: IndicatorResult,
    pc: Precomp
) -> IndicatorResult:
    """
    Pick the best parameters for an indicator (coarse→refine or full grid),
    then convert to `IndicatorResult`.

    Parameters
    ----------
    name : str
        Indicator name ("rsi", "mfi", "bb", "atr", "stoch", "ema", "macd", "vwap", "obv").
    base : IndicatorResult
        Placeholder for non-significant defaults; used as a fallback return.
    pc : Precomp
        Precomputed arrays for the ticker.

    Returns
    -------
    IndicatorResult
        Best configuration and its performance stats. If coarse screening is
        enabled and fails, returns weight=0 with default params.

    Flow
    ----
    - Determine gating flags: pro-trend, anti-trend, vol-required.
    - If `USE_COARSE_SCREEN` and indicator has a grid:
        run `_coarse_then_full_*`. If not passed, return defaults with weight 0.
    - Else run the full evaluator (`eval_*_grid` or single variants).
    - Convert best dict to `IndicatorResult` and set weight via `weight_from_stats`.
    """
   
    is_pro = (name in PRO_TREND)
   
    is_anti = (name in ANTI_TREND)
   
    vol_req = (name in VOL_REQUIRED)

    if name == "rsi" and USE_COARSE_SCREEN:
    
        best, passed = _coarse_then_full_rsi(
            pc = pc, 
            is_pro = is_pro, 
            is_anti = is_anti, 
            vol_req = vol_req
        )
    
        if not passed:

            return IndicatorResult(
                name = name, 
                params = default_params_for(name = name),
                sharpe = best["sharpe"],
                t_stat = best["t_stat"], 
                p_value = best["p_value"],
                weight = 0,
                n_trades = best["n_trades"]
            )
        
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"])
        
        return IndicatorResult(
            name = name,
            params = {
                "buy_thresh": best["buy_thresh"], 
                "sell_thresh": best["sell_thresh"]
            },
            sharpe = best["sharpe"],
            t_stat = best["t_stat"], 
            p_value = best["p_value"],
            weight = w,
            n_trades = best["n_trades"]
        )

    if name == "mfi" and USE_COARSE_SCREEN:
     
        best, passed = _coarse_then_full_mfi(
            pc = pc,
            is_pro = is_pro, 
            is_anti = is_anti, 
            vol_req = vol_req
        )
     
        if not passed:
     
            return IndicatorResult(
                name = name, 
                params = default_params_for(name = name),
                sharpe = best["sharpe"], 
                t_stat = best["t_stat"], 
                p_value = best["p_value"],
                weight = 0, 
                n_trades = best["n_trades"]
            )
        
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"],
            p_value = best["p_value"]
        )
        
        return IndicatorResult(
            name = name,
            params = {
                "buy_thresh": best["buy_thresh"], 
                "sell_thresh": best["sell_thresh"]
            },
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"],
            weight = w, 
            n_trades = best["n_trades"]
        )

    if name == "bb" and USE_COARSE_SCREEN:
        
        best, passed = _coarse_then_full_bb(
            pc = pc,
            is_pro = is_pro, 
            is_anti = is_anti, 
            vol_req = vol_req
        )
        
        if not passed:
        
            return IndicatorResult(
                name = name, 
                params = default_params_for(name = name),
                sharpe = best["sharpe"],
                t_stat = best["t_stat"], 
                p_value = best["p_value"],
                weight = 0,
                n_trades = best["n_trades"]
            )
       
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"],
            p_value = best["p_value"]
        )
       
        return IndicatorResult(
            name = name,
            params = {
                "num_std": best["num_std"]
            },
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"],
            weight = w, 
            n_trades = best["n_trades"]
        )

    if name == "atr" and USE_COARSE_SCREEN:
       
        best, passed = _coarse_then_full_atr(
            pc = pc,
            is_pro = is_pro, 
            is_anti = is_anti,
            vol_req = vol_req
        )
       
        if not passed:
       
            return IndicatorResult(
                name = name, 
                params = default_params_for(name = name),
                sharpe = best["sharpe"], 
                t_stat = best["t_stat"],
                p_value = best["p_value"],
                weight = 0, 
                n_trades = best["n_trades"]
            )
       
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"],
            p_value = best["p_value"]
        )
       
        return IndicatorResult(
            name = name,
            params = {
                "mult": best["mult"]
            },
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"],
            p_value = best["p_value"],
            weight = w, 
            n_trades = best["n_trades"]
        )

    if name == "stoch" and USE_COARSE_SCREEN:
      
        best, passed = _coarse_then_full_stoch(
            pc = pc,
            is_pro = is_pro, 
            is_anti = is_anti, 
            vol_req = vol_req
        )
      
        if not passed:
      
            return IndicatorResult(
                name = name, 
                params = default_params_for(name = name),
                sharpe = best["sharpe"],
                t_stat = best["t_stat"], 
                p_value = best["p_value"],
                weight = 0,
                n_trades = best["n_trades"]
            )
        
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"]
        )
        
        return IndicatorResult(
            name = name,
            params = {
                "lower": best["lower"],
                "upper": best["upper"]
            },
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"],
            weight = w,
            n_trades = best["n_trades"]
        )

    if name == "rsi":

        best = eval_rsi_grid(
            pc = pc, 
            buy_grid = np.array(RSI_BUY_GRID, dtype = float),
            sell_grid = np.array(RSI_SELL_GRID, dtype = float),
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req, 
            signal_lag = SIGNAL_LAG, 
            cooldown = COOLDOWN_BARS,
            conflict_mode = CONFLICT_MODE
        )

        w = weight_from_stats(
            sharpe = best["sharpe"],
            t_stat = best["t_stat"],
            p_value = best["p_value"]
        )

        return IndicatorResult(
            name = name,
            params = {
                "buy_thresh": best["buy_thresh"],
                "sell_thresh": best["sell_thresh"]
            },
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"],
            weight = w,
            n_trades = best["n_trades"]
        )

    if name == "mfi":
       
        best = eval_mfi_grid(
            pc = pc, 
            buy_grid = np.array(MFI_BUY_GRID, dtype = float), 
            sell_grid = np.array(MFI_SELL_GRID, dtype = float),
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req,
            signal_lag = SIGNAL_LAG, 
            cooldown = COOLDOWN_BARS,
            conflict_mode = CONFLICT_MODE
        )
       
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"]
        )
       
        return IndicatorResult(
            name = name,
            params = {
                "buy_thresh": best["buy_thresh"], 
                "sell_thresh": best["sell_thresh"]
            },
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"],
            p_value = best["p_value"],
            weight = w, 
            n_trades = best["n_trades"]
        )

    if name == "bb":
        
        best = eval_bb_grid(
            pc = pc, 
            num_std_grid = np.array(BB_STD_GRID, dtype=float),
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req, 
            signal_lag = SIGNAL_LAG, 
            cooldown = COOLDOWN_BARS,
            conflict_mode = CONFLICT_MODE
        )
        
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"]
        )
        
        return IndicatorResult(
            name = name, 
            params = {
                "num_std": best["num_std"]
            },
            sharpe = best["sharpe"],
            t_stat = best["t_stat"],
            p_value = best["p_value"],
            weight = w, 
            n_trades = best["n_trades"]
        )

    if name == "atr":
      
        best = eval_atr_grid(
            pc = pc, 
            mult_grid = np.array(ATR_MULT_GRID, dtype = float),
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req, 
            signal_lag = SIGNAL_LAG, 
            cooldown = COOLDOWN_BARS, 
            conflict_mode = CONFLICT_MODE
        )
      
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"], 
            p_value = best["p_value"]
        )
      
        return IndicatorResult(
            name = name, 
            params = {"mult": best["mult"]},
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"],
            p_value = best["p_value"],
            weight = w,
            n_trades = best["n_trades"]
        )

    if name == "stoch":
        
        best = eval_stoch_grid(
            pc = pc,
            lower_grid = np.array(STOCH_LOW_GRID, dtype = float), 
            upper_grid = np.array(STOCH_HIGH_GRID, dtype = float),
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req, 
            signal_lag = SIGNAL_LAG,
            cooldown = COOLDOWN_BARS,
            conflict_mode = CONFLICT_MODE
        )
        
        w = weight_from_stats(
            sharpe = best["sharpe"], 
            t_stat = best["t_stat"],
            p_value = best["p_value"]
        )
      
        return IndicatorResult(
            name = name,
            params = {
                "lower": best["lower"], 
                "upper": best["upper"]
            },
            sharpe = best["sharpe"],
            t_stat = best["t_stat"], 
            p_value = best["p_value"],
            weight = w, 
            n_trades = best["n_trades"]
        )

    if name == "ema":

        res = eval_ema_single(
            pc = pc,
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti,
            vol_required = vol_req, 
            signal_lag = SIGNAL_LAG,
            cooldown = COOLDOWN_BARS, 
            conflict_mode = CONFLICT_MODE
        )

        w = weight_from_stats(
            sharpe = res["sharpe"],
            t_stat = res["t_stat"], 
            p_value = res["p_value"]
        )

        return IndicatorResult(
            name = name, 
            params = {}, 
            sharpe = res["sharpe"],
            t_stat = res["t_stat"], 
            p_value = res["p_value"],
            weight = w, 
            n_trades = res["n_trades"]
        )

    if name == "macd":
       
        res = eval_macd_single(
            pc = pc, 
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req, 
            signal_lag = SIGNAL_LAG, 
            cooldown = COOLDOWN_BARS, 
            conflict_mode = CONFLICT_MODE
        )
       
        w = weight_from_stats(
            sharpe = res["sharpe"], 
            t_stat = res["t_stat"], 
            p_value = res["p_value"]
        )
       
        return IndicatorResult(
            name = name, 
            params = {}, 
            sharpe = res["sharpe"], 
            t_stat = res["t_stat"],
            p_value = res["p_value"],
            weight = w, 
            n_trades = res["n_trades"]
        )

    if name == "vwap":
       
        res = eval_vwap_single(
            pc = pc,
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req,
            signal_lag = SIGNAL_LAG,
            cooldown = COOLDOWN_BARS, 
            conflict_mode = CONFLICT_MODE
        )
       
        w = weight_from_stats(res["sharpe"], res["t_stat"], res["p_value"])
       
        return IndicatorResult(
            name = name, 
            params = {},
            sharpe = res["sharpe"],
            t_stat = res["t_stat"],
            p_value = res["p_value"],
            weight = w,
            n_trades = res["n_trades"]
        )

    if name == "obv":
       
        res = eval_obv_single(
            pc = pc, 
            is_pro_trend = is_pro, 
            is_anti_trend = is_anti, 
            vol_required = vol_req, 
            signal_lag = SIGNAL_LAG,
            cooldown = COOLDOWN_BARS, 
            conflict_mode = CONFLICT_MODE
        )
        
        w = weight_from_stats(
            sharpe = res["sharpe"], 
            t_stat = res["t_stat"],
            p_value = res["p_value"]
        )
       
        return IndicatorResult(
            name = name, 
            params = {}, 
            sharpe = res["sharpe"],
            t_stat = res["t_stat"],
            p_value = res["p_value"],
            weight = w, 
            n_trades = res["n_trades"]
        )

    return base


def calibrate_ticker(
    h: pd.Series, 
    l: pd.Series, 
    c: pd.Series, 
    v: pd.Series, 
    tk: str
) -> tuple[dict[str, dict], list[dict]]:
    """
    Calibrate all indicators for a single ticker and produce rows for persistence.

    Parameters
    ----------
    h, l, c, v : pd.Series
        Price/volume series (aligned).
    tk : str
        Ticker symbol.

    Returns
    -------
    (results, rows) : (dict[str, dict], list[dict])
        results[name] = {"params", "weight", "sharpe", "t_stat", "p_value", "n_trades"}
        rows : calibration rows ready to append to the calibration workbook.

    Notes
    -----
    - Builds a `Precomp`, then calls `tune_thresholds_if_significant` for each
    indicator in `INDICATOR_ORDER`.
    - Timestamps Start/End are made tz-naive for Excel persistence.
    """

    start = pd.Timestamp(c.index.min())
    
    end = pd.Timestamp(c.index.max())
    
    if start.tz is not None:
    
        start = start.tz_convert("UTC").tz_localize(None)
    
    if end.tz is not None:
    
        end = end.tz_convert("UTC").tz_localize(None)
    
    now_naive = pd.Timestamp.now(tz = "UTC").tz_localize(None)

    pc = precompute_all(
        h = h,
        l = l,
        c = c,
        v = v
    )

    results: dict[str, dict] = {}
    
    rows: list[dict] = []

    for name in INDICATOR_ORDER:
       
        base = IndicatorResult(
            name = name, 
            params = default_params_for(name = name), 
            sharpe = 0.0, 
            t_stat = 0.0, 
            p_value = 1.0,
            weight = 0,
            n_trades = 0
        )
        
        best = tune_thresholds_if_significant(
            name = name,
            base = base, 
            pc = pc
        )

        results[name] = {
            "params": best.params,
            "weight": best.weight,
            "sharpe": best.sharpe,
            "t_stat": best.t_stat,
            "p_value": best.p_value,
            "n_trades": best.n_trades
        }

        rows.append({
            "Ticker": tk,
            "Indicator": name,
            "ParamJSON": json.dumps(best.params),
            "Weight": best.weight,
            "Sharpe": best.sharpe,
            "TStat": best.t_stat,
            "PValue": best.p_value,
            "NTrades": best.n_trades,
            "Start": start,
            "End": end,
            "LastUpdated": now_naive
        })

    return results, rows


def _pos_for_indicator(
    name: str, 
    params: dict,
    pc: Precomp
) -> np.ndarray:
    """
    Build a single-scenario position vector for a given indicator and params.

    Parameters
    ----------
    name : str
        Indicator name.
    params : dict
        Parameterisation chosen for this ticker/indicator.
    pc : Precomp
        Precomputed arrays.

    Returns
    -------
    np.ndarray[int8], shape (T,)
        Position series in {-1,0,+1} produced by:
        1) constructing buy/sell pulse vectors for the given params,
        2) applying gating masks (trend/anti-trend/volume),
        3) resolving conflicts with `pulses_to_positions`.

    Signals (by indicator)
    ----------------------
    BB:
        upper = mid + nσ*std, lower = mid - nσ*std
        buy: C crosses over lower; sell: C crosses under upper

    ATR:
        up_th = hi_prev + m * ATR_{t-1}
        dn_th = lo_prev - m * ATR_{t-1}
        buy: C > up_th; sell: C < dn_th

    MFI / RSI:
        buy: CrossUp(MFI or RSI, buy_thresh)
        sell: CrossDown(MFI or RSI, sell_thresh)

    Stoch:
        buy: (K < L) & CrossUp(K,D)
        sell:(K > U) & CrossDown(K,D)

    EMA:
        buy: EMA_fast cross up EMA_slow; sell: cross down

    MACD:
        line vs signal cross

    VWAP:
        C vs rolling VWAP cross

    OBV divergence:
        buy:  (C < price_low_prev) & (OBV >= obv_low_prev)
        sell: (C > price_high_prev) & (OBV <= obv_high_prev)
    """

    
    is_pro = name in PRO_TREND
    
    is_anti = name in ANTI_TREND
    
    vol_req = name in VOL_REQUIRED

    T = pc.c.shape[0]
    
    gate = np.ones(T, dtype = np.bool_)
    
    if is_pro:   
        
        gate &= pc.trend_mask
    
    elif is_anti: 
        
        gate &= ~pc.trend_mask
    
    if vol_req:   
        
        gate &= pc.vol_ok

    if name == "bb":
       
        ns = float(params.get("num_std", BB_STD_GRID[2]))
       
        upper = pc.bb_mid + pc.bb_std * ns
       
        lower = pc.bb_mid - pc.bb_std * ns
       
        buy_mat = crosses_over(
            x = pc.c[:, None], 
            y = lower[:, None]
        )
       
        sell_mat = crosses_under(
            x = pc.c[:, None], 
            y = upper[:, None]
        )

    elif name == "atr":
       
        m = float(params.get("mult", ATR_MULTIPLIER))
       
        atr_prev = np.roll(pc.atr, 1)
        
        atr_prev[0] = np.nan
       
        up_th = pc.hi_prev + atr_prev * m
       
        dn_th = pc.lo_prev - atr_prev * m
       
        buy_mat = (pc.c[:, None] > up_th[:, None])
       
        sell_mat = (pc.c[:, None] < dn_th[:, None])

    elif name == "mfi":
       
        bt = float(params.get("buy_thresh", MFI_BUY_THRESH))
       
        st = float(params.get("sell_thresh", MFI_SELL_THRESH))
       
        mfi = pc.mfi
       
        mfi_prev = np.roll(mfi, 1)
        
        mfi_prev[0] = np.nan
       
        buy_mat = ((mfi[:, None] > bt) & (mfi_prev[:, None] <= bt))
       
        sell_mat = ((mfi[:, None] < st) & (mfi_prev[:, None] >= st))

    elif name == "stoch":
        
        lo = float(params.get("lower", 20.0))
        
        hi = float(params.get("upper", 80.0))
      
        k = pc.stoch_k[:, None]
        
        d = pc.stoch_d[:, None]
       
        buy_mat = (k < lo) & crosses_over(
            x = k, 
            y = d
        )
        
        sell_mat = (k > hi) & crosses_under(
            x = k, 
            y = d
        )

    elif name == "rsi":
      
        bt = float(params.get("buy_thresh", RSI_BUY_THRESH))
      
        st = float(params.get("sell_thresh", RSI_SELL_THRESH))
      
        rsi = pc.rsi
      
        rsi_prev = np.roll(rsi, 1)
        
        rsi_prev[0] = np.nan
      
        buy_mat = ((rsi[:, None] > bt) & (rsi_prev[:, None] <= bt))
      
        sell_mat = ((rsi[:, None] < st) & (rsi_prev[:, None] >= st))

    elif name == "ema":
       
        buy_mat = crosses_over(
            x = pc.ema_fast[:, None], 
            y = pc.ema_slow[:, None]
        )
       
        sell_mat = crosses_under(
            x = pc.ema_fast[:, None],
            y = pc.ema_slow[:, None]
        )

    elif name == "macd":
     
        buy_mat = crosses_over(
            x = pc.macd_line[:, None], 
            y = pc.macd_signal[:, None]
        )
     
        sell_mat = crosses_under(pc.macd_line[:, None], pc.macd_signal[:, None])

    elif name == "vwap":
   
        buy_mat = crosses_over(
            x = pc.c[:, None], 
            y = pc.rvwap[:, None]
        )
   
        sell_mat = crosses_under(
            x = pc.c[:, None], 
            y = pc.rvwap[:, None]
        )

    elif name == "obv":
    
        c2 = pc.c[:, None]
    
        buy_mat = (c2 < pc.price_low_prev[:, None]) & (pc.obv[:, None] >= pc.obv_low_prev[:, None])
    
        sell_mat = (c2 > pc.price_high_prev[:, None]) & (pc.obv[:, None] <= pc.obv_high_prev[:, None])

    else:

        return np.zeros(T, dtype = np.int8)

    if not gate.all():
       
        buy_mat &= gate[:, None]
       
        sell_mat &= gate[:, None]

    pos = pulses_to_positions(
        buys = buy_mat, 
        sells = sell_mat, 
        signal_lag = SIGNAL_LAG, 
        cooldown_bars = COOLDOWN_BARS, 
        mode_id = _mode_id(
            mode = CONFLICT_MODE
        )
    )
   
    return pos[:, 0] 


def score_for_ticker(
    data: dict[str, pd.DataFrame],
    tk: str, 
    calib_raw: dict[str, dict]
) -> pd.Series:
    """
    Compute the weighted composite signal score time series for one ticker.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Output of `load_data()` with keys: "high","low","close","volume".
    tk : str
        Ticker symbol.
    calib_raw : dict[str, dict]
        Calibration dict for this ticker (from cache or calibration pass).

    Returns
    -------
    pd.Series[int16]
        Per-bar integer score clipped to [-SCORE_CLAMP, SCORE_CLAMP].
        Index equals the price index; name equals `tk`.

    Algorithm
    ---------
    1) Create `Precomp`.
    2) For each indicator in `INDICATOR_ORDER`:
    - Get `weight` and `params` (defaults if missing).
    - If weight>0, build a position vector via `_pos_for_indicator`.
    - Add `weight * position` to the aggregate score.
    3) Clip the result to avoid extreme values.

    Notes
    -----
    Positions are in {-1,0,+1}. The composite score reflects weighted agreement
    between active indicators after gating and conflict resolution.
    """

    c = _col(
        df = data["close"], 
        tk = tk, 
        name = "Close"
    )
    
    h = _col(
        df = data["high"], 
        tk = tk, 
        name = "High"
    )
    
    l = _col(
        df = data["low"],    
        tk = tk, 
        name = "Low"
    )
    
    v = _col(
        df = data["volume"], 
        tk = tk, 
        name = "Volume"
    )
            
    pc = precompute_all(
        h = h, 
        l = l,
        c = c,
        v = v
    )

    calib: dict[str, dict] = {}

    for name in INDICATOR_ORDER:

        if name in calib_raw:

            calib[name] = calib_raw[name]

        else:

            calib[name] = {
                "params": default_params_for(name),
                "weight": DEFAULT_WEIGHTS.get(name, 1),
                "sharpe": 0.0,
                "t_stat": 0.0,
                "p_value": 1.0, 
                "n_trades": 0
            }

    score = np.zeros(pc.c.shape[0], dtype=np.int16)
   
    for name in INDICATOR_ORDER:
   
        w = int(calib[name].get("weight", 0))
   
        if w <= 0:
   
            continue
   
        params = calib[name].get("params", {})
   
        pos = _pos_for_indicator(
            name = name, 
            params = params, 
            pc = pc
        ).astype(np.int16)
   
        score = score + (w * pos)

    score = np.clip(score, -SCORE_CLAMP, SCORE_CLAMP)
  
    return pd.Series(score, index = pc.idx, dtype = "int16", name = tk)


def main() -> None:
    """
    Top-level entry point: load data, calibrate (optionally), score tickers, save output.

    Behavior
    --------
    - Load OHLCV from Excel.
    - For each ticker:
        - If UPDATE_PARAM: run `calibrate_ticker` → accumulate rows for persistence.
        Else try the cache; fall back to calibrate if missing.
        - Compute score series via `score_for_ticker`.
    - If UPDATE_PARAM: upsert all calibration rows via `write_calibration_rows`.
    - Concatenate the per-ticker score series into a DataFrame, filter by date
    (`config.YEAR_AGO`), and write to the "Signal Scores" sheet.

    Logging
    -------
    Logs number of tickers, skips with exceptions, and I/O steps.
    """
   
    data = load_data(
        excel_file = config.DATA_FILE
    )
    
    tickers = list(config.tickers)

    logger.info("Scoring %d tickers", len(tickers))

    series_list: List[pd.Series] = []
  
    skipped: List[str] = []
  
    all_rows: List[dict] = []

    for tk in tickers:
      
        try:
      
            c = _col(
                df = data["close"], 
                tk = tk, 
                name = "Close"
            )
            
            h = _col(
                df = data["high"], 
                tk = tk, 
                name = "High"
            )
            
            l = _col(
                df = data["low"],    
                tk = tk, 
                name = "Low"
            )
            
            v = _col(
                df = data["volume"], 
                tk = tk, 
                name = "Volume"
            )
            
            if UPDATE_PARAM:
      
                calib_raw, rows = calibrate_ticker(
                    h = h, 
                    l = l, 
                    c = c, 
                    v = v, 
                    tk = tk
                )
      
                all_rows.extend(rows)
      
            else:
      
                cached = get_cached_for_ticker(
                    tk = tk
                )
      
                if cached is None:
   
                    calib_raw, rows = calibrate_ticker(
                        h = h,
                        l = l, 
                        c = c, 
                        v = v, 
                        tk = tk
                    )
   
                    all_rows.extend(rows)
      
                else:
      
                    calib_raw = cached

            s = score_for_ticker(
                data = data, 
                tk = tk, 
                calib_raw = calib_raw
            )
          
            if s.empty:
          
                raise ValueError(f"Empty score series for {tk}")
          
            series_list.append(s)

        except Exception as e:
  
            logger.exception("Skipping %s due to error: %s", tk, e)
  
            skipped.append(tk)

    if UPDATE_PARAM and all_rows:
     
        logger.info("Writing %d calibration rows ...", len(all_rows))
     
        write_calibration_rows(
            rows = all_rows
        )

    if not series_list:
     
        raise RuntimeError("No valid score series produced; cannot build score DataFrame.")

    score_df = pd.concat(series_list, axis = 1).sort_index()
    
    score_df = score_df.loc[score_df.index > pd.to_datetime(config.YEAR_AGO)]

    logger.info("Writing results back to workbook (skipped %d tickers)", len(skipped))
   
    save_scores(
        excel_file = config.DATA_FILE, 
        scores = score_df
    )
   
    logger.info("Done – sheet 'Signal Scores' updated.")


if __name__ == "__main__":
   
    main()

