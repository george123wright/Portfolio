"""
Computes classic technical analysis signals from price data stored in Excel. 
It loads OHLCV sheets, scores each ticker and writes “Signal Scores” back to the workbook.
"""

from __future__ import annotations
import datetime as dt
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule
from openpyxl import load_workbook 
import config

EMA_FAST: int = 12
EMA_SLOW: int = 26

BB_WINDOW: int = 20
BB_STD: float = 2.0

STOCH_K: int = 14
STOCH_D: int = 3

ATR_WINDOW: int = 14
ATR_BREAK_WINDOW: int = 20
ATR_MULTIPLIER: float = 1.5

OBV_LOOKBACK: int = 20

ADX_WINDOW: int = 14
ADX_THRESHOLD: float = 25.0

RSI_BUY_THRESH: float = 30.0
RSI_SELL_THRESH: float = 70.0

SCORE_CLAMP: int = 10

VWAP_WINDOW: int = 20  

MFI_WINDOW: int = 14
MFI_BUY_THRESH: float = 20.0
MFI_SELL_THRESH: float = 80.0

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if not logger.handlers:

    _h = logging.StreamHandler()

    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(_h)


def read_sheet(
    excel_file: str | Path, 
    sheet: str, *, 
    multiheader: bool = False
) -> pd.DataFrame:
    """
    Read one sheet and return a DateTime‑indexed DataFrame.
    """
 
    try:
       
        hdr = [0, 1] if multiheader else 0
       
        df = pd.read_excel(excel_file, sheet_name=sheet, index_col=0, header=hdr, parse_dates=True)
    
    except Exception as exc:   
    
        logger.exception("Parsing %s!%s failed (%s)", excel_file, sheet, type(exc).__name__)
    
        raise

    if not pd.api.types.is_datetime64_any_dtype(df.index):
    
        raise ValueError(f"Sheet '{sheet}' has non‑datetime index")
    
    return df.sort_index()


def load_data(
    excel_file: str | Path
) -> Dict[str, pd.DataFrame]:
    """
    Return a dict of OHLCV plus pre‑computed indicators.
    """
    
    mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "rsi": "RSI",
        "macd": "MACD",
        "macd_signal": "MACD Signal",
    }
  
    multi = {"rsi", "macd", "macd_signal"}
 
    out: Dict[str, pd.DataFrame] = {}
 
    for key, sheet in mapping.items():
        
        out[key] = read_sheet(excel_file, sheet, multiheader=key in multi)
   
    return out


def get_series(
    df: pd.DataFrame, 
    level: str, 
    ticker: str
) -> pd.Series:
    """
    Return a single Series from *df* whether it has a MultiIndex or not.
    """
    
    if isinstance(df.columns, pd.MultiIndex):
       
        if (level, ticker) in df.columns:
       
            return df[(level, ticker)]
       
        logger.warning("%s not found at level '%s' – filling NaNs", ticker, level)
       
        return pd.Series(np.nan, index=df.index)

    if ticker in df.columns:
       
        return df[ticker]

    logger.warning("%s not found in single‑level DataFrame – filling NaNs", ticker)
    
    return pd.Series(np.nan, index=df.index)


def save_scores(
    excel_file: str | Path, 
    scores: pd.DataFrame
) -> None:
    """
    Write *scores* to *Signal Scores* sheet with traffic‑light formatting.
    """

    green = PatternFill("solid", start_color = "90EE90", end_color = "90EE90")
    red = PatternFill("solid", start_color = "FFC7CE", end_color = "FFC7CE")

    if Path(excel_file).exists():
       
        wb = load_workbook(excel_file)
    
    else:
    
        wb = load_workbook(filename=None)

    if "Signal Scores" in wb.sheetnames:
    
        ws = wb["Signal Scores"]
    
        wb.remove(ws)
    
    ws = wb.create_sheet("Signal Scores")

    for j, col in enumerate(["Date"] + scores.columns.tolist(), 1):
       
        ws.cell(row=1, column=j, value=col)
   
    for i, (day, row) in enumerate(scores.iterrows(), 2):
       
        ws.cell(row=i, column=1, value=day)
       
        for j, val in enumerate(row, 2):
       
            ws.cell(row=i, column=j, value=int(val) if pd.notna(val) else None)

    rng = f"B2:{get_column_letter(ws.max_column)}{ws.max_row}"
   
    rule_green = CellIsRule(operator="greaterThan", formula=["0"], fill=green)
    rule_red = CellIsRule(operator="lessThan", formula=["0"], fill=red)
  
    ws.conditional_formatting.add(rng, rule_green)
    ws.conditional_formatting.add(rng, rule_red)

    wb.save(excel_file)


def persistent_position(
    buy: pd.Series, 
    sell: pd.Series
) -> pd.Series:
    """
    Convert one‑bar *buy/sell* pulses into a persistent position series (1, 0, ‑1).
    """
    
    pos = pd.Series(np.nan, index=buy.index)
    
    pos[buy] = 1
    pos[sell] = -1
    
    return pos.ffill().fillna(0).astype("int8")


def macd_signals(
    macd: pd.Series, 
    signal: pd.Series
) -> Tuple[pd.Series, pd.Series]:
   
    buy = (macd > signal) & (macd.shift() <= signal.shift())
    sell = (macd < signal) & (macd.shift() >= signal.shift())
   
    return buy.fillna(False), sell.fillna(False)


def rsi_signals(
    rsi: pd.Series, 
    *, 
    buy_thresh: float = RSI_BUY_THRESH, 
    sell_thresh: float = RSI_SELL_THRESH
) -> Tuple[pd.Series, pd.Series]:
   
    buy = (rsi < buy_thresh) & (rsi.shift() >= buy_thresh)
    sell = (rsi > sell_thresh) & (rsi.shift() <= sell_thresh)
   
    return buy.fillna(False), sell.fillna(False)


def ema_crossover_signals(
    close: pd.Series, 
    *, 
    fast: int = EMA_FAST, 
    slow: int = EMA_SLOW
) -> Tuple[pd.Series, pd.Series]:
   
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
   
    buy = (ema_fast > ema_slow) & (ema_fast.shift() <= ema_slow.shift())
    sell = (ema_fast < ema_slow) & (ema_fast.shift() >= ema_slow.shift())
   
    return buy.fillna(False), sell.fillna(False)


def bollinger_signals(
    close: pd.Series, 
    *, 
    window: int = BB_WINDOW, 
    num_std: float = BB_STD
) -> Tuple[pd.Series, pd.Series]:
   
    mid = close.rolling(window).mean()
   
    std = close.rolling(window).std()
   
    upper = mid + num_std * std
    lower = mid - num_std * std
   
    buy = (close.shift() < lower) & (close >= lower)
    sell = (close.shift() > upper) & (close <= upper)
   
    return buy.fillna(False), sell.fillna(False)


def stochastic_signals(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    *, 
    k: int = STOCH_K, 
    d: int = STOCH_D
) -> Tuple[pd.Series, pd.Series]:
   
    low_k = low.rolling(k).min()
    high_k = high.rolling(k).max()
   
    pct_k = 100 * (close - low_k) / (high_k - low_k)
    pct_d = pct_k.rolling(d).mean()
   
    buy = (pct_k < 20) & (pct_k.shift() < pct_d.shift()) & (pct_k > pct_d)
    sell = (pct_k > 80) & (pct_k.shift() > pct_d.shift()) & (pct_k < pct_d)
   
    return buy.fillna(False), sell.fillna(False)


def atr_breakout_signals(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    *, 
    atr_window: int = ATR_WINDOW, 
    lookback: int = ATR_BREAK_WINDOW, 
    mult: float = ATR_MULTIPLIER
) -> Tuple[pd.Series, pd.Series]:
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_window).mean()
    
    hi = high.rolling(lookback).max().shift()
    lo = low.rolling(lookback).min().shift()
    
    buy = close > hi + mult * atr.shift()
    sell = close < lo - mult * atr.shift()
    
    return buy.fillna(False), sell.fillna(False)


def obv_divergence_signals(
    close: pd.Series, 
    volume: pd.Series, 
    *, 
    lookback: int = OBV_LOOKBACK
) -> Tuple[pd.Series, pd.Series]:
    """
    Divergence logic: bullish when price makes lower low but OBV does not.
    """
    
    price_diff = close.diff()
   
    direction = np.sign(price_diff).fillna(0.0)
   
    obv = pd.Series((direction * volume).cumsum(), index=close.index)

    price_high = close.rolling(lookback).max().shift()
    obv_high = obv.rolling(lookback).max().shift()
    
    price_low = close.rolling(lookback).min().shift()
    obv_low = obv.rolling(lookback).min().shift()

    buy = (close < price_low) & (obv > obv_low)
    sell = (close > price_high) & (obv < obv_high)

    return buy.fillna(False), sell.fillna(False)


def adx_series(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    *, 
    window: int = ADX_WINDOW
) -> pd.Series:
    """
    True Wilder ADX with protection against zero division.
    """
   
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
   
    atr = tr.ewm(alpha=1.0 / window, adjust=False).mean()

    up_move = high.diff().clip(lower=0)
    down_move = low.shift().sub(low).clip(lower=0)

    plus_dm = up_move.where(up_move > down_move, 0.0)
    minus_dm = down_move.where(down_move > up_move, 0.0)

    plus_di = 100 * plus_dm.ewm(alpha=1.0 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / window, adjust=False).mean() / atr.replace(0, np.nan)

    denom = plus_di + minus_di
    denom = denom.replace(0, np.nan)
   
    dx = 100 * (plus_di - minus_di).abs() / denom
   
    adx = dx.ewm(alpha=1.0 / window, adjust=False).mean()
   
    return adx


def mfi_signals(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    volume: pd.Series, 
    *, 
    window: int = MFI_WINDOW, 
    buy_thresh: float = MFI_BUY_THRESH, 
    sell_thresh: float = MFI_SELL_THRESH
) -> Tuple[pd.Series, pd.Series]:
    """
    Money Flow Index crossing thresholds with zero‑division protection
    """
    
    typical_price = (high + low + close) / 3.0
    
    money_flow = typical_price * volume
    
    pos_flow = money_flow.where(typical_price.diff() > 0, 0.0)
    neg_flow = money_flow.where(typical_price.diff() < 0, 0.0)

    pos_sum = pos_flow.rolling(window).sum()
    neg_sum = neg_flow.rolling(window).sum().replace(0, np.nan)
    
    mf_ratio = pos_sum / neg_sum
    mf_ratio = mf_ratio.replace([np.inf, -np.inf], np.nan)

    mfi = 100 - (100 / (1 + mf_ratio))
    
    buy = (mfi < buy_thresh) & (mfi.shift() >= buy_thresh)
    sell = (mfi > sell_thresh) & (mfi.shift() <= sell_thresh)
    
    return buy.fillna(False), sell.fillna(False)


def vwap_signals(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    volume: pd.Series, 
    *,
    window: int = VWAP_WINDOW
) -> Tuple[pd.Series, pd.Series]:
   
    tp = (high + low + close) / 3.0
   
    rolling_vwap = tp * volume
    rolling_vwap = rolling_vwap.rolling(window).sum() / volume.rolling(window).sum()
   
    buy = (close > rolling_vwap) & (close.shift() <= rolling_vwap.shift())
    sell = (close < rolling_vwap) & (close.shift() >= rolling_vwap.shift())
   
    return buy.fillna(False), sell.fillna(False)


def _latest_workbook(
    pattern: str = r"Portfolio_Optimisation_Data_(\d{4}-\d{2}-\d{2}).xlsx"
) -> Path | None:
      
    candidates: list[tuple[dt.date, Path]] = []
   
    reg = re.compile(pattern)
   
    for file in Path.cwd().glob("Portfolio_Optimisation_Data_*.xlsx"):
       
        m = reg.match(file.name)
   
        if m:
   
            try:
               
                d = dt.date.fromisoformat(m.group(1))
               
                candidates.append((d, file))
           
            except ValueError:
           
                continue
   
    if not candidates:
       
        return None
   
    candidates.sort(key=lambda t: t[0])
   
    latest_date, latest_file = candidates[-1]
   
    if latest_date < config.TODAY - dt.timedelta(days=60):
       
        logger.warning("Most recent workbook is %s – more than 60d old", latest_date)
   
    return latest_file


def score_for_ticker(
    data: dict[str, pd.DataFrame], 
    tk: str
) -> pd.Series:
   
    c = data["close"][tk]
    
    h = data["high"][tk]
    
    l = data["low"][tk]
    
    v = data["volume"][tk]

    rsi = get_series(
        df = data["rsi"], 
        level = "RSI", 
        ticker = tk
    )
    
    mac = get_series(
        df = data["macd"], 
        level = "MACD", 
        ticker = tk
    )
    
    sig = get_series(
        df = data["macd_signal"], 
        level = "MACD Signal", 
        ticker = tk
    )

    score = pd.Series(0, index=c.index, dtype="int8")

    adx_filter = (adx_series(
        high = h, 
        low = l, 
        close = c, 
        window = ADX_WINDOW
    ) > ADX_THRESHOLD).fillna(False)

    indicators = {
        "rsi": lambda: rsi_signals(rsi),
        "macd": lambda: macd_signals(mac, sig),
        "ema": lambda: ema_crossover_signals(c),
        "bb": lambda: bollinger_signals(c),
        "stoch": lambda: stochastic_signals(h, l, c),
        "atr": lambda: atr_breakout_signals(h, l, c),
        "obv": lambda: obv_divergence_signals(c, v),
        "mfi": lambda: mfi_signals(h, l, c, v),
    }

    trend_filtered = {"macd", "ema", "atr"}  

    for name, fn in indicators.items():
       
        buy, sell = fn()
       
        pos = persistent_position(
            buy = buy, 
            sell = sell
        )
       
        if name in trend_filtered:
            
            pos = pos.where(adx_filter, 0)
       
        score += pos

    return score.clip(-SCORE_CLAMP, SCORE_CLAMP)


def main(
    argv: list[str] | None = None
) -> None:
   
    argv = argv if argv is not None else sys.argv[1:]
   
    if argv:
       
        wb_path = Path(argv[0])
   
    else:
      
        wb_path = _latest_workbook()
   
        if wb_path is None:
            
            logger.error("No Portfolio_Optimisation_Data_*.xlsx found in %s", Path.cwd())
            
            sys.exit(1)
   
    if not wb_path.exists():
       
        logger.error("Workbook %s not found", wb_path)
       
        sys.exit(1)

    logger.info("Loading data from %s", wb_path)
   
    data = load_data(wb_path)
   
    tickers = data["close"].columns.tolist()

    logger.info("Scoring %d tickers", len(tickers))
   
    scores: Dict[str, pd.Series] = {tk: score_for_ticker(data, tk) for tk in tickers}
   
    score_df = pd.DataFrame(scores).sort_index()

    logger.info("Writing results back to workbook")
   
    save_scores(wb_path, score_df)
   
    logger.info("Done – sheet 'Signal Scores' updated.")


if __name__ == "__main__":
    main()
