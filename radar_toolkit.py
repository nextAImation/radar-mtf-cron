# -*- coding: utf-8 -*-
"""
RADAR v3.1 — Feature & Scoring Pipeline (radar_toolkit.py)

ROLE: Senior Quant Engineer
TASK: Implement feature-computation pipeline + scoring/tags/guards.

This module exposes:

Stage-1 (features):
    compute_indicators(df: pd.DataFrame, params: dict|None) -> pd.DataFrame

Required helpers (with exact names/signatures):
    compute_adx_di(df: pd.DataFrame, period:int=14) -> pd.DataFrame
    dynamic_resistance(df: pd.DataFrame, lookback:int=20, min_pullback:int=5) -> pd.Series
    anchored_vwap_from_last_swing_low(df: pd.DataFrame, lookback:int=20) -> pd.Series
    safe_zscore(s: pd.Series, win:int=60) -> pd.Series

Stage-2 (scoring/tags/guards):
    score_and_tags(df: pd.DataFrame, params: dict|None) -> dict

Additional helpers for regime/heat:
    compute_regime(df: pd.DataFrame, tf:str='1W'/'1M') -> pd.Series (1=up, 0=down)
    compute_heat(btc_df: pd.DataFrame|None, eth_df: pd.DataFrame|None) -> int  # 0/1/2

NEW (MTF helpers; pure, unit-testable):
    weekly_regime_from_1d(df_1d: pd.DataFrame) -> bool
    daily_setup_ok(df_1d: pd.DataFrame, min_score: int = 65, require_no_guards: bool = True) -> dict
    fourh_trigger_ok(df_4h: pd.DataFrame) -> bool
    mtf_decision(df_1d: pd.DataFrame, df_4h: pd.DataFrame, min_score:int=65, require_no_guards:bool=True) -> dict

Tags/Guards dictionary (for last row):
    {
        "score": int [0..100],
        "grade": "A"/"B"/"C"/"WATCH"/"⛔",
        "tags": list[str],     # e.g. ["RSI:52.1","ADX:24.3","ATR%:3.5","Fib:0.62","dist_res(ATRx):1.2",...]
        "guards": list[str],   # e.g. ["weak_volume","no_candle_confirm",...]
        "suggestion": str,     # "✅ BUY" / "⚠️ WATCH" / "⛔ REJECT"
        "stop": float|None, "tp1": float|None,"tp2": float|None,"tp3": float|None,
    }

NOTE: This module is "pure" wrt API fetching; It can compute on provided OHLCV DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Binance resilient fetch (fallback across multiple hosts) ---
import os, time
import requests  # ensure: pip install requests
import os, time, random, requests

# هاست‌های رسمی بایننس (برخی روی رنج‌های آی‌پی خاص بازترند)
BINANCE_BASES = [
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api-gcp.binance.com",
    "https://api.binance.com",
    "https://www.binance.com",  # وب‌دامین که اغلب Mirror هم دارد
]
# اگر خواستی دستی Override کنی:
OVERRIDE_BASE = os.environ.get("BINANCE_API_BASE")

REQ_SLEEP_SEC = float(os.environ.get("REQ_SLEEP_SEC", "0.3"))
MAX_RETRIES    = int(os.environ.get("BINANCE_MAX_RETRIES", "2"))

def fetch_klines_binance(symbol: str, interval: str, limit: int = 800, timeout: int = 20):
    """
    دریافت Klines با Fallback بین چند هاست برای دور زدن 451/403/429.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    bases = [OVERRIDE_BASE] if OVERRIDE_BASE else BINANCE_BASES[:]
    random.shuffle(bases)  # برای توزیع درخواست‌ها

    last_err = None
    for base in bases:
        url = f"{base}/api/v3/klines"
        for _ in range(MAX_RETRIES):
            try:
                r = requests.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                # خطاهای مربوط به بلاک/ریت‌لیمیت/کلودفلر
                if r.status_code in (451, 403, 429, 418, 520, 525):
                    last_err = f"{r.status_code} from {base}"
                    time.sleep(REQ_SLEEP_SEC)
                    continue
                r.raise_for_status()
            except Exception as e:
                last_err = f"{type(e).__name__}: {e} ({base})"
                time.sleep(REQ_SLEEP_SEC)
                continue
    raise RuntimeError(f"Failed to fetch klines for {symbol} {interval}. Last error: {last_err}")

BINANCE_BASES = [
    "https://api4.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api.binance.com",
]

REQ_SLEEP_SEC = float(os.environ.get("REQ_SLEEP_SEC", "0.25"))

def fetch_klines_binance(symbol: str, interval: str, limit: int = 800, timeout: int = 20):
    """
    Fetch Klines with fallback across several Binance API hosts.
    Retries next host on status codes 451/403/429/418/520/525 or on network errors.
    Returns r.json() on success; raises RuntimeError if all hosts fail.
    """
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    last_err = None
    for base in BINANCE_BASES:
        url = f"{base}/api/v3/klines"
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (451, 403, 429, 418, 520, 525):
                last_err = f"{r.status_code} from {base}"
                time.sleep(REQ_SLEEP_SEC)
                continue
            r.raise_for_status()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} ({base})"
            time.sleep(REQ_SLEEP_SEC)
            continue
    raise RuntimeError(f"Failed to fetch klines for {symbol} {interval}. Last error: {last_err}")

def klines_json_to_df(data):
    """Convert Binance klines JSON to a pandas DataFrame with standard columns.
    Returns DataFrame with index=time(UTC) and columns: open, high, low, close, vol
    """
    if not data:
        return pd.DataFrame(columns=["open","high","low","close","vol"])
    cols = [
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)[["openTime","open","high","low","close","volume"]].copy()
    df.rename(columns={"openTime":"time","volume":"vol"}, inplace=True)
    df["time"]  = pd.to_datetime(df["time"], unit="ms", utc=True)
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.set_index("time", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _sma(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=win).mean()

def _tr_df(df: pd.DataFrame) -> pd.DataFrame:
    # True Range components
    prev_close = df["close"].shift(1)
    high_low = (df["high"] - df["low"]).abs()
    high_pc  = (df["high"] - prev_close).abs()
    low_pc   = (df["low"] - prev_close).abs()
    tr = high_low.to_frame("hl")
    tr["hp"] = high_pc
    tr["lp"] = low_pc
    tr["tr"] = tr[["hl","hp","lp"]].max(axis=1)
    return tr

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b.replace(0, np.nan)
    return out.fillna(0.0)

def safe_zscore(s: pd.Series, win: int = 60) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=0)
    return _safe_div(s - mu, sd)

# ---------------------------------------------------------------------------
# Indicators
# -------------------------------------------------------------------
def compute_adx_di(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["high"]
    low  = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = (-low.diff())
    plus_dm  = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = _tr_df(df)["tr"]
    atr = tr.rolling(period, min_periods=period).mean()

    plus_di  = 100 * _safe_div(plus_dm.rolling(period, min_periods=period).mean(), atr)
    minus_di = 100 * _safe_div(minus_dm.rolling(period, min_periods=period).mean(), atr)
    dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) )
    adx = dx.rolling(period, min_periods=period).mean().fillna(0)

    out = pd.DataFrame({
        "adx14": adx,
        "di_plus14": plus_di.fillna(0),
        "di_minus14": minus_di.fillna(0),
    }, index=df.index)
    return out

def dynamic_resistance(df: pd.DataFrame, lookback: int = 20, min_pullback: int = 5) -> pd.Series:
    """
    Simple dynamic resistance candidate ~ recent swing-high under constraints.
    """
    highs = df["high"]
    # approximate pivot: highest high in last L, but ensure some pullback days
    roll_max = highs.rolling(lookback, min_periods=lookback).max()
    # enforce that current is not the same bar as max — need at least min_pullback bars since that high
    idxmax = highs.rolling(lookback, min_periods=lookback).apply(lambda x: np.argmax(x), raw=True)
    # position of the swing high relative to window end: 0..lookback-1
    pos = (lookback - 1) - idxmax
    cond = (pos >= min_pullback).fillna(False)
    dyn_res = roll_max.where(cond)
    return dyn_res

def anchored_vwap_from_last_swing_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    AVWAP anchored at most recent local swing low in last N bars (simple heuristic).
    """
    low = df["low"]
    # find local min in last 'lookback' (position)
    idxmin = low.rolling(lookback, min_periods=lookback).apply(lambda x: np.argmin(x), raw=True)
    # anchor index = current index - position
    pos = (lookback - 1) - idxmin
    # Price*Volume cumulative from anchor
    pv = (df["close"] * df["vol"]).copy()
    out = pd.Series(index=df.index, dtype="float64")
    # naive loop per row (ok for N<=5000 typical UI usage)
    csum_price = 0.0
    csum_vol = 0.0
    last_anchor = None
    for i, (t, p, v, pos_i) in enumerate(zip(df.index, df["close"].values, df["vol"].values, pos.values)):
        if np.isnan(pos_i):
            out.iloc[i] = np.nan
            continue
        # anchor at (i - pos_i)
        anchor_i = int(i - pos_i)
        if anchor_i != last_anchor:
            # reset from anchor
            csum_price = 0.0
            csum_vol = 0.0
            last_anchor = anchor_i
        csum_price += p * v
        csum_vol += v
        out.iloc[i] = csum_price / csum_vol if csum_vol > 0 else np.nan
    return out

# ---------------------------------------------------------------------------
# Feature computation
# -------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Adds vectorized features on a copy of df (expects columns: open, high, low, close, vol).
    """
    if df is None or df.empty:
        return df.copy()

    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["vol"]

    # EMAs
    df["ema20"]  = _ema(close, 20)
    df["ema50"]  = _ema(close, 50)
    df["ema200"] = _ema(close, 200)

    # ATR14, RSI14, ADX/DI
    tr = _tr_df(df)
    df["tr"] = tr["tr"]
    df["atr14"] = tr["tr"].rolling(14, min_periods=14).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = _safe_div(avg_gain, avg_loss.replace(0, np.nan))
    df["rsi14"] = 100 - (100 / (1 + rs.replace(0, np.nan))).fillna(0)

    adx_df = compute_adx_di(df, 14)
    df = df.join(adx_df)

    # Volume MAs & ratios
    df["vol_ma20"] = _sma(vol, 20)
    df["vol_ma60"] = _sma(vol, 60)
    df["vol_ratio"] = _safe_div(df["vol_ma20"], df["vol_ma60"])

    # Z-scores
    df["vol_z60"] = safe_zscore(vol, 60)
    df["tr_z60"]  = safe_zscore(df["tr"], 60)

    # Swings (20)
    df["sw_high20"] = high.rolling(20, min_periods=20).max()
    df["sw_low20"]  = low.rolling(20, min_periods=20).min()
    # Fib retrace (normalized pullback within [0..1]; 1=top, 0=bottom)
    rng = (df["sw_high20"] - df["sw_low20"]).replace(0, np.nan)
    df["fib_retrace"] = _safe_div(close - df["sw_low20"], rng).clip(0, 1)

    # Stacked/near EMA/candle_ok heuristics
    df["stacked"] = ((df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema200"])).astype(bool)
    df["near_ema"] = ((close - df["ema20"]).abs() <= 0.5*df["atr14"]).astype(bool)
    # simple candle confirmation: close>ema20 and range not too small; volume > 0.8*vol_ma20
    body = (df["close"] - df["open"]).abs()
    rng2 = (df["high"] - df["low"]).replace(0, np.nan)
    candle_valid = (df["close"] > df["ema20"]) & (_safe_div(body, rng2) > 0.3) & (vol > 0.8*df["vol_ma20"])
    df["candle_ok"] = candle_valid.fillna(False)

    # Dynamic resistance & distances
    df["dyn_res"] = dynamic_resistance(df, 20, 5)
    df["dist_res_atr"] = _safe_div(df["dyn_res"] - close, df["atr14"]).clip(lower=0)

    # Anchored VWAP
    df["avwap"] = anchored_vwap_from_last_swing_low(df, 20)

    return df

# ---------------------------------------------------------------------------
# Scoring / Tags / Guards
# -------------------------------------------------------------------
def score_and_tags(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Computes score/tags/guards for LAST ROW of df.
    """
    if df is None or df.empty:
        return {
            "score": 0, "grade": "⛔", "tags": [], "guards": ["no_data"],
            "suggestion": "⛔ REJECT", "stop": None, "tp1": None, "tp2": None, "tp3": None
        }

    p = {} if params is None else dict(params)
    # thresholds
    rsi_ok_lo  = float(p.get("RSI_OK_LO", 45))
    rsi_ok_hi  = float(p.get("RSI_OK_HI", 65))
    adx_min    = float(p.get("ADX_MIN", 18))
    atr_maxpct = float(p.get("ATR_PCT_MAX", 12))  # %
    fib_min    = float(p.get("FIB_MIN", 0.4))
    dist_res_min_atr = float(p.get("DIST_RES_MIN_ATR", 0.8))

    last = df.iloc[-1]
    px   = float(last["close"])
    ema20, ema50, ema200 = float(last["ema20"]), float(last["ema50"]), float(last["ema200"])
    rsi14 = float(last["rsi14"])
    adx14 = float(last["adx14"])
    atr14 = float(last["atr14"])
    vol_ratio = float(last["vol_ratio"])
    fib  = float(last["fib_retrace"])
    dist_res_atr = float(last["dist_res_atr"])
    stacked = bool(last["stacked"])
    near_ema = bool(last["near_ema"])
    candle_ok = bool(last["candle_ok"])

    tags = []
    guards = []

    # Base points
    score = 0
    # Trend stack
    if stacked: score += 15
    # RSI in mid zone (avoid overheat)
    if rsi_ok_lo <= rsi14 <= rsi_ok_hi: score += 15
    elif rsi14 < 35: guards.append("rsi_oversold")
    elif rsi14 > 70: guards.append("rsi_overbought")
    # ADX strength
    if adx14 >= adx_min: score += 15
    else: guards.append("weak_trend")
    # reasonable vol
    if vol_ratio >= 0.9: score += 10
    else: guards.append("weak_volume")
    # near ema pullback acceptance
    if near_ema: score += 10
    # fib zone preference
    if fib_min <= fib <= 0.8: score += 10
    elif fib > 0.85: guards.append("shallow_pullback")
    # headroom to resistance
    if dist_res_atr >= dist_res_min_atr: score += 15
    else: guards.append("near_resistance")
    # candle confirm
    if candle_ok: score += 10
    else: guards.append("no_candle_confirm")

    # ATR% (daily range) sanity vs price
    atr_pct = 100.0 * (atr14 / max(px, 1e-9))
    tags.extend([
        f"RSI:{rsi14:.1f}", f"ADX:{adx14:.1f}", f"ATR%:{atr_pct:.2f}",
        f"Fib:{fib:.2f}", f"dist_res(ATRx):{dist_res_atr:.2f}"
    ])

    if atr_pct > atr_maxpct:
        guards.append("too_volatile")

    # Grade mapping
    grade = "WATCH"
    if score >= 75 and not guards:
        grade = "A"
    elif score >= 65:
        grade = "B"
    elif score >= 55:
        grade = "C"
    else:
        grade = "WATCH"

    # Suggestion & basic stop/tp (illustrative)
    suggestion = "⚠️ WATCH"
    stop = tp1 = tp2 = tp3 = None

    if grade in ("A","B") and "near_resistance" not in guards and "too_volatile" not in guards:
        suggestion = "✅ BUY"
        # risk based stop at 1.8*ATR and TPs at 1.0/1.8/2.5 R
        R = 1.8 * atr14
        stop = px - R
        tp1 = px + 1.0*R
        tp2 = px + 1.8*R
        tp3 = px + 2.5*R
    elif grade == "C":
        suggestion = "⚠️ WATCH"
    else:
        suggestion = "⛔ REJECT"

    return {
        "score": int(round(score)),
        "grade": grade,
        "tags": tags,
        "guards": guards,
        "suggestion": suggestion,
        "stop": stop, "tp1": tp1, "tp2": tp2, "tp3": tp3
    }

# ---------------------------------------------------------------------------
# Regime / Heat
# -------------------------------------------------------------------
def compute_regime(df: pd.DataFrame, tf: str = "1W") -> pd.Series:
    """
    Weekly/Monthly regime (1=up, 0=down) derived from 1D DF via resampling if needed.
    """
    if df is None or df.empty:
        return pd.Series(dtype="int8")
    if tf not in ("1W","1M"):
        raise ValueError("tf must be '1W' or '1M'")
    # resample
    o = df["open"].resample(tf).first()
    h = df["high"].resample(tf).max()
    l = df["low"].resample(tf).min()
    c = df["close"].resample(tf).last()
    v = df["vol"].resample(tf).sum()
    wdf = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"vol":v}).dropna()
    wdf["ema20"]  = _ema(wdf["close"], 20)
    wdf["ema50"]  = _ema(wdf["close"], 50)
    wdf["ema200"] = _ema(wdf["close"], 200)
    regime = ((wdf["ema20"] > wdf["ema50"]) & (wdf["ema50"] > wdf["ema200"])).astype(int)
    return regime.reindex(df.index, method="ffill").fillna(0).astype(int)

def compute_heat(btc_df: pd.DataFrame | None, eth_df: pd.DataFrame | None) -> int:
    """
    Simple market heat: 0=cold, 1=neutral, 2=hot based on ADX/RSI states of BTC & ETH.
    """
    vals = []
    for df in (btc_df, eth_df):
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        adx = float(last.get("adx14", 0))
        rsi = float(last.get("rsi14", 50))
        hot = (adx >= 20 and 48 <= rsi <= 65)
        vals.append(2 if hot else (1 if 45 <= rsi <= 70 else 0))
    if not vals:
        return 1
    s = np.mean(vals)
    if s >= 1.5: return 2
    if s >= 0.8: return 1
    return 0

# ---------------------------------------------------------------------------
# MTF helpers (1W/1D/4H triad)
# -------------------------------------------------------------------
def weekly_regime_from_1d(df_1d: pd.DataFrame) -> bool:
    if df_1d is None or df_1d.empty:
        return False
    reg = compute_regime(df_1d, "1W")
    return bool(reg.iloc[-1] == 1)

def daily_setup_ok(df_1d: pd.DataFrame, min_score: int = 65, require_no_guards: bool = True) -> dict:
    dff = compute_indicators(df_1d)
    st = score_and_tags(dff)
    ok = st["score"] >= int(min_score)
    if require_no_guards:
        ok = ok and (len(st["guards"]) == 0)
    return {"ok": bool(ok), "score": st["score"], "grade": st["grade"], "guards": st["guards"], "tags": st["tags"]}

def fourh_trigger_ok(df_4h: pd.DataFrame) -> bool:
    """
    Minimal 4H trigger: close>ema20 and candle_ok True on last bar.
    """
    if df_4h is None or df_4h.empty:
        return False
    d4 = compute_indicators(df_4h)
    last = d4.iloc[-1]
    return bool( (last["close"] > last["ema20"]) and bool(last["candle_ok"]) )

def mtf_decision(df_1d: pd.DataFrame, df_4h: pd.DataFrame, min_score:int=65, require_no_guards:bool=True) -> dict:
    """
    Combines 1W regime (derived from 1D), 1D setup, and 4H trigger.
    """
    wk = weekly_regime_from_1d(df_1d)
    d = daily_setup_ok(df_1d, min_score=min_score, require_no_guards=require_no_guards)
    h4 = fourh_trigger_ok(df_4h)
    signal = "HOLD"
    if wk and d["ok"] and h4:
        signal = "BUY"
    return {
        "signal": signal,
        "weekly": wk, "daily_ok": d, "fourh_trigger": h4
    }

