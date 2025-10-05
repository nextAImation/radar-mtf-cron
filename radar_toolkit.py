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
    mtf_decision(df_1d: pd.DataFrame, df_4h: pd.DataFrame,
                 min_score: int = 65, require_no_guards: bool = True) -> dict

Constraints:
- No internet calls. Minimal deps (numpy, pandas).
- Handle short histories gracefully (NaN/inf where appropriate; never crash).

compute_indicators adds ALL of the following columns:
    ema20, ema50, ema200,
    atr14, rsi14, adx14, di_plus14, di_minus14,
    vol_ma20, vol_ma60, vol_ratio,
    tr, vol_z60, tr_z60,
    sw_high20, sw_low20, fib_retrace,
    stacked (bool), near_ema (bool), candle_ok (bool),
    dyn_res, dist_res_atr,
    avwap

score_and_tags computes for the LAST ROW:
- score: int in [0..100]
- grade: {"A","B","C","WATCH","⛔"}
- tags: list[str] with keys (formatted numbers):
  ["RSI","ADX","ATR%","Fib","dist_res(ATRx)","Stacked","NearEMA","CandleOK",
   "VolPB_OK","VolTrig_OK","QP","ET","vol_z","TR_z","Regime1W","Regime1M","Heat","RS:Top"]
- guards: list[str] using EXACT Persian whitelist:
  ["رژیم هفتگی نزولی","قدرت روند ضعیف (ADX)","نوسان خیلی بالا (ATR%)","فاصله تا مقاومت ناکافی",
   "حجم کندل تأییدی ضعیف","کندل تأییدی معتبر نیست",
   "Pullback خیلی کم؛ در ناحیه‌ی بالا (Fib>0.75)","Pullback خیلی عمیق (Fib<0.30)","بازار کل ضعیف (BTC Filter)"]

QP/ET (exact):
* QP: mean volume of last 3 red candles ≤ 0.85×vol_ma20 AND mean red body ≤ 40th percentile of bodies in last 60 bars.
* ET: vol_z60 ≥ +0.8 AND tr_z60 ≥ +0.5 AND High>High[-1] AND Close ≥ Low + 0.65*(H−L).
* VolPB_OK and VolTrig_OK reflect QP and ET respectively (boolean flags). Score gets +10 only if QP & ET both True.

Regime & Heat:
* compute_regime:
    1W: Close(1W) > EMA200(1W), daily fallback if short
    1M: EMA10(1M) > EMA20(1M)
* compute_heat(btc_df, eth_df):
    0 Risk-Off: (RSI<50) or (Close<EMA200) on BTC or ETH
    1 Neutral : default / mild
    2 Risk-On : RSI≥55 & EMA50 slope positive on both

GRADE mapping (if NO guards):
* score≥85 → "A"
* 75–84 → "B"
* 60–74 → "C"
* else → "WATCH"
"""
from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with adjust=False."""
    return s.ewm(span=span, adjust=False).mean()


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI implementation using EMA(alpha=1/period).
    """
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    Classic True Range = max(H-L, |H-C[-1]|, |L-C[-1]|).
    """
    high, low, close = df["high"], df["low"], df["close"]
    a = high - low
    b = (high - close.shift()).abs()
    c = (low - close.shift()).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    return tr


def _wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ATR via smoothing TR with EMA(alpha=1/period).
    """
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


# ---------------------------------------------------------------------------
# Required Helpers (exported)
# ---------------------------------------------------------------------------

def compute_adx_di(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute classic DI+/DI- and ADX using Wilder's smoothing.

    Returns a DataFrame with columns: di_plus14, di_minus14, adx14
    """
    if len(df) == 0:
        return pd.DataFrame(
            {"di_plus14": pd.Series(dtype=float),
             "di_minus14": pd.Series(dtype=float),
             "adx14": pd.Series(dtype=float)}
        )

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    atr = _wilder_atr(df.assign(high=high, low=low, close=close), period)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9))

    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    out = pd.DataFrame({
        "di_plus14": plus_di.fillna(0.0),
        "di_minus14": minus_di.fillna(0.0),
        "adx14": adx.fillna(0.0),
    }, index=df.index)
    return out


def dynamic_resistance(df: pd.DataFrame, lookback: int = 20, min_pullback: int = 5) -> pd.Series:
    """
    Per-bar distance to the most-relevant recent swing-high within last `lookback`
    bars that has at least `min_pullback` bars of lower-high pullback AFTER the pivot.
    If not found, +inf.
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float, index=df.index)

    highs = pd.to_numeric(df["high"], errors="coerce").values
    close = pd.to_numeric(df["close"], errors="coerce").values

    dyn = np.full(n, np.inf, dtype=float)

    for i in range(n):
        start = max(0, i - lookback + 1)
        end = i
        if end - start + 1 < lookback:
            dyn[i] = np.inf
            continue

        window_highs = highs[start:end + 1]
        if not np.isfinite(window_highs).all():
            dyn[i] = np.inf
            continue

        max_h = window_highs.max()
        found = np.inf
        for j in range(end, start - 1, -1):
            if not np.isfinite(highs[j]):
                continue
            if abs(highs[j] - max_h) < 1e-12:
                if j + min_pullback <= end:
                    segment = highs[j + 1: j + 1 + min_pullback]
                    if len(segment) == min_pullback and np.all(segment < highs[j]):
                        found = max(highs[j] - close[i], 0.0)
                        break
        dyn[i] = found

    return pd.Series(dyn, index=df.index)


def anchored_vwap_from_last_swing_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Anchored VWAP (AVWAP) from the last swing-low (rolling min over `lookback`).
    Before anchor: NaN. From anchor: cumsum(price*vol)/cumsum(vol).
    """
    if len(df) == 0:
        return pd.Series(dtype=float, index=df.index)

    sw_low20 = pd.to_numeric(df["low"], errors="coerce").rolling(lookback, min_periods=lookback).min()

    if not sw_low20.notna().any():
        return pd.Series(np.nan, index=df.index, dtype=float)

    anchor_idx = sw_low20[sw_low20.notna()].index.max()
    try:
        k = df.index.get_loc(anchor_idx)
    except Exception:
        k = max(len(df) - 1 - lookback, 0)

    price = pd.to_numeric(df["close"], errors="coerce").astype(float)
    vol = pd.to_numeric(df["vol"], errors="coerce").astype(float)

    avwap = pd.Series(np.nan, index=df.index, dtype=float)
    pv_cum = (price.iloc[k:] * vol.iloc[k:]).cumsum()
    v_cum = vol.iloc[k:].cumsum().replace(0, np.nan)
    avwap.iloc[k:] = (pv_cum / v_cum).values
    return avwap


def safe_zscore(s: pd.Series, win: int = 60) -> pd.Series:
    """
    Rolling z-score with safe denominator:
    z = (x - mean_win) / (std_win + 1e-9)
    """
    s = pd.to_numeric(s, errors="coerce")
    mean = s.rolling(win, min_periods=win).mean()
    std = s.rolling(win, min_periods=win).std(ddof=0)
    return (s - mean) / (std + 1e-9)


# ---------------------------------------------------------------------------
# Feature computation (Stage-1)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute RADAR v3.1 indicators/features (without scoring) and return augmented DataFrame.

    Required input columns: open, high, low, close, vol
    """
    if params is None:
        params = {}

    required = ["open", "high", "low", "close", "vol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Basic indicators
    out["ema20"] = _ema(out["close"], 20)
    out["ema50"] = _ema(out["close"], 50)
    out["ema200"] = _ema(out["close"], 200)

    out["atr14"] = _wilder_atr(out, 14)
    out["rsi14"] = _wilder_rsi(out["close"], 14)

    adx_df = compute_adx_di(out, period=14)
    out["di_plus14"] = adx_df["di_plus14"]
    out["di_minus14"] = adx_df["di_minus14"]
    out["adx14"] = adx_df["adx14"]

    # Volume features
    out["vol_ma20"] = out["vol"].rolling(20, min_periods=5).mean()
    out["vol_ma60"] = out["vol"].rolling(60, min_periods=20).mean()
    out["vol_ratio"] = (out["vol_ma20"] / (out["vol_ma60"] + 1e-9)).fillna(0.0)

    # True range & z-scores
    out["tr"] = out["high"] - out["low"]
    out["vol_z60"] = safe_zscore(out["vol"], 60)
    out["tr_z60"] = safe_zscore(out["tr"], 60)

    # Swings & Fib retracement
    out["sw_high20"] = out["high"].rolling(20, min_periods=20).max()
    out["sw_low20"] = out["low"].rolling(20, min_periods=20).min()
    rng = (out["sw_high20"] - out["sw_low20"]).clip(lower=1e-9)
    out["fib_retrace"] = ((out["close"] - out["sw_low20"]) / rng).clip(lower=0.0, upper=1.0)

    # Stacked trend
    out["stacked"] = ((out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"])).astype(bool)

    # NearEMA adaptive
    dist_e20 = (out["close"] - out["ema20"]).abs()
    dist_e50 = (out["close"] - out["ema50"]).abs()
    min_dist = pd.concat([dist_e20, dist_e50], axis=1).min(axis=1)
    k_base = 0.5
    k_relaxed = 0.7
    k = np.where((out["adx14"] >= 18) & (out["adx14"] < 25), k_relaxed, k_base)
    out["near_ema"] = (min_dist <= (pd.to_numeric(out["atr14"], errors="coerce") * k)).astype(bool)

    # Candle OK
    bar_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    pos = (out["close"] - out["low"]) / (bar_range + 1e-9)
    t = np.where(out["adx14"] >= 25, 0.70, 0.60)
    out["candle_ok"] = ((out["close"] > out["open"]) & (pos >= t)).astype(bool)

    # Dynamic resistance distance (ATR-normalized)
    out["dyn_res"] = dynamic_resistance(out, lookback=20, min_pullback=5)
    out["dist_res_atr"] = (out["dyn_res"] / (out["atr14"].abs() + 1e-9)).replace([np.inf, -np.inf], np.inf)

    # Anchored VWAP from last swing-low
    out["avwap"] = anchored_vwap_from_last_swing_low(out, lookback=20)

    return out


# ---------------------------------------------------------------------------
# Regime & Heat helpers
# ---------------------------------------------------------------------------

def compute_regime(df: pd.DataFrame, tf: str = "1W") -> pd.Series:
    """
    Compute regime boolean series mapped to the original index (1=up, 0=down).

    - 1W regime: Close(1W) > EMA200(1W), daily fallback if short.
    - 1M regime: EMA10(1M) > EMA20(1M).
    """
    close = pd.to_numeric(df["close"], errors="coerce")

    def daily_fallback_1w(_close: pd.Series) -> pd.Series:
        ema200d = _ema(_close, 200)
        return (_close > ema200d).astype(int)

    def daily_fallback_1m(_close: pd.Series) -> pd.Series:
        ema10d = _ema(_close, 10)
        ema20d = _ema(_close, 20)
        return (ema10d > ema20d).astype(int)

    if not isinstance(df.index, pd.DatetimeIndex):
        return daily_fallback_1w(close) if tf.upper() == "1W" else daily_fallback_1m(close)

    try:
        if tf.upper() == "1W":
            dfr = df[["close"]].resample("W").last()
            ema200w = _ema(dfr["close"], 200)
            regime_w = (dfr["close"] > ema200w).astype(int)
            return regime_w.reindex(df.index).ffill().fillna(0).astype(int)
        elif tf.upper() == "1M":
            # Use month-end ('ME') to avoid pandas 'M' deprecation
            dfr = df[["close"]].resample("ME").last()
            ema10m = _ema(dfr["close"], 10)
            ema20m = _ema(dfr["close"], 20)
            regime_m = (ema10m > ema20m).astype(int)
            return regime_m.reindex(df.index).ffill().fillna(0).astype(int)
        else:
            raise ValueError("tf must be '1W' or '1M'")
    except Exception:
        return daily_fallback_1w(close) if tf.upper() == "1W" else daily_fallback_1m(close)


def compute_heat(btc_df: Optional[pd.DataFrame], eth_df: Optional[pd.DataFrame]) -> int:
    """
    Compute market heat from BTC & ETH frames (0 Risk-Off, 1 Neutral, 2 Risk-On).
    Safe fallback: return 1 if inputs missing/short.
    """
    try:
        def ensure_ind(df: pd.DataFrame) -> pd.DataFrame:
            need = {"ema50", "ema200", "rsi14", "close"}
            if not need.issubset(df.columns):
                df = compute_indicators(df)
            return df

        if btc_df is None or eth_df is None:
            return 1

        btc = ensure_ind(btc_df.copy())
        eth = ensure_ind(eth_df.copy())
        if len(btc) < 15 or len(eth) < 15:
            return 1

        def slope_up(df_: pd.DataFrame, lookback: int = 10) -> bool:
            if len(df_) < lookback + 1:
                return False
            return (float(df_["ema50"].iloc[-1]) - float(df_["ema50"].iloc[-lookback - 1])) > 0

        # Risk-Off
        risk_off = ((float(btc["rsi14"].iloc[-1]) < 50) or (float(btc["close"].iloc[-1]) < float(btc["ema200"].iloc[-1])) or
                    (float(eth["rsi14"].iloc[-1]) < 50) or (float(eth["close"].iloc[-1]) < float(eth["ema200"].iloc[-1])))
        if risk_off:
            return 0

        # Risk-On
        risk_on = ((float(btc["rsi14"].iloc[-1]) >= 55) and slope_up(btc) and
                   (float(eth["rsi14"].iloc[-1]) >= 55) and slope_up(eth))
        if risk_on:
            return 2

        return 1
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# QP / ET flags
# ---------------------------------------------------------------------------

def _quiet_pullback_flag(df: pd.DataFrame) -> pd.Series:
    """
    QP: mean volume of last 3 red candles ≤ 0.85×vol_ma20 AND
        mean red body ≤ 40th percentile of bodies in last 60 bars.
    """
    body_abs = (df["close"] - df["open"]).abs()
    red = (df["close"] < df["open"]).astype(int)

    vol_red = (df["vol"] * red).rolling(3, min_periods=3).mean()
    cnt_red = red.rolling(3, min_periods=3).sum().replace(0, np.nan)
    mean_vol_red3 = vol_red / cnt_red

    bodies60 = body_abs.rolling(60, min_periods=20)
    p40 = bodies60.quantile(0.40).ffill()

    cond_vol = mean_vol_red3 <= (0.85 * df["vol_ma20"])
    cond_body = body_abs.rolling(3, min_periods=3).mean() <= p40
    return (cond_vol & cond_body).fillna(False)


def _expansion_trigger_flag(df: pd.DataFrame) -> pd.Series:
    """
    ET: vol_z60 ≥ +0.8 AND tr_z60 ≥ +0.5 AND High>High[-1] AND
        Close ≥ Low + 0.65*(H−L).
    """
    vol_ok = df["vol_z60"] >= 0.8
    tr_ok = df["tr_z60"] >= 0.5
    higher_high = df["high"] > df["high"].shift(1)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    pos = (df["close"] - df["low"]) / (rng + 1e-9)
    pos_ok = pos >= 0.65
    return (vol_ok & tr_ok & higher_high & pos_ok).fillna(False)


# ---------------------------------------------------------------------------
# Scoring / Tags / Guards (Stage-2)
# ---------------------------------------------------------------------------

def score_and_tags(df: pd.DataFrame, params: Optional[dict] = None) -> Dict:
    """
    Compute score/grade/tags/guards for the LAST ROW.
    """
    if params is None:
        params = {}
    is_alt = bool(params.get("is_alt", True))

    # Compute indicators/features first
    df_feat = compute_indicators(df, params)
    last = df_feat.iloc[-1]

    # Regimes
    reg1w_series = compute_regime(df_feat, "1W")
    reg1m_series = compute_regime(df_feat, "1M")
    reg1w = int(reg1w_series.iloc[-1]) if len(reg1w_series) else 0
    reg1m = int(reg1m_series.iloc[-1]) if len(reg1m_series) else 0

    # Heat
    if "heat" in params and params["heat"] in (0, 1, 2):
        heat = int(params["heat"])
    else:
        heat = compute_heat(params.get("btc_df"), params.get("eth_df"))

    # QP / ET booleans
    qp = int(_quiet_pullback_flag(df_feat).iloc[-1])
    et = int(_expansion_trigger_flag(df_feat).iloc[-1])

    # Distance-to-resistance check (adaptive)
    adx = float(last.get("adx14", np.nan)) if pd.notna(last.get("adx14", np.nan)) else 0.0
    dist_res_atr = float(last.get("dist_res_atr", np.inf))
    thr = 0.4 if (18 <= adx < 25) else 0.3
    dist_ok = (dist_res_atr >= thr)

    # Guards (exact Persian whitelist)
    guards: List[str] = []
    if reg1w == 0:
        guards.append("رژیم هفتگی نزولی")
    if adx < 18:
        guards.append("قدرت روند ضعیف (ADX)")
    atr_pct = float((last["atr14"] / (last["close"] + 1e-9) * 100)) if pd.notna(last["atr14"]) and pd.notna(last["close"]) else np.nan
    if pd.notna(atr_pct) and atr_pct > 10:
        guards.append("نوسان خیلی بالا (ATR%)")
    if not dist_ok:
        guards.append("فاصله تا مقاومت ناکافی")
    vol_z = float(last.get("vol_z60", 0.0)) if pd.notna(last.get("vol_z60", np.nan)) else 0.0
    if (et == 0) and (vol_z < 0):
        guards.append("حجم کندل تأییدی ضعیف")
    if not bool(last.get("candle_ok", False)):
        guards.append("کندل تأییدی معتبر نیست")
    fib = float(last.get("fib_retrace", np.nan)) if pd.notna(last.get("fib_retrace", np.nan)) else np.nan
    if pd.notna(fib) and fib > 0.75:
        guards.append("Pullback خیلی کم؛ در ناحیه‌ی بالا (Fib>0.75)")
    if pd.notna(fib) and fib < 0.30:
        guards.append("Pullback خیلی عمیق (Fib<0.30)")
    if (heat == 0) and is_alt:
        guards.append("بازار کل ضعیف (BTC Filter)")

    # Scoring components (sum to 0..100; WITHOUT RS-Top which is applied by scanner)
    score = 0
    if reg1w == 1:
        score += 15
    if adx >= 25:
        score += 10
    if bool(last.get("stacked", False)):
        score += 20
    if bool(last.get("near_ema", False)):
        score += 15
    if pd.notna(fib) and (0.5 <= fib <= 0.618):
        score += 10
    # RSI in [50,60] and rising vs 10 bars ago
    rsi_series = df_feat["rsi14"]
    rsi_last = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else np.nan
    rsi_prev = float(rsi_series.iloc[-10]) if len(rsi_series) > 10 and pd.notna(rsi_series.iloc[-10]) else np.nan
    if pd.notna(rsi_last) and 50 <= rsi_last <= 60 and pd.notna(rsi_prev) and (rsi_last > rsi_prev):
        score += 10
    # Volume signature: QP & ET
    if (qp == 1) and (et == 1):
        score += 10
    # Distance to resistance OK
    if dist_ok:
        score += 10

    # Grade mapping (if NO guards)
    if len(guards) == 0:
        if score >= 85:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 60:
            grade = "C"
        else:
            grade = "WATCH"
    else:
        grade = "⛔"

    # Build tags list (ordered, formatted)
    def fmt(x, n):
        try:
            return f"{float(x):.{n}f}"
        except Exception:
            return "NaN"

    atr_pct_val = (last["atr14"] / (last["close"] + 1e-9) * 100) if pd.notna(last["atr14"]) and pd.notna(last["close"]) else np.nan
    tags = [
        f"RSI={fmt(rsi_last,1)}",
        f"ADX={fmt(adx,1)}",
        f"ATR%={fmt(atr_pct_val,2)}",
        f"Fib={fmt(fib,2)}",
        f"dist_res(ATRx)={fmt(dist_res_atr,2)}",
        f"Stacked={int(bool(last.get('stacked', False)))}",
        f"NearEMA={int(bool(last.get('near_ema', False)))}",
        f"CandleOK={int(bool(last.get('candle_ok', False)))}",
        f"VolPB_OK={int(qp==1)}",
        f"VolTrig_OK={int(et==1)}",
        f"QP={int(qp==1)}",
        f"ET={int(et==1)}",
        f"vol_z={fmt(vol_z,2)}",
        f"TR_z={fmt(last.get('tr_z60', np.nan),2)}",
        f"Regime1W={int(reg1w==1)}",
        f"Regime1M={int(reg1m==1)}",
        f"Heat={int(heat)}",
        "RS:Top=0",
    ]

    return {
        "score": int(score),
        "grade": grade,
        "tags": tags,
        "guards": guards,
        "last": {
            "close": float(last.get("close", np.nan)) if pd.notna(last.get("close", np.nan)) else np.nan,
            "atr": float(last.get("atr14", np.nan)) if pd.notna(last.get("atr14", np.nan)) else np.nan,
            "high": float(last.get("high", np.nan)) if pd.notna(last.get("high", np.nan)) else np.nan,
            "low": float(last.get("low", np.nan)) if pd.notna(last.get("low", np.nan)) else np.nan,
        },
    }


# ---------------------------------------------------------------------------
# NEW — Minimal, composable MTF helpers (pure functions; no I/O)
# ---------------------------------------------------------------------------

def weekly_regime_from_1d(df_1d: "pd.DataFrame") -> bool:
    """
    Resample 1D → 1W (closed='right', label='right'), compute ema20_w, ema50_w, adx14_w,
    return True iff (ema20_w > ema50_w) AND (adx14_w >= 20). Safe False on short/NaN.
    """
    try:
        if not isinstance(df_1d.index, pd.DatetimeIndex) or len(df_1d) < 10:
            return False

        # Robust OHLCV weekly resample (right-closed, right-labeled)
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "vol": "sum",
        }
        w = (df_1d[["open", "high", "low", "close", "vol"]]
             .resample("W", label="right", closed="right")
             .agg(agg)
             .dropna(how="any"))

        if len(w) < 20:  # need some history to stabilize EMA20/50 & ADX14
            return False

        w = w.copy()
        w["ema20_w"] = _ema(pd.to_numeric(w["close"], errors="coerce"), 20)
        w["ema50_w"] = _ema(pd.to_numeric(w["close"], errors="coerce"), 50)
        adxw = compute_adx_di(w, period=14)
        w["adx14_w"] = adxw["adx14"]

        last = w.iloc[-1]
        ema20_w = float(last.get("ema20_w", np.nan))
        ema50_w = float(last.get("ema50_w", np.nan))
        adx14_w = float(last.get("adx14_w", np.nan))

        if not (np.isfinite(ema20_w) and np.isfinite(ema50_w) and np.isfinite(adx14_w)):
            return False

        return (ema20_w > ema50_w) and (adx14_w >= 20.0)
    except Exception:
        return False


def daily_setup_ok(df_1d: "pd.DataFrame", min_score: int = 65, require_no_guards: bool = True) -> dict:
    """
    Evaluate LAST 1D bar using existing compute_indicators/score_and_tags.
    Returns: {"ok":bool,"score":int,"grade":str,"guards":list,"tags":list}
    ok=True iff (score>=min_score) AND (require_no_guards → no guards).
    """
    try:
        st = score_and_tags(df_1d, params={})
        score = int(st.get("score", 0))
        guards = list(st.get("guards", [])) if st.get("guards") is not None else []
        grade = str(st.get("grade", "WATCH"))
        tags = list(st.get("tags", [])) if st.get("tags") is not None else []

        no_guards = (len(guards) == 0)
        ok = (score >= int(min_score)) and (no_guards if require_no_guards else True)

        return {"ok": bool(ok), "score": score, "grade": grade, "guards": guards, "tags": tags}
    except Exception:
        return {"ok": False, "score": 0, "grade": "WATCH", "guards": [], "tags": []}


def fourh_trigger_ok(df_4h: "pd.DataFrame") -> bool:
    """
    Compute indicators on 4H, extract last tags via score_and_tags,
    return True iff any tag startswith 'ET=' and endswith '1'.
    """
    try:
        st = score_and_tags(df_4h, params={})
        tags = list(st.get("tags", []))
        if not tags:
            return False
        for t in tags:
            # exact pattern requirement: startswith "ET=" and endswith "1"
            if isinstance(t, str) and t.startswith("ET=") and t.endswith("1"):
                return True
        return False
    except Exception:
        return False


def mtf_decision(df_1d: "pd.DataFrame", df_4h: "pd.DataFrame",
                 min_score: int = 65, require_no_guards: bool = True) -> dict:
    """
    Compose the MTF signal:
      if not weekly_uptrend → {"signal":"NO_TRADE","reason":"weekly_down_or_weak","score":score_1d}
      if daily_ok & fourh_ok → {"signal":"BUY","reason":"weekly+daily+4h","score":score_1d}
      if daily_ok & !fourh_ok → {"signal":"WATCH","reason":"weekly+daily_only","score":score_1d}
      else → {"signal":"NO_TRADE","reason":"daily_not_ok","score":score_1d}
    Must never raise on NaNs; default False for booleans if insufficient data.
    """
    try:
        wk = bool(weekly_regime_from_1d(df_1d))
        dres = daily_setup_ok(df_1d, min_score=min_score, require_no_guards=require_no_guards)
        score_1d = int(dres.get("score", 0))
        d_ok = bool(dres.get("ok", False))
        h4_ok = bool(fourh_trigger_ok(df_4h))

        if not wk:
            return {"signal": "NO_TRADE", "reason": "weekly_down_or_weak", "score": score_1d}
        if d_ok and h4_ok:
            return {"signal": "BUY", "reason": "weekly+daily+4h", "score": score_1d}
        if d_ok and not h4_ok:
            return {"signal": "WATCH", "reason": "weekly+daily_only", "score": score_1d}
        return {"signal": "NO_TRADE", "reason": "daily_not_ok", "score": score_1d}
    except Exception:
        return {"signal": "NO_TRADE", "reason": "error", "score": 0}


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Existing toolkit self-checks ----
    np.random.seed(7)
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(np.random.normal(0, 0.8, size=n))
    high = price + np.random.uniform(0.2, 1.2, size=n)
    low = price - np.random.uniform(0.2, 1.2, size=n)
    open_ = price + np.random.normal(0, 0.3, size=n)
    close = price + np.random.normal(0, 0.3, size=n)
    vol = np.random.lognormal(mean=10.0, sigma=0.35, size=n)

    base = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "vol": vol},
        index=idx
    )

    ind = compute_indicators(base.copy(), params={})
    required_cols = [
        "ema20", "ema50", "ema200",
        "atr14", "rsi14",
        "adx14", "di_plus14", "di_minus14",
        "vol_ma20", "vol_ma60", "vol_ratio",
        "tr", "vol_z60", "tr_z60",
        "sw_high20", "sw_low20", "fib_retrace",
        "stacked", "near_ema", "candle_ok",
        "dyn_res", "dist_res_atr",
        "avwap",
    ]
    miss = [c for c in required_cols if c not in ind.columns]
    assert not miss, f"Missing columns: {miss}"
    print("TOOLKIT_OK")

    res = score_and_tags(base.copy(), params={"heat": 1, "is_alt": True})
    print({"score": res["score"], "grade": res["grade"], "tags": res["tags"][:6], "guards": res["guards"]})

    flat = base.copy()
    flat.iloc[-15:, :] = flat.iloc[-16, :].values
    res2 = score_and_tags(flat, params={"heat": 1, "is_alt": True})
    print({"guards_low_adx": res2["guards"]})

    fib_hi = base.copy()
    fib_hi.loc[fib_hi.index[-1], "close"] = fib_hi["high"].rolling(20, min_periods=1).max().iloc[-1] * 0.999
    res3 = score_and_tags(fib_hi, params={"heat": 1, "is_alt": True})
    print({"guards_fib_hi": res3["guards"]})

    res4 = score_and_tags(base.copy(), params={"heat": 0, "is_alt": True})
    print({"guards_risk_off": res4["guards"]})

    print("SCORING_OK")

    # ---- NEW: Minimal MTF helpers self-test (synthetic bullish) ----
    # Build tiny-but-sufficient bullish 1D (~120 rows for stable weekly resample)
    n_d = 120
    idx_d = pd.date_range("2024-01-01", periods=n_d, freq="D")
    # Monotonic up drift + small noise
    close_d = 100 + np.linspace(0, 40, n_d) + np.random.normal(0, 0.2, n_d)
    open_d = close_d - np.random.uniform(0.1, 0.5, n_d)
    high_d = np.maximum(open_d, close_d) + np.random.uniform(0.2, 0.6, n_d)
    low_d = np.minimum(open_d, close_d) - np.random.uniform(0.2, 0.6, n_d)
    vol_d = np.random.lognormal(mean=10.0, sigma=0.25, size=n_d)
    df_1d = pd.DataFrame({"open": open_d, "high": high_d, "low": low_d, "close": close_d, "vol": vol_d},
                         index=idx_d)

    # Build bullish 4H with enough rows (>= 64) so z-scores exist; push a strong ET at the end
    n_h4 = 80
    idx_h4 = pd.date_range(idx_d[-40], periods=n_h4, freq="4H")  # overlaps recent days
    base_c = 140 + np.linspace(0, 12, n_h4) + np.random.normal(0, 0.15, n_h4)
    open_h4 = base_c - np.random.uniform(0.05, 0.25, n_h4)
    high_h4 = np.maximum(open_h4, base_c) + np.random.uniform(0.1, 0.4, n_h4)
    low_h4 = np.minimum(open_h4, base_c) - np.random.uniform(0.1, 0.4, n_h4)
    vol_h4 = np.random.lognormal(mean=9.6, sigma=0.30, size=n_h4)

    # Force the last bar to satisfy ET conditions: higher high, close near high, big vol & range
    high_h4[-1] = high_h4[-2] + 1.0
    low_h4[-1] = low_h4[-2] - 0.2
    base_c[-1] = high_h4[-1] - 0.05  # close near high
    open_h4[-1] = base_c[-1] - 0.3
    vol_h4[-1] = vol_h4[:-10].mean() * 2.5  # boost volume

    df_4h = pd.DataFrame({"open": open_h4, "high": high_h4, "low": low_h4, "close": base_c, "vol": vol_h4},
                         index=idx_h4)

    print("W:", weekly_regime_from_1d(df_1d))
    print("D:", daily_setup_ok(df_1d))
    print("H4:", fourh_trigger_ok(df_4h))
    print("MTF:", mtf_decision(df_1d, df_4h))
