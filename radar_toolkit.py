# -*- coding: utf-8 -*-
"""
RADAR v3.1 — Feature & Scoring Pipeline (radar_toolkit.py)

- Adds Yahoo Finance fetcher: fetch_klines_yf(symbol, interval="1d", limit=800)
- Keeps pure feature/scoring/MTF helpers (no Binance/requests).
"""

from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np
import pandas as pd

# ----------------------------- NEW: Yahoo Finance fetcher -----------------------------
# pip install yfinance
import yfinance as yf


def fetch_klines_yf(symbol: str, interval: str = "1d", limit: int = 800) -> pd.DataFrame:
    """
    دریافت داده‌های کندل از Yahoo Finance (جایگزین Binance)
    interval: '1d', '4h'  (Yahoo 4h ندارد؛ از 1h می‌گیریم و 4H Resample می‌کنیم)
    خروجی استاندارد: DataFrame با ایندکس زمانی UTC و ستون‌های:
        open, high, low, close, vol
    """
    tf_map = {
        "1d": "1d",
        "4h": "1h",  # Yahoo 4h نداره، پس باید Resample کنیم
        "1h": "1h",
    }
    yf_symbol = symbol.replace("USDT", "-USD")  # BTCUSDT → BTC-USD

    data = yf.download(
        yf_symbol,
        period="720d",                      # حدود دو سال
        interval=tf_map.get(interval, "1d"),
        progress=False,
        auto_adjust=True,                   # OHLC adjusted
    )

    if data is None or data.empty:
        raise ValueError(f"No data returned for {symbol} from Yahoo Finance")

    # اگر تایم‌فریم 4h خواستی، از 1h به 4h Resample می‌کنیم
    if interval == "4h":
        # اطمینان از DatetimeIndex (UTC)
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True, errors="coerce")
        else:
            data.index = data.index.tz_localize("UTC") if data.index.tz is None else data.index.tz_convert("UTC")

        data = data.resample("4H", label="right", closed="right").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(how="any")

    # ایندکس UTC و نگاشت به قالب استاندارد
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, utc=True, errors="coerce")
    else:
        data.index = data.index.tz_localize("UTC") if data.index.tz is None else data.index.tz_convert("UTC")

    out = pd.DataFrame({
        "open": pd.to_numeric(data["Open"], errors="coerce"),
        "high": pd.to_numeric(data["High"], errors="coerce"),
        "low":  pd.to_numeric(data["Low"],  errors="coerce"),
        "close":pd.to_numeric(data["Close"],errors="coerce"),
        "vol":  pd.to_numeric(data["Volume"],errors="coerce"),
    }, index=data.index).dropna()

    if limit and len(out) > limit:
        out = out.iloc[-limit:]

    return out


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
