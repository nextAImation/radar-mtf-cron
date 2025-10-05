# -*- coding: utf-8 -*-
"""
RADAR — Live Scanner to Telegram (MTF: Weekly + Daily + 4H)

• Fetch: OHLCV 1D & 4H from Yahoo Finance (via yfinance)
• Compute: weekly_regime, daily_setup_ok, 4H fast trigger
• Send: ranked list (most attractive → least) to Telegram with concise reasons + key levels + momentum 7d/30d.

ENV (optional):
  SYMBOLS, MTF_ENABLED, MIN_SCORE_D, REQUIRE_NO_GUARDS
  LIMIT_1D, LIMIT_4H
  REQ_SLEEP_SEC (default 0.25)
  TELEGRAM_API_BASE (default https://api.telegram.org)
  HTTPS_PROXY (proxy only for Telegram, optional)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np

from radar_toolkit import (
    fetch_klines_yf,        # ← NEW: Yahoo Finance fetcher
    compute_indicators,
    weekly_regime_from_1d,
    daily_setup_ok,
)

# =================== Telegram creds (HARD-CODED as requested) ===================
BOT_TOKEN = "7512369490:AAHiQqOzjLxh5zjcx3gmUT-hEj1196tIHfI"
CHAT_ID   = "-1002925489017"
print("DBG | BOT_TOKEN len:", len(BOT_TOKEN), " | CHAT_ID:", CHAT_ID)

# =================== Config & Defaults ===================
DFLT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,LINKUSDT,AVAXUSDT,MATICUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,ARBUSDT"

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", DFLT_SYMBOLS).split(",") if s.strip()]
MIN_SCORE_D = int(os.getenv("MIN_SCORE_D", "55"))
REQUIRE_NO_GUARDS = bool(int(os.getenv("REQUIRE_NO_GUARDS", "0")))
MTF_ENABLED = bool(int(os.getenv("MTF_ENABLED", "1")))
LIMIT_1D = int(os.getenv("LIMIT_1D", "900"))
LIMIT_4H = int(os.getenv("LIMIT_4H", "1800"))
REQ_SLEEP_SEC = float(os.getenv("REQ_SLEEP_SEC", "0.25"))
TELEGRAM_API_BASE = os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")  # optional, for Telegram only

EMO = {"ENTRY": "🟢", "WATCH": "🟡", "NO_TRADE": "⚫️", "ERROR": "⚠️"}

# =================== HTTP (proxy-aware) for Telegram only ===================
def _requests_session_with_proxy() -> requests.Session:
    sess = requests.Session()
    if HTTPS_PROXY:
        sess.proxies.update({"https": HTTPS_PROXY})
        print("DBG | Using HTTPS proxy for Telegram:", HTTPS_PROXY)
    return sess

HTTP = _requests_session_with_proxy()

# =================== Data loader (Yahoo Finance) ===================
def load_df(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Wraps radar_toolkit.fetch_klines_yf to standardize fetching.
    interval: '1d' or '4h' (4h is built from 1h via resample inside the helper)
    """
    return fetch_klines_yf(symbol, interval=interval, limit=limit)

# =================== Key levels (1D) ===================
def key_levels_1d(df_1d: pd.DataFrame, look1: int = 20, look2: int = 60) -> dict:
    if len(df_1d) < max(look1, look2):
        look1 = min(look1, len(df_1d))
        look2 = min(look2, len(df_1d))
    h1 = float(df_1d["high"].tail(look1).max())
    l1 = float(df_1d["low"].tail(look1).min())
    h2 = float(df_1d["high"].tail(look2).max())
    l2 = float(df_1d["low"].tail(look2).min())
    return {"R1": h1, "S1": l1, "R2": h2, "S2": l2}

# =================== Momentum (1D) ===================
def momentum_pct(df_1d: pd.DataFrame, window: int) -> float | None:
    """Return % change over N last closed days (close[t]/close[t-N]-1)*100."""
    if len(df_1d) <= window:
        return None
    try:
        c_now = float(df_1d["close"].iloc[-1])
        c_past = float(df_1d["close"].iloc[-window-1])  # -1 is last closed bar
        if c_past == 0:
            return None
        return (c_now / c_past - 1.0) * 100.0
    except Exception:
        return None

# =================== 4H ET fast flags ===================
def prep_4h_flags(df_4h_raw: pd.DataFrame) -> pd.DataFrame:
    if df_4h_raw is None or len(df_4h_raw) == 0:
        return pd.DataFrame()
    df4 = compute_indicators(df_4h_raw.copy())
    rng = (df4["high"] - df4["low"])
    cond = (
        (df4.get("vol_z60", 0) >= 0.8) &
        (df4.get("tr_z60", 0) >= 0.5) &
        (df4["high"] > df4["high"].shift(1)) &
        (df4["close"] >= df4["low"] + 0.65 * rng)
    )
    out = pd.DataFrame({"et_ok": cond.astype(bool)}, index=df4.index)
    return out

def fourh_ok_between(df_4h_flags: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
    if df_4h_flags is None or len(df_4h_flags) == 0:
        return False
    try:
        win = df_4h_flags.loc[(df_4h_flags.index >= t0) & (df_4h_flags.index < t1)]
    except Exception:
        return False
    return bool(len(win) > 0 and win["et_ok"].any())

# =================== Reasons & Evaluation ===================
def build_reason_tags(weekly_ok: bool, dchk: dict, fourh_ok: bool, mtf_enabled: bool) -> str:
    parts = []
    parts.append("W✓" if weekly_ok else "W✗")
    if isinstance(dchk, dict):
        score = int(dchk.get("score", 0))
        guards = dchk.get("guards", [])
        g = len(guards) if guards is not None else 0
        parts.append(f"D✓(S={score},G={g})" if dchk.get("ok", False) else f"D✗(S={score},G={g})")
    else:
        parts.append("D?")
    if mtf_enabled:
        parts.append("4H✓" if fourh_ok else "4H…")
    return " | ".join(parts)

def evaluate_symbol(symbol: str) -> dict:
    out = {
        "symbol": symbol, "status": "ERROR", "score": None,
        "weekly": None, "daily_ok": None, "fourh_ok": None,
        "reason": "", "close_1d": None, "t_1d": None,
        "levels": None, "m7": None, "m30": None,
    }
    try:
        # 1D
        df_1d = load_df(symbol, "1d", limit=LIMIT_1D)
        df_1d_ind = compute_indicators(df_1d)
        close_1d = float(df_1d_ind["close"].iloc[-1])
        t1d = df_1d_ind.index[-1]
        lv = key_levels_1d(df_1d_ind)
        m7  = momentum_pct(df_1d_ind, 7)
        m30 = momentum_pct(df_1d_ind, 30)

        if MTF_ENABLED:
            weekly_ok = bool(weekly_regime_from_1d(df_1d_ind))
            dchk = daily_setup_ok(df_1d_ind, min_score=MIN_SCORE_D, require_no_guards=REQUIRE_NO_GUARDS)
            daily_ok = bool(dchk.get("ok", False))
            score = int(dchk.get("score", 0))

            # 4H window aligned to the last 1D bar
            t0 = t1d
            t1 = t0 + pd.Timedelta(days=1)

            time.sleep(REQ_SLEEP_SEC)
            df_4h = load_df(symbol, "4h", limit=LIMIT_4H)
            flags_4h = prep_4h_flags(df_4h)
            f4 = fourh_ok_between(flags_4h, t0, t1)

            if weekly_ok and daily_ok and f4:
                status = "ENTRY"
            elif weekly_ok and daily_ok:
                status = "WATCH"
            else:
                status = "NO_TRADE"

            reason = build_reason_tags(weekly_ok, dchk, f4, True)
            out.update({
                "status": status, "score": score,
                "weekly": weekly_ok, "daily_ok": daily_ok, "fourh_ok": f4,
                "reason": reason, "close_1d": close_1d, "t_1d": t1d.strftime("%Y-%m-%d"),
                "levels": lv, "m7": m7, "m30": m30,
            })
        else:
            dchk = daily_setup_ok(df_1d_ind, min_score=MIN_SCORE_D, require_no_guards=REQUIRE_NO_GUARDS)
            daily_ok = bool(dchk.get("ok", False))
            score = int(dchk.get("score", 0))
            status = "WATCH" if daily_ok else "NO_TRADE"
            reason = build_reason_tags(False, dchk, False, False)
            out.update({
                "status": status, "score": score,
                "weekly": None, "daily_ok": daily_ok, "fourh_ok": None,
                "reason": reason, "close_1d": close_1d, "t_1d": t1d.strftime("%Y-%m-%d"),
                "levels": lv, "m7": m7, "m30": m30,
            })
        return out
    except Exception as e:
        out["reason"] = f"error: {e}"
        return out

# =================== Ranking & Formatting ===================
def attractiveness_rank(r: dict) -> tuple:
    st = r.get("status", "ERROR")
    weekly = bool(r.get("weekly")) if r.get("weekly") is not None else False
    daily_ok = bool(r.get("daily_ok")) if r.get("daily_ok") is not None else False
    score = int(r.get("score") or 0)

    if st == "ENTRY":
        group = 0
    elif st == "WATCH" and weekly:
        group = 1
    elif st == "NO_TRADE" and daily_ok:
        group = 2
    elif st == "NO_TRADE":
        group = 3
    else:
        group = 4  # ERROR

    return (group, -score)

def fmt_levels(lv: dict) -> str:
    if not isinstance(lv, dict): return "-"
    R1, R2, S1, S2 = lv["R1"], lv["R2"], lv["S1"], lv["S2"]
    def f(x):
        try:
            return f"{x:.4f}" if abs(x) < 1000 else f"{x:.2f}"
        except Exception:
            return "-"
    return f"R≈{f(R1)}/{f(R2)}  •  S≈{f(S1)}/{f(S2)}"

def fmt_pct(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    sign = "+" if x >= 0 else ""
    if abs(x) < 10:
        return f"{sign}{x:.2f}%"
    return f"{sign}{x:.1f}%"

def format_ranked_report(results: list[dict]) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    n = len(results)
    ranked = sorted(results, key=attractiveness_rank)

    c_entry = sum(1 for r in results if r["status"] == "ENTRY")
    c_watch = sum(1 for r in results if r["status"] == "WATCH")
    c_none  = sum(1 for r in results if r["status"] == "NO_TRADE")
    c_err   = sum(1 for r in results if r["status"] == "ERROR")

    head = (
        f"📡 *RADAR MTF* — از جذاب به کم‌جذاب\n"
        f"@ {now_utc} | 1D=1d • 4H=4h | نمادها: {n}\n"
        f"🟢 {c_entry}  🟡 {c_watch}  ⚫️ {c_none}" + (f"  ⚠️ {c_err}" if c_err else "") +
        "\n\n✨ *رتبه‌بندی امروز:*\n"
    )

    blocks = []
    for idx, r in enumerate(ranked, start=1):
        sym    = r["symbol"]
        st     = r.get("status", "ERROR")
        score  = r.get("score", "-")
        close  = r.get("close_1d")
        t1d    = r.get("t_1d", "-")
        reason = r.get("reason", "")
        lv     = r.get("levels")
        m7     = r.get("m7")
        m30    = r.get("m30")

        close_s = f"{close:.4f}" if isinstance(close, (int, float, np.floating)) else "-"
        emoji  = EMO.get(st, "⚪")
        levels_str = fmt_levels(lv)
        mom_str = f"Mom: 7d={fmt_pct(m7)} • 30d={fmt_pct(m30)}"

        block = (
            f"{idx:>2}) {emoji} *{sym}* — *{st}* | S={score} | C={close_s} (1D: {t1d})\n"
            f"    ↳ {reason}\n"
            f"    ▸ Levels: {levels_str}\n"
            f"    ▸ {mom_str}"
        )
        blocks.append(block)

    legend = (
        "\n\nℹ️ *Legend*: S=Score, C=Close  |  W✓/W✗=Weekly ok/not  |  D✓/D✗=Daily ok/not (S,G=guards)  |  4H✓/4H…=trigger  |  Mom=Momentum\n"
        "_criteria: weekly + daily score/guards + intraday 4H trigger_"
    )

    msg = head + ("\n\n".join(blocks)) + legend
    if len(msg) > 3800:
        msg = head + ("\n\n".join(blocks[:25])) + "\n\n_(truncated)_\n" + legend
    return msg

# =================== Telegram helpers (retries + proxy) ===================
def tg_send_message(text: str) -> None:
    base = TELEGRAM_API_BASE.rstrip("/")
    url = f"{base}/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    last_err = None
    for attempt in range(1, 3+1):
        try:
            r = HTTP.post(url, json=payload, timeout=25)
            r.raise_for_status()
            resp = r.json()
            if not isinstance(resp, dict) or not resp.get("ok", False):
                raise RuntimeError(f"Telegram API error: {resp}")
            print("✅ Telegram delivered.")
            return
        except Exception as e:
            last_err = e
            print(f"❌ Telegram send failed (try {attempt}/3):", e)
            try:
                print("RAW:", r.text[:1000])
            except Exception:
                pass
            time.sleep(1.2)
    raise SystemExit(f"Failed to send Telegram message after retries: {last_err}")

# =================== Main ===================
def main():
    print(f"SYMBOLS={','.join(SYMBOLS)} | MTF={int(MTF_ENABLED)} | MIN_SCORE_D={MIN_SCORE_D} | NO_GUARDS={int(REQUIRE_NO_GUARDS)}")
    results = []
    for i, s in enumerate(SYMBOLS, start=1):
        print(f"→ [{i}/{len(SYMBOLS)}] {s}")
        res = evaluate_symbol(s)
        print(res)
        results.append(res)
        time.sleep(REQ_SLEEP_SEC)

    msg = format_ranked_report(results)
    print("\n--- TELEGRAM MESSAGE ---\n" + msg)
    tg_send_message(msg)

if __name__ == "__main__":
    main()
