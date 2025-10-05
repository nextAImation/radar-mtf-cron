# --- Self-Test (manual run) ---
if __name__ == "__main__":
    import os
    from pathlib import Path

    # مسیر فولدر داده‌ها
    DATA_DIR = Path(r"D:\model\data")   # ← مسیر خودت
    SYMBOL = "ETHUSDT"
    INTERVAL = "1d"
    LIMIT = 800

    print(f"\n📡 Fetching {SYMBOL} ({INTERVAL}) ...")
    try:
        data = fetch_klines_binance(SYMBOL, INTERVAL, limit=LIMIT)
        df = klines_json_to_df(data)
        print(f"✅ Loaded {len(df)} bars from Binance")

        # --- ذخیره نسخه خام برای بررسی بعدی
        csv_path = DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv"
        df.to_csv(csv_path, index=True)
        print(f"💾 Saved to {csv_path}")

        # --- محاسبه اندیکاتورها و امتیاز
        df_feat = compute_indicators(df)
        out = score_and_tags(df_feat)

        print("\n✨ RADAR Summary:")
        print(f"Symbol: {SYMBOL} | Bars: {len(df_feat)}")
        print(f"Score: {out['score']} | Grade: {out['grade']} | Suggestion: {out['suggestion']}")
        print(f"Guards: {out['guards']}")
        print(f"Tags:   {out['tags']}")
        if out['stop']:
            print(f"TP1={out['tp1']:.2f} | TP2={out['tp2']:.2f} | TP3={out['tp3']:.2f} | STOP={out['stop']:.2f}")

    except Exception as e:
        print(f"❌ Error fetching or processing {SYMBOL}: {e}")
