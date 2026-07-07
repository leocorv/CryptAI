#!/usr/bin/env python3
"""CryptAI Data Collector v2 — Binance OHLCV multi-TF + 15s via resample"""
import os, sys, time, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import ccxt

BASE = "/mnt/hive_storage/CryptAI"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
# HYP via HyperLiquid API à part
TIMEFRAMES = {"1s": "1s", "15s": None, "1m": "1m", "1h": "1h", "4h": "4h", "1d": "1d"}
# 15s = None car on le resample depuis 1s

def resample_15s_from_1s(df_1s):
    """Resample 1s OHLCV to 15s candles"""
    df = df_1s.copy()
    df = df.set_index("timestamp")
    resampled = df.resample("15s").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    return resampled.reset_index()

def main():
    os.chdir(BASE)
    from dotenv import load_dotenv
    load_dotenv('.env')
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET")
    
    exchange = ccxt.binance({
        'apiKey': api_key, 'secret': secret,
        'enableRateLimit': True, 'options': {'defaultType': 'spot'}
    })
    
    date_str = datetime.utcnow().strftime('%Y%m%d')
    report = {"ts": datetime.utcnow().isoformat(), "fetched": []}
    
    for sym in SYMBOLS:
        safe_sym = sym.replace("/", "")
        for tf_name, tf_code in TIMEFRAMES.items():
            dest = Path(f"data/{safe_sym}/{tf_name}/{date_str}.csv")
            dest.parent.mkdir(parents=True, exist_ok=True)
            limit = 1000 if tf_code in ["1s", "1m", "15s"] else (500 if tf_code in ["1h","4h"] else 365)
            
            try:
                if tf_name == "15s":
                    # Fetch 1s and resample
                    ohlcv = exchange.fetch_ohlcv(sym, "1s", limit=1000)
                    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = resample_15s_from_1s(df)
                else:
                    since = exchange.parse8601(f"{datetime.utcnow().timestamp() - 86400*7:.0f}000")
                    ohlcv = exchange.fetch_ohlcv(sym, tf_code, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                
                df.to_csv(dest, index=False)
                report["fetched"].append(f"{safe_sym}/{tf_name}: {len(df)}c -> {dest}")
                print(f"[OK] {safe_sym}/{tf_name}: {len(df)} candles")
                time.sleep(exchange.rateLimit / 1000)
            except Exception as e:
                print(f"[SKIP] {safe_sym}/{tf_name}: {e}")
    
    with open("data/_latest_fetch.json","w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[DONE] {len(report['fetched'])} timeframes")

if __name__ == "__main__":
    main()