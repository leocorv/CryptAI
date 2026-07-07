#!/usr/bin/env python3
"""CryptAI — HyperLiquid data collector (ccxt)"""
import os, sys, time, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import ccxt

BASE = "/mnt/hive_storage/CryptAI"
HL_SYMBOL = "HYPE/USDC"
HL_TIMEFRAMES = {"1s": "1s", "15s": "15s", "1m": "1m", "1h": "1h", "4h": "4h", "1d": "1d"}

def collect_hyperliquid():
    exchange = ccxt.hyperliquid({"enableRateLimit": True})
    date_str = datetime.utcnow().strftime('%Y%m%d')
    results = []
    
    for tf_name, tf_code in HL_TIMEFRAMES.items():
        dest = Path(f"data/HYP/{tf_name}/{date_str}.csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        limit = 1000 if tf_code in ["1s", "15s", "1m"] else (500 if tf_code in ["1h","4h"] else 365)
        try:
            if tf_name == "15s":
                ohlcv = exchange.fetch_ohlcv(HL_SYMBOL, "1s", limit=1000)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp").resample("15s").agg({
                    "open":"first","high":"max","low":"min","close":"last","volume":"sum"
                }).dropna().reset_index()
            else:
                ohlcv = exchange.fetch_ohlcv(HL_SYMBOL, tf_code, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            df.to_csv(dest, index=False)
            results.append(f"HYP/{tf_name}: {len(df)}c")
            print(f"[OK] HYP/{tf_name}: {len(df)} candles")
            time.sleep(1)
        except Exception as e:
            print(f"[SKIP] HYP/{tf_name}: {e}")
    
    print(f"[HYP] {len(results)} timeframes collected")

if __name__ == "__main__":
    collect_hyperliquid()