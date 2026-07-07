#!/usr/bin/env python3
"""CryptAI Historical Massive — fetch 48h 1s + 30d 1m per symbol"""
import ccxt, os, time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

BASE = "/mnt/hive_storage/CryptAI"
exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
SYMBOLS = ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"]
TFS = {"1s": "1s", "1m": "1m", "15s": None}
LIMITS = {"1s": 1000, "1m": 1000}
HOURS = {"1s": 48, "1m": 720}

def main():
    total = 0
    for sym in SYMBOLS:
        safe = sym.replace("/", "")
        for tf_name, tf_code in TFS.items():
            limit = LIMITS.get(tf_name, 500)
            hours = HOURS.get(tf_name, 1)
            step_ms = 1000 if tf_name == "1s" else 60000
            n_batches = int((hours * 3600 * 1000) / (limit * step_ms))
            since = exchange.parse8601((datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ"))
            rows = 0
            dest = Path(f"{BASE}/data/{safe}/{tf_name}/historical.csv")
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            for b in range(n_batches):
                try:
                    actual_tf = "1s" if tf_name == "15s" else tf_code
                    ohlcv = exchange.fetch_ohlcv(sym, actual_tf, since=since, limit=limit)
                    if not ohlcv:
                        break
                    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    
                    if tf_name == "15s":
                        df = df.set_index("timestamp").resample("15s").agg({
                            "open": "first", "high": "max", "low": "min",
                            "close": "last", "volume": "sum"
                        }).dropna().reset_index()
                    
                    df.to_csv(dest, mode="a" if dest.exists() else "w",
                              header=not dest.exists(), index=False)
                    rows += len(df)
                    since = ohlcv[-1][0] + 1
                    
                    if b % 20 == 0:
                        print(f"  {safe}/{tf_name}: batch {b+1}/{n_batches} ({rows} rows)", flush=True)
                    time.sleep(0.1)
                except Exception as e:
                    if b % 50 == 0:
                        print(f"  {safe}/{tf_name} batch {b}: {e}", flush=True)
                    time.sleep(3)
            
            total += rows
            print(f"  DONE {safe}/{tf_name}: {rows} rows", flush=True)
    
    print(f"\nTOTAL: {total} rows across all symbols/timeframes", flush=True)

if __name__ == "__main__":
    main()