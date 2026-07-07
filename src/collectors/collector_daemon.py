#!/usr/bin/env python3
"""CryptAI Collector Daemon — runs forever, appends data"""
import ccxt, os, time, json, subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

BASE = "/mnt/hive_storage/CryptAI"
exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
SYMBOLS = ['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT']
TFS = {'1s':'1s','15s':'15s','1m':'1m','1h':'1h','4h':'4h','1d':'1d'}
LIMITS = {'1s':1000,'15s':1000,'1m':1000,'1h':500,'4h':500,'1d':365}

def main():
    print(f"[collector_daemon] PID={os.getpid()} — collecting every 60min")
    while True:
        dt = datetime.utcnow().strftime('%Y%m%d_%H%M')
        total = 0
        for sym in SYMBOLS:
            safe = sym.replace('/','')
            for tf_name, tf_code in TFS.items():
                limit = LIMITS.get(tf_name, 500)
                try:
                    if tf_name == '15s':
                        ohlcv = exchange.fetch_ohlcv(sym, '1s', limit=1000)
                        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df.set_index('timestamp').resample('15s').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
                    else:
                        ohlcv = exchange.fetch_ohlcv(sym, tf_code, limit=limit)
                        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    dest = Path(f'data/{safe}/{tf_name}/{dt}.csv')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(dest, index=False)
                    total += len(df)
                except Exception as e:
                    print(f"  {safe}/{tf_name}: {e}")
                time.sleep(1)
        
        # Also append to accumulative file
        with open(f'{BASE}/data/_stats.json', 'w') as f:
            json.dump({"ts": dt, "rows_collected": total}, f)
        
        gpu = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        print(f"[collector] {dt}: {total} rows | GPU: {gpu}", flush=True)
        time.sleep(3600)

if __name__ == "__main__":
    main()