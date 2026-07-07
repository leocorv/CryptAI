#!/usr/bin/env python3
"""CryptAI WebSocket Binance — 1s klines in real-time, daemon"""
import os, sys, json, time, asyncio, signal
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE = "/mnt/hive_storage/CryptAI"
SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt"]
STREAMS = [f"{s}@kline_1s" for s in SYMBOLS]
URL = f"wss://stream.binance.com:9443/stream?streams={'/'.join(STREAMS)}"

try:
    import websockets
except ImportError:
    os.system(f"{BASE}/venv/bin/pip install websockets -q")
    import websockets

buffer = {s: [] for s in SYMBOLS}
BUF_LIMIT = 1000
PID = os.getpid()

async def process_kline(symbol, data):
    k = data["k"]
    ts = datetime.utcfromtimestamp(k["t"] / 1000)
    row = [ts.isoformat(), float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"])]
    buffer[symbol].append(row)
    
    if len(buffer[symbol]) >= BUF_LIMIT:
        dt = ts.strftime("%Y%m%d_%H")
        dest = Path(f"{BASE}/data/{symbol.upper()}/1s/{dt}.csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(buffer[symbol], columns=["timestamp","open","high","low","close","volume"])
        df.to_csv(dest, mode="a", header=not dest.exists(), index=False)
        print(f"[ws] {symbol}: {len(buffer[symbol])} candles flushed to {dest}", flush=True)
        buffer[symbol] = []

async def handler():
    print(f"[ws_daemon] PID={PID} — streaming {len(SYMBOLS)} symbols @ 1s", flush=True)
    while True:
        try:
            async with websockets.connect(URL) as ws:
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("stream") and data.get("data", {}).get("e") == "kline":
                        sym = data["stream"].split("@")[0]
                        await process_kline(sym, data["data"])
        except Exception as e:
            print(f"[ws] Reconnecting in 5s: {e}", flush=True)
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(handler())
    except KeyboardInterrupt:
        print("[ws] Shutdown", flush=True)