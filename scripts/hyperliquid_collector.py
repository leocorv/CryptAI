#!/usr/bin/env python3
"""CryptAI Hyperliquid OHLCV collector."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

BASE = Path(os.getenv("CRYPTAI_HOME", Path(__file__).resolve().parents[1])).resolve()
HL_SYMBOL = "HYPE/USDC"
HL_TIMEFRAMES = {"1s": "1s", "15s": None, "1m": "1m", "1h": "1h", "4h": "4h", "1d": "1d"}


def fetch_frame(exchange, timeframe, limit):
    if timeframe not in (exchange.timeframes or {}):
        raise ValueError(f"timeframe '{timeframe}' is not supported by {exchange.id}")

    ohlcv = exchange.fetch_ohlcv(HL_SYMBOL, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def resample_15s_from_1s(df):
    return (
        df.set_index("timestamp")
        .resample("15s")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def collect_hyperliquid():
    exchange = ccxt.hyperliquid({"enableRateLimit": True})
    exchange.load_markets()

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    results = []

    for tf_name, tf_code in HL_TIMEFRAMES.items():
        dest = BASE / "data" / "HYP" / tf_name / f"{date_str}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        limit = 1000 if tf_name in {"1s", "15s", "1m"} else (500 if tf_name in {"1h", "4h"} else 365)

        try:
            if tf_name == "15s":
                df = resample_15s_from_1s(fetch_frame(exchange, "1s", 1000))
            else:
                df = fetch_frame(exchange, tf_code, limit)

            df.to_csv(dest, index=False)
            results.append(f"HYP/{tf_name}: {len(df)}c")
            print(f"[OK] HYP/{tf_name}: {len(df)} candles")
        except Exception as exc:
            print(f"[SKIP] HYP/{tf_name}: {exc}")

        time.sleep(exchange.rateLimit / 1000)

    print(f"[HYP] {len(results)} timeframes collected")


if __name__ == "__main__":
    collect_hyperliquid()
