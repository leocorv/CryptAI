#!/usr/bin/env python3
"""CryptAI Binance OHLCV collector.

Public market-data collection works without API credentials. If BINANCE_API_KEY and
BINANCE_SECRET are present they are passed to CCXT, but they are not required here.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
from dotenv import load_dotenv

BASE = Path(os.getenv("CRYPTAI_HOME", Path(__file__).resolve().parents[1])).resolve()
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
TIMEFRAMES = {"1s": "1s", "15s": None, "1m": "1m", "1h": "1h", "4h": "4h", "1d": "1d"}


def resample_15s_from_1s(df_1s):
    """Resample 1-second OHLCV candles into 15-second candles."""
    df = df_1s.copy().set_index("timestamp")
    return (
        df.resample("15s")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def build_exchange():
    load_dotenv(BASE / ".env")
    config = {
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }

    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET")
    if api_key:
        config["apiKey"] = api_key
    if secret:
        config["secret"] = secret

    exchange = ccxt.binance(config)
    exchange.load_markets()
    return exchange


def fetch_frame(exchange, symbol, timeframe, limit):
    if timeframe not in (exchange.timeframes or {}):
        raise ValueError(f"timeframe '{timeframe}' is not supported by {exchange.id}")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def main():
    exchange = build_exchange()
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    report = {"ts": now.isoformat(), "fetched": []}

    for symbol in SYMBOLS:
        safe_symbol = symbol.replace("/", "")

        for tf_name, tf_code in TIMEFRAMES.items():
            dest = BASE / "data" / safe_symbol / tf_name / f"{date_str}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            limit = 1000 if tf_name in {"1s", "15s", "1m"} else (500 if tf_name in {"1h", "4h"} else 365)

            try:
                if tf_name == "15s":
                    # 15s is derived only when the exchange exposes native 1s candles.
                    df = resample_15s_from_1s(fetch_frame(exchange, symbol, "1s", 1000))
                else:
                    df = fetch_frame(exchange, symbol, tf_code, limit)

                df.to_csv(dest, index=False)
                report["fetched"].append(f"{safe_symbol}/{tf_name}: {len(df)}c -> {dest}")
                print(f"[OK] {safe_symbol}/{tf_name}: {len(df)} candles")
            except Exception as exc:
                print(f"[SKIP] {safe_symbol}/{tf_name}: {exc}")

            time.sleep(exchange.rateLimit / 1000)

    report_path = BASE / "data" / "_latest_fetch.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f"\n[DONE] {len(report['fetched'])} timeframes")


if __name__ == "__main__":
    main()
