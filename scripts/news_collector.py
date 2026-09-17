#!/usr/bin/env python3
"""CryptAI lightweight market-context collector."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = Path(os.getenv("CRYPTAI_HOME", Path(__file__).resolve().parents[1])).resolve()
NEWS_DIR = BASE / "news"
CACHE_FILE = NEWS_DIR / "_cache.json"

SYMBOL_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance", "bnb"],
    "HYP": ["hyperliquid", "hyp", "hype"],
    "XRP": ["xrp", "ripple"],
}

# Low-volume, keyless CoinGecko public API endpoints used only as experimental
# market context. They are intentionally not treated as a production news feed.
SOURCES = [
    {
        "name": "coingecko_trending",
        "url": "https://api.coingecko.com/api/v3/search/trending",
        "type": "trending",
    },
    {
        "name": "coingecko_price_snapshot",
        "url": (
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,solana,binancecoin,ripple"
            "&vs_currencies=usd&include_24hr_change=true"
        ),
        "type": "price_snapshot",
    },
]


def fetch_news():
    """Fetch experimental market context from configured public sources."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    ts = datetime.now(timezone.utc).isoformat()

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "CryptAI-experimental/1.0",
            "Accept": "application/json",
        })

        for source in SOURCES:
            try:
                response = session.get(source["url"], timeout=10)
                response.raise_for_status()
                results.append({
                    "source": source["name"],
                    "type": source["type"],
                    "ts": ts,
                    "data": response.json(),
                })
                print(f"[context] {source['name']}: OK")
            except (requests.RequestException, ValueError) as exc:
                print(f"[context] {source['name']}: {exc}")
            time.sleep(1)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    outpath = NEWS_DIR / f"{date_str}.json"
    with outpath.open("w", encoding="utf-8") as handle:
        json.dump({"ts": ts, "items": results}, handle, indent=2, default=str)

    return results


def query_news_for_symbol(symbol, since_hours=24):
    """Find recent cached context that mentions a tracked symbol."""
    if not NEWS_DIR.exists():
        return []

    results = []
    cutoff = time.time() - since_hours * 3600
    keywords = SYMBOL_KEYWORDS.get(symbol.upper(), [symbol.lower()])

    for file_path in sorted(NEWS_DIR.glob("*.json")):
        if file_path == CACHE_FILE or file_path.stat().st_mtime < cutoff:
            continue

        try:
            with file_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        # Support both the cleaned-up "items" key and older cache snapshots.
        cached_items = data.get("items", data.get("articles", []))
        for item in cached_items:
            text = json.dumps(item).lower()
            if any(keyword in text for keyword in keywords):
                results.append(item)

    return results


def check_abnormal_move(current_price, previous_price, threshold_pct=2.0):
    """Return (is_abnormal, absolute_change_pct)."""
    if previous_price == 0:
        return False, 0.0

    change = abs((current_price - previous_price) / previous_price) * 100
    return change >= threshold_pct, change


if __name__ == "__main__":
    print("[context] Fetching crypto market context...")
    items = fetch_news()
    print(f"[context] {len(items)} sources polled")
