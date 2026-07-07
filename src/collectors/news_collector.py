#!/usr/bin/env python3
"""CryptAI News & Sentiment Collector — MCP-style pipeline for crypto context"""
import os, sys, json, time
from datetime import datetime
from pathlib import Path
import requests

BASE = "/mnt/hive_storage/CryptAI"
NEWS_DIR = f"{BASE}/news"
CACHE_FILE = f"{NEWS_DIR}/_cache.json"

SYMBOL_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance", "bnb"],
    "HYP": ["hyperliquid", "hyp", "hype"],
    "XRP": ["xrp", "ripple"]
}

# Free crypto news APIs (no key required or free tier)
SOURCES = [
    {
        "name": "coindesk",
        "url": "https://api.coindesk.com/v1/bpi/currentprice.json",
        "type": "price_snapshot"
    },
    {
        "name": "coingecko_trending",
        "url": "https://api.coingecko.com/api/v3/search/trending",
        "type": "trending"
    },
]

def fetch_news():
    """Fetch news from available free sources"""
    results = []
    ts = datetime.utcnow().isoformat()
    
    for source in SOURCES:
        try:
            r = requests.get(source["url"], timeout=10)
            if r.status_code == 200:
                data = r.json()
                results.append({
                    "source": source["name"],
                    "type": source["type"],
                    "ts": ts,
                    "data": data
                })
                print(f"[news] {source['name']}: OK")
            else:
                print(f"[news] {source['name']}: HTTP {r.status_code}")
        except Exception as e:
            print(f"[news] {source['name']}: {e}")
        time.sleep(1)
    
    # Save
    date_str = datetime.utcnow().strftime('%Y%m%d_%H')
    outpath = f"{NEWS_DIR}/{date_str}.json"
    with open(outpath, "w") as f:
        json.dump({"ts": ts, "articles": results}, f, indent=2, default=str)
    
    return results

def query_news_for_symbol(symbol, since_hours=24):
    """Find news context for a symbol in recent cache"""
    results = []
    cutoff = time.time() - since_hours * 3600
    keywords = SYMBOL_KEYWORDS.get(symbol, [symbol.lower()])
    
    for f in sorted(Path(NEWS_DIR).glob("*.json")):
        if f.name == "_cache.json":
            continue
        if f.stat().st_mtime < cutoff:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
        except:
            continue
        
        for article in data.get("articles", []):
            text = json.dumps(article).lower()
            if any(kw in text for kw in keywords):
                results.append(article)
    
    return results

def check_abnormal_move(current_price, previous_price, threshold_pct=2.0):
    """Detect if a price move is abnormal"""
    if previous_price == 0:
        return False
    change = abs((current_price - previous_price) / previous_price) * 100
    return change >= threshold_pct, change

if __name__ == "__main__":
    print("[news] Fetching crypto context...")
    articles = fetch_news()
    print(f"[news] {len(articles)} sources polled")