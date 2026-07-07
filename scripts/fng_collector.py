#!/usr/bin/env python3
"""CryptAI Fear & Greed + News fallback — Phase 3 infra"""
import os, sys, json, time, requests
from datetime import datetime

BASE = "/mnt/hive_storage/CryptAI"
NEWS_DIR = f"{BASE}/news"

def fear_greed_index():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {"index": int(data["data"][0]["value"]),
                    "classification": data["data"][0]["value_classification"],
                    "timestamp": data["data"][0]["timestamp"]}
    except:
        pass
    return None

def fetch_coindesk():
    """Coindesk RSS fallback with retry + backoff"""
    for attempt in range(3):
        try:
            r = requests.get("https://www.coindesk.com/arc/outboundfeeds/rss/", timeout=10)
            if r.status_code == 200:
                return r.text[:5000]
        except:
            time.sleep(2 ** attempt)
    return None

def store_snapshot():
    record = {"ts": datetime.utcnow().isoformat()}
    fg = fear_greed_index()
    if fg:
        record["fear_greed"] = fg
        print(f"[fng] Value: {fg['index']} ({fg['classification']})")
    else:
        print("[fng] Unavailable")
    
    cd = fetch_coindesk()
    if cd:
        record["coindesk"] = cd[:1000]
        print(f"[coindesk] RSS fetched ({len(cd)} chars)")
    
    outpath = f"{NEWS_DIR}/fng_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(outpath, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[news] Snapshot saved: {outpath}")

if __name__ == "__main__":
    store_snapshot()