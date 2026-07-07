#!/usr/bin/env python3
"""CryptAI Symbol Manager — add/remove tracking with audit log"""
import os, sys, json
from datetime import datetime

BASE = "/mnt/hive_storage/CryptAI"
CONFIG_FILE = f"{BASE}/config/symbols.json"
LOG_FILE = f"{BASE}/config/symbols.log"

BASE_SYMBOLS = {"BTC", "ETH", "SOL", "BNB", "HYP", "XRP"}

def load():
    with open(CONFIG_FILE) as f:
        return json.load(f)

def save(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def log(action, symbol, reason):
    entry = json.dumps({"ts": datetime.utcnow().isoformat(), "action": action,
                        "symbol": symbol, "reason": reason})
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")
    print(f"[symbols] {action}: {symbol} — {reason}")

def add_symbol(symbol, reason=""):
    cfg = load()
    sym = symbol.upper()
    if sym in cfg["base"]:
        log("BLOCKED", sym, "symbol de base — suppression interdite")
        return False
    if sym in cfg["added"]:
        log("SKIP", sym, "déjà ajouté")
        return False
    cfg["added"].append(sym)
    save(cfg)
    log("ADD", sym, reason)
    return True

def remove_symbol(symbol, reason=""):
    cfg = load()
    sym = symbol.upper()
    if sym in cfg["base"]:
        log("BLOCKED", sym, "symbol de base — suppression interdite")
        return False
    if sym not in cfg["added"]:
        log("SKIP", sym, "pas dans la liste des ajoutés")
        return False
    cfg["added"].remove(sym)
    cfg["removed"].append({"symbol": sym, "ts": datetime.utcnow().isoformat(), "reason": reason})
    save(cfg)
    log("REMOVE", sym, reason)
    return True

def get_all_symbols():
    cfg = load()
    return cfg["base"] + cfg["added"]

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "list"
    if action == "list":
        cfg = load()
        print(f"Base: {cfg['base']}")
        print(f"Added: {cfg['added']}")
        print(f"Removed: {cfg['removed']}")
    elif action == "add" and len(sys.argv) >= 3:
        reason = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "manual add"
        add_symbol(sys.argv[2], reason)
    elif action == "remove" and len(sys.argv) >= 3:
        reason = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "manual remove"
        remove_symbol(sys.argv[2], reason)