#!/usr/bin/env python3
"""CryptAI symbol manager with a small JSON audit trail."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(os.getenv("CRYPTAI_HOME", Path(__file__).resolve().parents[1])).resolve()
CONFIG_FILE = BASE / "config" / "symbols.json"
LOG_FILE = BASE / "config" / "symbols.log"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing symbol config: {CONFIG_FILE}")
    with CONFIG_FILE.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    cfg.setdefault("base", [])
    cfg.setdefault("added", [])
    cfg.setdefault("removed", [])
    return cfg


def save(cfg):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)


def log(action, symbol, reason):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({"ts": now_iso(), "action": action, "symbol": symbol, "reason": reason})
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")
    print(f"[symbols] {action}: {symbol} — {reason}")


def add_symbol(symbol, reason=""):
    cfg = load()
    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("symbol cannot be empty")
    if sym in cfg["base"]:
        log("SKIP", sym, "already part of the base symbol set")
        return False
    if sym in cfg["added"]:
        log("SKIP", sym, "already added")
        return False

    cfg["added"].append(sym)
    save(cfg)
    log("ADD", sym, reason)
    return True


def remove_symbol(symbol, reason=""):
    cfg = load()
    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("symbol cannot be empty")
    if sym in cfg["base"]:
        log("BLOCKED", sym, "base symbols cannot be removed")
        return False
    if sym not in cfg["added"]:
        log("SKIP", sym, "not present in the added symbol set")
        return False

    cfg["added"].remove(sym)
    cfg["removed"].append({"symbol": sym, "ts": now_iso(), "reason": reason})
    save(cfg)
    log("REMOVE", sym, reason)
    return True


def get_all_symbols():
    cfg = load()
    return cfg["base"] + cfg["added"]


def print_usage():
    print("Usage: symbol_manager.py [list | add SYMBOL [reason...] | remove SYMBOL [reason...]]")


def main():
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "list"

    if action == "list":
        cfg = load()
        print(f"Base: {cfg['base']}")
        print(f"Added: {cfg['added']}")
        print(f"Removed: {cfg['removed']}")
        return

    if action in {"add", "remove"}:
        if len(sys.argv) < 3:
            print_usage()
            raise SystemExit(2)
        reason = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else f"manual {action}"
        if action == "add":
            add_symbol(sys.argv[2], reason)
        else:
            remove_symbol(sys.argv[2], reason)
        return

    print_usage()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
