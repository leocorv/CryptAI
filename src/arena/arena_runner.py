#!/usr/bin/env python3
"""CryptAI Arena Runner — loads best champions and simulates trades"""
import os, sys, json, glob, time
import pandas as pd
import numpy as np
import torch

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib")
from features import build_features
from models import LSTMDirection, TransformerDirection, MLPDirection
from arena import Arena

CKPT_DIR = f"{BASE}/training/checkpoints"
ARENA_DIR = f"{BASE}/arena"

def load_champion(path):
    data = torch.load(path, map_location="cpu")
    cfg = data["config"]
    input_dim = 25  # from features.py
    
    if cfg["model"] == "lstm":
        m = LSTMDirection(input_dim, cfg.get("hidden_dim", 128), seq_len=cfg.get("seq_len", 96))
    elif cfg["model"] == "transformer":
        m = TransformerDirection(input_dim, cfg.get("hidden_dim", 128), seq_len=cfg.get("seq_len", 96))
    else:
        m = MLPDirection(input_dim * cfg.get("seq_len", 96))
    
    m.load_state_dict(data["state"])
    m.eval()
    return m, cfg

def champion_signal(model, cfg, row):
    """Generate signal: 0=sell, 1=hold, 2=buy"""
    seq_len = cfg.get("seq_len", 96)
    # Need sequence - this is called with single rows in arena
    # For now: return 1 (hold) as placeholder
    # Real implementation needs sequence buffer
    return 1

def run_arena(champion_paths):
    """Run all champions through the arena"""
    results = {}
    test_dfs = {}
    
    # Load test data (most recent 4h)
    for f in sorted(glob.glob(f"{BASE}/data/*/4h/*.csv"))[:1]:  # 1 symbol for now
        test_dfs["test"] = pd.read_csv(f, parse_dates=["timestamp"])
    
    for cpath in champion_paths[:5]:  # Top 5
        model, cfg = load_champion(cpath)
        name = os.path.basename(cpath).replace(".pth", "")
        
        # Build features for test data
        feats = build_features(test_dfs["test"])
        if feats.empty or len(feats) < cfg.get("seq_len", 96):
            print(f"  SKIP {name}: insufficient features ({len(feats)})")
            continue
        
        # Normalize
        vals = feats.values.astype(np.float32)
        for i in range(len(vals)):
            m = vals[i].mean(); s = vals[i].std() + 1e-8
            vals[i] = (vals[i] - m) / s
        
        # Generate signals
        seq_len = cfg.get("seq_len", 96)
        prices = test_dfs["test"]["close"].values[-len(feats):]
        signals = np.zeros(len(vals))
        positions = []  # Track for PnL
        
        for i in range(seq_len, len(vals)):
            x = torch.FloatTensor(vals[i-seq_len:i]).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                pred = out.argmax(1).item()
            signals[i] = pred
            
            # Simulate trade
            price = prices[i]
            if pred == 2:  # buy
                positions.append(("buy", price, i))
            elif pred == 0 and positions:  # sell
                buy_price = positions[-1][1]
                pnl = (price - buy_price) / buy_price * 100
                positions[-1] = (*positions[-1], pnl, price)
        
        # Arena metrics
        arena = Arena(initial_balance=1000.0)
        entry_price = None
        direction = 0
        for i in range(seq_len, len(vals)):
            price = prices[i]
            sig = signals[i]
            if sig == 2 and arena.position is None:
                arena.open_trade("ARENA", price, 1)
            elif sig == 0 and arena.position is not None:
                arena.close_trade(price)
        
        if arena.position is not None:
            arena.close_trade(prices[-1])
        
        metrics = arena.get_metrics()
        results[name] = metrics
        print(f"  {name}: Return={metrics['total_return']:.2%} | PF={metrics['profit_factor']:.2f} | DD={metrics['max_drawdown']:.2%} | Sharpe={metrics['sharpe']:.2f} | WR={metrics['win_rate']:.2%}")
        
        # Save champion to arena
        dest = f"{ARENA_DIR}/champions/{name}.json"
        with open(dest, "w") as f:
            json.dump({"config": cfg, "results": metrics, "path": cpath}, f, indent=2)
    
    return results

def main():
    os.makedirs(f"{ARENA_DIR}/champions", exist_ok=True)
    os.makedirs(f"{ARENA_DIR}/results", exist_ok=True)
    
    # Get best checkpoints (sorted by mtime = newest)
    checkpoints = sorted(glob.glob(f"{CKPT_DIR}/*.pth"), key=os.path.getmtime, reverse=True)
    print(f"Arena: {len(checkpoints)} champions available")
    
    results = run_arena(checkpoints[:5])
    
    # Save dashboard
    dashboard = {
        "ts": str(pd.Timestamp.utcnow()),
        "champions": len(results),
        "top_return": max((r["total_return"] for r in results.values()), default=0),
        "details": results
    }
    with open(f"{ARENA_DIR}/dashboard/latest.json", "w") as f:
        json.dump(dashboard, f, indent=2, default=str)
    
    print(f"\nDashboard: {ARENA_DIR}/dashboard/latest.json")
    return results

if __name__ == "__main__":
    main()