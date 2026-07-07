#!/usr/bin/env python3
"""CryptAI Batch Training v2 — launches champion training serially"""
import os, sys, json, time, subprocess
from datetime import datetime

BASE = "/mnt/hive_storage/CryptAI"
CKPT_DIR = f"{BASE}/training/checkpoints"
LOG_DIR = f"{BASE}/training/logs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CONFIGS = [
    {"model": "lstm", "sl": 48, "hd": 128, "ep": 100, "bs": 128, "lr": 1e-3},
    {"model": "lstm", "sl": 96, "hd": 256, "ep": 100, "bs": 64, "lr": 5e-4},
    {"model": "lstm", "sl": 192, "hd": 128, "ep": 80, "bs": 64, "lr": 1e-3},
    {"model": "transformer", "sl": 48, "hd": 128, "ep": 100, "bs": 128, "lr": 1e-3},
    {"model": "transformer", "sl": 96, "hd": 256, "ep": 100, "bs": 64, "lr": 5e-4},
    {"model": "transformer", "sl": 192, "hd": 128, "ep": 80, "bs": 64, "lr": 1e-3},
    {"model": "mlp", "sl": 96, "hd": 256, "ep": 100, "bs": 256, "lr": 1e-3},
    {"model": "lstm", "sl": 96, "hd": 128, "ep": 150, "bs": 128, "lr": 1e-3},
    {"model": "transformer", "sl": 96, "hd": 128, "ep": 150, "bs": 128, "lr": 1e-3},
    {"model": "lstm", "sl": 48, "hd": 256, "ep": 120, "bs": 64, "lr": 5e-4},
]

def main():
    print(f"CryptAI Batch Training v2 — {len(CONFIGS)} configs")
    gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"GPU: {gpu}")
    
    for i, cfg in enumerate(CONFIGS):
        tag = f"{cfg['model']}_sl{cfg['sl']}_h{cfg['hd']}_e{cfg['ep']}_{datetime.utcnow().strftime('%H%M%S')}"
        logfile = f"{LOG_DIR}/{tag}.log"
        ckpt = f"{CKPT_DIR}/{tag}.pth"
        
        print(f"\n[{i+1}/{len(CONFIGS)}] {tag}")
        cmd = f"{BASE}/venv/bin/python3 {BASE}/scripts/train_single.py {cfg["model"]} {cfg["sl"]} {cfg["hd"]} {cfg["ep"]} {cfg["bs"]} {cfg["lr"]} {ckpt}"
        
        with open(logfile, "w") as f:
            f.write(f"# Training {tag}\n# Config: {json.dumps(cfg)}\n# Started: {datetime.utcnow().isoformat()}\n")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        with open(logfile, "a") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        
        print(f"  Exit: {result.returncode} | Out: {result.stdout[-200:]}")
    
    print(f"\nDone: {len(CONFIGS)} configs")
    # Report top 5
    results = []
    for f in sorted(os.listdir(CKPT_DIR)):
        if f.endswith(".pth"):
            try:
                data = torch.load(f"{CKPT_DIR}/{f}", map_location="cpu")
                results.append((data.get("val_acc", 0), data.get("config", {}), f))
            except: pass
    results.sort(reverse=True)
    print("\nTop 5:")
    for i, (acc, cfg, fn) in enumerate(results[:5]):
        print(f"  {i+1}. {fn} — Acc: {acc:.3f} | {cfg.get('model','?')}")

if __name__ == "__main__":
    import torch
    main()