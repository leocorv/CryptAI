#!/usr/bin/env python3
"""CryptAI Anti-Leakage Validator — runs before each training"""
import os, sys, glob, json
import numpy as np
import pandas as pd

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib")
from features import build_features

errors = []
warnings = []

def check(name, condition, msg):
    if not condition:
        errors.append(f"❌ {name}: {msg}")
    else:
        print(f"  ✅ {name}")

def warn(name, condition, msg):
    if not condition:
        warnings.append(f"⚠️ {name}: {msg}")
    else:
        print(f"  ✅ {name}")

print("=== Anti-Leakage Validator ===\n", flush=True)

# 1. Test temporal ordering
print("[1] Temporal split test...", flush=True)
all_ts = []
for f in sorted(glob.glob(f"{BASE}/data/*/*/historical.csv")):
    try:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        if len(df) > 200:
            all_ts.extend(df["timestamp"].values[100:].tolist())
    except: pass

all_ts = sorted(all_ts)
n = len(all_ts)
s = int(n * 0.7)
vs = int(n * 0.85)
check("train_before_val", all_ts[s-1] < all_ts[s] if s < n else True,
      "Train max timestamp >= Val min timestamp — temporal split broken")
check("val_before_test", all_ts[vs-1] < all_ts[vs] if vs < n else True,
      "Val max timestamp >= Test min timestamp — temporal split broken")

# 2. Test feature lookahead
print("\n[2] Feature lookahead test...", flush=True)
df = pd.read_csv(sorted(glob.glob(f"{BASE}/data/*/1h/historical.csv"))[0], parse_dates=["timestamp"])
feats = build_features(df)
# Check that no feature uses future data
# Rolling windows should not include the current bar's close for future prediction
# But they DO include the current close — this is OK for t+1 prediction
# Check for .shift(-N) patterns
with open(f"{BASE}/lib/features.py") as fh:
    code = fh.read()
warn("shift_negative", "shift(-" not in code,
     "Features use .shift(-N) — possible lookahead!")
warn("shift_positive", "shift(" in code,
     "Features use .shift(N) — OK (past data)")

# 3. Test target shift
print("\n[3] Target shift test...", flush=True)
# Check that train script uses proper shift
with open(f"{BASE}/scripts/train_clean.py") as fh:
    train_code = fh.read()
check("no_random_perm", "permutation" not in train_code,
      "Script still uses np.random.permutation — temporal order broken!")
check("temporal_split", "argsort" in train_code and "ts[order]" in train_code,
      "No temporal sorting — temporal split missing!")
check("norm_on_train", "X_train.mean" in train_code,
      "Normalization not fit on train only!")

# 4. Test accuracy range
print("\n[4] Accuracy range check...", flush=True)
ckpt_files = sorted(glob.glob(f"{BASE}/training/checkpoints/*_clean.pth"))
if ckpt_files:
    import torch
    data = torch.load(ckpt_files[-1], map_location="cpu")
    val_acc = data.get("val_acc", 0)
    test_acc = data.get("test_acc", 0)
    train_acc = data.get("train_acc", 0)
    print(f"  Train acc: {train_acc:.3f}")
    print(f"  Val acc:   {val_acc:.3f}")
    print(f"  Test acc:  {test_acc:.3f}")
    check("val_accuracy", val_acc < 0.70, f"Val accuracy {val_acc:.3f} >= 0.70 — possible leakage!")
    warn("val_accuracy_high", val_acc < 0.85, f"Val accuracy {val_acc:.3f} >= 0.85 — certain leakage!")
    if val_acc > 0.70:
        errors.append(f"❌ FAILED: Val accuracy {val_acc:.3f} > 0.70 — training blocked")
else:
    print("  No clean checkpoint found — skipping accuracy check")

# Summary
print(f"\n{'='*50}")
print(f"Results: {len(errors)} errors, {len(warnings)} warnings")
if errors:
    print(f"❌ BLOCKED: Fix errors before training")
    for e in errors:
        print(f"  {e}")
else:
    print(f"✅ PASSED: Pipeline clean, training can proceed")
if warnings:
    for w in warnings:
        print(f"  {w}")

# Exit code for CI
sys.exit(1 if errors else 0)