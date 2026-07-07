#!/usr/bin/env python3
"""Batcher — trains LSTM (fits GPU), then Transformer, then TCN. Sequential, full GPU each."""
import os,sys,subprocess,time
BASE="/mnt/hive_storage/CryptAI"
VENV=f"{BASE}/venv/bin/python3"
SCRIPTS=[
    (f"{VENV} {BASE}/scripts/deep_lstm_train.py",f"{BASE}/training/logs/lstm_sat.log","LSTM"),
    (f"{VENV} {BASE}/scripts/deep_transformer_train.py",f"{BASE}/training/logs/transformer_sat.log","TFR"),
    (f"{VENV} {BASE}/scripts/tcn_train.py",f"{BASE}/training/logs/tcn_sat.log","TCN"),
]
for cmd,log,name in SCRIPTS:
    print(f"\n=== {name} ===",flush=True)
    with open(log,"w") as f:
        p=subprocess.Popen(cmd,shell=True,stdout=f,stderr=f)
    while p.poll() is None:
        r=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader"],capture_output=True,text=True)
        print(f"  [{name}] GPU: {r.stdout.strip()}",flush=True)
        time.sleep(30)
    print(f"  {name} exit={p.returncode}",flush=True)
print("\nAll done",flush=True)