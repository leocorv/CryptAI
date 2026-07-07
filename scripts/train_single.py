#!/usr/bin/env python3
"""CryptAI Single Champion Trainer — called by batch_train.py"""
import sys, os, json, time, glob
import pandas as pd
import numpy as np
import torch

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib")
from features import build_features, normalize
from models import LSTMDirection, TransformerDirection, MLPDirection

def train_champion(model_type, seq_len, hidden_dim, epochs, batch_size, lr, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data = {}
    for f in sorted(glob.glob(f"{BASE}/data/*/1h/*.csv")):
        try:
            sym = f.split("/")[-3]
            data[sym] = pd.read_csv(f, parse_dates=["timestamp"])
        except Exception as e:
            print(f"  SKIP {f}: {e}")
    
    if not data:
        print("NO DATA LOADED - trying alt paths")
        # Try direct path
        for f in sorted(glob.glob(f"{BASE}/data/*/1h/*.csv")):
            print(f"  FOUND: {f}")
    
    all_X, all_y = [], []
    for sym, df in data.items():
        feats = build_features(df)
        if len(feats) < seq_len + 10:
            continue
        close = df["close"].values[-len(feats):]
        future_ret = np.diff(close, prepend=close[0]) / (close + 1e-8)
        labels = np.where(future_ret < -0.002, 0, np.where(future_ret > 0.002, 2, 1))
        vals = feats.values.astype(np.float32)
        for i in range(len(vals) - seq_len):
            all_X.append(vals[i:i+seq_len])
            all_y.append(int(labels[i+seq_len]))
    
    if not all_X:
        print("ERROR: No training data generated")
        return None
    
    X = np.array(all_X)
    y = np.array(all_y)
    print(f"Data loaded: {len(X)} sequences, input_dim={X.shape[2]}, classes={len(set(y))}")
    
    # Normalize per-sample
    for i in range(len(X)):
        m = X[i].mean(axis=0, keepdims=True)
        s = X[i].std(axis=0, keepdims=True) + 1e-8
        X[i] = (X[i] - m) / s
    
    split = int(len(X) * 0.8)
    perm = np.random.permutation(len(X))
    X_train, X_val = X[perm[:split]], X[perm[split:]]
    y_train, y_val = y[perm[:split]], y[perm[split:]]
    
    input_dim = X.shape[2]
    if model_type == "lstm":
        model = LSTMDirection(input_dim, hidden_dim, seq_len=seq_len)
    elif model_type == "transformer":
        model = TransformerDirection(input_dim, hidden_dim, seq_len=seq_len)
    else:
        model = MLPDirection(input_dim * seq_len)
    
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    
    for ep in range(epochs):
        p = torch.randperm(len(X_train))
        total_loss = 0
        steps = 0
        for i in range(0, len(p), batch_size):
            idx = p[i:i+batch_size]
            xb = torch.FloatTensor(X_train[idx]).to(device)
            yb = torch.LongTensor(y_train[idx]).to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1
        
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                xv = torch.FloatTensor(X_val).to(device)
                yv = torch.LongTensor(y_val).to(device)
                val_loss = crit(model(xv), yv).item()
                acc = (model(xv).argmax(1) == yv).float().mean().item()
            model.train()
            elapsed = time.time() - t0
            print(f"EP {ep:>3}/{epochs} | Loss: {total_loss/steps:.4f} | VLoss: {val_loss:.4f} | Acc: {acc:.3f}  | {elapsed:.0f}s")
    
    # Save
    save_data = {
        "state": model.state_dict(),
        "config": {"model": model_type, "seq_len": seq_len, "hidden_dim": hidden_dim,
                   "epochs": epochs, "batch_size": batch_size, "lr": lr},
        "val_acc": acc,
        "val_loss": val_loss,
    }
    torch.save(save_data, save_path)
    print(f"Saved: {save_path} | ValAcc: {acc:.3f}")
    return save_data

if __name__ == "__main__":
    model_type = sys.argv[1]
    seq_len = int(sys.argv[2])
    hidden_dim = int(sys.argv[3])
    epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    lr = float(sys.argv[6])
    save_path = sys.argv[7]
    train_champion(model_type, seq_len, hidden_dim, epochs, batch_size, lr, save_path)