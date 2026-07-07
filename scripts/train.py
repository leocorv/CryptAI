#!/usr/bin/env python3
"""CryptAI Training Runner — GPU, permanent, champion generation"""
import os, sys, json, time, hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib")
from features import build_features, normalize
from models import LSTMDirection, TransformerDirection, MLPDirection

def load_latest_data():
    """Load latest CSV for all symbols/timeframes into a dict"""
    data = {}
    base = Path(f"{BASE}/data")
    for sym_dir in base.iterdir():
        if not sym_dir.is_dir() or sym_dir.name in ["_latest_fetch.json"]:
            continue
        for tf_dir in sym_dir.iterdir():
            csvs = sorted(tf_dir.glob("*.csv"))
            if csvs:
                df = pd.read_csv(csvs[-1], parse_dates=["timestamp"])
                data[f"{sym_dir.name}_{tf_dir.name}"] = df
    return data

def create_sequences(features: np.ndarray, labels: np.ndarray, seq_len=96):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)

def compute_labels(df: pd.DataFrame, horizon=5):
    """Direction labels: 0=sell, 1=hold, 2=buy based on future return"""
    future_ret = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.cut(future_ret, bins=[-np.inf, -0.002, 0.002, np.inf], labels=[0, 1, 2])
    return labels.astype(int)

def generate_champion_id(config: dict) -> str:
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

def train_champion(model_type="lstm", seq_len=96, hidden_dim=128, batch_size=64,
                   epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device} | Model: {model_type} | Epochs: {epochs}")
    
    # Load data
    data = load_latest_data()
    all_X, all_y = [], []
    for key, df in data.items():
        f = build_features(df)
        labels = compute_labels(df)
        f = f.iloc[:len(labels)]
        labels = labels.iloc[-len(f):]
        vals = f.values.astype(np.float32)
        labs = labels.values
        if len(vals) < seq_len + 10:
            continue
        X, y = create_sequences(vals, labs, seq_len)
        all_X.append(X)
        all_y.append(y)
        print(f"  {key}: {len(X)} sequences")
    
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    
    # Normalize per feature
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std
    
    # Train/val split
    split = int(len(X) * 0.8)
    perm = np.random.permutation(len(X))
    train_idx, val_idx = perm[:split], perm[split:]
    
    input_dim = X.shape[2]
    
    if model_type == "lstm":
        model = LSTMDirection(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len)
    elif model_type == "transformer":
        model = TransformerDirection(input_dim=input_dim, d_model=hidden_dim, seq_len=seq_len)
    else:
        model = MLPDirection(input_dim=input_dim * seq_len)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        # Mini-batch
        perm = np.random.permutation(len(train_idx))
        for i in range(0, len(perm), batch_size):
            idx = train_idx[perm[i:i+batch_size]]
            xb = torch.FloatTensor(X[idx]).to(device)
            yb = torch.LongTensor(y[idx]).to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                xv = torch.FloatTensor(X[val_idx]).to(device)
                yv = torch.LongTensor(y[val_idx]).to(device)
                out = model(xv)
                val_loss = criterion(out, yv).item()
                preds = out.argmax(dim=1)
                acc = (preds == yv).float().mean().item()
            model.train()
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>3}/{epochs} | Loss: {loss.item():.4f} | ValLoss: {val_loss:.4f} | Acc: {acc:.3f} | {elapsed:.0f}s")
    
    # Save
    torch.save(model.state_dict())
    
    return model

if __name__ == "__main__":
    train_champion()