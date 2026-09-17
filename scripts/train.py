#!/usr/bin/env python3
"""CryptAI training runner.

Experimental training pipeline for direction classifiers. The script is intentionally
self-contained and stores generated checkpoints outside Git-tracked source files.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

BASE = Path(os.getenv("CRYPTAI_HOME", Path(__file__).resolve().parents[1])).resolve()
LIB_DIR = BASE / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from features import build_features
from models import LSTMDirection, MLPDirection, TransformerDirection


def load_latest_data():
    """Load the newest CSV available for every symbol/timeframe pair."""
    data = {}
    data_dir = BASE / "data"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"No data directory found at {data_dir}. Run a collector first or set CRYPTAI_HOME."
        )

    for sym_dir in sorted(data_dir.iterdir()):
        if not sym_dir.is_dir():
            continue

        for tf_dir in sorted(sym_dir.iterdir()):
            if not tf_dir.is_dir():
                continue

            csvs = sorted(tf_dir.glob("*.csv"))
            if not csvs:
                continue

            df = pd.read_csv(csvs[-1], parse_dates=["timestamp"])
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            missing = required.difference(df.columns)
            if missing:
                print(f"[skip] {csvs[-1]} missing columns: {sorted(missing)}")
                continue

            data[f"{sym_dir.name}_{tf_dir.name}"] = df

    if not data:
        raise RuntimeError(f"No usable market CSV files found under {data_dir}.")

    return data


def create_sequences(features: np.ndarray, labels: np.ndarray, seq_len=96):
    """Convert feature rows into fixed-length sequences and next-step labels."""
    if len(features) <= seq_len:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x_values, y_values = [], []
    for i in range(len(features) - seq_len):
        x_values.append(features[i:i + seq_len])
        y_values.append(labels[i + seq_len])

    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.int64)


def compute_labels(df: pd.DataFrame, horizon=5):
    """Build direction labels while preserving unknown labels at the end as NaN.

    0 = sell, 1 = hold, 2 = buy.
    """
    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    labels = pd.Series(np.nan, index=df.index, dtype=np.float64)
    labels.loc[future_ret < -0.002] = 0
    labels.loc[(future_ret >= -0.002) & (future_ret <= 0.002)] = 1
    labels.loc[future_ret > 0.002] = 2
    return labels


def generate_champion_id(config: dict) -> str:
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def build_model(model_type: str, input_dim: int, hidden_dim: int, seq_len: int):
    if model_type == "lstm":
        return LSTMDirection(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len)
    if model_type == "transformer":
        return TransformerDirection(input_dim=input_dim, d_model=hidden_dim, seq_len=seq_len)
    if model_type == "mlp":
        return MLPDirection(input_dim=input_dim * seq_len)
    raise ValueError(f"Unknown model_type '{model_type}'. Expected: lstm, transformer or mlp.")


def evaluate(model, x_values, y_values, batch_size, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(x_values), batch_size):
            xb = torch.from_numpy(x_values[start:start + batch_size]).to(device)
            yb = torch.from_numpy(y_values[start:start + batch_size]).to(device)
            out = model(xb)
            loss = criterion(out, yb)

            total_loss += loss.item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)

    model.train()
    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, correct / total


def train_champion(model_type="lstm", seq_len=96, hidden_dim=128, batch_size=64,
                   epochs=50, lr=1e-3, seed=42):
    if seq_len <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("seq_len, batch_size and epochs must be positive")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Base: {BASE}")
    print(f"[train] Device: {device} | Model: {model_type} | Epochs: {epochs} | Seed: {seed}")

    data = load_latest_data()
    train_x_parts, train_y_parts = [], []
    val_x_parts, val_y_parts = [], []

    for key, df in data.items():
        feature_df = build_features(df)
        labels = compute_labels(df).reindex(feature_df.index)

        valid = labels.notna()
        feature_df = feature_df.loc[valid]
        labels = labels.loc[valid].astype(np.int64)

        if len(feature_df) < seq_len + 10:
            print(f"  {key}: skipped, only {len(feature_df)} usable rows")
            continue

        x_values, y_values = create_sequences(
            feature_df.to_numpy(dtype=np.float32),
            labels.to_numpy(dtype=np.int64),
            seq_len,
        )

        if len(x_values) < 2:
            print(f"  {key}: skipped, not enough sequences")
            continue

        # Chronological split with a purge gap so validation windows do not reuse
        # observations already present in the final training sequences.
        split = max(1, min(len(x_values) - 1, int(len(x_values) * 0.8)))
        val_start = split + seq_len
        if val_start >= len(x_values):
            print(f"  {key}: skipped, not enough sequences after validation purge gap")
            continue

        train_x_parts.append(x_values[:split])
        train_y_parts.append(y_values[:split])
        val_x_parts.append(x_values[val_start:])
        val_y_parts.append(y_values[val_start:])
        print(
            f"  {key}: {len(x_values)} sequences "
            f"({split} train / {seq_len} gap / {len(x_values) - val_start} val)"
        )

    if not train_x_parts or not val_x_parts:
        raise RuntimeError(
            "No dataset produced enough sequences for training, purge gap and validation. "
            "Collect more history or reduce seq_len."
        )

    train_x = np.concatenate(train_x_parts)
    train_y = np.concatenate(train_y_parts)
    val_x = np.concatenate(val_x_parts)
    val_y = np.concatenate(val_y_parts)

    # Fit normalization on training data only to avoid validation leakage.
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-8
    train_x = ((train_x - mean) / std).astype(np.float32)
    val_x = ((val_x - mean) / std).astype(np.float32)

    input_dim = train_x.shape[2]
    model = build_model(model_type, input_dim, hidden_dim, seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(len(train_x))
        epoch_loss = 0.0
        seen = 0

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.from_numpy(train_x[idx]).to(device)
            yb = torch.from_numpy(train_y[idx]).to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            seen += len(idx)

        if epoch % 10 == 0 or epoch == epochs - 1:
            val_loss, accuracy = evaluate(model, val_x, val_y, batch_size, device, criterion)
            elapsed = time.time() - t0
            train_loss = epoch_loss / max(seen, 1)
            print(
                f"  Epoch {epoch + 1:>3}/{epochs} | TrainLoss: {train_loss:.4f} "
                f"| ValLoss: {val_loss:.4f} | Acc: {accuracy:.3f} | {elapsed:.0f}s"
            )

    config = {
        "model_type": model_type,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "input_dim": input_dim,
    }
    champion_id = generate_champion_id(config)
    output_dir = BASE / "arena" / "champions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{champion_id}_{model_type}.pt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "normalization": {
                "mean": mean.reshape(-1).tolist(),
                "std": std.reshape(-1).tolist(),
            },
        },
        output_path,
    )
    print(f"[train] Saved champion: {output_path}")

    return model


if __name__ == "__main__":
    train_champion()
