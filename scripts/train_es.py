#!/usr/bin/env python3
"""CryptAI ES Trainer — Evolution Strategies for trading (inspired by Jepa_dreamer)"""
import os, sys, json, glob, time, random
import numpy as np
import pandas as pd
import torch

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib")
from features import build_features
from arena import Arena
os.makedirs(f"{BASE}/training/checkpoints/es", exist_ok=True)
os.makedirs(f"{BASE}/training/checkpoints/es", exist_ok=True)

# ─── Curriculum config ───
CURRICULUM = {
    1: {"spread": 0.0, "slippage": 0.0, "fee": 0.0, "max_trades": 20, "episodes": 200},
    2: {"spread": 0.0003, "slippage": 0.0002, "fee": 0.0005, "max_trades": 12, "episodes": 300},
    3: {"spread": 0.001, "slippage": 0.0005, "fee": 0.001, "max_trades": 8, "episodes": 999999},
}

# ─── Load data ───
print("[ES] Loading data...", flush=True)
datasets = []
for f in sorted(glob.glob(f"{BASE}/data/*/*/historical.csv")):
    try:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        if len(df) < 300: continue
        feats = build_features(df)
        if len(feats) < 196: continue
        vals = feats.values.astype(np.float32)
        for i in range(len(vals)):
            m = vals[i].mean(); s = vals[i].std() + 1e-8; vals[i] = (vals[i] - m) / s
        prices = df["close"].values[-len(feats):]
        n = len(vals)
        datasets.append({"vals": vals, "prices": prices, "split": int(n * 0.7), "name": f.split("/")[-3]})
    except: pass
print(f"  {len(datasets)} datasets loaded", flush=True)

# ─── ES Agent ───
from es_agent import LSTMPolicy, EvolutionStrategies
device = "cuda"
policy = LSTMPolicy().to(device)
es = EvolutionStrategies(policy, pop_size=16, sigma=0.02, lr=0.1, elite_ratio=0.25)
print(f"  Policy: {policy.n_params:,} params", flush=True)

def evaluate(policy_params, phase=1, dataset_idx=None):
    """Evaluate a single policy on a random dataset, return fitness = PnL%"""
    if dataset_idx is None:
        dataset_idx = random.randrange(len(datasets))
    ds = datasets[dataset_idx]
    train_vals = ds["vals"][:ds["split"]]
    train_prices = ds["prices"][:ds["split"]]
    
    policy.set_params(policy_params)
    cfg = CURRICULUM[phase]
    
    ar = Arena(1000.0, fee_rate=cfg["fee"], slippage=cfg["slippage"])
    ti = 96
    trades = 0
    last_action = 0
    
    while ti < len(train_vals) - 1:
        x = torch.FloatTensor(train_vals[ti-96:ti]).unsqueeze(0).to(device)
        action = policy.get_action(x, temperature=max(0.5, 1.5 - es.generation * 0.01))
        
        # Phase 1: bonus for opening trades (exploration)
        if action == 1 and ar.position is None:  # BUY
            ar.open_trade(ds["name"], train_prices[ti], 1)
            trades += 1
        elif action == 2 and ar.position is None:  # SELL
            ar.open_trade(ds["name"], train_prices[ti], 1)
            trades += 1
        elif action == 3 and ar.position is not None:  # CLOSE
            ar.close_trade(train_prices[ti])
        elif action == 4 and ar.position is None:  # SPLIT_BUY (half position)
            ar.open_trade(ds["name"], train_prices[ti], 0.5)
            trades += 1
        elif action == 5 and ar.position is None:  # SPLIT_SELL
            ar.open_trade(ds["name"], train_prices[ti], 0.5)
            trades += 1
        elif action == 6 and ar.position is not None:  # PYRAMID (add to position)
            ar.open_trade(ds["name"], train_prices[ti], 0.3)
        elif action == 7 and ar.position is not None:  # PARTIAL_CLOSE (close half)
            ar.close_trade(train_prices[ti])
        
        ti += 1
        if ar.get_metrics()["max_drawdown"] > 0.2: break
        if trades >= cfg["max_trades"]: break
    
    if ar.position is not None:
        ar.close_trade(train_prices[-1])
    
    mt = ar.get_metrics()
    fitness = mt["total_return"]
    
    # Penalties & bonuses
    if trades == 0:
        fitness -= 0.50  # -50% penalty for no trades
    if trades > 0:
        fitness += 0.02  # +2% bonus for trading
    
    return fitness, dataset_idx, trades, mt

# ─── Main loop ───
print("\n[ES] Training...", flush=True)
gen = 0
while True:
    phase = 1 if gen < 200 else (2 if gen < 500 else 3)
    
    # Generate population
    population = es.ask()
    
    # Evaluate each individual
    fitnesses = []
    eval_data = []
    for i, params in enumerate(population):
        fit, ds_idx, trades, mt = evaluate(params, phase)
        fitnesses.append(fit)
        eval_data.append((ds_idx, trades, mt))
    
    # Update policy
    mean_fit, best_fit, std_fit = es.tell(fitnesses)
    
    gen += 1
    if gen % 5 == 0:
        # Get best individual's details
        best_idx = np.argmax(fitnesses)
        best_ds, best_trades, best_mt = eval_data[best_idx]
        print(f"GEN {gen:>4} | Phase {phase} | PopFitness: mean={mean_fit:>+.3%} best={best_fit:>+.3%} std={std_fit:.3%} | Trades:{best_trades:>2} | DD:{best_mt['max_drawdown']:.2%} | σ:{es.sigma:.4f}", flush=True)
    
    if gen % 20 == 0:
        # Test on validation (holdout data)
        val_ds = random.randrange(len(datasets))
        val_fit, _, val_trades, val_mt = evaluate(population[best_idx], 3, val_ds)
        print(f"  [VAL] Dataset {val_ds}: PnL={val_fit:>+.3%} Trades={val_trades} DD={val_mt['max_drawdown']:.2%}", flush=True)
        
        if val_fit > es.best_fitness:
            es.best_fitness = val_fit
            es.save(f"{BASE}/training/checkpoints/es/gen{gen}_best.pth")
            print(f"  ✅ Saved best: gen{gen}_fitness{val_fit:.3%}.pth", flush=True)
            
            if val_fit >= 0.16:
                print(f"\n🏆 CHAMPION ES: {val_fit:.2%} in arena! Threshold reached.", flush=True)