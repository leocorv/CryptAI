#!/usr/bin/env python3
"""CryptAI ES v7 — OHLCV brut, le réseau apprend ses propres features"""
import os, sys, glob, time, random, json
import numpy as np
import torch

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib"); sys.path.insert(0, f"{BASE}/src/training")
from arena import Arena
from es_agent import BigPolicy as Model, EvolutionStrategies

os.makedirs(f"{BASE}/training/checkpoints/es", exist_ok=True)

# ─── Data loading (from pre-processed .npy — instant) ────────────────────────

DATA_DIR = f"{BASE}/training/data"

def load_datasets():
    seqs = np.load(f"{DATA_DIR}/datasets_seqs.npy", allow_pickle=True)
    prices = np.load(f"{DATA_DIR}/datasets_prices.npy", allow_pickle=True)
    names = np.load(f"{DATA_DIR}/datasets_names.npy", allow_pickle=True)
    ds = []
    for i in range(len(names)):
        ds.append({"seqs": seqs[i], "prices": prices[i], "name": str(names[i])})
    return ds

print("[ESv7] Loading data...", flush=True)
all_datasets = load_datasets()
random.seed(42)
random.shuffle(all_datasets)
split = max(3, int(len(all_datasets) * 0.75))
train_datasets = all_datasets[:split]
val_datasets = all_datasets[split:]
print(f"  {len(all_datasets)} total | {len(train_datasets)} train | {len(val_datasets)} val", flush=True)

device = "cuda"
torch.cuda.empty_cache()

# ─── Model & ES (bigger, better defaults) ────────────────────────────────────

policy = Model().to(device)  # input_dim=5 (OHLCV brut), n_actions=4, hidden=1024, 3 layers
es = EvolutionStrategies(policy, pop_size=64, sigma=0.2, lr=0.1,
                         elite_ratio=0.25, sigma_decay=0.98,
                         sigma_min=0.02, lr_decay=0.995, lr_min=0.01)
print(f"  Params: {policy.n_params:,} | pop={es.pop_size}", flush=True)
print(f"  sigma={es.sigma} | lr={es.lr} | elites={es.n_elite}", flush=True)

# ─── Evaluation ──────────────────────────────────────────────────────────────

def run_arena(params, dataset_list, fee_rate, max_trades=12):
    """Run policy on a set of datasets, return dict of metrics."""
    policy.set_params(params)
    all_returns = []
    all_drawdowns = []
    all_trades = []

    for ds in dataset_list:
        seqs = ds["seqs"]; p = ds["prices"]
        chunk = 1024
        all_a = []
        for s in range(0, len(seqs), chunk):
            x = torch.tensor(seqs[s:s+chunk], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = policy(x)
                all_a.extend(logits.argmax(-1).cpu().numpy())
        actions = np.array(all_a)

        ar = Arena(1000.0, fee_rate=fee_rate, slippage=0.0005)
        ti = 0; t = 0
        while ti < len(actions) and ti + 1 < len(p):
            a = int(actions[ti]); pr = float(p[ti + 1])
            if a == 0:
                pass  # hold
            elif a == 1 and ar.position is None:
                ar.open_trade(ds["name"], pr, 1)  # long
                t += 1
            elif a == 2 and ar.position is None:
                ar.open_trade(ds["name"], pr, -1)  # short
                t += 1
            elif a == 3 and ar.position is not None:
                ar.close_trade(pr)
            ti += 1
            if ar.get_metrics()["max_drawdown"] > 0.2: break
            if t >= max_trades: break
        if ar.position is not None:
            ar.close_trade(float(p[-1]))

        m = ar.get_metrics()
        all_returns.append(m["total_return"])
        all_drawdowns.append(m["max_drawdown"])
        all_trades.append(m["trade_count"])

    # Fitness = PnL × drawdown penalty - trade_penalty × n_trades
    # → force le réseau à n'ouvrir que quand il est vraiment confiant
    mean_ret = np.mean(all_returns) if all_returns else 0.0
    max_dd = max(all_drawdowns) if all_drawdowns else 0.0
    dd_penalty = max(0.0, 1.0 - max_dd * 2.0)  # 20% dd → 0.6x, 50%+ → 0.0
    total_trades = sum(all_trades)
    trade_penalty = 0.0005  # -0.05% par trade (dissuade le trading aléatoire)
    fitness = mean_ret * dd_penalty - trade_penalty * total_trades
    return fitness, mean_ret, 0.0, max_dd, total_trades


def evaluate(params, dataset_list, fee_rate, max_trades, n_datasets=3):
    """Fitness on a random subset of dataset_list."""
    subset = random.sample(dataset_list, min(n_datasets, len(dataset_list)))
    f, _, _, _, _ = run_arena(params, subset, fee_rate, max_trades)
    return f


def validate(params, dataset_list, fee_rate=0.001, max_trades=12):
    """Deterministic validation: run on ALL val datasets, return composite."""
    f, mean_ret, sharpe, dd, n_trades = run_arena(params, dataset_list, fee_rate, max_trades)
    return f, mean_ret, dd, n_trades


# ─── Curriculum ──────────────────────────────────────────────────────────────

def get_curriculum(gen):
    """Smooth fee ramp: 0 → 0.001 over 600 gens, max_trades: 20 → 8."""
    progress = min(1.0, gen / 600.0)
    fee = 0.001 * progress
    max_trades = max(8, int(20 - 12 * progress))
    return fee, max_trades


# ─── Main loop ───────────────────────────────────────────────────────────────

gen = 0
stall_count = 0
best_val_fitness = -float('inf')
last_save_gen = 0
print("[ESv7] Starting training loop...", flush=True)

while True:
    fee, max_trades = get_curriculum(gen)
    pop = es.ask()

    # Evaluate each member on a random subset of train datasets
    fits = [evaluate(p, train_datasets, fee, max_trades) for p in pop]
    mf, bf, sf = es.tell(fits)
    gen += 1

    # Log every 5 gens
    if gen % 5 == 0:
        vram = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        print(f"GEN{gen:>5} | fee={fee:.4f} maxT={max_trades} | "
              f"mean={mf:>+.3%} best={bf:>+.3%} | "
              f"σ={es.sigma:.4f} lr={es.lr:.4f} | VRAM={vram:.1f}GB",
              flush=True)

    # Validate every 10 gens on ALL val datasets (deterministic)
    if gen % 10 == 0:
        vf, vr, vdd, vt = validate(es.population[0], val_datasets, fee, max_trades)
        val_str = f"  [VAL] fit={vf:+.6f} PnL={vr:+.3%} DD={vdd:.1%} trades={vt}"

        if vf > best_val_fitness:
            best_val_fitness = vf
            stall_count = 0
            es.best_fitness = max(es.best_fitness, vf)
            es.save(f"{BASE}/training/checkpoints/es/gen{gen}_best.pth")
            print(f"{val_str}  ✅ SAVED gen{gen}", flush=True)
            if vf >= 0.05:
                print(f"  🏆 CHAMPION: fit={vf:.4f}", flush=True)
                # Also save as champion (overwrites)
                es.save(f"{BASE}/training/checkpoints/es/champion.pth")
        else:
            stall_count += 1
            print(f"{val_str}  (stall {stall_count})", flush=True)

        # Decay sigma when validation stalls for 30 checkpoints (300 gens)
        if stall_count >= 30 and es.sigma > es.sigma_min:
            es.decay_sigma()
            print(f"  📉 sigma decayed to {es.sigma:.4f}", flush=True)
            stall_count = 0  # reset counter after decay

# ─── Graceful shutdown note ──────────────────────────────────────────────────
# Kill with Ctrl+C, the last champion.pth is always the best.