#!/usr/bin/env python3
"""CryptAI Specialist — train ONE model per symbol with ES + hold bias"""
import os, sys, glob, time, random, json
import numpy as np
import torch

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib")
from arena import Arena

os.makedirs(f"{BASE}/training/checkpoints/specialists", exist_ok=True)
os.makedirs(f"{BASE}/training/data", exist_ok=True)

# ─── Medium model: enough capacity to learn real patterns ────────────────────

class SpecialistPolicy(torch.nn.Module):
    """Medium LSTM policy — learns micro-patterns for ONE symbol"""
    def __init__(self, input_dim=5, hidden=128, num_layers=1, seq_len=96, n_actions=4):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = torch.nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.attn = torch.nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, n_actions))
        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if 'weight' in n and p.dim() >= 2:
                torch.nn.init.orthogonal_(p, gain=0.5)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)
        # Strong hold bias: action 0 (hold) très favorisé
        with torch.no_grad():
            self.fc[-1].bias[0] = 1.5   # hold fortement favorisé
            self.fc[-1].bias[1] = -1.0  # long
            self.fc[-1].bias[2] = -1.0  # short
            self.fc[-1].bias[3] = -0.5  # close

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        o, _ = self.lstm(x)
        o2, _ = self.attn(o, o, o)
        f = o2.mean(dim=1)
        return self.fc(f)

    def set_params(self, flat):
        o = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[o:o+n].view(p.shape))
            o += n


class MiniES:
    """Minimal ES for specialist training"""
    def __init__(self, policy, pop_size=32, sigma=0.15, lr=0.05, elite_ratio=0.25):
        self.policy = policy
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.device = next(policy.parameters()).device
        self.n_params = policy.n_params
        self.n_elite = max(1, int(pop_size * elite_ratio))
        self.generation = 0
        self.best_fitness = -float('inf')

    def ask(self):
        self.epsilons = []
        self.population = []
        master = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        half = self.pop_size // 2
        for i in range(half):
            eps = torch.randn(self.n_params, device=self.device) * self.sigma
            self.epsilons.append(eps)
            self.population.append(master + eps)
            self.population.append(master - eps)
        return self.population

    def tell(self, fitnesses):
        f = np.array(fitnesses, dtype=np.float64)
        self.generation += 1
        self.best_fitness = max(self.best_fitness, float(f.max()))

        ranks = np.argsort(np.argsort(-f))
        w = np.maximum(0, self.n_elite - ranks)
        w = w / (w.sum() + 1e-8)
        w = w - w.mean()

        grad = torch.zeros(self.n_params, device=self.device)
        for eps, wi in zip(self.epsilons, w):
            grad = grad + wi * eps

        master = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        scaled_grad = self.lr / (self.sigma + 1e-8) * grad
        o = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_((master + scaled_grad)[o:o+n].view(p.shape))
            o += n

        return float(f.mean()), float(f.max()), float(f.std())

    def save(self, path):
        torch.save({
            'state': self.policy.state_dict(),
            'gen': self.generation,
            'best': self.best_fitness,
            'sigma': self.sigma,
        }, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(d['state'])
        self.generation = d['gen']
        self.best_fitness = d['best']
        self.sigma = d.get('sigma', self.sigma)


# ─── Data loading for ONE symbol ────────────────────────────────────────────

def load_symbol_data(symbol, max_files=5):
    """Load pre-processed data for a single symbol"""
    path = f"{BASE}/data/{symbol}/1s/*.csv"
    files = sorted(glob.glob(path))
    files = [f for f in files if "historical" not in f]
    ds = []
    for f in files[-max_files:]:
        try:
            df = __import__('pandas').read_csv(f)
            if len(df) < 300: continue
            ohlc = df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
            ohlc_log = np.log(ohlc + 1e-10)
            for col in range(ohlc_log.shape[1]):
                m = ohlc_log[:, col].mean()
                s = ohlc_log[:, col].std() + 1e-8
                ohlc_log[:, col] = (ohlc_log[:, col] - m) / s
            vals = ohlc_log.astype(np.float32)
            seqs = np.lib.stride_tricks.sliding_window_view(vals, 96, axis=0)
            seqs = np.moveaxis(seqs, -1, 1)
            prices = ohlc[95:, 3]
            ds.append({"seqs": seqs, "prices": prices, "name": symbol})
        except Exception as e:
            print(f"    [warn] {f}: {e}", flush=True)
    return ds

def get_curriculum(gen):
    """Fee 0→0.0005 over 500 gens, maxT 20→10"""
    progress = min(1.0, gen / 500.0)
    fee = 0.0005 * progress
    max_trades = max(10, int(20 - 10 * progress))
    return fee, max_trades


# ─── Evaluate on one symbol ─────────────────────────────────────────────────

def evaluate_specialist(params, datasets, fee, max_trades, device, policy):
    """Evaluate specialist — fitness = PnL + win_quality - inactivity_penalty"""
    policy.set_params(params)
    all_pnls = []
    trade_counts = []
    total_trades_all = 0

    for ds in datasets:
        seqs = ds["seqs"]
        prices = ds["prices"]
        all_a = []
        for s in range(0, len(seqs), 1024):
            x = torch.tensor(seqs[s:s+1024], dtype=torch.float32, device=device)
            with torch.no_grad():
                all_a.extend(policy(x).argmax(-1).cpu().numpy())

        ar = Arena(1000.0, fee_rate=fee, slippage=0.0005)
        ti = 0
        t = 0
        while ti < len(all_a) and ti + 1 < len(prices):
            a = int(all_a[ti])
            pr = float(prices[ti + 1])
            if a == 1 and ar.position is None:
                ar.open_trade(ds["name"], pr, 1)
                t += 1
            elif a == 2 and ar.position is None:
                ar.open_trade(ds["name"], pr, -1)
                t += 1
            elif a == 3 and ar.position is not None:
                ar.close_trade(pr)
            ti += 1
            if ar.get_metrics()["max_drawdown"] > 0.2: break
            if t >= max_trades: break
        if ar.position is not None:
            ar.close_trade(float(prices[-1]))

        # Collect per-trade PnLs
        for trade in ar.trades:
            all_pnls.append(trade.pnl)
        trade_counts.append(len(ar.trades))
        total_trades_all += len(ar.trades)

    # ─── New fitness: quality over quantity ───
    total_pnl = sum(all_pnls) if all_pnls else 0.0
    wins = sum(1 for p in all_pnls if p > 0)
    losses = sum(1 for p in all_pnls if p <= 0)

    # PnL en % du capital (1000) pour garder des valeurs stables
    pnl_pct = total_pnl / 1000.0

    # Fitness = PnL% + qualité des trades - pénalité d'inactivité
    trade_quality = wins * 0.02 - losses * 0.04
    inactivity = -0.05 if total_trades_all == 0 else 0.0

    fitness = pnl_pct + trade_quality + inactivity
    avg_pnl = total_pnl / max(1, len(datasets))
    return fitness, avg_pnl, 0.0, total_trades_all


# ─── Main ───────────────────────────────────────────────────────────────────

def train_specialist(symbol, generations=5000):
    print(f"\n{'='*60}", flush=True)
    print(f"🏋️  Training specialist: {symbol}", flush=True)
    print(f"{'='*60}", flush=True)

    device = "cuda"
    torch.cuda.empty_cache()

    # Load data for this symbol only
    all_data = load_symbol_data(symbol, max_files=5)
    if len(all_data) < 2:
        print(f"  ❌ Not enough data for {symbol}", flush=True)
        return

    # Split
    split = max(1, int(len(all_data) * 0.7))
    train_data = all_data[:split]
    val_data = all_data[split:]
    print(f"  Data: {len(all_data)} files | {len(train_data)} train | {len(val_data)} val", flush=True)

    # Model
    policy = SpecialistPolicy().to(device)
    es = MiniES(policy, pop_size=64, sigma=0.1, lr=0.05, elite_ratio=0.25)
    print(f"  Params: {policy.n_params:,} | pop={es.pop_size} | σ={es.sigma}", flush=True)

    # Training loop
    best_val = -float('inf')
    stall = 0
    for gen in range(generations):
        fee, max_t = get_curriculum(gen)
        pop = es.ask()

        fits = []
        for p in pop:
            subset = random.sample(train_data, min(2, len(train_data)))
            f, _, _, _ = evaluate_specialist(p, subset, fee, max_t, device, policy)
            fits.append(f)

        mf, bf, _ = es.tell(fits)

        if gen % 10 == 0:
            vf, vr, vdd, vt = evaluate_specialist(
                es.population[0], val_data, fee, max_t, device, policy)
            vram = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024

            if vf > best_val:
                best_val = vf
                stall = 0
                path = f"{BASE}/training/checkpoints/specialists/{symbol.lower()}_best.pth"
                es.save(path)
                print(f"GEN{gen:>4} {symbol} | mean={mf:>+.3%} best={bf:>+.3%} | "
                      f"VAL PnL={vr:>+.3%} trades={vt} | σ={es.sigma:.3f} | ✅", flush=True)
            else:
                stall += 1
                print(f"GEN{gen:>4} {symbol} | mean={mf:>+.3%} best={bf:>+.3%} | "
                      f"VAL PnL={vr:>+.3%} trades={vt} | σ={es.sigma:.3f} (stall {stall})", flush=True)

            # Decay sigma on plateau
            if stall >= 30 and es.sigma > 0.02:
                es.sigma = max(0.02, es.sigma * 0.95)
                stall = 0

    print(f"  ✅ {symbol} done — best_val={best_val:.6f}", flush=True)


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    for sym in symbols:
        train_specialist(sym)
