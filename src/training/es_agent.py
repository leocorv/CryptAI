#!/usr/bin/env python3
"""CryptAI ES Agent v7 — Clean policy, proper ES with adaptive sigma + LR"""
import torch
import numpy as np

class BigPolicy(torch.nn.Module):
    """LSTM + Attention policy — n_actions=4: [hold, long, short, close]
    Input: raw OHLCV (5 dims) — model learns its own features
    Bigger model (hidden=768, 2 layers) + hold bias initialization + high σ exploration"""
    def __init__(self, input_dim=5, hidden=768, num_layers=2, seq_len=96, n_actions=4):
        super().__init__()
        self.seq_len = seq_len; self.hidden = hidden
        self.lstm = torch.nn.LSTM(input_dim, hidden, num_layers,
                                  batch_first=True, dropout=0.1)
        self.attn = torch.nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions))
        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if 'weight' in n and p.dim() >= 2:
                torch.nn.init.orthogonal_(p, gain=0.5)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)
        # Explicitly bias final layer toward hold (action 0)
        with torch.no_grad():
            final_bias = self.fc[-1].bias
            final_bias[0] = 0.75   # hold favorisé
            final_bias[1] = -0.5   # long
            final_bias[2] = -0.5   # short
            final_bias[3] = -0.25  # close

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
            n = p.numel(); p.data.copy_(flat[o:o+n].view(p.shape)); o += n


class EvolutionStrategies:
    """ES with mirrored sampling, rank weighting, adaptive sigma + LR decay"""
    def __init__(self, policy, pop_size=64, sigma=0.2, lr=0.1, elite_ratio=0.25,
                 sigma_decay=0.98, sigma_min=0.02, lr_decay=0.995, lr_min=0.01):
        self.policy = policy; self.pop_size = pop_size
        self.sigma = sigma; self.sigma_init = sigma
        self.sigma_decay = sigma_decay; self.sigma_min = sigma_min
        self.lr = lr; self.lr_min = lr_min; self.lr_decay = lr_decay
        self.device = next(policy.parameters()).device
        self.n_params = policy.n_params
        self.n_elite = max(1, int(pop_size * elite_ratio))
        self.generation = 0; self.best_fitness = -float('inf')
        self.best_val_fitness = -float('inf')
        self.stall_counter = 0

    def ask(self):
        self.epsilons = []; self.population = []
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

        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        return float(f.mean()), float(f.max()), float(f.std())

    def decay_sigma(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def save(self, path):
        torch.save({
            'state': self.policy.state_dict(),
            'gen': self.generation,
            'best': self.best_fitness,
            'sigma': self.sigma,
            'lr': self.lr,
        }, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(d['state'])
        self.generation = d['gen']
        self.best_fitness = d['best']
        self.sigma = d.get('sigma', self.sigma)
        self.lr = d.get('lr', self.lr)
