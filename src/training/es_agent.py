#!/usr/bin/env python3
"""CryptAI ES Agent — Evolution Strategies for trading (adapted from Jepa_dreamer)"""
import torch
import torch.nn as nn
import numpy as np

class LSTMPolicy(nn.Module):
    """LSTM policy network — 2 layers, 128 hidden, frozen bias for HOLD/BUY/SELL"""
    def __init__(self, input_dim=25, hidden=128, num_layers=2, seq_len=96, n_actions=8):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)  # 8 actions: HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL, PYRAMID, PARTIAL_CLOSE
        )
        # Frozen bias buffer (non-learnable)
        self.register_buffer('action_bias', torch.tensor([
            -4.0,   # HOLD — never selected without position
            +3.0,   # BUY — always favored
            +3.0,   # SELL — always favored
            +0.5,   # CLOSE — learned
            +1.5,   # SPLIT_BUY — learned
            +1.5,   # SPLIT_SELL — learned
            0.0,    # PYRAMID — learned
            0.0,    # PARTIAL_CLOSE — learned
        ]))
        # Frozen action mask: 1 = frozen (not learned), 0 = learned
        self.register_buffer('frozen_mask', torch.tensor([
            1,  # HOLD frozen
            1,  # BUY frozen
            1,  # SELL frozen
            0,  # CLOSE learned
            0,  # SPLIT_BUY learned
            0,  # SPLIT_SELL learned
            0,  # PYRAMID learned
            0,  # PARTIAL_CLOSE learned
        ], dtype=torch.bool))
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # (batch, hidden)
        logits = self.fc(last)  # (batch, n_actions)
        # Apply frozen bias + mask
        logits = logits + self.action_bias.unsqueeze(0)
        # For frozen actions, override with bias only (no gradient flows through frozen)
        logits = torch.where(self.frozen_mask.unsqueeze(0), self.action_bias.unsqueeze(0), logits)
        return logits
    
    def get_action(self, x, temperature=1.0):
        """Sample action with temperature"""
        with torch.no_grad():
            logits = self.forward(x) / temperature
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample().item()
    
    def get_params(self):
        """Get model parameters as flat vector (for ES)"""
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_params(self, flat_params):
        """Set model parameters from flat vector"""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat_params[offset:offset+n].view(p.shape))
            offset += n
    
    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

# ─── Evolution Strategies ───
class EvolutionStrategies:
    """ES with antithetic sampling, rank-based selection, and adaptive noise"""
    def __init__(self, policy, pop_size=32, sigma=0.02, lr=0.1, elite_ratio=0.25):
        self.policy = policy
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.elite_ratio = elite_ratio
        self.n_elite = max(1, int(pop_size * elite_ratio))
        self.n_params = policy.n_params
        self.generation = 0
        self.best_fitness = -999
    
    def ask(self):
        """Generate population: antithetic pairs (epsilon and -epsilon)"""
        self.epsilons = []
        self.population = []
        master_params = self.policy.get_params()
        
        half = self.pop_size // 2
        for i in range(half):
            epsilon = torch.randn(self.n_params) * self.sigma
            self.epsilons.append(epsilon)
            # +epsilon
            p_plus = master_params + epsilon
            self.population.append(p_plus.clone())
            # -epsilon (antithetic)
            p_minus = master_params - epsilon
            self.population.append(p_minus.clone())
        
        # If odd pop_size, add one without antithetic
        if self.pop_size % 2 == 1:
            epsilon = torch.randn(self.n_params) * self.sigma
            self.epsilons.append(epsilon)
            self.population.append(master_params + epsilon)
        
        return self.population
    
    def tell(self, fitnesses):
        """Update policy using rank-based ES update"""
        fitnesses = np.array(fitnesses)
        self.generation += 1
        self.best_fitness = max(self.best_fitness, float(np.max(fitnesses)))
        
        # Rank-based weights
        ranks = np.argsort(np.argsort(-fitnesses))  # 0 = best
        weights = np.maximum(0, self.n_elite - ranks)
        weights = weights / (weights.sum() + 1e-8)
        
        # Compute gradient
        grad = torch.zeros(self.n_params)
        for i, (epsilon, w) in enumerate(zip(self.epsilons, weights)):
            grad += w * epsilon * fitnesses[i] * (1.0 / self.sigma)
        
        # Update policy
        master = self.policy.get_params()
        master += self.lr * grad
        self.policy.set_params(master)
        
        # Decay sigma over time
        if self.generation % 50 == 0 and self.sigma > 0.005:
            self.sigma *= 0.95
        
        return float(np.mean(fitnesses)), float(np.max(fitnesses)), float(np.std(fitnesses))
    
    def save(self, path):
        torch.save({
            'policy_state': self.policy.state_dict(),
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'sigma': self.sigma,
            'lr': self.lr,
        }, path)
    
    def load(self, path):
        data = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(data['policy_state'])
        self.generation = data['generation']
        self.best_fitness = data['best_fitness']
        self.sigma = data['sigma']
        self.lr = data['lr']