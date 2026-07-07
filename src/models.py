#!/usr/bin/env python3
"""CryptAI — Models: LSTM, Transformer, Ensemble Direction Classifier"""
import torch
import torch.nn as nn

# ─── Simple LSTM Direction Classifier ───
class LSTMDirection(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2, seq_len=96):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3)  # 0=sell, 1=hold, 2=buy
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(context)

# ─── Simple Transformer Direction Classifier ───
class TransformerDirection(nn.Module):
    def __init__(self, input_dim=64, d_model=128, nhead=4, num_layers=2, seq_len=96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4, dropout=0.1,
                                                   batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # global avg pool
        return self.classifier(x)

# ─── Simple MLP Baseline ───
class MLPDirection(nn.Module):
    def __init__(self, input_dim=64*96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# ─── Ensemble Model ───
class EnsembleDirection(nn.Module):
    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs).mean(dim=0)  # average logits