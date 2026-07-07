#!/usr/bin/env python3
"""CryptAI Advanced Models — GPU-saturating architectures"""
import torch
import torch.nn as nn

# ─── Deep LSTM (3-4 layers, 512-1024 hidden) ───
class DeepLSTM(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=512, num_layers=3, seq_len=96, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn).sum(dim=1)
        return self.classifier(context)

# ─── Deep Transformer (6-8 layers, multi-head) ───
class DeepTransformer(nn.Module):
    def __init__(self, input_dim=25, d_model=256, nhead=8, num_layers=6, seq_len=96, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x).mean(dim=1)
        return self.classifier(x)

# ─── GRU (alternative to LSTM) ───
class DeepGRU(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=512, num_layers=3, seq_len=96):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h[-1])

# ─── TCN (Temporal Convolutional Network) ───
class TCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dilation, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = out.transpose(1, 2)
        return self.norm(out)

class TCN(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=128, num_layers=4, seq_len=96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        dilations = [2**i for i in range(num_layers)]
        self.blocks = nn.Sequential(*[TCNBlock(hidden_dim, hidden_dim, d) for d in dilations])
        self.classifier = nn.Linear(hidden_dim, 3)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x.mean(dim=1))

# ─── Ensemble (all models) ───
class MegaEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    def forward(self, x):
        return torch.stack([m(x) for m in self.models]).mean(dim=0)