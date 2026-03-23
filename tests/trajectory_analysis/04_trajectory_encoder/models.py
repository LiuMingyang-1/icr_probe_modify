"""Trajectory encoder models: GRU, Transformer, Deep1DCNN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    """Bidirectional GRU treating 27 layers as a sequence."""

    def __init__(self, input_dim=27, hidden_size=32, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [batch, 27]
        x = x.unsqueeze(2)  # [batch, 27, 1]
        output, hidden = self.gru(x)
        # Take last hidden state from both directions
        # hidden: [num_layers*2, batch, hidden_size]
        h_fwd = hidden[-2]  # last layer forward
        h_bwd = hidden[-1]  # last layer backward
        h = torch.cat([h_fwd, h_bwd], dim=1)  # [batch, hidden*2]
        h = self.dropout(h)
        return torch.sigmoid(self.fc(h))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SmallTransformer(nn.Module):
    """Small Transformer encoder over layer dimension."""

    def __init__(self, input_dim=27, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.3, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [batch, 27]
        x = x.unsqueeze(2)          # [batch, 27, 1]
        x = self.embed(x)           # [batch, 27, d_model]
        x = self.pos_enc(x)
        x = self.transformer(x)     # [batch, 27, d_model]
        x = x.mean(dim=1)           # mean pool -> [batch, d_model]
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))


class Deep1DCNN(nn.Module):
    """Deeper 1D CNN encoder over layers."""

    def __init__(self, input_dim=27):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [batch, 27]
        x = x.unsqueeze(1)  # [batch, 1, 27]
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        x = self.pool(x).squeeze(2)  # [batch, 64]
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))
