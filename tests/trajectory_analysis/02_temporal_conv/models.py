"""Temporal convolution models for ICR trajectory classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMLP(nn.Module):
    """Simple MLP baseline on raw 27-layer trajectory."""

    def __init__(self, input_dim=27):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [batch, 27]
        out = F.leaky_relu(self.bn1(self.fc1(x)), 0.01)
        out = self.dropout1(out)
        out = F.leaky_relu(self.bn2(self.fc2(out)), 0.01)
        out = self.dropout2(out)
        return torch.sigmoid(self.fc3(out))


class TemporalCNN(nn.Module):
    """Multi-scale 1D CNN over layer dimension.

    Three parallel Conv1d branches with kernel sizes 3, 5, 7.
    """

    def __init__(self, input_dim=27, n_filters=16):
        super().__init__()
        self.conv3 = nn.Conv1d(1, n_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, n_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, n_filters, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(n_filters * 3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * 3, 1)
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
        c3 = F.leaky_relu(self.conv3(x), 0.01)  # [batch, n_filters, 27]
        c5 = F.leaky_relu(self.conv5(x), 0.01)
        c7 = F.leaky_relu(self.conv7(x), 0.01)
        out = torch.cat([c3, c5, c7], dim=1)     # [batch, 3*n_filters, 27]
        out = self.bn(out)
        out = out.mean(dim=2)                     # global avg pool -> [batch, 3*n_filters]
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))


class MultiScaleCNN(nn.Module):
    """Two-layer multi-scale CNN with residual connections."""

    def __init__(self, input_dim=27, n_filters=16):
        super().__init__()
        # Layer 1
        self.conv3_1 = nn.Conv1d(1, n_filters, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv1d(1, n_filters, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(n_filters * 2)

        # Layer 2
        self.conv3_2 = nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_filters * 4)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * 4, 1)
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

        # Layer 1
        c3 = F.leaky_relu(self.conv3_1(x), 0.01)
        c5 = F.leaky_relu(self.conv5_1(x), 0.01)
        out = torch.cat([c3, c5], dim=1)
        out = self.bn1(out)

        # Layer 2 with residual
        residual = out
        c3 = F.leaky_relu(self.conv3_2(out), 0.01)
        c5 = F.leaky_relu(self.conv5_2(out), 0.01)
        out = torch.cat([c3, c5], dim=1)
        out = self.bn2(out)

        out = out.mean(dim=2)  # global avg pool
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))
