from numpy import atleast_2d
import torch
import polars as pl
import torch.nn as nn
import sys
from models.time_encoders import *
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math


class LinearDecoder(nn.Module):
    def __init__(self, num_classes, hidden_size: int, **kwargs) -> None:
        super().__init__()

        self.internal_linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.projective_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.internal_linear(x)
        out = self.relu(out)
        y_hat = self.projective_linear(out)

        return y_hat


class LinearDecoderV2(nn.Module):
    def __init__(self, num_classes, hidden_size: int, dropout: float, **kwargs) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.internal_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.projective_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_internal = self.bn1(x)
        x_internal = self.relu(x_internal)
        x_internal = self.dropout(x_internal)
        x_internal = self.internal_linear(x_internal)
        x_internal = self.bn2(x_internal)
        x_internal = self.relu(x_internal)
        x_internal = self.dropout(x_internal)
        y_hat = self.projective_linear(x_internal)

        return y_hat.squeeze(0)
