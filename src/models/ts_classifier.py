from numpy import atleast_2d
import torch
import polars as pl
import torch.nn as nn
import sys
from models.time_encoder import *
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.encoder = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _, h_t = self.encoder(X)

        return h_t[-1]


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()

        d_model = input_size

        encoder_layer = TransformerEncoderLayer(
            batch_first=batch_first,
            d_model=d_model,
            nhead=4,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.linear = nn.Linear(
            in_features=d_model,
            out_features=hidden_size,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_encoded = self.encoder(X)
        X_encoded = self.dropout(X_encoded)
        X_mean = X_encoded[:, -1]
        X_linear = self.linear(X_mean)

        return X_linear


class TSDecoder2(nn.Module):
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


class TSDecoder(nn.Module):
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


def set_dynamic_size(nheads: int, T: int, input_size: int) -> int:
    size = nheads * T - input_size

    return size


class TSEncoder(nn.Module):
    def __init__(
        self,
        time_encoding: dict,
        ts_encoding: dict,
        input_size: int,
        num_features: int,
        **kwargs,
    ) -> None:

        super().__init__()
        self.time_encoder: nn.Module | None = None
        self.unsqueeze_timestamps = False
        self.time_encoding_size = time_encoding["time_encoding_size"]
        self.time_encoding_strategy = time_encoding["strategy"]
        self.projection = nn.Linear(num_features + self.time_encoding_size, input_size)

        if (self.time_encoding_strategy == "relative") or (
            self.time_encoding_strategy == "absolute"
        ):
            self.time_encoder = PositionalEncoding(**time_encoding)

        elif self.time_encoding_strategy == "timestamps":
            self.time_encoder = nn.Linear(1, self.time_encoding_size)
            self.unsqueeze_timestamps = True

        elif self.time_encoding_strategy == "none":
            self.linear = nn.Linear(num_features, input_size)

        else:
            Exception("Invalid time encoding strategy")

        encoder_class = getattr(sys.modules[__name__], ts_encoding["encoder_class"])
        self.encoder_wrapper = encoder_class(
            input_size=input_size,
            **ts_encoding,
        )

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        X = X.swapaxes(1, 2)

        if self.time_encoder is not None:
            timestamps = min_max_norm(timestamps)
            if self.unsqueeze_timestamps:
                timestamps = timestamps.unsqueeze(-1)

            encoded_timestamps = self.time_encoder(timestamps)
            X = torch.cat([X, encoded_timestamps], dim=-1)
            X = self.projection(X)
        else:
            X = self.linear(X)

        h_t = self.encoder_wrapper(X=X)

        return h_t


class TSClassifier(nn.Module):
    def __init__(
        self,
        encoder: dict,
        decoder: dict,
        num_classes: int,
        num_features: int,
        **kwargs,
    ) -> None:

        super().__init__()

        self.encoder = TSEncoder(num_features=num_features, **encoder)
        self.decoder = TSDecoder(num_classes=num_classes, **decoder)

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor):
        h_t = self.encoder(X, timestamps)
        y_hat = self.decoder(h_t.squeeze(0))
        return y_hat


def find_minimun_divisor(threshold: int, dividend: int) -> int:
    md = threshold
    if dividend <= threshold:
        return dividend
    while dividend % md != 0:
        md += 1
    return md


def min_max_norm(x: torch.Tensor) -> torch.Tensor:
    x_norm = (x - x.min()) / (x.max() - x.min())
    return x_norm
