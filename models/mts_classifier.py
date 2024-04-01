import torch
import polars as pl
import torch.nn as nn
import sys
from models.time_encoder import *


class MTSEncoder(nn.Module):
    def __init__(
        self,
        time_encoding: dict,
        hidden_size: int,
        **kwargs,
    ) -> None:

        super().__init__()
        self.time_encoder: nn.Module | None = None
        self.hidden_size = hidden_size

        time_encoding_size = 0
        if time_encoding is not None:
            time_encoding_class = getattr(
                sys.modules[__name__], time_encoding["time_encoding_class"]
            )
            time_encoding_size = time_encoding["time_encoding_size"]

        if time_encoding_size > 0:
            self.time_encoder = time_encoding_class(**time_encoding)


class LinearDecoder(nn.Module):
    def __init__(self, n_classes, hidden_size: int, **kwargs) -> None:
        super().__init__()

        self.internal_linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.projective_linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_internal = self.internal_linear(x)
        x_internal = self.relu(x_internal)
        y_hat = self.projective_linear(x_internal)

        return y_hat


class MTSClassifier(nn.Module):
    def __init__(
        self,
        encoder: dict,
        decoder: dict,
        **kwargs,
    ) -> None:

        super().__init__()

        self.encoder = MTSEncoder(**encoder)
        self.decoder = LinearDecoder(**decoder)

    def forward(self, X: torch.Tensor):
        x_encoded = self.encoder(X)
        y_hat = self.decoder(x_encoded)
        return y_hat
