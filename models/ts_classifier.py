import torch
import polars as pl
import torch.nn as nn
import sys
from models.time_encoder import *
from torch.nn import GRU, TransformerEncoder


class RNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        **kwargs,
    ) -> None:
        self.encoder = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )


class TSEncoder(nn.Module):
    def __init__(
        self,
        time_encoding: dict,
        ts_encoding: dict,
        num_classes: int,
        **kwargs,
    ) -> None:

        super().__init__()
        self.time_encoder: nn.Module | None = None

        self.time_encoding_size = 0
        if time_encoding is not None:
            time_encoding_class = getattr(
                sys.modules[__name__], time_encoding["time_encoding_class"]
            )
            self.time_encoding_size = time_encoding["time_encoding_size"]

        if self.time_encoding_size > 0:
            self.time_encoder = time_encoding_class(**time_encoding)

        encoder_class = getattr(sys.modules[__name__], ts_encoding["encoder_class"])

        self.encoder_wrapper = encoder_class(
            input_size=num_classes + self.time_encoding_size, **ts_encoding
        )

    def encode_timestamps(
        self, X: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:

        encoded_timestamps = []
        for i in range(X.shape[0]):
            timestamp = timestamps[i]
            t_inference = timestamp[-1] + 1
            x_rel_timestamp = torch.clone(timestamp)
            x_rel_timestamp = x_rel_timestamp.roll(-1)
            x_rel_timestamp[-1] = t_inference
            encoded_timestamps.append(
                self.time_encoder(
                    (x_rel_timestamp - t_inference).unsqueeze(-1),
                )
            )

        encoded_timestamps = torch.stack(encoded_timestamps, dim=0)
        X_encoding = torch.concat([X, encoded_timestamps], dim=-1)
        return X_encoding, encoded_timestamps

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        X = X.swapaxes(1, 2)
        if self.time_encoding_size > 0:
            X, encoded_timestamps = self.encode_timestamps(X, timestamps)

        X_encoded, _ = self.encoder_wrapper.encoder(X)

        return X_encoded[:, -1, :]


class LinearDecoder(nn.Module):
    def __init__(self, num_classes, hidden_size: int, **kwargs) -> None:
        super().__init__()

        self.internal_linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.projective_linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_internal = self.internal_linear(x)
        x_internal = self.relu(x_internal)
        y_hat = self.projective_linear(x_internal)

        return y_hat.squeeze()


class TSClassifier(nn.Module):
    def __init__(
        self,
        encoder: dict,
        decoder: dict,
        num_classes: int,
        **kwargs,
    ) -> None:

        super().__init__()

        self.encoder = TSEncoder(num_classes=num_classes, **encoder)
        self.decoder = LinearDecoder(num_classes=num_classes, **decoder)

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor):
        x_encoded = self.encoder(X, timestamps)
        y_hat = self.decoder(x_encoded)
        return y_hat
