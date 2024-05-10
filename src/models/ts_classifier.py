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
        time_encoding_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.encoder = GRU(
            input_size=input_size + time_encoding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(
        self, X: torch.Tensor, encoded_timestamps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        X = torch.cat([X, encoded_timestamps], dim=-1)
        _, h_t = self.encoder(X)

        return h_t[-1]


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        time_encoding_size: int,
        t_length: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        **kwargs,
    ) -> None:
        super().__init__()

        d_model = input_size + time_encoding_size

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

    def forward(
        self, X: torch.Tensor, encoded_timestamps: torch.Tensor
    ) -> torch.Tensor:
        X = torch.cat([X, encoded_timestamps], dim=-1)
        X_encoded = self.encoder(X)
        X_mean = torch.mean(X_encoded, dim=1)
        X_linear = self.linear(X_mean)

        return X_linear


# class TSEncoder(nn.Module):
#     def __init__(
#         self,
#         time_encoding: dict,
#         ts_encoding: dict,
#         num_features: int,
#         t_length: int,
#         **kwargs,
#     ) -> None:

#         super().__init__()
#         self.time_encoder: nn.Module | None = None

#         self.time_encoding_size = 0
#         if time_encoding is not None:
#             time_encoding_class = getattr(
#                 sys.modules[__name__], time_encoding["time_encoding_class"]
#             )
#             self.time_encoding_size = time_encoding["time_encoding_size"]

#         if self.time_encoding_size > 0:
#             self.time_encoder = time_encoding_class(**time_encoding)

#         encoder_class = getattr(sys.modules[__name__], ts_encoding["encoder_class"])

#         self.encoder_wrapper = encoder_class(
#             input_size=num_features + self.time_encoding_size,
#             t_length=t_length,
#             **ts_encoding,
#         )

#     def encode_timestamps(
#         self, X: torch.Tensor, timestamps: torch.Tensor
#     ) -> torch.Tensor:

#         encoded_timestamps = []
#         for i in range(X.shape[0]):
#             timestamp = timestamps[i]
#             t_inference = timestamp[-1] + 1
#             x_rel_timestamp = torch.clone(timestamp)
#             x_rel_timestamp = x_rel_timestamp.roll(-1)
#             x_rel_timestamp[-1] = t_inference
#             encoded_timestamps.append(
#                 self.time_encoder(
#                     (x_rel_timestamp - t_inference).unsqueeze(-1),
#                 )
#             )

#         encoded_timestamps = torch.stack(encoded_timestamps, dim=0)
#         X_encoding = torch.concat([X, encoded_timestamps], dim=-1)
#         return X_encoding, encoded_timestamps

#     def forward(self, X: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
#         X = X.swapaxes(1, 2)
#         if self.time_encoding_size > 0:
#             X, encoded_timestamps = self.encode_timestamps(X, timestamps)

#         X_encoded = self.encoder_wrapper(X)

#         return X_encoded


class TSDecoder(nn.Module):
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


class TSDecoder2(nn.Module):
    def __init__(self, num_classes, hidden_size: int, dropout: float, **kwargs) -> None:
        super().__init__()

        self.projective_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.projective_linear(x)

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
        t_length: int,
        # hidden_size: int,
        # num_layers: int,
        # batch_first: bool,
        # dropout: float,
        **kwargs,
    ) -> None:

        super().__init__()
        self.time_encoder: nn.Module | None = None

        if time_encoding is not None:
            if ts_encoding["encoder_class"] == "Transformer":
                time_encoding["time_encoding_size"] = set_dynamic_size(
                    nheads=4, T=30, input_size=input_size
                )

            self.time_encoder = PositionalEncoding(**time_encoding)

        encoder_class = getattr(sys.modules[__name__], ts_encoding["encoder_class"])
        self.encoder_wrapper = encoder_class(
            input_size=input_size,
            time_encoding_size=self.time_encoder.hidden_size,
            t_length=t_length,
            **ts_encoding,
        )

        self.dropout = nn.Dropout(p=ts_encoding["dropout"])

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        X = X.swapaxes(1, 2)
        if self.time_encoder is not None:
            encoded_timestamps = self.time_encoder(timestamps)

        h_t = self.encoder_wrapper(X=X, encoded_timestamps=encoded_timestamps)

        return h_t


class TSClassifier(nn.Module):
    def __init__(
        self,
        encoder: dict,
        decoder: dict,
        num_classes: int,
        num_features: int,
        t_length: int,
        **kwargs,
    ) -> None:

        super().__init__()

        self.encoder = TSEncoder(input_size=num_features, t_length=t_length, **encoder)
        self.decoder = TSDecoder(num_classes=num_classes, **decoder)

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor):
        h_t = self.encoder(X, timestamps)
        y_hat = self.decoder(h_t.squeeze(0))
        return y_hat
