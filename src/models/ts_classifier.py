import torch
import polars as pl
import torch.nn as nn
import sys
from models.time_encoder import *
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer


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
        self.encoder = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_encoded, _ = self.encoder(X)

        return X_encoded[:, -1, :]


class ARRNN(nn.Module):
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

        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(
        self, X: torch.Tensor, encoded_timestamps: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_sequence = torch.empty(
            (X.shape[0], X.shape[1] - 1, X.shape[2]), device=X.device
        )

        h_t = torch.zeros(
            (self.num_layers, X.shape[0], self.encoder.hidden_size), device=X.device
        )
        y_hat = torch.zeros((X.shape[0], 1, X.shape[2]), device=X.device)
        for i in range(0, X.shape[1] - 1):
            input = torch.cat(
                [
                    torch.where(mask[:, [i], None] | (i == 0), X[:, [i]], y_hat),
                    encoded_timestamps[:, [i + 1]],
                ],
                dim=-1,
            )

            out, h_t = self.encoder(input, h_t)

            y_hat = self.linear(out)
            output_sequence[:, i] = y_hat.squeeze(1)

        return output_sequence, h_t[-1]


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        t_length: int,
        **kwargs,
    ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            batch_first=batch_first,
            d_model=input_size,
            nhead=1,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.linear = nn.Linear(
            in_features=input_size * t_length, out_features=hidden_size
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_encoded = self.encoder(X)
        X_flat = torch.flatten(X_encoded, start_dim=1)
        X_linear = self.linear(X_flat)
        return X_linear


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, padding=2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=4, padding=2),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, dim=1)
        x_conv = self.conv_block(x)
        x_conv = x_conv.swapaxes(1, 2)
        return x_conv


class TSEncoder(nn.Module):
    def __init__(
        self,
        time_encoding: dict,
        ts_encoding: dict,
        num_features: int,
        t_length: int,
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
            input_size=num_features + self.time_encoding_size,
            t_length=t_length,
            **ts_encoding,
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

        X_encoded = self.encoder_wrapper(X)

        return X_encoded


class TSDecoder(nn.Module):
    def __init__(self, num_classes, hidden_size: int, **kwargs) -> None:
        super().__init__()

        self.internal_linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.projective_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_internal = self.internal_linear(x)
        x_internal = self.relu(x_internal)
        y_hat = self.projective_linear(x_internal)

        return y_hat


class TSAREncoderDecoder(nn.Module):
    def __init__(
        self,
        time_encoding: dict,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        **kwargs,
    ) -> None:

        super().__init__()
        self.time_encoder: nn.Module | None = None

        if time_encoding is not None:
            self.time_encoder = PositionalEncoding(**time_encoding)

        self.encoder_wrapper = ARRNN(
            input_size=input_size,
            time_encoding_size=self.time_encoder.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = torch.ones(
                (X.shape[0], X.shape[-1]), device=X.device, dtype=torch.bool
            )

        X = X.swapaxes(1, 2)
        if self.time_encoder is not None:
            encoded_timestamps = self.time_encoder(timestamps)

        X_encoded, h_t = self.encoder_wrapper(
            X=X, encoded_timestamps=encoded_timestamps, mask=mask
        )

        return X_encoded, h_t


class TSClassifier(nn.Module):
    def __init__(
        self,
        encoder: TSAREncoderDecoder,
        decoder: dict,
        num_classes: int,
        num_features: int,
        t_length: int,
        **kwargs,
    ) -> None:

        super().__init__()

        self.encoder = encoder
        self.decoder = TSDecoder(num_classes=num_classes, **decoder)

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor):
        _, x_encoded = self.encoder(X, timestamps)
        y_hat = self.decoder(x_encoded.squeeze(0))
        return y_hat
