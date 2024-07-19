from numpy import atleast_2d
import torch
import torch.nn as nn
import sys
from models.time_encoders import *
from models.encoders import *
from models.decoders import *
from models.utils import *


class TSDecoder(nn.Module):
    def __init__(
        self, num_classes, decoder_class: str, hidden_size: int, **kwargs
    ) -> None:
        super().__init__()

        decoder_class = getattr(sys.modules[__name__], decoder_class)
        self.decoder = decoder_class(
            num_classes=num_classes, hidden_size=hidden_size, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.decoder(x)
        return y_hat


class TSEncoder(nn.Module):
    def __init__(
        self,
        ts_encoding: dict,
        time_encoding: dict,
        encoder_class: str,
        num_features: int,
        t_length: int,
        **kwargs,
    ) -> None:

        super().__init__()
        self.time_encoder: nn.Module | None = None
        self.unsqueeze_timestamps = False
        self.input_size = ts_encoding["input_size"]
        self.time_encoding_size = time_encoding["time_encoding_size"]
        self.time_encoding_strategy = time_encoding["strategy"]
        self.time_encoding_class = time_encoding["time_encoding_class"]
        self.projection = nn.Linear(
            num_features + self.time_encoding_size, self.input_size
        )

        if self.time_encoding_class == "PositionalEncoding":
            self.time_encoder = PositionalEncoding(**time_encoding)

        elif self.time_encoding_class == "Linear":
            self.time_encoder = nn.Linear(1, self.time_encoding_size)
            self.unsqueeze_timestamps = True

        elif self.time_encoding_class == "Time2Vec":
            self.time_encoder = Time2Vec(1, self.time_encoding_size)
            self.unsqueeze_timestamps = True

        elif self.time_encoding_class == "tAPE":
            self.time_encoder = tAPE(
                self.time_encoding_size, max_len=t_length, **time_encoding
            )

        encoder_class = getattr(sys.modules[__name__], encoder_class)
        self.encoder_wrapper = encoder_class(
            seq_len=t_length,
            **ts_encoding,
            **time_encoding,
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

        h_t = self.encoder_wrapper(X=X)

        return h_t


class TSClassifier(nn.Module):
    def __init__(
        self,
        ts_encoding: dict,
        decoder: dict,
        num_classes: int,
        num_features: int,
        model_config: dict,
        t_length: int,
        **kwargs,
    ) -> None:

        super().__init__()

        self.encoder = TSEncoder(
            num_features=num_features,
            ts_encoding=ts_encoding,
            t_length=t_length,
            **model_config,
        )
        self.decoder = TSDecoder(num_classes=num_classes, **decoder, **model_config)

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor):
        h_t = self.encoder(X, timestamps)
        y_hat = self.decoder(h_t.squeeze(0))
        return y_hat
