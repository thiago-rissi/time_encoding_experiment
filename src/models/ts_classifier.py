from numpy import atleast_2d
import torch
import torch.nn as nn
import sys
from models.time_encoders import *
from models.encoders import *
from models.decoders import *
from models.utils import *


class TSDecoder(nn.Module):
    """
    Time Series Decoder module.

    Args:
        num_classes (int): The number of output classes.
        decoder_class (str): The name of the decoder class.
        hidden_size (int): The size of the hidden state.
        **kwargs: Additional keyword arguments to be passed to the decoder class.

    Attributes:
        decoder (nn.Module): The decoder module.

    """

    def __init__(
        self, num_classes, decoder_class: str, hidden_size: int, **kwargs
    ) -> None:
        super().__init__()

        decoder_class = getattr(sys.modules[__name__], decoder_class)
        self.decoder = decoder_class(
            num_classes=num_classes, hidden_size=hidden_size, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TSDecoder module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        y_hat = self.decoder(x)
        return y_hat


class TSEncoder(nn.Module):
    """
    Time Series Encoder module.

    Args:
        ts_encoding (dict): Dictionary containing parameters for time series encoding.
        time_encoding (dict): Dictionary containing parameters for time encoding.
        encoder_class (str): Name of the encoder class.
        num_features (int): Number of input features.
        t_length (int): Length of the time series.
        **kwargs: Additional keyword arguments.

    Attributes:
        time_encoder (nn.Module | None): Time encoder module.
        unsqueeze_timestamps (bool): Flag indicating whether to unsqueeze timestamps.
        input_size (int): Size of the input.
        time_encoding_size (int): Size of the time encoding.
        time_encoding_strategy (str): Strategy for time encoding.
        time_encoding_class (str): Class name of the time encoder.
        projection (nn.Linear): Linear projection layer.
        encoder_wrapper: Encoder wrapper module.

    """

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
        self.num_features = num_features
        self.projections = nn.ModuleList(
            [
                nn.Linear(1 + self.time_encoding_size, self.input_size)
                for _ in range(num_features)
            ]
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
            self.time_encoder = tAPE(max_len=t_length, **time_encoding)

        elif self.time_encoding_class == "Timestamps":
            self.time_encoder = lambda x: x.unsqueeze(-1)

        encoder_class = getattr(sys.modules[__name__], encoder_class)
        self.encoders = nn.ModuleList(
            [
                encoder_class(
                    seq_len=t_length,
                    **ts_encoding,
                    **time_encoding,
                )
                for _ in range(num_features)
            ]
        )

        self.gnn = GNN(
            num_node_features=self.input_size,
            hidden_size=400,
        )

    def encode_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode the timestamps.

        Args:
            timestamps (torch.Tensor): Timestamps tensor.

        Returns:
            torch.Tensor: Encoded timestamps tensor.

        """
        if self.time_encoding_class in ["Timestamps", "Linear", "Time2Vec"]:
            timestamps = min_max_norm(timestamps)

        if self.unsqueeze_timestamps:
            timestamps = timestamps.unsqueeze(-1)

        encoded_timestamps = self.time_encoder(timestamps)
        return encoded_timestamps

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TSEncoder module.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_features, t_length).
            timestamps (torch.Tensor): Timestamps tensor of shape (batch_size, t_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor and hidden state tensor.

        """

        X = X.swapaxes(1, 2)

        h_t = []
        for j in range(self.num_features):
            xj = X[:, :, j].unsqueeze(-1)
            if self.time_encoder is not None:
                encoded_timestamps = self.encode_timestamps(timestamps)
                xj = torch.concat(
                    [
                        xj,
                        encoded_timestamps,
                    ],
                    dim=-1,
                )
            xj = self.projections[j](xj)
            h_t.append(self.encoders[j](X=xj))

        h_t = torch.stack(h_t, dim=1)
        h_t = self.gnn(h_t)
        return h_t


class TSClassifier(nn.Module):
    """
    Time Series Classifier model.

    Args:
        ts_encoding (dict): Dictionary containing the parameters for time series encoding.
        decoder (dict): Dictionary containing the parameters for the decoder.
        num_classes (int): Number of classes for classification.
        num_features (int): Number of features in the input time series.
        model_config (dict): Dictionary containing additional model configuration parameters.
        t_length (int): Length of the time series.
        **kwargs: Additional keyword arguments.

    Attributes:
        encoder (TSEncoder): Time series encoder module.
        decoder (TSDecoder): Time series decoder module.
    """

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

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TSClassifier model.

        Args:
            X (torch.Tensor): Input time series tensor of shape (batch_size, num_features, t_length).
            timestamps (torch.Tensor): Timestamps tensor of shape (batch_size, t_length).

        Returns:
            torch.Tensor: Predicted class probabilities tensor of shape (batch_size, num_classes).
        """
        h_t = self.encoder(X, timestamps)
        y_hat = self.decoder(h_t.squeeze(0))
        return y_hat
