import numpy as np
from torch import nn
from models.time_encoders import (
    tAPE,
)
import torch
import torch.nn as nn
from models.time_encoders import *
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


class TransformerTorch(nn.Module):
    """
    TransformerTorch is a PyTorch module that implements a Transformer encoder.

    Args:
        input_size (int): The input size of the encoder.
        hidden_size (int): The output size of the linear layer.
        num_layers (int): The number of layers in the Transformer encoder.
        batch_first (bool): If True, the input and output tensors are provided as (batch, seq, feature).
                            If False, the input and output tensors are provided as (seq, batch, feature).
        dropout (float): The dropout probability.

    Attributes:
        encoder (TransformerEncoder): The Transformer encoder module.
        linear (nn.Linear): The linear layer to transform the encoded input.
        dropout (nn.Dropout): The dropout layer.

    """

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
        """
        Forward pass of the TransformerTorch module.

        Args:
            X (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the Transformer encoder,
                          dropout layer, and linear layer.

        """
        X_encoded = self.encoder(X)
        X_encoded = self.dropout(X_encoded)
        X_mean = X_encoded[:, -1]
        X_linear = self.linear(X_mean)

        return X_linear


class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) module.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers. Default is 1.
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature).
                            Default is False.
        **kwargs: Additional keyword arguments.

    Attributes:
        num_layers (int): Number of recurrent layers.
        encoder (nn.GRU): GRU encoder module.

    Methods:
        forward(X: torch.Tensor) -> torch.Tensor:
            Forward pass of the RNN module.

    """

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
        """
        Forward pass of the RNN module.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).

        """
        _, h_t = self.encoder(X)

        return h_t[-1]
