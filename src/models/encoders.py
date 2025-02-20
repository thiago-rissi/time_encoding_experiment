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
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from pytorch_tcn import TCN


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
        X_encoded = X_encoded[:, -1]
        X_linear = self.linear(X_encoded)

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


class GNN(torch.nn.Module):
    def __init__(self, num_node_features: int, hidden_size: int):
        super().__init__()
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = torch.concat(
            [x, torch.zeros((batch_size, 1, x.size(-1))).to(x.device)], dim=1
        )
        num_nodes = x.size(1)
        x = x.view(-1, x.size(-1))

        edge_index = torch.cat(
            [
                torch.tensor(
                    [
                        [i + b * num_nodes, j + b * num_nodes]
                        for i in range(num_nodes)
                        for j in range(num_nodes)
                        if i != j
                    ],
                    dtype=torch.long,
                ).t()
                for b in range(batch_size)
            ],
            dim=1,
        ).to(x.device)
        x = self.conv1(x, edge_index)
        h = self.conv2(x, edge_index)
        h = h.view(batch_size, num_nodes, -1)
        return h[:, -1, :]


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = TCN(
            num_inputs=input_size,
            num_channels=[3, 9, 27],
            kernel_size=3,
            input_shape="NLC",
        )
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(400)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_encoded = self.encoder(X)
        X_flattened = self.flatten(X_encoded)
        X_linear = self.linear(X_flattened)
        return X_linear
