import numpy as np
from torch import nn
from models.time_encoders import (
    tAPE,
    AbsolutePositionalEncoding,
    LearnablePositionalEncoding,
)
from models.attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec
import torch
import torch.nn as nn
from models.time_encoders import *
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        input_size: int,
        fix_pos_encode: str | None = None,
        rel_pos_encode: str | None = None,
        **kwargs,
    ):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        emb_size = input_size
        num_heads = 4
        dim_ff = 2048
        self.fix_pos_encode = fix_pos_encode
        self.rel_pos_encode = rel_pos_encode
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.LayerNorm(emb_size, eps=1e-5)

        if self.fix_pos_encode == "Sin":
            self.Fix_Position = tAPE(emb_size, dropout=0.1, max_len=seq_len)
        elif self.fix_pos_encode == "Learn":
            self.Fix_Position = LearnablePositionalEncoding(
                emb_size, dropout=0.1, max_len=seq_len
            )

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.rel_pos_encode == "Scalar":
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, 0.1)
        elif self.rel_pos_encode == "Vector":
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, 0.1)
        else:
            self.attention_layer = Attention(emb_size, num_heads, 0.1)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(0.1),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(
            in_features=emb_size,
            out_features=hidden_size,
        )

    def forward(self, X):
        x_src = self.embed_layer(X)
        if self.fix_pos_encode != None:
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.linear(out)
        # out = out.permute(1, 0, 2)
        # out = self.out(out[-1])

        return out


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
        X_hidden = X_encoded[:, -1]
        X_linear = self.linear(X_hidden)

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
