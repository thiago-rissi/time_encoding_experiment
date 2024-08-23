from numpy import atleast_2d
import torch
import polars as pl
import torch.nn as nn
import sys
from models.time_encoders import *
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math


class LinearDecoder(nn.Module):
    """
    LinearDecoder module that performs linear decoding for classification tasks.

    Args:
        num_classes (int): The number of output classes.
        hidden_size (int): The size of the hidden layer.
        **kwargs: Additional keyword arguments.

    Attributes:
        internal_linear (nn.Linear): The internal linear layer.
        relu (nn.ReLU): The ReLU activation function.
        projective_linear (nn.Linear): The projective linear layer.

    Methods:
        forward(x): Performs forward pass through the decoder.

    """

    def __init__(self, num_classes, hidden_size: int, **kwargs) -> None:
        super().__init__()

        self.internal_linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.projective_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        out = self.internal_linear(x)
        out = self.relu(out)
        y_hat = self.projective_linear(out)

        return y_hat


class LinearDecoderV2(nn.Module):
    """
    LinearDecoderV2 is a class that represents a linear decoder model.

    Args:
        num_classes (int): The number of output classes.
        hidden_size (int): The size of the hidden layer.
        dropout (float): The dropout rate.
        **kwargs: Additional keyword arguments.

    Attributes:
        bn1 (nn.BatchNorm1d): Batch normalization layer.
        bn2 (nn.BatchNorm1d): Batch normalization layer.
        internal_linear (nn.Linear): Linear layer for internal processing.
        dropout (nn.Dropout): Dropout layer.
        relu (nn.ReLU): ReLU activation function.
        projective_linear (nn.Linear): Linear layer for projecting to output classes.
    """

    def __init__(self, num_classes, hidden_size: int, dropout: float, **kwargs) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.internal_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.projective_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LinearDecoderV2 model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
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
