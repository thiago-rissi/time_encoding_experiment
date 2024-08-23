import torch
import torch.nn as nn
import math
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for time encoders.

    Args:
        time_encoding_size (int): The size of the time encoding.
        dropout (float): The dropout rate.
        **kwargs: Additional keyword arguments.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        hidden_size (int): The size of the time encoding.
        div_term_even (torch.Tensor): Divisor term for even indices.
        div_term_odd (torch.Tensor): Divisor term for odd indices.
    """

    def __init__(self, time_encoding_size: int, dropout: float, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = time_encoding_size
        self.div_term_even = torch.exp(
            torch.arange(0, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )
        self.div_term_odd = torch.exp(
            torch.arange(1, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )

    def forward(self, position):
        """
        Forward pass of the positional encoding module.

        Args:
            position (torch.Tensor): The input position tensor.

        Returns:
            torch.Tensor: The positional encoding tensor.
        """
        pe = torch.empty(*position.shape, self.hidden_size, device=position.device)
        pe[..., 0::2] = torch.sin(
            position.unsqueeze(-1) * self.div_term_even.to(position.device)
        )
        pe[..., 1::2] = torch.cos(
            position.unsqueeze(-1) * self.div_term_odd.to(position.device)
        )
        return self.dropout(pe)


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """
    Converts the input time sequence `tau` into a vector representation using the given parameters.

    Args:
        tau (torch.Tensor): Input time sequence tensor.
        f (torch.nn.Module): Activation function to be applied.
        out_features (int): Number of output features.
        w (torch.Tensor): Weight tensor for the first linear transformation.
        b (torch.Tensor): Bias tensor for the first linear transformation.
        w0 (torch.Tensor): Weight tensor for the second linear transformation.
        b0 (torch.Tensor): Bias tensor for the second linear transformation.
        arg (Any, optional): Additional argument for the activation function. Defaults to None.

    Returns:
        torch.Tensor: Vector representation of the input time sequence.
    """
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class Time2Vec(nn.Module):
    """
    Time2Vec module that applies time encoding to input features.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        **kargs: Additional keyword arguments.

    Attributes:
        out_features (int): Number of output features.
        w0 (torch.Tensor): Learnable weight parameter of shape (in_features, 1).
        b0 (torch.Tensor): Learnable bias parameter of shape (1).
        w (torch.Tensor): Learnable weight parameter of shape (in_features, out_features - 1).
        b (torch.Tensor): Learnable bias parameter of shape (out_features - 1).
        f (function): Activation function to be applied.

    Methods:
        forward(tau): Performs forward pass of the Time2Vec module.

    """

    def __init__(self, in_features, out_features, **kargs):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        """
        Performs forward pass of the Time2Vec module.

        Args:
            tau (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).

        """
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class tAPE(nn.Module):
    def __init__(self, time_encoding_size: int, max_len: int, dropout: float, **kwargs):
        """
        tAPE (Time-Absolute Positional Encoding) module.

        Args:
            time_encoding_size (int): The size of the time encoding.
            max_len (int): The maximum length of the input sequence.
            dropout (float): The dropout probability.
            **kwargs: Additional keyword arguments.

        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = time_encoding_size
        self.div_term_even = torch.exp(
            torch.arange(0, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )
        self.div_term_odd = torch.exp(
            torch.arange(1, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )
        self.max_len = max_len

    def forward(self, position):
        """
        Forward pass of the tAPE module.

        Args:
            position (torch.Tensor): The input position tensor.

        Returns:
            torch.Tensor: The encoded position tensor.

        """
        pe = torch.empty(*position.shape, self.hidden_size, device=position.device)
        pe[..., 0::2] = torch.sin(
            (position.unsqueeze(-1) * self.div_term_even.to(position.device))
            * (self.hidden_size / self.max_len)
        )
        pe[..., 1::2] = torch.cos(
            (position.unsqueeze(-1) * self.div_term_odd.to(position.device))
            * (self.hidden_size / self.max_len)
        )
        return self.dropout(pe)


class GRUEncoding(nn.Module):
    def __init__(
        self,
        time_encoding_size: int,
        **kwargs,
    ) -> None:
        super(GRUEncoding, self).__init__()
        self.encoder = GRU(
            input_size=1,
            hidden_size=time_encoding_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-1]
        _, h_t = self.encoder(t.unsqueeze(-1))
        out = h_t[-1].unsqueeze(1).expand(-1, seq_len, -1)
        return out


class AbsolutePositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

        # distance = torch.matmul(self.pe, self.pe[10])
        # import matplotlib.pyplot as plt

        # plt.plot(distance.detach().numpy())
        # plt.show()

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe
        # distance = torch.matmul(self.pe, self.pe.transpose(1,0))
        # distance_pd = pd.DataFrame(distance.cpu().detach().numpy())
        # distance_pd.to_csv('learn_position_distance.csv')
        return self.dropout(x)
