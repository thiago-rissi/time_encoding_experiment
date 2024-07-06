import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
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
        pe = torch.empty(*position.shape, self.hidden_size, device=position.device)
        pe[..., 0::2] = torch.sin(
            position.unsqueeze(-1) * self.div_term_even.to(position.device)
        )
        pe[..., 1::2] = torch.cos(
            position.unsqueeze(-1) * self.div_term_odd.to(position.device)
        )
        return self.dropout(pe)


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features, **kargs):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
