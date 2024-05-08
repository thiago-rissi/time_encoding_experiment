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
