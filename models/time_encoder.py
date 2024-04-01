import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, time_encoding_size: int, dropout: float, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = time_encoding_size
        self.div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )

    def forward(self, position):
        pe = torch.empty(position.shape[0], self.hidden_size, device=position.device)
        pe[:, 0::2] = torch.sin(position * self.div_term.to(position.device))
        pe[:, 1::2] = torch.cos(position * self.div_term.to(position.device))
        return self.dropout(pe)
