import torch
import numpy.typing as npt
from dataset.utils import *


class DeepDataset:
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_instances: int,
        n_variables: int,
        t_length: int,
        normalize: bool = False,
        statistics: dict | None = None,
    ) -> None:

        self.statistics = statistics

        self.n_instances = n_instances
        self.n_variables = n_variables
        self.t_length = t_length

        if (statistics == None) and (normalize == True):
            self.statistics = calculate_stats(X)

        if normalize == True:
            X = normalize(X, self.statistics)

        self.X = X
        self.y = y
        pass

    def __len__(self) -> int:
        return self.n_instances

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_i = self.X[idx]
        y_i = self.y[idx]

        return x_i, y_i
