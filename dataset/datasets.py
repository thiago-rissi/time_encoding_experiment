import torch
import numpy.typing as npt
from aeon.datasets import load_classification, load_from_tsfile, write_to_tsfile
from dataset.utils import *
import pathlib


class GeneralDataset:
    def __init__(self) -> None:
        pass


class DeepDataset:
    def __init__(
        self,
        dataset_path: pathlib.Path,
        device: torch.device,
        normalize: bool = False,
        statistics: dict | None = None,
    ) -> None:

        self.statistics = statistics
        X, y, metadata = load_from_tsfile(str(dataset_path), return_meta_data=True)

        X = torch.tensor(X, device=device, dtype=torch.float32)

        self.n_instances = X.shape[0]
        self.n_variables = X.shape[1]
        self.t_length = X.shape[2]
        self.encoding_order = metadata["class_values"]

        if (statistics == None) and (normalize == True):
            self.statistics = calculate_stats(X)

        if normalize == True:
            X = normalize(X, self.statistics)

        self.X = X
        self.y = encode_y(y, self.encoding_order, device)
        self.timestamps = torch.arange(self.t_length, device=device)
        self.num_classes = len(self.encoding_order)

    def __len__(self) -> int:
        return self.n_instances

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_i = self.X[idx]
        y_i = self.y[idx]

        ids = torch.where(~torch.isnan(x_i[0]))[0]

        return x_i[:, ids], y_i, self.timestamps[ids]
