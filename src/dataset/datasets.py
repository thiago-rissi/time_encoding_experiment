import torch
import numpy.typing as npt
from aeon.datasets import load_classification, load_from_tsfile, write_to_tsfile
from dataset.utils import *
import pathlib
import numpy as np
import pickle
from models.time_encoder import PositionalEncoding
from pre_process_code.rocket import apply_rocket


class GeneralDataset:
    def __init__(
        self,
        dataset_path: pathlib.Path,
        rocket: bool,
        task: str,
        feature_first: bool,
        dataset_name: str,
        add_encoding: bool,
        time_encoding_size: int,
        dropout: float,
        pmiss: int = 0,
    ) -> None:
        ts_path = (
            dataset_path / (dataset_name + "_train.ts")
            if task == "train"
            else dataset_path / (dataset_name + f"_{pmiss}.ts")
        )
        X, y = load_from_tsfile(str(ts_path))
        self.timestamps = np.ones(shape=(X.shape[0], X.shape[-1])) * np.arange(
            X.shape[-1]
        )

        if add_encoding:
            time_encoder = PositionalEncoding(
                time_encoding_size=time_encoding_size, dropout=dropout
            )
            X = aggregate_time_encoding(time_encoder, X, self.timestamps)

        if rocket:
            base_path = dataset_path.parent
            X = apply_rocket(X, base_path)

        self.X = X
        metadata = get_dataset_metadata(dataset_name)
        self.encoding_order = metadata["class_values"]
        self.y = encode_y(y, self.encoding_order, device=torch.device("cpu")).numpy()


class TorchDataset:
    def __init__(
        self,
        dataset_path: pathlib.Path,
        dataset_name: str,
        nan_strategy: str,
        device: torch.device,
        normalize: bool = False,
        statistics: dict | None = None,
    ) -> None:

        self.statistics = statistics
        X, y = load_from_tsfile(str(dataset_path))
        metadata = get_dataset_metadata(dataset_name)

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
        self.y = encode_y(y, self.encoding_order, device).long()
        self.timestamps = torch.arange(self.t_length, device=device)
        self.num_classes = len(self.encoding_order)
        self.nan_strategy = nan_strategy

    def __len__(self) -> int:
        return self.n_instances

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_i = self.X[idx]
        y_i = self.y[idx]

        ids = torch.where(~torch.isnan(x_i[0]))[0]

        return x_i[:, ids], y_i, self.timestamps[ids]
