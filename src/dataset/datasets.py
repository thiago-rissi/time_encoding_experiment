import torch
import numpy.typing as npt
from aeon.datasets import load_classification, load_from_tsfile, write_to_tsfile
from dataset.utils import *
import pathlib
import numpy as np
import pickle
from models.time_encoder import PositionalEncoding
from pre_process_code.rocket import apply_rocket


def sample_random_t_inference(
    min_timestamp: float,
    max_timestamp: float,
) -> torch.Tensor:
    base_random = torch.rand(1)

    lbound = min_timestamp
    ubound = max_timestamp
    interval = (ubound - lbound) * base_random
    t_inference = lbound + interval

    return t_inference


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
        time_encoding_strategy: str,
        normalize: bool = False,
        statistics: dict | None = None,
    ) -> None:

        self.statistics = statistics
        self.device = device
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        self.normalize = normalize

        X, y = load_from_tsfile(str(dataset_path))
        metadata = get_dataset_metadata(dataset_name)

        X = torch.tensor(X, device=device, dtype=torch.float32)

        self.n_instances = X.shape[0]
        self.n_variables = X.shape[1]
        self.t_length = X.shape[2]
        self.encoding_order = metadata["class_values"]

        self.X = X
        self.y = encode_y(y, self.encoding_order, device).long()
        self.timestamps = torch.arange(self.t_length, device=device)
        self.num_classes = len(self.encoding_order)
        self.nan_strategy = nan_strategy
        self.time_encoding_strategy = time_encoding_strategy

    def __len__(self) -> int:
        return self.n_instances

    def split_dataset(
        self, split_ratio: float
    ) -> tuple["TorchDataset", "TorchDataset"]:
        train_dataset = self.__class__(
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            nan_strategy=self.nan_strategy,
            device=self.device,
            normalize=self.normalize,
            statistics=self.statistics,
            time_encoding_strategy=self.time_encoding_strategy,
        )

        split_index = int(split_ratio * len(train_dataset))

        train_dataset.X = train_dataset.X[:split_index]
        train_dataset.y = train_dataset.y[:split_index]
        train_dataset.n_instances = train_dataset.X.shape[0]

        test_dataset = self.__class__(
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            nan_strategy=self.nan_strategy,
            device=self.device,
            normalize=self.normalize,
            statistics=self.statistics,
            time_encoding_strategy=self.time_encoding_strategy,
        )

        test_dataset.X = test_dataset.X[split_index:]
        test_dataset.y = test_dataset.y[split_index:]
        test_dataset.n_instances = test_dataset.X.shape[0]

        return train_dataset, test_dataset

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        t_inf = 0.0
        if self.time_encoding_strategy == "relative":
            t_inf = sample_random_t_inference(
                min_timestamp=0.0,
                max_timestamp=self.timestamps.max().item(),
            ).to(self.X.device)

        x_i = self.X[idx]
        y_i = self.y[idx]

        ids = torch.where(~torch.isnan(x_i[0]))[0]

        return x_i[:, ids], y_i, self.timestamps[ids] - t_inf


class TorchARDataset:
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
        X, _ = load_from_tsfile(str(dataset_path))
        metadata = get_dataset_metadata(dataset_name)

        X = torch.tensor(X, device=device, dtype=torch.float32)

        self.n_instances = X.shape[0]
        self.n_variables = X.shape[1]
        self.t_length = X.shape[2]

        if (statistics == None) and (normalize == True):
            self.statistics = calculate_stats(X)

        if normalize == True:
            X = normalize(X, self.statistics)

        self.X = X
        self.timestamps = torch.arange(self.t_length, device=device)
        self.nan_strategy = nan_strategy

    def __len__(self) -> int:
        return self.n_instances

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        t_inf = sample_random_t_inference(
            min_timestamp=0.0,
            max_timestamp=self.timestamps.max().item(),
        ).to(self.X.device)
        x_i = self.X[idx]

        mask = torch.ones(x_i.shape[1], dtype=torch.bool, device=self.X.device)

        # sample 0.2 of the data to be missing
        mask[torch.randperm(mask.shape[0])[: int(0.2 * mask.shape[0])]] = False

        ids = torch.where(~torch.isnan(x_i[0]))[0]

        return x_i[:, ids], self.timestamps[ids] - t_inf, mask[ids]
