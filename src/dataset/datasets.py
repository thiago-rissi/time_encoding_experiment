import torch
import numpy.typing as npt
from aeon.datasets import load_classification, load_from_tsfile, write_to_tsfile
from dataset.utils import *
import pathlib
import numpy as np
import pickle
from models.time_encoders import PositionalEncoding
from pre_process_node.rocket import apply_rocket


def sample_random_t_inference(
    min_timestamp: float,
    max_timestamp: float,
) -> torch.Tensor:
    """
    Generate a random timestamp for inference within the given range.

    Args:
        min_timestamp (float): The minimum timestamp value.
        max_timestamp (float): The maximum timestamp value.

    Returns:
        torch.Tensor: A randomly generated timestamp for inference.
    """
    base_random = torch.rand(1)

    lbound = min_timestamp
    ubound = max_timestamp
    interval = (ubound - lbound) * base_random
    t_inference = lbound + interval

    return t_inference


class GeneralDataset:
    """
    A class representing a general dataset.

    Args:
        dataset_path (pathlib.Path): The path to the dataset.
        rocket (bool): Whether to apply the ROCKET transformation.
        task (str): The task type, either "train" or "test".
        feature_first (bool): Whether the features are in the first dimension of the input data.
        dataset_name (str): The name of the dataset.
        add_encoding (bool): Whether to add time encoding to the input data.
        time_encoding_size (int): The size of the time encoding.
        dropout (float): The dropout rate for the time encoding.
        pmiss (int, optional): The percentage of missing values. Defaults to 0.

    Attributes:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The encoded target labels.
        encoding_order (list): The order of the class values.

    """

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
    """
    TorchDataset class represents a dataset for PyTorch models.

    Args:
        dataset_path (pathlib.Path): The path to the dataset.
        dataset_name (str): The name of the dataset.
        nan_strategy (str): The strategy to handle missing values.
        device (torch.device): The device to use for computation.
        time_encoding_strategy (str): The strategy to encode time information.
        normalize (bool, optional): Whether to normalize the data. Defaults to False.
        statistics (dict | None, optional): Statistics of the dataset. Defaults to None.

    Attributes:
        statistics (dict | None): Statistics of the dataset.
        device (torch.device): The device used for computation.
        dataset_path (pathlib.Path): The path to the dataset.
        dataset_name (str): The name of the dataset.
        normalize (bool): Whether the data is normalized.
        X (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        timestamps (torch.Tensor): The timestamps of the data.
        num_classes (int): The number of classes in the dataset.
        nan_strategy (str): The strategy to handle missing values.
        time_encoding_strategy (str): The strategy to encode time information.

    Methods:
        __len__(): Returns the number of instances in the dataset.
        split_dataset(split_ratio: float) -> tuple[TorchDataset, TorchDataset]: Splits the dataset into train and test datasets.
        __getitem__(idx: int) -> tuple[torch.Tensor, torch.Tensor]: Returns the data and target for a given index.

    """

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
        if self.time_encoding_strategy == "delta":
            t_inf = sample_random_t_inference(
                min_timestamp=0.0,
                max_timestamp=self.timestamps.max().item(),
            ).to(self.X.device)

        x_i = self.X[idx]
        y_i = self.y[idx]

        ids = torch.where(~torch.isnan(x_i[0]))[0]

        return x_i[:, ids], y_i, self.timestamps[ids] - t_inf
