import torch
from aeon.datasets import load_classification, load_from_tsfile, write_to_tsfile
import pathlib
import numpy.typing as npt
import numpy as np
from typing import Any


def normalize_ts(ts: torch.Tensor, stats: list[tuple[float, float]]) -> torch.Tensor:
    """
    Normalize a time series tensor using the provided statistics.

    Args:
        ts (torch.Tensor): The input time series tensor of shape (batch_size, num_channels, sequence_length).
        stats (list[tuple[float, float]]): The list of mean and standard deviation statistics for each channel.

    Returns:
        torch.Tensor: The normalized time series tensor.

    """

    for j in range(ts.shape[1]):
        ts_ = ts[:, j, :]
        mean, std = stats[j]
        ts_ = (ts_ - mean) / std
        ts[:, j, :] = ts_

    return ts


def calculate_stats(ts: torch.Tensor) -> list[tuple[float, float]]:
    """
    Calculate the mean and standard deviation for each column of a 3D tensor.

    Args:
        ts (torch.Tensor): The input tensor of shape (batch_size, num_columns, sequence_length).

    Returns:
        list[tuple[float, float]]: A list of tuples, where each tuple contains the mean and standard deviation
        for a column of the input tensor.

    """

    stats = []
    for j in range(ts.shape[1]):
        ts_ = ts[:, j, :]
        mean = torch.mean(ts_)
        std = torch.std(ts_)
        stats.append((mean, std))

    return stats


def encode_y(
    y: torch.Tensor, encoding_order: list, device: torch.device
) -> torch.Tensor:
    """
    Encodes the input tensor `y` based on the given `encoding_order` list and `device`.

    Args:
        y (torch.Tensor): The input tensor to be encoded.
        encoding_order (list): The list specifying the encoding order.
        device (torch.device): The device to be used for encoding.

    Returns:
        torch.Tensor: The encoded tensor.

    """
    y_encoded = []
    for y_i in y:
        id = encoding_order.index(y_i)
        y_encoded.append(id)

    return torch.tensor(y_encoded, device=device, dtype=torch.float32)


def get_dataset_metadata(dataset: str):
    """
    Retrieves the metadata of a dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        metadata: The metadata of the dataset.
    """

    base_path = pathlib.Path("data/primary")
    path = (base_path / dataset) / f"{dataset}_TRAIN.ts"
    _, _, metadata = load_from_tsfile(str(path), return_meta_data=True)

    return metadata


def aggregate_time_encoding(
    time_encoder: Any,
    X: npt.NDArray,
    timestamps: npt.NDArray,
) -> npt.NDArray:
    """
    Aggregates time encoding to the input data.

    Args:
        time_encoder (Any): The time encoder function.
        X (npt.NDArray): The input data.
        timestamps (npt.NDArray): The timestamps.

    Returns:
        npt.NDArray: The aggregated time encoded data.
    """
    encoded_timestamps = []
    for i in range(X.shape[0]):
        timestamp = timestamps[i]
        t_inference = timestamp[-1] + 1
        x_rel_timestamp = np.copy(timestamp)
        x_rel_timestamp = np.roll(x_rel_timestamp, shift=-1)
        x_rel_timestamp[-1] = t_inference
        encoded_timestamps.append(
            time_encoder(
                torch.tensor(
                    (x_rel_timestamp - t_inference), device=torch.device("cpu")
                ).unsqueeze(1)
            )
        )
    encoded_timestamps = torch.stack(encoded_timestamps, dim=0)
    X_encoded = np.concatenate([X.swapaxes(1, 2), encoded_timestamps.numpy()], axis=-1)
    X_encoded = X_encoded.swapaxes(1, 2)

    return X_encoded
