import torch
from aeon.datasets import load_classification, load_from_tsfile, write_to_tsfile
import pathlib


def normalize(ts: torch.Tensor, stats: list[tuple[float, float]]) -> torch.Tensor:

    for j in range(ts.shape[1]):
        ts_ = ts[:, j, :]
        mean, std = stats[j]
        ts_ = (ts_ - mean) / std
        ts[:, j, :] = ts_

    return ts


def calculate_stats(ts: torch.Tensor) -> list[tuple[float, float]]:
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
    y_encoded = []
    for y_i in y:
        id = encoding_order.index(y_i)
        y_ = id
        y_encoded.append(y_)
    return torch.tensor(y_encoded, device=device)


def encode_y(
    y: torch.Tensor, encoding_order: list, device: torch.device
) -> torch.Tensor:
    y_encoded = []
    for y_i in y:
        id = encoding_order.index(y_i)
        y_encoded.append(id)

    return torch.tensor(y_encoded, device=device, dtype=torch.float32)


def get_dataset_metadata(dataset: str):
    base_path = pathlib.Path("data/primary")
    path = (base_path / dataset) / f"{dataset}_TRAIN.ts"
    _, _, metadata = load_from_tsfile(str(path), return_meta_data=True)

    return metadata
