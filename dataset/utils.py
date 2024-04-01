import torch


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
