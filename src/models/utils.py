import torch


def find_minimun_divisor(threshold: int, dividend: int) -> int:
    md = threshold
    if dividend <= threshold:
        return dividend
    while dividend % md != 0:
        md += 1
    return md


def min_max_norm(x: torch.Tensor) -> torch.Tensor:
    x_norm = (x - x.min()) / (x.max() - x.min())
    return x_norm


def set_dynamic_size(nheads: int, T: int, input_size: int) -> int:
    size = nheads * T - input_size

    return size
