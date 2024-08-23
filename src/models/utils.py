import torch


def min_max_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input tensor using min-max normalization.

    Args:
        x (torch.Tensor): The input tensor to be normalized.

    Returns:
        torch.Tensor: The normalized tensor.

    """
    x_norm = (x - x.min()) / (x.max() - x.min())
    return x_norm
