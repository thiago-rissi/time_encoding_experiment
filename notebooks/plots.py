import matplotlib.pyplot as plt
import seaborn as sns
import numpy.typing as npt
import numpy as np


def compare_imputation(
    x_nan: npt.NDArray,
    x_imp: npt.NDArray,
    x: npt.NDArray,
    title: str,
    ts_id: int,
    ylabel: str = "Time Serie",
    xlabel: str = "Time",
):
    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    fig.suptitle(title)
    t = np.arange(x.shape[0])
    axes[0].plot(t, x[:, ts_id], label="original", color="blue")
    axes[1].plot(t, x_nan[:, ts_id], label="missing", color="red")
    axes[2].plot(t, x_imp[:, ts_id], label="imputed", color="orange")

    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)

    axes[2].set_xlabel(xlabel)

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.tight_layout()
