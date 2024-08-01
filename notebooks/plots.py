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


def plot_metric(
    results: dict[str, list],
    metric: str,
    pmiss: list = [0, 20, 40, 60, 70, 80, 90],
    title: str = "",
    save_name: str = "output",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, result in results.items():
        ax.plot(pmiss, result, label=name)

    ax.set_xlabel("Missing percentage")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    plt.tight_layout()
    plt.savefig(f"../notebooks/figures/{save_name}.png")
