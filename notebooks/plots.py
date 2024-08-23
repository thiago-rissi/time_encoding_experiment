import matplotlib.pyplot as plt
import seaborn as sns
import numpy.typing as npt
import numpy as np
import pathlib


def compare_imputation(
    x_nan: npt.NDArray,
    x_imp: npt.NDArray,
    x: npt.NDArray,
    title: str,
    ts_id: int,
    ylabel: str = "Time Serie",
    xlabel: str = "Time",
) -> None:
    """
    Compare the original, missing, and imputed time series data.

    Parameters:
        x_nan (npt.NDArray): The array containing the missing values.
        x_imp (npt.NDArray): The array containing the imputed values.
        x (npt.NDArray): The array containing the original values.
        title (str): The title of the plot.
        ts_id (int): The index of the time series to compare.
        ylabel (str, optional): The label for the y-axis. Defaults to "Time Serie".
        xlabel (str, optional): The label for the x-axis. Defaults to "Time".

    Returns:
        None
    """
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
    """
    Plot a metric against missing percentage.

    Args:
        results (dict[str, list]): A dictionary containing the results for different names.
        metric (str): The metric to be plotted.
        pmiss (list, optional): The missing percentage values. Defaults to [0, 20, 40, 60, 70, 80, 90].
        title (str, optional): The title of the plot. Defaults to "".
        save_name (str, optional): The name of the output file. Defaults to "output".

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, result in results.items():
        ax.plot(pmiss, result, label=name)

    ax.set_xlabel("Missing percentage (%)")
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
    pathlib.Path("../notebooks/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../notebooks/figures/{save_name}.png")
