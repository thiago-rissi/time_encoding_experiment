import polars as pl
import pathlib
import numpy as np
from typing import Any


def calculate_metrics(
    func_metric: function,
    base_path: pathlib.Path,
    datasets: list[str],
    models: list[str],
    pmisses=list[int],
) -> dict[str, list[float]]:

    results = {}
    for dataset, model in zip(datasets, models):
        metric_pmiss = np.empty(shape=len(pmisses))
        test = f"{model}_{dataset}"
        for i, pmiss in enumerate(pmisses):
            df_path = base_path / test + str(pmiss) + ".parquet"
            df = pl.read_csv(df_path)
            metric_pmiss[i] = func_metric(df["y_hat"], df["y"])
        results[test] = metric_pmiss

    model_mean = {}
    for model in models:
        metric_list = []
        for test, metrics in results.items():
            if model in test:
                metric_list.append(metrics)
        model_mean[model] = np.mean(metric_list)

    pmiss_result = {}
    for i in range(1, len(pmisses + 1)):
        models_pmiss = {model: [] for model in models}
        for model in models:
            for dataset in datasets:
                test = f"{model}_{dataset}"
                metrics = results[test]
                models_pmiss[model].append(metrics[i])

        pmiss_result[pmisses[i]] = models_pmiss

    return results, model_mean, pmiss_result


def calculate_mean_ranks(
    results: dict[str, Any], models: list[str], datasets: list[str]
) -> dict[str, list[float]]:

    ranks = {}
    for dataset in datasets:
        pass
