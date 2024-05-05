import polars as pl
import pathlib
import numpy as np
from typing import Any
from itertools import product
import pandas as pd


def calculate_metrics(
    func_metric: Any,
    base_path: pathlib.Path,
    datasets: list[str],
    models: list[str],
    pmisses: list[int],
    func_params: dict | None = None,
) -> dict[str, list[float]]:

    results = {}
    for dataset, model in product(datasets, models):
        metric_pmiss = np.empty(shape=len(pmisses))
        test = f"{model}_{dataset}"
        for i, pmiss in enumerate(pmisses):
            df_path = base_path / (test + "_" + str(pmiss) + ".parquet")
            df = pl.read_parquet(df_path)

            if func_params == None:
                metric_pmiss[i] = func_metric(df["y_hat"], df["y"])
            else:
                metric_pmiss[i] = func_metric(df["y_hat"], df["y"], **func_params)
        results[test] = metric_pmiss

    model_mean = {}
    for model in models:
        metric_list = []
        for test, metrics in results.items():
            if model in test:
                metric_list.append(metrics)
        metric_array = np.stack(metric_list, axis=0)
        model_mean[model] = np.mean(metric_array, axis=0)

    pmiss_result = {}
    for i in range(len(pmisses)):
        models_pmiss = {model: [] for model in models}
        for model in models:
            for dataset in datasets:
                test = f"{model}_{dataset}"
                metrics = results[test]
                models_pmiss[model].append(metrics[i])

        pmiss_result[pmisses[i]] = models_pmiss

    return results, model_mean, pmiss_result


def gather_metric_cd(
    func_metric: Any,
    metric_name: str,
    base_path: pathlib.Path,
    datasets: list[str],
    models: list[str],
    pmiss: int,
    func_params: dict | None = None,
) -> pd.DataFrame:
    results = {"classifier_name": [], "dataset_name": [], metric_name: []}
    for dataset, model in product(datasets, models):
        df_path = base_path / f"{model}_{dataset}_{pmiss}.parquet"
        df = pl.read_parquet(df_path)

        if func_params == None:
            metric = func_metric(df["y_hat"], df["y"])
        else:
            metric = func_metric(df["y_hat"], df["y"], **func_params)

        results["classifier_name"].append(model)
        results["dataset_name"].append(dataset)
        results[metric_name].append(metric)

    df = pd.DataFrame(results)

    return df
