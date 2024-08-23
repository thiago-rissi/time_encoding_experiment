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
) -> tuple[dict, dict, dict, dict]:
    """
    Calculate metrics for given datasets and models.

    Args:
        func_metric (Any): The metric function to calculate.
        base_path (pathlib.Path): The base path where the datasets are located.
        datasets (list[str]): The list of dataset names.
        models (list[str]): The list of model names.
        pmisses (list[int]): The list of pmiss values.
        func_params (dict | None, optional): Additional parameters for the metric function. Defaults to None.

    Returns:
        tuple[dict, dict, dict, dict]: A tuple containing the following:
            - results (dict): The calculated metrics for each combination of dataset and model.
            - model_mean (dict): The mean metric values for each model.
            - pmiss_result (dict): The metric values for each pmiss value and model.
            - datasets_results (dict): The metric values for each dataset and model.
    """
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
            if model == test.split("_")[0]:
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

    datasets_results = {}
    for dataset in datasets:
        models_results = {}
        for model in models:
            test = f"{model}_{dataset}"
            models_results[model] = results[test]
        datasets_results[dataset] = models_results

    return results, model_mean, pmiss_result, datasets_results


def gather_metric_cd(
    func_metric: Any,
    metric_name: str,
    base_path: pathlib.Path,
    datasets: list[str],
    models: list[str],
    pmiss: int,
    func_params: dict | None = None,
) -> pd.DataFrame:
    """
    Calculate the specified metric for different models and datasets.

    Args:
        func_metric (Any): The metric function to be used.
        metric_name (str): The name of the metric.
        base_path (pathlib.Path): The base path where the data files are located.
        datasets (list[str]): The list of dataset names.
        models (list[str]): The list of model names.
        pmiss (int): The value of pmiss.
        func_params (dict | None, optional): Additional parameters for the metric function. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metric values for each model and dataset.
    """
    results = {}
    for model in models:
        results[model] = {}
        for dataset in datasets:
            df_path = base_path / f"{model}_{dataset}_{pmiss}.parquet"
            df = pl.read_parquet(df_path)
            if func_params == None:
                metric = func_metric(df["y_hat"], df["y"])
            else:
                metric = func_metric(df["y_hat"], df["y"], **func_params)
            results[model][dataset] = metric

    df = pd.DataFrame(results)

    return df
