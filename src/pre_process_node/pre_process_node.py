from pre_process_node.utils import *


def pre_process(
    datasets: list[str],
    pmiss_list: list[float],
    imputer_name: str,
    impute: bool,
    primary_path: str,
    feature_path: str,
    train_ratio: float,
    nan_strategy: dict[str, str],
    max_iter: int,
    process_train: bool,
    window_mean: float,
    window_std: float,
    **kwargs,
) -> None:
    """
    Pre-processes the datasets by performing missing data imputation and feature engineering.

    Args:
        datasets (list[str]): List of dataset names to be pre-processed.
        pmiss_list (list[float]): List of missing data percentages to be considered.
        imputer_name (str): Name of the imputer class to be used for missing data imputation.
        impute (bool): Flag indicating whether missing data imputation should be performed.
        primary_path (str): Path to the primary directory where pre-processed datasets will be stored.
        feature_path (str): Path to the directory where feature engineering results will be stored.
        train_ratio (float): Ratio of training data to total data for splitting the datasets.
        nan_strategy (dict[str, str]): Dictionary mapping dataset names to nan strategy names.
        max_iter (int): Maximum number of iterations for the imputer.
        process_train (bool): Flag indicating whether to pre-process the training data without missing data.
        window_mean (float): percentage of the context window for considering as a normal distribution's mean, in order to sample missing gap's size in feature engineering.
        window_std (float): percentage of the context window for considering as a normal distribution's standard deviation, in order to sample missing gap's size in feature engineering.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    primary_path = pathlib.Path(primary_path)
    feature_path = pathlib.Path(feature_path)
    imputer_class = getattr(sys.modules[__name__], imputer_name)
    imputer = imputer_class(max_iter=max_iter)

    for dataset in datasets:
        print(f"Pre-processing dataset: {dataset}")
        initial_time = datetime.datetime.now()
        dataset_path = primary_path / dataset
        dataset_path.mkdir(exist_ok=True)
        out_path = feature_path / dataset
        out_path.mkdir(exist_ok=True)

        X_train, X_test, y_train, y_test = open_and_split(
            dataset, dataset_path, train_ratio
        )

        if process_train:
            # Firstly, pre processes dataset without missing data
            base_data(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                dataset=dataset,
                rocket_path=out_path,
                out_path=out_path / "0_missing",
            )

        for pmiss in pmiss_list:
            pre_process_step(
                X_test=X_test,
                y_test=y_test,
                pmiss=pmiss,
                out_path=out_path,
                dataset=dataset,
                nan_strategy=nan_strategy[dataset],
                imputer=imputer,
                impute=impute,
                window_mean=window_mean,
                window_std=window_std,
            )
        final_time = datetime.datetime.now()
        print(f"Total Execution time: {final_time - initial_time}")
