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
    **kwargs,
) -> None:

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
            )
        final_time = datetime.datetime.now()
        print(f"Total Execution time: {final_time - initial_time}")
