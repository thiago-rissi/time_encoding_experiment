from pre_process_code.imputation import *
from pre_process_code.rocket import *
import sys
from sklearn.model_selection import train_test_split


def open_and_split(
    dataset: str, dataset_path: pathlib.Path, train_ratio: float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    path_train = dataset_path / (dataset + "_TRAIN.ts")
    path_test = dataset_path / (dataset + "_TEST.ts")

    X_train, y_train = load_from_tsfile(str(path_train))
    X_test, y_test = load_from_tsfile(str(path_test))
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio)

    return X_train, X_test, y_train, y_test


def base_data(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    X_test: npt.NDArray,
    y_test: npt.NDArray,
    dataset: str,
    rocket_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    write_to_tsfile(
        X=X_train,
        y=y_train,
        problem_name=f"{dataset}_train.ts",
        path=out_path,
    )
    write_to_tsfile(
        X=X_test,
        y=y_test,
        problem_name=f"{dataset}_0.ts",
        path=out_path,
    )
    apply_rocket(
        X=X_train,
        y=y_train,
        base_path=rocket_path,
        output_path=out_path,
        X_label="X_train.npy",
        y_label="y_train.npy",
    )
    apply_rocket(
        X=X_test,
        y=y_test,
        base_path=rocket_path,
        output_path=out_path,
        X_label="X_test.npy",
        y_label="y_test.npy",
    )


def pre_process_step(
    X_test: npt.NDArray,
    y_test: npt.NDArray,
    pmiss: float,
    out_path: pathlib.Path,
    dataset: str,
    nan_strategy: str,
    imputer: Any,
) -> None:
    print(f"---> Missing percentage: {int(100*pmiss)}%")
    n_instances = X_test.shape[0]
    n_variables = X_test.shape[1]
    pmiss_path = out_path / f"{int(100*pmiss)}_missing"

    print("----> Creating missing points")
    X_nan = create_nan_dataset(
        X=X_test.copy(),
        pmiss=pmiss,
        n_instances=n_instances,
        n_variables=n_variables,
        nan_strategy=nan_strategy,
    )
    write_to_tsfile(
        X=X_nan.swapaxes(1, 2),
        y=y_test,
        problem_name=f"{dataset}_{int(100*pmiss)}_nan.ts",
        path=pmiss_path,
    )
    print("----> Imputing data points")
    X_imp = impute(X_nan=X_nan, n_instances=n_instances, imputer=imputer)
    write_to_tsfile(
        X=X_imp.swapaxes(1, 2),
        y=y_test,
        problem_name=f"{dataset}_{int(100*pmiss)}.ts",
        path=pmiss_path,
    )
    apply_rocket(
        X=X_imp,
        y=y_test,
        base_path=out_path,
        output_path=pmiss_path,
        X_label="X_test.npy",
        y_label="y_test.npy",
    )


def pre_process(
    datasets: list[str],
    pmiss_list: list[float],
    imputer_name: str,
    primary_path: str,
    feature_path: str,
    train_ratio: float,
    nan_strategy: dict[str, str],
    max_iter: int,
    **kwargs,
) -> None:

    primary_path = pathlib.Path(primary_path)
    feature_path = pathlib.Path(feature_path)
    imputer_class = getattr(sys.modules[__name__], imputer_name)

    for dataset in datasets:
        print(f"Pre-processing dataset: {dataset}")
        dataset_path = primary_path / dataset
        dataset_path.mkdir(exist_ok=True)
        out_path = feature_path / dataset
        out_path.mkdir(exist_ok=True)

        X_train, X_test, y_train, y_test = open_and_split(
            dataset, dataset_path, train_ratio
        )

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

        imputer = imputer_class(max_iter=max_iter)
        for pmiss in pmiss_list:
            pre_process_step(
                X_test=X_test,
                y_test=y_test,
                pmiss=pmiss,
                out_path=out_path,
                dataset=dataset,
                nan_strategy=nan_strategy[dataset],
                imputer=imputer,
            )
