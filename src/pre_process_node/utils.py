from pre_process_node.imputation import *
from pre_process_node.rocket import *
import sys
from sklearn.model_selection import train_test_split
import datetime


def open_and_split(
    dataset: str, dataset_path: pathlib.Path, train_ratio: float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Opens and splits a dataset into training and testing sets.

    Args:
        dataset (str): The name of the dataset.
        dataset_path (pathlib.Path): The path to the dataset.
        train_ratio (float): The ratio of the dataset to use for training.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]: A tuple containing the training and testing data and labels.
    """
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
    """
    Preprocesses and saves the base data for a given dataset.

    Args:
        X_train (npt.NDArray): The training input data.
        y_train (npt.NDArray): The training target data.
        X_test (npt.NDArray): The test input data.
        y_test (npt.NDArray): The test target data.
        dataset (str): The name of the dataset.
        rocket_path (pathlib.Path): The path to the rocket file.
        out_path (pathlib.Path): The output path to save the preprocessed data.

    Returns:
        None
    """
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


def pre_process_step(
    X_test: npt.NDArray,
    y_test: npt.NDArray,
    pmiss: float,
    out_path: pathlib.Path,
    dataset: str,
    nan_strategy: str,
    imputer: Any,
    impute: bool,
    window_mean: float,
    window_std: float,
) -> None:
    """
    Pre-processes the input data by creating missing points and imputing data points.

    Args:
        X_test (npt.NDArray): The input feature matrix.
        y_test (npt.NDArray): The input target vector.
        pmiss (float): The percentage of missing values to create.
        out_path (pathlib.Path): The output path to save the pre-processed data.
        dataset (str): The name of the dataset.
        nan_strategy (str): The strategy to create missing values.
        imputer (Any): The imputer object to use for imputation.
        impute (bool): Flag indicating whether to perform imputation or not.
        window_mean (float): percentage of the context window for considering as a normal distribution's mean, in order to sample missing gap's size in feature engineering.
        window_std (float): percentage of the context window for considering as a normal distribution's standard deviation, in order to sample missing gap's size in feature engineering.

    Returns:
        None
    """
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
        window_mean=window_mean,
        window_std=window_std,
    )
    write_to_tsfile(
        X=X_nan.swapaxes(1, 2),
        y=y_test,
        problem_name=f"{dataset}_{int(100*pmiss)}_nan.ts",
        path=pmiss_path,
    )

    if impute:
        print("----> Imputing data points")
        i_time = datetime.datetime.now()
        X_imp = impute(X_nan=X_nan, n_instances=n_instances, imputer=imputer)
        f_time = datetime.datetime.now()
        print(f"Imputation time: {f_time - i_time}")

        write_to_tsfile(
            X=X_imp.swapaxes(1, 2),
            y=y_test,
            problem_name=f"{dataset}_{int(100*pmiss)}.ts",
            path=pmiss_path,
        )
