import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from typing import Any
from aeon.datasets import load_classification, load_from_tsfile, write_to_tsfile
from xgboost import XGBRegressor
import cupy as cp


class MissForest:
    def __init__(
        self,
        max_iter,
        tol: float = 0.001,
        strategy: str = "mean",
        fill_values: float | None = None,
    ) -> None:

        self.regressor = XGBRegressor(random_state=42, device="cuda", n_jobs=-1)
        self.max_iter = max_iter
        self.tol = tol
        self.strategy = strategy
        self.fill_value = fill_values

    def init_values(self, X: npt.NDArray) -> npt.NDArray:

        imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        X = imputer.fit_transform(X)

        return X

    def step(
        self,
        X_: npt.NDArray,
        columns: set,
        c: float,
        known_ids: npt.NDArray,
        nan_ids: npt.NDArray,
        lenght: int,
    ) -> npt.NDArray:

        x_c = tuple(columns - {c})
        y = X_[known_ids, c]
        x = X_[known_ids][:, x_c]

        regressor = self.regressor.fit(cp.array(x), cp.array(y))

        nan_length = sum(nan_ids)
        id = random.randint(0, lenght - nan_length)
        test_ids = np.arange(id, id + nan_length)
        x_test = X_[test_ids][:, x_c]

        y_imp = regressor.predict(x_test)
        X_[nan_ids, c] = y_imp

    def calculate_epsilon(self, X_, X_old, X_0, known_ids) -> float:
        epsilon = np.max(np.abs(X_ - X_old)) / np.max(np.abs(X_0[known_ids]))
        return epsilon

    def fit_transform(self, X: npt.NDArray) -> npt.NDArray:
        n_iter = 0
        known_ids = ~np.isnan(X)
        nan_ids = np.isnan(X)
        epsilon = np.inf
        length = X.shape[0]
        columns = {i for i in range(X.shape[-1])}

        X_0 = self.init_values(X)

        X_old = X_0.copy()
        X_ = X_0.copy()

        while (n_iter <= self.max_iter) and (epsilon > self.tol):
            for c in range(len(columns)):
                self.step(X_, columns, c, known_ids[:, c], nan_ids[:, c], length)

            if n_iter > 0:
                epsilon = self.calculate_epsilon(X_, X_old, X_0, known_ids)

            X_old = X_.copy()
            n_iter += 1

        return X_


def create_nan_ts(
    ts: npt.NDArray,
    pmiss: float,
    strategy: str = "same",
    tol: float = 0.025,
    window_mean: float = 0.1,
    window_std: float = 0.02,
) -> npt.NDArray:
    miss_points = 0
    total_points = len(ts)
    mean = window_mean * total_points
    std = window_std * total_points
    n_insert = int(np.random.normal(mean, std))
    mask = np.zeros(total_points)

    pmiss_ = miss_points / total_points
    while pmiss_ < pmiss:
        id_insert = random.randint(0, total_points - 1)
        mask[id_insert : id_insert + n_insert] = 1
        pmiss_ = sum(mask) / total_points

    if pmiss_ > abs(pmiss - tol):
        n_remove = int(abs(pmiss - pmiss_) * total_points)
        mask[id_insert + n_insert - n_remove : id_insert + n_insert] = 0

    mask = mask.astype(bool)
    if strategy == "same":
        ts[mask, :] = np.nan
    elif strategy == "diff":
        ts[mask] = np.nan

    return ts


def create_nan_dataset(
    X: npt.NDArray,
    pmiss: float,
    n_instances: int,
    n_variables: int,
    nan_strategy: str = "same",
    window_mean: float = 0.1,
    window_std: float = 0.02,
):

    xs_nan = []
    for i in range(n_instances):
        x_i = X[i]

        if nan_strategy == "same":
            x_i_nan = create_nan_ts(
                x_i.swapaxes(0, 1),
                pmiss=pmiss,
                strategy=nan_strategy,
                window_mean=window_mean,
                window_std=window_std,
            )
        elif nan_strategy == "diff":
            x_i_nan = np.stack(
                [
                    create_nan_ts(x_i[j], pmiss=pmiss, strategy=nan_strategy)
                    for j in range(n_variables)
                ],
                axis=1,
            )

        xs_nan.append(x_i_nan)

    X_nan = np.stack(xs_nan)
    return X_nan


def impute(
    X_nan: npt.NDArray,
    n_instances: int,
    imputer: MissForest | IterativeImputer | Any,
) -> npt.NDArray:

    xs = []
    for i in tqdm(range(n_instances)):
        x_i_nan = X_nan[i]
        x_i_fitted = imputer.fit_transform(x_i_nan)
        xs.append(x_i_fitted)

    X_imputed = np.stack(xs)
    return X_imputed
