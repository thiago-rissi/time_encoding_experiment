from sktime.transformations.panel.rocket import Rocket
import numpy as np
import pickle
import pathlib
import numpy.typing as npt


def apply_rocket(
    X: npt.NDArray,
    y: npt.NDArray,
    base_path: pathlib.Path,
    output_path: pathlib.Path,
    X_label: str,
    y_label: str,
) -> None:

    rocket_path = base_path / "rocket.pkl"
    if rocket_path.exists():
        with open(rocket_path, "rb") as f:
            rocket = pickle.load(f)
    else:
        rocket = Rocket(n_jobs=-1, normalise=False)
        rocket = rocket.fit(X)
        with open(rocket_path, "wb") as f:
            pickle.dump(rocket, f)

    X_path = output_path / X_label
    y_path = output_path / y_label

    X_trans = rocket.transform(X)
    np.save(X_path, X_trans)
    np.save(y_path, y)
