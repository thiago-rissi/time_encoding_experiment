from sktime.transformations.panel.rocket import Rocket
import numpy as np
import pickle
import pathlib
import numpy.typing as npt


def apply_rocket(
    X: npt.NDArray,
    base_path: pathlib.Path,
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

    X_trans = rocket.transform(X)
    return X_trans
