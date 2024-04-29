import pathlib
import torch
import sys

import uniplot
from dataset.utils import *
from dataset.datasets import *
from test_code.testers import *
import os
from models.ts_classifier import TSAREncoderDecoder, TSClassifier
from sktime.classification.deep_learning import ResNetClassifier


def load_model(
    model_basepath: str, model: nn.Module, device: torch.device
) -> nn.Module:
    model_path = sorted(
        list(pathlib.Path(model_basepath).rglob("*.pkl")),
        key=lambda x: int(x.stem.split("_")[-1]),
    )[-1]
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def torch_test_step(
    pmiss_path: pathlib.Path,
    dataset_name: str,
    pmiss: float,
    model_name: str,
    config: dict,
    datasets_config: dict,
    torch_tester: dict,
    device: torch.device,
) -> float:
    test_path = (
        pmiss_path / f"{dataset_name}_{int(100*pmiss)}_nan.ts"
        if config["test_nan"] and pmiss != 0.0
        else pmiss_path / f"{dataset_name}_{int(100*pmiss)}.ts"
    )
    dataset = TorchDataset(
        dataset_path=test_path,
        dataset_name=dataset_name,
        nan_strategy=datasets_config["nan_strategy"][dataset_name],
        device=device,
    )

    ar_model = TSAREncoderDecoder(input_size=dataset.n_variables, **config)

    model = TSClassifier(
        num_classes=dataset.num_classes,
        num_features=dataset.n_variables,
        t_length=dataset.t_length,
        encoder=ar_model,
        **config,
    )

    model = load_model(
        model_basepath=os.path.join("data/models", model_name, dataset_name),
        model=model,
        device=device,
    )

    tester = TorchTester(model=model, model_name=model_name, **torch_tester)

    save_path = os.path.join(
        torch_tester["base_path"],
        f"{model_name}_{dataset_name}_{int(100*pmiss)}.parquet",
    )

    acc = tester.test(
        dataset=dataset,
        device=device,
        save_path=save_path,
        **torch_tester,
    )

    return acc


def general_step_tester(
    pmiss_path: pathlib.Path,
    dataset_name: str,
    pmiss: float,
    model_name: str,
    config: dict,
    general_tester: dict,
) -> None:
    dataset = GeneralDataset(
        dataset_path=pmiss_path,
        rocket=config["rocket"],
        task="test",
        feature_first=config["feature_first"],
        dataset_name=dataset_name,
        pmiss=int(100 * pmiss),
        add_encoding=general_tester["add_encoding"],
        time_encoding_size=general_tester["time_encoding_size"],
        dropout=general_tester["dropout"],
    )

    model_path = pathlib.Path("data/models")
    save_path = pathlib.Path(general_tester["base_path"])
    if general_tester["add_encoding"]:
        model_path = model_path / "time_encoding"
        save_path = save_path / "time_encoding"
        save_path.mkdir(exist_ok=True)

    if model_name == "ResNetClassifier":
        model = ResNetClassifier()
        model = model.load_from_path(model_path / f"{model_name}_{dataset_name}.zip")
    else:
        with open(
            model_path / f"{model_name}_{dataset_name}.pkl",
            "rb",
        ) as f:
            model = pickle.load(f)

    y_hat = model.predict(np.nan_to_num(dataset.X))
    print(accuracy_score(dataset.y, y_hat))
    results = pl.DataFrame({"y": dataset.y, "y_hat": y_hat})
    save_path = save_path / f"{model_name}_{dataset_name}_{int(100*pmiss)}.parquet"

    results.write_parquet(save_path)


def test(
    datasets: list[str],
    models: list[str],
    pmisses: list[float],
    models_config: dict,
    datasets_config: dict,
    torch_tester: dict,
    general_tester: dict,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = pathlib.Path("data/feature")
    all_results = {}
    for model_name in models:
        config = models_config[model_name]
        print(f"Testing model: {model_name}")

        for dataset_name in datasets:
            print(f"---> Dataset: {dataset_name}")
            dataset_path = base_path / dataset_name
            all_miss = []
            for pmiss in pmisses:
                print(f"---> Missing percentage: {int(100*pmiss)}")
                pmiss_path = dataset_path / f"{int(100*pmiss)}_missing"

                if config["torch"]:
                    acc = torch_test_step(
                        pmiss_path,
                        dataset_name,
                        pmiss,
                        model_name,
                        config,
                        datasets_config,
                        torch_tester,
                        device,
                    )
                    all_miss.append(acc)
                else:
                    general_step_tester(
                        pmiss_path,
                        dataset_name,
                        pmiss,
                        model_name,
                        config,
                        general_tester,
                    )

            uniplot.plot(
                xs=[pmisses],
                ys=[all_miss],
            )
