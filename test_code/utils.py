import pathlib
import torch
import sys
from dataset.utils import *
from dataset.datasets import *
from test_code.testers import *
import os
from models.ts_classifier import TSClassifier


def load_model(
    model_basepath: str, model: nn.Module, device: torch.device
) -> nn.Module:
    model_path = list(pathlib.Path(model_basepath).rglob("*.pkl"))[-1]
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def test(
    datasets: list[str],
    models: list[str],
    pmisses: list[float],
    models_config: dict,
    datasets_config: dict,
    deep_tester: dict,
    general_tester: dict,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = pathlib.Path("data/feature")
    for model_name in models:
        config = models_config[model_name]
        print(f"Testing model: {model_name}")

        for dataset_name in datasets:
            print(f"---> Dataset: {dataset_name}")
            dataset_path = base_path / dataset_name

            for pmiss in pmisses:
                print(f"---> Missing percentage: {int(100*pmiss)}")
                pmiss_path = dataset_path / f"{int(100*pmiss)}_missing"

                if config["deep"]:
                    test_path = (
                        pmiss_path / f"{dataset_name}_{int(100*pmiss)}_nan.ts"
                        if config["test_nan"] and pmiss != 0.0
                        else pmiss_path / f"{dataset_name}_{int(100*pmiss)}.ts"
                    )
                    dataset = DeepDataset(
                        dataset_path=test_path,
                        dataset_name=dataset_name,
                        nan_strategy=datasets_config["nan_strategy"][dataset_name],
                        device=device,
                    )
                    model_class = getattr(sys.modules[__name__], model_name)
                    model = model_class(num_classes=dataset.num_classes, **config)
                    model = load_model(
                        model_basepath=os.path.join(
                            "data/models", model_name, dataset_name
                        ),
                        model=model,
                        device=device,
                    )

                    tester = DeepTester(
                        model=model, model_name=model_name, **deep_tester
                    )

                    save_path = os.path.join(
                        deep_tester["base_path"],
                        f"{model_name}_{dataset_name}_{int(100*pmiss)}.parquet",
                    )

                    tester.test(
                        dataset=dataset,
                        device=device,
                        save_path=save_path,
                        **deep_tester,
                    )
                else:

                    dataset = GeneralDataset(
                        dataset_path=pmiss_path,
                        rocket=config["rocket"],
                        task="test",
                        feature_first=config["feature_first"],
                        dataset_name=dataset_name,
                        pmiss=int(100 * pmiss),
                    )

                    with open(
                        pathlib.Path("data/models")
                        / f"{model_name}_{dataset_name}.pkl",
                        "rb",
                    ) as f:
                        model = pickle.load(f)

                    y_hat = model.predict(dataset.X)
                    print(accuracy_score(dataset.y, y_hat))
                    results = pl.DataFrame({"y": dataset.y, "y_hat": y_hat})
                    save_path = os.path.join(
                        general_tester["base_path"],
                        f"{model_name}_{dataset_name}_{int(100*pmiss)}.parquet",
                    )
                    results.write_parquet(pathlib.Path(save_path))
