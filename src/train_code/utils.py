import pathlib
import torch
import sys
from models.ts_classifier import TSClassifier
from models.resnet import ResNet50
from train_code.trainers import *
from dataset.utils import *
from dataset.datasets import *
from sklearn.linear_model import RidgeClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.hybrid import HIVECOTEV2
import datetime


def torch_train_step(
    dataset_path: pathlib.Path,
    dataset_name: str,
    model_name: str,
    datasets_config: dict,
    config: dict,
    torch_trainer: dict,
    device: torch.device,
) -> None:
    dataset = TorchDataset(
        dataset_path=dataset_path / f"{dataset_name}_train.ts",
        dataset_name=dataset_name,
        nan_strategy=datasets_config["nan_strategy"][dataset_name],
        device=device,
    )
    model_class = getattr(sys.modules[__name__], model_name)
    model = model_class(
        num_classes=dataset.num_classes, num_features=dataset.n_variables, **config
    )

    trainer = TorchTrainer(model=model, **torch_trainer)
    save_path = (pathlib.Path(torch_trainer["base_path"]) / model_name) / dataset_name

    save_path.mkdir(parents=True, exist_ok=True)
    i_time = datetime.datetime.now()
    trainer.train(dataset=dataset, device=device, save_path=save_path, **torch_trainer)
    f_time = datetime.datetime.now()

    print(f"Training duration: {f_time - i_time}")


def general_train_step(
    dataset_path: pathlib.Path,
    dataset_name: str,
    model_name: str,
    config: dict,
    general_trainer: dict,
) -> None:
    dataset = GeneralDataset(
        dataset_path=dataset_path,
        rocket=config["rocket"],
        task="train",
        feature_first=config["feature_first"],
        dataset_name=dataset_name,
    )
    model_class = getattr(sys.modules[__name__], model_name)
    model = model_class(**config["params"])

    i_time = datetime.datetime.now()
    model = model.fit(dataset.X, dataset.y)
    f_time = datetime.datetime.now()
    print(f"Training duration: {f_time - i_time}")

    save_path = pathlib.Path(general_trainer["base_path"]) / (
        f"{model_name}_{dataset_name}.pkl"
    )

    with open(save_path, "wb") as f:
        pickle.dump(model, f)


def train(
    models_config: dict,
    datasets_config: dict,
    torch_trainer: dict,
    general_trainer: dict,
    datasets: list[str],
    models: list[str],
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = pathlib.Path("data/feature")
    for model_name in models:
        config = models_config[model_name]
        print(f"Training model: {model_name}")
        for dataset_name in datasets:
            print(f"---> Dataset: {dataset_name}")
            dataset_path = (base_path / dataset_name) / "0_missing"

            if config["torch"]:
                torch_train_step(
                    dataset_path,
                    dataset_name,
                    model_name,
                    datasets_config,
                    config,
                    torch_trainer,
                    device,
                )
            else:
                general_train_step(
                    dataset_path,
                    dataset_name,
                    model_name,
                    config,
                    general_trainer,
                )