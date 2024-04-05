import pathlib
import torch
import sys
from models.ts_classifier import TSClassifier
from train_code.trainers import *
from dataset.utils import *
from dataset.datasets import *
from sklearn.linear_model import RidgeClassifierCV


def train(
    models_config: dict,
    datasets_config: dict,
    deep_trainer: dict,
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

            if config["deep"]:
                dataset = DeepDataset(
                    dataset_path=dataset_path / f"{dataset_name}_train.ts",
                    nan_strategy=datasets_config["nan_strategy"][dataset],
                    device=device,
                )
                model_class = getattr(sys.modules[__name__], model_name)
                model = model_class(num_classes=dataset.num_classes, **config)

                trainer = DeepTrainer(model=model, **deep_trainer)
                save_path = (
                    pathlib.Path(deep_trainer["base_path"]) / model_name
                ) / dataset_name

                save_path.mkdir(parents=True, exist_ok=True)
                trainer.train(
                    dataset=dataset, device=device, save_path=save_path, **deep_trainer
                )
            else:
                dataset = GeneralDataset(
                    dataset_path=dataset_path,
                    rocket=config["rocket"],
                    task="train",
                    feature_first=config["feature_first"],
                    dataset_name=dataset_name,
                )
                model_class = getattr(sys.modules[__name__], model_name)
                model = model_class(**config["params"])
                model = model.fit(dataset.X, dataset.y)
                save_path = pathlib.Path(general_trainer["base_path"]) / (
                    model_name + ".pkl"
                )

                with open(save_path, "wb") as f:
                    pickle.dump(model, f)
