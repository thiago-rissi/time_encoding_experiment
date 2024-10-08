import pathlib
import torch
import sys
from models.ts_classifier import TSClassifier  # , TSAREncoderDecoder
from train_node.trainers import *
from dataset.utils import *
from dataset.datasets import *
from sklearn.linear_model import RidgeClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.hybrid import HIVECOTEV2
import datetime
import yaml


def torch_train_step(
    dataset_path: pathlib.Path,
    dataset_name: str,
    model_name: str,
    datasets_config: dict,
    config: dict,
    encoder: dict,
    decoder: dict,
    torch_trainer: dict,
    device: torch.device,
) -> None:
    """
    Trains a Torch model for classification.

    Args:
        dataset_path (pathlib.Path): The path to the dataset.
        dataset_name (str): The name of the dataset.
        model_name (str): The name of the model.
        datasets_config (dict): Configuration for the datasets.
        config (dict): Configuration for the model.
        encoder (dict): Configuration for the time encoding.
        decoder (dict): Configuration for the decoder.
        torch_trainer (dict): Configuration for the Torch trainer.
        device (torch.device): The device to train the model on.

    Returns:
        None
    """

    time_encoding_strategy = config["time_encoding"]["strategy"]

    class_trainer_config = torch_trainer["classification_training"]
    dataset = TorchDataset(
        dataset_path=dataset_path / f"{dataset_name}_train.ts",
        dataset_name=dataset_name,
        nan_strategy=datasets_config["nan_strategy"][dataset_name],
        device=device,
        time_encoding_strategy=time_encoding_strategy,
    )

    model = TSClassifier(
        num_classes=dataset.num_classes,
        num_features=dataset.n_variables,
        t_length=dataset.t_length,
        ts_encoding=encoder,
        decoder=decoder,
        model_config=config,
    )

    trainer = TorchTrainer(model=model, **class_trainer_config)
    save_path = (
        pathlib.Path(class_trainer_config["base_path"]) / model_name
    ) / dataset_name

    save_path.mkdir(parents=True, exist_ok=True)

    train_yml = save_path.parent / "train.yml"
    model_yml = save_path.parent / "model.yml"

    with open(train_yml, "w") as f:
        yaml.dump(torch_trainer, f)

    with open(model_yml, "w") as f:
        yaml.dump(config, f)

    i_time = datetime.datetime.now()
    print(f"Training Classification: {model_name}")
    trainer.train(
        dataset=dataset, device=device, save_path=save_path, **class_trainer_config
    )
    f_time = datetime.datetime.now()

    print(f"Training duration: {f_time - i_time}")


def general_train_step(
    dataset_path: pathlib.Path,
    dataset_name: str,
    model_name: str,
    config: dict,
    general_trainer: dict,
) -> None:
    """
    Train a model using the provided dataset.

    Args:
        dataset_path (pathlib.Path): The path to the dataset.
        dataset_name (str): The name of the dataset.
        model_name (str): The name of the model to train.
        config (dict): A dictionary containing configuration parameters.
        general_trainer (dict): A dictionary containing general training parameters.

    Returns:
        None
    """
    dataset = GeneralDataset(
        dataset_path=dataset_path,
        rocket=config["rocket"],
        task="train",
        feature_first=config["feature_first"],
        dataset_name=dataset_name,
        add_encoding=general_trainer["add_encoding"],
        time_encoding_size=general_trainer["time_encoding_size"],
        dropout=general_trainer["dropout"],
    )

    model_class = getattr(sys.modules[__name__], model_name)
    model = model_class(**config["params"])

    i_time = datetime.datetime.now()
    model = model.fit(dataset.X, dataset.y)
    f_time = datetime.datetime.now()
    print(f"Training duration: {f_time - i_time}")

    base_path = general_trainer["base_path"]
    if general_trainer["add_encoding"]:
        base_path = pathlib.Path(base_path)
        base_path = base_path / ("time_encoding")
        base_path.mkdir(exist_ok=True)

    if model_name == "ResNetClassifier":
        save_path = pathlib.Path(base_path) / (f"{model_name}_{dataset_name}")
        model.save(save_path)
    else:
        save_path = pathlib.Path(base_path) / (f"{model_name}_{dataset_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
