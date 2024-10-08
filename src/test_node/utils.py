import pathlib
import torch
import sys
import yaml
import uniplot
from dataset.utils import *
from dataset.datasets import *
from test_node.testers import *
import os
from models.ts_classifier import TSClassifier  # TSAREncoderDecoder,
from sktime.classification.deep_learning import ResNetClassifier


def load_model(
    model_basepath: str, model: nn.Module, device: torch.device
) -> nn.Module:
    """
    Load a trained model from the specified base path.

    Args:
        model_basepath (str): The base path where the model is saved.
        model (nn.Module): The model object to load the state dictionary into.
        device (torch.device): The device to load the model onto.

    Returns:
        nn.Module: The loaded model.
    """

    model_path = sorted(
        list(pathlib.Path(model_basepath).rglob("*best.pkl")),
        key=lambda x: int(x.stem.split("_")[-2]),
    )[-1]
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def torch_test_step(
    pmiss_path: pathlib.Path,
    dataset_name: str,
    pmiss: float,
    model_name: str,
    config: dict,
    encoder: dict,
    decoder: dict,
    datasets_config: dict,
    torch_tester: dict,
    device: torch.device,
    inf_sample_size: int,
) -> float:
    """
    Perform a test step using PyTorch.

    Args:
        pmiss_path (pathlib.Path): The path to the directory containing the test data.
        dataset_name (str): The name of the dataset.
        pmiss (float): The percentage of missing values.
        model_name (str): The name of the model.
        config (dict): The configuration settings.
        encoder (dict): The encoder settings.
        decoder (dict): The decoder settings.
        datasets_config (dict): The configuration settings for the datasets.
        torch_tester (dict): The settings for the torch tester.
        device (torch.device): The device to run the test on.
        inf_sample_size (int): The size of the inference sample.

    Returns:
        float: The accuracy of the test.
    """
    test_path = (
        pmiss_path / f"{dataset_name}_{int(100*pmiss)}_nan.ts"
        if config["test_nan"] and pmiss != 0.0
        else pmiss_path / f"{dataset_name}_{int(100*pmiss)}.ts"
    )

    time_encoding_strategy = config["time_encoding"]["strategy"]

    dataset = TorchDataset(
        dataset_path=test_path,
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
    """
    Perform a general step test.

    Args:
        pmiss_path (pathlib.Path): The path to the dataset.
        dataset_name (str): The name of the dataset.
        pmiss (float): The percentage of missing values.
        model_name (str): The name of the model.
        config (dict): The configuration settings.
        general_tester (dict): The general tester settings.

    Returns:
        None
    """
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
    acc = accuracy_score(dataset.y, y_hat)
    print(acc)
    results = pl.DataFrame({"y": dataset.y, "y_hat": y_hat})
    save_path = save_path / f"{model_name}_{dataset_name}_{int(100*pmiss)}.parquet"

    results.write_parquet(save_path)
    return acc
