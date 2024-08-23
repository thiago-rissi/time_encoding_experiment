from train_node.utils import *


def train(
    models_config: dict,
    datasets_config: dict,
    torch_trainer: dict,
    general_trainer: dict,
    datasets: list[str],
    models: list[str],
):
    """
    Train the specified models on the given datasets.

    Args:
        models_config (dict): A dictionary containing the configuration for each model.
        datasets_config (dict): A dictionary containing the configuration for each dataset.
        torch_trainer (dict): A dictionary containing the configuration for the torch trainer.
        general_trainer (dict): A dictionary containing the configuration for the general trainer.
        datasets (list[str]): A list of dataset names to train on.
        models (list[str]): A list of model names to train.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = pathlib.Path("data/feature")
    for model_name in models:
        config = models_config[model_name]
        encoder = models_config["Encoder"]
        decoder = models_config["Decoder"]
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
                    encoder,
                    decoder,
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
