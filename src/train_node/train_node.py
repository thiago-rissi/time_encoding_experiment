from train_node.utils import *


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
