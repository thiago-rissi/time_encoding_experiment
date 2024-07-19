from test_node.utils import *


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
        print(f"Testing model: {model_name}")

        config = models_config[model_name]
        encoder = models_config["Encoder"]
        decoder = models_config["Decoder"]
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
                        encoder,
                        decoder,
                        datasets_config,
                        torch_tester,
                        device,
                        torch_tester["inf_sample_size"],
                    )
                    all_miss.append(acc)
                else:
                    acc = general_step_tester(
                        pmiss_path,
                        dataset_name,
                        pmiss,
                        model_name,
                        config,
                        general_tester,
                    )
                    all_miss.append(acc)

            uniplot.plot(
                xs=[pmisses],
                ys=[all_miss],
            )
