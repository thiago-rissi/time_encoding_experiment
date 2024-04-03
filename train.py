import argparse
import pathlib
import yaml
import sys

from train_code.utils import train


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parses terminal's input.

    Parses terminal's input and directionates program's flow
    according to passed arguments.

    Args:
        args (list[str]): list of command's arguments included by user in the terminal

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="The config file for the dataset.",
        required=True,
    )

    return parser.parse_args(args)


def main(args: list[str]):
    """Receives args from terminal.

    Receives args from terminal and performs training accordingly.

    """
    args_ = parse_args(args)

    with open(pathlib.Path(args_.config), "r") as f:
        config = yaml.safe_load(f)

    with open(pathlib.Path("models/metadata.yml"), "r") as f:
        models_config = yaml.safe_load(f)

    train(models_config=models_config, **config)


if __name__ == "__main__":
    main(sys.argv[1:])
