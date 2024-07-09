import argparse
import pathlib
import yaml
import sys

from pre_process_node.pre_process_node import pre_process


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


def main_pre_process(args: list[str] | None = None, path: str | None = None):
    """Receives args from terminal.

    Receives args from terminal and performs training accordingly.

    """
    if path == None:
        args_ = parse_args(args)
        path = pathlib.Path(args_.config)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    pre_process(**config)


if __name__ == "__main__":
    main_pre_process(sys.argv[1:])
