from argparse import ArgumentParser


def _add_argument(parser: ArgumentParser, name: str, default: str, arg_help: str, type = int) -> None:
    parser.add_argument(
        name,
        default=default,
        help=f"{arg_help}. Defaults to '{default}'.",
        type=type
    )