##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import sys

from dependency_injector.wiring import Provide, inject

 # Internal
from .core.dataset_generator import DatasetGenerator
from .core.container import Container
from services.core.utils import _add_argument

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Generate a synthetic dataset.'
)

_add_argument(parser, "--type", "day", "The type of synthetic dataset", str)

##########################################################################################################
### Main
##########################################################################################################

@inject
def main(
    generator : DatasetGenerator = Provide[Container.dataset_generator]
) -> None:

    generator.add_args(parser)
    args = parser.parse_args()
    selected_generator = generator.get(args.type)
    if selected_generator is None:
        print(f"Invalid generator type: {args.type}")
        exit(2)

    selected_generator.generate(files = generator.files, args = args)

if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    main()
