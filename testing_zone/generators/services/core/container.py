from argparse import Namespace

from .dataset_generator import DatasetGenerator
from .generator_map import generator_map
from dependency_injector import containers, providers


class Container(containers.DeclarativeContainer):
    config = providers.Configuration(ini_files=["services/config.ini"])
    
    dataset_generator = providers.Singleton(
        DatasetGenerator,
        generator_map = generator_map,
        files = config.files
    )