"""
Machine Learning Pipeline Configuration

A library of configuration tools for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

import argparse
import os
import yaml

##########################################################################################################
### Library  
##########################################################################################################

class Config:
    """Pipeline configuration class"""

    def __init__(self):
        self.config = {}

    def get(self, key: str):
        return self.config.get(key)

    def set(self, key: str, value):
        self.config[key] = value
        return self

    def from_yaml(self, path: str):
        with open(path) as f:
            options: dict = yaml.safe_load(f)
            if options is None:
                options = {}
            self.extract_key_values(options)
    
    def from_parser(self, parser: argparse.ArgumentParser):
        args = parser.parse_args()
        arg_dict = vars(args)
        self.extract_key_values(arg_dict)

    def extract_key_values(self, options: dict):
        for k, v in options.items():
            self.config[k] = v
    
def get_config(base_dir: str, parser: argparse.ArgumentParser = None) -> Config:
    """
    Get a new configuration object.

    Parameters
    --------------
    base_dir: str
        The base directory for the ML pipeline project.

    Returns
    ---------
    config: Config
        The configuration object.
    """
    config = Config()
    config.from_yaml(f"{base_dir}/config.yaml")

    local_config = f"{base_dir}/config.local.yaml"
    if os.path.isfile(local_config):
        config.from_yaml(local_config)
    
    params = "params.yaml"
    if os.path.isfile(params):
        config.from_yaml(params)

    if parser is not None:
        config.from_parser(parser)

    return config

def add_argument(parser: argparse.ArgumentParser, name: str, default, arg_help: str, 
    type: type = int, choices: list = None) -> None:
    """
    Add argument to the argument parser.

    Parameters
    --------------
    parser: parser
        The argument parser.
    name: str
        The argument name.
    default: any
        The default value.
    arg_help: str
        The argument help text.
    type: type
        The argument type.
    choices: list
        Choices for multiple choice options.
    """

    parser.add_argument(
        name,
        default = default,
        help = f"{arg_help}. Defaults to '{default}'.",
        type = type,
        choices = choices
    )