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
        self.env_var_prefixes = ["MLFLOW_", "RAY_"]

    def get(self, key: str):
        return self.config.get(key)

    def set(self, key: str, value):
        self.config[key] = value
        return self

    def from_env(self):
        environment_vars = dict(os.environ)
        for k, v in environment_vars.items():
            for prefix in self.env_var_prefixes:
                if k.startswith(prefix): 
                    self.config[k] = v
            
    def from_yaml(self, path: str):
        with open(path) as f:
            options: dict = yaml.safe_load(f)
            if options is None:
                options = {}
            self.extract_key_values(options)
    
    def to_yaml(self, outfile: str, default_flow_style: bool = False):
        with open(outfile, 'w') as f:
            yaml.dump(self.config, f, default_flow_style = default_flow_style)

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

    Configuration Hierarchy:
        1. Environment variables
        2. config.global.yaml
        3. The project's params.yaml
        4. Arguments from parser 
        5. The scenario file.
        6. config.local.yaml

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
    config.from_env()
    config.from_yaml(f"{base_dir}/config.global.yaml")
    
    params = "params.yaml"
    if os.path.isfile(params):
        config.from_yaml(params)

    if parser is not None:
        config.from_parser(parser)

    scenario = config.get("scenario")
    if scenario is not None:
        config.from_yaml(f"{base_dir}/scenarios/{scenario}.yaml")

    local_config = f"{base_dir}/config.local.yaml"
    if os.path.isfile(local_config):
        config.from_yaml(local_config)
        
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