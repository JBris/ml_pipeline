"""
Machine Learning Pipeline Configuration

A library of configuration tools for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os
import yaml

from typing import Any

##########################################################################################################
### Library  
##########################################################################################################

class Config:
    """Pipeline configuration class"""

    def __init__(self):
        self.config = {}
        self.env_var_prefixes = ["MLFLOW_", "RAY_"]

    def get(self, key: str) -> Any:
        """Get a configuration value."""
        return self.config.get(key)

    def get_as(self, key: str, v_type: type) -> Any:
        """Get and cast a configuration value."""
        return v_type(self.config.get(key))

    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
        return self

    def from_env(self):
        """Add configuration values from environment variables."""
        environment_vars = dict(os.environ)
        for k, v in environment_vars.items():
            for prefix in self.env_var_prefixes:
                if k.startswith(prefix): 
                    self.config[k] = v
        return self
            
    def from_yaml(self, path: str):
        """Add configuration values from a YAML file."""
        with open(path) as f:
            options: dict = yaml.safe_load(f)
            if options is not None:
                self.extract_key_values(options)
        return self

    def to_yaml(self, outfile: str, default_flow_style: bool = False):
        """Export the configuration to a YAML file."""
        with open(outfile, 'w') as f:
            yaml.dump(self.config, f, default_flow_style = default_flow_style)
        return self

    def from_parser(self, parser: argparse.ArgumentParser):
        """Add configuration values from an argument parser."""
        args = parser.parse_args()
        arg_dict = vars(args)
        return self.extract_key_values(arg_dict)

    def extract_key_values(self, options: dict):
        """Extract keys and values from a dictionary, and add them to the configuration."""
        for k, v in options.items():
            self.config[k] = v
        return self
    
def get_config(base_dir: str, parser: argparse.ArgumentParser = None) -> Config:
    """
    Get a new configuration object.

    Configuration Hierarchy:
        1. config.global.yaml
        2. The project's params.yaml
        3. Arguments from parser 
        4. The scenario file
        5. config.local.yaml
        6. An optional config.yaml file to replicate pipelines

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
    
    if config.get("from_config") != ".":
        config.from_yaml(config.get("from_config"))

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