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

# Internal
from pipeline_lib.data import join_path

##########################################################################################################
### Library  
##########################################################################################################

# White list for environment variables
# Variables are filtered by prefix.
ENV_VAR_WHITE_LIST = ["ML_PIPELINE_", "MLFLOW_", "RAY_"]

class Config:
    """Pipeline configuration class"""

    def __init__(self):
        self.config = {}
        self.params_override = {} 
        self.env_var_prefixes = ENV_VAR_WHITE_LIST

    def get(self, k: str, export: bool = True) -> Any:
        """Get a configuration value."""
        v = self.config.get(k)
        if export:
            self.params_override[k] = v
        return v

    def get_as(self, k: str, v_type: type, export: bool = True) -> Any:
        """Get and cast a configuration value."""
        v = v_type(self.config.get(k))
        if export:
            self.params_override[k] = v
        return v

    def set(self, k: str, v: Any, export: bool = True):
        """Set a configuration value."""
        self.config[k] = v
        if export:
            self.params_override[k] = v
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

    def to_yaml(self, config: dict, outfile: str, default_flow_style: bool = False):
        """Export a dictionary to a YAML file."""
        with open(outfile, 'w') as f:
            yaml.dump(config, f, default_flow_style = default_flow_style)
        return self

    def export(self, path: str = "", default_flow_style: bool = False) -> str:
        """Export active configuration to a params.override.yaml file."""
        config_path = join_path(path, "params.override.yaml")
        self.to_yaml(self.params_override, config_path, default_flow_style)
        return config_path

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
        1. White-listed environment variables (see ENV_VAR_WHITE_LIST)
        2. params.global.yaml
        3. The project's params.yaml
        4. Arguments from parser 
        5. The scenario file
        6. params.local.yaml
        7. An optional params.override.yaml file to replicate pipelines

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
    config.from_yaml(f"{base_dir}/params.global.yaml")
    
    params = "params.yaml"
    if os.path.isfile(params):
        config.from_yaml(params)

    if parser is not None:
        config.from_parser(parser)

    scenario = config.get("scenario")
    if scenario is not None:
        scenario_file = f"{base_dir}/scenarios/{scenario}.yaml"
        if os.path.isfile(scenario_file):
            config.from_yaml(scenario_file)
        config.set("scenario", scenario.replace("/", "_"))

    local_config = f"{base_dir}/params.local.yaml"
    if os.path.isfile(local_config):
        config.from_yaml(local_config)
    
    if config.get("from_params") != ".":
        sizing_dir = config.get("base_dir")
        config.from_yaml(config.get("from_params"))
        config.set("base_dir", sizing_dir)
        
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