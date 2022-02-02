##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = '$description'
)
    
add_argument(parser, "--example", 0, "An example argument")
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_config", ".", "Override parameters using a config.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "$name"
CONFIG = get_config(base_dir, parser)
if CONFIG.get("from_config").strip() != ".":
    CONFIG.from_yaml(CONFIG.get("from_config"))

EXPERIMENT_NAME = f"{PROJECT_NAME}_{CONFIG.get('scenario')}"
BASE_DIR = CONFIG.get("base_dir")
if BASE_DIR is None:
    raise Exception(f"Directory not defined error: {BASE_DIR}")

##########################################################################################################
### Pipeline
##########################################################################################################

def main() -> None:
    # mlflow.set_tracking_uri(CONFIG.get("MLFLOW_TRACKING_URI")) # Enable tracking using MLFlow
    # with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
    example = CONFIG.get('example')
    with open("data/example.txt", "w") as f: 
        f.write(f"Example: {example}")
    print(f"Created example.txt" )

if __name__ == "__main__":
    main()
     