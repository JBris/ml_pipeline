##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import tempfile

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config
from pipeline_lib.pipelines import end_mlflow, get_experiment_name, init_mlflow

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = '$description'
)
    
add_argument(parser, "--example", 0, "An example argument")
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "$name"
CONFIG = get_config(base_dir, parser)

EXPERIMENT_NAME = get_experiment_name(PROJECT_NAME, CONFIG)
BASE_DIR = CONFIG.get("base_dir", False)
if BASE_DIR is None:
    raise Exception(f"Directory not defined error: {BASE_DIR}")

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

# Distributed
RUN_DISTRIBUTED = CONFIG.get("run_distributed")

# MLFlow
USE_MLFLOW = CONFIG.get("use_mlflow")

##########################################################################################################
### Pipeline
##########################################################################################################

def main() -> None:
    if USE_MLFLOW:
        import mlflow
        tmp_dir = init_mlflow(CONFIG)

    example = CONFIG.get('example')
    with open("data/example.txt", "w") as f: 
        f.write(f"Example: {example}")
    print(f"Created example.txt" )

    if USE_MLFLOW:
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir, CONFIG.get("author"))
        
if __name__ == "__main__":
    main()
     