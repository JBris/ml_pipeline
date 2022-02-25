##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import pandas as pd
import shutil
import tempfile

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.pipelines import create_local_directory, end_mlflow, get_experiment_name, init_mlflow

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Perform simple preprocessing of input data.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "preprocessing"
CONFIG = get_config(base_dir, parser)
EXPERIMENT_NAME = get_experiment_name(PROJECT_NAME, CONFIG)

# Data
DATA = Data()
FILE_NAME = DATA.get_filename(CONFIG)
TARGET_VAR = CONFIG.get("target")

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

# MLFlow
USE_MLFLOW = CONFIG.get("use_mlflow")

##########################################################################################################
### Pipeline
##########################################################################################################

def save_results(df: pd.DataFrame, path_prefix: str, copy_data_path: str):
    """Save preprocessing results."""
    preprocessed_fname = CONFIG.get("preprocessed_file_name")
    if preprocessed_fname is None:
        preprocessed_fname = f"{EXPERIMENT_NAME}_data.csv"

    data_file = join_path(path_prefix, preprocessed_fname)
    df.to_csv(data_file, index = False)
    if copy_data_path is not None:
        shutil.copy2(data_file, copy_data_path)

    config_file = CONFIG.export(path_prefix)
    return config_file, data_file

def main() -> None:
    if USE_MLFLOW:
        import mlflow
        tmp_dir = init_mlflow(CONFIG)
        save_dir = tmp_dir.name
    else:
        save_dir = create_local_directory(CONFIG)

    df = DATA.read_csv(FILE_NAME)

    # Drop nas for target
    if CONFIG.get("drop_target_nas") and TARGET_VAR is not None:
        df = df.dropna(subset = [TARGET_VAR])

    # Drop nas for all features
    if CONFIG.get("drop_nas"):
        df = df.dropna(axis = 0, how = 'any') 
    
    # Select subset of features
    include_features = CONFIG.get("include_features")
    drop_features = CONFIG.get("drop_features")
    if len(include_features) > 0:
        df = df[include_features]
    if len(drop_features) > 0:
        df = df.drop(columns = drop_features)

    # Cast columns
    col_as_type: dict = CONFIG.get("col_as_type")
    if col_as_type is not None:
        df = df.astype(col_as_type)

    # Perform query
    df = DATA.query(CONFIG, df)

    copy_data_path = CONFIG.get("copy_data_path")
    if USE_MLFLOW:
        config_file, data_file = save_results(df, save_dir, copy_data_path)
        mlflow.log_artifact(config_file)
        mlflow.log_artifact(data_file)
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir, CONFIG.get("author"))
    else:
        config_file, data_file = save_results(df, save_dir, copy_data_path)

if __name__ == "__main__":
    main()
     