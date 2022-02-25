##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys

from tpot import TPOTRegressor, TPOTClassifier

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.distributed import close_dask, init_dask
from pipeline_lib.estimator import EstimatorTask
from pipeline_lib.pipelines import create_local_directory, end_mlflow, get_experiment_name, init_mlflow

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'An AutoML project using the TPOT algorithm.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "tpot_automl"
CONFIG = get_config(base_dir, parser)
EXPERIMENT_NAME = get_experiment_name(PROJECT_NAME, CONFIG)

# Data
DATA = Data()
FILE_NAME = DATA.get_filename(CONFIG)
TARGET_VAR = CONFIG.get("target")

# Estimator
EST_TASK = CONFIG.get("est_task")

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

# Distributed
RUN_DISTRIBUTED = CONFIG.get("run_distributed")

# MLFlow
USE_MLFLOW = CONFIG.get("use_mlflow")

##########################################################################################################
### Pipeline
##########################################################################################################

def save_results(est, path_prefix: str):
    """Save TPOT results."""
    config_file = CONFIG.export(path_prefix)
    pipeline_file = join_path(path_prefix, f"{EXPERIMENT_NAME}.py")
    est.export(pipeline_file)
    return config_file, pipeline_file

def main() -> None:
    if USE_MLFLOW:
        import mlflow
        tmp_dir = init_mlflow(CONFIG)

    if RUN_DISTRIBUTED:
        client = init_dask()
        
    df = DATA.read_csv(FILE_NAME)
    df = DATA.query(CONFIG, df)
    columns = [TARGET_VAR]
    ignore_features = CONFIG.get("ignore_features")
    if type(ignore_features) is list:
        columns += ignore_features
    X = df.drop(columns = columns)
    y = df[TARGET_VAR].values

    save_dir = create_local_directory(CONFIG)
    kwargs = {
        "cv" : CONFIG.get("fold"), "random_state" : RANDOM_STATE, "use_dask" : RUN_DISTRIBUTED, 
        "verbosity" : 2, "warm_start" : False, "periodic_checkpoint_folder": save_dir
    }

    for config_arg in ["generations", "population_size", "n_jobs", "max_time_mins", "max_eval_time_mins",
        "config_dict", "early_stop"]:
        kwargs[config_arg] = CONFIG.get(config_arg)

    if EST_TASK == EstimatorTask.REGRESSION.value:
        est = TPOTRegressor(**kwargs)
    else:
        est = TPOTClassifier(**kwargs)

    est.fit(X, y)
    # if RUN_DISTRIBUTED:
    #     import joblib
    #     with joblib.parallel_backend("dask"): # @TODO Run using Ray backend instead?
    #         pipeline_optimizer.fit(X, y)
    # else:
    #     pipeline_optimizer.fit(X, y)
        
    if USE_MLFLOW:
        config_file, pipeline_file = save_results(est, tmp_dir.name)
        mlflow.log_artifact(config_file)
        mlflow.log_artifact(pipeline_file)
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir, CONFIG.get("author"))
    else:
        save_results(est, save_dir)

    if RUN_DISTRIBUTED:
        close_dask(client)

if __name__ == "__main__":
    main()
     