##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import tempfile

from tpot import TPOTRegressor, TPOTClassifier

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config
from pipeline_lib.data import Data, join_path

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'An AutoML project using the TPOT algorithm.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_config", ".", "Override parameters using a config.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "tpot_automl"
CONFIG = get_config(base_dir, parser)

EXPERIMENT_NAME = f"{PROJECT_NAME}_{CONFIG.get('scenario')}"
BASE_DIR = CONFIG.get("base_dir")
if BASE_DIR is None:
    raise Exception(f"Directory not defined error: {BASE_DIR}")

# Data
DATA = Data()
FILE_NAME = join_path(BASE_DIR, CONFIG.get("file_path"))
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

def get_regressor(**kwargs):
    """Return a TPOT regressor."""
    return TPOTRegressor(**kwargs)

def get_classifier(**kwargs):
    """Return a TPOT classifier."""
    return TPOTClassifier(**kwargs)

def main() -> None:
    if USE_MLFLOW:
        import mlflow
        mlflow.set_tracking_uri(CONFIG.get("MLFLOW_TRACKING_URI")) # Enable tracking using MLFlow
        mlflow.start_run()
        tmp_dir = tempfile.TemporaryDirectory()

    if RUN_DISTRIBUTED:
        import dask.distributed as dd
        import dask_mpi as dm
        # Initialise Dask cluster and store worker files in current work directory
        dm.initialize(local_directory=os.getcwd())
        client = dd.Client()

    df = DATA.read_csv(FILE_NAME)
    X = df.drop(TARGET_VAR, axis = 1)
    y = df[TARGET_VAR].values

    kwargs = {
        "generations" : CONFIG.get("generations"), 
        "population_size" : CONFIG.get("population_size"), 
        "cv" : CONFIG.get("k_fold"), 
        "random_state" : RANDOM_STATE, 
        "n_jobs" : -1, 
        "max_time_mins" : CONFIG.get("max_time_mins"), 
        "max_eval_time_mins" : CONFIG.get("max_eval_time_mins"), 
        "use_dask" : RUN_DISTRIBUTED, 
        "verbosity" : 2, 
        "warm_start" : False, 
        "config_dict" : CONFIG.get("config_dict")
    }
    if EST_TASK == "regression":
        pipeline_optimizer = get_regressor(**kwargs)
    else:
        pipeline_optimizer = get_classifier(**kwargs)

    pipeline_optimizer.fit(X, y)
    # if RUN_DISTRIBUTED:
    #     import joblib
    #     with joblib.parallel_backend("dask"):
    #         pipeline_optimizer.fit(X, y)
    # else:
    #     pipeline_optimizer.fit(X, y)

    if USE_MLFLOW:
        pipeline_file = join_path(tmp_dir.name, f"{EXPERIMENT_NAME}.py")
        pipeline_optimizer.export(pipeline_file)
        mlflow.log_artifact(pipeline_file)

        config_yaml = join_path(tmp_dir.name, "config.yaml")
        CONFIG.to_yaml(config_yaml)
        mlflow.log_artifact(config_yaml)
        mlflow.end_run()
        tmp_dir.cleanup()
    else:
        pipeline_optimizer.export(join_path("data", f"{EXPERIMENT_NAME}.py"))
        CONFIG.to_yaml(join_path("data", "config.yaml"))

    if RUN_DISTRIBUTED:
        client.close()

if __name__ == "__main__":
    main()
     