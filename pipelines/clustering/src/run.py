##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import tempfile

from pycaret.clustering import (create_model, assign_model, plot_model, tune_model)

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.estimator import unsupervised_setup

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Perform clustering on a dataset.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_config", ".", "Override parameters using a config.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "clustering"
CONFIG = get_config(base_dir, parser)

EXPERIMENT_NAME = f"{PROJECT_NAME}_{CONFIG.get('scenario')}"
BASE_DIR = CONFIG.get("base_dir")
if BASE_DIR is None:
    raise Exception(f"Directory not defined error: {BASE_DIR}")

# Data
DATA = Data()
FILE_NAME = join_path(BASE_DIR, CONFIG.get("file_path"))
TARGET_VAR = CONFIG.get("target")

# Model
MODEL = CONFIG.get("clustering_model") #kmeans, ap, meanshift, sc, hclust, dbscan, optics, birch, kmodes 

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
        mlflow.set_tracking_uri(CONFIG.get("MLFLOW_TRACKING_URI")) # Enable tracking using MLFlow
        mlflow.start_run()
        tmp_dir = tempfile.TemporaryDirectory()

    # Data split
    df = DATA.read_csv(FILE_NAME)

    # Data preprocessing
    est_setup = unsupervised_setup(df, CONFIG, EXPERIMENT_NAME, "clustering")

    # Estimator fitting
    model = create_model(MODEL, num_clusters = CONFIG.get("num_clusters"), ground_truth = CONFIG.get("ground_truth"))

    if TARGET_VAR:
        model = tune_model(model, supervised_target = TARGET_VAR, supervised_estimator = CONFIG.get("supervised_estimator"),
            optimize = CONFIG.get("evaluation_metric"), fold = CONFIG.get("k_fold"), custom_grid = CONFIG.get("custom_grid")) 

    assigned_df = assign_model(model)

    if USE_MLFLOW:
        config_yaml = join_path(tmp_dir.name, "config.yaml")
        CONFIG.to_yaml(config_yaml)
        mlflow.log_artifact(config_yaml)
        
        mlflow.sklearn.log_model(model, EXPERIMENT_NAME, registered_model_name = EXPERIMENT_NAME)
        clustering_plot = plot_model(model, plot = CONFIG.get("clustering_plot"), save = tmp_dir.name, 
            feature = CONFIG.get("label_feature"), label = True)
        mlflow.log_artifact(join_path(tmp_dir.name, clustering_plot))

        df_path = join_path(tmp_dir.name, f"{EXPERIMENT_NAME}.csv")
        assigned_df.to_csv(df_path)
        mlflow.log_artifact(df_path)

        mlflow.set_tag("project", PROJECT_NAME)
        mlflow.set_tag("experiment", EXPERIMENT_NAME)
        mlflow.end_run()
        tmp_dir.cleanup()
    else:
        from joblib import dump
        dump(model, join_path("data", f"{EXPERIMENT_NAME}.joblib")) 
        CONFIG.to_yaml(join_path("data", "config.yaml"))
        assigned_df.to_csv(join_path("data", f"{EXPERIMENT_NAME}.csv")) 
        plot_model(model, plot = CONFIG.get("clustering_plot"), save = "data", 
            feature = CONFIG.get("label_feature"), label = True)

if __name__ == "__main__":
    main()
     