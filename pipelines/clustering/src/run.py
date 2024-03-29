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
from pipeline_lib.estimator import EstimatorTask, get_unsupervised_custom_model, unsupervised_setup
from pipeline_lib.pipelines import (end_mlflow, get_experiment_name, init_mlflow, PlotParameters, 
    save_local_results, save_mlflow_results)

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Perform clustering on a dataset.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "clustering"
CONFIG = get_config(base_dir, parser)
EXPERIMENT_NAME = get_experiment_name(PROJECT_NAME, CONFIG)

# Data
DATA = Data()
FILE_NAME = DATA.get_filename(CONFIG)
TARGET_VAR = CONFIG.get("target")

# Model
if CONFIG.get("custom_clustering_model"):
    MODEL_ID = CONFIG.get("custom_clustering_model")
    MODEL = get_unsupervised_custom_model(MODEL_ID, EstimatorTask.CLUSTERING.value)
else:
    MODEL_ID = CONFIG.get("clustering_model")
    MODEL = MODEL_ID #kmeans, ap, meanshift, sc, hclust, dbscan, optics, birch, kmodes 

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
        tmp_dir = init_mlflow(CONFIG)

    # Data split
    df = DATA.read_csv(FILE_NAME)
    df = DATA.query(CONFIG, df)
    # Data preprocessing
    est_setup = unsupervised_setup(CONFIG, df, EXPERIMENT_NAME, EstimatorTask.CLUSTERING.value)
    # Estimator fitting
    model = create_model(MODEL, num_clusters = CONFIG.get("num_clusters"), ground_truth = CONFIG.get("ground_truth"))
    # Tune model
    if TARGET_VAR:
        model = tune_model(model, supervised_target = TARGET_VAR, supervised_estimator = CONFIG.get("supervised_estimator"),
            optimize = CONFIG.get("evaluation_metric"), fold = CONFIG.get("fold"), custom_grid = CONFIG.get("custom_grid").get(MODEL_ID)) 
    # Assign clusters
    assigned_df = assign_model(model)

    # Save results
    plot_params = PlotParameters(plot_model, CONFIG.get("label_feature"), [CONFIG.get("clustering_plot")])
    if USE_MLFLOW:
        save_mlflow_results(CONFIG, model, EXPERIMENT_NAME, tmp_dir, assigned_df, plot_params)        
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir, CONFIG.get("author"))
    else:
        save_local_results(CONFIG, model, EXPERIMENT_NAME, assigned_df, plot_params)

if __name__ == "__main__":
    main()
     