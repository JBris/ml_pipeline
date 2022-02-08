##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import tempfile

from pycaret.anomaly import (create_model, assign_model, plot_model, tune_model, models)

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.estimator import EstimatorTask, get_unsupervised_custom_model, unsupervised_setup
from pipeline_lib.pipelines import end_mlflow, init_mlflow, PlotParameters, save_local_results, save_mlflow_results

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Perform anomaly detection of datasets.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "anomaly"
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
if CONFIG.get("custom_anomaly_model"):
    MODEL = get_unsupervised_custom_model(CONFIG.get("custom_anomaly_model"), EstimatorTask.ANOMALY_DETECTION.value)
else:
    MODEL = CONFIG.get("anomaly_model") #abod, cluster, cof, histogram, knn, lof, svm, pca, mcd, sod, sos, iforest

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
    # Data preprocessing
    est_setup = unsupervised_setup(CONFIG, df, EXPERIMENT_NAME, EstimatorTask.ANOMALY_DETECTION.value)
    # Estimator fitting
    model = create_model(MODEL, fraction = CONFIG.get("contamination_fraction"))
    # Tune model
    if TARGET_VAR:
        model = tune_model(model, supervised_target = TARGET_VAR, supervised_estimator = CONFIG.get("supervised_estimator"),
            optimize = CONFIG.get("evaluation_metric"), fold = CONFIG.get("fold"), custom_grid = CONFIG.get("custom_grid")) 
    # Assign anomalies
    assigned_df = assign_model(model, score = True)

    # Save results
    plot_params = PlotParameters(plot_model, CONFIG.get("label_feature"), [CONFIG.get("anomaly_plot")])
    if USE_MLFLOW:
        save_mlflow_results(CONFIG, model, EXPERIMENT_NAME, tmp_dir, assigned_df, plot_params)        
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir)
    else:
        save_local_results(CONFIG, model, EXPERIMENT_NAME, assigned_df, plot_params)
        
if __name__ == "__main__":
    main()
     