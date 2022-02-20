##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
from typing import Callable, List
import pandas as pd

if "DISABLE_PLOTLY" in os.environ:
    sys.modules['plotly.express'] = {}

from pycaret.utils import check_metric

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import Config, add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.estimator import EstimatorTask, PyCaretClassifier, PyCaretRegressor, setup, train_ensemble_estimators
from pipeline_lib.pipelines import end_mlflow, init_mlflow, PlotParameters, save_local_results, save_mlflow_results

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Perform sizing estimation using ensemble models.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "ensemble_estimators"
CONFIG: Config = get_config(base_dir, parser)

EXPERIMENT_NAME = f"{PROJECT_NAME}_{CONFIG.get('scenario')}"
BASE_DIR = CONFIG.get("base_dir", False)
if BASE_DIR is None:
    raise Exception(f"Directory not defined error: {BASE_DIR}")

# Data
DATA = Data()
FILE_NAME = join_path(BASE_DIR, CONFIG.get("file_path"))
TARGET_VAR = CONFIG.get("target")

# Estimator
EST_TASK = CONFIG.get("est_task")
if EST_TASK == EstimatorTask.REGRESSION.value:
    ESTIMATOR = PyCaretRegressor()
else:
    ESTIMATOR = PyCaretClassifier()

# Distributed
RUN_DISTRIBUTED = CONFIG.get("run_distributed")

# Tuning
if RUN_DISTRIBUTED:
    SEARCH_ALGORITHM = CONFIG.get("distributed_search_algorithm")
    SEARCH_LIBRARY = CONFIG.get("distributed_search_library")
else:
    SEARCH_ALGORITHM = CONFIG.get("search_algorithm")
    SEARCH_LIBRARY = CONFIG.get("search_library")

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

# MLFlow
USE_MLFLOW = CONFIG.get("use_mlflow")

##########################################################################################################
### Pipeline
##########################################################################################################

def _get_prediction_metrics(metrics:dict, model, metrics_list: List[str], prefix: str, data: pd.DataFrame = None):
    predictions = ESTIMATOR.predict_model(model, data = data)
    for metric in metrics_list:
        metrics[f"{prefix}_{metric}"] = check_metric(predictions[TARGET_VAR], predictions.Label, metric)        
    return predictions

def _save_prediction_metrics(log_metric: Callable, metrics: dict, metrics_list: List[str], preds, prefix: str):
    prefixed_metrics = [ f"{prefix}_{metric}" for metric in metrics_list ]
    filtered_metrics = { k: metrics[k] for k in prefixed_metrics }

    for k, metric in filtered_metrics.items():
        log_metric(k, metric)
    for i, (y, predictions) in enumerate(zip(preds[TARGET_VAR], preds.Label)):
        log_metric(key = f"{prefix}_actual", value = y, step = i)
        log_metric(key = f"{prefix}_prediction", value = predictions, step = i)

def main() -> None:
    if RUN_DISTRIBUTED:
        import ray
        ray.init(address = CONFIG.get("RAY_ADDRESS"))  

    if USE_MLFLOW:
        import mlflow
        tmp_dir = init_mlflow(CONFIG)

    # Data split
    df = DATA.read_csv(FILE_NAME)
    data, data_unseen = DATA.train_test_split(df, frac = CONFIG.get("training_frac"), random_state = RANDOM_STATE)

    # Data preprocessing
    est_setup = setup(ESTIMATOR, CONFIG, data, EXPERIMENT_NAME)

    # Estimator fitting
    best_model, final_ensemble = train_ensemble_estimators(ESTIMATOR, CONFIG, SEARCH_ALGORITHM, SEARCH_LIBRARY)

    # Evaluate model
    metrics = {}
    if EST_TASK == EstimatorTask.REGRESSION.value:
        training_preds = _get_prediction_metrics(metrics, best_model, ["MAE", "MSE"], "training")
        testing_preds = _get_prediction_metrics(metrics, best_model, ["MAE", "MSE"], "testing", data_unseen)
        final_preds = _get_prediction_metrics(metrics, final_ensemble, ["MAE", "MSE"], "finalised", data_unseen)
        if USE_MLFLOW:
            _save_prediction_metrics(mlflow.log_metric, metrics, ["MAE", "MSE"], training_preds, "training")
            _save_prediction_metrics(mlflow.log_metric, metrics, ["MAE", "MSE"], testing_preds, "testing")
            _save_prediction_metrics(mlflow.log_metric, metrics, ["MAE", "MSE"], final_preds, "finalised")

    # Save results
    plot_params = PlotParameters(ESTIMATOR.plot_model, plots = ["residuals", "error"], model = best_model)
    if USE_MLFLOW:
        save_mlflow_results(CONFIG, final_ensemble, EXPERIMENT_NAME, tmp_dir, plot_params = plot_params)
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir)
    else:
        save_path = save_local_results(CONFIG, final_ensemble, EXPERIMENT_NAME, plot_params = plot_params)
        if len(metrics.keys()) > 0:
            pd.DataFrame(metrics, index = [0]).to_csv(join_path(save_path, f"{EXPERIMENT_NAME}_metrics.csv")) 

    if RUN_DISTRIBUTED:
        ray.shutdown()
        
if __name__ == "__main__":
    main()
     