##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import pandas as pd
import tempfile

from pycaret.utils import check_metric

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import Config, add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.estimator import EstimatorTask, PyCaretClassifier, PyCaretRegressor, setup
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
BASE_DIR = CONFIG.get("base_dir")
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

def _predict_reg_model(model, data = None):
    predictions = ESTIMATOR.predict_model(model, data = data)
    mae = check_metric(predictions[TARGET_VAR], predictions.Label, 'MAE')
    mse = check_metric(predictions[TARGET_VAR], predictions.Label, 'MSE')
    return predictions, mae, mse

def _save_reg_metrics(log_metric, mae, mse, preds, prefix):
    log_metric(f"{prefix}_mae", mae)
    log_metric(f"{prefix}_mse", mse)
    for i, (y, predictions) in enumerate(zip(preds[TARGET_VAR], preds.Label)):
        log_metric(key = f"{prefix}_actual", value = y, step = i)
        log_metric(key = f"{prefix}_prediction", value = predictions, step = i)

def main() -> None:
    # if RUN_DISTRIBUTED:
        # ray.init(address=os.environ["ip_head"])
        # print("Nodes in the Ray cluster:")
        # print(ray.nodes())
        # with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
        #     client = mlflow.tracking.MlflowClient()

        # ray.init(
        #     CONFIG.get("RAY_ADDRESS"),
        #             object_store_memory=200 * 1024 * 1024,
        # )

    if USE_MLFLOW:
        import mlflow
        tmp_dir = init_mlflow(CONFIG)

    # Data split
    df = DATA.read_csv(FILE_NAME)
    data, data_unseen = DATA.train_test_split(df, frac = CONFIG.get("training_frac"), random_state = RANDOM_STATE)

    # Data preprocessing
    est_setup = setup(ESTIMATOR, CONFIG, data, EXPERIMENT_NAME)

    # Estimator fitting
    best_model, final_ensemble = train_ensemble_estimators()

    # Evaluate model
    training_preds, training_mae, training_mse = _predict_reg_model(best_model)
    testing_preds, testing_mae, testing_mse = _predict_reg_model(best_model, data_unseen)
    final_preds, final_mae, final_mse = _predict_reg_model(final_ensemble, data_unseen)

    # Save results
    plot_params = PlotParameters(ESTIMATOR.plot_model, plots = ["residuals", "error"], model = best_model)
    if USE_MLFLOW:
        save_mlflow_results(CONFIG, final_ensemble, EXPERIMENT_NAME, tmp_dir, plot_params = plot_params)

        _save_reg_metrics(mlflow.log_metric, training_mae, training_mse, training_preds, "training")
        _save_reg_metrics(mlflow.log_metric, testing_mae, testing_mse, testing_preds, "testing")
        _save_reg_metrics(mlflow.log_metric, final_mae, final_mse, final_preds, "final")

        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir)
    else:
        save_local_results(CONFIG, final_ensemble, EXPERIMENT_NAME, plot_params = plot_params)

        pd.DataFrame({
            "training_mae": training_mae,
            "training_mse": training_mse,
            "testing_mae": testing_mae,
            "testing_mse": testing_mse,
            "final_mae": final_mae,
            "final_mse": final_mse
        }, index = [0]).to_csv(join_path("data", f"{EXPERIMENT_NAME}_metrics.csv")) 
        
if __name__ == "__main__":
    main()
     