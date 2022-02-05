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
from pipeline_lib.config import add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.estimator import PyCaretClassifier, PyCaretRegressor, setup

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Perform sizing estimation using ensemble models.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_config", ".", "Override parameters using a config.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "ensemble_estimators"
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
if EST_TASK == "regression":
    ESTIMATOR = PyCaretRegressor()
else:
    ESTIMATOR = PyCaretClassifier()
EVALUATION_METRIC = CONFIG.get("evaluation_metric")

# Distributed
RUN_DISTRIBUTED = CONFIG.get("run_distributed")

# Tuning
if RUN_DISTRIBUTED:
    SEARCH_ALGORITHM = CONFIG.get("distributed_search_algorithm")
    SEARCH_LIBRARY = CONFIG.get("distributed_search_library")
else:
    SEARCH_ALGORITHM = CONFIG.get("search_algorithm")
    SEARCH_LIBRARY = CONFIG.get("search_library")

N_ESTIMATORS = CONFIG.get("n_estimators")
N_ITER = CONFIG.get("n_iter")

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

# MLFlow
USE_MLFLOW = CONFIG.get("use_mlflow")

##########################################################################################################
### Pipeline
##########################################################################################################

def main() -> None:
    if RUN_DISTRIBUTED:
        import ray
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
        mlflow.set_tracking_uri(CONFIG.get("MLFLOW_TRACKING_URI")) # Enable tracking using MLFlow
        mlflow.start_run()
        tmp_dir = tempfile.TemporaryDirectory()

    # Data split
    df = DATA.read_csv(FILE_NAME)
    data, data_unseen = DATA.train_test_split(df, frac = CONFIG.get("training_frac"), random_state = RANDOM_STATE)

    # Data preprocessing
    est_setup = setup(ESTIMATOR, CONFIG, data, EXPERIMENT_NAME)

    # Estimator fitting
    top_models = ESTIMATOR.compare_models(n_select = CONFIG.get("n_select"), sort = EVALUATION_METRIC, turbo = CONFIG.get("turbo"))
    tuned_top = [ 
        ESTIMATOR.tune_model(model, search_algorithm = SEARCH_ALGORITHM, optimize = EVALUATION_METRIC,
            search_library = SEARCH_LIBRARY, n_iter = N_ITER, custom_grid = CONFIG.get("custom_grid")) 
        for model in top_models 
    ]

    # Train ensemble estimators
    meta_model = ESTIMATOR.create_model(CONFIG.get("meta_model"))
    tuned_meta_model = ESTIMATOR.tune_model(meta_model, search_algorithm = SEARCH_ALGORITHM, 
        optimize = EVALUATION_METRIC, search_library = SEARCH_LIBRARY, n_iter = N_ITER, custom_grid = CONFIG.get("custom_grid"))  
    stacking_ensemble = ESTIMATOR.stack_models(tuned_top, optimize = EVALUATION_METRIC, meta_model = tuned_meta_model)
    blending_ensemble = ESTIMATOR.blend_models(tuned_top, optimize = EVALUATION_METRIC, choose_better = True)
    boosting_ensemble = ESTIMATOR.ensemble_model(tuned_top[0], method = "Boosting", optimize = EVALUATION_METRIC, 
        choose_better = True, n_estimators = N_ESTIMATORS)
    bagging_ensemble = ESTIMATOR.ensemble_model(tuned_top[0], method = "Bagging", optimize = EVALUATION_METRIC, 
        choose_better = True, n_estimators = N_ESTIMATORS)
    boosted_top = [ 
        ESTIMATOR.ensemble_model(model, method = "Boosting", optimize = EVALUATION_METRIC, 
            choose_better = True, n_estimators = N_ESTIMATORS)
        for model in top_models 
    ]
    boosted_blending_ensemble = ESTIMATOR.blend_models(boosted_top, optimize = EVALUATION_METRIC, choose_better = True)
    best_model = ESTIMATOR.automl(optimize = EVALUATION_METRIC)        
    final_ensemble = ESTIMATOR.finalize_model(best_model)

    def _predict_model(model, data = None):
        predictions = ESTIMATOR.predict_model(model, data = data)
        mae = check_metric(predictions[TARGET_VAR], predictions.Label, 'MAE')
        mse = check_metric(predictions[TARGET_VAR], predictions.Label, 'MSE')
        return predictions, mae, mse
    
    # Evaluate model
    training_preds, training_mae, training_mse = _predict_model(best_model)
    testing_preds, testing_mae, testing_mse = _predict_model(best_model, data_unseen)
    final_preds, final_mae, final_mse = _predict_model(final_ensemble, data_unseen)

    # Save results
    if USE_MLFLOW:
        def _save_metrics(mae, mse, preds, prefix):
            mlflow.log_metric(f"{prefix}_mae", mae)
            mlflow.log_metric(f"{prefix}_mse", mse)
            for i, (y, predictions) in enumerate(zip(preds[TARGET_VAR], preds.Label)):
                mlflow.log_metric(key = f"{prefix}_actual", value = y, step = i)
                mlflow.log_metric(key = f"{prefix}_prediction", value = predictions, step = i)

        _save_metrics(training_mae, training_mse, training_preds, "training")
        _save_metrics(testing_mae, testing_mse, testing_preds, "testing")
        _save_metrics(final_mae, final_mse, final_preds, "final")

        config_yaml = join_path(tmp_dir.name, "config.yaml")
        CONFIG.to_yaml(config_yaml)
        mlflow.log_artifact(config_yaml)
    
        mlflow.sklearn.log_model(final_ensemble, EXPERIMENT_NAME, registered_model_name = EXPERIMENT_NAME)

        for plot in ["residuals", "error"]:
            ensemble_model_plot = ESTIMATOR.plot_model(best_model, plot = plot, save = tmp_dir.name)
            mlflow.log_artifact(join_path(tmp_dir.name, ensemble_model_plot))

        mlflow.set_tag("project", PROJECT_NAME)
        mlflow.set_tag("experiment", EXPERIMENT_NAME)
        mlflow.end_run()
        tmp_dir.cleanup()
    else:
        from joblib import dump
        dump(final_ensemble, join_path("data", f"{EXPERIMENT_NAME}.joblib")) 
        CONFIG.to_yaml(join_path("data", "config.yaml"))

        pd.DataFrame({
            "training_mae": training_mae,
            "training_mse": training_mse,
            "testing_mae": testing_mae,
            "testing_mse": testing_mse,
            "final_mae": final_mae,
            "final_mse": final_mse
        }, index = [0]).to_csv(join_path("data", f"{EXPERIMENT_NAME}_metrics.csv")) 
        
        for plot in ["residuals", "error"]:
            ESTIMATOR.plot_model(best_model, plot = plot, save = "data")

if __name__ == "__main__":
    main()
     