##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import mlflow
import os, sys
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

# Tuning
SEARCH_ALGORITHM = CONFIG.get("search_algorithm")
SEARCH_LIBRARY = CONFIG.get("search_library")
N_ESTIMATORS = CONFIG.get("n_estimators")
N_ITER = CONFIG.get("n_iter")

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

##########################################################################################################
### Pipeline
##########################################################################################################

def main() -> None:
    mlflow.set_tracking_uri(CONFIG.get("MLFLOW_TRACKING_URI"))
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
        client = mlflow.tracking.MlflowClient()

        # Data split
        df = DATA.read_csv(FILE_NAME)
        data, data_unseen = DATA.train_test_split(df, frac = CONFIG.get("training_frac"), random_state = RANDOM_STATE)

        # Data preprocessing
        est_setup = setup(ESTIMATOR, CONFIG, data, EXPERIMENT_NAME)

        # Estimator fitting
        top_models = ESTIMATOR.compare_models(n_select = CONFIG.get("n_select"), sort = EVALUATION_METRIC, turbo = CONFIG.get("turbo"))
        tuned_top = [ 
            ESTIMATOR.tune_model(model, search_algorithm = SEARCH_ALGORITHM, optimize = EVALUATION_METRIC,
                search_library = SEARCH_LIBRARY, n_iter = N_ITER) 
            for model in top_models 
        ]

        # Ensemble estimators
        meta_model = ESTIMATOR.create_model(CONFIG.get("meta_model"))
        tuned_meta_model = ESTIMATOR.tune_model(meta_model, search_algorithm = SEARCH_ALGORITHM, 
            optimize = EVALUATION_METRIC, search_library = SEARCH_LIBRARY, n_iter = N_ITER) 
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

        # Evaluate
        unseen_predictions = ESTIMATOR.predict_model(best_model, data = data_unseen)
        MAE = check_metric(unseen_predictions[TARGET_VAR], unseen_predictions.Label, 'MAE')
        MSE = check_metric(unseen_predictions[TARGET_VAR], unseen_predictions.Label, 'MSE')
        mlflow.log_metric("testing_mae", MAE)
        mlflow.log_metric("testing_mse", MSE)

        for i, (y, predictions) in enumerate(zip(unseen_predictions[TARGET_VAR], unseen_predictions.Label)):
            mlflow.log_metric(key = "testing_actual", value = y, step = i)
            mlflow.log_metric(key = "testing_prediction", value = predictions, step = i)
                
        config_yaml = join_path(tmp_dir, "config.yaml")
        CONFIG.to_yaml(config_yaml)
        mlflow.log_artifact(config_yaml)
        
        final_ensemble = ESTIMATOR.finalize_model(best_model)
        mlflow.sklearn.log_model(final_ensemble, EXPERIMENT_NAME, registered_model_name = EXPERIMENT_NAME)

        for plot in ["residuals", "error"]:
            ensemble_model_plot = ESTIMATOR.plot_model(final_ensemble, plot = plot, save = tmp_dir)
            mlflow.log_artifact(join_path(tmp_dir, ensemble_model_plot))


        mlflow.set_tag("project", PROJECT_NAME)
        mlflow.set_tag("experiment", EXPERIMENT_NAME)

if __name__ == "__main__":
    main()
     