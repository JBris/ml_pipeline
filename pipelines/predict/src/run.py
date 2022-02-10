##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import pandas as pd

from pycaret.utils import check_metric

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import Config, add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.estimator import EstimatorTask, PyCaretClassifier, PyCaretRegressor, setup, unsupervised_setup
from pipeline_lib.pipelines import end_mlflow, init_mlflow, PlotParameters, pipeline_plots

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Perform predictions using a trained model.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "predict"
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
MODEL_NAME = CONFIG.get("model_name")
if MODEL_NAME is None:
    raise Exception(f"Model name not defined.")

MODEL_VERSION = CONFIG.get("model_version")
MODEL_STAGE = CONFIG.get("model_stage")

EST_TASK = CONFIG.get("est_task")
if EST_TASK == EstimatorTask.REGRESSION.value:
    ESTIMATOR = PyCaretRegressor()
    load_model = ESTIMATOR.load_model
    predict_model = ESTIMATOR.predict_model
elif EST_TASK == EstimatorTask.CLASSIFICATION.value:
    ESTIMATOR = PyCaretClassifier()
    load_model = ESTIMATOR.load_model
    predict_model = ESTIMATOR.predict_model
elif EST_TASK == EstimatorTask.ANOMALY_DETECTION.value:
    from pycaret.anomaly import load_model, predict_model
    load_model = load_model
    predict_model = predict_model
else:
    from pycaret.clustering import load_model, predict_model
    load_model = load_model
    predict_model = predict_model

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

# MLFlow
USE_MLFLOW = CONFIG.get("use_mlflow")

##########################################################################################################
### Pipeline
##########################################################################################################

# def _predict_reg_model(model, data = None):
#     predictions = ESTIMATOR.predict_model(model, data = data)
#     mae = check_metric(predictions[TARGET_VAR], predictions.Label, 'MAE')
#     mse = check_metric(predictions[TARGET_VAR], predictions.Label, 'MSE')
#     return predictions, mae, mse

# def _save_reg_metrics(log_metric, mae, mse, preds, prefix):
#     log_metric(f"{prefix}_mae", mae)
#     log_metric(f"{prefix}_mse", mse)
#     for i, (y, predictions) in enumerate(zip(preds[TARGET_VAR], preds.Label)):
#         log_metric(key = f"{prefix}_actual", value = y, step = i)
#         log_metric(key = f"{prefix}_prediction", value = predictions, step = i)

def main() -> None:
    # Load data
    df = DATA.read_csv(FILE_NAME) 

    # Setup
    CONFIG.set("use_mlflow", False)
    if EST_TASK == EstimatorTask.REGRESSION.value or EST_TASK == EstimatorTask.CLASSIFICATION.value:
        setup(ESTIMATOR, CONFIG, df, EXPERIMENT_NAME)
    elif EST_TASK == EstimatorTask.ANOMALY_DETECTION.value:
        est_setup = unsupervised_setup(CONFIG, df, EXPERIMENT_NAME, EstimatorTask.ANOMALY_DETECTION.value)
        from pycaret.anomaly import plot_model
        plot_params = PlotParameters(plot_model, CONFIG.get("label_feature"), [CONFIG.get("anomaly_plot")])
    else:
        est_setup = unsupervised_setup(CONFIG, df, EXPERIMENT_NAME, EstimatorTask.CLUSTERING.value)
        from pycaret.clustering import plot_model
        plot_params = PlotParameters(plot_model, CONFIG.get("label_feature"), [CONFIG.get("clustering_plot")])
    CONFIG.set("use_mlflow", USE_MLFLOW)

    # # Load model
    if USE_MLFLOW:
        import mlflow
        tmp_dir = init_mlflow(CONFIG)
        save_dir = tmp_dir.name

        model_uri = f"models:/{MODEL_NAME}"
        if MODEL_VERSION:
            model_uri += f"/{MODEL_VERSION}"
        elif MODEL_STAGE:
            model_uri += f"/{MODEL_STAGE}"
        else:
            model_uri += "/latest"

        model = mlflow.sklearn.load_model(
            model_uri = model_uri
        )
    else:
         model = load_model(MODEL_NAME)
         save_dir = "data"

    # Perform predictions
    predictions = predict_model(model, df)
    df_path = join_path(save_dir, f"{EXPERIMENT_NAME}.csv")
    predictions.to_csv(df_path)

    # Override training data labels to create new plots
    if EST_TASK == EstimatorTask.ANOMALY_DETECTION.value:
        model.labels_ = predictions.Anomaly
    elif EST_TASK == EstimatorTask.CLUSTERING.value:
        model.labels_ = predictions.Cluster.str.lstrip("Cluster ").astype("int32")
    
    # Add visualisations
    pipeline_plots(plot_params, model, save_dir, USE_MLFLOW)

    # Save results
    # plot_params = PlotParameters(ESTIMATOR.plot_model, plots = ["residuals", "error"], model = best_model)
    if USE_MLFLOW:
        config_path = CONFIG.export(tmp_dir.name)
        mlflow.log_artifact(config_path)
        mlflow.log_artifact(df_path)
        mlflow.set_tag("est_task", EST_TASK)
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir)
    else:
        CONFIG.export("data")

        # pd.DataFrame({
        #     "training_mae": training_mae,
        #     "training_mse": training_mse,
        #     "testing_mae": testing_mae,
        #     "testing_mse": testing_mse,
        #     "final_mae": final_mae,
        #     "final_mse": final_mse
        # }, index = [0]).to_csv(join_path("data", f"{EXPERIMENT_NAME}_metrics.csv")) 
        
if __name__ == "__main__":
    main()
     