##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys

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
MODEL_PATH = CONFIG.get("model_path")
if MODEL_NAME is None and MODEL_PATH is None:
    raise Exception("A model name or model path must be provided.")

MODEL_VERSION = CONFIG.get("model_version")
MODEL_STAGE = CONFIG.get("model_stage")

EST_TASK = CONFIG.get("est_task")
if EST_TASK == EstimatorTask.REGRESSION.value:
    from pycaret.regression import load_model, predict_model
    ESTIMATOR = PyCaretRegressor()
elif EST_TASK == EstimatorTask.CLASSIFICATION.value:
    from pycaret.classification import load_model, predict_model
    ESTIMATOR = PyCaretClassifier()
elif EST_TASK == EstimatorTask.ANOMALY_DETECTION.value:
    from pycaret.anomaly import load_model, predict_model, plot_model
else:
    from pycaret.clustering import load_model, predict_model, plot_model

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

# MLFlow
USE_MLFLOW = CONFIG.get("use_mlflow")

##########################################################################################################
### Pipeline
##########################################################################################################

def _save_metrics(log_metric, metrics, preds):
    for k, v in metrics.items():
        log_metric(k, v)

    preds_sample = preds.sample(n = 300, random_state = RANDOM_STATE)
    for i, (y, predictions) in enumerate(zip(preds_sample[TARGET_VAR], preds_sample.Label)):
        log_metric(key = f"actual_value", value = y, step = i)
        log_metric(key = f"predicted_value", value = predictions, step = i)

def main() -> None:
    # Load data
    df = DATA.read_csv(FILE_NAME) 
    
    # Setup
    CONFIG.set("use_mlflow", False) # Skip MLFlow logging
    if EST_TASK == EstimatorTask.REGRESSION.value or EST_TASK == EstimatorTask.CLASSIFICATION.value:
        setup(ESTIMATOR, CONFIG, df, EXPERIMENT_NAME)
    elif EST_TASK == EstimatorTask.ANOMALY_DETECTION.value:
        est_setup = unsupervised_setup(CONFIG, df, EXPERIMENT_NAME, EstimatorTask.ANOMALY_DETECTION.value)
    else:
        est_setup = unsupervised_setup(CONFIG, df, EXPERIMENT_NAME, EstimatorTask.CLUSTERING.value)
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
         model = load_model(MODEL_PATH)
         save_dir = "data"

    # Perform predictions
    predictions = predict_model(model, df)
    df_path = join_path(save_dir, f"{EXPERIMENT_NAME}.csv")
    predictions.to_csv(df_path)

    metrics = {}
    # Replace existing training data labels to create new plots
    if EST_TASK == EstimatorTask.ANOMALY_DETECTION.value:
        model.labels_ = predictions.Anomaly
        plot_params = PlotParameters(plot_model, CONFIG.get("label_feature"), [CONFIG.get("anomaly_plot")])
    elif EST_TASK == EstimatorTask.CLUSTERING.value:
        model.labels_ = predictions.Cluster.str.lstrip("Cluster ").astype("int32")
        plot_params = PlotParameters(plot_model, CONFIG.get("label_feature"), [CONFIG.get("clustering_plot")])
    elif EST_TASK == EstimatorTask.REGRESSION.value:
        plot_params = PlotParameters(ESTIMATOR.plot_model, plots = ["residuals", "error"])
        metrics["mae"] = check_metric(predictions[TARGET_VAR], predictions.Label, 'MAE')
        metrics["mse"] = check_metric(predictions[TARGET_VAR], predictions.Label, 'MSE')

        if USE_MLFLOW:
            _save_metrics(mlflow.log_metric, metrics, predictions)

    # Add plots
    pipeline_plots(plot_params, model, save_dir, USE_MLFLOW)

    # Save results
    if USE_MLFLOW:
        config_path = CONFIG.export(tmp_dir.name)
        mlflow.log_artifact(config_path)
        mlflow.log_artifact(df_path)
        mlflow.set_tag("est_task", EST_TASK)
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir)
    else:
        CONFIG.export("data")
        if len(metrics.keys()) > 0:
            pd.DataFrame(metrics, index = [0]).to_csv(join_path("data", f"{EXPERIMENT_NAME}_metrics.csv")) 
        
if __name__ == "__main__":
    main()
     