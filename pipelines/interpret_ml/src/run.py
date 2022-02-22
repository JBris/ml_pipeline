##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import numpy as np
import os, sys
import pandas as pd

from pycaret.utils import check_metric

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.custom_estimators import CUSTOM_CLASSIFIERS, CUSTOM_REGRESSORS
from pipeline_lib.config import Config, add_argument, get_config
from pipeline_lib.data import Data, join_path
from pipeline_lib.estimator import EstimatorTask, PyCaretClassifier, PyCaretRegressor, setup
from pipeline_lib.pipelines import (create_local_directory, end_mlflow, init_mlflow, PlotParameters, 
    save_local_results, save_mlflow_results, pipeline_plots)

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Pipeline for interpretable machine learning.'
)
    
add_argument(parser, "--base_dir", ".", "The base project directory", str)
add_argument(parser, "--scenario", ".", "The pipeline scenario file", str)
add_argument(parser, "--from_params", ".", "Override parameters using a params.override.yaml file", str)

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "interpret_ml"
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
LINEAR_MODELS = ["lr", "ridge"]
TREE_MODEL = "dt"
EBM_KEY = "ebm"
if EST_TASK == EstimatorTask.REGRESSION.value:
    ESTIMATOR = PyCaretRegressor()
    LINEAR_MODELS.append("lasso")
    EBM = CUSTOM_REGRESSORS.get(EBM_KEY)
else:
    ESTIMATOR = PyCaretClassifier()
    EBM = CUSTOM_CLASSIFIERS.get(EBM_KEY)

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

def main() -> None:
    if RUN_DISTRIBUTED:
        import ray
        ray.init(address = CONFIG.get("RAY_ADDRESS"))  

    if USE_MLFLOW:
        import mlflow
        tmp_dir = init_mlflow(CONFIG)
        save_dir = tmp_dir.name
    else:
        save_dir = create_local_directory(CONFIG)

    # Data split
    df = DATA.read_csv(FILE_NAME) 
    df = DATA.query(CONFIG, df)
    # Data preprocessing
    est_setup = setup(ESTIMATOR, CONFIG, df, EXPERIMENT_NAME)

    # Interpretable model fitting
    coefficients = []
    custom_grid = CONFIG.get("custom_grid")
    evaluation_metric = CONFIG.get("evaluation_metric")
    n_iter = CONFIG.get("n_iter")
    early_stopping_algo = CONFIG.get("early_stopping_algo")
    early_stop = CONFIG.get("early_stop")
    plot_params = PlotParameters(ESTIMATOR.plot_model, plots = ["residuals", "error", "feature_all", "rfe"])

    for linear_model in LINEAR_MODELS:
        trained_lm = ESTIMATOR.create_model(linear_model)
        tuned_lm = ESTIMATOR.tune_model(trained_lm, search_algorithm = SEARCH_ALGORITHM, optimize = evaluation_metric,
            search_library = SEARCH_LIBRARY, n_iter = n_iter, custom_grid = custom_grid.get(linear_model), 
            early_stopping = early_stopping_algo, early_stopping_max_iters = early_stop, 
            choose_better = True) 

        plot_params.model = tuned_lm
        image_dir = join_path(save_dir, linear_model)  
        os.makedirs(image_dir, exist_ok = True)
        pipeline_plots(plot_params, tuned_lm, image_dir, USE_MLFLOW)
        coef = np.append(tuned_lm.intercept_, tuned_lm.coef_)
        coefficients.append(coef)
    
    coefficients_df = pd.DataFrame(coefficients, columns = np.append(["intercept"], ESTIMATOR.get_config('X_train').columns),
        index = LINEAR_MODELS)

    trained_tm = ESTIMATOR.create_model(TREE_MODEL)
    tuned_tm = ESTIMATOR.tune_model(trained_tm, search_algorithm = SEARCH_ALGORITHM, optimize = evaluation_metric,
        search_library = SEARCH_LIBRARY, n_iter = n_iter, custom_grid = custom_grid.get(TREE_MODEL), 
        early_stopping = early_stopping_algo, early_stopping_max_iters = early_stop, 
        choose_better = True) 

    plot_params.model = tuned_tm
    image_dir = join_path(save_dir, TREE_MODEL)  
    os.makedirs(image_dir, exist_ok = True)
    plot_params.plots = ["residuals", "error", "feature_all", "rfe"]
    pipeline_plots(plot_params, tuned_tm, image_dir, USE_MLFLOW)

    for plot in ["summary", "correlation", "reason", "pdp", "msa"]:
        ESTIMATOR.interpret_model(tuned_tm, plot = plot, save = image_dir)

    ebm_model = ESTIMATOR.create_model(EBM())
    if CONFIG.get("tune_ebm"):
        ebm_model = ESTIMATOR.tune_model(ebm_model, search_algorithm = SEARCH_ALGORITHM, optimize = evaluation_metric,
            search_library = SEARCH_LIBRARY, n_iter = n_iter, custom_grid = custom_grid.get(EBM_KEY), 
            early_stopping = early_stopping_algo, early_stopping_max_iters = early_stop, 
            choose_better = True) 
    
    image_dir = join_path(save_dir, EBM_KEY)  
    os.makedirs(image_dir, exist_ok = True)
    plot_params.model = ebm_model
    plot_params.plots = ["residuals", "error"]
    pipeline_plots(plot_params, ebm_model, image_dir, USE_MLFLOW)
    for plot in ["msa", "pdp"]:  
        ESTIMATOR.interpret_model(ebm_model, plot = plot, save = image_dir)

    ebm_model.explain_global().visualize().write_html(join_path(image_dir, "global_explanations.html"))
    
    # Save results
    if USE_MLFLOW:
        for model in LINEAR_MODELS + [TREE_MODEL, EBM_KEY]:
            mlflow.log_artifact(join_path(save_dir, model))
        save_mlflow_results(CONFIG, ebm_model, EXPERIMENT_NAME, tmp_dir, assigned_df = coefficients_df)
        end_mlflow(PROJECT_NAME, EXPERIMENT_NAME, tmp_dir, CONFIG.get("author"))
    else:
        save_local_results(CONFIG, ebm_model, EXPERIMENT_NAME, assigned_df = coefficients_df, save_path = save_dir)

    if RUN_DISTRIBUTED:
        ray.shutdown()
        
if __name__ == "__main__":
    main()
     