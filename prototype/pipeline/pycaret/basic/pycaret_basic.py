##########################################################################################################
### Imports
##########################################################################################################

import argparse
import configparser
import mlflow
import os
import pandas as pd
from pycaret.regression import (setup, compare_models, tune_model, get_logs, create_model,
get_config, ensemble_model, finalize_model, predict_model, plot_model, blend_models, stack_models)
from pycaret.utils import check_metric
import ray
import tempfile
import yaml

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Create PyCaret pipeline.'
)

def _add_argument(parser: argparse.ArgumentParser, name: str, default: str, arg_help: str, type = str) -> None:
    parser.add_argument(
        name,
        default=default,
        help=f"{arg_help}. Defaults to '{default}'.",
        type=type
    )

_add_argument(parser, "--input", None, "input")
_add_argument(parser, "--target", "Y", "target")
_add_argument(parser, "--config", "prepared", "prepared")
_add_argument(parser, "--config_yaml", None, "model")
_add_argument(parser, "--input_key", "features", "features")
_add_argument(parser, "--out_dir", "out", "out")
_add_argument(parser, "--n_select", 3, "n_select", int)
_add_argument(parser, "--n_estimators", 5, "n_estimators", int)
_add_argument(parser, "--model_name", "stacking_ensemble", "n_estimators")

args = parser.parse_args()

##########################################################################################################
### Constants
##########################################################################################################

CONFIG = configparser.ConfigParser()
CONFIG.read(args.config)
PARAMS = yaml.safe_load(open(args.config_yaml))["files"]

##########################################################################################################
### Main
##########################################################################################################

#ray.init()

with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
    
    # Create hold-out set
    df = pd.read_csv(PARAMS[args.input_key])
    data = df.sample(frac=0.9, random_state=PARAMS["random_state"])
    data_unseen = df.drop(data.index)
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)

    # Fit models
    reg = setup(data = data, target = args.target, session_id=123, fold_shuffle=True, imputation_type='iterative', silent = True) 
    top_models = compare_models(n_select = args.n_select, sort = "MSE")
    tuned_top = [ tune_model(model) for model in top_models ]
    dt = create_model('dt')
    tuned_df = tune_model(dt)
    #tuned_top = [ tune_model(model, search_algorithm="optuna", search_library="tune-sklearn") for model in top_models ]
    
    #ensemble_mod = stack_models(tuned_top, optimize = "MSE", meta_model = tuned_df)
    ensemble_mod = blend_models(tuned_top, optimize = "MSE", choose_better = True)
    #ensemble_mod = ensemble_model(tuned_top[0], method = "Boosting", optimize = "MSE", choose_better = True, n_estimators = 10)
    #ensemble_mod = ensemble_model(tuned_top[0], method = "Bagging", optimize = "MSE", choose_better = True, n_estimators = 10)

    # Evaluate
    unseen_predictions = predict_model(ensemble_mod, data = data_unseen)
    MAE = check_metric(unseen_predictions[args.target], unseen_predictions.Label, 'MAE')
    MSE = check_metric(unseen_predictions[args.target], unseen_predictions.Label, 'MSE')
    mlflow.log_metric("mae", MAE)
    mlflow.log_metric("mse", MSE)

    for i, (y, predictions) in enumerate(zip(unseen_predictions[args.target], unseen_predictions.Label)):
        mlflow.log_metric(key="actual", value=y, step=i)
        mlflow.log_metric(key="prediction", value=predictions, step=i)

    final_ensemble = finalize_model(ensemble_mod)
    ensemble_model_residuals = plot_model(final_ensemble, plot = 'residuals', save = tmp_dir)
    mlflow.log_artifact(os.path.join(tmp_dir, ensemble_model_residuals))
    mlflow.sklearn.log_model(final_ensemble, args.model_name, registered_model_name = args.model_name)

client = mlflow.tracking.MlflowClient()
client.set_tag(run.info.run_id, "model", args.model_name)
