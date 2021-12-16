##########################################################################################################
### Imports
##########################################################################################################

import argparse
import configparser
import mlflow
import os
import pandas as pd
from pycaret.regression import setup, compare_models, tune_model, get_logs, get_config
import ray
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

with mlflow.start_run() as run:
    
    # Create hold-out set
    df = pd.read_csv(PARAMS[args.input_key])
    data = df.sample(frac=0.9, random_state=PARAMS["random_state"])
    data_unseen = df.drop(data.index)
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)

    # Fit models
    reg = setup(data = data, target = args.target, session_id=123, fold_shuffle=True, imputation_type='iterative', silent = True) 
    top_models = compare_models(n_select = args.n_select, sort = "MSE")
    tuned_top = [ tune_model(model, search_algorithm="optuna", search_library="tune-sklearn") for model in top_models ]
    bagged_model = ensemble_model(tuned_top, n_estimators=args.n_estimators)
    final_dt = finalize_model(bagged_model)

    # Evaluate
    unseen_predictions = predict_model(bagged_model, data = data_unseen)
    MAE = check_metric(unseen_predictions[args.target], unseen_predictions.Label, 'MAE')
    MSE = check_metric(unseen_predictions[args.target], unseen_predictions.Label, 'MSE')
    mlflow.log_metric("mae", MAE)
    mlflow.log_metric("mse", MSE)
    plot_model(bagged_model, plot = 'parameter')
    