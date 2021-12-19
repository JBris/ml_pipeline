##########################################################################################################
### Imports
##########################################################################################################

import argparse
import configparser
from dvc.api import make_checkpoint
import mlflow
import os
import pandas as pd
from tpot import TPOTRegressor
import yaml

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Use TPOT automl.'
)

def _add_argument(parser: argparse.ArgumentParser, name: str, default: str, arg_help: str, type = str) -> None:
    parser.add_argument(
        name,
        default=default,
        help=f"{arg_help}. Defaults to '{default}'.",
        type=type
    )

_add_argument(parser, "--input", None, "input")
_add_argument(parser, "--config", "prepared", "prepared")
_add_argument(parser, "--config_yaml", None, "model")
_add_argument(parser, "--input_key", "features", "features")
_add_argument(parser, "--out_dir", "out", "out")

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

with mlflow.start_run() as run:
    df = pd.read_csv(PARAMS[args.input_key])
    X = df[["X", "FFMC"]]
    y = df.wind.values

    pipeline_optimizer = TPOTRegressor(generations=PARAMS["generations"], population_size=PARAMS["population_size"], 
        cv=PARAMS["cv"], random_state=PARAMS["random_state"], n_jobs = PARAMS["n_jobs"], 
        max_time_mins = PARAMS["max_time_mins"], max_eval_time_mins = PARAMS["max_eval_time_mins"], 
        use_dask = False, verbosity = 2, warm_start = False, config_dict = "TPOT light")

    pipeline_optimizer.fit(X, y)
    pipeline_optimizer.export(os.path.join(PARAMS['out_dir'], "pipeline_optimizer.py"))
    mlflow.log_artifact(os.path.join(PARAMS['out_dir'], "pipeline_optimizer.py"))
