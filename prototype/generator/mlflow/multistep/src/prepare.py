##########################################################################################################
### Imports
##########################################################################################################

import argparse
import mlflow
import os
import pandas as pd
from pprint import pprint
import random
import sys
import tempfile
import yaml

from sklearn.model_selection import train_test_split

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Generate a synthetic dataset.'
)

def _add_argument(parser: argparse.ArgumentParser, name: str, default: str, arg_help: str, type = int) -> None:
    parser.add_argument(
        name,
        default=default,
        help=f"{arg_help}. Defaults to '{default}'.",
        type=type
    )

_add_argument(parser, "--prepare_file", None, "Input file", str)
_add_argument(parser, "--prepare_output", None, "Output file", str)

args = parser.parse_args()

##########################################################################################################
### Main
##########################################################################################################

params = yaml.safe_load(open("params.yaml"))["prepare"]

 

with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
    #Test data set split ratio
    split = params["split"]
    random.seed(params["seed"])

    input = args.prepare_file

    df = pd.read_csv(input)
    train_output = os.path.join(tmp_dir, args.prepare_output, "train.csv")
    test_output = os.path.join(tmp_dir, args.prepare_output, "test.csv")
    print(train_output)

    X = df.drop("wind", axis = 1)
    y = df["wind"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = split, random_state = params["seed"]
    )

    X_train["wind"] = y_train
    X_test["wind"] = y_test

    os.makedirs(os.path.join(tmp_dir, args.prepare_output), exist_ok=True)

    X_train.to_csv(train_output, index = False)
    X_test.to_csv(test_output, index = False)

    mlflow.log_artifact(os.path.join(tmp_dir, args.prepare_output))

# Download artifacts
client = mlflow.tracking.MlflowClient()
pprint(os.listdir(client.download_artifacts(run.info.run_id, args.prepare_output)))

# List artifacts
pprint(client.list_artifacts(run.info.run_id, args.prepare_output))
