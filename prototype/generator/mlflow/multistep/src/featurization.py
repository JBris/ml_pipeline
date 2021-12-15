import argparse
import mlflow
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml

from sklearn.preprocessing import StandardScaler

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

_add_argument(parser, "--featurize_input", None, "Input file", str)
_add_argument(parser, "--featurize_output", None, "Output file", str)

args = parser.parse_args()

##########################################################################################################
### Main
##########################################################################################################

params = yaml.safe_load(open("params.yaml"))["featurize"]

np.set_printoptions(suppress=True)

with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
    train_input = os.path.join(args.featurize_input.strip("'"), "train.csv")
    test_input = os.path.join(args.featurize_input.strip("'"), "test.csv")
    train_output = os.path.join(tmp_dir, args.featurize_output, "train.csv")
    test_output = os.path.join(tmp_dir, args.featurize_output, "test.csv")

    mean = bool(params["mean"])
    std = bool(params["std"])
    scaler = StandardScaler(with_mean = mean, with_std = std)

    df_train = pd.read_csv(train_input).drop(["month", "day"], axis=1)
    df_test = pd.read_csv(test_input).drop(["month", "day"], axis=1)

    scaler.fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

    os.makedirs(os.path.join(tmp_dir, args.featurize_output), exist_ok=True)
    df_train.to_csv(train_output, index = False)
    df_test.to_csv(test_output, index = False)

    mlflow.log_artifact(os.path.join(tmp_dir, args.featurize_output))

