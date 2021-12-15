import json
import math
import os
import pandas as pd
import pickle
import sys

import mlflow
import mlflow.pyfunc
import tempfile
import sklearn.metrics as metrics

# if len(sys.argv) != 5:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython evaluate.py model features scores pred_actual \n")
#     sys.exit(1)

with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
    model_name = sys.argv[1].strip("'")
    version = 1
    test_file = os.path.join(sys.argv[2].strip("'"), "test.csv")
    scores_file = sys.argv[3].strip("'")
    pred_actual = sys.argv[4].strip("'")

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{version}"
    )

    df = pd.read_csv(test_file)
    X = df.drop("wind", axis = 1)
    y = df["wind"]

    predictions = model.predict(X)
    mse_score = metrics.mean_squared_error(y, predictions)
    mae_score = metrics.mean_absolute_error(y, predictions)

    mlflow.log_metric("mse", mse_score)
    mlflow.log_metric("mae", mae_score)

    for i, (y, predictions) in enumerate(zip(y, predictions)):
        mlflow.log_metric(key="actual", value=y, step=i)
        mlflow.log_metric(key="prediction", value=predictions, step=i)
