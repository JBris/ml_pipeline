import os
import pandas as pd
import pickle
import sys
import tempfile
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

params = yaml.safe_load(open("params.yaml"))["train"]

# if len(sys.argv) != 3:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython train.py features model\n")
#     sys.exit(1)

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
    train_input = os.path.join(sys.argv[1].strip("'"), "train.csv")
    model_name = sys.argv[2].strip("'")

    df = pd.read_csv(train_input)
    X = df.drop("wind", axis = 1)
    y = df["wind"]

    seed = params["seed"]
    n_est = params["n_est"]
    min_split = params["min_split"]

    clf = RandomForestRegressor(
        n_estimators=n_est, min_samples_split=min_split, n_jobs=-1, random_state=seed
    )

    clf.fit(X, y)

    signature = infer_signature(X, clf.predict(X))
    mlflow.sklearn.log_model(clf, model_name, signature=signature, registered_model_name = model_name)

client = mlflow.tracking.MlflowClient()
client.set_tag(run.info.run_id, "model", model_name)
