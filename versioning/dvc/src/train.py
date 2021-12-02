import os
import pandas as pd
import pickle
import sys

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

params = yaml.safe_load(open("params.yaml"))["train"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

train_input = os.path.join(sys.argv[1], "train.csv")
output = sys.argv[2]

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

with open(output, "wb") as fd:
    pickle.dump(clf, fd)
