import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml

from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open("params.yaml"))["featurize"]

np.set_printoptions(suppress=True)

if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
    sys.exit(1)

train_input = os.path.join(sys.argv[1], "train.csv")
test_input = os.path.join(sys.argv[1], "test.csv")
train_output = os.path.join(sys.argv[2], "train.csv")
test_output = os.path.join(sys.argv[2], "test.csv")

mean = bool(params["mean"])
std = bool(params["std"])
scaler = StandardScaler(with_mean = mean, with_std = std)

df_train = pd.read_csv(train_input).drop(["month", "day"], axis=1)
df_test = pd.read_csv(test_input).drop(["month", "day"], axis=1)
scaler.fit(df_train)
df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns)
df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

os.makedirs(sys.argv[2], exist_ok=True)
df_train.to_csv(train_output, index = False)
df_test.to_csv(test_output, index = False)
