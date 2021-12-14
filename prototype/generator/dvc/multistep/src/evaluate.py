import json
import math
import os
import pandas as pd
import pickle
import sys

import sklearn.metrics as metrics

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model features scores pred_actual \n")
    sys.exit(1)

model_file = sys.argv[1]
test_file = os.path.join(sys.argv[2], "test.csv")
scores_file = sys.argv[3]
pred_actual = sys.argv[4]

with open(model_file, "rb") as fd:
    model = pickle.load(fd)

df = pd.read_csv(test_file)
X = df.drop("wind", axis = 1)
y = df["wind"]

predictions = model.predict(X)
mse_score = metrics.mean_squared_error(y, predictions)
mae_score = metrics.mean_absolute_error(y, predictions)

with open(scores_file, "w") as fd:
    json.dump({"mse": mse_score, "mae": mae_score}, fd, indent=4)

with open(pred_actual, "w") as fd:
    json.dump(
        {
            "mae": [
                {"actual": y, "pred": predictions}
                for y, predictions in zip(y, predictions)
            ]
        },
        fd,
        indent=4,
    )
