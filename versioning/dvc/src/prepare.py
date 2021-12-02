import os
import pandas as pd
import random
import sys
import yaml

from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

# Test data set split ratio
split = params["split"]
random.seed(params["seed"])

input = sys.argv[1]

df = pd.read_csv(input)
train_output = os.path.join(sys.argv[2], "train.csv")
test_output = os.path.join(sys.argv[2], "test.csv")
print(train_output)

X = df.drop("wind", axis = 1)
y = df["wind"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = split, random_state = params["seed"]
)

X_train["wind"] = y_train
X_test["wind"] = y_test

os.makedirs(sys.argv[2], exist_ok=True)
X_train.to_csv(train_output, index = False)
X_test.to_csv(test_output, index = False)
