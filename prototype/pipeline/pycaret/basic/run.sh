#!/usr/bin/env bash

INPUT_KEY=forest_fires
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export RAY_ADDRESS=auto

# Use system environment
# --no-conda 

mlflow run --no-conda . -P config=../../config/config.ini -P config_yaml=../../config/config.yaml \
-P input_key=${INPUT_KEY} -P target=Y
