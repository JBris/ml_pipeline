#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Use system environment
# --no-conda 

mlflow run --no-conda . # -P evaluate_plot_data=plot.json -P evaluate_scores=evaluate.json
