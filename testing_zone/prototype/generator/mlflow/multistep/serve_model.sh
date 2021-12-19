#!/usr/bin/env bash

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Serve the production model from the model registry
mlflow models serve --no-conda -m "models:/forest_fires_rf/1" 