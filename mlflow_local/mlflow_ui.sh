#!/usr/bin/env bash

. ../.env

# Launch a local MLFlow server instance

echo "Host: ${MLFLOW_LOCAL_HOST}" 
echo "Port: ${MLFLOW_LOCAL_PORT}"

mlflow ui --host "${MLFLOW_LOCAL_HOST}" --port "${MLFLOW_LOCAL_PORT}" --serve-artifacts --default-artifact-root file:${ML_PIPELINE_DIR}/mlflow_local/mlruns --backend-store-uri sqlite:///mlflow.db 
