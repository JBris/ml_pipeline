#!/usr/bin/env bash

. ../.env

. ../env_hook.sh

# Launch a local MLFlow server instance

echo "Host: ${MLFLOW_LOCAL_HOST}" 
echo "Port: ${MLFLOW_LOCAL_PORT}"

mlflow server --serve-artifacts --host "${MLFLOW_LOCAL_HOST}" --port "${MLFLOW_LOCAL_PORT}" --default-artifact-root file:${PROJECT_DIR}/mlflow_local/mlruns --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" ${MLFLOW_EXPOSE_PROMETHEUS}
