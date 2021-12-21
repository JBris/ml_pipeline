#!/usr/bin/env bash

. ../.env

# Launch a local MLFlow server instance

mlflow server --host "${MLFLOW_LOCAL_HOST}" --port "${MLFLOW_LOCAL_PORT}" \
--default-artifact-root ./artifacts  \
--backend-store-uri sqlite:///mlflow.db # Uses an SQLite database 