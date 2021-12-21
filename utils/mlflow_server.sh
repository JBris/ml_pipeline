#!/usr/bin/env bash

# Launch a local MLFlow server instance

mlflow server --host 127.0.0.1 --port 5000 \
--default-artifact-root ./artifacts  \
--backend-store-uri sqlite:///mlflow.db