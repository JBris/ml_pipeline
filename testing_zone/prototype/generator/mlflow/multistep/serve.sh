#!/usr/bin/env bash

mlflow server --host 127.0.0.1 --port 5000 \
--default-artifact-root ./artifacts  \
--backend-store-uri sqlite:///mlflow.db
    #--gunicorn-opts "--log-level debug" \
    #--backend-store-uri ./mlruns
    