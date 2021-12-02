#!/usr/bin/env bash

mlflow run . -P learning_rate=0.2 -P colsample_bytree=0.8 -P subsample=0.9
mlflow ui