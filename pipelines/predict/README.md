# predict

## Table of Contents  

* [Introduction](#introduction)
* [Models](#models)
* [Parameters](#parameters)

### Introduction

Perform predictions using a trained model. 

### Models

Models may either be stored within the MLFlow model registry or serialised using [joblib](https://joblib.readthedocs.io/en/latest/index.html#. These saved models may then be loaded from MLFlow or deserialised for predictions. 

At the moment, this MLFlow project is intended to be used with [Scikit Learn](https://scikit-learn.org/) compatible models only, though future support for saved TensorFlow or PyTorch models may be added.

### Parameters

The following parameters are critical to the Predict MLFlow project.

* est_task: The prediction task that the loaded model is intended to perform. Can be "regression", "classification", "anomaly", or "clustering".
* model_name: The name of the model within the MLFlow model registry. 
* model_version: The version of the model within the MLFlow model registry.  
* model_stage: The deployment stage of the model within the MLFlow model registry. 
* model_path: The file path of the model when using local storage.
