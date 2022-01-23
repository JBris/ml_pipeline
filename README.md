# Machine Learning Pipeline

## Table of Contents  

* [Introduction](#introduction)<a name="introduction"/>
* [Dependencies](#dependencies)<a name="dependencies"/>
* [DVC](#dvc)<a name="dvc"/>
* [MLFlow](#mlflow)<a name="mlflow"/>
* [Ray](#ray)<a name="ray"/>
* [PyCaret](#pycaret)<a name="pycaret"/>
* [Optuna](#optuna)<a name="optuna"/>

### Introduction

This repository contains a machine learning pipeline framework for running fruit size prediction experiments.

Execute `init.sh` to initialise the pipeline tool. Several additional options are supported. Run `init.sh -h` for help.

Edit `.env` and `config.local.ini` to specify your local environment configuration as required.

[Visit the documentation page for more information.](https://planttech.atlassian.net/wiki/spaces/EK3/pages/9994731521/User+manual).

### Dependencies

The following core Python packages are required:

* DVC
* MLFlow
* Ray
* PyCaret
* Optuna

[See requirements.txt for the associated packages and versions.](requirements.txt)

### DVC

Data Version Control (DVC) is primarily being used as a data versioning tool for synthetic and real datasets, as well as other produced files (e.g. serialised models).

[More information can be found at https://dvc.org/](https://dvc.org/).

### MLFlow

MLFlow is a machine learning platform for experiment tracking, model versioning, and the logging of metrics.

[More information can be found at https://mlflow.org/](https://mlflow.org/).

### Ray

Ray has been included to facilitate the parallelised and distributed execution of the pipeline. More specifically, Tune enables hyperparameter tuning to be performed on a node cluster.

[More information can be found at https://www.ray.io/](https://www.ray.io/).

### PyCaret

PyCaret is a "low code" library that enables the construction of highly complex pipelines with minimal amounts of code required.

[More information can be found at https://pycaret.org/](https://pycaret.org/).

### Optuna

Optuna is an optimisation library that is primarily being used for hyperparameter tuning.

[More information can be found at https://optuna.org/](https://optuna.org/).
