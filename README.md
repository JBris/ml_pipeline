# Machine Learning Pipeline

## Table of Contents  

* [Introduction](#introduction)<a name="introduction"/>
* [Dependencies](#dependencies)<a name="dependencies"/>
* [DVC](#dvc)<a name="dvc"/>
* [MLFlow](#mlflow)<a name="mlflow"/>
* [Ray](#ray)<a name="ray"/>
* [PyCaret](#pycaret)<a name="pycaret"/>
* [Optuna](#optuna)<a name="optuna"/>
* [TPOT](#tpot)<a name="tpot"/>
* [InterpretML](#interpretml)<a name="interpretml"/>
* [Environment](#environment)<a name="environment"/>
* [Pipeline Library](#pipeline-library)<a name="pipeline-library"/>
* [Configuration](#configuration)<a name="configuration"/>
* [Pipelines](#pipelines)<a name="pipelines"/>
* [Scenarios](#scenarios)<a name="scenarios"/>
* [Templates](#templates)<a name="templates"/>

### Introduction

This repository contains a machine learning pipeline framework for running fruit size prediction experiments.

Execute `init.sh` to initialise the pipeline tool. Several additional options are supported. Run `init.sh -h` for help.

Edit `.env` and `params.local.ini` to specify your local environment configuration as required.

[Visit the documentation page for more information.](https://planttech.atlassian.net/wiki/spaces/EK3/pages/9994731521/User+manual).

### Dependencies

The following core Python packages are required:

* DVC
* MLFlow
* Ray
* PyCaret
* Optuna
* TPOT
* InterpretML

[See requirements.txt for the associated packages and versions.](requirements.txt)

### DVC

Data Version Control (DVC) is primarily being used as a data versioning tool for synthetic and real datasets, as well as other produced files (e.g. serialised models).

[More information can be found at https://dvc.org/](https://dvc.org/).

### MLFlow

MLFlow is a machine learning platform for experiment tracking, model versioning, and the logging of metrics.

[More information can be found at https://mlflow.org/](https://mlflow.org/).

A local instance of MLFlow can be launched using [mlflow_ui.sh](mlflow_local/mlflow_ui.sh). MLFlow project pipelines can be executed by running the [mlflow_run.sh](mlflow_local/mlflow_run.sh) script. Enter `mlflow_run.sh -h` for more information.

### Ray

Ray has been included to facilitate the parallelised and distributed execution of the pipeline. More specifically, Ray Tune enables hyperparameter tuning to be performed on within a node cluster. We are using it to rapidly speed up training time.

[More information can be found at https://www.ray.io/](https://www.ray.io/).

### PyCaret

PyCaret is a "low code" library that enables the construction of highly complex pipelines with minimal amounts of code required. Several estimator tasks are supported, including regression, classification, anomaly detection, and clustering. Experimental support for time series data is also included.

[More information can be found at https://pycaret.org/](https://pycaret.org/).

### Optuna

Optuna is an optimisation library that is primarily being used for hyperparameter tuning. It offers Bayesian optimisation using Tree-structured Parzen Estimator (TPES), as well as genetic algorithms such as Covariance matrix adaptation evolution strategy (CMA-ES) and Nondominated Sorting Genetic Algorithm II (NSGAII).

[More information can be found at https://optuna.org/](https://optuna.org/).

### TPOT

TPOT AutoML is a tool that uses genetic programming to automatically construct machine learning pipelines. These pipelines are built using the [Scikit Learn](https://scikit-learn.org/) machine learning library. 

[More information can be found at http://epistasislab.github.io/tpot/](http://epistasislab.github.io/tpot/).

### InterpretML

InterpretML is a machine learning library for interpretable machine learning. If offers metrics such as SHapley Additive exPlanations (SHAP) and Morris sensitivity scores, alongside the Explainable Boosting Machine (EBM) algorithm.

[More information can be found at https://interpret.ml/](https://interpret.ml/).

### Environment

When [init.sh](init.sh) is first executed, it will create both a *.env* file from [.env.example](.env.example), and an empty params.local.yaml file. This section will elaborate upon the *.env* file.

There are several important environment variables:

* ML_PIPELINE_DIR: The base directory for the ml-pipeline directory
* SIZING_DIR: The base directory for the fruit sizing project
* PIPELINE_SCENARIO: A scenario file from the scenario directory
* CONFIG_FILE: A params.override.yaml configuration file

### Pipeline Library

A library of reusable code is contained within [pipeline_lib](pipeline_lib). Code from this library can be imported into the pipeline scripts defined within the pipeline directory.

### Configuration

The pipeline configuration class is defined within [pipeline_lib/config.py](pipeline_lib/config.py). This acts as a unifying API for getting, setting, importing, and exporting pipeline parameter values defined within several formats:

* Environment variables - Environment variables available from a white list (currently supports Ray and MLFlow environment variables).
* YAML parameter files - Parameters can be specified in YAML files, which are then parsed and merged to the configuration dictionary.
* Script arguments - Arguments defined by the ArgumentParser of the argparse module can be merged into the configuration dictionary.

These various sources of parameter values are merged together into a single data dictionary. It is important to note the order that these sources are merged.

1. Environment variables.
2. [params.global.yaml](params.global.yaml)
3. The *params.yaml* file for each pipeline.
4. The arguments passed to the argument parser.
5. The scenario file.
6. [params.local.yaml](params.local.yaml)
7. A params.override.yaml file.

This means that the values in a params.override.yaml file will override those values in [params.global.yaml](params.global.yaml). 

We can think of the values in [params.global.yaml](params.global.yaml) and params.yaml as reasonable defaults. If more customised configuration is required, then it can be specified in a [scenarios](scenarios) file or [params.local.yaml](params.local.yaml). 

Note that [params.local.yaml](params.local.yaml) is intended to be used for pipeline exploration and tooling within a local user environment, and should not be saved using git (it's in [.gitignore](.gitignore) by default).

### Pipelines

Pipelines are defined within the [pipelines](pipelines) directory. These can optionally be initialised as DVC or MLFlow pipelines by running the [pipelines/new.sh](pipelines/new.sh) script. This will create the new pipeline directory, and add some scaffolding files, ensuring that your pipelines follow a consistent format. 

Type `new.sh -h` for more information.

### Scenarios

Scenario files can be defined in the [scenarios](scenarios) directory. These contain pipeline yaml files that you may want to use in the future.

Edit the *PIPELINE_SCENARIO* environment variable in .env to specify a particular scenario. You can also pass a scenario to a *run.py* script using the `--scenario` argument. When executing `mlflow run`, include a parameter override using `-P scenario=path/to/scenario`. 

### Templates

The [templates](templates) directory is used to store scaffolding files when initialising new projects. You can use these files to create a consistent scructure for each pipeline. This also minimises the amount of coding that is required when developing a pipeline - more time can be dedicated to developing the pipeline itself, and less time is required for constructing the MLproject file or configuration.
