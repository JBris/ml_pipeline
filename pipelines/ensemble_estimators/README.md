# ensemble_estimators

## Table of Contents  

* [Introduction](#introduction)<a name="introduction"/>
* [Models](#models)<a name="models"/>
* [Parameters](#parameters)<a name="parameters"/>
* [SLURM](#slurm)<a name="slurm"/>

### Introduction

Perform sizing estimation using ensemble models.

This MLFlow project makes use of [Scikit Learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [CatBoost](https://catboost.ai/), [XGBoost](https://xgboost.readthedocs.io/en/stable/), and PyCaret for regression and classsification tasks.

### Models

Several algorithms are supported internally by PyCaret. They can be utilised by passing the associated model type ID (contained within the brackets).

These include the following regressors:


Additional custom regressors include:


Additionally, the following classifiers are also included:

Additional custom classifiers include:


### Parameters

The following parameters are critical to the ensemble estimator MLFlow project.

*

### SLURM

The associated SLURM script will run Ray over SLURM for use on the NeSI high-performance computing (HPC) environment. It is particularly useful for training and tuning complex ensemble models for many iterations, which generally leads to a lengthy training time.
