# interpret_ml

## Table of Contents  

* [Introduction](#introduction)
* [Models](#models)
* [Parameters](#parameters)
* [SLURM](#slurm)

### Introduction

Pipeline for interpretable machine learning. This MLFlow project makes use of [Scikit Learn](https://scikit-learn.org/), [InterpretML](https://interpret.ml/), and PyCaret.

### Models

This project makes use of the following linear methods:

* Regression
  * Ordinary Least Squares Linear Regression
  * LASSO
  * Ridge Regression
* Classification
  * Logistic Regression
  * Ridge Classifier

The following non-linear methods are used:

* Regression
  * Decision Tree Regressor
* Classification
  * Decision Tree Classifier

Finally, the Explainable Boosting Machine (EBM) method is used. An EBM is a Generalised Additive Model (GAM) that fits boosted shallow decision trees to each feature, before performing automated feature interaction detection and feature importance.

[Click here for more information.](https://interpret.ml/docs/ebm.html)

### Parameters

The following parameters are critical to the interpretable machine learning MLFlow project.

* est_task: The prediction task that the loaded model is intended to perform. Can be "regression" or "classification".
* run_distributed: Whether to use Ray for distributed and parallelised computing.
* tune_ebm: Whether to perform hyperparameter optimisation of the EBM model. This is disabled by default.

### SLURM

The associated SLURM script will run Ray over SLURM for use on the NeSI high-performance computing (HPC) environment. It is particularly useful for training and tuning the EBM algorithm, which has a lengthy training time.
