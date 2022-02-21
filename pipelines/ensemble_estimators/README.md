# ensemble_estimators

## Table of Contents  

* [Introduction](#introduction)
* [Models](#models)
* [Parameters](#parameters)
* [SLURM](#slurm)

### Introduction

Perform sizing estimation using ensemble models.

This MLFlow project makes use of [Scikit Learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [CatBoost](https://catboost.ai/), [XGBoost](https://xgboost.readthedocs.io/en/stable/), and PyCaret for regression and classsification tasks.

### Models

Several algorithms are supported internally by PyCaret. They can be utilised by passing the associated model type ID (contained within the brackets).

These include the following regressors:

* Linear Regression (lr)
* Lasso Regression (lasso)
* Ridge Regression (ridge)
* Elastic Net (en)
* Least Angle Regression (lar)
* Lasso Least Angle Regression (llar)
* Orthogonal Matching Pursuit (omp)
* Bayesian Ridge (br)
* Automatic Relevance Determination (ard)
* Passive Aggressive Regressor (par)
* Random Sample Consensus (ransac)
* TheilSen Regressor (tr)
* Huber Regressor (huber)
* Kernel Ridge (kr)
* Support Vector Regression (svm)
* K Neighbors Regressor (knn)
* Decision Tree Regressor (dt)
* Random Forest Regressor (rf)
* Extra Trees Regressor (et)
* AdaBoost Regressor (ada)
* Gradient Boosting Regressor (gbr)
* MLP Regressor (mlp)
* Extreme Gradient Boosting (xgboost)
* Light Gradient Boosting Machine (lightgbm)
* CatBoost Regressor (catboost)

Additional custom regressors include:

* Explainable Boosting Regressor (ebm)
* Gamma Regressor (gamma)
* Gaussian Process Regressor (gp)
* Stochastic Gradient Descent (sgd)
* Tweedie Regressor (tweedie)

Moreover, the following classifiers are included:

* Logistic Regression (lr)
* K Neighbors Classifier (knn)
* Naive Bayes (nb)
* Decision Tree Classifier (dt)
* SVM - Linear Kernel (svm)
* SVM - Radial Kernel (rbfsvm)
* Gaussian Process Classifier (gpc)
* MLP Classifier (mlp)
* Ridge Classifier (ridge)
* Random Forest Classifier (rf)
* Quadratic Discriminant Analysis (qda)
* Ada Boost Classifier (ada)
* Gradient Boosting Classifier (gbc)
* Linear Discriminant Analysis (lda)
* Extra Trees Classifier (et)
* Extreme Gradient Boosting (xgboost)
* Light Gradient Boosting Machine (lightgbm)
* CatBoost Classifier (catboost)

Additional custom classifiers include:

* Explainable Boosting Classifier (ebm)

### Parameters

The following parameters are critical to the ensemble estimator MLFlow project.

* est_task: The prediction task that the loaded model is intended to perform. Can be "regression" or "classification".
* run_distributed: Whether to use Ray for distributed and parallelised computing.
* ensemble_methods: The list of ensembling methods to use. Defaults to "stacking", "blending", "boosting", "bagging", and "blended_boosting".
* meta_model: The meta-estimator for the stacking ensemble.
* n_select: Choose the top n models for ensembling.
* n_estimators: The number of estimators for boosting and bagging.
* n_iter: The number of iterations for hyperparameter tuning.
* turbo: Whether to train only fast-fitting models.
* include_estimators: A list of estimator types to include. Defaults to an empty list (meaning all supported estimator types are included).
* custom_regressors: A list of custom regressors to include in combination with PyCaret's regressors.
* custom_classifiers: A list of custom classifiers to include in combination with PyCaret's classifiers.
* custom_regressor_grid: The default hyperparameter grid for custom regressors.
* custom_classifier_grid: The default hyperparameter grid for custom classifiers.

### SLURM

The associated SLURM script will run Ray over SLURM for use on the NeSI high-performance computing (HPC) environment. It is particularly useful for training and tuning complex ensemble models for many iterations, which generally leads to a lengthy training time.
