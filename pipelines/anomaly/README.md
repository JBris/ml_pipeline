# anomaly

## Table of Contents  

* [Introduction](#introduction)
* [Models](#models)
* [Parameters](#parameters)

### Introduction

Perform anomaly detection of datasets.

This MLFlow project makes use of the [PyOD](https://pyod.readthedocs.io/en/latest/) package, alongside PyCaret, for performing scalable anomaly detection of multivariate data.

### Models

Several algorithms are supported internally by PyCaret. They can be utilised by passing the associated model type ID (contained within the brackets).

These include:

* Angle-Based Outlier Detection (abod)
* Clustering-based Outlier Detection (cluster)
* Connectivity-based Outlier Factor (cof)
* Histogram-Based Outlier Detection (histogram)
* K-Nearest Neighbors (knn)
* Local Outlier Factor (lof)
* Support Vector Machine (svm)
* Principal Component Analysis (pca)
* Minimum Covariance Determinant (mcd)
* Subspace Outlier Detection (sod)
* Stochastic Outlier Selection (sos)
* Isolation Forest (iforest)

Additional custom algorithms from PyOD include:

* Copula Based Outlier Detector (copod)
* Linear Method for Deviation-based Outlier Detection (lmdd)
* Local Correlation Integral (loci)
* Lightweight Online Detector of Anomalies (loda)

### Parameters

The following parameters are critical to the anomaly detection MLFlow project.

* anomaly_model: The anomaly detection model algorithm.
* custom_anomaly_model: The custom anomaly detection model algorithm (overrides anomaly_model).
* contamination_fraction: The estimated proportion of anomalous observations within the data.
* anomaly_plot: The dimensionality reduction algorithm to perform for 2D visualisations (defaults to UMAP).
