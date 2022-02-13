# clustering 

## Table of Contents  

* [Introduction](#introduction)<a name="introduction"/>
* [Models](#models)<a name="models"/>
* [Parameters](#parameters)<a name="parameters"/>

### Introduction

Perform clustering on a dataset

This MLFlow project makes use of the [Scikit Learn](https://scikit-learn.org/) package, alongside PyCaret, for performing clustering of data.

### Models

Several algorithms are supported internally by PyCaret. They can be utilised by passing the associated model type ID (contained within the brackets).

These include:

* K-means (kmeans)
* Affinity Propagation (ap)
* Mean-Shift (meanshift)
* Spectral Clustering (sc)
* Hierarchical Clustering (hclust)
* DBSCAN (dbscan)
* OPTICS (optics)
* BIRCH (birch)
* K-Modes (kmodes)

Additional custom algorithms from Scikit Learn include:

* Mini-Batch K-Means (mbkmeans)
* Gaussian Mixture Modelling (gmm)
* Bayesian Gaussian Mixture Modelling (bgmm)

### Parameters

The following parameters are critical to the clustering MLFlow project.

* clustering_model: The clustering model algorithm.
* custom_clustering_model: The custom clustering model algorithm (overrides clustering_model).
* num_clusters: The expected number of clusters within the data. Can be tuned using hyperparameter optimisation.
* clustering_plot: The dimensionality reduction algorithm to perform for 3D visualisations (defaults to T-SNE).
