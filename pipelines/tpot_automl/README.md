# tpot_automl

## Table of Contents  

* [Introduction](#introduction)
* [Models](#models)
* [Parameters](#parameters)
* [SLURM](#slurm)

### Introduction

An AutoML project using the [TPOT genetic algorithm](http://epistasislab.github.io/tpot/). TPOT will automatically construct a [Scikit Learn](https://scikit-learn.org/) machine learning pipeline. 

This package is integrated with [Dask](https://dask.org/) for distributed and parallelised construction of the pipeline.

### Models

By specifying the estimator task (using the "est_task" parameter), TPOT will attempt to make use of all the associated regression and classification algorithms available within Scikit Learn. Additional pre-processing methods will also be explored within the search space by the genetic algorithm.

Note that TPOT's performance increases as its execution time increases (and respective number of generations). It is intended to be run for at least one day, if not several.

### Parameters

The following parameters are critical to the TPOT AutoML MLFlow project.

* generations: Number of iterations to the run pipeline optimization process.   
* population_size: Number of individuals to retain in the genetic programming population every generation.
* max_time_mins: How many minutes TPOT has to optimize the pipeline. 
* max_eval_time_mins: How many minutes TPOT has to evaluate a single pipeline. 
* config_dict: A configuration dictionary for customizing the operators and parameters that TPOT searches in the optimization process. 
* run_distributed: Run TPOT over Dask for for distributed and parallelised computing.
* early_stop: How many generations TPOT checks whether there is no improvement in optimization process. Ends the optimization process if there is no improvement in the given number of generations. 

### SLURM

The associated SLURM script will run Dask over MPI for use on the NeSI high-performance computing (HPC) environment.
