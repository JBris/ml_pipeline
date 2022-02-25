# kiwi_datasets

## Table of Contents  

* [Introduction](#introduction)
* [DVC](#dvc)<a name="dvc"/>

### Introduction

Example Hayward and Gold kiwifruit datasets.

### DVC

Data Version Control (DVC) is being used to version a simple pipeline for adding interaction terms to the Hayward and Gold kiwifruit datasets from KPINS 4217 and 7154.

The pipeline is defined in [dvc.yaml](dvc.yaml), and can be reproduced by running [dvc.sh](dvc.sh).

There are six CSV files that are declared as dependencies within the pipeline. After processing, an additional six CSV files are creating where interaction terms are incorporated.

This provides a simple example of DVC for versioning a dataset and declaring a pipeline.
