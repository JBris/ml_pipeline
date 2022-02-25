# preprocessing

## Table of Contents  

* [Introduction](#introduction)
* [Parameters](#parameters)

### Introduction

Perform simple preprocessing of input data without the use of data versioning.

### Parameters

The following parameters are critical to the Preprocessing MLFlow project.

* drop_target_nas: Boolean to determine if rows should be dropped from the dataset when the target is null.
* drop_nas: Boolean to determine if rows should be dropped from the dataset if any value is null.
* include_features: A list of features to be included. Features outside of this list are dropped.
* drop_features: A list of features to be excluded. Features within this list are dropped.
* col_as_type: A dictionary of feature names and data types to cast them to.
* copy_data_path: An optional file path to copy the processed data to. If a relative path is provided, it will be relative to the preprocessing MLFlow directory.
