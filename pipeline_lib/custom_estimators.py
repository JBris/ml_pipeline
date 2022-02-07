"""
Machine Learning Pipeline Custom Estimators

A library of custom estimators for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External

from sklearn.linear_model import SGDRegressor, GammaRegressor

# Internal

##########################################################################################################
### Library  
##########################################################################################################

CUSTOM_REGRESSORS = {
    "sgd": SGDRegressor,
    "gamma": GammaRegressor
}

CUSTOM_CLASSIFIERS = {}
