"""
Machine Learning Pipeline estimators

A library of estimators for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

import abc
import pycaret.classification
import pycaret.regression

##########################################################################################################
### Library  
##########################################################################################################

class PyCaretEstimatorBase(metaclass = abc.ABCMeta):
    """Abstract base class for estimators."""
    @abc.abstractmethod
    def setup(self, **kwargs):
        return

    @abc.abstractmethod
    def compare_models(self, **kwargs):
        return

    @abc.abstractmethod
    def tune_model(self, estimator, **kwargs):
        return  

    @abc.abstractmethod
    def get_logs(self, **kwargs):
        return

    @abc.abstractmethod
    def create_model(self, estimator, **kwargs):
        return  

    @abc.abstractmethod
    def get_config(self, **kwargs):
        return

    @abc.abstractmethod
    def ensemble_model(self, estimator, **kwargs):
        return  

    @abc.abstractmethod
    def finalize_model(self, estimator, **kwargs):
        return 

    @abc.abstractmethod
    def predict_model(self, estimator, **kwargs):
        return  

    @abc.abstractmethod
    def plot_model(self, estimator, **kwargs):
        return  

    @abc.abstractmethod
    def blend_models(self, estimators: list, **kwargs):
        return

    @abc.abstractmethod
    def stack_models(self, estimators: list, **kwargs):
        return

    @abc.abstractmethod
    def automl(self, **kwargs):
        return

class PyCaretRegressor(PyCaretEstimatorBase):
    """Estimator for regression."""
    def setup(self, **kwargs):
        return pycaret.regression.setup(**kwargs)

    def compare_models(self, **kwargs):
        return pycaret.regression.compare_models(**kwargs)

    def tune_model(self, estimator, **kwargs):
        return pycaret.regression.tune_model(estimator, **kwargs)

    def get_logs(self, **kwargs):
        return pycaret.regression.get_logs(**kwargs)

    def create_model(self, estimator, **kwargs):
        return pycaret.regression.create_model(estimator, **kwargs)

    def get_config(self, **kwargs):
        return pycaret.regression.get_config(**kwargs)

    def ensemble_model(self, estimator, **kwargs):
        return pycaret.regression.ensemble_model(estimator, **kwargs)

    def finalize_model(self, estimator, **kwargs):
        return pycaret.regression.finalize_model(estimator, **kwargs)

    def predict_model(self, estimator, **kwargs):
        return pycaret.regression.predict_model(estimator, **kwargs)

    def plot_model(self, estimator, **kwargs):
        return pycaret.regression.plot_model(estimator, **kwargs)

    def blend_models(self, estimators: list, **kwargs):
        return pycaret.regression.blend_models(estimators, **kwargs)

    def stack_models(self, estimators: list, **kwargs):
        return pycaret.regression.stack_models(estimators, **kwargs)

    def automl(self, **kwargs):
        return pycaret.regression.automl(**kwargs)

class PyCaretClassifier(PyCaretEstimatorBase):
    """Estimator for classification."""
    def setup(self, **kwargs):
        return pycaret.classification.setup(**kwargs)

    def compare_models(self, **kwargs):
        return pycaret.classification.compare_models(**kwargs)

    def tune_model(self, estimator, **kwargs):
        return pycaret.classification.tune_model(estimator, **kwargs)

    def get_logs(self, **kwargs):
        return pycaret.classification.get_logs(**kwargs)

    def create_model(self, estimator, **kwargs):
        return pycaret.classification.create_model(estimator, **kwargs)

    def get_config(self, **kwargs):
        return pycaret.classification.get_config(**kwargs)

    def ensemble_model(self, estimator, **kwargs):
        return pycaret.classification.ensemble_model(estimator, **kwargs)

    def finalize_model(self, estimator, **kwargs):
        return pycaret.classification.finalize_model(estimator, **kwargs)

    def predict_model(self, estimator, **kwargs):
        return pycaret.classification.predict_model(estimator, **kwargs)

    def plot_model(self, estimator, **kwargs):
        return pycaret.classification.plot_model(estimator, **kwargs)

    def blend_models(self, estimators: list, **kwargs):
        return pycaret.classification.blend_models(estimators, **kwargs)

    def stack_models(self, estimators: list, **kwargs):
        return pycaret.classification.stack_models(estimators, **kwargs)

    def automl(self, **kwargs):
        return pycaret.classification.automl(**kwargs)
        