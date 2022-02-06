"""
Machine Learning Pipeline estimators

A library of estimators for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import abc
import pandas as pd
import pycaret.classification
import pycaret.regression

from enum import Enum, unique
from joblib import dump

# Internal
from pipeline_lib.config import Config
from pipeline_lib.data import join_path

##########################################################################################################
### Library  
##########################################################################################################

@unique
class EstimatorTask(Enum):
    """Enum for estimator tasks."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly"

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
    def interpret_model(self, estimator, **kwargs):
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

    @abc.abstractmethod
    def save_model(self, estimator, **kwargs):
        return

    @abc.abstractmethod
    def load_model(self, model_name: str):
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

    def interpret_model(self, estimator, **kwargs):
        return pycaret.regression.interpret_model(estimator, **kwargs)

    def blend_models(self, estimators: list, **kwargs):
        return pycaret.regression.blend_models(estimators, **kwargs)

    def stack_models(self, estimators: list, **kwargs):
        return pycaret.regression.stack_models(estimators, **kwargs)

    def automl(self, **kwargs):
        return pycaret.regression.automl(**kwargs)

    def save_model(self, estimator, **kwargs):
        return pycaret.regression.save_model(estimator, **kwargs)

    def load_model(self, model_name: str):
        return pycaret.regression.load_model(model_name)

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

    def interpret_model(self, estimator, **kwargs):
        return pycaret.classification.interpret_model(estimator, **kwargs)

    def blend_models(self, estimators: list, **kwargs):
        return pycaret.classification.blend_models(estimators, **kwargs)

    def stack_models(self, estimators: list, **kwargs):
        return pycaret.classification.stack_models(estimators, **kwargs)

    def automl(self, **kwargs):
        return pycaret.classification.automl(**kwargs)
        
    def save_model(self, estimator, **kwargs):
        return pycaret.classification.save_model(estimator, **kwargs)

    def load_model(self, model_name: str):
        return pycaret.classification.load_model(model_name)

def _get_setup_kwargs(config: Config, data: pd.DataFrame, experiment_name: str) -> dict:
    use_mlflow = config.get("use_mlflow")
    kwargs = { 
        "data": data,
        "experiment_name": experiment_name, 
        "log_experiment": use_mlflow,
        "log_plots": use_mlflow,
        "log_profile": use_mlflow,
        "log_data": use_mlflow,
        "profile": use_mlflow,
        "silent": True,
        "session_id": config.get("random_seed")
    }
    return kwargs

def setup(estimator: PyCaretEstimatorBase, config: Config, data: pd.DataFrame, experiment_name: str):
    kwargs = _get_setup_kwargs(config, data, experiment_name)
    kwargs["fold_shuffle"] = True

    for config_arg in ["target", "imputation_type", "iterative_imputation_iters", "fold", "fold_groups",
        "fold_strategy", "use_gpu", "polynomial_features", "polynomial_degree", "remove_multicollinearity", 
        "categorical_features", "ordinal_features", "numeric_features", "feature_selection", "feature_selection_method",
        "feature_selection_threshold", "feature_interaction", "feature_ratio", "interaction_threshold", "pca",
        "pca_method", "pca_components", "ignore_features"]:
        kwargs[config_arg] = config.get(config_arg)

    return estimator.setup(**kwargs) 

def unsupervised_setup(config: Config, data: pd.DataFrame, experiment_name: str, type: str = EstimatorTask.CLUSTERING.value):
    if type == EstimatorTask.ANOMALY_DETECTION.value:
        from pycaret.anomaly import setup
    else:
        from pycaret.clustering import setup
    
    kwargs = _get_setup_kwargs(config, data, experiment_name)
    for config_arg in ["imputation_type", "use_gpu", "remove_multicollinearity", "pca", "pca_method",
        "pca_components", "ignore_features"]:
        kwargs[config_arg] = config.get(config_arg)

    return setup(**kwargs) 

def save_local_model(model, experiment_name: str, path = "data") -> str:
    """Save the model to a local directory."""
    model_path = join_path(path, f"{experiment_name}.joblib")
    dump(model, model_path) 
    return model_path
