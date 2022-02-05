"""
Machine Learning Pipeline estimators

A library of estimators for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

import abc
import pandas as pd
import pycaret.classification
import pycaret.regression

from pipeline_lib.config import Config

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

def setup(estimator: PyCaretEstimatorBase, config: Config, data: pd.DataFrame, experiment_name: str):
    use_mlflow = config.get("use_mlflow")
    
    return estimator.setup(data = data, target = config.get("target"), fold_shuffle=True, 
        imputation_type = config.get("imputation_type"), iterative_imputation_iters = config.get("iterative_imputation_iters"),
        fold = config.get("k_fold"), fold_groups = config.get("fold_groups"),
        fold_strategy = config.get("fold_strategy"), use_gpu = True, polynomial_features = config.get("polynomial_features"), 
        polynomial_degree =  config.get("polynomial_degree"), remove_multicollinearity = config.get("remove_multicollinearity"), 
        categorical_features = config.get("categorical_features"), ordinal_features = config.get("ordinal_features"),
        numeric_features = config.get("numeric_features"), feature_selection = config.get("feature_selection"),  
        feature_selection_method = config.get("feature_selection_method"),
        feature_selection_threshold = config.get("feature_selection_threshold"), feature_interaction = config.get("feature_interaction"),
        feature_ratio = config.get("feature_ratio"), interaction_threshold = config.get("interaction_threshold"),
        pca = config.get("pca"), pca_method = config.get("pca_method"), pca_components = config.get("pca_components"),
        log_experiment = use_mlflow, experiment_name = experiment_name, ignore_features = config.get("ignore_features"), 
        log_plots = use_mlflow,  log_profile = use_mlflow, log_data = use_mlflow, silent = True, profile = use_mlflow, 
        session_id = config.get("random_seed")) 

def unsupervised_setup(data, config, experiment_name, type: str = "clustering"):
    use_mlflow = config.get("use_mlflow")
    if type == "anomaly":
        from pycaret.anomaly import setup
    else:
        from pycaret.clustering import setup
    
    return setup(data = data, imputation_type = config.get("imputation_type"),
        use_gpu = True, remove_multicollinearity = config.get("remove_multicollinearity"), log_experiment = use_mlflow, 
        pca = config.get("pca"), pca_method = config.get("pca_method"), pca_components = config.get("pca_components"),
        experiment_name = experiment_name, ignore_features = config.get("ignore_features"), log_plots = use_mlflow, 
        log_profile = use_mlflow, log_data = use_mlflow, silent = True, profile = use_mlflow, session_id = config.get("random_seed")) 
