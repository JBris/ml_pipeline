"""
Machine Learning Pipeline Estimators

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
import sklearn

from typing import List, Tuple

from enum import Enum, unique
from joblib import dump

# Internal
from pipeline_lib.custom_estimators import CUSTOM_ANOMALY_DETECTION, CUSTOM_CLASSIFIERS, CUSTOM_CLUSTERING, CUSTOM_REGRESSORS 
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
    def pull(self, **kwargs):
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

    @abc.abstractmethod
    def get_config(self, variable: str):
        return

class PyCaretRegressor(PyCaretEstimatorBase):
    """Estimator for regression."""
    def setup(self, **kwargs):
        return pycaret.regression.setup(**kwargs)

    def compare_models(self, **kwargs):
        return pycaret.regression.compare_models(**kwargs)

    def pull(self, **kwargs):
        return pycaret.regression.pull(**kwargs)

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

    def get_config(self, variable: str):
        return pycaret.regression.get_config(variable)

class PyCaretClassifier(PyCaretEstimatorBase):
    """Estimator for classification."""
    def setup(self, **kwargs):
        return pycaret.classification.setup(**kwargs)

    def compare_models(self, **kwargs):
        return pycaret.classification.compare_models(**kwargs)

    def pull(self, **kwargs):
        return pycaret.classification.pull(**kwargs)

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

    def get_config(self, variable: str):
        return pycaret.classification.get_config(variable)

def add_custom_estimators(estimator: PyCaretEstimatorBase, config: Config, search_algorithm: str, 
    search_library: str, tuned_top: List[sklearn.base.BaseEstimator]) -> List[sklearn.base.BaseEstimator]:    
    """Add custom est"""

    if config.get("est_task") == EstimatorTask.REGRESSION.value:
        custom_estimators = config.get("custom_regressors")
        available_estimators = CUSTOM_REGRESSORS
        custom_grid = config.get("custom_regressor_grid")
    else:
        custom_estimators = config.get("custom_classifiers")
        available_estimators = CUSTOM_CLASSIFIERS
        custom_grid = config.get("custom_classifier_grid")

    if len(custom_estimators) == 0:
        return tuned_top

    custom_grid = config.get_custom_grid(search_algorithm, search_library) 
    for k, v in custom_grid.items():
        custom_grid[k] = v

    model_list = []
    evaluation_metric = config.get("evaluation_metric")
    for custom_estimator in custom_estimators:
        if custom_estimator not in available_estimators:
            continue

        estimator_instance = available_estimators.get(custom_estimator)()
        trained_model = estimator.create_model(estimator_instance)
        tuned_model = estimator.tune_model(trained_model, search_algorithm = search_algorithm, optimize = evaluation_metric,
            search_library = search_library, n_iter = config.get("n_iter"), custom_grid = custom_grid.get(custom_estimator), 
            early_stopping = config.get("early_stopping_algo"), early_stopping_max_iters = config.get("early_stop"), 
            choose_better = True) 
        model_list.append(tuned_model)

    combined_models = estimator.compare_models(include = tuned_top + model_list, n_select = config.get("n_select"), 
        sort = evaluation_metric, turbo = config.get("turbo"))
    return combined_models

def train_ensemble_estimators(estimator: PyCaretEstimatorBase, config: Config, search_algorithm: str, 
    search_library: str) -> Tuple[sklearn.base.BaseEstimator]:    
    """Train several ensemble models, and return the best performing one."""
    evaluation_metric = config.get("evaluation_metric")
    n_estimators = config.get("n_estimators")
    n_iter = config.get("n_iter")
    custom_grid = config.get_custom_grid(search_algorithm, search_library) 
    ensemble_methods = config.get_as("ensemble_methods", set)

    # Train and tune estimators
    top_models = estimator.compare_models(include = config.get("include_estimators"), n_select = config.get("n_select"), 
        sort = evaluation_metric, turbo = config.get("turbo"))
    sorted_models = estimator.pull().index

    if type(top_models) is not list:
        top_models = [top_models]
    tuned_top = [ 
        estimator.tune_model(model, search_algorithm = search_algorithm, optimize = evaluation_metric,
            search_library = search_library, n_iter = n_iter, custom_grid = custom_grid.get(sorted_models[i]), 
            early_stopping = config.get("early_stopping_algo"), early_stopping_max_iters = config.get("early_stop"), 
            choose_better = True) 
        for i, model in enumerate(top_models) 
    ]

    tuned_top = add_custom_estimators(estimator, config, search_algorithm, search_library, tuned_top)
    
    # Train ensemble estimators
    if "stacking" in ensemble_methods:
        meta_model = estimator.create_model(config.get("meta_model"))
        tuned_meta_model = estimator.tune_model(meta_model, search_algorithm = search_algorithm, 
            optimize = evaluation_metric, search_library = search_library, n_iter = n_iter, custom_grid = custom_grid.get(meta_model)) 
        stacking_ensemble = estimator.stack_models(tuned_top, optimize = evaluation_metric, meta_model = tuned_meta_model)

    if "blending" in ensemble_methods:
        blending_ensemble = estimator.blend_models(tuned_top, optimize = evaluation_metric, choose_better = True)

    if "boosting" in ensemble_methods:
        boosting_ensemble = estimator.ensemble_model(tuned_top[0], method = "Boosting", optimize = evaluation_metric, 
            choose_better = True, n_estimators = n_estimators)

    if "bagging" in ensemble_methods:
        bagging_ensemble = estimator.ensemble_model(tuned_top[0], method = "Bagging", optimize = evaluation_metric, 
            choose_better = True, n_estimators = n_estimators)

    if "blended_boosting" in ensemble_methods:
        boosted_top = [ 
            estimator.ensemble_model(model, method = "Boosting", optimize = evaluation_metric, 
                choose_better = True, n_estimators = n_estimators)
            for model in top_models 
        ]
        boosted_blending_ensemble = estimator.blend_models(boosted_top, optimize = evaluation_metric, choose_better = True)

    if "blended_bagging" in ensemble_methods:
        bagging_top = [ 
            estimator.ensemble_model(model, method = "Bagging", optimize = evaluation_metric, 
                choose_better = True, n_estimators = n_estimators)
            for model in top_models 
        ]
        bagging_blending_ensemble = estimator.blend_models(bagging_top, optimize = evaluation_metric, choose_better = True)

    # Use AutoML to select best model in session
    best_model = estimator.automl(optimize = evaluation_metric)        
    final_ensemble = estimator.finalize_model(best_model)
    return best_model, final_ensemble

def _get_setup_kwargs(config: Config, data: pd.DataFrame, experiment_name: str) -> dict:
    """Get PyCaret model setup arguments from the experiment configuration."""
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
        "categorical_imputation": "mode",
        "session_id": config.get("random_seed")
    }

    config_args = ["imputation_type", "iterative_imputation_iters", "use_gpu", "remove_multicollinearity", 
        "multicollinearity_threshold", "pca", "pca_method", "pca_components", "ignore_features", "normalize", 
        "normalize_method", "transformation", "transformation_method", "group_features", "categorical_features", 
        "ordinal_features", "numeric_features", "high_cardinality_features", "date_features", "combine_rare_levels",
        "rare_level_threshold", "n_jobs"]
    for config_arg in config_args:
        kwargs[config_arg] = config.get(config_arg)

    return kwargs

def setup(estimator: PyCaretEstimatorBase, config: Config, data: pd.DataFrame, experiment_name: str):
    kwargs = _get_setup_kwargs(config, data, experiment_name)
    kwargs["fold_shuffle"] = True
    
    config_args = ["target", "fold", "fold_groups", "fold_strategy", "polynomial_features", "polynomial_degree", 
        "feature_selection", "feature_selection_method", "feature_selection_threshold", "feature_interaction", 
        "feature_ratio", "interaction_threshold", "remove_outliers", "outliers_threshold", "create_clusters",
        "cluster_iter"]
    for config_arg in config_args:
        kwargs[config_arg] = config.get(config_arg)

    return estimator.setup(**kwargs) 
    
def unsupervised_setup(config: Config, data: pd.DataFrame, experiment_name: str, type: str = EstimatorTask.CLUSTERING.value):
    if type == EstimatorTask.ANOMALY_DETECTION.value:
        from pycaret.anomaly import setup
    else:
        from pycaret.clustering import setup

    kwargs = _get_setup_kwargs(config, data, experiment_name)
    return setup(**kwargs) 

def get_unsupervised_custom_model(model_name: str, type: str, **kwargs) -> sklearn.base.BaseEstimator:
    if type == EstimatorTask.ANOMALY_DETECTION.value:
        return CUSTOM_ANOMALY_DETECTION.get(model_name)(**kwargs)
    else:
        return CUSTOM_CLUSTERING.get(model_name)(**kwargs)
    
def save_local_model(model, experiment_name: str, path = "data") -> str:
    """Save the model to a local directory."""
    model_path = join_path(path, f"{experiment_name}.joblib")
    dump(model, model_path) 
    return model_path
