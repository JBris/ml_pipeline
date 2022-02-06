"""
Machine Learning Pipeline Pipelines

A library for constructing pipeline components within the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
from typing import Callable, List

import sklearn
import mlflow
import os
import pandas as pd
import tempfile

from dataclasses import dataclass, field

# Internal
from pipeline_lib.config import Config
from pipeline_lib.data import join_path
from pipeline_lib.estimator import save_local_model 

##########################################################################################################
### Library  
##########################################################################################################

def init_mlflow(config: Config) -> tempfile.TemporaryDirectory:
    """Initialise the MLFlow run."""
    mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI")) # Enable tracking using MLFlow
    mlflow.start_run()
    tmp_dir = tempfile.TemporaryDirectory()
    return tmp_dir

def end_mlflow(project_name: str, experiment_name: str, tmp_dir: tempfile.TemporaryDirectory) -> None:
    """End the MLFlow run."""
    mlflow.set_tag("project", project_name)
    mlflow.set_tag("experiment", experiment_name)
    mlflow.end_run()
    tmp_dir.cleanup()

@dataclass
class PlotParameters:
    """Parameters for saving pipeline plots."""
    plot_model: Callable
    feature: str = None
    plots: List[str] = field(default_factory = list)
    model: sklearn.base.BaseEstimator = None

def pipeline_plots(plot_params: PlotParameters, default_model: sklearn.base.BaseEstimator, save: str,
    log_artifact = False) -> None:
    """Save pipeline plots from a plot parameter object."""    
    if plot_params.model is None:
        plot_params.model = default_model

    kwargs = { "save": save }
    if plot_params.feature is not None:
        kwargs["label"] = True
        kwargs["feature"] = plot_params.feature
    
    for plot in plot_params.plots:
        kwargs["plot"] = plot
        model_plot = plot_params.plot_model(plot_params.model, **kwargs)
        if log_artifact:
            mlflow.log_artifact(join_path(save, model_plot))

def save_local_results(config: Config, model, experiment_name: str, assigned_df: pd.DataFrame = None,
    plot_params: PlotParameters = None) -> None:
    """Save pipeline results to the local data directory."""
    save_local_model(model, experiment_name)
    config.export("data")

    if assigned_df is not None:
        assigned_df.to_csv(join_path("data", f"{experiment_name}.csv"))

    if plot_params is not None:
        pipeline_plots(plot_params, model, "data")

def save_mlflow_results(config: Config, model, experiment_name: str, tmp_dir: tempfile.TemporaryDirectory, 
    assigned_df: pd.DataFrame = None, plot_params: PlotParameters = None) -> None:
    """Save pipeline results to the MLFlow server."""
    
    config_path = config.export(tmp_dir.name)
    mlflow.log_artifact(config_path)

    mlflow.sklearn.log_model(model, experiment_name, registered_model_name = experiment_name)

    if assigned_df is not None:
        df_path = join_path(tmp_dir.name, f"{experiment_name}.csv")
        assigned_df.to_csv(df_path)
        mlflow.log_artifact(df_path)

    if plot_params is not None:
        pipeline_plots(plot_params, model, tmp_dir.name, True)