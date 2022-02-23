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
import time
import uuid

from dataclasses import dataclass, field

# Internal
from pipeline_lib.config import Config
from pipeline_lib.data import join_path
from pipeline_lib.estimator import save_local_model 

##########################################################################################################
### Library  
##########################################################################################################

def get_experiment_name(project_name: str, config: Config) -> str:
    """
    Get the name of the experiment.

    Parameters
    --------------
    project_name: str
        The name of the project.
    config: Config
        The pipeline configuration object.

    Returns
    ---------
    experiment_name: str
        The name of the experiment.
    """
    scenario = config.get('scenario')
    if scenario is None or scenario == ".":
        scenario = "experiment"

    experiment_name = f"{project_name}_{scenario}"
    return experiment_name

def init_mlflow(config: Config) -> tempfile.TemporaryDirectory:
    """Initialise the MLFlow run."""
    mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI")) # Enable tracking using MLFlow
    mlflow.start_run()
    tmp_dir = tempfile.TemporaryDirectory()
    return tmp_dir

def end_mlflow(project_name: str, experiment_name: str, tmp_dir: tempfile.TemporaryDirectory, author: str = None) -> None:
    """End the MLFlow run."""
    mlflow.set_tag("project", project_name)
    mlflow.set_tag("experiment", experiment_name)
    if author is not None:
        mlflow.set_tag("author", author)
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

def create_local_directory(config: Config) -> str:
    """
    Create a local directory to save pipeline results.

    Parameters
    --------------
    config: Config
        The pipeline configuration object.

    Returns
    ---------
    full_dir: str
        The full local directory path.
    """
    current_time = time.strftime("%Y-%m-%d-%H_%M", time.gmtime())
    dir_id = str(uuid.uuid1())
    local_dir_name = f"{current_time}_{dir_id}"
    # Add optional prefix
    local_data_prefix = config.get("local_data_prefix")
    if local_data_prefix is not None:
        local_dir_name = f"{local_data_prefix}-{local_dir_name}"

    full_dir = join_path("data", local_dir_name)
    os.makedirs(full_dir, exist_ok = True)
    return full_dir
    
def save_local_results(config: Config, model, experiment_name: str, assigned_df: pd.DataFrame = None,
    plot_params: PlotParameters = None, save_path: str = None) -> None:
    """Save pipeline results to the local data directory."""
    if save_path is None:
        save_path = create_local_directory(config)
    save_local_model(model, experiment_name, path = save_path)
    config.export(save_path)

    if assigned_df is not None:
        assigned_df.to_csv(join_path(save_path, f"{experiment_name}.csv"))

    if plot_params is not None:
        pipeline_plots(plot_params, model, save = save_path)

    return save_path

def save_mlflow_model(config: Config, model, experiment_name: str) -> None:
    """Save the model to the MLFlow model registry."""
    model_info = mlflow.sklearn.log_model(model, artifact_path = experiment_name)
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{experiment_name}" 
    model_version = mlflow.register_model(model_uri, experiment_name)

    model_description = config.get("model_description")
    if model_description is not None:
        author = config.get("author")
        if author is not None:
            model_description += f" - {author}"
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(experiment_name, model_version.version, model_description)

def save_mlflow_results(config: Config, model, experiment_name: str, tmp_dir: tempfile.TemporaryDirectory, 
    assigned_df: pd.DataFrame = None, plot_params: PlotParameters = None) -> None:
    """Save pipeline results to the MLFlow server."""
    
    config_path = config.export(tmp_dir.name)
    mlflow.log_artifact(config_path)
    
    # Saving model to MLFlow registry
    save_mlflow_model(config, model, experiment_name)

    if assigned_df is not None:
        df_path = join_path(tmp_dir.name, f"{experiment_name}.csv")
        assigned_df.to_csv(df_path)
        mlflow.log_artifact(df_path)

    if plot_params is not None:
        pipeline_plots(plot_params, model, tmp_dir.name, True)