##########################################################################################################
### Imports
##########################################################################################################

import argparse
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id
import os

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Generate a synthetic dataset.'
)

def _add_argument(parser: argparse.ArgumentParser, name: str, default: str, arg_help: str, type = str) -> None:
    parser.add_argument(
        name,
        default=default,
        help=f"{arg_help}. Defaults to '{default}'.",
        type=type
    )

_add_argument(parser, "--input", None, "input")
_add_argument(parser, "--prepared", "prepared", "prepared")
_add_argument(parser, "--model", None, "model")
_add_argument(parser, "--features", "features", "features")
_add_argument(parser, "--scores", None, "scores")
_add_argument(parser, "--plot_data", None, "plot_data")
_add_argument(parser, "--use_conda", None, 0, int)

args = parser.parse_args()
args.use_conda = bool(args.use_conda)

##########################################################################################################
### Main
##########################################################################################################

def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, use_conda = args.use_conda)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

def workflow():
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        prepare_run = _get_or_run(
            "prepare", {"prepare_file": args.input, "prepare_output": args.prepared}, git_commit
        )
        prepare_uri = os.path.join(prepare_run.info.artifact_uri, args.prepared)

        featurization_run = _get_or_run(
            "featurization", {"featurize_input": prepare_uri, "featurize_output": args.features}, git_commit
        )
        featurization_uri = os.path.join(featurization_run.info.artifact_uri, args.features)

        _get_or_run(
            "train", {"train_input": featurization_uri, "train_output": args.model}, git_commit
        )

        _get_or_run(
            "evaluate", {"evaluate_model": args.model, "evaluate_input": featurization_uri, 
            "evaluate_scores": args.scores, "evaluate_plot_data": args.plot_data}, git_commit
        )

if __name__ == "__main__":
    workflow()