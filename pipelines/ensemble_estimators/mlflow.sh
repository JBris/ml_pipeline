#!/usr/bin/env bash

. ../../.env

###################################################################
# Functions
###################################################################

usage() {
    echo """
    Usage: ${0} [MLFLOW_TRACKING_URI] 

    Parameters:
        MLFLOW_TRACKING_URI The URI for the MLFlow instance.
    """
}

help() {
    if [ "${1}" = "-h" ] || [ "${1}" = "-help" ]; then
        usage
        exit 1
    fi
}

###################################################################
# Main
###################################################################

# Exports
export MLFLOW_TRACKING_URI="${1:-${MLFLOW_TRACKING_URI}}"

#export RAY_ADDRESS=auto # Uncomment if using Ray

# Parameter overrides
example=-1

mlflow run ${MLFLOW_CONDA} . -P example=${example}  


