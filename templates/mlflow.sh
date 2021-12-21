#!/usr/bin/env bash

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
export MLFLOW_TRACKING_URI="${1:-http://127.0.0.1:5000}"

#export RAY_ADDRESS=auto # Uncomment if using Ray

# Parameter overrides
example=-1

# Flag to use system environment instead of Conda or Docker: --no-conda 

mlflow run --no-conda . -P example=${example}  


