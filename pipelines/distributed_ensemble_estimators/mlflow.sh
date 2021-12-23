#!/usr/bin/env bash

. ../../.env

###################################################################
# Variables
###################################################################

params=

###################################################################
# Functions
###################################################################

usage() {
    echo """
    Usage: ${0} [-d SIZING_DIR] [-r RAY_ADDRESS] [-r SCENARIO] [-u MLFLOW_TRACKING_URI] 

    Parameters:
        d   The root directory for the project.
        r   The Ray address.
        s   The pipeline scenario file.
        u   The URI for the MLFlow instance.
    """
}

help() {
    if [ "${1}" = "-h" ] || [ "${1}" = "-help" ]; then
        usage
        exit 1
    fi
}

add_param() {
    params="${params} -P ${1}=${2}"
}

###################################################################
# Main
###################################################################

while getopts ":d:r:s:u:" opt; do
    case $opt in
        d)
            SIZING_DIR=${OPTARG}
            ;;
        r)
            RAY_ADDRESS=${OPTARG}
            ;;
        s)
            PIPELINE_SCENARIO="${OPTARG}"
            ;;
        u)
            MLFLOW_TRACKING_URI=${OPTARG}
            ;;
        *) 
            echo 
            ;;
  esac
done

# Exports
export MLFLOW_TRACKING_URI
export RAY_ADDRESS
echo "MLFlow address: ${MLFLOW_TRACKING_URI}"
echo "Ray address: ${RAY_ADDRESS}"

# Target directory 
# Parameter overrides
[[ ! -z "${SIZING_DIR}" ]] && add_param base_dir "${SIZING_DIR}"
[[ ! -z "${PIPELINE_SCENARIO}" ]] && add_param scenario "${PIPELINE_SCENARIO}"

mlflow run ${MLFLOW_CONDA} . ${params}
