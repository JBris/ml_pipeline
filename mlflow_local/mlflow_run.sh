#!/usr/bin/env bash

. ../.env

###################################################################
# Variables
###################################################################

params=

###################################################################
# Functions
###################################################################

usage() {
    echo """
    Usage: ${0} [-c CONFIG_FILE] [-d SIZING_DIR] [-r RAY_ADDRESS] [-r SCENARIO] [-u MLFLOW_TRACKING_URI] [pipeline_directory]

    Parameters:
        pipeline_directory  The directory for MLflow pipelines.
        c   The optional config.yaml file.
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

help "${1}"

while getopts "c:d:r:s:u" opt; do
    case $opt in
        c)
            CONFIG_FILE=${OPTARG}
            ;;
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

shift $(($OPTIND - 1))

# Exports
export MLFLOW_TRACKING_URI
export RAY_ADDRESS
echo "MLFlow address: ${MLFLOW_TRACKING_URI}"
echo "Ray address: ${RAY_ADDRESS}"

# Parameter overrides
[[ ! -z "${CONFIG_FILE}" ]] && add_param from_config "${CONFIG_FILE}"
[[ ! -z "${SIZING_DIR}" ]] && add_param base_dir "${SIZING_DIR}"
[[ ! -z "${PIPELINE_SCENARIO}" ]] && add_param scenario "${PIPELINE_SCENARIO}"

mlflow run ${MLFLOW_CONDA} ../pipelines/${1} -b ${MLFLOW_BACKEND} ${params}
