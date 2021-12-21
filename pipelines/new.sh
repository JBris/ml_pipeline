#!/usr/bin/env bash

###################################################################
# Functions
###################################################################

usage() {
    echo """
    Usage: ${0} 
    
    Creates a new pipeline project and adds scaffolding files.
    Can be used for both DVC and MLFlow projects.

    """
}

help() {
    if [ "${1}" = "-h" ] || [ "${1}" = "-help" ]; then
        usage
        exit 1
    fi
}

create_project() {
    mkdir -p "${1}/src"
    mkdir -p "${1}/data"
    touch "${1}/data/.keep"

    envsubst '${name} ${description}' < ../templates/run.py > "${1}/src/run.py"
    envsubst '${name} ${description}' < ../templates/README.md > "${1}/README.md"
    cp ../templates/gitignore "${1}/.gitignore"
    cp ../templates/dockerignore "${1}/.dockerignore"
    cp ../templates/params.yaml "${1}/params.yaml"

    if [ "${2}" = "dvc" ]; then
        cp ../templates/dvc.sh "${1}/dvc.sh"
        cp ../templates/dvc.yaml "${1}/dvc.yaml"
        current_dir=$(pwd)
        cd "${1}"
        dvc init --subdir
        cd "${current_dir}"
    elif [ "${2}" = "mlflow" ]; then
        cp ../templates/mlflow.sh "${1}/mlflow.sh"
        envsubst '${name} ${description}' < ../templates/MLproject > "${1}/MLproject"
        touch "${1}/conda.yaml"
    fi
}

###################################################################
# Main
###################################################################

help "${1}"

while true; do
   read -p "Please enter the project name: " name
   if [ -z "$name" ]; then
      echo "Project name cannot be empty."
      echo
   else
      break
   fi
done


read -p "Please enter the project description [optional]: " description
read -p "Please enter the project type. Can be 'dvc', 'mlflow', or none [optional]: " project_type

echo "Creating ${name} in the pipelines directory..."
export name
export description
create_project "${name}" "${project_type}"
echo "Done"
