#!/usr/bin/env bash

###################################################################
# Functions
###################################################################

usage() {
    echo """
    Usage: ${0} [-c <string>] -d -e -p

    Options:
        c   Create a new virtual environment (with the specified name) including dependencies using Conda.
        d   Deploy the Docker stack.
        e   Recreate the .env file from .env.example.
        p   install Python requirements using pip
    """
}

help() {
    if [ "${1}" = "-h" ] || [ "${1}" = "-help" ]; then
        usage
        exit 1
    fi
}

create_file () {
    if [ ! -f "${1}" ]; then 
        cp "${2}" "${1}" 
        echo "Created ${1}"
    fi
}

###################################################################
# Main
###################################################################

help "${1}"

create_file config.local.ini config.local.ini.example
create_file .env  .env.example

while getopts "c:dep" opt; do
    case $opt in
        c)
            name=${OPTARG}
            conda env create -n "${name}" -f environment.yml python=3.8 || conda env update -n "${name}" -f environment.yml  
            ;;
        d)
            . .env
            ./init_docker.sh
            ;;
        e) 
            cp .env.example .env  
            echo "Created .env" 
            ;;
        p)
            python -m pip install -r requirements.txt
            ;;
        *) 
            echo 
            ;;
  esac
done

shift $(($OPTIND - 1))
echo "Done"
