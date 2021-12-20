#!/usr/bin/env bash

###################################################################
# Functions
###################################################################

usage() {
    echo """
    Usage: ${0} -d -e [-p <string>]

    Options:
        d   Deploy the Docker stack.
        e   Recreate the .env file from .env.example.
        p   Create a new virtual environment with dependencies using Conda.
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

while getopts "dep" opt; do
    case $opt in
        e) 
            cp .env.example .env  
            echo "Created .env" 
            ;;
        d)
            . .env
            ./init_docker.sh
            ;;
        p)
            conda env create -f environment.yml python=3.8 || conda env update -f environment.yml python=3.8
            ;;
        *) 
            echo 
            ;;
  esac
done

shift $(($OPTIND - 1))
echo "Done"
