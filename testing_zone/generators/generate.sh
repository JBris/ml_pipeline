#!/usr/bin/env bash

generator_type="$1"
args="${@:2}"
generator_types=("plugins" "services")

if [[ ! " ${generator_types[*]} " =~ " ${generator_type} " ]]; then
    echo "Error: Invalid generator type: ${generator_type}" 
    exit 1
fi

python -m "${generator_type}.generator" ${args}