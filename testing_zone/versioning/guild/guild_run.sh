#!/usr/bin/env bash

view="$1"
source_code="$2"
guild run train.py -y

if [[ "${view}" -eq 1 ]]; then
    guild view
fi

if [[ "${source_code}" -eq 1 ]]; then
   guild cat --sourcecode --path train.py
fi