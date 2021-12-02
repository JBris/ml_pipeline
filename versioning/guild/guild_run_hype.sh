#!/usr/bin/env bash

view="$1"
source_code="$2"

guild run train.py x='[-1,0,1]' -y
 
if [[ "${view}" -eq 1 ]]; then
    guild view
fi

if [[ "${source_code}" -eq 1 ]]; then
   guild cat --sourcecode --path train.py
fi

# Grid search
guild run train.py x=linspace[-0.6:0.6:4] -y

if [[ "${view}" -eq 1 ]]; then
    guild view
fi

if [[ "${source_code}" -eq 1 ]]; then
   guild cat --sourcecode --path train.py
fi

# Random search
guild run train.py x=[-2.0:2.0] --max-trials 5 -y

if [[ "${view}" -eq 1 ]]; then
    guild view
fi

if [[ "${source_code}" -eq 1 ]]; then
   guild cat --sourcecode --path train.py
fi

# Bayesian optimisation
guild run train.py x=[-2.0:2.0] --optimizer gp --max-trials 10 -y

if [[ "${view}" -eq 1 ]]; then
    guild view
fi

if [[ "${source_code}" -eq 1 ]]; then
   guild cat --sourcecode --path train.py
fi