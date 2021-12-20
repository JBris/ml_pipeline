#!/usr/bin/env bash

dvc run --force -n generate \
          -p day.seed,day.split,day.mean,day.std \
          -d day_generator.py -d ../../../data/day.csv  \
          -o out/days.csv \
          python day_generator.py

### For ../../../data/day.csv, we can do: 
# dvc add 
# dvc checkout

### To reproduce the pipeline
# dvc repro 