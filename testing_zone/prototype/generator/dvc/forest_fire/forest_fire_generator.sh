#!/usr/bin/env bash

dvc run --force -n generate \
          -p forest_fires.seed,forest_fires.split,forest_fires.mean,forest_fires.std \
          -d forest_fire_generator.py -d ../../../data/forestfires.csv  \
          -o out/forestfires.csv \
          python forest_fire_generator.py

### For ../../../data/forestfires.csv, we can do: 
# dvc add 
# dvc checkout

### To reproduce the pipeline
# dvc repro 