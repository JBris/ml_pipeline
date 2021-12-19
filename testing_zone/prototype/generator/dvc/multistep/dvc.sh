#!/usr/bin/env bash

dvc run --force -n prepare \
          -p prepare.seed,prepare.split \
          -d src/prepare.py -d ../../../data/forestfires.csv \
          -o data/prepared \
          python src/prepare.py ../../../data/forestfires.csv data/prepared

dvc run --force -n featurize \
          -p featurize.mean,featurize.std \
          -d src/featurization.py -d data/prepared \
          -o data/features \
          python src/featurization.py data/prepared data/features

dvc run --force -n train \
          -p train.seed,train.n_est,train.min_split \
          -d src/train.py -d data/features \
          -o model.pkl \
          python src/train.py data/features model.pkl

dvc run --force -n evaluate \
          -d src/evaluate.py -d model.pkl -d data/features \
          -M scores.json \
          --plots-no-cache pred_actual.json \
          python src/evaluate.py model.pkl \
                 data/features scores.json pred_actual.json  