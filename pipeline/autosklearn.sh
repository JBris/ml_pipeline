#!/usr/bin/env bash

docker pull mfeurer/auto-sklearn:master

docker run -it -v ${PWD}:/opt/nb -p 8888:8888 mfeurer/auto-sklearn:master /bin/bash -c "mkdir -p /opt/nb && jupyter notebook --notebook-dir=/opt/nb --port=8888 --no-browser --allow-root --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password=''"