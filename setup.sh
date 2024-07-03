#!/usr/bin/env bash

# Conda setup.
CONDA_ENV_NAME="sp-env"
conda create -y -n ${CONDA_ENV_NAME} python=3.9
conda init bash
source /miniconda/etc/profile.d/conda.sh
# source ~/.bashrc
conda activate ${CONDA_ENV_NAME}

# Env setup.
python3 -m pip install "poetry>=1.6.1"
poetry lock
poetry install

# Deps setup.
git submodule update --init --recursive
cd third_party/huggingface/transformers
git am ../../../transformers.patch
python3 -m pip install -e .
cd ../../../