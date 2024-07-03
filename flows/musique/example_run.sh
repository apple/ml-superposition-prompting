#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=false; poetry run python flows/musique/main.py -c flows/musique/configs/topk.yaml