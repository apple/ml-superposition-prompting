#!/usr/bin/env bash
set -Eeux
shopt -s expand_aliases

if ! [ -d .git ] || ! [ "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd)" ]; then echo "Must run from root repo"; exit; fi

# Download data into flows/musique.
cd flows/musique/
bash ../../third_party/StonyBrookNLP/musique/download_data.sh
cd ../../