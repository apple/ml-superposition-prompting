#!/usr/bin/env bash
set -Eeux
shopt -s expand_aliases

if ! [ -d .git ] || ! [ "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd)" ]; then echo "Must run from root repo"; exit; fi

wget -i flows/lost_in_the_middle/data/files.txt -P flows/lost_in_the_middle/data