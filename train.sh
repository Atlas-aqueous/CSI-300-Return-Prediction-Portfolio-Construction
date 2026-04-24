#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python -u "$SCRIPT_DIR/code/src/train.py"
