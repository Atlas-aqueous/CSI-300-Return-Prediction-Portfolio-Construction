#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$SCRIPT_DIR/model" "$SCRIPT_DIR/output" "$SCRIPT_DIR/temp"
echo "Init finished: directories are ready."
