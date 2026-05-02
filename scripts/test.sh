#!/bin/bash
# Run the metrics test suite.
set -e
cd "$(dirname "$0")/.."
python -m pytest tests/ -v "$@"
