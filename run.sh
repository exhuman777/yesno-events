#!/bin/bash
# Start dashboard with venv Python
cd "$(dirname "$0")"
source .venv/bin/activate
python dashboard4all.py "$@"
