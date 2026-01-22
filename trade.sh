#!/bin/bash
# Quick trading CLI - wraps trading.py with venv
cd "$(dirname "$0")"
source .venv/bin/activate
python trading.py "$@"
