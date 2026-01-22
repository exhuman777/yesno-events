#!/bin/bash
# YES/NO.EVENTS Vector Search CLI
cd "$(dirname "$0")"
source .venv/bin/activate
python search.py "$@"
