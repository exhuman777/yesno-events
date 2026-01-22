#!/bin/bash
# YES/NO.EVENTS Tweet Tracker CLI
cd "$(dirname "$0")"
source .venv/bin/activate
python tracker.py "$@"
