#!/bin/bash
# YES/NO.EVENTS - Polymarket Quant Terminal
cd "$(dirname "$0")"
source .venv/bin/activate
python app.py "$@"
