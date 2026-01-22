#!/bin/bash
# YES/NO.EVENTS Quant Models CLI
cd "$(dirname "$0")"
source .venv/bin/activate
python quant.py "$@"
