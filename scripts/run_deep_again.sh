#!/bin/bash
# Re-run the network with the fixed early stopping, at a scale this laptop can afford.
cd "$(dirname "$0")/.."
while pgrep -f bigeegnet.py >/dev/null; do sleep 20; done
mv results/deep.csv results/deep_prebugfix.csv 2>/dev/null
set -x
.venv/bin/python scripts/07_deep.py --folds 3 --epochs 90 > logs/07_deep.log 2>&1
echo "DEEP DONE"
