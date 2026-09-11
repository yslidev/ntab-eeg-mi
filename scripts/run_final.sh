#!/bin/bash
# Runs after the main chain: the sequence-confound test, then regenerate.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until grep -q "CHAIN DONE" logs/chain.log 2>/dev/null; do sleep 30; done
set -x
$PY scripts/20_missing_controls.py  > logs/20_missing.log  2>&1
$PY scripts/19_sequence_confound.py > logs/19_sequence.log 2>&1
$PY scripts/18_fill_video_script.py > logs/18_video.log    2>&1
$PY scripts/09_figures.py           > logs/09_figures.log  2>&1
$PY scripts/13_make_results_md.py   > logs/13_report.log   2>&1
echo "FINAL DONE"
