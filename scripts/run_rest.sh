#!/bin/bash
# Remaining analyses, in dependency order. Run after 05_main_regimes.py.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
set -x
$PY scripts/14_subject_bias.py     > logs/14_bias.log       2>&1
$PY scripts/06_controls.py         > logs/06_controls.log   2>&1
$PY scripts/12_interpretation.py   > logs/12_interp.log     2>&1
$PY scripts/17_naive_splits.py     > logs/17_naive.log      2>&1
$PY scripts/08_train_final.py      > logs/08_train.log      2>&1
$PY scripts/11_holdout_check.py    > logs/11_holdout.log    2>&1
$PY scripts/10_who_is_decodable.py > logs/10_decodable.log  2>&1
$PY scripts/09_figures.py          > logs/09_figures.log    2>&1
$PY scripts/13_make_results_md.py  > logs/13_report.log     2>&1
echo "CPU CHAIN DONE"
