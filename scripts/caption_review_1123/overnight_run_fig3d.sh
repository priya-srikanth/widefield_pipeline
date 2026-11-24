#!/bin/bash
# usage: run_fig3d.sh <classes> <outdir>
set -e
cd C:/Users/SabatiniLab/Github/wf_ccb_overnight1123
export PYTHONPATH=$PWD MKL_THREADING_LAYER=SEQUENTIAL
P=C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe
mkdir -p "$2"
$P scripts/c2_outcome_stats_aligned.py --out-dir "$2" --classes "$1" --n-boot 4000
$P scripts/c2_outcome_distribution_test.py --out-dir "$2" --classes "$1"
