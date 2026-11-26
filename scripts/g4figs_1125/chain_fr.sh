#!/bin/bash
# G4 readers chain (CC-B helper 11-25): one reader at a time, joblib <= 2 workers. Reads the bundle; writes only OUT.
cd /c/Users/SabatiniLab/Github/wf_ccb_g4figs1125
export PYTHONPATH=$PWD
PY=/c/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe
OUT="M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/figures_working/g4figs_ccB_20261125"
VIEW="$OUT/imaging_encoding_view/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
L=$OUT/logs; F=$OUT/figs; mkdir -p $F
B="M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/o2/enc_fourcam3d_20261124"
run() { n=$1; shift; echo "$(date +%T) START $n"; $PY scripts/g4figs_1125/g4run.py "$@" > $L/$n.log 2>&1; echo "$(date +%T) END $n rc=$?"; }

run fr_musall_A   scripts/fr_musall.py --mode A
run fr_trial_peth scripts/fr_trial_peth.py
run fr_contra     scripts/fr_contra_orofacial.py --mode A
echo "$(date +%T) FRCHAIN DONE"
