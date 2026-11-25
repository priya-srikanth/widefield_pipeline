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
run goal_fig1     scripts/rebuild_figs/fig_goal_summary.py --enc-dir "$VIEW" --only fig1 --out $F
run softmax       scripts/rebuild_figs/fig_softmax.py --out $F
run pooled        scripts/g4figs_1125/g4_pooled.py --tables "$VIEW" --out $F
run fr_pixsig     scripts/fr_pixel_significance.py --mode A --jobs 2
run variance_maps scripts/variance_maps.py --jobs 2
run fr_direction  scripts/fr_direction.py --jobs 2
run fr_mechanism  scripts/fr_direction_mechanism.py --bundle "$B/inputs" --jobs 2
run decomp_lick   scripts/lp_encoder_decompose.py --folder movement_first_exec-polar_licks-combined --align lick
run decomp_cue    scripts/lp_encoder_decompose.py --folder movement_first_exec-polar_licks-combined --align cue
echo "$(date +%T) CHAIN DONE"
