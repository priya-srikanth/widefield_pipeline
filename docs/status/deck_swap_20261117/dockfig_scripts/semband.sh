#!/usr/bin/env bash
# Plot-only redraw of the LOCLIN like-for-like curves (licks beyond the pre range kept; edge line carried) with the SEM as
# a faded band instead of error bars (Priya 11-17 "a version with the SEM as faded bar, like the like-for-like graphs").
# Figure data seeded (byte copy) from production into semband_<set>_20261117/<tag>/figdata -> new folders; production
# PNGs are not touched. No number changes.
H=$(dirname "$0")
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
TAG=hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
for fs in "allITI none ." "dock dock itifilter_dock" "gap2dock gap2dock itifilter_gap2dock"; do
  set -- $fs
  D=$I/semband_$1_20261117
  mkdir -p "$D/$TAG/figdata" "$D/logs"
  [ -e "$D/$TAG/figdata/lick_matched_loclin.pkl" ] || cp -p "$I/$TAG/$3/figdata/lick_matched_loclin.pkl" "$D/$TAG/figdata/"
  echo "$(date '+%F %T') seeded $D/$TAG/figdata/lick_matched_loclin.pkl from $I/$TAG/$3/figdata" >> "$D/logs/SEEDED_FROM.txt"
  bash "$H/run_step.sh" redraw_semband "semband_$1_20261117" "$2" scripts/lick_matched_readout.py --ref loclin --plot-only \
    --figs curves --sem-band || echo "FAILED $1"
done
echo "SEMBAND DONE"
