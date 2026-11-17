#!/usr/bin/env bash
# Batch 2 of the ITI 0..0.5 s re-render (11-17): area maps (docked = page 14.29-14.32; all-ITI split = recovery-markers
# input), recovery markers (inline image after 14.18), slide-95 session-z tables (slides 110 / 111). <= 3 workers.
H=C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/782a4a7a-7855-40ed-b449-494e8aa4b6bc/scratchpad/iti05
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
TAG=hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
( bash $H/run_step.sh area_maps_dock area_maps_dock_iti05_20261117 1 scripts/iti_area_maps.py
  bash $H/run_step.sh area_maps_split_dock area_maps_dock_iti05_20261117 1 scripts/iti_area_maps.py --split
  bash $H/run_step.sh area_maps_split_farcontra_dock area_maps_dock_iti05_20261117 1 scripts/iti_area_maps.py --split --spout far_contra ) &
( bash $H/run_step.sh area_maps_split area_maps_iti05_20261117 0 scripts/iti_area_maps.py --split
  bash $H/run_step.sh recovery_markers area_maps_iti05_20261117 0 scripts/lick_recovery_markers.py ) &
( EXTRA_ENV="WFIELD_SESSZ_ITI_ROOT=$I/$TAG WFIELD_SESSZ_OUT=$I/slide95_sessz_iti05_20261117" LOGFOLDER=slide95_sessz_iti05_20261117 \
    bash $H/run_step.sh slide95_sessz - 0 scripts/lick_slide95_sessz.py ) &
wait
echo BATCH2 DONE
