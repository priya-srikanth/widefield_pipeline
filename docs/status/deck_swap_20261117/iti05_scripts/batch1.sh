#!/usr/bin/env bash
# Batch 1 of the ITI 0..0.5 s re-render (11-17). Plot-only where the window is applied in the drawing (lateral ratio and
# vigor: both windows are already in the figure data / session tables), compute where it is applied in the analysis.
H=C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/782a4a7a-7855-40ed-b449-494e8aa4b6bc/scratchpad/iti05
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
TAG=hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
seed() {  # seed <src tag dir> <dst folder> <files...>: copy the figure data / tables a --plot-only run reads (never overwrite)
  src=$1; dst=$I/$2/$TAG; shift 2; mkdir -p "$dst/figdata" "$I/$2/logs"
  for f in "$@"; do [ -e "$dst/$f" ] || cp -p "$src/$f" "$dst/$f"; done
  echo "seeded $dst from $src: $*" >> "$I/$2/logs/SEEDED_FROM.txt"
}
seed "$I/$TAG" lateral_ratio_iti05_20261117 figdata/lick_lateral_ratio.pkl lateral_ratio_sessions.csv lateral_ratio_tests.csv lateral_ratio_models.csv
seed "$I/lateral_ratio_dock_20261117/$TAG" lateral_ratio_dock_iti05_20261117 figdata/lick_lateral_ratio.pkl lateral_ratio_sessions.csv lateral_ratio_tests.csv lateral_ratio_models.csv
seed "$I/$TAG" vigor_iti05_20261117 figdata/lick_vigor_perspout.pkl
bash $H/run_step.sh lateral_ratio_plot lateral_ratio_iti05_20261117 0 scripts/lick_lateral_ratio.py --plot-only &
bash $H/run_step.sh lateral_ratio_dock_plot lateral_ratio_dock_iti05_20261117 1 scripts/lick_lateral_ratio.py --plot-only &
bash $H/run_step.sh vigor_plot vigor_iti05_20261117 0 scripts/lick_vigor_perspout.py --plot-only &
wait
bash $H/run_step.sh cued_ramp cued_ramp_iti05_20261117 0 scripts/lick_cued_ramp.py &
bash $H/run_step.sh cued_ramp_dock itidock_iti05_20261117 1 scripts/lick_cued_ramp.py &
bash $H/run_step.sh balance_angle lateral_ratio_iti05_20261117 0 scripts/lick_balance_angle_lmm.py &
wait
echo BATCH1 DONE
