#!/usr/bin/env bash
# Docked decoded-vs-executed re-render (Priya 11-17). Three chains, one per output folder (<= 3 workers at a time):
#   dock_dockfig_20261117      WFIELD_ITI_FILTER=dock      (docked-spout ITI licks)
#   gap2dock_dockfig_20261117  WFIELD_ITI_FILTER=gap2dock  (docked AND >= 2 s since the previous lick = bout start after quiet)
#   allITI_dockfig_20261117    no filter                   (all-ITI twins of the NEW figures only: first-lick curves, angle
#                                                           curves, recovery link aim_z -- not in production)
# The docked where caches are reused from itidock_main_20261117 (byte copies, seed below); the gap2dock caches are built by
# iti_dock_where_preds.py from the production all-ITI caches (nothing refit).
H=$(dirname "$0")
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
TAG=hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
seed() {  # seed <src tag dir> <dst folder> <files...>: byte copies of input caches (never overwrite)
  src=$1; dst=$I/$2/$TAG; shift 2; mkdir -p "$dst/zscore_check" "$I/$2/logs"
  for f in "$@"; do [ -e "$dst/$f" ] || cp -p "$src/$f" "$dst/$f"; done
  echo "$(date '+%F %T') seeded $dst from $src: $*" >> "$I/$2/logs/SEEDED_FROM.txt"
}
seed "$I/itidock_main_20261117/$TAG" dock_dockfig_20261117 where_preds_splitr8.csv where_angle_reach_preds_splitr8.csv.gz \
     zscore_check/where_preds_z_splitr8.csv zscore_check/where_decoder_z.csv where_dockflag.csv
seed "$I/$TAG" allITI_dockfig_20261117 zscore_check/where_preds_z_splitr8.csv zscore_check/where_decoder_z.csv

R="bash $H/run_step.sh"
(
  $R matched_gauss_first dock_dockfig_20261117 dock scripts/lick_matched_readout.py --ref gauss --first-curves
  $R matched_axes_curves dock_dockfig_20261117 dock scripts/lick_matched_angle_reach.py --curves
  $R where_angle_reach dock_dockfig_20261117 dock scripts/iti_where_angle_reach.py
  $R rlink_aim_z dock_dockfig_20261117 dock scripts/iti_recovery_link.py --beh aim --z
  $R rlink_angle_z dock_dockfig_20261117 dock scripts/iti_recovery_link.py --beh angle --z
  $R scatter_z dock_dockfig_20261117 dock scripts/iti_where_scatter.py --z
  $R checkfigs dock_dockfig_20261117 dock scripts/iti_check_figures.py
  echo "DOCK CHAIN DONE"
) &
(
  $R dock_where_preds gap2dock_dockfig_20261117 gap2dock scripts/iti_dock_where_preds.py
  $R matched_gauss_first gap2dock_dockfig_20261117 gap2dock scripts/lick_matched_readout.py --ref gauss --first-curves
  $R matched_axes_curves gap2dock_dockfig_20261117 gap2dock scripts/lick_matched_angle_reach.py --curves
  $R where_angle_reach gap2dock_dockfig_20261117 gap2dock scripts/iti_where_angle_reach.py
  $R rlink_aim_z gap2dock_dockfig_20261117 gap2dock scripts/iti_recovery_link.py --beh aim --z
  $R rlink_angle_z gap2dock_dockfig_20261117 gap2dock scripts/iti_recovery_link.py --beh angle --z
  $R scatter_z gap2dock_dockfig_20261117 gap2dock scripts/iti_where_scatter.py --z
  $R checkfigs gap2dock_dockfig_20261117 gap2dock scripts/iti_check_figures.py
  echo "GAP2DOCK CHAIN DONE"
) &
(
  $R matched_gauss_first allITI_dockfig_20261117 none scripts/lick_matched_readout.py --ref gauss --first-curves --figs curves
  $R matched_axes_curves allITI_dockfig_20261117 none scripts/lick_matched_angle_reach.py --curves
  $R rlink_aim_z allITI_dockfig_20261117 none scripts/iti_recovery_link.py --beh aim --z --src "$I/allITI_dockfig_20261117/$TAG"
  echo "ALLITI CHAIN DONE"
) &
wait
echo "BATCH DONE"
