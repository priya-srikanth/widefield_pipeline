#!/usr/bin/env bash
# Plot-only redraws of the SIBLING figures after the footer fix reached them (trial_set_text.footer_bottom as the bottom
# of each tight_layout rect; recovery-link pooled legend stacked above the footer). Figure data only, no number changes.
H=$(dirname "$0")
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
TAG=hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
R="bash $H/run_step.sh"
for fs in "dock_dockfig_20261117 dock" "gap2dock_dockfig_20261117 gap2dock"; do
  set -- $fs
  $R redraw_rlink_aim_z $1 $2 scripts/iti_recovery_link.py --beh aim --z --plot-only
  $R redraw_rlink_angle_z $1 $2 scripts/iti_recovery_link.py --beh angle --z --plot-only
  $R redraw_scatter_z $1 $2 scripts/iti_where_scatter.py --z --plot-only
  $R redraw_checkfigs $1 $2 scripts/iti_check_figures.py --plot-only
  $R redraw_where_angle_reach $1 $2 scripts/iti_where_angle_reach.py --plot-only
done
$R redraw_rlink_aim_z allITI_dockfig_20261117 none scripts/iti_recovery_link.py --beh aim --z --plot-only \
   --src "$I/allITI_dockfig_20261117/$TAG"
echo "REDRAW2 DONE"
