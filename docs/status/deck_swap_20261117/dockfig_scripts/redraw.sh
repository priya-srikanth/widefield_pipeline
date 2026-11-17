#!/usr/bin/env bash
# Plot-only redraws after Priya's 11-17 feedback on lick_matched_gauss_curves: (1) bottom margin reserved for the purple
# trial-set footer (trial_set_text.footer_frac), (2) a note naming the animal missing where a curve is mostly dashed, and
# the per-epoch coverage table in the log. No number changes (figure data only).
#   - the three _dockfig folders (their own figure data, written by batch.sh)
#   - the PRODUCTION all-ITI curves: figure data seeded (byte copy) into allITI_curves_redraw_dockfig_20261117 -> new path,
#     the production PNG is not touched
H=$(dirname "$0")
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
TAG=hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
D=$I/allITI_curves_redraw_dockfig_20261117
mkdir -p "$D/$TAG/figdata" "$D/logs"
[ -e "$D/$TAG/figdata/lick_matched_gauss.pkl" ] || cp -p "$I/$TAG/figdata/lick_matched_gauss.pkl" "$D/$TAG/figdata/"
echo "$(date '+%F %T') seeded $D/$TAG/figdata/lick_matched_gauss.pkl from $I/$TAG/figdata" >> "$D/logs/SEEDED_FROM.txt"
R="bash $H/run_step.sh"
$R redraw_curves allITI_curves_redraw_dockfig_20261117 none scripts/lick_matched_readout.py --ref gauss --plot-only --figs curves
for fs in "dock_dockfig_20261117 dock" "gap2dock_dockfig_20261117 gap2dock" "allITI_dockfig_20261117 none"; do
  set -- $fs
  figs=curves,readout,perspout,animal; [ "$1" = allITI_dockfig_20261117 ] && figs=curves
  $R redraw_matched $1 $2 scripts/lick_matched_readout.py --ref gauss --plot-only --first-curves --figs $figs
  $R redraw_axes $1 $2 scripts/lick_matched_angle_reach.py --plot-only --curves
done
echo "REDRAW DONE"
