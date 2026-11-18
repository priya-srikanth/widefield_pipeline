#!/usr/bin/env bash
# Re-run of the two docked lick_matched_angle_reach steps of batch.sh (first try exit 1: the figure's sigma dict indexed the
# skipped 3-axis axes; fixed in compute), then the plot-only redraws (redraw.sh: footer margin + dashed-line notes).
H=$(dirname "$0")
R="bash $H/run_step.sh"
$R matched_axes_curves_try2 dock_dockfig_20261117 dock scripts/lick_matched_angle_reach.py --curves &
$R matched_axes_curves_try2 gap2dock_dockfig_20261117 gap2dock scripts/lick_matched_angle_reach.py --curves &
wait
bash $H/redraw.sh
echo "BATCH2 DONE"
