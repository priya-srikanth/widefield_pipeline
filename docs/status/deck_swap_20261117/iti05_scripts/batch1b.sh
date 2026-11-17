#!/usr/bin/env bash
# Batch 1 retry (try 1 failed: <folder>/<tag>/figures did not exist). Waits for batch 2 (3 workers max on the box).
H=C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/782a4a7a-7855-40ed-b449-494e8aa4b6bc/scratchpad/iti05
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
until grep -q "BATCH2 DONE" $H/batch2.out 2>/dev/null; do sleep 20; done
for f in lateral_ratio_iti05_20261117/logs/lateral_ratio_plot lateral_ratio_dock_iti05_20261117/logs/lateral_ratio_dock_plot \
         vigor_iti05_20261117/logs/vigor_plot cued_ramp_iti05_20261117/logs/cued_ramp itidock_iti05_20261117/logs/cued_ramp_dock \
         lateral_ratio_iti05_20261117/logs/balance_angle; do
  [ -e "$I/$f.log" ] && mv "$I/$f.log" "$I/$f.failed_try1_nodir.log"
done
bash $H/run_step.sh lateral_ratio_plot lateral_ratio_iti05_20261117 0 scripts/lick_lateral_ratio.py --plot-only &
bash $H/run_step.sh lateral_ratio_dock_plot lateral_ratio_dock_iti05_20261117 1 scripts/lick_lateral_ratio.py --plot-only &
bash $H/run_step.sh vigor_plot vigor_iti05_20261117 0 scripts/lick_vigor_perspout.py --plot-only &
wait
bash $H/run_step.sh cued_ramp cued_ramp_iti05_20261117 0 scripts/lick_cued_ramp.py &
bash $H/run_step.sh cued_ramp_dock itidock_iti05_20261117 1 scripts/lick_cued_ramp.py &
bash $H/run_step.sh balance_angle lateral_ratio_iti05_20261117 0 scripts/lick_balance_angle_lmm.py &
wait
echo BATCH1B DONE
