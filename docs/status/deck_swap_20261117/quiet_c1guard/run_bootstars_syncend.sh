#!/usr/bin/env bash
# Section-23 follow-ons: bootstrap-primary stars (Priya 11-18 "yes add"; quiet_boot_stars.py). Re-plot from the stored
# per-session outputs of run_all_followons_syncend.sh (summary / plot modes; the cheap whole-script ones re-run). Logs
# <run>/logs/<script>_bootstars.log.
R=$(dirname "$0")/run_followon_syncend.sh
export QS_LOG_SUFFIX=_bootstars
bash "$R" quiet_sync_index.py
bash "$R" quiet_f0_noise.py --jobs 3
bash "$R" quiet_sync_4to8.py
bash "$R" quiet_hemi_amp_recovery.py
bash "$R" quiet_breathing.py --summarise
bash "$R" quiet_event_width.py --summarise
bash "$R" quiet_lick_lfo.py --summarise
bash "$R" quiet_state_response.py --summarise
