#!/usr/bin/env bash
# Two waves, at most three scripts at once (shared box): per-session extractors, then the ones that read quiet_fluct output.
R=$(dirname "$0")/run_followon_syncend.sh
w() { while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done; bash "$R" "$@" & }
T0=$(date +%s)
w quiet_fluct.py --jobs 3
w quiet_eta_split.py --jobs 3
w quiet_quit_state.py --jobs 3
w quiet_breathing.py --jobs 3
w quiet_event_width.py --jobs 3
w quiet_state_response.py --jobs 3
w attempt_effort.py --jobs 3
w quiet_task_events_followup.py --jobs 3
w quiet_steinmetz_check.py
w quiet_hemi_amp_recovery.py
wait
w quiet_sync_index.py
w quiet_sync_4to8.py
w quiet_recovery_link.py
w quiet_f0_noise.py --jobs 3
w quiet_lick_lfo.py --jobs 3
wait
echo "all follow-ons $(( $(date +%s) - T0 )) s"
