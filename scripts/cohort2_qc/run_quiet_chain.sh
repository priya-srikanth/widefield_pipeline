#!/usr/bin/env bash
# THE quiet-event chain, one build end to end (Priya 11-18: "ensure that our code will align all, ie with a re-run we'll
# use the correct build"). Replaces the night-of runners in docs/status/deck_swap_20261117/quiet_c1guard/ (run_chain_*,
# run_followon_*, run_bootstars_*, task_events_summarise_prodptrue.py), which pointed at a worktree, hard-coded output
# folders, and read T13 p_true from ANOTHER key's attempt_coding tables -- the mismatch that kept Fig 23.23 pre-rebuild.
#
# Every 11-18 decision is in the code on main, not in this file: engaged / quit end at the LAST SYNC PULSE
# (QS_ENGAGED_END=lastsync; quit = quit .. last sync), common k = 4, cam1 far-from-mouth guard, quit-time cache,
# stable injection seed, bootstrap-primary stars + LMM dagger (quiet_boot_stars), the follow-up scored in production's
# frame. This file only fixes the ENV (the production env of the 11-17/18 runs) and the ORDER.
#
# usage: scripts/cohort2_qc/run_quiet_chain.sh <OUT_DIRNAME> [jobs]     e.g. v2_quiet_lp_<tag>_<YYYYMMDD>
#   run from a CLEAN checkout of the commit you want (refuses a dirty tree); code = this checkout, recorded per log.
# PREREQUISITES (checked; the run stops if missing):
#   1. attempt_coding precue tables (T13 p_true) under THIS env's key, all four animals. attempt_coding writes them and
#      may FREEZE pre-stroke decoders (rule 10), so it is NOT run here: run it (coordinate with the encoding window)
#      with this same env and WIDEFIELD_LICK_SOURCE=daq:  python -m wfield_local.attempt_coding --aligns precue
#   2. the face-movement series (fr_regressed_own/own_series) and LP licks for this tag (the upstream chains).
#   A re-run after new pose / 3-D inputs (e.g. four-camera 3-D) must redo those upstream steps first, under one tag.
# OUTPUT: <rebuild>/labcams/<tag>/cohort1_quiet_spikes/<OUT_DIRNAME>/, logs in its logs/; restyled figures in
#   its restyled_<date>/; attempt_effort writes cohort1_quiet_spikes/attempt_effort (script convention).
set -u
OUTNAME=${1:?usage: run_quiet_chain.sh <OUT_DIRNAME> [jobs]}
J=${2:-3}
W=$(cd "$(dirname "$0")/../.." && (pwd -W 2>/dev/null || pwd))   # Windows form (C:/...) for python on this box
cd "$W" || exit 2
if ! git diff --quiet || ! git diff --cached --quiet; then echo "refusing: $W has uncommitted changes"; exit 2; fi
CODE=$(git rev-parse --short HEAD)
PY=${PY:-C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe}
export FROZEN_LICKS_ALL=split
export PYTHONPATH="$W;$W/scripts"
export QS_CODE_DIR="$W/scripts/cohort2_qc"
export QS_LICK_SOURCE=daq+lp
export QS_V2_DIRNAME=$OUTNAME
export QS_ENGAGED_END=lastsync
export WFIELD_INPUT_CACHE=${WFIELD_INPUT_CACHE:-C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104}
export WIDEFIELD_FROZEN_MODEL_DIR=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/frozen_models
export WIDEFIELD_HEMO_VARIANT=meegkit_hpfit_lag_refitT_swapmask
export WIDEFIELD_LOCANMF_PROJECT=1
export WIDEFIELD_LP_FROZEN=lp
export WIDEFIELD_MAX_JOBS=4
export WIDEFIELD_PRE_FROM=0801
export WIDEFIELD_PROJECTION_CENTRE=wt
export WIDEFIELD_REBUILD_ROOT=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103
export WIDEFIELD_SESSION_CACHE=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/session_cache
export WIDEFIELD_SPOUT_SCORE_CENTRE=wt
export WIDEFIELD_TRIAL_END=lick
export WIDEFIELD_TRIAL_GATE=camera
export WIDEFIELD_TRIAL_STOPPED=both
export WIDEFIELD_TRIAL_WINDOW=cue_stop
export WIDEFIELD_WORKING_TRIALS=in_working
export PYTHONIOENCODING=utf-8
Q=$("$PY" -c "import sys; sys.path.insert(0, r'$W/scripts/cohort2_qc'); from quiet_area_spikes import OUT; print(OUT)")
LOGD="$Q/$OUTNAME/logs"

step() {   # step <log name> <script> [args...]   (env overrides via `VAR=... step ...`)
  local L=$1 S=$2; shift 2
  local LOG="$LOGD/$L.log" s; s=$(date +%s)
  { echo "# $S $* -- run_quiet_chain.sh $OUTNAME"; echo "# start $(date '+%F %T')  code $CODE"
    env | grep -E '^(FROZEN_|PYTHONPATH|QS_|WFIELD_|WIDEFIELD_)' | sort | sed 's/^/# env /'
    "$PY" -c "import wfield_local; print('# wfield_local', wfield_local.__file__)"; } > "$LOG"
  "$PY" "scripts/cohort2_qc/$S" "$@" >> "$LOG" 2>&1
  local rc=$?; echo "# end $(date '+%F %T') exit $rc" >> "$LOG"
  echo "$L exit $rc $(( $(date +%s) - s )) s"; return $rc
}

# ---- preflight: T13 p_true tables under THIS key (no cross-key reads, ever) ----
WIDEFIELD_LICK_SOURCE=daq "$PY" - <<'EOF' || exit 3
from wfield_local import attempt_coding as ac
miss = [str(ac.out_dir() / ac.trials_name("precue", a)) for a in ("PS92", "PS93", "PS94", "PS95")
        if not (ac.out_dir() / ac.trials_name("precue", a)).exists()]
if miss:
    raise SystemExit("PREFLIGHT: attempt_coding precue tables missing for this key -- run attempt_coding first:\n  "
                     + "\n  ".join(miss))
print("preflight: attempt_coding precue tables present for all four animals")
EOF
mkdir -p "$LOGD"
[ -f "$Q/engaged_end_20261118.csv" ] || step engaged_end quiet_engaged_end.py --jobs "$J" || exit 4

T0=$(date +%s)
# ---- base chain (order as the 11-18 sync-end run) ----
step quiet_v2 quiet_spikes_v2.py --jobs "$J" --k 4 || exit 5
step quiet_v2_fr quiet_spikes_v2_fr.py --jobs "$J" & P1=$!
( WIDEFIELD_LICK_SOURCE=daq step quiet_task_events_daq2s quiet_task_events.py --jobs "$J" ) & P2=$!
step quiet_hemo_check quiet_hemo_check.py --jobs "$J" & P3=$!
wait $P1 || exit 6
step quiet_eta_fr quiet_eta_fr.py --jobs "$J" & P4=$!
( step quiet_stillness quiet_stillness.py --jobs "$J" && step quiet_stillness_plot quiet_stillness.py --plot-only ) & P5=$!
step quiet_v2_examples quiet_spikes_v2_examples.py & P6=$!
step quiet_event_movie quiet_event_movie.py --pre PS92_0809 --post PS92_0820 --window intask & P7=$!
wait $P2 $P3 $P4 $P5 $P6 $P7
# ---- section-23 follow-ons (per-session extractors, then the ones reading quiet_fluct) ----
for a in "quiet_fluct.py" "quiet_eta_split.py" "quiet_quit_state.py" "quiet_breathing.py" "quiet_event_width.py" \
         "quiet_state_response.py" "attempt_effort.py" "quiet_steinmetz_check.py" "quiet_hemi_amp_recovery.py"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  step "${a%.py}" "$a" $( [ "$a" = quiet_steinmetz_check.py ] || [ "$a" = quiet_hemi_amp_recovery.py ] || echo --jobs "$J" ) &
done
wait
for a in quiet_sync_index.py quiet_sync_4to8.py quiet_recovery_link.py quiet_f0_noise.py quiet_lick_lfo.py; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  step "${a%.py}" "$a" $( [ "$a" = quiet_f0_noise.py ] || [ "$a" = quiet_lick_lfo.py ] && echo --jobs "$J" ) &
done
wait
# T13 follow-up: SAME key as T13 (DAQ 2-s window), stored decoders through load_or_fit only (no by-id switch)
WIDEFIELD_LICK_SOURCE=daq step quiet_task_events_followup quiet_task_events_followup.py --jobs "$J"
# ---- restyled deck / page figures (bootstrap stars) from this run's tables ----
QS_RESTYLE_BASE=$OUTNAME QS_RESTYLE_FOL=$OUTNAME step quiet_restyle_figs quiet_restyle_figs.py
echo "chain $OUTNAME done in $(( $(date +%s) - T0 )) s (code $CODE); logs $LOGD"
