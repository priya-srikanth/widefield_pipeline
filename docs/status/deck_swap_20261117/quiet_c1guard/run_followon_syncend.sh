#!/usr/bin/env bash
# Movement page section-23 figures still on the PRE-REBUILD runs (23.3, 23.7-10, 23.12-14, 23.16, 23.19, 23.20, 23.22-24),
# re-rendered on the sync-end run (Priya 11-18 "ok"; 3c swaps them, then the pre-rebuild v2_quiet / v2_quiet_lp can go).
# Env = run_chain_syncend.sh (QS_V2_DIRNAME=v2_quiet_lp_c1guard_syncend_20261118, QS_ENGAGED_END=lastsync); code wf_qstyle1118
# (quit-time cache, verified 11-18). usage: run_followon_syncend.sh <script.py> [args...]   (log = <run>/logs/<script>.log)
W=C:/Users/SabatiniLab/Github/wf_qstyle1118
Q=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/cohort1_quiet_spikes
export FROZEN_LICKS_ALL=split
export PYTHONPATH="C:\Users\SabatiniLab\Github\wf_qstyle1118;C:\Users\SabatiniLab\Github\wf_qstyle1118/scripts"
export QS_CODE_DIR=C:/Users/SabatiniLab/Github/wf_qstyle1118/scripts/cohort2_qc   # read by task_events_summarise_prodptrue.py
export QS_LICK_SOURCE=daq+lp
export QS_V2_DIRNAME=v2_quiet_lp_c1guard_syncend_20261118
export QS_ENGAGED_END=lastsync
export WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104
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
S=$1; shift; L=${S%.py}${QS_LOG_SUFFIX:-}   # QS_LOG_SUFFIX: keep a re-plot from overwriting the extraction log
LOGD=$Q/$QS_V2_DIRNAME/logs; mkdir -p "$LOGD"; LOG=$LOGD/$L.log
cd $W
{
  echo "# $S on the sync-end run (page section-23 follow-ons, Priya 11-18); env = run_chain_syncend.sh"
  echo "# start $(date '+%F %T')  code wf_qstyle1118 $(git rev-parse --short HEAD)$(git diff --quiet || echo ' +dirty')"
  echo "# cmd: python scripts/cohort2_qc/$S $*"
  env | grep -E '^(FROZEN_|PYTHONPATH|QS_|WFIELD_|WIDEFIELD_)' | sort | sed 's/^/# env /'
} > "$LOG"
s=$(date +%s)
C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe scripts/cohort2_qc/$S "$@" >> "$LOG" 2>&1
rc=$?; echo "# end $(date '+%F %T') exit $rc" >> "$LOG"; echo "$S exit $rc $(( $(date +%s) - s )) s"; exit $rc
