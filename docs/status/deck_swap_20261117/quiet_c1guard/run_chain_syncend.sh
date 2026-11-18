#!/usr/bin/env bash
# Quiet-event chain re-run with the analyzed time ending at the LAST SYNC PULSE (Priya 11-18, via 3c): engaged = before
# min(quit, last sync edge) (QS_ENGAGED_END=lastsync, quiet_engaged_end.py table). Supersedes run_chain_engend.sh (last
# cue + response window). Same env as the 11-17 deck pass otherwise; code = wf_qstyle1118 (branch
# fix/quiet-engaged-end-1118); output folder v2_quiet_lp_c1guard_syncend_20261118; base summary k 4.
# usage: run_chain_syncend.sh <v2|fr|eta|hemo|still|stillplot|task|tasksumm|examples|movie> [jobs]
W=C:/Users/SabatiniLab/Github/wf_qstyle1118
Q=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/cohort1_quiet_spikes
J=${2:-3}
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
case $1 in
  v2)       S=quiet_spikes_v2.py; A="--jobs $J --k 4"; L=quiet_v2 ;;
  examples) S=quiet_spikes_v2_examples.py; A=""; L=quiet_v2_examples ;;
  fr)       S=quiet_spikes_v2_fr.py; A="--jobs $J"; L=quiet_v2_fr ;;
  eta)      S=quiet_eta_fr.py; A="--jobs $J"; L=quiet_eta_fr ;;
  task)     S=quiet_task_events.py; A="--jobs $J"; L=quiet_task_events_daq2s; export WIDEFIELD_LICK_SOURCE=daq ;;
  # summary with the PRODUCTION p_true tables: the guard re-keys the csig and attempt_coding has no guard-keyed tables
  tasksumm) S=../../../wf_cleanup2/docs/status/deck_swap_20261117/quiet_c1guard/task_events_summarise_prodptrue.py; A=""
            L=quiet_task_events_summ_prodptrue; export WIDEFIELD_LICK_SOURCE=daq ;;
  # slides 154 / 157 (Priya 11-17 "re run both"; were on the pre-rebuild v2_quiet_lp)
  hemo)     S=quiet_hemo_check.py; A="--jobs $J"; L=quiet_hemo_check ;;
  still)    S=quiet_stillness.py; A="--jobs $J"; L=quiet_stillness ;;          # needs fr_regressed_own/sessions
  stillplot) S=quiet_stillness.py; A="--plot-only"; L=quiet_stillness_plot ;;  # the deck's _nolines redraw
  movie)    S=quiet_event_movie.py; A="--pre PS92_0809 --post PS92_0820 --window intask"; L=quiet_event_movie ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
LOGD=$Q/$QS_V2_DIRNAME/logs; mkdir -p "$LOGD"; LOG=$LOGD/$L.log
cd $W
{
  echo "# $1 on the cam1-guard run with the analyzed time ending at the last sync pulse (Priya 11-18); env = the 11-17 deck pass + QS_ENGAGED_END=lastsync"
  echo "# start $(date '+%F %T')  code wf_qstyle1118 $(git rev-parse --short HEAD)$(git diff --quiet || echo ' +dirty')"
  echo "# cmd: python scripts/cohort2_qc/$S $A"
  env | grep -E '^(FROZEN_|PYTHONPATH|QS_|WFIELD_|WIDEFIELD_)' | sort | sed 's/^/# env /'
  C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe -c "import wfield_local; print('# wfield_local', wfield_local.__file__)"
} > "$LOG"
C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe scripts/cohort2_qc/$S $A >> "$LOG" 2>&1
rc=$?; echo "# end $(date '+%F %T') exit $rc" >> "$LOG"; exit $rc
