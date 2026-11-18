#!/usr/bin/env bash
# Quiet-period events v2, cam1-guard per-session data (v2_quiet_lp_c1guard_20261117/sessions), SUMMARY at common k = 4
# (Priya 11-18 via 3c: goal Fig 23.18 threshold figure should show the decided k 4 gate). Env = run_quiet_v2_k375.sh;
# only QS_V2_DIRNAME and --k differ. Writes a new folder; the k 3.75 production folder is untouched.
W=C:/Users/SabatiniLab/Github/wf_c1guard1117
Q=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/cohort1_quiet_spikes
export FROZEN_LICKS_ALL=split
export PYTHONPATH="C:\Users\SabatiniLab\Github\wf_c1guard1117;C:\Users\SabatiniLab\Github\wf_c1guard1117/scripts"
export QS_LICK_SOURCE=daq+lp
export QS_V2_DIRNAME=v2_quiet_lp_c1guard_k4_20261118
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
LOGD=$Q/$QS_V2_DIRNAME/logs; mkdir -p "$LOGD"; LOG=$LOGD/quiet_v2.log
cd $W
{
  echo "# quiet_v2 cam1-guard re-run (Priya 11-17); env replayed from v2_quiet_lp_20261117/logs/quiet_v2.log"
  echo "# start $(date '+%F %T')  code wf_c1guard1117 $(git rev-parse --short HEAD)$(git diff --quiet || echo ' +dirty')"
  echo "# cmd: python scripts/cohort2_qc/quiet_spikes_v2.py --summarise --k 4 (common k 4 = the decided gate; Priya 11-18 via 3c, for goal Fig 23.18)"
  env | grep -E '^(FROZEN_|PYTHONPATH|QS_|WFIELD_|WIDEFIELD_)' | sort | sed 's/^/# env /'
  C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe -c "import wfield_local; print('# wfield_local', wfield_local.__file__)"
} > "$LOG"
C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe scripts/cohort2_qc/quiet_spikes_v2.py --summarise --k 4 >> "$LOG" 2>&1
rc=$?; echo "# end $(date '+%F %T') exit $rc" >> "$LOG"; exit $rc
