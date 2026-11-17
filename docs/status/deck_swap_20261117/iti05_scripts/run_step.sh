#!/usr/bin/env bash
# usage: run_step.sh <step> <out_folder_name> <dock:0|1> <python args...>
# Replays the env of the rebuild steps l_cued_ramp_g / l_lateral_ratio_g (render_daqlk logs; identical env blocks) and of
# the 11-17 docked runs (+ WFIELD_ITI_FILTER=dock); writes into iti_licks/<out_folder_name>/<tag>/ (WFIELD_ITI_OUT_W).
step=$1; folder=$2; dock=$3; shift 3
I=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks
export COMBINED_ENCODING_EXEC3D_TABLE=C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/pose3d_kin_licks_snapshot_20261023.csv.gz
export COMBINED_ENCODING_FR=master_pre_v1
export COMBINED_ENCODING_FR_GROUPS=master_v1:A
export COMBINED_ENCODING_LICKS=combined
export COMBINED_ENCODING_ONLY=full
export COMBINED_ENCODING_SUFFIX=_gate-attempted
export COMBINED_ENCODING_TONGUE=3d
export FROZEN_FR=1
export FROZEN_LICKSPOUT=1
export FROZEN_LICKS_ALL=split
export OROFACIAL_GATE=attempted
export PYTHONPATH="C:\\Users\\SabatiniLab\\Github\\wf_cleanup2;C:\\Users\\SabatiniLab\\Github\\wf_cleanup2/scripts"
export WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104
export WIDEFIELD_FROZEN_MODEL_DIR=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/frozen_models
export WIDEFIELD_HEMO_VARIANT=meegkit_hpfit_lag_refitT_swapmask
export WIDEFIELD_LOCANMF_PROJECT=1
export WIDEFIELD_LP_FROZEN=lp
export WIDEFIELD_MAX_JOBS=3
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
if [ "$folder" != "-" ]; then export WFIELD_ITI_OUT_W=$I/$folder; fi
if [ "$dock" = "1" ]; then export WFIELD_ITI_FILTER=dock; fi
[ -n "$EXTRA_ENV" ] && eval "export $EXTRA_ENV"
LOGD=$I/${LOGFOLDER:-$folder}/logs
mkdir -p "$LOGD"
LOG=$LOGD/$step.log
cd C:/Users/SabatiniLab/Github/wf_cleanup2
sha=$(git rev-parse --short HEAD); dirty=$(git diff --quiet -- scripts || echo " +dirty (ITI window 11-17 edits)")
{
  echo "# step $step  (ITI 0..0.5 s re-render, Priya 11-17; env replayed from render_daqlk l_cued_ramp_g / l_lateral_ratio_g${dock:+, dock=$dock})"
  echo "# start $(date '+%F %T')  code wf_cleanup2 $sha$dirty"
  echo "# cmd: C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe $*"
  env | grep -E '^(COMBINED_|FROZEN_|OROFACIAL_|PYTHONPATH|WFIELD_|WIDEFIELD_)' | sort | sed 's/^/# env /'
} > "$LOG"
C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe "$@" >> "$LOG" 2>&1
rc=$?
echo "# end $(date '+%F %T') exit $rc" >> "$LOG"
exit $rc
