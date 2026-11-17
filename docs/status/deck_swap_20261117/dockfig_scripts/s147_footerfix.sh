#!/usr/bin/env bash
# Slide 147 footer fix (Priya 11-17): re-render frozen_claims_trend.png with the trial-set footer margin reserved
# (trial_set_text.footer_bottom). No figure data existed, so --trend-only recomputes section C (no A / B bootstraps) and
# saves figdata/frozen_claims_trend.pkl; output in a NEW folder, the production render is untouched.
# Env = the '# env' block of render_daqlk/logs/z_frozen_claims_g.log (rebuild step), + PYTHONPATH=<worktree>;<worktree>/scripts.
E=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
OUT=$E/frozen_fr_compare_footerfix_20261117
export COMBINED_ENCODING_EXEC3D_TABLE=C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/pose3d_kin_licks_snapshot_20261023.csv.gz
export COMBINED_ENCODING_FR=master_pre_v1
export COMBINED_ENCODING_FR_GROUPS=master_v1:A
export COMBINED_ENCODING_LICKS=combined
export COMBINED_ENCODING_ONLY=full
export COMBINED_ENCODING_SUFFIX=_gate-attempted
export COMBINED_ENCODING_TONGUE=3d
export FROZEN_FR=1
export FROZEN_LICKS_ALL=split
export OROFACIAL_GATE=attempted
export WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104
export WIDEFIELD_FROZEN_MODEL_DIR=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/frozen_models
export WIDEFIELD_HEMO_VARIANT=meegkit_hpfit_lag_refitT_swapmask
export WIDEFIELD_LOCANMF_PROJECT=1
export WIDEFIELD_LP_FROZEN=lp
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
export WIDEFIELD_MAX_JOBS=3
export PYTHONPATH="C:\\Users\\SabatiniLab\\Github\\wf_cleanup2;C:\\Users\\SabatiniLab\\Github\\wf_cleanup2/scripts"
export PYTHONIOENCODING=utf-8
unset WFIELD_ITI_FILTER WFIELD_ITI_OUT_W
mkdir -p "$OUT/logs"
LOG=$OUT/logs/trend_${1:-trend-only}.log
cd C:/Users/SabatiniLab/Github/wf_cleanup2
{
  echo "# s147 footer fix  (env replayed from render_daqlk z_frozen_claims_g)"
  echo "# start $(date '+%F %T')  code wf_cleanup2 $(git rev-parse --short HEAD)$(git diff --quiet -- scripts || echo ' +dirty')"
  echo "# cmd: scripts/frozen_claims.py --out $OUT --${1:-trend-only}"
  env | grep -E '^(COMBINED_|FROZEN_|OROFACIAL_|PYTHONPATH|WFIELD_|WIDEFIELD_)' | sort | sed 's/^/# env /'
} > "$LOG"
C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe scripts/frozen_claims.py --out "$OUT" --${1:-trend-only} >> "$LOG" 2>&1
rc=$?
echo "# end $(date '+%F %T') exit $rc" >> "$LOG"
exit $rc
