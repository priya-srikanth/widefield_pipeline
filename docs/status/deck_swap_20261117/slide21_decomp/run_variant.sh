#!/usr/bin/env bash
# Slide-21 one-factor decomposition: run the cue/working arm (families acc 5r 13s) under the 11-09 rebuild env
# (g5_epoch_g, code 8aed3a7b) with ONE factor reverted. Usage: run_variant.sh <variant> [--dry]
# Isolation: outputs -> $D/<v>/grant_figures/<tag>/epoch ; session cache -> $D/<v>/<tag>/session_cache ;
# frozen store -> $D/frozen_models_<v> ; the write jail (jail/sitecustomize.py) refuses any share write outside $D.
set -u
V=$1; DRY=${2:-}
S=C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/782a4a7a-7855-40ed-b449-494e8aa4b6bc/scratchpad
D=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/decomp_slide21_20261117
PY=C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe
CODE=$S/code_8aed3a7b
[ "$V" = "C_lick" ] && CODE=$S/code_8aed3a7b_oldlick

# ---- the 11-09 rebuild env (g5_epoch_g.log head), verbatim
export FROZEN_LICKS_ALL=split
export WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104
export WIDEFIELD_HEMO_VARIANT=meegkit_hpfit_lag_refitT_swapmask
export WIDEFIELD_LOCANMF_PROJECT=1
export WIDEFIELD_LP_FROZEN=lp
export WIDEFIELD_PRE_FROM=0801
export WIDEFIELD_PROJECTION_CENTRE=wt
export WIDEFIELD_REBUILD_ROOT=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103
export WIDEFIELD_SPOUT_SCORE_CENTRE=wt
export WIDEFIELD_TRIAL_END=lick
export WIDEFIELD_TRIAL_GATE=camera
export WIDEFIELD_TRIAL_STOPPED=both
export WIDEFIELD_TRIAL_WINDOW=cue_stop
export WIDEFIELD_WORKING_TRIALS=in_working

# ---- the ONE reverted factor
case "$V" in
  R0) ;;                                                     # control: must reproduce the rebuild sidecars
  A_pre) unset WIDEFIELD_PRE_FROM ;;                         # June + August pre (11 / animal)
  B_gate) unset WIDEFIELD_TRIAL_GATE WIDEFIELD_TRIAL_WINDOW WIDEFIELD_TRIAL_END WIDEFIELD_TRIAL_STOPPED \
                WIDEFIELD_WORKING_TRIALS ;;                  # DAQ quit-tail gate, working = lick + miss-while-working
  C_lick) ;;                                                 # old imaging lick detector (code copy patched)
  D_scale) export WIDEFIELD_SPOUT_SCALING=session_z ;;       # pre-11-03 engaged-trial z-score (no bal_em / Wt)
  E_hemo) export WIDEFIELD_HEMO_VARIANT=meegkit_hpfit ;;     # the pre-rebuild hemo variant (projection kept;
                                                             # joint basis: in-fit sessions get their FITTED C again)
  E2_hemo_projall) export WIDEFIELD_HEMO_VARIANT=meegkit_hpfit WIDEFIELD_JOINT_PROJECT_ALL=1 ;;  # old hemo, all projected
  F_lnmf) unset WIDEFIELD_LOCANMF_PROJECT ;;                 # per-session LocaNMF = stored C (not projected)
  *) echo "unknown variant $V"; exit 2 ;;
esac

export PYTHONPATH="$CODE;$S/jail"
export PYTHONIOENCODING=utf-8
export SLIDE21_JAIL=$D
export WIDEFIELD_MAP_JOBS=2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WIDEFIELD_FROZEN_MODEL_DIR=$D/frozen_models_$V
TAG=$(cd "$CODE" && "$PY" -c "from wfield_local import lp_licks; print(lp_licks.output_tag() or '')")
export WIDEFIELD_SESSION_CACHE=$D/$V/$TAG/session_cache
OUT=$D/$V/grant_figures/epoch
LOG=$S/runs/$V.log
mkdir -p "$S/runs" "$WIDEFIELD_SESSION_CACHE" "$WIDEFIELD_FROZEN_MODEL_DIR"
{
  echo "# variant $V  start $(date '+%F %T')  code $(basename $CODE)  tag $TAG"
  echo "# free RAM GB: $(powershell -NoProfile -c '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)')"
  env | grep -E '^(WIDEFIELD_|WFIELD_|FROZEN_|SLIDE21|PYTHONPATH)' | sort | sed 's/^/# env /'
  echo "# cmd: python -m wfield_local.epoch_grant_figures --arm cue --only acc 5r 13s --output $OUT"
} > "$LOG"
[ -n "$DRY" ] && { cat "$LOG"; exit 0; }
cd "$CODE" && "$PY" -m wfield_local.epoch_grant_figures --arm cue --only acc 5r 13s --output "$OUT" >> "$LOG" 2>&1 &
PID=$!
echo "# bash pid $PID winpid $(cat /proc/$PID/winpid 2>/dev/null)" >> "$LOG"
wait $PID
RC=$?
echo "# end $(date '+%F %T') exit $RC" >> "$LOG"
exit $RC
