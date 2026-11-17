#!/usr/bin/env bash
# Batch 3: LMM hidden companions on the ITI 0..0.5 s window (wf_lmm_main, branch lmm-hidden-20261103). Waits for batches
# 1b and 2 (inputs: itidock_iti05 tables, lateral_ratio_dock_iti05 tables, slide95_sessz_iti05 LMM inputs).
H=C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/782a4a7a-7855-40ed-b449-494e8aa4b6bc/scratchpad/iti05
until grep -q "BATCH1B DONE" $H/batch1b.out 2>/dev/null && grep -q "BATCH2 DONE" $H/batch2.out 2>/dev/null; do sleep 20; done
G0=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/lmm_hidden_20261117
source $G0/logs/env_daqlk.sh
I=$IE/iti_licks
O=$G0/iti05_20261117
mkdir -p $O/logs
export PYTHONPATH=C:/Users/SabatiniLab/Github/wf_lmm_main
export LMM_HIDDEN_OUTDIR=$O
export LMM_HIDDEN_NAME_SUFFIX=_iti05
cd C:/Users/SabatiniLab/Github/wf_lmm_main
step() {  # step <name> <python args...> (extra env via EXTRA)
  n=$1; shift
  LOG=$O/logs/$n.log
  sha=$(git rev-parse --short HEAD); dirty=$(git diff --quiet -- scripts || echo " +dirty (ITI window 11-17 edits)")
  {
    echo "# step $n  start $(date '+%F %T')  code wf_lmm_main $sha$dirty (branch lmm-hidden-20261103)  ITI 0..0.5 s (Priya 11-17)"
    echo "# cmd: $PY $*"
    env | grep -E '^(FROZEN_|WFIELD_|WIDEFIELD_|LMM_HIDDEN|PYTHONPATH)' | sort | sed 's/^/# env /'
  } > $LOG
  $PY "$@" >> $LOG 2>&1
  echo "# end $(date '+%F %T') exit $?" >> $LOG
}
( export LMM_HIDDEN_ITI_VARIANT=$I/itidock_iti05_20261117/$TAG
  export LMM_HIDDEN_SESS_DIR=$IE/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801/sessions_roi8fr
  step iti_ramp_pooled_docked -m scripts.lmm_hidden.curve_iti_ramp_row1 --root $ITI --outdir $O ) &
( export LMM_HIDDEN_ITI_VARIANT=.
  step lateral_ratio_docked -m scripts.lmm_hidden.lick_lateral_ratio_auc_docked --root $I/lateral_ratio_dock_iti05_20261117/$TAG --outdir $O ) &
( step sessz_iti -m scripts.lmm_hidden.render $I/slide95_sessz_iti05_20261117/$TAG/lmm_iti_dock_sessz_sessions.csv \
    --outdir $O --name iti_ramp_row1_lmm_w0-0.5_sessz \
    --title "Docked-spout ITI bout-start licks (lick-response slide, ITI row), spouts pooled, SESSION-Z (SD_RISE variant c: / the session's SD over the same lick windows), response AUC 0..0.5 s / session SD / the animal's pre" \
    --ylabel "(AUC / session SD) / pre,
change vs pre" --cells ALM_right,ALM_left,tjM1_right,tjM1_left,SSp-m_right,SSp-m_left,cortex \
    --labels "ALM_right=ALM
contra,ALM_left=ALM
ipsi,tjM1_right=tjM1
contra,tjM1_left=tjM1
ipsi,SSp-m_right=SSp-m
contra,SSp-m_left=SSp-m
ipsi,cortex=cortex" ) &
wait
step splits $H/lmm_splits.py
echo BATCH3 DONE
