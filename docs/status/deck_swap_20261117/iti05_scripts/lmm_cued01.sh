#!/usr/bin/env bash
# LMM companion of slide 114 (Fig 14.7) on the cued per-spout window CUED_RESP_WIN 0..1.0 s (Priya 11-17).
G0=M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/lmm_hidden_20261117
source $G0/logs/env_daqlk.sh
O=$G0/cued01_20261117
mkdir -p $O/logs
export PYTHONPATH=C:/Users/SabatiniLab/Github/wf_lmm_main
export LMM_HIDDEN_OUTDIR=$O
export LMM_HIDDEN_NAME_SUFFIX=_cued01
export LMM_HIDDEN_SESS_DIR=$IE/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801/sessions_roi8fr
cd C:/Users/SabatiniLab/Github/wf_lmm_main
LOG=$O/logs/cued_ramp_perspout.log
sha=$(git rev-parse --short HEAD); dirty=$(git diff --quiet -- scripts || echo " +dirty (cued per-spout window 11-17 edits)")
{
  echo "# step cued_ramp_perspout  start $(date '+%F %T')  code wf_lmm_main $sha$dirty (branch lmm-hidden-20261103)  cued per-spout AUC 0..1.0 s (Priya 11-17)"
  echo "# cmd: $PY -m scripts.lmm_hidden.lick_cued_ramp_perspout --root $ITI --outdir $O"
  env | grep -E '^(FROZEN_|WFIELD_|WIDEFIELD_|LMM_HIDDEN|PYTHONPATH)' | sort | sed 's/^/# env /'
} > $LOG
$PY -m scripts.lmm_hidden.lick_cued_ramp_perspout --root $ITI --outdir $O >> $LOG 2>&1
echo "# end $(date '+%F %T') exit $?" >> $LOG
$PY -c "
import sys; sys.path.insert(0, 'C:/Users/SabatiniLab/Github/wf_lmm_main')
from pathlib import Path
from scripts.lmm_hidden.crop import drop_title
o = Path(r'$O'); drop_title(o / 'lick_cued_ramp_perspout_cued01.png', o / 'lick_cued_ramp_perspout_cued01_notitle.png'); print('notitle ok')
" >> $LOG 2>&1
echo LMM CUED01 DONE
