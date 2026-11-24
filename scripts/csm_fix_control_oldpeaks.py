"""CSM edge-fix CONTROL (2026-11-21; usage: python scripts/csm_fix_control_oldpeaks.py <new control dir>):  for the CSM-fix comparison: the CURRENT within_lick_correction code (camera working period etc.) on the
OCT 30 per-lick pickles (old zero-padded peaks) with the Oct 30 threshold. Separates what the edge fix changes from what
code changes since Oct 30 change. Reads the old licks read-only; writes only into the control dir."""
import os
import shutil
import sys
from pathlib import Path

CTRL = Path(sys.argv[1])
OLD = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
           "within_lick_correction")
os.environ["WLC_OUT_DIR"] = str(CTRL)
sys.path.insert(0, str(Path.cwd() / "scripts"))
import within_lick_correction as W  # noqa: E402

assert W.OUT == CTRL
CTRL.mkdir(parents=True, exist_ok=False)
shutil.copy2(OLD / "csm_threshold.json", CTRL / "csm_threshold.json")
W.LICKS = OLD / "licks"
_orig = W.load_licks
W.load_licks = lambda sids, cols=None, allow_stale=False: _orig(sids, cols, allow_stale=True)
W.cmd_summarise()
