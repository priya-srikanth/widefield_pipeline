"""quiet_task_events --summarise on the cam1-guard run, reading the PRODUCTION attempt_coding p_true tables.

DO NOT REUSE (11-18): a night-of patch that reads ANOTHER key's p_true tables (mixed builds; the cause of the Fig 23.23
mismatch). Re-runs use scripts/cohort2_qc/run_quiet_chain.sh, whose preflight requires attempt_coding tables under the
run's own key, and quiet_task_events now fails loudly when they are missing.

Why: the guard changes the camera-class signature (csig-3d7bbe921e -> csig-0f1b20ae2d), so trials_name() keys a file that
attempt_coding has not written yet. p_true here is the frozen pre-stroke decoder on the DAQ 2-s pre-cue window
(WIDEFIELD_LICK_SOURCE=daq), which no cam1 label enters; only the file key differs. Event flags / rates = guarded run.
Run with the env of run_downstream_k375.sh (task step); usage: python task_events_summarise_prodptrue.py"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("QS_CODE_DIR", "C:/Users/SabatiniLab/Github/wf_c1guard1117/scripts/cohort2_qc"))  # code worktree
sys.path.insert(0, str(W))
import quiet_task_events as q  # noqa: E402

PROD = "3d7bbe921e"
_orig = q.attempt_trials_name


def _prod(align, animal):
    n = _orig(align, animal)
    return n[:n.index("_csig-") + 6] + PROD + ".gz"


q.attempt_trials_name = _prod
print("# p_true tables read as:", _prod("precue", "PS92"))
sys.argv = [sys.argv[0], "--summarise"]
q.main()
