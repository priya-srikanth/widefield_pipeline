"""iti_retraction_bouts.trial_flags: a retraction bout = a gap-based bout START in [stop, dock); docked = in [dock, ts)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import iti_retraction_bouts as rb  # noqa: E402


def test_trial_flags():
    C = pd.DataFrame({"stop": [10.0, 20.0, 30.0], "dock": [10.5, 20.5, float("nan")], "trial_start": [12.0, 22.0, 32.0]})
    on = [9.0, 9.5, 10.2, 10.3,        # trial 1: bout from 9.0 runs on; a NEW bout at 10.2? no (gap 0.7) -> no retraction
          18.0, 20.1, 20.2, 21.0,      # trial 2: bout start 20.1 (gap 2.1) in retraction; 21.0 continues (no docked start)
          28.0, 30.2]                  # trial 3: no dock time -> no flag
    F = rb.trial_flags(on, C, gap=1.0)
    assert list(F.retraction_bout) == [False, True, False]
    assert list(F.docked_bout) == [False, False, False]
    assert list(F.cycle_ok) == [True, True, False]
    F2 = rb.trial_flags([5.0, 10.5, 10.6], C.iloc[:1], gap=1.0)    # bout starting exactly at arrival = docked
    assert bool(F2.docked_bout.iloc[0]) and not bool(F2.retraction_bout.iloc[0])
