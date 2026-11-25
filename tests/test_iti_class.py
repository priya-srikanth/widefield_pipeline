"""iti_dock_audit.iti_class / bout_starts: the docked definition of the 11-24 rebuild (Priya): docked = bouts STARTING at
or after dock arrival; a bout started during retraction is its own class (incl. its continuation into the docked
window); everything else stays "other"."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import iti_dock_audit as da  # noqa: E402

STOP, DOCK, TS = 10.0, 10.5, 12.0


def _cls(on_all, licks):
    bs = da.bout_starts(on_all, licks, gap=1.0)
    n = len(licks)
    return list(da.iti_class(licks, bs, np.full(n, STOP), np.full(n, DOCK), np.full(n, TS))), bs


def test_bout_started_during_retraction_is_excluded_from_docked():
    on = [5.0, 10.2, 10.4, 10.6, 10.8]          # bout starts at 10.2 (gap 5.2 s), during retraction
    cls, bs = _cls(on, np.array([10.2, 10.4, 10.6, 10.8]))
    assert np.allclose(bs, 10.2)
    assert cls == ["retraction"] * 4             # incl. the licks continuing into the docked window


def test_bout_whose_first_lick_lands_at_arrival_is_docked():
    on = [5.0, DOCK, 10.7, 10.9]
    cls, _ = _cls(on, np.array([DOCK, 10.7, 10.9]))
    assert cls == ["docked"] * 3


def test_bout_in_progress_from_the_trial_is_other():
    on = [9.5, 9.9, 10.3, 10.6]                  # started before the stop (response bout), runs into the dock
    cls, bs = _cls(on, np.array([10.3, 10.6]))
    assert np.allclose(bs, 9.5) and cls == ["other", "other"]


def test_after_trial_start_is_not_docked():
    on = [5.0, 11.5, 12.1]                       # docked bout start, one lick after the next trial_start
    cls, _ = _cls(on, np.array([11.5, 12.1]))
    assert cls == ["docked", "other"]


def test_lick_not_in_train_gives_nan_bout_start():
    bs = da.bout_starts([1.0, 3.0], np.array([2.0]), gap=1.0)
    assert np.isnan(bs[0])


def test_after_quiet_needs_docked_and_two_seconds():
    cls = np.array(["docked", "docked", "retraction", "other"])
    gap = np.array([2.0, 1.99, 5.0, 5.0])
    assert list(da.after_quiet(cls, gap, quiet=2.0)) == [True, False, False, False]
    assert not da.after_quiet(np.array(["docked"]), np.array([np.nan]), quiet=2.0)[0]
    assert list(da.after_quiet(cls, gap, quiet=1.0)) == [True, True, False, False]   # the 1 s (bout-start) flag
