"""Behavioural-state segments: the spout-position-agnostic SPECIFICITY CONTROL.

Priya, 2026-09-11: "we don't need position information for this at all -- spout position agnostic.
I'm more looking for evidence that not *all* decoding/encoding degrades post-stroke, with running as
an example", and then "should we try quiet vs running vs licking (all positions) frozen decoder?"

The lesion is ventrolateral STRIATAL, so no cortex is damaged. If the same undamaged cortex, imaged
through the same window on the same basis, still reports behavioural state normally while its report
of lick target has changed, then the target deficit is specific -- and the generic explanations
(window clouding, haemodynamic drift, arousal, basis drift) all fail at once, because every one of
them would degrade this readout too.

These tests pin the two things that make the unit legitimate: the MUTUAL EXCLUSIVITY Priya
specified, and the tiling that turns 1,795 running trials into 33,060 usable segments without
letting one long period dominate.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import locomotor_state as ls

FS = 10.0


def _ev(**kw):
    base = {"fs": FS, "running_starts": [], "running_stops": [],
            "quiet_starts": [], "quiet_stops": [], "lick_onsets": []}
    base.update(kw)
    return base


# ------------------------------------------------------------------ tiling

def test_a_five_second_bout_yields_five_one_second_segments():
    s, p = ls.segments([0], [50], FS, seg_s=1.0)
    assert len(s) == 5
    assert s.tolist() == [0, 10, 20, 30, 40]
    assert set(p.tolist()) == {0}, "every segment must carry its parent period's id"


def test_the_remainder_is_dropped_not_stretched():
    """A 0.4 s window binned into four is not the same feature as a 1 s one."""
    s, _p = ls.segments([0], [14], FS, seg_s=1.0)
    assert len(s) == 1


def test_a_period_shorter_than_the_window_contributes_nothing():
    """Quiet periods have a 0.5 s floor and a 1.10 s median; 42% of them are shorter than 1 s."""
    s, _p = ls.segments([0], [9], FS, seg_s=1.0)
    assert len(s) == 0


def test_the_cap_bounds_any_single_period():
    """The top 1% of QUIET periods supply 36.6% of all segments uncapped, and the top 5% supply 70%.

    Capping does NOT make the segments independent -- only clustering the bootstrap by period does
    that -- but it stops one 370 s stretch of an inactive animal being most of the class.
    """
    s, _p = ls.segments([0], [10_000], FS, seg_s=1.0, cap=8)
    assert len(s) == 8


def test_period_ids_never_collide_across_classes():
    """A caller grouping by period must not merge a running bout with a quiet period."""
    ev = _ev(running_starts=[0], running_stops=[50], quiet_starts=[100], quiet_stops=[140])
    _s, lab, per = ls.running_and_quiet_segments(ev)
    for a in set(per[lab == "running"].tolist()):
        assert a not in set(per[lab == "quiet"].tolist())


# ------------------------------------------------------------------ exclusivity

def test_a_running_segment_overlapping_a_lick_bout_is_dropped():
    """Priya's rule: "running only if not also licking, licking only if not also running"."""
    # licks at 2.0-2.4 s sit inside the running bout's third segment.
    ev = _ev(running_starts=[0], running_stops=[50],
             lick_onsets=[20, 22, 24], quiet_starts=[], quiet_stops=[])
    _s, lab, _p, n_drop = ls.three_way_segments(ev)
    assert n_drop >= 1, "the overlapping segment must be dropped, not assigned"
    # and the non-overlapping parts of the same bout survive
    assert (lab == "running").sum() >= 1


def test_the_three_classes_are_disjoint_in_time():
    rng = np.random.default_rng(0)
    ev = _ev(running_starts=[0, 200], running_stops=[50, 260],
             quiet_starts=[400], quiet_stops=[450],
             lick_onsets=sorted(rng.integers(600, 660, 12).tolist()))
    smp, lab, _p, _d = ls.three_way_segments(ev)
    n = round(ls.SEGMENT_S * FS)
    for i in range(len(smp)):
        for j in range(i + 1, len(smp)):
            if lab[i] == lab[j]:
                continue
            a0, a1 = smp[i], smp[i] + n
            b0, b1 = smp[j], smp[j] + n
            assert not (a0 < b1 and b0 < a1), f"{lab[i]} and {lab[j]} windows overlap"


def test_quiet_windows_must_be_CONTAINED_not_merely_overlapping():
    """Quiet asserts an ABSENCE, so a window half outside a quiet period is not quiet.

    `classify` uses containment for quiet and overlap for the two presence classes; this pins that
    asymmetry, which is easy to "simplify" away.
    """
    ev = _ev(quiet_starts=[100], quiet_stops=[120])
    lab, _c = ls.classify([95], (0, 10), ev)          # 95-105 straddles the start
    assert lab[0] is None
    lab, _c = ls.classify([105], (0, 10), ev)         # 105-115 is inside
    assert lab[0] == "quiet"


def test_lick_and_run_together_is_counted_not_absorbed():
    ev = _ev(running_starts=[0], running_stops=[100], lick_onsets=[5])
    _lab, counts = ls.classify([0], (0, 10), ev)
    assert counts[ls.DROPPED] == 1
    assert counts["running"] == 0 and counts["licking"] == 0


# ------------------------------------------------------------------ the exclusion

def test_the_crash_and_concat_session_is_excluded_by_name():
    """PS92 8/12's longest 'running bout' is 2,441 s -- 41 minutes, 29% of the session -- against a
    cohort maximum of 54 s everywhere else. That is the `concat_split_session` discontinuity read as
    sustained locomotion (`docs/EXPERIMENT_ERRORS.md`), and uncapped it would supply ~7% of the
    entire running class from one artefact."""
    assert "PS92_0812" in ls.EXCLUDE_SESSIONS


@pytest.mark.parametrize("name", list(ls.THREE_WAY))
def test_every_named_class_is_reachable(name):
    ev = _ev(running_starts=[0], running_stops=[50],
             quiet_starts=[200], quiet_stops=[250],
             lick_onsets=[400, 403, 406, 409])
    _s, lab, _p, _d = ls.three_way_segments(ev)
    assert name in set(lab.tolist()), f"{name} produced no segment from a period built for it"


def test_a_short_lick_bout_still_contributes_its_ONSET_window():
    """Lick bouts have a median of 0.37 s. Tiling strictly inside them keeps 22% of 101,018 bouts
    and biases the class toward sustained licking -- the same failure a 2 s window inflicts on
    quiet. Licking is an EVENT with a defined onset, so its window is anchored there and may run
    past the bout's end."""
    ev = _ev(lick_onsets=[400, 402, 404])                 # a 0.4 s bout at fs=10
    _s, lab, _p, _d = ls.three_way_segments(ev)
    assert (lab == "licking").sum() == 1


def test_quiet_is_NOT_onset_anchored():
    """A quiet window running past its period's end would spend part of itself in the movement or
    licking the period was buffered away from -- the one thing the class must not contain."""
    ev = _ev(quiet_starts=[0], quiet_stops=[6])           # 0.6 s, shorter than the window
    _s, lab, _p, _d = ls.three_way_segments(ev)
    assert (lab == "quiet").sum() == 0
