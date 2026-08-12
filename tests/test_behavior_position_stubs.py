"""The task-controller v47 (2026-08-10) stub rows must not reach the log<->DAQ aligner.

v47 fixed the old pos_idx overwriting (docs/GUI_TRIALS_LOGGING.md) by CLOSING the colliding row and
opening a fresh one -- but it keeps that stub in trials.csv: duplicate trial_id, sub-second, neither
hit nor miss. Left in, the log is LONGER than the DAQ cue stream, no integer trial-offset can align
them, and classify_cues_with_backup silently fails its >=0.9 validation -- disabling the dead-strobe
fallback exactly when a v47 session needs it.

Measured on real data (2026-08-12): filtering restores the log to the DAQ cue count EXACTLY
(PS94 8/10 1009->720, PS94 8/11 554->436) and best-offset agreement goes 21% -> 99%.
"""
from __future__ import annotations

from wfield_local.behavior_position import _scored_rows


def _row(tid, pos, hit=False, miss=False, start=0, end=8000):
    return {"trial_id": str(tid), "pos_idx": str(pos),
            "device_trial_start_ms": str(start), "device_trial_end_ms": str(end),
            "hit": "True" if hit else "False", "miss": "True" if miss else "False"}


def test_v47_stub_rows_are_dropped():
    """A stub (same trial_id, sub-second, unscored) is dropped; the real closed row survives."""
    rows = [
        _row(1, 3, hit=True, start=0, end=7000),
        _row(2, 5, start=7000, end=7600),            # v47 stub: unscored, 0.6 s, id reused below
        _row(2, 5, miss=True, start=7600, end=15000),
        _row(3, 1, hit=True, start=15000, end=22000),
    ]
    kept = _scored_rows(rows)
    assert len(kept) == 3, "exactly the stub should be removed"
    assert [r["pos_idx"] for r in kept] == ["3", "5", "1"]


def test_filter_is_a_noop_on_pre_v47_logs():
    """Pre-v47 sessions have no stubs (PS94 8/6 measured: 0), so the filter must not change them."""
    rows = [_row(i, i % 6, hit=(i % 2 == 0), miss=(i % 2 == 1)) for i in range(20)]
    assert len(_scored_rows(rows)) == 20


def test_never_closed_rows_still_dropped():
    rows = [_row(1, 2, hit=True), _row(2, 4, start=500, end=500), _row(3, 0, miss=True)]
    assert len(_scored_rows(rows)) == 2


def test_older_schema_without_hit_miss_is_left_alone():
    """Only the start/end filter applies when the CSV predates hit/miss columns -- never return []."""
    rows = [{"trial_id": "1", "pos_idx": "2", "device_trial_start_ms": "0", "device_trial_end_ms": "9"},
            {"trial_id": "2", "pos_idx": "3", "device_trial_start_ms": "9", "device_trial_end_ms": "9"}]
    kept = _scored_rows(rows)
    assert len(kept) == 1 and kept[0]["pos_idx"] == "2"


def test_truthiness_accepts_the_formats_the_controller_emits():
    for v in ("1", "True", "true", "TRUE", "yes", "y"):
        assert len(_scored_rows([_row(1, 0), {**_row(2, 1), "hit": v}])) == 1
    for v in ("0", "False", "false", "", "no"):
        assert len(_scored_rows([{**_row(1, 0), "hit": v}])) == 0
