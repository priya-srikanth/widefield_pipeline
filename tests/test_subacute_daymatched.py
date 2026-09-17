"""Guards for the day-matched subacute diagnostic.

The scientific claim this file protects is narrow and easy to break by editing: that re-binning by
POST-STROKE DAY selects the sessions it says it does, and that the statistic is the same Pearson r
the epoch-binned `rest_null` reports -- otherwise the two tables are not comparable and the whole
comparison is meaningless.

It also records WHY this module is absent from `test_rest_engagement.py::EXPOSED`. That list
asserts a module builds the engagement gate IN ITS OWN SOURCE, which is right for analyses that
collect rest frames themselves. This one collects nothing: it reads `maps_by_epoch`, which gates
internally. Asserting `engaged_by_cue` appears here would force a copy of the gate into a module
that must not have one -- exactly the templating that spread the original defect.
"""
from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

MOD = "scripts.rest_migration.subacute_daymatched"


def _mod():
    return pytest.importorskip(MOD)


def test_z_makes_a_dot_product_into_pearson_r():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    m = _mod()
    got = float(m._z(a) @ m._z(b))
    assert got == pytest.approx(float(np.corrcoef(a, b)[0, 1]), abs=1e-12)


def test_z_refuses_a_constant_vector():
    """A zero-variance map has no correlation with anything; it must drop out, not divide by zero."""
    assert _mod()._z(np.ones(500)) is None


def test_z_refuses_a_vector_with_too_few_finite_values():
    v = np.full(500, np.nan)
    v[:5] = 1.0
    assert _mod()._z(v) is None


def _store(days):
    """``{animal: {epoch: {position: {label: {ref: map}}}}}`` with one position and one reference.

    Values are the DAY itself, so a window's mean is the mean of the days it selected and the
    selection is readable straight off the number.
    """
    out = {}
    for an, per_epoch in days.items():
        for epoch, ds in per_epoch.items():
            for d in ds:
                lab = f"{an}_{d:04d}"
                (out.setdefault(an, {}).setdefault(epoch, {}).setdefault("close_L", {})
                    [lab]) = {"ref": np.full(4, float(d))}
    return out


def test_window_selects_on_day_and_ignores_the_epoch_label(monkeypatch):
    """THE POINT OF THE SCRIPT. A session in range counts even if it was labelled `subacute`.

    PS94's days 11-29 are labelled subacute only because it has no chronic bin, so day-selection
    must see them regardless of that label -- otherwise re-binning by day would just reproduce the
    epoch bins it exists to bypass.

    That is a statement about the SELECTOR, not a licence to put PS94 in chronic. The window-level
    `exclude` keeps it out of the chronic control for a separate reason: it has not plateaued. See
    `test_PS94_is_excluded_from_the_day_binned_chronic_window`.
    """
    m = _mod()
    monkeypatch.setattr(m, "_day_of", lambda _an, lab: int(lab.split("_")[-1]))
    store = _store({"PS94": {"acute": [1, 5], "subacute": [9, 11, 22]}})
    got, n = m._window(store, "ref", "close_L", 11, 29)
    assert n == {"PS94": 2}
    assert got["PS94"] == pytest.approx(np.full(4, 16.5))       # mean of 11 and 22


def test_window_bounds_are_inclusive(monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_day_of", lambda _an, lab: int(lab.split("_")[-1]))
    store = _store({"PS92": {"subacute": [5, 7, 9]}})
    _got, n = m._window(store, "ref", "close_L", 5, 9)
    assert n == {"PS92": 3}, "a session exactly on either bound must be included"


def test_window_never_takes_a_pre_stroke_session(monkeypatch):
    """`pre` is the SUBTRAHEND. Letting it into a post window would subtract a map from itself."""
    m = _mod()
    monkeypatch.setattr(m, "_day_of", lambda _an, lab: int(lab.split("_")[-1]))
    store = _store({"PS92": {"pre": [3], "acute": [3]}})
    _got, n = m._window(store, "ref", "close_L", 1, 7)
    assert n == {"PS92": 1}, "the pre epoch must be skipped even when its day is in range"


def test_window_drops_an_animal_with_no_session_in_range(monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_day_of", lambda _an, lab: int(lab.split("_")[-1]))
    store = _store({"PS92": {"acute": [1]}, "PS93": {"chronic": [22]}})
    got, _n = m._window(store, "ref", "close_L", 11, 29)
    assert set(got) == {"PS93"}


def test_windows_are_ordered_and_inclusive_ranges():
    for lab, lo, hi, _excl in _mod().WINDOWS:
        assert 1 <= lo <= hi, f"{lab} has an empty or pre-stroke range"


def test_PS94_is_excluded_from_the_day_binned_chronic_window():
    """PS94 HAS NOT PLATEAUED -- it fails the chronic rule on DRIFT, not residual, and is still
    rising through day 29 while the other three plateaued at day 11. Folding its days 11-29 into a
    day-binned chronic would put a still-recovering animal in the plateau bin: the subacute error
    moved one bin over. An earlier version of this script did exactly that and called it
    'recovering an animal for chronic'."""
    chronic = [w for w in _mod().WINDOWS if w[0].startswith("chronic")]
    assert chronic, "the chronic control window is gone"
    for _lab, _lo, _hi, excl in chronic:
        assert "PS94" in excl, "PS94 must be excluded from any day-binned chronic window"


def test_the_subacute_windows_exclude_nobody():
    """The subacute test is about the BIN, so dropping an animal there would answer a different
    question -- and PS94 is precisely the animal whose subacute placement is under test."""
    for lab, _lo, _hi, excl in _mod().WINDOWS:
        if lab.startswith("subacute"):
            assert excl == (), f"{lab} must keep all four animals"


def test_the_matched_subacute_windows_are_reachable_by_all_four_animals():
    """The diagnostic is worthless if the matched window is one animal's sessions again.

    Day sets are hard-coded from the registry as of 2026-09-17 deliberately: if a new session
    changes them, this test should FAIL and the window bounds should be re-chosen by hand rather
    than silently drifting to whatever the registry now allows.
    """
    days = {"PS92": {7, 9}, "PS93": {5, 7, 9}, "PS94": {9, 11, 15, 18, 22, 25, 29},
            "PS95": {2, 3, 4, 5, 7, 9}}
    for lab, lo, hi, _excl in _mod().WINDOWS:
        if not lab.startswith("subacute"):
            continue
        reach = [a for a, ds in days.items() if any(lo <= d <= hi for d in ds)]
        assert len(reach) == 4, f"{lab} is not reachable by all four animals, only {reach}"


def test_it_reads_through_maps_by_epoch_rather_than_collecting_rest_itself():
    """Why this module is NOT on EXPOSED -- see the module docstring."""
    code = inspect.getsource(_mod())
    assert "maps_by_epoch" in code, "must read the gated library path"
    assert "rest_frames_by_position" not in code, "must not collect rest frames itself"
    assert "engaged_by_cue" not in code, (
        "must NOT build its own gate -- it inherits one, and a second copy is how the "
        "original defect spread")


def test_it_never_reaches_the_raw_classifier():
    """The 0806 sessions collapse 6 positions to 4 under raw `_classify_cues`."""
    code = "\n".join(ln.split("#")[0] for ln in inspect.getsource(_mod()).splitlines())
    assert "_classify_cues(" not in code


def test_it_is_deliberately_absent_from_the_exposed_list():
    """If someone adds it there, the gate assertion will fail and the fix will look like
    'add engaged_by_cue to this file' -- which is the wrong fix. Fail here first, with a reason."""
    exposed = importlib.import_module("tests.test_rest_engagement").EXPOSED
    assert MOD not in exposed, (
        "subacute_daymatched inherits the gate through maps_by_epoch and must not build its own")
