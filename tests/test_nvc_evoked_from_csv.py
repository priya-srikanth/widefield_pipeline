"""`--from-csv` must re-derive the SAME numbers, and the trap is a missing measurement.

A session with too few trials gets an EMPTY CELL, not a zero. Written that way, read back by
`analysis_kit.read_rows` as **NaN** -- so the live path sees ``""`` and the re-derived path sees
``nan`` for the same session. The module's original filter was ``r.get(k, "") != ""``, which is
TRUE for NaN.

Measured against the real 2026-09-19 table, that difference is not cosmetic:

    nolick pre        34 real rows -> would have pooled 44
    nolick subacute   16           -> 18
    nolick chronic     7           -> 18        (11 of 18 are NaN)

and `nvc_evoked` says in its own prose that **the no-lick early 415 alone is the test**. So the
cheap re-run would have rewritten the headline from eleven missing measurements, reported a larger
n, and looked healthy doing it. That is what these tests exist for.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "nvc_evoked", ROOT / "scripts" / "rest_migration" / "nvc_evoked.py")
nvc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nvc)


@pytest.mark.parametrize("value", ["", None, float("nan"), np.nan, "nan"])
def test_a_missing_measurement_is_not_a_measurement(value):
    assert nvc._present(value) is False


@pytest.mark.parametrize("value", [0.0, -0.0, 1.5, -2.25, "0.0", "-1.5", 3])
def test_a_real_value_is_kept_including_zero(value):
    """Zero is a measurement. A filter that dropped it would bias every window toward its sign."""
    assert nvc._present(value) is True


def test_an_infinite_value_is_not_treated_as_a_measurement():
    assert nvc._present(float("inf")) is False
    assert nvc._present(float("-inf")) is False


def test_the_curves_round_trip_exactly(tmp_path):
    """The figure is drawn from these, so 'close enough' is not enough -- the whole point of the
    round trip is that a byte comparison of the PNG is a usable test."""
    t = np.linspace(-1.0, 3.0, 41)
    curves = {
        ("nolick", "pre"): [("PS92", np.sin(t), np.cos(t)), ("PS93", t * 0.5, t * -0.25)],
        ("lick", "chronic"): [("PS94", np.exp(-t ** 2), t ** 2)],
    }
    q = nvc._save_curves(curves, t, tmp_path)
    assert q is not None and q.parent.name == "data", "curves must not land flat"

    back, t_back = nvc._load_curves(tmp_path)
    assert np.array_equal(t, t_back)
    assert set(back) == set(curves)
    for key, entries in curves.items():
        assert [e[0] for e in back[key]] == [e[0] for e in entries], "animal order must survive"
        for (_, a470, a415), (_, b470, b415) in zip(entries, back[key]):
            assert np.array_equal(a470, b470)
            assert np.array_equal(a415, b415)


def test_absent_curves_report_rather_than_invent(tmp_path):
    """`--from-csv` without the .npz must leave the published figure alone, not redraw it from
    window averages. `_load_curves` returning None is what makes that branch reachable."""
    assert nvc._load_curves(tmp_path) == (None, None)


def test_nothing_is_written_when_there_is_nothing_to_write(tmp_path):
    assert nvc._save_curves({}, np.arange(3.0), tmp_path) is None
    assert nvc._save_curves({("a", "b"): [("PS92", [1.0], [2.0])]}, None, tmp_path) is None
    assert not (tmp_path / "data").exists() or not list((tmp_path / "data").glob("*.npz"))


# ------------------------------------------------------ the filter, exercised through main()

_HDR = ("label,animal,epoch,n_lick,n_nolick,"
        "lick_470_early,lick_470_late,lick_415_early,lick_415_late,"
        "nolick_470_early,nolick_470_late,nolick_415_early,nolick_415_late")


def _row(label, animal, epoch, nolick_vals):
    """One CSV row; `nolick_vals` of None means the window was not measurable -> empty cells."""
    lick = "1.0,1.0,0.5,0.5"
    no = "1.0,1.0,%s,0.5" % nolick_vals if nolick_vals is not None else ",,,"
    return f"{label},{animal},{epoch},10,10,{lick},{no}"


def test_main_from_csv_excludes_the_unmeasured_sessions(tmp_path):
    """The regression this whole file is about, driven through the real entry point.

    Four animals report a no-lick chronic value and four do not. If the filter counts the empty
    cells, `n_sessions` in the stats CSV reads 8 instead of 4 -- which is the shape of the real
    table, where `nolick chronic` was 7 real rows against 18 counted.
    """
    import csv as _csv

    d = tmp_path / "data"
    d.mkdir()
    lines = [_HDR]
    for i, an in enumerate(("PS92", "PS93", "PS94", "PS95")):
        lines.append(_row(f"{an}_090{i}", an, "chronic", "0.30"))
        lines.append(_row(f"{an}_091{i}", an, "chronic", None))   # not measurable
    (d / "epoch_16_nvc_evoked.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert nvc.main(["--from-csv", "--out", str(tmp_path)]) == 0
    with open(d / "epoch_16_nvc_evoked_stats.csv", newline="", encoding="utf-8") as fh:
        stats = {(r["cls"], r["epoch"]): r for r in _csv.DictReader(fh)}
    assert int(stats[("nolick", "chronic")]["n_sessions"]) == 4, (
        "the four rows with empty no-lick cells were pooled -- `_present` is not being used")


def test_main_from_csv_refuses_when_the_csv_is_absent(tmp_path):
    """It must not fall through to a full recompute, which is the cost --from-csv exists to avoid."""
    assert nvc.main(["--from-csv", "--out", str(tmp_path)]) == 1


def test_main_from_csv_does_not_rewrite_the_csv_it_just_read(tmp_path):
    """Re-emitting the input would launder a partial read into the file later runs trust."""
    d = tmp_path / "data"
    d.mkdir()
    src = d / "epoch_16_nvc_evoked.csv"
    body = "\n".join([_HDR] + [_row(f"PS92_090{i}", "PS92", "pre", "0.2") for i in range(4)]) + "\n"
    src.write_text(body, encoding="utf-8")
    before = src.read_bytes()
    assert nvc.main(["--from-csv", "--out", str(tmp_path)]) == 0
    assert src.read_bytes() == before
