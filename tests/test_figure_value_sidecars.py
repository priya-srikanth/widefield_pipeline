"""EVERY figure primitive must record the numbers it plots, beside it, as CSV.

Priya, 2026-09-12: "just make sure the code for all the sidecars exists."

THE GAP THIS CLOSES, and how it was found. `write_values` landed for the bar families in
7be02de/8aae500, and the audit stopped there. On 2026-09-12 a question about family 12b's bars --
does the post-stroke stopped pattern still resemble the pre-stroke STOPPED one -- had to be answered
by reading values off a rendered PNG, because `matrix_row`, `confusion_row`, `map_grid`,
`matrix_grid_by_animal` and both time-course primitives wrote no sidecar at all. Six of the eight
primitives in the module were silent, which covers every confusion, crossnobis, best-match
destination and time-course number in the deck.

`write_values`'s own docstring states the rule: quoting a figure should not be an act of eyesight.
These tests make that structural rather than aspirational -- the last one fails if a NEW primitive
is added without one, which is the only way the gap stays closed.
"""
from __future__ import annotations

import ast
import csv
import inspect
import pathlib

import numpy as np
import pytest

from wfield_local import epoch_figures as ef

LABELS = ["nI", "nM", "nC", "fI", "fM", "fC"]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _mats(seed=0, panels=("pre", "acute", "subacute", "chronic")):
    rng = np.random.default_rng(seed)
    return {e: rng.random((6, 6)) for e in panels}


# --------------------------------------------------------------------------- matrices


def test_a_matrix_sidecar_holds_every_cell_of_every_panel(tmp_path):
    mats = _mats()
    ef.write_matrix_values(mats, tmp_path / "m.png", labels=LABELS)
    rows = _rows(tmp_path / "m.csv")
    assert len(rows) == 4 * 36
    assert {r["panel"] for r in rows} == set(mats)
    assert {r["row"] for r in rows} == set(LABELS)


def test_the_matrix_delta_is_the_one_the_figure_draws(tmp_path):
    """`matrix_row` draws ``panel - pre``. Deriving it here is what stops the two disagreeing."""
    mats = _mats()
    ef.write_matrix_values(mats, tmp_path / "m.png", labels=LABELS)
    got = {(r["panel"], r["row"], r["col"]): r["delta_vs_pre"] for r in _rows(tmp_path / "m.csv")}
    for i, rl in enumerate(LABELS):
        for j, cl in enumerate(LABELS):
            assert got[("pre", rl, cl)] == "", "pre has no delta against itself"
            expect = mats["acute"][i, j] - mats["pre"][i, j]
            assert float(got[("acute", rl, cl)]) == pytest.approx(expect)


def test_a_confusion_sidecar_keeps_the_raw_count_beside_the_fraction(tmp_path):
    """A fraction alone cannot tell 1/1 from 40/40, and `confusion_row` plots the fraction."""
    rng = np.random.default_rng(1)
    counts = {e: rng.integers(1, 50, (6, 6)).astype(float) for e in ("pre", "acute")}
    norm = {e: M / M.sum(1, keepdims=True) for e, M in counts.items()}
    ef.write_matrix_values(norm, tmp_path / "c.png", labels=LABELS,
                           extra=counts, extra_name="count")
    rows = _rows(tmp_path / "c.csv")
    assert "count" in rows[0]
    r = [x for x in rows if x["panel"] == "pre" and x["row"] == "nI" and x["col"] == "nM"][0]
    assert float(r["count"]) == counts["pre"][0, 1]
    assert float(r["value"]) == pytest.approx(norm["pre"][0, 1])


def test_missing_labels_fall_back_to_indices_rather_than_crashing(tmp_path):
    ef.write_matrix_values({"pre": np.zeros((3, 3))}, tmp_path / "m.png", labels=None)
    assert {r["row"] for r in _rows(tmp_path / "m.csv")} == {"r0", "r1", "r2"}


def test_the_per_animal_grid_deltas_against_that_animals_own_pre(tmp_path):
    """The grid exists BECAUSE pooling hides the animal; a pooled pre would defeat it."""
    rng = np.random.default_rng(2)
    grid = {"PS94": {e: rng.random((6, 6)) for e in ("pre", "acute")},
            "PS95": {e: rng.random((6, 6)) for e in ("pre", "acute")}}
    ef.write_matrix_grid_values(grid, tmp_path / "g.png", labels=LABELS)
    rows = _rows(tmp_path / "g.csv")
    assert {r["animal"] for r in rows} == {"PS94", "PS95"}
    r = [x for x in rows if x["animal"] == "PS95" and x["panel"] == "acute"
         and x["row"] == "nI" and x["col"] == "nI"][0]
    assert float(r["delta_vs_pre"]) == pytest.approx(grid["PS95"]["acute"][0, 0]
                                                     - grid["PS95"]["pre"][0, 0])


def test_an_animal_with_no_pre_panel_still_writes_its_values(tmp_path):
    grid = {"PS92": {"acute": np.ones((2, 2))}}
    ef.write_matrix_grid_values(grid, tmp_path / "g.png", labels=["a", "b"])
    rows = _rows(tmp_path / "g.csv")
    assert len(rows) == 4 and all(r["delta_vs_pre"] == "" for r in rows)


# --------------------------------------------------------------------------- time courses


def test_a_time_course_sidecar_is_one_row_per_position_animal_day(tmp_path):
    per_day = {"nI": {"PS94": {0: 0.5, 3: 0.6}, "PS95": {1: 0.4}}, "fC": {"PS94": {0: 0.2}}}
    ef.write_series_values(per_day, tmp_path / "s.png")
    rows = _rows(tmp_path / "s.csv")
    assert len(rows) == 4
    assert {(r["position"], r["animal"], r["day"]) for r in rows} == {
        ("nI", "PS94", "0"), ("nI", "PS94", "3"), ("nI", "PS95", "1"), ("fC", "PS94", "0")}


def test_days_are_written_in_order_within_an_animal(tmp_path):
    """These traces are what the epoch boundaries were drawn FROM; out-of-order days obscure that."""
    per_day = {"nI": {"PS94": {10: 0.1, -2: 0.9, 3: 0.5}}}
    ef.write_series_values(per_day, tmp_path / "s.png")
    assert [r["day"] for r in _rows(tmp_path / "s.csv")] == ["-2", "3", "10"]


# --------------------------------------------------------------------------- maps


def test_a_map_sidecar_digests_each_cell_rather_than_dumping_pixels(tmp_path):
    """A map is ~10^5 pixels; the claims made from these figures are about scale and sign."""
    rng = np.random.default_rng(3)
    cells = {("far_L", "pre"): rng.random((8, 9)), ("far_L", "acute"): np.full((8, 9), np.nan)}
    ef.write_map_summary(cells, tmp_path / "mp.png")
    rows = _rows(tmp_path / "mp.csv")
    pre = [r for r in rows if r["col"] == "pre"][0]
    assert int(pre["n_finite"]) == 72
    assert float(pre["max"]) <= 1.0 and float(pre["min"]) >= 0.0
    blank = [r for r in rows if r["col"] == "acute"][0]
    assert int(blank["n_finite"]) == 0 and blank["mean"] == ""


# --------------------------------------------------------------------------- never cost a figure


@pytest.mark.parametrize("writer,payload", [
    (ef.write_matrix_values, {"pre": "not a matrix"}),
    (ef.write_matrix_grid_values, {"PS94": None}),
    (ef.write_series_values, {"nI": "not a dict"}),
    (ef.write_map_summary, {("a", "b"): "not an array"}),
])
def test_a_broken_payload_warns_and_returns_rather_than_raising(writer, payload, tmp_path):
    """`_save_png_svg`'s rule: a sidecar must never cost a figure that already rendered."""
    assert writer(payload, tmp_path / "x.png") is not None


# --------------------------------------------------------------------------- the structural guard


# --------------------------------------------------------------------------- the ANNOTATION layer
#
# Priya, 2026-09-12: "do the sidecars carry all information we would need to regenerate the figures
# (without redoing analysis)?" The value writers cover what a figure PLOTS. These cover what it
# STATES -- and some of that is data, not styling: how many animals and sessions stand behind an
# epoch, what chance level a bar is read against, what the Bonferroni mark was corrected by.


def test_the_meta_sidecar_records_the_counts_a_panel_states():
    """The acute panel is six PS94 sessions against one PS95 session; a redraw must be able to say so."""
    import tempfile
    from wfield_local.figure_meta import write_meta

    d = pathlib.Path(tempfile.mkdtemp())
    q = write_meta(d / "f.png", counts={"acute": {"PS94": 6, "PS95": 1}}, chance=1 / 6)
    rows = _rows(q)
    got = {(r["kind"], r["key"]): r["value"] for r in rows}
    assert got[("counts", "acute|PS94")] == "6"
    assert got[("counts", "acute|PS95")] == "1"
    assert got[("chance", "")].startswith("0.1666")


def test_a_field_that_does_not_apply_is_absent_rather_than_blank():
    """"Not applicable to this family" and "applicable and empty" are different facts."""
    import tempfile
    from wfield_local.figure_meta import write_meta

    d = pathlib.Path(tempfile.mkdtemp())
    q = write_meta(d / "f.png", chance=None, counts=None, title="t")
    assert {r["kind"] for r in _rows(q)} == {"title"}


def test_write_meta_returns_none_rather_than_an_empty_file():
    import tempfile
    from wfield_local.figure_meta import write_meta

    assert write_meta(pathlib.Path(tempfile.mkdtemp()) / "f.png") is None


def test_an_axis_limit_reads_as_lo_hi_not_as_zero_one():
    import tempfile
    from wfield_local.figure_meta import write_meta

    d = pathlib.Path(tempfile.mkdtemp())
    rows = _rows(write_meta(d / "f.png", ylim=(-1.05, 1.10)))
    assert {r["key"] for r in rows} == {"lo", "hi"}


def test_a_bar_figure_writes_values_sessions_and_meta_together(tmp_path):
    """All three, or the figure is only partly quotable."""
    labels = LABELS
    vals = {e: {p: (0.5, 0.4, 0.6) for p in labels} for e in ("pre", "acute")}
    pts = {e: {p: [("PS94", 0.5)] for p in labels} for e in ("pre", "acute")}
    ef.bar_row(vals, str(tmp_path), name="b", title="T", ylabel="acc", positions=labels,
               points=pts, chance=1 / 6, counts={"acute": {"PS94": 6}})
    got = {p.name for p in tmp_path.glob("b*")}
    assert {"b.png", "b.csv", "b_sessions.csv", "b_meta.csv"} <= got


def test_the_correction_divisor_behind_a_star_is_recorded():
    """`write_values` stores the MARK; without n_comparisons a reader sees ** and cannot check it."""
    import tempfile
    from wfield_local.figure_meta import write_meta

    d = pathlib.Path(tempfile.mkdtemp())
    rows = _rows(write_meta(d / "f.png", n_comparisons=6))
    assert [r["value"] for r in rows if r["kind"] == "n_comparisons"] == ["6"]


# --------------------------------------------------------------------------- the structural guard


#: Primitives that draw a figure but have no numbers of their own to record.
_EXEMPT: set[str] = set()


def test_every_primitive_that_saves_a_figure_also_writes_a_sidecar():
    """THE ONE THAT KEEPS THIS CLOSED.

    The six missing writers were not a decision; they were an audit that stopped at the bar
    families and was never revisited, and nothing in the repo could notice. This walks the module's
    AST: any top-level function calling `_save_png_svg` must also call one of the writers.
    """
    src = pathlib.Path(inspect.getfile(ef)).read_text(encoding="utf-8")
    writers = {"write_values", "write_contrast_values", "write_matrix_values",
               "write_matrix_grid_values", "write_series_values", "write_map_summary"}
    missing, no_meta = [], []
    for node in ast.parse(src).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        called = {n.func.id for n in ast.walk(node)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if "_save_png_svg" not in called or node.name in _EXEMPT:
            continue
        if not (called & writers):
            missing.append(node.name)
        if "write_meta" not in called:
            no_meta.append(node.name)
    assert not missing, (
        f"these primitives render a figure but record nothing: {missing}. Add a writer, or add the "
        f"name to _EXEMPT with a reason why the figure has no numbers of its own.")
    assert not no_meta, (
        f"these primitives record their VALUES but not what the figure STATES: {no_meta}. A redraw "
        f"would carry the right bars and be unable to name the chance level or the session counts. "
        f"Call figure_meta.write_meta.")
