"""`figure_cross` must draw every class it was given, at every grid shape it can take.

THE FAULT (2026-08-28). The panels were reshaped from one row of six to two rows of three -- a
legibility fix, since a 22.7in figure placed at 12.7in renders its 7pt ticks at 3.9pt -- but the
panel loop kept indexing `axes[0][k]`. From five classes onward that is `IndexError: index 3 is out
of bounds for axis 0 with size 3`.

IT HID FOR TWO REASONS, both worth pinning:

* THE LICK WINDOW NEVER FAILED. `CLASSES_LICK` has two entries, so `_nc` is 2, `_nr` is 1, and row 0
  IS the whole grid. Testing the reshape on the lick window would have passed.
* `_draw` CAUGHT IT. That wrapper exists so a plotting bug cannot discard 40 minutes of pooling, and
  it did its job -- the run completed and wrote its JSON. The cost is that the failure presented as
  MISSING FIGURES, not as an error, and a missing figure is invisible until someone measures which
  figures the deck actually placed.

So the test drives the function at every class count it can be handed and checks the axes, rather
than reading the diff. That is the third layout fault this month found by driving rather than
reading: the `_delta_grid` hspace comment that shipped over a gridspec that never got it, and a
`fig_asymmetry` fix applied to `fig_delta_trajectory` because both call
`plt.subplots(len(ANIMALS), 2, ...)`.
"""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import position_coding_directions as pcd  # noqa: E402


def _res(classes, animal="PS99"):
    """A `run_animal` result carrying `cross_matrix` for exactly these classes."""
    cm = {}
    for c in classes:
        cm[c] = {P: {D: {"mean": float(i == j)} for j, D in enumerate(pcd.BY_SEVERITY)}
                 for i, P in enumerate(pcd.BY_SEVERITY)}
    return {"animal": animal, "methods": {"dom": {"cross_matrix": cm}}}


@pytest.mark.parametrize("align,classes", [
    ("precue", pcd.CLASSES_FULL),      # 5 panels -> 2 rows of 3; the shape that broke
    ("cue", pcd.CLASSES_FULL),
    ("lick", pcd.CLASSES_LICK),        # 2 panels -> 1 row; the shape that always passed
])
def test_every_class_gets_a_panel(tmp_path, align, classes, monkeypatch):
    seen = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, axes = orig(*a, **k)
        seen["axes"] = axes
        seen["shape"] = np.asarray(axes).shape
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    q = pcd.figure_cross(_res(classes), tmp_path, align=align, meth="dom")
    assert q is not None and q.exists(), f"{align}: no figure written"

    n = len([c for c in classes])
    nr, nc = seen["shape"]
    assert nr * nc >= n, f"{align}: grid {nr}x{nc} cannot hold {n} panels"
    drawn = sum(1 for row in seen["axes"] for ax in row if ax.images)
    assert drawn == n, f"{align}: {drawn} panels drawn for {n} classes"


def test_the_spare_cell_is_switched_off(tmp_path, monkeypatch):
    """5 panels in a 2x3 grid leave one over. An unused axes draws as an empty framed box, which
    reads as a panel with no data rather than as no panel."""
    seen = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, axes = orig(*a, **k)
        seen["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    pcd.figure_cross(_res(pcd.CLASSES_FULL), tmp_path, align="precue", meth="dom")
    flat = [ax for row in seen["axes"] for ax in row]
    spare = [ax for ax in flat if not ax.images]
    assert len(spare) == len(flat) - len(pcd.CLASSES_FULL)
    for ax in spare:
        assert not ax.axison, "an unused cell is still drawing its frame"


def test_every_row_names_its_y_axis(tmp_path, monkeypatch):
    """With two rows, a y-label on panel 0 alone leaves the second row's axis unnamed -- the y-TICK
    rule already keyed on the column and the label did not."""
    seen = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, axes = orig(*a, **k)
        seen["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    pcd.figure_cross(_res(pcd.CLASSES_FULL), tmp_path, align="precue", meth="dom")
    for r, row in enumerate(seen["axes"]):
        if not any(ax.images for ax in row):
            continue
        assert row[0].get_ylabel(), f"row {r} has no y-axis label"


def test_the_loop_does_not_index_row_zero_only():
    """The specific regression, named. A reshape that leaves this behind fails only for >_nc
    panels, so it passes on the lick window and on any future two-class variant."""
    import inspect
    src = inspect.getsource(pcd.figure_cross)
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "axes[0][k]" not in body, "figure_cross indexes row 0 only; it will break past _nc panels"
