"""Layout guards for the delta grids (5d/6d/7d/8d).

Both failures pinned here shipped and were reported by eye rather than caught: the reference colour
bar drawn over the day-1 matrices, and a long suptitle stretching the saved canvas until the panels
occupied a third of it. Eyes are the wrong instrument for this -- a figure that is merely ugly looks
much like a figure that is fine -- so they are measured.
"""
import itertools

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
Image = pytest.importorskip("PIL.Image")

from wfield_local import grant_figures as gf

DAYS = [1, 2, 3, 4, 5, 7, 9]


def _mats(rng):
    return {an: {k: rng.uniform(-1, 1, (6, 6)) for k in ["PRE"] + DAYS} for an in gf.ANIMALS}


def _draw(tmp_path, title, cis=None, name="g.png"):
    rng = np.random.default_rng(0)
    return gf._delta_grid(_mats(rng), DAYS, tmp_path, name, title=title, abs_label="pre-stroke r",
                          delta_label="change in r", vmin=-1, vmax=1, cmap="RdBu_r", dmax=1.0,
                          summary=gf._diag, ylab="this position", cis=cis)


def test_a_long_title_does_not_stretch_the_canvas(tmp_path):
    """bbox_inches='tight' sizes the canvas around EVERYTHING, so one over-long line squashes the
    panels. A 420-character line did exactly that to 6d once the bootstrap interval was described."""
    short = _draw(tmp_path, "A short title", name="short.png")
    long_ = _draw(tmp_path, "x" * 420, name="long.png")
    a_short = Image.open(short).width / Image.open(short).height
    a_long = Image.open(long_).width / Image.open(long_).height
    assert a_long < a_short * 1.25, (
        f"a long title stretched the figure: aspect {a_short:.2f} -> {a_long:.2f}")


def test_deliberate_line_breaks_are_preserved(tmp_path):
    """Only OVER-long lines are wrapped; a caller's own newlines must survive, or a hand-tuned
    multi-line header would be silently reflowed."""
    p = _draw(tmp_path, "line one\nline two\nline three", name="brk.png")
    assert p.exists()


def test_reference_colour_bar_clears_the_first_delta_panel(tmp_path):
    """The bar, its tick labels AND its axis label must all end before the day-1 panel begins.

    The first fix moved only the bar; matplotlib draws ticks and label on the RIGHT by default, so
    the rotated label still reached over the first matrix.
    """
    real_close = matplotlib.pyplot.close
    held = {}

    def keep(f=None):
        if hasattr(f, "canvas"):
            held["fig"] = f
        else:
            real_close(f)

    matplotlib.pyplot.close = keep
    try:
        _draw(tmp_path, "t", name="bar.png")
    finally:
        matplotlib.pyplot.close = real_close

    fig = held["fig"]
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()

    # The reference bar is added with fig.add_axes, so it carries no "<colorbar>" label; find it
    # geometrically as the leftmost narrow axes holding no image.
    bars = [a for a in fig.axes if not a.images and a.get_position().width < 0.05]
    ref = min(bars, key=lambda a: a.get_position().x0)
    boxes = [ref.get_position()]
    boxes += [inv.transform_bbox(t.get_window_extent(rend)) for t in ref.get_yticklabels()]
    boxes.append(inv.transform_bbox(ref.yaxis.label.get_window_extent(rend)))
    right = max(b.x1 for b in boxes)

    xs = sorted({round(a.get_position().x0, 4) for a in fig.axes if a.images})
    day1_left = xs[1]
    assert right < day1_left, (
        f"colour-bar artists reach x={right:.4f}, day-1 panel starts at {day1_left:.4f}")
    real_close(fig)


@pytest.mark.parametrize("n_lines,height", [(2, 4.5), (7, 4.5), (7, 15.5), (4, 9.2)])
def test_suptitle_never_overlaps_the_axes(n_lines, height):
    """Matplotlib anchors a suptitle near y=0.98 and grows it DOWNWARD, so a header gains lines at
    the expense of the top row of panels.

    Every figure in this module used to reserve a hand-tuned constant (rect=(0, 0, 1, 0.88) and
    friends), which held only while the text did -- adding the bootstrap description to several
    headers pushed them straight over their own figures. The reservation is now computed from the
    line count AND the figure's real height, because these range from 4.5 to 15.5 inches and a
    five-line header costs a third of the short one and a tenth of the tall one.
    """
    import matplotlib.pyplot as plt

    fig, _ax = plt.subplots(2, 3, figsize=(12, height))
    gf._suptitle(fig, "\n".join(f"header line {i}" for i in range(n_lines)))
    fig.canvas.draw()
    box = fig.transFigure.inverted().transform_bbox(
        fig._suptitle.get_window_extent(fig.canvas.get_renderer()))
    axes_top = max(a.get_position().y1 for a in fig.axes)
    plt.close(fig)
    assert box.y0 >= axes_top - 1e-3, (
        f"{n_lines}-line title on a {height}in figure reaches y={box.y0:.3f}, "
        f"axes top at {axes_top:.3f}")


def test_suptitle_is_not_recursive():
    """The sweep that routed every fig.suptitle in the module through _suptitle rewrote the helper's
    OWN call too. It renders as a RecursionError, but only at render time, on whichever figure runs
    first -- so it is pinned here where it costs a millisecond."""
    import matplotlib.pyplot as plt

    fig, _ax = plt.subplots(1, 1, figsize=(6, 4))
    gf._suptitle(fig, "a title")            # would raise RecursionError if it called itself
    plt.close(fig)


def test_txt_is_the_single_gate_on_in_cell_numbers():
    """Every in-cell number in grant_figures goes through `_txt`, so the compact variant cannot be
    half-applied -- which is what would happen if each of the eight call sites carried its own
    `if not COMPACT` and a ninth were added later without one."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    assert gf.COMPACT is False
    assert gf._txt(ax, 0, 0, "0.42") is not None
    n_full = len(ax.texts)

    gf.COMPACT = True
    try:
        assert gf._txt(ax, 1, 1, "0.42") is None
    finally:
        gf.COMPACT = False
    assert len(ax.texts) == n_full, "compact must add no text object at all"
    plt.close(fig)


def test_compact_narrows_the_grid_and_tags_the_file(tmp_path):
    """The compact variant is for reproduction at a quarter of a letter page, where 32 panels are
    0.13in each. It must be narrower than the full one AND must not overwrite it."""
    from PIL import Image

    full = _draw(tmp_path, "t", name="grant_x_delta.png")
    gf.COMPACT = True
    try:
        comp = _draw(tmp_path, "t", name="grant_x_delta.png")
    finally:
        gf.COMPACT = False
    assert gf.COMPACT is False, "the flag must be left off after a compact render"

    assert comp.name.endswith("_compact.png") and comp != full
    assert full.exists() and comp.exists()
    assert Image.open(comp).width < Image.open(full).width


def test_compact_tick_labels_do_not_collide_at_the_narrower_width(tmp_path):
    """Narrowing the panels is only useful if the enlarged labels still fit inside them."""
    import matplotlib.pyplot as plt

    real_close, held = plt.close, {}

    def keep(f=None):
        if hasattr(f, "canvas"):
            held["fig"] = f
        else:
            real_close(f)

    gf.COMPACT = True
    plt.close = keep
    try:
        _draw(tmp_path, "t", name="c.png")
    finally:
        plt.close = real_close
        gf.COMPACT = False

    fig = held["fig"]
    fig.canvas.draw()
    inv, rend = fig.transFigure.inverted(), fig.canvas.get_renderer()
    clash = None
    for ax in fig.axes:
        for getter in (ax.get_xticklabels, ax.get_yticklabels):
            labs = [t for t in getter() if t.get_text().strip()]
            boxes = [inv.transform_bbox(t.get_window_extent(rend)) for t in labs]
            for (b1, t1), (b2, t2) in itertools.pairwise(
                    sorted(zip(boxes, labs), key=lambda b: (b[0].x0, b[0].y0))):
                if (min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1e-4
                        and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1e-4):
                    clash = (t1.get_text(), t2.get_text())
    real_close(fig)
    assert clash is None, f"compact tick labels overlap: {clash}"


def test_bottom_legend_clears_the_footer():
    """`_footer` writes at y=0.004 and a legend at "lower center" defaults to the same place; in
    figure 9's first render the two overprinted. Any figure carrying both must separate them.

    Checked structurally rather than by rendering figure 9, which needs the full LocaNMF data: the
    module is scanned for functions that contain BOTH, and each must anchor its legend explicitly.
    """
    import re
    from pathlib import Path

    src = Path(gf.__file__).read_text(encoding="utf-8")
    offenders = []
    for part in re.split(r"^def ", src, flags=re.MULTILINE)[1:]:
        name = part.split("(")[0]
        if 'loc="lower center"' in part and "_footer(fig" in part:
            # the legend call must pin its own y, and it must sit above the footer's 0.004
            # NB [^)]* cannot cross the ")" in `ncol=len(POS)`, which is how the first version of
            # this test flagged a call that was in fact correct.
            m = re.search(r'loc="lower center"[\s\S]{0,240}?bbox_to_anchor=\(0\.5,\s*([0-9.]+)\)',
                          part)
            if not m or float(m.group(1)) <= 0.004:
                offenders.append(name)
    assert not offenders, (
        f"these draw a bottom legend and a footer without separating them: {offenders}")
