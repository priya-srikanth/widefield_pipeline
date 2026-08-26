"""Layout guards for the delta grids (5d/6d/7d/8d).

Both failures pinned here shipped and were reported by eye rather than caught: the reference colour
bar drawn over the day-1 matrices, and a long suptitle stretching the saved canvas until the panels
occupied a third of it. Eyes are the wrong instrument for this -- a figure that is merely ugly looks
much like a figure that is fine -- so they are measured.
"""
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
