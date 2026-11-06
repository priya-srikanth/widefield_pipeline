"""sig_marks top-band stars (Priya 10-30 / 11-03): `star_top` + `reserve_headroom` grow the y range so the stars sit
above the data, and hide the top spine on every panel (the stars overlapped it)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from wfield_local import sig_marks as sm  # noqa: E402


@pytest.fixture
def fig():
    f = plt.figure()
    yield f
    plt.close(f)


def test_headroom_grows_by_rows(fig):
    a1, a2, a3 = fig.subplots(1, 3)
    for ax in (a1, a2, a3):
        ax.plot([0, 1], [0, 1])
        ax.set_ylim(0, 1)
    sm.star_top(a1, 0.5, "*", row=0)
    sm.star_top(a2, 0.5, "*", row=0)
    sm.star_top(a2, 0.5, "**", row=1)
    sm.reserve_headroom(fig)
    # data range [0, 1] must occupy exactly the bottom (1 - frac) of the axes
    f1 = sm.STEP * 1 + sm.PAD
    f2 = sm.STEP * 2 + sm.PAD
    assert a1.get_ylim() == pytest.approx((0, 1 / (1 - f1)))
    assert a2.get_ylim() == pytest.approx((0, 1 / (1 - f2)))
    assert a2.get_ylim()[1] > a1.get_ylim()[1]
    assert a3.get_ylim() == (0, 1)                 # no marks -> range untouched


def test_top_spine_hidden_on_every_axes(fig):
    a1, a2 = fig.subplots(1, 2)
    sm.star_top(a1, 0.0, "*")
    sm.reserve_headroom(fig)
    assert not a1.spines["top"].get_visible()
    assert not a2.spines["top"].get_visible()     # unmarked panels too
    assert a1.spines["bottom"].get_visible()


def test_shared_y_grown_once(fig):
    a1, a2 = fig.subplots(1, 2, sharey=True)
    a1.set_ylim(0, 1)
    sm.star_top(a1, 0.0, "*", row=0)
    sm.star_top(a2, 0.0, "*", row=0)
    sm.reserve_headroom(fig)
    f1 = sm.STEP + sm.PAD
    assert a1.get_ylim() == pytest.approx((0, 1 / (1 - f1)))
    assert a2.get_ylim() == a1.get_ylim()


def test_star_in_axes_fraction_top_band(fig):
    ax = fig.subplots()
    t0 = sm.star_top(ax, 0.3, "*", row=0)
    t1 = sm.star_top(ax, 0.3, "*", row=2)
    assert t0.get_position()[1] == pytest.approx(0.99)
    assert t1.get_position()[1] == pytest.approx(0.99 - 2 * sm.STEP)
    assert ax._wf_mark_rows == 3
