"""Epoch-colour convention helpers (PI decision A13, Priya 11-07): `cat_points.day_dots`, `epoch_mean_sem`,
`epoch_color`, `legend_handles`, `fit_ylim`.

Session points are OPEN markers, shape = animal, colour = epoch (the canonical `orofacial_cohort.EPOCH_LINE`), with no
line joining sessions; the per-epoch summary is a filled dot in the epoch colour with +- SEM across ANIMALS.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wfield_local import cat_points as cp  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402


@pytest.fixture
def ax():
    fig, a = plt.subplots()
    yield a
    plt.close(fig)


def _same(c1, c2) -> bool:
    return np.allclose(mcolors.to_rgba(c1)[:3], mcolors.to_rgba(c2)[:3])


def test_epoch_color_is_the_canonical_palette():
    for e, c in EPOCH_LINE.items():
        assert cp.epoch_color(e) == c
    assert cp.epoch_color(None) == cp.NO_EPOCH
    assert cp.epoch_color("unknown") == cp.NO_EPOCH


def test_epochs_of_days_uses_the_one_rule():
    from wfield_local.epoch_figures import epoch_of_day
    an, days = ["PS92", "PS92", "PS92", "PS95"], [-4, 2, 30, np.nan]
    assert cp.epochs_of_days(an, days) == [epoch_of_day("PS92", -4), epoch_of_day("PS92", 2),
                                           epoch_of_day("PS92", 30), None]


def test_day_dots_open_markers_shape_animal_colour_epoch_no_lines(ax):
    days = [-5, -3, 2, 3, 20, -4, 2, 25]
    vals = [1.0, 1.2, 0.2, 0.4, 0.9, 1.1, 0.1, 0.8]
    an = ["PS92"] * 5 + ["PS93"] * 3
    ep = ["pre", "pre", "acute", "acute", "chronic", "pre", "acute", "chronic"]
    cp.day_dots(ax, days, vals, an, ep, summary=False)
    lines = ax.get_lines()
    assert lines and not cp.no_cross_epoch_lines(ax)
    for ln in lines:
        assert ln.get_markerfacecolor() == "none"
        x, y = ln.get_data()
        # every point of one artist is one animal and one epoch
        idx = [i for i in range(len(days)) if days[i] in list(x) and vals[i] in list(y)]
        assert len({an[i] for i in idx}) == 1 and len({ep[i] for i in idx}) == 1
        assert ln.get_marker() == cp.animal_marker(an[idx[0]])
        assert _same(ln.get_markeredgecolor(), EPOCH_LINE[ep[idx[0]]])


def test_day_dots_summary_is_mean_sem_over_animals(ax):
    # PS92 has 2 acute sessions (0.2, 0.4 -> animal mean 0.3), PS93 one (0.1): summary = mean(0.3, 0.1), SEM over 2
    days = [2, 3, 2]
    vals = [0.2, 0.4, 0.1]
    an = ["PS92", "PS92", "PS93"]
    out = cp.day_dots(ax, days, vals, an, ["acute"] * 3)
    x, mu, se, n = out["acute"]
    assert n == 2
    assert mu == pytest.approx(0.2)
    assert se == pytest.approx(np.std([0.3, 0.1], ddof=1) / np.sqrt(2))
    assert x == pytest.approx(np.mean([2.5, 2.0]))
    # the summary is a filled dot in the epoch colour, drawn with no connecting line
    filled = [ln for ln in ax.get_lines() if ln.get_markerfacecolor() != "none"]
    assert filled and all(_same(ln.get_markerfacecolor(), EPOCH_LINE["acute"]) for ln in filled)
    assert not cp.no_cross_epoch_lines(ax)


def test_day_dots_default_epochs_and_nonfinite_dropped(ax):
    out = cp.day_dots(ax, [-3, 2, np.nan, 30], [1.0, 0.5, 9.0, np.nan], ["PS92"] * 4)
    assert set(out) == {"pre", "acute"}


def test_epoch_mean_sem_single_animal_has_no_bar(ax):
    mu, se, n = cp.epoch_mean_sem(ax, 1, [0.4, np.nan], "chronic")
    assert (mu, n) == (0.4, 1) and np.isnan(se)
    assert cp.epoch_mean_sem(ax, 1, [np.nan], "chronic") is None


def test_legend_handles_animals_then_epochs():
    h = cp.legend_handles(["PS93", "PS92"])
    assert [x.get_label() for x in h[:2]] == ["PS92 sessions", "PS93 sessions"]
    assert all(x.get_markerfacecolor() == "none" for x in h[:2])
    assert [x.get_label().split(":")[0] for x in h[2:]] == list(cp.EPOCH_ORDER)


def test_fit_ylim_grows_never_shrinks(ax):
    ax.set_ylim(0, 1)
    cp.fit_ylim(ax, [0.2, 0.5])
    assert ax.get_ylim() == (0, 1)
    cp.fit_ylim(ax, [-0.5, 1.5, np.nan])
    lo, hi = ax.get_ylim()
    assert lo < -0.5 and hi > 1.5


def test_no_cross_epoch_lines_flags_a_joined_series(ax):
    ax.plot([1, 2, 3], [1, 2, 3], "o-")
    ax.axhline(0)
    assert len(cp.no_cross_epoch_lines(ax)) == 1
