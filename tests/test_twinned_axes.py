"""`_twinned`: the layout checker must not report a twin axis as an overlap.

Figure 10 puts a second y-scale (mean rank) on its trend panel. The axes-overlap check reported all
four of them, on every window and class -- twenty fault lines describing a deliberate construction.
A checker that cries wolf trains the reader to skim past it, which is worse than not checking; and
these figures have already shipped real colour-bar-over-panel faults that a skimmed log would hide.

The exclusion has to be NARROW. Sharing an axis is not enough on its own: `plt.subplots(sharex=True)`
puts every panel of figure 8g into one shared group, so a siblings-only test would silently disable
the overlap check for an entire figure.
"""
import matplotlib.pyplot as plt
import pytest

# `grant_figures` selects the Agg backend on import, so plotting here is headless either way.
from wfield_local import grant_figures as gf


@pytest.fixture
def closefigs():
    yield
    plt.close("all")


def test_a_twinx_pair_is_recognised(closefigs):
    _fig, ax = plt.subplots()
    ax2 = ax.twinx()
    assert gf._twinned(ax, ax2)
    assert gf._twinned(ax2, ax), "the test must not depend on argument order"


def test_a_twiny_pair_is_recognised(closefigs):
    _fig, ax = plt.subplots()
    ax2 = ax.twiny()
    assert gf._twinned(ax, ax2)


def test_a_shared_grid_is_NOT_treated_as_twinned(closefigs):
    """THE FAILURE THIS GUARD EXISTS TO PREVENT. Figure 8g is built with sharex/sharey, so its
    panels are all siblings. If sharing alone counted, the axes-overlap check would go dark for the
    whole figure and a colour bar drawn over a panel would pass silently."""
    _fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    a, b = axes[0][0], axes[0][1]
    assert b in a.get_shared_x_axes().get_siblings(a), "precondition: they ARE siblings"
    assert not gf._twinned(a, b), "different rectangles are not a twin pair"


def test_unrelated_axes_at_the_same_place_are_not_excused(closefigs):
    """Two axes deliberately stacked at one rectangle, sharing nothing -- a colour bar dropped on
    top of a panel looks like this, and it must still be reported."""
    fig = plt.figure()
    a = fig.add_axes([0.1, 0.1, 0.5, 0.5])
    b = fig.add_axes([0.1, 0.1, 0.5, 0.5])
    assert not gf._twinned(a, b)


def test_the_checker_stays_quiet_on_a_twin_and_loud_on_a_real_overlap(closefigs):
    """End to end through `_overlaps`, because that is where the fault lines come from."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.twinx().plot([0, 1], [1, 0])
    names = {n for a, b, _ar in gf._overlaps(fig) for n in (a, b)}
    assert not any(n.startswith("AXES") for n in names), f"twin reported: {names}"

    over = fig.add_axes([0.3, 0.3, 0.3, 0.3])
    over.imshow([[0, 1], [1, 0]])
    hits = [(a, b) for a, b, _ar in gf._overlaps(fig)
            if a.startswith("AXES") and b.startswith("AXES")]
    assert hits, "an axes genuinely drawn over a panel must still be reported"
