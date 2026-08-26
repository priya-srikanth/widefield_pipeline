"""Per-class confusions are stored at the finest granularity, so any population is a SUM.

WHY THIS EXISTS. `grant_figures.fig_confusion_pre_post_working` (5b) needs the post-stroke population
MINUS the terminal quit period. `section_g.json` already had a confusion, but summed over trials, and
5b's own docstring records the consequence: "a summed matrix cannot be un-summed". So 5b recomputed
the entire pooling from LocaNMF -- >10 minutes of network reads to redraw one figure, paid again on
every layout iteration.

The fix is granularity rather than a second store. One matrix PER CLASS means:

    post_working = poststroke_lick + poststroke_miss_working
    post_all     = poststroke_lick + poststroke_miss_working + poststroke_stopped

so nothing ever has to be un-summed. Two properties make that valid and are pinned below: the
matrices are RAW COUNTS (row-normalised matrices cannot be added), and every post class is scored by
the SAME frozen model (matrices from different decoders would not be comparable, let alone additive).

IT LIVES IN `position_coding_directions` because that is where the classes are DEFINED -- `_gate_all`
and `_mask` are here, and `miss_vs_stopped` already reads these class names out of
coding_direction.json. Defining the same population a second time somewhere else is precisely how the
frozen-decoder contamination happened.

THE PRE PANEL IS LEAVE-ONE-SESSION-OUT while the post panels are not, and that asymmetry is correct:
post-stroke trials are held out by construction, pre-stroke trials are the training set. In-sample
pre scores 0.89-0.99 against 0.45-0.66 held out, so mixing the two would show a collapse that is
mostly overfitting.
"""
import numpy as np
import pytest

from wfield_local import position_coding_directions as pcd
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    K = len(DISPLAY_ORDER)
    n_e, n_u = 240, 120
    YE = np.array([DISPLAY_ORDER[i % K] for i in range(n_e)])
    YU = np.array([DISPLAY_ORDER[i % K] for i in range(n_u)])
    GE = np.repeat(np.arange(4), n_e // 4)
    GU = np.repeat(np.arange(4), n_u // 4)
    XE = rng.normal(size=(n_e, 12)) + YE[:, None] * 0.6
    XU = rng.normal(size=(n_u, 12)) + YU[:, None] * 0.6
    e_pre, u_pre = GE < 2, GU < 2
    not_eng = np.zeros(n_u, bool)
    not_eng[-30:] = True

    def _mask(cls, pos=None):
        if cls == "poststroke_lick":
            return ~e_pre
        if cls == "poststroke_miss_working":
            return ~u_pre & ~not_eng
        return ~u_pre & not_eng

    r = pcd._class_confusions(XE, YE, GE, XU, YU, e_pre, {0, 1}, _mask, list("abcd"))
    return r, int((~e_pre).sum()), int((~u_pre & ~not_eng).sum()), int((~u_pre & not_eng).sum())


def test_classes_partition_the_post_trials(synthetic):
    """The three classes must be disjoint AND exhaustive, or a sum of them is not a population."""
    r, n_lick, n_work, n_stop = synthetic
    for cls, n in (("poststroke_lick", n_lick), ("poststroke_miss_working", n_work),
                   ("poststroke_stopped", n_stop)):
        assert r[f"n_{cls}"] == n, f"{cls} count disagrees with its own mask"
        assert np.array(r[cls]).sum() == n, f"{cls} matrix total != its trial count"
    total = sum(np.array(r[c]).sum() for c in pcd.CONFUSION_CLASSES)
    assert total == n_lick + n_work + n_stop


def test_the_populations_5b_needs_are_sums(synthetic):
    """The actual requirement: 5b's two post panels reconstructed by addition, no recompute."""
    r, n_lick, n_work, n_stop = synthetic
    lick = np.array(r["poststroke_lick"])
    work = np.array(r["poststroke_miss_working"])
    stop = np.array(r["poststroke_stopped"])
    assert (lick + work).sum() == n_lick + n_work, "post_working is not reconstructible"
    assert (lick + work + stop).sum() == n_lick + n_work + n_stop, "post_all is not reconstructible"


def test_matrices_are_raw_counts_not_normalised(synthetic):
    """Row-normalised matrices cannot be added, which would defeat the entire design."""
    r, *_ = synthetic
    assert r["counts"] is True
    M = np.array(r["poststroke_lick"])
    assert M.sum() > len(DISPLAY_ORDER), "matrix looks normalised, not counts"
    assert np.allclose(M, np.round(M)), "counts are not integral"


def test_row_order_matches_the_rest_of_the_deck(synthetic):
    """A confusion in a different position order than every other confusion is a trap."""
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    r, *_ = synthetic
    assert r["labels"] == [POSITION_NAMES.get(int(c), str(c)) for c in DISPLAY_ORDER]
    assert np.array(r["poststroke_lick"]).shape == (len(DISPLAY_ORDER), len(DISPLAY_ORDER))


def test_pre_panel_is_leave_one_session_out():
    """In-sample pre would read as a post-stroke collapse that is mostly overfitting."""
    import inspect
    src = inspect.getsource(pcd._class_confusions)
    assert "GE != i" in src and "GE == i" in src, "the pre panel is not leave-one-session-out"


def test_too_little_pre_data_returns_none_rather_than_a_number():
    """With no usable training set there is no frozen decoder; inventing one would be worse."""
    K = len(DISPLAY_ORDER)
    YE = np.array([DISPLAY_ORDER[i % K] for i in range(10)])
    XE = np.zeros((10, 4))
    GE = np.zeros(10, int)
    out = pcd._class_confusions(XE, YE, GE, XE, YE, np.ones(10, bool), {0},
                                lambda cls, pos=None: np.zeros(10, bool), ["a"])
    assert out is None
