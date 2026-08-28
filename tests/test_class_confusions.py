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


@pytest.fixture
def with_rt():
    """The same synthetic data, plus a per-engaged-trial RT so the early/late split is computed."""
    rng = np.random.default_rng(0)
    K = len(DISPLAY_ORDER)
    n_e, n_u = 240, 120
    YE = np.array([DISPLAY_ORDER[i % K] for i in range(n_e)])
    YU = np.array([DISPLAY_ORDER[i % K] for i in range(n_u)])
    GE = np.repeat(np.arange(4), n_e // 4)
    XE = rng.normal(size=(n_e, 12)) + YE[:, None] * 0.6
    XU = rng.normal(size=(n_u, 12)) + YU[:, None] * 0.6
    e_pre, u_pre = GE < 2, np.repeat(np.arange(4), n_u // 4) < 2
    not_eng = np.zeros(n_u, bool)
    not_eng[-30:] = True
    # spread across the boundary on purpose: a split that puts everything on one side would pass a
    # sum check while testing nothing
    rt_e = rng.uniform(0.1, 3.4, size=n_e)

    def _mask(cls, pos=None):
        if cls == "poststroke_lick":
            return ~e_pre
        if cls == "poststroke_miss_working":
            return ~u_pre & ~not_eng
        return ~u_pre & not_eng

    r = pcd._class_confusions(XE, YE, GE, XU, YU, e_pre, {0, 1}, _mask, list("abcd"), rt_e=rt_e)
    return r, rt_e, ~e_pre


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


# --- early vs late rewarded ---------------------------------------------------------------------
# `decode.max_rt_s` is 3.5 s, so one "engaged" class holds a 0.2 s lick and a 3.0 s lick. Post-stroke
# the mass moves late, and that is the distinction the study is about: position coding preserved on
# LATE trials is plan intact / execution slow, degraded on late trials is a different result. The
# split is only trustworthy if it is a genuine PARTITION of the class it refines -- otherwise the two
# panels are two different populations that happen to be drawn side by side.


def test_early_plus_late_reconstructs_the_lick_class_exactly(with_rt):
    """THE acceptance test. Element-wise, not just in total: two matrices can sum to the right
    grand total while individual cells are wrong."""
    r, *_ = with_rt
    lick = np.array(r["poststroke_lick"])
    early = np.array(r["poststroke_lick_early"])
    late = np.array(r["poststroke_lick_late"])
    assert np.array_equal(early + late, lick), "early + late is not the lick class"
    assert r["n_poststroke_lick_early"] + r["n_poststroke_lick_late"] == r["n_poststroke_lick"]


def test_the_split_is_at_the_documented_boundary(with_rt):
    """A boundary that drifts from `RT_SPLIT_S` would make 'late' mean something different here
    than in `nolick_decoder`, which is the whole reason the constant is shared."""
    r, rt_e, post = with_rt
    assert r["rt_split_s"] == pcd.RT_SPLIT_S
    assert r["n_poststroke_lick_early"] == int((post & (rt_e < pcd.RT_SPLIT_S)).sum())
    assert r["n_poststroke_lick_late"] == int((post & (rt_e >= pcd.RT_SPLIT_S)).sum())
    assert min(r["n_poststroke_lick_early"], r["n_poststroke_lick_late"]) > 0, "one arm is empty"


def test_subclasses_are_not_siblings(with_rt):
    """If early/late joined `CONFUSION_CLASSES`, summing that tuple for 'all trials' would count
    every lick trial twice -- silently, and the result would still look like a confusion matrix."""
    r, *_ = with_rt
    subs = pcd.CONFUSION_SUBCLASSES["poststroke_lick"]
    assert not set(subs) & set(pcd.CONFUSION_CLASSES)
    total = sum(np.array(r[c]).sum() for c in pcd.CONFUSION_CLASSES)
    assert total == r["n_poststroke_lick"] + r["n_poststroke_miss_working"] + r["n_poststroke_stopped"]


def test_without_rt_there_is_no_split_rather_than_a_guess(synthetic):
    """The split is optional. A caller that cannot supply RT gets the old keys and nothing else --
    never a boundary inferred from something that is not reaction time."""
    r, *_ = synthetic
    for k in ("poststroke_lick_early", "poststroke_lick_late", "rt_split_s"):
        assert k not in r


def test_rt_alignment_is_length_checked_not_assumed():
    """An RT vector one trial out of step does not fail -- it mislabels the boundary trial of every
    session and still draws a plausible figure. `_rt_engaged` must refuse rather than trim."""
    class F:
        rts = {"a": np.array([0.5, 1.0, 2.5]), "b": np.array([1.1, 3.0])}
    XE = np.zeros((5, 3))
    assert pcd._rt_engaged(F(), ["a", "b"], XE).shape == (5,)
    assert pcd._rt_engaged(F(), ["a", "b"], np.zeros((6, 3))) is None, "a short RT vector was accepted"
    assert pcd._rt_engaged(F(), ["a"], XE) is None, "a missing session was accepted"

    class G:
        rts = {"a": np.array([0.5, np.nan, 2.5])}
    assert pcd._rt_engaged(G(), ["a"], np.zeros((3, 3))) is None, "a non-finite RT was accepted"


def test_rt_leaves_the_feature_builder_rather_than_being_rebuilt():
    """Reconstructing a trial filter outside `_trial_features` is how bugs 15-17 happened; one such
    mask came out 633 long against 575 kept trials. RT must ride out with the features."""
    import inspect

    from wfield_local import locanmf_position_decoder as lpd
    from wfield_local import precue_engagement_states as pes
    assert "with_rt" in inspect.signature(lpd._trial_features).parameters
    assert "with_rt" in inspect.signature(lpd.trial_features_cached).parameters
    assert "with_rt=True" in inspect.getsource(pes.features_with_indices)
    # and the flag must reach the cache key, or a with_rt entry would be served to a caller that
    # unpacks the shorter tuple
    assert "with_rt" in inspect.signature(lpd.feature_cache_kind).parameters


# --- the thin-arm guard --------------------------------------------------------------------------
# MEASURED 2026-08-28: the late arm is 3.4% of post-stroke rewarded trials, and only 26 (PS94) and
# 27 (PS95) in total. A 6x6 on 26 trials is ~4 per row, so a per-position recall there is 0.0 or
# 0.33 by arithmetic rather than by measurement. The figure must say so or a reader takes "late
# decodes at chance" for a result.


def test_a_thin_panel_is_marked_unreadable(tmp_path):
    """The red TOO FEW TO READ title, driven rather than grepped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(DISPLAY_ORDER)
    thin = np.zeros((K, K)); thin[0, 0] = 26          # 26 trials, as PS94 actually has
    fat = np.full((K, K), 50.0)
    res = {"animal": "PS94", "confusions": {
        "labels": [str(c) for c in DISPLAY_ORDER], "counts": True, "rt_split_s": 2.0,
        "prestroke_lick": fat.tolist(), "poststroke_lick": fat.tolist(),
        "poststroke_lick_early": fat.tolist(), "poststroke_lick_late": thin.tolist()}}
    seen = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, axes = orig(*a, **k)
        seen["axes"] = axes
        return fig, axes

    plt.subplots = spy
    try:
        assert pcd.figure_rt_split(res, tmp_path, align="cue") is not None
    finally:
        plt.subplots = orig
    titles = [ax.get_title() for row in seen["axes"] for ax in row]
    marked = [t for t in titles if "TOO FEW TO READ" in t]
    assert len(marked) == 1 and "n=26" in marked[0], f"the thin panel is not marked: {titles}"
    fatt = [t for t in titles if "n=1800" in t]
    assert fatt and not any("TOO FEW" in t for t in fatt), "a well-populated panel was marked"


def test_a_position_with_too_few_trials_gets_no_point():
    """Plotting 0.0 on a row of two trials is worse than plotting nothing: it looks measured.
    FLOOR_TRIALS is the same floor `_stats` and `_cells` use, so a cell too thin to report here is
    too thin to report anywhere in this module."""
    import inspect
    src = inspect.getsource(pcd.figure_rt_split)
    assert "rows >= FLOOR_TRIALS" in src, "the recall panel draws points on empty/near-empty rows"
    assert "rows >= MIN_TRIALS" in src, "there is no hollow-marker threshold"


def test_the_module_no_longer_predicts_a_late_shift():
    """The design assumed post-stroke mass moves late; the data says 3.4%. A comment asserting the
    refuted version is the exact failure this session has been fixing."""
    import inspect
    src = inspect.getsource(pcd)
    head = src[:src.index("def _rt_engaged")]
    assert "Post-stroke the mass moves late, and telling" not in head, (
        "the module still states the prediction the measurement refuted")
    assert "3.4%" in head, "the measured late fraction is not recorded where the constant is defined"
