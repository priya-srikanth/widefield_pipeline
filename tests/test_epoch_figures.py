"""Epoch pooling aggregates the existing collectors; it does not recompute, and it cannot disagree.

Priya, 2026-08-28: "I presume we can reuse saved processed session information from all the other
existing grant analyses -- I do NOT want this to have to re-create the wheel." That is the design and
these tests hold it: the epoch figures group `grant_figures`' per-day records and sum them, so they
hold the SAME objects the per-animal figures do and a second definition of any population cannot
appear.
"""
import numpy as np
import pytest

from wfield_local import config, epoch_figures as ef, epochs


def _fake(animal_days):
    """{animal: (pre_record, {day: record})} with recognisable per-day trial counts."""
    out = {}
    for an, days in animal_days.items():
        pre = (np.zeros(40, int), np.zeros(40, int), np.arange(40) % 4)
        by = {d: (np.full(n, 1), np.full(n, 1), np.arange(n) % 3) for d, n in days.items()}
        out[an] = (pre, by)
    return out


def test_the_two_day_definitions_agree_where_epochs_are_defined():
    """POST-stroke only, which is the whole domain an epoch boundary lives in.

    `grant_figures._day` is month*31 ORDERING and `epochs.days_since_stroke` is a calendar
    difference. Every post-stroke session sits in August, which has 31 days, so they agree exactly --
    and a September session would divide them, which is when an epoch boundary could otherwise move
    silently under a published figure. This is the guard for that.
    """
    from wfield_local.grant_figures import _day
    post = [l for l in config.pooled_labels() if epochs.epoch_of(l) != "pre"]
    assert post
    for lab in post:
        an, mmdd = config.animal_of(lab), lab.split("_")[-1]
        assert _day(an, mmdd) == epochs.days_since_stroke(lab), lab


def test_they_DISAGREE_before_the_stroke_and_that_is_harmless():
    """Pinned so nobody "fixes" one to match the other without knowing what they are changing.

    Month*31 accumulates a day of error per short month, so June dates come out one lower than the
    calendar: PS92_0606 is -73 by `_day` and -72 by calendar. It is harmless HERE because every
    pre-stroke day is negative and both therefore say 'pre'; it matters only as the x position of a
    pre-stroke point on `fig_behaviour`, whose own docstring records the approximation. Changing
    `_day` to calendar arithmetic would move published x positions on figures 1 and 1b.
    """
    from wfield_local.grant_figures import _day
    pre = [l for l in config.pooled_labels() if epochs.epoch_of(l) == "pre"]
    diffs = {_day(config.animal_of(l), l.split("_")[-1]) - epochs.days_since_stroke(l) for l in pre}
    assert diffs <= {0, -1}, f"an unexpected divergence appeared: {diffs}"
    for lab in pre:
        an = config.animal_of(lab)
        assert ef.epoch_of_day(an, _day(an, lab.split("_")[-1])) == "pre"
        assert epochs.epoch_of(lab) == "pre"


def test_epoch_of_day_matches_epoch_of_label():
    """The collectors key by day and `epochs` keys by label; one definition, two entry points."""
    for lab in config.pooled_labels():
        an = config.animal_of(lab)
        assert ef.epoch_of_day(an, epochs.days_since_stroke(lab)) == epochs.epoch_of(lab), lab


def test_a_day_in_the_gap_is_dropped_not_rounded():
    """PS92 day 6 and PS94 days 6 and 8 have no session today, but a future one must not be folded
    into whichever epoch is nearer -- that would invent a boundary nobody set."""
    assert ef.epoch_of_day("PS92", 6) is None
    assert ef.epoch_of_day("PS94", 8) is None
    per = _fake({"PS92": {5: 10, 6: 10, 7: 10}})
    grouped = ef.group_days_by_epoch(per)
    assert ("PS92", 6) not in grouped["acute"] + grouped["subacute"]
    assert ef.epoch_coverage(per)["unassigned"] == [("PS92", 6)]


def test_pooling_is_a_concatenation_of_the_same_records():
    """No recompute: the pooled epoch holds exactly the trials the per-day records hold."""
    per = _fake({"PS92": {1: 7, 2: 5}, "PS95": {1: 3}})
    y, p, b = ef.pool_records(per, "acute")
    assert len(y) == 7 + 5 + 3, "pooled trial count is not the sum of its days"
    assert len(p) == len(y) and len(b) == len(y)


def test_blocks_are_disambiguated_across_animals_and_days():
    """Two animals' block 3 are different clusters. Colliding them would let a block bootstrap treat
    unrelated trials as one correlated group and understate the interval."""
    per = _fake({"PS92": {1: 6}, "PS95": {1: 6}})
    _y, _p, b = ef.pool_records(per, "acute")
    assert len(set(b.tolist())) == 6, "block ids collided across animals"


def test_an_empty_epoch_returns_none_rather_than_an_empty_matrix():
    """An empty panel and a panel of zeros look identical once drawn; only one is honest."""
    per = _fake({"PS92": {1: 5}})
    assert ef.pool_records(per, "subacute") is None


def test_coverage_reports_the_imbalance_a_panel_must_state():
    """PS95 contributes ONE acute session and PS92 TWO subacute. A pooled panel that does not say so
    implies four equal contributors when one animal can dominate."""
    per = _fake({an: {epochs.days_since_stroke(l): 5
                      for l in config.pooled_labels(an) if epochs.epoch_of(l) != "pre"}
                 for an in ("PS92", "PS93", "PS94", "PS95")})
    cov = ef.epoch_coverage(per)
    assert cov["per_epoch"]["acute"] == {"PS92": 5, "PS93": 4, "PS94": 6, "PS95": 1}
    assert cov["per_epoch"]["subacute"] == {"PS92": 2, "PS93": 3, "PS94": 2, "PS95": 7}
    assert cov["unassigned"] == []


def test_behaviour_is_weighted_by_SESSION_as_asked(monkeypatch):
    """Priya, 2026-08-28: "weighted mean (ie just use each session's value)". Hits and trials sum
    over every session in the epoch, so a six-session animal outweighs a one-session animal. The
    alternative (mean of per-animal means) differs most in the acute panel, which is exactly where
    PS95 has one session and PS94 has six -- so the choice has to be visible, not implicit."""
    POS = ["far_R"]

    def sessions_of(an):
        return [(l.split("_")[-1], epochs.days_since_stroke(l)) for l in config.pooled_labels(an)]

    def metrics(an, mmdd):
        lab = f"{an}_{mmdd}"
        e = epochs.epoch_of(lab)
        rate = {"pre": 0.9, "acute": 0.1, "subacute": 0.5}.get(e, 0.0)
        n = 100 if an == "PS94" else 10          # PS94 dominates by trial count too
        return {"far_R": (rate, rate, rate, n)}

    out = ef.behaviour_by_epoch(metrics, sessions_of, positions=POS)
    for e, want in (("acute", 0.1), ("subacute", 0.5), ("pre", 0.9)):
        r, _lo, _hi, n, ns = out[e]["far_R"]
        assert r == pytest.approx(want, abs=0.02), f"{e}: {r}"
        assert ns == len([l for l in config.pooled_labels() if epochs.epoch_of(l) == e])
        assert n > 0


def test_it_adds_no_second_definition_of_any_population():
    """The reuse contract. This module must not fit a model, pool sessions, or build features -- if
    it did, its panels and the per-animal figures could drift apart while both looked defensible,
    which is how the frozen-decoder contamination survived eight days."""
    import inspect
    src = inspect.getsource(ef)
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for forbidden in ("pool_sessions(", ".fit(", "_trial_features", "make_pipeline"):
        assert forbidden not in body, f"epoch_figures recomputes ({forbidden}); it must aggregate"


# --- showing the imbalance rather than asserting it ----------------------------------------------
# Priya, 2026-08-28: one dot per session value, four colours, transparency for overlap. A note about
# session weighting is read once; a column of dots is read every time the figure is looked at.


def _ax():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots()[1]


def test_one_dot_per_session_value():
    ax = _ax()
    pts = [("PS92", 0.4), ("PS92", 0.5), ("PS94", 0.6)]
    assert ef.session_points(ax, 1.0, pts) == 3
    assert sum(len(c.get_offsets()) for c in ax.collections) == 3


def test_colours_come_from_the_single_source_of_truth():
    """The same animal must be the same colour here and in every per-animal figure beside it.
    Four colours chosen in this module would drift from `config.animal_color` the first time one
    changed."""
    import inspect
    assert "config.animal_color()" in inspect.getsource(ef.session_points)
    ax = _ax()
    ef.session_points(ax, 0.0, [("PS93", 0.5)])
    import matplotlib.colors as mc
    want = mc.to_rgb(config.animal_color()["PS93"])
    got = tuple(ax.collections[0].get_facecolor()[0][:3])
    assert all(abs(a - b) < 1e-6 for a, b in zip(want, got))


def test_the_spread_is_deterministic():
    """A random jitter makes two renders of the same data impossible to compare. Points are laid out
    by index, so the same data always draws the same picture."""
    pts = [("PS92", 0.4), ("PS93", 0.5), ("PS94", 0.6), ("PS95", 0.7)]
    xs = []
    for _ in range(2):
        ax = _ax()
        ef.session_points(ax, 2.0, pts)
        xs.append([tuple(c.get_offsets()[0]) for c in ax.collections])
    assert xs[0] == xs[1]


def test_a_single_session_sits_on_the_tick():
    """PS95 contributes ONE acute session. Offsetting a lone dot would misplace exactly the case the
    overlay exists to make visible."""
    ax = _ax()
    ef.session_points(ax, 3.0, [("PS95", 0.2)])
    assert ax.collections[0].get_offsets()[0][0] == pytest.approx(3.0)


def test_missing_values_are_skipped_not_plotted_as_zero():
    ax = _ax()
    assert ef.session_points(ax, 0.0, [("PS92", None), ("PS93", float("nan")), ("PS94", 0.3)]) == 1


def test_the_legend_can_carry_the_session_counts():
    """`PS95 (n=1)` is where the imbalance stops being merely visible and becomes unmissable."""
    ax = _ax()
    ef.animal_legend(ax, counts={"PS92": 5, "PS93": 4, "PS94": 6, "PS95": 1})
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "PS95 (n=1)" in labels and "PS94 (n=6)" in labels


def test_per_session_values_returns_one_entry_per_session():
    """The bridge from the collectors to the dots: one value per session, tagged with its animal, so
    the panel's dots and its pooled bar describe the same sessions."""
    per = _fake({"PS92": {1: 6, 2: 6}, "PS95": {1: 6}})
    got = ef.per_session_values(per, "acute", lambda an, day, rec: float(len(rec[0])))
    assert sorted(got) == [("PS92", 6.0), ("PS92", 6.0), ("PS95", 6.0)]
    assert len(ef.per_session_values(per, "pre", lambda a, d, r: 1.0)) == 2


# --------------------------------------------------------------------------- layout, measured

def _counts_matrix(k, seed, n):
    rng = np.random.default_rng(seed)
    M = rng.random((k, k)) + np.eye(k) * k
    return np.round(M / M.sum() * n)


def _row(tmp_path, **kw):
    """Render one `confusion_row` and hand back the Figure it saved, for measuring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure

    from wfield_local.grant_figures import CONF_LABELS, _short

    k = len(CONF_LABELS)
    counts = {"pre": _counts_matrix(k, 1, 9800), "acute": _counts_matrix(k, 2, 5200),
              "subacute": _counts_matrix(k, 3, 4600)}
    seen = []
    orig = matplotlib.figure.Figure.savefig

    def spy(self, *a, **k_):
        seen.append(self)
        return orig(self, *a, **k_)

    matplotlib.figure.Figure.savefig = spy
    try:
        ef.confusion_row(counts, tmp_path, name="probe", title="probe title",
                         labels=_short(CONF_LABELS), **kw)
    finally:
        matplotlib.figure.Figure.savefig = orig
    return seen[-1]


@pytest.mark.parametrize("delta", [False, True])
def test_the_canvas_is_the_placed_size_whatever_the_panel_count(tmp_path, delta):
    """A point size written in this module is the point size the reader gets.

    WHY THIS IS PINNED. The first version sized the canvas per panel -- 6.2in for three panels,
    8.27in for four -- so the deck, placing both in the same quarter-page column, rendered the
    four-panel row's ticks at 6.0pt against the three-panel row's 8.0pt. Two figures side by side
    with a 25% type difference, and `_overlaps` reported BOTH clean, because nothing collided. A
    layout fault that produces no collision is invisible to every check except this one.
    """
    fig = _row(tmp_path, delta=delta, chance=1 / 6)
    assert fig.get_size_inches()[0] == pytest.approx(ef.QUARTER_IN)


@pytest.mark.parametrize("delta,annotate", [(False, False), (True, False), (True, True)])
def test_nothing_collides_at_a_quarter_page(tmp_path, delta, annotate):
    """Driven, not read. Three layout faults this month were found only by rendering the figure.

    Uses the REAL position names: the digit fallback is far narrower than `far_center` rotated and
    could not find a crowding fault even where one existed.
    """
    from wfield_local.grant_figures import _overlaps

    fig = _row(tmp_path, delta=delta, annotate=annotate, chance=1 / 6)
    bad = _overlaps(fig)
    assert not bad, "; ".join(f"{a} {b}" for a, b, _ in bad[:6])


def test_the_delta_panels_are_measured_against_pre_not_against_each_other(tmp_path):
    """Priya, 2026-08-28: acute - pre AND subacute - pre.

    `subacute - acute` answers "did it recover from its worst point"; `subacute - pre` answers
    "has it returned to baseline", which is what a recovery figure is asked and what a reader
    assumes unless told otherwise. Sharing one reference also makes the two delta panels directly
    comparable to each other, which they are not when one is measured against pre and the other
    against acute.
    """
    fig = _row(tmp_path, delta=True)
    titles = [ax.get_title() for ax in fig.axes]
    assert "acute - pre" in titles and "subacute - pre" in titles, titles
    assert not any("subacute - acute" in t for t in titles), titles


def test_the_signed_panels_carry_a_scale(tmp_path):
    """A signed heatmap with no scale is a picture of signs: red at 0.03 looks like red at 0.30.

    The scale used to be printed in the delta title; it now lives on a colour bar, one per row,
    because the two rows are in different units -- recall in [0, 1] above, a signed change below --
    and a single shared bar would be wrong for one of them.
    """
    fig = _row(tmp_path, delta=True)
    labels = [ax.get_ylabel() for ax in fig.axes] + [ax.get_xlabel() for ax in fig.axes]
    assert any("change in recall" in x for x in labels), labels
    assert any("recall" == x for x in labels), labels


def test_every_drawn_row_has_its_positions_named(tmp_path):
    """The delta row starts under `acute`, so keying the y labels to column 0 left the whole row
    unlabelled -- six unnamed rows of a signed matrix, and no overlap check can see it."""
    fig = _row(tmp_path, delta=True)
    labelled = [ax for ax in fig.axes
                if [t.get_text() for t in ax.get_yticklabels() if t.get_text().strip()]]
    assert len(labelled) >= 2, "only one row of panels carries y tick labels"


def test_the_positions_are_named_not_numbered(tmp_path):
    """`labels` is a parameter because this renderer has no business deciding what the rows mean."""
    from wfield_local.grant_figures import CONF_LABELS, _short

    fig = _row(tmp_path, delta=False)
    got = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert got == _short(CONF_LABELS), got


# ---------------------------------------------------------- anatomical position labels

def test_ipsi_and_contra_come_from_the_lesion_side():
    """Priya, 2026-08-28: derive them, "in case of a change in stroke location".

    Every animal in this cohort is lesioned on the LEFT, so left maps to ipsi and right to contra
    uniformly and the rename is a rename. The data agrees rather than merely permitting it: the
    collapse is at far_R in all four animals, the position contralateral to a left lesion.
    """
    from wfield_local.grant_figures import CONF_LABELS

    assert ef.lesion_side() == "L"
    assert ef.anatomical_labels(CONF_LABELS) == ["nI", "nM", "nC", "fI", "fM", "fC"]
    assert ef.anatomical_labels(CONF_LABELS, short=False)[-1] == "far contra"


def test_a_right_lesion_inverts_the_mapping(monkeypatch):
    """The whole reason it is derived. A right-lesioned animal's far_R is IPSI."""
    monkeypatch.setattr(config, "animals", lambda: {"PSxx": {"stroke_laterality": "R"}})
    from wfield_local.grant_figures import CONF_LABELS
    assert ef.anatomical_labels(CONF_LABELS) == ["nC", "nM", "nI", "fC", "fM", "fI"]


def test_a_mixed_cohort_refuses_to_label_a_pooled_figure(monkeypatch):
    """Averaging ipsi with contra under one name is a wrong number wearing a correct-looking
    label. It must raise, not guess -- reflecting the position axis per animal is a real analysis
    change and has to be deliberate, not a silent consequence of adding a mouse."""
    monkeypatch.setattr(config, "animals",
                        lambda: {"A": {"stroke_laterality": "L"}, "B": {"stroke_laterality": "R"}})
    from wfield_local.grant_figures import CONF_LABELS
    with pytest.raises(ValueError, match="mixed lesion sides"):
        ef.anatomical_labels(CONF_LABELS)


def test_missing_laterality_raises_rather_than_defaulting(monkeypatch):
    monkeypatch.setattr(config, "animals", lambda: {"A": {}, "B": {}})
    from wfield_local.grant_figures import CONF_LABELS
    with pytest.raises(ValueError, match="stroke_laterality"):
        ef.anatomical_labels(CONF_LABELS)


# --------------------------------------------------- animals -> sessions -> blocks bootstrap

def _rec(rng, n_blocks, per_block, acc, k=6):
    """A (y_true, y_pred, blocks) record at a known accuracy, with real block structure."""
    y, p, b = [], [], []
    for blk in range(n_blocks):
        q = blk % k
        yy = np.full(per_block, q)
        pp = np.where(rng.random(per_block) < acc, q, (q + 1) % k)
        y.append(yy); p.append(pp); b.append(np.full(per_block, blk))
    return np.concatenate(y), np.concatenate(p), np.concatenate(b)


def _acc(y, p):
    return float(np.mean(y == p)) if len(y) else None


def test_the_contrast_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    per = {}
    for an in ("PS92", "PS93", "PS94", "PS95"):
        per[an] = (_rec(rng, 30, 6, 0.90), {1: _rec(rng, 30, 6, 0.50), 2: _rec(rng, 30, 6, 0.50)})
    got = ef.contrast_ci(per, "acute", "pre", _acc, rng=np.random.default_rng(1), n_boot=200)
    point, lo, hi, n = got
    assert -0.55 < point < -0.25, point
    assert lo < point < hi and hi < 0, (lo, point, hi)


def test_the_point_estimate_comes_from_the_data_not_the_bootstrap():
    """A figure prints the point; it must be the number in the data, not the resample mean."""
    rng = np.random.default_rng(2)
    per = {a: (_rec(rng, 20, 6, 0.9), {1: _rec(rng, 20, 6, 0.6)})
           for a in ("PS92", "PS93", "PS94", "PS95")}
    direct = _acc(*ef._pooled(per, "acute")) - _acc(*ef._pooled(per, "pre"))
    point, *_ = ef.contrast_ci(per, "acute", "pre", _acc, rng=np.random.default_rng(3), n_boot=120)
    assert point == pytest.approx(direct)


def test_blocks_are_the_unit_inside_a_session():
    """Trials inside one block share a position and a moment; resampling them independently would
    treat them as independent and understate the interval."""
    rng = np.random.default_rng(4)
    rec = _rec(rng, 12, 5, 0.8)
    blocks = ef._blocks_of(rec)
    assert len(blocks) == 12 and all(len(y) == 5 for y, _p in blocks)


def test_no_session_in_one_arm_does_not_crash_the_contrast():
    """An animal can contribute to one epoch and not the other -- PS95 has a single acute session
    and it could as easily have had none."""
    rng = np.random.default_rng(5)
    per = {"PS92": (_rec(rng, 20, 6, 0.9), {1: _rec(rng, 20, 6, 0.5)}),
           "PS93": (_rec(rng, 20, 6, 0.9), {}),           # no post-stroke session at all
           "PS94": (_rec(rng, 20, 6, 0.9), {1: _rec(rng, 20, 6, 0.5)})}
    got = ef.contrast_ci(per, "acute", "pre", _acc, rng=np.random.default_rng(6), n_boot=120)
    assert got is not None and got[1] < got[0] < got[2]


def test_every_per_arm_key_reaches_the_arm_loop():
    """`--only mat` and `--only scal` must not silently produce nothing.

    The arm loop is skipped entirely when none of its keys is wanted, and that guard listed only
    two of the four. `--only scal` therefore broke out immediately: empty output directory, exit 0,
    no error and no report. A completeness count cannot see this -- there is nothing to count.
    """
    import inspect

    from wfield_local import epoch_grant_figures as eg

    src = inspect.getsource(eg.main)
    keys = set(eg.MATRIX_FAMILIES and ["mat"]) | {"scal", "acc", "5c"}
    assert "ARM_KEYS" in src, "the arm-loop guard should be one named set, not a repeated literal"
    for k in keys:
        assert f'"{k}"' in src, f"{k} is not reachable from main()"


def test_a_bar_without_an_interval_is_a_bare_number_and_draws_nothing():
    """Why 8g, 10, 11 and the accuracy figures had no error bars.

    `bar_row` draws one only where a value arrives as a (value, lo, hi) tuple. Every family except
    behaviour passed a plain float, so the bars had no interval and nothing said so -- no warning,
    no gap, just an absent whisker on a figure that looked complete.
    """
    v, lo, hi = ef._value_and_ci(0.42)
    assert (v, lo, hi) == (0.42, None, None)
    v, lo, hi = ef._value_and_ci((0.42, 0.30, 0.55))
    assert lo == 0.30 and hi == 0.55


def test_value_draws_and_contrast_draws_share_one_resampling():
    """A bar's interval and the mark above it must not come from two different schemes."""
    rng = np.random.default_rng(3)
    per = {a: (_rec(rng, 20, 6, 0.9), {1: _rec(rng, 20, 6, 0.5)})
           for a in ("PS92", "PS93", "PS94", "PS95")}
    got = ef.value_draws(per, "acute", _acc, rng=np.random.default_rng(4), n_boot=200)
    assert got is not None
    point, draws = got
    assert 0.35 < point < 0.65
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert lo < point < hi
    assert ef.with_ci(got)[0] == pytest.approx(point)


def test_scalar_value_draws_gives_every_epoch_a_bar_interval():
    points = {"pre": {"k": [("PS92", 0.9), ("PS92", 0.88), ("PS93", 0.92)]},
              "acute": {"k": [("PS92", 0.5), ("PS93", 0.55)]}}
    for e in ("pre", "acute"):
        got = ef.scalar_value_draws(points, e, "k", rng=np.random.default_rng(5), n_boot=200)
        ci = ef.with_ci(got)
        assert ci is not None and ci[1] <= ci[0] <= ci[2]


# ------------------------------------------------------- running off the canvas

def test_off_canvas_sees_what_overlaps_cannot(tmp_path):
    """A label that collides with nothing but runs past the edge.

    `_overlaps` compares artists to EACH OTHER, so a lone label is clean however far past the edge
    it goes. That is how a rotated y label two inches long on a 1.55in axes, and a six-inch title
    on a 2.58in canvas, both shipped looking finished. Priya, 2026-08-28.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_ylabel("x" * 200, fontsize=20)          # far longer than the canvas is tall
    bad = ef.off_canvas(fig)
    plt.close(fig)
    assert bad, "a label running off the canvas was not detected"
    assert any(r[0].endswith("ylabel") for r in bad)
    assert max(abs(r[2]) for r in bad) > 0.5       # reported in INCHES over, not as a boolean


def test_long_text_is_wrapped_to_the_thing_that_bounds_it():
    """Titles are bounded by the canvas WIDTH, rotated y labels by the axes HEIGHT."""
    t, n = ef.wrap_title("word " * 60, 2.58, 8.5)
    assert n > 1 and max(len(x) for x in t.split(chr(10))) * 0.5 * 8.5 / 72 <= 2.58
    y, m = ef.wrap_ylabel("best-match accuracy (chance 0.17)", 1.6, 9.0)
    assert m == 2 and chr(10) in y
    short, one = ef.wrap_ylabel("hit rate", 1.6, 9.0)
    assert one == 1 and short == "hit rate"


def test_a_narrow_figure_is_narrower(tmp_path):
    """One bar group should not reserve the canvas six need -- but not at the plot's expense.

    The width floor makes one and two groups the SAME width, which is why this no longer asserts
    strict monotonicity across all three. Narrowing past the floor took the space out of the axes
    rather than out of the empty margin: the gutters are absolute, so a 2.58in canvas left a 0.57in
    plot. Both halves matter, so both are asserted.
    """
    assert ef.bar_figure_width(6) == pytest.approx(ef.QUARTER_IN, abs=0.05)
    assert ef.bar_figure_width(1) == ef.bar_figure_width(2) < ef.bar_figure_width(6)
    assert ef.bar_figure_width(1) < 3.6
    for n in (1, 2, 6):
        w = ef.bar_figure_width(n)
        assert w - ef.BAR_LEFT_IN - ef.BAR_RIGHT_IN >= ef.BAR_AXES_MIN_W_IN - 1e-9


def test_the_canvas_grows_with_its_text_instead_of_shrinking_the_plot(tmp_path):
    """A taller top band must make the FIGURE taller, never the data region shorter.

    The regression this pins: `fig_h` was the constant 2.60 and the axes took what the text left,
    so the figure carrying the most chrome got the least plot -- 8g ran a 0.68in axes under 1.92in
    of title, subtitle and two-level x labels. Priya, 2026-08-28: "the 8g figures are too short".
    """
    long_title = ("Geometry preserved against the pre-stroke reference, per position -- "
                  "ENL (pre-cue), lick + miss-while-working")
    two_line_sub = "N=4 animals, n=74 sessions" + chr(10) + "bootstrap: animals -> sessions"
    keys = ["nI", "nM", "nC", "fI", "fM", "fC"]
    vals = {e: {k: (0.5, 0.4, 0.6) for k in keys} for e in ("pre", "acute", "subacute")}
    heights, axes_h = [], []
    for title, sub, groups in ((("short"), None, None),
                               (long_title, two_line_sub,
                                [("Near", 0, 2), ("Far", 3, 5)])):
        p = ef.bar_row(vals, tmp_path, name=f"g{len(heights)}", ylabel="accuracy",
                       positions=keys, title=title, subtitle=sub, groups=groups,
                       tick_labels=(["Ipsi", "Middle", "Contra"] * 2 if groups else None))
        assert p is not None
        import matplotlib.image as mpimg
        im = mpimg.imread(str(p))
        heights.append(im.shape[0])
        axes_h.append(ef.bar_axes_height(ef.bar_figure_width(len(keys))))
    # more text -> taller canvas ...
    assert heights[1] > heights[0]
    # ... and the data region is untouched by how much text sits above it
    assert axes_h[0] == pytest.approx(axes_h[1])
    assert axes_h[0] >= ef.BAR_AXES_MIN_IN
