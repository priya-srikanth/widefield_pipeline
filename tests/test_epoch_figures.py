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
