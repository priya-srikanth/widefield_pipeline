"""The diagnostic figures render, and the numbers behind them mean what the slide notes say.

WHY SYNTHETIC. Each of these figures needs `pool_sessions` over a whole animal to draw for real --
about ten minutes of loading per animal -- so a bug in the plotting code would only surface at the
end of a nightly. Every one of them started life as a throwaway script, which is precisely the code
that reaches a deck untested.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import position_coding_directions as pcd

pytest.importorskip("matplotlib")

POS = pcd.BY_SEVERITY


def _res(animal, slopes=None, cosines=None, post=True):
    """A minimal run_animal() return carrying only what the diagnostic figures read."""
    slopes = slopes or dict.fromkeys(POS, -0.2)
    cosines = cosines or {p: (0.6 if p.startswith("close") else -0.6) for p in POS}
    positions = {}
    for p in POS:
        cell = {"cos_closefar": cosines[p],
                "prestroke_lick_by_quartile": [1.2, 1.0, 0.9, 1.2 + slopes[p]],
                "prestroke_lick_slope": slopes[p], "n_post": 80}
        if post:
            cell |= {"post_raw": 0.8, "post_unit": 0.75, "norm_ratio": 1.05}
        positions[p] = cell
    pw = [{"A": a, "B": b, "same_ring": pcd._ring(a) == pcd._ring(b),
           "cos_closefar": (0.3 if pcd._ring(a) == pcd._ring(b) else -0.8),
           "by_quartile": [0.1, 0.2, 0.3, 0.4], "slope": 0.3}
          for i, a in enumerate(POS) for b in POS[i + 1:]]
    rq = {ph: {p: [{"rate": r, "n": 60} for r in (0.99, 0.98, 0.95, 0.7)] for p in POS}
          for ph in ("pre", "post")}
    return {"animal": animal, "align": "lick", "window": "lick", "response_by_quartile": rq,
            "methods": {"dom_orth": {"diagnostics": {"positions": positions, "pairwise": pw,
                                                     "closefar_engagement_cos": -0.2}}}}


def test_each_diagnostic_figure_renders(tmp_path):
    r = _res("PS94")
    assert pcd.figure_engagement(r, tmp_path, align="lick") is not None
    assert pcd.figure_norm_unit(r, tmp_path, align="lick", meth="dom_orth") is not None
    every = {a: _res(a) for a in ("PS92", "PS93", "PS94", "PS95")}
    assert pcd.figure_cos_slope(every, tmp_path, align="lick", meth="dom_orth") is not None
    assert pcd.figure_pairwise_split(every, tmp_path, align="lick", meth="dom_orth") is not None
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_a_figure_with_nothing_to_draw_returns_none_rather_than_an_empty_png(tmp_path):
    """An empty axes in a deck reads as 'measured, found nothing', which is a different claim."""
    bare = {"animal": "PS92", "methods": {"dom_orth": {"diagnostics": {}}}}
    assert pcd.figure_engagement(bare, tmp_path, align="lick") is None
    assert pcd.figure_norm_unit(bare, tmp_path, align="lick", meth="dom_orth") is None
    assert pcd.figure_cos_slope({"PS92": bare}, tmp_path, align="lick", meth="dom_orth") is None
    assert pcd.figure_pairwise_split({"PS92": bare}, tmp_path, align="lick", meth="dom_orth") is None
    assert not list(tmp_path.glob("*.png"))


def test_post_stroke_only_animals_do_not_break_the_norm_unit_figure(tmp_path):
    """A position the animal never licked post-stroke has no post_raw, and must simply be absent."""
    r = _res("PS94", post=False)
    assert pcd.figure_norm_unit(r, tmp_path, align="lick", meth="dom_orth") is None


def test_ring_membership_matches_the_cohort_geometry():
    """`_ring` is what splits within- from cross-ring, so it must agree with the spout grid."""
    from wfield_local.spout_behavior import POSITIONS

    for q in POSITIONS:
        assert pcd._ring(q["name"]) == q["ring"], q["name"]


def test_slope_needs_three_quartiles_not_two():
    """A 'drift' from two bins is a difference, not a trend -- and the last quartile is exactly the
    one that empties when an animal stops responding, so two-point slopes would be common."""
    assert pcd._slope([1.0, None, None, 0.2]) is None
    assert pcd._slope([1.0, 0.8, 0.6, None]) == pytest.approx(-0.4)
    assert pcd._slope([1.0, 0.8, 0.6, 0.2]) == pytest.approx(-0.8)


def test_by_quartile_bins_cover_the_whole_session_including_the_last_trial():
    """The top bin is closed at 1.01, so a trial at fraction exactly 1.0 is counted rather than
    silently dropped -- there is always exactly one such trial per session."""
    frac = np.array([0.0, 0.3, 0.6, 1.0])
    got = pcd._by_quartile(frac, np.ones(4, bool), lambda m: int(m.sum()))
    assert got == [1, 1, 1, 1]


def test_a_failing_figure_does_not_take_the_run_down(capsys):
    """The analysis is the expensive part and must survive its own presentation layer.

    A one-line argument error in the cohort figure loop killed main() after every per-animal PNG was
    written and before coding_direction.json, discarding ~40 minutes of pooling (2026-08-22).
    """
    def boom(*_a, **_kw):
        raise TypeError("animal_color() takes 0 positional arguments but 1 was given")

    assert pcd._draw(boom, {}, "/nowhere") is None
    assert "!! boom: TypeError" in capsys.readouterr().out


def _res_sessions(animal="PS94", n_post=3):
    """A run_animal() return carrying the per-session pairwise and cross blocks."""
    r = _res(animal)
    R = r["methods"]["dom_orth"]
    labs = [f"{animal}_08{17 + i}" for i in range(n_post)]
    pairs = [f"{a}|{b}" for a in POS for b in POS if a != b]
    R["pairwise_axes"] = {k: {"spread": 1.4, "n_A": 200, "n_B": 200} for k in pairs}
    cell = {"n": 40, "mean": 0.6, "sem": 0.1, "sd": 0.6, "low_n": False}
    R["pairwise"] = {c: dict.fromkeys(pairs, cell)
                     for c in ("prestroke_lick", "poststroke_lick", "poststroke_miss_working")}
    R["pairwise_by_session"] = {
        "poststroke_miss_working": {
            k: {lab: {"n": 30, "mean": 0.4 + 0.1 * j, "sem": 0.12, "sd": 0.6, "low_n": False}
                for j, lab in enumerate(labs)} for k in pairs}}
    R["cross_matrix"] = {"prestroke_lick": {P: dict.fromkeys(POS, cell) for P in POS}}
    R["cross_by_session"] = {
        "poststroke_miss_working": {lab: {P: dict.fromkeys(POS, cell) for P in POS}
                                    for lab in labs}}
    return r, labs


def test_per_session_pairwise_draws_one_series_per_post_session(tmp_path):
    """The pooled figure averages a moving target; this is the same quantity resolved in time."""
    r, _labs = _res_sessions()
    q = pcd.figure_pairwise_sessions(r, tmp_path, align="lick", meth="dom_orth",
                                     cls="poststroke_miss_working")
    assert q is not None and q.exists()
    assert "poststroke_miss_working" in q.name and "pairsess" in q.name


def test_per_session_cross_matrix_draws_one_matrix_per_session(tmp_path):
    r, _labs = _res_sessions(n_post=4)
    q = pcd.figure_cross_sessions(r, tmp_path, align="lick", meth="dom_orth",
                                  cls="poststroke_miss_working")
    assert q is not None and q.exists() and "crosssess" in q.name


def test_a_class_with_no_per_session_data_returns_none(tmp_path):
    """An empty axes reads as "measured, found nothing", which is a different claim."""
    r, _labs = _res_sessions()
    for fn in (pcd.figure_pairwise_sessions, pcd.figure_cross_sessions):
        assert fn(r, tmp_path, align="lick", meth="dom_orth", cls="poststroke_stopped") is None
    assert not list(tmp_path.glob("*poststroke_stopped*"))


def test_the_per_session_cross_matrix_is_a_DIFFERENCE_from_the_baseline(tmp_path):
    """Raw off-diagonal magnitude means nothing -- neighbouring positions are already similar
    pre-stroke (far_center scores 0.76 on the far_R direction). A session identical to the baseline
    must therefore render as all-zero difference rather than as a strong pattern."""
    r, _labs = _res_sessions(n_post=1)
    assert pcd.figure_cross_sessions(r, tmp_path, align="lick", meth="dom_orth",
                                     cls="poststroke_miss_working") is not None
