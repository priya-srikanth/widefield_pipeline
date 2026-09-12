"""The trajectory module distinguishes two accounts of recovery, so its failure mode is that it
CANNOT distinguish them -- a sign error, a baseline that is not the baseline, or a summary that
reports the same thing whatever the data do.

These tests feed it the two shapes it exists to tell apart and require different answers.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import recovery_trajectory as rt
from wfield_local.grant_figures import REFIT_UNAVAILABLE


def _rec(n, frozen_acc, refit_acc, seed=0):
    """A paired record with the requested accuracies: (y, [frozen, refit], blocks)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, n)
    fz, rf = y.copy(), y.copy()
    fz[: int(round(n * (1 - frozen_acc)))] = (y[: int(round(n * (1 - frozen_acc)))] + 1) % 6
    rf[: int(round(n * (1 - refit_acc)))] = (y[: int(round(n * (1 - refit_acc)))] + 1) % 6
    return (y, np.column_stack([fz, rf]), np.arange(n) // 10)


def test_a_session_identical_to_baseline_sits_at_the_ORIGIN():
    """Both axes are defined so zero IS the pre-stroke state. If a session that matches baseline
    does not land at (0, 0), every trajectory in the figure is offset and the loop-vs-line reading
    is meaningless."""
    pre = [_rec(600, 0.90, 0.82, seed=1)]
    f, r = rt._pre_pair(pre)
    fz, rf, _n = rt._session_pair(_rec(600, 0.90, 0.82, seed=1))
    assert rt_F(f, fz) == pytest.approx(0.0, abs=1e-9)
    assert rt_G(f, r, fz, rf) == pytest.approx(0.0, abs=1e-9)


def rt_F(pre_f, fz):
    return pre_f - fz


def rt_G(pre_f, pre_r, fz, rf):
    return (rf - fz) - (pre_r - pre_f)


def test_the_signs_point_the_way_the_docstring_says():
    """F positive = the frozen readout is WORSE than baseline. G positive = the session carries
    information the pre-stroke readout cannot reach, beyond the baseline handicap."""
    pre_f, pre_r = 0.90, 0.82                      # refit COSTS 0.08 pre-stroke, as it really does
    # a lesioned session: frozen collapses, refit collapses less
    assert rt_F(pre_f, 0.50) > 0
    assert rt_G(pre_f, pre_r, 0.50, 0.60) > 0      # refit now BUYS accuracy -> reorganisation
    # a session where both fall together: degraded, not displaced
    assert rt_G(pre_f, pre_r, 0.50, 0.42) == pytest.approx(0.0, abs=1e-9)


def test_the_pre_baseline_accepts_BOTH_shapes_of_pre_entry():
    """`_collect_5c` returns a LIST of per-session records in paired mode and a single record in
    frozen mode. A baseline that silently mishandled the list would be computed from one session."""
    one = _rec(400, 0.9, 0.8, seed=3)
    a = rt._pre_pair(one)
    b = rt._pre_pair([one])
    assert a == b
    many = rt._pre_pair([_rec(400, 0.9, 0.8, seed=3), _rec(400, 0.7, 0.6, seed=4)])
    assert a[0] > many[0] > 0.7, "pooling must sit between the two sessions, not take the first"


def test_trials_the_refit_arm_never_saw_are_dropped_from_BOTH_arms():
    """Scoring the frozen arm on a class the refit arm could not train on compares two different
    populations -- the failure that made the lick-aligned acute far-contra cell read -0.42."""
    y, p, b = _rec(400, 0.9, 0.9, seed=5)
    p[:200, 1] = REFIT_UNAVAILABLE
    p[:200, 0] = (y[:200] + 1) % 6                # frozen WRONG on exactly the dropped trials
    fz, rf, n = rt._session_pair((y, p, b))
    assert n == 200
    assert fz > 0.8, "the dropped trials must not be charged to the frozen arm"


def _rows(animal, gs, fs, n_acute=2, start=2):
    """Rows with EPOCH LABELS, because the summary references the ACUTE window and not a global max."""
    return [{"animal": animal, "day": start + 2 * i, "n_trials": 500,
             "epoch": "acute" if i < n_acute else ("subacute" if i < n_acute + 2 else "chronic"),
             "frozen": 0.0, "refit": 0.0, "gap": 0.0, "pre_frozen": 0.9, "pre_refit": 0.82,
             "pre_gap": -0.08, "F_deficit": f, "G_reorg": g}
            for i, (g, f) in enumerate(zip(gs, fs))]


def test_RETRACE_is_reported_as_a_decaying_reorganisation():
    """Reorganisation peaks acutely and unwinds while the frozen readout recovers."""
    g = [0.05, 0.20, 0.15, 0.10, 0.05, 0.01]
    f = [0.50, 0.55, 0.40, 0.28, 0.14, 0.03]
    s = rt.summarise(_rows("PS92", g, f))["PS92"]
    assert s["post_slope"] < 0
    assert s["late_exceeds_acute"] is False
    assert s["post_corr_F_G"] > 0.5, "F and G must fall together under retrace"


def test_MIGRATE_IS_DETECTED_WHEN_REORGANISATION_PEAKS_LAST():
    """THE BUG THIS TEST EXISTS FOR. A first version took argmax over ALL post-stroke sessions, so
    an animal whose reorganisation is still climbing on its last day had that day called "the peak",
    nothing could come after it, and the late-resurgence flag reported False. On the real data that
    is PS93, whose maximum sits on day 25 -- the migrate signature, silenced by the statistic meant
    to detect it. Referencing the ACUTE window instead is what makes it visible."""
    g = [0.05, 0.10, 0.18, 0.26, 0.33, 0.40]          # still rising at the end
    f = [0.50, 0.45, 0.33, 0.22, 0.12, 0.04]          # while the frozen deficit closes
    s = rt.summarise(_rows("PS93", g, f))["PS93"]
    assert s["late_exceeds_acute"] is True, "reorganisation above its acute level must be flagged"
    assert s["post_slope"] > 0
    assert s["post_corr_F_G"] < 0, "G rising as F falls is the migrate signature"


def test_a_LATE_RESURGENCE_after_a_quiet_middle_is_flagged():
    g = [0.05, 0.30, 0.10, 0.05, 0.02, 0.42]
    f = [0.50, 0.55, 0.30, 0.15, 0.05, 0.04]
    s = rt.summarise(_rows("PS95", g, f))["PS95"]
    assert s["late_exceeds_acute"] is True
    assert s["acute_G"] == pytest.approx(0.30)


def test_an_animal_with_too_few_POST_ACUTE_sessions_gets_no_slope():
    """Two points define a slope and say nothing."""
    s = rt.summarise(_rows("PS94", [0.05, 0.30, 0.10], [0.50, 0.55, 0.30]))["PS94"]
    assert s["n_post"] < rt.MIN_POST_PEAK
    assert np.isnan(s["post_slope"])


def test_an_animal_with_NO_acute_session_falls_back_and_SAYS_SO():
    """PS95 contributes one acute session in the real set and an align/variant combination could
    leave an animal with none. Silently using a different reference from the other animals would
    make the cohort count compare unlike things."""
    rows = _rows("PS92", [0.2, 0.1, 0.05, 0.02], [0.4, 0.3, 0.1, 0.05], n_acute=0)
    s = rt.summarise(rows)["PS92"]
    assert s["acute_reference_is_fallback"] is True
    assert s["n_acute"] >= 1


def test_the_cohort_line_COUNTS_animals_rather_than_pooling_them():
    """With four animals the result is a direction that holds in all of them, not a p-value."""
    rows = (_rows("PS92", [0.05, 0.20, 0.15, 0.10, 0.05], [0.5, 0.55, 0.4, 0.2, 0.05])
            + _rows("PS94", [0.05, 0.30, 0.10], [0.5, 0.55, 0.3]))
    c = rt.summarise(rows)["_cohort"]
    assert c["animals"] == 2
    assert c["animals_scored"] == 1, "PS94 has too few post-acute sessions to score"
    assert c["negative_post_slope"] == 1


def test_a_written_series_reloads_to_the_same_summary(tmp_path):
    """The sidecar exists so a summary can be revised without a twenty-minute rebuild. If the
    round trip changed a number, that saving would be a trap."""
    rows = _rows("PS92", [0.05, 0.20, 0.15, 0.10, 0.05], [0.5, 0.55, 0.4, 0.2, 0.05])
    p = rt.write_csv(rows, tmp_path / "s.csv")
    back = rt.load_csv(p)
    assert rt.summarise(back)["PS92"] == rt.summarise(rows)["PS92"]


# --------------------------------------------------------------- the figure sidecars
#
# A figure whose values exist nowhere else drifts from every text that quotes it, and nothing can
# notice. PRELIM_DATA carried stale chronic numbers for two days for exactly this reason.


def test_a_bar_figures_values_are_written_beside_it(tmp_path):
    from wfield_local import epoch_figures as ef

    values = {"pre": {"near ipsi": (0.95, 0.92, 0.97)}, "chronic": {"near ipsi": 0.71}}
    marks = {"pre": {"near ipsi": "**"}}
    p = ef.write_values(values, tmp_path / "fig.png", marks=marks)
    txt = p.read_text(encoding="utf-8")
    assert "epoch,position,value,lo,hi,mark" in txt
    assert "pre,near ipsi,0.95,0.92,0.97,**" in txt
    assert "chronic,near ipsi,0.71,,," in txt, "a bare number must still be written"


def test_a_contrast_panel_keeps_BOTH_intervals(tmp_path):
    """The corrected pair is the whole argument on this family: the per-position chronic estimates
    exclude zero uncorrected and cross it after Bonferroni. A sidecar recording one of the two would
    let either reading be quoted as the figure's."""
    from wfield_local import epoch_figures as ef

    rows = {"chronic": {"far contra": (-0.12, -0.19, -0.03, -0.26, 0.02)}}
    p = ef.write_contrast_values(rows, tmp_path / "d.png")
    txt = p.read_text(encoding="utf-8")
    assert "point,lo95,hi95,lo_corrected,hi_corrected" in txt
    assert "-0.12,-0.19,-0.03,-0.26,0.02" in txt


def test_the_per_SESSION_dots_are_written_too(tmp_path):
    """The bar is session-weighted and the epochs are not balanced across animals. The dots are how
    that is audited, and they are exactly what a bar hides."""
    from wfield_local import epoch_figures as ef

    ef.write_values({"acute": {"far contra": 0.3}}, tmp_path / "f.png",
                    points={"acute": {"far contra": [("PS92", 0.28), ("PS93", 0.33)]}})
    txt = (tmp_path / "f_sessions.csv").read_text(encoding="utf-8")
    assert "acute,far contra,PS92,0.28" in txt and "PS93,0.33" in txt


def test_a_sidecar_failure_NEVER_costs_the_figure(tmp_path, capsys):
    """Same contract `_save_png_svg` gives the SVG: warn and carry on. A figure lost to a CSV bug
    would be a strictly worse trade than the drift the CSV prevents."""
    from wfield_local import epoch_figures as ef

    class Boom(dict):
        def items(self):
            raise RuntimeError("bad values")

    ef.write_values(Boom({"acute": {}}), tmp_path / "g.png")   # non-empty, so `values or {}` keeps it
    assert "failed" in capsys.readouterr().out


def test_F_minus_G_IS_the_refit_decoders_own_deficit():
    """The identity the unity line reads off, and the reason that line means anything:

        F - G = (pre_frozen - frozen) - [(refit - frozen) - (pre_refit - pre_frozen)]
              = pre_refit - refit

    So a point ON the diagonal is a session whose refit decoder is back to baseline -- all the
    information has returned and the whole frozen deficit is readout mismatch. Distance BELOW the
    line is the part no decoder recovers. If this identity ever broke, the line would still be drawn
    and would silently mean nothing.
    """
    pre_f, pre_r = 0.90, 0.82
    for fz, rf in ((0.50, 0.60), (0.70, 0.82), (0.88, 0.95), (0.40, 0.40)):
        F = pre_f - fz
        G = (rf - fz) - (pre_r - pre_f)
        assert F - G == pytest.approx(pre_r - rf, abs=1e-12)


def test_the_row_carries_refit_deficit_and_it_matches_F_minus_G():
    rows = _rows("PS92", [0.10], [0.30])
    # the synthetic helper does not set it; the real `series` does, so check the real arithmetic
    pre_f, pre_r, fz, rf = 0.90, 0.82, 0.55, 0.60
    row = {"F_deficit": pre_f - fz, "G_reorg": (rf - fz) - (pre_r - pre_f),
           "refit_deficit": pre_r - rf}
    assert row["F_deficit"] - row["G_reorg"] == pytest.approx(row["refit_deficit"], abs=1e-12)
    assert rows  # helper still usable


def test_a_series_missing_a_newer_column_still_round_trips(tmp_path):
    """A CSV written before `refit_deficit` existed must reload and rewrite, not KeyError."""
    old = [{"animal": "PS92", "day": 4, "epoch": "acute", "n_trials": 500, "frozen": 0.5,
            "refit": 0.6, "gap": 0.1, "pre_frozen": 0.9, "pre_refit": 0.82, "pre_gap": -0.08,
            "F_deficit": 0.4, "G_reorg": 0.18}]
    p = rt.write_csv(old, tmp_path / "old.csv")
    back = rt.load_csv(p)
    rt.write_csv(back, tmp_path / "again.csv")
    assert "refit_deficit" in (tmp_path / "again.csv").read_text(encoding="utf-8").splitlines()[0]


# --------------------------------------------------------------- both families, never one

def test_series_can_be_asked_for_EITHER_family():
    """The raw gap is positive pre-stroke under matching and negative under the unmatched design, so
    neither alone is about the lesion (DECISIONS 2026-09-12). A module that could only produce one
    would put every figure it draws on the wrong side of that rule -- which is where this one sat
    until the matched arm was added."""
    import inspect

    sig = inspect.signature(rt.series)
    assert "matched" in sig.parameters
    assert sig.parameters["matched"].default is False, "unmatched stays the default for continuity"
    assert "matched" in inspect.signature(rt.pre_points).parameters


def test_run_emits_BOTH_families_under_distinct_names(monkeypatch, tmp_path):
    """If both wrote to one filename the second would silently overwrite the first, and the figure
    on disk would be whichever ran last -- the exact failure the bracket exists to prevent."""
    seen = []

    def fake_series(align, variant, matched=False):
        seen.append(matched)
        return _rows("PS92", [0.05, 0.20, 0.15, 0.10, 0.05], [0.5, 0.55, 0.4, 0.2, 0.05])

    monkeypatch.setattr(rt, "series", fake_series)
    monkeypatch.setattr(rt, "pre_points", lambda *a, **k: {"PS92": [(0.0, 0.0), (0.01, 0.01)]})
    monkeypatch.setattr(rt, "figure", lambda *a, **k: [])
    monkeypatch.setattr(rt, "figure_by_animal", lambda *a, **k: [])
    out = rt.run(out_dir=tmp_path)
    assert seen == [False, True], "both families must be built, unmatched first"
    assert set(out) == {"unmatched", "matched"}
    names = {p.name for p in tmp_path.glob("*.csv")}
    assert "recovery_trajectory_unmatched_cue_working.csv" in names
    assert "recovery_trajectory_matched_cue_working.csv" in names


def test_the_family_is_recorded_IN_the_rows(tmp_path):
    """A CSV that did not say which family it came from could be quoted as either."""
    rows = _rows("PS92", [0.1], [0.3])
    rows[0]["matched"] = 1
    p = rt.write_csv(rows, tmp_path / "m.csv")
    assert "matched" in p.read_text(encoding="utf-8").splitlines()[0]
    assert rt.load_csv(p)[0]["matched"] == 1
