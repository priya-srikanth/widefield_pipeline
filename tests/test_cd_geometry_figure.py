"""The CIM geometry figure reads SAVED numbers and must not paper over a missing or stale one.

Its whole reason to exist is that the overlap was briefly read against 1.0 instead of against
chance, and that the magnitude was not read at all. So what is tested here is the arithmetic that
carries those two readings -- and the refusal to draw a dump that was made under other constants,
which is the one way a persisted result is worse than recomputing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import cd_geometry_figure as cg
from wfield_local import cd_trajectories as cdt


def _result(animal="PS95", align="precue", **over):
    t = 6
    res = {"animal": animal, "align": align, "method": "dom", "basis_id": "abc123def456",
           "ncomp": 95, "fs": 10.0, "span": (-3.0, 4.0), "pre_n": 2, "post_n": 4,
           "positions": [0, 1], "n_sessions": 3, "errors": [], "win_s": 2.0,
           "standardised": True, "smooth_s": cdt.SMOOTH_S, "orth": True, "gate": "lick",
           "reference": "contrast", "fit_on": ["success"], "cim_k": 2,
           # THE MASKED ARM IS THE DEFAULT, so the fixture has to be one -- `collect` looks up the
           # `cortexonly` tag now and an unmasked fixture would read as "no saved result".
           "mask_occluded": True, "n_dropped": 28,
           "surviving": {0: 0.61, 1: 0.88}, "anchor": 1.0,
           "cim_cos": {"pre": 1.0, "acute": 0.68, "subacute": 0.61, "chronic": 0.66},
           "cim_chance": 2.0 / 95.0,
           "cim_scale": {"pre": 1.0, "acute": 1.51, "subacute": 1.35, "chronic": 1.32},
           "traces": {("pre", 0, 0): {"mean": np.zeros(t), "n": 50, "iqr": np.ones(t),
                                      "lo": np.zeros(t), "hi": np.ones(t)}}}
    res.update(over)
    return res


def test_leak_is_the_AMPLITUDE_fraction_times_the_scale_not_the_power_fraction():
    """Overlap is a POWER fraction; the amplitude surviving the projection is its square root.
    Using the power fraction directly understates the residual, and understating it is what
    licenses trusting a post-stroke panel that should not be trusted.
    """
    assert cg.leak(0.64, None) == pytest.approx(0.6)          # sqrt(1 - 0.64)
    assert cg.leak(0.64, 1.5) == pytest.approx(0.9)           # and the magnitude multiplies


def test_a_GROWING_shared_response_makes_the_residual_WORSE_not_unchanged():
    """PS95's measured 1.51/1.35/1.32. The first residual table assumed 1.0, so every one of its
    ratios was optimistic -- the direction of that error is the point.
    """
    assert cg.leak(0.61, 1.35) > cg.leak(0.61, 1.0)
    assert cg.leak(0.61, 1.35) / cg.leak(0.61, 1.0) == pytest.approx(1.35)


def test_leak_is_defined_at_a_PERFECT_overlap_and_never_returns_a_nan_for_one():
    """A conserved mode is the common case and sqrt of a small negative is the obvious trap."""
    assert cg.leak(1.0, 1.4) == pytest.approx(0.0)
    assert cg.leak(1.0 + 1e-12, 1.0) == pytest.approx(0.0)
    assert np.isnan(cg.leak(None, 1.0))


def test_collect_reports_a_MISSING_dump_by_name_rather_than_drawing_three_animals(tmp_path):
    """The photobleach summary lost three animals silently; a figure that quietly drops an animal
    looks exactly like an animal with no effect (rule 8 -- the per-animal table comes first)."""
    cdt.save_result(_result("PS95"), tmp_path)
    got, problems = cg.collect(tmp_path, aligns=("precue",))
    assert set(got) == {("PS95", "precue")}
    assert len(problems) == 3
    assert all("no saved result" in p for p in problems)


def test_collect_refuses_a_STALE_dump_and_says_which(tmp_path):
    cdt.save_result(_result("PS95"), tmp_path)
    old = cdt.SMOOTH_S
    try:
        cdt.SMOOTH_S = old + 0.1
        got, problems = cg.collect(tmp_path, animals=("PS95",), aligns=("precue",))
    finally:
        cdt.SMOOTH_S = old
    assert got == {}
    assert len(problems) == 1 and "STALE" in problems[0]


def test_a_result_rendered_WITHOUT_orth_does_not_answer_an_orth_lookup(tmp_path):
    """`--orth off` and `--orth on` are separate tags, so a non-orth dump must not be picked up and
    drawn as though it carried a projected-out mode. It reads as absent, which is what it is."""
    cdt.save_result(_result("PS95", orth=False, cim_cos={}, cim_scale={}), tmp_path,
                    mask_occluded=True)
    got, problems = cg.collect(tmp_path, animals=("PS95",), aligns=("precue",))
    assert got == {}
    assert "no saved result" in problems[0]


def test_an_orth_result_whose_CIM_COULD_NOT_BE_FITTED_is_reported_not_drawn_flat(tmp_path):
    """The other way a row can be empty: orthogonalisation was asked for and the pre-stroke grand
    mean was unavailable, so there is no mode to compare across epochs. Silently drawing nothing
    would look like an animal whose mode did not move."""
    cdt.save_result(_result("PS95", cim_cos={}, cim_scale={}), tmp_path)
    got, problems = cg.collect(tmp_path, animals=("PS95",), aligns=("precue",))
    assert got == {}
    assert "no cim_cos" in problems[0]


def test_the_table_carries_the_CHANCE_LEVEL_on_every_row(tmp_path):
    """0.61 against 1.0 reads as a deficit; 0.61 against 0.021 reads as ~30x chance. The chance
    level is not optional context, so it is a column rather than a footnote."""
    cdt.save_result(_result("PS95"), tmp_path)
    got, _ = cg.collect(tmp_path, animals=("PS95",), aligns=("precue",))
    txt = cg.table(got)
    assert "chance" in txt.splitlines()[0]
    assert txt.count("0.021") == len(cg.POST)


def test_the_figure_draws_from_the_dumps_alone(tmp_path):
    for a in ("PS94", "PS95"):
        cdt.save_result(_result(a), tmp_path)
    got, _ = cg.collect(tmp_path, aligns=("precue",))
    png = cg.figure(got, tmp_path / "geom.png")
    assert Path(png).exists() and Path(png).stat().st_size > 10000


# ------------------------------------------------------------------ the cohort guard


def test_a_dump_from_a_DIFFERENT_COHORT_is_refused(tmp_path, monkeypatch):
    """THE FAILURE THIS EXISTS FOR, 2026-09-25. Three sessions were registered by the other machine
    while the CD arm was being computed -- three of four animals went from 26 curated sessions to 27,
    each gaining a CHRONIC session, and chronic is where the permutation found five of its six
    significant cells. Every figure rendered, every number was quoted, and the only symptom was a
    rejected `git push` an hour later.

    The dumps recorded the basis, the window, the smoothing and the gate, so a PARAMETER change could
    not be redrawn silently. They did not record WHICH SESSIONS went in.
    """
    monkeypatch.setattr(cdt, "cohort_digest", lambda _a: "26:aaaaaaaaaa")
    cdt.save_result(_result(), tmp_path, mask_occluded=True)
    monkeypatch.setattr(cdt, "cohort_digest", lambda _a: "27:bbbbbbbbbb")
    with pytest.raises(ValueError, match="STALE"):
        cdt.load_result(tmp_path, "PS95", "precue", orth=True, mask_occluded=True)


def test_a_dump_from_a_DIFFERENT_EPOCH_SPEC_is_refused(tmp_path, monkeypatch):
    """Separate from the cohort because they move independently. Registering a session re-derived
    PS94's `chronic_from` from 25 to None on 2026-09-25 before it was pinned back -- an hour either
    way and one animal would have had no chronic epoch, with four animals' worth of chronic results
    quietly built from three."""
    monkeypatch.setattr(cdt, "epoch_spec_digest", lambda _a: "aaaa111111")
    cdt.save_result(_result(), tmp_path, mask_occluded=True)
    monkeypatch.setattr(cdt, "epoch_spec_digest", lambda _a: "bbbb222222")
    with pytest.raises(ValueError, match="STALE"):
        cdt.load_result(tmp_path, "PS95", "precue", orth=True, mask_occluded=True)


def test_a_dump_predating_the_guard_reads_as_STALE_not_as_fine(tmp_path, monkeypatch):
    """Absent is not equal. A pre-2026-09-25 dump has no cohort field at all, and it genuinely
    describes an unknown cohort -- so "we cannot tell" must read as stale rather than as a match."""
    from wfield_local import results_store as rs

    cdt.save_result(_result(), tmp_path, mask_occluded=True)
    tag = cdt.result_tag("PS95", "precue", orth=True, mask_occluded=True)
    body = rs.load(tmp_path, cdt.RESULT_NAME, tag)
    body["_meta"].pop("cohort")
    payload = {k: v for k, v in body.items() if k != "_meta"}
    rs.save(tmp_path, cdt.RESULT_NAME, tag, payload,
            meta={k: v for k, v in body["_meta"].items() if k not in ("schema", "name", "tag")})
    with pytest.raises(ValueError, match="STALE"):
        cdt.load_result(tmp_path, "PS95", "precue", orth=True, mask_occluded=True)


def test_the_cohort_digest_carries_the_COUNT_in_the_clear():
    """The commonest change is "a session was registered", and a reader should see that without
    decoding a hash -- 26 vs 27 is the whole story most of the time."""
    d = cdt.cohort_digest("PS95")
    assert ":" in d and d.split(":")[0].isdigit()
