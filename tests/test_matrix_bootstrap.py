"""Per-cell bootstrap for the matrix families.

Priya, 2026-09-12: "did we give up on having any bootstrapping of fig 10 best-match matrices?"

The scalar arms of family 10 were always bootstrapped, and so was the DIAGONAL of the destination
matrix (`epoch_10cdiag`, drawn as bars through `_scalar_figure`). What never carried uncertainty was
the OFF-DIAGONAL -- which position a lost one moved toward -- and that is the substitution claim.

These tests pin the three things that make a cell mark mean what the caption says: the resampling
unit matches the rest of the deck, the delta stays paired on the animal draw, and the correction is
across the cells actually tested rather than across one.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import epoch_figures as ef
from wfield_local import matrix_bootstrap as mb

K = 6


@pytest.fixture()
def epochs_stub(monkeypatch):
    """Every post-stroke day is 'acute'; keeps these tests off the real epoch assignment."""
    monkeypatch.setattr(ef, "epoch_of_day", lambda an, d: "acute")


def _mats(cell=None, value=0.8, animals=("PS92", "PS93", "PS94", "PS95"), days=(3, 4, 5), noise=0.02):
    rng = np.random.default_rng(0)
    base = np.full((K, K), 1 / 6)
    out = {}
    for an in animals:
        by = {"PRE": np.clip(base + rng.normal(0, noise, (K, K)), 0, 1)}
        for d in days:
            M = base.copy()
            if cell is not None:
                M[cell] = value
            by[d] = np.clip(M + rng.normal(0, noise, (K, K)), 0, 1)
        out[an] = by
    return out


def _summary(mats, **kw):
    return mb.summarise(mb.by_epoch(mats), reference=1 / 6, seed=1, n_boot=400, **kw)


def test_grouping_matches_the_scalar_families(epochs_stub):
    """Same PRE key, same epoch_of_day -- or a cell interval and a bar interval describe
    different sessions while claiming to describe the same figure."""
    g = mb.by_epoch(_mats())
    assert set(g) == {"pre", "acute"}
    assert {a: len(v) for a, v in g["acute"].items()} == {a: 3 for a in
                                                          ("PS92", "PS93", "PS94", "PS95")}


def test_an_elevated_cell_marks_and_a_chance_cell_does_not(epochs_stub):
    s = _summary(_mats(cell=(5, 4)))
    assert s["acute"]["marks"][5, 4], "a cell at 0.8 against a 1/6 background must mark"
    assert not s["acute"]["marks"][0, 0], "a cell at chance must not"


def test_the_reference_is_chance_not_zero_on_the_absolute_row(epochs_stub):
    """Every cell of a fraction-of-sessions panel is above zero; testing against zero marks all 36."""
    g = mb.by_epoch(_mats())
    rng = np.random.default_rng(0)
    _pt, draws = mb.cell_draws(g, "acute", rng=rng, n_boot=400)
    assert (mb.cell_marks(draws, reference=0.0) != "").sum() == K * K
    assert (mb.cell_marks(draws, reference=1 / 6) != "").sum() < K * K


def test_pre_has_no_delta_against_itself(epochs_stub):
    assert "delta" not in _summary(_mats())["pre"]


def test_the_delta_is_paired_on_the_animal_draw(epochs_stub):
    """An animal present in only one epoch cannot supply a change and must not enter either side."""
    mats = _mats(cell=(5, 4))
    mats["PS99"] = {3: np.full((K, K), 0.9)}                      # post only, no PRE
    g = mb.by_epoch(mats)
    rng = np.random.default_rng(0)
    got = mb.delta_draws(g, "acute", "pre", rng=rng, n_boot=200)
    assert got is not None
    # PS99's 0.9 would drag the acute mean up if it leaked into the difference
    assert got[0][0, 0] == pytest.approx(0.0, abs=0.05)


def test_the_correction_is_across_the_cells_actually_tested(epochs_stub):
    """36 comparisons, not one. Correcting as though it were one is the error this exists to avoid."""
    g = mb.by_epoch(_mats(cell=(5, 4)))
    rng = np.random.default_rng(0)
    _pt, draws = mb.cell_draws(g, "acute", rng=rng, n_boot=600)
    wide = mb.cell_marks(draws, reference=1 / 6, n_comparisons=1)
    narrow = mb.cell_marks(draws, reference=1 / 6)                 # defaults to n tested
    n_corrected = lambda m: int((m == mb.MARK_CORRECTED).sum())    # noqa: E731
    assert n_corrected(narrow) <= n_corrected(wide)


def test_a_cell_that_is_nan_everywhere_stays_nan(epochs_stub):
    """A position gated out of an epoch has no value, which is not a value of nothing."""
    mats = _mats()
    for an in mats:
        for k in mats[an]:
            mats[an][k][2, 3] = np.nan
    s = _summary(mats)
    assert np.isnan(s["acute"]["point"][2, 3])
    assert s["acute"]["marks"][2, 3] == ""


def test_with_one_animal_the_animal_level_is_a_no_op(epochs_stub):
    """Resampling one animal always picks that animal, so only SESSIONS drive the interval.

    Worth pinning because it is the honest limit of a four-animal study: an epoch carried by a
    single animal gets an interval that describes its sessions and says nothing about the cohort.
    The point estimate must therefore be exactly that animal's session mean, with no shrinkage.
    """
    mats = _mats(animals=("PS94",))
    g = mb.by_epoch(mats)
    point, draws = mb.cell_draws(g, "acute", rng=np.random.default_rng(0), n_boot=200)
    expect = np.mean(np.stack([mats["PS94"][d] for d in (3, 4, 5)]), axis=0)
    assert np.allclose(point, expect)
    assert np.isfinite(mb.cell_ci(draws)[0]).all()


def test_more_sessions_narrow_the_interval(epochs_stub):
    """Sessions are the active level here; if they were not, adding them would change nothing."""
    def width(days):
        g = mb.by_epoch(_mats(days=days))
        _p, d = mb.cell_draws(g, "acute", rng=np.random.default_rng(0), n_boot=400)
        lo, hi = mb.cell_ci(d)
        return float(np.nanmean(hi - lo))

    assert width(tuple(range(3, 15))) < width((3, 4, 5))


def test_the_sidecar_carries_both_intervals_and_both_marks(epochs_stub, tmp_path):
    import csv

    s = _summary(_mats(cell=(5, 4)))
    q = mb.write_cell_values(s, tmp_path / "x.png", labels=list("abcdef"), reference=1 / 6)
    rows = list(csv.DictReader(q.open()))
    assert len(rows) == 2 * K * K
    acute = [r for r in rows if r["panel"] == "acute" and r["row"] == "f" and r["col"] == "e"][0]
    assert float(acute["lo95"]) < float(acute["value"]) < float(acute["hi95"])
    assert acute["delta_mark"] and acute["mark"]
    pre = [r for r in rows if r["panel"] == "pre"][0]
    assert pre["delta_vs_pre"] == "", "pre has no delta column content"


def test_the_mark_note_distinguishes_the_two_rows():
    """A bottom-row mark is about CHANGE, not about chance; the caption must not blur them."""
    note = mb.mark_note(1 / 6)
    assert "TOP ROW" in note and "BOTTOM ROW" in note
    assert "not a claim about chance" in note
