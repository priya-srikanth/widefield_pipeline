"""Guards for the 15h cohort statistic and the family-wise max-statistic.

The properties pinned here are the ones the arm's conclusions rest on, not the plumbing:
the sign convention (a rotated cell must score POSITIVE), that the max-statistic actually
exploits the coupling it was adopted for, and that a ragged family cannot silently shrink.
"""
import numpy as np
import pytest

from scripts.rest_migration.rotation_maps import load_draws, save_draws
from scripts.rest_migration.rotation_stats import (
    MIN_DRAWS,
    cell_z,
    cells_from_regions_csv,
    cohort_cosine,
    maxstat_threshold,
)

RNG = np.random.default_rng(0)


def _null(n=200, mu=0.8, sd=0.05):
    return RNG.normal(mu, sd, n)


# ---------------------------------------------------------------- cell_z

def test_rotated_cell_scores_positive():
    """THE SIGN CONVENTION. The null is two models of the SAME code, so a turned readout sits
    BELOW it and must come out positive -- if this flips, every conclusion in the arm inverts."""
    z, _, _ = cell_z(0.50, _null(mu=0.80, sd=0.05))
    assert z > 0


def test_unrotated_cell_scores_about_zero():
    null = _null(mu=0.80, sd=0.05)
    z, _, _ = cell_z(float(np.mean(null)), null)
    assert abs(z) < 0.2


def test_cell_with_too_few_draws_is_refused():
    """A cell estimated from a handful of draws has an sd that is mostly noise, and because the
    family threshold is a MAXIMUM one such cell raises the bar for every other cell in it."""
    assert cell_z(0.5, _null(n=MIN_DRAWS - 1))[0] is None


def test_zero_variance_null_is_refused_not_infinite():
    """`np.std` of a constant array is ~1e-16, NOT 0, so a `sd > 0` guard lets the cell through
    and it scores z ~ 1e15. Because the family threshold is a MAXIMUM, one such cell would set
    the threshold single-handedly and silence every real cell in the family."""
    z, _, sd = cell_z(0.5, np.full(100, 0.8))
    assert z is None
    assert sd == pytest.approx(0.0, abs=1e-12)


def test_near_degenerate_null_cannot_set_the_family_threshold():
    """The end-to-end version of the guard above: a degenerate cell must LEAVE the family rather
    than dominate its maximum."""
    fam = _family(4)
    k = next(iter(fam))
    fam[k][0]["cos_null"] = np.full(200, 0.8)
    thresh, _, n_cells = maxstat_threshold(fam)
    assert n_cells == 3
    assert thresh < 10


# ---------------------------------------------------------------- cohort

def _cells(obs, n_animals=4, boot_sd=0.04):
    return [{"cos_obs": obs, "cos_boot": RNG.normal(obs, boot_sd, 200), "_z": 1.0}
            for _ in range(n_animals)]


def test_cohort_ci_brackets_the_observed_mean():
    mean, lo, hi, n_an, _, inner = cohort_cosine(_cells(0.6), n_boot=500)
    assert lo < mean < hi and n_an == 4 and inner == "draws"


def test_animal_without_bootstrap_contributes_its_point_estimate():
    """DROPPING IT WOULD CHANGE THE COHORT COMPOSITION between cells, and a cell whose animals
    differ from its neighbour's is not comparable to it."""
    cells = _cells(0.6, n_animals=3)
    cells.append({"cos_obs": 0.2, "cos_boot": np.zeros(0), "_z": 1.0})
    mean, _, _, n_an, _, _ = cohort_cosine(cells, n_boot=300)
    assert n_an == 4
    assert mean == pytest.approx(np.mean([0.6, 0.6, 0.6, 0.2]), abs=1e-9)


def test_cohort_ci_narrows_with_more_animals():
    _, lo4, hi4, *_ = cohort_cosine(_cells(0.6, n_animals=4), n_boot=1000)
    _, lo12, hi12, *_ = cohort_cosine(_cells(0.6, n_animals=12), n_boot=1000)
    assert (hi12 - lo12) < (hi4 - lo4)


def test_empty_cell_returns_none():
    assert cohort_cosine([]) is None


# ---------------------------------------------------------------- max-statistic

def _family(n_cells, n_draws=200, correlated=False, n_animals=1):
    """One entry per (cell, animal). `correlated` makes every cell share one latent draw."""
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 1, n_draws)
    out = {}
    for c in range(n_cells):
        indep = rng.normal(0, 1, n_draws)
        base = shared if correlated else indep
        out[("arm", "acute - pre", c)] = [
            {"cos_obs": 0.5, "cos_null": 0.8 + 0.05 * base, "_z": 1.0}
            for _ in range(n_animals)]
    return out


def test_maxstat_exceeds_a_single_cell_threshold():
    """A maximum over many cells must sit above the same quantile of one of them -- that IS the
    multiplicity correction, and if it does not hold the threshold is not correcting anything."""
    one, _, _ = maxstat_threshold(_family(1))
    many, _, n_cells = maxstat_threshold(_family(20))
    assert n_cells == 20
    assert many > one


def test_correlated_cells_give_a_LOWER_threshold_than_independent_ones():
    """THE WHOLE ARGUMENT AGAINST BONFERRONI. The six positions are coupled -- multinomial class
    coefficients are identified only up to a constant shift across classes -- so the family's
    true maximum is smaller than an independence assumption implies. If this test fails the
    max-statistic is buying nothing and Bonferroni would have been honest."""
    indep, _, _ = maxstat_threshold(_family(20, correlated=False))
    corr, _, _ = maxstat_threshold(_family(20, correlated=True))
    assert corr < indep


def test_ragged_family_truncates_to_the_shortest_cell():
    """A ragged maximum would be taken over a changing number of cells and would drift downward
    as the longer cells ran on alone."""
    fam = _family(3, n_draws=200)
    k = next(iter(fam))
    fam[k][0]["cos_null"] = fam[k][0]["cos_null"][:60]
    thresh, n_draws, n_cells = maxstat_threshold(fam)
    assert n_draws == 60 and n_cells == 3 and thresh is not None


def test_cells_below_the_draw_floor_leave_the_family():
    fam = _family(3)
    k = next(iter(fam))
    fam[k][0]["cos_null"] = fam[k][0]["cos_null"][:MIN_DRAWS - 1]
    _, _, n_cells = maxstat_threshold(fam)
    assert n_cells == 2


def test_no_usable_cells_returns_none_not_a_crash():
    assert maxstat_threshold({})[0] is None


# ---------------------------------------------------------------- persistence

def test_draws_round_trip_preserves_order_and_raggedness(tmp_path):
    """Order is load-bearing: positions are comparable WITHIN a draw only because the draw index
    means the same thing for each of them."""
    draws = [
        {"arm": "cue", "animal": "PS92", "contrast": "acute - pre", "position": 0,
         "cos_obs": 0.5, "cos_null": np.arange(50, dtype=np.float32),
         "cos_boot": np.arange(200, dtype=np.float32)},
        {"arm": "cue", "animal": "PS93", "contrast": "acute - pre", "position": 5,
         "cos_obs": -0.1, "cos_null": np.arange(12, dtype=np.float32),
         "cos_boot": np.zeros(0, dtype=np.float32)},
    ]
    save_draws(draws, tmp_path)
    back = load_draws(tmp_path)
    assert [d["animal"] for d in back] == ["PS92", "PS93"]
    assert [d["position"] for d in back] == [0, 5]
    assert back[0]["cos_null"].size == 50 and back[1]["cos_null"].size == 12
    assert back[1]["cos_boot"].size == 0
    assert back[0]["cos_obs"] == pytest.approx(0.5)
    np.testing.assert_allclose(back[0]["cos_null"], np.arange(50))


def test_load_draws_returns_empty_when_absent(tmp_path):
    assert load_draws(tmp_path, "_nope") == []


# ---------------------------------------------------------------- reduced-table fallback

def _regions_csv(tmp_path, rows):
    import csv as _csv
    p = tmp_path / "epoch_15h_rotation_regions.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def _row(animal, pos, cos, lo, hi, region="MOp_left"):
    return {"arm": "cue", "animal": animal, "contrast": "acute - pre", "position": pos,
            "region": region, "cosine_pre_vs_epoch": cos, "cos_ci_lo": lo, "cos_ci_hi": hi}


def test_regions_csv_gives_one_cell_per_animal_not_per_region(tmp_path):
    """The table has one row per REGION; the cosine is per CELL and identical across them. Taking
    every row would count a single animal ~90 times and collapse the cohort interval to nothing."""
    rows = [_row("PS92", 0, 0.5, 0.4, 0.6, r) for r in ("MOp_left", "SSp-m_right", "VISp_left")]
    rows += [_row("PS93", 0, 0.3, 0.2, 0.4, r) for r in ("MOp_left", "SSp-m_right")]
    cells = cells_from_regions_csv(_regions_csv(tmp_path, rows))
    assert len(cells[("cue", "acute - pre", 0)]) == 2


def test_fallback_ci_is_marked_gaussian_never_exact(tmp_path):
    """An approximate interval must never be readable as an exact one."""
    rows = [_row(a, 0, 0.5, 0.4, 0.6) for a in ("PS92", "PS93", "PS94", "PS95")]
    cells = cells_from_regions_csv(_regions_csv(tmp_path, rows))
    *_, inner = cohort_cosine(cells[("cue", "acute - pre", 0)], n_boot=500)
    assert inner == "gaussian"


def test_fallback_recovers_a_sensible_sd_from_the_saved_ci(tmp_path):
    """sd = (hi - lo) / (2 * 1.96). A cell with a wide CI must end up with a wider cohort
    interval than one with a narrow CI, or the saved spread is being ignored."""
    narrow = [_row(a, 0, 0.5, 0.49, 0.51) for a in ("PS92", "PS93", "PS94", "PS95")]
    wide = [_row(a, 0, 0.5, 0.1, 0.9) for a in ("PS92", "PS93", "PS94", "PS95")]
    (tmp_path / "n").mkdir()
    (tmp_path / "w").mkdir()
    cn = cells_from_regions_csv(_regions_csv(tmp_path / "n", narrow))
    cw = cells_from_regions_csv(_regions_csv(tmp_path / "w", wide))
    _, lo_n, hi_n, *_ = cohort_cosine(cn[("cue", "acute - pre", 0)], n_boot=2000)
    _, lo_w, hi_w, *_ = cohort_cosine(cw[("cue", "acute - pre", 0)], n_boot=2000)
    assert (hi_w - lo_w) > (hi_n - lo_n)


def test_a_row_with_no_ci_falls_back_to_its_point_estimate(tmp_path):
    rows = [_row("PS92", 0, 0.5, "", "")]
    cells = cells_from_regions_csv(_regions_csv(tmp_path, rows))
    assert cells[("cue", "acute - pre", 0)][0]["cos_sd"] is None
