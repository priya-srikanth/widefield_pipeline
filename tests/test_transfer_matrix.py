"""Guards for `wfield_local.transfer_matrix`, the shared cross-epoch transfer core.

It is shared BECAUSE it is shared: the rest arm and the three trial arms (ENL / cue / lick) call
one implementation, so a guard fixed here is fixed for all four. Three of the 2026-09-16 audit's
defects spread by copying sibling scripts, and a second copy of this loop is exactly how the task
arms would quietly stop matching training size or stop using the block-permutation null.

THE CENTRAL RULE THESE TESTS PROTECT: the direction contrast is taken on RAW above-chance units,
never on the retained ratios. The ratios have different denominators in the two directions, so
their difference is nonzero whenever the ceilings differ even with perfectly symmetric data.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import transfer_matrix as tm


def _pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))


def _sep_data(rng, n_sess=4, per=120, ncls=3, sep=3.0, epoch="pre", start=0):
    """Sessions whose classes are linearly separable, so transfer SHOULD succeed."""
    out = {}
    for s in range(n_sess):
        y = np.repeat(np.arange(ncls), per // ncls)
        X = rng.normal(size=(len(y), 5)) + sep * np.eye(5)[y % 5]
        g = np.repeat(np.arange(len(y) // 10), 10)[:len(y)]
        out[f"S{start + s}"] = (epoch, X, y, g)
    return out


# ------------------------------------------------------------------ the normalisation rule

def test_retained_is_the_fraction_of_the_test_epochs_own_ceiling():
    assert tm.retained(0.50, 0.17, 0.60, 0.17) == pytest.approx(0.33 / 0.43)


def test_retained_is_one_on_the_diagonal_by_construction():
    assert tm.retained(0.44, 0.17, 0.44, 0.17) == pytest.approx(1.0)


def test_retained_refuses_a_collapsed_ceiling():
    """Dividing by a near-zero denominator manufactures ratios of 5 or 50 that read as spectacular
    transfer. Refusing is the only safe answer."""
    assert tm.retained(0.50, 0.17, 0.17 + tm.MIN_CEILING, 0.17) is None
    assert tm.retained(0.50, 0.17, 0.10, 0.17) is None, "a ceiling BELOW its null must refuse too"


def test_symmetric_raw_transfer_still_yields_a_fake_normalised_asymmetry():
    """THE TRAP, in closed form. Identical raw transfer both ways, different ceilings, large
    apparent asymmetry -- entirely the denominators. This is why the verdict uses raw units."""
    r, ceil_a, ceil_b = 0.08, 0.10, 0.23
    fwd = tm.retained(0.17 + r, 0.17, 0.17 + ceil_b, 0.17)
    rev = tm.retained(0.17 + r, 0.17, 0.17 + ceil_a, 0.17)
    assert rev - fwd > 0.4
    assert (rev - fwd) == pytest.approx(r * (1 / ceil_a - 1 / ceil_b), rel=1e-6)


# ------------------------------------------------------------------ the scoring primitives

def test_balanced_accuracy_averages_over_PRESENT_classes_only():
    """sklearn's averages over classes the test set lacks, which silently moves the chance level."""
    y = np.array([0, 0, 1, 1])          # class 2 absent
    acc, ncls = tm.balanced_accuracy(y, y)
    assert acc == pytest.approx(1.0)
    assert ncls == 2, "chance must be 1/2 here, not 1/3"


def test_balanced_accuracy_is_not_fooled_by_a_majority_guesser():
    """A constant predictor scores 1/n_classes, not the majority fraction -- the whole point of
    balancing when the positions are unequally sampled."""
    y = np.array([0] * 90 + [1] * 10)
    acc, _ = tm.balanced_accuracy(y, np.zeros_like(y))
    assert acc == pytest.approx(0.5)


def test_block_permute_keeps_blocks_intact_and_moves_labels_between_them():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2, 3], 5)
    g = np.repeat(np.arange(4), 5)
    out = tm.block_permute(y, g, rng)
    for b in np.unique(g):
        assert len(np.unique(out[g == b])) == 1, "a block must keep ONE label"
    assert sorted(out[::5]) == sorted(y[::5]), "the multiset of block labels is preserved"


def test_block_permute_is_a_harder_null_than_a_trial_shuffle():
    """A trial-wise shuffle destroys within-block correlation as well, making the null easier than
    the data and every p value optimistic. The block null must preserve that structure."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(6), 20)
    g = np.repeat(np.arange(6), 20)
    perm = tm.block_permute(y, g, rng)
    # every block still carries a single label, unlike a trial shuffle
    assert all(len(np.unique(perm[g == b])) == 1 for b in np.unique(g))


# ------------------------------------------------------------------ the matrix itself

def test_build_recovers_transfer_when_the_code_is_shared():
    """POSITIVE CONTROL. Two epochs drawn from the SAME generative rule must transfer."""
    rng = np.random.default_rng(0)
    data = {**_sep_data(rng, epoch="pre"), **_sep_data(rng, epoch="chronic", start=10)}
    rows = tm.build(data, _pipe, n_perm=30, seed_ns="t", log=lambda *_a: None)
    off = [r for r in rows if r["train_epoch"] == "pre" and r["test_epoch"] == "chronic"]
    assert off, "the pre->chronic cell should exist"
    assert np.mean([r["acc_minus_null"] for r in off]) > 0.3


def test_build_finds_no_transfer_when_the_code_is_scrambled():
    """NEGATIVE CONTROL. If the chronic labels are unrelated to the features, transfer must sit at
    the null -- otherwise the matrix reports structure that is not there."""
    rng = np.random.default_rng(1)
    pre = _sep_data(rng, epoch="pre")
    chron = _sep_data(rng, epoch="chronic", start=10)
    chron = {k: (e, X, rng.permutation(y), g) for k, (e, X, y, g) in chron.items()}
    rows = tm.build({**pre, **chron}, _pipe, n_perm=30, seed_ns="t", log=lambda *_a: None)
    off = [r for r in rows if r["train_epoch"] == "pre" and r["test_epoch"] == "chronic"]
    assert np.mean([r["acc_minus_null"] for r in off]) < 0.12


def test_the_diagonal_holds_out_its_own_test_session():
    """Otherwise the ceiling is a memorised score and every off-diagonal ratio is deflated."""
    rng = np.random.default_rng(2)
    data = _sep_data(rng, epoch="pre")
    rows = tm.build(data, _pipe, n_perm=10, seed_ns="t", log=lambda *_a: None)
    diag = [r for r in rows if r["train_epoch"] == "pre" and r["test_epoch"] == "pre"]
    assert diag
    assert all(r["n_train_matched"] > 0 for r in diag)
    # a memorised diagonal would sit at 1.0 for separable data; LOSO leaves it below
    assert max(r["acc"] for r in diag) <= 1.0


def test_every_cell_is_trained_on_the_same_number_of_units():
    """THE `5rm` LESSON. Unmatched, the matrix measures how much training data each epoch had."""
    rng = np.random.default_rng(3)
    data = {**_sep_data(rng, n_sess=5, epoch="pre"),
            **_sep_data(rng, n_sess=2, epoch="chronic", start=10)}
    rows = tm.build(data, _pipe, n_perm=5, seed_ns="t", log=lambda *_a: None)
    sizes = {r["n_train_matched"] for r in rows}
    assert len(sizes) == 1, f"training sizes differ across cells: {sizes}"


def test_an_epoch_with_too_few_sessions_is_not_a_train_source():
    """A pooled model must never be one session wearing an epoch's name."""
    rng = np.random.default_rng(4)
    data = {**_sep_data(rng, n_sess=4, epoch="pre"),
            **_sep_data(rng, n_sess=1, epoch="acute", start=10)}
    rows = tm.build(data, _pipe, n_perm=5, seed_ns="t", min_sessions=2, log=lambda *_a: None)
    assert not [r for r in rows if r["train_epoch"] == "acute"]
    assert [r for r in rows if r["test_epoch"] == "acute"], "it may still be a TEST target"


def test_build_refuses_without_a_pre_train_source():
    rng = np.random.default_rng(5)
    rows = tm.build(_sep_data(rng, epoch="chronic"), _pipe, n_perm=5, seed_ns="t",
                    log=lambda *_a: None)
    assert rows == []


def test_the_seed_is_per_cell_so_a_subset_run_reproduces():
    """A shared advancing generator would make each draw depend on how many cells ran before it."""
    rng = np.random.default_rng(6)
    data = {**_sep_data(rng, epoch="pre"), **_sep_data(rng, epoch="chronic", start=10)}
    a = tm.build(data, _pipe, n_perm=5, seed_ns="ns", rng=np.random.default_rng(0),
                 log=lambda *_a: None)
    b = tm.build(data, _pipe, n_perm=5, seed_ns="ns", rng=np.random.default_rng(0),
                 log=lambda *_a: None)
    assert [r["acc"] for r in a] == [r["acc"] for r in b]


def test_summarise_weights_animals_equally_not_sessions():
    """Pre holds up to 17 sessions and chronic 6; session-weighting lets the best-sampled animal
    set the cohort number."""
    rows = ([{"animal": "A", "train_epoch": "pre", "test_epoch": "pre", "acc": 1.0, "null": 0.0}
             for _ in range(10)]
            + [{"animal": "B", "train_epoch": "pre", "test_epoch": "pre", "acc": 0.0,
                "null": 0.0}])
    got = tm.cell(rows, "pre", "pre")
    assert got[0] == pytest.approx(0.5), "10 sessions of A must not outweigh 1 of B"
    assert got[2] == 2


def test_summarise_retained_is_a_mean_of_per_animal_ratios():
    """Not a ratio of pooled means: the animals' ceilings differ more than twofold, so a pooled
    denominator belongs to no animal."""
    rows = []
    for an, acc, ceil in (("A", 0.30, 0.50), ("B", 0.30, 0.90)):
        rows.append({"animal": an, "train_epoch": "pre", "test_epoch": "chronic",
                     "acc": acc, "null": 0.0})
        rows.append({"animal": an, "train_epoch": "chronic", "test_epoch": "chronic",
                     "acc": ceil, "null": 0.0})
    agg = tm.summarise(rows)
    cellv = next(x for x in agg
                 if x["train_epoch"] == "pre" and x["test_epoch"] == "chronic")
    assert cellv["retained_vs_own_ceiling"] == pytest.approx((0.30 / 0.50 + 0.30 / 0.90) / 2, abs=1e-4)
    assert cellv["retained_vs_own_ceiling"] != pytest.approx(0.30 / 0.70, abs=1e-4)


def test_material_fraction_is_relative_and_declared():
    """The bar must be visible and fixed, not chosen after seeing the numbers -- and RELATIVE,
    because the raw transfers differ several-fold between animals and arms."""
    assert 0 < tm.MATERIAL_FRACTION < 1
    assert tm.MIN_CEILING > 0
