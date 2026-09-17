"""Guards for the transfer-matrix arm (`epoch_15g`).

TWO OF THESE ARE REGRESSION TESTS FOR BUGS THAT ALREADY HAPPENED, both from assuming a callee's
contract instead of reading it, and both of which would have produced a plausible-looking wrong
result rather than a crash:

  * `matched_frozen` returns ``(fitted_model, n_used)``, NOT ``(X, y)``. Unpacked as features and
    labels it type-checks and runs; `np.unique` of an int then has length 1, every cell fails its
    class check, and the matrix comes out EMPTY -- which reads as a data problem, not a code one.
  * `_balanced_accuracy` returns ``(accuracy, n_classes)``, not a float. The pair reached
    `round()` and raised, but had the value been used one step earlier it would have silently
    become a tuple in the output.

The third guard is against templating. Three of the defects the 2026-09-16 audit found spread by
writing scripts from siblings; this module imports `15f`'s functions instead, and that has to stay
true or the two arms can drift apart while appearing to share a method.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

MOD = "scripts.rest_migration.substitute_code"


def _mod():
    return pytest.importorskip(MOD)


# ----------------------------------------------------------------- the callee contracts

def test_matched_frozen_returns_a_FITTED_MODEL_not_features():
    """REGRESSION. The caller must treat the first element as something it can `.predict` with."""
    rf = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 30)
    X = rng.normal(size=(90, 4)) + y[:, None]
    g = np.repeat(np.arange(9), 10)
    got = rf.matched_frozen(X, y, g, 60, rng)
    assert got is not None
    fit, n_used = got
    assert hasattr(fit, "predict"), "first element must be a fitted model"
    assert isinstance(n_used, int), "second element must be the period count"
    assert len(fit.predict(X)) == len(y)


def test_balanced_accuracy_returns_a_PAIR():
    """REGRESSION. `(acc, n_classes)` -- the class count sets the chance level and is not optional."""
    rf = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")
    y = np.array([0, 0, 1, 1, 2, 2])
    acc, ncls = rf._balanced_accuracy(y, y)
    assert acc == pytest.approx(1.0)
    assert ncls == 3


def test_score_unpacks_the_pair_and_reports_the_class_count():
    m = _mod()

    class _Fit:
        def predict(self, X):
            return np.asarray(X)

    y = np.array([0, 0, 1, 1, 2, 2])
    g = np.array([0, 0, 1, 1, 2, 2])
    bal = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")._balanced_accuracy
    acc, nul, p, ncls = m._score(_Fit(), y, y, g, np.random.default_rng(0), 20, bal,
                                 lambda yy, gg, rr, kind: yy)
    assert acc == pytest.approx(1.0)
    assert ncls == 3, "the class count must survive to the caller"
    assert 0.0 <= p <= 1.0
    assert isinstance(nul, float)


# ----------------------------------------------------------------- the normalisation

def test_retained_is_the_fraction_of_the_test_epochs_own_ceiling():
    m = _mod()
    # model reaches 0.33 above its null; the ceiling reaches 0.43 above its own -> 0.767
    assert m.retained(0.50, 0.17, 0.60, 0.17) == pytest.approx(0.33 / 0.43)


def test_retained_is_exactly_one_on_the_diagonal():
    """The diagonal is the ceiling, so it is 1.0 BY CONSTRUCTION and is not a result."""
    m = _mod()
    assert m.retained(0.44, 0.17, 0.44, 0.17) == pytest.approx(1.0)


def test_retained_refuses_a_collapsed_ceiling():
    """A test epoch whose own decoder barely beats its null has no meaningful denominator.

    Without this, dividing by ~0 manufactures ratios of 5 or 50 that read as spectacular transfer.
    `15f`'s acute cell is the live example of a small denominator dominating a ratio.
    """
    m = _mod()
    assert m.retained(0.50, 0.17, 0.18, 0.17) is None
    assert m.retained(0.50, 0.17, 0.17, 0.17) is None
    assert m.retained(0.50, 0.17, 0.10, 0.17) is None, "a ceiling BELOW its null must also refuse"


def test_retained_can_exceed_one_and_is_not_clipped():
    """A transfer that beats the test epoch's own refit is surprising, not impossible, and
    clipping it would hide exactly the case worth investigating."""
    m = _mod()
    assert m.retained(0.70, 0.17, 0.60, 0.17) > 1.0


def test_retained_goes_negative_when_transfer_is_below_null():
    m = _mod()
    assert m.retained(0.10, 0.17, 0.60, 0.17) < 0


# ----------------------------------------------------------------- anti-templating

def test_it_imports_15f_rather_than_reimplementing_it():
    """If these are ever copied in, the two arms can drift while appearing to share a method."""
    src = inspect.getsource(_mod())
    assert "from scripts.rest_migration.rest_frozen_decoder import" in src
    for name in ("_collect", "matched_frozen", "null_labels", "_balanced_accuracy", "_usable"):
        assert f"def {name}(" not in src, f"{name} must be IMPORTED from 15f, not redefined here"


def test_the_engagement_gate_is_not_optional_here():
    """`15f` has `--no-engagement-gate` for measuring the correction's size. This arm reports a
    headline result under one filename and must not be runnable ungated."""
    src = inspect.getsource(_mod())
    assert "no_engagement_gate" not in src
    assert "gate=True" in src, "collection must be explicitly gated"


def test_every_cell_matches_training_size():
    """Unmatched, pre trains on ~10-17 sessions and chronic on 6, so the matrix would measure
    training-set size as much as code transfer -- the error `5rm` exists to prevent."""
    src = inspect.getsource(_mod())
    assert "matched_frozen" in src
    assert "n_target" in src


def test_the_verdict_threshold_is_stated_in_the_source_and_is_relative():
    """The materiality bar must be visible and fixed, not chosen after seeing the result.

    It is RELATIVE (25% of that animal's own typical transfer) rather than absolute, because the
    raw above-chance transfers are small (0.055-0.108) and differ between animals, so one absolute
    cut-off would be lenient for a strong animal and impossible for a weak one.
    """
    src = inspect.getsource(_mod())
    assert "0.25" in src, "the relative asymmetry bar should be stated in the source"
    assert "ASYMMETRY" in src


# --------------------------------------------- the denominator trap, measured 2026-09-17
#
# The FIRST version of this arm took the replacement-vs-addition verdict on the difference of two
# RETAINED values. They are divided by DIFFERENT ceilings, so the contrast is dominated by how the
# ceilings differ rather than by the codes. On the real data the raw transfer is symmetric in all
# three animals (pre->chr 0.081/0.088/0.062 vs chr->pre 0.077/0.108/0.055) while the normalised
# asymmetry reads +0.404/+0.441/-0.001 -- and those are predicted almost exactly by the ceilings
# alone. The verdict now runs on raw above-chance, and these tests pin that.


def test_symmetric_raw_transfer_still_produces_a_fake_normalised_asymmetry():
    """THE TRAP, reproduced arithmetically. Identical raw transfer in both directions, different
    ceilings -> a large apparent asymmetry that is entirely the denominators."""
    m = _mod()
    r = 0.08                      # the SAME raw above-chance transfer in both directions
    ceil_pre, ceil_chr = 0.10, 0.23
    fwd = m.retained(0.17 + r, 0.17, 0.17 + ceil_chr, 0.17)   # pre -> chronic
    rev = m.retained(0.17 + r, 0.17, 0.17 + ceil_pre, 0.17)   # chronic -> pre
    assert rev - fwd > 0.4, "the artefact should be large -- that is why it fooled the first run"
    assert (rev - fwd) == pytest.approx(r * (1 / ceil_pre - 1 / ceil_chr), rel=1e-6), (
        "the fake asymmetry must equal the closed form r*(1/ceil_pre - 1/ceil_chr)")


def test_equal_ceilings_produce_no_fake_asymmetry():
    """The control: when the ceilings match, the normalised contrast is honest."""
    m = _mod()
    r = 0.08
    fwd = m.retained(0.17 + r, 0.17, 0.17 + 0.15, 0.17)
    rev = m.retained(0.17 + r, 0.17, 0.17 + 0.15, 0.17)
    assert rev - fwd == pytest.approx(0.0, abs=1e-12)


def test_the_verdict_is_taken_on_raw_above_chance_not_on_retained():
    """The fix must be WIRED. If someone re-points the verdict at the retained values, the arm
    silently returns to reporting the denominators as biology."""
    src = inspect.getsource(_mod())
    i = src.find("THE VERDICT")
    assert i > 0
    tail = src[i:]
    assert "raw_f" in tail and "raw_r" in tail, "the verdict must compute raw above-chance"
    assert "_material(raw_f, raw_r)" in tail, "materiality must be judged on the raw pair"


def test_addition_is_rejected_on_its_own_terms_in_the_source():
    """Independently of any asymmetry: addition predicts chronic->pre at ~the full pre ceiling.
    The observed 0.76/0.76/0.42 rejects it, and the script must say so rather than relying only
    on the symmetry argument."""
    src = inspect.getsource(_mod())
    assert "ADDITION is separately rejected" in src or "ADDITION ALSO FAILS" in src


def test_per_animal_rows_unpacking_is_consistent():
    """REGRESSION. `per_animal_rows` carries six fields; a stale five-field unpack raises only
    inside ONE verdict branch, which ruff cannot see and a passing run need not reach.

    THE NAME IS PART OF THE FIX. Both lists were originally called `per` in the same function --
    a list of floats in the aggregation and a list of 6-tuples in the verdict. Two shapes under
    one name in one scope is how a wrong unpack survives review, and it also made this test
    unwritable: matching `" in per"` caught `per_an.values()` and the float list too.
    """
    src = inspect.getsource(_mod())
    checked = 0
    for line in src.splitlines():
        if "per_animal_rows.append" in line or " in " not in line or "for " not in line:
            continue
        head, _, tail = line.partition(" in ")
        if not tail.split() or tail.split()[0].rstrip(":)],") != "per_animal_rows":
            continue
        names = head.split("for ", 1)[1]
        checked += 1
        assert names.count(",") == 5, f"expected a 6-field unpack, got: {line.strip()}"
    assert checked >= 2, "expected to find the per-animal unpack sites"


def test_the_two_result_lists_do_not_share_a_name():
    """The collision that made the above necessary must not come back."""
    src = inspect.getsource(_mod())
    assert "\n            per = [" not in src and "\n        per = []" not in src, (
        "use `ratios` for the aggregation floats and `per_animal_rows` for the verdict tuples")
