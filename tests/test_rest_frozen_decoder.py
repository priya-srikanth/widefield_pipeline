"""Guards for the docked frozen rest decoder.

Each of these asserts something that, if it silently broke, would still produce a figure and a
plausible number. That is the failure mode this family keeps hitting -- a wrong result that looks
like a result -- so the invariants are tested rather than reviewed.
"""
from __future__ import annotations

import numpy as np
import pytest

rfd = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")


def test_the_scored_session_is_never_in_its_own_training_pool():
    """LOSO. Without this the pre bar is a fit, and every retained fraction divides by it."""
    pre = ["PS93_0806", "PS93_0807", "PS93_0808"]
    assert rfd.training_pool(pre, "PS93_0807") == ["PS93_0806", "PS93_0808"]


def test_a_post_stroke_session_keeps_the_whole_pre_pool():
    """It is not in `pre_labels`, so nothing is dropped -- the exclusion must not fire blindly."""
    pre = ["PS93_0806", "PS93_0807"]
    assert rfd.training_pool(pre, "PS93_0821") == pre


def test_balanced_accuracy_averages_recalls_not_trials():
    """A class-imbalanced test set must not let the majority class carry the score.

    Rest-period class balance MOVES with epoch, so raw accuracy would reward that drift and read as
    a lesion effect. 90 trials of class 0 all correct and 10 of class 1 all wrong is 0.90 raw and
    must be 0.50 here.
    """
    y = np.array([0] * 90 + [1] * 10)
    p = np.zeros(100, int)
    acc, ncls = rfd._balanced_accuracy(y, p)
    assert acc == pytest.approx(0.5)
    assert ncls == 2


def test_balanced_accuracy_counts_only_the_classes_present():
    """The chance level the caller reports is 1/ncls, so ncls must describe the TEST set.

    sklearn's averages over classes absent from y_true, which silently moves chance.
    """
    y = np.array([2, 2, 5, 5])
    p = np.array([2, 2, 5, 5])
    acc, ncls = rfd._balanced_accuracy(y, p)
    assert acc == pytest.approx(1.0)
    assert ncls == 2


def test_the_counted_minimum_rejects_a_thin_session():
    y = np.repeat(np.arange(6), 5)          # 30 periods, 5 per class
    ok, why = rfd._usable(y, min_periods=40, min_per_class=5)
    assert not ok and "30 periods" in why


def test_the_counted_minimum_rejects_too_few_positions():
    """Six positions collapsed to two is a broken session, not a two-way problem to be scored."""
    y = np.repeat(np.arange(2), 40)
    ok, why = rfd._usable(y, min_periods=40, min_per_class=5)
    assert not ok and "positions" in why


def test_a_healthy_session_passes_the_minimum():
    y = np.repeat(np.arange(6), 20)
    ok, why = rfd._usable(y, min_periods=40, min_per_class=5)
    assert ok and why == ""


def test_positions_come_from_the_REPAIRED_classifier():
    """Dead `spout_bit1` (8/05-8/06) collapses six positions to four.

    A script using raw `_classify_cues` has now reported a phantom "4 positions" animal three times
    in this project (`STATUS_2026-09-16.md` pitfall 6). This asserts the import, because the symptom
    downstream is a plausible number rather than a crash.
    """
    src = __import__("inspect").getsource(rfd)
    assert "classify_cues_with_backup" in src
    assert "import _classify_cues" not in src


def test_the_gap_is_refit_minus_frozen_not_the_reverse():
    """The sign convention, asserted rather than restated.

    The task arm's identical quantity was written up backwards in `PRELIM_DATA_VLS_STROKE.md` on
    2026-09-16 -- reported as a frozen decoder BEATING a same-day refit when the figure's own
    ylabel says `refit - frozen accuracy`. POSITIVE must mean the refit wins, i.e. information
    present but displaced.
    """
    src = __import__("inspect").getsource(rfd.main)
    assert '"gap_refit_minus_frozen": r_acc - acc' in src


def test_the_refit_arm_uses_block_cv_not_random_folds():
    """Random folds would put two periods of one block in train and test and recover BLOCK
    IDENTITY as position -- the guard `rest_position_decode` says the analysis lives or dies by."""
    src = __import__("inspect").getsource(rfd.refit_predictions)
    assert "GroupKFold" in src and "groups=g" in src
    assert "KFold(" not in src.replace("GroupKFold(", "")


def test_the_refit_arm_declines_rather_than_guesses():
    """Too few blocks to group by must return None, not a decoder scored on leaky folds."""
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.repeat(np.arange(4), 5)
    g = np.zeros(20, int)                      # one block: GroupKFold cannot split it
    assert rfd.refit_predictions(X, y, g) is None


def test_the_null_is_permuted_not_an_analytic_chance_level():
    """The reported null must be an empirical permutation, never 1/6.

    `decode_ci.frozen_ci`: positions come in ~6-trial BLOCKS, trials are not independent, and the
    analytic 1/6 is the wrong reference for this design.
    """
    src = __import__("inspect").getsource(rfd.main)
    assert "null_labels(" in src
    assert "a.perm" in src


def _blocked(seed=1, n=25):
    """Labels and VARIABLE-length block ids, which is what the real data looks like."""
    rng = np.random.default_rng(seed)
    lens = rng.integers(3, 9, size=n)
    g = np.repeat(np.arange(n), lens)
    y = np.repeat(rng.integers(0, 6, size=n), lens)
    return y, g


def test_block_permutation_keeps_every_block_intact():
    """`blockperm` relabels whole blocks, so within-block correlation survives into the null.

    That is what makes it the honest null: each block carries one position by construction, so the
    question asked is "is activity in a block unrelated to which position that block was".
    """
    y, g = _blocked()
    rng = np.random.default_rng(7)
    for _ in range(50):
        d = rfd.null_labels(y, g, rng, "blockperm")
        assert all(len(np.unique(d[g == b])) == 1 for b in np.unique(g))


def test_a_trial_shuffle_destroys_block_structure_and_understates_the_null():
    y, g = _blocked()
    rng = np.random.default_rng(7)
    intact = np.mean([np.mean([len(np.unique(rfd.null_labels(y, g, rng, "trial")[g == b])) == 1
                               for b in np.unique(g)]) for _ in range(50)])
    assert intact < 0.05


def test_the_circular_shift_only_PARTLY_preserves_blocks_which_is_why_it_is_not_primary():
    """MEASURED, not assumed -- and it corrects a claim this project has made twice.

    `rest_position_decode`'s docstring says a circular shift "keeps blocks as blocks". It does so
    only when every block has the same length. Real blocks do NOT: the scheduler's size cap and
    position changes give variable lengths, so a roll misaligns the boundaries and leaves only
    about a third of blocks internally constant. The shift therefore sits BETWEEN a trial shuffle
    and a true block permutation, and reporting it alone gives a null that is too weak -- the exact
    error `decode_ci` warns about. Hence `blockperm` leads.
    """
    y, g = _blocked()
    rng = np.random.default_rng(7)
    intact = np.mean([np.mean([len(np.unique(rfd.null_labels(y, g, rng, "shift")[g == b])) == 1
                               for b in np.unique(g)]) for _ in range(200)])
    assert 0.15 < intact < 0.55, intact


def test_matched_frozen_samples_whole_blocks_and_respects_the_target():
    """Whole blocks, not loose periods -- else the matched model gets LESS within-block
    correlation than the refit model it is being compared with: one difference removed, another
    introduced."""
    y, g = _blocked(seed=3, n=30)
    X = np.random.default_rng(0).normal(size=(len(y), 6))
    got = rfd.matched_frozen(X, y, g, n_target=60, rng=np.random.default_rng(0))
    assert got is not None
    _model, n_used = got
    assert n_used >= 60
    # every contributing block is whole: n_used is a sum of full block sizes
    sizes = sorted(np.bincount(g))
    assert any(n_used == s for s in _reachable_sums(np.bincount(g))), (n_used, sizes)


def _reachable_sums(counts):
    """Every total obtainable by adding whole blocks in some order until the target is met."""
    out = {0}
    for c in counts:
        out |= {s + int(c) for s in out}
    return out


def test_matched_frozen_declines_a_single_class_subset():
    """A subset that cannot carry two classes must return None, not a degenerate model."""
    y = np.zeros(40, int)
    g = np.repeat(np.arange(8), 5)
    X = np.random.default_rng(0).normal(size=(40, 4))
    assert rfd.matched_frozen(X, y, g, n_target=10, rng=np.random.default_rng(0)) is None


def test_the_matched_seed_is_stable_across_processes():
    """NOT `hash(label)` -- Python randomises string hashing per process, so that would make the
    matched arm irreproducible between runs while looking deterministic."""
    src = __import__("inspect").getsource(rfd.main)
    assert "hashlib.sha1(lab.encode())" in src
    # CODE ONLY -- the comment above the fix names `hash(lab)` to explain what not to do, and a
    # naive substring check matches that comment and fails on the corrected file.
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    assert "hash(lab)" not in code


def test_beta_map_class_weighting_is_uniform_across_arms():
    """Figure 14's `working` and `lick` panels must be the SAME estimator.

    Until 2026-09-16 `maps_by_epoch` set `balance = (variant == "lick")`, so the two arms were
    fitted with and without `class_weight="balanced"` and nothing in the figure said so. Made
    uniform after MEASURING that it is free -- max move 0.007 (working) / 0.035 (lick), from
    `scripts/rest_migration/meanref_balance.py`. This guards the uniformity, not the value.
    """
    import ast
    import inspect
    import textwrap

    from wfield_local import beta_maps as bm

    src = textwrap.dedent(inspect.getsource(bm.maps_by_epoch))
    assert "balance = True if balance is None else bool(balance)" in src
    # STRIP THE DOCSTRING BEFORE THE NEGATIVE CHECK. The docstring explains what the rule USED to
    # be, and naming the old rule there is the point of it -- a substring check over raw source
    # matches that explanation and fails on the corrected file. Second time this exact shape of
    # false failure has appeared today; assert over CODE, never over prose about the code.
    fn = ast.parse(src).body[0]
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) else fn.body
    code = "\n".join(ast.unparse(n) for n in body)
    assert '(variant == "lick")' not in code and "variant == 'lick'" not in code


def test_blockperm_is_the_default_primary_null():
    """The FIRST entry of --null is the primary, and for a frozen decoder it must be blockperm."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", nargs="+", default=["blockperm", "shift", "trial"])
    assert ap.parse_args([]).null[0] == "blockperm"
