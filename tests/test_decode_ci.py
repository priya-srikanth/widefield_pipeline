"""Bootstrap CIs must respect the block structure, or they lie about precision.

Trials arrive in runs of ~6 at one position. Resampling individual trials treats those as independent
and produces intervals that are far too narrow; resampling BLOCKS preserves the dependence.
"""
from __future__ import annotations

import numpy as np

from wfield_local.decode_ci import bootstrap_recall, reference_band
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER


def _synthetic(n_blocks=40, per_block=6, acc=0.8, seed=0):
    """Blocked trials: one position per block, decoder correct with probability `acc`."""
    rng = np.random.default_rng(seed)
    y, pred, blk = [], [], []
    for b in range(n_blocks):
        p = DISPLAY_ORDER[b % len(DISPLAY_ORDER)]
        for _ in range(per_block):
            y.append(p); blk.append(b)
            if rng.random() < acc:
                pred.append(p)
            else:
                pred.append(int(rng.choice([q for q in DISPLAY_ORDER if q != p])))
    return np.array(y), np.array(pred), np.array(blk)


def test_point_estimate_matches_plain_accuracy():
    y, pred, blk = _synthetic()
    r = bootstrap_recall(y, pred, blk, n_boot=200)
    assert abs(r["accuracy"] - (y == pred).mean()) < 1e-12
    assert r["n_trials"] == len(y)
    assert r["n_effective"] == len(np.unique(blk)), "resampling units are BLOCKS, not trials"


def test_ci_brackets_the_point_estimate():
    y, pred, blk = _synthetic()
    r = bootstrap_recall(y, pred, blk, n_boot=500)
    lo, hi = r["accuracy_ci"]
    assert lo <= r["accuracy"] <= hi
    for i in range(len(r["positions"])):
        assert r["recall_ci_lo"][i] <= r["recall"][i] <= r["recall_ci_hi"][i]


def test_trial_bootstrap_is_overconfident_vs_block_bootstrap():
    """THE POINT OF THE MODULE: ignoring clustering narrows the interval dishonestly."""
    y, pred, blk = _synthetic(n_blocks=40, per_block=8)
    rb = bootstrap_recall(y, pred, blk, n_boot=800, by="block", seed=1)
    rt = bootstrap_recall(y, pred, blk, n_boot=800, by="trial", seed=1)
    wb = rb["accuracy_ci"][1] - rb["accuracy_ci"][0]
    wt = rt["accuracy_ci"][1] - rt["accuracy_ci"][0]
    assert wb > wt, f"block CI ({wb:.4f}) must be WIDER than trial CI ({wt:.4f})"


def test_more_blocks_narrows_the_interval():
    wide = bootstrap_recall(*_synthetic(n_blocks=12, seed=2), n_boot=600)
    narrow = bootstrap_recall(*_synthetic(n_blocks=96, seed=2), n_boot=600)
    assert ((narrow["accuracy_ci"][1] - narrow["accuracy_ci"][0])
            < (wide["accuracy_ci"][1] - wide["accuracy_ci"][0]))


def test_chance_decoder_ci_covers_chance():
    y, pred, blk = _synthetic(n_blocks=60, acc=1 / 6, seed=3)
    r = bootstrap_recall(y, pred, blk, n_boot=800)
    lo, hi = r["accuracy_ci"]
    assert lo <= r["chance"] <= hi, "a chance decoder's CI must cover 1/6"


def test_blocks_required_for_cluster_bootstrap():
    y, pred, _ = _synthetic()
    try:
        bootstrap_recall(y, pred, None, n_boot=10)
    except ValueError as e:
        assert "blocks are required" in str(e)
    else:
        raise AssertionError("must refuse a cluster bootstrap without blocks")


def test_reference_band_spans_the_sessions():
    cis = [bootstrap_recall(*_synthetic(seed=s, acc=0.7 + 0.02 * s), n_boot=200) for s in range(5)]
    band = reference_band(cis)
    pts = [c["accuracy"] for c in cis]
    assert band["n_sessions"] == 5
    assert band["min"] <= band["median"] <= band["max"]
    assert abs(band["min"] - min(pts)) < 1e-12 and abs(band["max"] - max(pts)) < 1e-12
    assert band["widest_ci"][0] <= band["min"] and band["widest_ci"][1] >= band["max"]


def test_frozen_ci_resamples_SESSIONS_not_trials():
    """A held-out DAY is the independent replicate for a LOSO decoder. Bootstrapping trials would
    treat ~500 correlated trials from one session as 500 observations and give a spuriously tight
    interval, which is how a cross-day claim gets overstated."""
    import numpy as np

    from wfield_local.decode_ci import frozen_ci

    rng = np.random.default_rng(0)
    sess = np.repeat(np.arange(8), 60)
    blk = np.tile(np.repeat(np.arange(10), 6), 8)
    y = np.tile(np.repeat(rng.integers(0, 6, 10), 6), 8)
    pred = np.where(rng.random(len(y)) < 0.5, y, rng.integers(0, 6, len(y)))
    r = frozen_ci(y, pred, sess, blk, n_boot=300, n_perm=300)
    assert r["n_sessions"] == 8, "the resampling unit must be the session"
    assert r["ci_lo"] < r["accuracy"] < r["ci_hi"]


def test_frozen_ci_empirical_chance_is_not_assumed_to_be_one_sixth():
    """The analytic 1/6 ignores block structure and the prediction distribution. The permutation null
    is the honest reference, and it does NOT come out at 1/6."""
    import numpy as np

    from wfield_local.decode_ci import frozen_ci

    rng = np.random.default_rng(1)
    sess = np.repeat(np.arange(6), 60)
    blk = np.tile(np.repeat(np.arange(10), 6), 6)
    y = np.tile(np.repeat(rng.integers(0, 6, 10), 6), 6)
    pred = np.full_like(y, 3)                      # a degenerate decoder that always says "3"
    r = frozen_ci(y, pred, sess, blk, n_boot=200, n_perm=400)
    assert r["chance_analytic"] == 1 / 6
    assert np.isfinite(r["chance_empirical"])
    # a constant predictor cannot beat its own permutation null
    assert r["p_perm"] > 0.01, "a degenerate predictor must not read as above chance"


def test_frozen_ci_without_blocks_reports_nan_chance_rather_than_guessing():
    import numpy as np

    from wfield_local.decode_ci import frozen_ci

    sess = np.repeat(np.arange(4), 20)
    y = np.tile(np.arange(5), 16)
    r = frozen_ci(y, y.copy(), sess, blocks=None, n_boot=100, n_perm=10)
    assert np.isnan(r["chance_empirical"]) and np.isnan(r["p_perm"])


def test_block_bootstrap_understates_the_interval_when_days_differ():
    """Blocks are NESTED in sessions. Resampling them holds the set of days effectively fixed, so the
    CI sees only within-day noise and cannot see that a different set of days would land elsewhere.

    With real between-day variability the block interval comes out ~3x too narrow, while the nested
    bootstrap (sessions, then blocks within each drawn session) recovers the session-level width. This
    is why the default unit is the SESSION: for a frozen decoder the held-out DAY is the replicate, and
    post-stroke application is by definition a new day.
    """
    import numpy as np

    from wfield_local.decode_ci import frozen_ci

    rng = np.random.default_rng(0)
    sess = np.repeat(np.arange(11), 300)
    blk = np.tile(np.repeat(np.arange(50), 6), 11)
    y = np.tile(np.repeat(rng.integers(0, 6, 50), 6), 11)
    day_acc = rng.uniform(0.30, 0.62, 11)                 # days genuinely differ
    pred = np.empty_like(y)
    for s in range(11):
        m = sess == s
        pred[m] = np.where(rng.random(m.sum()) < day_acc[s], y[m], rng.integers(0, 6, m.sum()))

    w = {}
    for by in ("session", "block", "nested"):
        r = frozen_ci(y, pred, sess, blk, n_boot=800, n_perm=1, by=by)
        w[by] = r["ci_hi"] - r["ci_lo"]
        assert r["ci_unit"] == by
    assert w["block"] < 0.5 * w["session"], f"block CI should be far narrower: {w}"
    assert w["nested"] > 0.8 * w["session"], f"nested must recover the between-day width: {w}"


def test_block_and_nested_require_block_ids():
    import numpy as np
    import pytest

    from wfield_local.decode_ci import frozen_ci

    sess = np.repeat(np.arange(4), 30)
    y = np.tile(np.arange(6), 20)
    for by in ("block", "nested"):
        with pytest.raises(ValueError):
            frozen_ci(y, y.copy(), sess, blocks=None, n_boot=50, n_perm=1, by=by)
