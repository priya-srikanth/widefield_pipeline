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
