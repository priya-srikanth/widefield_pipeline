"""The frozen-vs-refit arm: the paired record, the gap statistic, and what must NOT have moved.

The figure's whole claim rests on the two arms scoring the SAME trials -- if they do not, the
difference is a comparison of two trial sets and means nothing. These tests pin that, pin the
pairing through the block bootstrap, and pin that adding the modes left the frozen arm alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import epoch_figures as ef
from wfield_local import epoch_grant_figures as eg
from wfield_local import grant_figures as G


# --------------------------------------------------------------------------- the gap statistic

def test_gap_is_refit_minus_frozen_at_one_position():
    y = np.array([1, 1, 1, 1, 1, 1, 2, 2])
    #            frozen right 2/6, refit right 5/6 at position 1
    p = np.column_stack([np.array([1, 1, 3, 3, 3, 3, 2, 2]),
                         np.array([1, 1, 1, 1, 1, 3, 2, 2])])
    assert eg._gap_at(y, p, 1) == pytest.approx(5 / 6 - 2 / 6)
    assert eg._refit_at(y, p, 1) == pytest.approx(5 / 6)


def test_gap_is_none_when_the_class_is_too_thin():
    y = np.array([1, 1, 2, 2, 2, 2, 2, 2])
    p = np.column_stack([y, y])
    assert eg._gap_at(y, p, 1) is None          # 2 trials < the 5-trial floor
    assert eg._gap_at(y, p, 9) is None          # absent class
    assert eg._gap_at(y, p, None) is None


def test_gap_refuses_an_unpaired_record():
    """A 1-D prediction column silently taken as 'frozen' would report a gap of zero everywhere."""
    y = np.array([1] * 8)
    with pytest.raises(ValueError, match="paired"):
        eg._gap_at(y, y, 1)


def test_zero_gap_when_both_arms_agree():
    """Both decoders failing together is the DEGRADED reading, and it must read as 0, not as NaN."""
    y = np.array([1] * 8)
    wrong = np.array([4] * 8)
    assert eg._gap_at(y, np.column_stack([wrong, wrong]), 1) == 0.0


# ------------------------------------------------------------------- pairing through the bootstrap

def _paired_animal(rng, n_block=12, per_block=6):
    """One animal: pre = list of per-session records, one post day. Refit better by construction."""
    def rec(seed):
        r = np.random.default_rng(seed)
        y = np.repeat(np.arange(6), n_block * per_block // 6)
        blk = np.repeat(np.arange(n_block), per_block)[: len(y)]
        frozen = np.where(r.random(len(y)) < 0.3, y, (y + 1) % 6)
        refit = np.where(r.random(len(y)) < 0.9, y, (y + 1) % 6)
        return (y, np.column_stack([frozen, refit]), blk)
    return ([rec(1), rec(2)], {3: rec(4)})


def test_value_draws_survive_a_two_column_prediction():
    """`_blocks_of` masks and `pool_records` concatenates -- both must accept an (n, 2) p."""
    per_animal = {"PSxx": _paired_animal(np.random.default_rng(0))}
    got = ef.value_draws(per_animal, "pre",
                         lambda y, p: eg._gap_at(y, p, 0),
                         rng=np.random.default_rng(0), n_boot=200)
    assert got is not None
    point, draws = got
    assert point > 0.3                      # refit is better by construction
    assert len(draws) > 50
    assert np.isfinite(draws).all()


def test_pooling_keeps_the_two_columns_aligned():
    per_animal = {"PSxx": _paired_animal(np.random.default_rng(0))}
    y, p, _b = ef.pool_records(per_animal, "pre")
    assert p.ndim == 2 and p.shape == (len(y), 2)
    # the pooled gap equals the trial-weighted mean of the per-session gaps
    recs = per_animal["PSxx"][0]
    per_sess = [(len(r[0]), eg._gap_at(r[0], r[1], 0)) for r in recs]
    frozen_hits = sum(np.sum(r[1][r[0] == 0, 0] == 0) for r in recs)
    refit_hits = sum(np.sum(r[1][r[0] == 0, 1] == 0) for r in recs)
    n = sum(np.sum(r[0] == 0) for r in recs)
    assert eg._gap_at(y, p, 0) == pytest.approx((refit_hits - frozen_hits) / n)
    assert all(v is not None for _n, v in per_sess)


# --------------------------------------------------------------------------- the collector's modes

def test_collect_5c_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="frozen/refit/paired"):
        G._collect_5c("cue", "working", "whatever")


def test_frozen_mode_is_still_the_default():
    """Every existing caller passes two arguments; the default must not have become a paired record."""
    import inspect

    sig = inspect.signature(G._collect_5c)
    assert sig.parameters["mode"].default == "frozen"
    assert sig.parameters["variant"].default == "working"


def test_refit_pred_returns_none_rather_than_guessing():
    """A session that cannot support the split gets None, not a fallback to some other model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def pipe():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.zeros(20, int)                                  # one class
    assert G._refit_pred(pipe, X, y, np.arange(20)) is None
    y2 = np.tile([0, 1], 10)
    assert G._refit_pred(pipe, X, y2, np.zeros(20, int)) is None   # one block


def test_refit_pred_is_out_of_sample():
    """Predictions must come from folds that did not see the trial -- otherwise the gap is fitted."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def pipe():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))                # pure noise: no information at all
    y = np.repeat(np.arange(6), 20)
    blk = np.repeat(np.arange(12), 10)
    pred = G._refit_pred(pipe, X, y, blk)
    assert pred is not None and pred.shape == y.shape
    # an in-sample fit on 120x4 noise would score far above 1/6; a block-CV one cannot
    assert float(np.mean(pred == y)) < 0.45


# ------------------------------------------------------------------------------ the figure wiring

def test_reference_line_is_not_a_chance_level():
    """`reference` draws a zero line without relabelling the axis '(chance 0.00)'."""
    import inspect

    assert "reference" in inspect.signature(ef.bar_row).parameters
    src = inspect.getsource(eg._frozen_vs_refit)
    assert "reference=0.0" in src
    assert "chance=None" in src


def test_5r_is_an_arm_key_and_a_cli_choice():
    import inspect

    src = inspect.getsource(eg.main)
    assert '"5r"' in src
    assert 'ARM_KEYS = {"acc", "5c", "5r", "mat", "scal"}' in src


def test_pre_panel_counts_sessions_not_records():
    """The refit arm's pre entry is a LIST; counting it as one would print 'pre 4' under n=44."""
    import inspect

    src = inspect.getsource(eg._position_bars)
    assert "_n_pre" in src and "isinstance(pre, list)" in src
