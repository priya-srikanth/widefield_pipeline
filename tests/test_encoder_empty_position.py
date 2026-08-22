"""A position the animal abandoned must not delete the figure.

ROOT CAUSE of the FEVE collapse (2026-08-20 slide 21 "empty"; reproduced in the 2026-08-21 nightly
as "0 region(s) from 58 session(s)"):

    mm = y == p                 # p has NO trials -- the animal stopped going there
    mu = X[mm].mean(0)          # nan, with a "Mean of empty slice" warning
    betw += mm.sum() * (mu - gm) ** 2

``mm.sum()`` is 0, so this looks like it adds nothing. It adds ``0 * nan``, which is **nan**. Every
component of ``betw`` becomes nan, so that session's ``expl`` is nan for every region. Pooling sums
it into the shared axis, and ``nan > floor`` is False for EVERY region -- so one session with one
abandoned position deleted a 58-session, 64-region heatmap.

It arrived with the science: abandoning a position outright is the post-stroke phenotype, so the
bug could not appear until the lesions did.

Two layers are tested here: the arithmetic no longer produces nan, and the pooled axis survives a
nan even if some other path ever produces one again.
"""
import numpy as np

from wfield_local import locanmf_position_encoder as enc


def _betw(y, X, skip_empty):
    """The exact loop, with and without the fix."""
    gm = X.mean(0)
    betw = np.zeros(X.shape[1])
    for p in (0, 1, 2):
        mm = y == p
        if skip_empty and not mm.any():
            continue
        with np.errstate(invalid="ignore"):
            mu = X[mm].mean(0)
        betw += mm.sum() * (mu - gm) ** 2
    return betw


def test_the_old_arithmetic_really_did_produce_nan():
    """Pinning the mechanism, so nobody 'simplifies' the guard away later."""
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)          # position 2 never occurs
    assert np.isnan(_betw(y, X, skip_empty=False)).all()


def test_skipping_an_empty_position_keeps_the_sum_finite():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)
    b = _betw(y, X, skip_empty=True)
    assert np.isfinite(b).all() and (b > 0).any()


def test_an_empty_position_contributes_exactly_nothing():
    """Not merely finite -- the answer must equal the one with that class simply absent."""
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)
    with_empty = _betw(y, X, skip_empty=True)          # pos 2 present in the loop, no trials
    gm = X.mean(0)
    manual = np.zeros(2)
    for p in (0, 1):                                   # only the observed classes
        mm = y == p
        manual += mm.sum() * (X[mm].mean(0) - gm) ** 2
    assert np.allclose(with_empty, manual)


# ---------------------------------------------------------------- the pooled axis

def test_one_nan_session_no_longer_deletes_every_region():
    """Defence in depth: even if some path produces nan again, the axis must survive."""
    good = {f"R{i}": dict(expl=5.0, tot=10.0) for i in range(12)}
    res = {f"PS9{i % 4 + 2}_08{i:02d}": dict(good) for i in range(57)}
    res["PS94_0819"] = {f"R{i}": dict(expl=float("nan"), tot=10.0) for i in range(12)}
    regs = enc._feve_regions(res, floor=0.02)
    assert len(regs) == 12, "one poisoned session must not empty the axis"


def test_a_nan_total_is_also_survived():
    good = {"MOp": dict(expl=5.0, tot=10.0)}
    res = {f"s{i}": dict(good) for i in range(5)}
    res["bad"] = {"MOp": dict(expl=5.0, tot=float("nan"))}
    assert enc._feve_regions(res, floor=0.02) == ["MOp"]


def test_regions_genuinely_below_the_floor_are_still_dropped():
    """The guard must not turn into 'keep everything'."""
    res = {"s0": {"loud": dict(expl=5.0, tot=10.0), "quiet": dict(expl=0.001, tot=10.0)}}
    assert enc._feve_regions(res, floor=0.02) == ["loud"]
