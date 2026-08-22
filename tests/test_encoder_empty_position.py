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


# ---------------------------------------------------------------- per-position EV

def _ev(y, sstot, ssres, p, guard):
    """The scoring expression from ev_per_position, with and without the guard."""
    m = y == p
    if guard and not m.any():
        return np.nan
    return 1 - ssres[m].sum() / max(sstot[m].sum(), 1e-12)


def test_an_unvisited_position_used_to_score_a_perfect_one():
    """PS94_0820 far_R: ZERO engaged trials, reported EV = 1.0.

    Both sums are 0.0 over an empty mask, so `1 - 0.0 / max(0.0, 1e-12)` is exactly 1.0 -- a
    perfect score for a position the animal never visited, feeding the per-position x date EV
    matrix, where it would be the brightest cell in the figure."""
    y = np.array([0] * 5 + [1] * 5)
    sstot = np.ones(10); ssres = np.ones(10) * 0.5
    assert _ev(y, sstot, ssres, 2, guard=False) == 1.0


def test_an_unvisited_position_is_now_missing_not_perfect():
    y = np.array([0] * 5 + [1] * 5)
    sstot = np.ones(10); ssres = np.ones(10) * 0.5
    assert np.isnan(_ev(y, sstot, ssres, 2, guard=True))


def test_a_visited_position_is_unchanged_by_the_guard():
    y = np.array([0] * 5 + [1] * 5)
    sstot = np.ones(10); ssres = np.ones(10) * 0.5
    assert _ev(y, sstot, ssres, 0, guard=True) == _ev(y, sstot, ssres, 0, guard=False) == 0.5


def test_a_single_trial_position_still_reports_a_number():
    """One trial is noisy, not absent. Dropping it would hide a real (if weak) measurement."""
    y = np.array([0] * 9 + [1])
    sstot = np.ones(10); ssres = np.ones(10) * 0.25
    assert _ev(y, sstot, ssres, 1, guard=True) == 0.75


# ---------------------------------------------------------------- no fourth copy

def test_per_position_ev_is_computed_in_exactly_one_place():
    """A duplicate outlives the fix to its twin.

    ev_per_position was created to be the single implementation, folding in
    fig_ev_by_position_animal and the EV matrix. It missed fig_ev_by_position, which kept its own
    inline copy -- so when the empty-position guard was added on 2026-08-22, the per-date figure
    went on reporting PS94_0820 far_R = 1.0 for a position with ZERO engaged trials, and the
    published deck carried it. Two of three call sites were fixed and the figure still lied.
    """
    import ast
    import inspect

    from wfield_local import locanmf_position_encoder as m

    src = inspect.getsource(m)
    tree = ast.parse(src)
    owners = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(src, node) or ""
            if "ssres[" in seg and "max(sstot" in seg:
                owners.append(node.name)
    assert owners == ["ev_per_position"], (
        f"per-position EV is computed in {owners}; it must live only in ev_per_position so a guard "
        f"added there reaches every figure")
