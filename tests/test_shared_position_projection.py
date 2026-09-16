"""Guards for `15s`, the shared-position projection.

Both of the things this family lives or dies by are silent failures: a broken circularity guard
returns a confidently positive number, and a projection referenced to the wrong baseline returns
~1.0 by construction. Neither crashes, so both are tested rather than reviewed.
"""
from __future__ import annotations

import numpy as np
import pytest

sp = pytest.importorskip("scripts.rest_migration.shared_position_projection")


def test_block_parity_splits_on_time_order_not_on_id():
    """Ids from `block_ids` are not consecutive; `id % 2` would split unevenly and could correlate
    with position if a position's blocks happened to land on one parity."""
    blocks = np.array([7, 7, 12, 12, 30, 30, 31, 31])      # ordinals 0,0,1,1,2,2,3,3
    odd, even = sp.block_parity(blocks)
    assert list(even) == [True, True, False, False, True, True, False, False]
    assert list(odd) == [not x for x in even]


def test_block_parity_is_disjoint_and_total():
    blocks = np.repeat(np.array([3, 9, 14, 22, 100]), 4)
    odd, even = sp.block_parity(blocks)
    assert not (odd & even).any()
    assert (odd | even).all()


def test_projection_is_one_when_rest_equals_the_task_map():
    """The coefficient's definition: `shared = 1` means the task map adds nothing."""
    rng = np.random.default_rng(0)
    restw = rng.normal(size=(20, 20))
    trial = restw + rng.normal(size=(20, 20))
    coef, cos = sp.projection(trial, trial, restw)
    assert coef == pytest.approx(1.0)
    assert cos == pytest.approx(1.0)


def test_projection_is_zero_for_an_orthogonal_rest_map():
    a = np.zeros(400); a[:200] = 1.0
    b = np.zeros(400); b[200:] = 1.0
    restw = np.zeros(400)
    coef, cos = sp.projection(a, b, restw)
    assert coef == pytest.approx(0.0)
    assert cos == pytest.approx(0.0)


def test_projection_separates_shape_from_magnitude():
    """A rest map with the RIGHT SHAPE but a tenth of the amplitude must give coef ~0.1 and
    cosine ~1. Reporting only a correlation would call that a perfect match, which is the reason
    both are returned."""
    rng = np.random.default_rng(1)
    restw = np.zeros(500)
    trial = rng.normal(size=500)
    coef, cos = sp.projection(0.1 * trial, trial, restw)
    assert coef == pytest.approx(0.1)
    assert cos == pytest.approx(1.0)


def test_projection_is_referenced_to_restw_not_to_rest_p():
    """Referencing the trial map to its own rest would make the projection circular.

    `projection(rest, trial, restw)` must subtract `restw` from BOTH arguments -- shifting the
    baseline changes the answer, which is what proves it is used rather than ignored.
    """
    rng = np.random.default_rng(2)
    rest = rng.normal(size=300)
    trial = rng.normal(size=300)
    a, _ = sp.projection(rest, trial, np.zeros(300))
    b, _ = sp.projection(rest, trial, np.full(300, 5.0))
    assert not np.isclose(a, b)


def test_projection_declines_on_a_degenerate_task_map():
    """Zero-norm denominator must return NaN, not inf or a silent 0."""
    z = np.zeros(300)
    coef, cos = sp.projection(np.ones(300), z, z)
    assert np.isnan(coef) and np.isnan(cos)


def test_component_space_projection_equals_pixel_space():
    """The fast path must be PROVABLY identical, not assumed.

    `projection_components` exists only because the null rebuilds a trial map per position per
    permutation, and doing that as a (345600, K) matmul made a two-window cohort run ~4 h. A
    speed-up that quietly changed the number would be worse than the slow version, so the identity
    <u(a-w), u(b-w)> = (a-w)^T (u^T u) (b-w) is tested rather than trusted.
    """
    rng = np.random.default_rng(5)
    npix, K = 800, 12
    u = rng.normal(size=(npix, K))
    a_c, b_c, w_c = rng.normal(size=K), rng.normal(size=K), rng.normal(size=K)
    G = u.T @ u
    fast = sp.projection_components(a_c, b_c, w_c, G)
    slow = sp.projection(u @ a_c, u @ b_c, u @ w_c)
    assert fast[0] == pytest.approx(slow[0], rel=1e-9)
    assert fast[1] == pytest.approx(slow[1], rel=1e-9)


def test_component_projection_declines_a_degenerate_denominator():
    K = 6
    G = np.eye(K)
    z = np.zeros(K)
    coef, cos = sp.projection_components(np.ones(K), z, z, G)
    assert np.isnan(coef) and np.isnan(cos)


def test_the_guard_takes_rest_from_odd_blocks_and_trials_from_even():
    """The circularity guard, asserted against the source.

    `rest_p` and `trial_p` are disjoint FRAMES but share the session's drift, and shared drift alone
    produces a positive projection. If this split were dropped the family would still render, with
    a confidently wrong answer.
    """
    src = __import__("inspect").getsource(sp._session_terms)
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    assert "if not odd_c[prev]:" in code          # rest restricted to ODD position-blocks
    assert "(y == c) & even_t" in code            # trials restricted to EVEN


def test_the_null_is_a_circular_shift_of_the_block_to_position_map():
    """A label shuffle destroys block structure and gives an optimistically low null."""
    src = __import__("inspect").getsource(sp._session_terms)
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    assert "np.roll(" in code
    assert "rng.permutation(" not in code


def test_the_name_cannot_be_mistaken_for_a_reference_variant():
    """`epoch_15s_shared_position_*`, never `_PERPOSref_` -- a filename implying a fourth reference
    would invite exactly the per-position referencing the scope rejects."""
    src = __import__("inspect").getsource(sp.main)
    assert "epoch_15s_shared_position_" in src
    assert "ref_" not in src.split("csv")[0].lower().replace("reference", "")
