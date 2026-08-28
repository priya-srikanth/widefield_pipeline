"""Every pooled session gets a per-session engaged number, including post-stroke ones.

WHY (2026-08-28). fd67f63 restricted the ENGAGED arm's cross-validation to pre-stroke rows -- the
right fix, and the same one the frozen decoder got -- but the per-session loop kept indexing those
predictions with positions in the FULL pool. The first post-stroke session therefore indexed past
the end of a pre-only array and the entire nightly no-lick stage died on an IndexError, ~10 minutes
in, having written nothing.

The bug needed three things at once to appear: post-stroke sessions in the pool, a pre-only
training restriction, and a per-session loop over all of them. That combination first existed on
2026-08-28, so no existing test covered it and the failure surfaced only as a crashed nightly.

WHAT IS PINNED. Not just "does not crash" -- that would pass if post-stroke sessions were silently
dropped, which would quietly delete the comparison the module exists to make (`nolick_decoder`
line 474: "Post-stroke sessions in the pool are still SCORED, by a model that never saw one. That
is the measurement."). So this asserts COVERAGE: a per-session entry exists for post-stroke labels
too, and the pre-stroke predictions are still the leave-one-session-out ones rather than being
overwritten by the frozen model's.
"""
import numpy as np
import pytest

nd = pytest.importorskip("wfield_local.nolick_decoder")


def test_pooled_predictions_span_every_session_not_just_the_training_rows():
    """The assembly at the heart of the fix, in isolation from the analysis it feeds."""
    rng = np.random.default_rng(0)
    n_pre, n_post = 60, 25
    YE = rng.integers(0, 6, n_pre + n_post)
    m_pre = np.zeros(n_pre + n_post, dtype=bool)
    m_pre[:n_pre] = True

    pred_e = rng.integers(0, 6, n_pre)          # LOSO predictions: PRE rows only
    frozen = rng.integers(0, 6, n_post)         # frozen model on the POST rows

    pred_pooled = np.empty_like(YE)
    pred_pooled[m_pre] = pred_e
    pred_pooled[~m_pre] = frozen

    assert pred_pooled.shape == YE.shape, "one prediction per pooled trial, not per training trial"
    assert np.array_equal(pred_pooled[m_pre], pred_e), "pre rows keep their leave-one-session-out value"
    assert np.array_equal(pred_pooled[~m_pre], frozen), "post rows are scored by the frozen model"

    # The failure this replaces: indexing the pre-only array with a full-pool position.
    last = np.flatnonzero(~m_pre)[-1]
    with pytest.raises(IndexError):
        pred_e[last]
    pred_pooled[last]   # must not raise


def test_the_per_session_loop_indexes_the_pooled_array():
    """AST guard, not a text search: `pred_e[ie]` is the exact expression that crashed the nightly.

    Resolved structurally because the comment above the fix quotes the broken expression verbatim --
    a grep over source text matches the explanation as readily as a regression, and a guard that
    fires on its own documentation is one that gets deleted rather than fixed.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(nd.analyse_animal)))
    subscripts = {
        (n.value.id, n.slice.id)
        for n in ast.walk(tree)
        if isinstance(n, ast.Subscript)
        and isinstance(n.value, ast.Name) and isinstance(n.slice, ast.Name)
    }
    assert ("pred_e", "ie") not in subscripts, (
        "the per-session loop is indexing the PRE-ONLY predictions with full-pool positions again; "
        "it must index the pooled array so post-stroke sessions are scored rather than out of range")
    assert ("pred_pooled", "ie") in subscripts, (
        "per-session engaged arm must read the pooled predictions")


def test_the_raw_arms_handed_to_dissociation_ci_are_not_ragged():
    """`dissociation_ci` indexes y/pred/sess by the SAME positions, so they must share a length.

    The second failure from fd67f63: the tuple carried full-pool `YE`/`SE` beside a pre-only
    prediction column, and `pe[ie]` ran off the end ~10 minutes into the nightly -- past where the
    first fix had already got it. A length check here is the cheapest statement of the invariant
    that both bugs violated.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(nd.analyse_animal)))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value == "engaged" for k in node.keys)):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "engaged" and isinstance(v, ast.Tuple):
                names = [e.func.value.id for e in v.elts
                         if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute)
                         and isinstance(e.func.value, ast.Name)]
                if "YE" in names:
                    assert "pred_e" not in names, (
                        "the _raw engaged tuple pairs full-pool YE/SE with the PRE-ONLY prediction "
                        "column again; dissociation_ci indexes all three by the same positions")
                    assert "pred_pooled" in names
                    return
    pytest.fail("could not find the _raw engaged tuple to check")
