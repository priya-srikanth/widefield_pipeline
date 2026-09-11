"""The encoder CEILING: what the frozen encoder's R^2 is failing against.

`frozen EV` is an R^2 and acutely it is -0.388 post-cue -- "worse than predicting the mean", and
mute about what was achievable. `EV after rescale` frees the amplitude only, and `_enc_terms` warns
in its own docstring that a large rescaling gain means amplitude ONLY when the rescaled value is
high, because a code that is simply gone also recovers a lot. That ambiguity is what forced the
withdrawal of the "half amplitude, half shape" reading on 2026-09-10, and these tests pin the
construction that resolves it.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import grant_figures as G


def _session(rng, n=80, n_feat=12, tuning=1.0, noise=1.0):
    """One synthetic session: six positions with a fixed random tuning plus per-trial noise."""
    labels = list(G.CONF_LABELS)
    centres = {q: rng.normal(size=n_feat) * tuning for q in labels}
    return {q: centres[q] + rng.normal(size=(n, n_feat)) * noise for q in labels}


def test_a_refit_encoder_without_cross_validation_is_one_by_construction():
    """The reason this is a SPLIT-half and not simply 'refit on the session'.

    Ridge on a one-hot position design reduces to the per-position mean, so predicting a session's
    own means from themselves is an identity, not a prediction. If that ever scores below 1.0 the
    scorer has changed and this whole family means something else.
    """
    rng = np.random.default_rng(0)
    pat = _session(rng)
    means = G._means(pat)
    raw, _a, _gain, _per = G._enc_terms(means, means)
    assert raw == pytest.approx(1.0, abs=1e-9)


def test_the_ceiling_is_high_when_the_session_has_structure():
    rng = np.random.default_rng(1)
    strong = G._enc_ceiling(_session(rng, tuning=2.0, noise=1.0), np.random.default_rng(2))
    assert strong[0] > 0.5, f"a strongly tuned session should predict itself, got {strong[0]:.3f}"


def test_the_ceiling_collapses_when_there_is_nothing_to_predict():
    """Pure noise has no position structure, so no template -- its own included -- can predict it."""
    rng = np.random.default_rng(3)
    flat = G._enc_ceiling(_session(rng, tuning=0.0, noise=1.0), np.random.default_rng(4))
    assert flat[0] < 0.2, f"noise should have no ceiling, got {flat[0]:.3f}"


def test_the_ceiling_separates_a_MOVED_code_from_a_LOST_one():
    """The discrimination the frozen EV alone cannot make, on two sessions it scores identically.

    MOVED: strong tuning, but a different tuning from the reference -- the frozen template fails and
    the ceiling is high. LOST: no tuning at all -- the frozen template fails and the ceiling is low.
    `frozen EV` is poor in both; only the ceiling tells them apart.
    """
    rng = np.random.default_rng(5)
    ref = G._means(_session(rng, tuning=2.0, noise=0.3))

    moved = _session(np.random.default_rng(6), tuning=2.0, noise=0.3)
    lost = _session(np.random.default_rng(7), tuning=0.0, noise=0.3)

    frozen_moved = G._enc_terms(G._means(moved), ref)[0]
    frozen_lost = G._enc_terms(G._means(lost), ref)[0]
    assert frozen_moved < 0.2 and frozen_lost < 0.2, "the frozen template should fail on both"

    ceil_moved = G._enc_ceiling(moved, np.random.default_rng(8))[0]
    ceil_lost = G._enc_ceiling(lost, np.random.default_rng(9))[0]
    assert ceil_moved - ceil_lost > 0.4, (
        f"the ceiling did not separate moved ({ceil_moved:.3f}) from lost ({ceil_lost:.3f})")


def test_both_orderings_are_averaged():
    """A-vs-B and B-vs-A are two estimates of one quantity; using one is estimation noise drawn as
    structure, the argument `_split_half_matrix` already makes."""
    import inspect

    src = inspect.getsource(G._enc_ceiling)
    assert "((A, B), (B, A))" in src


def test_the_ceiling_is_pessimistic_and_says_so():
    """It carries half-session noise on BOTH sides where the frozen arm has it on one.

    On the real pre-cue arm this makes the frozen EV (0.330) EXCEED its own ceiling (0.090), which
    is impossible for a true ceiling and is the signal that the construction has run out of signal.
    The docstring has to keep saying so, because a reader who takes it as a ceiling in that arm will
    conclude the template is better than achievable.
    """
    import inspect

    doc = inspect.getdoc(G._enc_ceiling) or ""
    assert "PESSIMISTIC" in doc
    assert "BELOW" in doc


def test_pre_is_averaged_per_session_not_computed_on_the_pool():
    """A ceiling on ten pooled sessions measures something no single post-stroke session can reach."""
    import inspect

    doc = inspect.getdoc(G._enc_ceiling_tables) or ""
    assert "PER-SESSION" in doc and "pooled" in doc
