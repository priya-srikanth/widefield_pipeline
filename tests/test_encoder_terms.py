"""The frozen encoder's amplitude/tuning split, on data where the answer is constructed.

Figures 6, 7, 8 and 8b have circled one question for a fortnight: did the post-stroke code MOVE, or
did it merely get SMALLER? Every measure so far answers half of it -- correlations are blind to
amplitude, distances are dominated by it. `_enc_terms` fits one gain for the session and reports
what is left, so the two are separate numbers. These tests build each case on purpose and check the
decomposition names it.
"""
import numpy as np
import pytest

from wfield_local import grant_figures as gf

L = gf.CONF_LABELS
DIM = 30


def _patterns(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return {q: rng.normal(0, scale, size=DIM) for q in L}


def test_an_unchanged_session_scores_one_with_a_gain_of_one():
    p = _patterns()
    raw, a, gain, per = gf._enc_terms(p, p)
    assert raw == pytest.approx(1.0, abs=1e-9)
    assert a == pytest.approx(1.0, abs=1e-9)
    assert gain == pytest.approx(1.0, abs=1e-9)
    assert all(v == pytest.approx(1.0, abs=1e-9) for v in per.values())


def test_a_pure_amplitude_loss_is_charged_to_amplitude_and_not_to_tuning():
    """THE CASE THE WHOLE DECOMPOSITION EXISTS FOR. Every pattern halved: the frozen encoder fails
    badly, but nothing has moved, and `gain` must say so."""
    p = _patterns()
    m = {q: 0.5 * v for q, v in p.items()}
    raw, a, gain, per = gf._enc_terms(m, p)
    assert a == pytest.approx(0.5, abs=1e-9), "the fitted gain must recover the true scaling"
    assert gain == pytest.approx(1.0, abs=1e-9), "after rescaling, nothing is left to explain"
    assert raw < 0.1, f"the raw frozen encoder should fail here, got {raw:.3f}"
    assert gain - raw > 0.8, "the failure must be attributed to AMPLITUDE"
    assert all(v == pytest.approx(1.0, abs=1e-9) for v in per.values())


def test_a_pure_tuning_change_survives_rescaling():
    """Patterns replaced with unrelated ones. No gain can rescue that, and `gain` must not try."""
    p, m = _patterns(seed=0), _patterns(seed=1)
    raw, a, gain, _per = gf._enc_terms(m, p)
    assert gain < 0.35, f"an unrelated code must not be explained by a gain, got {gain:.3f}"
    assert raw < 0, "predicting an unrelated pattern is WORSE than predicting nothing"
    assert abs(a) < 0.35, f"the best gain for an unrelated pattern collapses to ~0, got {a:.3f}"


def test_gain_minus_raw_is_not_by_itself_an_amplitude_claim():
    """THE TRAP IN THE DECOMPOSITION, pinned so the caption cannot drift back to the wrong reading.

    `gain - raw` is what rescaling RECOVERS. A halved code recovers a lot (same shape, wrong size).
    An unrelated code ALSO recovers a lot -- the fitted gain collapses towards zero and predicting
    nothing beats predicting something wrong. The two are opposite findings with a similar
    difference, and only `gain` itself separates them.
    """
    p = _patterns()
    halved = {q: 0.5 * v for q, v in p.items()}
    unrelated = _patterns(seed=1)

    r_h, _a, g_h, _p = gf._enc_terms(halved, p)
    r_u, _a2, g_u, _p2 = gf._enc_terms(unrelated, p)

    assert g_h - r_h > 0.8 and g_u - r_u > 0.8, "both recover a lot by rescaling"
    assert g_h > 0.95 and g_u < 0.35, (
        f"only `gain` separates them: halved {g_h:.3f} vs unrelated {g_u:.3f}")


def test_one_position_moving_is_localised_to_that_position():
    """far_R replaced, the rest untouched: the session-level terms drop a little, and the
    per-position term must put the loss where it happened rather than smearing it."""
    p = _patterns()
    m = dict(p)
    m["far_R"] = _patterns(seed=9)["far_R"]
    _raw, _a, _gain, per = gf._enc_terms(m, p)
    assert per["far_R"] < 0.4, f"far_R moved and must read low, got {per['far_R']:.3f}"
    others = [v for q, v in per.items() if q != "far_R"]
    assert min(others) > 0.7, f"the spared positions must stay high, got min {min(others):.3f}"


def test_a_session_wide_offset_is_charged_to_neither_term():
    """A shift shared by every position carries NO position information and the encoder is not being
    asked to predict it. Centring on the session's own across-position mean is what removes it; if
    it leaked in, every session with a different F0 would read as a tuning change."""
    p = _patterns()
    off = np.random.default_rng(4).normal(0, 3.0, size=DIM)
    m = {q: v + off for q, v in p.items()}
    raw, a, gain, _per = gf._enc_terms(m, p)
    assert raw == pytest.approx(1.0, abs=1e-9)
    assert a == pytest.approx(1.0, abs=1e-9)
    assert gain == pytest.approx(1.0, abs=1e-9)


def test_the_gain_is_one_number_for_the_session():
    """A PER-POSITION gain would absorb exactly the position-specific amplitude loss that IS the
    deficit, and the decomposition would report nothing. Here far_R alone is halved: the session
    gain must stay near 1 and the loss must show up in far_R's per-position term."""
    p = _patterns()
    m = dict(p)
    m["far_R"] = 0.5 * p["far_R"]
    _raw, a, _gain, per = gf._enc_terms(m, p)
    assert 0.85 < a < 1.0, f"one position halved must barely move the session gain, got {a:.3f}"
    assert per["far_R"] < min(v for q, v in per.items() if q != "far_R")


def test_too_few_shared_positions_is_declined():
    p = _patterns()
    raw, a, gain, per = gf._enc_terms({L[0]: p[L[0]]}, p)
    assert np.isnan(raw) and np.isnan(a) and np.isnan(gain) and per == {}


def test_gain_never_scores_below_raw():
    """`gain` refits a free parameter on the same data `raw` uses fixed at 1, so it is an upper
    bound by construction. If it ever came out lower the residual algebra would be wrong."""
    rng = np.random.default_rng(0)
    for seed in range(15):
        p = _patterns(seed=seed)
        m = {q: rng.normal(0, 1) * v + rng.normal(0, 0.4, size=DIM) for q, v in p.items()}
        raw, _a, gain, _per = gf._enc_terms(m, p)
        assert gain >= raw - 1e-9, f"seed {seed}: gain {gain:.4f} < raw {raw:.4f}"
