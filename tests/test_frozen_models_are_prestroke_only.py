"""A frozen model must never be trained on a post-stroke session.

THE BUG (found 2026-08-26, confirmed by Priya: "the frozen models were supposed to be pre stroke
only"). `pooled_frozen_loso` predicted with

    cross_val_predict(_pipe(), XE, YE, cv=LeaveOneGroupOut(), groups=GE)

over EVERY pooled session, and `pooled_frozen_encoder` used `tr = ~te`. Both were correct when
written -- every curated session was pre-stroke, because the strokes had not happened. As post-stroke
nights registered they joined the pool silently, and by 8/26 six of PS92's 18 pooled sessions were
post-stroke: ~30% of the training data behind a number whose entire purpose is to be a lesion-free
baseline. The reported transfer cost had already drifted from the +0.140 in DECISIONS (11 pre-stroke
sessions) to +0.123.

WHY IT MATTERS BEYOND TIDINESS. The argument in DECISIONS is: transfer cost is positive pre-stroke,
so a frozen decoder does not decay across days on its own, so post-stroke degradation is attributable
to the lesion. That inference needs the reference to be lesion-free. For the ENCODER it is worse --
its residual on post-stroke trials IS the representational-change readout, so training on those
trials fits the model to the very data whose departure from it is the result.

WHAT IS DELIBERATELY NOT RESTRICTED: pooling for FEATURE ALIGNMENT still uses every session. That is
what puts post-stroke sessions in a comparable feature space at all. Only training is pre-only.
"""
import inspect

from wfield_local import locanmf_frozen_decoder as fd


def _code(fn) -> str:
    """Source with comments and blank lines stripped.

    These tests search for the OLD buggy expressions, and the fix deliberately quotes those same
    expressions in comments to explain what changed. Searching raw source would therefore fail on the
    very documentation that makes the fix understandable -- and the obvious "fix" would be to delete
    the explanation. Strip comments and assert against the code.
    """
    out = []
    for line in inspect.getsource(fn).splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if stripped:
            out.append(stripped)
    return "\n".join(out)


def test_decoder_no_longer_trains_on_every_pooled_session():
    code = _code(fd.pooled_frozen_loso)
    assert "LeaveOneGroupOut()" not in code, (
        "the decoder is back to leave-one-out over ALL pooled sessions, which trains on post-stroke "
        "data")
    assert "session_phase" in code, "no phase filter in the decoder's training set"


def test_encoder_no_longer_trains_on_every_other_session():
    code = _code(fd.pooled_frozen_encoder)
    assert "tr = ~te" not in code, "encoder trains on every other session, post-stroke included"
    assert "session_phase" in code, "no phase filter in the encoder's training set"


def test_the_result_declares_what_it_was_trained_on():
    """The contamination stayed invisible because nothing in the output said what the pool was.
    A consumer must never again have to infer it."""
    src = inspect.getsource(fd.pooled_frozen_loso)
    for key in ('"training_phase"', '"n_pre_sessions"', '"n_post_sessions"', '"pre_labels"'):
        assert key in src, f"the result does not record {key}"


def test_the_reference_band_is_computed_on_prestroke_sessions_only():
    """transfer_cost = loso - mean_within, and BOTH terms must be pre-stroke. A mixed mean_within
    would quietly re-import the contamination through the other side of the subtraction."""
    src = inspect.getsource(fd.pooled_frozen_loso)
    assert "within_pre" in src, "mean_within is not restricted to pre-stroke sessions"
    assert "m_pre" in src, "loso_accuracy is not restricted to pre-stroke sessions"


def test_too_few_prestroke_sessions_is_refused_not_approximated():
    """With <2 pre-stroke sessions there is no cross-day reference to be had. Returning a number
    computed some other way would be worse than returning nothing."""
    for fn in (fd.pooled_frozen_loso, fd.pooled_frozen_encoder):
        src = inspect.getsource(fn)
        assert "len(pre_i) < 2" in src, f"{fn.__name__} does not refuse an unbuildable reference"


def test_post_stroke_sessions_are_still_scored():
    """The fix must not delete the post-stroke rows -- they are the point. Every pooled session keeps
    a per_session entry; only the TRAINING set is restricted."""
    src = inspect.getsource(fd.pooled_frozen_loso)
    assert "for i, lab in enumerate(kept)" in src, (
        "per_session no longer iterates every pooled session, so post-stroke rows -- and the deck "
        "figures built from them -- would disappear")
