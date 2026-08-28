"""The `_trial_features` disk cache: what its key must separate, and what it must refuse to key.

WHY A DISK CACHE AND NOT `lru_cache` (measured 2026-08-27). `_trial_features` is the per-session
workhorse every downstream analysis is built from and had no memoisation at any level: ~5 rebuilds per
distinct session in a grant render, and the uncached calls account for roughly 6 of the nightly's
9.62 h on top of most of the 8-10 h render. The nightly is 17 SEPARATE PROCESSES -- `cli()` shells out
to `python -m <module>` -- so an `lru_cache` is discarded at every step boundary and would fix only
the render. Only `session_cache` crosses a process, and it also carries results between NIGHTS, which
is where most of the win is.

WHY THE KEY IS THE DANGEROUS PART. `_trial_features` feeds the decoder, the encoder, RSA and
cross-mouse, so a key that conflates two calls is wrong numbers everywhere -- silently, across
processes, persisting for days. These tests target exactly the collisions that would do that.
"""
import types

import pytest

from wfield_local import locanmf_position_decoder as D

_UNSAFE_IN_A_FILENAME = set('/\\:*?"<>| ')


def args(align="cue", post_s=2.0, source="locanmf", **kw):
    base = dict(align=align, source=source, post_s=post_s, pre_s=1.0, fs=30.0, max_rt=2.0,
                baseline="precue", bins=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def kind(a, *, signal_key="own", nolick_ref="cue", with_precue_licks=False, with_indices=False):
    return D.feature_cache_kind(a, signal_key=signal_key, nolick_ref=nolick_ref,
                                with_precue_licks=with_precue_licks, with_indices=with_indices)


# ---------------------------------------------------------------------------------------------
# What the key must SEPARATE.
# ---------------------------------------------------------------------------------------------

def test_signal_provenance_separates_the_key():
    """THE one that matters most. The same session and the same args give completely different
    features depending on whether the signal is that session's own LocaNMF fit or a projection onto
    a shared joint basis -- and `session_signature` stats neither the basis nor anything moving with
    it. Without this the cache would serve one for the other, across processes, for days."""
    assert kind(args()) != kind(args(), signal_key="basis:76d884873920")
    assert kind(args(), signal_key="basis:aaa") != kind(args(), signal_key="basis:bbb")


@pytest.mark.parametrize("field,value", [
    ("align", "lick"), ("post_s", 1.0), ("pre_s", 2.0), ("fs", 31.23),
    ("max_rt", 3.5), ("baseline", "none"), ("source", "roi"),
])
def test_every_result_changing_arg_separates_the_key(field, value):
    """`max_rt` is not hypothetical: `decode.max_rt_s` went 2.0 -> 3.5 s on 2026-08-21 and
    invalidated every number computed before it, because it redefines 'engaged'."""
    assert kind(args()) != kind(args(**{field: value}))


@pytest.mark.parametrize("kw", [{"nolick_ref": "would_be_lick"}, {"with_precue_licks": True},
                                {"with_indices": True}])
def test_the_shape_and_placement_kwargs_separate_the_key(kw):
    """`with_precue_licks` and `with_indices` change the RETURN SHAPE; `nolick_ref` moves where a
    no-lick trial's window starts. Any of them missing serves a caller something it did not ask for."""
    assert kind(args()) != kind(args(), **kw)


def test_resolved_bins_separate_the_key():
    """`_bins_for` falls through to `defaults.yaml decode.bins` per alignment, and `session_signature`
    stats DATA files only -- a defaults.yaml edit changes the result and moves no mtime it looks at.
    So the RESOLVED value has to be in the kind, not the raw `args.bins`."""
    assert kind(args(bins=4)) != kind(args(bins=8))


def test_identical_inputs_give_an_identical_key():
    """A key that varied run to run would be a cache that never hits while looking like it works."""
    assert kind(args()) == kind(args())


def test_the_kind_is_filesystem_safe_and_bounded():
    """The kind lands in a FILENAME. A raw spec would push a Windows path over its limit and make the
    cache directory unreadable; the readable prefix is for a human scanning it, the digest is what
    makes it correct."""
    k = kind(args(), signal_key="basis:76d884873920")
    assert k.startswith("tf-cue-") and len(k) < 40
    assert not (set(k) & _UNSAFE_IN_A_FILENAME)


# ---------------------------------------------------------------------------------------------
# What it must REFUSE to key. Both fail-safe: compute rather than guess.
# ---------------------------------------------------------------------------------------------

def _spy(monkeypatch):
    calls = {"compute": 0, "cached": 0}

    def fake_tf(s, a, **kw):
        calls["compute"] += 1
        return ("features",)

    def fake_cached(session, kind_, compute, params=None, verbose=True):
        calls["cached"] += 1
        return compute()

    monkeypatch.setattr(D, "_trial_features", fake_tf)
    monkeypatch.setattr(D.session_cache, "cached", fake_cached)
    return calls


def test_an_injected_signal_without_a_key_is_not_cached(monkeypatch):
    """The array itself cannot go in a key -- it is the expensive thing we are avoiding building, and
    hashing ~100 MB per call would cost more than the rebuild. A caller that injects a signal must say
    where it came from; one that does not gets a correct uncached answer, not a fast wrong one."""
    calls = _spy(monkeypatch)
    D.trial_features_cached({"label": "PS99_0101", "mc": "x"}, args(), signal=[[1.0]])
    assert calls["cached"] == 0 and calls["compute"] == 1


def test_a_deferred_signal_without_a_key_is_not_cached_either(monkeypatch):
    """`signal_fn` is the same injection deferred, so it carries the same requirement. Checking only
    `signal` would leave the lazy path -- the one every joint figure now uses -- unguarded."""
    calls = _spy(monkeypatch)
    D.trial_features_cached({"label": "PS99_0101", "mc": "x"}, args(),
                            signal_fn=lambda: ([[1.0]], [0]))
    assert calls["cached"] == 0 and calls["compute"] == 1


def test_a_non_locanmf_source_is_not_cached(monkeypatch):
    """`_build_signal` then reads `U_atlas.npy` and the SVTcorr, and `session_signature` stats
    neither, so a re-preprocess would not invalidate the entry. Those sources are diagnostics, so
    leaving them uncached is honest where widening a shared signature would not be."""
    calls = _spy(monkeypatch)
    D.trial_features_cached({"label": "PS99_0101", "mc": "x"}, args(source="roi"))
    assert calls["cached"] == 0 and calls["compute"] == 1


def test_the_normal_path_IS_cached(monkeypatch):
    """The refusals must not be so broad that nothing is cached -- which would pass every test above
    while saving nothing."""
    calls = _spy(monkeypatch)
    D.trial_features_cached({"label": "PS99_0101", "mc": "x"}, args())
    assert calls["cached"] == 1
    calls2 = _spy(monkeypatch)
    D.trial_features_cached({"label": "PS99_0101", "mc": "x"}, args(),
                            signal_fn=lambda: ([[1.0]], [0]), signal_key="basis:abc")
    assert calls2["cached"] == 1


def test_the_deferred_signal_is_not_built_on_a_cache_hit(monkeypatch):
    """The whole saving. Projecting a session costs a U/SVT load and a ~100 MB result; if it happened
    before the cache lookup, a hit would still pay for it and the render would not get faster."""
    built = {"n": 0}

    def sig():
        built["n"] += 1
        return ([[1.0]], [0])

    monkeypatch.setattr(D.session_cache, "cached", lambda *a, **k: "HIT")
    out = D.trial_features_cached({"label": "PS99_0101", "mc": "x"}, args(),
                                  signal_fn=sig, signal_key="basis:abc")
    assert out == "HIT" and built["n"] == 0, "the signal was built despite a cache hit"
