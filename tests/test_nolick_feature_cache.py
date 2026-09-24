"""The `session_features` disk cache must not serve features built under superseded rules.

`session_cache` keys on a signature of the session's DATA files. Two things that change these
features live in `defaults.yaml` instead and move no mtime it stats -- and both of them changed on
2026-09-24, which is precisely when this cache was introduced. A key that omitted either would let a
warm entry silently undo the strict lick-free gate, or carry a `lead_lick` split computed at an
interval nobody asked for.

`locanmf_position_decoder.feature_cache_kind` records the same failure: `decode.max_rt_s` going
2.0 -> 3.5 s invalidated every previously computed number while moving no file it looked at.
"""
from types import SimpleNamespace

import pytest

from wfield_local import nolick_decoder as nd


def _args(**kw):
    base = dict(align="precue", post_s=2.0, pre_s=1.0, fs=31.23, max_rt=2.0,
                baseline="none", source="locanmf")
    base.update(kw)
    return SimpleNamespace(**base)


def _kind(args=None, *, signal_key="own", lead_s=1.0):
    return nd.session_features_cache_kind(args or _args(), signal_key=signal_key, lead_s=lead_s)


def test_the_same_inputs_give_the_same_kind():
    assert _kind() == _kind()


def test_lead_s_is_in_the_key():
    """It sets `lead_lick`, so an entry written at 1.0 s holds a different split from one at 0.5 s."""
    assert _kind(lead_s=1.0) != _kind(lead_s=0.5)


def test_the_lickfree_setting_is_in_the_key(monkeypatch):
    """The gate lives in defaults.yaml, so nothing `session_cache` stats changes when it flips.

    Without this, an entry written before 2026-09-24 -- fixed window, no gate -- would be served
    afterwards and silently undo the strict gate.
    """
    got = []
    real = nd.config.defaults

    def fake():
        d = dict(real())
        dec = dict(d["decode"])
        dec["precue_lickfree"] = got and got[0]
        d["decode"] = dec
        return d

    got.append(True)
    monkeypatch.setattr(nd.config, "defaults", fake)
    on = _kind()
    got[0] = False
    off = _kind()
    assert on != off


def test_signal_provenance_is_in_the_key():
    """Own LocaNMF fit vs a projection onto a shared joint basis are different features."""
    assert _kind(signal_key="own") != _kind(signal_key="joint:abc123")
    assert _kind(signal_key="joint:abc123") != _kind(signal_key="joint:def456")


@pytest.mark.parametrize("field,value", [
    ("align", "cue"), ("post_s", 1.5), ("max_rt", 3.5), ("source", "roi"),
    ("baseline", "precue"), ("fs", 30.0), ("pre_s", 0.5),
])
def test_every_arg_that_changes_the_result_is_in_the_key(field, value):
    assert _kind(_args(**{field: value})) != _kind()


def test_response_window_is_in_the_key():
    """`categorize` prefers `args.response_window_s` over the session's own value when present."""
    assert _kind(_args(response_window_s=3.5)) != _kind(_args(response_window_s=2.5))


def test_the_kind_is_short_and_readable():
    """It lands in a FILENAME; a full spec would push a Windows path over its limit."""
    k = _kind()
    assert k.startswith("sf-precue-") and len(k) < 32


def test_an_injected_signal_without_provenance_is_not_cached(monkeypatch):
    """Fail-safe: compute correctly rather than key on something that cannot describe the signal."""
    calls = []
    monkeypatch.setattr(nd, "session_features", lambda *a, **k: calls.append(1) or ("F", "reg"))
    monkeypatch.setattr(nd.session_cache, "cached",
                        lambda *a, **k: pytest.fail("must not consult the cache"))
    out = nd.session_features_cached(object(), _args(), signal=[[0.0]], lead_s=1.0)
    assert out == ("F", "reg") and len(calls) == 1


def test_a_non_locanmf_source_without_injection_is_not_cached(monkeypatch):
    """`_build_signal` then reads files the session signature does not stat."""
    monkeypatch.setattr(nd, "session_features", lambda *a, **k: ("F", "reg"))
    monkeypatch.setattr(nd.session_cache, "cached",
                        lambda *a, **k: pytest.fail("must not consult the cache"))
    assert nd.session_features_cached(object(), _args(source="roi"), lead_s=1.0) == ("F", "reg")


def test_a_deferred_signal_fn_is_not_called_on_a_cache_hit(monkeypatch):
    """The projection is the expensive half; deferring it is the whole saving on the joint path."""
    monkeypatch.setattr(nd.session_cache, "cached", lambda s, kind, compute, **k: "WARM")
    boom = lambda: pytest.fail("signal_fn must not run when the cache hits")   # noqa: E731
    assert nd.session_features_cached(object(), _args(), signal_fn=boom,
                                      signal_key="joint:abc", lead_s=1.0) == "WARM"
