"""The encoder reads trial features through the session cache, and gets the SAME features.

Until 2026-11-06 `locanmf_position_encoder` called plain `_trial_features` at five sites, so every
date's encoder process rebuilt each of ~122 pooled sessions' features about four times (~17 s
each), while the decoders beside it read the same features from `trial_features_cached`. The switch
is only worth having if it changes no number, so that is what is pinned here: cached (cold, and
warm from the pickle) equals uncached element for element, exactly -- not to a tolerance.

No real session is loaded: `_trial_features` is replaced by a deterministic fake, and the cache
lives in a tmp dir with a fixed signature.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wfield_local import locanmf_position_decoder as dec
from wfield_local import locanmf_position_encoder as enc
from wfield_local import session_cache

SESSION = {"label": "PS99_0101", "mc": "X:/nowhere", "h5": "", "behavior_trials": ""}


def _fake_features(s, args, signal=None, feat_region=None, with_precue_licks=False,
                   with_indices=False, with_rt=False, nolick_ref="cue"):
    """Same 6-tuple shape as `_trial_features`, deterministic in (label, args)."""
    seed = sum(map(ord, s["label"])) + int(10 * args.post_s)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 7))
    y = np.tile(np.array([0, 1, 2, 3, 4, 5]), 10)
    g = np.repeat(np.arange(10), 6)
    Xn = rng.normal(size=(12, 7)).astype(np.float32)
    yn = np.arange(12) % 6
    reg = np.arange(7, dtype=np.int64) + 100
    return X, y, g, Xn, yn, reg


@pytest.fixture
def fake(monkeypatch, tmp_path):
    calls = []

    def _counting(*a, **k):
        calls.append(1)
        return _fake_features(*a, **k)

    monkeypatch.delenv("WIDEFIELD_NO_CACHE", raising=False)
    monkeypatch.setattr(dec, "_trial_features", _counting)
    monkeypatch.setattr(session_cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(session_cache, "session_signature", lambda s, params=None: "fixedsig")
    monkeypatch.setattr(enc, "_FEAT", {})
    return calls


def _assert_identical(a, b):
    assert type(a) is type(b) and len(a) == len(b)
    for u, v in zip(a, b):
        u, v = np.asarray(u), np.asarray(v)
        assert u.dtype == v.dtype and u.shape == v.shape
        assert np.array_equal(u, v)


def test_cached_equals_uncached_cold_and_warm(fake):
    args = enc._args(2.0)
    ref = _fake_features(SESSION, args)
    cold = dec.trial_features_cached(SESSION, args)
    warm = dec.trial_features_cached(SESSION, args)
    _assert_identical(cold, ref)
    _assert_identical(warm, ref)                       # read back from the pickle
    assert len(fake) == 1, "the warm call must be a cache read, not a rebuild"
    assert list((session_cache.CACHE_DIR).glob("PS99_0101__tf-lick-*.pkl")), "kind is tf-<align>-<digest>"


def test_encoder_features_build_once_per_process(fake):
    ref = _fake_features(SESSION, enc._args(2.0))
    outs = [enc._features(SESSION, enc._args(2.0)) for _ in range(4)]
    for o in outs:
        _assert_identical(o, ref)
    assert len(fake) == 1


def test_different_args_are_different_entries(fake):
    a = enc._features(SESSION, enc._args(2.0))
    b = enc._features(SESSION, enc._args(1.5))
    assert not np.array_equal(a[0], b[0])
    _assert_identical(b, _fake_features(SESSION, enc._args(1.5)))
    assert len(fake) == 2


def test_refused_source_is_computed_not_cached(fake):
    """The wrapper's refusal (a diagnostic non-locanmf source, no injected signal; ``roi`` is cached since 11-07)
    still holds through `_features`'s disk tier: computed every time, nothing written."""
    args = SimpleNamespace(**{**vars(enc._args(2.0)), "source": "svt"})
    r1 = dec.trial_features_cached(SESSION, args)
    r2 = dec.trial_features_cached(SESSION, args)
    _assert_identical(r1, r2)
    assert len(fake) == 2
    assert not session_cache.CACHE_DIR.exists() or not list(session_cache.CACHE_DIR.glob("*.pkl"))


def test_no_plain_trial_features_call_left_in_the_encoder():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(enc))
    plain = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_trial_features"]
    assert plain == [], f"plain _trial_features calls at lines {plain}; use _features"


# ------------------------------------------------------------- pooled figures drawn once

def test_skip_pooled_stops_before_the_pooled_figures(monkeypatch, tmp_path):
    drawn = []
    monkeypatch.setattr(enc, "SESSIONS", [])
    for name in ("fig_ev_by_position", "fig_ev_ceiling_by_position", "fig_quiet_drift"):
        monkeypatch.setattr(enc, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub")))
    monkeypatch.setattr(enc, "fig_region_feve_pooled", lambda *a, **k: drawn.append("pooled"))
    monkeypatch.setattr(enc, "fig_region_feve_sessions", lambda *a, **k: drawn.append("sessions"))
    monkeypatch.setattr(enc, "fig_ev_matrix", lambda *a, **k: drawn.append("matrix"))
    base = ["enc", "--output", str(tmp_path), "--date", "0101"]
    monkeypatch.setattr(sys, "argv", base + ["--skip-pooled"])
    assert enc.main() == 0 and drawn == []
    monkeypatch.setattr(sys, "argv", base)
    assert enc.main() == 0 and drawn == ["pooled", "sessions", "matrix"]


def test_nightly_draws_pooled_on_the_last_date_only(monkeypatch):
    from wfield_local import nightly_figs as nf

    ran = []
    monkeypatch.setattr(nf, "_cli_many", lambda cmds: ran.append(list(cmds)))
    monkeypatch.setattr(nf, "_per_day_inproc", lambda *a, **k: None)
    nf._per_day_figs(["0101", "0102", "0103"], "OUT", "0101,0102,0103", only=None)
    encs = ran[-1]
    assert [c[2] for c in encs] == ["0101", "0102", "0103"]
    assert ["--skip-pooled" in c for c in encs] == [True, True, False]
    # apart from the flag, every encoder command is what it was
    for c in encs:
        assert c[:2] == ("wfield_local.locanmf_position_encoder", "--date")
        assert "--pool-dates" in c and c[c.index("--pool-dates") + 1] == "0101,0102,0103"
