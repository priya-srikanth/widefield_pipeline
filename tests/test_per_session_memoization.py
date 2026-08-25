"""Per-session quantities are memoized, and memoizing them must not move a single number.

Measured 2026-08-25 over one night: `spatial_reorganisation` 36 min, `evoked_amplitude` 18 min,
`fixed_scale_maps` 18 min -- ~72 min recomputing values that CANNOT have changed. Each is a property
of one session's own recording: crossnobis is computed inside a single session (DECISIONS
2026-08-19/20), and PS92_0606's evoked amplitude cannot move because PS93 was recorded on 8/25.

The risk of caching is serving a result for the wrong question, so these tests pin the two things
that would make that happen:

  WRONG PARAMS   a cue-aligned result served for a precue request, or an engaged-only matrix served
                 where post_all_trials=True was asked for. The `kind` string must separate them.
  WRONG INPUTS   a result kept after the session's own data changed. `session_signature` covers the
                 LocaNMF C, the h5 and the behavior trials, so that is inherited -- pinned here so a
                 refactor cannot quietly drop it.

What is NOT cached is equally load-bearing: the cross-session layers built on top (the pre-stroke
band, the common colour scale, the mirror test) still see every new night.
"""
import inspect

import pytest

from wfield_local import evoked_amplitude, fixed_scale_maps, session_cache, spatial_reorganisation

WRAPPED = [
    (evoked_amplitude, "session_amplitudes", "_session_amplitudes"),
    (fixed_scale_maps, "_position_maps", "_position_maps_uncached"),
    (spatial_reorganisation, "_area_matrix", "_area_matrix_uncached"),
]


@pytest.mark.parametrize("mod,pub,raw", WRAPPED)
def test_the_uncached_implementation_is_still_present(mod, pub, raw):
    """The real computation must survive the wrapper -- caching is a shortcut, not a replacement."""
    assert callable(getattr(mod, raw)), f"{mod.__name__}.{raw} is gone"
    assert callable(getattr(mod, pub))


@pytest.mark.parametrize("mod,pub,raw", WRAPPED)
def test_the_wrapper_delegates_and_returns_verbatim(mod, pub, raw, monkeypatch):
    """No transformation on the way through: cached must equal uncached, exactly."""
    sentinel = {"value": object()}
    monkeypatch.setattr(mod, raw, lambda *a, **k: sentinel)
    monkeypatch.setattr(session_cache, "cached",
                        lambda session, kind, compute, params=None, verbose=True: compute())
    got = getattr(mod, pub)({"label": "PS92_0606", "mc": "x"})
    assert got is sentinel


@pytest.mark.parametrize("mod,pub,raw", WRAPPED)
def test_different_params_get_different_cache_keys(mod, pub, raw, monkeypatch):
    """A cue result must never be served for a precue question."""
    seen = []
    monkeypatch.setattr(mod, raw, lambda *a, **k: None)
    monkeypatch.setattr(session_cache, "cached",
                        lambda session, kind, compute, params=None, verbose=True: seen.append(kind))
    s = {"label": "PS92_0606", "mc": "x"}
    getattr(mod, pub)(s, align="cue")
    getattr(mod, pub)(s, align="precue")
    assert len(set(seen)) == 2, f"{mod.__name__}.{pub} reuses one key across alignments: {seen}"


def test_area_matrix_key_separates_the_trial_set():
    """post_all_trials is the difference between "the phenotype" and "a minority subset" for a
    post-stroke session (PS94 8/18 is 40% engaged), so it cannot share a key."""
    src = inspect.getsource(spatial_reorganisation._area_matrix)
    assert "post_all_trials" in src.split("session_cache.cached")[1].split("lambda")[0], (
        "the cache kind must include post_all_trials")


def test_signature_still_tracks_the_session_inputs():
    """Inherited from session_cache, pinned so a refactor cannot silently drop it."""
    src = inspect.getsource(session_cache.session_signature)
    for token in ("locanmf_C.npy", 'session.get("h5"', "CACHE_VERSION"):
        assert token in src, f"session_signature no longer covers {token}"


def test_cache_can_be_disabled_for_an_equivalence_check():
    """WIDEFIELD_NO_CACHE is how a cached-vs-uncached comparison is run on real data."""
    assert "WIDEFIELD_NO_CACHE" in inspect.getsource(session_cache._disabled)
