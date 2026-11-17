"""Allen-ROI features are disk-cached, keyed on the stats of every file the ROI signal is built from (11-07).

Uncached, section G rebuilt every session at every alignment in every unit off MICROSCOPE (~19 s per session-alignment,
~15 min per unit). The original refusal was that `session_signature` cannot see `U_atlas.npy` / the SVTcorr; the key now
carries their (path, mtime, size), so a re-made input is a different kind. Other non-LocaNMF sources stay uncached.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from wfield_local import locanmf_position_decoder as lpd


@pytest.fixture
def session(tmp_path, monkeypatch):
    ad = tmp_path / "wfield_local_results" / "allen_aligned_affine8v1"
    ad.mkdir(parents=True)
    for f in ("U_atlas.npy", "allen_area_atlas_native_grid.npy", "allen_brain_mask_native_grid.npy"):
        (ad / f).write_bytes(b"x")
    svt = tmp_path / "SVTcorr.npy"
    svt.write_bytes(b"y")
    monkeypatch.setattr(lpd.config, "svtcorr_path", lambda mc: svt)
    return {"label": "PSX_0101", "mc": str(tmp_path)}, ad, svt


def _calls(monkeypatch):
    seen = {"cached": [], "computed": 0}

    def fake_tf(s, args, **k):
        seen["computed"] += 1
        return ("features", args.source)

    def fake_cached(s, kind, compute, **k):
        seen["cached"].append(kind)
        return compute()

    monkeypatch.setattr(lpd, "_trial_features", fake_tf)
    monkeypatch.setattr(lpd.session_cache, "cached", fake_cached)
    monkeypatch.setattr(lpd.lp_licks, "apply_enl_window", lambda a: a)
    return seen


def _args(source):
    return SimpleNamespace(align="cue", post_s=2.0, pre_s=1.0, fs=31.23, max_rt=3.5, baseline=None, source=source)


def test_roi_key_lists_every_roi_input(session):
    s, ad, svt = session
    got = lpd.roi_signal_inputs(s)
    assert [os.path.basename(f) for f in got[:3]] == ["U_atlas.npy", "allen_area_atlas_native_grid.npy",
                                                      "allen_brain_mask_native_grid.npy"]
    assert got[3] == str(svt)


@pytest.mark.parametrize("which", ["U_atlas.npy", "allen_area_atlas_native_grid.npy",
                                   "allen_brain_mask_native_grid.npy", "SVT"])
def test_roi_key_moves_when_any_input_is_remade(session, which):
    s, ad, svt = session
    k0 = lpd.roi_signal_key(s)
    f = svt if which == "SVT" else ad / which
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert lpd.roi_signal_key(s) != k0
    assert lpd.roi_signal_key(s) == lpd.roi_signal_key(s)          # deterministic


def test_roi_source_goes_through_the_cache_with_the_input_key(session, monkeypatch):
    s, ad, svt = session
    seen = _calls(monkeypatch)
    out = lpd.trial_features_cached(s, _args("roi"))
    assert out == ("features", "roi") and seen["computed"] == 1
    want = lpd.feature_cache_kind(_args("roi"), signal_key=lpd.roi_signal_key(s), nolick_ref="cue",
                                  with_precue_licks=False, with_indices=False)
    assert seen["cached"] == [want]


def test_roi_kind_differs_from_locanmf_and_from_a_remade_input(session, monkeypatch):
    s, ad, svt = session
    seen = _calls(monkeypatch)
    lpd.trial_features_cached(s, _args("roi"))
    lpd.trial_features_cached(s, _args("locanmf"))
    st = os.stat(svt)
    os.utime(svt, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    lpd.trial_features_cached(s, _args("roi"))
    roi0, loc, roi1 = seen["cached"]
    assert len({roi0, loc, roi1}) == 3


def test_other_non_locanmf_sources_stay_uncached(session, monkeypatch):
    s, ad, svt = session
    seen = _calls(monkeypatch)
    assert lpd.trial_features_cached(s, _args("svt")) == ("features", "svt")
    assert seen["cached"] == [] and seen["computed"] == 1
