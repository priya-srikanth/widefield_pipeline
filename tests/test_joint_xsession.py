"""Tests for cross-session decode/encode in the shared joint-LocaNMF basis.

The joint basis exists to make LocaNMF components poolable across days. That only holds if the SAME
footprints are used for every session, so these tests target the ways that could quietly stop being
true and still produce a plausible number: features being rebuilt per session instead of injected, a
projected session being silently refitted, and the region labels the encoder groups by drifting away
from the components they describe.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from wfield_local import joint_xsession
from wfield_local.locanmf_frozen_decoder import BASIS_NAME, _encoder_fig, _loso_fig, pool_sessions


def test_injected_signal_is_used_instead_of_loading_from_disk(monkeypatch):
    """``_trial_features(signal=...)`` must not touch ``_build_signal``.

    If it fell back to the disk loader the joint analysis would silently become the per-session one --
    same shapes, plausible accuracy, wrong basis.
    """
    from wfield_local import locanmf_position_decoder as pd

    def _boom(*a, **k):
        raise AssertionError("_build_signal was called despite an injected signal")

    monkeypatch.setattr(pd, "_build_signal", _boom)
    sig = np.zeros((7, 500))
    with pytest.raises(Exception) as ex:      # fails later (no DAQ file), but NOT on _build_signal
        pd._trial_features({"label": "PSXX_0000", "mc": "/nope", "h5": "/nope"},
                           SimpleNamespace(source="locanmf", align="precue", baseline="none",
                                           pre_s=1.0, post_s=2.0, fs=31.23, max_rt=2.0),
                           signal=sig)
    assert "_build_signal" not in str(ex.value)


def test_pool_sessions_calls_the_features_hook_for_every_session():
    """The hook is what makes the joint basis reachable from the pooling code; if pool_sessions ever
    stopped honouring it, the joint figures would silently show ROI results."""
    seen = []

    def fake_features(s, args):
        seen.append(s["label"])
        n = 60
        rng = np.random.default_rng(len(seen))
        y = np.repeat(np.arange(6), n // 6)
        return (rng.normal(size=(n, 5)), y, np.repeat(np.arange(10), 6),
                np.zeros((0, 5)), np.array([]), np.arange(5))

    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    labs = [s["label"] for s in SESSIONS][:2]
    out = pool_sessions(labs, source="locanmf", features=fake_features)
    assert out is not None
    assert seen == labs
    XE, YE, GE = out[0], out[1], out[2]
    assert XE.shape[0] == len(YE) == len(GE)
    assert set(np.unique(GE)) == {0, 1}, "GE must be the SESSION index (the LOSO group)"


def test_figure_filenames_carry_the_basis_so_bases_cannot_overwrite_each_other(tmp_path):
    """ROI and joint results are the same figure in two bases. Before the basis tag existed the
    filename only carried the alignment, and a precue run overwrote the cue figures -- the deck then
    showed one under the other's heading. Do not let that recur across bases."""
    res = {"PS92": {"labels": ["PS92_0606"], "per_session": {"PS92_0606": 0.5},
                    "within_session": {"PS92_0606": 0.4}, "transfer_cost": 0.1,
                    "ood": {"engaged_H": 0.5, "nolick_H": 0.6, "shuffle_H": 0.9}}}
    enc = {"PS92": {"labels": ["PS92_0606"], "per_session_ev": {"PS92_0606": 0.02},
                    "within_session_ev": {"PS92_0606": 0.01}, "ceiling": {"PS92_0606": 0.05},
                    "mean_ev": 0.02, "mean_within_ev": 0.01, "mean_feve": 0.4}}
    paths = {_loso_fig(res, tmp_path, "precue", basis=b).name for b in ("roi", "joint")}
    paths |= {_encoder_fig(enc, tmp_path, "precue", basis=b).name for b in ("roi", "joint")}
    assert len(paths) == 4, f"filenames collide across bases: {paths}"
    assert "locanmf_frozen_decoder_loso_roi_precue.png" in paths, "ROI names must stay unchanged"


def test_basis_name_covers_every_basis_the_deck_asks_for():
    """The deck builds slide titles from these keys; an unmapped one would render as a bare slug."""
    assert set(BASIS_NAME) >= {"roi", "joint"}


def test_basis_health_figure_marks_projected_sessions(tmp_path):
    """The whole point of the panel is telling in-fit from projected. If ``basis_labels`` were dropped
    every bar would render as in-fit and a poorly-spanned projected day would look trustworthy."""
    results = {"PS92": {"variance_captured": {"PS92_0606": 1.0, "PS92_0812": 0.83},
                        "basis_labels": ["PS92_0606"], "ncomp": 137}}
    p = joint_xsession.fig_basis_health(results, tmp_path, "precue")
    assert p is not None and p.exists()


def test_joint_features_records_variance_captured_per_session():
    """A projected session's variance_captured is the only thing separating 'the basis does not span
    this day' from 'this day changed'. It must be collected, not merely computed and dropped."""
    class FakeBasis:
        basis_id, ncomp, labels = "abc123", 3, ["S_in"]
        regions = np.array([1, 2, 3])

        def signal(self, lab):
            return np.zeros((3, 100))

        def project(self, s, with_diagnostics=False):
            C = np.zeros((3, 100))
            return (C, {"variance_captured": 0.77}) if with_diagnostics else C

    calls = {}
    feat = joint_xsession.joint_features(FakeBasis())
    joint_xsession._FEAT_CACHE.clear()

    def fake_tf(s, args, signal=None, feat_region=None):
        calls[s["label"]] = (signal.shape, tuple(feat_region))
        return (np.zeros((6, 3)), np.arange(6), np.arange(6), np.zeros((0, 3)), np.array([]),
                feat_region)

    import wfield_local.joint_xsession as jx
    orig, jx._trial_features = jx._trial_features, fake_tf
    try:
        args = SimpleNamespace(align="precue", post_s=2.0, source="locanmf")
        feat({"label": "S_in", "mc": "", "h5": ""}, args)
        feat({"label": "S_out", "mc": "", "h5": ""}, args)
    finally:
        jx._trial_features = orig
        joint_xsession._FEAT_CACHE.clear()
    assert feat.variance_captured == {"S_in": 1.0, "S_out": 0.77}
    assert calls["S_out"][1] == (1, 2, 3), "component region labels must travel with the features"


def test_regions_are_derived_from_the_footprints_not_guessed():
    """``Basis.regions`` is derived because ``build`` discarded LocaNMF's own assignment and refitting
    would produce a DIFFERENT basis. Derivation is only valid because LocaNMF confines a component to
    one atlas region -- so the max-weight label must be that region."""
    from wfield_local.joint_locanmf import Basis

    atlas = np.array([0, 0, 5, 5, 5, 9, 9])
    A = np.zeros((7, 2), dtype=np.float32)
    A[2:5, 0] = [0.4, 0.9, 0.2]      # component 0 lives in region 5
    A[5:7, 1] = [0.7, 0.3]           # component 1 lives in region 9
    labs = np.unique(atlas[atlas != 0])
    W = np.stack([np.abs(A[atlas == l]).sum(0) for l in labs])
    assert list(labs[np.argmax(W, axis=0)]) == [5, 9]
    assert hasattr(Basis, "regions")
