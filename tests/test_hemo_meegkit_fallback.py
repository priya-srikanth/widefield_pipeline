"""meegkit's robust detrend flags outliers against a GLOBAL std across all channels, so the dominant
SVD component (component 0, ~10x the amplitude of the rest) can have every sample flagged; its weight
column collapses to all-zero and meegkit.regress then references an unset local `V` -> UnboundLocalError.
It killed the PS92 20260922 nightly. remove_drift now catches that and detrends each component on its
own (same method, per-component outlier std), which cannot cross-contaminate. The batch path is left
untouched, so every session that already succeeds is bit-for-bit unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfield_local.hemo_variants as H

# meegkit lives in the IMAGING box's `wfield` env, not this analysis box's `locanmf` env (ground
# rule 7: per-machine envs differ by design -- this box READS the meegkit-computed SVTcorr, it never
# runs the detrend, so meegkit is deliberately not installed here). Skip the module where it is
# absent rather than fail collection with ModuleNotFoundError -- which was blocking every push from
# the analysis box, including LocaNMF session registrations.
pytest.importorskip("meegkit")


def _patch_meegkit(monkeypatch, fail_batch):
    """Stand in for meegkit.detrend: raise meegkit's bug on the batch (2-D w) call, succeed per
    component (single-column w). Returns the input unchanged as the 'trend-removed' output so the
    test asserts control flow, not numerics."""
    import meegkit.detrend as md

    calls = {"batch": 0, "per_component": 0}

    def fake_detrend(x, order, w=None, **kw):
        if w is not None and w.ndim == 2 and w.shape[1] > 1:
            calls["batch"] += 1
            if fail_batch:
                raise UnboundLocalError("cannot access local variable 'V'")
        else:
            calls["per_component"] += 1
        return np.asarray(x, dtype=np.float64), np.asarray(w), None

    monkeypatch.setattr(md, "detrend", fake_detrend)
    return calls


def test_batch_failure_falls_back_to_per_component(monkeypatch):
    calls = _patch_meegkit(monkeypatch, fail_batch=True)
    H._MEEGKIT_FELL_BACK = False
    X = np.random.RandomState(0).randn(4, 200)          # (K=4 components, T=200)
    m = np.ones(200, bool)

    out = H.remove_drift(X, "meegkit_hpfit", mask=m)

    assert out.shape == X.shape
    assert np.isfinite(out).all()
    assert calls["batch"] == 1 and calls["per_component"] == X.shape[0]  # one batch try, then per-comp
    assert H._MEEGKIT_FELL_BACK is True


def test_batch_success_takes_the_batch_path_unchanged(monkeypatch):
    calls = _patch_meegkit(monkeypatch, fail_batch=False)
    H._MEEGKIT_FELL_BACK = False
    X = np.random.RandomState(1).randn(4, 200)
    m = np.ones(200, bool)

    out = H.remove_drift(X, "meegkit_hpfit", mask=m)

    assert out.shape == X.shape
    assert calls["batch"] == 1 and calls["per_component"] == 0            # no fallback
    assert H._MEEGKIT_FELL_BACK is False
