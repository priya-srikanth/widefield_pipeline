"""Merge review B1 / N1 / N4 / N8 (11-03): stored LOSO models are keyed by SESSION LABEL, so a pool that lacks one pre
session (the LP pool without PS93_0813) still scores every pre session with the model that left THAT session out."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from wfield_local import spout_scaling as ss


def _pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))


def _pool(labels, seed=0):
    rng = np.random.default_rng(seed)
    GE = np.repeat(np.arange(len(labels)), 90)
    YE = np.tile(np.arange(6), 15 * len(labels))
    XE = rng.normal(size=(len(GE), 5)) + 0.8 * YE[:, None]
    return XE, YE, GE


def test_stored_loso_is_label_keyed_and_survives_a_missing_middle_session(monkeypatch):
    monkeypatch.setenv(ss.SPOUT_SCALING_ENV, "session_z")
    full = ["PS93_0812", "PS93_0813", "PS93_0814", "PS93_0815"]
    XE, YE, GE = _pool(full)
    box = {}

    def store_a(fit_all):
        box["pay"] = fit_all()
        return box["pay"]

    a = ss.FrozenFrame(XE, YE, GE, [0, 1, 2, 3], _pipe, store=store_a, labels=dict(enumerate(full)))
    assert sorted(box["pay"]["loso"]) == sorted(full)            # stored keys are LABELS
    # the LP pool: PS93_0813 (no video) is absent, so the later sessions' pooled ids shift down by one
    lp = [full[0], full[2], full[3]]
    keep = GE != 1
    XB, YB = XE[keep], YE[keep]
    GB = np.searchsorted([0, 2, 3], GE[keep])
    b = ss.FrozenFrame(XB, YB, GB, [0, 1, 2], _pipe, store=lambda fit_all: box["pay"], labels=dict(enumerate(lp)))
    for i, lab in enumerate(lp):
        assert b.loso[i] is box["pay"]["loso"][lab]                # id i -> the model that left `lab` out
    assert b.loso[1] is a.loso[2]                                  # PS93_0814: id 2 in A, id 1 in B
    with pytest.raises(ValueError, match="labels="):
        ss.FrozenFrame(XE, YE, GE, [0, 1], _pipe, store=store_a)    # a store without labels is refused


def test_every_store_hook_site_passes_labels():
    from wfield_local import nolick_decoder, position_coding_directions, poststroke_compare, precue_engagement_states
    assert "labels={k: lab for lab, k in _sid.items()}" in inspect.getsource(nolick_decoder.analyse_animal)
    assert "labels=dict(enumerate(kept))" in inspect.getsource(position_coding_directions._class_confusions)
    assert "labels=dict(enumerate(kept))" in inspect.getsource(precue_engagement_states)
    assert 'labels=dict(enumerate(d["kept"]))' in inspect.getsource(poststroke_compare.decode_matched)
    src = inspect.getsource(precue_engagement_states)
    assert "loso[kept[si]] = _disc().fit(" in src and 'dpay["loso"][kept[si]]' in src   # the -disc payload too
    assert not hasattr(poststroke_compare, "_fit_lickonly")


def test_nolick_train_data_carries_the_pooled_label_set():
    from wfield_local import nolick_decoder
    src = inspect.getsource(nolick_decoder._stored_nolick_reference)
    assert "tuple(sorted(set(np.asarray(SE).tolist())))" in src


def test_n2_camera_errors_are_not_swallowed():
    from wfield_local import grant_confusion, position_coding_directions
    assert "reraise_under_camera(ex, f\"5c {an}" in inspect.getsource(grant_confusion._collect_5c_cached)
    assert "reraise_under_camera(ex, f\"coding_dirs {animal}" in inspect.getsource(position_coding_directions.run_animal)


def test_n8_bundle_and_frame_keys_carry_gate_and_centre():
    from wfield_local import grant_kit
    assert "_tck.gate_mode(), _tck.run_selection_tag()" in inspect.getsource(grant_kit._pooled_bundle)
    assert "spout_scaling.centre_rows() if sc else 'z'" in inspect.getsource(grant_kit._frozen_frame)


def test_n4_drift_matched_draws_are_stored():
    jad = pytest.importorskip("scripts.june_august_drift")
    src = inspect.getsource(jad.decoder_rows)
    assert '"drift_matched"' in src and "frozen_models.load_or_fit(mm_spec" in src
    assert "got = mf(" in src and "_matched_frozen(_pipe, ff.XT[tr]" not in src


def test_section_g_store_errors_propagate_and_unkeyed_pools_refuse_in_a_tagged_run(monkeypatch):
    """Fifth review D-B1: `frozen(d)` no longer swallows a store error into an in-memory fit, and a pool without an
    identity is refused in a tagged run (allowed, as before, in the default run)."""
    from wfield_local import poststroke_compare as pc
    monkeypatch.setattr(pc, "frozen_decoder_models", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("store")))
    d = {"XE": np.zeros((4, 2)), "YE": np.zeros(4), "GE": np.zeros(4, int), "XU": np.zeros((0, 2)), "YU": np.zeros(0),
         "GU": np.zeros(0, int), "kept": ["PS92_0806"], "pre_i": {0},
         "align": "cue", "source": "roi", "_frozen_cache": {}}
    with pytest.raises(RuntimeError, match="store"):
        pc.frozen(d)
    for k in ("WIDEFIELD_TRIAL_GATE", "WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LICK_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    assert pc.frozen({"XE": None}) is None                       # default run: the old in-memory path
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    with pytest.raises(pc.SectionGRefused):
        pc.frozen({"XE": None})


def test_collect_5c_cache_is_keyed_by_the_run(monkeypatch):
    """Re-verify item 4: the 5c collector's cache key carries the run tag / selection and the bal_em center."""
    from wfield_local import grant_confusion as gc
    seen = []
    monkeypatch.setattr(gc, "_collect_5c_cached", lambda align, variant, mode, run: seen.append(run) or ({}, []))
    for k in ("WIDEFIELD_TRIAL_GATE", "WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LICK_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    gc._collect_5c("cue")
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    gc._collect_5c("cue")
    assert seen[0] != seen[1] and "sel-camera" in seen[1][0]
