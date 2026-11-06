"""The frozen ROI step builds ONE pool per (animal, alignment) and both consumers read it unchanged;
the two map scripts read the memoised map through the identical call.

WHY (audit items 2 and 6, 2026-11-06). `pooled_frozen_loso` and `pooled_frozen_encoder` each called
`pool_sessions(source="roi")` with the same arguments, and ROI features are not disk-cached, so the
nightly built every pool twice (~19 s a session x 122 x 3 alignments, twice). Sharing it is only safe
if the shared pool is the pool each consumer would have built itself -- under the camera gate the
decoder pools with `working_trials.roi_features_with_indices` and the encoder with the default
builder, so that is pinned here with EXACT equality, on synthetic sessions (no data from M:).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from wfield_local import locanmf_frozen_decoder as fd
from wfield_local import locanmf_position_decoder as lpd

LABELS = ["PSX_0801", "PSX_0802", "PSX_0803"]
REGS = np.array(["MOp", "SSp", "VISp", "MOs"] * 2)          # two sub-bins, tiled like the real ROI vector


def _fake_sessions():
    return [{"label": lab, "mc": f"C:/nowhere/{lab}"} for lab in LABELS]


def _fake_trial_features(s, args, *, with_indices=False, nolick_ref="cue", **_kw):
    """Deterministic per-session (X, y, g, Xnl, ynl, reg) [+ engaged / no-lick cue indices]."""
    seed = LABELS.index(s["label"]) + 7 * len(str(args.align))
    rng = np.random.RandomState(seed)
    n, nn = 30 + seed, 5 + seed % 3
    reg = REGS if s["label"] != "PSX_0802" else REGS[::-1].copy()   # a different column order, so alignment runs
    X = rng.randn(n, len(reg)) * (1 + seed) + seed
    y = np.repeat(np.arange(6), n // 6 + 1)[:n]
    g = np.arange(n) // 6
    Xnl = rng.randn(nn, len(reg))
    ynl = rng.randint(0, 6, nn)
    base = (X, y, g, Xnl, ynl, reg)
    if with_indices:
        return base + (np.arange(n) * 2, np.arange(nn) * 2 + 1)
    return base


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(fd, "SESSIONS", _fake_sessions())
    monkeypatch.setattr(fd, "trial_features_cached", _fake_trial_features)
    monkeypatch.setattr(lpd, "trial_features_cached", _fake_trial_features)   # what roi_features_with_indices imports


@pytest.fixture
def camera_gate(monkeypatch):
    from wfield_local import trial_classes, working_trials
    monkeypatch.setattr(trial_classes, "camera_gate_on", lambda: True)
    monkeypatch.setattr(trial_classes, "working_selection", lambda: "in_working")
    monkeypatch.setattr(working_trials, "is_camera", lambda sel: True)


def _assert_pools_identical(a, b):
    assert a is not None and b is not None
    assert len(a) == len(b) == 9
    for i in (0, 1, 2, 4, 5, 8):                               # XE YE GE XU YU GU
        np.testing.assert_array_equal(a[i], b[i])
        assert np.asarray(a[i]).dtype == np.asarray(b[i]).dtype
    assert len(a[3]) == len(b[3])                              # BE: per-session block ids
    for x, y in zip(a[3], b[3]):
        np.testing.assert_array_equal(x, y)
    assert list(a[6]) == list(b[6])                            # kept labels
    assert list(a[7]) == list(b[7])                            # common regions


@pytest.mark.parametrize("align", ["cue", "precue", "lick"])
def test_shared_pool_equals_each_consumers_own_pool_under_the_camera_gate(synthetic, camera_gate, align):
    # the ENCODER's own pool: default builder (features=None)
    enc_own = fd.pool_sessions(LABELS, source="roi", align=align, post_s=2.0)
    # the DECODER's own pool: its feature choice, built the way pooled_frozen_loso builds it
    dec_feat = fd.loso_features(None, "roi", align)
    dec_own = fd.pool_sessions(LABELS, source="roi", align=align, post_s=2.0, zscore=True, features=dec_feat)
    shared, feats = fd.frozen_pool(LABELS, source="roi", align=align, post_s=2.0)
    _assert_pools_identical(shared, enc_own)
    _assert_pools_identical(shared, dec_own)
    if align == "lick":
        assert feats is None                                   # the lick window never takes the camera builder
    else:
        # the decoder's working-trial mapping needs the indices, recorded on the builder that made the pool
        assert set(feats.indices) == set(LABELS)
        assert getattr(feats, "basis_id", "missing") is None   # so the encoder spec's basis_id is unchanged


def test_shared_pool_equals_the_default_pool_with_the_gate_off(synthetic, monkeypatch):
    from wfield_local import trial_classes
    monkeypatch.setattr(trial_classes, "camera_gate_on", lambda: False)
    shared, feats = fd.frozen_pool(LABELS, source="roi", align="cue")
    assert feats is None
    _assert_pools_identical(shared, fd.pool_sessions(LABELS, source="roi", align="cue"))


def test_frozen_pair_builds_once_and_hands_both_consumers_the_same_pool(monkeypatch):
    calls, got = [], {}
    sentinel_feat = object()
    pooled = tuple(np.arange(3) for _ in range(9))

    def fake_pool(labels, **kw):
        calls.append((list(labels), kw))
        return pooled

    monkeypatch.setattr(fd, "pool_sessions", fake_pool)
    monkeypatch.setattr(fd, "loso_features", lambda f=None, s="roi", a="cue": sentinel_feat)

    def fake_enc(labels, **kw):
        got["enc"] = kw
        assert "dec" not in got                                # the encoder scores first
        return {"mean_ev": 0.1}

    def fake_dec(labels, **kw):
        got["dec"] = kw
        return {"loso_accuracy": 0.5}

    monkeypatch.setattr(fd, "pooled_frozen_encoder", fake_enc)
    monkeypatch.setattr(fd, "pooled_frozen_loso", fake_dec)
    dec, enc = fd.frozen_pair(LABELS, align="precue")
    assert (dec, enc) == ({"loso_accuracy": 0.5}, {"mean_ev": 0.1})
    # ONE build, with exactly the arguments both consumers used to build with themselves
    assert calls == [(LABELS, dict(source="roi", align="precue", post_s=2.0, zscore=True, features=sentinel_feat))]
    for k in ("enc", "dec"):
        assert got[k]["pooled"] is pooled
        assert got[k]["features"] is sentinel_feat
        assert (got[k]["source"], got[k]["align"], got[k]["post_s"], got[k]["verbose"]) == ("roi", "precue", 2.0, False)


def test_consumers_without_a_pool_still_build_their_own_unchanged(monkeypatch):
    from wfield_local import trial_classes
    monkeypatch.setattr(trial_classes, "camera_gate_on", lambda: False)
    seen = []
    monkeypatch.setattr(fd, "pool_sessions", lambda labels, **kw: seen.append(kw))
    assert fd.pooled_frozen_encoder(LABELS, align="cue") is None
    assert fd.pooled_frozen_loso(LABELS, align="cue") is None
    assert seen == [dict(source="roi", align="cue", post_s=2.0, features=None),
                    dict(source="roi", align="cue", post_s=2.0, zscore=True, features=None)]


def test_loso_refuses_a_shared_pool_without_its_builder_under_the_gate(synthetic, camera_gate):
    shared, _feats = fd.frozen_pool(LABELS, source="roi", align="cue")
    with pytest.raises(ValueError, match="features"):
        fd.pooled_frozen_loso(LABELS, align="cue", pooled=shared)


def test_frozen_pairs_serial_is_in_order_and_in_process(monkeypatch):
    order = []
    monkeypatch.setattr(fd, "frozen_pair", lambda labels, align: order.append((labels[0], align)) or (align, labels[0]))
    units = [("PS93", "cue", ["PS93_0801", "PS93_0802"]), ("PS92", "cue", ["PS92_0801", "PS92_0802"])]
    out = fd.frozen_pairs(units, jobs=1)
    assert order == [("PS93_0801", "cue"), ("PS92_0801", "cue")]          # given order, no sort, no spawn
    assert out == {("PS93", "cue"): ("cue", "PS93_0801"), ("PS92", "cue"): ("cue", "PS92_0801")}


def test_frozen_pairs_fanned_restores_the_parents_blas(monkeypatch):
    import threadpoolctl

    from wfield_local import analysis_kit, parallel

    def blas():
        return [i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"]

    before_env = {v: os.environ.get(v) for v in fd._BLAS_VARS}
    before = blas()

    def fake_fan(items, worker, **kw):
        assert worker is fd._frozen_pair_unit and kw["jobs"] <= 8
        parallel.pin_blas(1)                                   # what fan_out does to the parent, unrestored
        return sorted((it, ("d", it[0])) for it in items), []

    monkeypatch.setattr(analysis_kit, "fan_sessions", fake_fan)
    units = [("PS93", "lick", ["a", "b"]), ("PS92", "cue", ["c", "d"])]
    out = fd.frozen_pairs(units, jobs=12)
    assert out == {("PS92", "cue"): ("d", "PS92"), ("PS93", "lick"): ("d", "PS93")}
    assert blas() == before
    assert {v: os.environ.get(v) for v in fd._BLAS_VARS} == before_env


def test_frozen_pairs_fanned_raises_on_a_failed_unit(monkeypatch):
    from wfield_local import analysis_kit
    monkeypatch.setattr(analysis_kit, "fan_sessions",
                        lambda items, worker, **kw: ([], [(items[0], "ValueError: boom")]))
    with pytest.raises(RuntimeError, match="boom"):
        fd.frozen_pairs([("PS92", "cue", ["a", "b"]), ("PS93", "cue", ["c", "d"])], jobs=2)


def test_nightly_frozen_units_sorted_scoped_and_pooled(monkeypatch):
    from wfield_local import config, nightly_figs
    sess = [{"label": lab} for lab in ("PS94_0801", "PS92_0801", "PS92_0802", "PS93_0801", "PS94_0802", "PS95_0801")]
    monkeypatch.setattr(nightly_figs, "SESSIONS", sess)
    pooled = {"PS92": ["PS92_0801", "PS92_0802", "PS92_0817"], "PS93": ["PS93_0801"],
              "PS94": ["PS94_0801", "PS94_0802"], "PS95": ["PS95_0801", "PS95_0802"]}
    monkeypatch.setattr(config, "pooled_labels", lambda a: pooled[a])
    u = nightly_figs._frozen_units(("cue", "lick"), ["0801", "0802"], None)
    assert u["cue"] == [("PS92", "cue", ["PS92_0801", "PS92_0802"]), ("PS94", "cue", ["PS94_0801", "PS94_0802"]),
                        ("PS95", "cue", ["PS95_0801", "PS95_0802"])]      # PS93 <2 sessions, 0817 not in from
    assert [x[0] for x in u["lick"]] == ["PS92", "PS94", "PS95"]
    assert [x[0] for x in nightly_figs._frozen_units(("cue",), ["0801", "0802"], ["PS94"])["cue"]] == ["PS94"]


# ---------------------------------------------------------------------------------------------
# B: the map scripts read `session_raw_maps_cached` -- the identical call, served from the cache
# ---------------------------------------------------------------------------------------------

def _fake_raw_maps_factory(calls):
    def fake(session, align, *, post_s=2.0, variant="working"):
        calls.append((session["label"], align, post_s, variant))
        rng = np.random.RandomState(len(calls) * 0 + 3)
        q = {p: rng.randn(4, 5) for p in ("close_L", "far_R")}
        return {"raw": q, "raw_precue": {p: m * 0.5 for p, m in q.items()}, "quiet_flat": rng.randn(4, 5),
                "raw_restw": {p: m - 1 for p, m in q.items()}, "used": {"close_L": 20, "far_R": 21}}
    return fake


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            _same(a[k], b[k])
        else:
            np.testing.assert_array_equal(a[k], b[k])


@pytest.mark.parametrize("kw", [dict(align="cue", variant="working"),
                                dict(align="cue", post_s=2.0, variant="working")])
def test_cached_map_is_the_uncached_call(monkeypatch, tmp_path, kw):
    from wfield_local import position_reference_maps as prm
    from wfield_local import session_cache
    monkeypatch.setattr(session_cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("WIDEFIELD_NO_CACHE", raising=False)
    calls = []
    monkeypatch.setattr(prm, "session_raw_maps", _fake_raw_maps_factory(calls))
    s = {"label": "PSX_0801", "mc": str(tmp_path / "mc")}
    kw = dict(kw)
    align = kw.pop("align")
    direct = prm.session_raw_maps(s, align, **kw)
    miss = prm.session_raw_maps_cached(s, align, **kw)
    hit = prm.session_raw_maps_cached(s, align, **kw)
    # the cache computes through the SAME function with the SAME arguments, once; the hit is the pickled result
    assert calls == [("PSX_0801", "cue", 2.0, "working")] * 2
    _same(direct, miss)
    _same(direct, hit)


def test_chronic_stability_amplitudes_use_the_cached_identical_call(monkeypatch):
    import importlib

    from wfield_local import beta_maps, position_reference_maps as prm
    from wfield_local import locanmf_cue_lick_analysis as lca
    cs = importlib.import_module("scripts.chronic_stability")
    monkeypatch.setattr(lca, "SESSIONS", [{"label": "PSX_0801", "mc": "C:/nowhere"}])
    monkeypatch.setattr(beta_maps, "stat_mask", lambda: np.ones((4, 5), bool))
    unc, cac = [], []
    fake = _fake_raw_maps_factory(unc)
    monkeypatch.setattr(prm, "session_raw_maps", fake)

    def fake_cached(session, align, *, post_s=2.0, variant="working"):
        cac.append((session["label"], align, post_s, variant))
        return fake(session, align, post_s=post_s, variant=variant)

    monkeypatch.setattr(prm, "session_raw_maps_cached", fake_cached)
    got = cs._session_map_amplitudes("PSX_0801")
    # the call the script used to make: session_raw_maps(s, "cue", variant="working") -> post_s 2.0
    assert cac == [("PSX_0801", "cue", 2.0, "working")]
    want = {q: float(np.nanmean(np.asarray(m, float)[np.ones((4, 5), bool)]))
            for q, m in prm.reference_maps(fake({"label": "PSX_0801"}, "cue"), "precue").items()}
    assert got == want


def test_flatpool_vs_restw_calls_the_cached_map_with_the_same_arguments():
    import inspect

    from scripts.rest_migration import flatpool_vs_restw as fv
    src = inspect.getsource(fv.main)
    assert 'prm.session_raw_maps_cached(s, a.align, post_s=a.post_s, variant="working")' in src
    assert "prm.session_raw_maps(" not in src
