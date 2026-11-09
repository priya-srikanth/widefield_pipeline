"""Stale-read guards on caches whose content depends on the analysis masks but whose key does not (11-09).

After the rebuild glue-mask fix (GLUE_MASK_FIX_2026-11-09) every skip-if-exists cache built on `load_imaging`,
`roi_masks`, the brain / stat / painted masks or `component_exclusion` must store the masks' digest and refuse a
mismatch; a cache with no digest is refused inside a rebuild root and accepted at the default root. Each test fakes the
digest (`mask_keys.signature`), so nothing here reads the share.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture
def masks(monkeypatch):
    """Current masks = "glue_out"; default root (no rebuild root). `masks.rebuild()` switches the root on."""
    from wfield_local import mask_keys, rebuild_root

    class M:
        def set(self, sig):
            monkeypatch.setattr(mask_keys, "signature", lambda: sig)

        def rebuild(self):
            monkeypatch.setattr(rebuild_root, "active", lambda: True)
    monkeypatch.delenv(rebuild_root.ENV, raising=False)
    m = M()
    m.set("glue_out")
    return m


# ------------------------------------------------------------------------------------------------ the rule itself
def test_check_rule(masks, tmp_path):
    from wfield_local import mask_keys
    f = tmp_path / "x.npz"
    np.savez(f, **mask_keys.stamp())
    mask_keys.check_npz(f)                                                 # same masks: reused
    np.savez(f, mask_sig="glue_in")
    with pytest.raises(RuntimeError, match="analysis masks glue_in"):
        mask_keys.check_npz(f)                                             # other masks: refused
    np.savez(f, a=np.zeros(2))
    mask_keys.check_npz(f)                                                 # no digest, default root: accepted
    masks.rebuild()
    with pytest.raises(RuntimeError, match="before the 11-09"):
        mask_keys.check_npz(f)                                             # no digest, rebuild root: refused
    mask_keys.check_obj({"mask_sig": "glue_out"}, f)                       # pickled dicts too
    with pytest.raises(RuntimeError):
        mask_keys.check_obj({"mask_sig": "glue_in"}, f)


def test_exclusion_signature_follows_the_per_animal_mask(monkeypatch):
    from wfield_local import component_exclusion, mask_keys
    m = np.zeros((6, 7), bool)
    monkeypatch.setattr(component_exclusion, "excluded_pixels", lambda animal, include_bulbs=True: m)
    a = mask_keys.exclusion_signature("PS92")
    m[2, 3] = True
    assert mask_keys.exclusion_signature("PS92") != a


# ------------------------------------------------------------------------------------------------ item 2: ITI extracts
def _stale_and_legacy(masks, f, call, **fields):
    """A cached file on other masks is refused; without a digest it is refused only in a rebuild root."""
    np.savez(f, mask_sig="glue_in", **fields)
    with pytest.raises(RuntimeError, match="masks"):
        call()
    np.savez(f, mask_sig="glue_out", **fields)
    assert call()["cached"]
    np.savez(f, **fields)
    assert call()["cached"]
    masks.rebuild()
    with pytest.raises(RuntimeError, match="before the 11-09"):
        call()


def test_lick_matched_pixel_maps_extract(masks, tmp_path):
    import lick_matched_pixel_maps as m
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.extract(("L", str(tmp_path))))


def test_lick_cue_to_lick_extract(masks, tmp_path, monkeypatch):
    import lick_cue_to_lick_extract as m
    monkeypatch.setattr(m, "SESS", tmp_path)
    monkeypatch.setattr(m, "SESS_NL", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.session_worker("L"), version=m.VERSION)


def test_lick_cue_to_lick_extract_restnl(masks, tmp_path, monkeypatch):
    import lick_cue_to_lick_extract as m
    monkeypatch.setattr(m, "SESS_NL", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.session_worker_restnl("L"), version=m.VERSION)


def test_lick_interhemispheric_rest_extract(masks, tmp_path, monkeypatch):
    import lick_interhemispheric as m
    monkeypatch.setattr(m, "REST_DIR", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.rest_extract("L"))


def test_restw_strict_check(masks, tmp_path, monkeypatch):
    import restw_strict_check as m
    monkeypatch.setattr(m, "SESS", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.session_worker(("L", [])), version=m.VERSION)


@pytest.mark.parametrize("var", ["restw", "restnl"])
def test_lick_precue_restw(masks, tmp_path, monkeypatch, var):
    import lick_precue_restw as m
    monkeypatch.setattr(m.il, "OUT_W", tmp_path)
    d = tmp_path / m._cache(var)
    d.mkdir(parents=True)
    _stale_and_legacy(masks, d / "L.npz", lambda: m._session("L", var), version=m.VERSION)


def test_disengagement_img_worker(masks, tmp_path, monkeypatch):
    import disengagement_tests as m
    monkeypatch.setattr(m, "SESS", tmp_path)
    _stale_and_legacy(masks, tmp_path / "img_L.npz", lambda: m.img_worker("L"), version=m.VERSION)


def test_iti_hemo_check(masks, tmp_path, monkeypatch):
    import iti_hemo_check as m
    monkeypatch.setattr(m, "CACHE", tmp_path)
    _stale_and_legacy(masks, tmp_path / "L.npz", lambda: m.work("L"), cache_version=m.CACHE_VERSION)


@pytest.mark.parametrize("script", ["lick_matched_pixel_maps", "lick_cue_to_lick_extract", "lick_interhemispheric",
                                    "restw_strict_check", "lick_precue_restw", "disengagement_tests",
                                    "iti_hemo_check"])
def test_every_extract_writes_its_digest(script):
    """The writers stamp the digest (a cache written without it would be refused in a rebuild root forever)."""
    src = (REPO / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert "mask_keys.stamp()" in src


# ------------------------------------------------------------------------------------------------ item 1: O2 bundle
def test_o2_bundle_masks(masks, monkeypatch):
    from wfield_local import o2_bundle as ob
    ob.verify_masks({"mask_sig": "glue_out"})                              # desktop: same masks
    with pytest.raises(ob.BundleMismatch, match="glue_in"):
        ob.verify_masks({"mask_sig": "glue_in"})
    ob.verify_masks({"name": "legacy"})                                    # legacy, default root: accepted
    with pytest.raises(ob.BundleMismatch, match="before the 11-09"):
        ob.verify_masks({"name": "enc_rebuild_20261104b", "rebuild_root": "M:/x/rebuild"})
    # runner (no share): each pickle against bundle.json's own record
    ob.check_inputs({"mask_sig": "A"}, {"mask_sig": "A"}, "L")
    with pytest.raises(ob.BundleMismatch):
        ob.check_inputs({"mask_sig": "A"}, {"mask_sig": "B"}, "L")
    with pytest.raises(ob.BundleMismatch):
        ob.check_inputs({"mask_sig": "A"}, {}, "L")                       # pickle without a digest in a signed bundle
    with pytest.raises(ob.BundleMismatch, match="rebuild root"):
        ob.check_inputs({"rebuild_root": "M:/x/rebuild"}, {}, "L")
    ob.check_inputs({"rebuild_root": ""}, {}, "L")                         # legacy default-root bundle
    masks.rebuild()
    with pytest.raises(ob.BundleMismatch):
        ob.verify_masks({"name": "legacy"})                                # any unsigned bundle, rebuild root active


def test_o2_runner_and_collect_wire_the_checks():
    import o2_encoding as oe
    assert "o2_bundle.check_inputs(" in oe.RUNNER
    src = (REPO / "scripts" / "o2_encoding.py").read_text(encoding="utf-8")
    assert src.count("o2_bundle.verify_masks(") >= 3                       # resume, check / collect, refresh-code
    assert "x[mask_keys.FIELD] = now" in src


def test_o2_refuse_other_run(masks, monkeypatch):
    import o2_encoding as oe
    from wfield_local import o2_bundle as ob
    monkeypatch.setattr(ob, "verify", lambda rec: dict(rec))
    assert oe._refuse_other_run({"identity": {"run_tag": "t"}, "mask_sig": "glue_out"}) == {"run_tag": "t"}
    with pytest.raises(SystemExit, match="glue_in"):
        oe._refuse_other_run({"identity": {}, "mask_sig": "glue_in"})




# ------------------------------------------------------------------------------------------------ item 3: frozen fit/
def test_frozen_encoder_fit_pickles(masks, tmp_path):
    import pickle

    from wfield_local import frozen_encoder as fe
    (tmp_path / "spatial_PS92.pkl").write_bytes(pickle.dumps({"P": np.eye(2), "mask_sig": "glue_out"}))
    assert fe._spatial_of(str(tmp_path), "PS92")["P"].shape == (2, 2)
    (tmp_path / "spatial_PS92.pkl").write_bytes(pickle.dumps({"P": np.eye(2), "mask_sig": "glue_in"}))
    with pytest.raises(RuntimeError, match="glue_in"):
        fe._spatial_of(str(tmp_path), "PS92")
    fe.check_saved_masks([{"label": "a"}], tmp_path / "pre_stats.pkl")            # legacy, default root
    assert fe._stamp_masks({})["mask_sig"] == "glue_out"
    masks.rebuild()
    with pytest.raises(RuntimeError, match="before the 11-09"):
        fe.check_saved_masks([{"label": "a", "mask_sig": "glue_out"}, {"label": "b"}], tmp_path / "pre_stats.pkl")


def test_frozen_encoder_main_checks_reused_stats():
    src = (REPO / "wfield_local" / "frozen_encoder.py").read_text(encoding="utf-8")
    assert "check_saved_masks(saved, sp)" in src                                   # --reuse-stats / --fit-from
    assert "check_saved_masks(roi.values(), spr)" in src                           # pre_stats_roi.pkl
    assert src.count("_stamp_masks(") >= 4                                         # pre_stats, spatial x2, roi


def test_epoch_kernels_stats_worker(masks, tmp_path):
    import pickle

    import epoch_kernels as ek
    f = tmp_path / "L.pkl"
    f.write_bytes(pickle.dumps({"label": "L", "mask_sig": "glue_in"}))
    with pytest.raises(RuntimeError, match="glue_in"):
        ek.stats_worker(("L", "tp", None, str(tmp_path)))
    f.write_bytes(pickle.dumps({"label": "L", "mask_sig": "glue_out"}))
    assert ek.stats_worker(("L", "tp", None, str(tmp_path))) == "L"
    f.write_bytes(pickle.dumps({"label": "L"}))
    assert ek.stats_worker(("L", "tp", None, str(tmp_path))) == "L"
    masks.rebuild()
    with pytest.raises(RuntimeError, match="before the 11-09"):
        ek.stats_worker(("L", "tp", None, str(tmp_path)))


# ------------------------------------------------------------------------------------------------ item 4: state space
def test_state_space_binned_cache_is_mask_keyed(masks, tmp_path, monkeypatch):
    import state_space_sim as S
    monkeypatch.setattr(S, "BASIS", "allen")
    monkeypatch.setattr(S, "CACHE", tmp_path / "cache")
    assert S._bcache() == tmp_path / "cache" / "mglue_out"                         # never the pre-fix cache/<align>/
    p = tmp_path / "PS92_0820.npz"
    T, n = len(S.LAGS), 3
    fields = dict(cols=np.array(["ALM_left", "cortex"]), first_cue=np.zeros((n, T, 2)),
                  first_peth=np.zeros((n, len(S.LAGS_P), 2)), first_pos=np.array(["a", "b", "c"]),
                  epoch="pre", animal="PS92", first_rt=np.zeros(n))
    np.savez(p, **fields)                                                          # pre-fix rebuild extract: no digest
    with pytest.raises(RuntimeError, match="before the 11-09"):
        S.fill_cache([str(p)])                                                     # refused without the profile too
    np.savez(p, mask_sig="glue_in", **fields)
    with pytest.raises(RuntimeError, match="glue_in"):
        S.fill_cache([str(p)])
    np.savez(p, mask_sig="glue_out", **fields)
    S.fill_cache([str(p)])
    assert (tmp_path / "cache" / "mglue_out" / S.ALIGN / "PS92_0820.npz").exists()
    monkeypatch.setattr(S, "BASIS", "svd")
    assert S._bcache() == tmp_path / "cache"                                       # component bases: not on the masks


def test_state_space_lnmf_keep_is_keyed_on_the_exclusion_mask(masks, tmp_path, monkeypatch):
    import types

    import state_space_sim as S
    from wfield_local import component_exclusion as cx
    from wfield_local import joint_locanmf as jl
    monkeypatch.setattr(S, "CACHE", tmp_path)
    monkeypatch.setattr(jl, "pinned", lambda an: {"id": "b1"})
    monkeypatch.setattr(jl, "load", lambda an, bid=None, warn_stale=False: types.SimpleNamespace())
    ex = {"m": np.zeros((4, 4), bool)}
    monkeypatch.setattr(cx, "excluded_pixels", lambda an, include_bulbs=True: ex["m"])
    monkeypatch.setattr(cx, "occluded", lambda basis, an: np.array([False, ex["m"].any()]))
    np.save(tmp_path / "lnmf_keep_PS92_b1.npy", np.array([True, True]))           # a pre-fix file: never read
    assert S._lnmf_keep("PS92").tolist() == [True, True]
    ex["m"] = ex["m"].copy()
    ex["m"][1, 1] = True                                                           # the glue now in: a new file
    assert S._lnmf_keep("PS92").tolist() == [True, False]
    assert len(list(tmp_path.glob("lnmf_keep_PS92_b1_x*.npy"))) == 2


def test_state_space_svd_basis_mask_checked(tmp_path, monkeypatch):
    import state_space_sim as S
    from wfield_local import beta_maps
    (tmp_path / "basis").mkdir()
    (tmp_path / "sessions").mkdir()
    bm = np.zeros((4, 5), bool)
    bm[1:3, 1:4] = True
    np.save(tmp_path / "basis" / "mask.npy", bm.ravel())
    monkeypatch.setattr(S, "SVD", tmp_path / "sessions")
    monkeypatch.setattr(S, "BASIS", "svd")
    monkeypatch.setattr(S, "_SVD_OK", [])
    monkeypatch.setattr(beta_maps, "brain_mask", lambda: bm)
    S._check_svd_mask()
    S._SVD_OK.clear()
    bm2 = bm.copy()
    bm2[0, 0] = True
    monkeypatch.setattr(beta_maps, "brain_mask", lambda: bm2)
    with pytest.raises(RuntimeError, match="another brain mask"):
        S._check_svd_mask()


# ------------------------------------------------------------------------------------------------ item 5: DAQ_RUN_FP
def test_lp_daq_cross_check_refuses_a_daq_model_on_other_masks(masks, monkeypatch, tmp_path):
    from wfield_local import frozen_models as fm
    from wfield_local import mask_keys
    for k in ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_LP_FROZEN",
              "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_GATE_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(fm, "server_dir", lambda: None)
    tr = ["PS92_0810", "PS92_0811"]

    def spec(X):
        return fm.make_spec("PS92", "decoder", align="cue", source="locanmf", train_labels=tr, train_data=X)
    masks.set("glue_in")                                               # the DAQ run froze before the fix
    daq = spec(np.ones((2, 2)))
    fm.load_or_fit(daq, lambda: {"daq": "glue_in"}, log=lambda *_: None)
    man = next((tmp_path / "store").rglob("manifest.json"))
    assert json.loads(man.read_text(encoding="utf-8"))["mask_sig"] == "glue_in"                   # recorded in the manifest, not the spec id
    masks.set("glue_out")
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_LP_FROZEN", "daq")
    lp_side = spec(np.zeros((3, 2)))
    assert lp_side["train_fp"] == fm.DAQ_RUN_FP
    with pytest.raises(RuntimeError, match="other analysis masks"):
        fm.load_or_fit(lp_side, lambda: pytest.fail("must not fit"), log=lambda *_: None)
    # a manifest from before 11-09 (no mask_sig): accepted at the default root, refused in a rebuild root
    m = json.loads(man.read_text(encoding="utf-8"))
    m.pop(mask_keys.FIELD)
    man.write_text(json.dumps(m), encoding="utf-8")
    assert fm.load_or_fit(lp_side, lambda: pytest.fail("must not fit"), log=lambda *_: None)[0] == {"daq": "glue_in"}
    masks.rebuild()
    with pytest.raises(RuntimeError, match="other analysis masks"):
        fm.find_daq_trained(lp_side)
