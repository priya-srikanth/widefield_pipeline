"""Cohort-1 rebuild (2026-11-03) switches: one hemo switch, pinned joint basis, project_all on the basis variant,
run-mode tags, cache prune guards, per-session LocaNMF projection. The DEFAULT must be byte-identical to before;
non-default modes must agree everywhere and never mix / overwrite / prune the default."""
from __future__ import annotations

import copy
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from wfield_local import config

MODE_ENVS = ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LOCANMF_VARIANT", "WIDEFIELD_LOCANMF_PROJECT", "WIDEFIELD_PRE_FROM",
             "WIDEFIELD_MAP_REFS", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_TRIAL_WINDOW", "WIDEFIELD_JOINT_PROJECT_ALL",
             "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_GATE_SOURCE", "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_LP_FROZEN")
VAR = "meegkit_hpfit_lag_refitT_swapmask"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os
    for k in list(os.environ):
        if k in MODE_ENVS or re.match(r"^WIDEFIELD_[A-Z0-9_]+_FROZEN$", k):
            monkeypatch.delenv(k, raising=False)


def _yaml_variant(monkeypatch, v):
    """defaults.yaml hemo.variant flipped to ``v`` (the env var unset)."""
    d = copy.deepcopy(config.defaults())
    d["hemo"]["variant"] = v
    monkeypatch.setattr(config, "defaults", lambda session=None: d)


# ------------------------------------------------------------------------------------------- (a) one hemo switch
def test_default_paths_unchanged():
    from wfield_local import combined_encoding as ce
    from wfield_local import hemo_variants as hv
    assert config.hemo_variant() == "meegkit_hpfit"
    assert config.hemo_dir_name() == "hemo_meegkit_hpfit"
    assert ce.hemo_dir() == "hemo_meegkit_hpfit"
    assert hv.ADOPTED == "meegkit_hpfit" and hv.adopted() == "meegkit_hpfit"
    assert config.locanmf_dir_name() == config.defaults()["locanmf"]["output_dir_name"] \
        == "locanmf_affine8v1_hemo_meegkit_hpfit"
    assert config.svtcorr_path("X:/s/motion_corrected") == \
        "X:/s/motion_corrected/wfield_local_results/hemo_meegkit_hpfit/SVTcorr.npy"
    assert config.hemo_tag() == ""


def _agree(v):
    from wfield_local import combined_encoding as ce
    from wfield_local import hemo_variants as hv
    hv = importlib.reload(hv)
    assert config.hemo_variant() == v
    assert ce.hemo_dir() == f"hemo_{v}" == config.hemo_dir_name()
    assert hv.ADOPTED == v == hv.adopted()
    assert config.locanmf_dir_name() == f"locanmf_affine8v1_hemo_{v}"
    assert config.svtcorr_path("X:/m").endswith(f"/hemo_{v}/SVTcorr.npy")
    assert config.hemo_manifest_path("X:/m").endswith(f"wfield_local_results/hemo_{v}/manifest.json")
    assert config.hemo_tag() == f"hv-{v}"


def test_non_default_variant_by_env_agrees_everywhere(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    try:
        _agree(VAR)
    finally:
        monkeypatch.delenv("WIDEFIELD_HEMO_VARIANT")
        importlib.reload(importlib.import_module("wfield_local.hemo_variants"))


def test_non_default_variant_by_yaml_agrees_everywhere(monkeypatch):
    _yaml_variant(monkeypatch, VAR)
    try:
        _agree(VAR)
    finally:
        monkeypatch.undo()
        importlib.reload(importlib.import_module("wfield_local.hemo_variants"))


def test_locanmf_variant_override_still_wins(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    monkeypatch.setenv("WIDEFIELD_LOCANMF_VARIANT", "meegkit_hpfit")
    assert config.locanmf_dir_name() == "locanmf_affine8v1_hemo_meegkit_hpfit"


def test_build_default_is_a_variants_key():
    from wfield_local import hemo_variants as hv
    assert hv.BUILD_DEFAULT in hv.VARIANTS


def test_no_hardcoded_adopted_read_paths():
    """No analysis script locates its data with a literal adopted dir or the module constant (use config)."""
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in ["scripts/slowband_decode_check.py", "wfield_local/combined_encoding.py", "wfield_local/postcue_window_test.py"]:
        src = (root / f).read_text(encoding="utf-8")
        if re.search(r'hemo_\{(hv\.)?ADOPTED\}|"hemo_meegkit_hpfit"|"/hemo_meegkit_hpfit/', src):
            bad.append(f)
    assert not bad, bad


# ------------------------------------------------------------------------------------------- (b) joint basis
class _FakeBasis:
    def __init__(self, animal="PS92", bid="76d884873920", hemo=None, labels=("PS92_0806",)):
        from wfield_local import joint_locanmf as jl
        self.manifest = {"animal": animal, "basis_id": bid, "labels": list(labels), "ncomp": 2, "rank": 2}
        if hemo is not None:
            self.manifest["hemo_variant"] = hemo
        self.basis_id = bid
        self._jl = jl


def test_project_all_default_off_and_variant_on(monkeypatch):
    from wfield_local import joint_locanmf as jl
    b = _FakeBasis()
    assert jl.basis_hemo_variant(b) == "meegkit_hpfit"
    assert jl.project_all(b) is False and jl.project_all() is False
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    assert jl.project_all(b) is True


def test_project_all_on_when_switch_is_made_in_yaml(monkeypatch):
    """Trap T3: the old rule compared the env var with defaults.yaml, so a yaml flip left project_all OFF."""
    from wfield_local import joint_locanmf as jl
    _yaml_variant(monkeypatch, VAR)
    assert jl.project_all(_FakeBasis()) is True


def test_project_all_off_for_a_basis_built_on_the_variant(monkeypatch):
    from wfield_local import joint_locanmf as jl
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    assert jl.project_all(_FakeBasis(bid="newbasis0000", hemo=VAR)) is False
    monkeypatch.setenv("WIDEFIELD_JOINT_PROJECT_ALL", "1")
    assert jl.project_all(_FakeBasis(bid="newbasis0000", hemo=VAR)) is True


def _real_basis(tmp_path, bid, built, labels=("PS92_0806",), hemo=None):
    from wfield_local import joint_locanmf as jl
    d = tmp_path / "PS92" / bid
    d.mkdir(parents=True)
    man = {"basis_id": bid, "animal": "PS92", "labels": list(labels), "ncomp": 2, "rank": 2, "built_utc": built}
    if hemo:
        man["hemo_variant"] = hemo
    (d / "manifest.json").write_text(json.dumps(man))
    A = np.zeros((2, 2, 2), np.float32)
    A[0, 0, 0] = A[1, 1, 1] = 1.0
    np.save(d / "A.npy", A)
    np.save(d / f"C_{labels[0]}.npy", np.ones((2, 5), np.float32))
    return jl.Basis(d)


def test_pinned_basis_wins_over_newer(monkeypatch, tmp_path):
    from wfield_local import joint_locanmf as jl
    monkeypatch.setattr(jl, "BASIS_DIR", tmp_path)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: None)
    _real_basis(tmp_path, "76d884873920", "2026-08-15T00:00:00Z")
    _real_basis(tmp_path, "aaaaaaaaaaaa", "2026-11-03T00:00:00Z")
    assert jl.pinned("PS92")["id"] == "76d884873920"
    assert jl.load("PS92").basis_id == "76d884873920"
    assert jl.load("PS92", bid="aaaaaaaaaaaa").basis_id == "aaaaaaaaaaaa"     # an explicit id still wins


def test_unpinned_animal_keeps_newest(monkeypatch, tmp_path):
    from wfield_local import joint_locanmf as jl
    d = copy.deepcopy(config.defaults())
    d["joint_basis"]["pins"].pop("PS92")
    monkeypatch.setattr(config, "defaults", lambda session=None: d)
    monkeypatch.setattr(jl, "BASIS_DIR", tmp_path)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: None)
    _real_basis(tmp_path, "76d884873920", "2026-08-15T00:00:00Z")
    _real_basis(tmp_path, "aaaaaaaaaaaa", "2026-11-03T00:00:00Z")
    assert jl.load("PS92").basis_id == "aaaaaaaaaaaa"


def test_variant_never_mixes_stored_and_projected_C(monkeypatch, tmp_path):
    from wfield_local import joint_locanmf as jl
    b = _real_basis(tmp_path, "76d884873920", "2026-08-15T00:00:00Z")
    monkeypatch.setattr(type(b), "regions", property(lambda self: np.array([1, 2])))
    sess = {"label": "PS92_0806", "mc": "X:/none"}
    projected = []
    monkeypatch.setattr(b, "project", lambda s, with_diagnostics=False, **k: (
        projected.append(s["label"]) or ((np.zeros((2, 5)), {"variance_captured": 1.0}) if with_diagnostics
                                          else np.zeros((2, 5)))))
    # default: in-fit session served its stored C
    assert np.all(b.signal_or_project(sess) != 0) and projected == []
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    with pytest.raises(RuntimeError):
        b.signal("PS92_0806")                                  # stored C refused under another variant
    b.signal_or_project(sess)
    src = jl.BasisSource(b, sess)
    monkeypatch.setattr("wfield_local.session_cache.cached", lambda s, k, f, **kw: f())
    src.signal()
    assert projected == ["PS92_0806", "PS92_0806"]             # in-fit session PROJECTED in both paths
    assert src.key.endswith(":projall")


def test_nolick_joint_signal_projects_under_variant(monkeypatch, tmp_path):
    from wfield_local import nolick_decoder as nd
    b = _real_basis(tmp_path, "76d884873920", "2026-08-15T00:00:00Z")
    called = []
    monkeypatch.setattr(b, "project", lambda s, with_diagnostics=False: (called.append(1) or
                                                                        (np.zeros((2, 5)), {"variance_captured": .5})))
    monkeypatch.setattr(type(b), "regions", property(lambda self: np.array([1, 2])))
    nd._joint_signal(b, {"label": "PS92_0806", "mc": "X:/none"})
    assert called == []
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    nd._joint_signal(b, {"label": "PS92_0806", "mc": "X:/none"})
    assert called == [1]


# ------------------------------------------------------------------------------------------- (c) run-mode tags
def test_default_run_has_no_tag():
    from wfield_local import lp_licks, run_mode
    assert run_mode.parts() == [] and run_mode.tag() == "" and run_mode.cache_tag() == ""
    assert lp_licks.output_tag() is None
    assert lp_licks.output_dir(Path("X:/a/grant_figures/epoch")) == Path("X:/a/grant_figures/epoch")
    assert run_mode.keyed_name("grant_2.json") == "grant_2.json"
    assert run_mode.subdir(Path("X:/a")) == Path("X:/a")


def test_every_mode_gets_a_part(monkeypatch):
    from wfield_local import lp_licks, run_mode
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    monkeypatch.setenv("WIDEFIELD_LOCANMF_PROJECT", "1")
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.setenv("WIDEFIELD_MAP_REFS", "restnl, rest")
    monkeypatch.setenv("WIDEFIELD_EPOCH_FROZEN", "0")
    monkeypatch.setenv("WIDEFIELD_NOLICK_FROZEN", "off")
    assert run_mode.parts() == [f"hv-{VAR}", "lnmfproj", "pre0801", "sel-camera-cue_stop", "refs-restnl+rest",
                                "inmem-epoch+nolick"]
    assert run_mode.cache_parts() == [f"hv-{VAR}", "lnmfproj", "sel-camera-cue_stop"]
    t = run_mode.tag()
    assert lp_licks.output_tag() == t
    assert lp_licks.output_dir(Path("X:/a/grant_figures/epoch")) == Path("X:/a/grant_figures") / t / "epoch"
    assert run_mode.keyed_name("grant_2.json") == f"grant_2__{t}.json"


def test_lp_tag_combines(monkeypatch):
    from wfield_local import lp_licks
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    monkeypatch.setenv("WIDEFIELD_LP_FROZEN", "daq")
    assert lp_licks.output_tag() == "LP_licks_pre0801_lpdaqtrained"


def test_writeguard_refuses_untagged_share_write_in_variant_mode(monkeypatch):
    from wfield_local import writeguard
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    with pytest.raises(writeguard.WriteGuardError):
        writeguard.assert_writable("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/x.png")
    writeguard.assert_writable(f"M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/hv-{VAR}/epoch/x.png")


def test_trial_gate_default_daq():
    from wfield_local import trial_classes as tc
    assert tc.gate_mode() == "daq" and tc.run_selection_tag() is None
    assert tc.run_selection() == "sel-daq-quittail"


# ------------------------------------------------------------------------------------------- (d) prune guards
def _fake_session(tmp_path):
    return {"label": "PS92_0806", "mc": str(tmp_path / "mc"), "h5": "", "behavior_trials": ""}


def test_session_cache_default_kind_unchanged_and_variant_never_prunes_it(monkeypatch, tmp_path):
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("WIDEFIELD_NO_CACHE", raising=False)
    s = _fake_session(tmp_path)
    assert sc.effective_kind("tf-cue-abc") == "tf-cue-abc"
    sc.cached(s, "tf-cue-abc", lambda: 1, verbose=False)
    files = sorted(p.name for p in (tmp_path / "cache").glob("*.pkl"))
    assert len(files) == 1 and files[0].startswith("PS92_0806__tf-cue-abc__")
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    assert sc.effective_kind("tf-cue-abc") == f"tf-cue-abc-hv-{VAR}"
    assert sc.cached(s, "tf-cue-abc", lambda: 2, verbose=False) == 2
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    assert sc.cached(s, "tf-cue-abc", lambda: 3, verbose=False) == 3
    names = sorted(p.name for p in (tmp_path / "cache").glob("*.pkl"))
    assert files[0] in names and len(names) == 3                 # nothing pruned across modes
    monkeypatch.delenv("WIDEFIELD_HEMO_VARIANT")
    monkeypatch.delenv("WIDEFIELD_TRIAL_GATE")
    assert sc.cached(s, "tf-cue-abc", lambda: 9, verbose=False) == 1     # the default entry is still served


def test_restnl_stem(monkeypatch):
    from wfield_local import restnl_mask as rm
    assert rm._stem("PS92_0806") == "PS92_0806__restnl"
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    assert rm._stem("PS92_0806") == f"PS92_0806__restnl-hv-{VAR}"
    # the prune glob of one mode never matches the other's files
    assert not Path(f"PS92_0806__restnl-hv-{VAR}__k.npz").match("PS92_0806__restnl__*.npz")


# ------------------------------------------------------------------------------------------- (j) projection path
def test_project_C_recovers_exact_C():
    from wfield_local import locanmf_session as ls
    rng = np.random.default_rng(0)
    H, W, n, K, T = 6, 5, 3, 4, 50
    A = rng.random((H, W, n)).astype(np.float32)
    A[0, 0, :] = np.nan                                         # outside every footprint
    Af = np.nan_to_num(A.reshape(-1, n)).astype(np.float64)
    M = rng.standard_normal((n, K))
    Uf = Af @ M                                                 # spatial basis inside the footprints' span
    SVT = rng.standard_normal((K, T))
    C = M @ SVT                                                 # so U @ SVT = A @ C exactly
    np.testing.assert_allclose(ls.project_C(A, Uf, SVT), C, atol=1e-5)


def test_load_C_default_is_the_stored_array(monkeypatch, tmp_path):
    from wfield_local import locanmf_session as ls
    mc = tmp_path / "mc"
    d = mc / config.locanmf_dir_name()
    d.mkdir(parents=True)
    C = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(d / "PS92_0806_locanmf_C.npy", C)
    out = ls.load_C({"label": "PS92_0806", "mc": str(mc)})
    assert out.dtype == C.dtype and np.array_equal(out, C)


def test_variant_without_decomposition_fails_loudly(monkeypatch, tmp_path):
    from wfield_local import locanmf_session as ls
    mc = tmp_path / "mc"
    (mc / "locanmf_affine8v1_hemo_meegkit_hpfit").mkdir(parents=True)
    np.save(mc / "locanmf_affine8v1_hemo_meegkit_hpfit" / "PS92_0806_locanmf_C.npy", np.ones((2, 2)))
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    with pytest.raises(FileNotFoundError):
        ls.load_C({"label": "PS92_0806", "mc": str(mc)})       # never the adopted C under the variant's label
    monkeypatch.setenv("WIDEFIELD_LOCANMF_PROJECT", "1")
    assert config.locanmf_dir_name() == "locanmf_affine8v1_hemo_meegkit_hpfit"   # footprints from the stored dir


def test_no_direct_C_loads_outside_locanmf_session():
    root = Path(__file__).resolve().parents[1] / "wfield_local"
    bad = [p.name for p in root.glob("*.py") if p.name not in ("locanmf_session.py", "locanmf_lick_aligned.py")
           and re.search(r"np\.load\([^\n]*_locanmf_C\.npy", p.read_text(encoding="utf-8"))]
    assert not bad, bad


# ------------------------------------------------------------------------------------------- (i) nolick summarize
def test_nolick_summarize_survives_an_arm_too_small_to_evaluate():
    from wfield_local import nolick_analysis as na
    res = {"cue": {"engaged": {"n": 40, "balanced_accuracy": .5, "bal_null_mean": .17, "bal_p": .01, "accuracy": .5,
                               "raw_null_mean": .17, "majority_class_floor": .2},
                   "undetected_working": {"n": 12, "note": "too few trials to evaluate (need >=30)"}}}
    txt = na.summarize(res, fh=__import__("io").StringIO())
    assert "undetected_working" in txt and "not evaluated" in txt and "engaged" in txt


# ------------------------------------------------------------------------------------------- (f) camera trial gate
def _old_class_select(variant, sess_e, sess_u, not_eng):
    """FROZEN COPY of grant_kit._class_select before the camera rule (rebuild 11-03)."""
    e_none = np.zeros(len(sess_e), bool)
    u_none = np.zeros(len(sess_u), bool)
    if variant == "stopped":
        return e_none, (sess_u & not_eng) if len(sess_u) else u_none
    if variant == "working":
        return sess_e, (sess_u & ~not_eng) if len(sess_u) else u_none
    return sess_e, u_none


def test_class_select_default_is_byte_identical():
    from wfield_local.grant_kit import _class_select
    rng = np.random.default_rng(1)
    for n_u in (0, 40):
        se, su, ne = rng.random(50) > .5, rng.random(n_u) > .5, rng.random(n_u) > .7
        for v in ("lick", "working", "stopped"):
            a, b = _class_select(v, se, su, ne), _old_class_select(v, se, su, ne)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            a2 = _class_select(v, se, su, ne, cls=(np.array(["x"] * 50), np.array(["x"] * n_u)))
            assert np.array_equal(a2[0], b[0]) and np.array_equal(a2[1], b[1])     # cls ignored under daq


def test_class_select_camera_rule(monkeypatch):
    from wfield_local.grant_kit import _class_select
    ce = np.array(["contact", "contact", "unknown", "incomplete"], object)
    cu = np.array(["jaw_only", "no_response", "unknown", "incomplete"], object)
    se, su, ne = np.ones(4, bool), np.ones(4, bool), np.array([False, True, True, False])
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    with pytest.raises(RuntimeError):
        _class_select("working", se, su, ne)                               # no silent DAQ fallback
    w = _class_select("working", se, su, ne, cls=(ce, cu))
    assert w[0].tolist() == [True, True, False, True] and w[1].tolist() == [True, False, False, True]
    s = _class_select("stopped", se, su, ne, cls=(ce, cu))
    assert not s[0].any() and s[1].tolist() == [False, True, False, False]
    lk = _class_select("lick", se, su, ne, cls=(ce, cu))
    assert lk[0].all() and not lk[1].any()                                 # contact-only: unchanged (PS93_0813)
    d = _class_select("working", se, su, ne, cls=(ce, cu), rule="daq")     # the quit-tail robustness row
    o = _old_class_select("working", se, su, ne)
    assert np.array_equal(d[0], o[0]) and np.array_equal(d[1], o[1])


def test_camera_rows(monkeypatch):
    from wfield_local import trial_classes as tc
    cls = np.array(["contact", "incomplete", "jaw_only", "no_response", "unknown"], object)
    monkeypatch.setattr(tc, "row_classes", lambda lab, idx: cls[np.asarray(idx, int)])
    ie, inl = np.array([0, 4]), np.array([1, 2, 3, 4])
    assert [m.tolist() for m in tc.camera_rows("L", ie, inl, "working")] == [[True, False], [True, True, False, False]]
    assert [m.tolist() for m in tc.camera_rows("L", ie, inl, "miss_working")] == [[False, False],
                                                                                [True, True, False, False]]
    assert [m.tolist() for m in tc.camera_rows("L", ie, inl, "stopped")] == [[False, False], [False, False, True, False]]


def _old_cue_classes(table, cue_s, tol=0.020):
    """FROZEN COPY of scripts/jaw_only_incomplete_figs.cue_classes before the move (rebuild 11-03)."""
    cue_s = np.asarray(cue_s, float)
    out = np.full(len(cue_s), "unknown", dtype=object)
    if table is None or not len(table):
        return out
    t = table.dropna(subset=["cue_daq"]).sort_values("cue_daq")
    tv, cv = t.cue_daq.to_numpy(float), t.cls.to_numpy(object)
    j = np.clip(np.searchsorted(tv, cue_s), 1, len(tv) - 1) if len(tv) > 1 else np.zeros(len(cue_s), int)
    if len(tv) > 1:
        j = np.where(np.abs(tv[j - 1] - cue_s) <= np.abs(tv[j] - cue_s), j - 1, j)
    ok = np.abs(tv[j] - cue_s) <= tol
    out[ok] = cv[j[ok]]
    return out


def test_cue_classes_moved_identically():
    import pandas as pd

    from wfield_local import trial_classes as tc
    rng = np.random.default_rng(2)
    tv = np.sort(rng.random(30) * 100)
    T = pd.DataFrame({"cue_daq": tv, "cls": rng.choice(list(tc.CLASSES), 30)})
    q = np.concatenate([tv + rng.normal(0, .01, 30), rng.random(10) * 100])
    assert tc.cue_classes(T, q).tolist() == _old_cue_classes(T, q).tolist()


# ------------------------------------------------------------------------------------------- (g) per-block pre
def test_apply_pre_from(monkeypatch):
    import os
    assert config.apply_pre_from(None) is None and "WIDEFIELD_PRE_FROM" not in os.environ
    assert config.apply_pre_from("0801") == "0801" and os.environ["WIDEFIELD_PRE_FROM"] == "0801"
    with pytest.raises(SystemExit):
        config.require_full_pre("drift decoder")
    assert config.apply_pre_from("all") is None and "WIDEFIELD_PRE_FROM" not in os.environ
    config.require_full_pre("drift decoder")
    with pytest.raises(ValueError):
        config.apply_pre_from("Aug")


def test_pre_from_is_not_a_config_default():
    assert "pre_from" not in json.dumps(config.defaults().get("decode", {}))
    assert config.pre_from() is None


# ------------------------------------------------------------------------------------------- (h) cache rebuild
def test_legacy_date8_shim_is_gone(monkeypatch):
    from wfield_local import position_reference_maps as prm
    a = prm._raw_maps_code()
    monkeypatch.setenv("WIDEFIELD_PRM_LEGACY_DATE8", "1")
    assert prm._raw_maps_code() == a
    assert not hasattr(prm, "LEGACY_DATE8_ENV")


def test_map_digest_closure_is_narrow():
    from wfield_local import map_collect, precue_engagement_states as pes
    src = map_collect.closure(pes, "engagement_gate").source()
    assert "def engagement_gate" in src
    assert "def run_animal" not in src and "def _stored" not in src            # unrelated functions are not hashed
    with pytest.raises(KeyError):
        map_collect.closure(pes, "no_such_function").source()


def test_engagement_states_store_opt_out(monkeypatch):
    from wfield_local import precue_engagement_states as pes
    monkeypatch.setenv("WIDEFIELD_ENGSTATES_FROZEN", "0")
    assert pes._stored("PS92", "precue-posdec", None, ["PS92_0806"], {0}, 3, None, lambda: "fit", 2.0) == "fit"


def test_engagement_states_store_freezes(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from wfield_local import frozen_models as fm
    from wfield_local import precue_engagement_states as pes
    monkeypatch.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(fm, "server_dir", lambda: None)
    b = SimpleNamespace(basis_id="76d884873920")
    n = [0]

    def fit():
        n[0] += 1
        return {"m": 1}
    args = ("PS92", "precue-posdec", b, ["PS92_0806"], {0}, 3, (np.ones((2, 3)), np.array([0, 1])), fit, 2.0)
    assert pes._stored(*args) == {"m": 1} and pes._stored(*args) == {"m": 1} and n[0] == 1


# ------------------------------------------------------------------------------------------- (k) map npz tag
def test_maps_dir_tag(monkeypatch):
    from wfield_local import preprocess_deck as pd_
    assert config.maps_dir_tag() == "affine8v1"
    assert pd_._tagged_map_dir("spout_trial_averages_affine8v1") == "spout_trial_averages_affine8v1"
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", VAR)
    assert config.maps_dir_tag() == f"affine8v1_hv-{VAR}"
    assert pd_._tagged_map_dir("lick_aligned_affine8v1") == f"lick_aligned_affine8v1_hv-{VAR}"
    assert pd_._tagged_map_dir("motion_qc") == "motion_qc"


# ------------------------------------------------------------------------------------------- (c) fixed-name JSONs
def test_fixed_name_jsons_are_mode_tagged(monkeypatch):
    from wfield_local.grant_kit import _mode_name
    assert _mode_name("section_g.json") == "section_g.json"
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    assert _mode_name("section_g.json") == "section_g__pre0801.json"
