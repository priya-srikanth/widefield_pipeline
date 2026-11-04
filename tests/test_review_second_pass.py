"""Review, second pass (2026-11-03 evening, REBUILD_CODE_REVIEW_20261103.md "Second pass"): N1-N5, N8, N9, N11 and the
finished partial fixes. Every store / folder here is a temp dir; nothing reads the shares."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from wfield_local import frozen_models as fm
from wfield_local import run_mode
from wfield_local import trial_classes as TC

REPO = Path(__file__).resolve().parents[1]
MODES = ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_LP_FROZEN",
         "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_GATE_SOURCE", "WIDEFIELD_PRE_FROM", "WIDEFIELD_MAP_REFS",
         "WIDEFIELD_JOINT_PROJECT_ALL", "WIDEFIELD_LOCANMF_PROJECT", "WIDEFIELD_ITI_FROZEN", "WIDEFIELD_EPOCH_FROZEN")


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for k in MODES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(fm, "server_dir", lambda: None)
    monkeypatch.setattr(TC, "class_source_signature", lambda *a, **k: "sigtest")


# ------------------------------------------------------------------------------------------------------------ N1
def test_lp_family_out_dirs_keep_the_default_lp_folder(monkeypatch):
    """N1: an LP-only family (attempt_coding / attempt_maps) under LP ALONE writes the pre-rebuild folder; any other
    mode gets its own."""
    from wfield_local import attempt_coding, attempt_maps, lp_licks
    base = lp_licks._run_dir() / "imaging_encoding"
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    assert attempt_coding.out_dir() == base / "attempt_coding"
    assert attempt_maps.out_dir() == base / "attempt_maps"
    assert attempt_maps.out_dir("restnl") == base / "attempt_maps_restnl"
    monkeypatch.setenv("WIDEFIELD_GATE_SOURCE", "lp")
    assert attempt_coding.out_dir() == base / "attempt_coding" / "gateLP"
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    assert attempt_coding.out_dir().name.startswith("gateLP_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopRun6")


#: readers of the attempt_coding / attempt_maps outputs: each must resolve its folder through the WRITER's out_dir
READERS = ("scripts/rebuild_figs/fig_goal_summary.py", "scripts/rebuild_figs/fig_goal_epochs.py",
           "scripts/early_predictors.py", "scripts/recovery_timing.py", "scripts/precue_class_matched.py",
           "scripts/decoder_retention_sessions.py", "scripts/lick_enl_target_test.py", "scripts/kinematic_normality.py",
           "scripts/cohort2_qc/quiet_task_events.py")


def test_attempt_readers_resolve_through_the_writer():
    """N1: no reader builds `<E> / "attempt_coding"` / `"attempt_maps"` itself (it would read the default folder in a
    tagged run); they call `attempt_coding.out_dir()` / `attempt_maps.out_dir()` (or `run_mode.subdir(...,
    lp_family=True)` for the explicit attempt_coding_pp folder)."""
    lit = re.compile(r"/\s*\"attempt_(?:coding|maps)\"")
    for rel in READERS:
        src = (REPO / rel).read_text(encoding="utf-8")
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        assert not lit.search(code), rel
        assert "out_dir()" in code or "lp_family=True" in code, rel


@pytest.mark.parametrize("env", [{}, {"WIDEFIELD_GATE_SOURCE": "lp"}, {"WIDEFIELD_TRIAL_GATE": "camera"},
                                 {"WIDEFIELD_HEMO_VARIANT": "meegkit_hpfit_lag_refitT_swapmask"}])
def test_reader_writer_pairs_resolve_the_same_folder(monkeypatch, env):
    """N1 / N4: under every mode, a reader of a writer's folder gets the SAME path: attempt_coding (writer out_dir vs
    the lp_family subdir readers), figures_working (H2 writers' routed_output vs grant_kit._fig_root)."""
    from wfield_local import attempt_coding, grant_kit, lp_licks
    from wfield_local.paths import PathResolver
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    base = lp_licks._run_dir() / "imaging_encoding"
    assert attempt_coding.out_dir() == run_mode.subdir(base / "attempt_coding", lp_family=True)
    fw = Path(PathResolver().root("figures_working"))
    assert run_mode.routed_output(fw, make=False, guard=False) == grant_kit._fig_root()
    # an explicit --output that already names the tag folder is NOT tagged twice (the nightly passes it down)
    tagged = run_mode.routed_output(fw, make=False, guard=False)
    assert run_mode.routed_output(tagged, make=False, guard=False) == tagged


def test_routed_output_creates_the_tagged_folder(monkeypatch, tmp_path):
    """N5: a tagged writer's folder is created; the default run creates nothing new."""
    from wfield_local import writeguard
    monkeypatch.setattr(writeguard, "assert_writable", lambda p: None)
    root = tmp_path / "fw"
    assert run_mode.routed_output(root, guard=False) == root and not root.exists()
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    p = run_mode.routed_output(root, guard=False)
    assert p.parent == root and p.is_dir()


# ------------------------------------------------------------------------------------------------------------ N2
def test_lp_final_daq_cross_check_finds_its_model(monkeypatch):
    """N2: the DAQ run (lp_final_render.setup(source="daq"): no gate-LP) freezes under sel-daq-quittail; the LP run
    with the LP-inclusive gate and WIDEFIELD_LP_FROZEN=daq must find that model. A camera-gate LP run must NOT."""
    kw = dict(align="cue", source="locanmf", train_labels=["PS92_0810", "PS92_0811"])
    daq = fm.make_spec("PS92", "epoch_decoder", train_data=np.ones((3, 2)), **kw)
    assert daq["selection"] == "sel-daq-quittail"
    fm.load_or_fit(daq, lambda: {"daq": True}, log=lambda *_: None)
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_GATE_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_LP_FROZEN", "daq")
    lp = fm.make_spec("PS92", "epoch_decoder", train_data=np.zeros((5, 2)), **kw)
    assert lp["selection"] == "sel-daq-quittail-gateLP"
    payload, status = fm.load_or_fit(lp, lambda: pytest.fail("must not fit"), log=lambda *_: None)
    assert payload == {"daq": True} and status == fm.STATUS_HIT
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    cam = fm.make_spec("PS92", "epoch_decoder", train_data=np.zeros((5, 2)), **kw)
    with pytest.raises(RuntimeError, match="DAQ"):
        fm.load_or_fit(cam, lambda: None, log=lambda *_: None)


# ------------------------------------------------------------------------------------------------------------ N3
def test_writeguard_allows_frozen_stores_and_the_session_cache(monkeypatch, tmp_path):
    from wfield_local import config, session_cache, writeguard

    class R:
        def root(self, name):
            return str(tmp_path / "cue_lick")
    monkeypatch.setattr(config, "resolver", lambda: R())
    monkeypatch.setattr(session_cache, "CACHE_DIR", tmp_path / "cue_lick" / ".cache")
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_GATE_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    fw = tmp_path / "cue_lick"
    assert not writeguard.is_writable(fw / "coding_direction_cue.png")                  # a default-named figure
    assert writeguard.is_writable(fw / "frozen_models_lpfinal" / "PS92" / "x")           # lp_final's store
    monkeypatch.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(fw / "my_store"))
    assert writeguard.is_writable(fw / "my_store" / "PS92" / "x")
    assert writeguard.is_writable(fw / ".cache" / "X_0000__k__sig.pkl")


# ------------------------------------------------------------------------------------------------------------ N8
def test_restnl_key_covers_yaml_and_helper_code(monkeypatch):
    import copy

    from wfield_local import config, restnl_mask
    src = REPO / "wfield_local" / "restnl_mask.py"
    key_src = ast.get_source_segment(src.read_text(encoding="utf-8"),
                                     next(n for n in ast.parse(src.read_text(encoding="utf-8")).body
                                          if isinstance(n, ast.FunctionDef) and n.name == "_key"))
    assert "_helper_code()" in key_src and "map_yaml_subset()" in key_src
    from wfield_local import map_collect
    base = copy.deepcopy(config.defaults())
    a = map_collect.map_yaml_subset(base)
    base2 = copy.deepcopy(base)
    base2["segmentation"]["rest"]["min_rest_s"] = 123.0
    assert map_collect.map_yaml_subset(base2) != a
    assert isinstance(restnl_mask._helper_code(), str) and len(restnl_mask._helper_code()) == 12


# ------------------------------------------------------------------------------------------------------------ N9
def test_map_collect_fails_a_camera_run_with_any_failed_session(monkeypatch):
    from wfield_local import map_collect

    def worker(it):
        if it[0] == "B":
            raise ValueError("bad class table")
        return it[0]
    res, fail = map_collect.collect([("A",), ("B",)], worker, jobs=1)
    assert list(res) == ["A"] and len(fail) == 1                       # daq: logged and continued, as before
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    with pytest.raises(RuntimeError, match="1 session job"):
        map_collect.collect([("A",), ("B",)], worker, jobs=1)
    with pytest.raises(RuntimeError):
        TC.reraise_under_camera(ValueError("x"), "here")


# ----------------------------------------------------------------------------------------------------------- N11
def test_iti_extracts_follow_only_the_data_modes(monkeypatch):
    """N11: XOUT (the skip-if-exists extracts) moves only with the data parts; OUT (outputs) with the full tag."""
    from wfield_local import config
    E = Path("E")
    for env, same in (({}, True), ({"WIDEFIELD_PRE_FROM": "0801"}, False), ({"WIDEFIELD_ITI_FROZEN": "0"}, False)):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        x = run_mode.subdir(E / "iti_licks", include=("hemo", "lnmf", "projall"))
        o = run_mode.subdir(E / "iti_licks")
        assert x == E / "iti_licks" and (o == x) == same, env
        for k in env:
            monkeypatch.delenv(k)
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", "meegkit_hpfit_lag_refitT_swapmask")
    assert run_mode.subdir(E / "iti_licks", include=("hemo", "lnmf", "projall")) == E / "iti_licks" / config.hemo_tag()
    src = (REPO / "scripts" / "iti_licks.py").read_text(encoding="utf-8")
    assert "OUT / SESS_NAME" not in src.replace("XOUT / SESS_NAME", "")


# ----------------------------------------------------------------------------------------------- partial fixes / LOW
def test_one_definition_of_stopped_and_unknown_variants_raise(monkeypatch):
    from wfield_local.grant_kit import _class_select
    ce = np.array(["stopped", "contact"], object)
    cu = np.array(["stopped", "jaw_only"], object)
    s = TC.camera_select("stopped", ce, cu, np.ones(2, bool), np.ones(2, bool))
    assert not s[0].any() and s[1].tolist() == [True, False]          # engaged rows are never stopped (= camera_rows)
    with pytest.raises(ValueError):
        TC.camera_select("miss_working", ce, cu, np.ones(2, bool), np.ones(2, bool))
    with pytest.raises(ValueError):
        _class_select("miss_working", np.ones(2, bool), np.ones(2, bool), np.zeros(2, bool))
    monkeypatch.setattr(TC, "row_classes", lambda lab, idx: np.array(["contact"] * len(idx), object))
    with pytest.raises(ValueError):
        TC.camera_rows("L", np.arange(1), np.arange(1), "bogus")


def test_lp_final_settings_do_not_touch_the_environment(monkeypatch):
    """M10: `settings()` / `scoped()` resolve the render env without changing this process outside the block."""
    import os
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import lp_final_render
    env = lp_final_render.settings(source="lp")
    assert env["WIDEFIELD_LICK_SOURCE"] == "lp" and "WIDEFIELD_LICK_SOURCE" not in os.environ
    with lp_final_render.scoped():
        assert os.environ.get("WIDEFIELD_GATE_SOURCE") == "lp"
    assert "WIDEFIELD_GATE_SOURCE" not in os.environ and "WIDEFIELD_PRE_FROM" not in os.environ


def test_retired_iti_nmatch_check_is_gone():
    assert not (REPO / "scripts" / "iti_nmatch_check.py").exists()
    assert (REPO / "scripts" / "archive" / "iti_nmatch_check.py").exists()


# ------------------------------------------------------------------------------------- L1: per-kind row perturbation
def _perturbed(X):
    Y = np.array(X, float, copy=True)
    Y.flat[0] += 1.0
    return Y


def test_attempt_axes_change_one_row_new_id(monkeypatch):
    from wfield_local import attempt_coding
    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 3))
    G = np.repeat([0, 1, 2], 8)
    sp = np.array(list(attempt_coding.SPOUTS) * 4)
    ref = np.ones(24, bool)
    kept = ["PS92_0810", "PS92_0811", "PS92_0812"]
    seen = []
    monkeypatch.setattr(fm, "load_or_fit", lambda spec, fit, **k: (seen.append(fm.spec_id(spec)) or ({}, "hit")))
    attempt_coding.stored_axes("PS92", "cue", 2.0, X, G, sp, ref, kept, "b", verbose=False)
    attempt_coding.stored_axes("PS92", "cue", 2.0, _perturbed(X), G, sp, ref, kept, "b", verbose=False)
    assert len(seen) == 2 and seen[0] != seen[1]


def test_engagement_states_change_one_row_new_id(monkeypatch):
    from types import SimpleNamespace

    from wfield_local import precue_engagement_states as pes
    seen = []
    monkeypatch.setattr(fm, "load_or_fit", lambda spec, fit, **k: (seen.append(fm.spec_id(spec)) or ({}, "hit")))
    b = SimpleNamespace(basis_id="76d884873920")
    X = np.arange(12.).reshape(4, 3)
    for data in (X, _perturbed(X)):
        pes._stored("PS92", "precue-posdec", b, ["PS92_0810"], {0}, 3, (data, np.arange(4)), lambda: {}, 2.0)
    assert len(seen) == 2 and seen[0] != seen[1]


def test_frozen_decoder_models_change_one_row_new_id(monkeypatch):
    from wfield_local import locanmf_frozen_decoder as lfd
    seen = []
    monkeypatch.setattr(fm, "load_or_fit", lambda spec, fit, **k: (seen.append(fm.spec_id(spec)) or (
        {"full": None, "loso": {}}, "hit")))
    rng = np.random.default_rng(1)
    XE = rng.standard_normal((12, 3))
    YE = np.tile([1, 2, 3], 4)
    GE = np.repeat([0, 1], 6)
    kept = ["PS92_0810", "PS92_0811"]
    for X in (XE, _perturbed(XE)):
        try:
            lfd.frozen_decoder_models(X, YE, GE, kept, [0, 1], align="cue", source="locanmf", post_s=2.0,
                                      basis_id="b")
        except Exception:                                              # noqa: BLE001 - only the spec id is checked
            pass
    assert len(seen) >= 2 and seen[0] != seen[1]
