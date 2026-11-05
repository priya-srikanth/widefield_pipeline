"""REBUILD_CODE 8.54: an O2 bundle recorded WITHOUT a non-input run-tag part (`o2_bundle.NON_INPUT_TAG_PARTS`, today only
"daqlk", 8.53) is accepted by `verify` / `check` / `collect`; anything else still refuses, and collect routes the outputs
into the CURRENT (``_daqlk``) folder with a provenance note."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest

SEL = "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70"
OLD_TAG = f"hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_{SEL}"      # bundle enc_rebuild_20261104b's


def _mod():
    spec = importlib.util.spec_from_file_location("o2e", Path(__file__).resolve().parents[1] / "scripts" / "o2_encoding.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def profile(monkeypatch):
    from wfield_local import render_profile
    from wfield_local import trial_classes as tc
    for k in list(os.environ):
        if k.startswith(("COMBINED_ENCODING_", "OROFACIAL_", "WIDEFIELD_", "WFIELD_", "LOCANMF_")):
            monkeypatch.delenv(k)
    for k, v in render_profile.load("cohort1_rebuild_20261103")["env"].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(tc, "class_source_signature", lambda *a, **k: "sig")


def _recorded(**over):
    from wfield_local import o2_bundle
    ident = o2_bundle.identity()
    return {**ident, "run_tag": OLD_TAG, **over}


def test_the_list_is_explicit():
    from wfield_local import config, o2_bundle
    assert o2_bundle.NON_INPUT_TAG_PARTS == (config.LICK_DETECTOR_TAG,) == ("daqlk",)


def test_current_tag_is_the_old_one_plus_daqlk(profile):
    from wfield_local import o2_bundle, run_mode
    assert run_mode.tag() == OLD_TAG + "_daqlk"
    assert o2_bundle.tag_without_non_input_parts() == OLD_TAG


def test_exact_identity_still_passes_unchanged(profile):
    from wfield_local import o2_bundle
    ident = o2_bundle.identity()
    assert o2_bundle.verify(ident) == ident


def test_bundle_without_daqlk_is_accepted(profile, capsys):
    from wfield_local import o2_bundle
    got = o2_bundle.verify(_recorded())
    assert got["run_tag"] == OLD_TAG + "_daqlk" and got["recorded_run_tag"] == OLD_TAG
    assert got["accepted_parts"] == ["daqlk"]
    assert "non-input part(s) ['daqlk']" in capsys.readouterr().out
    assert _mod()._refuse_other_run({"identity": _recorded()})["accepted_parts"] == ["daqlk"]


@pytest.mark.parametrize("bad", [
    {"run_tag": OLD_TAG.replace("_pre0801", "")},                         # another part missing too
    {"run_tag": OLD_TAG.replace("stopA2jex", "stopA2jsb")},               # another rule
    {"run_tag": OLD_TAG + "_daqlkx"},                                     # no fuzzy match
    {"run_tag": OLD_TAG + "_daql"},
    {"run_tag": OLD_TAG, "selection": "sel-daq-quittail"},                # tag ok, selection differs
    {"run_tag": OLD_TAG, "area_basis": "other"},                          # tag ok, area basis differs
])
def test_anything_else_is_refused(profile, bad):
    from wfield_local import o2_bundle
    with pytest.raises(o2_bundle.BundleMismatch):
        o2_bundle.verify(_recorded(**bad))
    with pytest.raises(SystemExit, match="not of the current run"):
        _mod()._refuse_other_run({"identity": _recorded(**bad)})


def test_a_later_run_without_daqlk_does_not_accept_a_daqlk_bundle(profile, monkeypatch):
    """The acceptance is one way: recorded = current MINUS the listed parts, never plus."""
    from wfield_local import o2_bundle, run_mode
    monkeypatch.setattr(run_mode, "lickdet_part", lambda: "")
    idx = [n for n, _ in run_mode._ALL].index("lickdet")
    monkeypatch.setattr(run_mode, "_ALL", run_mode._ALL[:idx] + (("lickdet", lambda: ""),) + run_mode._ALL[idx + 1:])
    assert run_mode.tag() == OLD_TAG
    with pytest.raises(o2_bundle.BundleMismatch):
        o2_bundle.verify(_recorded(run_tag=OLD_TAG + "_daqlk"))


def test_collect_routes_into_the_daqlk_folder_with_provenance(profile, tmp_path, monkeypatch):
    from wfield_local import combined_encoding as ce
    from wfield_local import writeguard
    m = _mod()
    d = tmp_path / "enc_rebuild_20261104b"
    (d / "out" / "main" / "sessions").mkdir(parents=True)
    (d / "bundle.json").write_text(json.dumps({"identity": _recorded()}))
    (d / "out" / "main" / "run.json").write_text(json.dumps({"order": "movement_first", "bundle_env": {}}))
    (d / "out" / "main" / "sessions" / "PS95_0917.json").write_text(json.dumps({"ok": 1}))
    seen = {}

    def out_dir(rv, order):
        from wfield_local import run_mode
        seen["tag"] = run_mode.tag()
        return tmp_path / "imaging_encoding" / run_mode.tag() / order
    monkeypatch.setattr(m, "bundle_dir", lambda rv, name: d)
    monkeypatch.setattr(ce, "out_dir", out_dir)
    monkeypatch.setattr(writeguard, "assert_writable", lambda p: None)
    assert m.cmd_collect(argparse.Namespace(name="enc_rebuild_20261104b", tag="main", run=None)) == 0
    dst = tmp_path / "imaging_encoding" / (OLD_TAG + "_daqlk") / "movement_first"
    assert seen["tag"].endswith("_daqlk") and (dst / "sessions" / "PS95_0917.json").exists()
    prov = json.loads((dst / "collected_enc_rebuild_20261104b__main.json").read_text())
    assert prov["recorded_run_tag"] == OLD_TAG and prov["accepted_parts"] == ["daqlk"]
    assert prov["run_tag"] == OLD_TAG + "_daqlk"


def test_collect_move_leaves_no_second_copy(profile, tmp_path, monkeypatch):
    """collect --move (Priya 11-05): the outputs are moved into the run folder, the bundle's sessions/ is left empty
    with a MOVED_TO.txt pointer, the provenance json records moved=True, and a JSON with a stray trailing byte is still
    repaired after the move."""
    from wfield_local import combined_encoding as ce
    from wfield_local import writeguard
    m = _mod()
    d = tmp_path / "enc_rebuild_20261104b"
    ses = d / "out" / "main" / "sessions"
    ses.mkdir(parents=True)
    (d / "bundle.json").write_text(json.dumps({"identity": _recorded()}))
    (d / "out" / "main" / "run.json").write_text(json.dumps({"order": "movement_first", "bundle_env": {}}))
    (ses / "PS95_0917.json").write_text(json.dumps({"ok": 1}) + "}")
    (ses / "PS95_0917.npz").write_bytes(b"\x00\x01\x02")
    monkeypatch.setattr(m, "bundle_dir", lambda rv, name: d)
    monkeypatch.setattr(ce, "out_dir", lambda rv, order: tmp_path / "enc" / order)
    monkeypatch.setattr(writeguard, "assert_writable", lambda p: None)
    assert m.cmd_collect(argparse.Namespace(name="enc_rebuild_20261104b", tag="main", run=None, move=True)) == 0
    dst = tmp_path / "enc" / "movement_first"
    assert sorted(p.name for p in (dst / "sessions").iterdir()) == ["PS95_0917.json", "PS95_0917.npz"]
    assert (dst / "sessions" / "PS95_0917.npz").read_bytes() == b"\x00\x01\x02"
    assert json.loads((dst / "sessions" / "PS95_0917.json").read_text()) == {"ok": 1}
    assert list(ses.iterdir()) == []
    assert str(dst / "sessions") in (d / "out" / "main" / "MOVED_TO.txt").read_text()
    assert json.loads((dst / "collected_enc_rebuild_20261104b__main.json").read_text())["moved"] is True
