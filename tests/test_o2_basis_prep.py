"""11-26 (O2 basis-frame prep): `o2_frozen collect` accepts the rule-11 bundles' recorded `out_root` (o2f_wave1a2 /
o2f_wave1b2 were refused with "frame folder ... differs"), and the split-row readers find the basis twin by an explicit
--basis-run (a twin collected into an O2 bundle's frames/ is not a sibling of its area run), refuse a named twin that is
absent / misnamed / incomplete, and print which twin they used (the sibling lookup alone skipped the rows silently)."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import encoder_split_rows as esr  # noqa: E402
import o2_frozen as of  # noqa: E402

TAG = of.FRAMES["spont_frA_basis"]["tag"]


# ------------------------------------------------------------------------------------------- collect: declared root
def test_declared_out_root_spellings():
    assert of.declared_out_root({"test_out_root": ""}) is None                 # a routed full bundle (main's spelling)
    assert of.declared_out_root({"test_out_root": "C:/t"}) == "C:/t"           # a TEST subset bundle
    assert of.declared_out_root({"out_root": "C:/b/frames"}) == "C:/b/frames"  # the rule-11 bundles (05e363a4)
    assert of.declared_out_root({"out_root": "C:/b/frames", "test_out_root": "C:/b/frames"}) == "C:/b/frames"
    with pytest.raises(SystemExit, match="two frame roots"):
        of.declared_out_root({"out_root": "C:/a", "test_out_root": "C:/b"})


def _bundle(tmp: Path, key: str, rkey: str, mode_files=("PS92_0806",)):
    """A minimal bundle whose run.json declares its frames root under ``rkey`` (out_root | test_out_root)."""
    d = tmp / "o2" / "bndl"
    r = d / "runs" / key
    frames = tmp / "o2" / "bndl" / "frames"
    fdir = frames / f"frozen_encoder_{of.FRAMES[key]['tag']}"
    (fdir / "fit").mkdir(parents=True)
    ps = fdir / "fit" / "pre_stats.pkl"
    ps.write_bytes(b"x" * 10)
    (d / "bundle.json").parent.mkdir(parents=True, exist_ok=True)
    (d / "bundle.json").write_text(json.dumps({"name": "bndl", "commit": "c0ffee", "mask_sig": "m"}))
    sess = r / "out" / "joint" / "sessions"
    sess.mkdir(parents=True)
    for lab in mode_files:
        (sess / f"{lab}.json").write_text(json.dumps({"label": lab}))
        np.savez(sess / f"{lab}.npz", a=np.arange(3))
    (r / "labels.txt").write_text("\n".join(mode_files))
    rm = {"identity": {}, "frame_dir": str(fdir), "variants": ["joint"], "svf_variants": ["joint"],
          "pre_stats": {"size": ps.stat().st_size, "mtime": ps.stat().st_mtime}, rkey: str(frames)}
    (r / "run.json").write_text(json.dumps(rm))
    return d, fdir


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    from wfield_local import frozen_encoder as fe
    from wfield_local import o2_bundle
    monkeypatch.setattr(o2_bundle, "verify_masks", lambda meta: None)
    monkeypatch.setattr(of.op, "verify_identity", lambda ident: None)
    monkeypatch.setattr(of.op, "_bundle_dir", lambda name: tmp_path / "o2" / name)
    monkeypatch.setattr(fe, "summarise_musall", lambda out, variants: None)
    monkeypatch.setattr(fe, "summarise_drop", lambda out: None)
    monkeypatch.setattr(fe, "out_root", lambda: tmp_path / "PRODUCTION_imaging_encoding")
    monkeypatch.delenv("FROZEN_OUT_ROOT", raising=False)
    return tmp_path


@pytest.mark.parametrize("rkey", ["out_root", "test_out_root"])
def test_collect_reads_declared_root(stubbed, rkey):
    key = "spont_frA_basis"
    d, fdir = _bundle(stubbed, key, rkey)
    a = types.SimpleNamespace(frame=key, name="bndl", mode="predict", partial=False)
    assert of.cmd_collect(a) == 0
    assert (fdir / "joint" / "sessions" / "PS92_0806.npz").exists()
    assert (fdir / "collected_o2_bndl_predict.json").exists()
    assert not (stubbed / "PRODUCTION_imaging_encoding").exists()
    assert "FROZEN_OUT_ROOT" not in os.environ                  # predict: no summary re-routing needed


def test_collect_routed_bundle_unchanged(stubbed):
    """A routed bundle (test_out_root "") still resolves the frame folder from fe.out_root(), as before."""
    key = "spont_frA_basis"
    d, fdir = _bundle(stubbed, key, "test_out_root")
    rm = json.loads((d / "runs" / key / "run.json").read_text())
    rm["test_out_root"] = ""
    (d / "runs" / key / "run.json").write_text(json.dumps(rm))
    a = types.SimpleNamespace(frame=key, name="bndl", mode="predict", partial=False)
    with pytest.raises(SystemExit, match="differs from the bundle's"):     # its frame_dir is not under fe.out_root()
        of.cmd_collect(a)


def test_collect_out_root_routes_svf_summaries(stubbed, monkeypatch):
    key = "spont_frA_basis"
    d, fdir = _bundle(stubbed, key, "out_root")
    monkeypatch.delitem(sys.modules, "fr_musall", raising=False)
    seen = {}
    fake = types.ModuleType("session_vs_frozen")
    fake.main = lambda argv: seen.setdefault("root", os.environ.get("FROZEN_OUT_ROOT"))
    monkeypatch.setitem(sys.modules, "session_vs_frozen", fake)
    sv = d / "runs" / key / "svf" / "sessions"
    sv.mkdir(parents=True)
    (sv / "PS92_0806.json").write_text("{}")
    a = types.SimpleNamespace(frame=key, name="bndl", mode="svf", partial=False)
    assert of.cmd_collect(a) == 0
    assert seen["root"] == str(fdir.parent)
    monkeypatch.delenv("FROZEN_OUT_ROOT", raising=False)


# --------------------------------------------------------------------------------------------- readers: basis twin
def _twin(tmp: Path, run_name: str, labels, fit=True, name=None):
    t = tmp / "bundle" / "frames" / (name or f"{run_name}_basis")
    (t / "joint" / "sessions").mkdir(parents=True)
    if fit:
        (t / "fit").mkdir()
    for lab in labels:
        (t / "joint" / "sessions" / f"{lab}.json").write_text("{}")
        (t / "joint" / "sessions" / f"{lab}.npz").write_bytes(b"")
    return t


def test_basis_twin_explicit_not_sibling(tmp_path, capsys):
    run = tmp_path / "LP" / f"frozen_encoder_{TAG[:-6]}"          # the area run elsewhere
    run.mkdir(parents=True)
    t = _twin(tmp_path, run.name, ["PS92_0806", "PS93_0806"])
    assert esr.basis_twin(run, str(t), ["PS92_0806", "PS93_0806"]) == t
    assert f"basis twin {t} (--basis-run)" in capsys.readouterr().out


def test_basis_twin_explicit_refusals(tmp_path):
    run = tmp_path / "LP" / "frozen_encoder_x_spont_frA"
    run.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no such folder"):
        esr.basis_twin(run, str(tmp_path / "nope"), ["PS92_0806"])
    wrong = _twin(tmp_path, run.name, ["PS92_0806"], name="frozen_encoder_x_spont_frA_spd_basis")
    with pytest.raises(SystemExit, match="not the twin"):
        esr.basis_twin(run, str(wrong), ["PS92_0806"])
    nofit = _twin(tmp_path / "b2", run.name, ["PS92_0806"], fit=False)
    with pytest.raises(SystemExit, match="no fit/"):
        esr.basis_twin(run, str(nofit), ["PS92_0806"])
    part = _twin(tmp_path / "b3", run.name, ["PS92_0806"])
    with pytest.raises(SystemExit, match="1 of 2 sessions"):
        esr.basis_twin(run, str(part), ["PS92_0806", "PS93_0806"])


def test_basis_twin_sibling_and_absent_are_printed(tmp_path, capsys):
    run = tmp_path / "frozen_encoder_x_spont_frA"
    run.mkdir()
    assert esr.basis_twin(run, None, ["PS92_0806"]) is None
    assert "split rows: NONE" in capsys.readouterr().out          # old frames: no rows, but never silently
    sib = tmp_path / f"{run.name}_basis"
    (sib / "joint" / "sessions").mkdir(parents=True)
    for s in ("json", "npz"):
        (sib / "joint" / "sessions" / f"PS92_0806.{s}").write_text("")
    assert esr.basis_twin(run, None, ["PS92_0806"]) == sib
    assert "(sibling)" in capsys.readouterr().out
    assert esr.basis_twin(run, "none", ["PS92_0806"]) is None
    assert "--basis-run none" in capsys.readouterr().out


def test_specificity_session_rows_uses_named_twin(tmp_path, monkeypatch):
    import specificity_control as sc
    run = tmp_path / "LP" / "frozen_encoder_x_spont_frA"
    (run / "joint" / "sessions").mkdir(parents=True)
    labs = ["PS92_0806", "PS92_0807"]
    off = np.linspace(-1, 2, 31)
    for lab in labs:
        (run / "joint" / "sessions" / f"{lab}.json").write_text(json.dumps(
            {"label": lab, "animal": "PS92", "epoch": "pre", "peth_parts": ["data", "full"], "areas": ["MOs_left"]}))
        np.savez(run / "joint" / "sessions" / f"{lab}.npz", peth_off_s=off, peth=np.ones((2, 6, 1, 31)),
                 peth_lick=np.ones((2, 6, 1, 31)))
    from wfield_local import config
    monkeypatch.setattr(config, "early_pre", lambda lab: False)
    twin = _twin(tmp_path, run.name, labs)
    got = {}

    def fake_rows(run_, labels, variant="joint", by_spout=False, basis=None):
        got["basis"] = basis
        return pd.DataFrame([{"animal": "PS92", "label": lb, "epoch": "pre", "window": w, "area": a, "data": 0.0,
                              "full": 0.0, "residual": 0.0} for lb in labels for w in ("cue", "lick")
                             for a in esr.SPLIT_AREAS])
    monkeypatch.setattr(esr, "residual_rows", fake_rows)
    T = sc.session_rows(run, basis_run=str(twin))
    assert got["basis"] == twin
    assert set(esr.SPLIT_AREAS) <= set(T.area)
    # rows missing for one session -> refused, not drawn without it
    monkeypatch.setattr(esr, "residual_rows", lambda *a, **k: fake_rows(*a, **k).query("label == 'PS92_0806'"))
    with pytest.raises(SystemExit, match="split rows incomplete"):
        sc.session_rows(run, basis_run=str(twin))
