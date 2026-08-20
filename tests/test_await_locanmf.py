"""Tests for wfield_local.await_locanmf: sessions.yaml text-insertion (must preserve format) + detection."""
import os
import sys
from pathlib import Path

import pytest
import yaml

from wfield_local import await_locanmf as aw


def test_ensure_conda_prefix_derives_from_interpreter(monkeypatch):
    # unset -> derived from sys.executable's env dir (so the git pre-push hook can find pytest)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "executable", r"C:/Users/x/.conda/envs/locanmf/python.exe")
    aw._ensure_conda_prefix()
    assert Path(os.environ["CONDA_PREFIX"]) == Path(r"C:/Users/x/.conda/envs/locanmf")
    # posix-style bin/ layout resolves up one level
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "executable", "/home/x/miniconda3/envs/locanmf/bin/python")
    aw._ensure_conda_prefix()
    assert Path(os.environ["CONDA_PREFIX"]) == Path("/home/x/miniconda3/envs/locanmf")
    # already set -> left untouched
    monkeypatch.setenv("CONDA_PREFIX", "/preset/env")
    aw._ensure_conda_prefix()
    assert os.environ["CONDA_PREFIX"] == "/preset/env"


SAMPLE = (
    "# header comment line one\n"
    "# header comment line two\n"
    "sessions:\n"
    "  PS92:\n"
    '    "0807":\n'
    '      mc: "20260807/PS92_x/motion_corrected"\n'
    '      h5: "20260807/PS92_x.h5"\n'
    "      regime: B\n"
    "      fmdir: null\n"
    "  PS93:\n"
    '    "0807":\n'
    '      mc: "20260807/PS93_x/motion_corrected"\n'
    '      h5: "20260807/PS93_x.h5"\n'
    "      regime: B\n"
    "      fmdir: null\n"
)


def test_insert_appends_to_block_and_parses():
    out = aw.insert_session_entry(SAMPLE, "PS92", "0809",
                                  "20260809/PS92_y/motion_corrected", "20260809/PS92_y.h5", "B", None)
    d = yaml.safe_load(out)
    # new date present under PS92, PS93 untouched, MMDD keys stay strings
    assert set(d["sessions"]["PS92"]) == {"0807", "0809"}
    assert set(d["sessions"]["PS93"]) == {"0807"}
    assert all(isinstance(k, str) for k in d["sessions"]["PS92"])
    e = d["sessions"]["PS92"]["0809"]
    assert e == {"mc": "20260809/PS92_y/motion_corrected", "h5": "20260809/PS92_y.h5",
                 "regime": "B", "fmdir": None}
    # header comments preserved
    assert out.startswith("# header comment line one\n")


def test_insert_into_last_animal_block_at_eof():
    out = aw.insert_session_entry(SAMPLE, "PS93", "0809",
                                  "20260809/PS93_y/motion_corrected", "20260809/PS93_y.h5", "A", None)
    d = yaml.safe_load(out)
    assert set(d["sessions"]["PS93"]) == {"0807", "0809"}
    assert d["sessions"]["PS93"]["0809"]["regime"] == "A"


def test_insert_is_idempotent():
    once = aw.insert_session_entry(SAMPLE, "PS92", "0809", "m", "h", "B", None)
    twice = aw.insert_session_entry(once, "PS92", "0809", "m", "h", "B", None)
    assert once == twice


def test_insert_fmdir_nonnull_quoted():
    out = aw.insert_session_entry(SAMPLE, "PS92", "0809", "m", "h", "B", "20260809/PS92_y/rescue")
    assert yaml.safe_load(out)["sessions"]["PS92"]["0809"]["fmdir"] == "20260809/PS92_y/rescue"


def test_insert_on_real_sessions_yaml_round_trips():
    text = aw.SESSIONS_YAML.read_text(encoding="utf-8")
    for a in ["PS92", "PS93", "PS94", "PS95"]:
        text = aw.insert_session_entry(text, a, "0899",
                                       f"20260899/{a}_z/motion_corrected", f"20260899/{a}_z.h5", "B", None)
    d = yaml.safe_load(text)                      # must still parse
    for a in ["PS92", "PS93", "PS94", "PS95"]:
        assert "0899" in d["sessions"][a]
        assert all(isinstance(k, str) for k in d["sessions"][a])   # MMDD keys stay quoted strings
    assert text.startswith("# Widefield imaging sessions")          # header comment preserved


class _FakeRV:
    def __init__(self, labcams, daq):
        self._m = {"labcams": labcams, "daq_recorder_output": daq}

    def resolve(self, root, rel):
        return str(self._m[root] / rel)


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_discover_ready_session(tmp_path, monkeypatch):
    monkeypatch.setattr(aw.config, "load_sessions", lambda *a, **k: [])   # nothing registered
    lab, daq = tmp_path / "labcams", tmp_path / "daq"
    sd = lab / "20260809" / "PS92_20260809_120000" / "motion_corrected"
    res = sd / "wfield_local_results"
    _touch(Path(aw.config.svtcorr_path(sd)))          # the VARIANT SVTcorr, not the bare one
    _touch(res / str(aw.config.defaults()["locanmf"]["allen_dir_name"]) / "U_atlas.npy")
    _touch(sd / "pco_daq_led_cleanpairs_frame_map.npz")               # regime B marker
    _touch(daq / "20260809" / "PS92_20260809_120500.h5")

    got = aw.discover(_FakeRV(lab, daq), "20260809", ["PS92", "PS93"])
    assert len(got) == 1 and got[0]["animal"] == "PS92"
    e = got[0]
    assert e["regime"] == "B" and not e["locanmf_done"] and not e["registered"]
    # the fit input and the output dir must both come from config, so the decomposition the poller
    # writes is the one every consumer reads (2026-08-20: they did not, and a whole night's 8/19
    # figures failed with FileNotFoundError against locanmf_affine8v1_final)
    assert Path(e["svt"]) == Path(aw.config.svtcorr_path(sd))
    assert Path(e["locanmf_out"]) == Path(aw.config.locanmf_dir(sd))
    assert e["mc_rel"] == "20260809/PS92_20260809_120000/motion_corrected"
    assert e["h5_rel"] == "20260809/PS92_20260809_120500.h5"


def test_discover_skips_when_inputs_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(aw.config, "load_sessions", lambda *a, **k: [])
    lab, daq = tmp_path / "labcams", tmp_path / "daq"
    sd = lab / "20260809" / "PS92_20260809_120000" / "motion_corrected"
    _touch(Path(aw.config.svtcorr_path(sd)))                          # U_atlas.npy absent -> not ready
    got = aw.discover(_FakeRV(lab, daq), "20260809", ["PS92"])
    assert got == []


def test_discover_ignores_bare_svtcorr_when_variant_configured(tmp_path, monkeypatch):
    """A bare wfield_local_results/SVTcorr.npy must NOT count as inputs-ready.

    That file is the superseded zerophase product. Treating it as ready is what made the poller fit
    LocaNMF to the wrong data on 2026-08-19 and write it to a directory no consumer reads, so every
    downstream figure raised FileNotFoundError.

    The fitted SVT path in the run's summary.json is the evidence, NOT the component count: that
    count swings 94-196 across days for one animal on the SAME input, so it cannot distinguish the
    two variants and must not be used as a check.
    """
    monkeypatch.setattr(aw.config, "load_sessions", lambda *a, **k: [])
    if aw.config.hemo_variant() is None:
        pytest.skip("no hemo variant configured -- the bare SVTcorr IS the canonical input")
    lab, daq = tmp_path / "labcams", tmp_path / "daq"
    sd = lab / "20260809" / "PS92_20260809_120000" / "motion_corrected"
    res = sd / "wfield_local_results"
    _touch(res / "SVTcorr.npy")                                       # bare file only
    _touch(res / str(aw.config.defaults()["locanmf"]["allen_dir_name"]) / "U_atlas.npy")
    _touch(daq / "20260809" / "PS92_20260809_120500.h5")
    assert aw.discover(_FakeRV(lab, daq), "20260809", ["PS92"]) == []
