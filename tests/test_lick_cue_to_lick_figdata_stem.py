"""lick_cue_to_lick figdata stems keep tau and the rest-variant suffix (11-18: "_tau0.6_restnlc1" was cut to "_tau0")."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import figdata as _fd  # noqa: E402
import lick_cue_to_lick as c2l  # noqa: E402


def test_tau_and_variant_survive_fd_path(monkeypatch, tmp_path):
    stem = "lick_cue_to_lick_B_pre_models_tau0.6"
    paths = {}
    for sfx in ("", "_restnl", "_restnlc1"):
        monkeypatch.setattr(c2l, "SFX", sfx)
        p = c2l.save_fd(stem, {"sfx": sfx}, tmp_path)
        assert p == _fd.fd_path(c2l.fd_stem(stem), tmp_path)
        assert p.name == f"lick_cue_to_lick_B_pre_models_tau0p6{sfx}.pkl"
        paths[sfx] = p
    assert len(set(paths.values())) == 3                     # no two variants share a file
    for sfx in paths:
        monkeypatch.setattr(c2l, "SFX", sfx)
        assert c2l.load_fd(stem, tmp_path) == {"sfx": sfx}


def test_dotless_stems_unchanged(monkeypatch):
    monkeypatch.setattr(c2l, "SFX", "_restnl")
    assert c2l.fd_stem("lick_cue_to_lick_B_farcontra_on_pre") == "lick_cue_to_lick_B_farcontra_on_pre_restnl"
