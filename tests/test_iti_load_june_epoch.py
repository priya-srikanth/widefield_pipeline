"""iti_licks._load reads June reference files whose epoch was saved as None (an object array), and nothing else.

Before 11-08 the extractor wrote ``epoch=epochs.epoch_of(label)``, which is None for a session before WIDEFIELD_PRE_FROM;
numpy saves that as an object array and the strict loader refused all 12 June files (l_reembed, 11-07).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def il(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import iti_licks
    return iti_licks


def test_none_epoch_reads_as_empty_string(il, tmp_path):
    np.savez_compressed(tmp_path / "PSX_0606.npz", epoch=None, animal="PSX", first_rt=np.arange(3.0))
    d = il._load("PSX_0606", tmp_path)
    assert d["epoch"].dtype.kind == "U" and str(d["epoch"]) == ""
    assert str(d["animal"]) == "PSX" and np.array_equal(d["first_rt"], np.arange(3.0))


def test_plain_file_unchanged(il, tmp_path):
    np.savez_compressed(tmp_path / "PSX_0810.npz", epoch="pre", animal="PSX")
    assert str(il._load("PSX_0810", tmp_path)["epoch"]) == "pre"


def test_any_other_object_array_is_still_refused(il, tmp_path):
    np.savez_compressed(tmp_path / "PSX_0607.npz", epoch="pre", cols=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError, match="only a None epoch"):
        il._load("PSX_0607", tmp_path)


def test_writer_never_saves_a_none_epoch():
    src = (REPO / "scripts" / "iti_licks.py").read_text(encoding="utf-8")
    assert "epoch=epochs.epoch_of(label) or \"\"" in src
    assert "epoch=epochs.epoch_of(label)," not in src
    sys.modules.pop("iti_licks", None)
