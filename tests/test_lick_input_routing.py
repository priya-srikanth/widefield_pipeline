"""11-21 four-camera re-run: `lp_licks` input routing (cam4 run + versioned cam1 event folder) is opt-in, byte-identical
by default, and enters the LP cache token when set."""
from pathlib import Path

import pytest

from wfield_local import lp_licks as L


class _RV:
    def __init__(self, root):
        self._r = Path(root)

    def root(self, name):
        return self._r / name


def test_default_routing_is_unchanged(monkeypatch, tmp_path):
    from wfield_local import orofacial_cohort as oc
    monkeypatch.delenv(L.INPUT_RUN_ENV, raising=False)
    monkeypatch.delenv(L.CAM1_DIR_ENV, raising=False)
    assert L.input_run() == oc.RUN
    assert L.cam1_events_dir(_RV(tmp_path)) == tmp_path / "dlc" / "session_poses" / "cam1_licks"
    assert L._routing_token() == ""


def test_routing_env_moves_inputs_and_token(monkeypatch, tmp_path):
    monkeypatch.setenv(L.INPUT_RUN_ENV, "lp_r4_heatmap_20261013__clean_v37")
    monkeypatch.setenv(L.CAM1_DIR_ENV, "cam1_licks_v37")
    rv = _RV(tmp_path)
    sd, f4, f1 = L._inputs("PS93_0814", rv)
    assert "clean_v37" in str(sd) and f4.name == "licks_all.csv"
    assert f1 == tmp_path / "dlc" / "session_poses" / "cam1_licks_v37" / "PS93_20260814_events.csv"
    assert "clean_v37" in L._routing_token() and "cam1_licks_v37" in L._routing_token()


def test_cam1_dir_must_be_a_cam1_licks_variant(monkeypatch, tmp_path):
    monkeypatch.setenv(L.CAM1_DIR_ENV, "../elsewhere")
    with pytest.raises(ValueError):
        L.cam1_events_dir(_RV(tmp_path))
