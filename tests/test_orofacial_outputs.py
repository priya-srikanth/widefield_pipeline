"""orofacial_outputs: the layout, the run's params-digest guard, and that windows.npz text columns load without
pickle."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wfield_local import orofacial_outputs as oo


class _RV:
    def __init__(self, r):
        self.r = r

    def root(self, name):
        return str(self.r)


def test_run_name_and_session_layout(tmp_path):
    assert oo.run_name("LP_r4_heatmap_20261013") == f"lp_r4_heatmap_20261013__clean_{oo.CLEANING_VERSION}"
    d = oo.session_dir(_RV(tmp_path), "runA", "PS93", "20260814")
    assert d == tmp_path / "runA" / "sessions" / "PS93" / "20260814"


def test_a_run_refuses_sessions_made_under_different_params(tmp_path, monkeypatch):
    monkeypatch.setattr(oo, "params_digest", lambda: "aaa")
    monkeypatch.setattr("wfield_local.writeguard.assert_writable", lambda p: None)
    oo.ensure_run(_RV(tmp_path), "runA", {"model_scorer": "x"})
    assert json.loads((tmp_path / "runA" / "run.json").read_text())["params_digest"] == "aaa"
    oo.ensure_run(_RV(tmp_path), "runA", {})                     # same params: fine
    monkeypatch.setattr(oo, "params_digest", lambda: "bbb")
    with pytest.raises(SystemExit, match="bump CLEANING_VERSION"):
        oo.ensure_run(_RV(tmp_path), "runA", {})


def test_windows_text_columns_round_trip_without_pickle(tmp_path):
    ix = pd.DataFrame({"trial_k": [0, 0, 1], "position": ["far_L", "far_L", "close_R"], "t_ms": [0.0, 4.0, 0.0]})
    np.savez_compressed(tmp_path / "w.npz", **{f"ix_{c}": (ix[c].to_numpy() if pd.api.types.is_numeric_dtype(ix[c])
                                                            else np.asarray(ix[c].astype(str), dtype=str))
                                                         for c in ix.columns})
    w = np.load(tmp_path / "w.npz", allow_pickle=False)
    back = pd.DataFrame({k[3:]: w[k] for k in w.files})
    assert list(back.position) == ["far_L", "far_L", "close_R"] and back.trial_k.tolist() == [0, 0, 1]
