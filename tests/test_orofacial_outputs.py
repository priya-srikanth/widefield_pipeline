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


def test_onset_context_by_onset():
    # the PRE-11-04 label (kept for the encoder's lick lists): pre_cue = strobe <= onset < cue; post_cue = cue <= onset
    # <= stop; iti = before the strobe / after the stop
    on = [-9000.0, -100.0, 0.0, 3500.0, 3600.0, -100.0, 50.0]
    strobe = [-7000.0, -7000.0, -7000.0, -7000.0, -7000.0, np.nan, -7000.0]
    stop = [3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, np.nan]
    assert oo.onset_context(on, strobe, stop).tolist() == ["iti", "pre_cue", "post_cue", "post_cue", "iti", "pre_cue",
                                                           "post_cue"]


def test_lick_context_by_peak_with_the_0_70_exclusion():
    # 11-04: by PEAK; pre_cue = strobe <= peak < cue; excluded_0_70 = cue <= peak < cue + 70 ms; post_cue = cue + 70 ms
    # <= peak <= stop; iti otherwise
    pk = [-9000.0, -0.1, 0.0, 69.9, 70.0, 3500.0, 3600.0, -100.0, 50.0, 80.0]
    strobe = [-7000.0] * 7 + [np.nan, -7000.0, -7000.0]
    stop = [3500.0] * 9 + [np.nan]
    assert oo.lick_context(pk, strobe, stop).tolist() == [
        "iti", "pre_cue", "excluded_0_70", "excluded_0_70", "post_cue", "post_cue", "iti", "pre_cue", "excluded_0_70",
        "post_cue"]
    assert oo.CONTEXTS == ("pre_cue", "excluded_0_70", "post_cue", "iti")


def test_onset_and_peak_labels_differ_only_around_the_cue():
    # a camera lick with onset 62 ms / peak 90 ms is a real response: post_cue under BOTH labels (onset rule not used);
    # onset -20 / peak 30: pre_cue by onset (the encoder list keeps it), excluded by peak
    on, pk = np.array([62.0, -20.0, -500.0]), np.array([90.0, 30.0, -450.0])
    sb, sp = np.full(3, -5000.0), np.full(3, 3500.0)
    assert oo.lick_context(pk, sb, sp).tolist() == ["post_cue", "excluded_0_70", "pre_cue"]
    assert oo.onset_context(on, sb, sp).tolist() == ["post_cue", "pre_cue", "pre_cue"]


def test_encoder_context_prefers_the_onset_label():
    new = pd.DataFrame({"context": ["excluded_0_70", "pre_cue"], "onset_context": ["pre_cue", "pre_cue"]})
    old = pd.DataFrame({"context": ["pre_cue", "pre_cue"]})                  # a pre-11-04 licks_all.csv
    assert oo.encoder_context(new).tolist() == ["pre_cue", "pre_cue"] == oo.encoder_context(old).tolist()


def test_engagement_keep_attempted_gates_read_trial_attempts(monkeypatch):
    # encoding window via Priya 10-20 late: the attempted gate reads trial_attempts.csv instead of re-deriving it
    from wfield_local import orofacial_cohort as oc
    T = pd.DataFrame({"label": ["PS92_0811"] * 4, "trial_id": [1, 2, 3, 4], "attempted": [True, True, False, True],
                      "canonical": [True, False, True, True]})
    monkeypatch.setitem(oc._ATTEMPTS, "T", T)
    monkeypatch.setattr(oc, "ENGAGED_ONLY", True)
    keep, lab = oc.engagement_keep("PS92", "20260811", None, gate="attempted")
    assert keep == {1, 2, 4} and lab.startswith("attempted")
    keep, _ = oc.engagement_keep("PS92", "20260811", None, gate="canonical_attempted")
    assert keep == {1, 4}
    assert oc.engagement_keep("PS93", "20260811", None, gate="attempted")[0] is None
    with pytest.raises(ValueError):
        oc.engagement_keep("PS92", "20260811", None, gate="nope")
