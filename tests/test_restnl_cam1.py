"""RESTnlc1 (Priya 11-17): the RESTnl rest mask with the cam1-only LP licks also removed (`wfield_local.restnl_licks`).

(1) the cam4cam1 mask removes a SUPERSET of the cam4 mask's samples (same session, real `restnl_mask.lickfree_mask` with its
file reads stubbed); (2) water droplets are NOT licks (they stay in rest); (3) a cam4-matched cam1 event is not counted
twice; (4) the default (cam4) cache names are unchanged and the cam4cam1 ones differ; (5) a session without cam1 events
falls back to cam4 only and is recorded by name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

FS = 1000.0
N = 60_000                     # 60 s of DAQ samples


def _cue_table():
    return pd.DataFrame({"trial_id": [1, 2, 3], "position": ["far_R"] * 3,
                         "cue_daq": [10.0, 30.0, 50.0]}).set_index("trial_id")


def _events(tmp_path):
    """cam1 events: a cam1-only lick (trial 1, +2.0 s -> 12.0 s), a cam4-matched one (trial 2, +0.5 s -> 30.5 s, the same
    time as a cam4 lick) and a water droplet (trial 3, +4.0 s -> 54.0 s)."""
    E = pd.DataFrame({"trial_id": [1, 2, 3], "on_ms": [2000.0, 500.0, 4000.0],
                      "cam4_matched": [False, True, False], "droplet": [False, False, True]})
    f1 = tmp_path / "PS99_20260801_events.csv"
    E.to_csv(f1, index=False)
    return f1


def test_read_times_drops_droplets_and_matched(tmp_path, monkeypatch):
    from wfield_local import lp_licks
    from wfield_local import orofacial_cohort as oc
    f4 = tmp_path / "licks_all.csv"
    pd.DataFrame({"daq_on_s": [5.0, 30.5, 40.0]}).to_csv(f4, index=False)
    f1 = _events(tmp_path)
    monkeypatch.setattr(oc, "_cue_table", lambda sd: _cue_table().reset_index())
    t, _cue = lp_licks._read_times("PS99_0801", tmp_path, f4, f1)
    assert 12.0 in t                                   # the cam1-only lick is in the LP train
    assert not np.any(np.isclose(t, 54.0))             # the droplet is not a lick
    assert np.sum(np.isclose(t, 30.5)) == 1            # the cam4-matched cam1 event is not added a second time


def test_union_counts_cam1_only_once():
    from wfield_local import restnl_licks as rl
    cam4 = np.array([5.0, 30.5, 40.0])
    lp = np.array([5.0, 12.0, 30.5, 40.0])             # = what lp_lick_times returns (cam4 + cam1-only)
    u, n = rl.union_onsets(cam4, lp)
    assert n == 1 and u.tolist() == [5.0, 12.0, 30.5, 40.0]


def _stub_session(monkeypatch, tmp_path, lp_train, cam1_exists=True):
    from wfield_local import analysis_kit as ak
    from wfield_local import lp_licks
    from wfield_local import rest_by_position as rbp
    from wfield_local import restnl_licks as rl
    from wfield_local import restnl_mask as rnm
    rest0 = np.ones(N, bool)
    f4 = tmp_path / "licks_all.csv"
    pd.DataFrame({"daq_on_s": [5.0, 30.5, 40.0]}).to_csv(f4, index=False)
    monkeypatch.setattr(rbp, "_session_daq", lambda s: (rest0.copy(), None, None, None, None, None))
    monkeypatch.setattr(rbp, "_engaged_trials", lambda *a, **k: np.ones(3, bool))
    monkeypatch.setattr(ak, "daq_rate", lambda s: FS)
    monkeypatch.setattr(rnm, "daq_lick_onsets", lambda s, fs: np.array([20_000], np.int64))
    monkeypatch.setattr(rnm, "licks_all_path", lambda label: f4)
    monkeypatch.setattr(lp_licks, "lp_lick_times", lambda label, rv=None: (np.asarray(lp_train, float), None))
    c1 = tmp_path / "PS99_20260801_events.csv"
    if cam1_exists:
        c1.write_text("x\n")
    monkeypatch.setattr(rl, "cam1_events_path", lambda label: c1)
    return {"label": "PS99_0801"}


def test_cam1_mask_is_superset_and_droplet_stays_rest(tmp_path, monkeypatch):
    from wfield_local import restnl_licks as rl
    from wfield_local import restnl_mask as rnm
    # the LP train as lp_lick_times builds it from _events(): cam4 + the cam1-only 12.0 s (droplet 54.0 s already out)
    s = _stub_session(monkeypatch, tmp_path, [5.0, 12.0, 30.5, 40.0])
    with rl.active("cam4"):
        m4, _p, i4 = rnm.lickfree_mask(s)
    with rl.active("cam4cam1"):
        m1, _p, i1 = rnm.lickfree_mask(s)
    assert "n_cam1_only" not in i4                      # default info unchanged
    assert i1["n_cam1_only"] == 1 and i1["n_cam4_licks"] == 4
    assert not np.any(m1 & ~m4)                         # cam4cam1 keeps no sample the cam4 mask removed (superset)
    assert m4[12_000] and not m1[12_000]                # the cam1-only lick (and its buffer) leaves rest
    assert not m1[11_500] and not m1[14_000] and m1[11_400] and m1[14_100]   # [-0.5, +2.0] s widening
    assert m1[54_000] and m4[54_000]                    # the droplet time stays rest in both versions


def test_no_cam1_events_falls_back_to_cam4(tmp_path, monkeypatch):
    from wfield_local import restnl_licks as rl
    from wfield_local import restnl_mask as rnm
    s = _stub_session(monkeypatch, tmp_path, [5.0, 12.0, 30.5, 40.0], cam1_exists=False)
    with rl.active("cam4"):
        m4, _p, _i = rnm.lickfree_mask(s)
    with rl.active("cam4cam1"):
        m1, _p, i1 = rnm.lickfree_mask(s)
    assert np.array_equal(m4, m1) and i1["n_cam1_only"] == -1
    assert "PS99_0801" in rl.missing_cam1()


def test_cache_names_default_unchanged(monkeypatch):
    from wfield_local import restnl_licks as rl
    from wfield_local import restnl_mask as rnm
    from wfield_local import run_mode
    monkeypatch.setattr(run_mode, "cache_tag", lambda: "")
    monkeypatch.delenv(rl.ENV, raising=False)
    monkeypatch.delenv("WIDEFIELD_GATE_SOURCE", raising=False)
    d4, s4 = rnm.cache_dir(), rnm._stem("PS92_0810")
    assert d4.name.endswith("_restnl") and s4 == "PS92_0810__restnl"
    with rl.active("cam4cam1"):
        d1, s1 = rnm.cache_dir(), rnm._stem("PS92_0810")
    assert d1.name.endswith("_restnlc1") and s1 == "PS92_0810__restnlc1" and d1 != d4


def test_bad_lick_set_refused(monkeypatch):
    from wfield_local import restnl_licks as rl
    monkeypatch.setenv(rl.ENV, "cam1")
    with pytest.raises(ValueError):
        rl.lick_set()
