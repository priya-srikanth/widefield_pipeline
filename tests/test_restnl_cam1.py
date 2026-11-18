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


def _events_all(path):
    """cam1 events_all: a cam1-only lick at 12.0 s (ITI context), a cam4-matched one at 30.5 s (the same time as a cam4
    lick) and a cam1-only water droplet at 54.0 s."""
    pd.DataFrame({"daq_on_s": [12.0, 30.5, 54.0], "cam4_matched": [False, True, False],
                  "droplet": [False, False, True], "spread_px": [9.0, 9.0, 1.0], "range_px": [40.0, 40.0, 2.0],
                  "med_out_px": [30.0, 30.0, 60.0]}).to_csv(path, index=False)
    return path


def test_cam1_only_onsets_drop_droplets_and_matched(tmp_path):
    from wfield_local import restnl_licks as rl
    E = pd.read_csv(_events_all(tmp_path / "e.csv"))
    assert rl.cam1_only_onsets(E).tolist() == [12.0]    # droplet and cam4-matched events are not cam1-only licks
    E2 = E.drop(columns="droplet")                       # no droplet column -> cam1_licks.droplet_flags decides
    from wfield_local.cam1_licks import droplet_flags
    exp = np.sort(E2.daq_on_s.to_numpy()[~E2.cam4_matched.to_numpy() & ~droplet_flags(E2, ~E2.cam4_matched.to_numpy())])
    assert rl.cam1_only_onsets(E2).tolist() == exp.tolist()


def test_union_counts_cam1_only_once():
    from wfield_local import restnl_licks as rl
    cam4 = np.array([5.0, 30.5, 40.0])
    c1 = np.array([12.0, 30.5])                        # a cam1 onset at a cam4 time is not a second lick
    u, n = rl.union_onsets(cam4, c1)
    assert n == 1 and u.tolist() == [5.0, 12.0, 30.5, 40.0]


def _stub_session(monkeypatch, tmp_path, cam1_exists=True):
    from wfield_local import analysis_kit as ak
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
    c1 = tmp_path / "PS99_20260801_events_all.csv"
    if cam1_exists:
        _events_all(c1)
    monkeypatch.setattr(rl, "cam1_events_path", lambda label: c1)
    return {"label": "PS99_0801"}


def test_cam1_mask_is_superset_and_droplet_stays_rest(tmp_path, monkeypatch):
    from wfield_local import restnl_licks as rl
    from wfield_local import restnl_mask as rnm
    s = _stub_session(monkeypatch, tmp_path)
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
    s = _stub_session(monkeypatch, tmp_path, cam1_exists=False)
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
    assert d1.name.endswith("_" + rl.CACHE_TAG) and s1 == f"PS92_0810__{rl.CACHE_TAG}" and d1 != d4


def test_bad_lick_set_refused(monkeypatch):
    from wfield_local import restnl_licks as rl
    monkeypatch.setenv(rl.ENV, "cam1")
    with pytest.raises(ValueError):
        rl.lick_set()
