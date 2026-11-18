"""THE cam1 lick rule, `cam1_licks.keep_mask` (Priya 11-17): one definition of "this cam1 event counts as a lick"
(not a water droplet, not a far-from-mouth label), shared by every reader of the cam1 event tables.

(1) a droplet row is dropped, a far cam1-only row is dropped, a cam4-matched row is untouched (even when far);
(2) no droplet column -> `droplet_flags` decides; no med_out_px column -> no far guard, printed once per session;
(3) `restnl_licks.cam1_only_onsets` is unchanged by the refactor (frozen copy of the v3 body, EXACT equality) and its
    cache digest stays the v3 one (alias), while any other source would not map to it;
(4) `trial_classes.session_trials` drops a far cam1-only event from the incomplete set AND from the jaw-only "seen" set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local import cam1_licks as C1

FAR = C1.FAR_FROM_MOUTH_PX


def _table():
    # rows: 0 real cam1-only lick; 1 droplet; 2 far cam1-only; 3 cam4-matched AND far (a cam4 lick: kept);
    # 4 exactly at the threshold (not "<": kept); 5 cam1-only, med_out NaN (kept); 6 cam4-matched NaN flag = cam1-only
    return pd.DataFrame({
        "daq_on_s": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "cam4_matched": [False, False, False, True, False, False, np.nan],
        "droplet": [False, True, False, False, False, False, False],
        "spread_px": [9.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0],
        "range_px": [40.0, 2.0, 2.0, 40.0, 40.0, 40.0, 40.0],
        "med_out_px": [-50.0, 60.0, FAR - 1.0, FAR - 200.0, FAR, np.nan, FAR - 5.0]})


def test_keep_mask_rows():
    k = C1.keep_mask(_table(), label="T1")
    assert k.tolist() == [True, False, False, True, True, True, False]


def test_keep_mask_cam4_matched_argument_overrides_column():
    E = _table()
    m4 = np.array([False, False, True, True, False, False, False])     # row 2 now cam4-matched -> far guard off
    assert C1.keep_mask(E, m4).tolist() == [True, False, True, True, True, True, False]


def test_missing_droplet_column_uses_droplet_flags():
    E = _table().drop(columns="droplet")
    m4 = C1.flag_col(E.cam4_matched)
    exp = ~C1.droplet_flags(E, ~m4) & ~(~m4 & (E.med_out_px.to_numpy(float) < FAR))
    assert C1.keep_mask(E).tolist() == exp.tolist()
    assert not C1.keep_mask(E)[1]                                       # the static point away from the mouth


def test_missing_med_out_column_no_far_guard_printed_once(capsys):
    E = _table().drop(columns="med_out_px")
    C1._NO_FAR_COL.discard("PS99_0101")
    k1 = C1.keep_mask(E, label="PS99_0101")
    k2 = C1.keep_mask(E, label="PS99_0101")
    assert k1.tolist() == k2.tolist() == (~E.droplet.to_numpy()).tolist()   # droplets still out, far rows kept
    assert capsys.readouterr().out.count("PS99_0101: no med_out_px column") == 1


def test_flag_col_parses_strings_and_nan():
    assert C1.flag_col(pd.Series(["True", "false", np.nan, "TRUE", True, False])).tolist() == \
        [True, False, False, True, True, False]


def _v3_cam1_only_onsets(E) -> np.ndarray:
    """FROZEN copy of `restnl_licks.cam1_only_onsets` at 5a43b845 (RESTnlc1 v3), before the rule moved to keep_mask."""
    m4 = E["cam4_matched"].astype(str).str.lower().eq("true").to_numpy()
    if "droplet" in E:
        dr = E["droplet"].astype(str).str.lower().eq("true").to_numpy()
    else:
        from wfield_local.cam1_licks import droplet_flags
        dr = droplet_flags(E, ~m4)
    far = np.zeros(len(E), bool)
    if "med_out_px" in E:
        far = (pd.to_numeric(E["med_out_px"], errors="coerce") < -120.0).to_numpy()
    t = pd.to_numeric(E["daq_on_s"], errors="coerce").to_numpy(float)[~m4 & ~dr & ~far]
    return np.sort(t[np.isfinite(t)])


def test_restnl_cam1_only_onsets_unchanged_by_refactor(tmp_path):
    from wfield_local import restnl_licks as rl
    rng = np.random.default_rng(17)
    n = 400
    E = pd.DataFrame({"daq_on_s": np.round(rng.uniform(0, 3000, n), 3),
                      "cam4_matched": rng.choice([True, False, np.nan], n, p=[.3, .65, .05]),
                      "droplet": rng.choice([True, False], n, p=[.1, .9]),
                      "spread_px": rng.uniform(0, 8, n), "range_px": rng.uniform(0, 30, n),
                      "med_out_px": rng.uniform(-400, 120, n)})
    E.loc[rng.choice(n, 20, replace=False), "daq_on_s"] = np.nan
    E.loc[rng.choice(n, 20, replace=False), "med_out_px"] = np.nan
    E.to_csv(tmp_path / "e.csv", index=False)                         # through a CSV, as the real reader sees it
    E = pd.read_csv(tmp_path / "e.csv")
    for X in (E, E.drop(columns="droplet"), E.drop(columns="med_out_px")):
        assert np.array_equal(rl.cam1_only_onsets(X), _v3_cam1_only_onsets(X))
    assert rl.FAR_FROM_MOUTH_PX == FAR == -120.0


def test_restnl_digest_is_v3_alias_and_only_for_this_source(monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    from wfield_local import restnl_licks as rl
    # fresh interpreter: other tests in the suite may monkeypatch / reload modules the closure hashes
    root = Path(__file__).resolve().parents[1]
    d = subprocess.run([sys.executable, "-c", "from wfield_local import restnl_licks as r; print(r.code_digest())"],
                       cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    assert d == "e34a85c2b4b0"                                         # RESTnlc1 v3 caches stay warm
    # any other source (here: keep_mask's closure changed) misses the alias -> a new digest = invalidation
    from wfield_local import map_collect
    real = map_collect._Closure.source
    monkeypatch.setattr(map_collect._Closure, "source", lambda self: real(self) + "\n# changed\n")
    assert rl.code_digest() != "e34a85c2b4b0"


def test_session_trials_far_event_not_a_lick_nor_seen(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import test_trial_classes as TT
    from wfield_local import trial_classes as TC
    v36, c1 = TT._session(tmp_path)
    f = c1 / "PS92_20260818_events.csv"
    E = pd.read_csv(f)
    E["spread_px"], E["range_px"], E["med_out_px"] = 9.0, 40.0, -50.0
    base = TC.session_trials("PS92_0818", v36=v36, c1=c1).set_index("trial_id").cls
    assert base[3] == "incomplete" and base[4] == "incomplete"
    # trial 3: its cam1-only lick becomes a far label; trial 4: its jaw-only-trial "seen" row becomes far + cam1-only
    E.loc[E.trial_id == 3, "med_out_px"] = FAR - 10.0
    E.loc[E.trial_id == 4, ["med_out_px", "cam4_matched"]] = [FAR - 10.0, False]
    E.to_csv(f, index=False)
    T = TC.session_trials("PS92_0818", v36=v36, c1=c1).set_index("trial_id")
    assert T.cls[3] == "no_response" and T.n_licks[3] == 0              # cam4_moved 0 / cam1_moved 0 in CASES
    assert T.cls[4] == "jaw_only"                                       # jaw moved (1, 1), tongue no longer seen
    assert (T.cls.drop([3, 4]) == base.drop([3, 4])).all()
