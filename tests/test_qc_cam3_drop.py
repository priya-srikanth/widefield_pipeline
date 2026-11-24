"""Forced side-camera exclusion (`pose3d_side.SIDE_OFF_ENV`, opt-in, default unchanged) and the cam3-drop QC compare
(`scripts/qc_cam3_drop.py`, Priya 11-22: "does leaving cam3 out change conclusions?")."""
import numpy as np
import pandas as pd
import pytest

from wfield_local import pose3d_side as PS


def test_forced_off_default_is_empty_and_validates(monkeypatch):
    monkeypatch.delenv(PS.SIDE_OFF_ENV, raising=False)
    assert PS.forced_off() == ()
    monkeypatch.setenv(PS.SIDE_OFF_ENV, "cam3")
    assert PS.forced_off() == ("cam3",)
    assert PS.forced_off("cam2, cam3") == ("cam2", "cam3")
    with pytest.raises(ValueError):
        PS.forced_off("cam4")


def test_apply_forced_off_fails_only_the_named_camera_and_keeps_numbers():
    gate = {"cam2": {"rms600": 0.6, "max600": 1.0, "pass": True}, "cam3": {"rms600": 1.1, "max600": 1.66, "pass": True}}
    g = PS.apply_forced_off(gate, ("cam3",))
    assert g["cam2"] == gate["cam2"] and not g["cam3"]["pass"] and g["cam3"]["forced_off"]
    assert g["cam3"]["rms600"] == 1.1 and gate["cam3"]["pass"]               # no mutation
    assert PS.apply_forced_off(gate, ()) == gate


def test_pose3d_forced_off_needs_out_and_skips_side_events():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "pose3d.py").read_text(encoding="utf-8")
    assert "need --out" in src and 'four_cam != "clips" and not forced' in src


def _licks(session, vals):
    rows = []
    for i, (pos, idx, side, out, down, reach, kin) in enumerate(vals):
        rows.append({"session": session, "lick_key": f"{i}:0.0", "position": pos, "lick_idx": idx, "kin_ok": kin,
                     "reach_mm": reach, "speed_peak": 100.0 + i, "side_ml": side, "out_ap": out, "down_dv": down})
    return pd.DataFrame(rows)


def test_aim_and_compare_tables_match_by_lick_key():
    import scripts.qc_cam3_drop as Q
    assert np.isclose(Q.aim_deg(1.0, 0.0, 1.0), 45.0) and np.isclose(Q.aim_deg(-1.0, 3.0, 4.0), np.degrees(np.arctan2(-1, 5)))
    prod = _licks("PS95_20260819", [("far_R", 0, 1.0, 0.0, 1.0, 5.0, True), ("far_R", 1, 0.0, 0.0, 1.0, 6.0, True)])
    alt = prod.copy()
    alt.loc[0, "reach_mm"] = 5.5
    alt.loc[1, "kin_ok"] = False
    alt = alt.iloc[::-1].reset_index(drop=True)                               # order must not matter
    pre = pd.DataFrame({"animal": ["PS95"] * 2, "position": ["far_R"] * 2, "licks": ["first", "all"],
                        "reach_mm": [4.0, 4.0], "speed_peak": [100.0, 100.0], "aim_deg": [0.0, 0.0]})
    T = Q.compare_tables(prod, alt, {"PS95_20260819": "subacute"}, pre)
    f = T[T.licks == "first"].iloc[0]
    assert np.isclose(f.d_reach_mm, 0.5) and np.isclose(f.eff_reach_mm, 1.0) and np.isclose(f.ratio_reach_mm, 0.5)
    a = T[T.licks == "all"].iloc[0]
    assert np.isclose(a.frac_kin_prod, 1.0) and np.isclose(a.d_frac_kin, -0.5)
    assert np.isclose(a.reach_mm_prod, 5.5) and np.isclose(a.reach_mm_alt, 5.5)


def test_forced_on_and_pnp_exclude_parse_and_validate(monkeypatch):
    for k in (PS.SIDE_ON_ENV, PS.PNP_EXCLUDE_ENV):
        monkeypatch.delenv(k, raising=False)
    assert PS.forced_on() == () and PS.pnp_exclude() == {}
    monkeypatch.setenv(PS.SIDE_ON_ENV, "cam3")
    assert PS.forced_on() == ("cam3",)
    with pytest.raises(ValueError):
        PS.forced_on("cam1")
    monkeypatch.setenv(PS.PNP_EXCLUDE_ENV, "cam3:close_center,close_R")
    assert PS.pnp_exclude() == {"cam3": ("close_center", "close_R")}
    assert PS.pnp_exclude("cam2:far_L; cam3:close_R") == {"cam2": ("far_L",), "cam3": ("close_R",)}
    for bad in ("cam4:far_L", "cam3:far_X", "cam3:", "cam3:close_center,close_R,close_L"):
        with pytest.raises(ValueError):
            PS.pnp_exclude(bad)


def test_apply_forced_on_passes_only_the_named_camera_and_keeps_numbers():
    gate = {"cam2": {"rms600": 0.6, "max600": 1.0, "pass": True}, "cam3": {"rms600": 1.56, "max600": 2.33, "pass": False}}
    g = PS.apply_forced_on(gate, ("cam3",))
    assert g["cam2"] == gate["cam2"] and g["cam3"]["pass"] and g["cam3"]["forced_on"]
    assert g["cam3"]["rms600"] == 1.56 and not gate["cam3"]["pass"]           # no mutation; the real gate is kept
    assert PS.apply_forced_on(gate, ()) == gate


class _Cam:
    """Just enough of aniposelib.Camera for `pose_subset`: pinhole K, no distortion, pose via Rodrigues."""

    def __init__(self, name, rvec, tvec):
        self.name, self.r, self.t = name, np.asarray(rvec, float), np.asarray(tvec, float)
        self.K = np.array([[5500.0, 0, 300], [0, 5500.0, 300], [0, 0, 1]])

    def get_name(self):
        return self.name

    def get_camera_matrix(self):
        return self.K

    def get_distortions(self):
        return np.zeros(5)

    def get_rotation(self):
        return self.r

    def get_translation(self):
        return self.t

    def set_rotation(self, r):
        self.r = np.asarray(r, float)

    def set_translation(self, t):
        self.t = np.asarray(t, float)

    def project(self, X):
        import cv2
        return cv2.projectPoints(np.asarray(X, float), self.r, self.t, self.K, np.zeros(5))[0][:, 0]


class _Group:
    def __init__(self, cams):
        self.cameras = cams


def test_pose_subset_ignores_biased_spouts_and_reports_all_six():
    from wfield_local import spout_world as SW
    geom = {"dist_close": "2", "dist_far": "4", "az_center": "0.0", "az_left": "-60", "az_right": "60",
            "down_angle": "45", "head_roll": "0"}
    true = _Cam("cam3", [0.1, -1.4, 0.05], [0.5, 0.8, 150.0])
    obj = np.array([SW.commanded_world(geom, p) for p in SW.POSITIONS])
    px = true.project(obj)
    med = {p: tuple(px[k]) for k, p in enumerate(SW.POSITIONS)}
    med["close_center"] = (med["close_center"][0] - 2.0, med["close_center"][1])   # chin-fur bias on two spouts
    med["close_R"] = (med["close_R"][0] + 2.0, med["close_R"][1])
    start = _Group([_Cam("cam3", [0.12, -1.38, 0.04], [0.6, 0.7, 151.0])])          # near the truth (six-spout pose)
    g, err = PS.pose_subset(start, "cam3", med, geom, ("close_center", "close_R"))
    c = g.cameras[0]
    assert np.allclose(c.r, true.r, atol=1e-4) and np.allclose(c.t, true.t, atol=1e-2)
    assert set(err) == set(SW.POSITIONS)
    assert max(err[p] for p in ("far_center", "far_L", "far_R", "close_L")) < 1e-3
    assert np.isclose(err["close_center"], 2.0, atol=1e-2) and np.isclose(err["close_R"], 2.0, atol=1e-2)
    assert start.cameras[0].t[2] == 151.0                                           # the input group is not modified
    with pytest.raises(ValueError):
        PS.pose_subset(start, "cam3", med, geom, ("close_center", "close_R", "close_L"))


def test_tongue_jitter_is_zero_on_straight_motion_and_sees_a_kink():
    import scripts.qc_cam3_drop as Q
    t = np.arange(10) * 4.0
    F = pd.DataFrame({"trial_id": 1, "t_ms": t, "tongue_ml": 0.1 * t, "tongue_ap": 0.0, "tongue_dv": -0.05 * t,
                      "tongue_views": 2})
    P = pd.DataFrame({"trial_id": [1, 1], "lick_idx": [0, 1], "kin_ok": [True, True], "on_ms": [0.0, 12.0],
                      "off_ms": [8.0, 36.0]})
    j = Q.tongue_jitter(F, P)
    assert j["n"] == 6 and np.isclose(j["med"], 0.0)                 # centers 12..32 ms (lick 1 only; 36 has no t+1)
    F.loc[5, "tongue_ap"] = 0.3                                       # one frame off by 0.3 mm
    F.loc[8, "tongue_views"] = -1                                     # a one-view fill breaks its triplets
    j = Q.tongue_jitter(F, P)
    assert j["n"] == 4 and np.isclose(j["med"], 0.3) and np.isclose(j["p95"], 0.555, atol=0.01)


def test_splice_text_replaces_only_the_named_sessions_in_place():
    from scripts.swap_cam3_on_20261123 import splice_text
    old = pd.DataFrame({"session": ["A", "B", "B", "C"], "x": ["1.5", "2", "3", "", ], "y": ["a", "b", "c", "d"]})
    new = pd.DataFrame({"session": ["B", "B", "B", "Z"], "x": [7.0, 8.0, 9.0, 0.0], "extra": [1, 2, 3, 4]})
    S = splice_text(old, new, ["B"])
    assert list(S.columns) == ["session", "x", "y"]
    assert S.session.tolist() == ["A", "B", "B", "B", "C"]
    assert S.x.tolist() == ["1.5", "7.0", "8.0", "9.0", ""] and S.y.tolist() == ["a", "", "", "", "d"]
    S2 = splice_text(old, new, ["Q"])                                   # nothing to replace: unchanged
    assert S2.equals(old.reset_index(drop=True))


def test_splice_measures_keeps_order_numbering_and_unmatched_rows():
    from scripts.swap_cam3_on_20261123 import splice_measures
    old = pd.DataFrame({"session": ["A", "B", "B", "B"], "lick_key": ["1", "1", "2", "3"], "lick_idx": ["0", "0", "1", "2"],
                        "lick_idx_fused": ["5", "6", "7", "8"], "reach_mm": ["4", "4", "4", "4"]})
    new = pd.DataFrame({"session": ["B", "B", "A"], "lick_key": ["3", "1", "1"], "lick_idx": [9, 9, 9],
                        "lick_idx_fused": [0, 0, 0], "reach_mm": [5.5, 6.5, 7.5]})
    S = splice_measures(old, new, ["B"])
    assert S.reach_mm.tolist() == ["4", "6.5", "4", "5.5"]                 # B/2 unmatched keeps its old row; A untouched
    assert S.lick_idx.tolist() == old.lick_idx.tolist() and S.lick_idx_fused.tolist() == old.lick_idx_fused.tolist()
