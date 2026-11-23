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
    assert "needs --out" in src and 'four_cam != "clips" and not off' in src


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
