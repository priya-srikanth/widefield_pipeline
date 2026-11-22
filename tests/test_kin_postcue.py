"""Four-camera licks everywhere (11-22): the post-cue view of the fused kinematics and the lick-set switch."""
import numpy as np
import pandas as pd
import pytest


def _fused():
    return pd.DataFrame({
        "session": ["S1"] * 6, "trial_id": [1, 1, 1, 1, 2, 2], "lick_idx": [0, 1, 2, 3, 0, 1],
        "context": ["pre_cue", "post_cue", "post_cue", "iti", "post_cue", "post_cue"],
        "t_peak_cam4": [-300, 200, 330, 5000, 150, 290], "lick_key": ["1:-300.0", "1:200.0", "1:330.0", "1:5000.0",
                                                                     "2:150.0", "2:290.0"],
        "n_views": [2, 1, 3, 2, 2, 2], "cams": ["cam1+cam4", "cam4", "cam1+cam2+cam4", "cam1+cam4", "cam1+cam4",
                                               "cam1+cam4"], "n_3d": [10, 0, 12, 9, 8, 7], "cam4_matched": [True, False, True, True, True, True],
        "reach_mm": [4.0, 1.8, 5.0, 4.4, 4.5, 4.6], "speed_peak": [150.0, 90.0, 180.0, 160.0, 170.0, 175.0],
        "t_peak_3d": [-300.0, 196.0, 330.0, 5000.0, 150.0, 290.0]})


def test_postcue_renumbers_within_trial_and_joins_spans():
    from scripts.pose3d_kin_postcue import postcue_licks
    spans = pd.DataFrame({"session": "S1", "lick_key": ["1:200.0", "1:330.0", "2:150.0", "2:290.0", "1:-300.0"],
                          "on_ms": [180.0, 300.0, 120.0, 260.0, -330.0], "off_ms": [230.0, 380.0, 200.0, 340.0, -250.0]})
    P = postcue_licks(_fused(), spans)
    assert list(P.context.unique()) == ["post_cue"]
    assert P.lick_idx.tolist() == [0, 1, 0, 1]
    assert P.lick_idx_fused.tolist() == [1, 2, 0, 1]
    assert P.on_ms.tolist() == [180.0, 300.0, 120.0, 260.0]


def test_single_view_rows_kept_with_nan_kinematics():
    from scripts.pose3d_kin_postcue import mask_single_view, postcue_licks
    P = mask_single_view(postcue_licks(_fused()))
    assert len(P) == 4                                    # counts unchanged
    first = P[(P.trial_id == 1) & (P.lick_idx == 0)].iloc[0]
    assert not first.kin_ok and np.isnan(first.reach_mm) and np.isnan(first.t_peak_3d)   # not promoted, just blank
    assert P.kin_ok.sum() == 3
    assert mask_single_view(postcue_licks(_fused()), 0).reach_mm.notna().all()


def test_kin_name_switch(monkeypatch):
    from wfield_local import session_licks as SL
    monkeypatch.setattr(SL, "fused_on", lambda: True)
    monkeypatch.delenv(SL.KIN_ENV, raising=False)
    assert SL.kin_name() == SL.KIN_POSTCUE and SL.kin_sfx() == "_fused"
    monkeypatch.setenv(SL.KIN_ENV, "cam4")
    assert SL.kin_name() == "kinematics" and SL.kin_sfx() == ""
    monkeypatch.setenv(SL.KIN_ENV, "bogus")
    with pytest.raises(ValueError):
        SL.kin_name()
    monkeypatch.delenv(SL.KIN_ENV)
    monkeypatch.setattr(SL, "fused_on", lambda: False)
    assert SL.kin_name() == "kinematics"


def test_summary_counts_both_ways():
    from scripts.c2_lick3d_session_summary import summarise
    from scripts.pose3d_kin_postcue import mask_single_view, postcue_licks
    P = mask_single_view(postcue_licks(_fused())).assign(position="far_R", side_ml=0.0, animal="PS92", date="20260810",
                                                         epoch="pre")
    P["session"] = "PS92_20260810"
    S = summarise(P).iloc[0]
    assert (S.n_first, S.n_first_rows, S.n_all, S.n_all_rows) == (1, 2, 3, 4)


def test_cam1_only_events_are_not_licks():
    from scripts.pose3d_kin_postcue import postcue_licks
    L = _fused()
    L.loc[1, "cams"] = "cam1"                             # trial 1's first post-cue event: cam1 only
    P = postcue_licks(L)
    t1 = P[P.trial_id == 1]
    assert len(t1) == 1 and t1.lick_idx.tolist() == [0] and t1.lick_key.tolist() == ["1:330.0"]   # next real lick = first
    assert len(postcue_licks(L, drop_cam1_only=False)) == 4
