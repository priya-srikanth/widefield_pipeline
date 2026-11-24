"""scripts/pose3d_kinematics: the fused-train mode keeps every lick (NaN measures where < 3 3-D frames), carries
lick_key, and the frame-coverage gate skips the licks of gated trials with the reason."""
import numpy as np
import pandas as pd

from scripts import pose3d_kinematics as K
from wfield_local import frame_coverage as FC
from wfield_local import session_licks as SL

GEO = {"dist_close": 2.0, "dist_far": 4.0}


def _frames():
    t = np.arange(0.0, 400.0, 4.0)
    P = np.c_[np.zeros_like(t), 1.0 + t / 100.0, -1.0 - t / 200.0]
    P[(t > 200) & (t < 380)] = np.nan                          # the second lick has only 1 3-D frame (t = 380+)
    return pd.DataFrame({"trial_id": 1, "t_ms": t, "tongue_ml": P[:, 0], "tongue_ap": P[:, 1], "tongue_dv": P[:, 2],
                         "position": "far_L"})


def _licks():
    C = pd.DataFrame({"trial_id": [1, 1], "lick_idx": [0, 1], "position": ["far_L", "far_L"], "on_ms": [40.0, 300.0],
                      "off_ms": [120.0, 380.0], "t_ms": [100.0, 380.0], "context": ["post_cue", "post_cue"],
                      "contact": [True, False], "cam4_matched": [True, False]})
    C["lick_key"] = SL.lick_key(C.trial_id, C.t_ms)
    return C


def _patch_targets(monkeypatch):
    from wfield_local import spout_world as SW
    monkeypatch.setattr(SW, "commanded_world", lambda geo, p: np.array([1.0, 3.0, -2.0]))
    monkeypatch.setattr(SW, "POSITIONS", ("far_L",))


def test_fused_mode_keeps_short_licks_with_nan_measures(monkeypatch):
    _patch_targets(monkeypatch)
    L = K.session_licks(_frames(), _licks(), GEO, np.zeros(3), keep_short=True)
    assert len(L) == 2 and L.lick_key.tolist() == ["1:100.0", "1:380.0"]
    assert L.n_3d.tolist()[1] < 3 and np.isnan(L.reach_mm.iloc[1]) and np.isfinite(L.reach_mm.iloc[0])
    D = K.session_licks(_frames(), _licks(), GEO, np.zeros(3))                     # the other modes drop it
    assert len(D) == 1 and D.lick_key.tolist() == ["1:100.0"]


def test_gate_by_coverage_records_reason():
    L = pd.DataFrame({"trial_id": [1, 2, 2, 3], "lick_key": ["a", "b", "c", "d"]})
    cov = pd.DataFrame({"trial_id": [1, 2], "cov_union": [0.99, 0.4], "gated": [False, True],
                        "gate_reason": ["", FC.GATE_REASON]})
    keep, gated = K.gate_by_coverage(L, cov)
    assert keep.lick_key.tolist() == ["a", "d"]                                   # trial 3 has no coverage row: kept
    assert gated.lick_key.tolist() == ["b", "c"] and (gated.gate_reason == FC.GATE_REASON).all()
    np.testing.assert_allclose(gated.cov_union, [0.4, 0.4])


def test_out_names_and_overwrite_refusal(tmp_path, monkeypatch):
    assert K.OUT_NAMES["fused"] == "kinematics_fused" and K.OUT_NAMES["all"] == "kinematics_all"
    (tmp_path / "licks.csv.gz").write_bytes(b"")
    monkeypatch.setenv(SL.FUSED_ENV, "off")
    monkeypatch.setattr(K, "pose3d_dir", lambda rv=None: tmp_path)
    try:
        K.main(["PS92:20260818", "--out", str(tmp_path)])
    except SystemExit as e:
        assert "exists" in str(e)
    else:
        raise AssertionError("expected a refusal")
    try:
        K.main(["--licks", "fused", "--out", str(tmp_path / "x")])
    except SystemExit as e:
        assert SL.FUSED_ENV in str(e)
    else:
        raise AssertionError("fused mode without a fused lick set must refuse")
