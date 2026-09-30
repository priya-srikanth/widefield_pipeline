"""The round-4 pick rules on synthetic tracks, and the target/context/promoted rows they produce."""
import numpy as np

from wfield_local.dlc_context_frames import CONTEXT, CONTEXT_LABEL
from wfield_local.dlc_hard_frames import ROUND, _spikes, candidates, rows_for, select

N = 300


def _still():
    """A resting face: nose at (340, 260), jaw 130 px below, no tongue, spout parked."""
    one, z = np.ones(N), np.zeros(N)
    return {"nose": (340 * one, 260 * one, one), "jaw": (350 * one, 390 * one, one),
            "tongue": (345 * one, 400 * one, z), "spout": (330 * one, 530 * one, one)}


def _kinds(P):
    return [k for _, k, _ in candidates(P, rest_open=130.0, lat_med=10.0, lat_mad=2.0)]


def test_resting_face_gives_nothing():
    assert _kinds(_still()) == []


def test_small_opening_without_tongue_is_incomplete_lick():
    P = _still()
    jx, jy, jp = P["jaw"]
    bump = 20 * np.exp(-0.5 * ((np.arange(N) - 150) / 6) ** 2)      # 20 px jaw drop
    P["jaw"] = (jx, jy + bump, jp)
    got = candidates(P, 130.0, 10.0, 2.0)
    assert [(i, k) for i, k, _ in got] == [(150, "incomplete_tongue")]


def test_same_opening_with_tongue_seen_is_not_picked():
    P = _still()
    jx, jy, jp = P["jaw"]
    P["jaw"] = (jx, jy + 20 * np.exp(-0.5 * ((np.arange(N) - 150) / 6) ** 2), jp)
    tx, ty, tp = P["tongue"]
    tp = tp.copy()
    tp[148:153] = 0.9
    P["tongue"] = (tx, ty, tp)
    assert "incomplete_tongue" not in _kinds(P)


def test_full_opening_is_not_incomplete():
    P = _still()
    jx, jy, jp = P["jaw"]
    P["jaw"] = (jx, jy + 45 * np.exp(-0.5 * ((np.arange(N) - 150) / 6) ** 2), jp)
    assert "incomplete_tongue" not in _kinds(P)


def test_single_frame_jump_is_a_spike_but_a_real_move_is_not():
    x = np.full(20, 100.0)
    y = np.full(20, 100.0)
    p = np.ones(20)
    x[10] = 140                                     # one-frame excursion
    x[15:] = 140                                    # a genuine step: neighbours disagree afterwards
    s = _spikes(x, y, p)
    assert s[10] and not s[15] and s.sum() == 1


def test_select_keeps_spacing_and_caps():
    cands = [(100, "incomplete_tongue", 20.0), (120, "incomplete_tongue", 25.0),
             (400, "incomplete_tongue", 15.0), (410, "erratic_jaw", 100.0), (900, "tricky_spout", 0.3)]
    got = select(cands, caps={"incomplete_tongue": 4, "erratic_tongue": 1, "erratic_jaw": 1, "tricky_spout": 1})
    frames = sorted(f for f, _, _ in got)
    assert frames == [120, 400, 900]               # 100 too close to 120; jaw 410 too close to 400


def test_rows_centre_context_and_promotion():
    tpl = {"fs_daq": 5000.0, "slope_daqSample_per_camFrame": 20.0, "intercept_daqSample": 0.0}
    trial = {"cue_s": 3.0, "trial_id": 5, "pos_name": "far_L"}
    rows = rows_for([(1000, "incomplete_tongue", 20.0, trial), (5000, "tricky_spout", 0.3, trial)],
                    {"animal": "PS95", "date": "20260907", "cam": "cam4", "epoch": "chronic", "video_stem": "cam4_x"},
                    tpl, 10_000)
    by = {r["frame"]: r for r in rows}
    assert by[1000]["category"] == ROUND and by[1000]["phase"] == "incomplete_tongue"
    assert {f for f, r in by.items() if r["category"] == CONTEXT_LABEL} == {997, 1003}
    assert {f for f, r in by.items() if r["category"] == CONTEXT} == {996, 998, 999, 1001, 1002, 1004}
    assert by[5000]["category"] == ROUND and not any(4990 < f < 5010 and f != 5000 for f in by)   # spout: no context
    assert abs(by[1000]["t_from_cue_s"] - (1000 / 250 - 3.0)) < 1e-9
