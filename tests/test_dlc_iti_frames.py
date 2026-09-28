"""The between-trial spout frame rule, pinned (recovered into the repo 2026-09-28 from the 09-26 scratch run).

No network here: `predict` and `read_frame` are injected, so these test the RULE -- which gaps, which
frame in a gap, the manifest schema -- not DeepLabCut.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import dlc_iti_frames as iti
from wfield_local.dlc_frames import MANIFEST_COLUMNS

POS = ["far_L", "far_center", "far_R", "close_L", "close_center", "close_R"]


def _trials(seq):
    """A trial table whose position sequence is `seq`; trials 10 s apart, trial_start 4 s before each cue."""
    n = len(seq)
    return pd.DataFrame({"trial_id": np.arange(n), "pos_name": seq,
                         "cue_s": 10.0 + 10.0 * np.arange(n),
                         "trial_start_s": 6.0 + 10.0 * np.arange(n)})


def _tpl(n_frames=100_000):
    # DAQ 5 kHz, camera 250 fps: 20 DAQ samples per frame, zero intercept
    return {"fs_daq": 5000.0, "fps_cam": 250.0, "slope_daqSample_per_camFrame": 20.0,
            "intercept_daqSample": 0.0, "n_cam_frames": n_frames}


def test_only_gaps_that_follow_a_position_change_are_candidates():
    by = iti.position_change_gaps(["far_L", "far_L", "far_R", "far_R", "close_L"])
    assert by == {"far_R": [1], "close_L": [3]}, "keyed by the NEW position, indexed by the trial before the gap"


def test_two_gaps_per_position_deterministic_under_the_seed():
    by = {p: list(range(10 * k, 10 * k + 6)) for k, p in enumerate(POS)}
    a = iti.pick_gaps(by, np.random.default_rng(92))
    b = iti.pick_gaps(by, np.random.default_rng(92))
    assert a == b and len(a) == 12
    for k, p in enumerate(sorted(POS)):
        assert all(g in by[p] for g in a[2 * k:2 * k + 2]), "each position contributes exactly two of its own gaps"


def test_transit_is_the_largest_jump_and_doubt_is_the_lowest_likelihood():
    xs = np.array([300, 301, 302, 340, 341, 342.0])            # the spout moves between samples 2 and 3
    ps = np.array([0.9, 0.9, 0.9, 0.7, 0.3, 0.9])
    assert iti.choose_in_gap(xs, ps) == (3, 4)


def test_a_frame_that_failed_to_decode_is_the_doubt_frame_and_never_the_transit():
    xs = np.array([300, np.nan, 302.0])
    ps = np.array([0.9, 0.0, 0.9])
    k_move, k_doubt = iti.choose_in_gap(xs, ps)
    assert k_doubt == 1
    assert k_move in (0, 2)


def test_scan_rows_carry_the_manifest_schema_and_skip_short_gaps():
    seq = ["far_L", "far_R", "far_R", "close_L"]                # gaps after trial 0 (-> far_R) and 2 (-> close_L)
    t = _trials(seq)
    t.loc[3, "trial_start_s"] = t.loc[2, "cue_s"] + 3.5 + 0.5    # gap after trial 2 is only 0.5 s: skipped
    calls = []

    def read_frame(video, f):
        calls.append(f)
        return np.zeros((680, 680, 3), np.uint8)

    def predict(im):                                            # x jumps at the 4th sample, confidence dips at the 6th
        k = len(calls) - 1
        return (300.0 if k < 3 else 340.0), (0.2 if k == 5 else 0.9)

    rows = iti.scan_trials(t, _tpl(), "v.avi", "PS95", "20260907", "cam4_x", "chronic", predict,
                           read_frame=read_frame)
    assert {r["phase"] for r in rows} == {iti.PHASE_MOVE, iti.PHASE_DOUBT}
    assert {r["position"] for r in rows} == {"far_R"}, "the 0.5 s gap contributed nothing"
    for r in rows:
        assert set(MANIFEST_COLUMNS) <= set(r), r.keys()
        assert r["image"] == f"img{r['frame']:07d}.png" and r["category"] == "iti" and r["cam"] == "cam4"
        assert r["trial_id"] == 1, "the gap is attributed to the trial it leads INTO"
    # the gap runs from cue+3.5 s to the next trial_start, sampled every 12 frames
    f0, f1 = round((10.0 + 3.5) * 250), round(16.0 * 250)
    assert calls == list(range(f0, f1, 12))
    move = next(r for r in rows if r["phase"] == iti.PHASE_MOVE)
    doubt = next(r for r in rows if r["phase"] == iti.PHASE_DOUBT)
    assert move["frame"] == f0 + 3 * 12 and move["spout_dx"] == 40.0
    assert doubt["frame"] == f0 + 5 * 12 and doubt["spout_p"] == 0.2


def test_the_recorded_2026_09_26_picks_have_the_documented_shape():
    """The rows the guide was built from: 47 frames, 2 sessions, all six positions, both phases."""
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "docs" / "dlc_iti_rows_20260926.csv")
    assert len(df) == 47 and df.video_stem.nunique() == 2
    assert set(df.position) == set(POS) and set(df.phase) == {iti.PHASE_MOVE, iti.PHASE_DOUBT}
    assert df.groupby("video_stem").position.nunique().min() == 6, "all six positions in EACH folder"
    assert (df[df.phase == iti.PHASE_DOUBT].spout_p < 0.6).mean() > 0.3, "the doubt frames are where the network is unsure"
