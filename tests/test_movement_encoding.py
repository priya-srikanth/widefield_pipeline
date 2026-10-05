"""movement_encoding / movement_inputs on a synthetic session with KNOWN kernels: the design matrix puts events and
signals on the right imaging frames, grouped ridge recovers the kernels, CV folds are trial blocks, variance
partitioning attributes each output to the group that drives it, a frozen model transfers with its training
standardisation, and the residual removes exactly the chosen groups."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import config
from wfield_local import movement_encoding as ME
from wfield_local import movement_inputs as MI

FS_IMG = 20.0        # imaging frames / s
FPS_CAM = 250.0


@pytest.fixture(autouse=True)
def _no_yaml(monkeypatch):
    monkeypatch.setattr(config, "defaults", lambda session=None: {})


def _session(seed=0, n_trials=120, scale_movement=1.0):
    """Trials every ~6 s; cue at trial start + 1 s, position cycles through 3; tongue onset 0.25-0.6 s after the
    cue on 80 % of trials; a continuous protrusion bump per lick (camera rate). Outputs: 0 = cue_far_L kernel only,
    1 = tongue-onset kernel only, 2 = continuous protrusion only, 3 = noise."""
    rng = np.random.default_rng(seed)
    starts = np.cumsum(rng.uniform(5.5, 6.5, n_trials))
    T_end = starts[-1] + 8
    ft = np.arange(0, T_end, 1 / FS_IMG)
    cue = starts + 1.0
    pos = np.array(["far_L", "close_center", "far_R"])[np.arange(n_trials) % 3]
    lick = rng.random(n_trials) < 0.8
    onset = cue[lick] + rng.uniform(0.25, 0.6, lick.sum())
    tcam = np.arange(0, T_end, 1 / FPS_CAM)
    prot = np.zeros_like(tcam)
    for o in onset:
        prot += 120 * np.exp(-0.5 * ((tcam - (o + 0.05)) / 0.03) ** 2) * scale_movement
    k_cue = np.exp(-np.arange(0, 1.0, 1 / FS_IMG) / 0.3)                   # 20 frames
    k_lick = np.exp(-np.arange(0, 1.0, 1 / FS_IMG) / 0.2)
    Y = rng.normal(0, 0.05, (len(ft), 4))
    for c in cue[pos == "far_L"]:
        i = int(round(c * FS_IMG))
        Y[i:i + len(k_cue), 0] += k_cue[:max(0, min(len(k_cue), len(ft) - i))]
    for o in onset:
        i = int(round(o * FS_IMG))
        Y[i:i + len(k_lick), 1] += k_lick[:max(0, min(len(k_lick), len(ft) - i))]
    Y[:, 2] += ME.bin_to_frames(tcam, prot, ft) / 100.0
    cues = pd.DataFrame({"cue_s": cue, "pos_name": pos})
    return ft, starts, cues, onset, tcam, prot, Y, k_cue, k_lick


def _inputs(ft, starts, cues, onset, tcam, prot):
    return MI.build_inputs(ft, cues=cues, events={"tongue_onset": onset},
                           video_signals={"tongue_protrusion": prot}, video_t_s=tcam, trial_starts_s=starts,
                           overrides={"lags_s": {"cue": [0.0, 1.0], "tongue_onset": [0.0, 1.0]},
                                      "continuous_lags_s": [0.0]})


def test_helpers_put_things_on_the_right_frames():
    ft = np.arange(0, 10, 0.05)
    assert list(ME.nearest_frame([0.0, 0.051, 9.95, 12.0, -1.0], ft)) == [0, 1, 199, -1, -1]
    t = np.arange(0, 10, 0.004)
    b = ME.bin_to_frames(t, np.where(t < 5, 1.0, 3.0), ft)
    assert np.allclose(b[:90], 1.0) and np.allclose(b[110:], 3.0)
    tpl = {"intercept_daqSample": 1000.0, "slope_daqSample_per_camFrame": 20.0, "fs_daq": 5000.0, "fps_cam": 250.0}
    from wfield_local import dlc_frames
    for f in (0, 1234, 99999):                                         # inverse of dlc_frames.frame_of
        assert dlc_frames.frame_of(tpl, float(MI.cam_frames_to_daq_s(tpl, f)), 0.0) == f


def test_design_has_one_column_per_lag_and_group_labels():
    ft, starts, cues, onset, tcam, prot, *_ = _session()
    d = ME.build_design(_inputs(ft, starts, cues, onset, tcam, prot))
    assert set(d.regressor) == {"cue_far_L", "cue_close_center", "cue_far_R", "tongue_onset", "tongue_protrusion"}
    assert (d.regressor == "tongue_onset").sum() == 21                  # 0 .. 1 s at 20 Hz
    assert set(d.group) == {"task", "lick_events", "tongue"}
    col = d.X[:, d.names.index("tongue_onset@+0.000")]
    assert col.sum() == len(onset)                                      # every onset on one frame


def test_folds_are_contiguous_trial_blocks():
    ft, starts, *_ = _session()
    f = ME.trial_block_folds(ft, starts, 5)
    assert set(f) == {0, 1, 2, 3, 4} and np.all(np.diff(f) >= 0)        # monotone = contiguous blocks


def test_fit_recovers_kernels_and_partition_attributes_correctly():
    ft, starts, cues, onset, tcam, prot, Y, k_cue, k_lick = _session()
    inp = _inputs(ft, starts, cues, onset, tcam, prot)
    d = ME.build_design(inp)
    folds = ME.trial_block_folds(ft, starts, 5)
    m = ME.fit(d, Y, folds=folds, grid=(0.1, 1.0, 10.0, 100.0))
    ks = ME.kernels(m)
    sd_cue = m.sd[np.isin(m.regressor, ["cue_far_L"])]
    est_cue = ks["cue_far_L"][1][:, 0] / sd_cue                          # back to indicator units
    assert np.corrcoef(est_cue[:len(k_cue)], k_cue)[0, 1] > 0.95
    est_lick = ks["tongue_onset"][1][:, 1] / m.sd[m.regressor == "tongue_onset"]
    assert np.corrcoef(est_lick[:len(k_lick)], k_lick)[0, 1] > 0.95
    vp = ME.variance_partition(d, Y, folds, m.alphas, out_names=["cue", "lick", "prot", "noise"]).set_index(
        ["output", "group"])
    assert vp.loc[("cue", "task"), "unique"] > 0.5
    assert vp.loc[("lick", "lick_events"), "r2_alone"] > 0.7 and vp.loc[("prot", "tongue"), "r2_alone"] > 0.7
    assert abs(vp.loc[("cue", "tongue"), "unique"]) < 0.05 and abs(vp.loc[("prot", "task"), "unique"]) < 0.05
    assert vp.loc[("noise", "task"), "r2_full"] < 0.05
    # the two movement groups overlap (stereotyped licks) -> partition movement as one block vs the task
    vp2 = ME.variance_partition(d, Y, folds, m.alphas, out_names=["cue", "lick", "prot", "noise"],
                                partitions={"task": ["task"], "movement": ["lick_events", "tongue"]}
                                ).set_index(["output", "group"])
    assert vp2.loc[("lick", "movement"), "unique"] > 0.5 and vp2.loc[("prot", "movement"), "unique"] > 0.5
    assert vp2.loc[("cue", "task"), "unique"] > 0.5 and abs(vp2.loc[("cue", "movement"), "unique"]) < 0.05


def test_frozen_model_transfers_and_residual_removes_chosen_groups():
    ft, starts, cues, onset, tcam, prot, Y, *_ = _session(seed=0)
    d = ME.build_design(_inputs(ft, starts, cues, onset, tcam, prot))
    m = ME.fit(d, Y, alphas={"task": 1.0, "lick_events": 1.0, "tongue": 1.0})
    ft2, st2, cues2, on2, tc2, pr2, Y2, *_ = _session(seed=1)
    d2 = ME.build_design(_inputs(ft2, st2, cues2, on2, tc2, pr2))
    assert ME.r2_score(Y2, ME.predict(m, d2))[:3].min() > 0.5            # frozen weights predict a new session
    res = ME.residual(m, d2, Y2, remove_groups=["lick_events", "tongue"])
    r2_mov_before = ME.r2_score(Y2[:, 1:3], ME.predict(m, d2, groups=["lick_events", "tongue"])[:, 1:3])
    r2_mov_after = ME.r2_score(res[:, 1:3], ME.predict(m, d2, groups=["lick_events", "tongue"])[:, 1:3])
    assert r2_mov_before.min() > 0.5 and r2_mov_after.max() < 0.05      # movement explained away
    cue_part = ME.predict(m, d2, groups=["task"], intercept=False)[:, 0]
    assert np.corrcoef(res[:, 0], cue_part)[0, 1] > 0.8                  # the target part is untouched
    with pytest.raises(ValueError):
        ME.predict(m, ME.build_design(MI.build_inputs(ft2, cues=cues2)))   # different columns -> refuse


def test_modulated_event_carries_only_the_modulation():
    ft = np.arange(0, 100, 0.05)
    t = np.arange(5, 95, 3.0)
    amp = np.where(np.arange(len(t)) % 2, 10.0, -10.0) + 30.0           # mean 30, +-10
    inp = MI.build_inputs(ft, modulated={"tongue_onset_x_angle": (t, amp)},
                          overrides={"lags_s": {"tongue_onset_x_angle": [0.0, 0.0]}})
    col = ME.build_design(inp).X[:, 0]
    assert set(np.round(col[col != 0], 6)) == {10.0, -10.0}              # centred: the mean is not modelled


def test_pose_signals_fill_tongue_in_at_lip_level():
    x = np.array([300.0, 300.0, 330.0, np.nan])
    y = np.array([140.0, 200.0, 240.0, np.nan])
    s = MI.pose_signals(x, y, [False, False, False, True], np.zeros(4), origin=(300.0, 100.0), ap_axis=(0.0, -1.0),
                        fps=250.0, lip_px=40.0)
    assert s["tongue_protrusion"][3] == 40.0 and np.isnan(s["tongue_lr"][3])
    assert s["tongue_protrusion"][1] == pytest.approx(100.0) and s["tongue_lr"][2] == pytest.approx(30.0)


def test_cv_residual_removes_movement_and_keeps_the_target():
    ft, starts, cues, onset, tcam, prot, Y, *_ = _session()
    mov = ME.build_design(MI.build_inputs(ft, events={"tongue_onset": onset}, video_signals={"tongue_protrusion": prot},
                                          video_t_s=tcam, trial_starts_s=starts,
                                          overrides={"lags_s": {"tongue_onset": [0.0, 1.0]}, "continuous_lags_s": [0.0]}))
    folds = ME.trial_block_folds(ft, starts, 5)
    R = ME.cv_residual(mov, Y, folds, {"lick_events": 1.0, "tongue": 1.0})
    assert np.var(R[:, 1]) < 0.3 * np.var(Y[:, 1]) and np.var(R[:, 2]) < 0.3 * np.var(Y[:, 2])   # movement gone
    assert np.var(R[:, 0]) > 0.9 * np.var(Y[:, 0])                                               # target kept
