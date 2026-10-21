"""Pins the 10-19 / 10-20 options of the combined encoding model (wfield_local/combined_encoding.py): the DROP-p per-spout
model, the polar 3-D coordinates, the combined-lick timing kernel and the output-folder naming. Synthetic session, no
share access."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import combined_encoding as ce

POS = ("close_contra", "close_ipsi", "close_mid", "far_contra", "far_ipsi", "far_mid")


def _behaviour(n_trials: int = 48, seed: int = 0, **extra) -> tuple[ce.Behaviour, np.ndarray]:
    """Trials every 8 s, spout arrival 3 s before the cue, stop 3 s after it; one lick per trial at +0.4 s."""
    rng = np.random.default_rng(seed)
    cue = 10.0 + 8.0 * np.arange(n_trials)
    pos = np.array([POS[i % 6] for i in rng.permutation(n_trials)])
    T = pd.DataFrame({"trial_id": np.arange(1, n_trials + 1), "pos": pos, "strobe_s": cue - 3.0, "cue_s": cue,
                      "stop_s": cue + 3.0, "engaged": True})
    ft = np.arange(0.0, cue[-1] + 10.0, 0.1)
    lick = cue + 0.4
    z = np.zeros_like(ft)
    analog = {k: (ft, rng.normal(size=len(ft)) * 0 + z) for k in
              ("tongue_x", "tongue_y", "tongue_prot", "tongue_speed", "jaw_x", "jaw_y", "jaw_speed", "nose_x",
               "nose_y", "running")}
    B = ce.Behaviour(T, lick, lick + 0.05, pos, np.zeros(n_trials), lick + 0.02, np.zeros(0), analog, "test",
                     **extra)
    return B, ft


def _frames_of(T: pd.DataFrame, ft: np.ndarray, p: str) -> np.ndarray:
    """Frames inside spout p's trial windows (strobe .. stop)."""
    m = np.zeros(len(ft), bool)
    for r in T[T.pos == p].itertuples():
        m |= (ft >= r.strobe_s) & (ft <= r.stop_s)
    return m


# ------------------------------------------------------------------------------------------------ DROP-p
def test_default_position_blocks_sum_to_zero():
    """The reason DROP-p cannot just delete p's columns: the six deviation blocks sum to zero, so any one is redundant."""
    B, ft = _behaviour()
    X = ce.raw_blocks(B, ft)["pos_cue"]
    k = X.shape[1] // 6
    S = sum(X[:, i * k:(i + 1) * k] for i in range(6))
    assert np.abs(S).max() < 1e-4


@pytest.mark.parametrize("p", ["far_contra", "close_mid"])
def test_droppos_removes_one_spout_and_is_zero_on_its_trials(p):
    B, ft = _behaviour()
    full = ce.raw_blocks(B, ft)
    drop = ce.raw_blocks(B, ft, drop_pos=p)
    for v in ce.POS_VARS:
        assert drop[v].shape[1] == full[v].shape[1] * 5 // 6
        # p's trials get only the shared response: every remaining deviation column is zero there
        assert np.abs(drop[v][_frames_of(B.trials, ft, p)]).max() < 1e-5
        # the other variables are untouched
    assert np.array_equal(drop["licks"], full["licks"])


def test_droppos_loses_exactly_one_dimension_per_kernel_column():
    """[shared, full deviations] vs [shared, drop-p deviations]: rank falls by one kernel's worth (p constrained)."""
    B, ft = _behaviour()
    full, drop = ce.raw_blocks(B, ft), ce.raw_blocks(B, ft, drop_pos="far_mid")
    for v, sv in (("pos_cue", "cue"), ("pos_enl", "arrival")):
        rk = np.linalg.matrix_rank(full[sv].astype(np.float64), tol=1e-6)    # one kernel's EFFECTIVE rank (< columns)
        r_full = np.linalg.matrix_rank(np.hstack([full[sv], full[v]]).astype(np.float64), tol=1e-6)
        r_drop = np.linalg.matrix_rank(np.hstack([drop[sv], drop[v]]).astype(np.float64), tol=1e-6)
        assert r_full == 6 * rk                       # shared + 5 independent deviations = six spouts
        assert r_drop == 5 * rk                       # p constrained to the mean of the others: one spout fewer


def test_droppos_toy_constraint_is_mean_of_others():
    """Indicator toy (one lag): p's fitted value = the trial-weighted mean of the other spouts' fitted values."""
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 6, 900)
    X = np.eye(6)[lab]
    S = np.ones(len(lab))
    m = X.mean(0)
    p = 2
    D = [X[:, q] - m[q] / (1 - m[p]) * (S - X[:, p]) for q in range(6) if q != p]
    A = np.column_stack([S, *D])
    y = rng.normal(size=6)[lab] + rng.normal(0, 0.1, len(lab))
    f = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    others = [q for q in range(6) if q != p]
    w = np.array([(lab == q).sum() for q in others], float)
    assert np.isclose(f[lab == p].mean(), np.average([f[lab == q].mean() for q in others], weights=w), atol=1e-8)
    assert np.linalg.matrix_rank(A) == 5


# ------------------------------------------------------------------------------------------------ polar
def test_polar_known_directions():
    lat, lat2, tilt, dist = ce._polar([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
    raw_lat = lat + np.nanmean(np.degrees(np.arctan2([0, 1, 0], np.hypot([0, 0, 1], [1, 1, 1]))))
    assert np.allclose(raw_lat, [0.0, 45.0, 0.0])                  # straight down / 45 deg to the mouse's right / forward
    assert np.allclose(tilt, [0.0, 0.0, 45.0])
    assert np.allclose(dist, [1.0, np.sqrt(2), np.sqrt(2)])
    assert np.allclose(lat2, lat ** 2)
    assert abs(np.nanmean(lat)) < 1e-12                            # centred before squaring


# ------------------------------------------------------------------------------------------------ licks
def test_lick_modes_default_and_invalid():
    assert ce._cam1_lick_events(None, "PS92", "20260807", pd.DataFrame(), "cam4") == (None, None)
    with pytest.raises(ValueError):
        ce._cam1_lick_events(None, "PS92", "20260807", pd.DataFrame(), "cam1only")


def test_combined_and_lips_lick_kernels():
    B0, ft = _behaviour()
    width = ce.raw_blocks(B0, ft)["licks"].shape[1]
    extra = B0.trials.cue_s.to_numpy()[::4] + 1.5
    Bc, _ = _behaviour(extra_lick_on_s=extra)
    Lc = ce.raw_blocks(Bc, ft)["licks"]
    assert Lc.shape[1] == width                                     # same kernels, more events
    assert not np.array_equal(Lc, ce.raw_blocks(B0, ft)["licks"])
    Bl, _ = _behaviour(extra_lick_on_s=extra, lips_on_s=extra + 0.7)
    assert ce.raw_blocks(Bl, ft)["licks"].shape[1] == width * 3 // 2   # + one lips kernel (lick, contact, lips)


def test_exec_mode_invalid():
    with pytest.raises(ValueError):
        ce._exec_3d(pd.DataFrame({"trial_id": [1], "lick_idx": [0]}), np.array([0.0]), np.array([1.0]), "4d", "x")


# ------------------------------------------------------------------------------------------------ binned contest
def _binned(monkeypatch, exec_shift: dict, **extra):
    """Target geometry on the cam4 plane; each lick lands at its spout's (angle, reach) shifted per exec_shift."""
    geo = {p: (a, r) for p, a, r in zip(POS, (-30, 30, 0, -45, 45, 0), (10, 10, 10, 20, 20, 20))}
    B, ft = _behaviour(target_angle={p: g[0] for p, g in geo.items()},
                       target_dist={p: g[1] for p, g in geo.items()}, **extra)
    land = [exec_shift.get(p, p) for p in B.lick_pos]
    B.lick_angle = np.array([geo[q][0] for q in land], float)
    B.lick_reach = np.array([geo[q][1] for q in land], float)
    monkeypatch.setenv(ce.BINS_ENV, "1")
    return B, ft


def test_exec_bins_from_geometry_not_labels(monkeypatch):
    B, _ = _binned(monkeypatch, {"far_contra": "close_ipsi"})        # far contra licks all land on close ipsi
    tl, el, frame = ce.exec_bins(B)
    assert frame == "cam4_angle_reach"
    assert (tl == B.lick_pos).all()
    m = B.lick_pos == "far_contra"
    assert (el[m] == "close_ipsi").all() and (el[~m] == B.lick_pos[~m]).all()
    assert not (el == "far_contra").any()                             # bins never calibrated on the labels


def test_exec_bins_unusable_licks_blank(monkeypatch):
    B, _ = _binned(monkeypatch, {})
    B.lick_angle[:5] = np.nan
    tl, el, _ = ce.exec_bins(B)
    assert (tl[:5] == "").all() and (el[:5] == "").all() and (tl[5:] != "").all()


def test_binned_blocks_matched_and_deviation_coded(monkeypatch):
    B, ft = _binned(monkeypatch, {})
    R = ce.raw_blocks(B, ft)
    lick_k = ce._event_cols(B.lick_on_s, ft, ce.LAGS["lick"]).shape[1]
    for name in ("lick_bin_tgt", "lick_bin_exec"):
        X = R[name]
        assert X.shape[1] == 6 * lick_k                               # six free kernels on each side
        assert np.abs(sum(X[:, i * lick_k:(i + 1) * lick_k] for i in range(6))).max() < 1e-4   # sum to zero
    assert np.allclose(R["lick_bin_tgt"], R["lick_bin_exec"])        # licks on target -> identical blocks
    B2, _ = _binned(monkeypatch, {"far_contra": "close_ipsi"})
    R2 = ce.raw_blocks(B2, ft)
    assert not np.allclose(R2["lick_bin_tgt"], R2["lick_bin_exec"])


def test_binned_blocks_off_by_default(monkeypatch):
    monkeypatch.delenv(ce.BINS_ENV, raising=False)
    B, ft = _behaviour()
    assert "lick_bin_tgt" not in ce.raw_blocks(B, ft)


# ------------------------------------------------------------------------------------------------ folder naming
@pytest.mark.parametrize("env, suffix", [
    ({}, ""), ({ce.ONLY_ENV: "nokin:"}, "_subset-nokin"), ({ce.ONLY_ENV: "droppos-"}, "_subset-droppos-"),
    ({ce.ONLY_ENV: "nokin:", ce.EXEC_ENV: "polar"}, "_subset-nokin_exec-polar"),
    ({ce.ONLY_ENV: "nokin:", ce.LICKS_ENV: "lips"}, "_subset-nokin_licks-lips"),
    ({ce.ONLY_ENV: "nokin:", ce.LICKS_ENV: "cam4"}, "_subset-nokin"),
    ({ce.ONLY_ENV: "nokin:", ce.EXEC_ENV: "polar", ce.BINS_ENV: "1"}, "_subset-nokin_exec-polar_bins"),
    ({ce.FR_ENV: "master_pre_v1"}, "_fr-master_pre_v1"),
    ({ce.VIDEO_ENV: "cam1,cam4", ce.FR_ENV: "master_pre_v1"}, "_video-cam1-cam4_fr-master_pre_v1"),
])
def test_out_dir_names(monkeypatch, env, suffix):
    from wfield_local import orofacial_outputs as oo
    for k in (ce.ONLY_ENV, ce.EXEC_ENV, ce.LICKS_ENV, ce.BINS_ENV, ce.MATCH_ENV, ce.SUFFIX_ENV, ce.VIDEO_ENV, ce.FR_ENV):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(oo, "root", lambda rv: Path("ROOT"))
    monkeypatch.setattr(ce, "_run", lambda: "RUN")
    assert ce.out_dir(None, "movement_first") == Path("ROOT") / "RUN" / "imaging_encoding" / f"movement_first{suffix}"


def test_soft_exec_bins_sum_to_zero_and_weights(monkeypatch, tmp_path):
    """Calibrated 3-D bins: soft weights sum to 1 per lick; the soft executed blocks still sum to zero."""
    B, ft = _behaviour()
    rng = np.random.default_rng(1)
    land = {p: np.array([x, y, z]) for p, x, y, z in zip(POS, (0.5, -0.5, 0, 1.5, -1.5, 0), (0, 0, .5, .8, .5, 1.6),
                                                         (3.9, 3.7, 3.8, 5.0, 4.7, 4.2))}
    xyz = np.array([land[p] for p in B.lick_pos]) + rng.normal(0, 0.3, (len(B.lick_pos), 3))
    B.exec_xyz, B.session = [xyz[:, 0], xyz[:, 1], xyz[:, 2]], "PS92_20260809"
    B.lick_angle = np.zeros(len(B.lick_pos))
    cal = pd.DataFrame([{"animal": "PS92", "calib_for": "default", "position": p, "n": 100, "ml": v[0], "ap": v[1],
                         "down": v[2], "sd_ml": .5, "sd_ap": .5, "sd_down": .9} for p, v in land.items()])
    cal.to_csv(tmp_path / "cal.csv", index=False)
    monkeypatch.setenv(ce.BINS_CALIB_ENV, str(tmp_path / "cal.csv"))
    monkeypatch.setenv(ce.BINS_ENV, "1")
    tl, el, frame, W, P = ce.exec_bins(B, weights=True)
    assert frame == "3d_calibrated" and np.allclose(W.sum(1), 1) and np.mean(tl == el) > 0.6
    X = ce.raw_blocks(B, ft)["lick_bin_exec"]
    k = X.shape[1] // 6
    assert np.abs(sum(X[:, i * k:(i + 1) * k] for i in range(6))).max() < 1e-4


@pytest.mark.parametrize("mode", ["shared", "split"])
def test_licks_all_modes(monkeypatch, mode):
    """Whole-session licks: shared = same kernels, more events; split = + one pre-cue and one ITI kernel."""
    B0, ft = _behaviour()
    width = ce.raw_blocks(B0, ft)["licks"].shape[1]
    B, _ = _behaviour()
    cue = B.trials.cue_s.to_numpy()
    B.context_lick_on_s = {"pre_cue": cue[::2] - 1.0, "iti": cue[1::3] + 4.0}
    monkeypatch.setenv(ce.LICKS_ALL_ENV, mode)
    X = ce.raw_blocks(B, ft)["licks"]
    k = width // 2                                                   # lick + contact kernels, same width each
    if mode == "shared":
        assert X.shape[1] == width and not np.array_equal(X, ce.raw_blocks(B0, ft)["licks"])
    else:
        assert X.shape[1] == width + 2 * k


def test_licks_all_invalid_mode(monkeypatch):
    B, ft = _behaviour()
    B.context_lick_on_s = {"pre_cue": np.zeros(0), "iti": np.zeros(0)}
    monkeypatch.setenv(ce.LICKS_ALL_ENV, "everything")
    with pytest.raises(ValueError):
        ce.raw_blocks(B, ft)
