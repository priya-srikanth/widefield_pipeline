"""tongue_kinematics: per-trial features come out right, the angle has the stated geometry, and the whole v7
per-trial path + features equal stroke_orofacial's (source loaded by file path; see test_tongue_detect)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_tongue_detect import FPS, source_fixture, synth_session  # noqa: F401  (fixture "source")
from wfield_local import orofacial_clean as oc
from wfield_local import tongue_detect as td
from wfield_local import tongue_kinematics as tk


@pytest.fixture(autouse=True)
def _no_yaml(monkeypatch):
    """Tests run on DEFAULTS, not on whatever configs/defaults.yaml carries (another agent may add the block)."""
    monkeypatch.setattr(td, "config_overrides", lambda: {})


def _trials(cues):
    return [tk.Trial(trial_id=k, cue_frame=float(c), position=tk.POSITIONS[k % 6]) for k, c in enumerate(cues)]


# =========================================================================== features

def test_per_trial_features_on_a_clean_lick_train():
    y, x, fm, lk, cues, truth = synth_session(seed=3, extras=False)
    r = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS)
    pt = r.per_trial.set_index("trial_id")
    for k in range(len(cues)):
        tt = sorted(t for (kk, _, t) in truth if kk == k)
        row = pt.loc[k]
        assert row.n_kept_licks == len(tt)                            # every synthetic lick kept, nothing else
        assert abs(row.lick1_t_peak_ms - tt[0]) <= 8.0 and abs(row.lick2_t_peak_ms - tt[1]) <= 8.0
        assert row.lick1_y_peak > 100
        assert row.n_licks == len(tt)                                 # slope count on the cleaned slice
        nv = min(5, len(tt))
        vy = np.array([row[f"vy_peak_lick{j + 1}"] for j in range(5)])
        assert np.all(np.isfinite(vy[:nv]) & (vy[:nv] > 1000)) and np.all(np.isnan(vy[nv:]))
        assert row.n_bouts == 1 and row.n_multi_lick_bouts == 1     # ILIs 140-212 ms < 300 ms
        assert row.n_licks_in_first_multi_bout == len(tt)
        assert 135 < row.mean_within_bout_ili_ms < 217 if len(tt) >= 3 else np.isnan(row.mean_within_bout_ili_ms)
        assert row.lick_rate_hz == pytest.approx(len(tt) / 4.93)
    assert len(r.per_lick) == len(truth) and len(r.bouts) == len(cues)
    assert set(r.per_lick.source) <= {"lmax", "bounded", "legacy"}
    assert np.isnan(r.per_lick.peak_angle_deg).all() and r.angle_per_frame is None   # no SpoutFrame -> no angles
    traces = r.licking_dynamics_traces
    assert set(traces.position) == set(tk.POSITIONS) and len(traces) == 6 * 247


def test_fix_detect_offset_reads_the_peak_frame():
    """Theirs reads frame-indexed values one frame early; `fix_detect_offset` reads them AT the peak time."""
    y, x, fm, lk, cues, _truth = synth_session(seed=3, extras=False)
    x = x + np.arange(len(x)) * 0.01                                  # x differs frame to frame
    old = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS)
    new = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS,
                                       params_override={"fix_detect_offset": True})
    n_checked = 0
    for ro, rn in zip(old.trial_results, new.trial_results):
        assert [k["t_ms"] for k in ro.kept_licks] == [k["t_ms"] for k in rn.kept_licks]   # times untouched
        for ko, kn in zip(ro.kept_licks, rn.kept_licks):
            f = int(np.argmin(np.abs(rn.t_ms - kn["t_ms"])))
            assert rn.t_ms[f] == pytest.approx(kn["t_ms"])
            assert kn["x"] == pytest.approx(rn.x_clean[f], nan_ok=True)
            assert ko["x"] == pytest.approx(ro.x_clean[f - 1], nan_ok=True)
            assert kn["rise_start_ms"] - ko["rise_start_ms"] == pytest.approx(1000.0 / FPS)
            n_checked += bool(np.isfinite(kn["x"]) and np.isfinite(ko["x"]) and kn["x"] != ko["x"])
    assert n_checked > 20


def test_angle_geometry():
    f = tk.SpoutFrame(origin=(300.0, 100.0), ap_axis=(0.0, -5.0))   # spouts BELOW the mouth; normalised
    a = tk.compute_signed_tongue_angle_deg(np.array([300.0, 400.0, 200.0]), np.array([200.0, 200.0, 200.0]), f)
    np.testing.assert_allclose(a, [0.0, 45.0, -45.0], atol=1e-12)    # straight out = 0, image-right = +
    g = tk.SpoutFrame(origin=(0.0, 0.0), ap_axis=(1.0, -1.0))         # tilted axis: out of the mouth = down-left
    np.testing.assert_allclose(tk.compute_signed_tongue_angle_deg(-10.0, 10.0, g), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        tk.SpoutFrame(origin=(0, 0), ap_axis=(0, 0))


def test_lick_angles_through_the_pipeline_are_positive_for_a_rightward_tongue():
    y, x, fm, lk, cues, _ = synth_session(seed=5, extras=False)
    X0, Y0 = 320.0, 250.0
    frame = tk.SpoutFrame(origin=(X0, Y0 - 50.0), ap_axis=(0.0, -1.0))   # mouth 50 px above the retracted tip
    r = tk.compute_tongue_kinematics(np.abs(x), y, fm, lk, _trials(cues), fps=FPS, X0=X0, Y0=Y0, spout_frame=frame)
    assert (r.per_lick.peak_angle_deg > 0).all() and (r.per_lick.angle_smoothed_at_peak_deg > 0).all()
    assert (r.per_trial.lick1_angle_at_ypeak > 0).all() and (r.per_trial.angle_max_signed_lick1 > 0).all()
    np.testing.assert_allclose(r.per_lick.x_img - r.per_lick.x, X0)
    assert np.isfinite(r.angle_per_frame).all()
    with pytest.raises(ValueError):
        tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), spout_frame=frame)   # no X0/Y0


def test_trial_inputs():
    with pytest.raises(ValueError):
        tk.Trial(trial_id=0, cue_frame=10.0, position="L")
    df = pd.DataFrame({"trial_id": [1, 2], "cue_frame": [100.25, np.nan], "position": ["far_L", "close_R"]})
    assert tk.trials_from_frame(df) == [tk.Trial(1, 100.25, "far_L")]
    arr = np.arange(1000.0)
    s, t, lo, hi = tk.extract_trial_slice(arr, 100.0, (70.0, 8000.0), FPS)        # integer cue == theirs
    assert lo == 117 and t[0] == 68.0 and hi == 999 and len(s) == hi - lo + 1
    s, t, lo, _ = tk.extract_trial_slice(arr, 100.6, (70.0, 400.0), FPS)          # float cue: exact t
    assert lo == 118 and t[0] == pytest.approx(69.6)


def test_integration_with_orofacial_clean():
    """Raw pose -> orofacial_clean v5p3 -> tongue_kinematics: the sign convention (protrusion = +y_final) and the
    fill codes line up, and every synthetic lick is found."""
    y, x, fm, _, cues, truth = synth_session(seed=7, extras=False)
    n = len(y)
    rng = np.random.default_rng(0)
    vis = fm != 3
    df = pd.DataFrame({"tongue_x": 320.0 + x, "tongue_y": 250.0 + y,
                       "tongue_likelihood": np.where(vis, rng.uniform(0.96, 1.0, n), 0.05)})
    p = {"x_range_pm": 150.0, "x_range_source": "lk", "lk_thr": 0.6, "n_neigh": 2, "neighbor_ms": 16.0,
         "max_gap_ms": 56.0, "interp_n_side": 6, "interp_min_total_context": 6, "require_bracketed_by_hilk": True,
         "interpolation_method": "pchip", "interpolation_fallback": "linear",
         "baseline": {"force": False, "y0_lk_thr": 0.95, "y0_percentile": 5.0, "pool_win_ms": [0.0, 5000.0]}}
    c = oc.clean_bodypart(df, "tongue", cues, FPS, p=p)
    r = tk.from_clean(c, _trials(cues), spout_frame=tk.SpoutFrame((c.X0, c.Y0 - 50.0), (0.0, -1.0)))
    assert len(r.per_lick) == len(truth)
    np.testing.assert_allclose(r.per_lick.y_img, r.per_lick.y + c.Y0)


# =========================================================================== parity vs source

def _their_params(p):
    g = {k: v for k, v in p["gates"].items() if k != "e_min_robust_std_px"}
    tp = {"preclean": p["preclean"], "detector": p["detector"], "detector_v2": p["detector_v2"], "gates": g,
          "detector_per_cell": {}, "bout_per_cell": {}, "bout": p["bout"],
          "direction_thresholds": {"x_left_thr_px": p["direction_thresholds"]["img_right_thr_px"],
                                   "x_right_thr_px": p["direction_thresholds"]["img_left_thr_px"]},
          "n_licks_categorical": p["n_licks_categorical"], "n_licks_binning": p["n_licks_binning"],
          "max_licks_per_trial": p["max_licks_per_trial"], "licking_dynamics": p["licking_dynamics"],
          "angle": p["angle"], "licking_dyn_detect_win_ms": [70.0, 8000.0], "angle_detect_win_ms": [70.0, 8000.0]}
    for k in ("lick12_detect_win_ms", "peak_velocity_detect_win_ms", "lick_count_detect_win_ms",
              "lick_count_apply_win_ms", "lick_count_bin_win_ms", "licking_dyn_apply_win_ms", "angle_apply_win_ms"):
        tp[k] = p[k]
    return {"dlc_kinematics": {"tongue_pertrial": tp}}


def _theirs(src, y, x, fm, lk, cues, eye):
    """Their `_build_tongue_pertrial_core` v7 path, re-glued from their own functions (the driver itself needs
    their loaders). Returns per-trial dicts, per-trial kept licks, bout summary rows, traces, angle."""
    stp, sang = src["tongue_pertrial"], src["_angle"]
    params = _their_params(tk.params())
    intp, base = td.interp_masks(fm)
    n = len(y)
    res = []
    for k, c in enumerate(cues):
        sl = {nm: stp._extract_trial_slice(a, c, (70.0, 8000.0), FPS) for nm, a in
              (("x", x), ("y", y), ("i", intp), ("b", base), ("l", lk))}
        lo = max(0, int(c + np.floor(70.0 / 1000.0 * FPS)))
        hi = min(n - 1, int(c + np.ceil(8000.0 / 1000.0 * FPS)))
        res.append(stp._run_v7_pipeline_per_trial(
            sl["y"][0], sl["x"][0], sl["i"][0], sl["b"][0], sl["l"][0], sl["x"][1], FPS, trial_index=k, side="L",
            tone_frame_idx=c, session_frame_lo=lo, session_frame_hi=hi, eye_x_in_spout_frame=eye[0],
            eye_y_in_spout_frame=eye[1], params=params))
    df = pd.DataFrame({"x_final": x, "y_final": y, "is_baseline_fill": base})
    xb, yb = stp._v7_cleaned_bundle_arrays(df, res)
    angle = sang._extract_per_frame_angle_smoothed(xb, yb, eye[0], eye[1], FPS, params=params)
    rows, by_pos, bouts = [], {pos: [] for pos in tk.POSITIONS}, []
    edges = np.linspace(0, 1, 73)
    for k, (c, r) in enumerate(zip(cues, res)):
        xs, t_ms = stp._extract_trial_slice(x, c, (70.0, 8000.0), FPS)
        ys, _ = stp._extract_trial_slice(y, c, (70.0, 8000.0), FPS)
        l12 = stp._extract_lick1_lick2(ys, xs, t_ms, FPS, params=params, v7_trial_result=r)
        vel = stp._extract_peak_velocity_first5(ys, xs, t_ms, FPS, params=params, v7_trial_result=r)
        nl = stp._extract_n_licks_family(r.y_clean, r.x_clean, t_ms, FPS, params=params)
        ldr = stp._extract_licking_dynamics_reductions(ys, xs, t_ms, FPS, params=params, kept_licks=r.kept_licks)
        own = sang._extract_cell50_own_bout_scalars(ys, xs, t_ms, FPS, params=params, kept_licks=r.kept_licks)
        a1 = sang._extract_angle_at_lick_timestamps(angle, FPS, c, l12["lick1_t_peak_ms"])
        a2 = sang._extract_angle_at_lick_timestamps(angle, FPS, c, l12["lick2_t_peak_ms"])
        av = [sang._extract_angle_at_lick_timestamps(angle, FPS, c, t) for t in vel["lick_t_peaks_bylick"]]
        am = sang._extract_angle_max_signed_per_lick(angle, FPS, c, vel["lick_t_peaks_bylick"], 52.0)
        rows.append(dict(l12=l12, vel=vel, nl=nl, ldr=ldr, own=own, a1=a1, a2=a2, av=av, am=am))
        lo_, hi_ = params["dlc_kinematics"]["tongue_pertrial"]["licking_dyn_apply_win_ms"]
        by_pos[tk.POSITIONS[k % 6]].append(np.asarray(sorted(
            lk_["t_ms"] for lk_ in r.kept_licks if lo_ <= lk_["t_ms"] <= hi_), dtype=float))
        _, b, _ = stp._build_visual_bundle_per_trial(
            k, "L", x_final=xb, y_final=yb, angle_per_frame=angle, tone_frame_idx=c, fps=FPS,
            plot_win_ms=(-180.0, 8000.0), detect_win_ms=(70.0, 3000.0), first_peak_align_win_ms=(-180.0, 1000.0),
            detector_params=params["dlc_kinematics"]["tongue_pertrial"]["detector"], max_ili_ms=300.0,
            single_bout_win_ms=(-120.0, 200.0), max_licks_per_trial=40, phase_bin_edges=edges,
            phase_bin_centers=0.5 * (edges[:-1] + edges[1:]), kept_licks=r.kept_licks)
        bouts += [dict(bb, trial_id=k) for bb in b if np.isnan(bb["peak_t_ms"])]
    traces = stp._extract_licking_dynamics_traces(by_pos, params=params)
    return res, rows, bouts, traces, angle, (xb, yb)


def _same(a, b, exact=True):
    if exact:
        np.testing.assert_array_equal(np.asarray(a, float), np.asarray(b, float))
    else:
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-9)


def test_parity_full_v7_path_and_features(source):
    """Ten randomised sessions: kept licks, every per-trial feature, bout rows, traces and angles equal theirs.

    Angles are compared to 1e-9 not exactly: their reference vector is (-eye) and ours the unit (-ap_axis), the
    same direction at a different length, which atan2 ignores only up to rounding. Everything else is exact.
    Also asserts that the random sessions actually reached the branches worth checking (imputation, every
    pre-clean rule, gate M), so a pass is not a pass over trivially clean data.
    """
    eye = (5.0, -60.0)
    frame = tk.SpoutFrame(origin=eye, ap_axis=eye)          # theirs: ref = eye -> origin(0,0) = -eye direction
    hits = dict(imputed=0, artifact=0, frame_out=0, x_joint=0, x_hard=0, M=0, D=0, F=0, legacy=0)
    for seed in range(10):
        y, x, fm, lk, cues, _ = synth_session(seed)
        tres, trows, tbouts, ttraces, tangle, (txb, tyb) = _theirs(source, y, x, fm, lk, cues, eye)
        ours = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS, X0=0.0, Y0=0.0, spout_frame=frame)
        _same(ours.x_clean, txb)
        _same(ours.y_clean, tyb)
        _same(ours.angle_per_frame, tangle, exact=False)
        pt = ours.per_trial
        for k, (tr, orr) in enumerate(zip(tres, ours.trial_results)):
            assert len(tr.kept_licks) == len(orr.kept_licks)
            for a, b in zip(orr.kept_licks, tr.kept_licks):
                assert set(b) <= set(a)
                for f in b:
                    if f in ("source", "imputed", "lick_idx", "rise_start_frame", "fall_end_frame"):
                        assert a[f] == b[f], f
                    else:
                        _same(a[f], b[f], exact=(f != "peak_angle_deg"))
            row, t = pt.iloc[k], trows[k]
            for f, v in t["l12"].items():
                _same(row[f], v)
            for j in range(5):
                _same(row[f"vy_peak_lick{j + 1}"], t["vel"]["vy_peaks_bylick"][j])
                _same(row[f"vxy_peak_lick{j + 1}"], t["vel"]["vxy_peaks_bylick"][j])
                _same(row[f"t_peak_lick{j + 1}_ms"], t["vel"]["lick_t_peaks_bylick"][j])
                _same(row[f"angle_at_vpeak_lick{j + 1}"], t["av"][j], exact=False)
                _same(row[f"angle_max_signed_lick{j + 1}"], t["am"][j], exact=False)
                assert row[f"n_licks_bin{j + 1}"] == t["nl"]["n_licks_per_bin"][j]
            assert (row.n_licks, row.n_img_right, row.n_img_left, row.n_other, row.lick_ge10) == (
                t["nl"]["n_licks"], t["nl"]["n_left"], t["nl"]["n_right"], t["nl"]["n_other"], t["nl"]["lick_ge10"])
            for f, v in {**t["ldr"], **t["own"]}.items():
                _same(row[f], v)
            _same(row.lick1_angle_at_ypeak, t["a1"], exact=False)
            _same(row.lick2_angle_at_ypeak, t["a2"], exact=False)
            d = orr.preclean_diag
            hits["artifact"] += d.n_artifact_clusters
            hits["frame_out"] += d.n_frame_outliers
            hits["x_joint"] += d.n_x_joint_outliers
            hits["x_hard"] += d.n_x_hard_outliers
            hits["imputed"] += sum(lk_["imputed"] for lk_ in orr.kept_licks)
            hits["M"] += sum(dd["gate"] == "M" for dd in orr.decisions)
            hits["D"] += sum(dd["gate"] == "D" for dd in orr.decisions)
            hits["F"] += sum(dd["gate"] == "F" for dd in orr.decisions)
            hits["legacy"] += sum(lk_["source"] == "legacy" for lk_ in orr.kept_licks)
        ob = ours.bouts
        assert len(ob) == len(tbouts)
        for (_, a), b in zip(ob.iterrows(), tbouts):
            assert a.trial_id == b["trial_id"] and a.bout_idx == b["bout_idx"] and a.n_peaks == b["n_peaks"]
            for f in ("multi_bout_rank", "bout_t0_ms", "bout_t1_ms", "peak_t0_ms", "peak_t1_ms", "peak_y0", "peak_y1"):
                _same(a[f], b[f])
            for f in ("peak_angle0", "peak_angle1"):
                _same(a[f], b[f], exact=False)
        tt = ttraces.rename(columns={"side": "position"})
        cols = ["timebin_ms", "lick_rate_per_t", "ili_within_bout_rate_per_t", "p_lick_per_t", "p_within_bout_per_t"]
        _same(ours.licking_dynamics_traces[cols].to_numpy(), tt[cols].to_numpy())
        assert ours.licking_dynamics_traces.position.tolist() == tt.position.tolist()
    # (gate E's "drop" outcome -- imputed peak below min_peak_y_abs / no anchors -- is not reached here; the
    # imputers' None paths are covered by the gate-level parity in test_tongue_detect.)
    for k in hits:
        assert hits[k] > 0, f"parity sessions never exercised {k}: {hits}"


# =========================================================================== OURS: geometry / phase / velocity

def test_lick_extent_and_spout_coords():
    y = np.array([0, 0, 10, 30, 60, 80, 70, 40, 20, 0, 0], float)
    base = y == 0
    assert tk.lick_extent(y, base, 5) == (2, 8)                      # visible run around the peak
    f = tk.SpoutFrame(origin=(300.0, 100.0), ap_axis=(0.0, -1.0))
    ap, lr = tk.spout_coords(np.array([300.0, 400.0]), np.array([200.0, 200.0]), f)
    np.testing.assert_allclose(ap, [100, 100])
    np.testing.assert_allclose(lr, [0, 100])                         # image-right = + lr
    np.testing.assert_allclose(np.degrees(np.arctan2(lr, ap)),
                               tk.compute_signed_tongue_angle_deg(np.array([300.0, 400.0]), np.array([200.0, 200.0]), f))


def test_visible_rise_velocity_geometry_and_phase_table():
    y, x, fm, lk, cues, _truth = synth_session(seed=3, extras=False)
    X0, Y0 = 320.0, 250.0
    frame = tk.SpoutFrame(origin=(X0, Y0 - 30.0), ap_axis=(0.0, -1.0))
    ov = {"velocity_window": "visible_rise", "lick_geometry": True, "fix_detect_offset": True}
    old = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS, X0=X0, Y0=Y0, spout_frame=frame)
    new = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS, X0=X0, Y0=Y0, spout_frame=frame,
                                       params_override=ov)
    pl = new.per_lick
    assert {"on_frame", "off_frame", "protrusion_px", "ap_px", "lr_px", "max_retract_velocity_y_px_per_s"} <= set(pl.columns)
    assert "protrusion_px" not in old.per_lick.columns and old.lick_phase is None
    assert (pl.on_ms <= pl.t_ms).all() and (pl.off_ms >= pl.t_ms).all() and (pl.on_ms < pl.t_ms).mean() > 0.9
    np.testing.assert_allclose(pl.protrusion_px, np.hypot(pl.ap_px, pl.lr_px))
    np.testing.assert_allclose(pl.ap_px, pl.y + 30.0, atol=1e-9)    # mouth 30 px above the zero, axis straight down
    # the whole rise sees the fastest part of the lick, the inherited window may not
    vn, vo = pl.max_velocity_y_px_per_s, old.per_lick.max_velocity_y_px_per_s
    ok = vn.notna() & vo.notna()                                     # NaN: a lick first visible at its peak
    assert ok.mean() > 0.9 and (vn[ok] >= vo[ok] - 1e-6).mean() > 0.9   # a gap mid-rise can shorten a rise
    ph = new.lick_phase
    assert set(np.round(ph.phase, 6)) == set(np.round(np.linspace(0, 1, 21), 6))
    peak = ph[np.isclose(ph.phase, 0.5)].set_index(["trial_id", "lick_idx"]).protrusion_px
    edge = ph[np.isclose(ph.phase, 0.0)].set_index(["trial_id", "lick_idx"]).protrusion_px
    assert (peak.dropna() > edge.reindex(peak.dropna().index)).all()


def test_angle_max_signed_within_lick_ignores_the_next_lick():
    ang = np.zeros(200)
    ang[95:106] = 5.0                                                # this lick: +5 deg
    ang[110:120] = -40.0                                             # the NEXT lick, inside +-13 frames of 100
    t = np.array([(100 - 50) / FPS * 1000.0])                        # lick peak at frame 100, cue at 50
    assert tk.angle_max_signed_per_lick(ang, FPS, 50, t, 52.0)[0] == -40.0
    assert tk.angle_max_signed_per_lick(ang, FPS, 50, t, 52.0, extents=[(95, 106)])[0] == 5.0


def test_contacts_spout_reference_and_reach_accuracy():
    """OURS: per-lick DAQ contact, per-trial spout tip, delta angle / tip-to-spout / reach fraction."""
    y, x, fm, lk, cues, _truth = synth_session(seed=3, extras=False)
    X0, Y0 = 320.0, 250.0
    frame = tk.SpoutFrame(origin=(X0, Y0), ap_axis=(0.0, -1.0))          # mouth at the zero, spouts straight down
    n = len(y)
    spout = (np.full(n, X0 + 100.0), np.full(n, Y0 + 100.0), np.ones(n))   # spout tip at +45 deg, 141 px out
    tr = _trials(cues)
    ov = {"lick_geometry": True, "fix_detect_offset": True}
    base = tk.compute_tongue_kinematics(x, y, fm, lk, tr, fps=FPS, X0=X0, Y0=Y0, spout_frame=frame, params_override=ov)
    first = {t.trial_id: base.per_lick[base.per_lick.trial_id == t.trial_id].t_ms.iloc[0] for t in tr}
    contacts = {k: np.array([t + 20.0]) for k, t in first.items()}         # one contact, 20 ms after lick 1
    r = tk.compute_tongue_kinematics(x, y, fm, lk, tr, fps=FPS, X0=X0, Y0=Y0, spout_frame=frame, params_override=ov,
                                     spout_xy=spout, contacts_ms=contacts)
    pl, pt = r.per_lick, r.per_trial
    assert (pl.groupby("trial_id").contact.sum() == 1).all()               # only lick 1 of each trial
    assert (pt.n_daq_contacts == 1).all() and (pt.n_licks_contact == 1).all()
    np.testing.assert_allclose(pt.spout_angle_deg, 45.0)
    np.testing.assert_allclose(pt.spout_dist_px, np.hypot(100, 100))
    ang = np.degrees(np.arctan2(pl.lr_px, pl.ap_px))
    np.testing.assert_allclose(pl.delta_angle_deg, ang - 45.0)
    np.testing.assert_allclose(pl.tip_to_spout_px, np.hypot(pl.ap_px - 100, pl.lr_px - 100))
    np.testing.assert_allclose(pl.reach_frac, pl.protrusion_px / np.hypot(100, 100))
    np.testing.assert_allclose(r.lick_phase.delta_angle_deg, r.lick_phase.angle_deg - 45.0)


def test_detect_on_protrusion_keeps_image_geometry():
    """OURS: detection on the tongue-mouth distance finds the same licks on near-straight licks, and the
    per-lick y is then the distance while angles / ap / lr still come from image coordinates."""
    y, x, fm, lk, cues, _truth = synth_session(seed=3, extras=False)
    X0, Y0 = 320.0, 250.0
    frame = tk.SpoutFrame(origin=(X0, Y0), ap_axis=(0.0, -1.0))
    ov = {"lick_geometry": True, "fix_detect_offset": True}
    a = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS, X0=X0, Y0=Y0, spout_frame=frame,
                                     params_override=ov)
    b = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS, X0=X0, Y0=Y0, spout_frame=frame,
                                     params_override={**ov, "detect_on": "protrusion"})
    assert abs(len(a.per_lick) - len(b.per_lick)) <= 1
    m = a.per_lick.merge(b.per_lick, on=["trial_id", "t_ms"], suffixes=("_y", "_d"))
    assert len(m) >= len(a.per_lick) - 1
    np.testing.assert_allclose(m.y_d, np.hypot(m.x_d, m.y_image), rtol=1e-6)     # y = distance from the mouth
    np.testing.assert_allclose(m.y_image, m.y_y)                                    # image y unchanged
    np.testing.assert_allclose(m.peak_angle_deg_d, m.peak_angle_deg_y)
    with pytest.raises(ValueError):
        tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS, params_override={"detect_on": "protrusion"})


def test_centered_phase_is_real_time_around_the_peak():
    """OURS (stroke_orofacial's centered phase): phase 0.5 = the peak frame, the window spans cycle_ms in real
    time, and the default cycle is the session's median within-bout ILI."""
    y, x, fm, lk, cues, _truth = synth_session(seed=3, extras=False)
    X0, Y0 = 320.0, 250.0
    frame = tk.SpoutFrame(origin=(X0, Y0), ap_axis=(0.0, -1.0))
    ov = {"lick_geometry": True, "fix_detect_offset": True, "phase": {"mode": "centered", "n_points": 41}}
    r = tk.compute_tongue_kinematics(x, y, fm, lk, _trials(cues), fps=FPS, X0=X0, Y0=Y0, spout_frame=frame,
                                     params_override=ov)
    ph = r.lick_phase
    ili = tk.within_bout_ili_ms(r.trial_results)
    assert np.allclose(ph.cycle_ms, np.median(ili)) and 100 < np.median(ili) < 300
    peak = ph[np.isclose(ph.phase, 0.5)].merge(r.per_lick[["trial_id", "lick_idx", "protrusion_px"]],
                                               on=["trial_id", "lick_idx"], suffixes=("", "_lick"))
    np.testing.assert_allclose(peak.protrusion_px, peak.protrusion_px_lick, atol=1e-6)   # phase .5 = the peak
    fixed = tk.lick_phase_table(r.trial_results, X0, Y0, frame, FPS, n_points=41, mode="centered", cycle_ms=120.0)
    assert np.allclose(fixed.cycle_ms, 120.0) and len(fixed) == len(ph)
