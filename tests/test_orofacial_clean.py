"""orofacial_clean: each ported stage does what the stroke_orofacial v5p3 / v3.4 code does, on synthetic tracks."""
import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc

P = {"x_range_pm": 150.0, "x_range_source": "lk", "lk_thr": 0.6, "n_neigh": 2, "neighbor_ms": 16.0,
     "max_gap_ms": 56.0, "interp_n_side": 6, "interp_min_total_context": 6, "require_bracketed_by_hilk": True,
     "interpolation_method": "pchip", "interpolation_fallback": "linear",
     "baseline": {"force": False, "y0_lk_thr": 0.95, "y0_percentile": 5.0, "pool_win_ms": [0.0, 5000.0]}}


def _df(x, y, lk, part="tongue"):
    return pd.DataFrame({f"{part}_x": x, f"{part}_y": y, f"{part}_likelihood": lk})


def test_xrange_drops_far_points_and_keeps_near():
    x = np.full(50, 300.0)
    x[10] = 600.0
    xo, yo = oc._stage_xrange(x, np.ones(50), np.ones(50), x_range_pm=150, x_range_source="lk", lk_thr=0.6)
    assert np.isnan(xo[10]) and np.isnan(yo[10]) and np.isfinite(xo[11])


def test_isolated_confident_point_is_dropped():
    x = np.full(40, np.nan)
    x[5] = 1.0                                   # alone
    x[20:25] = 2.0                               # a run: each has >= 2 neighbours within +-4 frames
    xo, _ = oc._stage_drop(x, x.copy(), n_neigh=2, neighbor_f=4)
    assert np.isnan(xo[5]) and np.all(np.isfinite(xo[20:25]))


def test_short_gap_filled_long_gap_not():
    n = 200
    y = np.sin(np.arange(n) / 10.0) * 50 + 400
    x = np.full(n, 300.0)
    lk = np.ones(n)
    lk[50:55] = 0.1                               # 5 frames = 20 ms  -> filled (<= 56 ms = 14 f)
    lk[120:150] = 0.1                             # 30 frames = 120 ms -> not filled -> baseline
    c = oc.clean_bodypart(_df(x, y, lk), "tongue", centers=[0], p=P)
    assert np.all(np.isin(c.fill_method[50:55], (oc.FILL_PCHIP, oc.FILL_LINEAR)))
    assert np.allclose(c.y_interp[50:55], y[50:55], atol=1.0)          # PCHIP follows the sine
    assert np.all(c.fill_method[120:150] == oc.FILL_BASELINE)
    assert np.all(np.isnan(c.y_masked[120:150])) and np.all(c.y_final[120:150] == 0.0)


def test_baseline_is_lowest_y_of_confident_pool():
    n = 1500
    y = np.full(n, 450.0)
    y[::7] = 380.0                                # tongue in / jaw closed: smallest y
    c = oc.clean_bodypart(_df(np.full(n, 300.0), y, np.ones(n)), "tongue", centers=[10], p=P)
    assert c.Y0 == 380.0 and c.X0 == 300.0
    assert c.y_final.min() == 0.0


def test_lp_image_centre_points_never_survive():
    n = 300
    y = np.full(n, 450.0)
    x = np.full(n, 300.0)
    lk = np.ones(n)
    y[100:200], x[100:200], lk[100:200] = 338.0, 338.0, 0.0     # LP "hidden" = image centre, p ~ 0
    c = oc.clean_bodypart(_df(x, y, lk), "tongue", centers=[0], p=P)
    assert np.all(np.isnan(c.y_masked[100:200]))


def test_jaw_v34_y_ceiling_and_isolated():
    n = 1000
    rng = np.random.default_rng(0)
    y = 420 + rng.normal(0, 1, n)
    x = 350 + rng.normal(0, 1, n)
    lk = np.ones(n)
    y[500] = 700.0                                 # far above any ceiling relative to the cue baseline
    lk[300:310] = 0.1
    lk[311:400] = 0.1                              # leaves frame 310 as an isolated 1-frame cluster
    q = {"y_ceiling_px": 125.0, "x_ceiling_px": 40.0, "isolated_max_n": 5, "isolated_radius_ms": 30.0,
         "frame_jump_thr_px": 70.0, "frame_outlier_radius_ms": 20.0, "max_gap_ms_pchip": 120.0,
         "baseline_win_ms": [-500.0, 0.0]}
    pj = {**P, "n_neigh": 0, "max_gap_ms": 4.0}    # keep raw frames raw so v3.4 sees them
    c = oc.jaw_v34(oc.clean_bodypart(_df(x, y, lk, "jaw"), "jaw", centers=[200, 600, 900], p=pj), [200, 600, 900], q)
    assert c.extra["v34_wiped_y"][500] or c.extra["v34_wiped_frame_outlier"][500]
    assert c.extra["v34_wiped_isolated"][310]
    assert c.fill_method[500] != oc.FILL_NONE


def test_jaw_v34_x_frame_outlier_keeps_smooth_excursion():
    """A smooth 60-px lateral excursion survives the x jump test; a 3-frame side-flip of the same size does not."""
    n = 1000
    rng = np.random.default_rng(1)
    y = 420 + rng.normal(0, 0.5, n)
    x = 350 + rng.normal(0, 0.5, n)
    x[400:500] += 60 * np.sin(np.linspace(0, np.pi, 100))      # real: ~2 px/frame at most
    x[700:703] += 60                                            # flip: one-frame jump in and out
    lk = np.ones(n)
    q = {"y_ceiling_px": 125.0, "x_ceiling_px": 1e9, "isolated_max_n": 5, "isolated_radius_ms": 30.0,
         "frame_jump_thr_px": 70.0, "frame_outlier_radius_ms": 20.0, "frame_jump_thr_x_px": 25.0,
         "max_gap_ms_pchip": 120.0, "baseline_win_ms": [-500.0, 0.0]}
    pj = {**P, "n_neigh": 0, "max_gap_ms": 4.0, "x_range_pm": 1e9}
    c = oc.jaw_v34(oc.clean_bodypart(_df(x, y, lk, "jaw"), "jaw", centers=[200, 600, 900], p=pj), [200, 600, 900], q)
    wx = c.extra["v34_wiped_frame_outlier_x"]
    assert not wx[400:500].any()
    assert wx[700:703].all()
    # a 9-frame flip (longer than half the 11-frame median window) needs the bracketed-step rule
    x2 = x.copy()
    x2[800:809] -= 55
    c2 = oc.jaw_v34(oc.clean_bodypart(_df(x2, y, lk, "jaw"), "jaw", centers=[200, 600, 900], p=pj), [200, 600, 900],
                    {**q, "x_flip_step_px": 30.0, "x_flip_max_ms": 200.0})
    w2 = c2.extra["v34_wiped_flip_x"]
    assert c2.fill_method[804] == oc.FILL_NONE and abs(c2.x_final[804] - c2.x_final[795]) < 5   # y kept, x refilled
    assert w2[800:809].all() and not w2[400:500].any() and not w2[795:800].any() and not w2[809:815].any()
    # flanks OFF the resting jaw (both on the wrong landmark): the run is spared
    x3 = x.copy()
    x3[780:830] -= 50                                           # long wrong-side stretch ...
    x3[800:809] += 50                                           # ... with a short return to the right point inside
    c3 = oc.jaw_v34(oc.clean_bodypart(_df(x3, y, lk, "jaw"), "jaw", centers=[200, 600, 900], p=pj), [200, 600, 900],
                    {**q, "x_flip_step_px": 30.0, "x_flip_max_ms": 200.0})
    assert not c3.extra["v34_wiped_flip_x"][801:808].any()
    q_off = {k: v for k, v in q.items() if k != "frame_jump_thr_x_px"}
    c0 = oc.jaw_v34(oc.clean_bodypart(_df(x, y, lk, "jaw"), "jaw", centers=[200, 600, 900], p=pj), [200, 600, 900],
                    q_off)
    assert not c0.extra["v34_wiped_frame_outlier_x"].any()


def test_interlick_gap_gets_mouth_fill_but_rod_gap_keeps_interpolation():
    """Retract -> gap -> protrude near the mouth = between licks (mouth fill); a gap out at the spout is interpolated."""
    n = 400
    x = np.full(n, 300.0)
    y = np.full(n, np.nan)
    lk = np.zeros(n)
    # lick 1 retracting into the mouth (r 60 -> 50), gap 10 frames, lick 2 protruding (50 -> 60): mouth at (300, 300)
    y[100:110] = 300 + np.linspace(80, 50, 10)
    y[120:130] = 300 + np.linspace(50, 80, 10)
    # tongue out at the spout (r ~150) on both sides of a gap, still retract-then-protrude in radius
    y[200:210] = 300 + np.linspace(160, 150, 10)
    y[220:230] = 300 + np.linspace(150, 160, 10)
    lk[np.isfinite(y)] = 1.0
    y = np.where(np.isfinite(y), y, 300.0)
    pt = {**P, "n_neigh": 0, "max_gap_ms": 60.0, "interp_veto_interlick": True, "interlick_k": 2,
          "interlick_speed_px": 3.0, "interlick_near_px": 80.0}
    c = oc.clean_bodypart(_df(x, y, lk), "tongue", centers=[0], p=pt, x0y0=(300.0, 300.0))
    assert (c.fill_method[110:120] == oc.FILL_BASELINE).all()
    assert (c.fill_method[210:220] != oc.FILL_BASELINE).all()
    c0 = oc.clean_bodypart(_df(x, y, lk), "tongue", centers=[0], p={**pt, "interp_veto_interlick": False},
                           x0y0=(300.0, 300.0))
    assert (c0.fill_method[110:120] != oc.FILL_BASELINE).all()


def test_jaw_on_far_tongue_tip_is_made_invisible():
    n = 50
    df = pd.DataFrame({"jaw_x": np.full(n, 350.0), "jaw_y": np.full(n, 420.0), "jaw_likelihood": np.ones(n),
                       "tongue_x": np.full(n, 350.0), "tongue_y": np.full(n, 430.0), "tongue_likelihood": np.ones(n)})
    df.loc[10, ["jaw_x", "jaw_y", "tongue_x", "tongue_y"]] = [360.0, 500.0, 365.0, 505.0]   # riding a far tongue
    df.loc[20, ["jaw_x", "jaw_y"]] = [352.0, 425.0]                                         # near a near tongue: kept
    q = {"tongue_ride_px": 25.0, "tongue_out_px": 60.0}
    out = oc.jaw_off_tongue(df, q)
    assert out.jaw_likelihood[10] == 0.0 and out.jaw_likelihood[20] == 1.0 and (out.jaw_likelihood != 0).sum() == n - 1
    assert oc.jaw_off_tongue(df, {"tongue_ride_px": None}) is df


def test_snippet_pads_and_mean_sem():
    a = np.arange(10.0)
    assert np.isnan(oc.snippet(a, 1, 3, 2)[:2]).all() and oc.snippet(a, 1, 3, 2)[2] == 0.0
    m, s = oc.mean_sem(np.array([[1.0, np.nan], [3.0, np.nan]]))
    assert m[0] == 2.0 and np.isnan(m[1]) and np.isclose(s[0], 1.0 / np.sqrt(2))
