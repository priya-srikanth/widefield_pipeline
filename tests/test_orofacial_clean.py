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


def test_snippet_pads_and_mean_sem():
    a = np.arange(10.0)
    assert np.isnan(oc.snippet(a, 1, 3, 2)[:2]).all() and oc.snippet(a, 1, 3, 2)[2] == 0.0
    m, s = oc.mean_sem(np.array([[1.0, np.nan], [3.0, np.nan]]))
    assert m[0] == 2.0 and np.isnan(m[1]) and np.isclose(s[0], 1.0 / np.sqrt(2))
