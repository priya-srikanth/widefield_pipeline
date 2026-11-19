"""`camera_licks`: fusing per-camera lick events (one lick per overlapping span, peak by camera priority, views)."""
import numpy as np
import pandas as pd

from wfield_local import camera_licks as CL


def _ev(cam, rows):
    return pd.DataFrame([{"trial_id": t, "position": "far_L", "on_ms": a, "off_ms": b, "t_peak_ms": p, "cam": cam}
                         for t, a, b, p in rows])


def test_fuse_merges_overlaps_and_keeps_priority_peak():
    ev = {"cam1": _ev("cam1", [(1, 100, 160, 140), (1, 400, 430, 415)]),
          "cam4": _ev("cam4", [(1, 110, 170, 150)]),
          "cam2": _ev("cam2", [(1, 175, 200, 190), (1, 600, 640, 620), (2, 100, 150, 120)]),
          "cam3": _ev("cam3", [])}
    U = CL.fuse(ev)
    t1 = U[U.trial_id == 1]
    assert list(t1.on_ms) == [100, 400, 600]                 # cam2 175 is within 12 ms of the 170 end -> same lick
    first = t1.iloc[0]
    assert first.off_ms == 200 and first.t_peak_ms == 150      # union span; cam4's peak wins
    assert first.cams == "cam1+cam2+cam4" and first.n_views == 3
    assert t1.iloc[2].cams == "cam2" and not t1.iloc[2].seen_cam4
    assert list(U[U.trial_id == 2].lick_idx) == [0]            # trials never chain into each other
    assert len(CL.lick_train(U, min_views=2)) == 1


def test_fuse_chains_through_a_bridging_event_and_respects_tol():
    ev = {"cam4": _ev("cam4", [(1, 0, 50, 20), (1, 90, 120, 100)]), "cam2": _ev("cam2", [(1, 45, 95, 70)])}
    assert len(CL.fuse(ev)) == 1                               # cam2 bridges the two cam4 licks
    ev = {"cam4": _ev("cam4", [(1, 0, 50, 20), (1, 63, 90, 70)])}
    assert len(CL.fuse(ev)) == 2 and len(CL.fuse(ev, tol_ms=13)) == 1


def test_fuse_empty_and_view_table():
    assert len(CL.fuse({"cam4": _ev("cam4", [])})) == 0
    U = CL.fuse({"cam4": _ev("cam4", [(1, 0, 40, 20)]), "cam3": _ev("cam3", [(1, 200, 240, 220)])})
    V = CL.view_table(U)
    assert V.loc["cam3", "side_only"] == 1 and V.loc["cam4", "side_only"] == 0 and np.isclose(V.share.sum(), 1)


def test_within_frames_keeps_clip_span():
    F = pd.DataFrame({"trial_id": [1, 1, 2], "t_ms": [-200.0, 1500.0, 0.0]})
    E = _ev("cam4", [(1, -5000, -4900, -4950), (1, 1400, 1600, 1500), (3, 0, 50, 20)])
    assert list(CL.within_frames(E, F).on_ms) == [1400]


def test_static_events_count_only_when_another_camera_confirms():
    s2 = _ev("cam2", [(1, 100, 112, 104), (1, 500, 512, 504)]).assign(static=True)
    U = CL.fuse({"cam4": _ev("cam4", [(1, 90, 150, 120)]), "cam2": s2})
    assert list(U.static) == [False, True]                    # held on the spout with cam4 / a static label alone
    assert list(CL.lick_train(U).on_ms) == [90]
    assert CL.view_table(U).loc["cam2", "static"] == 1


def test_droplet_rule_scales_to_another_view():
    from wfield_local import cam1_licks as C1
    E = pd.DataFrame({"spread_px": [5.0], "range_px": [20.0], "med_out_px": [10.0]})
    assert not C1.droplet_flags(E, [True])[0] and C1.droplet_flags(E, [True], scale=2.0)[0]
