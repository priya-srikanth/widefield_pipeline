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
    assert list(U.guarded) == [False, True]                    # held on the spout with cam4 / a static label alone
    assert list(CL.lick_train(U).on_ms) == [90]
    assert CL.view_table(U).loc["cam2", "guarded"] == 1


def test_droplet_rule_scales_to_another_view():
    from wfield_local import cam1_licks as C1
    E = pd.DataFrame({"spread_px": [5.0], "range_px": [20.0], "med_out_px": [10.0]})
    assert not C1.droplet_flags(E, [True])[0] and C1.droplet_flags(E, [True], scale=2.0)[0]


def test_far_guard_counts_like_static():
    s3 = _ev("cam3", [(1, 500, 540, 520)]).assign(far=True)
    U = CL.fuse({"cam3": s3, "cam2": _ev("cam2", [(1, 505, 545, 525)])})
    assert list(U.guarded) == [False] and len(CL.lick_train(U)) == 1     # confirmed by cam2 -> a lick
    assert len(CL.lick_train(CL.fuse({"cam3": s3}))) == 0


# ---- 11-21 pilot: splitting back-to-back licks and D2 as revised (any camera; long cam1-only dropped) ----

def _frames(trial, t0, r_mm):
    """A pose3d-like frames table: one trial, 4 ms frames from t0, tongue straight out (ml = distance)."""
    r = np.asarray(r_mm, float)
    t = t0 + 4.0 * np.arange(len(r))
    return pd.DataFrame({"trial_id": trial, "position": "far_L", "t_ms": t, "stop_ms": 5000.0,
                         "tongue_ml": r, "tongue_ap": np.where(np.isnan(r), np.nan, 0.0),
                         "tongue_dv": np.where(np.isnan(r), np.nan, 0.0),
                         "tongue_src": 1, "tongue_rep1": 1.0, "tongue_rep4": 1.0})


def test_bridged_licks_stay_merged_without_cuts_and_split_with_event_cuts():
    ev = {"cam4": _ev("cam4", [(1, 0, 50, 20), (1, 90, 120, 100)]), "cam2": _ev("cam2", [(1, 45, 95, 70)])}
    assert len(CL.fuse(ev)) == 1                               # chaining alone: cam2 bridges the two cam4 licks
    cuts = CL.event_cuts(ev)
    np.testing.assert_allclose(cuts[1], [70.0])                # middle of the gap 50 .. 90
    U = CL.fuse(ev, cuts=cuts)
    assert list(U.t_peak_ms) == [20, 100] and list(U.off_ms) == [50, 120]
    assert list(U.cams) == ["cam4", "cam2+cam4"]               # cam2's peak (70) falls in the second piece
    assert U.iloc[1].on_ms == 70                               # its span clipped at the cut


def test_event_cuts_handle_overlapping_cam4_spans_and_cam1_precedence():
    ev = {"cam4": _ev("cam4", [(1, 312, 388, 320), (1, 344, 464, 432)]),       # v36 spans overlap (PS93 0814)
          "cam1": _ev("cam1", [(1, 300, 330, 310), (1, 340, 360, 350), (1, 600, 620, 610), (1, 700, 720, 710)])}
    c = CL.event_cuts(ev)[1]
    assert np.isclose(c, 366.0).any()                          # middle of the 344 .. 388 overlap, between the peaks
    assert not ((c > 312) & (c < 366)).any()                   # cam1's 335 cut sits inside a cam4 lick -> dropped
    assert np.isclose(c, 660.0).any()                          # cam1 cut where cam4 saw nothing
    assert len(CL.fuse(ev)) == 3 and len(CL.fuse(ev, cuts=CL.event_cuts(ev))) == 4


def test_event_cuts_move_to_the_tongue_return_and_skip_a_held_tongue():
    ev = {"cam4": _ev("cam4", [(1, 0, 40, 20), (1, 60, 120, 100)])}
    r = [1, 2, 3, 4, 3, 3, 3, 2, 1, 0.5, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3.5, 4, 3]  # 0 .. 104 ms; minimum at 36 ms
    c = CL.event_cuts(ev, S=_frames(1, 0.0, r))[1]
    np.testing.assert_allclose(c, [35.5])                      # not the span midpoint (50): the return toward the mouth
    held = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]    # no return: the minimum is the first peak's frame
    c2 = CL.event_cuts({"cam4": _ev("cam4", [(1, 0, 8, 8), (1, 52, 70, 60)])}, S=_frames(1, 0.0, held))[1]
    assert c2[0] <= 8                                          # cut at the track edge -> no split inside the run
    gap = r[:8] + [np.nan] * 3 + r[11:]                        # track broken between the peaks -> cut in the break
    c3 = CL.event_cuts(ev, S=_frames(1, 0.0, gap).dropna(subset=["tongue_ml"]))[1]
    assert 28 < c3[0] < 44


def test_lick_peaks_splits_a_run_at_cuts():
    from wfield_local import pose3d as P3
    rig = {"geometry": None}
    S = _frames(1, 0.0, [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2])
    import wfield_local.spout_world as SW
    orig = SW.commanded_world
    SW.commanded_world = lambda geo, p: np.array([5.0, 0.0, 0.0])
    try:
        assert len(P3.lick_peaks(S, rig)) == 1
        # peaks 24 ms apart: below the physiological floor, so only split with the floor off (min_ili_ms=0)
        L = P3.lick_peaks(S, rig, cuts={1: np.array([23.5])}, min_ili_ms=0.0)
        assert list(L.t_onset_ms) == [0.0, 24.0] and list(L.t_peak_ms) == [12.0, 36.0]
        assert len(P3.lick_peaks(S, rig, cuts={1: np.array([23.5])})) == 1      # default 60 ms floor merges them
    finally:
        SW.commanded_world = orig


def test_lick_peaks_floor_merges_a_dropout_fragment_and_keeps_real_licks():
    """11-21 (Priya): the 60 ms video floor merges two runs of one lick split by a one-frame dropout (peaks < 60 ms
    apart), measured over the union of their frames, and never merges two real licks (peaks >= 60 ms apart)."""
    from wfield_local import pose3d as P3
    rig = {"geometry": None}
    import wfield_local.spout_world as SW
    orig = SW.commanded_world
    SW.commanded_world = lambda geo, p: np.array([5.0, 0.0, 0.0])
    try:
        frag = [1, 2, 3, 4, np.nan, 4.5, 3, 2, 1]                  # dropout at 16 ms; peaks 12 / 20 ms -> one lick
        S = _frames(1, 0.0, frag).dropna(subset=["tongue_ml"])
        assert len(P3.lick_peaks(S, rig, min_ili_ms=0.0)) == 2
        L = P3.lick_peaks(S, rig)
        assert len(L) == 1 and L.t_peak_ms.iloc[0] == 20.0 and L.reach_mm.iloc[0] == 4.5
        two = [1, 2, 3, 4, 3, 2, 1] + [np.nan] * 4 + [1, 2, 3, 4, 3, 2, 1]   # peaks 12 / 56 ms (44 ms apart)
        S2 = _frames(1, 0.0, two).dropna(subset=["tongue_ml"])
        assert len(P3.lick_peaks(S2, rig)) == 1                          # under the 60 ms floor -> one lick
        assert len(P3.lick_peaks(S2, rig, min_ili_ms=40.0)) == 2
        two = [1, 2, 3, 4, 3, 2, 1] + [np.nan] * 9 + [1, 2, 3, 4, 3, 2, 1]   # peaks 12 / 76 ms (64 ms apart) -> two licks
        S3 = _frames(1, 0.0, two).dropna(subset=["tongue_ml"])
        assert len(P3.lick_peaks(S3, rig)) == 2
        assert P3.POSE3D_MIN_ILI_MS == 60.0 and CL.MIN_ILI_MS == 60.0
    finally:
        SW.commanded_world = orig


def test_mouth_cuts_prominence_and_ili_floor():
    r = [1, 2, 3, 4, 3, 2, 1.5, 2, 3, 4, 4, 3]                 # a 2.5 mm return between two extensions 24 ms apart
    S = _frames(1, 0.0, r)
    assert len(CL.mouth_cuts(S, prominence_mm=1.0, min_ili_ms=20).get(1, [])) == 1
    assert len(CL.mouth_cuts(S, prominence_mm=3.0).get(1, [])) == 0          # not prominent enough
    assert len(CL.mouth_cuts(S, prominence_mm=1.0, min_ili_ms=40).get(1, [])) == 0   # peaks < 40 ms apart


def test_lick_train_d2_revised_drops_long_cam1_only():
    U = CL.fuse({"cam1": _ev("cam1", [(1, 0, 20, 10), (1, 100, 450, 200)]), "cam4": _ev("cam4", [(1, 600, 1000, 700)])})
    assert len(CL.lick_train(U)) == 2                          # short cam1-only lick kept; 350 ms cam1-only dropped
    assert len(CL.lick_train(U, long_cam1_only_ms=None)) == 3
    assert len(CL.lick_train(U, min_views=2)) == 0


def test_bundle_out_carries_the_pose3d_run_tag(tmp_path, monkeypatch):
    """11-21: the whole-session camera_licks folder takes the 3-D run tag, so a tagged run never overwrites the pilot's
    untagged camera_licks/<bundle>/ tables; an invalid tag is refused."""
    import pytest

    from scripts import camera_licks as SCL

    class _RV:
        def root(self, _k):
            return str(tmp_path)

    from wfield_local import pose3d
    monkeypatch.setattr(pose3d, "ADOPTED_SOURCE", "2cam")        # the pre-adoption semantics (no default tag)
    monkeypatch.setattr(pose3d, "ADOPTED_TAG", "")
    monkeypatch.delenv("WIDEFIELD_POSE3D_TAG", raising=False)
    assert SCL._bundle_out(_RV(), "bund").name == "bund"
    monkeypatch.setenv("WIDEFIELD_POSE3D_TAG", "_3net_20261121")
    p = SCL._bundle_out(_RV(), "bund")
    assert p.name == "bund_3net_20261121" and p.is_dir()
    monkeypatch.setenv("WIDEFIELD_POSE3D_TAG", "../x")
    with pytest.raises(ValueError):
        SCL._bundle_out(_RV(), "bund")


def test_saved_side_events_round_trip_keeps_guards_boolean(tmp_path):
    """11-21: the four-camera production run writes the side events once; the --saved fusion reads them back per camera
    with the static / far guards as bool (a CSV round trip gives strings / NaN)."""
    from scripts.camera_licks import _saved_side
    E = pd.DataFrame({"trial_id": [1, 1, 2], "on_ms": [0, 50, 10], "off_ms": [20, 70, 30], "t_peak_ms": [10, 60, 20],
                      "static": [True, False, None], "far": [False, None, True], "cam": ["cam2", "cam3", "cam2"]})
    E.to_csv(tmp_path / "PS93_20260814_side_events.csv", index=False)
    ev = _saved_side(tmp_path, "PS93", "20260814")
    assert set(ev) == {"cam2", "cam3"} and len(ev["cam2"]) == 2
    assert ev["cam2"].static.dtype == bool and list(ev["cam2"].static) == [True, False]
    assert list(ev["cam2"].far) == [False, True] and ev["cam3"].far.tolist() == [False]


def _fused(rows):
    """Fused-lick rows: (trial, on, off, cams)."""
    U = pd.DataFrame([{"trial_id": t, "on_ms": a, "off_ms": b, "cams": c} for t, a, b, c in rows])
    for c in CL.CAMS:
        U[f"seen_{c}"] = U.cams.str.split("+").map(lambda s, c=c: c in s)
    U["guarded"] = False
    return U


def test_track_flags_needs_confidence_and_a_tip_in_front_of_the_mouth():
    f = CL.track_flags([0.9, 0.3, 0.9, 0.9], [10.0, 10.0, -5.0, 0.0], [50.0, 50.0, 50.0, np.nan])
    assert list(f) == [True, False, False, True]                     # lk < 0.4 / behind the mouth line drop; NaN spout ok
    assert list(CL.track_flags([0.9], [10.0], [3.0], spout_min_px=5.0)) == [False]


def test_tracked_frames_counts_tracked_frames_inside_the_span():
    U = _fused([(1, 100, 140, "cam2"), (2, 0, 20, "cam3")])
    F = pd.DataFrame({"trial_id": [1, 1, 1, 1, 2], "t_ms": [96.0, 104.0, 120.0, 150.0, 8.0],
                      "track": [True, True, False, True, True]})
    assert list(CL.tracked_frames(U, F)) == [1, 1]                    # 96 is outside 100-2; 150 outside 140+2
    assert list(CL.tracked_frames(U, None)) == [0, 0]


def test_confirm_side_only_reasons_in_precedence_order():
    U = _fused([(1, 0, 40, "cam4"),             # camera event
                (1, 45, 60, "cam2"),            # touches the cam4 lick -> cut fragment (even if tracked)
                (1, 300, 340, "cam2"),          # one side view, cam4 tracked -> dropped (failed blind check)
                (1, 500, 540, "cam3+cam2"),     # both side views (tracked or not) -> kept
                (1, 700, 740, "cam2+cam3"),     # both side views, nobody tracked -> kept
                (1, 900, 940, "cam2"),          # one side view, untracked -> dropped
                (2, 0, 30, "cam3")])            # guarded
    U.loc[6, "guarded"] = True
    tr = {"cam4": [0, 5, 3, 0, 0, 1, 0], "cam1": [0, 0, 0, 2, 0, 0, 0]}
    R = CL.confirm_side_only(U, tr)
    assert list(R.reason) == ["camera_event", "cut_fragment", "one_view_tracked", "two_side_views", "two_side_views",
                              "unconfirmed_one_side_view", "guarded"]
    assert list(R.keep) == [True, False, False, True, True, False, False]
    assert set(R.reason) <= set(CL.REASONS)
    assert list(R.tracked_cam4) == tr["cam4"]
    # the first version (not adopted): tracking confirms on its own
    F = CL.confirm_side_only(U, tr, track_confirms=True)
    assert list(F.reason[:6]) == ["camera_event", "cut_fragment", "tracked_cam4", "tracked_cam1", "two_side_views",
                                  "unconfirmed_one_side_view"]
    assert not CL.confirm_side_only(U, tr, two_view=False).keep.iloc[4]


def test_lick_train_honors_the_table_gaps_keep():
    U = _fused([(1, 0, 40, "cam4"), (1, 300, 340, "cam2")])
    U["n_views"] = 1
    R = CL.confirm_side_only(U, {})
    assert len(CL.lick_train(R)) == 1 and len(CL.lick_train(U)) == 2


def test_anchor_bundles_match_the_3d_production_bundles():
    from scripts.pose3d import PROD
    assert CL.ANCHOR_BUNDLES == {c: PROD[c] for c in CL.ANCHOR_BUNDLES}
