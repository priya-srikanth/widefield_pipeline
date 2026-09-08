"""What a labelling set can get silently wrong, pinned.

A bad labelling set does not fail -- it trains a network that tracks well where it was labelled and
badly everywhere else, and "badly everywhere else" here means POST-STROKE, which is where the
result lives. So the tests below are mostly about coverage and determinism, not about pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import dlc_frames as df

# --------------------------------------------------------------------------- selection

def _trials(n_per_pos=4, positions=("far_L", "far_center", "far_R", "close_L", "close_center",
                                    "close_R")):
    rows, tid = [], 0
    for pos in positions:
        for k in range(n_per_pos):
            rows.append({"trial_id": tid, "pos_name": pos, "cue_s": 10.0 + tid,
                         "trial_start_s": 8.0 + tid,
                         "cat": ["success", "working", "stopped"][k % 3]})
            tid += 1
    return pd.DataFrame(rows)


def test_every_spout_position_is_represented():
    """Position is the design's own stratification -- a set missing one cannot report on it."""
    picks = df.select_trials(_trials(), n_cells=6, rng=np.random.default_rng(0))
    assert set(picks["pos_name"]) == {"far_L", "far_center", "far_R",
                                      "close_L", "close_center", "close_R"}


def test_failed_and_stopped_trials_are_not_all_dropped():
    """Labelling only successful trials teaches the network the posture of a mouse that is licking.

    Post-stroke the interesting frames are the ones where it tries and fails, and those are exactly
    the frames a success-only set never showed it.
    """
    cats = set(df.select_trials(_trials(), 6, np.random.default_rng(0))["cat"])
    assert len(cats) >= 2, f"only {cats} represented across six positions"


def test_a_position_with_only_one_category_still_yields_a_frame():
    """The rotation PREFERS a category, it does not require one."""
    t = _trials(n_per_pos=2)
    t.loc[t["pos_name"] == "far_R", "cat"] = "stopped"
    picks = df.select_trials(t, 6, np.random.default_rng(0))
    assert "far_R" in set(picks["pos_name"])


def test_selection_is_deterministic_and_session_scoped():
    """Re-running must re-pick the SAME frames, or a second run grows a divergent labelling set.

    And the stream must depend only on the session, so extracting a new date cannot change what a
    date somebody has already annotated would pick.
    """
    t = _trials()
    a = df.select_trials(t, 6, df._rng("PS94", "20260907", "cam4"))
    b = df.select_trials(t, 6, df._rng("PS94", "20260907", "cam4"))
    assert list(a["trial_id"]) == list(b["trial_id"])

    other = df.select_trials(t, 6, df._rng("PS94", "20260903", "cam4"))
    cam1 = df.select_trials(t, 6, df._rng("PS94", "20260907", "cam1"))
    assert (list(a["trial_id"]) != list(other["trial_id"])
            or list(a["trial_id"]) != list(cam1["trial_id"])), "streams are not independent"


# --------------------------------------------------------------------------- frame mapping

def _tpl(fs=5000.0, fps=250.0, n=1_500_000):
    return {"fs_daq": fs, "fps_cam": fps, "slope_daqSample_per_camFrame": fs / fps,
            "intercept_daqSample": 0.0, "n_cam_frames": n, "quality_ok": True}


def test_phase_offsets_land_where_the_cue_says():
    tpl = _tpl()
    cue = 100.0                                   # seconds of DAQ time
    at_cue = df.frame_of(tpl, cue, 0.0)
    assert at_cue == 25_000                       # 100 s x 250 fps
    assert df.frame_of(tpl, cue, -0.60) == at_cue - 150
    assert df.frame_of(tpl, cue, 1.50) == at_cue + 375


def test_frame_numbers_are_python_ints():
    """They index `cv2.CAP_PROP_POS_FRAMES` and land in the manifest; `2481.0` is a filename bug."""
    tpl = {k: (np.float64(v) if isinstance(v, float) else v) for k, v in _tpl().items()}
    f = df.frame_of(tpl, np.float64(100.0), np.float64(-0.6))
    assert type(f) is int


# --------------------------------------------------------------------------- coverage

def test_pick_middle_prefers_the_centre_but_falls_back_outward():
    """The bug this exists for: the camera alignment templates start on 2026-06-06 while the `pre`
    epoch opens on 2026-05-27, so the MEDIAN pre-stroke session was untemplated for every animal and
    the cohort walk produced a labelling set with no pre-stroke frames -- and said it succeeded.
    """
    sessions = [(f"2026060{i}", f"s{i}") for i in range(1, 8)]     # 0601..0607, middle = 0604
    assert df._pick_middle(sessions, lambda s: True)[0] == "20260604"

    late_only = df._pick_middle(sessions, lambda s: s[0] >= "20260606")
    assert late_only[0] == "20260606", "should take the CLOSEST usable, not the first or last"

    assert df._pick_middle(sessions, lambda s: False) is None


def test_a_session_missing_one_camera_template_is_not_usable(tmp_path, monkeypatch):
    """Views that get triangulated together have to be labelled on the SAME sessions."""
    root = tmp_path / "alignment_templates"
    for cam in ("cam4", "cam1"):
        (root / cam / "PS94").mkdir(parents=True)
    np.savez(root / "cam4" / "PS94" / "20260907.npz", quality_ok=True)

    class RV:
        def root(self, name):
            assert name == "alignment_templates"
            return str(root)

    assert df.usable("PS94", "20260907", ["cam4"], RV())
    assert not df.usable("PS94", "20260907", ["cam4", "cam1"], RV())


def test_a_failed_template_is_not_usable(tmp_path):
    root = tmp_path / "alignment_templates"
    (root / "cam4" / "PS94").mkdir(parents=True)
    np.savez(root / "cam4" / "PS94" / "20260907.npz", quality_ok=False)

    class RV:
        def root(self, name):
            return str(root)

    assert not df.usable("PS94", "20260907", ["cam4"], RV())


# --------------------------------------------------------------------------- lick-locked frames

def test_lick_frames_are_spread_over_spout_position():
    """A tongue labelled only where the animal licks most trains a network that finds the tongue
    best exactly where the behaviour is already easiest."""
    t = _trials(n_per_pos=2)
    licks = np.concatenate([t["cue_s"].to_numpy() + off for off in (0.3, 0.6, 0.9)])
    picks = df.select_licks(t, licks, n_per_pos=2, window_s=3.5, rng=np.random.default_rng(0))
    assert {str(r["pos_name"]) for r, _ in picks} == set(t["pos_name"])


def test_only_licks_INSIDE_the_response_window_are_used():
    """A lick 6 s after the cue is ITI licking, not a response to this trial's spout position."""
    t = _trials(n_per_pos=1)
    t["cue_s"] = 10.0 + 20.0 * np.arange(len(t))          # well-separated, so only the bound bites
    late = t["cue_s"].to_numpy() + 6.0
    assert df.select_licks(t, late, 2, 3.5, np.random.default_rng(0)) == []

    inside = t["cue_s"].to_numpy() + 0.5
    assert len(df.select_licks(t, inside, 2, 3.5, np.random.default_rng(0))) == len(t)


def test_the_window_is_bounded_by_the_NEXT_CUE_not_just_its_length():
    """`daq_trials`' own rule. Trials here are 1 s apart while the window is 3.5 s, so without the
    bound a lick belonging to the next trial -- at a DIFFERENT spout position -- is attributed to
    this one, and the frame gets filed under a position the tongue was not reaching for.
    """
    t = pd.DataFrame([
        {"trial_id": 0, "pos_name": "far_L", "cue_s": 10.0, "trial_start_s": 9.0, "cat": "success"},
        {"trial_id": 1, "pos_name": "far_R", "cue_s": 11.0, "trial_start_s": 10.5, "cat": "success"},
    ])
    picks = df.select_licks(t, np.array([11.5]), 2, 3.5, np.random.default_rng(0))
    assert [str(r["pos_name"]) for r, _ in picks] == ["far_R"], "1.5 s past far_L's cue, but far_R's trial"


def test_a_session_with_no_licks_contributes_no_lick_frames():
    """Post-stroke sessions where the animal barely licks must degrade, not raise."""
    assert df.select_licks(_trials(), np.array([]), 2, 3.5, np.random.default_rng(0)) == []


def test_the_lick_offsets_span_the_tongue_out_epoch_not_a_symmetric_window():
    """Measured on PS94 at 12 ms steps from -28 to +84 ms: the tongue is still IN through most of
    the pre-onset half, is broadest at +24..+48 ms, and is retracting by +72..+84. A symmetric
    +/-28 ms window spends half its frames before the tongue appears and stops before the peak.
    """
    offs = df.lick_offsets_s()
    assert len(offs) > 1, "one offset per lick teaches the network a single tongue posture"
    assert offs == sorted(offs)
    assert min(offs) < 0, "at least one frame with the tongue still in, as a negative case"
    assert max(offs) >= 0.040, "must reach peak extension, not stop at +28 ms"
    assert max(offs) <= 0.080, "past ~72 ms the tongue is retracting"
    assert len(set(offs)) == len(offs), "a repeated offset is a duplicate frame to label"


def test_offsets_are_far_enough_apart_to_be_different_postures():
    """At 250 fps, adjacent frames are near-duplicates; labelling both is wasted effort."""
    offs = df.lick_offsets_s()
    gaps = np.diff(offs)
    assert (gaps * 250 >= 3).all(), f"offsets closer than 3 frames apart: {gaps * 1000} ms"


def test_the_lick_cache_is_invalidated_when_the_detection_params_change(tmp_path, monkeypatch):
    """A cache of DERIVED event times is only valid while the settings that derived them hold.

    Re-decoding the cohort is ~20 minutes, paid every time an offset is tuned, so caching is worth
    it -- but a lick threshold change has to be a MISS, not a warning nobody reads.
    """
    monkeypatch.setattr(df, "out_root", lambda rv=None: tmp_path)
    p0 = {"thresh_upper": 2.5, "thresh_lower": 0.5, "min_ili_ms": 40}
    df._store_licks("PS94", "20260831", np.array([1.0, 2.0, 3.0]), p0)

    assert df._cached_licks("PS94", "20260831", p0).tolist() == [1.0, 2.0, 3.0]
    assert df._cached_licks("PS94", "20260831", {**p0, "thresh_upper": 3.0}) is None
    assert df._cached_licks("PS94", "20260606", p0) is None, "a session never cached is a miss"


def test_a_corrupt_lick_cache_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(df, "out_root", lambda rv=None: tmp_path)
    p = df._lick_cache_path("PS94", "20260831")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not an npz")
    assert df._cached_licks("PS94", "20260831", {"a": 1}) is None


def test_a_scalar_lick_offset_config_still_works(monkeypatch):
    """Older configs wrote `lick_offset_s: 0.008`; reading one must not become a crash."""
    real = df._cfg()
    frames = {k: v for k, v in real["frames"].items() if k != "lick_offsets_s"}
    frames["lick_offset_s"] = 0.008
    monkeypatch.setattr(df, "_cfg", lambda: {**real, "frames": frames})
    assert df.lick_offsets_s() == [0.008]


def test_a_lick_frame_maps_through_the_same_template_as_a_cue_frame():
    """frame_of takes a DAQ TIME, not specifically a cue -- both live on the one DAQ clock."""
    tpl = _tpl()
    lick_t = 100.0
    assert df.frame_of(tpl, lick_t, 0.008) == 25_000 + 2
    assert df.frame_of(tpl, lick_t, -0.016) == 25_000 - 4


# --------------------------------------------------------------------------- manifest

def _row(stem="cam4_x", frame=10, **kw):
    r = {c: "" for c in df.MANIFEST_COLUMNS}
    r.update(video_stem=stem, frame=frame, animal="PS94", cam="cam4", **kw)
    return r


def test_the_manifest_appends_rather_than_replacing(tmp_path):
    """The set is built a date at a time; a rewrite would orphan the provenance of earlier frames."""
    dest = tmp_path / "frame_manifest.csv"
    df.write_manifest([_row(frame=1), _row(frame=2)], dest=dest)
    df.write_manifest([_row(frame=3)], dest=dest)

    got = pd.read_csv(dest)
    assert sorted(got["frame"]) == [1, 2, 3]


def test_re_extracting_a_date_does_not_duplicate_its_rows(tmp_path):
    dest = tmp_path / "frame_manifest.csv"
    rows = [_row(frame=1), _row(frame=2)]
    df.write_manifest(rows, dest=dest)
    df.write_manifest(rows, dest=dest)
    assert len(pd.read_csv(dest)) == 2


def test_the_manifest_carries_the_provenance_needed_to_audit_the_set(tmp_path):
    """"Which epochs is this network trained on?" gets asked the first time post-stroke tracking
    looks worse, and it has to be answerable from the file rather than from memory."""
    dest = tmp_path / "frame_manifest.csv"
    df.write_manifest([_row(epoch="acute", position="far_R", category="stopped", phase="early",
                            date="20260820", trial_id=7, t_from_cue_s=0.4)], dest=dest)
    got = pd.read_csv(dest).iloc[0]
    for field in ("animal", "date", "cam", "epoch", "position", "category", "phase", "trial_id"):
        assert str(got[field]) not in ("", "nan"), f"{field} missing from the manifest"


# --------------------------------------------------------------------------- config

def test_names_are_shared_where_the_views_OVERLAP():
    """Triangulation matches keypoints by NAME, so the overlap is what can ever become 3D.

    The sets are per-view because the views genuinely differ -- cam1 looks up from below and has no
    nose -- but a part seen by two cameras must carry the SAME name in both or it is two parts.
    """
    bps = df.bodyparts()
    assert "spout" in bps and "L_spout" not in bps and "R_spout" not in bps
    assert bps == sorted(set(bps), key=bps.index), "duplicate bodypart in the union"

    shared = df.shared_bodyparts()
    assert {"jaw", "tongue", "spout"} <= set(shared), "the cohort-wide 3D parts"
    assert "nose" not in shared, "cam4 only -- it stays 2D, and pretending otherwise hides that"
    assert "L_eye" not in shared and "R_eye" not in shared, "one side view each"


def test_each_view_declares_only_what_it_can_see():
    assert df.bodyparts("cam1") == ["jaw", "tongue", "spout"], "bottom view: no nose, no whiskers"
    assert "nose" in df.bodyparts("cam4")
    for cam in df.cameras():
        assert df.bodyparts(cam), f"{cam} declares no bodyparts"
        assert set(df.bodyparts(cam)) <= set(df.bodyparts()), f"{cam} has a part outside the union"


def test_an_unknown_camera_asks_for_nothing_rather_than_everything():
    """A typo in --cam must not silently seed the full frontal set onto some other view."""
    assert df.bodyparts("cam9") == []


def test_the_phase_offsets_sit_inside_the_example_clip_window():
    """A labelled frame outside the clip window is one nobody can go and look at."""
    from wfield_local.behavior_clips import POST_S, PRE_S

    for name, off in df.phases().items():
        assert -PRE_S <= off <= POST_S, f"phase {name} at {off}s is outside the clip window"


@pytest.mark.parametrize("key", ["per_session", "seed"])
def test_the_extraction_parameters_come_from_config(key):
    assert isinstance(getattr(df, key)(), int)
