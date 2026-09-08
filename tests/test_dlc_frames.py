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


def test_the_lick_offset_lands_a_couple_of_frames_AFTER_contact():
    """At the onset the tongue is at the spout and maximally occluded by it; a little later it is
    still out and better separated. Later than ~40 ms and it is retracting."""
    off = df.lick_offset_s()
    assert 0 < off < 0.040
    assert round(off * 250) == 2, "two frames at 250 fps"


def test_a_lick_frame_maps_through_the_same_template_as_a_cue_frame():
    """frame_of takes a DAQ TIME, not specifically a cue -- both live on the one DAQ clock."""
    tpl = _tpl()
    lick_t = 100.0
    assert df.frame_of(tpl, lick_t, df.lick_offset_s()) == 25_000 + 2


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

def test_the_bodypart_set_is_shared_across_cameras():
    """Triangulation matches keypoints by NAME, so a per-view set cannot be lifted to 3D later."""
    bps = df.bodyparts()
    assert "spout" in bps and "L_spout" not in bps and "R_spout" not in bps
    assert "L_eye" not in bps and "R_eye" not in bps, "not in frame on cam1/cam4"
    assert bps == sorted(set(bps), key=bps.index), "duplicate bodypart"


def test_the_phase_offsets_sit_inside_the_example_clip_window():
    """A labelled frame outside the clip window is one nobody can go and look at."""
    from wfield_local.behavior_clips import POST_S, PRE_S

    for name, off in df.phases().items():
        assert -PRE_S <= off <= POST_S, f"phase {name} at {off}s is outside the clip window"


@pytest.mark.parametrize("key", ["per_session", "seed"])
def test_the_extraction_parameters_come_from_config(key):
    assert isinstance(getattr(df, key)(), int)
