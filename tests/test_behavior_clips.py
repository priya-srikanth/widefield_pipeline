"""Guards for the annotated example clips and their place in the camera nightly."""
import inspect

import numpy as np
import pandas as pd
import pytest

from wfield_local import behavior_clips as bc


def test_the_window_covers_the_whole_response_window():
    """3.0 s post-cue would end before the 3500 ms window the animal is scored on."""
    assert bc.POST_S >= 3.5
    assert bc.PRE_S >= 1.0


def test_phases_are_named_from_the_trial_s_own_enl():
    """ENL is jittered 2-3 s per trial, so the label has to come from that trial, not a constant."""
    assert bc.phase_at(-0.5, enl_len=2.4) == "ENL"
    assert bc.phase_at(-2.9, enl_len=2.4) == "ITI"       # older than this trial's ENL
    assert bc.phase_at(-2.9, enl_len=3.0) == "ENL"       # same instant, longer ENL
    assert bc.phase_at(0.05, enl_len=2.4) == "Cue"
    assert bc.phase_at(0.5, enl_len=2.4) == "Response"
    assert bc.phase_at(3.4, enl_len=2.4) == "Response"


def test_categories_recompute_engagement_rather_than_trusting_the_column():
    """The gate was backdated on 2026-09-07; a persisted trials.csv may predate it."""
    src = inspect.getsource(bc.categorise)
    assert "reference_engagement" in src
    assert '"engaged"' not in src and "'engaged'" not in src


def test_a_hit_is_a_success_and_an_engaged_miss_is_working():
    d = pd.DataFrame({
        "hit": [1, 0, 0], "responded": [True, False, False],
        "pos_name": ["close_L", "close_L", "far_R"],
        "trial_id": [1, 2, 3], "cue_s": [1.0, 2.0, 3.0], "trial_start_s": [0.0, 1.0, 2.0]})
    cat = bc.categorise(d)
    assert cat.iloc[0] == "success"
    assert set(cat.iloc[1:]) <= {"working", "stopped"}


def test_clip_generation_uses_a_flat_cap():
    """The library is uncurated ON PURPOSE (Priya: 'don't change the clip generation, just the
    deck'). Pushing presentation choices in here means re-encoding video when they change."""
    d = pd.DataFrame({"pos_name": ["far_R"] * 9, "cat": ["success"] * 3 + ["working"] * 3
                      + ["stopped"] * 3})
    cells = {(p, c): t for p, c, _a, t in bc._cell_counts(d, 5)}
    assert cells[("far_R", "success")] == 3
    assert cells[("far_R", "stopped")] == 3          # not curated down here


def test_the_camera_nightly_cuts_clips_after_behaviour():
    """It needs the alignment template AND the trial table, so it cannot run earlier."""
    from wfield_local import camera_nightly as cn

    src = inspect.getsource(cn.run)
    assert "behavior_clips.run(" in src
    assert src.index("do_behavior") < src.index("do_clips")
    assert "do_clips" in inspect.signature(cn.run).parameters


def test_a_session_without_a_good_template_is_skipped_not_guessed(tmp_path, monkeypatch):
    """The 8/18 folder swap was caught by `quality_ok`; cutting anyway would have hidden it."""
    class RV:
        def root(self, name):
            return str(tmp_path / name)

    rv = RV()
    p = tmp_path / "alignment_templates" / bc.CAM / "PS94"
    p.mkdir(parents=True)
    np.savez(p / "20260820.npz", quality_ok=False, fs_daq=5000.0, fps_cam=250.0,
             n_cam_frames=10, slope_daqSample_per_camFrame=20.0, intercept_daqSample=0.0)
    assert bc.session_clips("PS94", "20260820", "sid", "acute", rv=rv) == 0
