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


def test_the_analysis_deck_states_its_trial_population():
    """"Engaged" means two different things in this deck, and the slides never said which.

    The decode/encode/frozen/RSA families split on a lick detected within `max_rt`; sections G/H/I
    split the no-lick trials with the ENGAGEMENT GATE. A reader comparing a frozen-decoder curve
    against a section-G contrast is comparing two trial SETS, not two results.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "wfield_local"
           / "locanmf_analysis_deck.py").read_text(encoding="utf-8")
    assert "TRIALS_LICK" in src and "TRIALS_NOLICK" in src
    assert "max_rt 3.5 s" in src, "the lick label must name the actual window"
    assert "ENGAGEMENT GATE" in src, "the working label must name the gate"
    # the ambiguous families are the ones that must carry it
    assert src.count("trials=TRIALS_LICK") >= 12
    assert "def title(s, text, sub=None, trials=None):" in src


def test_the_clip_deck_keeps_every_position_in_its_own_slot():
    """Packing shown clips from the top-left would move a position between slides."""
    from wfield_local import behavior_clip_deck as d

    assert list(d.POS_ORDER).index("far_center") == 4     # bottom row, centre column
    assert list(d.POS_ORDER).index("far_R") == 5          # bottom row, right column
    src = __import__("inspect").getsource(d.build)
    assert "for i, pos in enumerate(POS_ORDER)" in src, "grid must be indexed by position"
    assert "-- none --" in src, "an empty slot must say so rather than go blank"


def test_the_clip_deck_curates_but_the_library_does_not():
    from wfield_local import behavior_clip_deck as d

    assert d.deck_cap("far_R", "working") == 5            # the deficit earns a run
    assert d.deck_cap("close_L", "working") == 2
    assert d.deck_cap("far_R", "success") == 2            # a hit looks like a hit
    assert d.deck_cap("far_R", "stopped") == 2


def test_an_empty_trial_class_still_gets_a_slide():
    """Skipping it silently reads as a rendering gap, not as a result.

    PS92 09-04 hit 378 of 378 trials, so it has no working and no stopped trials at all -- and the
    deck simply had no slides for them, which is what prompted the question.
    """
    from wfield_local import behavior_clip_deck as d

    for cat in ("success", "working", "stopped"):
        note = d._absence_note(cat)
        assert note and len(note) > 40
    assert "378" in d._absence_note("working"), "say what an empty working class means"
    src = __import__("inspect").getsource(d.build)
    assert "if depth == 0:" in src and "NO TRIALS IN THIS CLASS" in src


def test_the_deck_derives_its_epoch_label_rather_than_reading_the_folder():
    """`chronic_from` is recomputed each run, so a folder can carry an epoch the animal has left."""
    from wfield_local import behavior_clip_deck as d

    src = __import__("inspect").getsource(d._sessions_for)
    assert "epochs.epoch_of(" in src
    assert "lab or epoch_dir.name" in src, "fall back to the folder, never crash"


def test_stale_epoch_folders_are_re_filed():
    """Three PS92 sessions sat under subacute/ after the animal's chronic boundary landed."""
    from wfield_local import behavior_clips as bc

    src = __import__("inspect").getsource(bc.refile_stale_epochs)
    assert "epochs.epoch_of(" in src
    assert "assert_writable" in src
    run_src = __import__("inspect").getsource(bc.run)
    assert "refile_stale_epochs(" in run_src, "the nightly must re-file before it cuts"


def test_the_camera_nightly_rebuilds_the_clip_decks_after_cutting():
    from wfield_local import camera_nightly as cn

    src = __import__("inspect").getsource(cn.run)
    assert "behavior_clip_deck.run(" in src
    assert src.index("do_clips") < src.index("do_clip_deck")
    assert "do_clip_deck" in __import__("inspect").signature(cn.run).parameters
