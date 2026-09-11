"""A calibration fails silently or not at all: every triangulated point downstream inherits it, and
nothing in a 3D trajectory looks wrong the way a bad image does.

So these tests are about the ways a pooled solve could produce confident, wrong geometry -- mixing
boards where the object points are indexed by id, pooling extrinsics across a rig that moved, or
accepting a K that fits beautifully and is not identifiable.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from wfield_local import dlc_anipose as da

#: aniposelib is installed in the `dlc` env only -- the production env on both boxes is `locanmf`,
#: and the pre-push hook runs pytest from whichever one is active. The pure-logic tests below (pose
#: counting, identifiability, the gate's verdicts) must keep running everywhere; only the ones that
#: actually construct an aniposelib board or CameraGroup are skipped.
requires_aniposelib = pytest.mark.skipif(
    importlib.util.find_spec("aniposelib") is None,
    reason="aniposelib is installed in the `dlc` env only")


def _row(frame, n_corners, ids=None):
    """A minimal aniposelib-shaped detection row."""
    ids = np.arange(n_corners) if ids is None else np.asarray(ids)
    return {"framenum": frame, "ids": ids.reshape(-1, 1),
            "corners": np.zeros((len(ids), 1, 2), np.float32)}


# ------------------------------------------------------------------ the library's own thresholds

@requires_aniposelib
def test_the_mirrored_thresholds_still_match_aniposelib():
    """These constants are COPIES of numbers inside aniposelib, and the gate reports them as the bar
    a camera must clear. An upgrade that moved either one would leave the gate quietly lying about
    what the solve will accept -- passing cameras the solve then drops, which is exactly the failure
    the marker-based gate produced on 2026-09-10."""
    import inspect

    from aniposelib.cameras import CameraGroup

    sig = inspect.signature(CameraGroup.calibrate_rows)
    assert sig.parameters["min_corners_intrinsic"].default == da.ANIPOSE_MIN_CORNERS_INTRINSIC

    src = inspect.getsource(CameraGroup.calibrate_rows)
    assert f"'ids'].size >= {da.ANIPOSE_MIN_CORNERS_BUNDLE}" in src, \
        "aniposelib's bundle-adjustment corner threshold moved; ANIPOSE_MIN_CORNERS_BUNDLE is stale"


def test_our_old_threshold_was_LOWER_than_the_one_that_decides():
    """dlc_calibrate gated at 6 corners; aniposelib admits a row at 8 and initialises intrinsics at
    9. A camera can therefore clear our gate and still be dropped by the solver."""
    from wfield_local.dlc_calibrate import MIN_CORNERS

    assert MIN_CORNERS < da.ANIPOSE_MIN_CORNERS_BUNDLE <= da.ANIPOSE_MIN_CORNERS_INTRINSIC


# ------------------------------------------------------------------ poses, not frames

def test_a_held_board_is_ONE_pose_however_many_frames_it_spans():
    """At 250 fps a board held still for two seconds is 500 frames and one view of the board."""
    held = [_row(f, 12) for f in range(500)]
    assert da.pose_counts(held, 8) == 1


def test_poses_are_counted_only_among_rows_that_clear_the_bar():
    rows = [_row(0, 12), _row(500, 4), _row(1000, 12)]
    assert da.pose_counts(rows, 8) == 2, "the 4-corner row must not count as a pose"
    assert da.pose_counts(rows, 4) == 3


def test_one_per_pose_keeps_one_row_from_each_distinct_view():
    rows = [_row(f, 12) for f in (0, 10, 20, 500, 510, 5000)]
    kept = [da._framenum(r) for r in da._one_per_pose(rows)]
    assert kept == [0, 500, 5000]


def test_framenum_survives_anipose_prefixing():
    """aniposelib keys rows as (prefix, n) once more than one video feeds a camera; a pose counter
    that compared tuples would order them by prefix and collapse every recording into one run."""
    assert da._framenum({"framenum": (3, 1750)}) == 1750
    assert da._framenum({"framenum": 1750}) == 1750


# ------------------------------------------------------------------ identifiability

def test_the_measured_cam1_solve_is_REJECTED_despite_a_low_reprojection_error():
    """The real numbers from Widefield_calibration_20260910: a principal point at (1354, 402) on a
    600x600 frame, reported alongside 1.95 px of reprojection error. RMS cannot see this."""
    K = np.array([[4375.0, 0, 1354.1], [0, 4375.0, 401.9], [0, 0, 1.0]])
    q = da.intrinsics_quality(K, np.array([-1.161, 0, 0, 0, 0]), (600, 600))
    assert not q["identifiable"]
    assert any("OUTSIDE" in p for p in q["problems"])


def test_pinning_the_principal_point_does_not_launder_cam1():
    """With pp pinned the same recording returns k1 = -4.05, which is not a lens. The
    non-identifiability moves into whatever parameter is left free; it does not go away."""
    K = np.array([[5099.1, 0, 299.5], [0, 5099.1, 299.5], [0, 0, 1.0]])
    q = da.intrinsics_quality(K, np.array([-4.052, 0, 0, 0, 0]), (600, 600))
    assert not q["identifiable"]
    assert any("not a lens" in p for p in q["problems"])


def test_the_measured_cam2_solve_with_a_pinned_principal_point_is_ACCEPTED():
    """The other half of the finding: cam2 sees the 40 mm board properly and returns f=1519,
    k1=-0.21. A check that rejected everything would be useless."""
    K = np.array([[1519.0, 0, 299.5], [0, 1519.0, 224.5], [0, 0, 1.0]])
    q = da.intrinsics_quality(K, np.array([-0.206, 0, 0, 0, 0]), (600, 450))
    assert q["identifiable"], q["problems"]


def test_a_NON_SQUARE_PIXEL_is_refused_even_with_the_principal_point_pinned():
    """The measured cam2 solve AFTER --fix-principal-point: fx=1458, fy=1617.

    The sharpest of the three checks, because it tests against a fact about the sensor rather than a
    plausible range -- Blackfly pixels are square, so fx and fy are the same length in the same units.
    A 10% disagreement is the degeneracy moving into the parameter left free when the principal point
    was pinned, which is why pinning it is a way to keep working and not a way to fix anything.
    """
    K = np.array([[1457.8, 0, 299.5], [0, 1616.8, 224.5], [0, 0, 1.0]])
    q = da.intrinsics_quality(K, np.array([-0.039, 0, 0, 0, 0]), (600, 450))
    assert not q["identifiable"]
    assert any("non-square pixel" in p for p in q["problems"])
    assert q["aspect_error"] > 0.09


def test_square_pixels_within_tolerance_pass():
    K = np.array([[1519.0, 0, 299.5], [0, 1521.0, 224.5], [0, 0, 1.0]])
    q = da.intrinsics_quality(K, np.array([-0.206, 0, 0, 0, 0]), (600, 450))
    assert q["identifiable"], q["problems"]


def test_a_principal_point_inside_the_frame_but_far_off_centre_is_still_refused():
    K = np.array([[1500.0, 0, 560.0], [0, 1500.0, 225.0], [0, 0, 1.0]])
    q = da.intrinsics_quality(K, np.zeros(5), (600, 450))
    assert not q["identifiable"] and any("image-widths from" in p for p in q["problems"])


# ------------------------------------------------------------------ what may and may not be pooled

def _recordings(spec_a, spec_b, cams=("cam1", "cam2")):
    def one(spec):
        return (list(cams), {c: [] for c in cams}, {c: (600, 600) for c in cams}, spec)
    from pathlib import Path
    return {Path("A"): one(spec_a), Path("B"): one(spec_b)}


SPEC_40 = {"squares_x": 6, "squares_y": 6, "square_mm": 6.67, "marker_mm": 5.07,
           "dictionary": "DICT_4X4_50"}
SPEC_26 = {"squares_x": 6, "squares_y": 6, "square_mm": 4.33, "marker_mm": 3.29,
           "dictionary": "DICT_4X4_50"}


def test_extrinsics_REFUSE_to_pool_recordings_that_used_different_boards():
    """aniposelib indexes object points by ChArUco corner id against ONE board. Both our boards are
    DICT_4X4_50 with overlapping ids, so mixing them does not error -- it reads a 26 mm board's
    corners against 40 mm object points and returns a confident, wrong scale."""
    from pathlib import Path

    recs = _recordings(SPEC_40, SPEC_26)
    with pytest.raises(ValueError, match="DIFFERENT boards"):
        da._rows_for(recs, [Path("A"), Path("B")], ["cam1", "cam2"])


@requires_aniposelib
def test_extrinsics_DO_pool_recordings_that_shared_a_board():
    from pathlib import Path

    recs = _recordings(SPEC_40, dict(SPEC_40))
    rows, board = da._rows_for(recs, [Path("A"), Path("B")], ["cam1", "cam2"])
    assert len(rows) == 2 and board.squaresX == 6


def test_intrinsics_refuse_to_pool_a_camera_whose_frame_size_changed():
    """A different ROI is a different optical configuration; its principal point is somewhere else
    in the frame and pooling the two would average two unrelated cameras.

    Checked before any board is constructed, so it is refused on the cheap fact and stays testable
    without aniposelib installed."""
    from pathlib import Path

    recs = _recordings(SPEC_40, SPEC_26)
    recs[Path("B")][2]["cam1"] = (680, 680)
    with pytest.raises(ValueError, match="frame size differs"):
        da.pooled_intrinsics(recs, "cam1")


def test_a_board_with_no_recorded_geometry_is_REFUSED_not_guessed(tmp_path, monkeypatch):
    """camera_calibration_20260805's board.yaml is all zeros -- the geometry was never recorded and
    is not recoverable from the video. Marker ids 0-37 are equally consistent with a 7x11, a 9x9 and
    an 8x10 board, and square_mm sets the metric scale of every 3D distance downstream."""
    import yaml

    (tmp_path / "board.yaml").write_text(yaml.safe_dump(
        {"squares_x": 0, "squares_y": 0, "square_mm": 0.0, "marker_mm": 0.0,
         "dictionary": "DICT_4X4_50"}))
    with pytest.raises(ValueError, match="no usable geometry"):
        da.detect_rows(tmp_path, step=10)


def test_a_MISSING_board_yaml_is_announced_rather_than_silently_defaulted(tmp_path, capsys):
    """`board_for` falls back to configs/defaults.yaml when a recording carries no sidecar -- the
    right default, and dangerous in silence. Both our boards are 6x6 DICT_4X4_50 and differ only in
    millimetres, so the fallback yields the same corner ids at the wrong scale and every metric
    result is off by the ratio with nothing downstream able to tell."""
    with pytest.raises(FileNotFoundError):   # no videos in tmp_path; the warning fires first
        da.detect_rows(tmp_path, step=25)
    assert "no board.yaml" in capsys.readouterr().out


# ------------------------------------------------------------------ the rig-unmoved assumption

@requires_aniposelib
def test_pooled_extrinsics_are_refused_when_the_assumption_CANNOT_BE_TESTED(monkeypatch):
    """The 2026-08-05 / 2026-09-10 case exactly. Pooling extrinsics asserts the cameras did not move
    in five weeks; with no pair solvable in both recordings there is no measurement that could
    contradict it. Untestable is not the same as true."""
    monkeypatch.setattr(da, "detect_rows",
                        lambda d, *a, **k: (["cam1"], {"cam1": []}, {"cam1": (600, 600)}, SPEC_40))
    monkeypatch.setattr(da, "pooled_intrinsics",
                        lambda *a, **k: (np.eye(3), np.zeros(5), 0.5, 30, {"A": 30}))
    monkeypatch.setattr(da, "intrinsics_quality",
                        lambda *a, **k: {"identifiable": True, "problems": []})
    monkeypatch.setattr(da, "cross_recording_agreement", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="untestable"):
        da.solve(["A", "B"], assume_rig_unmoved=True)


@requires_aniposelib
def test_pooled_extrinsics_are_refused_when_the_rig_DID_move(monkeypatch):
    monkeypatch.setattr(da, "detect_rows",
                        lambda d, *a, **k: (["cam1"], {"cam1": []}, {"cam1": (600, 600)}, SPEC_40))
    monkeypatch.setattr(da, "pooled_intrinsics",
                        lambda *a, **k: (np.eye(3), np.zeros(5), 0.5, 30, {"A": 30}))
    monkeypatch.setattr(da, "intrinsics_quality",
                        lambda *a, **k: {"identifiable": True, "problems": []})
    monkeypatch.setattr(da, "cross_recording_agreement", lambda *a, **k: [
        {"pair": "cam1-cam4", "deg": 4.2, "mm": 9.1, "ok": False, "a": "A", "b": "B",
         "n_a": 20, "n_b": 20}])
    with pytest.raises(RuntimeError, match="rig-unmoved test FAILED"):
        da.solve(["A", "B"], assume_rig_unmoved=True)


def test_the_drift_tolerance_is_tighter_than_the_tracking_it_serves():
    """A degree of rotation at ~100 mm is ~1.7 mm of transverse error. There is no point pooling a
    rig that moved more than the precision the 3D tracking is trying to reach."""
    assert np.degrees(np.arctan(da.MAX_RIG_DRIFT_MM / 100.0)) > da.MAX_RIG_DRIFT_DEG


# ------------------------------------------------------------------ the gate's verdicts

def test_the_gate_says_TOO_LARGE_when_a_camera_never_reaches_enough_corners():
    """cam1 on 2026-09-10: markers everywhere, never more than five interpolated corners, because a
    ChArUco corner needs its four surrounding markers and cam1 only ever sees a fragment."""
    from pathlib import Path

    recs = {Path("A"): (["cam1"], {"cam1": [_row(f, 5) for f in (0, 500, 1000)]},
                        {"cam1": (600, 600)}, SPEC_40)}
    v = da.gate(recs)[str(Path("A"))]["cameras"]["cam1"]
    assert "too" in v["verdict"].lower() and "LARGE" in v["verdict"]


def test_the_gate_says_WEAK_when_the_corners_are_there_but_the_board_was_held():
    from pathlib import Path

    recs = {Path("A"): (["cam2"], {"cam2": [_row(f, 20) for f in range(400)]},
                        {"cam2": (600, 450)}, SPEC_40)}
    v = da.gate(recs)[str(Path("A"))]["cameras"]["cam2"]
    assert v["verdict"].startswith("WEAK") and v["poses_intrinsic"] == 1


def test_the_gate_passes_a_camera_with_enough_corners_in_enough_poses():
    from pathlib import Path

    rows = [_row(f, 20) for f in range(0, da.MIN_POSES * 500, 500)]
    recs = {Path("A"): (["cam2"], {"cam2": rows}, {"cam2": (600, 450)}, SPEC_40)}
    assert da.gate(recs)[str(Path("A"))]["cameras"]["cam2"]["verdict"] == "OK"


# ------------------------------------------------------------------ caching

def test_the_detection_cache_key_moves_with_anything_that_changes_detection():
    """A cache that survived a board change would answer a question about one board with detections
    from another."""
    assert da._digest(SPEC_40, 10) != da._digest(SPEC_26, 10)
    assert da._digest(SPEC_40, 10) != da._digest(SPEC_40, 25)
    assert da._digest(SPEC_40, 10) == da._digest(dict(SPEC_40), 10)


# ------------------------------------------------------------------ the principal point is MEASURED

def test_the_principal_point_comes_from_the_ROI_not_the_frame_centre():
    """Every camera records a crop off a 1280x1024 sensor and the optical axis is at the SENSOR
    centre. cam1's ROI is 600x600 at (440,312), so its principal point is at (200,200) in frame
    coordinates -- 141 px from the frame centre, which is error the solve would have to absorb."""
    assert da.measured_principal_point("cam1", (600, 600)) == (200.0, 200.0)
    assert da.measured_principal_point("cam2", (600, 450)) == (452.0, 276.0)
    assert da.measured_principal_point("cam4", (680, 680)) == (332.0, 380.0)


def test_a_RECROPPED_camera_falls_back_rather_than_applying_a_stale_offset():
    """cam2 and cam3 are both 600x450 and their recorded offsets differ by 156 px, so a stale entry
    is not a small error -- it is a confident 150 px offset in the wrong direction, which is worse
    than assuming the frame centre. The ROI size is checked against the actual video."""
    assert da.measured_principal_point("cam1", (800, 600)) is None
    assert da.measured_principal_point("cam9", (600, 600)) is None


def test_the_identifiability_check_scores_against_the_MEASURED_axis():
    """cam2's optical axis is 0.25 image-widths from its frame centre. Scored against the centre, a
    CORRECT solve would be reported as a failure."""
    K = np.array([[1779.0, 0, 452.0], [0, 1779.0, 276.0], [0, 0, 1.0]])
    d = np.array([-0.1, 0, 0, 0, 0])
    assert da.intrinsics_quality(K, d, (600, 450), expect_pp=(452.0, 276.0))["identifiable"]
    assert not da.intrinsics_quality(K, d, (600, 450))["identifiable"], \
        "against the frame centre the same solve must look wrong -- that is why expect_pp exists"
