"""The ChArUco survey is a GATE, so the ways it could wrongly say "go" are what is pinned here.

Every test below is a way the 2026-08-05 recording could have been passed as usable.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wfield_local import dlc_calibration as dc


#: Samples this far apart count as separate board POSES (POSE_GAP_S at 250 fps is 125 frames).
POSE_STRIDE = 250


def _survey(counts: dict[str, list[int]], stride: int = POSE_STRIDE) -> pd.DataFrame:
    """A survey frame from ``{cam: [n_markers per sample]}``; all cams share the frame grid.

    ``stride`` is frames between samples. The default puts each sample in its OWN pose, which is what
    most of these tests want; pass a small stride to build one long pose instead -- the distinction
    the module now turns on.
    """
    rows = []
    for cam, seq in counts.items():
        for i, n in enumerate(seq):
            rows.append({"cam": cam, "frame": i * stride, "n_markers": n,
                         "ids": " ".join(map(str, range(n)))})
    return pd.DataFrame(rows, columns=["cam", "frame", "n_markers", "ids"])


def test_a_board_held_STILL_is_one_pose_however_many_frames_it_fills():
    """The overcounting half of the frame-vs-pose correction.

    cam4 in the real recording has 8,969 usable sampled frames and FOUR distinct poses, because the
    board sat in front of it. A frame count called that the best camera in the rig; it is the worst
    constrained.
    """
    still = _survey({"cam1": [8] * 400}, stride=1)      # 400 consecutive frames = 1.6 s
    assert dc.per_camera(still).loc[0, "poses"] == 1
    assert not bool(dc.per_camera(still).loc[0, "ok"]), "400 frames of one pose is not a calibration"

    swept = _survey({"cam1": [8] * 400})                # 400 samples 1 s apart = 400 poses
    assert dc.per_camera(swept).loc[0, "poses"] == 400
    assert bool(dc.per_camera(swept).loc[0, "ok"])


def test_the_pose_count_does_not_depend_on_the_SAMPLING_STEP():
    """The undercounting half. cam1-cam2 read 11 co-visible frames at step 25 and 69 at step 5 --
    same recording, opposite verdict, because the threshold was on a step-dependent quantity."""
    coarse = dc.count_poses([0, 5000, 10000])                       # 3 poses, sparsely sampled
    dense = dc.count_poses([0, 10, 20, 5000, 5010, 10000, 10010])   # same 3, densely sampled
    assert coarse == dense == 3


def test_two_cameras_that_never_see_the_board_at_the_same_time_are_not_a_pair():
    """The failure that actually happened, in miniature.

    Both cameras clear the per-camera bar comfortably -- so a report that only counted detections
    per camera would call this recording fine. They are useless as a pair because the board is
    never in both at once, which is the only thing that constrains their relative pose.
    """
    n = 400
    df = _survey({"cam1": [8, 0] * n, "cam2": [0, 8] * n})
    cams = dc.per_camera(df)
    assert cams["ok"].all(), "both cameras individually clear MIN_CAM_FRAMES"

    pairs = dc.per_pair(df)
    assert int(pairs.loc[0, "both"]) == 0
    assert not bool(pairs.loc[0, "ok"])
    ok, comps = dc.connected(pairs, ["cam1", "cam2"])
    assert not ok and comps == [["cam1"], ["cam2"]]


def test_connectivity_does_not_require_every_pair():
    """cam2 and cam3 look at opposite sides of the animal and may share NO field of view.

    A demand for all six pairs would condemn a rig geometry that is perfectly calibratable through
    a chain, so connectivity -- not completeness -- is the criterion.
    """
    pairs = pd.DataFrame([
        {"cam_a": "cam1", "cam_b": "cam2", "both": 900, "ok": True},
        {"cam_a": "cam1", "cam_b": "cam3", "both": 900, "ok": True},
        {"cam_a": "cam1", "cam_b": "cam4", "both": 900, "ok": True},
        {"cam_a": "cam2", "cam_b": "cam3", "both": 0, "ok": False},
        {"cam_a": "cam2", "cam_b": "cam4", "both": 0, "ok": False},
        {"cam_a": "cam3", "cam_b": "cam4", "both": 0, "ok": False},
    ])
    ok, comps = dc.connected(pairs, ["cam1", "cam2", "cam3", "cam4"])
    assert ok and comps == [["cam1", "cam2", "cam3", "cam4"]]


def test_the_20260805_shape_is_reported_as_not_connected():
    """Regression on the real recording: one good pair, two orphans, and a verdict that says so."""
    n = 600
    df = _survey({
        "cam1": [8] * n + [0] * n,
        "cam4": [15] * n + [0] * n,
        "cam2": [1] * (2 * n - 5) + [8] * 5,     # sees markers often, 4+ almost never
        "cam3": [0] * (2 * n - 2) + [8] * 2,
    })
    pairs = dc.per_pair(df)
    ok, comps = dc.connected(pairs, ["cam1", "cam2", "cam3", "cam4"])
    assert not ok
    assert ["cam1", "cam4"] in comps and ["cam2"] in comps and ["cam3"] in comps

    text = "\n".join(dc.report_lines(df, "camera_calibration_20260805", 25))
    assert "NOT CONNECTED" in text
    assert "Fix: re-record" in text


def test_a_connected_graph_says_so_and_offers_no_fix():
    n = 400
    df = _survey({"cam1": [8] * n, "cam2": [8] * n})
    text = "\n".join(dc.report_lines(df, "camera_calibration_20260901", 25))
    assert "CONNECTED" in text and "NOT CONNECTED" not in text
    assert "Fix:" not in text


def test_a_camera_that_only_ever_sees_three_markers_is_unusable():
    """Three markers cannot pin a board pose, and MIN_MARKERS is the floor -- not a soft preference."""
    df = _survey({"cam1": [3] * 2000, "cam2": [8] * 2000})
    cams = dc.per_camera(df).set_index("cam")
    assert not bool(cams.loc["cam1", "ok"])
    assert cams.loc["cam1", "any_pct"] == 100.0, "detects markers on every frame, still unusable"


def _survey_ppb(counts: dict[str, list[int]], ppb: dict[str, float]) -> pd.DataFrame:
    """Like ``_survey`` but carrying the marker scale, so the two failure modes can be told apart."""
    df = _survey(counts)
    df["px_per_bit"] = [ppb.get(c, 0.0) if n > 0 else 0.0
                        for c, n in zip(df["cam"], df["n_markers"], strict=True)]
    df["n_rejected"] = [0 if n > 0 else 12 for n in df["n_markers"]]
    return df


def test_a_board_that_is_TOO_SMALL_reads_differently_from_one_never_shown():
    """The distinction the first version of this module missed, and Priya caught.

    A board present but unreadable and a board never presented both score ~0 usable frames, and they
    have opposite fixes: print it bigger versus sweep it into view. `px_per_bit` separates them --
    a DICT_4X4 marker is 6 cells across and below ~3 px/cell the square is still FOUND and then
    rejected, so the camera reports markers-seen but nothing decoded.
    """
    n = 400
    small = _survey_ppb({"cam1": [8] * n, "cam2": [1] * n},          # sees it, cannot read it
                        {"cam1": 5.2, "cam2": 2.1})
    text = "\n".join(dc.report_lines(small, "camera_calibration_20260805", 25))
    assert "BOARD TOO SMALL" in text
    assert "1.4x larger" in text, "the shortfall is quantified, not just named"
    assert "NEVER PRESENTED" not in text

    absent = _survey_ppb({"cam1": [8] * n, "cam2": [0] * n}, {"cam1": 5.2})
    text2 = "\n".join(dc.report_lines(absent, "camera_calibration_20260805", 25))
    assert "NEVER PRESENTED" in text2
    assert "BOARD TOO SMALL" not in text2


def test_the_fix_line_says_re_sweeping_will_not_help_when_the_board_is_too_small():
    n = 400
    df = _survey_ppb({"cam1": [8] * n, "cam2": [1] * n}, {"cam1": 5.2, "cam2": 1.5})
    text = "\n".join(dc.report_lines(df, "camera_calibration_20260805", 25))
    assert "re-sweeping the same board will not help" in text
    assert "2.0x larger" in text


def test_a_survey_without_the_scale_column_still_reports(tmp_path):
    """Older survey CSVs predate px_per_bit; reading one must not become a crash."""
    text = "\n".join(dc.report_lines(_survey({"cam1": [8] * 400, "cam2": [0] * 400}),
                                     "camera_calibration_20260805", 25))
    assert "RESULT:" in text


def test_the_newest_calibration_is_chosen_by_NAME_not_mtime(tmp_path):
    """Re-running the dropped-frame QC touches an old directory; that must not make it 'newest'."""
    old = tmp_path / "camera_calibration_20260805"
    new = tmp_path / "camera_calibration_20260910"
    for d in (old, new):
        d.mkdir()
    # Touch the OLD one last, the way a QC re-run would.
    (new / "cam1.avi").write_bytes(b"")
    (old / "dropped_frames_summary.txt").write_text("re-run", encoding="utf-8")

    assert dc.find_calibration_dir(root=tmp_path).name == "camera_calibration_20260910"


def test_an_unrelated_sibling_directory_is_not_mistaken_for_a_calibration(tmp_path):
    (tmp_path / "widefield").mkdir()
    (tmp_path / "Baseline_screenshots").mkdir()
    (tmp_path / "camera_calibration_20260805").mkdir()
    assert dc.find_calibration_dir(root=tmp_path).name == "camera_calibration_20260805"

    (tmp_path / "camera_calibration_20260805").rmdir()
    with pytest.raises(FileNotFoundError):
        dc.find_calibration_dir(root=tmp_path)


def test_the_calibration_root_is_the_parent_of_the_widefield_subdir():
    """`camera_calibration_*` sits BESIDE `widefield/`, not inside it.

    Pointing the root at `behavior_cameras` (which ends in `/Widefield`) would find nothing, and
    `find_calibration_dir` would raise on a recording that is present on disk.
    """
    from wfield_local.paths import PathResolver
    for machine in ("analysis", "imaging", "mac"):
        r = PathResolver(machine=machine)
        cal, cams = r.root("camera_calibration"), r.root("behavior_cameras")
        assert cams.lower().startswith(cal.lower())
        assert cams.rstrip("/").lower().endswith("/widefield")
