"""11-06 (queue v9k k_spout_error): `pose3d_kinematics.session_result` cached into the unrouted
`session_poses/pose3d/_cache/kinematics_<mode>`. Its 10-26 entries went stale when pose3d.py changed (11-04), and the
recompute's write was refused by the rebuild fence, so every 3-D reader that calls it (spout_error, phase, vs_cam4,
summary_fig, jaw_crosscheck, kinematics_pooled_figs) failed under the rebuild root. It now caches through
`trial_classes.kin_cache_dir`, the folder every other 3-D kinematics writer uses.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_session_result_caches_through_kin_cache_dir():
    src = (REPO / "scripts" / "pose3d_kinematics.py").read_text(encoding="utf-8")
    assert 'tc.kin_cache_dir(d3, f"kinematics_{mode}")' in src
    assert 'd3 / "_cache"' not in src


def test_kin_cache_dir_routes_under_a_rebuild_root(tmp_path, monkeypatch):
    from wfield_local import trial_classes as tc
    d3 = tmp_path / "session_poses" / "pose3d"
    monkeypatch.delenv("WIDEFIELD_REBUILD_ROOT", raising=False)
    assert tc.kin_cache_dir(d3, "kinematics_cam4") == d3 / "_cache" / "kinematics_cam4"


def test_jaw_crosscheck_writes_and_is_read_through_kin_out_dir():
    """11-06 (v9k2 k_jaw_xcheck): the writer wrote session_poses/pose3d/crosscheck directly (refused by the rebuild fence);
    writer and the pose-validation reader now share `kin_out_dir(..., "crosscheck", script="pose3d_jaw_crosscheck")`."""
    w = (REPO / "scripts" / "pose3d_jaw_crosscheck.py").read_text(encoding="utf-8")
    r = (REPO / "scripts" / "rebuild_figs" / "fig_pose_validation_sessions.py").read_text(encoding="utf-8")
    key = 'kin_out_dir(d3, "crosscheck", script="pose3d_jaw_crosscheck")'
    assert key in w and 'd3 / "crosscheck"' not in w
    assert 'kin_out_dir(P3, "crosscheck", script="pose3d_jaw_crosscheck") / "jaw_agreement.csv"' in r
