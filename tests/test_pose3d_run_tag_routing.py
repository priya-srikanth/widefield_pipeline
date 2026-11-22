"""11-21 four-camera re-run: outputs of a run on a non-default 3-D source go to their own tagged folders (VLS-paper
refresh rule: new names only); the default two-camera run is byte-identical."""
import importlib
import sys
from pathlib import Path


def _clean(monkeypatch):
    for k in ("WIDEFIELD_POSE3D_SOURCE", "WIDEFIELD_POSE3D_TAG"):
        monkeypatch.delenv(k, raising=False)


def test_default_2cam_run_is_untagged(monkeypatch):
    from wfield_local import pose3d, run_mode
    _clean(monkeypatch)
    assert run_mode.pose3d_part() == "" and "p3d" not in run_mode.tag()
    assert pose3d.pose3d_dir(root="R").name == "pose3d"
    monkeypatch.setenv("WIDEFIELD_POSE3D_TAG", "_3net_20261121")     # a tag never applies to 2cam
    assert run_mode.pose3d_part() == "" and pose3d.pose3d_dir(root="R").name == "pose3d"


def test_4cam_run_tags_outputs_and_reads_the_tagged_3d(monkeypatch, tmp_path):
    from wfield_local import pose3d, run_mode
    _clean(monkeypatch)
    monkeypatch.setenv("WIDEFIELD_POSE3D_SOURCE", "4cam")
    monkeypatch.setenv("WIDEFIELD_POSE3D_TAG", "_3net_20261121")
    assert pose3d.pose3d_dir(root="R").name == "pose3d_4cam_3net_20261121"
    assert run_mode.pose3d_part() == "p3d-4cam_3net_20261121"
    assert "p3d-4cam_3net_20261121" in run_mode.tag()
    assert "p3d-4cam_3net_20261121" not in run_mode.cache_parts()   # output-only: imaging caches unaffected
    assert run_mode.subdir(tmp_path) != Path(tmp_path)              # routed output roots move under the tag


def test_within_lick_correction_and_lick_correction_read_pose3d_dir():
    """No literal session_poses/pose3d reads left in the Phase A gate-4 scripts (they follow the selected source)."""
    root = Path(__file__).resolve().parents[1] / "scripts"
    for f in ("within_lick_correction.py", "lick_correction.py"):
        src = (root / f).read_text(encoding="utf-8")
        assert '"session_poses" / "pose3d"' not in src, f
        assert "pose3d_dir(" in src or "kin_all_path(" in src, f      # 11-22: + the lick set (iti_licks.kin_all_path)


def test_within_lick_correction_out_root_moves_with_the_4cam_tag(monkeypatch):
    _clean(monkeypatch)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import within_lick_correction as W
    W = importlib.reload(W)
    base = W.out_root()
    monkeypatch.setenv("WIDEFIELD_POSE3D_SOURCE", "4cam")
    monkeypatch.setenv("WIDEFIELD_POSE3D_TAG", "_3net_20261121")
    tagged = W.out_root()
    assert tagged != base and "p3d-4cam_3net_20261121" in str(tagged)
