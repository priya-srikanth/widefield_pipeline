"""Tests for the camera nightly orchestrator (wfield_local.camera_nightly)."""
from pathlib import Path

from wfield_local import behavior_events, camera_nightly as cn
from wfield_local import camera_sync, dropframe_qc, spout_behavior


class _RV:
    """Minimal PathResolver stub. staging/resolve/root map under `base` (a local tmp dir, so the
    writeguard treats it as a non-share path and allows the copy)."""
    machine = "analysis"

    def __init__(self, base):
        self.base = Path(base)

    def staging(self, name):
        return str(self.base / ("cam" if "camera" in name else "logs"))

    def resolve(self, root, rel):
        return str(self.base / "server" / root / rel)

    def root(self, name):
        return str(self.base / "server" / name)


def _patch_steps(monkeypatch):
    calls = []
    monkeypatch.setattr(dropframe_qc, "run", lambda *a, **k: calls.append(("drop", a, k)))
    monkeypatch.setattr(camera_sync, "run", lambda *a, **k: calls.append(("align", a, k)))
    monkeypatch.setattr(behavior_events, "run", lambda *a, **k: calls.append(("events", a, k)))
    monkeypatch.setattr(spout_behavior, "run", lambda *a, **k: calls.append(("behavior", a, k)))
    return calls


def test_dispatch_order_and_animals_threading(tmp_path, monkeypatch):
    calls = _patch_steps(monkeypatch)
    cn.run("20260807", _RV(tmp_path), animals=["PS94"], do_copy=False)     # no staging -> skip copy
    assert [c[0] for c in calls] == ["drop", "align", "events", "behavior"]   # QC->align->events->figs
    assert all(c[2].get("animals") == ["PS94"] for c in calls)            # animals threaded through
    assert calls[0][1][0] == str(tmp_path / "server" / "behavior_cameras" / "20260807")


def test_skip_flags(tmp_path, monkeypatch):
    calls = _patch_steps(monkeypatch)
    cn.run("20260807", _RV(tmp_path), do_copy=False, do_dropframe=False)
    assert [c[0] for c in calls] == ["align", "events", "behavior"]
    calls.clear()
    cn.run("20260807", _RV(tmp_path), do_copy=False, do_align=False)
    assert [c[0] for c in calls] == ["drop", "events", "behavior"]
    calls.clear()
    cn.run("20260807", _RV(tmp_path), do_copy=False, do_behavior=False, do_events=False)
    assert [c[0] for c in calls] == ["drop", "align"]


def test_upload_copies_size_verified_and_never_deletes_source(tmp_path, monkeypatch):
    _patch_steps(monkeypatch)
    rv = _RV(tmp_path)
    # fake D: staging: camera D:/cam/<date>/<PSxx>/cam1.csv ; behavior log D:/logs/<PSxx>_<date>_x/trials.csv
    cam = tmp_path / "cam" / "20260807" / "PS94"
    cam.mkdir(parents=True)
    (cam / "cam1_2026-08-07T00_00_00.csv").write_text("1,2,12\n")
    (cam / "cam1_2026-08-07T00_00_00.avi").write_bytes(b"\x00" * 2048)
    log = tmp_path / "logs" / "PS94_20260807_101010"
    log.mkdir(parents=True)
    (log / "trials.csv").write_text("pos_idx,pos_name\n0,far_L\n")

    res, fails = cn.upload("20260807", rv, dry=False)
    assert not fails and res["ok"] == 3 and res["FAIL"] == 0
    # copied to the server tree, byte-for-byte
    dst_cam = tmp_path / "server" / "behavior_cameras" / "20260807" / "PS94" / "cam1_2026-08-07T00_00_00.avi"
    dst_log = tmp_path / "server" / "behavior_logs" / "PS94_20260807_101010" / "trials.csv"
    assert dst_cam.exists() and dst_cam.stat().st_size == 2048
    assert dst_log.read_text().startswith("pos_idx")
    # source (D:) is UNTOUCHED
    assert (cam / "cam1_2026-08-07T00_00_00.avi").exists() and (log / "trials.csv").exists()
    # idempotent: a second upload skips (same size)
    res2, _ = cn.upload("20260807", rv, dry=False)
    assert res2["skip"] == 3 and res2["ok"] == 0


def test_upload_animal_filter_and_dry_run(tmp_path, monkeypatch):
    _patch_steps(monkeypatch)
    rv = _RV(tmp_path)
    for a in ("PS94", "PS95"):
        d = tmp_path / "cam" / "20260807" / a
        d.mkdir(parents=True)
        (d / "cam1_x.csv").write_text("1,2,12\n")
    res, _ = cn.upload("20260807", rv, animals=["PS94"], dry=True)
    assert res["dry"] == 1 and res["ok"] == 0            # only PS94, and dry -> nothing written
    assert not (tmp_path / "server").exists()
