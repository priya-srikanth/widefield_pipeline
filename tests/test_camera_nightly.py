"""Tests for the camera nightly orchestrator (wfield_local.camera_nightly)."""
from wfield_local import camera_nightly as cn
from wfield_local import camera_sync, dropframe_qc


class _RV:
    """Minimal PathResolver stub (the real step functions are monkeypatched out)."""
    machine = "analysis"

    def resolve(self, root, rel):
        return f"/{root}/{rel}"

    def root(self, name):
        return f"/{name}"


def _patch(monkeypatch):
    calls = []
    monkeypatch.setattr(dropframe_qc, "run", lambda *a, **k: calls.append(("drop", a, k)))
    monkeypatch.setattr(camera_sync, "run", lambda *a, **k: calls.append(("align", a, k)))
    return calls


def test_run_dispatches_both_in_order_and_threads_animals(monkeypatch):
    calls = _patch(monkeypatch)
    cn.run("20260807", _RV(), animals=["PS94"])
    assert [c[0] for c in calls] == ["drop", "align"]           # QC before alignment
    assert calls[0][2].get("animals") == ["PS94"]               # --only threaded to dropframe
    assert calls[1][2].get("animals") == ["PS94"]               # ...and to camera_sync
    assert calls[0][1][0] == "/behavior_cameras/20260807"       # dropframe scans the date's cam dir


def test_skip_flags(monkeypatch):
    calls = _patch(monkeypatch)
    cn.run("20260807", _RV(), do_dropframe=False)
    assert [c[0] for c in calls] == ["align"]
    calls.clear()
    cn.run("20260807", _RV(), do_align=False)
    assert [c[0] for c in calls] == ["drop"]
