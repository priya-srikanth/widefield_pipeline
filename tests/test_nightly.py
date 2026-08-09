"""Tests for the machine-dispatched nightly wrapper (wfield_local.nightly)."""
from wfield_local import nightly


def _capture(monkeypatch):
    cmds = []
    monkeypatch.setattr(nightly, "_run", lambda args, dry: cmds.append([str(a) for a in args]))
    monkeypatch.setattr(nightly, "_available_dates", lambda rv: [])   # explicit dates pass through
    return cmds


def test_imaging_sequence_never_cleans(monkeypatch):
    cmds = _capture(monkeypatch)
    nightly.main(["20260807", "20260808", "--machine", "imaging"])
    steps = [c[0] for c in cmds]
    assert steps[0] == "wfield_local.preprocess"
    assert "wfield_local.preprocess_deck" in steps
    # archive + verify per date, in order after preprocess
    assert ["wfield_local.archive_day", "archive", "--date", "20260807", "--machine", "imaging"] in cmds
    assert ["wfield_local.archive_day", "verify", "--date", "20260808", "--machine", "imaging"] in cmds
    # NEVER auto-delete
    assert not any("clean" in c or "--execute" in c for c in cmds)


def test_analysis_sequence_camera_before_figs(monkeypatch):
    cmds = _capture(monkeypatch)
    nightly.main(["20260807", "--machine", "analysis", "--only", "PS94", "--from", "0606,0807"])
    steps = [c[0] for c in cmds]
    # camera_nightly (which uploads cameras + behavior logs) runs before nightly_figs
    assert steps == ["wfield_local.camera_nightly", "wfield_local.nightly_figs"]
    assert cmds[0][:2] == ["wfield_local.camera_nightly", "20260807"]
    assert "--only" in cmds[0] and "PS94" in cmds[0]
    assert "--from" in cmds[1] and "0606,0807" in cmds[1]


def test_skip_flags(monkeypatch):
    cmds = _capture(monkeypatch)
    nightly.main(["20260807", "--machine", "analysis", "--skip-camera"])
    assert [c[0] for c in cmds] == ["wfield_local.nightly_figs"]
    cmds.clear()
    nightly.main(["20260807", "--machine", "imaging", "--skip-deck", "--skip-archive"])
    assert [c[0] for c in cmds] == ["wfield_local.preprocess"]
