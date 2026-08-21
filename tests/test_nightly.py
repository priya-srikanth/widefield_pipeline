"""Tests for the machine-dispatched nightly wrapper (wfield_local.nightly)."""
from wfield_local import nightly


def _capture(monkeypatch):
    cmds = []
    monkeypatch.setattr(nightly, "_run", lambda args, dry: cmds.append([str(a) for a in args]))
    monkeypatch.setattr(nightly, "_available_dates", lambda rv: [])   # explicit dates pass through
    # these tests assert on the COMMAND SEQUENCE, not on whether raw data exists; the
    # raw-presence guard has its own tests below.
    monkeypatch.setattr(nightly, "_assert_has_raw", lambda dates, rv: None)
    return cmds


def test_imaging_sequence_never_cleans(monkeypatch):
    cmds = _capture(monkeypatch)
    nightly.main(["20260807", "20260808", "--machine", "imaging"])
    steps = [c[0] for c in cmds]
    # DAQ .h5 pushed to MICROSCOPE FIRST (both dates), before preprocess (byte-verified: --hash)
    assert cmds[0] == ["wfield_local.archive_day", "upload-daq", "--date", "20260807", "--hash", "--machine", "imaging"]
    assert ["wfield_local.archive_day", "upload-daq", "--date", "20260808", "--hash", "--machine", "imaging"] in cmds
    assert steps.index("wfield_local.preprocess") > steps.index("wfield_local.archive_day")
    assert "wfield_local.preprocess_deck" in steps
    # archive + verify per date, in order after preprocess (byte-verified: --hash)
    assert ["wfield_local.archive_day", "archive", "--date", "20260807", "--hash", "--machine", "imaging"] in cmds
    assert ["wfield_local.archive_day", "verify", "--date", "20260808", "--hash", "--machine", "imaging"] in cmds
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


def test_analysis_figs_gated_on_locanmf_registration(monkeypatch):
    # Stage 2 (figs) defers unless the date's sessions are registered in configs/sessions.yaml
    # (the proxy for "LocaNMF done"). The gate calls the REAL config.load_sessions (not mocked).
    cmds = _capture(monkeypatch)
    nightly.main(["20261225", "--machine", "analysis"])   # 12/25 is never a recording date
    assert [c[0] for c in cmds] == ["wfield_local.camera_nightly"]  # camera ran; figs DEFERRED
    # --figs overrides the gate (curated-refresh on demand)
    cmds.clear()
    nightly.main(["20261225", "--machine", "analysis", "--figs"])
    assert [c[0] for c in cmds] == ["wfield_local.camera_nightly", "wfield_local.nightly_figs"]
    # a REGISTERED date still runs figs by default
    cmds.clear()
    nightly.main(["20260807", "--machine", "analysis"])
    assert "wfield_local.nightly_figs" in [c[0] for c in cmds]


def test_perday_figs_incomplete_backfill_detection(tmp_path, monkeypatch):
    from wfield_local import nightly_figs as nf
    # a registered date whose per-day cue-decode fig is absent -> incomplete (would be backfilled)
    monkeypatch.setattr(nf.config, "load_sessions", lambda dates=None, **k: [{"label": "PS92_0809"}])
    assert nf._perday_figs_incomplete(str(tmp_path), "0809") is True
    (tmp_path / "locanmf_position_session_PS92_0809_locanmf_cue_base-none_cv-block.png").write_bytes(b"x")
    assert nf._perday_figs_incomplete(str(tmp_path), "0809") is False   # fig now present -> complete
    # a date with no registered sessions is never flagged (nothing to generate)
    monkeypatch.setattr(nf.config, "load_sessions", lambda dates=None, **k: [])
    assert nf._perday_figs_incomplete(str(tmp_path), "0810") is False


def test_publish_figs_incremental_pngs_only(tmp_path):
    from wfield_local import nightly_figs as nf
    src = tmp_path / "out"; src.mkdir()
    (src / "a.png").write_bytes(b"aaaa")
    (src / "b.png").write_bytes(b"bbbb")
    (src / "notes.txt").write_bytes(b"x")            # non-PNG must be ignored
    dst = tmp_path / "dst"

    class _RV:
        def root(self, k): return str(dst)

    assert nf._publish_figs(str(src), _RV()) == 2     # both PNGs copied
    assert (dst / "a.png").exists() and (dst / "b.png").exists() and not (dst / "notes.txt").exists()
    assert nf._publish_figs(str(src), _RV()) == 0     # unchanged -> nothing re-copied
    (src / "a.png").write_bytes(b"aaaaAAAA")          # size change -> only that one re-copies
    assert nf._publish_figs(str(src), _RV()) == 1


def test_analysis_await_locanmf_hands_off_to_poller(monkeypatch):
    cmds = _capture(monkeypatch)
    nightly.main(["20261225", "--machine", "analysis", "--await-locanmf", "--dry-run"])
    steps = [c[0] for c in cmds]
    assert steps == ["wfield_local.camera_nightly", "wfield_local.await_locanmf"]
    # dry-run hands the poller a single, no-write pass
    await_cmd = cmds[1]
    assert "20261225" in await_cmd and "--once" in await_cmd and "--dry-run" in await_cmd


def test_skip_flags(monkeypatch):
    cmds = _capture(monkeypatch)
    nightly.main(["20260807", "--machine", "analysis", "--skip-camera"])
    assert [c[0] for c in cmds] == ["wfield_local.nightly_figs"]
    cmds.clear()
    nightly.main(["20260807", "--machine", "imaging", "--skip-deck", "--skip-archive"])
    assert [c[0] for c in cmds] == ["wfield_local.archive_day", "wfield_local.preprocess"]  # upload-daq, then preprocess
    cmds.clear()
    nightly.main(["20260807", "--machine", "imaging", "--skip-deck", "--skip-archive", "--skip-daq-upload"])
    assert [c[0] for c in cmds] == ["wfield_local.preprocess"]


def test_imaging_branch_refuses_when_there_is_no_raw_data(tmp_path, monkeypatch):
    """The analysis box resolves to machine='imaging' and must NOT run the preprocessing branch.

    `detect_machine` tests `E:/labcams_data` FIRST, and the analysis box has that directory -- empty.
    So `nightly <date>` there dispatched to upload-daq -> preprocess -> preprocess_deck -> archive on
    a machine holding no raw data. Path resolution is fine on that box (N: really is MICROSCOPE); it
    is the dispatch that is wrong, so the guard asks the question the branch actually depends on:
    is there anything to preprocess?
    """
    import pytest

    from wfield_local import nightly

    monkeypatch.setattr(nightly, "list_raw_dates", lambda rv: ["20260819"], raising=False)
    monkeypatch.setattr("wfield_local.preprocess.list_raw_dates", lambda rv: ["20260819"])

    class _RV:
        machine = "imaging"

        def root(self, name):
            return str(tmp_path / name)

    with pytest.raises(SystemExit, match="no raw acquisition data"):
        nightly._assert_has_raw(["20260820"], _RV())


def test_imaging_branch_proceeds_when_the_raw_is_there(tmp_path, monkeypatch):
    from wfield_local import nightly

    monkeypatch.setattr("wfield_local.preprocess.list_raw_dates", lambda rv: ["20260819", "20260820"])

    class _RV:
        machine = "imaging"

        def root(self, name):
            return str(tmp_path / name)

    nightly._assert_has_raw(["20260820"], _RV())      # must not raise
