"""LP arm: a never-filmed session contributes nothing instead of failing the camera-gated map collection (2026-11-06).

Render g5_restnl_maps_lp died in `map_collect.collect` (review N9: any failed job is fatal under the camera gate) on
``PS93_0813: LPUnavailable: no LP lick tables`` -- PS93_0813 has no behavior video (docs/EXPERIMENT_ERRORS.md), so no LP
licks and no camera working period. Under the contact arm it already contributes nothing. The ref-map / beta-map job
wrappers now return that same empty result for such a session, and ONLY for it: a filmed session with missing LP tables
still fails. Synthetic: no share access.
"""
import pytest

from wfield_local import beta_maps as bm
from wfield_local import lp_licks, map_collect, trial_classes
from wfield_local import position_reference_maps as prm
from wfield_local.paths import PathResolver

S = {"label": "AX_0813", "mc": ""}


def _raise_lp(*_a, **_k):
    raise lp_licks.LPUnavailable("AX_0813: no LP lick tables in <somewhere>")


@pytest.fixture
def cams(tmp_path, monkeypatch):
    """A behavior-camera root on tmp; no camera classes unless a test says otherwise."""
    root = tmp_path / "Behavior_Cameras"
    root.mkdir()
    monkeypatch.setattr(PathResolver, "root", lambda self, key: str(root) if key == "behavior_cameras" else "")
    monkeypatch.setattr(trial_classes, "session_trials", lambda label, *a, **k: None)
    monkeypatch.setattr(trial_classes, "camera_gate_on", lambda: True)
    monkeypatch.setattr(prm, "session_raw_maps_cached", _raise_lp)
    monkeypatch.setattr(bm, "session_maps_cached", _raise_lp)
    return root


def _prm_job():
    return ("AX_0813", S, "cue", 2.0, "working")


def _bm_job():
    return ("AX_0813", S, "cue", 2.0, None, "working")


def test_no_video_session_contributes_nothing(cams, capsys):
    by, fail = map_collect.collect([_prm_job()], prm._raw_maps_job, label="ref-map", jobs=1)
    assert fail == [] and by["AX_0813"] == dict(prm._empty_parts(), skipped_no_video=True)
    assert "AX_0813: skipped: no LP licks (no video)" in capsys.readouterr().out
    by, fail = map_collect.collect([_bm_job()], bm._session_maps_job, label="beta-map", jobs=1)
    assert fail == [] and by["AX_0813"] == ({}, {})


def test_empty_parts_matches_the_contact_arm_empty_session():
    assert prm._empty_parts() == prm._as_parts(({}, {}, None, None, {}, {}))


@pytest.mark.parametrize("worker,job", [(prm._raw_maps_job, _prm_job()), (bm._session_maps_job, _bm_job())])
def test_filmed_session_missing_lp_tables_stays_fatal(cams, worker, job):
    (cams / "20260813" / "AX").mkdir(parents=True)
    (cams / "20260813" / "AX" / "cam4_rec.avi").write_bytes(b"")
    with pytest.raises(RuntimeError, match="camera trial gate: 1 .*LPUnavailable"):
        map_collect.collect([job], worker, label="x", jobs=1)


def test_session_with_camera_classes_stays_fatal(cams, monkeypatch):
    monkeypatch.setattr(trial_classes, "session_trials", lambda label, *a, **k: object())
    with pytest.raises(RuntimeError, match="LPUnavailable"):
        map_collect.collect([_prm_job()], prm._raw_maps_job, label="ref-map", jobs=1)


def test_unreadable_camera_root_stays_fatal(cams, monkeypatch, tmp_path):
    monkeypatch.setattr(PathResolver, "root", lambda self, key: str(tmp_path / "absent"))
    assert not map_collect.session_has_no_video("AX_0813")
    with pytest.raises(RuntimeError, match="LPUnavailable"):
        map_collect.collect([_prm_job()], prm._raw_maps_job, label="ref-map", jobs=1)


def test_other_errors_are_untouched(cams, monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("bad data")
    monkeypatch.setattr(prm, "session_raw_maps_cached", boom)
    with pytest.raises(RuntimeError, match="ValueError: bad data"):
        map_collect.collect([_prm_job()], prm._raw_maps_job, label="ref-map", jobs=1)


def test_skipped_sessions_are_named_and_counted_in_the_caption(cams, monkeypatch, capsys):
    """Priya 11-17 (15r REST-reference LP lick maps, PS93_0813): the collection drops a never-filmed session, NAMES it in
    the log and the 15r / 15rpa captions COUNT it; a filmed session is kept."""
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local import grant_figures, locanmf_cue_lick_analysis

    good = {"label": "AX_0806", "mc": ""}

    def maps(session, *_a, **_k):
        if session["label"] == "AX_0813":
            _raise_lp()
        return prm._empty_parts()            # filmed session, no position over the floor: kept, contributes nothing

    monkeypatch.setattr(prm, "session_raw_maps_cached", maps)
    monkeypatch.setattr(grant_figures, "ANIMALS", ("AX",))
    monkeypatch.setattr(grant_figures, "_day", lambda an, d: -1 if d == "0806" else 5)
    monkeypatch.setattr(ef, "epoch_of_day", lambda an, d: "acute")
    monkeypatch.setattr(locanmf_cue_lick_analysis, "SESSIONS", [good, S])
    monkeypatch.setattr(config, "phase_labels", lambda ph: ["AX_0806"] if ph == "pre" else ["AX_0813"])
    monkeypatch.setattr(prm, "derived_active", lambda: ())
    monkeypatch.setenv("WIDEFIELD_MAP_JOBS", "1")
    prm.maps_by_epoch.cache_clear()
    try:
        prm.maps_by_epoch("lick", "lick")
    finally:
        prm.maps_by_epoch.cache_clear()
    out = capsys.readouterr().out
    assert "ref-map: 1 session(s) skipped, no camera / LP lick table (no behavior video): AX_0813" in out
    assert prm.skipped_no_video("lick", "lick") == ["AX_0813"]
    assert "no camera / LP lick table (never filmed): 1 (AX_0813)" in prm.skipped_caption("lick", "lick")
    assert prm.skipped_caption("cue", "nothing-run").startswith("Sessions skipped for having no camera / LP lick "
                                                                 "table: 0")
