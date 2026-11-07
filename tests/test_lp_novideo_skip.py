"""LP arm: a never-filmed session contributes nothing instead of failing the camera-gated map collection (2026-11-06).

Render g5_restnl_maps_lp died in `map_collect.collect` (review N9: any failed job is fatal under the camera gate) on
``PS93_0813: LPUnavailable: no LP lick tables`` -- PS93_0813 has no behaviour video (docs/EXPERIMENT_ERRORS.md), so no LP
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
    """A behaviour-camera root on tmp; no camera classes unless a test says otherwise."""
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
    assert fail == [] and by["AX_0813"] == prm._empty_parts()
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
