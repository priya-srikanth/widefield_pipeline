"""Priya 11-05 (relayed by a8 via the coordinator): behavior-only figures move to the CAMERA working period +
August-only pre.

Pins:
  * `trial_classes.camera_quit` (A6, ENGAGEMENT_CAMERA_2026-11-05): first cue of the tail after the last camera lick, a
    quit only with >= 6 tail cues starting >= 300 s after the first cue; no video -> no camera quit (has_classes False);
  * disengagement_tests takes its quit from it (DISENGAGEMENT_QUIT=daq = the prereg's DAQ row) and leaves no-video
    sessions out of the camera rows;
  * revup_behaviour reads THIS run's attempts table (`lp_licks.attempts_path`), never the untagged qc one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from wfield_local import trial_classes as tc

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _forget_new_modules():
    """Importing disengagement_tests sets WIDEFIELD_PRE_FROM=0801 before its config imports; modules first imported then
    keep the August-only values for the rest of the session (broke test_rebuild_root / test_lp_gate_option when run
    after this file). monkeypatch.delenv on an UNSET variable records nothing to undo, so that setdefault outlived the
    test: restore the whole environment and drop every module this test imported so later tests import them fresh."""
    before, env = set(sys.modules), dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(env)
    for k in set(sys.modules) - before:
        sys.modules.pop(k, None)


def _stub(monkeypatch, cls, kind):
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs, w=None: np.asarray(cls, dtype=object))
    monkeypatch.setattr(tc, "stop_kind_for", lambda lab, cs, w=None: np.asarray(kind, dtype=object))
    monkeypatch.setattr(tc, "stop_run_min", lambda: 6)
    monkeypatch.setattr(tc, "stop_min_start_s", lambda: 300.0)


def test_camera_quit_first_tail_cue(monkeypatch):
    n = 60
    cls = ["contact"] * 50 + ["stopped"] * 10
    kind = [""] * 50 + ["tail"] * 10
    _stub(monkeypatch, cls, kind)
    q = tc.camera_quit("PS92_0810", np.arange(n) * 10.0)
    assert q == {"quit_cue": 50, "censored": False, "has_classes": True, "n_tail": 10}


def test_camera_quit_censored_short_or_early_tail(monkeypatch):
    _stub(monkeypatch, ["contact"] * 55 + ["stopped"] * 5, [""] * 55 + ["tail"] * 5)       # 5 < 6 tail cues
    assert tc.camera_quit("PS92_0810", np.arange(60) * 10.0)["censored"]
    _stub(monkeypatch, ["contact"] * 20 + ["stopped"] * 40, [""] * 20 + ["tail"] * 40)     # starts at 200 s < 300 s
    assert tc.camera_quit("PS92_0810", np.arange(60) * 10.0)["censored"]


def test_camera_quit_no_video(monkeypatch):
    _stub(monkeypatch, ["unknown"] * 60, [""] * 60)
    q = tc.camera_quit("PS93_0813", np.arange(60) * 10.0)
    assert q["censored"] and not q["has_classes"] and q["quit_cue"] == -1


@pytest.fixture
def dis(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    monkeypatch.delenv("DISENGAGEMENT_QUIT", raising=False)
    monkeypatch.delenv("WIDEFIELD_PRE_FROM", raising=False)     # the module setdefaults it at import; restored after
    sys.modules.pop("disengagement_tests", None)
    import disengagement_tests as dt
    return dt


def test_disengagement_quit_is_the_camera_quit(dis, monkeypatch):
    assert dis.QUIT == "camera"
    rows = [{"order": o, "elapsed_s": float(o * 10)} for o in range(0, 60, 2)]       # every other cue is a trial row
    monkeypatch.setattr(tc, "camera_quit", lambda lab, cs: {"quit_cue": 51, "censored": False, "has_classes": True,
                                                             "n_tail": 9})
    q = dis.camera_quit_rows("PS92_0810", np.arange(60) * 10.0, rows)
    assert q["quit_idx"] == 26 and q["quit_t"] == 520.0 and not q["censored"]           # first row with order >= 51
    s = {"label": "PS92_0810", "quit_idx": 5, "quit_t": 50.0, "censored": False, "quit_cam": q}
    a = dis.apply_quit(s, "camera")
    assert (a["quit_idx"], a["quit_t"], a["censored"]) == (26, 520.0, False)
    assert dis.apply_quit(s, "daq") is s                                                # the labelled DAQ row
    nv = {**s, "quit_cam": {**q, "has_classes": False}}
    assert dis.apply_quit(nv, "camera") is None                                         # no video -> DAQ-only row
    with pytest.raises(SystemExit):
        dis.apply_quit({k: v for k, v in s.items() if k != "quit_cam"}, "camera")       # stale cache refused


def test_disengagement_daq_row_writes_beside_the_primary(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    monkeypatch.setenv("DISENGAGEMENT_QUIT", "daq")
    monkeypatch.delenv("WIDEFIELD_PRE_FROM", raising=False)
    sys.modules.pop("disengagement_tests", None)
    import disengagement_tests as dt
    try:
        assert dt.OUT.name == "disengagement_daqquit" and dt.SESS.parent.name == "disengagement"
    finally:
        sys.modules.pop("disengagement_tests", None)


def test_revup_behaviour_reads_the_runs_attempts_table(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("revup_behaviour", None)
    import revup_behaviour as rb

    from wfield_local import lp_licks
    monkeypatch.setattr(lp_licks, "attempts_path", lambda rv=None: tmp_path / "sel-camera" / "trial_attempts.csv")
    assert rb.attempts_csv() == tmp_path / "sel-camera" / "trial_attempts.csv"
    src = (REPO / "scripts" / "revup_behaviour.py").read_text(encoding="utf-8")
    assert 'QC / "trial_attempts.csv"' not in src
