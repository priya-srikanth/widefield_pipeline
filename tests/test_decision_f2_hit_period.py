"""Priya 11-05, DECISIONS_PENDING_MERGE_2026-11-04 (f) 2: the hit-rate period is the CAMERA working period (first cue
-> last camera lick, minus the stopped runs); the DAQ quit-tail rule is a labeled robustness row. Fig 0.2
(`scripts/rebuild_figs/fig_behaviour_hits.py`) draws the camera rate as the primary and the DAQ one as
`behaviour_hits_by_spout_daq_quittail`.

Pins:
  * the camera classes of a session put the mid-session stop run AND the tail after the last camera lick outside the
    working period, and `trial_attempts.session_summary`'s frac_hit is contact / working-period trials (denominator =
    contact + incomplete + jaw_only + no_response; Priya 11-04 answer 2);
  * fig_behaviour_hits reads exactly that column for the primary and refuses a table built without the camera gate;
  * it writes the primary and the robustness figure under different names, the robustness one labeled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _path(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "rebuild_figs"))
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)


def _session():
    from wfield_local import trial_classes as TC
    raw = np.array(["contact"] * 40 + ["no_response"] * 7 + ["contact"] * 10 + ["incomplete", "jaw_only", "no_response"]
                   + ["contact"] * 5 + ["no_response"] * 8, dtype=object)
    pos = np.array(["close_L"] * len(raw), dtype=object)
    cls, kind = TC.classify_session(raw, None, pos, rule="both", min_run=6, end="lick", cue_s=np.arange(len(raw)) * 10.,
                                    min_start_s=300, lick_rule="none", return_kind=True)
    return raw, cls, kind


def test_working_period_ends_at_last_camera_lick_minus_stopped_runs():
    import trial_attempts as ta
    raw, cls, kind = _session()
    cls = pd.Series(cls).replace(ta.TA_NAMES).to_numpy()
    working = ~np.isin(cls, ta.RUN_EXCLUDED)                     # trial_attempts' `canonical` under the camera gate
    assert not working[40:47].any() and set(kind[40:47]) == {"run"}           # the mid-session stop run
    last_lick = int(np.flatnonzero(np.isin(raw, ["contact", "incomplete"]))[-1])
    assert working[: last_lick + 1].sum() == len(raw[: last_lick + 1]) - 7
    assert not working[last_lick + 1:].any() and set(kind[last_lick + 1:]) == {"tail"}
    # the isolated jaw_only / no_response stay IN the working period as non-hits (denominator, Priya 11-04 answer 2)
    assert working[57:60].all()

    n = len(raw)
    D = pd.DataFrame({"label": "PS92_0810", "pos": "far_contra", "cls": cls, "cls_raw": raw, "canonical": working,
                      "attempted": np.isin(cls, ta.ATTEMPT_CLASSES), "n_licks": 1, "n_nocontact": 0,
                      "contact": 300.0, "combined": 250.0, "has_cam1": True, "cam4": 250.0, "jaw_onset_ms": np.nan,
                      "angle_first": 0.0}, index=range(n))
    W = ta.session_summary(D)
    assert W.frac_hit_far_contra.iloc[0] == pytest.approx(55 / 58)     # contact / (contact+incomplete+jaw+no_resp)


def test_fig02_primary_is_the_camera_working_rate():
    import fig_behaviour_hits as fbh
    labs = ["PS92_0810", "PS94_0820"]
    W = pd.DataFrame({"label": labs, "frac_stopped_far_contra": [0.1, 0.2]})
    for i, p in enumerate(fbh.POS):
        W[f"frac_hit_{p}"] = [0.1 * i, 0.5]
    M = fbh.camera_metrics(W)
    assert list(M.date) == ["20260810", "20260820"] and list(M.animal) == ["PS92", "PS94"]
    for i, c in enumerate(fbh.POS.values()):
        assert list(M[f"hit__{c}"]) == [pytest.approx(0.1 * i), 0.5]
    with pytest.raises(SystemExit):                              # a DAQ-gated / untagged attempts table is refused
        fbh.camera_metrics(W.drop(columns=["frac_stopped_far_contra"]))


def test_fig02_writes_primary_and_labelled_robustness(monkeypatch, tmp_path):
    import fig_behaviour_hits as fbh
    calls = []

    def fake_compute(src, metrics=None):
        calls.append("camera" if metrics is not None else "daq")
        T = pd.DataFrame([dict(animal=a, epoch=e, spout=s, hit=0.5, n_sessions=1)
                          for a in fbh.AN for e in fbh.EP for s in fbh.POS])
        M = pd.DataFrame([dict(animal=a, epoch=e, **{f"hit__{c}": 0.5 for c in fbh.POS.values()})
                          for a in fbh.AN for e in fbh.EP])
        return {"T": T, "M": M, "src": str(src)}

    monkeypatch.setattr(fbh, "compute", fake_compute)
    monkeypatch.setattr(fbh, "camera_metrics", lambda src: pd.DataFrame())
    assert fbh.main(["--out", str(tmp_path), "--attempts", str(tmp_path / "a.csv"),
                     "--metrics", str(tmp_path / "m.csv")]) == 0
    assert calls == ["camera", "daq"]
    for n in (fbh.NAME, fbh.NAME_DAQ):
        assert (tmp_path / f"{n}.png").exists() and (tmp_path / f"{n}.csv").exists()
    assert fbh.NAME_DAQ != fbh.NAME and "ROBUSTNESS" in fbh.TITLE[fbh.NAME_DAQ] and fbh.TITLE[fbh.NAME] is None
    assert "camera working period" in fbh.YLABEL[fbh.NAME]
    assert sys.modules["fig_behaviour_hits"].CAMERA_COL == "frac_hit_{pos}"
