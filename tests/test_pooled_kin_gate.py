"""Priya 11-07 (fix/pooled-kin-gate-20261107): the pooled 3-D table gap. `kinematics_pooled_figs` (kinematics page Fig 7.5,
Fig 9.6, the 3-D rows of the section 3 / 5 / 7 tables) read the UNGATED 10-23 `pose3d/kinematics/sessions.csv`. It now
re-makes the per-session medians over CAMERA-WORKING-PERIOD licks / jaw trials (`sessions3d` ->
`pose3d_summary_fig.gated_sessions` -> `trial_classes.gate_kin_licks`, A20); ``WFIELD_KIN_GATE=legacy`` keeps the old
table. Synthetic inputs only, no share reads.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ("kinematics_pooled_figs", "scripts.kinematics_pooled_figs", "pose3d_summary_fig",
           "scripts.pose3d_summary_fig")

from wfield_local import trial_classes as tc  # noqa: E402

# cue order: 2 contact, incomplete, jaw_only, no_response, stopped run, after the last camera lick (trials 1..5 working)
CLS = np.array(["contact", "contact", "incomplete", "jaw_only", "no_response", "stopped", "stopped_licking",
                "after_end", "unknown"], dtype=object)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(str(REPO))
    yield
    for m in SCRIPTS:
        sys.modules.pop(m, None)


@pytest.fixture
def d3(tmp_path, monkeypatch):
    """PS92_0810 with camera classes (working trials 1..5); PS93_0813 with no video. The snapshot tables hold both
    working and non-working trials; sessions.csv = the ungated medians (the 10-23 table)."""
    full = np.arange(len(CLS)) * 10.0
    T = pd.DataFrame({"trial_id": np.arange(len(CLS)) + 1, "position": ["far_R"] * len(CLS), "cue_daq": full + 0.004,
                      "cls": ["raw"] * len(CLS)})
    monkeypatch.setattr(tc, "session_trials", lambda lab, *a, **k: T if lab.startswith("PS92") else None)
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda lab: full)
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs, w=None: CLS.copy() if lab.startswith("PS92")
                        else np.array(["unknown"] * len(cs), dtype=object))
    d = tmp_path / "pose3d"
    (d / "kinematics").mkdir(parents=True)
    L = pd.DataFrame({"session": ["PS92_20260810"] * 4 + ["PS93_20260813"] * 2, "animal": ["PS92"] * 4 + ["PS93"] * 2,
                      "date": ["20260810"] * 4 + ["20260813"] * 2, "epoch": "pre",
                      "position": ["far_L", "far_L", "far_R", "far_R", "far_L", "far_R"],
                      "trial_id": [1, 6, 2, 9, 1, 2], "miss_mm": [1.0, 100.0, 3.0, 100.0, 5.0, 5.0],
                      "midline_err_ipsi": [2.0, -50.0, 4.0, -50.0, 0.0, 0.0]})
    J = pd.DataFrame({"session": ["PS92_20260810"] * 4, "animal": "PS92", "date": "20260810", "epoch": "pre",
                      "position": ["far_L", "far_R", "far_L", "far_R"], "trial_id": [1, 2, 6, 9],
                      "lr_wide_mm": [1.0, -1.0, 50.0, 50.0], "open_max_mm": [2.0, 4.0, 99.0, 99.0]})
    L.to_csv(d / "kinematics" / "licks.csv.gz", index=False)
    J.to_csv(d / "kinematics" / "jaw_trials.csv.gz", index=False)
    pd.DataFrame([{"session": "PS92_20260810", "animal": "PS92", "date": "20260810", "epoch": "pre",
                   "miss_mm|far_L": 50.5, "jaw_far_dir_mm": 0.0},
                  {"session": "PS93_20260813", "animal": "PS93", "date": "20260813", "epoch": "pre",
                   "miss_mm|far_L": 5.0}]).to_csv(d / "kinematics" / "sessions.csv", index=False)
    import scripts.kinematics_pooled_figs as K
    monkeypatch.setattr(K, "pose3d_dir", lambda *a: d)
    K._S3D.clear()
    return d


def test_sessions3d_is_over_working_period_trials(d3):
    import scripts.kinematics_pooled_figs as K
    S = K.sessions3d()
    assert S.session.tolist() == ["PS92_20260810"]                       # PS93: no camera classes, contributes nothing
    r = S.iloc[0]
    assert r["miss_mm|far_L"] == 1.0 and r["miss_mm|far_R"] == 3.0        # trials 6 / 9 (after the period) out
    assert r["midline_err_ipsi|far_L"] == 2.0
    assert r["open_max_mm|far_L"] == 2.0 and r["jaw_far_dir_mm"] == pytest.approx(1.0)
    assert str(r["date"]) == "20260810"


def test_sessions3d_legacy_reads_the_old_table(d3, monkeypatch):
    import scripts.kinematics_pooled_figs as K
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    S = K.sessions3d()
    assert S.session.tolist() == ["PS92_20260810", "PS93_20260813"]
    assert S["miss_mm|far_L"].tolist() == [50.5, 5.0]


def test_pooled_figs_never_read_the_ungated_table_unfiltered():
    src = (REPO / "scripts" / "kinematics_pooled_figs.py").read_text(encoding="utf-8")
    hits = [m.start() for m in re.finditer(r'"kinematics"\s*/\s*"sessions\.csv"', src)]
    assert len(hits) == 1                                                  # only the legacy branch of sessions3d
    body = src[src.index("def sessions3d"):src.index("def fig_3d")]
    assert src.index("def sessions3d") < hits[0] < src.index("def fig_3d")
    assert body.index('if g == "working"') < body.index("gated_sessions(") < body.index('"sessions.csv"')
    for fn in ("def fig_3d", "def fig_days"):
        f = src[src.index(fn):]
        f = f[:f.index("\ndef ", 5)]
        assert "S = sessions3d()" in f and "read_csv" not in f.split("S = sessions3d()")[0], fn
