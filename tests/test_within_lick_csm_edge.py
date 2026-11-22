"""CSM edge-peak fix (2026-11-21, Priya: "yes fix the measure"): `within_lick_correction._peaks` counts only INTERIOR
local maxima of the real speed trace. Until then the trace was zero-padded, so a track cut while speed was still rising
(or falling from a high value) counted its first / last frame as a speed peak: truncation of the track, not a
corrective submovement (71-89 % of the two-camera August-pre CSMs; qc_phaseA/csm_examples README Part 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def W(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import within_lick_correction as w
    yield w
    sys.modules.pop("within_lick_correction", None)


def test_monotonic_rising_trace_has_no_csm(W):
    x = np.linspace(10, 300, 15)                   # a run cut while the tongue still accelerates
    k = 7
    pre, post = W._peaks(x, k)
    assert pre == () and post == ()
    assert W.csm_count(pre, post, 10) == 0


def test_edge_maxima_never_count(W):
    # rises to the LAST frame and falls from the FIRST frame: both edges are maxima only under zero padding
    x = np.r_[200.0, 150, 100, 120, 260, 120, 60, 80, 140, 220]
    pre, post = W._peaks(x, 4)
    assert pre == () and post == ()                 # the only interior maximum (frame 4) is k itself: in neither phase
    x2 = np.r_[300.0, 100, 50, 40, 30, 20]          # padded: frame 0 was a "peak" of prominence 300
    assert W._peaks(x2, 5) == ((), ())


def test_interior_bump_is_counted(W):
    # protrusion: main speed peak at frame 3, then a second interior bump at frame 7 before the extension peak k = 10
    x = np.r_[20.0, 80, 160, 220, 150, 90, 120, 180, 110, 60, 30, 70, 150, 90, 40]
    k = 10
    pre, post = W._peaks(x, k)
    assert len(pre) == 2 and len(post) == 1
    assert W.csm_count(pre, post, 30) == 1           # one extra peak in the protrusion phase = one CSM
    assert W.csm_count(pre, post, 200) == 0          # the bump's prominence (90) is below this threshold
    assert min(pre) == pytest.approx(90.0)           # bump prominence on the real trace: 180 - max(90, 30)


def test_prominence_is_measured_on_the_real_trace(W):
    # padding let the base of an isolated interior peak fall to 0; on the real trace it is the higher flank minimum
    x = np.r_[100.0, 120, 200, 130, 110]
    pre, _ = W._peaks(x, 4)
    assert pre == (pytest.approx(90.0),)              # 200 - max(100, 110), not 200 - 0


def test_stale_pickles_are_refused_and_re_extracted(W, monkeypatch, tmp_path):
    from wfield_local import trial_classes as tc
    monkeypatch.setattr(W, "LICKS", tmp_path)
    monkeypatch.setattr(tc, "working_trial_ids", lambda lab: ({1: True}, True))
    pd.DataFrame({"sid": "PS92_20260810", "trial_id": [1], "pk_sp": [((60.0,), ())]}).to_pickle(
        tmp_path / "PS92_20260810.pkl")
    assert W._peaks_version(tmp_path / "PS92_20260810.pkl") == 1
    with pytest.raises(SystemExit, match="peak measure v1"):
        W.load_licks(["PS92_20260810"])
    assert len(W.load_licks(["PS92_20260810"], allow_stale=True)) == 1
    pd.DataFrame({"sid": "PS92_20260810", "trial_id": [1], "pk_sp": [((60.0,), ())],
                  "peaks_v": W.PEAKS_VERSION}).to_pickle(tmp_path / "PS92_20260810.pkl")
    assert len(W.load_licks(["PS92_20260810"])) == 1


def test_grid_reaches_below_the_prereg_floor(W):
    assert min(W.GRID) <= 10 and all(t in W.GRID for t in range(30, 201, 10))


def test_out_dir_env_wins(W, monkeypatch, tmp_path):
    monkeypatch.setenv(W.OUT_ENV, str(tmp_path / "x_csmfix"))
    assert W.out_root() == tmp_path / "x_csmfix"
