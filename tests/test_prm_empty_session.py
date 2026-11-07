"""An empty-trial session must not crash the 15r reference maps (2026-11-06).

`session_raw_maps` returns the retired 6-tuple ``({}, {}, None, None, {}, {})`` when a session has no trials in the
arm, and the session cache stores it as-is. `maps_by_epoch` indexed it by key, so render g5_epoch lost the whole 15r
family on the cue and ENL arms (``TypeError tuple indices must be integers or slices, not str``; PS93_0813 has zero
working trials under the camera gate). The tuple must now read as an empty session: skipped, everything else kept.
"""
import numpy as np
import pytest

from wfield_local import position_reference_maps as prm

EMPTY = ({}, {}, None, None, {}, {})


def test_as_parts_normalises_the_empty_tuple():
    p = prm._as_parts(EMPTY)
    assert p["raw"] == {} and p["raw_rest"] == {} and p["trial_mean"] is None
    d = {"raw": {"far_R": np.zeros(2)}}
    assert prm._as_parts(d) is d
    with pytest.raises(TypeError):
        prm._as_parts(({"far_R": np.zeros(2)}, {}))       # a NON-empty tuple is not silently dropped


@pytest.mark.parametrize("ref", prm.REFERENCES)
def test_reference_maps_accepts_the_empty_tuple(ref):
    assert prm.reference_maps(EMPTY, ref) == {}


def test_maps_by_epoch_skips_an_empty_tuple_session(monkeypatch):
    from wfield_local import beta_maps as bm
    from wfield_local import config, grant_figures, locanmf_cue_lick_analysis, map_collect

    m = np.ones((2, 2), np.float32)
    good = {"raw": {"far_R": m}, "used": {"far_R": 25}, "quiet_flat": None, "trial_mean": m * 0,
            "raw_precue": {}, "used_precue": {}, "raw_rest": {"far_R": m}, "used_rest": {"far_R": 25},
            "raw_restw": {}, "used_restw": {}, "restw_positions": []}
    sessions = [{"label": "AX_0801", "mc": ""}, {"label": "AX_0802", "mc": ""}]
    monkeypatch.setattr(grant_figures, "ANIMALS", ["AX"])
    monkeypatch.setattr(grant_figures, "_day", lambda an, d: -1)
    monkeypatch.setattr(locanmf_cue_lick_analysis, "SESSIONS", sessions)
    monkeypatch.setattr(config, "phase_labels", lambda ph: ["AX_0801", "AX_0802"] if ph == "pre" else [])
    monkeypatch.setattr(prm, "derived_active", lambda: ())
    monkeypatch.setattr(bm, "split_half_reliability", lambda d: float("nan"))
    monkeypatch.setattr(map_collect, "collect",
                        lambda jobs, worker, label=None: ({"AX_0801": good, "AX_0802": EMPTY}, []))
    prm.maps_by_epoch.cache_clear()
    try:
        out, _rel, n_out = prm.maps_by_epoch("cue", "working")
    finally:
        prm.maps_by_epoch.cache_clear()
    assert list(out["AX"]["pre"]["far_R"]) == ["AX_0801"]
    assert n_out["AX"]["pre"]["far_R"] == {"AX_0801": 25}
