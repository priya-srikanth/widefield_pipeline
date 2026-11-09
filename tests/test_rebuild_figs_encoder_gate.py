"""rebuild_figs encoder captions follow the gate the encoder PRODUCERS recorded (Priya 11-09, item C).

The queue fitted every encoder under OROFACIAL_GATE=attempted; the render profile does not set it, so a plain replot used
to caption the code default "reference". `_common.encoder_gate` now reads the gate from the producer outputs (combined
encoder session JSONs, frozen_encoder_gate-<g> run folders) and refuses a conflicting env.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def common(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_", "OROFACIAL_GATE")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "rebuild_figs"))
    import _common
    from wfield_local import run_mode
    monkeypatch.setattr(run_mode, "encoding_out_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(_common, "_RECORDED", {})
    return _common, tmp_path


def _producers(E: Path, combined="attempted (trial_attempts.csv)", frozen=("gate-attempted_v7g3_t3d_spont",)):
    s = E / "movement_first_subset-nokin_exec-polar_licks-combined_bins" / "sessions"
    s.mkdir(parents=True)
    (s / "PS92_20260801.json").write_text(json.dumps({"gate": combined}), encoding="utf-8")
    for t in frozen:
        (E / f"frozen_encoder_{t}").mkdir()


def test_recorded_gate_wins_over_the_code_default(common):
    c, E = common
    _producers(E)
    assert c.encoder_gate() == "attempted"
    assert "OROFACIAL_GATE=attempted, as recorded by the encoder outputs" in c.encoder_trials()


def test_matching_env_ok_conflicting_env_refused(common, monkeypatch):
    c, E = common
    _producers(E)
    monkeypatch.setenv("OROFACIAL_GATE", "attempted")
    assert c.encoder_gate() == "attempted"
    monkeypatch.setenv("OROFACIAL_GATE", "reference")
    with pytest.raises(SystemExit, match="fitted under gate 'attempted'"):
        c.encoder_gate()


def test_producers_disagreeing_is_an_error(common):
    c, E = common
    _producers(E, frozen=("gate-attempted_v7g3_t3d", "gate-canonical_attempted_v7g3_t3d"))
    with pytest.raises(SystemExit, match="different gates"):
        c.encoder_gate()


def test_no_producer_outputs_falls_back_to_env(common, monkeypatch):
    c, _E = common
    assert c.encoder_gate() == "reference"
    monkeypatch.setenv("OROFACIAL_GATE", "attempted")
    assert c.encoder_gate() == "attempted" and "this process's env" in c.encoder_trials()


def test_gate_tokens():
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "rebuild_figs"))
    try:
        import _common as c
    finally:
        sys.path.remove(str(REPO / "scripts" / "rebuild_figs"))
    assert c._gate_token("attempted (trial_attempts.csv)") == "attempted"
    assert c._gate_token("reference") == "reference"
    assert c._gate_token("frozen_encoder_gate-canonical_attempted_v7") == "canonical_attempted"
    assert c._gate_token("none (no trial_attempts rows for PS92_0801)") is None
