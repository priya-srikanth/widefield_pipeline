"""Review, third pass (2026-11-03 night): T1 / T2 / T4 / T5 / T8 (T3 / T7 / T11 in test_priya_decisions_1103.py, T10 in
test_parallel_ordering_strict.py). Paths are resolved only; nothing is written to a share."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from wfield_local import run_mode
from wfield_local import trial_classes as TC

REPO = Path(__file__).resolve().parents[1]
MODES = ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_LP_FROZEN",
         "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_GATE_SOURCE", "WIDEFIELD_PRE_FROM", "WIDEFIELD_MAP_REFS",
         "WIDEFIELD_JOINT_PROJECT_ALL", "WIDEFIELD_LOCANMF_PROJECT", "WIDEFIELD_ITI_FROZEN", "WFIELD_ITI_OUT_W")
ENVS = [{}, {"WIDEFIELD_GATE_SOURCE": "lp"}, {"WIDEFIELD_TRIAL_GATE": "camera"}, {"WIDEFIELD_PRE_FROM": "0801"},
        {"WIDEFIELD_HEMO_VARIANT": "meegkit_hpfit_lag_refitT_swapmask"}, {"WIDEFIELD_JOINT_PROJECT_ALL": "1"}]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in MODES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(TC, "class_source_signature", lambda *a, **k: "sig")


@pytest.mark.parametrize("env", ENVS)
def test_t1_attempt_outputs_are_writable_in_every_lp_mode(monkeypatch, env):
    """T1: the attempt writers' own folders pass the writeguard under LP alone and every LP + mode combination."""
    from wfield_local import attempt_coding, attempt_maps, writeguard
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a, b = attempt_coding.out_dir(), attempt_maps.out_dir("restnl")
    pp = run_mode.routed_output(a.parent / "attempt_coding_pp", guard=False, lp_family=True, make=False)
    for p in (a, b, pp):
        assert writeguard.is_writable(p / "_"), (env, p)
    if env:                                               # a default-named write is still refused in a tagged mode
        assert not writeguard.is_writable(a.parent / "attempt_coding" / "trials_cue_PS92.csv.gz") or a.name == \
            "attempt_coding"


@pytest.mark.parametrize("env", ENVS + [{"WIDEFIELD_LICK_SOURCE": "lp"}, {"WIDEFIELD_ITI_FROZEN": "0"}])
def test_t2_iti_extracts_and_sd_tables(monkeypatch, env):
    """T2: the extract folder is writable in every mode (content-keyed store); the SD tables' writer path equals the
    reader path (`_sd_dir`); outputs stay in the fully tagged OUT."""
    from wfield_local import writeguard
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.path.insert(0, str(REPO / "scripts"))
    for m in [m for m in list(sys.modules) if m in ("iti_licks", "fr_musall")]:
        del sys.modules[m]
    import iti_licks as il
    assert writeguard.is_writable(il.XOUT / il.SESS_NAME / "_")
    assert writeguard.is_writable(il.sd_table_path(il.SD_NAME))
    assert il._sd_dir(None) == il.sd_table_path(il.SD_NAME).parent
    src = (REPO / "scripts" / "iti_zscore_check.py").read_text(encoding="utf-8")
    assert "il.sd_table_path(il.SD_NAME)" in src
    assert "il.sd_table_path(il.SPLIT_NAME)" in (REPO / "scripts" / "iti_area_split_sd.py").read_text(encoding="utf-8")
    assert il.OUT == run_mode.subdir(il.E / "iti_licks")
    for m in [m for m in list(sys.modules) if m in ("iti_licks", "fr_musall")]:
        del sys.modules[m]


def test_t4_restnl_missing_baseline_is_built(monkeypatch, tmp_path):
    from wfield_local import restnl_mask
    calls = []
    state = {"built": False}
    monkeypatch.setattr(restnl_mask, "cache_path", lambda label, s=None: tmp_path / f"{label}.npz")
    monkeypatch.setattr(restnl_mask, "baseline", lambda label: {"b_nl": 1} if state["built"] else None)

    def build(label, measure=False):
        calls.append(label)
        state["built"] = True
        return {}
    monkeypatch.setattr(restnl_mask, "build_session", build)
    assert restnl_mask.baseline_or_build("PS92_0810") == {"b_nl": 1} and calls == ["PS92_0810"]
    src = (REPO / "wfield_local" / "attempt_maps.py").read_text(encoding="utf-8")
    assert "if not maps:" in src and "baseline_or_build" in src
    assert "RESTnl reference requested but no session" in (REPO / "wfield_local" /
                                                          "position_reference_maps.py").read_text(encoding="utf-8")


def test_t5_tagged_deck_is_never_the_default_deck(monkeypatch):
    from wfield_local import config, writeguard
    from wfield_local.paths import deck_dir
    fw = deck_dir("analysis", config.resolver())                 # decks_out since 2026-11-06 (was cohort_out)
    default = fw / "spout_position_analysis_summary.pptx"
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    routed = run_mode.routed_output(fw, guard=False, make=False) / default.name
    assert routed != default and writeguard.is_writable(routed) and not writeguard.is_writable(default)


def test_t8_rest_period_ends_at_the_last_lick_with_an_uncovered_tail(monkeypatch):
    cls = np.array(["contact", "incomplete", "stopped", "stopped", "unknown", "unknown"], object)
    monkeypatch.setattr(TC, "cue_classes_for", lambda label, cue_s, window=None: cls)
    assert TC.camera_engaged_period("X_0000", np.arange(6.0)).tolist() == [True, True, False, False, False, False]
