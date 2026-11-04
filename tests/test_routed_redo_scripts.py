"""Coordinator 11-04: the previously REFUSING REDO scripts are routed -- no require_default, outputs under the run tag
(run_mode / lp_licks / frozen_encoder routing helpers), behaviour inputs from the untagged behaviour qc dir."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: family 1 (the encoder family + its attempt / epoch readers)
FAMILY_2 = ["enl_fr_removed_decode", "fr_direction_mechanism", "fr_encoding_summary", "fr_musall"]
FAMILY_3 = ["layer2_matched_n", "layer2_random_splits", "summary_figures_1021", "within_lick_correction"]
FAMILY_4 = ["learned_compensation", "misdirected_lick_counts", "enl_lick_audit", "face_prelaunch_aim", "lick_interhemispheric", "lick_template_test",
            "misdirected_lick_decode", "position_coding_trend_recovery", "prelaunch_aim_tracking",
            "prelick_hemisphere_tuning", "qc_running_speed", "slowband_caveat_figure"]
FAMILY_1 = ["best_match_destination", "combined_encoding_figures", "compare_map_tests", "coupling_by_class",
            "encoder_trialavg_r2", "encoding_key_figures", "encoding_mixed", "exec3d_pilot_figure",
            "exec_angle_spread_check", "exec_vs_neural_match", "frozen_area_maps", "frozen_group_leakage",
            "frozen_musall_figures", "frozen_pixel_maps", "frozen_r2_maps", "frozen_residual_significance",
            "frozen_residual_trend", "frozen_running_compare", "frozen_spout_maps", "imaging_baseline_compare",
            "imaging_per_animal_tables", "lp_encoder_decompose", "relabel_confusion", "softmax_neural_vs_lick",
            "spout_mid_maps", "spout_reliability", "trial_neural_vs_exec", "within_spout_deviation",
            "within_spout_mixed"]
ROUTED = re.compile(r"encoding_out_root\(|out_root\(\)|output_dir\(|routed_output\(|run_mode\.subdir\(")
UNROUTED_ROOT = re.compile(r"oo\.encoding_root\((oc\.RUN|a\.run|run), PathResolver\(\)\)|_encoding_root\(\)\s*$")


@pytest.mark.parametrize("name", FAMILY_1 + FAMILY_2 + FAMILY_3 + FAMILY_4)
def test_family_1_is_routed(name):
    src = (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assert "require_default(" not in src, name
    assert ROUTED.search(src), name
    assert "widefield_pipeline" not in src, f"{name} pins the MAIN checkout on sys.path"


def test_encoding_out_root_is_the_encoders_tagged_root(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)
    from wfield_local import frozen_encoder as fe
    from wfield_local import lp_licks, run_mode
    base = run_mode.behaviour_qc_dir().parent
    assert run_mode.encoding_out_root() == base == fe.out_root()                  # the default run: untagged
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    t = lp_licks.output_tag()
    assert run_mode.encoding_out_root() == base / t == fe.out_root()             # tagged: <imaging_encoding>/<tag>
    assert run_mode.behaviour_qc_dir() == base / "qc"                            # behaviour inputs stay untagged


def test_trial_attempts_is_routed_and_the_deficit_readers_follow():
    src = (REPO / "scripts" / "trial_attempts.py").read_text(encoding="utf-8")
    assert "require_default(" not in src and "out = run_mode.attempts_dir()" in src
    for name in ("encoding_mixed", "within_spout_mixed"):
        assert "run_mode.deficit_qc_dir()" in (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8"), name


def test_every_attempts_reader_reads_this_runs_table(monkeypatch):
    """Coordinator 11-04: under the profile the attempts table is the routed one (camera classes), keyed by the
    selection only, so every arm / block reads the same table; never the pre-rebuild table."""
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)
    from wfield_local import lp_licks, run_mode
    from wfield_local import trial_classes as tc
    base = run_mode.behaviour_qc_dir()
    assert lp_licks.attempts_path() == base / "trial_attempts.csv"                 # the default run: as before
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    sel = tc.run_selection_tag()
    p = lp_licks.attempts_path()
    assert p == base / sel / "trial_attempts.csv" and run_mode.deficit_qc_dir() == p.parent
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")                               # an LP block: the same table
    assert lp_licks.attempts_path() == p
    for name in ("orofacial_cohort", "combined_encoding"):
        assert "lp_licks.attempts_path(" in (REPO / "wfield_local" / f"{name}.py").read_text(encoding="utf-8"), name
    seeds = (REPO / "configs" / "paths.yaml").read_text(encoding="utf-8")
    assert "- lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/trial_attempts.csv" not in seeds


ATTEMPT_READERS = ["cohort2_qc/attempt_effort", "cohort2_qc/quiet_lick_lfo", "cohort2_qc/quiet_task_events",
                   "interhemi_crossarea", "matched_residual", "recovery_timing", "relabel_fc_by_class",
                   "state_spout_check", "stopped_trial_traces", "summary_figures_1021"]


@pytest.mark.parametrize("name", ATTEMPT_READERS)
def test_redo_attempt_readers_use_this_runs_table(name):
    src = (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assert "_lp_licks.attempts_path()" in src and not re.search(r"/ \"qc\" / \"trial_attempts\.csv\"|/trial_attempts\.csv\"", src)
