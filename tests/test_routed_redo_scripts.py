"""Coordinator 11-04: the previously REFUSING REDO scripts are routed -- no require_default, outputs under the run tag
(run_mode / lp_licks / frozen_encoder routing helpers), behaviour inputs from the untagged behaviour qc dir."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: family 1 (the encoder family + its attempt / epoch readers)
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


@pytest.mark.parametrize("name", FAMILY_1)
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
