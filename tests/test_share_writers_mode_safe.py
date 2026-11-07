"""Every script that writes under a share / figures_working root is mode-safe (coordinator 11-03, before the render).

The rebuild renders in a NON-DEFAULT run mode (hemo variant + camera gate), so no writer may put mode-dependent output
into a default folder. Each such script either routes (run_mode / lp_licks.output_dir / routed_output / a routed root
such as il.OUT or attempt_coding.out_dir) or REFUSES (`run_mode.require_default`). Two exemptions, listed here so a
new writer fails until someone classifies it:
  BEHAVIOR  writes behavior / pose / FaceRhythm / camera products that do not depend on any imaging run mode
             (no imaging module, no SVTcorr / LocaNMF / encoder input): the same file in every mode.
  INDIRECT   writes only through an already-routed root (il.OUT, attempt_coding.out_dir(), ...).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: `save_png_svg(` is `fig.savefig(` routed through `figure_layout` (PNG + SVG twin, 2026-11-06): still a writer.
W = re.compile(r"\.(to_csv|savefig|write_text|write_bytes|to_pickle|to_parquet)\(|np\.save(z|z_compressed)?\(|save_fd\(|pd\.to_pickle\(|"
               r"save_png_svg\(")
SHARE = re.compile(r"M:/MICROSCOPE|oo\.root\(|PathResolver\(\)\.root\(|_run_dir\(\)|encoding_root\(|_enc_dir\(|"
                   r"root\(\"(cohort_out|orofacial|dlc|figures_working|behavior_out|labcams)")
ROUTED = re.compile(r"run_mode|output_dir\(|routed_output|lp_licks\.output_tag|require_default|fe\.out_root\(\)")  # + encoder root
BEHAVIOUR = {
    "apparatus_mouth.py", "cam1_clean_qc.py", "cam1_incomplete_licks.py", "cam1_lick_aim.py", "cam1_licks.py",
    "cam4_timing_check.py", "face_rhythm_aligned_maps.py", "face_rhythm_asym_aligned.py", "face_rhythm_band_maps.py",
    "face_rhythm_bout_split.py", "face_rhythm_evr_epochs.py", "face_rhythm_factor_clips.py",
    "face_rhythm_lick_epochs.py", "face_rhythm_match.py", "face_rhythm_nontask_state.py", "face_rhythm_param_qc.py",
    "face_rhythm_position_maps.py", "face_rhythm_prep.py", "face_rhythm_readouts.py", "face_rhythm_validate.py",
    "lick_floor_qc.py", "lick_path_softmax.py",
    "lmm_curve_summaries.py",
    # kinematics_pooled_figs / orofacial_cover_sensitivity / orofacial_robustness_table / pose3d_jaw_crosscheck: no
    # longer here -- they read / write the cam4 cohort through orofacial_cohort.cohort_dir (routed, gate-named; A27 11-06)
    "pose3d.py", "pose3d_crosscheck.py", "pose3d_cue_traces.py",
    "pose3d_jaw_depth_model_check.py", "pose3d_kinematics.py", "pose3d_phase.py", "pose3d_spout_error.py",
    "pose3d_vs_cam4.py", "pose_cue_traces.py",   # pose_kinematics_demo: share reads moved to kinematics_chain 11-05
    "tongue_nocontact_qc.py", "trial_class_raw_share_fig.py",
    # main's side-camera / 4-camera 3-D scripts (pose only), classified at the 11-05 rebuild merge
    "pose3d_side_check.py", "pose3d_side_loo.py", "pose3d_side_video.py", "side_clean_qc.py", "side_label_offsets.py",
    # main's FaceRhythm spout-in-mesh control (video factors only, writes the FR readouts dir), 11-05 merge
    "fr_spout_mesh_control.py",
}
INDIRECT = {
    "coupling_vigour_sm.py", "decoder_retention_sessions.py", "lick_correction.py",
    "lick_firstlick_extract.py", "lick_vigor_perspout.py", "precue_class_matched.py", "recovery_timing.py",
    "restw_strict_check.py", "session_pace_table.py",
}


def test_every_share_writer_is_routed_refusing_or_classified():
    unrouted = set()
    for p in sorted((REPO / "scripts").glob("*.py")):
        t = p.read_text(encoding="utf-8", errors="replace")
        if W.search(t) and SHARE.search(t) and not ROUTED.search(t):
            unrouted.add(p.name)
    assert unrouted == BEHAVIOUR | INDIRECT, {"unclassified": sorted(unrouted - BEHAVIOUR - INDIRECT),
                                              "stale": sorted((BEHAVIOUR | INDIRECT) - unrouted)}


def test_require_default_refuses_in_a_tagged_mode(monkeypatch):
    import pytest

    from wfield_local import run_mode
    monkeypatch.delenv("WIDEFIELD_TRIAL_GATE", raising=False)
    monkeypatch.delenv("WIDEFIELD_HEMO_VARIANT", raising=False)
    run_mode.require_default("x")
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    with pytest.raises(RuntimeError):
        run_mode.require_default("x")


#: Scripts whose inputs AND outputs are paths given on the command line (main's A3 / LMM figure scripts): outside the
#: share-root scan above, so registered here. Each must check the run tag of the folder it reads (fifth review N5).
ARGPATH_READERS = ("a3_lmm_imaging_stars.py", "a3_lmm_preview_fig.py", "lmm_change_fig.py", "lmm_per_lick_fig.py")


def test_argument_path_readers_check_the_input_run_tag():
    for f in ARGPATH_READERS:
        assert "run_mode.require_input_tag(" in (REPO / "scripts" / f).read_text(encoding="utf-8"), f


def test_require_input_tag(monkeypatch, tmp_path):
    import pytest

    from wfield_local import run_mode
    for k in ("WIDEFIELD_TRIAL_GATE", "WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LICK_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    assert run_mode.require_input_tag(tmp_path / "grant_figures" / "epoch" / "data", "t") == ""
    with pytest.raises(SystemExit, match="run tag"):
        run_mode.require_input_tag(tmp_path / "LP_licks" / "epoch" / "data", "t")
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    assert run_mode.require_input_tag(tmp_path / "LP_licks" / "epoch" / "data", "t") == "LP_licks"
    with pytest.raises(SystemExit):
        run_mode.require_input_tag(tmp_path / "grant_figures" / "epoch" / "data", "t")


def test_a3_imaging_stars_no_fittable_cell_is_an_empty_result(tmp_path):
    import subprocess
    import sys
    import pandas as pd
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"epoch": ["pre", "acute"], "position": ["a", "a"], "animal": ["PS92", "PS92"],
                  "value": [1.0, 2.0]}).to_csv(data / "fig_sessions.csv", index=False)
    pd.DataFrame({"epoch": ["acute"], "position": ["a"], "mark": [""]}).to_csv(data / "fig.csv", index=False)
    out = tmp_path / "o.csv"
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_"))}
    env["PYTHONPATH"] = str(REPO)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "a3_lmm_imaging_stars.py"), str(data), str(out)],
                       capture_output=True, text=True, cwd=str(REPO), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "empty result" in r.stdout and pd.read_csv(out).empty
