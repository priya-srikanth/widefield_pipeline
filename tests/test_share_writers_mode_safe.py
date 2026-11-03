"""Every script that writes under a share / figures_working root is mode-safe (coordinator 11-03, before the render).

The rebuild renders in a NON-DEFAULT run mode (hemo variant + camera gate), so no writer may put mode-dependent output
into a default folder. Each such script either routes (run_mode / lp_licks.output_dir / routed_output / a routed root
such as il.OUT or attempt_coding.out_dir) or REFUSES (`run_mode.require_default`). Two exemptions, listed here so a
new writer fails until someone classifies it:
  BEHAVIOUR  writes behaviour / pose / FaceRhythm / camera products that do not depend on any imaging run mode
             (no imaging module, no SVTcorr / LocaNMF / encoder input): the same file in every mode.
  INDIRECT   writes only through an already-routed root (il.OUT, attempt_coding.out_dir(), ...).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
W = re.compile(r"\.(to_csv|savefig|write_text|write_bytes|to_pickle|to_parquet)\(|np\.save(z|z_compressed)?\(|save_fd\(|pd\.to_pickle\(")
SHARE = re.compile(r"M:/MICROSCOPE|oo\.root\(|PathResolver\(\)\.root\(|_run_dir\(\)|"
                   r"root\(\"(cohort_out|orofacial|dlc|figures_working|behavior_out|labcams)")
ROUTED = re.compile(r"run_mode|output_dir\(|routed_output|lp_licks\.output_tag|require_default")
BEHAVIOUR = {
    "apparatus_mouth.py", "cam1_clean_qc.py", "cam1_incomplete_licks.py", "cam1_lick_aim.py", "cam1_licks.py",
    "cam4_timing_check.py", "enl_video_spout_check.py", "face_rhythm_aligned_maps.py", "face_rhythm_asym_aligned.py",
    "face_rhythm_bout_split.py", "face_rhythm_evr_epochs.py", "face_rhythm_factor_clips.py",
    "face_rhythm_lick_epochs.py", "face_rhythm_match.py", "face_rhythm_nontask_state.py", "face_rhythm_param_qc.py",
    "face_rhythm_position_maps.py", "face_rhythm_prep.py", "face_rhythm_readouts.py", "face_rhythm_validate.py",
    "jaw_only_incomplete_figs.py", "kinematics_pooled_figs.py", "lick_floor_qc.py", "lick_path_softmax.py",
    "orofacial_cover_sensitivity.py", "orofacial_robustness_table.py", "pose3d.py", "pose3d_close_dir_sheet.py",
    "pose3d_crosscheck.py", "pose3d_cue_traces.py", "pose3d_figures.py", "pose3d_jaw_crosscheck.py",
    "pose3d_jaw_depth_model_check.py", "pose3d_kinematics.py", "pose3d_phase.py", "pose3d_spout_error.py",
    "pose3d_summary_fig.py", "pose3d_vs_cam4.py", "pose_cue_traces.py", "pose_kinematics_demo.py",
    "tongue_nocontact_qc.py", "trial_class_raw_share_fig.py",
}
INDIRECT = {
    "coupling_vigour_sm.py", "decoder_retention_sessions.py", "iti_first_lick.py", "lick_correction.py",
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
