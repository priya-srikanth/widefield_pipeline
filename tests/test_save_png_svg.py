"""Every page / deck figure is also saved as SVG (Priya, 2026-11-06) -- `figure_layout.save_png_svg`.

TWO HALVES. The helper itself (the PNG byte-for-byte what a plain `fig.savefig` writes, the SVG at
`svg/<name>.svg` with text kept as text, an SVG failure never costing the PNG), and a STATIC check that every
writer whose PNG is on a results page or a deck saves through it, so a new `fig.savefig(` in one of them fails.

WHERE THE WRITER LIST CAME FROM (2026-11-06): the png stems on the results pages and decks --
`docs/status/rebuild_20261103/MASTER_MANIFEST.csv` (page / PI-deck / analysis-deck rows),
`docs/status/reanalysis_20261105/REPLACEMENT_PASS_MAP_2026-11-05.csv`, the goal page and `deck_registry` --
mapped to the script that calls `savefig` for them. Writers that save through a SHARED helper (`grant_kit._save`,
`epoch_figures._save_png_svg`, `orofacial_cohort.save_png_svg`, `face_rhythm_figures.save`) are covered by that
helper, which is in SHARED. Deliberately NOT listed: the per-day `labcams/analysis_figures` and preprocessing-deck
writers (per-session nightly outputs, many separate save calls, no shared save point), and `histology_area` /
`recovery_trajectory` (each already writes its own SVG).
"""
from __future__ import annotations

import ast
import io
import pathlib
import re
import warnings

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from wfield_local import figure_layout as fl  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Writers whose `fig.savefig(` calls were routed through `save_png_svg` (2026-11-06 sweep).
PAGE_FIGURE_WRITERS = tuple("""
    scripts/arrival_to_cue_traces.py
    scripts/cam1_lick_aim.py
    scripts/cd_cross_animal_figure.py
    scripts/cd_geometry_figure.py
    scripts/cd_loo_figures.py
    scripts/cd_loo_pooled.py
    scripts/cd_migration.py
    scripts/chronic_drift_vs_reorg.py
    scripts/cohort2_qc/quiet_eta_fr.py
    scripts/cohort2_qc/quiet_eta_split.py
    scripts/cohort2_qc/quiet_f0_noise.py
    scripts/cohort2_qc/quiet_fluct.py
    scripts/cohort2_qc/quiet_spikes_v2.py
    scripts/cohort2_qc/quiet_spikes_v2_examples.py
    scripts/cohort2_qc/quiet_spikes_v2_fr.py
    scripts/cohort2_qc/quiet_state_response.py
    scripts/cohort2_qc/quiet_sync_4to8.py
    scripts/cohort2_qc/quiet_task_events_followup.py
    scripts/cohort2_qc/seizure_events_video.py
    scripts/contest_lines.py
    scripts/coupling_by_class.py
    scripts/coupling_vigour_sm.py
    scripts/cued_misdirected_licks.py
    scripts/decoder_retention_sessions.py
    scripts/encoding_key_figures.py
    scripts/engagement_by_epoch.py
    scripts/enl_decode_figure.py
    scripts/enl_lick_audit.py
    scripts/enl_movement_multisession.py
    scripts/epoch_kernels.py
    scripts/exec_vs_neural_match.py
    scripts/face_offset_partial.py
    scripts/face_prelaunch_aim.py
    scripts/fr_direction.py
    scripts/fr_direction_mechanism.py
    scripts/fr_encoding_summary.py
    scripts/fr_enl_decode.py
    scripts/fr_musall.py
    scripts/fr_pixel_significance.py
    scripts/fr_trial_peth.py
    scripts/frozen_claims.py
    scripts/frozen_fr_compare.py
    scripts/frozen_fr_musall.py
    scripts/frozen_kernel_maps.py
    scripts/frozen_peth_trial_maps.py
    scripts/frozen_pixel_musall.py
    scripts/frozen_pixel_unique_sig.py
    scripts/frozen_posinfo.py
    scripts/frozen_r2_maps.py
    scripts/frozen_spout_maps.py
    scripts/imaging_page/s_per_animal.py
    scripts/interhemi_crossarea.py
    scripts/iti_area_maps.py
    scripts/iti_basis_decoder.py
    scripts/iti_check_figures.py
    scripts/iti_decoded_decomposition.py
    scripts/iti_decoder_pattern_stats.py
    scripts/iti_decoder_patterns.py
    scripts/iti_first_lick.py
    scripts/iti_hemo_check.py
    scripts/iti_lick_order.py
    scripts/iti_lick_rate.py
    scripts/iti_licks.py
    scripts/iti_overcorrection.py
    scripts/iti_recovery_link.py
    scripts/iti_recovery_marker.py
    scripts/iti_recovery_measure.py
    scripts/iti_session_cluster_tests.py
    scripts/iti_session_pattern_tests.py
    scripts/iti_spout_animal_panels.py
    scripts/iti_spout_pattern_stats.py
    scripts/iti_spout_patterns.py
    scripts/iti_where_angle_reach.py
    scripts/iti_where_decoder.py
    scripts/iti_where_drift.py
    scripts/iti_where_mae.py
    scripts/iti_where_per_spout.py
    scripts/iti_where_scatter.py
    scripts/jaw_lateral.py
    scripts/jaw_only_incomplete_figs.py
    scripts/june_august_drift.py
    scripts/key_figure.py
    scripts/kinematics_pooled_figs.py
    scripts/lick_aim_deficit.py
    scripts/lick_area_importance.py
    scripts/lick_area_tuning.py
    scripts/lick_area_unique_maps.py
    scripts/lick_balance_angle_lmm.py
    scripts/lick_balance_blind_decoder.py
    scripts/lick_balance_contact.py
    scripts/lick_balance_explainer.py
    scripts/lick_balance_imaging_control.py
    scripts/lick_balance_imaging_control_primary.py
    scripts/lick_balance_maps.py
    scripts/lick_balance_matched_error.py
    scripts/lick_balance_nested.py
    scripts/lick_balance_offset_lmm.py
    scripts/lick_balance_offset_primary.py
    scripts/lick_balance_order.py
    scripts/lick_balance_order_primary.py
    scripts/lick_balance_recovery.py
    scripts/lick_balance_reference.py
    scripts/lick_balance_regions.py
    scripts/lick_balance_subsets.py
    scripts/lick_balance_target_model.py
    scripts/lick_balance_trial.py
    scripts/lick_behaviour_decoder.py
    scripts/lick_block_compare.py
    scripts/lick_ci_spout_epoch.py
    scripts/lick_code_noisematch.py
    scripts/lick_code_rotation_summary.py
    scripts/lick_code_subspace.py
    scripts/lick_correction.py
    scripts/lick_covariate_reference.py
    scripts/lick_cue_to_lick.py
    scripts/lick_cue_to_lick_precue.py
    scripts/lick_cued_aim_by_spout.py
    scripts/lick_cued_ramp.py
    scripts/lick_enl_angle_decode.py
    scripts/lick_enl_offset_explore.py
    scripts/lick_enl_target_test.py
    scripts/lick_enlv_balance_offset.py
    scripts/lick_enlv_curves.py
    scripts/lick_enlv_target_code.py
    scripts/lick_interhemispheric.py
    scripts/lick_iti_dock_compare.py
    scripts/lick_iti_filter_compare.py
    scripts/lick_iti_prelick.py
    scripts/lick_iti_ramp_animals.py
    scripts/lick_iti_recovery_search.py
    scripts/lick_lateral_ratio.py
    scripts/lick_matched_angle_reach.py
    scripts/lick_matched_pixel_maps.py
    scripts/lick_matched_readout.py
    scripts/lick_nonlinear_decoder.py
    scripts/lick_offset_contrib_map.py
    scripts/lick_offset_terms_recovery.py
    scripts/lick_offset_where.py
    scripts/lick_position_per_spout.py
    scripts/lick_precue_restw.py
    scripts/lick_precue_restw_traces.py
    scripts/lick_recovery_markers.py
    scripts/lick_recovery_primary.py
    scripts/lick_refit_decoders.py
    scripts/lick_rim_overlay.py
    scripts/lick_roi_candidates.py
    scripts/lick_roi_check.py
    scripts/lick_roi_foci.py
    scripts/lick_same_target_place.py
    scripts/lick_session_level_maps.py
    scripts/lick_session_levels_scale_compare.py
    scripts/lick_session_pixel_lmm.py
    scripts/lick_vigor_perspout.py
    scripts/matched_residual.py
    scripts/movement_null_plan_maps.py
    scripts/partial_refit.py
    scripts/pose3d_cue_traces.py
    scripts/pose3d_phase.py
    scripts/pose3d_spout_error.py
    scripts/pose3d_summary_fig.py
    scripts/precue_class_matched.py
    scripts/rebuild_figs/fig_behaviour_hits.py
    scripts/rebuild_figs/fig_decoding_sessions.py
    scripts/rebuild_figs/fig_farcontra_attempts_sessions.py
    scripts/rebuild_figs/fig_frozen_refit_pooled.py
    scripts/rebuild_figs/fig_goal_epochs.py
    scripts/rebuild_figs/fig_goal_summary.py
    scripts/rebuild_figs/fig_layer2.py
    scripts/rebuild_figs/fig_pooled.py
    scripts/rebuild_figs/fig_pose_validation_sessions.py
    scripts/rebuild_figs/fig_resid16.py
    scripts/rebuild_figs/fig_resid_per_animal.py
    scripts/rebuild_figs/fig_resid_spout.py
    scripts/rebuild_figs/fig_rsplit.py
    scripts/rebuild_figs/fig_softmax.py
    scripts/rebuild_figs/fig_softmax_stats.py
    scripts/rebuild_figs/fig_state_position_sessions.py
    scripts/rebuild_figs/fig_state_ps94_diagnostic.py
    scripts/recovered_20261103/fig_goal_epochs.py
    scripts/recovered_20261103/fig_goal_summary.py
    scripts/recovered_20261103/per_animal_fig.py
    scripts/recovery_timing.py
    scripts/relabel_confusion.py
    scripts/relabel_fc_by_class.py
    scripts/remap_checks.py
    scripts/residual_roi.py
    scripts/rest_migration/channel_evoked_sign.py
    scripts/rest_migration/channel_position_maps.py
    scripts/rest_migration/channel_vessel_sign.py
    scripts/rest_migration/lick_bout_structure.py
    scripts/rest_migration/nvc_evoked.py
    scripts/rest_migration/quit_prodrome.py
    scripts/rest_migration/reference_family_figure.py
    scripts/rest_migration/rest_baseline_epoch_drift.py
    scripts/rest_migration/rest_frozen_decoder.py
    scripts/rest_migration/rest_frozen_decoder_nl.py
    scripts/rest_migration/rotation_maps.py
    scripts/rest_migration/shared_position_projection.py
    scripts/rest_migration/transfer_arms.py
    scripts/restw_strict_check.py
    scripts/revup_behaviour.py
    scripts/revup_followup.py
    scripts/revup_neural.py
    scripts/session_vs_frozen.py
    scripts/simple_key_figures.py
    scripts/slowband_caveat_figure.py
    scripts/specificity_control.py
    scripts/spout_unique_maps.py
    scripts/stopped_trial_traces.py
    scripts/te_readout.py
    scripts/trial_class_raw_share_fig.py
    scripts/trial_neural_vs_exec.py
    scripts/variance_maps.py
    scripts/within_lick_correction.py
    wfield_local/attempt_maps.py
    wfield_local/cd_trajectories.py
    wfield_local/face_rhythm_figures.py
    wfield_local/fixed_scale_maps.py
    wfield_local/locanmf_decoder_weights.py
    wfield_local/locanmf_frozen_decoder.py
    wfield_local/mixed_models.py
    wfield_local/plot_poststroke.py
    wfield_local/position_coding_directions.py
    wfield_local/spatial_reorganisation.py
    wfield_local/spout_behavior.py
""".split())

#: The shared save points page figures go through; each must call `save_png_svg` itself.
SHARED = ("wfield_local/grant_kit.py", "wfield_local/epoch_figures.py", "wfield_local/orofacial_cohort.py",
          "wfield_local/face_rhythm_figures.py")


def _figure():
    """A deterministic figure with text, lines, an image and a legend -- what page figures are made of."""
    import matplotlib.pyplot as plt
    import numpy as np
    fig, (a, b) = plt.subplots(1, 2, figsize=(4, 2))
    a.plot([0, 1, 2], [0, 1, 0.5], label="probe line")
    a.set_title("SVG probe title")
    a.legend()
    b.imshow(np.arange(16.0).reshape(4, 4))
    fig.suptitle("suptitle text")
    return fig


@pytest.mark.parametrize("kw", [dict(dpi=110), dict(dpi=150, bbox_inches="tight"), dict(dpi=200, facecolor="white"),
                                dict(dpi=120, bbox_inches="tight", pad_inches=0.05, transparent=True), dict()])
def test_png_is_byte_for_byte_what_plain_savefig_writes(tmp_path, kw):
    import matplotlib.pyplot as plt
    f1 = _figure()
    f1.savefig(tmp_path / "plain.png", **kw)
    plt.close(f1)
    f2 = _figure()
    (tmp_path / "routed").mkdir()
    fl.save_png_svg(f2, tmp_path / "routed" / "plain.png", **kw)
    plt.close(f2)
    assert (tmp_path / "plain.png").read_bytes() == (tmp_path / "routed" / "plain.png").read_bytes()
    assert (tmp_path / "routed" / "svg" / "plain.svg").is_file()


def test_svg_twin_lands_in_svg_dir_with_text_kept_as_text(tmp_path):
    import matplotlib.pyplot as plt
    fig = _figure()
    q = tmp_path / "fig_x.png"
    fl.save_png_svg(fig, str(q), dpi=150, bbox_inches="tight")     # a str path works too
    plt.close(fig)
    svg = tmp_path / "svg" / "fig_x.svg"
    assert q.is_file() and svg == fl.svg_path(q, mkdir=False) and svg.is_file()
    assert not (tmp_path / "fig_x.svg").exists()                  # never flat beside the PNG
    s = svg.read_text(encoding="utf-8")
    assert "<text" in s and "SVG probe title" in s                 # svg.fonttype none: text, not glyph paths
    assert matplotlib.rcParams["svg.fonttype"] == "path"           # the rc change is scoped to the SVG call


def test_raster_only_kwargs_do_not_reach_the_svg_call(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    fig = _figure()
    calls = []
    orig = fig.savefig

    def spy(fname, *a, **k):
        calls.append((pathlib.Path(fname).suffix, dict(k), matplotlib.rcParams["svg.fonttype"]))
        return orig(fname, *a, **k)

    monkeypatch.setattr(fig, "savefig", spy)
    fl.save_png_svg(fig, tmp_path / "k.png", dpi=90, metadata={"Software": "x"}, bbox_inches="tight")
    plt.close(fig)
    assert calls[0] == (".png", dict(dpi=90, metadata={"Software": "x"}, bbox_inches="tight"), "path")
    assert calls[1] == (".svg", dict(bbox_inches="tight"), "none")


def test_svg_failure_never_costs_the_png(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    fig = _figure()
    orig = fig.savefig

    def boom(fname, *a, **k):
        if str(fname).endswith(".svg"):
            pathlib.Path(fname).write_text("partial", encoding="utf-8")
            raise ValueError("cannot vectorise")
        return orig(fname, *a, **k)

    monkeypatch.setattr(fig, "savefig", boom)
    with pytest.warns(RuntimeWarning, match="SVG twin not written"):
        fl.save_png_svg(fig, tmp_path / "f.png", dpi=100)
    plt.close(fig)
    assert (tmp_path / "f.png").stat().st_size > 0
    assert not (tmp_path / "svg" / "f.svg").exists()              # no broken twin left beside a good PNG


def test_unwritable_svg_dir_is_tolerated(tmp_path):
    import matplotlib.pyplot as plt
    (tmp_path / "svg").write_text("a FILE where the svg dir would go", encoding="utf-8")
    fig = _figure()
    with pytest.warns(RuntimeWarning):
        fl.save_png_svg(fig, tmp_path / "g.png", dpi=100)
    plt.close(fig)
    assert (tmp_path / "g.png").is_file()


def test_non_png_targets_get_no_twin(tmp_path):
    import matplotlib.pyplot as plt
    fig = _figure()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fl.save_png_svg(fig, tmp_path / "a.pdf")
        fl.save_png_svg(fig, tmp_path / "b.svg")
        buf = io.BytesIO()
        fl.save_png_svg(fig, buf, format="png")
    plt.close(fig)
    assert buf.getvalue()[:4] == b"\x89PNG"
    assert not (tmp_path / "svg").exists()


def test_shared_save_points_write_the_twin(tmp_path):
    import matplotlib.pyplot as plt

    from wfield_local import face_rhythm_figures, orofacial_cohort
    fig = _figure()
    orofacial_cohort.save_png_svg(fig, tmp_path / "o.png")
    face_rhythm_figures.save(fig, tmp_path / "r.png", methods="probe")   # closes fig
    plt.close("all")
    for n in ("o", "r"):
        assert (tmp_path / f"{n}.png").is_file()
        assert "<text" in (tmp_path / "svg" / f"{n}.svg").read_text(encoding="utf-8")


def _orofacial(fig, q):
    from wfield_local import orofacial_cohort
    orofacial_cohort.save_png_svg(fig, q)


def _face_rhythm(fig, q):
    from wfield_local import face_rhythm_figures
    face_rhythm_figures.save(fig, q, methods="probe")


def _epoch(fig, q):
    from wfield_local import epoch_figures
    epoch_figures._save_png_svg(fig, q)


@pytest.mark.parametrize("writer,before", [(_orofacial, dict(dpi=130)), (_face_rhythm, dict(dpi=110)),
                                           (_epoch, dict(dpi=200))])
def test_shared_writers_png_unchanged(tmp_path, writer, before, monkeypatch):
    """Each shared save point's PNG against the exact call it made before 2026-11-06."""
    import matplotlib.pyplot as plt
    for k in [k for k in __import__("os").environ if k.startswith(("WIDEFIELD_", "WFIELD_"))]:
        monkeypatch.delenv(k)
    f1 = _figure()
    f1.savefig(tmp_path / "before.png", **before)
    plt.close(f1)
    (tmp_path / "after").mkdir()
    writer(_figure(), tmp_path / "after" / "before.png")
    plt.close("all")
    assert (tmp_path / "before.png").read_bytes() == (tmp_path / "after" / "before.png").read_bytes()
    assert (tmp_path / "after" / "svg" / "before.svg").is_file()


def _savefig_calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "savefig"]


def _calls_helper(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == "save_png_svg") or \
               (isinstance(f, ast.Attribute) and f.attr == "save_png_svg"):
                return True
    return False


@pytest.mark.parametrize("rel", PAGE_FIGURE_WRITERS + SHARED)
def test_every_page_figure_writer_saves_through_the_helper(rel):
    p = ROOT / rel
    assert p.is_file(), f"{rel} is listed as a page-figure writer but does not exist (renamed? update the list)"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    direct = [n.lineno for n in _savefig_calls(tree)]
    assert not direct, (f"{rel}: direct savefig at line(s) {direct} -- a page figure saved this way gets no SVG; "
                        "use `figure_layout.save_png_svg(fig, png, **same_kwargs)`")
    assert _calls_helper(tree), f"{rel} no longer calls save_png_svg"


def test_the_writer_list_is_not_vacuous():
    assert len(PAGE_FIGURE_WRITERS) > 200 and len(set(PAGE_FIGURE_WRITERS)) == len(PAGE_FIGURE_WRITERS)


def test_the_helper_does_not_call_itself():
    """The trap `grant_kit._save` records twice: a sweep that rewrites the helper's own savefig recurses."""
    src = (ROOT / "wfield_local" / "figure_layout.py").read_text(encoding="utf-8")
    body = src.split("def save_png_svg(", 1)[1].split("\ndef ", 1)[0]
    assert "fig.savefig(fname" in body and not re.search(r"\bsave_png_svg\(fig", body)
