"""Review 7A (working rows, captions, PS93_0813 visibility, NB4 / NB6) and 7B (PRE_FROM blocks, primary-tag inputs,
June selection, deck tagged paths) at 6a85304d."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from wfield_local import run_mode
from wfield_local import spout_scaling as ss

REPO = Path(__file__).resolve().parents[1]      # never the cwd (the suite also runs from tests/)

PROFILE_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_TRIAL_WINDOW": "cue_stop", "WIDEFIELD_TRIAL_END": "lick",
               "WIDEFIELD_TRIAL_STOPPED": "run", "WIDEFIELD_PRE_FROM": "0801"}


@pytest.fixture
def camera(monkeypatch):
    for k, v in PROFILE_ENV.items():
        monkeypatch.setenv(k, v)
    import os
    for k in list(os.environ):              # review 8A NB-e: every owned variable, so the tags do not depend on the shell
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")) and k not in PROFILE_ENV:
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ------------------------------------------------------------------------------------------------------ 7A
def test_5b_never_reads_the_quit_tail_json_under_a_camera_selection():
    from wfield_local import grant_confusion as gc
    src = inspect.getsource(gc.fig_confusion_pre_post_working)
    assert "if _wt.is_camera():\n            return None" in src.replace("\r\n", "\n")       # main's guard, verbatim
    assert '_frozen_frame(bd, an, align, sel="working")' in src
    assert "WORKING trials ({_wc})" in src and "c+i+jaw" in src


def test_class_note_describes_the_active_selection(camera):
    from wfield_local.grant_kit import _class_note
    t = _class_note("working")
    assert "contact + incomplete" in t and "jaw_only excluded" in t and "last camera lick" in t
    assert "PRE-STROKE REFERENCE IS LICK TRIALS" in _class_note("working", pattern=True)
    camera.setenv("WIDEFIELD_WORKING_TRIALS", "in_working_jaw")
    assert "+ jaw_only" in _class_note("working")
    camera.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    assert _class_note("working") == "LICK + miss-while-working (quit period removed)"


def test_pattern_families_caption_their_lick_pre_reference():
    for mod in ("grant_encoder", "grant_geometry", "grant_matching", "grant_similarity"):
        src = (REPO / f"wfield_local/{mod}.py").read_text(encoding="utf-8")
        assert "_class_note(v)" not in src and "_class_note(v, pattern=True)" in src, mod


def test_no_video_session_is_logged_and_a_post_session_without_rows_raises(camera, capsys):
    from wfield_local import locanmf_frozen_decoder as lfd
    ss.WORKING_EXCLUDED.discard("PS93_0813")
    ss.note_working_excluded("PS93_0813", "test")
    assert "PS93_0813" in ss.WORKING_EXCLUDED and "EXCLUDED from the working selection" in capsys.readouterr().out
    rng = np.random.default_rng(0)
    GE = np.repeat(np.arange(3), 30)
    YE = np.tile(np.arange(6), 15)
    XE = rng.normal(size=(90, 3))
    work = {"XE": XE, "YE": YE, "GE": GE, "XU": np.zeros((0, 3)), "YU": np.zeros(0), "GU": np.zeros(0, int),
            "work_e": GE != 2, "work_u": np.zeros(0, bool)}

    class M:
        classes_ = np.arange(6)

        def predict_proba(self, X):
            return np.full((len(X), 6), 1 / 6)
    kept = ["PS92_0806", "PS92_0807", "PS92_0820"]                     # pre, pre, POST (by phase)
    models = {"full": M(), "loso": {kept[0]: M()}, "trials": "sel", "pre_labels": [kept[0]],
              "pi_train": {"full": np.full(6, 1 / 6), "loso": {kept[0]: np.full(6, 1 / 6)}}}
    with pytest.raises(ss.BalEmCentreRefused, match="post session PS92_0820"):
        lfd.frozen_scored_frame(models, XE, GE, np.zeros((0, 3)), np.zeros(0, int), kept, [0], work=work)
    # review 8A B1: a PRE session with 0 working rows is excluded and logged, also when the caller's pre list was
    # narrowed to the sessions with a LOSO model (pooled_frozen_loso)
    work["work_e"] = GE != 1
    ss.WORKING_EXCLUDED.discard(kept[1])
    lfd.frozen_scored_frame(models, XE, GE, np.zeros((0, 3)), np.zeros(0, int), kept, [0], work=work)
    assert kept[1] in ss.WORKING_EXCLUDED and "EXCLUDED" in capsys.readouterr().out


def test_pooled_frozen_loso_survives_a_pre_session_without_working_rows(camera, monkeypatch, tmp_path, capsys):
    """Review 8A B1: the pooled_frozen_loso sequence (section D / joint_xsession / nightly ROI decoder) with a no-video
    PRE session (PS93_0813: 0 working rows) must not raise; it logs EXCLUDED and leaves that session out of the band."""
    from wfield_local import locanmf_frozen_decoder as lfd
    monkeypatch.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path / "fm"))
    kept = ["PS93_0806", "PS93_0807", "PS93_0813", "PS93_0820"]
    rng = np.random.default_rng(1)
    GE = np.repeat(np.arange(4), 60)
    YE = np.tile(np.arange(6), 40)
    XE = rng.normal(size=(len(GE), 4)) + YE[:, None]
    GU = np.repeat(np.arange(4), 12)
    YU = np.tile(np.arange(6), 8)
    XU = rng.normal(size=(len(GU), 4)) + YU[:, None]
    BE = [np.repeat(np.arange(10), 6) for _ in kept]
    monkeypatch.setattr(lfd, "pool_sessions",
                        lambda *a, **k: (XE, YE, GE, BE, XU, YU, kept, np.arange(4), GU))
    monkeypatch.setattr(lfd, "pool_work", lambda XE_, YE_, GE_, XU_, YU_, GU_, kept_, f, al: {
        "XE": XE_, "YE": YE_, "GE": GE_, "XU": XU_, "YU": YU_, "GU": GU_,
        "work_e": GE_ != 2, "work_u": GU_ != 2})
    ss.WORKING_EXCLUDED.discard("PS93_0813")
    res = lfd.pooled_frozen_loso(kept, source="roi", align="cue", verbose=False)
    assert "PS93_0813" in ss.WORKING_EXCLUDED and "PS93_0813" in capsys.readouterr().out
    assert "PS93_0813" not in res["per_session"] and "PS93_0820" in res["per_session"]


def test_collect_7_reraises_under_camera_and_is_keyed_by_the_selection():
    from wfield_local import grant_kit as gk
    src = inspect.getsource(gk._collect_7_cached)
    assert "reraise_under_camera" in src
    assert "_collect_7_cached(align, variant, min_trials, field, run)" in inspect.getsource(gk._collect_7)


def test_store_log_line_names_the_lick_source_and_selection():
    from wfield_local import grant_confusion as gc
    src = inspect.getsource(gc._stored_epoch_decoder)
    assert "lick source" in src and "selection {_sel}" in src and "trained on {spec.get('lick_source'" not in src


def test_working_trials_stays_byte_identical_to_main():
    import subprocess
    try:
        blob = subprocess.run(["git", "rev-parse", "origin/main:wfield_local/working_trials.py"], capture_output=True,
                              text=True, check=True, cwd=REPO).stdout.strip()
        mine = subprocess.run(["git", "hash-object", "wfield_local/working_trials.py"], capture_output=True, text=True,
                              check=True, cwd=REPO).stdout.strip()
    except Exception as ex:                                              # noqa: BLE001 - no git / no origin here
        pytest.skip(f"git unavailable: {ex}")
    assert blob == mine


# ------------------------------------------------------------------------------------------------------ 7B
def test_profile_blocks_unset_pre_from():
    from wfield_local import render_profile as rp
    for b in ("june_august_drift", "rest_frozen_decoder_nl", "map_drift_prepre", "restnl_mask"):
        assert rp.block_env("cohort1_rebuild_20261103", b) == {"WIDEFIELD_PRE_FROM": None}, b


def test_june_dir_and_primary_dir(camera, tmp_path):
    from wfield_local import lp_licks
    lab = tmp_path / "cohort_out"                     # the DERIVED root (review 8B B1), never the labcams input tree
    primary = lp_licks.output_tag()
    assert "pre0801" in primary
    # the june block's own tag (PRE_FROM unset), for the script and the deck alike
    jd = run_mode.june_august_drift_dir(lab)
    assert jd.name == "june_august_drift" and "pre0801" not in jd.parent.name and jd.parent.name.endswith("stopRun6j")
    camera.delenv("WIDEFIELD_PRE_FROM")                                  # inside the block
    assert run_mode.june_august_drift_dir(lab) == jd
    # the primary run's epoch dir, read from inside the block
    assert run_mode.primary_output_dir(lab / "grant_figures" / "epoch") == \
        lab / "grant_figures" / primary / "epoch"
    # review 8B N1 / N5: from inside an LP (or ENL) block, both still resolve to the DAQ-arm folders
    camera.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    assert run_mode.june_august_drift_dir(lab) == jd
    assert run_mode.primary_output_dir(lab / "grant_figures" / "epoch") == lab / "grant_figures" / primary / "epoch"
    camera.delenv("WIDEFIELD_LICK_SOURCE")
    for k in PROFILE_ENV:
        camera.delenv(k, raising=False)
    assert run_mode.june_august_drift_dir(lab) == lab / "chronic_stability" / "june_august_drift"   # default run
    assert run_mode.primary_output_dir(lab / "x") == lab / "x"


def test_june_august_drift_refuses_a_missing_primary_input_and_a_pre_from_run(camera, tmp_path):
    import scripts.june_august_drift as jad
    camera.delenv("WIDEFIELD_PRE_FROM")
    with pytest.raises(FileNotFoundError, match="primary run"):
        jad.primary_chronic_csv(tmp_path)
    camera.setenv("WIDEFIELD_PRE_FROM", "0801")
    with pytest.raises(RuntimeError, match="june_august_drift block"):
        jad.main([])


def test_reembedding_selects_june_explicitly():
    src = (REPO / "scripts/reembedding_test.py").read_text(encoding="utf-8")
    assert "with run_mode.with_env(WIDEFIELD_PRE_FROM=None)" in src and "no June reference session" in src


def test_deck_reads_the_tagged_grant_dir_and_counts_missing(camera):
    from wfield_local import lp_licks
    from wfield_local import locanmf_analysis_deck as d
    src = inspect.getsource(d.build_analysis_deck)
    assert "Path(_lpl.output_dir(_grant_root))" in src and "_strict" in src and "june_august_drift_dir" in src
    g = Path("G")
    lp = d._tagged_k("cohort_out", "grant_figures/LP_licks/epoch/x.png", g)
    with run_mode.with_env(WIDEFIELD_LICK_SOURCE="lp"):
        assert lp == Path(lp_licks.output_dir(g / "epoch")) / "x.png"
    assert lp.parent.parent.name.startswith("LP_licks") and lp.parent.parent.name != "LP_licks"
    assert d._tagged_k("labcams", "chronic_stability/whatever.png", g) is None


def test_reembedding_cache_is_keyed_by_the_output_tag():
    src = (REPO / "scripts/reembedding_test.py").read_text(encoding="utf-8")
    assert "sessions_cache_{t}.pkl" in src and "lp_licks.output_tag()" in src



# ------------------------------------------------------------------------------------------------------ 8A / 8B
def test_june_script_and_deck_use_the_derived_cohort_out_root():
    src = (REPO / "scripts/june_august_drift.py").read_text(encoding="utf-8")
    assert 'PathResolver().root("cohort_out")' in src and 'root("labcams")' not in src
    dsrc = (REPO / "wfield_local/locanmf_analysis_deck.py").read_text(encoding="utf-8")
    assert "run_mode.june_august_drift_dir(_grant_root.parent)" in dsrc        # _grant_root = cohort_out/grant_figures


def test_recovery_trajectory_and_chronic_stability_write_tagged():
    from wfield_local import recovery_trajectory as rt
    assert "lp_licks.output_dir(" in inspect.getsource(rt.run)
    src = (REPO / "scripts/chronic_stability.py").read_text(encoding="utf-8")
    assert "require_default" not in src and "_lpl.output_dir(" in src and "_rm.subdir(" in src


def test_design_note_and_5b_titles_follow_the_selection(camera):
    assert "(contact + incomplete)" in tc_mod().design_note()
    camera.setenv("WIDEFIELD_WORKING_TRIALS", "in_working_jaw")
    assert "(contact + incomplete + jaw_only)" in tc_mod().design_note()
    from wfield_local import grant_confusion as gc
    assert "c+i+jaw" in inspect.getsource(gc.fig_confusion_pre_post_working)


def test_working_excluded_comes_from_the_classes(camera, monkeypatch):
    t = tc_mod()
    t._no_class_sessions.cache_clear()
    monkeypatch.setattr(t, "session_trials", lambda lab: None if lab == "PS93_0813" else object())
    assert t.working_excluded_sessions(["PS92_0806", "PS93_0813"]) == ["PS93_0813"]
    t._no_class_sessions.cache_clear()


def test_deck_k_daq_enl_entry_is_not_the_lp_figure(camera):
    from wfield_local import locanmf_analysis_deck as d
    camera.setenv("WIDEFIELD_LICK_SOURCE", "lp")                       # an LP deck block
    enl = d._tagged_k("cohort_out", "grant_figures/DAQ_ENL1s/epoch/x.png", Path("G"))
    assert "LP_licks" not in str(enl) and "ENL" in str(enl)


def test_store_log_names_the_gate():
    from wfield_local import grant_confusion as gc
    assert "camera gate" in inspect.getsource(gc._stored_epoch_decoder)


def tc_mod():
    from wfield_local import trial_classes
    return trial_classes


# ------------------------------------------------------------------------------------------------------ 11-04 follow-up
def test_chronic_drift_vs_reorg_is_routed_like_chronic_stability():
    src = (REPO / "scripts/chronic_drift_vs_reorg.py").read_text(encoding="utf-8")
    assert 'run_mode.subdir(Path(PathResolver().root("cohort_out")) / "chronic_stability")' in src
    assert "require_default" not in src and "assert_writable(out_dir)" in src


def test_render_profile_check_refuses_code_outside_the_worktree(tmp_path):
    from wfield_local import render_profile as rp
    d = rp.load("cohort1_rebuild_20261103")
    root = Path(d["code_root"])
    ok, line = rp.code_check(d, str(root / "wfield_local" / "__init__.py"))
    assert ok and "wfield_local:" in line
    # the MAIN checkout (a .git DIRECTORY: where the editable install points) is refused
    main = tmp_path / "main"
    (main / ".git").mkdir(parents=True)
    (main / "wfield_local").mkdir()
    ok, line = rp.code_check(d, str(main / "wfield_local" / "__init__.py"))
    assert not ok and "REFUSED" in line and "PYTHONPATH" in line
    # an unrelated linked worktree (another repository's common dir) is refused too
    other = tmp_path / "other"
    (other / "wfield_local").mkdir(parents=True)
    (other / ".git").write_text("gitdir: " + str(tmp_path / "elsewhere" / ".git" / "worktrees" / "x"))
    assert not rp.code_check(d, str(other / "wfield_local" / "__init__.py"))[0]
    # without a rebuild root nothing is enforced
    assert rp.code_check({"code_root": str(root)}, str(main / "wfield_local" / "__init__.py"))[0]
