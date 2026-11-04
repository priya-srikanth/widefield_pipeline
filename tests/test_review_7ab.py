"""Review 7A (working rows, captions, PS93_0813 visibility, NB4 / NB6) and 7B (PRE_FROM blocks, primary-tag inputs,
June selection, deck tagged paths) at 6a85304d."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from wfield_local import run_mode
from wfield_local import spout_scaling as ss

PROFILE_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_TRIAL_WINDOW": "cue_stop", "WIDEFIELD_TRIAL_END": "lick",
               "WIDEFIELD_TRIAL_STOPPED": "run", "WIDEFIELD_PRE_FROM": "0801"}


@pytest.fixture
def camera(monkeypatch):
    for k, v in PROFILE_ENV.items():
        monkeypatch.setenv(k, v)
    for k in ("WIDEFIELD_WORKING_TRIALS", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_REBUILD_ROOT",
              "WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LOCANMF_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ------------------------------------------------------------------------------------------------------ 7A
def test_5b_never_reads_the_quit_tail_json_under_a_camera_selection():
    from wfield_local import grant_confusion as gc
    src = inspect.getsource(gc.fig_confusion_pre_post_working)
    assert "if _wt.is_camera():\n            return None" in src.replace("\r\n", "\n")       # main's guard, verbatim
    assert '_frozen_frame(bd, an, align, sel="working")' in src
    assert "WORKING trials (c+i)" in src


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
        src = Path(f"wfield_local/{mod}.py").read_text(encoding="utf-8")
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
    models = {"full": M(), "loso": {"a": M()}, "trials": "sel", "pre_labels": ["a"],
              "pi_train": {"full": np.full(6, 1 / 6), "loso": {"a": np.full(6, 1 / 6)}}}
    with pytest.raises(ss.BalEmCentreRefused, match="post session c"):
        lfd.frozen_scored_frame(models, XE, GE, np.zeros((0, 3)), np.zeros(0, int), ["a", "b", "c"], [0], work=work)


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
                              text=True, check=True).stdout.strip()
        mine = subprocess.run(["git", "hash-object", "wfield_local/working_trials.py"], capture_output=True, text=True,
                              check=True).stdout.strip()
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
    lab = tmp_path / "labcams"
    primary = lp_licks.output_tag()
    assert "pre0801" in primary
    # the june block's own tag (PRE_FROM unset), for the script and the deck alike
    jd = run_mode.june_august_drift_dir(lab)
    assert jd.name == "june_august_drift" and "pre0801" not in jd.parent.name and jd.parent.name.endswith("stopRun6")
    camera.delenv("WIDEFIELD_PRE_FROM")                                  # inside the block
    assert run_mode.june_august_drift_dir(lab) == jd
    # the primary run's epoch dir, read from inside the block
    assert run_mode.primary_output_dir(lab / "grant_figures" / "epoch") == \
        lab / "grant_figures" / primary / "epoch"
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
    src = Path("scripts/reembedding_test.py").read_text(encoding="utf-8")
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
    src = Path("scripts/reembedding_test.py").read_text(encoding="utf-8")
    assert "sessions_cache_{t}.pkl" in src and "lp_licks.output_tag()" in src
