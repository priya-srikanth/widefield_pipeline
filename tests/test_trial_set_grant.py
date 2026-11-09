"""Trial-set captions on the GRANT figures (Priya 11-09: "EVERY figure must state EXACTLY which trials were included /
excluded"). `grant_kit._save` resolves the sentence of every grant figure from its file stem
(`grant_kit.trial_set_sentence`), so the serial render, a spawned worker and a direct call all carry it. Synthetic
inputs only: nothing here reads real data or writes outside tmp_path."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wfield_local import grant_kit as gk  # noqa: E402
from wfield_local import trial_set_text as tst  # noqa: E402

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}


@pytest.fixture(autouse=True)
def _no_share_stamp(monkeypatch):
    """`lp_licks.stamp` reads the run's coverage table on M: under the camera gate: a plain purple run stamp."""
    from wfield_local import lp_licks
    monkeypatch.setattr(lp_licks, "stamp", lambda fig: fig.text(0.995, 0.002, "RUN STAMP", color=tst.RUN_STAMP_COLOR))
    # the footer's session coverage resolves every registered session's path on the share (minutes on this box)
    monkeypatch.setattr(gk, "coverage_note", lambda *a, **k: "coverage (stub)")


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    yield


@pytest.fixture
def daq(monkeypatch):
    for k in CAMERA_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    yield


@pytest.fixture
def only():
    yield gk.set_only
    gk.set_only(None, None)


def _body(v):
    return tst.sentence(v).split(": ", 1)[1]


def _flat(s):
    return " ".join(str(s).split())


def _caption(fig):
    t = [a for a in fig.texts if a.get_gid() == tst.GID]
    assert len(t) == 1, [a.get_text() for a in fig.texts]
    return _flat(t[0].get_text()).removesuffix(" RUN STAMP")


# ------------------------------------------------------------------------------------------------ the sentences
@pytest.mark.parametrize("stem", ["grant_6_pattern_cue_working", "grant_6b_pattern_per_session_precue_working",
                                  "grant_7b_reliability_cue_working", "grant_8d_crossnobis_delta_cue_working",
                                  "grant_10_best_match_precue_working", "grant_11_encoder_gain_shape_cue_working"])
def test_pattern_family_working_states_lick_pre_reference(camera, stem):
    s = gk.trial_set_sentence(stem, 10)
    assert s == (f"trials: POST-STROKE {_body('working')}  AND  PRE-STROKE REFERENCE: {_body('lick')}; "
                 "session x position cells with < 10 trials dropped")
    assert "camera contact + incomplete" in s and "lick + miss-while-working" not in s


def test_pattern_family_lick_window(camera):
    assert gk.trial_set_sentence("grant_8_crossnobis_lick_lick", 10) == (
        f"trials: pre- and post-stroke sessions, {_body('lick')}; session x position cells with < 10 trials dropped")


def test_5c_pre_panel_follows_the_camera_working_frame(camera):
    s = gk.trial_set_sentence("grant_5c_confusion_per_session_cue_working")
    assert s == ("trials: pre-stroke sessions (each held out, leave-one-session-out) and post-stroke sessions, "
                 + _body("working"))


def test_5c_under_the_daq_gate_pre_is_lick(daq):
    s = gk.trial_set_sentence("grant_5d_confusion_delta_precue_working")
    assert "DAQ quit-tail rule" in s and "camera contact" not in s
    assert s.endswith("leave-one-session-out): " + _body("lick"))


def test_5b_camera(camera):
    s = gk.trial_set_sentence("grant_5b_confusion_working_cue")
    assert s.startswith("trials: LEFT = PRE-stroke sessions, each held out (leave-one-session-out): " + _body("working"))
    assert "MIDDLE = every POST-stroke trial" in s and s.endswith("RIGHT = POST-STROKE " + _body("working"))


def test_behaviour_and_json_free_families(camera):
    assert "engaged trials of the behavior per-position table" in gk.trial_set_sentence("grant_1_behaviour_by_position")
    assert gk.trial_set_sentence("grant_1b_behaviour_pre_collapsed").startswith("trials: engaged trials")
    s3a = gk.trial_set_sentence("grant_3a_coding_retained")
    assert _body("miss_working") in s3a and _body("lick") in s3a and "< 0.5" in s3a and ">= 10 engaged" in s3a
    assert "section_g arm 'lickonly'" in gk.trial_set_sentence("grant_3b_frozen_vs_within")


def test_not_a_grant_stem():
    assert gk.trial_set_sentence("epoch_15r_map_cue_working") is None
    assert gk.trial_set_sentence("grant_99_unknown") is None


# ------------------------------------------------------------------------------------------------ JSON-fed
@pytest.fixture
def figroot(tmp_path, monkeypatch):
    root = tmp_path / "figs"
    root.mkdir()
    monkeypatch.setattr(gk, "_fig_root", lambda: root)
    monkeypatch.setattr(gk, "_mode_name", lambda n: n)
    return root


def test_joint_decoder_figures_read_the_scored_set(camera, figroot):
    (figroot / "joint_xsession_decoder_cue.json").write_text(json.dumps(
        {an: {"scored_set": "working"} for an in gk.ANIMALS}), encoding="utf-8")
    (figroot / "joint_xsession_decoder_lick.json").write_text(json.dumps(
        {"PS92": {"scored_set": "engaged"}, "PS93": {}}), encoding="utf-8")
    s4 = gk.trial_set_sentence("grant_4_confusion_prestroke_cue")
    assert s4 == ("trials: PRE-STROKE sessions only (config phase pre), each scored by a decoder that never saw it "
                  f"(leave-one-session-out); post-cue: {_body('working')}")
    s2 = gk.trial_set_sentence("grant_2_prestroke_crossday_decoding")
    assert f"post-cue: {_body('working')}  AND  post-lick: {_body('lick')}" in s2
    assert gk.trial_set_sentence("grant_4_confusion_prestroke_precue") is None          # no JSON, no claim


def test_section_g_arms(camera, figroot):
    blk = {"pre_arm": "camera WORKING trials (contact + incomplete), leave-one-session-out",
           "pre_nolick_arm": "PRE-stroke NO-LICK trials, decoder trained on other sessions' engaged",
           "post_arm": "ALL trials (engaged + no-lick)"}
    (figroot / "section_g.json").write_text(json.dumps(
        {"PS92_0820": {"arms": {"all": {"confusion": {"post-cue": blk}}}}}), encoding="utf-8")
    s = gk.trial_set_sentence("grant_5_confusion_pre_post_postcue")
    assert s.startswith("trials (LP licks): LEFT = PRE-stroke camera WORKING trials (contact + incomplete)")
    assert "RIGHT = POST-stroke ALL trials (engaged + no-lick)" in s
    assert gk.trial_set_sentence("grant_5_confusion_pre_post_precue") is None


# ------------------------------------------------------------------------------------------------ on the figure
def _render_through_save(path, min_trials=7):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([0, 1], [0, 1])
    gk._save(fig, path, dpi=50)
    return fig


def test_save_stamps_the_stem_sentence_with_the_callers_min_trials(camera, tmp_path):
    fig = _render_through_save(tmp_path / "grant_7_splithalf_cue_working.png", min_trials=7)
    assert _caption(fig) == _flat(gk.trial_set_sentence("grant_7_splithalf_cue_working", 7))
    assert (tmp_path / "grant_7_splithalf_cue_working.png").exists()
    plt.close(fig)


def test_an_activated_set_wins(camera, tmp_path):
    with tst.active(None, text="events: something narrower"):
        fig = _render_through_save(tmp_path / "grant_7_splithalf_cue_working.png")
    assert _caption(fig) == "events: something narrower"
    plt.close(fig)


def test_non_grant_stem_is_unstamped(tmp_path):
    fig = _render_through_save(tmp_path / "other_figure.png")
    assert not [a for a in fig.texts if a.get_gid() == tst.GID]
    plt.close(fig)


@pytest.fixture
def grab(monkeypatch):
    figs = []
    real = gk._save

    def _rec(fig, path, **kw):
        figs.append((path, fig))
        return real(fig, path, **kw)
    return figs, _rec


def test_real_delta_grid_figure_carries_the_sentence(camera, tmp_path, monkeypatch, only, grab):
    """6d through the real `fig_pattern_delta` -> `_delta_grid` -> `_save` (the generic grant_kit grid)."""
    from wfield_local import grant_similarity as gs
    figs, rec = grab
    M = np.eye(6)
    monkeypatch.setattr(gs, "_matrices_pattern", lambda a, v, m: ({"PS92": {"PRE": M, 1: M * 0.5}}, [1]))
    monkeypatch.setattr(gs, "_delta_cis", lambda *a, **k: {})
    monkeypatch.setattr(gk, "_save", rec)
    only("cue", "working")
    made = gs.fig_pattern_delta(tmp_path, min_trials=8)
    assert made and figs
    path, fig = figs[0]
    assert path.name == "grant_6d_pattern_delta_cue_working.png"
    assert _caption(fig) == _flat(gk.trial_set_sentence(path.stem, 8))
    assert "< 8 trials dropped" in _caption(fig)


def test_real_prestroke_confusion_figure_carries_the_sentence(camera, tmp_path, figroot, monkeypatch, only, grab):
    """4 through the real `fig_confusion_prestroke` with a synthetic joint-decoder JSON."""
    from wfield_local import config
    from wfield_local import grant_confusion as gc
    figs, rec = grab
    pre = {an: f"{an}_0806" for an in gk.ANIMALS}                 # synthetic labels (phase_labels reads the share)
    monkeypatch.setattr(config, "phase_labels", lambda phase: list(pre.values()) if phase == "pre" else [])
    C = (np.eye(6) * 8 + 1).tolist()
    (figroot / "joint_xsession_decoder_cue.json").write_text(json.dumps(
        {an: {"scored_set": "working", "confusion": {lab: C}} for an, lab in pre.items()}), encoding="utf-8")
    monkeypatch.setattr(gc, "_fig_root", lambda: figroot)
    monkeypatch.setattr(gc, "_mode_name", lambda n: n)
    monkeypatch.setattr(gc, "_save", rec)
    only("cue", None)
    p = gc.fig_confusion_prestroke(tmp_path)
    assert p and p.name == "grant_4_confusion_prestroke_cue.png"
    _path, fig = figs[0]
    assert _caption(fig) == _flat(gk.trial_set_sentence("grant_4_confusion_prestroke_cue"))
    assert _body("working")[:40] in _caption(fig)
