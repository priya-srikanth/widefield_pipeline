"""Follow-ups of the 11-04 port (coordinator): C ON with the per-readout split, run-tag parts for non-default scoring
options and the profile arms, the G4 primary figure."""
import json
import os

import numpy as np
import pytest

from wfield_local import position_coding_directions as pcd
from wfield_local import run_mode
from wfield_local import spout_scaling as ss
from wfield_local import working_trials as wt

OPTS = ("WIDEFIELD_SPOUT_SCORE_CENTRE", "WIDEFIELD_PROJECTION_FRAME", "WIDEFIELD_PROJECTION_CENTRE",
        "WIDEFIELD_SECTION_D_SCORED", "WIDEFIELD_SECTION_G_GROUPS", "WIDEFIELD_G4_DISC", "WIDEFIELD_G4_CENTRE",
        "WIDEFIELD_G9_CLASSES", "WIDEFIELD_STATE_DISC_SCALING", "WIDEFIELD_STATE_DISC_CENTRE")


@pytest.fixture
def clean(monkeypatch):
    for k in OPTS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_default_scoring_has_no_tag_part(clean):
    assert run_mode.scoring_part() == ""


@pytest.mark.parametrize("env,part", [
    ({"WIDEFIELD_SPOUT_SCORE_CENTRE": "n"}, "sc-ctrN"),
    ({"WIDEFIELD_SPOUT_SCORE_CENTRE": "cto"}, "sc-ctrCto"),
    ({"WIDEFIELD_SPOUT_SCORE_CENTRE": "working"}, "sc-ctrW"),
    ({"WIDEFIELD_PROJECTION_FRAME": "session_z"}, "sc-projz"),
    ({"WIDEFIELD_PROJECTION_CENTRE": "cto"}, "sc-pctrCto"),
    ({"WIDEFIELD_SECTION_D_SCORED": "engaged"}, "sc-dEng"),
    ({"WIDEFIELD_SECTION_G_GROUPS": "daq"}, "sc-gDaq"),
    ({"WIDEFIELD_G4_DISC": "daq"}, "sc-g4Daq"),
    ({"WIDEFIELD_G4_CENTRE": "working"}, "sc-g4cWorking"),
    ({"WIDEFIELD_G9_CLASSES": "daq"}, "sc-g9Daq"),
    ({"WIDEFIELD_STATE_DISC_SCALING": "spout_frame", "WIDEFIELD_STATE_DISC_CENTRE": "engaged"}, "sc-sdSpout+sdcEng"),
])
def test_each_non_default_option_tags_the_run(clean, env, part):
    for k, v in env.items():
        clean.setenv(k, v)
    assert run_mode.scoring_part() == part
    assert run_mode.input_run_tag(f"x/{part}/y") == part


def test_profile_centre_arms_write_to_their_own_folders(monkeypatch):
    from wfield_local import render_profile
    for arm, part in (("centreN", "_sc-ctrN+pctrN"), ("centreCto", "_sc-ctrCto+pctrCto")):
        env = dict(os.environ)
        d = render_profile.apply("cohort1_rebuild_20261103", env, arm=arm)
        for k in list(os.environ):
            if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
                monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
                monkeypatch.setenv(k, v)
        monkeypatch.setattr(run_mode, "csig_part", lambda: "")
        assert run_mode.tag() == d["arms"][arm]["expected_run_tag"] if "arms" in d else True
        assert run_mode.tag().endswith(part)


# ------------------------------------------------------------------------------------------- C: the split
def test_contact_z_centres_each_session_on_its_contact_rows():
    rng = np.random.default_rng(0)
    XE, XU = rng.normal(size=(100, 3)) + 4, rng.normal(size=(40, 3)) - 2
    GE, GU = np.repeat([0, 1], 50), np.repeat([0, 1], 20)
    ce = np.where(rng.random(100) < .8, "contact", "incomplete").astype(object)
    cu = np.full(40, "none", dtype=object)
    A, B = ss.contact_z(XE, XU, GE, GU, ce, cu)
    for i in (0, 1):
        np.testing.assert_allclose(A[(GE == i) & (ce == "contact")].mean(0), 0, atol=1e-9)
    assert ss.contact_z(XE, XU, GE, GU) == (XE, XU)                  # DAQ gate: the production z


def test_engagement_axis_comes_from_the_eax_rows():
    rng = np.random.default_rng(1)
    XE, XU = rng.normal(size=(120, 4)), rng.normal(size=(60, 4)) + 1
    en = np.array(pcd.BY_SEVERITY * 20)
    GE, e_pre = np.repeat([0, 1], 60), np.ones(120, bool)
    u_pre = np.ones(60, bool)
    EX, EU = XE * 2 + 3, XU * 2 + 3
    r = pcd.fit_reference(XE, en, e_pre, GE, XU, u_pre, {0, 1}, ["A_1", "A_2"], ("dom",), e_rows=(EX, EU))
    np.testing.assert_allclose(r["e_axis"], pcd.engagement_axis(EX, EU))
    r0 = pcd.fit_reference(XE, en, e_pre, GE, XU, u_pre, {0, 1}, ["A_1", "A_2"], ("dom",))
    np.testing.assert_allclose(r0["e_axis"], pcd.engagement_axis(XE, XU))


def test_projection_work_refuses_a_camera_pool_without_indices(clean):
    clean.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    clean.setenv("WIDEFIELD_WORKING_TRIALS", "in_working")
    with pytest.raises(RuntimeError):
        wt.projection_work(["A_1"], object(), np.zeros(3), np.zeros(2), "cue")
    assert wt.projection_work(["A_1"], object(), np.zeros(3), np.zeros(2), "lick") is None    # lick gate


# ------------------------------------------------------------------------------------------- G4 figure
def test_g4_figure_builds_and_replots(tmp_path):
    from wfield_local import g4_figures as gf
    rng = np.random.default_rng(2)
    rec = {}
    for an in ("PS92", "PS93", "PS94", "PS95"):
        for k in range(3):
            post = {g: {"frac_spout_balanced": float(rng.random()), "n": 50, "pre_reference": .6,
                        "pre_reference_is": g if g != "after_end" else "working"} for g in wt.G_GROUPS}
            pre_ref = {g: {"per_session": list(rng.random(4))} for g in wt.G_GROUPS}
            rec[f"{an}_09{k}"] = {"phase_tag": "post", "animal": an,
                                  "g4_camclass_cue": {"post": post, "pre_reference": pre_ref, "frame": "camclass/z"},
                                  "fits_engaged_cue": {"post_value": .4, "engaged": {"mean": .9},
                                                       "nolick": {"mean": .2}}}
    eps = {f"{an}_09{k}": ep for an in ("PS92", "PS93", "PS94", "PS95")
           for k, ep in enumerate(("acute", "subacute", "chronic"))}
    C, D, P = gf.g4_table(rec, eps.get)
    assert len(C) == 4 * 3 * len(wt.G_GROUPS) and len(D) == 12
    fig = gf.draw(C, D, P, tmp_path / "g4.png")
    assert (tmp_path / "g4.png").exists() and not fig.axes[0].spines["top"].get_visible()
    (tmp_path / "s.json").write_text(json.dumps(rec))
    import wfield_local.epochs as E
    orig = E.epoch_of
    try:
        E.epoch_of = eps.get
        assert gf.main(["--section-g", str(tmp_path / "s.json"), "--out", str(tmp_path)]) == 0
    finally:
        E.epoch_of = orig
    assert gf.main(["--out", str(tmp_path), "--plot-only"]) == 0
