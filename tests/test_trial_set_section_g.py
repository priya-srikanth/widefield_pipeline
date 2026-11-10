"""Section G figures carry the trial set they draw (Priya 11-09): every figure group of `section_g_figures`,
`section_g_retained` and `g4_figures` saves under its own `trial_set_text` sentence. Synthetic records only."""
from __future__ import annotations

import pandas as pd
import pytest
from matplotlib.figure import Figure

from wfield_local import g4_figures as g4f
from wfield_local import section_g_figures as sgf
from wfield_local import section_g_retained as sgr
from wfield_local import trial_set_text as tst

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}
POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("WIDEFIELD_SECTION_G_GROUPS", raising=False)
    yield


@pytest.fixture
def saved(monkeypatch):
    """Every Figure.savefig of the test: (file name, the trial-set text on the figure, whitespace-normalized)."""
    tst.install_savefig_hook()
    orig = Figure.savefig
    out = []

    def rec(self, fname, *a, **k):
        r = orig(self, fname, *a, **k)
        out.append((str(fname), [" ".join(t.get_text().split()) for t in self.texts if t.get_gid() == tst.GID]))
        return r
    monkeypatch.setattr(Figure, "savefig", rec)
    return out


def _n(s):
    return " ".join(s.split())


def _carries(saved, stem, sentence):
    hits = [txt for f, txt in saved if stem in f and f.endswith(".png")]
    assert hits, f"no PNG saved for {stem}: {[f for f, _ in saved]}"
    for txt in hits:
        assert txt and _n(sentence) in txt[0], (stem, txt)


# ------------------------------------------------------------------------------------------- the sentences
def test_sentences_under_the_camera_env(camera):
    work = tst.sentence("working")
    assert work.startswith("trials: camera contact + incomplete (LP licks)")
    assert sgf.camera_grouping() is True
    m_all = sgf.matched_sentence("all", True)
    assert m_all.startswith("trials: pre band = pre-stroke camera WORKING trials (camera contact + incomplete")
    assert "post = EVERY cued trial of the session at all six spouts" in m_all
    lo = sgf.matched_sentence("lickonly", True)
    assert "preserved spouts (>= 20 camera WORKING trials there)" in lo and "post-lick = lick trials (LP licks" in lo
    # the within-session (recoding_test) and similarity sets are the lick-source ENGAGED rows, not camera working
    assert "camera WORKING trials (" not in sgf.within_sentence("lickonly", True)
    assert sgf.within_sentence("lickonly", True).startswith("trials: pre band and post = lick trials (LP licks")
    assert "camera WORKING trials (" not in sgf.similarity_sentence("all", True)
    # the lick window has no camera grouping and no no-lick panels
    c_lick = sgf.confusion_sentence("lickonly", "lick", True)
    assert "no-lick" not in c_lick and "camera WORKING" not in c_lick
    c_cue = sgf.confusion_sentence("all", "cue", True)
    assert "PRE no-lick = pre-stroke no-lick trials (no LP licks within" in c_cue
    assert sgf.g13_sentence([{"train_classes": {"groups": "camera"}}]).startswith(
        "trials: discriminator trained on pre-stroke camera WORKING trials")
    assert "vs no-lick trials" in sgf.g13_sentence([{"train_classes": {"groups": "daq"}}])
    assert sgf.counts_sentence().startswith("trials: every cued trial of each session, per spout")


def test_camera_grouping_read_off_the_records(camera):
    daq = {"S": {"arms": {"all": {"post-cue": {"pre_band_rows": "engaged"}}}}}
    cam = {"S": {"arms": {"all": {"confusion": {"post-cue": {"groups": "camera"}}}}}}
    assert sgf.camera_grouping(daq) is False
    assert sgf.camera_grouping(cam) is True
    assert "pre band = pre-stroke lick trials" in sgf.matched_sentence("all", sgf.camera_grouping(daq))


def test_retained_sentence_names_both_arms(camera):
    s = sgr.trial_set_sentence({})
    assert s.startswith("trials: ALL arm: pre band = pre-stroke lick trials")
    assert "  AND  LICK-ONLY arm: pre band and post = lick trials" in s


def test_g4_sentence_left_and_right(camera):
    s = g4f.trial_set_sentence()
    assert s.startswith("trials: G4 (left): every trial of the post session at its preserved spouts")
    assert "each session z-scored on its camera contact rows" in s
    assert "never trained on: jaw only / unknown (jaw unjudged) / quit" in s
    D = pd.DataFrame([dict(train="camera")])
    assert "AND  right panel: discriminator trained on pre-stroke camera WORKING" in g4f.trial_set_sentence(None, D)


# ------------------------------------------------------------------------------------------- on the saved figures
def _within(label, post, pre_mean, n):
    return {"n_positions": n, "chance": 1.0 / n, "post_arm": "engaged only",
            "within_pre_band": {"mean": pre_mean, "sd": 0.05, "min": pre_mean - 0.1, "max": pre_mean + 0.1, "n": 5},
            "post": {label: {"within_accuracy": post, "z": (post - pre_mean) / 0.05,
                             "inside_pre_range": abs(post - pre_mean) <= 0.1}}}


def _rec():
    lab = "ZZ1_0102"
    counts = {"pre": {"engaged": {p: 50.0 for p in POS}, "undetected": {p: 5.0 for p in POS}},
              "post": {"engaged": {p: 20 for p in POS}, "undetected": {p: 10 for p in POS}}}
    return {lab: {"animal": "ZZ1", "label": lab, "phase_tag": "post", "counts": counts,
                  "arms": {"all": {"chance": 1 / 6, "post-cue within-session": _within(lab, 0.5, 0.6, 6)},
                           "lickonly": {"chance": 1 / 4, "post-cue within-session": _within(lab, 0.6, 0.7, 4)}}}}


def test_section_g_family_figures_carry_their_sentences(camera, saved, tmp_path):
    made = sgf._render_family(_rec(), tmp_path, "section_g", "POST-STROKE")
    assert made
    _carries(saved, "section_g_counts", sgf.counts_sentence())
    _carries(saved, "section_g_grid_all", sgf.within_sentence("all", True))
    _carries(saved, "section_g_grid_lickonly", sgf.within_sentence("lickonly", True))
    # nothing leaks out of the family: no trial set is active afterwards
    assert tst.current() is None


def test_retained_figure_carries_both_arms(camera, saved, tmp_path):
    q = sgr.render(_rec(), tmp_path)
    assert q
    _carries(saved, "section_g_retained", sgr.trial_set_sentence(_rec()))


def test_g4_figure_carries_its_sentence(camera, saved, tmp_path):
    C = pd.DataFrame([dict(animal="PS92", session="PS92_0818", epoch="acute", align=al, group=g, value=0.5, n=30,
                           pre_reference=0.6, pre_reference_is=g, frame="camclass/z-contact/pos-contact+incomplete/"
                           "neg-no_response") for al in g4f.ALIGNS for g in ("contact", "no_response", "jaw_only")])
    D = pd.DataFrame([dict(animal="PS92", session="PS92_0818", epoch="acute", align=al, post_nolick=0.4,
                           train="camera", ref_engaged=0.7, ref_nolick=0.3) for al in g4f.ALIGNS])
    P = pd.DataFrame([dict(animal="PS92", align=al, group="contact", value=0.6) for al in g4f.ALIGNS])
    g4f.draw(C, D, P, tmp_path / "section_g_g4_camclass.png")
    _carries(saved, "section_g_g4_camclass", g4f.trial_set_sentence(C, D, groups=["contact", "jaw_only",
                                                                                  "no_response"]))
