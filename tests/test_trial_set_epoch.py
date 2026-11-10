"""The trial set of every pooled EPOCH family (`epoch_grant_figures`, Priya 11-09: "EVERY figure must state EXACTLY which
trials were included / excluded").

The arm loop activates the ARM's class; families that draw another set are wrapped in `_trial_set(<family>)` with the
sentence `epoch_kit` builds from the collectors' own rules. Checked here, synthetic only (no data, no M:):

  * the sentence of every family under the production camera env (and the DAQ-gate composition),
  * the driver (`main`, one arm, every family replaced by a fake that saves through the real epoch save path) stamps
    each family's figures with ITS set, and the behavior families (1b / 1b daq / 1c) with theirs,
  * 13n (two sets on one figure) is drawn under the union while the rest of family 13 carries the segments.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from wfield_local import epoch_figures as ef  # noqa: E402
from wfield_local import epoch_grant_figures as g  # noqa: E402
from wfield_local import epoch_kit as ek  # noqa: E402
from wfield_local import trial_set_text as tst  # noqa: E402

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    yield


def _norm(s):
    return " ".join(str(s).split())


def _body(s):
    return s.split(": ", 1)[1]


# ------------------------------------------------------------------------------------------------ the sentences
@pytest.mark.parametrize("fam", ["acc", "5c", "5cr", "5r", "5rm", "5ro", "5rmo", "14m"])
@pytest.mark.parametrize("align,variant", [("precue", "working"), ("cue", "working"), ("lick", "lick"),
                                           ("cue", "lick"), ("precue", "lick")])
def test_decoder_families_are_the_arm_class_under_camera(camera, fam, align, variant):
    """Camera working frame: the frozen decoder trains, scores and fills its PRE panel on the arm's class."""
    assert g.trial_set_sentence(fam, align, variant) == tst.sentence(variant)


@pytest.mark.parametrize("fam", ["mat", "scal", "10e", "10cs"])
def test_pattern_families_name_the_lick_pre_reference(camera, fam):
    """`_collect_7`: post-stroke = the arm's class, PRE-stroke reference + PRE panel = LICK trials in every class."""
    s = g.trial_set_sentence(fam, "cue", "working")
    assert s == (f"trials: post-stroke = {_body(tst.sentence('working'))}  AND  "
                 f"{ek.PATTERN_PRE_ROLE} = {_body(tst.sentence('lick'))}")
    assert g.trial_set_sentence(fam, "lick", "lick") == tst.sentence("lick")
    assert g.trial_set_sentence(fam, "precue", "lick") == tst.sentence("lick")


@pytest.mark.parametrize("fam", ["12s", "12b"])
def test_stopped_families_name_stopped_and_the_engaged_template(camera, fam):
    from wfield_local import grant_figures as G
    s = g.trial_set_sentence(fam, "cue", "working")
    assert s.startswith(f"trials: post-stroke + pre-stroke = {_body(tst.sentence('stopped'))}; ")
    assert s.endswith(f"  AND  scored against the pre-stroke template = {_body(tst.sentence('lick'))}")
    if fam == "12s":
        assert f">= {G.MIN_STOPPED} stopped trials" in s and f">= {G.MIN_STOPPED_REF} pre-stroke stopped" in s
    else:
        assert ">= 20 stopped trials (all positions pooled)" in s


def test_state_family_is_segments_not_trials(camera):
    from wfield_local import locomotor_decoder as ld
    from wfield_local import locomotor_state as ls
    s = g.trial_set_sentence("13s", "cue", "working")
    assert s.startswith("segments (not trials): ")
    assert f"{ls.SEGMENT_S:g} s windows" in s and f"at most {ls.MAX_SEGMENTS_PER_PERIOD} per period" in s
    assert "REST (behavior_events rest mask" in s and "RUNNING (treadmill" in s
    assert f"first DAQ lick within {ls.LICK_MAX_LATENCY_S:g} s of the cue" in s
    assert f"sessions with < {ld.MIN_PER_CLASS} segments of any class" in s
    both = ek.state_vs_position_text()
    assert both == f"{s}  AND  position bar {tst.sentence('working')}"


def test_evoked_family_is_every_cued_trial(camera):
    s = g.trial_set_sentence("15e", "cue", "working")
    assert s.startswith("trials: every cued trial at each position") and "no engagement / camera class" in s


def test_behaviour_families(camera):
    from wfield_local.epoch_behaviour import MIN_POSITION_TRIALS
    cam = g.trial_set_sentence("1b")
    assert cam.startswith(tst.sentence("working_period") + "; hit = camera contact")
    assert f">= {MIN_POSITION_TRIALS} trials" in cam
    daq = g.trial_set_sentence("1b daq")
    assert daq == g.trial_set_sentence("1c")
    assert "ENGAGED trials" in daq and "DAQ quit-tail rule on DAQ licks" in daq and "camera" not in daq


def test_daq_gate_composes_the_lick_pre_side(camera, monkeypatch):
    """Outside the camera working frame the frozen decoder trains on (and its PRE panel holds) the LICK rows."""
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    s = g.trial_set_sentence("acc", "cue", "working")
    assert s == (f"trials: post-stroke = {_body(tst.sentence('working'))}  AND  "
                 f"{ek.DECODER_PRE_ROLE} = {_body(tst.sentence('lick'))}")
    assert ek.REFIT_PRE_ROLE in g.trial_set_sentence("5cr", "cue", "working")
    assert g.trial_set_sentence("acc", "lick", "lick") == tst.sentence("lick")
    # 1b's camera period is the video classes whatever the gate: still named, never the DAQ wording
    cam = g.trial_set_sentence("1b")
    assert "camera working period" in cam and "OUT_OF_WORKING_PERIOD" in cam


def test_daq_trained_cross_check_is_named(camera, monkeypatch):
    monkeypatch.setenv("WIDEFIELD_LP_FROZEN", "daq")
    assert g.trial_set_sentence("acc", "cue", "working").endswith("DAQ-TRAINED model (fitted on the DAQ arm's "
                                                                   "pre-stroke trials, DAQ licks)")
    assert "DAQ-TRAINED" not in g.trial_set_sentence("5cr", "cue", "working")      # the refit has no frozen model


# ------------------------------------------------------------------------------------------------ the driver
ARM_FAMS = ("acc", "5c", "5cr", "5r", "5rm", "5ro", "5rmo", "10e", "10cs", "12s", "12b", "13s", "14m", "15e",
            "mat", "scal")


@pytest.fixture
def fake_render(monkeypatch, tmp_path):
    """Every family of the driver replaced by a fake that draws one figure and saves it through the REAL epoch save path
    (`epoch_figures._save_png_svg`), recording the trial-set text the saved figure carries."""
    from wfield_local import lp_licks
    monkeypatch.setattr(lp_licks, "stamp", lambda fig: None)
    seen = {}

    def _draw(key):
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot([0, 1], [0, 1])
        q = tmp_path / "figs" / f"{key.replace(' ', '_').replace('/', '_')}.png"
        q.parent.mkdir(parents=True, exist_ok=True)
        ef._save_png_svg(fig, q, dpi=30)
        seen[key] = _norm(" ".join(t.get_text() for t in fig.texts if t.get_gid() == tst.GID))
        plt.close(fig)
        return q

    def fam(key, many=False):
        def f(*a, **k):
            # (out, align, variant, wname) for most; the 5c / acc collectors pass per_animal first
            align, variant = next((a[i], a[i + 1]) for i in range(len(a) - 1)
                                  if a[i] in ("precue", "cue", "lick") and a[i + 1] in tst.VARIANTS)
            p = _draw(f"{key} {align}/{variant}")
            return [p] if many else p
        return f

    monkeypatch.setattr(g, "_epoch_arm", lambda align, variant: {})
    monkeypatch.setattr(g.ef, "epoch_coverage", lambda per_animal: {"unassigned": [], "n": 0, "per_epoch": {}})
    for name, key, many in [("_per_position_accuracy", "acc", False), ("_confusion_rows", "5c", True),
                            ("_refit_confusion_rows", "5cr", True), ("_fig_15_evoked_maps", "15e", False),
                            ("_fig_15pa_evoked_maps_by_animal", "15pa", False), ("_fig_14_beta_maps", "14m", False),
                            ("_fig_14pa_beta_maps_by_animal", "14pa", False), ("_fig_14z_beta_vs_zero", "14z", False),
                            ("_fig_13_state", "13s", True), ("_fig_10cs", "10cs", False),
                            ("_fig_12b_stopped_pooled", "12b", False), ("_fig_12_stopped", "12s", False),
                            ("_fig_10e_best_match_grid", "10e", True), ("_frozen_vs_refit", "5r", True),
                            ("_frozen_vs_refit_matched", "5rm", True), ("_frozen_vs_refit_overall", "5ro", True),
                            ("_frozen_vs_refit_overall_matched", "5rmo", True)]:
        monkeypatch.setattr(g, name, fam(key, many))
    monkeypatch.setattr(g, "SCALAR_FAMILIES", (("9", fam("scal")),))
    monkeypatch.setattr(g, "MATRIX_FAMILIES", (("6", "_matrices_pattern", "u", "viridis", None, "stem"),))
    monkeypatch.setattr(g, "_matrix_family", lambda key, coll, unit, cmap, scale, stem, out, align, variant, wname:
                        _draw(f"mat {align}/{variant}"))
    monkeypatch.setattr(g, "fig_behaviour", lambda out, period="camera": _draw("1b" if period == "camera" else "1b daq"))
    monkeypatch.setattr(g, "fig_behaviour_timecourse", lambda out: _draw("1c"))
    return seen


EXPECT = {"acc": "acc", "5c": "5c", "5cr": "5cr", "5r": "5r", "5rm": "5rm", "5ro": "5ro", "5rmo": "5rmo",
          "10e": "10e", "10cs": "10cs", "12s": "12s", "12b": "12b", "13s": "13s", "14m": "14m", "14pa": "14m",
          "14z": "14m", "15e": "15e", "15pa": "15e", "mat": "mat", "scal": "scal"}


@pytest.mark.parametrize("arm,align,variant", [("cue", "cue", "working"), ("lick", "lick", "lick")])
def test_driver_stamps_each_family_with_its_set(camera, fake_render, tmp_path, arm, align, variant):
    rc = g.main(["--arm", arm, "--only", *ARM_FAMS, "1b", "1c", "--output", str(tmp_path / "out")])
    assert rc == 0
    seen = fake_render
    for key, fam in EXPECT.items():
        if fam in ("12s", "12b") and variant != "working":
            assert f"{key} {align}/{variant}" not in seen               # working arms only (one figure per alignment)
            continue
        got = seen[f"{key} {align}/{variant}"]
        assert got == _norm(g.trial_set_sentence(fam, align, variant)), key
    for key in ("1b", "1b daq", "1c"):
        assert seen[key] == _norm(g.trial_set_sentence(key)), key
    assert tst.current() is None                                         # nothing leaks past the render


def test_behaviour_families_in_parallel_parent(camera, fake_render, tmp_path):
    """`_parallel_arms` renders 1b / 1c in the parent, each under its own set (no arm family wanted: no child starts)."""
    import argparse
    g._parallel_arms(argparse.Namespace(parallel_arms=2, output=None), {"1b", "1c"}, tmp_path)
    assert fake_render["1b daq"] == _norm(g.trial_set_sentence("1b daq"))
    assert fake_render["1b"] == _norm(g.trial_set_sentence("1b"))
    assert fake_render["1c"] == _norm(g.trial_set_sentence("1c"))


def test_a_caption_that_cannot_be_built_is_no_caption(camera, fake_render, monkeypatch, tmp_path):
    """Never the arm's set on a figure that draws another one: a failing sentence leaves the figure uncaptioned."""
    monkeypatch.setattr(ek, "stopped_trial_text", lambda *a, **k: 1 / 0)
    with tst.active("working", "cue"):
        with g._trial_set("12s", "cue", "working"):
            g._fig_12_stopped(tmp_path, "cue", "working", "x")
    assert fake_render["12s cue/working"] == ""


# ------------------------------------------------------------------------------------------------ 13n
def test_13n_carries_both_sets_and_the_rest_of_13_the_segments(camera, monkeypatch, tmp_path):
    import numpy as np

    from wfield_local import epoch_state as es
    from wfield_local import locomotor_decoder as ld
    got = {}

    def rec(name):
        got[name] = tst.current_sentence()
        return tmp_path / f"{name}.png"

    store = {"PS94": {"PRE": [{"balacc": 0.9, "per_class": {"quiet": .9, "running": .9, "licking": .9},
                               "counts": {"quiet": 5, "running": 5, "licking": 5}, "confusion": np.eye(3)}],
                      3: {"balacc": 0.8, "per_class": {"quiet": .8, "running": .8, "licking": .8},
                          "counts": {"quiet": 5, "running": 5, "licking": 5}, "confusion": np.eye(3)}}}
    monkeypatch.setattr(ld, "by_animal_day", lambda: (store, [3]))
    monkeypatch.setattr(ef, "epoch_of_day", lambda an, d: "acute")
    monkeypatch.setattr(es, "_position_accuracy_by_epoch", lambda align, variant: {"pre": 0.9, "acute": 0.5})
    monkeypatch.setattr(es, "_scalar_figure", lambda out, *, name, **k: rec(name))
    monkeypatch.setattr(ef, "bar_row", lambda vals, out, *, name, **k: rec(name))
    monkeypatch.setattr(ef, "confusion_row", lambda conf, out, *, name, **k: rec(name))
    with tst.active("working", "cue"):
        with g._trial_set("13s", "cue", "working"):
            es._fig_13_state(tmp_path, "cue", "working", "x")
    assert got["epoch_13n_state_vs_position_cue"] == ek.state_vs_position_text()
    for name in ("epoch_13_state_decoder_cue", "epoch_13c_state_confusion_cue",
                 "epoch_13pos_state_decoder_by_class_cue"):
        assert got[name] == ek.state_segment_text(), name
