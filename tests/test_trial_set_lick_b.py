"""Trial-set captions of the lick scripts, part B (Priya 11-09: "EVERY figure must state EXACTLY which trials were
included / excluded"): cue-to-lick, cued first licks / ramps, the ENL pre-cue window family, ENLv licks, ITI filters,
bout starts. Each driver declares its sentence in main() (checked by stopping main at the declaration), the sentences
are built from the drivers' own constants, and representative figures saved through the drivers' real save paths carry
the sentence (gid "trial_set"). Synthetic / monkeypatched only: nothing reads real data or writes outside tmp_path."""
from __future__ import annotations

import importlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from wfield_local import trial_set_text as tst  # noqa: E402

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("WFIELD_ITI_FILTER", raising=False)
    monkeypatch.delenv("WFIELD_RECSEARCH_SET", raising=False)
    yield


class _Stop(Exception):
    pass


def _declared(monkeypatch, mod, call):
    """Run ``call`` (a driver's main) until its `tst.declare`; return the declared text."""
    got = []
    real = tst.declare

    def rec(variant=None, align=None, *, text=None):
        real(variant, align, text=text)
        got.append(tst.current_sentence())
        raise _Stop

    monkeypatch.setattr(tst, "declare", rec)
    with pytest.raises(_Stop):
        call()
    assert len(got) == 1
    return got[0]


def _stamp_texts(fig):
    return [t.get_text() for t in fig.texts if t.get_gid() == tst.GID]


def _unwrap(txt: str) -> str:
    return " ".join(txt.split())


# ------------------------------------------------------------------------------------------------ main() declares
# (module, argv, expected sentence from the module's own helper, a phrase the sentence must hold)
DRIVERS = [
    ("lick_cue_to_lick", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "latency 0.05-3.6 s"),
    ("lick_cue_to_lick_precue", ["x", "--plot-only"], lambda m: m.trial_set_sentence("restw"), "any first-lick phase"),
    ("lick_cued_aim_by_spout", ["x"], lambda m: m.trial_set_sentence(), "FIRST post-cue lick of each trial"),
    ("lick_cued_ramp", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "ITI bout-start licks"),
    ("lick_enl_angle_decode", ["x", "--plot-only"], lambda m: m.trial_set_sentence(1.0, m.LF_PRIMARY),
     "sliding lick-free 1 s window"),
    ("lick_enl_offset_explore", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "sliding lick-free 1 s"),
    ("lick_enl_target_test", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "attempt classes pre_hit"),
    ("lick_enlv_balance_offset", ["x", "--plot-only"], lambda m: m.variant_sentence(m.PRIMARY), "[on1: PRIMARY"),
    ("lick_enlv_curves", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "[enlv_a]"),
    ("lick_enlv_target_code", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "[on1_clean"),
    ("lick_iti_dock_compare", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "[dock: DOCKED"),
    ("lick_iti_filter_compare", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "decoder refit on the set"),
    ("lick_iti_prelick", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "FIRST lick after >= 2 s quiet"),
    ("lick_iti_ramp_animals", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "no lick onset in the preceding 2 s"),
    ("lick_iti_recovery_search", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "[ITI first after quiet]"),
    ("lick_lateral_ratio", ["x", "--plot-only"], lambda m: m.trial_set_sentence(), "previous-trial spout cells"),
]


@pytest.mark.parametrize("name, argv, expect, phrase", DRIVERS, ids=[d[0] for d in DRIVERS])
def test_main_declares_the_driver_sentence(camera, monkeypatch, tmp_path, name, argv, expect, phrase):
    mod = importlib.import_module(name)
    monkeypatch.setattr(sys, "argv", argv)
    for attr in ("OUT", "FIG"):                       # mains that make their output folder before declaring
        if name in ("lick_enl_target_test", "lick_enlv_balance_offset", "lick_enl_angle_decode",
                    "lick_enl_offset_explore") and hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, tmp_path / attr)
    txt = _declared(monkeypatch, mod, mod.main)
    assert txt == expect(mod)
    assert phrase in txt
    # the lick families never describe the camera working classes (their trials are the engagement gate / lick events)
    assert "camera contact + incomplete" not in txt


def test_floor_qc_declares_its_lick_draw(camera, monkeypatch):
    import lick_floor_qc as fq
    txt = _declared(monkeypatch, fq, lambda: fq.main(["--floor", "3", "--per-sheet", "12", "--seed", "4"]))
    assert txt == fq.trial_set_sentence(3.0, 12, 4)
    assert "min_peak_y_abs = 3 px" in txt and "seed 4" in txt and f"{fq.CTRL_BAND[0]:g}-{fq.CTRL_BAND[1]:g} px" in txt


# ------------------------------------------------------------------------------------------------ the words
def test_engaged_gate_wording_follows_the_switch(camera, monkeypatch):
    import lick_cue_to_lick_extract as cx

    from wfield_local import orofacial_cohort as oc
    assert cx.engaged_text().startswith("engaged trials (behavior pipeline reference-position gate")
    assert "not the camera classes" in cx.engaged_text()
    monkeypatch.setattr(oc, "ENGAGEMENT_GATE", "attempted")
    assert cx.engaged_text() == "attempted trials (hit + incomplete, trial_attempts.csv)"
    monkeypatch.setattr(oc, "ENGAGED_ONLY", False)
    assert cx.engaged_text() == "all trials (no engagement gate)"


def test_cue_to_lick_stems_narrow_the_set(camera):
    import lick_cue_to_lick as c2l
    base = c2l.trial_set_sentence()
    c03 = c2l.trial_set_sentence("lick_cue_to_lick_C_w03_contra")
    assert "latency >= 0.3 s" in c03 and "latency >= 0.3 s" not in base
    assert "latency >= 0.5 s" in c2l.trial_set_sentence("lick_cue_to_lick_C_w05_ipsi")
    b = c2l.trial_set_sentence("lick_cue_to_lick_B_pre_models")
    assert f"not yet licked by t + {c2l.LAG_EXCL:g} s" in b
    assert "cue-locked template" in c2l.trial_set_sentence("lick_cue_to_lick_threshold_simple")


def test_iti_filter_reaches_the_bout_start_sentence(camera, monkeypatch):
    import lick_cued_ramp as cr
    assert "ITI filter" not in cr.iti_bout_sentence()
    monkeypatch.setattr(cr.il, "ITI_SUBSET", "dock")
    assert "ITI filter dock: onset while the spout is DOCKED" in cr.iti_bout_sentence()


def test_enl_lickfree_rule_words(camera):
    import lick_enl_angle_decode as ead
    assert "sliding lick-free 0.5 s window" in ead.trial_set_sentence(0.5, "slide")
    fixed = ead.trial_set_sentence(1.0, "2s")
    assert f"no licks_all onset (any context) in [cue - {ead.LF_LONG:g} s, cue)" in fixed


def test_enlv_variant_rules(camera):
    import lick_enlv_balance_offset as bo
    assert "bout starts only (clean_start" in bo.variant_sentence(bo.BOUTS)
    assert "no cue rule" in " ".join(bo.variant_keep("on1_nocuerule"))
    assert "non-contact licks only" in " ".join(bo.variant_keep("on1_noncontact"))
    sw = bo.figure_sentence("lick_enlv_sweep_simple")
    assert all(f"[{v}]" in sw for v in bo.SWEEP_SIMPLE_V)
    cmp_ = bo.figure_sentence("lick_enlv_compare")
    assert "ITI filter dock" in cmp_ and "ITI filter notdock" in cmp_


def test_recovery_search_cued_mode(camera, monkeypatch):
    import lick_iti_recovery_search as rs
    monkeypatch.setattr(rs, "CUED", True)
    txt = rs.trial_set_sentence()
    assert txt.startswith("events: post-cue licks") and "lick order 0" in txt


# ------------------------------------------------------------------------------------------------ real save paths
def test_filter_compare_figure_carries_the_sentence(camera, monkeypatch, tmp_path):
    import lick_iti_filter_compare as fc
    monkeypatch.setattr(fc.il, "OUT", tmp_path)
    (tmp_path / "figures").mkdir()
    rows = [dict(set=lab, epoch=ep, point=0.1, lo=0.0, hi=0.2, n_dir=3, n=4, n_licks=10, **{a: 0.1 for a in fc.AN})
            for _t, lab in fc.SETS for ep in fc.EP]
    tst.declare(text=fc.trial_set_sentence())
    figs = []
    real = plt.Figure.savefig

    def keep(self, *a, **k):
        figs.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", keep)
    fc.plot({"R": pd.DataFrame(rows)})
    assert (tmp_path / "figures" / "lick_iti_filter_compare.png").exists()
    assert [_unwrap(t) for t in _stamp_texts(figs[0])] == [_unwrap(fc.trial_set_sentence())]


def test_cued_ramp_baseline_sens_is_narrowed(camera, monkeypatch, tmp_path):
    import lick_cued_ramp as cr
    monkeypatch.setattr(cr.il, "OUT_W", tmp_path)
    (tmp_path / "figures").mkdir()
    tst.declare(text=cr.trial_set_sentence())
    figs = []
    real = plt.Figure.savefig

    def keep(self, *a, **k):
        figs.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", keep)
    T = pd.DataFrame(columns=["set", "baseline", "roi", "value", "epoch", "delta", "lo", "hi", "n_dir", "n"])
    cr.draw_baseline_sens({"T": T})
    got = [_unwrap(t) for t in _stamp_texts(figs[0])]
    assert got == [_unwrap(cr.trial_set_sentence("sens"))]
    assert "ISOLATED subset" in got[0]
    assert tst.current_sentence() == cr.trial_set_sentence()          # the narrowing closed after the save


@pytest.mark.parametrize("name, key, which", [
    ("lick_cued_ramp", "lick_iti_ramp_perspout.png", "iti_spout"),
    ("lick_cued_ramp", "lick_cue_perspout_windows.png", "cued_spout"),
    ("lick_cued_ramp", "lick_ramp_windows_cued.png", "cued"),
    ("lick_cued_ramp", "lick_ramp_ratio.png", "all"),
    ("lick_lateral_ratio", "lick_lateral_ratio_traces_iti.png", "iti"),
    ("lick_lateral_ratio", "lick_lateral_ratio_side_traces_cue.png", "cue"),
    ("lick_lateral_ratio", "lick_lateral_ratio_auc.png", None),
])
def test_per_figure_narrowing_by_name(camera, tmp_path, name, key, which):
    mod = importlib.import_module(name)
    tst.declare(text=mod.trial_set_sentence())
    fig = plt.figure(figsize=(6, 3))
    with mod._trial_set_for(tmp_path / key):
        fig.savefig(tmp_path / key)
    plt.close(fig)
    assert [_unwrap(t) for t in _stamp_texts(fig)] == [_unwrap(mod.trial_set_sentence(which))]


def test_narrowing_is_a_noop_without_a_declaration(camera, tmp_path):
    import lick_cue_to_lick as c2l
    fig = plt.figure(figsize=(4, 2))
    with c2l._trial_set_for("lick_cue_to_lick_C_w03_contra"):
        fig.savefig(tmp_path / "a.png")
    plt.close(fig)
    assert _stamp_texts(fig) == []


@pytest.mark.parametrize("name, part, expect", [
    ("lick_cue_to_lick", "lick_cue_to_lick_C_w05_contra", lambda m: m.trial_set_sentence("lick_cue_to_lick_C_w05_contra")),
    ("lick_enl_target_test", "A", lambda m: m.trial_set_sentence("A")),
    ("lick_enl_target_test", "C", lambda m: m.trial_set_sentence("C")),
    ("lick_enlv_balance_offset", "lick_enlv_arrival", lambda m: m.figure_sentence("lick_enlv_arrival")),
])
def test_driver_narrowing_on_save(camera, tmp_path, name, part, expect):
    mod = importlib.import_module(name)
    tst.declare(text="events: the process declaration")
    fig = plt.figure(figsize=(6, 3))
    with mod._trial_set_for(part):
        fig.savefig(tmp_path / "f.png")
    plt.close(fig)
    assert [_unwrap(t) for t in _stamp_texts(fig)] == [_unwrap(expect(mod))]


def test_enlv_curves_combined_figure_names_six_sets(camera):
    import lick_enlv_curves as ec
    txt = ec.trial_set_sentence(ec.ALL_PANELS)
    assert txt.count("  AND  ") == 5
    assert "FIRST post-cue lick of each trial" in txt and "FIRST lick after >= 2 s quiet" in txt
    four = ec.trial_set_sentence()
    assert four.count("  AND  ") == 3


def test_sentences_use_the_driver_constants(camera, monkeypatch):
    import lick_cued_ramp as cr
    import lick_enlv_curves as ec
    monkeypatch.setattr(cr.il, "GAP_MAIN", 3.0)
    assert "no lick onset in the preceding 3 s" in cr.iti_bout_sentence()
    monkeypatch.setattr(ec, "ENLV_A_ONSET", 1.5)
    assert "onset >= 1.5 s after arrival" in ec.set_sentence("enlv_a")
    assert np.isclose(ec.ENLV_A_CUE_END, ec.bo.VARIANTS[ec.bo.PRIMARY][3])
