"""Trial-set captions of the lick drivers, part C (Priya 11-09: "EVERY figure must state EXACTLY which trials were
included / excluded"). Each driver declares its lick / trial / frame set in main() (`trial_set_text.declare`), narrowed
per figure where its figures draw different sets; the sentence is built from the driver's own constants.

Synthetic / monkeypatched only: no driver reads real data or writes outside tmp_path."""
from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import trial_set_text as tst  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("WFIELD_ITI_FILTER", raising=False)
    yield


def _flat(s: str) -> str:
    return " ".join(str(s).split())


def _hits(fig) -> list[str]:
    return [_flat(t.get_text()) for t in fig.texts if t.get_gid() == tst.GID]


class _Sentinel(Exception):
    pass


def _raise(*_a, **_k):
    raise _Sentinel


# ------------------------------------------------------------------------------------------------ the sentences
#: driver -> phrases its sentence must carry (its own set rules, from its own constants)
MUST = {
    "lick_matched_angle_reach": ["ITI licks", "post-cue licks", ">= 20 such licks per session",
                                 "GAUSSIAN-reference panels"],
    "lick_matched_pixel_maps": ["FIRST post-cue lick", "first lick after >= 2 s quiet", "ITI filter far1",
                                "2.5-97.5 % range", ">= 10 kept licks"],
    "lick_matched_readout": ["ITI licks", "at most 1500 per session", "spout known", ">= 20 such licks per session"],
    "lick_nonlinear_decoder": [">= 40 such licks per session", "every 3-D coordinate finite"],
    "lick_offset_contrib_map": ["Gaussian weight > 1e-6 (sigma 0.1 mm)"],
    "lick_offset_terms_recovery": ["recovery axes: far-contra FIRST post-cue lick aim"],
    "lick_offset_where": ["sigma 0.1 mm", "post-cue licks"],
    "lick_path_softmax": ["pose3d_kin_licks_snapshot_20261023", ">= 8 tracked points"],
    "lick_position_per_spout": ["session x spout cells with >= 5 licks"],
    "lick_precue_restw": ["trials: engaged cued trials", "no lick-free requirement", "RESTW rest frames",
                          "restdock05", ">= 200 frames each", ">= 10 such trials"],
    "lick_precue_restw_traces": ["trials: engaged cued trials", "RESTW rest frames"],
    "lick_recovery_markers": ["ITI first after quiet", "cued first lick", "response-AUC", "recovery axes"],
    "lick_recovery_primary": ["FIRST post-cue lick", "preceding 2 s", "recovery axes"],
    "lick_refit_decoders": [">= 40 such licks per session", "every 3-D coordinate finite"],
    "lick_rim_overlay": ["ITI bout-start licks", "no lick onset in the preceding 2 s", "pre-stroke sessions only"],
    "lick_roi_candidates": ["pre-stroke sessions only (PRE level maps)"],
    "lick_roi_check": ["pre-stroke sessions only (PRE level maps)"],
    "lick_roi_foci": ["pre-stroke sessions only (PRE level maps)"],
    "lick_same_target_place": ["same-target rows"],
    "lick_session_level_maps": ["FIRST subset", ">= 40 licks per session map", "fit share >= 0.05"],
    "lick_session_levels_scale_compare": ["FIRST subset", ">= 40 licks per session map"],
    "lick_session_pixel_lmm": ["FIRST subset", "aim deficit"],
    "lick_sspn_checks": ["recovery axes"],
    "lick_trajectories_3d": ["up to 5 evenly spaced chronic"],
    "lick_vigor_perspout": ["FIRST lick of each engaged cued trial", "ITI bout-start licks", "within 30 ms"],
}


@pytest.mark.parametrize("name", sorted(MUST))
def test_driver_sentence(camera, name):
    M = importlib.import_module(name)
    s = M.trial_set_sentence()
    assert s.startswith(("events: ", "trials: ")), s
    for p in MUST[name]:
        assert p in s, f"{name}: {p!r} missing from {s!r}"
    if name != "lick_matched_pixel_maps":                 # no ITI filter in the run -> none claimed
        assert "ITI filter" not in s


def test_template_match_sentence(camera):
    import lick_template_match as M
    s = M.trial_set_sentence(["PS93:20260814"], ["PS93:20260821"], ["far_R"], 20)
    assert "cued to far_R in PS93:20260821" in s and "reach >= 20 px" in s and "PS93:20260814" in s


def test_iti_filter_follows_what_the_driver_applies(camera, monkeypatch):
    import iti_lick_order as lo
    import lick_matched_readout as mr
    import lick_recovery_markers as rm
    import lick_refit_decoders as rd
    monkeypatch.setenv("WFIELD_ITI_FILTER", "dock")       # a SUBSET filter: matched_readout applies it ...
    assert "ITI filter dock" in mr.trial_set_sentence()
    assert "ITI filter" not in rd.trial_set_sentence()    # ... the where-decoder never does
    monkeypatch.setattr(lo, "ITI_FILTER", "gap2")         # a REFIT filter: the KEY cache readers carry it
    assert "ITI filter gap2" in rm.trial_set_sentence()


def test_per_set_sentences_differ(camera):
    import lick_matched_pixel_maps as pm
    import lick_offset_contrib_map as oc
    import lick_precue_restw as pr
    import lick_session_level_maps as lv
    assert pm.trial_set_sentence("trial").startswith("events: FIRST post-cue lick")
    assert "ITI filter far1" in pm.trial_set_sentence("iti_far1") and "far1" not in pm.trial_set_sentence("iti")
    assert "ITI licks" not in oc.trial_set_sentence("trial").split("excluded")[0]
    assert "raw kernel weight >= 3" in oc.trial_set_sentence(coord="polar")
    far = pr.trial_set_sentence("far_contra", "restnl")
    assert "far-contra target only" in far and ">= 5 such trials" in far and "RESTnl" in far
    assert lv.trial_set_sentence("sp_iti").startswith("events: ITI licks") and "post-cue licks (" not in \
        lv.trial_set_sentence("sp_iti")


# ------------------------------------------------------------------------------------------------ the declaration
#: driver -> (functions stubbed to save a figure, where its plot-only switch / figure data come from)
REPLOT = {
    "lick_refit_decoders": ["plot"],
    "lick_matched_angle_reach": ["plot"],
    "lick_position_per_spout": ["plot", "plot_by_epoch"],
    "lick_nonlinear_decoder": ["plot"],
    "lick_offset_where": ["plot"],
    "lick_recovery_markers": ["plot"],
    "lick_recovery_primary": ["plot"],
    "lick_vigor_perspout": ["plot"],
    "lick_same_target_place": ["draw"],
    "lick_matched_readout": ["plot", "plot_animal"],
    "lick_rim_overlay": ["plot"],
    "lick_matched_pixel_maps": ["plot"],
    "lick_offset_contrib_map": ["plot"],
    "lick_session_levels_scale_compare": ["plot_maps", "plot_global"],
}


@pytest.mark.parametrize("name", sorted(REPLOT))
def test_plot_only_replot_carries_the_sentence(camera, monkeypatch, tmp_path, name):
    """The --plot-only / figdata path declares the driver's set: every figure it saves carries the sentence."""
    import figdata
    M = importlib.import_module(name)
    fd = defaultdict(lambda: None)
    for mod in (M, figdata):
        if hasattr(mod, "plot_only"):
            monkeypatch.setattr(mod, "plot_only", lambda: True)
        if hasattr(mod, "load_fd"):
            monkeypatch.setattr(mod, "load_fd", lambda *a, **k: fd)
    if hasattr(M, "FIG"):
        monkeypatch.setattr(M, "FIG", tmp_path / "figures")
    if hasattr(M, "_runs"):
        monkeypatch.setattr(M, "_runs", lambda: {})
    monkeypatch.setattr(sys, "argv", [name, "--plot-only"])
    saved = []

    def stub(fn):
        def f(*_a, **_k):
            fig = plt.figure(figsize=(4, 3))
            fig.savefig(tmp_path / f"{fn}.png")
            saved.append((fn, fig, tst.current_sentence()))
        return f
    for fn in REPLOT[name]:
        monkeypatch.setattr(M, fn, stub(fn))
    assert M.main() == 0
    assert saved
    for fn, fig, cur in saved:
        assert _hits(fig) == [_flat(cur)], fn
        plt.close(fig)
    want = _flat(M.trial_set_sentence())
    if name == "lick_session_levels_scale_compare":       # the map figures narrow it per family
        assert _flat(saved[-1][2]) == want                # plot_global: every family
        assert _flat(saved[0][2]) == _flat(M.trial_set_sentence("lat"))
    else:
        assert all(_flat(c) == want for _f, _g, c in saved)


def test_session_level_maps_narrow_per_family(camera, monkeypatch, tmp_path):
    import lick_session_level_maps as lv
    monkeypatch.setattr(lv, "plot_only", lambda: True)
    monkeypatch.setattr(lv, "load_fd", lambda *a, **k: {})
    seen = []

    def stub(D):
        fig = plt.figure(figsize=(4, 3))
        fig.savefig(tmp_path / "x.png")
        seen.append((_hits(fig)[0], tst.current_sentence()))
        plt.close(fig)
    monkeypatch.setattr(lv, "plot_levels", stub)
    monkeypatch.setattr(lv, "plot_change", stub)
    assert lv.main() == 0
    assert len(seen) == 2 * len(lv.FAMS)
    for k, fam in enumerate(lv.FAMS):
        for hit, cur in seen[2 * k: 2 * k + 2]:
            assert hit == _flat(lv.trial_set_sentence(fam)) == _flat(cur)
    assert _flat(tst.current_sentence()) == _flat(lv.trial_set_sentence())     # back to the declaration


@pytest.mark.parametrize("name,var", [("lick_precue_restw", "restw"), ("lick_precue_restw", "restnl"),
                                      ("lick_precue_restw_traces", "restw")])
def test_precue_draw_activates_its_figure_set(camera, monkeypatch, name, var):
    """draw(stem) narrows the declaration to the stem's spout set and rest variant before drawing."""
    M = importlib.import_module(name)
    fd = {"sfx": "_restnl" if var == "restnl" else "", "tag": "_farcontra"}
    monkeypatch.setattr(M, "load_fd", lambda *a, **k: fd)
    tst.declare(text="declared")
    stem = "lick_cued_precue_restw_maps_farcontra" + fd["sfx"]
    with pytest.raises(KeyError):                         # the synthetic figure data stops it after the activation
        M.draw(stem)
    assert _flat(tst.current_sentence()) == _flat(M.trial_set_sentence("far_contra", var))


@pytest.mark.parametrize("name", ["lick_precue_restw", "lick_precue_restw_traces"])
def test_precue_main_declares(camera, monkeypatch, name):
    M = importlib.import_module(name)
    monkeypatch.setattr(M, "plot_only", lambda: True)
    monkeypatch.setattr(M, "draw", lambda stem: None)
    monkeypatch.setattr(sys, "argv", [name, "--plot-only"])
    assert M.main() == 0
    assert _flat(tst.current_sentence()) == _flat(M.trial_set_sentence(None, "restw"))


def _declared_before_load(M, monkeypatch, argv, call):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_Sentinel):
        call()
    return tst.current_sentence()


@pytest.mark.parametrize("name", ["lick_roi_candidates", "lick_roi_check", "lick_roi_foci"])
def test_roi_drivers_declare_before_the_data(camera, monkeypatch, name):
    import iti_licks as il
    M = importlib.import_module(name)
    monkeypatch.setattr(il, "labels", _raise)
    cur = _declared_before_load(M, monkeypatch, [name], M.main)
    assert _flat(cur) == _flat(M.trial_set_sentence())


@pytest.mark.parametrize("name", ["lick_trajectories_3d", "lick_path_softmax", "lick_offset_terms_recovery"])
def test_csv_drivers_declare_before_the_data(camera, monkeypatch, name):
    from wfield_local import run_mode
    M = importlib.import_module(name)
    monkeypatch.setattr(run_mode, "require_default", lambda what: None)
    monkeypatch.setattr(M.pd, "read_csv", _raise)
    cur = _declared_before_load(M, monkeypatch, [name], M.main)
    assert _flat(cur) == _flat(M.trial_set_sentence())


def test_template_match_declares_and_narrows(camera, monkeypatch, tmp_path):
    import lick_template_match as M

    from wfield_local import run_mode
    monkeypatch.setattr(run_mode, "session_poses_out", lambda script, default: tmp_path)
    monkeypatch.setattr(M, "load_session", _raise)
    argv = ["x", "--pre", "PS93:20260814", "--post", "PS93:20260821", "--cued", "far_R"]
    cur = _declared_before_load(M, monkeypatch, argv, lambda: M.main(argv[1:]))
    assert "cued to far_R in PS93:20260821" in cur


# ------------------------------------------------------------------------------------------------ real figure functions
def test_position_per_spout_real_figures_carry_it(camera, monkeypatch, tmp_path):
    """The driver's real plot functions (save_png_svg path) on synthetic per-animal means."""
    import iti_licks as il
    import iti_lick_order as lo
    import lick_position_per_spout as M
    rng = np.random.default_rng(0)
    rows = [dict(set=st, animal=a, epoch=ep, pos=p, act=rng.normal(), dec=rng.normal())
            for st in M.SETS for a in ("PS92", "PS93") for ep in M.EP for p in lo.POS]
    monkeypatch.setattr(M, "plot_only", lambda: True)
    monkeypatch.setattr(M, "load_fd", lambda *a, **k: {"A": pd.DataFrame(rows)})
    monkeypatch.setattr(il, "OUT_W", tmp_path)
    (tmp_path / "figures").mkdir()
    closed = []
    monkeypatch.setattr(plt, "close", lambda fig=None: closed.append(fig))
    assert M.main() == 0
    assert len(closed) == 2 and (tmp_path / "figures" / "lick_position_per_spout.png").exists()
    for fig in closed:
        assert _hits(fig) == [_flat(M.trial_set_sentence())]


def test_sspn_checks_real_figure_plain_savefig(camera, monkeypatch, tmp_path):
    """A plain fig.savefig writer (lick_sspn_checks.plot) through main, on synthetic check tables."""
    import iti_licks as il
    import lick_sspn_checks as M
    rows = [dict(check=c, set=st, y=yv, measure=m, p_within=0.5, **{f"r_{a}": 0.1 for a in M.AN})
            for c in ("1", "2", "3") for st, yv in M.TARGETS for m in ("SSp-n", "ALM")]
    A = pd.DataFrame(rows)
    for k in ("check1", "check2", "check3"):
        monkeypatch.setattr(M, k, lambda c=k[-1]: A[A.check == c].reset_index(drop=True))
    monkeypatch.setattr(il, "OUT_W", tmp_path)
    (tmp_path / "figures").mkdir()
    closed = []
    monkeypatch.setattr(plt, "close", lambda fig=None: closed.append(fig))
    assert M.main() == 0
    assert (tmp_path / "figures" / "lick_sspn_checks.png").exists()
    assert _hits(closed[0]) == [_flat(M.trial_set_sentence())]


def test_no_declaration_at_import(camera):
    for name in [*MUST, "lick_template_match"]:
        importlib.import_module(name)
    assert tst.current() is None
