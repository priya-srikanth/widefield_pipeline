"""Every ITI-lick figure states its LICK set (Priya 11-09), part A of the ITI scripts.

Pins, under the production camera env: the sentence each driver builds from its own constants (iti_licks GAP_MAIN /
CUE_EXCL / N_TRIAL_MAX, the ITI filters, the per-session minimums), that each driver DECLARES it on its command-line path
(the --plot-only replot included) before any figure is drawn, and that figures saved through the drivers' real figure
functions carry it (gid "trial_set"): the per-figure-group captions of iti_licks (ramp / direction / frozen / first
licks; none on the ROI placement; none when an importer draws without `main`), the per-stem sets of iti_check_figures,
a plain ``fig.savefig`` driver (iti_decoder_contrib, iti_extension_dist), the shared save path (iti_first_lick) and the
F0 frame set of iti_hemo_check. Synthetic figure data only; every write goes to tmp_path.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from wfield_local import trial_set_text as tst  # noqa: E402

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("WFIELD_ITI_FILTER", raising=False)
    yield


@pytest.fixture
def il(monkeypatch, tmp_path):
    import iti_licks
    monkeypatch.delenv(iti_licks.TS_ENV, raising=False)             # restored after the test (main switches it on)
    monkeypatch.setattr(iti_licks, "OUT", tmp_path)
    monkeypatch.setattr(iti_licks, "OUT_W", tmp_path)
    (tmp_path / "figures").mkdir()
    return iti_licks


class _Stop(Exception):
    pass


def _stamps(fig) -> list[str]:
    return [t.get_text() for t in fig.texts if t.get_gid() == tst.GID]


def _flat(s: str) -> str:
    return " ".join(s.split())


def _record(monkeypatch, mod, name="save_png_svg"):
    """Wrap ``mod.<name>`` (a save function taking the figure first): the real save runs, then the figure's trial-set
    stamps are recorded per file name."""
    real = getattr(mod, name)
    got = {}

    def rec(fig, path, *a, **k):
        out = real(fig, path, *a, **k)
        got[getattr(path, "name", str(path))] = [_flat(t) for t in _stamps(fig)]
        return out
    monkeypatch.setattr(mod, name, rec)
    return got


def _capture(box: list):
    """A stand-in for a driver's first data / draw step: records the active sentence, then stops the run."""
    def f(*_a, **_k):
        box.append(tst.current_sentence())
        raise _Stop
    return f


# ------------------------------------------------------------------------------------------------ the sentences
def test_bout_start_sentence_from_the_constants(camera, il):
    s = il.trial_set_bouts(il.GAP_MAIN)
    a, b = il.CUE_EXCL
    assert s.startswith("events: ITI bout-start licks (3-D whole-session lick table, context iti")
    assert f"no lick onset in the preceding {il.GAP_MAIN:g} s" in s
    assert f"within [{a:+g}, {b:+g}] s of the onset" in s and "sessions with >= 10 such licks" in s
    assert s.endswith("excluded: later licks of a bout, post-cue (trial) licks")
    assert "ITI filter" not in s                                          # iti_licks applies no filter
    assert "ISOLATED" in il.trial_set_sentence("ramp_isolated") and "ISOLATED" not in il.trial_set_sentence("ramp")


def test_iti_licks_groups(camera, il):
    d = il.trial_set_sentence("direction")
    assert "  AND  " in d and "per-session decoder: sessions with >= 50 ITI licks" in d
    assert f"a random {il.N_TRIAL_MAX} per session" in d and "drawn to the session's ITI-lick count" in d
    f = il.trial_set_sentence("frozen")
    assert "drawn to the session's ITI-lick count" not in f and f.count("sessions with >= 20 such licks") == 2
    assert il.trial_set_sentence("first").startswith("events: trial FIRST licks")
    assert "engaged trials only (orofacial engagement gate" in il.trial_set_sentence("first")
    assert il.trial_set_sentence("rois") is None
    with pytest.raises(ValueError):
        il.trial_set_sentence("nope")


def test_lick_order_states_the_refit_or_subset_filter(camera, monkeypatch, il):
    import iti_lick_order as lo
    assert "ITI filter" not in lo.trial_set_sentence()
    monkeypatch.setattr(lo, "ITI_FILTER", "gap2far1")
    s = lo.trial_set_sentence()
    assert "ITI filter gap2far1: " + tst.ITI_FILTER_TEXT["gap2far1"] in s
    assert s.split("  AND  ")[1].count("ITI filter") == 0                 # the cued half carries no ITI filter
    monkeypatch.setattr(lo, "ITI_FILTER", "")
    monkeypatch.setattr(il, "ITI_SUBSET", "dock")
    assert "ITI filter dock: " + tst.ITI_FILTER_TEXT["dock"] in lo.trial_set_sentence()
    assert f"gaps >= {il.BOUT_GAP:g} s" in lo.trial_set_sentence()
    assert "previous trial's spout known" not in lo.trial_set_sentence().split("  AND  ")[1]


def test_area_maps_sets(camera, monkeypatch, il):
    import iti_area_maps as am
    assert am.trial_set_sentence() == il.trial_set_bouts(il.GAP_MAIN)
    assert "previous trial at far_contra only; sessions with >= 5" in am.trial_set_sentence(True, False, "far_contra")
    cu = am.trial_set_sentence(True, True, "far_contra")
    assert cu.startswith("events: trial FIRST licks") and "target spout far_contra only" in cu
    monkeypatch.setattr(il, "ITI_SUBSET", "notdock")
    assert "ITI filter notdock" in am.trial_set_sentence()


def test_lick_rate_is_the_camera_working_period(camera):
    import iti_lick_rate as lr
    s = lr.trial_set_sentence()
    assert s.startswith("events: ITI licks (cam4 licks_all.csv, context iti")
    assert "camera working period (first cue to the last camera lick, stopped runs (A2 + lick excise) removed)" in s
    assert f"clipped to 0..{lr.GAP_MAX:g} s" in s and "sessions without camera classes" in s


def test_check_figures_per_stem(camera, il):
    import iti_check_figures as cf
    assert cf.trial_set_sentence("lick_iti_check_f0_sessions") == il.trial_set_bouts(il.GAP_MAIN)
    assert cf.trial_set_sentence("lick_iti_check_error_per_lick") == il.trial_set_iti()
    assert cf.trial_set_sentence("lick_cued_check_rmse_sessions") == il.trial_set_cued()
    assert cf.trial_set_sentence("lick_check_rmse") == tst.union(il.trial_set_iti(), il.trial_set_cued())
    for stem in cf.STEMS:                                                 # every figure of the script has a set
        assert cf.trial_set_sentence(stem).startswith("events: ")


@pytest.mark.parametrize("mod", ["iti_basis_decoder", "iti_decoded_decomposition", "iti_decoder_contrib",
                                 "iti_decoder_patterns", "iti_extension_dist", "iti_first_lick", "iti_hemo_check",
                                 "iti_lick_order"])
def test_every_driver_sentence_is_an_event_set(camera, mod, il):
    import importlib
    s = importlib.import_module(mod).trial_set_sentence()
    assert s.startswith("events: ") and "excluded: " in s


# ------------------------------------------------------------------------------------------------ declared on the CLI path
def _plot_only_argv(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["x", "--plot-only", *extra])


@pytest.mark.parametrize("mod, patch, call", [
    ("iti_first_lick", "plot", lambda m: m.main()),
    ("iti_decoder_contrib", "plot", lambda m: m.main()),
    ("iti_decoder_patterns", "plot", lambda m: m.main()),
    ("iti_extension_dist", "draw", lambda m: m.main()),
    ("iti_lick_rate", "plot", lambda m: m.cli()),
])
def test_plot_only_path_declares(camera, monkeypatch, il, mod, patch, call):
    import importlib
    m = importlib.import_module(mod)
    box = []
    _plot_only_argv(monkeypatch)
    monkeypatch.setattr(m, "load_fd", lambda *_a, **_k: {"T": None})
    monkeypatch.setattr(m, patch, _capture(box))
    with pytest.raises(_Stop):
        call(m)
    assert box == [m.trial_set_sentence()]


def test_pattern_stats_declares_the_pattern_set(camera, monkeypatch, il):
    import iti_decoder_pattern_stats as ps
    import iti_decoder_patterns as dp
    box = []
    _plot_only_argv(monkeypatch)
    monkeypatch.setattr(ps, "load_fd", lambda *_a, **_k: {})
    monkeypatch.setattr(ps, "plot_all", _capture(box))
    with pytest.raises(_Stop):
        ps.main()
    assert box == [dp.trial_set_sentence()]


def test_basis_decoder_and_hemo_check_declare(camera, monkeypatch, il):
    import iti_basis_decoder as bd
    import iti_hemo_check as hc

    from wfield_local import writeguard
    box = []
    monkeypatch.setattr(bd, "load_fd", lambda *_a, **_k: {})
    monkeypatch.setattr(bd, "plot", _capture(box))
    with pytest.raises(_Stop):
        bd.main(["--plot-only"])
    monkeypatch.setattr(writeguard, "assert_writable", _capture(box))    # hemo main's first step after the declaration
    with pytest.raises(_Stop):
        hc.main(["--plot-only"])
    assert box == [bd.trial_set_sentence(), hc.trial_set_sentence()]


def test_lick_order_and_decomposition_declare_before_the_cache(camera, monkeypatch, il):
    import iti_decoded_decomposition as dd
    import iti_lick_order as lo
    box = []
    monkeypatch.setattr(lo, "preds_path", _capture(box))
    with pytest.raises(_Stop):
        lo.main()
    with pytest.raises(_Stop):
        dd.main()
    assert box == [lo.trial_set_sentence(), dd.trial_set_sentence()]
    assert "spouts with >= 8 licks" in box[1]


@pytest.mark.parametrize("argv, fn, args", [
    ([], "draw_all", ()),
    (["--compare"], "draw_all", ()),
    (["--split", "--cued", "--spout", "far_contra"], "draw_maps", (True, True, "far_contra")),
    (["--split", "--spout", "far_ipsi"], "draw_maps", (True, False, "far_ipsi")),
])
def test_area_maps_cli_declares(camera, monkeypatch, il, argv, fn, args):
    import iti_area_maps as am
    box = []
    _plot_only_argv(monkeypatch, *argv)
    monkeypatch.setattr(am, fn, _capture(box))
    with pytest.raises(_Stop):
        am.cli()
    assert box == [am.trial_set_sentence(*args)]


# ------------------------------------------------------------------------------------------------ on the saved figures
def _direction_fd():
    x = np.round(np.arange(-0.5, 0.5001, 0.05), 3)
    cur = [("pre", x, np.zeros_like(x), x * 0 - 0.1, x * 0 + 0.1, 3)]
    return {"panels": {(tg, s): {"sig": {"rows": [], "recs": []}, "curves": cur}
                       for tg in ("direction", "relative") for s in ("iti", "trial")},
            "frozen": {"direction": None, "relative": None}}


def _ramp_fd(il):
    lags = il.LAGS_P
    return {"lags": lags, "what": "w", "panels": [{"roi": r, "curves": [("pre", lags * 0, lags * 0 + 0.1, 3)],
                                                  "sig": None} for r in ("ALM_right", "ALM_left")]}


def test_iti_licks_figures_carry_their_group_set(camera, monkeypatch, il):
    import figdata
    got = _record(monkeypatch, il)
    fds = {il.tagged("lick_direction_decoding"): _direction_fd(), "lick_iti_ramp_peth": _ramp_fd(il),
           "lick_iti_ramp_peth_isolated": _ramp_fd(il),
           "lick_iti_rois": {"img": np.ones((4, 4)), "outline": np.zeros((4, 4)), "roi_frac": 0.5, "roi_frac_alm": 0.4},
           "fr_lick_variance": {"panels": [(0, 0, "t", None, {"eps": [], "stars": []})], "off": []}}
    monkeypatch.setattr(figdata, "load_fd", lambda stem, *a, **k: fds[stem])
    for fam in ("--plot-direction", "--ramp", "--rois-only"):
        assert il.main(["--plot-only", fam]) == 0
    il.controls_plot(only=("fr_lick_variance",))                          # TS_ENV is still on (main ran)
    assert got[il.tagged("lick_direction_decoding.png")] == [_flat(il.trial_set_sentence("direction"))]
    assert got["lick_iti_ramp_peth.png"] == [_flat(il.trial_set_sentence("ramp"))]
    assert got["lick_iti_ramp_peth_isolated.png"] == [_flat(il.trial_set_sentence("ramp_isolated"))]
    assert got["fr_lick_variance.png"] == [_flat(il.trial_set_sentence("first"))]
    assert got["lick_iti_rois.png"] == []                                 # ROI placement: no lick set
    assert (il.OUT / "figures" / "lick_iti_ramp_peth.png").exists()


def test_iti_licks_draw_without_main_states_nothing(camera, monkeypatch, il):
    """An importer that calls a draw function (no `main`) gets no caption from this module."""
    got = _record(monkeypatch, il)
    il.direction_draw(_direction_fd())
    assert got[il.tagged("lick_direction_decoding.png")] == []


def test_check_figures_stamp_per_stem(camera, monkeypatch, il):
    import iti_check_figures as cf
    monkeypatch.setattr(cf, "WOUT", il.OUT)
    got = _record(monkeypatch, cf)
    p = {"i": 0, "j": 0, "title": "t", "span0": -3, "x": np.arange(3.0), "y": np.arange(3.0), "label": None,
         "ref": None, "pm": None, "x2": None}
    fds = {n: dict(kind="course", name=n, nrows=1, panels=[p], suptitle="s")
           for n in ("lick_iti_check_f0_sessions", "lick_cued_check_rmse_sessions")}
    monkeypatch.setattr(cf, "STEMS", list(fds))
    monkeypatch.setattr(cf, "load_fd", lambda stem, *a, **k: fds[stem])
    _plot_only_argv(monkeypatch)
    assert cf.main() == 0
    assert got["lick_iti_check_f0_sessions.png"] == [_flat(il.trial_set_bouts(il.GAP_MAIN))]
    assert got["lick_cued_check_rmse_sessions.png"] == [_flat(il.trial_set_cued())]


def test_plain_savefig_drivers_carry_it(camera, monkeypatch, il):
    """iti_decoder_contrib / iti_extension_dist save with a plain fig.savefig: the declaration's savefig hook."""
    import matplotlib.pyplot as plt

    import iti_decoder_contrib as dc
    import iti_extension_dist as ed
    got = []
    real = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: (got.append([_flat(t) for t in _stamps(fig)]), real(fig)))
    per = pd.DataFrame({"PS92": [0.1, -0.2]}, index=["ALM_right", "tjM1_left"])
    rows = [dict(title="t", order=list(per.index),
                 eps={ep: dict(per=per, tot=pd.Series({"PS92": -0.1})) for ep in dc.EP})]
    tst.declare(text=dc.trial_set_sentence())
    dc.plot({"rows": rows})
    cells = {(st, v, a): dict(own=1.0, x={"pre": np.ones(25)}) for st, v, _l in ed.PANELS for a in ed.AN}
    tst.declare(text=ed.trial_set_sentence())
    ed.draw({"cells": cells, "cohort": {"act_sag": 1.0, "act_reach": 1.0}, "top": {"act_sag": 3.0, "act_reach": 3.0},
             "rows": []})
    assert got == [[_flat(dc.trial_set_sentence())], [_flat(ed.trial_set_sentence())]]
    assert (il.OUT / "figures" / il.tagged("lick_decoder_contrib.png")).exists()


def test_first_lick_figure_through_the_shared_save(camera, monkeypatch, il):
    import iti_first_lick as fl
    got = _record(monkeypatch, fl)
    T = pd.DataFrame([dict(scope=sc, measure=m, cls=c, epoch=ep, point=0.1, lo=0.0, hi=0.2, n_dir=3, n=4)
                      for sc in ("far contra", "all spouts") for m in ("landing", "decoded", "lfl") for c in ("first", "later")
                      for ep in fl.EP] + [dict(scope=sc, measure="first_minus_later", cls="diff", epoch=ep, point=0.1,
                                               lo=0.0, hi=0.2, n_dir=3, n=4) for sc in ("far contra", "all spouts")
                                          for ep in fl.EP])
    monkeypatch.setattr(fl, "load_fd", lambda *_a, **_k: {"T": T})
    _plot_only_argv(monkeypatch)
    assert fl.main() == 0
    assert got["lick_cued_first_lick.png"] == [_flat(fl.trial_set_sentence())]


def test_hemo_f0_map_states_its_frame_set(camera, monkeypatch, il):
    import frozen_area_maps
    import iti_hemo_check as hc
    got = _record(monkeypatch, hc)
    monkeypatch.setattr(frozen_area_maps, "paint", lambda img, nm, v: np.zeros(img.shape))
    tst.declare(text=hc.trial_set_sentence())                             # the driver's lick set ...
    pans = {(i, j): ("f0_470_rel" if i == 0 else "f0_415_rel", ep, {}, 0, 0.01)
            for i in range(2) for j, ep in enumerate(hc.POST)}
    hc.draw_f0({"img": np.zeros((4, 4)), "nm": [], "ol": np.zeros((4, 4)), "panels": pans})
    assert got["lick_iti_hemo_check_f0_areas.png"] == [_flat(hc.F0_SET)]  # ... is replaced by the frame set here
    assert hc.F0_SET.startswith("frames: ")
