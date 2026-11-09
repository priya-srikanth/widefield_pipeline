"""Trial-set captions of the ITI lick drivers, part B (Priya 11-09: "EVERY figure must state EXACTLY which trials were
included / excluded").

Every driver declares the LICK-EVENT set its figures draw (ITI licks of the session files, cued post-cue licks, first
cued licks, bout starts, spout maps, the subset ITI filter only where the driver really reads the filtered copy) and,
where one driver draws different sets, activates the narrower sentence per figure. No real data is read: every input is
synthetic or monkeypatched and every figure goes to tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

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


def _import(name):
    import importlib
    return importlib.import_module(name)


class Rec:
    """Stub for a figure function: records the trial-set sentence active when it is called."""

    def __init__(self):
        self.calls = []

    def fn(self, tag):
        def f(*a, **k):
            self.calls.append((tag, tst.current_sentence()))
        return f


def _flat(s: str) -> str:
    return " ".join(str(s).split())


def _caption(fig) -> str:
    t = [x for x in fig.texts if x.get_gid() == tst.GID]
    assert len(t) == 1, "figure carries no (or two) trial-set captions"
    return _flat(t[0].get_text())


# ------------------------------------------------------------------------------------------------ the shared words
def test_decoder_sets_sentence(camera):
    import iti_licks as il
    import iti_where_decoder as wd
    s = wd.trial_set_sentence()
    assert s.startswith("events: ITI licks (3-D whole-session lick table, context iti")
    assert "  AND  post-cue licks (3-D lick table, context post_cue" in s
    assert f"at most {il.N_TRIAL_MAX} per session (random draw, seed 0)" in s
    assert f"sessions with >= {wd.MIN_SESSION_LICKS} such licks" in s
    assert "lick peak >= 1 s inside the imaging" in s
    assert "every trial (no engagement / camera-class gate)" in s
    assert "ITI filter" not in s                      # the decoder applies no ITI filter


def test_subset_filter_only_where_the_filtered_copy_is_read(camera, monkeypatch, tmp_path):
    import iti_licks as il
    import iti_where_decoder as wd
    monkeypatch.setattr(il, "ITI_SUBSET", "dock")
    monkeypatch.setattr(il, "OUT", tmp_path)
    monkeypatch.setattr(il, "OUT_W", tmp_path / "itifilter_dock")
    assert wd.subset_in(tmp_path / "itifilter_dock" / "zscore_check" / "x.csv") == "dock"
    assert wd.subset_in(tmp_path / "where_preds.csv") == ""
    z = wd.trial_set_sentence(z=True)
    assert "ITI filter dock: onset while the spout is DOCKED" in z
    assert "ITI filter" not in wd.trial_set_sentence(z=False)
    # cued rows are never filtered: the filter sits in the ITI half only
    assert "ITI filter" not in z.split("  AND  ")[1]


def test_where_decoder_main_block_declares(camera):
    import iti_where_decoder as wd
    wd.declare_trial_set(False)
    assert tst.current_sentence() == wd.trial_set_sentence(False)


# ------------------------------------------------------------------------------------------------ per driver
def _common(monkeypatch, mod, fd):
    if hasattr(mod, "plot_only"):
        monkeypatch.setattr(mod, "plot_only", lambda: True)
    monkeypatch.setattr(mod, "load_fd", lambda stem, *a, **k: fd(stem))
    import iti_licks as il
    monkeypatch.setattr(il, "split_only", lambda *_: None)


def test_where_scatter_per_figure_sets(camera, monkeypatch):
    import iti_where_scatter as ws
    r = Rec()
    _common(monkeypatch, ws, lambda stem: {"st": "trial" if "cued" in stem else "iti"})
    monkeypatch.setattr(ws, "plot", lambda F: r.fn(F["st"])())
    monkeypatch.setattr(ws, "plot_delta", r.fn("delta"))
    monkeypatch.setattr(sys, "argv", ["x", "--plot-only"])
    ws.main()
    got = dict(r.calls)
    assert got["trial"] == ws.trial_set_sentence(False, "trial") and got["trial"].startswith("events: post-cue licks")
    assert got["iti"] == ws.trial_set_sentence(False, "iti") and got["iti"].startswith("events: ITI licks")
    assert "grouped by the spout nearest the actual landing" in got["iti"]
    assert got["delta"] == ws.trial_set_sentence(False) and "  AND  " in got["delta"]


@pytest.mark.parametrize("modname, fd, stubs, argv, expect", [
    ("iti_where_angle_reach", {}, ["plot"], None, lambda m: m.trial_set_sentence()),
    ("iti_where_mae", {"S": None, "T": None}, ["plot"], None, lambda m: m.trial_set_sentence()),
    ("iti_where_per_spout", {"T": None, "S": None, "tag": ""}, ["plot"], None, lambda m: m.trial_set_sentence(False)),
    ("iti_overcorrection", {"S": None, "T": None, "angle": False}, ["plot"], [], lambda m: m.trial_set_sentence()),
    ("iti_recovery_measure", {"S": None, "aim": "first"}, ["plot"], None, lambda m: m.trial_set_sentence("first")),
    ("iti_recovery_marker", {}, ["plot"], None, lambda m: m.trial_set_sentence(m.BEH)),
    ("iti_spout_patterns", {}, ["plot"], None, lambda m: m.set_caption("trial")),
    ("iti_spout_animal_panels", {}, ["plot"], None, lambda m: m.trial_set_sentence()),
    ("iti_spout_pattern_stats", {}, ["plot_all"], ["--iti"], lambda m: m.set_caption("iti")),
    ("iti_where_drift", {}, ["plot"], ["--plot-only", "--june"], lambda m: m.trial_set_sentence(True)),
])
def test_driver_declares_its_set(camera, monkeypatch, modname, fd, stubs, argv, expect):
    mod = _import(modname)
    r = Rec()
    _common(monkeypatch, mod, lambda stem: dict(fd))
    for s in stubs:
        monkeypatch.setattr(mod, s, r.fn(s))
    monkeypatch.setattr(sys, "argv", ["x", "--plot-only"])
    mod.main(argv) if argv is not None else mod.main()
    assert r.calls and all(c[1] == expect(mod) for c in r.calls)
    assert r.calls[0][1].startswith("events: ")


def test_spout_sets_name_the_right_licks(camera):
    import iti_spout_patterns as sp
    c, i = sp.set_caption("trial"), sp.set_caption("iti")
    assert c.startswith("events: post-cue licks") and "ITI licks (3-D whole" not in c
    assert i.startswith("events: ITI licks") and "previous trial's spout" in i
    assert f">= {sp.MIN_LICKS} such licks at the six spouts" in c


def test_recovery_measure_first_vs_all(camera):
    import iti_recovery_measure as rm
    f, a = rm.trial_set_sentence("first"), rm.trial_set_sentence("all")
    assert "FIRST post-cue lick of each trial (lick_idx 0)" in f and "excluded: later licks of the trial" in f
    assert "FIRST" not in a and f">= {rm.MIN_LICKS['all']} licks per session" in a


def test_recovery_link_declares_decoder_sets_and_behaviour(camera, monkeypatch):
    import iti_recovery_link as rl
    r = Rec()
    monkeypatch.setattr(rl, "load_fd", lambda *a, **k: {})
    import iti_licks as il
    monkeypatch.setattr(il, "split_only", lambda *_: None)
    for s in ("plot_course", "plot_pooled", "plot_session_scatter"):
        monkeypatch.setattr(rl, s, r.fn(s))
    rl.main(["--plot-only", "--beh", "hit"])
    s = rl.trial_set_sentence("hit")
    assert [c[1] for c in r.calls] == [s] * 3
    assert "behavior axis: far-contra hit rate / pre (camera contact share of far-contra trials; trials: all trials " \
           "in the working period" in s


def test_session_pattern_tests_per_family(camera, monkeypatch):
    import iti_session_pattern_tests as spt
    r = Rec()
    rev = {v: k for k, v in spt.CHANGE_FIG.items()}
    _common(monkeypatch, spt, lambda stem: {"fam": rev.get(stem)})
    monkeypatch.setattr(spt, "plot_maplevel", r.fn("maplevel"))
    monkeypatch.setattr(spt, "plot_similarity", r.fn("similarity"))
    monkeypatch.setattr(spt, "plot_change", lambda D: r.fn(D["fam"])())
    spt.main()
    got = dict(r.calls)
    assert got["maplevel"] == got["similarity"] == spt.trial_set_sentence()
    assert got["spout_cued"].startswith("events: post-cue licks") and "  AND  " not in got["spout_cued"]
    assert got["spout_iti"].startswith("events: ITI licks") and "  AND  " not in got["spout_iti"]
    assert "  AND  " in got["position"] and "within-spout target" in got["within_spout"]


def test_session_cluster_tests_per_figure(camera, monkeypatch):
    import iti_session_cluster_tests as sct
    r = Rec()
    rev = {v: k for k, v in sct.FIG_NAME.items()}
    _common(monkeypatch, sct, lambda stem: {"fam": rev.get(stem), "st": stem.split("_")[1]})
    monkeypatch.setattr(sct, "plot_maps", lambda D: r.fn(D["fam"])())
    monkeypatch.setattr(sct, "plot_shape_d", r.fn("shape_d"))
    monkeypatch.setattr(sct, "plot_shape_days", lambda D: r.fn("days_" + D["st"])())
    sct.main()
    got = dict(r.calls)
    assert got["days_cued"] == got["spout_cued"] == sct.trial_set_sentence("cued")
    assert got["days_iti"] == got["spout_iti"] == sct.trial_set_sentence("iti")
    assert got["shape_d"] == sct.trial_set_sentence("shape_d") and "  AND  " in got["shape_d"]


def test_pattern_area_stats_per_family(camera, monkeypatch, tmp_path):
    import iti_pattern_area_stats as pas
    r = Rec()
    rev = {v: k for k, v in pas.FIG_NAME.items()}
    _common(monkeypatch, pas, lambda stem: {"label": rev[stem]})
    monkeypatch.setattr(pas, "fd_path", lambda fn: tmp_path)          # exists
    monkeypatch.setattr(pas, "plot", lambda D, levels=False: r.fn((D["label"], levels))())
    pas.main()
    got = dict(r.calls)
    for lab in ("position", "spout", "spout_iti"):
        assert got[(lab, False)] == got[(lab, True)] == pas.trial_set_sentence(lab)
    assert pas.trial_set_sentence("spout").startswith("events: post-cue licks")


def test_lmm_figures_per_group(camera, monkeypatch):
    import iti_lmm_figures as lf
    import iti_recovery_measure as rm
    r = Rec()
    monkeypatch.setattr(rm, "aim_deficit", lambda *a, **k: {})
    monkeypatch.setattr(lf, "_run_group", lambda name, dfc: r.fn(name)())
    monkeypatch.setattr(sys, "argv", ["x"])
    lf.main()
    got = dict(r.calls)
    assert set(got) == set(lf.GROUPS)
    assert got["response"].startswith("events: ITI bout-start licks") and "preceding 2 s" in got["response"]
    assert got["behaviour"].startswith("events: FIRST post-cue lick of each trial")
    assert "FIRST = the first lick of its bout" in got["lickclass"]
    assert "  AND  " in got["readout"] and "  AND  " in got["perspout"]


def test_lmm_refit_filter_reaches_only_like_for_like(camera, monkeypatch):
    import iti_lick_order as lo
    import iti_lmm_figures as lf
    monkeypatch.setattr(lo, "ITI_FILTER", "gap2")
    s = lf.trial_set_sentence("readout")
    assert "ITI filter gap2: no lick onset in the preceding 2 s" in s
    assert "the ITI filter applies to the like-for-like panels only" in s
    assert "ITI filter gap2" in lf.trial_set_sentence("lickclass")
    assert "ITI filter" not in lf.trial_set_sentence("response")


# ------------------------------------------------------------------------------------------------ real figures
def _capture(monkeypatch, mod):
    from wfield_local import figure_layout
    figs = []

    def save(fig, fname, *a, **k):
        figs.append(fig)
        return figure_layout.save_png_svg(fig, fname, *a, **k)
    monkeypatch.setattr(mod, "save_png_svg", save)
    return figs


def test_where_mae_figure_carries_sentence(camera, monkeypatch, tmp_path):
    import iti_licks as il
    import iti_where_mae as mae
    (tmp_path / "figures").mkdir()
    monkeypatch.setattr(il, "OUT", tmp_path)
    figs = _capture(monkeypatch, mae)
    rng = np.random.default_rng(0)
    rows = []
    for st in ("iti", "trial"):
        for a in ("PS92", "PS93"):
            for ep in mae.EP4:
                for k in range(3):
                    e = rng.normal(0.2, 0.05, (30, 2))
                    rows.append(pd.DataFrame({"set": st, "animal": a, "epoch": ep, "label": f"{a}_{ep}{k}",
                                              "act_lat": 0, "act_sag": 4, "act_out": 1, "dec_lat": e[:, 0],
                                              "dec_sag": 4 + e[:, 1], "dec_out": 1}))
    S = mae.sessions(pd.concat(rows, ignore_index=True))
    tst.declare(text=mae.trial_set_sentence())
    mae.plot(S, mae.tests(S))
    assert (tmp_path / "figures" / "lick_where_mae.png").exists()
    assert _caption(figs[0]) == _flat(mae.trial_set_sentence())


def test_recovery_measure_figure_carries_sentence(camera, monkeypatch, tmp_path):
    import iti_licks as il
    import iti_recovery_measure as rm
    (tmp_path / "figures").mkdir()
    monkeypatch.setattr(il, "OUT", tmp_path)
    figs = _capture(monkeypatch, rm)
    S = pd.DataFrame({"animal": np.repeat(["PS92", "PS93"], 4), "epoch": ["pre", "acute", "subacute", "chronic"] * 2,
                      "day": [-3, 2, 9, 30] * 2, "weighted_signed": np.linspace(-3, 1, 8),
                      "far_contra_signed": np.linspace(-5, 0, 8)})
    monkeypatch.setattr(rm, "plot_only", lambda: True)
    monkeypatch.setattr(rm, "load_fd", lambda stem: {"S": S, "aim": "first"})
    rm.main()
    assert (tmp_path / "figures" / "lick_cued_recovery_measure.png").exists()
    assert _caption(figs[0]) == _flat(rm.trial_set_sentence("first"))


def test_where_scatter_real_figures_carry_their_set(camera, monkeypatch, tmp_path):
    import iti_licks as il
    import iti_where_scatter as ws
    (tmp_path / "figures").mkdir()
    monkeypatch.setattr(ws, "WOUT", tmp_path)
    monkeypatch.setattr(il, "split_only", lambda *_: None)
    figs = _capture(monkeypatch, ws)
    rng = np.random.default_rng(1)
    rows = []
    for st in ("iti", "trial"):
        for a in ws.AN:
            for ep in ws.EP:
                for p in ws.POS[:2]:
                    n = 8
                    xy = rng.normal(0, 0.3, (n, 4))
                    rows.append(pd.DataFrame({"set": st, "label": f"{a}_{ep}", "animal": a, "epoch": ep, "pos": p,
                                              "near_pos": p, "act_lat": xy[:, 0], "act_sag": 4 + xy[:, 1],
                                              "dec_lat": xy[:, 2], "dec_sag": 4 + xy[:, 3]}))
    R = pd.concat(rows, ignore_index=True)
    F = {st: ws.scatter_data(R, st, g, n_) for st, g, n_ in ws.SCATTER_SETS}
    monkeypatch.setattr(ws, "plot_only", lambda: True)
    monkeypatch.setattr(ws, "load_fd", lambda stem: F["trial" if "cued" in stem else "iti"] if "scatter" in stem else {})
    monkeypatch.setattr(ws, "plot_delta", lambda F_: None)
    monkeypatch.setattr(sys, "argv", ["x", "--plot-only"])
    ws.main()
    assert len(figs) == 2
    assert _caption(figs[0]) == _flat(ws.trial_set_sentence(False, "trial"))
    assert _caption(figs[1]) == _flat(ws.trial_set_sentence(False, "iti"))
    assert (tmp_path / "figures" / "lick_cued_where_scatter.png").exists()


def test_plain_savefig_of_a_unit_carries_the_unit_sentence(camera, monkeypatch, tmp_path):
    """The per-unit activation reaches a plain fig.savefig too (lmm groups draw through mixed_models.plot_fits)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import iti_lmm_figures as lf
    import iti_recovery_measure as rm
    figs = []

    def draw(name, dfc):
        fig, _ax = plt.subplots()
        fig.savefig(tmp_path / f"{name}.png")
        figs.append((name, fig))
    monkeypatch.setattr(rm, "aim_deficit", lambda *a, **k: {})
    monkeypatch.setattr(lf, "_run_group", draw)
    monkeypatch.setattr(sys, "argv", ["x", "--groups", "response", "behaviour"])
    lf.main()
    for name, fig in figs:
        assert _caption(fig) == _flat(lf.trial_set_sentence(name))
