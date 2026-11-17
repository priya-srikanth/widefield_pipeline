"""Trial-set captions (Priya 11-09) on the CD family, the rebuilt page-figure drivers and the gain-bio figures.

Synthetic / monkeypatched only: no driver reads data. Each driver must DECLARE its trial set in main() after argparse
(so the --plot-only replot stamps too), and a figure saved through its real save path must carry the sentence (a
figure text with gid "trial_set").
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from wfield_local import trial_set_text as tst  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RF = REPO / "scripts" / "rebuild_figs"
GB = REPO / "scripts" / "gain_bio"

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}


@pytest.fixture
def camera(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_", "GAIN_BIO_", "OROFACIAL_GATE")):
            monkeypatch.delenv(k, raising=False)
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(RF))
    monkeypatch.syspath_prepend(str(GB))
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import _common                                   # no producer outputs here: the encoder gate falls back to the env
    monkeypatch.setitem(_common._RECORDED, "v", {})
    yield


def _stamp_of(fig) -> str | None:
    t = [x.get_text() for x in fig.texts if x.get_gid() == tst.GID]
    return " ".join(t[0].split()) if t else None     # the stamp wraps the sentence (and folds double spaces)


def _flat(s: str) -> str:
    return " ".join(s.split())


class _Stop(Exception):
    pass


# ------------------------------------------------------------------------------------------------ CD family
def test_cd_lick_gate_is_success_only(camera):
    from wfield_local import cd_trajectories as cdt
    s = cdt.trial_set_sentence("cue", "lick")
    assert s.startswith("trials: lick trials (LP licks:")
    assert "excluded: no-lick trials" in s and "late hits" in s


def test_cd_lick_or_working_adds_the_camera_miss_working_rows(camera):
    """categorize's camera branch: undetected cues whose camera class is in sel_classes("miss_working")."""
    from wfield_local import cd_trajectories as cdt
    s = cdt.trial_set_sentence("cue", "lick_or_working")
    mw = tst.sentence("miss_working").split(": ", 1)[1].partition("; excluded: ")[0]
    assert mw in s and "  AND  " in s
    assert "no-lick trials of class contact, jaw-only, no response, unknown" in s
    assert "lick + no-lick trials outside the terminal quit period" not in s


def test_cd_daq_gate_words(monkeypatch):
    for k in CAMERA_ENV:
        monkeypatch.delenv(k, raising=False)
    from wfield_local import cd_trajectories as cdt
    s = cdt.trial_set_sentence("cue", "lick_or_working")
    assert "DAQ quit-tail rule" in s and "no-lick trials in the terminal quit period" in s


def test_cd_lick_alignment_forces_success(camera):
    from wfield_local import cd_trajectories as cdt
    s = cdt.trial_set_sentence("lick", "lick_or_working")
    assert "gate is forced to lick trials" in s and "  AND  " not in s
    assert cdt.trial_set_sentence("precue", "lick").count("lick-free window is dropped") == 1
    multi = cdt.trial_set_sentence(["precue", "cue", "lick"], "lick_or_working")
    assert "pre-cue arm:" in multi and "lick-aligned arm: lick trials only" in multi


def test_cd_worker_stamps_each_unit(camera, monkeypatch, tmp_path):
    """`_render_animal` (the spawned worker): every layout figure of an (align, gate) unit carries that unit's set."""
    from wfield_local import cd_trajectories as cdt
    seen = {}

    def fake_layout(res, png):
        fig = plt.figure()
        fig.savefig(png)
        seen[Path(png).name] = _stamp_of(fig)
        plt.close(fig)
        return png

    monkeypatch.setattr(cdt, "LAYOUTS", {"fake": fake_layout})
    monkeypatch.setattr(cdt, "load_result", lambda *a, **k: {"animal": "PS95"})
    item = {"animal": "PS95", "aligns": ["cue", "lick"], "gates": ["lick_or_working"], "orths": [False],
            "layouts": ["fake"], "method": "dom", "reference": "contrast", "replot": True, "out": str(tmp_path)}
    out = cdt._render_animal(item)
    assert not out["errors"], out
    assert len(seen) == 2
    for name, stamp in seen.items():
        al = "cue" if "_cue_" in name else "lick"
        assert stamp == _flat(cdt.trial_set_sentence(al, "lick_or_working")), name
    assert tst.current() is None                      # the unit's set does not leak out of the worker


def test_cd_geometry_declares_and_stamps(camera, monkeypatch, tmp_path):
    import scripts.cd_geometry_figure as g
    monkeypatch.setattr(g, "collect", lambda *a, **k: ({("PS95", "cue"): {}, ("PS95", "lick"): {}}, []))
    monkeypatch.setattr(g, "load_nulls", lambda *a, **k: {})
    monkeypatch.setattr(g, "table", lambda *a, **k: "")
    got = {}

    def fake_figure(got_, out, nulls=None, normalise=False):
        fig = plt.figure()
        g.save_png_svg(fig, out)
        got["stamp"] = _stamp_of(fig)
        plt.close(fig)
        return out

    monkeypatch.setattr(g, "figure", fake_figure)
    assert g.main(["--dir", str(tmp_path), "--gate", "lick_or_working"]) == 0
    assert got["stamp"] == _flat(g.trial_set_sentence(["cue", "lick"], "lick_or_working"))


# ------------------------------------------------------------------------------------------------ rebuild_figs drivers
#: driver -> (extra argv, a phrase its sentence must hold under the camera env)
DRIVERS = {
    "fig_behaviour_hits": ([], "all trials in the working period"),
    "fig_decoding_sessions": ([], "engaged trials (first lick within"),
    "fig_enl_move": ([], "engaged trials (next lick within the engaged cut; DAQ licks (the 10-13 CSV))"),
    "fig_farcontra_attempts_sessions": ([], "licks without contact: every v36 cam4 lick"),
    "fig_frozen_refit_pooled": ([], "camera contact + incomplete (LP licks)"),
    "fig_goal_epochs": ([], "far-contra trials (LP licks; camera classes"),
    "fig_goal_summary": ([], "far-contra lick rows"),
    "fig_layer2": ([], "imaging frames from spout strobe - 0.5 s to trial stop + 1 s"),
    "fig_pooled": ([], "imaging frames from spout strobe"),
    "fig_pose_validation_sessions": ([], "3-D licks of the August pre sessions"),
    "fig_refit_matched_pre": ([], "camera contact + incomplete (LP licks)"),
    "fig_resid16": ([], "each spout (fig 7d)"),
    "fig_resid_per_animal": ([], "at far contra"),
    "fig_resid_spout": (["--spout", "far_contra"], "at spout far_contra"),
    "fig_rsplit": ([], "imaging frames from spout strobe"),
    "fig_s_per_animal": ([], "1 s segments of rest"),
    "fig_softmax": ([], "every usable cam4 post-cue LP lick"),
    "fig_softmax_stats": ([], "every usable cam4 post-cue LP lick"),
    "fig_state_position_sessions": ([], "position readout, cue window 0-2 s"),
    "fig_state_ps94_diagnostic": ([], "1 s segments of rest"),
}


def _sentence_of(m, argv):
    if m.__name__ == "fig_resid_spout":
        return m.trial_set_sentence("far_contra")
    if m.__name__ == "fig_enl_move":
        return m.trial_set_sentence(None, 60)
    if m.__name__ == "fig_behaviour_hits":
        return m.trial_set_sentence(m.NAME)
    return m.trial_set_sentence()


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_driver_declares_on_the_plot_only_path(camera, monkeypatch, tmp_path, name):
    """main() declares the driver's sentence right after argparse: the --plot-only replot is covered."""
    argv, phrase = DRIVERS[name]
    m = importlib.import_module(name)
    import _common

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(_common, "load_figdata", stop)
    with pytest.raises(_Stop):
        m.main(["--out", str(tmp_path), "--plot-only", *argv])
    s = tst.current_sentence()
    assert s == _sentence_of(m, argv)
    assert phrase in s, (name, s)


def test_save_with_wins_over_the_declaration(camera, tmp_path):
    import _common
    _common.declare("trials: DRIVER")
    fig = plt.figure()
    _common.save_with("frames: ONE FIGURE", fig, tmp_path / "a.png")
    assert _stamp_of(fig) == "frames: ONE FIGURE"
    fig2 = plt.figure()
    from wfield_local.figure_layout import save_png_svg
    save_png_svg(fig2, tmp_path / "b.png")
    assert _stamp_of(fig2) == "trials: DRIVER"


def test_goal_summary_each_figure_its_own_set(camera, monkeypatch, tmp_path):
    import _common
    import fig_goal_summary as g
    stamps = {}

    def drawer(k):
        def f(_fd, png):
            fig = plt.figure()
            g.save_png_svg(fig, png)
            stamps[k] = _stamp_of(fig)
            plt.close(fig)
        return f

    monkeypatch.setattr(_common, "load_figdata", lambda *a: {k: None for k in g.FIGS})
    monkeypatch.setattr(g, "DRAW", {k: (drawer(k), f"{k}.png") for k in g.FIGS})
    assert g.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert set(stamps) == set(g.FIGS)
    for k, s in stamps.items():
        assert s == _flat(g.SENTENCES[k]()), k
    assert stamps["fig1"].startswith("frames:") and stamps["fig2"].startswith("trials: far-contra trials")


def test_behaviour_hits_primary_vs_daq_robustness(camera, monkeypatch, tmp_path):
    import _common
    import fig_behaviour_hits as b
    stamps = {}

    def fake_draw(fd, png, name=b.NAME):
        fig = plt.figure()
        b.save_png_svg(fig, png)
        stamps[name] = _stamp_of(fig)
        plt.close(fig)

    monkeypatch.setattr(_common, "load_figdata", lambda *a: {})
    monkeypatch.setattr(b, "draw", fake_draw)
    assert b.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert "hit = camera contact" in stamps[b.NAME]
    assert "DAQ-strobe trials before the session's engagement cut" in stamps[b.NAME_DAQ]


def test_pooled_and_softmax_figure_sets(camera):
    import fig_pooled as p
    import fig_softmax as s
    assert set(p.SETS) == {"fig1p", "fig10p", "fig2c", "fig4p", "fig12p", "fig5p", "fig5c", "fig6p"}
    assert p.SETS["fig2c"]().startswith("trials: far-contra trials")
    assert "at far contra with the trial's first post-cue lick" in p.SETS["fig4p"]()
    assert s.width_sentence().startswith("frames: imaging frames") and s.trial_set_sentence().startswith("licks:")


def test_encoder_gate_words(camera, monkeypatch):
    import _common
    assert "reference gate" in _common.encoder_trials()
    monkeypatch.setenv("OROFACIAL_GATE", "attempted")
    inc = tst.sentence("working").split(": ", 1)[1].partition("; excluded: ")[0]
    assert inc in _common.encoder_trials() and "OROFACIAL_GATE=attempted" in _common.encoder_trials()


def _spy(monkeypatch):
    stamps = {}
    orig = plt.Figure.savefig

    def spy(self, fname, *a, **k):
        r = orig(self, fname, *a, **k)
        stamps[Path(str(fname)).name] = _stamp_of(self)
        return r

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    return stamps


def test_real_plot_only_renders_carry_the_set(camera, monkeypatch, tmp_path):
    """Real --plot-only renders (synthetic figdata, as tests/test_page_fig_scripts.py): Layer 2 (save_png_svg) and the
    softmax landing stats carry their driver's sentence on every PNG."""
    import _common
    import fig_layer2
    import fig_softmax_stats as fs
    stamps = _spy(monkeypatch)
    rng = np.random.default_rng(0)
    rows = [dict(fit="combined", area=ar, epoch=ep, animal=an, change_ref=rng.normal(),
                 **{f"change_{g}": rng.normal() for g in fig_layer2.G})
            for an in fig_layer2.AN for ar in ("cortex", "SSp-m_left") for ep in fig_layer2.EP for _ in range(3)]
    M = pd.DataFrame([dict(animal=an, chronic_subset=s, epoch="chronic", change_ref=0.1)
                      for an in fig_layer2.AN for s in ("first", "last")])
    _common.save_figdata(tmp_path, fig_layer2.NAME, {"W": pd.DataFrame(rows), "M": M})
    assert fig_layer2.main(["--out", str(tmp_path), "--plot-only"]) == 0
    for png in ("fig5_layer2_change.png", "fig6_layer2_groups.png"):
        assert stamps[png] == _flat(fig_layer2.trial_set_sentence()), png
    SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
    sess = [{"animal": an, "epoch": ep, "bins": {"soft_weight": {t: {e: float(rng.uniform()) for e in SP} for t in SP}}}
            for an in ("PS92", "PS93", "PS94", "PS95") for ep in ("pre", "acute", "subacute", "chronic")
            for _ in range(2)]
    _common.save_figdata(tmp_path, fs.NAME, {"sessions": sess})
    assert fs.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert stamps["fig9c_softmax_landing_stats.png"] == _flat(fs.trial_set_sentence())


# ------------------------------------------------------------------------------------------------ gain_bio
def test_gain_bio_frame_sets(camera):
    import gain_bio as gb
    r5 = gb.frame_set_sentence("R5")
    assert r5.startswith("frames: R5 = ") and "camera trial gate" in r5 and "swapped frame pair" in r5
    assert f"> {gb.FAR_S:g} s from every quiet event" in gb.frame_set_sentence("R5far")
    task = gb.frame_set_sentence("TASK")
    assert f"cue .. cue + {gb.TASK_S:g} s" in task and "all trials in the working period" in task
    with pytest.raises(ValueError):
        gb.frame_set_sentence("NOPE")


def test_gain_fig4_declares_and_stamps(camera, monkeypatch, tmp_path):
    import _common
    import fig_sessions as fs
    import gb_common
    got = {}

    def fake_draw(fd, out_png):
        fig = plt.figure()
        fig.savefig(out_png)
        got["s"] = _stamp_of(fig)
        plt.close(fig)
        return out_png

    monkeypatch.setattr(_common, "load_figdata", lambda *a: {})
    monkeypatch.setattr(fs, "draw", fake_draw)
    monkeypatch.setattr(gb_common, "guard", lambda p: (Path(p).mkdir(parents=True, exist_ok=True), Path(p))[1])
    assert fs.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert got["s"] == _flat(fs.trial_set_sentence())
    assert "panel C y-axis" in got["s"]


def test_gain_maps_each_figure_its_set(camera, monkeypatch, tmp_path):
    """The real `fig_maps` (synthetic maps): fig 1 = R5, fig 2 = TASK."""
    import summarise_gain_bio as sg
    from wfield_local import beta_maps
    monkeypatch.setattr(beta_maps, "brain_mask", lambda: None)
    monkeypatch.setattr(sg, "FIG", tmp_path)
    stamps = _spy(monkeypatch)
    S = pd.DataFrame([dict(label=f"{a}_{e}", animal=a, epoch=e) for a in sg.AN for e in sg.EPS])
    M = {(st, "r470_hp"): {lab: np.ones((4, 4)) for lab in S.label} for st in ("R5", "TASK")}
    sg.fig_maps(S, M, "R5", [(("R5", "r470_hp"), "raw")], "f1.png", "t")
    sg.fig_maps(S, M, "TASK", [(("TASK", "r470_hp"), "raw")], "f2.png", "t")
    assert stamps["f1.png"] == _flat(sg.map_frame_set("R5")) and stamps["f2.png"] == _flat(sg.map_frame_set("TASK"))
    assert tst.current() is None

