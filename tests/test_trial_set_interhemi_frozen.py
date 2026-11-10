"""Trial-set captions of the interhemispheric family and the frozen-encoder figure scripts (Priya 11-09: "EVERY figure
must state EXACTLY which trials were included / excluded").

Pins: (1) the frozen encoder's trial / frame sentence (`frozen_encoder.trial_set_sentence`: the modeled trials under the
run's engagement gate, the modeled frames, the Musall windows, the PETH alignments, the TE / spont designs read from the
run tag); (2) each frozen_* driver's main DECLARES the sentence of the sets it draws, after argparse (the declaration is
intercepted, nothing after it runs); (3) figures saved through the drivers' real figure functions carry the sentence of
THEIR set: the plain ``fig.savefig`` path (frozen_residual_trend, per window), ``save_png_svg`` (frozen_fr_musall, per
window), the interhemispheric ``_save`` (declared set) and its narrowed figures (plot_raw = lick sets only,
interhemi_crossarea plot2 sets); (4) the event-set sentences of lick_interhemispheric / interhemi_crossarea are built
from the drivers' own constants. Synthetic inputs only; every output goes to tmp_path.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import trial_set_text as tst  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

#: the production camera run (tests/test_trial_set_text.py CAMERA_ENV)
CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}
CAMERA_TRIALS = ("trials: camera contact + incomplete (LP licks; trial_attempts 'attempted', gate attempted), within the "
                 "working period (first-last camera lick), stopped runs (A2 + lick excise) and quit tail removed; "
                 "excluded: jaw-only, no response, unknown")
TAG = "gate-attempted_v7g3_t3d_spont_frA"


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    yield


def _norm(s: str) -> str:
    return " ".join(str(s).split())


def _stamped(fig) -> list[str]:
    return [_norm(t.get_text()) for t in fig.texts if t.get_gid() == tst.GID]


# ------------------------------------------------------------------------------------------------ frozen encoder words
def test_modeled_trials_camera_run(camera):
    from wfield_local import frozen_encoder as fe
    assert fe.modeled_trials_text(TAG) == CAMERA_TRIALS
    # the gate comes from the run tag, not the reader's environment
    assert "gate canonical_attempted" in fe.modeled_trials_text("frozen_encoder_gate-canonical_attempted_v7g3_t3d")


def test_modeled_trials_reference_and_daq_gate(monkeypatch):
    from wfield_local import frozen_encoder as fe
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    assert fe.modeled_trials_text("gate-reference_v7g3_t3d").startswith("trials: the behavior pipeline's engaged trials")
    s = fe.modeled_trials_text("gate-attempted_v7g3_t3d")
    assert s.startswith("trials: contact + incomplete trials (trial_attempts 'attempted', gate attempted")
    assert "no working-period cut" in s and "excluded: jaw-only, no response, unknown" in s


def test_frozen_scopes(camera):
    from wfield_local import frozen_encoder as fe
    frames = ("frames: each modeled trial's strobe (no strobe: cue - 3 s) - 0.5 s .. stop + 1 s; ITI frames outside "
              "those spans excluded")
    assert fe.trial_set_sentence("all", tag=TAG) == f"{CAMERA_TRIALS}; {frames}"
    assert fe.trial_set_sentence("fit", tag=TAG) == f"fit on the animal's pre sessions; {CAMERA_TRIALS}; {frames}"
    assert fe.trial_set_sentence("musall", tag=TAG, align="cue") == (
        f"{CAMERA_TRIALS}; frames: cue 0 .. +1.5 s of each modeled trial")
    assert fe.trial_set_sentence("musall", tag=TAG, align="cue", pixel=True).endswith(
        "; excluded: trials whose window runs past the recording")
    lick = fe.trial_set_sentence("musall", tag=TAG, align="lick")
    assert lick.startswith(f"{CAMERA_TRIALS}; frames: first post-cue lick -0.1 .. +1 s of each modeled trial with one")
    assert "at a close spout the first DAQ contact" in lick                          # spont design: trial lick timing
    assert lick.endswith("excluded: trials without a first lick")
    nospont = fe.trial_set_sentence("musall", tag="gate-attempted_v7g3_t3d", align="lick")
    assert "first camera lick onset (cam4 + cam1-only) in cue .. stop (no stop: cue + 3.5 s)" in nospont
    assert "DAQ contact" not in nospont
    peth = fe.trial_set_sentence("peth", tag=TAG, align="lick", spout="far_contra")
    assert peth.startswith(f"{CAMERA_TRIALS}; spout: far_contra trials only; lick-aligned PETH over the modeled trials")
    assert fe.trial_set_sentence("peth", tag=TAG, align="cue", spout="all").endswith(
        "spout: mean over the session's spouts (sessions with >= 4 spouts); cue-aligned PETH over every modeled trial")
    te = fe.trial_set_sentence("all", tag="gate-attempted_v7g3_t3d_frA_te_basis")
    assert te.endswith("frames: every frame from the first modeled trial's strobe - 0.5 s to the last one's stop + 3 s "
                       "(ITI frames included)")
    assert fe.trial_set_sentence("own", tag=TAG).endswith("per position: frames of that spout's own trials")
    with pytest.raises(ValueError):
        fe.trial_set_sentence("musall", tag=TAG, align="precue")


# ------------------------------------------------------------------------------------------------ frozen drivers declare
class _Declared(Exception):
    pass


def _fe_s(scope, tag, align=None, spout=None, pixel=False):
    from wfield_local import frozen_encoder as fe
    return fe.trial_set_sentence(scope, tag=tag, align=align, spout=spout, pixel=pixel)


def _u(*s):
    return tst.union(*s, unit="trials")


def _expected(name: str) -> str:
    claims = importlib.import_module("frozen_claims")
    ffc = importlib.import_module("frozen_fr_compare")
    t = "gate-attempted_T1"
    return {
        "frozen_area_maps": _u(_fe_s("peth", t, "lick", "far_contra"), _fe_s("peth", t, "cue", "far_contra")),
        "frozen_claims": _u(*(claims.trial_set_sentence(w) for w in ("cue", "lick"))),
        "frozen_fr_compare": _u(*(ffc.trial_set_sentence(w) for w in ("cue", "lick"))),
        "frozen_fr_musall": _u(*(_fe_s("musall", importlib.import_module("frozen_fr_musall").RUNS["noFR"], w)
                                 for w in ("cue", "lick"))),
        "frozen_kernel_maps": _fe_s("fit", t),
        "frozen_musall_figures": _u(_fe_s("musall", t, "cue"), _fe_s("musall", t, "lick")),
        "frozen_peth_trial_maps": _u(_fe_s("musall", t, "cue"), _fe_s("musall", t, "lick")),
        "frozen_pixel_maps": _u(_fe_s("peth", t, "lick", "far_contra"), _fe_s("peth", t, "cue", "far_contra")),
        "frozen_pixel_musall": _fe_s("musall", t, "lick", pixel=True),
        "frozen_pixel_unique_sig": _fe_s("all", t),
        "frozen_posinfo": _u(_fe_s("musall", t, "cue"), _fe_s("musall", t, "lick")),
        "frozen_r2_maps": _fe_s("all", t),
        "frozen_residual_significance": _u(_fe_s("peth", t, "lick", "all"), _fe_s("peth", t, "cue", "all")),
        "frozen_residual_trend": _u(_fe_s("peth", t, "lick", "far_contra"), _fe_s("peth", t, "cue", "far_contra")),
        "frozen_spout_maps": _fe_s("peth", t, "cue", "each"),
    }[name]


DRIVERS = [("frozen_area_maps", ["--tag", "gate-attempted_T1"]), ("frozen_claims", None), ("frozen_fr_compare", None),
           ("frozen_fr_musall", None), ("frozen_kernel_maps", ["--tag", "gate-attempted_T1"]),
           ("frozen_musall_figures", ["--tag", "gate-attempted_T1"]), ("frozen_peth_trial_maps", ["--tag", "gate-attempted_T1"]),
           ("frozen_pixel_maps", ["--tag", "gate-attempted_T1"]), ("frozen_pixel_musall", ["--tag", "gate-attempted_T1", "--window", "lick"]),
           ("frozen_pixel_unique_sig", ["--tag", "gate-attempted_T1"]), ("frozen_posinfo", ["--tag", "gate-attempted_T1"]),
           ("frozen_r2_maps", ["--tag", "gate-attempted_T1"]), ("frozen_residual_significance", ["--tag", "gate-attempted_T1", "--spout", "all"]),
           ("frozen_residual_trend", ["--tag", "gate-attempted_T1"]), ("frozen_spout_maps", ["--tag", "gate-attempted_T1", "--align", "cue"])]


@pytest.mark.parametrize("name,argv", DRIVERS, ids=[d[0] for d in DRIVERS])
def test_frozen_driver_declares_its_set(name, argv, camera, monkeypatch):
    mod = importlib.import_module(name)

    def fake_declare(variant=None, align=None, *, text=None):
        raise _Declared(text)
    monkeypatch.setattr(tst, "declare", fake_declare)
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(_Declared) as ex:
        mod.main() if argv is None else mod.main(argv)
    got = ex.value.args[0]
    assert got == _expected(name)
    assert "camera contact + incomplete" in got and "excluded: jaw-only, no response, unknown" in got


# ------------------------------------------------------------------------------------------------ frozen figures carry it
def _musall_session(root: Path, folder: str, variant: str, label: str, animal: str, epoch: str, rng) -> None:
    from frozen_fr_compare import AREAS
    d = root / folder / variant / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    r2 = {w: {m: list(rng.normal(0.2, 0.05, len(AREAS))) for m in ("frame", "trial", "avg")} for w in ("cue", "lick")}
    (d / f"{label}.json").write_text(json.dumps({"label": label, "animal": animal, "epoch": epoch, "areas": AREAS,
                                                 "r2_musall": r2}), encoding="utf-8")


def test_fr_musall_figures_carry_their_window(camera, monkeypatch, tmp_path):
    """save_png_svg path, one figure per Musall window, each stamped with ITS window."""
    m = importlib.import_module("frozen_fr_musall")
    monkeypatch.setattr(m, "E", tmp_path)
    monkeypatch.setattr(importlib.import_module("frozen_fr_compare"), "E", tmp_path)   # session_r2 reads its own E
    monkeypatch.setattr(m, "Q", m.Q[:2])                 # two quantities keep the bootstrap small
    rng = np.random.default_rng(0)
    for _fr, folder in m.RUNS.items():
        for v in ("joint", *(v_ for _q, v_, _l in m.Q)):
            for an in ("PS92", "PS93", "PS94", "PS95"):
                for ep in ("pre", "acute", "subacute", "chronic"):
                    for k in range(2):
                        _musall_session(tmp_path, folder, v, f"{an}_{ep}{k}", an, ep, rng)
    seen = {}
    real = m.save_png_svg

    def rec(fig, path, *a, **k):
        out = real(fig, path, *a, **k)
        seen[Path(path).name] = _stamped(fig)
        return out
    monkeypatch.setattr(m, "save_png_svg", rec)
    assert m.main() == 0
    plt.close("all")
    for w in ("cue", "lick"):
        assert seen[f"frozen_musall_{w}.png"] == [_norm(m.trial_set_sentence(w))]
        assert (tmp_path / "frozen_fr_compare" / "figures" / f"frozen_musall_{w}.png").exists()
    assert "first post-cue lick" in seen["frozen_musall_lick.png"][0]
    assert "first post-cue lick" not in seen["frozen_musall_cue.png"][0]


def test_residual_trend_plain_savefig_carries_its_window(camera, monkeypatch, tmp_path):
    """Plain fig.savefig path (the savefig hook), one figure per window, far-contra trials."""
    m = importlib.import_module("frozen_residual_trend")
    from wfield_local import config, epochs
    from wfield_local import frozen_encoder as fe
    monkeypatch.setattr(fe, "out_root", lambda: tmp_path)
    monkeypatch.setattr(config, "early_pre", lambda label: False)
    monkeypatch.setattr(epochs, "days_since_stroke", lambda label: int(label.split("_d")[1]))
    run = tmp_path / "frozen_encoder_gate-attempted_T1" / "joint" / "sessions"
    run.mkdir(parents=True)
    off = np.round(np.arange(-1.0, 2.0001, 0.1), 3)
    rng = np.random.default_rng(1)
    areas, pos = ["SSp-m_left", "SSp-m_right", "cortex"], ["far_contra", "far_ipsi"]
    for an in ("PS92", "PS93"):
        for ep, days in (("pre", (-8, -6, -4, -2)), ("acute", (2, 4)), ("chronic", (30, 35, 40, 45))):
            for dd in days:
                lab = f"{an}_d{dd}"
                (run / f"{lab}.json").write_text(json.dumps({"label": lab, "animal": an, "epoch": ep,
                                                             "peth_parts": ["data", "full"], "pos_order": pos,
                                                             "areas": areas}), encoding="utf-8")
                P = rng.normal(0, 1e-3, (2, len(pos), len(areas), len(off)))
                np.savez(run / f"{lab}.npz", peth=P, peth_lick=P, peth_off_s=off)
    seen = {}
    real = matplotlib.figure.Figure.savefig

    def rec(self, fname, *a, **k):
        out = real(self, fname, *a, **k)            # the installed hook (stamp) runs inside `real`
        seen[Path(fname).name] = _stamped(self)
        return out
    tst.install_savefig_hook()
    real = matplotlib.figure.Figure.savefig
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", rec)
    assert m.main(["--tag", "gate-attempted_T1"]) == 0
    plt.close("all")
    for w in ("lick", "cue"):
        assert seen[f"residual_trend_{w}.png"] == [_norm(m.trial_set_sentence("gate-attempted_T1", w))]
        assert "spout: far_contra trials only" in seen[f"residual_trend_{w}.png"][0]


# ------------------------------------------------------------------------------------------------ interhemispheric
@pytest.fixture
def interhemi(monkeypatch):
    """lick_interhemispheric + interhemi_crossarea imported under the PRIMARY rest variant (restnl), back to the
    default after the test (as tests/test_interhemi_rest_variant.py)."""
    monkeypatch.setattr(sys, "argv", ["x", "--rest-variant", "restnl"])
    mods = []
    for n in ("lick_interhemispheric", "interhemi_crossarea"):
        mods.append(importlib.reload(sys.modules[n]) if n in sys.modules else importlib.import_module(n))
    yield mods
    monkeypatch.setattr(sys, "argv", ["x"])
    for n in ("lick_interhemispheric", "interhemi_crossarea"):
        importlib.reload(sys.modules[n])


def test_interhemi_event_sets_from_driver_constants(interhemi, camera):
    lih, ica = interhemi
    import iti_licks as il
    import restnl
    s = lih.trial_set_sentence()
    assert s.startswith("events: cued = the FIRST post-cue lick of each lick bout")
    assert f"bout = lick onsets < {il.BOUT_GAP:g} s apart" in s and f"no {il.N_TRIAL_MAX} cap" in s
    for st in ("iti_all", "iti_far1", "iti_gap2"):
        assert f"{st} = the decoder's {tst.ITI_LICK_WHAT}" in s
    assert tst.iti_filter_text("far1") in s and tst.iti_filter_text("gap2") in s
    assert f"rest = one pseudo-event at the center of each rest period >= {lih.REST_MIN_SPAN:g} s" in s
    assert restnl.DEF[: restnl.DEF.index(restnl._WDEF)].rstrip(", ") in s       # the RESTnl mask, from restnl.DEF
    assert restnl._WDEF not in s                                                 # (its weighting does not apply)
    assert "sessions without licks_all.csv" in s
    assert s.endswith(f"a set enters a session only with >= {lih.MIN_N} events")
    raw = lih.trial_set_sentence(lih.LICK_SETS)
    assert "rest =" not in raw and "iti_gap2 =" in raw
    simple = lih.trial_set_sentence(("cued", "rest"))
    assert "iti_all =" not in simple and "rest =" in simple
    xa = ica.trial_set_sentence(ica.SETS1)
    assert f"iti_dock = the decoder's {tst.ITI_LICK_WHAT}; {tst.iti_filter_text('dock')}" in xa
    assert "iti_far1" not in xa


def test_interhemi_restdock05_rest_frames(monkeypatch, camera):
    monkeypatch.setattr(sys, "argv", ["x"])
    lih = importlib.reload(sys.modules["lick_interhemispheric"]) if "lick_interhemispheric" in sys.modules else \
        importlib.import_module("lick_interhemispheric")
    from wfield_local import config
    r = config.defaults()["segmentation"]["rest"]
    s = lih.set_sentence("rest")
    assert f"frames of the {r['variant']} rest mask (DAQ licks excluded" in s and "RESTnl" not in s


def test_interhemi_save_paths_carry_their_sets(interhemi, camera, monkeypatch, tmp_path):
    lih, ica = interhemi
    tst.declare(text=lih.trial_set_sentence())
    fig = plt.figure()
    fig.suptitle("t")
    lih._save(fig, tmp_path / "a.png", dpi=40)                                   # the declared set (every set)
    assert _stamped(fig) == [_norm(lih.trial_set_sentence())]
    assert (tmp_path / "a.png").exists()
    plt.close(fig)
    tst.declare(text=ica.trial_set_sentence(ica.SETS1))
    for sets, name in ((None, "b.png"), (ica.PLOT2_SETS, "c.png"), (ica.SETS2, "d.png")):
        fig = plt.figure()
        fig.suptitle("t")
        ica._save(fig, tmp_path / name, dpi=40, sets=sets)
        assert _stamped(fig) == [_norm(ica.trial_set_sentence(ica.SETS1 if sets is None else sets))]
        plt.close(fig)
    assert "iti_all =" not in _norm(ica.trial_set_sentence(ica.PLOT2_SETS))


def test_plot_raw_is_captioned_lick_sets_only(interhemi, camera, monkeypatch, tmp_path):
    """The RAW figures (no rest reference) draw the lick sets only: plot_raw's caption names no rest set even under the
    driver's every-set declaration."""
    lih, _ica = interhemi
    monkeypatch.setattr(lih, "FIGD", tmp_path / "figdata")
    monkeypatch.setattr(lih, "FIGS", tmp_path / "figures")
    (tmp_path / "figdata").mkdir()
    rows = []
    for m, tau, win, _l in lih.RAW_ROWS:
        for st in lih.LICK_SETS:
            for ep in lih.EPS[1:]:
                for p in lih.PAIRS:
                    rows.append({"set": st, "metric": m, "tau": tau, "win": win, "epoch": ep, "pair": p,
                                 "delta": 0.01, "lo": -0.01, "hi": 0.03, "excl0": False})
    rc = []
    for tgt in ("aim", "share"):
        for st in lih.LICK_SETS:
            for m, tau in (("nc_post", 0.0), ("cc0", 0.0), ("asym", 0.6)):
                rc.append({"target": tgt, "pair": "oro", "win": lih._win(st), "set": st, "tau": tau, "metric": m,
                           "r_pooled_within": 0.1, "p_perm": 0.5, "n_same_sign": 3, "n_animals": 4})
    with open(tmp_path / "figdata" / lih.RAW_NAMES["figdata"], "wb") as fh:
        import pickle
        pickle.dump({"T": pd.DataFrame(rows), "Rc": pd.DataFrame(rc)}, fh)
    seen = {}
    real = lih.save_png_svg

    def rec(fig, path, *a, **k):
        out = real(fig, path, *a, **k)
        seen[Path(path).name] = _stamped(fig)
        return out
    monkeypatch.setattr(lih, "save_png_svg", rec)
    tst.declare(text=lih.trial_set_sentence())
    lih.plot_raw()
    plt.close("all")
    want = [_norm(lih.trial_set_sentence(lih.LICK_SETS))]
    assert seen[lih.RAW_NAMES["fig"]] == want and seen[lih.RAW_NAMES["fig_rec"]] == want
    assert tst.current_sentence() == lih.trial_set_sentence()                    # the narrowing closed with the call
