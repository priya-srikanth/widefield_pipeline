"""THE LICK SET OF EVERY FIGURE, lick family part A (Priya 11-09: "EVERY figure must state EXACTLY which trials were
included / excluded").

The lick drivers draw LICK EVENTS, not trial classes. Each declares its set in main() (`trial_set_text.declare(text=
trial_set_sentence())`), built from the constants of the code that selects the licks (`lick_covariate_reference`
builders over iti_licks / iti_lick_order; `lick_balance_target_model.lick_set_sentence` for baltarget_licks;
`lick_balance_subsets.set_sentence` for the primary balance sets; `lick_area_importance.refit_lick_set` for the refit
decoders; `lick_code_subspace.lick_sets_sentence` for the lick-code family). Synthetic only: no driver reads real data.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wfield_local import trial_set_text as tst  # noqa: E402

CAMERA_ENV = {"WIDEFIELD_TRIAL_GATE": "camera", "WIDEFIELD_WORKING_TRIALS": "in_working",
              "WIDEFIELD_TRIAL_STOPPED": "both", "WIDEFIELD_STOP_LICKS": "excise", "WIDEFIELD_STOP_TAIL": "lick_trial",
              "WIDEFIELD_TRIAL_END": "lick", "WIDEFIELD_LICK_SOURCE": "lp", "WIDEFIELD_GATE_SOURCE": "daq"}

DRIVERS = """lick_acad_midline_control lick_aim_deficit lick_area_importance lick_area_tuning lick_area_unique_maps
lick_balance_angle_lmm lick_balance_blind_decoder lick_balance_contact lick_balance_explainer lick_balance_imaging_control
lick_balance_imaging_control_primary lick_balance_maps lick_balance_maps_svd lick_balance_matched_error lick_balance_nested
lick_balance_offset_lmm lick_balance_offset_primary lick_balance_order lick_balance_order_primary lick_balance_recovery
lick_balance_reference lick_balance_regions lick_balance_subsets lick_balance_target_model lick_balance_trial
lick_balance_variants lick_behaviour_decoder lick_block_compare lick_ci_spout_epoch lick_code_noisematch
lick_code_rotation_summary lick_code_subspace lick_correction lick_covariate_reference""".split()

KEY_ITI = "ITI licks (3-D whole-session lick table, context iti"
KEY_CUED = "post-cue licks (3-D lick table, context post_cue"
FIRST = "FIRST post-cue lick of each trial (lick_idx 0)"
#: what each driver's sentence must say (beyond the shared "events: " shape)
MUST = {
    "lick_acad_midline_control": [KEY_ITI, KEY_CUED, "a random 1500 post-cue licks"],
    "lick_aim_deficit": [FIRST, ">= 5 per spout x session", ">= 10 per spout x session"],
    "lick_area_importance": ["every 3-D tip coordinate", "sessions with >= 40 such licks"],
    "lick_area_tuning": [FIRST, "no lick onset (any licks_all.csv lick) in the preceding 2 s", "ITI filter far1",
                         "inside the animal's pre 2.5-97.5 percentile"],
    "lick_area_unique_maps": ["every 3-D tip coordinate"],
    "lick_balance_angle_lmm": ["bout start (index 0", "first post-cue lick (lick_idx 0) vs later licks"],
    "lick_balance_blind_decoder": [KEY_ITI, KEY_CUED],
    "lick_balance_contact": ["START their lick bout", "no 1500 cap", "strict miss"],
    "lick_balance_explainer": [FIRST, "animal PS94 only"],
    "lick_balance_imaging_control": ["ENGAGED trial", "ITI bout-start licks", "preceding 2 s"],
    "lick_balance_imaging_control_primary": [FIRST, "preceding 2 s"],
    "lick_balance_maps": [">= 2 sessions and >= 100 licks"],
    "lick_balance_maps_svd": [">= 2 sessions and >= 100 licks"],
    "lick_balance_matched_error": [FIRST, "sessions with >= 10 kept licks"],
    "lick_balance_nested": [FIRST, KEY_CUED, "preceding 2 s"],
    "lick_balance_offset_lmm": [KEY_ITI, KEY_CUED],
    "lick_balance_offset_primary": ["balance axis", "offset axis"],
    "lick_balance_order": ["with >= 20 licks", "order capped at 9"],
    "lick_balance_order_primary": [">= 10 licks", KEY_CUED],
    "lick_balance_recovery": ["far-contra target only"],
    "lick_balance_reference": [KEY_ITI, KEY_CUED],
    "lick_balance_regions": [KEY_ITI, KEY_CUED],
    "lick_balance_subsets": [FIRST, "preceding 2 s"],
    "lick_balance_target_model": [FIRST, "preceding 2 s", "ALM + tjM1 + SSp-m + SSp-n + SSs"],
    "lick_balance_trial": [FIRST],
    "lick_balance_variants": [FIRST, "preceding 2 s"],
    "lick_behaviour_decoder": ["common set", "FaceRhythm"],
    "lick_block_compare": ["every 3-D tip coordinate"],
    "lick_ci_spout_epoch": [FIRST, "session x spout cells with >= 5 licks"],
    "lick_code_noisematch": [FIRST, "ITI filter far1", "side_ml inside"],
    "lick_code_rotation_summary": ["enlv_on1", "enlv_bout", "ITI filter far1"],
    "lick_code_subspace": [FIRST, "ITI filter far1"],
    "lick_correction": ["PAIRS k -> k+1", "<= 0.35 s"],
    "lick_covariate_reference": ["common set", "8 covariates"],
}


@pytest.fixture
def camera(monkeypatch):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("WFIELD_ITI_FILTER", raising=False)
    yield


def _mod(name):
    return importlib.import_module(name)


def _stamped(fig) -> list[str]:
    return [t.get_text() for t in fig.texts if t.get_gid() == tst.GID]


def _flat(s: str) -> str:
    return " ".join(s.split())


# ------------------------------------------------------------------------------------------------ every driver
@pytest.mark.parametrize("name", DRIVERS)
def test_driver_sentence_and_main_declares_it(name, camera):
    M = _mod(name)
    s = M.trial_set_sentence()
    assert s.startswith("events: ")
    assert "excluded:" in s or name == "lick_code_rotation_summary"
    for frag in MUST[name]:
        assert frag in s, (name, frag)
    src = inspect.getsource(M.main)
    assert "tst.declare(text=trial_set_sentence())" in src
    # declared before any plot-only / replot branch (so a figdata replot is captioned too)
    i = src.index("tst.declare(")
    for tok in ("plot_only()", '"--plot-only" in sys.argv', "a.plot_only", "a.summarise", "--dots-only"):
        if tok in src:
            assert i < src.index(tok), (name, tok)


def test_no_declaration_at_import(camera):
    for name in DRIVERS:
        _mod(name)
    assert tst.current() is None


# ------------------------------------------------------------------------------------------------ built from the code
def test_key_set_follows_the_extraction_constants(camera, monkeypatch):
    import iti_licks as il
    import lick_covariate_reference as cv
    monkeypatch.setattr(il, "N_TRIAL_MAX", 777)
    s = cv.key_lick_set()
    assert "a random 777 post-cue licks per session" in s
    assert f"sessions with >= {cv._pred_min_licks()} such licks" in s
    assert cv._pred_min_licks() == 20                  # the literal of iti_lick_order.predictions today


def test_refit_filter_and_subset_filter(camera, monkeypatch):
    import iti_licks as il
    import iti_lick_order as lo
    import lick_balance_blind_decoder as bd
    import lick_balance_reference as lbr
    monkeypatch.setattr(lo, "ITI_FILTER", "gap2")      # a REFIT filter reaches every KEY-set reader
    assert "ITI filter gap2: no lick onset in the preceding 2 s" in lbr.trial_set_sentence()
    monkeypatch.setattr(lo, "ITI_FILTER", "")
    monkeypatch.setattr(il, "ITI_SUBSET", "dock")      # a SUBSET filter only where the driver applies it
    assert "ITI filter dock" in bd.trial_set_sentence()
    assert "ITI filter dock" not in lbr.trial_set_sentence()


def test_gap_and_switches_are_read_not_copied(camera, monkeypatch):
    import iti_licks as il
    import lick_balance_subsets as lbs
    import lick_balance_target_model as tm
    monkeypatch.setattr(il, "GAP_MAIN", 3.0)
    assert "in the preceding 3 s" in tm.trial_set_sentence()
    monkeypatch.setattr(lbs, "FULL", True)
    monkeypatch.setattr(lbs, "INSIDE", True)
    s = lbs.trial_set_sentence()
    assert "START their lick bout" in s and "pre 2.5-97.5 % lick-angle range" in s
    # drivers that build their own sets ignore the subsets switches
    import lick_balance_trial as bt
    assert "START their lick bout" not in bt.trial_set_sentence()


def test_lick_code_family(camera, monkeypatch):
    import lick_code_subspace as lcs
    s = lcs.lick_sets_sentence("enlv")
    assert "onset >= 1 s after the spout arrival" in s and "peak > 0.5 s before the cue" in s
    assert "bout start (clean start" in s
    assert "enlv" not in lcs.lick_sets_sentence("")


# ------------------------------------------------------------------------------------------------ on the figure
#: drivers whose main is `if plot_only(): plot(load_fd(...))` -- the declaration must reach the replot path
REPLOT = ["lick_balance_reference", "lick_balance_regions", "lick_balance_offset_lmm", "lick_balance_order",
          "lick_balance_maps", "lick_balance_maps_svd", "lick_balance_recovery", "lick_balance_variants",
          "lick_balance_angle_lmm", "lick_area_importance", "lick_area_unique_maps", "lick_block_compare",
          "lick_balance_explainer"]


@pytest.mark.parametrize("name", REPLOT)
def test_replot_path_figure_carries_the_lick_set(name, camera, monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = _mod(name)
    seen = {}

    def fake_plot(F, *a, **k):                         # a plain plt.savefig: covered by the savefig hook
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.savefig(tmp_path / f"{name}.png")
        seen["fig"] = fig
        plt.close(fig)

    monkeypatch.setattr(M, "plot_only", lambda: True)
    monkeypatch.setattr(M, "load_fd", lambda *a, **k: {})
    monkeypatch.setattr(M, "plot", fake_plot)
    if hasattr(M, "_pre_cloud"):
        monkeypatch.setattr(M, "_pre_cloud", lambda: None)
    assert M.main() == 0
    assert (tmp_path / f"{name}.png").exists()
    assert [_flat(t) for t in _stamped(seen["fig"])] == [_flat(M.trial_set_sentence())]


def _capture(monkeypatch, M):
    """Wrap the module's save_png_svg (the real one runs) and record the saved figures."""
    figs = []
    real = M.save_png_svg

    def rec(fig, fname, *a, **k):
        figs.append((Path(fname).name, fig))
        return real(fig, fname, *a, **k)
    monkeypatch.setattr(M, "save_png_svg", rec)
    return figs


def test_nested_real_figure(camera, monkeypatch, tmp_path):
    """lick_balance_nested.main end to end on a synthetic balvar_sessions.csv (its real figure code)."""
    import iti_licks as il
    import lick_balance_nested as bn
    import lick_recovery_markers as lrm
    rng = np.random.default_rng(0)
    rows = []
    for st in bn.TARGETS:
        for a in lrm.AN:
            for k in range(6):
                rows.append({"set": st, "animal": a, "epoch": "acute" if k < 3 else "chronic", "day": 3 + 10 * k,
                             "aim": rng.normal(), "iti_ang": rng.normal(),
                             "diff:raw_0-1|oro": rng.normal(), "diff:bs_0-1|oro": rng.normal(),
                             "ratio:bs_0-1|oro": rng.normal(), "diff:bs_1-2|oro": rng.normal()})
    (tmp_path / "figures").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "balvar_sessions.csv", index=False)
    monkeypatch.setattr(il, "OUT_W", tmp_path)
    monkeypatch.setattr(bn, "N_NULL", 2)
    figs = _capture(monkeypatch, bn)
    assert bn.main() == 0
    (stem, fig), = figs
    assert stem == "lick_balance_nested.png" and (tmp_path / "figures" / stem).exists()
    assert [_flat(t) for t in _stamped(fig)] == [_flat(bn.trial_set_sentence())]


def test_order_primary_real_figure(camera, monkeypatch, tmp_path):
    """lick_balance_order_primary --plot-only on synthetic tables: its real figure carries both KEY sets."""
    import iti_licks as il
    import lick_balance_order_primary as bo
    rows = []
    for st in bo.SETS:
        for a in ("PS92", "PS93", "PS94", "PS95"):
            for ep in bo.EPS:
                for o in range(10):
                    rows.append({"set": st, "label": f"{a}_{ep}", "animal": a, "epoch": ep, "ord_c": o, "n": 20,
                                 "obs_c": 2.0, "pred_c": 1.5, "obs_i": 1.0, "pred_i": 1.0})
    (tmp_path / "figures").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "balsub_order.csv", index=False)
    pd.DataFrame([{"set": "cued", "ord_c": 0, "epoch": "acute", "sig": True}]).to_csv(tmp_path / "balorder_primary.csv",
                                                                                       index=False)
    monkeypatch.setattr(il, "OUT_W", tmp_path)
    monkeypatch.setattr(bo, "plot_only", lambda: True)
    figs = _capture(monkeypatch, bo)
    assert bo.main() == 0
    (stem, fig), = figs
    s = _flat(" ".join(_stamped(fig)))
    assert s == _flat(bo.trial_set_sentence()) and KEY_CUED in s and KEY_ITI in s


def test_maps_one_set_per_figure(camera, monkeypatch, tmp_path):
    """lick_balance_maps.plot: each figure states ITS set only (cued figure: post-cue licks, no ITI licks)."""
    import iti_licks as il
    import lick_balance_maps as bm
    img = np.full((20, 20), -1)
    img[2:9, 2:9], img[2:9, 11:18] = 0, 1
    nm = ["ALM_left", "ALM_right"]
    M_ = pd.DataFrame([{"set": st, "epoch": ep, "pair": "ALM", **{m: 0.1 for m, _t in bm.MEASURES}}
                       for st in ("cued", "ITI") for ep in il.EP4])
    (tmp_path / "figures").mkdir()
    monkeypatch.setattr(il, "OUT_W", tmp_path)
    figs = _capture(monkeypatch, bm)
    tst.declare(text=bm.trial_set_sentence())
    bm.plot({"M": M_, "img": img, "nm": nm})
    got = {stem: _flat(" ".join(_stamped(fig))) for stem, fig in figs}
    assert got["lick_balance_maps_cued.png"] == _flat(bm.trial_set_sentence("cued"))
    assert got["lick_balance_maps_iti.png"] == _flat(bm.trial_set_sentence("ITI"))
    assert KEY_ITI not in got["lick_balance_maps_cued.png"] and KEY_CUED not in got["lick_balance_maps_iti.png"]
    assert tst.current_sentence() == bm.trial_set_sentence()           # the narrower set closed again
