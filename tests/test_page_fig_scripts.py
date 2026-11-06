"""Coordinator 11-04 (RECONCILIATION section 7): the page / PI-deck figure scripts brought into the repo
(`scripts/rebuild_figs/`) -- smoke tests: each imports, has the _common contract (--out / --plot-only), carries no
hard-coded machine / adopted-run / scratch path in its CODE, and the small ones draw from synthetic figdata."""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RF = REPO / "scripts" / "rebuild_figs"
NEW = ["fig_layer2", "fig_rsplit", "fig_pooled", "fig_softmax", "fig_softmax_stats", "fig_resid_per_animal",
       "fig_resid_spout", "fig_resid16", "fig_farcontra_attempts_sessions", "fig_pose_validation_sessions"]
ALL = NEW + ["fig_goal_summary", "fig_goal_epochs", "fig_behaviour_hits", "fig_decoding_sessions",
             "fig_frozen_refit_pooled", "fig_state_position_sessions", "fig_state_ps94_diagnostic"]
BAD = ("M:/", "C:/Users", "wf_attempt", "widefield_pipeline", "scratchpad", "lp_r4_heatmap")


@pytest.fixture(autouse=True)
def _path(monkeypatch):
    monkeypatch.syspath_prepend(str(RF))
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)


def _code_strings(path: Path) -> list[str]:
    """Every string constant in the module EXCEPT docstrings (provenance notes may name the old paths)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:
            docs.add(id(node.body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


@pytest.mark.parametrize("name", ALL)
def test_no_hard_coded_paths_in_code(name):
    hits = [s for s in _code_strings(RF / f"{name}.py") if any(b in s for b in BAD) and "PROVENANCE" not in s]
    assert not hits, (name, hits)


@pytest.mark.parametrize("name", NEW)
def test_imports_and_has_the_contract(name, capsys):
    m = importlib.import_module(name)
    with pytest.raises(SystemExit) as e:
        m.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--out" in out and "--plot-only" in out


def test_layer2_and_rsplit_draw_from_figdata(tmp_path):
    import _common
    import fig_layer2
    import fig_rsplit
    rng = np.random.default_rng(0)
    G = fig_layer2.G
    rows = [dict(fit="combined", area=ar, epoch=ep, animal=an, change_ref=rng.normal(), **{f"change_{g}": rng.normal()
                                                                                          for g in G})
            for an in fig_layer2.AN for ar in ("cortex", "SSp-m_left") for ep in fig_layer2.EP for _ in range(3)]
    M = pd.DataFrame([dict(animal=an, chronic_subset=s, epoch="chronic", change_ref=0.1)
                      for an in fig_layer2.AN for s in ("first", "last")])
    _common.save_figdata(tmp_path, fig_layer2.NAME, {"W": pd.DataFrame(rows), "M": M})
    assert fig_layer2.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert (tmp_path / "fig5_layer2_change.png").exists() and (tmp_path / "fig6_layer2_groups.png").exists()
    T = {an: pd.DataFrame([dict(split=s, area=ar, epoch=ep, change_ref=rng.normal())
                           for s in range(5) for ar in fig_rsplit.AR for ep in fig_rsplit.EP]) for an in fig_rsplit.AN}
    _common.save_figdata(tmp_path, fig_rsplit.NAME, {"T": T})
    assert fig_rsplit.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert (tmp_path / "random_splits_summary.csv").exists()


def test_softmax_stats_draws_from_figdata(tmp_path):
    import _common
    import fig_softmax_stats as fs
    rng = np.random.default_rng(1)
    SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
    sess = [{"animal": an, "epoch": ep,
             "bins": {"soft_weight": {t: {e: float(rng.uniform()) for e in SP} for t in SP}}}
            for an in ("PS92", "PS93", "PS94", "PS95") for ep in ("pre", "acute", "subacute", "chronic") for _ in range(2)]
    _common.save_figdata(tmp_path, fs.NAME, {"sessions": sess})
    assert fs.main(["--out", str(tmp_path), "--plot-only"]) == 0
    assert (tmp_path / "fig9c_softmax_landing_stats.png").exists()


def test_resid16_and_resid_spout_draw_from_figdata(tmp_path):
    import _common
    import fig_resid16
    import fig_resid_spout
    F, H, W = 16, 64, 64
    npz = {(sp, al, ep): (np.random.default_rng(0).normal(size=(H, W)), np.zeros((H // F, W // F), bool), H, W)
           for sp in ["all", *fig_resid16.SP] for al in ("lick", "cue") for ep in fig_resid16.EP}
    R = pd.DataFrame([dict(spout=sp, window=al, epoch=ep, sig=0, n_bins=16, up=0, down=0)
                      for sp in ["all", *fig_resid16.SP] for al in ("lick", "cue") for ep in fig_resid16.EP])
    fd = {"R": R, "outline": np.zeros((H, W)), "npz": npz, "F": F}
    _common.save_figdata(tmp_path, f"resid{F}", fd)
    assert fig_resid16.main(["--out", str(tmp_path), "--plot-only", "--bin", "16"]) == 0
    _common.save_figdata(tmp_path, f"resid_far_contra_{F}", fd)
    assert fig_resid_spout.main(["--out", str(tmp_path), "--plot-only", "--spout", "far_contra"]) == 0
    assert (tmp_path / "fig7_residual_far_contra_16px.png").exists()


def test_default_out_is_run_tagged_under_a_rebuild_root(monkeypatch, tmp_path):
    import _common
    from wfield_local import lp_licks
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path / "rb"))
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    assert _common.default_out() == tmp_path / "rb" / "rebuild_figs" / lp_licks.output_tag()


def test_inputs_follow_the_run(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    import fig_farcontra_attempts_sessions as ff
    import fig_layer2
    import fig_resid_per_animal
    from wfield_local import frozen_encoder as fe
    from wfield_local import lp_licks, run_mode
    t = lp_licks.output_tag()
    assert t in fig_layer2.allsession_dir("x").parts and fig_resid_per_animal.encoder_run("y").parent == fe.out_root()
    assert ff.attempts_sessions() == run_mode.attempts_dir() / "trial_attempts_sessions.csv"
    assert sys.modules  # keep the import side effects explicit
