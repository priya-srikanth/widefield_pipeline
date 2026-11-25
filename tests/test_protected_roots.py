"""CLAUDE.md rule 11 (Priya 11-24): protected-folder registry, its allowances and its enforcement points. Pure path
logic plus writes into tmp folders only (a fixture registry names tmp folders as protected local roots)."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from wfield_local import protected_roots as pr
from wfield_local import writeguard

PROD = "M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (pr.RUN_OUT_ENV, pr.ALLOW_PRODUCTION_ENV, pr.ALLOW_FREEZE_ENV, pr.MODE_ENV, pr.REGISTRY_ENV,
              "WIDEFIELD_REBUILD_ROOT", "WIDEFIELD_FROZEN_MODEL_DIR", "WIDEFIELD_FROZEN_LOAD_ONLY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pr, "_REG", None)
    monkeypatch.setattr(pr, "_LOGGED", set())
    monkeypatch.setattr("sys.argv", ["pytest"])


@pytest.fixture
def local_reg(tmp_path, monkeypatch):
    """A fixture registry: tmp/prod (production), tmp/store (frozen store)."""
    prod, store = tmp_path / "prod", tmp_path / "store"
    prod.mkdir()
    store.mkdir()
    f = tmp_path / "reg.yaml"
    f.write_text(textwrap.dedent(f"""
        share: []
        local:
          - path: "{prod.as_posix()}"
            kind: production
            hint: "set MY_OUT"
          - path: "{store.as_posix()}"
            kind: frozen_store
        """), encoding="utf-8")
    monkeypatch.setenv(pr.REGISTRY_ENV, str(f))
    monkeypatch.setattr(pr, "_REG", None)
    return prod, store


# ------------------------------------------------------------------------------------------------ registry matching
def test_production_iti_licks_and_figures_are_protected_on_every_mount():
    for root in ("M:", "N:", "//research.files.med.harvard.edu/Neurobio"):
        p = root + PROD.split("M:", 1)[1] + "/iti_licks/figures/revup_x.png"
        assert pr.refusal(p), p
    assert pr.refusal("/Volumes/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding"
                      "/iti_licks/where_preds_splitr8.csv")


def test_new_run_folders_beside_production_are_not_protected():
    assert pr.refusal(PROD + "/iti_licks_readers3_laneE_20261125/where_preds.csv") is None
    assert pr.refusal(PROD + "/revup_fourcam_jaw3dmouth_wp_ccB1123/figures/a.png") is None
    assert pr.refusal("C:/Users/SabatiniLab/scratch/x.csv") is None


def test_patch_manuscript_and_frozen_stores_are_protected():
    for p in (PROD + "/iti_licks_patched_fusedorder_laneE_20261124/where_preds_splitr8.csv",
              "M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/main.tex",
              "M:/MICROSCOPE/Priya/Widefield/labcams/frozen_models/PS92/x/model.joblib",
              "M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/frozen_models/PS93/m",
              "M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/frozen_models_lpfinal/PS94/m"):
        assert pr.refusal(p), p


def test_fake_share_tree_under_temp_is_local():
    assert pr._key("C:/Users/x/AppData/Local/Temp/t0/MICROSCOPE/Priya/Widefield/labcams/frozen_models/a")[0] == "local"
    assert pr._key("M:/MICROSCOPE/Priya/Widefield")[0] == "share"
    assert pr._key("//?/UNC/research.files.med.harvard.edu/Neurobio/MICROSCOPE/Priya/Widefield")[0] == "share"


# ------------------------------------------------------------------------------------------------ allowances
def test_run_out_grants_only_inside_a_new_folder(monkeypatch):
    rebuild_tbl = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/labcams/grant_figures/x.png"
    assert pr.refusal(rebuild_tbl)
    monkeypatch.setenv(pr.RUN_OUT_ENV, "M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/labcams")
    assert pr.refusal(rebuild_tbl), "a RUN_OUT inside a protected root is not a new folder"
    monkeypatch.setenv(pr.RUN_OUT_ENV, PROD + "/iti_licks")
    assert "grants nothing" in pr.refusal(PROD + "/iti_licks/figures/a.png")
    monkeypatch.setenv(pr.RUN_OUT_ENV, PROD)                    # a parent of protected roots
    assert pr.refusal(PROD + "/iti_licks/figures/a.png")


def test_rebuild_root_run_writes_its_own_root_but_not_its_frozen_store(monkeypatch):
    rb = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam"
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", rb)
    assert pr.refusal(rb + "/labcams/grant_figures/x.png") is None
    assert pr.refusal(rb + "/frozen_models/PS92/m/model.joblib")
    assert pr.refusal("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/x.png"), "another rebuild root"


def test_allow_production_write_logs_reason(monkeypatch, capsys, tmp_path):
    log = tmp_path / "prod_writes.log"
    monkeypatch.setenv(pr.ALLOW_PRODUCTION_ENV, "Priya 11-25: restore revup figures")
    monkeypatch.setenv(pr.LOG_ENV, str(log))
    assert pr.refusal(PROD + "/iti_licks/figures/a.png") is None
    assert pr.refusal(PROD + "/iti_licks/figures/b.png") is None           # logged once per root
    err = capsys.readouterr().err
    assert err.count("PRODUCTION WRITE") == 1 and "restore revup figures" in err
    assert "restore revup figures" in log.read_text(encoding="utf-8")
    assert pr.refusal("M:/MICROSCOPE/Priya/Widefield/labcams/frozen_models/PS92/m"), "never opens a frozen store"


def test_allow_freeze_env_and_flag(monkeypatch):
    p = "M:/MICROSCOPE/Priya/Widefield/labcams/frozen_models/PS92/m/model.joblib"
    assert pr.refusal(p)
    monkeypatch.setenv(pr.ALLOW_FREEZE_ENV, "1")
    assert pr.refusal(p) is None
    monkeypatch.delenv(pr.ALLOW_FREEZE_ENV)
    monkeypatch.setattr("sys.argv", ["x.py", "--allow-freeze"])
    assert pr.refusal(p) is None and os.environ.get(pr.ALLOW_FREEZE_ENV) == "1"   # inherited by spawned workers


# ------------------------------------------------------------------------------------------------ enforcement points
def test_writeguard_assert_writable_refuses_protected():
    with pytest.raises(writeguard.WriteGuardError, match="rule 11"):
        writeguard.assert_writable(PROD + "/iti_licks/figures/a.png")
    writeguard.assert_writable(PROD + "/iti_licks_new_run_20261125/figures/a.png")


def test_audit_hook_blocks_open_mkdir_replace_and_pandas(local_reg, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    prod, store = local_reg
    pr.install()
    with pytest.raises(pr.ProtectedWriteError):
        open(prod / "a.txt", "w")
    with pytest.raises(pr.ProtectedWriteError):
        (prod / "figures").mkdir(parents=True, exist_ok=True)
    with pytest.raises(pr.ProtectedWriteError):
        pd.DataFrame({"a": [1]}).to_csv(prod / "t.csv")
    with pytest.raises(pr.ProtectedWriteError):
        np.save(prod / "x.npy", np.zeros(2))
    src = tmp_path / "src.txt"
    src.write_text("x", encoding="utf-8")
    with pytest.raises(pr.ProtectedWriteError):
        os.replace(src, prod / "moved.txt")
    with pytest.raises(pr.ProtectedWriteError):
        open(store / "m.joblib", "wb")
    assert not any(prod.iterdir()) and not any(store.iterdir())
    # reads stay free; a declared run folder elsewhere is untouched by the guard
    assert pr.refusal(tmp_path / "new_run" / "a.csv") is None
    monkeypatch.setenv(pr.ALLOW_FREEZE_ENV, "1")
    with open(store / "m.joblib", "wb") as fh:
        fh.write(b"x")
    monkeypatch.setenv(pr.MODE_ENV, "warn")
    with open(prod / "warned.txt", "w") as fh:                       # warn mode prints, does not raise
        fh.write("x")


def test_save_png_svg_refuses_protected(local_reg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.figure_layout import save_png_svg
    prod, _ = local_reg
    fig = plt.figure()
    with pytest.raises(writeguard.WriteGuardError):
        save_png_svg(fig, prod / "f.png")
    plt.close(fig)


def test_require_out_refuses_before_compute(local_reg, tmp_path):
    prod, _ = local_reg
    with pytest.raises(SystemExit, match="not writable"):
        pr.require_out(prod / "revup")
    assert pr.require_out(tmp_path / "fresh") == tmp_path / "fresh"


def test_registry_file_parses_and_names_the_required_roots():
    pats = [e["pattern"] for e in pr.registry() if e["where"] == "share"]
    for need in ("imaging_encoding/iti_licks", "vls_stroke_paper", "labcams/frozen_models*", "LP_analysis/*/frozen_models*",
                 "iti_licks_patched_fusedorder_laneE_20261124", "iti_licks_patched_ps95_laneE_20261124"):
        assert any(p.endswith(need) for p in pats), need
    kinds = {e["kind"] for e in pr.registry()}
    assert kinds <= {"production", "manuscript_lineage", "patch", "manuscript", "frozen_store", "superseded"}


def test_package_import_installs_the_hook():
    import wfield_local  # noqa: F401
    assert pr._installed
    assert Path(pr.__file__).parent == Path(writeguard.__file__).parent
