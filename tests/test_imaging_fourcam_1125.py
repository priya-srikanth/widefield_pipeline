"""Four-camera imaging queue fixes (CC-B, 2026-11-25; run_records/2026-11-25/IMAGING_FAILED_STEPS_20261125.md).

* f1_slowband computed 108 sessions, then failed at to_csv: the routed qc folder did not exist. The folder is now
  created BEFORE the compute.
* quiet_restyle_figs wrote under a hard-wired rebuild_20261103 root (the four-camera fence refused it): it now resolves
  through the quiet chain's own `quiet_area_spikes.OUT`.
* restnl_position_coverage wrote into a production folder from a literal path and read a legacy cache: OUT is now the
  run-routed RESTnl root in a tagged run (production only in the untagged default run), explicit --out wins.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_slowband_creates_its_output_folder_before_the_compute():
    src = (REPO / "scripts" / "slowband_decode_check.py").read_text(encoding="utf-8")
    body = src[src.index("def main("):]
    assert "Path(out).mkdir(parents=True, exist_ok=True)" in body
    assert body.index("Path(out).mkdir(") < body.index("ak.fan_sessions(")


def test_quiet_restyle_root_is_the_quiet_chains_own_resolver(monkeypatch, tmp_path):
    monkeypatch.setenv("QS_OUT_ROOT", str(tmp_path / "qroot"))
    qdir = str(REPO / "scripts" / "cohort2_qc")
    monkeypatch.syspath_prepend(qdir)
    for m in ("quiet_area_spikes", "quiet_restyle_figs"):
        sys.modules.pop(m, None)
    rf = importlib.import_module("quiet_restyle_figs")
    assert rf.Q == tmp_path / "qroot"
    assert rf.BASE.parent == rf.Q and rf.OUT.parent == rf.BASE
    src = (REPO / "scripts" / "cohort2_qc" / "quiet_restyle_figs.py").read_text(encoding="utf-8")
    assert "rebuild_20261103/labcams" not in src.replace("# ", "")  # no literal old root outside comments
    for m in ("quiet_area_spikes", "quiet_restyle_figs"):
        sys.modules.pop(m, None)


@pytest.fixture
def cov(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("restnl_position_coverage", None)
    mod = importlib.import_module("restnl_position_coverage")
    yield mod
    sys.modules.pop("restnl_position_coverage", None)


def test_restnl_coverage_explicit_out_wins(cov, tmp_path, monkeypatch):
    monkeypatch.delenv("RESTNL_COV_SCRATCH", raising=False)
    out, scr = cov.resolve_paths(tmp_path / "o")
    assert out == tmp_path / "o" and scr == tmp_path / "o" / "scratch"
    out, scr = cov.resolve_paths(tmp_path / "o", tmp_path / "s")
    assert scr == tmp_path / "s"


def test_restnl_coverage_tagged_run_never_defaults_to_production(cov, tmp_path, monkeypatch):
    from wfield_local import run_mode
    monkeypatch.delenv("RESTNL_COV_SCRATCH", raising=False)
    monkeypatch.setattr(run_mode, "cache_tag", lambda: "sometag")
    import restnl_maps
    monkeypatch.setattr(restnl_maps, "out_dir", lambda root=None: tmp_path / "routed")
    out, _ = cov.resolve_paths()
    assert out == tmp_path / "routed" / "restnl_position_coverage_check"
    assert out != cov.LEGACY_OUT
    monkeypatch.setattr(run_mode, "cache_tag", lambda: "")
    out, _ = cov.resolve_paths()
    assert out == cov.LEGACY_OUT                      # the untagged default run keeps its old folder


def test_restnl_coverage_needs_a_stage_and_has_no_literal_cache():
    src = (REPO / "scripts" / "restnl_position_coverage.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert "CACHE" not in names                       # the legacy RESTnl cache dir literal is gone
    assert ".widefield_session_cache_restnl" not in src
    assert "require_default" not in src               # routed instead of refused


def test_position_blocks_write_to_the_runs_own_tagged_qc(monkeypatch, tmp_path):
    """Rule 11 (coordinator 11-25): a tagged run writes the block tables under ITS tag, never the untagged qc."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("position_block_durations", None)
    pb = importlib.import_module("position_block_durations")
    from wfield_local import lp_licks, run_mode
    monkeypatch.setattr(lp_licks, "output_tag", lambda *a, **k: "sometag")
    monkeypatch.setattr(run_mode, "encoding_out_root", lambda run=None: tmp_path / "enc" / "sometag")
    out = pb.default_out()
    assert out == tmp_path / "enc" / "sometag" / "qc"
    out.mkdir(parents=True)
    (out / "position_block_summary.csv").write_text("label,median")
    with pytest.raises(SystemExit):
        pb.default_out()                                   # never replaces a table in a tagged run
    assert pb.default_out(out=tmp_path / "new") == tmp_path / "new"
    sys.modules.pop("position_block_durations", None)


def test_slowband_figure_reads_blocks_from_arg_then_tagged_qc_then_behavior_qc(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("slowband_caveat_figure", None)
    sf = importlib.import_module("slowband_caveat_figure")
    from wfield_local import run_mode
    monkeypatch.setattr(run_mode, "behaviour_qc_dir", lambda: tmp_path / "beh")
    q = tmp_path / "tag" / "qc"
    q.mkdir(parents=True)
    assert sf.blocks_path(q) == tmp_path / "beh" / "position_block_summary.csv"
    (q / "position_block_summary.csv").write_text("label,median")
    assert sf.blocks_path(q) == q / "position_block_summary.csv"
    assert sf.blocks_path(q, str(tmp_path / "x.csv")) == tmp_path / "x.csv"
    sys.modules.pop("slowband_caveat_figure", None)
