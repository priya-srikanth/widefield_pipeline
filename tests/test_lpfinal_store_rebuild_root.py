"""The LP arm's frozen-model store follows an active rebuild root (the write guard refuses the box default)."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("lp_final_render_t", REPO / "scripts" / "lp_final_render.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_store_is_inside_the_rebuild_root(monkeypatch, tmp_path):
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path / "rb"))
    env = _mod().settings(source="lp", window=1.0, trained="lp")
    assert Path(env["WIDEFIELD_FROZEN_MODEL_DIR"]) == tmp_path / "rb" / "frozen_models_lpfinal"


def test_store_without_a_rebuild_root_is_beside_figures_working(monkeypatch, tmp_path):
    monkeypatch.delenv("WIDEFIELD_REBUILD_ROOT", raising=False)
    monkeypatch.setenv("WIDEFIELD_FIGURES_WORKING", str(tmp_path / "fw"))
    env = _mod().settings(source="lp", window=1.0, trained="lp")
    assert Path(env["WIDEFIELD_FROZEN_MODEL_DIR"]) == tmp_path / "fw" / "frozen_models_lpfinal"


def test_pre_session_count_ignores_no_lick_run_ids():
    import numpy as np
    blocks = np.array([0 * 1_000_000 + 3, 0 * 1_000_000 + 4, 2 * 1_000_000 + 1, -1, -2, -7])
    assert _mod().pre_sessions(blocks) == 2
