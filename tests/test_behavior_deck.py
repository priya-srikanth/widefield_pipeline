"""Smoke test for the behavior deck builder (wfield_local.behavior_deck)."""
import pytest

from wfield_local import behavior_deck as bd

pytest.importorskip("pptx")
from PIL import Image  # noqa: E402


def _png(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), "white").save(p)


def test_builds_and_skips_missing(tmp_path):
    out = tmp_path / "deck.pptx"
    summary = bd.build_behavior_deck(tmp_path / "bs", out, animals=["PS92", "PS93"])
    assert out.exists()
    assert summary["slides"] >= 5                    # title + 2 animal dividers + summaries + cross-animal
    assert summary["figures_present"] == 0           # nothing on disk
    assert summary["figures_missing"] > 0


def test_places_present_figures_grouped(tmp_path):
    root = tmp_path / "bs"
    for d, t in (("20260817", "100000"), ("20260818", "110000")):
        _png(root / f"sessions/PS92/{d}/PS92_{d}_{t}_behavior.png")
        _png(root / f"sessions/PS92/{d}/PS92_{d}_{t}_licking.png")
    for suffix, _ttl, _sub in bd.ACROSS_METRICS:      # split-out per-metric cross-session figures
        _png(root / f"cohort/by_animal/PS92_{suffix}_across_sessions.png")
    _png(root / "cohort/cohort_behavior.png")
    summary = bd.build_behavior_deck(root, tmp_path / "d.pptx", animals=["PS92"])
    assert summary["figures_present"] == 11          # 2 behavior + 2 licking + 6 per-metric + cohort
    assert summary["figures_missing"] == 0


def test_prefers_concat_session_figure(tmp_path):
    root = tmp_path / "bs"
    _png(root / "sessions/PS92/20260812/PS92_20260812_152647_behavior.png")   # raw crash segment
    _png(root / "sessions/PS92/20260812/PS92_20260812_concat_behavior.png")   # rejoined session
    assert bd._fig(root, "PS92", "20260812", "behavior").name == "PS92_20260812_concat_behavior.png"


def test_session_dates_sorted_and_filtered(tmp_path):
    root = tmp_path / "bs"
    for d in ("20260818", "20260806", "notadate"):
        (root / "sessions/PS92" / d).mkdir(parents=True)
    assert bd._session_dates(root, "PS92") == ["20260806", "20260818"]
