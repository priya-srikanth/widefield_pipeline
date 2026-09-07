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
        for kind, _lbl, _sub, _note in bd.SESSION_FIGS:      # behavior, licking
            _png(root / f"sessions/PS92/{d}/PS92_{d}_{t}_{kind}.png")
    _png(root / "cohort/by_animal/PS92_raster_grid_p1.png")   # tiled task rasters (8 sessions/slide)
    for suffix, _ttl, _sub in bd.ACROSS_METRICS:             # split-out per-metric cross-session figures
        _png(root / f"cohort/by_animal/PS92_{suffix}_across_sessions.png")
    _png(root / "cohort/cohort_behavior.png")
    summary = bd.build_behavior_deck(root, tmp_path / "d.pptx", animals=["PS92"])
    # 2 days x 2 per-session figs (behavior, licking) + 1 raster-grid page + 6 per-metric + cohort
    assert summary["figures_present"] == 2 * len(bd.SESSION_FIGS) + 1 + 6 + 1 == 12
    assert summary["figures_missing"] == 0


def test_cumulative_task_raster_grid_gets_a_slide_with_a_note(tmp_path):
    """The rig-GUI raster is on the deck as a tiled per-animal grid, and its speaker note must say what
    the colours mean and where the outcomes came from — 'hit' and 'a reward was delivered' differ."""
    from pptx import Presentation
    root = tmp_path / "bs"
    _png(root / "cohort/by_animal/PS92_raster_grid_p1.png")
    _png(root / "cohort/by_animal/PS92_raster_grid_p2.png")   # a second page -> a second slide
    out = tmp_path / "d.pptx"
    bd.build_behavior_deck(root, out, animals=["PS92"])
    prs = Presentation(str(out))
    titles = [sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame]
    assert sum("cumulative task rasters" in t.lower() for t in titles) == 2   # both grid pages placed
    notes = [s.notes_slide.notes_text_frame.text for s in prs.slides if s.has_notes_slide]
    raster_note = next(n for n in notes if "Cumulative task raster" in n)
    for must in ("GREEN", "RED", "auto_after_delay", "NOT engagement-gated",
                 "DAQ recorder", "LONGEST session"):
        assert must in raster_note
    # the free/auto/manual RING was dropped (Priya, 2026-08-22). The note must not describe a marker
    # the figure does not draw -- a legend for something absent is worse than no legend.
    assert "TEAL RING" not in raster_note


def test_session_fig_kinds_match_the_producer():
    """Deck kinds and spout_behavior's file stems must agree, or a figure silently never appears."""
    from wfield_local import spout_behavior as sb
    kinds = {k for k, _l, _s, _n in bd.SESSION_FIGS}
    assert kinds == {"behavior", "licking"}                   # the raster moved to a tiled grid
    assert sb.plot_animal_raster_grid.__doc__.count("raster_grid")


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


def test_epoch_section_is_opt_in_and_placed_when_asked(tmp_path):
    """The behaviour-by-epoch figures go on the deck, but ONLY when a caller passes the directory.

    Both halves matter. `spout_behavior` builds this deck in camera-nightly step 4, BEFORE the
    epoch boundaries are resolved in step 5, so a deck built there must not carry an epoch section
    at all rather than one drawn on last night's boundaries -- `camera_nightly` rebuilds it after
    the resolve. And deriving the directory inside the builder would make this very test read the
    real MICROSCOPE share instead of its tmp root.
    """
    from pptx import Presentation
    root = tmp_path / "bs"
    ep = tmp_path / "epoch"
    for stem, _t, _s in bd.EPOCH_FIGS:
        _png(ep / f"{stem}.png")

    without = bd.build_behavior_deck(root, tmp_path / "a.pptx", animals=["PS92"])
    with_ep = bd.build_behavior_deck(root, tmp_path / "b.pptx", animals=["PS92"], epoch_dir=ep)

    assert with_ep["figures_present"] == without["figures_present"] + len(bd.EPOCH_FIGS)
    assert with_ep["slides"] == without["slides"] + 1 + len(bd.EPOCH_FIGS)   # +divider

    titles = [sh.text_frame.text for s in Presentation(str(tmp_path / "b.pptx")).slides
              for sh in s.shapes if sh.has_text_frame]
    assert any("Behaviour by epoch" in t for t in titles)
    assert any("days since lesion" in t for t in titles)
    bare = [sh.text_frame.text for s in Presentation(str(tmp_path / "a.pptx")).slides
            for sh in s.shapes if sh.has_text_frame]
    assert not any("Behaviour by epoch" in t for t in bare)
