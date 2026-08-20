"""Smoke test for the refined analysis deck builder (wfield_local.locanmf_analysis_deck)."""
from wfield_local import locanmf_analysis_deck as ad

pptx = __import__("pytest").importorskip("pptx")


def test_builds_and_skips_missing(tmp_path):
    # empty figure dir -> deck still builds (title + dividers + slides), all figures reported missing
    out = tmp_path / "deck.pptx"
    summary = ad.build_analysis_deck(tmp_path / "figs", out, dates=["0606", "0807"], animals=["PS92", "PS93"])
    assert out.exists()
    assert summary["slides"] > 5                       # title + 4 dividers + per-animal/summary/RSA slides
    assert summary["figures_present"] == 0             # no figures on disk
    assert summary["figures_missing"] > 0
    assert summary["tag"] == "0606-0807"


def test_places_present_figures(tmp_path):
    figs = tmp_path / "figs"
    figs.mkdir()
    # a 1x1 png the builder can place
    from PIL import Image
    Image.new("RGB", (4, 4), "white").save(figs / "locanmf_rsa_crossnobis_0606-0807.png")
    Image.new("RGB", (4, 4), "white").save(figs / "locanmf_decoder_rolling_by_animal_PS92.png")
    summary = ad.build_analysis_deck(figs, tmp_path / "d.pptx", dates=["0606", "0807"], animals=["PS92"])
    assert summary["figures_present"] == 2


def test_deck_carries_no_stale_inflation_warning():
    """The deck once opened with a slide titled "PRE-CUE numbers in this deck are inflated ~2x", plus
    three slide titles repeating it. After the 2026-08-14 correction those numbers ARE the corrected
    ones, so the warning became false -- and a stale warning is worse than none: it tells a reader to
    discount numbers that are now right. Priya spotted it still in the built deck.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "wfield_local" / "locanmf_analysis_deck.py").read_text(encoding="utf-8")
    bad = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue                                  # explanatory comments may name the old text
        if re.search(r"inflated ~2|~half this|filter-inflated|are inflated by", line):
            bad.append(f"{i}: {line.strip()[:70]}")
    assert not bad, "stale inflation warning still rendered into the deck:\n  " + "\n  ".join(bad)


def test_incomplete_rebuild_refuses_to_overwrite_an_existing_deck(tmp_path):
    """The 2026-08-19 failure: a deck with holes published itself over a good one.

    await_locanmf fitted LocaNMF to the superseded SVTcorr and wrote it where nothing reads, so the
    whole 8/19 column raised FileNotFoundError -- and the build shipped anyway at 20 missing.
    """
    import pytest

    from wfield_local.locanmf_analysis_deck import DeckIncomplete, _refuse_incomplete_overwrite

    p = tmp_path / "spout_position_analysis_summary.pptx"
    p.write_bytes(b"x" * 4096)
    with pytest.raises(DeckIncomplete, match="missing 2 figure"):
        _refuse_incomplete_overwrite(p, ["locanmf_encoder_quiet_drift_0819.png",
                                         "locanmf_decoder_rolling_cue_0819.png"])


def test_the_blocked_deck_names_every_missing_figure(tmp_path):
    """A count alone cannot be acted on -- the caller truncates, so the names ride on the exception."""
    import pytest

    from wfield_local.locanmf_analysis_deck import DeckIncomplete, _refuse_incomplete_overwrite

    p = tmp_path / "deck.pptx"
    p.write_bytes(b"x" * 4096)
    missing = [f"fig_{i}.png" for i in range(25)]
    with pytest.raises(DeckIncomplete) as ei:
        _refuse_incomplete_overwrite(p, missing)
    assert ei.value.missing_figures == missing              # every one, not just the 20 shown
    assert "... and 5 more" in str(ei.value)


def test_a_complete_rebuild_is_allowed(tmp_path):
    from wfield_local.locanmf_analysis_deck import _refuse_incomplete_overwrite

    p = tmp_path / "deck.pptx"
    p.write_bytes(b"x" * 4096)
    _refuse_incomplete_overwrite(p, [])                     # nothing missing -> publish


def test_allow_missing_tolerates_a_deliberate_gap(tmp_path):
    from wfield_local.locanmf_analysis_deck import _refuse_incomplete_overwrite

    p = tmp_path / "deck.pptx"
    p.write_bytes(b"x" * 4096)
    _refuse_incomplete_overwrite(p, ["one.png"], allow_missing=1)


def test_a_brand_new_deck_may_have_gaps(tmp_path):
    """A tree still filling up is a real case: there is no good deck to destroy yet."""
    from wfield_local.locanmf_analysis_deck import _refuse_incomplete_overwrite

    _refuse_incomplete_overwrite(tmp_path / "does_not_exist.pptx", ["a.png", "b.png"])
