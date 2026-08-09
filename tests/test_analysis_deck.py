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
