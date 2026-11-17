"""Outdated slots are left out of a TAGGED (rebuild) deck build and only there (Priya 11-17, deck_registry
REBUILD_DROPPED_EPOCH / REBUILD_DROPPED_LP / REBUILD_DROPS_SECTIONS_EF); the 15f rest frozen decoder slot is kept."""
import pytest

pytest.importorskip("pptx")

from wfield_local import deck_registry as reg  # noqa: E402
from wfield_local import locanmf_analysis_deck as ad  # noqa: E402
from wfield_local import run_mode  # noqa: E402


def _png(p):
    from PIL import Image
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "white").save(p)


def _build(tmp_path, monkeypatch, tag):
    monkeypatch.setattr(run_mode, "tag", lambda *a, **k: tag)
    ep = tmp_path / "grant" / "epoch"
    for name in ("epoch_15g_transfer_by_window.png", "epoch_26_licks_per_bout.png",
                 "epoch_15f_rest_frozen_restdock05_durmatched_gapmatched.png"):
        _png(ep / name)
    out = tmp_path / f"deck_{tag or 'untagged'}.pptx"
    s = ad.build_analysis_deck(tmp_path / "figs", out, dates=["0806", "0807"], animals=["PS92"],
                               grant_dir=tmp_path / "grant", within_day=False)
    from pptx import Presentation
    titles = [sh.text_frame.text for sl in Presentation(str(out)).slides for sh in sl.shapes
              if sh.has_text_frame and sh.text_frame.text]
    return s, " | ".join(titles)


def test_untagged_build_unchanged(tmp_path, monkeypatch):
    s, titles = _build(tmp_path, monkeypatch, "")
    assert s["figures_present"] == 3                       # all three epoch figures placed
    assert any("cross_mouse" in str(m) for m in s["missing_figures"])     # section E still expected
    assert any("locanmf_rsa" in str(m) for m in s["missing_figures"])     # section F still expected


def test_tagged_build_drops_outdated_slots_keeps_15f(tmp_path, monkeypatch):
    s, titles = _build(tmp_path, monkeypatch, "T")
    assert s["figures_present"] == 1                       # only 15f
    assert not any("cross_mouse" in str(m) or "locanmf_rsa" in str(m) or "within_animal" in str(m)
                   for m in s["missing_figures"])
    assert "Cross-session summary" not in titles and "RSA" not in titles


def test_drop_lists_are_registered_slots_and_spare_15f():
    pats = {p for p, _t, _l in reg.EPOCH_FIGURES}
    assert reg.REBUILD_DROPPED_EPOCH <= pats
    assert not any(p.startswith("epoch_15f_") for p in reg.REBUILD_DROPPED_EPOCH)
    assert reg.REBUILD_DROPPED_LP <= {rel for _r, rel, _t, _l in reg.LP_GATED_FIGURES}
    assert len(reg.REBUILD_DROPPED_EPOCH) == 31 and len(reg.REBUILD_DROPPED_LP) == 3
