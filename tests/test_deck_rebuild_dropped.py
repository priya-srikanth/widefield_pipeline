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


def test_tagged_build_reads_15f_from_its_rebuild_file(tmp_path, monkeypatch):
    """15f re-pointed to the RESTnlc1 v3 render (deck_registry.EPOCH_FIGURES_REBUILD_FILE, 11-17)."""
    pat = "epoch_15f_rest_frozen_restdock05_durmatched_gapmatched.png"
    from PIL import Image
    new = tmp_path / "grant" / "epoch" / reg.EPOCH_FIGURES_REBUILD_FILE[pat]
    new.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), "red").save(new)              # distinguishable from the old-named white 4x4
    s, titles = _build(tmp_path, monkeypatch, "T")
    assert s["figures_present"] == 1                       # one 15f slide: the old-named file is NOT also placed
    from pptx import Presentation
    notes = [sl.notes_slide.notes_text_frame.text for sl in Presentation(str(tmp_path / "deck_T.pptx")).slides
             if sl.has_notes_slide]
    assert any("RESTnlc1 v3" in n for n in notes)          # the rebuild legend rides with the re-pointed image
    blobs = [sh.image.blob for sl in Presentation(str(tmp_path / "deck_T.pptx")).slides for sh in sl.shapes
             if sh.shape_type == 13]
    assert new.read_bytes() in blobs                       # the v3 render is the placed picture
    assert pat in reg.EPOCH_FIGURES_REBUILD_LEGEND
    assert set(reg.EPOCH_FIGURES_REBUILD_FILE) <= {p for p, _t, _l in reg.EPOCH_FIGURES}


def test_tagged_build_reads_15f_confusion_from_its_rebuild_file(tmp_path, monkeypatch):
    """The 15f confusion slot also reads the RESTnlc1 v3 floor-30 / central-90% render, with its own rebuild legend (11-18)."""
    main = "epoch_15f_rest_frozen_restdock05_durmatched_gapmatched.png"
    pat = "epoch_15f_rest_frozen_restdock05_durmatched_gapmatched_confusion.png"
    assert reg.EPOCH_FIGURES_REBUILD_FILE[pat] == (
        "rest15f_restnlc1all_floor30_pct5_20261118/allpre/rest15f_restnlc1v3wk_allpre_durgap_floor30_pct5_confusion.png")
    from PIL import Image
    ep = tmp_path / "grant" / "epoch"
    files = {}
    for p, color in ((main, "red"), (pat, "blue")):
        f = ep / reg.EPOCH_FIGURES_REBUILD_FILE[p]
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color).save(f)
        files[p] = f
    s, _titles = _build(tmp_path, monkeypatch, "T")
    assert s["figures_present"] == 2                       # the 15f main + confusion slides, nothing else
    from pptx import Presentation
    prs = Presentation(str(tmp_path / "deck_T.pptx"))
    blobs = [sh.image.blob for sl in prs.slides for sh in sl.shapes if sh.shape_type == 13]
    assert files[pat].read_bytes() in blobs
    notes = [sl.notes_slide.notes_text_frame.text for sl in prs.slides if sl.has_notes_slide]
    leg = reg.EPOCH_FIGURES_REBUILD_LEGEND[pat]
    assert any("Where the frozen rest decoder's errors go" in n for n in notes)
    # the v3 numbers, not the retired restdock05 / DAQ-only reading (far-contra "holds up", .56 -> .67)
    assert "pre 6160, acute 1375, subacute 1964, chronic 4725" in leg          # floor 30, central 90 % (11-18)
    assert ".41 pre, .25 acute, .33 subacute, .23 chronic" in leg
    assert "central 90 %" in leg
    assert ">= 30 matched periods post-stroke" in leg
    assert "HOLDS UP" not in leg and "0.67" not in leg
