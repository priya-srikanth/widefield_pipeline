"""The auto-built decks resolve under ``presentations/pipeline`` (paths.yaml ``decks_out``; Priya 2026-11-06)."""
from __future__ import annotations

from pathlib import Path

import pytest

from wfield_local import paths
from wfield_local.paths import PathResolver, deck_dir, deck_history_dir


@pytest.fixture
def cohort(monkeypatch):
    def _set(name):
        monkeypatch.setenv(paths.COHORT_ENV, name)
    monkeypatch.delenv(paths.COHORT_ENV, raising=False)
    monkeypatch.delenv(PathResolver.env_var(paths.DECKS_OUT), raising=False)
    monkeypatch.delenv("WIDEFIELD_REBUILD_ROOT", raising=False)
    return _set


@pytest.mark.parametrize("machine, base", [("analysis", "M:/MICROSCOPE/Priya"), ("imaging", "N:/MICROSCOPE/Priya"),
                                           ("mac", "/Volumes/Priya")])
def test_every_deck_kind_resolves_under_presentations_pipeline(cohort, machine, base):
    rv = PathResolver(machine=machine)
    for kind in paths.DECK_KINDS:
        assert deck_dir(kind, rv).as_posix() == f"{base}/Widefield/presentations/pipeline/{kind}"


def test_cohort2_decks_are_scoped(cohort):
    cohort("cohort2")
    rv = PathResolver(machine="analysis")
    assert deck_dir("preprocessing", rv).as_posix() == \
        "M:/MICROSCOPE/Priya/Widefield/presentations/pipeline/cohort2/preprocessing"


def test_unknown_kind_refused(cohort):
    with pytest.raises(ValueError):
        deck_dir("labcams", PathResolver(machine="analysis"))


def test_history_keeps_the_tag_folder_and_falls_back_beside_an_explicit_out(cohort, tmp_path):
    rv = PathResolver(machine="analysis")
    root = Path("M:/MICROSCOPE/Priya/Widefield/presentations/pipeline")
    assert deck_history_dir(root / "analysis" / "x.pptx", rv) == root / "history"
    assert deck_history_dir(root / "preprocessing" / "x.pptx", rv) == root / "history"
    assert deck_history_dir(root / "analysis" / "camgate" / "x.pptx", rv) == root / "history" / "camgate"
    assert deck_history_dir(tmp_path / "x.pptx", rv) == tmp_path / "deck_history"


def test_default_deck_writers_use_deck_dir():
    """The CLI defaults name the decks_out kind, not labcams / behavior_out / LP_analysis (a regression guard)."""
    repo = Path(__file__).resolve().parents[1] / "wfield_local"
    for mod, kind in [("locanmf_analysis_deck", "analysis"), ("nightly_figs", "analysis"), ("deck_split", "analysis"),
                      ("preprocess_deck", "preprocessing"), ("behavior_deck", "behavior"),
                      ("spout_behavior", "behavior"), ("behavior_clip_deck", "behavior"),
                      ("orofacial_deck", "orofacial")]:
        assert f'deck_dir("{kind}"' in (repo / f"{mod}.py").read_text(encoding="utf-8"), mod
