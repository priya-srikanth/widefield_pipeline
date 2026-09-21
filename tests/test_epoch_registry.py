"""The deck's epoch registry is now importable, so assert what a regex-scrape could not.

WHY THIS FILE EXISTS. `_EPOCH` was a LOCAL inside `build_analysis_deck` until 2026-09-21, so the
only way to ask "what does the deck place?" was to scrape this module's source --
`scripts/deck_figure_coverage.py` still does exactly that for the other two decks. A scrape cannot
see a name built by f-string, and `epoch_grant_figures` builds every one of its names that way
(`name=f"epoch_{key}_frozen_refit_overall_{align}_{variant}"`), which is why `DECISIONS.md`
records the coverage check reporting live figures as orphans.

WHAT IS CHECKED HERE AND WHAT IS NOT. Everything below is derived from SOURCE, so it runs in CI
with no share mounted. The disk half -- which rendered files match no registry entry -- cannot be
a unit test, because it depends on what a nightly happened to produce; that stays in
`deck_figure_coverage.py`, which is a tool for looking, not a gate. Measured on 2026-09-21 for the
record: 402 files in `grant_figures/epoch`, 109 registered, **33 unregistered** (the engineering
handoff said 25; it has grown), and **0 dead registry entries**.
"""
from __future__ import annotations

import ast
import fnmatch
from pathlib import Path

import pytest

from wfield_local.locanmf_analysis_deck import EPOCH_FIGURES

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "wfield_local" / "epoch_grant_figures.py"


def _renderer_name_templates():
    """Every `name=` a figure is written under, with f-string slots collapsed to ``*``.

    The renderer never writes a literal `.png`; it passes a stem as `name=` and a helper appends
    the suffix. Reading the keyword is therefore the only way to know what it emits without
    running it.
    """
    tree = ast.parse(RENDERER.read_text(encoding="utf-8"))
    out = set()
    for n in ast.walk(tree):
        if not (isinstance(n, ast.keyword) and n.arg == "name"):
            continue
        v = n.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out.add(v.value)
        elif isinstance(v, ast.JoinedStr):
            out.add("".join(p.value if isinstance(p, ast.Constant) else "*" for p in v.values))
    return sorted(t for t in out if t.startswith("epoch"))


def test_the_registry_is_importable_and_populated():
    """The whole point of 2.3. If this fails, something put it back inside a function."""
    assert isinstance(EPOCH_FIGURES, tuple)
    assert len(EPOCH_FIGURES) >= 100, len(EPOCH_FIGURES)


def test_every_entry_is_a_well_formed_triple():
    """`(pattern, title, legend)`. The deck unpacks exactly three, and a malformed entry would
    take the whole epoch section down at render time rather than here."""
    bad = [e for e in EPOCH_FIGURES
           if not (isinstance(e, tuple) and len(e) == 3
                   and isinstance(e[0], str) and e[0].endswith(".png")
                   and isinstance(e[1], str) and e[1].strip()
                   and (e[2] is None or isinstance(e[2], str)))]
    assert not bad, bad[:3]


def test_no_pattern_is_registered_twice():
    """A duplicate places the same figure on two slides, which has happened in this deck before
    and reads as a rendering bug rather than a registry one."""
    pats = [e[0] for e in EPOCH_FIGURES]
    dupes = sorted({p for p in pats if pats.count(p) > 1})
    assert not dupes, dupes


def test_the_renderer_emits_nothing_the_registry_has_never_heard_of():
    """**THE CHECK THE LIFT WAS FOR.**

    A figure family the deck was never told about is invisible from both ends: the deck's
    missing-figure counter only counts slides it TRIED to build, and `if not path.exists():
    continue` is silent by construction. Two such families were found by hand on 2026-08-24 and a
    third on 2026-09-09; finding them by hand is not a method.

    Matching is glob-against-glob in both directions because BOTH sides carry wildcards -- the
    registry entry may be a glob the deck expands with `_epoch.glob`, and the renderer template has
    `*` wherever an f-string slot was. That is looser than exact matching and deliberately so: the
    failure this guards against is a whole family with no entry at all, not a per-file placement.
    """
    reg = [e[0] for e in EPOCH_FIGURES]
    orphans = []
    for t in _renderer_name_templates():
        cand = t + ".png"
        if not any(fnmatch.fnmatch(cand, p) or fnmatch.fnmatch(p, cand) for p in reg):
            orphans.append(cand)
    assert not orphans, (
        "epoch_grant_figures renders these and the deck has no entry for them, so they will "
        "never be placed and nothing will say so:\n  " + "\n  ".join(orphans))


def test_the_renderer_templates_were_actually_found():
    """A guard on the guard: if `name=` is renamed, the test above passes vacuously."""
    assert len(_renderer_name_templates()) >= 30


@pytest.mark.parametrize("entry", EPOCH_FIGURES, ids=lambda e: e[0])
def test_a_legend_is_prose_not_a_placeholder(entry):
    """Every slide carries a legend or inherits `_CI_LEGEND`. An entry with a one-word legend is
    almost always a stub someone meant to come back to -- and the deck will happily ship it."""
    _pat, _title, legend = entry
    if legend is None:
        return
    assert len(legend.split()) >= 8, legend
