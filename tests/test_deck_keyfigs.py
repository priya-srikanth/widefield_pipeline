"""The rebuild deck's page key figures and section K captions (Priya 11-17) are well-formed: every entry names a page,
a path under the rebuild root (or the share above it), and a caption that states its provenance (window, licks,
baseline / pre, statistic, n) in American spelling. Pure data checks: no share access."""
import re

import pytest

from wfield_local import deck_keyfigs as K
from wfield_local import deck_registry as R

BRITISH = re.compile(r"\b(colour|behaviour(?!_hits)|centre|normalis|analyse[sd]?\b|modell)", re.I)


@pytest.mark.parametrize("e", K.KEY_FIGURES_PAGES, ids=lambda e: f"{e[0][:6]}-{e[2]}")
def test_page_key_figure_entry(e):
    page, url, lab, base, path, title, licks, cap, what = e
    assert page in ("Spout (Target) Coding", "State-Space Recovery", "Licking Kinematics")
    assert url.startswith("https://claude.ai/artifact/") and lab.startswith("Fig ")
    assert base in ("rebuild", "share") and not path.startswith(("M:", "/", "C:"))
    assert base == "share" or "rebuild" in what
    assert licks.startswith("licks: ") and title
    low = cap.lower()
    assert "pre" in low and ("licks" in low or "lick" in low) and ("session" in low)
    assert not BRITISH.search(cap.replace("behaviour_hits_by_spout", "")), BRITISH.search(cap)


def test_pages_grouped_in_order():
    order = [e[0] for e in K.KEY_FIGURES_PAGES]
    firsts = [p for i, p in enumerate(order) if i == 0 or order[i - 1] != p]
    assert firsts == ["Spout (Target) Coding", "State-Space Recovery", "Licking Kinematics"]


def test_enl_figure_is_25_primary():
    """Coordinator 11-17: the 1.0 s ENL figure is Fig 2.5, the primary window (2.6 = the 1.5 s window-length check)."""
    labs = {e[2]: e for e in K.KEY_FIGURES_PAGES if e[0] == "Spout (Target) Coding"}
    assert "Fig 2.5" in labs and "Fig 2.8" not in labs and "primary" in labs["Fig 2.5"][5]


def test_goal_alternates_cover_the_three_missing_key_figures():
    assert set(K.KEY_FIGURES_REBUILD_ALT) == {"Key figure 2", "Fig 12.2", "Fig 14.18"}
    goal = {k[2] for k in R.KEY_FIGURES}
    assert set(K.KEY_FIGURES_REBUILD_ALT) <= goal
    for rel, cap in K.KEY_FIGURES_REBUILD_ALT.values():
        assert "{T}" in rel and "Script:" in cap and "pre" in cap


def test_section_k_rebuild_captions_cover_registered_slots():
    slots = {r.rsplit("/", 1)[-1] for _root, r, _t, _l in R.LP_GATED_FIGURES}
    assert set(R.LP_GATED_REBUILD) <= slots
    for name, (rel, cap) in R.LP_GATED_REBUILD.items():
        assert rel is None or "{T}" in rel
        assert not cap.startswith("__"), f"{name}: caption placeholder not filled"
        assert "WHAT IT SHOWS" not in cap and not BRITISH.search(cap)
