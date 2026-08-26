"""The shared grant bundle is built once per (animal, alignment), not once per figure.

MEASURED 2026-08-26. `_collect_7` appears 14 times in grant_figures and loops over 4 animals, so a
full render called `_pooled_bundle` 56 times -- while only 4 animals x 3 alignments = 12 are
distinct. Every build loads the joint basis, PROJECTS ~18 sessions onto it and re-derives the
engagement gate, which is the dominant cost of figures 6, 6b, 6d, 7, 7b, 7d, 8, 8b, 8d, 8e.

Nothing between two calls can change the result within one process, so the other 44 builds were pure
repetition -- which is why a layout iteration on those families cost hours.

WHY THIS IS ALSO A CORRECTNESS IMPROVEMENT, not only a speed one. `_pooled_bundle`'s own docstring
says it was extracted because the code was "character-identical" in two figures, and "a third and
fourth copy is how two figures that claim to describe the same trials quietly stop doing so". A memo
finishes that argument: the figures no longer merely compute the same bundle, they hold the same
object, so they cannot disagree even in principle.

IN-PROCESS, NOT ON DISK, deliberately. The bundle holds pooled feature matrices for every session --
persisting it would write hundreds of MB per key and reintroduce the staleness question. A render is
one process, so an in-process memo captures the whole duplication with none of that.
"""
import inspect

from wfield_local import grant_figures as g


def test_the_bundle_is_memoised():
    src = inspect.getsource(g._pooled_bundle)
    assert "if key in _BUNDLE_CACHE" in src, "no cache lookup"
    assert "_BUNDLE_CACHE[key] = bundle" in src, "result is never stored"


def test_the_key_separates_animal_and_alignment():
    """One key per (animal, align). Collapsing either would serve one animal's trials for another --
    the late-binding-closure bug this module already documents, in a different disguise."""
    src = inspect.getsource(g._pooled_bundle)
    assert "key = (an, align)" in src, "the cache key is not (animal, alignment)"


def test_no_path_returns_without_storing():
    """A `return` between the lookup and the store would silently disable the memo for that branch,
    leaving a figure paying full cost while the counter said it was cached."""
    src = inspect.getsource(g._pooled_bundle)
    returns = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("return")]
    assert returns, "no return found; test needs updating"
    for r in returns:
        assert "bundle" in r or "_BUNDLE_CACHE" in r, f"return bypasses the cache: {r}"


def test_cache_is_a_module_level_dict():
    assert isinstance(g._BUNDLE_CACHE, dict)


def test_the_duplication_this_removes_is_real():
    """Pins the CONDITION, so the test keeps its meaning: many _collect_7 call sites against few
    distinct bundles is what makes the memo worth having."""
    src = inspect.getsource(g)
    calls = src.count("_collect_7(")
    assert calls >= 8, (
        f"_collect_7 is called only {calls} times; if the module was restructured, re-measure "
        f"whether the memo still pays for itself")
