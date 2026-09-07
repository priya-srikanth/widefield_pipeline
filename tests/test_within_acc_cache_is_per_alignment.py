"""The within-session pre-stroke band is per ALIGNMENT, not shared across windows.

WHY (2026-09-07, found by Priya from deck slide 115). `_within_accuracy` memoises on
``within_acc__<positions>__k<folds>`` and the alignment was absent, so the pre-cue, post-cue and
post-lick calls for one pre-stroke session all hashed to the SAME entry: whichever ran first served
the other two. In the published `section_g.json` every window carried an identical
``within_pre_band`` -- mean 0.7457034401132404 for PS92 in all of them -- while the post-stroke
values differed correctly, because those are not cached.

WHAT IT COST. G2c ("the PLAN survives, EXECUTION does not") drew the POST-cue band on its pre-cue
panel. Pre-cue's true within-session mean is around 0.39 against the 0.746 being shown, so the
dissociation the figure exists to demonstrate was flattened, and every post-stroke z-score was
measured against the wrong reference.

The computation was never wrong -- the KEY was. `params=None` with everything result-changing
folded into `kind` is the house idiom precisely so this is checkable by eye, and every other cached
kind in the repo already carried the alignment.
"""
import inspect
import re

from wfield_local import poststroke_compare as pc


def test_align_is_a_parameter_of_the_cached_function():
    assert "align" in inspect.signature(pc._within_accuracy).parameters, (
        "_within_accuracy cannot key on the alignment unless it is given it")


def test_the_cache_kind_carries_the_alignment():
    src = inspect.getsource(pc._within_accuracy)
    kinds = re.findall(r'f"(within_acc__[^"]*)"', src)
    assert kinds, "could not find the within_acc cache kind"
    for k in kinds:
        assert "{align}" in k, (
            f"cache kind {k!r} omits the alignment, so pre-cue/post-cue/post-lick collide on one "
            f"entry and one window's band is served for all three")


def test_every_cached_kind_in_poststroke_compare_names_its_alignment():
    """The sibling check: this module's cached results are all per-alignment quantities."""
    src = inspect.getsource(pc)
    for kind in re.findall(r'session_cache\.cached\(\s*[^,]+,\s*f"([^"]+)"', src):
        assert "{align}" in kind, f"cached kind {kind!r} is alignment-dependent but unkeyed on it"
