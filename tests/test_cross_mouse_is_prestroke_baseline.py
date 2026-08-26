"""Both cross-mouse figures are BASELINES and are scoped to pre-stroke sessions.

`fig_within_animal_consistency` declares itself a reference in its own docstring -- "the within-animal
noise floor a post-stroke change must exceed" -- and was built from every session, 39% of them
post-stroke by 2026-08-26. A floor that includes the post-stroke sessions is inflated by the change it
exists to be exceeded by. The failure runs in the conservative direction (a real effect is harder to
clear) but the reference is still measuring the wrong thing.

`fig_cross_mouse` asks whether the mice DIFFER from each other, motivated by PS93's right orofacial
deficit. Three of its six metrics are lateralisation measures -- 3-way laterality, L-vs-R spout
decodability, SSp-left vs SSp-right -- and DECISIONS 2026-08-19 records lateralisation collapsing
post-stroke in PS94 specifically. Pooling across the lesion means a "between-mouse difference" can be
a between-lesion-severity difference, aimed straight at what the module measures.

THE SCOPE IS DEFINED IN THE MODULE, NOT AT THE CALL SITE. The frozen-decoder contamination survived
eight days because the CLI and the nightly passed different date sets, so whichever path you read
looked defensible. A module that defines its own scope cannot be handed the wrong one.

NOT restricted: the per-session metrics, which are computed within a single session and are unaffected
by which other sessions exist.
"""
import inspect

import pytest

from wfield_local import config, locanmf_cross_mouse as cm


def test_both_figures_default_to_prestroke():
    for fn in (cm.fig_cross_mouse, cm.fig_within_animal_consistency):
        sig = inspect.signature(fn)
        assert "phase" in sig.parameters, f"{fn.__name__} has no phase parameter"
        assert sig.parameters["phase"].default == "pre", (
            f"{fn.__name__} does not default to pre-stroke")


def test_the_default_pool_excludes_post_stroke():
    by = cm._session_labels(None, "pre")
    assert by, "no sessions selected at all"
    post = set(config.phase_labels("post"))
    for mouse, labs in by.items():
        bad = [l for l in labs if l in post]
        assert not bad, f"{mouse}: post-stroke sessions in the baseline pool: {bad}"


def test_post_stroke_is_still_reachable_deliberately():
    """Restricting the DEFAULT must not remove the ability to ask the post-stroke question on
    purpose -- that would trade one blind spot for another."""
    by_all = cm._session_labels(None, "all")
    by_pre = cm._session_labels(None, "pre")
    assert sum(len(v) for v in by_all.values()) > sum(len(v) for v in by_pre.values()), (
        "phase='all' returns no more sessions than pre; the escape hatch does not work")


def test_the_scope_is_decided_inside_the_module():
    """Pinning the lesson, not just the behaviour: a caller must not be able to widen the pool by
    passing a date list, which is exactly how the frozen models stayed contaminated."""
    src = inspect.getsource(cm._session_labels)
    assert 'phase_labels' in src, "the module no longer decides its own phase scope"
    for fn in (cm.fig_cross_mouse, cm.fig_within_animal_consistency):
        body = inspect.getsource(fn)
        assert "_session_labels(" in body, (
            f"{fn.__name__} builds its own session list again instead of going through the one "
            f"place that applies the phase filter")


def test_per_session_metrics_are_not_phase_aware():
    """Each metric is computed within one session; making them phase-aware would be meaningless and
    would break the cache key."""
    src = inspect.getsource(cm._per_session_compute)
    assert "phase" not in src, "per-session computation became phase-aware"
