"""Session-dot shapes (cat_points.animal_marker): one shape per animal, read from the cohort animals files.

Code review 2026-11-04: the old name-hash fallback gave PS118 and PS121 the same marker ('>'), so two cohort-2 animals
were indistinguishable in every session-dot figure. Markers are now explicit `marker:` keys in configs/animals.yaml
and configs/cohort2/animals.yaml. Cohort-1 shapes are pinned to what published figures used.
"""
from __future__ import annotations

import pytest

from wfield_local import cat_points as cp
from wfield_local import config

#: the shapes every cohort-1 figure was drawn with before the yaml key existed (cat_points / orofacial_cohort dicts)
COHORT1_PUBLISHED = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
#: what the old hash rule gave cohort 2 (PS121 changes: it collided with PS118)
COHORT2_OLD_HASH = {"PS118": ">", "PS120": "<", "PS121": ">", "PS122": "v"}


def _cohorts():
    return config._load_file("cohorts.yaml")["cohorts"]


def test_cohort1_markers_unchanged():
    assert {a: cp.animal_marker(a) for a in COHORT1_PUBLISHED} == COHORT1_PUBLISHED
    assert cp.ANIMAL_MARKER == COHORT1_PUBLISHED


def test_cohort1_matches_orofacial_cohort_dict():
    oc = pytest.importorskip("wfield_local.orofacial_cohort")
    assert {a: cp.animal_marker(a) for a in oc.ANIMAL_MARKER} == oc.ANIMAL_MARKER


@pytest.mark.parametrize("name", sorted(config._load_file("cohorts.yaml")["cohorts"]))
def test_markers_explicit_and_distinct_within_cohort(name):
    spec = _cohorts()[name]
    animals = spec["animals"]
    meta = config._load_file(spec["animals_file"])["animals"]
    missing = [a for a in animals if not (meta.get(a) or {}).get("marker")]
    assert not missing, f"{name}: give these animals an explicit `marker:` in {spec['animals_file']}: {missing}"
    shapes = [cp.animal_marker(a) for a in animals]
    assert len(set(shapes)) == len(shapes), f"{name}: duplicate session-dot shapes {dict(zip(animals, shapes))}"


def test_markers_distinct_across_cohorts():
    """Not required by the never-pool rule, but cheap: a figure that ever shows both cohorts stays readable."""
    allm = cp.configured_markers()
    assert len(set(allm.values())) == len(allm), allm


def test_cohort2_only_the_colliding_animal_moved():
    new = {a: cp.animal_marker(a) for a in COHORT2_OLD_HASH}
    changed = {a for a in new if new[a] != COHORT2_OLD_HASH[a]}
    assert changed == {"PS121"}


def test_unlisted_animal_falls_back_to_hash_rule():
    a = "PS999"
    assert a not in cp.configured_markers()
    assert cp.animal_marker(a) == cp._SPARE[sum(map(ord, a)) % len(cp._SPARE)]
