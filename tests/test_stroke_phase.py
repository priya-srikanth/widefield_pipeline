"""Post-stroke sessions must never enter a pre-stroke pool.

`curated_dates()` is "every registered date minus exclusions", and its docstring says a newly
registered night joins AUTOMATICALLY. That was correct while every night was pre-stroke. From the
moment a lesion exists it is a live hazard, because this one list feeds the joint LocaNMF bases, the
frozen decoder's training pool, and the no-lick pre-stroke reference — the three things the whole
post-stroke comparison is measured against. A single post-stroke session leaking into any of them
destroys the comparison silently, and every number would still look plausible.

These tests exercise the guard with a stroke date SET, because with it null the code path is inert
and a test would pass against the unguarded version.
"""
from __future__ import annotations

import pytest

from wfield_local import config


@pytest.fixture
def lesioned(monkeypatch):
    """Cohort with a lesion after 8/14, the convention in animals.yaml."""
    base = {a: dict(v) for a, v in config.animals().items()}
    for a in base:
        base[a]["stroke_date"] = "20260814"
    monkeypatch.setattr(config, "animals", lambda: base)
    return base


def test_without_a_stroke_date_nothing_changes(monkeypatch):
    """KNOWN-GOOD case: while the cohort is pre-stroke the guard must be inert."""
    base = {a: dict(v) for a, v in config.animals().items()}
    for a in base:
        base[a]["stroke_date"] = None
    monkeypatch.setattr(config, "animals", lambda: base)
    assert config.stroke_cutoff() is None
    assert config.curated_dates() == config.curated_dates(phase="all")


def test_post_stroke_dates_are_excluded_from_the_default_curated_set(lesioned):
    """The regression this exists for: 8/17 must not appear in the pre-stroke pool."""
    pre = config.curated_dates()
    assert "0814" in pre, "the stroke_date session itself is BASELINE and must be kept"
    assert all(d <= "0814" for d in pre), f"post-stroke date leaked into the pre-stroke pool: {pre}"


def test_post_phase_returns_only_post_stroke_dates(lesioned):
    post = config.curated_dates(phase="post")
    assert all(d > "0814" for d in post)
    assert not set(post) & set(config.curated_dates()), "pre and post must not overlap"


def test_all_phase_recovers_the_unguarded_behaviour(lesioned):
    every = config.curated_dates(phase="all")
    assert set(every) == set(config.curated_dates()) | set(config.curated_dates(phase="post"))


def test_cutoff_is_the_EARLIEST_stroke_date_for_a_staggered_cohort(monkeypatch):
    """Pooled references must be safe for every animal, so the cohort cutoff is the earliest.

    A later cutoff would let an animal lesioned earlier contribute post-stroke sessions to a pooled
    pre-stroke reference.
    """
    base = {a: dict(v) for a, v in config.animals().items()}
    for a, d in zip(sorted(base), ("20260814", "20260812", "20260814", "20260814")):
        base[a]["stroke_date"] = d
    monkeypatch.setattr(config, "animals", lambda: base)
    assert config.stroke_cutoff() == "0812"
    assert all(d <= "0812" for d in config.curated_dates())


def test_stroke_date_accepts_both_date_spellings(monkeypatch):
    base = {a: dict(v) for a, v in config.animals().items()}
    first = sorted(base)[0]
    base[first]["stroke_date"] = "20260814"
    monkeypatch.setattr(config, "animals", lambda: base)
    assert config.stroke_date(first) == "0814"
    base[first]["stroke_date"] = "0814"
    assert config.stroke_date(first) == "0814"
    base[first]["stroke_date"] = None
    assert config.stroke_date(first) is None
