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


# ---------------------------------------------------------------------------------------------
# THREE states, not two: a lesion that did not take leaves sessions belonging to neither phase
# ---------------------------------------------------------------------------------------------
def test_the_real_cohort_state_2026_08_17():
    """PS92/PS93 8/17 is EXCLUDED (lesion 8/16 gave no deficit, redone after that session);
    PS94/PS95 8/17 is POST. Pinned because a two-state model would force them into a pool."""
    assert config.stroke_cutoff() == "0816", "stroke_date is the LESION date, not the last session"
    for a in ("PS92", "PS93"):
        assert config.session_phase(a, "0817") == "excluded"
    for a in ("PS94", "PS95"):
        assert config.session_phase(a, "0817") == "post"
    for a in ("PS92", "PS93", "PS94", "PS95"):
        assert config.session_phase(a, "0814") == "pre", "the stroke_date session is BASELINE"


def test_excluded_sessions_are_in_no_pooled_phase():
    """The property that matters: excluded means absent from BOTH pools, not silently pre."""
    pre, post = config.phase_labels("pre"), config.phase_labels("post")
    for lab in ("PS92_0817", "PS93_0817"):
        assert lab not in pre and lab not in post, f"{lab} leaked into a pooled phase"


def test_phase_labels_resolves_per_animal_not_per_date():
    """8/17 is post for two animals and excluded for two -- a cohort date list cannot say that."""
    post = config.phase_labels("post")
    if post:                                    # once 8/17 is registered
        assert all(l.startswith(("PS94", "PS95")) for l in post if l.endswith("0817"))


def test_pre_stroke_pool_is_unchanged_by_the_lesion_metadata():
    """The reference must be built from exactly the 11 curated pre-stroke dates.

    The last of them is 8/14 while the lesion is 8/16: no sessions fell in between, so the pool is
    the same whichever date is stored. That coincidence is WHY the distinction needs a test -- it
    would hide a wrong stroke_date here and only surface if a same-day session were ever recorded.
    """
    pre = config.curated_dates()
    assert pre[-1] == "0814" and len(pre) == 11
    assert config.stroke_cutoff() == "0816"


# ---------------------------------------------------------------------------------------------
# POST-STROKE SUMMARIES: the excluded sessions may be PROJECTED and shown per-session, but must
# never enter a pooled summary (Priya, 2026-08-18).
# ---------------------------------------------------------------------------------------------
def test_post_stroke_summary_label_set_excludes_the_failed_lesions():
    """`phase_labels("post")` is the ONLY sanctioned way to build a post-stroke pool.

    The tempting shortcut -- select sessions by DATE, e.g. label.endswith("0817") -- sweeps in
    PS92 and PS93, whose 8/17 lesion produced no deficit. They are fine to project onto the joint
    bases and fine to show individually; they must not dilute a post-stroke summary.
    """
    post = set(config.phase_labels("post"))
    by_date = {s["label"] for s in config.load_sessions() if s["label"].endswith("0817")}
    # the 0817 sessions that ARE post-stroke are exactly PS94/PS95; PS92/PS93 were lesioned that
    # evening, after their session. Asserted on the 0817 subset so registering later dates (8/18
    # onward, where every animal is post-stroke) does not break the guard.
    assert {l for l in post if l.endswith("0817")} == {"PS94_0817", "PS95_0817"}
    leaked = (by_date - post) & {"PS92_0817", "PS93_0817"}
    assert leaked, "fixture check: the date-based shortcut should differ from the phase-based one"
    for lab in ("PS92_0817", "PS93_0817"):
        assert lab not in post, f"{lab} would dilute a post-stroke summary"


def test_excluded_sessions_are_still_registered_and_projectable():
    """Excluded from POOLS, not from the pipeline: they must still resolve as sessions so their
    per-session figures and joint projections can be produced."""
    labs = {s["label"] for s in config.load_sessions()}
    for lab in ("PS92_0817", "PS93_0817"):
        assert lab in labs, f"{lab} must remain registered so it can be projected and plotted"


# ---------------------------------------------------------------------------------------------
# The MIRROR-IMAGE hazard: a post-stroke session silently dropped from a post-stroke analysis.
#
# The tests above guard the direction everyone thinks of -- post-stroke data leaking into a
# pre-stroke pool. The opposite leak is just as damaging and much quieter. `evoked_amplitude`
# curated its session list with a hardcoded `set(curated_dates()) | {"0817"}`, which was correct on
# the day it was written and wrong the moment 8/18 was registered: PS92 and PS93, whose EFFECTIVE
# lesion is 8/18, produced no post-stroke row at all, and PS94/PS95 were quietly truncated to day 1.
# Nothing failed -- the summary just printed "--- PS92 ---" with nothing under it.
#
# Curation exists to keep noisy early sessions out of the REFERENCE BAND. Applied to a post-stroke
# session it deletes the measurement instead.
# ---------------------------------------------------------------------------------------------

def test_curation_never_drops_a_post_stroke_session(lesioned, monkeypatch):
    from wfield_local import evoked_amplitude as ea

    labels = ["PS94_0812", "PS94_0813", "PS94_0817", "PS94_0818"]
    monkeypatch.setattr(config, "load_sessions", lambda: [{"label": x} for x in labels])
    # curation that admits ONLY pre-stroke dates -- the situation the bug created
    monkeypatch.setattr(config, "curated_dates", lambda *a, **k: ["0812", "0813"])

    kept = {s["label"] for s in ea.sessions_to_measure(curated_only=True)}
    non_pre = {x for x in labels if config.session_phase(x[:4], x.split("_")[-1]) != "pre"}
    assert non_pre, "fixture must contain at least one non-pre session or this proves nothing"
    assert non_pre <= kept, f"curation dropped post-stroke sessions: {sorted(non_pre - kept)}"


def test_curation_still_filters_the_pre_stroke_reference(lesioned, monkeypatch):
    """The other half: curation must keep doing its job on the PRE side.

    PS95_0605 has a mean |amplitude| of 16.3 against ~0.53 everywhere else; including it put PS95's
    band at [0.15, 18.09], inside which no post-stroke value could ever fall. A fix that spared
    post-stroke dates by disabling curation entirely would silently reintroduce that.
    """
    from wfield_local import evoked_amplitude as ea

    labels = ["PS95_0605", "PS95_0812", "PS95_0817"]
    monkeypatch.setattr(config, "load_sessions", lambda: [{"label": x} for x in labels])
    monkeypatch.setattr(config, "curated_dates", lambda *a, **k: ["0812"])

    kept = {s["label"] for s in ea.sessions_to_measure(curated_only=True)}
    assert "PS95_0605" not in kept, "non-curated PRE-stroke date must still be excluded"
    assert {"PS95_0812", "PS95_0817"} <= kept
