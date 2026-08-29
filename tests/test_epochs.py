"""Epoch assignment is by DAYS SINCE STROKE, and reproduces the specification exactly.

Priya specified (2026-08-28): PS92 acute 1-5 / subacute 7+, PS93 acute 1-4 / subacute 5+,
PS94 acute 1-7 / subacute 9+, PS95 acute 1 / subacute 2+.

READ AS SESSION INDEX THAT IS IMPOSSIBLE. PS94 has eight post-stroke sessions and no ninth, so
"subacute 9+" would be empty and the animal would contribute nothing to a subacute panel -- silently,
since an empty arm and an unpopulated one look identical on a pooled figure. Read as DAYS SINCE
STROKE it is exact: PS94's day 9 is 0825 and day 11 is 0827.

The two readings also disagree about the lesion dates. PS94/PS95 were lesioned 0816 and PS92/PS93 on
0817 (their 0816 attempt produced no deficit and was redone after the 0817 session), so the same
calendar date is a different post-stroke day for different animals. A session-index reading would
align 0818 across all four; a day reading does not, and the day reading is the one that makes the
recovery stage comparable.
"""
import datetime as dt

import pytest

from wfield_local import config, epochs

#: Priya's specification, transcribed as {animal: (acute days, first subacute day)}.
SPEC = {"PS92": (range(1, 6), 7), "PS93": (range(1, 5), 5),
        "PS94": (range(1, 8), 9), "PS95": (range(1, 2), 2)}


def test_the_stored_spec_is_the_one_priya_gave():
    for animal, (acute, sub) in SPEC.items():
        s = epochs.EPOCH_SPEC[animal]
        assert s["acute"] == (min(acute), max(acute)), animal
        assert s["subacute_from"] == sub, animal


def test_every_post_stroke_session_lands_in_exactly_one_epoch():
    """No CURRENT session falls in the gap between the acute range and the first subacute day --
    days 6 and 8 simply were not recorded. If a future session lands there this fails, which is
    correct: assigning it to the nearer neighbour would invent a boundary nobody set."""
    unassigned = [l for l in config.pooled_labels() if epochs.epoch_of(l) is None]
    assert not unassigned, f"sessions in no epoch: {unassigned}"


def test_days_are_counted_from_each_animals_own_lesion():
    """PS94/PS95 were lesioned a day before PS92/PS93, so the same date is a different post-stroke
    day. Counting from a shared date would shift two animals by one and move PS95 out of acute."""
    assert epochs.days_since_stroke("PS94_0817") == 1
    assert epochs.days_since_stroke("PS92_0817") == 0      # lesion follows that day's session
    assert epochs.days_since_stroke("PS92_0818") == 1
    assert epochs.epoch_of("PS94_0817") == "acute"
    assert epochs.epoch_of("PS95_0818") == "subacute"      # day 2, PS95 recovers fastest


def test_it_reproduces_the_day_lists_priya_wrote():
    """The whole specification, re-derived from dates and checked day by day."""
    for animal, (acute, sub) in SPEC.items():
        sd = config.stroke_date(animal)
        base = dt.date(2026, int(sd[:2]), int(sd[2:]))
        for lab in config.pooled_labels(animal):
            if config.session_phase(animal, lab.split("_")[-1]) != "post":
                continue
            mmdd = lab.split("_")[-1]
            n = (dt.date(2026, int(mmdd[:2]), int(mmdd[2:])) - base).days
            want = "acute" if n in acute else ("subacute" if n >= sub else None)
            assert epochs.epoch_of(lab) == want, f"{lab} (day {n})"


def test_the_excluded_sessions_belong_to_no_epoch():
    """PS92_0817 / PS93_0817 are a lesion attempt that produced no deficit -- neither baseline nor
    post-stroke. They must not be swept into 'acute' by being early."""
    for lab in ("PS92_0817", "PS93_0817"):
        assert config.session_phase(config.animal_of(lab), lab[-4:]) == "excluded"
        assert epochs.epoch_of(lab) is None
    pooled = {l for v in epochs.labels_by_epoch().values() for l in v}
    assert not (pooled & {"PS92_0817", "PS93_0817"})


def test_the_pool_is_the_sanctioned_one():
    """`labels_by_epoch` builds from config.pooled_labels, so the 0817 exclusion is by construction
    rather than by epoch_of having to catch it -- the same single pool definition the frozen decoder
    is keyed on."""
    import inspect
    assert "config.pooled_labels" in inspect.getsource(epochs.labels_by_epoch)


def test_the_epoch_counts_are_what_the_pooled_figures_will_show():
    """A pooled panel is only interpretable if its n is known, and NO epoch is balanced across
    animals, so any figure must state its per-animal n rather than imply four equal contributors.

    Asserts that IMBALANCE and the pre/post accounting, which are durable as the cohort grows -- NOT a
    snapshot of the exact counts. The subacute counts increase every time a post-stroke session is
    registered (that stale snapshot is what blocked the nightly registration push, 2026-08-28); the
    assignment logic itself is pinned by the spec / day-list / no-gap tests above."""
    by = epochs.labels_by_epoch()
    ANIMALS = ("PS92", "PS93", "PS94", "PS95")
    per = {e: {a: sum(1 for l in v if l.startswith(a)) for a in ANIMALS} for e, v in by.items()}
    for e in ("acute", "subacute"):
        assert len(set(per[e].values())) > 1, (
            f"{e} epoch is balanced across animals ({per[e]}); a pooled figure could imply four "
            f"equal contributors, so it must state per-animal n")
    assert per["acute"]["PS95"] == 1              # PS95 acute is spec'd as day 1 only -- one session
    # every acute/subacute label is a post-stroke pooled label; pre is exactly the remainder
    n_post = len(by["acute"]) + len(by["subacute"])
    assert n_post == sum(1 for l in config.pooled_labels()
                         if epochs.epoch_of(l) in ("acute", "subacute"))
    assert len(by["pre"]) == len(config.pooled_labels()) - n_post


def test_the_ordering_is_the_plot_order():
    assert epochs.EPOCHS == ("pre", "acute", "subacute")


def test_the_behavioural_rule_reports_rather_than_reassigns(monkeypatch):
    """The rule is a CHECK. If it silently reassigned, every published epoch boundary would move the
    moment a session was registered or a behaviour metric changed."""
    acc = {}
    for lab in config.pooled_labels():
        e = epochs.epoch_of(lab)
        acc[lab] = {"far_R": 0.8 if e == "pre" else (0.05 if e == "acute" else 0.6)}
    rep = epochs.verify_against_behaviour(acc)
    assert all(r["agree"] for r in rep.values()), rep
    # now make one session disagree; the stored assignment must NOT move
    bad = epochs.labels_by_epoch()["subacute"][0]
    acc[bad] = {"far_R": 0.01}
    rep2 = epochs.verify_against_behaviour(acc)
    animal = config.animal_of(bad)
    assert rep2[animal]["agree"] is False
    assert any(d["label"] == bad for d in rep2[animal]["disagreements"])
    assert epochs.epoch_of(bad) == "subacute", "the rule reassigned instead of reporting"


def test_a_missing_baseline_is_reported_not_guessed():
    rep = epochs.verify_against_behaviour({})
    assert all(r["agree"] is None for r in rep.values())


def test_adding_chronic_needs_one_entry_per_animal():
    """The extension path, asserted so it stays cheap: the epoch names and the per-animal spec are
    the only two places a new epoch appears."""
    import inspect
    src = inspect.getsource(epochs)
    assert "ADDING \"CHRONIC\"" in src or 'ADDING "CHRONIC"' in src
