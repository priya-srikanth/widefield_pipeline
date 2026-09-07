"""The CHRONIC boundary: three conditions, persistence, and an AND across two measures.

Priya, 2026-09-07: "maybe we should have a separate intersection (AND) plateau requirement for
chronic, so both hit rate and lick number reach a plateau?"

Each test below is one of the four ways a naive plateau rule gets this wrong, and every one of them
is taken from a REAL series in this cohort rather than invented -- they are the cases that forced
each condition into `epochs.CHRONIC_RULE`, so a future simplification that drops one fails here with
the animal's own numbers rather than an abstract counterexample.
"""
import pytest

from wfield_local import config, epochs
from wfield_local import epoch_figures as ef

#: PS94 far_R licks/trial as a fraction of its pre-stroke baseline, days 1-18 (engaged trials).
#: Dead flat from day 9 -- and dead flat at 47% of baseline.
PS94_LICKS = [0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.45, 0.58, 0.40, 0.46]
PS94_LICK_SD = 0.341

#: PS95 far_R hit rate as a fraction of baseline, days 1-18. The linear TREND from day 2 onward is
#: shallow, but the series swings 72-105% inside that window: a noisy climb, not a plateau.
PS95_HIT = [0.01, 0.84, 0.82, 0.97, 0.84, 0.72, 0.80, 0.92, 1.05, 1.04]
PS95_HIT_SD = 0.076

#: PS93 far_R licks. Overshoots to 129% of baseline and then declines. Ratios are deliberately NOT
#: capped at 1.0 -- Priya: "changes > 1 are still informative for whether things may be changing."
PS93_LICKS = [0.08, 0.14, 0.09, 0.04, 0.41, 0.84, 0.84, 1.26, 1.29, 1.18]
PS93_LICK_SD = 0.086

#: PS92 far_R hit rate. The one series in the cohort that genuinely plateaus.
PS92_HIT = [0.00, 0.08, 0.02, 0.06, 0.07, 0.50, 0.85, 1.02, 1.02, 1.04]
PS92_HIT_SD = 0.050


def _plateau(series, sd, which="hit"):
    return epochs._plateau_index(series, sd, epochs.CHRONIC_LEVEL_MIN[which])


def test_a_real_plateau_is_found():
    """PS92's hit rate: flat, at baseline, and settled from index 7 (day 11) onward."""
    assert _plateau(PS92_HIT, PS92_HIT_SD) == 7


def test_flat_but_impaired_is_not_a_plateau():
    """THE LEVEL CONDITION. PS94's licking is flatter than PS92's from day 9 -- slope -1.5%/session
    against a tolerance of 17% -- and sits at 47% of baseline throughout. A slope-only rule calls
    that chronic, which would report an animal with a halved lick vigour as recovered."""
    assert _plateau(PS94_LICKS, PS94_LICK_SD, "licks") is None
    # and it is the LEVEL that rejects it, not the slope: drop the bar under 47% and it qualifies
    assert epochs._plateau_index(PS94_LICKS, PS94_LICK_SD, 0.30) is not None


def test_a_flat_fit_through_scattered_points_is_not_a_plateau():
    """THE SETTLED CONDITION, isolated: a series that oscillates hard about a perfectly flat fit.

    Slope is 0.0000 and the level is 104% of baseline, so FLAT and RECOVERED both pass and a
    two-condition rule would call this stable. It alternates 80%-120% every session. Residual
    scatter is the only condition that can see the difference between this and a real plateau."""
    scattered = [1.2, 0.8, 1.2, 0.8, 1.2]
    sd = 0.05
    slope, resid = epochs._fit_line(scattered)
    assert abs(slope) <= epochs.CHRONIC_K_SD * sd, "flat: the fit is dead level"
    assert sum(scattered) / len(scattered) >= epochs.CHRONIC_LEVEL_MIN["hit"], "recovered"
    assert resid > epochs.CHRONIC_K_RES * sd, "settled is the condition that must reject it"
    assert epochs._plateau_index(scattered, sd, epochs.CHRONIC_LEVEL_MIN["hit"]) is None


def test_the_plateau_must_persist():
    """THE PERSISTENCE REQUIREMENT, on the real series that forced it.

    PS95's hit rate satisfies all three conditions at index 1 (day 2) and at NO OTHER index -- the
    predicate is not monotone in the start date. A "first index that passes" rule therefore reports
    PS95 as chronic from post-stroke day 2, on a series running 84 82 97 84 72 80 92 105 104 that
    is plainly still climbing. Requiring every later start to pass as well reports nothing.

    Tested at the 0.80 level bar rather than hit rate's real 0.90, deliberately: at 0.90 the level
    condition also rejects index 1, which would mask whether persistence works at all."""
    def passes(i):
        tail = PS95_HIT[i:]
        slope, resid = epochs._fit_line(tail)
        return (abs(slope) <= epochs.CHRONIC_K_SD * PS95_HIT_SD
                and sum(tail) / len(tail) >= 0.80
                and resid <= epochs.CHRONIC_K_RES * PS95_HIT_SD)

    lucky = [i for i in range(len(PS95_HIT) - 2) if passes(i)]
    assert lucky == [1], f"expected exactly one lucky window, got {lucky}"
    assert epochs._plateau_index(PS95_HIT, PS95_HIT_SD, 0.80) is None, (
        "persistence did not reject a window that passes at one index and fails at every other")


def test_ratios_are_not_capped_at_one():
    """PS93's licking overshoots to 129% of baseline and is coming back DOWN at -4.4%/session.

    That decline is why PS93 is not chronic. Capping ratios at 1.0 would flatten 1.26/1.29/1.18 to
    1.00/1.00/1.00 and hand PS93 a chronic stamp for an above-baseline excursion -- which is the
    proposal Priya rejected: "changes > 1 are still informative for whether things may be
    changing." This pins the uncapped behaviour so a later "tidy-up" cannot reintroduce it."""
    assert _plateau(PS93_LICKS, PS93_LICK_SD, "licks") is None
    capped = [min(1.0, v) for v in PS93_LICKS]
    assert epochs._plateau_index(capped, PS93_LICK_SD, epochs.CHRONIC_LEVEL_MIN["licks"]) is not None, (
        "capping no longer changes the verdict; this test has stopped guarding anything")


def test_one_measure_plateauing_is_not_enough():
    """THE AND. PS93's hit rate plateaus at day 11 and PS95's licking at day 2, but neither animal
    is chronic, because the other measure has not. An OR -- or a weighted composite, which behaves
    like an OR when one term is quiet -- would call both chronic."""
    post = sorted((l for l in config.pooled_labels("PS92") if epochs.epoch_of(l) != "pre"),
                  key=lambda x: x.split("_")[-1])
    pre = [l for l in config.phase_labels("pre") if config.animal_of(l) == "PS92"]
    # THE BASELINE MUST SCATTER, AND BY A REALISTIC AMOUNT. Every tolerance in the rule is a
    # multiple of the pre-stroke SD: a constant baseline gives sd=0 and nothing can ever plateau,
    # and a baseline that scatters by 1% gives a tolerance 5x tighter than PS92's real one, so even
    # PS92's genuine plateau would fail. These reproduce PS92's actual pre-stroke SD of 0.050.
    hit = {l: {"far_R": 0.96 + 0.048 * (1 if i % 2 else -1)} for i, l in enumerate(pre)}
    lick = {l: {"far_R": 5.0 + 0.75 * (1 if i % 2 else -1)} for i, l in enumerate(pre)}
    for i, l in enumerate(post):
        hit[l] = {"far_R": 0.96 * (PS92_HIT[i] if i < len(PS92_HIT) else 1.02)}
        lick[l] = {"far_R": 1.0 + 0.75 * i}          # climbing forever: never flat
    rep = epochs.derive_chronic_boundaries(hit, lick)["PS92"]
    assert rep["hit"]["day"] is not None, "the hit series was supposed to plateau"
    assert rep["licks"]["day"] is None, "the lick series was supposed never to plateau"
    assert rep["derived_day"] is None, "AND requires both"


def test_it_reports_rather_than_reassigns():
    """Same contract as `verify_against_behaviour`, and it matters MORE here: chronic sits at the
    END of the series, where every new session lands. PS93 and PS95 are both close enough to
    qualifying that a few more sessions could flip them, and they must flip by someone editing
    EPOCH_SPEC, not by a nightly run quietly redrawing a published panel."""
    pre = [l for l in config.phase_labels("pre") if config.animal_of(l) == "PS94"]
    post = sorted((l for l in config.pooled_labels("PS94") if epochs.epoch_of(l) != "pre"),
                  key=lambda x: x.split("_")[-1])
    # a behaviour table in which PS94 looks perfectly recovered from its first post-stroke session.
    # The baseline scatters (see the note in the AND test): a flat baseline zeroes every tolerance.
    hit = {l: {"far_R": 0.95 + 0.02 * (i % 2)} for i, l in enumerate(pre)}
    lick = {l: {"far_R": 5.0 + 0.2 * (i % 2)} for i, l in enumerate(pre)}
    for i, l in enumerate(post):
        hit[l] = {"far_R": 0.96}
        lick[l] = {"far_R": 5.1}
    rep = epochs.derive_chronic_boundaries(hit, lick)["PS94"]
    assert rep["derived_day"] is not None, "the fixture was supposed to look recovered"
    assert rep["agree"] is False, "a derived/stored mismatch must be REPORTED"
    assert epochs.EPOCH_SPEC["PS94"]["chronic_from"] is None, "the rule reassigned the stored spec"
    assert epochs.epoch_of(post[-1]) == "subacute", "the rule moved a session"


def test_a_missing_series_is_reported_not_guessed():
    rep = epochs.derive_chronic_boundaries({}, {})
    assert all(r["derived_day"] is None for r in rep.values())
    assert rep["PS92"]["agree"] is False          # PS92 HAS a stored boundary; absent data != None
    assert rep["PS94"]["agree"] is True           # PS94 has none, and none was derived


@pytest.mark.parametrize("label", [l for l in config.pooled_labels()])
def test_the_two_copies_of_the_rule_agree(label):
    """`epoch_figures.epoch_of_day` keeps its own copy of the boundary logic, keyed on day number
    rather than label. Two copies of a rule drift; this pins them together for every pooled session.

    The failure this guards is specific and silent: `epoch_of_day` testing subacute before chronic
    would return "subacute" for every chronic day, so the pooled panels would disagree with
    `epoch_table` and the deck's counts while every figure still rendered."""
    day = epochs.days_since_stroke(label)
    expect = epochs.epoch_of(label)
    if expect == "pre" or day is None:
        return
    assert ef.epoch_of_day(config.animal_of(label), int(day)) == expect, f"{label} day {day}"
