"""Post-stroke EPOCHS: pre / acute / subacute / chronic, one definition for every figure using them.

Priya, 2026-08-28, mirroring `stroke_orofacial_pipeline`: pooled cross-animal figures should stratify
post-stroke sessions by RECOVERY STAGE rather than plot a linear time axis. A linear axis makes four
animals with different session cadences and different lesion dates incomparable at every x position;
epochs make them poolable.

THE UNIT IS DAYS SINCE THAT ANIMAL'S OWN STROKE, NOT SESSION INDEX. This is the whole reason the
definition lives in one place. The two are not the same and quietly disagree:

  * PS92 and PS93 have SEVEN post-stroke sessions, PS94 and PS95 have EIGHT.
  * The lesion dates differ -- PS94/PS95 on 0816, PS92/PS93 on 0817 (`config.stroke_date`), because
    PS92/PS93's 0816 attempt produced no deficit and was redone after the 0817 session.
  * Sessions are not daily. There is no session on PS92's day 6, nor on PS94's days 6 or 8.

Read as session INDEX, Priya's specification ("PS94 acute 1-7, subacute 9+") is impossible -- PS94
has eight sessions and no ninth. Read as DAYS SINCE STROKE it is exact, and the apparent gaps are
simply days nobody recorded. Verified below and pinned in `tests/test_epochs.py`.

WHERE THE BOUNDARIES COME FROM, and it differs by epoch since 2026-09-07:

  * ACUTE / SUBACUTE are DECLARED in `configs/animals.yaml`, beside each animal's `stroke_date`.
    The rule is behavioural -- acute = the days on which far_R accuracy is below 25% of that
    animal's pre-stroke baseline -- and `verify_against_behaviour` re-derives it and reports
    agreement, but it is a CHECK, not the source. It currently reproduces the declared boundaries
    exactly on all four animals, so deriving them would change nothing today while adding a way for
    a published acute boundary to move.
  * CHRONIC is DERIVED from behaviour every run (Priya, 2026-09-07: *"I'd like the pipeline to run
    the epoch definitions and just determine if we have met 'chronic' criteria, order sessions into
    epochs appropriately, and analyze"*). See the DERIVED BOUNDARIES section below.

The standing objection to deriving a boundary is that it can move between two runs and silently
redraw published panels -- the failure class this codebase keeps finding (`curated_dates`, the
frozen models, the 0817 pooling). Deriving does not remove that risk, so it is answered directly
rather than avoided: every run writes the boundaries it used to `epoch_boundaries.json` beside the
deck, the nightly diffs that file against the previous run and logs every boundary that moved, and
`WIDEFIELD_EPOCHS_PINNED=1` reproduces an older figure set under `EPOCH_SPEC` exactly.

THE PIPELINE NEVER WRITES `configs/animals.yaml`. It is version-controlled, both machines push
`main`, and most of its value is hand-written comments a YAML dump would delete. So the declared
`chronic_from` there can fall behind what behaviour now implies. `epoch_audit` reports that every
run and prints the lines to paste; promoting is a deliberate human step.

THE PRECEDENCE, because "fallback" is too vague to act on:

    pinned                        -> `configs/animals.yaml`
    normal run                    -> derived, then written to `epoch_boundaries.json`
    derivation unavailable/crashed-> the LAST DERIVED `epoch_boundaries.json`
    no artifact at all            -> `configs/animals.yaml`

A stale `chronic_from` in the YAML therefore does NOT silently revert tonight's epochs when a run
fails -- the last good derivation carries. It bites in two places only, both about REPRODUCING a
figure set rather than producing one: a fresh clone with no artifact, and `WIDEFIELD_EPOCHS_PINNED=1`.

WHERE EACH THING LIVES, since there are now three files and they are easy to confuse:
  * `configs/defaults.yaml epochs.*`   -- the RULE (thresholds, tolerances). Human-edited.
  * `configs/animals.yaml <an>.epochs` -- the per-animal FALLBACK boundaries. Human-edited.
  * `<labcams>/epoch_boundaries.json`  -- what the last run DERIVED and built its figures on.
                                          Machine-written every run; never hand-edited.

CHRONIC WAS ADDED 2026-09-07, and this docstring used to promise it would cost one entry per animal
in `EPOCH_SPEC` plus its name in `EPOCHS`, with nothing else needing to know. That was not true, and
the four places it was wrong are worth naming for whoever adds a fifth epoch:

  * `epoch_figures.epoch_of_day` keeps its OWN copy of the boundary logic, keyed on day number
    rather than label, and needed the same new branch in the same order
    (`tests/test_chronic_epoch.py` now pins the two copies together for every pooled session);
  * `epoch_figures.EPOCH_GREY` needed a fourth colour, or every chronic bar would have silently
    fallen back to the acute grey;
  * `timecourse_panel`'s `boundaries` argument was a 2-tuple per animal;
  * `verify_against_behaviour` compared derived and stored labels DIRECTLY, so every chronic session
    would have been reported as a disagreement with the acute rule -- a rule that cannot say
    "chronic" and was never meant to adjudicate it.

None of those would have raised. Three would have rendered a wrong figure and the fourth a wrong
audit, which is why the parity test exists rather than a comment asking the next person to be careful.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from wfield_local import config

#: Ordered, and the order is what every pooled figure plots left to right.
EPOCHS = ("pre", "acute", "subacute", "chronic")

#: Per animal: (acute days inclusive, first subacute day, first chronic day), in DAYS SINCE
#: `config.stroke_date`. FROM `configs/animals.yaml`, beside each animal's `stroke_date` -- these
#: are per-animal facts about the experiment and belong with the rest of them, not in a Python dict
#: (CLAUDE.md rule 3; the same reason the hardcoded SESSIONS and ANIMAL_COLOR were retired).
#:
#: `acute` and `subacute_from` ARE THE SOURCE -- nothing derives them, so removing them leaves
#: those two boundaries with no definition at all and `epoch_of` returns None for every post-stroke
#: session. `chronic_from` here is a LAST-RESORT SEED: it is derived every run and published to
#: `epoch_boundaries.json`, and applies only when pinned or when no artifact exists at all (a fresh
#: clone, or a box that has never run). A crashed or behaviour-less run uses the last DERIVED file,
#: not this.
#:
#: ``chronic_from: null`` is an ASSERTION, not an omission: that animal was tested against
#: `CHRONIC_RULE` and has not stabilised. Three of the four have not, for three different reasons --
#: PS93's licking overshot baseline and is still coming back down, PS94 is DECLINING, PS95's hit
#: rate is still climbing. Written explicitly so a reader cannot mistake it for a gap.
EPOCH_SPEC = config.epoch_spec()

_EP = config.defaults().get("epochs") or {}
_CHRONIC = _EP.get("chronic") or {}

#: The behavioural rule the boundaries were derived from, for `verify_against_behaviour`.
ACUTE_FRACTION = float(_EP.get("acute_fraction", 0.25))
RULE_POSITION = str(_EP.get("rule_position", "far_R"))
ACUTE_RULE = (f"{RULE_POSITION} accuracy < {ACUTE_FRACTION:.0%} of that animal's pre-stroke "
              "baseline")

# --- the chronic rule ---------------------------------------------------------------------------
# Priya, 2026-09-07: "maybe we should have a separate intersection (AND) plateau requirement for
# chronic, so both hit rate and lick number reach a plateau?"
#
# chronic = the first session from which BOTH far_R hit rate AND far_R licks/trial are stable, each
# tested on its own with the SAME three conditions, on ENGAGED trials only. The two are NOT averaged
# into a composite. Averaging was tried and rejected: PS94 sits at 91% hit rate while its licking is
# at 45% of baseline, so the two series are measuring different things (accuracy and vigour) and no
# weighting of them is correct -- at a noise-optimal weight the composite is hit rate to within two
# points and the vigour deficit vanishes; at equal weight PS94 reads 68% and its accuracy is
# misrepresented. An AND keeps both visible and requires both.
#
# THE THREE CONDITIONS, each applied to a candidate tail (this session onward):
#
#   1. FLAT       |slope| <= CHRONIC_K_SD x that series' OWN pre-stroke SD, per session.
#                 Scaled per series because licking is ~3.5x noisier than hit rate in units of
#                 fraction-of-baseline; one absolute tolerance would make the lick test ~3.5x
#                 stricter and would mostly report that licking is noisy. At K_SD = 1/3 and a
#                 3-session minimum tail (2 intervals) the permitted drift across the shortest
#                 acceptable window is 2/3 of a baseline SD.
#
#                 K_SD = 0.5 WAS TRIED AND REJECTED, and the reason is worth keeping because the
#                 argument for it was seductive: it makes the permitted drift exactly 1.0 baseline
#                 SD, which reads better than 0.67. It also moves PS93's lick series from failing
#                 by 55% of tolerance to failing by 3.6% -- one session from flipping an animal
#                 into chronic, on a cohort that gains a session most nights. The tidier anchor was
#                 not worth a boundary that fragile. Every other decisive margin is comfortable at
#                 either value; PS93 is the one that discriminates, so it chose the constant.
#   2. RECOVERED  mean of the tail >= CHRONIC_LEVEL_MIN[series] x baseline. Flat is not recovered:
#                 PS94's licking is dead flat at 47% of baseline from day 9, and slope alone calls
#                 that a plateau. "Stably impaired" is not "chronic".
#   3. SETTLED    residual scatter about the fitted line <= CHRONIC_K_RES x pre-stroke SD. A flat
#                 FIT through scattered points is a noisy climb, not a plateau: PS95's hit rate has
#                 a shallow trend from day 2 while swinging 72-105% inside the window.
#
# AND PERSISTENCE: the plateau is the earliest session such that EVERY later start also satisfies
# all three. Without it the predicate is not monotone in the start date -- PS95 passes at day 2 and
# fails at day 3 -- so "first session that passes" latches onto a lucky early window.
#
# TWO KNOWN LIMITATIONS, left in deliberately and recorded here rather than fixed:
#
#   * THE SLOPE IS PER SESSION INDEX, NOT PER DAY. Sessions are not evenly spaced (gaps run 1, 1, 1,
#     1, 2, 2, 2, 4, 3 days), so a 3-session window spans 2-7 calendar days and the qualifying late
#     windows are the widest. In per-day terms the test is ~3.5x more permissive exactly where
#     plateaus get declared, which biases mildly TOWARD calling chronic. Kept because the pre-stroke
#     SD that sets the tolerance is itself session-to-session scatter, so per-session keeps both
#     sides of the comparison in matching units; rescaling only one side would be worse. All four
#     animals share an identical schedule, so nothing is confounded BETWEEN animals.
#   * PS94'S LICK BASELINE IS A POOR YARDSTICK. Its pre-stroke licks/trial scatter by 34% of
#     baseline, 2-4x every other animal, which is why the level bar is a fixed fraction rather than
#     SD-scaled: `1 - 2*SD` would put PS94's threshold at 32% of baseline and pass its clearly
#     impaired 47% licking. The fixed fraction is not consistent in noise units across animals
#     (80% is 0.6 SD for PS94 and 2.3 SD for PS93) and that is the accepted cost.
#
# ENGAGEMENT IS NOT SETTLED SCIENCE HERE. Both series are computed on engaged trials only
# (`spout_behavior.reference_engagement`). Priya, 2026-09-05: post-stroke disengagement "is a real
# and tricky one - I think this is a post-stroke phenomenon but it is hard to decide if I should
# include it in the recovery timeline". Gating EXCLUDES it from the timeline, so an animal that
# recovers accuracy but works fewer trials still reaches chronic. That is a choice, not a fact; PS94
# is the case where it matters most.
#: Every constant below comes from `configs/defaults.yaml epochs.chronic`, so the thresholds are
#: tunable without a code edit and the reasoning for each sits beside the value it justifies.
CHRONIC_K_SD = float(_CHRONIC.get("k_sd", 1.0 / 3.0))
CHRONIC_K_RES = float(_CHRONIC.get("k_res", 1.5))
CHRONIC_MIN_TAIL = int(_CHRONIC.get("min_tail", 3))
CHRONIC_LEVEL_MIN = {k: float(v) for k, v in
                     (_CHRONIC.get("level_min") or {"hit": 0.90, "licks": 0.80}).items()}

#: BUILT FROM THE CONSTANTS, not written out beside them. This string is stamped into
#: `epoch_boundaries.json` and into the deck's section I divider as the rule a figure set claims to
#: be the output of -- so a hand-written copy would go stale the first time a threshold was tuned in
#: YAML, and would then be asserting something false in a published deck.
CHRONIC_RULE = (
    f"{RULE_POSITION} hit rate AND licks/trial both flat (|slope| <= {CHRONIC_K_SD:.4g} x "
    f"pre-stroke SD/session), recovered (>= "
    + " / ".join(f"{100 * CHRONIC_LEVEL_MIN[k]:.0f}%" for k in ("hit", "licks")
                 if k in CHRONIC_LEVEL_MIN)
    + f" of baseline) and settled (residual <= {CHRONIC_K_RES:.4g} x pre-stroke SD) from this "
      f"session onward, on engaged trials")
#: Per series, as a fraction of that animal's pre-stroke baseline. Hit rate uses Priya's original
#: by-eye bar ("> 90% pre-stroke baseline"), which also lands at a consistent 1.3-2.0 SD across
#: animals. Licks are looser in fraction terms because they are intrinsically noisier; 1.00 would
#: demand full lick recovery and reject PS92 at 97%. Empirically the levels barely matter -- every
#: combination from 0.75/0.60 to 0.95/0.90 gives the same four boundaries.

# --- DERIVED boundaries -------------------------------------------------------------------------
# Priya, 2026-09-07: "I'd like the pipeline to run the epoch definitions and just determine if we
# have met 'chronic' criteria, order sessions into epochs appropriately, and analyze."
#
# So `chronic_from` is now DERIVED FROM BEHAVIOUR each run rather than hand-edited. `EPOCH_SPEC`
# remains as the fallback and as the acute/subacute specification, which stays stored: the acute
# rule reproduces its boundaries exactly on all four animals, so deriving it would change nothing
# today while adding a way for a published acute boundary to move.
#
# THE OBJECTION TO DERIVING WAS NEVER "it might be wrong". It was that a boundary can move between
# two runs and silently redraw published panels with no record of why. Deriving does not remove
# that risk, so the risk is answered directly instead:
#
#   * every run WRITES the boundaries it used to `epoch_boundaries.json` beside the deck, so a
#     figure set can always be asked which epochs it was built from;
#   * the nightly DIFFS that file against the previous run and logs every boundary that moved,
#     which is the "record of why" the stored spec used to provide by refusing to change;
#   * `WIDEFIELD_EPOCHS_PINNED=1` disables the whole mechanism and falls back to `EPOCH_SPEC`, for
#     reproducing an older figure set exactly.
#
# IT MUST GO THROUGH A FILE, not a module global. `nightly_figs` runs `grant_figures` and
# `epoch_grant_figures` as SUBPROCESSES via `cli()`, which re-import this module fresh -- a value
# set in the parent does not propagate, and half the deck would then be built on stored boundaries
# and half on derived ones. That failure would render cleanly and be invisible.
BOUNDARIES_FILE = "epoch_boundaries.json"

#: What THIS process resolved, or None if it has not looked yet. `_RESOLVE_TRIED` separates "looked
#: and found nothing" from "has not looked", so a missing file is not re-stat'ed on every call.
_RESOLVED: dict | None = None
_RESOLVE_TRIED: bool = False
_RESOLVED_SOURCE: str | None = None


def pinned() -> bool:
    """True when derivation is disabled and `EPOCH_SPEC` is authoritative.

    Set `WIDEFIELD_EPOCHS_PINNED=1` to rebuild an older figure set under the boundaries it was
    published with, rather than under whatever behaviour now implies.
    """
    return os.environ.get("WIDEFIELD_EPOCHS_PINNED", "") not in ("", "0")


def boundaries_path() -> Path | None:
    """Where the derived boundaries live: beside the deck, on the share both boxes read.

    Overridable with `WIDEFIELD_EPOCH_BOUNDARIES`, which the tests use so the suite never depends
    on -- or is perturbed by -- whatever the last real run wrote.
    """
    override = os.environ.get("WIDEFIELD_EPOCH_BOUNDARIES")
    if override:
        return Path(override)
    try:
        return Path(config.resolver().root("labcams")) / BOUNDARIES_FILE
    except Exception:                                             # noqa: BLE001
        return None                # no resolver on this box -> stored spec, not a crash


def set_resolved(mapping, *, source="explicit") -> None:
    """Install derived boundaries for this process. ``{animal: {"chronic_from": int|None}}``."""
    global _RESOLVED, _RESOLVE_TRIED, _RESOLVED_SOURCE
    _RESOLVED = {a: dict(v) for a, v in (mapping or {}).items()}
    _RESOLVE_TRIED = True
    _RESOLVED_SOURCE = source


def clear_resolved() -> None:
    """Forget any derived boundaries AND the fact that we looked. Mainly for tests."""
    global _RESOLVED, _RESOLVE_TRIED, _RESOLVED_SOURCE
    _RESOLVED, _RESOLVE_TRIED, _RESOLVED_SOURCE = None, False, None


def resolved_source() -> str:
    """``'stored'``, ``'pinned'``, or the path the boundaries were loaded from. For captions."""
    _ensure_resolved()
    if pinned():
        return "pinned"
    return _RESOLVED_SOURCE or "stored"


def save_boundaries(mapping, path=None) -> Path | None:
    """Write the boundaries this run used, so a figure set can be asked what it was built from."""
    path = Path(path) if path is not None else boundaries_path()
    if path is None:
        return None
    # THE RULE AND ITS CONSTANTS TRAVEL WITH THE NUMBERS. A file saying only "PS92: 11" is
    # unfalsifiable a month later -- it cannot be checked against the rule it claims to be the
    # output of, and a constant that changed in between would leave no trace.
    payload = {"chronic_rule": CHRONIC_RULE,
               "constants": {"K_SD": CHRONIC_K_SD, "K_RES": CHRONIC_K_RES,
                             "LEVEL_MIN": CHRONIC_LEVEL_MIN, "MIN_TAIL": CHRONIC_MIN_TAIL},
               "boundaries": {a: dict(v) for a, v in sorted((mapping or {}).items())}}
    from wfield_local import writeguard
    writeguard.assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_boundaries(path=None):
    """``{animal: {...}}`` from the boundaries file, or None if absent/unreadable.

    Unreadable is deliberately the same as absent: a truncated file from a killed run must fall
    back to the stored spec, not abort every figure in the deck.
    """
    path = Path(path) if path is not None else boundaries_path()
    if path is None or not path.exists():
        return None
    try:
        return (json.loads(path.read_text(encoding="utf-8")) or {}).get("boundaries") or None
    except Exception:                                             # noqa: BLE001
        return None


def _ensure_resolved() -> None:
    global _RESOLVE_TRIED, _RESOLVED, _RESOLVED_SOURCE
    if _RESOLVE_TRIED or pinned():
        return
    _RESOLVE_TRIED = True
    p = boundaries_path()
    got = load_boundaries(p)
    if got:
        _RESOLVED = {a: dict(v) for a, v in got.items()}
        _RESOLVED_SOURCE = str(p)


def spec_for(animal: str) -> dict | None:
    """The epoch specification in force for one animal: stored, overlaid with anything derived.

    A MERGE, not a replacement. The derived file carries only `chronic_from`; acute and
    subacute_from continue to come from `EPOCH_SPEC`, and a derived file that somehow lacked a key
    must not delete a boundary that every published figure depends on.
    """
    base = EPOCH_SPEC.get(animal)
    if base is None:
        return None
    _ensure_resolved()
    extra = (_RESOLVED or {}).get(animal)
    if not extra:
        return base
    out = dict(base)
    for k in ("chronic_from",):
        if k in extra:
            out[k] = extra[k]
    return out


def _date(mmdd: str) -> dt.date:
    """MMDD in the study year. The cohort is a single 2026 season; `config` stores MMDD throughout."""
    return dt.date(2026, int(mmdd[:2]), int(mmdd[2:]))


def days_since_stroke(label: str) -> int | None:
    """Days from this animal's OWN lesion to this session, or None if it has no stroke date.

    Positive only after the lesion: `config.stroke_date` is the day the lesion was INDUCED and the
    lesion follows that day's session, so day 0 is still baseline.
    """
    animal = config.animal_of(label)
    sd = config.stroke_date(animal)
    if not sd:
        return None
    return (_date(label.split("_")[-1]) - _date(sd)).days


def epoch_of(label: str) -> str | None:
    """``'pre'`` / ``'acute'`` / ``'subacute'`` / ``'chronic'``, or None.

    None means "this session belongs to no epoch", and it is returned rather than guessed in three
    distinct cases that must not be conflated:

      * the session's phase is ``'excluded'`` -- PS92_0817 and PS93_0817, a lesion attempt that
        produced no deficit, neither baseline nor post-stroke (`config.session_phase`);
      * the animal has no epoch specification;
      * the day falls in the gap between the acute range and the first subacute day. No CURRENT
        session does (days 6 and 8 were simply not recorded), but a future one could, and assigning
        it to whichever neighbour is nearer would invent a boundary Priya did not set.

    CHRONIC IS TESTED BEFORE SUBACUTE and the ranges deliberately overlap: PS92's ``subacute_from``
    is 7 and its ``chronic_from`` is 11, so days 7-10 are subacute and 11+ are chronic. Ordering the
    checks the other way round would make ``chronic_from`` dead code. An animal whose
    ``chronic_from`` is None stays subacute forever, which is the intended reading -- it has not
    stabilised, not that its late sessions are unassigned.
    """
    animal = config.animal_of(label)
    phase = config.session_phase(animal, label.split("_")[-1])
    if phase == "pre":
        return "pre"
    if phase != "post":
        return None
    spec = spec_for(animal)
    n = days_since_stroke(label)
    if spec is None or n is None:
        return None
    lo, hi = spec["acute"]
    if lo <= n <= hi:
        return "acute"
    chronic_from = spec.get("chronic_from")
    if chronic_from is not None and n >= chronic_from:
        return "chronic"
    if n >= spec["subacute_from"]:
        return "subacute"
    return None


def labels_by_epoch(animal: str | None = None) -> dict[str, list[str]]:
    """``{epoch: [session label, ...]}`` over the POOLED set, sorted by date.

    Built from `config.pooled_labels`, so PS92_0817 and PS93_0817 are excluded by construction
    rather than by `epoch_of` having to catch them -- the same single definition of "the sanctioned
    pool" that the frozen decoder is keyed on.
    """
    out: dict[str, list[str]] = {e: [] for e in EPOCHS}
    for lab in config.pooled_labels(animal):
        e = epoch_of(lab)
        if e in out:
            out[e].append(lab)
    return {e: sorted(v, key=lambda x: x.split("_")[-1]) for e, v in out.items()}


def epoch_table() -> list[dict]:
    """One row per pooled session: animal, date, days since stroke, epoch. For captions and audit."""
    rows = []
    for lab in config.pooled_labels():
        rows.append({"label": lab, "animal": config.animal_of(lab), "date": lab.split("_")[-1],
                     "day": days_since_stroke(lab), "epoch": epoch_of(lab)})
    return sorted(rows, key=lambda r: (r["animal"], r["date"]))


def verify_against_behaviour(accuracy_by_session, *, position=RULE_POSITION,
                             fraction=ACUTE_FRACTION):
    """Re-derive the acute/subacute split from behaviour and compare it with `EPOCH_SPEC`.

    ``accuracy_by_session`` maps ``label -> {position_name: accuracy}``. The baseline is that
    animal's mean `position` accuracy over its PRE-stroke sessions; a post-stroke session is acute
    when its accuracy at that position is below ``fraction`` of it.

    Returns ``{animal: {"agree": bool, "derived": {...}, "stored": {...}, "disagreements": [...]}}``.

    IT REPORTS, IT DOES NOT REASSIGN. The stored specification is Priya's call and is what figures
    use; this exists so that if behaviour and specification ever diverge, someone is told rather than
    the figures quietly following whichever the code happened to consult. A rule evaluated at figure
    time would move published epoch boundaries the moment a session was registered.
    """
    out = {}
    for animal, spec in EPOCH_SPEC.items():
        pre = [l for l in config.phase_labels("pre") if config.animal_of(l) == animal]
        base = [accuracy_by_session[l].get(position) for l in pre
                if l in accuracy_by_session and accuracy_by_session[l].get(position) is not None]
        if not base:
            out[animal] = {"agree": None, "note": f"no pre-stroke {position} accuracy available"}
            continue
        thresh = fraction * (sum(base) / len(base))
        derived, stored, bad = {}, {}, []
        for lab in config.pooled_labels(animal):
            if epoch_of(lab) == "pre":
                continue
            acc = (accuracy_by_session.get(lab) or {}).get(position)
            if acc is None:
                continue
            derived[lab] = "acute" if acc < thresh else "subacute"
            stored[lab] = epoch_of(lab)
            # COMPARED ON ACUTE-NESS ONLY, because that is the whole of what this rule decides.
            # `derived` can never say "chronic" -- the acute rule is a single threshold on far_R
            # accuracy and knows nothing about plateaus -- so comparing the labels directly would
            # report every one of PS92's chronic sessions as a disagreement and drown the signal
            # this function exists to raise. `derive_chronic_boundaries` checks the other boundary.
            if stored[lab] is not None and (derived[lab] == "acute") != (stored[lab] == "acute"):
                bad.append({"label": lab, "day": days_since_stroke(lab), position: acc,
                            "threshold": thresh, "derived": derived[lab], "stored": stored[lab]})
        out[animal] = {"agree": not bad, "threshold": thresh, "baseline_n": len(base),
                       "derived": derived, "stored": stored, "disagreements": bad}
    return out


def _pstdev(values) -> float:
    """Population SD. Pure Python so this definition module keeps its two-import footprint."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _fit_line(ys):
    """``(slope, rms residual)`` of a least-squares line through ``ys`` against SESSION INDEX.

    Index, not date. See the "TWO KNOWN LIMITATIONS" note above `CHRONIC_RULE`: the tolerance this
    slope is compared against is a session-to-session SD, so both sides stay in matching units, at
    the cost of treating a 4-day gap like a 1-day one.
    """
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = 0.0 if sxx == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = (sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    return slope, resid


def _plateau_index(series, sd_pre, level_min):
    """Earliest index whose tail -- AND every later tail -- is flat, recovered and settled.

    The persistence requirement is not decoration. The three conditions are not monotone in the
    start index: PS95's hit rate passes at day 2 (slope 2.3% against a 2.5% tolerance) and fails at
    day 3 (2.8%), on a series that runs 84 82 97 84 72 80 92 105 104 and is plainly still climbing.
    "First index that passes" would report day 2. "First index from which it never stops passing"
    reports nothing, which is correct.
    """
    if sd_pre <= 0:
        return None
    cand = [i for i in range(len(series)) if len(series) - i >= CHRONIC_MIN_TAIL]
    ok = {}
    for i in cand:
        tail = series[i:]
        slope, resid = _fit_line(tail)
        ok[i] = (abs(slope) <= CHRONIC_K_SD * sd_pre
                 and sum(tail) / len(tail) >= level_min
                 and resid <= CHRONIC_K_RES * sd_pre)
    for i in cand:
        if ok[i] and all(ok[j] for j in cand if j >= i):
            return i
    return None


def _normalised_series(by_session, animal, position):
    """``(pre_normalised, [(day, normalised), ...])`` for one animal and one behavioural measure.

    Everything is expressed as a fraction of that animal's OWN pre-stroke mean, which is what makes
    the per-series pre-stroke SD a meaningful tolerance. Ratios are NOT capped at 1.0 -- Priya,
    2026-09-07: "changes > 1 are still informative for whether things may be changing." PS93's
    licking overshoots to 129% of baseline and then declines, and that decline is exactly why it
    does not qualify as chronic; clipping would have hidden it.
    """
    pre_labels = [l for l in config.phase_labels("pre") if config.animal_of(l) == animal]
    base = [(by_session.get(l) or {}).get(position) for l in pre_labels]
    base = [v for v in base if v is not None]
    if not base:
        return [], []
    mean = sum(base) / len(base)
    if mean <= 0:
        return [], []
    post = sorted((l for l in config.pooled_labels(animal) if epoch_of(l) != "pre"),
                  key=lambda x: x.split("_")[-1])
    out = []
    for lab in post:
        v = (by_session.get(lab) or {}).get(position)
        day = days_since_stroke(lab)
        if v is not None and day is not None:
            out.append((day, v / mean))
    return [v / mean for v in base], out


def derive_chronic_boundaries(hit_by_session, licks_by_session, *, position=RULE_POSITION):
    """Re-derive each animal's chronic boundary from behaviour and compare it with `EPOCH_SPEC`.

    ``hit_by_session`` and ``licks_by_session`` both map ``label -> {position_name: value}``, the
    same shape `verify_against_behaviour` takes. Both must already be restricted to ENGAGED trials;
    this function cannot tell whether they were, which is why `CHRONIC_RULE` says so and
    `poststroke_behaviour` gates both before calling.

    Returns ``{animal: {"agree": bool, "derived_day": int|None, "stored_day": int|None,
    "hit": {...}, "licks": {...}}}``, where each per-series dict carries the plateau day, the
    pre-stroke SD that set its tolerances, and the level it plateaued at.

    IT REPORTS, IT DOES NOT REASSIGN, for the same reason `verify_against_behaviour` does not: a
    rule evaluated at figure time would move a published epoch boundary the moment a session was
    added. That risk is sharper for chronic than for acute, because chronic sits at the END of the
    series where every new session lands -- PS93 and PS95 are both close enough to qualifying that a
    few more sessions could flip them, and they must flip by someone editing `EPOCH_SPEC` and
    re-running, not by a figure quietly redrawing itself.
    """
    out = {}
    for animal, spec in EPOCH_SPEC.items():
        per_series, days = {}, []
        for name, table in (("hit", hit_by_session), ("licks", licks_by_session)):
            pre, post = _normalised_series(table, animal, position)
            if not pre or len(post) < CHRONIC_MIN_TAIL:
                per_series[name] = {"day": None, "note": f"no usable {name} series"}
                days.append(None)
                continue
            sd = _pstdev(pre)
            idx = _plateau_index([v for _d, v in post], sd, CHRONIC_LEVEL_MIN[name])
            day = post[idx][0] if idx is not None else None
            tail = [v for _d, v in post[idx:]] if idx is not None else []
            per_series[name] = {"day": day, "sd_pre": sd, "n_post": len(post),
                                "level": (sum(tail) / len(tail)) if tail else None}
            days.append(day)
        # AND: chronic begins at the LATER of the two plateaus, and only if BOTH exist.
        derived = max(days) if all(d is not None for d in days) else None
        stored = spec.get("chronic_from")
        out[animal] = {"agree": derived == stored, "derived_day": derived, "stored_day": stored,
                       **per_series}
    return out
