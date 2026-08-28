"""Post-stroke EPOCHS: pre / acute / subacute, one definition for every figure that uses them.

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

WHERE THE BOUNDARIES CAME FROM. The rule is behavioural: acute = the days on which far_R accuracy is
below 25% of that animal's pre-stroke baseline. `verify_against_behaviour` re-derives the boundaries
from the behaviour tables and reports whether they agree with the stored specification; it is a CHECK,
not the source. The specification is stored explicitly because a rule evaluated at figure time would
silently redraw every epoch boundary the moment a session was added or a behaviour metric changed --
and an epoch boundary that moves under a published figure is exactly the class of failure this
codebase keeps finding (`curated_dates`, the frozen models, the 0817 pooling).

ADDING "CHRONIC" is one entry per animal in `EPOCH_SPEC` plus its name in `EPOCHS`; nothing else in
the pooled figures needs to know.
"""
from __future__ import annotations

import datetime as dt

from wfield_local import config

#: Ordered, and the order is what every pooled figure plots left to right.
EPOCHS = ("pre", "acute", "subacute")

#: Per animal: (acute days inclusive, first subacute day), in DAYS SINCE `config.stroke_date`.
#: Priya, 2026-08-28. A day outside both ranges is deliberately unassigned -- see `epoch_of`.
EPOCH_SPEC = {
    "PS92": {"acute": (1, 5), "subacute_from": 7},
    "PS93": {"acute": (1, 4), "subacute_from": 5},
    "PS94": {"acute": (1, 7), "subacute_from": 9},
    "PS95": {"acute": (1, 1), "subacute_from": 2},
}

#: The behavioural rule the boundaries were derived from, for `verify_against_behaviour`.
ACUTE_RULE = "far_R accuracy < 25% of that animal's pre-stroke baseline"
ACUTE_FRACTION = 0.25
RULE_POSITION = "far_R"


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
    """``'pre'`` / ``'acute'`` / ``'subacute'``, or None.

    None means "this session belongs to no epoch", and it is returned rather than guessed in three
    distinct cases that must not be conflated:

      * the session's phase is ``'excluded'`` -- PS92_0817 and PS93_0817, a lesion attempt that
        produced no deficit, neither baseline nor post-stroke (`config.session_phase`);
      * the animal has no epoch specification;
      * the day falls in the gap between the acute range and the first subacute day. No CURRENT
        session does (days 6 and 8 were simply not recorded), but a future one could, and assigning
        it to whichever neighbour is nearer would invent a boundary Priya did not set.
    """
    animal = config.animal_of(label)
    phase = config.session_phase(animal, label.split("_")[-1])
    if phase == "pre":
        return "pre"
    if phase != "post":
        return None
    spec = EPOCH_SPEC.get(animal)
    n = days_since_stroke(label)
    if spec is None or n is None:
        return None
    lo, hi = spec["acute"]
    if lo <= n <= hi:
        return "acute"
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
            if stored[lab] is not None and derived[lab] != stored[lab]:
                bad.append({"label": lab, "day": days_since_stroke(lab), position: acc,
                            "threshold": thresh, "derived": derived[lab], "stored": stored[lab]})
        out[animal] = {"agree": not bad, "threshold": thresh, "baseline_n": len(base),
                       "derived": derived, "stored": stored, "disagreements": bad}
    return out
