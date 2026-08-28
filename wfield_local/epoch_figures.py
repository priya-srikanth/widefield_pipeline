"""Pooled CROSS-ANIMAL grant figures, stratified by recovery EPOCH instead of a time axis.

Priya, 2026-08-28. Three panels -- pre-stroke, acute, subacute -- pooled over all four animals, sized
to be read at a QUARTER PAGE or smaller.

NOTHING HERE RECOMPUTES ANYTHING. Every population these figures need already exists in
`grant_figures`' collectors, and the collectors already carry the trial-inclusion variant Priya asked
for:

  * `_collect_5c(align, variant)` -> ``{animal: (pre-stroke LOSO record, {day: record})}``, where a
    record is ``(y_true, y_pred, blocks)`` at TRIAL level and ``variant='working'`` is exactly
    "lick plus miss-while-working". Keyed by DAY, which is what an epoch is defined on.
  * `_collect_7(align, variant, min_trials)` -> per-day post trials kept as trials, pre-stroke
    sessions kept separate.
  * `_pooled_bundle(animal, align)` -> the shared joint-basis load behind figures 6, 6b, 7 and 8,
    memoised per (animal, alignment).
  * `_position_metrics(animal, mmdd)` -> per-position ``(hit_rate, ci_lo, ci_hi, n)`` behaviour.

So an epoch figure is an AGGREGATION: group the existing per-day records by `epochs.epoch_of`, then
concatenate. Trial-level records concatenate across animals directly; confusion counts are raw, so
pooling is a sum. That is the same summability that made the early/late split free, and it is why
this module adds no new analysis and cannot disagree with the per-animal figures -- it holds the same
objects.

QUARTER PAGE IS A SIZE CONSTRAINT ON TYPE, NOT ON INCHES. What a reader sees is
``fontsize x (placed width / figure width)``. A figure placed at a quarter page is ~6.2in wide, so a
6.2in figure renders 1:1 and an 11in one halves every label. These figures are therefore SMALL in
inches with LARGE fonts -- the opposite of the instinct, and the same measurement that drove
`coding_cross_` from 22.7in to 10.8in.

DAYS COME FROM `epochs.days_since_stroke`, not from `grant_figures._day`. The two agree for this
cohort and are not the same function: `_day` is month*31 ordering, exact only while every date sits
in a 31-day month, and its own docstring says so. `tests/test_epoch_figures.py` asserts they still
agree, so a September session divides them loudly rather than silently moving an epoch boundary.
"""
from __future__ import annotations

import numpy as np

from wfield_local import config, epochs

#: Panel order, left to right. `epochs.EPOCHS` is the single definition; restated for readability.
PANELS = epochs.EPOCHS

#: Placed at a quarter page (~6.2in on a 13.33in slide). Figures are built at this width so type
#: arrives at ~1:1; see the module docstring.
QUARTER_IN = 6.2

#: Font sizes for a figure that will NOT be scaled down. Roughly 1.6x the per-animal figures', which
#: are built at 10-16in and shrink to fit.
FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 11.0, 10.0, 9.0, 8.5


def epoch_of_day(animal: str, day: int) -> str | None:
    """Epoch for a post-stroke DAY NUMBER, the key `grant_figures`' collectors use.

    Pre-stroke days are negative and return ``'pre'``. Delegates the post-stroke boundaries to
    `epochs.EPOCH_SPEC` so there is one definition; a day between the acute range and the first
    subacute day returns None rather than being rounded to a neighbour.
    """
    if day < 0:
        return "pre"
    spec = epochs.EPOCH_SPEC.get(animal)
    if spec is None:
        return None
    lo, hi = spec["acute"]
    if lo <= day <= hi:
        return "acute"
    if day >= spec["subacute_from"]:
        return "subacute"
    return None


def group_days_by_epoch(per_animal: dict) -> dict[str, list[tuple[str, int]]]:
    """``{epoch: [(animal, day), ...]}`` from a collector's ``{animal: (pre, {day: rec})}``.

    Days that belong to no epoch are DROPPED and reported by `epoch_coverage`, never folded into a
    neighbouring panel.
    """
    out: dict[str, list[tuple[str, int]]] = {e: [] for e in PANELS if e != "pre"}
    for an, (_pre, by_day) in per_animal.items():
        for day in sorted(by_day):
            e = epoch_of_day(an, int(day))
            if e in out:
                out[e].append((an, int(day)))
    return out


def epoch_coverage(per_animal: dict) -> dict:
    """Per-epoch session counts per animal, plus anything unassigned.

    EVERY POOLED PANEL MUST BE ABLE TO STATE THIS. The epochs are not balanced -- PS95 contributes
    ONE acute session and PS92 TWO subacute ones -- so a panel that does not say so implies four
    equal contributors when one animal can dominate.
    """
    grouped = group_days_by_epoch(per_animal)
    cov = {e: {an: sum(1 for a, _ in v if a == an) for an in sorted(per_animal)}
           for e, v in grouped.items()}
    unassigned = [(an, int(d)) for an, (_p, bd) in per_animal.items() for d in bd
                  if epoch_of_day(an, int(d)) is None]
    return {"per_epoch": cov, "n": {e: len(v) for e, v in grouped.items()},
            "unassigned": sorted(unassigned)}


def pool_records(per_animal: dict, epoch: str) -> tuple | None:
    """Concatenate every ``(y_true, y_pred, blocks)`` record in one epoch, across animals.

    Legitimate because each record's predictions come from that animal's OWN frozen pre-stroke
    decoder scored on its own trials, so pooling counts positions, not models -- the same reason
    `_class_confusions` stores raw counts. Blocks are made unique per (animal, day) so a downstream
    block bootstrap cannot treat two animals' block 3 as one cluster.
    """
    if epoch == "pre":
        recs = [rec for _an, (pre, _bd) in per_animal.items() if (rec := pre) is not None]
        offs = list(range(len(recs)))
    else:
        recs, offs, k = [], [], 0
        for an, day in group_days_by_epoch(per_animal).get(epoch, []):
            r = per_animal[an][1].get(day)
            if r is not None:
                recs.append(r)
                offs.append(k)
            k += 1
    if not recs:
        return None
    y = np.concatenate([r[0] for r in recs])
    p = np.concatenate([r[1] for r in recs])
    b = np.concatenate([np.asarray(r[2]) + 10_000 * (i + 1) for i, r in zip(offs, recs)])
    return y, p, b


def behaviour_by_epoch(position_metrics, sessions_of, *, positions):
    """Per-position hit rate per epoch, POOLED ACROSS ANIMALS AND WEIGHTED BY SESSION.

    Priya, 2026-08-28: "weighted mean (ie just use each session's value)" -- so hits and trials are
    summed over every (animal, session) in the epoch and the rate is their ratio, exactly as
    `grant_figures.fig_behaviour` already computes its own baseline point. A session with more trials
    therefore counts for more, and an animal contributing six acute sessions counts for more than one
    contributing a single session. That is the requested weighting and it must be stated on the
    figure, because the alternative (mean of per-animal means) differs substantially in the acute
    panel where PS95 has one session and PS94 has six.

    ``position_metrics(animal, mmdd)`` and ``sessions_of(animal)`` are injected rather than imported
    so this is testable without the behaviour tree, and so the caller decides which store to read.
    Returns ``{epoch: {position: (rate, lo, hi, n_trials, n_sessions)}}``.
    """
    from wfield_local.grant_figures import _wilson

    acc = {e: {p: [0, 0, 0] for p in positions} for e in PANELS}   # hits, n, sessions
    for an in config.animals():
        for mmdd, day in sessions_of(an):
            e = epoch_of_day(an, int(day))
            if e not in acc:
                continue
            met = position_metrics(an, mmdd) or {}
            for pos in positions:
                m = met.get(pos)
                if not m or m[3] < 5:
                    continue
                acc[e][pos][0] += round(m[0] * m[3])
                acc[e][pos][1] += m[3]
                acc[e][pos][2] += 1
    out = {}
    for e, per_pos in acc.items():
        out[e] = {}
        for pos, (h, n, ns) in per_pos.items():
            if not n:
                continue
            r, lo, hi = _wilson(h, n)
            out[e][pos] = (r, lo, hi, n, ns)
    return out


def counts_by_epoch(per_animal: dict) -> dict:
    """``{epoch: confusion counts}``, pooled across animals. Raw counts, so a panel is a SUM."""
    from wfield_local.grant_figures import _counts
    return {e: _counts(pool_records(per_animal, e)) for e in PANELS}
