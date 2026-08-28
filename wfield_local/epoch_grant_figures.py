"""Render the pooled cross-animal EPOCH figures.

Priya, 2026-08-28: pooled versions of the grant figures with three panels -- pre-stroke, acute and
subacute -- instead of a linear time axis, sized to be read at a quarter page or smaller. And:
*"I do NOT want this to have to re-create the wheel."*

SO THIS MODULE COMPUTES NOTHING. It calls `grant_figures`' existing collectors, groups their
per-day records by `epochs.epoch_of`, and hands the groups to `epoch_figures`' renderers. Every
population here is the SAME OBJECT the per-animal figures are drawn from, which is the only way two
figures captioned with the same trials can be relied on to contain them -- `tests/
test_epoch_figures.py` forbids this module from fitting a model or pooling sessions itself.

WHAT POOLING MEANS DIFFERS BY MEASURE, and the difference is not cosmetic:

  * `_collect_5c` returns TRIAL-LEVEL records `(y_true, y_pred, blocks)` scored by each animal's own
    frozen pre-stroke decoder. Pooling those is a CONCATENATION and the confusion matrix a SUM,
    because raw counts add. Every trial counts once, so a session with more trials counts for more.
  * `_matrices_*` return matrices that have already been reduced. Pooling those is a MEAN OVER
    SESSIONS -- which is exactly the weighting Priya asked for (*"weighted mean (ie just use each
    session's value)"*), and it is why the session dots matter: the acute panel is six PS94
    sessions against one PS95 session.

Run: ``python -m wfield_local.epoch_grant_figures [--only 1b 5c ...] [--output DIR]``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config, epoch_figures as ef
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

#: (display name, alignment, trial class). The lick window admits only lick trials -- a trial with
#: no detected lick has no lick to align to -- which is `grant_figures._variants`' rule, restated
#: here as data because these figures name the class in their titles.
ARMS = (("ENL", "precue", "working", "ENL (pre-cue), lick + miss-while-working"),
        ("cue", "cue", "working", "post-cue, lick + miss-while-working"),
        ("lick", "lick", "lick", "post-lick, lick trials only"))

CHANCE = 1.0 / 6.0


def _short_labels():
    from wfield_local.grant_figures import CONF_LABELS, _short
    return _short(CONF_LABELS)


def _counts_line(per_epoch):
    """``pre 92:11 93:11 ... | acute 92:5 ... 95:1 | subacute ...`` for a figure's subtitle.

    THE PER-EPOCH BREAKDOWN, not a per-animal total. A total is what hides the imbalance: PS95
    contributes 19 sessions overall and exactly ONE of them is acute, which is the fact a reader
    of the acute panel needs and the only place it can be stated on a bar figure is here.
    """
    parts = []
    for e in ef.PANELS:
        per = per_epoch.get(e) or {}
        if not any(per.values()):
            continue
        inner = " ".join(f"{a[-2:]}:{c}" for a, c in sorted(per.items()) if c)
        parts.append(f"{e} n={sum(per.values())} ({inner})")
    return "   |   ".join(parts)


# --------------------------------------------------------------------------------- behaviour

def fig_behaviour(out_dir):
    """1b pooled: hit rate per spout position, per epoch, weighted by session."""
    from wfield_local.grant_figures import _position_metrics, _sessions, CONF_LABELS, _short

    by = ef.behaviour_by_epoch(_position_metrics, lambda an: _sessions(an), positions=CONF_LABELS)
    if not any(by.values()):
        return None
    short = dict(zip(CONF_LABELS, _short(CONF_LABELS)))
    values = {e: {short[p]: v for p, v in per.items()} for e, per in by.items() if per}

    # ONE DOT PER SESSION, from the same store the bars are summed from -- not a second pass over
    # the behaviour tree, which could disagree with the bars it sits on.
    points = {e: {short[p]: [] for p in CONF_LABELS} for e in values}
    counts = {}
    for an in config.animals():
        for mmdd, day in _sessions(an):
            e = ef.epoch_of_day(an, int(day))
            if e not in points:
                continue
            met = _position_metrics(an, mmdd) or {}
            seen = False
            for p in CONF_LABELS:
                m = met.get(p)
                if m and m[3] >= 5:
                    points[e][short[p]].append((an, float(m[0])))
                    seen = True
            if seen:
                counts.setdefault(e, {})
                counts[e][an] = counts[e].get(an, 0) + 1
    return ef.bar_row(values, out_dir, name="epoch_1b_behaviour_by_position",
                      title="Behaviour: hit rate by spout position, pooled across animals "
                            "(one dot per session)",
                      subtitle=_counts_line(counts),
                      ylabel="hit rate", positions=_short(CONF_LABELS), points=points,
                      # HEADROOM ABOVE 1.0 so a session at ceiling is a visible dot rather than a
                      # smear on the spine -- and the close positions sit at ceiling throughout.
                      ylim=(0.0, 1.06))


# ------------------------------------------------------------- decoding, from `_collect_5c`

def _accuracy_of(record, position=None):
    """Accuracy of one record, overall or at one true position. `None` when the class is absent."""
    y, p = np.asarray(record[0]), np.asarray(record[1])
    if position is not None:
        m = (y == position)
        if m.sum() < 5:
            return None
        y, p = y[m], p[m]
    return float(np.mean(y == p)) if len(y) else None


def _per_position_accuracy(per_animal, out_dir, disp, align, variant, wname):
    """Per-position accuracy of the frozen pre-stroke decoder, by epoch."""
    from wfield_local.grant_figures import CONF_LABELS, _short

    short = dict(zip(CONF_LABELS, _short(CONF_LABELS)))
    values, points = {}, {}
    for e in ef.PANELS:
        rec = ef.pool_records(per_animal, e)
        if rec is None:
            continue
        values[e] = {}
        points[e] = {}
        for q in CONF_LABELS:
            a = _accuracy_of(rec, q)
            if a is not None:
                values[e][short[q]] = a
            # one dot per session, from the SAME records the pooled bar sums
            pts = ef.per_session_values(per_animal, e, lambda _an, _d, r, q=q: _accuracy_of(r, q))
            points[e][short[q]] = [(an, v) for an, v in pts if v is not None]
    if not values:
        return None
    cov = ef.epoch_coverage(per_animal)
    per_epoch = dict(cov["per_epoch"])
    per_epoch["pre"] = {an: (1 if per_animal.get(an, (None, {}))[0] is not None else 0)
                        for an in per_animal}
    return ef.bar_row(
        values, out_dir, name=f"epoch_acc_by_position_{align}_{variant}",
        title=f"Per-position decoding accuracy, {wname} -- pooled across animals",
        subtitle=_counts_line(per_epoch),
        ylabel="accuracy", positions=_short(CONF_LABELS), points=points,
        chance=CHANCE, ylim=(0.0, 1.06))


def _confusion_rows(per_animal, out_dir, disp, align, variant, wname):
    """5c/5d pooled: the frozen pre-stroke decoder's confusion per epoch, and the delta."""
    counts = ef.counts_by_epoch(per_animal)
    counts = {e: M for e, M in counts.items() if M is not None and np.asarray(M).sum()}
    if not counts:
        return []
    cov = ef.epoch_coverage(per_animal)["per_epoch"]
    made = []
    p = ef.confusion_row(
        counts, out_dir, name=f"epoch_5c_frozen_confusion_{align}_{variant}",
        title=f"Frozen PRE-stroke decoder, pooled across animals -- {wname}",
        coverage=cov, delta=True, chance=CHANCE, labels=_short_labels())
    if p:
        made.append(p)
    return made


# ------------------------------------------------------------------------------ the driver

def _epoch_arm(align, variant):
    """`_collect_5c`'s per-animal records, with anything outside an epoch reported not dropped."""
    from wfield_local.grant_figures import _collect_5c

    per_animal, _days = _collect_5c(align, variant)
    return per_animal


def main(argv=None) -> int:
    from wfield_local.console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--only", nargs="+", default=None,
                    choices=("1b", "acc", "5c"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1b", "acc", "5c"))

    if "1b" in want:
        try:
            print(f"  wrote {fig_behaviour(out)}", flush=True)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 1b: {type(ex).__name__} {str(ex)[:160]}", flush=True)

    for disp, align, variant, wname in ARMS:
        if not (want & {"acc", "5c"}):
            break
        try:
            per_animal = _epoch_arm(align, variant)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! collect {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                  flush=True)
            continue
        cov = ef.epoch_coverage(per_animal)
        # UNASSIGNED DAYS ARE REPORTED, never folded into a neighbouring panel. A day between the
        # acute range and the first subacute day belongs to neither, and silently rounding it to
        # one would move a boundary Priya set.
        if cov["unassigned"]:
            print(f"  .. {align}/{variant} days in no epoch: {cov['unassigned']}", flush=True)
        print(f"  .. {align}/{variant} sessions per epoch: {cov['n']} {cov['per_epoch']}",
              flush=True)
        if "acc" in want:
            try:
                print(f"  wrote {_per_position_accuracy(per_animal, out, disp, align, variant, wname)}",
                      flush=True)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! acc {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "5c" in want:
            try:
                for p in _confusion_rows(per_animal, out, disp, align, variant, wname):
                    print(f"  wrote {p}", flush=True)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 5c {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
