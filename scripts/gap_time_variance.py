"""How much of the frozen-to-refit gap is a function of time since lesion?

WHY THIS EXISTS. The K08 ranking model (Aim 1) relates per-session behaviour to the
frozen-to-refit gap at each node and asks which node tracks recovery most closely. Both the
gap and behaviour change over the weeks after stroke, so a raw gap-to-behaviour correlation
is confounded by their shared time course. The standard fix is to enter days since lesion,
or each animal's own trajectory, as a covariate. That fix has its own failure mode: if the
gap is very nearly a deterministic function of time, detrending removes almost all of its
variance and the gap coefficient is then estimated from noise -- OVER-ADJUSTMENT, which
looks like a null result and is not one.

This script measures which regime we are in, per animal, by regressing the per-session gap
on that animal's own days since lesion. High R2 means detrending is expensive and the
analysis should lean on between-animal differences in recovery RATE instead. Low R2 means
detrending is nearly free, and it also means the shared-time-trend confound was small to
begin with, so neither failure mode bites.

Run it per alignment and variant; `cue`/`working` and `lick`/`lick` are the two that matter
for the grant. Per-node is the number the ranking model actually needs -- see NEXT STEP.

    python scripts/gap_time_variance.py --align cue --variant working
    python scripts/gap_time_variance.py --align lick --variant lick --out gap_time_lick.csv

WHAT THE GAP IS. `grant_figures._collect_5c(align, variant, "paired_matched")` returns, per
animal, (pre records, {day: record}) where each record is (y_true, y_pred, blocks) and
y_pred has column 0 = frozen (pre-stroke model applied unchanged) and column 1 = refit
(within-session block-CV). Gap = refit accuracy - frozen accuracy, paired at the trial
level, via `epoch_grant_figures._overall_of(record, "gap")` so this agrees exactly with the
5rmo panel rather than recomputing accuracy a second way. "paired_matched" gives the frozen
arm the same training-set size as the refit arm, so the gap is not inflated by the
ten-sessions-against-one handicap; pass --unmatched to see the raw version too.

READ adj R2, NOT R2. With ~17 post-stroke sessions and one predictor, a predictor made of
pure noise earns R2 ~ 1/(n-1) ~ 0.06 on average. Adjusted R2 at or below zero means days
since lesion explains no more than a random regressor would.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fits(day, y):
    """(R2, adjR2, slope, resid_sd) for linear, log and quadratic fits of y on day."""
    day = np.asarray(day, float)
    y = np.asarray(y, float)
    n = len(y)
    out = {}
    for name, cols in (("linear", [day]),
                       ("log", [np.log(np.maximum(day, 0.5))]),
                       ("quadratic", [day, day ** 2])):
        k = len(cols)
        if n <= k + 1:
            out[name] = (float("nan"),) * 4
            continue
        X = np.column_stack([np.ones(n)] + cols)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        r2 = 1.0 - res.var() / y.var() if y.var() > 0 else float("nan")
        adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
        out[name] = (r2, adj, float(b[1]), float(res.std(ddof=k + 1)))
    return out


def collect(align, variant, mode):
    """[(animal, day, gap)] for every POST-stroke session, using the deck's own gap."""
    from wfield_local import grant_figures as G
    from wfield_local import epoch_grant_figures as EG

    per_animal, _days = G._collect_5c(align, variant, mode)
    rows = []
    for an, (_pre, by_day) in per_animal.items():
        for day, rec in sorted(by_day.items()):
            if rec is None or int(day) <= 0:      # day <= 0 is pre-stroke
                continue
            gap = EG._overall_of(rec, "gap")
            if gap is None or not np.isfinite(gap):
                continue
            rows.append((an, int(day), float(gap)))
    return rows


def report(rows, label):
    from collections import defaultdict
    by = defaultdict(list)
    for an, day, gap in rows:
        by[an].append((day, gap))
    print(f"\n=== {label} ===")
    print(f"{'animal':8s}{'n':>4}{'R2':>8}{'adjR2':>8}{'slope/day':>12}"
          f"{'gap SD':>9}{'resid SD':>10}{'logR2':>8}{'quadR2':>8}")
    for an in sorted(by):
        d, g = zip(*sorted(by[an]))
        f = _fits(d, g)
        r2, adj, slope, rsd = f["linear"]
        print(f"{an:8s}{len(g):>4}{r2:>8.3f}{adj:>8.3f}{slope:>12.5f}"
              f"{np.std(g, ddof=1):>9.4f}{rsd:>10.4f}{f['log'][0]:>8.3f}{f['quadratic'][0]:>8.3f}")
    if by:
        n = int(np.mean([len(v) for v in by.values()]))
        print(f"\nnull expectation for R2, 1 predictor, n~{n}: {1 / max(n - 1, 2):.3f}")
        print("adj R2 <= 0 means days since lesion explains no more than a random regressor.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", default="cue", choices=("precue", "cue", "lick"))
    ap.add_argument("--variant", default="working", choices=("working", "lick"))
    ap.add_argument("--unmatched", action="store_true",
                    help="also report the raw gap (frozen trained on all of pre)")
    ap.add_argument("--out", default=None, help="write the per-session rows to this CSV")
    a = ap.parse_args(argv)

    modes = [("paired_matched", "training-set matched gap")]
    if a.unmatched:
        modes.append(("paired", "raw gap (refit minus full-pre frozen)"))

    all_rows = []
    for mode, name in modes:
        rows = collect(a.align, a.variant, mode)
        if not rows:
            print(f"no post-stroke sessions returned for {a.align}/{a.variant}/{mode}")
            continue
        report(rows, f"{name} -- {a.align}-aligned, {a.variant} trials")
        all_rows += [(mode, *r) for r in rows]

    if a.out and all_rows:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["mode", "animal", "day_since_lesion", "gap_refit_minus_frozen"])
            w.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} rows to {a.out}")
    return 0


# NEXT STEP, and the reason this file is not the final answer. `_collect_5c` pools over the
# whole cortical field, so this measures the gap of ONE readout. The ranking model in Aim 1
# operates on individual LocaNMF components, and a single component can trend with time far
# more strongly than the pooled signal does. To get the number the grant actually needs,
# rebuild the paired records per component (restrict the design matrix to one component's
# columns before `_pipe().fit`) and run `_fits` on each. Until that is done, treat the
# figures here as an order-of-magnitude check, not as the per-node result.
if __name__ == "__main__":
    raise SystemExit(main())
