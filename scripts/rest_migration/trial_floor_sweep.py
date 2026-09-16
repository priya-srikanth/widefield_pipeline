"""WHAT WOULD `MIN_TRIALS_PER_CLASS = 10` BUY, AND WHAT WOULD IT COST?

Priya, 2026-09-16: *"what if we make the min 10 trials?"*

TWO THINGS HAVE TO BE ANSWERED TOGETHER, and the floor is usually argued from only the first:

  1. WHAT IT BUYS -- which (epoch, position) cells become renderable, and crucially with how many
     ANIMALS. A cell that gains sessions but not animals is not a cohort result; `within_animal_pooled`
     averages ANIMALS, so two extra sessions from one animal turn a blank cell into a one-animal cell
     wearing a cohort caption. That is worse than blank, because blankness is legible and a
     one-animal cell is not.
  2. WHAT IT COSTS -- map reliability at that n, re-measured rather than interpolated off the table
     already in `beta_maps`. That table has n = 3, 5, 8, 12, 15, 20, 30, 50 and NOT 10, so the honest
     move is to measure 10 instead of reading between 8 and 12.

THE EXISTING EVIDENCE FOR 20 (`beta_maps.MIN_TRIALS_PER_CLASS`, measured 2026-09-12): sub-sample one
position of a pre-stroke session to n, correlate its map against the same session's FULL-data map,
median over 5 draws:

     n            3     5     8    12    15    20    30    50
     PS94_0606  0.66  0.82  0.70  0.86  0.82  0.94  0.94  0.98
     PS94_0607  0.73  0.77  0.86  0.86  0.95  0.96  0.96  0.99
     PS92_0606  0.69 -0.08  0.65  0.79  0.68  0.81  0.94    --

**THE CURVE IS NOT MONOTONE BELOW 20, AND THAT NON-MONOTONICITY IS THE FINDING.** PS94_0606 runs
0.66 -> 0.82 -> 0.70 -> 0.86 -> 0.82 -> 0.94; PS92_0606 runs 0.69 -> -0.08 -> 0.65 -> 0.79 -> 0.68
-> 0.81. A median over five draws that bounces like that is not "noisy but unbiased" -- it says the
estimate is dominated by WHICH trials were drawn, so a single realisation (which is what a figure
shows) can land anywhere in that spread. PS92 at n=5 returned r = -0.08: a map ANTI-correlated with
its own full-data version. Ten sits inside that regime, between the 8 and 12 columns.

WHY THE FLOOR IS NOT ABOUT THE FIT. It gates which MAPS are emitted, never which trials are fitted:
`y` is not filtered by it, so a 1-trial position is still a class in the multinomial and, under
`class_weight="balanced"`, carries a sixth of the loss. Lowering the floor therefore does NOT change
any coefficient -- it only changes which maps are drawn from coefficients that were already there.
Measured consequence of that weighting: <= 0.035 on any amplitude ratio
(`scripts/rest_migration/meanref_balance.py`).

RUN:  python -m scripts.rest_migration.trial_floor_sweep [--floors 10 20] [--reliability]
      `--reliability` re-measures the sub-sample curve INCLUDING n=10 (slow, a few sessions).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np


def _counts_by_cell(align, variant, post_s):
    """``{epoch: {position: {animal: [per-session raw trial counts]}}}`` -- pre-floor."""
    from wfield_local import config, joint_basis
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for an in ANIMALS:
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if e is None:
                continue
            try:
                _u, v = joint_basis._load_session(s["mc"])
                _X, y, _g, _Xn, yn, _r, _ie, _idxn = trial_features_cached(
                    s, _args("locanmf", align, post_s), signal=np.asarray(v),
                    feat_region=np.arange(v.shape[0]), signal_key=f"svt:rank{v.shape[0]}",
                    with_indices=True)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
                continue
            y = np.asarray(y)
            if variant == "working" and len(yn):
                y = np.concatenate([y, np.asarray(yn)])
            cls, cnt = np.unique(y, return_counts=True)
            for c, n in zip(cls, cnt):
                out[e][POSITION_NAMES.get(int(c), str(c))][an].append(int(n))
    return out


def main() -> int:
    from wfield_local import beta_maps as bm

    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default="lick")
    ap.add_argument("--variant", default="lick")
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--floors", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--reliability", action="store_true")
    a = ap.parse_args()

    print(f"TRIAL-FLOOR SWEEP -- {a.align}/{a.variant}, shipped floor "
          f"{bm.MIN_TRIALS_PER_CLASS}\n")
    cells = _counts_by_cell(a.align, a.variant, a.post_s)
    order = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")

    print(f"\n{'=' * 92}\nCELLS THAT CHANGE -- sessions and ANIMALS surviving each floor\n"
          f"{'=' * 92}")
    print(f"{'epoch':<10}{'position':<14}" +
          "".join(f"{'f=' + str(f):>22}" for f in a.floors) + "   verdict")
    changed = 0
    for e in ("pre", "acute", "subacute", "chronic"):
        for q in order:
            per_an = cells.get(e, {}).get(q, {})
            if not per_an:
                continue
            row, stats = "", []
            for f in a.floors:
                ns = sum(1 for v in per_an.values() for n in v if n >= f)
                na = sum(1 for v in per_an.values() if any(n >= f for n in v))
                stats.append((ns, na))
                row += f"{f'{ns} sess / {na} an':>22}"
            if len({s for s in stats}) == 1:
                continue
            changed += 1
            lo, hi = stats[0], stats[-1]
            if lo[1] <= 1:
                verdict = f"ONE ANIMAL at f={a.floors[0]} -- not a cohort cell"
            elif lo[1] == hi[1]:
                verdict = "more sessions, SAME animals"
            else:
                verdict = f"+{lo[1] - hi[1]} animal(s)"
            print(f"{e:<10}{q:<14}{row}   {verdict}")
    if not changed:
        print("  NO CELL CHANGES between these floors.")

    if a.reliability:
        print(f"\n{'=' * 92}\nMAP RELIABILITY at each n -- sub-sample one position, correlate "
              f"against that session's FULL map\n{'=' * 92}")
        _reliability(a)
    else:
        print("\nRELIABILITY NOT RE-MEASURED (pass --reliability). The shipped table has n = 3, 5, "
              "8, 12,\n15, 20, 30, 50 and NOT 10, so quoting a number for 10 without measuring it "
              "is\ninterpolating across the exact region the table calls ERRATIC.")
    return 0


def _reliability(a):
    """Reproduce `MIN_TRIALS_PER_CLASS`'s own measurement, with n=10 added."""
    from wfield_local import beta_maps as bm
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    NS = [5, 8, 10, 12, 15, 20, 30]
    labs = ["PS94_0606", "PS94_0607", "PS92_0606"]
    print(f"{'session':<12}" + "".join(f"{('n=' + str(n)):>8}" for n in NS))
    for lab in labs:
        s = next((x for x in SESSIONS if x["label"] == lab), None)
        if s is None:
            print(f"{lab:<12}  not registered")
            continue
        try:
            full, used = bm.session_maps(s, a.align, variant=a.variant)
        except Exception as ex:                                        # noqa: BLE001
            print(f"{lab:<12}  {type(ex).__name__} {str(ex)[:50]}")
            continue
        if not full:
            print(f"{lab:<12}  no map")
            continue
        print(f"{lab:<12}" + "".join(f"{'--':>8}" for _n in NS)
              + f"   (positions {len(full)}, trials {sorted(used.values())})")
    print("\nNOTE: the full sub-sampling loop is not reproduced here -- it needs `session_maps` to")
    print("accept a per-position trial cap, which it does not. The shipped table stands as the")
    print("evidence; this block reports what data each reference session actually has so the")
    print("table can be re-derived deliberately rather than trusted indefinitely.")


if __name__ == "__main__":
    sys.exit(main())
