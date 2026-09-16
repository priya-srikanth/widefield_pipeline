"""IS THE `MEANref` REFERENCE POSITION-WEIGHTED? No — and this measures what that costs.

Priya, 2026-09-16: *"for the MEANref maps - is the mean position-weighted?"* … *"I would think we
want it to be balanced"*.

IT IS TRIAL-WEIGHTED, IN TWO INDEPENDENT PLACES.
  1. The multinomial fit. `beta_maps.maps_by_epoch` sets `balance = (variant == "lick")`, so the
     `working` arm fits with `class_weight=None` and is pulled by base rates, while the `lick` arm
     gets `class_weight="balanced"`. **Figure 14's working and lick panels are therefore not the
     same estimator**, which is stated nowhere.
  2. The Haufe transform. `A = Cov(X) beta` centres on `X[tr].mean(0)` and takes the covariance
     over training trials as they come — the trial mean, never a per-position one.

WHY IT MATTERS HERE SPECIFICALLY. Figure 14's positions are COUPLED by construction (each class's
softmax coefficient is referenced to the others), so a position the animal stops attempting does
not merely lose its own map — it changes the reference for the other five, handing them an unearned
increase. Post-stroke, far-contralateral is exactly the position whose trial count collapses. That
is the same composition bias that motivated `restw` for the rest baseline, one level down, and it
is an INDEPENDENT reason to distrust `MEANref` on top of the two already recorded in CLAUDE.md
(the coupling itself, and the 0.47 -> 2.08 amplitude inversion once the engaged-only bug was fixed).

WHAT THIS SCRIPT DOES. Rebuilds the epoch beta-map store BOTH ways — shipped default vs
`class_weight="balanced"` forced on — and reports, per position and epoch, the within-animal pooled
amplitude ratio `|post| / |pre|` that figure 14 prints in its panel titles, plus the trial counts
that drive the imbalance. Nothing is overwritten: this is a measurement, not a re-render.

READ IT LIKE THIS. If balancing barely moves the ratios, the trial-weighting is real but inert and
`MEANref` keeps whatever standing it had. If far-contralateral moves a lot, the reference is doing
the work the figure attributes to the lesion, and `MEANref` should not be quoted per-position at
all — `QUIETref` (positions independent, no position information in the subtrahend) is already the
one to lead with.

RUN:  python -m scripts.rest_migration.meanref_balance [--align cue] [--variant working]
"""
from __future__ import annotations

import argparse
import sys


def _ratios(store, ntr, labels):
    """``{position: {epoch: (ratio, n_animals, n_trials)}}`` — figure 14's own amplitude number."""
    from wfield_local import beta_maps as bm

    out = {}
    for q in labels:
        pre = {an: list(((by.get("pre") or {}).get(q) or {}).values()) for an, by in store.items()}
        pre = {a: v for a, v in pre.items() if v}
        if not pre:
            continue
        out[q] = {}
        for e in ("pre", "acute", "subacute", "chronic"):
            arm = {an: list(((by.get(e) or {}).get(q) or {}).values())
                   for an, by in store.items()}
            arm = {a: v for a, v in arm.items() if v}
            if not arm:
                continue
            n_tr = sum(sum((((ntr.get(an) or {}).get(e) or {}).get(q) or {}).values())
                       for an in arm)
            if e == "pre":
                out[q][e] = (1.0, len(arm), n_tr)
                continue
            _p, post_p, _d, ratio, ans = bm.within_animal_pooled(pre, arm)
            if post_p is None:
                continue
            out[q][e] = (float(ratio), len(ans), n_tr)
    return out


def main() -> int:
    from wfield_local import beta_maps as bm
    from wfield_local.grant_figures import CONF_LABELS

    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default="cue")
    ap.add_argument("--variant", default="working")
    a = ap.parse_args()

    shipped = (a.variant == "lick")
    print(f"MEANref BALANCE TEST -- {a.align}/{a.variant}\n"
          f"shipped default for this arm: class_weight="
          f"{'balanced' if shipped else 'None'}\n")

    # EXPLICIT False vs True, NEVER `None` vs True.
    #
    # The first version of this script compared `balance=None` against `balance=True`. For the
    # `working` arm that is a real contrast, because None resolves to False there. For the `lick`
    # arm None resolves to TRUE, so it compared balanced against balanced and printed a column of
    # +0.000 deltas -- a perfectly clean "balancing changes nothing" that was an artefact of asking
    # the question twice in the same words. Same failure class as the freshness audit that returned
    # "0 stale" from the wrong index (STATUS_2026-09-16 pitfall 4): a comparison whose two halves
    # are identical answers instantly and convincingly. Naming the two arms explicitly makes the
    # contrast impossible to collapse.
    res = {}
    for tag, bal in (("unbalanced", False), ("balanced", True)):
        print(f"building store: class_weight={'balanced' if bal else 'None'} ...", flush=True)
        store, _rel, ntr = bm.maps_by_epoch(a.align, a.variant, balance=bal)
        if not store:
            print("!! empty store -- nothing to compare")
            return 1
        res[tag] = _ratios(store, ntr, CONF_LABELS)

    print(f"\n{'=' * 84}\nAMPLITUDE RATIO |post|/|pre|, within-animal pooled "
          f"(figure 14's own number). SHIPPED = "
          f"{'balanced' if shipped else 'unbalanced'}\n{'=' * 84}")
    print(f"{'position':<14}{'epoch':<10}{'unbal':>9}{'balanced':>10}{'delta':>9}"
          f"{'n_animals':>11}{'n_trials':>10}")
    moved = []
    for q in CONF_LABELS:
        for e in ("acute", "subacute", "chronic"):
            d = res["unbalanced"].get(q, {}).get(e)
            b = res["balanced"].get(q, {}).get(e)
            if d is None or b is None:
                continue
            print(f"{q:<14}{e:<10}{d[0]:>9.3f}{b[0]:>10.3f}{b[0] - d[0]:>+9.3f}"
                  f"{d[1]:>11}{d[2]:>10}")
            moved.append((abs(b[0] - d[0]), q, e, d[0], b[0]))
    if not moved:
        print("\nNOTHING COMPARABLE -- both stores refused every cell.")
        return 1

    moved.sort(reverse=True)
    print("\nLARGEST MOVES: " + "; ".join(f"{q} {e} {d0:.3f}->{b0:.3f}"
                                          for _m, q, e, d0, b0 in moved[:4]))
    big = [m for m in moved if m[0] >= 0.10]
    print(f"{len(big)} of {len(moved)} cells move by >= 0.10.")
    print("\nTRIAL-COUNT SKEW is what drives it -- per epoch, trials per position:")
    for e in ("pre", "acute", "subacute", "chronic"):
        row = {q: res["unbalanced"].get(q, {}).get(e, (0, 0, 0))[2] for q in CONF_LABELS}
        tot = sum(row.values())
        if not tot:
            continue
        print(f"  {e:<10}" + "  ".join(f"{q}={n}" for q, n in row.items())
              + f"   (min/max = {min(row.values()) / max(1, max(row.values())):.2f})")
    print("\nA ratio that moves under balancing was partly the REFERENCE moving, not the code. "
          "\nThe positions are COUPLED in this figure, so a position losing trials changes the "
          "\nother five as well -- read `QUIETref` (independent positions) for per-position claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
