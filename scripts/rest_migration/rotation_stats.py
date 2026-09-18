"""15h post-hoc statistics: the COHORT cosine and the FAMILY-WISE threshold.

Reads `epoch_15h_rotation_draws<tag>.npz` and writes `epoch_15h_rotation_cohort<tag>.csv`.
REFUSES if the draws are absent rather than recomputing them -- the same discipline as
`rotation_maps --replot` and `reference_family_figure`, and for the same reason: silently
repaying a two-hour refit to answer a question about a statistic is how an afternoon goes.

WHAT WAS MISSING, AND WHY IT IS NOT MERELY A GAP. `epoch_15h_rotation_regions.csv` carries three
statistics per cell -- `cos_p_vs_noise`, `gain_resid_p`, and a bootstrap CI -- and all three are
PER ANIMAL. The arm's cohort summary was therefore "animals with >= 3 of 6 positions
significant", a replication count. `15k` already rejected exactly that construction for itself:
`cohort_delta` exists there because an any-animal rule gives a per-cell false-positive rate of
~1 - 0.95^4 = 18%, and because "significant in all three families" could otherwise mean three
different animals. The same argument applies here, so this module gives 15h the same two things
15k and the map families already have.

    1. A COHORT CI, never a p. With four animals an animal-level sign-flip null has 2^4 = 16
       assignments, so 0.0625 is the smallest attainable p and NO cell could reach 0.05 at any
       effect size. A CI has no such floor -- it can exclude zero.

    2. A FAMILY-WISE threshold from the PERMUTATION MAX-STATISTIC, not Bonferroni. The six
       positions within a window are not independent: `LogisticRegression` is multinomial, so
       the class coefficients are identified only up to a constant shift across classes and the
       six patterns are coupled by construction. Bonferroni assumes independence and is
       conservative here by an unknown amount. The max-statistic lets the data's own covariance
       perform the correction.

THE FAMILY IS window x contrast x position = 72. ANIMALS ARE NOT IN IT. They are replicates of
one hypothesis, not separate hypotheses; correcting across them asks "does far-contra rotate
acutely in cue" as four different questions. That is why the cohort statistic comes first -- a
72-cell family needs one value per cell, which means aggregating over animals rather than
testing each.

AND THE DRAW COUNT DOES NOT BIND THIS THE WAY IT BOUND BONFERRONI. The earlier note that ~1,440
draws were needed was about a per-cell empirical p having to beat alpha/n, where the floor is
1/(n_draws+1) = 0.00498 at 200 draws. A max-statistic needs no such thing: it needs enough draws
to estimate the 95th percentile of the max distribution, and 200 is workable (1,000 is better and
costs nothing extra to collect, since the draws are already being computed).

RUN AS:  python -m scripts.rest_migration.rotation_stats [--tag _foo] [--boot 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.rest_migration.rotation_maps import load_draws
from wfield_local.paths import PathResolver

#: Minimum draws a cell needs before it is allowed into the family. A cell estimated from three
#: draws has an sd that is mostly noise, and because the family threshold is a MAXIMUM, one such
#: cell raises the bar for every other cell in it -- the same power leak the brain mask was added
#: to stop when bulb components were setting the maximum.
MIN_DRAWS = 20

#: A null sd below this counts as ZERO. `np.std` of a constant array is ~1e-17, not 0, so a
#: `sd > 0` guard lets a degenerate cell through and it returns a z of ~1e15 -- which, because the
#: family threshold is a MAXIMUM, would set the threshold single-handedly and silence every real
#: cell. This repo has already been bitten by a zero-variance denominator once (the four-way-broken
#: cluster-permutation test). Cosines are O(1), so an absolute floor is the right form.
MIN_NULL_SD = 1e-9


def cell_z(cos_obs, cos_null):
    """``(z, mu, sd)`` -- how far the observed cosine sits BELOW its own estimation-noise null.

    SIGNED SO THAT POSITIVE MEANS MORE ROTATION. The null is the cosine between two models of the
    SAME pre-stroke code, so it is the ceiling an unrotated epoch would reach; an epoch that has
    turned scores LOWER. Writing it as `(mu - obs) / sd` keeps the tested direction positive and
    one-sided, which is what `cos_p_vs_noise` already tests per cell.
    """
    v = np.asarray(cos_null, float)
    if v.size < MIN_DRAWS:
        return None, None, None
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    if not np.isfinite(sd) or sd <= MIN_NULL_SD:
        return None, mu, sd
    return (mu - float(cos_obs)) / sd, mu, sd


def cohort_cosine(cells, n_boot=2000, seed=7, alpha=0.05):
    """``(mean, lo, hi, n_animals, n_below_ceiling)`` for one (arm, contrast, position) cell.

    THE NESTED BOOTSTRAP, with the inner level already paid for. `cells` is one entry per animal,
    each carrying `cos_boot` -- that animal's own SESSION resamples, computed during the run. A
    draw resamples ANIMALS with replacement and then takes one of that animal's session resamples,
    which is the project's standard animals -> sessions nesting; sessions sharpen each animal's
    estimate without being counted as independent animals.

    AN ANIMAL WITH NO BOOTSTRAP DRAWS CONTRIBUTES ITS POINT ESTIMATE rather than being dropped.
    Dropping it would change the COHORT COMPOSITION between cells, and a cell whose animals differ
    from its neighbour's is not comparable to it -- that is the same trap the 15k vocabulary
    intersection exists to avoid.
    """
    if not cells:
        return None
    rng = np.random.default_rng(seed)
    pools = [np.asarray(c["cos_boot"], float) if np.size(c["cos_boot"]) >= MIN_DRAWS
             else np.asarray([c["cos_obs"]], float) for c in cells]
    n_an = len(pools)
    idx = rng.integers(0, n_an, (n_boot, n_an))
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = np.mean([pools[j][rng.integers(0, pools[j].size)] for j in idx[b]])
    obs = float(np.mean([c["cos_obs"] for c in cells]))
    return (obs,
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))),
            n_an,
            int(sum(1 for c in cells if c.get("_z") is not None and c["_z"] > 0)))


def maxstat_threshold(by_cell, alpha=0.05):
    """The FWE threshold on cohort z, from the per-draw MAXIMUM across the whole family.

    Construction, and it is the one the map families already use (`significant_components`,
    and `_MEANref_`'s legend: "the maximum |z| across bins is taken per draw; the 95th percentile
    of those maxima is the threshold"):

        1. standardise every draw of every cell by THAT CELL'S own null moments, so cells with
           different noise levels are on one scale;
        2. average over animals within a draw, giving a cohort z per cell per draw;
        3. take the maximum over the family's cells, per draw;
        4. the threshold is the 95th percentile of those maxima.

    WHY IT IS CORRECT THAT THE ANIMAL DRAWS ARE INDEPENDENT ACROSS ARMS. `null_delta` seeds on
    `f"{arm}|{animal}"`, so draw d of the ENL arm shares nothing with draw d of the cue arm.
    Averaging independent nulls still gives the null OF THE AVERAGE, which is what step 2 needs;
    what independence costs is only that the cross-ARM correlation is not captured, making the
    threshold conservative across arms and exact within one. Stated rather than hidden, because
    a max-statistic's whole claim is that it respects the family's covariance.

    Returns ``(threshold, n_draws_used, n_cells)``.
    """
    usable = {k: v for k, v in by_cell.items() if v}
    if not usable:
        return None, 0, 0
    per_cell = []
    for cells in usable.values():
        cols = []
        for c in cells:
            v = np.asarray(c["cos_null"], float)
            if v.size < MIN_DRAWS:
                continue
            mu, sd = v.mean(), v.std(ddof=1)
            if not np.isfinite(sd) or sd <= MIN_NULL_SD:
                continue
            cols.append((mu - v) / sd)
        if cols:
            n = min(c.size for c in cols)
            per_cell.append(np.mean([c[:n] for c in cols], axis=0))
    if not per_cell:
        return None, 0, 0
    # TRUNCATE TO THE SHORTEST CELL so every draw index holds a value for every cell. A ragged
    # maximum would be taken over a changing number of cells and would drift downward as the
    # longer cells ran on alone.
    n = min(c.size for c in per_cell)
    stack = np.stack([c[:n] for c in per_cell])          # (cells, draws)
    maxima = stack.max(axis=0)
    return float(np.percentile(maxima, 100 * (1 - alpha))), int(n), int(stack.shape[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--boot", type=int, default=2000, help="cohort bootstrap draws")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")

    draws = load_draws(out_dir, a.tag)
    if not draws:
        print(f"!! no draws at {out_dir}/epoch_15h_rotation_draws{a.tag}.npz -- run "
              f"`rotation_maps` once with this tag first. REFUSING to recompute them here.")
        return 1
    print(f"15h COHORT STATISTICS -- {len(draws)} animal-cells from "
          f"epoch_15h_rotation_draws{a.tag}.npz")

    for d in draws:
        d["_z"], d["_mu"], d["_sd"] = cell_z(d["cos_obs"], d["cos_null"])

    by_cell = defaultdict(list)
    for d in draws:
        by_cell[(d["arm"], d["contrast"], int(d["position"]))].append(d)

    thresh, n_draws, n_cells = maxstat_threshold(by_cell, alpha=a.alpha)
    print(f"   FAMILY: {n_cells} cells (window x contrast x position), {n_draws} draws")
    print(f"   max-statistic FWE threshold at alpha={a.alpha}: z = "
          + (f"{thresh:.3f}" if thresh is not None else "NOT COMPUTABLE"))
    if n_cells and n_cells != 72:
        print(f"   NB: family is {n_cells}, not the nominal 72 -- cells are dropped where an "
              f"animal has no epoch data (PS94 has no chronic) or too few draws")

    rows, n_sig = [], 0
    for (arm, contrast, pos), cells in sorted(by_cell.items()):
        coh = cohort_cosine(cells, n_boot=a.boot, seed=7, alpha=a.alpha)
        if coh is None:
            continue
        mean, lo, hi, n_an, n_rot = coh
        zs = [c["_z"] for c in cells if c["_z"] is not None]
        zc = float(np.mean(zs)) if zs else None
        sig = bool(zc is not None and thresh is not None and zc > thresh)
        n_sig += sig
        rows.append({
            "arm": arm, "contrast": contrast, "position": pos,
            "cohort_cosine": round(mean, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_animals": n_an,
            "n_animals_rotated": n_rot,
            "cohort_z_vs_noise": (round(zc, 3) if zc is not None else None),
            "maxstat_threshold": (round(thresh, 3) if thresh is not None else None),
            "fwe_significant": sig,
            "correction": "maxstat",
            "family_n_cells": n_cells,
            "n_null_draws": n_draws,
        })

    p = out_dir / f"epoch_15h_rotation_cohort{a.tag}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"   {n_sig}/{len(rows)} cells clear the family-wise threshold")
    print(f"   {sum(r['ci_excludes_zero'] for r in rows)}/{len(rows)} cohort CIs exclude zero")
    print(f"[15h] wrote {p}")
    (out_dir / f"epoch_15h_rotation_cohort{a.tag}_meta.json").write_text(
        json.dumps({"alpha": a.alpha, "n_boot": a.boot, "family_n_cells": n_cells,
                    "n_null_draws": n_draws, "correction": "maxstat",
                    "family": "window x contrast x position; animals are replicates"}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
