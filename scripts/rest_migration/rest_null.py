"""Does each animal change the same PART of cortex? The cross-position null, on the REST baseline.

THE QUESTION. Every significance test in this arm asks whether a POOLED effect differs from zero
given the spread across four animals. This asks something different and harder: do the four animals
change the same place? It uses no significance machinery -- it is a correlation between animals --
so it is an independent line of evidence rather than a restatement.

THE STATISTIC. For each (reference, position, epoch): take each animal's OWN `epoch - pre` map,
then the MEAN PAIRWISE PEARSON r between animals, over the 129,770 px of `stat_mask`. Pearson r is
invariant to a positive scale factor per animal, so the within-animal RMS normalisation cannot move
it -- it measures SHAPE agreement, not size.

WHY IT NEEDS A NULL, and this is the part that matters. **The PRE-STROKE maps already correlate
between animals at r ~= 0.88.** Four mice share gross cortical anatomy, one Allen warp, one imaging
geometry and one set of optical artefacts. A large positive r is this statistic's RESTING STATE, not
a finding. Reading a ranking off the raw numbers -- which is what I did the first time -- mistakes
the shared anatomy for a shared result.

THE NULL. Keep every animal's real delta map, but let each animal independently draw one of the six
POSITIONS at that epoch. Anatomy, warp, rim, glue and every other between-animal commonality survive
the shuffle untouched; the ONLY thing destroyed is "the animals changed in the same
POSITION-SPECIFIC way". So:

    observed >> null   the agreement is about THIS position's change     <- a result
    observed ~= null   the agreement is position-INDEPENDENT shared offset -- the animals do move
                       together, but they would move together at any position, so it says nothing
                       about this one

That second case is not hypothetical: it is exactly what the retired reward-anchored baseline
produced at chronic (observed +0.494 against a null of +0.497), and it is why that column was
declared uninterpretable.

WHAT IT CANNOT TELL US. It is blind to effect SIZE -- four animals can agree precisely on a tiny
change. It says nothing about whether a change is non-zero; that is the bootstrap's job. And with
2,000 draws over 72 cells the p values are a screen, not a verdict: Bonferroni would demand
p < 0.0007.

RUN AS:  python -m scripts.rest_migration.rest_null
"""
from __future__ import annotations

import csv
import itertools
import pathlib
import sys
import time

import numpy as np

ALIGN, VARIANT = "cue", "working"
EPOCHS = ("acute", "subacute", "chronic")
N_NULL = 2000
SEED = 0


def _per_animal(store, ref, q, epoch):
    d = {an: [m[ref] for m in ((by.get(epoch) or {}).get(q) or {}).values() if ref in m]
         for an, by in store.items()}
    return {a: np.mean(v, axis=0) for a, v in d.items() if v}


def _z(v):
    """Mean-removed and unit-norm, so a dot product IS Pearson r."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return None
    v = np.where(ok, v, 0.0) - v[ok].mean()
    n = float(np.linalg.norm(v))
    return None if n == 0 else v / n


def main() -> int:
    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.paths import PathResolver

    t0 = time.time()
    mask = bm.stat_mask()
    rng = np.random.default_rng(SEED)
    print(f"[null] stat mask {int(mask.sum())} px; loading maps ({ALIGN}/{VARIANT})", flush=True)
    store, _rel, _ntr = prm.maps_by_epoch(ALIGN, VARIANT)
    print(f"[null]   .. {len(store)} animals, {time.time() - t0:.0f}s", flush=True)

    rows = []
    for ref in prm.REFERENCES:
        pre = {q: _per_animal(store, ref, q, "pre") for q in CONF_LABELS}
        for q in CONF_LABELS:
            zz = {a: _z(m[mask]) for a, m in pre[q].items()}
            zz = {a: v for a, v in zz.items() if v is not None}
            ans = sorted(zz)
            rs = [float(zz[a] @ zz[b]) for a, b in itertools.combinations(ans, 2)]
            if rs:
                rows.append(dict(reference=ref, position=q, contrast="pre (baseline)",
                                 n_animals=len(ans), mean_r=round(float(np.mean(rs)), 4),
                                 null_mean=None, null_p=None))
        for e in EPOCHS:
            # ONE Gram matrix per epoch: 4 animals x 6 positions z-scored once, so every null draw
            # is a table lookup rather than a 130k-element correlation.
            vecs, index = [], {}
            for q in CONF_LABELS:
                post = _per_animal(store, ref, q, e)
                for a in sorted(set(pre[q]) & set(post)):
                    z = _z((post[a] - pre[q][a])[mask])
                    if z is not None:
                        index[(a, q)] = len(vecs)
                        vecs.append(z)
            if not vecs:
                continue
            R = np.asarray(vecs) @ np.asarray(vecs).T
            animals = sorted({a for a, _q in index})
            for q in CONF_LABELS:
                ans = [a for a in animals if (a, q) in index]
                if len(ans) < 2:
                    continue
                pairs = list(itertools.combinations(range(len(ans)), 2))
                obs = float(np.mean([R[index[(ans[i], q)], index[(ans[j], q)]] for i, j in pairs]))
                pool = {a: [p for p in CONF_LABELS if (a, p) in index] for a in ans}
                nulls = []
                for _ in range(N_NULL):
                    pick = [index[(a, pool[a][rng.integers(len(pool[a]))])] for a in ans]
                    nulls.append(float(np.mean([R[pick[i], pick[j]] for i, j in pairs])))
                rows.append(dict(reference=ref, position=q, contrast=f"{e} - pre",
                                 n_animals=len(ans), mean_r=round(obs, 4),
                                 null_mean=round(float(np.mean(nulls)), 4),
                                 null_p=round(float(np.mean([x >= obs for x in nulls])), 4)))
        print(f"  .. {ref} done ({time.time() - t0:.0f}s)", flush=True)

    out = pathlib.Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
    from wfield_local import figure_layout as fl
    p = fl.sidecar_for(out, "epoch_15x_between_animal_null_RESTref", ".csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[null] wrote {p}", flush=True)

    for e in EPOCHS:
        print(f"\n== {e} - pre ==  observed r (null mean, p)")
        print(f"{'position':<15}" + "".join(f"{r:>28}" for r in prm.REFERENCES))
        for q in CONF_LABELS:
            cells = []
            for r in prm.REFERENCES:
                m = [x for x in rows if x["reference"] == r and x["position"] == q
                     and x["contrast"] == f"{e} - pre"]
                cells.append(f"{m[0]['mean_r']:>+10.3f} ({m[0]['null_mean']:+.3f}, p={m[0]['null_p']:.3f})"
                             if m else f"{'--':>28}")
            print(f"{q:<15}" + "".join(cells))
    print(f"\n[null] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
