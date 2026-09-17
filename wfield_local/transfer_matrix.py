"""Cross-epoch decoder TRANSFER, for any window: rest, ENL, post-cue or post-lick.

THE QUESTION. A frozen pre-stroke decoder that fails post-stroke is consistent with several
geometries, and the usual `pre -> *` row cannot tell them apart. Filling the whole matrix can:

    train on epoch A  ->  score on epoch B, for every ordered pair

    ROTATION     A and B separate the positions along DIFFERENT directions. Both off-diagonal
                 directions fail, symmetrically.
    ADDITION     B kept A's directions and added new ones. Then a B-trained model reads A at
                 nearly A's own ceiling, while A's model misses what B added -- ASYMMETRIC.
    DEGRADATION  B carries little position at all. Then B's own DIAGONAL is near chance, and no
                 transfer into it can be interpreted.

THE UNIT IS ONE TRIAL (task arms) OR ONE REST PERIOD (rest arm), featurised in that animal's JOINT
LocaNMF basis. The joint basis is not a nicety: per-session SVD components are DIFFERENT CORTICAL
PATCHES on different days, so a model frozen in one session's basis and applied to another is not a
degraded decoder but a meaningless one, reading component 7 as though it were the same tissue.

THREE THINGS MAKE A CELL MEAN ANYTHING, and all three are enforced here rather than left to callers:

  * TRAINING SIZE IS MATCHED IN EVERY CELL. The epochs hold very different session counts (pre
    10-17, chronic 6, PS94 chronic 0), so an unmatched matrix measures how much training data each
    epoch happened to have. Every cell draws the same number of whole BLOCKS. This is the `5rm`
    lesson: unmatched, the pre-stroke gap carries a training-size handicap with no lesion in it.
  * THE DIAGONAL HOLDS OUT ITS OWN TEST SESSION, so it is a real ceiling and not a memorised one.
  * THE NULL IS A BLOCK PERMUTATION, not the analytic 1/n. Positions run in blocks, so two units
    from one block share whatever slow drift the session has; shuffling labels ACROSS blocks keeps
    that structure inside the null.

**THE DIRECTION CONTRAST IS TAKEN ON RAW ABOVE-CHANCE, NEVER ON THE RETAINED RATIOS.** `retained`
below divides by the TEST epoch's own ceiling, which is a good description of one cell and a trap
across two: the two directions have different denominators, so

    retained(B->A) - retained(A->B) = r * (1/ceil_A - 1/ceil_B)      for symmetric raw r

is nonzero whenever the ceilings differ, with no asymmetry in the data at all. Measured on the rest
arm 2026-09-17: raw transfer symmetric in 3/3 animals, yet the normalised asymmetry read +0.404 /
+0.441 / -0.001 and the closed form above predicts +0.432 / +0.331 / +0.053 from the ceilings alone.
The first version of that analysis printed "consistent with ADDITION" and was reporting its own
denominators. See DECISIONS.md.

EXTRACTED FROM THE REST ARM so the task arms cannot drift from it. Three of the defects the
2026-09-16 audit found spread by writing scripts from siblings as templates, carrying the prose and
dropping the guards; there is one implementation here and the callers differ only in how they build
``{session_label: (epoch, X, y, blocks)}``.
"""
from __future__ import annotations

import hashlib

import numpy as np

EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")
#: Printed header. A constant because a backslash inside an f-string EXPRESSION is a Python 3.12
#: feature and this box runs 3.10, where it is a SyntaxError rather than a style complaint.
TRAIN_TEST = "train \\ test"
#: An asymmetry counts as material only at >= this fraction of that animal's own typical transfer.
#: RELATIVE rather than absolute because the raw above-chance transfers are small (0.05-0.11 on the
#: rest arm) and differ between animals, so one absolute cut-off would be lenient for a strong
#: animal and unreachable for a weak one. Stated here, before any run, so a verdict cannot be
#: fitted to the numbers it turned out to produce.
MATERIAL_FRACTION = 0.25
#: A ceiling at or below this above chance has no usable denominator; `retained` refuses instead of
#: manufacturing a ratio of 5 or 50 out of a near-zero divisor.
MIN_CEILING = 0.02


def retained(acc, null, ceiling_acc, ceiling_null):
    """``(acc - null) / (ceiling - ceiling_null)``, or None when the ceiling is not above chance.

    DESCRIPTIVE ONLY. Correct for "how much of B's decodable position does A's model capture";
    never valid as a contrast between two directions -- see the module docstring.
    """
    den = ceiling_acc - ceiling_null
    if den <= MIN_CEILING:
        return None
    return (acc - null) / den


def balanced_accuracy(y_true, y_pred):
    """``(mean per-class recall over the classes PRESENT, n_classes)``.

    Not sklearn's: that averages over classes the test set does not contain, which silently moves
    the chance level of the number being reported. Averaging over present classes keeps chance at
    1/n_classes and the count is returned so the caller can say which denominator it used.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cls = np.unique(y_true)
    if not len(cls):
        return float("nan"), 0
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in cls])), len(cls)


def block_permute(y, g, rng):
    """Labels permuted BETWEEN blocks, block structure intact.

    Each block keeps its own size and its own internal label; what is destroyed is which POSITION a
    block carried. A trial-wise shuffle would also destroy the within-block correlation, making the
    null easier than the data and every p value optimistic.
    """
    y, g = np.asarray(y), np.asarray(g)
    blocks = np.unique(g)
    lab = {b: y[g == b][0] for b in blocks}
    shuffled = rng.permutation([lab[b] for b in blocks])
    out = np.empty_like(y)
    for b, v in zip(blocks, shuffled):
        out[g == b] = v
    return out


def _matched(pipe_fn, X, y, g, n_target, rng):
    """Fit on a size-matched random subset of whole BLOCKS. ``(fitted, n_used)`` or None.

    WHOLE BLOCKS, not loose units: blocks are the unit every other resampling here uses, and
    sampling loose rows would hand the matched model a training set with LESS within-block
    correlation than its comparators.
    """
    g = np.asarray(g)
    y = np.asarray(y)
    keep = np.zeros(len(y), bool)
    for b in rng.permutation(np.unique(g)):
        keep |= (g == b)
        if keep.sum() >= n_target:
            break
    if keep.sum() < 2 or len(np.unique(y[keep])) < 2:
        return None
    return pipe_fn().fit(X[keep], y[keep]), int(keep.sum())


def build(data, pipe_fn, *, n_perm=200, seed_ns="", min_sessions=2, rng=None, log=print):
    """One arm's transfer matrix.

    ``data`` maps ``session_label -> (epoch, X, y, blocks)``. Returns a list of per-cell rows.
    """
    rng = rng or np.random.default_rng(0)
    by_ep = {e: [k for k, v in data.items() if v[0] == e] for e in EPOCH_ORDER}
    trainable = [e for e in EPOCH_ORDER if len(by_ep[e]) >= min_sessions]
    log("   usable: " + ", ".join(f"{e}={len(by_ep[e])}" for e in EPOCH_ORDER))
    if "pre" not in trainable:
        log("   !! pre is not a usable train source -- arm skipped")
        return []

    # ONE n_target FOR THE WHOLE ANIMAL, taken as the smallest pool any cell would otherwise get
    # (including the diagonal's leave-one-out shrinkage), so every cell of its matrix is trained on
    # the same amount of data and the cells are comparable to each other.
    sizes = [sum(len(data[k][2]) for k in by_ep[tr] if k != te)
             for tr in trainable for te in data
             if len([k for k in by_ep[tr] if k != te]) >= min_sessions]
    if not sizes:
        log("   !! no cell has a usable training pool")
        return []
    n_target = int(min(sizes))
    log(f"   training pool matched to {n_target} units in every cell")

    rows = []
    for tr_ep in trainable:
        for te_lab, (te_ep, Xte, yte, gte) in data.items():
            pool = [k for k in by_ep[tr_ep] if k != te_lab]
            if len(pool) < min_sessions:
                continue
            Xtr = np.concatenate([data[k][1] for k in pool])
            ytr = np.concatenate([data[k][2] for k in pool])
            # BLOCK IDS MADE UNIQUE ACROSS SESSIONS before matching: they restart per session, so a
            # sampler pooling two sessions would treat two different blocks as one.
            gtr = np.concatenate([np.asarray(data[k][3], np.int64) + 1_000_000 * (i + 1)
                                  for i, k in enumerate(pool)])
            # SEEDED PER CELL. A shared advancing generator would make each draw depend on how many
            # cells ran before it, so re-running on a subset of animals would silently change
            # results that are supposed to be identical.
            seed = int(hashlib.sha1(f"{seed_ns}|{tr_ep}|{te_lab}".encode()).hexdigest()[:8], 16)
            got = _matched(pipe_fn, Xtr, ytr, gtr, n_target, np.random.default_rng(seed))
            if got is None:
                continue
            fit, n_used = got
            if len(getattr(fit, "classes_", [])) < 3:
                continue
            pred = fit.predict(Xte)
            acc, ncls = balanced_accuracy(yte, pred)
            draws = [balanced_accuracy(block_permute(yte, gte, rng), pred)[0]
                     for _ in range(n_perm)]
            rows.append({"train_epoch": tr_ep, "test_epoch": te_ep, "test_session": te_lab,
                         "n_test": len(yte), "n_classes": ncls, "n_train_matched": n_used,
                         "acc": round(acc, 4), "null": round(float(np.mean(draws)), 4),
                         "p": round(float(np.mean([d >= acc for d in draws])), 4),
                         "above_chance": round(acc - float(np.mean(draws)), 4)})
    return rows


def cell(rows, tr, te, animal=None):
    """``(acc, null, n_animals, animals, n_sessions)`` -- animals weighted EQUALLY, not sessions.

    Pre holds up to 17 sessions and chronic 6, so a session-weighted mean would let the
    best-sampled animal set the cohort number.
    """
    sel = [r for r in rows if r["train_epoch"] == tr and r["test_epoch"] == te
           and (animal is None or r["animal"] == animal)]
    if not sel:
        return None
    per = {}
    for r in sel:
        per.setdefault(r["animal"], []).append(r)
    accs = [float(np.mean([r["acc"] for r in v])) for v in per.values()]
    nuls = [float(np.mean([r["null"] for r in v])) for v in per.values()]
    return float(np.mean(accs)), float(np.mean(nuls)), len(per), sorted(per), len(sel)


def summarise(rows):
    """Per-(train, test) aggregate. ``retained`` is a MEAN OF PER-ANIMAL RATIOS, not a ratio of
    pooled means: the animals' ceilings differ more than twofold, so a pooled denominator belongs
    to no animal and every other cohort statistic in this project is animal-level."""
    agg = []
    for tr in EPOCH_ORDER:
        for te in EPOCH_ORDER:
            c = cell(rows, tr, te)
            if c is None:
                continue
            ratios = []
            for an in c[3]:
                x, ceil = cell(rows, tr, te, an), cell(rows, te, te, an)
                if x and ceil:
                    v = retained(x[0], x[1], ceil[0], ceil[1])
                    if v is not None:
                        ratios.append(v)
            agg.append({"train_epoch": tr, "test_epoch": te, "acc": round(c[0], 4),
                        "null": round(c[1], 4), "above_chance": round(c[0] - c[1], 4),
                        "retained_vs_own_ceiling": (round(float(np.mean(ratios)), 4)
                                                    if ratios else None),
                        "retained_n_animals": len(ratios),
                        "n_animals": c[2], "animals": "|".join(c[3]), "n_sessions": c[4]})
    return agg


def report(rows, agg, a_ep="pre", b_ep="chronic", log=print):
    """Print both matrices and the RAW-units verdict for one ordered pair. Returns the verdict."""
    log("\nTRANSFER MATRIX -- balanced accuracy (null), rows TRAIN, cols TEST")
    log(f"{TRAIN_TEST:<12}" + "".join(f"{e:>22}" for e in EPOCH_ORDER))
    for tr in EPOCH_ORDER:
        cells = []
        for te in EPOCH_ORDER:
            m = [x for x in agg if x["train_epoch"] == tr and x["test_epoch"] == te]
            cells.append(f"{m[0]['acc']:>13.3f} ({m[0]['null']:.2f})" if m else f"{'--':>22}")
        log(f"{tr:<12}" + "".join(cells))

    log("\nRETAINED vs the TEST epoch's own ceiling -- DESCRIPTIVE ONLY, never across directions")
    log(f"{TRAIN_TEST:<12}" + "".join(f"{e:>14}" for e in EPOCH_ORDER))
    for tr in EPOCH_ORDER:
        cells = []
        for te in EPOCH_ORDER:
            m = [x for x in agg if x["train_epoch"] == tr and x["test_epoch"] == te]
            v = m[0]["retained_vs_own_ceiling"] if m else None
            cells.append(f"{v:>14.3f}" if v is not None else f"{'--':>14}")
        log(f"{tr:<12}" + "".join(cells))

    log("\n" + "=" * 78)
    per_animal = []
    for an in sorted({r["animal"] for r in rows}):
        fwd, rev = cell(rows, a_ep, b_ep, an), cell(rows, b_ep, a_ep, an)
        ca, cb = cell(rows, a_ep, a_ep, an), cell(rows, b_ep, b_ep, an)
        if not all([fwd, rev, ca, cb]):
            log(f"   {an}: no {b_ep} data -- contributes nothing to this contrast")
            continue
        raw_f, raw_r = fwd[0] - fwd[1], rev[0] - rev[1]
        rf, rr = retained(*fwd[:2], *cb[:2]), retained(*rev[:2], *ca[:2])
        if rf is None or rr is None:
            log(f"   {an}: ceiling collapsed -- refused")
            continue
        gain = (cb[0] - cb[1]) - (ca[0] - ca[1])
        per_animal.append((an, raw_f, raw_r, rf, rr, gain))
        log(f"   {an}: RAW {a_ep}->{b_ep} {raw_f:+.4f}  {b_ep}->{a_ep} {raw_r:+.4f}  "
            f"raw asym {raw_r - raw_f:+.4f}")
        log(f"          (normalised {rf:+.3f} / {rr:+.3f}, ceiling gain {gain:+.4f})")

    def material(raw_f, raw_r):
        return abs(raw_r - raw_f) >= MATERIAL_FRACTION * max(abs(raw_f), abs(raw_r), 1e-9)

    n_asym = sum(1 for _a, f, r, _rf, _rr, _g in per_animal if material(f, r))
    log("")
    if not per_animal:
        verdict = "NOT COMPUTABLE"
        log(f"=> {verdict}: no animal has both directions.")
    elif n_asym == 0:
        verdict = "SYMMETRIC"
        log(f"=> SYMMETRIC in {len(per_animal)}/{len(per_animal)} animals on RAW transfer:")
        log("   neither model preferentially reads the other's epoch, so the codes share a")
        log("   PARTIAL, direction-independent overlap -- a ROTATION rather than an ADDITION.")
        log("   ADDITION predicts the reverse direction at ~the full ceiling (retained ~ 1.0);")
        log("   observed " + " / ".join(f"{rr:.2f}" for _a, _f, _r, _rf, rr, _g in per_animal))
    elif n_asym == len(per_animal):
        verdict = "ASYMMETRIC"
        log(f"=> ASYMMETRIC on RAW transfer in {n_asym}/{len(per_animal)} animals.")
    else:
        verdict = "SPLIT"
        log(f"=> SPLIT on RAW transfer -- {n_asym} of {len(per_animal)}. NOT a result; report the")
        log("   per-animal values rather than the cohort mean.")
    log("=" * 78)
    return verdict, per_animal
