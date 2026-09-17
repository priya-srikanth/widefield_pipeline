"""Is the chronic rest code a REPLACEMENT for the pre-stroke one, or an ADDITION to it?

THE CLAIM THIS TESTS. `15f` reports that chronically the within-session refit reaches 1.181 of its
own pre-stroke level while the frozen pre-stroke model reaches 0.611, and the arm concludes
"position is in chronic rest, read by something other than the pre-stroke code -- REPLACEMENT".
That conclusion is an INFERENCE FROM A DIVERGENCE, not an identification: it says the pre-stroke
readout stopped working, and says nothing about what the new one is. `docs/STATUS_2026-09-16.md`
lists "identify the substitute code" as the central open question for exactly this reason.

WHY ONE MORE CELL SETTLES THE FRAMING. `15f` fills only the `pre -> *` ROW of what is really a
TRANSFER MATRIX. The missing cell that matters is `chronic -> pre`, and the two hypotheses make
opposite predictions for it:

    REPLACEMENT / ROTATION   the chronic code occupies different directions. A chronic-trained
                             model does not read pre-stroke rest either. BOTH directions fail --
                             SYMMETRIC.

    ADDITION                 the chronic code CONTAINS the pre-stroke directions and adds new ones.
                             A chronic-trained model still reads pre-stroke rest; the pre-stroke
                             model misses only what was added. ASYMMETRIC: chronic -> pre is high,
                             pre -> chronic is not.

**The asymmetry is the diagnostic**, and it is not reachable from any number `15f` currently
produces. If the answer is ADDITION then "replacement" is the wrong word in every document that
uses it, and the substitute is a set of ADDED dimensions rather than a moved code -- which is a
different search.

NORMALISATION -- AND THE TRAP IN IT, WHICH THIS SCRIPT WALKED INTO ON ITS FIRST RUN. Each cell is
also reported as a fraction of the TEST epoch's OWN within-epoch ceiling:

    retained(A -> B) = (acc(A -> B) - null) / (acc(B -> B) - null)

That is a fine DESCRIPTIVE quantity -- "how much of B's decodable position does model A capture" --
and the diagonal is 1.0 by construction. **It must NOT be used for the direction contrast.**
`retained(pre->chronic)` and `retained(chronic->pre)` are divided by DIFFERENT denominators, so
their difference is dominated by how the two ceilings differ rather than by the codes:

    retained(chr->pre) - retained(pre->chr) = r * (1/ceil_pre - 1/ceil_chr)   for symmetric raw r

Measured 2026-09-17, and it is not a small correction. The RAW above-chance transfer is very nearly
symmetric in every animal (pre->chr 0.081 / 0.088 / 0.062 against chr->pre 0.077 / 0.108 / 0.055 for
PS92 / PS93 / PS95), yet the NORMALISED asymmetry reads +0.404 / +0.441 / -0.001 -- and the formula
above predicts +0.432 / +0.331 / +0.053 from the denominators alone. The apparent "addition"
signature was the ceiling difference: chronic rest is MORE decodable than pre-stroke rest in PS92
and PS93 (ceiling gain +0.128 and +0.132 above chance) and barely more in PS95 (+0.018), which is
exactly the ordering of the fake asymmetry.

**THE VERDICT IS THEREFORE TAKEN ON RAW ABOVE-CHANCE TRANSFER**, where the two directions share
units and no denominator enters. The retained matrix is still printed, because "the pre-stroke model
captures a third of chronic's position information" is worth knowing -- it just cannot be compared
across directions.

ADDITION ALSO FAILS ON ITS OWN TERMS, independently of any asymmetry: if chronic contained the
pre-stroke directions and added to them, a chronic-trained model would read pre-stroke rest at
NEARLY ITS FULL CEILING, `retained(chronic->pre) ~ 1.0`. It is 0.76 / 0.76 / 0.42.

TRAINING-SET SIZE IS MATCHED IN EVERY CELL, and skipping this would repeat a known error. The task
arm learned it at `5rm`: unmatched, a model trained on ten pre-stroke sessions is compared against
one trained on four fifths of one, and the resulting gap carries a training-size handicap with no
lesion in it -- which flipped the sign of the pre-stroke gap when it was finally matched. Here the
epochs hold very different session counts (pre ~10-17, chronic 6, PS94 chronic 0), so every cell
draws the SAME number of whole training BLOCKS via `matched_frozen`.

WHAT THIS CANNOT DO. It cannot say WHERE the substitute lives -- that needs the Haufe-transformed
decoder PATTERN projected through the joint basis, which is the natural follow-on and is
deliberately not attempted here. Identify whether a substitute exists before characterising one.

IT IMPORTS FROM `rest_frozen_decoder` RATHER THAN COPYING IT. Three of the defects the 2026-09-16
audit found spread by writing scripts from siblings as templates, carrying prose and structure
while dropping guards. Every shared step here -- collection, the engagement gate, the repaired
classifier, block-matched training, balanced accuracy, the block-permutation null -- is the SAME
function object `15f` calls, so the two arms cannot drift apart.

RUN AS:  python -m scripts.rest_migration.substitute_code [--perm 200] [--animals PS92 PS93]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np

EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")
#: Header for the printed matrices, held in a constant rather than written inline: a backslash
#: inside an f-string EXPRESSION is a Python 3.12 feature, and this box runs 3.10 where it is a
#: SyntaxError rather than a style complaint.
TRAIN_TEST = "train \\ test"


def retained(acc, null, ceiling_acc, ceiling_null):
    """``(acc - null) / (ceiling - ceiling_null)``, or None when the ceiling is not above chance.

    RETURNS None RATHER THAN A LARGE NUMBER when the denominator collapses. A test epoch whose own
    within-epoch decoder barely beats its null has no meaningful ceiling, and dividing by it
    manufactures ratios of 5 or 50 that read as spectacular transfer. `15f` hit exactly this: its
    acute cell moves by up to 0.21 per animal because acute is 16 sessions with PS95 contributing
    one, and a small denominator is most of why.
    """
    den = ceiling_acc - ceiling_null
    if den <= 0.02:
        return None
    return (acc - null) / den


def _score(fit, X, y, g, rng, n_perm, bal, nullf):
    """``(balanced accuracy, null mean, p, n_classes)`` for one fitted model on one session.

    `_balanced_accuracy` RETURNS A PAIR, `(acc, n_classes)`, and the second element is not
    decoration: it averages over the classes PRESENT in `y_true`, so chance is 1/n_classes and a
    session missing a position has a DIFFERENT chance level from one that is not. It is carried
    into the row rather than dropped, because a cell scored over five positions is not comparable
    to one scored over six and nothing else in the output would reveal that.
    """
    pred = fit.predict(X)
    acc, ncls = bal(y, pred)
    draws = [bal(nullf(y, g, rng, "blockperm"), pred)[0] for _ in range(n_perm)]
    return acc, float(np.mean(draws)), float(np.mean([d >= acc for d in draws])), int(ncls)


def main() -> int:
    # THE SAME FUNCTION OBJECTS `15f` USES -- see the module docstring on why these are imported
    # and not copied.
    from scripts.rest_migration.rest_frozen_decoder import (
        _balanced_accuracy,
        _collect,
        _epoch_dir,
        _usable,
        matched_frozen,
        null_labels,
    )
    from wfield_local import config, epochs, joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--min-periods", type=int, default=40)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--min-sessions-per-epoch", type=int, default=2,
                    help="an epoch needs this many usable sessions to be a TRAIN source, so a "
                         "pooled model is never one session wearing an epoch's name")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    a.out = a.out or _epoch_dir()

    from wfield_local.quiet_periods import quiet_variant
    variant = quiet_variant() or "retired"
    stem = f"epoch_15g_transfer_{variant}{a.tag}"
    rng = np.random.default_rng(0)
    t0 = time.time()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    animals = a.animals or sorted({s["label"].split("_")[0] for s in SESSIONS if s["label"] in want})
    print(f"SUBSTITUTE CODE -- transfer matrix, variant {variant}, {len(animals)} animals, "
          f"{a.perm} permutations\n")

    rows, skipped = [], []
    for an in animals:
        todo = [s for s in SESSIONS
                if s["label"] in want and s["label"].startswith(an) and s.get("h5")]
        try:
            basis = joint_locanmf.load(an, sessions=SESSIONS)
        except Exception as ex:                                          # noqa: BLE001
            print(f"!! {an}: no joint basis ({type(ex).__name__}) -- SKIPPED\n")
            skipped.append(f"{an}: no joint basis")
            continue
        print(f"{an}: basis {basis.basis_id}, {basis.ncomp} components, {len(todo)} sessions")

        data = {}
        for s in todo:
            ep = epochs.epoch_of(s["label"])
            if ep is None:
                continue
            got, why = _collect(s, basis, a.bins, verbose=False, gate=True)
            if got is None:
                skipped.append(why)
                continue
            X, y, g, _dur, _gap, _hz = got
            ok, bad = _usable(y, a.min_periods, a.min_per_class)
            if not ok:
                skipped.append(f"{s['label']}: {bad}")
                continue
            data[s["label"]] = (ep, X, y, g)
        by_ep = {e: [k for k, v in data.items() if v[0] == e] for e in EPOCH_ORDER}
        print("   usable: " + ", ".join(f"{e}={len(by_ep[e])}" for e in EPOCH_ORDER), flush=True)

        # A TRAIN SOURCE NEEDS >= min_sessions_per_epoch, and on the DIAGONAL it needs one more,
        # because the test session is held out of its own training pool. Stated per epoch rather
        # than assumed: PS94 has NO chronic session at all (it has not plateaued -- see
        # DECISIONS.md), so its chronic row and column are absent by construction, not dropped.
        trainable = [e for e in EPOCH_ORDER if len(by_ep[e]) >= a.min_sessions_per_epoch]
        if "pre" not in trainable:
            print(f"!! {an}: pre is not a usable train source -- SKIPPED\n")
            skipped.append(f"{an}: pre not trainable")
            continue

        # ONE n_target FOR THE WHOLE ANIMAL so every cell of its matrix is trained on the same
        # amount of data. Taken as the smallest pool any cell would otherwise get, including the
        # diagonal's leave-one-out shrinkage.
        sizes = []
        for tr_ep in trainable:
            for te_lab in data:
                pool = [k for k in by_ep[tr_ep] if k != te_lab]
                if len(pool) >= a.min_sessions_per_epoch:
                    sizes.append(sum(len(data[k][2]) for k in pool))
        if not sizes:
            skipped.append(f"{an}: no cell has a usable training pool")
            continue
        n_target = int(min(sizes))
        print(f"   training pool matched to {n_target} periods in every cell", flush=True)

        for tr_ep in trainable:
            for te_lab, (te_ep, Xte, yte, gte) in data.items():
                pool = [k for k in by_ep[tr_ep] if k != te_lab]
                if len(pool) < a.min_sessions_per_epoch:
                    continue
                Xtr = np.concatenate([data[k][1] for k in pool])
                ytr = np.concatenate([data[k][2] for k in pool])
                # BLOCK IDS MADE UNIQUE ACROSS SESSIONS before matching, exactly as `15f` does:
                # they restart at 0 in every session, so a sampler pooling two sessions would treat
                # two different blocks as one and draw a unit that does not exist.
                gtr = np.concatenate([np.asarray(data[k][3], np.int64) + 1_000_000 * (i + 1)
                                      for i, k in enumerate(pool)])
                # `matched_frozen` RETURNS A FITTED MODEL, not features -- `(fitted, n_used)`.
                # Unpacking it as `(X, y)` type-checks and runs, and then `np.unique` of an int
                # is length 1, so every cell would silently fail its class check and the matrix
                # would come out EMPTY. Caught before the first run by reading the callee rather
                # than inferring it from the name.
                #
                # SEEDED PER CELL, mirroring `15f`: a shared advancing generator would make each
                # draw depend on how many cells ran before it, so re-running with `--animals` on a
                # subset would silently change results that are supposed to be identical.
                seed = int(hashlib.sha1(f"{an}|{tr_ep}|{te_lab}".encode()).hexdigest()[:8], 16)
                sub = matched_frozen(Xtr, ytr, gtr, n_target, np.random.default_rng(seed))
                if sub is None:
                    continue
                fit, n_used = sub
                if len(getattr(fit, "classes_", [])) < 3:
                    continue
                acc, nul, p, ncls = _score(fit, Xte, yte, gte, rng, a.perm,
                                           _balanced_accuracy, null_labels)
                rows.append({"animal": an, "train_epoch": tr_ep, "test_epoch": te_ep,
                             "test_session": te_lab, "n_test": len(yte), "n_classes": ncls,
                             "n_train_matched": n_used, "acc": round(acc, 4),
                             "null": round(nul, 4), "p": round(p, 4),
                             "acc_minus_null": round(acc - nul, 4)})
        print(f"   .. {len([r for r in rows if r['animal'] == an])} cells "
              f"({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        print("no cells computed")
        return 1

    out = Path(a.out)
    p = out / f"{stem}_sessions.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[15g] wrote {p}")

    # ---- aggregate: mean over test sessions, then over animals ------------------------------
    def cell(tr, te, animal=None):
        sel = [r for r in rows if r["train_epoch"] == tr and r["test_epoch"] == te
               and (animal is None or r["animal"] == animal)]
        if not sel:
            return None
        per_an = {}
        for r in sel:
            per_an.setdefault(r["animal"], []).append(r)
        # ANIMALS ARE WEIGHTED EQUALLY, not sessions: pre holds up to 17 sessions and chronic 6,
        # so a session-weighted mean would let the best-sampled animal set the cohort number.
        accs = [np.mean([r["acc"] for r in v]) for v in per_an.values()]
        nuls = [np.mean([r["null"] for r in v]) for v in per_an.values()]
        return (float(np.mean(accs)), float(np.mean(nuls)), len(per_an),
                sorted(per_an), len(sel))

    def per_animal_cell(tr, te, an):
        """``(acc, null)`` for ONE animal's cell, or None. The unit the cohort mean averages."""
        c = cell(tr, te, animal=an)
        return None if c is None else (c[0], c[1])

    agg = []
    for tr in EPOCH_ORDER:
        for te in EPOCH_ORDER:
            c = cell(tr, te)
            if c is None:
                continue
            # MEAN OF RATIOS, NOT RATIO OF MEANS, and the two genuinely disagree here. Pooling
            # accuracies across animals first and dividing by a pooled ceiling gave +0.173 for the
            # headline asymmetry where the per-animal mean is +0.282 -- because the animals' own
            # ceilings differ by more than twofold (PS92's pre ceiling is 0.102 above chance,
            # PS93's chronic 0.275), so a pooled denominator belongs to no animal. Every other
            # cohort statistic in this project is animal-level for the same reason.
            ratios = [retained(*per_animal_cell(tr, te, an), *per_animal_cell(te, te, an))
                   for an in c[3] if per_animal_cell(tr, te, an) and per_animal_cell(te, te, an)]
            ratios = [x for x in ratios if x is not None]
            r = float(np.mean(ratios)) if ratios else None
            agg.append({"train_epoch": tr, "test_epoch": te, "acc": round(c[0], 4),
                        "null": round(c[1], 4), "acc_minus_null": round(c[0] - c[1], 4),
                        "retained_vs_own_ceiling": (None if r is None else round(r, 4)),
                        "retained_n_animals": len(ratios),
                        "retained_per_animal": "|".join(f"{x:+.3f}" for x in ratios),
                        "n_animals": c[2], "animals": "|".join(c[3]), "n_sessions": c[4]})
    p2 = out / f"{stem}_matrix.csv"
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    print(f"[15g] wrote {p2}\n")

    print("TRANSFER MATRIX -- balanced accuracy (null), rows TRAIN, cols TEST")
    print(f"{TRAIN_TEST:<12}" + "".join(f"{e:>22}" for e in EPOCH_ORDER))
    for tr in EPOCH_ORDER:
        cells = []
        for te in EPOCH_ORDER:
            m = [x for x in agg if x["train_epoch"] == tr and x["test_epoch"] == te]
            cells.append(f"{m[0]['acc']:>13.3f} ({m[0]['null']:.2f})" if m else f"{'--':>22}")
        print(f"{tr:<12}" + "".join(cells))

    print("\nRETAINED vs the TEST epoch's OWN ceiling (diagonal is 1.0 by construction)")
    print(f"{TRAIN_TEST:<12}" + "".join(f"{e:>14}" for e in EPOCH_ORDER))
    for tr in EPOCH_ORDER:
        cells = []
        for te in EPOCH_ORDER:
            m = [x for x in agg if x["train_epoch"] == tr and x["test_epoch"] == te]
            v = m[0]["retained_vs_own_ceiling"] if m else None
            cells.append(f"{v:>14.3f}" if v is not None else f"{'--':>14}")
        print(f"{tr:<12}" + "".join(cells))

    # ---- THE VERDICT -------------------------------------------------------------------------
    def rv(tr, te):
        m = [x for x in agg if x["train_epoch"] == tr and x["test_epoch"] == te]
        return m[0]["retained_vs_own_ceiling"] if m and m[0]["retained_vs_own_ceiling"] else None

    print("\n" + "=" * 78)
    fwd, rev = rv("pre", "chronic"), rv("chronic", "pre")
    if fwd is None or rev is None:
        print("VERDICT: not computable -- one direction is missing "
              f"(pre->chronic {fwd}, chronic->pre {rev})")
    else:
        print(f"pre -> chronic  retained {fwd:+.3f}     chronic -> pre  retained {rev:+.3f}")
        print(f"ASYMMETRY (chronic->pre minus pre->chronic): {rev - fwd:+.3f}")

        # PER-ANIMAL FIRST, because a cohort mean cannot tell a consistent effect from one animal
        # carrying two. Measured 2026-09-17: the cohort asymmetry is +0.282 and it is +0.404 /
        # +0.441 / -0.001 across PS92 / PS93 / PS95 -- PS95 shows EXACTLY NONE. An earlier version
        # of this block printed "ADDITION" off the cohort number alone and would have published a
        # 2-of-3 split as a finding.
        per_animal_rows = []
        for an in sorted({r["animal"] for r in rows}):
            f_c, r_c = per_animal_cell("pre", "chronic", an), per_animal_cell("chronic", "pre", an)
            cp, cc = per_animal_cell("pre", "pre", an), per_animal_cell("chronic", "chronic", an)
            if not all([f_c, r_c, cp, cc]):
                print(f"   {an}: no chronic data -- contributes nothing to this contrast")
                continue
            rf, rr = retained(*f_c, *cc), retained(*r_c, *cp)
            if rf is None or rr is None:
                print(f"   {an}: ceiling collapsed -- refused")
                continue
            # THE CONTRAST IS TAKEN ON RAW ABOVE-CHANCE, not on the two retained values. They have
            # different denominators, so their difference mostly measures the ceilings -- see the
            # module docstring for the measurement that established this.
            raw_f, raw_r = f_c[0] - f_c[1], r_c[0] - r_c[1]
            gain = (cc[0] - cc[1]) - (cp[0] - cp[1])
            per_animal_rows.append((an, raw_f, raw_r, rf, rr, gain))
            print(f"   {an}: RAW pre->chr {raw_f:+.4f}  chr->pre {raw_r:+.4f}  "
                  f"raw asym {raw_r - raw_f:+.4f}")
            print(f"          (normalised {rf:+.3f} / {rr:+.3f}, asym {rr - rf:+.3f} -- "
                  f"ceiling gain {gain:+.4f})")
        print()
        # THE BAR IS ON RAW UNITS and is stated rather than tuned: an asymmetry counts as material
        # only if it is at least a quarter of the typical transfer in that animal, which is the
        # scale at which it could mean anything. `n_asym` counts animals clearing it in EITHER
        # direction, because a consistent reverse asymmetry would also be a finding.
        def _material(raw_f, raw_r):
            """An asymmetry counts only if it is >= 25% of that animal's typical transfer."""
            typical = max(abs(raw_f), abs(raw_r), 1e-9)
            return abs(raw_r - raw_f) >= 0.25 * typical

        n_asym = sum(1 for _a, raw_f, raw_r, _rf, _rr, _g in per_animal_rows if _material(raw_f, raw_r))
        if not per_animal_rows:
            print("=> NOT COMPUTABLE: no animal has both directions.")
        elif n_asym == 0:
            print(f"=> SYMMETRIC in {len(per_animal_rows)}/{len(per_animal_rows)} animals on RAW transfer. Neither model")
            print("   preferentially reads the other's epoch, so the codes share a PARTIAL,")
            print("   direction-independent overlap -- consistent with a ROTATION of the code")
            print("   rather than an ADDITION to it.")
            print("   ADDITION is separately rejected: it predicts chronic->pre at nearly the full")
            print("   pre-stroke ceiling (retained ~ 1.0); observed "
                  + " / ".join(f"{rr:.2f}" for _a, _f, _r, _rf, rr, _g in per_animal_rows) + ".")
        elif n_asym == len(per_animal_rows):
            print(f"=> ASYMMETRIC on RAW transfer in {n_asym}/{len(per_animal_rows)} animals. Report the")
            print("   direction and check the training pools before interpreting.")
        else:
            print(f"=> SPLIT on RAW transfer -- {n_asym} of {len(per_animal_rows)}. NOT a result; report")
            print("   the per-animal values above rather than the cohort mean.")
        print()
        print("   NOTE the normalised asymmetry above is LARGER and points at 'addition'. It is an")
        print("   artefact of dividing the two directions by different ceilings; the raw numbers")
        print("   are the ones that carry the geometry. See the module docstring.")
    print("=" * 78)
    if skipped:
        print(f"\n{len(skipped)} session(s)/animal(s) skipped; first few:")
        for s in skipped[:6]:
            print("   ", s)
    print(f"\n[15g] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
