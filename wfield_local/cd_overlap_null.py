"""The TRIAL-MATCHED pre-to-pre null for the condition-independent mode: overlap AND amplitude.

Priya, 2026-09-24: *"so we want to compare post-stroke vs pre-stroke cosine to pre-stroke vs
pre-stroke LOSO cosine?"* -- yes, and then *"trial-matched?"* -- yes, and that is the whole design.

WHY THE NUMBERS ARE UNREADABLE WITHOUT THIS. `cd_trajectories` reports, per epoch, the subspace
overlap of that epoch's condition-independent mode with the pre-stroke one, and the ratio of their
magnitudes. Neither has a scale of its own:

  * an overlap of 0.61 is not "39% rotated away" -- two subspaces estimated from DIFFERENT SESSIONS
    do not fully overlap even when nothing changed, because each is estimated with noise;
  * a magnitude ratio above 1.0 is not necessarily a larger response -- `dev_norm` is a NORM, so
    estimation noise inflates it, and a post-stroke epoch has fewer trials per session. **The bias
    points the same way as the finding**, which is the one configuration that cannot be waved away.

MEASURED and unanimous, so the question is quantitative rather than directional: the magnitude ratio
exceeded 1.0 in 12 of 12 cells (4 animals x 3 epochs, pre-cue), mean 1.41.

WHY TRIALS AND NOT SESSIONS. `scripts/rest_migration/rest_baseline_epoch_drift.py` is the
construction this copies and it matches on SESSION COUNT, which is right for its question -- it
calibrates BETWEEN-session drift. The bias here is WITHIN-session estimation noise and that scales
with trials: a post-stroke session with 80 success trials and a pre-stroke one with 600 are one
session each while their grand means differ in noise by sqrt(7.5). Matching sessions alone would
leave the entire confound in place and return a reassuringly high null. So the match is on TOTAL
TRIALS, with session count matched too where a subset can do both, and the residual mismatch in each
is reported rather than assumed small.

THE CONSTRUCTION, and the point is that observed and null share a reference:

    B   a subset of PRE-STROKE sessions whose summed trial count matches epoch E's
    A   the remaining pre-stroke sessions
    null       overlap(CIM(A), CIM(B))        ||dev(B)|| / ||dev(A)||
    observed   overlap(CIM(A), CIM(E))        ||dev(E)|| / ||dev(A)||

Both sides are computed against the SAME A, so the only thing differing between them is whether the
second group is post-stroke or pre-stroke. Reading the observation against a null built on the FULL
pre set instead would hand the null a quieter reference than the observation gets -- conservative for
overlap, ANTI-conservative for amplitude, i.e. flattering the exact number in question.

K IS FIXED ACROSS THE COMPARISON, from the full pre-stroke set. `subspace_overlap` normalises by the
first basis's dimension and chance is K/n, so letting each group choose its own K by variance
explained would vary the dimension AND the chance level between the observation and its own null.

WHAT IS NOT CLAIMED. The draws are a resampling of which pre-stroke sessions stand in for E, not a
permutation of an exchangeable label, so `p_not_worse` is reported as the FRACTION OF DRAWS in which
the observation was no worse than its matched null -- not as a p-value. At 11-13 pre-stroke sessions
per animal the draws are also not independent; they share sessions by construction.

    python -m wfield_local.cd_overlap_null --animal PS95 --align precue
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import cd_trajectories as cdt

#: Draws of (A, B) per epoch. The pool is 11-13 pre-stroke sessions, so the number of DISTINCT
#: subsets of a given size is in the hundreds -- more draws buy resolution on the matching, not
#: independent evidence.
N_DRAW = 200

#: Hill-climbing swaps per draw when matching trial counts. A random subset of the right SIZE is
#: usually far off in TRIALS; swapping one session at a time for the one that reduces the gap
#: converges in a handful of steps on a pool this small.
N_SWAP = 40

#: STOP CLIMBING ONCE THE MATCH IS THIS GOOD, as a fraction of the epoch's trial total.
#:
#: NOT AN OPTIMISER, AND THE DIFFERENCE IS THE WHOLE NULL. Climbing to the single best subset makes
#: every draw return the SAME split: measured, all 200 draws identical, so the null had zero width
#: and its interval was a point. The draws have to sample the space of ADEQUATELY matched splits,
#: not find the best one. Stopping at a tolerance keeps the randomness of the start.
#:
#: 0.05 IS AFFORDABLE ON REAL DATA. Each animal has 11 pre-stroke sessions of varied size, and the
#: best achievable match is 0.0-0.7% of the target in 9 of the 12 (animal, epoch) cells. PS93 is the
#: exception at 6.9% acute and 10.6% subacute -- its pre-stroke sessions are unusually uniform
#: (453-565 trials), so no subset sum lands close. Those two cells are matched as well as the data
#: allows and the achieved gap is reported per cell.
MATCH_TOL = 0.05


def trial_matched_split(counts, n_target, k_sessions=None, rng=None, n_swap=N_SWAP,
                        tol=MATCH_TOL):
    """``(idx_B, idx_A)`` -- a disjoint split of pre-stroke sessions, B matched to `n_target` trials.

    `counts` is each pre-stroke session's trial count. `k_sessions` is a PREFERENCE for |B|, taken
    when it can also match the trials; trials win, because the bias being calibrated is within-session
    estimation noise and that scales with trials, not sessions.

    GREEDY WITH A TOLERANCE, NOT EXHAUSTIVE. Two reasons it stops early rather than optimising: the
    draws must VARY (see `MATCH_TOL`), and the achieved counts are reported, so a cell the data
    cannot match is visible in the output rather than absorbed into it.

    WHOLE SESSIONS ARE THE ONLY UNIT AVAILABLE, which bounds what this can do. Only the per-session
    grand MEANS are cached, not the trials, so B's trial total moves in steps of one session. Where
    an epoch's total falls below the smallest pre-stroke session no subset can reach it at all --
    pre-stroke sessions of 500-640 trials cannot match an epoch total of 270, and the closest subset
    is 1610. That does not arise in this cohort (the smallest epoch total is 454 against pre-stroke
    sessions from 221), but it is the failure mode to check before reusing this elsewhere.
    """
    rng = np.random.default_rng() if rng is None else rng
    counts = np.asarray(counts, float)
    n = len(counts)
    if n < 2:
        return None
    if k_sessions is None:
        # smallest subset that can reach the target, so B is not forced to overshoot
        order = rng.permutation(n)
        run, k_sessions = 0.0, n - 1
        for i, j in enumerate(order, start=1):
            run += counts[j]
            if run >= n_target:
                k_sessions = i
                break
    k = int(np.clip(k_sessions, 1, n - 1))
    idx = rng.permutation(n)
    B, A = list(idx[:k]), list(idx[k:])
    good = float(tol) * float(n_target)
    for _ in range(n_swap):
        gap = abs(counts[B].sum() - n_target)
        if gap <= good:
            break                       # GOOD ENOUGH -- stop here, so draws differ from each other
        best, bgap = None, gap
        for bi, b in enumerate(B):
            for ai, a in enumerate(A):
                g = abs(counts[B].sum() - counts[b] + counts[a] - n_target)
                if g < bgap - 1e-9:
                    bgap, best = g, (bi, ai)
        if best is None:
            break
        bi, ai = best
        B[bi], A[ai] = A[ai], B[bi]
    return np.asarray(B, int), np.asarray(A, int)


def _cim_and_norm(chunks, t, stats, k):
    """``(basis, ||dev||)`` for a set of ``(G, n)`` session means, at a FIXED rank."""
    G = cdt.pooled_mean(chunks, stats)
    if G is None:
        return None, float("nan")
    return cdt.condition_independent_modes(G, t, k=k), cdt.dev_norm(G, t)


def epoch_null(pre, ep_chunks, t, stats=None, k=None, n_draw=N_DRAW, rng=None,
               match_sessions=True):
    """Observed and matched-null overlap/scale for ONE epoch, over `n_draw` trial-matched splits.

    Returns a dict of arrays over draws plus the achieved match, so the mismatch is auditable.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    counts = [int(c[1]) for c in pre]
    n_target = float(sum(int(c[1]) for c in ep_chunks))
    k_sessions = len(ep_chunks) if match_sessions else None
    if len(pre) < 2 or not ep_chunks:
        return None

    cimE, devE = _cim_and_norm(ep_chunks, t, stats, k)
    out = {"obs_cos": [], "null_cos": [], "obs_scale": [], "null_scale": [],
           "n_B": [], "n_A": [], "k_B": []}
    for _ in range(n_draw):
        split = trial_matched_split(counts, n_target, k_sessions, rng=rng)
        if split is None:
            continue
        iB, iA = split
        # A SESSION-COUNT MATCH THAT CANNOT REACH THE TRIAL TARGET IS ABANDONED FOR THIS DRAW, not
        # silently kept: an epoch with more sessions than the pre-stroke set has spare would
        # otherwise contribute draws matched on neither.
        if not len(iB) or not len(iA):
            continue
        cimB, devB = _cim_and_norm([pre[i] for i in iB], t, stats, k)
        cimA, devA = _cim_and_norm([pre[i] for i in iA], t, stats, k)
        if cimA is None or cimB is None or cimE is None or not devA:
            continue
        out["obs_cos"].append(cdt.subspace_overlap(cimA, cimE))
        out["null_cos"].append(cdt.subspace_overlap(cimA, cimB))
        out["obs_scale"].append(devE / devA)
        out["null_scale"].append(devB / devA)
        out["n_B"].append(sum(counts[i] for i in iB))
        out["n_A"].append(sum(counts[i] for i in iA))
        out["k_B"].append(len(iB))
    if not out["obs_cos"]:
        return None
    res = {k_: np.asarray(v, float) for k_, v in out.items()}
    res["n_target"] = n_target
    res["k_target"] = len(ep_chunks)
    res["k"] = int(k) if k else None
    # PAIRED WITHIN A DRAW, because both sides share that draw's A. Differencing first removes the
    # draw-to-draw variation in the reference, which is the largest term and is common to both.
    res["d_cos"] = res["obs_cos"] - res["null_cos"]
    res["d_scale"] = res["obs_scale"] - res["null_scale"]
    # NOT A p-VALUE: the fraction of draws in which the observation was no worse than its matched
    # null. The draws resample which pre-stroke sessions stand in for E; they do not permute an
    # exchangeable label, and they share sessions.
    res["p_not_worse_cos"] = float(np.mean(res["d_cos"] >= 0))
    res["p_not_bigger_scale"] = float(np.mean(res["d_scale"] <= 0))
    return res


def run_animal(animal, align="precue", gate="lick", n_draw=N_DRAW, seed=0, verbose=True):
    """Every post-stroke epoch's matched null for one animal, off the CACHED per-session means.

    NOTHING IS PROJECTED HERE when `cd_trajectories` has been run for this (animal, align, gate):
    `grand_means` reads the same `cdgm2-` cache entries, so this is arithmetic on what the render
    already wrote.
    """
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs, joint_locanmf, session_cache
    from wfield_local.locanmf_frozen_decoder import _args

    basis = joint_locanmf.load(animal)
    post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    args = _args(source="roi", align=align, post_s=post_s)
    win_n = int(round(args.post_s * args.fs))
    pre_n, post_frames = cdt.span_frames(align, args.fs)
    use = cdt.LICK_ALIGNED_CLASSES if align == "lick" else cdt.GATES[gate]
    akind = cdt.arms_cache_kind(args, align)

    book, Xf = [], []
    for s in [x for x in ak.curated_sessions() if x["label"].startswith(animal)]:
        try:
            arms = session_cache.cached(s, akind,
                                        lambda s=s: cdt.session_arms(s, args, basis, align),
                                        verbose=False)
        except Exception:
            continue
        book.append((s, epochs.epoch_of(s["label"]), arms))
    # THE SAME FROZEN Z-SCORING FRAME the observation used -- pre-stroke window means, standardised
    # once. A null computed in a different frame would not be comparable to what it calibrates.
    for s, ep, arms in book:
        if ep != "pre" or not any(arms[c]["y"] for c in use):
            continue
        fit_at = [f for c in use for f in arms[c]["fit"]]
        Xf.append(session_cache.cached(
            s, f"cdfit-{align}-{'+'.join(use)}-{basis.basis_id[:8]}-{win_n}",
            lambda s=s, arms=arms, fit_at=fit_at: cdt.window_means(
                cdt._OnceSignal(joint_locanmf.BasisSource(basis, s))(), fit_at, win_n),
            verbose=False))
    if not Xf:
        return None
    stats = cdt.component_stats(np.vstack(Xf)) if cdt.STANDARDISE else None

    gm = cdt.grand_means(book, use, basis, align, pre_n, post_frames)
    if not gm.get("pre"):
        return None
    t = np.arange(-pre_n, post_frames) / args.fs
    # K FROM THE FULL PRE-STROKE SET, then held fixed everywhere (see the module docstring).
    cim_pre = cdt.condition_independent_modes(cdt.pooled_mean(gm["pre"], stats), t)
    if cim_pre is None:
        return None
    k = int(cim_pre.shape[1])

    out = {"animal": animal, "align": align, "gate": gate, "k": k,
           "ncomp": int(basis.ncomp), "chance": cdt.subspace_chance(int(basis.ncomp), k),
           "n_pre_sessions": len(gm["pre"]),
           "n_pre_trials": int(sum(c[1] for c in gm["pre"])), "epochs": {}}
    rng = np.random.default_rng(seed)
    for ep in ("acute", "subacute", "chronic"):
        if not gm.get(ep):
            continue
        r = epoch_null(gm["pre"], gm[ep], t, stats=stats, k=k, n_draw=n_draw, rng=rng)
        if r is not None:
            out["epochs"][ep] = r
    if verbose:
        print(report(out), flush=True)
    return out


def _q(v):
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def report(out):
    """The per-animal table. RULE 8: printed per animal before anything is pooled over four."""
    L = [f"{out['animal']} {out['align']} gate={out['gate']}  K={out['k']}  "
         f"chance={out['chance']:.3f}  pre: {out['n_pre_sessions']} sessions / "
         f"{out['n_pre_trials']} trials",
         f"  {'epoch':9s} {'match (trials B/E, sess)':26s} {'OVERLAP obs':>12s} "
         f"{'matched null':>22s} {'SCALE obs':>10s} {'matched null':>22s} {'frac no worse':>13s}"]
    for ep, r in out["epochs"].items():
        oc, nc = _q(r["obs_cos"]), _q(r["null_cos"])
        os_, ns = _q(r["obs_scale"]), _q(r["null_scale"])
        match = (f"{np.median(r['n_B']):.0f}/{r['n_target']:.0f}, "
                 f"{np.median(r['k_B']):.0f}/{r['k_target']:.0f}")
        L.append(f"  {ep:9s} {match:26s} {oc[0]:12.3f} "
                 f"{f'{nc[0]:.3f} [{nc[1]:.3f},{nc[2]:.3f}]':>22s} {os_[0]:10.3f} "
                 f"{f'{ns[0]:.3f} [{ns[1]:.3f},{ns[2]:.3f}]':>22s} "
                 f"{r['p_not_worse_cos']:6.2f}/{r['p_not_bigger_scale']:<6.2f}")
    L.append("  frac no worse = draws where observed overlap was NOT below / scale NOT above its "
             "matched null. NOT a p-value (see the module docstring).")
    return "\n".join(L)


def save(out, out_dir):
    """Persist through `results_store`, like every other figure input in this arm."""
    from wfield_local import results_store as rs

    payload = {"epochs": {ep: {k_: v for k_, v in r.items()} for ep, r in out["epochs"].items()},
               **{k_: v for k_, v in out.items() if k_ != "epochs"}}
    return rs.save(out_dir, "cd_overlap_null",
                   f"{out['animal']}_{out['align']}_{out['gate']}", payload,
                   meta={"course_version": cdt.COURSE_VERSION, "cim_var": cdt.CIM_VAR,
                         "k": out["k"], "chance": out["chance"], "n_draw": N_DRAW,
                         "animal": out["animal"], "align": out["align"], "gate": out["gate"]})


def _one(item):
    """Module-level worker: the animal is the parallel unit, as in `cd_trajectories` (rule 6)."""
    from wfield_local import cd_overlap_null as m

    out = m.run_animal(item["animal"], item["align"], item["gate"], n_draw=item["n_draw"],
                       seed=item["seed"], verbose=False)
    if out is None:
        return None
    if item.get("out"):
        m.save(out, item["out"])
    return m.report(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--align", nargs="+", default=["precue"], choices=("precue", "cue", "lick"))
    ap.add_argument("--gate", default="lick", choices=tuple(cdt.GATES))
    ap.add_argument("--draws", type=int, default=N_DRAW)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="directory to persist into (its results/ subdir); default is the "
                         "server analysis-figure root")
    args = ap.parse_args(argv)

    # ONE definition of where this arm's output lives, shared with `cd_trajectories` -- so the
    # null's dumps land beside the results they calibrate rather than in a second place.
    out = Path(args.out) if args.out else cdt.default_out()
    out.mkdir(parents=True, exist_ok=True)

    items = [{"animal": a, "align": al, "gate": args.gate, "n_draw": args.draws,
              "seed": args.seed, "out": str(out)}
             for a in (args.animal or ["PS92", "PS93", "PS94", "PS95"]) for al in args.align]

    from wfield_local import analysis_kit as ak

    res, fail = ak.fan_sessions(items, _one, jobs=args.jobs,
                               key=lambda it: (it["animal"], it["align"]), label="animal")
    for _it, val in res:
        if val:
            print(val, flush=True)
    for f in fail:
        print(f"!! {f}", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
