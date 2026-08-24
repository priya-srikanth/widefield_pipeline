"""How each PAIRWISE position axis changes after the lesion -- per pair, per post-stroke session.

    python -m wfield_local.position_axes [--animals PS94 ...] [--align precue] [--output <dir>]

Priya, 2026-08-23: "what is changing about how lateralized tongue movements are encoded in cortex?",
then "across positions", then "we need to do this on a per-session basis since things evolve over
time". This is that analysis. For every pair of spout positions it fits the axis separating them on
PRE-STROKE lick trials and asks three questions of the post-stroke data, which must be kept apart:

    IS IT DIFFERENT?      cos(pre-stroke axis, post-stroke axis)
    DOES IT STILL EXIST?  the post-stroke axis's OWN split-half reliability
    IS IT THE LESION?     the same comparison with the CLASS held fixed (lick vs lick)

WHY "DOES IT STILL EXIST" IS SEPARATE. A low cosine with the pre-stroke axis shows the post-stroke
axis is DIFFERENT. It cannot show that a coherent new axis exists -- a direction fitted through trials
carrying no position information at all also scores near zero. PS94's far lateral cosine of +0.053 is
exactly what nothing looks like. The first version of this analysis reported that the lateral axis
"re-forms" with only the cosine in hand, which does not support it. Measured, the answer differs by
cell: PS92's far MISS axis reproduces itself at +0.781 while its close MISS axis reproduces at +0.078.

RELIABILITY IS ALSO A CEILING, and getting this wrong inflated an earlier result. An axis cannot
resemble another axis more than it resembles itself, so an observed cosine must be judged against
what two axes of THAT reliability would score if they were measuring the SAME thing:

    expected cos if unchanged  =  sqrt(r_pre * r_post)
    disattenuated ratio        =  observed cos / that

Judging the observed cosine against the PRE-STROKE floor alone ignores the post-stroke axis's own
noise. Doing that made PS94 (raw 0.756 against a 0.952 floor) look clearly worse than PS95, when
disattenuated they are 0.87 and 0.84 -- indistinguishable. PS93 stays an outlier either way (0.37).

WHY PER SESSION. The pooled view averages every post-stroke day into one number, and the classes
move: PS95's pooled coding value of +1.05 was a near-perfect first day and a halved fifth one. A
recovery and a collapse average to "no change".

WHY THE FLOOR IS SPLIT-HALF OF PRE-STROKE LICK, subsampled to the comparison's own n -- same animal,
same state, no lesion. Pre-stroke NO-LICK was the obvious null and is a bad one: those trials are the
sated tail, "a fundamentally different animal state" (Priya), and orthogonalising against the
engagement axis removes only its linear component.

THE WINDOW IS PRE-CUE and lick-free by construction, so NEITHER class contains a movement and the
lick/no-lick difference is not a movement difference. The class confound is handled by the same-class
control instead of by discarding the miss trials -- those are the only place the contraversive
positions survive at all, since the animal stopped licking there (far_R|far_L is n=5 for PS94 in the
lick class against n=399 in the miss class).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.paths import PathResolver
from wfield_local.position_coding_directions import (
    BY_SEVERITY,
    _gate_all,
    direction,
    engagement_axis,
    orthogonalise,
)
from wfield_local.precue_engagement_states import features_with_indices

FAR = ("far_R", "far_center", "far_L")
SIDE = {"far_R": "R", "close_R": "R", "far_L": "L", "close_L": "L",
        "far_center": "C", "close_center": "C"}

MIN_FIT = 20        #: trials per position below which an axis is not fitted
MIN_PRE = 40        #: pre-stroke trials per position needed for the reference axis
DRAWS = 30          #: split-half repeats (per session, so kept modest)
NO_AXIS = 0.25      #: own-reliability below this means there is no axis to compare
#: RELIABILITY BELOW THIS MAKES THE DISATTENUATED RATIO UNUSABLE. Dividing by sqrt(r_pre * r_post)
#: amplifies the noise in BOTH estimates, and at per-session trial counts that noise dominates:
#: measured on the 2026-08-23 cohort, 31% of per-session ratios came out ABOVE 1.0 -- an impossible
#: value, since an axis cannot resemble another more than its own ceiling -- against 9% of pooled
#: ones. Per session only 39% of cells clear this bar (median reliability +0.45) against 59% pooled
#: (+0.56). Cells below it are reported as too noisy rather than given a verdict they cannot support.
MIN_REL = 0.5


def pair_type(a, b):
    """What the contrast VARIES -- the thing the six position labels hide."""
    same_ring = (a in FAR) == (b in FAR)
    sa, sb = SIDE[a], SIDE[b]
    if same_ring:
        return "lateral" if "C" not in (sa, sb) else "lateral-centre"
    return "distance" if sa == sb else "diagonal"


PAIRS = [(a, b) for a, b in itertools.combinations(BY_SEVERITY, 2)]


def axis(L, R, e_ax):
    w = direction(L, R, "dom")
    return orthogonalise(w, e_ax) if e_ax is not None else w


def split_half(L, R, e_ax, n_l, n_r, rng, draws=DRAWS):
    """Mean cos between axes fitted on two DISJOINT halves -- the axis's reliability at that n.

    Disjoint or nothing: overlapping halves share trials, which drags the estimate toward 1 and would
    make every real comparison look like a change.
    """
    if n_l < 5 or n_r < 5 or 2 * n_l > len(L) or 2 * n_r > len(R):
        return None
    out = []
    for _ in range(draws):
        iL, iR = rng.permutation(len(L)), rng.permutation(len(R))
        out.append(float(axis(L[iL[:n_l]], R[iR[:n_r]], e_ax)
                         @ axis(L[iL[n_l:2 * n_l]], R[iR[n_r:2 * n_r]], e_ax)))
    return float(np.mean(out))


def compare(w_pre, pL, pR, qL, qR, e_ax, rng):
    """One cell: the observed cosine, both reliabilities, and the disattenuated ratio."""
    if len(qL) < MIN_FIT or len(qR) < MIN_FIT:
        return {"n": [len(qL), len(qR)], "too_few": True}
    cos = float(w_pre @ axis(qL, qR, e_ax))
    r_pre = split_half(pL, pR, e_ax, min(len(qL), len(pL) // 2), min(len(qR), len(pR) // 2), rng)
    r_post = split_half(qL, qR, e_ax, len(qL) // 2, len(qR) // 2, rng)
    ratio = None
    if r_pre is not None and r_post is not None and r_pre > 0 and r_post > 0:
        ratio = cos / float(np.sqrt(r_pre * r_post))
    return {"n": [len(qL), len(qR)], "cos": cos, "r_pre": r_pre, "r_post": r_post,
            "disattenuated": ratio}


def verdict(cell):
    """The reading the numbers license -- never a headline the cosine alone cannot support."""
    if cell.get("too_few"):
        return f"n={cell['n']} not fitted"
    if cell.get("r_post") is None or cell.get("r_pre") is None:
        return f"cos {cell['cos']:+.2f} (no reliability -- too few to split)"
    if cell["r_post"] < NO_AXIS:
        return f"NO AXIS post-stroke (reliability {cell['r_post']:+.2f}) -- nothing to compare"
    d = cell.get("disattenuated")
    if d is None:
        return f"cos {cell['cos']:+.2f}, reliability {cell['r_post']:+.2f}"
    if min(cell["r_pre"], cell["r_post"]) < MIN_REL:
        return (f"TOO NOISY (reliability pre {cell['r_pre']:+.2f} / post {cell['r_post']:+.2f}; "
                f"cos {cell['cos']:+.2f})")
    if d > 1.0:
        return (f"AT CEILING (disatt {d:.2f} > 1 -- indistinguishable from unchanged, and the "
                f"estimate is unstable)")
    if d >= 0.9:
        return f"UNCHANGED (disatt {d:.2f}, reliability {cell['r_post']:+.2f})"
    if d >= 0.7:
        return f"modestly changed (disatt {d:.2f}, reliability {cell['r_post']:+.2f})"
    return f"CHANGED, stable new axis (disatt {d:.2f}, reliability {cell['r_post']:+.2f})"



def decompose(w_post, refs):
    """What a changed axis is MADE OF, in the coordinate system of pre-stroke axes.

    "The axis changed" is a statement about an ANGLE and says nothing about where the code went.
    This projects the post-stroke axis onto interpretable pre-stroke references -- its OWN axis (the
    part that survives), every OTHER position axis (has far_R|far_L become close_R|close_L, i.e. one
    generic left-right axis instead of a per-ring one?), the close-vs-far axis (has a lateral
    contrast become a distance one?) and the engagement axis (has it become a state signal?).

    THE RESIDUAL IS THE POINT. `refs` are not orthogonal to each other, so the individual cosines
    overlap and must not be summed. What IS well defined is how much of the axis lies outside their
    span: residual = 1 - ||projection onto span(refs)||^2, by least squares. A changed axis that is
    near-orthogonal to EVERY pre-stroke reference has moved into structure that did not previously
    exist -- a far stronger claim than "it changed", and one the cosines alone cannot support.
    """
    w = np.asarray(w_post, float)
    names = list(refs)
    R = np.stack([np.asarray(refs[k], float) for k in names])            # (nref, nfeat)
    cos = {k: float(w @ R[i]) for i, k in enumerate(names)}
    # least-squares projection onto the span; rcond kept explicit so a rank-deficient ref set
    # (two references that are nearly the same axis) degrades gracefully rather than exploding
    coef, *_ = np.linalg.lstsq(R.T, w, rcond=1e-8)
    proj = R.T @ coef
    resid = float(max(0.0, 1.0 - float(proj @ proj)))
    return {"cos": cos, "residual_outside_span": resid}



def prestroke_null(XE, en, GE, pre_i, e_ax, rng, block=2):
    """The NULL every post-stroke cosine must beat: pooled pre-stroke vs a HELD-OUT pre-stroke block.

    STRUCTURALLY MATCHED, which the earlier nulls were not. The post-stroke comparison is (pooled
    pre-stroke) vs (a post-stroke subset); a session-to-session drift rate answers a different
    question and OVERSTATES the null, because pooling eleven sessions averages drift out and gives a
    more stable reference than any single session. Holding out a block and pooling the rest is the
    identical operation with no lesion in it.

    BLOCKS OF TWO because one session is not enough in every animal: PS92's median per-session axis
    reliability is +0.47, right at the 0.5 gate, so single-session holdouts yielded 1-2 usable cells
    and none at all at the matched 8/14 gap. Two sessions took it to 18 (Priya, 2026-08-23).

    Returns per-block medians plus the pooled null. A block containing a KNOWN-DEGRADED session drags
    it down -- 8/13 is the documented case (PS95 recorded single-channel for 32 min; 197/871 cues fell
    outside the surviving imaging span, and after the coverage fix it still reached only 0.78 against
    ~0.90 for that animal). Excluding it lifted PS92's null from +0.73 to +0.79, so the blocks are
    returned individually rather than only as a summary.
    """
    idx = sorted(pre_i)
    blocks = [idx[i:i + block] for i in range(0, len(idx), block)]
    out = {}
    for blk in blocks:
        vals = []
        rest = np.isin(GE, [i for i in idx if i not in blk])
        inb = np.isin(GE, blk)
        for a, b in PAIRS:
            pL, pR = XE[rest & (en == a)], XE[rest & (en == b)]
            hL, hR = XE[inb & (en == a)], XE[inb & (en == b)]
            if min(len(pL), len(pR)) < MIN_PRE or min(len(hL), len(hR)) < MIN_FIT:
                continue
            r_pool = split_half(pL, pR, e_ax, min(len(hL), len(pL) // 2),
                                min(len(hR), len(pR) // 2), rng)
            r_held = split_half(hL, hR, e_ax, len(hL) // 2, len(hR) // 2, rng)
            if not r_pool or not r_held or min(r_pool, r_held) < MIN_REL:
                continue
            dis = float(axis(pL, pR, e_ax) @ axis(hL, hR, e_ax)) / float(np.sqrt(r_pool * r_held))
            if dis <= 1.0:
                vals.append(dis)
        if vals:
            out["+".join(str(i) for i in blk)] = {"n": len(vals),
                                                  "median": float(np.median(vals)),
                                                  "values": [float(v) for v in vals]}
    allv = [v for r in out.values() for v in r["values"]]
    return {"blocks": out,
            "null_median": (float(np.median(allv)) if allv else None),
            "n_cells": len(allv)}


def run_animal(animal, align="precue", seed=0, pool_n=1):
    rng = np.random.default_rng(seed)
    pre = [x for x in config.phase_labels("pre") if x.startswith(animal)]
    post = [x for x in config.phase_labels("post") if x.startswith(animal)]
    if not pre or not post:
        return None
    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    feat = features_with_indices(basis, nolick_ref="cue")
    post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    pooled = pool_sessions(pre + post, source="locanmf", align=align, post_s=post_s, features=feat)
    if pooled is None:
        return None
    XE, YE, GE, _B, XU, YU, kept, _c, GU = pooled
    g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
    if g is None:
        return None
    not_eng, _a, _b = g
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    e_pre = np.isin(GE, list(pre_i))
    GU = np.asarray(GU)
    u_pre = np.isin(GU, list(pre_i)) if len(GU) else np.zeros(0, bool)
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU]) if len(YU) else np.zeros(0, str)
    e_ax = engagement_axis(XE[e_pre], XU[u_pre]) if len(u_pre) and u_pre.sum() >= 10 else None
    miss = (~u_pre) & ~not_eng
    # STOPPED IS THE CONTROL FOR THE MISS RESULT (Priya, 2026-08-23). A changed axis on
    # miss-while-working trials is consistent with a failed PLAN -- and equally with a STATE change
    # that happens to be commoner on failed trials. Those separate here: changed in miss but intact
    # in stopped is specific to attempted-and-failed; changed in both is not about the plan at all.
    # It matters because misses concentrate LATE in a session, exactly where the close-vs-far state
    # drift lives. Stopped is terminal by construction, so its counts are small and end-skewed --
    # a null there needs its reliability read before it is called "intact".
    stopped = (~u_pre) & not_eng

    def all_at(pos, pre_phase, include_stopped=True, blk=None):
        """Every trial at a position REGARDLESS OF OUTCOME, lick and no-lick together.

        THE OUTCOME SPLIT CUTS THE TRIALS THE WRONG WAY for the positions that matter. far_R lives
        in the MISS class post-stroke (355-399 trials) while its partners live in the LICK class, so
        no pair can contrast a failing position against a performed one -- PS94's far_R|far_L miss
        cell is n=[116, 24], limited by the position the animal still performs (Priya, 2026-08-23:
        "we need to be able to use data from the most affected spout positions!").

        In the PRE-CUE window this is also the more defensible axis: nothing has happened yet, so
        splitting by outcome conditions on the future. What it cannot do is separate position from
        outcome-correlated state, because post-stroke the outcome composition differs BY position --
        far_R is mostly misses, close mostly licks.
        """
        me = (e_pre if pre_phase else ~e_pre) & (en == pos)
        mu = (u_pre if pre_phase else ~u_pre) & (un == pos) if len(un) else np.zeros(0, bool)
        if blk is not None:
            me = me & np.isin(GE, blk)
            if len(un):
                mu = mu & np.isin(GU, blk)
        if len(un) and not include_stopped:
            # STOPPED trials are the terminal non-recovering run -- end-of-session concentrated, and
            # therefore carrying the close-vs-far state drift that contaminates exactly this kind of
            # comparison. Excluded from BOTH phases, because the pre-stroke equivalent is the sated
            # tail, "a fundamentally different animal state" (Priya, 2026-08-23).
            mu = mu & ~not_eng
        parts = [XE[me]] + ([XU[mu]] if len(un) and mu.any() else [])
        return np.vstack(parts) if parts else XE[me]
    post_idx = sorted((i for i in range(len(kept)) if i not in pre_i),
                      key=lambda i: kept[i].split("_")[1])
    # CONSECUTIVE BLOCKS of pool_n sessions. Per session the measure mostly fails for want of
    # trials -- median split-half reliability +0.45, only 39% of cells clearing 0.5, against +0.56
    # and 59% pooled over everything. Blocking trades time resolution for the reliability the
    # disattenuation needs, and consecutive (not sliding) blocks keep every trial in exactly one
    # cell so the blocks stay independent (Priya, 2026-08-23).
    blocks = [post_idx[i:i + pool_n] for i in range(0, len(post_idx), pool_n)]

    # ---- reference axes, all fitted PRE-STROKE, for decomposing whatever the post-stroke axis
    # turned into. Every position pair, plus the two task dimensions and the engagement axis.
    refs = {}
    for x, y in PAIRS:
        aL, aR = XE[e_pre & (en == x)], XE[e_pre & (en == y)]
        if len(aL) >= MIN_PRE and len(aR) >= MIN_PRE:
            refs[f"{x}|{y}"] = axis(aL, aR, e_ax)
    farm = e_pre & np.isin(en, FAR)
    closem = e_pre & ~np.isin(en, FAR) & np.isin(en, BY_SEVERITY)
    if farm.sum() >= MIN_PRE and closem.sum() >= MIN_PRE:
        refs["AXIS_close_vs_far"] = axis(XE[closem], XE[farm], e_ax)
    if e_ax is not None:
        refs["AXIS_engagement"] = np.asarray(e_ax, float)

    out = {"animal": animal, "align": align, "basis_id": basis.basis_id,
           "reference_axes": sorted(refs),
           "prestroke_null": prestroke_null(XE, en, GE, pre_i, e_ax, rng),
           "pairs": {}}
    for a, b in PAIRS:
        pL, pR = XE[e_pre & (en == a)], XE[e_pre & (en == b)]
        if len(pL) < MIN_PRE or len(pR) < MIN_PRE:
            continue
        w_pre = axis(pL, pR, e_ax)
        rec = {"type": pair_type(a, b), "n_pre": [len(pL), len(pR)],
               "pooled": {}, "sessions": {}}
        rec["pooled"]["poststroke_lick"] = compare(
            w_pre, pL, pR, XE[(~e_pre) & (en == a)], XE[(~e_pre) & (en == b)], e_ax, rng)
        rec["pooled"]["poststroke_miss_working"] = compare(
            w_pre, pL, pR, XU[miss & (un == a)], XU[miss & (un == b)], e_ax, rng)
        rec["pooled"]["poststroke_stopped"] = compare(
            w_pre, pL, pR, XU[stopped & (un == a)], XU[stopped & (un == b)], e_ax, rng)
        # ALL TRIALS, outcome-blind -- the only arm in which the most affected positions have
        # enough trials on BOTH sides of a contrast. Its reference is also outcome-blind.
        # BOTH VARIANTS, so the effect of including the terminal quit period is visible rather
        # than assumed. `_working` is the one to prefer; `all` is kept as its comparison.
        blind = {}
        for tag, keep_stopped in (("poststroke_all", True), ("poststroke_all_working", False)):
            aA = all_at(a, True, keep_stopped)
            aB = all_at(b, True, keep_stopped)
            if min(len(aA), len(aB)) < MIN_PRE:
                continue
            blind[tag] = (axis(aA, aB, e_ax), aA, aB, keep_stopped)
            rec["pooled"][tag] = compare(blind[tag][0], aA, aB,
                                         all_at(a, False, keep_stopped),
                                         all_at(b, False, keep_stopped), e_ax, rng)
        # WHAT the pooled post-stroke axis is made of, per class. Only where an axis exists at all:
        # decomposing a direction fitted through noise describes the noise.
        rec["decomposition"] = {}
        for cls, (qL, qR) in (("poststroke_lick", (XE[(~e_pre) & (en == a)],
                                                   XE[(~e_pre) & (en == b)])),
                              ("poststroke_miss_working", (XU[miss & (un == a)],
                                                           XU[miss & (un == b)])),
                              ("poststroke_stopped", (XU[stopped & (un == a)],
                                                      XU[stopped & (un == b)]))):
            cell = rec["pooled"][cls]
            if cell.get("too_few") or (cell.get("r_post") or 0) < NO_AXIS:
                continue
            rec["decomposition"][cls] = decompose(axis(qL, qR, e_ax), refs)

        for blk in blocks:
            lab = "+".join(kept[i].split("_")[1] for i in blk)
            inE, inU = np.isin(GE, blk), (np.isin(GU, blk) if len(GU) else np.zeros(0, bool))
            rec["sessions"][lab] = {
                "poststroke_lick": compare(w_pre, pL, pR, XE[(~e_pre) & inE & (en == a)],
                                           XE[(~e_pre) & inE & (en == b)], e_ax, rng),
                "poststroke_miss_working": compare(w_pre, pL, pR, XU[miss & inU & (un == a)],
                                                   XU[miss & inU & (un == b)], e_ax, rng),
                "poststroke_stopped": compare(w_pre, pL, pR, XU[stopped & inU & (un == a)],
                                              XU[stopped & inU & (un == b)], e_ax, rng)}
            # The outcome-blind arms per block too: the per-block view is what separated PS95's
            # recovery from PS93's progression, and it is worth nothing if it cannot be read in the
            # one arm where the most affected positions have trials on both sides.
            for tag, (w_blind, aA, aB, keep_stopped) in blind.items():
                rec["sessions"][lab][tag] = compare(
                    w_blind, aA, aB, all_at(a, False, keep_stopped, blk=blk),
                    all_at(b, False, keep_stopped, blk=blk), e_ax, rng)
        out["pairs"][f"{a}|{b}"] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--align", default="precue", choices=("precue", "cue", "lick"))
    ap.add_argument("--class", dest="cls", default="poststroke_lick",
                    choices=("poststroke_lick", "poststroke_miss_working",
                             "poststroke_stopped", "poststroke_all",
                             "poststroke_all_working"),
                    help="which class to PRINT; both are always computed and stored")
    ap.add_argument("--pool", type=int, default=1, metavar="N",
                    help="pool N CONSECUTIVE post-stroke sessions per cell (default 1). Per session "
                         "the measure is usually too noisy to interpret; N=2 roughly doubles the "
                         "trials per split-half.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for an in (config.normalize_animals(args.animals) or list(config.animals())):
        try:
            r = run_animal(an, align=args.align, pool_n=args.pool)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  !! {an}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            continue
        if not r:
            continue
        res[an] = r
        print(f"\n=== {an}  {args.align}  [{args.cls}]", flush=True)
        for key, rec in r["pairs"].items():
            print(f"  {key:<26}({rec['type']})", flush=True)
            cell = rec["pooled"].get(args.cls)
            if cell is None:                       # arm not computed for this pair (too few trials)
                print(f"      {'POOLED':<14}not computed for this class", flush=True)
                continue
            print(f"      {'POOLED':<14}{verdict(cell)}", flush=True)
            for lab, cells in rec["sessions"].items():
                if args.cls in cells:
                    print(f"      {lab.split('_')[-1]:<14}{verdict(cells[args.cls])}", flush=True)
    if res:
        p = out / (f"position_axes_{args.align}.json" if args.pool == 1
                   else f"position_axes_{args.align}_pool{args.pool}.json")
        p.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
