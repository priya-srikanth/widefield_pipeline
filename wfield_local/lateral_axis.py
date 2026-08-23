"""How the LATERAL (left-vs-right spout) component of the position code changes after the lesion.

    python -m wfield_local.lateral_axis [--animals PS94 ...] [--align precue] [--output <dir>]

Priya, 2026-08-23: "what is changing about how lateralized tongue movements are encoded in cortex?"
The six-label position code cannot answer that directly -- it confounds SIDE with DISTANCE -- so this
splits out the lateral component and asks three questions about it that must be kept apart:

    IS IT DIFFERENT?      cos(pre-stroke axis, post-stroke axis), against a matched noise floor
    DOES IT STILL EXIST?  the post-stroke axis's OWN split-half reproducibility
    IS IT THE LESION?     the same comparison with the CLASS held fixed (lick vs lick)

THE AXIS. w = mean(LEFT trials) - mean(RIGHT trials), difference-of-means, unit-normalised,
orthogonalised against the pre-stroke engagement axis, in the (LocaNMF component x time sub-bin)
feature space. Every animal in this cohort is lesioned LEFT, so RIGHT spouts are contraversive.

FITTED WITHIN A RING, always. Pooling close_L+far_L against close_R+far_R removes the close-vs-far
dimension only if the two sets match in close/far composition, and post-stroke they need not:
measured imbalance was fine for PS94 (+0.01) and PS95 (-0.04) but -0.17 for PS93 and -0.11 for PS92,
because misses concentrate at far positions. Within a ring no distance component can enter, and the
split also asks whether lateral encoding degrades at far positions specifically or everywhere.

WHY EVERY COSINE NEEDS A FLOOR. A cosine of 0.3 means one thing if two halves of clean pre-stroke
data reproduce each other at 0.85 and nothing at all if they reproduce each other at 0.35. The floor
is SPLIT-HALF of pre-stroke LICK subsampled to the comparison's own n -- same animal, same state, no
lesion. Pre-stroke NO-LICK was the obvious null and is a bad one: those trials are the sated tail,
"a fundamentally different animal state" (Priya), and orthogonalising against the engagement axis
removes only its linear component.

WHY "DOES IT STILL EXIST" IS A SEPARATE QUESTION. A low cosine with the pre-stroke axis shows the
post-stroke axis is DIFFERENT. It cannot show that a coherent new axis exists -- a fitted direction
through trials carrying no lateral information at all would also score near zero. PS94's far value of
+0.053 is exactly what nothing looks like. The distinguishing measurement is the post-stroke axis's
own split-half: near 0 means the axis is GONE, high means a STABLE NEW axis near-orthogonal to the
old one. The first version of this analysis reported "the lateral axis re-forms" with only the first
number in hand, which does not support it. Measured, the answer differs by cell -- PS92's far MISS
axis reproduces itself at +0.781 while its close MISS axis reproduces at +0.078 -- so the word has to
be earned per cell rather than asserted for the cohort.

THE WINDOW IS PRE-CUE and lick-free by construction, so NEITHER class contains a movement and the
lick/no-lick difference is not a movement difference. That is why the class confound is addressed by
the same-class control rather than by discarding the miss trials: those trials are the only place the
contraversive positions survive at all (far_R|far_L is n=5 for PS94 in the lick class and n=399 in
the miss class), because the animal stopped licking there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.paths import PathResolver
from wfield_local.position_coding_directions import (
    _gate_all,
    direction,
    engagement_axis,
    orthogonalise,
)
from wfield_local.precue_engagement_states import features_with_indices

#: the two pure-lateral contrasts. Never pooled -- see the module docstring.
RINGS = {"far": ("far_L", "far_R"), "close": ("close_L", "close_R")}
MIN_FIT = 20            #: trials per side below which an axis is not fitted at all
MIN_PRE = 40            #: pre-stroke trials per side needed for the reference axis
DRAWS = 40              #: split-half repeats


def side_axis(L, R, e_ax):
    """Unit vector separating LEFT-spout from RIGHT-spout trials, engagement projected out."""
    w = direction(L, R, "dom")
    return orthogonalise(w, e_ax) if e_ax is not None else w


def split_half(L, R, e_ax, n_l, n_r, rng, draws=DRAWS):
    """cos between axes fitted on two DISJOINT halves of the same trials.

    Used two ways, and they answer different questions. On PRE-STROKE trials subsampled to a
    comparison's n it is the NOISE FLOOR -- what a cosine can reach with that much data. On the
    POST-STROKE trials themselves it asks whether a coherent axis exists there at all.
    """
    if n_l < 5 or n_r < 5 or 2 * n_l > len(L) or 2 * n_r > len(R):
        return None
    out = []
    for _ in range(draws):
        iL, iR = rng.permutation(len(L)), rng.permutation(len(R))
        a = side_axis(L[iL[:n_l]], R[iR[:n_r]], e_ax)
        b = side_axis(L[iL[n_l:2 * n_l]], R[iR[n_r:2 * n_r]], e_ax)
        out.append(float(a @ b))
    return np.asarray(out)


def _band(v):
    return None if v is None else {"mean": float(v.mean()),
                                   "p5": float(np.percentile(v, 5)),
                                   "p95": float(np.percentile(v, 95))}


def run_animal(animal, align="precue", seed=0):
    """Every lateral-axis measure for one animal, as a plain dict. None if it cannot be built."""
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
    u_pre = np.isin(np.asarray(GU), list(pre_i)) if len(GU) else np.zeros(0, bool)
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU]) if len(YU) else np.zeros(0, str)
    e_ax = (engagement_axis(XE[e_pre], XU[u_pre])
            if len(u_pre) and u_pre.sum() >= 10 else None)
    reg = np.asarray(basis.regions)
    ncomp = int(basis.ncomp)

    def hemi(w):
        """Fraction of |w| carried by LEFT-hemisphere components (+code = _left, the same rule
        hemispheric_intensity uses; LocaNMF constrains each component to one area and side)."""
        mag = np.sqrt((np.asarray(w).reshape(-1, ncomp) ** 2).sum(0))
        tot = mag.sum() or 1.0
        return {"left": float(mag[reg > 0].sum() / tot), "right": float(mag[reg < 0].sum() / tot)}

    out = {"animal": animal, "align": align, "basis_id": basis.basis_id, "rings": {}}
    miss = (~u_pre) & ~not_eng
    for ring, (Lp, Rp) in RINGS.items():
        pL, pR = XE[e_pre & (en == Lp)], XE[e_pre & (en == Rp)]
        if len(pL) < MIN_PRE or len(pR) < MIN_PRE:
            continue
        w_pre = side_axis(pL, pR, e_ax)
        rec = {"n_pre": [len(pL), len(pR)], "hemisphere_pre": hemi(w_pre), "classes": {}}
        arms = {"poststroke_lick": (XE[(~e_pre) & (en == Lp)], XE[(~e_pre) & (en == Rp)]),
                "poststroke_miss_working": (XU[miss & (un == Lp)], XU[miss & (un == Rp)])}
        for cls, (qL, qR) in arms.items():
            if len(qL) < MIN_FIT or len(qR) < MIN_FIT:
                rec["classes"][cls] = {"n": [len(qL), len(qR)], "too_few": True}
                continue
            w_q = side_axis(qL, qR, e_ax)
            rec["classes"][cls] = {
                "n": [len(qL), len(qR)],
                "cos_with_prestroke": float(w_pre @ w_q),
                # the floor this cosine has to beat, at ITS OWN trial count
                "noise_floor": _band(split_half(pL, pR, e_ax,
                                                min(len(qL), len(pL) // 2),
                                                min(len(qR), len(pR) // 2), rng)),
                # does a coherent axis exist in this class at all?
                "own_reproducibility": _band(split_half(qL, qR, e_ax,
                                                        len(qL) // 2, len(qR) // 2, rng)),
                "hemisphere": hemi(w_q),
            }
        out["rings"][ring] = rec
    return out


def verdict(cell):
    """One line per cell, saying which of the three questions the numbers actually answer."""
    if cell.get("too_few"):
        return f"n={cell['n']} -- not fitted"
    c, fl, rep = cell["cos_with_prestroke"], cell.get("noise_floor"), cell.get("own_reproducibility")
    if fl is None or rep is None:
        return f"cos {c:+.3f} (no floor -- too few trials to split)"
    if rep["mean"] < 0.25:
        return (f"cos {c:+.3f}; the axis DOES NOT REPRODUCE ITSELF "
                f"({rep['mean']:+.3f}) -- no lateral code here to compare")
    if c >= fl["p5"]:
        return f"cos {c:+.3f} WITHIN the floor {fl['p5']:+.3f}-{fl['p95']:+.3f} -- no detected change"
    return (f"cos {c:+.3f} BELOW the floor {fl['p5']:+.3f}-{fl['p95']:+.3f}, and the axis reproduces "
            f"at {rep['mean']:+.3f} -- a stable DIFFERENT axis")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--align", default="precue", choices=("precue", "cue", "lick"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for an in (config.normalize_animals(args.animals) or list(config.animals())):
        try:
            r = run_animal(an, align=args.align)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  !! {an}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            continue
        if not r:
            continue
        res[an] = r
        print(f"=== {an}  ({args.align})", flush=True)
        for ring, rec in r["rings"].items():
            print(f"  {ring:<6} pre-stroke n {rec['n_pre'][0]}/{rec['n_pre'][1]}  "
                  f"LEFT-hemisphere share {rec['hemisphere_pre']['left']:.2f}", flush=True)
            for cls, cell in rec["classes"].items():
                print(f"      {cls:<26}{verdict(cell)}", flush=True)
    if res:
        p = out / f"lateral_axis_{args.align}.json"
        p.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
