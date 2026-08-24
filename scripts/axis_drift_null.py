"""THE DRIFT NULL: how much does a position axis change between PRE-STROKE sessions?

Every "CHANGED" verdict so far compares a post-stroke axis against a POOLED pre-stroke reference,
with a SAME-DAY split-half as the floor. That floor measures sampling noise, not time. If pre-stroke
axes already drift day to day -- and representational drift is well documented in healthy animals
(Ziv 2013, Driscoll 2017; Gallego 2020 argues the manifold is stable while tuning drifts) -- then a
post-stroke cosine of 0.5 may be unremarkable. This measures the pre-stroke rate directly.

DISATTENUATED, because per-session axes are noisy: a raw cross-session cosine is capped by each
session's own reliability, so cos(i,j) / sqrt(r_i * r_j) is what isolates real change from sampling.

THE JUNE-AUGUST GAP IS THE NATURAL EXPERIMENT (Priya, 2026-08-23). The curated pre-stroke set is
6/6-6/8 and then 8/6 onward, so the same animals give within-week comparisons (1-8 days) and
across-two-month ones (~60 days) with no lesion in between.
"""
import itertools
from collections import defaultdict

import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.position_axes import PAIRS, axis, split_half
from wfield_local.precue_engagement_states import features_with_indices

MIN_SESS = 25          # trials per position per session to fit an axis at all
rng = np.random.default_rng(0)


def day(lab):
    m = lab.split("_")[1]
    return int(m[:2]) * 31 + int(m[2:])


for animal in ("PS92", "PS93", "PS94", "PS95"):
    pre = [x for x in config.phase_labels("pre") if x.startswith(animal)]
    post = [x for x in config.phase_labels("post") if x.startswith(animal)]
    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    feat = features_with_indices(basis, nolick_ref="cue")
    XE, YE, GE, _B, XU, YU, kept, _c, _GU = pool_sessions(
        pre + post, source="locanmf", align="precue", post_s=2.0, features=feat)
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])

    per = defaultdict(dict)      # pair -> session idx -> (w, reliability)
    for a, b in PAIRS:
        for i in sorted(pre_i, key=lambda k: kept[k]):
            L = XE[(GE == i) & (en == a)]
            R = XE[(GE == i) & (en == b)]
            if len(L) < MIN_SESS or len(R) < MIN_SESS:
                continue
            r = split_half(L, R, None, len(L) // 2, len(R) // 2, rng)
            if r is None or r <= 0.05:
                continue
            per[f"{a}|{b}"][i] = (axis(L, R, None), r)

    buckets = defaultdict(list)
    for sess in per.values():
        for i, j in itertools.combinations(sorted(sess), 2):
            (wi, ri), (wj, rj) = sess[i], sess[j]
            gap = abs(day(kept[j]) - day(kept[i]))
            dis = float(wi @ wj) / float(np.sqrt(ri * rj))
            tag = ("JUNE-AUGUST" if gap > 30 else
                   ("within 1-3 d" if gap <= 3 else "within 4-10 d"))
            buckets[tag].append(dis)
    rels = [r for sess in per.values() for _w, r in sess.values()]
    print("=" * 78)
    print(f"{animal}: {len(per)} pairs with per-session axes, median session reliability "
          f"{np.median(rels):+.2f}" if rels else f"{animal}: no per-session axes")
    for tag in ("within 1-3 d", "within 4-10 d", "JUNE-AUGUST"):
        v = np.array(buckets.get(tag, []))
        if len(v):
            print(f"   {tag:<15} n={len(v):<5} disattenuated cos {np.median(v):+.2f}  "
                  f"[{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}]")
