"""Two-session holdout, and what 8/13 does to the null.

TWO SESSIONS AT A TIME because PS92 could not be measured one at a time: its median per-session axis
reliability is +0.47, right at the 0.5 gate, so 1-2 cells survived per held-out session and none at
the matched 8/14 gap. Doubling the held-out trials should lift reliability toward ~0.64.

8/13 IS A KNOWN BAD SESSION and is still in the curated set. PS95 8/13 was recorded single-channel
for 32 min; the repair left 197/871 cues (23%) before any surviving imaging frame, and after the
coverage fix it recovered to 0.78 cue-aligned against ~0.90 for that animal -- i.e. still below its
own norm (docs/EXPERIMENT_ERRORS.md). It is also the extreme outlier in the single-session holdout:
PS95 -0.08 against 0.32-0.93, PS93 +0.53 against 0.89-0.99. Reported with and without.
"""
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.position_axes import MIN_FIT, MIN_PRE, PAIRS, axis, split_half
from wfield_local.precue_engagement_states import features_with_indices

rng = np.random.default_rng(0)

for animal in ("PS92", "PS93", "PS94", "PS95"):
    pre = [x for x in config.phase_labels("pre") if x.startswith(animal)]
    post = [x for x in config.phase_labels("post") if x.startswith(animal)]
    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    feat = features_with_indices(basis, nolick_ref="cue")
    XE, YE, GE, _B, _XU, _YU, kept, _c, _GU = pool_sessions(
        pre + post, source="locanmf", align="precue", post_s=2.0, features=feat)
    pre_i = sorted({i for i, l in enumerate(kept) if l in set(pre)}, key=lambda k: kept[k])
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    e_pre = np.isin(GE, pre_i)
    blocks = [pre_i[i:i + 2] for i in range(0, len(pre_i), 2)]

    print("=" * 88)
    print(f"{animal}: pooled-vs-held-out TWO pre-stroke sessions, disattenuated")
    keep_all, keep_no813 = [], []
    for blk in blocks:
        labs = [kept[i].split("_")[1] for i in blk]
        vals = []
        for a, b in PAIRS:
            rest = e_pre & ~np.isin(GE, blk)
            inb = np.isin(GE, blk)
            pL, pR = XE[rest & (en == a)], XE[rest & (en == b)]
            hL, hR = XE[inb & (en == a)], XE[inb & (en == b)]
            if min(len(pL), len(pR)) < MIN_PRE or min(len(hL), len(hR)) < MIN_FIT:
                continue
            r_pool = split_half(pL, pR, None, min(len(hL), len(pL) // 2),
                                min(len(hR), len(pR) // 2), rng)
            r_held = split_half(hL, hR, None, len(hL) // 2, len(hR) // 2, rng)
            if not r_pool or not r_held or min(r_pool, r_held) < 0.5:
                continue
            dis = float(axis(pL, pR, None) @ axis(hL, hR, None)) / float(np.sqrt(r_pool * r_held))
            if dis <= 1.0:
                vals.append(dis)
        if not vals:
            print(f"   {'+'.join(labs):<12} no cells passed")
            continue
        flag = "  <-- contains 8/13" if "0813" in labs else ""
        print(f"   {'+'.join(labs):<12} n={len(vals):<3} median {np.median(vals):+.2f}"
              f"   [{np.percentile(vals, 25):+.2f}, {np.percentile(vals, 75):+.2f}]{flag}")
        keep_all += vals
        if "0813" not in labs:
            keep_no813 += vals
    if keep_all:
        print(f"   NULL over all blocks      median {np.median(keep_all):+.2f} (n={len(keep_all)})")
    if keep_no813 and len(keep_no813) != len(keep_all):
        print(f"   NULL excluding 8/13       median {np.median(keep_no813):+.2f} "
              f"(n={len(keep_no813)})")
