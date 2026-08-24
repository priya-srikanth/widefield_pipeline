"""THE CONTROL THE OUTCOME-BLIND ARM NEEDS: can OUTCOME COMPOSITION alone rotate a position axis?

The outcome-blind arm (`position_axes` classes `poststroke_all*`) exists because the outcome split
cuts the trials the wrong way for the positions that matter -- far_R lives in the MISS class
post-stroke while its partners live in the LICK class, so no pair can contrast a failing position
against a performed one. Pooling every trial at a position regardless of outcome fixes the counts.

WHAT IT CANNOT FIX BY ITSELF. Post-stroke the outcome composition differs BY POSITION: far_R is
almost all misses, the close positions almost all licks. The pre-stroke reference is mixed but
mostly licks. So a "changed" verdict on far_R|far_L could be the lesion changing the position code,
or it could be the ordinary lick/no-lick difference showing up because the two sides of the contrast
are drawn from different outcome mixtures.

THIS MEASURES THAT DIRECTLY, INSIDE PRE-STROKE, where there is no lesion. For each pair (a, b) it
builds the same asymmetry by hand -- position `a` from MISS-while-working trials, position `b` from
LICK trials, both PRE-STROKE -- and asks how far that axis sits from the pre-stroke outcome-blind
axis for the same pair, disattenuated exactly as `position_axes` does.

    disatt ~ 0.9   composition asymmetry does not rotate the axis; the post-stroke result stands
    disatt ~ 0.4   composition alone reproduces the post-stroke effect; the arm is confounded

The pre-cue window is lick-free by construction, so neither side contains a movement -- what is
being tested is whether the STATE that precedes a miss is enough to move the axis.

Read against the matched drift null (`axis_holdout_null.py`), not against 1.0: this comparison has
its own noise, and both numbers are on the same disattenuated scale.
"""
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.position_axes import MIN_FIT, MIN_PRE, MIN_REL, PAIRS, axis, compare
from wfield_local.position_coding_directions import _gate_all, engagement_axis
from wfield_local.precue_engagement_states import features_with_indices

ALIGN = "precue"


def blind(pos, XE, XU, e_pre, en, un, p_miss):
    """Every PRE-STROKE trial at a position, lick and miss-while-working together."""
    parts = [XE[e_pre & (en == pos)]]
    m = p_miss & (un == pos) if len(un) else np.zeros(0, bool)
    if len(un) and m.any():
        parts.append(XU[m])
    return np.vstack(parts)

for animal in ("PS92", "PS93", "PS94", "PS95"):
    pre = [x for x in config.phase_labels("pre") if x.startswith(animal)]
    post = [x for x in config.phase_labels("post") if x.startswith(animal)]
    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    feat = features_with_indices(basis, nolick_ref="cue")
    XE, YE, GE, _B, XU, YU, kept, _c, GU = pool_sessions(
        pre + post, source="locanmf", align=ALIGN, post_s=2.0, features=feat)
    g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
    if g is None:
        continue
    not_eng, _a, _b = g
    rng = np.random.default_rng(0)
    pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
    e_pre = np.isin(GE, list(pre_i))
    GU = np.asarray(GU)
    u_pre = np.isin(GU, list(pre_i)) if len(GU) else np.zeros(0, bool)
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU]) if len(YU) else np.zeros(0, str)
    e_ax = engagement_axis(XE[e_pre], XU[u_pre]) if len(u_pre) and u_pre.sum() >= 10 else None
    # PRE-STROKE MISS-WHILE-WORKING: no-lick trials that are not part of the terminal sated tail.
    # Excluding the tail matters here more than anywhere -- an axis fitted on the last twenty minutes
    # of a session against licks from the whole session would be measuring satiety, not outcome.
    p_miss = u_pre & ~not_eng

    print("=" * 92)
    print(f"{animal}: composition null -- pre-stroke MISS(a) vs LICK(b) against the "
          f"outcome-blind pre-stroke axis")
    vals = []
    for a, b in PAIRS:
        mA = XU[p_miss & (un == a)] if len(un) else np.zeros((0, XE.shape[1]))
        lB = XE[e_pre & (en == b)]
        allA = blind(a, XE, XU, e_pre, en, un, p_miss)
        allB = blind(b, XE, XU, e_pre, en, un, p_miss)
        if min(len(allA), len(allB)) < MIN_PRE or min(len(mA), len(lB)) < MIN_FIT:
            continue
        cell = compare(axis(allA, allB, e_ax), allA, allB, mA, lB, e_ax, rng)
        d, rp, rq = cell.get("disattenuated"), cell.get("r_pre"), cell.get("r_post")
        if d is None or rp is None or rq is None or min(rp, rq) < MIN_REL:
            continue
        vals.append(d)
        print(f"   {a + '|' + b:<26} n=[{len(mA):>4},{len(lB):>4}]  disatt {d:+.2f}   "
              f"(cos {cell['cos']:+.2f}, r_pre {rp:+.2f}, r_post {rq:+.2f})")
    if vals:
        print(f"   COMPOSITION NULL over {len(vals)} pairs: median {np.median(vals):+.2f}   "
              f"[{np.percentile(vals, 25):+.2f}, {np.percentile(vals, 75):+.2f}]")
    else:
        print("   no pair had enough pre-stroke miss-while-working trials to test")
