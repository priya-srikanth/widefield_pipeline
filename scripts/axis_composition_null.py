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

THE TEST IS BUILT INSIDE PRE-STROKE, where there is no lesion: for each pair (a, b) take position
`a` from NO-LICK trials and position `b` from LICK trials -- the same asymmetry, by hand -- and
measure how far that axis sits from the pre-stroke outcome-blind axis for the same pair,
disattenuated exactly as `position_axes` does. Position `a` is the more affected of the two in every
pair, because PAIRS is built in BY_SEVERITY order, so the asymmetry points the same way as the
post-stroke one.

    disatt ~ 0.9   composition asymmetry does not rotate the axis; the post-stroke result stands
    disatt ~ 0.4   composition alone reproduces the post-stroke effect; the arm is confounded

TWO ARMS, because the obvious one turns out to be unmeasurable:

  MISS-WHILE-WORKING -- the exact analogue of the post-stroke miss. Pre-stroke it barely exists:
      these animals almost never tried and failed before the lesion, so nearly every pre-stroke
      no-lick trial belongs to the terminal sated tail. PS92 has no testable pair at all. The
      "NOT TESTABLE" branch prints the per-position counts, because the absence is itself the
      finding.

  DISENGAGED (the sated tail) -- Priya's suggestion, 2026-08-23, and the arm that can actually be
      run. It is ONE-SIDED and must be read that way. The sated tail is a FAR larger state
      difference than a miss ("a fundamentally different animal state"), so:
          it does NOT rotate the axis  ->  a mere outcome difference cannot either. Conservative,
                                           and the post-stroke far_R result stands.
          it DOES rotate the axis      ->  inconclusive, NOT a refutation: satiety is not the
                                           post-stroke miss state, and this over-states the effect.
      An UPPER BOUND on composition sensitivity, not an estimate of it.

WHAT IS ALREADY DEFENDED. Every axis here is orthogonalised against the engagement axis (fitted
pre-stroke lick vs pre-stroke no-lick), so the LINEAR engagement component is already removed from
both the reference and the test axis. What this control measures is what SURVIVES that -- the part
of the state difference that is not one global direction.

The pre-cue window is lick-free by construction, so neither side contains a movement: what is being
tested is whether the STATE preceding a non-lick is enough to move the position axis.

Read the numbers against the matched drift null (`axis_holdout_null.py`), not against 1.0 -- this
comparison has its own noise, and both are on the same disattenuated scale.
"""
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.position_axes import MIN_FIT, MIN_PRE, MIN_REL, PAIRS, axis, compare
from wfield_local.position_coding_directions import BY_SEVERITY, _gate_all, engagement_axis
from wfield_local.precue_engagement_states import features_with_indices

ALIGN = "precue"


def blind(pos, XE, XU, e_pre, en, un, p_work):
    """The outcome-blind PRE-STROKE reference: lick + miss-while-working, sated tail EXCLUDED.

    Excluded because `poststroke_all_working` -- the arm being controlled -- excludes it on both
    sides. The reference has to be the same population, or the control tests the wrong thing.
    """
    parts = [XE[e_pre & (en == pos)]]
    m = p_work & (un == pos) if len(un) else np.zeros(0, bool)
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
    p_work = u_pre & ~not_eng          # tried and missed -- pre-stroke this is nearly empty
    p_stop = u_pre & not_eng           # the terminal sated tail -- the upper-bound arm

    print("=" * 96)
    print(f"{animal}: composition null -- pre-stroke NO-LICK(a) vs LICK(b) against the "
          f"outcome-blind pre-stroke axis")
    print(f"   pre-stroke no-lick trials: {int(u_pre.sum())} total = "
          f"{int(p_work.sum())} miss-while-working + {int(p_stop.sum())} sated tail")
    UPPER = "DISENGAGED / sated tail (UPPER BOUND -- a larger state difference than a miss)"
    for arm, mask in (("MISS-WHILE-WORKING (the exact analogue)", p_work), (UPPER, p_stop)):
        print(f"   -- {arm}")
        vals = []
        for a, b in PAIRS:
            mA = XU[mask & (un == a)] if len(un) else np.zeros((0, XE.shape[1]))
            lB = XE[e_pre & (en == b)]
            allA = blind(a, XE, XU, e_pre, en, un, p_work)
            allB = blind(b, XE, XU, e_pre, en, un, p_work)
            if min(len(allA), len(allB)) < MIN_PRE or min(len(mA), len(lB)) < MIN_FIT:
                continue
            cell = compare(axis(allA, allB, e_ax), allA, allB, mA, lB, e_ax, rng)
            d, rp, rq = cell.get("disattenuated"), cell.get("r_pre"), cell.get("r_post")
            if d is None or rp is None or rq is None or min(rp, rq) < MIN_REL:
                continue
            vals.append(d)
            print(f"      {a + '|' + b:<26} n=[{len(mA):>4},{len(lB):>4}]  disatt {d:+.2f}   "
                  f"(cos {cell['cos']:+.2f}, r_pre {rp:+.2f}, r_post {rq:+.2f})")
        if vals:
            print(f"      MEDIAN over {len(vals)} pairs: {np.median(vals):+.2f}   "
                  f"[{np.percentile(vals, 25):+.2f}, {np.percentile(vals, 75):+.2f}]")
        else:
            per_pos = ({q: int((mask & (un == q)).sum()) for q in BY_SEVERITY} if len(un) else {})
            print(f"      NOT TESTABLE -- per position: {per_pos} "
                  f"(need {MIN_FIT} at the no-lick side, and reliability >= {MIN_REL})")
