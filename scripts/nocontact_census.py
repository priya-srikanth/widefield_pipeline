"""WHAT IS THE PRE-STROKE "NO-LICK" POPULATION, PER POSITION? Not what its label says.

    python scripts/nocontact_census.py

`position_axes` splits post-stroke no-lick trials into MISS-WHILE-WORKING and STOPPED, and applies
the same two masks to the PRE-STROKE side to build the outcome-blind reference and the engagement
axis. The post-stroke names were carried over to the pre-stroke side without being earned (Priya,
2026-08-24: "how are you calling miss-while-working pre-stroke? This was something we defined
post-stroke").

WHAT THE PRE-STROKE MASK ACTUALLY IS: a trial with NO SPOUT CONTACT that is not part of the terminal
disengagement run. Pre-stroke there is no motor deficit, so it is not "tried and failed" -- and
because licks are detected by CONTACT, an OFF-TARGET LICK (leftward into the air, no spout) lands in
it indistinguishably from an inattentive trial. This is the same contact-vs-attempt ambiguity already
recorded for the post-stroke classes; it transfers.

WHY IT IS NOT COSMETIC. The engagement axis is `mean(pre-stroke lick) − mean(pre-stroke no-lick)`,
and in PS92 and PS93 that no-lick side is 95%/93% this population. If it is largely off-target
licking, `e_ax` is a MOTOR-ERROR axis, not a motivational one -- and EVERY position axis in that
animal is orthogonalised against it, so motor-error structure is removed from all of them. Post-
stroke misses are also licks-without-contact, so this could suppress the very effect being looked
for, and would present as broad unlateralised change rather than a far_R-specific one. That is
exactly PS93's profile.

THE TESTABLE PREDICTION Priya raised: PS93 often licked LEFTWARD without spout contact pre-stroke,
so its no-contact trials should be enriched at far_L rather than spread evenly. This counts them.
An enrichment at the position CONTRALATERAL to the animal's later deficit, present BEFORE any
lesion, is behavioural evidence that these are off-target licks and not inattention.
"""
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.position_coding_directions import BY_SEVERITY, _gate_all
from wfield_local.precue_engagement_states import features_with_indices

for animal in ("PS92", "PS93", "PS94", "PS95"):
    pre = [x for x in config.phase_labels("pre") if x.startswith(animal)]
    post = [x for x in config.phase_labels("post") if x.startswith(animal)]
    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    feat = features_with_indices(basis, nolick_ref="cue")
    XE, YE, GE, _B, XU, YU, kept, _c, GU = pool_sessions(
        pre + post, source="locanmf", align="precue", post_s=2.0, features=feat)
    g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
    if g is None:
        continue
    not_eng, _a, _b = g
    pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
    e_pre = np.isin(GE, list(pre_i))
    GU = np.asarray(GU)
    u_pre = np.isin(GU, list(pre_i)) if len(GU) else np.zeros(0, bool)
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU]) if len(YU) else np.zeros(0, str)

    print("=" * 96)
    print(f"{animal}: PRE-STROKE trials per position -- contact, no-contact-while-working, tail")
    print(f"   {'position':<14}{'contact':>9}{'no-contact':>12}{'tail':>8}{'no-contact %':>14}")
    rates = {}
    for pos in BY_SEVERITY:
        n_lick = int((e_pre & (en == pos)).sum())
        n_work = int((u_pre & ~not_eng & (un == pos)).sum()) if len(un) else 0
        n_tail = int((u_pre & not_eng & (un == pos)).sum()) if len(un) else 0
        tot = n_lick + n_work + n_tail
        rates[pos] = n_work / tot if tot else 0.0
        print(f"   {pos:<14}{n_lick:>9}{n_work:>12}{n_tail:>8}{rates[pos]:>13.1%}")
    hi = max(rates, key=rates.get)
    ratio = rates[hi] / max(min(rates.values()), 1e-9)
    print(f"   -> highest no-contact rate at {hi} ({rates[hi]:.1%}), "
          f"{ratio:.1f}x the lowest position")
