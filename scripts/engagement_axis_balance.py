"""HOW MUCH POSITION INFORMATION IS THE "ENGAGEMENT" AXIS CARRYING?

    python scripts/engagement_axis_balance.py

The engagement axis is `mean(pre-stroke lick) - mean(pre-stroke no-lick)`, and every position axis in
every arm is orthogonalised against it. That is only a state correction if the no-lick trials are
POSITION-NEUTRAL. They are not: PS93's pre-stroke no-contact rate is 26.8% at far_L against 0.4% at
close_R (`scripts/nocontact_census.py`), because it licked leftward without spout contact before any
lesion and licks are detected by contact.

THREE NUMBERS PER ANIMAL:

  cos(e_ax, balanced)   the current axis against the POSITION-BALANCED one (mean over positions of
                        each position's own lick-minus-no-lick difference). 1.0 means the imbalance
                        does not matter; well below 1 means the axis is partly a position contrast.
  cos(e_ax, far_L|rest) how much of the current axis IS the far_L-versus-rest direction -- the
                        specific contamination PS93's census predicts.
  worst pair            the position pair whose axis loses the most when orthogonalised against the
                        current axis, i.e. where the subtraction is doing the most work.

Nothing is rewritten: `engagement_axis` takes the balanced form only when handed position labels,
and no analysis passes them yet. This measures whether it should.
"""
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.position_axes import MIN_PRE, PAIRS
from wfield_local.position_coding_directions import (
    _gate_all,
    direction,
    engagement_axis,
    orthogonalise,
)
from wfield_local.precue_engagement_states import features_with_indices


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


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
    _not_eng, _a, _b = g
    pre_i = {i for i, lab in enumerate(kept) if lab in set(pre)}
    e_pre = np.isin(GE, list(pre_i))
    GU = np.asarray(GU)
    u_pre = np.isin(GU, list(pre_i)) if len(GU) else np.zeros(0, bool)
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU]) if len(YU) else np.zeros(0, str)

    XL, XN = XE[e_pre], XU[u_pre]
    pl, pn = en[e_pre], un[u_pre]
    e_now = engagement_axis(XL, XN)
    e_bal = engagement_axis(XL, XN, pl, pn)

    fl = unit(XE[e_pre & (en == "far_L")].mean(0) - XE[e_pre & (en != "far_L")].mean(0))
    print("=" * 88)
    print(f"{animal}   cos(current, balanced) = {float(e_now @ e_bal):+.3f}    "
          f"cos(current, far_L-vs-rest) = {float(e_now @ fl):+.3f}    "
          f"cos(balanced, far_L-vs-rest) = {float(e_bal @ fl):+.3f}")
    rows = []
    for a, b in PAIRS:
        L, R = XE[e_pre & (en == a)], XE[e_pre & (en == b)]
        if min(len(L), len(R)) < MIN_PRE:
            continue
        w = direction(L, R, "dom")
        rows.append((f"{a}|{b}", abs(float(w @ e_now)),
                     float(orthogonalise(w, e_now) @ orthogonalise(w, e_bal))))
    for key, loss, agree in sorted(rows, key=lambda r: -r[1])[:3]:
        print(f"   {key:<26} |cos with engagement axis| {loss:.3f}   "
              f"current-vs-balanced orthogonalised axes agree {agree:+.3f}")
