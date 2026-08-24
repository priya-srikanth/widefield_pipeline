"""ON- vs OFF-MANIFOLD: does post-stroke activity leave the pre-stroke subspace, or rearrange in it?

Priya, 2026-08-23. The Sadtler/Golub/Batista distinction: a WITHIN-manifold rearrangement keeps the
population in its existing covariance subspace and remaps the code inside it -- learnable in hours.
An OUTSIDE-manifold excursion requires activity patterns the circuit does not natively produce --
learned over days or not at all (Oby 2019). The two have different recovery prognoses, which is why
the distinction is worth making rather than just saying "the code changed".

WHAT WAS MEASURED BEFORE, AND WHY IT IS NOT THIS. The earlier residual projected changed CODING AXES
onto the span of pre-stroke CODING AXES. Coding axes are discriminant directions; the manifold is the
dominant COVARIANCE structure. Activity can sit entirely on-manifold while a coding axis rotates
within it, so that residual is closer to output-potent vs output-null (Kaufman 2014) than to on/off
manifold.

THE MANIFOLD HERE is defined on the LocaNMF COMPONENT space: every (trial, time-bin) is one point in
ncomp dimensions, PCA on pre-stroke points, top k explaining VAR_TARGET.

TWO QUESTIONS, kept apart:
  1. ACTIVITY -- what fraction of post-stroke activity variance lies on the pre-stroke manifold,
     against a CROSS-VALIDATED pre-stroke ceiling (held-out pre-stroke sessions, which bounds how
     high this can go given sampling alone).
  2. THE CODE -- what fraction of a coding axis lies within the manifold, post-stroke vs pre-stroke.
     This is the Sadtler question proper: within-manifold remap or outside-manifold excursion.

CAVEAT THAT LIMITS BOTH. The joint LocaNMF basis is ALREADY a fixed, anatomically constrained
dimensionality reduction (90 components). Anything genuinely outside its span is invisible here, so
"off-manifold" means off the pre-stroke manifold WITHIN the LocaNMF basis -- a lower bound on any
real excursion.
"""
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.position_axes import PAIRS, axis
from wfield_local.precue_engagement_states import features_with_indices

VAR_TARGET = 0.90
MIN_FIT = 25


def manifold(X, ncomp, target=VAR_TARGET):
    """(basis, k) from points reshaped to (trial*bin, ncomp). Mean-centred, as PCA requires."""
    P = X.reshape(-1, ncomp)
    mu = P.mean(0)
    _U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
    var = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12)
    k = int(np.searchsorted(var, target) + 1)
    return Vt[:k], k, mu


def vaf(X, B, mu, ncomp):
    """Variance of X accounted for by the subspace B (rows orthonormal)."""
    P = X.reshape(-1, ncomp) - mu
    tot = float((P ** 2).sum())
    if tot <= 0:
        return float("nan")
    return float(((P @ B.T) ** 2).sum() / tot)


def axis_in_manifold(w, B, ncomp):
    """Fraction of a coding axis lying IN the manifold, averaged over its time sub-bins."""
    W = np.asarray(w, float).reshape(-1, ncomp)          # (nbins, ncomp)
    out = []
    for row in W:
        n2 = float(row @ row)
        if n2 > 1e-12:
            out.append(float(((row @ B.T) ** 2).sum() / n2))
    return float(np.mean(out)) if out else float("nan")


for animal in ("PS92", "PS93", "PS94", "PS95"):
    pre = [x for x in config.phase_labels("pre") if x.startswith(animal)]
    post = [x for x in config.phase_labels("post") if x.startswith(animal)]
    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    ncomp = int(basis.ncomp)
    feat = features_with_indices(basis, nolick_ref="cue")
    XE, YE, GE, _B, _XU, _YU, kept, _c, _GU = pool_sessions(
        pre + post, source="locanmf", align="precue", post_s=2.0, features=feat)
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    e_pre = np.isin(GE, list(pre_i))
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])

    B, k, mu = manifold(XE[e_pre], ncomp)
    # CROSS-VALIDATED CEILING: rebuild the manifold leaving one pre-stroke session out, then score
    # that session. Bounds how high the post-stroke number could go on sampling alone.
    held = []
    for i in sorted(pre_i):
        tr = e_pre & (GE != i)
        if tr.sum() < 200 or (GE == i).sum() < 50:
            continue
        Bi, _ki, mui = manifold(XE[tr], ncomp)
        held.append(vaf(XE[(GE == i)], Bi, mui, ncomp))
    post_vaf = vaf(XE[~e_pre], B, mu, ncomp) if (~e_pre).sum() else float("nan")

    print("=" * 82)
    print(f"{animal}: manifold = {k}/{ncomp} dims for {VAR_TARGET:.0%} of pre-stroke variance")
    print(f"   ACTIVITY on the pre-stroke manifold:  post-stroke VAF {post_vaf:.3f}   "
          f"held-out pre-stroke ceiling {np.median(held):.3f} "
          f"[{min(held):.3f}-{max(held):.3f}]" if held else "")

    rows = []
    for a, b in PAIRS:
        pL, pR = XE[e_pre & (en == a)], XE[e_pre & (en == b)]
        qL, qR = XE[(~e_pre) & (en == a)], XE[(~e_pre) & (en == b)]
        if min(len(pL), len(pR)) < 40 or min(len(qL), len(qR)) < MIN_FIT:
            continue
        rows.append((f"{a}|{b}",
                     axis_in_manifold(axis(pL, pR, None), B, ncomp),
                     axis_in_manifold(axis(qL, qR, None), B, ncomp)))
    if rows:
        pv = np.array([r[1] for r in rows])
        qv = np.array([r[2] for r in rows])
        print(f"   THE CODE inside the manifold:  pre-stroke axes {np.median(pv):.3f}   "
              f"post-stroke axes {np.median(qv):.3f}   ({len(rows)} pairs)")
        worst = sorted(rows, key=lambda r: r[2])[:3]
        for key, p_, q_ in worst:
            print(f"      {key:<26} pre {p_:.3f} -> post {q_:.3f}")
