"""Per-position pre-stroke coding directions, and where the post-stroke failure classes fall on them.

Priya, 2026-08-20. REPORTED, not a filter: nothing here changes what any existing analysis uses.

VOCABULARY. "pre-cue"/"post-cue" collided with "pre-stroke"/"post-stroke", so the WINDOW is named
**ENL / cue / lick** and the PHASE stays **pre-stroke / post-stroke**. The internal align tokens are
unchanged (``precue``/``cue``/``lick``) because they are baked into 292 figure filenames, the
``section_g.json`` keys and the per-session caches; renaming those would strand every existing
figure under a name nothing reads, which is the exact failure this deck spent 2026-08-20 clearing.

THE DIRECTION. For each spout position P, a direction is fitted on PRE-STROKE trials WITH A
SUCCESSFUL LICK in the response window: P against the other positions. It is therefore "the pattern
that precedes/accompanies a successful lick to P" -- defined entirely where behaviour is intact.

WHY ONE DIRECTION PER POSITION RATHER THAN ONE ENGAGEMENT AXIS. The two post-stroke phenomena differ
enormously in position composition -- MISS-WHILE-WORKING is 34-44% far_R, STOPPED is near-uniform, total
variation 0.31-0.65 between them -- and ENL activity CARRIES position. A single position-blind axis
therefore compares the spout, not the state, which is what produced a spurious PS95 "effect" on the
first pass. Fitting per position and comparing only WITHIN a position removes that by construction.

THE TWO POST-STROKE FAILURE MODES (Priya, 2026-08-20):
  MISS WHILE WORKING  the animal is still working the task and fails to lick at THIS position.
                      Position-specific, and graded by SEVERITY: far_R > far_center > far_L >
                      close_R > close_center > close_L -- contraversive within each ring, far
                      worse than close throughout.
  STOPPED             the animal has quit for the day and licks nowhere. Verified position-GENERAL:
                      inside that window the response rate is ~0 at every position, close included.

WHAT EACH WINDOW CAN ANSWER, AND WHAT IT CANNOT:
  ENL   all five classes. Nothing has happened yet, and the window is already lick-free by
        construction (``decode.precue_lickfree``), so it is the clean one.
  cue   all five classes. Note that a lick trial contains its lick from ~140 ms (median first-lick
        latency is 0.137-0.255 s pre-stroke, minimum 0.109 s), so there is NO movement-free cue
        window to retreat to. The per-position construction is what keeps this interpretable:
        movement is common to every training class, so it cannot define the direction.
  lick  ONLY the classes that have licks. A no-lick trial has no lick to align to. This matters
        operationally: at ``align="lick"`` ``_trial_features`` still RETURNS no-lick trials, but
        referenced to the cue instead -- so they arrive populated, plausible, and on a different
        alignment from everything they would be compared with. They are excluded explicitly here
        rather than assumed absent (the same trap as the post-lick confusion bug of 2026-08-20).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local import config, joint_locanmf
from wfield_local.locanmf_frozen_decoder import pool_sessions
from wfield_local.paths import PathResolver
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
from wfield_local.precue_engagement_states import (
    _disc,
    _positions_for,
    _session_tables,
    engagement_gate,
    features_with_indices,
)

#: internal align token -> the name used in every figure, label and printed line.
ALIGNS = (("precue", "ENL"), ("cue", "cue"), ("lick", "lick"))

#: most impaired first (Priya, 2026-08-20).
SEVERITY = {"far_R": 1, "far_center": 2, "far_L": 3,
            "close_R": 4, "close_center": 5, "close_L": 6}
BY_SEVERITY = [p for p, _ in sorted(SEVERITY.items(), key=lambda kv: kv[1])]

#: classes each window can carry. `lick` drops both no-lick classes: no lick, no alignment point.
CLASSES_FULL = ("prestroke_lick", "prestroke_nolick", "poststroke_lick", "poststroke_miss_working", "poststroke_stopped")
CLASSES_LICK = ("prestroke_lick", "poststroke_lick")
MIN_TRIALS = 12      # below this a value is drawn HOLLOW, not dropped
FLOOR_TRIALS = 3     # below this there is nothing to average at all


# ---------------------------------------------------------------------------------------------
# THE DIRECTION, AND WHAT IS REPORTED ON IT
# ---------------------------------------------------------------------------------------------
# The feature space has one axis per (LocaNMF component, time sub-bin) -- 90 x 4 = 360 for an ENL
# window, 8 bins for lick -- so a direction is a weight per component PER MOMENT in the window.
#
# TWO CONSTRUCTIONS, because they fail differently:
#   dom  w = mean(P) - mean(not P), unit-normalised. The literature's "coding direction": no
#        hyperparameter, ignores covariance, blunt but has nothing to tune.
#   lr   the logistic weight vector, unit-normalised. Accounts for covariance so it usually
#        separates better, but the geometry depends on C, which is an arbitrary choice.
# Reported together: if the two disagree about an ordering, that disagreement is the finding.
#
# LINEAR PROJECTION, NOT A PROBABILITY (Priya, 2026-08-21). predict_proba was carried over from the
# earlier engagement axis and was never a considered choice. A sigmoid SATURATES, and these
# directions are strong -- PS94 close_R/close_L reach AUC 0.98, so pre-stroke lick already sits in
# the flat region. Degradation measured from a saturated reference is understated, and unevenly so
# between positions of different separability, which corrupts exactly the two orderings this
# analysis is for. The squashing constant also depends on C and the feature scale, so probabilities
# are not commensurable across the panels that get compared side by side.
#
# POLE-NORMALISED so the number means something: 0 = pre-stroke NOT-this-position, 1 = pre-stroke
# LICK at this position. Every value then reads as "fraction of the normal position-P signature",
# comparable across positions and animals, and values outside [0, 1] are informative rather than
# clipped.
#: "_orth" variants are the SAME construction with the engagement axis projected out. Kept as
#: separate methods rather than replacing the originals, so both are on disk and comparable.
CD_METHODS = ("dom", "lr", "dom_orth", "lr_orth")


def engagement_axis(X_lick, X_nolick):
    """Unit vector separating pre-stroke LICK from pre-stroke NO-LICK.

    Built PRE-STROKE, where "no lick" is unambiguous -- there is no motor deficit to confuse with
    intent. Whether the same axis describes post-stroke non-responding is an assumption, not a fact,
    and it is the main thing to distrust about the orthogonalised variants.
    """
    e = np.asarray(X_lick).mean(0) - np.asarray(X_nolick).mean(0)
    n = float(np.linalg.norm(e))
    return e / n if n > 0 else e


def orthogonalise(w, e):
    """Gram-Schmidt: remove w's component along e, renormalise.

    After this w.e == 0, so a trial's score CANNOT move because it was a lick or a no-lick trial.
    The cost is real: any position information lying along e goes with it, so the projection answers
    the narrower question "how much position structure is there that engagement cannot explain".
    A class that still separates is strong evidence; one that stops separating is ambiguous, not
    negative.
    """
    w2 = np.asarray(w) - float(np.asarray(w) @ np.asarray(e)) * np.asarray(e)
    n = float(np.linalg.norm(w2))
    return w2 / n if n > 0 else w2


def direction(Xp, Xn, method="dom"):
    """Unit vector separating Xp from Xn in (component x bin) space."""
    if method == "dom":
        w = np.asarray(Xp).mean(0) - np.asarray(Xn).mean(0)
    else:
        m = _disc().fit(np.vstack([Xn, Xp]),
                        np.concatenate([np.zeros(len(Xn)), np.ones(len(Xp))]))
        w = np.asarray(m[-1].coef_).ravel()
        sc = m[0]                       # undo the StandardScaler so w lives in feature space
        w = w / np.where(sc.scale_ > 0, sc.scale_, 1.0)
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


def poles(Xp, Xn, w):
    """The two pre-stroke anchors on this axis: (zero point = not-P, unit point = P)."""
    return float(np.mean(np.asarray(Xn) @ w)), float(np.mean(np.asarray(Xp) @ w))


def project(X, w, p0, p1):
    """Pole-normalised projection: 0 = pre-stroke not-P, 1 = pre-stroke lick at P."""
    if len(X) == 0:
        return np.empty(0)
    v = np.asarray(X) @ w
    d = (p1 - p0)
    return (v - p0) / d if abs(d) > 1e-12 else v - p0


def _gate_all(feat, kept, XE, YE, GE, XU, YU, GU):
    """`not_eng` per no-lick trial, or None if the bookkeeping does not line up."""
    tables = _session_tables(feat, kept)
    if not tables:
        return None
    not_eng = []
    for si in range(len(kept)):
        t = tables.get(si)
        if t is None:
            continue
        ie, inl = feat.indices[kept[si]]
        pos = _positions_for(tables, si, YE, GE, YU, GU, ie, inl)
        ne = engagement_gate(t["order"], t["responded"], pos)
        bne = {int(k): bool(v) for k, v in zip(t["order"], ne)}
        not_eng += [bne.get(int(k), False) for k in inl]
    not_eng = np.array(not_eng, bool)
    return not_eng if len(not_eng) == len(XU) else None


def _stats(vals):
    """mean, SEM and SD over the trials in one (session, position, class) cell.

    SEM rather than SD on the plot: every point IS a mean, and the question is how well that mean is
    pinned down. SD is carried in the JSON as well so the trial-to-trial spread stays available --
    post-stroke variability is itself a candidate readout, and a SEM alone would hide it.
    """
    v = np.asarray(vals, float)
    n = int(v.size)
    if n < FLOOR_TRIALS:
        return {"n": n, "mean": None, "sem": None, "sd": None, "low_n": True}
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": float(np.mean(v)), "sem": (sd / np.sqrt(n)) if n > 1 else None,
            "sd": sd, "low_n": n < MIN_TRIALS}


def _cells(vals):
    """mean / SEM / SD over one cell's trials. SEM on the plot (each point IS a mean); SD kept in
    the JSON because post-stroke variability is itself a candidate readout."""
    v = np.asarray(vals, float)
    n = int(v.size)
    if n < FLOOR_TRIALS:
        return {"n": n, "mean": None, "sem": None, "sd": None, "low_n": True}
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": float(np.mean(v)), "sem": (sd / np.sqrt(n)) if n > 1 else None,
            "sd": sd, "low_n": n < MIN_TRIALS}


def run_animal(animal, align="precue", verbose=True, methods=CD_METHODS):
    """Per-position directions for one animal: time course, 6x6 cross matrix, and pairwise axes."""
    disp = dict(ALIGNS)[align]
    post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    if not pre or not post:
        return None
    try:
        from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
        basis = joint_locanmf.load(animal, sessions=_ALL)
    except FileNotFoundError as ex:
        print(f"[coding_dirs] {animal}: {ex}", flush=True)
        return None

    feat = features_with_indices(basis)
    pooled = pool_sessions(pre + post, source="locanmf", align=align, post_s=post_s, features=feat)
    if pooled is None:
        return None
    XE, YE, GE, _BE, XU, YU, kept, _c, GU = pooled
    YU = YU.astype(int)
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    e_pre, u_pre = np.isin(GE, list(pre_i)), np.isin(GU, list(pre_i))
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU])

    use_nolick = align != "lick"
    not_eng = None
    if use_nolick:
        not_eng = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
        if not_eng is None:
            print(f"[coding_dirs] {animal} {disp}: trial bookkeeping mismatch -- skipped", flush=True)
            return None

    order = sorted(range(len(kept)), key=lambda i: kept[i].split("_")[1])
    sessions = [{"label": kept[i], "date": kept[i].split("_")[1],
                 "phase": ("pre" if i in pre_i else "post")} for i in order]

    def trials(cls, pos=None, sess=None):
        """Feature rows for one class, optionally restricted to a position and/or session."""
        if cls == "prestroke_lick":
            m = e_pre.copy()
        elif cls == "poststroke_lick":
            m = ~e_pre
        elif cls == "prestroke_nolick":
            m = u_pre.copy()
        elif cls == "poststroke_miss_working":
            m = ~u_pre & ~not_eng
        else:
            m = ~u_pre & not_eng
        X, names, g = ((XE, en, GE) if cls in ("prestroke_lick", "poststroke_lick") else (XU, un, GU))
        if pos is not None:
            m = m & (names == pos)
        if sess is not None:
            m = m & (g == sess)
        return X[m]

    classes = CLASSES_LICK if not use_nolick else CLASSES_FULL
    out = {"animal": animal, "align": align, "window": disp, "basis_id": basis.basis_id,
           "ncomp": int(basis.ncomp), "n_features": int(XE.shape[1]), "sessions": sessions,
           "methods": {}}

    # the engagement axis, from pre-stroke lick vs pre-stroke no-lick
    e_axis = (engagement_axis(XE[e_pre], XU[u_pre])
              if (use_nolick and e_pre.sum() and u_pre.sum()) else None)

    for meth in methods:
        base_m, do_orth = (meth[:-5], True) if meth.endswith("_orth") else (meth, False)
        if do_orth and e_axis is None:
            continue
        res = {"positions": {}, "cross_matrix": {}, "pairwise": {},
               "orthogonalised": do_orth, "base_method": base_m}
        W, W_loso = {}, {}
        for D in BY_SEVERITY:
            Xp, Xn = XE[e_pre & (en == D)], XE[e_pre & (en != D)]
            if len(Xp) < 20 or len(Xn) < 20:
                continue
            w_raw = direction(Xp, Xn, base_m)
            # CONTAMINATION: how much engagement each position axis carries, before anything is
            # removed. This is the number that decides whether orthogonalising is warranted.
            cos_e = float(w_raw @ e_axis) if e_axis is not None else None
            w = orthogonalise(w_raw, e_axis) if do_orth else w_raw
            p0, p1 = poles(Xp, Xn, w)
            W[D] = (w, p0, p1)
            # RAW poles kept: a normalised value far outside [0, 1] is either a genuine offset or a
            # near-zero denominator, and only the raw spread distinguishes those two.
            res.setdefault("axes", {})[D] = {
                "pole_notP": p0, "pole_P": p1, "spread": p1 - p0,
                "engagement_cos": cos_e,
                "raw_prestroke_nolick": (float(np.mean(XU[u_pre] @ w)) if u_pre.sum() else None)}
            # HELD-OUT DIRECTIONS for the class the direction is FITTED ON. Pre-stroke LICK trials
            # are the training set, so scoring them on the full-fit direction puts them at their own
            # optimum and every other class looks worse than it is by construction. Every other
            # class -- pre-stroke NO-LICK included, since training uses only the lick arm -- is
            # already out of sample and uses the full fit.
            for i in pre_i:
                tr = e_pre & (GE != i)
                Ap, An = XE[tr & (en == D)], XE[tr & (en != D)]
                if len(Ap) < 20 or len(An) < 20:
                    continue
                wi = direction(Ap, An, base_m)
                if do_orth:
                    wi = orthogonalise(wi, e_axis)
                W_loso[(D, i)] = (wi, *poles(Ap, An, wi))

        def proj(cls, X, D, sess=None, _W=W, _WL=W_loso):
            """Project with the RIGHT direction: held-out for pre-stroke lick, full fit otherwise."""
            if cls == "prestroke_lick" and sess is not None and (D, sess) in _WL:
                w_, a_, b_ = _WL[(D, sess)]
            elif D in _W:
                w_, a_, b_ = _W[D]
            else:
                return np.empty(0)
            return project(X, w_, a_, b_)

        def proj_pre_lick_pooled(D, pos=None, _p=None):
            """Pre-stroke lick pooled across sessions, each scored on ITS held-out direction."""
            chunks = []
            for i in pre_i:
                m = e_pre & (GE == i) & ((en == pos) if pos else np.ones(len(en), bool))
                if m.sum():
                    chunks.append(proj("prestroke_lick", XE[m], D, sess=i))
            return np.concatenate(chunks) if chunks else np.empty(0)

        # ---- 1. TIME COURSE on each position's OWN direction (the diagonal) -------------------
        for P in BY_SEVERITY:
            if P not in W:
                res["positions"][P] = {"severity_rank": SEVERITY[P]}
                continue
            rec = {"severity_rank": SEVERITY[P], "by_session": {}}
            for i in order:
                cell = {}
                for c in classes:
                    if (i in pre_i) != c.startswith("prestroke"):
                        continue
                    cell[c] = _cells(proj(c, trials(c, P, i), P, sess=i))
                rec["by_session"][kept[i]] = cell
            res["positions"][P] = rec

        # ---- 2. CROSS MATRIX: every class at true position P, on EVERY direction D ------------
        # Priya, 2026-08-21: "see if the far center post stroke activity aligns better with
        # pre-stroke far R coding direction". The diagonal is (1); the OFF-diagonal is the question.
        for c in classes:
            mat = {}
            for P in BY_SEVERITY:
                row = {}
                for D in BY_SEVERITY:
                    if D not in W:
                        continue
                    row[D] = _cells(proj_pre_lick_pooled(D, P) if c == "prestroke_lick"
                                    else proj(c, trials(c, P), D))
                mat[P] = row
            res["cross_matrix"][c] = mat

        # ---- 3. PAIRWISE axes: A vs B directly, no five-way mixture in the contrast -----------
        # Sharper for the remapping question: "not-P" above averages five positions, so a trial can
        # look unlike P without the axis saying WHICH other position it resembles.
        for c in classes:
            pw = {}
            for A in BY_SEVERITY:
                for B in BY_SEVERITY:
                    if A == B:
                        continue
                    XA, XB = XE[e_pre & (en == A)], XE[e_pre & (en == B)]
                    if len(XA) < 20 or len(XB) < 20:
                        continue
                    if c == "prestroke_lick":
                        # held out session by session, same reason as above
                        ch = []
                        for i in pre_i:
                            tr = e_pre & (GE != i)
                            Ap, Bp = XE[tr & (en == A)], XE[tr & (en == B)]
                            if len(Ap) < 20 or len(Bp) < 20:
                                continue
                            wi = direction(Ap, Bp, base_m)
                            if do_orth:
                                wi = orthogonalise(wi, e_axis)
                            r0, r1 = poles(Ap, Bp, wi)
                            m = e_pre & (GE == i) & (en == A)
                            if m.sum():
                                ch.append(project(XE[m], wi, r0, r1))
                        pw[f"{A}|{B}"] = _cells(np.concatenate(ch) if ch else [])
                        continue
                    wab = direction(XA, XB, base_m)
                    if do_orth:
                        wab = orthogonalise(wab, e_axis)
                    q0, q1 = poles(XA, XB, wab)        # 0 = pre-stroke B, 1 = pre-stroke A
                    pw[f"{A}|{B}"] = _cells(project(trials(c, A), wab, q0, q1))
            res["pairwise"][c] = pw
        out["methods"][meth] = res

    if verbose:
        nd = len(out["methods"].get(methods[0], {}).get("positions", {}))
        print(f"  {animal} [{disp}]: {nd}/6 positions, {len(sessions)} sessions, "
              f"{out['n_features']} features, methods={list(methods)}", flush=True)
    return out


STYLE = {"prestroke_lick": ("tab:blue", "o", "pre-stroke LICK (held out)"),
         "prestroke_nolick": ("tab:grey", "s", "pre-stroke NO-LICK (sated / not working)"),
         "poststroke_lick": ("tab:green", "^", "post-stroke LICK"),
         "poststroke_miss_working": ("tab:red", "D", "post-stroke MISS while still working"),
         "poststroke_stopped": ("tab:purple", "v", "post-stroke STOPPED (quit for the day)")}


def figure_animal(res, out, align="precue", meth="dom"):
    """ONE figure per animal: a panel per spout position, each class over TIME.

    Per animal and over time because the post-stroke course is expected to MOVE (Priya,
    2026-08-21) -- pooling would average a recovery and a collapse into the same number.
    """
    if not res or meth not in res.get("methods", {}):
        return None
    disp, R = dict(ALIGNS)[align], res["methods"][meth]
    classes = CLASSES_LICK if align == "lick" else CLASSES_FULL
    sess = res["sessions"]
    x = np.arange(len(sess))
    bx = next((i - 0.5 for i, sn in enumerate(sess) if sn["phase"] == "post"), None)

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.8), squeeze=False, sharey=True, sharex=True)
    for k, P in enumerate(BY_SEVERITY):
        ax = axes[k // 3][k % 3]
        bys = (R["positions"].get(P) or {}).get("by_session") or {}
        for c in classes:
            col, mk, lab = STYLE[c]
            cells = [(bys.get(sn["label"], {}).get(c) or {}) for sn in sess]
            ys = [cc.get("mean") for cc in cells]
            es = [cc.get("sem") for cc in cells]
            ax.errorbar(x, [np.nan if y is None else y for y in ys],
                        yerr=[0 if e is None else e for e in es], fmt="-", marker=mk, color=col,
                        ecolor=col, elinewidth=1.1, capsize=2.5, ms=5.5, lw=1.4,
                        label=(lab if k == 0 else None))
            for xi, cc in zip(x, cells):
                y, n, low = cc.get("mean"), cc.get("n", 0), cc.get("low_n")
                if y is None and n:
                    ax.text(xi, 0.02, f"n={n}", ha="center", va="bottom", fontsize=5,
                            rotation=90, color=col, style="italic",
                            transform=ax.get_xaxis_transform())
                elif y is not None and low:
                    ax.plot([xi], [y], marker=mk, ms=7, markerfacecolor="none",
                            markeredgecolor=col, markeredgewidth=1.5, zorder=4)
        # the two pre-stroke anchors the scale is built from
        ax.axhline(1.0, color="tab:blue", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0.0, color="k", ls=":", lw=1.0, alpha=0.7)
        if bx is not None:
            ax.axvline(bx, color="k", ls="--", lw=1.4)
            ax.text(bx, 1.005, " stroke", ha="left", va="bottom", fontsize=7.5, fontweight="bold",
                    transform=ax.get_xaxis_transform())
        ax.set_title(f"{P}   (rank {(R['positions'].get(P) or {}).get('severity_rank', '?')} of 6)",
                     fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([sn["date"] for sn in sess], rotation=60, ha="right", fontsize=6.5)
        ax.grid(alpha=0.25)
        if k % 3 == 0:
            ax.set_ylabel("projection  (0 = pre-stroke not-P, 1 = pre-stroke lick at P)")
    fig.legend(loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle(
        f"{res['animal']} \u2014 {disp} window, {meth.upper()} coding direction. Each position's "
        f"PRE-STROKE successful-lick direction, every class over time.\nLINEAR projection, "
        f"pole-normalised: 0 = pre-stroke NOT-this-position, 1 = pre-stroke LICK here (dotted "
        f"lines). Panels MOST IMPAIRED first. Bars = SEM. Hollow = <{MIN_TRIALS} trials.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    q = Path(out) / f"coding_direction_{disp}_{meth}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_cross(res, out, align="precue", meth="dom"):
    """6x6: every class at TRUE position (rows) scored on EVERY direction (columns).

    THE BASELINE PANEL IS THE POINT. Neighbouring positions are intrinsically similar before any
    stroke -- pre-stroke far_center already scores 0.76 on the far_R direction -- so a raw
    off-diagonal value says nothing on its own. Every post-stroke panel is therefore drawn as a
    DIFFERENCE from the pre-stroke LICK matrix, and remapping is a departure from that baseline
    rather than a large number (Priya, 2026-08-21).

    READ WITH THE ONE-VS-REST CAVEAT. "Not P" mixes the five other positions, and for the MIDDLE
    positions that mixture is majority-far, so the axis becomes largely close-vs-far: PS94's
    close_center direction orders close_L 1.23 > close_R 0.83 > close_center 0.71, i.e. the position
    it is named for is only third on its own axis. The pairwise figure is the sharper instrument for
    remapping; this one is context.
    """
    if not res or meth not in res.get("methods", {}):
        return None
    disp, R = dict(ALIGNS)[align], res["methods"][meth]
    cm = R.get("cross_matrix", {})
    classes = [c for c in (CLASSES_LICK if align == "lick" else CLASSES_FULL) if c in cm]
    if "prestroke_lick" not in classes:
        return None

    def mat(c):
        M = np.full((len(BY_SEVERITY), len(BY_SEVERITY)), np.nan)
        for i, P in enumerate(BY_SEVERITY):
            for j, D in enumerate(BY_SEVERITY):
                cell = (cm[c].get(P) or {}).get(D) or {}
                if cell.get("mean") is not None:
                    M[i, j] = cell["mean"]
        return M

    base = mat("prestroke_lick")
    others = [c for c in classes if c != "prestroke_lick"]
    mats = [("prestroke_lick", base, False)] + [(c, mat(c) - base, True) for c in others]
    # robust symmetric scale for the difference panels: the raw outliers otherwise own the colourbar
    d = np.concatenate([m[np.isfinite(m)].ravel() for _, m, isd in mats if isd] or [np.zeros(1)])
    lim = float(np.nanpercentile(np.abs(d), 95)) if d.size else 1.0
    lim = max(lim, 0.2)

    fig, axes = plt.subplots(1, len(mats), figsize=(4.3 * len(mats) + 1.2, 5.1), squeeze=False)
    for k, (c, M, isdiff) in enumerate(mats):
        ax = axes[0][k]
        im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r",
                       vmin=(-lim if isdiff else -0.3), vmax=(lim if isdiff else 1.3))
        for i in range(len(BY_SEVERITY)):
            for j in range(len(BY_SEVERITY)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}" if isdiff else f"{M[i, j]:.2f}",
                            ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(BY_SEVERITY)))
        ax.set_xticklabels(BY_SEVERITY, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(BY_SEVERITY)))
        ax.set_yticklabels(BY_SEVERITY if k == 0 else [], fontsize=7)
        ax.set_title(("BASELINE: " if not isdiff else "minus baseline: ") + STYLE[c][2],
                     fontsize=8.5)
        ax.set_xlabel("scored on THIS position's direction", fontsize=8)
        if k == 0:
            ax.set_ylabel("TRUE spout position", fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{res['animal']} \u2014 {disp}, {meth.upper()}. Panel 1 is the PRE-STROKE baseline (1 = "
        f"that column's own position); the rest are DIFFERENCES from it.\nRed = more like the "
        f"column's position than pre-stroke, blue = less. A row going red OFF the diagonal is a "
        f"remapping. CAVEAT: one-vs-rest axes for MIDDLE positions are largely close-vs-far \u2014 "
        f"see the pairwise figure.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    q = Path(out) / f"coding_cross_{disp}_{meth}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_pairwise(res, out, align="precue", meth="dom"):
    """A-vs-B axes: for trials truly at A, how far toward B does each class sit?

    Sharper than one-vs-rest for remapping: "not-A" averages five positions, so it can say a trial
    is unlike A without saying WHICH position it resembles. Here 1 = pre-stroke A, 0 = pre-stroke B.
    """
    if not res or meth not in res.get("methods", {}):
        return None
    disp, R = dict(ALIGNS)[align], res["methods"][meth]
    classes = [c for c in (CLASSES_LICK if align == "lick" else CLASSES_FULL)
               if c in R.get("pairwise", {})]
    if not classes:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.2), squeeze=False, sharey=True)
    for k, A in enumerate(BY_SEVERITY):
        ax = axes[k // 3][k % 3]
        others = [B for B in BY_SEVERITY if B != A]
        x = np.arange(len(others))
        for c in classes:
            col, mk, lab = STYLE[c]
            cells = [(R["pairwise"][c].get(f"{A}|{B}") or {}) for B in others]
            ys = [cc.get("mean") for cc in cells]
            es = [cc.get("sem") for cc in cells]
            ax.errorbar(x, [np.nan if y is None else y for y in ys],
                        yerr=[0 if e is None else e for e in es], fmt="-", marker=mk, color=col,
                        ecolor=col, elinewidth=1.0, capsize=2.5, ms=5.5, lw=1.3,
                        label=(lab if k == 0 else None))
        ax.axhline(1.0, color="tab:blue", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0.0, color="k", ls=":", lw=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(others, rotation=55, ha="right", fontsize=7.5)
        ax.set_title(f"trials truly at {A}", fontsize=9.5)
        ax.grid(alpha=0.25)
        if k % 3 == 0:
            ax.set_ylabel("1 = pre-stroke THIS position,  0 = pre-stroke the OTHER one")
    fig.legend(loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle(f"{res['animal']} \u2014 {disp}, {meth.upper()}. PAIRWISE axes \u2014 the sharper remapping instrument: each contrast is A vs B alone, "
                 f"panel's position, how far toward each OTHER position does the class sit?\n"
                 f"Dropping toward 0 against a particular partner is that trial set looking like "
                 f"THAT position. Bars = SEM.", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    q = Path(out) / f"coding_pairwise_{disp}_{meth}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--windows", nargs="+", default=["ENL", "cue", "lick"],
                    choices=("ENL", "cue", "lick"))
    # DEFAULT IS THE dom PAIR. The lr variants need a logistic fit per held-out session per pair --
    # ~330 extra fits per animal-window for the pairwise arm alone -- which is fine for a one-off
    # check and far too slow for a nightly step.
    ap.add_argument("--methods", nargs="+", default=["dom", "dom_orth"], choices=CD_METHODS)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    out.mkdir(parents=True, exist_ok=True)
    animals = config.normalize_animals(args.animals) or [a for a in config.animals()]
    want = set(args.windows)
    everything = {}
    for align, disp in ALIGNS:
        if disp not in want:
            continue
        print(f"=== {disp} window (align={align}) ===", flush=True)
        res = {}
        for an in animals:
            try:
                res[an] = run_animal(an, align=align, methods=tuple(args.methods))
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! {an} [{disp}]: {type(ex).__name__} {str(ex)[:90]}", flush=True)
                res[an] = None
        everything[disp] = res
        # ONE FIGURE PER ANIMAL: the post-stroke course is expected to move, and four animals on
        # one axes cannot show six positions x five classes x N sessions each.
        for an in sorted(res):
            if not res[an]:
                continue
            for meth in res[an].get("methods", {}):
                for fn in (figure_animal, figure_cross, figure_pairwise):
                    q = fn(res[an], out, align=align, meth=meth)
                    if q:
                        print(f"  wrote {q}", flush=True)
    (out / "coding_direction.json").write_text(
        json.dumps(everything, indent=1, default=float), encoding="utf-8")
    print("wrote coding_direction.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
