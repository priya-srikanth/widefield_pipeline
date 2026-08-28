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
  MISS WHILE WORKING  no spout contact at THIS position, while the animal is still contacting at
                      others. Position-specific, and graded by SEVERITY: far_R > far_center >
                      far_L > close_R > close_center > close_L -- contraversive within each ring,
                      far worse than close throughout.
  STOPPED             no spout contact anywhere, for a sustained non-recovering run. Position-
                      GENERAL: inside that window the response rate is ~0 at every position, close
                      included.

BOTH ARE DEFINED BY SPOUT CONTACT, NOT BY ATTEMPTING (Priya, 2026-08-23: "its not clear the animal
isnt trying in the 'stopped' trials - i still sometimes see reactive jaw movement"). A "lick" here is
a detected event on `lick_analog`, i.e. the tongue REACHING the spout, and `engagement_gate` sees
only that. A jaw movement, a tongue protrusion that falls short, or a mistimed reach all produce zero
lick events and are indistinguishable from lying still.

That matters most exactly where the deficit is. If the lesion impairs REACHING the contraversive
spout, "failed to contact" is the deficit's signature rather than evidence about intention -- so
STOPPED is NOT a "not trying" control, and a class labelled by non-response cannot support a claim
about whether a plan formed. The fix is movement onset from the Blackfly VIDEO rather than contact
from the spout sensor, which would split the miss class into attempted-but-no-contact (a motor
deficit) and no-attempt (a plan deficit); those have opposite predictions here and no DAQ-only
analysis can separate them. See DECISIONS.md.

The `poststroke_lick` class is unaffected -- those trials have a contact by definition -- which is a
further reason to prefer the same-class pre-vs-post comparison wherever both are available.

WHAT EACH WINDOW CAN ANSWER, AND WHAT IT CANNOT:
  ENL   all five classes. Nothing has happened yet, and the window is already lick-free by
        construction (``decode.precue_lickfree``), so it is the clean one.
  cue   all five classes. Note that a lick trial contains its lick from ~140 ms (median first-lick
        latency is 0.137-0.255 s pre-stroke, minimum 0.109 s), so there is NO movement-free cue
        window to retreat to. The per-position construction is what keeps this interpretable:
        movement is common to every training class, so it cannot define the direction.
  lick  all five classes, but the no-lick ones sit at an INFERRED time. A no-lick trial has no
        lick to align to, so its window starts at the cue plus this session's own median RT at that
        position -- "when the lick would have been" (Priya, 2026-08-21). That makes the arms
        comparable; it does not make the no-lick time measured. Read those two classes as an
        inference, and most cautiously post-stroke, where latency is long and variable.

        The default reference remains the CUE and must not be used here: it offsets the arms by the
        whole reaction time (a median of 2.439 s at post-stroke far_R against a 2 s window, so no
        overlap at all), and `_trial_features` returns those trials populated and plausible-looking
        either way -- the same trap as the post-lick confusion bug of 2026-08-20.

        THE ENGAGEMENT AXIS IS A LICKING AXIS HERE. Built in this window it separates trials WITH a
        movement from trials without one, so the orthogonalised variants ask "what position
        structure survives once movement presence is removed". That is the right question for the
        no-lick classes and a conservative one for the lick classes: licks to different spouts
        differ in kinematics, so position and position-specific MOVEMENT are not separable in this
        window by any projection.
"""
from __future__ import annotations

import argparse
import itertools
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
#: two-character position labels for dense axes (c/f = close/far, L/C/R). One
#: definition, imported rather than re-spelled -- three copies of a label map is
#: how two figures come to disagree about what 'fC' means.
from wfield_local.locanmf_frozen_decoder import POS_SHORT  # noqa: E402

#: classes each window can carry. Every window now carries all five: the lick window places a
#: no-lick trial at its would-be-lick time (see the module docstring), so CLASSES_LICK is kept
#: only for callers that deliberately want the two lick classes alone.
CLASSES_FULL = ("prestroke_lick", "prestroke_nolick", "poststroke_lick", "poststroke_miss_working", "poststroke_stopped")
CLASSES_LICK = ("prestroke_lick", "poststroke_lick")
#: Flag a value as thinly-supported. 10 to match this project's existing rule -- plot_poststroke
#: MIN_N is 10 and G2b red-hatches below it -- rather than inventing a second threshold for the
#: same idea. (It was 12, chosen for no reason; Priya asked why, and there was no answer.)
MIN_TRIALS = 10
FLOOR_TRIALS = 3     # below this there is nothing to average at all

#: DO-NOT-DRAW rule for the within-session figure, on PRECISION rather than count. The pole
#: separation is 1.0 by construction, and the measured within-class SD of the projection is ~1.08
#: (median over 168 well-populated cells), so n=12 buys a SEM of 0.31 -- a third of the entire
#: scale, enough to invent a shape from noise. A cell is drawn only if its OWN SEM is under this,
#: which adapts to the real spread instead of assuming one; at the median SD it corresponds to
#: n >= 20.
MAX_SEM_DRAWN = 0.25

#: Figure kinds this module writes, as they appear in the filenames the deck reads. Declared rather
#: than left implicit so `tests/test_analysis_deck.py` can assert the deck never names a kind nothing
#: produces -- the ORPHAN condition that froze four section-G slides for two days in August, invisible
#: to every other check because the file was present and simply never updated.
#:   PER-ANIMAL   coding_<kind>_<window>_<method>_<animal>.png   (engagement omits the method)
#:   COHORT       coding_<kind>_<window>_<method>.png
FIGURE_KINDS = ("direction", "pooled", "within", "cross", "pairwise", "normunit")
#: per-post-stroke-session pairwise, one figure PER CLASS (five classes on one figure
#: with N sessions each would be unreadable). Named coding_pairsess_<win>_<meth>_<cls>_<animal>.
PAIRSESS_CLASSES = ("poststroke_lick", "poststroke_miss_working", "poststroke_stopped")
#: Figures that carry NO method in their filename, because they do not come from a coding direction
#: at all -- `engagement` is behaviour and `rtsplit` reads the frozen decoder's stored confusions.
#: Getting this wrong is not cosmetic: the deck once built every name with the method interpolated,
#: so `coding_engagement_<win>_<animal>.png` never matched and TWELVE behaviour slides were silently
#: absent (2026-08-24). `tests/test_analysis_deck.py` checks the deck's literals against this tuple.
FIGURE_KINDS_NOMETHOD = ("engagement", "rtsplit")
COHORT_FIGURE_KINDS = ("cosslope", "pairsplit")


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


def engagement_axis(X_lick, X_nolick, pos_lick=None, pos_nolick=None):
    """Unit vector separating pre-stroke LICK from pre-stroke NO-LICK.

    Built PRE-STROKE, where "no lick" is unambiguous -- there is no motor deficit to confuse with
    intent. Whether the same axis describes post-stroke non-responding is an assumption, not a fact,
    and it is the main thing to distrust about the orthogonalised variants.

    THE UNSTATED ASSUMPTION IS THAT NO-LICK TRIALS ARE POSITION-NEUTRAL, and in one animal they are
    emphatically not (Priya, 2026-08-24; `scripts/nocontact_census.py`). Pre-stroke no-contact rate
    by position:

        PS93   far_L 26.8% (268 trials)  far_center 14.1%  far_R 4.3%  close 0.4-2.4%   -> 64x
        PS92   far_L  6.2%               ...                                            -> 4.4x
        PS94   far_R  6.2% (the other direction)                                        -> 2.9x
        PS95   far_L  7.7%               near-flat                                      -> 1.9x

    PS93 often licked LEFTWARD WITHOUT SPOUT CONTACT before any lesion -- its known orofacial
    deficit -- and licks are detected BY CONTACT, so those trials land in the no-lick arm. 54% of
    its no-lick side is far_L and 82% is far positions, while the lick side is balanced across all
    six. The resulting "engagement" axis therefore carries a large far_L-versus-rest component, and
    orthogonalising against it removes far_L POSITION structure from all fifteen pairs. That is a
    mechanical candidate for PS93's broad, unlateralised profile: not a state effect measured, but
    position information subtracted.

    ``pos_lick``/``pos_nolick`` (position labels per trial) switch to the POSITION-BALANCED axis:
    the mean over positions of each position's own lick-minus-no-lick difference, so a position
    contributing ten times the no-lick trials no longer contributes ten times the axis. Positions
    with no trials on one side are skipped. Off by default -- every result on disk predates this,
    and the two must be COMPARED rather than silently swapped.
    """
    XL, XN = np.asarray(X_lick), np.asarray(X_nolick)
    if pos_lick is not None and pos_nolick is not None:
        pl, pn = np.asarray(pos_lick), np.asarray(pos_nolick)
        diffs = [XL[pl == p].mean(0) - XN[pn == p].mean(0)
                 for p in np.unique(pl) if (pl == p).any() and (pn == p).any()]
        e = np.mean(diffs, axis=0) if diffs else XL.mean(0) - XN.mean(0)
    else:
        e = XL.mean(0) - XN.mean(0)
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



# --------------------------------------------------------------------------------------------
# DIAGNOSTICS. Each of these answers "why does that panel read the way it does", and each was a
# throwaway script before it was a figure -- which is exactly how a measured number becomes stale
# prose in a speaker note that nothing regenerates (Priya, 2026-08-22).
# --------------------------------------------------------------------------------------------

QUARTILES = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
QLABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]

#: rings of the 2x3 spout grid. A pair is WITHIN-ring when both spouts sit at the same distance and
#: differ only in laterality (close_L vs close_R); CROSS-ring when it spans the distance dimension
#: (far_R vs close_R). The distinction matters because the within-session state drift runs ALONG
#: close-vs-far, so a contrast that spans it inherits the drift and one that does not, does not.
FAR = ("far_R", "far_center", "far_L")


def _ring(p):
    return "far" if p in FAR else "close"


def _pos_color(name):
    """The cohort's canonical per-position colour (side = hue, ring = lightness).

    Borrowed from `spout_behavior` rather than reinvented: a position that is crimson in the
    behaviour deck and something else here is a reader's problem, not a palette preference. Lazy
    import so this module does not pull the behaviour stack unless a figure asks for a colour.
    """
    from wfield_local.spout_behavior import POSITIONS, pos_color
    idx = next((q["idx"] for q in POSITIONS if q["name"] == name), None)
    return pos_color(idx) if idx is not None else "tab:grey"


def _slope(vals):
    """Last usable quartile minus the first. None if under three quartiles have a value."""
    v = [x for x in vals if x is not None]
    return (v[-1] - v[0]) if len(v) >= 3 else None


def _by_quartile(frac, mask, fn):
    """fn applied to each within-session quartile of the trials in ``mask``."""
    return [fn(mask & (frac >= lo) & (frac < hi)) for lo, hi in QUARTILES]


def response_by_quartile(feat, kept, YE, GE, YU, GU, pre_i):
    """Response rate per position per within-session quartile, split pre/post stroke.

    THE BEHAVIOURAL COUNTERPART of the within-session neural figure, and the thing that decides how
    to read it. Built from the SAME trial indices the features come from (feat.indices), so "where
    in the session" cannot disagree between the two -- reconstructing a trial filter elsewhere is
    how bugs 15-17 happened.
    """
    tables = _session_tables(feat, kept)
    acc = {ph: {p: [[] for _ in QUARTILES] for p in BY_SEVERITY} for ph in ("pre", "post")}
    for si in range(len(kept)):
        t = tables.get(si)
        if t is None:
            continue
        ie, inl = feat.indices[kept[si]]
        pos = _positions_for(tables, si, YE, GE, YU, GU, ie, inl)
        span = max(int(t["order"].max()), 1)
        fr = np.array([min(int(k) / span, 1.0) for k in t["order"]], float)
        ph = "pre" if si in pre_i else "post"
        for p in BY_SEVERITY:
            mp = pos == p
            for qi, (lo, hi) in enumerate(QUARTILES):
                acc[ph][p][qi].extend(t["responded"][mp & (fr >= lo) & (fr < hi)].tolist())
    out = {}
    for ph, d in acc.items():
        out[ph] = {p: [{"rate": (float(np.mean(v)) if v else None), "n": len(v)} for v in rows]
                   for p, rows in d.items()}
    return out


def _diagnostics(XE, en, e_pre, frac_e, W, base_m, do_orth, e_axis):
    """Everything that explains a panel rather than being one.

    closefar      how much of each position's axis is the close-vs-far dimension
    slopes        that position's within-session drift on its own axis (pre-stroke LICK)
    pairwise      the same two numbers for every A-vs-B axis, tagged within-ring or cross-ring
    norm_unit     post-stroke LICK scored raw and on UNIT-NORMALISED trials, plus the norm ratio
    """
    d = {}
    far_m = e_pre & np.isin(en, FAR)
    close_m = e_pre & ~np.isin(en, FAR)
    if far_m.sum() < 20 or close_m.sum() < 20:
        return d
    cf = XE[close_m].mean(0) - XE[far_m].mean(0)
    cf = cf / max(float(np.linalg.norm(cf)), 1e-12)
    d["closefar_engagement_cos"] = (float(cf @ e_axis) if e_axis is not None else None)

    def unit(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n > 0, n, 1.0)

    d["positions"] = {}
    for P in BY_SEVERITY:
        if P not in W:
            continue
        w, p0, p1 = W[P]
        mp = e_pre & (en == P)
        pre_q = _by_quartile(frac_e, mp,
                             lambda m, _w=w, _a=p0, _b=p1: (
                                 float(np.mean(project(XE[m], _w, _a, _b)))
                                 if m.sum() >= MIN_TRIALS else None))
        post_m = (~e_pre) & (en == P)
        cell = {"cos_closefar": float(w @ cf),
                "prestroke_lick_by_quartile": pre_q,
                "prestroke_lick_slope": _slope(pre_q),
                "n_post": int(post_m.sum())}
        if post_m.sum() >= MIN_TRIALS:
            q0, q1 = poles(unit(XE[mp]), unit(XE[e_pre & (en != P)]), w)
            cell["post_raw"] = float(np.mean(project(XE[post_m], w, p0, p1)))
            cell["post_unit"] = float(np.mean(project(unit(XE[post_m]), w, q0, q1)))
            cell["norm_ratio"] = float(np.linalg.norm(XE[post_m], axis=1).mean()
                                       / np.linalg.norm(XE[mp], axis=1).mean())
        d["positions"][P] = cell

    d["pairwise"] = []
    for A, B in itertools.combinations(BY_SEVERITY, 2):
        mA, mB = e_pre & (en == A), e_pre & (en == B)
        if mA.sum() < 20 or mB.sum() < 20:
            continue
        wab = direction(XE[mA], XE[mB], base_m)
        if do_orth and e_axis is not None:
            wab = orthogonalise(wab, e_axis)
        r0, r1 = poles(XE[mA], XE[mB], wab)      # 0 = B, 1 = A
        q = _by_quartile(frac_e, mA,
                         lambda m, _w=wab, _a=r0, _b=r1: (
                             float(np.mean(project(XE[m], _w, _a, _b)))
                             if m.sum() >= MIN_TRIALS else None))
        d["pairwise"].append({"A": A, "B": B, "same_ring": _ring(A) == _ring(B),
                              "cos_closefar": float(wab @ cf),
                              "by_quartile": q, "slope": _slope(q)})
    return d


#: Post-stroke classes written to `confusions`, finest-grain first. These PARTITION the post-stroke
#: trials: they are mutually exclusive and exhaustive, so any population a caller wants is a SUM over
#: a subset -- "all working" is lick+miss_working, "all trials" is all three -- which is exactly what
#: a pre-summed matrix cannot provide.
CONFUSION_CLASSES = ("poststroke_lick", "poststroke_miss_working", "poststroke_stopped")

#: REFINEMENTS of a class in `CONFUSION_CLASSES`, deliberately kept OUT of that tuple.
#:
#: `poststroke_lick_early` / `_late` partition `poststroke_lick` at `RT_SPLIT_S`; they are not a
#: fourth and fifth sibling. Putting them in `CONFUSION_CLASSES` would have broken the invariant that
#: file's whole design rests on -- a caller summing the tuple to get "all trials" would have counted
#: every lick trial twice, silently, and the result would still have looked like a confusion matrix.
#: Written as a mapping so a consumer can discover the relationship rather than hardcode it.
CONFUSION_SUBCLASSES = {"poststroke_lick": ("poststroke_lick_early", "poststroke_lick_late")}

#: Boundary between an early and a late rewarded lick, in seconds from the cue.
#:
#: 2.0 s, which is `nolick_decoder._args`'s engaged cut and the value `decode.max_rt_s` held until
#: 2026-08-21, so "late" means the same thing in both places. NOT the session's median RT: that is
#: session-relative, so a "late" trial on one day would be an "early" trial on the next, and it could
#: not be compared across days nor against the `late_rewarded` category the no-lick reference already
#: defines. A fixed boundary is comparable; an adaptive one is a different measurement per session.
#:
#: The split exists because `decode.max_rt_s` is now 3.5 s, so a 0.2 s lick and a 3.0 s lick sit in
#: one "engaged" class, and telling those apart is the study's question: position coding preserved on
#: LATE trials is plan intact / execution slow; degraded on late trials is a different result.
#:
#: MEASURED 2026-08-28, AND THE ANSWER SURPRISED THE DESIGN. This was built expecting the post-stroke
#: mass to move late. It does not: the late arm is 1029 of 30645 post-stroke rewarded trials (3.4%),
#: and only 26 for PS94 and 27 for PS95 -- the two most impaired animals. They do not lick SLOWLY,
#: they lick fast or not at all, which is the same conclusion the no-detected-lick arm reaches from
#: the other side. The split is still worth computing (PS92 and PS93 carry enough to read, and an
#: empty arm is an answer), but any figure drawn from it MUST mark its n -- see `figure_rt_split`,
#: where a panel below `len(positions) * MIN_TRIALS` is titled TOO FEW TO READ in red.
RT_SPLIT_S = 2.0


def _rt_engaged(feat, kept, XE):
    """First-lick latency in seconds, one row per ENGAGED feature row, or None.

    Concatenated in `kept` order, which is the order `pool_sessions` stacks XE in -- and then LENGTH
    CHECKED against XE. The check is the point: an RT vector one trial out of step would not fail,
    it would mislabel the boundary trial of every session and still produce a plausible figure.
    Returning None on any mismatch means the caller writes no split at all, which is recoverable;
    writing a misaligned one is not.
    """
    rts = getattr(feat, "rts", None)
    if not rts:
        return None
    out = []
    for lab in kept:
        v = rts.get(lab)
        if v is None:
            return None
        out.append(np.asarray(v, float))
    rt = np.concatenate(out) if out else np.array([], float)
    if rt.size != len(XE) or not np.all(np.isfinite(rt)):
        return None
    return rt


def _class_confusions(XE, YE, GE, XU, YU, e_pre, pre_i, _mask, kept, rt_e=None):
    """Confusion of the frozen PRE-stroke decoder, one counts matrix per class.

    Rows are the TRUE position and columns the predicted one, both in DISPLAY_ORDER, as raw counts
    rather than row-normalised -- a caller summing two classes must be able to add them, and
    normalised rows cannot be added.

    THE PRE PANEL IS LEAVE-ONE-SESSION-OUT and the post panels are not, which is not an
    inconsistency: post-stroke trials are held out by construction (the decoder never saw a
    post-stroke session), while pre-stroke trials are the training set. Scoring them in-sample gives
    0.89-0.99 against 0.45-0.66 held out, so a reader comparing a post value against an in-sample pre
    value would read a collapse that is mostly overfitting (recorded in
    `grant_figures.fig_confusion_pre_post_working`).
    """
    # This module imports only POSITION_NAMES, not the order -- take it from the one place that
    # defines it, so these matrices are in the SAME row order as every other confusion in the deck.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

    def pipe():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))

    def counts(y_true, y_pred):
        M = np.zeros((len(DISPLAY_ORDER), len(DISPLAY_ORDER)), float)
        idx = {c: i for i, c in enumerate(DISPLAY_ORDER)}
        for t, q in zip(y_true, y_pred):
            if int(t) in idx and int(q) in idx:
                M[idx[int(t)], idx[int(q)]] += 1
        return M

    if e_pre.sum() < 20 or len(np.unique(YE[e_pre])) < 2:
        return None
    res = {"labels": [POSITION_NAMES.get(int(c), str(c)) for c in DISPLAY_ORDER],
           "counts": True, "note": "raw counts; sum classes to build any population"}

    # PRE, leave-one-session-out
    Cpre = np.zeros((len(DISPLAY_ORDER), len(DISPLAY_ORDER)), float)
    got = False
    for i in sorted(pre_i):
        tr, te = e_pre & (GE != i), e_pre & (GE == i)
        if te.sum() < 5 or len(np.unique(YE[tr])) < 2:
            continue
        Cpre += counts(YE[te], pipe().fit(XE[tr], YE[tr]).predict(XE[te]))
        got = True
    res["prestroke_lick"] = Cpre.tolist() if got else None

    # POST, one matrix per class, all scored by the SAME frozen model fitted on every pre-stroke
    # engaged trial -- so the classes are directly comparable and their sums are meaningful.
    clf = pipe().fit(XE[e_pre], YE[e_pre])
    for cls in CONFUSION_CLASSES:
        m = _mask(cls)
        X, y = ((XE, YE) if cls == "poststroke_lick" else (XU, YU))
        res[cls] = counts(y[m], clf.predict(X[m])).tolist() if m.any() else None
        res[f"n_{cls}"] = int(m.sum())

    # EARLY vs LATE REWARDED, a partition of `poststroke_lick` at RT_SPLIT_S.
    #
    # Built from the SAME mask and scored by the SAME `clf`, so `early + late == poststroke_lick`
    # element-wise by construction rather than by two recipes agreeing -- and the assertion below
    # checks the construction held rather than trusting it. Two places computing "the same"
    # population and drifting apart is how the frozen-decoder contamination happened.
    if rt_e is not None:
        m_lick = _mask("poststroke_lick")
        early = m_lick & (rt_e < RT_SPLIT_S)
        late = m_lick & (rt_e >= RT_SPLIT_S)
        if int(early.sum()) + int(late.sum()) != int(m_lick.sum()):
            # Only reachable via a non-finite RT, which `_rt_engaged` already refuses -- so if it
            # ever fires, the partition is broken somewhere new and the safe answer is no split.
            print(f"  [coding_dirs] early+late != lick ({int(early.sum())}+{int(late.sum())} vs "
                  f"{int(m_lick.sum())}) -- early/late split omitted", flush=True)
        else:
            res["rt_split_s"] = RT_SPLIT_S
            for cls, m in (("poststroke_lick_early", early), ("poststroke_lick_late", late)):
                res[cls] = counts(YE[m], clf.predict(XE[m])).tolist() if m.any() else None
                res[f"n_{cls}"] = int(m.sum())
    return res


def _gate_all(feat, kept, XE, YE, GE, XU, YU, GU):
    """(not_eng, frac_nolick, frac_eng), or None if the trial bookkeeping does not line up.

    ``frac_*`` is each trial's position WITHIN its session, 0 at the first trial and 1 at the last.
    It comes from the same indices the gate uses, so "where in the session" and "before or after the
    quit" can never disagree about a trial.
    """
    tables = _session_tables(feat, kept)
    if not tables:
        return None
    not_eng, frac_nl, frac_e = [], [], []
    for si in range(len(kept)):
        t = tables.get(si)
        if t is None:
            continue
        ie, inl = feat.indices[kept[si]]
        pos = _positions_for(tables, si, YE, GE, YU, GU, ie, inl)
        ne = engagement_gate(t["order"], t["responded"], pos)
        bne = {int(k): bool(v) for k, v in zip(t["order"], ne)}
        span = max(int(t["order"].max()), 1)
        not_eng += [bne.get(int(k), False) for k in inl]
        frac_nl += [min(int(k) / span, 1.0) for k in inl]
        frac_e += [min(int(k) / span, 1.0) for k in ie]
    not_eng = np.array(not_eng, bool)
    if len(not_eng) != len(XU) or len(frac_e) != len(XE):
        return None
    return not_eng, np.array(frac_nl, float), np.array(frac_e, float)


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

    # LICK WINDOW: no-lick trials are referenced to WHEN THE LICK WOULD HAVE BEEN (this
    # session's own median RT at that position), not to the cue. Cue-referencing offsets the
    # two arms by the whole reaction time -- a median of 2.439 s at post-stroke far_R, where a
    # 2 s window means they do not overlap at all.
    feat = features_with_indices(basis,
                                 nolick_ref=("would_be_lick" if align == "lick" else "cue"))
    pooled = pool_sessions(pre + post, source="locanmf", align=align, post_s=post_s, features=feat)
    if pooled is None:
        return None
    XE, YE, GE, _BE, XU, YU, kept, _c, GU = pooled
    YU = YU.astype(int)
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    e_pre, u_pre = np.isin(GE, list(pre_i)), np.isin(GU, list(pre_i))
    en = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YE])
    un = np.array([POSITION_NAMES.get(int(v), str(v)) for v in YU])

    # every window can now carry the no-lick classes; at "lick" their window is INFERRED
    use_nolick = True
    _g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
    if _g is None:
        print(f"[coding_dirs] {animal} {disp}: trial bookkeeping mismatch -- skipped", flush=True)
        return None
    not_eng, frac_nl, frac_e = _g

    order = sorted(range(len(kept)), key=lambda i: kept[i].split("_")[1])
    # VARIANCE CAPTURED PER SESSION, stored so a declining per-session coding value can be tested
    # against the obvious artefact: the joint basis has FIXED footprints, and a session not in the
    # fitting set is PROJECTED onto them. If the basis describes later days progressively worse,
    # every projection shrinks and manufactures a decline with no neural change (Priya, 2026-08-23).
    # `in_basis` matters: those sessions are recorded as 1.0 by CONSTRUCTION, not measured, so only
    # the projected ones can be compared -- which is fine, since post-stroke sessions are projected.
    _vc = getattr(feat, "variance_captured", {}) or {}
    sessions = [{"label": kept[i], "date": kept[i].split("_")[1],
                 "phase": ("pre" if i in pre_i else "post"),
                 "in_basis": bool(kept[i] in basis.labels),
                 "variance_captured": (None if kept[i] not in _vc else float(_vc[kept[i]]))}
                for i in order]

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

    def _mask(cls, pos=None):
        """Row mask for one class (same definitions as `trials`), optionally at one position."""
        if cls == "prestroke_lick":
            m, names = e_pre.copy(), en
        elif cls == "poststroke_lick":
            m, names = ~e_pre, en
        elif cls == "prestroke_nolick":
            m, names = u_pre.copy(), un
        elif cls == "poststroke_miss_working":
            m, names = ~u_pre & ~not_eng, un
        else:
            m, names = ~u_pre & not_eng, un
        return m & (names == pos) if pos is not None else m

    classes = CLASSES_FULL
    out = {"animal": animal, "align": align, "window": disp, "basis_id": basis.basis_id,
           "ncomp": int(basis.ncomp), "n_features": int(XE.shape[1]), "sessions": sessions,
           "methods": {}}
    # BEHAVIOUR, from the same trial indices the features came from. Method-independent, so it is
    # computed once here rather than inside the per-method loop.
    try:
        out["response_by_quartile"] = response_by_quartile(feat, kept, YE, GE, YU, GU, pre_i)
    except Exception as ex:                                          # noqa: BLE001
        print(f"  [coding_dirs] {animal}: response-rate diagnostic unavailable ({ex})", flush=True)

    # PER-CLASS CONFUSIONS OF THE FROZEN PRE-STROKE DECODER (Priya, 2026-08-26).
    #
    # WHY HERE. `section_g.json` already stores a confusion, but summed over trials -- and
    # `grant_figures.fig_confusion_pre_post_working` records why that is unusable: "a summed matrix
    # cannot be un-summed". So 5b recomputed the whole pooling from LocaNMF, which costs >10 min of
    # network reads to redraw a figure, and every layout iteration on it paid that again.
    #
    # The fix is granularity, not a second store: writing ONE MATRIX PER CLASS means any population
    # a caller wants is a SUM of these, so nothing has to be un-summed. 5b's "post_working" is
    # lick + miss_working; its "post_all" is those plus stopped.
    #
    # And it belongs in THIS file because this is where the classes are defined -- `_gate_all` and
    # `_mask` live here, and `miss_vs_stopped` already reads these class names out of
    # coding_direction.json. A second definition elsewhere is how the frozen-decoder contamination
    # happened: two places computing "the same" population and drifting apart.
    try:
        out["confusions"] = _class_confusions(XE, YE, GE, XU, YU, e_pre, pre_i, _mask, kept,
                                              rt_e=_rt_engaged(feat, kept, XE))
    except Exception as ex:                                          # noqa: BLE001
        print(f"  [coding_dirs] {animal}: per-class confusions unavailable ({ex})", flush=True)

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

        # ---- 2a2. THE SAME MATRIX, PER POST-STROKE SESSION ----------------------------------
        # Pooled, a recovery and a collapse average to "no change". Same directions, same poles --
        # only the trials projected change (Priya, 2026-08-23).
        res["cross_by_session"] = {}
        for c in ("poststroke_lick", "poststroke_miss_working", "poststroke_stopped"):
            if c not in classes:
                continue
            per = {}
            for i in order:
                if i in pre_i:
                    continue
                mat_i = {}
                for P in BY_SEVERITY:
                    row = {}
                    for D in BY_SEVERITY:
                        if D not in W:
                            continue
                        row[D] = _cells(proj(c, trials(c, P, sess=i), D))
                    mat_i[P] = row
                per[kept[i]] = mat_i
            res["cross_by_session"][c] = per

        # ---- 2b. POOLED across every session of the phase, and BINNED WITHIN a session --------
        # Pooled because the per-session view showed the classes are noisy at that grain -- swings
        # as large as any trend. Within-session because engagement is GRADED: a miss-while-working
        # trial just before the animal quits is not the same state as one at trial 50, and the
        # session-level split cannot see that (Priya, 2026-08-21).
        QS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
        for c in classes:
            fr = frac_e if c in ("prestroke_lick", "poststroke_lick") else frac_nl
            pooled, within = {}, {}
            for P in BY_SEVERITY:
                if P not in W:
                    continue
                w, p0, p1 = W[P]
                m = _mask(c, P)
                pooled[P] = _cells(proj(c, (XE if c in ("prestroke_lick", "poststroke_lick")
                                            else XU)[m], P))
                rows = []
                for lo, hi in QS:
                    mq = m & (fr >= lo) & (fr < hi)
                    cell = _cells(proj(c, (XE if c in ("prestroke_lick", "poststroke_lick")
                                           else XU)[mq], P))
                    cell["quartile"] = f"{int(lo * 100)}-{int(hi * 100 if hi <= 1 else 100)}%"
                    rows.append(cell)
                within[P] = rows
            res.setdefault("pooled", {})[c] = pooled
            res.setdefault("within_session", {})[c] = within

        # ---- 3. PAIRWISE axes: A vs B directly, no five-way mixture in the contrast -----------
        # Sharper for the remapping question: "not-P" above averages five positions, so a trial can
        # look unlike P without the axis saying WHICH other position it resembles.
        # THE DENOMINATOR OF EVERY PAIRWISE VALUE, recorded so the figure can show it. The
        # projection divides by the pre-stroke separation of A and B, so a pair that was barely
        # distinguishable to begin with AMPLIFIES the same absolute displacement -- which is exactly
        # where the largest |values| turn up, and was invisible on the figure (Priya, 2026-08-23).
        res["pairwise_axes"] = {}
        PW = {}          # key -> (w, pole_B, pole_A), fitted once and reused
        for A in BY_SEVERITY:
            for B in BY_SEVERITY:
                if A == B:
                    continue
                XA, XB = XE[e_pre & (en == A)], XE[e_pre & (en == B)]
                if len(XA) < 20 or len(XB) < 20:
                    continue
                wab = direction(XA, XB, base_m)
                if do_orth and e_axis is not None:
                    wab = orthogonalise(wab, e_axis)
                a0, a1 = poles(XA, XB, wab)
                res["pairwise_axes"][f"{A}|{B}"] = {
                    "spread": float(a1 - a0), "n_A": len(XA), "n_B": len(XB)}
                PW[f"{A}|{B}"] = (wab, a0, a1)

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
                    got = PW.get(f"{A}|{B}")       # fitted once above; 0 = pre-stroke B, 1 = A
                    if got is None:
                        continue
                    wab, q0, q1 = got
                    pw[f"{A}|{B}"] = _cells(project(trials(c, A), wab, q0, q1))
            res["pairwise"][c] = pw

        # ---- 3b. PAIRWISE, PER POST-STROKE SESSION ------------------------------------------
        # The pooled pairwise figure averages every post-stroke day into one number, and this repo
        # already knows those classes MOVE -- PS94's miss-while-working swings +1.05 to -0.68
        # between adjacent sessions. Pooling was chosen deliberately to tame that noise, and its
        # honest cost is that a recovery and a collapse average to "no change" (Priya, 2026-08-23).
        # Same axes, no refitting: only the trials being projected change.
        res["pairwise_by_session"] = {}
        for c in ("poststroke_lick", "poststroke_miss_working", "poststroke_stopped"):
            if c not in classes:
                continue
            by = {}
            for key, (wab, q0, q1) in PW.items():
                A = key.split("|")[0]
                rows = {}
                for i in order:
                    if i in pre_i:
                        continue
                    rows[kept[i]] = _cells(project(trials(c, A, sess=i), wab, q0, q1))
                by[key] = rows
            res["pairwise_by_session"][c] = by

        # ---- 4. DIAGNOSTICS: why a panel reads the way it does -------------------------------
        # Folded in HERE rather than given their own module because every one of them needs the
        # feature matrices that are already in memory. Run separately they cost a second ~40 min
        # of loading to recompute what this loop already has.
        res["diagnostics"] = _diagnostics(XE, en, e_pre, frac_e, W, base_m, do_orth, e_axis)
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
    classes = CLASSES_FULL
    sess = res["sessions"]
    x = np.arange(len(sess))
    bx = next((i - 0.5 for i, sn in enumerate(sess) if sn["phase"] == "post"), None)

    # 17.5in placed at 12.7in is a 73% reduction, so 6.5pt date labels arrived at 4.7pt.
    # 12.2in renders at ~100% and the labels below are enlarged to match.
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.4), squeeze=False, sharey=True,
                             sharex=True, gridspec_kw={"hspace": 0.34, "wspace": 0.10})
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
                    ax.text(xi, 0.02, f"n={n}", ha="center", va="bottom", fontsize=6.5,
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
        ax.set_xticklabels([sn["date"] for sn in sess], rotation=60, ha="right", fontsize=8.5)
        ax.grid(alpha=0.25)
        if k % 3 == 0:
            ax.set_ylabel("projection", fontsize=9)
    # ROBUST Y-LIMITS. A class with a handful of trials in one session can sit far outside [0, 1]
    # with a SEM to match -- pre-stroke no-lick reached ~5 on two PS94 sessions -- and a shared axis
    # then squashes every real line into a thin band. Clip to the bulk; outliers run off the top.
    _vals = []
    for _P in BY_SEVERITY:
        for _cell in ((R["positions"].get(_P) or {}).get("by_session") or {}).values():
            for _c in _cell.values():
                if isinstance(_c, dict) and _c.get("mean") is not None:
                    _vals.append(_c["mean"])
    if _vals:
        _lo, _hi = np.percentile(_vals, [2, 98])
        _pad = 0.15 * max(_hi - _lo, 0.5)
        axes[0][0].set_ylim(min(_lo - _pad, -0.35), max(_hi + _pad, 1.35))
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

    # SIZED FOR THE SLIDE IT IS PLACED ON (Priya, 2026-08-28). Six matrices in one row at
    # 4.3in each made a 22.7in figure placed at 12.7in -- a 56% reduction, so its 7pt tick
    # labels reached the reader at 3.9pt. Two rows of three at 3.2in is 10.8in wide and
    # renders at ~100%, so 8.5pt labels arrive as 8.5pt. Fewer inches, larger type: the
    # figure is smaller and MORE legible, because what a reader sees is
    # fontsize x (placed width / figure width).
    _nc = min(len(mats), 3)
    _nr = (len(mats) + _nc - 1) // _nc
    fig, axes = plt.subplots(_nr, _nc, figsize=(3.2 * _nc + 1.1, 3.5 * _nr + 1.1),
                             squeeze=False, gridspec_kw={"hspace": 0.45, "wspace": 0.16})
    for k, (c, M, isdiff) in enumerate(mats):
        # TWO-DIMENSIONAL, because the grid is. This said `axes[0][k]` -- correct while the panels
        # were one row of six, and an IndexError from the moment they became two rows of three on
        # 2026-08-28 (`index 3 is out of bounds for axis 0 with size 3`). It only ever fired on ENL
        # and cue: the LICK window carries two classes, so `_nc` is 2, `_nr` is 1, and row 0 is the
        # whole grid. `_draw` caught it, so the run continued and the deck simply had no
        # `coding_cross_*` for those windows -- an absence, not an error, which is why measuring the
        # placed figures is the only check that finds this class of fault.
        ax = axes[k // _nc][k % _nc]
        im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r",
                       vmin=(-lim if isdiff else -0.3), vmax=(lim if isdiff else 1.3))
        for i in range(len(BY_SEVERITY)):
            for j in range(len(BY_SEVERITY)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}" if isdiff else f"{M[i, j]:.2f}",
                            ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(BY_SEVERITY)))
        # VERTICAL, CENTRED. At 55 degrees with ha="right" a long label extends LEFT of its tick
        # and, with five panels across, is clipped by the neighbouring axes -- "far_center" and
        # "close_center" both rendered as "center" and "close_R" as "-lose_R", so two columns
        # appeared to have the same label (Priya, 2026-08-23). Vertical labels extend only
        # downward, which nothing else occupies.
        ax.set_xticklabels([POS_SHORT.get(p_, p_) for p_ in BY_SEVERITY], rotation=90,
                           ha="center", fontsize=9)
        ax.set_yticks(range(len(BY_SEVERITY)))
        ax.set_yticklabels([POS_SHORT.get(p_, p_) for p_ in BY_SEVERITY]
                           if k % _nc == 0 else [], fontsize=9)
        ax.set_title(("BASELINE: " if not isdiff else "minus baseline: ") + STYLE[c][2],
                     fontsize=8.5)
        ax.set_xlabel("scored on THIS position's direction", fontsize=8)
        # ONCE PER ROW, not once per figure -- with two rows, a single y-label on panel 0 leaves the
        # second row's axis unnamed. Matches the y-TICK rule three lines above, which already keyed
        # on the column.
        if k % _nc == 0:
            ax.set_ylabel("TRUE spout position", fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    # A 5-panel set in a 2x3 grid leaves one cell over; an unused axes draws as an empty framed box
    # that reads as a panel with no data rather than as no panel.
    for k in range(len(mats), _nr * _nc):
        axes[k // _nc][k % _nc].set_axis_off()
    fig.suptitle(
        f"{res['animal']} \u2014 {disp}, {meth.upper()}. Panel 1 is the PRE-STROKE baseline (1 = "
        f"that column's own position); the rest are DIFFERENCES from it.\nRed = more like the "
        f"column's position than pre-stroke, blue = less. A row going red OFF the diagonal is a "
        f"remapping. CAVEAT: one-vs-rest axes for MIDDLE positions are largely close-vs-far \u2014 "
        f"see the pairwise figure.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.85))
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
    classes = [c for c in CLASSES_FULL
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
        # WHERE THE PARTNER POSITION ACTUALLY SITS (Priya, 2026-08-23: "why aren't we showing the
        # compared position for reference?"). The 0 line is pre-stroke B BY DEFINITION -- it is the
        # pole, not a measurement -- so it says nothing about where B ended up. Post-stroke B is the
        # reference that matters: if A has moved to 0.4 and B is still at 0.05 the two are still
        # apart, and if B has come up to 0.35 they have converged. The value is already in the
        # B|A cell, where the same axis is anchored the other way round; by linearity
        # proj_B|A = 1 - proj_A|B exactly, so it is read back with 1 - x rather than recomputed.
        partner = []
        for B in others:
            cc = (R["pairwise"].get("poststroke_lick", {}).get(f"{B}|{A}") or {})
            partner.append(None if cc.get("mean") is None else 1.0 - cc["mean"])
        if any(v is not None for v in partner):
            ax.plot(x, [np.nan if v is None else v for v in partner], ls="--", lw=1.2,
                    color="tab:green", marker="x", ms=6, alpha=0.75,
                    label=("post-stroke LICK at the PARTNER position" if k == 0 else None))
        ax.axhline(1.0, color="tab:blue", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0.0, color="k", ls=":", lw=1.0, alpha=0.7)
        ax.set_xticks(x)
        # THE SEPARATION IS THE DENOMINATOR, so a small one amplifies. Shown under each partner
        # rather than left to be inferred: the largest |values| on this figure sit against the
        # partners a position was least distinguishable from pre-stroke.
        axes_meta = R.get("pairwise_axes") or {}
        ticks = []
        for B in others:
            sp = (axes_meta.get(f"{A}|{B}") or {}).get("spread")
            ticks.append(B if sp is None else f"{B}\n(sep {sp:.2f})")
        ax.set_xticklabels(ticks, rotation=55, ha="right", fontsize=7)
        ax.set_title(f"trials truly at {A}", fontsize=9.5)
        ax.grid(alpha=0.25)
        if k % 3 == 0:
            # SHORT. The full sentence collided with itself between the two rows; the meaning of the
            # 0 and 1 anchors belongs in the suptitle, where it is written once.
            ax.set_ylabel("projection", fontsize=9)
    fig.legend(loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle(
        f"{res['animal']} \u2014 {disp}, {meth.upper()}. PAIRWISE axes: each contrast is A vs B "
        f"ALONE. For trials truly at the panel's position, how far toward each OTHER position does "
        f"each class sit?\n"
        f"SCALE: 1 = pre-stroke lick at THIS position, 0 = pre-stroke lick at the OTHER one. Both "
        f"are DEFINITIONS -- the axis is anchored on them -- so the flat blue line at 1 is the "
        f"anchor, not a result; only its scatter is data.\n"
        f"Dropping toward 0 against a particular partner is that trial set looking like THAT "
        f"position. The dashed green x marks where post-stroke lick at the PARTNER position "
        f"actually sits, so convergence can be read directly rather than inferred. Bars = SEM.\n"
        f"(sep N) under each partner is the PRE-STROKE separation of that pair -- the denominator. "
        f"A small separation amplifies the same absolute displacement, which is where the "
        f"largest values on this figure come from.",
        fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    q = Path(out) / f"coding_pairwise_{disp}_{meth}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q



def figure_pairwise_sessions(res, out, align="precue", meth="dom", cls="poststroke_miss_working"):
    """ONE post-stroke class, one panel per position, a dot per POST-STROKE SESSION.

    WHY THIS IS SEPARATE FROM `figure_pairwise` RATHER THAN ANOTHER LINE ON IT. That figure already
    carries five classes; adding N sessions to each would be 5 x N lines per panel and unreadable.
    Splitting by class keeps a panel to one class over time, which is the question the figure is for
    (Priya, 2026-08-23: "show dots for each post-stroke session").

    SEQUENTIAL COLOUR = DATE, so time reads left-to-right in the legend and dark-to-light in the
    panel. The pooled value is drawn as a grey line behind the dots: if the dots straddle it widely,
    the pooled pairwise figure is averaging a moving target and should not be read as a state.
    """
    if not res or meth not in res.get("methods", {}):
        return None
    disp, R = dict(ALIGNS)[align], res["methods"][meth]
    by = (R.get("pairwise_by_session") or {}).get(cls) or {}
    if not by:
        return None
    labs = sorted({lab for rows in by.values() for lab in rows})
    if not labs:
        return None
    cmap = plt.get_cmap("viridis")
    col = {lab: cmap(0.12 + 0.76 * (i / max(len(labs) - 1, 1))) for i, lab in enumerate(labs)}

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.4), squeeze=False, sharey=True)
    drew = False
    for k, A in enumerate(BY_SEVERITY):
        ax = axes[k // 3][k % 3]
        others = [B for B in BY_SEVERITY if B != A]
        x = np.arange(len(others))
        # POOLED, behind: the number the other pairwise figure shows
        pooled = [((R.get("pairwise", {}).get(cls, {}).get(f"{A}|{B}") or {}).get("mean"))
                  for B in others]
        if any(v is not None for v in pooled):
            ax.plot(x, [np.nan if v is None else v for v in pooled], color="0.55", lw=3.0,
                    alpha=0.55, zorder=1, label=("POOLED over all sessions" if k == 0 else None))
        for lab in labs:
            ys, es = [], []
            for B in others:
                cc = (by.get(f"{A}|{B}") or {}).get(lab) or {}
                ys.append(cc.get("mean"))
                es.append(cc.get("sem"))
            if all(v is None for v in ys):
                continue
            drew = True
            ax.errorbar(x, [np.nan if v is None else v for v in ys],
                        yerr=[0 if e is None else e for e in es], fmt="-o", ms=5, lw=1.2,
                        color=col[lab], ecolor=col[lab], elinewidth=0.9, capsize=2, zorder=3,
                        label=(lab.split("_")[-1] if k == 0 else None))
        ax.axhline(1.0, color="tab:blue", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(0.0, color="k", ls=":", lw=1.0, alpha=0.7)
        ax.set_xticks(x)
        meta = R.get("pairwise_axes") or {}
        ax.set_xticklabels(
            [f"{B}\n(sep {(meta.get(f'{A}|{B}') or {}).get('spread', float('nan')):.2f})"
             for B in others], rotation=55, ha="right", fontsize=7)
        ax.set_title(f"trials truly at {A}", fontsize=9.5)
        ax.grid(alpha=0.25)
        if k % 3 == 0:
            ax.set_ylabel("projection", fontsize=9)
    if not drew:
        plt.close(fig)
        return None
    fig.legend(loc="lower center", ncol=min(len(labs) + 1, 8), fontsize=8, frameon=False)
    fig.suptitle(
        f"{res['animal']} \u2014 {disp}, {meth.upper()}: {STYLE[cls][2]}, ONE DOT PER POST-STROKE "
        f"SESSION.\n1 = pre-stroke lick at THIS position, 0 = pre-stroke lick at the OTHER one; "
        f"both are DEFINITIONS, so the anchors are not results. Dropping toward 0 against a "
        f"particular partner is that trial set looking like THAT position \u2014 but a drop against "
        f"EVERY partner is loss of this position's pattern, not resemblance to all of them.\n"
        f"Grey = the pooled value the other pairwise figure shows. Dots straddling it widely mean "
        f"that figure is averaging a moving target. (sep N) is the pre-stroke separation of the "
        f"pair, i.e. the denominator: a small one amplifies.", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.86))
    q = Path(out) / f"coding_pairsess_{disp}_{meth}_{cls}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_cross_sessions(res, out, align="precue", meth="dom", cls="poststroke_miss_working"):
    """ONE post-stroke class, one 6x6 matrix PER SESSION, each a difference from the baseline.

    The pooled cross matrix is one number per cell over every post-stroke day. This is the same
    quantity resolved in time, which is the only way to tell a settled state from a moving one.
    Chunked so a panel stays the same size however long the cohort runs.
    """
    if not res or meth not in res.get("methods", {}):
        return None
    disp, R = dict(ALIGNS)[align], res["methods"][meth]
    per = (R.get("cross_by_session") or {}).get(cls) or {}
    base = (R.get("cross_matrix") or {}).get("prestroke_lick") or {}
    if not per or not base:
        return None
    labs = sorted(per)
    B = np.array([[(base.get(P, {}).get(D) or {}).get("mean", np.nan) for D in BY_SEVERITY]
                  for P in BY_SEVERITY], float)

    mats = []
    for lab in labs:
        M = np.array([[(per[lab].get(P, {}).get(D) or {}).get("mean", np.nan)
                       for D in BY_SEVERITY] for P in BY_SEVERITY], float)
        if np.isfinite(M).any():
            mats.append((lab, M - B))
    if not mats:
        return None
    lim = float(np.nanpercentile(np.abs(np.array([m for _l, m in mats])), 98)) or 1.0

    # MORE SESSIONS PER FIGURE, SMALLER PANELS, LARGER TEXT (Priya, 2026-08-28). The across-session
    # view is PREFERRED over the pooled one -- pooling post-stroke merges early-deficit days with
    # late-recovered days, which is the very thing this figure exists to separate -- so the answer to
    # its slide cost is to make the figure denser and more legible, not to drop it.
    #
    # WHY THIS IS A NET GAIN IN LEGIBILITY DESPITE SMALLER PANELS. The figure is placed at a FIXED
    # 12.7in on the slide, so what a reader sees is fontsize x (12.7 / figure width). At 4 columns of
    # 4.0in the figure was 16in wide and its 6.5pt ticks rendered at ~5.1pt. At 6 columns of 2.25in
    # it is 13.5in and 8pt ticks render at ~7.5pt -- half again as large, with two more sessions
    # visible at once.
    n = len(mats)
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.25 * cols + 0.9, 2.55 * rows + 1.0),
                             squeeze=False, gridspec_kw={"hspace": 0.42, "wspace": 0.18})
    short = [POS_SHORT.get(p, p) for p in BY_SEVERITY]
    im = None
    for k, (lab, M) in enumerate(mats):
        ax = axes[k // cols][k % cols]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for r in range(len(BY_SEVERITY)):
            for cc in range(len(BY_SEVERITY)):
                if np.isfinite(M[r, cc]):
                    ax.text(cc, r, f"{M[r, cc]:+.1f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(BY_SEVERITY)))
        # Bottom-most panel IN EACH COLUMN, not the last row: the final row is short whenever the
        # session count is not a multiple of `cols`, which would leave most columns with no x axis.
        ax.set_xticklabels(short if k + cols >= n else [], rotation=90, ha="center", fontsize=8)
        ax.set_yticks(range(len(BY_SEVERITY)))
        ax.set_yticklabels(short if k % cols == 0 else [], fontsize=8)
        ax.set_title(lab.split("_")[-1], fontsize=10)
        ax.tick_params(length=2, pad=1)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    # ONE COLOUR BAR, not one per panel. Every panel already shares vmin/vmax=+/-lim, so N of them
    # asserted a per-panel scale that does not exist while taking width from the maps themselves.
    # Added after the layout is final so it cannot be displaced by it.
    fig.tight_layout(rect=(0, 0, 0.93, 0.86))
    cax = fig.add_axes([0.945, 0.22, 0.012, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("difference from the pre-stroke baseline", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{res['animal']} \u2014 {disp}, {meth.upper()}: {STYLE[cls][2]}, MINUS the pre-stroke "
        f"baseline, ONE MATRIX PER SESSION.\nRows = TRUE spout position, columns = the direction "
        f"scored on. Red = more like the column's position than pre-stroke, blue = less. A row "
        f"going red OFF the diagonal is a remapping; a row going blue ACROSS is loss of that "
        f"position's pattern. Read across sessions: a settled deficit looks the same each day, a "
        f"moving one does not.", fontsize=10)
    q = Path(out) / f"coding_crosssess_{disp}_{meth}_{cls}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q

def figure_pooled(res, out, align="precue", meth="dom"):
    """Every class per position, POOLED over all sessions of its phase.

    The per-session view is the honest one but it is noisy at that grain -- PS94's
    miss-while-working swings +1.05 to -0.68 between adjacent sessions with intervals to match.
    Pooling collapses that; read it WITH the time course, never instead of it, or a class that
    swung wildly and a class that sat still look identical here.
    """
    if not res or meth not in res.get("methods", {}):
        return None
    disp, R = dict(ALIGNS)[align], res["methods"][meth]
    if "pooled" not in R:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    x = np.arange(len(BY_SEVERITY))
    for c in CLASSES_FULL:
        if c not in R["pooled"]:
            continue
        col, mk, lab = STYLE[c]
        cells = [(R["pooled"][c].get(P) or {}) for P in BY_SEVERITY]
        ax.errorbar(x, [cc.get("mean") if cc.get("mean") is not None else np.nan for cc in cells],
                    yerr=[cc.get("sem") or 0 for cc in cells], fmt="-", marker=mk, color=col,
                    ecolor=col, capsize=3, ms=6.5, lw=1.6, label=lab)
        for xi, cc in zip(x, cells):
            if cc.get("mean") is not None and cc.get("low_n"):
                ax.plot([xi], [cc["mean"]], marker=mk, ms=8.5, markerfacecolor="none",
                        markeredgecolor=col, markeredgewidth=1.6, zorder=4)
    ax.axhline(1.0, color="tab:blue", ls=":", lw=1.1)
    ax.axhline(0.0, color="k", ls=":", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(BY_SEVERITY, rotation=30, ha="right")
    ax.set_ylabel("projection")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"{res['animal']} \u2014 {disp}, {meth.upper()}: POOLED over sessions. "
                 f"0 = pre-stroke not-this-position, 1 = pre-stroke lick here.\n"
                 f"Positions MOST IMPAIRED first. Bars = SEM. Read beside the per-session figure: "
                 f"pooling hides whether a class was steady or swinging.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    q = Path(out) / f"coding_pooled_{disp}_{meth}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_rt_split(res, out, align="precue", meth="dom"):
    """Frozen pre-stroke decoder on EARLY vs LATE rewarded post-stroke trials.

    THE QUESTION. `decode.max_rt_s` is 3.5 s, so "engaged" holds a 0.2 s lick and a 3.0 s lick in one
    class, and post-stroke the mass moves late. Position coding PRESERVED on late trials means the
    plan is intact and execution is slow; DEGRADED on late trials is a different result. Every other
    figure in this deck averages the two together and so cannot distinguish them.

    METHOD-INDEPENDENT, like `figure_engagement`: these confusions come from the frozen decoder, not
    from a coding direction, so there is one per animal per window rather than one per method. The
    `meth` argument is accepted and ignored so the driver can call it in the same loop shape.

    The two post panels are a PARTITION of the lick class -- `early + late` is the `poststroke_lick`
    matrix cell for cell (`tests/test_class_confusions.py`) -- so the reader can add them back and
    recover the familiar figure, and any difference between the panels is a real regrouping of the
    same trials rather than two differently-selected populations.
    """
    c = (res or {}).get("confusions") or {}
    if c.get("poststroke_lick_early") is None and c.get("poststroke_lick_late") is None:
        return None
    disp = dict(ALIGNS)[align]
    short = [POS_SHORT.get(p, p) for p in c["labels"]]
    cut = c.get("rt_split_s", RT_SPLIT_S)

    panels = [("prestroke_lick", "PRE-stroke LICK\n(leave-one-session-out)"),
              ("poststroke_lick_early", f"POST-stroke EARLY\nlick < {cut:g} s"),
              ("poststroke_lick_late", f"POST-stroke LATE\nlick \u2265 {cut:g} s")]

    # 10.4in for four panels rather than a wider canvas: the deck places this ~9in, so a figure this
    # size renders 9pt type at ~7.8pt. A 16in version of the same axes would land at 5pt.
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.95), squeeze=False,
                             gridspec_kw={"wspace": 0.40, "width_ratios": [1, 1, 1, 1.25]})
    axes = axes[0]
    im = None
    for ax, (key, title) in zip(axes[:3], panels):
        M = c.get(key)
        if M is None:
            ax.set_axis_off()
            ax.set_title(f"{title}\nno trials", fontsize=8)
            continue
        M = np.array(M, float)
        n = int(M.sum())
        # ROW-NORMALISED FOR DISPLAY ONLY. The stored matrices are raw counts precisely so they stay
        # addable; normalising here is what makes three panels of very different n comparable by eye.
        rows = M.sum(1, keepdims=True)
        P = np.divide(M, rows, out=np.zeros_like(M), where=rows > 0)
        im = ax.imshow(P, vmin=0, vmax=1, cmap="viridis")
        acc = float(np.trace(M) / n) if n else float("nan")
        # LOW n IS PART OF THE TITLE, not a footnote. Measured 2026-08-28 on the real pool: the LATE
        # arm is 3.4% of post-stroke rewarded trials overall and only 26 (PS94) and 27 (PS95) trials
        # in total, so its 6x6 rests on ~4 trials per row. Without this a reader sees "late decodes
        # at chance" and reads a RESULT, when the honest statement is that it cannot be measured.
        _thin = ("  -- TOO FEW TO READ" if n < len(short) * MIN_TRIALS else "")
        ax.set_title(f"{title}\nn={n}, acc={acc:.2f}{_thin}", fontsize=8,
                     color=("#b2182b" if _thin else "black"))
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=7, rotation=90)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short if ax is axes[0] else [], fontsize=7)
        ax.set_xlabel("predicted", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("true position", fontsize=8)

    # FOURTH PANEL: per-position recall, early vs late, which is the actual comparison. Reading it
    # off two heatmaps means comparing colours across panels, and that is exactly the judgement the
    # eye is worst at.
    ax = axes[3]
    E, L = c.get("poststroke_lick_early"), c.get("poststroke_lick_late")
    x = np.arange(len(short))
    for M, col, lab in ((E, "#2166ac", f"early (<{cut:g}s)"),
                        (L, "#b2182b", f"late (\u2265{cut:g}s)")):
        if M is None:
            continue
        M = np.array(M, float)
        rows = M.sum(1)
        # A POSITION WITH TOO FEW TRIALS IS A GAP, NOT A ZERO. Plotting 0 recall on an empty row
        # reads as "the decoder failed here" when nothing was asked of it -- and plotting 0.0 or 1.0
        # on a row of two trials is worse, because it looks like a measurement. `FLOOR_TRIALS` is the
        # same floor `_stats` and `_cells` use, so a cell that is too thin to report here is too thin
        # to report anywhere in this module.
        rec = np.divide(np.diag(M), rows, out=np.full(len(rows), np.nan),
                        where=rows >= FLOOR_TRIALS)
        ax.plot(x, rec, "-", color=col, lw=1.5, label=f"{lab}  n={int(M.sum())}")
        # HOLLOW BELOW `MIN_TRIALS`, the house idiom from `figure_pooled`: the point is drawn because
        # it exists, and marked because it should not be read as firmly as a filled one.
        solid = rows >= MIN_TRIALS
        ax.plot(x[solid], rec[solid], "o", color=col, ms=5)
        thin = (~solid) & np.isfinite(rec)
        ax.plot(x[thin], rec[thin], "o", ms=6, markerfacecolor="none", markeredgecolor=col,
                markeredgewidth=1.5)
        for xi in x[thin]:
            ax.annotate(f"n={int(rows[xi])}", (xi, rec[xi]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=5.5, color=col)
    ax.axhline(1 / len(short), color="k", ls=":", lw=1.1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7, rotation=90)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("recall", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title(f"per-position recall  (hollow = n<{MIN_TRIALS}; "
                 f"no point = n<{FLOOR_TRIALS})", fontsize=7.5)

    fig.suptitle(f"{res['animal']} \u2014 {disp}: frozen PRE-stroke decoder on post-stroke "
                 f"rewarded trials, split at {cut:g} s. Same decoder, same trials as the LICK "
                 f"class, regrouped by reaction time.", fontsize=9.5, y=0.965)
    # ABSOLUTE MARGINS, NOT `tight_layout`. Two reasons, both learned the hard way on 2026-08-28.
    # `imshow` fixes an axes aspect, which makes the figure "not compatible with tight_layout" --
    # matplotlib warns once per panel and the nightly log fills with it, which is how a real
    # [layout] line stops being noticed. And a negotiated layout is not reproducible: `_delta_grid`
    # was clean at 10 post-stroke days and overlapping at 12 because its margins were fractional.
    # These numbers are checked by driving the function, not by reading them (see
    # scratchpad/smoke_rtsplit.py and `_overlaps`).
    fig.subplots_adjust(left=0.055, right=0.918, top=0.735, bottom=0.225, wspace=0.40)
    if im is not None:
        # Its OWN axes, added last, or the colorbar steals width from the final panel -- the fault
        # found on the section D grid on 2026-08-28.
        cax = fig.add_axes([0.935, 0.225, 0.012, 0.51])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("P(predicted | true)", fontsize=7.5)
        cb.ax.tick_params(labelsize=6.5)
    q = Path(out) / f"coding_rtsplit_{disp}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_within_session(res, out, align="precue", meth="dom"):
    """Each class across the COURSE of a session, in within-session quartiles.

    Engagement is graded, not binary: a miss-while-working trial just before the animal quits is not
    the same state as one at trial 50, and a session-level split cannot see that. Trials are pooled
    across sessions of a phase and binned by position WITHIN their session, so this reads as "how
    does the state evolve over a typical session" (Priya, 2026-08-21).
    """
    if not res or meth not in res.get("methods", {}):
        return None
    disp, R = dict(ALIGNS)[align], res["methods"][meth]
    if "within_session" not in R:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2), squeeze=False, sharey=True, sharex=True)
    for k, P in enumerate(BY_SEVERITY):
        ax = axes[k // 3][k % 3]
        for c in CLASSES_FULL:
            rows = (R["within_session"].get(c) or {}).get(P)
            if not rows:
                continue
            col, mk, lab = STYLE[c]
            xq = np.arange(len(rows))
            # A THIN BIN IS NOT PLOTTED HERE, only counted. Hollow markers are enough in the
            # per-session figure, where a low-n point tells you the class exists at all; here they
            # are not. STOPPED is terminal BY DEFINITION, so its early bins are guaranteed thin --
            # 0 trials in the first quartile at every position, then 4-14 in the second -- and those
            # few trials landed at +2.2 and -2.3, dominating the eye and inventing a within-session
            # shape out of one session's tail (Priya, 2026-08-21). Same rule G2b uses: say "n=", do
            # not draw a value you would not weigh.
            def _thin(r):
                return (r.get("mean") is None or r.get("sem") is None
                        or r["sem"] > MAX_SEM_DRAWN)

            ys = [np.nan if _thin(r) else r["mean"] for r in rows]
            es = [0 if _thin(r) else r["sem"] for r in rows]
            ax.errorbar(xq, ys, yerr=es, fmt="-", marker=mk, color=col, ecolor=col, capsize=2.5,
                        ms=5.5, lw=1.4, label=(lab if k == 0 else None))
            for xi, r in zip(xq, rows):
                if _thin(r):
                    ax.text(xi, 0.02, f"n={r.get('n', 0)}", ha="center", va="bottom", fontsize=5.5,
                            rotation=90, color=col, style="italic",
                            transform=ax.get_xaxis_transform())
        ax.axhline(1.0, color="tab:blue", ls=":", lw=1.0)
        ax.axhline(0.0, color="k", ls=":", lw=1.0)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["0-25%", "25-50%", "50-75%", "75-100%"], fontsize=8)
        ax.set_title(P, fontsize=10)
        ax.grid(alpha=0.25)
        if k % 3 == 0:
            ax.set_ylabel("projection", fontsize=9)
        if k >= 3:
            ax.set_xlabel("position within session", fontsize=9)
    fig.legend(loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle(f"{res['animal']} \u2014 {disp}, {meth.upper()}: over the COURSE of a session. "
                 f"Trials pooled across sessions of a phase, binned by where they fall within "
                 f"their own session.\nA state that drifts as the animal tires shows here and "
                 f"cannot show in a session-level split. Bars = SEM; hollow = few trials.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    q = Path(out) / f"coding_within_{disp}_{meth}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q



# ---------------------------------------------------------------------------------------------
# DIAGNOSTIC FIGURES. Each of these was a throwaway script that produced a number, and the number
# then lived in a speaker note that nothing regenerated -- the same failure mode as any other stale
# prose, except invisible to the staleness manifest because there was no file to age
# (Priya, 2026-08-22: "ensure each figure's analysis is clearly explained in the slide notes").
# ---------------------------------------------------------------------------------------------

def figure_engagement(res, out, align="precue", meth="dom"):
    """BEHAVIOUR: response rate per position across the course of a session, pre and post stroke.

    This is the figure that decides how to read the within-session neural panel. Pre-stroke, PS94 and
    PS95 lose a quarter to a third of their responding by the last quartile while PS92 and PS93 lose
    under a tenth -- and the two that disengage are exactly the two whose neural projection drifts.
    """
    disp = dict(ALIGNS)[align]
    rq = (res or {}).get("response_by_quartile")
    if not rq:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True, squeeze=False)
    for ax, ph, ttl in ((axes[0][0], "pre", "PRE-stroke"), (axes[0][1], "post", "POST-stroke")):
        rows = rq.get(ph) or {}
        any_pt = False
        for P in BY_SEVERITY:
            cells = rows.get(P) or []
            xs = [i for i, c in enumerate(cells) if c.get("rate") is not None]
            ys = [cells[i]["rate"] for i in xs]
            if len(xs) < 2:
                continue
            any_pt = True
            ax.plot(xs, ys, marker="o", ms=5, lw=1.6, color=_pos_color(P), label=P)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks(range(4)); ax.set_xticklabels(QLABELS, fontsize=8)
        ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
        ax.set_xlabel("position within session")
        if not any_pt:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    axes[0][0].set_ylabel("response rate")
    axes[0][0].legend(fontsize=7.5, ncol=2, frameon=False)
    # THE WINDOW BELONGS IN THE TITLE even though this panel is behaviour. `disp` reached the
    # FILENAME and not the caption, so the three files (ENL / cue / lick) were captioned
    # identically -- and they are NOT the same figure: each alignment keeps a different trial set
    # (pre-cue drops trials with no lick-free window, lick drops positions with no engaged trial),
    # so the response rates can differ between them. Audited 2026-08-24.
    fig.suptitle(f"{res['animal']} \u2014 BEHAVIOUR: response rate by position over the COURSE of a "
                 f"session, on the trials the {disp} window keeps.\nA terminal collapse here is "
                 f"DISENGAGEMENT (reward is auto-held after a miss run), and it is what a "
                 f"within-session neural decline has to be read against.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    q = Path(out) / f"coding_engagement_{disp}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_norm_unit(res, out, align="precue", meth="dom"):
    """Is the post-stroke value DIRECTION or MAGNITUDE?

    x.w grows either because the trial points more along w (position structure) or because it sits
    further from its session's engaged centroid (everything else). The two are not independent, so
    correlating the projection with the norm cannot separate them -- re-projecting UNIT-NORMALISED
    trials can, being blind to magnitude and sensitive only to direction.
    """
    disp = dict(ALIGNS)[align]
    dg = ((res or {}).get("methods", {}).get(meth) or {}).get("diagnostics") or {}
    pos = dg.get("positions") or {}
    have = [P for P in BY_SEVERITY if (pos.get(P) or {}).get("post_raw") is not None]
    if not have:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False)
    ax = axes[0][0]
    x = np.arange(len(have))
    ax.bar(x - 0.2, [pos[P]["post_raw"] for P in have], 0.4, label="raw  x\u00b7w", color="tab:green")
    ax.bar(x + 0.2, [pos[P]["post_unit"] for P in have], 0.4, label="unit-norm  cos(x,w)",
           color="tab:olive")
    ax.axhline(1.0, ls=":", lw=1, color="tab:blue")
    ax.axhline(0.0, ls=":", lw=1, color="k")
    ax.set_xticks(x); ax.set_xticklabels(have, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("post-stroke LICK projection")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3, axis="y")
    ax.set_title("1 = pre-stroke lick here, 0 = pre-stroke not-here", fontsize=9)

    ax2 = axes[0][1]
    ax2.scatter([pos[P]["norm_ratio"] for P in have], [pos[P]["post_raw"] for P in have],
                s=45, color="tab:green", label="raw")
    ax2.scatter([pos[P]["norm_ratio"] for P in have], [pos[P]["post_unit"] for P in have],
                s=45, marker="s", facecolors="none", edgecolors="tab:olive", label="unit-norm")
    for P in have:
        ax2.annotate(P, (pos[P]["norm_ratio"], pos[P]["post_raw"]), fontsize=6.5,
                     xytext=(3, 3), textcoords="offset points")
    lim = [min(0.7, *[pos[P]["norm_ratio"] for P in have]), max(1.6, *[pos[P]["norm_ratio"] for P in have])]
    ax2.plot(lim, lim, ls="--", lw=1, color="grey", label="pure gain (proj = ratio)")
    ax2.set_xlabel("post/pre feature-norm ratio"); ax2.set_ylabel("projection")
    ax2.legend(fontsize=7.5, frameon=False); ax2.grid(alpha=0.3)
    ax2.set_title("on the dashed line = magnitude alone", fontsize=9)

    fig.suptitle(f"{res['animal']} \u2014 {disp}, {meth.upper()}: is the post-stroke value DIRECTION "
                 f"or MAGNITUDE?\nIf the two bars agree, magnitude contributes nothing and the value "
                 f"is directional \u2014 real position structure.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    q = Path(out) / f"coding_normunit_{disp}_{meth}_{res['animal']}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_cos_slope(everything, out, align="precue", meth="dom"):
    """WHY the within-session decline is close-specific when the disengagement is not.

    THE ONE-LINE VERSION. x = how much a position's axis is really the close-vs-far dimension.
    y = how much that position's own pre-stroke lick trials drift over the session. If a uniform
    state shift runs along close-vs-far, then the more an axis points along that dimension, the more
    drift it must show -- a downward slope here. Nothing about a position's own coding has to change
    for that to happen, which is the point: the decline is a property of the AXIS, not the spout.
    """
    disp = dict(ALIGNS)[align]
    pts = []
    for an, res in sorted((everything or {}).items()):
        dg = ((res or {}).get("methods", {}).get(meth) or {}).get("diagnostics") or {}
        for P, c in (dg.get("positions") or {}).items():
            if c.get("prestroke_lick_slope") is not None:
                pts.append((an, P, c["cos_closefar"], c["prestroke_lick_slope"]))
    if len(pts) < 6:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for an in sorted({p[0] for p in pts}):
        sub = [p for p in pts if p[0] == an]
        for mk, ring in (("o", "close"), ("^", "far")):
            g = [p for p in sub if _ring(p[1]) == ring]
            if g:
                ax.scatter([p[2] for p in g], [p[3] for p in g], s=52, marker=mk,
                           color=config.animal_color().get(an, 'k'),
                           label=(an if ring == "close" else None))
        for p in sub:
            ax.annotate(p[1], (p[2], p[3]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    xs = np.array([p[2] for p in pts]); ys = np.array([p[3] for p in pts])
    r = float(np.corrcoef(xs, ys)[0, 1])
    b, a = np.polyfit(xs, ys, 1)
    gx = np.linspace(xs.min(), xs.max(), 20)
    ax.plot(gx, a + b * gx, ls="--", lw=1.3, color="k")
    ax.axhline(0, ls=":", lw=1, color="grey"); ax.axvline(0, ls=":", lw=1, color="grey")
    ax.set_xlabel("cos(position axis, close-vs-far axis)\n"
                  "negative = the axis points toward FAR, positive = toward CLOSE", fontsize=9)
    ax.set_ylabel("within-session drift of PRE-STROKE LICK\n(last quartile minus first)", fontsize=9)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    ax.set_title(f"{disp}, {meth.upper()}: the drift is a property of the AXIS, not the spout   "
                 f"(n={len(pts)}, r={r:+.3f})", fontsize=10)
    fig.tight_layout()
    q = Path(out) / f"coding_cosslope_{disp}_{meth}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def figure_pairwise_split(everything, out, align="precue", meth="dom"):
    """WHICH pairwise cells are safe to read: within-ring, not cross-ring.

    A pairwise axis contrasts two spouts directly. When both sit at the same distance (within-ring)
    it barely touches the close-vs-far dimension; when it spans the rings it does. The prediction is
    that only the cross-ring cells inherit the drift, and only in animals that HAVE a state drift --
    which is why this is split by animal as well as by pair type. Pooled, the effect vanishes.
    """
    disp = dict(ALIGNS)[align]
    rows = []
    for an, res in sorted((everything or {}).items()):
        dg = ((res or {}).get("methods", {}).get(meth) or {}).get("diagnostics") or {}
        for pr in (dg.get("pairwise") or []):
            if pr.get("slope") is not None:
                rows.append((an, bool(pr["same_ring"]), float(pr["cos_closefar"]), float(pr["slope"])))
    if len(rows) < 8:
        return None
    animals = sorted({r[0] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), squeeze=False)

    ax = axes[0][0]
    for j, same in enumerate((True, False)):
        v = [abs(r[2]) for r in rows if r[1] is same]
        ax.bar(j, np.mean(v) if v else 0, 0.55,
               color=("tab:blue" if same else "tab:orange"))
        ax.scatter(np.full(len(v), j) + np.linspace(-0.18, 0.18, len(v)), v, s=12, color="k", zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["within-ring\n(close-close, far-far)",
                                               "cross-ring\n(one of each)"], fontsize=8.5)
    ax.set_ylabel("|cos| with the close-vs-far axis")
    ax.set_title("how much of the distance dimension each pair type carries", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[0][1]
    w = 0.36
    for j, same in enumerate((True, False)):
        for k, an in enumerate(animals):
            v = [r[3] for r in rows if r[1] is same and r[0] == an]
            if not v:
                continue
            xpos = k + (-w / 2 if same else w / 2)
            npos = sum(1 for x in v if x > 0)
            ax2.bar(xpos, np.mean(v), w * 0.9,
                    color=("tab:blue" if same else "tab:orange"),
                    label=(("within-ring" if same else "cross-ring") if k == 0 else None))
            ax2.scatter(np.full(len(v), xpos) + np.linspace(-0.1, 0.1, len(v)), v, s=10,
                        color="k", zorder=3)
            ax2.annotate(f"{npos}/{len(v)}", (xpos, 0), fontsize=6.5, ha="center", va="bottom"
                         if np.mean(v) < 0 else "top", xytext=(0, 2 if np.mean(v) < 0 else -2),
                         textcoords="offset points")
    ax2.axhline(0, lw=1, color="k")
    ax2.set_xticks(range(len(animals))); ax2.set_xticklabels(animals, fontsize=9)
    ax2.set_ylabel("within-session drift (last quartile minus first)")
    ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("drift per animal. numbers = how many of the pairs went POSITIVE\n"
                  "(A is the FAR position in every cross-ring pair)", fontsize=9)

    fig.suptitle(f"{disp}, {meth.upper()}: WITHIN-RING pairwise cells are the ones to read. A "
                 f"cross-ring axis spans the close-vs-far dimension and inherits the session drift "
                 f"\u2014 but only in an animal that HAS one, which is why this is split by animal.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    q = Path(out) / f"coding_pairsplit_{disp}_{meth}.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def _draw(fn, *a, **kw):
    """Call one figure function, reporting a failure instead of taking the run down with it.

    A figure raising used to kill main() AFTER every per-animal PNG was written and BEFORE
    coding_direction.json -- so a one-line plotting bug discarded ~40 minutes of pooling and left the
    figures with no data file beside them (2026-08-22, an argument-count error in the cohort loop).
    The analysis is the expensive part and must survive its own presentation layer.
    """
    try:
        q = fn(*a, **kw)
    except Exception as ex:                                          # noqa: BLE001
        print(f"  !! {fn.__name__}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        return None
    if q:
        print(f"  wrote {q}", flush=True)
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
            # behaviour is method-independent -- one per animal, not one per method
            _draw(figure_engagement, res[an], out, align=align)
            # so is the RT split: it reads the frozen decoder's confusions, not a coding direction
            _draw(figure_rt_split, res[an], out, align=align)
            for meth in res[an].get("methods", {}):
                for fn in (figure_animal, figure_pooled, figure_within_session,
                           figure_cross, figure_pairwise, figure_norm_unit):
                    _draw(fn, res[an], out, align=align, meth=meth)
                for _cls in PAIRSESS_CLASSES:
                    _draw(figure_pairwise_sessions, res[an], out, align=align, meth=meth, cls=_cls)
                    _draw(figure_cross_sessions, res[an], out, align=align, meth=meth, cls=_cls)
        # COHORT-level diagnostics: both describe how the AXES behave across animals, so neither can
        # be drawn per animal -- the pairwise split in particular VANISHES when animals are pooled,
        # which is the whole reason it is a figure.
        for meth in sorted({m for r in res.values() if r for m in r.get("methods", {})}):
            for fn in (figure_cos_slope, figure_pairwise_split):
                _draw(fn, {a: r for a, r in res.items() if r}, out, align=align, meth=meth)
    (out / "coding_direction.json").write_text(
        json.dumps(everything, indent=1, default=float), encoding="utf-8")
    print("wrote coding_direction.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
