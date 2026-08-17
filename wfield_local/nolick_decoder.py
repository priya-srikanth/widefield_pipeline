"""Per-session, per-animal and cross-session decoding of trials with NO DETECTED LICK.

Runs the analysis described in `nolick_analysis` over real sessions and writes the frozen pre-stroke
reference. The scientific framing, the naming, and the reason the obvious statistics are wrong all
live in that module's docstring; this one is the plumbing.

WHAT IS SHARED AND WHAT IS NOT. Features come from `locanmf_position_decoder`'s own helpers
(`_build_signal`, `_window_feature`, `_bins_for`), so an engaged trial here is byte-identical to an
engaged trial in the deck. Only the CATEGORISATION is new, which is the whole point of the module:

    engaged      first detected lick within decode.max_rt_s of the cue      (the decoder's set)
    late         first detected lick AFTER max_rt but within LATE_S         (NEW -- see below)
    undetected   no detected lick at all within LATE_S                      (NEW)

The pipeline's existing "no-lick" arm is `late + undetected` lumped together. Splitting them matters
post-stroke: a slowed-but-completed movement and an absent one are different injuries, and lumping
them guarantees the analysis cannot tell them apart. Pre-stroke the late group is small; that is
itself the reference value.

The engaged arm is additionally split at the median RT into fast/slow, which gives a graded
within-engaged handle on the same axis without changing what "engaged" means.

CAVEAT CARRIED IN THE DATA, NOT JUST THE PROSE. "Undetected" is not "no attempt": the sensor needs
contact, so a short lick registers as nothing. PS93 reaches far_L poorly (pre-existing rightward
tongue bias), so its far-position undetected trials are substantially attempted-and-short. Per-
position output is therefore always written, and `ATTEMPT_CONFOUNDED` records the animal/positions
where this is known to apply, so a later DLC pass can target them rather than rediscover them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from wfield_local import config, nolick_analysis as na
from wfield_local.behavior_position import classify_cues_with_backup
from wfield_local.locanmf_crossanimal_dff import _frames
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import (
    _bins_for,
    _build_signal,
    _load_cue_events,
    _window_feature,
)
from wfield_local.locanmf_frozen_decoder import _pipe
from wfield_local.plot_lick_aligned_averages import (
    DISPLAY_ORDER,
    POSITION_NAMES,
    _load_daq_events,
)

FS = 31.23
LATE_S = 5.0          # a lick this long after the cue is a late response, not a different trial
CATEGORIES = ("engaged", "late", "undetected")

# Known non-stroke reasons a position's "undetected" trials over-represent failed EXECUTION rather
# than absent intent. Recorded so the DLC/facial-tracking pass has a target list.
ATTEMPT_CONFOUNDED = {
    "PS93": {"positions": ["far_L", "far_center"],
             "why": "pre-existing rightward tongue bias; licks occur but frequently do not reach "
                    "the far-left spout (Priya, 2026-08-17). Awaiting DLC/FR to disambiguate."},
}


def _args(source="locanmf", align="cue", post_s=2.0, bins=None):
    return SimpleNamespace(source=source, align=align, baseline="none", pre_s=1.0,
                           post_s=post_s, fs=FS, max_rt=2.0, bins=bins)


def categorize(s, args):
    """Per-cue category, position code and block id -- no imaging touched.

    Returns (codes, cat, blk, rt_s, cue_f) with `cat` one of CATEGORIES or "" for cues with no
    usable position label.
    """
    cue = _load_cue_events(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, _csmp = _frames(s, cue, lk)
    codes = classify_cues_with_backup(s, cue)

    blk = np.full(cue_f.size, -1, int)
    b, prev = -1, None
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        if prev is None or codes[k] != prev:
            b += 1
        blk[k] = b
        prev = int(codes[k])

    ls = np.sort(lick_f)
    j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1)
    rt_n = first - cue_f
    rt_s = np.where(first > 0, rt_n / args.fs, np.nan)

    maxrt_n = int(round(args.max_rt * args.fs))
    late_n = int(round(LATE_S * args.fs))
    cat = np.full(cue_f.size, "", dtype=object)
    for k in range(cue_f.size):
        if codes[k] < 0 or cue_f[k] < 0:
            continue
        if first[k] > 0 and 0 < rt_n[k] <= maxrt_n:
            cat[k] = "engaged"
        elif first[k] > 0 and maxrt_n < rt_n[k] <= late_n:
            cat[k] = "late"
        else:
            cat[k] = "undetected"
    return codes, cat, blk, rt_s, cue_f


def session_features(s, args, signal=None, feat_region=None):
    """Feature matrix per category, using the decoder's own window builder.

    Every category is windowed CUE-referenced (or pre-cue referenced), never lick-referenced --
    trials without a detected lick have no lick to align to, so a lick-aligned comparison between
    the arms cannot exist. That is a property of the question, not a limitation of the code.

    ``signal``/``feat_region`` inject an already-built (nfeat, T) signal instead of loading
    ``args.source`` from disk -- the joint-LocaNMF path, exactly as `locanmf_position_decoder`'s
    own `_trial_features` does it. Everything downstream is then identical by construction, which is
    the point: the basis is the only thing that differs.
    """
    if signal is not None:
        sig = np.asarray(signal)
        feat_reg = (np.arange(sig.shape[0]) if feat_region is None else np.asarray(feat_region))
    else:
        sig, feat_reg = _build_signal(s, args.source)
    nfeat, T = sig.shape
    codes, cat, blk, rt_s, cue_f = categorize(s, args)
    bins = _bins_for(args)
    post_n = int(round(args.post_s * args.fs))

    out = {c: {"X": [], "y": [], "g": [], "rt": []} for c in CATEGORIES}
    for k in range(cue_f.size):
        if not cat[k]:
            continue
        c0 = int(cue_f[k])
        ref0 = c0 - post_n if args.align == "precue" else c0
        if ref0 < 0 or ref0 + post_n > T:
            continue
        d = out[cat[k]]
        d["X"].append(_window_feature(sig, ref0, post_n, bins, 0.0))
        d["y"].append(int(codes[k]))
        d["g"].append(int(blk[k]))
        d["rt"].append(float(rt_s[k]))
    del sig
    if bins > 1:
        feat_reg = np.tile(feat_reg, bins)
    for c in CATEGORIES:
        d = out[c]
        d["X"] = np.array(d["X"]); d["y"] = np.array(d["y"], int)
        d["g"] = np.array(d["g"], int); d["rt"] = np.array(d["rt"], float)
    return out, feat_reg


def analyse_session(s, align="cue", source="locanmf", post_s=2.0, n_perm=na.N_PERM, verbose=True):
    """Train on engaged (block-CV for its own score), apply the fitted model to the other arms."""
    args = _args(source=source, align=align, post_s=post_s)
    F, _feat_reg = session_features(s, args)
    E = F["engaged"]
    res = {"label": s["label"], "align": align, "source": source,
           "n_by_category": {c: int(F[c]["y"].size) for c in CATEGORIES}}
    if E["y"].size < 30 or np.unique(E["y"]).size < 2:
        res["skipped"] = "too few engaged trials to train"
        return res

    ng = min(5, int(np.unique(E["g"]).size))
    pred_e = (cross_val_predict(_pipe(), E["X"], E["y"], cv=GroupKFold(ng), groups=E["g"])
              if ng >= 2 else np.full(E["y"].shape, -1))
    eng_frac = {POSITION_NAMES[c]: float((E["y"] == c).mean()) for c in DISPLAY_ORDER}
    res["engaged"] = na.evaluate_arm(E["y"], pred_e, n_perm=n_perm)

    clf = _pipe().fit(E["X"], E["y"])
    for c in ("late", "undetected"):
        d = F[c]
        if d["y"].size == 0:
            res[c] = {"n": 0}
            continue
        res[c] = na.evaluate_arm(d["y"], clf.predict(d["X"]), target_frac=eng_frac, n_perm=n_perm)

    # the pipeline's historical arm = late + undetected pooled, kept so the new split can be
    # reconciled against every number already in the decks
    pooled_y = np.concatenate([F["late"]["y"], F["undetected"]["y"]]) if (
        F["late"]["y"].size + F["undetected"]["y"].size) else np.array([], int)
    if pooled_y.size:
        pooled_X = np.concatenate([x for x in (F["late"]["X"], F["undetected"]["X"]) if x.size])
        res["nolick_pooled"] = na.evaluate_arm(pooled_y, clf.predict(pooled_X),
                                               target_frac=eng_frac, n_perm=n_perm)
        res["compare"] = na.compare_arms(res["engaged"], res["nolick_pooled"])

    # graded handle WITHIN engaged: does a slower response already carry a weaker code?
    if E["rt"].size >= 60 and np.isfinite(E["rt"]).any():
        med = float(np.nanmedian(E["rt"]))
        for nm, m in (("engaged_fast", E["rt"] <= med), ("engaged_slow", E["rt"] > med)):
            if m.sum() >= 20:
                res[nm] = na.evaluate_arm(E["y"][m], pred_e[m], n_perm=n_perm)
        res["engaged_rt_median_s"] = med

    an = s["label"][:4]
    if an in ATTEMPT_CONFOUNDED:
        res["attempt_confounded"] = ATTEMPT_CONFOUNDED[an]
    if verbose:
        print(f"{s['label']} [{align}] " + "  ".join(
            f"{c}={res['n_by_category'][c]}" for c in CATEGORIES), flush=True)
    return res


def _joint_signal(basis, s):
    """(signal, regions, variance_captured) for one session in a FIXED joint basis.

    A session in the fit keeps its fitted time courses; a new one is PROJECTED onto the same frozen
    footprints. Never refitted -- a refit over a grown session set is a different reference frame,
    and post-stroke that would make the comparison meaningless.
    """
    if s["label"] in basis.labels:
        return basis.signal(s["label"]), basis.regions, 1.0
    sig, diag = basis.project(s, with_diagnostics=True)
    return sig, basis.regions, float(diag["variance_captured"])


def analyse_animal(animal, dates=None, align="cue", source="roi", post_s=2.0,
                   n_perm=na.N_PERM, verbose=True, basis=None):
    """Pool an animal's curated sessions, then train ONE model and apply it to the other arms.

    Pooling before fitting (rather than averaging per-session results) is deliberate: the undetected
    arm is small in some sessions and a per-session accuracy on 8 trials is not a measurement. The
    engaged score stays leave-one-SESSION-out so it remains an out-of-sample number.

    SOURCE MUST BE ROI, for the same reason the frozen decoder's is: LocaNMF components are fitted
    per session and differ in both count and identity, so column j is a different thing on different
    days and stacking them is meaningless (it fails loudly here rather than silently mixing).
    Allen-ROI features are atlas-anchored, so column j is the same cortical area every day.

    Features are z-scored PER SESSION using that session's ENGAGED trials only, exactly as the
    frozen decoder does. Without it a day with a larger F0 or better SNR would shift every arm
    together and the pooled model would partly be learning which day a trial came from. Using the
    engaged trials to define the scaling (not all trials) keeps the transform independent of the
    arm being tested.
    """
    if basis is None and source != "roi":
        raise ValueError(
            f"analyse_animal pools across sessions and needs a basis that is the SAME on every day; "
            f"source={source!r} is fitted per session. Use source='roi', or pass a joint basis "
            f"(joint_locanmf.load) whose footprints are frozen and days projected onto them.")
    dates = set(dates or config.curated_dates())
    labs = [s for s in SESSIONS if s["label"][:4] == animal and s["label"][-4:] in dates]
    args = _args(source=source, align=align, post_s=post_s)
    # per session: the z-scored features of each category, kept beside that session's region labels
    # so the columns can be reconciled afterwards
    per_sess, var_cap = [], {}
    for s in labs:
        try:
            if basis is not None:
                sig, regs_j, vc = _joint_signal(basis, s)
                var_cap[s["label"]] = vc
                F, feat_reg = session_features(s, args, signal=sig, feat_region=regs_j)
                del sig
            else:
                F, feat_reg = session_features(s, args)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        E = F["engaged"]["X"]
        if E.size == 0:
            continue
        mu, sd = E.mean(0), E.std(0)
        sd[sd == 0] = 1.0
        per_sess.append({"label": s["label"], "reg": list(np.asarray(feat_reg)),
                         "cats": {c: ((F[c]["X"] - mu) / sd if F[c]["y"].size else None,
                                      F[c]["y"]) for c in CATEGORIES}})
        if verbose:
            print(f"  {s['label']}: " + " ".join(f"{c}={F[c]['y'].size}" for c in CATEGORIES),
                  flush=True)

    # Sessions can differ in which atlas areas survive registration, so restrict every session to
    # the areas ALL of them have, in one shared order -- otherwise column j is still not the same
    # area everywhere and the per-session z-scoring above would not save it.
    acc = {c: {"X": [], "y": [], "sess": []} for c in CATEGORIES}
    if per_sess:
        first = per_sess[0]["reg"]
        others = [set(p["reg"]) for p in per_sess[1:]]
        common = [r for r in first if all(r in o for o in others)]
        if len(common) < len(first):
            print(f"  [{animal} {align}] restricting to {len(common)}/{len(first)} features "
                  f"present in all {len(per_sess)} sessions", flush=True)
        for p in per_sess:
            idx = [p["reg"].index(k) for k in common]
            for c in CATEGORIES:
                X, y = p["cats"][c]
                if X is None or not y.size:
                    continue
                acc[c]["X"].append(X[:, idx])
                acc[c]["y"].append(y)
                acc[c]["sess"].append(np.full(y.size, p["label"], dtype=object))

    def _cat(k, f):
        return np.concatenate(acc[k][f]) if acc[k][f] else np.array([])

    XE, YE, SE = _cat("engaged", "X"), _cat("engaged", "y").astype(int), _cat("engaged", "sess")
    res = {"animal": animal, "align": align, "n_sessions": len(labs),
           "source": "joint" if basis is not None else source,
           "n_by_category": {c: int(_cat(c, "y").size) for c in CATEGORIES}}
    if basis is not None:
        # a projected day that decodes poorly AND captures little variance is a basis problem, not a
        # coding one; without this the two are indistinguishable on the figure
        res["basis_id"] = basis.basis_id
        res["variance_captured"] = var_cap
    if YE.size < 100:
        res["skipped"] = "too few pooled engaged trials"
        return res

    from sklearn.model_selection import LeaveOneGroupOut
    pred_e = cross_val_predict(_pipe(), XE, YE, cv=LeaveOneGroupOut(), groups=SE)
    res["engaged"] = na.evaluate_arm(YE, pred_e, n_perm=n_perm)
    res["engaged"]["loso_accuracy"] = float(accuracy_score(YE, pred_e))
    eng_frac = {POSITION_NAMES[c]: float((YE == c).mean()) for c in DISPLAY_ORDER}

    clf = _pipe().fit(XE, YE)
    for c in ("late", "undetected"):
        Y = _cat(c, "y").astype(int)
        if not Y.size:
            res[c] = {"n": 0}
            continue
        res[c] = na.evaluate_arm(Y, clf.predict(_cat(c, "X")), target_frac=eng_frac, n_perm=n_perm)

    Yp = np.concatenate([_cat("late", "y"), _cat("undetected", "y")]).astype(int)
    if Yp.size:
        Xp = np.concatenate([x for x in (_cat("late", "X"), _cat("undetected", "X")) if x.size])
        res["nolick_pooled"] = na.evaluate_arm(Yp, clf.predict(Xp), target_frac=eng_frac,
                                               n_perm=n_perm)
        res["compare"] = na.compare_arms(res["engaged"], res["nolick_pooled"])

    # PER SESSION, from the SAME frozen model rather than 61 separate refits. Two reasons, and the
    # cost saving is the lesser one: a per-session refit answers "could a model fitted here decode
    # here", while a post-stroke session will be scored by a model it had no part in fitting, so
    # this is the quantity that will actually be compared. The engaged arm uses the leave-one-
    # SESSION-out predictions, so it stays out-of-sample too.
    #
    # A session's undetected arm is often a dozen trials, which is not a measurement. Each one
    # therefore carries its own permutation null and n; anything read off a single session should be
    # read against those, not against the pooled number.
    sess_labels = sorted(set(SE.tolist()))
    per = {}
    idx_e = {lab: np.flatnonzero(SE == lab) for lab in sess_labels}
    cat_sess = {c: _cat(c, "sess") for c in CATEGORIES}
    cat_y = {c: _cat(c, "y") for c in CATEGORIES}
    cat_X = {c: _cat(c, "X") for c in CATEGORIES}
    for lab in sess_labels:
        d = {}
        ie = idx_e[lab]
        if ie.size >= 20:
            d["engaged"] = na.evaluate_arm(YE[ie], pred_e[ie], n_perm=max(200, n_perm // 4))
        for c in ("late", "undetected"):
            if not cat_y[c].size:
                continue
            m = np.flatnonzero(cat_sess[c] == lab)
            if m.size == 0:
                d[c] = {"n": 0}
                continue
            d[c] = na.evaluate_arm(cat_y[c][m].astype(int), clf.predict(cat_X[c][m]),
                                   n_perm=max(200, n_perm // 4))
        nl = [c for c in ("late", "undetected") if isinstance(d.get(c), dict) and d[c].get("n")]
        if nl and "engaged" in d:
            ym = np.concatenate([cat_y[c][np.flatnonzero(cat_sess[c] == lab)] for c in nl]).astype(int)
            xm = np.concatenate([cat_X[c][np.flatnonzero(cat_sess[c] == lab)] for c in nl])
            d["nolick_pooled"] = na.evaluate_arm(ym, clf.predict(xm),
                                                 n_perm=max(200, n_perm // 4))
            d["compare"] = na.compare_arms(d["engaged"], d["nolick_pooled"])
        if basis is not None and lab in var_cap:
            d["variance_captured"] = var_cap[lab]
        per[lab] = d
    res["per_session"] = per
    if animal in ATTEMPT_CONFOUNDED:
        res["attempt_confounded"] = ATTEMPT_CONFOUNDED[animal]
    return res


BASES = ("roi", "joint")


def build_reference(animals=None, dates=None, bases=BASES, out=None, n_perm=na.N_PERM):
    """The frozen PRE-STROKE reference: both bases, both alignments, all animals, written once.

    Both alignments are always computed even if a caller only wants one, because the discriminating
    quantity is the pre-cue/post-cue CONTRAST and a reference containing only half of it would
    invite exactly the comparison it cannot support.

    TWO BASES, for the reason the frozen decoder uses two: a cross-day claim that holds in only one
    parcellation is a claim about the parcellation. Allen-ROI is atlas-anchored and assumption-light;
    the joint LocaNMF basis is data-driven, higher-dimensional, and fitted to these animals -- if the
    pre-cue code survives without a lick in ROI features but not in joint components (or the other
    way round), that is about the features, and it needs to be visible rather than a coin-flip of
    which basis someone happened to run. A missing joint basis is REPORTED and skipped, never
    refitted on the fly: a refit over a grown session set is a different reference frame.
    """
    from wfield_local import joint_locanmf

    animals = config.normalize_animals(animals) or ["PS92", "PS93", "PS94", "PS95"]
    dates = sorted(dates or config.curated_dates())
    ref = {"kind": "pre-stroke no-detected-lick reference", "dates": dates, "bases": list(bases),
           "max_rt_s": 2.0, "late_s": LATE_S, "n_perm": n_perm,
           "categories": {"engaged": "first detected lick within max_rt_s",
                          "late": f"first detected lick between max_rt_s and {LATE_S}s",
                          "undetected": f"no detected lick within {LATE_S}s "
                                        "(NOT the same as no attempt -- see ATTEMPT_CONFOUNDED)"},
           "attempt_confounded": ATTEMPT_CONFOUNDED, "by_basis": {}}
    for bkey in bases:
        ref["by_basis"][bkey] = {"animals": {}}
        for an in animals:
            basis = None
            if bkey == "joint":
                try:
                    basis = joint_locanmf.load(an)
                except Exception as ex:                                # noqa: BLE001
                    print(f"[{an}] no joint basis ({type(ex).__name__}: {str(ex)[:60]}) -- skipped",
                          flush=True)
                    continue
            slot = ref["by_basis"][bkey]["animals"].setdefault(an, {})
            for al in ("precue", "cue"):
                print(f"[{bkey} {an} {al}]", flush=True)
                slot[al] = analyse_animal(an, dates=dates, align=al, n_perm=n_perm, basis=basis)
            pc, cc = slot["precue"].get("compare"), slot["cue"].get("compare")
            if pc and cc:
                # pass the ARMS too, so the verdict rests on each arm's own permutation p rather
                # than on a ratio whose denominator might simply be small
                slot["interpretation"] = na.interpret(
                    pc, cc, slot["precue"].get("nolick_pooled"), slot["cue"].get("nolick_pooled"))
            na.summarize({"precue": slot["precue"], "cue": slot["cue"],
                          "interpretation": slot.get("interpretation", "")})
    # a single agreed verdict per animal, or an explicit note that the bases disagree -- so nobody
    # has to reconcile two sections by eye and nobody can quote the convenient one
    ref["consensus"] = {}
    for an in animals:
        got = {b: ref["by_basis"][b]["animals"].get(an, {}).get("interpretation")
               for b in bases if an in ref["by_basis"].get(b, {}).get("animals", {})}
        vals = {v for v in got.values() if v}
        ref["consensus"][an] = (vals.pop() if len(vals) == 1 else
                                {"DISAGREEMENT": got} if vals else None)
    if out:
        na.write_reference(ref, Path(out))
        print(f"wrote {out}", flush=True)
    return ref


def reinterpret(ref):
    """Recompute every verdict from the STORED arms, in place.

    The reference holds all the numbers a verdict is derived from, so changing how they are read
    does not require recomputing them -- which matters, since the compute is hours and the reading
    rule is the part most likely to be revised. It also means an old reference can be re-read under
    a new rule and the two compared, instead of one silently replacing the other.
    """
    for _bkey, block in ref.get("by_basis", {}).items():
        for _an, slot in block.get("animals", {}).items():
            pc = (slot.get("precue") or {}).get("compare")
            cc = (slot.get("cue") or {}).get("compare")
            if pc and cc:
                slot["interpretation"] = na.interpret(
                    pc, cc, (slot["precue"] or {}).get("nolick_pooled"),
                    (slot["cue"] or {}).get("nolick_pooled"))
            for lab, d in (slot.get("cue") or {}).get("per_session", {}).items():
                pd_ = ((slot.get("precue") or {}).get("per_session", {}) or {}).get(lab, {})
                if d.get("compare") and pd_.get("compare"):
                    d["interpretation"] = na.interpret(
                        pd_["compare"], d["compare"], pd_.get("nolick_pooled"),
                        d.get("nolick_pooled"))
    animals = sorted({a for b in ref.get("by_basis", {}).values() for a in b.get("animals", {})})
    ref["consensus"] = {}
    for an in animals:
        got = {b: blk["animals"][an].get("interpretation")
               for b, blk in ref["by_basis"].items() if an in blk.get("animals", {})}
        # compare the VERDICT CLASS, not the sentence -- the sentences carry per-basis numbers and
        # would never be equal, which would report disagreement on every animal
        cls = {v.split(":")[0].replace("consistent with ", "").strip()
               for v in got.values() if v}
        ref["consensus"][an] = (got[list(got)[0]] if len(cls) == 1 else
                                {"DISAGREEMENT": got} if cls else None)

    # THE THRESHOLD-FREE SUMMARY, and the one to quote. A per-animal verdict is a binary label on a
    # continuous ratio, so an animal near the cut flips basis to basis and the consensus reports
    # "DISAGREEMENT" for what is really 1.4x versus 1.6x. Counting how many animal x basis
    # comparisons put pre-cue survival ABOVE post-cue asks the underlying question directly and
    # depends on no threshold at all -- so a reader can see whether a split verdict means the bases
    # contradict each other or merely straddle a line someone drew.
    ratios, above = {}, 0
    for bkey, blk in ref.get("by_basis", {}).items():
        for an, slot in blk.get("animals", {}).items():
            p = ((slot.get("precue") or {}).get("compare") or {}).get("survival_ratio")
            c = ((slot.get("cue") or {}).get("compare") or {}).get("survival_ratio")
            if p is None or c is None or not (np.isfinite(p) and np.isfinite(c)):
                continue
            ratios[f"{an}/{bkey}"] = round(float(p / c), 2) if c else None
            above += int(p > c)
    ref["direction_consistency"] = {
        "n_comparisons": len(ratios), "n_precue_above_postcue": above,
        "dissociation_ratio": ratios,
        "statement": (f"pre-cue survival exceeds post-cue in {above}/{len(ratios)} animal x basis "
                      f"comparisons" + (f"; ratios {min(ratios.values()):.1f}-{max(ratios.values()):.1f}x"
                                        if ratios else "")),
        "note": "threshold-free. Per-animal verdicts binarize this and can split at the margin.",
    }
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--dates", default=None, help="comma/range spec; default = curated dates")
    ap.add_argument("--bases", nargs="+", default=list(BASES), choices=list(BASES),
                    help="feature bases to run. Both by default: a cross-day claim that holds in "
                         "only one parcellation is a claim about the parcellation.")
    ap.add_argument("--n-perm", type=int, default=na.N_PERM)
    ap.add_argument("--out", default=None, help="write the frozen reference JSON here")
    ap.add_argument("--reinterpret", default=None, metavar="JSON",
                    help="re-derive verdicts from an EXISTING reference's stored arms and rewrite "
                         "it, without recomputing. Use when the reading rule changes, not the data.")
    a = ap.parse_args()
    if a.reinterpret:
        import json as _json
        pth = Path(a.reinterpret)
        ref = reinterpret(_json.loads(pth.read_text()))
        pth.write_text(_json.dumps(ref, indent=2))
        for an, v in (ref.get("consensus") or {}).items():
            print(f"  {an}: {v if isinstance(v, str) else 'BASES DISAGREE: ' + str(v)}", flush=True)
        print(f"rewrote {pth}", flush=True)
        return
    dates = config.expand_dates(a.dates, width=4) if a.dates else None
    build_reference(animals=a.animals, dates=dates, bases=a.bases, out=a.out, n_perm=a.n_perm)


if __name__ == "__main__":
    main()
