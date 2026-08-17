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


def session_features(s, args):
    """Feature matrix per category, using the decoder's own window builder.

    Every category is windowed CUE-referenced (or pre-cue referenced), never lick-referenced --
    trials without a detected lick have no lick to align to, so a lick-aligned comparison between
    the arms cannot exist. That is a property of the question, not a limitation of the code.
    """
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


def analyse_animal(animal, dates=None, align="cue", source="locanmf", post_s=2.0,
                   n_perm=na.N_PERM, verbose=True):
    """Pool an animal's curated sessions, then train ONE model and apply it to the other arms.

    Pooling before fitting (rather than averaging per-session results) is deliberate: the undetected
    arm is small in some sessions and a per-session accuracy on 8 trials is not a measurement. The
    engaged score stays leave-one-SESSION-out so it remains an out-of-sample number.
    """
    dates = set(dates or config.curated_dates())
    labs = [s for s in SESSIONS if s["label"][:4] == animal and s["label"][-4:] in dates]
    args = _args(source=source, align=align, post_s=post_s)
    acc = {c: {"X": [], "y": [], "sess": []} for c in CATEGORIES}
    for s in labs:
        try:
            F, _ = session_features(s, args)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        for c in CATEGORIES:
            if F[c]["y"].size:
                acc[c]["X"].append(F[c]["X"]); acc[c]["y"].append(F[c]["y"])
                acc[c]["sess"].append(np.full(F[c]["y"].size, s["label"], dtype=object))
        if verbose:
            print(f"  {s['label']}: " + " ".join(f"{c}={F[c]['y'].size}" for c in CATEGORIES),
                  flush=True)

    def _cat(k, f):
        return np.concatenate(acc[k][f]) if acc[k][f] else np.array([])

    XE, YE, SE = _cat("engaged", "X"), _cat("engaged", "y").astype(int), _cat("engaged", "sess")
    res = {"animal": animal, "align": align, "source": source, "n_sessions": len(labs),
           "n_by_category": {c: int(_cat(c, "y").size) for c in CATEGORIES}}
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
    if animal in ATTEMPT_CONFOUNDED:
        res["attempt_confounded"] = ATTEMPT_CONFOUNDED[animal]
    return res


def build_reference(animals=None, dates=None, source="locanmf", out=None, n_perm=na.N_PERM):
    """The frozen PRE-STROKE reference: both alignments, all animals, written once.

    Both alignments are always computed even if a caller only wants one, because the discriminating
    quantity is the pre-cue/post-cue CONTRAST and a reference containing only half of it would
    invite exactly the comparison it cannot support.
    """
    animals = config.normalize_animals(animals) or ["PS92", "PS93", "PS94", "PS95"]
    dates = sorted(dates or config.curated_dates())
    ref = {"kind": "pre-stroke no-detected-lick reference", "dates": dates, "source": source,
           "max_rt_s": 2.0, "late_s": LATE_S, "n_perm": n_perm,
           "categories": {"engaged": "first detected lick within max_rt_s",
                          "late": f"first detected lick between max_rt_s and {LATE_S}s",
                          "undetected": f"no detected lick within {LATE_S}s "
                                        "(NOT the same as no attempt -- see ATTEMPT_CONFOUNDED)"},
           "attempt_confounded": ATTEMPT_CONFOUNDED, "animals": {}}
    for an in animals:
        ref["animals"][an] = {}
        for al in ("precue", "cue"):
            print(f"[{an} {al}]", flush=True)
            r = analyse_animal(an, dates=dates, align=al, source=source, n_perm=n_perm)
            ref["animals"][an][al] = r
        pc, cc = (ref["animals"][an]["precue"].get("compare"),
                  ref["animals"][an]["cue"].get("compare"))
        if pc and cc:
            ref["animals"][an]["interpretation"] = na.interpret(pc, cc)
        na.summarize({"precue": ref["animals"][an]["precue"], "cue": ref["animals"][an]["cue"],
                      "interpretation": ref["animals"][an].get("interpretation", "")}
                     if pc and cc else {})
    if out:
        na.write_reference(ref, Path(out))
        print(f"wrote {out}", flush=True)
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--dates", default=None, help="comma/range spec; default = curated dates")
    ap.add_argument("--source", default="locanmf", choices=("locanmf", "roi"))
    ap.add_argument("--n-perm", type=int, default=na.N_PERM)
    ap.add_argument("--out", default=None, help="write the frozen reference JSON here")
    a = ap.parse_args()
    dates = config.expand_dates(a.dates, width=4) if a.dates else None
    build_reference(animals=a.animals, dates=dates, source=a.source, out=a.out, n_perm=a.n_perm)


if __name__ == "__main__":
    main()
