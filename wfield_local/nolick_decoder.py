"""Per-session, per-animal and cross-session decoding of trials with NO DETECTED LICK.

Runs the analysis described in `nolick_analysis` over real sessions and writes the frozen pre-stroke
reference. The scientific framing, the naming, and the reason the obvious statistics are wrong all
live in that module's docstring; this one is the plumbing.

WHAT IS SHARED AND WHAT IS NOT. Features come from `locanmf_position_decoder`'s own helpers
(`_build_signal`, `_window_feature`, `_bins_for`), so an engaged trial here is byte-identical to an
engaged trial in the deck. Only the CATEGORISATION is new, which is the whole point of the module:

    engaged        first detected lick within decode.max_rt_s of the cue    (the decoder's set)
    late_rewarded  lick after max_rt but within the RESPONSE WINDOW          (a hit the decoder drops)
    undetected     no detected lick within the response window

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
    is_engaged,
)
from wfield_local.locanmf_frozen_decoder import _pipe
from wfield_local.plot_lick_aligned_averages import (
    DISPLAY_ORDER,
    POSITION_NAMES,
    _load_daq_events,
)

FS = 31.23
CATEGORIES = ("engaged", "late_rewarded", "undetected")

# NOTHING IS ANALYSED AFTER THE RESPONSE WINDOW (Priya, 2026-08-17). The spout begins MOVING once
# the window closes, so any window extending past it samples the next trial's setup rather than this
# trial's behaviour. The response window is therefore a hard ceiling on every category boundary and
# every feature window here, not a parameter to be widened for statistical convenience.
#
# It also is NOT the decoder's engaged cut. decode.max_rt_s is 2.0 s while the task has run 3500 ms
# throughout (configs/defaults.yaml:120, read per session from that session's gui_config.json), so a
# lick at 2.5 s is a REWARDED HIT that the decoder calls "no lick". Both cuts are reported (Priya's
# choice, 2026-08-17): 2.0 s keeps every existing deck number comparable, the response window is
# what the task means by a response.
DEFAULT_RESPONSE_WINDOW_S = 3.5

# Known non-stroke reasons a position's "undetected" trials over-represent failed EXECUTION rather
# than absent intent. Recorded so the DLC/facial-tracking pass has a target list.
ATTEMPT_CONFOUNDED = {
    "PS93": {"positions": ["far_L", "far_center"],
             "why": "pre-existing rightward tongue bias; licks occur but frequently do not reach "
                    "the far-left spout (Priya, 2026-08-17). Awaiting DLC/FR to disambiguate."},
}


def s_label(s):
    return s.get("label", "?") if isinstance(s, dict) else "?"


def _args(source="locanmf", align="cue", post_s=2.0, bins=None):
    return SimpleNamespace(source=source, align=align, baseline="none", pre_s=1.0,
                           post_s=post_s, fs=FS, max_rt=2.0, bins=bins)


def response_window_for(s, default=DEFAULT_RESPONSE_WINDOW_S):
    """This session's response window in seconds, from its own gui_config.json.

    Read per session rather than taken from defaults because it is a TASK setting that has been
    changed at the rig; `daq_trials.response_window_s` is the pipeline's one resolver for it, so the
    behaviour scoring and this analysis cannot disagree about what a response is.

    THE DATE MUST COME FROM THIS SESSION. An earlier version globbed `{animal}_*` when it could not
    resolve the date and took the first match, so every session silently read its animal's EARLIEST
    config -- all 44 curated sessions reported 3.0 s when the file says 3500 ms. It failed the same
    way for every session, which is exactly why it looked like a consistent finding rather than a
    bug. The date is taken from the session's own DAQ path, and an unresolvable one raises rather
    than falling back to a wrong-but-plausible neighbour.
    """
    import glob as _glob
    import re as _re

    from wfield_local import daq_trials

    animal = s["label"][:4]
    try:
        for cand in config.load_sessions():
            if cand["label"] == s["label"] and cand.get("behavior_trials"):
                win, _src = daq_trials.response_window_s(Path(cand["behavior_trials"]).parent, default)
                return float(win)
        m = _re.search(rf"{animal}_(\d{{8}})_", str(s.get("h5") or "")) or \
            _re.search(r"[/\\](\d{8})[/\\]", str(s.get("h5") or s.get("mc") or ""))
        if not m:
            raise ValueError(f"cannot resolve a date for {s['label']} from its DAQ path")
        yyyymmdd = m.group(1)
        root = config.resolver().root("behavior_logs")
        hits = sorted(_glob.glob(f"{root}/{animal}_{yyyymmdd}_*/gui_config.json"))
        if hits:
            win, _src = daq_trials.response_window_s(Path(hits[0]).parent, default)
            return float(win)
        print(f"  [{s['label']}] no gui_config.json for {yyyymmdd} -> default {default}s", flush=True)
    except Exception as ex:                                            # noqa: BLE001
        print(f"  [{s['label']}] response window lookup failed ({type(ex).__name__}: "
              f"{str(ex)[:60]}) -> {default}s", flush=True)
    return float(default)


def category_for_rt(rt_s, max_rt_s, response_window_s):
    """The category boundary rule, pure so it can be tested without a session.

    `rt_s` is NaN (or non-positive) when no lick was detected at all. The response window is a hard
    ceiling: a lick after it arrives while the spout is already moving, so it is `undetected` --
    there is deliberately no arm past the window (Priya, 2026-08-17).
    """
    cut = min(max_rt_s, response_window_s)      # the engaged cut can never exceed the window
    if rt_s is None or not np.isfinite(rt_s) or rt_s <= 0:
        return "undetected"
    # the SAME predicate the production decoder uses, imported rather than restated
    if is_engaged(1.0, rt_s, cut):
        return "engaged"
    if rt_s <= response_window_s:
        return "late_rewarded"
    return "undetected"


def categorize(s, args):
    """Per-cue category, position code and block id -- no imaging touched.

    Returns (codes, cat, blk, rt_s, cue_f) with `cat` one of CATEGORIES or "" for cues with no
    usable position label.

    Boundaries, all bounded by the response window (see the module constants):

        engaged        first detected lick within args.max_rt of the cue
        late_rewarded  first detected lick between args.max_rt and the response window -- SLOW but
                       still a hit by the task's own definition, which the decoder discards
        undetected     no detected lick within the response window

    There is deliberately NO arm past the response window. A lick at 4 s arrives while the spout is
    already moving, so it belongs to no trial cleanly; such cues count as `undetected`, which is what
    the task scores them as.
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

    rw_s = getattr(args, "response_window_s", None) or response_window_for(s)
    maxrt_n = int(round(min(args.max_rt, rw_s) * args.fs))   # the cut can never exceed the window
    rw_n = int(round(rw_s * args.fs))
    cat = np.full(cue_f.size, "", dtype=object)
    for k in range(cue_f.size):
        if codes[k] < 0 or cue_f[k] < 0:
            continue
        cat[k] = category_for_rt(rt_s[k] if first[k] > 0 else float("nan"),
                                 args.max_rt, rw_s)
    # ENGAGEMENT STATE, per trial. Pre-stroke, a run of misses at the END of a session is satiation,
    # not motor failure (Priya, 2026-08-17) -- so the undetected arm is a MIXTURE of "no plan formed"
    # (sated) and "plan formed, movement fell short" (e.g. PS93 far_L). Pooling them dilutes exactly
    # the contrast this module measures, and post-stroke the mixture will be different again, so the
    # comparison would be between two different blends.
    #
    # flag_engagement is the pipeline's existing gate and is reused rather than reimplemented; it
    # already has the property this needs -- a patch of hard-POSITION misses does not trip it while
    # the animal keeps hitting easy positions, so PS93's far_L attempts stay in the engaged period
    # instead of being written off as disengagement.
    from wfield_local.spout_behavior import flag_engagement
    eng = config.defaults()["behavior"]["engagement"]
    responded = np.array([c == "engaged" or c == "late_rewarded" for c in cat], bool)
    try:
        sess_eng, _info = flag_engagement(responded, window=eng["window_trials"],
                                          min_rate=eng["min_response_rate"],
                                          tail_min_misses=eng["tail_min_misses"])
    except Exception as ex:                                            # noqa: BLE001
        print(f"  [{s_label(s)}] engagement gate failed ({type(ex).__name__}) -> all engaged",
              flush=True)
        sess_eng = np.ones(cue_f.size, bool)
    return codes, cat, blk, rt_s, cue_f, np.asarray(sess_eng, bool)


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
    codes, cat, blk, rt_s, cue_f, sess_eng = categorize(s, args)
    bins = _bins_for(args)
    post_n = int(round(args.post_s * args.fs))

    out = {c: {"X": [], "y": [], "g": [], "rt": [], "sess_eng": []} for c in CATEGORIES}
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
        d["sess_eng"].append(bool(sess_eng[k]))
    del sig
    if bins > 1:
        feat_reg = np.tile(feat_reg, bins)
    for c in CATEGORIES:
        d = out[c]
        d["X"] = np.array(d["X"]); d["y"] = np.array(d["y"], int)
        d["g"] = np.array(d["g"], int); d["rt"] = np.array(d["rt"], float)
        d["sess_eng"] = np.array(d["sess_eng"], bool)
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
    for c in ("late_rewarded", "undetected"):
        d = F[c]
        if d["y"].size == 0:
            res[c] = {"n": 0}
            continue
        res[c] = na.evaluate_arm(d["y"], clf.predict(d["X"]), target_frac=eng_frac, n_perm=n_perm)

    # the pipeline's historical arm = late + undetected pooled, kept so the new split can be
    # reconciled against every number already in the decks
    pooled_y = np.concatenate([F["late_rewarded"]["y"], F["undetected"]["y"]]) if (
        F["late_rewarded"]["y"].size + F["undetected"]["y"].size) else np.array([], int)
    if pooled_y.size:
        pooled_X = np.concatenate([x for x in (F["late_rewarded"]["X"], F["undetected"]["X"]) if x.size])
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
                   n_perm=na.N_PERM, verbose=True, basis=None, max_rt=2.0, return_raw=False):
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
    # max_rt is the ENGAGED cut and is deliberately variable: 2.0 s is the decoder's convention,
    # `None` means "use this session's own response window", which is what the task calls a response.
    args.max_rt = float("inf") if max_rt is None else float(max_rt)
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
                                      F[c]["y"], F[c]["sess_eng"]) for c in CATEGORIES}})
        if verbose:
            print(f"  {s['label']}: " + " ".join(f"{c}={F[c]['y'].size}" for c in CATEGORIES),
                  flush=True)

    # Sessions can differ in which atlas areas survive registration, so restrict every session to
    # the areas ALL of them have, in one shared order -- otherwise column j is still not the same
    # area everywhere and the per-session z-scoring above would not save it.
    acc = {c: {"X": [], "y": [], "sess": [], "eng": []} for c in CATEGORIES}
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
                X, y, eg = p["cats"][c]
                if X is None or not y.size:
                    continue
                acc[c]["X"].append(X[:, idx])
                acc[c]["y"].append(y)
                acc[c]["eng"].append(eg)
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
    for c in ("late_rewarded", "undetected"):
        Y = _cat(c, "y").astype(int)
        if not Y.size:
            res[c] = {"n": 0}
            continue
        res[c] = na.evaluate_arm(Y, clf.predict(_cat(c, "X")), target_frac=eng_frac, n_perm=n_perm)

    # SPLIT THE UNDETECTED ARM BY ENGAGEMENT STATE. A miss inside a working stretch is a candidate
    # "tried and fell short"; a miss in the satiation tail is "stopped working". Pre-stroke they are
    # pooled in every previous version of this analysis, which blends the two mechanisms the module
    # exists to separate -- and post-stroke the blend will differ again, so the comparison would be
    # between two different mixtures rather than between two conditions.
    Yu_all, Xu_all = _cat("undetected", "y"), _cat("undetected", "X")
    Eu = _cat("undetected", "eng")
    if Yu_all.size and Eu.size == Yu_all.size:
        for nm, m in (("undetected_working", Eu.astype(bool)),
                      ("undetected_sated", ~Eu.astype(bool))):
            if m.sum() >= 30:
                res[nm] = na.evaluate_arm(Yu_all[m].astype(int), clf.predict(Xu_all[m]),
                                          target_frac=eng_frac, n_perm=n_perm)
            else:
                res[nm] = {"n": int(m.sum()),
                           "note": "too few trials to evaluate (need >=30)"}

    Yp = np.concatenate([_cat("late_rewarded", "y"), _cat("undetected", "y")]).astype(int)
    if Yp.size:
        Xp = np.concatenate([x for x in (_cat("late_rewarded", "X"), _cat("undetected", "X")) if x.size])
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
        for c in ("late_rewarded", "undetected"):
            if not cat_y[c].size:
                continue
            m = np.flatnonzero(cat_sess[c] == lab)
            if m.size == 0:
                d[c] = {"n": 0}
                continue
            d[c] = na.evaluate_arm(cat_y[c][m].astype(int), clf.predict(cat_X[c][m]),
                                   n_perm=max(200, n_perm // 4))
        nl = [c for c in ("late_rewarded", "undetected") if isinstance(d.get(c), dict) and d[c].get("n")]
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
    res["max_rt_s"] = None if max_rt is None else float(max_rt)
    if return_raw and Yp.size:
        res["_raw"] = {"engaged": (YE.tolist(), pred_e.tolist(), SE.tolist()),
                       "nolick": (Yp.tolist(), clf.predict(Xp).tolist(),
                                  np.concatenate([_cat("late_rewarded", "sess"),
                                                  _cat("undetected", "sess")]).tolist())}
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
           "max_rt_s": 2.0, "response_window_s": "per session (gui_config.json)",
           "no_analysis_past_response_window": True, "n_perm": n_perm,
           "categories": {"engaged": "first detected lick within max_rt_s",
                          "late_rewarded": "lick after max_rt_s but within the response "
                                           "window -- a HIT by the task's definition that the "
                                           "decoder's 2.0s cut discards",
                          "undetected": "no detected lick within the response window (NOT the same "
                                        "as no attempt -- see ATTEMPT_CONFOUNDED). Licks after the "
                                        "window are counted here: the spout is already moving, so "
                                        "they belong to no trial cleanly and are never analysed"},
           "attempt_confounded": ATTEMPT_CONFOUNDED, "by_basis": {}}
    # Both ENGAGED CUTS x both BASES. The cut is folded into the key rather than nesting another
    # level, so every downstream consumer (figures, consensus, direction_consistency) keeps working
    # and each panel is self-labelling. `None` means "this session's own response window".
    combos = [(bk, cut, nm) for bk in bases
              for cut, nm in ((2.0, "2.0s"), (None, "respwin"))]
    for bkey, max_rt, cutname in combos:
        key = bkey if cutname == "2.0s" else f"{bkey}_{cutname}"
        ref["by_basis"][key] = {"animals": {}, "basis": bkey, "engaged_cut": cutname}
        for an in animals:
            basis = None
            if bkey == "joint":
                try:
                    basis = joint_locanmf.load(an)
                except Exception as ex:                                # noqa: BLE001
                    print(f"[{an}] no joint basis ({type(ex).__name__}: {str(ex)[:60]}) -- skipped",
                          flush=True)
                    continue
            slot = ref["by_basis"][key]["animals"].setdefault(an, {})
            for al in ("precue", "cue"):
                print(f"[{bkey} {an} {al} cut={cutname}]", flush=True)
                slot[al] = analyse_animal(an, dates=dates, align=al, n_perm=n_perm, basis=basis,
                                          max_rt=max_rt, return_raw=True)
            pc, cc = slot["precue"].get("compare"), slot["cue"].get("compare")
            if pc and cc:
                # pass the ARMS too, so the verdict rests on each arm's own permutation p rather
                # than on a ratio whose denominator might simply be small
                slot["interpretation"] = na.interpret(
                    pc, cc, slot["precue"].get("nolick_pooled"), slot["cue"].get("nolick_pooled"))
            # the threshold-free test: does the pre-cue MINUS post-cue survival difference exclude
            # zero under a paired bootstrap over sessions?
            slot["dissociation_ci"] = na.dissociation_ci(slot["precue"].pop("_raw", None),
                                                         slot["cue"].pop("_raw", None))
            slot["precue"].pop("_raw", None); slot["cue"].pop("_raw", None)
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
    # derive verdicts/consensus/direction_consistency through the SAME path a reinterpret uses, so a
    # fresh build and a re-read can never disagree about them (they did: a fresh build had no
    # direction_consistency at all, because that block lived only in reinterpret)
    reinterpret(ref)
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
            # normalise an older CI block to the current schema. dissociation_ci itself cannot be
            # recomputed here (it needs the per-trial arrays, which are deliberately not stored), but
            # the VERDICT field can be derived from the stored one-sided p -- so a reference written
            # before the convention was pinned does not sit next to a newer one with a different key
            # and no way to tell which was the test.
            ci = slot.get("dissociation_ci")
            if isinstance(ci, dict) and ci.get("n_boot") and "significant" not in ci:
                ci["significant"] = bool(ci.get("p_one_sided", 1.0) < na.ALPHA)
                ci["test"] = "one-sided (directional); derived on re-read from the stored p"
                ci["difference_ci_is"] = "two-sided 95%, DESCRIPTIVE -- not the test"
                if "excludes_zero" in ci:
                    ci["ci95_two_sided_excludes_zero"] = ci.pop("excludes_zero")
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
    # Consensus is across BASES WITHIN ONE CUT. Comparing across cuts would report "disagreement"
    # for the thing both cuts exist to show -- they are different questions (the decoder's 2.0 s
    # convention vs the task's own response window), not two measurements of one.
    cuts = sorted({blk.get("engaged_cut", "2.0s") for blk in ref["by_basis"].values()})
    ref["consensus"] = {}
    for cut in cuts:
        keys = [k for k, blk in ref["by_basis"].items() if blk.get("engaged_cut", "2.0s") == cut]
        slot = ref["consensus"].setdefault(cut, {})
        for an in animals:
            got = {k: ref["by_basis"][k]["animals"][an].get("interpretation")
                   for k in keys if an in ref["by_basis"][k].get("animals", {})}
            cls = {v.split(":")[0].replace("consistent with ", "").strip()
                   for v in got.values() if v}
            slot[an] = (got[list(got)[0]] if len(cls) == 1 else
                        {"DISAGREEMENT": got} if cls else None)
    for an in animals:
        got = {b: blk["animals"][an].get("interpretation")
               for b, blk in ref["by_basis"].items() if an in blk.get("animals", {})}
        del got

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
        d = ref.get("direction_consistency") or {}
        if d.get("statement"):
            print(f"  DIRECTION (threshold-free): {d['statement']}", flush=True)
        for cut, per in (ref.get("consensus") or {}).items():
            print(f"  [engaged cut = {cut}]", flush=True)
            for an, v in (per or {}).items():
                if isinstance(v, str):
                    print(f"    {an}: {v}", flush=True)
                else:
                    got = (v or {}).get("DISAGREEMENT", {})
                    print(f"    {an}: BASES DISAGREE -- " + "; ".join(
                        f"{k}={x.split(':')[0]}" for k, x in got.items()), flush=True)
        print(f"rewrote {pth}", flush=True)
        return
    dates = config.expand_dates(a.dates, width=4) if a.dates else None
    build_reference(animals=a.animals, dates=dates, bases=a.bases, out=a.out, n_perm=a.n_perm)


if __name__ == "__main__":
    main()
