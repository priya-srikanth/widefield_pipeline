"""Post-stroke vs pre-stroke, POSITION-MATCHED, in the four readout conditions.

WHY MATCHING IS THE WHOLE ANALYSIS. On 8/17 both animals stopped attempting the far positions --
PS94 has ZERO engaged trials at far_center and far_R (from ~106 each pre-stroke), PS95 has 10 and 1.
So a frozen 6-position model is being scored on 4 positions, and the raw accuracy drop (PS94 0.699
vs a 0.937 pre-stroke band) is mostly TRIAL COMPOSITION, not coding. Every comparison here is
restricted to positions the animal still attempts post-stroke, in BOTH phases, so the question is
"did coding change where behaviour survived" rather than "did behaviour change".

Three questions (Priya, 2026-08-18):

  1. Do post-stroke PRE-CUE no-lick trials look like pre-stroke LICKING or pre-stroke NON-LICKING
     trials? Tested directly: train a discriminator on pre-stroke engaged-vs-undetected pre-cue
     patterns, POSITION-BALANCED so it cannot simply learn "far", then apply it to post-stroke
     undetected trials and report which side they fall on.
  2. How similar are the POST-CUE models? Per-position mean-pattern correlation between phases,
     plus the frozen decoder's per-position recall.
  3. At the PRESERVED positions, does POST-LICK look the same?
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config, nolick_analysis as na
from wfield_local.locanmf_frozen_decoder import _pipe, pool_sessions
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

MIN_POST = 20          # a position needs this many post-stroke engaged trials to be "preserved"


def _pooled(animal, align, source="roi"):
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    if not pre or not post:
        return None
    p = pool_sessions(pre + post, source=source, align=align, post_s=2.0)
    if p is None:
        return None
    XE, YE, GE, BE, XU, YU, kept, common, GU = p
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    post_i = {i for i, l in enumerate(kept) if l not in set(pre)}
    return dict(XE=XE, YE=YE, GE=GE, XU=XU, YU=YU.astype(int), GU=GU, kept=kept,
                pre_i=pre_i, post_i=post_i)


def preserved_positions(d):
    """Positions the animal still attempts post-stroke -- the only ones a comparison can use."""
    m = np.isin(d["GE"], list(d["post_i"]))
    return [c for c in DISPLAY_ORDER if int((d["YE"][m] == c).sum()) >= MIN_POST]


def decode_matched(d, keep):
    """Frozen pre-stroke decoder, scored on the SAME positions in both phases."""
    kp = np.isin(d["YE"], keep)
    tr = np.isin(d["GE"], list(d["pre_i"])) & kp
    te = np.isin(d["GE"], list(d["post_i"])) & kp
    if tr.sum() < 50 or te.sum() < 20:
        return None
    clf = _pipe().fit(d["XE"][tr], d["YE"][tr])
    # pre-stroke baseline under the SAME restriction, leave-one-session-out
    pre_pred = cross_val_predict(_pipe(), d["XE"][tr], d["YE"][tr],
                                 cv=LeaveOneGroupOut(), groups=d["GE"][tr])
    per = {}
    for i in sorted(d["pre_i"]):
        m = d["GE"][tr] == i
        if m.sum():
            per[d["kept"][i]] = float(accuracy_score(d["YE"][tr][m], pre_pred[m]))
    post_pred = clf.predict(d["XE"][te])
    out = na.evaluate_arm(d["YE"][te], post_pred, n_perm=1000, labels=keep)
    out["accuracy"] = float(accuracy_score(d["YE"][te], post_pred))
    v = np.array(list(per.values()), float)
    out["pre_band"] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                       "n_sessions": len(per)}
    out["below_every_pre_session"] = bool(out["accuracy"] < out["pre_band"]["min"])
    out["pre_recall"] = na.per_position_recall(d["YE"][tr], pre_pred, labels=keep)
    return out


def looks_like_which(d, keep, seed=0):
    """Do POST-stroke no-lick trials resemble PRE-stroke LICKING or PRE-stroke NON-LICKING trials?

    A discriminator trained on pre-stroke engaged vs undetected would otherwise learn POSITION --
    undetected trials are overwhelmingly far pre-stroke, and post-stroke no-lick trials are far too,
    so an unmatched classifier would answer "far" and look like an answer. The two training classes
    are therefore position-BALANCED: within each kept position, both classes are subsampled to the
    same count, so position carries no information about the label.
    """
    rng = np.random.RandomState(seed)
    pre_e = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], keep)
    pre_u = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], keep)
    post_u = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], keep)
    if pre_u.sum() < 30 or post_u.sum() < 20:
        return {"note": "too few no-lick trials to discriminate",
                "n_pre_undetected": int(pre_u.sum()), "n_post_undetected": int(post_u.sum())}
    Xe, ye = d["XE"][pre_e], d["YE"][pre_e]
    Xu, yu = d["XU"][pre_u], d["YU"][pre_u]
    xs, lab = [], []
    for c in keep:                      # position-balanced within each position
        ie, iu = np.flatnonzero(ye == c), np.flatnonzero(yu == c)
        n = min(len(ie), len(iu))
        if n < 5:
            continue
        xs.append(Xe[rng.choice(ie, n, replace=False)]); lab.append(np.ones(n))
        xs.append(Xu[rng.choice(iu, n, replace=False)]); lab.append(np.zeros(n))
    if not xs:
        return {"note": "no position had >=5 of both classes"}
    X, y = np.vstack(xs), np.concatenate(lab)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X, y)
    # how well does it separate the two PRE-stroke states at all? without this the post-stroke
    # answer is uninterpretable -- a coin-flip discriminator says nothing.
    sep = float(accuracy_score(y, cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)), X, y, cv=5)))
    p_post = float(clf.predict(d["XU"][post_u]).mean())

    # THE CONTROL THIS TEST IS WORTHLESS WITHOUT. The boundary is trained entirely on PRE-stroke
    # data, so post-stroke trials are out of distribution: any global post-lesion shift could push
    # them to one side whatever their trial state. Applying the SAME boundary to post-stroke ENGAGED
    # trials says whether it still tracks licking after the lesion. If engaged and undetected land at
    # the same rate, the classifier is reporting "post-stroke", not "licking", and the headline
    # number means nothing.
    post_e = np.isin(d["GE"], list(d["post_i"])) & np.isin(d["YE"], keep)
    p_post_eng = float(clf.predict(d["XE"][post_e]).mean()) if post_e.sum() >= 20 else float("nan")
    pre_e_held = float(clf.predict(Xe).mean())          # pre-stroke engaged, for scale
    pre_u_held = float(clf.predict(Xu).mean())          # pre-stroke undetected, for scale
    gap = p_post_eng - p_post if np.isfinite(p_post_eng) else float("nan")
    return {"n_train_per_class": int((y == 1).sum()), "n_post_undetected": int(post_u.sum()),
            "pre_separability_cv": sep,
            "post_undetected_frac_classified_ENGAGED_like": p_post,
            "CONTROL_post_engaged_frac_engaged_like": p_post_eng,
            "n_post_engaged": int(post_e.sum()),
            "scale_pre_engaged": pre_e_held, "scale_pre_undetected": pre_u_held,
            "engaged_minus_undetected_post": gap,
            "boundary_still_discriminates_post": bool(np.isfinite(gap) and gap > 0.10),
            "reads_as": ("engaged-like (licking)" if p_post > 0.5 else "undetected-like (non-licking)")}


def pattern_similarity(d, keep):
    """Per-position mean-pattern correlation between phases -- is the code the SAME code?"""
    out = {}
    for c in keep:
        a = d["XE"][np.isin(d["GE"], list(d["pre_i"])) & (d["YE"] == c)]
        b = d["XE"][np.isin(d["GE"], list(d["post_i"])) & (d["YE"] == c)]
        if len(a) < 5 or len(b) < 5:
            continue
        ma, mb = a.mean(0), b.mean(0)
        out[POSITION_NAMES[c]] = {"r": float(np.corrcoef(ma, mb)[0, 1]),
                                  "n_pre": int(len(a)), "n_post": int(len(b))}
    return out


def precue_lick_mask(s, args):
    """Mask over the trials `_trial_features` KEEPS, True where the fixed pre-cue window had licks.

    The caller decides which side it wants; this only reports which trials had licks.

    Built by replaying the same keep-logic with the same helpers, then ASSERTED against the feature
    matrix length by the caller. Replicating a trial filter is how two code paths silently come to
    disagree (bugs 15-17 were all versions of that), so the assertion is the point: if this drifts
    from `_trial_features`, the run stops instead of pairing the wrong mask with the wrong trials.
    """
    import numpy as _np
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.locanmf_position_decoder import _load_cue_events, precue_window_start
    from wfield_local.plot_lick_aligned_averages import _load_daq_events

    cue = _load_cue_events(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, _ = _frames(s, cue, lk)
    codes = classify_cues_with_backup(s, cue, verbose=False)
    ls = _np.sort(_np.asarray(lick_f))
    post_n = int(round(args.post_s * args.fs))
    T = None
    keep, has = [], []
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        c0 = int(cue_f[k])
        ref0 = precue_window_start(c0, _np.nan, ls, post_n, lickfree=False)
        if ref0 is None or ref0 < 0:
            continue
        keep.append(k)
        has.append(bool(_np.any((ls >= ref0) & (ls < ref0 + post_n))))
    return _np.array(has, bool), _np.array(keep, int)


def crossed_confusion(d, labels=DISPLAY_ORDER):
    """Full confusion of the FROZEN pre-stroke decoder applied to post-stroke trials.

    Scalar accuracy discards what a lateralised lesion should produce: WHERE the errors go. A
    contralesional deficit that shifts far_R trials onto far_L looks identical, in accuracy, to one
    that scatters them. The 6x6 distinguishes those, and its diagonal is the per-position recall
    table, so the two are one object rather than two.

    Rows are TRUE position, columns PREDICTED, row-normalised. Rows with no post-stroke trials are
    returned as NaN rather than zeros -- PS94 has none at far_center or far_R, and a row of zeros
    would read as "always wrong" instead of "never attempted".
    """
    tr = np.isin(d["GE"], list(d["pre_i"]))
    te = np.isin(d["GE"], list(d["post_i"]))
    clf = _pipe().fit(d["XE"][tr], d["YE"][tr])
    out = {}
    for phase, m, pred in (("pre", tr, cross_val_predict(_pipe(), d["XE"][tr], d["YE"][tr],
                                                         cv=LeaveOneGroupOut(), groups=d["GE"][tr])),
                           ("post", te, None)):
        p = pred if pred is not None else clf.predict(d["XE"][m])
        y = d["YE"][m]
        M = np.full((len(labels), len(labels)), np.nan)
        n = []
        for i, c in enumerate(labels):
            sel = y == c
            n.append(int(sel.sum()))
            if sel.sum():
                for j, c2 in enumerate(labels):
                    M[i, j] = float((p[sel] == c2).mean())
        out[phase] = {"matrix": M.tolist(), "n_per_true_position": n,
                      "positions": [POSITION_NAMES[c] for c in labels]}
    return out


def undetected_state_split(d, keep, seed=0):
    """THE ARBITRATING TEST: does the pre-stroke engaged/undetected boundary still work post-stroke?

    Priya (2026-08-18): post-stroke undetected trials classify as engaged-like (0.71, 0.86) and sit
    far from pre-stroke undetected (0.085, 0.147), so the failure to separate post-stroke engaged
    from post-stroke undetected may not mean the boundary went blind -- it may mean there is nothing
    left to separate, because the animal IS in an engaged-like state on trials it fails to execute.

    That reading and "a global post-stroke shift pushes everything to the engaged side" are
    distinguishable. Split the POST-stroke undetected trials by the engagement gate and apply the
    same boundary:

        disengaged trials classify LOW  -> the boundary still tracks state; the WORKING trials being
                                           engaged-like is a real execution-failure signature
        disengaged trials classify HIGH -> everything post-stroke reads engaged-like regardless, and
                                           the headline is a global shift

    Returns None when the post-stroke session has too few disengaged trials to decide, which is a
    real possibility and must not be reported as either answer.
    """
    from wfield_local import nolick_decoder as nd
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    base = looks_like_which(d, keep, seed=seed)
    if "post_undetected_frac_classified_ENGAGED_like" not in base:
        return base

    # rebuild the boundary exactly as looks_like_which does, then score the two post subsets
    rng = np.random.RandomState(seed)
    pre_e = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], keep)
    pre_u = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], keep)
    Xe, ye = d["XE"][pre_e], d["YE"][pre_e]
    Xu, yu = d["XU"][pre_u], d["YU"][pre_u]
    xs, lab = [], []
    for c in keep:
        ie, iu = np.flatnonzero(ye == c), np.flatnonzero(yu == c)
        n = min(len(ie), len(iu))
        if n < 5:
            continue
        xs.append(Xe[rng.choice(ie, n, replace=False)]); lab.append(np.ones(n))
        xs.append(Xu[rng.choice(iu, n, replace=False)]); lab.append(np.zeros(n))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(
        np.vstack(xs), np.concatenate(lab))

    # engagement state for each post-stroke undetected trial, from the pipeline's own gate
    out = dict(base)
    for i in sorted(d["post_i"]):
        lab_i = d["kept"][i]
        s = next((x for x in SESSIONS if x["label"] == lab_i), None)
        if s is None:
            continue
        args = nd._args(align="precue"); args.max_rt = 2.0
        codes, cat, blk, rt_s, cue_f, sess_eng = nd.categorize(s, args)
        und = np.flatnonzero((cat == "undetected") & np.isin(codes, keep))
        if not und.size:
            continue
        m = (d["GU"] == i) & np.isin(d["YU"], keep)
        if m.sum() != und.size:
            out[f"{lab_i}_split"] = {"note": f"trial-count mismatch ({int(m.sum())} pooled vs "
                                             f"{und.size} categorised); split not attempted"}
            continue
        eng_state = sess_eng[und]
        p = clf.predict(d["XU"][m])
        res = {"n_working": int(eng_state.sum()), "n_disengaged": int((~eng_state).sum())}
        if eng_state.sum() >= 10:
            res["working_frac_engaged_like"] = float(p[eng_state].mean())
        if (~eng_state).sum() >= 10:
            res["disengaged_frac_engaged_like"] = float(p[~eng_state].mean())
        if "working_frac_engaged_like" in res and "disengaged_frac_engaged_like" in res:
            gap = res["working_frac_engaged_like"] - res["disengaged_frac_engaged_like"]
            res["working_minus_disengaged"] = gap
            res["verdict"] = ("boundary STILL tracks state -> engaged-like WORKING trials are a real "
                              "execution-failure signature" if gap > 0.10 else
                              "no separation -> consistent with a global post-stroke shift")
        else:
            res["verdict"] = "too few trials in one state to decide"
        out[f"{lab_i}_split"] = res
    return out


def poststroke_engagement(s, reference_positions, window=15, min_rate=0.5):
    """Engagement judged ONLY at positions the animal can still reach (Priya, 2026-08-18).

    THE PRE-STROKE GATE IS INVALID AFTER A LESION. `flag_engagement` calls an animal disengaged when
    its trailing response rate collapses -- and post-stroke that collapses BECAUSE the animal cannot
    reach the far positions. It would therefore label motor failure as disengagement, i.e. label the
    effect being measured as the confound, and every "no plan formed" conclusion downstream would be
    circular.

    Judged instead at the PRESERVED positions (PS94/PS95 8/17: close_L, close_center, close_R,
    far_L). A miss there is genuine disengagement; a miss at an impaired position says nothing about
    motivation. Which positions count is empirical per animal and per session, so it is passed in
    rather than hardcoded.

    Returns (engaged_bool per trial, info). Trials at non-reference positions inherit the state of
    the reference-position trials around them, since that is what the estimate is FOR.
    """
    from wfield_local import config, nolick_decoder as nd

    args = nd._args(align="cue")
    args.max_rt = 2.0
    codes, cat, blk, rt_s, cue_f, _pre_gate = nd.categorize(s, args)
    codes = np.asarray(codes)
    responded = np.array([c in ("engaged", "late_rewarded") for c in cat], bool)
    is_ref = np.isin(codes, list(reference_positions))

    eng = np.ones(codes.size, bool)
    ref_idx = np.flatnonzero(is_ref & (codes >= 0))
    if ref_idx.size < window:
        return eng, {"note": "too few reference-position trials to gate", "n_ref": int(ref_idx.size)}
    # rolling response rate over REFERENCE trials only, then mapped back onto the full trial order
    r = responded[ref_idx].astype(float)
    roll = np.array([r[max(0, i - window + 1):i + 1].mean() for i in range(r.size)])
    low = roll < min_rate
    for j, k in enumerate(ref_idx):
        eng[k] = not low[j]
    # non-reference trials take the state of the nearest preceding reference trial
    last = True
    for k in range(codes.size):
        if is_ref[k] and codes[k] >= 0:
            last = eng[k]
        else:
            eng[k] = last
    return eng, {"n_ref": int(ref_idx.size), "ref_response_rate": float(responded[ref_idx].mean()),
                 "frac_disengaged": float((~eng).mean()),
                 "reference_positions": sorted(int(c) for c in reference_positions)}
