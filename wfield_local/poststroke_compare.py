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
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config, nolick_analysis as na
from wfield_local.locanmf_frozen_decoder import _pipe, pool_sessions
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

MIN_POST = 20          # a position needs this many post-stroke engaged trials to be "preserved"

#: Positions used to judge post-stroke ENGAGEMENT: close_L and close_center only (Priya,
#: 2026-08-18). Deliberately NOT every position that survives on 8/17. close_R is CONTRALESIONAL
#: (both lesions are left-sided) and far_L carries the distance effect, so either could be impaired
#: -- and a reference position that is itself impaired turns the gate back into the thing it exists
#: to avoid, labelling motor failure as disengagement. These two are the least likely to be affected
#: by a left VLS lesion, so a miss there is the strongest available evidence of genuine
#: disengagement. Empirical: revisit if an animal turns out to miss them.
REFERENCE_POSITIONS = [c for c in DISPLAY_ORDER
                       if POSITION_NAMES[c] in ("close_L", "close_center")]


def excluded_labels(animal):
    """Sessions for `animal` whose phase is neither pre nor post.

    Currently PS92_0817 / PS93_0817: the 8/16 lesion produced no deficit and was redone AFTER that
    session, so it is baseline-like but not a clean baseline, and post-lesion but not post-deficit.
    They are analysable and must never be pooled.
    """
    out = []
    for s in config.load_sessions():
        lab = s["label"]
        if not lab.startswith(animal + "_"):
            continue
        if config.session_phase(animal, lab.split("_")[-1]) == "excluded":
            out.append(lab)
    return sorted(out)


def _pooled(animal, align, source="roi", post_labels=None):
    """Pre-stroke pool + a comparison session, aligned and z-scored together.

    `post_labels` overrides the comparison arm. Leave it None for real post-stroke work -- the default
    is `phase_labels("post")`, which is the only sanctioned pool. Pass it ONLY to analyse sessions that
    resolve to the 'excluded' phase, and label the output accordingly; `excluded_labels()` supplies
    them. Results built this way must not enter any pooled post-stroke summary (Priya, 2026-08-18).
    """
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    if post_labels is None:
        post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    else:
        post = [l for l in post_labels if l.startswith(animal)]
    if not pre or not post:
        return None
    p = pool_sessions(pre + post, source=source, align=align, post_s=2.0)
    if p is None:
        return None
    XE, YE, GE, BE, XU, YU, kept, common, GU = p
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    post_i = {i for i, l in enumerate(kept) if l not in set(pre)}
    return dict(XE=XE, YE=YE, GE=GE, BE=BE, XU=XU, YU=YU.astype(int), GU=GU, kept=kept,
                pre_i=pre_i, post_i=post_i)


def preserved_positions(d, session=None, combine="intersection"):
    """Positions the animal still attempts post-stroke -- the only ones a comparison can use.

    PER SESSION, because "still attempts" is a behavioural state that changes day to day (Priya,
    2026-08-19). Pass `session` (an index into d["kept"]) for one session's set.

    WITHOUT `session`, sessions are combined by INTERSECTION: positions attempted on EVERY
    post-stroke day. An earlier version pooled the trial counts across sessions, which is the UNION,
    and it produced a concrete error -- PS95 attempted far_center/far_R on 8/18 (99 and 84 trials) but
    not on 8/17 (10 and 1), so the pooled set was six positions and PS95's 8/17 numbers were computed
    over a position with ONE engaged trial. It also silently moved that result's chance level from
    0.25 to 0.167 when 8/18 was registered, with nothing about 8/17 having changed.

    A pooled statistic has to be defensible for every session inside it, which is what the
    intersection guarantees and the union cannot.
    """
    def _for(idx):
        m = np.isin(d["GE"], list(idx))
        return {c for c in DISPLAY_ORDER if int((d["YE"][m] == c).sum()) >= MIN_POST}

    if session is not None:
        return [c for c in DISPLAY_ORDER if c in _for([session])]
    per = [_for([i]) for i in sorted(d["post_i"])]
    if not per:
        return []
    if combine == "union":                       # never the default; here only to be explicit
        keep = set.union(*per)
    elif combine == "pooled":                    # the old behaviour, kept so it can be compared
        keep = _for(list(d["post_i"]))
    else:
        keep = set.intersection(*per)
    return [c for c in DISPLAY_ORDER if c in keep]


def preserved_positions_by_session(d):
    """{session label -> preserved positions} for every post-stroke session, for reporting."""
    return {d["kept"][i]: preserved_positions(d, session=i) for i in sorted(d["post_i"])}


def decode_matched(d, keep, post_all_trials=True):
    """Frozen pre-stroke decoder, scored on the SAME positions in both phases.

    POST ARM USES ALL TRIALS by default (Priya, 2026-08-18/19): the missing licks ARE the phenotype,
    and with the no-lick trials included every position has data, so the arm is scored over all SIX
    positions rather than the lick-defined `keep` set. An engaged-only version excludes the positions
    the lesion abolished -- PS94 has ZERO engaged and ~105 no-lick trials at far_center and far_R --
    so it can only ever describe the part of the phenotype that still works.

    `keep` still governs the LICK-ONLY arm, where it is forced: a position with no engaged trials
    cannot be decoded from engaged trials. The two arms therefore differ in chance level and callers
    must say which is shown.
    """
    pos = list(DISPLAY_ORDER) if post_all_trials else list(keep)
    kp = np.isin(d["YE"], pos)
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
    if post_all_trials:
        teu = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], pos)
        Xpost = np.vstack([d["XE"][te], d["XU"][teu]])
        ypost = np.concatenate([d["YE"][te], d["YU"][teu]])
    else:
        Xpost, ypost = d["XE"][te], d["YE"][te]
    post_pred = clf.predict(Xpost)
    out = na.evaluate_arm(ypost, post_pred, n_perm=1000, labels=pos)
    out["accuracy"] = float(accuracy_score(ypost, post_pred))
    out["post_arm"] = "ALL trials" if post_all_trials else "lick-only"
    out["positions_scored"] = [POSITION_NAMES[c] for c in pos]
    out["n_post"] = int(len(ypost))
    v = np.array(list(per.values()), float)
    out["pre_band"] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                       "n_sessions": len(per)}
    out["below_every_pre_session"] = bool(out["accuracy"] < out["pre_band"]["min"])
    out["pre_recall"] = na.per_position_recall(d["YE"][tr], pre_pred, labels=pos)
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
    Xe, ye, ge = d["XE"][pre_e], d["YE"][pre_e], d["GE"][pre_e]
    Xu, yu, gu = d["XU"][pre_u], d["YU"][pre_u], d["GU"][pre_u]
    xs, lab, grps = [], [], []
    for c in keep:                      # position-balanced within each position
        ie, iu = np.flatnonzero(ye == c), np.flatnonzero(yu == c)
        n = min(len(ie), len(iu))
        if n < 5:
            continue
        se = rng.choice(ie, n, replace=False)
        su = rng.choice(iu, n, replace=False)
        xs.append(Xe[se]); lab.append(np.ones(n)); grps.append(ge[se])
        xs.append(Xu[su]); lab.append(np.zeros(n)); grps.append(gu[su])
    if not xs:
        return {"note": "no position had >=5 of both classes"}
    X, y, grp = np.vstack(xs), np.concatenate(lab), np.concatenate(grps)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X, y)
    # how well does it separate the two PRE-stroke states at all? without this the post-stroke
    # answer is uninterpretable -- a coin-flip discriminator says nothing.
    # GROUPED by session. An ungrouped cv=5 puts trials from one session on both sides of a fold, and
    # session identity is exactly what a whole-brain feature vector can memorise; every other CV in
    # this pipeline groups for that reason. Falls back to ungrouped only if the balanced training set
    # somehow spans fewer than two sessions, and says so in the output.
    def _mk():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))

    # BOTH estimates, because they answer different questions and the grouped one can be too coarse
    # here: GroupKFold holds out a whole SESSION, and with ~11 pre-stroke sessions a fold can lose an
    # entire position's worth of one class. Ungrouped = "can these two states be separated at all";
    # grouped = "can they be separated in a session the model has never seen", which is the
    # conservative number and the one to quote when the answer must generalise across days.
    sep_trial = float(accuracy_score(y, cross_val_predict(_mk(), X, y, cv=5)))
    ngrp = len(np.unique(grp))
    if ngrp >= 2:
        sep_session = float(accuracy_score(y, cross_val_predict(
            _mk(), X, y, cv=GroupKFold(n_splits=min(5, ngrp)), groups=grp)))
        sep_grouped = True
    else:
        sep_session = float("nan")
        sep_grouped = False
    sep = sep_session if sep_grouped else sep_trial
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

    # Interval on the CONTROL gap. It decides whether the headline number may be read at all, so a
    # point estimate either side of a 0.10 threshold was never enough. Stratified bootstrap: resample
    # the post-stroke engaged and no-lick arms independently, since they are separate samples.
    gap_ci = [float("nan"), float("nan")]
    if np.isfinite(p_post_eng):
        pe = clf.predict(d["XE"][post_e]).astype(float)
        pu = clf.predict(d["XU"][post_u]).astype(float)
        boots = [pe[rng.randint(0, len(pe), len(pe))].mean()
                 - pu[rng.randint(0, len(pu), len(pu))].mean() for _ in range(2000)]
        gap_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    return {"n_train_per_class": int((y == 1).sum()), "n_post_undetected": int(post_u.sum()),
            "pre_separability_cv": sep,
            "pre_separability_cv_grouped_by_session": sep_grouped,
            "pre_separability_cv_session_grouped": sep_session,
            "pre_separability_cv_trialwise": sep_trial,
            "n_train_sessions": int(ngrp),
            "control_gap_ci95": gap_ci,
            "post_undetected_frac_classified_ENGAGED_like": p_post,
            "CONTROL_post_engaged_frac_engaged_like": p_post_eng,
            "n_post_engaged": int(post_e.sum()),
            "scale_pre_engaged": pre_e_held, "scale_pre_undetected": pre_u_held,
            "engaged_minus_undetected_post": gap,
            # the WHOLE interval must clear zero: a positive point estimate whose interval spans
            # zero is not a control that passed, it is a control that was not measured precisely
            # enough to say either way
            "boundary_still_discriminates_post": bool(
                np.isfinite(gap) and gap > 0.10 and gap_ci[0] > 0),
            "reads_as": ("engaged-like (licking)" if p_post > 0.5 else "undetected-like (non-licking)")}



def fits_engaged_distribution(d, keep, seed=0, n_boot=2000):
    """Does the POST-stroke no-lick session fall inside the PRE-stroke ENGAGED distribution?

    See the module note on `looks_like_which`: that function's control assumes post-stroke engaged and
    no-lick trials should separate, which is precisely what the execution-failure hypothesis denies.
    This one makes no such assumption. The reference distributions come from PRE-stroke sessions, where
    the truth is known, and the post-stroke session is placed against them.

    Returns per-session reference values, their ranges, and where the post-stroke value sits.
    """
    rng = np.random.RandomState(seed)
    pre_e_all = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], keep)
    pre_u_all = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], keep)

    def balanced_fit(exclude_session=None):
        """Position-balanced engaged-vs-no-lick discriminator, optionally holding out one session."""
        e = pre_e_all & (d["GE"] != exclude_session if exclude_session is not None else True)
        u = pre_u_all & (d["GU"] != exclude_session if exclude_session is not None else True)
        Xe, ye = d["XE"][e], d["YE"][e]
        Xu, yu = d["XU"][u], d["YU"][u]
        xs, lab = [], []
        for c in keep:
            ie, iu = np.flatnonzero(ye == c), np.flatnonzero(yu == c)
            n = min(len(ie), len(iu))
            if n < 5:
                continue
            xs.append(Xe[rng.choice(ie, n, replace=False)]); lab.append(np.ones(n))
            xs.append(Xu[rng.choice(iu, n, replace=False)]); lab.append(np.zeros(n))
        if not xs:
            return None
        X, y = np.vstack(xs), np.concatenate(lab)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X, y)

    ref = {"engaged": {}, "nolick": {}}
    for i in sorted(d["pre_i"]):
        clf = balanced_fit(exclude_session=i)
        if clf is None:
            continue
        me = pre_e_all & (d["GE"] == i)
        mu = pre_u_all & (d["GU"] == i)
        if me.sum() >= 20:
            ref["engaged"][d["kept"][i]] = float(clf.predict(d["XE"][me]).mean())
        if mu.sum() >= 10:
            ref["nolick"][d["kept"][i]] = float(clf.predict(d["XU"][mu]).mean())
    if len(ref["engaged"]) < 3 or len(ref["nolick"]) < 3:
        return {"note": "too few pre-stroke sessions to build a reference distribution",
                "n_engaged_sessions": len(ref["engaged"]), "n_nolick_sessions": len(ref["nolick"])}

    clf_all = balanced_fit()
    post_u = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], keep)
    post_e = np.isin(d["GE"], list(d["post_i"])) & np.isin(d["YE"], keep)
    if post_u.sum() < 10 or clf_all is None:
        return {"note": "too few post-stroke no-lick trials", "n_post_nolick": int(post_u.sum())}
    pred_u = clf_all.predict(d["XU"][post_u]).astype(float)
    P = float(pred_u.mean())

    out = {"post_value": P, "n_post_nolick": int(post_u.sum()),
           "post_engaged_value": float(clf_all.predict(d["XE"][post_e]).mean())
                                 if post_e.sum() >= 20 else float("nan"),
           "reference_engaged_per_session": ref["engaged"],
           "reference_nolick_per_session": ref["nolick"]}
    # a bootstrap interval on the post-stroke value itself, so the comparison is interval-to-interval
    boots = [pred_u[rng.randint(0, len(pred_u), len(pred_u))].mean() for _ in range(n_boot)]
    out["post_value_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    for arm in ("engaged", "nolick"):
        v = np.array(list(ref[arm].values()), float)
        lo, hi = float(v.min()), float(v.max())
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        out[arm] = {"n_sessions": len(v), "mean": mu, "sd": sd, "min": lo, "max": hi,
                    "z_of_post": (P - mu) / sd if sd else float("nan"),
                    "post_inside_range": bool(lo <= P <= hi),
                    # fraction of pre-stroke sessions this post value exceeds
                    "percentile_of_post": float((v < P).mean())}
    ie, inl = out["engaged"]["post_inside_range"], out["nolick"]["post_inside_range"]
    e_lo, e_hi = out["engaged"]["min"], out["engaged"]["max"]
    n_lo, n_hi = out["nolick"]["min"], out["nolick"]["max"]
    # Where P sits BETWEEN the two references is a real and common state, and the first version of
    # this verdict collapsed it into "OFF-SCALE", which reads as a failed measurement. PS94 pre-cue
    # is exactly this: 0.662, above every pre-stroke no-lick session and below every engaged one.
    between = (not ie) and (not inl) and (n_hi < P < e_lo or e_hi < P < n_lo)
    if between:
        span = abs(e_lo - n_hi) if n_hi < e_lo else abs(n_lo - e_hi)
        frac = (P - n_hi) / span if (span and n_hi < e_lo) else float("nan")
        out["fraction_of_gap_toward_engaged"] = float(frac)
    out["verdict"] = (
        "post-stroke NO-LICK trials fall inside the pre-stroke ENGAGED distribution and OUTSIDE the "
        "pre-stroke no-lick one -> indistinguishable from successful trials: plan intact, execution "
        "failed" if (ie and not inl) else
        "post-stroke no-lick trials look like ordinary pre-stroke FAILURES" if (inl and not ie) else
        "AMBIGUOUS: the post value sits inside both reference distributions -- they overlap too much "
        "to separate" if (ie and inl) else
        (f"INTERMEDIATE: above every pre-stroke no-lick session and below every engaged one "
         f"({out.get('fraction_of_gap_toward_engaged', float('nan')):.0%} of the way toward engaged) "
         f"-> shifted toward successful trials but not indistinguishable from them") if between else
        "OFF-SCALE: outside BOTH pre-stroke distributions and outside the gap between them, so "
        "neither reference describes it")
    return out

def pattern_similarity(d, keep, post_all_trials=True):
    """POST arm uses ALL trials by default and is scored over every position -- see decode_matched.

    Per-position mean-pattern correlation between phases -- is the code the SAME code?"""
    # ALL six positions when the post arm is all-trials: restricting to `keep` would silently drop
    # far_center and far_R for PS94, which are the positions this comparison most needs to describe.
    pos = list(DISPLAY_ORDER) if post_all_trials else list(keep)
    out = {}
    for c in pos:
        a = d["XE"][np.isin(d["GE"], list(d["pre_i"])) & (d["YE"] == c)]
        b = d["XE"][np.isin(d["GE"], list(d["post_i"])) & (d["YE"] == c)]
        if post_all_trials:
            bu = d["XU"][np.isin(d["GU"], list(d["post_i"])) & (d["YU"] == c)]
            if len(bu):
                b = np.vstack([b, bu]) if len(b) else bu
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


def crossed_confusion(d, labels=DISPLAY_ORDER, post_all_trials=False):
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

    # PRE: engaged only, leave-one-session-out -- the reference for a code with a successful movement.
    pre_y = d["YE"][tr]
    pre_p = cross_val_predict(_pipe(), d["XE"][tr], d["YE"][tr], cv=LeaveOneGroupOut(),
                              groups=d["GE"][tr])
    pre_nolick = np.zeros(len(pre_y), bool)

    # POST: engaged, or engaged + no-lick when post_all_trials. The no-lick trials are the ONLY ones
    # that exist at an abandoned position, so without them those rows cannot be filled at all.
    if post_all_trials:
        teu = np.isin(d["GU"], list(d["post_i"]))
        post_y = np.concatenate([d["YE"][te], d["YU"][teu]])
        post_X = np.vstack([d["XE"][te], d["XU"][teu]])
        post_nolick = np.concatenate([np.zeros(int(te.sum()), bool), np.ones(int(teu.sum()), bool)])
    else:
        post_y, post_X = d["YE"][te], d["XE"][te]
        post_nolick = np.zeros(len(post_y), bool)
    post_p = clf.predict(post_X)

    out = {"post_arm": "ALL trials (engaged + no-lick)" if post_all_trials else "engaged only",
           "pre_arm": "engaged only, leave-one-session-out"}
    for phase, y, p, nl in (("pre", pre_y, pre_p, pre_nolick),
                            ("post", post_y, post_p, post_nolick)):
        M = np.full((len(labels), len(labels)), np.nan)
        n, n_nolick = [], []
        for i, c in enumerate(labels):
            sel = y == c
            n.append(int(sel.sum()))
            # how much of this row is carried by trials with no detected lick: a row that is 100%
            # no-lick is a different kind of evidence from one that is 10%, and the matrix cannot
            # show that by itself
            n_nolick.append(int(nl[sel].sum()))
            if sel.sum():
                for j, c2 in enumerate(labels):
                    M[i, j] = float((p[sel] == c2).mean())
        out[phase] = {"matrix": M.tolist(), "n_per_true_position": n,
                      "n_nolick_per_true_position": n_nolick,
                      "positions": [POSITION_NAMES[c] for c in labels]}
    return out


#: POST-STROKE ENGAGEMENT FILTERING IS RETIRED (Priya, 2026-08-18).
#:
#: There is no valid post-stroke construction of "disengaged", so no analysis may split on one.
#:
#: The pre-stroke gate was invalid for the obvious reason (it reads motor failure as lost motivation
#: -- it called 59% of PS94 8/17 disengaged against 6.8% for a reference-position gate). The
#: reference-position gate that replaced it is better but ALSO not validated: its 29 "disengaged"
#: PS94 trials are undetected trials falling where the trailing response rate at close_L/close_center
#: dipped below 0.5 over 15 reference trials, and a short run of MOTOR failures produces that dip
#: just as readily as a motivational lapse. Nothing in the spout data distinguishes them.
#:
#: And the construction has no general form: in a severe stroke EVERY position may be impaired, so
#: there is no spared reference to anchor engagement on and spout contact cannot define it at all.
#:
#: Consequence for results already computed: `undetected_state_split`'s comparison class was never
#: established, so its output (PS94 working-minus-disengaged -0.060) is UNINTERPRETABLE, not
#: negative. It must not be quoted as evidence for a global post-stroke shift -- which is how I first
#: read it. PS95 returned UNDECIDABLE (0 disengaged) and that reading was right for the wrong reason.
#:
#: Post-stroke analyses therefore use ALL trials (nolick_analysis.SANCTIONED_MISMATCHES). The
#: question the split was meant to answer -- execution failure vs disengagement -- is better asked
#: WITHIN the post-stroke session by contrasting undetected trials at IMPAIRED vs PRESERVED
#: positions, which needs no engagement label. DLC tongue-protrusion data settles it directly.
POSTSTROKE_ENGAGEMENT_FILTERING = False


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

    RETAINED AS A DESCRIPTIVE STATISTIC ONLY -- see POSTSTROKE_ENGAGEMENT_FILTERING above. Reporting
    "the animal responded on 89% (PS94) / 97% (PS95) of trials at spared positions" is sound and worth
    showing. SPLITTING trials on it is not, because a local dip in that rate is indistinguishable from
    a run of motor failures. Callers that filter on the returned mask are the error this guards.

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


def nolick_state_vector(kept, args, reference_positions=None):
    """Engagement state + lick category for every trial in the pooled NO-LICK arm, in pool order.

    Built by asking `_trial_features` which cues it kept (``with_indices``) and indexing the
    per-cue arrays at those positions -- never by rebuilding the filter. That reconciliation is
    exact: PS94 8/17's no-lick arm is 354 = 353 undetected + 1 late_rewarded, PS95's 273 = 265 + 8,
    which is precisely the discrepancy that made the first attempt refuse to run.
    """
    from wfield_local import nolick_decoder as nd
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_position_decoder import _trial_features

    refs = REFERENCE_POSITIONS if reference_positions is None else reference_positions
    eng, cats = [], []
    for lab in kept:
        s_ = next((x for x in SESSIONS if x["label"] == lab), None)
        if s_ is None:
            continue
        *_rest, idx_e, idx_nl = _trial_features(s_, args, with_indices=True)
        ca = nd._args(align=args.align)
        ca.max_rt = args.max_rt
        codes, cat, _blk, _rt, _cf, _pre = nd.categorize(s_, ca)
        st, _info = poststroke_engagement(s_, refs)
        eng.append(np.asarray(st)[idx_nl])
        cats.append(np.asarray(cat, dtype=object)[idx_nl])
    return (np.concatenate(eng) if eng else np.array([], bool),
            np.concatenate(cats) if cats else np.array([], dtype=object))


def undetected_state_split(d, keep, args, seed=0):
    """THE ARBITRATING TEST: is the engaged-like reading real, or a global post-stroke shift?

    Priya (2026-08-18) read the numbers the other way round from me and was right: post-stroke
    undetected trials classify at 0.71-0.86 against pre-stroke undetected at 0.085-0.147, so they do
    not resemble unengaged trials. My control asked whether the boundary separates post-stroke
    engaged from post-stroke DISENGAGED -- but under a valid gate there are almost no disengaged
    trials to separate (PS95: none at all), so the absence of separation was the absence of a second
    class, not a blind boundary.

    This splits the post-stroke no-lick trials by the REFERENCE-POSITION gate (close_L,
    close_center -- positions a left VLS lesion should spare) and scores each subset:

        working trials engaged-like AND disengaged trials lower -> execution failure, boundary works
        both equally high                                       -> global shift, not interpretable
        too few disengaged trials                               -> UNDECIDABLE, and reported as such
    """
    base = looks_like_which(d, keep, seed=seed)
    if "post_undetected_frac_classified_ENGAGED_like" not in base:
        return base
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

    state, cats = nolick_state_vector(d["kept"], args)
    out = dict(base)
    if state.size != d["GU"].size:
        out["split"] = {"note": f"state vector {state.size} vs pooled no-lick {d['GU'].size}; "
                                f"refusing to pair them"}
        return out
    post_m = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], keep)
    p = clf.predict(d["XU"])
    w = post_m & state
    dis = post_m & ~state
    res = {"n_working": int(w.sum()), "n_disengaged": int(dis.sum()),
           "n_late_rewarded_in_arm": int((cats[post_m] == "late_rewarded").sum())}
    if w.sum() >= 10:
        res["working_frac_engaged_like"] = float(p[w].mean())
    if dis.sum() >= 10:
        res["disengaged_frac_engaged_like"] = float(p[dis].mean())
    if "working_frac_engaged_like" in res and "disengaged_frac_engaged_like" in res:
        gap = res["working_frac_engaged_like"] - res["disengaged_frac_engaged_like"]
        res["working_minus_disengaged"] = gap
        res["verdict"] = ("boundary STILL tracks state -> engaged-like WORKING trials are a real "
                          "execution-failure signature" if gap > 0.10 else
                          "no separation -> consistent with a global post-stroke shift")
    else:
        res["verdict"] = ("UNDECIDABLE: too few disengaged trials under the reference-position gate. "
                          "That is itself informative -- the animal was working almost throughout.")
    out["split"] = res
    return out


def impaired_nolick_readout(d, keep, alignment="precue", n_perm=2000):
    """Was a PLAN formed on post-stroke trials where no lick was detected? (Priya, 2026-08-18.)

    This replaces `undetected_state_split`, which needed a "disengaged" label that has no valid
    post-stroke construction (see POSTSTROKE_ENGAGEMENT_FILTERING). The question it was asking is
    answerable WITHIN the post-stroke session and without any engagement label, by splitting the
    no-lick trials on something measured rather than inferred: whether the TRUE position was one the
    animal still reaches (preserved) or one it has stopped reaching (impaired).

    Apply the frozen pre-stroke decoder to post-stroke NO-LICK trials and ask whether it still reads
    out the correct position:

      * above chance at IMPAIRED positions -> the position was represented and the lick did not
        happen: EXECUTION failure with the plan intact. This is the strong result, because those are
        exactly the trials a motor-output account predicts and a "no plan" account does not.
      * at chance everywhere -> no readable plan on no-lick trials; execution failure is not
        supported (though absence of decodable signal is weaker evidence than presence).

    The preserved-position arm is the internal control: no-lick trials there are the animal's own
    baseline lapses at positions it can still reach, so it calibrates the impaired arm within the
    same session, the same recording, and the same decoder.

    CAVEAT that cannot be removed from spout data alone: "no lick detected" is not "no tongue
    protrusion". PS93's rightward bias already produces licks that occur without reaching far_L. DLC
    tongue tracking replaces this inference with a measurement; until then a null here is ambiguous
    between "no plan" and "plan plus a protrusion the spout never saw".
    """
    impaired = [c for c in DISPLAY_ORDER if c not in keep]
    tr = np.isin(d["GE"], list(d["pre_i"]))
    clf = _pipe().fit(d["XE"][tr], d["YE"][tr])
    post_u = np.isin(d["GU"], list(d["post_i"]))
    out = {"preserved_positions": [POSITION_NAMES[c] for c in keep],
           "impaired_positions": [POSITION_NAMES[c] for c in impaired]}
    for name, labs in (("preserved", keep), ("impaired", impaired)):
        m = post_u & np.isin(d["YU"], labs)
        if m.sum() < MIN_POST:
            out[name] = {"n": int(m.sum()), "note": f"fewer than {MIN_POST} no-lick trials"}
            continue
        y, pred = d["YU"][m], clf.predict(d["XU"][m])
        # Scored over ALL six labels: the frozen decoder may predict any position, and restricting
        # the label set would hand it a chance level it was not operating under.
        r = na.evaluate_arm(y, pred, n_perm=n_perm, labels=list(DISPLAY_ORDER))
        r["n"] = int(m.sum())
        r["per_position"] = na.per_position_recall(y, pred, labels=list(labs))
        out[name] = r
    a, b = out.get("impaired", {}), out.get("preserved", {})
    if "balanced_accuracy" not in a:
        out["verdict"] = "UNDECIDABLE -- too few no-lick trials at impaired positions"
        return out
    if "balanced_accuracy" in b:
        out["impaired_minus_preserved"] = float(a["balanced_accuracy"] - b["balanced_accuracy"])

    # MULTIPLICITY: this function is run per alignment x arm, so a whole pass is 2 alignments x
    # 2 arms x n_animals tests. A single uncorrected p just under 0.05 is not a result, and the first
    # version of this verdict declared "execution failure" from PS95's p=0.017 (x8 = 0.14).
    p = a.get("bal_p", 1.0)
    out["bal_p_bonferroni_x8"] = float(min(1.0, p * 8))
    survives = out["bal_p_bonferroni_x8"] < 0.05

    if alignment == "precue":
        # The clean index of a PLAN: before the cue there is no movement to explain the signal.
        out["verdict"] = (
            "PRE-CUE position code present on no-lick trials at impaired positions -> plan formed, "
            "movement failed" if survives else
            "no decodable PRE-CUE plan on impaired no-lick trials -> execution failure NOT supported "
            "by this test" + ("" if p >= 0.05 else " once corrected for the 8 tests in a pass"))
    else:
        # POST-cue is NOT a plan readout on these trials. The spout is physically present throughout,
        # so a position-specific sensory response exists whether or not anything was planned, and
        # post-cue decoding is in any case largely lick-driven -- which is why the pre-stroke
        # dissociation used pre-cue survival as the discriminating quantity.
        out["verdict"] = (
            "post-cue position code present on impaired no-lick trials -- AMBIGUOUS: the spout is "
            "present, so this may be a sensory response rather than a plan. Read the pre-cue arm."
            if survives else
            "no post-cue position code on impaired no-lick trials")
    out["verdict"] += ". CAVEAT: 'no lick detected' is not 'no tongue protrusion' -- DLC settles it."
    return out


def recoding_test(d, keep, min_trials=40, n_splits=5, post_all_trials=True):
    """Is the position code LOST, or RECODED? Frozen pre-stroke decoder vs a within-session one.

    THE DISTINCTION THIS EXISTS TO DRAW. `decode_matched` shows PS94's frozen pre-stroke decoder
    falling below every pre-stroke session on 8/17, and that was reported as a decoding deficit. But a
    frozen decoder fails for two quite different reasons: the information is gone, or the information
    is there in a DIFFERENT code that the old model cannot read. Training a decoder on the post-stroke
    session itself separates them -- if it recovers normal accuracy, the information is intact and only
    the mapping changed.

    THE COMPARISON MUST BE POSITION-MATCHED, and this is the trap it was built to avoid. PS94 8/17 has
    engaged trials at 4 positions where its pre-stroke sessions have 6, so an unmatched within-session
    comparison pits a 4-way problem (chance 0.25) against 6-way ones (chance 0.167) and inflates the
    post-stroke side. That is the same trial-composition error that produced a spurious 'PS94 neural
    deficit' headline earlier in this project, running the other way. Pre-stroke sessions are therefore
    restricted to the SAME positions before their band is computed.

    Both arms use GroupKFold on the real position blocks, so the within-session number carries the same
    block-CV convention as everything else in the deck.
    """
    # POST-STROKE SESSIONS USE ALL TRIALS (Priya, 2026-08-18): the missing licks ARE the phenotype, so
    # filtering to engaged trials removes the effect being measured -- and post-stroke that is not a
    # small correction, PS94 8/18 is only 40% engaged. Pre-stroke keeps the engaged cut, which is the
    # mismatch declared in nolick_analysis.SANCTIONED_MISMATCHES. An earlier version of this function
    # used engaged trials on BOTH sides, contradicting the decision it was written under.
    all_pos = list(DISPLAY_ORDER)
    scored = all_pos if post_all_trials else list(keep)
    rows = []
    for i in sorted(d["pre_i"]) + sorted(d["post_i"]):
        is_post = i in d["post_i"]
        if is_post and post_all_trials:
            # ALL SIX POSITIONS. Restricting to `keep` would drop exactly the positions the lesion
            # abolished -- the ones this arm exists to examine (Priya, 2026-08-19). PS94 far_center
            # and far_R have ZERO engaged trials and ~105 no-lick trials each.
            me = (d["GE"] == i) & np.isin(d["YE"], all_pos)
            mu = (d["GU"] == i) & np.isin(d["YU"], all_pos)
            X = np.vstack([d["XE"][me], d["XU"][mu]])
            y = np.concatenate([d["YE"][me], d["YU"][mu]])
            # the block vector must be filtered by the SAME position set as the trials it labels --
            # this line still said `keep` after the trials moved to all_pos, giving 600 trials and
            # 597 block ids
            ge = np.asarray(d["BE"][i])[np.isin(d["YE"][d["GE"] == i], all_pos)]
            # no-lick trials carry no block id in BE; give them their own groups so they cannot
            # straddle a CV fold with the engaged trials they sit between
            gb = np.concatenate([ge, np.arange(len(ge), len(ge) + int(mu.sum()))])
            m = np.ones(len(y), bool)
        else:
            # pre-stroke sessions are scored over whichever set this arm uses, so the band and the
            # post value are always on the same positions and the same chance level
            pos = all_pos if post_all_trials else keep
            m = (d["GE"] == i) & np.isin(d["YE"], pos)
            if m.sum() < min_trials:
                continue
            y, X = d["YE"][m], d["XE"][m]
            gb = np.asarray(d["BE"][i])[np.isin(d["YE"][d["GE"] == i], pos)]
        if len(y) < min_trials:
            continue
        ng = min(n_splits, int(np.unique(gb).size))
        if ng < 2:
            continue
        acc = float(accuracy_score(y, cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=gb)))
        rows.append({"label": d["kept"][i], "within_accuracy": acc, "n": int(len(y)),
                     "post": is_post})
    pre = np.array([r["within_accuracy"] for r in rows if not r["post"]], float)
    post = [r for r in rows if r["post"]]
    if len(pre) < 3 or not post:
        return {"note": "not enough sessions", "n_pre": int(len(pre)), "n_post": len(post)}
    band = {"mean": float(pre.mean()), "sd": float(pre.std(ddof=1)),
            "min": float(pre.min()), "max": float(pre.max()), "n": int(len(pre))}
    out = {"n_positions": len(scored), "chance": 1.0 / len(scored), "within_pre_band": band,
           "positions_scored": [POSITION_NAMES[c] for c in scored],
           "post_arm": "ALL trials" if post_all_trials else "engaged only", "per_session": rows}
    for r in post:
        z = (r["within_accuracy"] - band["mean"]) / band["sd"] if band["sd"] else float("nan")
        out.setdefault("post", {})[r["label"]] = {
            "within_accuracy": r["within_accuracy"], "z": float(z),
            "inside_pre_range": bool(band["min"] <= r["within_accuracy"] <= band["max"])}
    # DIRECTION MATTERS. The first version tested only `inside_pre_range` and so called PS95 cue
    # (z=+1.4, ABOVE every pre-stroke session) "impaired" -- outside on the HIGH side is the opposite
    # of impairment. Only a value below the pre-stroke range is evidence of degraded information.
    vals = list(out.get("post", {}).values())
    below = [v for v in vals if v["z"] < 0 and not v["inside_pre_range"]]
    above = [v for v in vals if v["z"] > 0 and not v["inside_pre_range"]]
    if below:
        out["verdict"] = ("within-session decoding is ALSO impaired -> the position information "
                          "itself is degraded, not merely recoded")
    elif above:
        out["verdict"] = ("within-session decoding is ABOVE the pre-stroke range -> the information "
                          "is intact and if anything sharper; the frozen decoder fails because the "
                          "CODE CHANGED: RECODING, not loss")
    else:
        out["verdict"] = ("within-session decoding is NORMAL -> the position information is INTACT "
                          "and the frozen decoder fails because the CODE CHANGED: RECODING, not loss")
    out["information_degraded"] = bool(below)
    return out
