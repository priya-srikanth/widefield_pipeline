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
    return {"n_train_per_class": int((y == 1).sum()), "n_post_undetected": int(post_u.sum()),
            "pre_separability_cv": sep,
            "post_undetected_frac_classified_ENGAGED_like": p_post,
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


def precue_lick_mask(s, args, want_licks):
    """Mask over the trials `_trial_features` KEEPS, True where the fixed pre-cue window had licks.

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
