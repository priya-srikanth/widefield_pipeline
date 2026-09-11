"""The FROZEN pre-stroke behavioural-state decoder: does cortex still report quiet/running/licking?

THE SPECIFICITY CONTROL, and it has to be the FROZEN arm rather than the within-session one.

Priya, 2026-09-11, asked for "evidence that not *all* decoding/encoding degrades post-stroke". The
obvious version of that -- refit a state decoder inside each session and show it stays high -- does
not work, and the reason is measured rather than suspected: within session, binary running-vs-quiet
decodes at AUROC 0.99-1.00 and the three-way problem at macro-AUROC 0.98-1.00. That is a CEILING,
and a ceiling cannot demonstrate preservation; a reviewer reads it as "the task was too easy to
fail" and is right to.

The position claim rests on a FROZEN PRE-STROKE MODEL failing on post-stroke data. So the control
has to be the same object: freeze a pre-stroke state decoder, apply it to each post-stroke session,
and put its accuracy on the same axis as the frozen position decoder's. Cross-session generalisation
is the hard part of the position result, and the state arm has to survive exactly that.

    frozen POSITION decoder, by epoch    0.886 pre -> 0.523 acute  (collapses)
    frozen STATE decoder, by epoch       <- if this holds, the deficit is specific

WHY IT IS CREDIBLE THAT IT HOLDS, and why that is not circular: the lesion is ventrolateral
STRIATAL. No cortex is damaged, the window is the same, the basis is the same, and the animals run
MORE acutely than before (5.1% of session against 3.1% pre-stroke), so the state arm is not being
rescued by having more data at baseline than after.

BALANCED ACCURACY AND MACRO AUROC, never raw accuracy. The classes are wildly unbalanced within a
session -- PS94 8/06 holds 12 running segments against 324 quiet -- and the balance CHANGES across
epochs (quiet runs 3.4% of a pre-stroke session, 15.1% acutely, 0.7% chronically). Raw accuracy
under a shifting base rate is not comparable across the epochs this figure exists to compare.

THE BOOTSTRAP MUST CLUSTER BY PERIOD. Segments from one bout are not independent observations; the
same rule the position families apply to trials inside a scheduler block.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import locomotor_state as ls

#: Fewest segments of EVERY class a session needs before it is scored at all. A session that holds
#: three quiet segments can produce a balanced accuracy, and it would be noise with a number on it.
MIN_PER_CLASS = 15


def _pipe():
    """The same shape as the position decoder's: standardise, then multinomial logistic."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def fit_frozen(Xs, ys, classes=ls.THREE_WAY):
    """Fit one state decoder on POOLED pre-stroke segments. None if a class is missing."""
    if not len(ys):
        return None
    y = np.asarray(ys)
    if any((y == c).sum() < MIN_PER_CLASS for c in classes):
        return None
    return _pipe().fit(np.asarray(Xs), np.array([list(classes).index(v) for v in y]))


def score(model, X, y, classes=ls.THREE_WAY):
    """``{"balacc":…, "auroc":…, "n":…, "per_class":{…}}`` or None when a class is absent.

    PER-CLASS RECALL IS RETURNED ALONGSIDE THE SUMMARY, because a pooled balanced accuracy can hide
    the interesting failure: a state decoder that keeps quiet and running but loses LICKING would be
    reporting a change in the task-engaged state specifically, which is a different claim from
    "cortex still reports behavioural state" and should not be averaged into it.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    if model is None or not len(y):
        return None
    y = np.asarray(y)
    present = [c for c in classes if (y == c).sum()]
    if len(present) < 2:
        return None
    yi = np.array([list(classes).index(v) for v in y])
    P = model.predict_proba(np.asarray(X))
    full = np.zeros((len(yi), len(classes)))
    for j, c in enumerate(model.classes_):
        full[:, int(c)] = P[:, j]
    pred = full.argmax(1)
    out = {"n": len(yi),
           "balacc": float(balanced_accuracy_score(yi, pred)),
           "per_class": {c: float((pred[y == c] == list(classes).index(c)).mean())
                         for c in present},
           "counts": {c: int((y == c).sum()) for c in classes}}
    try:
        # ONE-VS-REST OVER THE CLASSES ACTUALLY PRESENT. Passing all three to `roc_auc_score` when
        # a session holds two raises rather than returning a partial score, and a session missing a
        # class is common here -- chronic sessions run 0.7% quiet.
        idx = [list(classes).index(c) for c in present]
        sub = full[:, idx]
        sub = sub / np.clip(sub.sum(1, keepdims=True), 1e-12, None)
        out["auroc"] = float(roc_auc_score(
            np.array([present.index(v) for v in y]), sub if len(present) > 2 else sub[:, 1],
            multi_class="ovr", average="macro") if len(present) > 2 else
            roc_auc_score(np.array([present.index(v) for v in y]), sub[:, 1]))
    except Exception:                                                  # noqa: BLE001
        out["auroc"] = float("nan")
    out["chance_balacc"] = 1.0 / len(present)
    out["classes_present"] = list(present)
    return out


#: Imaging frame rate the segment windows are cut at. The corrected (paired) rate, matching what
#: `_trial_features` uses for the trial arms -- the two must agree or a "1 s window" is two lengths.
FS_IMG = 31.23


@lru_cache(maxsize=4)
def by_animal_day(fs_img=FS_IMG):
    """``({animal: {"PRE": [score, ...], day: score}}, days)`` for the FROZEN state decoder.

    PRE IS LEAVE-ONE-SESSION-OUT and post-stroke days are scored by a model frozen on ALL pre-stroke
    segments, which is exactly the discipline `_collect_5c` uses for position. Without the LOSO the
    pre column would be a session scored partly against itself and the baseline would be too easy --
    the failure `_pre_reference` exists to prevent.

    CACHED BECAUSE IT PROJECTS EVERY SESSION ONTO THE JOINT BASIS, which is the expensive step
    (~10 min for the cohort) and is shared by every figure that reads this.
    """
    import numpy as np

    from wfield_local import behavior_events as be
    from wfield_local import config, joint_locanmf
    from wfield_local import locomotor_features as lf
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.paths import PathResolver

    rv = PathResolver()
    out, all_days = {}, set()
    for an in ANIMALS:
        try:
            basis = joint_locanmf.load(an, sessions=SESSIONS)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! state {an}: basis {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        got = {}
        for s in [x for x in SESSIONS if x["label"] in want]:
            lab = s["label"]
            mmdd = lab.split("_")[-1]
            ev = be.get_or_compute(rv, an, "2026" + mmdd)
            if not ev:
                continue
            try:
                sig, _r = joint_locanmf.BasisSource(basis, s).signal()
                X, y, g = lf.segment_features(s, np.asarray(sig), ev, fs_img=fs_img)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! state {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                continue
            if not len(y):
                continue
            d = _day(an, mmdd)
            if d is None:
                continue
            got[lab] = (X, y, g, int(d))
        pre = [(X, y) for X, y, _g, d in got.values() if d <= 0]
        if not pre:
            continue
        frozen = fit_frozen(np.vstack([a for a, _b in pre]),
                            np.concatenate([b for _a, b in pre]))
        rec = {"PRE": []}
        for lab, (X, y, _g, d) in sorted(got.items()):
            if d <= 0:
                keep = [(a, b) for l2, (a, b, _c, dq) in got.items() if l2 != lab and dq <= 0]
                if not keep:
                    continue
                m = fit_frozen(np.vstack([a for a, _b in keep]),
                               np.concatenate([b for _a, b in keep]))
            else:
                m = frozen
            r = score(m, X, y)
            if r is None:
                continue
            if d <= 0:
                rec["PRE"].append(r)
            else:
                rec[d] = r
                all_days.add(d)
        if len(rec) > 1 and rec["PRE"]:
            out[an] = rec
    return out, sorted(all_days)
