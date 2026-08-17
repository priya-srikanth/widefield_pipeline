"""Do trials WITHOUT a detected lick still carry the position code?

THE QUESTION THIS EXISTS FOR. Post-stroke, an animal will fail trials. A failed trial can mean the
plan was never formed (engagement/attention) or that it was formed and the movement failed (motor).
Those are different injuries and they look identical in the behaviour log. Widefield can separate
them, because the two make opposite predictions about WHEN in the trial the position code survives:

    plan intact, execution fails  ->  PRE-CUE code preserved, POST-CUE code collapses
    plan never formed             ->  both collapse

Pre-stroke no-detected-lick trials are the reference. They are not a curiosity: they are the only
pre-stroke data of the same kind as a post-stroke failure, and the reference has to be fixed BEFORE
post-stroke data exists or every later choice is made with the answer in view.

WHY "no_detected_lick" AND NOT "no lick". The lick sensor fires on contact at the spout, so a lick
that is EXECUTED but falls short registers as nothing at all. PS93 has a pre-existing rightward
tongue bias and reaches far_L poorly (Priya, 2026-08-17), so a large share of its far-position
"no-lick" trials are near-certainly attempted-and-short, not unattempted. The category is named for
what is measured -- absence of a DETECTION -- so that DLC/facial-tracking can later split it into
attempted vs not without renaming anything or invalidating stored results. Until then, treat these
trials as a MIXTURE, and read the per-position breakdown rather than the pooled number.

PS93's far_L is in fact the most valuable cell in the whole table: a within-subject, PRE-stroke
instance of "plan intact, execution fails", whose ground truth comes from a tongue bias that has
nothing to do with the stroke. Whatever signature the post-stroke analysis claims should already be
visible there.

THE STATISTICS, AND WHY THE OBVIOUS VERSION IS WRONG. The frozen decoder previously judged no-lick
accuracy against uniform 1/6 and reported "above chance" for all four animals. That null is not
valid here, because BOTH sides are biased:

  * the TRIALS are not uniform over positions -- animals decline far positions, and PS93's no-lick
    trials are 49% far_center, 25% far_L;
  * the DECODER's predictions on these trials are not uniform either -- PS94 places 33% of them on
    a single position.

Two biases that happen to overlap produce above-chance accuracy with no information whatsoever. For
PS93 a constant "always guess far_center" scores 0.490, far above the 0.293 actually measured.

So this module reports, in order:

  1. BALANCED accuracy (macro-recall) as the headline. Under the null its expectation is EXACTLY
     1/6 no matter how skewed the trials or the predictions are: E[recall_c] = P(pred=c) = q_c, and
     the macro-average is (1/6)*sum(q_c) = 1/6 because the q_c sum to one. It is the one summary
     that both biases cannot move.
  2. RAW accuracy against a PERMUTATION null computed on these trials -- labels shuffled, model and
     predictions untouched -- which reproduces the collision of the two biases and so measures what
     is left over. Its expectation is sum(q_c * p_c), which is where 0.211 rather than 0.167 comes
     from for PS93.
  3. A POSITION-MATCHED subsample as an independent check, since (1) and (2) are corrections and a
     reader may reasonably want the version with the confound physically removed rather than
     modelled.

An above-chance PRE-CUE result here is expected and biologically meaningful (Priya, 2026-08-17): the
animal can know where the spout is and still not lick. The discriminating comparison is not
"pre-cue vs chance" but "pre-cue vs post-cue WITHIN the same trials", which is why every quantity is
computed for both alignments and reported as a ratio against the engaged trials of the same session.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

CHANCE = 1.0 / len(DISPLAY_ORDER)
N_PERM = 2000


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------
def balanced_accuracy(y_true, y_pred, labels=DISPLAY_ORDER):
    """Macro-averaged recall: the mean of per-class recall over classes PRESENT in y_true.

    This is the headline metric because its null expectation is exactly 1/len(labels) regardless of
    how the trials or the predictions are distributed (see module docstring). Classes absent from
    y_true are skipped rather than scored 0 -- with a skewed no-lick set some positions can have no
    trials at all, and counting them as failures would penalise the animal for what it declined.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rec = [float((y_pred[m] == c).mean()) for c in labels if (m := (y_true == c)).any()]
    return float(np.mean(rec)) if rec else float("nan")


def per_position_recall(y_true, y_pred, labels=DISPLAY_ORDER):
    out = {}
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for c in labels:
        m = y_true == c
        out[POSITION_NAMES[c]] = {"n": int(m.sum()),
                                  "recall": float((y_pred[m] == c).mean()) if m.any() else float("nan")}
    return out


def majority_class_floor(y_true, labels=DISPLAY_ORDER):
    """Accuracy of the best CONSTANT predictor. Not a null -- a sanity bound.

    If a reported accuracy sits below this, an uninformative rule beats the decoder outright and no
    claim of preserved coding should be made from the raw number.
    """
    y_true = np.asarray(y_true)
    if not y_true.size:
        return float("nan")
    return float(max((y_true == c).mean() for c in labels))


def permutation_null(y_true, y_pred, n_perm=N_PERM, seed=0, labels=DISPLAY_ORDER):
    """Null distribution for raw AND balanced accuracy, with the model's predictions held fixed.

    Shuffling the LABELS (not the predictions) keeps the decoder's prediction bias exactly as it is
    and destroys only the trial-to-label correspondence, so the null inherits both skews. This is
    the difference between a null of 0.211 and a null of 0.167 for PS93.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = y_true.size
    if n == 0:
        nan2 = [float("nan")] * 2
        return {"raw_null_mean": float("nan"), "raw_null_ci": nan2, "raw_p": float("nan"),
                "bal_null_mean": float("nan"), "bal_null_ci": nan2, "bal_p": float("nan"), "n_perm": 0}
    rng = np.random.RandomState(seed)
    raw_obs = float((y_pred == y_true).mean())
    bal_obs = balanced_accuracy(y_true, y_pred, labels)
    raw, bal = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        ysh = rng.permutation(y_true)
        raw[i] = (y_pred == ysh).mean()
        bal[i] = balanced_accuracy(ysh, y_pred, labels)
    # +1 in numerator and denominator: with n_perm draws a p of exactly 0 is not a value the test
    # can produce, and reporting one invites it to be read as certainty.
    return {"raw_null_mean": float(raw.mean()),
            "raw_null_ci": [float(np.percentile(raw, 2.5)), float(np.percentile(raw, 97.5))],
            "raw_p": float((np.sum(raw >= raw_obs) + 1) / (n_perm + 1)),
            "bal_null_mean": float(np.nanmean(bal)),
            "bal_null_ci": [float(np.nanpercentile(bal, 2.5)), float(np.nanpercentile(bal, 97.5))],
            "bal_p": float((np.sum(bal >= bal_obs) + 1) / (n_perm + 1)),
            "n_perm": int(n_perm)}


def match_profile(y, target_frac, seed=0, labels=DISPLAY_ORDER, n_draws=200):
    """Indices of a subsample of `y` whose position profile matches `target_frac`.

    The correction-free version of the imbalance fix: rather than modelling the skew, remove it. The
    subsample is the largest one that can hit the target exactly, so it is bounded by the scarcest
    position relative to its target share -- which for PS93 is severe, and is exactly why this is a
    robustness check and not the headline.

    Returns a LIST of index arrays (n_draws resamples), because a single draw of a small subsample
    is noisy and the caller should average over draws rather than trust one.
    """
    y = np.asarray(y)
    tf = np.asarray([target_frac[POSITION_NAMES[c]] if isinstance(target_frac, dict) else target_frac[i]
                     for i, c in enumerate(labels)], float)
    tf = tf / tf.sum()
    have = np.array([(y == c).sum() for c in labels], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(tf > 0, have / np.where(tf > 0, tf, np.nan), np.inf)
    total = int(np.floor(np.nanmin(cap)))
    want = np.floor(tf * total).astype(int)
    if total <= 0 or want.sum() == 0:
        return []
    rng = np.random.RandomState(seed)
    idx_by_c = {c: np.flatnonzero(y == c) for c in labels}
    draws = []
    for _ in range(n_draws):
        pick = [rng.choice(idx_by_c[c], size=w, replace=False) for c, w in zip(labels, want) if w > 0]
        draws.append(np.sort(np.concatenate(pick)))
    return draws


# --------------------------------------------------------------------------------------------------
# one arm of one session
# --------------------------------------------------------------------------------------------------
def evaluate_arm(y_true, y_pred, target_frac=None, n_perm=N_PERM, seed=0, labels=DISPLAY_ORDER):
    """Every number this module reports for one set of trials and one set of predictions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = int(y_true.size)
    out = {"n": n,
           "accuracy": float((y_pred == y_true).mean()) if n else float("nan"),
           "balanced_accuracy": balanced_accuracy(y_true, y_pred, labels),
           "majority_class_floor": majority_class_floor(y_true, labels),
           "position_frac": {POSITION_NAMES[c]: float((y_true == c).mean()) if n else float("nan")
                             for c in labels},
           "pred_frac": {POSITION_NAMES[c]: float((y_pred == c).mean()) if n else float("nan")
                         for c in labels},
           "recall_by_position": per_position_recall(y_true, y_pred, labels),
           "chance_uniform": CHANCE}
    out.update(permutation_null(y_true, y_pred, n_perm=n_perm, seed=seed, labels=labels))
    out["above_null_raw"] = bool(n and out["accuracy"] > out["raw_null_ci"][1])
    out["above_null_balanced"] = bool(n and out["balanced_accuracy"] > out["bal_null_ci"][1])
    # The old flag, kept ONLY so its disagreement with the corrected one is visible in the record
    # rather than silently replaced. It is the claim this module exists to retire.
    out["above_uniform_chance_DEPRECATED"] = bool(n and out["accuracy"] > CHANCE)

    if target_frac is not None and n:
        draws = match_profile(y_true, target_frac, seed=seed, labels=labels)
        if draws:
            accs = [float((y_pred[d] == y_true[d]).mean()) for d in draws]
            bals = [balanced_accuracy(y_true[d], y_pred[d], labels) for d in draws]
            out["matched"] = {"n_per_draw": int(len(draws[0])), "n_draws": len(draws),
                              "accuracy": float(np.mean(accs)),
                              "accuracy_ci": [float(np.percentile(accs, 2.5)),
                                              float(np.percentile(accs, 97.5))],
                              "balanced_accuracy": float(np.nanmean(bals))}
        else:
            out["matched"] = {"n_per_draw": 0, "note": "no subsample can hit the target profile"}
    return out


def compare_arms(engaged, nolick):
    """The contrast the whole module is for: how much of the engaged code survives without a lick.

    Reported as a RATIO of balanced accuracies above chance, not of raw accuracies, so that neither
    the trial skew nor the prediction skew leaks into the comparison. Ratios are on the
    above-chance part because a ratio of raw accuracies treats 1/6 as zero and inflates everything.
    """
    def _above(d):
        b = d.get("balanced_accuracy", float("nan"))
        return b - CHANCE
    e, u = _above(engaged), _above(nolick)
    return {"engaged_balanced_above_chance": e, "nolick_balanced_above_chance": u,
            "survival_ratio": float(u / e) if e and np.isfinite(e) and e > 0 else float("nan")}


DISSOCIATION_MIN = 1.5     # pre-cue must survive this many times better than post-cue
ALPHA = 0.05


def interpret(precue_cmp, cue_cmp, precue_arm=None, cue_arm=None):
    """Turn the two survival ratios into the hypothesis they support -- explicitly, not by eye.

    THE QUANTITY IS THE CONTRAST, NOT TWO ABSOLUTE LEVELS. The first version of this function
    thresholded each survival ratio at 0.5 independently, and it mislabelled the very data it was
    written for: PS92 survives 0.401 pre-cue against 0.143 post-cue -- a 2.8x dissociation, LARGER
    than PS93's 1.8x -- yet scored "no plan formed" purely because 0.401 < 0.5, while PS93's 0.627
    passed. How strong an animal's pre-cue code is and whether it OUTLIVES the movement are
    different questions, and only the second distinguishes the two injuries.

    Evidence that the pre-cue code is present at all comes from that arm's own permutation p-value
    when the arm dicts are supplied, not from the ratio -- a ratio can look healthy because its
    denominator is small. Without them the verdict still computes but says it is unverified, rather
    than quietly implying a significance test that never ran.
    """
    p, c = precue_cmp["survival_ratio"], cue_cmp["survival_ratio"]
    if not (np.isfinite(p) and np.isfinite(c)):
        return "indeterminate: a survival ratio is undefined (engaged decoding at or below chance)"

    pre_sig = None if precue_arm is None else bool(precue_arm.get("bal_p", 1.0) < ALPHA)
    ratio = (p / c) if c > 0 else float("inf")
    qual = "" if pre_sig is not None else "  [level unverified: no permutation p supplied]"

    if pre_sig is False:
        return ("consistent with NO PLAN FORMED: the pre-cue position code is not above its own "
                f"permutation null on trials without a detected lick (p>={ALPHA}), so there is no "
                f"preserved code for the movement to have failed to execute")
    if ratio >= DISSOCIATION_MIN:
        return (f"consistent with PLAN INTACT, EXECUTION FAILED: the pre-cue code survives "
                f"{ratio:.1f}x better than the post-cue code without a detected lick "
                f"({p:.2f} vs {c:.2f}){qual}")
    if c >= 0.5:
        return ("UNEXPECTED: the post-cue code survives about as well as the pre-cue code without a "
                "detected lick. Either the post-cue decode is not movement-driven, or these trials "
                "contain undetected licks -- check the per-position breakdown before interpreting")
    return (f"NO CLEAR DISSOCIATION: pre-cue {p:.2f} vs post-cue {c:.2f} ({ratio:.1f}x, below the "
            f"{DISSOCIATION_MIN}x threshold) -- both arms degrade together{qual}")


ARMS = ("engaged", "engaged_fast", "engaged_slow", "late", "undetected", "nolick_pooled")


def summarize(res, fh=None):
    """Human-readable block. Printed by the CLI and by the nightly run.

    Arms that a session does not have (too few late trials, no RT split) are skipped rather than
    printed as NaN -- an absent arm and a measured-but-empty one should not look the same.
    """
    lines = []
    for al in ("precue", "cue"):
        if al not in res or not isinstance(res[al], dict):
            continue
        r = res[al]
        lines.append(f"  [{al}]")
        for arm in ARMS:
            d = r.get(arm)
            if not isinstance(d, dict) or not d.get("n"):
                continue
            mt = d.get("matched") or {}
            mstr = (f"  matched acc={mt['accuracy']:.3f} (n={mt['n_per_draw']})"
                    if mt.get("n_per_draw") else "")
            lines.append(f"    {arm:14s} n={d['n']:5d}  bal={d['balanced_accuracy']:.3f} "
                         f"(null {d['bal_null_mean']:.3f}, p={d['bal_p']:.4f})  "
                         f"raw={d['accuracy']:.3f} (null {d['raw_null_mean']:.3f}, "
                         f"floor {d['majority_class_floor']:.3f}){mstr}")
        if isinstance(r.get("compare"), dict):
            lines.append(f"    survival ratio (no-detected-lick / engaged, above chance) = "
                         f"{r['compare']['survival_ratio']:.3f}")
    if res.get("interpretation"):
        lines.append(f"  => {res['interpretation']}")
    txt = "\n".join(lines)
    if fh is None:
        print(txt, flush=True)
    else:
        fh.write(txt + "\n")
    return txt


def write_reference(res, path):
    """Freeze the pre-stroke reference to JSON. Written once, before post-stroke data exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    return path
