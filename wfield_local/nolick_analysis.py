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


DISSOCIATION_MIN_DIFF = 0.15   # label threshold on the DIFFERENCE; dissociation_ci is the real test
ALPHA = 0.05
N_BOOT = 2000


def _survival(y_e, p_e, y_u, p_u, labels=DISPLAY_ORDER):
    e = balanced_accuracy(y_e, p_e, labels) - CHANCE
    u = balanced_accuracy(y_u, p_u, labels) - CHANCE
    return (u / e) if (np.isfinite(e) and e > 0) else np.nan


def dissociation_ci(precue_raw, cue_raw, n_boot=N_BOOT, alpha=ALPHA, seed=0,
                    labels=DISPLAY_ORDER, by="session"):
    """Is the pre-cue/post-cue dissociation real for this animal? Cluster bootstrap over SESSIONS.

    THIS REPLACES THE THRESHOLD. `interpret`'s 1.5x cut turns a continuous quantity into a label,
    and the cut is where PS92 (1.23x) and PS94 (1.57x) land on opposite sides of a line nobody
    measured -- so the consensus reported "bases disagree" for a marginal difference. Asking instead
    whether the pre-cue MINUS post-cue survival difference excludes zero needs no threshold, and
    gives each animal an uncertainty rather than a verdict.

    SESSIONS are the resampling unit, not trials: the claim generalises to an unseen DAY, and trials
    within a session are not independent (~6-trial position blocks, shared F0, shared engagement
    state). Resampling trials would produce intervals far too narrow -- the same reasoning as
    decode_ci.frozen_ci, which resamples sessions for the same reason.

    The draw is PAIRED: each bootstrap replicate uses the SAME session sample for both alignments,
    because pre-cue and post-cue are measured on the same trials and an unpaired interval would
    inflate the variance of their difference with between-session variation common to both.

    `precue_raw` / `cue_raw` are dicts with ``engaged`` and ``nolick`` entries, each a
    ``(y_true, y_pred, session_label)`` triple.
    """
    def _ok(d):
        return d and all(k in d for k in ("engaged", "nolick"))

    if not (_ok(precue_raw) and _ok(cue_raw)):
        return {"n_boot": 0, "note": "raw per-trial arrays unavailable"}

    sess = sorted(set(np.asarray(precue_raw["engaged"][2]).tolist()))
    if len(sess) < 3:
        return {"n_boot": 0, "n_sessions": len(sess),
                "note": "fewer than 3 sessions: a session bootstrap would be meaningless"}

    rng = np.random.RandomState(seed)

    def _idx(arr_sess, chosen, arr_blk=None):
        # sessions are drawn WITH replacement, so a session picked twice must contribute its trials
        # twice -- concatenating per-session index blocks does that; a boolean mask would not.
        per = {s: np.flatnonzero(np.asarray(arr_sess) == s) for s in set(chosen)}
        if by == "session" or arr_blk is None:
            return (np.concatenate([per[s] for s in chosen if per[s].size])
                    if chosen else np.array([], int))
        # NESTED: having drawn a session, also resample the BLOCKS within it. A pure session
        # bootstrap treats a drawn session's trials as known exactly, which is fine at ~500 trials
        # and not fine at the 4-6 an undetected arm can contribute. The outer level still dominates;
        # this adds the inner one rather than replacing it.
        out = []
        blk = np.asarray(arr_blk)
        for s_ in chosen:
            ix = per[s_]
            if not ix.size:
                continue
            bs = np.unique(blk[ix])
            drawn = rng.choice(bs, size=len(bs), replace=True)
            for b_ in drawn:
                out.append(ix[blk[ix] == b_])
        return np.concatenate(out) if out else np.array([], int)

    diffs, pres, cues = [], [], []
    for _ in range(n_boot):
        pick = list(rng.choice(sess, size=len(sess), replace=True))
        vals = {}
        for nm, raw in (("pre", precue_raw), ("cue", cue_raw)):
            ye, pe, se = (np.asarray(x) for x in raw["engaged"][:3])
            yu, pu, su = (np.asarray(x) for x in raw["nolick"][:3])
            be = raw["engaged"][3] if len(raw["engaged"]) > 3 else None
            bu = raw["nolick"][3] if len(raw["nolick"]) > 3 else None
            ie, iu = _idx(se, pick, be), _idx(su, pick, bu)
            vals[nm] = (np.nan if (ie.size == 0 or iu.size == 0)
                        else _survival(ye[ie], pe[ie], yu[iu], pu[iu], labels))
        if np.isfinite(vals["pre"]) and np.isfinite(vals["cue"]):
            pres.append(vals["pre"]); cues.append(vals["cue"])
            diffs.append(vals["pre"] - vals["cue"])
    if len(diffs) < max(50, n_boot // 10):
        return {"n_boot": len(diffs), "n_sessions": len(sess),
                "note": "too many bootstrap replicates were undefined to report an interval"}
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one = float((np.sum(d <= 0) + 1) / (len(d) + 1))
    lb_one = float(np.percentile(d, 100 * alpha))       # one-sided lower bound, matches p_one
    # ONE CONVENTION, STATED. The hypothesis is directional -- pre-cue OUTLIVES post-cue -- so the
    # test is one-sided and `significant` keys off it. The two-sided interval is kept because it is
    # what a reader expects to see, but it is DESCRIPTIVE: at the margin the two disagree (joint PS94
    # measured p=0.037 with a 95% two-sided CI containing zero), and reporting both without saying
    # which is the test is an invitation to quote whichever looks better.
    return {"n_boot": int(len(d)), "n_sessions": len(sess), "resampling_unit": by,
            "precue_survival_mean": float(np.mean(pres)),
            "cue_survival_mean": float(np.mean(cues)),
            "difference_mean": float(d.mean()),
            "difference_ci": [float(lo), float(hi)],
            "difference_ci_is": "two-sided 95%, DESCRIPTIVE -- not the test",
            "difference_lower_bound_one_sided": lb_one,
            "p_one_sided": p_one,
            "test": "one-sided (directional hypothesis: precue survival > postcue survival)",
            "significant": bool(p_one < alpha),
            "ci95_two_sided_excludes_zero": bool(lo > 0 or hi < 0),
            "direction": "precue > postcue" if d.mean() > 0 else "postcue >= precue"}


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
    # ratio only where the denominator is meaningfully above zero; post-cue survival is ~0 by
    # hypothesis, so p/c is unstable exactly when the dissociation is clearest (see
    # direction_consistency). The DIFFERENCE decides; the ratio is quoted when it is finite.
    ratio = (p / c) if c > 0.05 else float("inf")
    diff = p - c
    qual = "" if pre_sig is not None else "  [level unverified: no permutation p supplied]"

    if pre_sig is False:
        return ("consistent with NO PLAN FORMED: the pre-cue position code is not above its own "
                f"permutation null on trials without a detected lick (p>={ALPHA}), so there is no "
                f"preserved code for the movement to have failed to execute")
    if diff >= DISSOCIATION_MIN_DIFF:
        how = f"{ratio:.1f}x" if np.isfinite(ratio) else "with the post-cue code at ~zero"
        return (f"consistent with PLAN INTACT, EXECUTION FAILED: the pre-cue code survives {how} "
                f"better than the post-cue code without a detected lick "
                f"({p:.2f} vs {c:.2f}, difference {diff:+.2f}){qual}")
    if c >= 0.5:
        return ("UNEXPECTED: the post-cue code survives about as well as the pre-cue code without a "
                "detected lick. Either the post-cue decode is not movement-driven, or these trials "
                "contain undetected licks -- check the per-position breakdown before interpreting")
    return (f"NO CLEAR DISSOCIATION: pre-cue {p:.2f} vs post-cue {c:.2f} (difference {diff:+.2f}, "
            f"below the {DISSOCIATION_MIN_DIFF:+.2f} threshold) -- both arms degrade together{qual}")


ARMS = ("engaged", "engaged_fast", "engaged_slow", "late_rewarded", "undetected",
        "undetected_working", "undetected_disengaged", "nolick_pooled")


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


# --------------------------------------------------------------------------------------------------
# trial-selection criteria, and the guard that stops them being mixed
# --------------------------------------------------------------------------------------------------
#: A pre/post-stroke comparison is only meaningful if BOTH sides selected trials the same way.
#: Priya (2026-08-17) wants pre-stroke on lick-restricted trials but post-stroke on ALL trials,
#: because after the stroke a missing detection may be a tongue protrusion that fell short rather
#: than an absent attempt -- undecidable until DLC/FR lands. That is the right scientific instinct
#: and it is also the setup for a confound that mimics the very effect being measured:
#:
#:   engaged-only pre  vs  all-trials post   -> post scores lower partly because its trial set now
#:                                              includes trials the pre-stroke set excluded. No
#:                                              stroke required to produce the difference.
#:   engaged-only both                       -> post-stroke "engaged" is SURVIVORSHIP-SELECTED: it
#:                                              keeps exactly the trials where the movement still
#:                                              worked, biasing toward preserved function.
#:
#: Neither criterion is safe alone, so compute both and compare like with like. The guard below makes
#: that structural rather than remembered.
CRITERIA = {
    "engaged_2s": "first detected lick within decode.max_rt_s (2.0 s) of the cue",
    "engaged_respwin": "first detected lick within that session's response window",
    "all_trials": "every cue with a usable position label, licked or not",
    "engaged_reference_positions": "engaged as judged ONLY at reference positions the deficit is "
                                   "expected to spare (post-stroke motivation control; which "
                                   "positions is a phenotype question, hence parameterised)",
}


def reference_position_engagement(codes, cat, reference_positions):
    """Engagement judged only at positions the deficit is expected to spare.

    Post-stroke, "did not lick" at an impaired position confounds motivation with motor failure. At a
    position the animal can still reach, a miss is much more likely to be genuine disengagement. This
    returns the detected-lick rate restricted to `reference_positions`, so engagement can be measured
    where the deficit is not acting -- Priya's close_L / close_center proposal, left parameterised
    because which positions are spared is an empirical question about the phenotype, not a constant.
    """
    codes, cat = np.asarray(codes), np.asarray(cat)
    ref = np.isin(codes, list(reference_positions))
    if not ref.any():
        return {"n": 0, "engaged_rate": float("nan"),
                "note": "no trials at the reference positions"}
    eng = (cat[ref] == "engaged")
    return {"n": int(ref.sum()), "engaged_rate": float(eng.mean()),
            "reference_positions": sorted(reference_positions)}


def assert_comparable(a, b, what="comparison"):
    """Refuse to compare two results whose trial-selection criteria differ.

    Raises rather than warns. A warning would be read past, and the failure it guards against --
    a selection change that looks exactly like a stroke effect -- is not recoverable after the fact
    because the two numbers are individually correct.
    """
    ca, cb = (a or {}).get("criterion"), (b or {}).get("criterion")
    if ca is None or cb is None:
        raise ValueError(
            f"{what}: refusing to compare results without a recorded trial-selection criterion "
            f"(got {ca!r} and {cb!r}). Every result must carry `criterion`; see CRITERIA.")
    if ca != cb:
        raise ValueError(
            f"{what}: trial-selection criteria differ ({ca!r} vs {cb!r}). Comparing these would "
            f"confound the selection change with the effect. Recompute both sides under ONE "
            f"criterion -- compute all of them and compare like with like.")
    return True
