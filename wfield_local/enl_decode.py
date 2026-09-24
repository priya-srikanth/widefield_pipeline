"""Is the pre-cue position code still there when the animal has STOPPED? The sensory-vs-plan test.

Priya, 2026-09-24. `enl_states` sets up the classes; this is the arm that can carry a claim on its
own, and the reason it can is that it never compares the two classes' AMPLITUDES.

THE QUESTION THIS SETTLES THAT NOTHING ELSE IN THE PROJECT DOES. CLAUDE.md records the standing
limit on the pre-cue readout:

    "Deliberately NOT called a 'maintained motor plan': the spout arrives ~3 s before the cue, so a
     sustained sensory response and a held intention are temporally coextensive and this design
     cannot separate them."

Timing cannot separate them, so this separates them by BEHAVIOUR instead: remove the intention and
leave the stimulus. On a stopped trial the spout is still in position and the animal is no longer
going to move. **Position decodable from stopped ENL is direct evidence the pre-cue code is not
only a held plan** -- the first statement of that kind this project can make.

WHY DECODING AND NOT THE AMPLITUDE DIFFERENCE. A global state change -- satiety, drowsiness, less
fidgeting -- moves total ENL amplitude, and `enl_states` cannot tell that from a lost motor plan.
It CANNOT produce above-chance position decoding: being sleepier lowers all six positions, it does
not make far_L distinguishable from close_R. Position-specificity is the property that survives the
confound, which is why the claim lives here and the subtraction is reported beside it as
descriptive.

It also sidesteps the time confound entirely. `enl_states.adjacent_window` can only shrink the
working/stopped separation, never remove it (the gate makes the classes disjoint in time by
construction); a within-class decode against its own permutation null compares nothing across
classes and so inherits none of it.

FOUR READOUTS, WEAKEST TO STRONGEST:

  1  WITHIN-STOPPED   decode position from stopped ENL, against a per-arm permutation null.
                      Above null = the representation is present with no plan being formed.
  2  RATIO            stopped accuracy as a fraction of miss_working accuracy, with the CLASS
                      PROFILE MATCHED between arms (`nolick_analysis.match_profile`). Without that
                      match the comparison is partly an imbalance artefact: a position the animal
                      reaches poorly contributes more misses, so the two arms do not see the same
                      six-way problem.
  3  TRANSFER         train on miss_working, test on stopped. If a decoder built where a plan WAS
                      formed reads position out of trials where one was not, it is the SAME
                      representation rather than two coincidentally-decodable ones. If (1) works
                      and (3) fails, the code is present but ROTATED -- a different answer, and an
                      interesting one.
  4  SHARED DECODER   ONE decoder trained on `success`, blocks held out across ALL arms, scoring
                      `success`, `miss_working` and `stopped` with the same fitted model. Readout 2
                      divides two SEPARATELY FIT decoders, so its numerator and denominator differ
                      in training-set size as well as in biology -- and at the real trial counts
                      both fits sit at their nulls and the ratio is undefined. This one holds the
                      training set fixed, so **the ratio it yields is like-for-like**, and the
                      `stopped`/`miss_working` form of it is outcome-matched. It is the readout that
                      answers "how much".

POSITIONS ARE POOLED, AND THAT COSTS NOTHING HERE. Pre-stroke only PS94 and PS95 clear a
ten-trials-per-position floor (`scripts/enl_state_counts.py`, 2026-09-24: PS92 has TEN stopped
trials pre-stroke in total, three positions at zero). Pooling means not reporting per-position
effect sizes; the position LABELS are still what the decoder predicts, so the sensory question is
untouched. Pooled, PS93/PS94/PS95 all clear -- three animals, which is CLAUDE.md rule 8's bar.

WHAT A POSITIVE RESULT DOES NOT LICENSE. Accuracy is not linear in information, so "40% of the ENL
signal is sensory" is not a sentence this supports; "the position code survives at X% of its
working-trial accuracy, both above null" is. And attention rides with engagement -- a stopped animal
may attend the spout less -- so a reduced-but-present code could be attenuated sensory rather than
sensory-minus-plan. Nothing here separates those two.

CLI::

    python -m wfield_local.enl_decode --epoch pre                  # the claim
    python -m wfield_local.enl_decode --epoch pre --transfer       # + readout 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import nolick_analysis as na

#: Pre-cue, always -- the ENL is the pre-cue period by definition. `enl_states.align()` says the
#: same thing for the amplitude arm; both are stated rather than defaulted so a caller cannot
#: quietly score this on a cue-aligned window that contains the response.
ALIGN = "precue"

#: The basis for the POOLED analysis, and it cannot be plain `locanmf`. LocaNMF components are
#: SESSION-SPECIFIC, so their count differs from day to day -- PS95's pre-stroke sessions gave 568
#: features on one day and 492 on another, and `pool_arms` refused to concatenate them. CLAUDE.md
#: names this "the cross-day problem" and gives two answers; this uses the better one.
#:
#: **THE PER-ANIMAL FROZEN JOINT BASIS** (`joint_locanmf.load(animal)`, ~87-95 components) is one
#: fixed set of footprints for that animal: a session in the fit keeps its fitted time courses and
#: any other is PROJECTED onto the same frozen footprints, never refitted. That makes features
#: comparable across days AND keeps the reference frame fixed before the manipulation, which is the
#: frozen-model discipline the post-stroke arm depends on (rule 10).
#:
#: ROI is the registration-free alternative and is the FALLBACK, not the default: Allen regions are
#: far coarser than ~90 components, and the first PS95 run under ROI put both within-arm decodes at
#: null while a transfer read the same stopped trials well above it. A null under a coarse basis is
#: weak evidence of absence.
POOLED_SOURCE = "joint"

#: Fallback when an animal has no saved joint basis. ROI is coarse -- Allen regions against the
#: basis's ~90 components -- and on PS95 pre-stroke the within-arm decodes sat at null under ROI
#: while a transfer read the same trials at 0.267, so the basis is worth using wherever it exists.
FALLBACK_SOURCE = "roi"

#: The two classes, in the order the ratio is taken: `stopped` over `miss_working`.
ARMS = ("miss_working", "stopped")

#: Below this many trials, a WITHIN-ARM decode is reported as UNDERPOWERED rather than as a result.
#: Not a filter -- the number is still printed, because "we could not test it" and "we tested it and
#: found nothing" are different facts and only one of them is evidence of absence.
#:
#: **RAISED 40 -> 500 ON MEASUREMENT, 2026-09-24.** The first value was a guess and it was far too
#: low: on PS95 pre-stroke the within-arm decodes sat at null with n=283 (`miss_working`, p=0.23)
#: and n=495 (`stopped`, p=0.47), and the check passed both as adequately powered. They are not
#: absent-signal results -- a decoder trained on the 5974 success trials reads position out of those
#: SAME 495 stopped trials at 0.267 against a null of 0.167, p < 0.0005. So the stopped trials carry
#: position information and the within-arm failure is training-set size, which is exactly the
#: distinction this constant exists to protect and exactly the one it got wrong.
#:
#: 500 is the smallest value consistent with what was measured (495 was insufficient); it is a floor
#: below which a within-arm result must not be read as absence, NOT a promise that 500 suffices.
#: **Use the TRANSFER readout at these trial counts** -- it trains on a large arm and tests on the
#: scarce one, so it is the readout that works where a 6-way within-arm fit does not.
MIN_TRIALS = 500


def shared_profile(y_by_arm, labels=None):
    """The position profile both arms are matched to: the ELEMENTWISE MINIMUM share, renormalised.

    Matching arm A to arm B's profile and arm B to its own would compare a subsample against a full
    sample. Matching both to the same target makes the two decodes the same six-way problem, which
    is the only way the ratio in readout 2 means anything.

    The minimum (rather than, say, the mean) is what both arms can actually supply: a target share
    above what an arm has for some position makes `match_profile` bound the whole subsample by that
    position's scarcity, and the draw collapses.
    """
    labels = labels or na.DISPLAY_ORDER
    fr = []
    for y in y_by_arm:
        y = np.asarray(y)
        n = max(int(y.size), 1)
        fr.append(np.array([(y == c).sum() / n for c in labels], float))
    m = np.minimum.reduce(fr)
    return m / m.sum() if m.sum() > 0 else np.full(len(labels), 1.0 / len(labels))


def underpowered(y, min_trials=MIN_TRIALS, labels=None):
    """``(bool, reason)`` -- is a WITHIN-ARM decode on this arm too thin to read as absence?

    Applies to readouts 1 and 2 only. A transfer result is not governed by this: it trains
    elsewhere, so the scarce arm only has to be large enough to SCORE, which needs far fewer trials
    than fitting a six-way decoder on it.
    """
    labels = labels or na.DISPLAY_ORDER
    y = np.asarray(y)
    if y.size < min_trials:
        return True, f"{y.size} trials < {min_trials}"
    present = sum(1 for c in labels if (y == c).sum() > 0)
    if present < len(labels):
        return True, f"only {present}/{len(labels)} positions represented"
    return False, ""


def ratio(stopped_arm, working_arm, den_name="working"):
    """Stopped decoding as a fraction of another arm's, both measured ABOVE THEIR OWN NULL.

    Two things this gets right that the obvious version does not.

    **The baseline is each arm's PERMUTATION NULL, not uniform chance.** `nolick_analysis` exists
    partly to retire the uniform-chance comparison -- it ships the old flag as
    `above_uniform_chance_DEPRECATED`, "the claim this module exists to retire" -- because the
    null moves with the decoder's prediction bias and the arm's position profile, which differ
    between these two arms by construction (a position the animal reaches poorly contributes more
    misses). A ratio taken against a shared 1/6 would charge that difference to the biology.

    Measured on synthetic data while building this: under LABEL SKEW ALONE the BALANCED null stays
    at ~1/6, because balanced accuracy is macro-averaged recall and is robust to imbalance by
    construction; it is the RAW null that rises (`nolick_analysis`'s 0.211-vs-0.167 figure for PS93
    is the raw one). The balanced null moves when the PREDICTOR is biased, which is the case here
    and is exactly what shuffling labels with predictions held fixed is designed to capture. Using
    each arm's own measured null rather than reasoning about which effect dominates is the point.

    **The guard is `above_null_balanced`, not a positive numerator.** A first version tested
    ``working - chance > 0`` and returned a ratio of 100 for an arm sitting AT chance, because
    ``0.167 - 1/6`` is a tiny positive rather than zero. A fraction of a quantity that is not
    itself established is not a number, so the denominator has to have cleared its own null.

    Returns ``(value, note)``; ``value`` is None when the ratio is undefined and ``note`` says why.
    """
    if not working_arm.get("above_null_balanced"):
        return None, (f"{den_name} arm is not above its own null -- there is no established effect "
                      "to take a fraction of")
    w = working_arm["balanced_accuracy"] - working_arm["bal_null_mean"]
    s_ = stopped_arm["balanced_accuracy"] - stopped_arm["bal_null_mean"]
    if not np.isfinite(w) or w <= 0:
        return None, f"{den_name} arm at or below its null"
    note = "" if stopped_arm.get("above_null_balanced") else         "stopped arm is NOT above its own null -- the ratio sizes an effect that is not established"
    return float(s_ / w), note


def arms_for_session(s, align=ALIGN, source=POOLED_SOURCE, post_s=2.0, basis=None):
    """``{arm: {X, y, g}}`` for one session -- the SHARED categorisation, re-split, not a new one.

    `nolick_decoder.session_features` already returns `sess_eng` per trial, computed from
    `precue_engagement_states.engagement_gate` in the imaging universe. So the two classes this
    module needs are a re-split of a category that exists rather than a parallel path, and
    `enl_states.states_for` must never be used to build features -- it works on the behaviour
    table and would be a second definition of the same quantity (rule 9).

    **`late_rewarded` IS EXCLUDED, and that is a correction to this analysis's first design.**
    A trial where the animal licked after `max_rt` but still inside the response window is a HIT:
    the movement happened and reward arrived. `enl_states.states_for` scores it `working`, because
    by the decoder's rule it is not a success and the animal was still engaged -- which quietly
    puts a rewarded, executed trial in the class whose whole purpose is to be matched to `stopped`
    on outcome. `nolick_decoder` splits it out and this uses that split.

        miss_working   undetected + engaged      no lick anywhere in the response window
        stopped        undetected + not engaged  the same, inside the terminal collapse
    """
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.nolick_decoder import _joint_signal, session_features

    args = _args(source=(FALLBACK_SOURCE if source == "joint" else source),
                 align=align, post_s=post_s)
    if basis is not None:
        # Projected onto the animal's FROZEN footprints -- `_joint_signal` keeps a fitted session's
        # own time courses and projects any other, and never refits. `session_features` takes the
        # injected signal by the same route `locanmf_position_decoder._trial_features` does, so
        # everything downstream is identical by construction and the basis is the only difference.
        sig, regs, _vc = _joint_signal(basis, s)
        F, _feat = session_features(s, args, signal=sig, feat_region=regs)
    else:
        F, _feat = session_features(s, args)
    u = F["undetected"]
    out = {}
    for arm, m in (("miss_working", u["sess_eng"]), ("stopped", ~u["sess_eng"])):
        out[arm] = {"X": u["X"][m] if u["y"].size else u["X"],
                    "y": u["y"][m] if u["y"].size else u["y"],
                    "g": u["g"][m] if u["y"].size else u["g"],
                    # carried through so `enl_lick_control` splits on the SAME per-trial flag the
                    # window builder computed, rather than rebuilding one alongside it (rule 9, and
                    # the failure mode `locanmf_position_decoder` records for externally rebuilt
                    # masks: one came out 633 long against 575 kept trials)
                    "lead_lick": u["lead_lick"][m] if u["y"].size else u["lead_lick"]}
    out["success"] = {"X": F["engaged"]["X"], "y": F["engaged"]["y"], "g": F["engaged"]["g"],
                      "lead_lick": F["engaged"]["lead_lick"]}
    return out


def pool_arms(per_session):
    """Concatenate per-session arms, keeping BLOCK GROUPS UNIQUE ACROSS SESSIONS.

    Block ids restart at 0 every session, so concatenating them unchanged would let `GroupKFold`
    put block 3 of one session and block 3 of another in the same fold -- which is not leakage of
    trials but is leakage of session, and a session is exactly the level the decoder is supposed to
    generalise across. Offsetting makes every block globally distinct.
    """
    out = {}
    for arm in ("miss_working", "stopped", "success"):
        Xs, ys, gs, ll, off = [], [], [], [], 0
        for d in per_session:
            a = d.get(arm)
            if a is None or not len(a["y"]):
                continue
            Xs.append(a["X"]); ys.append(a["y"]); gs.append(np.asarray(a["g"]) + off)
            ll.append(np.asarray(a.get("lead_lick", np.zeros(len(a["y"]), bool)), bool))
            off += int(np.asarray(a["g"]).max()) + 1
        out[arm] = ({"X": np.concatenate(Xs), "y": np.concatenate(ys), "g": np.concatenate(gs),
                     "lead_lick": np.concatenate(ll)}
                    if Xs else {"X": np.empty((0, 0)), "y": np.array([], int),
                                "g": np.array([], int), "lead_lick": np.array([], bool)})
    return out


def decode_within(arm, n_splits=5):
    """Block-CV predictions for one arm, or None if it cannot be cross-validated.

    `GroupKFold` on the position BLOCK, the same grouping the production decoder uses: adjacent
    trials share a spout position, so a random split would put near-duplicate trials on both sides
    and report a generalisation score that is partly memorisation.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict

    from wfield_local.locanmf_frozen_decoder import _pipe

    y, g = np.asarray(arm["y"]), np.asarray(arm["g"])
    if y.size == 0 or np.unique(y).size < 2:
        return None
    ng = min(n_splits, int(np.unique(g).size))
    if ng < 2:
        return None
    return cross_val_predict(_pipe(), arm["X"], y, cv=GroupKFold(ng), groups=g)


def transfer(train_arm, test_arm):
    """Readout 3: fit on ``train_arm``, predict ``test_arm``. Predictions, or None."""
    from wfield_local.locanmf_frozen_decoder import _pipe

    if not len(train_arm["y"]) or not len(test_arm["y"]):
        return None
    if np.unique(train_arm["y"]).size < 2:
        return None
    return _pipe().fit(train_arm["X"], train_arm["y"]).predict(test_arm["X"])


def shared_decoder(train, tests, n_splits=5):
    """Readout 4. ONE decoder, one training set, block held out, scoring SEVERAL test arms.

    This is the readout that makes a ratio mean something at these trial counts, and it exists
    because readout 2 could not be computed. Readout 2 fits a separate decoder on each arm and then
    divides, so its denominator and numerator differ in TRAINING-SET SIZE as well as in biology --
    and on PS95 pre-stroke both fits were too thin to clear their own nulls (n=283 and n=495), so
    the ratio was undefined while a transfer read those SAME stopped trials well above null. The
    within-arm nulls were a statement about how many trials a six-way fit needs, not about cortex.

    So: train once on the LARGE arm (`success`, ~6k trials), and in each fold score every test arm's
    trials that live in the HELD-OUT BLOCKS. Because the decoder and its training data are identical
    across test arms, the only thing that differs between them is the trials being read, which is
    the comparison the question actually asks for.

    **Blocks are held out across ALL arms at once, not just the training one.** A stopped trial and
    a success trial from the same block share a spout position and sit seconds apart; scoring a
    stopped trial with a model that trained on success trials from its own block would report
    memorisation. Partitioning by block over the union means each trial in any arm is predicted by
    exactly the fold in which its block was held out -- so `success` here is a genuine cross-
    validated score, directly comparable to the other arms rather than a training-set fit.

    Two ratios follow, and the OUTCOME-MATCHED one is the primary:

        stopped / miss_working   both are no-lick trials, so outcome, reward and movement are
                                 matched and only engagement differs. This is the sensory-vs-plan
                                 fraction.
        stopped / success        the ceiling: what this basis and window deliver when position
                                 information is present AND a plan was formed AND reward followed.
                                 A scale reference, not an outcome-matched contrast.

    Returns ``(preds, coverage)``. ``preds[name]`` is an object array aligned to that arm's ``y``
    with None where a trial's block never appeared as a held-out block (possible for a test arm
    whose blocks are absent from the training arm); ``coverage[name]`` is the fraction predicted, so
    a thinly-covered arm is visible rather than silently scored on a subset.
    """
    from sklearn.model_selection import GroupKFold

    from wfield_local.locanmf_frozen_decoder import _pipe

    Xtr, ytr = np.asarray(train["X"]), np.asarray(train["y"])
    gtr = np.asarray(train["g"])
    if ytr.size == 0 or np.unique(ytr).size < 2:
        return None, None
    ng = min(n_splits, int(np.unique(gtr).size))
    if ng < 2:
        return None, None

    preds = {k: np.full(np.asarray(v["y"]).size, None, dtype=object) for k, v in tests.items()}
    for tr, te in GroupKFold(ng).split(Xtr, ytr, groups=gtr):
        if np.unique(ytr[tr]).size < 2:
            continue
        held = np.unique(gtr[te])
        fit = _pipe().fit(Xtr[tr], ytr[tr])
        for k, v in tests.items():
            sel = np.flatnonzero(np.isin(np.asarray(v["g"]), held))
            if sel.size:
                preds[k][sel] = fit.predict(np.asarray(v["X"])[sel])
    cover = {k: (float((p != None).mean()) if p.size else 0.0)  # noqa: E711 -- object array
             for k, p in preds.items()}
    return preds, cover


def common_positions(arms, labels=None):
    """The positions present in EVERY arm -- the only label set on which a ratio is like-for-like.

    `nolick_analysis.balanced_accuracy` is macro-recall over the classes PRESENT IN ``y_true``, and
    absent classes are skipped rather than scored zero (deliberately: with a skewed no-lick set,
    counting a position the animal never saw as a failure would penalise it for what it declined).
    The consequence for a RATIO is that two arms with different sets of present positions are
    averages over different things, so dividing them charges the difference in label set to the
    biology.

    It is not hypothetical. PS92 pre-stroke has SIX stopped trials, and its within-arm balanced null
    printed as **0.333** rather than ~1/6 -- the arm had about three positions in it, so the metric
    was a three-way macro-recall being divided by a six-way one. Restricting both arms to the
    positions they share makes the comparison one quantity again, and `positions_dropped` records
    what that cost.
    """
    labels = labels or na.DISPLAY_ORDER
    return [c for c in labels
            if all((np.asarray(a["y"]) == c).any() for a in arms)]


def scored_arm(arm, pred, labels=None):
    """The trials of one shared-decoder arm that are actually scorable: ``{y, pred, g}`` or None.

    Two restrictions, both recorded rather than silent: to trials whose BLOCK fell in a held-out
    fold, and to the positions in ``labels`` (`common_positions`). Predictions OUTSIDE ``labels``
    are kept as predictions -- they simply count as errors -- so restricting the label set never
    flatters the decoder by hiding its confusions.

    Split out from the scoring so the bootstrap resamples EXACTLY the trials that were scored. Two
    separate mask computations would be two definitions of "the scored set" (rule 9).
    """
    labels = list(labels or na.DISPLAY_ORDER)
    y = np.asarray(arm["y"])
    if pred is None or not y.size or len(labels) < 2:
        return None
    m = np.flatnonzero((pred != None) & np.isin(y, labels))                  # noqa: E711
    if m.size == 0:
        return None
    return {"y": y[m], "pred": np.asarray(pred[m], dtype=y.dtype),
            "g": np.asarray(arm["g"])[m], "n_total": int(y.size),
            "coverage": float(np.mean(pred != None))}                        # noqa: E711


def _score_shared(sc, target_frac, n_perm, labels):
    """`evaluate_arm` on a `scored_arm`, or a reason the arm was not scorable."""
    if sc is None:
        return {"skipped": "no trial fell in a held-out block at a shared position"}
    out = na.evaluate_arm(sc["y"], sc["pred"], target_frac=target_frac, n_perm=n_perm,
                          labels=list(labels))
    out["n_total"] = sc["n_total"]
    out["coverage"] = sc["coverage"]
    out["positions_used"] = len(labels)
    return out


def bootstrap_arms(scored, nulls, labels, n_boot=2000, seed=0, alpha=0.05, pairs=None):
    """PAIRED cluster bootstrap over blocks -- CIs on each arm, and on the DIFFERENCES between arms.

    Everything reported before this was an arm against its OWN permutation null, which answers "is
    there a position code here" and says nothing about whether two arms DIFFER. "`miss_working` is
    above null but well below `success`" needs a test of the gap, and a ratio quoted without an
    interval invites a 0.42-vs-0.80 spread across two animals to be read as a real difference.

    **BLOCKS ARE THE RESAMPLING UNIT, NOT TRIALS.** `decode_ci` records why: the task presents
    positions in runs of ~6, so resampling trials would treat six correlated observations as six
    independent ones and return intervals that are far too narrow. `n_effective` is the honest
    sample size and it is much smaller than the trial count.

    **THE DRAW IS PAIRED ACROSS ARMS**, and that is the part `decode_ci.bootstrap_recall` has no
    need for. The three arms come from the same sessions and the SAME BLOCKS, so their accuracies
    are correlated; resampling each arm independently would inflate the interval on a difference by
    discarding that correlation. One draw of block ids is applied to every arm at once.

    The permutation nulls are held FIXED at their measured means rather than recomputed per draw --
    2000 bootstraps x 2000 permutations is not affordable, and the null is a property of the
    predictor and the label distribution rather than of the resample. It means the ratio intervals
    below carry the bootstrap's uncertainty and not the null's.

    Ratios are computed per draw, so a draw whose DENOMINATOR falls to or below its null yields no
    ratio; `frac_undefined` reports how often that happened. A ratio whose interval rests on a
    denominator that keeps collapsing is not a measurement, and that has to be visible.
    """
    # Generic over arm names so `enl_lick_control` can pass its clean/lead arms. Insertion order is
    # kept, and for the canonical {success, miss_working, stopped} the derived pairs are exactly the
    # three this used to hardcode.
    names = [k for k, v in scored.items() if v is not None]
    if len(names) < 2:
        return {"skipped": "fewer than two scorable arms"}
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    units = np.unique(np.concatenate([scored[k]["g"] for k in names]))
    idx = {k: {u: np.flatnonzero(scored[k]["g"] == u) for u in units} for k in names}
    rng = np.random.default_rng(seed)

    bal = {k: np.full(n_boot, np.nan) for k in names}
    for b in range(n_boot):
        pick = rng.choice(units, size=units.size, replace=True)
        for k in names:
            sel = np.concatenate([idx[k][u] for u in pick]) if units.size else np.array([], int)
            if sel.size:
                bal[k][b] = na.balanced_accuracy(scored[k]["y"][sel], scored[k]["pred"][sel],
                                                 list(labels))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {"n_boot": int(n_boot), "n_effective": int(units.size), "alpha": alpha,
           "arms": {k: {"ci": [float(np.nanpercentile(bal[k], lo)),
                               float(np.nanpercentile(bal[k], hi))]} for k in names}}

    out["differences"], out["ratios"] = {}, {}
    for a, b_ in pairs:
        if a not in names or b_ not in names:
            continue
        d = bal[a] - bal[b_]                                  # positive => the first arm is higher
        good = np.isfinite(d)
        # two-sided bootstrap p for "the arms do not differ"
        p = 2 * min(float(np.mean(d[good] <= 0)), float(np.mean(d[good] >= 0))) if good.any() else 1.0
        out["differences"][f"{a}_minus_{b_}"] = {
            "value": float(np.nanmean(d)),
            "ci": [float(np.nanpercentile(d, lo)), float(np.nanpercentile(d, hi))],
            "p": float(min(1.0, p))}

        num, den = bal[b_] - nulls[b_], bal[a] - nulls[a]     # scarcer arm over richer arm
        r = np.where(den > 0, num / np.where(den > 0, den, np.nan), np.nan)
        out["ratios"][f"{b_}_over_{a}"] = {
            "ci": [float(np.nanpercentile(r, lo)), float(np.nanpercentile(r, hi))]
            if np.isfinite(r).any() else [float("nan")] * 2,
            "frac_undefined": float(np.mean(~np.isfinite(r)))}
    return out


def analyse(pooled, n_perm=na.N_PERM, do_transfer=True, n_boot=2000):
    """The four readouts for one pooled animal x epoch. Never raises on a thin arm; reports it."""
    res = {"n": {a: int(len(pooled[a]["y"])) for a in ("miss_working", "stopped", "success")}}
    for arm in ARMS:
        under, why = underpowered(pooled[arm]["y"])
        res.setdefault("power", {})[arm] = {"underpowered": under, "reason": why}

    tf = shared_profile([pooled[a]["y"] for a in ARMS])
    for arm in ARMS:                                            # readouts 1 and 2
        pred = decode_within(pooled[arm])
        res[arm] = (na.evaluate_arm(pooled[arm]["y"], pred, target_frac=tf, n_perm=n_perm)
                    if pred is not None else {"n": int(len(pooled[arm]["y"])),
                                              "skipped": "not cross-validatable"})
    if "skipped" not in res["stopped"] and "skipped" not in res["miss_working"]:
        val, note = ratio(res["stopped"], res["miss_working"])
        res["ratio"] = {"value": val, "note": note}

    if do_transfer:                                             # readout 3
        for src in ("miss_working", "success"):
            pred = transfer(pooled[src], pooled["stopped"])
            res[f"transfer_{src}_to_stopped"] = (
                na.evaluate_arm(pooled["stopped"]["y"], pred, target_frac=tf, n_perm=n_perm)
                if pred is not None else {"skipped": "no usable training arm"})

    # Readout 4: the like-for-like ratio. Profile matched across ALL THREE arms it compares, not
    # just the two in ARMS -- otherwise the success arm is scored on its own (much richer) profile
    # and part of the ratio is that difference rather than the biology.
    names = ("success", "miss_working", "stopped")
    lab4 = common_positions([pooled[a] for a in names])
    tf4 = shared_profile([pooled[a]["y"] for a in names], labels=lab4)
    preds, cover = shared_decoder(pooled["success"], {a: pooled[a] for a in names})
    scored = {a: (scored_arm(pooled[a], preds[a], labels=lab4) if preds is not None else None)
              for a in names}
    res["shared"] = {a: _score_shared(scored[a], tf4, n_perm, lab4) for a in names}
    res["shared_coverage"] = cover or {}
    res["shared_positions"] = {
        "used": [na.POSITION_NAMES[c] for c in lab4],
        "dropped": [na.POSITION_NAMES[c] for c in na.DISPLAY_ORDER if c not in lab4]}
    # THE LADDER. Each rung removes one thing while the SPOUT STAYS IN POSITION throughout, so the
    # sensory drive is common to all three arms and every ratio is asking what the removed component
    # was contributing.
    #
    #   success        sensory + plan + execution + reward
    #   miss_working   sensory + engagement, no movement    <- `working_vs_success` sizes this step
    #   stopped        sensory, no engagement, no movement  <- `vs_working` sizes this one
    #
    # `working_vs_success` is the rung that survives the trial counts: `miss_working` clears its own
    # null in every animal while `stopped` does so in two, so it is the only ratio here that reaches
    # rule 8's three-animal bar.
    #
    # **BUT `success` IS THE TRAINING ARM**, and that is not symmetric. Block hold-out makes its
    # score a genuine cross-validated one, yet the decoder is still FIT TO SUCCESS-TRIAL STATISTICS
    # and the other two arms are scored out of distribution. Any shift between arms -- overall
    # activity, noise, hemodynamics -- costs accuracy on its own, so **every ratio with `success` in
    # the denominator is biased DOWNWARD** and reads as a lower bound.
    #
    # Which is the argument for `vs_working` being the primary readout: its numerator and
    # denominator are BOTH out of distribution relative to the training set, so the domain-shift
    # cost largely cancels between them. It is also why `vs_success` comes out so much lower than
    # `vs_working` on the same animal, and that gap should not be read as biology.
    for key, num, den in (("clean_ratio_vs_working", "stopped", "miss_working"),
                          ("clean_ratio_vs_success", "stopped", "success"),
                          ("clean_ratio_working_vs_success", "miss_working", "success")):
        n_, d_ = res["shared"][num], res["shared"][den]
        if "skipped" in n_ or "skipped" in d_:
            res[key] = {"value": None, "note": "an arm was not scored"}
        else:
            val, note = ratio(n_, d_, den_name=den)
            res[key] = {"value": val, "note": note, "numerator": num, "denominator": den,
                        "training_arm_denominator": den == "success"}

    nulls = {a: res["shared"][a].get("bal_null_mean", float("nan")) for a in names}
    res["bootstrap"] = bootstrap_arms({a: scored[a] for a in names}, nulls, lab4, n_boot=n_boot)
    return res


def sessions_for(animal, epoch):
    """Curated sessions for one animal x epoch, in `load_sessions()` order.

    ORDER IS PRESERVED, NOT SORTED. `analysis_kit.curated_sessions` keeps `load_sessions()`'s order
    deliberately -- rule 9 records that "tidying" it moved every published CI, because a bootstrap
    pool built by iterating it draws differently under the same seed. Nothing here bootstraps yet,
    but the pooled feature matrix is built in this order and the CV folds follow from it.
    """
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs

    return [s for s in ak.curated_sessions()
            if s["label"].startswith(animal) and epochs.epoch_of(s["label"]) == epoch]


def _session_arms(item):
    """Module-level worker (rule 6: spawn pickles by name). One session -> its arms, or None."""
    from wfield_local import config, enl_decode

    s = next((x for x in config.load_sessions() if x["label"] == item["label"]), None)
    if s is None:
        return None
    try:
        return enl_decode.arms_for_session(s, align=item["align"], source=item["source"],
                                           post_s=item["post_s"])
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"[:140]}


def analyse_animal(animal, epoch="pre", *, source=POOLED_SOURCE, post_s=2.0, n_perm=na.N_PERM,
                   do_transfer=True, jobs=None, verbose=True):
    """The three readouts for one animal x epoch, positions POOLED. Returns a result dict.

    Positions are pooled in the sense that matters: per-position effect sizes are not reported, and
    the position LABELS are still what the decoder predicts. See the module docstring for why that
    leaves the sensory question intact while making the pre-stroke cell answerable at all.
    """
    from wfield_local import analysis_kit as ak

    labs = [s["label"] for s in sessions_for(animal, epoch)]
    if not labs:
        return {"animal": animal, "epoch": epoch, "skipped": "no curated sessions"}
    basis, per, errs = None, [], []
    if source == "joint":
        from wfield_local import joint_locanmf
        try:
            basis = joint_locanmf.load(animal)
        except Exception as exc:
            errs.append(f"no joint basis for {animal} ({type(exc).__name__}) -- "
                        f"falling back to {FALLBACK_SOURCE}")
            source = FALLBACK_SOURCE

    if basis is not None:
        # SERIAL on purpose. The basis is one shared object served from MICROSCOPE; handing it to
        # eight spawned workers would pickle it eight times and read it over the network eight
        # times, for a loop that costs ~9 s a session. Rule 6 says fan a per-session loop out, and
        # it also says the unit is the caller's judgement -- here the shared object is the cost.
        from wfield_local import config
        for lab in labs:
            sess = next((x for x in config.load_sessions() if x["label"] == lab), None)
            if sess is None:
                errs.append(f"{lab}: no session record")
                continue
            try:
                per.append(arms_for_session(sess, align=ALIGN, source=source, post_s=post_s,
                                            basis=basis))
            except Exception as exc:
                errs.append(f"{lab}: {type(exc).__name__}: {exc}"[:140])
    else:
        items = [{"label": lab, "align": ALIGN, "source": source, "post_s": post_s} for lab in labs]
        res, fail = ak.fan_sessions(items, _session_arms, jobs=jobs, key=lambda it: it["label"])
        errs += [f["label"] if isinstance(f, dict) else str(f) for f in fail]
        for item, val in res:
            if val is None:
                errs.append(f"{item['label']}: no session record")
            elif "__error__" in val:
                errs.append(f"{item['label']}: {val['__error__']}")
            else:
                per.append(val)
    if not per:
        return {"animal": animal, "epoch": epoch, "skipped": "no usable sessions", "errors": errs}

    pooled = pool_arms(per)
    out = analyse(pooled, n_perm=n_perm, do_transfer=do_transfer)
    out.update({"animal": animal, "epoch": epoch, "align": ALIGN, "source": source,
                "basis_id": getattr(basis, "basis_id", None),
                "ncomp": getattr(basis, "ncomp", None),
                "n_sessions": len(per), "sessions": labs, "errors": errs})
    if verbose:
        report(out)
    return out


def report(r, fh=None):
    """Human-readable summary. Prints the POWER verdict before the numbers, deliberately.

    Rule 8's habit, applied to a single cell: an arm that could not be tested and an arm that was
    tested and found nothing print differently, because only the second is evidence of absence.
    """
    pr = (lambda *a: print(*a, file=fh, flush=True)) if fh else (lambda *a: print(*a, flush=True))
    pr("")
    bas = ("" if not r.get("basis_id") else f" {str(r['basis_id'])[:12]} {r.get('ncomp')}c")
    pr(f"=== {r.get('animal')} {r.get('epoch')} "
       f"[{r.get('align')}/{r.get('source')}{bas}] "
       f"{r.get('n_sessions', 0)} session(s) ===")
    if r.get("skipped"):
        pr(f"  SKIPPED: {r['skipped']}")
        return
    for e in r.get("errors", [])[:5]:
        pr(f"  ! {e}")
    pr("  trials: " + "  ".join(f"{k}={v}" for k, v in r["n"].items()))
    for arm in ARMS:
        p_ = r["power"][arm]
        if p_["underpowered"]:
            pr(f"  {arm:13s} UNDERPOWERED ({p_['reason']}) -- not evidence of absence")
    for arm in ARMS:
        d = r.get(arm, {})
        if "skipped" in d:
            pr(f"  {arm:13s} skipped: {d['skipped']}")
            continue
        flag = "ABOVE NULL" if d["above_null_balanced"] else "at null"
        pr(f"  {arm:13s} n={d['n']:4d}  bal={d['balanced_accuracy']:.3f}  "
           f"null={d['bal_null_mean']:.3f}  p={d['bal_p']:.4f}  {flag}")
    if "ratio" in r:
        v, note = r["ratio"]["value"], r["ratio"]["note"]
        pr(f"  ratio (stopped/working, above own nulls): "
           f"{'undefined' if v is None else f'{v:.3f}'}")
        if note:
            pr(f"      {note}")
    for k in ("transfer_miss_working_to_stopped", "transfer_success_to_stopped"):
        d = r.get(k, {})
        if d and "skipped" not in d:
            pr(f"  {k:34s} bal={d['balanced_accuracy']:.3f} null={d['bal_null_mean']:.3f} "
               f"p={d['bal_p']:.4f} {'ABOVE NULL' if d['above_null_balanced'] else 'at null'}")
    if "shared" in r:
        pr("  -- readout 4: ONE success-trained decoder, blocks held out across all arms --")
        sp = r.get("shared_positions", {})
        if sp.get("dropped"):
            pr(f"  POSITIONS RESTRICTED to {len(sp['used'])}/6 shared by every arm; "
               f"dropped {', '.join(sp['dropped'])} -- the ratio is NOT a six-way comparison")
        for a in ("success", "miss_working", "stopped"):
            d = r["shared"].get(a, {})
            if "skipped" in d:
                pr(f"  {a:13s} skipped: {d['skipped']}")
                continue
            pr(f"  {a:13s} n={d['n']:4d}/{d['n_total']:4d}  bal={d['balanced_accuracy']:.3f}  "
               f"null={d['bal_null_mean']:.3f}  p={d['bal_p']:.4f}  "
               f"{'ABOVE NULL' if d['above_null_balanced'] else 'at null'}")
        bs = r.get("bootstrap", {})
        for a in ("success", "miss_working", "stopped"):
            ci = bs.get("arms", {}).get(a, {}).get("ci")
            if ci:
                pr(f"  {a:13s} 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
                   f"(paired block bootstrap, n_eff={bs.get('n_effective')} blocks)")
        for k, d in bs.get("differences", {}).items():
            star = "DIFFER" if d["p"] < 0.05 else "not distinguishable"
            pr(f"  diff {k:28s} {d['value']:+.3f}  CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  "
               f"p={d['p']:.4f}  {star}")
        bratio = bs.get("ratios", {})
        for key, num, den, lab in (
                ("clean_ratio_vs_working", "stopped", "miss_working", "OUTCOME-MATCHED"),
                ("clean_ratio_vs_success", "stopped", "success", "ceiling; den is TRAINING arm"),
                ("clean_ratio_working_vs_success", "miss_working", "success",
                 "ceiling; den is TRAINING arm")):
            d = r.get(key, {})
            v = d.get("value")
            ci = bratio.get(f"{num}_over_{den}", {})
            ci_s = ("" if not ci.get("ci") or not np.isfinite(ci["ci"][0]) else
                    f"  CI [{ci['ci'][0]:.3f}, {ci['ci'][1]:.3f}]")
            und = ci.get("frac_undefined", 0.0)
            pr(f"  ratio {num[:4]}/{den[:4]} {lab:28s} "
               f"{'undefined' if v is None else f'{v:.3f}'}{ci_s}"
               + (f"  [{und:.0%} of draws undefined]" if und > 0.02 else ""))
            if d.get("note"):
                pr(f"      {d['note']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", default="pre", help="pre / acute / subacute / chronic")
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--transfer", action="store_true", help="also run readout 3")
    ap.add_argument("--n-perm", type=int, default=na.N_PERM)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default=POOLED_SOURCE)
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    animals = args.animal or ["PS92", "PS93", "PS94", "PS95"]
    results = [analyse_animal(a, args.epoch, source=args.source, post_s=args.post_s,
                              n_perm=args.n_perm, do_transfer=args.transfer, jobs=args.jobs)
               for a in animals]
    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
        print("")
        print(f"[enl_decode] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
