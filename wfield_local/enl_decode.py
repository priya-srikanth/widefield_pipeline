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

THREE READOUTS, WEAKEST TO STRONGEST:

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

import numpy as np

from wfield_local import nolick_analysis as na

#: Pre-cue, always -- the ENL is the pre-cue period by definition. `enl_states.align()` says the
#: same thing for the amplitude arm; both are stated rather than defaulted so a caller cannot
#: quietly score this on a cue-aligned window that contains the response.
ALIGN = "precue"

#: The two classes, in the order the ratio is taken: `stopped` over `miss_working`.
ARMS = ("miss_working", "stopped")

#: Below this many trials in an arm, a decode is reported as UNDERPOWERED rather than as a result.
#: Not a filter -- the number is still printed, because "we could not test it" and "we tested it and
#: found nothing" are different facts and only one of them is evidence of absence.
MIN_TRIALS = 40


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
    """``(bool, reason)`` -- is this arm too thin to be read as evidence of absence?"""
    labels = labels or na.DISPLAY_ORDER
    y = np.asarray(y)
    if y.size < min_trials:
        return True, f"{y.size} trials < {min_trials}"
    present = sum(1 for c in labels if (y == c).sum() > 0)
    if present < len(labels):
        return True, f"only {present}/{len(labels)} positions represented"
    return False, ""


def ratio(stopped_arm, working_arm):
    """Stopped decoding as a fraction of working decoding, both measured ABOVE THEIR OWN NULL.

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
        return None, ("working arm is not above its own null -- there is no established effect to "
                      "take a fraction of")
    w = working_arm["balanced_accuracy"] - working_arm["bal_null_mean"]
    s_ = stopped_arm["balanced_accuracy"] - stopped_arm["bal_null_mean"]
    if not np.isfinite(w) or w <= 0:
        return None, "working arm at or below its null"
    note = "" if stopped_arm.get("above_null_balanced") else         "stopped arm is NOT above its own null -- the ratio sizes an effect that is not established"
    return float(s_ / w), note


def arms_for_session(s, align=ALIGN, source="locanmf", post_s=2.0):
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
    from wfield_local.nolick_decoder import session_features

    F, _feat = session_features(s, _args(source=source, align=align, post_s=post_s))
    u = F["undetected"]
    out = {}
    for arm, m in (("miss_working", u["sess_eng"]), ("stopped", ~u["sess_eng"])):
        out[arm] = {"X": u["X"][m] if u["y"].size else u["X"],
                    "y": u["y"][m] if u["y"].size else u["y"],
                    "g": u["g"][m] if u["y"].size else u["g"]}
    out["success"] = {"X": F["engaged"]["X"], "y": F["engaged"]["y"], "g": F["engaged"]["g"]}
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
        Xs, ys, gs, off = [], [], [], 0
        for d in per_session:
            a = d.get(arm)
            if a is None or not len(a["y"]):
                continue
            Xs.append(a["X"]); ys.append(a["y"]); gs.append(np.asarray(a["g"]) + off)
            off += int(np.asarray(a["g"]).max()) + 1
        out[arm] = ({"X": np.concatenate(Xs), "y": np.concatenate(ys), "g": np.concatenate(gs)}
                    if Xs else {"X": np.empty((0, 0)), "y": np.array([], int), "g": np.array([], int)})
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


def analyse(pooled, n_perm=na.N_PERM, do_transfer=True):
    """The three readouts for one pooled animal x epoch. Never raises on a thin arm; reports it."""
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
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", default="pre", help="pre / acute / subacute / chronic")
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--transfer", action="store_true", help="also run readout 3")
    ap.add_argument("--n-perm", type=int, default=na.N_PERM)
    ap.add_argument("--out", default=None)
    ap.parse_args(argv)
    raise SystemExit("enl_decode: the session-loading driver is not wired yet; the readouts, the "
                     "profile matching and the ratio are. See the module docstring.")


if __name__ == "__main__":
    raise SystemExit(main())
