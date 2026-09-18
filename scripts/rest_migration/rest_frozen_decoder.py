"""THE DOCKED FROZEN DECODER: does the PRE-STROKE rest code survive, or is it replaced?

`rest_position_decode` established that position is decodable from the inter-trial interval
(93/94 sessions above their own null) and that the cohort trajectory dips and recovers. It cannot
say what recovered. Every one of its decoders is fitted WITHIN the session it scores, so a chronic
session scoring well says only that THAT DAY'S rest carries position -- by whatever code that day
happens to use. Recovery and replacement produce the identical number.

A frozen model separates them. Train on PRE-STROKE rest, freeze, apply to post-stroke rest:

    retained high   -> the pre-stroke rest code SURVIVES and is still read the same way
    retained low BUT per-session decode high -> the code was REPLACED; a new one carries position
    both low        -> rest position information is genuinely lost

The middle row is the one no existing figure can reach, and it is the reason this arm exists.

THE BASIS HAD TO CHANGE, AND THIS IS THE TRAP THIS FILE EXISTS TO AVOID
----------------------------------------------------------------------
`rest_position_decode` builds features from `joint_basis._load_session`, i.e. each session's OWN
SVD temporal components. That is CORRECT there -- a within-session decoder never compares two days
-- and FATAL here. Component *i* of PS93_0606 and component *i* of PS93_0821 are different cortical
patches, so a model frozen in one session's basis and applied in another is not a degraded decoder,
it is a meaningless one, and it would return a low number that reads exactly like a lesion effect.

So this arm uses the persisted JOINT LocaNMF basis (`joint_locanmf.load`), where component *i* is
the same footprint on every day of that animal -- the same basis the task-side frozen decoder uses
(`grant_figures._pooled_bundle`). That also makes the two arms commensurable: same animal, same
footprints, same estimator, only the WINDOW changes (rest interval vs post-cue trial).

WHAT IS HELD FIXED, so the comparison to the task arm and to the state control is honest
-----------------------------------------------------------------------------------------
* BALANCED ACCURACY, the mean of the six row recalls, chance 1/6. Rest-period class balance moves
  with epoch (an acute animal rests more, and not uniformly across positions), and raw accuracy
  would reward that drift. This is the scoring `BEHAVIOURAL_STATE_CONTROL.md` uses.
* Reported as the FRACTION OF ABOVE-CHANCE PERFORMANCE RETAINED, `(acc - chance)/(1 - chance)`
  normalised to the animal's own pre value -- the same normalisation that lets a 6-way position
  problem be compared with a 3-way state one there. Quoting raw accuracies across arms of different
  cardinality is the error that normalisation exists to prevent.
* THE PRE BASELINE IS LEAVE-ONE-SESSION-OUT. A frozen model scored on a session inside its own
  training set is not a baseline, it is a fit, and every post-stroke number would then be compared
  against an inflated reference. This mirrors the task arm's `_collect_5c` discipline exactly.
* VARIABLE-LENGTH WINDOW. Rest periods are not fixed duration (median ~1.1 s, tail to minutes), so
  each period is split into `--bins` equal parts and each part averaged. Feature width is constant
  while the window is not -- which is what lets a 0.8 s period and a 40 s period be one design
  matrix. It is also the rule `rest_position_decode` already uses, so the two arms see the same
  features.
* COUNTED MINIMUMS, stated rather than implied: `--min-periods` usable rest periods per scored
  session and `--min-per-class` per position, in BOTH the training pool and the scored session.
  A session that fails is NAMED in the output, never silently dropped -- an unstated drop is how a
  cohort mean comes to describe a different set of animals than its caption claims.
* WORKING TRIALS ONLY, and only rest periods BETWEEN two trials of the same position (the bracketing
  cue and the next trial_start must agree). Block-boundary periods are a separate question, and
  mixing them in would let "the next position is predictable" masquerade as a persistent trace.

THE NULL IS PERMUTED, NEVER THE ANALYTIC 1/6. Positions come in ~6-trial BLOCKS, so periods are not
independent and 1/6 is the wrong reference for this design (`decode_ci.frozen_ci` makes the same
point for the task arm). Labels are permuted with the model's PREDICTIONS HELD FIXED -- the
repo-wide rule, and the only coherent choice for a frozen model, since refitting per permutation
would be a null for a different estimator.

THE PRIMARY NULL IS `blockperm`, matching `decode_ci.frozen_ci`, because this IS a frozen decoder.
It permutes the BLOCK -> POSITION map with each block left intact, so within-block correlation and
drift stay inside the null. `shift` (`rest_position_decode`'s null) and `trial`
(`nolick_analysis.permutation_null`, the ENL/cue/lick per-session decoders') are computed alongside
and reported, so the three can be compared rather than argued about.

MEASURED, AND IT CORRECTS A CLAIM THIS PROJECT HAS MADE TWICE: a circular shift does NOT "keep
blocks as blocks" once blocks have unequal lengths, which real ones do. Rolling misaligns the
boundaries and leaves only ~31% of blocks internally constant, against 100% for `blockperm` and
~1% for a trial shuffle -- so the shift sits BETWEEN the two and gives a null that is too weak.
A further trap: under a trial shuffle the BALANCED-accuracy null is ~1/ncls by construction,
because balancing removes the very class skew the shuffle carries. A balanced null sitting at 1/6
is therefore not evidence the permutation is working. See `tests/test_rest_frozen_decoder.py`.

WHAT THIS ARM STILL CANNOT SAY. With no spout present, an above-null read is either a persistent
trace of the position just licked at or anticipation of the next one; within a block both point at
the same position. That ambiguity is `rest_position_decode --boundary`'s business and is unchanged
here -- freezing the model does not resolve it.

RUN:  python -m scripts.rest_migration.rest_frozen_decoder [--perm 50] [--bins 4]
                                                           [--animals PS92 ...] [--out DIR]
Writes `rest_frozen_decoder_<variant>.{csv,png}` plus a per-session csv, so no number here lives
only in stdout.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

CHANCE = 1.0 / 6.0


def _frame_samples(mc, fmdir, regime, pco):
    """DAQ sample index of every BLUE frame -- the same mapping `rest_position_decode` uses."""
    if regime == "B":
        fm = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_frame_map.npz"))
        summ = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_summary.json"))
        if not fm or not summ:
            return None
        with open(summ[0]) as fh:
            off = int(json.load(fh)["chosen_exposure_offset"])
        z = np.load(fm[0])
        return pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
    return pco[np.arange(len(pco) // 2) * 2]


def _balanced_accuracy(y_true, y_pred):
    """Mean of the per-class recalls over the classes PRESENT in `y_true`.

    Not sklearn's, deliberately: sklearn warns and returns a value averaged over classes the test
    set does not contain, which silently changes the chance level of the number being reported.
    Averaging over present classes only keeps chance at 1/len(classes present), and the caller
    reports that denominator.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cls = np.unique(y_true)
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in cls])), len(cls)


def _fit_frozen(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    m.fit(X, y)
    return m


def _epoch_dir():
    """The shared `grant_figures/epoch` directory every other epoch figure writes to.

    NOT a local scratch path. These outputs were landing in `E:/cue_lick/rest_migration/`, which is
    this box's disk -- so every rest-arm result was invisible from the other machine and from the
    deck (Priya, 2026-09-16: "our new rest figures should join all our other figures"). Resolved
    through `PathResolver` exactly as `rest_position_vs_drift` does for `epoch_15x`, so it is
    correct on either box rather than correct on the one it was written on.
    """
    from wfield_local.paths import PathResolver
    return Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"


def lick_gap_s(period_start_smp, period_stop_smp, lick_onsets_smp, fs):
    """Seconds from each rest period to the NEAREST DETECTED lick outside it.

    THE CONFOUND THIS MEASURES, AND ITS LIMIT (Priya, 2026-09-16): *"PS92 is VERY licky pre-stroke
    and during recovery so much of the 'rest' probably includes licks without spout contact (due to
    docked spout position)."*

    The lick channel is ANALOG WITH A THRESHOLD (`lick_detection.thresh_upper` on `lick_analog`), so
    it registers CONTACT. Once the spout docks it is out of reach, and an animal that keeps licking
    at nothing produces NO deflection. `lick_buffer_s` is keyed on detected licks, so it cannot
    exclude what was never detected: a very licky animal's "rest" can contain ongoing orofacial
    motor activity that the mask believes is absent. That would inflate rest decoding for that
    animal specifically, and PS92 is both the licky one AND the outlier with the highest frozen
    retained fraction -- the two facts have to be prised apart rather than noted.

    WHAT THIS CAN AND CANNOT DO. Undetected licks are, by construction, invisible to every DAQ
    measure, so no analysis on this side can count them. What it CAN exploit is that continuation
    licking is temporally CLUSTERED: bouts run on past the last contact, so the periods most likely
    to carry undetected licking are those that start soon after a detected one. Stratifying by this
    gap therefore tests the confound without being able to measure it directly -- if a rest position
    signal is really continuation licking, it should live in the short-gap stratum and thin out in
    the long-gap one. A flat profile does not prove the confound absent; it bounds it.

    THE ONLY MEASUREMENT THAT SETTLES IT is tongue tracking from the behaviour cameras (DLC), which
    is registered as parked work. Say so wherever this control is quoted.
    """
    if lick_onsets_smp.size == 0:
        return np.full(len(period_start_smp), np.inf)
    lo = np.sort(np.asarray(lick_onsets_smp, np.int64))
    out = np.empty(len(period_start_smp), float)
    for i, (aa, bb) in enumerate(zip(period_start_smp, period_stop_smp)):
        j = np.searchsorted(lo, aa)
        before = (aa - lo[j - 1]) if j > 0 else np.inf
        k = np.searchsorted(lo, bb, "right")
        after = (lo[k] - bb) if k < lo.size else np.inf
        out[i] = min(before, after) / fs
    return out


def null_labels(y, g, rng, kind):
    """One draw of label-side nulls, with the model's PREDICTIONS HELD FIXED.

    Holding predictions fixed is the repo-wide rule (`nolick_analysis.permutation_null`,
    `decode_ci.frozen_ci`): shuffling labels rather than predictions keeps the decoder's own
    prediction bias intact, so the null inherits the skew instead of being flattered by it. For a
    FROZEN model it is also the only coherent choice -- refitting per permutation would be a null
    for a different estimator.

    THREE CONVENTIONS EXIST IN THIS PROJECT AND THEY ARE NOT INTERCHANGEABLE:

    ``shift``     circular shift of the time-ordered labels -- `rest_position_decode`'s null.
                  Keeps runs as runs, so block structure and slow drift stay INSIDE the null.
    ``blockperm`` permute the BLOCK -> POSITION map, keeping each block intact -- `decode_ci`'s
                  null for the FROZEN task decoders, and the one this arm should be read against
                  since it IS a frozen decoder. Each block carries one position by construction, so
                  this is the honest "activity in a block is unrelated to which position that block
                  was". `decode_ci` warns in terms: trial-level shuffling destroys within-block
                  correlation and UNDERSTATES the null, "the error that makes 1/6 look defensible".
    ``trial``     i.i.d. shuffle -- `nolick_analysis.permutation_null`, used by the per-session
                  ENL/cue/lick decoders where the reported statistic is RAW accuracy and the point
                  is to inherit the class skew (PS93: null 0.211 rather than 0.167).

    NOTE ON BALANCED ACCURACY. Under a trial-level shuffle the balanced-accuracy null is ~1/ncls by
    construction, because balancing removes exactly the class skew the shuffle was there to carry.
    So a balanced null sitting at 1/6 is NOT evidence the permutation is working -- it is what a
    too-weak null looks like too. `blockperm` is what distinguishes them, which is why both are
    computed rather than one being argued for.
    """
    y = np.asarray(y)
    if kind == "shift":
        return np.roll(y, int(rng.integers(1, len(y))))
    if kind == "trial":
        return rng.permutation(y)
    if kind == "blockperm":
        g = np.asarray(g)
        blocks = np.unique(g)
        # each block's position: taken from its first period, since a block carries one position
        lab_of = {b: y[np.flatnonzero(g == b)[0]] for b in blocks}
        shuffled = rng.permutation([lab_of[b] for b in blocks])
        m = dict(zip(blocks, shuffled))
        return np.array([m[b] for b in g])
    raise ValueError(f"unknown null kind {kind!r}")


def matched_frozen(X, y, g, n_target, rng):
    """The frozen model refitted on a SIZE-MATCHED random subset of pre-stroke BLOCKS.

    THE SAME HANDICAP THE TASK ARM'S `5rm` FAMILY EXISTS TO REMOVE, and the rest arm has it worse.
    The frozen model trains on every pre-stroke session of the animal (~10,000 rest periods) and
    the refit on four fifths of one (~300), so the PRE gap is negative -- measured -0.116 here,
    against the task arm's -0.073 post-cue -- purely from training-set size, with no lesion in it.
    A raw gap read against that baseline charges the lesion for a handicap the design imposed.
    Matching leaves WHICH SESSIONS the data came from as the only difference between the arms.

    WHOLE BLOCKS, NOT LOOSE PERIODS -- exactly `grant_figures._matched_frozen`'s rule. Blocks are
    the unit every other resampling here uses, and sampling loose periods would hand the matched
    model a training set with LESS within-block correlation than the refit model's: one difference
    removed, another introduced.

    SEEDED PER SCORED SESSION, not per animal. The task side learned this the hard way -- one seed
    per animal makes `permutation` return the same block ORDER every time, so every session of that
    animal is scored by very nearly the same matched model: one draw presented as many, and one
    unlucky subset biases the whole animal.

    Returns ``(fitted, n_used)`` or None when the subset cannot carry two classes.
    """
    g = np.asarray(g)
    keep = np.zeros(len(y), bool)
    for b in rng.permutation(np.unique(g)):
        keep |= (g == b)
        if keep.sum() >= n_target:
            break
    if keep.sum() < 2 or len(np.unique(np.asarray(y)[keep])) < 2:
        return None
    return _fit_frozen(X[keep], np.asarray(y)[keep]), int(keep.sum())


def refit_predictions(X, y, g):
    """Within-session block-CV predictions on the SAME periods the frozen model just scored.

    THIS IS WHAT SEPARATES RECOVERY FROM REPLACEMENT, and it has to be paired to do so. The frozen
    arm asks "does the PRE-STROKE code survive"; this asks "is position recoverable AT ALL from this
    session's rest, by any linear readout". Read together:

        frozen low, refit low    -> rest position information is genuinely DEGRADED
        frozen low, refit high   -> the information is PRESENT and the pre-stroke readout no longer
                                    points at it: REPLACEMENT, not recovery
        frozen high              -> the pre-stroke code survives

    Comparing the frozen number against `rest_position_decode`'s published per-session values would
    NOT do this. That family scores different periods, in each session's OWN SVD basis, under a
    different normalisation -- three differences between the two halves of one contrast. Here both
    arms see one X, one y, one block vector, so the difference is the estimator and nothing else.

    BLOCK-CV, NOT RANDOM CV, for the reason `rest_position_decode` lives by: positions run in
    ~6-trial blocks, so two rest periods from one block share whatever drift the session has, and
    random folds would put them in train and test and recover BLOCK IDENTITY as position. Returns
    None when the session cannot support the split rather than guessing.
    """
    from sklearn.model_selection import GroupKFold

    y, g = np.asarray(y), np.asarray(g)
    if len(np.unique(y)) < 3 or len(np.unique(g)) < 4:
        return None
    pred = np.empty_like(y)
    try:
        for tr, te in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                return None
            pred[te] = _fit_frozen(X[tr], y[tr]).predict(X[te])
    except Exception:                                                    # noqa: BLE001
        return None
    return pred


def _collect(session, basis, bins, verbose=True, gate=True):
    """Rest-period features for ONE session in the animal's JOINT basis.

    Returns (X, y, order) or None. `order` is the time order of the periods, which the
    circular-shift null needs and which the collection already produces (the mask is scanned
    forward), so it is returned rather than re-derived.
    """
    import h5py

    from wfield_local import behavior_events as be
    from wfield_local import config, daq_io
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.quiet_periods import quiet_dir

    lab = session["label"]
    qs = sorted(glob.glob(f"{quiet_dir(session['mc'])}/*quiet_sample.npy"))
    if not qs:
        return None, f"{lab}: no rest mask"
    try:
        rest = np.load(qs[0]).astype(bool)
        with h5py.File(session["h5"], "r") as f:
            dn = [x.decode() for x in f["digital/channel_names"][:]]
            packed = f["digital/packed_samples"][:, 0]
            # THE SESSION'S OWN RATE, not `defaults.yaml`'s 5000. Durations here are compared
            # ACROSS sessions, so a per-session rate that ever differed from the global would
            # rescale one animal's rest periods against another's and the duration control would
            # be silently measuring the rig instead of the animal.
            rate = float(f.attrs["sample_rate_hz"])
        pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
        ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
        cue = _load_cue_events(session["h5"])
        # THE REPAIRED CLASSIFIER, NOT `_classify_cues`. Dead `spout_bit1` (8/05-8/06) collapses six
        # positions to four, and a script that used the raw classifier has now reported a phantom
        # "4 positions" animal three separate times in this project. `classify_cues_with_backup`
        # repairs from the behaviour log.
        codes = np.asarray(classify_cues_with_backup(session, cue))
        cs = np.asarray(cue["cue_samples"], np.int64)
        fs_samp = _frame_samples(session["mc"], session.get("fmdir"), session.get("regime"), pco)
        C = np.asarray(basis.signal_or_project(session), dtype=np.float32)
        # THE CANONICAL LICKS, not a re-detection. `behavior_events` is the one identity both the
        # behaviour and imaging halves read, and the rest mask itself was built from these onsets --
        # re-detecting here with different params would make the lick-proximity control disagree
        # with the buffer it is auditing.
        # THE ENGAGEMENT GATE, aligned to the CUE index the collector uses below.
        engaged = None
        if gate:
            from wfield_local.rest_engagement import engaged_by_cue
            engaged, gate_note = engaged_by_cue(session, cs, codes)
            if "UNGATED" in gate_note:
                print(f"  !! {lab}: {gate_note}", flush=True)
        _an, _mmdd = lab.split("_")[0], lab.split("_")[1]
        _ev = be.get_or_compute(config.resolver(), _an, f"2026{_mmdd}")
        licks = np.asarray(_ev["lick_onsets"], np.int64) if _ev is not None else np.zeros(0, np.int64)
    except Exception as ex:                                              # noqa: BLE001
        return None, f"{lab}: {type(ex).__name__} {str(ex)[:60]}"
    if fs_samp is None:
        return None, f"{lab}: no frame map"

    from wfield_local.block_ids import block_ids, block_size_max_for

    T = C.shape[1]
    f_of = np.clip(fs_samp, 0, rest.shape[0] - 1)
    pad = np.concatenate([[0], rest.view(np.int8), [0]])
    dif = np.diff(pad)
    blk = block_ids(np.asarray(codes), block_size_max_for(session))

    X, y, g, dur, p_aa, p_bb = [], [], [], [], [], []
    for aa, bb in zip(np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)):
        prev = np.searchsorted(cs, aa, "right") - 1
        nxt = np.searchsorted(ts, bb, "left")
        if prev < 0 or nxt >= len(ts):
            continue
        nc = np.searchsorted(cs, ts[nxt], "left")
        if nc >= len(codes) or prev >= len(codes):
            continue
        if codes[prev] != codes[nc] or codes[prev] < 0:
            continue                       # block-boundary periods are a different question
        # THE ENGAGEMENT GATE, which this file CLAIMED in prose and did not apply until 2026-09-16.
        # Both bracketing trials must be WORKING: a sated animal's rest is a different state, it is
        # most abundant exactly where the animal has stopped working, and the quit period LENGTHENS
        # post-stroke -- a confound that moves with the independent variable. See
        # `wfield_local/rest_engagement.py`.
        if engaged is not None and not (engaged[prev] and engaged[nc]):
            continue
        fr = np.flatnonzero((f_of >= aa) & (f_of < bb))
        fr = fr[fr < T]
        if fr.size < bins:
            continue
        parts = np.array_split(fr, bins)
        X.append(np.concatenate([C[:, p].mean(1) for p in parts]))
        y.append(int(codes[prev]))
        g.append(int(blk[prev]))
        # PERIOD DURATION IN SECONDS, carried so `--match-duration` can be measured rather than
        # assumed.
        dur.append(float(bb - aa) / rate)
        p_aa.append(int(aa))
        p_bb.append(int(bb))
    if not y:
        return None, f"{lab}: 0 usable rest periods"
    gap = lick_gap_s(np.asarray(p_aa), np.asarray(p_bb), licks, rate)
    # SESSION-LEVEL LICKINESS, so "PS92 is very licky" is a number rather than an impression.
    lick_hz = float(licks.size) / (float(rest.shape[0]) / rate) if rest.shape[0] else float("nan")
    return (np.asarray(X, np.float32), np.asarray(y), np.asarray(g),
            np.asarray(dur, float), gap, lick_hz), None


def training_pool(pre_labels, scored_label):
    """Every pre-stroke session of the animal, MINUS the one being scored.

    That single exclusion is the whole difference between a baseline and a fit. Without it the pre
    bar is a frozen model scored on its own training data, every retained fraction below divides by
    an inflated reference, and each post-stroke epoch looks worse than it is. Extracted from `main`
    so it can be asserted rather than reviewed -- the task arm's equivalent
    (`grant_figures._collect_5c`, "PRE: leave-one-session-out among pre-stroke") is the discipline
    being mirrored.
    """
    return [k for k in pre_labels if k != scored_label]


def _usable(y, min_periods, min_per_class):
    """(ok, why). The counted minimum, applied identically to training pools and scored sessions."""
    if len(y) < min_periods:
        return False, f"only {len(y)} periods (< {min_periods})"
    cls, cnt = np.unique(y, return_counts=True)
    keep = cls[cnt >= min_per_class]
    if len(keep) < 3:
        return False, f"only {len(keep)} positions with >= {min_per_class} periods"
    return True, ""


def _confusion(conf, order, out, variant, stem, codes):
    """The frozen rest decoder's CONFUSION per epoch -- the rest-side `epoch_5c_frozen_confusion`.

    Priya, 2026-09-16: *"did you make decoder matrices for the restdock analyses?"* No -- the arm
    reported balanced accuracy, retained fractions and the refit gap, all of which are SCALARS. A
    scalar says how much position information survives; it cannot say WHERE it goes, and on the task
    side that question has had its own figure since figure 5c.

    WHY IT MATTERS HERE SPECIFICALLY. Rest accuracy falls to 0.332 retained acutely. Two very
    different things produce that number: errors scattering uniformly (the code DEGRADES) or errors
    collapsing onto particular positions (the code SHIFTS -- e.g. far-contra read as near, which is
    the spatial signature the task arm's confusions already show). The retained fraction is
    identical either way.

    RAW COUNTS ARE STORED, row-normalisation is for display only, exactly as `confusion_row`
    requires -- raw counts stay addable, which is what makes pooling across sessions a sum.
    """
    from wfield_local import epoch_figures as ef

    counts = {e: conf[e] for e in order if e in conf and np.asarray(conf[e]).sum()}
    if not counts:
        return None
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    # PERMUTE INTO THE CANONICAL ORDER, do not merely relabel. The matrices are built on
    # `CODES = [0..5]`, and `POSITION_NAMES` puts close_CENTER at 0 -- so an axis in code order is
    # TRANSPOSED against `CONF_LABELS`, which is what every neighbouring figure in this deck uses.
    # Relabelling without permuting would put correct-looking names on the wrong rows, and a
    # confusion matrix read one row out is worse than no confusion matrix: the off-diagonal IS the
    # claim here ("far-contra read as near"), so a transposed axis invents a substitution.
    rank = {q: i for i, q in enumerate(CONF_LABELS)}
    idx = sorted(range(len(codes)),
                 key=lambda i: rank.get(POSITION_NAMES.get(int(codes[i]), ""), 99))
    counts = {e: np.asarray(m)[np.ix_(idx, idx)] for e, m in counts.items()}
    # SHORT ANATOMICAL LABELS, as the task-side frozen figures use. `anatomical_labels` derives
    # ipsi/contra from `stroke_laterality` rather than hardcoding it, so this cannot silently go
    # backwards for a right-lesioned animal -- it raises instead.
    labels = ef.anatomical_labels(
        [POSITION_NAMES.get(int(codes[i]), str(codes[i])) for i in idx], short=True)
    return ef.confusion_row(
        counts, out, name=f"{stem}_confusion",
        title=f"Frozen PRE-stroke REST decoder, pooled across animals -- variant {variant}",
        delta=True, chance=CHANCE, labels=labels)


def _plot(rows, per_session, order, out, variant, stem, n_skipped):
    """Three panels, because the cohort mean has hidden a reversal in this family before.

    The withdrawn "rest follows the same acute dip" claim came from reading a trajectory off a
    pooled average while PS95 was going the other way (`STATUS_2026-09-16.md` pitfall 5). Panel 2
    therefore draws every animal separately and is not optional decoration.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config as _cfg

    colors = _cfg.animal_color()
    animals = sorted({r["animal"] for r in per_session})
    # TALLER THAN IT WAS, because the legends now sit BELOW the axes rather than inside them.
    # In-axes legends on these three panels collided with the data they described -- panel 3 draws
    # two horizontal reference lines across the full width, so there is no empty corner for a
    # legend to occupy and matplotlib's "best" placement lands it on top of a line every time.
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.6))
    x = np.arange(len(order))
    LEG = {"fontsize": 10, "frameon": False, "loc": "upper center", "borderaxespad": 0.0}

    # 1. balanced accuracy against the circular-shift null
    ax[0].plot(x, [r["acc"] for r in rows], "o-", color="k", lw=2, label="frozen decoder")
    ax[0].plot(x, [r["null"] for r in rows], "s--", color="0.55", lw=1.4,
               label="circular-shift null")
    for i, r in enumerate(rows):
        v = [s["acc"] for s in per_session if s["epoch"] == r["epoch"]]
        ax[0].scatter([i] * len(v), v, s=14, color="0.7", zorder=1)
    ax[0].axhline(CHANCE, color="r", ls=":", lw=1, label="chance (1/6)")
    ax[0].set_ylabel("balanced accuracy", fontsize=11)
    ax[0].set_title("Frozen pre-stroke rest decoder", fontsize=12, fontweight="bold")
    ax[0].legend(bbox_to_anchor=(0.5, -0.13), ncol=2, **LEG)

    # 2. per animal, retained fraction -- the panel that would have caught the withdrawn claim
    for an in animals:
        pf = [r["above_chance_frac"] for r in per_session
              if r["animal"] == an and r["epoch"] == "pre"]
        if not pf:
            continue
        ys = []
        for e in order:
            v = [r["above_chance_frac"] for r in per_session
                 if r["animal"] == an and r["epoch"] == e]
            ys.append(np.mean(v) / np.mean(pf) if v else np.nan)
        ax[1].plot(x, ys, "o-", color=colors.get(an, "0.4"), lw=1.8, label=an)
    ax[1].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[1].set_ylabel("above-chance fraction retained (vs own pre)", fontsize=11)
    ax[1].set_title("Per animal -- read this before any trajectory",
                    fontsize=12, fontweight="bold")
    ax[1].legend(bbox_to_anchor=(0.5, -0.13), ncol=len(animals) or 1, **LEG)

    # 3. the cohort retained fraction beside the TASK arm, the comparison the number is for
    ax[2].plot(x, [r["retained"] for r in rows], "o-", color="k", lw=2, label="REST (this arm)")
    ax[2].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[2].set_ylabel("above-chance fraction retained", fontsize=11)
    ax[2].set_title("Rest vs the task and state arms", fontsize=12, fontweight="bold")
    # REFERENCE VALUES, NOT RE-MEASURED HERE -- they are BEHAVIOURAL_STATE_CONTROL.md's, quoted so
    # the rest number is read against something. Marked as quoted on the figure itself so nobody
    # takes them for output of this script.
    for nm, val, col in (("task position (quoted)", 0.496, "#b2182b"),
                         ("state (quoted)", 0.902, "#2166ac")):
        ax[2].axhline(val, color=col, ls="--", lw=1.2, label=nm)
    ax[2].legend(bbox_to_anchor=(0.5, -0.13), ncol=1, **LEG)

    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels(order, fontsize=10)
        a_.tick_params(axis="y", labelsize=10)
        a_.spines[["top", "right"]].set_visible(False)
    ns = ", ".join(f"{r['epoch']} {r['n_sessions']}" for r in rows)
    fig.suptitle(f"Position decoded from REST by a FROZEN pre-stroke model "
                 f"(variant {variant}) -- sessions: {ns}; {n_skipped} skipped", fontsize=11)
    # ROOM AT THE BOTTOM FOR THE LEGENDS, which are now outside the axes.
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    p = out / f"{stem}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    from wfield_local import config, epochs, joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=50)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--min-periods", type=int, default=40)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--match-duration", action="store_true",
                    help="restrict train AND test to the central range of the PRE-STROKE rest-"
                         "period duration distribution -- the control for 'acute animals rest "
                         "longer', which shifts the feature distribution with no position in it")
    ap.add_argument("--duration-pct", type=float, default=10.0,
                    help="percentile trimmed from each tail of the pre duration distribution")
    ap.add_argument("--no-engagement-gate", action="store_true",
                    help="keep the pre-2026-09-16 behaviour: do NOT require both "
                         "bracketing trials to be engaged. For measuring the size "
                         "of the correction, never for a reported result.")
    ap.add_argument("--match-train", action="store_true",
                    help="ALSO fit the frozen model on a size-matched random subset of pre-stroke "
                         "BLOCKS, the counterpart of the task arm's 5rm family. Without it the "
                         "gap carries a training-set-size handicap (~10,000 periods vs ~300) and "
                         "the pre gap is not zero by construction.")
    ap.add_argument("--match-lickgap", action="store_true",
                    help="restrict train AND test to the central range of the PRE-STROKE "
                         "lick-gap distribution. The control for 'acute animals lick less, so "
                         "their rest sits further from licks' -- which otherwise confounds "
                         "composition with code loss, since decoding depends on that gap.")
    ap.add_argument("--lick-far-s", type=float, default=3.0,
                    help="ABSOLUTE gap (s) defining the 'very far from any detected lick' stratum. "
                         "The median split is animal-relative and too weak on its own: a licky "
                         "animal's median is ~1 s, which licking bouts outlast.")
    ap.add_argument("--null", nargs="+", default=["blockperm", "shift", "trial"],
                    choices=["blockperm", "shift", "trial"],
                    help="label-side null(s), predictions held fixed. FIRST is the primary. "
                         "Default leads with blockperm -- decode_ci's convention for the frozen "
                         "task decoders, which this arm is the rest-side counterpart of.")
    ap.add_argument("--refit", action="store_true",
                    help="also fit a WITHIN-SESSION block-CV decoder on the same rest periods; the "
                         "refit-minus-frozen gap is what separates recovery from replacement")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to the shared "
                         "grant_figures/epoch where every other epoch figure lives")
    a = ap.parse_args()
    a.out = a.out or _epoch_dir()

    from wfield_local.quiet_periods import quiet_variant
    variant = quiet_variant() or "retired"
    # THE DURATION-MATCHED RUN MUST NOT OVERWRITE THE UNMATCHED ONE. Two analyses that answer
    # different questions sharing one filename is how a figure comes to disagree with the caption
    # that was written for the other run.
    if a.tag:
        stem = f"epoch_15f_rest_frozen_{variant}{a.tag}"
    else:
        stem = (f"epoch_15f_rest_frozen_{variant}"
                + ("_durmatched" if a.match_duration else "")
                + ("_gapmatched" if a.match_lickgap else ""))
    rng = np.random.default_rng(0)
    t0 = time.time()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    animals = a.animals or sorted({s["label"].split("_")[0] for s in SESSIONS
                                   if s["label"] in want})
    print(f"REST FROZEN DECODER -- variant {variant}, {len(animals)} animals, "
          f"bins={a.bins}, min {a.min_periods} periods / {a.min_per_class} per class\n")

    rows, per_session, skipped = [], [], []
    # POOLED CONFUSION COUNTS, raw, one MxM per epoch over a FIXED code order. The order is fixed
    # up front rather than taken from each session's `np.unique`, because a session missing a
    # position would otherwise contribute an MxM whose axes mean something different -- matrices
    # that are summed have to be indexed the same way.
    CODES = [0, 1, 2, 3, 4, 5]
    cidx = {c: i for i, c in enumerate(CODES)}
    conf = {e: np.zeros((len(CODES), len(CODES)), np.int64) for e in epochs.EPOCHS}
    for an in animals:
        todo = [s for s in SESSIONS
                if s["label"] in want and s["label"].startswith(an) and s.get("h5")]
        try:
            basis = joint_locanmf.load(an, sessions=SESSIONS)
        except Exception as ex:                                          # noqa: BLE001
            # SAID OUT LOUD -- a silent skip here drops a whole animal from a cohort mean.
            print(f"!! {an}: no joint basis ({type(ex).__name__} {str(ex)[:70]}) -- SKIPPED\n")
            skipped.append(f"{an}: no joint basis")
            continue
        print(f"{an}: basis {basis.basis_id}, {basis.ncomp} components, {len(todo)} sessions")

        data = {}
        for s in todo:
            ep = epochs.epoch_of(s["label"])
            if ep is None:
                skipped.append(f"{s['label']}: no epoch")
                continue
            got, why = _collect(s, basis, a.bins, gate=not a.no_engagement_gate)
            if got is None:
                skipped.append(why)
                continue
            X, y, g, dur, gap, lick_hz = got
            ok, bad = _usable(y, a.min_periods, a.min_per_class)
            if not ok:
                skipped.append(f"{s['label']}: {bad}")
                continue
            data[s["label"]] = (ep, X, y, g, dur, gap, lick_hz)
            print(f"  .. {s['label']} [{ep}] {len(y):>4} rest periods, "
                  f"{len(np.unique(y))} positions, median {np.median(dur):.2f}s, "
                  f"{lick_hz:.2f} licks/s, median lick gap {np.median(gap):.2f}s", flush=True)

        pre_labs = [k for k, v in data.items() if v[0] == "pre"]
        if len(pre_labs) < 2:
            print(f"!! {an}: {len(pre_labs)} usable pre-stroke sessions -- LOSO impossible, "
                  f"animal SKIPPED\n")
            skipped.append(f"{an}: {len(pre_labs)} usable pre sessions")
            continue

        # THE DURATION WINDOW, AND WHY IT IS NOT OPTIONAL DECORATION. An acute animal rests far
        # more (quiet goes from 3.4% of a session to 15.1%, BEHAVIOURAL_STATE_CONTROL.md), so a
        # model trained on pre-stroke rest periods is applied to periods of a DIFFERENT LENGTH.
        # Each period is averaged into `bins` parts, so a longer period's features average over a
        # longer stretch of cortex time: the feature distribution shifts for a reason that has
        # nothing to do with position, and it would read as a lesion effect. `--match-duration`
        # restricts BOTH the training pool and the scored session to the central range of the
        # PRE-STROKE duration distribution, so the two are on one footing. Off by default because
        # it discards data; reported alongside, never instead.
        lo_s = hi_s = None
        if a.match_duration:
            pre_dur = np.concatenate([data[k][4] for k in pre_labs])
            lo_s, hi_s = (float(np.percentile(pre_dur, a.duration_pct)),
                          float(np.percentile(pre_dur, 100 - a.duration_pct)))
            print(f"  duration window from PRE: [{lo_s:.2f}, {hi_s:.2f}] s "
                  f"({a.duration_pct}-{100 - a.duration_pct} pct of {len(pre_dur)} pre periods)")

        # THE LICK-PROXIMITY WINDOW -- the control that the >=3s STRATUM showed was needed and that
        # stratification itself cannot deliver. Measured 2026-09-16: in PS92/PS93/PS94 roughly half
        # to three quarters of the PRE-STROKE rest position signal sits within 3 s of a detected
        # lick, and the gap distribution MOVES with epoch (PS93 median 1.21 s pre -> 4.58 s acute,
        # PS94 1.00 -> 4.06) because acute animals lick less. Decoding that depends on proximity,
        # measured over epochs whose proximity differs, confounds composition with code loss.
        #
        # STRATIFYING AT >=3s does not fix it: for the licky animals that stratum is a 7-19% tail
        # and the retained fractions computed inside it go unusable (PS92 subacute -1.697). MATCHING
        # keeps the full sample and equalises the distribution instead, exactly as --match-duration
        # does for period length.
        glo = ghi = None
        if a.match_lickgap:
            pre_gap = np.concatenate([data[k][5] for k in pre_labs])
            pre_gap = pre_gap[np.isfinite(pre_gap)]
            glo, ghi = (float(np.percentile(pre_gap, a.duration_pct)),
                        float(np.percentile(pre_gap, 100 - a.duration_pct)))
            print(f"  lick-gap window from PRE: [{glo:.2f}, {ghi:.2f}] s "
                  f"({a.duration_pct}-{100 - a.duration_pct} pct of {len(pre_gap)} pre periods)")

        for lab, (ep, X, y, g, dur, gap, lick_hz) in data.items():
            train = training_pool(pre_labs, lab)
            Xt = np.concatenate([data[k][1] for k in train])
            yt = np.concatenate([data[k][2] for k in train])
            # BLOCK IDS MADE UNIQUE ACROSS SESSIONS, the same way `_pooled_bundle` does it. They
            # restart per session, so a sampler that pooled two sessions' block 3 would treat one
            # id as one block and draw a unit that does not exist.
            gt_pool = np.concatenate([np.asarray(data[k][3], np.int64) + 1_000_000 * (i + 1)
                                      for i, k in enumerate(train)])
            keep_test = np.ones(len(y), bool)
            if lo_s is not None:
                dt = np.concatenate([data[k][4] for k in train])
                m_tr = (dt >= lo_s) & (dt <= hi_s)
                Xt, yt, gt_pool = Xt[m_tr], yt[m_tr], gt_pool[m_tr]
                keep_test = (dur >= lo_s) & (dur <= hi_s)
            if glo is not None:
                gt = np.concatenate([data[k][5] for k in train])
                if lo_s is not None:
                    gt = gt[m_tr]
                m_g = (gt >= glo) & (gt <= ghi)
                Xt, yt, gt_pool = Xt[m_g], yt[m_g], gt_pool[m_g]
                keep_test &= (gap >= glo) & (gap <= ghi)
            ok, bad = _usable(yt, a.min_periods, a.min_per_class)
            if not ok:
                skipped.append(f"{lab}: training pool {bad}")
                continue
            # Score only positions the frozen model was actually trained on. A class it never saw
            # is not a failure of the decoder, and scoring it as one is the same mistake
            # `_gap_at` fixes on the task side with REFIT_UNAVAILABLE.
            m_test = np.isin(y, np.unique(yt)) & keep_test
            if m_test.sum() < a.min_periods:
                skipped.append(f"{lab}: only {int(m_test.sum())} periods in trained classes")
                continue
            model = _fit_frozen(Xt, yt)
            pred = model.predict(X[m_test])
            acc, ncls = _balanced_accuracy(y[m_test], pred)
            chance = 1.0 / ncls
            yt_test, gt_test = y[m_test], g[m_test]
            for t_c, p_c in zip(yt_test, pred):
                if int(t_c) in cidx and int(p_c) in cidx:
                    conf[ep][cidx[int(t_c)], cidx[int(p_c)]] += 1
            null_stats = {}
            for kind in a.null:
                draws = [_balanced_accuracy(null_labels(yt_test, gt_test, rng, kind), pred)[0]
                         for _ in range(int(a.perm))]
                null_stats[kind] = (float(np.mean(draws)),
                                    (1 + sum(1 for x in draws if x >= acc)) / (1 + len(draws)),
                                    float(np.percentile(draws, 95)))
            primary = a.null[0]
            nm, p, n95 = null_stats[primary]
            # THE PAIRED REFIT ARM, on the SAME periods. `gap` is REFIT MINUS FROZEN, the same sign
            # convention the task side uses (`epoch_grant_figures._gap_at`, ylabel
            # "refit - frozen accuracy"): POSITIVE means the session's own readout finds position
            # the frozen one cannot -- information present but DISPLACED. Asserted in
            # tests/test_rest_frozen_decoder.py because this convention was misread once already.
            rf = refit_predictions(X[m_test], y[m_test], g[m_test]) if a.refit else None
            r_acc = _balanced_accuracy(y[m_test], rf)[0] if rf is not None else float("nan")
            # THE TRAINING-SET-MATCHED FROZEN ARM (`5rm`'s counterpart). The refit model trains on
            # (k-1)/k of THIS session; give the frozen model the same number of periods, drawn as
            # whole pre-stroke blocks, and the size handicap stops contaminating the gap.
            m_acc = float("nan")
            m_used = 0
            if a.match_train:
                k = min(5, int(np.unique(g[m_test]).size))
                n_target = round(int(m_test.sum()) * max(k - 1, 1) / max(k, 1))
                # SEEDED PER SCORED SESSION, and NOT with `hash(lab)` -- Python randomises string
                # hashing per process, so that would make the matched arm irreproducible between
                # runs while looking deterministic. A stable digest of the label instead.
                seed = int(hashlib.sha1(lab.encode()).hexdigest()[:8], 16)
                got = matched_frozen(Xt, yt, gt_pool, n_target, np.random.default_rng(seed))
                if got is not None:
                    m_acc = _balanced_accuracy(y[m_test], got[0].predict(X[m_test]))[0]
                    m_used = got[1]
            # LICK-PROXIMITY STRATIFICATION: the same frozen predictions, split by how long after a
            # detected lick each period sits. Median split within the session, so it is a contrast
            # and not a threshold argument. If the rest position signal is continuation licking, it
            # should sit in the NEAR half; see `lick_gap_s` for what this can and cannot establish.
            gp = gap[m_test]
            fin = np.isfinite(gp)
            acc_near = acc_far = float("nan")
            acc_veryfar = float("nan")
            n_veryfar = 0
            if fin.sum() >= 2 * a.min_per_class:
                cut = float(np.median(gp[fin]))
                near, far = fin & (gp <= cut), fin & (gp > cut)
                if near.sum() >= a.min_periods // 2 and far.sum() >= a.min_periods // 2:
                    acc_near = _balanced_accuracy(yt_test[near], pred[near])[0]
                    acc_far = _balanced_accuracy(yt_test[far], pred[far])[0]
                # THE MEDIAN SPLIT IS TOO WEAK ON ITS OWN, and saying so is the point of this
                # second stratum. PS92's median gap is ~1.0 s, so its "far" half is barely far --
                # licking bouts run well past that, and a confound living at 1-2 s would sit in
                # BOTH halves and cancel. `--lick-far-s` is an ABSOLUTE floor, so the stratum means
                # the same thing in every animal regardless of how licky it is, which the median
                # split by construction does not.
                vf = fin & (gp >= a.lick_far_s)
                n_veryfar = int(vf.sum())
                if n_veryfar >= a.min_periods // 2 and len(np.unique(yt_test[vf])) >= 3:
                    acc_veryfar = _balanced_accuracy(yt_test[vf], pred[vf])[0]
            per_session.append({"animal": an, "label": lab, "epoch": ep, "n": int(m_test.sum()),
                                "n_classes": ncls, "acc": acc, "null": nm, "p": p,
                                "above_chance_frac": (acc - chance) / (1 - chance),
                                "refit_acc": r_acc,
                                "refit_above_chance": (r_acc - chance) / (1 - chance),
                                "gap_refit_minus_frozen": r_acc - acc,
                                "matched_frozen_acc": m_acc, "matched_n_train": m_used,
                                "matched_above_chance": (m_acc - chance) / (1 - chance),
                                "gap_refit_minus_matched": r_acc - m_acc,
                                "n_train_full": len(yt),
                                "median_dur_s": float(np.median(dur)),
                                "median_dur_scored_s": float(np.median(dur[m_test])),
                                "lick_hz": lick_hz,
                                "median_lick_gap_s": float(np.median(gp[fin])) if fin.any() else float("nan"),
                                "acc_near_lick": acc_near, "acc_far_lick": acc_far,
                                "acc_veryfar_lick": acc_veryfar, "n_veryfar": n_veryfar,
                                "null_kind": primary, "null_p95": n95,
                                **{f"null_{k}": v[0] for k, v in null_stats.items()},
                                **{f"p_{k}": v[1] for k, v in null_stats.items()}})
            extra = f", refit {r_acc:.3f} (gap {r_acc - acc:+.3f})" if rf is not None else ""
            others = "".join(f" {k}={v[0]:.3f}/p{v[1]:.3f}"
                             for k, v in null_stats.items() if k != primary)
            print(f"  == {lab} [{ep}] frozen acc {acc:.3f} (null[{primary}] {nm:.3f} p95 {n95:.3f}, "
                  f"p={p:.3f}, {ncls} classes, n={int(m_test.sum())}){extra}{others}", flush=True)
        print(flush=True)

    if not per_session:
        print("NOTHING WAS SCORED -- a failed run, not a negative result.")
        for x in skipped[:20]:
            print("   skipped:", x)
        return 1

    # ---- cohort table -------------------------------------------------------------------------
    order = [e for e in epochs.EPOCHS if any(r["epoch"] == e for r in per_session)]
    print(f"\n{'=' * 88}\nFROZEN PRE-STROKE REST DECODER, applied across epochs "
          f"(variant {variant})\n{'=' * 88}")
    print(f"{'epoch':<10}{'n sess':>8}{'bal acc':>10}{'null':>9}{'above-chance':>14}"
          f"{'retained':>11}{'sess>null':>12}{'med dur':>12}")
    pre_frac = None
    for e in order:
        v = [r for r in per_session if r["epoch"] == e]
        acc = float(np.mean([r["acc"] for r in v]))
        nm = float(np.mean([r["null"] for r in v]))
        frac = float(np.mean([r["above_chance_frac"] for r in v]))
        if e == "pre":
            pre_frac = frac
        ret = (frac / pre_frac) if pre_frac else float("nan")
        n_above = sum(1 for r in v if r["acc"] > r["null"])
        md = float(np.median([r["median_dur_scored_s"] for r in v]))
        r_acc = float(np.nanmean([r["refit_acc"] for r in v]))
        r_frac = float(np.nanmean([r["refit_above_chance"] for r in v]))
        m_acc = float(np.nanmean([r["matched_frozen_acc"] for r in v]))
        m_frac = float(np.nanmean([r["matched_above_chance"] for r in v]))
        rows.append({"epoch": e, "n_sessions": len(v), "acc": acc, "null": nm,
                     "above_chance_frac": frac, "retained": ret,
                     "refit_acc": r_acc, "refit_above_chance": r_frac,
                     "gap_refit_minus_frozen": r_acc - acc,
                     "matched_frozen_acc": m_acc, "matched_above_chance": m_frac,
                     "gap_refit_minus_matched": r_acc - m_acc,
                     "median_dur_s": md, "duration_matched": bool(a.match_duration)})
        print(f"{e:<10}{len(v):>8}{acc:>10.3f}{nm:>9.3f}{frac:>14.3f}{ret:>11.3f}"
              f"{n_above:>9}/{len(v):<3}{md:>12.2f}")

    print("\nRETAINED = this epoch's above-chance fraction divided by PRE's, the normalisation")
    print("BEHAVIOURAL_STATE_CONTROL.md uses. Read it against the TASK arm's 0.86 -> 0.43.")
    print("The pre row is LEAVE-ONE-SESSION-OUT, so it is a baseline and not a fit.")
    # THE CONFOUND, PRINTED WHETHER OR NOT IT IS CONTROLLED FOR. If the median scored period
    # length moves with epoch, the frozen model is being applied to windows it was not trained on,
    # and the retained fraction carries that as well as the lesion. Stating it in the unmatched
    # run is what makes `--match-duration` a check someone actually runs.
    spread = max(r["median_dur_s"] for r in rows) - min(r["median_dur_s"] for r in rows)
    by_ep = ", ".join("{} {:.2f}s".format(r["epoch"], r["median_dur_s"]) for r in rows)
    print(f"\nMEDIAN SCORED REST-PERIOD LENGTH moves {spread:.2f}s across epochs ({by_ep}).")
    if a.refit and np.isfinite(rows[0]["refit_acc"]):
        pre_rf = rows[0]["refit_above_chance"]
        print(f"\n{'=' * 88}\nRECOVERY OR REPLACEMENT -- the same periods, refit WITHIN each "
              f"session (block-CV)\n{'=' * 88}")
        pre_mt = rows[0]["matched_above_chance"]
        print(f"{'epoch':<10}{'frozen':>9}{'matched':>9}{'refit':>9}{'gap':>9}{'gapM':>9}"
              f"{'froz ret':>10}{'matchret':>10}{'refit ret':>11}")
        for r in rows:
            rf_ret = (r["refit_above_chance"] / pre_rf) if pre_rf else float("nan")
            mt_ret = (r["matched_above_chance"] / pre_mt) if pre_mt else float("nan")
            print(f"{r['epoch']:<10}{r['acc']:>9.3f}{r['matched_frozen_acc']:>9.3f}"
                  f"{r['refit_acc']:>9.3f}{r['gap_refit_minus_frozen']:>+9.3f}"
                  f"{r['gap_refit_minus_matched']:>+9.3f}"
                  f"{r['retained']:>10.3f}{mt_ret:>10.3f}{rf_ret:>11.3f}")
        print("\n`matched` = frozen fitted on a SIZE-MATCHED subset of pre-stroke BLOCKS, so `gapM`")
        print("is refit-minus-frozen with the training-set-size handicap removed -- the rest-side")
        print("counterpart of the task arm's 5rm family. Read `gapM`, not `gap`.")
        print("\nGAP IS REFIT MINUS FROZEN, the task arm's sign convention: POSITIVE = the session's")
        print("own readout finds position the frozen pre-stroke model cannot -- present but")
        print("DISPLACED. Both retained fractions low = the code is genuinely degraded; frozen low")
        print("with refit retained = REPLACEMENT rather than recovery.")
    # ---- THE LICKINESS CONTROL (Priya, 2026-09-16) ---------------------------------------------
    print(f"\n{'=' * 88}\nUNDETECTED-LICKING CONTROL -- the spout docks out of reach, so licks at "
          f"nothing are INVISIBLE\n{'=' * 88}")
    print(f"{'animal':<8}{'epoch':<10}{'licks/s':>9}{'lick gap':>10}{'near':>8}{'far':>8}"
          f"{'far-near':>10}{f'>={a.lick_far_s:g}s':>9}{'n':>7}{'vf-all':>9}")
    for an in sorted({r["animal"] for r in per_session}):
        for e in order:
            v = [r for r in per_session if r["animal"] == an and r["epoch"] == e]
            if not v:
                continue
            nr = float(np.nanmean([r["acc_near_lick"] for r in v]))
            fr = float(np.nanmean([r["acc_far_lick"] for r in v]))
            vf = float(np.nanmean([r["acc_veryfar_lick"] for r in v]))
            allacc = float(np.mean([r["acc"] for r in v]))
            print(f"{an:<8}{e:<10}{np.mean([r['lick_hz'] for r in v]):>9.2f}"
                  f"{np.nanmedian([r['median_lick_gap_s'] for r in v]):>10.2f}"
                  f"{nr:>8.3f}{fr:>8.3f}{fr - nr:>+10.3f}"
                  f"{vf:>9.3f}{int(np.sum([r['n_veryfar'] for r in v])):>7}{vf - allacc:>+9.3f}")
    print("\nNEAR/FAR = the SAME frozen predictions, split at the session's median gap to the")
    print(f"nearest DETECTED lick; the >={a.lick_far_s:g}s column is an ABSOLUTE stratum, which means the")
    print("same thing in every animal where the median split does not. `vf-all` is that stratum")
    print("against the session's overall accuracy: near zero = the signal does NOT depend on")
    print("being close to a lick. If rest position decoding were continuation licking, it would")
    print("concentrate NEAR and thin out FAR, i.e. far-near strongly negative. This BOUNDS the")
    print("confound; it cannot measure undetected licks, which no DAQ channel sees. Only tongue")
    print("tracking from the behaviour cameras (DLC, PARKED) settles it -- say so when quoting.")
    if a.match_duration:
        print("This run IS duration-matched: train and test were restricted to the pre-stroke "
              f"{a.duration_pct}-{100 - a.duration_pct} percentile window.")
    elif spread > 0.25:
        print("NOT duration-matched. Re-run with --match-duration before quoting the retained "
              "fraction -- an acute animal rests longer, and the features average over that.")

    # ---- per animal, because a cohort mean has hidden a reversal in this family before ----------
    print(f"\n{'animal':<8}" + "".join(f"{e:>12}" for e in order) + "   (retained)")
    for an in sorted({r["animal"] for r in per_session}):
        pf = [r["above_chance_frac"] for r in per_session if r["animal"] == an and r["epoch"] == "pre"]
        line = f"{an:<8}"
        for e in order:
            v = [r["above_chance_frac"] for r in per_session
                 if r["animal"] == an and r["epoch"] == e]
            if not v or not pf:
                line += f"{'--':>12}"
            else:
                line += f"{np.mean(v) / np.mean(pf):>12.3f}"
        print(line)

    a.out.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    cp = a.out / f"{stem}.csv"
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sp = a.out / f"{stem}_sessions.csv"
    with open(sp, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(per_session[0]))
        w.writeheader()
        w.writerows(per_session)
    print(f"\nwrote {cp}\nwrote {sp}")
    fp = _plot(rows, per_session, order, a.out, variant, stem, len(skipped))
    print(f"wrote {fp}")
    cf = _confusion(conf, order, a.out, variant, stem, CODES)
    print(f"wrote {cf}" if cf else "!! no confusion matrix -- every epoch was empty")
    # THE RAW COUNTS AS CSV TOO. The figure is row-normalised for display; a reader asking "how many
    # far-contra rest periods were read as near" needs the integers, and a claim whose only support
    # is a colour scale is not checkable.
    if cf:
        import csv as _c
        cp2 = a.out / f"{stem}_confusion.csv"
        with open(cp2, "w", newline="", encoding="utf-8") as fh:
            w = _c.writer(fh)
            w.writerow(["epoch", "true", "pred", "count"])
            for e in order:
                for i, tc in enumerate(CODES):
                    for j, pc in enumerate(CODES):
                        w.writerow([e, tc, pc, int(conf[e][i, j])])
        print(f"wrote {cp2}")

    print(f"\ntested {len(per_session)} sessions; {len(skipped)} skipped")
    for x in skipped[:12]:
        print("   skipped:", x)
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
