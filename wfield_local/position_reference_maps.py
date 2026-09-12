"""Per-position cortical maps under THREE different references -- because the reference is the claim.

Priya, 2026-09-12, on figure 14: "in acute there may be less ss-ul/ll activity in far-center trials,
which makes the near ipsi acute trial map look as though there is a relative *increase* in ss-ul/ll
activity compared to pre-stroke." Then: "does that average across all trials make sense? should we
compare it to 'quiet' instead?" and "maybe let's trial comparison the one vs rest vs one vs quiet".

THAT IS THE WHOLE POINT OF THIS MODULE. A position map is always activity MINUS SOMETHING, and
which something decides what a red pixel means:

    MEAN reference    minus the mean over ALL trials in the window (what the decoder centres on,
    "one vs rest"     and therefore what figure 14 draws). THE SIX MAPS ARE NOT INDEPENDENT: one
                      position losing drive lowers the reference and hands the other five an
                      increase they did not earn. Kept here NOT as a preferred view but as the
                      thing being tested -- it is figure 14's reference, rendered in raw evoked
                      activity with no decoder in the path, so the artefact can be seen directly.

    QUIET reference   minus that session's quiet-period baseline -- slow treadmill, no licking,
    "one vs quiet"    buffered, as `behavior_events` defines it and `quiet_periods` writes it per
                      corrected frame. The subtrahend is ONE MAP PER SESSION, identical for all six
                      positions, so subtracting it CANNOT couple them. This is the reference that
                      answers "is this position's cortex driven at all", rather than "is it driven
                      more than the others".

    SELF reference    minus that position's own pre-cue window (`position_evoked_maps`, figure 15).
    "within trial"    Also independent, and the tightest control for slow drift, but it measures the
                      EVOKED CHANGE and so is blind to a sustained shift that is already present
                      before the cue.

WHY QUIET AND SELF ARE BOTH WORTH HAVING, since both keep the positions independent. The pre-cue
window is inside the trial: the animal may already be running, or anticipating, and on a cued task
that anticipation is itself position-blind but not state-blind. The quiet baseline is outside the
task entirely. If a position's map looks the same under both, drift and anticipation are not
carrying it; if it looks different, the difference localises which of the two is responsible.

ONE LOAD, ALL THREE. The expensive part is the session's U and SVT (~100 MB each), so this computes
the raw window means per position AND the quiet baseline in a single pass and derives the references
afterwards. That also guarantees every reference is expressed in the SAME hemodynamic product --
the position maps written by `framemap_event_maps` predate the `meegkit_hpfit` switch for some
sessions, and subtracting a baseline computed from one correction from maps computed under another
is a silent unit error rather than a loud one.

THE TRIAL SELECTION IS FIGURE 14'S. Same `_args("locanmf", align, post_s)` window, same `working` /
`lick` class definitions including the no-lick arm, same `MIN_TRIALS_PER_CLASS` floor -- so a
difference between this figure and figure 14 is the REFERENCE and nothing else. That is the only
way the comparison answers anything.
"""
from __future__ import annotations

import glob
from functools import lru_cache

import numpy as np

from wfield_local.beta_maps import (
    MAP_SHAPE, MIN_TRIALS_PER_CLASS, _quit_mask, _runs_to_blocks,
)

#: The references this module can express a position map in. `self` is `position_evoked_maps`'
#: territory and is named here only so the three can be listed together in one place.
REFERENCES = ("mean", "quiet")


def session_quiet_svt(session, svt):
    """Mean SVT over this session's quiet frames, or None if it has no quiet mask.

    Returns None rather than raising: a session without `quiet_affine8v1` should cost that
    session's QUIET column and nothing else -- its mean-referenced maps are unaffected.
    """
    from wfield_local.quiet_periods import quiet_baseline_svt

    qf = glob.glob(f"{session['mc']}/quiet_affine8v1/*quiet_frame.npy")
    if not qf:
        return None
    try:
        return quiet_baseline_svt(np.asarray(svt), np.load(qf[0]))
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! quiet baseline {session['label']}: {type(ex).__name__} {str(ex)[:70]}",
              flush=True)
        return None


def session_raw_maps(session, align, *, post_s=2.0, variant="working"):
    """``({position: map}, {position: n_trials}, quiet_map_or_None, all_trial_mean_map)``.

    The maps are ABSOLUTE window means -- `U @ mean(SVT over the window)` for that position's
    trials -- so they are not yet interpretable on their own. `reference_maps` turns them into one
    of the two referenced forms; keeping the raw form is what lets both be derived from one load.

    NO DECODER, NO CROSS-VALIDATION, NO BALANCING. This is a trial average, and the three things
    figure 14 needs (a fit, folds, class weights) exist there to answer "what distinguishes the
    positions". Averaging answers "what happens on this position's trials", which is the question a
    reference is supposed to make answerable.
    """
    from wfield_local import joint_basis
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    u, v = joint_basis._load_session(session["mc"])
    X, y, _g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
        session, _args("locanmf", align, post_s), signal=np.asarray(v),
        feat_region=np.arange(v.shape[0]), signal_key=f"svt:rank{v.shape[0]}",
        with_indices=True)
    X, y = np.asarray(X), np.asarray(y)
    if variant == "working" and len(yn):
        # IDENTICAL to `beta_maps.session_maps`: `working` is engaged PLUS miss-while-working, and
        # the no-lick arm comes back separately. Dropping it here would silently make this a
        # lick-trial map -- the exact bug that made figure 14 refuse the far-contra acute cell.
        keep_n = ~_quit_mask(session, idx_e, idx_n, y, yn)
        if keep_n.any():
            X = np.vstack([X, np.asarray(Xn)[keep_n]])
            y = np.concatenate([y, np.asarray(yn)[keep_n]])
    if not len(y):
        return {}, {}, None, None

    K = u.shape[1]
    if X.shape[1] % K:
        raise ValueError(f"{X.shape[1]} features is not a multiple of {K} SVT components")
    n_bins = X.shape[1] // K

    def _map(feat):
        # BINS AVERAGED, as figure 14 does: the per-bin maps are the response's trajectory within
        # the window, this is its summary, and they share a spatial basis so averaging is defined.
        return (u @ np.asarray(feat).reshape(n_bins, K).mean(0)).reshape(MAP_SHAPE)

    raw, used = {}, {}
    for q in CONF_LABELS:
        c = code_of.get(q)
        if c is None:
            continue
        sel = y == c
        # SAME FLOOR AS FIGURE 14 (20 trials), for the same reason: below it the map is erratic
        # rather than merely noisy. A position the animal has stopped attempting loses its OWN
        # column and takes nothing else with it.
        if int(sel.sum()) < MIN_TRIALS_PER_CLASS:
            continue
        raw[q] = _map(X[sel].mean(0))
        used[q] = int(sel.sum())

    # THE TRIAL MEAN, NOT THE MEAN OF THE SIX MAPS. Figure 14's decoder centres on the mean over
    # TRIALS, so a position with few trials contributes little to the reference -- which is exactly
    # the mechanism that makes the near positions look raised when far-contra collapses. Averaging
    # the six maps instead would be a balanced reference and would HIDE the artefact this figure
    # exists to expose.
    trial_mean = _map(X.mean(0))
    qsvt = session_quiet_svt(session, v)
    quiet = None if qsvt is None else (u @ np.asarray(qsvt)).reshape(MAP_SHAPE)
    return raw, used, quiet, trial_mean


def reference_maps(raw, quiet, trial_mean, reference):
    """Turn `session_raw_maps`' absolute window means into one referenced form."""
    if reference == "mean":
        if trial_mean is None:
            return {}
        return {q: m - trial_mean for q, m in raw.items()}
    if reference == "quiet":
        if quiet is None:
            return {}
        return {q: m - quiet for q, m in raw.items()}
    raise ValueError(f"unknown reference {reference!r}; expected one of {REFERENCES}")


@lru_cache(maxsize=8)
def maps_by_epoch(align, variant, post_s=2.0):
    """``({animal: {epoch: {position: {label: {reference: map}}}}}, reliability, trial counts)``.

    CACHED, and keyed on the ARM rather than on the reference, because both references come out of
    one pass over the sessions. Asking for the quiet figure after the mean figure costs nothing.
    """
    from wfield_local import beta_maps as bm
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    out, rel, n_out = {}, {}, {}
    n_noquiet = 0
    for an in ANIMALS:
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        per, ntr = {}, {}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if e is None:
                continue
            try:
                raw, used, quiet, tmean = session_raw_maps(s, align, post_s=post_s,
                                                           variant=variant)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! ref-map {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                continue
            if not raw:
                print(f"  .. ref-map {s['label']} {e}: no position reached "
                      f"{MIN_TRIALS_PER_CLASS} trials", flush=True)
                continue
            if quiet is None:
                n_noquiet += 1
            byref = {r: reference_maps(raw, quiet, tmean, r) for r in REFERENCES}
            for q in raw:
                got = {r: byref[r][q] for r in REFERENCES if q in byref[r]}
                if got:
                    per.setdefault(e, {}).setdefault(q, {})[s["label"]] = got
                    ntr.setdefault(e, {}).setdefault(q, {})[s["label"]] = used.get(q, 0)
        if per:
            out[an], n_out[an] = per, ntr
            rel[an] = {e: {q: {r: bm.split_half_reliability(
                                    {k: d[r] for k, d in by_s.items() if r in d})
                               for r in REFERENCES}
                           for q, by_s in by_q.items()}
                       for e, by_q in per.items()}
    # SAID OUT LOUD. A session with no quiet mask silently drops out of the quiet figure only, so
    # the two references would be built on different session sets with nothing on the figure to say
    # so -- which is how a difference between them gets read as biology.
    if n_noquiet:
        print(f"  .. ref-map: {n_noquiet} session(s) have no quiet mask -- MEAN reference only",
              flush=True)
    return out, rel, n_out


def pooled(store, reference, q, epoch):
    """``(map, n_animals, n_sessions)`` -- each animal's epoch mean, then the mean over animals.

    THE POOLING RULE EVERY FAMILY HERE USES, stated once: an animal with more sessions must not
    dominate, so the average is over ANIMALS and each animal's contribution is its own session mean.
    """
    per, ns = [], 0
    for _an, by_e in store.items():
        got = {k: d[reference] for k, d in ((by_e.get(epoch) or {}).get(q) or {}).items()
               if reference in d}
        if got:
            per.append(np.mean(list(got.values()), axis=0))
            ns += len(got)
    if not per:
        return None, 0, 0
    return np.mean(per, axis=0), len(per), ns


def by_animal(store, reference, q, epoch):
    """``{animal: that animal's epoch-mean map}`` -- the input the permutation test wants."""
    out = {}
    for an, by_e in store.items():
        got = [d[reference] for d in ((by_e.get(epoch) or {}).get(q) or {}).values()
               if reference in d]
        if got:
            out[an] = got
    return out
