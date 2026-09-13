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

#: The references this module can express a position map in. ALL THREE ARE COMPUTED HERE as of
#: 2026-09-12, which is the whole point -- see `session_raw_maps` on why `precue` moved in from
#: `position_evoked_maps`.
REFERENCES = ("mean", "rest", "precue")

#: How a reference is NAMED IN A FILENAME. `rest` resolves through the mask variant actually in
#: use, so a figure built on the retired 8 s-post-reward definition is called `_REWARD8ref_` and one
#: built on the new between-trials/not-running/not-licking definition is called `_RESTref_`.
#:
#: THE FILENAME HAS TO CARRY THE BASELINE, for the same reason `hemo_<variant>/` does
#: (docs/PREPROCESSING_DECISION.md). During this migration two definitions of the same subtrahend
#: exist side by side on the share, and a figure that says only "rest" would be unreadable a week
#: later -- exactly the ambiguity that made "quiet" have to be retired as a word. Priya,
#: 2026-09-12: "i worry that having the older things named quiet will get confusing."
def reference_tag(reference):
    """The filename token for a reference -- baseline-explicit for `rest`."""
    from wfield_local.quiet_periods import quiet_variant

    if reference != "rest":
        return reference.upper()
    return "REST" if quiet_variant() else "REWARD8"


#: Bins the session is split into for the time-local rest baseline. Matches
#: `locanmf_position_encoder._quiet_baseline`, which has used this shape since it was written.
REST_BASELINE_BINS = 12


def session_rest_svt_timelocal(session, svt, nbins=REST_BASELINE_BINS):
    """``(K, T)`` rest baseline that TRACKS DRIFT, or None -- the encoder's construction.

    MEASURED REASON THIS REPLACED A SESSION MEAN (2026-09-13). Rest was found to differ between
    spout positions in 6/6 positions, up to 1,042 of 2,022 bins. The obvious reading -- that the
    rest baseline carries position information and so cannot be a valid subtrahend -- turned out to
    be wrong, and the control that settled it is worth stating: positions are presented in ~6-trial
    BLOCKS, so position is confounded with TIME-WITHIN-SESSION. Splitting each position's rest at
    the session midpoint gave

        DRIFT    (same position, early vs late)      RMS 0.00282
        POSITION (different positions, matched time) RMS 0.00288      ratio 1.02

    i.e. rest differs between positions by EXACTLY as much as the same position's rest differs from
    itself across the session. It is drift aliased onto the block structure, not position coding.

    A SINGLE SESSION MEAN CANNOT REMOVE THAT, because each position's trials cluster at particular
    times and the mean is flat. A time-local baseline can, and `locanmf_position_encoder` has used
    one all along -- "bin the session into nbins, take the median of quiet frames per bin,
    interpolate to every frame -> tracks slow drift (photobleaching / state)". The map reference and
    the encoder were computing the same quantity two different ways; this makes them agree, on the
    encoder's side, which is the correct one.

    MEDIAN PER BIN, not mean: a bin with few rest frames should not be dragged by one outlier, and
    the encoder uses the median for the same reason. Bins with NO rest frames are interpolated
    across rather than dropped, so the baseline is defined at every frame.
    """
    from wfield_local.quiet_periods import quiet_frame_path

    qf = quiet_frame_path(session["mc"])
    if not qf:
        return None
    try:
        import numpy as _np

        q = _np.load(qf).astype(bool)
        V = np.asarray(svt)
        T = V.shape[1]
        L = min(q.shape[0], T)
        qm = np.zeros(T, bool)
        qm[:L] = q[:L]
        qi = np.flatnonzero(qm)
        if qi.size < 2 * nbins:
            return None
        edges = np.linspace(0, T, int(nbins) + 1)
        cent = (edges[:-1] + edges[1:]) / 2.0
        bm_ = np.full((V.shape[0], int(nbins)), np.nan)
        for b in range(int(nbins)):
            sel = qi[(qi >= edges[b]) & (qi < edges[b + 1])]
            if sel.size:
                bm_[:, b] = np.median(V[:, sel], axis=1)
        out = np.empty((V.shape[0], T), dtype=float)
        x = np.arange(T)
        for k in range(V.shape[0]):
            ok = np.isfinite(bm_[k])
            if not ok.any():
                return None
            out[k] = np.interp(x, cent[ok], bm_[k][ok])
        return out
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! time-local rest {session['label']}: {type(ex).__name__} {str(ex)[:60]}",
              flush=True)
        return None


def session_quiet_svt(session, svt):
    """Mean SVT over this session's quiet frames, or None if it has no quiet mask.

    Returns None rather than raising: a session without `quiet_affine8v1` should cost that
    session's QUIET column and nothing else -- its mean-referenced maps are unaffected.
    """
    from wfield_local.quiet_periods import quiet_baseline_svt, quiet_frame_path

    qf = quiet_frame_path(session["mc"])
    if not qf:
        return None
    try:
        return quiet_baseline_svt(np.asarray(svt), np.load(qf))
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! quiet baseline {session['label']}: {type(ex).__name__} {str(ex)[:70]}",
              flush=True)
        return None


def _working_xy(session, align, post_s, variant, baseline, v):
    """``(X, y)`` for one baseline setting -- figure 14's trial selection, exactly.

    FACTORED OUT so the no-baseline and pre-cue-baseline loads cannot drift apart. They must agree
    on the window, the class definition and the no-lick arm, or the reference stops being the only
    difference between the maps -- which is the entire claim this module makes.
    """
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    X, y, _g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
        session, _args("locanmf", align, post_s, baseline=baseline), signal=np.asarray(v),
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
    return X, y


def session_raw_maps(session, align, *, post_s=2.0, variant="working"):
    """``(raw, used, quiet, trial_mean, raw_precue, used_precue)``.

    `raw` are ABSOLUTE window means -- `U @ mean(SVT over the window)` for that position's trials --
    so they are not interpretable on their own. `reference_maps` turns them into a referenced form;
    keeping the raw form is what lets every reference come from one load.

    `raw_precue` is the same quantity with each trial's OWN 1.0 s pre-cue mean already subtracted,
    from a second `trial_features` load at ``baseline="precue"``.

    WHY THE PRE-CUE REFERENCE IS COMPUTED HERE RATHER THAN READ FROM THE NPZ (Priya, 2026-09-12, of
    the three-reference consistency comparison: "clarify this - we may need to fix it"). It was a
    real confound and it is now gone. `position_evoked_maps` aggregates the `delta` field of
    `*_spout_positions_1s_pre_post_delta_maps.npz`, which the imaging box writes as **1 s post minus
    1 s pre**. The other references here are the deck's own window -- cue-aligned **0 to +2.0 s**,
    62 frames at 31.23 Hz in 4 averaged bins, no baseline, on figure 14's trial selection (engaged
    plus miss-while-working, max_rt 3.5 s, a 20-trial floor per position). So "PRECUE vs QUIET" had
    been comparing

        reference   1 s pre-cue mean   vs  session quiet baseline     <- the intended contrast
        window      1 s post           vs  2 s post                   <- and three uncontrolled ones
        trials      preprocessing's    vs  the deck's engaged set
        pipeline    the imaging box's, at whatever hemo correction that session had

    -- four differences wearing one name, so a gap in between-animal consistency could have been
    any of them. `trial_features` has supported ``baseline="precue"`` all along
    (`locanmf_position_decoder`: ``subtract = args.baseline == "precue"``, which removes
    ``sig[:, c0 - pre_n:c0].mean(1)`` from every bin), so the clean version costs one extra CACHED
    feature build per session and nothing else.

    THE TRIAL SETS ARE NOT QUITE IDENTICAL, and that is the one residual difference. A trial whose
    cue sits closer to the recording start than `pre_s` HAS NO PRE-CUE WINDOW and `trial_features`
    drops it under that baseline only. `used_precue` reports the per-position count so the gap is
    visible rather than assumed.

    THIS DOES NOT RETIRE `position_evoked_maps`. That module remains the record of what the
    preprocessing product contains and what figure 15 has always drawn -- and the two can now be
    compared, which is the only way to find out how much of figure 15 was the WINDOW rather than
    the reference.

    NO DECODER, NO CROSS-VALIDATION, NO BALANCING. This is a trial average, and the three things
    figure 14 needs (a fit, folds, class weights) exist there to answer "what distinguishes the
    positions". Averaging answers "what happens on this position's trials", which is the question a
    reference is supposed to make answerable.
    """
    from wfield_local import joint_basis
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    u, v = joint_basis._load_session(session["mc"])
    X, y = _working_xy(session, align, post_s, variant, "none", v)
    if not len(y):
        return {}, {}, None, None, {}, {}

    K = u.shape[1]
    if X.shape[1] % K:
        raise ValueError(f"{X.shape[1]} features is not a multiple of {K} SVT components")
    n_bins = X.shape[1] // K

    def _map(feat):
        # BINS AVERAGED, as figure 14 does: the per-bin maps are the response's trajectory within
        # the window, this is its summary, and they share a spatial basis so averaging is defined.
        return (u @ np.asarray(feat).reshape(n_bins, K).mean(0)).reshape(MAP_SHAPE)

    def _per_position(Xa, ya):
        r, n = {}, {}
        for q in CONF_LABELS:
            c = code_of.get(q)
            if c is None:
                continue
            sel = np.asarray(ya) == c
            # SAME FLOOR AS FIGURE 14 (20 trials), for the same reason: below it the map is erratic
            # rather than merely noisy. A position the animal has stopped attempting loses its OWN
            # column and takes nothing else with it.
            if int(sel.sum()) < MIN_TRIALS_PER_CLASS:
                continue
            r[q], n[q] = _map(np.asarray(Xa)[sel].mean(0)), int(sel.sum())
        return r, n

    raw, used = _per_position(X, y)

    # THE SECOND LOAD. A failure costs the PRE-CUE column of this session and nothing else -- the
    # other two references are already in hand and must not be lost with it.
    raw_pc, used_pc = {}, {}
    try:
        Xp, yp = _working_xy(session, align, post_s, variant, "precue", v)
        if len(yp) and np.asarray(Xp).shape[1] == X.shape[1]:
            raw_pc, used_pc = _per_position(Xp, yp)
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! precue-referenced maps {session['label']}: "
              f"{type(ex).__name__} {str(ex)[:70]}", flush=True)

    # THE TRIAL MEAN, NOT THE MEAN OF THE SIX MAPS. Figure 14's decoder centres on the mean over
    # TRIALS, so a position with few trials contributes little to the reference -- which is exactly
    # the mechanism that makes the near positions look raised when far-contra collapses. Averaging
    # the six maps instead would be a balanced reference and would HIDE the artefact this figure
    # exists to expose.
    trial_mean = _map(X.mean(0))
    qsvt = session_quiet_svt(session, v)
    quiet = None if qsvt is None else (u @ np.asarray(qsvt)).reshape(MAP_SHAPE)
    return raw, used, quiet, trial_mean, raw_pc, used_pc


def reference_maps(raw, quiet, trial_mean, reference, raw_precue=None):
    """Turn `session_raw_maps`' window means into one referenced form.

    `precue` arrives ALREADY referenced, and it has to: a pre-cue baseline is per trial, so the
    subtraction can only happen inside `trial_features`. By the time the trials are averaged the
    information needed to remove it is gone.
    """
    if reference == "mean":
        if trial_mean is None:
            return {}
        return {q: m - trial_mean for q, m in raw.items()}
    if reference == "rest":
        if quiet is None:
            return {}
        return {q: m - quiet for q, m in raw.items()}
    if reference == "precue":
        return dict(raw_precue or {})
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
    n_noquiet = n_noprecue = 0
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
                raw, used, quiet, tmean, raw_pc, used_pc = session_raw_maps(
                    s, align, post_s=post_s, variant=variant)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! ref-map {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                continue
            if not raw:
                print(f"  .. ref-map {s['label']} {e}: no position reached "
                      f"{MIN_TRIALS_PER_CLASS} trials", flush=True)
                continue
            if quiet is None:
                n_noquiet += 1
            if not raw_pc:
                n_noprecue += 1
            byref = {r: reference_maps(raw, quiet, tmean, r, raw_pc) for r in REFERENCES}
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
    if n_noprecue:
        print(f"  .. ref-map: {n_noprecue} session(s) produced no pre-cue-referenced maps",
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
