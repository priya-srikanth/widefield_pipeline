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
REFERENCES = ("mean", "rest", "restw", "precue")

#: `restw` JOINED REFERENCES ON 2026-09-13, in the same commit as `session_restw_svt` and
#: as its `epoch_grant_figures._REF_TEXT` entry. It had been named here weeks earlier and listed in
#: REFERENCES ahead of its builder, which took down EVERY 15r render for twenty minutes --
#: `maps_by_epoch` iterates REFERENCES and `reference_maps` raises on an unknown name. THREE things
#: have to land together for a new reference: the builder, this tuple, and the caption table (a
#: missing `_REF_TEXT` key raises per-arm AFTER the earlier references are written, so the render
#: exits 0 having produced fewer figures than asked -- that is how `precue` was silently lost).

#: `restw` -- the POSITION-WEIGHTED ITI average (Priya, 2026-09-13: "let's add another possible
#: 'quiet' - position-weighted ITI average").
#:
#: THE DEFECT IT FIXES, and it is a real one that gets WORSE post-stroke. `rest` subtracts the
#: session's rest baseline, which is a mean over rest FRAMES -- so a position contributing more rest
#: frames pulls the baseline toward its own resting state. Pre-stroke the six positions contribute
#: roughly equally and this hardly matters. AFTER THE LESION IT MATTERS A LOT: the animal stops
#: attempting the far positions, those blocks shorten or vanish, and the unweighted mean drifts
#: toward the NEAR positions' rest. The baseline then changes with the deficit -- which is precisely
#: the failure that retired the 8 s-post-reward definition, arriving by a different route.
#:
#: WHAT IT IS: build each position's own time-local rest baseline, then average the SIX, equally.
#: The result is still ONE subtrahend, IDENTICAL for all six positions, so it cannot couple them --
#: the property `mean` lacks and the whole reason `rest` is the primary reference. What changes is
#: that its composition no longer tracks which positions the animal still works.
#:
#: HOW IT DIFFERS FROM A PER-POSITION BASELINE, which is a separate thing and NOT this. Subtracting
#: each position's OWN rest would remove the between-trial position signal entirely -- and that
#: signal is now known to exist (`rest_position_permutation` on `restdock05`, 2026-09-15:
#: observed/null 1.622, 44/44 sessions). `restw` keeps it; a per-position reference
#: would delete it. The difference between the two is the measurement of it.
REST_WEIGHTED = "restw"

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

    if reference not in ("rest", "restw"):
        return reference.upper()
    # THE WEIGHTING IS PART OF THE BASELINE'S IDENTITY, so it goes in the filename with it. A
    # `_RESTWref_` figure and a `_RESTref_` one differ in what was subtracted, not in how it was
    # drawn, and two files that differ that way must not share a name.
    base = "REST" if quiet_variant() else "REWARD8"
    return base + "W" if reference == "restw" else base


#: Bins the session is split into for the time-local rest baseline. Matches
#: `locanmf_position_encoder._quiet_baseline`, which has used this shape since it was written.
REST_BASELINE_BINS = 12


def _timelocal_from_mask(V, qm, nbins):
    """``(K, T)`` time-local baseline from a boolean frame mask, or None if it cannot be formed.

    THE ONE PLACE THE RULE LIVES: bin the session into `nbins`, take the MEDIAN of the masked frames
    in each bin, interpolate to every frame. Median and not mean, because a bin with few rest frames
    should not be dragged by one outlier -- `locanmf_position_encoder._quiet_baseline_local` uses
    the median for the same reason and this exists so the two constructions cannot diverge.

    Bins with NO masked frame are interpolated ACROSS rather than dropped, so the baseline is defined
    at every frame. A component with no usable bin at all returns None rather than a guess.
    """
    T = V.shape[1]
    qi = np.flatnonzero(qm[:T])
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


def session_rest_svt_timelocal(session, svt, nbins=REST_BASELINE_BINS, frames=None):
    """``(K, T)`` rest baseline that TRACKS DRIFT, or None -- the encoder's construction.

    MEASURED REASON THIS REPLACED A SESSION MEAN (2026-09-13): the session mean is FLAT and cannot
    remove drift, while each position's trials cluster at particular times, so a flat subtrahend
    leaves every position carrying its blocks' share of the session's drift.

    A PARAGRAPH THAT STOOD HERE IS WITHDRAWN. It argued, from `rest_position_vs_drift`'s ratio of
    1.02, that rest's position differences were "drift aliased onto the block structure, not
    position coding". That inference does not hold -- the two contrasts it compared are not matched
    on time separation, and a ratio of magnitudes is not a test. A circular-shift permutation that
    keeps the block-time structure INSIDE the null gives observed/null **1.622 over 44 pre-stroke
    sessions, above null in 44/44** (`restdock05`, re-measured 2026-09-15; the earlier 1.429 41/44
    and 1.449 39/42 were on superseded rest definitions): REST DOES CARRY POSITION INFORMATION. See
    `scripts/rest_migration/rest_position_permutation.py` and DECISIONS.md 2026-09-13.

    THAT DOES NOT INVALIDATE THIS BASELINE -- it is still identical for all six positions and so
    cannot couple them, which is the property `mean` lacks. It does mean the subtrahend is not
    position-NEUTRAL, which is what `restw` (position-weighted) and a per-position reference exist
    to address from two different directions.

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

    # ``frames`` OVERRIDES THE MASK, for `restw`: the same construction, run on one position's rest
    # frames instead of all of them. Passed as a boolean array over frames, already aligned to the
    # signal, so this function stays the single place the binning/median/interpolate rule lives.
    if frames is not None:
        try:
            V = np.asarray(svt)
            return _timelocal_from_mask(V, np.asarray(frames, bool), int(nbins))
        except Exception:                                              # noqa: BLE001
            return None
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
        return _timelocal_from_mask(V, qm, int(nbins))
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! time-local rest {session['label']}: {type(ex).__name__} {str(ex)[:60]}",
              flush=True)
        return None


def session_restw_svt(session, svt, nbins=REST_BASELINE_BINS, docked=False):
    """``(baseline, codes_used)`` -- the POSITION-WEIGHTED FLAT rest baseline, or ``(None, [])``.

    Build each position's OWN FLAT rest level -- the median over that position's rest frames, one
    number per component, NOT binned over session time -- then average those with EQUAL WEIGHT. The
    result is still ONE subtrahend, identical for all six positions, so it cannot couple them --
    the property `mean` lacks and the reason `rest` is the primary reference. What changes is that
    its COMPOSITION no longer tracks which positions the animal still works.

    THE DEFECT THIS FIXES GETS WORSE AFTER THE LESION, which is what makes it a correctness fix and
    not a refinement. `rest` averages over rest FRAMES, so a position contributing more rest frames
    pulls the baseline toward its own resting state. Pre-stroke the six contribute roughly equally.
    Post-stroke the animal stops attempting the far positions, their blocks shorten or vanish, and
    the frame-weighted mean drifts toward the NEAR positions' rest -- so the baseline changes WITH
    the deficit. That is the same failure that retired the 8 s-post-reward definition, arriving by a
    different route.

    AND IT MATTERS MORE THAN WHEN IT WAS DESIGNED, because of a finding made after. `restw` was
    specified while rest was believed to be position-neutral. It is not: the circular-shift
    permutation gives observed/null **1.622 on `restdock05`, 44/44 sessions, and 4/4 animals**
    (PS92 1.444, PS93 1.996, PS94 1.662, PS95 1.500; mean over animals 1.650, 0 sessions skipped).
    Re-measured 2026-09-15 -- the effect is STRONGER on the current definition than the 1.449
    (39/42) measured on its predecessor, and every session now clears its own null. A
    frame-weighted rest average
    therefore CARRIES POSITION, and subtracting it partially cancels the effect under test by an
    amount that varies session to session with block composition.

    WHAT THIS IS NOT: a per-position baseline. Referencing each position to its OWN rest would
    remove the between-trial position signal entirely -- that signal is the persistence trace
    measured at +0.0754 across 4/4 animals -- and report a null. `restw` keeps it. The difference
    between the two is the measurement of it (Priya raised both on 2026-09-13; this is the one
    chosen, and the other is the trap).

    RETURNS THE CONTRIBUTING POSITIONS, not just the array, because with fewer than
    `MIN_POSITIONS_FOR_WEIGHTED` of them the quantity stops being what its name says and the
    session loses this column -- see that constant.
    """
    from wfield_local.rest_by_position import rest_frames_by_position, restw_from_frames

    V = np.asarray(svt)
    T = V.shape[1]
    per, info = rest_frames_by_position(session, T, docked=docked)
    if info.get("error"):
        print(f"  !! restw {session['label']}: {info['error']}", flush=True)
        return None, []
    # FLAT PER POSITION, NOT TIME-LOCAL, SINCE 2026-09-14 -- and this REPLACED a per-position
    # time-local construction that was measured and rejected the same week.
    #
    # `flat_vs_timelocal` compared the two baselines on the claims the REST reference actually
    # carries. Acute/pre amplitude ratios: 1.415/1.344, 1.115/1.059, 1.230/1.169, 0.952/0.902,
    # 0.636/0.598, far_R 0.483/0.445 -- same ordering, same monotone near->far gradient. Between-
    # animal agreement at far-contra: acute 0.812 vs 0.838, against a null of 0.07. TIME-LOCAL
    # MOVES NO CONCLUSION, so the simpler estimator wins.
    #
    # AND IT REMOVES THE DEFECT THAT MADE THE OTHER VERSION UNSOUND. A per-position TIME-LOCAL
    # baseline needs each position estimated within each time bin, and positions occupy only
    # 66% of bins on average (min 17%, fully covered 2/30 position-sessions, 16/30 time-clustered)
    # -- so a third of the session was INTERPOLATED per position. Flat needs no bins: each position
    # holds 1,000-3,000 rest frames across a session, pooled over all of it.
    #
    # THE TWO CHOICES ARE SEPARABLE AND WERE TANGLED. Temporal (flat) and composition
    # (position-weighted) are independent; only their PRODUCT was ill-posed. The weighting stays
    # because it is justified by its own measurement -- per-position ITI survival spans 24.4%-56.7%
    # PRE-STROKE (2.3x) and the spread is epoch-dependent, so an unweighted baseline's composition
    # changes with epoch even though its definition does not (`position_survival`).
    #
    # MEDIAN, not mean, for the same reason `_timelocal_from_mask` uses one: a position whose rest
    # contains a few outlier frames should not have its level set by them.
    #
    # NOTE `nbins` IS NOW UNUSED and kept only so callers do not break; it is not a silent no-op,
    # it is recorded here as deliberate.
    #
    # THE ESTIMATOR ITSELF LIVES IN `rest_by_position.restw_from_frames`, not here, because it is
    # basis-agnostic and `locanmf_position_encoder` needs the SAME one on the LocaNMF `C`. While it
    # was duplicated the encoder's copy had drifted into a 24-bin time-local UNWEIGHTED median, so
    # two figures captioned "above rest" subtracted different quantities (found 2026-09-14).
    return restw_from_frames(V, per, label=session["label"])


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


def _working_xy(session, align, post_s, variant, baseline, v, *, signal_key=None):
    """``(X, y)`` for one baseline setting -- figure 14's trial selection, exactly.

    FACTORED OUT so the no-baseline and pre-cue-baseline loads cannot drift apart. They must agree
    on the window, the class definition and the no-lick arm, or the reference stops being the only
    difference between the maps -- which is the entire claim this module makes.

    ``signal_key`` MUST BE OVERRIDDEN WHENEVER ``v`` IS NOT THE SESSION'S PLAIN SVT. The disk cache
    keys on it and on nothing about the array's contents, so passing a drift-removed signal under
    the default key would make the cache serve rest-subtracted features to the plain path and back
    again -- silently, across processes, for as long as the entry survived. That is the exact
    failure `feature_cache_kind`'s `signal_key` exists to prevent; it only works if callers use it.
    """
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    X, y, _g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
        session, _args("locanmf", align, post_s, baseline=baseline), signal=np.asarray(v),
        feat_region=np.arange(v.shape[0]),
        signal_key=signal_key or f"svt:rank{v.shape[0]}",
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

    # ------------------------------------------------------------------ the REST reference
    # THE THIRD LOAD, and the reason it is a load rather than one more subtraction at the end.
    #
    # Until 2026-09-13 the rest reference was ONE SESSION MEAN subtracted from every position's
    # map. That is flat, and a flat subtrahend cannot remove DRIFT -- which is exactly what the
    # rest baseline was then shown to contain: `rest_position_vs_drift` measured the same position's
    # rest early-vs-late at RMS 0.00282 against 0.00288 between positions at matched time, a ratio
    # of 1.02. Positions run in ~6-trial BLOCKS, so each position's trials cluster at particular
    # times and a flat subtrahend leaves every one of them carrying its blocks' share of the drift.
    #
    # THE FIX IS APPLIED TO THE SIGNAL, NOT TO THE MAPS, and that is what makes it correct rather
    # than approximate. Subtracting a time-local baseline from the AVERAGED map would need it
    # evaluated at those trials' times, which this function does not carry; subtracting it from the
    # SVT before `trial_features` sees it means every trial's window is referenced to the baseline
    # AT ITS OWN MOMENT, and the averaging that follows is then over already-referenced trials.
    # `locanmf_position_encoder._quiet_baseline_local` has always done it this way.
    #
    # THE RESULT ARRIVES ALREADY REFERENCED, like `precue` and unlike `mean` -- see `reference_maps`.
    # A failure costs this session's REST column and nothing else.
    raw_rest, used_rest = {}, {}
    base = session_rest_svt_timelocal(session, v)
    if base is not None:
        try:
            Xr, yr = _working_xy(session, align, post_s, variant, "none",
                                 np.asarray(v) - np.asarray(base),
                                 signal_key=f"svt:rank{v.shape[0]}:restlocal{REST_BASELINE_BINS}")
            if len(yr) and np.asarray(Xr).shape[1] == X.shape[1]:
                raw_rest, used_rest = _per_position(Xr, yr)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! rest-referenced maps {session['label']}: "
                  f"{type(ex).__name__} {str(ex)[:70]}", flush=True)

    # ------------------------------------------------------- the POSITION-WEIGHTED REST reference
    # THE FOURTH LOAD. Same construction as REST above -- subtract from the SVT, not from the maps,
    # so each trial is referenced at its own moment -- differing ONLY in that the baseline gives
    # each position equal weight instead of each rest FRAME equal weight. That single difference is
    # the whole reference, which is what makes `rest` vs `restw` a clean comparison.
    # A failure costs this session's RESTW column and nothing else.
    raw_restw, used_restw = {}, {}
    basew, restw_positions = session_restw_svt(session, v)
    if basew is not None:
        try:
            Xw, yw = _working_xy(session, align, post_s, variant, "none",
                                 np.asarray(v) - np.asarray(basew),
                                 # THE CONTRIBUTING POSITIONS ARE PART OF THE KEY, not their
                                 # count: a baseline averaged over {0,1,2,3} and one over
                                 # {2,3,4,5} are different subtrahends of the same length, and a
                                 # count-keyed cache would serve one for the other across
                                 # sessions. Position sets DO differ post-stroke -- that is the
                                 # condition `restw` exists for.
                                 signal_key=f"svt:rank{v.shape[0]}:restwflat"
                                            f":p{'-'.join(str(c) for c in restw_positions)}")
            if len(yw) and np.asarray(Xw).shape[1] == X.shape[1]:
                raw_restw, used_restw = _per_position(Xw, yw)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! restw-referenced maps {session['label']}: "
                  f"{type(ex).__name__} {str(ex)[:70]}", flush=True)

    # THE FLAT SESSION MEAN IS STILL RETURNED, and it is no longer the rest reference. It is kept
    # because `_fig_15r_reference_maps` prints the retired form's provenance, and because a reader
    # comparing the two wants the superseded quantity to still exist rather than be described.
    qsvt = session_quiet_svt(session, v)
    quiet = None if qsvt is None else (u @ np.asarray(qsvt)).reshape(MAP_SHAPE)
    return {"raw": raw, "used": used, "quiet_flat": quiet, "trial_mean": trial_mean,
            "raw_precue": raw_pc, "used_precue": used_pc,
            "raw_rest": raw_rest, "used_rest": used_rest,
            "raw_restw": raw_restw, "used_restw": used_restw,
            "restw_positions": restw_positions}


def reference_maps(parts, reference):
    """Turn one `session_raw_maps` result into one referenced form. ``parts`` is its dict.

    TWO OF THE THREE ARRIVE ALREADY REFERENCED, and for the same reason: their subtrahend is
    PER TRIAL, so the subtraction can only happen inside `trial_features`, and by the time the
    trials are averaged the information needed to remove it is gone.

      mean    subtracted HERE. The subtrahend is one map -- the mean over all trials in the window
              -- so it is a property of the session, not of a trial, and the six positions are
              COUPLED through it by construction.
      rest    subtracted from the SVT before the features are built, at each trial's OWN time, so
              the baseline tracks drift (2026-09-13; see `session_raw_maps`). Until that date this
              was one flat session mean subtracted here, which could not remove drift at all.
      precue  subtracted per trial inside `trial_features` at ``baseline="precue"``.

    A DICT RATHER THAN SIX POSITIONAL ARGUMENTS, since 2026-09-13. The tuple form had reached six
    elements and was about to reach eight, and its two map dicts and two count dicts are
    interchangeable at the call site with nothing to catch a transposition -- a swap would have
    produced a complete, plausible, wrong figure.
    """
    if reference == "mean":
        tm = parts.get("trial_mean")
        if tm is None:
            return {}
        return {q: m - tm for q, m in (parts.get("raw") or {}).items()}
    if reference == "rest":
        return dict(parts.get("raw_rest") or {})
    if reference == REST_WEIGHTED:
        return dict(parts.get("raw_restw") or {})
    if reference == "precue":
        return dict(parts.get("raw_precue") or {})
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
    n_noquiet = n_noprecue = n_norestw = 0
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
                parts = session_raw_maps(s, align, post_s=post_s, variant=variant)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! ref-map {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                continue
            raw, used = parts["raw"], parts["used"]
            if not raw:
                print(f"  .. ref-map {s['label']} {e}: no position reached "
                      f"{MIN_TRIALS_PER_CLASS} trials", flush=True)
                continue
            # COUNTED ON THE REST MAPS, not on the flat session mean. Since 2026-09-13 the rest
            # reference comes from its own feature build, so a session can have a quiet MASK and
            # still contribute no rest column -- the thing the tally has to report is the column
            # that is missing from the figure, not the input that happened to exist.
            if not parts["raw_rest"]:
                n_noquiet += 1
            if not parts.get("raw_restw"):
                n_norestw += 1
            if not parts["raw_precue"]:
                n_noprecue += 1
            byref = {r: reference_maps(parts, r) for r in REFERENCES}
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
    if n_norestw:
        # SAID OUT LOUD FOR THE SAME REASON as the two above, and it matters MORE here: a session
        # drops its RESTW column when fewer than `MIN_POSITIONS_FOR_WEIGHTED` positions still have
        # a usable rest baseline, which is a POST-STROKE condition. So the sessions this silently
        # removes are exactly the most affected ones, and `restw` would then be built on a healthier
        # session set than `rest` while sitting beside it on the same figure.
        #
        # THE THRESHOLD IS READ FROM THE CONSTANT, NOT SPELLED OUT. This line said "<4" until
        # 2026-09-15, five weeks after the gate moved to 6 -- a runtime message stating a threshold
        # the code no longer applies, which is worse than no message.
        from wfield_local.rest_by_position import MIN_POSITIONS_FOR_WEIGHTED
        print(f"  .. ref-map: {n_norestw} session(s) produced no position-weighted rest maps "
              f"(<{MIN_POSITIONS_FOR_WEIGHTED} positions with a usable rest baseline) -- "
              f"these are dropped from RESTW ONLY", flush=True)
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
