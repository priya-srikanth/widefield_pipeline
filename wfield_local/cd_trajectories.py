"""Per-position coding directions projected FRAME BY FRAME -- the CD trajectory.

Priya, 2026-09-24: *"In my 2pRAM_pipeline I've been doing a lot of CD analyses and plotting the CD
for different trial types/epochs... build a CD for each lick direction and plot all [trials] for each
spout position trial type aligned to lick? same for ENL etc"* -- and on how it is done there:
*"I think we're getting scalar CD but then plotting the projections onto the CD over time."*

Exactly that. `position_coding_directions` already fits per-position directions on the per-animal
frozen joint basis and reports a SCALAR per trial per window. This module reuses its `direction`,
`poles` and `project` unchanged and adds the only missing step: projecting the signal at EVERY FRAME
onto one fixed direction, so a trial becomes a time course instead of a number.

WHY A SEPARATE DIRECTION HAS TO BE FITTED, AND WHY THAT IS NOT A COMPROMISE.
`position_coding_directions` fits in (component x TIME SUB-BIN) space -- its own comment: "a
direction is a weight per component PER MOMENT in the window", 90 x 4 = 360 for an ENL window. Such
a vector CANNOT be applied to a single frame, and collapsing it would misrepresent what the
sub-binned classifier learnt. More importantly, a direction with time structure and a trajectory are
redundant: projecting frame t with weights fitted for a different moment is not a coherent readout.
**A trajectory requires a time-INVARIANT direction**, so this module fits on the window MEAN
(``bins=1``) and lets all the time structure live in the projection, which is the standard
construction and the one the 2p pipeline uses.

The cost is small and already measured. `configs/defaults.yaml` records pre-cue sub-binning as
UNESTABLISHED: "+0.009, better in 23/44 -- a coin flip, NOT the +0.032 the 16-session pilot
reported... `precue: 4` is kept only because changing it would move every pre-cue number again for
no demonstrated gain." So for the ENL window a mean-feature direction gives up essentially nothing.
Post-lick is where width matters (0.25 s wins, +0.009, 12/16), so read LICK-aligned trajectories
knowing their direction is blunter than the sub-binned one scored elsewhere.

NOTHING EXISTING IS REPLACED. `position_coding_directions` keeps its four sub-binned methods; this
is a fifth object beside them, in the spirit of that module's own note that "_orth variants are the
SAME construction with the engagement axis projected out. Kept as separate methods rather than
replacing the originals, so both are on disk and comparable."

THE DIRECTION IS DEFINED WHERE BEHAVIOUR IS INTACT: PRE-STROKE trials with a successful lick, P
against not-P, matching `position_coding_directions` exactly. Pole-normalised so 0 = pre-stroke
not-P and 1 = pre-stroke lick at P; every trajectory then reads as "fraction of the normal
position-P signature" and is comparable across positions, animals and epochs.

WHAT EACH ALIGNMENT CAN AND CANNOT SAY -- inherited from `position_coding_directions`, unchanged:

  precue (ENL)  the clean one. The window is lick-free by construction (`decode.precue_lickfree`).
  cue           a lick trial contains its lick from ~140 ms, so there is NO movement-free cue
                window to retreat to. Per-position construction is what keeps it interpretable:
                movement is common to every training class, so it cannot define the direction.
  lick          NO-LICK CLASSES HAVE NO LICK TO ALIGN TO and are therefore NOT PLOTTED here. The
                static module can place them at an inferred time (cue + that session's median RT at
                that position); a TRAJECTORY through an inferred time would draw a curve whose
                x-axis is a guess, one that grows with the latency, and post-stroke the latency is
                long and variable. Reporting a number with a caveat is defensible; drawing a
                time course through it is not.

AND THE ONE THING WIDEFIELD CANNOT DO THAT 2p CAN. The haemodynamic response is slow -- seconds
wide -- so these trajectories are that kernel convolved with whatever the underlying signal is.
Onset times, ramp slopes and the ORDER of two nearby events are NOT readable from them the way they
are from a 2p CD trajectory. Read amplitude and gross time course; do not read timing.

THE 2 s FITTING WINDOW IS NOW MEASURED, NOT INHERITED (Priya, 2026-09-24: *"should we try a tighter
pre-cue window, eg the 1s before cue?"*). Widths swept on the pre-cue direction, scoring d' =
(p1 - p0) / pooled per-trial sd, which is what a coding direction is FOR -- pole separation in units
of trial scatter:

    width   PS92 d'  (n)      PS95 d'  (n)
    0.5 s     0.255  (3911)     0.336  (5974)
    1.0 s     0.255  (3911)     0.337  (5974)
    1.5 s     0.280  (3898)     0.344  (5970)
    2.0 s     0.353  (3829)     0.357  (5958)      <- the default
    3.0 s     0.411  ( 458)     0.412  (5081)

**A TIGHTER WINDOW IS WORSE**, monotonically and in both animals -- 1 s costs 28% of d' in PS92.
`defaults.yaml` says "position information concentrates in the final second before the cue", but
that is about where the cue-evoked INCREMENT lives relative to a baseline; it does not make a 1 s
window a better direction. The pre-cue code is sustained and the haemodynamics are slow, so
averaging more frames buys more than the extra second dilutes.

**WIDER SCORES BETTER AND IS STILL REFUSED**, for two reasons. PS92 keeps only 458 of 3911 trials at
3 s, because the lick-free gate cannot find a clean 3 s window: the spout arrives ~3 s before the
cue, so there is almost no slack. And a 3 s window spans essentially the whole strobe-to-cue
interval, so it swallows the SPOUT-ARRIVAL TRANSIENT -- the direction would then be dominated by the
arrival response rather than the maintained pre-cue code, a different quantity, and one that
reintroduces the movement confound `enl_lick_control` exists to test.

Absolute d' is small (0.25-0.41): these are one-vs-rest on SINGLE trials, consistent with the ~0.5
balanced accuracy the six-way decoder reaches. The direction is real; single trials are noisy, which
is why the band is the CI of the mean.

CLI::

    python -m wfield_local.cd_trajectories --animal PS95 --align precue
    python -m wfield_local.cd_trajectories --align precue cue --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import position_coding_directions as pcd
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

#: Seconds before/after the alignment event to draw. The ENL window is 2 s and the spout arrives
#: ~3 s before the cue, so `precue` reaches back past spout arrival on purpose: the trajectory
#: should show the code appearing, not start after it has.
SPAN = {"precue": (-3.0, 4.0), "cue": (-3.0, 4.0), "lick": (-3.0, 4.0)}

#: Classes drawn per panel. `stopped` is included but is the scarce one everywhere
#: (pre-stroke 6/40/326/495), so `min_trials` marks rather than drops it.
CLASSES = ("success", "miss_working", "stopped")

#: A class with fewer than this many trials in a cell is drawn DASHED and labelled, never omitted --
#: "could not test" and "tested and found nothing" are different facts (the rule this whole ENL arm
#: follows).
MIN_TRIALS = 10

#: Lick-aligned no-lick classes are not drawn at all. See the module docstring: their alignment time
#: is an inference, and a trajectory through an inferred time has a guessed x-axis.
LICK_ALIGNED_CLASSES = ("success",)

#: Width of the boxcar the projected course is smoothed with, in seconds, and CENTRED.
#:
#: TWO WRONG VALUES CAME BEFORE THIS ONE, and they fail in opposite directions.
#:
#: **0.5 s centred** (v1) -- judged too noisy, but for a reason since removed. The bands were the
#: INTERQUARTILE SPREAD OF TRIALS on UNSTANDARDISED components, giving +/-5 to +/-10 on an axis
#: whose full signature is 1. With components z-scored, trials averaged by MEAN, and the band the
#: CI OF THE MEAN (~sd/sqrt(n), n in the hundreds), that objection no longer holds.
#:
#: **2.0 s trailing** (v2) -- an OVERCORRECTION that destroyed what the figure is for. Matching the
#: feature width makes t = 0 exactly the scalar CD score, which is a satisfying anchor and a useless
#: trajectory: a 2 s boxcar turns a step into a 2 s linear RAMP, so every lick-aligned panel rose
#: from t = 0 and was still climbing at +2 s. That was the boxcar's impulse response, not the
#: animal's. Priya, 2026-09-24: *"I'd expect some peak in the ENL and after cue"* and *"it should
#: probably rise higher with lick onset"* -- both were being smoothed flat.
#:
#: THE ANCHOR SURVIVES ANYWAY, and this is what v2 got wrong conceptually: the poles constrain the
#: AVERAGE OVER THE FITTING WINDOW, not the value at every instant. The check is therefore that the
#: pre-stroke success trace averages ~1 over [-2, 0), reported as `anchor` on every result.
#: Demanding it instantaneously is what forced the wide window.
SMOOTH_S = 0.4

#: Centred, not trailing. A trailing window lags every feature by half its width, which for a
#: trajectory is a timing artefact in the one axis the figure is about. v2 used trailing so t = 0
#: would coincide with the feature window ending at the cue; with the anchor now an average over
#: [-2, 0) that reason is gone. It also carried an off-by-one: `np.convolve(..., "full")[i]` is
#: ALREADY the window ending at i, so taking `full[n-1:]` returned a LEADING window and shifted the
#: whole trace one window early.
TRAILING = False

#: BUMP THIS WHENEVER `cd_courses` OR `smooth` CHANGES WHAT THEY COMPUTE.
#:
#: `courses_cache_kind` hashes the INPUTS -- align, method, basis, window, directions, standardising
#: stats -- and none of those move when the code between them does. Fixing the trailing-window
#: off-by-one changed every course on disk and the key did not notice: the re-run returned the
#: buggy values to twelve significant figures and would have been read as "the fix did nothing".
#: This is the local counterpart of `session_cache.CACHE_VERSION`, kept separate so a change here
#: does not invalidate every other cached kind in the repo.
#:
#:   1  first version: 0.5 s centred boxcar, 25-75 band
#:   2  trailing window of the FEATURE WIDTH (off-by-one fixed), components z-scored, CI of median
COURSE_VERSION = 3

#: Trials are aggregated by MEAN, with a 95% CI of the mean.
#:
#: THE MEDIAN WAS TRIED FIRST AND IS WRONG HERE, for a reason worth keeping. The poles DEFINE the
#: axis as 0 = pre-stroke not-P and 1 = pre-stroke lick at P, and that anchor is a MEAN. Checked
#: against PS92's own fit features: the mean of pre-stroke success is **1.000 at every position**,
#: exactly, by construction -- while the median is 2.13, 1.48, 1.07, -0.03, -0.63, 1.48. The
#: per-trial projection is strongly skewed, so a median line does not sit where the axis label says,
#: and three of six positions read BELOW ZERO for pre-stroke success alone. That is a figure whose
#: y-axis lies.
#:
#: The median was chosen when each point was a 0.5 s average and single-frame artefacts dominated
#: (per-frame sd 0.04-0.08 against extremes near +/-5). That reason is gone: each point is now a
#: full 2 s window -- what the poles are calibrated on, and what `position_coding_directions`
#: already reports as a mean. Using the mean also keeps the two modules on one definition.
#:
#: THE BAND IS THE UNCERTAINTY OF THE MEAN, NOT THE SPREAD OF TRIALS. Single-trial CD projections in
#: widefield scatter far wider than the effect (PS92's interquartile width is 1.1-3.7 on an axis
#: whose full signature is 1.0), so a trial-spread band says "single trials are noisy" -- true of
#: every cell, and an answer to nobody's question. The `iqr` is still stored per cell so that
#: spread is recorded rather than lost.
BAND_Z = 1.96


#: Standardise each LocaNMF component before the direction is computed. Priya, 2026-09-24: *"are
#: we normalizing locaNMF components properly for use in a CD? this should be similar to using a
#: single-cell population for CD."*
#:
#: MEASURED ON PS95's 95-COMPONENT BASIS, and it is not a subtlety: per-component sd spans **107x**
#: from smallest to largest (p95/p5 = 12x), and the top FIVE components hold **82%** of total
#: variance, the top ten 86%. `pcd.direction(method="dom")` is ``mean(Xp) - mean(Xn)`` normalised as
#: a WHOLE VECTOR, so without this the "coding direction" is set by about five components on
#: amplitude alone, regardless of how position-informative any of them is.
#:
#: The single-cell analogue is the point: a 2p CD is built on dF/F traces of broadly comparable
#: scale, or z-scored first, so every neuron contributes by its RELIABILITY. LocaNMF components have
#: no such property -- a large bright parcel and a small dim one are not commensurable.
#:
#: `method="lr"` already standardises internally (`_disc` is StandardScaler + LogisticRegression,
#: and `pcd.direction` un-scales the weights afterwards), so this brings `dom` into line with it.
#:
#: THE STATISTICS ARE FROZEN, from PRE-STROKE trials only, and reused for every epoch. Standardising
#: per session would erase real between-session amplitude differences; per epoch would normalise
#: away exactly the post-stroke change being measured. A fixed reference frame set before the
#: manipulation is the same discipline the frozen basis itself exists for (rule 10).
STANDARDISE = True


def component_stats(X):
    """``(mu, sd)`` per component from the PRE-STROKE fit features. Zero-variance -> sd 1."""
    X = np.asarray(X, float)
    ok = np.isfinite(X).all(1)
    mu = X[ok].mean(0)
    sd = X[ok].std(0)
    return mu, np.where(sd > 0, sd, 1.0)


def smooth(v, fs, seconds=None, trailing=TRAILING):
    """Boxcar over the CD time course. TRAILING by default, so a value at t summarises [t-w, t].

    Edges shrink the window rather than pad it, so a constant input stays constant everywhere and
    the ends are not dragged toward zero.
    """
    n = max(1, int(round((seconds if seconds is not None else 2.0) * fs)))
    v = np.asarray(v, float)
    if n < 2:
        return v
    k = np.ones(n)
    num = np.convolve(v, k, mode="full")
    den = np.convolve(np.ones_like(v), k, mode="full")
    # `full[i]` for i < len(v) is ALREADY the window ENDING at i, so trailing takes offset 0.
    # Taking `n-1` (the first version) returns the window ending at i+n-1 -- a LEADING window, the
    # whole trace shifted 2 s early, so t=0 showed the POST-cue response rather than the ENL. Caught
    # by checking the anchor the poles define: pre-stroke success at t=0 must be ~1, and it was
    # -0.96 to +3.44. Do not "simplify" this to mode="same", which is the centred case.
    off = 0 if trailing else (n - 1) // 2
    return (num[off:off + v.size] / den[off:off + v.size])


def span_frames(align, fs):
    """``(pre_n, post_n)`` frame counts for the drawn span."""
    lo, hi = SPAN[align]
    return int(round(-lo * fs)), int(round(hi * fs))


def window_means(sig, ref0s, post_n):
    """``(n_trials, ncomp)`` -- the WINDOW MEAN per component, i.e. a ``bins=1`` feature.

    This is what makes the direction time-invariant and therefore projectable frame by frame. It is
    deliberately NOT `_window_feature(..., bins=4)`, which returns (component x sub-bin).
    """
    out = np.full((len(ref0s), sig.shape[0]), np.nan)
    for i, r in enumerate(ref0s):
        if r is None or r < 0 or r + post_n > sig.shape[1]:
            continue
        out[i] = sig[:, int(r):int(r) + post_n].mean(1)
    return out


def fit_directions(X, y, labels, method="dom", stats=None):
    """``{position: (w, p0, p1)}`` from PRE-STROKE successful-lick trials, P against not-P.

    `pcd.direction` and `pcd.poles` unchanged -- they are basis-agnostic, so handing them a
    ``bins=1`` feature gives a weight per COMPONENT rather than per (component, sub-bin), which is
    the whole point. Positions with too few trials on either side are omitted and reported.
    """
    out = {}
    X, y = np.asarray(X, float), np.asarray(y)
    ok = np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if stats is not None:                      # z-score in the FROZEN pre-stroke reference frame
        mu, sd = stats
        X = (X - mu) / sd
    for p in labels:
        m = y == p
        if m.sum() < MIN_TRIALS or (~m).sum() < MIN_TRIALS:
            continue
        w = pcd.direction(X[m], X[~m], method=method)
        p0, p1 = pcd.poles(X[m], X[~m], w)
        out[int(p)] = (w, p0, p1)
    return out


def trajectory(sig, align_f, w, p0, p1, pre_n, post_n, fs=None):
    """``(n_trials, pre_n + post_n)`` pole-normalised projections, one row per trial.

    The whole signal is projected ONCE (``w @ sig``, a single matvec over the session) and trials
    are then sliced out of it. Projecting per trial would repeat the same multiply for every
    overlapping window.
    """
    v = np.asarray(w) @ np.asarray(sig)                       # (T,) the session's own CD time course
    if fs:
        v = smooth(v, fs)                                     # before normalising; linear either way
    d = p1 - p0
    v = (v - p0) / d if abs(d) > 1e-12 else v - p0
    out = np.full((len(align_f), pre_n + post_n), np.nan)
    for i, f in enumerate(align_f):
        if f is None or not np.isfinite(f):
            continue
        a, b = int(f) - pre_n, int(f) + post_n
        if a < 0 or b > v.size:
            continue
        out[i] = v[a:b]
    return out


def arms_cache_kind(args, align):
    """Cache kind for `session_arms`, which touches NO imaging -- only the DAQ.

    Added after timing the course cache: cold 212 s, warm 143 s for PS95. The projection WAS being
    skipped; what remained was `categorize` re-reading cue and lick events from every session's h5
    on the server, once per run and once per alignment. The bookkeeping it returns is a few thousand
    ints, so caching it is nearly free and removes the DAQ read entirely on a warm pass.

    `lickfree` and the response window are in the key for the reason
    `nolick_decoder.session_features_cache_kind` gives: both change which trials survive and neither
    moves any mtime `session_cache.session_signature` stats.
    """
    import hashlib

    from wfield_local import config

    spec = {k: getattr(args, k, None)
            for k in ("align", "post_s", "pre_s", "fs", "max_rt", "response_window_s")}
    spec["align_event"] = align
    spec["lickfree"] = bool(config.defaults()["decode"].get("precue_lickfree", True))
    digest = hashlib.sha1(repr(sorted(spec.items(), key=repr)).encode()).hexdigest()[:12]
    return f"cdarms-{align}-{digest}"


def courses_cache_kind(align, method, basis_key, dirs, win_s, stats=None):
    """Cache kind for one session's per-position CD TIME COURSES.

    WHY CACHE THIS AND NOT THE SIGNAL. The projection is the expensive step -- a U/SVT load and a
    ~100 MB result over the network -- but the result is far too big to memoise per session. The CD
    time courses derived from it are ``n_positions x T`` floats, ~7 MB for a session, so caching
    THEM makes a warm re-run skip the projection entirely. That is the same trade
    `joint_locanmf.BasisSource` is built for: "deferring it behind a callable means a warm cache
    never touches the basis at all".

    THE DIRECTIONS MUST BE IN THE KEY. They are fitted from data, so a course computed under one
    set of weights is simply a different quantity from one computed under another -- and nothing
    `session_cache.session_signature` stats would change when the fit does. Hashing (w, p0, p1) per
    position makes a refit miss the cache instead of silently reusing the old projection.

    The window width and TRAILING are in too: they are applied before the courses are stored, so
    a cached course carries whatever averaging was configured when it was written.
    """
    import hashlib

    h = hashlib.sha1()
    h.update(f"v{COURSE_VERSION}|{align}|{method}|{basis_key}|{win_s}|{SMOOTH_S}|{TRAILING}"
             .encode())
    if stats is not None:                      # the courses are computed in this reference frame
        h.update(np.asarray(stats[0], float).tobytes())
        h.update(np.asarray(stats[1], float).tobytes())
    for p in sorted(dirs):
        w, p0, p1 = dirs[p]
        h.update(np.asarray(w, float).tobytes())
        h.update(f"{p}|{p0!r}|{p1!r}".encode())
    return f"cdcourse-{align}-{h.hexdigest()[:12]}"


def cd_courses(sig, dirs, fs, win_s, stats=None):
    """``{position: (T,)}`` -- the pole-normalised, smoothed CD time course for each position.

    One matvec per position over the whole session. Slicing trials out of these is then free, which
    is why this and not a per-trial projection: overlapping windows would repeat the same multiply.
    """
    out = {}
    sig = np.asarray(sig)
    if stats is not None:
        # The SAME frozen transform the direction was fitted under. Applying a direction fitted in
        # z-space to raw components would silently reweight it by the component scales again.
        mu, sd = stats
        sig = (sig - mu[:, None]) / sd[:, None]
    for p, (w, p0, p1) in dirs.items():
        v = np.asarray(w) @ sig
        v = smooth(v, fs, SMOOTH_S)
        d = p1 - p0
        out[int(p)] = ((v - p0) / d if abs(d) > 1e-12 else v - p0).astype(np.float32)
    return out


def slice_trials(course, align_f, pre_n, post_n):
    """``(n_trials, pre_n + post_n)`` from an already-projected course. NaN where a trial overruns."""
    out = np.full((len(align_f), pre_n + post_n), np.nan, np.float32)
    for i, f in enumerate(align_f):
        if f is None or not np.isfinite(f):
            continue
        a, b = int(f) - pre_n, int(f) + post_n
        if a < 0 or b > course.size:
            continue
        out[i] = course[a:b]
    return out


def session_arms(s, args, basis, align):
    """``{class: {"fit": ref0s, "at": align_frames, "y": codes}}`` for one session -- DAQ only.

    THE TRIAL SET COMES FROM `nolick_decoder.categorize`, not from a classification written here.
    That function already defines `engaged` / `late_rewarded` / `undetected` and the per-session
    engagement gate, and `enl_decode` splits `undetected` by it into `miss_working` / `stopped`.
    Re-deriving any of that locally is how `enl_state_counts` ended up with a 2.0 s response window
    against the decoder's 3.5 s and reported 67 stopped trials where the decode saw 40 (rule 9).

    TWO FRAME SETS PER TRIAL, and they are different things:

      ``fit``  where the direction's feature window STARTS -- `precue_window_start` for the ENL
               (slid to a lick-free gap, or None meaning drop), the cue otherwise.
      ``at``   where the trajectory is CENTRED. Always a real observed event: the CUE for `precue`
               and `cue`, the FIRST LICK for `lick`. A slid feature window does not move it, so
               panels stay on a common x-axis.

    So the align token chooses WHICH WINDOW THE DIRECTION IS FITTED ON; `precue` and `cue` are both
    drawn against the cue and differ only in that.
    """
    from wfield_local.locanmf_position_decoder import precue_window_start
    from wfield_local.nolick_decoder import categorize

    # NO SIGNAL IS TOUCHED HERE. This is pure trial bookkeeping off the DAQ, and separating it from
    # the projection is what lets a warm cache skip the projection entirely -- the reason
    # `joint_locanmf.BasisSource` exists. It also stops the caller holding every session's ~120 MB
    # signal at once, which the first version did.
    codes, cat, _blk, _rt_s, cue_f, sess_eng, ls, strobe_f = categorize(s, args, with_licks=True)
    post_n = int(round(args.post_s * args.fs))
    lickfree = bool(args.align == "precue")

    j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1)

    out = {c: {"fit": [], "at": [], "y": []} for c in CLASSES}
    for k in range(cue_f.size):
        if not cat[k] or codes[k] < 0 or int(cue_f[k]) < 0:
            continue
        if cat[k] == "engaged":
            cls = "success"
        elif cat[k] == "undetected":
            cls = "miss_working" if sess_eng[k] else "stopped"
        else:
            continue                                   # late_rewarded: a HIT, excluded as elsewhere
        c0 = int(cue_f[k])
        if align == "precue":
            ref0 = precue_window_start(c0, strobe_f[k], ls, post_n, lickfree=lickfree)
            if ref0 is None:
                continue                               # no lick-free window exists -> drop
        else:
            ref0 = c0
        at = c0 if align in ("precue", "cue") else (int(first[k]) if first[k] > 0 else None)
        if at is None:
            continue
        out[cls]["fit"].append(ref0)
        out[cls]["at"].append(at)
        out[cls]["y"].append(int(codes[k]))
    return out


def analyse_animal(animal, align="precue", *, method="dom", post_s=None, verbose=True):
    """Per-position directions from PRE-STROKE success, and trajectories for every class x epoch.

    TWO PASSES OVER THE SESSIONS, and the projection is deferred in both.

    Pass 1 needs only a ``bins=1`` window mean per trial to fit the directions; pass 2 needs the
    per-position CD time courses. Neither needs the ~120 MB signal kept afterwards, and the first
    version of this function held EVERY session's signal at once -- ~3 GB for an animal -- purely
    because it computed both passes from one list.

    `joint_locanmf.BasisSource` supplies the signal lazily and carries the `basis:{id}` key that
    makes caching safe, and `courses_cache_kind` memoises the courses, so a re-run projects nothing.
    """
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs, joint_locanmf, session_cache
    from wfield_local.locanmf_frozen_decoder import _args

    basis = joint_locanmf.load(animal)
    # PER-ALIGNMENT, FROM CONFIG -- `decode.{align}_post_s`, exactly as
    # `position_coding_directions.run_animal` resolves it. This was hardcoded to 2.0, which happens
    # to be the current value for all three alignments, so it agreed by luck rather than by
    # construction; if any one of them is retuned the two modules would silently diverge. That is
    # the same shape as `enl_state_counts` hardcoding a 2.0 s response window against the decoder's
    # 3.5 s (rule 9).
    if post_s is None:
        post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    args = _args(source="roi", align=align, post_s=post_s)
    win_n = int(round(args.post_s * args.fs))
    pre_n, post_frames = span_frames(align, args.fs)
    # THE CURATED SET, in `load_sessions()` ORDER -- the same source `enl_decode.sessions_for` uses.
    # Iterating `config.load_sessions()` directly picks up sessions with no SVTcorr on disk, and is
    # a second definition of "which sessions this cohort is" (rule 9). Order is preserved.
    sess = [s for s in ak.curated_sessions() if s["label"].startswith(animal)]

    book, errs = [], []
    akind = arms_cache_kind(args, align)
    for s in sess:
        try:
            arms = session_cache.cached(
                s, akind, lambda s=s: session_arms(s, args, basis, align), verbose=False)
            book.append((s, epochs.epoch_of(s["label"]), arms))
        except Exception as exc:
            errs.append(f"{s['label']}: {type(exc).__name__}: {exc}"[:140])
    if verbose:
        for s, ep, arms in book:
            print(f"  {s['label']} [{ep}] "
                  + " ".join(f"{c}={len(arms[c]['y'])}" for c in CLASSES), flush=True)

    # ---- PASS 1: the direction, from PRE-STROKE SUCCESS only, window means, P vs not-P ----------
    Xf, yf = [], []
    for s, ep, arms in book:
        if ep != "pre" or not arms["success"]["y"]:
            continue
        src = joint_locanmf.BasisSource(basis, s)
        feats = session_cache.cached(
            s, f"cdfit-{align}-{basis.basis_id[:8]}-{win_n}",
            lambda src=src, arms=arms: window_means(src.signal()[0], arms["success"]["fit"], win_n),
            verbose=False)
        Xf.append(feats)
        yf.append(np.asarray(arms["success"]["y"]))
    if not Xf:
        return {"animal": animal, "align": align, "skipped": "no pre-stroke success trials",
                "errors": errs}
    Xall = np.vstack(Xf)
    stats = component_stats(Xall) if STANDARDISE else None
    dirs = fit_directions(Xall, np.concatenate(yf), DISPLAY_ORDER, method=method, stats=stats)
    if not dirs:
        return {"animal": animal, "align": align, "skipped": "no position could be fitted",
                "errors": errs}

    # ---- PASS 2: the trajectories, from CACHED per-position CD time courses ----------------------
    kind = courses_cache_kind(align, method, f"basis:{basis.basis_id}", dirs, args.post_s,
                              stats=stats)
    draw = LICK_ALIGNED_CLASSES if align == "lick" else CLASSES
    acc = {}
    for s, ep, arms in book:
        if not any(arms[c]["y"] for c in draw):
            continue
        src = joint_locanmf.BasisSource(basis, s)
        courses = session_cache.cached(
            s, kind,
            lambda src=src: cd_courses(src.signal()[0], dirs, args.fs, args.post_s, stats=stats),
            verbose=False)
        for cls in draw:
            ys = np.asarray(arms[cls]["y"])
            if not ys.size:
                continue
            at = np.asarray(arms[cls]["at"])
            for p, course in courses.items():
                m = ys == p
                if m.any():
                    acc.setdefault((ep, cls, p), []).append(
                        slice_trials(course, at[m], pre_n, post_frames))

    out = {"animal": animal, "align": align, "method": method,
           "basis_id": basis.basis_id, "ncomp": int(basis.ncomp),
           "fs": float(args.fs), "span": SPAN[align], "pre_n": pre_n, "post_n": post_frames,
           "positions": sorted(dirs), "n_sessions": len(book), "errors": errs,
           "win_s": float(args.post_s), "standardised": bool(STANDARDISE),
           "smooth_s": float(SMOOTH_S), "drawn_classes": list(draw), "traces": {}}
    for key, chunks in acc.items():
        A = np.vstack(chunks)
        n = int(np.isfinite(A).any(1).sum())
        with np.errstate(invalid="ignore"):
            m = np.nanmean(A, 0)
            iqr = np.nanpercentile(A, 75, axis=0) - np.nanpercentile(A, 25, axis=0)
            se = np.nanstd(A, 0) / max(np.sqrt(n), 1.0)
            out["traces"][key] = {"mean": m, "n": n, "iqr": iqr,
                                  "lo": m - BAND_Z * se, "hi": m + BAND_Z * se}
    out["anchor"] = _anchor(out, args.post_s)
    return out


def _anchor(res, win_s):
    """Pre-stroke SUCCESS averaged over the FITTING window, pooled over positions -- must be ~1.

    The poles define 1 as "pre-stroke lick at P", and they do so for the WINDOW MEAN. A short
    smoothing boxcar therefore does NOT have to hit 1 at any single instant; what must hold is that
    the trace averages to 1 across the window the direction was fitted on. Reported on every result
    and printed in the figure title, so a construction error shows up as a number rather than as a
    shape somebody has to doubt.
    """
    t = (np.arange(-res["pre_n"], res["post_n"]) / res["fs"])
    # WHERE THE FITTING WINDOW SITS RELATIVE TO t = 0 DEPENDS ON THE ALIGNMENT, and getting this
    # wrong makes a correct figure look broken: the pre-cue window ENDS at the cue, so it is
    # [-win, 0), while the cue and lick windows START at their event, so they are [0, +win).
    # Checked against [-2,0) for `lick` the anchor read 0.22 and nothing was wrong with the data.
    m = ((t >= -win_s) & (t < 0.0)) if res["align"] == "precue" else ((t >= 0.0) & (t < win_s))
    vals = [np.nanmean(d["mean"][m]) for k, d in res["traces"].items()
            if k[0] == "pre" and k[1] == "success"]
    return float(np.nanmean(vals)) if vals else float("nan")


STYLE = {"success": ("tab:blue", "success (lick)"),
         "miss_working": ("tab:orange", "miss while working"),
         "stopped": ("tab:red", "stopped")}
EPOCHS = ("pre", "acute", "subacute", "chronic")


def figure(res, out):
    """Positions down, epochs across; one trace per class. x = 0 is the ALIGNMENT EVENT."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES

    pos = res["positions"]
    eps = [e for e in EPOCHS if any(k[0] == e for k in res["traces"])]
    if not pos or not eps:
        raise SystemExit("[cd_traj] nothing to draw")
    t = (np.arange(-res["pre_n"], res["post_n"]) / res["fs"])
    fig, axes = plt.subplots(len(pos), len(eps), figsize=(3.0 * len(eps), 1.9 * len(pos)),
                             squeeze=False, sharex=True, sharey="row", constrained_layout=True)
    for i, p in enumerate(pos):
        for j, ep in enumerate(eps):
            ax = axes[i][j]
            # THE POLES ARE THE CROSS-ROW CALIBRATION, so they are drawn darker than before: with
            # sharey="row" each position has its own scale, and these two lines are what keeps the
            # panels comparable by eye. 0 = pre-stroke not-P, 1 = pre-stroke lick at P.
            ax.axhline(0, color="0.55", lw=0.8)
            ax.axhline(1, color="0.55", lw=0.8, ls=":")
            ax.axvline(0, color="0.4", lw=0.8)
            for cls in res["drawn_classes"]:
                d = res["traces"].get((ep, cls, p))
                if d is None:
                    continue
                col, lab = STYLE[cls]
                # THIN CELLS ARE DASHED AND LABELLED, NEVER DROPPED
                thin = d["n"] < MIN_TRIALS
                ax.plot(t, d["mean"], color=col, lw=1.3, ls="--" if thin else "-",
                        label=f"{lab} (n={d['n']})" + (" THIN" if thin else ""))
                if not thin:
                    ax.fill_between(t, d["lo"], d["hi"], color=col, alpha=0.18, linewidth=0)
            if i == 0:
                ax.set_title(ep, fontsize=9)
            if j == 0:
                ax.set_ylabel(POSITION_NAMES.get(p, str(p)), fontsize=8)
            ax.legend(fontsize=5.5, frameon=False, loc="upper left")
            ax.tick_params(labelsize=7)
    zero = {"precue": "cue", "cue": "cue", "lick": "first lick"}[res["align"]]
    for ax in axes[-1]:
        ax.set_xlabel(f"s from {zero}", fontsize=8)
    fig.suptitle(
        f"{res['animal']} — projection onto the per-position {res['align'].upper()} coding "
        f"direction, frame by frame\n"
        f"direction fitted on PRE-STROKE SUCCESS (window mean, time-INVARIANT"
        + (", components Z-SCORED in a frozen pre-stroke frame" if res.get("standardised") else
           ", components NOT standardised") + ") · "
        f"0 = pre-stroke not-P, 1 = pre-stroke lick at P · basis {str(res['basis_id'])[:12]} "
        f"{res['ncomp']}c\n"
        f"{res.get('smooth_s', float('nan')):g}s centred boxcar · MEAN over trials, band = 95% CI "
        f"OF THE MEAN (not trial spread) · anchor: pre-stroke success averages "
        f"{res.get('anchor', float('nan')):.2f} over "
        f"{'[-%g,0)' % res['win_s'] if res['align'] == 'precue' else '[0,%g)' % res['win_s']}\n"
        f"Y SCALED PER ROW (positions differ up to ~4x: close_R spans 3.5 units, far_L 11) — the "
        f"0 and 1 lines are the shared calibration\n"
        f"HAEMODYNAMICS ARE SLOW — read amplitude and gross time course, NOT onset or ordering"
        + ("" if res["align"] != "lick" else
           " · no-lick classes omitted: their alignment time would be inferred"),
        fontsize=8.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--align", nargs="+", default=["precue"],
                    choices=("precue", "cue", "lick"))
    ap.add_argument("--method", default="dom", choices=("dom", "lr"))
    ap.add_argument("--post-s", type=float, default=None,
                    help="override decode.{align}_post_s; default reads it per alignment")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.out:
        out = Path(args.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("figures_working"))
    for animal in (args.animal or ["PS92", "PS93", "PS94", "PS95"]):
        for align in args.align:
            try:
                res = analyse_animal(animal, align, method=args.method, post_s=args.post_s)
            except Exception as exc:
                print(f"[cd_traj] {animal} {align}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if res.get("skipped"):
                print(f"[cd_traj] {animal} {align}: SKIPPED {res['skipped']}", flush=True)
                continue
            p = figure(res, out / f"cd_traj_{animal}_{align}_{args.method}.png")
            print(f"[cd_traj] -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
