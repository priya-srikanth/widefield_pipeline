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

#: Every class the trial bookkeeping knows about. What actually gets FITTED and PLOTTED is set by
#: the GATE below -- `stopped` is never in a gate, being the scarce one everywhere (pre-stroke
#: 6/40/326/495).
CLASSES = ("success", "miss_working", "stopped")

#: THE GATE: which trials the direction is fitted on AND which trials the trace is built from.
#: Priya, 2026-09-24: *"let's do a 'lick' gated version and a 'lick or working' gated version (will
#: actually be interested to see if miss-while-working looks like lick)"*, and on scope: *"traces
#: should be the same trials its trained on"*.
#:
#: So the two versions are self-contained analyses, not one axis with two sets of traces. If
#: `lick_or_working` looks like `lick`, miss-while-working trials carry the same position code as
#: licks -- and `cos(w_lick, w_lick_or_working)` per position, reported as `gate_cos`, is that
#: statement as a number rather than an impression from two figures.
#:
#: THE CLASSES ARE POOLED AT THE TRIAL LEVEL, not averaged separately and combined: the two have
#: very different n (pre-stroke success is 4-6k against 130-500 miss_working), so averaging the
#: class means would silently weight a 130-trial class equally with a 6000-trial one.
GATES = {"lick": ("success",), "lick_or_working": ("success", "miss_working")}

#: A class with fewer than this many trials in a cell is drawn DASHED and labelled, never omitted --
#: "could not test" and "tested and found nothing" are different facts (the rule this whole ENL arm
#: follows).
MIN_TRIALS = 10

#: LICK ALIGNMENT FORCES THE GATE BACK TO SUCCESS, whatever was asked for. A no-lick trial has no
#: lick to align to; the static module can place it at an inferred time (cue + that session's median
#: RT at that position) but a TRAJECTORY through an inferred time has a guessed x-axis that grows
#: with the latency, and post-stroke the latency is long and variable.
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
#:
#: 0.2 s IS MEASURED (Priya, 2026-09-24: *"should we try narrower than 0.4s centered?"*). Peak
#: amplitude and latency of PS95's lick-aligned trial-averaged traces, against NO smoothing:
#:
#:     width   close_center       far_center            worst-case amplitude cost
#:     0.00     2.61 @ +1.99      2.72 @ +0.38          --
#:     0.10     2.60 @ +1.99      2.66 @ +0.38          -2%
#:     0.20     2.59 @ +1.99      2.61 @ +0.32          -4%
#:     0.40     2.58 @ +1.99      2.50 @ +0.38          -8%
#:     0.80     2.53 @ +1.99      2.12 @ +0.42         -22%
#:     1.60     2.38 @ +1.99      1.32 @ +0.70         -51%, and latency shifts +0.32 -> +0.70
#:
#: The cost is NOT uniform: the slow close-position peaks (latency +1.4 to +2.0 s) lose under 1% even
#: at 0.4 s, while the FAST far-position transients (+0.32 to +0.51 s) lose 4-8% because the boxcar
#: is comparable to their width. 0.2 s halves that and costs nothing visible, since the band is the
#: CI of the mean at n = 400-1000. Below 0.2 s there is no return -- 0.10 and 0.00 sit within 1% of
#: it everywhere, which is the 1-2 s haemodynamic kernel asserting itself.
SMOOTH_S = 0.2

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
COURSE_VERSION = 4

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


#: Baseline interval for the condition-independent mode, seconds relative to the alignment event.
#: Far enough back that the event response has not begun.
CIM_BASELINE = (-3.0, -2.0)


def _event_average(sig, at, pre_n, post_n):
    """``(mean (ncomp, T) over trials, n)`` -- the event-triggered average, or None."""
    sig = np.asarray(sig)
    keep = [sig[:, int(f) - pre_n:int(f) + post_n] for f in at
            if np.isfinite(f) and int(f) - pre_n >= 0 and int(f) + post_n <= sig.shape[1]]
    if not keep:
        return None
    return np.mean(np.stack(keep), 0).astype(np.float32), len(keep)


#: Fraction of the grand-mean trajectory's variance the projected-out subspace must capture.
#:
#: K = 1 WAS NOT ENOUGH, and the measurement says so directly. Decomposing PS95's pre-cue traces
#: into a shared part and a position-specific one, AFTER orthogonalising against a single mode, the
#: shared column still ran -0.47 to +1.83 against a position-specific signal of +1.0 to +1.9 -- the
#: same order as the thing being measured. The shared response is a TIME COURSE occupying several
#: dimensions; removing one direction removes one of them.
#:
#: It matters more post-stroke. The mode also ROTATES: PS93's per-epoch cosine against pre is 0.86
#: acute / 0.84 subacute, and a cosine of 0.86 leaves sin = 0.51 of the shared magnitude. Against a
#: shared component ~10x the position-specific one, half of it is still several times the signal --
#: which is why PS93's dips survived K=1 orthogonalisation post-stroke.
#:
#: 0.90 rather than 0.95 or 0.99: each extra dimension removed also takes any position information
#: lying along it, the cost `position_coding_directions.orthogonalise` states for its own use of
#: this move. `cim_k` is reported on every result so the number is visible rather than assumed.
CIM_VAR = 0.90

#: Hard ceiling on K regardless of variance, so a flat spectrum cannot eat the whole space.
CIM_KMAX = 8


def condition_independent_modes(G, t, baseline=CIM_BASELINE, var=CIM_VAR, kmax=CIM_KMAX):
    """``(ncomp, K)`` orthonormal basis for the grand-mean response. ``None`` if it is flat.

    WHY A SUBSPACE AND NOT A DIRECTION. The per-position directions are ONE-VS-REST,
    ``w_P = mean(P) - mean(not-P)``, so across the six they very nearly cancel -- MEASURED on PS95,
    the six unit vectors sum to a vector of length **0.289** where six aligned ones would give 6.0.
    A signal common to every trial therefore CANNOT load positively on all six: the geometry forces
    it positive on some and negative on others. That is what the close-position dips and the
    far-position negative deflections are. Decomposed on PS95 pre-cue, before any orthogonalisation:

        position       observed    SHARED   POS-SPEC
        close_center    +14.08    +14.32      -0.24
        far_center       -6.43     -5.56      -0.87
        close_R          +6.11     +5.01      +1.10

    -- the dramatic position differences are the shared term; the position-specific one is ~1-2
    everywhere. Projecting out the K leading components of that shared time course, rather than just
    its single largest, is what leaves the position-specific part behind.

    K is chosen by variance explained (`CIM_VAR`) and capped (`CIM_KMAX`), from the SVD of the grand
    mean's deviation from its own pre-event baseline, so no interval or rank is hand-picked.
    """
    G = np.asarray(G, float)
    t = np.asarray(t, float)
    base = (t >= baseline[0]) & (t < baseline[1])
    if not base.any():
        base = t < (t.min() + 0.5)
    dev = G - G[:, base].mean(1)[:, None]
    if not np.isfinite(dev).all() or not dev.any():
        return None
    U, sv, _ = np.linalg.svd(dev, full_matrices=False)
    power = sv ** 2
    if power.sum() <= 0:
        return None
    k = int(np.searchsorted(np.cumsum(power) / power.sum(), var) + 1)
    return U[:, :max(1, min(k, kmax, U.shape[1]))]


def subspace_chance(ncomp, k):
    """Expected `subspace_overlap` for two UNRELATED K-dim subspaces of an n-dim space: K/n.

    WITHOUT THIS THE METRIC IS UNREADABLE, and it was briefly misread here. Two random 2-dimensional
    subspaces of the 95-dimensional joint basis overlap at **0.021**, not 0 -- so a measured 0.61 is
    thirty times chance, i.e. the shared mode is strongly CONSERVED across epochs, not "a third of
    it rotated away". The residual-leak reading and the conservation reading are both true and they
    answer different questions:

      * biologically -- 0.61 against 0.021 says the global mode survives the lesion nearly intact;
      * for the ARTIFACT correction -- ~39% of the subspace's power is still unremoved, and against
        a shared component ~10x the position-specific signal, that is several times the signal.

    Only the second licenses discounting a post-stroke panel. The first must not be read as a
    deficit.
    """
    return float(k) / float(max(ncomp, 1))


def subspace_overlap(A, B):
    """Mean squared cosine of the principal angles between two orthonormal bases, in [0, 1].

    The subspace generalisation of the single-vector cosine: 1 means the epoch's shared-response
    subspace is the one that was projected out, and `subspace_chance` -- NOT zero -- is what
    "unrelated" looks like. Reported per epoch as `cim_cos`, beside `cim_chance`.

    A SUBSPACE ROTATES WHERE A SINGLE VECTOR APPEARS NOT TO. Measured on PS95 pre-cue, the K=1
    cosines ran 0.88/0.95/0.88 across the post-stroke epochs while the K=2 subspace overlap ran
    0.68/0.61/0.66 -- one vector can stay well aligned while the plane it lies in turns. That is why
    the diagnostic moved to subspaces when K did.
    """
    if A is None or B is None:
        return None
    M = np.asarray(A).T @ np.asarray(B)
    return float((M ** 2).sum() / max(A.shape[1], 1))



#: How the direction's SUBTRAHEND is chosen -- what position P is contrasted AGAINST.
#:
#: Priya, 2026-09-24: *"is there no way to do CD relative to rest instead of relative to the other
#: positions?"*
#:
#:   ``contrast``  w_P = mean(P) - mean(NOT-P). Position-specific by construction, and the default
#:                 everywhere else in the project. Its failure mode is the one this module spent the
#:                 day chasing: the six directions SUM TO ~ZERO (measured 0.289 of a possible 6.0),
#:                 so any condition-independent signal is FORCED positive on some positions and
#:                 negative on others. That is what the close-position dips and the far-position
#:                 negative deflections are -- decomposed on PS95, close_center reads +14.08 of
#:                 which +14.32 is shared and -0.24 position-specific.
#:
#:   ``rest``      w_P = mean(P) - the TIME-LOCAL rest baseline, identical for all six positions.
#:                 The six directions no longer cancel, so nothing is forced to split sign and the
#:                 artifact cannot arise. The cost is the mirror image: the shared response is now
#:                 ADDED to every position rather than cancelled across them, so the traces are
#:                 dominated by what is common and are LESS position-specific.
#:
#:   ``restw``     the same with each position referenced to ITS OWN rest frames
#:                 (`rest_by_position.rest_frames_by_position`). Removes position differences in the
#:                 baseline -- but `position_reference_maps` records that REST ITSELF CARRIES
#:                 POSITION INFORMATION (observed/null 1.634 over 44 pre-stroke sessions, above null
#:                 in 43/44), so this subtrahend can remove part of the signal being measured. Use
#:                 it as a conservative bound, not as the headline.
#:
#: BASELINE DRIFT IS NOT THE OBJECTION. `rest_baseline_epoch_drift` measures `restw` moving between
#: epochs at 0.074-0.432 of the evoked norm, but the component ALIGNED with the signal has median
#: |bias| 0.07 and is signed both ways, with 1 of 11 cells significant and running the "wrong" way.
#: DECISIONS.md: "the baseline does move; the movement is not pointed at the signal."
REFERENCES = ("contrast", "rest", "restw")

#: Bins the session is split into for the time-local rest baseline. Matches
#: `position_reference_maps.REST_BASELINE_BINS` and `locanmf_position_encoder._quiet_baseline`;
#: imported rather than redefined would be better still, but that module pulls in matplotlib.
REST_BINS = 12


def rest_baseline(session, sig, reference, nbins=REST_BINS):
    """``{code: (ncomp,) baseline}`` or ``{None: (ncomp,)}`` for the flat one. ``None`` if absent.

    REUSES `position_reference_maps._timelocal_from_mask`, which is basis-agnostic -- it bins the
    session, takes the MEDIAN of rest frames per bin and interpolates, and it does that on whatever
    ``(nfeat, T)`` array it is handed. Passing the joint-basis signal keeps the binning rule in the
    one place it already lives rather than making a second copy of it here (rule 9).

    The mask is the pipeline's own: `quiet_periods.quiet_frame_path` for rest frames, ANDed with
    `rest_engagement.engaged_frame_mask` so the terminal quit period is excluded -- 3.1% of rest
    frames pre-stroke but 18.7% acute, so an ungated baseline would change composition WITH the
    deficit it is subtracted from.
    """
    import numpy as _np

    from wfield_local.position_reference_maps import _timelocal_from_mask
    from wfield_local.quiet_periods import quiet_frame_path
    from wfield_local.rest_engagement import engaged_frame_mask

    T = int(_np.asarray(sig).shape[1])
    eng, _note = engaged_frame_mask(session, T)

    if reference == "restw":
        from wfield_local.rest_by_position import rest_frames_by_position

        per, _info = rest_frames_by_position(session, T, engaged_only=True)
        out = {}
        for code, idx in (per or {}).items():
            m = _np.zeros(T, bool)
            m[_np.asarray(idx, int)] = True
            m &= eng
            if m.sum() < nbins:
                continue
            b = _timelocal_from_mask(_np.asarray(sig), m, nbins)
            if b is not None:
                out[int(code)] = _np.asarray(b).mean(1)      # flat over the session
        return out or None

    qf = quiet_frame_path(session["mc"])
    if not qf:
        return None
    q = _np.load(qf).astype(bool)
    m = _np.zeros(T, bool)
    L = min(q.shape[0], T)
    m[:L] = q[:L]
    m &= eng
    if not m.any():
        return None
    b = _timelocal_from_mask(_np.asarray(sig), m, nbins)
    return None if b is None else {None: _np.asarray(b).mean(1)}


def directions_vs_rest(X, y, base, labels, stats=None, cim=None):
    """``{position: (w, p0, p1)}`` with the SUBTRAHEND a rest baseline, not the other positions.

    ``base`` is ``{None: v}`` for the flat reference or ``{code: v}`` for the per-position one.
    Poles keep their meaning -- 0 is the subtrahend, 1 is pre-stroke lick at P -- so the axis label
    on every figure stays true and the anchor check still applies.
    """
    labels = list(labels)
    X, y = np.asarray(X, float), np.asarray(y)
    ok = np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if stats is not None:
        X = (X - stats[0]) / stats[1]
    out = {}
    for p in labels:
        m = y == p
        if m.sum() < MIN_TRIALS:
            continue
        b = base.get(int(p), base.get(None))
        if b is None:
            continue
        b = (np.asarray(b) - stats[0]) / stats[1] if stats is not None else np.asarray(b)
        w = np.asarray(X[m].mean(0)) - b
        n = float(np.linalg.norm(w))
        if n <= 0:
            continue
        w = w / n
        if cim is not None:
            for e_ in np.asarray(cim).T:
                w = pcd.orthogonalise(w, e_)
        out[int(p)] = (w, float(b @ w), float(X[m].mean(0) @ w))
    return out


def fit_directions(X, y, labels, method="dom", stats=None, cim=None, with_surviving=False):
    """``{position: (w, p0, p1)}`` from PRE-STROKE successful-lick trials, P against not-P.

    `pcd.direction` and `pcd.poles` unchanged -- they are basis-agnostic, so handing them a
    ``bins=1`` feature gives a weight per COMPONENT rather than per (component, sub-bin), which is
    the whole point. Positions with too few trials on either side are omitted and reported.
    """
    out, surviving = {}, {}
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
        if cim is None:
            p0, p1 = pcd.poles(X[m], X[~m], w)
            out[int(p)] = (w, p0, p1)
            continue
        # THE SCALE STAYS THE UNROTATED GAP. Rotating shrinks the separation, and recomputing the
        # poles on the residual RESCALES WHATEVER SURVIVES BACK UP TO 1 -- which manufactures the
        # inflation Priya spotted: "the residual after orthogonalizing is so small that the small
        # denominator blows everything up".
        #
        # MEASURED on PS95 pre-cue (K=2), gap before -> after, and how much of the original
        # direction lay in the shared subspace:
        #
        #     close_L        1.565 -> 1.383  (88%)   |w.CIM| 0.47
        #     far_R          1.761 -> 1.156  (66%)   |w.CIM| 0.75
        #     close_center   0.737 -> 0.448  (61%)   |w.CIM| 0.80
        #
        # close_center is worst on BOTH counts at once -- the smallest separation to begin with and
        # the most of it inside the shared subspace -- so its recomputed denominator is 3.1x smaller
        # than close_L's and everything divided by it is inflated 3.1x. far_R is the control: nearly
        # the same alignment and loss, but it started large, so it lands fine.
        #
        # Dividing by the ORIGINAL gap instead makes the axis "fraction of the original position-P
        # signature", so a direction that barely survives draws SMALL, which is the truth. The
        # per-position anchor then reads the SURVIVING FRACTION (close_center 0.607) rather than a
        # meaningless 1.00, and `surviving` records it.
        _q0, _q1 = pcd.poles(X[m], X[~m], w)      # unpacked, not np.diff: that returns a 1-element
        g_plain = float(_q1) - float(_q0)         # array and float() on one is deprecated
        for e_ in np.asarray(cim).T:          # sequential Gram-Schmidt over the whole subspace
            w = pcd.orthogonalise(w, e_)
        p0, p1 = pcd.poles(X[m], X[~m], w)
        surviving[int(p)] = (p1 - p0) / g_plain if g_plain else float("nan")
        out[int(p)] = (w, p0, p0 + g_plain)
    # EXPLICIT SECOND RETURN, not a key in `out`. Every caller iterates `dirs.items()` unpacking a
    # 3-tuple, so smuggling a dict in under a reserved key would crash them at a distance.
    return (out, surviving) if with_surviving else out


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


class _OnceSignal:
    """Load a session's projection AT MOST ONCE, however many cached products ask for it.

    Pass 2 wants two things per session -- the CD courses and the event-triggered average for the
    per-epoch CIM cosine -- and each is cached under its own kind. Calling `BasisSource.signal()`
    twice would re-project a ~100 MB array over the network for the second; on a warm cache neither
    call should happen at all. This defers and shares.
    """

    def __init__(self, src):
        self.src = src
        self._v = None

    def __call__(self):
        if self._v is None:
            self._v = self.src.signal()[0]
        return self._v


def cim_scale(by_epoch, t, baseline=CIM_BASELINE):
    """``{epoch: ||grand-mean deviation|| / pre}`` -- the MAGNITUDE the cosine cannot see.

    Priya, 2026-09-24: *"the cosine will not read out amplitude changes though, right"* -- right,
    and that is a blind spot worth naming rather than a detail. Subspace overlap is SCALE-INVARIANT:
    a response that keeps its orientation exactly and halves in size scores 1.00, which for a lesion
    study reports "unchanged" about the very thing most likely to change.

    So orientation and magnitude are reported side by side, and they DISSOCIATE in informative ways:

        overlap ~1, scale ~1     nothing moved
        overlap ~1, scale < 1    same geometry, weaker drive
        overlap < 1, scale ~1    REORGANISATION -- the code went somewhere else
        both < 1                 mixed, and neither number alone would have said so

    A decoder score conflates all four, because every one of them lowers accuracy. This is the main
    thing the subspace view adds over "can position still be read out".

    It also repairs the residual-leak estimate. The unremoved shared amplitude is
    ``sqrt(1 - overlap) * ||shared||`` -- the second factor is THIS, and using the pre-stroke value
    for a post-stroke epoch (as the first estimate did) is only right if the scale is 1.

    Frobenius norm of the deviation from the pre-event baseline, divided by pre's, so it is a pure
    ratio and the arbitrary units of the z-scored basis cancel.
    """
    out, ref = {}, None
    t = np.asarray(t, float)
    base = (t >= baseline[0]) & (t < baseline[1])
    if not base.any():
        base = t < (t.min() + 0.5)
    for ep, G in by_epoch.items():
        G = np.asarray(G, float)
        out[ep] = float(np.linalg.norm(G - G[:, base].mean(1)[:, None]))
    ref = out.get("pre")
    if not ref:
        return {}
    return {ep: v / ref for ep, v in out.items()}


def cim_rotation(by_epoch, t):
    """``{epoch: cos(CIM_epoch, CIM_pre)}`` -- how far the condition-independent mode turns.

    WHY THIS IS REPORTED RATHER THAN ASSUMED. `--orth` projects out ONE mode, fitted on PRE-STROKE
    sessions and applied to every epoch, because a per-epoch mode would be a moving reference frame
    that subtracts away the post-stroke change being measured (rule 10). The cost is that the
    removal is exact pre-stroke by construction and only approximate afterwards: if the common mode
    ROTATES after the lesion, the post-stroke panels keep a residual share of it.

    MEASURED, and it is not hypothetical. PS93 pre-cue, minimum of the success trace over [0, 4] s,
    plain -> orthogonalised:

        close_center   pre -3.64 -> +0.22    acute -5.56 -> -1.82
                       subacute -7.14 -> -5.13   chronic -6.85 -> -6.45

    The correction is total at pre, most of the way at acute, and almost nothing by chronic --
    degrading exactly in proportion to distance from the epoch it was fitted on. This cosine is the
    diagnostic for that: near 1 means the projection still removes what it removed pre-stroke; well
    below 1 means residual leak, and the panel is weaker evidence than its pre-stroke counterpart.
    """
    pre = by_epoch.get("pre")
    if pre is None:
        return {}
    c0 = condition_independent_modes(pre, t)
    if c0 is None:
        return {}
    return {ep: subspace_overlap(c0, condition_independent_modes(G, t))
            for ep, G in by_epoch.items()}


def analyse_animal(animal, align="precue", *, method="dom", post_s=None, orth=False,
                   gate="lick", reference="contrast", verbose=True):
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
    # LICK ALIGNMENT OVERRIDES THE GATE -- see LICK_ALIGNED_CLASSES.
    use = LICK_ALIGNED_CLASSES if align == "lick" else GATES[gate]
    Xf, yf, Gs, Gn, Rb = [], [], [], [], {}
    for s, ep, arms in book:
        if ep != "pre" or not any(arms[c]["y"] for c in use):
            continue
        get = _OnceSignal(joint_locanmf.BasisSource(basis, s))
        # THE GATE IS IN THE CACHE KEY: it changes which trials the features are built from, and
        # `session_signature` has no way to see it.
        fit_at = [f for c in use for f in arms[c]["fit"]]
        fit_y = [v for c in use for v in arms[c]["y"]]
        feats = session_cache.cached(
            s, f"cdfit-{align}-{'+'.join(use)}-{basis.basis_id[:8]}-{win_n}",
            lambda get=get, fit_at=fit_at: window_means(get(), fit_at, win_n),
            verbose=False)
        Xf.append(feats)
        yf.append(np.asarray(fit_y))
        if reference != "contrast":
            # Cached per session: the mask work and the per-bin median are not free, and the
            # signal is already in hand here.
            rb = session_cache.cached(
                s, f"cdrest-{reference}-{basis.basis_id[:8]}-{REST_BINS}",
                lambda get=get, s=s: rest_baseline(s, get(), reference),
                verbose=False)
            if rb:
                for k_, v_ in rb.items():
                    Rb.setdefault(k_, []).append(np.asarray(v_, float))
        if orth:
            # The EVENT-TRIGGERED AVERAGE over this session's pre-stroke success trials,
            # pooled over positions. Accumulated HERE because pass 1 already holds the
            # signal; computing it in pass 2 would be circular, since pass 2 needs the
            # directions that the CIM helps define.
            # THE SAME TRIALS THE TRACE USES, so the mode being projected out is the shared
            # response of what is actually plotted.
            gate_at = [f for c in use for f in arms[c]["at"]]
            gm = session_cache.cached(
                s, f"cdgm-{align}-{'+'.join(use)}-{basis.basis_id[:8]}-{pre_n}-{post_frames}",
                lambda get=get, gate_at=gate_at: _event_average(get(), gate_at,
                                                                pre_n, post_frames),
                verbose=False)
            if gm is not None:
                Gs.append(gm[0])
                Gn.append(gm[1])
    if not Xf:
        return {"animal": animal, "align": align, "skipped": "no pre-stroke success trials",
                "errors": errs}
    Xall = np.vstack(Xf)
    stats = component_stats(Xall) if STANDARDISE else None
    cim = None
    if orth and Gs:
        wts = np.asarray(Gn, float)
        G = np.tensordot(wts / wts.sum(), np.stack(Gs), axes=(0, 0))   # n-weighted grand mean
        if stats is not None:
            G = (G - stats[0][:, None]) / stats[1][:, None]
        cim = condition_independent_modes(G, np.arange(-pre_n, post_frames) / args.fs)
    if reference == "contrast":
        dirs, surviving = fit_directions(Xall, np.concatenate(yf), DISPLAY_ORDER, method=method,
                                         stats=stats, cim=cim, with_surviving=True)
    elif not Rb:
        return {"animal": animal, "align": align,
                "skipped": f"no {reference} baseline on any pre-stroke session",
                "errors": errs}
    else:
        # SESSION-AVERAGED, matching how `Xall` pools sessions. A per-session subtrahend would
        # be a different reference frame per day, which is what the frozen basis exists to
        # avoid (rule 10).
        base = {k_: np.mean(np.stack(v_), 0) for k_, v_ in Rb.items()}
        dirs = directions_vs_rest(Xall, np.concatenate(yf), base, DISPLAY_ORDER,
                                  stats=stats, cim=cim)
        surviving = {}      # the rest references do not renormalise, so nothing collapses
    if not dirs:
        return {"animal": animal, "align": align, "skipped": "no position could be fitted",
                "errors": errs}

    # ---- PASS 2: the trajectories, from CACHED per-position CD time courses ----------------------
    kind = courses_cache_kind(align, f"{method}-{reference}", f"basis:{basis.basis_id}",
                              dirs, args.post_s, stats=stats)
    draw = use
    acc, Ge = {}, {}
    for s, ep, arms in book:
        if not any(arms[c]["y"] for c in draw):
            continue
        get = _OnceSignal(joint_locanmf.BasisSource(basis, s))
        courses = session_cache.cached(
            s, kind,
            lambda get=get: cd_courses(get(), dirs, args.fs, args.post_s, stats=stats),
            verbose=False)
        if orth and arms["success"]["y"]:
            # THE SAME TRIALS THE TRACE USES, so the mode being projected out is the shared
            # response of what is actually plotted.
            gate_at = [f for c in use for f in arms[c]["at"]]
            gm = session_cache.cached(
                s, f"cdgm-{align}-{'+'.join(use)}-{basis.basis_id[:8]}-{pre_n}-{post_frames}",
                lambda get=get, arms=arms: _event_average(get(), arms["success"]["at"],
                                                          pre_n, post_frames),
                verbose=False)
            if gm is not None:
                Ge.setdefault(ep, []).append(gm)
        for cls in draw:
            ys = np.asarray(arms[cls]["y"])
            if not ys.size:
                continue
            at = np.asarray(arms[cls]["at"])
            # POOLED ACROSS THE GATE'S CLASSES, at the TRIAL level -- one bucket, not one per class.
            #
            # THE FULL CROSS: every position's trials on every position's CD. The courses are
            # already computed for all six directions over the whole session, so slicing trials of
            # ANY position out of ANY course is free. The diagonal is the per-position trace; the
            # off-diagonal is the selectivity the `cross` layout draws.
            for cd_p, course in courses.items():
                for tr_p in np.unique(ys):
                    m = ys == tr_p
                    if m.any():
                        acc.setdefault((ep, int(cd_p), int(tr_p)), []).append(
                            slice_trials(course, at[m], pre_n, post_frames))

    out = {"animal": animal, "align": align, "method": method,
           "basis_id": basis.basis_id, "ncomp": int(basis.ncomp),
           "fs": float(args.fs), "span": SPAN[align], "pre_n": pre_n, "post_n": post_frames,
           "positions": sorted(dirs), "n_sessions": len(book), "errors": errs,
           "win_s": float(args.post_s), "standardised": bool(STANDARDISE),
           "smooth_s": float(SMOOTH_S), "orth": bool(orth and cim is not None),
           "gate": gate, "reference": reference, "fit_on": list(use),
           "cim_k": (None if cim is None else int(np.asarray(cim).shape[1])),
           # How much of each position's ORIGINAL separation survived the projection. The traces are
           # scaled by the unrotated gap, so a low value means the panel is genuinely small, not
           # that it was rescaled -- and it is the per-position anchor under `--orth`.
           "surviving": {int(k): float(v) for k, v in (surviving or {}).items()},
           "traces": {}}
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
    if Ge:
        tt = np.arange(-pre_n, post_frames) / args.fs
        by_ep = {}
        for ep_, chunks in Ge.items():
            wq = np.asarray([c[1] for c in chunks], float)
            Gq = np.tensordot(wq / wq.sum(), np.stack([c[0] for c in chunks]), axes=(0, 0))
            by_ep[ep_] = ((Gq - stats[0][:, None]) / stats[1][:, None]
                          if stats is not None else Gq)
        out["cim_cos"] = cim_rotation(by_ep, tt)
        out["cim_chance"] = subspace_chance(int(basis.ncomp), int(out["cim_k"] or 1))
        # ORIENTATION AND MAGNITUDE TOGETHER -- the cosine is scale-invariant and would score a
        # halved-but-unrotated response as unchanged.
        out["cim_scale"] = cim_scale(by_ep, tt)
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
            if k[0] == "pre" and k[1] == k[2]]
    return float(np.nanmean(vals)) if vals else float("nan")


EPOCHS = ("pre", "acute", "subacute", "chronic")

#: Epoch colours for the OVERLAY layout. Sequential rather than categorical: the four epochs are
#: ordered in time, and a categorical palette would hide that.
EPOCH_COLOR = {"pre": "#111111", "acute": "#D95F02", "subacute": "#7570B3", "chronic": "#1B9E77"}

#: Position colours for the CROSS layout, where six traces share a panel. The point of that panel
#: is which ONE of the six rises, so the on-diagonal trace is drawn heavy and the rest light.
POS_COLOR = {1: "#4C72B0", 0: "#DD8452", 2: "#55A868", 4: "#C44E52", 3: "#8172B3", 5: "#937860"}


def _axis_furniture(ax, res, t):
    """The 0/1 poles and the event line. Shared by every layout so they cannot drift apart."""
    ax.axhline(0, color="0.55", lw=0.8)
    ax.axhline(1, color="0.55", lw=0.8, ls=":")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=7)


def _suptitle(res, extra=""):
    zero = {"precue": "cue", "cue": "cue", "lick": "first lick"}[res["align"]]
    cos = res.get("cim_cos") or {}
    cos_s = ("" if not cos else f"  ·  CIM K={res.get('cim_k')} overlap vs pre: "
             + " ".join(f"{e[:3]}={cos[e]:.2f}" for e in EPOCHS if cos.get(e) is not None)
             + f" (chance {res.get('cim_chance', float('nan')):.3f})")
    win = ("[-%g,0)" % res["win_s"]) if res["align"] == "precue" else ("[0,%g)" % res["win_s"])
    return (
        f"{res['animal']} — per-position {res['align'].upper()} coding direction, projected frame "
        f"by frame{extra}\n"
        f"fitted on PRE-STROKE {'+'.join(res['fit_on']).upper()} vs "
        f"{{'contrast': 'OTHER POSITIONS', 'rest': 'REST (flat, time-local)', 'restw': 'ITS OWN REST'}}[res.get('reference', 'contrast')]"
        f" (window mean, time-INVARIANT, "
        f"components Z-SCORED in a frozen pre-stroke frame"
        + (", CONDITION-INDEPENDENT MODE PROJECTED OUT" if res.get("orth") else "")
        + f") · basis {str(res['basis_id'])[:12]} {res['ncomp']}c\n"
        f"{res.get('smooth_s', float('nan')):g}s centred boxcar · MEAN over trials, band = 95% CI "
        f"OF THE MEAN · anchor: {res.get('anchor', float('nan')):.2f} over {win}{cos_s}\n"
        f"0 = pre-stroke not-P, 1 = pre-stroke lick at P · x = 0 is the {zero} · HAEMODYNAMICS ARE "
        f"SLOW — read amplitude and gross time course, NOT onset or ordering")


def figure_epochs(res, out):
    """LAYOUT 1: one panel per spout position, the four epochs OVERLAID.

    Priya, 2026-09-24: *"plot each spout position with overlaid pre-stroke and post-stroke epochs
    (so color coded with appropriate alpha to see overlay, one per position)"*.

    The previous layout put epochs in columns, which makes "did this position's code change after
    the lesion" a comparison across four separate axes. Overlaid, it is one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES

    pos = res["positions"]
    t = np.arange(-res["pre_n"], res["post_n"]) / res["fs"]
    nc = 3
    nr = int(np.ceil(len(pos) / nc))
    # ONE Y-AXIS ACROSS ALL SIX PANELS (Priya, 2026-09-24: "can we keep y axis the same in the
    # comparison plots"). This layout exists to compare epochs, and the pole normalisation already
    # makes the positions commensurable -- 1 means the same thing in every panel -- so a per-panel
    # scale would make two positions with different amplitudes look alike. The cost is that a
    # low-amplitude position is drawn small rather than filled to its own axis; that is the correct
    # impression here, unlike in the cross layout where the question is within-panel selectivity.
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 2.9 * nr), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    for k, p in enumerate(pos):
        ax = axes[k // nc][k % nc]
        _axis_furniture(ax, res, t)
        for ep in EPOCHS:
            d = res["traces"].get((ep, p, p))
            if d is None:
                continue
            thin = d["n"] < MIN_TRIALS
            ax.plot(t, d["mean"], color=EPOCH_COLOR[ep], lw=1.5, alpha=0.85,
                    ls="--" if thin else "-",
                    label=f"{ep} (n={d['n']})" + (" THIN" if thin else ""))
            if not thin:
                # alpha low enough that four overlapping bands stay separable
                ax.fill_between(t, d["lo"], d["hi"], color=EPOCH_COLOR[ep], alpha=0.16, lw=0)
        # The surviving fraction goes ON THE PANEL, because a position whose direction barely
        # survived the projection draws small for a reason the reader cannot otherwise see.
        sv = (res.get("surviving") or {}).get(p)
        ax.set_title(POSITION_NAMES.get(p, str(p))
                     + ("" if sv is None else f"   (survives orth: {sv:.0%})"),
                     fontsize=9,
                     color="tab:red" if (sv is not None and sv < 0.7) else "black")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for k in range(len(pos), nr * nc):
        axes[k // nc][k % nc].set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("s", fontsize=8)
    for r in range(nr):
        axes[r][0].set_ylabel("projection", fontsize=8)
    fig.suptitle(_suptitle(res, " — EPOCHS OVERLAID"), fontsize=8.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_cross(res, out):
    """LAYOUT 2: the full 6x6 -- every position's trials projected onto every position's CD.

    Priya, 2026-09-24: *"project each position's activity onto each position's CD (so per epoch 6
    plots, one per position CD, with overlaid 6 projections of the 6 trial types)"*.

    Rows are epochs, columns are the CD being projected onto, and each panel overlays the six spout
    positions' trials. **The diagonal trace is the one the panel is named for** and is drawn heavy;
    the other five are the comparison. A selective direction shows one trace rising and five flat --
    which is the claim "this is a position code" made visible, rather than inferred from a decoder
    score.

    It costs nothing extra to compute: the courses already exist for all six directions over the
    whole session, so slicing any position's trials out of any course is free.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES

    pos = res["positions"]
    eps = [e for e in EPOCHS if any(k[0] == e for k in res["traces"])]
    t = np.arange(-res["pre_n"], res["post_n"]) / res["fs"]
    fig, axes = plt.subplots(len(eps), len(pos), figsize=(2.9 * len(pos), 2.5 * len(eps)),
                             squeeze=False, sharex=True, sharey="row", constrained_layout=True)
    for i, ep in enumerate(eps):
        for j, cd_p in enumerate(pos):
            ax = axes[i][j]
            _axis_furniture(ax, res, t)
            for tr_p in pos:
                d = res["traces"].get((ep, cd_p, tr_p))
                if d is None:
                    continue
                on = tr_p == cd_p
                ax.plot(t, d["mean"], color=POS_COLOR.get(tr_p, "0.5"),
                        lw=2.0 if on else 1.0, alpha=1.0 if on else 0.65,
                        ls="-" if d["n"] >= MIN_TRIALS else "--",
                        label=(POSITION_NAMES.get(tr_p, str(tr_p))
                               + (" (own)" if on else "") + f" n={d['n']}"))
                if on and d["n"] >= MIN_TRIALS:
                    ax.fill_between(t, d["lo"], d["hi"], color=POS_COLOR.get(tr_p, "0.5"),
                                    alpha=0.18, lw=0)
            if i == 0:
                sv = (res.get("surviving") or {}).get(cd_p)
                ax.set_title("CD: " + POSITION_NAMES.get(cd_p, str(cd_p))
                             + ("" if sv is None else f"\n(survives orth: {sv:.0%})"),
                             fontsize=8.5,
                             color="tab:red" if (sv is not None and sv < 0.7) else "black")
            if j == 0:
                ax.set_ylabel(ep, fontsize=9)
            if i == 0 and j == len(pos) - 1:
                ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=1)
    for ax in axes[-1]:
        ax.set_xlabel("s", fontsize=8)
    fig.suptitle(_suptitle(res, " — 6x6 CROSS-PROJECTION (heavy = the CD's own position)"),
                 fontsize=8.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


#: The layouts `--layout` can ask for.
LAYOUTS = {"epochs": figure_epochs, "cross": figure_cross}


def _render_animal(item):
    """Every figure for ONE animal. Module level so spawn can pickle it by name (rule 6).

    THE ANIMAL IS THE PARALLEL UNIT, and the loops inside it stay serial deliberately. Each animal
    has its OWN frozen joint basis -- a shared object served from MICROSCOPE -- so animals are
    independent and fan out cleanly, while the alignments, gates and layouts within an animal reuse
    that one basis and the caches it warms. Fanning over the inner combinations instead would reload
    the same ~100 MB basis once per combination and throw away every warm `cdarms-`/`cdfit-`/
    `cdcourse-` entry the previous combination just wrote.
    """
    from pathlib import Path as _P

    from wfield_local import cd_trajectories as cdt

    out, made, errs = _P(item["out"]), [], []
    for align in item["aligns"]:
        for gate in item["gates"]:
            for orth in item["orths"]:
                try:
                    res = cdt.analyse_animal(item["animal"], align, method=item["method"],
                                             post_s=item["post_s"], orth=orth, gate=gate,
                                             reference=item["reference"], verbose=False)
                except Exception as exc:
                    errs.append(f"{item['animal']} {align} {gate} orth={orth}: "
                                f"{type(exc).__name__}: {exc}"[:160])
                    continue
                if res.get("skipped"):
                    errs.append(f"{item['animal']} {align} {gate}: SKIPPED {res['skipped']}")
                    continue
                tag = "_".join([item["method"], item["reference"], gate]
                               + (["orth"] if orth else []))
                for lay in item["layouts"]:
                    try:
                        made.append(str(cdt.LAYOUTS[lay](
                            res, out / f"cd_{lay}_{item['animal']}_{align}_{tag}.png")))
                    except Exception as exc:
                        errs.append(f"{item['animal']} {align} {lay}: "
                                    f"{type(exc).__name__}: {exc}"[:160])
    return {"made": made, "errors": errs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--align", nargs="+", default=["precue"],
                    choices=("precue", "cue", "lick"))
    ap.add_argument("--layout", nargs="+", default=["epochs", "cross"],
                    choices=tuple(LAYOUTS),
                    help="epochs = one panel per position with the four epochs OVERLAID; "
                         "cross = the full 6x6, every position's trials on every position's CD")
    ap.add_argument("--gate", nargs="+", default=["lick"], choices=tuple(GATES),
                    help="which trials the direction is fitted on AND the trace is built from. "
                         "`lick` = success only; `lick_or_working` = success + miss-while-working "
                         "pooled at the trial level. Run both to see whether miss-while-working "
                         "carries the same position code as licks.")
    ap.add_argument("--method", default="dom", choices=("dom", "lr"))
    ap.add_argument("--reference", nargs="+", default=["contrast"], choices=REFERENCES,
                    help="what position P is contrasted AGAINST. `contrast` = the other five positions (the default everywhere else, and the one whose directions sum to ~0 and so force a shared signal to split sign). `rest` = the flat time-local rest baseline, identical for all six, so nothing cancels. `restw` = each position against ITS OWN rest frames -- conservative, since rest itself carries position information.")
    ap.add_argument("--orth", nargs="+", default=["off"], choices=("off", "on"),
                    help="project the CONDITION-INDEPENDENT MODE out of every direction. The "
                         "one-vs-rest directions nearly cancel (their sum is 0.289 of a possible "
                         "6.0), so the shared response is forced positive on some positions and "
                         "negative on others -- which is what the close-position dips are. Give "
                         "both to render both; neither replaces the other, because orthogonalising "
                         "also removes any real position information lying along that mode.")
    ap.add_argument("--post-s", type=float, default=None,
                    help="override decode.{align}_post_s; default reads it per alignment")
    ap.add_argument("--jobs", type=int, default=None, help="parallel animals (default: cores-2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.out:
        out = Path(args.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("figures_working"))
    Path(out).mkdir(parents=True, exist_ok=True)

    animals = args.animal or ["PS92", "PS93", "PS94", "PS95"]
    items = [{"animal": a, "aligns": list(args.align), "layouts": list(args.layout),
              "gates": list(args.gate), "orths": [o == "on" for o in args.orth],
              "method": args.method, "reference": r, "post_s": args.post_s,
              "out": str(out)}
             for a in animals for r in args.reference]
    n = len(items) * len(args.align) * len(args.gate) * len(args.orth) * len(args.layout)
    # KEY ON (animal, reference): the same animal under two references is two independent
    # jobs, but they share the basis and every `cdarms-`/`cdfit-` entry, so the sort keeps
    # them adjacent rather than interleaved with other animals.
    print(f"[cd_traj] {len(items)} animal(s) -> {n} figure(s)", flush=True)

    from wfield_local import analysis_kit as ak

    # `fan_sessions` sorts by the ITEM and a dict is not orderable, so the key is required -- and
    # the sort is not optional: completion order would reorder the log between identical runs.
    res, fail = ak.fan_sessions(items, _render_animal, jobs=args.jobs,
                                key=lambda it: (it["animal"], it["reference"]),
                                label="animal")
    made, errs = [], [f"{f}" for f in fail]
    for _it, val in res:
        if isinstance(val, dict):
            made += val["made"]
            errs += val["errors"]
    for m in made:
        print(f"[cd_traj] -> {m}", flush=True)
    for e in errs:
        print(f"[cd_traj] !! {e}", flush=True)
    print(f"[cd_traj] {len(made)} figure(s), {len(errs)} problem(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
