"""WHERE the position code rotates -- Haufe decoder PATTERNS in the shared joint basis.

WHAT THE TRANSFER MATRIX LEFT OPEN. `epoch_15g` established that the chronic code is a PARTIAL
ROTATION of the pre-stroke one (transfer symmetric in both directions, ~0.78-0.82 of ceiling for
cue and lick, 0.64/0.57 for ENL) and that every window ends ABOVE its pre-stroke ceiling. Both are
scalars. Neither says WHICH CORTEX the rotated fifth occupies, or where the extra information came
from. This does.

WHAT A HAUFE PATTERN IS, and why the decoder's weights are the wrong thing to plot. A weight is a
FILTER: a logistic regression can put large weight on a channel carrying no position signal at all,
purely to cancel correlated noise in a channel that does -- a suppressor variable. So a weight map
answers "what does the readout multiply?". The Haufe transform (Haufe et al. 2014) converts it to a
PATTERN, ``A = Cov(X) @ beta``, the covariance between each channel and the decoder's output, which
is the anatomical question: "which cortex actually co-varies with position". **In this dataset the
filter and the pattern correlate at only r = 0.245**, so this is not a refinement.

HOW THIS DIFFERS FROM `epoch_14` / `epoch_15r`, WHICH ARE ALSO HAUFE MAPS AND ALREADY EXIST. Those
fit PER SESSION in that session's OWN SVD basis and reference each epoch's map to a baseline; they
answer "where is each position's pattern, and how did it change". This fits ONE model per EPOCH in
the SHARED JOINT basis -- the same models the transfer matrix scores -- so the maps are directly
comparable across epochs and the difference between two of them IS the rotation the matrix measured.
Different question, same transform. Do not read one as a re-render of the other.

SHAPE, NOT GAIN. Each position's pattern is normalised to unit length before differencing, because
a decoder trained on a higher-SNR epoch produces a larger pattern for reasons that have nothing to
do with rotation, and the transfer result is about DIRECTIONS. The un-normalised amplitudes are in
the CSV for anyone who wants the gain question instead; `evoked_amplitude` and the rescale analyses
already own that one.

THE COSINE IS THE SUMMARY, THE MAP IS THE LOCALISATION. ``cos(pattern_pre, pattern_epoch)`` per
position is "how far did this position's readout direction turn"; the map shows where the turn
lives. Reported for acute, subacute AND chronic against pre (Priya, 2026-09-17), because the
matrix's own diagonal shows the three epochs are not on one trajectory -- acute DIPS and chronic
overshoots.

RUN AS:  python -m scripts.rest_migration.rotation_maps [--arms ENL cue lick rest]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from scripts.rest_migration.transfer_arms import ARMS, collect_rest, collect_task


def _xyg(data, labels):
    """Pool sessions into ``(X, y, blocks)``, block ids made unique across sessions."""
    X = np.concatenate([data[k][1] for k in labels])
    y = np.concatenate([data[k][2] for k in labels])
    g = np.concatenate([np.asarray(data[k][3], np.int64) + 1_000_000 * (i + 1)
                        for i, k in enumerate(labels)])
    return X, y, g


def _patterns_from(fit, X, basis, positions):
    """``({position: unit pattern}, {position: raw norm})`` for one fitted model."""
    from wfield_local import transfer_matrix as tm

    pats, norms = {}, {}
    for pos in positions:
        raw = tm.haufe_pattern(fit, X, pos)
        if raw is None:
            continue
        comp = tm.fold_bins(raw, basis.ncomp)
        nrm = float(np.linalg.norm(comp))
        if nrm > 0:
            pats[pos], norms[pos] = comp / nrm, nrm
    return pats, norms


def epoch_patterns(data, basis, pipe_fn, *, min_sessions=1, seed_ns="", log=print):
    """``(patterns, norms, ceiling)`` -- per epoch, per position, unit-norm Haufe patterns.

    ONE MODEL PER EPOCH, trained on that epoch's pooled sessions at the SAME block-matched size the
    transfer matrix uses, so these are the very models whose transfer was measured.

    ``min_sessions`` IS 1 HERE, NOT 2 (Priya, 2026-09-17: "PS95 should be included in acute data").
    The transfer matrix uses 2 because an epoch there is a TRAIN SOURCE and a pooled model must not
    be one session wearing an epoch's name. A PATTERN is a different object: PS95 has exactly one
    acute session, and its acute pattern is noisier but not invalid. Dropping it silently removed a
    whole animal from the acute column. Single-session epochs are reported with ``n_sessions`` in
    the CSV and cannot carry a split-half reliability (that needs >= 4), so their corrected cosine
    is None -- visible, rather than absent.

    ``ceiling`` IS WHAT MAKES THE COSINES READABLE, and the first version of this analysis omitted
    it. Two decoders fit on finite data disagree even when nothing changed, so a cross-epoch cosine
    is attenuated by estimation noise by an unknown amount -- 0.64 could be a large rotation or a
    noisy estimate of none. The ceiling splits the PRE-STROKE sessions into two DISJOINT halves,
    fits each on its own half, and takes the cosine between them: what an unchanged code yields
    with this much data. **Read every cross-epoch cosine against it, never against 1.0.**

    SPLIT BY SESSION, NOT BY TRIAL. A trial-level split leaves both halves carrying the same day's
    drift, which inflates the ceiling and would make every real rotation look larger than it is.
    """
    import hashlib

    from wfield_local import transfer_matrix as tm

    by_ep = {e: [k for k, v in data.items() if v[0] == e] for e in tm.EPOCH_ORDER}
    usable = [e for e in tm.EPOCH_ORDER if len(by_ep[e]) >= min_sessions]
    if "pre" not in usable:
        return {}, {}, {}, {}, {}      # arity must match the success path -- five, not three
    n_target = int(min(sum(len(data[k][2]) for k in by_ep[e]) for e in usable))

    pats, norms, n_sess = {}, {}, {}
    for ep in usable:
        X, y, g = _xyg(data, by_ep[ep])
        seed = int(hashlib.sha1(f"{seed_ns}|{ep}".encode()).hexdigest()[:8], 16)
        got = tm._matched(pipe_fn, X, y, g, n_target, np.random.default_rng(seed))
        if got is None:
            continue
        pp, nn = _patterns_from(got[0], X, basis, sorted(np.unique(y).tolist()))
        if pp:
            pats[ep], norms[ep] = pp, nn
            n_sess[ep] = len(by_ep[ep])

    # RELIABILITY PER EPOCH, not just for pre -- and the first version's pre-only ceiling was
    # actively misleading. It split pre in half, so its models saw 5-6 sessions while the epoch
    # models saw 11, making it systematically PESSIMISTIC: for PS92's ENL arm the "ceiling" came
    # out at +0.260 while the cross-epoch cosines were 0.410-0.437, i.e. ABOVE their own ceiling,
    # which is not interpretable at all.
    #
    # Each epoch's own split-half reliability is measured instead, and the cross-epoch cosine is
    # corrected for attenuation in BOTH: cos / sqrt(rel_pre * rel_epoch). That is the standard
    # correction and it handles epochs of different reliability, which is exactly the situation
    # here (pre has 11 sessions, acute 4-6, subacute 2-7).
    reliability = {}
    for ep in usable:
        labs = sorted(by_ep[ep])
        if len(labs) < 4:
            continue                      # cannot split; reliability stays unknown for this epoch
        rng = np.random.default_rng(
            int(hashlib.sha1(f"{seed_ns}|rel|{ep}".encode()).hexdigest()[:8], 16))
        shuf = [str(x) for x in rng.permutation(labs)]
        halves, half = [shuf[: len(shuf) // 2], shuf[len(shuf) // 2:]], []
        for hi, hl in enumerate(halves):
            Xh, yh, gh = _xyg(data, hl)
            got = tm._matched(pipe_fn, Xh, yh, gh, min(n_target, len(yh)),
                              np.random.default_rng(1000 + hi))
            if got is None:
                half = []
                break
            half.append(_patterns_from(got[0], Xh, basis, sorted(np.unique(yh).tolist()))[0])
        if len(half) == 2:
            reliability[ep] = {pos: float(half[0][pos] @ half[1][pos])
                               for pos in sorted(set(half[0]) & set(half[1]))}
    ceiling = reliability.get("pre", {})
    log(f"   patterns for {sorted(pats)}; split-half reliability "
        + (", ".join(f"{e}:{np.mean(list(v.values())):+.2f}"
                     for e, v in sorted(reliability.items())) or "NOT COMPUTABLE"))
    # THE HALVES ARE RETURNED, not just their cosine, because the |difference| between them is the
    # NULL MAP -- what "where did it change" looks like when nothing changed. Measured 2026-09-17:
    # the null map and the acute-minus-pre map correlate at r = +0.83 (PS92) and +0.81 (PS93), so
    # the localisation is mostly basis-shaped estimation noise and CANNOT carry a regional claim
    # on its own. Retrosplenial topping every panel was this, not the lesion.
    return pats, norms, ceiling, reliability, n_sess


def null_delta(data, basis, pipe_fn, *, n_draws=12, n_target=None, seed_ns="", log=print):
    """``{position: (n_draws, ncomp)}`` of |pattern change| between two PRE-STROKE halves.

    THE FIX FOR THE MAP CONFOUND, and the confound is real: measured 2026-09-17, the raw
    |acute - pre| map correlates with this null at **r = +0.83**, so two thirds of its spatial
    structure is present when nothing changed. Retrosplenial topped every panel because the basis
    puts its largest components there -- footprint mass spans 67x across components -- and a
    component-space unit norm projects them to correspondingly large pixel mass whatever their
    coefficient. The bright region was where the NOISE is, not where the lesion acted.

    So the map is not compared against zero. Each COMPONENT is compared against ITS OWN null: a
    large, noisy component has to clear a correspondingly large bar, which removes the area
    weighting rather than merely flagging it.

    Each draw splits the pre-stroke sessions into two disjoint halves BY SESSION, fits a model on
    each, and records the per-component |difference| of their sign-aligned unit patterns. Different
    split every draw, so the spread reflects which sessions land together as well as the fit noise.
    """
    import hashlib

    from wfield_local import transfer_matrix as tm

    pre_labs = sorted([k for k, v in data.items() if v[0] == "pre"])
    if len(pre_labs) < 4:
        log("   null: NOT COMPUTABLE (<4 pre sessions)")
        return {}
    out, cos_out, gain_out = {}, {}, {}
    for d in range(n_draws):
        rng = np.random.default_rng(
            int(hashlib.sha1(f"{seed_ns}|null{d}".encode()).hexdigest()[:8], 16))
        shuf = [str(x) for x in rng.permutation(pre_labs)]
        halves, ok = [shuf[: len(shuf) // 2], shuf[len(shuf) // 2:]], True
        pats, half_norms = [], []
        for hi, labs in enumerate(halves):
            X, y, g = _xyg(data, labs)
            tgt = min(n_target or len(y), len(y))
            got = tm._matched(pipe_fn, X, y, g, tgt, np.random.default_rng(7919 * d + hi))
            if got is None:
                ok = False
                break
            pp, nn = _patterns_from(got[0], X, basis, sorted(np.unique(y).tolist()))
            pats.append(pp)
            half_norms.append(nn)
        if not ok:
            continue
        # RESIDUAL LOG-GAIN between the two halves: log(n1/n0) per position, MINUS the mean across
        # positions. The subtraction removes the epoch-wide scale factor, and that is the whole
        # point: measured 2026-09-17, the common factor is 1.11x the position-specific spread, so
        # a test on RAW gain would mostly be testing how loud the epoch was rather than whether
        # this position's pattern changed relative to its neighbours.
        shared_n = sorted(set(half_norms[0]) & set(half_norms[1]))
        if len(shared_n) >= 2:
            lg = {q: float(np.log(half_norms[1][q] / half_norms[0][q])) for q in shared_n}
            mu = float(np.mean(list(lg.values())))
            for q, v in lg.items():
                gain_out.setdefault(q, []).append(v - mu)
        for pos in sorted(set(pats[0]) & set(pats[1])):
            c = float(pats[0][pos] @ pats[1][pos])
            dd = np.abs((pats[1][pos] if c >= 0 else -pats[1][pos]) - pats[0][pos])
            out.setdefault(pos, []).append(dd)
            # THE COSINE IS KEPT, not just the difference map. These draws ARE the null for "two
            # models of the SAME code, differing only by estimation noise", so the distribution of
            # their cosines is exactly what an observed cross-epoch cosine has to be tested
            # against. The first version computed them and threw them away, which is why the
            # rotation numbers were point estimates with no p attached.
            cos_out.setdefault(pos, []).append(c)
    res = {p: np.asarray(v) for p, v in out.items() if len(v) >= 3}
    cos_res = {p: np.asarray(v) for p, v in cos_out.items() if len(v) >= 3}
    gain_res = {p: np.asarray(v) for p, v in gain_out.items() if len(v) >= 3}
    log(f"   null: {len(res)} positions x {min((len(v) for v in out.values()), default=0)} draws")
    return res, cos_res, gain_res


def gain_p(observed_resid, null_resid):
    """TWO-SIDED p on a position's amplitude change RELATIVE TO ITS NEIGHBOURS.

    ``observed_resid`` is log(gain) minus the mean log(gain) over the six positions, so the
    epoch-wide scale is already divided out; the null is the same residual measured between two
    halves of the PRE-STROKE data, where no epoch difference exists. Two-sided because a position
    may grow or shrink relative to the others and both are findings.

    THE RESIDUAL FORM IS NOT OPTIONAL. Raw gain is ~half epoch-wide scale (the common factor is
    1.11x the position-specific spread), so a test against a raw-gain null would report the
    epoch's loudness as a position effect.
    """
    if null_resid is None or len(null_resid) < 3:
        return None
    n = len(null_resid)
    return (1 + int(np.sum(np.abs(np.asarray(null_resid)) >= abs(observed_resid)))) / (1 + n)


def cosine_p(observed, null_cos):
    """One-sided p that a cosine this LOW arises from estimation noise alone.

    ``(1 + #{null <= observed}) / (1 + n)`` -- the add-one form, so p is never 0 and the floor is
    an honest function of how many draws were run. With 12 draws nothing can beat p = 0.077, which
    is why the reported run uses far more.

    ONE-SIDED AND DOWNWARD BY DESIGN: the hypothesis is that the readout ROTATED, which can only
    push the cosine DOWN relative to two noise-separated estimates of an unchanged code. A cosine
    ABOVE the null is not evidence of anti-rotation, it is a lucky draw.
    """
    if null_cos is None or len(null_cos) < 3:
        return None
    n = len(null_cos)
    return (1 + int(np.sum(np.asarray(null_cos) <= observed))) / (1 + n)


def excess_z(real_delta, null_draws):
    """Per-component z of the observed change against its OWN null. None if the null is unusable.

    ``(real - null_mean) / null_sd``. A component whose change is ordinary for the noise scores ~0
    however large it is in absolute terms, which is precisely what the raw map failed to do.
    """
    if null_draws is None or len(null_draws) < 3:
        return None
    mu, sd = null_draws.mean(0), null_draws.std(0)
    # THE SD IS FLOORED, and without it the map is unreadable. From ~10 draws a component can get a
    # near-zero SD by luck, and dividing by it produced z values above 30 -- which are a small
    # denominator, not a large effect, and they set the colour scale for every other panel. The
    # floor is the 25th percentile of the non-zero SDs in this position's own null, so it adapts to
    # the arm's noise level instead of being a constant that would be lenient for one arm and
    # crushing for another.
    pos_sd = sd[sd > 1e-12]
    floor = float(np.percentile(pos_sd, 25)) if pos_sd.size else np.inf
    sd = np.maximum(sd, floor)
    sd = np.where(sd > 1e-12, sd, np.inf)      # a component with no null spread cannot be scored
    return (np.asarray(real_delta) - mu) / sd


#: Fraction of a component's footprint MASS that must fall inside `beta_maps.stat_mask` for it to
#: enter the analysis at all. Raised 0.5 -> 0.75 on 2026-09-17 (Priya: *"we should exclude non-map,
#: olfactory, and glue masked pixels. These should also be excluded from statistical analysis"*,
#: then *"sure we can tighten to 0.75"*).
#:
#: WHY 0.5 WAS TOO LOOSE. Out-of-mask PIXELS were already excluded from every average, so a
#: half-outside component could not import bulb signal directly. What it could do is carry a full
#: vote on HALF the evidence: its value was estimated from only its surviving pixels, making it
#: noisier than an interior component while weighing the same in its region's mean and in the
#: max-statistic family. A noisy unit with a full vote is how a threshold gets set by the component
#: that deserves it least.
#:
#: THE COST, measured: components admitted go 55/49/52/48 -> 48/43/45/44 (PS92/93/94/95), about 12%.
#: 0.90 would cost 33% (37/34/30/32) and start dropping genuinely cortical components whose
#: footprints simply reach the mask edge, which is why it was not taken further.
MIN_IN_MASK_FRAC = 0.85

#: SECOND, INDEPENDENT CRITERION: how much of a component's mass must fall in the ERODED
#: `stat_mask` -- the trustworthy interior -- for it to be TESTABLE at all.
#:
#: TWO CRITERIA BECAUSE THERE ARE TWO FAILURE MODES, and one threshold on one mask cannot separate
#: them. `MIN_IN_MASK_FRAC` on `brain_mask` asks IS THIS CORTEX (bulbs and painted glue are already
#: out of that mask). This asks IS ENOUGH OF IT AWAY FROM THE WINDOW RIM, where `U` is smallest,
#: the Allen warp least constrained, and the artefact CONSISTENT ACROSS ANIMALS -- which a
#: between-animal denominator rewards rather than rejects.
#:
#: WHY NOT JUST GATE ON `stat_mask`, WHICH IS WHAT THE FIRST VERSION DID (2026-09-17, superseded
#: the same evening): the erosion is 16 px and it removes 30% of the SSp-m ATLAS REGION, so a gate
#: built on it selects against LATERAL cortex per se. PS93's SSp-m_left components are 97.5% inside
#: `brain_mask` and were excluded at 0.75 purely because 27-29% of their mass lies in that rim --
#: and because the `15k` vocabulary is an INTERSECTION over animals, that removed ipsilesional
#: mouth cortex for all four, in a cohort whose deficit is orofacial.
#:
#: MEASURED ON THE EIGHT REGIONS THE RELAXATION ADMITS (Priya, 2026-09-17: "I think FRP, RSP, VIS
#: are probably not real", "VIS right in particular is tough to interpret bc the glue covers
#: VISl"). She is right, and the two criteria say so for three different reasons:
#:
#:     SSp-m_left    brain 0.975-0.989   stat 0.688-0.808   glue 0.011-0.026   ADMITTED
#:     RSPagl_left   brain 0.923-0.957   stat 0.621-0.865   glue 0.043-0.077   admitted
#:     VISal_right   brain 0.961-0.987   stat 0.623-0.682   glue 0.013-0.039   admitted
#:     PL_*          brain 0.891-1.000   stat 0.410-0.679   glue 0            REJECTED, half rim
#:     FRP_*         brain 0.798-1.000   stat 0.000-0.165   glue 0            REJECTED, ALL rim
#:     VISp_right    brain 0.758-0.847   stat 0.541-0.701   glue 0.153-0.242   REJECTED, glue
#:
#: FRP's entire ATLAS FOOTPRINT is inside the rim -- 389 px per side, 0 of them in `stat_mask` --
#: so it is not a region this window can test at any threshold. VISp_right loses 31.3% of its
#: footprint to the painted glue, which is why its components cannot reach 0.85 on `brain_mask`:
#: the glue is already subtracted there, so occlusion shows up as missing mass. Both are caught by
#: criteria that state what is wrong rather than by a number tuned until the list looked right.
MIN_IN_STAT_FRAC = 0.50


def in_mask_components(basis, min_frac=None, eroded=False):
    """Boolean over components: which sit INSIDE `beta_maps.brain_mask`.

    THE MASK BELONGS IN THE STATISTICS, NOT ONLY IN THE DISPLAY (Priya, 2026-09-17). The maps were
    already masked when drawn, but the TEST ran over every component -- including ones whose
    footprint lies in the olfactory bulbs or on the glue/window edge, where `U` is smallest and the
    Allen warp least constrained. Those components then (a) could be flagged significant and
    outlined, and (b) entered the max-statistic family, inflating the threshold for the components
    that ARE in cortex. Both are wrong, and the second silently costs power everywhere else.

    A component counts as in-mask when at least `min_frac` of its footprint MASS falls inside.
    Mass, not pixel count, because footprints are graded -- a component whose tail brushes the mask
    should not qualify on area alone.

    THE MASK IS `brain_mask`, NOT `stat_mask`, AND THAT WAS A BUG UNTIL 2026-09-17 EVENING. It
    gated on `stat_mask`, which is `brain_mask` ERODED BY 16 px -- an erosion that exists for a
    different purpose entirely: it stops PIXEL-LEVEL contours being drawn in the window rim, where
    edge enrichment was measured at ~2x (see `beta_maps.stat_mask`). Applied to a COMPONENT it
    asks a question it was never built to answer, and it penalises a region for being LATERAL
    rather than for being untrustworthy.

    WHAT IT COST, measured on PS93's SSp-m_left: **97.5% of that component's mass is inside
    `brain_mask`** -- only 2.5% is genuinely off-brain -- but **27-29% lies in the 16 px rim**, so
    it scored 0.69-0.71 and failed a 0.75 gate. Since the `15k` vocabulary is an INTERSECTION over
    animals, one animal's failure removed SSp-m_left for all four, and SSp-m_left is ipsilesional
    mouth cortex in the cohort whose deficit is orofacial. The systematic version: the SSp-m ATLAS
    REGION loses 30% of its pixels to the erosion, and every SSp-m component in every animal loses
    17-29% of its mass -- so the gate was selecting against lateral cortex per se.

    `brain_mask` STILL EXCLUDES EVERYTHING THE GATE WAS BUILT FOR: it is `allen_mask` minus
    `EXCLUDE_REGIONS` (the olfactory bulbs) minus the hand-painted fibre-glue occlusion. Only the
    16 px rim comes back. Pass ``eroded=True`` to reproduce the superseded behaviour.

    BUT THE RIM STILL HAS TO BE ACCOUNTED FOR, so a SECOND criterion does it explicitly:
    `MIN_IN_STAT_FRAC` of the mass must fall in the eroded interior. Relaxing the first criterion
    alone admitted FRP -- whose entire 389 px atlas footprint lies in the rim, 0 px in `stat_mask`
    -- which is a region this window cannot test at any threshold, not a region the old gate was
    unfairly excluding. Two failure modes, two criteria, each saying what is actually wrong.
    """
    from wfield_local import beta_maps as bm

    # RESOLVED AT CALL TIME, not bound as a default. A default argument is evaluated when the
    # function is DEFINED, so `--min-in-mask-frac` could never reach it and the two threshold runs
    # would silently be the same run twice. Reading the module global here makes the flag live.
    if min_frac is None:
        min_frac = MIN_IN_MASK_FRAC
    A = np.nan_to_num(np.asarray(basis.A, dtype=np.float32))
    flat = np.abs(A.reshape(-1, basis.ncomp))
    m = np.asarray(bm.stat_mask() if eroded else bm.brain_mask(), bool)
    interior = None if eroded else np.asarray(bm.stat_mask(), bool)
    if m.shape != A.shape[:2]:
        # LOUD, NEVER SILENT. Returning all-True here removes the mask from the statistics
        # entirely -- every bulb and glue-edge component re-enters the max-statistic family and
        # the threshold rises for the components that matter. This repo has already been bitten
        # once by the U / U_atlas grid mismatch (460x480 vs 540x640), and that failure looks
        # exactly like a run that worked.
        print(f"   !! GRID MISMATCH: stat_mask {m.shape} vs basis {A.shape[:2]} -- "
              f"THE MASK IS NOT IN THE STATISTICS FOR THIS ANIMAL", flush=True)
        return np.ones(basis.ncomp, bool)
    total = np.where(flat.sum(0) > 0, flat.sum(0), np.inf)
    keep = np.divide(flat[m.ravel(), :].sum(0), total) >= min_frac
    if interior is not None and interior.shape == A.shape[:2]:
        # THE SECOND CRITERION. Without it, relaxing the first admits components that are
        # essentially ALL RIM -- FRP scores 0.80-1.00 on `brain_mask` and 0.00-0.17 on
        # `stat_mask`, because its whole atlas footprint lies inside the eroded band.
        keep &= np.divide(flat[interior.ravel(), :].sum(0), total) >= MIN_IN_STAT_FRAC
    return keep


def bootstrap_cosine_ci(data, basis, pipe_fn, ep, *, n_boot=200, n_target=None,
                        seed_ns="", alpha=0.05, log=print, return_draws=False):
    """``{position: (lo, hi)}`` -- percentile CI on cos(pre, epoch) by RESAMPLING SESSIONS.

    THE WHISKERS THIS REPLACES WERE NOT A CI. They were the min-max RANGE across three or four
    animals, which on a bar chart reads as an error bar and is not one (Priya, 2026-09-17: "are
    these bars really CI? it just looks like animal spread").

    WHY IT NEEDS A REFIT PER RESAMPLE, and why `stats.permutation.bootstrap_ci` in
    stroke_orofacial cannot be called directly. That helper bootstraps a SAMPLE OF VALUES; the
    cosine here is ONE number per cell, computed from a model fitted on all of that epoch's
    sessions. There is no sample to resample without refitting, so each draw resamples SESSIONS
    with replacement, refits both the pre and the epoch model, and recomputes the cosine. The
    interval then carries session-level variability, which is the replicate this project's nested
    bootstrap resamples.

    PERCENTILE, NOT BCa. BCa needs a jackknife acceleration estimate over the same unit; with 4-11
    sessions per epoch that estimate is itself unstable, and the bias correction would be noise.
    Percentile is the honest choice at this n.

    WITHIN ANIMAL. This is each animal's own interval, not a cohort CI -- with three animals
    carrying chronic data a resample-animals interval would be three points wide and would imply
    a precision the design does not have. The cohort bar stays a mean with its per-animal points.

    THAT LAST SENTENCE IS NO LONGER THE WHOLE STORY, and the reason is worth stating here rather
    than only in the new module. "A mean with its per-animal points" is what `15k` already
    rejected for itself: `cohort_delta` was added there precisely because counting how many
    animals clear a threshold is "a replication count wearing a cohort test's clothes", with a
    per-cell false-positive rate of ~1 - 0.95^4 = 18% under an any-animal rule. The same argument
    applies to this arm. `rotation_stats.py` builds the cohort statistic from the draws this
    function returns under `return_draws`, as a CI and never a p -- with four animals an
    animal-level sign-flip null has 2^4 = 16 assignments, so 0.0625 is its floor and no cell
    could reach 0.05 at any effect size. The narrowness objection above stands and is the reason
    the cohort number is reported as an interval with its n_animals beside it.
    """
    import hashlib

    from wfield_local import transfer_matrix as tm

    pre_labs = [k for k, v in data.items() if v[0] == "pre"]
    ep_labs = [k for k, v in data.items() if v[0] == ep]
    if not pre_labs or not ep_labs:
        return {}
    acc = {}
    for b in range(n_boot):
        rng = np.random.default_rng(
            int(hashlib.sha1(f"{seed_ns}|boot|{ep}|{b}".encode()).hexdigest()[:8], 16))
        got = {}
        for name, labs in (("pre", pre_labs), (ep, ep_labs)):
            pick = [labs[i] for i in rng.integers(0, len(labs), len(labs))]
            X, y, g = _xyg(data, pick)
            fit = tm._matched(pipe_fn, X, y, g, min(n_target or len(y), len(y)),
                              np.random.default_rng(31 * b + len(name)))
            if fit is None:
                got = {}
                break
            got[name] = _patterns_from(fit[0], X, basis, sorted(np.unique(y).tolist()))[0]
        if len(got) != 2:
            continue
        for pos in sorted(set(got["pre"]) & set(got[ep])):
            acc.setdefault(pos, []).append(float(got["pre"][pos] @ got[ep][pos]))
    out = {pos: (float(np.percentile(v, 100 * alpha / 2)),
                 float(np.percentile(v, 100 * (1 - alpha / 2))))
           for pos, v in acc.items() if len(v) >= 20}
    log(f"   bootstrap {ep}: {len(out)} positions x "
        f"{min((len(v) for v in acc.values()), default=0)} resamples")
    if return_draws:
        # THE DRAWS, NOT JUST THEIR PERCENTILES. A cohort statistic resamples ANIMALS and then
        # sessions within animal; these are already the session-level resamples, so keeping them
        # makes the animal level a re-read rather than a second two-hour refit. Same reasoning as
        # `epoch_15k_region_vectors_*.npz` -- "any later statistic is a re-read, not another pass".
        return out, {pos: np.asarray(v, np.float32) for pos, v in acc.items() if len(v) >= 20}
    return out


def significant_components(real_delta, null_draws, alpha=0.05, keep=None):
    """Boolean over components: which changed MORE than the family-wise null allows.

    THE COMPONENT IS THE UNIT OF TEST, not the pixel. The pixel map is a PROJECTION of component
    values through overlapping footprints, so a contour drawn at a z threshold on the projected map
    is not a significance statement about that pixel -- neighbouring pixels are not independent
    tests, they are the same components seen through different mixing weights.

    FAMILY-WISE BY MAX-STATISTIC, over the components within this position. Each draw contributes
    its single most extreme component z; the threshold is the (1-alpha) quantile of those maxima.
    That respects the coupling between components -- they share the drift and the basis, so their
    draws are correlated and the max distribution is narrower than independence would imply --
    and it costs no extra draws, because it reuses the ones already computed. Bonferroni over ~90
    components would be far more conservative for no gain.
    """
    if null_draws is None or len(null_draws) < 10:
        return None
    mu, sd = null_draws.mean(0), null_draws.std(0)
    pos_sd = sd[sd > 1e-12]
    floor = float(np.percentile(pos_sd, 25)) if pos_sd.size else np.inf
    sd = np.where(sd > 1e-12, np.maximum(sd, floor), np.inf)
    z_null = (null_draws - mu) / sd                      # (n_draws, ncomp)
    # OUT-OF-MASK COMPONENTS ARE EXCLUDED FROM THE FAMILY, not merely hidden afterwards. Leaving
    # them in lets an olfactory-bulb or glue-edge component set the per-draw maximum and raise the
    # threshold for every cortical component -- a power loss paid by the components we care about.
    keep = np.ones(z_null.shape[1], bool) if keep is None else np.asarray(keep, bool)
    if not keep.any():
        return None
    thresh = float(np.percentile(np.nanmax(z_null[:, keep], axis=1), 100 * (1 - alpha)))
    z_obs = (np.asarray(real_delta) - mu) / sd
    return (z_obs > thresh) & keep


def component_outline(sig, basis, frac=0.35, mask=True):
    """Pixel mask of the significant components' territory, for contouring.

    A component's footprint is graded, so "its territory" needs a cut: `frac` of that component's
    own peak. Per component rather than global, because footprint mass spans 67x and a global cut
    would erase the small ones entirely.
    """
    if sig is None or not np.any(sig):
        return None
    A = np.nan_to_num(np.asarray(basis.A, dtype=np.float32))
    H, W = A.shape[0], A.shape[1]
    flat = A.reshape(-1, basis.ncomp)
    out = np.zeros(H * W, bool)
    for c in np.flatnonzero(np.asarray(sig)):
        col = np.abs(flat[:, c])
        pk = col.max()
        if pk > 0:
            out |= col >= frac * pk
    out = out.reshape(H, W)
    if mask:
        # INTERSECT WITH THE BRAIN MASK. A kept component can still have a tail reaching into the
        # bulbs or the window edge, and an outline there reads as a finding in tissue the analysis
        # has already declared unusable.
        from wfield_local import beta_maps as bm
        m = np.asarray(bm.stat_mask(), bool)
        if m.shape == out.shape:
            out &= m
    return out


def to_pixels(comp_pattern, basis, mask=True):
    """Component-space pattern -> ``(H, W)`` cortical map through the SHARED footprints.

    MASKED TO `beta_maps.stat_mask` BY DEFAULT -- the eroded brain mask every other map family in
    this project uses. It removes the olfactory bulbs and the glue/window edge, which are exactly
    where `U` is smallest and the Allen warp least constrained, so they carry partial-volume
    mixing and read as signal (Priya, 2026-09-12 and again 2026-09-17). Outside the mask is NaN
    rather than zero, so a masked pixel cannot drag a mean or set a colour limit.
    """
    A = np.nan_to_num(np.asarray(basis.A, dtype=np.float32))
    H, W = A.shape[0], A.shape[1]
    out = (A.reshape(-1, basis.ncomp) @ np.asarray(comp_pattern, np.float32)).reshape(H, W)
    if mask:
        from wfield_local import beta_maps as bm
        m = np.asarray(bm.stat_mask(), bool)
        if m.shape == out.shape:
            out = np.where(m, out, np.nan)
    return out


def by_region(comp_pattern, basis):
    """``{allen_region: summed |pattern|}`` -- the regional answer, without a pixel map.

    Uses the basis's OWN region assignment (`Basis.regions`, derived from the footprints), so the
    labels are the same ones every other joint-basis figure uses.
    """
    out = {}
    regs = list(basis.regions)
    for c, r in enumerate(regs):
        out[str(r)] = out.get(str(r), 0.0) + abs(float(comp_pattern[c]))
    return out


def main() -> int:
    # DECLARED HERE, at the top, because `global` must precede every use of the name in the
    # function and MIN_IN_MASK_FRAC is read below as the argparse default.
    global MIN_IN_MASK_FRAC, MIN_IN_STAT_FRAC
    from wfield_local import config, joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_frozen_decoder import _pipe
    from wfield_local.paths import PathResolver

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["ENL", "cue", "lick"], choices=list(ARMS))
    ap.add_argument("--bins", type=int, default=4, help="rest arm only")
    ap.add_argument("--boot", type=int, default=0,
                    help="session-resample bootstrap draws for the cosine CI (0 = off). Each draw "
                         "REFITS both models, so this roughly doubles runtime at --boot == "
                         "--null-draws.")
    ap.add_argument("--null-draws", type=int, default=12,
                    help="pre-stroke split-half draws building the per-component null the maps "
                         "are scored against; below ~6 the SD is too noisy to divide by")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--min-in-mask-frac", type=float, default=MIN_IN_MASK_FRAC,
                    help="fraction of a component's footprint mass that must sit inside "
                         "BRAIN_MASK (bulbs and painted glue already removed). Adopted 0.85, "
                         "paired with --min-in-stat-frac. ALWAYS pair with --tag, or two "
                         "settings overwrite each other's figures, CSV and cache.")
    ap.add_argument("--min-in-stat-frac", type=float, default=MIN_IN_STAT_FRAC,
                    help="SECOND criterion: fraction of the mass that must sit in the ERODED "
                         "stat_mask, so a component that is essentially all window-rim (FRP) is "
                         "excluded even though it is entirely on brain. Adopted 0.50.")
    ap.add_argument("--eroded-gate", action="store_true",
                    help="reproduce the SUPERSEDED single-criterion gate on stat_mask, which "
                         "excluded SSp-m_left by penalising lateral cortex for being lateral.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--replot", action="store_true",
                    help="REDRAW from epoch_15h_rotation_cache<tag>.npz without recomputing. "
                         "Seconds instead of ~30 min; use for any change to how the result is "
                         "DRAWN. Refuses if the cache is absent -- it will not silently recompute "
                         "and charge you half an hour for a colour change.")
    a = ap.parse_args()
    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    animals = a.animals or sorted({s["label"].split("_")[0] for s in SESSIONS if s["label"] in want})
    t0 = time.time()

    if a.replot:
        cache, rows, sha = load_cache(out_dir, a.tag)
        if cache is None:
            print(f"!! no cache at {out_dir}/epoch_15h_rotation_cache{a.tag}.npz -- run once "
                  f"WITHOUT --replot first. Refusing to recompute silently.")
            return 1
        print(f"REPLOT from cache: {len(cache)} cells computed by git {sha}\n")
        maps, outlines = rebuild(cache, log=lambda m: print(m, flush=True))
        fig = _figure(maps, rows, out_dir, a.tag, outlines=outlines)
        plane = _gain_rotation_figure(rows, out_dir, a.tag)
        print(f"\n[15h] wrote {fig}\n[15h] wrote {plane}\n[15h] replot {time.time() - t0:.0f}s")
        return 0

    MIN_IN_MASK_FRAC = float(a.min_in_mask_frac)
    MIN_IN_STAT_FRAC = float(a.min_in_stat_frac)
    # STAMPED NOW, NOT WHEN THE CACHE IS WRITTEN. A commit made DURING a two-hour run would
    # otherwise be recorded as the code that produced the result; see `git_sha`.
    sha0 = git_sha()
    print(f"ROTATION MAPS -- Haufe patterns in the joint basis, arms {a.arms}  git {sha0}")
    # THE GATE IDENTITY GOES IN THE HEADER, not just its threshold. Two different masks have now
    # been used here, and a run stamped only with a number is indistinguishable from the other one.
    print(f"   GATE: mass in {'stat_mask (SUPERSEDED, eroded)' if a.eroded_gate else 'brain_mask'}"
          f" >= {MIN_IN_MASK_FRAC}"
          + ("" if a.eroded_gate else f" AND mass in stat_mask >= {MIN_IN_STAT_FRAC}")
          + f"   (tag {a.tag!r})\n")

    rows, maps, outlines, cache, draws = [], {}, {}, [], []
    for arm in a.arms:
        align, variant = ARMS[arm]
        print(f"\n== ARM {arm}")
        for an in animals:
            try:
                basis = joint_locanmf.load(an, sessions=SESSIONS)
                data = (collect_rest(an, a.bins) if arm == "rest"
                        else collect_task(an, align, variant))
            except Exception as ex:                                      # noqa: BLE001
                print(f"   !! {an}: {type(ex).__name__} {str(ex)[:80]} -- SKIPPED", flush=True)
                continue
            if not data:
                continue
            print(f"   {an}:", flush=True)
            keep_comp = in_mask_components(basis, eroded=a.eroded_gate)
            print(f"   {int(keep_comp.sum())}/{basis.ncomp} components inside the brain mask "
                  f"(olfactory bulbs + glue edge excluded from the TEST family)", flush=True)
            pats, norms, ceiling, reliability, n_sess = epoch_patterns(
                data, basis, _pipe, seed_ns=f"{arm}|{an}", log=lambda m: print(m, flush=True))
            if "pre" not in pats:
                continue
            boots, boot_draws = {}, {}
            if a.boot:
                for _ep in ("acute", "subacute", "chronic"):
                    if _ep in pats:
                        boots[_ep], boot_draws[_ep] = bootstrap_cosine_ci(
                            data, basis, _pipe, _ep, n_boot=a.boot, seed_ns=f"{arm}|{an}",
                            log=lambda m: print(m, flush=True), return_draws=True)
            nulls, cos_nulls, gain_nulls = null_delta(
                data, basis, _pipe, n_draws=a.null_draws, seed_ns=f"{arm}|{an}",
                log=lambda m: print(m, flush=True))
            for ep in ("acute", "subacute", "chronic"):
                if ep not in pats:
                    continue
                shared = sorted(set(pats["pre"]) & set(pats[ep]))
                if not shared:
                    continue
                zmaps, cosines = [], []
                for pos in shared:
                    a_pre, a_ep = pats["pre"][pos], pats[ep][pos]
                    cos = float(a_pre @ a_ep)
                    cosines.append(cos)
                    # SIGN-ALIGNED before differencing. A Haufe pattern's overall sign is set by
                    # the class coding, and a flipped sign would register as a total rotation.
                    d = np.abs((a_ep if cos >= 0 else -a_ep) - a_pre)
                    z = excess_z(d, nulls.get(pos))
                    sig = significant_components(d, nulls.get(pos), keep=keep_comp)
                    if z is not None:
                        zmaps.append((pos, z, component_outline(sig, basis)))
                        # THE REPLOT CACHE, in COMPONENT space. Stored as `z` and `sig` rather
                        # than the rendered pixel maps: two orders of magnitude smaller, and it is
                        # the actual RESULT rather than a picture of one. `to_pixels` and
                        # `component_outline` rebuild the pixels at replot from the same basis,
                        # so a cached render cannot drift from a live one.
                        cache.append({"arm": arm, "contrast": f"{ep} - pre", "position": int(pos),
                                      "animal": an, "z": np.asarray(z, np.float32),
                                      "sig": (np.zeros(0, bool) if sig is None
                                              else np.asarray(sig, bool))})
                    rel_a = reliability.get("pre", {}).get(pos)
                    rel_b = reliability.get(ep, {}).get(pos)
                    # CORRECTED FOR ATTENUATION IN BOTH EPOCHS. None when either reliability is
                    # unknown or non-positive -- a negative reliability means the split-half
                    # estimate is pure noise and no correction can rescue the cosine.
                    corr = None
                    if rel_a and rel_b and rel_a > 0 and rel_b > 0:
                        corr = round(cos / float(np.sqrt(rel_a * rel_b)), 4)
                    pv = cosine_p(cos, cos_nulls.get(pos))
                    # RESIDUAL log-gain: this position's amplitude change relative to the mean
                    # change across positions, so the epoch-wide scale cancels.
                    gp = None
                    if norms.get("pre", {}).get(pos) and norms.get(ep, {}).get(pos):
                        allp = sorted(set(norms["pre"]) & set(norms[ep]))
                        lgs = [float(np.log(norms[ep][q] / norms["pre"][q])) for q in allp]
                        resid = float(np.log(norms[ep][pos] / norms["pre"][pos])) - float(
                            np.mean(lgs))
                        gp = gain_p(resid, gain_nulls.get(pos))
                    # GAIN, the other half of the answer. The cosine is SHAPE with the global
                    # scale divided out, so a position that merely weakened scores no rotation --
                    # correctly, but that is not the whole story. Amplitude and rotation are
                    # dissociable here: PS93 acute far_center loses 82% of its amplitude at a
                    # cosine of 0.37, while far_R keeps 76% of its amplitude at a cosine of -0.29.
                    gain = (norms[ep][pos] / norms["pre"][pos]
                            if norms.get("pre", {}).get(pos) else None)
                    # THE DRAWS BEHIND THE TWO p VALUES AND THE CI, kept so the COHORT statistic
                    # and the family-wise threshold are a re-read rather than a second run. Both
                    # need per-draw values across cells and neither can be recovered from the
                    # reduced z/sig in the replot cache. `cos_null` is per (arm, animal, position)
                    # and shared across contrasts -- it is a pre-vs-pre null, so it does not
                    # depend on the epoch -- and it is stored per row so the draw ORDER, which is
                    # what makes positions comparable within a draw, survives the round trip.
                    draws.append({
                        "arm": arm, "animal": an, "contrast": f"{ep} - pre", "position": int(pos),
                        "cos_obs": float(cos),
                        "cos_null": np.asarray(cos_nulls.get(pos, []), np.float32),
                        "cos_boot": np.asarray(boot_draws.get(ep, {}).get(pos, []), np.float32),
                    })
                    for reg, v in by_region(d, basis).items():
                        rows.append({"arm": arm, "animal": an, "contrast": f"{ep} - pre",
                                     "position": int(pos), "region": reg,
                                     "abs_delta": round(v, 6),
                                     "cosine_pre_vs_epoch": round(cos, 4),
                                     "rel_pre": (round(rel_a, 4) if rel_a else None),
                                     "rel_epoch": (round(rel_b, 4) if rel_b else None),
                                     "cos_attenuation_corrected": corr,
                                     "cos_p_vs_noise": (round(pv, 4) if pv else None),
                                     "gain_epoch_over_pre": (round(gain, 4) if gain else None),
                                     "gain_resid_p": (round(gp, 4) if gp else None),
                                     "cos_ci_lo": (round(boots[ep][pos][0], 4)
                                                   if ep in boots and pos in boots[ep] else None),
                                     "cos_ci_hi": (round(boots[ep][pos][1], 4)
                                                   if ep in boots and pos in boots[ep] else None),
                                     "n_sessions_epoch": n_sess.get(ep),
                                     "n_sessions_pre": n_sess.get("pre"),
                                     "noise_ceiling_cos": (round(ceiling[pos], 4)
                                                           if pos in ceiling else None),
                                     "norm_pre": round(norms["pre"][pos], 4),
                                     "norm_epoch": round(norms[ep][pos], 4)})
                # PER POSITION, not averaged over them. A mean over the six would have hidden the
                # cue arm's far-contralateral cosine of -0.042 among five values near +0.6.
                for pos, z, outline in zmaps:
                    maps.setdefault((arm, f"{ep} - pre", pos), []).append(to_pixels(z, basis))
                    if outline is not None:
                        outlines.setdefault((arm, f"{ep} - pre", pos), []).append(outline)
                cb = np.mean([ceiling[p] for p in shared if p in ceiling]) if ceiling else np.nan
                print(f"      {ep}-pre: mean cos {np.mean(cosines):+.3f} "
                      f"(noise ceiling {cb:+.3f}) over {len(shared)} pos", flush=True)

        # CHECKPOINT AFTER EVERY ARM. A four-arm run is ~2.5 h and the cache used to be written
        # only at the very end, so a crash in arm 4 discarded arms 1-3 AND the cache that exists
        # to make a redraw free. Each arm now persists everything computed so far, and --replot
        # after an interrupted run draws the arms that finished.
        if rows:
            save_cache(cache, rows, out_dir, a.tag, args=a, sha=sha0)
            save_draws(draws, out_dir, a.tag)
            print(f"[15h] checkpoint after arm {arm}: {write_csv(rows, out_dir, a.tag)}",
                  flush=True)

    if not rows:
        print("no patterns computed")
        return 1
    p = write_csv(rows, out_dir, a.tag)
    print(f"\n[15h] wrote {p}")

    print("\nROTATION (cosine between the pre-stroke and the epoch pattern, 1.0 = no turn)")
    print(f"{'arm':<7}{'contrast':<18}" + "".join(f"{an:>10}" for an in animals))
    for arm in a.arms:
        for ep in ("acute", "subacute", "chronic"):
            sel = [r for r in rows if r["arm"] == arm and r["contrast"] == f"{ep} - pre"]
            if not sel:
                continue
            cells = []
            for an in animals:
                v = {(r["position"]): r["cosine_pre_vs_epoch"]
                     for r in sel if r["animal"] == an}
                cells.append(f"{np.mean(list(v.values())):>10.3f}" if v else f"{'--':>10}")
            print(f"{arm:<7}{ep + ' - pre':<18}" + "".join(cells))

    save_cache(cache, rows, out_dir, a.tag, args=a, sha=sha0)
    save_draws(draws, out_dir, a.tag)
    fig = _figure(maps, rows, out_dir, a.tag, outlines=outlines)
    plane = _gain_rotation_figure(rows, out_dir, a.tag)
    print(f"\n[15h] wrote {fig}\n[15h] wrote {plane}\n[15h] {time.time() - t0:.0f}s")
    return 0


def git_sha() -> str:
    """Short HEAD sha, ``-dirty`` when the tree has uncommitted changes.

    CALL THIS AT STARTUP, NOT AT SAVE TIME. `save_cache` used to read it when it wrote, hours
    after the run began -- so a commit made DURING a two-hour run got stamped onto results the
    commit had no part in, which is the provenance lie the stamp exists to prevent (2026-09-17:
    the cache claimed 568d935, a commit made mid-run; the code that ran was bc6d055).
    """
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip() or "unknown"
        # "-dirty" IS THE WHOLE POINT. With uncommitted changes the HEAD sha names code that is
        # NOT what ran, which is the same lie the missing stamp told on 2026-09-17 -- a figure
        # claiming provenance it does not have is worse than one claiming none.
        if subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                          text=True, timeout=10, check=False).stdout.strip():
            sha += "-dirty"
        return sha
    except Exception:                                                  # noqa: BLE001
        return "unknown"


def write_csv(rows, out_dir, tag=""):
    """The per-cell table. Split out of `main` so a per-arm checkpoint writes the same file."""
    p = out_dir / f"epoch_15h_rotation_regions{tag}.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def save_cache(cache, rows, out_dir, tag="", args=None, sha=None):
    """Write the COMPONENT-space results so the figures can be redrawn without recomputing.

    THIS EXISTS BECAUSE THE FIGURE COST THREE FULL PASSES IN ONE DAY, two of them purely to change
    how something was DRAWN. A ~30-minute recomputation to move a colour is the kind of cost that
    makes a known-wrong figure stay published, which is exactly what happened on 2026-09-17: the
    mask fix landed at 13:51 and the run that wrote the figures had imported the module at 13:39,
    so the published contours were unmasked and re-rendering them meant paying the whole null again.

    COMPONENT SPACE, NOT PIXELS. `z` is one value per component (~90 floats) against 540x640
    per pixel map -- the cache is kilobytes instead of hundreds of megabytes -- and it is the
    result itself, so `--replot` re-projects through the SAME `to_pixels` / `component_outline`
    the live path uses. A cached render could drift from a live one; a cached result cannot.

    THE GIT SHA IS STAMPED IN, because the failure above was invisible precisely for want of it:
    nothing on the figure said which code drew it.
    """
    import json

    if not cache:
        return None
    # `sha` COMES FROM THE CALLER, captured when the run STARTED -- see `git_sha`.
    sha = sha or git_sha()
    meta = [{k: c[k] for k in ("arm", "contrast", "position", "animal")} for c in cache]
    blobs = {}
    for i, c in enumerate(cache):
        blobs[f"z{i}"] = c["z"]
        blobs[f"sig{i}"] = c["sig"]
    p = out_dir / f"epoch_15h_rotation_cache{tag}.npz"
    np.savez_compressed(p, meta=json.dumps(meta), rows=json.dumps(rows, default=str),
                        git_sha=sha, args=json.dumps(vars(args) if args else {}, default=str),
                        n=len(cache), **blobs)
    print(f"[15h] wrote {p}  ({len(cache)} cells, git {sha}) -- redraw with --replot", flush=True)
    return p


def save_draws(draws, out_dir, tag=""):
    """Persist the per-cell null and bootstrap draws to ``epoch_15h_rotation_draws<tag>.npz``.

    WHY A SEPARATE FILE FROM THE REPLOT CACHE. The cache holds the REDUCED result -- `z` and
    `sig` per cell -- which is all a redraw needs and is two orders of magnitude smaller. The
    cohort CI and the family-wise threshold need the DRAWS, and neither is recoverable from a
    reduction. Keeping them apart is what lets `--replot` stay a 5 s operation.

    RAGGED BY CONSTRUCTION, and stored ragged rather than padded: a cell's null and bootstrap
    counts differ (`--null-draws` vs `--boot`), draws that failed to fit are dropped per cell,
    and a cell with no bootstrap gets an empty array. Padding would let a short cell be read as
    a full one with zeros in it, and a zero cosine is a meaningful value here.
    """
    import json

    if not draws:
        return None
    out, meta = {"n": len(draws)}, []
    for i, d in enumerate(draws):
        meta.append({k: (int(d[k]) if k == "position" else d[k])
                     for k in ("arm", "animal", "contrast", "position", "cos_obs")})
        out[f"null{i}"] = np.asarray(d["cos_null"], np.float32)
        out[f"boot{i}"] = np.asarray(d["cos_boot"], np.float32)
    p = out_dir / f"epoch_15h_rotation_draws{tag}.npz"
    np.savez_compressed(p, meta=json.dumps(meta), **out)
    print(f"[15h] wrote {p}  ({len(draws)} cells of draws)", flush=True)
    return p


def load_draws(out_dir, tag=""):
    """``[{arm, animal, contrast, position, cos_obs, cos_null, cos_boot}]``, or ``[]``."""
    import json

    p = Path(out_dir) / f"epoch_15h_rotation_draws{tag}.npz"
    if not p.exists():
        return []
    with np.load(p, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        return [{**m, "cos_null": f[f"null{i}"], "cos_boot": f[f"boot{i}"]}
                for i, m in enumerate(meta)]


def load_cache(out_dir, tag=""):
    """``(cache, rows, sha)`` from `save_cache`, or ``(None, None, None)``."""
    import json

    p = out_dir / f"epoch_15h_rotation_cache{tag}.npz"
    if not p.exists():
        return None, None, None
    with np.load(p, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        rows = json.loads(str(f["rows"]))
        sha = str(f["git_sha"])
        cache = [{**m, "z": f[f"z{i}"], "sig": f[f"sig{i}"]} for i, m in enumerate(meta)]
    return cache, rows, sha


def rebuild(cache, log=print):
    """``(maps, outlines)`` re-projected from the cached component-space results.

    One basis load per animal -- seconds, against the ~30 minutes the null costs.
    """
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    maps, outlines, bases = {}, {}, {}
    for c in cache:
        an = c["animal"]
        if an not in bases:
            bases[an] = joint_locanmf.load(an, sessions=SESSIONS)
            log(f"   basis {an}: {bases[an].ncomp} components")
        basis = bases[an]
        key = (c["arm"], c["contrast"], int(c["position"]))
        maps.setdefault(key, []).append(to_pixels(c["z"], basis))
        sig = c["sig"]
        if sig.size:
            ol = component_outline(sig, basis)
            if ol is not None:
                outlines.setdefault(key, []).append(ol)
    return maps, outlines


def _gain_rotation_figure(rows, out_dir, tag=""):
    """ROTATION against GAIN, one point per (animal, position), panelled by arm and epoch.

    THE TWO ARE DISSOCIABLE AND BOTH MATTER (Priya, 2026-09-17: "but amplitude is also important
    -- how can we assess both?"). The cosine is SHAPE with the global scale divided out, so a
    position that merely weakened scores no rotation; the gain is that missing half. Measured on
    PS93 acute: far_center keeps 18% of its pre-stroke amplitude at a cosine of 0.37, while far_R
    keeps 76% at a cosine of -0.29. One collapsed, the other turned -- and a single number cannot
    tell those apart.

    THE PLANE MAKES THE FOUR OUTCOMES READABLE:

        gain ~ 1, cosine ~ ceiling   nothing happened
        gain LOW, cosine ~ ceiling   the code WEAKENED but kept its shape
        gain ~ 1, cosine LOW         the code ROTATED at preserved strength
        gain LOW, cosine LOW         both
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    # POSITION BY MARKER, animal by colour. The whole result is position-specific -- far-contra
    # turns most acutely, near-middle chronically -- and with colour already spent on animals a
    # reader could not tell which point was which (Priya, 2026-09-17). Marker-for-one-factor,
    # colour-for-another is the convention `spout_behavior` already uses for its six positions.
    MARKERS = ["o", "s", "^", "v", "D", "P"]
    _pretty = dict(zip(CONF_LABELS, [x.title() for x in ef.anatomical_labels(CONF_LABELS,
                                                                            short=False)]))
    _rank = {lab: i for i, lab in enumerate(CONF_LABELS)}
    pos_rank = {c: _rank.get(POSITION_NAMES.get(c, ""), 99) for c in range(6)}
    pos_name = {c: _pretty.get(POSITION_NAMES.get(c, ""), str(c)) for c in range(6)}

    cons = ["acute - pre", "subacute - pre", "chronic - pre"]
    arms = [x for x in ARMS if any(r["arm"] == x for r in rows)]
    colors = config.animal_color()
    seen = {}
    for r in rows:                       # collapse the region axis
        g, c = r.get("gain_epoch_over_pre"), r.get("cosine_pre_vs_epoch")
        if g in (None, "") or c in (None, ""):
            continue
        pv = r.get("cos_p_vs_noise")
        seen[(r["arm"], r["contrast"], r["animal"], int(r["position"]))] = (
            float(g), float(c), (float(pv) if pv not in (None, "") else None))

    ceils = {}
    for arm in arms:
        v = [float(r["noise_ceiling_cos"]) for r in rows
             if r["arm"] == arm and r.get("noise_ceiling_cos") not in (None, "")]
        ceils[arm] = float(np.mean(v)) if v else None

    # THE HEADER IS A FIXED HEIGHT IN INCHES, not a fraction of the figure. As a fraction it
    # collapses when few arms are drawn: at one arm `top=0.90` left about half an inch for a
    # five-line caption, which then sat on the panel titles. Reserving inches makes the band the
    # same physical size whether one arm is plotted or four.
    HEAD_IN, FOOT_IN = 2.0, 1.45
    body_in = 3.2 * len(arms)
    fig_h = body_in + HEAD_IN + FOOT_IN
    gains = [v[0] for v in seen.values() if v[0] and np.isfinite(v[0]) and v[0] > 0]
    xlo = min(gains) / 1.6 if gains else 0.08
    xhi = max(gains) * 1.6 if gains else 3.0

    fig, axes = plt.subplots(len(arms), len(cons),
                             figsize=(3.5 * len(cons) + 1.6, fig_h),
                             squeeze=False)
    # HEADER AND FOOTER BANDS RESERVED. `bbox_inches="tight"` crops to the artists and does
    # not separate them, so a three-line caption at 0.945 lands on panel titles sitting at
    # 0.86, and a legend at the figure bottom lands on the x-labels.
    fig.subplots_adjust(top=1 - HEAD_IN / fig_h, bottom=FOOT_IN / fig_h,
                        hspace=0.34, wspace=0.24)
    for i, arm in enumerate(arms):
        for j, con in enumerate(cons):
            ax = axes[i][j]
            pts = {k: v for k, v in seen.items() if k[0] == arm and k[1] == con}
            for (_a, _c, an, _pos), (g, c, pv) in pts.items():
                mk = MARKERS[pos_rank.get(_pos, 0) % len(MARKERS)]
                # ANIMAL COLOURS from the config, because that is what they already mean.
                # FILLED = the rotation beats the split-half null at p < 0.05; OPEN = it does not,
                # so the point is a measurement without a claim attached. Drawing them alike would
                # let an untested point read as a result.
                sig = pv is not None and pv < 0.05
                ax.scatter(g, c, s=44, marker=mk, label=an,
                           color=colors.get(an, "0.4") if sig else "none",
                           edgecolor=colors.get(an, "0.4"),
                           linewidth=0.8 if sig else 1.4, alpha=0.9)
            # THE QUADRANT DIVIDERS. Vertical at gain = 1 (weakened | strengthened). Horizontal
            # at THIS ARM's mean split-half reliability, not at zero or some round number:
            # "rotated" has to mean "turned further than estimation noise alone would turn it",
            # and that floor differs by arm (rest is far noisier than lick).
            ceil_arm = ceils.get(arm)
            ax.axvline(1.0, color="k", lw=0.7, ls=":")
            if ceil_arm is not None:
                ax.axhline(ceil_arm, color="k", lw=0.9, ls="--")
                ax.axhspan(-1.0, ceil_arm, color="0.85", alpha=0.35, lw=0, zorder=0)
                if j == 0:
                    ax.text(0.09, ceil_arm + 0.03, "noise ceiling", fontsize=8.5,
                            style="italic", va="bottom")
            else:
                ax.axhline(0, color="k", lw=0.7)
            # CORNER LABELS, drawn once per panel so a reader never has to reconstruct the axes.
            lo = ceil_arm if ceil_arm is not None else 0.0
            for (xx, yy, ha, va, txt) in (
                    (0.30, 1.30, "center", "top", "WEAKENED|shape kept"),
                    (1.75, 1.30, "center", "top", "STRONGER|shape kept"),
                    (0.30, -1.30, "center", "bottom", "WEAKENED|+ ROTATED"),
                    (1.75, -1.30, "center", "bottom", "STRONGER|+ ROTATED")):
                # the pipe is a line break; a literal newline inside these tuples does not
                # survive the editing round-trip, so it is substituted at draw time
                ax.text(xx, yy, txt.replace("|", chr(10)), fontsize=8.5, ha=ha,
                        va=va, color="0.35", fontweight="bold", zorder=1)
            del lo
            ax.set_xscale("log")
            # LIMITS FROM THE DATA. Hard-coded at (0.08, 3.0) these clipped real points off the
            # edge -- a figure that silently drops observations is worse than one that looks
            # untidy. A decade-and-a-bit of padding either side keeps the 1.0 reference centred.
            ax.set_xlim(xlo, xhi)
            # ROOM FOR THE QUADRANT LABELS OUTSIDE THE DATA. A cosine is bounded to
            # [-1, 1], so the margin beyond +-1 can never hold a point -- putting the
            # labels there makes "does the label cover a datum" impossible rather than
            # unlikely, which corner placement only achieved by luck.
            ax.set_ylim(-1.34, 1.34)
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.grid(alpha=0.22)
            if i == 0:
                ax.set_title(con, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{arm}\ncos(pre, epoch)", fontsize=11, fontweight="bold")
            if i == len(arms) - 1:
                ax.set_xlabel("gain  |epoch| / |pre|   (log)", fontsize=11)
            ax.tick_params(labelsize=10)
    # HANDLES FROM EVERY PANEL, not just the first. Built from `axes[0][0]` the legend listed
    # PS92-94 and omitted PS95, whose points are plainly in the lower panels -- that first panel is
    # ENL/acute, and PS95 contributes ONE acute session so it is absent there. A legend that omits
    # a plotted animal is worse than no legend.
    uniq = {}
    for row in axes:
        for ax_ in row:
            for hh, ll in zip(*ax_.get_legend_handles_labels()):
                uniq.setdefault(ll, hh)
    uniq = {k: uniq[k] for k in sorted(uniq)}
    # TWO LEGENDS, because two factors are encoded: COLOUR is the animal, SHAPE is the spout
    # position. One combined legend would need 4 x 6 entries to say what two of 4 and 6 do.
    from matplotlib.lines import Line2D
    pos_keys = sorted(pos_rank, key=lambda c: pos_rank[c])[: len(MARKERS)]
    shape_h = [Line2D([], [], color="0.35", marker=MARKERS[i], linestyle="none",
                      markersize=7, label=pos_name[c])
               for i, c in enumerate(pos_keys)]
    # ALL-OPEN IN THE ANIMAL KEY. Handles harvested from the axes inherit whichever fill state the
    # first plotted point happened to have, so an animal read as "significant" purely because its
    # first cell was. Fill carries MEANING here (p < 0.05) and must not be spent on identity.
    animal_h = [Line2D([], [], marker="o", linestyle="none", markersize=8,
                       markerfacecolor="none", markeredgecolor=colors.get(a, "0.4"),
                       markeredgewidth=1.4, label=a)
                for a in sorted(uniq)]
    leg1 = fig.legend(handles=animal_h, fontsize=11, frameon=False,
                      ncol=max(len(animal_h), 1), loc="lower center",
                      bbox_to_anchor=(0.28, 0.10 / fig_h), title="animal (fill = p < 0.05)")
    leg1.get_title().set_fontsize(10)
    leg2 = fig.legend(handles=shape_h, fontsize=10, frameon=False, ncol=3,
                      loc="lower center", bbox_to_anchor=(0.72, 0.04 / fig_h),
                      title="spout position")
    leg2.get_title().set_fontsize(10)
    fig.add_artist(leg1)
    fig.text(0.5, 1 - 0.22 / fig_h, "ROTATION vs GAIN -- the two halves of 'the code changed'",
             ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 1 - 0.62 / fig_h,
             # WRAPPED SHORT, ~105 characters a line. These ran wider than the panels they
             # describe, so the figure was being sized by its caption rather than by its data.
             "One point per animal x spout position. X = the epoch's Haufe-pattern amplitude "
             "relative to pre-stroke\n"
             "(1.0 = unchanged, dotted). Y = cosine between the two patterns (1.0 = no rotation). "
             "FILL = p < 0.05 vs the split-half null.\n"
             "Gain ABOVE 1.0 is STRONGER than pre-stroke, not merely preserved -- the chronic "
             "panels sit there.\n"
             "GAIN IS THE DECODER PATTERN'S NORM, NOT AN EVOKED RESPONSE, and NOT the deficit "
             "measure: a position can become more\n"
             "SEPARABLE while its response collapses (acute far-contra is dominated by "
             "unattempted trials -- the epoch_14 MEANref inversion).\n"
             "For the deficit read the REST-referenced maps and the encoder. Colours are ANIMALS, "
             "from configs/animals.yaml.",
             ha="center", va="top", fontsize=10)
    out = out_dir / f"epoch_15h_gain_vs_rotation{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def _figure(maps, rows, out_dir, tag="", outlines=None):
    """ONE FIGURE PER ARM: rows = spout position, columns = epoch contrast, cells = the z-map.

    PER POSITION, because averaging over positions is what hid the cue arm's far-contralateral
    cosine of -0.042 among five values near +0.6 -- and far-contralateral is the lesion-relevant
    one, so the average was hiding exactly the cell worth seeing.

    THE CELLS ARE EXCESS-OVER-NOISE, not raw change. The raw |change| map correlates with its own
    pre-stroke null at r = +0.83, so it localises the BASIS rather than the lesion; each component
    is scored against its own split-half null instead. See `null_delta`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import beta_maps as _bm
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local import transfer_matrix as tm
    from wfield_local.grant_figures import CONF_LABELS

    # THE ANALYSED REGION, loaded once and drawn on every panel. `None` only if the grid does not
    # match, which is itself worth seeing rather than silently skipping -- the shape guards in
    # `to_pixels` / `component_outline` / `in_mask_components` all fall back to NO MASKING when the
    # grids disagree, so a mismatch would remove the mask from the statistics without an error.
    try:
        _SM = np.asarray(_bm.stat_mask(), bool)
    except Exception:                                                  # noqa: BLE001
        _SM = None
    # THE ALLEN BOUNDARIES, loaded once for the whole figure. `atlas_edges` takes no arguments and
    # returns the parcellation on the SHARED grid these maps live on. A missing atlas costs the
    # outlines, never the figure -- the same contract `beta_maps` states for it.
    from wfield_local.atlas_overlay import overlay_regions as _overlay_regions
    try:
        _EDGES = _bm.atlas_edges()
    except Exception:                                                  # noqa: BLE001
        _EDGES = None
    if _EDGES is None:
        print("   !! atlas edges unavailable -- panels drawn WITHOUT Allen boundaries", flush=True)
    if _SM is None:
        print("   !! stat_mask unavailable -- panels drawn WITHOUT the analysed-region edge",
              flush=True)
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    colors = config.animal_color()

    # THE CANONICAL ORDER AND NAMES, not a numeric sort. Every other position figure in this
    # project reads Near Ipsi -> Far Contra (`CONF_LABELS` order, `anatomical_labels` naming), and
    # `POSITION_NAMES` codes do NOT follow it -- 0 is close_CENTER, so sorting by code puts Middle
    # before Ipsi and silently transposes this figure against its neighbours.
    _pretty = dict(zip(CONF_LABELS, [x.title() for x in ef.anatomical_labels(CONF_LABELS,
                                                                            short=False)]))
    _rank = {lab: i for i, lab in enumerate(CONF_LABELS)}

    def _key(pos):
        return _rank.get(POSITION_NAMES.get(pos, ""), 99)

    def _name(pos):
        return _pretty.get(POSITION_NAMES.get(pos, ""), str(pos))

    cons = ["acute - pre", "subacute - pre", "chronic - pre"]
    made = []
    for arm in sorted({k[0] for k in maps}, key=lambda x: list(ARMS).index(x)):
        poss = sorted({k[2] for k in maps if k[0] == arm}, key=_key)
        if not poss:
            continue
        fig = plt.figure(figsize=(3.1 * len(cons) + 6.0, 2.5 * len(poss) + 2.6))
        # A SPACER COLUMN between the colorbar and the bar panel. Without it the bar panel's
        # position tick-labels land on top of the colorbar's ticks -- one `wspace` serves every
        # gap, and it has to stay small to keep the map columns adjacent.
        # WIDER GUTTERS EITHER SIDE OF THE COLORBAR. Its label sat hard against the bar panel's
        # y-axis label; one `wspace` serves every gap, so the room has to come from spacer columns.
        gs = fig.add_gridspec(len(poss), len(cons) + 4,
                              width_ratios=[1] * len(cons) + [0.16, 0.07, 0.95, 2.0],
                              top=0.855, bottom=0.05, hspace=0.12, wspace=0.10)
        vals = [np.nanmean(np.asarray(v), axis=0) for k, v in maps.items() if k[0] == arm]
        flat = np.concatenate([v.ravel() for v in vals])
        flat = flat[np.isfinite(flat)]
        vmax = float(np.nanpercentile(np.abs(flat), 99)) if flat.size else 1.0
        im = None
        for i, pos in enumerate(poss):
            for j, con in enumerate(cons):
                ax = fig.add_subplot(gs[i, j])
                v = maps.get((arm, con, pos))
                ax.set_xticks([])
                ax.set_yticks([])
                if not v:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, color="0.6")
                    continue
                cm = plt.get_cmap(tm.CMAP_CHANGE).copy()
                # OUTSIDE-THE-MASK MUST NOT LOOK LIKE ZERO-CHANGE. `set_bad("white")` on a
                # diverging map whose MIDPOINT is also white made the two indistinguishable, so
                # the apparent "brain silhouette" was the non-zero region rather than the analysed
                # region -- and every contour sitting over near-zero cortex read as a contour
                # outside the brain (Priya, 2026-09-17). Grey is outside the analysis; white is
                # inside it and unchanged.
                cm.set_bad("0.90")
                im = ax.imshow(np.nanmean(np.asarray(v), axis=0), cmap=cm,
                               vmin=-vmax, vmax=vmax)
                # THE ALLEN PARCELLATION, as every other map family in this deck draws it. This
                # arm never had it (no commit on any branch ever referenced `atlas_edges` here),
                # which is why a significant blob could only be located by eye against a
                # neighbouring figure. Priya, 2026-09-18. Drawn UNDER the significance contours
                # and the mask edge so it can never be mistaken for either: these are ANATOMY,
                # not statistics, and the two must stay visually separable.
                if _EDGES is not None:
                    _overlay_regions(ax, _EDGES)
                # AND DRAW THE ANALYSED REGION'S EDGE, so "is that contour inside the mask" is a
                # question the figure answers instead of one a reader has to trust.
                if _SM is not None:
                    ax.contour(_SM.astype(float), levels=[0.5], colors="0.55",
                               linewidths=0.6, linestyles="--")
                # CONTOUR THE COMPONENTS THAT CLEAR THE FAMILY-WISE NULL, not a threshold on the
                # projected z -- the pixel map mixes components, so a contour on it would outline
                # the basis rather than the result. Drawn where a MAJORITY of animals agree, so a
                # single animal cannot put a ring on the cohort figure.
                ol = (outlines or {}).get((arm, con, pos))
                if ol:
                    frac = np.mean(np.asarray(ol, float), axis=0)
                    # BELT AND BRACES: re-intersect at DRAW time. The per-animal outlines are
                    # already masked, but this is the last point before ink, and a leak here is
                    # the one failure a reader cannot audit from the figure.
                    if _SM is not None and frac.shape == _SM.shape:
                        leaked = int(((frac > 0.5) & ~_SM).sum())
                        if leaked:
                            print(f"   !! {arm} {con} pos {pos}: {leaked} outline px OUTSIDE "
                                  f"stat_mask -- clipped at draw time", flush=True)
                        frac = np.where(_SM, frac, 0.0)
                    if np.any(frac > 0.5):
                        # SAME TREATMENT AS THE POSITION MAPS: a white underlay carries the edge
                        # across the saturated ends of the diverging ramp, black carries it across
                        # the pale middle. 0.9 pt of plain black read well over white and was
                        # close to invisible inside a deep red or deep blue blob -- which is
                        # exactly where a significant component tends to sit.
                        ax.contour(frac, levels=[0.5], colors="white", linewidths=2.6, alpha=0.85)
                        ax.contour(frac, levels=[0.5], colors="black", linewidths=1.5)
                if i == 0:
                    # MAIN LABELS AT THE MAP FAMILIES' SIZE. Epoch across the top, spout
                    # position down the side; these two are read before anything else in the
                    # panel, and they sat smaller than the annotations around them.
                    ax.set_title(con, fontsize=12.5, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(_name(pos), fontsize=12.5, fontweight="bold")
        if im is not None:
            # SHORTER THAN THE COLUMN: a full-height bar on a six-row figure is a metre of
            # gradient for a scale that needs an inch. Centred on the middle rows.
            lo = max(0, len(poss) // 2 - 1)
            cax = fig.add_subplot(gs[lo:lo + 2, len(cons) + 1])
            fig.colorbar(im, cax=cax).set_label(
                "change in readout pattern, z vs its own pre-stroke split-half null", fontsize=10)

        # right: the cosine per position, against the noise ceiling
        ax = fig.add_subplot(gs[:, len(cons) + 3])
        w = 0.82 / len(cons)
        # PER-ANIMAL SPREAD AND PER-CELL SIGNIFICANCE ON THE BARS. The bar is a cohort mean over
        # three or four animals and, drawn bare, carries no uncertainty at all -- which is how a
        # mean of three becomes "the result". Each animal's own value is overlaid as a dot, FILLED
        # when that animal's cosine beats its split-half null at p < 0.05 and OPEN when it does
        # not, and the range across animals is drawn as a whisker.
        #
        # NO CI BAR ON PURPOSE. With n = 3-4 animals a parametric interval would imply a precision
        # the design does not have; the individual points ARE the honest display, and they are the
        # unit this project's nested bootstrap resamples.
        for j, con in enumerate(cons):
            ys, spread, per_an, ci = [], [], [], []
            for pos in poss:
                sel = [r for r in rows
                       if r["arm"] == arm and r["contrast"] == con and r["position"] == pos]
                vals = [(r["animal"], float(r["cosine_pre_vs_epoch"]),
                         (float(r["cos_p_vs_noise"])
                          if r.get("cos_p_vs_noise") not in (None, "") else None))
                        for r in sel]
                # one row per REGION per cell, so collapse to one value per animal
                uniq = {a: (c, pv) for a, c, pv in vals}
                ys.append(float(np.mean([c for c, _ in uniq.values()])) if uniq else np.nan)
                spread.append((min((c for c, _ in uniq.values()), default=np.nan),
                               max((c for c, _ in uniq.values()), default=np.nan)))
                per_an.append(uniq)
                # cohort CI = the mean of the per-animal bootstrap bounds, kept only when EVERY
                # contributing animal has one; a partial mean would mix intervals with points.
                los = [float(r["cos_ci_lo"]) for r in sel if r.get("cos_ci_lo") not in (None, "")]
                his = [float(r["cos_ci_hi"]) for r in sel if r.get("cos_ci_hi") not in (None, "")]
                ci.append((float(np.mean(los)), float(np.mean(his)))
                          if los and his and len(los) == len(his) else None)
            yy = np.arange(len(poss)) - (j - (len(cons) - 1) / 2) * w
            ax.barh(yy, ys, height=w, label=con.replace(" - pre", ""),
                    color=tm.epoch_color(con.replace(" - pre", "")), zorder=2)
            for k, (loh, uniq) in enumerate(zip(spread, per_an)):
                # THE WHISKER IS A BOOTSTRAP CI WHEN ONE EXISTS, and the ANIMAL RANGE otherwise --
                # never silently one dressed as the other. The range is the min-max over three or
                # four animals; on a bar chart that reads as an error bar and is not one.
                if ci[k] is not None:
                    ax.plot(list(ci[k]), [yy[k], yy[k]], color="0.15", lw=1.6, zorder=3,
                            solid_capstyle="butt")
                elif np.isfinite(loh[0]):
                    ax.plot([loh[0], loh[1]], [yy[k], yy[k]], color="0.55", lw=0.9, zorder=3,
                            ls=":")
                for an_, (c, pv) in sorted(uniq.items()):
                    sig = pv is not None and pv < 0.05
                    ax.scatter(c, yy[k], s=16, zorder=4,
                               color=colors.get(an_, "0.4") if sig else "none",
                               edgecolor=colors.get(an_, "0.4"), linewidth=0.8)
        ceil = [r["noise_ceiling_cos"] for r in rows
                if r["arm"] == arm and r["noise_ceiling_cos"] not in (None, "")]
        if ceil:
            ax.axvline(float(np.mean([float(c) for c in ceil])), color="k", ls="--", lw=1.4,
                       label="noise ceiling")
        ax.set_yticks(range(len(poss)), [_name(p) for p in poss], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("cosine(pre pattern, epoch pattern)", fontsize=11)
        ax.set_title("HOW FAR THE READOUT TURNED\n(read against the dashed ceiling, not 1.0)",
                     fontsize=11, fontweight="bold")
        ax.axvline(0, color="k", lw=0.8)
        # BELOW THE AXES so it cannot sit on the bars or collide with the title.
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.06))
        ax.grid(axis="x", alpha=0.25)

        fig.text(0.5, 0.985, f"WHERE the position readout changes -- {arm} window",
                 ha="center", va="top", fontsize=15, fontweight="bold")
        fig.text(0.5, 0.945,
                 "Haufe patterns (A = Cov(X)b) from one block-matched decoder per epoch, in the "
                 "shared joint LocaNMF basis; unit-normalised per position, so this is SHAPE, not "
                 "gain.\n"
                 "BLACK OUTLINES = components clearing a family-wise max-statistic null at "
                 "p < 0.05, in a majority of animals. The test is per COMPONENT, not per pixel: "
                 "the map mixes components, so a\n"
                 "threshold on it would outline the basis.\n"
                 "MAPS ARE EXCESS OVER NOISE: each component is scored against its own "
                 "pre-stroke split-half null, because the RAW change map correlates with that null "
                 "at r = 0.83 and localises the basis, not the lesion.\n"
                 "A decoder WEIGHT is a filter, not a pattern (r = 0.245 here). Cohort mean; "
                 "three animals carry chronic data.\n"
                 "THE MAP AND THE COSINE ARE NOT THE SAME QUANTITY: the map is scale-FREE (each "
                 "component against its own noise), the cosine is scale-WEIGHTED. A position can "
                 "look dramatic and barely turn.\n"
                 "Measured on PS93 cue acute -- far_center has "
                 "mean |z| 11.1 with only 20% of its change in the top-weight components and a "
                 "cosine of 0.37, while far_R has mean |z| 3.7, 37% in the top weights, and a "
                 "cosine of -0.29. Judge rotation by the BARS, not by brightness.",
                 ha="center", va="top", fontsize=10)
        out = out_dir / f"epoch_15h_rotation_maps_{arm}{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(out.name)
    return ", ".join(made)


if __name__ == "__main__":
    raise SystemExit(main())
