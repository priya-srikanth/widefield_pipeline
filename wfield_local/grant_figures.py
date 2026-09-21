"""GRANT FIGURES — a small, self-contained set for a progress report and a new application.

    python -m wfield_local.grant_figures [--output <dir>] [--only 1 1b 2 2b 3a 3b 4 5 5b 5c 6 6b]

Priya, 2026-08-24. Deliberately NOT deck figures: the deck exists to be interrogated and carries every
caveat on the slide, which is right there and wrong here. These are meant to be read in ten seconds by
someone who has not been in the weeds, so each one makes ONE point, and the caveats live in this
docstring and in DECISIONS.md rather than on the axes.

WHAT IS SHOWN

  1  BEHAVIOUR at all six spout positions, per animal, against days-from-lesion. Engaged hit rate --
     the "stopped" (terminal quit) trials are excluded, which is what `hit_rate` in the per-position
     metrics CSV already means -- with Wilson CIs. This is the deficit the rest of the work is about.

  2  PRE-STROKE CROSS-SESSION DECODING in the shared joint-LocaNMF basis, ENL / cue / lick. Each bar
     is leave-one-session-out accuracy pooled over the curated pre-stroke sessions, the error bar is
     the 95% CI across HELD-OUT SESSIONS (not across trials -- the session is the unit that
     generalisation is claimed over), and every held-out session is plotted as a point. Chance is
     1/6.

  3a POST-STROKE CODING RETAINED, per animal, per window, over days. y = the projection of that
     position's own trials onto its own pre-stroke coding direction, pole-normalised so 1.0 = the
     pre-stroke lick signature and 0 = the other positions. Error bars are SEM over trials. This is
     the "how much of the normal code is left" view.

  3b FROZEN vs WITHIN-SESSION DECODING, per animal, per window, over days. The frozen pre-stroke
     decoder asks whether the OLD code still reads out; a decoder trained on the post-stroke session
     itself asks whether position information is present AT ALL. Frozen falling while within-session
     holds up is reorganisation rather than loss -- the two lines and the gap between them are the
     point. Error bars are binomial 95% CIs on each session's own trial count.

EXCLUSIONS, and they are not cosmetic
  * PS92_0817 and PS93_0817 are dropped everywhere. They follow the 8/16 laser that did NOT take, so
    they are neither baseline nor post-stroke, and `config.session_phase` already labels them
    "excluded" -- this module asks the config rather than hardcoding dates.
  * PS94/PS95 were lesioned 2026-08-16 and PS92/PS93 2026-08-17, so DAY-FROM-LESION is per animal
    (`config.stroke_date`). Plotting against calendar date would misalign the cohort by a day.

WHAT THESE FIGURES DO NOT SAY, kept here so it is not lost when they are pasted into a document:
  * "Miss" is defined by SPOUT CONTACT, so an off-target lick counts as a miss in panel 1 and sits in
    the miss class in panel 3. The DAQ cannot distinguish "did not try" from "tried and missed".
  * Panel 3a's y can exceed 1.0. The projection rises either because a trial points more along the
    direction or because it sits further from the session centroid; above-1 values mean "at least
    intact", not "better than pre-stroke".
  * Panel 2 is PRE-STROKE only and is a capability claim (we can decode), not a lesion result.
"""
from __future__ import annotations

import argparse
import os
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from wfield_local.console import use_utf8_stdout

# THE FIGURE FAMILIES, one module each (2026-09-21). What is left in THIS module is the
# REGISTRY and the DRIVER: JOBS, the unit decomposition, the process pool, the CLI. It
# knows which figures exist and how to render them in parallel; it no longer knows how
# any single one of them is drawn.
# THE FIGURE FAMILIES, one module each (2026-09-21). What is left in THIS module is the
# REGISTRY and the DRIVER: JOBS, the unit decomposition, the process pool, the CLI.
#
# EVERY NAME IS RE-EXPORTED, not just the JOBS entry points. 31 of the 49 that moved are
# addressed from outside the grant_* modules -- `epoch_grant_figures` and `epoch_figures`
# import a dozen of them, and a dozen test modules more. Moving one is a rename of a
# public-ish name whether or not it starts with an underscore; `tests/test_grant_kit.py`
# pins the list against the pre-split revision so it cannot quietly shrink.
from wfield_local.grant_behaviour import (  # noqa: F401
    BASELINE_BEFORE,
    BASELINE_X,
    _wilson,
    fig_behaviour,
    fig_behaviour_collapsed,
    fig_prestroke_decoding,
    fig_prestroke_decoding_cohort,
)
from wfield_local.grant_confusion import (  # noqa: F401
    MIN_REFIT_CLASS,
    MIN_REFIT_SHARE,
    REFIT_UNAVAILABLE,
    _acc_ci,
    _collect_5c,
    _counts,
    _draw_5c,
    _matched_frozen,
    _refit_pred,
    _Stored,
    fig_confusion_delta,
    fig_confusion_per_session,
    fig_confusion_pre_post,
    fig_confusion_pre_post_working,
    fig_confusion_prestroke,
)
from wfield_local.grant_encoder import (  # noqa: F401
    FROZEN_WINDOWS,
    _binom_ci,
    _enc_ci,
    _enc_scores,
    _enc_tables,
    _enc_terms,
    fig_coding_retained,
    fig_encoder_gain_shape,
    fig_frozen_vs_within,
)
from wfield_local.grant_geometry import (  # noqa: F401
    _asymmetry_ci,
    _asymmetry_one,
    _crossnobis_cross,
    _crossnobis_within,
    _fast_rdm,
    _halves,
    _lw_cov,
    _matrices_crossnobis,
    _mats_crossnobis,
    _rdm_ci,
    _rdm_one,
    _rdm_pct,
    _rdm_rows,
    _rdm_scores,
    _triu_vals,
    _whitener,
    fig_asymmetry,
    fig_crossnobis_cross,
    fig_crossnobis_delta,
    fig_crossnobis_geometry,
    fig_delta_trajectory,
    fig_geometry_by_position,
)

# THE SHARED MACHINERY lives in `grant_kit` (see its docstring): twenty helpers that call
# nothing outside themselves and that forty-five functions here call into. Imported by name
# so an unused one is visible; re-exported because tests and sibling scripts read some of
# them off this module.
# EVERY NAME THAT MOVED IS RE-EXPORTED, not only the ones still called from this module.
# `_runs_to_blocks`, `_delta_diag_ci` and `_delta_diag_one` are now used only by their kit
# siblings, so a list built from 'what does grant_figures still reference' dropped them --
# and three test modules that read them off `grant_figures` broke. `tests/test_grant_kit.py`
# now asserts this list stays complete against the pre-split revision.
from wfield_local.grant_kit import (  # noqa: F401
    _BUNDLE_CACHE,
    ANIMALS,
    BOOT_CACHE_VERSION,
    CONF_LABELS,
    N_BOOT_DELTA,
    N_BOOT_RDM,
    N_LOO_DRAW,
    POS,
    POS_SHORT,
    WINDOWS,
    _anchor,
    _best_match,
    _block_boot,
    _block_index,
    _boot_cached,
    _cd_labels,
    _class_note,
    _class_select,
    _collect_7,
    _colw,
    _corr_matrix,
    _day,
    _delta_cis,
    _delta_diag_ci,
    _delta_diag_one,
    _delta_grid,
    _diag,
    _digest,
    _excludes_zero,
    _feed,
    _fig_root,
    _fit_bottom,
    _fit_header,
    _footer,
    _impaired,
    _matrices_pattern,
    _mats_pattern,
    _means,
    _nanmean_stack,
    _out,
    _overlaps,
    _pct3,
    _pooled_bundle,
    _position_metrics,
    _pre_reference,
    _runs_to_blocks,
    _save,
    _seed,
    _session_trials,
    _sessions,
    _sg_labels,
    _short,
    _suptitle,
    _twinned,
    _txt,
    _variants,
    _windows,
    coverage_note,
    only,
    pos_style,
    set_only,
)
from wfield_local.grant_matching import (  # noqa: F401
    _match_tables,
    _pre_loo_matrices,
    fig_best_match,
    fig_best_match_by_session,
)
from wfield_local.grant_similarity import (  # noqa: F401
    MIN_REL,
    N_BOOT,
    SPLIT_REPS,
    _disatt_one,
    _disattenuated_ci,
    _matrices_splithalf,
    _mats_splithalf,
    _mean_pattern,
    _pattern_stats,
    _reliability,
    _split_half,
    _split_half_matrix,
    _strat_mean,
    fig_pattern_delta,
    fig_pattern_similarity,
    fig_pattern_similarity_per_session,
    fig_reliability_verdict,
    fig_splithalf_delta,
    fig_splithalf_matrix,
)
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

# ------------------------------------------------------------------ 3a. coding retained
# ---------------------------------------------------------------------------------------------
# 7 / 7b: IS THE "LOST CODE" JUST A NOISIER ONE?  (Priya, 2026-08-25)
# ---------------------------------------------------------------------------------------------










@lru_cache(maxsize=12)
def pre_session_counts(align, variant):
    """``{animal: n}`` -- pre-stroke sessions that have ANY trial of this class.

    NOT the same as "pre-stroke sessions", and the difference is the whole reason this exists. For
    `lick` and `working` every pre-stroke session qualifies, so it reproduces the epoch assignment.
    For `stopped` it does not come close: a well-trained pre-stroke animal barely quits, and
    post-cue the counts are PS92 1, PS93 1, PS94 6, PS95 7 out of eleven sessions each. A subtitle
    stating 44 over a panel built from 15 is a 3x overstatement of the baseline's footing, and it
    is invisible on the figure itself.
    """
    out = {}
    for an in ANIMALS:
        try:
            bd = _pooled_bundle(an, align)
        except Exception as ex:                                          # noqa: BLE001
            # SAID OUT LOUD. A silent skip here understates the pre baseline by one whole animal
            # and the figure still draws, which is the same class of failure this function exists
            # to fix.
            print(f"  !! pre-counts {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            continue
        n = 0
        for i in sorted(bd["pre_i"]):
            if any(len(_session_trials(bd, i, q, variant)) for q in CONF_LABELS):
                n += 1
        if n:
            out[an] = n
    return out














def _split_half_asymmetry(src, rng, labels=None):
    """Mean |corr(A_P, B_Q) - corr(A_Q, B_P)| over off-diagonal pairs -- a pure noise read.

    The two orderings estimate the same thing, so any difference is sampling noise in the split.
    Useful as a diagnostic on how much to trust a panel; deliberately NOT drawn into the matrix.
    """
    labels = labels or CONF_LABELS
    halves = {}
    for q in labels:
        Z = src.get(q)
        if Z is None or len(Z) < 4:
            continue
        idx = rng.permutation(len(Z))
        h = len(Z) // 2
        halves[q] = (Z[idx[:h]].mean(0), Z[idx[h:2 * h]].mean(0))
    d = []
    ks = [q for q in labels if q in halves]
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            p, q = ks[a], ks[b]
            r1 = float(np.corrcoef(halves[p][0], halves[q][1])[0, 1])
            r2 = float(np.corrcoef(halves[q][0], halves[p][1])[0, 1])
            if np.isfinite(r1) and np.isfinite(r2):
                d.append(abs(r1 - r2))
    return float(np.mean(d)) if d else np.nan








def _pre_pool_blk(pre_b, exclude=None):
    """Pooled pre-stroke BLOCK ids per position, mirroring `_pre_reference` for the trial arrays."""
    acc = {}
    for s, pat in pre_b.items():
        if s == exclude:
            continue
        for q, b in pat.items():
            acc.setdefault(q, []).append(np.asarray(b))
    return {q: np.concatenate(v) for q, v in acc.items()}








# ---------------------------------------------------------------------------------------------
# 8 / 8b: THE CROSSNOBIS VERSION -- and what it can and cannot say per position
# ---------------------------------------------------------------------------------------------
#
# Priya, 2026-08-25: "would this still give us any per-position information though? or just
# overall representational geometry similarity?"  BOTH, but they are different questions and the
# two figures below separate them deliberately.
#
#   8   CROSS-SET distances d(post at P, pre at Q). The direct translation of figure 6: per
#       position, fully, with the diagonal reading "did this position's pattern move". Noise-
#       UNBIASED, which figure 6's correlation is not. Still sensitive to a global amplitude
#       change, exactly as figure 6 is -- an unbiased distance between two patterns is not a
#       gain-invariant one.
#
#   8b  SECOND-ORDER: the within-set 6x6 RDM for each session correlated against the pre-stroke
#       RDM. This is RSA proper, and it IS gain-invariant, because scaling every distance leaves
#       a correlation between RDMs unchanged. Per-position information survives as each
#       position's ROW -- its five distances to the other positions -- so "is far_R still
#       arranged the way it was relative to everything else" is answerable, while "did far_R's
#       pattern move" is not. That is the trade: 8 keeps the position and loses gain-invariance,
#       8b keeps gain-invariance and can only speak about a position's RELATIONS.
#
# Together they bracket the headline pattern result. If the graded drop at every position in
# figure 6 were a global amplitude change, 8 would show it and 8b would NOT.






















# ---------------------------------------------------------------------------------------------
# DELTA VIEWS: every post-stroke panel as a DIFFERENCE from the pre-stroke reference
# ---------------------------------------------------------------------------------------------
#
# Priya, 2026-08-25: "make additional versions of fig 5, 6, 7 as differences from prestroke --
# show pre-stroke for reference, but then subsequent columns expressed as deltas."
#
# WHY THIS IS WORTH A SEPARATE FIGURE RATHER THAN A READER'S SUBTRACTION. The absolute panels ask
# the eye to hold a six-by-six reference in memory and compare it with a panel three columns away,
# and the pre-stroke reference is NOT uniform -- close positions are intrinsically more confusable
# than far ones, and every animal's baseline has its own texture. A cell that reads "0.4, low" may
# be 0.4 against a baseline of 0.45 (nothing happened) or against 0.9 (half the code gone). The
# delta answers that directly and the absolute panel cannot.
#
# WHAT IS SUBTRACTED, in every case, is the SAME pre-stroke reference the absolute figure draws in
# its first column -- leave-one-session-out, so it is one session against other days exactly like
# the post columns, and NOT a random half of the pooled trials (see `_collect_7`). Getting this
# wrong would put a floor under every delta.
#
# NOT r-SQUARED. Priya's phrasing was "deltas of r2 / accuracy"; for the correlation figures the
# quantity differenced is r ITSELF, because the sign carries the result -- a far_R pattern moving
# ONTO far_L shows as a positive off-diagonal, and squaring would erase exactly that. Accuracy
# figures difference the row-normalised probability, which is already on [0, 1].


# ---------------------------------------------------------------------------------------------
# BLOCK BOOTSTRAP for the delta figures (Priya, 2026-08-25)
# ---------------------------------------------------------------------------------------------
#
# WHAT IS RESAMPLED, AND WHAT IS NOT.
#
#   BLOCKS, not trials. Trials adjacent in time share arousal, satiety and drift, so an i.i.d.
#   trial bootstrap treats correlated samples as independent and returns intervals that are too
#   narrow. The scheduler's own ~6-trial position blocks are the natural unit and the pipeline
#   already uses them for GroupKFold. A block belongs to ONE position by construction (a new block
#   starts when the position changes), so resampling a session's blocks also resamples each
#   position's trial count -- which is right: how many trials a position got is itself uncertain.
#
#   SESSIONS ARE HELD FIXED. Days are not exchangeable while an animal is recovering: PS94's
#   figure-8 diagonal runs 0.99 -> 0.46 across one week, so there is no single post-stroke value for
#   an interval to be about. Resampling days would fold that trajectory into "sampling noise" and
#   quote an interval for a state that does not exist. The interval therefore means "how well
#   determined GIVEN THESE DAYS" and licenses no generalisation to other days -- the trajectory in
#   the per-session panels is what speaks to that. Same decision, same reasoning, as the pattern
#   bootstrap of 2026-08-25 (DECISIONS.md).
#
# WHY BOOTSTRAP AND NOT A PERMUTATION, for the asymmetry question specifically: permuting position
# labels equalises the condition means, so the true distances collapse toward zero -- but the
# sampling variance of a crossnobis distance scales with the true difference vector, so a real and
# perfectly SYMMETRIC separation still yields a larger |D - D.T| than permuted data does. The
# permuted null therefore sits too low and would call ordinary noise "asymmetry". A bootstrap
# interval on D[P,Q] - D[Q,P] has no such problem.











#: Minimum STOPPED trials at one position in one session before that cell contributes a mean
#: pattern. Lower than the engaged families' 10 by design: the quit period is short by definition
#: and a 10-trial floor discards most of it. Measured 2026-09-11, post-cue, per (session, position):
#: the post-stroke stopped sets run 15-369 trials per SESSION spread over six positions, so ~5 is
#: where most sessions still contribute all six cells.
MIN_STOPPED = 5

#: Total pre-stroke STOPPED trials an animal needs before it may define the no-lesion control, on
#: top of needing at least four of the six positions. Post-cue the four animals hold 6, 40, 326 and
#: 495, so this admits PS94 and PS95 and refuses the two whose whole baseline is one session.
MIN_STOPPED_REF = 100


@lru_cache(maxsize=6)
def _collect_stopped(align, min_trials=MIN_STOPPED):
    """Mean patterns on the trials the engagement gate THROWS AWAY: the terminal quit period.

    Priya, 2026-09-11: "do we already have (or can we add) a post-stroke vs pre-stroke 'stopped'
    trials pattern similarity analysis? We will probably have to pool pre-stroke trials to get
    enough n." We did not -- `flag_engagement` has only ever been a filter -- and yes, pre-stroke
    has to be pooled.

    Returns ``({animal: {"WORK_REF": means, "PRE_STOPPED": means|None, day: means}}, days)``.

    WHAT IS BEING ASKED, and why it needs TWO references rather than one. A post-stroke stopped
    pattern that no longer resembles the pre-stroke template has two explanations that this design
    can separate only because the pre-stroke animal ALSO stops:

        post-stopped vs pre-stroke WORKING template   how far the code is from the intact one
        PRE-stopped  vs the same template             how far STOPPING ALONE moves it, no lesion

    The second is the control, and without it any post-stroke result is a statement about arousal
    and satiety as much as about the lesion. It is why `WORK_REF` is the engaged pre-stroke
    reference every other family uses rather than a stopped one: both stopped sets are scored
    against the SAME template, so their difference is attributable.

    THE PRE-STROKE STOPPED SET IS POOLED ACROSS SESSIONS AND IS THIN, UNEQUALLY. Post-cue, per
    animal: PS92 6 trials in 1 of 11 sessions, PS93 40 in 1, PS94 326 in 6, PS95 495 in 7. PS92 and
    PS93 cannot support a control at all -- one trial per position is not a mean pattern -- so the
    control exists for PS94 and PS95 and the figure must say which animals it rests on rather than
    drawing four columns and letting two of them be noise. A well-trained pre-stroke animal barely
    quits, which is the same fact that makes the gate worth having.

    POST-STROKE IS PER SESSION, like every other collector here, so an epoch is a mean over sessions
    rather than a mean over trials -- otherwise a 369-trial session would outvote a 15-trial one by
    a factor of 25 within its own epoch.
    """
    out, all_days = {}, set()
    for an in ANIMALS:
        try:
            bd = _pooled_bundle(an, align)
        except Exception as ex:                                          # noqa: BLE001
            print(f"  !! stopped {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            continue
        rec, pre_pool = {}, {}
        work = {}
        for i, lab in enumerate(bd["kept"]):
            mmdd = lab.split("_")[-1]
            if i in bd["pre_i"]:
                for q in CONF_LABELS:
                    Z = _session_trials(bd, i, q, "lick")
                    if len(Z):
                        work.setdefault(q, []).append(Z)
                    S = _session_trials(bd, i, q, "stopped")
                    if len(S):
                        pre_pool.setdefault(q, []).append((i, S))
                continue
            pat = {q: Z for q in CONF_LABELS
                   if len(Z := _session_trials(bd, i, q, "stopped")) >= min_trials}
            # TWO POSITIONS IS THE FLOOR for a correlation matrix to mean anything; one position
            # gives a 1x1 comparison dressed as a 6x6.
            if len(pat) >= 2:
                day = _day(an, mmdd)
                rec[day] = _means(pat)
                all_days.add(day)
        if not work:
            continue
        d = {"WORK_REF": _means({q: np.vstack(v) for q, v in work.items()})}
        pre_s = {q: np.vstack([z for _i, z in v]) for q, v in pre_pool.items()
                 if sum(len(z) for _i, z in v) >= min_trials}
        # THE RAW TOTAL, before the per-position floor, because that is what "this animal barely
        # quits" means and it is what the caption reports. Counting after the floor printed
        # "PS92 0 trials" for an animal that has six -- true of the surviving cells and false of
        # the animal.
        n_pre_tr = sum(len(z) for v in pre_pool.values() for _i, z in v)
        n_pre_ss = len({i for v in pre_pool.values() for i, _z in v})
        # A CONTROL NEEDS ALL SIX POSITIONS AND A REAL TRIAL COUNT, not two positions and forty
        # trials. The first version gated on ">= 2 positions with >= 5 trials", which let PS93's
        # forty pre-stroke stopped trials -- spread over six positions, from ONE session -- stand as
        # a no-lesion baseline, while the caption asserted PS93 had been excluded. A control that
        # thin is indistinguishable from noise and the figure would have shown a delta against it.
        ok = len(pre_s) >= 4 and n_pre_tr >= MIN_STOPPED_REF
        d["PRE_STOPPED"] = _means(pre_s) if ok else None
        d["PRE_STOPPED_N"] = (n_pre_tr, n_pre_ss)
        d.update(rec)
        out[an] = d
    return out, sorted(all_days)


@lru_cache(maxsize=6)
def _collect_stopped_pooled(align, min_trials=20):
    """POSITION-AGNOSTIC stopped-trial patterns: one mean vector per session, all positions pooled.

    Priya, 2026-09-11: "I more was thinking about mean pattern similarity or encoder similarity for
    ALL stopped trials (rather than per position)."

    WHY POOLING IS THE RIGHT MOVE HERE, in numbers. The per-position stopped arm divides each
    session's quit period six ways, and the quit period is short by definition. Pooled across
    animals the post-cue stopped sets are 867 trials pre-stroke, 1,984 acute, 1,935 subacute and
    359 chronic; split per position the chronic cell falls to 36-75 trials. Pooling uses all of a
    session's stopped trials for one measurement instead of a sixth of them for each of six, and
    the question -- does the cortical pattern during the quit period still look like the pre-stroke
    one -- does not need the position axis at all.

    Returns ``({animal: {"REF": v, "PRE_BY_SESS": {mmdd: v}, day: v}}, days)`` where each value is
    a 380-vector: the mean over that session's stopped trials, no position split.

    TWO REFERENCES, BECAUSE THEY TRADE OFF AND NEITHER IS STRICTLY BETTER:

      REF           the pre-stroke ENGAGED pooled mean. Available for all four animals, but it
                    compares a QUITTING animal to a WORKING one, so the pre-stroke stopped column
                    is needed as the "quitting alone" control before anything can be attributed
                    to the lesion.
      PRE_BY_SESS   the pre-stroke STOPPED pattern of each pre-stroke session. Priya, 2026-09-11:
                    "I want to compare post-stroke stopped to pre-stroke stopped" -- state-matched
                    on both sides, which is the cleaner contrast. But only PS94 (326 trials) and
                    PS95 (495) can build one; PS92 has 6 and PS93 has 40, because a well-trained
                    pre-stroke animal barely quits. No amount of pooling fixes that.

    KEPT PER SESSION, NOT POOLED, for the reason `_collect_7` keeps pre-stroke sessions apart: a
    pre-stroke session scored against a pool that CONTAINS IT is scored partly against itself, and
    its bar is then too high by construction. The caller builds a leave-one-session-out reference
    for the pre column and the full pool for post-stroke days.

    A HIGHER FLOOR THAN THE PER-POSITION ARM (20 trials, not 5): a session contributes ONE number
    here, so there is no reason to accept a cell built from five trials, and the whole point of
    pooling is that it does not have to.
    """
    out, all_days = {}, set()
    for an in ANIMALS:
        try:
            bd = _pooled_bundle(an, align)
        except Exception as ex:                                          # noqa: BLE001
            print(f"  !! stopped-pooled {an} {align}: {type(ex).__name__} {str(ex)[:80]}",
                  flush=True)
            continue
        rec, ref_parts, pre_by_sess, n_pre = {}, [], {}, 0
        for i, lab in enumerate(bd["kept"]):
            mmdd = lab.split("_")[-1]
            stop = [_session_trials(bd, i, q, "stopped") for q in CONF_LABELS]
            stop = [Z for Z in stop if len(Z)]
            if i in bd["pre_i"]:
                eng = [_session_trials(bd, i, q, "lick") for q in CONF_LABELS]
                eng = [Z for Z in eng if len(Z)]
                if eng:
                    ref_parts.append(np.vstack(eng))
                if stop:
                    Z = np.vstack(stop)
                    n_pre += len(Z)
                    # PER SESSION and gated on the SESSION's own count, so a leave-one-out
                    # reference is a real pattern rather than a handful of trials.
                    if len(Z) >= min_trials:
                        pre_by_sess[mmdd] = Z.mean(0)
                continue
            if not stop:
                continue
            Z = np.vstack(stop)
            if len(Z) < min_trials:
                continue
            day = _day(an, mmdd)
            if day is None:
                continue
            rec[day] = Z.mean(0)
            all_days.add(day)
        if not ref_parts or not rec:
            continue
        d = {"REF": np.vstack(ref_parts).mean(0), "PRE_BY_SESS": pre_by_sess,
             "PRE_STOPPED_N": int(n_pre)}
        d.update(rec)
        out[an] = d
    return out, sorted(all_days)












@lru_cache(maxsize=6)
def _matrices_crossnobis_rownorm(align, variant, min_trials=10):
    """`_matrices_crossnobis_rowcentred` with every row also divided by its OWN SD across columns.

    WHY ROW-CENTRING ALONE IS NOT SCALE-FREE. Writing the row out,

        d(P,Q) - mean_Q d(P,.) = -2 mu_postP . (mu_preQ - mean mu_pre)
                                  + (|mu_preQ|^2 - mean |mu_pre|^2)

    the first term scales LINEARLY with |mu_postP| and the second does not depend on P at all. So
    row-centring removes amplitude from the row's OFFSET and leaves it MULTIPLYING the row's shape:
    double P's response and the whole row-centred profile doubles. Priya, 2026-09-10: "any universal
    gain should change the shape anyway, right?" -- right in the sense that matters for DIRECTION,
    since scaling a profile preserves its rank order and its signs, which is what the "moved toward"
    claim rests on. Wrong for MAGNITUDE: a row-centred value cannot be compared across epochs or
    positions whose amplitude differs, and the encoder says amplitude fell hard acutely (fitted
    a: 0.94 pre to 0.36 acute, post-cue).

    Dividing each row by its own SD fixes that: every row becomes a z-profile, so a cell reads "how
    many row-SDs from this row's mean", and epochs become magnitude-comparable.

    WHAT IT DELIBERATELY THROWS AWAY: how FAR a position moved. A row that barely moved and a row
    that moved enormously have the same z-profile if they moved in the same direction. This is the
    companion to the row-centred family, not its replacement -- read direction here and distance
    there.
    """
    mats, days = _matrices_crossnobis_rowcentred(align, variant, min_trials)
    out = {}
    for an, by_key in mats.items():
        d = {}
        for key, M in by_key.items():
            A = np.asarray(M, float).copy()
            for i in range(A.shape[0]):
                row = A[i]
                if not np.isfinite(row).any():
                    continue
                sd = np.nanstd(row)
                # A ZERO-SD ROW IS NOT A FLAT PROFILE WORTH DIVIDING: it is one surviving cell, or a
                # degenerate row. NaN it rather than emitting an inf that a colour map will render
                # as the strongest effect on the figure.
                A[i] = row / sd if np.isfinite(sd) and sd > 1e-12 else np.nan
            if np.isfinite(A).any():
                d[key] = A
        if d:
            out[an] = d
    return out, days


@lru_cache(maxsize=6)
def _matrices_best_match_destination(align, variant, min_trials=10):
    """{animal: {"PRE"|day: M}} where M[i, j] = 1 if position i's BEST pre-stroke match was j.

    THE ARGMAX OF `_matrices_pattern`, ONE-HOT. Figure 10b reduces that argmax to a single number --
    "was it still itself" -- which answers whether the code moved and says nothing about WHERE it
    went. Averaged over sessions within an epoch, this matrix is the destination distribution: the
    diagonal is 10b, and the off-diagonal mass shows whether a position's code went to ONE other
    position or scattered evenly, which is the difference between substitution and collapse.

    WHY THIS IS THE MOST LEGIBLE FORM OF "MOVED TOWARD" (Priya, 2026-09-10, asking for exactly this):
    it needs no sign convention, no row-centring and no units. A cell is a fraction of sessions.
    Everything else in the 6/7/8 families requires the reader to hold "is larger better here" in
    their head; a destination matrix does not.

    A ROW WITH NO FINITE VALUES SCORES NOTHING rather than voting for column 0. `np.nanargmax` on an
    all-NaN row raises, and catching that to return 0 would have manufactured a systematic pull
    toward the first position out of missing data -- the same failure `_fig_10b` guards.

    THE PRE COLUMN IS ONE-HOTTED PER SESSION AND THEN AVERAGED, and it is NOT built from
    `_matrices_pattern`'s "PRE" entry for that reason. That entry is the MEAN of the
    leave-one-session-out correlation matrices, and `argmax(mean) != mean(argmax)`: one-hotting the
    averaged matrix asks "does the average pre-stroke session match itself", which is 6/6 for every
    animal and makes the baseline a perfect identity. Every post-stroke epoch is a mean of
    PER-SESSION one-hots, so scoring pre the other way compares a session against an average and
    charges the difference to the lesion. Built from `_collect_7` directly so both sides are the
    same construction.
    """
    store, days = _collect_7(align, variant, min_trials)

    def _onehot(M):
        A = np.asarray(M, float)
        H = np.full(A.shape, np.nan)
        for i in range(A.shape[0]):
            if not np.isfinite(A[i]).any():
                continue
            H[i] = 0.0
            H[i, int(np.nanargmax(A[i]))] = 1.0
        return H if np.isfinite(H).any() else None

    out = {}
    for an, (pre_by_sess, by_day) in store.items():
        ref_m = _means(_pre_reference(pre_by_sess))
        loo = [_onehot(_corr_matrix(_means(pat), _means(_pre_reference(pre_by_sess, exclude=s))))
               for s, pat in pre_by_sess.items()]
        d = {}
        base = _nanmean_stack([h for h in loo if h is not None])
        if base is not None:
            d["PRE"] = base
        for day, pat in by_day.items():
            h = _onehot(_corr_matrix(_means(pat), ref_m))
            if h is not None:
                d[day] = h
        if d:
            out[an] = d
    return out, days






















def _matrices_crossnobis_rowcentred(align, variant, min_trials=10):
    """`_matrices_crossnobis` with every ROW centred: WHICH position did it move toward.

    A NAMED COLLECTOR RATHER THAN A KWARG because `MATRIX_FAMILIES` resolves collectors by name
    through `getattr`, and a family is the unit that gets a matrix, a diagonal and a delta panel.
    One line of delegation keeps the arithmetic in exactly one place.

    WHAT IT BUYS, and why the label-level measures do not already give it (Priya, 2026-09-09). The
    decoder confusion and the best-match fraction say which position the READOUT assigns; they are
    label-level and can move for reasons that are not representational. The raw distance row is
    confounded the other way: `d(post P, pre Q) = |mu_postP|^2 - 2 mu_postP . mu_preQ + |mu_preQ|^2`
    has a first term depending only on P, so a pure amplitude change in P shifts its distance to
    EVERY pre-stroke position equally and paints a uniform row that looks like "moved toward all
    six". Row-centring removes that term and leaves the CONTRAST within the row, which is where a
    substitution actually lives -- so a far-contra row going negative under the far-middle column is
    that position's pattern having moved toward far-middle, not merely away from itself.
    """
    return _matrices_crossnobis(align, variant, min_trials=min_trials, row_centre=True)














































#: Random half-splits averaged per session. MEASURED rather than guessed (Priya, 2026-09-10: "how
#: did we decide on 8 splits?" -- it had been a guess, and the honest answer was that nobody had
#: checked). Re-running PS92_0606 post-cue under 20 different seeds:
#:
#:     repeats    1      2      4      8     16     32     64
#:     seed SD  0.147  0.133  0.076  0.045  0.038  0.025  0.017
#:
#: falling as 1/sqrt(r), as it must. Eight was ADEQUATE -- a per-session SD of 0.045 pools to about
#: 0.007-0.011 at the epoch level, ~5% of the acute effect -- but the splits themselves cost almost
#: nothing next to `_collect_7`, which is memoised and dominates the runtime. Thirty-two cuts the
#: seed-dependence 1.8x for no measurable time. Effect on the pooled post-cue gaps of moving 8 -> 32:
#: -0.000 pre, -0.005 acute, -0.016 subacute, +0.003 chronic, so no conclusion turns on it.
ENC_CEILING_REPEATS = 32


def _enc_ceiling(pat, rng, repeats=ENC_CEILING_REPEATS):
    """THE CEILING the frozen encoder is failing against: this session predicting ITSELF, held out.

    WHY THE FROZEN ENCODER NEEDS ONE. `raw` is an R^2 and acutely it is -0.388, which says "worse
    than predicting the mean" and nothing about whether anything could have done better in that
    session. The existing companion, `gain`, frees the AMPLITUDE only -- and `_enc_terms`' own
    docstring warns that a large `gain - raw` is an amplitude story only when `gain` itself is high,
    because a code that is simply GONE also recovers a lot under rescaling. That ambiguity is what
    forced the withdrawal of the "half amplitude, half shape" reading on 2026-09-10. A ceiling
    resolves it: `ceiling - raw` is how much of the failure is the TEMPLATE being wrong, and
    `1 - ceiling` is how much is the session having no position information to predict at all.

    A REFIT ENCODER MUST BE CROSS-VALIDATED OR IT IS 1.0 BY CONSTRUCTION. Ridge on a one-hot
    position design reduces to the per-position mean (see `_enc_terms`), so "refit within the
    session", predicting that session's own means from themselves, is an identity. Splitting the
    trials is what makes it a prediction -- which also makes this the split-half family scored in the
    ENCODER's units rather than as a correlation. Deliberately: a correlation is gain-blind and `raw`
    is not, so the two cannot otherwise be read against each other.

    BOTH ORDERINGS, AVERAGED, over `repeats` random splits. A against B and B against A are two
    estimates of one quantity differing only in which random half landed on which side;
    `_split_half_matrix` makes the same argument for the same reason.

    IT IS A PESSIMISTIC CEILING AND MUST BE READ AS ONE. Both sides are half-session means, while
    `raw` scores a noisy session against a reference pooled over ~10 sessions. The ceiling therefore
    carries noise on both sides where the frozen arm has it on one, and can sit BELOW `raw`
    pre-stroke. That is the same training-set-size bracketing the matched frozen DECODER arm exposed,
    where matching flipped the pre-stroke gap from -0.073 to +0.090. Read `ceiling - raw` against its
    own pre-stroke value, never against zero.
    """
    return _enc_half_scores(pat, rng, repeats=repeats)["ceiling"]


def _enc_half_scores(pat, rng, repeats=ENC_CEILING_REPEATS, ref_pool=None):
    """``{"ceiling": terms, "matched": terms|None}`` -- both scored on the SAME half-session means.

    THE CEILING AND THE MATCHED FROZEN ARM DIFFER IN ONE THING AND MUST DIFFER IN ONE THING ONLY.
    Both score half A of this session. The ceiling scores it against half B of the SAME session; the
    matched arm scores it against an equally sized random draw from the PRE-STROKE pool. Same scored
    side, same number of reference trials, same estimator -- so the difference is which sessions the
    reference came from, and nothing else.

    WHY IT IS NEEDED. The unmatched frozen arm scores the full session against a reference pooled
    over ~10 pre-stroke sessions, which carries far less noise than a half-session reference. On the
    post-cue window that asymmetry is tolerable; on the PRE-CUE window it inverts the comparison
    outright -- frozen EV 0.330 against a ceiling of 0.090, a frozen arm beating its own ceiling,
    which is impossible for a real ceiling and is a fact about the construction rather than the
    data. Matching removes the asymmetry, at the cost of a noisier reference for both.

    COMPUTED TOGETHER, not in two passes, so the two share `halves` exactly. Two passes with the same
    seed would coincide only for as long as nobody changed the call order, and the pairing is the
    whole point.
    """
    labels = [q for q in CONF_LABELS if q in pat and len(pat[q]) >= 4]
    if len(labels) < 2:
        nan = (np.nan, np.nan, np.nan, {})
        return {"ceiling": nan, "matched": None if ref_pool is None else nan}
    got, got_m = [], []
    for _ in range(repeats):
        halves = {}
        for q in labels:
            Z = np.asarray(pat[q])
            idx = rng.permutation(len(Z))
            h = len(Z) // 2
            halves[q] = (Z[idx[:h]].mean(0), Z[idx[h:2 * h]].mean(0))
        A = {q: v[0] for q, v in halves.items()}
        B = {q: v[1] for q, v in halves.items()}
        for m, ref in ((A, B), (B, A)):
            t = _enc_terms(m, ref)
            if np.isfinite(t[0]):
                got.append(t)
        if ref_pool is not None:
            # SAME n PER POSITION as that position's half, drawn without replacement from the pool.
            # Per position rather than one global n: the halves differ in size between positions
            # whenever the animal worked them unevenly, which post-stroke it always does.
            mref = {}
            for q in labels:
                R = np.asarray(ref_pool.get(q, []))
                h = len(np.asarray(pat[q])) // 2
                if len(R) < 2 or h < 1:
                    continue
                take = min(h, len(R))
                mref[q] = R[rng.permutation(len(R))[:take]].mean(0)
            if len(mref) >= 2:
                for m in (A, B):
                    t = _enc_terms({q: m[q] for q in mref}, mref)
                    if np.isfinite(t[0]):
                        got_m.append(t)

    def _pool(ts):
        if not ts:
            return (np.nan, np.nan, np.nan, {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per = {q: float(np.nanmean([t[3][q] for t in ts if q in t[3]]))
                   for q in labels if any(q in t[3] for t in ts)}
            return (float(np.nanmean([t[0] for t in ts])),
                    float(np.nanmean([t[1] for t in ts])),
                    float(np.nanmean([t[2] for t in ts])), per)

    return {"ceiling": _pool(got), "matched": (_pool(got_m) if ref_pool is not None else None)}


@lru_cache(maxsize=6)
def _enc_ceiling_tables(align, variant, min_trials=10):
    """{animal: {"PRE"|day: (ceiling, a, gain, per-position)}} -- the within-session encoder ceiling.

    PRE IS PER-SESSION AND THEN AVERAGED, not a ceiling computed on the pooled reference. The ceiling
    is a statement about ONE session's own repeatability, so pooling ten of them would measure
    something else and would sit far above anything a single post-stroke session could reach.
    """
    ceil_t, _matched, days = _enc_half_tables(align, variant, min_trials)
    return ceil_t, days


@lru_cache(maxsize=6)
def _enc_matched_tables(align, variant, min_trials=10):
    """The SIZE-MATCHED frozen arm: same half-session scored side, reference drawn to the same size.

    Read against `_enc_ceiling_tables` rather than against `_enc_tables`: those two share the scored
    side and the reference size, so `ceiling - matched` is the cost of the reference coming from
    OTHER sessions, with training-set size held constant. At pre-stroke that difference is the
    cross-session generalisation cost and nothing else, which is the baseline every post-stroke epoch
    has to be read against.
    """
    _ceil, matched, days = _enc_half_tables(align, variant, min_trials)
    return matched, days


@lru_cache(maxsize=6)
def _enc_half_tables(align, variant, min_trials=10):
    """``(ceiling_tables, matched_tables, days)`` -- one pass, so the two arms share their halves.

    PRE IS PER-SESSION AND THEN AVERAGED, and its reference is LEAVE-ONE-SESSION-OUT: a pre-stroke
    session must not draw its matched reference from a pool that contains itself, or the matched arm
    is scored partly against its own trials and the baseline it defines is too easy.
    """
    store, days = _collect_7(align, variant, min_trials)
    out, out_m = {}, {}
    for an, (pre_by_sess, by_day) in store.items():
        rng = np.random.default_rng(_seed(an, align, variant))
        rec, rec_m = {}, {}

        def _avg(ts):
            ts = [t for t in ts if t is not None and np.isfinite(t[0])]
            if not ts:
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per = {q: float(np.nanmean([t[3][q] for t in ts if q in t[3]]))
                       for q in CONF_LABELS if any(q in t[3] for t in ts)}
                return (float(np.nanmean([t[0] for t in ts])),
                        float(np.nanmean([t[1] for t in ts])),
                        float(np.nanmean([t[2] for t in ts])), per)

        pre, pre_m = [], []
        for sess, pat in pre_by_sess.items():
            d = _enc_half_scores(pat, rng, ref_pool=_pre_reference(pre_by_sess, exclude=sess))
            pre.append(d["ceiling"])
            pre_m.append(d["matched"])
        if (t := _avg(pre)) is not None:
            rec["PRE"] = t
        if (t := _avg(pre_m)) is not None:
            rec_m["PRE"] = t

        full = _pre_reference(pre_by_sess)
        for day, pat in by_day.items():
            d = _enc_half_scores(pat, rng, ref_pool=full)
            if np.isfinite(d["ceiling"][0]):
                rec[day] = d["ceiling"]
            if d["matched"] is not None and np.isfinite(d["matched"][0]):
                rec_m[day] = d["matched"]
        if len(rec) > 1:
            out[an] = rec
        if len(rec_m) > 1:
            out_m[an] = rec_m
    return out, out_m, days
















#: (key, function) for every grant figure, in render order. AT MODULE SCOPE because a worker
#: process re-imports this module and looks a job up BY KEY -- a table built inside `main()` is
#: not reachable from a spawned child.
JOBS = (("1", fig_behaviour), ("1b", fig_behaviour_collapsed),
        ("2", fig_prestroke_decoding), ("2b", fig_prestroke_decoding_cohort),
        ("3a", fig_coding_retained), ("3b", fig_frozen_vs_within),
        ("4", fig_confusion_prestroke), ("5", fig_confusion_pre_post),
        ("5b", fig_confusion_pre_post_working),
        ("5c", fig_confusion_per_session), ("5d", fig_confusion_delta),
        ("6", fig_pattern_similarity),
        ("6b", fig_pattern_similarity_per_session), ("6d", fig_pattern_delta),
        ("7", fig_splithalf_matrix), ("7b", fig_reliability_verdict),
        ("7d", fig_splithalf_delta),
        ("8", fig_crossnobis_cross), ("8b", fig_crossnobis_geometry),
        ("8d", fig_crossnobis_delta), ("8e", fig_asymmetry), ("8g", fig_geometry_by_position),
        ("9", fig_delta_trajectory),
        ("10", fig_best_match), ("10b", fig_best_match_by_session),
        ("11", fig_encoder_gain_shape))

ALL_KEYS = tuple(k for k, _ in JOBS)

#: Measured shares of a full serial render (2026-08-28, 5.79 h wall, from the output mtimes). Used
#: ONLY to start the long units first, which is what decides the makespan of a fixed pool: 7b and
#: 8d are each ~30 minutes per unit, and a pool that picks them up last finishes half an hour after
#: it had nothing else to do. Wrong numbers here cost scheduling, never correctness.
_COST_HINT = {"7b": 31.2, "8d": 26.6, "6d": 11.9, "7d": 10.5, "8b": 9.5, "8e": 5.0, "5c": 4.0,
              "6": 0.5, "8": 0.5}


def _splits(fn) -> tuple[bool, bool]:
    """``(splits by alignment, splits by trial class)`` for one figure function.

    Answered from the SOURCE -- does it iterate `_windows()`, does it iterate `_variants(` --
    rather than from a hand-kept list, because a hand-kept list is wrong the first time somebody
    adds a figure.

    THE TWO ARE NOT THE SAME QUESTION, and assuming they were produced the first bug this parallel
    driver had. Figure 4 loops over alignments but not over trial classes, and its filename carries
    no variant: asking for `4[precue/lick]` and `4[precue/working]` as separate units had two
    workers rendering the identical figure and writing the identical path at the same time. Not a
    slow render -- a torn PNG, and one that would look merely "missing" in the deck.
    """
    import inspect
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False, False
    return "_windows()" in src, "_variants(" in src


def render_units(want=None):
    """[(key, align, variant)] -- the independent units of a render, longest first.

    A windowed figure contributes one unit per (alignment, trial class); everything else one unit
    with `None` for both. The units are independent by construction: each writes its own PNG and
    shares nothing but the on-disk session cache.
    """
    want = set(want or ALL_KEYS)
    units = []
    for key, fn in JOBS:
        if key not in want:
            continue
        windowed, varianted = _splits(fn)
        if not windowed:
            units.append((key, None, None))
            continue
        for _d, align, _w in WINDOWS:
            if not varianted:
                units.append((key, align, None))
                continue
            # THE SAME RULE AS THE COLLECTORS, from the same function rather than a second copy
            # of the conditional -- which is how this line came to be missing the `_ONLY_VARIANT`
            # filter that `_variants` applies, so `--only-variant` narrowed the figures but not the
            # unit list the scheduler planned.
            units.extend((key, align, v) for v in _variants(align))
    units.sort(key=lambda u: -_COST_HINT.get(u[0], 0.0))
    return units


def _render_unit(spec):
    """Render ONE unit in this process. Top-level and picklable, so a spawned worker can run it."""
    key, align, variant, out = spec
    # THROUGH THE SETTER, not `global`. These live in `grant_kit` since 2026-09-21 and the readers
    # are there; assigning a same-named global here would bind a second variable that nothing
    # reads, and every worker would silently render EVERY alignment instead of its own unit.
    set_only(align, variant)
    fn = dict(JOBS)[key]
    tag = key if align is None else f"{key}[{align}/{variant}]"
    try:
        p = fn(Path(out))
        return (tag, str(p) if p else None, None)
    except Exception as ex:                                            # noqa: BLE001
        return (tag, None, f"{type(ex).__name__} {str(ex)[:160]}")


# Thread pinning and worker counts live in `wfield_local.parallel` now: `poststroke_section_g`,
# `joint_xsession` and `locanmf_frozen_decoder` fan out too, and four copies of "how many workers,
# how many BLAS threads, what happens when one dies" is how they drift. Re-exported under the old
# names so this module's callers and tests are unaffected.
from wfield_local.parallel import BLAS_THREADS  # noqa: E402
from wfield_local.parallel import default_jobs as _default_jobs  # noqa: E402
from wfield_local.parallel import pin_blas as _pin_blas


def _run_parallel(units, out, n_jobs, threads_per_worker):
    """Fan the units over a PROCESS pool. Returns (n_written, [failures]).

    PROCESSES, NOT THREADS, and not negotiable: pyplot keeps a global figure registry and the GIL
    serialises the numpy that dominates this render anyway. The backend is Agg, set at import.

    Each worker is capped to `threads_per_worker` BLAS threads. Left uncapped, every worker's numpy
    grabs the whole box: the serial render already averages 1.5 cores from BLAS alone, so ten
    unconstrained workers oversubscribe 24 cores rather than scale on them.
    """
    import concurrent.futures as cf
    import os as _os

    env = dict(_os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[var] = str(threads_per_worker)
    written, failures = 0, []
    try:
        ctx = __import__("multiprocessing").get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
            futs = {pool.submit(_render_unit, (k, a, v, str(out))): (k, a, v)
                    for k, a, v in units}
            claimed = {}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                tag, path, err = fut.result()
                if err:
                    failures.append((tag, err))
                    print(f"  !! [{i}/{len(units)}] {tag}: {err}", flush=True)
                else:
                    written += 1
                    # TWO UNITS, ONE FILE is the failure mode of a wrong unit decomposition, and
                    # it is silent: the loser's bytes are simply gone and the PNG may be torn.
                    # `_splits` is meant to prevent it; this catches the case where a new figure
                    # breaks the assumption anyway, which a static check cannot see.
                    for p in (path if isinstance(path, (list, tuple)) else [path]):
                        if p is None:
                            continue
                        if p in claimed:
                            failures.append((tag, f"wrote {p}, already written by {claimed[p]} "
                                                  f"-- two units share one output path"))
                            print(f"  !! COLLISION {tag} and {claimed[p]} both wrote {p}",
                                  flush=True)
                        claimed[p] = tag
                    print(f"  [{i}/{len(units)}] {tag}: {path or 'no data'}", flush=True)
    finally:
        _os.environ.clear()
        _os.environ.update(env)
    return written, failures


def main(argv=None) -> int:
    # BEFORE ANY FIGURE IS DRAWN. `_save` names the offending tick labels when it reports an
    # overlap, and matplotlib writes a negative one with U+2212, which cp1252 cannot encode --
    # so on Windows the layout reporter killed exactly those figures that had a fault to report,
    # before savefig, leaving the previous render's PNG in place. See wfield_local/console.py.
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--only", nargs="+", default=None, choices=ALL_KEYS)
    ap.add_argument("--jobs", "-j", type=int, default=None, metavar="N",
                    help=f"render N units in parallel (default {_default_jobs()} on this box; "
                         f"pass 1 for the serial render). A unit is one (figure, alignment, "
                         f"trial class).")
    ap.add_argument("--threads-per-worker", type=int, default=None, metavar="N",
                    help="BLAS threads per worker (default: cores // jobs, capped at 2)")
    ap.add_argument("--window", default=None, choices=tuple(w[1] for w in WINDOWS),
                    help="render only this alignment")
    ap.add_argument("--variant", default=None, choices=("lick", "working"),
                    help="render only this trial class")
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ALL_KEYS)

    set_only(args.window, args.variant)

    n_jobs = _default_jobs() if args.jobs is None else args.jobs
    # BOTH PATHS, not just the parallel one: an uncapped serial render produced a different PNG
    # from the identical parallel render, and the escape hatch must not change the answer.
    _pin_blas(args.threads_per_worker or BLAS_THREADS)
    if n_jobs > 1:
        # THE UNITS ARE INDEPENDENT AND THE COST IS ALL IN THE BOOTSTRAPS. Measured on the
        # 2026-08-28 serial render: 7b, 8d, 6d, 7d, 8b and 8e together are 94.7% of 5.79 hours,
        # and each of them writes exactly five files -- one per (alignment, trial class) -- at
        # 10-37 minutes apiece. Collection is ~20 s per unit against that, which is why the unit
        # can be the FIGURE rather than the figure family: a worker re-collecting what a sibling
        # already collected wastes seconds to save half an hour.
        units = [(k, a, v) for k, a, v in render_units(want)]
        if args.window:
            units = [u for u in units if u[1] in (None, args.window)]
        if args.variant:
            units = [u for u in units if u[2] in (None, args.variant)]
        cores = os.cpu_count() or 4
        tpw = args.threads_per_worker or max(1, min(2, cores // max(1, n_jobs)))
        print(f"  {len(units)} units, {n_jobs} workers, {tpw} BLAS thread(s) each "
              f"({cores} cores)", flush=True)
        written, failures = _run_parallel(units, out, n_jobs, tpw)
        print(f"  {written} units wrote, {len(failures)} failed", flush=True)
        for tag, err in failures:
            print(f"    FAILED {tag}: {err}", flush=True)
        return 1 if failures else 0

    for key, fn in JOBS:
        if key not in want:
            continue
        try:
            p = fn(out)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {key}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
            continue
        print(f"  {'wrote ' + str(p) if p else f'{key}: no data'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
