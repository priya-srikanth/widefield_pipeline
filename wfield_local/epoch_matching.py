"""POOLED EPOCH FIGURES — MATCHING

Best-match destination and the cross-position matching grids.

Split out of `epoch_grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS
FROM THE CALL GRAPH: every function here is reached from this family's entry points and from no
other. Anything shared with a sibling is in `epoch_kit`, so this imports from there and never
from `epoch_grant_figures` -- that direction would be a cycle.

Entry points, dispatched by `epoch_grant_figures.main`:
  - `_fig_10`
  - `_fig_10b`
  - `_fig_10cs`
  - `_fig_10e_best_match_grid`
  - `_fig_11`
  - `_fig_11c`
  - `_fig_11cpos`
"""
from __future__ import annotations

import warnings

import numpy as np

from wfield_local import epoch_figures as ef
from wfield_local.epoch_kit import (  # noqa: F401
    _MEAN_NOTE,
    N_BOOT,
    _accuracy_at,
    _groups,
    _long_labels,
    _minor,
    _pre_counts,
    _scalar_figure,
    _seed_for,
    _session_counts,
    _short_labels,
    _totals,
)


def _fig_10e_best_match_grid(out_dir, align, variant, wname):
    """10e: which pre-stroke position each position matches best -- PER ANIMAL, PER EPOCH.

    Priya, 2026-09-11, pointing at `grant_10_best_match`: "I want this best-match confusion matrices
    for epoch data". That grant figure gives each animal a PRE panel and one pooled POST panel; this
    is the same quantity cut by EPOCH instead, so a substitution can be watched appearing and
    resolving within an animal rather than only in the four-animal average that 10c draws.

    WHY PER ANIMAL MATTERS HERE SPECIFICALLY. The pooled 10c panel averages four animals whose
    epochs rest on very unequal session counts -- PS95 contributes one acute session and PS94 six --
    so a pooled off-diagonal cell can be one animal's whole story. The grid cannot hide that: each
    panel carries its own n, and a panel resting on one session looks like one session.

    FRACTIONS, NOT COUNTS, in every panel including pre. `grant_10` prints counts and has to warn in
    its own caption that the two panels' totals differ so only the percentage is comparable; here
    every cell is the fraction of that animal's sessions in that epoch whose best match was that
    column, which is the same construction everywhere and directly comparable across the grid.

    THE PRE PANEL IS A MEAN OF PER-SESSION ONE-HOTS, leave-one-session-out -- never the argmax of
    the averaged correlation matrix, which is a perfect identity for every animal and would draw a
    clean diagonal as the baseline. That error cost the best-match headline 0.02 when it was found
    on 2026-09-10; `_matrices_best_match_destination` carries the fix and this figure inherits it.
    """
    from wfield_local import grant_figures as G

    mats, _days = G._matrices_best_match_destination(align, variant)
    if not mats:
        return None
    store, _d2 = G._collect_7(align, variant, 10)

    grid, counts = {}, {}
    for an, by_key in mats.items():
        per_epoch, n_by = {}, {}
        if by_key.get("PRE") is not None:
            per_epoch["pre"] = np.asarray(by_key["PRE"], float)
            n_by["pre"] = len((store.get(an) or ({}, {}))[0])
        bucket = {}
        for key, M in by_key.items():
            if key == "PRE":
                continue
            e = ef.epoch_of_day(an, int(key))
            if e:
                bucket.setdefault(e, []).append(np.asarray(M, float))
        for e, stack in bucket.items():
            # nanmean over sessions: a row with no finite value in one session must not drag the
            # other sessions' vote toward zero, and `_matrices_best_match_destination` leaves such
            # a row NaN rather than one-hotting column 0. An ALL-NaN row across every session of an
            # epoch is legitimate (that position was never scorable) and numpy warns about it;
            # the warning is the expected case here, not a symptom.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_epoch[e] = np.nanmean(np.stack(stack), axis=0)
            n_by[e] = len(stack)
        if per_epoch:
            grid[an] = per_epoch
            counts[an] = n_by
    if not grid:
        return None

    made = []
    p = ef.matrix_grid_by_animal(
        grid, out_dir, name=f"epoch_10e_best_match_grid_{align}_{variant}",
        title=(f"Which PRE-STROKE position does each position match BEST, by animal and epoch "
               f"-- {wname}"),
        labels=_short_labels(),
        subtitle=("Cell = fraction of that animal's sessions in that epoch whose best pre-stroke "
                  "match was that column. Rows are the TRUE position, columns the pre-stroke "
                  "pattern matched. pre is leave-one-session-out and is NOT 100% -- read every "
                  "post-stroke panel against that animal's own pre panel, never against a perfect "
                  "diagonal. Argmax is invariant to a monotone change across a row, so the uniform "
                  "amplitude shifts that dominate the crossnobis families cannot move this."),
        unit="fraction of sessions", counts=counts, diag_label="self")
    if p:
        made.append(p)

    # THE DELTA GRID, as its own figure rather than as extra columns (Priya, 2026-09-11: "make sure
    # the best match matrices have delta matrices too"). `matrix_row` puts its deltas BENEATH, which
    # works because it has one row; this grid already spends its four rows on animals, and seven
    # columns at QUARTER_IN gives 0.6in panels that cannot carry the per-cell annotation the whole
    # figure is read through. Two figures at full size beats one at half.
    #
    # EACH ANIMAL AGAINST ITS OWN PRE, never against a pooled baseline or a perfect diagonal: the
    # leave-one-session-out pre panel is 92-100% here depending on the animal, and charging PS93 the
    # 8% its own baseline already misses would attribute it to the lesion.
    dgrid, dcounts = {}, {}
    for an, per_epoch in grid.items():
        base = per_epoch.get("pre")
        if base is None:
            continue
        d = {e: np.asarray(M, float) - np.asarray(base, float)
             for e, M in per_epoch.items() if e != "pre"}
        if d:
            dgrid[an] = d
            dcounts[an] = {e: n for e, n in counts[an].items() if e != "pre"}
    if dgrid:
        lim = max((float(np.nanmax(np.abs(M))) for by in dgrid.values() for M in by.values()
                   if np.isfinite(M).any()), default=1.0) or 1.0
        q = ef.matrix_grid_by_animal(
            dgrid, out_dir, name=f"epoch_10edelta_best_match_grid_{align}_{variant}",
            title=(f"Where each position's best match MOVED, by animal and epoch -- change from "
                   f"that animal's own pre-stroke -- {wname}"),
            labels=_short_labels(), cmap="RdBu_r", vmin=-lim, vmax=lim,
            subtitle=("Each panel is that epoch minus THAT ANIMAL's own leave-one-session-out pre "
                      "panel, so a perfect pre-stroke diagonal is not assumed and an animal whose "
                      "baseline already misses is not charged for it. BLUE on the diagonal = the "
                      "position stopped matching itself; RED off the diagonal in the SAME ROW names "
                      "where it went instead. A row that goes blue on the diagonal without any red "
                      "cell is a position that scattered rather than substituted."),
            unit="change in fraction of sessions", counts=dcounts,
            diag_label="self", diag_fmt="{:+.0%}")
        if q:
            made.append(q)
    return made or None
def _fig_10cs(out_dir, align, variant, wname):
    """10cs: the best-match DESTINATION matrix with per-cell bootstrap marks.

    Priya, 2026-09-12: "did we give up on having any bootstrapping of fig 10 best-match matrices?"

    WHAT THIS ADDS THAT 10c AND 10cdiag DO NOT. 10c draws the same matrix with no uncertainty at
    all; 10cdiag tests the DIAGONAL as bars, so "did position P still match itself" is already
    answered with intervals. Neither touches the OFF-DIAGONAL, which is where the substitution claim
    lives -- "acute far-contra now best-matches far-middle" was a colour, not a test. Every cell here
    carries a nested animals->sessions interval from the same bootstrap the bar families use.

    THE REFERENCE IS CHANCE, NOT ZERO, on the absolute row: a cell is the fraction of sessions whose
    best match landed there, so the question is whether it beats the 1/6 a coin gives. The delta row
    is against zero in the ordinary way. `matrix_bootstrap.mark_note` states both, once.
    """
    from wfield_local import grant_figures as G
    from wfield_local import matrix_bootstrap as mb

    mats, _days = G._matrices_best_match_destination(align, variant)
    if not mats:
        return None
    _means, cov = ef.mean_matrix_by_epoch(mats)
    chance = 1.0 / len(_short_labels())
    return mb.figure(
        mats, out_dir, name=f"epoch_10cs_best_match_destination_marked_{align}_{variant}",
        title=f"Where the best match went, with per-cell tests -- {wname}",
        labels=_short_labels(), unit="fraction of sessions", reference=chance,
        vmin=0.0, vmax=1.0, coverage=cov, pre_sessions=_pre_counts(align, variant),
        seed=_seed_for("10cs", align, variant, "cells"),
        subtitle=(f"Rows are this session's position, columns the pre-stroke pattern it matched "
                  f"best. {mb.mark_note(chance)}"))
def _fig_10(out_dir, align, variant, wname):
    """10 pooled: best-match accuracy and the correct position's rank, by epoch.

    BUILT ON `_matrices_pattern`, not `_match_tables`. Priya asked for a pre-stroke bar and
    `_match_tables` cannot give one: its per-day dict is post-stroke only, with pre held separately
    as a 6x6 of counts from which a mean rank cannot be recovered. `_matrices_pattern` is keyed
    "PRE"|day, so the baseline is a session like any other -- and it is the same source 10b reads,
    so the two figures cannot disagree about what a best match is.

    `_best_match` is `grant_figures`' own, used rather than reimplemented: it takes the WHOLE ROW,
    which is what separates "the code is gone" from "the code moved somewhere specific".
    """
    from wfield_local import grant_figures as G

    mats, _days = G._matrices_pattern(align, variant)
    if not mats:
        return None
    ACC, RANK = "best-match accuracy", "rank of correct position"

    def value_of(M, key):
        best, rank = G._best_match(np.asarray(M, float))
        ok = np.isfinite(rank)
        if not ok.any():
            return None
        if key == ACC:
            return float(np.mean(best[ok] == np.flatnonzero(ok)))
        return float(np.nanmean(rank))

    values, points = ef.scalar_by_epoch(mats, value_of, keys=[ACC, RANK])
    if not values:
        return None
    made = None
    for key, ylim, chance in ((ACC, (0.0, 1.10), 1.0 / 6.0), (RANK, (0.0, 6.4), None)):
        v = {e: {key: d[key]} for e, d in values.items() if key in d}
        pt = {e: {key: points[e][key]} for e in v}
        if not v:
            continue
        got = _scalar_figure(
            out_dir, name=f"epoch_10_best_match_{'acc' if key == ACC else 'rank'}_{align}_{variant}",
            title=f"Best match against the pre-stroke patterns -- {wname}",
            ylabel=key, keys=[key], values=v, points=pt, chance=chance, ylim=ylim,
            delta_name=(f"epoch_10delta_best_match_"
                        f"{'acc' if key == ACC else 'rank'}_{align}_{variant}"),
            delta_title=f"Change from pre-stroke in {key} -- {wname}")
        made = made or got
    return made
def _fig_10b(out_dir, align, variant, wname):
    """10b pooled: is the best-matching pre-stroke pattern the CORRECT position, per position?

    Read off `_matrices_pattern` -- ``{animal: {"PRE"|day: M}}`` where ``M[i, j]`` is the
    correlation of this session's pattern at position i with the pre-stroke pattern at j. The
    session scores 1 at position i when j = i is the argmax of that row.

    A ROW OF ALL-NaN SCORES NOTHING rather than scoring zero: a position gated out for too few
    trials has no best match, and counting that as "matched the wrong position" would turn missing
    data into evidence of reorganisation, which is the direction that would flatter the result.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    mats, _days = G._matrices_pattern(align, variant)
    if not mats:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))
    idx = {short[q]: i for i, q in enumerate(CONF_LABELS)}

    def value_of(M, key):
        i = idx[key]
        row = np.asarray(M, float)[i]
        if not np.isfinite(row).any():
            return None
        return float(np.nanargmax(row) == i)

    values, points = ef.scalar_by_epoch(mats, value_of, keys=_short_labels())
    if not values:
        return None
    return _scalar_figure(
        out_dir, name=f"epoch_10b_best_match_by_position_{align}_{variant}",
        title=f"Fraction of sessions whose best pre-stroke match is the correct position -- {wname}",
        ylabel="fraction of sessions", keys=_short_labels(), values=values, points=points,
        tick_labels=_minor(), groups=_groups(), chance=1.0 / len(CONF_LABELS), ylim=(0.0, 1.10),
        delta_name=f"epoch_10bdelta_best_match_by_position_{align}_{variant}",
        delta_title=f"Change from pre-stroke in best-match fraction -- {wname}")
def _fig_11(out_dir, align, variant, wname):
    """11 pooled: the encoder's explained variance and gain, by epoch.

    `_enc_tables` is keyed "PRE"|day, so the pre-stroke baseline is a bar here without any special
    handling -- unlike figure 10, whose collector holds pre separately and had to be rebuilt on
    `_matrices_pattern` to get one.
    """
    from wfield_local import grant_figures as G

    tab, _days = G._enc_tables(align, variant)
    if not tab:
        return None
    # THREE THINGS WERE CALLED "gain" IN THIS FAMILY and Priya, reasonably, could not tell them
    # apart: the explained variance AFTER one best rescale (this figure's second bar), the fitted
    # amplitude factor `a` (the 11amp figure), and the quantity 11pos calls "shape r2" -- which is
    # the SAME number as this figure's second bar, per position. Renamed to say what each is.
    LABELS = ["frozen EV", "EV after rescale"]
    take = {"frozen EV": 0, "EV after rescale": 2}

    def value_of(payload, key):
        try:
            v = payload[take[key]]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None else float(v)

    values, points = ef.scalar_by_epoch(tab, value_of, keys=LABELS)
    if not values:
        return None
    made = _scalar_figure(
        out_dir, name=f"epoch_11_encoder_gain_shape_{align}_{variant}",
        title=(f"Encoder explained variance, before and after ONE best rescale -- {wname}"),
        ylabel="value", keys=LABELS, values=values, points=points, ylim=(0.0, 1.30),
        delta_name=f"epoch_11delta_encoder_gain_shape_{align}_{variant}",
        delta_title=f"Change from pre-stroke in encoder explained variance -- {wname}")

    # THE FITTED GAIN ITSELF, H11's middle panel, which the epoch version had been dropping.
    # `_enc_terms` returns (raw, a, gain, per) and this figure took only 0 and 2 -- transfer before
    # rescaling and after -- so `a`, the amplitude term the whole decomposition exists to isolate,
    # was computed nightly and never shown by epoch.
    #
    # ITS OWN FIGURE, NOT A THIRD BAR BESIDE raw AND gain. Those two are variance-explained on
    # [0, 1]; `a` is a RATIO around 1.0 and unbounded above. Sharing one axis would either squash
    # the r2 pair into the bottom third or clip `a` -- the same category error as the crossnobis
    # bars fixed earlier today, where a bound belonging to one quantity was applied to another.
    #
    # 1.0 IS THE MEANINGFUL VALUE, not zero: a = 1 means no amplitude change, a < 1 a weaker code,
    # a > 1 a stronger one. It is deliberately NOT passed as `chance=1.0` -- that would draw the
    # line but also append "(chance 1.00)" to the y-label, and a gain of 1 is not a chance level.
    def _amp(payload, key):
        try:
            v = payload[1]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None or not np.isfinite(v) else float(v)

    avals, apoints = ef.scalar_by_epoch(tab, _amp, keys=["gain"])
    if avals:
        try:
            _scalar_figure(
                out_dir, name=f"epoch_11amp_encoder_amplitude_{align}_{variant}",
                title=(f"Fitted AMPLITUDE factor a by epoch (1.0 = no amplitude change; "
                       f"below 1 = smaller) -- {wname}"),
                ylabel="amplitude factor a", keys=["gain"], values=avals, points=apoints,
                ylim=None,
                delta_name=f"epoch_11ampdelta_encoder_amplitude_{align}_{variant}",
                delta_title=f"Change from pre-stroke in the fitted amplitude factor a -- {wname}")
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 11amp {align}/{variant}: {type(ex).__name__} {str(ex)[:120]}", flush=True)

    # PER POSITION, the way the decoding families already are (Priya, 2026-09-07). `_enc_terms`
    # already returns `{position: shape r2 after the gain}` as its fourth term and this figure was
    # reading only terms 0 and 2, so the per-position tuning was computed every night and shown to
    # nobody.
    #
    # SHAPE r2 IS THE RIGHT PER-POSITION QUANTITY AND THE GAIN IS NOT. `_enc_terms` fits ONE gain
    # per session deliberately -- "a per-position gain would absorb the position-specific amplitude
    # loss that IS the deficit, and the decomposition would say nothing". Shape r2 asks the question
    # that survives that: with the session's amplitude change already divided out, is THIS position
    # still predicted by its pre-stroke pattern?
    #
    # AND IT IS THE PER-POSITION COMPLEMENT OF THE DECODING RECALL. Recall asks whether a position
    # stays DISCRIMINABLE from the other five; shape r2 asks whether its pattern is still the
    # pre-stroke one. A position can stay decodable on a changed pattern, so the two dissociating is
    # the measurement behind "decodable but re-geometried" -- read them side by side.
    short = dict(zip(G.CONF_LABELS, _short_labels()))

    def _shape_at(payload, key):
        try:
            per = payload[3] or {}
        except Exception:                                              # noqa: BLE001
            return None
        for q, sh in short.items():
            if sh == key:
                v = per.get(q)
                return None if v is None or not np.isfinite(v) else float(v)
        return None

    pvals, ppoints = ef.scalar_by_epoch(tab, _shape_at, keys=_short_labels())
    if pvals:
        try:
            _scalar_figure(
                out_dir, name=f"epoch_11pos_encoder_shape_{align}_{variant}",
                title=(f"Encoder EV after rescale (= shape r2), per position -- {wname}"),
                ylabel="EV after rescale", keys=_short_labels(),
                values=pvals, points=ppoints, tick_labels=_minor(), groups=_groups(),
                # r2 is bounded above by 1 and NOT below: a pattern unrelated to its reference goes
                # sharply negative, so a fixed floor would clip exactly the positions that lost
                # their tuning -- the ones the figure exists to find.
                ylim=None,
                delta_name=f"epoch_11posdelta_encoder_shape_{align}_{variant}",
                delta_title=f"Change from pre-stroke in per-position encoder shape r² -- {wname}")
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 11pos {align}/{variant}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
    return made
def _fig_11c(out_dir, align, variant, wname):
    """11c: the encoder's CEILING, and how much of its failure is the TEMPLATE being wrong.

    THE NUMBER THIS FIGURE EXISTS TO MAKE READABLE. Figure 11's `frozen EV` is an R^2, and acutely it
    is -0.388 post-cue: "worse than predicting the mean", and mute about what was achievable in that
    session. `EV after rescale` frees the AMPLITUDE only, and a large gap between the two is an
    amplitude story only when the rescaled value is HIGH -- a code that is simply gone also recovers
    a lot under rescaling. That ambiguity forced the withdrawal of the "half amplitude, half shape"
    reading on 2026-09-10.

    THREE BARS, ALL DIRECTLY MEASURED, all in the same units, all scoring the SAME half-session:

        ceiling          scored against the OTHER half of the same session -- how much position
                         structure the session has at all
        frozen (matched) scored against an equally sized draw from the pre-stroke pool -- the same
                         estimator and the same number of reference trials, differing only in WHICH
                         SESSIONS the reference came from
        frozen (all pre) scored against the whole pre-stroke pool: the encoder as it is actually
                         used elsewhere in this deck, and the only one of the three that is not
                         size-matched

    READ `ceiling - frozen (matched)`, NOT `ceiling - frozen (all pre)`. Only the first holds
    training-set size constant, so only the first isolates "the template came from other sessions".
    And read it against its own PRE value rather than against zero: at pre-stroke that difference is
    the cross-session generalisation cost with no lesion involved, and it is large -- 0.276 post-cue.

    WHY THE MATCHED ARM HAD TO BE ADDED. Without it the pre-cue window was uninterpretable: frozen EV
    0.330 against a ceiling of 0.090, a frozen arm beating its own ceiling, which is impossible for a
    real ceiling and was a fact about the construction rather than the data. Matching removes the
    asymmetry and the inversion goes with it, in all three windows.

    THE PRE-CUE WINDOW IS NOW SELF-CONSISTENT BUT STILL NEARLY SIGNAL-FREE: its ceiling is 0.117
    pre-stroke and hovers near zero afterwards, so the gap there is a ratio of two near-zero
    quantities. Consistent is not the same as informative.
    """
    from wfield_local import grant_figures as G

    raw_tab, _d1 = G._enc_tables(align, variant)
    cei_tab, _d2 = G._enc_ceiling_tables(align, variant)
    mat_tab, _d3 = G._enc_matched_tables(align, variant)
    if not cei_tab or not mat_tab:
        return None

    def _first(payload, _key):
        """THE AFTER-RESCALE SCORE, index 2 -- not the raw one. Priya: "what does negative variance
        explained mean though".

        It means the template's prediction is FURTHER from the data than predicting zero, i.e. than
        assuming no position tuning at all. And at these amplitudes that is almost entirely an
        amplitude statement: with PERFECT shape and only a scale mismatch,
        ``R2 = 1 - (1-a)^2 / a^2``, which is 0.000 at a = 0.5 and -2.13 at the a = 0.361 observed
        acutely. So a raw comparison between the ceiling and the matched arm is confounded: the
        ceiling's two halves have matched amplitude BY CONSTRUCTION (a = 0.75-0.89) while the matched
        frozen arm scores a shrunken post-stroke pattern against a full-amplitude pre-stroke template
        (a = 0.286 acutely in this arm, 0.361 in the unmatched one). The raw gap would be reporting
        amplitude and calling it template mismatch.

        The after-rescale score is amplitude-free on both sides and cannot go negative -- a = 0 is
        always available, which scores exactly 0 -- so this figure is about SHAPE and nothing else.
        The amplitude story is figure 11amp's, where it belongs.
        """
        try:
            v = payload[2]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None or not np.isfinite(v) else float(v)

    KEYS = ["ceiling", "frozen (matched)", "frozen (all pre)"]
    got = {}
    for key, tab in zip(KEYS, (cei_tab, mat_tab, raw_tab)):
        vals, pts = ef.scalar_by_epoch(tab, _first, keys=[key.split(" ")[0]])
        got[key] = (vals, pts, key.split(" ")[0])

    values, points = {}, {}
    for e in ef.PANELS:
        row, pt = {}, {}
        for key, (vals, pts, inner) in got.items():
            v = (vals.get(e) or {}).get(inner)
            if v is not None:
                row[key] = v
                pt[key] = (pts.get(e) or {}).get(inner, [])
        if row:
            values[e], points[e] = row, pt
    if not values:
        return None
    return _scalar_figure(
        out_dir, name=f"epoch_11c_encoder_ceiling_{align}_{variant}",
        title=(f"Encoder SHAPE ceiling vs the frozen template, training-set matched "
               f"-- {wname}"),
        ylabel="EV after rescale (shape only)", keys=KEYS, values=values,
        points=points, ylim=(0.0, 1.05),
        # WRAPPED, because "frozen (matched)" and "frozen (all pre)" collide at this axis width and
        # the collision lands on the two bars a reader most needs to tell apart.
        tick_labels=["ceiling", "frozen\n(matched)", "frozen\n(all pre)"],
        delta_name=f"epoch_11cdelta_encoder_ceiling_{align}_{variant}",
        delta_title=f"Encoder ceiling and frozen template, change from pre-stroke -- {wname}")
def _fig_11cpos(out_dir, align, variant, wname):
    """11cpos: the encoder SHAPE ceiling and the frozen template, PER POSITION.

    11c pools the six positions into one number per epoch, which is the wrong shape for the question
    the lesion poses: far-contralateral is the position the deficit lives at, and a pooled ceiling
    averages it with five positions that barely moved. This is the same three quantities per
    position.

    TWO FIGURES. The first is the ceiling alone -- how much SHAPE each position's own trials can
    predict, which is a statement about that position's data and not about the template. The second
    is the CAPTURED FRACTION, `matched / ceiling`: of the shape a position's own trials can predict,
    how much does the pre-stroke template get? A fraction rather than a difference, because the
    positions do not share a ceiling and a raw gap of 0.2 means something different at a position
    whose ceiling is 0.3 than at one whose ceiling is 0.8.

    AFTER-RESCALE THROUGHOUT, for the reason `_fig_11c` gives at length: a raw score at these
    amplitudes is dominated by the encoder not being allowed to rescale, and the ceiling's halves
    have matched amplitude by construction while the frozen arm's reference does not.

    THE FRACTION IS CLIPPED FOR DISPLAY AND NOT FOR COMPUTATION. Where a ceiling is near zero the
    ratio is unstable and can exceed 1 or go negative; those cells are dropped rather than drawn,
    because a 300% "captured" bar is an artefact of a small denominator and reads as a result.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    cei_tab, _d1 = G._enc_ceiling_tables(align, variant)
    mat_tab, _d2 = G._enc_matched_tables(align, variant)
    if not cei_tab or not mat_tab:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))

    def _per(payload, key):
        try:
            per = payload[3] or {}
        except Exception:                                              # noqa: BLE001
            return None
        for q, sh in short.items():
            if sh == key:
                v = per.get(q)
                return None if v is None or not np.isfinite(v) else float(v)
        return None

    cvals, cpts = ef.scalar_by_epoch(cei_tab, _per, keys=_short_labels())
    mvals, _mp = ef.scalar_by_epoch(mat_tab, _per, keys=_short_labels())
    if not cvals:
        return None
    made = []
    p1 = _scalar_figure(
        out_dir, name=f"epoch_11cpos_encoder_ceiling_by_position_{align}_{variant}",
        title=f"Encoder SHAPE ceiling per position -- {wname}",
        ylabel="EV after rescale (shape)", keys=_short_labels(), values=cvals, points=cpts,
        tick_labels=_minor(), groups=_groups(), ylim=(0.0, 1.05),
        delta_name=f"epoch_11cposdelta_encoder_ceiling_by_position_{align}_{variant}",
        delta_title=f"Encoder shape ceiling per position, change from pre-stroke -- {wname}")
    if p1:
        made.append(p1)

    #: A per-session ceiling below this makes that session's captured FRACTION a ratio of noise.
    #: Applied PER SESSION, not to the pooled value: one session whose far-contra ceiling collapsed
    #: would otherwise drag a pooled ratio that the other fifteen support.
    MIN_CEILING = 0.10

    # THE FRACTION IS BUILT PER SESSION AND THEN POOLED, not computed from the pooled ceiling and
    # the pooled match. Priya, 2026-09-10: "why are there no sem or stats on the per position
    # fraction explained vs ceiling?" Because the first version divided one pooled number by
    # another, which has no distribution behind it and therefore no interval and no dots -- the one
    # bar family in this section that could not be argued with. A per-session ratio has both, and
    # goes through the same animals -> sessions bootstrap as every other bar here.
    #
    # A SYNTHETIC TABLE rather than a new collector: `scalar_by_epoch` wants one table whose payload
    # it can reduce, and the two arms are already keyed identically by (animal, PRE|day), so the
    # ratio can be formed session by session and handed back in the same shape. Slots 0-2 are NaN
    # because nothing reads them here; slot 3 is the per-position dict the accessor expects.
    frac_tab = {}
    for an, by_key in cei_tab.items():
        rec = {}
        for key, cpay in by_key.items():
            mpay = (mat_tab.get(an) or {}).get(key)
            if mpay is None:
                continue
            cper, mper = (cpay[3] or {}), (mpay[3] or {})
            row = {}
            for q in CONF_LABELS:
                c, m = cper.get(q), mper.get(q)
                if c is None or m is None or not np.isfinite(c) or not np.isfinite(m):
                    continue
                if c < MIN_CEILING:
                    continue
                row[q] = float(max(0.0, min(1.0, m / c)))
            if row:
                rec[key] = (np.nan, np.nan, np.nan, row)
        if len(rec) > 1:
            frac_tab[an] = rec

    if frac_tab:
        fvals, fpts = ef.scalar_by_epoch(frac_tab, _per, keys=_short_labels())
        if fvals:
            p2 = _scalar_figure(
                out_dir, name=f"epoch_11cfrac_encoder_captured_by_position_{align}_{variant}",
                title=(f"Of the shape a position's own trials can predict, how much does the "
                       f"PRE-STROKE template capture? -- {wname}"),
                ylabel="matched / ceiling", keys=_short_labels(), values=fvals, points=fpts,
                tick_labels=_minor(), groups=_groups(), ylim=(0.0, 1.10),
                delta_name=f"epoch_11cfracdelta_encoder_captured_by_position_{align}_{variant}",
                delta_title=f"Captured fraction per position, change from pre-stroke -- {wname}")
            if p2:
                made.append(p2)
    return made or None
