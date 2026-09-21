"""POOLED EPOCH FIGURES — ACCURACY

Per-position accuracy, the confusion rows, and the frozen-vs-refit contrast.

Split out of `epoch_grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS
FROM THE CALL GRAPH: every function here is reached from this family's entry points and from no
other. Anything shared with a sibling is in `epoch_kit`, so this imports from there and never
from `epoch_grant_figures` -- that direction would be a cycle.

Entry points, dispatched by `epoch_grant_figures.main`:
  - `_confusion_rows`
  - `_refit_confusion_rows`
  - `_per_position_accuracy`
  - `_matrix_family`
  - `_frozen_vs_refit`
  - `_frozen_vs_refit_matched`
  - `_frozen_vs_refit_overall`
  - `_frozen_vs_refit_overall_matched`
  - `_epoch_arm`
  - `_report`
  - `_fig_8g`
  - `_fig_9`
"""
from __future__ import annotations

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

CHANCE = 1.0 / 6.0
def _named(record):
    """A record's ``(true, predicted)`` as POSITION NAMES.

    `_collect_5c` stores the NUMERIC position codes the decoder was fitted on, not the display
    names -- which is why `grant_figures._counts` maps through `POSITION_NAMES` before indexing
    `CONF_LABELS`. Comparing the raw codes to a name silently matches nothing: the first run of
    this figure produced empty panels and reported "wrote None", with no error anywhere.
    """
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    def nm(v):
        return np.array([POSITION_NAMES.get(int(x), str(x)) for x in np.asarray(v)])

    return nm(record[0]), nm(record[1])
def _code_of(position):
    """The NUMERIC code for a position name.

    `contrast_draws` gets its arrays straight from `pool_records`, which returns the records as
    stored -- the integer codes the decoder was fitted on, NOT display names. Comparing those to
    "close_L" matches nothing and returns an empty figure with no error, which is exactly the
    fault that produced three blank panels earlier today. Working in codes here keeps the mapping
    in one place instead of naming twenty thousand trials per bootstrap draw.
    """
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    for code, nm in POSITION_NAMES.items():
        if nm == position:
            return int(code)
    return None
def _accuracy_of(record, position=None):
    """Accuracy of one record, overall or at one true position. `None` when the class is absent."""
    if record is None:
        return None
    y, p = _named(record)
    if position is not None:
        m = (y == position)
        if m.sum() < 5:
            return None
        y, p = y[m], p[m]
    return float(np.mean(y == p)) if len(y) else None
def _gap_at(y, p, code):
    """Refit-minus-frozen accuracy at one true position CODE, from a PAIRED record.

    ``p`` is the (n, 2) array `grant_figures._collect_5c(mode="paired")` returns -- column 0 the
    frozen pre-stroke decoder, column 1 the within-session refit -- so the difference is taken on
    the SAME trials and survives every level of the block bootstrap paired.

    Trials whose class the session could not TRAIN on are dropped from both arms together (the
    refit column carries `grant_figures.REFIT_UNAVAILABLE`), which keeps the pairing while refusing
    to score a decoder on a class it never saw. Without it the lick-aligned acute far-contralateral
    cell reads -0.42 -- the mouse not licking, presented as the code being gone.

    Positive = the session's own decoder reads a position the frozen one cannot: information
    PRESENT but unreadable by the pre-stroke model. Zero = both fail together, which is what a
    genuinely degraded code looks like.
    """
    from wfield_local.grant_figures import REFIT_UNAVAILABLE

    if code is None:
        return None
    y, p = np.asarray(y), np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("_gap_at needs a paired record: use _collect_5c(..., mode='paired')")
    m = (y == code) & (p[:, 1] != REFIT_UNAVAILABLE)
    if m.sum() < 5:
        return None
    return float(np.mean(p[m, 1] == y[m]) - np.mean(p[m, 0] == y[m]))
def _gap_of(record, position):
    """`_gap_at` for one whole record, at one position NAME. `None` when the class is absent."""
    if record is None:
        return None
    return _gap_at(record[0], record[1], _code_of(position))
def _refit_at(y, p, code):
    """Accuracy of the REFIT column of a paired record at one true position code.

    Trials the session could not train on (`grant_figures.MIN_REFIT_CLASS`) are DROPPED, not scored
    as errors: the refit arm was never asked about them. Dropping them from both arms is what keeps
    this comparable to `_gap_at`, which drops the same trials.
    """
    from wfield_local.grant_figures import REFIT_UNAVAILABLE

    if code is None:
        return None
    y, p = np.asarray(y), np.asarray(p)
    m = (y == code) & (p[:, 1] != REFIT_UNAVAILABLE)
    if m.sum() < 5:
        return None
    return float(np.mean(p[m, 1] == y[m]))
def _refit_of(record, position):
    if record is None:
        return None
    return _refit_at(record[0], record[1], _code_of(position))
def _per_position_accuracy(per_animal, out_dir, disp, align, variant, wname):
    """Per-position accuracy of the frozen pre-stroke decoder, by epoch."""
    return _position_bars(
        per_animal, out_dir, align, variant, wname,
        name=f"epoch_acc_by_position_{align}_{variant}",
        title=f"Per-position decoding accuracy, {wname} -- pooled across animals",
        delta_name=f"epoch_accdelta_by_position_{align}_{variant}",
        delta_title=f"Change from pre-stroke in per-position accuracy, {wname}",
        ylabel="accuracy", delta_ylabel="accuracy - pre",
        stat_at=_accuracy_at, value_of=_accuracy_of,
        chance=CHANCE, ylim=(0.0, 1.10),
        notes=["pre panel is leave-one-session-out within each animal, pooled across animals"])
def _position_bars(per_animal, out_dir, align, variant, wname, *, name, title, delta_name,
                   delta_title, ylabel, delta_ylabel, stat_at, value_of, chance, ylim, notes,
                   reference=None):
    """ONE path for every per-position bar row built from decoder records.

    Frozen accuracy, refit accuracy and the paired refit-minus-frozen gap differ only in which
    statistic reduces a trial set, so the resampling, the marks, the Bonferroni correction across
    the twelve contrasts and the companion interval panel are written once. Three copies would
    agree today and diverge the first time one of them gained a correction -- the same argument
    `_scalar_figure` already makes for the scalar families.

    ``stat_at(y, p, code) -> float | None`` reduces a POOLED trial set at one position;
    ``value_of(record, position_name) -> float | None`` reduces ONE session for its dot.
    """
    from wfield_local.grant_figures import CONF_LABELS

    short = dict(zip(CONF_LABELS, _short_labels()))
    values, points = {}, {}
    for e in ef.PANELS:
        rec = ef.pool_records(per_animal, e)
        if rec is None:
            continue
        values[e] = {}
        points[e] = {}
        for q in CONF_LABELS:
            # THE BAR'S OWN INTERVAL, from the same animals -> sessions -> blocks resampling the
            # marks use. Passing a bare number drew no error bar at all, silently: `bar_row` draws
            # one only where a value arrives as a (value, lo, hi) tuple.
            got = ef.value_draws(
                per_animal, e, lambda y, pr, c=_code_of(q): stat_at(y, pr, c),
                rng=np.random.default_rng(_seed_for(align, variant, e, f"{q}-value")),
                n_boot=N_BOOT)
            a = ef.with_ci(got) if got is not None else value_of(rec, q)
            if a is not None:
                values[e][short[q]] = a
            # one dot per session, from the SAME records the pooled bar sums
            pts = ef.per_session_values(per_animal, e, lambda _an, _d, r, q=q: value_of(r, q))
            points[e][short[q]] = [(an, v) for an, v in pts if v is not None]
    if not values:
        return None
    cov = ef.epoch_coverage(per_animal)
    per_epoch = dict(cov["per_epoch"])
    # A LIST OF PRE RECORDS IS A LIST OF SESSIONS. The frozen arm's pre entry is one
    # leave-one-session-out pool per animal and counts as one; the refit arm's is one record per
    # pre-stroke session, and counting that as one would print "pre 4" under a bar resting on 44.
    def _n_pre(pre):
        return 0 if pre is None else (len(pre) if isinstance(pre, list) else 1)

    per_epoch["pre"] = {an: _n_pre(per_animal.get(an, (None, {}))[0]) for an in per_animal}

    # THE CONTRAST AGAINST PRE-STROKE, per position, resampling animals -> sessions -> blocks.
    # Draws are taken ONCE per contrast and summarised at both an uncorrected and a corrected
    # level, so the corrected interval necessarily contains the uncorrected one -- two separate
    # bootstraps could not guarantee that and could print a corrected interval that was narrower.
    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    n_comp = sum(len(values[e]) for e in post)          # every contrast this figure draws
    marks, rows = {}, {}
    for e in post:
        marks[e], rows[e] = {}, {}
        for q in CONF_LABELS:
            key = short[q]
            if key not in values[e]:
                continue
            got = ef.contrast_draws(
                per_animal, e, "pre",
                lambda y, pr, c=_code_of(q): stat_at(y, pr, c),
                # SEEDED PER (window, class, epoch, position), stably: the same figure has to give
                # the same interval on every render, and `hash()` is salted per process.
                rng=np.random.default_rng(_seed_for(align, variant, e, q)), n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            marks[e][key] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][key] = (point, float(lo), float(hi), float(clo), float(chi))

    # SESSION COUNTS COME FROM THE EPOCH ASSIGNMENT, not from the panel's construction. The pre
    # panel here is four leave-one-session-out records -- one per animal -- so counting the panel
    # would print "pre 4" beside "acute 16" and imply the baseline rests on four sessions when it
    # rests on forty-four.
    from wfield_local import grant_figures as _G
    sub = ef.stats_line(_session_counts(_G.pre_session_counts(align, variant)),
                        blocks=ef.block_counts(per_animal), n_boot=N_BOOT, notes=list(notes))
    made = ef.bar_row(
        values, out_dir, name=name, title=title,
        subtitle=sub, counts=_totals(per_epoch), marks=marks, points=points,
        ylabel=ylabel, positions=_short_labels(), tick_labels=_minor(), groups=_groups(),
        chance=chance, ylim=ylim, reference=reference)
    if any(rows.values()):
        ef.contrast_panel(
            rows, out_dir, name=delta_name, title=delta_title,
            subtitle=sub, ylabel=delta_ylabel, positions=_short_labels(),
            tick_labels=_minor(), groups=_groups(), n_comparisons=n_comp)
    return made
#: The three arms of the overall (all-positions) frozen-vs-refit figure, in drawing order.
#: Named rather than positional because the delta panel's caption has to say which sign means what.
OVERALL_ARMS = ("frozen", "refit", "gap")
def _overall_at(y, p, arm):
    """One arm's statistic on a POOLED paired trial set, all positions together.

    The per-position family (`_gap_at`, `_refit_at`) exists because the deficit is graded across
    positions. This is the same reduction with the position index dropped, and it is the quantity
    with the POWER: a position gets a sixth of the trials, and at chronic that is the difference
    between an interval of 0.076 and one of 0.056 against an effect of ~0.075 (DECISIONS,
    2026-09-12 power decomposition).

    Trials the refit arm could not train on are dropped from BOTH arms, exactly as `_gap_at` drops
    them, so the three arms are computed on one trial set and `gap` is `refit - frozen` on it.
    """
    from wfield_local.grant_figures import REFIT_UNAVAILABLE

    y, p = np.asarray(y), np.asarray(p)
    if p.ndim != 2:
        return None
    m = p[:, 1] != REFIT_UNAVAILABLE
    if m.sum() < 20:
        return None
    fz = float((p[m, 0] == y[m]).mean())
    rf = float((p[m, 1] == y[m]).mean())
    return {"frozen": fz, "refit": rf, "gap": rf - fz}[arm]
def _overall_of(record, arm):
    """`_overall_at` for one whole record -- one session's dot."""
    if record is None:
        return None
    return _overall_at(record[0], record[1], arm)
def _frozen_vs_refit_overall(out_dir, align, variant, wname, *, matched=False):
    """5ro: the frozen and refit arms POOLED OVER POSITIONS, as bars by epoch plus a delta panel.

    Priya, 2026-09-12: a version of F and G "with pre-stroke and post-stroke epoch binning and dots
    for sessions color coded for animal (like our other bar graphs) and a delta line graph compared
    to 0 with stats labels (as for other graph families)".

    **This is the panel the power analysis says to lead with.** `recovery_trajectory` shows the same
    two quantities per session, which is the right picture of a ROUTE but carries no intervals; the
    per-position family carries intervals but spends five sixths of its trials on a resolution the
    cohort cannot support at chronic. Pooling over positions is what makes the interval narrower
    than the effect.

    **SIGN CONVENTION, because it differs from `recovery_trajectory` on purpose.** The delta panel
    is ``epoch - pre``, as every other contrast panel in this deck is, so the frozen arm goes
    NEGATIVE when the readout is worse. `recovery_trajectory`'s ``F`` is the same quantity with the
    sign flipped (``pre - frozen``) so that both of its axes move positive with the lesion. ``G``
    there IS the gap arm here, unflipped. Two conventions is one more than ideal; the alternative
    was a contrast panel whose bars point the opposite way from every other contrast panel.
    """
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant,
                                      "paired_matched" if matched else "paired")
    if not per_animal:
        return None
    values, points = {}, {}
    for e in ef.PANELS:
        rec = ef.pool_records(per_animal, e)
        if rec is None:
            continue
        values[e], points[e] = {}, {}
        for arm in OVERALL_ARMS:
            got = ef.value_draws(
                per_animal, e, lambda y, pr, a=arm: _overall_at(y, pr, a),
                rng=np.random.default_rng(_seed_for(align, variant, e, f"overall-{arm}-value")),
                n_boot=N_BOOT)
            v = ef.with_ci(got) if got is not None else _overall_of(rec, arm)
            if v is not None:
                values[e][arm] = v
            pts = ef.per_session_values(per_animal, e,
                                        lambda _an, _d, r, a=arm: _overall_of(r, a))
            points[e][arm] = [(an, v) for an, v in pts if v is not None]
    if not values:
        return None

    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    n_comp = sum(len(values[e]) for e in post)
    marks, rows = {}, {}
    for e in post:
        marks[e], rows[e] = {}, {}
        for arm in OVERALL_ARMS:
            if arm not in values[e]:
                continue
            got = ef.contrast_draws(
                per_animal, e, "pre", lambda y, pr, a=arm: _overall_at(y, pr, a),
                rng=np.random.default_rng(_seed_for(align, variant, e, f"overall-{arm}")),
                n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            marks[e][arm] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][arm] = (point, float(lo), float(hi), float(clo), float(chi))

    key = "5rmo" if matched else "5ro"
    what = " (training-set MATCHED)" if matched else ""
    sub = ef.stats_line(_session_counts(G.pre_session_counts(align, variant)),
                        blocks=ef.block_counts(per_animal), n_boot=N_BOOT,
                        notes=["pooled over all six positions -- the per-position family spends "
                               "5/6 of its trials on a resolution this cohort cannot support "
                               "chronically",
                               "gap = refit - frozen, paired within trial",
                               "delta panel is epoch - pre, so the frozen arm goes NEGATIVE when "
                               "the readout is worse"])
    made = ef.bar_row(
        values, out_dir, name=f"epoch_{key}_frozen_refit_overall_{align}_{variant}",
        title=f"Frozen vs refit decoding, pooled over positions{what}, {wname}",
        subtitle=sub, counts=None, marks=marks, points=points,
        ylabel="accuracy (gap: refit - frozen)", positions=list(OVERALL_ARMS),
        tick_labels=["frozen", "refit", "gap"], chance=None, reference=0.0, ylim=None)
    if any(rows.values()):
        ef.contrast_panel(
            rows, out_dir, name=f"epoch_{key}delta_frozen_refit_overall_{align}_{variant}",
            title=f"Change from pre-stroke, pooled over positions{what}, {wname}",
            subtitle=sub, ylabel="epoch - pre", positions=list(OVERALL_ARMS),
            tick_labels=["frozen", "refit", "gap"], n_comparisons=n_comp)
    # A LIST, because the render loop does `for q in (fn(...) or [])` and `ef.bar_row`
    # returns ONE Path. Returning the Path itself raised
    #     TypeError: 'WindowsPath' object is not iterable
    # on every align/variant -- AFTER both figures and both sidecars had been written, so
    # the outputs looked complete on disk while every combination was logged as failed.
    # `_frozen_vs_refit` returns a list for the same reason; this had to match it.
    return [made] if made else None
def _frozen_vs_refit_overall_matched(out_dir, align, variant, wname):
    """5rmo: the pooled-over-positions contrast with both arms given the same training-set size."""
    return _frozen_vs_refit_overall(out_dir, align, variant, wname, matched=True)
def _frozen_vs_refit_matched(out_dir, align, variant, wname):
    """5rm: the same contrast with the two arms given the SAME AMOUNT of training data.

    Priya, 2026-09-10: "build the training set-matched version too". The unmatched pre bar is a
    training-set-size handicap -- ten pre-stroke sessions against four fifths of one -- so the raw
    gap cannot be read directly. Matching removes it, and what it exposes is that the baseline does
    not go to zero: it goes the OTHER WAY. At equal training-set size the refit arm is +0.090 BETTER
    pre-stroke (post-cue), because training within a session shares that session's own nuisance
    structure -- alignment, haemodynamics, arousal, the LocaNMF projection -- while the matched
    frozen model has to generalise across days.

    SO THE NO-LESION BASELINE IS BRACKETED RATHER THAN KNOWN: -0.073 unmatched (frozen has 10x the
    data) and +0.090 matched (frozen must cross sessions). Both bounds are real effects and neither
    is "the" answer, which is why both families are drawn and both are read as
    epoch-minus-pre. The headline survives either way -- far-contralateral has the largest
    lesion-attributable recoverable component, +0.196 unmatched and +0.158 matched.

    ONLY THE GAP IS DRAWN HERE. The refit-accuracy pair is 5r's and is identical under matching --
    see `_frozen_vs_refit`. And matching does not merely shrink the gap, it SHARPENS the
    dissociation; post-cue acute minus pre, per position:

                        nI       nM       nC       fI       fM       fC
        5r          +0.027  +0.132*  +0.168**  +0.034   +0.044  +0.196*
        5rm         +0.093   +0.039  +0.128*  -0.036  -0.132*  +0.158**

    Far-contralateral keeps a significant positive recoverable component once the handicap is
    removed, while FAR-MIDDLE turns significantly NEGATIVE: refitting buys less there than it bought
    before the lesion. That is "the code is degraded" as a positive finding rather than as an absent
    one, and it is invisible in the unmatched family, where far-middle is a flat +0.044.
    """
    return _frozen_vs_refit(out_dir, align, variant, wname, matched=True)
def _frozen_vs_refit(out_dir, align, variant, wname, *, matched=False):
    """5r: is the position code LOST, or present and MISREAD by the pre-stroke model?

    THE QUESTION THE FROZEN DECODER CANNOT ANSWER ALONE. A frozen decoder that fails post-stroke is
    consistent with two opposite readings -- the information is gone, or the information is there
    and the pre-stroke readout no longer points at it -- and every other figure in this set attacks
    that ambiguity indirectly (split-half reliability says the pattern is still repeatable;
    crossnobis says it moved; the confusions say where it went). This attacks it directly: refit a
    decoder WITHIN each session, on the same trials, with the same estimator and the same block
    grouping, and ask whether the position becomes decodable again.

        gap ~ 0, both low   -> the code is degraded. No model recovers it.
        gap > 0             -> the code is present and DISPLACED: only the frozen readout fails.

    THE PRE PANEL IS THE CONTROL AND IS NOT ZERO BY CONSTRUCTION. The frozen arm trains on ten
    pre-stroke sessions and the refit arm on one, so a gap exists at pre from training-set size
    alone, with no lesion involved. The post-stroke claim is the gap at that epoch MINUS the gap at
    pre, which is what the companion contrast panel plots -- reading the raw gap as the effect
    would charge the lesion for the handicap the design imposes.

    Trials are IDENTICAL between the two arms by construction (`mode="paired"` fills two columns of
    one record), so the difference is paired at the trial level and every bootstrap level inherits
    the pairing.
    """
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant,
                                      "paired_matched" if matched else "paired")
    if not per_animal:
        return None
    key = "5rm" if matched else "5r"
    what = (" (training-set MATCHED)" if matched else "")
    note_arm = (["frozen arm trained on a size-matched random subset of pre-stroke BLOCKS",
                 "the pre gap is cross-session generalisation, not training-set size"]
                if matched else
                ["each session scored by a 5-fold block-CV decoder fitted on ITSELF",
                 "the pre gap is the training-set-size handicap, not an effect"])
    made = []
    # THE MATCHED ARM DOES NOT DRAW THE REFIT-ACCURACY PAIR, because it cannot differ from 5r's.
    # `paired_matched` replaces the FROZEN model only; `_refit_pred` is called with the same X, y
    # and blocks, and `_seed_for` is keyed on (align, variant, epoch, position) rather than on the
    # figure name, so the bars, the intervals and the marks are identical by construction. Measured
    # 2026-09-11, post-cue acute-minus-pre, both families: -0.207* -0.141* -0.145* -0.339**
    # -0.363** -0.378** -- identical to three decimals AND in every mark. Drawing them twice put two
    # slides in the deck whose "(training-set MATCHED)" title promised a second analysis that did
    # not exist, and invited exactly the comparison this comment now forecloses. What matching CAN
    # move is the GAP, and it moves it a great deal -- see the second call below.
    if not matched:
        p = _position_bars(
            per_animal, out_dir, align, variant, wname,
            name=f"epoch_{key}_refit_by_position_{align}_{variant}",
            title=f"Per-position accuracy of a WITHIN-SESSION refit decoder{what}, {wname}",
            delta_name=f"epoch_{key}delta_refit_by_position_{align}_{variant}",
            delta_title=f"Change from pre-stroke in refit decoding accuracy, {wname}",
            ylabel="accuracy (refit within session)", delta_ylabel="accuracy - pre",
            stat_at=_refit_at, value_of=_refit_of, chance=CHANCE, ylim=(0.0, 1.10),
            notes=note_arm + ["pre panel is one record per pre-stroke session, not a pooled LOSO "
                              "record"])
        if p:
            made.append(p)
    q = _position_bars(
        per_animal, out_dir, align, variant, wname,
        name=f"epoch_{key}gap_frozen_vs_refit_{align}_{variant}",
        title=f"Recoverable information: refit minus frozen accuracy{what}, {wname}",
        delta_name=f"epoch_{key}gapdelta_frozen_vs_refit_{align}_{variant}",
        delta_title=f"Change from pre-stroke in the refit-minus-frozen gap{what}, {wname}",
        ylabel="refit - frozen accuracy", delta_ylabel="gap - pre gap",
        stat_at=_gap_at, value_of=_gap_of,
        # A DIFFERENCE HAS NO CHANCE LEVEL and no natural range: zero is the reference and the
        # axis autoscales, because clipping a negative gap to a [0, 1] window would hide the
        # reading that matters most -- both decoders failing together.
        chance=None, reference=0.0, ylim=None,
        notes=["paired within trial: both arms score the SAME trials"] + note_arm[1:])
    if q:
        made.append(q)
    return made or None
def _confusion_rows(per_animal, out_dir, disp, align, variant, wname):
    """5c/5d pooled: the frozen pre-stroke decoder's confusion per epoch, and the delta."""
    counts = ef.counts_by_epoch(per_animal)
    counts = {e: M for e, M in counts.items() if M is not None and np.asarray(M).sum()}
    if not counts:
        return []
    cov = ef.epoch_coverage(per_animal)["per_epoch"]
    made = []
    p = ef.confusion_row(
        counts, out_dir, name=f"epoch_5c_frozen_confusion_{align}_{variant}",
        title=f"Frozen PRE-stroke decoder, pooled across animals -- {wname}",
        coverage=cov, delta=True, chance=CHANCE, labels=_short_labels(),
        # PRE is a leave-one-out REFERENCE, not one session -- epoch_figures._panel_counts
        pre_sessions=_pre_counts(align, variant))
    if p:
        made.append(p)
    return made
def _refit_confusion_rows(out_dir, align, variant, wname):
    """5cr: the WITHIN-SESSION REFIT decoder's confusion per epoch, and its change from pre.

    Priya, 2026-09-11: "can we add the refit confusion matrices to the deck". The 5r family reduces
    the refit decoder to a per-position ACCURACY -- the diagonal -- and the whole point of the
    frozen family's confusion panel is that the diagonal is not the interesting part: WHERE the
    errors go is. 5c answers that for the frozen decoder and nothing answered it for the refit one.

    WHAT THE PAIR SEPARATES, read against 5c panel for panel:

        5c off-diagonal, 5cr diagonal restored   the code is INTACT and the pre-stroke readout is
                                                 pointing at the wrong place -- displacement
        both off-diagonal, and in the SAME cells the code itself is confusable with that neighbour;
                                                 no readout recovers it -- degradation
        5cr off-diagonal in DIFFERENT cells      the within-session structure has reorganised rather
                                                 than simply weakened

    THE TWO PANELS ARE NOT ON THE SAME FOOTING and the difference is not cosmetic: the frozen arm
    trains on ten pre-stroke sessions and this one on four fifths of one, so its PRE panel is
    already worse than 5c's with no lesion involved. Read each family against ITS OWN pre column --
    which is what the delta row beneath does -- and never a 5cr cell against a 5c cell directly.

    SESSIONS THE REFIT COULD NOT BE FITTED ON ARE ABSENT, not zero: `_refit_pred` returns None for a
    session that cannot carry a 5-fold block split, so a thin epoch here rests on fewer sessions
    than the same epoch in 5c. The panel titles carry the per-animal session counts for that reason.
    """
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant, "refit")
    if not per_animal:
        return []
    counts = ef.counts_by_epoch(per_animal)
    counts = {e: M for e, M in counts.items() if M is not None and np.asarray(M).sum()}
    if not counts:
        return []
    p = ef.confusion_row(
        counts, out_dir, name=f"epoch_5cr_refit_confusion_{align}_{variant}",
        title=(f"WITHIN-SESSION REFIT decoder, pooled across animals -- {wname}"),
        coverage=ef.epoch_coverage(per_animal)["per_epoch"], delta=True, chance=CHANCE,
        labels=_short_labels())
    return [p] if p else []
def _matrix_family(key, collector, unit, cmap, scale, stem, out_dir, align, variant, wname):
    """One matrix family's pooled epoch row, from the collector the per-animal figures already use.

    NOTHING IS RECOMPUTED: `_matrices_pattern`, `_matrices_splithalf` and `_matrices_crossnobis`
    all bottom out in `_collect_7`, keyed by "PRE" or day, which is what an epoch is defined on.
    """
    from wfield_local import grant_figures as G

    mats, _days = getattr(G, collector)(align, variant)
    if not mats:
        return None
    pooled, cov = ef.mean_matrix_by_epoch(mats)
    if not pooled:
        return None
    vmin, vmax = (scale if scale else (None, None))
    sub = ef.stats_line(_session_counts(),
                        notes=["pooled as a MEAN OVER SESSIONS: these matrices are already "
                               "reduced, so a session is the unit and cannot be re-weighted "
                               "by its trial count"])

    # THE DIAGONAL AS BARS, beside the matrix (Priya, 2026-08-28). The matrix carries the whole
    # structure -- where a position's pattern went, not only that it left -- and the diagonal
    # carries the headline: how much each position still resembles its own pre-stroke pattern.
    # One is not a summary of the other, which is why both are drawn: a diagonal that falls with a
    # flat off-diagonal is a code that faded, and the same diagonal with mass at a neighbour is a
    # code that MOVED. Same statistics as every other bar family here.
    from wfield_local.grant_figures import CONF_LABELS

    short = dict(zip(CONF_LABELS, _short_labels()))
    dpos = {short[q]: i for i, q in enumerate(CONF_LABELS)}

    def _diag(M, key):
        A = np.asarray(M, float)
        i = dpos[key]
        v = A[i, i] if i < A.shape[0] else np.nan
        return None if not np.isfinite(v) else float(v)

    dvals, dpoints = ef.scalar_by_epoch(mats, _diag, keys=_short_labels())
    if dvals:
        try:
            _scalar_figure(
                out_dir, name=f"epoch_{key}diag_{collector.strip('_')}_{align}_{variant}",
                title=f"{stem}: own-position value by epoch -- {wname}",
                ylabel=unit, keys=_short_labels(), values=dvals, points=dpoints,
                tick_labels=_minor(), groups=_groups(),
                # THE BARS INHERIT THE FAMILY'S SCALE DECISION. `MATRIX_FAMILIES` sets a fixed range
                # only where the quantity has one -- a correlation does, a crossnobis distance does
                # not -- and the matrix already honours that by passing vmin/vmax through as None.
                # The diagonal bars used to fall back to (-0.05, 1.10) instead, which is a
                # CORRELATION's bound applied to a distance: crossnobis values above 1.10 were drawn
                # clipped, bars and 95% intervals cut off at the axis (Priya, deck slides 590-592).
                # None here means autoscale, which is the same answer the matrix arrives at.
                ylim=(None if scale is None
                      else (min(-0.05, float(vmin)), float(vmax))),
                delta_name=f"epoch_{key}diagdelta_{collector.strip('_')}_{align}_{variant}",
                delta_title=f"{stem}: change from pre-stroke in the own-position value -- {wname}")
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {key}diag {align}/{variant}: {type(ex).__name__} {str(ex)[:120]}",
                  flush=True)
    return ef.matrix_row(
        pooled, out_dir, name=f"epoch_{key}_{collector.strip('_')}_{align}_{variant}",
        title=f"{stem} -- {wname}", labels=_short_labels(), cmap=cmap,
        vmin=vmin, vmax=vmax, unit=unit, coverage=cov, subtitle=sub, delta=True,
        pre_sessions=_pre_counts(align, variant))
def _fig_8g(out_dir, align, variant, wname):
    """8g pooled: per-position RDM row correlation against the pre-stroke geometry, by epoch.

    `_rdm_rows` gives ``{animal: {"PRE"|day: (per-position row r, whole r, n)}}`` -- one number per
    position per day, which is a bar figure once grouped.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    rows, _days = G._rdm_rows(align, variant)
    if not rows:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))
    idx = {short[q]: i for i, q in enumerate(CONF_LABELS)}

    def value_of(payload, key):
        try:
            per = payload[0]
            v = per.get(key) if isinstance(per, dict) else per[idx[key]]
        except Exception:                                              # noqa: BLE001
            return None
        return None if v is None else float(v)

    values, points = ef.scalar_by_epoch(rows, value_of, keys=_short_labels())
    if not values:
        return None
    return _scalar_figure(
        out_dir, name=f"epoch_8g_geometry_by_position_{align}_{variant}",
        title=f"Per-position RDM row correlation with the pre-stroke geometry -- {wname}",
        ylabel="row correlation", keys=_short_labels(), values=values, points=points,
        tick_labels=_minor(), groups=_groups(),
        # THE FULL RANGE OF A CORRELATION, not a floor chosen from the engaged arms. This was
        # (-0.2, 1.10), and on cue/stopped several chronic bars and most of the acute error bars run
        # BELOW -0.2 -- they were drawn truncated at the axis, so a bar that reaches -0.45 looked
        # identical to one that reaches -0.2 and the intervals simply ended in mid-air. A clipped
        # bar is worse than a compressed one: it misreports the value rather than the emphasis.
        # MATRIX_FAMILIES already states the rule this follows -- fix the scale only where the
        # quantity has a natural range, and a correlation does.
        ylim=(-1.05, 1.10),
        delta_name=f"epoch_8gdelta_geometry_by_position_{align}_{variant}",
        delta_title=f"Change from pre-stroke in per-position row correlation -- {wname}")
def _fig_9(out_dir, align, variant, wname):
    """9 pooled: the per-position change in pattern similarity, by epoch.

    THE ONE FAMILY THAT CARRIES ITS OWN BOOTSTRAP. Every other scalar family reads a collector and
    is reduced by the time it arrives here; `_delta_cis(..., full=True)` computes the day-minus-pre
    difference by resampling blocks within session, and returns a record per day holding the
    interval and the median for the mean diagonal and for each position separately.

    AND ITS VALUES ARE ALREADY DIFFERENCES, so there is no pre-stroke bar and no contrast against
    one -- pre minus pre is zero by construction, and drawing it would invite a reader to compare
    two real panels against a column of exact zeros. The question here is whether each epoch's
    change differs from ZERO, so the mark tests the value's own interval rather than a contrast.
    """
    from wfield_local import grant_figures as G
    from wfield_local.grant_figures import CONF_LABELS

    cis = G._delta_cis(align, variant, 10, G._mats_pattern, "9", full=True)
    if not cis:
        return None
    short = dict(zip(CONF_LABELS, _short_labels()))

    def value_of(rec, key):
        try:
            v = (rec.get("pos") or {}).get(_inv_short[key])
        except AttributeError:
            return None
        return None if not v else float(v[2])          # the median of the day's own draws

    _inv_short = {short[q]: q for q in CONF_LABELS}
    values, points = ef.scalar_by_epoch(cis, value_of, keys=_short_labels())
    if not values:
        return None

    # MARKS TEST THE VALUE AGAINST ZERO, not against a pre bar there is no room for.
    n_comp = sum(len(values[e]) for e in values)
    marks, rows = {}, {}
    for e in list(values):
        marks[e], rows[e] = {}, {}
        for k in list(values[e]):
            got = ef.scalar_value_draws(
                points, e, k, rng=np.random.default_rng(_seed_for("9", align, variant, f"{e}-{k}")),
                n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            values[e][k] = ef.with_ci(got)
            marks[e][k] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][k] = (point, float(lo), float(hi), float(clo), float(chi))

    sub = ef.stats_line(_session_counts(), n_boot=N_BOOT, notes=[
        _MEAN_NOTE,
        "values are ALREADY day-minus-pre differences, so there is no pre-stroke bar; the mark "
        "asks whether that epoch's change differs from zero"])
    made = ef.bar_row(
        values, out_dir, name=f"epoch_9_delta_trajectory_{align}_{variant}",
        title=f"Change in own-position pattern similarity from pre-stroke -- {wname}",
        subtitle=sub, marks=marks, ylabel="similarity - pre",
        positions=_short_labels(), tick_labels=_minor(), groups=_groups(), points=points,
        # AUTOSCALED, for the same reason as the crossnobis bars above: (-0.65, 0.35) was a window
        # fitted to the data as it stood when the figure was written, and a recovery trajectory that
        # moved outside it was drawn clipped rather than drawn larger. A change-from-baseline has no
        # natural range either.
        counts=_totals(_session_counts()), ylim=None)
    if any(rows.values()):
        ef.contrast_panel(
            rows, out_dir, name=f"epoch_9ci_delta_trajectory_{align}_{variant}",
            title=f"Change in own-position pattern similarity, with intervals -- {wname}",
            subtitle=sub, ylabel="similarity - pre", positions=_short_labels(),
            tick_labels=_minor(), groups=_groups(), n_comparisons=n_comp)
    return made
def _epoch_arm(align, variant):
    """`_collect_5c`'s per-animal records, with anything outside an epoch reported not dropped."""
    from wfield_local.grant_figures import _collect_5c

    per_animal, _days = _collect_5c(align, variant)
    return per_animal
def _report(tag, path):
    """A figure that returned None is an ABSENCE, and it has to say so.

    `print(f"wrote {fn(...)}")` renders a None return as "wrote None", which scans as success in a
    log nobody reads closely -- and that is exactly how the per-position accuracy figure came back
    empty three times without anyone noticing. Same class as `_draw` swallowing a plotting error so
    a broken figure reports as zero missing.
    """
    if path:
        print(f"  wrote {path}", flush=True)
    else:
        print(f"  ?? {tag}: NO FIGURE -- the renderer found no data to draw", flush=True)
