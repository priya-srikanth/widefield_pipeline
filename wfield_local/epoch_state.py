"""POOLED EPOCH FIGURES — STATE

The stopped/working split and the behavioural-state decoder.

Split out of `epoch_grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS
FROM THE CALL GRAPH: every function here is reached from this family's entry points and from no
other. Anything shared with a sibling is in `epoch_kit`, so this imports from there and never
from `epoch_grant_figures` -- that direction would be a cycle.

Entry points, dispatched by `epoch_grant_figures.main`:
  - `_fig_12_stopped`
  - `_fig_12b_stopped_pooled`
  - `_fig_13_state`
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


def _fig_12_stopped(out_dir, align, variant, wname):
    """12: the trials the engagement gate THROWS AWAY -- does the position code survive quitting?

    Answer, measured: no, and it never did. This figure reports a NULL, which is why it sits at the
    end of the section rather than inside the argument.

    Priya, 2026-09-11: "do we already have a post-stroke vs pre-stroke 'stopped' trials pattern
    similarity analysis?" We did not. `flag_engagement` has only ever been a filter, and every
    figure in this deck is built on `~not_eng`; this is the complement.

    ``variant`` IS IGNORED AND MUST BE. The stopped set is defined by the gate, not by whether the
    animal licked -- a trial in the quit period is by construction a non-response -- so there is no
    lick/working distinction to make and drawing the same figure twice under two class labels would
    imply one. The figure renders once, on the `working` pass only.

    THREE PANELS PER EPOCH ROW, all correlated against the SAME pre-stroke ENGAGED template:

        PRE (stopped)   the CONTROL: how far stopping alone moves the pattern, with no lesion
        each epoch      the post-stroke stopped pattern against that same template

    THE CONTROL IS THE POINT. A post-stroke stopped pattern that no longer resembles the template
    is uninterpretable on its own, because a quitting animal is differently aroused, differently
    sated and differently postured whether or not it has a lesion. Only the difference between the
    two stopped columns is attributable to the lesion.

    AND THE CONTROL RESTS ON TWO ANIMALS. Post-cue, pre-stroke stopped trials number 6 (PS92), 40
    (PS93), 326 (PS94) and 495 (PS95); the first two are one session each and cannot carry a mean
    pattern. `_collect_stopped` returns None for them rather than a noisy one, and the subtitle
    names which animals the pre column rests on -- a four-animal-looking panel resting on two is
    the failure mode this whole section's per-animal counts exist to prevent.
    """
    if variant != "working":
        return None
    from wfield_local import grant_figures as G

    store, _days = G._collect_stopped(align)
    if not store:
        return None

    per_epoch, contributors = {}, {}
    pre_rows, pre_animals, pre_n, excluded = [], [], {}, []
    for an, rec in sorted(store.items()):
        ref = rec.get("WORK_REF")
        if not ref:
            continue
        n_tr, n_ss = rec.get("PRE_STOPPED_N", (0, 0))
        if rec.get("PRE_STOPPED"):
            pre_rows.append(G._corr_matrix(rec["PRE_STOPPED"], ref))
            pre_animals.append(an)
            pre_n[an] = n_ss
        else:
            excluded.append(f"{an} {n_tr}")
        for key, means in rec.items():
            if key in ("WORK_REF", "PRE_STOPPED", "PRE_STOPPED_N"):
                continue
            e = ef.epoch_of_day(an, int(key))
            if not e or e == "pre":
                continue
            per_epoch.setdefault(e, []).append(G._corr_matrix(means, ref))
            contributors.setdefault(e, {}).setdefault(an, 0)
            contributors[e][an] += 1

    mats = {}
    if pre_rows:
        mats["pre"] = G._nanmean_stack(pre_rows)
    for e, rows in per_epoch.items():
        mats[e] = G._nanmean_stack(rows)
    if len([m for m in mats.values() if m is not None]) < 2:
        return None

    cov = {e: dict(by) for e, by in contributors.items()}
    if pre_animals:
        # SESSIONS, not "1". The pre reference pools an animal's pre-stroke stopped trials into ONE
        # pattern, and reporting that as one contributing session -- which the first version did --
        # understates the baseline as badly as the stopped decoder arm's subtitle overstated it.
        cov["pre"] = dict(pre_n)
    n_pre = ", ".join(pre_animals) if pre_animals else "NONE"
    n_out = ("; excluded " + ", ".join(f"{x} trials" for x in excluded)) if excluded else ""

    # THE HEADLINE IS THE PRE PANEL, and it has to be stated or the delta row invites a reading it
    # cannot support. If the diagonal is at zero BEFORE the lesion then stopped trials carry no
    # recoverable position pattern at all, every post-stroke panel is a second near-zero number,
    # and their difference is a difference of noise however strongly the diverging colour map
    # renders it. Measured rather than asserted, so the sentence cannot outlive the result.
    _pre = mats.get("pre")
    _d = float(np.nanmean(np.diag(np.asarray(_pre, float)))) if _pre is not None else float("nan")
    verdict = ("READ THE PRE PANEL FIRST. Its diagonal is %.2f -- at zero before any lesion -- so "
               "stopped trials carry essentially NO position pattern in the first place, every "
               "post-stroke panel is a second near-zero number, and the change row below is a "
               "difference between two of them. Do not read structure into it. This is a NEGATIVE "
               "result about the quit period and it agrees with the stopped decoder arm, whose "
               "pre-stroke accuracy is 0.18-0.31 against a chance of 0.167."
               % _d) if np.isfinite(_d) and abs(_d) < 0.15 else ""
    return ef.matrix_row(
        mats, out_dir, name=f"epoch_12_stopped_pattern_{align}",
        title=(f"STOPPED trials: does the position pattern survive the animal quitting? -- "
               f"{wname}"),
        labels=_short_labels(), unit="pattern correlation", vmin=-1.0, vmax=1.0,
        coverage=cov, pre_sessions=_pre_counts(align, variant), delta=True, annotate=False,
        subtitle=("Trials inside the terminal quit period ONLY -- the set every other figure in "
                  "this section removes. Every panel is correlated against the SAME reference: "
                  "that animal's pre-stroke ENGAGED mean pattern. The pre column is the CONTROL, "
                  "pre-stroke stopped trials against that template, and it measures how far "
                  "QUITTING ALONE moves the pattern with no lesion involved; only the difference "
                  f"between it and a post-stroke column is attributable to the lesion. It rests on "
                  f"{n_pre}{n_out} -- an animal needs four of the six positions and "
                  f"{G.MIN_STOPPED_REF} pre-stroke stopped trials to define the control, because a "
                  f"well-trained pre-stroke animal barely quits and a baseline built from one "
                  f"session is noise wearing the word 'control'. Cells with fewer than "
                  f"{G.MIN_STOPPED} stopped trials at that position in that session are absent, "
                  f"not zero. {verdict}"))
def _retained(acc, chance):
    """Fraction of the ABOVE-CHANCE range a score holds: ``(acc - chance) / (1 - chance)``.

    THE ONLY HONEST WAY TO PUT A SIX-WAY AND A THREE-WAY PROBLEM ON ONE AXIS. Position decoding has
    a chance of 1/6 and the behavioural-state decoder 1/3, so their raw accuracies are not
    comparable and their raw DROPS are not either: the same absolute fall means something different
    when the floor is 0.167 than when it is 0.333. Normalising asks the one question both can
    answer -- of the performance this readout had above chance, how much survived?
    """
    if acc is None:
        return None
    if np.ndim(acc):
        # A CALLER HANDED A CI TUPLE WHERE A POINT ESTIMATE BELONGS -- almost certainly because
        # `_scalar_figure` rewrote its `values` in place (see that function). Say which mistake it
        # is; the bare numpy error is "truth value of an array is ambiguous", which names neither
        # the variable nor the cause.
        raise TypeError(f"_retained needs a scalar accuracy, got {type(acc).__name__} of "
                        f"shape {np.shape(acc)} -- did _scalar_figure already overwrite it?")
    if not np.isfinite(acc):
        return None
    return float((acc - chance) / (1.0 - chance))
def _position_accuracy_by_epoch(align, variant):
    """``{epoch: BALANCED accuracy}`` of the FROZEN position decoder, from the 5c confusion counts.

    Read off the same matrices the 5c panel draws rather than recomputed, so the two figures can
    never disagree about the number this whole comparison hinges on.

    BALANCED -- the mean of the six row recalls -- and NOT the trial-weighted `trace / total`.
    The first version used the trial-weighted form and it made the comparison in 13n unfair in a
    way that was being disclosed as a caveat instead of fixed: the STATE arm is scored with
    `balanced_accuracy_score` because its class balance moves with epoch, so scoring position the
    other way put two different estimators on one axis. It also disagreed with the 5c panel's own
    printed accuracy (0.859 against the 0.89 on the figure), which is the row-recall mean.

    IT MATTERS HERE SPECIFICALLY because the post-stroke position sets are skewed by construction --
    PS93's are 49% far_center -- so a trial-weighted accuracy is pulled toward whichever positions
    the animal still attempts. The same reason `nolick_analysis` made balanced accuracy the headline
    on 2026-08-17.
    """
    from wfield_local import grant_figures as G

    per_animal, _d = G._collect_5c(align, variant, "frozen")
    if not per_animal:
        return {}
    out = {}
    for e, M in ef.counts_by_epoch(per_animal).items():
        if M is None:
            continue
        A = np.asarray(M, float)
        rows = A.sum(1)
        ok = rows > 0
        if ok.any():
            out[e] = float(np.mean(np.diag(A)[ok] / rows[ok]))
    return out
def _state_epoch_values(store, field, keys):
    """``(values, points)`` over epochs for one field of the state-decoder records.

    THE PRE COLUMN IS EVERY LEAVE-ONE-OUT RECORD, not one per animal: taking the first would throw
    away ten of each animal's eleven pre-stroke sessions and rest the baseline on four numbers.
    """
    vals, pts = {}, {}
    for e in ef.PANELS:
        row, pt = {}, {}
        for k in keys:
            got = []
            for an, rec in sorted(store.items()):
                src = (rec.get("PRE", []) if e == "pre" else
                       [r for day, r in rec.items()
                        if day != "PRE" and ef.epoch_of_day(an, int(day)) == e])
                for r in src:
                    v = (r.get(field) if field != "per_class"
                         else (r.get("per_class") or {}).get(k))
                    if v is not None and np.isfinite(v):
                        got.append((an, float(v)))
            if got:
                row[k] = float(np.mean([v for _a, v in got]))
                pt[k] = got
        if row:
            vals[e], pts[e] = row, pt
    return vals, pts
def _state_class_share(store):
    """``{epoch: {class: share}}`` -- what fraction of each epoch's SEGMENTS each class supplies.

    MEASURED, NOT ASSERTED, and that is the whole point of this function existing. Until
    2026-09-13 three captions and one figure subtitle asserted "quiet is 3.4% of a pre session,
    15.1% acute, 0.7% chronic" as a literal. Those numbers were measured once, by hand, on the
    RETIRED reward-anchored quiet definition, and survived the REST migration unchanged because a
    hard-coded string cannot go stale loudly -- a re-render reprinted them onto a figure built from
    different segments. Priya, 2026-09-13: "fix those at the source."

    THE QUANTITY IS DELIBERATELY NOT THE OLD ONE. "% of a session" was a TIME fraction; this is the
    share of SEGMENTS, which is what a balanced-accuracy caveat is actually about -- the estimator
    sees segments, not seconds, and the cap on segments per period means the two are not
    proportional. Say "of segments" wherever it is printed.

    `score` stores raw per-class counts on every record precisely so this can be summed rather than
    averaged: a mean of per-session shares and a share computed on pooled counts are different
    numbers, and only the second describes what the pooled estimator saw.
    """
    out = {}
    for e in ef.PANELS:
        tot = {}
        for an, rec in sorted(store.items()):
            src = (rec.get("PRE", []) if e == "pre" else
                   [r for day, r in rec.items()
                    if day != "PRE" and ef.epoch_of_day(an, int(day)) == e])
            for r in src:
                for c, n in (r.get("counts") or {}).items():
                    tot[c] = tot.get(c, 0) + int(n)
        n = sum(tot.values())
        if n:
            out[e] = {c: v / n for c, v in tot.items()}
    return out
def _state_balance_line(store, cls="quiet", label="REST"):
    """The one-line class-balance caveat, with its numbers measured off ``store``.

    Returns a sentence naming this class's share of segments in each epoch it has one, so a figure
    subtitle and a deck note can carry the same measured statement instead of two copies of a
    literal that only one of them will ever remember to update.
    """
    share = _state_class_share(store)
    got = [(e, share[e][cls]) for e in ef.PANELS if e in share and cls in share[e]]
    if not got:
        return "BALANCED accuracy: the class balance moves with epoch. Full method in the notes"
    body = ", ".join(f"{v * 100:.1f}% {e}" for e, v in got)
    return (f"BALANCED accuracy: the class balance moves with epoch ({label} is {body} of "
            f"segments). Full method in the speaker notes")
def _fig_12b_stopped_pooled(out_dir, align, variant, wname):
    """12b: do STOPPED trials still look like pre-stroke cortex? POOLED over positions.

    Priya, 2026-09-11, on why the per-position stopped arm is underpowered: "I more was thinking
    about mean pattern similarity or encoder similarity for ALL stopped trials (rather than per
    position)."

    THE POWER ARGUMENT, in numbers. The quit period is short by definition, and the per-position arm
    divides it six ways. Pooled across animals the post-cue stopped sets hold 867 trials pre-stroke,
    1,984 acute, 1,935 subacute and 359 chronic; split per position the chronic cell falls to 36-75
    trials, which is not a mean pattern. This uses ALL of a session's stopped trials for ONE
    measurement, and the question it asks -- does the cortical pattern during the quit period still
    resemble the pre-stroke one -- never needed the position axis.

    TWO BARS, and the second is what makes the first readable:

      similarity   corr(this session's pooled stopped pattern, that animal's pre-stroke ENGAGED
                   pooled pattern). Both stopped columns are scored against the SAME reference, so
                   the PRE-stroke stopped bar measures how far QUITTING ALONE moves the pattern with
                   no lesion involved, and only the difference between it and a post-stroke bar is
                   attributable to the lesion.
      reliability  the split-half correlation of the session's OWN stopped trials -- the ceiling
                   this measure can reach. A similarity of 0.4 against a ceiling of 0.45 and the
                   same 0.4 against a ceiling of 0.9 are opposite results, and the per-position
                   stopped arm had no ceiling at all.

    WHY IT IS NOT AN ENCODER. "Explained variance" in the encoder families is variance ACROSS
    POSITIONS -- `_enc_terms` centres a 6 x 380 matrix down the position axis and the denominator is
    the between-position sum of squares. Pool the positions away and that denominator is zero by
    construction, so an encoder EV has nothing left to explain. Correlation against a reference
    pattern is the measure that survives pooling, which is why this is the pattern-similarity family
    and not the encoder one.
    """
    if variant != "working":
        return None
    from wfield_local import grant_figures as G

    store, _days = G._collect_stopped_pooled(align)
    if not store:
        return None

    def _corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size != b.size or not np.std(a) or not np.std(b):
            return None
        return float(np.corrcoef(a, b)[0, 1])

    VS_ENG = "vs pre-stroke ENGAGED"
    VS_STOP = "vs pre-stroke STOPPED"
    KEYS = [VS_STOP, VS_ENG]

    def _mean_of(by_sess, exclude=None):
        """Pooled pre-stroke stopped pattern, optionally leaving one session out."""
        v = [x for k, x in by_sess.items() if k != exclude]
        return np.mean(np.stack(v), axis=0) if v else None

    vals, pts, cov = {}, {}, {}
    for e in ef.PANELS:
        got = {k: [] for k in KEYS}
        for an, rec in sorted(store.items()):
            ref, by_sess = rec.get("REF"), (rec.get("PRE_BY_SESS") or {})
            if ref is None:
                continue
            if e == "pre":
                # LEAVE-ONE-SESSION-OUT on BOTH arms. A pre-stroke session scored against a pool
                # containing itself is scored partly against itself, and the whole pre bar -- which
                # is the control every post-stroke bar is read against -- would be too high.
                for mmdd, v in by_sess.items():
                    r = _corr(v, ref)
                    if r is not None:
                        got[VS_ENG].append((an, r))
                    other = _mean_of(by_sess, exclude=mmdd)
                    if other is not None:
                        r2 = _corr(v, other)
                        if r2 is not None:
                            got[VS_STOP].append((an, r2))
                continue
            full = _mean_of(by_sess)
            for day, v in rec.items():
                if day in ("REF", "PRE_BY_SESS", "PRE_STOPPED_N"):
                    continue
                if ef.epoch_of_day(an, int(day)) != e:
                    continue
                r = _corr(v, ref)
                if r is not None:
                    got[VS_ENG].append((an, r))
                if full is not None:
                    r2 = _corr(v, full)
                    if r2 is not None:
                        got[VS_STOP].append((an, r2))
        row = {k: float(np.mean([v for _a, v in got[k]])) for k in KEYS if got[k]}
        if row:
            vals[e] = row
            pts[e] = {k: got[k] for k in row}
            cov[e] = {a: sum(1 for x, _v in got[VS_ENG] if x == a)
                      for a in sorted({x for x, _v in got[VS_ENG]})}
    if len(vals) < 2:
        return None

    excl = [f"{an} {rec.get('PRE_STOPPED_N', 0)}" for an, rec in sorted(store.items())
            if not (rec.get("PRE_BY_SESS") or {})]
    has = sorted(an for an, rec in store.items() if (rec.get("PRE_BY_SESS") or {}))
    return _scalar_figure(
        out_dir, name=f"epoch_12b_stopped_pooled_similarity_{align}",
        title=("STOPPED trials, POOLED over positions: does the quit-period pattern still look "
               f"like pre-stroke cortex? -- {wname}"),
        ylabel="correlation with that pre-stroke reference",
        keys=KEYS, values=vals, points=pts,
        # SHORT, because two bar groups leave a narrow axis and "vs pre-stroke STOPPED" overran
        # into its neighbour. The full reference is named in the notes below, where there is room.
        tick_labels=["STOPPED", "ENGAGED"], ylim=(-0.4, 1.05),
        session_counts=cov,
        # TWO LINES. The method is in the docstring and the speaker note; a fourteen-line block
        # left the axes a sixth of the figure.
        notes=["a session's stopped trials pooled into ONE mean pattern, no position split. "
               "STOPPED = vs pre-stroke stopped (state-matched); ENGAGED = vs pre-stroke engaged, "
               "whose pre bar is the 'quitting alone' control. Pre is leave-one-session-out",
               "867 stopped trials pre, 1,984 acute, 1,935 subacute, 359 chronic"]
        + ([f"THE LEFT BAR RESTS ON {', '.join(has)} ONLY: "
            + ", ".join(f"{x} pre-stroke stopped trials" for x in excl)
            + " is too few to build a reference from, and a well-trained pre-stroke animal barely "
              "quits, so pooling cannot fix it"] if excl else []),
        delta_name=f"epoch_12bdelta_stopped_pooled_similarity_{align}",
        delta_title="Pooled stopped-trial similarity, change from pre-stroke")
def _fig_13_state(out_dir, align, variant, wname):
    """13: DOES EVERYTHING DEGRADE, OR ONLY THE TARGET? The frozen behavioural-state decoder.

    Priya, 2026-09-11: "I'm more looking for evidence that not *all* decoding/encoding degrades
    post stroke, with running as an example."

    THE LESION IS VENTROLATERAL STRIATAL. No cortex is damaged anywhere in the field of view, the
    imaging window is the same one, the LocaNMF basis is the same. So if the frozen pre-stroke
    POSITION decoder collapses while a frozen pre-stroke BEHAVIOURAL-STATE decoder built from the
    identical features does not, the target deficit is SPECIFIC -- and the generic explanations a
    reader reaches for first (window clouding, haemodynamic drift, arousal, basis drift, "a lesion
    was made and everything got worse") all fail at once, because every one of them would degrade
    this readout too.

    IT HAD TO BE THE FROZEN ARM, which is a measurement and not a preference. Refit WITHIN a
    session, running-vs-quiet decodes at AUROC 0.99-1.00 and the three-way problem at macro-AUROC
    0.98-1.00. A ceiling cannot demonstrate preservation; a reader sees "the task was too easy to
    fail" and is right. The position claim rests on a frozen pre-stroke model failing on
    post-stroke data, so the control must be the same object carrying the same cross-session
    generalisation burden.

    METHOD IN FULL, because nothing else in this deck is built this way:

      UNIT      a ONE-SECOND window, not a trial. Trial-level labelling gives ~17 running trials
                per session, which decodes nothing; tiling the bouts gives 22,986 running, 28,324
                REST and 38,270 licking one-second segments over the 91 sessions this figure uses
                (inside imaging coverage, PS92 8/12 excluded; `scripts/rest_migration/
                state_time_bins.py`). The length is set by REST and not chosen: the 2 s window
                every other family here uses fits 19.4% of rest bouts while 1 s fits 84.6%.

                RE-MEASURED 2026-09-13 ON THE REST DEFINITION. The previous figures -- 33,060
                running, 41,549 "quiet", 1.10 s median, 17% / 58% -- were measured on the RETIRED
                reward-anchored quiet definition and are not comparable: that definition excluded
                8 s after every reward, which in this task's 3.5 s response window removed most of
                each inter-trial interval AND anchored the category on the animal's PERFORMANCE.
                The 1 s window survived the re-derivation on its own merits (84.6% against the 58%
                that chose it); see docs/REST_BASELINE_MIGRATION.md.
      FEATURES  four 0.25 s sub-bins x 95 LocaNMF components = 380 columns, the SAME width as the
                trial-aligned arms and on the SAME joint basis. No per-segment baseline: a segment
                inside a running bout has no "before" that is not also running.
      CLASSES   REST / running / licking, MUTUALLY EXCLUSIVE per Priya's rule -- running only if
                not also licking, licking only if not also running, rest only inside a
                `behavior_events` REST period: between trials, not running and not licking, from
                cue + response_window + 0.5 s to the next `trial_start`. Overlapping segments are
                DROPPED and counted, never assigned.
      LICKING   anchored at the trial's FIRST POST-CUE LICK (`lick_mode="postcue"`, the default
                since 2026-09-12), which is the same anchor every position decoder in this deck
                uses, so the licking class and the position trials observe ONE event rather than
                two that share a word. The window runs 1 s from that anchor, matching the other
                two classes so no decoder can separate them on window length. Running and rest are
                tiled, at most 8 segments per period so no single long period dominates.
      MODEL     multinomial logistic on standardised features, frozen on ALL pre-stroke segments.
                The pre column is leave-one-session-out, as everywhere else here.
      SCORE     BALANCED accuracy against a chance of 1/3, never raw: quiet runs 3.4% of a
                pre-stroke session, 15.1% acutely and 0.7% chronically, and raw accuracy under a
                base rate that moves that much is not comparable across the epochs being compared.
      EXCLUDED  PS92 8/12, whose longest "running bout" is 2,441 s -- 41 minutes, 29% of the
                session, against a cohort maximum of 54 s. That is the crash+concat discontinuity
                (`docs/EXPERIMENT_ERRORS.md`) read as sustained locomotion.

    TWO CONFOUNDS CHECKED BEFORE THIS WAS DRAWN. Session TIME alone separates the classes at AUROC
    0.165-0.752, near chance, and restricting to the range where the classes overlap in time leaves
    the cortical score unchanged, so it is reading cortex rather than drift. And the animals RUN
    MORE after the lesion (5.1% of session acutely against 3.1% pre-stroke), so the state arm is
    not rescued by having more data at baseline than afterwards.

    THE LIMIT, stated here because it is easy to miss: licking windows are locked to a behavioural
    TRANSITION while running and quiet are sampled from inside sustained STATES, so a decoder could
    separate them partly on transient-versus-sustained rather than on which behaviour it is. This
    answers "does cortex still distinguish behavioural state at all", which is what the control
    needs; it is not a clean three-way contrast of matched epochs.
    """
    # ONE ALIGNMENT ONLY. These segments are not trials and have no cue to align to, so there is no
    # pre-cue/post-cue/post-lick distinction to make -- rendering the same figure under three arm
    # labels would imply three analyses where there is one.
    if variant != "working" or align != "cue":
        return None
    from wfield_local import locomotor_decoder as ld

    store, _days = ld.by_animal_day()
    if not store:
        return None

    vals, pts = _state_epoch_values(store, "balacc", ["state"])
    if not vals:
        return None
    # SNAPSHOTTED BEFORE `_scalar_figure` TOUCHES IT: that helper stores each bar's bootstrap
    # interval back into `values`, so these floats become (point, lo, hi) tuples the moment the
    # first figure is drawn. 13n needs the plain numbers.
    balacc = {e: float(row["state"]) for e, row in vals.items() if "state" in row}

    # THREE LINES ON THE CANVAS, NOT EIGHT. The full method is in this function's docstring and in
    # the deck speaker notes, which is where a reader who wants it goes; repeating it here wrapped
    # to thirteen lines and left the axes a fifth of the figure's height, so the one thing the
    # figure exists to show was the smallest thing on it. Keep what a reader CANNOT infer from the
    # title -- the unit, the class rule, and the fact that the model is frozen.
    NOTES = ["unit is a 1 s SEGMENT, not a trial: 4 x 0.25 s bins x 95 components, joint basis",
             "rest / running / licking, mutually exclusive; frozen on ALL pre-stroke segments, "
             "pre column leave-one-session-out",
             _state_balance_line(store)]
    counts = {e: {a: sum(1 for x, _v in pts[e]["state"] if x == a)
                  for a in sorted({x for x, _v in pts[e]["state"]})} for e in pts}
    made = []
    p = _scalar_figure(
        out_dir, name="epoch_13_state_decoder_cue",
        # NOT "preserved". The acute bar carries ** and chronic *, so the state decoder's fall IS
        # detectable -- it is 0.945 -> 0.875, small but not nothing. The claim this figure supports
        # is that it falls FAR LESS than position does, which is what 13n quantifies; a title
        # saying "preserved" would be contradicted by the marks on its own bars.
        title=("Does EVERYTHING degrade? Frozen pre-stroke BEHAVIOURAL-STATE decoder "
               "(quiet / running / licking), spout-position agnostic"),
        # `bar_row` APPENDS THE CHANCE LEVEL ITSELF, so naming it here too rendered
        # "balanced accuracy (chance 1/3) (chance 0.33)" down the side of the axes.
        ylabel="balanced accuracy", keys=["state"], values=vals, points=pts,
        tick_labels=["frozen state\ndecoder"], ylim=(0.0, 1.05), chance=1.0 / 3.0,
        notes=NOTES, session_counts=counts,
        delta_name="epoch_13delta_state_decoder_cue",
        delta_title="Frozen state decoder, change from pre-stroke")
    if p:
        made.append(p)

    pos = _position_accuracy_by_epoch(align, variant)
    if pos:
        cvals = {}
        for e in ef.PANELS:
            row = {}
            r_pos = _retained(pos.get(e), 1.0 / 6.0)
            if r_pos is not None:
                row["position (6-way)"] = r_pos
            if e in balacc:
                r_st = _retained(balacc[e], 1.0 / 3.0)
                if r_st is not None:
                    row["state (3-way)"] = r_st
            if row:
                cvals[e] = row
        if cvals:
            # DRAWN DIRECTLY, NOT THROUGH `_scalar_figure`, and this is not a shortcut. That
            # helper bootstraps every bar from its per-SESSION points and marks the contrasts; this
            # figure has no per-session points by construction -- each bar is ONE pooled number
            # derived from a pooled accuracy, and the position bar is not even per-session in
            # origin (it is the trace of a summed confusion matrix). Feeding it empty point lists
            # produced a numpy truth-value error, which was the right failure: the honest fix is
            # not to fake points but to draw the bars with NO marks and say in the subtitle that
            # there are none, rather than to emit error bars the data cannot support.
            q = ef.bar_row(
                cvals, out_dir, name="epoch_13n_state_vs_position_cue",
                title=("Of the performance each readout had ABOVE CHANCE, how much survived? "
                       "-- frozen decoders, post-cue"),
                ylabel="fraction of above-chance performance retained",
                positions=["position (6-way)", "state (3-way)"],
                tick_labels=["position\n(6-way)", "state\n(3-way)"], ylim=(0.0, 1.05),
                # THE COUNTS ARE THE STATE ARM'S, and they are the honest ones to print: the
                # position bar pools the same sessions minus PS92 8/12, which only this arm
                # excludes. Passing {} printed "N=0 animals, n=0 sessions" over a figure built
                # from eighty-nine.
                subtitle=ef.stats_line(counts, notes=[
                    "(accuracy - chance) / (1 - chance). Chance is 1/6 for the six-way position "
                    "decoder and 1/3 for the three-way state decoder, so raw accuracies -- and raw "
                    "DROPS -- are not comparable",
                    "BOTH BARS ARE BALANCED ACCURACY: mean of the row recalls for position, "
                    "sklearn balanced_accuracy_score for state. Both frozen on pre-stroke, pre "
                    "column leave-one-session-out",
                    "WHY NOT TRIAL-WEIGHTED for position (Priya asked): because THIS figure divides "
                    "by (1 - chance), and a trial-weighted chance level is not 1/6 -- it moves with "
                    "the trial mix, and the post-stroke mix is skewed by construction (PS93's "
                    "trials are 49% far_center, where always-guess-far_center scores 0.490). "
                    "Balanced accuracy has a null of exactly 1/6 however skewed either side is. The "
                    "trial-weighted accuracies are 0.859 / 0.525 / 0.749 / 0.833 -- they barely "
                    "differ here, so nothing in the reading turns on it; the 5c panel prints them",
                    "position 0.86 -> 0.43 acutely, losing 50%; state 0.92 -> 0.81, losing 11%; "
                    "RUNNING alone 0.98 -> 0.95, losing 3%",
                    "NO INTERVALS AND NO MARKS: each bar is one pooled number. The per-session "
                    "distribution is on the figures either side of this one"]))
            if q:
                made.append(q)

    # ---------------------------------------------------------------- confusion, per epoch
    # Priya, 2026-09-11: "can we add confusion matrices for the state decoder?" The per-class panel
    # gives the DIAGONAL -- how often each class is recalled -- and says nothing about where the
    # errors go, which is exactly the argument 5c makes for the position decoder. A quiet segment
    # misread as LICKING is a different failure from one misread as RUNNING: the first says the
    # post-stroke immobile animal looks task-engaged to the readout, the second that it looks like
    # it is moving.
    from wfield_local import locomotor_state as _ls

    conf = {}
    for e in ef.PANELS:
        acc = None
        for an, rec in sorted(store.items()):
            src = (rec.get("PRE", []) if e == "pre" else
                   [r for day, r in rec.items()
                    if day != "PRE" and ef.epoch_of_day(an, int(day)) == e])
            for r in src:
                M = r.get("confusion")
                if M is None:
                    continue
                acc = M.copy() if acc is None else acc + M
        if acc is not None and acc.sum():
            conf[e] = acc
    if len(conf) > 1:
        c = ef.confusion_row(
            conf, out_dir, name="epoch_13c_state_confusion_cue",
            title=("Frozen BEHAVIOURAL-STATE decoder, confusion by epoch -- "
                   "where do the errors go?"),
            coverage={e: dict(counts.get(e, {})) for e in conf}, delta=True,
            chance=1.0 / 3.0, labels=list(_ls.THREE_WAY))
        if c:
            made.append(c)

    CLS = ["quiet", "running", "licking"]
    pvals, ppts = _state_epoch_values(store, "per_class", CLS)
    if pvals:
        # SNAPSHOT BEFORE `_scalar_figure` REWRITES `pvals` into (point, lo, hi) tuples -- the same
        # trap `balacc` is snapshotted for above. Building the note text afterwards would format a
        # tuple into the caption.
        _rec = {c: [(e, float(pvals[e][c])) for e in ef.PANELS
                    if e in pvals and c in pvals[e]] for c in CLS}

        def _line(c):
            return " / ".join(f"{v:.2f}" for _e, v in _rec[c]) or "--"

        # MEASURED, NOT ASSERTED. These three lines used to be literals: "0.98 / 0.95 / 0.94 / 0.97"
        # for running and "0.86 -> 0.70 acutely" for rest. By the 2026-09-12 render the figure held
        # 0.99/0.96/0.89/0.98 and 0.97 -> 0.84 -- the caption had drifted off its own bars and
        # overstated the fall it was pointing at. A re-render moves both together now.
        _q = dict(_rec["quiet"])
        _qmove = (f"({_q.get('pre', float('nan')):.2f} pre -> "
                  f"{_q.get('acute', float('nan')):.2f} acutely)" if "pre" in _q else "")
        r = _scalar_figure(
            out_dir, name="epoch_13pos_state_decoder_by_class_cue",
            title="Frozen state decoder, RECALL PER CLASS -- which behavioural state changed?",
            ylabel="recall", keys=CLS, values=pvals, points=ppts, tick_labels=CLS,
            ylim=(0.0, 1.05), session_counts=counts, notes=[
                NOTES[0],
                f"RUNNING IS THE CLEAN EXAMPLE: {_line('running')}, flat at every epoch "
                f"(pre / acute / subacute / chronic); LICKING {_line('licking')}",
                f"REST IS THE ONE THAT MOVES {_qmove} -- and it is also a much larger share of a "
                f"post-stroke session, so a post-stroke animal sitting still may be in a genuinely "
                f"different state. A finding about immobility, not a failed control",
                _state_balance_line(store)],
            delta_name="epoch_13posdelta_state_decoder_by_class_cue",
            delta_title="State decoder recall per class, change from pre-stroke")
        if r:
            made.append(r)
    return made or None
