"""THE POOLED EPOCH RENDERER'S SHARED HELPERS.

Split out of `epoch_grant_figures` on 2026-09-21, alongside the five figure-family modules. These
are the helpers reached from MORE THAN ONE family -- scalar-figure layout, the session and pre
counts every panel states, the label shorteners, and the per-day seed. A helper used by one family
travels with it; one used across families has to live above all of them.

`_seed_for` is the reason several of these are shared rather than copied: a day's interval must
not depend on how many days preceded it in the run, or rendering a subset silently changes the
days that remain. One seed function, derived per day.
"""
from __future__ import annotations

import numpy as np

from wfield_local import epoch_figures as ef


def _session_counts(pre_override=None):
    """``{epoch: {animal: sessions}}`` from the epoch assignment itself.

    One definition of "how many sessions is this epoch", shared by every figure, rather than each
    one counting whatever it happens to hold.

    ``pre_override`` REPLACES THE PRE COUNT WITH WHAT THE PANEL ACTUALLY HOLDS, and only the pre
    count. The post-stroke epochs are stated from the assignment on purpose -- a family that drops
    a thin session should still say how many sessions the epoch HAS, with the drop visible in the
    block count. The pre panel is different: for the `stopped` class it is not a subset of the
    pre-stroke sessions but a small minority of them, because a well-trained pre-stroke animal
    barely quits. The first stopped render claimed "pre 44 (92:11 93:11 94:11 95:11)" over a panel
    built from FIFTEEN sessions, and the only visible sign was the block count falling from 3,688
    to 169 (Priya's stopped-class request, 2026-09-11). A count that is wrong by 3x is worse than
    no count.
    """
    from wfield_local import epochs
    out = {}
    for e, labels in epochs.labels_by_epoch().items():
        per = {}
        for lab in labels:
            an = lab.split("_")[0]
            per[an] = per.get(an, 0) + 1
        out[e] = per
    if pre_override:
        out["pre"] = dict(pre_override)
    return out
def _long_labels():
    """Full position names, title-cased: "Near Ipsi" ... "Far Contra".

    For panel TITLES, where there is room and where the two-level x axis is unavailable. Figure
    1c's six panels read "Ipsi / Middle / Contra" twice over, which is ambiguous the moment the
    Near and Far groups are not visibly bracketed together (Priya, 2026-08-28).
    """
    from wfield_local.grant_figures import CONF_LABELS
    return [x.title() for x in ef.anatomical_labels(CONF_LABELS, short=False)]
def _minor():
    """Per-tick labels for a bar figure: Ipsi / Middle / Contra."""
    from wfield_local.grant_figures import CONF_LABELS
    return ef.split_labels(CONF_LABELS)[0]
def _groups():
    """The second x level: a rule under each of Near and Far."""
    from wfield_local.grant_figures import CONF_LABELS
    return ef.split_labels(CONF_LABELS)[1]
def _short_labels():
    """Position labels for an axis: nI/nM/nC, fI/fM/fC.

    ANATOMY, NOT THE RIG -- and derived from `stroke_laterality` rather than hardcoded, so a
    right-lesioned animal raises instead of inheriting a label that would be backwards for it.
    """
    from wfield_local.grant_figures import CONF_LABELS
    return ef.anatomical_labels(CONF_LABELS)
def _totals(per_epoch):
    """``{animal: sessions across all epochs}`` -- the legend's count.

    Deliberately a different number from the subtitle's, and both are wanted: the legend says how
    many dots of a color are on the figure, the subtitle says how they split across the panels.
    """
    out = {}
    for per in per_epoch.values():
        for an, c in (per or {}).items():
            out[an] = out.get(an, 0) + c
    return out
#: Bootstrap draws per contrast. Well past where the 95% percentile stops moving; the outer
#: animal draw has only 35 distinct multisets, so more draws buy resolution on the inner levels
#: and nothing whatever on the outer one.
N_BOOT = 2000
def _seed_for(align, variant, epoch, position) -> int:
    """A stable seed, for the same reason `grant_figures._seed` exists: `hash()` is salted per
    process, so seeding a bootstrap with it gives a different interval on every render."""
    from wfield_local.grant_figures import _seed
    return _seed("epoch-contrast", align, variant, epoch, position)
def _accuracy_at(y, p, code):
    """Accuracy at one true position CODE, on raw (unnamed) arrays."""
    if code is None:
        return None
    y, p = np.asarray(y), np.asarray(p)
    m = (y == code)
    if not m.any():
        return None
    return float(np.mean(y[m] == p[m]))
def _pre_counts(align, variant):
    """Per-animal PRE-STROKE SESSION counts, for the panel labels.

    The already-reduced families build ONE pre matrix per animal by averaging that animal's
    pre-stroke sessions leave-one-out, so a panel's own coverage reads 92:1 where the subtitle says
    92:11. See .
    """
    from wfield_local import grant_figures as _G
    return _G.pre_session_counts(align, variant)
def _scalar_figure(out_dir, *, name, title, ylabel, keys, values, points, tick_labels=None,
                   groups=None, chance=None, ylim=(0.0, 1.06), delta_name=None,
                   delta_title=None, delta_ylabel=None, notes=None, session_counts=None):
    """Bar row + marks + the epoch-minus-pre companion panel, for a per-session scalar family.

    ONE PATH FOR ALL OF THEM. 8g, 10, 10b and 11 differ only in which collector fills `values` and
    `points`, so the statistics, the marks and the companion panel are written once. Four copies
    would agree today and diverge the first time one of them gained a correction.

    ``notes`` EXTENDS the standard subtitle rather than replacing it, so a family with an unusual
    method can state it without any family losing the two lines every one of them needs (what the
    mean is over, and what the bootstrap resamples).

    **IT REWRITES ``values`` IN PLACE.** Every bar gets a bootstrap interval, and the interval is
    stored back as ``values[epoch][key] = (point, lo, hi)`` -- so a caller that reads its own
    ``values`` AFTER calling this gets a tuple where it put a float. That cost an afternoon: figure
    13n read ``vals[e]["state"]`` to compute a retention and got an array, and `np.isfinite` on an
    array is an array, so `if not np.isfinite(...)` raised "truth value ambiguous". Snapshot
    anything you need afterwards BEFORE the call. (Left in place rather than fixed by copying,
    because four existing families depend on the mutation to draw their intervals.)

    ``session_counts`` OVERRIDES the epoch assignment's counts. Every family here is built on
    TRIALS, so the assignment's per-epoch session counts describe them exactly; the behavioral-
    state family is built on one-second SEGMENTS that are not trials and whose sessions are a
    different set, and a subtitle that states the assignment's counts over that panel is simply
    wrong. The same correction `_position_bars` needed for the `stopped` class.
    """
    post = [e for e in ef.PANELS if e != "pre" and values.get(e)]
    if not post or "pre" not in values:
        # NO PRE, NO CONTRAST -- and say so, rather than draw bars with no marks and let a reader
        # assume the test was run and came back null.
        return ef.bar_row(values, out_dir, name=name, title=title,
                          subtitle=ef.stats_line(
                              session_counts if session_counts is not None
                              else _session_counts(), notes=list(notes or []) + [
                              _MEAN_NOTE, "no pre-stroke arm in this collector: no contrast drawn"]),
                          ylabel=ylabel, positions=keys, tick_labels=tick_labels, groups=groups,
                          points=points, counts=_totals(_session_counts()), chance=chance,
                          ylim=ylim)
    n_comp = sum(len(values[e]) for e in post)
    # EVERY BAR GETS ITS OWN INTERVAL, including pre. Same resampling as the contrasts, so a bar
    # and the mark above it cannot come from two different schemes.
    for e in list(values):
        for k in list(values[e]):
            got = ef.scalar_value_draws(
                points, e, k, rng=np.random.default_rng(_seed_for(name, "value", e, k)),
                n_boot=N_BOOT)
            ci = ef.with_ci(got)
            if ci is not None:
                values[e][k] = ci
    marks, rows, thin_marks = {}, {}, set()
    for e in post:
        marks[e], rows[e] = {}, {}
        for k in keys:
            if k not in values[e]:
                continue
            # A MARK NEEDS TWO ANIMALS. With one, the outer bootstrap level has no variance to
            # draw on and the interval is within-animal scatter wearing a star -- see
            # `ef.contrast_animals`. The BAR and its interval still appear; only the mark is
            # withheld, and the subtitle says how many epochs were affected.
            if ef.contrast_animals(points, e, "pre", k) < ef.MIN_ANIMALS_FOR_MARK:
                # PER (epoch, key). Tracking it per EPOCH silenced a sound bar because its
                # NEIGHBOUR was thin -- 12b's acute ENGAGED arm lost its mark to the STOPPED arm.
                thin_marks.add(f"{e}/{k}")
                continue
            got = ef.scalar_contrast_draws(
                points, e, "pre", k,
                rng=np.random.default_rng(_seed_for(name, "scalar", e, k)), n_boot=N_BOOT)
            if got is None:
                continue
            point, draws = got
            marks[e][k] = ef.contrast_marks(draws, n_comparisons=n_comp)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            a = 0.05 / max(1, n_comp)
            clo, chi = np.percentile(draws, [100 * a / 2, 100 * (1 - a / 2)])
            rows[e][k] = (point, float(lo), float(hi), float(clo), float(chi))
    counts = session_counts if session_counts is not None else _session_counts()
    thin_note = ([f"NO MARK on {', '.join(sorted(thin_marks))} (epoch/bar): fewer than "
                  f"{ef.MIN_ANIMALS_FOR_MARK} animals contribute, so the animal level of the "
                  f"bootstrap has no variance and a star would assert more than one animal can "
                  f"support. The bar and its interval are still shown"] if thin_marks else [])
    sub = ef.stats_line(counts, n_boot=N_BOOT, notes=thin_note + [
        _MEAN_NOTE,
        "bootstrap: animals -> sessions. No block level: these values are one number per session, "
        "so the trial reduction already happened inside the collector"] + list(notes or []))
    made = ef.bar_row(values, out_dir, name=name, title=title, subtitle=sub, marks=marks,
                      ylabel=ylabel, positions=keys, tick_labels=tick_labels, groups=groups,
                      points=points, counts=_totals(counts), chance=chance, ylim=ylim)
    if any(rows.values()):
        ef.contrast_panel(rows, out_dir, name=delta_name or f"{name}_delta",
                          title=delta_title or f"Change from pre-stroke -- {title}",
                          subtitle=sub, ylabel=delta_ylabel or f"{ylabel} - pre",
                          positions=keys, tick_labels=tick_labels, groups=groups,
                          n_comparisons=n_comp, points=ef.session_deltas(points))
    return made
#: Stated on every already-reduced family, because it is the one thing that separates them from
#: the confusion figures: those pool by SUMMING raw counts (every trial once), these by AVERAGING
#: one value per session (every session once). Same figure shape, different weighting.
_MEAN_NOTE = ("pooled as a MEAN OVER SESSIONS: these values are already reduced per session, so a "
              "session is the unit and cannot be re-weighted by its trial count")


# ------------------------------------------------------------------ the trial set of each family
# Priya 11-09: "EVERY figure must state EXACTLY which trials were included / excluded". The arm loop of
# `epoch_grant_figures.main` activates the ARM's class (`trial_set_text.activate(variant, align)`), which is exact only
# for a family that draws that class on BOTH sides of the pre/post comparison. These return the sentence a family
# really draws, from the selection rules the collectors apply (cited per function); the driver wraps each family in
# `trial_set_text.active(None, text=...)`. Captions only: nothing here is in a cache key.
#: what the pre side of a pattern family is (`grant_kit._collect_7`: pre-stroke trials are the LICK set in every class)
PATTERN_PRE_ROLE = "pre-stroke reference + PRE panel"
#: the pre side of a frozen spout decoder outside the camera working frame (`grant_kit._frozen_frame` sel "engaged")
DECODER_PRE_ROLE = "frozen decoder training set + PRE panel (held-out pre-stroke sessions)"
#: the pre side of the within-session refit decoder (5cr) outside the camera working frame
REFIT_PRE_ROLE = "PRE panel (each pre-stroke session refit on itself)"


def _body(sentence: str) -> str:
    return sentence.split(": ", 1)[1] if ": " in sentence else sentence


def _two_sides(post_variant: str, pre_variant: str, pre_role: str) -> str:
    """POST-stroke trials of ``post_variant`` and PRE-stroke trials of ``pre_variant`` in one sentence."""
    from wfield_local import trial_set_text as tst
    return tst.union(f"post-stroke = {_body(tst.sentence(post_variant))}",
                     f"{pre_role} = {_body(tst.sentence(pre_variant))}", unit="trials")


def working_frame_on() -> bool:
    """True when the frozen spout decoder of the `working` class trains and scores on the camera WORKING rows on both
    sides (`grant_kit._working_masks`: camera gate on AND a camera working selection); else it trains on the engaged
    (lick) rows and its pre panel is the lick set (the DAQ gate)."""
    from wfield_local import trial_classes as tc
    from wfield_local import working_trials as wt
    return bool(tc.camera_gate_on() and wt.is_camera(tc.working_selection()))


def decoder_trial_text(align: str, variant: str, *, refit_only: bool = False) -> str:
    """Families acc / 5c / 5cr / 5r / 5rm / 5ro / 5rmo (`grant_confusion._collect_5c`, frozen / refit / paired /
    paired_matched): post-stroke rows = the arm's class (`_class_select`). The frozen model's training rows and the pre
    panel are the arm's class too under the camera working frame (`_frozen_frame(sel="working")`), else the ENGAGED
    (lick) rows. The within-session refit (5cr, the refit column of 5r*) is fitted and block-CV scored on the scored
    session's own rows, i.e. the same set; ``refit_only`` (5cr) = no frozen model on the figure."""
    from wfield_local import frozen_models
    from wfield_local import trial_set_text as tst
    if variant == "lick" or align == "lick" or (variant == "working" and working_frame_on()):
        out = tst.sentence(variant)
    else:
        out = _two_sides(variant, "lick", REFIT_PRE_ROLE if refit_only else DECODER_PRE_ROLE)
    if not refit_only and frozen_models.daq_trained_in_lp():
        # WIDEFIELD_LP_FROZEN=daq cross-check: the LP arm LOADS the DAQ run's model (its own pre trials)
        out += "; frozen decoder = the DAQ-TRAINED model (fitted on the DAQ arm's pre-stroke trials, DAQ licks)"
    return out


def pattern_trial_text(align: str, variant: str) -> str:
    """Families mat (6 / 7 / 8 / 8rc / 10c / 8rz), scal (8g / 9 / 10 / 10b / 11 / 11c / 11cpos), 10e, 10cs: all built on
    `grant_kit._collect_7`, whose post-stroke side is the arm's class and whose PRE-STROKE side (the reference and the
    PRE panel) is the LICK set in every class, camera gate included."""
    from wfield_local import trial_set_text as tst
    if variant == "lick":
        return tst.sentence("lick")
    return _two_sides(variant, "lick", PATTERN_PRE_ROLE)


def stopped_trial_text(align: str, family: str = "12s") -> str:
    """Families 12s / 12b (`grant_figures._collect_stopped` / `_collect_stopped_pooled`): the STOPPED trials,
    post-stroke per session and pre-stroke pooled (12s; per session, leave-one-out, 12b), each scored against the
    animal's pre-stroke ENGAGED template, built from its LICK trials (`_session_trials(..., "lick")`). The floors are
    the collectors' own (`MIN_STOPPED` / `MIN_STOPPED_REF`; `_collect_stopped_pooled`'s ``min_trials`` default)."""
    from wfield_local import grant_figures as G
    from wfield_local import trial_set_text as tst
    if family == "12b":
        n = (getattr(G._collect_stopped_pooled, "__wrapped__", G._collect_stopped_pooled).__defaults__ or (20,))[0]
        floor = f"a session enters with >= {n} stopped trials (all positions pooled)"
    else:
        floor = (f"a session x position cell enters with >= {G.MIN_STOPPED} stopped trials (>= 2 positions); the "
                 f"pre-stroke stopped control only for animals with >= {G.MIN_STOPPED_REF} pre-stroke stopped trials")
    return tst.union(f"post-stroke + pre-stroke = {_body(tst.sentence('stopped'))}; {floor}",
                     f"scored against the pre-stroke template = {_body(tst.sentence('lick'))}", unit="trials")


def _rest_text() -> str:
    from wfield_local import config
    try:
        seg = config.defaults()["segmentation"]
        r, run = seg["rest"], seg["running"]
    except Exception:                                                  # noqa: BLE001 - a caption never kills a figure
        return "REST (behavior_events rest mask), RUNNING (treadmill running bouts)"
    dock = ", spout docked (dock -> next trial start)" if r.get("docked") else ""
    lb = r.get("lick_buffer_s", ("?", "?"))
    tb = r.get("treadmill_buffer_s", ("?", "?"))
    return (f"REST (behavior_events rest mask {r.get('variant', '')}: outside trials{dock}, treadmill below "
            f"{r.get('speed_mm_s', '?')} mm/s (buffer {tb[0]} s before / {tb[1]} s after), no DAQ lick from {lb[0]} s "
            f"before to {lb[1]} s after), RUNNING "
            f"(treadmill >= {run.get('thresh_speed_mm_s', '?')} mm/s for >= {run.get('min_duration_s', '?')} s)")


def state_segment_text() -> str:
    """Family 13s (`locomotor_decoder.by_animal_day`): 1 s SEGMENTS, not trials (`locomotor_state.three_way_segments`
    with lick_mode="postcue", `locomotor_features.segment_features`)."""
    from wfield_local import locomotor_decoder as ld
    from wfield_local import locomotor_state as ls
    from wfield_local import trial_set_text as tst
    excl = ["segments where running and licking overlap",
            f"sessions with < {ld.MIN_PER_CLASS} segments of any class"]
    if ls.EXCLUDE_SESSIONS:
        excl.append("sessions " + ", ".join(ls.EXCLUDE_SESSIONS))
    return tst.events(
        f"{ls.SEGMENT_S:g} s windows ({ls.SEGMENT_BINS} bins), at most {ls.MAX_SEGMENTS_PER_PERIOD} per period, of "
        f"three mutually exclusive states: {_rest_text()}, LICKING ({ls.LICK_POSTCUE_S:g} s from each trial's first "
        f"DAQ lick within {ls.LICK_MAX_LATENCY_S:g} s of the cue)",
        keep=["every pre- and post-stroke session (no trial class gate)"], excluded=excl, unit="segments (not trials)")


def state_vs_position_text() -> str:
    """Figure 13n: the state decoder's segments AND the post-cue frozen position decoder's trials (5c, cue /
    working)."""
    return f"{state_segment_text()}  AND  position bar {decoder_trial_text('cue', 'working')}"


def evoked_trial_text() -> str:
    """Family 15e (`position_evoked_maps` <- `framemap_event_maps.run_cue` npz): EVERY cued trial at each position."""
    from wfield_local import position_evoked_maps as pem
    return (f"trials: every cued trial at each position (framemap_event_maps cue maps, field {pem.FIELD}: no "
            "engagement / camera class / lick gate), cue inside the imaging coverage with contiguous frames over the "
            "pre / post window; excluded: cues outside the imaging span or across a frame gap, cues without a "
            "position code")


def behaviour_trial_text(period: str = "camera", *, min_trials: int = 5) -> str:
    """Families 1b (``period`` "camera" = primary, "daq" = the quit-tail robustness row) and 1c ("daq": the series the
    epoch boundaries were drawn from). ``min_trials`` = the per-session per-position floor of the figure."""
    from wfield_local import trial_set_text as tst
    floor = f"a session's position counts with >= {min_trials} trials"
    if period == "camera":
        # the camera working period whatever the run's trial gate (`trial_classes.camera_period_trials` reads the video
        # classes directly); `sentence("working_period")` names it under the camera gate only
        try:
            per = tst.sentence(tst.WORKING_PERIOD)
        except ValueError:
            from wfield_local import trial_classes as tc
            per = (f"trials: all trials in the camera working period (first-{tc.end_anchor()} anchor; trial_classes "
                   f"OUT_OF_WORKING_PERIOD removed: {', '.join(tc.OUT_OF_WORKING_PERIOD)})")
        return f"{per}; hit = camera contact; {floor}; sessions without camera classes (no video) left out"
    return ("trials: behavior-pipeline ENGAGED trials (position_metrics.csv trials_engaged / trials.csv engaged: DAQ "
            f"quit-tail rule on DAQ licks); hit = DAQ hit; {floor}; excluded: the terminal quit period")
