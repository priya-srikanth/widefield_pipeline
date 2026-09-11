"""Per-trial LOCOMOTOR STATE: running, licking, or quiet -- as three MUTUALLY EXCLUSIVE classes.

Priya, 2026-09-11: "let's add a running vs quiet vs licking decoder/encoder analysis", and then,
asked how the three should relate: "mutually exclusive groups, so running only if not also licking,
licking only if not also running, and quiet can be during ITI only".

WHY THE EXCLUSIVITY RULE HAD TO BE STATED. The three names do not describe one partition on their
own. Licking is a TASK EVENT and running is a LOCOMOTOR STATE, so a trial can be both and most
licking trials are; and `quiet` as `behavior_events` defines it is already *slow treadmill AND
not-near-lick/reward, buffered*, so no licking trial can ever be quiet. Three overlapping sets drawn
as three bars would invite a reading none of them supports. The rule above makes them a partition:

    LICKING   a lick in the window, and NO running bout overlapping it
    RUNNING   a running bout overlapping the window, and NO lick in it
    QUIET     neither, and the window lies inside a `behavior_events` quiet period
    (dropped)  a lick AND running together -- the overlap the rule excludes, counted and reported

THE CLOCK IS ALREADY SHARED, which is the only reason this is cheap. `_trial_features` resolves each
cue to a DAQ SAMPLE (`csmp`) and `behavior_events` stores `running_starts/stops`,
`quiet_starts/stops` and `lick_onsets` as DAQ sample indices at the same `fs`. No frame map, no
clock conversion, and nothing cached needs its `CACHE_VERSION` bumped: this reads the same DAQ
events `_trial_features` reads and joins on an index both sides already have.

THE WINDOW IS THE ANALYSIS WINDOW, not the trial. A trial is several seconds long and an animal can
start running inside it; asking "was it running during the trial" of a 6 s trial and of a 2 s
analysis window are different questions, and the one the decoder cares about is the window its
features come from.
"""
from __future__ import annotations

import numpy as np

#: Classes, in the order figures should draw them.
STATES = ("licking", "running", "quiet")

#: The overlap class the exclusivity rule discards: a lick AND a running bout in the same window.
#: Never silently -- `classify` returns its count so callers can report what the rule cost.
DROPPED = "lick+run"


def _in_any_bout(t0, t1, starts, stops):
    """Does [t0, t1) overlap any [start, stop) bout? Vectorised over bouts, looped over trials."""
    if not len(starts):
        return np.zeros(len(t0), bool)
    starts = np.asarray(starts, np.int64)
    stops = np.asarray(stops, np.int64)
    # A bout overlaps the window iff it starts before the window ends AND ends after it begins.
    # `searchsorted` on the (sorted) starts bounds the candidates; bouts are few, so the simple
    # form is fast enough and is obviously correct, which matters more here.
    out = np.zeros(len(t0), bool)
    for i, (a, b) in enumerate(zip(t0, t1)):
        out[i] = bool(np.any((starts < b) & (stops > a)))
    return out


def _fully_inside(t0, t1, starts, stops):
    """Is [t0, t1) CONTAINED in some bout? Stricter than overlap, and right for QUIET.

    Quiet is the class that asserts an ABSENCE -- no movement, no licking, buffered away from both.
    A window that merely touches a quiet period spends part of itself outside one, which is the
    thing quiet is defined to exclude. Running and licking assert a PRESENCE and take overlap.
    """
    if not len(starts):
        return np.zeros(len(t0), bool)
    starts = np.asarray(starts, np.int64)
    stops = np.asarray(stops, np.int64)
    out = np.zeros(len(t0), bool)
    for i, (a, b) in enumerate(zip(t0, t1)):
        out[i] = bool(np.any((starts <= a) & (stops >= b)))
    return out


def classify(cue_samples, window, events, lick_onsets=None):
    """``(labels, counts)`` -- one of STATES or None per trial, plus a count per class.

    ``cue_samples``  DAQ sample of each trial's alignment point, one per trial, in trial order.
    ``window``       ``(pre, post)`` in SAMPLES relative to that point; the analysis window.
    ``events``       a loaded `behavior_events` dict.
    ``lick_onsets``  override, for the rare caller with its own lick identity; defaults to the
                     canonical `events["lick_onsets"]`, which is the whole point of that file.

    A trial is None when it satisfies none of the three -- not quiet, not running, no lick, e.g. a
    window in a slow but unbuffered stretch next to a reward. Those are NOT forced into quiet: the
    buffer is what makes quiet mean something, and widening it here would be redefining the class at
    the call site.
    """
    c = np.asarray(cue_samples, np.int64)
    pre, post = int(window[0]), int(window[1])
    t0, t1 = c - pre, c + post

    licks = np.asarray(events.get("lick_onsets") if lick_onsets is None else lick_onsets, np.int64)
    has_lick = np.zeros(len(c), bool)
    if len(licks):
        lo = np.searchsorted(np.sort(licks), t0, "left")
        hi = np.searchsorted(np.sort(licks), t1, "left")
        has_lick = hi > lo

    running = _in_any_bout(t0, t1, events.get("running_starts", []), events.get("running_stops", []))
    quiet = _fully_inside(t0, t1, events.get("quiet_starts", []), events.get("quiet_stops", []))

    labels = np.full(len(c), None, dtype=object)
    both = has_lick & running
    labels[has_lick & ~running] = "licking"
    labels[running & ~has_lick] = "running"
    # QUIET LAST AND ONLY WHERE NOTHING ELSE APPLIES. `behavior_events` already buffers quiet away
    # from licks and rewards, so this overlap should be empty; asserting the ordering rather than
    # trusting it costs nothing and means a change to the buffer cannot silently create a trial
    # that is both quiet and licking.
    labels[quiet & ~has_lick & ~running] = "quiet"
    counts = {s: int((labels == s).sum()) for s in STATES}
    counts[DROPPED] = int(both.sum())
    counts["unclassified"] = int(sum(1 for x in labels if x is None)) - counts[DROPPED]
    return labels, counts


# ---------------------------------------------------------------------------------------------
# SEGMENTS: the unit of the running-vs-quiet analysis, and it is NOT the trial.
#
# Priya, 2026-09-11: "we don't need position information for this at all -- spout position
# agnostic. I'm more looking for evidence that not *all* decoding/encoding degrades post-stroke,
# with running as an example."
#
# That makes this a SPECIFICITY CONTROL, and it changes the unit. A trial-level split gives 1,795
# running and 3,151 quiet trials across 107 sessions -- ~17 running trials per session, which
# decodes nothing. Tiling the bouts themselves gives 33,060 and 41,549 one-second segments.
#
# WHY ONE SECOND, MEASURED RATHER THAN CHOSEN. Running bouts have a median of 3.73 s and a floor of
# 2.0 s (`segmentation.running.min_duration_s`), so any window fits them. QUIET PERIODS HAVE A
# MEDIAN OF 1.10 s and a floor of 0.5 s, and that is what sets the window: a 2 s window -- matching
# every other family here -- fits only 17% of quiet periods, while 1 s fits 58%. Four 0.25 s bins in
# that second give 95 x 4 = 380 feature columns, the SAME width as the 2 s x 0.5 s trial arms, so
# the two are comparable in size even though a model cannot be transferred between them.
#
# WHY THE CAP. Segments from one bout are not independent, and the distribution is skewed: the top
# 1% of running bouts supply 12.7% of all segments and the top 1% of QUIET periods supply 36.6%.
# Capping bounds any single period's vote. It does not make the segments independent -- only
# clustering the bootstrap by PERIOD does that, which is the caller's job and is why `period_id`
# comes back alongside.
# ---------------------------------------------------------------------------------------------

#: Analysis window per segment, seconds. Set by the QUIET distribution, not by preference.
SEGMENT_S = 1.0

#: Sub-bins inside it. Four, to match the 380-column width of the trial-aligned arms.
SEGMENT_BINS = 4

#: Most segments any one bout or quiet period may contribute.
MAX_SEGMENTS_PER_PERIOD = 8

#: Sessions whose treadmill trace cannot be segmented and must not contribute.
#:
#: PS92 8/12 is the crash+concat session (`docs/EXPERIMENT_ERRORS.md`): acquisition died after
#: ~9.6 min and the remainder was joined with `concat_split_session`. Its longest "running bout" is
#: 2,441 s -- 41 minutes, 29% of the session -- against a cohort maximum of 54 s everywhere else.
#: That is the concatenation discontinuity read as sustained locomotion. Uncapped it would supply
#: 2,441 segments, ~7% of the entire running class, from one artefact.
EXCLUDE_SESSIONS = ("PS92_0812",)


def segments(starts, stops, fs, *, seg_s=SEGMENT_S, cap=MAX_SEGMENTS_PER_PERIOD,
             onset_anchored=False):
    """``(seg_start_samples, period_id)`` -- non-overlapping windows tiled inside each period.

    Tiling starts at each period's START rather than centring, so a long bout's segments are
    deterministic and a re-run cannot shift them; and it drops the remainder rather than stretching
    the last window, because a 0.4 s window binned into four is not the same feature as a 1 s one.

    ``onset_anchored`` KEEPS PERIODS SHORTER THAN THE WINDOW, by emitting one window from the
    period's onset and letting it run past the end. That is right for LICKING and wrong for the
    other two, and the difference is not a convenience:

        licking   an EVENT with a defined onset. Lick bouts have a median duration of 0.37 s, so a
                  1 s window tiled strictly inside them would keep 22% of 101,018 bouts and bias
                  the class toward sustained licking -- the same failure a 2 s window inflicts on
                  quiet. The question is whether cortex reports the event, and the second after its
                  onset is exactly where that answer lives.
        running   a sustained STATE with no privileged moment, and a 2.0 s floor by construction
                  (`segmentation.running.min_duration_s`), so every bout tiles cleanly.
        quiet     a sustained state asserting an ABSENCE. A window running past its end would spend
                  part of itself in the movement or licking the period was buffered away from,
                  which is the one thing the class must not contain.

    THE ASYMMETRY IS REAL AND MUST BE STATED WHEREVER THIS IS PLOTTED: licking windows are locked to
    a behavioural transition while the other two are sampled from within sustained states, so a
    decoder could in principle separate them on transient-versus-sustained rather than on which
    behaviour it is. That is a limit on the interpretation, not a defect -- the control asks whether
    cortex still distinguishes behavioural states at all.
    """
    n = round(seg_s * fs)
    out_s, out_p = [], []
    for pid, (a, b) in enumerate(zip(np.asarray(starts, np.int64), np.asarray(stops, np.int64))):
        k = min(int((b - a) // n), int(cap))
        if onset_anchored and k == 0:
            k = 1
        for j in range(k):
            out_s.append(a + j * n)
            out_p.append(pid)
    return np.asarray(out_s, np.int64), np.asarray(out_p, np.int64)


def running_and_quiet_segments(events, *, seg_s=SEGMENT_S, cap=MAX_SEGMENTS_PER_PERIOD):
    """``(sample, label, period_id)`` for every usable segment in one session.

    `period_id` is made unique ACROSS the two classes, so a caller grouping by it cannot merge a
    running bout with a quiet period that happened to be numbered the same -- the identical failure
    `_pooled_bundle` guards when it offsets block ids by session.
    """
    fs = float(events.get("fs", 5000.0))
    rs, rp = events.get("running_starts", []), events.get("running_stops", [])
    qs, qp = events.get("quiet_starts", []), events.get("quiet_stops", [])
    s_r, p_r = segments(rs, rp, fs, seg_s=seg_s, cap=cap)
    s_q, p_q = segments(qs, qp, fs, seg_s=seg_s, cap=cap)
    sample = np.concatenate([s_r, s_q])
    label = np.array(["running"] * len(s_r) + ["quiet"] * len(s_q), dtype=object)
    period = np.concatenate([p_r, p_q + (p_r.max() + 1 if len(p_r) else 0) + 1_000_000])
    order = np.argsort(sample, kind="stable")
    return sample[order], label[order], period[order]


#: The three-way label set. Priya, 2026-09-11: "should we try quiet vs running vs licking (all
#: positions) frozen decoder?" -- yes, and for a reason the binary version exposed: running vs quiet
#: decodes at AUROC 0.99-1.00 within session, which is a CEILING, and a ceiling cannot demonstrate
#: preservation. Three classes at a chance of 1/3 have room to fall, and the problem is then
#: structurally parallel to the six-way position decoder: same estimator, same frozen-model logic,
#: a different label.
THREE_WAY = ("quiet", "running", "licking")


def lick_periods(events, max_ili_s=0.3, min_bout_licks=2):
    """``(starts, stops)`` in DAQ samples for lick BOUTS, from the canonical lick onsets.

    Reuses `spout_behavior.segment_bouts` -- the same rule, the same parameters out of
    `defaults.yaml behavior.licks` -- rather than a second bout definition, because two analyses
    disagreeing about what a lick bout is would make their trial counts incomparable for no reason.
    """
    from wfield_local.spout_behavior import segment_bouts

    fs = float(events.get("fs", 5000.0))
    on = np.asarray(events.get("lick_onsets", []), np.int64)
    if not len(on):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    bouts = segment_bouts(np.sort(on) / fs, max_ili_s, min_bout_licks)
    if not bouts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = np.array([b[0] for b in bouts], float) * fs
    b = np.array([b[1] for b in bouts], float) * fs
    return a.astype(np.int64), b.astype(np.int64)


def three_way_segments(events, *, seg_s=SEGMENT_S, cap=MAX_SEGMENTS_PER_PERIOD,
                       max_ili_s=0.3, min_bout_licks=2):
    """``(sample, label, period_id)`` over quiet / running / licking, MUTUALLY EXCLUSIVE.

    Priya's rule, 2026-09-11: "running only if not also licking, licking only if not also running,
    and quiet can be during ITI only". `behavior_events` already buffers quiet away from licks and
    rewards, so quiet cannot collide with licking by construction; what needs enforcing is the
    running/licking overlap, and it is enforced by DROPPING those segments rather than assigning
    them, since assigning them would decide by tie-break what the rule says is undecidable.

    A LICK BOUT IS SHORT AND A LICKING SEGMENT IS THEREFORE MOSTLY ONE BOUT, which is the intended
    unit: the question is whether cortex still reports "this animal is licking", not how long it
    licked for. The same cap applies to all three classes so no class can be built from a handful of
    very long periods -- the concentration that makes the QUIET class dangerous untiled, where the
    top 5% of periods supply 70% of the segments.

    THE LICKING CLASS IS CUE-LOCKED AND THE OTHER TWO ARE NOT. Licking happens on trials, so a
    licking window sits a fixed distance after a cue while running and quiet windows fall anywhere.
    A decoder separating them is therefore reading "licking state OR its cue-locked context", and
    that is a limit on the interpretation, not a bug -- the control asks whether cortex still
    distinguishes behavioural states at all, and it does not need the three to be matched in time
    to answer that. Say it rather than let a reader assume otherwise.
    """
    fs = float(events.get("fs", 5000.0))
    ls_, lp = lick_periods(events, max_ili_s, min_bout_licks)
    rs, rp = np.asarray(events.get("running_starts", []), np.int64), \
        np.asarray(events.get("running_stops", []), np.int64)
    qs, qp = np.asarray(events.get("quiet_starts", []), np.int64), \
        np.asarray(events.get("quiet_stops", []), np.int64)

    out_s, out_l, out_p, base = [], [], [], 0
    for name, (a, b) in (("quiet", (qs, qp)), ("running", (rs, rp)), ("licking", (ls_, lp))):
        smp, pid = segments(a, b, fs, seg_s=seg_s, cap=cap, onset_anchored=(name == "licking"))
        out_s.append(smp)
        out_l.append(np.array([name] * len(smp), dtype=object))
        # UNIQUE ACROSS CLASSES, so a caller grouping by period cannot merge a running bout with a
        # quiet period that happens to carry the same index.
        out_p.append(pid + base)
        base += (int(pid.max()) + 1 if len(pid) else 0) + 1_000_000
    sample = np.concatenate(out_s) if out_s else np.zeros(0, np.int64)
    label = np.concatenate(out_l) if out_l else np.zeros(0, object)
    period = np.concatenate(out_p) if out_p else np.zeros(0, np.int64)

    # THE EXCLUSIVITY RULE, applied to the SEGMENTS rather than to the period edges: a running bout
    # and a lick bout can overlap partially, so dropping whole periods would discard clean segments
    # at their ends.
    n = round(seg_s * fs)
    drop = np.zeros(len(sample), bool)
    if len(ls_) and len(rs):
        for i, (t0, lab) in enumerate(zip(sample, label)):
            t1 = t0 + n
            if lab == "running":
                drop[i] = bool(np.any((ls_ < t1) & (lp > t0)))
            elif lab == "licking":
                drop[i] = bool(np.any((rs < t1) & (rp > t0)))
    keep = ~drop
    order = np.argsort(sample[keep], kind="stable")
    return sample[keep][order], label[keep][order], period[keep][order], int(drop.sum())
