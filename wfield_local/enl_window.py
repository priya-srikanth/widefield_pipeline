"""The ENL (pre-cue) LICK-FREE window: the ONE search the ENL decoders and the combined encoder's "enl" scoring window
share. Moved verbatim from `locanmf_position_decoder` on 2026-11-04 (coordinator: the encoder's enl R2 window = the
decoders' lick-free ENL window), so the O2 encoder can use it without that module's sklearn / matplotlib / session
registry imports. `locanmf_position_decoder` re-exports both functions; nothing else changed.

The window LENGTH is the decoders' too (`enl_spec`): `lp_licks.enl_post_s()`, THE ENL window definition
(WIDEFIELD_ENL_WINDOW_S if set, else 1.0 s under the LP lick source, else `defaults.yaml decode.precue_post_s` = 2.0 s,
the DAQ source). Time-based callers (scripts that hold cue / lick TIMES, not frames) reach `precue_window_start` through
`lp_licks.enl_window_starts_s`, which runs it on a 1 ms grid: one search everywhere (merge of rebuild/encoder-classes
and rebuild/enl-window, 11-04).
"""
from __future__ import annotations

import numpy as np


def enl_spec() -> dict:
    """{"source": lp | daq, "length_s", "lickfree"}: the ENL decoders' window for this process's lick source."""
    from wfield_local import config, lp_licks
    d = config.defaults()["decode"]
    return {"source": lp_licks.lick_source(), "length_s": lp_licks.enl_post_s(),
            "lickfree": bool(d.get("precue_lickfree", True))}


def lickfree_window(cue_f, strobe_f, licks, win_n):
    """Latest ``win_n``-frame window before the cue containing NO licks, or None.

    Moved here from `precue_lickfree` on 2026-08-17 when Priya decided the HEADLINE pre-cue number
    should be lick-free: the rule now governs the production decoder, so it belongs beside it rather
    than in the module that used to be its only consumer.

    A FIXED window ending at the cue throws a trial away for a single lick anywhere in it -- including
    one 200 ms before the cue, when 2 s of clean data sits just earlier in the same enforced no-lick
    period. Instead: take the lick-free GAPS in [strobe, cue] and use the last ``win_n`` frames of the
    LATEST gap long enough to hold the window. Closest to the cue is preferred, being most informative
    about the upcoming action.

    BOUNDED AT THE STROBE, deliberately. Before the spout arrives this trial's position does not exist,
    and because the task avoids recent repeats, prior-trial activity carries real information about the
    upcoming position -- a window straying earlier would smuggle that in and look like a pre-cue code.
    """
    lo = int(np.ceil(strobe_f)) if np.isfinite(strobe_f) else None
    hi = int(cue_f)
    if lo is None or hi - lo < win_n:
        return None
    inside = np.sort(licks[(licks >= lo) & (licks < hi)])
    edges = np.concatenate(([lo], inside, [hi]))
    for i in range(len(edges) - 2, -1, -1):
        gap_start = int(edges[i]) + (1 if i > 0 else 0)      # a lick occupies its own frame
        gap_end = int(edges[i + 1])
        if gap_end - gap_start >= win_n:
            return gap_end - win_n
    return None


def precue_window_start(c0, strobe_f, licks_sorted, win_n, lickfree=True):
    """Start frame of this trial's PRE-CUE window, or None meaning DROP the trial.

    THE HEADLINE PRE-CUE NUMBER IS LICK-FREE (Priya, 2026-08-17). Previously only the Section C
    control restricted; the headline used every engaged trial and relied on the ENL to keep the
    window quiet, which it does 90.8-99.5% of the time -- but "almost always" is per SESSION, and
    PS93 8/9 falls to 76%, in the animal whose licking is already atypical. A readout whose whole
    claim is motor-independence should not rest on a task contingency holding on average.

    Trials with a clean fixed window keep it, so the common case stays exactly cue-aligned. Trials
    with a lick in it slide to the latest clean gap. Trials with no clean window ANYWHERE are
    DROPPED rather than kept-and-flagged -- for a headline number, "mostly clean" is the thing being
    retired.
    """
    fixed = int(c0) - win_n
    if not lickfree:
        return fixed if fixed >= 0 else None
    if fixed < 0:
        return None
    if not np.any((licks_sorted >= fixed) & (licks_sorted < fixed + win_n)):
        return fixed                                          # already clean; keep it cue-aligned
    return lickfree_window(c0, strobe_f, licks_sorted, win_n)


# ------------------------------------------------------------------------------------------ trace families (11-05)
#: THE SPOUT-ARRIVAL SETTLE (Priya 11-04 ~20:00, the slide decision's guard): a slid window must START at least this long
#: after the trial's position strobe -- the encoder's ENL-state settle -- so the spout-arrival transient stays out
SETTLE_S = 0.3


def sliding_lags(cue_s, strobe_s, lick_s, win_s: float, *, end_lag: float = 0.0, span_lo: float = -np.inf,
                 settle_s: float = SETTLE_S):
    """``(lo, hi)``: per trial the lag window (s from the cue) of its LICK-FREE ENL window, NaN = drop the trial -- for
    the families that average cue-ALIGNED TRACES on a lag grid (Priya 11-04 ~20:00: the between-trial lick ENL family
    and fr_enl_decode also SLIDE, like the imaging ENL decoders).

    THE SEARCH IS THE DECODERS' (`precue_window_start` through `lp_licks.enl_window_starts_s`, rule 9): the window
    [cue - win_s, cue) when it holds no lick of ``lick_s``, else the latest lick-free gap in [strobe, cue) long enough for
    it, else drop. Then the guard of the decision: a window starting before ``strobe + settle_s`` or before the family's
    extracted span (``span_lo``, e.g. -2.0 s for traces that start at cue - 2 s) is DROPPED, never clipped. A trial
    without a strobe keeps only an unslid (clean, cue-ending) window. ``hi`` = ``lo + win_s`` capped at ``end_lag``
    (the families that stop their window before the cue, e.g. -0.10 s so interpolation never reaches a post-cue frame,
    keep that end on the unslid window; a slid window already ends at the lick that moved it)."""
    from wfield_local import lp_licks
    cue = np.asarray(cue_s, float)
    strobe = np.asarray(strobe_s, float)
    w0 = lp_licks.enl_window_starts_s(cue, strobe, lick_s, float(win_s))
    lo = w0 - cue
    # the unslid window, snapped to exactly -win_s (the search runs on a 1 ms grid, so w0 - cue is off by < 0.5 ms)
    fixed = np.isfinite(lo) & (np.abs(lo + float(win_s)) < 1e-3)
    lo = np.where(fixed, -float(win_s), lo)
    ok = np.isfinite(lo) & (lo >= span_lo - 1e-9)
    ok &= np.where(np.isfinite(strobe), w0 >= strobe + settle_s - 1e-9, fixed)
    lo = np.where(ok, lo, np.nan)
    hi = np.where(ok, np.minimum(lo + float(win_s), end_lag), np.nan)
    return lo, hi


def window_mean(X, lags, lo, hi):
    """``X`` (trials x lags x columns) averaged over each trial's own lag window [lo, hi] (inclusive, as the families'
    fixed masks); NaN rows for dropped trials (lo NaN)."""
    X = np.asarray(X, float)
    lags = np.asarray(lags, float)
    out = np.full((X.shape[0],) + X.shape[2:], np.nan)
    for i, (a, b) in enumerate(zip(np.asarray(lo, float), np.asarray(hi, float))):
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        m = (lags >= a - 1e-9) & (lags <= b + 1e-9)
        if m.any():
            out[i] = np.nanmean(X[i, m], 0)
    return out


def strobe_for_cues(label: str, cue_s) -> np.ndarray:
    """Per DAQ cue time (s) of ``label``, its trial's position strobe time (DAQ s), NaN where no trial matches: from the
    v36 per-session windows.npz (`trial_classes._trial_times`), matched by cue time within `trial_classes.CUE_MATCH_S`
    exactly as the camera classes are (`trial_classes.cue_classes`). For families whose stored trial files carry no
    strobe."""
    from pathlib import Path

    from wfield_local import config
    from wfield_local import trial_classes as tc
    v36, _c1 = tc._paths(None, None)
    sd = Path(v36) / config.animal_of(label) / ("2026" + label.split("_")[-1])
    T = tc._trial_times(sd)
    T = T.assign(cls=(T.cue_daq + T.strobe_ms / 1000.0).astype(object))
    got = tc.cue_classes(T, np.asarray(cue_s, float))
    return np.array([float(v) if v != "unknown" else np.nan for v in got], float)
