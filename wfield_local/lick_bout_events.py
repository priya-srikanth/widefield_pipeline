"""Lick BOUT ONSETS while the spout is at a known position — the motor readout's event set.

WHY THIS EXISTS. The lick-aligned decoder currently uses exactly ONE lick per trial: the first one
strictly after the cue, and only if it lands within decode.max_rt_s. Sessions have 5.4-15.1 detected
licks per trial, so it discards roughly 80-93% of the lick events available. Priya (2026-08-17) asked
whether we should instead use all licks while the spout sits at the correct position -- for example
when a mouse misses reward on one trial, the water stays on the spout, and it licks as the spout
ARRIVES for the next trial, before that trial's cue.

BOUT ONSETS, NOT EVERY LICK (Priya's call, 2026-08-17). Licks inside a bout are ~5-7 Hz and highly
correlated; treating each as an independent observation would multiply n by 5-15 while adding almost
no independent information, and would make any cross-validated estimate anticonservative. A bout
onset is the movement INITIATION, which is the event a motor readout is about. `segment_bouts` from
spout_behavior does the splitting, so bouts here and bouts in the behaviour figures are the same
objects.

TWO CORRECTIONS TO HOW LICKS GET A POSITION.

1. LABEL BY THE SPOUT STROBE, NOT THE PRECEDING CUE. `framemap_event_maps` assigns each lick the
   position of the most recent CUE when a behavior-log override is supplied. The spout moves and
   strobes BEFORE the cue, so a lick in the arrival window carries the PREVIOUS trial's label --
   precisely the case Priya described. Because positions run in ~6-trial blocks the label is usually
   still right by luck; measured 2026-08-17 the genuinely mislabelled fraction is PS93 7.3%,
   PS94 5.0%, PS95 0.8% of all licks, concentrated at block transitions. Labelling by the most recent
   STROBE is correct by construction and needs no luck.

2. NOTHING AFTER THE RESPONSE WINDOW. The spout starts moving once the window closes, so the
   eligible interval for trial k is [strobe_k, cue_k + response_window]. That admits the arrival and
   ENL period (where the leftover-water licks live) and the response window, and excludes the
   movement.

PHASE IS KEPT, NOT COLLAPSED. Each bout onset is labelled `approach` (strobe -> cue) or `response`
(cue -> window close). They are different behavioural contexts -- uncued and possibly consummatory
versus cued and reward-directed -- and pooling them silently is the same mistake as pooling late with
undetected. The caller decides; this module reports.
"""
from __future__ import annotations

import numpy as np

from wfield_local import config
from wfield_local.spout_behavior import segment_bouts

PHASES = ("approach", "response")


def bout_onsets_with_position(cue_samples, strobe_samples, lick_samples, codes, sample_rate_hz,
                              response_window_s, max_ili_s=None, min_bout_licks=None):
    """Bout onsets (in SAMPLES) with the spout position that was actually present at the time.

    ``codes`` is the per-CUE position code (the pipeline's repaired labels), so the behaviour log
    still supplies the position VALUE; only the timing attribution changes -- a lick is assigned to
    the trial whose STROBE most recently preceded it, not the trial whose cue did.

    Returns a dict of equal-length arrays: ``onset_sample``, ``code``, ``trial`` (index into
    ``codes``), ``phase``, ``n_licks`` (bout size), ``t_from_cue_s``.
    """
    lick = np.sort(np.asarray(lick_samples, dtype=float))
    cue = np.asarray(cue_samples, dtype=float)
    strobe = np.sort(np.asarray(strobe_samples, dtype=float))
    codes = np.asarray(codes)
    fs = float(sample_rate_hz)

    beh = config.defaults()["behavior"]["licking"]
    max_ili = float(max_ili_s if max_ili_s is not None else beh["max_ili_ms"] / 1000.0)
    min_licks = int(min_bout_licks if min_bout_licks is not None else beh["min_bout_licks"])

    bouts = segment_bouts(lick / fs, max_ili, min_licks)
    if not bouts:
        return {k: np.array([], dtype=(object if k == "phase" else float))
                for k in ("onset_sample", "code", "trial", "phase", "n_licks", "t_from_cue_s")}
    onset = np.array([b[0] for b in bouts]) * fs
    nlicks = np.array([b[2] for b in bouts], dtype=int)

    # trial = the cue that FOLLOWS the strobe most recently preceding this onset. Going via the
    # strobe is the whole point; going via the cue directly is the bug this module avoids.
    si = np.searchsorted(strobe, onset, side="right") - 1
    has_strobe = si >= 0
    strobe_t = np.where(has_strobe, strobe[np.clip(si, 0, strobe.size - 1)], np.nan)
    trial = np.searchsorted(cue, strobe_t, side="right")      # first cue at/after that strobe
    ok = has_strobe & (trial < cue.size) & (trial < codes.size)

    t_cue = np.where(ok, cue[np.clip(trial, 0, cue.size - 1)], np.nan)
    t_from_cue = (onset - t_cue) / fs
    # eligible: after the spout arrived, and not past the response window (spout moves after it)
    ok &= (onset >= strobe_t) & (t_from_cue <= response_window_s)
    phase = np.where(t_from_cue < 0, "approach", "response").astype(object)

    code = np.where(ok, codes[np.clip(trial, 0, max(codes.size - 1, 0))], -1)
    ok &= code >= 0
    return {"onset_sample": onset[ok], "code": code[ok].astype(int), "trial": trial[ok].astype(int),
            "phase": phase[ok], "n_licks": nlicks[ok], "t_from_cue_s": t_from_cue[ok]}


def summarize(ev):
    """Counts by phase and position -- what the extra data actually buys, before any decoding."""
    if not len(ev["code"]):
        return {"n_bouts": 0}
    out = {"n_bouts": int(len(ev["code"])),
           "by_phase": {p: int((ev["phase"] == p).sum()) for p in PHASES},
           "median_bout_licks": float(np.median(ev["n_licks"])),
           "n_trials_represented": int(len(np.unique(ev["trial"])))}
    out["bouts_per_trial"] = out["n_bouts"] / max(out["n_trials_represented"], 1)
    return out
