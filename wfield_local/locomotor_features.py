"""Cortical features for RUNNING and QUIET segments -- the spout-position-agnostic control arm.

Priya, 2026-09-11: "we don't need position information for this at all -- spout position agnostic.
I'm more looking for evidence that not *all* decoding/encoding degrades post-stroke, with running
as an example."

WHAT THE ARGUMENT IS. The lesion is VENTROLATERAL STRIATAL. No cortex is damaged, the imaging window
is the same one, the LocaNMF basis is the same, and the animals keep running -- median 3.1% of a
pre-stroke session against 5.1% acutely and 6.3% subacutely, so there is MORE running data exactly
where the position code collapses. If the same undamaged cortex still reports running normally while
its report of lick target has changed, then the target deficit is specific, and the generic
explanations a reviewer reaches for first -- window clouding, haemodynamic drift, arousal, basis
drift, "you cut a hole in the brain and everything got worse" -- are all ruled out at once, because
every one of them would degrade the running readout too.

WHY THIS IS A SEPARATE MODULE AND NOT A `variant`. Every trial class in `grant_figures` --
`lick`, `working`, `stopped` -- is a SUBSET OF THE TRIALS, selected by a mask over rows that
`_trial_features` already produced. These are not trials. A running bout has no cue, no position and
no alignment point; it is a window at an arbitrary DAQ sample. Threading that through
`_session_trials` would mean giving the trial pipeline a row that is not a trial, and the first
thing to break would be the position label everything downstream assumes.

THE FRAME MAPPING IS THE ONLY SUBTLE PART and it is borrowed, not reinvented: `_frames` already
resolves a session's DAQ-sample-to-frame relation in both acquisition regimes, and
`_window_feature` already turns a frame index into the binned feature vector every other family
uses. This module is the join between them.
"""
from __future__ import annotations

import numpy as np

from wfield_local import locomotor_state as ls
from wfield_local.locanmf_crossanimal_dff import _frames
from wfield_local.locanmf_position_decoder import _window_feature
from wfield_local.plot_lick_aligned_averages import _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events


def sample_to_frame(s, samples):
    """DAQ sample -> imaging frame index, in whichever regime this session was acquired in.

    Returns -1 where the sample falls outside the imaging coverage, matching `_frames`' own
    convention so a caller that already drops negative frames needs no new rule.
    """
    cue = _load_cue_events(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    _cf, _lf, csmp = _frames(s, cue, lk)
    samples = np.asarray(samples, np.int64)
    if csmp is not None:
        from wfield_local.framemap_event_maps import _nearest_corrected_frame, coverage_mask
        f = _nearest_corrected_frame(samples, np.asarray(csmp))
        return np.where(coverage_mask(samples, csmp), f, -1)
    # REGIME A: frames come from the DAQ's own PCO exposure pulses, two per corrected frame.
    from wfield_local.plot_spout_trial_averages import _event_frame_indices_from_pco
    pco = np.asarray(cue["pco_samples"])
    if pco.size < 2:
        return np.full(samples.shape, -1, np.int64)
    f = _event_frame_indices_from_pco(samples, pco) // 2
    inside = (samples >= pco[0]) & (samples <= pco[-1])
    return np.where(inside, f, -1)


def segment_features(s, signal, events, *, fs_img, seg_s=ls.SEGMENT_S, bins=ls.SEGMENT_BINS,
                     cap=ls.MAX_SEGMENTS_PER_PERIOD, three_way=True):
    """``(X, y, period)`` for one session's behavioural-state segments.

    ``three_way`` (the default) labels quiet / running / LICKING; False gives the binary
    running-vs-quiet arm. The binary one decodes at AUROC 0.99-1.00 within session -- a CEILING,
    which cannot show preservation -- so the three-way problem is the one the figures use and the
    binary one is kept as the simpler companion.

    ``signal`` is components x frames on the SAME basis the trial arms use, so the two analyses are
    two readouts of one feature space rather than two feature spaces.

    NO PER-SEGMENT BASELINE SUBTRACTION, and this is a real difference from the trial arms. A trial
    has a defined pre-cue period to subtract; a segment inside a five-second running bout has no
    "before" that is not also running, and subtracting the first bin from the rest would remove the
    sustained component that IS the running signal. The session-level z-score `pool_sessions`
    applies is what keeps sessions comparable, and it is applied by the caller exactly as for
    trials.

    Segments whose window leaves the imaging coverage are dropped, not clipped: a clipped window
    starts at frame 0 and is decoded from an arbitrary moment (the failure `coverage_mask` exists
    for, which cost PS95 8/13 23% of its trials before it was caught).
    """
    if s["label"] in ls.EXCLUDE_SESSIONS:
        return (np.zeros((0, signal.shape[0] * bins)), np.zeros(0, object), np.zeros(0, np.int64))
    if three_way:
        smp, lab, per, _n_drop = ls.three_way_segments(events, seg_s=seg_s, cap=cap)
    else:
        smp, lab, per = ls.running_and_quiet_segments(events, seg_s=seg_s, cap=cap)
    if not len(smp):
        return (np.zeros((0, signal.shape[0] * bins)), np.zeros(0, object), np.zeros(0, np.int64))
    f0 = sample_to_frame(s, smp)
    post_n = max(1, round(seg_s * fs_img))
    T = signal.shape[1]
    keep = (f0 >= 0) & (f0 + post_n <= T)
    if not keep.any():
        return (np.zeros((0, signal.shape[0] * bins)), np.zeros(0, object), np.zeros(0, np.int64))
    nb = min(int(bins), int(post_n))
    X = np.array([_window_feature(signal, int(w0), post_n, nb, 0.0) for w0 in f0[keep]])
    return X, lab[keep], per[keep]
