"""Decode spout position from LICK BOUT ONSETS, and measure what the extra events actually buy.

The production lick-aligned decoder uses one lick per trial (first after the cue, within
decode.max_rt_s). `lick_bout_events` supplies 1.16-4.24 bout onsets per trial instead, correctly
attributed to the spout position present at the time. This module turns those into features and asks
the only question worth asking: does the extra data improve the readout, or just enlarge n?

THREE THINGS THIS GETS RIGHT THAT A NAIVE VERSION WOULD NOT.

1. CV GROUPS ARE TRIALS, NOT BOUTS. Several bouts come from one trial and share its position, its
   engagement state and its noise. Splitting them across folds would let the model see a trial in
   training and be tested on the same trial's other bouts -- leakage that inflates accuracy exactly
   in proportion to how much extra data the method appears to add. Grouping by trial is what makes
   the comparison against the one-lick baseline honest rather than flattering.

2. THE FEATURE WINDOW IS BOUNDED BY THE RESPONSE WINDOW, not just the onset. A bout at cue+3.4 s with
   a 2 s window would read 1.9 s of spout movement (Priya, 2026-08-17: nothing after the response
   window). Bouts whose window would overrun are DROPPED and counted, so the cost of the bound is
   visible instead of silently changing which trials contribute.

3. PHASES ARE SCORED SEPARATELY. `approach` bouts (pre-cue, e.g. licking leftover water as the spout
   arrives) and `response` bouts are different behavioural contexts, and the animals differ wildly in
   their mix -- PS92 is 74% approach, PS95 6%. A pooled number would be a different quantity per
   animal.

WHAT A POSITIVE RESULT WOULD MEAN. Approach-phase bouts occur BEFORE the cue, so decoding position
from them is a motor/sensory readout of a spout the animal is already interacting with -- not a
prediction of an upcoming action. It is a cleaner motor reference than the post-cue lick, not a
second pre-cue code, and should not be read as one.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from wfield_local import lick_bout_events as lb, nolick_analysis as na, nolick_decoder as nd
from wfield_local.behavior_position import classify_cues_with_backup
from wfield_local.locanmf_crossanimal_dff import _frames
from wfield_local.locanmf_frozen_decoder import _pipe
from wfield_local.locanmf_position_decoder import _bins_for, _build_signal, _window_feature
from wfield_local.plot_lick_aligned_averages import _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue

POST_S = 0.5          # short by design: bouts recur every few hundred ms, so a 2 s window would
                      # straddle the next one and stop being "aligned to this movement"


def bout_features(s, source="roi", post_s=POST_S, bins=None):
    """(X, y, groups, phase, dropped) aligned to bout onsets, in frames of the corrected movie."""
    args = nd._args(source=source, align="lick", post_s=post_s, bins=bins)
    ce = _load_cue(s["h5"])
    le = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    codes = classify_cues_with_backup(s, ce)
    rw = nd.response_window_for(s)
    ev = lb.bout_onsets_with_position(ce["cue_samples"], ce["strobe_samples"], le["lick_samples"],
                                      codes, ce["sample_rate_hz"], rw)
    if not len(ev["code"]):
        return None

    # bouts whose feature window would run past the response window read spout movement -> drop
    keep = (ev["t_from_cue_s"] + post_s) <= rw
    dropped = int((~keep).sum())
    ev = {k: np.asarray(v)[keep] for k, v in ev.items()}
    if not len(ev["code"]):
        return None

    # onsets are DAQ samples; map them to corrected frames the same way every other event is mapped
    cue_f, _lick_f, _cs = _frames(s, ce, le)
    _cf2, bout_f, _cs2 = _frames(s, ce, {**le, "lick_samples": ev["onset_sample"]})
    sig, feat_reg = _build_signal(s, source)
    nb = _bins_for(args)
    post_n = int(round(post_s * args.fs))
    T = sig.shape[1]

    X, y, g, ph = [], [], [], []
    for i, fr in enumerate(np.asarray(bout_f, int)):
        if fr < 0 or fr + post_n > T:
            continue
        X.append(_window_feature(sig, int(fr), post_n, nb, 0.0))
        y.append(int(ev["code"][i]))
        g.append(int(ev["trial"][i]))          # GROUP = trial, see module docstring
        ph.append(str(ev["phase"][i]))
    del sig
    if not X:
        return None
    return (np.array(X), np.array(y, int), np.array(g, int), np.array(ph, dtype=object), dropped)


def analyse_session(s, source="roi", post_s=POST_S, n_perm=500, verbose=True):
    """Bout-onset decoding for one session, overall and per phase."""
    out = bout_features(s, source=source, post_s=post_s)
    if out is None:
        return {"label": s["label"], "skipped": "no usable bouts"}
    X, y, g, ph, dropped = out
    res = {"label": s["label"], "source": source, "post_s": post_s,
           "n_bouts": int(len(y)), "n_trials": int(len(np.unique(g))),
           "n_dropped_past_window": dropped,
           "bouts_per_trial": float(len(y) / max(len(np.unique(g)), 1)),
           "criterion": "bout_onsets"}

    def _score(m, tag):
        if m.sum() < 40 or len(np.unique(y[m])) < 2 or len(np.unique(g[m])) < 2:
            return {"n": int(m.sum()), "note": "too few bouts/trials to cross-validate"}
        ng = min(5, int(np.unique(g[m]).size))
        pred = cross_val_predict(_pipe(), X[m], y[m], cv=GroupKFold(ng), groups=g[m])
        d = na.evaluate_arm(y[m], pred, n_perm=n_perm)
        d["accuracy_raw"] = float(accuracy_score(y[m], pred))
        d["n_trials"] = int(np.unique(g[m]).size)
        return d

    res["all"] = _score(np.ones(len(y), bool), "all")
    for p in lb.PHASES:
        res[p] = _score(ph == p, p)
    if verbose:
        a = res["all"]
        print(f"{s['label']}: {res['n_bouts']} bouts / {res['n_trials']} trials "
              f"({res['bouts_per_trial']:.2f} per trial, {dropped} dropped past the window)  "
              f"bal={a.get('balanced_accuracy', float('nan')):.3f} "
              f"(null {a.get('bal_null_mean', float('nan')):.3f}, p={a.get('bal_p', float('nan')):.4f})",
              flush=True)
    return res
