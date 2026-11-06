"""Per-trial engagement rows and the terminal-quit detector for ONE session -- library code.

Moved VERBATIM from `scripts/rest_migration/engagement_decomposition.py` (`session_trials`, `near_codes`) and
`scripts/rest_migration/quit_point.py` (`session_quit` and its helpers) on 2026-11-05 (code review 11-04,
modularity item 1): `analysis_kit.session_behavior` -- the shared quit / trial definition -- imported them from
`scripts/`, so archiving a "one-off" script would have broken production numbers. The scripts re-export these names,
so every existing import keeps working. The reasoning behind each rule is in the two scripts' module docstrings
(won't vs can't, why no engagement gate; the step, right-censoring, the common horizon); read them before changing
anything here.
"""
from __future__ import annotations

import numpy as np

def session_trials(s, resp_s):
    """Per-trial rows for ONE session: position, hit, latency, anticipatory licks, trial order.

    Built here rather than via `daq_trials.build_trials` for one reason: positions come from
    `classify_cues_with_backup`, the REPAIRED classifier. The raw strobe collapses 6 positions onto
    4 on the August sessions with a dead `spout_bit1`, and those are pre-stroke days -- scoring a
    position-specific question on merged labels would answer a different question.
    """
    from wfield_local import daq_io
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.rest_by_position import _session_daq

    cue = _load_cue_events(s["h5"])
    cs = np.asarray(cue["cue_samples"], float)
    codes = np.asarray(classify_cues_with_backup(s, cue, verbose=False))
    # LICK COUNTS (n_licks_trial, n_licks_pre, cum_licks) use THE behaviour detector (coordinator 11-03, rule 9;
    # `analysis_kit.lick_count_samples`). The RESPONSE detection (hit / latency, which the engagement gate and the quit
    # point are scored on) keeps the deck's discriminator (`analysis_kit.lick_samples`), unchanged.
    from wfield_local import analysis_kit as _ak
    licks = np.sort(np.asarray(_ak.lick_count_samples(s), float))
    licks_resp = np.sort(np.asarray(_ak.lick_samples(s), float))
    with daq_io.open_daq(s["h5"]) as f:
        sr, _ = daq_io.session_attrs(f)
    _r, _cs2, _c, ts, _fs, _sy = _session_daq(s)
    ts = np.sort(np.asarray(ts, float))

    # THE ENL WINDOW OPENS AT THE LAST trial_start AT OR BEFORE THE CUE. Falling back to the cue
    # itself when there is none gives an EMPTY pre window, which scores as "no anticipation" -- so
    # those trials are dropped rather than counted as unmotivated.
    j = np.searchsorted(ts, cs, side="right") - 1
    pre_open = np.where(j >= 0, ts[np.clip(j, 0, None)], np.nan)

    nxt = np.append(cs[1:], np.inf)
    n = cs.size
    # THE TRIAL STOP (Priya 11-03): the DAQ trial_end TTL, else the task rule (`daq_trials.trial_stops`); LICKS PER
    # TRIAL = total licks from the cue to it (`n_licks_trial`), the window every licks-per-trial readout uses
    from wfield_local import daq_trials
    with daq_io.open_daq(s["h5"]) as f:
        te = daq_io.analog_channel(f, "trial_end", required=False)
    te_s = (daq_io.rising_edges((te > 2.5).astype(np.int8)) / sr) if te is not None else None
    stop_s, _src = daq_trials.trial_stops(cs / sr, licks / sr, resp_s, te_s)
    n_trial = daq_trials.licks_to_stop(cs / sr, stop_s, licks / sr)
    rows = []
    for k in range(n):
        if codes[k] < 0 or not np.isfinite(pre_open[k]) or pre_open[k] >= cs[k]:
            continue
        end = min(cs[k] + resp_s * sr, nxt[k])
        post = licks_resp[(licks_resp >= cs[k]) & (licks_resp <= end)]
        npre = int(((licks >= pre_open[k]) & (licks < cs[k])).sum())
        rows.append(dict(pos=int(codes[k]), order=k, frac=k / max(n - 1, 1),
                         n_licks_trial=int(n_trial[k]), stop_sample=float(stop_s[k] * sr),
                         hit=int(post.size > 0), n_licks_pre=npre,
                         anticipated=int(npre > 0),
                         # EFFORT AND TIME, SEPARATELY. See `licks_vs_time`.
                         cum_licks=float(np.searchsorted(licks, cs[k])),
                         elapsed_s=float(cs[k] - cs[0]) / sr,
                         latency_s=(float(post[0] - cs[k]) / sr if post.size else np.nan)))
    # WIDEFIELD_GATE_SOURCE=lp (2026-11-01, `wfield_local/gate_source.py`): a SEPARATE `gate_hit` column = this row's
    # hit OR a DAQ contact OR an LP attempt within `resp_s` -- what `quit_point.session_quit` scores the gate on. ADDED
    # ONLY UNDER THE SWITCH, so the default rows are unchanged; `hit` itself (hit rates, sharpness) never moves.
    from wfield_local import gate_source
    if gate_source.is_lp() and rows:
        order = np.array([r["order"] for r in rows])
        gh = gate_source.gate_responded(s["label"], order, np.array([bool(r["hit"]) for r in rows]), resp_s,
                                        cue["cue_samples"], cue["sample_rate_hz"])
        for r, g in zip(rows, gh):
            r["gate_hit"] = int(g)
    return rows


def near_codes():
    """Position codes for the NEAR spouts -- derived from the names, not written down as {0,1,2}."""
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    return {c for c, nm in POSITION_NAMES.items() if nm.startswith("close")}


EARLY_S = 600.0          # s; the fixed early window the independent lick rate is measured over


def _early_rate(rows, early_s=EARLY_S):
    """``(licks/min over the first `early_s`, licks/trial over those trials, n_trials)``.

    **THE POINT OF THIS COLUMN IS THAT IT IS NOT DERIVED FROM THE QUIT POINT**, which makes it the
    only non-circular predictor available for the fatigue question.

    `quit_cum_licks / quit_elapsed_s` looks like a lick rate and must never be used as one: it is
    `L / T`, so regressing it on `T` is guaranteed negative and on `L` guaranteed positive, for
    arithmetic rather than biological reasons. A FIXED early window is independent of when the quit
    occurs, so the two competing predictions become testable and OPPOSITE:

        MOTOR FATIGUE, quit at a fixed LICK COUNT  ->  quit TIME  ~ 1/rate, log-log slope -1
                                                       quit LICKS ~ const,  log-log slope  0
        TIME-DRIVEN, quit at a fixed TIME          ->  quit TIME  ~ const,  log-log slope  0
                                                       quit LICKS ~ rate,   log-log slope +1
    """
    early = [r for r in rows if float(r["elapsed_s"]) <= early_s]
    if len(early) < 5:
        return float("nan"), float("nan"), len(early)
    span = max(float(early[-1]["elapsed_s"]), 1.0) / 60.0
    licks = float(early[-1]["cum_licks"]) - float(early[0]["cum_licks"])
    return licks / span, licks / max(len(early) - 1, 1), len(early)


def session_quit(s, rows, resp_s=None):     # resp_s unused: `rows` already carry hit (A1)
    """``dict`` describing this session's terminal disengagement, or ``None`` if it never quits.

    Returns what had ACCUMULATED at the quit trial -- time, licks, rewards -- because those are the
    candidate threshold variables, and which of them is conserved across epochs is the question.
    """
    from wfield_local import daq_io
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.precue_engagement_states import engagement_gate

    order = np.array([r["order"] for r in rows])
    responded = np.array([bool(r["hit"]) for r in rows])
    # WIDEFIELD_GATE_SOURCE=lp: `session_trials` adds `gate_hit` (hit | DAQ contact | LP attempt within resp_s) only
    # under that switch; the gate is scored on it. Absent (the default) -> the line above, unchanged. hit_before /
    # hit_after / step_sharpness below stay on `hit` (the option moves the quit, never a hit rate).
    if rows and "gate_hit" in rows[0]:
        responded = np.array([bool(r["gate_hit"]) for r in rows])
    pos = np.array([POSITION_NAMES.get(r["pos"], str(r["pos"])) for r in rows])
    from wfield_local import trial_classes as _tc
    with _tc.daq_quit_rule("quit_point: the S28 quit (a measure of engagement)"):
        ne = np.asarray(engagement_gate(order, responded, pos), bool)
    dur = float(rows[-1]["elapsed_s"])
    er, el, en = _early_rate(rows)

    # RIGHT-CENSORED, NOT ABSENT (Priya, 2026-09-19: *"count the full session time if there is no
    # disengaged tail"*). **THE QUIT-DETECTION RATE IS CONFOUNDED BY SESSION-TERMINATION POLICY**:
    # pre-stroke Priya ran "up to 2 hrs or until they stopped licking", so a pre-stroke session
    # often ENDED AT THE QUIT and leaves no tail for a gate that requires a sustained non-recovering
    # run. Post-stroke was a strict 120 min / 100 trials per position, so the tail is always there.
    #
    # THE POLICY CHANGE IS VISIBLE IN THE DURATIONS: pre spans 74-167 min (IQR 89-143) against
    # acute's 100-126 (IQR 104-121). Comparing DETECTION RATES across that is comparing how long
    # the experimenter kept recording, and the 0.34-pre against 0.75-acute this module first
    # reported is mostly that.
    #
    # SO THE OUTCOME IS TIME ENGAGED, WITH CENSORING. No detected quit -> time engaged is the WHOLE
    # SESSION and the observation is RIGHT-CENSORED: **IF THE ANIMAL NEVER QUIT, it would have gone
    # on longer, so the recorded value is a LOWER BOUND on true engagement.**
    #
    # THIS COMMENT FIRST SAID UPPER BOUND, WHICH WAS BACKWARDS (Priya: *"isn't the pre-time engaged
    # a LOWER bound... if the animal never quit"*). Right-censoring means the event has not happened
    # yet, so the observation bounds the truth from BELOW. The upper-bound reasoning applied only to
    # the OTHER subpopulation -- sessions the animal DID quit, stopped so promptly that no tail
    # survives for the gate, where the recorded duration is approximately the true quit time.
    # Pre-stroke "no detected quit" is a MIXTURE of the two, and both are <= the truth.
    #
    # THE CONSERVATISM SURVIVES BY THE OPPOSITE ARGUMENT, and it is what licenses any claim here:
    # pre is ~66% censored (29/44) against acute's ~25% (4/16), so measured pre UNDERSTATES true pre
    # engagement more than measured acute understates acute. A measured pre > acute therefore
    # UNDERSTATES the true gap. A post-stroke SHORTFALL is trustworthy; a post-stroke EXCESS could be
    # differential censoring alone and means nothing.
    if not ne.any():
        return dict(quit_trial=-1, n_trials=len(rows), quit_frac=float("nan"),
                    quit_elapsed_s=float("nan"), quit_cum_licks=float("nan"),
                    quit_cum_rewards=float("nan"), hit_before=float("nan"),
                    hit_after=float("nan"), step_sharpness=float("nan"),
                    session_dur_s=dur, time_engaged_s=dur, censored=True,
                    frac_engaged=1.0, early_lpm=er, early_lpt=el, early_n=en)
    # THE QUIT POINT IS THE FIRST TRIAL OF THE TERMINAL RUN. `engagement_gate` already requires
    # non-recovery, so scanning back from the end to the first False is the start of that run.
    k = len(ne)
    while k > 0 and ne[k - 1]:
        k -= 1
    if k == 0 or k >= len(rows):          # quit at trial 0 is a failed session, not a quit
        return None

    # rewards, for the satiety branch -- the only one of the three that needs a channel we do not
    # otherwise read here
    n_rew = None
    try:
        with daq_io.open_daq(s["h5"]) as f:
            sr, _ = daq_io.session_attrs(f)
            rv = daq_io.analog_channel(f, "reward_ttl", required=False)
        if rv is not None:
            edges = daq_io.rising_edges((np.asarray(rv) > 2.5).astype(np.int8))
            n_rew = int((np.asarray(edges, float) / sr <= rows[k]["elapsed_s"]).sum())
    except Exception:                                                # noqa: BLE001
        pass

    return dict(quit_trial=int(k), n_trials=len(rows),
                quit_frac=float(k / max(len(rows) - 1, 1)),
                quit_elapsed_s=float(rows[k]["elapsed_s"]),
                quit_cum_licks=float(rows[k]["cum_licks"]),
                quit_cum_rewards=(float(n_rew) if n_rew is not None else float("nan")),
                hit_before=float(np.mean([r["hit"] for r in rows[:k]])),
                hit_after=float(np.mean([r["hit"] for r in rows[k:]])),
                # STEP SHARPNESS: how much of the total drop happens in the 20 trials around the
                # quit. Near 1 = a genuine step. Near the fraction of trials spanned = a ramp.
                step_sharpness=_sharpness(rows, k),
                early_lpm=er, early_lpt=el, early_n=en,
                session_dur_s=dur, time_engaged_s=float(rows[k]["elapsed_s"]),
                censored=False, frac_engaged=float(rows[k]["elapsed_s"] / dur) if dur else 1.0)


def _sharpness(rows, k, w=10):
    h = np.array([r["hit"] for r in rows], float)
    pre, post = h[:k], h[k:]
    if pre.size < w or post.size < w:
        return float("nan")
    total = pre.mean() - post.mean()
    if abs(total) < 1e-9:
        return float("nan")
    local = h[max(0, k - w):k].mean() - h[k:k + w].mean()
    return float(local / total)
