"""THE ENGAGEMENT GATE FOR THE REST ARM — one definition, resolved by every rest analysis.

WHAT WAS WRONG (found 2026-09-16, Priya: *"all only on working trials?"*). Every rest analysis in
this project DOCUMENTED an engagement gate and NONE of them applied one. `rest_position_decode`'s
docstring says in terms:

    "WORKING TRIALS ONLY. A rest period counts only if the trials bracketing it are both WORKING --
     the animal responded, or missed while still attempting. The terminal quit period is excluded,
     for the same reason every other family excludes it: a sated animal's rest is a different state,
     and it concentrates at the end of the session where drift is largest."

and its own stdout header prints "(working trials, block-CV, circular-shift null)". The only filter
it actually applied was that the two bracketing trials AGREE ON POSITION. `rest_position_permutation`
(finding 11, observed/null 1.622), `rest_frozen_decoder` (15f) and `shared_position_projection`
(15s) all inherited the same claim and the same omission -- 15f by being written from
`rest_position_decode` as a template, copying the prose along with the structure.

WHY IT IS NOT A STALE COMMENT. `flag_engagement`'s terminal quit period is exactly where rest is
most ABUNDANT -- the animal has stopped working, so the inter-trial intervals lengthen and multiply
-- and it sits at the END of the session, where drift is largest. Both of those push in the same
direction as the effects being measured. Worse, **the quit period grows post-stroke**, so its share
of rest periods is not constant across epochs: a confound that moves WITH the independent variable.

THE GATE IS `precue_engagement_states.engagement_gate`, not a fourth re-derivation. That is the one
the imaging deck already uses (`beta_maps._quit_mask`, `grant_figures._gate_all`), so the rest arm's
`working` now means what `working` means everywhere else. A second implementation would agree today
and diverge the first time either moved -- the failure `quiet_periods.rest_mask` records having
already happened once, when two modules computed the rest mask independently and "a comment in the
second claimed they agreed".

REPORT BOTH. `--no-engagement-gate` keeps the ungated behaviour so the SIZE of the correction can be
measured rather than asserted. A fix whose magnitude is unknown is not yet a finding.
"""
from __future__ import annotations

import numpy as np


def engaged_by_cue(session, cue_samples, codes, response_window_s=None):
    """``bool[n_cues]`` — True where that trial is ENGAGED (working), False inside the quit period.

    Aligned to the CUE index, which is what the rest collectors carry: they locate a rest period's
    bracketing trials as indices into `cue_samples`/`codes`, so the gate has to be indexed the same
    way or it would be applied to the wrong trials while looking correct.

    ``responded`` is "a lick inside the session's real response window", read per session from
    `gui_config.json timing.response_window` by `daq_trials` -- never `defaults.yaml`'s 2.0 s, which
    was never the task's window (docs/GUI_TRIALS_LOGGING.md).

    FAILS OPEN, LOUDLY. If the gate cannot be built the session keeps every rest period and the
    caller is told, rather than silently losing the session or silently losing the gate. Both of
    those failure modes have happened in this project; a stated fallback has not.
    """
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.precue_engagement_states import engagement_gate

    n = len(np.asarray(cue_samples))
    if n == 0:
        return np.zeros(0, bool), "no cues"
    try:
        responded = _responded_per_cue(session, cue_samples, response_window_s)
    except Exception as ex:                                              # noqa: BLE001
        return np.ones(n, bool), f"gate unavailable ({type(ex).__name__} {str(ex)[:40]}) -- UNGATED"
    if responded is None or len(responded) != n:
        return np.ones(n, bool), "gate unavailable (no lick/response data) -- UNGATED"
    order = np.arange(n)
    pos = np.array([POSITION_NAMES.get(int(c), str(c)) for c in np.asarray(codes)[:n]])
    not_eng = engagement_gate(order, np.asarray(responded, bool), pos)
    eng = ~np.asarray(not_eng, bool)
    return eng, f"{int((~eng).sum())}/{n} trials in the quit period"


def engaged_frame_mask(session, n_frames):
    """``bool[n_frames]`` — True for frames BEFORE the terminal quit period begins.

    THE ASYMMETRY THIS CLOSES (found 2026-09-16, Priya: *"the rest maps used for the map deltas --
    are those engagement gated?"*). In every rest-referenced map family the TRIAL side is gated and
    the REST side is not: `_working_xy(..., variant="working")` applies `beta_maps._quit_mask`,
    while `session_rest_svt_timelocal` / `session_restw_svt` average rest frames straight off the
    mask, and `quiet_periods.rest_mask` has no engagement term. So the minuend excludes the quit
    period and the subtrahend includes it.

    WHY THAT IS THE SAME ARGUMENT `restw` WAS BUILT ON, one axis over. `session_restw_svt`'s own
    docstring: *"`rest` averages over rest FRAMES, so a position contributing more rest frames pulls
    the baseline toward its own resting state ... Post-stroke the animal stops attempting the far
    positions ... so the baseline changes WITH the deficit."* Substitute "behavioural state" for
    "position": MEASURED, the quit period's share of rest FRAMES runs 3.1% pre, **18.7% acute**,
    17.7% subacute, 4.7% chronic (worst session PS94_0819 at 60.6%). `restw` fixed the position
    axis; the engagement axis went unfixed because the trial side was already gated and nothing
    compared the two.

    A SINGLE TERMINAL BOUNDARY IS CORRECT HERE because `engagement_gate` requires a NON-RECOVERING
    collapse -- only the final sustained one counts -- so the quit period is a suffix of the session
    by construction. This would be wrong for `spout_behavior.flag_engagement`, which unions a
    terminal tail with a rolling-rate gate and can therefore flag mid-session bouts.

    FRAME MAPPING is the same rule `quiet_periods` uses to build `*_quiet_frame.npy` in the first
    place: regime B maps through `original_frame_index_ch0 + offset`, regime A takes every second
    pco edge. Deriving it differently here would put the gate a few frames off the boundary the mask
    itself was cut on.

    FAILS OPEN, LOUDLY, like `engaged_by_cue`.
    """
    import glob
    import json

    import h5py

    from wfield_local import daq_io
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events

    n_frames = int(n_frames)
    keep = np.ones(n_frames, bool)
    try:
        with h5py.File(session["h5"], "r") as f:
            dn = [x.decode() for x in f["digital/channel_names"][:]]
            packed = f["digital/packed_samples"][:, 0]
        pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
        cue = _load_cue_events(session["h5"])
        cs = np.asarray(cue["cue_samples"], np.int64)
        codes = np.asarray(classify_cues_with_backup(session, cue, verbose=False))
        eng, note = engaged_by_cue(session, cs, codes)
        if "UNGATED" in note or bool(eng.all()):
            return keep, note if "UNGATED" in note else "no quit period"
        first_quit_sample = int(cs[int(np.flatnonzero(~eng)[0])])

        if session.get("regime") == "B":
            fm = sorted(glob.glob(f"{session.get('fmdir') or session['mc']}"
                                  f"/*cleanpairs_frame_map.npz"))
            summ = sorted(glob.glob(f"{session.get('fmdir') or session['mc']}"
                                    f"/*cleanpairs_summary.json"))
            if not fm or not summ:
                return keep, "no frame map -- UNGATED"
            with open(summ[0]) as fh:
                off = int(json.load(fh)["chosen_exposure_offset"])
            z = np.load(fm[0])
            fs_samp = pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
        else:
            fs_samp = pco[np.arange(len(pco) // 2) * 2]
    except Exception as ex:                                              # noqa: BLE001
        return keep, f"gate unavailable ({type(ex).__name__} {str(ex)[:40]}) -- UNGATED"

    m = min(len(fs_samp), n_frames)
    keep[:m] = np.asarray(fs_samp[:m]) < first_quit_sample
    if m < n_frames:
        # Frames past the mapping are past the end of the recording; keeping them would put
        # unmapped frames on the engaged side of a gate that never saw them.
        keep[m:] = False
    return keep, f"{int((~keep).sum())}/{n_frames} frames in the quit period"


def _responded_per_cue(session, cue_samples, response_window_s=None):
    """Did the animal lick within the response window of each cue? From the CANONICAL events."""
    from wfield_local import behavior_events as be
    from wfield_local import config

    lab = session["label"]
    an, mmdd = lab.split("_")[0], lab.split("_")[1]
    ev = be.get_or_compute(config.resolver(), an, f"2026{mmdd}")
    if ev is None:
        return None
    fs = float(ev["fs"])
    licks = np.asarray(ev["lick_onsets"], np.int64)
    if response_window_s is None:
        response_window_s = _response_window_s(session)
    w = round(float(response_window_s) * fs)
    cs = np.asarray(cue_samples, np.int64)
    # searchsorted rather than a loop: n_cues x n_licks is ~500 x ~15,000 per session and this runs
    # for every session of every rest analysis.
    lo = np.searchsorted(licks, cs, "left")
    hi = np.searchsorted(licks, cs + w, "right")
    return hi > lo


def _response_window_s(session):
    """The session's REAL response window (3.5 s in every session to date), never a config default.

    `defaults.yaml`'s 2.0 s was never the task's window and is only a fallback -- reading it as the
    truth is a documented error (docs/GUI_TRIALS_LOGGING.md).
    """
    import json
    from pathlib import Path

    from wfield_local import config

    d = session.get("behavior_dir") or session.get("mc")
    for cand in (Path(str(d)) / "gui_config.json",) if d else ():
        if cand.exists():
            try:
                t = json.loads(cand.read_text()).get("timing", {})
                if t.get("response_window"):
                    return float(t["response_window"])
            except Exception as ex:                                      # noqa: BLE001
                # SAID OUT LOUD. A malformed gui_config silently falling back to the config default
                # would substitute a 2.0 s window the task never used, which is a documented error
                # rather than a harmless default.
                print(f"  !! response_window from {cand}: "
                      f"{type(ex).__name__} {str(ex)[:50]} -- falling back", flush=True)
    try:
        return float(config.defaults()["behavior"]["response_window_s"])
    except Exception:                                                    # noqa: BLE001
        return 3.5
