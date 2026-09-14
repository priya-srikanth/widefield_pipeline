"""Rest frames SPLIT BY THE POSITION they belong to -- the input `restw` is built from.

WHY THIS IS A MODULE AND NOT A HELPER INSIDE ITS CALLER. Five scripts in `scripts/rest_migration/`
each carry their own copy of `_frame_samples`, and the rest-period labelling rule was about to
acquire a sixth copy inside `position_reference_maps`. The last time one quantity had two
implementations in this repo -- the flat map baseline against the encoder's time-local one -- they
disagreed for months and the map side was the wrong one (DECISIONS.md, 2026-09-13). One home.

WHAT "THE POSITION A REST PERIOD BELONGS TO" MEANS, and why it is defined conservatively. A rest
period sits between two trials. It is labelled ONLY when the preceding and following trials are at
the SAME position, so the label is unambiguous. Boundary periods -- where the position changes --
are DISCARDED here on purpose: they are the ones that separate persistence from anticipation, they
are the subject of `rest_block_boundary`, and a baseline is not the place to take a position on
that question. They are ~1/6 of periods.

THE LABEL IS RETROSPECTIVE-OR-PROSPECTIVE BY CONSTRUCTION and this module does not care, because
within a block those are the same position. That is exactly why the boundary test exists separately.
"""
from __future__ import annotations

import glob
import json

import numpy as np

#: Minimum positions that must contribute before a weighted baseline is worth calling weighted.
#: BELOW THIS IT IS NOT THE QUANTITY IT CLAIMS TO BE. `restw` exists to stop the baseline's
#: composition tracking which positions the animal still works; if only three positions survive,
#: an equal average over those three still tracks the deficit -- just less. Returning None makes
#: that session lose its RESTW column rather than quietly report a differently-defined baseline
#: under the same name. Four of six is the floor because PS94 has ZERO engaged trials at two
#: positions in some epochs (G1b), and excluding those sessions entirely would be a stronger
#: intervention than this reference needs.
MIN_POSITIONS_FOR_WEIGHTED = 4

#: Rest frames a position needs before it may contribute a level to the weighted baseline.
#: EQUAL WEIGHTING AMPLIFIES THE THINNEST ESTIMATE, which is the hazard it buys along with the
#: benefit: a position represented by 30 frames would count as much as one represented by 3,000.
#: 200 frames is ~6.5 s at 31 Hz, comfortably above the level at which a median is stable, and well
#: below the 1,000-3,000 a position normally holds -- so it excludes only genuinely starved
#: positions. Measured context: per-position ITI survival runs 24.4%-56.7% pre-stroke
#: (`position_survival`), so thin positions are a real occurrence, not a hypothetical.
MIN_FRAMES_PER_POSITION = 200


def frame_samples(mc, fmdir, regime, pco):
    """DAQ sample index of every imaging FRAME, or None.

    THE CANONICAL COPY. Regime B sessions were acquired as clean pairs and carry an explicit frame
    map plus a chosen exposure offset; everything else is strict blue/violet alternation and the
    blue frames are the even exposures. Five scripts had this inline; they should import it.
    """
    if regime == "B":
        fm = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_frame_map.npz"))
        summ = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_summary.json"))
        if not fm or not summ:
            return None
        with open(summ[0]) as fh:
            off = int(json.load(fh)["chosen_exposure_offset"])
        z = np.load(fm[0])
        return pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
    return pco[np.arange(len(pco) // 2) * 2]


def _session_daq(session):
    """``(rest_sample_mask, cue_samples, codes, trial_starts, frame_samples, sync_edges)``.

    One read of the session's DAQ and rest mask. Raises rather than returning a partial tuple --
    every caller needs all of it, and a half-loaded session producing a plausible baseline is the
    failure mode this repo has hit most often.
    """
    import h5py

    from wfield_local import daq_io, joint_basis  # noqa: F401  (joint_basis kept for parity)
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.quiet_periods import quiet_dir

    qs = sorted(glob.glob(f"{quiet_dir(session['mc'])}/*quiet_sample.npy"))
    if not qs:
        raise FileNotFoundError("no rest sample mask")
    rest = np.load(qs[0]).astype(bool)
    with h5py.File(session["h5"], "r") as f:
        dn = [x.decode() for x in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
    pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
    ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
    sync = daq_io.rising_edges((packed >> dn.index("sync")) & 1)
    cue = _load_cue_events(session["h5"])
    codes = np.asarray(_classify_cues(cue["cue_samples"], cue["strobe_samples"],
                                      cue["strobe_codes"]))
    cs = np.asarray(cue["cue_samples"], np.int64)
    fs_samp = frame_samples(session["mc"], session.get("fmdir"), session.get("regime"), pco)
    if fs_samp is None:
        raise FileNotFoundError("no frame map")
    return rest, cs, codes, ts, fs_samp, sync


def _apply_docked(session, rest, cs, codes, ts, sync):
    """Intersect the rest mask with the STRICT docked window. ``(mask, reconstructed)``.

    DOCKED IS A SUBSET OF REST, NEVER A REPLACEMENT. It constrains where the SPOUT is and says
    nothing about the animal, so the not-running / not-licking conditions still have to hold.

    Falls back to a DAQ-only reconstruction when the behaviour log's clock will not align, and
    reports that it did -- a reconstructed window uses the per-position travel table, and a session
    built that way must be identifiable so a result can be re-run without it.
    """
    from wfield_local import config
    from wfield_local.docked_periods import docked_mask_any
    from wfield_local.spout_behavior import discover_sessions

    an, mmdd = session["label"].split("_")[0], session["label"].split("_")[1]
    cands = discover_sessions(config.resolver(), f"2026{mmdd}", [an])
    dm, source = docked_mask_any(cands[0] if cands else None, sync, cs, codes, ts, rest.shape[0])
    if dm is None:
        return None, False
    return rest & dm[: rest.shape[0]], source == "reconstructed"


def rest_frames_by_position(session, n_frames, *, docked=False):
    """``({code: frame_index_array}, info)`` -- rest frames grouped by the position they sit at.

    ``n_frames`` bounds the result to the signal's length, so the indices are directly usable
    against an ``(K, T)`` SVT.

    ``info`` carries ``n_periods``, ``reconstructed`` and ``error``; an empty dict result with a
    populated ``error`` is a LOAD FAILURE, not a session with no rest -- the two must not be
    confused by a caller deciding whether to drop a column.
    """
    info = {"n_periods": 0, "reconstructed": False, "error": None}
    try:
        rest, cs, codes, ts, fs_samp, sync = _session_daq(session)
        if docked:
            rest, rebuilt = _apply_docked(session, rest, cs, codes, ts, sync)
            if rest is None:
                info["error"] = "no docked window even reconstructed"
                return {}, info
            info["reconstructed"] = rebuilt
    except Exception as ex:                                            # noqa: BLE001
        info["error"] = f"{type(ex).__name__} {str(ex)[:70]}"
        return {}, info

    pad = np.concatenate([[0], rest.view(np.int8), [0]])
    dif = np.diff(pad)
    f_of = np.clip(fs_samp, 0, rest.shape[0] - 1)
    out = {}
    for aa, bb in zip(np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)):
        prev = np.searchsorted(cs, aa, "right") - 1
        nxt = np.searchsorted(ts, bb, "left")
        if prev < 0 or nxt >= len(ts):
            continue
        nc = np.searchsorted(cs, ts[nxt], "left")
        # THE LABEL MUST BE UNAMBIGUOUS: preceding and following trial at the same position, or the
        # period is dropped. See the module docstring on why boundary periods are not the baseline's
        # business.
        if nc >= len(codes) or prev >= len(codes):
            continue
        if codes[prev] != codes[nc] or codes[prev] < 0:
            continue
        fr = np.flatnonzero((f_of >= aa) & (f_of < bb))
        fr = fr[fr < int(n_frames)]
        if fr.size:
            out.setdefault(int(codes[prev]), []).append(fr)
            info["n_periods"] += 1
    return {c: np.concatenate(v) for c, v in out.items()}, info
