"""Per-position EVOKED cortical maps -- the position-independent answer to "where".

WHY THIS EXISTS AND WHY IT OUTRANKS THE DECODER MAPS. `beta_maps` renders a one-vs-rest decoder
pattern, and Priya found the structural limit in it (2026-09-12): "in acute there may be less
ss-ul/ll activity in far-center trials, which makes the near ipsi acute trial map look as though
there is a relative *increase* in ss-ul/ll activity compared to pre-stroke". The Haufe pattern is a
covariance against the mean over ALL SIX positions, so one position losing drive lowers the
reference and hands every other position an increase it did not earn. The six maps are not
independent, and four of six rows in figure 14 cannot be read as written.

THE REFERENCE HERE IS WITHIN TRIAL AND PER POSITION: each map is that position's own
`post-cue mean - pre-cue mean`. Far-contralateral collapsing cannot leak into near-ipsilateral's
map, because near-ipsilateral's map never looks at far-contralateral's trials.

NOTHING IS RECOMPUTED. `framemap_event_maps` already writes these per session as
`*_spout_positions_1s_pre_post_delta_maps.npz` -- 18 arrays, six positions x {pre, post, delta}, as
540 x 640 Allen-aligned pixel maps, 123 of them on the share. This module aggregates them by epoch.

WHAT IT COSTS RELATIVE TO THE DECODER MAPS, stated so the two are not confused:

    decoder pattern (beta_maps)   isolates what DISTINGUISHES positions, but one-vs-rest couples
                                  them; a row that goes UP cannot be trusted
    evoked map (here)             positions independent, but shows the WHOLE task-evoked response
                                  at that position -- cue, licking, movement, arousal -- not only
                                  the part that carries target identity

They answer different questions and disagreeing is informative rather than contradictory: a change
visible here and absent in the decoder map is a change in drive that does not carry position
information, and the reverse is a change in tuning without a change in drive.
"""
from __future__ import annotations

import glob
from functools import lru_cache

import numpy as np

#: Map shape on the shared Allen grid, as `framemap_event_maps` writes them.
MAP_SHAPE = (540, 640)

#: Which of the three arrays per position to aggregate. `delta` is post-cue minus pre-cue -- the
#: EVOKED response, and the one whose reference is within trial. `pre` and `post` are the raw window
#: means and are kept reachable for diagnostics only: their absolute level moves with F0 and
#: photobleaching across a session, which is exactly what the subtraction removes.
FIELD = "delta"


def session_npz(session):
    """Path to this session's per-position map file, or None."""
    hits = glob.glob(f"{session['mc']}/spout_trial_averages_affine8v1/"
                     f"*_spout_positions_1s_pre_post_delta_maps.npz")
    return hits[0] if hits else None


def session_position_maps(session, field=FIELD):
    """``{position: (540, 640)}`` for one session, or {} when the file is absent.

    RETURNS {} RATHER THAN RAISING for a missing file. These npz are written by the imaging box's
    preprocessing, not by this analysis, so a session can legitimately lack one -- and losing a
    session should cost that session, not the figure.
    """
    f = session_npz(session)
    if not f:
        return {}
    out = {}
    with np.load(f) as z:
        for k in z.files:
            if not k.endswith(f"_{field}"):
                continue
            arr = np.asarray(z[k], float)
            if arr.shape == MAP_SHAPE:
                out[k[: -len(f"_{field}")]] = arr
    return out


@lru_cache(maxsize=4)
def maps_by_epoch(field=FIELD):
    """``({animal: {epoch: {position: {label: map}}}}, {animal: {epoch: n_sessions}})``.

    Epoch assignment is the deck's own, from each animal's lesion date -- the same `epoch_of_day`
    every other family here uses, so a session cannot sit in one epoch on this figure and another
    one elsewhere.
    """
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    out, counts = {}, {}
    for an in ANIMALS:
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        per, n = {}, {}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if e is None:
                continue
            got = session_position_maps(s, field)
            if not got:
                continue
            for q, m in got.items():
                per.setdefault(e, {}).setdefault(q, {})[s["label"]] = m
            n[e] = n.get(e, 0) + 1
        if per:
            out[an], counts[an] = per, n
    return out, counts
