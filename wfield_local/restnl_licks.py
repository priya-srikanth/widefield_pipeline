"""WHICH VIDEO LICKS LEAVE THE RESTnl REST MASK -- the RESTnl lick-set switch (Priya 11-17).

Priya 11-17: lick-free rest (RESTnl) should exclude DAQ + cam4 + cam1 licks (cam1-only events, water droplets removed),
the same lick train as the within-trial lick-free windows. Before 11-17 RESTnl removed DAQ licks and cam4 `licks_all.csv`
licks only. Both versions stay available:

  cam4       (DEFAULT, the RESTnl of every output up to 11-17) -- DAQ lick onsets + cam4 `licks_all.csv` (`daq_on_s`).
  cam4cam1   ("RESTnlc1") -- DAQ + cam4 + the cam1-only events of the WHOLE trial windows: cam1
             `cam1_licks/<a>_<date>_events_all.csv` (every context: pre-cue, post-cue, ITI; ~98-99 % of the video, the
             same windows as cam4 licks_all), rows NOT cam4-matched and NOT water droplets (`droplet` column =
             `cam1_licks.droplet_flags`; recomputed from it if the column is absent), `daq_on_s`. UNION with the cam4
             table read by the mask, so the cam1 version removes a SUPERSET of the cam4 version's rest frames (both are
             floored by the same min_rest_s rule, which is monotone). NOT `lp_licks.lp_lick_times`: its cam1 file
             (`<a>_<date>_events.csv`) holds POST-CUE events only (cue .. trial stop), so it misses the cam1 licks
             that sit in rest (coordinator 11-17, docs/status/LICK_SOURCE_AUDIT_20261117.md; the first version of this
             module used it and added ~nothing -- PS92_0814: 415 cam1-only onsets, 9 rest samples).
             A session without an events_all file gets cam4 only, and is LOGGED by name (`missing_cam1()`); a session
             without licks_all.csv has no RESTnl at all (unchanged, e.g. PS93_0813).

Selected by env `WIDEFIELD_RESTNL_LICKS` (so spawned workers inherit it) or `active("cam4cam1")` (sets the env for the
block). Consumers: `restnl_mask` (marked hook lines -- the default digest / cache names are unchanged), the derived map
reference "restnlc1" (`position_reference_maps`), `attempt_maps --rest-variant restnlc1`, `rest_frozen_decoder_nl --mask
restnlc1`, `scripts/restw_strict_check.py` + `scripts/restnl.py` (area-level RESTnl; folder `restw_check_c1`, suffix
`_restnlc1`), the interhemispheric rest set and the state-space rest baseline. Every cam4cam1 product has its own cache
dir / file stem / output name, so no cam4 output is ever overwritten or read as cam4cam1.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import sys

import numpy as np

ENV = "WIDEFIELD_RESTNL_LICKS"
SETS = ("cam4", "cam4cam1")
DEFAULT = "cam4"
#: name / tag of the cam4cam1 version (cache dirs, file stems, reference names, output suffixes)
TAG = "restnlc1"
NAME = "RESTnlc1"
#: cache / intermediate-folder token of THIS definition (v2 = cam1 events_all). The v1 (post-cue events.csv) files of
#: the first 14:31 timing run stay where they are (session_cache_restnlc1/, restw_check_c1/) and are never read.
CACHE_TAG = "restnlc1v3"
CHECK_DIR = "restw_check_c1v3"
TEXT = {"cam4": "DAQ spout-contact licks + cam4 LP video licks",
        "cam4cam1": "DAQ spout-contact licks + cam4 LP video licks + cam1-only LP events (water droplets and far-from-mouth labels removed)"}
SHORT = {"cam4": "DAQ + cam4", "cam4cam1": "DAQ + cam4 + cam1"}
VERSION = 3
#: cam1-only events whose median label position is this far BEHIND the mouth line (px, med_out_px) are static
#: tracking artifacts, not licks: Priya's blind review 11-17 scored 7 / 8 no-tongue test events below it and 0 / 30
#: tongue events (docs/status/CAM1_EXTRA_LICKS_CHECK_20261117.md). v3 = this guard; v2 outputs are superseded.
FAR_FROM_MOUTH_PX = -120.0

_MISSING: dict = {}


def lick_set() -> str:
    v = os.environ.get(ENV, DEFAULT).strip().lower() or DEFAULT
    if v not in SETS:
        raise ValueError(f"{ENV}={v!r}; expected one of {SETS}")
    return v


def is_c1() -> bool:
    return lick_set() == "cam4cam1"


@contextlib.contextmanager
def active(which: str):
    """Run a block under one lick set (env, so spawned workers see it too); restores the previous value."""
    if which not in SETS:
        raise ValueError(f"lick set {which!r} not in {SETS}")
    old = os.environ.get(ENV)
    os.environ[ENV] = which
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old


def text() -> str:
    return TEXT[lick_set()]


def cam1_events_path(label: str):
    """cam1 `events_all.csv` of a session (whole trial windows; may not exist)."""
    from pathlib import Path

    from wfield_local import config
    from wfield_local.paths import PathResolver
    an, d = label.split("_")
    return Path(PathResolver().root("dlc")) / "session_poses" / "cam1_licks" / f"{an}_{config.date8(d)}_events_all.csv"


def cam1_only_onsets(E) -> np.ndarray:
    """DAQ onsets (s) of the cam1-only, non-droplet, not-far-from-mouth (med_out_px >= FAR_FROM_MOUTH_PX) events of an events_all table (finite only, sorted)."""
    import pandas as pd
    m4 = E["cam4_matched"].astype(str).str.lower().eq("true").to_numpy()
    if "droplet" in E:
        dr = E["droplet"].astype(str).str.lower().eq("true").to_numpy()
    else:
        from wfield_local.cam1_licks import droplet_flags
        dr = droplet_flags(E, ~m4)
    far = np.zeros(len(E), bool)
    if "med_out_px" in E:
        far = (pd.to_numeric(E["med_out_px"], errors="coerce") < FAR_FROM_MOUTH_PX).to_numpy()
    t = pd.to_numeric(E["daq_on_s"], errors="coerce").to_numpy(float)[~m4 & ~dr & ~far]
    return np.sort(t[np.isfinite(t)])


def union_onsets(cam4_s, lp_s) -> tuple[np.ndarray, int]:
    """``(sorted unique union of the two trains in s, number of onsets the second train adds)``."""
    c4 = np.asarray(cam4_s, float)
    c4 = c4[np.isfinite(c4)]
    lp = np.asarray(lp_s, float)
    lp = lp[np.isfinite(lp)]
    u = np.union1d(c4, lp)
    return u, int(len(u) - len(np.unique(c4)))


def extend(label: str, cam4_s) -> tuple[np.ndarray, int]:
    """``(onsets in DAQ s, n cam1-only onsets added)`` -- the cam4cam1 train for one session: cam4 `licks_all.csv` UNION
    the cam1-only, non-droplet events of cam1 `events_all.csv`. Without that file the cam4 train is returned unchanged
    and the session is recorded in `missing_cam1()` (n added = -1)."""
    import pandas as pd
    f = cam1_events_path(label)
    if not f.exists():
        _MISSING[label] = "no cam1 events_all file"
        print(f"[restnlc1] {label}: no cam1 events_all -> cam4 only", flush=True)
        return np.asarray(cam4_s, float), -1
    E = pd.read_csv(f, usecols=lambda c: c in ("daq_on_s", "cam4_matched", "droplet", "spread_px", "range_px",
                                                "med_out_px"))
    return union_onsets(cam4_s, cam1_only_onsets(E))


def missing_cam1() -> dict:
    """{label: reason} of the sessions this process served cam4-only under cam4cam1."""
    return dict(_MISSING)


def code_digest() -> str:
    """This module's source: enters every cam4cam1 cache key."""
    return hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode()).hexdigest()[:12]


def key_part(label: str) -> str:
    """Per-session cache-key part of the cam4cam1 version: code + the cam1 events_all file identity."""
    from wfield_local import map_collect
    return f"c1v{VERSION}:{code_digest()}:{map_collect.stat_sig(cam1_events_path(label))}"
