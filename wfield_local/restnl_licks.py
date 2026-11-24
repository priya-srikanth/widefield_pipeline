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

  fused      ("RESTnlf", 11-22; Priya 11-21 Q1 = (b): the fused four-camera train defines the lick set for the
             lick-free windows too) -- DAQ + EVERY lick of the fused train (`session_licks`, every context; cam1-only
             and confirmed side-only licks are in it), replacing the cam4 table. The DEFAULT whenever a fused lick set is
             selected (`session_licks.fused_on()`) and the env does not name another set. Own cache dir / stem.

Selected by env `WIDEFIELD_RESTNL_LICKS` (so spawned workers inherit it) or `active("cam4cam1")` (sets the env for the
block); unset -> "fused" under a fused lick set, else "cam4". Consumers: `restnl_mask` (marked hook lines -- the default digest / cache names are unchanged), the derived map
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

from wfield_local import cam1_licks as _C1

ENV = "WIDEFIELD_RESTNL_LICKS"
SETS = ("cam4", "cam4cam1", "fused")
DEFAULT = "cam4"
#: the fused version's names (11-22)
FUSED_TAG = "restnlf"
FUSED_NAME = "RESTnlf"
FUSED_CACHE_TAG = "restnlfv1"
#: name / tag of the cam4cam1 version (cache dirs, file stems, reference names, output suffixes)
TAG = "restnlc1"
NAME = "RESTnlc1"
#: cache / intermediate-folder token of THIS definition (v2 = cam1 events_all). The v1 (post-cue events.csv) files of
#: the first 14:31 timing run stay where they are (session_cache_restnlc1/, restw_check_c1/) and are never read.
CACHE_TAG = "restnlc1v3"
CHECK_DIR = "restw_check_c1v3"
TEXT = {"cam4": "DAQ spout-contact licks + cam4 LP video licks",
        "cam4cam1": "DAQ spout-contact licks + cam4 LP video licks + cam1-only LP events (water droplets and far-from-mouth labels removed)",
        "fused": "DAQ spout-contact licks + every lick of the fused four-camera video train (cam4, cam1, cam2, cam3)"}
SHORT = {"cam4": "DAQ + cam4", "cam4cam1": "DAQ + cam4 + cam1", "fused": "DAQ + fused 4-camera"}
VERSION = 3
#: cam1-only events whose median label position is this far BEHIND the mouth line (px, med_out_px) are static
#: tracking artifacts, not licks: Priya's blind review 11-17 scored 7 / 8 no-tongue test events below it and 0 / 30
#: tongue events (docs/status/CAM1_EXTRA_LICKS_CHECK_20261117.md). v3 = this guard; v2 outputs are superseded.
#: Since 11-17 (late) the rule lives in ONE place, `cam1_licks.keep_mask` (droplet + far guard), shared by every cam1
#: event reader; this name re-exports its threshold.
FAR_FROM_MOUTH_PX = _C1.FAR_FROM_MOUTH_PX

_MISSING: dict = {}


def lick_set() -> str:
    from wfield_local import session_licks
    dflt = "fused" if session_licks.fused_on() else DEFAULT
    v = os.environ.get(ENV, dflt).strip().lower() or dflt
    if v not in SETS:
        raise ValueError(f"{ENV}={v!r}; expected one of {SETS}")
    return v


def is_c1() -> bool:
    return lick_set() == "cam4cam1"


def is_fused() -> bool:
    return lick_set() == "fused"


def fused_onsets(label: str) -> np.ndarray:
    """DAQ onsets (s) of every lick of the session's fused train (`session_licks.session_onsets_s`); raises
    `session_licks.SessionLicksUnavailable` when the session has none (-> no RESTnl, as without licks_all.csv)."""
    from wfield_local import session_licks
    return session_licks.session_onsets_s(label)


def fused_key_part(label: str) -> str:
    """Per-session cache-key part of the fused version: code + the fused train's identity (`session_licks.signature`)
    + the session's fused table."""
    from wfield_local import map_collect, session_licks
    an, d = session_licks._split(label)
    f = session_licks.fused_dir() / f"{an}_{d}_fused.csv"
    return f"fv1:{code_digest()}:{session_licks.signature()}:{map_collect.stat_sig(f)}"


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
    from wfield_local import config
    from wfield_local.paths import PathResolver
    an, d = label.split("_")
    from wfield_local import lp_licks          # 11-22: the adopted / routed cam1 folder (was the literal cam1_licks)
    return lp_licks.cam1_events_dir(PathResolver()) / f"{an}_{config.date8(d)}_events_all.csv"


def cam1_only_onsets(E, label: str | None = None) -> np.ndarray:
    """DAQ onsets (s) of the cam1-only (not cam4-matched) events of an events_all table that count as licks under
    `cam1_licks.keep_mask` (not a water droplet, not a far-from-mouth label; finite only, sorted)."""
    import pandas as pd
    m4 = _C1.flag_col(E["cam4_matched"])
    t = pd.to_numeric(E["daq_on_s"], errors="coerce").to_numpy(float)[~m4 & _C1.keep_mask(E, m4, label=label)]
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
    return union_onsets(cam4_s, cam1_only_onsets(E, label))


def missing_cam1() -> dict:
    """{label: reason} of the sessions this process served cam4-only under cam4cam1."""
    return dict(_MISSING)


def code_digest() -> str:
    """This module's source + the closure of `cam1_licks.keep_mask`: enters every cam4cam1
    cache key. (11-17: the move of the rule into `keep_mask` changed this digest, so RESTnlc1 caches made by the v3 code before it
    recompute once; an alias that kept them warm was dropped as fragile.)"""
    from wfield_local import map_collect
    src = inspect.getsource(sys.modules[__name__])
    src += map_collect.closure(_C1, "keep_mask").source()
    d = hashlib.sha1(src.encode()).hexdigest()[:12]
    return d


def key_part(label: str) -> str:
    """Per-session cache-key part of the cam4cam1 version: code + the cam1 events_all file identity."""
    from wfield_local import map_collect
    return f"c1v{VERSION}:{code_digest()}:{map_collect.stat_sig(cam1_events_path(label))}"
