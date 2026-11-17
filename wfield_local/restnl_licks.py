"""WHICH VIDEO LICKS LEAVE THE RESTnl REST MASK -- the RESTnl lick-set switch (Priya 11-17).

Priya 11-17: lick-free rest (RESTnl) should exclude DAQ + cam4 + cam1 licks (cam1-only events, water droplets removed),
the same lick train as the within-trial lick-free windows. Before 11-17 RESTnl removed DAQ licks and cam4 `licks_all.csv`
licks only. Both versions stay available:

  cam4       (DEFAULT, the RESTnl of every output up to 11-17) -- DAQ lick onsets + cam4 `licks_all.csv` (`daq_on_s`).
  cam4cam1   ("RESTnlc1") -- DAQ + cam4 + the cam1-only events: the LP lick train `lp_licks.lp_lick_times` (cam4
             licks_all every context + cam1 events that are NOT cam4-matched and NOT water droplets, inside the camera
             trial windows). Its union with the cam4 table read by the mask, so the cam1 version removes a SUPERSET of
             the cam4 version's rest frames (both are floored by the same min_rest_s rule, which is monotone).
             A session without cam1 events (or without the LP trial table) gets cam4 only, and is LOGGED by name
             (`missing_cam1()`); a session without licks_all.csv has no RESTnl at all (unchanged, e.g. PS93_0813).

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
TEXT = {"cam4": "DAQ spout-contact licks + cam4 LP video licks",
        "cam4cam1": "DAQ spout-contact licks + cam4 LP video licks + cam1-only LP events (water droplets removed)"}
SHORT = {"cam4": "DAQ + cam4", "cam4cam1": "DAQ + cam4 + cam1"}
VERSION = 1

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
    from wfield_local import lp_licks
    from wfield_local.paths import PathResolver
    return lp_licks._inputs(label, PathResolver())[2]


def union_onsets(cam4_s, lp_s) -> tuple[np.ndarray, int]:
    """``(sorted unique union of the two trains in s, number of onsets the LP train adds)``. The LP train already
    holds the cam4 table, so the added count is the cam1-only events (cam4-matched cam1 events are not in it)."""
    c4 = np.asarray(cam4_s, float)
    c4 = c4[np.isfinite(c4)]
    lp = np.asarray(lp_s, float)
    lp = lp[np.isfinite(lp)]
    u = np.union1d(c4, lp)
    return u, int(len(u) - len(np.unique(c4)))


def extend(label: str, cam4_s) -> tuple[np.ndarray, int]:
    """``(onsets in DAQ s, n cam1-only onsets added)`` -- the cam4cam1 train for one session: cam4 `licks_all.csv` UNION
    `lp_licks.lp_lick_times` (cam4 + cam1-only, droplets removed). Without cam1 events (no events file, or no LP trial
    table) the cam4 train is returned unchanged and the session is recorded in `missing_cam1()` (n added = -1)."""
    from wfield_local import lp_licks
    if not cam1_events_path(label).exists():
        _MISSING[label] = "no cam1 events file"
        print(f"[restnlc1] {label}: no cam1 events -> cam4 only", flush=True)
        return np.asarray(cam4_s, float), -1
    try:
        lp_s, _cue = lp_licks.lp_lick_times(label)
    except lp_licks.LPUnavailable as ex:
        _MISSING[label] = f"LP tables unavailable: {str(ex)[:80]}"
        print(f"[restnlc1] {label}: LP lick tables unavailable -> cam4 only ({ex})", flush=True)
        return np.asarray(cam4_s, float), -1
    return union_onsets(cam4_s, lp_s)


def missing_cam1() -> dict:
    """{label: reason} of the sessions this process served cam4-only under cam4cam1."""
    return dict(_MISSING)


def code_digest() -> str:
    """This module + the LP train's definition (`lp_licks._read_times`, LP_VERSION): enters every cam4cam1 cache key."""
    from wfield_local import lp_licks
    src = inspect.getsource(sys.modules[__name__]) + inspect.getsource(lp_licks._read_times) + repr(lp_licks.LP_VERSION)
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def key_part(label: str) -> str:
    """Per-session cache-key part of the cam4cam1 version: code + the cam1 events file identity."""
    from wfield_local import map_collect
    return f"c1v{VERSION}:{code_digest()}:{map_collect.stat_sig(cam1_events_path(label))}"
