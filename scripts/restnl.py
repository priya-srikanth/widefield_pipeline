"""RESTnl = the LICK-FREE ("no-lick") rest baseline adopted for the rest-referenced LEVEL results (Priya 10-29).

Definition (built by `scripts/restw_strict_check.py`, variant `strict_w50`): the restdock05 rest mask (the RESTW mask)
minus EVERY cam4 lick of licks_all.csv (contact or not, any context) and every DAQ lick onset, each with a buffer of
0.5 s before / 2.0 s after; the mask's min_rest_s floor re-applied; engagement-gated rest frames
(`rest_by_position.rest_frames_by_position(docked=False, engaged_only=True)`); per position the MEDIAN over its rest
frames, the six positions averaged with equal weight (RESTW's weighting), all 6 positions required with >= 50 frames
each (RESTW: 200). 97 / 107 sessions qualify; the other 10 are left OUT of every RESTnl statistic.

This module only READS the per-session result `<iti_licks.OUT>/restw_check/sessions/<label>.npz` (`strict__posmed`,
`strict__nfr_pos`); the mask itself is rebuilt by `restw_strict_check.session_worker` (masks are never written to disk).
The scripts that use it take `--rest-variant restnl` (default `restw`, unchanged) and write `_restnl` file names.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

SRC = il.OUT / "restw_check" / "sessions"
SRC_VERSION = 3                 # restw_strict_check.VERSION the levels were built with
MIN_FR = 50
SFX = "_restnl"
NAME = "RESTnl"
DEF = ("RESTnl = no-lick rest: restdock05 rest minus every cam4 / DAQ lick [0.5 s before, 2.0 s after], engagement-gated, "
       "position-weighted median (>= 50 frames per position)")


def variant() -> str:
    """`--rest-variant restw|restnl` from argv (default restw)."""
    v = sys.argv[sys.argv.index("--rest-variant") + 1] if "--rest-variant" in sys.argv else "restw"
    if v not in ("restw", "restnl"):
        raise SystemExit(f"--rest-variant must be restw or restnl, not {v!r}")
    return v


def level(label: str, cols=None) -> tuple[np.ndarray, bool]:
    """(RESTnl level per column, ok). `cols` (optional) = column names to return, in that order (NaN if absent)."""
    z = np.load(SRC / f"{label}.npz")
    if int(z["version"]) != SRC_VERSION:
        raise RuntimeError(f"{label}: restw_check session version {int(z['version'])} != {SRC_VERSION}")
    cs = [str(c) for c in z["cols"]]
    lev = np.asarray(z["strict__posmed"], float).mean(0)
    ok = bool((np.asarray(z["strict__nfr_pos"]) >= MIN_FR).all()) and bool(np.isfinite(lev).all())
    if cols is None:
        return lev, ok
    return np.array([lev[cs.index(c)] if c in cs else np.nan for c in cols], float), ok


def relabel(txt: str) -> str:
    """Figure text written for RESTW -> the RESTnl wording."""
    for a, b in (("position-weighted flat rest median, restdock05, engagement-gated", DEF),
                 ("(position-weighted flat rest median, restdock05, engagement-gated)", f"({DEF})"),
                 ("RESTW", NAME), ("restw", NAME), ("Restw", NAME)):
        txt = txt.replace(a, b)
    return txt
