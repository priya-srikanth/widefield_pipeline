"""RESTnl = the LICK-FREE ("no-lick") rest baseline adopted for the rest-referenced LEVEL results (Priya 10-29).

MASK (built by `scripts/restw_strict_check.py`, mask `strict`): the restdock05 rest mask (the RESTW mask) minus EVERY
cam4 lick of licks_all.csv (contact or not, any context) and every DAQ lick onset, each with a buffer of 0.5 s before /
2.0 s after; the mask's min_rest_s floor re-applied; engagement-gated rest frames
(`rest_by_position.rest_frames_by_position(docked=False, engaged_only=True)`); per position the MEDIAN over its rest
frames.

WEIGHTING RULE (the choice, 10-30; sweep = scripts/restnl_rescue.py, <iti_licks>/restw_check/rescue/):
  wr      (DEFAULT, rescue variant `s20_f50_wr`): equal-weight mean of the per-position medians over the positions
          with >= 50 frames, RENORMALIZED over the positions present; >= 4 of 6 positions required.
          Keeps 107 / 107 sessions (PS93 pre 7); headlines H1-H3 unchanged vs strict (rescue_headlines.csv).
  strict  (the 10-29 rule, `strict_w50`; select with `--restnl-rule strict` or env RESTNL_RULE=strict): equal-weight
          mean of all 6 per-position medians, all 6 required with >= 50 frames each. Keeps 97 / 107 (PS93 pre 2).
The 2.0 s post-lick buffer is kept in both: the rescue sweep's lick-triggered traces show the lick transient has not
decayed by 1.0-1.5 s, so the buffer was not shortened; only the all-6-positions requirement was relaxed.

This module only READS the per-session result `<iti_licks.OUT>/restw_check/sessions/<label>.npz` (`strict__posmed`,
`strict__nfr_pos`); the mask itself is rebuilt by `restw_strict_check.session_worker` (masks are never written to disk).
The scripts that use it take `--rest-variant restnl` (default `restw`, unchanged). Outputs: rule wr -> suffix `_restnl`
(these names held the strict rule until 10-30, superseded); rule strict -> suffix `_restnl_strict`. The per-session
caches that store the level are kept apart per rule via `cache()` (strict = the original dirs, wr = `<dir>_wr`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

# LICK SET (Priya 11-17, `wfield_local.restnl_licks`): WIDEFIELD_RESTNL_LICKS=cam4cam1 -> RESTnlc1 = the mask also
# removes the cam1-only LP events (droplets removed). Its levels come from restw_check_c1/ (restw_strict_check under the
# same switch), its outputs carry the suffix _restnlc1 and its per-session caches `<name>_wr_c1` / `<name>_c1`, so no
# cam4 RESTnl file is read as, or overwritten by, a cam4cam1 one. Scripts keep `--rest-variant restnl`.
from wfield_local import restnl_licks as _rl  # noqa: E402
C1 = _rl.is_c1()
CHECK = _rl.CHECK_DIR if C1 else "restw_check"
SRC = il.OUT / CHECK / "sessions"
SRC_VERSION = 3                 # restw_strict_check.VERSION the levels were built with
MIN_FR = 50
MIN_POS_WR = 4                  # rule wr: positions (of 6) with >= MIN_FR frames required
RULES = ("wr", "strict")
NAME = "RESTnlc1" if C1 else "RESTnl"


def rule() -> str:
    """`--restnl-rule wr|strict` from argv, else env RESTNL_RULE, else wr (the default since 10-30)."""
    r = sys.argv[sys.argv.index("--restnl-rule") + 1] if "--restnl-rule" in sys.argv else os.environ.get("RESTNL_RULE", "wr")
    if r not in RULES:
        raise SystemExit(f"--restnl-rule must be one of {RULES}, not {r!r}")
    return r


RULE = rule()
SFX = ("_restnlc1" if C1 else "_restnl") + ("" if RULE == "wr" else "_strict")
_WDEF = ("mean of the per-position medians over positions with >= 50 frames (>= 4 of 6, renormalized)" if RULE == "wr"
         else "position-weighted median, all 6 positions >= 50 frames")
DEF = ((f"{NAME} = no-lick rest: restdock05 rest minus every cam4 / cam1-only (droplets and far-from-mouth labels removed) / DAQ lick [0.5 s before, "
        "2.0 s after], engagement-gated, ") if C1 else
       "RESTnl = no-lick rest: restdock05 rest minus every cam4 / DAQ lick [0.5 s before, 2.0 s after], engagement-gated, "
       ) + _WDEF


def cache(name: str) -> str:
    """Per-session cache dir name for the active rule (strict = the original dir, wr = `<name>_wr`)."""
    return (name if RULE == "strict" else f"{name}_wr") + ("_c1" if C1 else "")


def variant() -> str:
    """`--rest-variant restw|restnl` from argv (default restw)."""
    v = sys.argv[sys.argv.index("--rest-variant") + 1] if "--rest-variant" in sys.argv else "restw"
    if v == "restnlc1":              # interhemispheric argv (11-17): RESTnl under the cam4cam1 lick set
        if not C1:
            raise SystemExit("--rest-variant restnlc1 needs WIDEFIELD_RESTNL_LICKS=cam4cam1 set before import")
        return "restnl"
    if v not in ("restw", "restnl"):
        raise SystemExit(f"--rest-variant must be restw or restnl, not {v!r}")
    return v


def level(label: str, cols=None, rule_: str | None = None, src: Path | None = None) -> tuple[np.ndarray, bool]:
    """(RESTnl level per column, ok) under the active rule (or `rule_`). `cols` (optional) = column names to return,
    in that order (NaN if absent). `src` = the restw_check sessions folder to read (default `SRC`, this run's); an LP
    lick-source run passes the lick-source-free one (`lick_precue_restw.rest_root`), since the RESTnl mask already
    removes every DAQ and cam4 lick whatever the run's lick source."""
    r = rule_ or RULE
    _f = Path(src or SRC) / f"{label}.npz"
    z = np.load(_f)
    from wfield_local import mask_keys                  # restw_check is mask-dependent: stale-read guard (11-09)
    mask_keys.check_obj(z, _f)
    if int(z["version"]) != SRC_VERSION:
        raise RuntimeError(f"{label}: restw_check session version {int(z['version'])} != {SRC_VERSION}")
    cs = [str(c) for c in z["cols"]]
    pm = np.asarray(z["strict__posmed"], float)
    npos = np.asarray(z["strict__nfr_pos"])
    if r == "strict":
        lev = pm.mean(0)
        ok = bool((npos >= MIN_FR).all())
    else:
        use = npos >= MIN_FR
        lev = pm[use].mean(0) if use.any() else np.full(pm.shape[1], np.nan)
        ok = bool(use.sum() >= MIN_POS_WR)
    ok = ok and bool(np.isfinite(lev).all())
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
