"""Cohort-2 (six-position) per-session 3-D lick summary, UNGATED -- the table behind VLS manuscript Fig 4B (first-lick
reach) and Fig 4E (all-lick peak speed), `c2_lick3d_session_summary_ungated.csv`. Until 2026-11-22 this table was made by
hand (11-19, no producing script); this script is its recipe, reverse-engineered from the hand-made copy and checked
against it cell for cell (DECISIONS 2026-11-22 "Hand-made manuscript tables").

Input: the 3-D lick table of `scripts/pose3d_kinematics.py` (``<pose3d_dir()>/kinematics/licks.csv.gz`` by default = the
cam4-window lick set of the adopted 3-D run; ``--kin-dir kinematics_fused`` for the fused four-camera train, or
``--licks PATH`` for any other table). One row per lick with ``trial_id``, ``lick_idx`` (0 = the trial's first post-cue
lick), ``position``, ``reach_mm``, ``side_ml``, ``speed_peak``, ``animal``, ``date``, ``session``, ``epoch``.

Recipe (one row per session x position; sorted by animal, date, position):
  * sessions: every session of the table except early pre sessions (`config.early_pre`: the August-only baseline,
    June and 0805 dropped); epoch = the table's own ``epoch``;
  * UNGATED: every lick row of the table -- no working-trial gate, no engagement gate, no camera-class gate, no
    minimum lick count (the panels apply their own n >= 20 cut: `c2_scalar_family_aligned.lick3d`);
  * first-lick set = rows with ``lick_idx == 0``: ``n_first`` = their count, ``reach_first`` / ``side_first`` /
    ``speed_first`` = their medians of ``reach_mm`` / ``side_ml`` / ``speed_peak`` (median skips NaN speed);
  * ``speed_first5`` = median over trials of the per-trial mean ``speed_peak`` of licks ``lick_idx < 5``;
    ``n_trials5`` = the number of trials with any such lick;
  * all-lick set = every row: ``n_all`` = count, ``reach_all`` / ``side_all`` / ``speed_all`` = medians.
Units as the input (mm, mm/s); ``side_ml`` keeps the input sign (+ = mouse right). No statistics here: the session
changes and their tests are made by `scripts/c2_scalar_family_aligned.py` (``--lick3d``).

Per-animal table first (CLAUDE.md rule 8): epoch means of the session medians, printed before anything is written.

Usage:
  python scripts/c2_lick3d_session_summary.py [--kin-dir kinematics|kinematics_fused | --licks PATH] [--out CSV]
                                              [--force]
Default output: ``c2_lick3d_session_summary_ungated.csv`` next to the input licks table. Refuses to overwrite unless
``--force``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

NAME = "c2_lick3d_session_summary_ungated.csv"
KEY = ["animal", "date", "session", "epoch", "position"]
COLS = KEY + ["n_first", "reach_first", "side_first", "speed_first", "speed_first5", "n_trials5", "n_all", "reach_all",
              "side_all", "speed_all", "n_first_rows", "n_all_rows"]
FIRST5 = 5


def default_licks(kin_dir: str | None = None) -> Path:
    """``<pose3d_dir()>/<kin_dir>/licks.csv.gz``; default folder = `session_licks.kin_name()` (11-22: the fused
    four-camera post-cue table, ``kinematics_fused_postcue``; ``WIDEFIELD_KIN_LICKS=cam4`` or ``--kin-dir kinematics``
    for the cam4 windows)."""
    from wfield_local import session_licks as SL
    from wfield_local.pose3d import pose3d_dir
    return pose3d_dir() / (kin_dir or SL.kin_name()) / "licks.csv.gz"


def summarise(L: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    """The session x position summary of one 3-D lick table (see the module docstring). ``stat`` (11-23, Priya: session
    MEANS over licks, as the angles and the two-tone cohort): "median" (the 11-19 recipe, default) or "mean" -- the
    per-session statistic of every lick measure (and of the per-trial first-5 means); counts unchanged."""
    agg = (lambda x: x.median()) if stat == "median" else (lambda x: x.mean())
    from wfield_local import config
    L = L[~((L.epoch == "pre") & L.date.astype(str).map(config.early_pre))]
    rows = []
    for key, d in L.groupby(KEY, sort=True):
        f = d[d.lick_idx == 0]
        t5 = d[d.lick_idx < FIRST5].groupby("trial_id").speed_peak.mean()
        # n_first / n_all = licks WITH 3-D kinematics (finite reach): on the cam4 tables every row had them, so this is
        # the recipe's n; on the fused post-cue tables a lick without 3-D support keeps its row with NaN measures, and
        # n_first_rows / n_all_rows count every lick (11-22: counts reported both ways)
        rows.append(dict(zip(KEY, key), n_first=int(f.reach_mm.notna().sum()), reach_first=agg(f.reach_mm),
                         side_first=agg(f.side_ml), speed_first=agg(f.speed_peak), speed_first5=agg(t5),
                         n_trials5=len(t5), n_all=int(d.reach_mm.notna().sum()), reach_all=agg(d.reach_mm),
                         side_all=agg(d.side_ml), speed_all=agg(d.speed_peak), n_first_rows=len(f),
                         n_all_rows=len(d)))
    S = pd.DataFrame(rows, columns=COLS)
    return S.sort_values(["animal", "date", "position"], kind="stable").reset_index(drop=True)


def per_animal(S: pd.DataFrame) -> pd.DataFrame:
    cols = ["reach_first", "speed_first", "speed_all", "reach_all"]
    return (S.groupby(["animal", "epoch"])[cols].mean().unstack("epoch")
            .reindex(columns=["pre", "acute", "subacute", "chronic"], level=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--kin-dir", default=None,
                   help="3-D kinematics folder under pose3d_dir() (default session_licks.kin_name(): "
                        "kinematics_fused_postcue = the fused four-camera train, post-cue; kinematics = cam4 windows, "
                        "the manuscript's 11-19 recipe). NOT kinematics_fused: it holds pre-cue / ITI licks too")
    g.add_argument("--licks", type=Path, default=None, help="explicit 3-D lick table (licks.csv.gz)")
    ap.add_argument("--out", type=Path, default=None, help=f"output CSV (default: <licks folder>/{NAME})")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--stat", choices=("median", "mean"), default="median",
                    help="per-session statistic over licks (11-23: 'mean' = the session mean, Priya)")
    a = ap.parse_args(argv)
    src = a.licks or default_licks(a.kin_dir)
    out = a.out or src.parent / NAME
    if out.exists() and not a.force:
        sys.exit(f"refusing to overwrite {out} (--force)")
    L = pd.read_csv(src, usecols=["trial_id", "lick_idx", "position", "reach_mm", "side_ml", "speed_peak", "animal",
                                  "date", "session", "epoch"])
    S = summarise(L, a.stat)
    pd.set_option("display.width", 220)
    print(f"input {src}: {len(L)} licks -> {len(S)} session x position rows, {S.session.nunique()} sessions")
    print(f"== per animal (rule 8): epoch means of the session x position {a.stat}s (mm, mm/s)")
    print(per_animal(S).round(2).to_string())
    out.parent.mkdir(parents=True, exist_ok=True)
    S.to_csv(out, index=False, lineterminator="\n")   # LF line ends, as the hand-made copy (else CRLF on Windows)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
