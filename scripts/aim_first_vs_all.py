"""Aim offset by spout x epoch, FIRST cued lick vs ALL cued licks side by side -- the tables behind VLS manuscript Fig 4D
(all-lick aim row), `iti_licks/aim_first_vs_all_{sessions,tests}.csv`. Made by hand on 2026-10-30 (written next to
`aim_by_spout_*` from the same session, no producing script was committed); this script is its recipe, reverse-engineered
from the hand-made copy and checked against it cell for cell (DECISIONS 2026-11-22 "Hand-made manuscript tables").

Aim = THE aim definition (`iti_recovery_measure.aim_licks`, CLAUDE.md rule 9): per cued lick of the where-decoder lick
table, 3-D angle atan2(lateral, sagittal) at the extension peak minus the animal's mean over ALL its pre cued licks at
that spout (deg; sign as `lick_cued_aim_by_spout`, + = contralesional); ``first`` = the trial's first post-cue lick.

sessions table, one row per session x spout (animal, label, epoch, pos), each lick set on its own:
  * FIRST set (first == True), kept when >= `iti_recovery_measure.MIN_LICKS["first"]` (5) licks: ``mean_first``,
    ``median_first``, ``n_first``;
  * ALL set (every cued lick), kept when >= `MIN_LICKS["all"]` (10): ``mean_all``, ``median_all``, ``n_all``;
  * a row needs both sets (inner join);
  * ``chg_<set>`` = ``mean_<set>`` minus the animal's mean of its pre-session ``mean_<set>`` at that spout (change vs
    own pre); ``diff`` = ``chg_first - chg_all``.
tests table, one row per spout x epoch (pre = the pre sessions vs their own mean, the noise floor):
  * ``chg_first`` / ``chg_all`` / ``diff`` = animal-weighted mean (mean of per-animal means of the session values);
  * 95% CI of each = `lick_cued_aim_by_spout.boot` (nested animals -> sessions, animal-weighted, 2,000 draws), three
    INDEPENDENT bootstraps per cell in the order chg_first, chg_all, diff, ONE rng (default_rng(0)) shared over cells
    in spout (canonical) x epoch order -- the draw order is load-bearing for reproducing the CIs;
  * ``sig`` = post epoch and the ``diff`` CI excludes 0; ``sig_bonf`` = the same on the Bonferroni CI over 18 tests
    (6 spouts x 3 post epochs, the `lick_cued_aim_by_spout` bar); ``diff_<animal>`` = each animal's mean ``diff``.
The manuscript's Fig 4D stars do NOT come from these CIs: `scripts/c2_scalar_family_aligned.py` re-tests ``chg_all``
in the one cohort-2 scalar family (MWU + `analysis_kit.boot_delta`; `aim_first_vs_all_tests_aligned.csv`).
Bootstrap note (rule 9): this reuses the committed `lick_cued_aim_by_spout.boot`, the draw the hand-made table used;
analysis_kit has no animal-weighted LEVEL bootstrap with this draw order (`boot_ci` is flat-pool).

Per-animal table first (CLAUDE.md rule 8): ``diff`` / ``chg_all`` per animal x spout x epoch before the bootstrap.

Usage:
  python scripts/aim_first_vs_all.py [--preds where_preds_z<tag>.csv --first-preds first_lick_preds<tag>.csv.gz]
                                     [--out DIR] [--force]
Inputs default to this run's where-decoder caches (`iti_licks.pred_file`, under the adopted run's routed iti_licks);
outputs default to `iti_licks.OUT_W` (the routed iti_licks folder `c2_scalar_family_aligned --aim` reads). Refuses to
overwrite either output unless ``--force``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_recovery_measure as rm  # noqa: E402
from lick_cued_aim_by_spout import boot  # noqa: E402

AN, EPS, POS = rm.AN, rm.EPS, rm.POS
KEY = ["animal", "label", "epoch", "pos"]
SETS = ("chg_first", "chg_all", "diff")
N_TESTS = 18                                   # 6 spouts x 3 post epochs (lick_cued_aim_by_spout's Bonferroni bar)


def _set_table(q: pd.DataFrame, min_n: int, sfx: str) -> pd.DataFrame:
    g = q.groupby(KEY).off.agg(["mean", "median", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    base = g[g.epoch == "pre"].groupby(["animal", "pos"])["mean"].mean()
    g["chg"] = g["mean"] - base.reindex(pd.MultiIndex.from_arrays([g.animal, g.pos])).to_numpy()
    return g.rename(columns={"mean": f"mean_{sfx}", "median": f"median_{sfx}", "size": f"n_{sfx}", "chg": f"chg_{sfx}"})


def session_table(p: pd.DataFrame) -> pd.DataFrame:
    S = _set_table(p[p["first"]], rm.MIN_LICKS["first"], "first").merge(_set_table(p, rm.MIN_LICKS["all"], "all"),
                                                                         on=KEY)
    S["diff"] = S.chg_first - S.chg_all
    return S


def tests(S: pd.DataFrame, n_boot: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    T = []
    for pos in POS:
        for ep in EPS:
            d = S[(S.pos == pos) & (S.epoch == ep)]
            if d.empty:
                continue
            r = dict(pos=pos, epoch=ep, n_animals=d.animal.nunique(), n_sessions=len(d))
            for c in SETS:
                bs = boot(d.assign(chg=d[c]), rng, n_boot)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                r.update({c: d.groupby("animal")[c].mean().mean(), f"{c}_lo": lo, f"{c}_hi": hi})
                if c == "diff":
                    lob, hib = np.percentile(bs, [100 * 0.025 / N_TESTS, 100 * (1 - 0.025 / N_TESTS)])
                    r["sig"] = ep != "pre" and (lo > 0 or hi < 0)
                    r["sig_bonf"] = ep != "pre" and (lob > 0 or hib < 0)
            r.update({f"diff_{a}": d[d.animal == a]["diff"].mean() for a in AN})
            T.append(r)
    return pd.DataFrame(T)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, default=None, help="where-decoder lick table (default: il.pred_file cache)")
    ap.add_argument("--first-preds", type=Path, default=None, help="first-lick table (default: il.pred_file cache)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: il.OUT_W, the routed iti_licks)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="bootstrap draws per cell and set (default lick_cued_aim_by_spout.NB = 2000; same seed)")
    a = ap.parse_args(argv)
    out = a.out or il.OUT_W
    targets = {k: out / f"aim_first_vs_all_{k}.csv" for k in ("sessions", "tests")}
    for f in targets.values():
        if f.exists() and not a.force:
            sys.exit(f"refusing to overwrite {f} (--force)")
    p = rm.aim_licks(a.preds, a.first_preds)
    S = session_table(p)
    pd.set_option("display.width", 220)
    print(f"{len(p)} cued licks -> {len(S)} session x spout rows, {S.label.nunique()} sessions")
    print("== per animal first (rule 8): mean session diff (first - all) and chg_all, deg vs own pre")
    A = S.groupby(["pos", "epoch", "animal"])[["diff", "chg_all"]].mean().unstack("animal")
    print(A.reindex([(p_, e) for p_ in POS for e in EPS if (p_, e) in A.index]).round(1).to_string())
    T = tests(S, a.n_boot)
    print("\n== pooled (animal-weighted), 95% CI; sig = diff CI excludes 0 (post epochs)")
    print(T[["pos", "epoch", "chg_first", "chg_all", "diff", "diff_lo", "diff_hi", "sig", "sig_bonf"]].round(2)
          .to_string(index=False))
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(targets["sessions"], index=False)
    T.to_csv(targets["tests"], index=False)
    print("wrote", *targets.values())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
