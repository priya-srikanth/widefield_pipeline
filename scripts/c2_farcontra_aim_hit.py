"""Fig 4F TABLES (six-position cohort): far-contralesional first-lick aim against far-contralesional hit rate.

    python scripts/c2_farcontra_aim_hit.py --aim <dir with aim_by_spout_sessions.csv> --out <dir>
        [--metrics <cohort_session_metrics.csv>]

Session table = the paper's `manuscript_code/figures/make_farcontra_aim_hit.py` recipe, reproduced here so it can run on
any aim table: far-contra rows of aim_by_spout_sessions.csv (chg = first-lick aim change vs own pre, deg, + = contra)
joined by animal + MMDD to the DAQ behavior pipeline's cohort_session_metrics.csv (hit__far_R, hitall__far_R);
hit_rel = hit__far_R / the animal's mean pre-stroke hit__far_R.
Stats (as the Fig 4F legend): Spearman rho (and p) of chg vs hit__far_R per animal over its POST-stroke sessions
(`c2_panels.aimhit_single`); in post-stroke sessions with hit_rel >= 0.9, the mean residual chg with a nested bootstrap
(animals, then sessions within animal; animal-weighted mean of per-animal means; 5,000 draws, seed 0) and the
per-animal means. Tables only (the plot is made on the other machine).
Writes OUT/farcontra_aim_hit_sessions.csv, OUT/farcontra_aim_hit_stats.csv (refuses to overwrite unless --force).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield/behavior_summary/cohort/cohort_session_metrics.csv")
NB = 5000


def session_table(aim_dir: Path, metrics: Path) -> pd.DataFrame:
    a = pd.read_csv(aim_dir / "aim_by_spout_sessions.csv")
    m = pd.read_csv(metrics)
    a = a[a.pos == "far_contra"][["animal", "label", "epoch", "chg", "n", "day"]].copy()
    a["mmdd"] = a.label.str.split("_").str[1]
    m["mmdd"] = m.date.astype(str).str.replace("-", "").str[-4:]
    d = a.merge(m[["animal", "mmdd", "hit__far_R", "hitall__far_R"]], on=["animal", "mmdd"], how="left").drop(columns="mmdd")
    d["hit_rel"] = d.hit__far_R / d.animal.map(d[d.epoch == "pre"].groupby("animal").hit__far_R.mean())
    return d


def nested_boot(d: pd.DataFrame, col: str, rng) -> np.ndarray:
    groups = [g[col].to_numpy() for _, g in sorted(d.groupby("animal"))]
    out = np.empty(NB)
    for b in range(NB):
        pick = rng.integers(len(groups), size=len(groups))
        out[b] = np.mean([rng.choice(groups[i], size=len(groups[i])).mean() for i in pick])
    return out


def stats_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    post = d[d.epoch != "pre"].dropna(subset=["chg", "hit__far_R"])
    for an, g in sorted(post.groupby("animal")):
        r, p = stats.spearmanr(g.chg, g.hit__far_R)
        rows.append(dict(stat="spearman_rho_post", animal=an, value=r, p=p, n=len(g)))
    rec = post[post.hit_rel >= 0.9]
    for an, g in sorted(rec.groupby("animal")):
        rows.append(dict(stat="recovered_mean_chg", animal=an, value=g.chg.mean(), n=len(g)))
    bs = nested_boot(rec, "chg", np.random.default_rng(0))
    rows.append(dict(stat="recovered_mean_chg", animal="pooled", value=rec.groupby("animal").chg.mean().mean(),
                     lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n=len(rec)))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aim", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, default=METRICS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    f1, f2 = a.out / "farcontra_aim_hit_sessions.csv", a.out / "farcontra_aim_hit_stats.csv"
    for f in (f1, f2):
        if f.exists() and not a.force:
            sys.exit(f"refusing to overwrite {f} (--force)")
    d = session_table(a.aim, a.metrics)
    T = stats_table(d)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    a.out.mkdir(parents=True, exist_ok=True)
    d.to_csv(f1, index=False)
    T.to_csv(f2, index=False)
    print("wrote", f1, f2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
