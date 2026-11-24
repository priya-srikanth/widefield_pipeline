"""Six-position cohort (PS92-95): the angle correlations reported in the VLS paper (Fig 4F and the angle-latency
supplement), ported from the paper's manuscript_code (figures/make_farcontra_aim_hit.py, the Spearman part of
figures/c2_panels.py aimhit_single, and figures/c2_angle_latency.py; Priya 2026-11-21: every six-position analysis step
lives here so the 4-camera / 3-network 3D update regenerates the tables from repo code; the manuscript only plots).

  aim_hit        far-contralesional first-lick aim change (iti_licks aim_by_spout_sessions.csv: chg, deg vs the animal's
                 own pre, + = contralesional) joined by animal + session date to the far-contralesional hit rate
                 (behavior cohort_session_metrics.csv: hit__far_R), with hit_rel = hit rate / the animal's mean pre hit
                 rate. Per animal over post-stroke sessions: Spearman rho and p of aim vs hit rate. Recovered sessions
                 (post-stroke, hit_rel >= 0.9): number, per-animal means, the ANIMAL-WEIGHTED mean (mean of the animal
                 means; the manuscript's -7.3 deg) with an animal-then-session nested bootstrap 95% CI of that mean
                 (B = 5000, seed 0), and the session-pooled mean for reference.
  angle_latency  per animal over post-stroke sessions (iti_licks revup/revup_track_sessions.csv: aim, lat_med_ms):
                 Spearman rho raw, and rho / p after regressing both measures on log(post-stroke day, >= 1) within the
                 animal (the shared recovery trend removed).
Inputs are 3D-kinematic (aim) and must be regenerated with the 4-camera / 3-network tracking
(docs/MANUSCRIPT_C2_REFRESH.md).

Writes (refuses to overwrite) under --out-dir (default M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_correlations):
  farcontra_aim_hit_sessions.csv  aim_hit_per_animal.csv  aim_hit_recovered.csv  angle_latency_per_animal.csv
Usage: python scripts/c2_manuscript_correlations.py [--aim CSV] [--metrics CSV] [--revup CSV] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wfield_local import writeguard  # noqa: E402

SRV = Path("M:/MICROSCOPE/Priya")
# 11-22 (plan item 3 / runbook bug: steps 5 / 6 / 8 passed only --out-dir, so every input stayed v36 / two-camera):
# the DEFAULT follows the adopted run (`orofacial_cohort.RUN`, routed like the producers); pass it explicitly anyway
from wfield_local import orofacial_outputs as _oo  # noqa: E402
ITI = _oo.encoding_root() / "iti_licks"
METRICS = SRV / "Behavior_logs/Widefield/behavior_summary/cohort/cohort_session_metrics.csv"
OUT_DIR = SRV / "Widefield/manuscript_stats/c2_correlations"
REL_RECOVERED = 0.9                       # recovered = far-contra hit rate >= 90% of the animal's own pre mean
N_BOOT, SEED = 5000, 0


def aim_hit_table(aim_csv: Path, metrics_csv: Path) -> pd.DataFrame:
    """Join far-contra first-lick aim to far-contra hit rate per session (as make_farcontra_aim_hit.py)."""
    a = pd.read_csv(aim_csv)
    m = pd.read_csv(metrics_csv)
    a = a[a.pos == "far_contra"][["animal", "label", "epoch", "chg", "n", "day"]].copy()
    a["mmdd"] = a.label.str.split("_").str[1]
    m["mmdd"] = m.date.astype(str).str.replace("-", "").str[-4:]
    d = a.merge(m[["animal", "mmdd", "hit__far_R", "hitall__far_R"]], on=["animal", "mmdd"], how="left",
                validate="one_to_one").drop(columns="mmdd")
    d["hit_rel"] = d.hit__far_R / d.animal.map(d[d.epoch == "pre"].groupby("animal").hit__far_R.mean())
    return d


def nested_boot_mean(d: pd.DataFrame, col: str, rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of the animal-weighted mean: resample animals, then sessions within each drawn animal, and average the
    drawn animals' session means (each animal counts once, as in the manuscript's estimate)."""
    groups = [g[col].to_numpy(float) for _, g in d.groupby("animal")]
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draw = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        stats[b] = np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in draw])
    return tuple(np.percentile(stats, [2.5, 97.5]))


def angle_latency(revup_csv: Path) -> pd.DataFrame:
    """Per animal: Spearman aim vs first-lick latency, raw and detrended on log(day) (as c2_angle_latency.py)."""
    d = pd.read_csv(revup_csv)
    d = d[d.epoch != "pre"].dropna(subset=["aim", "lat_med_ms"])

    def resid(y, x):
        X = np.c_[np.ones_like(x), x]
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    rows = []
    for an, g in d.groupby("animal"):
        r0, p0 = spearmanr(g.aim, g.lat_med_ms)
        ld = np.log(g.day.clip(lower=1).to_numpy(float))
        r1, p1 = spearmanr(resid(g.aim.to_numpy(float), ld), resid(g.lat_med_ms.to_numpy(float), ld))
        rows.append({"animal": an, "n_sessions": len(g), "rho": r0, "p": p0, "rho_detrended": r1, "p_detrended": p1})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aim", type=Path, default=ITI / "aim_by_spout_sessions.csv")
    ap.add_argument("--metrics", type=Path, default=METRICS)
    ap.add_argument("--revup", type=Path, default=ITI / "revup" / "revup_track_sessions.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    outs = {k: a.out_dir / f"{k}.csv" for k in ("farcontra_aim_hit_sessions", "aim_hit_per_animal",
                                                "aim_hit_recovered", "angle_latency_per_animal")}
    clash = [p for p in outs.values() if p.exists()]
    if clash:
        sys.exit("refusing to overwrite: " + ", ".join(map(str, clash)))
    d = aim_hit_table(a.aim, a.metrics)
    post = d[d.epoch != "pre"]
    per = []
    for an, g in post.groupby("animal"):
        r, p = spearmanr(g.chg, g.hit__far_R)
        per.append({"animal": an, "n_post_sessions": len(g), "rho": r, "p": p})
    per = pd.DataFrame(per)
    rec = post[post.hit_rel >= REL_RECOVERED]
    am = rec.groupby("animal").chg.mean()
    lo, hi = nested_boot_mean(rec, "chg", np.random.default_rng(SEED))
    recov = pd.DataFrame([{"n_sessions": len(rec), "animal_weighted_mean_chg": am.mean(), "boot_lo": lo, "boot_hi": hi,
                           "session_pooled_mean_chg": rec.chg.mean(),
                           "animal_mean_min": am.min(), "animal_mean_max": am.max(),
                           **{f"mean_{k}": v for k, v in am.items()}}])
    al = angle_latency(a.revup)
    print(per.round(3).to_string(index=False), recov.round(2).to_string(index=False), al.round(3).to_string(index=False),
          sep="\n\n", flush=True)
    writeguard.assert_writable(a.out_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for k, df in zip(outs, (d, per, recov, al)):
        df.to_csv(outs[k], index=False)
    print("wrote", a.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
