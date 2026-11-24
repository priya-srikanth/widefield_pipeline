"""Six-position within-bout inter-lick interval with the cross-cohort definition (caption review item 29, CC-B 2026-11-23).

Priya (item 29): one ILI definition for both cohorts = the session MEDIAN of the within-bout intervals, a within-bout
interval being the interval between consecutive licks of the same trial that is <= 300 ms. The six-position S10C value
(`fourcam_lick_metrics.session_rows`) is already the session x position median of the pooled within-trial intervals
between consecutive post-cue fused-lick peaks, but with ``< 300`` ms. Here only the bound changes to ``<= 300`` ms
(peaks are on the 4-ms camera grid, so an interval of exactly 300 ms occurs). Same licks (fused four-camera train,
post_cue context = peak >= cue + 70 ms and before the trial stop, cam1-only licks excluded), same trial gate
(`orofacial_cohort.cohort_keep`, camera working period), same sessions and positions.

Check first: the ``< 300`` value recomputed here equals the stored ``ili_ms`` of position_metrics_fourcam.csv for every
row. Tests: `orofacial_stats.session_change_tests` with the cohort's metric list, the other two fused metrics as stored,
n_boot = 4000, exactly as fourcam_lick_metrics.main; then the ILI p values are swapped into the six-position scalar
family (`--family`) and BH is recomputed over the whole family (checked to reproduce the stored q_family first).

Writes (refuses to overwrite) under --out-dir: c2_ili_le300_position_metrics.csv, c2_ili_le300_stats.csv,
c2_ili_le300_family_old_vs_new.csv.
Usage: python scripts/c2_ili_le300.py --out-dir DIR --fourcam-dir <lick_metrics_boot4000> --family <c2_scalar_family_aligned.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ILI_MAX_MS = 300.0
N_BOOT = 4000


def session_ili(job) -> list[dict]:
    """fan_sessions worker: per position, the median within-trial interval, < 300 (old) and <= 300 (new)."""
    animal, date = job
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import session_licks as SL
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    sd = oo.session_dir(rv, oc.RUN, animal, date)
    keep, _gate = oc.cohort_keep(animal, date, sd)
    A = SL.session_licks(f"{animal}_{date}", rv, include_cam1_only=True)
    A = A[(A.context == "post_cue") & (A.cams.astype(str) != "cam1")]
    if keep is not None:
        A = A[A.trial_id.astype(int).isin(keep)]
    rows = []
    for pos, g in A.groupby("position"):
        g = g.sort_values(["trial_id", "t_ms"])
        ili = g.groupby("trial_id").t_ms.diff().dropna()
        lt, le = ili[ili < ILI_MAX_MS], ili[ili <= ILI_MAX_MS]
        rows.append({"animal": animal, "date": date, "position": pos,
                     "ili_lt300": float(lt.median()) if len(lt) else np.nan,
                     "ili_le300": float(le.median()) if len(le) else np.nan,
                     "n_int_lt300": int(len(lt)), "n_int_le300": int(len(le)), "n_int_eq300": int((ili == ILI_MAX_MS).sum())})
    return rows


def bh(p: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_stats as st
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fourcam-dir", type=Path, required=True)
    ap.add_argument("--family", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    outs = {k: a.out_dir / f"{k}.csv" for k in ("c2_ili_le300_position_metrics", "c2_ili_le300_stats",
                                                 "c2_ili_le300_family_old_vs_new")}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash:
        sys.exit("refusing to overwrite " + ", ".join(clash))
    tables = oc.cohort_dir("tables", run=oc.RUN, rv=PathResolver())
    if not (tables / "position_metrics.csv").exists():
        tables = tables / "prepost_working"
    Pm = pd.read_csv(tables / "position_metrics.csv", dtype={"date": str})
    P4 = pd.read_csv(a.fourcam_dir / "position_metrics_fourcam.csv", dtype={"date": str})
    stored = pd.read_csv(a.fourcam_dir / "change_vs_pre_stats_contra_fourcam.csv")
    assert (P4[["animal", "date", "position"]].to_numpy() == Pm[["animal", "date", "position"]].to_numpy()).all()
    sess = sorted({(r.animal, r.date) for r in Pm.itertuples()})
    res, fails = ak.fan_sessions(sess, session_ili, jobs=a.jobs, key=lambda j: f"{j[0]}_{j[1]}")
    for j, e in fails:
        print(f"{j}: FAILED {e}")
    F = pd.DataFrame([r for _, rr in res for r in rr])
    P = P4.merge(F, on=["animal", "date", "position"], how="left")
    d = (P.ili_lt300 - P.ili_ms).abs()
    print(f"check < 300 vs stored ili_ms: rows {len(P)}, max |diff| {np.nanmax(d):.3g}, "
          f"NaN mismatch {int((P.ili_lt300.isna() != P.ili_ms.isna()).sum())}")
    if np.nanmax(d) > 1e-6 or (P.ili_lt300.isna() != P.ili_ms.isna()).any():
        sys.exit("recomputed < 300 ILI does not reproduce the stored table; nothing written")
    print(f"intervals of exactly 300 ms: {int(P.n_int_eq300.sum())} of {int(P.n_int_le300.sum())}; rows whose median "
          f"changes: {int((P.ili_le300 != P.ili_lt300).sum())}")
    ms_ = list(dict.fromkeys([*oc.COHORT_SCALARS, *(m for g in oc.POS_GROUPS.values() for m in g)]))
    Pt = Pm.copy()
    for m in ("latency_ms", "licks_per_resp", "ili_ms"):
        Pt[m] = P4[m].to_numpy()
    R_old = st.session_change_tests(Pt, ms_, n_boot=N_BOOT)
    key = ["metric", "pos", "epoch"]
    c = R_old[R_old.metric == "ili_ms"].merge(stored[stored.metric == "ili_ms"], on=key, suffixes=("", "_st"))
    print("replay of the stored fused ILI rows: max |d boot_change|",
          float(np.nanmax(np.abs(c.boot_change - c.boot_change_st))), "agree identical", bool((c.agree == c.agree_st).all()))
    Pt["ili_ms"] = P.ili_le300.to_numpy()
    R = st.session_change_tests(Pt, ms_, n_boot=N_BOOT)
    R = R[R.metric == "ili_ms"].reset_index(drop=True)
    fam = pd.read_csv(a.family)
    q0 = bh(fam.p.to_numpy(float))
    print("family BH reproduces stored q_family: max |dq|", float(np.nanmax(np.abs(q0 - fam.q_family))))
    new = fam.copy()
    sel = new.metric == "ili_ms"
    rr = R.set_index(["pos", "epoch"])
    for i in new.index[sel]:
        k = (new.at[i, "pos"], new.at[i, "epoch"])
        for col in ("p", "boot_change", "boot_lo", "boot_hi", "mean_change", "n_sessions", "n_pre_sessions"):
            if col in rr.columns:
                new.at[i, col] = rr.at[k, col]
    new["q_family"] = bh(new.p.to_numpy(float))
    new["boot_sig"] = (new.boot_lo > 0) | (new.boot_hi < 0)
    new["star_new"] = np.where((new.q_family < 0.05) & new.boot_sig, "**", "")
    old_star = np.where((fam.q_family < 0.05) & ((fam.boot_lo > 0) | (fam.boot_hi < 0)), "**", "")
    cmp = fam[["metric", "pos", "epoch", "boot_change", "boot_lo", "boot_hi", "q_family", "star_aligned"]].assign(
        star_old_recomputed=old_star).join(new[["boot_change", "boot_lo", "boot_hi", "q_family", "star_new"]],
                                           rsuffix="_new")
    cmp["change"] = np.where((cmp.star_old_recomputed == "**") & (cmp.star_new == ""), "LOST",
                             np.where((cmp.star_old_recomputed == "") & (cmp.star_new == "**"), "GAINED", ""))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    P.to_csv(outs["c2_ili_le300_position_metrics"], index=False)
    R.to_csv(outs["c2_ili_le300_stats"], index=False)
    cmp.to_csv(outs["c2_ili_le300_family_old_vs_new"], index=False)
    pd.set_option("display.width", 250)
    print("star_aligned == recomputed old star:", bool((cmp.star_aligned.fillna("") == cmp.star_old_recomputed).all()))
    print(cmp[cmp.metric == "ili_ms"].round(3).to_string(index=False))
    print("other cells changing:", cmp[(cmp.metric != "ili_ms") & (cmp.change != "")][["metric", "pos", "epoch", "change"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
