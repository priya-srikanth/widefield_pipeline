"""Four-camera lick counts / timing for Fig 4A (Priya 11-22: four-camera licks for every figure): first-lick latency,
licks per responded trial and within-bout ILI per session x spout on the FUSED lick train, tested exactly as the cam4
cohort rows they replace.

    python scripts/fourcam_lick_metrics.py [--out-dir DIR] [--force]

Until now these three Fig 4A metrics came from the cam4 2-D cohort table (`orofacial_cohort`, cam4 LP lick windows):
`<run>/cohort/tables/prepost_working/stats/change_vs_pre_stats_contra.csv`, read by `c2_scalar_family_aligned.pipeline_rows`.

RECIPE (one row per session x position of the cohort's position_metrics.csv: same sessions, epochs, positions, the same
trial gate = `orofacial_cohort.cohort_keep` (camera working period), August pre baseline):
  licks   = `session_licks.session_licks` (the fused four-camera train), context post_cue (peak >= cue + 70 ms, before the
            trial stop, as the cam4 post-cue table), EXCLUDING cam1-only licks (``cams == 'cam1'``; Priya 11-22: "not
            licks" -- median 0 3-D frames, reach 1.85 mm, DAQ contact 0.3 %, late in the trial)
  latency_ms       median over trials of the FIRST post-cue lick's ONSET (on_ms: ms from the cue; = daq_on_s - cue)
  latency_peak_ms  the same at the lick PEAK (the cam4 cohort's latency_ms definition; for the like-for-like check)
  licks_per_resp   mean number of post-cue licks over trials with >= 1 post-cue lick ("responded")
  ili_ms           median within-trial interval between consecutive post-cue lick PEAKS, intervals < ILI_MAX_MS (300 ms)
  n_licks, n_resp_trials, n_cam1_only (the excluded licks, reported)
TESTS: `orofacial_stats.session_change_tests` on the cohort's position table with ONLY these columns replaced and the
cohort's metric list / order / seed (`orofacial_cohort.contra_stats` replays it the same way), so every other draw is the
cohort's; the three metrics are not image-signed, so the + = contra table equals the + = ipsi one. A replay of the
unmodified cohort table must reproduce the stored cam4 rows (checked, printed) before anything is written.
Output (new names only): ``position_metrics_fourcam.csv``, ``change_vs_pre_stats_contra_fourcam.csv`` (the three metrics'
rows, columns as the cohort's) and ``per_animal_fourcam.csv`` (rule 8: epoch means of the per-session change, fused
next to cam4).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("latency_ms", "licks_per_resp", "ili_ms")
OUT_DEFAULT = Path("M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_4cam_sync60_20261122")


def session_rows(job) -> list[dict]:
    """Per position of one session: the fused metrics (module docstring). fan_sessions worker."""
    animal, date = job
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import session_licks as SL
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    sd = oo.session_dir(rv, oc.RUN, animal, date)
    keep, _gate = oc.cohort_keep(animal, date, sd)
    A = SL.session_licks(f"{animal}_{date}", rv)
    A = A[A.context == "post_cue"]
    c1 = A.cams.astype(str) == "cam1"
    n_c1 = A[c1].groupby("position").size()
    A = A[~c1]
    if keep is not None:
        A = A[A.trial_id.astype(int).isin(keep)]
    rows = []
    for pos, g in A.groupby("position"):
        g = g.sort_values(["trial_id", "t_ms"])
        by = g.groupby("trial_id")
        rt = by.size()
        ili = by.t_ms.diff()
        ili = ili[ili < oc.ILI_MAX_MS]
        rows.append({"animal": animal, "date": date, "position": pos,
                     "latency_ms": float(by.on_ms.min().median()), "latency_peak_ms": float(by.t_ms.min().median()),
                     "licks_per_resp": float(rt.mean()), "ili_ms": float(ili.median()) if len(ili) else np.nan,
                     "n_licks": int(len(g)), "n_resp_trials": int(len(rt)), "n_cam1_only": int(n_c1.get(pos, 0))})
    return rows


def per_animal(P: pd.DataFrame, cols) -> pd.DataFrame:
    """Rule 8: per animal x epoch, the mean over sessions x positions of each session's change from its animal's
    August pre mean at the same position."""
    out = {}
    for c in cols:
        base = P[P.epoch == "pre"].groupby(["animal", "pos"])[c].mean()
        d = P[P.epoch != "pre"].copy()
        d["chg"] = d[c] - base.reindex(pd.MultiIndex.from_frame(d[["animal", "pos"]])).to_numpy()
        out[c] = d.groupby(["animal", "epoch"]).chg.mean()
    return pd.DataFrame(out).unstack("epoch").reindex(columns=["acute", "subacute", "chronic"], level=1)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_stats as st
    from wfield_local import session_licks as SL
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", type=Path, default=None, help="cohort tables folder (default <run>/cohort/tables/"
                                                              "prepost_working)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not SL.fused_on():
        raise SystemExit("no fused lick set selected")
    tables = a.cohort or oc.cohort_dir("tables", run=oc.RUN, rv=PathResolver())
    if not (tables / "position_metrics.csv").exists():
        tables = tables / "prepost_working"
    Pm = pd.read_csv(tables / "position_metrics.csv", dtype={"date": str})
    stored = pd.read_csv(tables / "stats" / "change_vs_pre_stats_contra.csv")
    outs = {n: a.out_dir / n for n in ("position_metrics_fourcam.csv", "change_vs_pre_stats_contra_fourcam.csv",
                                       "per_animal_fourcam.csv")}
    for f in outs.values():
        if f.exists() and not a.force:
            raise SystemExit(f"{f} exists: --force to overwrite")
    ms_ = list(dict.fromkeys([*oc.COHORT_SCALARS, *(m for g in oc.POS_GROUPS.values() for m in g)]))
    # (1) replay of the unmodified cohort table reproduces the stored cam4 rows (stored at 4 significant digits)
    R0 = st.session_change_tests(Pm, ms_)
    key = ["metric", "pos", "epoch"]
    c = R0[R0.metric.isin(METRICS)].merge(stored, on=key, suffixes=("", "_st"))
    rel = max(float(np.nanmax(np.abs(c[k] - c[k + "_st"]) / np.maximum(np.abs(c[k + "_st"]), 1e-12)))
              for k in ("boot_change", "boot_lo", "boot_hi", "p", "q_fdr"))
    same = bool((c.agree == c.agree_st).all())
    print(f"replay of the cam4 cohort rows: {len(c)} rows, max relative diff {rel:.2g}, agree identical: {same}")
    if not same or rel > 1e-3:
        raise SystemExit("the replay does not reproduce the stored cam4 rows; nothing written")
    # (2) the fused metrics per session x position
    sess = sorted({(r.animal, r.date) for r in Pm.itertuples()})
    res, fails = ak.fan_sessions(sess, session_rows, jobs=a.jobs, key=lambda j: f"{j[0]}_{j[1]}")
    for j, e in fails:
        print(f"{j}: FAILED {e}")
    F = pd.DataFrame([r for _, rr in res for r in rr])
    P4 = Pm.merge(F, on=["animal", "date", "position"], how="left", suffixes=("_cam4", ""))
    miss = P4.licks_per_resp.isna() & P4.licks_per_resp_cam4.notna()
    print(f"{len(F)} fused session x position rows; cohort rows without a fused value: {int(miss.sum())}; cam1-only "
          f"post-cue licks excluded: {int(F.n_cam1_only.sum())} of {int(F.n_licks.sum() + F.n_cam1_only.sum())}")
    Pt = Pm.copy()
    for m in METRICS:
        Pt[m] = P4[m].to_numpy()
    R4 = st.session_change_tests(Pt, ms_)
    R4 = R4[R4.metric.isin(METRICS)].reset_index(drop=True)
    pa = per_animal(P4, list(METRICS) + [f"{m}_cam4" for m in METRICS] + ["latency_peak_ms"])
    pd.set_option("display.width", 250)
    print("== per animal (rule 8): epoch means of the per-session change from pre (fused, then cam4)")
    print(pa.round(2).to_string())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    P4.to_csv(outs["position_metrics_fourcam.csv"], index=False)
    R4.to_csv(outs["change_vs_pre_stats_contra_fourcam.csv"], index=False, float_format="%.4g")
    pa.to_csv(outs["per_animal_fourcam.csv"])
    cmp = R4.merge(stored[stored.metric.isin(METRICS)], on=key, suffixes=("", "_cam4"))
    flips = cmp[cmp.agree != cmp.agree_cam4]
    print(f"star changes vs the cam4 cohort rows ({len(cmp)} cells):")
    print(flips[key + ["boot_change_cam4", "boot_change", "q_fdr_cam4", "q_fdr", "agree_cam4", "agree"]].to_string()
          if len(flips) else "  none")
    for f in outs.values():
        print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
