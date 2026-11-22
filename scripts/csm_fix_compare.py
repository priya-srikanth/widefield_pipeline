"""CSM edge-peak fix (2026-11-21, Priya: "yes fix the measure"): compare the regenerated CSM tables with the tables the
VLS manuscript cites. READ ONLY on the cited tables; writes one comparison CSV per table + a summary into --out.

Tables compared (cited -> regenerated):
  within_lick_correction/csm_threshold.json      threshold rule (theta_sp / theta_ml, grid rows at the shared thetas)
  within_lick_correction/primary_holm.csv        P1-P3 boot_delta + Holm (the pipeline's own primary tests)
  within_lick_correction/boot.csv                every boot_table cell (metric x group x order x outcome x epoch)
  within_lick_correction/per_animal.csv          per-animal epoch means (rule 8; far contra / all, any order / outcome)
  manuscript_stats/c2_recovery_mechanics_v2_stats.csv   S16 stats (session MWU + boot, pooled-family q, star)
  manuscript_code/data/c2_scalar_family_aligned_with_recovery.csv   the pooled BH family: star flips of every cell
A CONTROL run (the current code on the Oct 30 per-lick pickles, old zero-padded peaks, Oct 30 threshold) is compared
the same way when given (--control), so a change can be attributed to the fix or to code changes since Oct 30 (e.g. the
11-05 camera working period).

Flags (Priya's tolerances): shift beyond 1 deg (angles), 0.02 (rates / fractions), 10 % otherwise; star flips either way;
n changes (animals / sessions / licks).

Usage: python scripts/csm_fix_compare.py --new-wlc DIR --new-stats DIR --out DIR [--control DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRV = Path("M:/MICROSCOPE/Priya")
ENC = SRV / "Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding"
CITED_WLC = ENC / "within_lick_correction"
CITED_STATS = SRV / "Widefield/manuscript_stats"
CITED_FAMILY = SRV / "stroke_orofacial_pipeline_outputs/vls_stroke_paper/manuscript_code/data/" \
                     "c2_scalar_family_aligned_with_recovery.csv"
ANGLES = {"e_L", "e_E", "corr", "toward", "abs_corr", "theta_C_e", "D_off", "cued_mean", "GAP", "step_b"}
RATES = {"csm_share", "csm_false", "csm_lat_share", "csm_share_lo", "csm_share_hi", "csm_excess", "onset_cov",
         "full_cov", "single_share", "g_w", "obs_share", "false", "sur_smooth_share"}

NO_SHIFT_FLAG = {"p", "q_fdr", "p_holm", "pipe_p_holm", "q_family_pooled", "lo", "hi", "boot_lo", "boot_hi",
                 "delta_lo", "delta_hi", "level_lo", "level_hi"}


def tol_flag(metric: str, a, b) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(np.isfinite(a) != np.isfinite(b))
    if metric in ANGLES:
        return abs(b - a) > 1.0
    if metric in RATES:
        return abs(b - a) > 0.02
    return abs(b - a) > 0.10 * max(abs(a), 1e-12)


def compare(A: pd.DataFrame, B: pd.DataFrame, keys: list, vals: list, star: str | None, ns: list,
            metric_col="metric") -> pd.DataFrame:
    M = A.merge(B, on=keys, how="outer", suffixes=("_cited", "_new"), indicator=True)
    rows = []
    for _, r in M.iterrows():
        o = {k: r[k] for k in keys}
        o["in"] = r["_merge"]
        m = r[metric_col] if metric_col in keys else ""
        flags = []
        for v in vals:
            a, b = float(r.get(f"{v}_cited", np.nan)), float(r.get(f"{v}_new", np.nan))
            o[f"{v}_cited"], o[f"{v}_new"] = a, b
            if v not in NO_SHIFT_FLAG and tol_flag(m, a, b):      # p / q / CI bounds: reported, judged via the star
                flags.append(f"shift:{v}")
        if star:
            sa, sb = r.get(f"{star}_cited"), r.get(f"{star}_new")
            o["star_cited"], o["star_new"] = sa, sb
            if pd.notna(sa) and pd.notna(sb) and bool(sa) != bool(sb):
                flags.append("STAR_LOST" if bool(sa) else "STAR_GAINED")
        for n in ns:
            a, b = r.get(f"{n}_cited"), r.get(f"{n}_new")
            o[f"{n}_cited"], o[f"{n}_new"] = a, b
            if pd.notna(a) and pd.notna(b) and a != b:
                flags.append(f"n:{n}")
        if r["_merge"] != "both":
            flags.append(f"only_{r['_merge']}")
        o["flags"] = ";".join(flags)
        rows.append(o)
    return pd.DataFrame(rows)


def holm_star(H: pd.DataFrame) -> pd.DataFrame:
    H = H.copy()
    H["ci_excl0"] = (H.lo > 0) | (H.hi < 0)
    return H


def run(new_wlc: Path, new_stats: Path, out: Path, tag: str, summary: list) -> None:
    # 1. threshold
    ca, cb = (json.loads((d / "csm_threshold.json").read_text()) for d in (CITED_WLC, new_wlc))
    rows = []
    for kind in ("sp", "ml"):
        ga = {r["theta"]: r for r in ca["grid"][kind]}
        gb = {r["theta"]: r for r in cb["grid"][kind]}
        for th in sorted(set(ga) | set(gb)):
            rows.append({"kind": kind, "theta": th, **{f"{c}_cited": ga.get(th, {}).get(c, np.nan) for c in
                                                       ("false", "obs_share", "sur_smooth_share")},
                         **{f"{c}_new": gb.get(th, {}).get(c, np.nan) for c in ("false", "obs_share", "sur_smooth_share")}})
    T = pd.DataFrame(rows)
    T.to_csv(out / f"cmp_{tag}_csm_threshold.csv", index=False, float_format="%.4f")
    summary.append(f"[{tag}] threshold theta_sp {ca['theta_sp']} -> {cb['theta_sp']}, theta_ml {ca['theta_ml']} -> "
                   f"{cb['theta_ml']}; calibration licks {ca['n_licks']} -> {cb['n_licks']}, sessions "
                   f"{len(ca['sessions'])} -> {len(cb['sessions'])}")
    # 2. primary_holm
    A, B = (holm_star(pd.read_csv(d / "primary_holm.csv")) for d in (CITED_WLC, new_wlc))
    A["star"], B["star"] = A.p_holm < 0.05, B.p_holm < 0.05
    P = compare(A, B, ["metric", "epoch"], ["delta", "lo", "hi", "p", "p_holm"], "star", [])
    P["ci_excl0_cited"] = A.set_index(["metric", "epoch"]).loc[list(zip(P.metric, P.epoch)), "ci_excl0"].to_numpy()
    P["ci_excl0_new"] = B.set_index(["metric", "epoch"]).loc[list(zip(P.metric, P.epoch)), "ci_excl0"].to_numpy()
    P.to_csv(out / f"cmp_{tag}_primary_holm.csv", index=False, float_format="%.4f")
    # 3. boot.csv (the pipeline criterion: change = CI excludes 0 AND >= 3/4 animals same sign)
    keys = ["context", "group", "order", "outcome", "metric", "epoch"]
    A, B = (pd.read_csv(d / "boot.csv") for d in (CITED_WLC, new_wlc))
    Bt = compare(A, B, keys, ["level", "delta", "delta_lo", "delta_hi"], "change", ["n_animals", "n_paired"])
    Bt.to_csv(out / f"cmp_{tag}_boot.csv", index=False, float_format="%.4f")
    # 4. per-animal
    keys = ["context", "group", "order", "outcome", "animal", "epoch"]
    A, B = (pd.read_csv(d / "per_animal.csv") for d in (CITED_WLC, new_wlc))
    mets = [c[:-5] for c in A.columns if c.endswith("_mean") and c.startswith("csm")] + ["e_L", "e_E", "corr"]
    A_, B_ = (d[keys + [f"{m}_mean" for m in mets] + [f"{m}_count" for m in mets]] for d in (A, B))
    A_ = A_.melt(id_vars=keys, value_vars=[f"{m}_mean" for m in mets], var_name="metric", value_name="v")
    B_ = B_.melt(id_vars=keys, value_vars=[f"{m}_mean" for m in mets], var_name="metric", value_name="v")
    for d in (A_, B_):
        d["metric"] = d.metric.str[:-5]
    sel = lambda d: d[(d.context == "cued") & d.group.isin(["far_contra", "all"]) & (d.order == "any")  # noqa: E731
                      & (d.outcome == "any")]
    PA = compare(sel(A_), sel(B_), keys + ["metric"], ["v"], None, [])
    PA.to_csv(out / f"cmp_{tag}_per_animal.csv", index=False, float_format="%.4f")
    if new_stats is None:
        return
    # 5. S16 stats
    keys = ["metric", "epoch"]
    A, B = (pd.read_csv(d / "c2_recovery_mechanics_v2_stats.csv") for d in (CITED_STATS, new_stats))
    S = compare(A, B, keys, ["boot_change", "boot_lo", "boot_hi", "mean_change", "p", "q_fdr", "pipe_delta",
                             "pipe_p_holm"], "star", ["n_animals", "n_sessions", "n_pre_sessions"])
    S.to_csv(out / f"cmp_{tag}_c2_recovery_mechanics_v2_stats.csv", index=False, float_format="%.4f")
    # 6. pooled family
    A = pd.read_csv(CITED_FAMILY)
    B = pd.read_csv(new_stats / CITED_FAMILY.name)
    for d in (A, B):
        d["star_pooled"] = (d.agree_pooled == "both").astype(int)
    F = compare(A, B, ["metric", "pos", "epoch"], ["q_family_pooled"], "star_pooled", [])
    F.to_csv(out / f"cmp_{tag}_scalar_family_with_recovery.csv", index=False, float_format="%.5f")
    for name, D in (("primary_holm", P), ("boot (all cells)", Bt), ("S16 stats", S), ("pooled family", F)):
        fl = D[D["flags"].astype(str).str.len() > 0]
        st = D[D["flags"].astype(str).str.contains("STAR")]
        summary.append(f"[{tag}] {name}: {len(D)} rows, {len(fl)} flagged, {len(st)} star flips")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new-wlc", type=Path, required=True)
    ap.add_argument("--new-stats", type=Path, required=True)
    ap.add_argument("--control", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from wfield_local import writeguard
    writeguard.assert_writable(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.glob("cmp_*.csv")):
        sys.exit(f"refusing to overwrite comparison tables in {a.out}")
    summary: list[str] = []
    run(a.new_wlc, a.new_stats, a.out, "fix", summary)
    if a.control is not None:
        run(a.control, None, a.out, "control_oldpeaks", summary)
    (a.out / "cmp_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
