"""DO EARLY CORTICAL FEATURES PREDICT THE CHRONIC OUTCOME PER ANIMAL? (Priya 10-25.) EXPLORATORY at n = 4.

One value per animal. Early (acute) features: the first post-stroke session's orofacial residual (frozen frame + FR,
pre-SD units, mean of SSp-m / MOp both hemispheres; cue and lick), its mean over days 1-3, acute frozen-decoder far-contra
retention on ALL far-contra trials and on HITS (lick window), acute trial-by-trial coupling (all far-contra trials,
landing), and lesion burden (% of ipsilesional caudoputamen, docs/HISTOLOGY.md) as the structural reference. Outcomes:
behavioural half-recovery day (`recovery_timing.py`), chronic far-contra hit rate (relative to pre), chronic cortical
far-contra code on hits, chronic per-session gap (cortex - behaviour), chronic orofacial residual (cue).

Statistic: Spearman rho with the EXACT permutation p over all 4! = 24 orderings (two-sided; a perfect ordering gives
p = 2/24 = 0.083 two-sided, 1/24 = 0.042 one-sided): nothing short of a perfect ordering is significant, so the output is
a ranked list of candidate predictors to PRE-REGISTER for the next cohort, not a result.

Usage: python scripts/early_predictors.py
Output: <E>/attempt_coding/early_predictors{,_table}.csv, figures/early_predictors.png
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
A = E / "attempt_coding"
AN = ["PS92", "PS93", "PS94", "PS95"]
ORO = ["SSp-m_left", "SSp-m_right", "MOp_left", "MOp_right"]
LESION = {"PS92": 46.7, "PS93": 45.0, "PS94": 67.7, "PS95": 81.9}      # docs/HISTOLOGY.md, 2026-10-21
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}


def exact_spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perms = [float(np.corrcoef(rx, np.array(p))[0, 1]) for p in itertools.permutations(ry)]
    return rho, float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


def features() -> pd.DataFrame:
    sp = pd.read_csv(E / "specificity_control" / "gate-attempted_v7g3_t3d_frA" / "sessions.csv")
    sp = sp[sp.area.isin(ORO) & (sp.label.str[-4:] >= "0801")]
    days = pd.read_csv(A / "recovery_timing_series_lick.csv")[["label", "day"]].drop_duplicates()
    sp = sp.merge(days, on="label", how="left")
    ser = pd.read_csv(A / "recovery_timing_series_lick.csv")
    allc = pd.read_csv(E / "attempt_coding_pp" / "trial_neural_vs_exec_coupling.csv")
    half = pd.read_csv(A / "recovery_timing_halftimes_lick.csv").set_index("animal")
    T = pd.concat([pd.read_csv(A / f"trials_lick_{an}.csv.gz") for an in AN])
    T = T[(T.method == "dom") & (T.spout == "far_contra") & (T.label.str[-4:] >= "0801")]
    ref = T[T.cls == "pre_hit"].groupby("animal").p_true.mean()
    T["ret"] = (T.p_true - 1 / 6) / (T.animal.map(ref) - 1 / 6)
    rows = []
    for an in AN:
        s = sp[sp.animal == an]
        post = s[s.day > 0]
        d1 = post.day.min()
        r = {"animal": an, "lesion_pct": LESION[an]}
        for w in ("cue", "lick"):
            r[f"resid_day1_{w}"] = post[(post.day == d1) & (post.window == w)].residual_z.mean()
            r[f"resid_d1to3_{w}"] = post[(post.day <= 3) & (post.window == w)].residual_z.mean()
        r["resid_chronic_cue"] = s[(s.epoch == "chronic") & (s.window == "cue")].residual_z.mean()
        r["ret_acute_all"] = T[(T.animal == an) & (T.epoch == "acute")].ret.mean()
        h = T[(T.animal == an) & (T.epoch == "acute") & (T.cls == "hit")]
        r["ret_acute_hits"] = h.ret.mean() if len(h) >= 5 else np.nan
        c = allc[(allc.animal == an) & (allc.epoch == "acute") & (allc.exec == "landing")]
        r["coupling_acute"] = float(c.r.iloc[0]) if len(c) else np.nan
        g = ser[ser.animal == an]
        r["beh_t_half"] = half.loc[an, "t_half_behaviour"]
        r["beh_chronic"] = g[g.epoch == "chronic"].beh.mean()
        r["cortex_chronic_hits"] = g[g.epoch == "chronic"].neu.mean()
        r["gap_chronic"] = (g[g.epoch == "chronic"].neu - g[g.epoch == "chronic"].beh).mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("animal")


PRED = ["lesion_pct", "resid_day1_cue", "resid_day1_lick", "resid_d1to3_cue", "resid_d1to3_lick", "ret_acute_all",
        "ret_acute_hits", "coupling_acute"]
OUT = ["beh_t_half", "beh_chronic", "cortex_chronic_hits", "gap_chronic", "resid_chronic_cue"]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = features()
    pd.set_option("display.width", 240)
    print("== per animal (rule 8)\n" + F.round(3).T.to_string())
    rows = []
    for p in PRED:
        for o in OUT:
            ok = F[[p, o]].dropna()
            if len(ok) == 4:
                rho, pv = exact_spearman(ok[p].to_numpy(), ok[o].to_numpy())
                rows.append({"predictor": p, "outcome": o, "rho": rho, "p_exact_two_sided": pv,
                             "order": " < ".join(ok.sort_values(p).index)})
            else:
                rows.append({"predictor": p, "outcome": o, "rho": np.nan, "p_exact_two_sided": np.nan,
                             "order": f"{len(ok)} animals"})
    R = pd.DataFrame(rows)
    F.to_csv(A / "early_predictors_table.csv")
    R.to_csv(A / "early_predictors.csv", index=False)
    print("== Spearman rho, exact permutation p (24 orderings; perfect ordering = 0.083 two-sided)")
    print(R.pivot_table(index="predictor", columns="outcome", values="rho").reindex(PRED).round(2).to_string())
    print(R[R.rho.abs() >= 0.99].to_string(index=False))

    fig, axs = plt.subplots(len(OUT), len(PRED), figsize=(2.1 * len(PRED), 2.0 * len(OUT)), squeeze=False)
    for i, o in enumerate(OUT):
        for j, p in enumerate(PRED):
            ax = axs[i, j]
            for an in AN:
                if np.isfinite(F.loc[an, p]) and np.isfinite(F.loc[an, o]):
                    ax.scatter(F.loc[an, p], F.loc[an, o], color=ACOL[an], s=30, edgecolor="k", lw=0.4)
            r = R[(R.predictor == p) & (R.outcome == o)].iloc[0]
            if np.isfinite(r.rho):
                ax.text(0.03, 0.97, f"rho {r.rho:+.1f}", transform=ax.transAxes, fontsize=7, va="top",
                        fontweight="bold" if abs(r.rho) > 0.99 else "normal")
            ax.tick_params(labelsize=6)
            if i == len(OUT) - 1:
                ax.set_xlabel(p, fontsize=7)
            if j == 0:
                ax.set_ylabel(o, fontsize=7)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc=ACOL[a], mec="k", label=a) for a in AN], loc="upper right",
               fontsize=8, frameon=False)
    fig.suptitle("EXPLORATORY (n = 4): early features (columns) vs chronic outcome (rows), one dot per animal; Spearman rho "
                 "(bold = perfect ordering, exact p = 0.083 two-sided / 0.042 one-sided)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.95, 0.95))
    fig.savefig(A / "figures" / "early_predictors.png", dpi=110)
    print("->", A / "figures" / "early_predictors.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
