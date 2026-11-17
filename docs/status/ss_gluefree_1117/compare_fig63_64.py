"""Old vs new statistics behind State-Space page Figs 6.3 (step vs gradual) / 6.4 (per position), SVD + LocaNMF bases.

OLD = <new root>/replay_1108_inputs (current step / per-position code on the 11-08 trials / trajectories / pertrial tables;
its PNGs are byte-identical to the page v40 figures). NEW = <new root> (the 11-17 re-run). Per animal first, then pooled.
The per-spout partial r on log(day) (Fig 6.4 caption: "removing log(days) within animal") is not written by
state_space_per_position.py; it is recomputed here: within animal and spout, post-stroke sessions, x (or y) and aim
each residualized on [1, ln(day)], r of the residuals. `--check-allen` prints it for the production Allen table (the
caption's numbers) as a check of this reconstruction.

usage: python compare_fig63_64.py <new root> [--check-allen <production state_space_recovery root>]
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AN = ["PS92", "PS93", "PS94", "PS95"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
AIM = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
           "iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-"
           "incomplete-stopA2jexrt3500c70_daqlk/aim_deficit_sessions.csv")


def partial_logday(s):
    """per animal x spout x readout: within-animal partial r(readout, aim | ln day), post-stroke sessions."""
    post = s[s.epoch != "pre"].dropna(subset=["aim", "day"])
    rows = []
    for (an, p), g in post.groupby(["animal", "spout"]):
        X = np.column_stack([np.ones(len(g)), np.log(np.maximum(g.day.to_numpy(float), 0.5))])
        for c in ("x", "y"):
            h = g.dropna(subset=[c])
            if len(h) < 5:
                continue
            Xh = X[g[c].notna().to_numpy()]
            rx = h[c].to_numpy(float) - Xh @ np.linalg.lstsq(Xh, h[c].to_numpy(float), rcond=None)[0]
            ra = h.aim.to_numpy(float) - Xh @ np.linalg.lstsq(Xh, h.aim.to_numpy(float), rcond=None)[0]
            rows.append({"animal": an, "spout": p, "readout": c, "partial_r": float(np.corrcoef(rx, ra)[0, 1])})
    return pd.DataFrame(rows)


def step_tables(root, tag):
    A = pd.read_csv(root / f"step_vs_gradual{tag}" / "against_recovery.csv")
    for an in AN:
        A[f"r_{an}"] = [ast.literal_eval(d).get(an, np.nan) for d in A.r_per_animal]
    M = pd.read_csv(root / f"step_vs_gradual{tag}" / "time_models.csv")
    return A.drop(columns="r_per_animal"), M


def perpos_tables(root, tag):
    F = pd.read_csv(root / f"per_position{tag}" / "per_position_fits.csv")
    S = pd.read_csv(root / f"per_position{tag}" / "per_position_sessions.csv")
    return F, partial_logday(S)


def main():
    new = Path(sys.argv[1])
    old = new / "replay_1108_inputs"
    out = new / "compare"
    out.mkdir(exist_ok=True)
    if "--check-allen" in sys.argv:
        prod = Path(sys.argv[sys.argv.index("--check-allen") + 1])
        P = perpos_tables(prod, "_lick")[1]
        print("ALLEN production partial r (caption: x PS92 .58-.74, others .0-.47; y PS94 only):")
        print(P.pivot_table(index=["readout", "spout"], columns="animal", values="partial_r").reindex(POS, level=1).round(2))
    for tag in ("_lick_svd", "_lick_lnmfbm"):
        Ao, Mo = step_tables(old, tag)
        An, Mn = step_tables(new, tag)
        A = Ao.merge(An, on="readout", suffixes=("_old", "_new"))
        A.to_csv(out / f"step_against_recovery{tag}.csv", index=False)
        W = (Mo[["animal", "readout", "best"]].merge(Mn[["animal", "readout", "best"]], on=["animal", "readout"],
                                                   suffixes=("_old", "_new")))
        W.to_csv(out / f"step_time_models_best{tag}.csv", index=False)
        Fo, Po = perpos_tables(old, tag)
        Fn, Pn = perpos_tables(new, tag)
        F = Fo.merge(Fn, on=["spout", "readout", "vs"], suffixes=("_old", "_new"))
        F.to_csv(out / f"perpos_fits{tag}.csv", index=False)
        P = Po.merge(Pn, on=["animal", "spout", "readout"], suffixes=("_old", "_new"))
        P.to_csv(out / f"perpos_partial_logday{tag}.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 4):
            print(f"\n######## {tag}")
            print("== Fig 6.3 per-animal r vs aim (old -> new)")
            print(A[["readout"] + [f"r_{a}_{k}" for a in AN for k in ("old", "new")]].to_string(index=False))
            print("== Fig 6.3 pooled: circular-shift p, within-epoch LMM p, within-animal slope")
            print(A[["readout", "within_slope_old", "within_slope_new", "shuffle_p_old", "shuffle_p_new",
                     "p_given_epoch_old", "p_given_epoch_new", "lmm_p_old", "lmm_p_new"]].to_string(index=False))
            print("== Fig 6.3 time models: wins (old / new) of", len(W), "fits")
            print(pd.concat([W.best_old.value_counts().rename("old"), W.best_new.value_counts().rename("new")], axis=1))
            print("changed winners:\n", W[W.best_old != W.best_new].to_string(index=False))
            print("== Fig 6.4 per spout, per-animal r and circular-shift p (old -> new)")
            F["star_old"] = F.shuffle_p_old < .05
            F["star_new"] = F.shuffle_p_new < .05
            print(F[["spout", "readout", "vs"] + [f"r_{a}_{k}" for a in AN for k in ("old", "new")]
                    + ["shuffle_p_old", "shuffle_p_new", "star_old", "star_new"]].to_string(index=False))
            print("== Fig 6.4 partial r on log(day), x and y vs aim (old -> new)")
            print(P.pivot_table(index=["readout", "spout"], columns="animal",
                                values=["partial_r_old", "partial_r_new"]).round(2).to_string())


if __name__ == "__main__":
    main()
