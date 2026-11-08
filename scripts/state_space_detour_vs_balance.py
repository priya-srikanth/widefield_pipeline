"""Is the state-space ACUTE-DEPARTURE axis just the 0b hemispheric BALANCE marker in another basis? (FINDINGS 0c item 3's
open caveat; Priya 11-08 "proceed through the rest", item 1.)

Per post-stroke session: y = mean position on the acute-departure axis (`state_space_trajectories` days_sessions.csv; the
pre -> chronic / acute-departure frame `state_space_recovery_trials.frames`), the PRIMARY recovery marker of FINDINGS 0b
(cued first lick, angle-matched gain ratio ALM + tjM1 + SSp-m + SSp-n + SSs, 0.1..0.5 s; `lick_recovery_primary`
recprim_sessions.csv, panel "cued first lick -> aim") and the far-contra first-lick aim (deg vs own pre,
aim_deficit_sessions.csv). CAVEAT: recprim_sessions.csv exists only for the PRE-REBUILD run (original hemo variant, all
trials gate) -- the rebuild never re-rendered it; the session labels are the same.
Within animal (rule 8 table first), post-stroke sessions: r(y, aim), r(balance, aim), r(y, balance); partial r of each
predictor with aim given the other (both residualized on the other within animal, pooled within-animal slope / r), with a
within-animal shuffle of the tested predictor's residual (2000 draws); share of y's within-animal variance explained by
balance. Output <state_space_recovery>/detour_vs_balance{tag}/. EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_sim as S  # noqa: E402
import state_space_step_vs_gradual as SG  # noqa: E402

RECPRIM = (S.ROOT.parent / "LP_analysis" / "lp_r4_heatmap_20261013__clean_v36" / "imaging_encoding" / "iti_licks"
           / "recprim_sessions.csv")      # PRE-REBUILD run (see docstring)
OUT = S.ROOT / f"detour_vs_balance{S.tag()}"
N_SHUF = 2000


def load():
    y = pd.read_csv(S.ROOT / f"trajectories{S.tag()}" / "days_sessions.csv")[["label", "animal", "epoch", "day", "x", "y"]]
    b = pd.read_csv(RECPRIM).query("panel == 'cued first lick -> aim'")[["label", "marker"]].rename(columns={"marker": "bal"})
    a = pd.read_csv(SG.AIM)[["label", "farcontra_first"]].rename(columns={"farcontra_first": "aim"})
    return y.merge(b, on="label", how="left").merge(a, on="label", how="left")


def demean(d, cols):
    return d[cols] - d.groupby("animal")[cols].transform("mean")


def resid(d, v, on):
    """v residualized on `on` within animal (per-animal OLS)."""
    out = pd.Series(np.nan, index=d.index)
    for _, g in d.groupby("animal"):
        X = np.column_stack([np.ones(len(g)), g[on]])
        out[g.index] = g[v] - X @ np.linalg.lstsq(X, g[v].to_numpy(), rcond=None)[0]
    return out


def partial(d, pred, ctrl, target, rng):
    """Pooled within-animal partial r of pred with target given ctrl, + within-animal shuffle p (pred residual permuted)."""
    rp, rt = resid(d, pred, ctrl), resid(d, target, ctrl)
    r0 = float(np.corrcoef(rp, rt)[0, 1])
    null = np.empty(N_SHUF)
    an = d.animal.to_numpy()
    for i in range(N_SHUF):
        sh = rp.copy()
        for a in np.unique(an):
            m = an == a
            sh[m] = rng.permutation(sh[m].to_numpy())
        null[i] = np.corrcoef(sh, rt)[0, 1]
    return r0, float((np.abs(null) >= abs(r0)).mean())


def main():
    rng = np.random.default_rng(0)
    d = load()
    post = d[(d.epoch != "pre")].dropna(subset=["y", "bal", "aim"]).reset_index(drop=True)
    print(f"post-stroke sessions with y + balance + aim: {len(post)} "
          f"({post.groupby('animal').size().to_dict()}); missing balance: {d[d.epoch != 'pre'].bal.isna().sum()}")
    rows = []
    for an, g in post.groupby("animal"):
        c = g[["y", "bal", "aim", "x"]].corr()
        pr = {f"p_{v}_aim|{w}": float(np.corrcoef(resid(g, v, w), resid(g, "aim", w))[0, 1])
              for v, w in (("y", "bal"), ("bal", "y"), ("x", "bal"))}
        rows.append({"animal": an, "n": len(g), "r_y_aim": c.loc["y", "aim"], "r_bal_aim": c.loc["bal", "aim"],
                     "r_y_bal": c.loc["y", "bal"], "r_x_aim": c.loc["x", "aim"], "r_x_bal": c.loc["x", "bal"], **pr})
    A = pd.DataFrame(rows)
    print("\n== per animal (rule 8), post-stroke sessions ==")
    print(A.round(2).to_string(index=False))
    W = demean(post, ["y", "bal", "aim", "x"])
    pooled = W.corr()
    T = []
    for pred, ctrl in (("y", "bal"), ("bal", "y"), ("x", "bal"), ("bal", "x")):
        r, p = partial(post, pred, ctrl, "aim", rng)
        T.append({"test": f"{pred} -> aim | {ctrl}", "partial_r": r, "shuffle_p": p,
                  "zero_order_r_within": float(pooled.loc[pred, "aim"])})
    T.append({"test": "y ~ bal (within-animal R2)", "partial_r": float(pooled.loc["y", "bal"] ** 2), "shuffle_p": np.nan,
              "zero_order_r_within": float(pooled.loc["y", "bal"])})
    T = pd.DataFrame(T)
    print("\n== pooled within-animal ==")
    print(T.round(3).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    post.to_csv(OUT / "sessions.csv", index=False)
    A.to_csv(OUT / "per_animal.csv", index=False)
    T.to_csv(OUT / "tests.csv", index=False)


if __name__ == "__main__":
    main()
