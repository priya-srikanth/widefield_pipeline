"""Is the state-space ACUTE-DEPARTURE axis just the 0b hemispheric BALANCE marker in another basis? (FINDINGS 0c item 3's
open caveat; Priya 11-08 "proceed through the rest", item 1.)

Per post-stroke session: y = mean position on the acute-departure axis (`state_space_trajectories` days_sessions.csv; the
pre -> chronic / acute-departure frame `state_space_recovery_trials.frames`), the PRIMARY recovery marker of FINDINGS 0b
(cued first lick, angle-matched gain ratio ALM + tjM1 + SSp-m + SSp-n + SSs, 0.1..0.5 s; `lick_recovery_primary`
recprim_sessions.csv, panel "cued first lick -> aim") and the far-contra first-lick aim (deg vs own pre,
aim_deficit_sessions.csv). Balance and aim both come from the rebuild's run tables (`S.RECPRIM`, `S.ITI_W`; 11-16 path
fix: until then RECPRIM was built under `ROOT.parent`, which the 11-12 reroute turned into the rebuild root, where no
such file exists; the 11-08 numbers used the PRE-REBUILD copy, original hemo variant, all-trials gate).
Within animal (rule 8 table first), post-stroke sessions: r(y, aim), r(balance, aim), r(y, balance); partial r of each
predictor with aim given the other (both residualized on the other within animal, pooled within-animal slope / r), with a
within-animal circular shift of the tested predictor's residual over the animal's day-sorted sessions (2000 draws;
a free shuffle until 11-16); share of y's within-animal variance explained by
balance. Output <state_space_recovery>/detour_vs_balance{tag}/. EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
import state_space_step_vs_gradual as SG  # noqa: E402

RECPRIM = S.RECPRIM                       # the rebuild's primary balance marker (see docstring)
OUT = S.ROOT / f"detour_vs_balance{S.tag()}"
N_SHUF = 2000


def load():
    y = pd.read_csv(S.require(S.ROOT / f"trajectories{S.tag()}" / "days_sessions.csv", "trajectories table"))[
        ["label", "animal", "epoch", "day", "x", "y"]]
    b = pd.read_csv(S.require(RECPRIM, "primary balance marker (lick_recovery_primary)")).query(
        "panel == 'cued first lick -> aim'")[["label", "marker"]].rename(columns={"marker": "bal"})
    a = pd.read_csv(S.require(SG.AIM, "far-contra aim table"))[["label", "farcontra_first"]].rename(
        columns={"farcontra_first": "aim"})
    d = y.merge(b, on="label", how="left").merge(a, on="label", how="left")
    post = d[d.epoch != "pre"]
    for c in ("bal", "aim"):
        if post[c].isna().all():
            raise ValueError(f"no post-stroke session matched a '{c}' value: label mismatch between the inputs?")
    return d


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
    """Pooled within-animal partial r of pred with target given ctrl, + within-animal circular-shift p (pred residual
    rotated over days)."""
    rp, rt = resid(d, pred, ctrl), resid(d, target, ctrl)
    r0 = float(np.corrcoef(rp, rt)[0, 1])
    _, p = kit.within_shuffle_p(lambda a: np.corrcoef(a, rt)[0, 1], rp.to_numpy(), d.animal.to_numpy(), rng, N_SHUF,
                                order=d.day.to_numpy(), mode="circular")      # rotated over days (Priya 11-16)
    return r0, p


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
