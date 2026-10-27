"""Decoding ERROR in mm per session (Priya 10-26: "let's add the MAE / RMSE - but this will be subject to pull to
middle"). Companion to page 3.15 (r).

Same frozen where-decoder and per-lick predictions as 3.15-3.22 (`where_preds.csv`; Allen areas, lags 0..0.2 s, pre
scored leave-one-session-out). Per session: mean absolute error (MAE) and root-mean-square error (RMSE) of decoded vs
actual lateral position, sagittal extension, and the 2-D distance. Unlike r, these include EVERYTHING that separates
decoded from actual: worse lick-to-lick tracking, a constant offset (3.21 / 3.22) and the decoder's pull toward the
middle (present before the stroke; it grows when licks land further from the centre, so part of a post-stroke error
increase can come from where licks land rather than from cortex). One dot per session (all animals pooled), raw mm
and divided by the animal's pre mean; black = mean +- SEM over sessions; * = paired nested bootstrap excluding 0.
Outputs <E>/iti_licks/where_mae_{sessions,tests}.csv, figures/iti_where_mae.png. Usage: python scripts/iti_where_mae.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP4 = ["pre", "acute", "subacute", "chronic"]
COL = {"pre": "0.25", "acute": "#e7298a", "subacute": "#d4a017", "chronic": "#1b9e77"}
TARGETS = {"lat": "LATERAL", "sag": "SAGITTAL", "2d": "2-D distance"}


def sessions(p: pd.DataFrame) -> pd.DataFrame:
    p = p.dropna(subset=["act_lat", "act_sag", "dec_lat", "dec_sag"]).copy()
    p["err_lat"], p["err_sag"] = p.dec_lat - p.act_lat, p.dec_sag - p.act_sag
    p["err_2d"] = np.hypot(p.err_lat, p.err_sag)
    rows = []
    for (st, a, ep, x), g in p.groupby(["set", "animal", "epoch", "label"]):
        r = dict(set=st, animal=a, epoch=ep, label=x, n=len(g))
        for t in TARGETS:
            e = g[f"err_{t}"].to_numpy()
            r[f"mae_{t}"], r[f"rmse_{t}"] = np.abs(e).mean(), np.sqrt((e ** 2).mean())
        rows.append(r)
    S = pd.DataFrame(rows)
    for c in [c for c in S if c.startswith(("mae_", "rmse_"))]:
        pre =S[S.epoch == "pre"].groupby(["set", "animal"])[c].mean().rename("pm")
        S[f"{c}_norm"] = S[c] / S.join(pre, on=["set", "animal"]).pm
    return S


def tests(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in ("iti", "trial"):
        x = S[S.set == st]
        for c in [c for c in S if c.startswith(("mae_", "rmse_"))]:
            prd = {a: g[c].to_numpy() for a, g in x[x.epoch == "pre"].groupby("animal")}
            pre_a = x[x.epoch == "pre"].groupby("animal")[c].mean()
            for ep in EP4[1:]:
                post = {a: g[c].to_numpy() for a, g in x[x.epoch == ep].groupby("animal")}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = x[x.epoch == ep].groupby("animal")[c].mean() - pre_a
                rows.append(dict(set=st, measure=c, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                 n_up=int((am.dropna() > 0).sum()), n=int(am.notna().sum())))
    return pd.DataFrame(rows)


def plot(S: pd.DataFrame, T: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(st, kind) for st in ("iti", "trial") for kind in ("mae", "mae_norm")]
    fig, axs = plt.subplots(len(rows), len(TARGETS), figsize=(4.4 * len(TARGETS), 3.1 * len(rows)), squeeze=False)
    sl = {"iti": "ITI (self-initiated) licks", "trial": "CUED in-trial licks"}
    rng = np.random.default_rng(0)
    for i, (st, kind) in enumerate(rows):
        for j, t in enumerate(TARGETS):
            ax = axs[i, j]
            c = f"mae_{t}" + ("_norm" if kind.endswith("norm") else "")
            for k, ep in enumerate(EP4):
                v = S[(S.set == st) & (S.epoch == ep)][c].dropna().to_numpy()
                ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=9, color=COL[ep], alpha=0.5, lw=0)
                ax.errorbar(k + 0.3, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="k", ms=4,
                            capsize=2, lw=1.2)
                tt = T[(T.set == st) & (T.measure == c) & (T.epoch == ep)]
                if len(tt) and (tt.lo.iloc[0] > 0 or tt.hi.iloc[0] < 0):
                    ax.text(k, 1.0, "*", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
            if kind.endswith("norm"):
                ax.axhline(1, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(len(EP4)))
            ax.set_xticklabels(["pre", "acute", "sub", "chronic"], fontsize=8)
            if i == 0:
                ax.set_title(f"{TARGETS[t]} position error", fontsize=10)
            ax.set_ylabel(f"{sl[st]}\nMAE, decoded vs actual\n" + ("/ animal's pre mean (pre = 1)" if
                                                                     kind.endswith("norm") else "(mm)"), fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("NEURAL: FROZEN pre-stroke lick-position decoder, ERROR in mm (mean absolute error per session; Allen "
                 "areas, lags 0..0.2 s;\nsame licks and predictions as 3.15-3.22). One dot per session, all animals "
                 "pooled; black = mean +- SEM; * = paired nested bootstrap (animals > sessions) excludes 0.\nUnlike r "
                 "(3.15), error INCLUDES the decoder's pull toward the middle and any constant offset,\nso it rises "
                 "when licks land further from the centre even with unchanged cortex.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(il.OUT / "figures" / "iti_where_mae.png", dpi=95)
    plt.close(fig)


def main() -> int:
    S = sessions(pd.read_csv(il.OUT / "where_preds.csv"))
    S.to_csv(il.OUT / "where_mae_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): MAE (mm) by epoch")
    print(S.groupby(["set", "animal", "epoch"])[["mae_lat", "mae_sag", "mae_2d"]].mean().unstack("epoch")
          .reindex(columns=EP4, level=1).round(2).to_string())
    T = tests(S)
    T.to_csv(il.OUT / "where_mae_tests.csv", index=False)
    print(T[T.measure.isin(["mae_lat", "mae_sag", "mae_2d", "rmse_lat"])].round(3).to_string(index=False))
    plot(S, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
