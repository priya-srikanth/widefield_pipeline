"""MISDIRECTED far-contra licks: do far-right-trial licks that land LEFT look like left licks or right licks in cortex?
(Priya 10-27: "PS94 has far R trials where the animal licks left. do licks that land left look like neural left lick
activity or right lick activity?")

Cued licks, Allen-area activity at lags 0..0.2 s from the lick peak, z-scored within session (KEY row, session_sd.csv).
Per animal, groups of licks:
  MIS  target far contra (far R), landing LEFT of the animal's pre mean landing at the middle far spout
  LEFT target close / far ipsi (left spouts), landing in the same lateral range as that session's MIS licks
  RIGHT target far contra, landing at the animal's usual far-contra place (within 0.5 mm of its pre mean)
Read-outs (per session, post-stroke; animal's pre sessions build the references):
  (1) TEMPLATE score: s = (x - mu_L) . (mu_R - mu_L) / |mu_R - mu_L|^2 on the animal's PRE templates (mu_L = pre
      left-target licks, mu_R = pre on-target far-R licks): 0 = looks like a left lick, 1 = looks like a right lick.
      Same-session LEFT and RIGHT licks are scored too (drift control: where post-stroke left / right licks fall).
  (2) FROZEN DECODER (KEY-row lateral decoder, first_lick_preds): decoded lateral of MIS licks minus decoded lateral of
      same-session LEFT licks with the same landing (landing bins of 0.25 mm) -- 0 = read as a left lick, > 0 = read
      toward the right / target.
Output <E>/iti_licks/misdirected_{licks,sessions}.csv, figures/lick_cued_misdirected_licks.png.
Usage: python scripts/cued_misdirected_licks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_first_lick as fl  # noqa: E402
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
LEFT_T = ("close_ipsi", "far_ipsi")
MIN = 5


def lick_table() -> pd.DataFrame:
    from iti_decoder_contrib import feature_names
    from wfield_local import orofacial_cohort as oc
    SD = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load("areas")
    rows, feats = [], []
    for x, d in sorted(D.items()):
        X, C = d["trial"]
        pos = np.array([q if q in fl.POS or "_" not in q else oc.pos_key(q, "L") for q in d["trial_pos"].astype(str)])
        ok = np.isfinite(C[:, 0]) & np.isin(pos, fl.POS)
        Xz = X[ok] / SD.loc[x, names].to_numpy(float)[None, :]
        Xz = Xz - Xz.mean(0)                                   # within-session centring (drift / baseline)
        feats.append(Xz)
        rows.append(pd.DataFrame({"animal": d["animal"], "label": x, "epoch": d["epoch"], "pos": pos[ok],
                                  "act": C[ok, 0], "t": il._load(x)["tr_t"].astype(float)[ok]}))
    T = pd.concat(rows, ignore_index=True)
    F = np.vstack(feats)
    return T, F


def classify(T: pd.DataFrame) -> pd.DataFrame:
    T = T.copy()
    T["grp"] = ""
    for _a, g in T.groupby("animal"):
        pre = g[g.epoch == "pre"].groupby("pos").act.mean()
        mid, fc = pre["far_mid"], pre["far_contra"]
        i = g.index
        T.loc[i[(g.pos == "far_contra") & (g.act < mid)], "grp"] = "MIS"
        T.loc[i[(g.pos == "far_contra") & ((g.act - fc).abs() < 0.5)], "grp"] = "RIGHT"
        T.loc[i[g.pos.isin(LEFT_T)], "grp"] = "LEFT"
    return T


def scores(T: pd.DataFrame, F: np.ndarray) -> pd.DataFrame:
    T = T.copy()
    T["s"] = np.nan
    for _a, g in T.groupby("animal"):
        pre = g[g.epoch == "pre"]
        muL = F[pre.index[pre.grp == "LEFT"]].mean(0)
        muR = F[pre.index[pre.grp == "RIGHT"]].mean(0)
        w = muR - muL
        T.loc[g.index, "s"] = (F[g.index] - muL) @ w / (w @ w)
    # pre sessions: score with templates that leave the session out (otherwise pre LEFT / RIGHT sit at 0 / 1 by fit)
    for _a, g in T.groupby("animal"):
        pre = g[g.epoch == "pre"]
        for lab in pre.label.unique():
            o = pre[pre.label != lab]
            w = F[o.index[o.grp == "RIGHT"]].mean(0) - F[o.index[o.grp == "LEFT"]].mean(0)
            muL = F[o.index[o.grp == "LEFT"]].mean(0)
            idx = pre.index[pre.label == lab]
            T.loc[idx, "s"] = (F[idx] - muL) @ w / (w @ w)
    return T


def decoder_diff(T: pd.DataFrame) -> pd.DataFrame:
    P = pd.read_csv(il.OUT / "first_lick_preds.csv.gz")
    P = P.rename(columns={"dec": "dec_lat"})
    T = T.merge(P[["label", "act", "dec_lat"]].drop_duplicates(["label", "act"]), on=["label", "act"], how="left")
    rows = []
    for (a, lab, ep), g in T.groupby(["animal", "label", "epoch"]):
        m, lft = g[g.grp == "MIS"], g[(g.grp == "LEFT")]
        r = dict(animal=a, label=lab, epoch=ep, n_mis=len(m), n_left=len(lft), n_right=int((g.grp == "RIGHT").sum()),
                 s_mis=m.s.mean() if len(m) >= MIN else np.nan, s_left=lft.s.mean() if len(lft) >= MIN else np.nan,
                 s_right=g[g.grp == "RIGHT"].s.mean() if (g.grp == "RIGHT").sum() >= MIN else np.nan)
        if len(m) >= MIN and len(lft) >= MIN:
            b = np.arange(-4, 4.01, 0.25)
            mb, lb = pd.cut(m.act, b), pd.cut(lft.act, b)
            dm, dl = m.groupby(mb, observed=True).dec_lat.mean(), lft.groupby(lb, observed=True).dec_lat.mean()
            cm = m.groupby(mb, observed=True).size()
            both = dm.index.intersection(dl.index)
            if len(both):
                wts = cm.reindex(both).to_numpy(float)
                r["dec_mis_minus_left"] = float(np.average(dm.reindex(both) - dl.reindex(both), weights=wts))
                r["n_matched"] = int(wts.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def plot(S: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.6), squeeze=False)
    for j, a in enumerate(AN):
        g = S[S.animal == a]
        ax = axs[0, j]
        for k, (c, lab, m) in enumerate((("s_left", "left-target licks", "o"), ("s_mis", "far-R licks landing LEFT", "D"),
                                         ("s_right", "far-R licks on target", "s"))):
            for e_i, ep in enumerate(["pre", *EP]):
                v = g[g.epoch == ep][c].dropna()
                if len(v):
                    ax.errorbar(e_i + (k - 1) * 0.18, v.mean(), yerr=v.sem() if len(v) > 1 else 0, fmt=m,
                                color=col[ep], mec="k", mew=0.5, ms=6, capsize=2,
                                label=lab if (e_i == 0 and j == 0) or (ep == "acute" and j == 0 and c == "s_mis") else None)
                    ax.text(e_i + (k - 1) * 0.18, v.mean(), f" {len(v)}", fontsize=6, va="center")
        ax.axhline(0, color="0.5", lw=0.7, ls="--")
        ax.axhline(1, color="0.5", lw=0.7, ls="--")
        ax.set_xticks(range(4), ["pre", *EP])
        ax.set_ylabel("template score (0 = left lick, 1 = right lick)", fontsize=8)
        ax.set_title(f"{a}: cortical pattern of each lick group\n(per-session means +/- SEM; n = sessions)", fontsize=9)
        ax = axs[1, j]
        for e_i, ep in enumerate(EP):
            v = g[g.epoch == ep].dec_mis_minus_left.dropna()
            if len(v):
                ax.errorbar(e_i, v.mean(), yerr=v.sem() if len(v) > 1 else 0, fmt="D", color=col[ep], mec="k", ms=6,
                            capsize=2)
                ax.plot(np.full(len(v), e_i) + 0.15, v, ".", color=col[ep], alpha=0.6)
                ax.text(e_i, ax.get_ylim()[1] if False else v.mean(), f"  n={len(v)}", fontsize=7, va="bottom")
        ax.axhline(0, color="0.5", lw=0.7, ls="--")
        ax.set_xticks(range(3), EP)
        ax.set_ylabel("decoded lateral, far-R-landing-left minus\nleft-target licks with the same landing (mm)", fontsize=8)
        ax.set_title(f"{a}: frozen decoder, landing-matched\n(0 = read as a left lick, + = read toward the right)",
                     fontsize=9)
        for r_ in axs[:, j]:
            for s_ in ("top", "right"):
                r_.spines[s_].set_visible(False)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="center left")
    fig.suptitle("MISDIRECTED far-contra (far R) cued licks that land LEFT: left-lick or right-lick cortex? Row 1: score on "
                 "the animal's pre templates (left-target vs on-target far-R licks, Allen areas, z-scored);\nrow 2: frozen "
                 "lateral decoder vs left-target licks with the same landing, same session. Lesion LEFT; far R = "
                 "contralesional. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT / "figures" / "lick_cued_misdirected_licks.png", dpi=100)
    plt.close(fig)


def main() -> int:
    T, F = lick_table()
    T = scores(classify(T), F)
    S = decoder_diff(T)
    T.drop(columns=[]).to_csv(il.OUT / "misdirected_licks.csv.gz", index=False)
    S.to_csv(il.OUT / "misdirected_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== licks per group x epoch")
    print(T[T.grp != ""].groupby(["animal", "epoch", "grp"]).size().unstack("grp").to_string())
    print("== per animal x epoch (rule 8): mean of session values")
    print(S.groupby(["animal", "epoch"])[["s_left", "s_mis", "s_right", "dec_mis_minus_left"]].mean().round(2).to_string())
    plot(S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
