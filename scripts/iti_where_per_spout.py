"""Per-spout decoding error and r (Priya 10-26: "should we look at r and lateral / distance error also per spout
position? (ITI using spout position preceding the ITI)").

Frozen where-decoder per lick (`where_preds.csv`, from `iti_where_scatter.py`). Grouping: in-trial licks by the trial's
TARGET spout; ITI licks by the PREVIOUS trial's spout (`iti_pos`, fixed before the lick -- unlike the nearest-spout
grouping of 3.20 / 3.21, which depends on where the lick landed). Per spout, change vs pre:
  beyond_lat / beyond_sag  decoded minus where the animal's pre-stroke pull alone would put the lick (per animal x set x
                           axis, decoded = a + b * actual fit on pre licks; `iti_where_beyond_pull.py`); + lateral =
                           contralesional, - sagittal = closer to the mouth;
  r_lat                    per-session r between decoded and actual lateral position within the spout's licks
                           (within-spout range is small, so r is low and spread-sensitive; read the offsets first).
Cells need >= 15 licks per session x spout. Per-animal table first; paired nested bootstrap (animals > sessions).
`--z` uses the within-session z-scored predictions (`iti_zscore_check.py`).
Outputs <E>/iti_licks/per_spout[_z]_{by_animal,tests}.csv, figures/lick_where_per_spout[_z].png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

POS = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")   # canonical order
EP = ["acute", "subacute", "chronic"]
MEAS = {"beyond_lat": "LATERAL offset, mm\n(+ = read contra)",
        "beyond_sag": "SAGITTAL offset, mm\n(- = read closer)",
        "r_lat": "frozen r, lateral\n(within spout)"}
EPC = {"acute": "#e7298a", "subacute": "#d4a017", "chronic": "#1b9e77"}
MIN_N = 15


def table(p: pd.DataFrame) -> pd.DataFrame:
    p = p.dropna(subset=["act_lat", "act_sag", "dec_lat", "dec_sag"]).copy()
    for _key, g in p.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        for ax_ in ("lat", "sag"):
            b, c = np.polyfit(pre[f"act_{ax_}"], pre[f"dec_{ax_}"], 1)
            p.loc[g.index, f"beyond_{ax_}"] = g[f"dec_{ax_}"] - (c + b * g[f"act_{ax_}"])
    p = p[p.pos.isin(POS)]

    def cell(g):
        r = np.corrcoef(g.dec_lat, g.act_lat)[0, 1] if g.act_lat.std() > 0 else np.nan
        return pd.Series({"beyond_lat": g.beyond_lat.mean(), "beyond_sag": g.beyond_sag.mean(), "r_lat": r,
                          "n": len(g)})
    s = p.groupby(["set", "animal", "epoch", "label", "pos"]).apply(cell, include_groups=False).reset_index()
    return s[s.n >= MIN_N]


def tests(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per = [], []
    for (st, pos), g in s.groupby(["set", "pos"]):
        for m in MEAS:
            prd = {a: x[m].dropna().to_numpy() for a, x in g[g.epoch == "pre"].groupby("animal")}
            pre_a = g[g.epoch == "pre"].groupby("animal")[m].mean()
            for ep in EP:
                h = g[g.epoch == ep]
                post = {a: x[m].dropna().to_numpy() for a, x in h.groupby("animal")}
                post = {a: v for a, v in post.items() if len(v) and a in prd and len(prd[a])}
                am = h.groupby("animal")[m].mean() - pre_a
                for a, v in am.dropna().items():
                    per.append(dict(set=st, pos=pos, measure=m, epoch=ep, animal=a, change=v))
                ci = ak.boot_delta(post, prd, np.random.default_rng(0)) if len(post) >= 2 else None
                rows.append(dict(set=st, pos=pos, measure=m, epoch=ep,
                                 point=ci.point if ci else np.nan, lo=ci.lo if ci else np.nan,
                                 hi=ci.hi if ci else np.nan, n_pos=int((am.dropna() > 0).sum()),
                                 n=int(am.notna().sum())))
    return pd.DataFrame(rows), pd.DataFrame(per)


def session_changes(s: pd.DataFrame) -> pd.DataFrame:
    """Per session x spout: value minus its animal's PRE mean for that spout and set (pre sessions included, so their
    spread is visible)."""
    out = []
    for m in MEAS:
        pre = s[s.epoch == "pre"].groupby(["set", "animal", "pos"])[m].mean().rename("pre_m")
        x = s.join(pre, on=["set", "animal", "pos"]).dropna(subset=["pre_m", m])
        out.append(x.assign(measure=m, change=x[m] - x.pre_m)[["set", "animal", "epoch", "label", "pos", "measure",
                                                                 "change"]])
    return pd.concat(out, ignore_index=True)


def plot(T: pd.DataFrame, S: pd.DataFrame, tag: str) -> None:
    """3.9 layout: one dot per SESSION (change from its animal's pre mean, all animals pooled), bar = mean +- SEM over
    sessions, * = paired nested bootstrap (animals > sessions) excluding 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep4 = ["pre", *EP]
    col = {"pre": "0.25", **EPC}
    rows = [(st, m) for st in ("trial", "iti") for m in MEAS]
    fig, axs = plt.subplots(len(rows), len(POS), figsize=(2.9 * len(POS), 2.5 * len(rows)), squeeze=False, sharey="row")
    sl = {"trial": "CUED (by target)", "iti": "ITI (by previous spout)"}
    rng = np.random.default_rng(0)
    for i, (st, m) in enumerate(rows):
        for j, pos in enumerate(POS):
            ax = axs[i, j]
            ax.axhline(0, color="0.6", lw=0.7)
            for k, ep in enumerate(ep4):
                v = S[(S.set == st) & (S.pos == pos) & (S.measure == m) & (S.epoch == ep)].change.to_numpy()
                if not len(v):
                    continue
                ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=8, color=col[ep], alpha=0.45, lw=0)
                mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                ax.errorbar(k + 0.3, mu, yerr=se, fmt="o", color="k", ms=4, capsize=2, lw=1.2)
                t = T[(T.set == st) & (T.pos == pos) & (T.measure == m) & (T.epoch == ep)]
                if ep != "pre" and len(t) and np.isfinite(t.point.iloc[0]) and (t.lo.iloc[0] > 0 or t.hi.iloc[0] < 0):
                    ax.text(k, 1.0, "*", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
            ax.set_xticks(range(len(ep4)))
            ax.set_xticklabels(["pre", "acu", "sub", "chr"], fontsize=8)
            if i == 0:
                ax.set_title(pos.replace("_", " "), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sl[st]}\n{MEAS[m]}" + ("\n[decoder on Z-SCORED features]" if tag else ""), fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle(("Z-SCORED FEATURES (each area / its frame-wise SD in the session, pre and post alike) -- " if tag else "")
                 + "NEURAL: frozen pre-stroke lick-position decoder PER SPOUT (Allen areas, lags 0..0.2 s)."
                 "\nOne dot per SESSION = change from its "
                 "animal's pre mean (all animals pooled; pre sessions show the pre spread); black = mean +- SEM over "
                 "sessions;\n* = paired nested bootstrap (animals > sessions) excludes 0. Cued licks grouped by target, "
                 "ITI licks by the previous trial's spout (fixed before the lick).\nOffset = decoded minus where "
                 "the animal's pre-stroke pull toward the middle alone would put the lick.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(il.OUT / "figures" / f"lick_where_per_spout{tag}.png", dpi=95)
    plt.close(fig)


def main() -> int:
    z = "--z" in sys.argv
    tag = "_z" if z else ""
    p = pd.read_csv(il.OUT / "zscore_check" / "where_preds_z.csv" if z else il.OUT / "where_preds.csv")
    s = table(p)
    T, P = tests(s)
    pd.set_option("display.width", 230)
    print("== per animal (rule 8): change vs pre, chronic")
    print(P[P.epoch == "chronic"].pivot_table(index=["set", "measure", "pos"], columns="animal", values="change")
          .reindex(POS, level=2).round(2).to_string())
    print("\n== pooled (nested bootstrap)")
    print(T.pivot_table(index=["set", "measure", "pos"], columns="epoch", values="point").reindex(columns=EP)
          .reindex(POS, level=2).round(2).to_string())
    sig = T[(T.lo > 0) | (T.hi < 0)]
    print("\n== cells whose CI excludes 0\n" + sig.round(3).to_string(index=False))
    P.to_csv(il.OUT / f"per_spout{tag}_by_animal.csv", index=False)
    T.to_csv(il.OUT / f"per_spout{tag}_tests.csv", index=False)
    S = session_changes(s)
    S.to_csv(il.OUT / f"per_spout{tag}_by_session.csv", index=False)
    plot(T, S, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
