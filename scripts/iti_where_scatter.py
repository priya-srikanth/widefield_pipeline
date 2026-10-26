"""Decoded vs ACTUAL lick landing in 2-D, per spout and epoch (Priya 10-26: "plot the actual 3d (or 2d) decoded vs true
lick positions for given spout positions per epoch per position").

Same frozen decoder as page 3.15 (`iti_where_decoder.py`): per animal, ridge on the animal's AUGUST pre licks (Allen-area
activity averaged over lags 0..0.2 s from the lick peak), one decoder per axis -- LATERAL (side_ml) and OUTWARD (out_ap),
mm, mouth-relative 3-D tip at the lick peak. Pre sessions are scored leave-one-session-out, post sessions with the full
pre fit. Decoding is PER LICK; the page's r is computed per session over its licks. 2-D, not 3-D: height is not decoded
(the six spouts differ in lateral x outward only).

Groups:
  in-trial licks  by the trial's TARGET spout;
  ITI licks       by the spout their ACTUAL landing is nearest to (each animal's pre in-trial landing mean per spout),
                  since an ITI lick has no target (the spout is retracted).
Per panel, per spout: actual mean (open circle) and decoded mean (filled) with an arrow actual -> decoded and 1-SD
ellipses (actual solid, decoded dashed). Ridge predictions are SHRUNK toward the mean, so decoded points sit closer to
the centre even pre-stroke: read each epoch against the pre column, not against the actual points.
Outputs <E>/iti_licks/where_preds.csv (one row per lick), figures/iti_where_scatter_{trial,iti}.png.
Usage: python scripts/iti_where_scatter.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
POS = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")   # canonical order
PCOL = {"close_ipsi": "#7fb3e0", "close_mid": "#b0b0b0", "close_contra": "#f0a08a",
        "far_ipsi": "#1f5fa8", "far_mid": "#555555", "far_contra": "#c0392b"}


def predictions() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    D = wd.load("areas")
    pos = {x: {"iti": il._load(x)["iti_pos"].astype(str), "trial": il._load(x)["tr_pos"].astype(str)} for x in D}
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            S = {}
            for x, d in D.items():
                if d["animal"] != a:
                    continue
                X, C = d[st]
                ok = np.isfinite(C[:, :2]).all(1)
                if ok.sum() >= 20:
                    S[x] = (d["epoch"], X[ok], C[ok, :2], pos[x][st][ok])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            fits = {}
            for held in [*pre, None]:
                trn = [q for q in pre if q != held]
                Xt, Yt = np.vstack([S[q][1] for q in trn]), np.vstack([S[q][2] for q in trn])
                sc = StandardScaler().fit(Xt)
                fits[held] = (sc, [RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), Yt[:, k]) for k in range(2)])
            for x, (ep, X, Y, P) in S.items():
                sc, ms = fits[x if x in pre else None]
                Pr = np.column_stack([m.predict(sc.transform(X)) for m in ms])
                rows.append(pd.DataFrame({"set": st, "label": x, "animal": a, "epoch": ep, "pos": P,
                                          "act_lat": Y[:, 0], "act_out": Y[:, 1], "dec_lat": Pr[:, 0], "dec_out": Pr[:, 1]}))
    return _groups(pd.concat(rows, ignore_index=True))


def _groups(R: pd.DataFrame) -> pd.DataFrame:
    """Lesion-relative spout names for every lick (in-trial `tr_pos` holds the rig's names, close_L ...; all four lesions
    are LEFT, animals.yaml) and, for ITI licks, the spout nearest the ACTUAL landing (animal's pre in-trial mean)."""
    from wfield_local import orofacial_cohort as oc
    R = R.copy()
    R["pos"] = [p if p in POS or not isinstance(p, str) or "_" not in p else oc.pos_key(p, "L") for p in R.pos]
    cen = R[(R.set == "trial") & (R.epoch == "pre")].groupby(["animal", "pos"])[["act_lat", "act_out"]].mean()
    near = np.full(len(R), "", dtype=object)
    for a in AN:
        if a not in cen.index.get_level_values(0):
            continue
        c = cen.loc[a].reindex(list(POS)).dropna()
        m = (R.animal == a).to_numpy()
        if not m.any() or c.empty:
            continue
        d2 = ((R.loc[m, ["act_lat", "act_out"]].to_numpy()[:, None, :] - c.to_numpy()[None]) ** 2).sum(-1)
        near[m] = np.asarray(c.index)[d2.argmin(1)]
    R["near_pos"] = near
    return R


def _ellipse(ax, xy, color, ls):
    from matplotlib.patches import Ellipse
    if len(xy) < 5:
        return
    C = np.cov(xy.T)
    w, v = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(xy.mean(0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fill=False, ec=color, lw=1.1, ls=ls,
                         alpha=0.8))


def plot(R: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for st, gcol, gname in (("trial", "pos", "TARGET spout"), ("iti", "near_pos", "spout nearest the ACTUAL landing")):
        T = R[R.set == st]
        rows = [("all animals", T), *[(a, T[T.animal == a]) for a in AN]]
        fig, axs = plt.subplots(len(rows), len(EP), figsize=(4.3 * len(EP), 3.9 * len(rows)), squeeze=False,
                                sharex="row", sharey="row")
        for i, (rname, g) in enumerate(rows):
            for j, ep in enumerate(EP):
                ax = axs[i, j]
                h = g[g.epoch == ep]
                for p in POS:
                    q = h[h[gcol] == p]
                    if len(q) < 5:
                        continue
                    A, Dd = q[["act_lat", "act_out"]].to_numpy(), q[["dec_lat", "dec_out"]].to_numpy()
                    _ellipse(ax, A, PCOL[p], "-")
                    _ellipse(ax, Dd, PCOL[p], "--")
                    ma, md = A.mean(0), Dd.mean(0)
                    ax.annotate("", md, ma, arrowprops=dict(arrowstyle="->", color=PCOL[p], lw=1.6))
                    ax.scatter(*ma, s=55, facecolor="w", edgecolor=PCOL[p], lw=2, zorder=4)
                    ax.scatter(*md, s=55, color=PCOL[p], zorder=5)
                ax.set_aspect("equal", adjustable="box")
                ax.axhline(0, color="0.9", lw=0.6, zorder=0)
                ax.axvline(0, color="0.9", lw=0.6, zorder=0)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                ax.set_title(f"{rname} | {ep} ({h.label.nunique()} sessions, {len(h)} licks)", fontsize=9)
                if i == len(rows) - 1:
                    ax.set_xlabel("lateral tip position side_ml (mm)", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("outward tip position out_ap (mm)", fontsize=8.5)
        hd = [Line2D([], [], marker="o", ls="", color=PCOL[p], label=p.replace("_", " ")) for p in POS]
        hd += [Line2D([], [], marker="o", ls="", mfc="w", mec="k", label="actual mean (solid ellipse = 1 SD)"),
               Line2D([], [], marker="o", ls="", color="k", label="decoded mean (dashed ellipse = 1 SD)")]
        fig.legend(handles=hd, loc="upper center", ncol=8, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.965))
        fig.suptitle(f"NEURAL: decoded vs actual lick landing, {'in-trial' if st == 'trial' else 'ITI (self-initiated)'} licks, "
                     f"grouped by {gname}.\nFROZEN pre-stroke decoder (fit once on each animal's August pre licks, never refit; "
                     "one ridge per axis, Allen-area features, lags 0..0.2 s);\nper lick, decoded in 2-D (lateral x outward, "
                     "mm). Arrow = actual mean -> decoded mean. Ridge shrinks predictions toward the centre, so compare each "
                     "epoch with PRE, not with the actual points.", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(il.OUT / "figures" / f"iti_where_scatter_{st}.png", dpi=95)
        plt.close(fig)


def main() -> int:
    if "--plot-only" in sys.argv:
        plot(_groups(pd.read_csv(il.OUT / "where_preds.csv")))
        return 0
    R = predictions()
    R.to_csv(il.OUT / "where_preds.csv", index=False)
    pd.set_option("display.width", 200)
    print("== licks per set / epoch / animal (rule 8)")
    print(R.groupby(["set", "animal", "epoch"]).size().unstack("epoch").reindex(columns=EP).to_string())
    plot(R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
