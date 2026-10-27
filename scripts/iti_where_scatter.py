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
                Yc = C[:, [0, 3, 1]]                      # lateral, SAGITTAL extension, outward (wd.coords columns)
                ok = np.isfinite(Yc).all(1)
                if ok.sum() >= 20:
                    S[x] = (d["epoch"], X[ok], Yc[ok], pos[x][st][ok])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            fits = {}
            for held in [*pre, None]:
                trn = [q for q in pre if q != held]
                Xt, Yt = np.vstack([S[q][1] for q in trn]), np.vstack([S[q][2] for q in trn])
                sc = StandardScaler().fit(Xt)
                fits[held] = (sc, [RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), Yt[:, k]) for k in range(3)])
            for x, (ep, X, Y, P) in S.items():
                sc, ms = fits[x if x in pre else None]
                Pr = np.column_stack([m.predict(sc.transform(X)) for m in ms])
                rows.append(pd.DataFrame({"set": st, "label": x, "animal": a, "epoch": ep, "pos": P,
                                          "act_lat": Y[:, 0], "act_sag": Y[:, 1], "act_out": Y[:, 2],
                                          "dec_lat": Pr[:, 0], "dec_sag": Pr[:, 1], "dec_out": Pr[:, 2]}))
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


def _ellipse(ax, xy, color, ls, filled=False):
    """1-SD ellipse; ``filled`` = the ACTUAL landing spread (light shading), else a thin faint outline (decoded)."""
    from matplotlib.patches import Ellipse
    if len(xy) < 5:
        return
    C = np.cov(xy.T)
    w, v = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    kw = (dict(fill=True, fc=color, ec=color, alpha=0.13, lw=0.6) if filled
          else dict(fill=False, ec=color, lw=0.8, ls=ls, alpha=0.45))
    ax.add_patch(Ellipse(xy.mean(0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, **kw))


#: groups for the error-change figure: (set, grouping column, licks), "pre landing" = licks within 0.25 mm of the animal's
#: pre landing mean for their spout (actual position held fixed; see iti_where_signed_error.py)
DELTA_ROWS = (("trial", "pos", "all"), ("trial", "pos", "on"), ("iti", "near_pos", "all"), ("iti", "near_pos", "on"))


def error_change(R: pd.DataFrame) -> pd.DataFrame:
    """Per session x spout: decoded - actual (lateral, sagittal) and the actual / decoded means, MINUS the animal's pre
    value for that spout (pre sessions minus the pre mean too, for the bootstrap). + lateral = contralesional; - sagittal =
    decoded closer to the mouth than the lick. Subtracting pre removes the ridge's shrinkage toward the centre."""
    out = []
    for st, gcol, sel in DELTA_ROWS:
        T = R[R.set == st].copy()
        T["spout"] = T[gcol]
        T = T[T.spout.isin(POS)]
        land = T[T.epoch == "pre"].groupby(["animal", "spout"])[["act_lat", "act_sag"]].mean()
        T = T.join(land, on=["animal", "spout"], rsuffix="_pl")
        if sel == "on":
            T = T[np.hypot(T.act_lat - T.act_lat_pl, T.act_sag - T.act_sag_pl) < 0.25]
        T["e_lat"], T["e_sag"] = T.dec_lat - T.act_lat, T.dec_sag - T.act_sag
        V = ["e_lat", "e_sag", "act_lat", "act_sag", "dec_lat", "dec_sag"]
        s = T.groupby(["animal", "epoch", "label", "spout"])[V].mean().join(
            T.groupby(["animal", "epoch", "label", "spout"]).size().rename("n")).reset_index()
        s = s[s.n >= 5]
        pre = s[s.epoch == "pre"].groupby(["animal", "spout"])[V].mean()
        s = s.join(pre, on=["animal", "spout"], rsuffix="_pre").dropna(subset=["e_lat_pre"])
        for v in V:
            s[f"d_{v}"] = s[v] - s[f"{v}_pre"]
        out.append(s.assign(set=st, licks=sel))
    return pd.concat(out, ignore_index=True)


def plot_delta(R: pd.DataFrame) -> None:
    """Change in the decoding ERROR vector vs pre (Priya 10-26: 'a normalized to pre figure of the delta direction
    change'): per row (lick set x lick selection) and epoch, one dot per spout (mean of animals), faint = animals, black
    cross = all spouts pooled with 95 % nested CI (animals > sessions) on each axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import analysis_kit as ak
    S = error_change(R)
    eps = EP[1:]
    names = {("trial", "all"): "in-trial licks, all (by TARGET)", ("trial", "on"): "in-trial licks landing at the pre spot",
             ("iti", "all"): "ITI licks, all (by nearest spout)", ("iti", "on"): "ITI licks landing at the pre spot"}
    tests, per_animal = [], []
    fig, axs = plt.subplots(len(DELTA_ROWS), len(eps), figsize=(4.2 * len(eps), 3.9 * len(DELTA_ROWS)), squeeze=False,
                            sharex=True, sharey=True)
    for i, (st, _g, sel) in enumerate(DELTA_ROWS):
        G = S[(S.set == st) & (S.licks == sel)]
        ses = G.groupby(["animal", "epoch", "label"])[["d_e_lat", "d_e_sag", "d_act_lat", "d_act_sag", "d_dec_lat",
                                                        "d_dec_sag"]].mean().reset_index()
        pa = ses[ses.epoch.isin(eps)].groupby(["animal", "epoch"]).mean(numeric_only=True)
        per_animal.append(pa.assign(set=st, licks=sel).reset_index())
        for j, ep in enumerate(eps):
            ax = axs[i, j]
            ax.axhline(0, color="0.75", lw=0.7)
            ax.axvline(0, color="0.75", lw=0.7)
            h = G[G.epoch == ep]
            for p in POS:
                q = h[h.spout == p].groupby("animal")[["d_e_lat", "d_e_sag"]].mean()
                if q.empty:
                    continue
                ax.scatter(q.d_e_lat, q.d_e_sag, s=12, color=PCOL[p], alpha=0.35, lw=0)
                ax.scatter(q.d_e_lat.mean(), q.d_e_sag.mean(), s=60, color=PCOL[p], ec="k", lw=0.5, zorder=4)
            ci = {}
            for ax_ in ("lat", "sag"):
                post = {a: g[f"d_e_{ax_}"].to_numpy() for a, g in ses[ses.epoch == ep].groupby("animal")}
                prd = {a: g[f"d_e_{ax_}"].to_numpy() for a, g in ses[ses.epoch == "pre"].groupby("animal")}
                ci[ax_] = ak.boot_delta(post, prd, np.random.default_rng(0))
            if ci["lat"] is not None and ci["sag"] is not None:
                ax.errorbar(ci["lat"].point, ci["sag"].point, xerr=[[ci["lat"].point - ci["lat"].lo],
                                                                    [ci["lat"].hi - ci["lat"].point]],
                            yerr=[[ci["sag"].point - ci["sag"].lo], [ci["sag"].hi - ci["sag"].point]],
                            fmt="s", color="k", ms=6, capsize=3, lw=1.4, zorder=6)
                star = lambda c: "*" if c.lo > 0 or c.hi < 0 else ""  # noqa: E731
                ax.text(0.03, 0.97, f"lateral {ci['lat'].point:+.2f}{star(ci['lat'])}\nsagittal {ci['sag'].point:+.2f}"
                        f"{star(ci['sag'])} mm", transform=ax.transAxes, va="top", fontsize=8.5)
                for ax_ in ("lat", "sag"):
                    tests.append(dict(set=st, licks=sel, epoch=ep, axis=ax_, point=ci[ax_].point, lo=ci[ax_].lo,
                                      hi=ci[ax_].hi, n_pos=int((pa.xs(ep, level="epoch")[f"d_e_{ax_}"] > 0).sum()),
                                      n=ci[ax_].n_animals))
            ax.set_title(f"{names[(st, sel)]} | {ep}", fontsize=9)
            if i == len(DELTA_ROWS) - 1:
                ax.set_xlabel("change in LATERAL error (mm)\n<- decoded more ipsi | decoded more CONTRA ->", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("change in SAGITTAL error (mm)\n<- decoded closer | decoded further ->", fontsize=8.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    hd = [Line2D([], [], marker="o", ls="", color=PCOL[p], mec="k", mew=0.5, label=p.replace("_", " ")) for p in POS]
    hd += [Line2D([], [], marker="o", ls="", color="0.6", alpha=0.4, label="each animal"),
           Line2D([], [], marker="s", ls="-", color="k", label="all spouts, 95% nested CI (* excludes 0)")]
    fig.suptitle("NEURAL: how the decoded lick position moved relative to the real one, vs PRE. FROZEN pre-stroke decoder "
                 "(Allen areas, lags 0..0.2 s).\nPer spout: (decoded - actual) in the epoch minus the same in pre (removes "
                 "the decoder's pre-stroke shrinkage toward the centre). 0,0 = same error as before the stroke.",
                 fontsize=10, y=0.995)
    fig.legend(handles=hd, loc="upper center", ncol=8, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(il.OUT / "figures" / "iti_where_delta.png", dpi=95)
    plt.close(fig)
    pd.set_option("display.width", 220)
    P = pd.concat(per_animal, ignore_index=True)
    print("== per animal (rule 8): change vs pre, mm (+ lateral = contra, - sagittal = closer)")
    print(P.round(2).to_string(index=False))
    P.to_csv(il.OUT / "where_delta_by_animal.csv", index=False)
    Tt = pd.DataFrame(tests)
    print(Tt.round(3).to_string(index=False))
    Tt.to_csv(il.OUT / "where_delta_tests.csv", index=False)


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
                    A, Dd = q[["act_lat", "act_sag"]].to_numpy(), q[["dec_lat", "dec_sag"]].to_numpy()
                    _ellipse(ax, A, PCOL[p], "-", filled=True)
                    _ellipse(ax, Dd, PCOL[p], ":")
                    ma, md = A.mean(0), Dd.mean(0)
                    ax.annotate("", md, ma, arrowprops=dict(arrowstyle="->", color=PCOL[p], lw=1.6))
                    ax.scatter(*ma, s=50, facecolor="w", edgecolor=PCOL[p], lw=2, zorder=4)
                    ax.scatter(*md, s=50, color=PCOL[p], zorder=5)
                ax.set_aspect("equal", adjustable="box")
                ax.axhline(0, color="0.9", lw=0.6, zorder=0)
                ax.axvline(0, color="0.9", lw=0.6, zorder=0)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                ax.set_title(f"{rname} | {ep} ({h.label.nunique()} sessions, {len(h)} licks)", fontsize=9)
                if i == len(rows) - 1:
                    ax.set_xlabel("lateral tip position side_ml (mm)", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("sagittal extension (mm)", fontsize=8.5)
        from matplotlib.patches import Patch
        hd = [Line2D([], [], marker="o", ls="", color=PCOL[p], label=p.replace("_", " ")) for p in POS]
        hd += [Line2D([], [], marker="o", ls="", mfc="w", mec="k", mew=2, label="open circle = where the licks LANDED (mean)"),
               Line2D([], [], marker="o", ls="", color="k", label="filled dot = where cortex DECODED them (mean)"),
               Line2D([], [], marker=r"$\rightarrow$", ls="", color="k", ms=12, label="arrow = landed -> decoded"),
               Patch(fc="0.5", alpha=0.25, label="shaded ellipse = landing spread (1 SD)"),
               Line2D([], [], ls=":", color="k", alpha=0.6, label="dotted ellipse = decoded spread (1 SD)")]
        fig.suptitle(f"NEURAL: decoded vs actual lick landing, {'in-trial' if st == 'trial' else 'ITI (self-initiated)'} licks, "
                     f"grouped by {gname}.\nFROZEN pre-stroke decoder (fit once on each animal's August pre licks, never refit; "
                     "one ridge per axis, Allen-area features, lags 0..0.2 s); per lick, lateral x sagittal extension (mm), "
                     "+ lateral = contralesional.\nThe decoder shrinks predictions toward the centre even before the stroke, "
                     "so compare each epoch with PRE (change figure: iti_where_delta.png).", fontsize=10.5, y=0.997)
        fig.legend(handles=hd, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.962))
        fig.tight_layout(rect=(0, 0, 1, 0.925))
        fig.savefig(il.OUT / "figures" / f"iti_where_scatter_{st}.png", dpi=95)
        plt.close(fig)


def main() -> int:
    if "--plot-only" in sys.argv:
        R = _groups(pd.read_csv(il.OUT / "where_preds.csv"))
        plot(R)
        plot_delta(R)
        return 0
    R = predictions()
    R.to_csv(il.OUT / "where_preds.csv", index=False)
    pd.set_option("display.width", 200)
    print("== licks per set / epoch / animal (rule 8)")
    print(R.groupby(["set", "animal", "epoch"]).size().unstack("epoch").reindex(columns=EP).to_string())
    plot(R)
    plot_delta(R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
