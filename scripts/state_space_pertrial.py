"""Per-trial views of the state-space results (Priya 11-07: "should some plots have per-trial (instead of per-session)
information" -> "add the 3 worth showing per trial"). Alignment from WF_SS_ALIGN; first-lick trials (contact +
incomplete); statistics stay at the session level (trials are not independent units).

A. DAYS PLANE, PER TRIAL: every trial's (x, y) in the per-spout frames of state_space_trajectories (x = pre (0) ->
   chronic (1) line, y = acute departure orthogonal to it, pre -> chronic units); per animal and pooled, 50 % and 90 %
   highest-density contours per epoch (smoothed 2-D histogram). Does the acute detour move the whole cloud or a subset?
B. WITHIN SESSION: each trial's x and y against its order in the session (trial rank / (n - 1), quintiles); per session
   the least-squares slope of x and y on trial order. Pre sessions = the within-session drift that is not stroke.
C. RESPONSE SIZE PER TRIAL: full-space (62 areas) response norm per trial (state_space_sim.BAL_BINS response minus
   baseline), log2 vs the animal x spout pre median; ECDFs per spout (thin = sessions, thick = epoch pool) and the
   change of the 10th / 50th / 90th percentile vs own pre (uniform shift vs a subset of large trials).
Output: Widefield/state_space_recovery/pertrial{tag}/, page_figs/ss_pt_{plane,order,size}_<suffix>.png. Exploratory.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import state_space_contact as C  # noqa: E402
import state_space_page_figs as F  # noqa: E402
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import epochs  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402  (PNG + svg/ twin, Priya 11-06)

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
OUT = ROOT / f"pertrial{S.tag()}"
FIGS = ROOT / "page_figs"
SUF = S.tag().lstrip("_") or "cue"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def collect():
    base, resp = S.BAL_BINS[S.ALIGN]
    data = {}
    for f in sorted(glob.glob(S.ITI + "/PS9*.npz")):
        Y, pos, cols, ep, an = S.load(f)
        lab = Path(f).stem
        data.setdefault(an, []).append((lab, ep, epochs.days_since_stroke(lab), Y, pos))
    rows = []
    for an, sess in sorted(data.items()):
        sess = sorted(sess, key=lambda s: s[2])
        P, mu, fr = C.frames(sess)
        for lab, ep, day, Y, pos in sess:
            n = len(pos)
            order = np.arange(n) / max(n - 1, 1)
            size = np.linalg.norm(Y[:, resp].mean(1) - Y[:, base].mean(1), axis=1)
            z = ((Y - mu) @ P).reshape(n, -1)
            for p, (zp, u, v, L, _) in fr.items():
                m = np.flatnonzero(pos == p)
                if len(m) < R.MIN_TRIALS:
                    continue
                q = z[m] - zp
                for i, j in enumerate(m):
                    rows.append({"animal": an, "label": lab, "epoch": ep, "day": day, "spout": p, "trial": int(j),
                                 "order": float(order[j]), "x": float(q[i] @ u / L), "y": float(q[i] @ v / L),
                                 "size": float(size[j])})
    D = pd.DataFrame(rows)
    pre_med = D[D.epoch == "pre"].groupby(["animal", "spout"])["size"].median()
    D["lsize"] = np.log2(D["size"].to_numpy() / D.set_index(["animal", "spout"]).index.map(pre_med).to_numpy())
    return D


def hdr_contours(ax, x, y, color, lims, lw=1.8):
    """50 % (solid) and 90 % (dashed) highest-density regions of a smoothed 2-D histogram."""
    if len(x) < 30:
        return
    H, xe, ye = np.histogram2d(x, y, bins=80, range=lims)
    H = gaussian_filter(H, 2.0)
    Hs = np.sort(H.ravel())[::-1]
    cs = np.cumsum(Hs) / Hs.sum()
    lv = [Hs[np.searchsorted(cs, q)] for q in (0.9, 0.5)]
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, H.T, levels=sorted(set(lv)), colors=[color], linewidths=[lw * 0.6, lw],
               linestyles=["--", "-"] if len(set(lv)) == 2 else ["-"])


def fig_plane(D):
    lims = [[-2.5, 3.5], [-2.0, 3.0]]
    an_list = sorted(D.animal.unique())
    fig, axs = plt.subplots(1, 5, figsize=(19, 4.2), sharex=True, sharey=True)
    for ax, key in zip(axs, ["all animals"] + an_list):
        g = D if key == "all animals" else D[D.animal == key]
        for ep in R.EPOCHS:
            e = g[g.epoch == ep]
            hdr_contours(ax, e.x.to_numpy(), e.y.to_numpy(), EPOCH_LINE[ep], lims)
            ax.plot(e.x.mean(), e.y.mean(), "o", color=EPOCH_LINE[ep], ms=6, mec="w", zorder=5)
        for xx in (0, 1):
            ax.axvline(xx, color="0.8", lw=0.8, zorder=0)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(key)
        ax.set_xlabel("along pre (0) -> chronic (1)\n(fraction of pre -> chronic distance)")
    axs[0].set_ylabel("acute departure, orthogonal\n(same units)")
    hs = [plt.Line2D([], [], color=EPOCH_LINE[e], lw=2, label=e) for e in R.EPOCHS]
    hs += [plt.Line2D([], [], color="0.3", lw=2, label="50 % of trials"),
           plt.Line2D([], [], color="0.3", lw=1, ls="--", label="90 % of trials"),
           plt.Line2D([], [], marker="o", color="0.3", lw=0, label="epoch mean")]
    fig.legend(handles=hs, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(f"Every trial in the pre -> chronic / acute-departure plane: highest-density regions per epoch "
                 f"({F.ALIGN_LABEL[S.ALIGN]})", y=1.13)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_pt_plane_{SUF}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def slopes(D):
    out = []
    for (an, lab, ep, day), g in D.groupby(["animal", "label", "epoch", "day"]):
        if len(g) < 20:
            continue
        r = {"animal": an, "label": lab, "epoch": ep, "day": day}
        for c in ("x", "y", "lsize"):
            # within-spout: remove each spout's session mean so the slope is not spout composition over the session
            v = g[c] - g.groupby("spout")[c].transform("mean")
            r[f"slope_{c}"] = float(np.polyfit(g.order, v, 1)[0])
        out.append(r)
    return pd.DataFrame(out)


def fig_order(D, Sl):
    rng = np.random.default_rng(0)
    D = D.assign(q=np.minimum((D.order * 5).astype(int), 4))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), gridspec_kw={"width_ratios": [1.3, 1.3, 1, 1]})
    for ax, c, t in ((axs[0], "x", "position pre (0) -> chronic (1)\n(fraction of pre -> chronic distance)"),
                     (axs[1], "y", "acute-departure axis\n(same units)")):
        for k, ep in enumerate(R.EPOCHS):
            s = D[D.epoch == ep].groupby(["animal", "label", "q"])[c].mean().reset_index()
            pts, lo, hi = [], [], []
            for q in range(5):
                h = s[s.q == q]
                iv = F.kit.boot_ci({a: g[c].tolist() for a, g in h.groupby("animal")}, rng)
                pts.append(iv.point)
                lo.append(iv.lo)
                hi.append(iv.hi)
            xq = np.arange(5) + (k - 1.5) * 0.06
            ax.fill_between(xq, lo, hi, color=EPOCH_LINE[ep], alpha=0.15, lw=0)
            ax.plot(xq, pts, "o-", color=EPOCH_LINE[ep], ms=4, lw=1.6, label=ep)
        ax.set_xticks(range(5), ["1st", "2nd", "3rd", "4th", "5th"])
        ax.set_xlabel("fifth of the session (trial order)")
        ax.set_title(t)
    axs[0].legend(frameon=False, fontsize=7.5)
    for ax, c, t in ((axs[2], "slope_x", "within-session slope of position\n(\u0394 per session, vs own pre)"),
                     (axs[3], "slope_y", "within-session slope, acute-departure axis\n(\u0394 per session, vs own pre)")):
        F.epoch_panel(ax, Sl, c, t, rng)
        ax.set_ylabel("")
    F.legend_animals_epochs(fig, y=1.08)
    fig.suptitle(f"Within a session: does the trial cloud move from the first to the last trials? "
                 f"({F.ALIGN_LABEL[S.ALIGN]}; within spout)", y=1.15)
    fig.tight_layout()
    F.flush_marks(fig)
    F.sig_marks.reserve_headroom(fig)
    save_png_svg(fig, FIGS / f"ss_pt_order_{SUF}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_size(D):
    rng = np.random.default_rng(0)
    xs = np.linspace(-3, 3, 241)
    fig, axs = plt.subplots(2, 6, figsize=(18, 6.6), gridspec_kw={"height_ratios": [1.3, 1]})
    qrows = []
    for j, p in enumerate(S.POS):
        ax = axs[0, j]
        g = D[D.spout == p]
        for ep in R.EPOCHS:
            e = g[g.epoch == ep]
            for _, s in e.groupby("label"):
                ax.plot(xs, np.searchsorted(np.sort(s.lsize), xs) / len(s), color=EPOCH_LINE[ep], lw=0.4, alpha=0.25)
            if len(e):
                ax.plot(xs, np.searchsorted(np.sort(e.lsize), xs) / len(e), color=EPOCH_LINE[ep], lw=2.2)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_title(p.replace("_", " "))
        ax.set_xlabel("log2 response size vs own pre median")
        # percentile shifts per session -> change vs own pre
        for (an, lab, ep), s in g.groupby(["animal", "label", "epoch"]):
            if len(s) >= 10:
                qrows.append({"animal": an, "label": lab, "epoch": ep, "spout": p,
                              **{f"q{q}": float(np.percentile(s.lsize, q)) for q in (10, 50, 90)}})
    axs[0, 0].set_ylabel("cumulative share of trials\n(thin = sessions, thick = epoch)")
    Q = pd.DataFrame(qrows)
    for j, p in enumerate(S.POS):
        ax = axs[1, j]
        ax.axhline(0, color="0.6", lw=0.8)
        g = Q[Q.spout == p]
        for k, qn in enumerate(("q10", "q50", "q90")):
            ch = F.change(g, qn, rng)
            for i, ep in enumerate(("acute", "subacute", "chronic")):
                pt, lo, hi, up, n = ch[ep]
                x = i + (k - 1) * 0.25
                ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt=["v", "o", "^"][k], color=EPOCH_LINE[ep], ms=5,
                            lw=1.5, mfc="w" if k != 1 else EPOCH_LINE[ep])
        ax.set_xticks(range(3), ["acute", "subacute", "chronic"], fontsize=8)
        if j == 0:
            ax.set_ylabel("\u0394 percentile vs own pre\n(log2 response size)")
    hs = [plt.Line2D([], [], marker=m, ls="", color="0.3", mfc=f, label=lb) for m, f, lb in
          (("v", "w", "10th percentile"), ("o", "0.3", "median"), ("^", "w", "90th percentile"))]
    hs += [plt.Line2D([], [], color=EPOCH_LINE[e], lw=2, label=e) for e in R.EPOCHS]
    fig.legend(handles=hs, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Per-trial response size in the full 62-area space, per spout ({F.ALIGN_LABEL[S.ALIGN]})", y=1.07)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_pt_size_{SUF}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Q


def fig_order_aim(D, Sl):
    """Within-session trial order, colored by RECOVERY (Priya 11-07: "with color grading on the recovery axis (eg
    viridis)"): each post-stroke session's per-fifth mean x / y as a thin line colored by its far-contra aim vs own pre
    (deg; viridis, yellow = recovered), pre sessions gray; right: each session's within-session slope against its aim."""
    import matplotlib as mpl

    import state_space_step_vs_gradual as SG
    aim = pd.read_csv(SG.AIM)[["label", "farcontra_first"]].rename(columns={"farcontra_first": "aim"})
    D = D.assign(q=np.minimum((D.order * 5).astype(int), 4)).merge(aim, on="label", how="left")
    Sl = Sl.merge(aim, on="label", how="left")
    post = D[D.epoch != "pre"]
    norm = mpl.colors.Normalize(vmin=np.nanpercentile(post.aim, 2), vmax=np.nanpercentile(post.aim, 98))
    cmap = mpl.colormaps["viridis"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, c, t in ((axs[0], "x", "position pre (0) -> chronic (1)\n(fraction of pre -> chronic distance)"),
                     (axs[1], "y", "acute-departure axis\n(same units)")):
        s = D.groupby(["label", "epoch", "q"])[c].mean().reset_index().merge(aim, on="label", how="left")
        for (_lab, ep), g in s.groupby(["label", "epoch"]):
            g = g.sort_values("q")
            col = "0.75" if ep == "pre" else cmap(norm(g.aim.iloc[0])) if np.isfinite(g.aim.iloc[0]) else "0.5"
            ax.plot(g.q, g[c], color=col, lw=0.9 if ep != "pre" else 0.6, alpha=0.85, zorder=1 if ep == "pre" else 2)
        ax.set_xticks(range(5), ["1st", "2nd", "3rd", "4th", "5th"])
        ax.set_xlabel("fifth of the session (trial order)")
        ax.set_title(t)
    for ax, c, t in ((axs[2], "slope_x", "within-session slope of position\n(per session, first -> last trial)"),
                     (axs[3], "slope_y", "within-session slope, acute-departure axis\n(per session)")):
        ax.axhline(0, color="0.6", lw=0.8)
        pre = Sl[Sl.epoch == "pre"]
        ax.scatter(pre.aim, pre[c], color="0.75", s=14, zorder=1)
        for an, g in Sl[Sl.epoch != "pre"].groupby("animal"):
            ax.scatter(g.aim, g[c], marker=F.ANIMAL_MARKER[an], c=cmap(norm(g.aim)), s=26, edgecolors="k", lw=0.4,
                       zorder=2)
        ax.set_xlabel("far-contra aim vs own pre (deg)")
        ax.set_title(t)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs, shrink=0.8, pad=0.01, label="far-contra aim vs own pre (deg; - = ipsilesional)")
    hs = [plt.Line2D([], [], marker=F.ANIMAL_MARKER[a], ls="", mfc="0.6", mec="k", label=a) for a in sorted(F.ANIMAL_MARKER)]
    hs += [plt.Line2D([], [], color="0.75", lw=1, label="pre sessions")]
    fig.legend(handles=hs, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.45, 1.06))
    fig.suptitle(f"Within a session, colored by behavioral recovery ({F.ALIGN_LABEL[S.ALIGN]}; within spout)", y=1.13)
    save_png_svg(fig, FIGS / f"ss_pt_order_aim_{SUF}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = collect()
    D.to_csv(OUT / "pertrial.csv.gz", index=False)
    Sl = slopes(D)
    Sl.to_csv(OUT / "within_session_slopes.csv", index=False)
    fig_plane(D)
    fig_order(D, Sl)
    fig_order_aim(D, Sl)
    Q = fig_size(D)
    Q.to_csv(OUT / "size_percentiles.csv", index=False)
    rng = np.random.default_rng(0)
    with pd.option_context("display.width", 220, "display.precision", 3):
        for c in ("slope_x", "slope_y", "slope_lsize"):
            pre = Sl[Sl.epoch == "pre"].groupby("animal")[c].mean().round(3).to_dict()
            print(c, "pre level per animal:", pre, "| change vs pre:",
                  {e: (round(v[0], 3), v[3], v[4], bool(v[1] > 0 or v[2] < 0)) for e, v in F.change(Sl, c, rng).items()})
        for p in S.POS:
            g = Q[Q.spout == p]
            print(p, {qn: {e: round(v[0], 2) for e, v in F.change(g, qn, rng).items()} for qn in ("q10", "q50", "q90")})


if __name__ == "__main__":
    main()
