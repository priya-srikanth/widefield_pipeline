"""LIKE-FOR-LIKE (landing-matched) lick read-out -- the PRIMARY offset measure from 10-27 (Priya: "let's do the
like-for-like lick analysis"; replaces 'offset beyond the pull', whose answer depended on which pull slope was assumed).

Per lick (frozen KEY-row lateral decoder per lick set, z-scored Allen areas -- with the ALM / tjM1 split by default since
10-27 late, iti_lick_order.SOURCE; `iti_lick_order.preds_path()`; outputs tagged iti_lick_order.SRC_TAG = "" for the
canonical split basis, "_plain" for plain Allen areas):
  dLand = landing - the animal's PRE mean landing at the spout (cued: target spout; ITI: previous trial's spout)
  dDec  = decoded - the animal's PRE mean decoded position at the same spout
  REF   = the animal's PRE curve: mean dDec of its pre licks (all spouts, same lick set) in landing-deviation bins
          (BINS, mm); pre sessions are scored leave-one-session-out by the decoder, so the pre curve is out-of-sample
  resid = dDec - REF(bin of dLand): how the lick is read compared with pre-stroke licks that landed the SAME distance
          off their spout (+ = read more contralesional). No pull slope enters.
Session means of resid (all spouts / per spout / first vs later lick), change vs pre = paired nested bootstrap
(analysis_kit.boot_delta), per-animal table first (rule 8). Pre sessions sit near 0 by construction (their own curve);
their spread is the noise level.
Outputs <E>/iti_licks/matched_{sessions,tests}.csv, figures/lick_matched_{curves,readout,perspout}.png.
--ref loclin (10-27 late, Priya "sounds good"): LOCAL-LINEAR Gaussian reference with the edge line carried beyond the pre
range (scripts/likeforlike.py): every lick gets a reference, `beyond` flags licks outside the pre range; extra scope
"all spouts, inside pre range" (sensitivity), matched_loclin*_coverage.csv (share beyond per epoch) and
matched_loclin*_refboot.csv (epoch effect over references refit on resampled pre sessions).
10-28 (overnight queue 5d, approved backlog): per-spout figure gains an ITI FIRST-lick row (bout start); new figure
figures/lick_matched<TAG>_animal_change.png (per-animal change vs pre with session spread, `plot_animal`); figure data
figdata/lick_matched<TAG>.pkl (--plot-only redraws every figure from it). Writes go to iti_licks.OUT_W (= OUT unless
WFIELD_ITI_OUT_W redirects them); --animals PS94[,PS95] restricts a test run.
Usage: python scripts/lick_matched_readout.py [--ref place|spout|pooled|gauss|loclin] [--plot-only] [--animals A,B]
       (default place; see REF)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import likeforlike as lfl  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP = ["acute", "subacute", "chronic"]
EPS = ["pre", *EP]
POS = lo.POS
SETS = {"iti": "ITI", "trial": "cued"}
BINS = [-4, -2, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 4]
MIN_BIN = 20          # pre licks needed in a bin for its reference value
MIN = 8               # licks per session cell
FINE = np.arange(-4, 4.01, 0.25)    # 0.25-mm bins for the same-spout and same-place references
MIN_FINE = 15
# reference (Priya 10-27): "place" = pre licks of any spout that landed at the same absolute lateral place (covers the
# acute large misses; asks whether a lick is read like any lick that landed there) -- PRIMARY; "spout" = pre licks of
# the same spout with the same deviation (abnormal vs pre; no reference for acute large misses); "pooled" = pre licks of
# any spout with the same deviation from their own spout (first version)
REF = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "place"
TAG = ("" if REF == "pooled" else f"_{REF}") + lo.SRC_TAG      # + decoder source (iti_lick_order.SOURCE)
ANIMALS = sys.argv[sys.argv.index("--animals") + 1].split(",") if "--animals" in sys.argv else None
# --figs curves,readout,perspout,animal: draw only these (default all) -- a rerun that adds the per-spout row / the
# per-animal figure need not re-render the other page figures
FIGS = set(sys.argv[sys.argv.index("--figs") + 1].split(",")) if "--figs" in sys.argv else \
    {"curves", "readout", "perspout", "animal"}
SIGMAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)   # Gaussian widths tried (mm); chosen by held-out pre-session prediction
PLACE_REFS = ("place", "gauss", "loclin")      # references on the absolute landing place (not deviation from the spout)
INSIDE = "all spouts, inside pre range"
REFLAB = {"loclin": "pre licks (any spout) by a Gaussian-weighted LOCAL LINE on landing (SAME PLACE; edge line carried "
                    "beyond the pre range)",
          "gauss": "pre licks (any spout) weighted by a Gaussian on landing distance (SAME PLACE, no bins)",
          "place": "pre licks (any spout) that landed at the SAME PLACE",
          "spout": "pre licks of the SAME SPOUT with the same landing deviation",
          "pooled": "pre licks (any spout) with the same landing deviation from their spout"}[REF]


def _kernel_mean(x_ref: np.ndarray, y_ref: np.ndarray, x: np.ndarray, sigma: float, chunk: int = 4000) -> np.ndarray:
    """Nadaraya-Watson: Gaussian-weighted mean of y_ref at each x (weights exp(-(x - x_ref)^2 / 2 sigma^2)); NaN where
    the total weight is < 3 (fewer than ~3 reference licks within ~1 sigma)."""
    out = np.full(len(x), np.nan)
    for i in range(0, len(x), chunk):
        w = np.exp(-0.5 * ((x[i:i + chunk, None] - x_ref[None, :]) / sigma) ** 2)
        sw = w.sum(1)
        out[i:i + chunk] = np.where(sw >= 3, (w @ y_ref) / np.maximum(sw, 1e-12), np.nan)
    return out


def gauss_ref(P: pd.DataFrame, sigma: float) -> np.ndarray:
    """Kernel reference per lick: the animal's PRE licks of the same set (any spout) weighted by landing distance.
    Pre-session licks use the OTHER pre sessions only (leave-one-session-out), so pre sits at ~0 out of sample."""
    ref = np.full(len(P), np.nan)
    for (_st, _a), g in P.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        post = g[g.epoch != "pre"]
        ref[P.index.get_indexer(post.index)] = _kernel_mean(pre.act.to_numpy(), pre.dec.to_numpy(), post.act.to_numpy(),
                                                            sigma)
        for lab, h in pre.groupby("label"):
            o = pre[pre.label != lab]
            ref[P.index.get_indexer(h.index)] = _kernel_mean(o.act.to_numpy(), o.dec.to_numpy(), h.act.to_numpy(), sigma)
    return ref


def choose_sigma(P: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Width = the SIGMAS value with the smallest held-out PRE prediction error (leave-one-pre-session-out), pooled over
    sets and animals -- chosen on pre-stroke data only. Returns (sigma, table of mean squared errors)."""
    Q = P[P.epoch == "pre"]
    rows = []
    for sg in SIGMAS:
        r = gauss_ref(Q.copy(), sg)
        e = (Q.dec.to_numpy() - r) ** 2
        rows.append(dict(sigma=sg, mse=float(np.nanmean(e)), coverage=float(np.isfinite(r).mean())))
    T = pd.DataFrame(rows)
    return float(T.loc[T.mse.idxmin(), "sigma"]), T


def per_lick() -> pd.DataFrame:
    P = pd.read_csv(lo.preds_path())
    if il.ITI_SUBSET:                                  # docked-spout etc. subset of the frozen predictions (10-29)
        keep = il.iti_subset_keep(P)
        print(f"ITI subset filter {il.ITI_SUBSET}: keeping {int(keep[(P.set == 'iti').to_numpy()].sum())} of "
              f"{int((P.set == 'iti').sum())} ITI licks", flush=True)
        P = P[keep].reset_index(drop=True)
    if ANIMALS:                                        # test runs on a subset (the pre references are per animal)
        P = P[P.animal.isin(ANIMALS)].reset_index(drop=True)
    ref = P[P.epoch == "pre"].groupby(["set", "animal", "pos"])[["act", "dec"]].mean()
    P = P.join(ref, on=["set", "animal", "pos"], rsuffix="_ref")
    P["dLand"], P["dDec"] = P.act - P.act_ref, P.dec - P.dec_ref
    P["bin"] = pd.cut(P.dLand, BINS)
    pre = P[P.epoch == "pre"].groupby(["set", "animal", "bin"], observed=True).dDec.agg(["mean", "size"])
    pre = pre[pre["size"] >= MIN_BIN]["mean"]
    P["resid_pooled"] = P.dDec - pre.reindex(pd.MultiIndex.from_arrays([P.set, P.animal, P.bin])).to_numpy()
    # SAME SPOUT: pre licks of the same spout with the same deviation (fine bins; acute large misses have none)
    P["bin_f"] = pd.cut(P.dLand, FINE)
    ps = P[P.epoch == "pre"].groupby(["set", "animal", "pos", "bin_f"], observed=True).dDec.agg(["mean", "size"])
    ps = ps[ps["size"] >= MIN_FINE]["mean"]
    P["resid_spout"] = P.dDec - ps.reindex(pd.MultiIndex.from_arrays([P.set, P.animal, P.pos, P.bin_f])).to_numpy()
    # SAME PLACE: pre licks of ANY spout that landed at the same absolute lateral place; absolute decoded positions
    P["bin_a"] = pd.cut(P.act, FINE)
    pa = P[P.epoch == "pre"].groupby(["set", "animal", "bin_a"], observed=True).dec.agg(["mean", "size"])
    pa = pa[pa["size"] >= MIN_FINE]["mean"]
    P["resid_place"] = P.dec - pa.reindex(pd.MultiIndex.from_arrays([P.set, P.animal, P.bin_a])).to_numpy()
    if REF == "gauss":
        sg, cv = choose_sigma(P)
        print("Gaussian width by held-out pre-session prediction (mse, coverage):")
        print(cv.round(4).to_string(index=False))
        print(f"-> sigma = {sg} mm")
        cv.to_csv(il.OUT_W / f"matched_gauss{lo.SRC_TAG}_sigma_cv.csv", index=False)
        P["resid_gauss"] = P.dec - gauss_ref(P, sg)
        P.attrs["sigma"] = sg
    if REF == "loclin":
        sg, cv = lfl.choose(P, "act", "dec", SIGMAS)
        print("local-linear vs weighted-mean reference, held-out pre error:")
        print(cv.round(4).to_string(index=False))
        print(f"-> local-linear sigma = {sg} mm")
        cv.to_csv(il.OUT_W / f"matched_loclin{lo.SRC_TAG}_sigma_cv.csv", index=False)
        R = lfl.ref_table(P, "act", "dec", sg)
        P["resid_loclin"], P["beyond"] = P.dec - R.ref, R.beyond
        P.attrs["sigma"] = sg
    P["resid"] = P[f"resid_{REF}"]
    P["cls"] = np.where(P.order == 0, "first", "later")
    return P


CURVE_GRID = np.round(np.arange(-3.0, 3.001, 0.05), 3)   # landing grid of the smoothed curves (mm)
CURVE_MIN_W = 10.0     # Gaussian weight (~licks within ~1 sigma) an animal x epoch needs at a grid point to be drawn


def curves(P: pd.DataFrame) -> pd.DataFrame:
    """Read-out vs landing per epoch: absolute place (REF place) or deviation from the spout (pooled / spout).
    REF gauss (Priya 10-28 "there aren't discrete bins anymore right?"): Gaussian-SMOOTHED curves with the offset's own
    sigma on CURVE_GRID (column smooth=True), no display bins; the significance strip keeps 0.25-mm bins."""
    if REF == "gauss" and "sigma" in P.attrs:
        sg, rows = P.attrs["sigma"], []
        for (st, ep, a), g in P.groupby(["set", "epoch", "animal"]):
            x_, y_ = g.act.to_numpy(float), g.dec.to_numpy(float)
            ok = np.isfinite(x_) & np.isfinite(y_)
            w = np.exp(-0.5 * ((CURVE_GRID[:, None] - x_[None, ok]) / sg) ** 2)
            sw = w.sum(1)
            yy = np.where(sw >= CURVE_MIN_W, (w @ y_[ok]) / np.maximum(sw, 1e-12), np.nan)
            rows += [dict(set=st, epoch=ep, animal=a, mid=m, y=v, n=s, smooth=True, sigma=sg)
                     for m, v, s in zip(CURVE_GRID, yy, sw) if np.isfinite(v)]
        return pd.DataFrame(rows)
    x, y, b = ("act", "dec", "bin_a") if REF in PLACE_REFS else ("dLand", "dDec", "bin")
    A = P.groupby(["set", "epoch", b, "animal"], observed=True).agg(y=(y, "mean"), n=(y, "size"))
    A = A[A.n >= MIN_BIN].reset_index()
    A["mid"] = [v.mid for v in A[b]]
    return A


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    Q = P.dropna(subset=["resid"])
    cells = [("all spouts", "all", Q)] + [(p, "all", Q[Q.pos == p]) for p in POS] + \
            [("all spouts", c, Q[Q.cls == c]) for c in ("first", "later")] + \
            [(p, c, Q[(Q.pos == p) & (Q.cls == c)]) for p in POS for c in ("first", "later")]
    if REF == "loclin":                     # sensitivity: post licks inside the pre landing range only
        Qi = Q[(Q.beyond == 0) | (Q.epoch == "pre")]
        cells += [(INSIDE, "all", Qi)] + [(INSIDE, c, Qi[Qi.cls == c]) for c in ("first", "later")]
    for scope, cls, g in cells:
        for (st, a, lab, ep), h in g.groupby(["set", "animal", "label", "epoch"]):
            if len(h) >= MIN:
                rows.append(dict(scope=scope, cls=cls, set=st, animal=a, label=lab, epoch=ep, n=len(h),
                                 resid=h.resid.mean(), dLand=h.dLand.mean(), dDec=h.dDec.mean(),
                                 matched_share=float(len(h)) / max(1, ((P.label == lab) & (P.set == st)).sum())))
    return pd.DataFrame(rows)


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (scope, cls, st), g in S.groupby(["scope", "cls", "set"]):
        for val in ("resid", "dLand", "dDec"):
            prd = {a: h[val].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            pm = g[g.epoch == "pre"].groupby("animal")[val].mean()
            for ep in EP:
                post = {a: h[val].to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                if not post:
                    continue
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = (g[g.epoch == ep].groupby("animal")[val].mean() - pm).dropna()
                out.append(dict(scope=scope, cls=cls, set=st, measure=val, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am),
                                **{f"animal_{a}": v for a, v in am.items()}))
    return pd.DataFrame(out)


def bin_tests(P: pd.DataFrame) -> pd.DataFrame:
    """Per landing bin of the curve figure: session mean of resid in that bin (>= 5 licks), change vs pre by the paired
    nested bootstrap, p from the same draws (analysis_kit.boot_p), BH across bins within set x epoch."""
    b = "bin_a" if REF in PLACE_REFS else "bin"
    Q = P.dropna(subset=["resid"])
    S = Q.groupby(["set", b, "animal", "label", "epoch"], observed=True).resid.agg(["mean", "size"]).reset_index()
    S = S[S["size"] >= 5]
    rows = []
    for (st, bb), g in S.groupby(["set", b], observed=True):
        prd = {a: h["mean"].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
        for ep in EP:
            post = {a: h["mean"].to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
            if len(post) < 3:
                continue
            ci, draws = ak.boot_delta(post, prd, np.random.default_rng(0), draws=True)
            rows.append(dict(set=st, epoch=ep, mid=bb.mid, point=ci.point, p=ak.boot_p(draws), n=len(post)))
    T = pd.DataFrame(rows, columns=["set", "epoch", "mid", "point", "p", "n"])    # empty with < 3 animals (test runs)
    T["sig"] = False
    for _k, idx in T.groupby(["set", "epoch"]).groups.items():
        T.loc[idx, "sig"] = ak.bh(T.loc[idx, "p"].to_numpy())
    return T


def _dots(ax, g: pd.DataFrame, k: int, c) -> None:
    """One small dot per SESSION (jittered), one outlined marker per ANIMAL (mean of its sessions), bar = mean of the
    animals (the animal-weighted point the paired nested bootstrap tests)."""
    if g.empty:
        return
    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    rng = np.random.default_rng(k)
    am = g.groupby("animal").resid.mean()
    for q, (a, h) in enumerate(g.groupby("animal")):
        mk = ANIMAL_MARKER.get(a, "o")
        ax.plot(k + rng.uniform(-0.22, 0.22, len(h)), h.resid, mk, color=c, ms=3, alpha=0.4, mew=0)
        ax.plot(k - 0.12 + 0.08 * q, am[a], mk, mfc="none", mec=c, mew=1.3, ms=7)
    ax.plot([k - 0.25, k + 0.25], [am.mean()] * 2, color=c, lw=2.5)


def _stars(ax, x, r, col):
    """Change-vs-pre label in a strip ABOVE the data (axes coordinates), so it never sits on the markers."""
    sig = r.lo > 0 or r.hi < 0
    ax.text(x, 1.01, f"{r.point:+.2f}{'*' if sig else ''}\n{int(r.n_dir)}/{int(r.n)}", ha="center", va="bottom",
            fontsize=6.5, color=col, fontweight="bold" if sig else "normal", transform=ax.get_xaxis_transform(),
            clip_on=False)


def plot(A: pd.DataFrame, S: pd.DataFrame, T: pd.DataFrame, BT: pd.DataFrame | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    epl = [Line2D([], [], color=col[e], lw=2, marker="o", label=e) for e in EPS]
    # 1. curves
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), squeeze=False)
    for j, st in enumerate(SETS):
        ax = axs[0, j]
        smooth = "smooth" in A.columns
        for ep in EPS:
            q = A[(A.set == st) & (A.epoch == ep)].groupby("mid").y
            mu, se, nn = q.mean(), q.sem(), q.count()
            if smooth:   # Gaussian-smoothed: solid + SEM band where all 4 animals contribute; thin dashed where 3
                # (an animal entering / leaving the mean makes a step, so the 3-animal stretches are marked as such)
                m4 = mu.where(nn >= 4)
                ax.plot(mu.index.to_numpy(float), mu.where(nn >= 3), color=col[ep], lw=0.9, ls="--", zorder=3)
                ax.plot(m4.index.to_numpy(float), m4, color=col[ep], lw=2.0, zorder=4)
                ax.fill_between(m4.index.to_numpy(float), m4 - se, m4 + se, color=col[ep], alpha=0.08, lw=0, zorder=2)
            else:
                ax.errorbar(mu.index.astype(float), mu, yerr=se, fmt="o-", color=col[ep], ms=4, capsize=2, lw=1.3,
                            zorder=3)
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.axvline(0, color="0.7", lw=0.6, ls=":")
        # significance strip: one row per post epoch, translucent bar over bins whose change vs pre survives BH
        for r_i, ep in enumerate(EP):
            bt = BT[(BT.set == st) & (BT.epoch == ep) & BT.sig] if BT is not None and len(BT) else []
            for m_ in (bt.mid if len(bt) else []):
                ax.axvspan(m_ - 0.125, m_ + 0.125, ymin=0.975 - 0.018 * r_i, ymax=0.987 - 0.018 * r_i, color=col[ep],
                           alpha=0.35, lw=0, zorder=0)                  # thin, BEHIND the data and SEM bars
        ax.set_xlabel("landing, lateral (mm, + = contralesional)" if REF in PLACE_REFS else
                      "landing deviation from the animal's pre mean at the spout (mm, + = contralesional)", fontsize=8)
        ax.set_ylabel(("decoded lateral" if REF in PLACE_REFS else "read-out deviation (decoded")
                      + " (mm, + = contralesional)", fontsize=8)
        ax.set_title(f"{SETS[st]} licks: read-out vs landing{' place' if REF in PLACE_REFS else ' deviation'}\n" +
                     (f"(Gaussian-smoothed, sigma {A['sigma'].iloc[0]:g} mm; mean +/- SEM of 4 animals;\n"
                      "dashed = only 3 animals there)" if smooth else
                      f"(bins with >= {MIN_BIN} licks per animal; mean +/- SEM over animals)"), fontsize=9)
        ax.legend(handles=epl, fontsize=8, frameon=False, loc="lower right")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE: decoded position for a given landing, before and after the stroke (frozen "
                 + ("NON-LINEAR (RBF kernel ridge)" if lo.SOURCE == "nonlinear" else "KEY-row") + " "
                 "lateral decoder).\nA curve shifted UP = read more contralesional than " + REFLAB + ".\nBars at the top = bins whose change vs pre survives BH (paired nested bootstrap; rows acute / subacute / chronic). PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    if "curves" in FIGS:
        save_png_svg(fig, il.OUT_W / "figures" / f"lick_matched{TAG}_curves.png", dpi=100)
    plt.close(fig)
    # 2. pooled read-out: all spouts, first / later, far contra first / later
    panels = [("all spouts", "all", "all spouts, all licks"), ("all spouts", "first", "all spouts, FIRST lick"),
              ("all spouts", "later", "all spouts, later licks"), ("far_contra", "all", "far contra, all licks"),
              ("far_contra", "first", "far contra, FIRST lick"), ("far_contra", "later", "far contra, later licks")]
    if REF == "loclin":
        panels.insert(1, (INSIDE, "all", "all spouts, INSIDE the pre range\n(no extrapolated reference)"))
    fig, axs = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 7.6), squeeze=False)
    for i, st in enumerate(SETS):
        for j, (scope, cls, title) in enumerate(panels):
            ax = axs[i, j]
            g = S[(S.set == st) & (S.scope == scope) & (S.cls == cls)]
            t = T[(T.set == st) & (T.scope == scope) & (T.cls == cls) & (T.measure == "resid")].set_index("epoch")
            for k, ep in enumerate(EPS):
                _dots(ax, g[g.epoch == ep], k, col[ep])
                if ep in t.index:
                    _stars(ax, k, t.loc[ep], col[ep])
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(4), EPS, fontsize=7.5)
            ax.set_title(f"{SETS[st]}: {title}", fontsize=8.5, pad=22)
            if j == 0:
                # short label (the reference is spelled out in the suptitle): the long one ran across rows (Priya 10-28)
                ax.set_ylabel(("read-out vs pre licks\nat the same place" if REF in PLACE_REFS else
                               "read-out vs pre licks") + "\n(mm, + = contralesional)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE read-out (no pull slope): each lick's read-out minus that of the animal's " + REFLAB +
                 ".\nsmall filled = SESSIONS, open = animal means, shape = animal (circle PS92, square PS93, triangle PS94, diamond PS95), bar = mean of animals; "
                 "text = change vs pre, 95% paired nested bootstrap, * = CI excludes 0 (bold), k/n animals. "
                 "PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if "readout" in FIGS:
        save_png_svg(fig, il.OUT_W / "figures" / f"lick_matched{TAG}_readout.png", dpi=100)
    plt.close(fig)
    # 3. per spout
    # Priya 10-27 (cued first-lick row); ITI FIRST-lick row added 10-28 (overnight queue 5d): ITI order 0 = the start of
    # a lick bout (no lick for >= 1 s, iti_lick_order._bout_idx), the self-initiated launch -- as meaningful as the cued
    # first lick, and >= MIN per spout per session in every animal
    rows = [("iti", "all", "ITI"), ("iti", "first", "ITI, FIRST lick (bout start)"), ("trial", "all", "cued"),
            ("trial", "first", "cued, FIRST lick")]
    fig, axs = plt.subplots(len(rows), 6, figsize=(19, 3.8 * len(rows)), squeeze=False)
    for i, (st, cls_, rlab) in enumerate(rows):
        for j, p in enumerate(POS):
            ax = axs[i, j]
            g = S[(S.set == st) & (S.scope == p) & (S.cls == cls_)]
            t = T[(T.set == st) & (T.scope == p) & (T.cls == cls_) & (T.measure == "resid")].set_index("epoch")
            for k, ep in enumerate(EPS):
                _dots(ax, g[g.epoch == ep], k, col[ep])
                if ep in t.index:
                    _stars(ax, k, t.loc[ep], col[ep])
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(4), EPS, fontsize=7.5)
            ax.set_title(f"{rlab}: {p.replace('_', ' ')}", fontsize=9, pad=22)
            if j == 0:                                   # short label (4 rows since 10-28; reference named in the title)
                ax.set_ylabel(("read-out vs pre licks\nat the same place" if REF in PLACE_REFS else
                               "read-out vs pre licks\nwith the same deviation") + "\n(mm, + = contralesional)",
                              fontsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE read-out PER SPOUT (canonical order; cued by target, ITI by the previous trial's spout). "
                 "Reference = " + REFLAB + ".\nsmall filled = SESSIONS, open = animal means, shape = animal (circle PS92, square PS93, triangle PS94, diamond PS95), bar = "
                 "mean; change vs pre, 95% paired nested bootstrap, * = CI excludes 0 (bold), k/n animals. PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if "perspout" in FIGS:
        save_png_svg(fig, il.OUT_W / "figures" / f"lick_matched{TAG}_perspout.png", dpi=100)
    plt.close(fig)


ANIMAL_ROWS = [("iti", "all", "ITI, all licks"), ("iti", "first", "ITI, FIRST lick (bout start)"),
               ("trial", "all", "cued, all licks"), ("trial", "first", "cued, FIRST lick")]


def plot_animal(S: pd.DataFrame, T: pd.DataFrame) -> None:
    """PER-ANIMAL change vs pre (overnight queue 5d, approved backlog): for each animal, every post session's offset
    minus the animal's pre mean (small dots), the animal's change (open marker = mean of its session values - pre mean,
    the quantity the paired bootstrap averages over animals) and its PRE spread (grey bar = +/- 1 SD of its pre session
    values around 0: the noise level an epoch value has to clear). Pre sessions sit near 0 by construction (leave-one-
    session-out), so a pre-vs-epoch scatter would be a vertical line; the change with session spread is the clearer view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    animals = sorted(S.animal.unique())
    fig, axs = plt.subplots(len(ANIMAL_ROWS), 3, figsize=(15, 3.4 * len(ANIMAL_ROWS)), squeeze=False, sharey="row")
    rng = np.random.default_rng(0)
    for i, (st, cls, rlab) in enumerate(ANIMAL_ROWS):
        g = S[(S.set == st) & (S.scope == "all spouts") & (S.cls == cls)]
        pre = g[g.epoch == "pre"].groupby("animal").resid.agg(["mean", "std"])
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            c = EPOCH_LINE[ep]
            for k, a in enumerate(animals):
                if a not in pre.index:
                    continue
                sd = pre.loc[a, "std"] if np.isfinite(pre.loc[a, "std"]) else 0.0
                ax.add_patch(plt.Rectangle((k - 0.3, -sd), 0.6, 2 * sd, color="0.85", lw=0, zorder=0))
                h = g[(g.epoch == ep) & (g.animal == a)].resid.to_numpy() - pre.loc[a, "mean"]
                if not len(h):
                    continue
                mk = ANIMAL_MARKER.get(a, "o")
                ax.plot(k + rng.uniform(-0.18, 0.18, len(h)), h, mk, color=c, ms=3.5, alpha=0.5, mew=0)
                ax.plot(k, h.mean(), mk, mfc="none", mec=c, mew=1.6, ms=10)
                ax.text(k, 1.01, f"{h.mean():+.2f}\n(n={len(h)})", ha="center", va="bottom", fontsize=6.8, color=c,
                        transform=ax.get_xaxis_transform(), clip_on=False)
            t = T[(T.set == st) & (T.scope == "all spouts") & (T.cls == cls) & (T.measure == "resid") & (T.epoch == ep)]
            if len(t):
                r = t.iloc[0]
                sig = r.lo > 0 or r.hi < 0
                kp = len(animals) + 0.3                        # POOLED column (Priya 10-28): mean of animals + 95% CI
                ax.axvline(len(animals) - 0.35, color="0.85", lw=0.8)
                ax.errorbar(kp, r.point, yerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o", color="k", ms=7,
                            mfc=c if sig else "white", mec="k", capsize=3, lw=1.6, zorder=4)
                ax.text(kp, 1.01, f"{r.point:+.2f}{'*' if sig else ''}\n{int(r.n_dir)}/{int(r.n)}", ha="center",
                        va="bottom", fontsize=6.8, color="k", fontweight="bold" if sig else "normal",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                ax.text(0.99, 0.02, f"mean of animals {r.point:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]{'*' if sig else ''}, "
                        f"{int(r.n_dir)}/{int(r.n)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                        fontweight="bold" if sig else "normal")
            ax.axhline(0, color="0.5", lw=0.7, ls="--")
            ax.set_xticks([*range(len(animals)), len(animals) + 0.3], [*animals, "POOLED\n(mean, 95% CI)"], fontsize=8)
            ax.set_xlim(-0.6, len(animals) + 0.8)
            ax.set_title(f"{rlab}: {ep} - pre", fontsize=9, pad=24, color=c)
            if j == 0:
                ax.set_ylabel("session offset - animal's pre mean\n(mm, + = contralesional)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE offset PER ANIMAL, change vs pre (reference = " + REFLAB + ").\nsmall dots = post "
                 "SESSIONS minus the animal's pre mean, open marker = the animal's change (text: value, n sessions), grey "
                 "bar = +/- 1 SD of the animal's PRE sessions (noise level);\nright column / bottom right = POOLED: mean of animals, 95% paired "
                 "nested bootstrap CI, * = excludes 0, k/n animals. PROVISIONAL", fontsize=9.5)
    lo._layout(fig, bottom=0.01)
    save_png_svg(fig, il.OUT_W / "figures" / f"lick_matched{TAG}_animal_change.png", dpi=100)
    plt.close(fig)


def main() -> int:
    from figdata import load_fd, plot_only, save_fd
    if plot_only():                                    # figdata convention (10-28): redraw from the saved figure data
        D = load_fd(f"lick_matched{TAG}")
        plot(D["A"], D["S"], D["T"], D["BT"])
        if "animal" in FIGS:
            plot_animal(D["S"], D["T"])
        return 0
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    P = per_lick()
    print("share of licks with a reference bin, by set x epoch:")
    print(P.groupby(["set", "epoch"]).resid.apply(lambda s: s.notna().mean()).round(3).unstack().to_string())
    A = curves(P)
    S = sessions(P)
    S.to_csv(il.OUT_W / f"matched{TAG}_sessions.csv", index=False)
    T = tests(S)
    T.to_csv(il.OUT_W / f"matched{TAG}_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print("== per animal (rule 8) + nested bootstrap, resid")
    cols = ["scope", "cls", "set", "epoch", "point", "lo", "hi", "n_dir", "n"] + [c for c in T.columns if c.startswith("animal_")]
    print(T[T.measure == "resid"][cols].round(2).to_string(index=False))
    BT = bin_tests(P)
    BT.to_csv(il.OUT_W / f"matched{TAG}_bin_tests.csv", index=False)
    if REF == "loclin":
        C = lfl.coverage(P, P.beyond.to_numpy())
        C.to_csv(il.OUT_W / f"matched{TAG}_coverage.csv", index=False)
        print("== share of licks beyond the pre landing range (median mm beyond):")
        print(C.round(3).to_string(index=False))
        RB = lfl.boot_reference(P, "act", "dec", P.attrs["sigma"])
        RB.to_csv(il.OUT_W / f"matched{TAG}_refboot.csv", index=False)
        print("== epoch effect with the reference refit on resampled pre sessions (2.5-97.5 %):")
        print(RB.round(3).to_string(index=False))
    save_fd(f"lick_matched{TAG}", dict(A=A, S=S, T=T, BT=BT))
    plot(A, S, T, BT)
    if "animal" in FIGS:
        plot_animal(S, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
