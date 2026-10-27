"""LIKE-FOR-LIKE (landing-matched) lick read-out -- the PRIMARY offset measure from 10-27 (Priya: "let's do the
like-for-like lick analysis"; replaces 'offset beyond the pull', whose answer depended on which pull slope was assumed).

Per lick (frozen KEY-row lateral decoder per lick set, z-scored Allen areas; `lick_order_preds.csv.gz`):
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
Usage: python scripts/lick_matched_readout.py [--ref place|spout|pooled]   (default place; see REF)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402

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
TAG = "" if REF == "pooled" else f"_{REF}"
REFLAB = {"place": "pre licks (any spout) that landed at the SAME PLACE",
          "spout": "pre licks of the SAME SPOUT with the same landing deviation",
          "pooled": "pre licks (any spout) with the same landing deviation from their spout"}[REF]


def per_lick() -> pd.DataFrame:
    P = pd.read_csv(il.OUT / "lick_order_preds.csv.gz")
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
    P["resid"] = P[f"resid_{REF}"]
    P["cls"] = np.where(P.order == 0, "first", "later")
    return P


def curves(P: pd.DataFrame) -> pd.DataFrame:
    """Read-out vs landing per epoch: absolute place (REF place) or deviation from the spout (pooled / spout)."""
    x, y, b = ("act", "dec", "bin_a") if REF == "place" else ("dLand", "dDec", "bin")
    A = P.groupby(["set", "epoch", b, "animal"], observed=True).agg(y=(y, "mean"), n=(y, "size"))
    A = A[A.n >= MIN_BIN].reset_index()
    A["mid"] = [v.mid for v in A[b]]
    return A


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    Q = P.dropna(subset=["resid"])
    cells = [("all spouts", "all", Q)] + [(p, "all", Q[Q.pos == p]) for p in POS] + \
            [("all spouts", c, Q[Q.cls == c]) for c in ("first", "later")] + \
            [("far_contra", c, Q[(Q.pos == "far_contra") & (Q.cls == c)]) for c in ("first", "later")]
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
    b = "bin_a" if REF == "place" else "bin"
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
    T = pd.DataFrame(rows)
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
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for j, st in enumerate(SETS):
        ax = axs[0, j]
        for ep in EPS:
            q = A[(A.set == st) & (A.epoch == ep)].groupby("mid").y
            mu, se = q.mean(), q.sem()
            ax.errorbar(mu.index.astype(float), mu, yerr=se, fmt="o-", color=col[ep], ms=4, capsize=2, lw=1.3)
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.axvline(0, color="0.7", lw=0.6, ls=":")
        # significance strip: one row per post epoch, translucent bar over bins whose change vs pre survives BH
        for r_i, ep in enumerate(EP):
            bt = BT[(BT.set == st) & (BT.epoch == ep) & BT.sig] if BT is not None and len(BT) else []
            for m_ in (bt.mid if len(bt) else []):
                ax.axvspan(m_ - 0.125, m_ + 0.125, ymin=0.955 - 0.035 * r_i, ymax=0.985 - 0.035 * r_i, color=col[ep],
                           alpha=0.5, lw=0)
        ax.set_xlabel("landing, lateral (mm, + = contralesional)" if REF == "place" else
                      "landing deviation from the animal's pre mean at the spout (mm, + = contralesional)", fontsize=8)
        ax.set_ylabel(("decoded lateral" if REF == "place" else "read-out deviation (decoded")
                      + " (mm, + = contralesional)", fontsize=8)
        ax.set_title(f"{SETS[st]} licks: read-out vs landing{' place' if REF == 'place' else ' deviation'}\n(bins with >= {MIN_BIN} licks per animal; "
                     "mean +/- SEM over animals)", fontsize=9)
        ax.legend(handles=epl, fontsize=8, frameon=False, loc="lower right")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE: decoded position for a given landing, before and after the stroke (frozen KEY-row "
                 "lateral decoder).\nA curve shifted UP = read more contralesional than " + REFLAB + ".\nBars at the top = bins whose change vs pre survives BH (paired nested bootstrap; rows acute / subacute / chronic). PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(il.OUT / "figures" / f"lick_matched{TAG}_curves.png", dpi=100)
    plt.close(fig)
    # 2. pooled read-out: all spouts, first / later, far contra first / later
    panels = [("all spouts", "all", "all spouts, all licks"), ("all spouts", "first", "all spouts, FIRST lick"),
              ("all spouts", "later", "all spouts, later licks"), ("far_contra", "all", "far contra, all licks"),
              ("far_contra", "first", "far contra, FIRST lick"), ("far_contra", "later", "far contra, later licks")]
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
                ax.set_ylabel("read-out vs " + REFLAB.replace("pre licks ", "pre licks\n", 1)
                              + "\n(mm, + = contralesional)", fontsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE read-out (no pull slope): each lick's read-out minus that of the animal's " + REFLAB +
                 ".\nsmall filled = SESSIONS, open = animal means, shape = animal (circle PS92, square PS93, triangle PS94, diamond PS95), bar = mean of animals; "
                 "text = change vs pre, 95% paired nested bootstrap, * = CI excludes 0 (bold), k/n animals. "
                 "PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT / "figures" / f"lick_matched{TAG}_readout.png", dpi=100)
    plt.close(fig)
    # 3. per spout
    fig, axs = plt.subplots(2, 6, figsize=(19, 7.6), squeeze=False)
    for i, st in enumerate(SETS):
        for j, p in enumerate(POS):
            ax = axs[i, j]
            g = S[(S.set == st) & (S.scope == p) & (S.cls == "all")]
            t = T[(T.set == st) & (T.scope == p) & (T.cls == "all") & (T.measure == "resid")].set_index("epoch")
            for k, ep in enumerate(EPS):
                _dots(ax, g[g.epoch == ep], k, col[ep])
                if ep in t.index:
                    _stars(ax, k, t.loc[ep], col[ep])
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(4), EPS, fontsize=7.5)
            ax.set_title(f"{SETS[st]}: {p.replace('_', ' ')}", fontsize=9, pad=22)
            if j == 0:
                ax.set_ylabel("read-out vs " + REFLAB.replace("pre licks ", "pre licks\n", 1)
                              + "\n(mm, + = contralesional)", fontsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE read-out PER SPOUT (canonical order; cued by target, ITI by the previous trial's spout). "
                 "Reference = " + REFLAB + ".\nsmall filled = SESSIONS, open = animal means, shape = animal (circle PS92, square PS93, triangle PS94, diamond PS95), bar = "
                 "mean; change vs pre, 95% paired nested bootstrap, * = CI excludes 0 (bold), k/n animals. PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT / "figures" / f"lick_matched{TAG}_perspout.png", dpi=100)
    plt.close(fig)


def main() -> int:
    P = per_lick()
    print("share of licks with a reference bin, by set x epoch:")
    print(P.groupby(["set", "epoch"]).resid.apply(lambda s: s.notna().mean()).round(3).unstack().to_string())
    A = curves(P)
    S = sessions(P)
    S.to_csv(il.OUT / f"matched{TAG}_sessions.csv", index=False)
    T = tests(S)
    T.to_csv(il.OUT / f"matched{TAG}_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print("== per animal (rule 8) + nested bootstrap, resid")
    cols = ["scope", "cls", "set", "epoch", "point", "lo", "hi", "n_dir", "n"] + [c for c in T.columns if c.startswith("animal_")]
    print(T[T.measure == "resid"][cols].round(2).to_string(index=False))
    BT = bin_tests(P)
    BT.to_csv(il.OUT / f"matched{TAG}_bin_tests.csv", index=False)
    plot(A, S, T, BT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
