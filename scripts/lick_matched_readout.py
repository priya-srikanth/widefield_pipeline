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


def _stars(ax, x, r, col):
    sig = r.lo > 0 or r.hi < 0
    ax.text(x, r.hi, f"{r.point:+.2f}{'*' if sig else ''}\n{int(r.n_dir)}/{int(r.n)}", ha="center", va="bottom",
            fontsize=6.5, color=col, fontweight="bold" if sig else "normal")


def plot(A: pd.DataFrame, S: pd.DataFrame, T: pd.DataFrame) -> None:
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
        ax.set_xlabel("landing, lateral (mm, + = contralesional)" if REF == "place" else
                      "landing deviation from the animal's pre mean at the spout (mm, + = contralesional)", fontsize=8)
        ax.set_ylabel(("decoded lateral" if REF == "place" else "read-out deviation (decoded")
                      + " (mm, + = contralesional)", fontsize=8)
        ax.set_title(f"{SETS[st]} licks: read-out vs landing{' place' if REF == 'place' else ' deviation'}\n(bins with >= {MIN_BIN} licks per animal; "
                     "mean +/- SEM over animals)", fontsize=9)
        ax.legend(handles=epl, fontsize=8, frameon=False, loc="upper left")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE: decoded position for a given landing, before and after the stroke (frozen KEY-row "
                 "lateral decoder).\nA curve shifted UP = read more contralesional than " + REFLAB + ". PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(il.OUT / "figures" / f"lick_matched{TAG}_curves.png", dpi=100)
    plt.close(fig)
    # 2. pooled read-out: all spouts, first / later, far contra first / later
    panels = [("all spouts", "all", "all spouts, all licks"), ("all spouts", "first", "all spouts, FIRST lick"),
              ("all spouts", "later", "all spouts, later licks"), ("far_contra", "all", "far contra, all licks"),
              ("far_contra", "first", "far contra, FIRST lick"), ("far_contra", "later", "far contra, later licks")]
    fig, axs = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 7.6), squeeze=False, sharey="row")
    for i, st in enumerate(SETS):
        for j, (scope, cls, title) in enumerate(panels):
            ax = axs[i, j]
            g = S[(S.set == st) & (S.scope == scope) & (S.cls == cls)]
            t = T[(T.set == st) & (T.scope == scope) & (T.cls == cls) & (T.measure == "resid")].set_index("epoch")
            for k, ep in enumerate(EPS):
                v = g[g.epoch == ep].groupby("animal").resid.mean()
                ax.plot(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, "o", color=col[ep], ms=4, alpha=0.8)
                ax.plot([k - 0.2, k + 0.2], [v.mean()] * 2, color=col[ep], lw=2.5)
                if ep in t.index:
                    _stars(ax, k, t.loc[ep], col[ep])
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(4), EPS, fontsize=7.5)
            ax.set_title(f"{SETS[st]}: {title}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("read-out vs " + REFLAB.replace("pre licks ", "pre licks\n", 1)
                              + "\n(mm, + = contralesional)", fontsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE read-out (no pull slope): each lick's read-out minus that of the animal's " + REFLAB +
                 ".\nDots = animals (mean of sessions), bar = mean; "
                 "text = change vs pre, 95% paired nested bootstrap, * = CI excludes 0 (bold), k/n animals. "
                 "PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT / "figures" / f"lick_matched{TAG}_readout.png", dpi=100)
    plt.close(fig)
    # 3. per spout
    fig, axs = plt.subplots(2, 6, figsize=(19, 7.6), squeeze=False, sharey="row")
    for i, st in enumerate(SETS):
        for j, p in enumerate(POS):
            ax = axs[i, j]
            g = S[(S.set == st) & (S.scope == p) & (S.cls == "all")]
            t = T[(T.set == st) & (T.scope == p) & (T.cls == "all") & (T.measure == "resid")].set_index("epoch")
            for k, ep in enumerate(EPS):
                v = g[g.epoch == ep].groupby("animal").resid.mean()
                ax.plot(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, "o", color=col[ep], ms=4, alpha=0.8)
                if len(v):
                    ax.plot([k - 0.2, k + 0.2], [v.mean()] * 2, color=col[ep], lw=2.5)
                if ep in t.index:
                    _stars(ax, k, t.loc[ep], col[ep])
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(4), EPS, fontsize=7.5)
            ax.set_title(f"{SETS[st]}: {p.replace('_', ' ')}", fontsize=9)
            if j == 0:
                ax.set_ylabel("read-out vs " + REFLAB.replace("pre licks ", "pre licks\n", 1)
                              + "\n(mm, + = contralesional)", fontsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE read-out PER SPOUT (canonical order; cued by target, ITI by the previous trial's spout). "
                 "Reference = " + REFLAB + ".\nDots = animals, bar = "
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
    plot(A, S, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
