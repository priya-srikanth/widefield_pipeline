"""PER-SPOUT DECOMPOSITION of the decoded lateral shift (Priya 10-27: "if our effect was just a 'pull' the direction would
be different for ipsi vs contra spouts" -> "try the per-spout decomposition"). No pull slope is assumed.

Per session x lick set (ITI by previous spout / cued by target), per spout s (>= MIN licks):
  dDec_s  = session mean decoded position at s - the animal's pre mean decoded position at s    (NEURAL)
  dLand_s = session mean landing at s - the animal's pre mean landing at s                        (BEHAVIOUR)
  x_s     = the animal's pre mean landing at s - the animal's pre mean over all licks             (spout's place, mm)
OLS over the spouts:  dDec_s = SHIFT + GAIN * x_s + TRACK * dLand_s
  SHIFT  uniform read-out shift, the same at every spout (+ = contralesional) -- the "contralesional read-out" proper
  GAIN   change in the read-out's lateral spread (- = compression toward the centre: ipsi spouts read more contra,
         contra spouts more ipsi; the map-level face of a lower lateral r)
  TRACK  how much of each spout's landing change the read-out follows (the within-session pull slope, estimated, not
         assumed)
Variants: without TRACK (shift + gain only) and with TRACK fixed at the animal's pre within-spout slope, to show what the
free TRACK term trades against. Pre sessions (decoded leave-one-session-out) give the noise level of each term; change vs
pre = paired nested bootstrap (analysis_kit.boot_delta), per-animal table first (rule 8).
Input: lick_order_preds.csv.gz (KEY-row lateral decoder per set, z-scored Allen areas). Outputs <E>/iti_licks/
decomposition_{sessions,tests}.csv, figures/lick_decoded_decomposition.png.
Usage: python scripts/iti_decoded_decomposition.py
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
MIN = 8
TERMS = ("shift", "gain", "track")
SETS = {"iti": "ITI", "trial": "cued"}


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (st, a), g in P.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        ref = pre.groupby("pos")[["act", "dec"]].mean()
        x = ref.act - pre.act.mean()
        xw = pre.act - pre.groupby(["label", "pos"]).act.transform("mean")
        yw = pre.dec - pre.groupby(["label", "pos"]).dec.transform("mean")
        b_w = float((xw * yw).sum() / (xw ** 2).sum())
        for (lab, ep), h in g.groupby(["label", "epoch"]):
            m = h.groupby("pos").agg(n=("act", "size"), act=("act", "mean"), dec=("dec", "mean"))
            m = m[(m.n >= MIN) & m.index.isin(ref.index)]
            if len(m) < 4:
                continue
            dd = m.dec - ref.dec.reindex(m.index)
            dl = m.act - ref.act.reindex(m.index)
            xs = x.reindex(m.index)
            r = dict(set=st, animal=a, label=lab, epoch=ep, n_spouts=len(m), b_within=b_w,
                     mean_dDec=float(dd.mean()), mean_dLand=float(dl.mean()))
            X = np.column_stack([np.ones(len(m)), xs, dl])
            c = np.linalg.lstsq(X, dd.to_numpy(), rcond=None)[0]
            r.update(shift=c[0], gain=c[1], track=c[2])
            c2 = np.linalg.lstsq(X[:, :2], dd.to_numpy(), rcond=None)[0]
            r.update(shift_notrack=c2[0], gain_notrack=c2[1])
            c3 = np.linalg.lstsq(X[:, :2], (dd - b_w * dl).to_numpy(), rcond=None)[0]
            r.update(shift_fixw=c3[0], gain_fixw=c3[1])
            r["corr_x_dLand"] = float(np.corrcoef(xs, dl)[0, 1]) if dl.std() > 0 else np.nan
            for s_, v in dd.items():
                r[f"dDec_{s_}"] = v
            rows.append(r)
    return pd.DataFrame(rows)


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    cols = ["shift", "gain", "track", "shift_notrack", "gain_notrack", "shift_fixw", "gain_fixw", "mean_dDec", "mean_dLand"]
    for st, g in S.groupby("set"):
        for col in cols:
            prd = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            pm = g[g.epoch == "pre"].groupby("animal")[col].mean()
            out.append(dict(set=st, measure=col, epoch="pre", level=g[g.epoch == "pre"][col].mean(), point=np.nan,
                            lo=np.nan, hi=np.nan, n_dir=np.nan, n=len(pm)))
            for ep in EP:
                post = {a: h[col].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                post = {a: v for a, v in post.items() if len(v)}
                if not post:
                    continue
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = (g[g.epoch == ep].groupby("animal")[col].mean() - pm).dropna()
                out.append(dict(set=st, measure=col, epoch=ep, level=g[g.epoch == ep][col].mean(), point=ci.point,
                                lo=ci.lo, hi=ci.hi, n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am)))
    return pd.DataFrame(out)


def plot(S: pd.DataFrame, T: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    POS = lo.POS
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.4), squeeze=False)
    lab = {"shift": "SHIFT: uniform read-out shift (mm)", "gain": "GAIN: change in read-out spread (mm / mm)",
           "track": "TRACK: share of landing change followed"}
    for i, st in enumerate(SETS):
        g = S[S.set == st]
        ax = axs[i, 0]
        A = g.groupby(["epoch", "animal"])[[f"dDec_{p}" for p in POS]].mean()
        for ep in EPS:
            if ep not in A.index.get_level_values(0):
                continue
            q = A.loc[ep]
            ax.errorbar(np.arange(6) + (EPS.index(ep) - 1.5) * 0.08, q.mean(), yerr=q.sem(), fmt="o-", color=col[ep],
                        ms=4, lw=1.3, capsize=2)
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.set_xticks(range(6), [p.replace("_", "\n") for p in POS], fontsize=7.5)
        ax.set_ylabel("decoded change vs pre (mm, + = contra)", fontsize=8)
        ax.set_title(f"{SETS[st]}: decoded position change per spout\n(mean +/- SEM over animals; raw, no pull model)",
                     fontsize=8.5)
        for j, term in enumerate(TERMS):
            ax = axs[i, 1 + j]
            t = T[(T.set == st) & (T.measure == term)].set_index("epoch")
            for _a, h in g.groupby("animal"):
                am = h.groupby("epoch")[term].mean().reindex(EPS)
                ax.plot(range(4), am, "-", color="0.75", lw=0.8)
            lv = g.groupby("epoch")[term].mean().reindex(EPS)
            ax.plot(range(4), lv, "o", color="k", ms=0)
            for k, ep in enumerate(EPS):
                ax.plot(k, lv[ep], "o", color=col[ep], ms=7, mec="k", mew=0.5)
                if ep != "pre" and ep in t.index and np.isfinite(t.loc[ep, "point"]):
                    r = t.loc[ep]
                    sig = r.lo > 0 or r.hi < 0
                    ax.text(k, ax.get_ylim()[1], f"{r.point:+.2f}{'*' if sig else ''}\n{int(r.n_dir)}/{int(r.n)}",
                            ha="center", va="top", fontsize=7, color=col[ep], fontweight="bold" if sig else "normal")
            ax.axhline(0 if term != "track" else lv["pre"], color="0.7", lw=0.6, ls=":")
            ax.set_xticks(range(4), EPS)
            ax.set_ylabel(lab[term], fontsize=8)
            ax.set_title(f"{SETS[st]}: {lab[term]}\nsession fits, grey = animals; text = change vs pre", fontsize=8.5)
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ + 0.25 * (hi_ - lo_))
        for ax in axs[i]:
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=col[e], marker="o", lw=1.3, label=e) for e in EPS], loc="lower center",
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Per-spout DECOMPOSITION of the decoded lateral shift (no pull slope assumed): per session, over spouts, "
                 "dDecoded_s = SHIFT + GAIN x (spout's pre landing - centre) + TRACK x dLanding_s.\nSHIFT = same at every "
                 "spout (+ = contralesional); GAIN < 0 = read-out compressed toward the centre; TRACK = share of the "
                 "landing change the read-out follows.\n95% paired nested bootstrap of the change vs pre, * = CI excludes "
                 "0 (bold), k/n animals. Frozen KEY-row decoder per set. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.87))
    fig.savefig(il.OUT / "figures" / "lick_decoded_decomposition.png", dpi=100)
    plt.close(fig)


def main() -> int:
    P = pd.read_csv(il.OUT / "lick_order_preds.csv.gz")
    S = sessions(P)
    S.to_csv(il.OUT / "decomposition_sessions.csv", index=False)
    pd.set_option("display.width", 230)
    print("== per animal (rule 8): shift / gain / track by epoch")
    for st in SETS:
        for term in (*TERMS, "shift_notrack", "shift_fixw"):
            print(f"-- {SETS[st]} {term}")
            print(S[S.set == st].pivot_table(index="animal", columns="epoch", values=term).reindex(columns=EPS)
                  .round(3).to_string())
    print("x vs dLand correlation (median per set x epoch):")
    print(S.groupby(["set", "epoch"]).corr_x_dLand.median().round(2).unstack().to_string())
    T = tests(S)
    T.to_csv(il.OUT / "decomposition_tests.csv", index=False)
    print(T[T.epoch != "pre"].round(3).to_string(index=False))
    plot(S, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
