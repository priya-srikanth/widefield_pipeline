"""T20: is quiet cortex more SYNCHRONIZED after the stroke? (cohort1, 2026-11-01)

Priya's original hypothesis (handoff section 7): VLS stroke -> less tonic thalamocortical drive. Less-driven cortex is more
synchronized (large slow fluctuations, less fast activity; Poulet et al. 2012). dF/F cannot see tonic level, but it can see
synchronization. Index per area = log10(mean PSD 1-4 Hz / mean PSD 8-15 Hz) of the quiet engaged trace (quiet_fluct
spectra: Welch 4-s segments, dF/F % not detrended). Change vs the animal's August pre; ipsilesional vs contralesional
orofacial and limb; whole-cortex mean as the comparison. Reduced drive predicts a rise, strongest in orofacial cortex and,
if it comes through the ipsilesional thalamic loop, ipsilesionally.

    python scripts/cohort2_qc/quiet_sync_index.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_fluct import FL, TRACE_COLS, load  # noqa: E402

OUT = FL.parent / "sync_index"
EP = ["pre", "acute", "subacute", "chronic"]
LO, HI = (1.0, 4.0), (8.0, 15.0)


def main():
    from wfield_local import mixed_models as mm
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    D = load()
    rows = []
    for k, d in D.items():
        fr, P = d["freqs"], d["psd"]
        lo = (fr >= LO[0]) & (fr < LO[1])
        hi = (fr >= HI[0]) & (fr < HI[1])
        for j, c in enumerate(TRACE_COLS):
            rows.append(dict(label=k, animal=k[:4], epoch=d["epoch"], column=c,
                             sync=float(np.log10(np.nanmean(P[j, lo]) / np.nanmean(P[j, hi]))),
                             p_lo=float(np.log10(np.nanmean(P[j, lo]))), p_hi=float(np.log10(np.nanmean(P[j, hi])))))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "sync_sessions.csv", index=False)
    pre = S[S.epoch == "pre"].groupby(["animal", "column"])[["sync", "p_lo", "p_hi"]].mean()
    for m in ("sync", "p_lo", "p_hi"):
        S[f"d_{m}"] = S[m] - [pre.loc[(a, c), m] for a, c in zip(S.animal, S.column)]
    for m, lab in (("d_sync", "sync index change (log10 1-4 / 8-15 Hz)"), ("d_p_lo", "1-4 Hz power change (log10)"),
                   ("d_p_hi", "8-15 Hz power change (log10)")):
        print(f"\n{lab}, animal x epoch mean:\n",
              S[S.epoch != "pre"].pivot_table(index=["column", "animal"], columns="epoch", values=m, aggfunc="mean")
              .reindex(columns=EP[1:]).round(3).to_string())
    L = mm.add_time_covariates(S[["label", "column"]].assign(measure="sync", y=S.sync.to_numpy()), august_pre=True)
    T = mm.lmm_table(L.dropna(subset=["y"]), ["measure", "column"], models=("epoch",), print_animals=False,
                     fdr_by=("measure",))
    W = S.pivot_table(index=["label", "animal", "epoch"], columns="column", values="sync").reset_index()
    for g in ("orofacial", "limb"):
        W[f"{g}_ipsi_minus_contra"] = W[f"{g}_left"] - W[f"{g}_right"]
    W["orofacial_minus_cortex"] = W["orofacial"] - W["cortex"]
    L2 = pd.concat([W[["label"]].assign(measure="contrast", column=c, y=W[c].to_numpy())
                    for c in ("orofacial_ipsi_minus_contra", "limb_ipsi_minus_contra", "orofacial_minus_cortex")])
    L2 = mm.add_time_covariates(L2, august_pre=True)
    T2 = mm.lmm_table(L2.dropna(subset=["y"]), ["measure", "column"], models=("epoch",), print_animals=True,
                      fdr_by=("measure",))
    T = pd.concat([T, T2])
    T.to_csv(OUT / "lmm_sync.csv", index=False)
    print(T[T.term != "Intercept"][["measure", "column", "term", "coef", "se", "p", "q"]].round(4).to_string())
    figure(S, T)
    (OUT / "README.txt").write_text(__doc__)


def figure(S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    cols = ["orofacial_left", "orofacial_right", "limb_left", "limb_right", "cortex"]
    names = ["orofacial ipsi", "orofacial contra", "limb ipsi", "limb contra", "whole cortex"]
    fig, ax = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 4), sharey=True)
    animals = sorted(S.animal.unique())
    for a, c, nm in zip(ax, cols, names):
        G = S[(S.column == c) & (S.epoch != "pre")]
        for k, ep in enumerate(EP[1:]):
            g = G[G.epoch == ep]
            for an, x in g.groupby("animal"):
                a.plot(np.full(len(x), k + (animals.index(an) - 1.5) * 0.08), x.d_sync, ANIMAL_MARKER[an], mfc="none",
                       mec=EPOCH_LINE[ep], ms=4, alpha=0.7)
            m = g.groupby("animal").d_sync.mean()
            a.bar(k, m.mean(), width=0.5, color=EPOCH_LINE[ep], alpha=0.3)
            t = T[(T.measure == "sync") & (T.column == c) & T.term.str.contains(ep)]
            if len(t) and float(t.q.iloc[0]) < 0.05:
                sig_marks.star_top(a, k, "**" if float(t.q.iloc[0]) < 0.01 else "*")
        a.axhline(0, color="0.5", lw=0.8)
        a.set_xticks(range(3))
        a.set_xticklabels(EP[1:], fontsize=8)
        a.set_title(nm, fontsize=9)
    ax[0].set_ylabel("change vs pre, log10(PSD 1-4 Hz / 8-15 Hz)")
    sig_marks.reserve_headroom(fig)
    fig.suptitle("T20: quiet-period synchronization index (slow / fast power) after the stroke; a less thalamically driven "
                 "cortex is more synchronized. Sessions (shape = animal), bar = mean of animals; stars = LMM vs pre, FDR",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "quiet_sync_index.png", dpi=110)
    print("wrote", OUT / "quiet_sync_index.png")


if __name__ == "__main__":
    main()
