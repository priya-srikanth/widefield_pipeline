"""FROZEN-DECODER RETENTION PER SESSION (Priya 10-25: session-level versions of the per-animal results; page Fig 7c).

Per far-contra target trial the frozen pre-stroke spout decoder gives P(true spout) (`attempt_coding` trials, LOSO for
pre). Retention = (P - 1/6) / (the animal's pre-stroke hit mean - 1/6): 1 = as well coded as that animal's pre-stroke
hits, 0 = chance. Per SESSION and class (hits = pre_hit + hit; failed attempts = incomplete) with >= MIN_TRIALS trials;
windows pre-cue (ENL), cue and lick. August pre only.

Figures (<E>/attempt_coding/figures/): decoder_retention_sessions.png (retention vs days from stroke) and
decoder_retention_pooled.png (pooled per epoch, one dot per session; numbers = change vs pre-stroke hits on session values,
`analysis_kit.boot_delta`, animals -> sessions; failed attempts have no pre-stroke counterpart, so their change is against
pre hits). Cells with too few trials per session (acute hits, ~4 per session) also show each animal's retention over ALL
its trials of the cell (hollow diamonds), which are not part of the test.
Usage: python scripts/decoder_retention_sessions.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
A = E / "attempt_coding"
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
WIN = [("precue", "pre-cue (ENL)"), ("cue", "cue window"), ("lick", "lick window")]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
CLS = [("hit", "hits", "#2c3e50"), ("failed", "failed attempts", "#c0392b")]
MIN_TRIALS = 5


def load() -> pd.DataFrame:
    out = []
    for w, _l in WIN:
        T = pd.concat([pd.read_csv(A / f"trials_{w}_{an}.csv.gz") for an in AN])
        T = T[(T.method == "dom") & (T.spout == "far_contra") & T.cls.isin(["pre_hit", "hit", "incomplete"])
              & (T.label.str[-4:] >= "0801")].copy()
        T["cls2"] = np.where(T.cls == "incomplete", "failed", "hit")
        ref = T[T.cls == "pre_hit"].groupby("animal").p_true.mean()
        T["ret"] = (T.p_true - 1 / 6) / (T.animal.map(ref) - 1 / 6)
        out.append(T.assign(window=w))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import analysis_kit as ak
    T = load()
    S = (T.groupby(["window", "animal", "epoch", "cls2", "label"])
         .agg(day=("day", "first"), ret=("ret", "mean"), p_true=("p_true", "mean"), n=("ret", "size")).reset_index())
    S = S[S.n >= MIN_TRIALS]
    Aall = T.groupby(["window", "animal", "epoch", "cls2"]).agg(ret=("ret", "mean"), n=("ret", "size")).reset_index()
    out = A / "figures"
    out.mkdir(exist_ok=True)
    S.to_csv(A / "decoder_retention_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): retention over all trials of the cell (n)")
    print(Aall.pivot_table(index=["window", "animal"], columns=["epoch", "cls2"], values="ret")
          .reindex(columns=EP, level=0).round(2).to_string())
    print(Aall.pivot_table(index=["window", "animal"], columns=["epoch", "cls2"], values="n")
          .reindex(columns=EP, level=0).to_string())

    def legend(fig, extra=()):
        h = [Line2D([], [], marker="o", ls="", mfc=c, mec=c, label=lab) for _k, lab, c in CLS]
        h += [Line2D([], [], marker=MK[a], ls="", mfc="w", mec="k", label=a) for a in AN] + list(extra)
        fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8.5, frameon=False)

    # FIG 1: retention vs days from stroke
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    for k, (w, wl) in enumerate(WIN):
        ax = axs[k]
        ax.axvspan(S.day.min() - 2, 0, color="0.93", lw=0, zorder=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        for c, _lab, col in CLS:
            for an in AN:
                v = S[(S.window == w) & (S.cls2 == c) & (S.animal == an)].sort_values("day")
                if len(v):
                    ax.plot(v.day, v.ret, color=col, lw=0.6, alpha=0.35)
                    ax.scatter(v.day, v.ret, marker=MK[an], s=26, facecolor=col if c == "hit" else "w", edgecolor=col, lw=1.1,
                               zorder=5)
        ax.set_title(wl, fontsize=10)
        ax.set_xlabel("days from stroke (grey = pre-stroke)")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("far-contra code retained\n(P(true) - 1/6) / (pre-stroke hits - 1/6)")
    legend(fig)
    fig.suptitle(f"NEURAL: frozen pre-stroke decoder on far-contra trials, PER SESSION (>= {MIN_TRIALS} trials of the class); 1 = the "
                 "animal's pre-stroke hits, 0 = chance (dashed = 1)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out / "decoder_retention_sessions.png", dpi=120)
    plt.close(fig)

    # FIG 2: pooled per epoch, one dot per session; change vs pre-stroke hits
    xo = {"hit": -0.18, "failed": 0.18}
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.0), sharey=True)
    for k, (w, wl) in enumerate(WIN):
        ax = axs[k]
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ref = {an: S[(S.window == w) & (S.epoch == "pre") & (S.cls2 == "hit") & (S.animal == an)].ret.to_numpy() for an in AN}
        ref = {an: v for an, v in ref.items() if len(v)}
        for i, ep in enumerate(EP):
            for c, _lab, col in CLS:
                xx = i + xo[c]
                v = S[(S.window == w) & (S.epoch == ep) & (S.cls2 == c)]
                by = {an: v[v.animal == an].ret.to_numpy() for an in AN if (v.animal == an).any()}
                for j, an in enumerate(AN):
                    if an in by:
                        xs = xx + (j - 1.5) * 0.04 + np.random.default_rng(10 * i + j).uniform(-0.015, 0.015, len(by[an]))
                        ax.scatter(xs, by[an], marker=MK[an], s=18, facecolor=col if c == "hit" else "w", edgecolor=col, lw=1.0,
                                   zorder=4)
                if len(by) >= 2:
                    ax.plot([xx - 0.15, xx + 0.15], [np.mean([b.mean() for b in by.values()])] * 2, color=col, lw=2.2, zorder=5)
                else:                                   # too few sessions: each animal's retention over all its trials
                    aa = Aall[(Aall.window == w) & (Aall.epoch == ep) & (Aall.cls2 == c) & (Aall.n >= 3)]
                    for _, r in aa.iterrows():
                        ax.scatter(xx + (AN.index(r.animal) - 1.5) * 0.04, r.ret, marker="D", s=40, facecolor="none",
                                   edgecolor=col, lw=1.2, zorder=4)
                com = [an for an in by if an in ref]
                if ep != "pre" and len(com) >= 2:
                    iv = ak.boot_delta({an: by[an] for an in com}, {an: ref[an] for an in com}, np.random.default_rng(0))
                    if iv is not None:
                        rows.append({"window": w, "epoch": ep, "class": c, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                                     "animals": "/".join(com), "n_sessions": sum(len(by[an]) for an in com)})
                        sig = iv.lo > 0 or iv.hi < 0
                        ax.text(xx, 2.18, f"{iv.point:+.2f}\n[{iv.lo:+.2f},\n{iv.hi:+.2f}]", ha="center", va="top", fontsize=6.5,
                                color=col, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(4)); ax.set_xticklabels(EP); ax.set_ylim(-1.0, 2.2)
        ax.set_title(wl, fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("far-contra code retained\n(1 = pre-stroke hits, 0 = chance)")
    R = pd.DataFrame(rows)
    R.to_csv(A / "decoder_retention_pooled.csv", index=False)
    print("== change vs pre-stroke hits (session values, nested bootstrap animals > sessions)\n" + R.round(3).to_string(index=False))
    legend(fig, [Line2D([], [], color="0.3", lw=2.2, label="mean of animals"),
                 Line2D([], [], marker="D", ls="", mfc="none", mec="0.3", label="animal, all trials (too few per session)")])
    fig.suptitle(f"NEURAL: frozen-decoder far-contra retention pooled per epoch, one dot per SESSION (>= {MIN_TRIALS} trials). Numbers = "
                 "change vs pre-stroke hits,\n95% nested CI (animals > sessions; bold = excludes 0); failed attempts have no pre-stroke "
                 "counterpart", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(out / "decoder_retention_pooled.png", dpi=120)
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
