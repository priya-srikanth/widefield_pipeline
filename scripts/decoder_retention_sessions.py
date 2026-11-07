"""FROZEN-DECODER RETENTION PER SESSION (Priya 10-25: session-level versions of the per-animal results; page Fig 7c).

Per far-contra target trial the frozen pre-stroke spout decoder gives P(true spout) (`attempt_coding` trials, LOSO for
pre). Retention = (P - 1/6) / (the animal's pre-stroke hit mean - 1/6): 1 = as well coded as that animal's pre-stroke
hits, 0 = chance. Per SESSION and class (hits = pre_hit + hit; failed attempts = incomplete) with >= MIN_TRIALS trials;
windows pre-cue (ENL), cue and lick. August pre only.

Figures (<E>/attempt_coding/figures/): decoder_retention_sessions.png (retention vs days from stroke; rows = class) and
decoder_retention_pooled.png (pooled per epoch, one dot per session; numbers = change vs pre-stroke hits on session values,
`analysis_kit.boot_delta`, animals -> sessions; failed attempts have no pre-stroke counterpart, so their change is against
pre hits). Cells with too few trials per session (acute hits, ~4 per session) also show each animal's retention over ALL
its trials of the cell (hollow diamonds), which are not part of the test.
EPOCH-COLOUR CONVENTION (PI decision A13, Priya 11-07): every session point is an OPEN marker, shape = animal, colour =
epoch; no line joins sessions; the epoch summary is a filled dot in the epoch colour +- SEM over animals (each animal's
sessions averaged first). Trial class is told apart by panel row (days figure) or x offset (pooled figure).
Usage: python scripts/decoder_retention_sessions.py
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)


import numpy as np
import pandas as pd
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # rebuild root (11-03)

E = _encoding_root()
from wfield_local import attempt_coding as _ac  # noqa: E402
from wfield_local.attempt_coding import trials_name as attempt_trials_name  # noqa: E402

A = _ac.out_dir()   # the writer's folder under the run mode (review N1 / N10)
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
WIN = [("precue", "pre-cue (ENL)"), ("cue", "cue window"), ("lick", "lick window")]
CLS = [("hit", "hits", "#2c3e50"), ("failed", "failed attempts", "#c0392b")]   # colours unused since 11-07 (A13)
MIN_TRIALS = 5


def load() -> pd.DataFrame:
    from wfield_local import config
    out = []
    for w, _l in WIN:
        T = pd.concat([pd.read_csv(A / attempt_trials_name(w, an)) for an in AN])
        T = T[(T.method == "dom") & (T.spout == "far_contra") & T.cls.isin(["pre_hit", "hit", "incomplete"])
              & (~T.label.map(config.early_pre))].copy()
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
    from wfield_local import cat_points as cp
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
        h = cp.legend_handles(AN, EP) + list(extra)
        fig.legend(handles=h, loc="lower center", ncol=int(np.ceil(len(h) / 2)), fontsize=8.5, frameon=False)

    # FIG 1: retention vs days from stroke; rows = trial class (colour is the epoch's, A13)
    fig, axs = plt.subplots(len(CLS), 3, figsize=(17, 7.6), sharey=True, sharex=True, squeeze=False)
    for r_, (c, clab, _col) in enumerate(CLS):
        for k, (w, wl) in enumerate(WIN):
            ax = axs[r_, k]
            ax.axvline(0, color="0.6", lw=0.7, ls=":")
            ax.axhline(0, color="k", lw=0.6)
            ax.axhline(1, color="0.5", lw=0.8, ls="--")
            v = S[(S.window == w) & (S.cls2 == c)]
            cp.day_dots(ax, v.day, v.ret, v.animal, v.epoch)
            ax.set_title(f"{clab}, {wl}", fontsize=10)
            if r_ == len(CLS) - 1:
                ax.set_xlabel("days from stroke")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if k == 0:
                ax.set_ylabel(f"{clab}: far-contra code retained\n(P(true) - 1/6) / (pre-stroke hits - 1/6)", fontsize=9)
    legend(fig)
    fig.suptitle(f"NEURAL: frozen pre-stroke decoder on far-contra trials, PER SESSION (>= {MIN_TRIALS} trials of the class); 1 = the "
                 "animal's pre-stroke hits, 0 = chance (dashed = 1).\nOpen markers = sessions (shape = animal, colour = "
                 "epoch); filled = epoch mean +- SEM over animals", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_png_svg(fig, out / "decoder_retention_sessions.png", dpi=120)
    plt.close(fig)

    # FIG 2: pooled per epoch, one dot per session; change vs pre-stroke hits
    xo = {"hit": -0.18, "failed": 0.18}
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.0), sharey=True)
    allv = [-1.0, 2.0]                                    # the y range keeps every plotted value (A13: no clipping)
    for k, (w, wl) in enumerate(WIN):
        ax = axs[k]
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ref = {an: S[(S.window == w) & (S.epoch == "pre") & (S.cls2 == "hit") & (S.animal == an)].ret.to_numpy() for an in AN}
        ref = {an: v for an, v in ref.items() if len(v)}
        for i, ep in enumerate(EP):
            for c, _lab, _col in CLS:
                xx = i + xo[c]
                v = S[(S.window == w) & (S.epoch == ep) & (S.cls2 == c)]
                by = {an: v[v.animal == an].ret.to_numpy() for an in AN if (v.animal == an).any()}
                for j, an in enumerate(AN):
                    if an in by:
                        xs = xx + (j - 1.5) * 0.04 + np.random.default_rng(10 * i + j).uniform(-0.015, 0.015, len(by[an]))
                        ax.plot(xs, by[an], ls="none", marker=cp.animal_marker(an), ms=4, mfc="none",
                                mec=cp.epoch_color(ep), mew=0.9, alpha=0.8, zorder=4)
                        allv += list(by[an])
                if len(by) >= 2:                        # epoch summary: filled dot +- SEM over animals (A13)
                    cp.epoch_mean_sem(ax, xx, [b.mean() for b in by.values()], ep)
                else:                                   # too few sessions: each animal's retention over all its trials
                    aa = Aall[(Aall.window == w) & (Aall.epoch == ep) & (Aall.cls2 == c) & (Aall.n >= 3)]
                    for _, r in aa.iterrows():
                        ax.scatter(xx + (AN.index(r.animal) - 1.5) * 0.04, r.ret, marker="D", s=40, facecolor="none",
                                   edgecolor=cp.epoch_color(ep), lw=1.2, zorder=4)
                        allv.append(r.ret)
                com = [an for an in by if an in ref]
                if ep != "pre" and len(com) >= 2:
                    iv = ak.boot_delta({an: by[an] for an in com}, {an: ref[an] for an in com}, np.random.default_rng(0))
                    if iv is not None:
                        rows.append({"window": w, "epoch": ep, "class": c, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                                     "animals": "/".join(com), "n_sessions": sum(len(by[an]) for an in com)})
                        sig = iv.lo > 0 or iv.hi < 0
                        ax.text(xx, 0.995, f"{iv.point:+.2f}\n[{iv.lo:+.2f},\n{iv.hi:+.2f}]", ha="center", va="top",
                                fontsize=6.5, color=cp.epoch_color(ep), fontweight="bold" if sig else "normal",
                                transform=ax.get_xaxis_transform())
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{e}\nhits | failed" for e in EP])
        ax.set_title(wl, fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("far-contra code retained\n(1 = pre-stroke hits, 0 = chance)")
    lo_, hi_ = float(np.nanmin(allv)), float(np.nanmax(allv))  # every point inside; top 22 % kept for the numbers
    pad = 0.05 * (hi_ - lo_)
    axs[0].set_ylim(lo_ - pad, hi_ + pad + (hi_ - lo_ + 2 * pad) * 0.22 / 0.78)
    R = pd.DataFrame(rows)
    R.to_csv(A / "decoder_retention_pooled.csv", index=False)
    print("== change vs pre-stroke hits (session values, nested bootstrap animals > sessions)\n" + R.round(3).to_string(index=False))
    legend(fig, [Line2D([], [], marker="D", ls="", mfc="none", mec="0.3", label="animal, all trials (too few per session)")])
    fig.suptitle(f"NEURAL: frozen-decoder far-contra retention pooled per epoch, one dot per SESSION (>= {MIN_TRIALS} trials; left of each "
                 "epoch = hits, right = failed attempts). Numbers = "
                 "change vs pre-stroke hits,\n95% nested CI (animals > sessions; bold = excludes 0); failed attempts have no pre-stroke "
                 "counterpart", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    save_png_svg(fig, out / "decoder_retention_pooled.png", dpi=120)
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
