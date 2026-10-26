"""RECOVERY TIMING, HITS ONLY (Priya 10-25: "how strong is the evidence that cortex recovers at / after behaviour" -> run
the hits-only test; and "can we predict animals' recovery from cortical activity?").

Per post-stroke SESSION: behaviour = far-contra hit rate (hits / far-contra trials, `qc/trial_attempts.csv`); neural =
frozen-decoder far-contra retention on HITS only (`attempt_coding/decoder_retention_sessions.csv`, sessions with >= 5
far-contra hits; removes the composition dependence -- failed attempts carry no code). Both normalised within animal to
its pre-stroke level (behaviour: pre far-contra hit rate; neural: 1 = pre hits by construction).

  1. per animal: exponential recovery y(d) = a - (a - b) exp(-d / tau) fitted to each measure's post sessions; half-recovery
     day = tau ln 2; 95 % CI from resampling the post sessions (1000 draws); neural - behaviour difference with its CI.
  2. session-to-session lagged models (post sessions in day order, consecutive pairs within animal; animal fixed effects):
     behaviour[t] ~ behaviour[t-1] + neural[t-1]   (does cortex predict the NEXT session's behaviour beyond behaviour?)
     neural[t]    ~ neural[t-1] + behaviour[t-1]   (does behaviour predict the next session's cortex beyond cortex?)
     OLS with animal dummies, cluster-robust SE by animal is meaningless at n = 4, so a session-level bootstrap within
     animal is used for the CIs. With 4 animals, between-animal prediction (who recovers fastest) is descriptive only.
Caveat: acute sessions often have < 5 far-contra hits, so the neural series starts later than the behavioural one.

Usage: python scripts/recovery_timing.py [--window lick]
Output: <E>/attempt_coding/recovery_timing_*.csv, figures/recovery_timing.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
A = E / "attempt_coding"
AN = ["PS92", "PS93", "PS94", "PS95"]
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}


def series(window: str) -> pd.DataFrame:
    T = pd.read_csv(E / "qc" / "trial_attempts.csv", usecols=["label", "pos", "cls", "animal", "epoch", "day"])
    T = T[(T.pos == "far_contra") & (T.label.str[-4:] >= "0801")]
    beh = T.assign(hit=T.cls.isin(["hit", "pre_hit"])).groupby(["animal", "epoch", "label", "day"]).hit.mean().reset_index()
    pre = beh[beh.epoch == "pre"].groupby("animal").hit.mean()
    beh["beh"] = beh.hit / beh.animal.map(pre)
    N = pd.read_csv(A / "decoder_retention_sessions.csv")
    N = N[(N.window == window) & (N.cls2 == "hit")][["label", "ret", "n"]].rename(columns={"ret": "neu", "n": "n_hits"})
    return beh.merge(N, on="label", how="left").sort_values(["animal", "day"]).reset_index(drop=True)


def exp_fit(d, y):
    from scipy.optimize import curve_fit
    ok = np.isfinite(y)
    d, y = d[ok], y[ok]
    if len(d) < 4:
        return np.nan

    def f(x, a, b, tau):
        return a - (a - b) * np.exp(-x / tau)
    try:
        (a, b, tau), _ = curve_fit(f, d, y, p0=(np.nanmax(y), np.nanmin(y), 5.0), bounds=([-2, -2, 0.3], [3, 3, 200]),
                                   maxfev=20000)
        return tau * np.log(2)
    except (RuntimeError, ValueError):
        return np.nan


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="lick", choices=("precue", "cue", "lick"))
    a = ap.parse_args(argv)
    S = series(a.window)
    post = S[S.epoch != "pre"].copy()
    rng = np.random.default_rng(0)
    rows = []
    for an in AN:
        g = post[post.animal == an]
        d = g.day.to_numpy(float)
        th_b, th_n = exp_fit(d, g.beh.to_numpy(float)), exp_fit(d, g.neu.to_numpy(float))
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(g), len(g))
            bs.append((exp_fit(d[i], g.beh.to_numpy(float)[i]), exp_fit(d[i], g.neu.to_numpy(float)[i])))
        bs = np.array(bs)
        diff = bs[:, 1] - bs[:, 0]
        ok = np.isfinite(diff)
        q = lambda x: np.nanpercentile(x, [2.5, 97.5]) if np.isfinite(x).sum() > 50 else (np.nan, np.nan)  # noqa: E731
        rows.append({"animal": an, "t_half_behaviour": th_b, "beh_lo": q(bs[:, 0])[0], "beh_hi": q(bs[:, 0])[1],
                     "t_half_neural": th_n, "neu_lo": q(bs[:, 1])[0], "neu_hi": q(bs[:, 1])[1],
                     "neural_minus_behaviour": th_n - th_b, "diff_lo": q(diff)[0], "diff_hi": q(diff)[1],
                     "share_neural_later": float((diff[ok] > 0).mean()) if ok.any() else np.nan,
                     "first_neural_day": float(g[np.isfinite(g.neu)].day.min()), "n_post_sessions": len(g),
                     "n_neural_sessions": int(np.isfinite(g.neu).sum())})
    H = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): half-recovery day (exponential fit), 95% CI from resampling post sessions")
    print(H.round(2).to_string(index=False))

    # lagged models, consecutive post sessions within animal (both measures present)
    L = []
    for an, g in post.groupby("animal"):
        g = g.sort_values("day").reset_index(drop=True)
        for i in range(1, len(g)):
            L.append({"animal": an, "day": g.day[i], "beh": g.beh[i], "neu": g.neu[i], "beh_prev": g.beh[i - 1],
                      "neu_prev": g.neu[i - 1], "gap": g.day[i] - g.day[i - 1]})
    L = pd.DataFrame(L).dropna()

    def fit(df, y, xs):
        X = np.column_stack([df[x].to_numpy(float) for x in xs] + [(df.animal == an).to_numpy(float) for an in AN])
        X = X[:, np.abs(X).sum(0) > 0]
        beta = np.linalg.lstsq(X, df[y].to_numpy(float), rcond=None)[0]
        return beta[:len(xs)]

    out = []
    for y, xs in (("beh", ["beh_prev", "neu_prev"]), ("neu", ["neu_prev", "beh_prev"])):
        b0 = fit(L, y, xs)
        bb = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(np.flatnonzero(L.animal == an), (L.animal == an).sum())
                                  for an in L.animal.unique()])
            bb.append(fit(L.iloc[idx], y, xs))
        bb = np.array(bb)
        for k, x in enumerate(xs):
            lo, hi = np.percentile(bb[:, k], [2.5, 97.5])
            out.append({"outcome": y + "[t]", "predictor": x.replace("_prev", "[t-1]"), "beta": b0[k], "lo": lo, "hi": hi,
                        "n_pairs": len(L)})
    M = pd.DataFrame(out)
    print("== session-to-session lagged models (animal fixed effects; bootstrap within animal)")
    print(M.round(3).to_string(index=False))
    H.to_csv(A / f"recovery_timing_halftimes_{a.window}.csv", index=False)
    M.to_csv(A / f"recovery_timing_lagged_{a.window}.csv", index=False)
    S.to_csv(A / f"recovery_timing_series_{a.window}.csv", index=False)

    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), sharey=True)
    for k, an in enumerate(AN):
        ax = axs[k]
        g = S[S.animal == an]
        ax.axvspan(g.day.min() - 1, 0, color="0.93", lw=0)
        ax.axhline(1, color="0.5", ls="--", lw=0.8)
        ax.plot(g.day, g.beh, "-o", color="0.25", ms=3.5, lw=1, label="behaviour: far-contra hit rate / pre")
        ax.plot(g.day, g.neu, "-s", color=ACOL[an], ms=4, lw=1.2, label="cortex: far-contra code on hits / pre")
        r = H[H.animal == an].iloc[0]
        ax.set_title(f"{an}: t1/2 behaviour {r.t_half_behaviour:.1f} d, cortex {r.t_half_neural:.1f} d\n"
                     f"cortex - behaviour {r.neural_minus_behaviour:+.1f} d [{r.diff_lo:+.1f}, {r.diff_hi:+.1f}]", fontsize=8.5)
        ax.set_xlabel("days from stroke")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("relative to the animal's pre-stroke level")
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Recovery timing, HITS ONLY ({a.window} window): behaviour vs the cortical far-contra code on successful trials; "
                 "half-recovery day from exponential fits, 95% CI by resampling sessions", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(A / "figures" / f"recovery_timing_{a.window}.png", dpi=120)
    plt.close(fig)
    print("->", A / "figures" / f"recovery_timing_{a.window}.png")
    pooled_figures(S, a.window, plt)
    return 0


#: day bins for pooling across animals (sessions fall on nearly the same days; narrow early, wider late)
DAY_BINS = [(-13, -1), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 9), (10, 12), (13, 16), (17, 20), (21, 24),
            (25, 28), (29, 33), (34, 38), (39, 44), (45, 51)]
EP4 = ["pre", "acute", "subacute", "chronic"]


def _boot_mean_of_animals(by: dict, rng, n=2000):
    """Mean of animal means and 95 % CI, resampling animals then sessions (levels; animal-weighted)."""
    an = [k for k, v in by.items() if len(v)]
    if not an:
        return np.nan, np.nan, np.nan
    point = float(np.mean([np.mean(by[k]) for k in an]))
    if len(an) < 2:
        return point, np.nan, np.nan
    d = [np.mean([np.mean(rng.choice(by[k], len(by[k]))) for k in rng.choice(an, len(an))]) for _ in range(n)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return point, float(lo), float(hi)


def pooled_figures(S: pd.DataFrame, window: str, plt) -> None:
    """POOLED ACROSS ANIMALS (Priya 10-25): per-animal-normalised behaviour and cortex (hits) (a) by day bin and (b) by
    epoch with one dot per session, plus the per-session GAP cortex - behaviour (does behaviour recover more than cortex?)."""
    from matplotlib.lines import Line2D

    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(0)
    S = S.copy()
    S["gap"] = S.neu - S.beh
    MEAS = [("beh", "behaviour: far-contra hit rate", "0.25"), ("neu", "cortex: far-contra code on hits", "#c0392b")]
    # (a) by day bin
    rows = []
    for lo, hi in DAY_BINS:
        g = S[(S.day >= lo) & (S.day <= hi)]
        x = float(np.mean([lo, hi])) if lo > 0 else -6.0
        for m, _l, _c in MEAS:
            by = {an: g[g.animal == an][m].dropna().to_numpy() for an in AN}
            pt, l_, h_ = _boot_mean_of_animals(by, rng)
            rows.append({"bin": f"{lo}..{hi}", "x": x, "measure": m, "mean": pt, "lo": l_, "hi": h_,
                         "n_animals": sum(len(v) > 0 for v in by.values()),
                         **{an: (float(np.mean(v)) if len(v) else np.nan) for an, v in by.items()}})
    Dd = pd.DataFrame(rows)
    Dd.to_csv(A / f"recovery_timing_pooled_day_{window}.csv", index=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.4))
    ax.axvspan(-13, 0, color="0.93", lw=0)
    ax.axhline(1, color="0.5", ls="--", lw=0.8)
    for m, lab, col in MEAS:
        t = Dd[Dd.measure == m].sort_values("x")
        for an in AN:
            ax.plot(t.x, t[an], color=col, lw=0.6, alpha=0.3)
        ax.fill_between(t.x, t.lo, t.hi, color=col, alpha=0.15, lw=0)
        ax.plot(t.x, t["mean"], "-o", color=col, lw=2, ms=4, label=lab)
    ax.set_xlabel("days from stroke (binned; pre = all August pre sessions at x = -6)")
    ax.set_ylabel("relative to the animal's pre-stroke level")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", label=lab) for _m, lab, c in MEAS]
              + [Line2D([], [], color="0.5", lw=0.6, alpha=0.5, label="each animal")], fontsize=8.5, frameon=False,
              loc="lower right")
    ax.set_title(f"Recovery, pooled across animals ({window} window, hits only for cortex): mean of animals, band = 95% CI "
                 "(animals > sessions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(A / "figures" / f"recovery_timing_pooled_day_{window}.png", dpi=120)
    plt.close(fig)

    # (b) by epoch, one dot per session; change vs pre (boot_delta); the gap panel
    PAN = MEAS + [("gap", "gap: cortex - behaviour", "#2471a3")]
    MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    tst = []
    for k, (m, lab, col) in enumerate(PAN):
        ax = axs[k]
        ax.axhline(1 if m != "gap" else 0, color="0.5", ls="--", lw=0.8)
        ref = {an: S[(S.animal == an) & (S.epoch == "pre")][m].dropna().to_numpy() for an in AN}
        ref = {an: v for an, v in ref.items() if len(v)}
        for i, ep in enumerate(EP4):
            g = S[S.epoch == ep]
            by = {an: g[g.animal == an][m].dropna().to_numpy() for an in AN}
            by = {an: v for an, v in by.items() if len(v)}
            for j, an in enumerate(AN):
                if an in by:
                    xs = i + (j - 1.5) * 0.07 + np.random.default_rng(10 * i + j).uniform(-0.02, 0.02, len(by[an]))
                    ax.scatter(xs, by[an], marker=MK[an], s=20, color=col, alpha=0.75, edgecolor="k", lw=0.3, zorder=4)
            if by:
                ax.plot([i - 0.3, i + 0.3], [np.mean([v.mean() for v in by.values()])] * 2, color="k", lw=2.2, zorder=5)
            com = [an for an in by if an in ref]
            if ep != "pre" and len(com) >= 2:
                iv = ak.boot_delta({an: by[an] for an in com}, {an: ref[an] for an in com}, np.random.default_rng(0))
                if iv is not None:
                    sig = iv.lo > 0 or iv.hi < 0
                    tst.append({"measure": m, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                                "animals": "/".join(com), "n_sessions": sum(len(by[an]) for an in com)})
                    ax.text(i, 1.42 if m != "gap" else 0.62, f"{iv.point:+.2f}\n[{iv.lo:+.2f}, {iv.hi:+.2f}]", ha="center",
                            va="top", fontsize=7.5, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(4)); ax.set_xticklabels(EP4)
        ax.set_ylim(*((-0.3, 1.45) if m != "gap" else (-1.1, 0.65)))
        ax.set_title(lab, fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("relative to the animal's pre-stroke level")
    Tt = pd.DataFrame(tst)
    Tt.to_csv(A / f"recovery_timing_pooled_epoch_{window}.csv", index=False)
    print("== pooled by epoch: change vs pre (nested bootstrap animals > sessions)\n" + Tt.round(3).to_string(index=False))
    fig.legend(handles=[Line2D([], [], marker=MK[an], ls="", mfc="0.6", mec="k", label=an) for an in AN]
               + [Line2D([], [], color="k", lw=2.2, label="mean of animals")], loc="lower center", ncol=5, fontsize=8.5,
               frameon=False)
    fig.suptitle(f"Recovery by epoch, one dot per session, normalised within animal ({window} window; cortex = hits only). "
                 "Numbers = change vs pre, 95% nested CI (bold = excludes 0); acute cortex: sessions with >= 5 hits only",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(A / "figures" / f"recovery_timing_pooled_epoch_{window}.png", dpi=120)
    plt.close(fig)
    print("->", A / "figures" / f"recovery_timing_pooled_epoch_{window}.png")


if __name__ == "__main__":
    raise SystemExit(main())
