"""RECOVERY TIMING, HITS ONLY (Priya 10-25: "how strong is the evidence that cortex recovers at / after behaviour" -> run
the hits-only test; and "can we predict animals' recovery from cortical activity?").

Per post-stroke SESSION: behaviour = far-contra hit rate, THE Fig 0.2 / 1b rate (Priya 11-06, A21): column
``frac_hit_far_contra`` of the run's trial_attempts_sessions.csv (`trial_attempts.session_summary`: camera class contact
/ far-contra trials inside the CAMERA WORKING PERIOD -- denominator contact + incomplete + jaw_only + no_response; stopped,
stopped-licking, early-run, after-end and unknown trials excluded); a table built without the camera gate is refused.
Before 11-06 the denominator was every far-contra trial_attempts row (no period). Neural =
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
       python scripts/recovery_timing.py --aim [--plot-only]   (Fig 0.12 + far-contra AIM panel, 10-30 -> figures/
       recovery_timing_pooled_day_<window>_aim.png, recovery_timing_pooled_day_<window>_aim.csv,
       recovery_timing_aim_per_animal_<window>.csv, figdata/recovery_timing_pooled_day_<window>_aim.pkl)
Output: <E>/attempt_coding/recovery_timing_*.csv, figures/recovery_timing.png
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse

import numpy as np
import pandas as pd
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # rebuild root (11-03)

E = _encoding_root()
from wfield_local import attempt_coding as _ac  # noqa: E402
from wfield_local import lp_licks as _lp_licks  # noqa: E402  (coordinator 11-04: the run's attempts table)

A = _ac.out_dir()   # the writer's folder under the run mode (review N1 / N10)
AN = ["PS92", "PS93", "PS94", "PS95"]
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}


#: THE camera working-period hit rate of `trial_attempts.session_summary` (as fig_behaviour_hits.CAMERA_COL)
HIT_COL = "frac_hit_far_contra"
DATA_HIT = ("far-contra hit rate = contact / camera-working-period trials (first cue -> last camera lick, stopped runs "
            "A2 + excise out; as Fig 0.2 / 1b)")


def far_contra_hit(W: pd.DataFrame) -> pd.DataFrame:
    """[label, hit] from the run's trial_attempts_sessions.csv ``W`` (`trial_attempts.session_summary`): ``HIT_COL``, the
    camera working-period hit rate. Never re-derived here (A21). REFUSES a table built without the camera trial gate
    (no ``frac_stopped_*`` columns), like fig_behaviour_hits.camera_metrics; sessions without far-contra trials drop."""
    if not any(c.startswith("frac_stopped_") for c in W.columns):
        raise SystemExit("trial_attempts_sessions.csv is not camera-gated (no frac_stopped_* columns) -- build it with "
                         "scripts/trial_attempts.py under WIDEFIELD_TRIAL_GATE=camera (the render profile)")
    return W[["label", HIT_COL]].rename(columns={HIT_COL: "hit"}).dropna(subset=["hit"])


def series(window: str) -> pd.DataFrame:
    from wfield_local import config
    f = _lp_licks.attempts_path()
    T = pd.read_csv(f, usecols=["label", "animal", "epoch", "day"]).drop_duplicates("label")
    T = T[~T.label.map(config.early_pre)]
    H = far_contra_hit(pd.read_csv(f.with_name("trial_attempts_sessions.csv")))
    beh = T.merge(H, on="label", how="inner")[["animal", "epoch", "label", "day", "hit"]]
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
                 f"{DATA_HIT}; half-recovery day from exponential fits, 95% CI by resampling sessions", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, A / "figures" / f"recovery_timing_{a.window}.png", dpi=120)
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
                 "(animals > sessions)\n" + DATA_HIT, fontsize=10)
    fig.tight_layout()
    save_png_svg(fig, A / "figures" / f"recovery_timing_pooled_day_{window}.png", dpi=120)
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
                 "Numbers = change vs pre, 95% nested CI (bold = excludes 0); acute cortex: sessions with >= 5 hits only\n"
                 + DATA_HIT,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    save_png_svg(fig, A / "figures" / f"recovery_timing_pooled_epoch_{window}.png", dpi=120)
    plt.close(fig)
    print("->", A / "figures" / f"recovery_timing_pooled_epoch_{window}.png")


import iti_licks as _il  # noqa: E402
AIM_CSV = _il.OUT / "aim_by_spout_sessions.csv"                  # il.OUT: the ITI family's mode-routed root (N10)      # scripts/lick_cued_aim_by_spout.py (THE aim definition)
AIM_COL = "#2471a3"


def aim_series() -> pd.DataFrame:
    """Per session: far-contra FIRST post-cue lick 3-D aim offset, change vs the animal's own (August) pre mean, deg,
    + = contralesional (iti_recovery_measure.aim_licks via lick_cued_aim_by_spout.session_spout; >= 5 first licks)."""
    from wfield_local import config
    G = pd.read_csv(AIM_CSV)
    G = G[(G.pos == "far_contra") & (~G.label.map(config.early_pre))]
    return G[["animal", "label", "epoch", "day", "chg", "n"]].rename(columns={"chg": "aim"}).sort_values(["animal", "day"])


def aim_compute(window: str) -> dict:
    """Fig 0.12 + AIM (Priya 10-30: "can 0.12 include the far contra lick angle index?"). Panel 1 = the existing pooled
    day-bin figure (hit rate / pre, code on hits / pre; same rng order, so it reproduces recovery_timing_pooled_day);
    panel 2 = far-contra aim change vs own pre (deg) by the same day bins, per-animal + animal-weighted mean + 95% CI.
    Also per animal: acute / subacute / chronic means, recovered fraction of the acute deficit
    (x - acute) / (pre - acute) per measure, and exponential half-recovery days (aim scaled by the animal's |acute|)."""
    S = pd.read_csv(A / f"recovery_timing_series_{window}.csv")
    G = aim_series()
    rng = np.random.default_rng(0)
    rows = []
    for lo, hi in DAY_BINS:                                    # == pooled_figures (a), same draw order
        g = S[(S.day >= lo) & (S.day <= hi)]
        x = float(np.mean([lo, hi])) if lo > 0 else -6.0
        for m in ("beh", "neu"):
            by = {an: g[g.animal == an][m].dropna().to_numpy() for an in AN}
            pt, l_, h_ = _boot_mean_of_animals(by, rng)
            rows.append({"bin": f"{lo}..{hi}", "x": x, "measure": m, "mean": pt, "lo": l_, "hi": h_,
                         "n_animals": sum(len(v) > 0 for v in by.values()),
                         **{an: (float(np.mean(v)) if len(v) else np.nan) for an, v in by.items()}})
    rng2 = np.random.default_rng(1)
    for lo, hi in DAY_BINS:
        g = G[(G.day >= lo) & (G.day <= hi)]
        x = float(np.mean([lo, hi])) if lo > 0 else -6.0
        by = {an: g[g.animal == an].aim.dropna().to_numpy() for an in AN}
        pt, l_, h_ = _boot_mean_of_animals(by, rng2)
        rows.append({"bin": f"{lo}..{hi}", "x": x, "measure": "aim", "mean": pt, "lo": l_, "hi": h_,
                     "n_animals": sum(len(v) > 0 for v in by.values()),
                     **{an: (float(np.mean(v)) if len(v) else np.nan) for an, v in by.items()}})
    D = pd.DataFrame(rows)
    # per animal (rule 8): epoch means, recovered fraction of the acute deficit, half-recovery day
    M = S.merge(G[["label", "aim"]], on="label", how="outer")
    M["animal"] = M.label.str[:4]
    M["epoch"] = M.epoch.fillna(M.label.map(G.set_index("label").epoch))
    M["day"] = M.day.fillna(M.label.map(G.set_index("label").day))
    pa = []
    for an in AN:
        g = M[M.animal == an]
        e = g.groupby("epoch")[["beh", "neu", "aim"]].mean().reindex(EP4)
        r = {"animal": an}
        post = g[g.epoch != "pre"]
        for m in ("beh", "neu", "aim"):
            for ep in EP4:
                r[f"{m}_{ep}"] = e.loc[ep, m]
            pre_v = 0.0 if m == "aim" else 1.0                # pre reference (beh/neu normalised to pre; aim = change vs pre)
            ac = e.loc["acute", m]
            for ep in ("subacute", "chronic"):
                r[f"{m}_recov_{ep}"] = (e.loc[ep, m] - ac) / (pre_v - ac) if np.isfinite(ac) and ac != pre_v else np.nan
            y = post[m].to_numpy(float)
            if m == "aim":
                y = 1 + y / abs(ac) if np.isfinite(ac) and ac != 0 else y * np.nan   # 1 = pre, 0 = acute deficit
            r[f"{m}_t_half"] = exp_fit(post.day.to_numpy(float), y)
        pa.append(r)
    PA = pd.DataFrame(pa)
    return {"D": D, "PA": PA, "window": window}


def aim_draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    D, window = F["D"], F["window"]
    MEAS = [("beh", "far-contra hit rate / pre", "0.25"), ("neu", "cortex: far-contra frozen code on hits / pre", "#c0392b")]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [1, 0.9]})
    for a_ in (ax, bx):
        a_.axvspan(-13, 0, color="0.93", lw=0)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    ax.axhline(1, color="0.5", ls="--", lw=0.8)
    for m, _lab, col in MEAS:
        t = D[D.measure == m].sort_values("x")
        for an in AN:
            ax.plot(t.x, t[an], color=col, lw=0.6, alpha=0.3)
        ax.fill_between(t.x, t.lo, t.hi, color=col, alpha=0.15, lw=0)
        ax.plot(t.x, t["mean"], "-o", color=col, lw=2, ms=4)
    ax.set_ylabel("relative to the animal's\npre-stroke level")
    ax.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", label=lab) for _m, lab, c in MEAS]
              + [Line2D([], [], color="0.5", lw=0.6, alpha=0.5, label="each animal (faint)")], fontsize=8.5,
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("A  far-contra hit rate and frozen decoder code on hits (as Fig 0.12)", fontsize=10, loc="left")
    bx.axhline(0, color="0.5", ls="--", lw=0.8)
    t = D[D.measure == "aim"].sort_values("x")
    for an in AN:
        bx.plot(t.x, t[an], color=AIM_COL, lw=0.6, alpha=0.3)
    bx.fill_between(t.x, t.lo, t.hi, color=AIM_COL, alpha=0.15, lw=0)
    bx.plot(t.x, t["mean"], "-o", color=AIM_COL, lw=2, ms=4)
    bx.set_ylabel("far-contra aim offset,\nchange vs own pre (deg)\n+ = contralesional")
    bx.legend(handles=[Line2D([], [], color=AIM_COL, lw=2, marker="o", label="far-contra first-lick aim (3-D angle)"),
                       Line2D([], [], color=AIM_COL, lw=0.6, alpha=0.5, label="each animal (faint)")], fontsize=8.5,
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    bx.set_title("B  far-contra aim: first post-cue lick, 3-D angle minus the animal's pre mean (- = shifted ipsilesional)",
                 fontsize=10, loc="left")
    bx.set_xlabel("days from stroke (binned; pre = all August pre sessions at x = -6)")
    fig.suptitle(f"Recovery pooled across animals by day bin ({window} window; cortex = hits only): mean of animals, "
                 "band = 95% CI (animals > sessions). n = 4, PROVISIONAL\n" + DATA_HIT, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = A / "figures" / f"recovery_timing_pooled_day_{window}_aim.png"
    save_png_svg(fig, out, dpi=120)
    plt.close(fig)
    print("->", out)


def aim_main(window: str, plot_only: bool) -> int:
    from figdata import load_fd, save_fd
    stem, root = f"recovery_timing_pooled_day_{window}_aim", A / "figdata"
    if not plot_only:
        F = aim_compute(window)
        pd.set_option("display.width", 250)
        print("== per animal (rule 8): epoch means (beh, neu = / pre; aim = deg vs pre), recovered fraction of the acute "
              "deficit, half-recovery day\n" + F["PA"].round(2).to_string(index=False))
        F["D"].to_csv(A / f"recovery_timing_pooled_day_{window}_aim.csv", index=False)
        F["PA"].to_csv(A / f"recovery_timing_aim_per_animal_{window}.csv", index=False)
        ref = pd.read_csv(A / f"recovery_timing_pooled_day_{window}.csv")
        mine = F["D"][F["D"].measure != "aim"].reset_index(drop=True)
        print("== check vs recovery_timing_pooled_day (beh/neu): max |diff| =",
              float(np.nanmax(np.abs(mine[["mean", "lo", "hi"]].to_numpy() - ref[["mean", "lo", "hi"]].to_numpy()))))
        save_fd(stem, F, root)
    aim_draw(load_fd(stem, root))
    return 0


if __name__ == "__main__":
    import sys
    if "--aim" in sys.argv:          # Fig 0.12 + aim panel: python scripts/recovery_timing.py --aim [--plot-only] [--window lick]
        w = sys.argv[sys.argv.index("--window") + 1] if "--window" in sys.argv else "lick"
        raise SystemExit(aim_main(w, "--plot-only" in sys.argv))
    raise SystemExit(main())
