"""Step or gradual? (Priya 11-07: "it seems like there's an abrupt transition acute to subacute, but can we show some of
this as lmm against recovery (eg far contra aim offset) or in time, to see if it's gradual vs a step").

Per session (first-lick-aligned state-space readouts; WF_SS_ALIGN / WF_SS_XFORM select the run) joined to the
behavioural recovery axis: far-contra FIRST-cued-lick aim vs the animal's pre landing (deg, - = ipsilesional;
`aim_deficit_sessions.csv` farcontra_first, the 14.16 recovery axis, rebuild _daqlk run).

1. AGAINST RECOVERY, post-stroke sessions: per-animal r (printed first, rule 8); within-animal slope of readout on aim
   (animal fixed effects) with a within-animal shuffle null (aim permuted within animal, 2000 draws); descriptive
   MixedLM readout ~ aim + (1 | animal); and readout ~ aim + C(epoch) + (1 | animal): does aim explain the readout
   WITHIN epochs (gradual coupling) or only through the epoch step?
2. AGAINST TIME, post-stroke sessions, per animal, leave-one-session-out squared error of
     step     value = a + b [day >= c]                      (c on a 1-day grid)
     ramp     value = a + b clip((day - d0) / (d1 - d0), 0, 1)
     exp      value = a + b (1 - exp(-day / tau))
     log      value = a + b ln(day)                          (Priya 11-07: "should we try log curves?")
   with 2-4 fitted parameters by grid + least squares; the same for the aim itself, so the neural and the
   behavioural time courses are compared on one footing. Reported: which model wins per animal, and the fitted step
   day / ramp span / tau.
Output: Widefield/state_space_recovery/step_vs_gradual{tag}/, page_figs/ss_step_{suffix}.png. Exploratory.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import state_space_page_figs as F  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402  (PNG + svg/ twin, Priya 11-06)

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
OUT = ROOT / f"step_vs_gradual{S.tag()}"
SUF = S.tag().lstrip("_") or "cue"
AIM = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
           "iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-"
           "incomplete-stopA2jexrt3500c70_daqlk/aim_deficit_sessions.csv")
READ = [("x", "position pre (0) -> chronic (1)\n(fraction of pre -> chronic distance)"),
        ("y", "acute-departure axis\n(same units)"),
        ("dist_pre", "distance from pre sessions\n(pre-pre median = 1)"),
        ("r", "pattern r with pre\n(correlation)"),
        ("amp", "amplitude ratio\n(pre-pre \u2248 1)")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def load():
    tag = S.tag()
    s = pd.read_csv(ROOT / f"trials{tag}" / "state_space_trials_sessions.csv")[["label", "animal", "epoch", "day", "r",
                                                                                 "amp", "dist_pre"]]
    xy = pd.read_csv(ROOT / f"trajectories{tag}" / "days_sessions.csv")[["label", "x", "y"]]
    aim = pd.read_csv(AIM)[["label", "farcontra_first"]].rename(columns={"farcontra_first": "aim"})
    return s.merge(xy, on="label", how="left").merge(aim, on="label", how="left")


def within_slope(d, c):
    g = d.dropna(subset=[c, "aim"])
    x = g.aim - g.groupby("animal").aim.transform("mean")
    y = g[c] - g.groupby("animal")[c].transform("mean")
    return float((x * y).sum() / (x ** 2).sum()), g


def shuffle_p(d, c, rng, n=2000):
    b, g = within_slope(d, c)
    null = []
    for _ in range(n):
        h = g.copy()
        h["aim"] = h.groupby("animal").aim.transform(lambda v: rng.permutation(v.to_numpy()))
        null.append(within_slope(h, c)[0])
    null = np.asarray(null)
    return b, float((np.abs(null) >= abs(b)).mean())


def mixed(d, c, formula):
    import statsmodels.formula.api as smf
    g = d.dropna(subset=[c, "aim"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula.format(c=c), g, groups=g.animal).fit(reml=True)
    return float(m.params["aim"]), float(m.pvalues["aim"])


# ---------------------------------------------------------------- time models

def design(day, kind, par):
    if kind == "step":
        return (day >= par).astype(float)
    if kind == "ramp":
        d0, d1 = par
        return np.clip((day - d0) / max(d1 - d0, 1e-9), 0, 1)
    if kind == "log":
        return np.log(np.maximum(day, 0.5))
    return 1 - np.exp(-day / par)


MODELS = ("step", "ramp", "exp", "log")
GRID = {"step": [c for c in np.arange(1.5, 30.5, 1.0)],
        "ramp": [(a, b) for a in np.arange(0, 20, 1.0) for b in np.arange(a + 2, 40, 2.0)],
        "exp": list(np.geomspace(0.5, 40, 40)), "log": [None]}


def fit(day, v, kind):
    best = None
    for par in GRID[kind]:
        X = np.column_stack([np.ones_like(day), design(day, kind, par)])
        coef, *_ = np.linalg.lstsq(X, v, rcond=None)
        sse = float(((v - X @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, par, coef)
    return best


def loso(day, v, kind):
    err = []
    for i in range(len(day)):
        m = np.arange(len(day)) != i
        _, par, coef = fit(day[m], v[m], kind)
        err.append((v[i] - (coef[0] + coef[1] * design(day[i:i + 1], kind, par)[0])) ** 2)
    return float(np.mean(err))


def time_models(d, c):
    rows = []
    for an, g in d[d.epoch != "pre"].dropna(subset=[c]).groupby("animal"):
        day, v = g.day.to_numpy(float), g[c].to_numpy(float)
        if len(g) < 8:
            continue
        r = {"animal": an, "readout": c, "n": len(g)}
        for k in MODELS:
            r[f"cv_{k}"] = loso(day, v, k)
            r[f"par_{k}"] = fit(day, v, k)[1]
        r["best"] = min(MODELS, key=lambda k: r[f"cv_{k}"])
        rows.append(r)
    return pd.DataFrame(rows)


def figure(d, models):
    cols = READ + [("aim", "far-contra aim vs own pre\n(deg, - = ipsilesional)")]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.3 * len(cols), 7.2))
    for j, (c, t) in enumerate(cols):
        ax = axs[0, j]                                       # against time (log days), per-animal log fits
        pre = d[d.epoch == "pre"][c].dropna()
        ax.axhspan(pre.mean() - pre.std(), pre.mean() + pre.std(), color="0.88", zorder=0, lw=0)
        ax.axhline(pre.mean(), color="0.6", lw=0.8, zorder=0)
        dd = np.geomspace(1, 52, 200)
        curves = []
        for an, g in d[d.epoch != "pre"].groupby("animal"):
            ax.scatter(g.day, g[c], marker=ANIMAL_MARKER[an], facecolors="none",
                       edgecolors=[EPOCH_LINE[e] for e in g.epoch], s=16, lw=0.8, zorder=3)
            gp = g.dropna(subset=[c])
            if len(gp) >= 8:
                _, par, coef = fit(gp.day.to_numpy(float), gp[c].to_numpy(float), "log")
                curves.append(coef[0] + coef[1] * design(dd, "log", par))
                ax.plot(dd, curves[-1], color="0.55", lw=0.8, zorder=2)
        if curves:
            ax.plot(dd, np.mean(curves, axis=0), color="k", lw=2.2, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        mm = models[models.readout == c]
        wins = ", ".join(f"{k} {int((mm.best == k).sum())}" for k in MODELS if (mm.best == k).any())
        ax.set_title(f"{t}\nbest by CV (of 4 animals): {wins}", fontsize=8.5)
        ax.set_xlabel("days from stroke (log)")
        if j < len(READ):                                                       # against recovery
            ax = axs[1, j]
            for an, g in d[d.epoch != "pre"].groupby("animal"):
                ax.scatter(g.aim, g[c], marker=ANIMAL_MARKER[an], facecolors="none",
                           edgecolors=[EPOCH_LINE[e] for e in g.epoch], s=18, lw=0.9)
                gg = g.dropna(subset=[c, "aim"])
                if len(gg) > 3:
                    b = np.polyfit(gg.aim, gg[c], 1)
                    xx = np.array([gg.aim.min(), gg.aim.max()])
                    ax.plot(xx, np.polyval(b, xx), color="0.35", lw=0.9)
            pre = d[d.epoch == "pre"]
            ax.scatter(pre.aim, pre[c], color="0.75", s=10, zorder=0)
            ax.set_xlabel("far-contra aim vs own pre (deg)")
            rr = [g[[c, "aim"]].dropna().corr().iloc[0, 1] for _, g in d[d.epoch != "pre"].groupby("animal")]
            ax.set_title(t.split("\n")[0] + " vs recovery\nwithin-animal r " + " / ".join(f"{v:.2f}" for v in rr),
                         fontsize=8.5)
    axs[1, -1].axis("off")
    hs = [plt.Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", mec="k", label=a) for a in sorted(ANIMAL_MARKER)]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EPOCH_LINE[e], label=e) for e in EPOCH_LINE]
    hs += [plt.Line2D([], [], color="0.55", lw=0.8, label="log fit per animal (top) / line per animal (bottom)"),
           plt.Line2D([], [], color="k", lw=2.2, label="mean of the 4 log fits"),
           plt.Line2D([], [], color="0.88", lw=6, label="pre mean \u00b1 SD"),
           plt.Line2D([], [], marker="o", ls="", color="0.75", label="pre sessions (bottom)")]
    fig.legend(handles=hs, loc="upper center", ncol=11, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Step or gradual? State-space readouts against time and against far-contra aim recovery "
                 f"({F.ALIGN_LABEL[S.ALIGN]}{'; ' + S.XFORM if S.XFORM else ''})", y=1.09)
    fig.tight_layout()
    save_png_svg(fig, ROOT / "page_figs" / f"ss_step_{SUF}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    d = load()
    post = d[d.epoch != "pre"]
    rows = []
    for c, _ in READ:
        r_an = {an: round(float(g[[c, "aim"]].dropna().corr().iloc[0, 1]), 2) for an, g in post.groupby("animal")}
        b, p = shuffle_p(post, c, rng)
        b_lmm, p_lmm = mixed(post, c, "{c} ~ aim")
        b_ep, p_ep = mixed(post, c, "{c} ~ aim + C(epoch)")
        rows.append({"readout": c, "r_per_animal": r_an, "within_slope": b, "shuffle_p": p, "lmm_slope": b_lmm,
                     "lmm_p": p_lmm, "slope_given_epoch": b_ep, "p_given_epoch": p_ep})
    A = pd.DataFrame(rows)
    models = pd.concat([time_models(d, c) for c, _ in READ + [("aim", "")]])
    A.to_csv(OUT / "against_recovery.csv", index=False)
    models.to_csv(OUT / "time_models.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60, "display.precision", 3):
        print(A.to_string(index=False))
        print(models[["animal", "readout", "n", "cv_step", "cv_ramp", "cv_exp", "best", "par_step", "par_ramp",
                      "par_exp"]].to_string(index=False))
        print(models.groupby("readout").best.value_counts().unstack(fill_value=0))
    figure(d, models)


if __name__ == "__main__":
    main()
