"""Aim offset BY SPOUT x EPOCH (Priya 10-29: "behavioural deficit by angle offset from target over time by epochs, as an
addition to the spout contact data"; for the kinematics page).

THE aim definition (iti_recovery_measure.aim_licks, rule 9): per cued lick, 3-D angle atan2(lateral, sagittal) at the
extension peak minus the animal's pre mean angle at the same target spout; PRIMARY lick set = the FIRST post-cue lick of
each trial; a spout counts in a session with >= 5 such licks. Per session x spout: mean offset, minus the animal's mean
over its pre sessions at that spout (change vs own pre, deg). SIGN + = contralesional, the encoder goal page's convention (Priya 10-30:
align signs; the kinematics page's p_angle is + = ipsi and is to be aligned to this).

Fig A (POOLED): 6 spouts (canonical order) x epoch; dots = sessions (open, shape = animal, colour = epoch; A13, Priya
11-07); big marker = animal-weighted mean
(mean of per-animal epoch means), 95% CI = nested bootstrap animals -> sessions (2,000 draws); * CI excludes 0
(uncorrected), ** after Bonferroni over 18 tests (6 spouts x 3 post epochs). Pre column = pre sessions vs their own
mean (the noise floor).
Both figures carry a 7th panel, the WEIGHTED aim (iti_recovery_measure.weighted: far contra 3, far mid 2, others 1;
the goal page's behaviour axis), same change-vs-own-pre treatment (Priya 10-30).
Fig B (over time): per spout, change vs pre against day after stroke, one open point per session (shape = animal,
colour = epoch, no line), filled = epoch mean +- SEM over animals, plus the POOLED
cross-animal fit of the goal page's shared TIME model (wfield_local.mixed_models: y ~ post + post:lday, animal random
effects, lday = log days since stroke, 0 at day 1; the same model as goal-page Fig 16.9), fit per panel; BH-FDR over the
7 panels per term. Fit table: aim_by_spout_time_lmm.csv.
Usage: python scripts/lick_cued_aim_by_spout.py
Outputs: <E>/iti_licks/aim_by_spout_{sessions,tests,time_lmm}.csv, figures/lick_cued_aim_by_spout{,_days}.png
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_recovery_measure as rm  # noqa: E402

AN, EPS, POS = rm.AN, rm.EPS, rm.POS
POST = EPS[1:]
PANELS = POS + ["weighted"]
LAB = {**{p: p.replace("_", " ") for p in POS}, "weighted": "WEIGHTED aim\n(far contra 3 : far mid 2 : others 1)"}
#: THE epoch palette (orofacial_cohort.EPOCH_LINE; A13, Priya 11-07 -- the private copy had drifted). The private animal
#: colour dict is gone (config rule 3): animals are told apart by marker shape (cat_points.animal_marker).
from wfield_local.orofacial_cohort import EPOCH_LINE as ECOL  # noqa: E402
MINN = rm.MIN_LICKS["first"]
NB = 2000


def session_spout() -> pd.DataFrame:
    p = rm.aim_licks()
    p = p[p["first"]]
    g = p.groupby(["animal", "label", "epoch", "pos"]).off.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= MINN].rename(columns={"mean": "off", "size": "n"})
    w = p.groupby(["animal", "label", "epoch"]).apply(lambda x: rm.weighted(x, MINN)).rename("off").reset_index()
    w = w.dropna().assign(pos="weighted", n=p.groupby("label").size().reindex(w.label).to_numpy())
    g = pd.concat([g, w], ignore_index=True)
    base = g[g.epoch == "pre"].groupby(["animal", "pos"]).off.mean()
    g["chg"] = g.off - base.reindex(pd.MultiIndex.from_arrays([g.animal, g.pos])).to_numpy()  # + = contra
    from iti_check_figures import _days
    return g.merge(_days(), on="label", how="left")


def boot(d: pd.DataFrame, rng) -> np.ndarray:
    groups = [x.chg.to_numpy() for _, x in d.groupby("animal")]
    out = np.empty(NB)
    for b in range(NB):
        pick = rng.integers(len(groups), size=len(groups))
        out[b] = np.mean([rng.choice(groups[i], size=len(groups[i])).mean() for i in pick])
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import cat_points as cp
    S = session_spout()
    S.to_csv(il.OUT / "aim_by_spout_sessions.csv", index=False)
    rng = np.random.default_rng(0)
    T = []
    for pos in PANELS:
        for ep in EPS:
            d = S[(S.pos == pos) & (S.epoch == ep)]
            if d.empty:
                continue
            m = d.groupby("animal").chg.mean().mean()
            bs = boot(d, rng)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lob, hib = np.percentile(bs, [100 * 0.025 / 18, 100 * (1 - 0.025 / 18)])  # 18 = the 6 spouts (weighted: same bar)
            T.append(dict(pos=pos, epoch=ep, mean=m, lo=lo, hi=hi, lo_bonf=lob, hi_bonf=hib, n_animals=d.animal.nunique(),
                          n_sessions=len(d), sig=ep != "pre" and (lo > 0 or hi < 0),
                          sig_bonf=ep != "pre" and (lob > 0 or hib < 0),
                          **{f"{a}": d[d.animal == a].chg.mean() for a in AN}))
    T = pd.DataFrame(T)
    T.to_csv(il.OUT / "aim_by_spout_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal first (rule 8): change vs own pre, deg, + = contralesional")
    print(T[["pos", "epoch"] + AN].round(1).to_string(index=False))
    print("\n== pooled (animal-weighted)")
    print(T[["pos", "epoch", "mean", "lo", "hi", "sig", "sig_bonf", "n_animals", "n_sessions"]].round(1).to_string(index=False))

    # ---- Fig A: pooled, spout x epoch ----
    fig, axs = plt.subplots(1, 7, figsize=(19.5, 4.4), sharey=True)
    for ax, pos in zip(axs, PANELS):
        for k, ep in enumerate(EPS):
            d = S[(S.pos == pos) & (S.epoch == ep)]
            x = k + rng.uniform(-0.18, 0.18, len(d))      # same draw as before (rng is not reused for any number)
            for a in sorted(d.animal.unique()):
                kk = (d.animal == a).to_numpy()
                ax.plot(x[kk], d.chg.to_numpy()[kk], ls="none", marker=cp.animal_marker(a), ms=3.4, mfc="none",
                        mec=ECOL[ep], mew=0.8, alpha=0.65)
            r = T[(T.pos == pos) & (T.epoch == ep)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(k + 0.32, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt="o", ms=7,
                        color=ECOL[ep], capsize=3, lw=1.8)
            if r.sig:
                # Priya 10-30 figure rule: stars in a fixed band at the TOP of the panel (x = data, y = axes fraction),
                # never on the data or the pre / 0 line (same rule as wfield_local/sig_marks.star_top)
                ax.text(k + 0.32, 0.97, "**" if r.sig_bonf else "*", transform=ax.get_xaxis_transform(), ha="center",
                        va="top", color=ECOL[ep], fontsize=14, fontweight="bold")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["pre", "acute", "subac", "chron"], fontsize=8.5)
        ax.set_title(LAB[pos], fontsize=11 if pos != "weighted" else 9.5, fontweight="bold" if pos == "weighted" else None)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel("first-lick aim offset, change vs own pre\n(deg; + = contralesional)")
    lo_, hi_ = axs[0].get_ylim()                         # sharey: one range; the top band holds the stars, no data
    axs[0].set_ylim(lo_, hi_ + 0.14 * (hi_ - lo_))
    hs = cp.legend_handles(AN, EPS, sessions_label="{a} sessions", epoch_label="{e}: animal-weighted mean, 95% CI")
    fig.legend(handles=hs, loc="lower center", ncol=8, frameon=False, fontsize=8.5)
    fig.suptitle("Aim offset by spout (3-D angle of the FIRST cued lick minus the animal's pre landing at that spout), "
                 "POOLED (last panel = the weighted aim); dots = sessions; marker = animal-weighted mean, nested-bootstrap 95% CI;\n* CI excludes 0, "
                 "** after Bonferroni over 18 tests; pre column = pre sessions vs their own mean (noise floor)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    save_png_svg(fig, il.OUT / "figures" / "lick_cued_aim_by_spout.png", dpi=100)

    # ---- Fig B: per animal over days ----
    from wfield_local import mixed_models as mm
    # fit on the UNCENTRED offset (as Fig 16.9): centring each animal on its own pre first removes the between-animal
    # step / slope variance the random-effects ladder needs, and every panel fell back to ~1 (SEs too small). Plotted
    # relative to the fitted pre level, i.e. on the change-vs-pre axis.
    L = mm.add_time_covariates(S.drop(columns=["day"]).rename(columns={"off": "y"}))
    M = mm.lmm_table(L, ["pos"], models=("time",), print_animals=False, fdr_by=())
    M.to_csv(il.OUT / "aim_by_spout_time_lmm.csv", index=False)
    print("\n== pooled TIME model per panel (y ~ post + post:lday, animal random effects; q = BH over 7 panels)")
    print(M[["pos", "term", "coef", "se", "z", "q", "re", "n"]].round(3).to_string(index=False))
    fig, axs = plt.subplots(1, 7, figsize=(19.5, 4.6), sharey=True)
    for ax, pos in zip(axs, PANELS):
        d = S[S.pos == pos]
        cp.day_dots(ax, d.day, d.chg, d.animal, d.epoch, ms=3.2, summary_ms=5.5)
        c = M[M.pos == pos].set_index("term")
        if len(c):
            dd = np.linspace(1, S.day.max(), 200)
            ax.plot(dd, c.coef["post"] + c.coef["post:lday"] * np.log(dd), "k-", lw=2.2,
                    label="pooled fit")
            ax.plot([S.day.min(), -0.5], [0, 0], "k-", lw=2.2)
            st = lambda t, c=c: f"{c.coef[t]:+.1f} (q {c.q[t]:.3f}){'*' if c.q[t] < 0.05 else ''}"  # noqa: E731
            ax.text(0.97, 0.03, f"step {st('post')}\nrecovery/log-day {st('post:lday')}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.axhline(0, color="0.6", lw=0.7, ls=":")
        ax.set_title(LAB[pos], fontsize=11 if pos != "weighted" else 9.5, fontweight="bold" if pos == "weighted" else None)
        ax.set_xlabel("day after stroke")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel("first-lick aim offset, change vs own pre\n(deg; + = contralesional)")
    hs = cp.legend_handles(AN, EPS, sessions_label="{a} sessions", epoch_label="{e}: mean +- SEM over animals")
    hs += [plt.Line2D([], [], color="k", lw=2.2, label="pooled fit")]
    fig.legend(handles=hs, loc="lower center", ncol=9, frameon=False, fontsize=8.5)
    fig.suptitle("Aim offset by spout over days: open = one point per session with >= 5 first licks at the spout (shape = "
                 "animal, colour = epoch); filled = epoch mean +- SEM over animals;\nBLACK = POOLED cross-animal time model y ~ post + post:lday (animal random effects; the goal "
                 "page's Fig 16.9 model),\nstep = day-1 change vs pre, recovery = change per e-fold of days; q = BH over "
                 "the 7 panels", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    save_png_svg(fig, il.OUT / "figures" / "lick_cued_aim_by_spout_days.png", dpi=100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
