"""PRIMARY recovery marker (Priya 10-28: "proceed with the model including SSs"; "we'll use first ITI lick after quiet and
first post-cue lick based windows"): the hemispheric balance as an angle-matched GAIN RATIO, pooled over the orofacial
areas + SSs (ALM, tjM1, SSp-m, SSp-n, SSs), on
  cued: the FIRST post-cue lick, 0.1..0.5 s after it
  ITI:  the FIRST ITI lick after >= 2 s without licking, the same 0.1..0.5 s (held-out window check in
        lick_balance_subsets: every window from 0.1 s gives ~0.40 for ITI; 0.1..0.5 best for both, 0.70 / 0.42)
Gain ratio = log2(sum contra / sum of its pre-line prediction) - log2(same, ipsi), the prediction = the animal's PRE line on
lick angle per lick set x lick order (pre sessions leave-one-pre-session-out, so pre values are a real noise floor);
session values come from lick_balance_subsets (balsub_sessions.csv; per-area sums, pooled here).
Targets: cued -> far-contra first-lick aim (deg vs pre, key figure 1A); ITI -> ITI's own axis (session mean ITI lick
angle vs pre) and the cued aim.
Models (post-stroke sessions; per-animal r first, rule 8): concurrent MixedLM y ~ within + between + log(day) +
(1 | animal) and with a per-animal spline in log(day); predictive (next-session change ~ marker + current y + log(day));
cross-validated: leave-one-ANIMAL-out with a within-animal shuffle null, within-animal BLOCKED, leave-one-session-out
animal unknown / known (lick_recovery_markers.models).
Outputs <OUT_W>/recprim_sessions.csv, recprim_models.csv, recprim_cv.csv; figure lick_recovery_primary.png
(figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
CUED_WIN = "ratio_010_050"             # Priya 10-28: one window for both; 0.1..0.5 best for both (held-out window check)
ITI_WIN = "ratio_010_050"
PANELS = [(lrm.C1SET, CUED_WIN, "aim", "far-contra first-lick aim (deg vs pre)"),
          (lrm.QSET, ITI_WIN, "iti_ang", "ITI lick angle vs pre (deg)"),
          (lrm.QSET, ITI_WIN, "aim", "far-contra first-lick aim (deg vs pre)")]
AN = lrm.AN
REF = os.environ.get("WFIELD_BAL_REF", "angle")         # pre-line covariate (lick_balance_subsets.REF)
REFMETHOD = os.environ.get("WFIELD_BAL_REFMETHOD", "line")
INSIDE = os.environ.get("WFIELD_BAL_INSIDE", "") == "1"
SFX = ("" if REF == "angle" else f"_{REF}") + ("" if REFMETHOD == "line" else f"_{REFMETHOD}") + ("_inside" if INSIDE else "") + ("_kin" if os.environ.get("WFIELD_BAL_KIN", "") == "1" else "") + ("_full" if os.environ.get("WFIELD_BAL_FULL", "") == "1" else "")

# DISPLAY convention for the balance (Priya 10-29), shared by the balance figure scripts. Stored values / CSVs keep the
# HEMISPHERIC sign (marker = log2 contra gain - log2 ipsi gain; + = contralesional larger). On the key figures it is shown
# as IMPLIED DIRECTION = -balance, so it reads like lick angle, aim and the decoder offset: in the pre-stroke code each
# hemisphere responds more to licks toward the opposite side, so a lower contra/ipsi balance looks like a more
# contralesional lick (+ = read contralesional, as the decoder's like-for-like offset). DISPLAY ONLY: flip plotted values,
# reference lines, slopes and r's drawn from them; cvR2 / p do not flip. WFIELD_BAL_DISPLAY=hemisphere = the old look.
BAL_DISPLAY = os.environ.get("WFIELD_BAL_DISPLAY", "direction")
if BAL_DISPLAY not in ("direction", "hemisphere"):
    raise SystemExit(f"WFIELD_BAL_DISPLAY must be 'direction' or 'hemisphere', not {BAL_DISPLAY!r}")
BAL_SIGN = -1.0 if BAL_DISPLAY == "direction" else 1.0
BAL_SHORT = "implied direction" if BAL_DISPLAY == "direction" else "balance"
BAL_LABEL = ("implied direction (+ = activity looks like a more contralesional lick)\n= -(contralesional - ipsilesional "
             "balance)" if BAL_DISPLAY == "direction" else
             "hemispheric balance: log2(contra gain) - log2(ipsi gain)\n(+ = contralesional hemisphere larger)")


def bal_disp(v):
    """Balance (hemispheric sign, as stored) -> the value to PLOT under BAL_DISPLAY. Also use it on anything drawn from
    the balance with the balance's sign: a slope / r against it (not on cvR2 or p)."""
    return BAL_SIGN * v


def marker(S: pd.DataFrame, dn: str) -> pd.Series:
    g = []
    for s in ("c", "i"):
        o = S[[f"{dn}|{a}|obs_{s}" for a in POOL]].sum(1)
        p = S[[f"{dn}|{a}|pred_{s}" for a in POOL]].sum(1)
        g.append(np.where((o > 0) & (p > 0), np.log2(o.clip(lower=1e-12) / p.clip(lower=1e-12)), np.nan))
    return pd.Series(g[0] - g[1], index=S.index)


def compute() -> dict:
    import statsmodels.formula.api as smf
    S = pd.read_csv(il.OUT_W / f"balsub_sessions{SFX}.csv")
    rows = []
    for st, dn, yv, _yl in PANELS:
        t = S[S.set == st].copy()
        rows.append(pd.DataFrame({"panel": f"{st} -> {yv}", "set": st, "def": dn, "yname": yv, "label": t.label,
                                  "animal": t.animal, "epoch": t.epoch, "day": t.day, "y": t[yv],
                                  "marker": marker(t, dn), "n_licks": t.get("n_licks", np.nan)}))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(il.OUT_W / f"recprim_sessions{SFX}.csv", index=False)
    M, CV, ANr, LC = [], [], [], []
    for pn, G in T.groupby("panel", sort=False):
        post = G[(G.epoch != "pre") & (G.day > 0)].dropna(subset=["y", "marker"]).sort_values(["animal", "day"]).copy()
        post["logday"] = np.log(post.day)
        for a_, h in post.groupby("animal"):
            ANr.append({"panel": pn, "animal": a_, "n": len(h), "r": np.corrcoef(h.marker, h.y)[0, 1],
                        "r_days": np.corrcoef(h.logday, h.y)[0, 1]})
        post["w"] = post.marker - post.groupby("animal").marker.transform("mean")
        post["b"] = post.groupby("animal").marker.transform("mean")
        post["y_next"] = post.groupby("animal").y.shift(-1)
        post["d_y"] = post.y_next - post.y
        for name, f, data in (("concurrent, log(day)", "y ~ w + b + logday", post),
                              ("concurrent, per-animal spline", "y ~ w + b + C(animal):bs(logday, df=3)", post),
                              ("predictive (next-session change)", "d_y ~ w + b + y + logday", post.dropna(subset=["d_y"]))):
            try:
                m = smf.mixedlm(f, data, groups=data["animal"]).fit(reml=False)
                ci = m.conf_int()
                M.append({"panel": pn, "model": name, "n": len(data), "slope": m.fe_params["w"], "lo": ci.loc["w", 0],
                          "hi": ci.loc["w", 1], "p": m.pvalues["w"]})
            except Exception as ex:                                    # noqa: BLE001
                M.append({"panel": pn, "model": f"{name} FAILED {ex}", "n": len(data)})
        F = lrm.models(G[["set", "label", "animal", "epoch", "day"]].assign(aim=G.y, marker=G.marker))
        CV.append(F["CV"].assign(panel=pn))
        # LICK COUNT (Priya 10-28 "how do we account for variable lick number"): the gain is a ratio of sums over the
        # same licks, so counts cancel in the value; here: does the count track the marker or the recovery axis, and do
        # a minimum count / count weighting change the cross-animal result?
        if post.n_licks.notna().any():
            import lick_balance_nested as bn
            x, y, grp, n = post.marker.to_numpy(), post.y.to_numpy(), post.animal.to_numpy(), post.n_licks.to_numpy(float)
            m20 = n >= 20
            pred = np.full(len(y), np.nan)
            for a_ in np.unique(grp):                    # count-weighted leave-one-animal-out OLS
                te = grp == a_
                Xw = np.column_stack([np.ones((~te).sum()), x[~te]]) * np.sqrt(n[~te])[:, None]
                b = np.linalg.lstsq(Xw, y[~te] * np.sqrt(n[~te]), rcond=None)[0]
                pred[te] = b[0] + b[1] * x[te]
            LC.append({"panel": pn, "n_median": float(np.median(n)), "n_min": float(n.min()),
                       "n_lt20": int((~m20).sum()),
                       "r_n_marker_within": float(np.nanmean([np.corrcoef(n[grp == a], x[grp == a])[0, 1]
                                                              for a in np.unique(grp)])),
                       "r_n_y_within": float(np.nanmean([np.corrcoef(n[grp == a], y[grp == a])[0, 1] for a in np.unique(grp)])),
                       "loao_all": bn._loao_r2(x, y, grp),
                       "loao_min20": bn._loao_r2(x[m20], y[m20], grp[m20]) if m20.sum() > 20 else np.nan,
                       "loao_count_weighted": 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)})
    M, CV, ANr = pd.DataFrame(M), pd.concat(CV, ignore_index=True), pd.DataFrame(ANr)
    LC = pd.DataFrame(LC)
    LC.to_csv(il.OUT_W / f"recprim_lickcount{SFX}.csv", index=False)
    print("== lick count per session (counts cancel in the gain; checks)\n" + LC.round(3).to_string(index=False))
    M.to_csv(il.OUT_W / f"recprim_models{SFX}.csv", index=False)
    CV.to_csv(il.OUT_W / f"recprim_cv{SFX}.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): r of marker / of log(day) with the recovery axis\n"
          + ANr.pivot_table(index="panel", columns="animal", values=["r", "r_days"], sort=False).round(2).to_string())
    print("== mixed models\n" + M.round(4).to_string(index=False))
    print("== cross-validated\n" + CV[["panel", "pred", "loao", "p_shuffle", "loblk", "loblk_plus_days", "loso_unknown",
                                       "loso_known", "loso_known_plus_days"]].round(3).to_string(index=False))
    return {"T": T, "M": M, "CV": CV, "AN": ANr}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    T, M, CV, ANr = F["T"], F["M"], F["CV"], F["AN"]
    fig, axs = plt.subplots(2, len(PANELS), figsize=(6.2 * len(PANELS), 11), squeeze=False)
    for j, (st, dn, yv, yl) in enumerate(PANELS):
        pn = f"{st} -> {yv}"
        G = T[T.panel == pn]
        ax = axs[0, j]
        pre = G[G.epoch == "pre"]
        ax.plot(bal_disp(pre.marker), pre.y, "o", color="0.82", mec="0.6", mew=0.4, ms=6, zorder=0)
        for (ep, a_), h in G[G.epoch != "pre"].groupby(["epoch", "animal"]):
            ax.plot(bal_disp(h.marker), h.y, ANIMAL_MARKER.get(a_, "o"), color=EPOCH_LINE[ep], ms=6.5, mec="k", mew=0.3)
        r = ANr[ANr.panel == pn].set_index("animal")
        ttl = f"{pn}\nper-animal r " + " / ".join(f"{bal_disp(r.r.get(a, np.nan)):+.2f}" for a in AN)
        for name in ("concurrent, log(day)", "concurrent, per-animal spline", "predictive (next-session change)"):
            m = M[(M.panel == pn) & (M.model == name)]
            if len(m) and "p" in m and np.isfinite(m.p.iloc[0]):
                s_ = f"{name}: {bal_disp(m.slope.iloc[0]):+.1f}, p {m.p.iloc[0]:.2g}"
                ttl += "\n" + (rf"$\mathbf{{{s_.replace(' ', '~')}}}$" if m.p.iloc[0] < 0.05 else s_)
        ax.set_title(ttl, fontsize=8)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        w0, w1 = (int(x) / 100 for x in dn.split("_")[1:])
        ax.set_xlabel(f"{BAL_LABEL.replace(chr(10), ' ')}\ngain ratio, oro + SSs ({w0:g}-{w1:g} s after the first lick; "
                      "log2, vs pre)", fontsize=7.5)
        ax.set_ylabel(yl, fontsize=8.5)
        ax = axs[1, j]
        c = CV[(CV.panel == pn)].set_index("pred")
        if "marker" in c.index:
            cm, cd = c.loc["marker"], c.loc["days"]
            labs = ["unseen animal\n(LOAO)", "within-animal\nblocked", "blocked,\n+ days", "LOSO\nanimal unknown",
                    "LOSO\nanimal known"]
            vm = [cm.loao, cm.loblk, cm.loblk_plus_days, cm.loso_unknown, cm.loso_known]
            vd = [cd.loao, cd.loblk, np.nan, cd.loso_unknown, cd.loso_known]
            x = np.arange(len(labs))
            ax.bar(x - 0.2, vd, width=0.38, color="0.6", label="days since stroke")
            ax.bar(x + 0.2, vm, width=0.38, color="#C44E52", label=f"{BAL_SHORT} (oro + SSs)")
            ax.plot([0.2], [cm.null95], "k_", ms=14, mew=1.5)
            ax.text(0.2, max(cm.loao, 0) + 0.04, f"p {cm.p_shuffle:.2g}", ha="center", fontsize=7.5)
            ax.set_xticks(x, labs, fontsize=7.5)
            ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-0.6, 1)
        ax.set_title(f"{pn}: cross-validated R²", fontsize=9)
        for a_ in axs[:, j]:
            for s_ in ("top", "right"):
                a_.spines[s_].set_visible(False)
    h = [Line2D([], [], color="0.82", marker="o", mec="0.6", lw=0, label="pre (left-out reference)")]
    h += [Line2D([], [], color=EPOCH_LINE[e], marker="o", lw=0, label=e) for e in ("acute", "subacute", "chronic")]
    h += [Line2D([], [], color="0.3", marker=ANIMAL_MARKER.get(a, "o"), lw=0, label=a) for a in AN]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8.5, frameon=False)
    shown = ("\nShown as " + BAL_LABEL.replace("\n", " ") + "; slopes and r are vs the value shown"
             if BAL_DISPLAY == "direction" else "")
    fig.suptitle("PRIMARY recovery marker: hemispheric balance (contralesional vs ipsilesional, angle-matched gain ratio) "
                 "of orofacial cortex + SSs after the FIRST lick (cued: first post-cue lick; ITI: first lick after >= 2 s "
                 f"quiet).{shown}\nTop: one point per session, gray = pre (not fitted), mixed models with log(day) (bold "
                 "p < 0.05). Bottom: cross-validated R² vs days since stroke. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91 if shown else 0.93))
    save_png_svg(fig, il.OUT_W / "figures" / f"lick_recovery_primary{SFX}.png", dpi=95)
    plt.close(fig)


def trial_set_sentence() -> str:
    """THE lick sets of the figure (Priya 11-09): lick_balance_subsets' cued FIRST post-cue lick and ITI first lick after
    >= il.GAP_MAIN s quiet (KEY per-lick cache rows, the run's ITI filter), + the recovery axes."""
    import lick_refit_decoders as rd
    tst = rd._tst()
    extra = ["post-stroke licks outside the animal's pre 2.5-97.5 % angle range dropped (WFIELD_BAL_INSIDE=1)"] \
        if INSIDE else []
    if os.environ.get("WFIELD_BAL_FULL", "") == "1":
        cued = tst.cued_licks(first_only=True, extra=["EVERY first post-cue lick (lick_firstlick_extract, no cap; "
                                                      "WFIELD_BAL_FULL=1)", *extra])
    else:
        cued = rd.cued_set(first_only=True, extra=[*rd.KEY_ROWS, *extra])
    return tst.union(
        cued,
        rd.iti_set(filt=None, extra=[*rd.KEY_ROWS, f"no lick onset (licks_all.csv) in the preceding {il.GAP_MAIN:g} s",
                                     *extra]),
        rd.recovery_axes())


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if plot_only():
        plot(load_fd(f"lick_recovery_primary{SFX}"))
        return 0
    F = compute()
    save_fd(f"lick_recovery_primary{SFX}", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
